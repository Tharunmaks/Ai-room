"""The Lab: upload data, untrained models learn it, predictions come back. All offline and quick."""

import csv
import io
import json
import math
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_room.lab import samples
from ai_room.lab.automl import Trained, split_rows, train_table
from ai_room.lab.data import DataError, Encoder, profile, read_table, read_text, sniff_kind
from ai_room.lab.learners import Baseline, Boosted, Neural, OneShot, Task, feature_edges, find_bundles, grow_tree, linear, tree_leaves
from ai_room.lab.textgen import CharModel, Passages, TextModel, passages_of, train_text
from ai_room.server import create_app


def table_bytes(header, rows):
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)
    return buf.getvalue().encode()


def rings(n=400, seed=0):
    """Two classes no straight line can separate: inside vs outside a circle, plus a useless column."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        x, y = rng.uniform(-1, 1, 2)
        rows.append([f"{x:.4f}", f"{y:.4f}", f"{rng.normal():.3f}", "in" if x * x + y * y < 0.45 else "out"])
    return ["x", "y", "noise", "where"], rows


def xy_data(n=500, seed=0, kind="classification"):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    if kind == "classification":
        y = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > 0.6)).astype(np.int64)
    else:
        y = (np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.normal(size=n)).astype(np.float32)
        y = (y - y.mean()) / y.std()
    return X, y


# ------------------------------------------------------------------ reading and understanding uploads
def test_reads_csv_tsv_json_jsonl_and_headerless_files():
    assert read_table(b"a,b\n1,x\n2,y\n", "t.csv") == (["a", "b"], [["1", "x"], ["2", "y"]])
    assert read_table(b"a\tb\n1\tx\n", "t.tsv") == (["a", "b"], [["1", "x"]])
    cols, rows = read_table(b'[{"a": 1, "b": {"c": 2}}, {"a": 3}]', "t.json")
    assert cols == ["a", "b"] and rows == [["1", '{"c": 2}'], ["3", ""]]
    assert read_table(b'{"a": 1}\n{"a": 2, "z": "q"}\n', "t.jsonl") == (["a", "z"], [["1", ""], ["2", "q"]])
    cols, rows = read_table(b"1,2\n3,4\n", "t.csv")
    assert cols == ["column 1", "column 2"] and len(rows) == 2
    assert read_table(b"a;b\n1;2\n3;4\n", "t.csv")[0] == ["a", "b"]
    assert read_table(b"a,a\n1,2\n", "t.csv")[0] == ["a", "a (2)"]


def test_sniffs_tables_from_text():
    assert sniff_kind("d.csv", b"a,b\n1,2") == "table"
    assert sniff_kind("notes.txt", b"Once upon a time, there was a cat.\nIt sat.\n") == "text"
    assert sniff_kind("data.txt", b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n") == "table"


def test_text_upload_keeps_spacing_and_reads_zips():
    docs = read_text(b"def f():\n    return 1\n", "code.py")
    assert docs == [("code", "def f():\n    return 1\n")]
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "first text")
        z.writestr("b.md", "# second\n\ntext")
        z.writestr(".hidden/c.txt", "skip me")
    assert [t for t, _ in read_text(buf.getvalue(), "books.zip")] == ["a", "b"]
    with pytest.raises(DataError):
        read_text(b"   ", "empty.txt")


def test_profile_names_what_each_column_holds():
    rows = [[str(i), f"{i * 1.5}", ["red", "green", "blue"][i % 3], f"2024-01-{i % 28 + 1:02d}", "same",
             f"a long free text sentence number {i} about many different things", f"user{i}@x.org", ""] for i in range(60)]
    prof = {p["name"]: p["kind"] for p in profile(["id", "num", "colour", "day", "const", "notes", "email", "blank"], rows)}
    assert prof == {"id": "id", "num": "number", "colour": "category", "day": "date", "const": "constant", "notes": "text", "email": "id", "blank": "empty"}


def test_encoder_learns_from_training_rows_only_and_copes_with_new_values():
    cols = ["n", "c", "t"]
    train = [["1", "a", "good film"], ["3", "b", "bad film"], ["", "a", "good"], ["5", "b", "bad"]]
    enc = Encoder(cols, {"n": "number", "c": "category", "t": "text"}).fit(train)
    X = enc.transform([["2", "zzz", "unseen words"], ["", "a", ""]])
    assert X.dense.shape == (2, len(enc.names))
    assert "n is missing" in enc.names and X.dense[1, enc.names.index("n is missing")] == 1
    assert X.dense[0, enc.names.index("c = a")] == 0 and X.dense[1, enc.names.index("c = a")] == 1
    assert X.has_text and np.diff(X.sp_indptr).tolist()[1] == 0          # empty text → no words
    sub = X.take(np.array([1, 0]))
    assert sub.dense[0].tolist() == X.dense[1].tolist() and np.diff(sub.sp_indptr).tolist() == np.diff(X.sp_indptr)[::-1].tolist()


def test_split_is_stratified_and_disjoint():
    y = np.array([0] * 70 + [1] * 30)
    tr, va, te = split_rows(len(y), y, seed=1)
    assert len(set(tr) | set(va) | set(te)) == 100 and not set(tr) & set(te) and not set(va) & set(te)
    assert abs(y[te].mean() - 0.3) < 0.06 and abs(y[tr].mean() - 0.3) < 0.06


# ------------------------------------------------------------------ the learners
@pytest.mark.parametrize("make", [lambda: OneShot(), lambda: Boosted(), lambda: Neural(), linear])
def test_every_learner_starts_untrained_and_learns(make):
    from ai_room.lab.data import XData
    X, y = xy_data(600, kind="classification")
    l = make()
    if l.key == "linear":                  # a straight line cannot solve the XOR-like task, so give it one it can
        y = (X[:, 0] + 0.5 * X[:, 1] > 0.1).astype(np.int64)
    task = Task("classification", 2)
    l.fit(XData(X[:400]), y[:400], XData(X[400:500]), y[400:500], task, time.perf_counter() + 20, lambda **k: None, None)
    acc = (l.predict(XData(X[500:])).argmax(1) == y[500:]).mean()
    assert acc > 0.85, (l.key, acc)


@pytest.mark.parametrize("make", [lambda: OneShot(), lambda: Boosted(), lambda: Neural()])
def test_regression_learners_fit_a_curve(make):
    from ai_room.lab.data import XData
    X, y = xy_data(800, kind="regression")
    l = make()
    l.fit(XData(X[:500]), y[:500], XData(X[500:650]), y[500:650], Task("regression"), time.perf_counter() + 20, lambda **k: None, None)
    p = l.predict(XData(X[650:]))
    r2 = 1 - ((p - y[650:]) ** 2).mean() / y[650:].var()
    assert r2 > 0.8, (l.key, r2)


def test_tree_leaves_hold_the_newton_step_of_the_rows_routed_to_them():
    rng = np.random.default_rng(1)
    Xb = rng.integers(0, 8, (3, 400)).astype(np.uint8)
    y = (Xb[0] > 3) * 2.0 + (Xb[1] > 5) + rng.normal(0, .1, 400)
    g, h = -y[:, None], np.ones((400, 1))
    feat, thr, val, gains = grow_tree(Xb, g, h, 3, 8, 1.0, 5, h_const=1.0)
    assert feat[0] == 0 and thr[0] == 3 and gains[0] > gains[2]
    leaves = tree_leaves(feat, thr, Xb, 3)
    for leaf in np.unique(leaves):
        s = leaves == leaf
        assert val[leaf, 0] == pytest.approx(y[s].sum() / (s.sum() + 1.0))


def test_one_hot_columns_are_bundled_and_skewed_columns_keep_their_bins():
    X = np.zeros((300, 5), np.float32)
    X[np.arange(300), np.arange(300) % 3] = 1          # three one-hot columns: never on together
    X[:, 3] = np.arange(300)
    X[:, 4] = np.where(np.arange(300) % 2, 1.0, 0.0)
    X[::2, 4] = 0
    plain, bundles = find_bundles(X)
    assert 3 in plain and len(bundles) == 1 and sorted(bundles[0][0].tolist()) == [0, 1, 2]
    mostly_zero = np.concatenate([np.zeros(9000), np.random.default_rng(0).lognormal(8, 1, 1000)]).astype(np.float32)
    assert len(feature_edges(mostly_zero, 63)) > 50           # plain quantiles would give only a handful
    assert feature_edges(np.array([1, 2, 2, 3], np.float32), 63).tolist() == [1.5, 2.5]


# ------------------------------------------------------------------ training a table end to end
def test_train_table_beats_no_learning_and_reports_honestly():
    cols, rows = rings(500)
    events = []
    m = train_table(cols, rows, "where", budget=6, progress=events.append)
    r = m.report
    assert r["task"] == "classification" and r["classes"] == ["in", "out"]
    assert r["split"]["train"] + r["split"]["valid"] + r["split"]["test"] == 500
    assert r["test"]["accuracy"] > 0.85 and r["test"]["accuracy"] > r["baseline_test"]["accuracy"] + 0.15
    assert {b["learner"] for b in r["leaderboard"]} == {"baseline", "oneshot", "linear", "boost", "mlp"}
    assert r["importance"][-1]["column"] == "noise"
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stage" and "plan" in kinds and kinds[-1] == "done" and kinds.count("learner_done") == 5
    assert any(e["type"] == "progress" for e in events)
    p = m.predict([{"x": 0, "y": 0, "noise": 1}, {"x": 0.95, "y": 0.95}])
    assert [x["prediction"] for x in p] == ["in", "out"] and abs(sum(t["p"] for t in p[0]["top"]) - 1) < 1e-5


def test_regression_target_and_save_load_give_identical_predictions(tmp_path):
    fn, raw = samples.make("houses")
    cols, rows = read_table(raw, fn)
    m = train_table(cols, rows[:700], "price", budget=5)
    assert m.task.kind == "regression" and m.report["test"]["r2"] > 0.8
    m.save(tmp_path / "m")
    m2 = Trained.load(tmp_path / "m")
    probe = [dict(zip(cols, r)) for r in rows[700:720]]
    assert [p["prediction"] for p in m.predict(probe)] == pytest.approx([p["prediction"] for p in m2.predict(probe)], rel=1e-5)
    assert m.predict([{"size_m2": "250", "area": "Riverside"}])[0]["prediction"] > m.predict([{"size_m2": "40", "area": "Industrial"}])[0]["prediction"]


def test_text_column_is_learned_by_the_linear_model(tmp_path):
    fn, raw = samples.make("reviews")
    cols, rows = read_table(raw, fn)
    m = train_table(cols, rows[:800], "sentiment", budget=5)
    assert m.report["test"]["accuracy"] > 0.85 and m.report["text_columns"] == 1
    m.save(tmp_path / "r")
    m2 = Trained.load(tmp_path / "r")
    good = m2.predict([{"review": "The kettle is great. Five stars. Would buy again."}])[0]
    bad = m2.predict([{"review": "The lamp is terrible and broken. Sent it back."}])[0]
    assert good["prediction"] == "positive" and bad["prediction"] == "negative"


def test_impossible_requests_are_explained():
    cols, rows = rings(50)
    with pytest.raises(DataError, match="no column"):
        train_table(cols, rows, "nope")
    with pytest.raises(DataError, match="same"):
        train_table(["a", "b"], [[str(i), "x"] for i in range(30)], "b")
    with pytest.raises(DataError, match="At least 10"):
        train_table(cols, rows[:5], "where")
    stop = threading.Event()
    stop.set()
    with pytest.raises(DataError, match="stopped"):
        train_table(cols, rows, "where", stop=stop)


# ------------------------------------------------------------------ text models
STORY = ("The little robot walked to the market every morning. It bought apples, bread and a newspaper. "
         "On rainy days the robot carried a red umbrella, and the baker always waved. ") * 40 + \
        "\n\nThe lighthouse keeper counted ships at night. He wrote their names in a blue notebook.\n" * 20


def test_char_model_learns_text_in_one_pass_and_writes_like_it(tmp_path):
    tm = train_text([("story", STORY)])
    r = tm.report
    assert r["valid_bits_per_char"] < r["letter_frequency_bits_per_char"] < r["random_bits_per_char"]
    assert 1 <= r["order"] <= 10 and r["chars"] == len(STORY.strip())
    p = tm.model.next_distribution([1, 2, 3])
    assert abs(p.sum() - 1) < 1e-9 and p[0] == 0
    out = tm.model.generate("The little ", 120, 0.5, seed=3)
    assert len(out) == 120 and set(out) <= set(STORY)
    assert tm.model.bits("The robot bought apples and bread.") < tm.model.bits("Zxq vvb kqpz wwy jjj.")
    tm.save(tmp_path / "t")
    tm2 = TextModel.load(tmp_path / "t")
    assert tm2.model.generate("The", 80, 0.8, seed=5) == tm.model.generate("The", 80, 0.8, seed=5)
    assert tm2.search("lighthouse keeper notebook")[0]["text"].startswith("The lighthouse keeper")


def test_text_too_short_is_refused():
    with pytest.raises(ValueError, match="too short"):
        CharModel().fit("hello")


def test_passages_split_long_text_and_search_ranks_by_words():
    docs = [("a", "Cats purr softly.\n\nDogs bark loudly at night.\n\n" + "Filler sentence here. " * 80)]
    ps = passages_of(docs, size=200)
    assert len(ps) > 3 and all(len(p["text"]) < 400 for p in ps)
    hits = Passages(ps).search("why do dogs bark", 2)
    assert "Dogs bark" in hits[0]["text"]
    assert Passages(ps).search("") == []
    ranked = Passages([{"title": "q", "text": "How are you? What are they? Who are we?"}, {"title": "b", "text": "Bins split a feature into ranges."}])
    assert ranked.search("how are bins chosen")[0]["title"] == "b"        # 'how' and 'are' do not outvote 'bins'


# ------------------------------------------------------------------ the web API
@pytest.fixture
def lab(tmp_path):
    return TestClient(create_app(str(tmp_path / "db.sqlite")))


def wait_job(c, jid, timeout=60):
    for _ in range(int(timeout / 0.1)):
        j = c.get(f"/api/lab/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError("training did not finish")


def test_api_upload_train_predict_and_delete(lab):
    cols, rows = rings(300)
    up = lab.post("/api/lab/datasets?filename=rings.csv", content=table_bytes(cols, rows)).json()
    assert up["kind"] == "table" and up["rows"] == 300 and up["target"] == "where" and up["task"] == "classification"
    assert [d["id"] for d in lab.get("/api/lab/datasets").json()] == [up["id"]]
    job = lab.post("/api/lab/train", json={"dataset": up["id"], "speed": "quick", "ignore": ["noise"]}).json()
    j = wait_job(lab, job["id"])
    assert j["status"] == "done" and j["model"], j
    seen = lab.get(f"/api/lab/jobs/{job['id']}?after=0").json()
    assert seen["next"] == len(seen["events"]) > 5
    card = lab.get(f"/api/lab/models/{j['model']}").json()
    assert card["type"] == "table" and card["score"]["accuracy"] > 0.8 and "noise" not in [i["name"] for i in card["inputs"]]
    assert card["curve"] and card["report"]["leaderboard"]
    pred = lab.post(f"/api/lab/models/{j['model']}/predict", json={"rows": [{"x": 0, "y": 0}]}).json()
    assert pred["predictions"][0]["prediction"] == "in"
    r = lab.post(f"/api/lab/models/{j['model']}/predict_file?filename=new.csv", content=b"x,y\n0,0\n0.97,-0.97\n")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    out = list(csv.reader(io.StringIO(r.text)))
    assert out[0] == ["x", "y", "predicted where", "confidence"] and [row[2] for row in out[1:]] == ["in", "out"]
    assert lab.delete(f"/api/lab/models/{j['model']}").json()["ok"] and lab.get("/api/lab/models").json() == []
    assert lab.delete(f"/api/lab/datasets/{up['id']}").json()["ok"] and lab.get("/api/lab/datasets").json() == []


def test_api_text_model_writes_and_answers(lab):
    up = lab.post("/api/lab/datasets?filename=story.txt", content=STORY.encode()).json()
    assert up["kind"] == "text" and up["chars"] == len(STORY)
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": up["id"]}).json()["id"])
    assert j["status"] == "done", j
    card = lab.get(f"/api/lab/models/{j['model']}").json()
    assert card["type"] == "text" and card["score"]["bits_per_char"] < card["score"]["random_bits_per_char"]
    w = lab.post(f"/api/lab/models/{j['model']}/predict", json={"prompt": "The ", "length": 50, "seed": 1}).json()
    assert len(w["text"]) == 50
    a = lab.post(f"/api/lab/models/{j['model']}/predict", json={"query": "umbrella rainy"}).json()
    assert "umbrella" in a["passages"][0]["text"] and "text" not in a
    assert lab.post(f"/api/lab/models/{j['model']}/predict_file", content=b"a\n1\n").status_code == 400


def test_api_samples_errors_and_safety(lab):
    info = lab.get("/api/lab/info").json()
    assert set(info["speeds"]) == {"quick", "balanced", "thorough"} and {s["key"] for s in info["samples"]} == set(samples.SAMPLES)
    sp = lab.post("/api/lab/samples/spirals").json()
    assert sp["rows"] == 1200 and sp["target"] == "arm" and sp["sample"] == "spirals"
    assert lab.post("/api/lab/samples/nope").status_code == 404
    assert lab.post("/api/lab/datasets?filename=e.csv", content=b"").status_code == 400
    assert lab.post("/api/lab/datasets?filename=pics.zip", content=b"not a zip").status_code == 400
    assert lab.post("/api/lab/train", json={"dataset": sp["id"], "target": "nope"}).status_code == 400
    assert lab.post("/api/lab/train", json={"dataset": sp["id"], "speed": "warp"}).status_code == 400
    assert lab.get("/api/lab/datasets/..%2F..%2Fetc").status_code in (400, 404)
    assert lab.get("/api/lab/models/bad id!").status_code == 400
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": sp["id"], "target": "x", "task": "classification"}).json()["id"])
    assert j["status"] == "error" and "too many" in j["error"]


def test_cli_trains_predicts_and_lists(tmp_path, capsys):
    from ai_room.cli import main
    cols, rows = rings(300)
    data = tmp_path / "rings.csv"
    data.write_bytes(table_bytes(cols, rows))
    lab_dir = tmp_path / "lab"
    main(["lab", "train", str(data), "--speed", "quick", "--data", str(lab_dir)])
    out = capsys.readouterr().out
    assert "Winner:" in out and "never saw" in out
    mid = next((lab_dir / "models").iterdir()).name
    new = tmp_path / "new.csv"
    new.write_text("x,y\n0,0\n0.97,0.97\n")
    main(["lab", "predict", mid, str(new), "--data", str(lab_dir), "--out", str(tmp_path / "p.csv")])
    assert [r[-2] for r in csv.reader(open(tmp_path / "p.csv"))][1:] == ["in", "out"]
    main(["lab", "models", "--data", str(lab_dir)])
    assert mid in capsys.readouterr().out
    card = json.loads((lab_dir / "models" / mid / "card.json").read_text())
    assert card["type"] == "table" and card["inputs"]
