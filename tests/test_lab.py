"""The Lab: upload data, untrained models learn it, predictions come back. All offline and quick."""

import csv
import io
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_room.lab import samples
from ai_room.lab.automl import Trained, split_rows, train_table
from ai_room.lab.data import DataError, Encoder, profile, read_table, read_text, sniff_kind
from ai_room.lab.learners import Boosted, Neural, OneShot, Task, feature_edges, find_bundles, grow_tree, linear, tree_leaves
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
    assert [s["key"] for s in info["speeds"]] == ["fast", "balanced", "thorough"] and info["speeds"][0]["label"] == "5 min per GB"
    assert info["max_bytes"] == 5 * 10**12 and info["max_label"] == "5.0 TB" and info["seconds_per_mb"] == 0.3 and info["min_seconds"] == 0.5
    assert {s["key"] for s in info["samples"]} == set(samples.SAMPLES)
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


# ------------------------------------------------------------------ size limits and the 1 second per MB budget
def test_budget_is_five_minutes_per_gigabyte():
    from ai_room.lab.automl import RATE_LABEL, budget_for
    assert RATE_LABEL == "5 min per GB"
    assert budget_for("fast", 1e9) == pytest.approx(300)                                  # 1 GB → 5 minutes
    assert budget_for("fast", 0) == 0.5 and budget_for("fast", 1_000_000) == 0.5           # at least half a second
    assert budget_for("fast", 7_500_000) == pytest.approx(2.25) and budget_for("quick", 50e6) == pytest.approx(15)
    assert budget_for("fast", 5e12) == pytest.approx(1_500_000)                           # 5 TB → 25,000 minutes at most
    assert budget_for("balanced", 1e6) == 30 and budget_for("balanced", 200e6) == pytest.approx(60)


def test_upload_streams_to_disk_and_enforces_the_size_cap_and_free_space(lab, monkeypatch):
    import ai_room.lab.api as api
    cols, rows = rings(200)
    body = table_bytes(cols, rows)
    monkeypatch.setattr(api, "MAX_BYTES", len(body) - 1)
    r = lab.post("/api/lab/datasets?filename=big.csv", content=body)
    assert r.status_code == 413 and "limit" in r.json()["detail"]
    monkeypatch.setattr(api, "MAX_BYTES", 5 * 10**12)

    class Tiny:
        free = 1000
    monkeypatch.setattr(api.shutil, "disk_usage", lambda p: Tiny)
    r = lab.post("/api/lab/datasets?filename=big.csv", content=body)
    assert r.status_code == 507 and "disk space" in r.json()["detail"]
    monkeypatch.undo()
    assert lab.get("/api/lab/datasets").json() == []                 # nothing half-written is left behind
    up = lab.post("/api/lab/datasets?filename=ok.csv", content=body).json()
    assert up["rows"] == 200 and up["rows_exact"] and up["budget_seconds"] == 0.5


def test_a_training_job_finishes_within_five_minutes_per_gigabyte(lab):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40_000, 6))
    y = np.where(X[:, 0] * X[:, 1] + X[:, 2] > 0, "yes", "no")
    body = table_bytes([f"f{i}" for i in range(6)] + ["label"], [[f"{v:.5f}" for v in x] + [t] for x, t in zip(X, y)])
    up = lab.post("/api/lab/datasets?filename=mid.csv", content=body).json()
    budget = up["budget_seconds"]
    assert budget == pytest.approx(len(body) / 1e6 * 0.3, abs=0.01) and budget > 0.6
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": up["id"]}).json()["id"])
    assert j["status"] == "done" and j["budget"] == pytest.approx(budget, abs=0.01)
    assert j["seconds"] <= budget * 1.15 + 0.3, (j["seconds"], budget)
    card = lab.get(f"/api/lab/models/{j['model']}").json()
    assert card["score"]["accuracy"] > 0.85 and card["budget"] == pytest.approx(budget, abs=0.01)


def test_tables_with_more_cells_than_memory_are_sampled(tmp_path):
    from ai_room.lab.ingest import load_table
    n = 500_000
    a = np.random.default_rng(0).normal(size=(n, 2))
    lines = ["a,b,c"] + [f"{x:.5f},{z:.5f},{'p' if x + z > 0 else 'q'}" for x, z in a]
    path = tmp_path / "big.csv"
    path.write_text("\n".join(lines) + "\n")
    assert path.stat().st_size > 8 << 20                                   # big enough to be read by random seeks
    t = load_table(path, "big.csv", limit_cells=30_000)
    assert t.sampled and not t.exact and 9_000 <= t.n <= 10_000 and abs(t.total_rows - n) / n < 0.1
    assert set(t.col("c").tolist()) == {"p", "q"} and np.isfinite(t.col("a").astype(float)).all()
    assert len(set(zip(t.col("a").tolist(), t.col("b").tolist()))) == t.n            # every row at most once
    half = load_table(path, "big.csv", limit_cells=600_000)                          # 40% of the rows: one sequential pass
    assert half.sampled and 180_000 <= half.n <= 210_000 and len(set(zip(half.col("a").tolist(), half.col("b").tolist()))) == half.n
    full = load_table(path, "big.csv", limit_cells=10**9)
    assert full.n == n and not full.sampled
    small = tmp_path / "small.csv"
    small.write_text("x,y\n" + "\n".join(f"{i},{i % 3}" for i in range(2000)) + "\n")
    s = load_table(small, "small.csv", limit_cells=1000)                  # small files: loaded, then a random sample kept
    assert s.sampled and s.n == 1000 and s.total_rows == 2000                 # never fewer than 1,000 rows


def test_all_numeric_files_take_the_fast_parser(tmp_path):
    from ai_room.lab.ingest import load_table
    rng = np.random.default_rng(1)
    rows = [",".join(f"{v:.6f}" for v in r) for r in rng.normal(size=(150_000, 8))]
    path = tmp_path / "nums.csv"
    path.write_text("a,b,c,d,e,f,g,h\n" + "\n".join(rows) + "\n")
    t = load_table(path, "nums.csv")
    assert t.n == 150_000 and all(c.dtype == np.float64 for c in t.data)
    prof = profile(t)
    assert all(p["kind"] == "number" for p in prof)


def test_big_texts_are_sampled_evenly_and_text_learning_meets_its_deadline(tmp_path):
    from ai_room.lab.ingest import load_text
    path = tmp_path / "big.txt"
    path.write_text(("".join(f"Chapter {i}. " + STORY[:2000] + "\n" for i in range(3000))))
    docs, info = load_text(path, "big.txt", max_chars=1_000_000)
    assert info["sampled"] and info["pieces"] == 1 and len(docs) == 1
    docs, info = load_text(path, "big.txt", max_chars=3_000_000)
    assert not info["sampled"] or info["pieces"] >= 2
    text = STORY * 30
    t0 = time.perf_counter()
    m = CharModel().fit(text, deadline=t0 + 0.4)
    assert time.perf_counter() - t0 < 1.0 and 1 <= m.order <= 10 and m.report["learned_chars"] <= len(text)
    assert abs(m.next_distribution([1, 2]).sum() - 1) < 1e-9


# ------------------------------------------------------------------ Excel
def make_xlsx(rows, date_col=None) -> bytes:
    """A minimal real .xlsx (shared strings, a date style, booleans, an inline string), written by hand."""
    import zipfile
    strings, cells_xml = [], []
    def col(i):
        return chr(65 + i)
    for r, row in enumerate(rows, 1):
        cs = []
        for i, v in enumerate(row):
            ref = f"{col(i)}{r}"
            if v is None:
                continue
            if isinstance(v, bool):
                cs.append(f'<c r="{ref}" t="b"><v>{int(v)}</v></c>')
            elif isinstance(v, (int, float)):
                style = ' s="1"' if date_col is not None and i == date_col and r > 1 else ""
                cs.append(f'<c r="{ref}"{style}><v>{v}</v></c>')
            elif v.startswith("inline:"):
                cs.append(f'<c r="{ref}" t="inlineStr"><is><t>{v[7:]}</t></is></c>')
            else:
                if v not in strings:
                    strings.append(v)
                cs.append(f'<c r="{ref}" t="s"><v>{strings.index(v)}</v></c>')
        cells_xml.append(f'<row r="{r}">{"".join(cs)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rel}><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/></Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst {ns}>' + "".join(f"<si><t>{t}</t></si>" for t in strings) + "</sst>")
        z.writestr("xl/styles.xml", f'<styleSheet {ns}><cellXfs><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet {ns}><dimension ref="A1:D{len(rows)}"/><sheetData>{"".join(cells_xml)}</sheetData></worksheet>')
    return buf.getvalue()


def test_excel_workbooks_are_read_with_dates_booleans_and_gaps():
    from ai_room.lab.xlsx import Workbook
    raw = make_xlsx([["name", "joined", "active", "score"], ["Ana", 45000, True, 3.5], ["Bo", 45366, False, None], ["inline:Cy", 45001.5, True, 7]], date_col=1)
    cols, rows = read_table(raw, "people.xlsx")
    assert cols == ["name", "joined", "active", "score"]
    assert rows == [["Ana", "2023-03-15", "TRUE", "3.5"], ["Bo", "2024-03-15", "FALSE", ""], ["Cy", "2023-03-16 12:00:00", "TRUE", "7"]]
    assert Workbook(io.BytesIO(raw)).dimension_rows() == 4
    with pytest.raises(DataError, match="xlsx"):
        read_table(b"not a zip", "old.xls")


def test_an_excel_upload_trains_like_a_csv(lab):
    cols, rows = rings(240)
    data = [cols] + [[float(r[0]), float(r[1]), float(r[2]), r[3]] for r in rows]
    up = lab.post("/api/lab/datasets?filename=rings.xlsx", content=make_xlsx(data)).json()
    assert up["kind"] == "table" and up["rows"] == 240 and up["target"] == "where"
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": up["id"], "speed": "balanced"}).json()["id"])
    assert j["status"] == "done", j
    r = lab.post(f"/api/lab/models/{j['model']}/predict_file?filename=new.xlsx", content=make_xlsx([["x", "y"], [0, 0], [0.97, 0.97]]))
    assert r.status_code == 200 and [row[2] for row in csv.reader(io.StringIO(r.text))][1:] == ["in", "out"]


# ------------------------------------------------------------------ pictures
def shapes_zip(tmp_path, per_class=40):
    raw = samples.shapes(per_class=per_class, size=40)
    path = tmp_path / "shapes.zip"
    path.write_bytes(raw)
    return path


def test_pictures_become_features_and_a_picture_model_learns_them(tmp_path):
    from ai_room.lab.images import FEATURES, describe_zip, image_table, load_image_zip
    path = shapes_zip(tmp_path)
    assert describe_zip(path) == {"classes": {"circles": 40, "squares": 40, "triangles": 40}, "images": 120}
    t = load_image_zip(path)
    assert t.n == 120 and len(t.columns) == FEATURES + 1 and t.columns[-1] == "category"
    m = train_table(t, None, "category", budget=8, image=True)
    r = m.report
    assert r["image"]["groups"] == ["shape", "colour layout", "edges", "colour mix"] and r["test"]["accuracy"] > r["baseline_test"]["accuracy"]
    assert {i["column"] for i in r["importance"]} <= {"shape", "colour layout", "edges", "colour mix"}
    import zipfile
    z = zipfile.ZipFile(path)
    one = z.read(next(n for n in z.namelist() if "circles/" in n))
    p = m.predict(image_table([one]))[0]
    assert p["prediction"] in ("circles", "squares", "triangles") and abs(sum(x["p"] for x in p["top"]) - 1) < 1e-5


def test_picture_datasets_are_explained_and_bad_uploads_refused(tmp_path):
    from ai_room.lab.images import describe_zip
    import zipfile
    one = tmp_path / "one.zip"
    with zipfile.ZipFile(one, "w") as z:
        z.writestr("cats/a.png", b"x")
        z.writestr("cats/b.png", b"y")
    with pytest.raises(DataError, match="one folder per category"):
        describe_zip(one)


def test_api_learns_pictures_and_answers_about_a_new_one(lab, tmp_path):
    sp = lab.post("/api/lab/samples/shapes").json()
    assert sp["kind"] == "images" and sp["images"] == 600 and sp["classes"] == {"circles": 200, "squares": 200, "triangles": 200}
    png = lab.get(f"/api/lab/datasets/{sp['id']}/preview.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": sp["id"], "speed": "balanced"}).json()["id"])
    assert j["status"] == "done", j
    card = lab.get(f"/api/lab/models/{j['model']}").json()
    assert card["type"] == "image" and card["inputs"] == [] and card["score"]["accuracy"] > 0.5
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(samples.shapes(per_class=3, size=40, seed=99)))
    pic = z.read(z.namelist()[0])
    r = lab.post(f"/api/lab/models/{j['model']}/predict_image", content=pic).json()
    assert r["prediction"]["prediction"] in ("circles", "squares", "triangles")
    assert lab.post(f"/api/lab/models/{j['model']}/predict_image", content=b"not a picture").status_code == 400
    out = lab.post(f"/api/lab/models/{j['model']}/predict_file?filename=batch.zip", content=samples.shapes(per_class=2, size=40, seed=7))
    rows = list(csv.reader(io.StringIO(out.text)))
    assert out.status_code == 200 and rows[0] == ["file", "predicted category", "confidence"] and len(rows) == 7
    assert lab.post("/api/lab/datasets?filename=cat.jpg", content=pic).status_code == 400   # one picture: explain zips


# ------------------------------------------------------------------ friendlier model handling
def test_models_can_be_renamed_downloaded_and_imported(lab):
    cols, rows = rings(200)
    up = lab.post("/api/lab/datasets?filename=rings.tsv", content=table_bytes(cols, rows).replace(b",", b"\t")).json()
    assert up["kind"] == "table" and up["columns"] == cols                      # pasted from a spreadsheet: tab-separated
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": up["id"]}).json()["id"])
    mid = j["model"]
    assert lab.patch(f"/api/lab/models/{mid}", json={"name": "  Inside the ring  "}).json()["name"] == "Inside the ring"
    assert lab.patch(f"/api/lab/models/{mid}", json={"name": "   "}).status_code == 400
    z = lab.get(f"/api/lab/models/{mid}/download")
    assert z.status_code == 200 and "Inside_the_ring" in z.headers["content-disposition"]
    imp = lab.post("/api/lab/models/import", content=z.content).json()
    assert imp["id"] != mid and imp["name"] == "Inside the ring" and imp["curve"]
    a = lab.post(f"/api/lab/models/{mid}/predict", json={"rows": [{"x": 0.1, "y": 0}]}).json()
    b = lab.post(f"/api/lab/models/{imp['id']}/predict", json={"rows": [{"x": 0.1, "y": 0}]}).json()
    assert a["predictions"] == b["predictions"]
    assert lab.post("/api/lab/models/import", content=b"junk").status_code == 400
    assert len(lab.get("/api/lab/models").json()) == 2


def test_text_models_saved_before_column_hashing_still_predict(tmp_path):
    fn, raw = samples.make("reviews")
    cols, rows = read_table(raw, fn)
    m = train_table(cols, rows[:400], "sentiment", budget=3)
    m.save(tmp_path / "v2")
    meta = json.loads((tmp_path / "v2" / "model.json").read_text())
    for s in meta["encoder"]["spec"]:
        s.pop("v", None)                                     # what a model from the previous version looks like
    (tmp_path / "v2" / "model.json").write_text(json.dumps(meta))
    old = Trained.load(tmp_path / "v2")
    p = old.predict([{"review": "The kettle is great. Five stars."}])[0]
    assert p["prediction"] in ("positive", "negative") and 0 < p["confidence"] <= 1


# ------------------------------------------------------------------ sounds and videos
def wav_bytes(x, rate=16000, width=2, channels=1):
    import wave
    x = np.clip(np.asarray(x, np.float64), -1, 1)
    if channels > 1:
        x = np.repeat(x[:, None], channels, 1).ravel()
    if width == 1:
        pcm = ((x * 127) + 128).astype(np.uint8).tobytes()
    elif width == 2:
        pcm = (x * 32767).astype("<i2").tobytes()
    elif width == 3:
        v = (x * 8388607).astype("<i4")
        pcm = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes()
    else:
        pcm = (x * 2147483647).astype("<i4").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def test_wav_files_of_every_common_kind_are_read_at_16_khz():
    from ai_room.lab import media
    t = np.arange(22050) / 22050
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)                           # one second of A
    for width in (1, 2, 3, 4):
        for channels in (1, 2):
            x = media.decode_audio(wav_bytes(tone, 22050, width, channels), "a.wav")
            assert abs(len(x) - 16000) <= 1 and 0.4 < np.abs(x).max() < 0.6, (width, channels)
            spec = np.abs(np.fft.rfft(x))
            assert abs(np.argmax(spec) * 16000 / len(x) - 440) < 3          # still the same note after resampling
    f = media.audio_features(media.decode_audio(wav_bytes(tone), "a.wav"))
    assert f.shape == (media.AUDIO_FEATURES,) == (212,) and np.isfinite(f).all()
    assert media.audio_features(np.zeros(10, np.float32)).shape == (212,)       # a tiny, silent clip still works
    with pytest.raises(DataError):
        media.decode_audio(b"RIFF not really", "broken.wav")


def test_zips_of_sounds_videos_and_pictures_are_told_apart():
    from ai_room.lab import media
    z = lambda raw: __import__("zipfile").ZipFile(io.BytesIO(raw))
    assert media.sniff_zip(z(samples.sounds(per_class=3))) == "sounds"
    assert media.sniff_zip(z(samples.motions(per_class=3, frames=4))) == "videos"
    assert media.sniff_zip(z(samples.shapes(per_class=3))) == "images"
    assert media.zip_kind(["a/1.mp4", "b/2.mov", "b/3.webm"]) == "video"
    assert media.zip_kind(["a/1.wav", "b/2.mp3", "notes.txt"]) == "audio"
    assert media.zip_kind(["a/1.csv", "b/2.csv"]) is None
    assert sniff_kind("s.zip", samples.sounds(per_class=3)) == "sounds"


def test_sounds_become_numbers_and_a_sound_model_learns_them(tmp_path):
    from ai_room.lab import media
    path = tmp_path / "sounds.zip"
    path.write_bytes(samples.sounds(per_class=30))
    info = media.describe(path, "audio")
    assert info == {"classes": {"chirps": 30, "chords": 30, "drums": 30, "whistles": 30}, "items": 120, "media": "audio"}
    t = media.load_zip(path, "audio")
    assert t.n == 120 and len(t.columns) == media.AUDIO_FEATURES + 1 and t.columns[-1] == "category"
    m = train_table(t, None, "category", budget=6, image="audio")
    r = m.report
    assert r["media"] == "audio" and r["audio"]["groups"] == list(media.AUDIO_GROUPS)
    assert r["test"]["accuracy"] >= 0.8 > r["baseline_test"]["accuracy"]
    assert {i["column"] for i in r["importance"]} <= set(media.AUDIO_GROUPS)
    whistle = media.decode_audio(__import__("zipfile").ZipFile(path).read("sounds/whistles/whistle_000.wav"), "w.wav")
    p = m.predict(media.media_table("audio", [("w.wav", wav_bytes(whistle))]))[0]
    assert p["prediction"] in info["classes"] and abs(sum(x["p"] for x in p["top"]) - 1) < 1e-5
    png = media.preview_sheet(path, "audio")
    assert png[:4] == b"\x89PNG"


def test_animated_gifs_become_look_and_motion_and_a_video_model_learns_them(tmp_path):
    from ai_room.lab import media
    path = tmp_path / "motions.zip"
    path.write_bytes(samples.motions(per_class=25, frames=8))
    assert media.describe(path, "video")["classes"] == {"bouncing": 25, "growing": 25, "sliding": 25, "spinning": 25}
    raw = __import__("zipfile").ZipFile(path).read("motions/sliding/sliding_000.gif")
    frames, sound = media.decode_video(raw, "a.gif")
    assert len(frames) == 8 and sound is None
    f = media.video_features(frames, sound)
    assert f.shape == (media.VIDEO_FEATURES,) == (1172,) and not f[-media.AUDIO_FEATURES - 1:].any()   # no sound: zeros
    still = media.video_features(frames[:1] * 8, None)
    assert np.abs(still[936:936 + media.MOTION]).max() == 0                     # nothing moves: no motion
    t = media.load_zip(path, "video")
    m = train_table(t, None, "category", budget=6, image="video")
    r = m.report
    assert r["media"] == "video" and r["test"]["accuracy"] >= 0.7 > r["baseline_test"]["accuracy"]
    assert media.preview_sheet(path, "video")[:4] == b"\x89PNG"


needs_ffmpeg = pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg is not installed")


@needs_ffmpeg
def test_mp3_sounds_and_mp4_videos_with_a_sound_track_are_read_through_ffmpeg(tmp_path):
    import subprocess
    from ai_room.lab import media
    t = np.arange(16000) / 16000
    (tmp_path / "a.wav").write_bytes(wav_bytes(0.5 * np.sin(2 * np.pi * 660 * t)))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_path / "a.wav"), str(tmp_path / "a.mp3")], check=True)
    x = media.decode_audio((tmp_path / "a.mp3").read_bytes(), "a.mp3")
    assert 15000 < len(x) < 17500 and abs(np.argmax(np.abs(np.fft.rfft(x))) * 16000 / len(x) - 660) < 5
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", "-shortest", "-pix_fmt", "yuv420p", str(tmp_path / "v.mp4")], check=True)
    frames, sound = media.decode_video((tmp_path / "v.mp4").read_bytes(), "v.mp4")
    assert len(frames) == media.FRAMES and frames[0].size == (64, 64) and sound is not None and len(sound) > 16000
    f = media.video_features(frames, sound)
    assert f[-1] == 1.0 and np.abs(f[936:936 + media.MOTION]).max() > 0     # it has sound, and things move


def test_api_learns_sounds_and_videos_and_answers_about_new_ones(lab):
    sp = lab.post("/api/lab/samples/sounds").json()
    assert sp["kind"] == "sounds" and sp["items"] == 400 and sp["media"] == "audio" and len(sp["classes"]) == 4
    assert lab.get(f"/api/lab/datasets/{sp['id']}/preview.png").content[:4] == b"\x89PNG"
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": sp["id"]}).json()["id"])
    assert j["status"] == "done", j
    card = lab.get(f"/api/lab/models/{j['model']}").json()
    assert card["type"] == "audio" and card["score"]["accuracy"] > 0.8
    t = np.arange(8000) / 8000
    r = lab.post(f"/api/lab/models/{j['model']}/predict_media?filename=mine.wav", content=wav_bytes(np.sin(2 * np.pi * 600 * t), 8000)).json()
    assert r["media"] == "audio" and r["prediction"]["prediction"] in sp["classes"]
    assert lab.post(f"/api/lab/models/{j['model']}/predict_media?filename=x.wav", content=b"nope").status_code == 400
    out = lab.post(f"/api/lab/models/{j['model']}/predict_file?filename=more.zip", content=samples.sounds(per_class=2, seed=5))
    rows = list(csv.reader(io.StringIO(out.text)))
    assert out.status_code == 200 and rows[0] == ["file", "predicted category", "confidence"] and len(rows) == 9
    assert lab.post("/api/lab/datasets?filename=bark.wav", content=wav_bytes(t)).status_code == 400   # one sound: explain zips

    vp = lab.post("/api/lab/samples/motions").json()
    assert vp["kind"] == "videos" and vp["items"] == 320 and vp["media"] == "video"
    j = wait_job(lab, lab.post("/api/lab/train", json={"dataset": vp["id"]}).json()["id"])
    assert j["status"] == "done", j
    card = lab.get(f"/api/lab/models/{j['model']}").json()
    assert card["type"] == "video" and card["score"]["accuracy"] > 0.6
    gif = __import__("zipfile").ZipFile(io.BytesIO(samples.motions(per_class=1, frames=6, seed=3))).read("motions/bouncing/bouncing_000.gif")
    r = lab.post(f"/api/lab/models/{j['model']}/predict_media?filename=clip.gif", content=gif).json()
    assert r["media"] == "video" and r["prediction"]["prediction"] in vp["classes"]
