"""`ai-room lab …`: the Lab from the terminal. Models land in the same folder the web app uses, so they show up there too."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .api import write_card
from .automl import SECONDS_PER_MB, SPEEDS, Trained, budget_for, new_id, train_table
from .data import DataError
from .ingest import human, load_table, load_text, sniff_path
from .textgen import MAX_CHARS, TextModel, train_text


def lab_dir(data: str | None = None) -> Path:
    db = os.environ.get("AI_ROOM_DB", "data/ai_room.db")
    return Path(data) if data else Path(os.path.dirname(db) or ".") / "lab"


def find_model(spec: str, data: str | None = None) -> Path:
    p = Path(spec)
    if (p / "model.json").exists():
        return p
    q = lab_dir(data) / "models" / spec
    if (q / "model.json").exists():
        return q
    raise SystemExit(f"No model at '{spec}'. List them with: ai-room lab models")


def load(folder: Path):
    import json
    kind = json.loads((folder / "model.json").read_text()).get("type")
    return TextModel.load(folder) if kind == "text" else Trained.load(folder)


def _printer():
    t0 = time.perf_counter()

    def show(e: dict):
        k = e.get("type")
        if k == "stage":
            print(f"  · {e['text']}", flush=True)
        elif k == "learner_start":
            print(f"  ▸ {e['label']} …", end="", flush=True)
        elif k == "learner_done":
            v = e["valid"]
            score = f"accuracy {v['accuracy']:.1%}" if "accuracy" in v else f"R² {v['r2']:.3f}"
            print(f" {score} on the progress-check rows, {e['seconds']:.2f} s", flush=True)
        elif k == "learner_failed":
            print(f" failed: {e['error']}", flush=True)
    return show, t0


def cmd_train(a) -> None:
    path = Path(a.file)
    if not path.exists():
        raise SystemExit(f"No such file: {path}")
    speed = "fast" if a.speed == "quick" else a.speed
    size = path.stat().st_size
    budget = budget_for(speed, size)
    show, t0 = _printer()
    deadline = t0 + budget
    say = lambda text: show({"type": "stage", "text": text})
    try:
        kind = sniff_path(path, path.name)
        print(f"Learning {path.name} ({human(size)}) from scratch as {'text' if kind == 'text' else 'a table'}; "
              f"time allowed: {budget:.1f} s ({SECONDS_PER_MB:g} s per MB{', at least 1 s' if budget <= 1 else ''}) …", flush=True)
        if kind == "text":
            docs, _ = load_text(path, path.name, MAX_CHARS, say=say)
            model = train_text(docs, progress=show, deadline=deadline - 0.12 * budget)
        elif kind == "table":
            table = load_table(path, path.name, say=say)
            model = train_table(table, None, a.target, task=a.task, ignore=[c.strip() for c in (a.ignore or "").split(",") if c.strip()],
                                progress=show, deadline=deadline - 0.03 * budget)
        else:
            raise SystemExit("Folders of images are not supported yet.")
    except (DataError, ValueError) as e:
        raise SystemExit(str(e))
    folder = Path(a.out) if a.out else lab_dir(a.data) / "models" / new_id()
    model.save(folder)
    secs = time.perf_counter() - t0
    write_card(folder, model, name=a.name or path.stem, dataset_name=path.name, seconds=secs, speed=speed, budget=budget, size=size)
    r = model.report
    print()
    if kind == "text":
        print(f"Learned {r['learned_chars']:,} of {r['chars']:,} characters: {r['valid_bits_per_char']:.2f} bits per character on held-back text "
              f"(random guessing: {r['random_bits_per_char']:.2f}; context {r['order']} characters).")
        print(f"Done in {secs:.2f} s of the {budget:.1f} s allowed.  Try it:  ai-room lab write {folder.name} \"Once upon\"")
    else:
        key, fmt = ("accuracy", "{:.1%}") if r["task"] == "classification" else ("r2", "{:.3f}")
        base = fmt.format(r["baseline_test"][key]) if r["baseline_test"] else "—"
        print(f"Winner: {r['winner_label']}. On {r['split']['test']:,} rows it never saw: {key} {fmt.format(r['test'][key])} (no learning: {base}).")
        print(f"Done in {secs:.2f} s of the {budget:.1f} s allowed; the first model was ready after {r['first_model_seconds']:.2f} s.")
        if r.get("sampled"):
            print(f"The file has about {r['file_rows']:,} rows; it learned from a random sample of {r['rows']:,}.")
        if r["importance"]:
            print("Relies most on: " + ", ".join(f"{i['column']} ({i['share']:.0%})" for i in r["importance"][:5]))
        print(f"Use it:  ai-room lab predict {folder.name} new_rows.csv")
    print(f"Saved to {folder}")


def cmd_predict(a) -> None:
    from .api import predict_table_file
    m = load(find_model(a.model, a.data))
    if isinstance(m, TextModel):
        raise SystemExit("That is a text model: use `ai-room lab write` or `ai-room lab ask`.")
    path = Path(a.file)
    out = Path(a.out) if a.out else None
    import tempfile
    target = out or Path(tempfile.mkstemp(suffix=".csv")[1])
    n = predict_table_file(m, path, path.name, target)
    if out:
        print(f"Wrote {n:,} predictions to {out}")
    else:
        sys.stdout.write(target.read_text(encoding="utf-8"))
        target.unlink()


def cmd_write(a) -> None:
    m = load(find_model(a.model, a.data))
    if not isinstance(m, TextModel):
        raise SystemExit("That model predicts a column; use `ai-room lab predict`.")
    prompt = a.prompt or ""
    print(prompt + m.model.generate(prompt, a.length, a.temperature, a.seed))


def cmd_ask(a) -> None:
    m = load(find_model(a.model, a.data))
    if not isinstance(m, TextModel):
        raise SystemExit("Only text models can be searched.")
    hits = m.search(" ".join(a.question), a.k)
    if not hits:
        print("Nothing in the text matches those words.")
    for h in hits:
        print(f"\n[{h['score']:.1f}] {h['title']}\n{h['text'][:800]}")


def cmd_models(a) -> None:
    import json
    root = lab_dir(a.data) / "models"
    cards = []
    for p in root.glob("*/card.json") if root.exists() else []:
        try:
            cards.append(json.loads(p.read_text()))
        except ValueError:
            pass
    if not cards:
        print("No models yet. Train one:  ai-room lab train data.csv")
    for c in sorted(cards, key=lambda c: -c["created"]):
        if c["type"] == "text":
            what = f"text · {c['score']['bits_per_char']:.2f} bits/char"
        else:
            s = c["score"]
            what = f"predicts {c['target']} · " + (f"accuracy {s['accuracy']:.1%}" if "accuracy" in s else f"R² {s['r2']:.3f}")
        print(f"{c['id']}  {c['name'][:28]:28}  {what}  ({c['seconds']:.1f} s)")


def add_parser(sub) -> None:
    lab = sub.add_parser("lab", help="train untrained models on your own data, fast, on the CPU")
    ls = lab.add_subparsers(dest="lab_cmd", required=True)
    t = ls.add_parser("train", help="learn a table (CSV/TSV/JSON) or a text (txt/md/pdf/docx/epub/zip)")
    t.add_argument("file")
    t.add_argument("--target", help="column to predict (default: guessed, usually the last)")
    t.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    t.add_argument("--ignore", help="comma-separated columns to leave out")
    t.add_argument("--speed", choices=list(SPEEDS) + ["quick"], default="fast",
                   help="time allowed: fast = 1 s per MB of data (at least 1 s); balanced ≥ 30 s; thorough ≥ 2 min")
    t.add_argument("--name")
    t.add_argument("--out", help="folder to save the model in (default: data/lab/models/<id>)")
    p = ls.add_parser("predict", help="add predictions to every row of a table")
    p.add_argument("model", help="model id (see `ai-room lab models`) or folder")
    p.add_argument("file")
    p.add_argument("--out", help="CSV to write (default: print)")
    w = ls.add_parser("write", help="let a text model write")
    w.add_argument("model")
    w.add_argument("prompt", nargs="?")
    w.add_argument("--length", type=int, default=400)
    w.add_argument("--temperature", type=float, default=0.8)
    w.add_argument("--seed", type=int)
    q = ls.add_parser("ask", help="find the passages of a learned text that answer a question")
    q.add_argument("model")
    q.add_argument("question", nargs="+")
    q.add_argument("-k", type=int, default=3)
    m = ls.add_parser("models", help="list trained models")
    for x in (t, p, w, q, m):
        x.add_argument("--data", help="Lab folder (default: data/lab next to AI_ROOM_DB)")


def run(a) -> None:
    {"train": cmd_train, "predict": cmd_predict, "write": cmd_write, "ask": cmd_ask, "models": cmd_models}[a.lab_cmd](a)
