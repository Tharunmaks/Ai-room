"""Command line entry point.

    ai-room serve                                   # web UI on http://127.0.0.1:8765
    ai-room run --student ollama:qwen2.5:14b --persona socratic --subjects math,physics,code
    ai-room library fetch astronomy history         # download textbooks (Wikipedia, Wikibooks)
    ai-room library import history ./books          # your own txt/md/html/pdf/epub/docx files
    ai-room bank import questions.jsonl --subject biology
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

from .classroom import ClassConfig, Classroom
from .curriculum import DEFAULT_SUBJECTS, SUBJECTS
from .memory import Store
from .teacher import PRESETS, Library
from .teacher.fetch import fetch_subject
from .teacher.library import parse_bank_text


def _student(spec: str) -> dict:
    backend, _, model = spec.partition(":")
    if backend == "demo":
        return {"backend": "demo", "model": "simulated-student"}
    if not model:
        raise SystemExit(f"Student spec must look like backend:model (e.g. ollama:qwen2.5:14b) or 'demo', got '{spec}'")
    return {"backend": backend, "model": model}


def _print_new(store: Store, cid: str, stop: threading.Event) -> None:
    last = 0
    while not stop.is_set():
        for m in store.messages(cid, last):
            last = m["id"]
            if m["role"] == "system":
                print(f"\n── {m['content']} ──")
            else:
                print(f"\n[{m['speaker']}]\n{m['content']}")
        stop.wait(0.5)


def _library(args) -> None:
    lib = Library(args.library)
    if args.action == "status":
        stats = lib.stats()
        if not stats:
            print("The library is empty. Try: ai-room library fetch astronomy")
        for key, st in sorted(stats.items()):
            print(f"{key:24} {st['docs']:4} books  {st['passages']:6} passages  {st['questions']:5} bank questions")
    elif args.action == "fetch":
        for subject in args.args:
            if subject not in SUBJECTS:
                raise SystemExit(f"Unknown subject '{subject}'. Known: {', '.join(sorted(SUBJECTS))}")
            print(f"Fetching textbooks for {SUBJECTS[subject].name} …")
            summary = fetch_subject(lib, subject, progress=lambda m, i, t: print(f"  [{i}/{t}] {m}", flush=True))
            print(f"  added {summary['added']}, already had {summary['skipped']}" + (f", problems: {summary['errors'][:2]}" if summary["errors"] else ""))
    elif args.action == "import":
        if len(args.args) != 2:
            raise SystemExit("usage: ai-room library import SUBJECT PATH")
        subject, path = args.args
        if subject not in SUBJECTS:
            raise SystemExit(f"Unknown subject '{subject}'")
        for r in lib.import_path(subject, path):
            print(("  ✗ " + r["title"] + ": " + r["error"]) if "error" in r else f"  ✓ {r['title']} ({r['chars']:,} characters)")
    elif args.action == "search":
        if len(args.args) < 2:
            raise SystemExit("usage: ai-room library search SUBJECT QUESTION…")
        for p in lib.search(args.args[0], " ".join(args.args[1:]), k=4):
            print(f"\n[{p.score:.1f}] {p.title} — {p.heading}\n{p.text[:500]}")
    elif args.action == "clear":
        for s in args.args:
            lib.clear_subject(s)
            print(f"Cleared {s}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ai-room")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--db", default="data/ai_room.db")

    r = sub.add_parser("run", help="run a classroom in the terminal")
    r.add_argument("--db", default="data/ai_room.db")
    r.add_argument("--library", default="data/library.db")
    r.add_argument("--resume", help="id of an existing classroom")
    r.add_argument("--name", default="Terminal Classroom")
    r.add_argument("--student", default="ollama:qwen2.5:14b", help="backend:model (≤27B), or 'demo'")
    r.add_argument("--student-params", type=float, help="student size in B if not in its name")
    r.add_argument("--persona", default="patient", choices=list(PRESETS))
    r.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    r.add_argument("--questions", type=int, default=10)
    r.add_argument("--attempts", type=int, default=5)
    r.add_argument("--no-fetch", action="store_true", help="don't download textbooks automatically")
    r.add_argument("--until", choices=["class", "all"], default="class")

    lb = sub.add_parser("library", help="manage the textbook library")
    lb.add_argument("action", choices=["status", "fetch", "import", "search", "clear"])
    lb.add_argument("args", nargs="*")
    lb.add_argument("--library", default="data/library.db")

    bk = sub.add_parser("bank", help="import a question bank (JSON / JSON-lines)")
    bk.add_argument("action", choices=["import"])
    bk.add_argument("file")
    bk.add_argument("--subject")
    bk.add_argument("--library", default="data/library.db")

    t = sub.add_parser("train", help="really fine-tune a small open model on the CPU (LoRA), with a before/after exam")
    t.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct", help="Hugging Face model id or local folder")
    t.add_argument("--subjects", default="math", help="comma-separated subjects with exact problem generators")
    t.add_argument("--grades", default="1-3", help="classes, e.g. 1-3 or 2,5")
    t.add_argument("--examples", type=int, default=800, help="training problems (answers computed by code)")
    t.add_argument("--eval", type=int, default=100, help="held-out exam questions (0 to skip)")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--max-minutes", type=float, default=0, help="stop training after this long (0 = no limit)")
    t.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    t.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--out", default="data/trained/model")

    bo = sub.add_parser("boost", help="make a model answer better WITHOUT training (notebook, mistakes, calculator, voting) and measure it")
    bo.add_argument("--student", required=True, help="backend:model, e.g. ollama:qwen2.5:1.5b or openai:qwen2.5-1.5b-instruct (≤27B)")
    bo.add_argument("--base-url", help="server address, e.g. http://127.0.0.1:8080/v1 for llama.cpp")
    bo.add_argument("--student-params", type=float, help="model size in billions if it is not in the name")
    bo.add_argument("--subjects", default="math")
    bo.add_argument("--grades", default="1-3")
    bo.add_argument("--eval", type=int, default=100, help="held-out exam questions (same exam as `ai-room train` for the same seed)")
    bo.add_argument("--study", type=int, default=300, help="worked examples the student studies into its notebook")
    bo.add_argument("--practice", type=int, default=60, help="practice questions; each mistake becomes a notebook page")
    bo.add_argument("--pages", type=int, default=3, help="notebook pages re-read before each question")
    bo.add_argument("--methods", default="plain,notes,mistakes,calculator", help="stacked in order; any of plain,notes,mistakes,calculator,vote")
    bo.add_argument("--votes", type=int, default=5)
    bo.add_argument("--parallel", type=int, default=1, help="questions in flight at once (helps on servers that batch requests)")
    bo.add_argument("--max-tokens", type=int, default=256)
    bo.add_argument("--seed", type=int, default=7)
    bo.add_argument("--out", default="data/boost/report.json")

    q = sub.add_parser("ask", help="ask a model saved by `ai-room train` a question")
    q.add_argument("model_dir", help="e.g. data/trained/model/merged")
    q.add_argument("question")

    a = p.parse_args(argv)

    if a.cmd == "train":
        from .train_cpu import TrainConfig, parse_grades, train
        rep = train(TrainConfig(model=a.model, subjects=[x.strip() for x in a.subjects.split(",") if x.strip()], grades=parse_grades(a.grades), examples=a.examples,
                                eval_examples=a.eval, skip_eval=a.eval == 0, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, lora_r=a.lora_r, lora_alpha=2 * a.lora_r,
                                max_minutes=a.max_minutes, threads=a.threads, dtype=a.dtype, seed=a.seed, out=a.out))
        if "before" in rep:
            print(f"\nHeld-out exam: {rep['before']['accuracy']:.1%} before → {rep['after']['accuracy']:.1%} after training "
                  f"({rep['train']['steps']} steps, {rep['train']['seconds'] / 60:.1f} min, loss {rep['train']['first_loss']} → {rep['train']['last_loss']}).")
        return
    if a.cmd == "boost":
        from .boost import BoostConfig, run, summary_table
        from .llm import ModelTooLargeError, check_student_size, make_llm
        from .train_cpu import parse_grades
        spec = _student(a.student)
        if spec["backend"] == "demo":
            raise SystemExit("boost needs a real model (ollama:… or openai:…); the demo student only copies the answer key")
        llm = make_llm(spec["backend"], spec["model"], a.base_url)
        try:
            check_student_size(llm, a.student_params)
        except ModelTooLargeError as e:
            raise SystemExit(str(e))
        rep = run(llm, BoostConfig(subjects=[x.strip() for x in a.subjects.split(",") if x.strip()], grades=parse_grades(a.grades), eval_examples=a.eval,
                                   study_examples=a.study, practice=a.practice, pages=a.pages, votes=a.votes, methods=[m.strip() for m in a.methods.split(",") if m.strip()],
                                   parallel=a.parallel, max_tokens=a.max_tokens, seed=a.seed, out=a.out))
        print("\n" + summary_table(rep))
        return
    if a.cmd == "ask":
        from .train_cpu import ask
        print(ask(a.model_dir, a.question))
        return

    if a.cmd == "serve":
        import uvicorn

        from .server import create_app
        os.environ.setdefault("AI_ROOM_LIBRARY", os.path.join(os.path.dirname(a.db) or ".", "library.db"))
        uvicorn.run(create_app(a.db), host=a.host, port=a.port)
        return
    if a.cmd == "library":
        _library(a)
        return
    if a.cmd == "bank":
        lib = Library(a.library)
        with open(a.file, encoding="utf-8") as f:
            items = parse_bank_text(f.read(), a.subject)
        by: dict[str, list] = {}
        for subj, item in items:
            by.setdefault(subj, []).append(item)
        print("added", {s: lib.add_questions(s, q, source=os.path.basename(a.file)) for s, q in by.items()} or "nothing (check the format and --subject)")
        return

    store = Store(a.db)
    library = Library(a.library)
    if a.resume:
        cid = a.resume
    else:
        cfg = ClassConfig.from_dict({
            "student": _student(a.student), "student_params_b": a.student_params, "persona": a.persona, "subjects": a.subjects.split(","),
            "questions_per_exam": a.questions, "max_attempts": a.attempts, "auto_fetch": not a.no_fetch,
        })
        cid = store.create_classroom(a.name, cfg.to_dict(redact=False))
        print(f"Created classroom {cid} (resume later with --resume {cid})")
    try:
        room = Classroom(store, cid, library)
    except Exception as e:
        print(f"Cannot open classroom: {e}", file=sys.stderr)
        raise SystemExit(1)
    stop = threading.Event()
    printer = threading.Thread(target=_print_new, args=(store, cid, stop), daemon=True)
    printer.start()
    try:
        room.run_all() if a.until == "all" else room.run_class()
    except KeyboardInterrupt:
        print("\nStopped. Resume with --resume", cid)
    finally:
        stop.set()
        printer.join(timeout=2)
        print("\nReport:", room.report())


if __name__ == "__main__":
    main()
