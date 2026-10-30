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

    a = p.parse_args(argv)

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
