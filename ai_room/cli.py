"""Command line entry point.

    ai-room serve                      # web UI on http://127.0.0.1:8765
    ai-room run --student ollama:qwen2.5:14b --teacher ollama:gemma3:27b \
                --subjects math,physics,logic,code,research --persona socratic
"""

from __future__ import annotations

import argparse
import sys
import threading

from .classroom import ClassConfig, Classroom
from .curriculum import DEFAULT_SUBJECTS
from .memory import Store
from .personas import DEFAULT_PERSONA, TEACHER_PERSONAS


def _model(spec: str) -> dict:
    backend, _, model = spec.partition(":")
    if not model:
        raise SystemExit(f"Model spec must look like backend:model, got '{spec}'")
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ai-room")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--db", default="data/ai_room.db")

    r = sub.add_parser("run", help="run a classroom in the terminal")
    r.add_argument("--db", default="data/ai_room.db")
    r.add_argument("--resume", help="id of an existing classroom")
    r.add_argument("--name", default="Terminal Classroom")
    r.add_argument("--student", default="ollama:qwen2.5:14b", help="backend:model (≤27B)")
    r.add_argument("--student-params", type=float, help="student size in B if not in its name")
    r.add_argument("--teacher", default="ollama:gemma3:27b", help="backend:model")
    r.add_argument("--persona", default=DEFAULT_PERSONA, choices=[*TEACHER_PERSONAS, "custom"])
    r.add_argument("--custom-style", help="teacher personality when --persona custom")
    r.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    r.add_argument("--questions", type=int, default=10)
    r.add_argument("--attempts", type=int, default=5)
    r.add_argument("--until", choices=["class", "all"], default="class")
    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from .server import create_app
        uvicorn.run(create_app(a.db), host=a.host, port=a.port)
        return

    store = Store(a.db)
    if a.resume:
        cid = a.resume
    else:
        cfg = ClassConfig.from_dict({
            "student": _model(a.student), "teacher": _model(a.teacher), "student_params_b": a.student_params,
            "persona": a.persona, "custom_style": a.custom_style, "subjects": a.subjects.split(","),
            "questions_per_exam": a.questions, "max_attempts": a.attempts,
        })
        cid = store.create_classroom(a.name, cfg.to_dict(redact=False))
        print(f"Created classroom {cid} (resume later with --resume {cid})")
    try:
        room = Classroom(store, cid)
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
