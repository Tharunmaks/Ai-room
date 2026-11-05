"""Better answers without training: the student studies, keeps a notebook and learns from its mistakes. Its weights never change.

It is measured the same way as `ai-room train`: the same held-out exam (the same questions with the same seed, the same
grader) is taken with and without each method, so the two approaches can be compared directly. It runs on any backend
AI Room supports — Ollama or an OpenAI-compatible server such as llama.cpp — so it needs no PyTorch and works on a phone.

The methods are stacked one at a time, so you can see what each one adds:
  plain       the model alone.
  notes       before each question it re-reads the most similar pages of its notebook: worked examples it studied, whose
              answers were computed by code. Pages are found with BM25. This is in-context learning — how the classroom works.
  mistakes    first it sits a practice test; every wrong answer becomes a notebook page with the correct method.
  calculator  it may give its final answer as an arithmetic expression, which AI Room's calculator works out exactly.
  vote        it answers several times and the most common answer wins (self-consistency).

The exam tells the student each question's topic (as classroom exams do), and that topic is part of the notebook search.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .grading import extract_final, grade_question
from .llm import LLM, Message
from .memory import tokenize
from .train_cpu import make_problems, prompt_of, target_of

METHODS = ("plain", "notes", "mistakes", "calculator", "vote")


@dataclass
class BoostConfig:
    subjects: list[str] = field(default_factory=lambda: ["math"])
    grades: list[int] = field(default_factory=lambda: [1, 2, 3])
    eval_examples: int = 100        # the held-out exam; with the same seed it is the exact exam `ai-room train` uses
    study_examples: int = 300       # worked examples written into the notebook
    practice: int = 60              # practice questions; each mistake becomes a notebook page
    pages: int = 3                  # notebook pages re-read before each question
    votes: int = 5                  # answers per question for "vote"
    methods: list[str] = field(default_factory=lambda: ["plain", "notes", "mistakes", "calculator"])
    parallel: int = 1               # questions answered at the same time (more can raise total tokens/second)
    max_tokens: int = 256
    seed: int = 7
    out: str = "data/boost/report.json"


# ------------------------------------------------------------------ the notebook
class Notebook:
    """Pages the student has written, searchable with BM25."""

    def __init__(self):
        self.pages: list[dict] = []

    def add(self, text: str, topic: str, kind: str):
        toks = tokenize(text + " " + topic)
        self.pages.append({"text": text, "topic": topic, "kind": kind, "tf": Counter(toks), "len": len(toks)})

    def search(self, query: str, k: int) -> list[dict]:
        if not self.pages or k <= 0:
            return []
        q = set(tokenize(query))
        n = len(self.pages)
        avg = sum(p["len"] for p in self.pages) / n
        df = Counter(w for p in self.pages for w in p["tf"])
        scored = []
        for i, p in enumerate(self.pages):
            s = 0.0
            for w in q:
                f = p["tf"].get(w)
                if f:
                    s += math.log(1 + (n - df[w] + .5) / (df[w] + .5)) * f * 2.2 / (f + 1.2 * (.25 + .75 * p["len"] / avg))
            if s > 0:
                scored.append((s + (0.5 if p["kind"] == "mistake" else 0), -i, p))   # a lesson learned the hard way is worth a look
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, _, p in scored[:k]]


def study_page(q: dict) -> str:
    return f"Q: {q['question']}" + ("\n" + "\n".join(q["options"]) if q.get("options") else "") + f"\nA: {target_of(q)}"


def mistake_page(q: dict, given: str) -> str:
    return (f"Q: {q['question']}" + ("\n" + "\n".join(q["options"]) if q.get("options") else "")
            + f"\nI answered {given or '(nothing)'}, which was wrong. Correct working: {target_of(q)}")


CALC_TIP = "\nTip: you may write the final answer as an arithmetic expression, e.g. FINAL ANSWER: (120 - 45) * 3 — AI Room's calculator works it out exactly."


def build_prompt(q: dict, notebook: Notebook | None, pages: int, calculator: bool) -> str:
    base = prompt_of(q) + (CALC_TIP if calculator and q["type"] == "numeric" else "")
    found = notebook.search(q["question"] + " " + q.get("family", ""), pages) if notebook else []
    if not found:
        return f"Topic: {q.get('family', '')}\n{base}" if notebook else base
    notes = "\n\n".join(f"Page {i} ({p['topic']}{', a mistake you made' if p['kind'] == 'mistake' else ''})\n{p['text']}" for i, p in enumerate(found, 1))
    return f"Pages from your notebook (worked examples you studied):\n\n{notes}\n\nUse the same method. Topic: {q.get('family', '')}\n{base}"


# ------------------------------------------------------------------ answering
_SIMPLE = re.compile(r"\s*-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*%?\s*")
_EXPR = re.compile(r"[\d\s.+\-*/×÷^()]+")


def apply_calculator(q: dict, reply: str) -> tuple[str, bool]:
    """If the final answer is an arithmetic expression, replace it with its exact value."""
    if q["type"] != "numeric":
        return reply, False
    given = extract_final(reply).strip().rstrip(".")
    if not given or _SIMPLE.fullmatch(given) or not _EXPR.fullmatch(given) or not re.search(r"\d\s*[-+*/×÷^]\s*[\d(]", given):
        return reply, False
    from .tools.t_math import evaluate, fmt
    try:
        v = evaluate(given)
    except Exception:  # an unparseable expression is just a wrong answer
        return reply, False
    if isinstance(v, complex) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return reply, False
    v = fmt(round(v, 10)) if isinstance(v, float) else v
    return reply + f"\n[calculator: {given} = {v}]\nFINAL ANSWER: {v}", True


def _norm(ans: str) -> str:
    return re.sub(r"\s+", " ", ans.strip().strip(".").lower())


class Runner:
    def __init__(self, llm: LLM, cfg: BoostConfig, log: Callable[[str], None]):
        self.llm, self.cfg, self.log = llm, cfg, log

    def ask(self, prompt: str, temperature: float) -> tuple[str, dict]:
        t = time.perf_counter()
        try:
            text, usage = self.llm.chat_with_usage([Message("user", prompt)], temperature=temperature, max_tokens=self.cfg.max_tokens)
        except Exception as e:  # a dropped request counts as a wrong answer; the run goes on
            text, usage = f"(the model did not answer: {e.__class__.__name__})", {}
        usage["wall_seconds"] = time.perf_counter() - t
        return text, usage

    def answer(self, q: dict, notebook: Notebook | None, calculator: bool, votes: int) -> dict:
        prompt = build_prompt(q, notebook, self.cfg.pages, calculator)
        replies, usages = [], []
        for i in range(max(1, votes)):
            text, u = self.ask(prompt, 0.0 if votes <= 1 else 0.7)
            if calculator:
                text, _ = apply_calculator(q, text)
            replies.append(text)
            usages.append(u)
        if len(replies) > 1:
            finals = [_norm(extract_final(r)) for r in replies]
            best = Counter(f for f in finals if f).most_common(1)
            reply = next((r for r, f in zip(replies, finals) if best and f == best[0][0]), replies[0])
        else:
            reply = replies[0]
        score, given = grade_question(q, reply)
        return {"question": q["question"], "family": q.get("family"), "expected": q["answer"], "given": given, "correct": bool(score),
                "used_calculator": "[calculator:" in reply, "reply": reply[-400:], "usage": usages}

    def sit(self, name: str, problems: list[dict], notebook: Notebook | None, calculator: bool = False, votes: int = 1) -> dict:
        t0 = time.perf_counter()
        done = []

        def one(q):
            r = self.answer(q, notebook, calculator, votes)
            done.append(1)
            if len(done) % 10 == 0 or len(done) == len(problems):
                self.log(f"  {name}: {len(done)}/{len(problems)} answered")
            return r
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.parallel)) as ex:
            rows = list(ex.map(one, problems))
        wall = time.perf_counter() - t0
        us = [u for r in rows for u in r["usage"]]
        ptok = sum(u.get("prompt_tokens", 0) for u in us)
        ctok = sum(u.get("completion_tokens", 0) for u in us)
        csec = sum(u.get("completion_seconds", 0) for u in us)
        psec = sum(u.get("prompt_seconds", 0) for u in us)
        fam: dict[str, list[int]] = {}
        for r in rows:
            fam.setdefault(r["family"], []).append(int(r["correct"]))
        return {"name": name, "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)), "n": len(rows), "seconds": round(wall, 1),
                "speed": {"prompt_tokens": ptok, "completion_tokens": ctok,
                          "generation_tokens_per_second": round(ctok / csec, 1) if csec else None,          # one request at a time, as the engine measured it
                          "prompt_tokens_per_second": round(ptok / psec, 1) if psec else None,
                          "total_tokens_per_second_wall": round((ptok + ctok) / wall, 1) if wall and (ptok + ctok) else None,
                          "generated_tokens_per_second_wall": round(ctok / wall, 1) if wall and ctok else None},
                "by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(fam.items())}, "rows": rows}


# ------------------------------------------------------------------ the run
def run(llm: LLM, cfg: BoostConfig, log: Callable[[str], None] = print) -> dict:
    bad = [m for m in cfg.methods if m not in METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {', '.join(METHODS)}")
    exam = make_problems(cfg.subjects, cfg.grades, cfg.eval_examples, cfg.seed + 1000)
    seen = {q["question"] for q in exam}
    study = make_problems(cfg.subjects, cfg.grades, cfg.study_examples, cfg.seed, avoid=seen)
    seen |= {q["question"] for q in study}
    practice = make_problems(cfg.subjects, cfg.grades, cfg.practice, cfg.seed + 2000, avoid=seen) if "mistakes" in cfg.methods else []
    log(f"Student: {llm.name} · held-out exam: {len(exam)} questions (the same exam as `ai-room train` with seed {cfg.seed}) · weights are never changed")
    report: dict = {"config": asdict(cfg), "student": llm.name, "results": [], "started": time.time()}

    notebook = Notebook()
    calculator, votes, stack = False, 1, []
    for m in [x for x in METHODS if x in cfg.methods]:
        if m == "notes":
            for q in study:
                notebook.add(study_page(q), q.get("family", ""), "study")
            log(f"Studied {len(study)} worked examples → {len(notebook.pages)} notebook pages")
        elif m == "mistakes":
            log(f"Practice test ({len(practice)} questions, not part of the exam)…")
            pr = Runner(llm, cfg, log).sit("practice", practice, notebook if notebook.pages else None, calculator)
            wrong = [(q, r) for q, r in zip(practice, pr["rows"]) if not r["correct"]]
            for q, r in wrong:
                notebook.add(mistake_page(q, r["given"]), q.get("family", ""), "mistake")
            report["practice"] = {"accuracy": pr["accuracy"], "mistakes_written": len(wrong)}
            log(f"  practice: {pr['accuracy']:.0%} right · wrote {len(wrong)} mistake pages")
        elif m == "calculator":
            calculator = True
        elif m == "vote":
            votes = cfg.votes
        stack.append(m)
        label = " + ".join(stack)
        log(f"Exam: {label}")
        res = Runner(llm, cfg, log).sit(label, exam, notebook if notebook.pages else None, calculator, votes)
        report["results"].append(res)
        sp = res["speed"]
        log(f"  {label}: {res['accuracy']:.1%} · {res['seconds']:.0f}s · generation {sp['generation_tokens_per_second'] or '?'} tok/s · all tokens {sp['total_tokens_per_second_wall'] or '?'} tok/s")
    report["seconds_total"] = round(time.time() - report.pop("started"), 1)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log(f"Saved the full report (every question and answer) to {out}")
    return report


def summary_table(report: dict) -> str:
    lines = [f"{'method':<44}{'accuracy':>9}{'gen tok/s':>11}{'all tok/s':>11}{'seconds':>9}"]
    for r in report["results"]:
        sp = r["speed"]
        lines.append(f"{r['name']:<44}{r['accuracy']:>8.1%}{(sp['generation_tokens_per_second'] or 0):>11.1f}{(sp['total_tokens_per_second_wall'] or 0):>11.1f}{r['seconds']:>9.0f}")
    return "\n".join(lines)
