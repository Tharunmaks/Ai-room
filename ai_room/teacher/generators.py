"""Exact problem generators — the part of the teacher that needs no AI.

A *family* is one kind of problem (e.g. "solve a linear equation"). It carries
a short lesson card and a `make(rng, t)` function that builds a fresh problem
whose answer key is computed by code, so it is correct by construction. `t`
is 0..1: how far into the family's grade range the exam is (bigger numbers,
harder variants).
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable


@dataclass(frozen=True)
class Family:
    id: str
    title: str
    card: str
    subjects: dict[str, tuple[int, int]]  # subject key -> (first class, last class)
    make: Callable[[random.Random, float], dict]
    pitfalls: str = ""


FAMILIES: dict[str, Family] = {}


def family(id: str, title: str, subjects: dict[str, tuple[int, int]], card: str, pitfalls: str = ""):
    def deco(fn):
        if id in FAMILIES:
            raise ValueError(f"duplicate family {id}")
        FAMILIES[id] = Family(id, title, card.strip(), subjects, fn, pitfalls.strip())
        return fn
    return deco


def families_for(subject: str, grade: int) -> list[Family]:
    return [f for f in FAMILIES.values() if subject in f.subjects and f.subjects[subject][0] <= grade <= f.subjects[subject][1]]


def has_generators(subject: str) -> bool:
    return any(subject in f.subjects for f in FAMILIES.values())


def generate(fam: Family, subject: str, grade: int, rng: random.Random) -> dict:
    lo, hi = fam.subjects[subject]
    t = 0.0 if hi == lo else (grade - lo) / (hi - lo)
    q = fam.make(rng, max(0.0, min(1.0, t)))
    q.setdefault("type", "numeric")
    if q["type"] == "numeric" and re.search(r"significant figures|decimal|\(3 decimals\)", q["question"]):
        q["tol"] = 0.01  # the key is rounded, so accept answers within 1%
    q["concept"] = f"fam:{fam.id}"
    q["family"] = fam.title
    return q


# -- helpers used by the family modules ----------------------------------------

def Q(type: str, question: str, answer, explanation: str = "", **extra) -> dict:
    return {"type": type, "question": question, "answer": str(answer), "explanation": explanation, **extra}


def scale(t: float, lo: int, hi: int) -> int:
    return int(round(lo + t * (hi - lo)))


def fmt(x: float, sig: int = 3) -> str:
    """Format a float to `sig` significant figures; integers stay integers."""
    if isinstance(x, int) or (isinstance(x, float) and abs(x - round(x)) < 1e-9 and abs(x) < 1e12):
        return str(int(round(x)))
    if x == 0:
        return "0"
    digits = sig - int(math.floor(math.log10(abs(x)))) - 1
    r = round(x, digits)
    if abs(r) >= 1e6 or abs(r) < 1e-3:
        return f"{r:.{max(sig - 1, 0)}e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{r:.{max(digits, 0)}f}"


def frac(f: Fraction) -> str:
    return str(f)


def mcq(rng: random.Random, question: str, correct: str, wrong: list[str], explanation: str = "", **extra) -> dict:
    """Multiple choice with the correct option at a random position."""
    wrong = [w for i, w in enumerate(wrong) if w != correct and w not in wrong[:i]][:3]
    opts = wrong + [correct]
    rng.shuffle(opts)
    letters = "ABCD"
    return Q("mcq", question, letters[opts.index(correct)], explanation,
             options=[f"{letters[i]}) {o}" for i, o in enumerate(opts)], **extra)


def yesno(rng: random.Random, question: str, truth: bool, explanation: str = "") -> dict:
    return mcq(rng, question, "Yes" if truth else "No", ["No" if truth else "Yes"], explanation)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            return False
    return True


def prime_factors(n: int) -> list[int]:
    out, p = [], 2
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out


def poly_str(coeffs: list[int], var: str = "x") -> str:
    """[a, b, c] -> 'ax² + bx + c' (highest power first)."""
    sup = {0: "", 1: "", 2: "²", 3: "³", 4: "⁴", 5: "⁵"}
    terms, n = [], len(coeffs) - 1
    for i, c in enumerate(coeffs):
        p = n - i
        if c == 0:
            continue
        body = (var + sup[p]) if p > 0 else ""
        mag = abs(c)
        coef = "" if (mag == 1 and p > 0) else str(mag)
        t = f"{coef}{body}"
        terms.append(("-" if c < 0 else "+", t))
    if not terms:
        return "0"
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, t in terms[1:]:
        s += f" {'−' if sign == '-' else '+'} {t}"
    return s.replace("-", "−")


def seed_for(*parts) -> int:
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def sci(x: float, sig: int = 3) -> str:
    """Readable scientific notation for question text: 6.42×10^23."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if -3 <= e <= 5:
        return f"{x:.{sig}g}"
    m = f"{x / 10 ** e:.{sig - 1}f}".rstrip("0").rstrip(".")
    return f"{m}×10^{e}"
