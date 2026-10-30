"""Deterministic grading helpers: final-answer extraction, numeric and
multiple-choice comparison, and sandboxed execution of code answers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

_FINAL = re.compile(r"final\s*answer\s*[:=\-]\s*(.+)", re.IGNORECASE)
_NUM = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*[eE]\s*[-+]?\d+|\s*[x×]\s*10\^\s*[-+]?\d+)?(?:\s*/\s*\d+)?")


def extract_final(text: str) -> str:
    """The student's final answer: the last 'FINAL ANSWER:' line, else the last line."""
    matches = _FINAL.findall(text or "")
    if matches:
        ans = matches[-1].strip()
    else:
        lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
        ans = lines[-1] if lines else ""
    return ans.strip().strip("*").strip("`").strip().rstrip(".")


def parse_number(text: str) -> float | None:
    s = (text or "").replace("−", "-").replace("\\times", "x").replace("$", "")
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    m = _NUM.search(s)
    if not m:
        return None
    tok = m.group(0).replace(",", "").replace(" ", "")
    pct = s[m.end():m.end() + 2].strip().startswith("%")
    try:
        if re.search(r"[x×]10\^", tok):
            base, exp = re.split(r"[x×]10\^", tok)
            val = float(base) * 10 ** float(exp)
        elif "/" in tok:
            val = float(Fraction(tok))
        else:
            val = float(tok)
    except (ValueError, ZeroDivisionError):
        return None
    return val / 100 if pct else val


def numeric_match(given: str, expected: str, rel_tol: float = 0.01) -> bool:
    g, e = parse_number(given), parse_number(expected)
    if g is None or e is None:
        return False
    if e == 0:
        return abs(g) < 1e-6
    if abs(g - e) <= rel_tol * abs(e):
        return True
    # Accept percentages written as either 25% or 25 when the key says 0.25 (and vice versa).
    return abs(g / 100 - e) <= rel_tol * abs(e) or abs(g * 100 - e) <= rel_tol * abs(e)


def mcq_match(given: str, expected: str) -> bool:
    letter = re.search(r"\b([A-H])\b", (given or "").upper().replace("(", " ").replace(")", " "))
    exp = re.search(r"[A-H]", (expected or "").upper())
    return bool(letter and exp and letter.group(1) == exp.group(0))


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text or ""


def run_code_tests(code: str, tests: list[str], timeout: float = 10.0) -> tuple[int, int, str]:
    """Run the student's code followed by each assert in its own process.

    Returns (passed, total, log). This runs untrusted model output: only
    isolation is a separate Python process with a timeout in an empty temp
    directory, so run AI Room inside a container or VM if that matters to you.
    """
    if not tests:
        return 0, 0, "no tests"
    passed, log = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for i, test in enumerate(tests):
            path = os.path.join(tmp, f"t{i}.py")
            with open(path, "w") as f:
                f.write(code + "\n\n" + test + "\n")
            try:
                r = subprocess.run([sys.executable, "-I", path], cwd=tmp, capture_output=True,
                                   text=True, timeout=timeout, env={"PATH": os.environ.get("PATH", "")})
                ok = r.returncode == 0
                if not ok:
                    log.append(f"FAIL {test}: {(r.stderr or r.stdout).strip().splitlines()[-1:] }")
            except subprocess.TimeoutExpired:
                ok = False
                log.append(f"TIMEOUT {test}")
            passed += ok
    return passed, len(tests), "\n".join(log) or "all tests passed"
