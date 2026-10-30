import random
import re
from fractions import Fraction

import pytest

import ai_room.teacher  # noqa: F401  (registers every family)
from ai_room.curriculum import SUBJECTS
from ai_room.grading import cloze_match, exact_match, mcq_match, numeric_match, run_code_tests
from ai_room.teacher.code_tasks import TASKS
from ai_room.teacher.generators import FAMILIES, families_for, generate, has_generators


def all_cases():
    for fam in FAMILIES.values():
        for subject, (lo, hi) in fam.subjects.items():
            assert subject in SUBJECTS, (fam.id, subject)
            assert 1 <= lo <= hi <= 12, fam.id
            yield fam, subject, lo, hi


def test_every_family_produces_valid_questions_in_every_class_it_covers():
    count = 0
    for fam, subject, lo, hi in all_cases():
        for g in range(lo, hi + 1):
            for seed in range(6):
                q = generate(fam, subject, g, random.Random(seed))
                count += 1
                assert q["question"].strip() and q["answer"] != "" and q["type"] in ("numeric", "mcq", "cloze", "code"), (fam.id, q)
                assert not re.search(r"\b(None|nan|inf)\b", q["question"]), q["question"]
                if q["type"] == "numeric":
                    assert numeric_match(q["answer"], q["answer"], exact=False if q.get("tol") else None), q
                elif q["type"] == "mcq":
                    assert q["answer"] in "ABCD" and len(q["options"]) >= 2
                    assert len({o[3:] for o in q["options"]}) == len(q["options"]), "duplicate options"
                    assert any(o.startswith(q["answer"]) for o in q["options"])
                elif q["type"] == "cloze":
                    assert cloze_match(q["answer"], q["answer"]) or exact_match(q["answer"], q["answer"])
                else:
                    assert q["tests"] and q["function_name"] in q["question"]
    assert count > 5000


def test_questions_are_reproducible_and_varied():
    fam = FAMILIES["math.lineq"]
    a = generate(fam, "math", 6, random.Random(5))
    assert a == generate(fam, "math", 6, random.Random(5))
    assert len({generate(fam, "math", 6, random.Random(s))["question"] for s in range(30)}) > 20


def test_subject_coverage():
    assert len(FAMILIES) >= 190
    with_gen = [k for k in SUBJECTS if has_generators(k)]
    assert len(with_gen) >= 85
    for core in ("math", "physics", "logic", "code", "research", "chemistry", "biology", "cs"):
        assert all(families_for(core, g) for g in range(1, 13)), core
    for key in ("spanish", "french", "german", "mandarin", "hindi"):
        assert all(families_for(key, g) for g in range(1, 13)), key


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.name)
def test_reference_solutions_pass_their_own_tests(task):
    for seed in range(5):
        tests = task.build_tests(random.Random(seed))
        passed, total, log = run_code_tests("import heapq, math, itertools, collections\n" + task.source(), tests, timeout=10)
        assert total > 0 and passed == total, log


def test_wrong_code_fails_its_tests():
    task = next(t for t in TASKS if t.name == "gcd")
    tests = task.build_tests(random.Random(1))
    assert run_code_tests("def gcd(a, b):\n    return 1", tests)[0] < len(tests)


# ---- independent re-computation of answer keys --------------------------------------------------

def poly_to_py(s: str) -> str:
    s = s.replace("−", "-").replace("²", "**2").replace("³", "**3").replace("⁴", "**4").replace("⁵", "**5")
    s = re.sub(r"(\d)([a-z])", r"\1*\2", s)
    return s


def questions(fid, subject, grade, n=25):
    fam = FAMILIES[fid]
    return [generate(fam, subject, grade, random.Random(s)) for s in range(n)]


def test_derivative_keys_match_numerical_differentiation():
    for q in questions("math.derivative", "math", 9):
        m = re.match(r"If f\(x\) = (.+), what is f′\((-?\d+)\)\?", q["question"])
        f = eval("lambda x: " + poly_to_py(m.group(1)))
        x0, h = int(m.group(2)), 1e-4
        assert abs((f(x0 + h) - f(x0 - h)) / (2 * h) - float(q["answer"])) < 1e-2, q


def test_integral_keys_match_numerical_integration():
    for q in questions("math.integral", "math", 10):
        m = re.match(r"Evaluate ∫ from 0 to (\d+) of \((.+)\) dx", q["question"])
        f = eval("lambda x: " + poly_to_py(m.group(2)))
        k, n = int(m.group(1)), 2000
        simpson = sum((f(i * k / n) + 4 * f((i + .5) * k / n) + f((i + 1) * k / n)) * (k / n) / 6 for i in range(n))
        assert abs(simpson - float(Fraction(q["answer"]))) < 1e-6, q


def test_quadratic_keys_match_the_quadratic_formula():
    import math
    for q in questions("math.quadratic", "math", 8, 60):
        f = eval("lambda x: " + poly_to_py(q["question"].rsplit(" of ", 1)[1].split(" = 0")[0]))
        # recover a, b, c by evaluating the polynomial, independently of how it was printed
        c_, a_, b_ = f(0), (f(1) + f(-1) - 2 * f(0)) / 2, (f(1) - f(-1)) / 2
        disc = b_ * b_ - 4 * a_ * c_
        want = float(q["answer"])
        if "larger root" in q["question"]:
            assert abs((-b_ + math.sqrt(disc)) / (2 * a_) - want) < 1e-9, q
        elif "sum of the roots" in q["question"]:
            assert abs(-b_ / a_ - want) < 1e-9
        elif "product" in q["question"]:
            assert abs(c_ / a_ - want) < 1e-9
        else:
            assert abs(disc - want) < 1e-9


def test_system_keys_satisfy_both_equations():
    for q in questions("math.systems", "math", 7, 40):
        m = re.match(r"Solve the system: (.+) = (-?\d+) and (.+) = (-?\d+)\. What is (.+)\?", q["question"])
        sol = re.search(r"x = (-?\d+), y = (-?\d+)", q["explanation"])
        x, y = int(sol.group(1)), int(sol.group(2))
        f1 = eval("lambda x, y: " + poly_to_py(m.group(1)))
        f2 = eval("lambda x, y: " + poly_to_py(m.group(3)))
        assert f1(x, y) == int(m.group(2)) and f2(x, y) == int(m.group(4)), q
        want = {"x": x, "y": y, "x + y": x + y}[m.group(5)]
        assert want == int(q["answer"])


def test_matrix_determinant_keys():
    for q in questions("math.matrices", "math", 9, 60):
        m = re.search(r"determinant of (?:the 3×3 matrix )?\[([^\]]+)\]", q["question"])
        if not m:
            continue
        rows = [[int(v) for v in r.split()] for r in m.group(1).split(";")]
        if len(rows) == 2:
            det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        else:
            r = rows
            det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
        assert det == int(q["answer"]), q


def test_eigenvalue_keys_satisfy_the_characteristic_equation():
    for q in questions("math.eigen", "math", 11, 40):
        rows = [[int(v) for v in r.split()] for r in re.search(r"\[([^\]]+)\]", q["question"]).group(1).split(";")]
        lam = int(q["answer"])
        det = (rows[0][0] - lam) * (rows[1][1] - lam) - rows[0][1] * rows[1][0]
        assert det == 0, q


def test_number_theory_keys():
    for q in questions("nt.congruence", "number_theory", 6, 40):
        a, b, m = map(int, re.search(r"with (\d+)x ≡ (\d+) \(mod (\d+)\)", q["question"]).groups())
        x = int(q["answer"])
        assert (a * x - b) % m == 0 and 0 <= x < m
        assert all((a * y - b) % m for y in range(x))
    for q in questions("nt.crt", "number_theory", 9, 40):
        m = re.match(r"Find the smallest positive x with x ≡ (\d+) \(mod (\d+)\) and x ≡ (\d+) \(mod (\d+)\)\.", q["question"])
        if m:
            a, m1, b, m2 = map(int, m.groups())
            x = int(q["answer"])
            assert x % m1 == a and x % m2 == b and all(not (y % m1 == a and y % m2 == b) for y in range(1, x))


def test_sql_keys_come_from_sqlite_and_logic_keys_from_search():
    for q in questions("cs.sql", "databases", 8, 30):
        assert q["answer"].lstrip("-").isdigit()
    for q in questions("logic.knights", "logic", 7, 30):
        assert (q["type"] == "mcq" and q["answer"] in "ABCD") or q["answer"] in ("0", "1", "2", "3")


def test_language_keys():
    from ai_room.teacher.gen_lang import ES_PRES, _de_participle, _de_present

    assert _de_present("arbeiten") == ["arbeite", "arbeitest", "arbeitet", "arbeiten", "arbeitet", "arbeiten"]
    assert _de_present("tanzen")[1] == "tanzt" and _de_present("spielen")[1] == "spielst"
    assert _de_participle("arbeiten") == "gearbeitet" and _de_participle("spielen") == "gespielt"
    assert ES_PRES["ar"] == ["o", "as", "a", "amos", "áis", "an"]
    es = FAMILIES["lang.conj.spanish"]
    answers = {generate(es, "spanish", 5, random.Random(s))["question"]: generate(es, "spanish", 5, random.Random(s))["answer"] for s in range(300)}
    assert answers["Conjugate the Spanish verb 'hablar' for 'yo' in the present tense."] == "hablo"
    fr = FAMILIES["lang.conj.french"]
    q = next(generate(fr, "french", 4, random.Random(s)) for s in range(400) if "'finir' for 'nous' in the present" in generate(fr, "french", 4, random.Random(s))["question"])
    assert q["answer"] == "finissons"


def test_subject_modes_match_what_the_readme_promises():
    from ai_room.curriculum import NUM_GRADES
    full = [k for k in SUBJECTS if all(families_for(k, g) for g in range(1, NUM_GRADES + 1))]
    none = [k for k in SUBJECTS if not has_generators(k)]
    assert len(full) >= 79 and len(none) <= 12 and len(TASKS) >= 53
    assert set(none) >= {"history", "philosophy", "ethics", "classics"}
