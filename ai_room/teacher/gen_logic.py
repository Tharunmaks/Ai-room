"""Logical-thinking families. Every answer is found by exhaustive search, so keys are exact."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

from .generators import Q, family, mcq, scale, yesno

IMP = lambda a, b: (not a) or b  # noqa: E731


# ---------------------------------------------------------------- patterns

@family("logic.patterns", "Number patterns", {"logic": (1, 5), "discrete_math": (1, 2), "research": (1, 2)},
        "Look at the differences between neighbours. Constant difference → arithmetic (add d). Constant ratio → geometric (multiply by r). "
        "Differences that themselves grow by a constant → quadratic (squares, triangular numbers). Each term the sum of the previous two → Fibonacci-like.",
        "Spotting a pattern from only two numbers — check it against every term.")
def patterns(rng, t):
    kind = rng.choice(["arith", "geo", "squares", "tri", "fib", "alt"][: 2 + int(t * 4)])
    if kind == "arith":
        a, d = rng.randint(1, 20), rng.randint(2, 12)
        seq = [a + i * d for i in range(5)]
        return Q("numeric", f"What comes next: {', '.join(map(str, seq))}, ?", a + 5 * d, f"add {d} each time")
    if kind == "geo":
        a, r = rng.randint(1, 5), rng.randint(2, 4)
        seq = [a * r ** i for i in range(5)]
        return Q("numeric", f"What comes next: {', '.join(map(str, seq))}, ?", a * r ** 5, f"multiply by {r}")
    if kind == "squares":
        k = rng.randint(1, 4)
        seq = [(k + i) ** 2 for i in range(5)]
        return Q("numeric", f"What comes next: {', '.join(map(str, seq))}, ?", (k + 5) ** 2, "perfect squares")
    if kind == "tri":
        seq = [(i + 1) * (i + 2) // 2 for i in range(5)]
        return Q("numeric", f"What comes next: {', '.join(map(str, seq))}, ?", 6 * 7 // 2, "triangular numbers: add 2, 3, 4, …")
    if kind == "fib":
        a, b = rng.randint(1, 4), rng.randint(2, 6)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
        return Q("numeric", f"What comes next: {', '.join(map(str, seq))}, ?", seq[-1] + seq[-2], "each term is the sum of the previous two")
    a, d1, d2 = rng.randint(1, 10), rng.randint(2, 6), rng.randint(1, 5)
    seq = [a]
    for i in range(5):
        seq.append(seq[-1] + (d1 if i % 2 == 0 else d2))
    return Q("numeric", f"What comes next: {', '.join(map(str, seq))}, ?", seq[-1] + d1, f"alternately add {d1} and {d2}")


# ---------------------------------------------------------------- propositional logic

def _expr(rng, names, depth):
    if depth == 0 or rng.random() < .2:
        v = rng.choice(names)
        return (v, v) if rng.random() < .75 else (f"¬{v}", f"(not {v})")
    op = rng.choice(["∧", "∨", "→", "⊕"])
    (s1, p1), (s2, p2) = _expr(rng, names, depth - 1), _expr(rng, names, depth - 1)
    py = {"∧": f"({p1} and {p2})", "∨": f"({p1} or {p2})", "→": f"IMP({p1}, {p2})", "⊕": f"({p1} != {p2})"}[op]
    return (f"({s1} {op} {s2})", py)


def _eval(py, env):
    return bool(eval(py, {"IMP": IMP}, env))


@family("logic.truth", "Evaluating logical statements", {"logic": (2, 6), "cs": (3, 5), "discrete_math": (1, 3)},
        "∧ (and) is true only if both sides are true; ∨ (or) is true if at least one is; ¬ flips the value; p → q is false only when p is true and q is false; p ⊕ q (xor) is true when exactly one side is true.",
        "Reading 'p → q' as 'p and q'.")
def truth(rng, t):
    names = ["p", "q", "r"][: 2 + (t > .5)]
    s, py = _expr(rng, names, 1 + int(t * 2))
    env = {n: rng.random() < .5 for n in names}
    given = ", ".join(f"{n} = {'True' if v else 'False'}" for n, v in env.items())
    res = _eval(py, env)
    return mcq(rng, f"Given {given}, what is the value of {s}?", "True" if res else "False", ["False" if res else "True"], "evaluate inside out")


@family("logic.truthtable", "Truth tables", {"logic": (4, 8), "cs": (4, 6)},
        "A statement with n variables has 2ⁿ rows in its truth table. Count the rows where the whole statement is true. A tautology is true in every row; a contradiction in none.",
        "Forgetting rows where a variable is false.")
def truthtable(rng, t):
    names = ["p", "q", "r"][: 2 + (rng.random() < .5)]
    while True:
        s, py = _expr(rng, names, 2)
        if sum(n in s for n in names) >= 2:
            break
    rows = sum(_eval(py, dict(zip(names, vals))) for vals in itertools.product([True, False], repeat=len(names)))
    return Q("numeric", f"In how many of the {2 ** len(names)} rows of the truth table is {s} true?", rows, "enumerate every row")


def _models(n_elems=3):
    return list(itertools.product(range(8), repeat=n_elems))


def _pred(kind, X, Y, model):
    xi, yi = X, Y
    inx = lambda e, i: bool(e >> i & 1)  # noqa: E731
    if kind == "A":
        return all((not inx(e, xi)) or inx(e, yi) for e in model)
    if kind == "E":
        return all(not (inx(e, xi) and inx(e, yi)) for e in model)
    if kind == "I":
        return any(inx(e, xi) and inx(e, yi) for e in model)
    return any(inx(e, xi) and not inx(e, yi) for e in model)


@family("logic.syllogism", "Syllogisms and valid arguments", {"logic": (4, 8)},
        "A categorical argument is valid when the conclusion must be true whenever the premises are. All A are B (A); No A are B (E); Some A are B (I); Some A are not B (O). "
        "Test validity by looking for a counter-example: a situation where the premises are true but the conclusion is false.",
        "Judging validity by whether the conclusion sounds true instead of whether it follows.")
def syllogism(rng, t):
    nouns = rng.sample(["dogs", "mammals", "pets", "artists", "poets", "teachers", "birds", "singers", "robots", "machines"], 3)
    S, M, P = nouns
    words = {"A": "All {} are {}", "E": "No {} are {}", "I": "Some {} are {}", "O": "Some {} are not {}"}
    while True:
        m1, m2, c = rng.choice("AEIO"), rng.choice("AEIO"), rng.choice("AEIO")
        fig = rng.randint(1, 4)
        # term indices: S = 0, M = 1, P = 2. Major premise first, then minor.
        pr1 = (1, 2) if fig in (1, 3) else (2, 1)
        pr2 = (0, 1) if fig in (1, 2) else (1, 0)
        valid = not any(_pred(m1, *pr1, model) and _pred(m2, *pr2, model) and not _pred(c, 0, 2, model) for model in _models(4))
        if valid == (rng.random() < .5):  # keep answers balanced
            break
    prem1 = words[m1].format(*(nouns[i] for i in pr1))
    prem2 = words[m2].format(*(nouns[i] for i in pr2))
    concl = words[c].format(S, P)
    return yesno(rng, f"Premises: '{prem1}.' and '{prem2}.' Conclusion: '{concl}.' Is the argument valid?", valid,
                 "valid" if valid else "a counter-example exists")


@family("logic.venn", "Venn diagrams and inclusion–exclusion", {"logic": (3, 7), "discrete_math": (3, 6)},
        "For three sets: |A∪B∪C| = |A| + |B| + |C| − |A∩B| − |A∩C| − |B∩C| + |A∩B∩C|. Fill the middle of a Venn diagram first, then work outwards.",
        "Subtracting overlaps but forgetting to add back the triple overlap.")
def venn(rng, t):
    a, b, c = rng.randint(15, 40), rng.randint(15, 40), rng.randint(15, 40)
    ab, ac, bc = rng.randint(4, 10), rng.randint(4, 10), rng.randint(4, 10)
    abc = rng.randint(1, 3)
    total = a + b + c - ab - ac - bc + abc
    return Q("numeric", f"In a survey, {a} like tea, {b} like coffee, {c} like juice; {ab} like tea and coffee, {ac} tea and juice, {bc} coffee and juice; {abc} like all three. How many like at least one?", total, "inclusion–exclusion")


@family("logic.statements", "Converse, inverse and contrapositive", {"logic": (3, 7)},
        "For 'If P then Q': converse = 'If Q then P'; inverse = 'If not P then not Q'; contrapositive = 'If not Q then not P'. Only the contrapositive is always equivalent to the original.",
        "Treating the converse as equivalent to the original.")
def statements(rng, t):
    pairs = [("it rains", "the ground is wet"), ("a number is divisible by 4", "it is even"), ("a shape is a square", "it has four sides"),
             ("you study", "you pass"), ("an animal is a dog", "it is a mammal"), ("the light is on", "the circuit is closed")]
    P, Qq = rng.choice(pairs)
    kind = rng.choice(["contrapositive", "converse", "inverse"])
    forms = {"contrapositive": f"If it is not true that {Qq}, then it is not true that {P}",
             "converse": f"If {Qq}, then {P}", "inverse": f"If it is not true that {P}, then it is not true that {Qq}"}
    wrong = [v for k, v in forms.items() if k != kind] + [f"If {P}, then it is not true that {Qq}"]
    return mcq(rng, f"The statement is: 'If {P}, then {Qq}.' Which option is its {kind}?", forms[kind], wrong, f"{kind} swaps and/or negates the parts")


# ---------------------------------------------------------------- puzzles

@family("logic.knights", "Knights and knaves", {"logic": (5, 9)},
        "Knights always tell the truth; knaves always lie. Try each possible assignment: a statement made by a knight must be true, and one made by a knave must be false. Keep only assignments where every statement fits.",
        "Stopping at the first assignment that works without checking whether others do.")
def knights(rng, t):
    names = ["A", "B", "C"][: 2 + (t > .4)]
    n = len(names)
    for _ in range(500):
        stmts = []
        for who in range(n):
            k = rng.choice(["is_knight", "is_knave", "same", "at_least_knave", "exactly_one_knight"])
            tgt = rng.choice([i for i in range(n) if i != who])
            stmts.append((who, k, tgt))

        def holds(assign, s):
            who, k, tgt = s
            if k == "is_knight":
                return assign[tgt]
            if k == "is_knave":
                return not assign[tgt]
            if k == "same":
                return assign[who] == assign[tgt]
            if k == "at_least_knave":
                return not all(assign)
            return sum(assign) == 1

        sols = [a for a in itertools.product([True, False], repeat=n) if all(holds(a, s) == a[s[0]] for s in stmts)]
        if len(sols) == 1:
            text = {"is_knight": "{t} is a knight", "is_knave": "{t} is a knave", "same": "{t} and I are the same type",
                    "at_least_knave": "at least one of us is a knave", "exactly_one_knight": "exactly one of us is a knight"}
            lines = "; ".join(f"{names[w]} says: \"{text[k].format(t=names[tg])}\"" for w, k, tg in stmts)
            who = rng.randrange(n)
            if rng.random() < .5:
                return Q("numeric", f"{n} islanders, each a knight (always truthful) or a knave (always lies). {lines}. How many are knights?", sum(sols[0]), "only one assignment is consistent")
            return mcq(rng, f"{n} islanders, each a knight or a knave. {lines}. What is {names[who]}?", "Knight" if sols[0][who] else "Knave",
                       ["Knave" if sols[0][who] else "Knight"], "only one assignment is consistent")
    return Q("numeric", "How many knights are there if A says 'B is a knave' and B says 'A is a knight'?", 1, "exactly one of them is a knight")


@family("logic.ordering", "Ordering and constraint puzzles", {"logic": (6, 10)},
        "List every arrangement that satisfies the first clue, then eliminate arrangements using each further clue. With enough clues exactly one arrangement survives.",
        "Misreading 'next to' (adjacent) as 'somewhere to the left of'.")
def ordering(rng, t):
    people = rng.sample(["Ana", "Ben", "Cy", "Dee", "Eli", "Fay"], 4)
    for _ in range(500):
        truth = rng.sample(people, 4)  # positions 1..4 left→right
        pos = {p: i for i, p in enumerate(truth)}
        clues = []
        for _ in range(rng.randint(4, 5)):
            a, b = rng.sample(people, 2)
            k = rng.choice(["left", "adjacent", "not_end", "not_adjacent"])
            if k == "left" and pos[a] < pos[b]:
                clues.append((k, a, b, f"{a} is somewhere to the left of {b}"))
            elif k == "adjacent" and abs(pos[a] - pos[b]) == 1:
                clues.append((k, a, b, f"{a} is next to {b}"))
            elif k == "not_end" and pos[a] not in (0, 3):
                clues.append((k, a, None, f"{a} is not at either end"))
            elif k == "not_adjacent" and abs(pos[a] - pos[b]) > 1:
                clues.append((k, a, b, f"{a} is not next to {b}"))

        def ok(perm):
            p = {x: i for i, x in enumerate(perm)}
            for k, a, b, _ in clues:
                if k == "left" and not p[a] < p[b]:
                    return False
                if k == "adjacent" and abs(p[a] - p[b]) != 1:
                    return False
                if k == "not_end" and p[a] in (0, 3):
                    return False
                if k == "not_adjacent" and abs(p[a] - p[b]) <= 1:
                    return False
            return True

        sols = [pm for pm in itertools.permutations(people) if ok(pm)]
        if len(sols) == 1 and len(clues) >= 3:
            slot = rng.randint(1, 4)
            return mcq(rng, f"Four people stand in a line (positions 1–4 from the left): {', '.join(people)}. Clues: " + "; ".join(c[3] for c in clues) + f". Who is in position {slot}?",
                       sols[0][slot - 1], [p for p in people if p != sols[0][slot - 1]], "only one arrangement fits all clues")
    return Q("numeric", "How many arrangements are there of 3 people in a line?", 6, "3!")


_POOL = ["p and q", "q and p", "not (not p or not q)", "p or q", "not (not p and not q)", "(not p) or q", "not (p and not q)", "p <= q", "(not q) <= (not p)",
         "p and (q or r)", "(p and q) or (p and r)", "p or (q and r)", "(p or q) and (p or r)", "not (p or q)", "(not p) and (not q)", "not (p and q)", "(not p) or (not q)",
         "p != q", "(p or q) and not (p and q)", "p == q", "(p and q) or (not p and not q)", "q <= p", "p and not q", "not p or q or r", "(p <= q) and (q <= p)"]


@family("logic.equivalence", "Logical equivalence", {"logic": (6, 10), "cs": (5, 7)},
        "Two statements are equivalent if they have the same truth table. Key laws: De Morgan ¬(p∧q) ≡ ¬p∨¬q, ¬(p∨q) ≡ ¬p∧¬q; implication p→q ≡ ¬p∨q ≡ ¬q→¬p; distribution p∧(q∨r) ≡ (p∧q)∨(p∧r).",
        "Assuming p→q is equivalent to q→p.")
def equivalence(rng, t):
    def table(e):
        return tuple(bool(eval(e, {}, dict(p=p, q=q, r=r))) for p, q, r in itertools.product([True, False], repeat=3))

    def show(e):
        return (e.replace("<=", "→").replace("!=", "⊕").replace("==", "↔").replace("and", "∧").replace("or", "∨").replace("not ", "¬"))

    groups = defaultdict(list)
    for e in _POOL:
        groups[table(e)].append(e)
    multi = [g for g in groups.values() if len(g) >= 2]
    g = rng.choice(multi)
    target, correct = rng.sample(g, 2)
    wrong_pool = [e for e in _POOL if table(e) != table(target)]
    wrong = rng.sample(wrong_pool, 3)
    return mcq(rng, f"Which statement is logically equivalent to {show(target)}?", show(correct), [show(w) for w in wrong], "compare truth tables")


@family("logic.game", "Game theory", {"logic": (9, 11), "economics": (9, 12), "microeconomics": (9, 12)},
        "In a two-player game a pure-strategy Nash equilibrium is a cell where neither player can gain by switching their own choice. Check each cell: is the row player's payoff the best in its column, and the column player's the best in its row?",
        "Looking for the best outcome for both instead of mutual best responses.")
def game(rng, t):
    A = [[(rng.randint(0, 5), rng.randint(0, 5)) for _ in range(2)] for _ in range(2)]
    ne = []
    for i in range(2):
        for j in range(2):
            if A[i][j][0] >= A[1 - i][j][0] and A[i][j][1] >= A[i][1 - j][1]:
                ne.append((i, j))
    rows = "; ".join(f"Row{i + 1}: " + ", ".join(f"({A[i][j][0]},{A[i][j][1]})" for j in range(2)) for i in range(2))
    return Q("numeric", f"A 2×2 game lists (row payoff, column payoff) for each column choice. {rows}. How many pure-strategy Nash equilibria does it have?", len(ne), f"equilibria at {ne}")


@family("logic.sat", "Satisfiability", {"logic": (10, 12), "cs": (9, 11)},
        "A CNF formula is an AND of clauses, each an OR of literals. It is satisfiable if some assignment makes every clause true. Count satisfying assignments by trying all 2ⁿ.",
        "Counting an assignment that satisfies only some clauses.")
def sat(rng, t):
    n = 4
    clauses = []
    for _ in range(rng.randint(4, 6)):
        vs = rng.sample(range(n), 3)
        clauses.append([(v, rng.random() < .5) for v in vs])
    cnt = sum(all(any(a[v] == s for v, s in c) for c in clauses) for a in itertools.product([True, False], repeat=n))
    text = " ∧ ".join("(" + " ∨ ".join(("" if s else "¬") + "xyzw"[v] for v, s in c) + ")" for c in clauses)
    return Q("numeric", f"How many of the 16 assignments to x, y, z, w satisfy {text}?", cnt, "enumerate all 16")


@family("logic.computation", "Computation puzzles", {"logic": (9, 12), "cs": (8, 12), "discrete_math": (9, 12)},
        "A cellular automaton updates every cell from its neighbours by a fixed rule. The Collatz process: if n is even halve it, otherwise replace it by 3n + 1; count steps until it reaches 1. Simulating step by step is the only reliable method.",
        "Skipping steps — simulate carefully.")
def computation(rng, t):
    if rng.random() < .5:
        n = rng.randint(6, 60)
        steps, x = 0, n
        while x != 1:
            x = x // 2 if x % 2 == 0 else 3 * x + 1
            steps += 1
        return Q("numeric", f"How many steps does the Collatz process take to reach 1 starting from {n}?", steps, "simulate")
    rule, steps = rng.choice([30, 90, 110]), rng.randint(2, 5)
    w = 11
    cells = [0] * w
    cells[w // 2] = 1
    for _ in range(steps):
        cells = [(rule >> (cells[i - 1] * 4 + cells[i] * 2 + cells[(i + 1) % w])) & 1 if 0 < i < w - 1 else 0 for i in range(w)]
    return Q("numeric", f"A row of 11 cells (edges stay 0) starts with only the middle cell on. Rule {rule} is applied {steps} times (new cell = bit [4·left + 2·self + right] of {rule}). How many cells are on?", sum(cells), "simulate each step")
