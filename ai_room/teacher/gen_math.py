"""Mathematics families: arithmetic → algebra → calculus → analysis-flavoured problems."""

from __future__ import annotations

import math
from fractions import Fraction

from .generators import Q, family, fmt, is_prime, mcq, poly_str, prime_factors, scale

SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"}


def sub(n: int) -> str:
    return "".join(SUB[c] for c in str(n))


def lin(a: int, b: int, vx="x", vy="y") -> str:
    out = f"{'' if abs(a) == 1 else abs(a)}{vx}"
    out = ("−" if a < 0 else "") + out
    if b:
        out += f" {'−' if b < 0 else '+'} {'' if abs(b) == 1 else abs(b)}{vy}"
    return out


def mat(rows: list[list]) -> str:
    return "[" + "; ".join(" ".join(str(v) for v in r) for r in rows) + "]"


MAT_NOTE = "(a matrix is written row by row; rows are separated by ';')"


# ---------------------------------------------------------------- arithmetic

@family("math.addsub", "Adding and subtracting whole numbers", {"math": (1, 3), "discrete_math": (1, 1)},
        "Line the numbers up by place value (ones, tens, hundreds) and work column by column, carrying "
        "(adding) or borrowing (subtracting) when a column overflows.\nExample: 47 + 38 → 7+8 = 15, write 5 carry 1; 4+3+1 = 8 → 85.",
        "Forgetting to carry or borrow; subtracting the bigger digit from the smaller digit in a column.")
def addsub(rng, t):
    hi = scale(t, 20, 9999)
    a, b = rng.randint(1, hi), rng.randint(1, hi)
    if rng.random() < .5:
        return Q("numeric", f"What is {a} + {b}?", a + b, f"{a} + {b} = {a + b}")
    a, b = max(a, b), min(a, b)
    return Q("numeric", f"What is {a} − {b}?", a - b, f"{a} − {b} = {a - b}")


@family("math.muldiv", "Multiplication and division", {"math": (2, 4)},
        "Multiplication is repeated addition: 6 × 4 = 6+6+6+6 = 24. Division asks how many equal groups fit: 24 ÷ 4 = 6. "
        "When a number does not divide evenly the leftover is the remainder: 25 = 4×6 + 1.",
        "Reading the remainder as the quotient; multiplying instead of dividing in word problems.")
def muldiv(rng, t):
    hi = scale(t, 9, 99)
    a, b = rng.randint(2, hi), rng.randint(2, 12 if t < .5 else 30)
    k = rng.random()
    if k < .5:
        return Q("numeric", f"What is {a} × {b}?", a * b, f"{a} × {b} = {a * b}")
    if k < .8:
        return Q("numeric", f"What is {a * b} ÷ {a}?", b, f"{a} × {b} = {a * b}, so {a * b} ÷ {a} = {b}")
    n = a * b + rng.randint(1, a - 1)
    return Q("numeric", f"What is the remainder when {n} is divided by {a}?", n % a,
             f"{n} = {a} × {n // a} + {n % a}")


@family("math.orderops", "Order of operations", {"math": (3, 5)},
        "Work in this order: (1) brackets, (2) powers, (3) multiplication and division left to right, (4) addition and subtraction "
        "left to right.\nExample: 2 + 3 × 4 = 2 + 12 = 14, not 20.",
        "Adding before multiplying; ignoring brackets.")
def orderops(rng, t):
    hi = scale(t, 9, 30)
    a, b, c, d = (rng.randint(2, hi) for _ in range(4))
    forms = [
        (f"{a} + {b} × {c}", a + b * c, f"multiply first: {b} × {c} = {b * c}; then {a} + {b * c}"),
        (f"({a} + {b}) × {c}", (a + b) * c, f"brackets first: {a} + {b} = {a + b}; then × {c}"),
        (f"{a} × {b} − {c} × {d}", a * b - c * d, f"{a * b} − {c * d}"),
        (f"{a} + {b} × ({c} − {d})", a + b * (c - d), f"brackets: {c} − {d} = {c - d}; × {b} = {b * (c - d)}; + {a}"),
    ]
    s, v, e = rng.choice(forms)
    return Q("numeric", f"Evaluate {s}.", v, f"{s}: {e} = {v}")


@family("math.fractions", "Fractions", {"math": (3, 5)},
        "To add or subtract fractions, use a common denominator: 1/2 + 1/3 = 3/6 + 2/6 = 5/6. To multiply, multiply tops and bottoms. "
        "To divide, flip the second fraction and multiply. Always reduce to lowest terms.",
        "Adding denominators; forgetting to simplify.")
def fractions_(rng, t):
    q1, q2 = rng.randint(2, scale(t, 6, 12)), rng.randint(2, scale(t, 6, 12))
    p1, p2 = rng.randint(1, q1 - 1), rng.randint(1, q2 - 1)
    a, b = Fraction(p1, q1), Fraction(p2, q2)
    op = rng.choice(["+", "−", "×", "÷"] if t > .4 else ["+", "×"])
    if op == "−" and a < b:
        p1, q1, p2, q2, a, b = p2, q2, p1, q1, b, a
    val = {"+": a + b, "−": a - b, "×": a * b, "÷": a / b}[op]
    return Q("numeric", f"Compute {p1}/{q1} {op} {p2}/{q2}. Give the answer as a fraction in lowest terms (like 5/6) or a whole number.",
             val, f"{p1}/{q1} {op} {p2}/{q2} = {val}")


@family("math.percent", "Percentages", {"math": (3, 6)},
        "Percent means 'out of 100'. x% of y = y × x/100. Percent change = (new − old) / old × 100.\nExample: 20% of 150 = 150 × 0.20 = 30.",
        "Dividing by the new value instead of the old one when finding percent change.")
def percent(rng, t):
    if rng.random() < .6:
        p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
        base = 20 * rng.randint(1, scale(t, 10, 60))
        return Q("numeric", f"What is {p}% of {base}?", fmt(p * base / 100), f"{base} × {p}/100 = {fmt(p * base / 100)}")
    old = 20 * rng.randint(2, scale(t, 10, 50))
    pct = rng.choice([10, 20, 25, 50, 5, 15])
    up = rng.random() < .5
    new = old * (100 + pct if up else 100 - pct) // 100
    return Q("numeric", f"A price goes from {old} to {new}. What is the percent {'increase' if up else 'decrease'}? (number only, no % sign)",
             pct, f"|{new} − {old}| / {old} × 100 = {pct}")


@family("math.ratio", "Ratios and proportions", {"math": (4, 6)},
        "A ratio compares two quantities: a : b. Equivalent ratios are made by multiplying both parts by the same number. "
        "If a : b = c : x then x = b × c / a.",
        "Adding the same number to both parts instead of multiplying.")
def ratio(rng, t):
    a, b = rng.randint(1, 9), rng.randint(2, 12)
    k = rng.randint(2, scale(t, 6, 20))
    return Q("numeric", f"The ratio of apples to oranges is {a} : {b}. If there are {a * k} apples, how many oranges are there?", b * k,
             f"Scale factor {k}: {b} × {k} = {b * k}")


@family("math.negatives", "Negative numbers", {"math": (4, 5)},
        "On the number line, adding a negative moves left; subtracting a negative moves right: 5 − (−3) = 8. "
        "A negative times a negative is positive; a negative times a positive is negative.",
        "Treating 'minus a negative' as 'minus'.")
def negatives(rng, t):
    a, b, c = (rng.choice([n for n in range(-20, 21) if n]) for _ in range(3))
    forms = [(f"{a} − ({b}) + ({c})", a - b + c), (f"({a}) × ({b})", a * b), (f"{a} + ({b}) − ({c})", a + b - c)]
    s, v = rng.choice(forms)
    return Q("numeric", f"Evaluate {s}.", v, f"{s} = {v}")


@family("math.exponents", "Exponents and square roots", {"math": (4, 6), "number_theory": (1, 2)},
        "a^n means a multiplied by itself n times. Rules: a^m × a^n = a^(m+n), a^m ÷ a^n = a^(m−n), (a^m)^n = a^(mn). "
        "√n is the number whose square is n.",
        "Multiplying the base by the exponent (3^4 ≠ 12).")
def exponents(rng, t):
    k = rng.random()
    if k < .4:
        b, e = rng.randint(2, scale(t, 5, 12)), rng.randint(2, scale(t, 3, 5))
        return Q("numeric", f"What is {b}^{e}?", b ** e, f"{b}^{e} = {b ** e}")
    if k < .7:
        r = rng.randint(2, scale(t, 12, 30))
        return Q("numeric", f"What is the square root of {r * r}?", r, f"{r} × {r} = {r * r}")
    b, m, n = rng.randint(2, 5), rng.randint(5, 9), rng.randint(2, 4)
    return Q("numeric", f"What is {b}^{m} ÷ {b}^{n}?", b ** (m - n), f"{b}^({m}−{n}) = {b}^{m - n} = {b ** (m - n)}")


@family("math.primes", "Primes and factorisation", {"math": (4, 6), "number_theory": (1, 5)},
        "A prime has exactly two factors: 1 and itself (2, 3, 5, 7, 11, …). Every integer > 1 is a unique product of primes: 60 = 2 × 2 × 3 × 5.",
        "Calling 1 a prime; forgetting that 2 is prime.")
def primes(rng, t):
    if rng.random() < .4:
        lo = rng.randint(1, 50)
        hi = lo + scale(t, 15, 40)
        cnt = sum(is_prime(n) for n in range(lo, hi + 1))
        return Q("numeric", f"How many prime numbers are there from {lo} to {hi} (inclusive)?", cnt,
                 "Primes: " + ", ".join(str(n) for n in range(lo, hi + 1) if is_prime(n)))
    ps = [rng.choice([2, 3, 5, 7, 11, 13, 17, 19][:4 + int(t * 4)]) for _ in range(rng.randint(2, 3 + int(t * 2)))]
    n = math.prod(ps)
    return Q("numeric", f"What is the largest prime factor of {n}?", max(prime_factors(n)),
             f"{n} = " + " × ".join(map(str, prime_factors(n))))


@family("math.gcdlcm", "GCD and LCM", {"math": (4, 6), "number_theory": (2, 4)},
        "GCD (greatest common divisor) is the largest number dividing both; LCM (least common multiple) is the smallest number both divide. "
        "Euclid: gcd(a, b) = gcd(b, a mod b). Also gcd × lcm = a × b.",
        "Mixing up GCD and LCM.")
def gcdlcm(rng, t):
    g = rng.randint(2, scale(t, 6, 15))
    x, y = rng.sample(range(2, 10), 2)
    a, b = g * x, g * y
    if rng.random() < .5:
        return Q("numeric", f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b), f"gcd({a}, {b}) = {math.gcd(a, b)}")
    return Q("numeric", f"What is the least common multiple of {a} and {b}?", math.lcm(a, b),
             f"lcm = {a} × {b} ÷ gcd = {a * b} ÷ {math.gcd(a, b)} = {math.lcm(a, b)}")


# ---------------------------------------------------------------- algebra

@family("math.lineq", "Linear equations", {"math": (5, 7)},
        "Keep the equation balanced: do the same thing to both sides until x is alone.\nExample: 3x + 5 = 20 → 3x = 15 → x = 5.",
        "Changing only one side; sign errors when moving terms.")
def lineq(rng, t):
    x = rng.choice([n for n in range(-10, 21) if n])
    if rng.random() < .6 or t < .4:
        a, b = rng.randint(2, 9), rng.randint(-15, 15)
        c = a * x + b
        return Q("numeric", f"Solve for x: {lin(a, 0)}{' + ' if b >= 0 else ' − '}{abs(b)} = {c}", x,
                 f"{lin(a, 0)} = {c - b} → x = {x}")
    a, c = rng.randint(3, 9), rng.randint(1, 2)
    b = rng.choice([n for n in range(-10, 11) if n])
    d = (a - c) * x + b
    cx = "x" if c == 1 else f"{c}x"
    lhs = f"{a}x {'+' if b > 0 else '−'} {abs(b)}"
    rhs = f"{cx} {'+' if d > 0 else '−'} {abs(d)}" if d else cx
    return Q("numeric", f"Solve for x: {lhs} = {rhs}", x, f"{a - c}x = {d - b} → x = {x}")


@family("math.pythag", "The Pythagorean theorem", {"math": (6, 8), "geometry": (3, 6)},
        "In a right triangle with legs a, b and hypotenuse c: a² + b² = c². Common triples: 3-4-5, 5-12-13, 8-15-17, 7-24-25.",
        "Adding the legs instead of their squares; using the formula on a non-right triangle.")
def pythag(rng, t):
    a, b, c = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)])
    k = rng.randint(1, 4)
    a, b, c = a * k, b * k, c * k
    if t > .6 and rng.random() < .5:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        h = math.hypot(a, b)
        return Q("numeric", f"A right triangle has legs {a} and {b}. How long is the hypotenuse? (2 decimal places)", f"{h:.2f}",
                 f"√({a}² + {b}²) = √{a * a + b * b} ≈ {h:.2f}")
    if rng.random() < .5:
        return Q("numeric", f"A right triangle has legs {a} and {b}. How long is the hypotenuse?", c, f"√({a}² + {b}²) = {c}")
    return Q("numeric", f"A right triangle has hypotenuse {c} and one leg {a}. How long is the other leg?", b, f"√({c}² − {a}²) = {b}")


@family("math.quadratic", "Quadratic equations", {"math": (6, 9)},
        "A quadratic ax² + bx + c = 0 factors as a(x − p)(x − q) with roots p, q. Sum of roots = −b/a, product = c/a. "
        "The discriminant b² − 4ac tells how many real roots there are; the quadratic formula is x = (−b ± √(b² − 4ac)) / 2a.",
        "Sign errors when reading roots off (x − p) factors.")
def quadratic(rng, t):
    p, q = rng.sample([n for n in range(-9, 10) if n], 2)
    a = 1 if t < .5 else rng.choice([1, 2, 3])
    b, c = -a * (p + q), a * p * q
    eq = poly_str([a, b, c]) + " = 0"
    k = rng.random()
    if k < .4:
        return Q("numeric", f"What is the larger root of {eq}?", max(p, q), f"roots are {p} and {q}")
    if k < .6:
        return Q("numeric", f"What is the sum of the roots of {eq}?", p + q, f"−b/a = {p + q}")
    if k < .8:
        return Q("numeric", f"What is the product of the roots of {eq}?", p * q, f"c/a = {p * q}")
    return Q("numeric", f"What is the discriminant b² − 4ac of {eq}?", b * b - 4 * a * c, f"{b}² − 4·{a}·{c} = {b * b - 4 * a * c}")


@family("math.systems", "Systems of linear equations", {"math": (6, 8), "linear_algebra": (1, 2)},
        "Two equations, two unknowns. Elimination: scale the equations so one variable cancels when you add or subtract them, solve, then substitute back.",
        "Solving for one variable and forgetting to substitute back.")
def systems(rng, t):
    while True:
        x, y = rng.randint(-6, 9), rng.randint(-6, 9)
        a1, b1, a2, b2 = (rng.choice([n for n in range(-5, 6) if n]) for _ in range(4))
        if a1 * b2 - a2 * b1 != 0:
            break
    c1, c2 = a1 * x + b1 * y, a2 * x + b2 * y
    ask = rng.choice(["x", "y", "x + y"])
    ans = {"x": x, "y": y, "x + y": x + y}[ask]
    return Q("numeric", f"Solve the system: {lin(a1, b1)} = {c1} and {lin(a2, b2)} = {c2}. What is {ask}?", ans,
             f"The solution is x = {x}, y = {y}")


@family("math.sequences", "Sequences and series", {"math": (6, 9), "discrete_math": (4, 7)},
        "Arithmetic: aₙ = a₁ + (n−1)d, sum of n terms = n(2a₁ + (n−1)d)/2. Geometric: aₙ = a₁rⁿ⁻¹, sum = a₁(rⁿ − 1)/(r − 1).",
        "Using n instead of n−1 in the nth-term formula.")
def sequences(rng, t):
    kind = rng.choice(["arith", "sum", "geo"] if t > .3 else ["arith"])
    n = rng.randint(5, scale(t, 10, 25))
    if kind == "geo":
        a1, r, n = rng.randint(1, 5), rng.choice([2, 3, -2]), rng.randint(4, 7)
        seq = ", ".join(str(a1 * r ** i) for i in range(4))
        return Q("numeric", f"In the geometric sequence {seq}, … what is term number {n}?", a1 * r ** (n - 1), f"a₁·r^(n−1) = {a1}·{r}^{n - 1}")
    a1, d = rng.randint(-5, 20), rng.randint(2, 9)
    seq = ", ".join(str(a1 + i * d) for i in range(4))
    if kind == "arith":
        return Q("numeric", f"In the arithmetic sequence {seq}, … what is term number {n}?", a1 + (n - 1) * d, f"a₁ + (n−1)d = {a1} + {n - 1}·{d}")
    return Q("numeric", f"What is the sum of the first {n} terms of {seq}, …?", n * (2 * a1 + (n - 1) * d) // 2, f"n(2a₁ + (n−1)d)/2")


@family("math.logs", "Logarithms", {"math": (7, 9)},
        "log_b(x) is the exponent you raise b to in order to get x: log₂ 8 = 3 because 2³ = 8. log(xy) = log x + log y; log(x/y) = log x − log y; log(xⁿ) = n log x.",
        "Confusing log_b(x) with b^x.")
def logs(rng, t):
    b = rng.choice([2, 3, 5, 10])
    e1, e2 = rng.randint(1, 5), rng.randint(1, 4)
    if rng.random() < .5:
        return Q("numeric", f"What is log{sub(b)}({b ** e1})?", e1, f"{b}^{e1} = {b ** e1}")
    return Q("numeric", f"What is log{sub(b)}({b ** e1}) + log{sub(b)}({b ** e2})?", e1 + e2, f"{e1} + {e2} = {e1 + e2} (or log of the product {b ** (e1 + e2)})")


@family("math.trig", "Trigonometry", {"math": (7, 9), "geometry": (6, 9)},
        "SOH-CAH-TOA: sin = opposite/hypotenuse, cos = adjacent/hypotenuse, tan = opposite/adjacent. Special angles: sin 30° = 1/2, cos 60° = 1/2, "
        "sin 45° = cos 45° = √2/2 ≈ 0.707, sin 60° = cos 30° = √3/2 ≈ 0.866, tan 45° = 1. π radians = 180°.",
        "Calculator in radian mode when the angle is in degrees.")
def trig(rng, t):
    k = rng.random()
    if k < .4:
        fn, ang = rng.choice(["sin", "cos", "tan"]), rng.choice([0, 30, 45, 60, 90])
        if fn == "tan" and ang == 90:
            ang = 45
        v = getattr(math, {"sin": "sin", "cos": "cos", "tan": "tan"}[fn])(math.radians(ang))
        return Q("numeric", f"What is {fn} {ang}°? (3 decimal places)", f"{v:.3f}", f"{fn} {ang}° = {v:.3f}")
    if k < .7:
        deg = rng.choice([30, 45, 60, 90, 120, 135, 150, 180, 270, 360])
        return Q("numeric", f"Convert {deg}° to radians. Give k where the angle is kπ radians (as a fraction like 2/3 or a whole number).",
                 Fraction(deg, 180), f"{deg}/180 = {Fraction(deg, 180)}")
    h, ang = rng.randint(4, 40), rng.choice([30, 45, 60])
    v = h * math.sin(math.radians(ang))
    return Q("numeric", f"A right triangle has hypotenuse {h} and an angle of {ang}°. How long is the side opposite that angle? (2 decimal places)",
             f"{v:.2f}", f"{h} × sin {ang}° = {v:.2f}")


@family("math.combinatorics", "Counting and combinations", {"math": (7, 10), "discrete_math": (3, 7)},
        "Permutations (order matters): P(n,r) = n!/(n−r)!. Combinations (order doesn't): C(n,r) = n!/(r!(n−r)!). "
        "Arrangements of a word with repeated letters: n! divided by the factorial of each repeat count.",
        "Using permutations when order does not matter.")
def combinatorics(rng, t):
    k = rng.random()
    n = rng.randint(5, scale(t, 8, 14))
    r = rng.randint(2, min(5, n - 1))
    if k < .4:
        return Q("numeric", f"In how many ways can a committee of {r} be chosen from {n} people?", math.comb(n, r), f"C({n},{r}) = {math.comb(n, r)}")
    if k < .65:
        return Q("numeric", f"In how many ways can {r} runners finish 1st, 2nd, … from a field of {n}?", math.perm(n, r), f"P({n},{r}) = {math.perm(n, r)}")
    w = rng.choice(["LEVEL", "BANANA", "APPLE", "MISSISSIPPI", "COMMITTEE", "STATISTICS", "BOOKKEEPER"])
    cnt = math.factorial(len(w))
    for ch in set(w):
        cnt //= math.factorial(w.count(ch))
    return Q("numeric", f"How many distinct arrangements of the letters of {w} are there?", cnt, f"{len(w)}! divided by repeat factorials = {cnt}")


@family("math.derivative", "Derivatives of polynomials", {"math": (8, 10), "calculus": (2, 6)},
        "The derivative gives the slope of a curve. Power rule: d/dx xⁿ = n·xⁿ⁻¹; constants differentiate to 0; derivative of a sum is the sum of derivatives.\nExample: f = 3x² + 5x → f′ = 6x + 5.",
        "Forgetting to lower the power; differentiating constants as if they were variables.")
def derivative(rng, t):
    n = rng.randint(2, scale(t, 3, 5))
    a, b, c = rng.randint(1, 6), rng.randint(-9, 9), rng.randint(-9, 9)
    co = [0] * (n + 1)
    co[0], co[n - 1], co[n] = a, co[n - 1] + b, c
    x0 = rng.randint(-3, 3)
    dco = [c_ * (n - i) for i, c_ in enumerate(co[:-1])]
    val = sum(c_ * x0 ** (len(dco) - 1 - i) for i, c_ in enumerate(dco))
    return Q("numeric", f"If f(x) = {poly_str(co)}, what is f′({x0})?", val, f"f′(x) = {poly_str(dco)}; f′({x0}) = {val}")


@family("calc.chain", "Chain and product rules", {"calculus": (4, 7), "math": (9, 10)},
        "Chain rule: d/dx [f(g(x))] = f′(g(x))·g′(x). Product rule: (uv)′ = u′v + uv′.\nExample: (2x+1)³ → 3(2x+1)²·2.",
        "Forgetting the inner derivative.")
def chain(rng, t):
    a, b, n, x0 = rng.randint(2, 5), rng.randint(-4, 4), rng.randint(2, 4), rng.randint(-2, 2)
    if rng.random() < .5:
        v = n * a * (a * x0 + b) ** (n - 1)
        return Q("numeric", f"If f(x) = ({a}x {'+' if b >= 0 else '−'} {abs(b)})^{n}, what is f′({x0})?", v, f"f′(x) = {n}·{a}·({a}x {'+' if b >= 0 else '−'} {abs(b)})^{n - 1}")
    m = rng.randint(2, 3)
    v = m * x0 ** (m - 1) * (a * x0 + b) + a * x0 ** m
    return Q("numeric", f"If f(x) = x^{m}·({a}x {'+' if b >= 0 else '−'} {abs(b)}), what is f′({x0})?", v, f"product rule: {m}x^{m - 1}·({a}x {'+' if b >= 0 else '−'} {abs(b)}) + {a}x^{m}")


@family("calc.optim", "Maxima and minima", {"calculus": (5, 8), "optimization": (1, 4)},
        "At a maximum or minimum a smooth function has f′(x) = 0. For ax² + bx + c the turning point is x = −b/2a; it is a maximum if a < 0, a minimum if a > 0.",
        "Reporting x when the question asks for the maximum value.")
def optim(rng, t):
    a, h, k = rng.randint(1, 5), rng.randint(-5, 8), rng.randint(-20, 30)
    sgn = rng.choice([-1, 1])
    co = [sgn * a, -2 * sgn * a * h, sgn * a * h * h + k]
    which = "maximum" if sgn < 0 else "minimum"
    if rng.random() < .5:
        return Q("numeric", f"For f(x) = {poly_str(co)}, at what x does the {which} occur?", h, f"x = −b/2a = {h}")
    return Q("numeric", f"What is the {which} value of f(x) = {poly_str(co)}?", k, f"vertex at x = {h}, f({h}) = {k}")


@family("math.integral", "Definite integrals of polynomials", {"math": (9, 11), "calculus": (6, 9)},
        "Power rule for integrals: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C. The definite integral ∫ₐᵇ f dx = F(b) − F(a), where F is an antiderivative.\nExample: ∫₀² x² dx = 8/3.",
        "Forgetting to raise the power before dividing; mixing up the order F(b) − F(a).")
def integral(rng, t):
    a, b, c = rng.randint(1, 4), rng.randint(-4, 4), rng.randint(-4, 4)
    k = rng.randint(1, 4)
    v = Fraction(a * k ** 3, 3) + Fraction(b * k ** 2, 2) + c * k
    return Q("numeric", f"Evaluate ∫ from 0 to {k} of ({poly_str([a, b, c])}) dx. (exact fraction like 7/3, or a whole number)", v,
             f"antiderivative F(x) = (a/3)x³ + (b/2)x² + cx with a = {a}, b = {b}, c = {c}; F({k}) − F(0) = {v}")


@family("math.limits", "Limits", {"math": (8, 10), "calculus": (1, 4)},
        "A limit asks what value f(x) approaches. If direct substitution gives 0/0, factor and cancel: (x²−a²)/(x−a) = x + a → 2a. "
        "As x → ∞ a ratio of polynomials tends to the ratio of leading coefficients when the degrees match.",
        "Plugging in the limit point when it gives 0/0 and stopping there.")
def limits(rng, t):
    k = rng.random()
    a = rng.randint(1, 9)
    if k < .4:
        return Q("numeric", f"Find the limit as x → {a} of (x² − {a * a}) / (x − {a}).", 2 * a, f"= x + {a} → {2 * a}")
    if k < .75:
        p, s = rng.randint(1, 9), rng.randint(1, 9)
        return Q("numeric", f"Find the limit as x → ∞ of ({p}x² + {rng.randint(1, 9)}x + 1) / ({s}x² + {rng.randint(1, 9)}). (fraction or whole number)", Fraction(p, s),
                 f"leading coefficients: {p}/{s}")
    c0, c1 = rng.randint(1, 5), rng.randint(1, 5)
    return Q("numeric", f"Find the limit as x → {a} of {c0}x² + {c1}x.", c0 * a * a + c1 * a, "substitute directly")


@family("math.matrices", "Matrices and determinants", {"math": (9, 11), "linear_algebra": (2, 5)},
        "Determinant of [a b; c d] is ad − bc. For 3×3, expand along a row. Matrix product entry (i,j) = row i of A · column j of B. Trace = sum of diagonal entries.",
        "Multiplying matrices entry-by-entry; wrong sign in the cofactor expansion.")
def matrices(rng, t):
    k = rng.random()
    A = [[rng.randint(-5, 9) for _ in range(2)] for _ in range(2)]
    if k < .35:
        d = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        return Q("numeric", f"What is the determinant of {mat(A)}? {MAT_NOTE}", d, f"ad − bc = {d}")
    if k < .6 or t < .3:
        B = [[rng.randint(-4, 6) for _ in range(2)] for _ in range(2)]
        i, j = rng.randint(0, 1), rng.randint(0, 1)
        v = A[i][0] * B[0][j] + A[i][1] * B[1][j]
        return Q("numeric", f"For A = {mat(A)} and B = {mat(B)}, what is entry (row {i + 1}, column {j + 1}) of AB? {MAT_NOTE}", v,
                 f"row {i + 1} of A · column {j + 1} of B = {v}")
    M = [[rng.randint(-3, 5) for _ in range(3)] for _ in range(3)]
    d = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    return Q("numeric", f"What is the determinant of the 3×3 matrix {mat(M)}? {MAT_NOTE}", d, "expand along the first row")


@family("math.vectors", "Vectors", {"math": (9, 11), "linear_algebra": (1, 3)},
        "Add vectors component by component. Dot product u·v = u₁v₁ + u₂v₂ + u₃v₃; it is 0 exactly when u ⊥ v. Length |v| = √(v·v).",
        "Taking |u||v| for the dot product.")
def vectors(rng, t):
    n = 2 if t < .4 else 3
    u = [rng.randint(-6, 8) for _ in range(n)]
    v = [rng.randint(-6, 8) for _ in range(n)]
    if rng.random() < .6:
        d = sum(a * b for a, b in zip(u, v))
        return Q("numeric", f"What is the dot product of ({', '.join(map(str, u))}) and ({', '.join(map(str, v))})?", d, f"sum of products = {d}")
    w, k = rng.choice([(1, 2, 2), (2, 3, 6), (1, 4, 8), (2, 6, 9), (4, 4, 7)]), rng.randint(1, 3)
    vec = [c * k for c in w]
    return Q("numeric", f"What is the length of the vector ({', '.join(map(str, vec))})?", math.isqrt(sum(c * c for c in vec)),
             f"√({' + '.join(f'{c}²' for c in vec)})")


@family("math.eigen", "Eigenvalues", {"math": (10, 12), "linear_algebra": (5, 12)},
        "Av = λv. The eigenvalues are the roots of det(A − λI) = 0. For a 2×2: λ² − (trace)λ + det = 0, so the eigenvalues sum to the trace and multiply to the determinant.",
        "Forgetting that eigenvalues add up to the trace — a handy check.")
def eigen(rng, t):
    l1, l2 = rng.sample(range(-4, 8), 2)
    p = rng.randint(-3, 5)
    s = l1 + l2 - p
    r = p * s - l1 * l2
    A = [[p, 1], [r, s]]
    big = rng.random() < .5
    return Q("numeric", f"What is the {'larger' if big else 'smaller'} eigenvalue of {mat(A)}? {MAT_NOTE}", max(l1, l2) if big else min(l1, l2),
             f"trace = {p + s} and det = {p * s - r}, so the characteristic equation is λ² − (trace)λ + det = 0 → λ = {l1} and {l2}")


@family("math.complex", "Complex numbers", {"math": (11, 12)},
        "i² = −1. (a+bi)(c+di) = (ac − bd) + (ad + bc)i. The modulus is |a+bi| = √(a²+b²) and (a+bi)(a−bi) = a² + b².",
        "Forgetting that i² = −1 when multiplying.")
def complex_(rng, t):
    a, b, c, d = (rng.choice([n for n in range(-5, 7) if n]) for _ in range(4))
    z1, z2 = f"{a} {'+' if b > 0 else '−'} {abs(b)}i", f"{c} {'+' if d > 0 else '−'} {abs(d)}i"
    k = rng.random()
    if k < .4:
        return Q("numeric", f"What is the real part of ({z1})({z2})?", a * c - b * d, f"ac − bd = {a * c - b * d}")
    if k < .8:
        return Q("numeric", f"What is the imaginary part of ({z1})({z2})?", a * d + b * c, f"ad + bc = {a * d + b * c}")
    return Q("numeric", f"What is |{z1}|²?", a * a + b * b, f"a² + b² = {a * a + b * b}")


@family("math.series", "Infinite series", {"math": (10, 12), "calculus": (9, 10)},
        "A geometric series a + ar + ar² + … converges when |r| < 1, to a/(1 − r). Taylor: eˣ = Σ xⁿ/n!, so the coefficient of xⁿ in e^(kx) is kⁿ/n!.",
        "Applying the formula when |r| ≥ 1.")
def series(rng, t):
    a, k = rng.randint(1, 9), rng.randint(2, 5)
    if rng.random() < .6:
        terms = " + ".join(str(Fraction(a, k ** i)) for i in range(4))
        return Q("numeric", f"What is the sum of the infinite series {terms} + …? (fraction or whole number)", Fraction(a * k, k - 1),
                 f"a/(1−r) with a = {a}, r = 1/{k}")
    c, n = rng.randint(2, 4), rng.randint(2, 5)
    return Q("numeric", f"What is the coefficient of x^{n} in the Taylor series of e^({c}x)? (fraction or whole number)", Fraction(c ** n, math.factorial(n)),
             f"{c}^{n}/{n}!")


@family("math.modular", "Modular arithmetic", {"math": (10, 12), "number_theory": (4, 7), "discrete_math": (6, 9)},
        "a mod m is the remainder after dividing a by m. Congruent numbers share a remainder. You can reduce at every step: (ab) mod m = ((a mod m)(b mod m)) mod m. "
        "The inverse of a mod m exists exactly when gcd(a, m) = 1.",
        "Reducing the exponent as if it were also mod m (use Fermat/Euler instead).")
def modular(rng, t):
    m = rng.choice([7, 11, 13, 17, 19, 23, 29])
    a = rng.randint(2, 25)
    if rng.random() < .5:
        b = rng.randint(5, 40)
        return Q("numeric", f"What is {a}^{b} mod {m}?", pow(a, b, m), f"computed by repeated squaring = {pow(a, b, m)}")
    while math.gcd(a, m) != 1:
        a += 1
    return Q("numeric", f"What is the multiplicative inverse of {a} modulo {m}? (the number x in 1..{m - 1} with {a}x ≡ 1 mod {m})", pow(a, -1, m),
             f"{a} × {pow(a, -1, m)} = {a * pow(a, -1, m)} ≡ 1 (mod {m})")


@family("math.ode", "Differential equations", {"math": (11, 12), "calculus": (10, 12)},
        "To solve y′ = f(x) with y(0) = y₀, integrate: y = ∫f dx + C and use the initial condition to fix C. For y′ = ky the solution is y = y₀e^(kx).",
        "Forgetting the constant of integration.")
def ode(rng, t):
    if rng.random() < .6:
        a, b, y0, x1 = rng.randint(1, 4) * 2, rng.randint(-5, 6), rng.randint(-5, 5), rng.randint(1, 4)
        return Q("numeric", f"Solve dy/dx = {a}x {'+' if b >= 0 else '−'} {abs(b)} with y(0) = {y0}. What is y({x1})?", a * x1 * x1 // 2 + b * x1 + y0,
                 f"y = {a // 2}x² + {b}x + {y0}")
    k, y0 = rng.choice([1, 2, 3, 4]), rng.randint(1, 6)
    return Q("numeric", f"Solve dy/dx = {k}y with y(0) = {y0}. What is y(1/{k})? (3 significant figures)", fmt(y0 * math.e), f"y = {y0}e^({k}x) → {y0}e")
