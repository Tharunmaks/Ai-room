"""Calculators and science helpers: arithmetic, algebra, statistics, units, constants, finance, physics, chemistry."""

from __future__ import annotations

import ast
import cmath
import itertools
import math
import operator
import re
import statistics as st
from fractions import Fraction

from .registry import ToolError, reg

# ------------------------------------------------------------------ safe expression evaluator

_FUNCS = {n: getattr(math, n) for n in (
    "sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh sqrt exp log log2 log10 floor ceil fabs factorial gcd lcm comb perm "
    "degrees radians hypot isqrt trunc erf erfc gamma lgamma copysign fmod pow expm1 log1p cbrt".split()) if hasattr(math, n)}
_FUNCS |= {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "int": int, "float": float, "ln": math.log, "sign": lambda x: (x > 0) - (x < 0)}
_CONST = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "phi": (1 + 5 ** .5) / 2}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate(expr: str, **names) -> float:
    """Evaluate an arithmetic expression without exec/eval — only numbers, operators and math functions."""
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr.strip()).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"cannot parse '{expr}': {e.msg}")

    def go(n):
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            if n.id in _CONST:
                return _CONST[n.id]
            raise ToolError(f"unknown name '{n.id}'")
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = go(n.left), go(n.right)
            if isinstance(n.op, ast.Pow) and isinstance(b, (int, float)) and abs(b) > 10000:
                raise ToolError("exponent too large")
            if isinstance(n.op, ast.Pow) and isinstance(a, int) and isinstance(b, int) and abs(a) > 1 and abs(b) * math.log2(abs(a)) > 100000:
                raise ToolError("result would be astronomically large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](go(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[go(a) for a in n.args])
        if isinstance(n, (ast.Tuple, ast.List)):
            return [go(x) for x in n.elts]
        raise ToolError("only numbers, + - * / // % **, parentheses and math functions (sin, sqrt, log, …) are allowed")

    try:
        return go(tree)
    except (ZeroDivisionError, OverflowError, ValueError) as e:
        raise ToolError(f"math error: {e}")


def fmt(x):
    if isinstance(x, float):
        return int(x) if x.is_integer() and abs(x) < 1e15 else round(x, 12)
    return x


def num_list(v) -> list[float]:
    if isinstance(v, str):
        v = [x for x in v.replace(";", ",").replace("\n", ",").replace(" ", ",").split(",") if x != ""]
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        raise ToolError("expected a list of numbers like [1, 2, 3]")


C = "math"
reg("calculator", "Evaluate an arithmetic expression like (3+4)*2^3/7 or sqrt(2)*sin(pi/4). Supports + - * / // % ** and math functions.", C, "expression:str",
    lambda expression: fmt(evaluate(expression)), tags=("google calculator", "compute", "arithmetic"))
reg("percent_of", "What is X percent of Y?", C, "percent:float, of:float", lambda percent, of: fmt(percent * of / 100))
reg("percent_change", "Percentage change from an old value to a new value.", C, "old:float, new:float", lambda old, new: fmt((new - old) / old * 100))
reg("what_percent", "What percentage is part of whole?", C, "part:float, whole:float", lambda part, whole: fmt(part / whole * 100))
reg("add_percent", "Increase a value by a percentage (tax, markup).", C, "value:float, percent:float", lambda value, percent: fmt(value * (1 + percent / 100)))
reg("subtract_percent", "Decrease a value by a percentage (discount).", C, "value:float, percent:float", lambda value, percent: fmt(value * (1 - percent / 100)))
reg("round_number", "Round a number to some decimal places.", C, "value:float, places:int=0", lambda value, places: round(value, places))
reg("power", "Raise a base to an exponent.", C, "base:float, exponent:float", lambda base, exponent: fmt(base ** exponent))
reg("square_root", "Square root of a number (complex for negatives).", C, "value:float", lambda value: fmt(math.sqrt(value)) if value >= 0 else str(cmath.sqrt(value)))
reg("cube_root", "Cube root of a number.", C, "value:float", lambda value: fmt(math.copysign(abs(value) ** (1 / 3), value)))
reg("nth_root", "The n-th root of a number.", C, "value:float, n:int", lambda value, n: fmt(value ** (1 / n)))
reg("logarithm", "Logarithm of a value in any base (default natural).", C, "value:float, base:float=2.718281828459045", lambda value, base: fmt(math.log(value, base)))
reg("exponential", "e raised to a power.", C, "x:float", lambda x: fmt(math.exp(x)))
reg("absolute_value", "Absolute value.", C, "x:float", lambda x: abs(x))
reg("factorial", "n factorial.", C, "n:int", lambda n: math.factorial(n) if 0 <= n <= 5000 else (_ for _ in ()).throw(ToolError("n must be 0..5000")))
reg("gcd", "Greatest common divisor of integers.", C, "numbers:list", lambda numbers: math.gcd(*[int(x) for x in numbers]))
reg("lcm", "Least common multiple of integers.", C, "numbers:list", lambda numbers: math.lcm(*[int(x) for x in numbers]))
reg("combinations", "nCr — ways to choose r of n.", C, "n:int, r:int", lambda n, r: math.comb(n, r))
reg("permutations", "nPr — ordered arrangements of r from n.", C, "n:int, r:int", lambda n, r: math.perm(n, r))
reg("fibonacci", "The n-th Fibonacci number (0-indexed).", C, "n:int", lambda n: (lambda a: a)(_fib(n)))
reg("is_prime", "Is the integer prime?", C, "n:int", lambda n: _isprime(n))
reg("next_prime", "Smallest prime greater than n.", C, "n:int", lambda n: next(k for k in itertools.count(max(2, n + 1)) if _isprime(k)))
reg("primes_up_to", "All primes up to n (n ≤ 1,000,000).", C, "n:int", lambda n: _sieve(n))
reg("prime_factors", "Prime factorisation of an integer.", C, "n:int", lambda n: _pf(n))
reg("divisors", "All divisors of an integer.", C, "n:int", lambda n: sorted({d for i in range(1, math.isqrt(abs(n)) + 1) if n % i == 0 for d in (i, n // i)}))
reg("euler_totient", "Euler's totient φ(n).", C, "n:int", lambda n: sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1))
reg("mod_pow", "Compute base^exp mod m efficiently.", C, "base:int, exp:int, mod:int", lambda base, exp, mod: pow(base, exp, mod))
reg("mod_inverse", "Modular inverse of a modulo m.", C, "a:int, mod:int", lambda a, mod: pow(a, -1, mod))
reg("collatz_sequence", "The Collatz sequence from n.", C, "n:int", lambda n: _collatz(n))
reg("convert_base", "Convert an integer string between bases 2–36.", C, "value:str, from_base:int=10, to_base:int=2", lambda value, from_base, to_base: _base(value, from_base, to_base))
reg("to_roman", "Integer 1–3999 to a Roman numeral.", C, "n:int", lambda n: _roman(n))
reg("from_roman", "Roman numeral to integer.", C, "numeral:str", lambda numeral: _unroman(numeral))
reg("number_to_words", "Spell a whole number in English words.", C, "n:int", lambda n: _words(n))
reg("simplify_fraction", "Reduce a fraction to lowest terms.", C, "numerator:int, denominator:int", lambda numerator, denominator: str(Fraction(numerator, denominator)))
reg("decimal_to_fraction", "Best fraction for a decimal.", C, "value:float, max_denominator:int=1000", lambda value, max_denominator: str(Fraction(value).limit_denominator(max_denominator)))
reg("fraction_arithmetic", "Add, subtract, multiply or divide two fractions like 1/2 and 3/4.", C, "a:str, op:str, b:str",
    lambda a, op, b: str({"+": Fraction(a) + Fraction(b), "-": Fraction(a) - Fraction(b), "*": Fraction(a) * Fraction(b), "/": Fraction(a) / Fraction(b)}[op]))
reg("ratio_simplify", "Simplify a ratio of integers.", C, "numbers:list", lambda numbers: ":".join(str(int(x) // math.gcd(*[int(y) for y in numbers])) for x in numbers))
reg("proportion_solve", "Solve a/b = c/x for x.", C, "a:float, b:float, c:float", lambda a, b, c: fmt(b * c / a))
reg("degrees_to_radians", "Degrees to radians.", C, "degrees:float", lambda degrees: math.radians(degrees))
reg("radians_to_degrees", "Radians to degrees.", C, "radians:float", lambda radians: math.degrees(radians))
for _f in ("sin", "cos", "tan", "asin", "acos", "atan"):
    reg(f"trig_{_f}", f"{_f} of a value; angle in degrees for sin/cos/tan, result in degrees for asin/acos/atan.", C, "x:float",
        (lambda f: (lambda x: fmt(round(getattr(math, f)(math.radians(x)), 12)) if f in ("sin", "cos", "tan") else fmt(math.degrees(getattr(math, f)(x)))))(_f))
reg("hypotenuse", "Hypotenuse of a right triangle from its legs.", C, "a:float, b:float", lambda a, b: fmt(math.hypot(a, b)))
reg("complex_arithmetic", "Add/subtract/multiply/divide two complex numbers like 3+4j.", C, "a:str, op:str, b:str",
    lambda a, op, b: str({"+": complex(a) + complex(b), "-": complex(a) - complex(b), "*": complex(a) * complex(b), "/": complex(a) / complex(b)}[op]))
reg("complex_polar", "Modulus and angle (degrees) of a complex number.", C, "z:str", lambda z: {"modulus": abs(complex(z)), "angle_degrees": math.degrees(cmath.phase(complex(z)))})

# ---- algebra and calculus
reg("solve_linear", "Solve a·x + b = 0.", "algebra", "a:float, b:float", lambda a, b: fmt(-b / a))
reg("solve_quadratic", "Roots of a·x² + b·x + c = 0 (real or complex).", "algebra", "a:float, b:float, c:float",
    lambda a, b, c: [str(r) if isinstance(r, complex) else fmt(r) for r in _quad(a, b, c)])
reg("solve_cubic", "Roots of a·x³ + b·x² + c·x + d = 0 (numerical).", "algebra", "a:float, b:float, c:float, d:float", lambda a, b, c, d: _poly_roots([a, b, c, d]))
reg("polynomial_roots", "Numerical roots of a polynomial, coefficients highest power first.", "algebra", "coefficients:list", lambda coefficients: _poly_roots(num_list(coefficients)))
reg("polynomial_eval", "Evaluate a polynomial at x (coefficients highest power first).", "algebra", "coefficients:list, x:float", lambda coefficients, x: fmt(_horner(num_list(coefficients), x)))
reg("polynomial_derivative", "Derivative of a polynomial (coefficients highest power first).", "algebra", "coefficients:list", lambda coefficients: _pder(num_list(coefficients)))
reg("polynomial_integral", "Antiderivative of a polynomial, constant 0.", "algebra", "coefficients:list", lambda coefficients: _pint(num_list(coefficients)))
reg("solve_system_2x2", "Solve a1x+b1y=c1 and a2x+b2y=c2.", "algebra", "a1:float, b1:float, c1:float, a2:float, b2:float, c2:float", lambda a1, b1, c1, a2, b2, c2: _sys2(a1, b1, c1, a2, b2, c2))
reg("solve_linear_system", "Solve A·x = b by Gaussian elimination. A is a list of rows.", "algebra", "A:list, b:list", lambda A, b: [fmt(v) for v in _gauss(A, num_list(b))])
reg("find_root", "Find a root of f(x) = 0 near a starting guess (Newton with bisection fallback). Use x as the variable, e.g. 'x**3 - 2*x - 5'.", "algebra", "expression:str, guess:float=1.0", lambda expression, guess: fmt(_root(expression, guess)))
reg("numerical_derivative", "Derivative of f(x) at a point, numerically.", "calculus", "expression:str, x:float", lambda expression, x: fmt(round((evaluate(expression, x=x + 1e-6) - evaluate(expression, x=x - 1e-6)) / 2e-6, 6)))
reg("numerical_integral", "Definite integral of f(x) from a to b by Simpson's rule.", "calculus", "expression:str, a:float, b:float, steps:int=2000", lambda expression, a, b, steps: fmt(round(_simpson(expression, a, b, steps), 10)))
reg("limit_estimate", "Estimate the limit of f(x) as x approaches a.", "calculus", "expression:str, a:float", lambda expression, a: fmt(round((evaluate(expression, x=a + 1e-7) + evaluate(expression, x=a - 1e-7)) / 2, 6)))
reg("series_sum", "Sum f(n) for n from a to b.", "calculus", "expression:str, a:int, b:int", lambda expression, a, b: fmt(sum(evaluate(expression, n=n) for n in range(a, min(b, a + 100000) + 1))))
reg("arithmetic_sequence", "n-th term and sum of an arithmetic sequence.", "algebra", "first:float, difference:float, n:int", lambda first, difference, n: {"term": first + (n - 1) * difference, "sum": n * (2 * first + (n - 1) * difference) / 2})
reg("geometric_sequence", "n-th term and sum of a geometric sequence.", "algebra", "first:float, ratio:float, n:int", lambda first, ratio, n: {"term": first * ratio ** (n - 1), "sum": first * n if ratio == 1 else first * (ratio ** n - 1) / (ratio - 1)})

# ---- vectors and matrices
reg("vector_dot", "Dot product of two vectors.", "linear algebra", "a:list, b:list", lambda a, b: fmt(sum(x * y for x, y in zip(num_list(a), num_list(b)))))
reg("vector_cross", "Cross product of two 3-D vectors.", "linear algebra", "a:list, b:list", lambda a, b: _cross(num_list(a), num_list(b)))
reg("vector_norm", "Length of a vector.", "linear algebra", "v:list", lambda v: fmt(math.sqrt(sum(x * x for x in num_list(v)))))
reg("vector_angle", "Angle in degrees between two vectors.", "linear algebra", "a:list, b:list", lambda a, b: fmt(math.degrees(math.acos(sum(x * y for x, y in zip(num_list(a), num_list(b))) / (math.hypot(*num_list(a)) * math.hypot(*num_list(b)))))))
reg("matrix_multiply", "Multiply two matrices (lists of rows).", "linear algebra", "A:list, B:list", lambda A, B: [[fmt(sum(a * b for a, b in zip(r, c))) for c in zip(*B)] for r in A])
reg("matrix_transpose", "Transpose a matrix.", "linear algebra", "A:list", lambda A: [list(r) for r in zip(*A)])
reg("matrix_determinant", "Determinant of a square matrix.", "linear algebra", "A:list", lambda A: fmt(_det([[Fraction(x).limit_denominator(10 ** 9) for x in r] for r in A])))
reg("matrix_inverse", "Inverse of a square matrix.", "linear algebra", "A:list", lambda A: _inv(A))
reg("matrix_rank", "Rank of a matrix.", "linear algebra", "A:list", lambda A: _rank(A))
reg("matrix_trace", "Trace of a square matrix.", "linear algebra", "A:list", lambda A: fmt(sum(A[i][i] for i in range(len(A)))))
reg("matrix_power", "Integer power of a square matrix.", "linear algebra", "A:list, n:int", lambda A, n: _mpow(A, n))
reg("identity_matrix", "n×n identity matrix.", "linear algebra", "n:int", lambda n: [[1 if i == j else 0 for j in range(n)] for i in range(n)])
reg("eigenvalues_2x2", "Eigenvalues of a 2×2 matrix.", "linear algebra", "a:float, b:float, c:float, d:float", lambda a, b, c, d: [str(r) if isinstance(r, complex) else fmt(r) for r in _quad(1, -(a + d), a * d - b * c)])

# ---- statistics and probability
S = "statistics"
reg("stats_summary", "Count, mean, median, mode, min, max, range, variance and standard deviation of numbers.", S, "numbers:list",
    lambda numbers: (lambda xs: {"count": len(xs), "mean": fmt(st.fmean(xs)), "median": fmt(st.median(xs)), "min": min(xs), "max": max(xs), "range": max(xs) - min(xs),
                                 "stdev_sample": fmt(st.stdev(xs)) if len(xs) > 1 else None, "stdev_population": fmt(st.pstdev(xs)), "sum": fmt(sum(xs))})(num_list(numbers)))
reg("mean", "Arithmetic mean.", S, "numbers:list", lambda numbers: fmt(st.fmean(num_list(numbers))))
reg("median", "Median.", S, "numbers:list", lambda numbers: fmt(st.median(num_list(numbers))))
reg("mode", "Most common value(s).", S, "numbers:list", lambda numbers: st.multimode(num_list(numbers)))
reg("variance", "Sample variance.", S, "numbers:list", lambda numbers: fmt(st.variance(num_list(numbers))))
reg("std_dev", "Sample standard deviation.", S, "numbers:list", lambda numbers: fmt(st.stdev(num_list(numbers))))
reg("geometric_mean", "Geometric mean.", S, "numbers:list", lambda numbers: fmt(st.geometric_mean(num_list(numbers))))
reg("harmonic_mean", "Harmonic mean.", S, "numbers:list", lambda numbers: fmt(st.harmonic_mean(num_list(numbers))))
reg("percentile", "The p-th percentile (0–100) with linear interpolation.", S, "numbers:list, p:float", lambda numbers, p: fmt(_pctl(sorted(num_list(numbers)), p)))
reg("quartiles", "Q1, median, Q3 and IQR.", S, "numbers:list", lambda numbers: (lambda xs: {"q1": _pctl(xs, 25), "median": _pctl(xs, 50), "q3": _pctl(xs, 75), "iqr": _pctl(xs, 75) - _pctl(xs, 25)})(sorted(num_list(numbers))))
reg("z_score", "z-score of a value.", S, "value:float, mean:float, std:float", lambda value, mean, std: fmt((value - mean) / std))
reg("correlation", "Pearson correlation of two lists.", S, "x:list, y:list", lambda x, y: fmt(st.correlation(num_list(x), num_list(y))))
reg("covariance", "Sample covariance of two lists.", S, "x:list, y:list", lambda x, y: fmt(st.covariance(num_list(x), num_list(y))))
reg("linear_regression", "Least-squares line y = a + b·x.", S, "x:list, y:list", lambda x, y: (lambda r: {"slope": fmt(r.slope), "intercept": fmt(r.intercept), "r": fmt(st.correlation(num_list(x), num_list(y)))})(st.linear_regression(num_list(x), num_list(y))))
reg("normal_pdf", "Normal density at x.", S, "x:float, mean:float=0, std:float=1", lambda x, mean, std: st.NormalDist(mean, std).pdf(x))
reg("normal_cdf", "P(X ≤ x) for a normal distribution.", S, "x:float, mean:float=0, std:float=1", lambda x, mean, std: st.NormalDist(mean, std).cdf(x))
reg("normal_inverse", "x such that P(X ≤ x) = p for a normal distribution.", S, "p:float, mean:float=0, std:float=1", lambda p, mean, std: st.NormalDist(mean, std).inv_cdf(p))
reg("binomial_pmf", "P(X = k) for Binomial(n, p).", "probability", "n:int, k:int, p:float", lambda n, k, p: math.comb(n, k) * p ** k * (1 - p) ** (n - k))
reg("binomial_cdf", "P(X ≤ k) for Binomial(n, p).", "probability", "n:int, k:int, p:float", lambda n, k, p: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1)))
reg("poisson_pmf", "P(X = k) for Poisson(λ).", "probability", "k:int, lam:float", lambda k, lam: math.exp(-lam) * lam ** k / math.factorial(k))
reg("geometric_pmf", "P(first success on trial k).", "probability", "k:int, p:float", lambda k, p: (1 - p) ** (k - 1) * p)
reg("bayes_theorem", "P(A|B) from P(B|A), P(A) and P(B).", "probability", "p_b_given_a:float, p_a:float, p_b:float", lambda p_b_given_a, p_a, p_b: p_b_given_a * p_a / p_b)
reg("confidence_interval", "95% (or other) confidence interval for a mean (normal approximation).", S, "mean:float, std:float, n:int, confidence:float=0.95",
    lambda mean, std, n, confidence: (lambda z: [mean - z * std / math.sqrt(n), mean + z * std / math.sqrt(n)])(st.NormalDist().inv_cdf(0.5 + confidence / 2)))
reg("sample_size", "Sample size for a margin of error.", S, "std:float, margin:float, confidence:float=0.95", lambda std, margin, confidence: math.ceil((st.NormalDist().inv_cdf(0.5 + confidence / 2) * std / margin) ** 2))
reg("t_statistic", "One-sample t statistic.", S, "sample_mean:float, mu0:float, std:float, n:int", lambda sample_mean, mu0, std, n: fmt((sample_mean - mu0) / (std / math.sqrt(n))))
reg("chi_square", "Chi-square statistic from observed and expected counts.", S, "observed:list, expected:list", lambda observed, expected: fmt(sum((o - e) ** 2 / e for o, e in zip(num_list(observed), num_list(expected)))))
reg("expected_value", "Expected value of a discrete distribution.", "probability", "values:list, probabilities:list", lambda values, probabilities: fmt(sum(v * p for v, p in zip(num_list(values), num_list(probabilities)))))
reg("entropy_bits", "Shannon entropy in bits.", "probability", "probabilities:list", lambda probabilities: fmt(-sum(p * math.log2(p) for p in num_list(probabilities) if p > 0)))
reg("moving_average", "Moving average with a window.", S, "numbers:list, window:int=3", lambda numbers, window: [fmt(round(sum(xs[i:i + window]) / window, 10)) for xs in [num_list(numbers)] for i in range(len(xs) - window + 1)])
reg("normalize_list", "Scale numbers to 0–1 (min-max).", S, "numbers:list", lambda numbers: [fmt(round((x - min(xs)) / (max(xs) - min(xs)), 10)) for xs in [num_list(numbers)] for x in xs])
reg("cumulative_sum", "Running totals.", S, "numbers:list", lambda numbers: [fmt(v) for v in itertools.accumulate(num_list(numbers))])
reg("histogram_counts", "Count values into equal-width bins.", S, "numbers:list, bins:int=5", lambda numbers, bins: _hist(num_list(numbers), bins))
reg("outliers_iqr", "Values outside 1.5×IQR.", S, "numbers:list", lambda numbers: (lambda xs: [x for x in xs if x < _pctl(xs, 25) - 1.5 * (_pctl(xs, 75) - _pctl(xs, 25)) or x > _pctl(xs, 75) + 1.5 * (_pctl(xs, 75) - _pctl(xs, 25))])(sorted(num_list(numbers))))
reg("dice_probability", "Probability that n dice sum to a target.", "probability", "dice:int, target:int, sides:int=6", lambda dice, target, sides: _dice(dice, target, sides))
reg("birthday_probability", "Chance two of n people share a birthday.", "probability", "people:int", lambda people: 1 - math.prod((365 - i) / 365 for i in range(people)))
reg("random_number", "A random number in a range (seeded for repeatability).", "probability", "low:float=0, high:float=1, seed:int?", lambda low, high, seed: __import__("random").Random(seed).uniform(low, high))
reg("random_integers", "Random integers in a range.", "probability", "count:int, low:int=1, high:int=100, seed:int?", lambda count, low, high, seed: (lambda r: [r.randint(low, high) for _ in range(min(count, 1000))])(__import__("random").Random(seed)))
reg("shuffle_list", "Shuffle items.", "probability", "items:list, seed:int?", lambda items, seed: (lambda r, xs: (r.shuffle(xs), xs)[1])(__import__("random").Random(seed), list(items)))
reg("coin_flips", "Simulate coin flips.", "probability", "count:int=10, seed:int?", lambda count, seed: (lambda r: ["H" if r.random() < .5 else "T" for _ in range(min(count, 1000))])(__import__("random").Random(seed)))

# ---- geometry
G = "geometry"
reg("circle_area", "Area of a circle from its radius.", G, "radius:float", lambda radius: fmt(math.pi * radius ** 2))
reg("circle_circumference", "Circumference from the radius.", G, "radius:float", lambda radius: fmt(2 * math.pi * radius))
reg("triangle_area_heron", "Area of a triangle from its three sides.", G, "a:float, b:float, c:float", lambda a, b, c: fmt(math.sqrt((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))))
reg("triangle_area_base_height", "Area from base and height.", G, "base:float, height:float", lambda base, height: fmt(base * height / 2))
reg("rectangle_area", "Area and perimeter of a rectangle.", G, "width:float, height:float", lambda width, height: {"area": width * height, "perimeter": 2 * (width + height), "diagonal": math.hypot(width, height)})
reg("trapezoid_area", "Area of a trapezoid.", G, "a:float, b:float, height:float", lambda a, b, height: fmt((a + b) * height / 2))
reg("polygon_area", "Area of a polygon from its vertices [[x,y],…] (shoelace).", G, "points:list", lambda points: fmt(abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))) / 2))
reg("regular_polygon", "Area, perimeter and interior angle of a regular polygon.", G, "sides:int, side_length:float", lambda sides, side_length: {"area": fmt(sides * side_length ** 2 / (4 * math.tan(math.pi / sides))), "perimeter": sides * side_length, "interior_angle": (sides - 2) * 180 / sides})
reg("sphere", "Volume and surface area of a sphere.", G, "radius:float", lambda radius: {"volume": fmt(4 / 3 * math.pi * radius ** 3), "surface_area": fmt(4 * math.pi * radius ** 2)})
reg("cylinder", "Volume and surface area of a cylinder.", G, "radius:float, height:float", lambda radius, height: {"volume": fmt(math.pi * radius ** 2 * height), "surface_area": fmt(2 * math.pi * radius * (radius + height))})
reg("cone", "Volume and surface area of a cone.", G, "radius:float, height:float", lambda radius, height: {"volume": fmt(math.pi * radius ** 2 * height / 3), "surface_area": fmt(math.pi * radius * (radius + math.hypot(radius, height)))})
reg("box", "Volume and surface area of a box.", G, "length:float, width:float, height:float", lambda length, width, height: {"volume": length * width * height, "surface_area": 2 * (length * width + width * height + length * height)})
reg("distance_2d", "Distance between two points.", G, "x1:float, y1:float, x2:float, y2:float", lambda x1, y1, x2, y2: fmt(math.hypot(x2 - x1, y2 - y1)))
reg("distance_3d", "Distance between two 3-D points.", G, "a:list, b:list", lambda a, b: fmt(math.dist(num_list(a), num_list(b))))
reg("midpoint", "Midpoint of two points.", G, "x1:float, y1:float, x2:float, y2:float", lambda x1, y1, x2, y2: [(x1 + x2) / 2, (y1 + y2) / 2])
reg("slope_of_line", "Slope and intercept of the line through two points.", G, "x1:float, y1:float, x2:float, y2:float", lambda x1, y1, x2, y2: {"slope": fmt((y2 - y1) / (x2 - x1)), "intercept": fmt(y1 - (y2 - y1) / (x2 - x1) * x1)})
reg("haversine_distance", "Great-circle distance in km between two latitude/longitude points.", G, "lat1:float, lon1:float, lat2:float, lon2:float",
    lambda lat1, lon1, lat2, lon2: fmt(round(2 * 6371.0088 * math.asin(math.sqrt(math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)), 4)))
reg("law_of_cosines", "Third side of a triangle from two sides and the included angle (degrees).", G, "a:float, b:float, angle:float", lambda a, b, angle: fmt(math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(angle)))))
reg("circle_from_three_points", "Centre and radius of the circle through three points.", G, "p1:list, p2:list, p3:list", lambda p1, p2, p3: _circ(p1, p2, p3))

# ---- units and constants
UNITS = {
    "length": {"m": 1, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "um": 1e-6, "nm": 1e-9, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344, "nmi": 1852, "ly": 9.4607304725808e15, "au": 1.495978707e11, "pc": 3.0856775814914e16, "angstrom": 1e-10, "mil": 2.54e-5},
    "mass": {"kg": 1, "g": 1e-3, "mg": 1e-6, "t": 1000, "lb": 0.45359237, "oz": 0.028349523125, "st": 6.35029318, "ct": 2e-4, "u": 1.66053906660e-27, "slug": 14.5939029},
    "time": {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "min": 60, "h": 3600, "day": 86400, "week": 604800, "month": 2629800, "year": 31557600},
    "area": {"m2": 1, "cm2": 1e-4, "mm2": 1e-6, "km2": 1e6, "in2": 6.4516e-4, "ft2": 0.09290304, "yd2": 0.83612736, "acre": 4046.8564224, "ha": 1e4, "mi2": 2589988.110336},
    "volume": {"m3": 1, "l": 1e-3, "ml": 1e-6, "cm3": 1e-6, "gal": 3.785411784e-3, "qt": 9.46352946e-4, "pt": 4.73176473e-4, "cup": 2.365882365e-4, "floz": 2.95735295625e-5, "tbsp": 1.478676478125e-5, "tsp": 4.92892159375e-6, "gal_uk": 4.54609e-3, "bbl": 0.158987294928},
    "speed": {"m/s": 1, "km/h": 1 / 3.6, "mph": 0.44704, "kn": 0.514444, "ft/s": 0.3048, "mach": 343, "c": 299792458},
    "pressure": {"pa": 1, "kpa": 1e3, "mpa": 1e6, "bar": 1e5, "atm": 101325, "psi": 6894.757293168, "mmhg": 133.322387415, "torr": 133.322368421},
    "energy": {"j": 1, "kj": 1e3, "cal": 4.184, "kcal": 4184, "wh": 3600, "kwh": 3.6e6, "ev": 1.602176634e-19, "btu": 1055.05585262, "erg": 1e-7, "ftlbf": 1.3558179483314},
    "power": {"w": 1, "kw": 1e3, "mw": 1e6, "hp": 745.69987158227, "btu/h": 0.29307107},
    "data": {"bit": 1, "b": 8, "kb": 8e3, "mb": 8e6, "gb": 8e9, "tb": 8e12, "pb": 8e15, "kib": 8192, "mib": 8 * 1048576, "gib": 8 * 1073741824, "tib": 8 * 1099511627776},
    "angle": {"rad": 1, "deg": math.pi / 180, "grad": math.pi / 200, "turn": math.tau, "arcmin": math.pi / 10800, "arcsec": math.pi / 648000},
    "frequency": {"hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "rpm": 1 / 60},
    "force": {"n": 1, "kn": 1e3, "lbf": 4.4482216152605, "dyn": 1e-5, "kgf": 9.80665},
}


def convert(value: float, frm: str, to: str, category: str | None = None) -> float:
    f, t = frm.strip().lower().replace("²", "2").replace("³", "3"), to.strip().lower().replace("²", "2").replace("³", "3")
    if f in ("c", "f", "k", "r", "°c", "°f", "celsius", "fahrenheit", "kelvin", "rankine") or t in ("c", "f", "k", "r", "°c", "°f", "celsius", "fahrenheit", "kelvin", "rankine"):
        norm = {"c": "c", "°c": "c", "celsius": "c", "f": "f", "°f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k", "r": "r", "rankine": "r"}
        a, b = norm.get(f), norm.get(t)
        if not a or not b:
            raise ToolError("temperature units: C, F, K, R")
        k = {"c": value + 273.15, "f": (value - 32) * 5 / 9 + 273.15, "k": value, "r": value * 5 / 9}[a]
        return {"c": k - 273.15, "f": (k - 273.15) * 9 / 5 + 32, "k": k, "r": k * 9 / 5}[b]
    for cat, table in UNITS.items():
        if (category in (None, cat)) and f in table and t in table:
            return value * table[f] / table[t]
    raise ToolError(f"cannot convert '{frm}' to '{to}'. Known units: " + "; ".join(f"{c}: {', '.join(t)}" for c, t in UNITS.items()))


reg("convert_units", "Convert between units: length, mass, time, area, volume, speed, pressure, energy, power, data, angle, frequency, force, temperature.", "units", "value:float, from_unit:str, to_unit:str",
    lambda value, from_unit, to_unit: fmt(float(f"{convert(value, from_unit, to_unit):.12g}")), tags=("google", "converter"))
for _cat in list(UNITS) + ["temperature"]:
    reg(f"convert_{_cat}", f"Convert {_cat} units" + (f" ({', '.join(UNITS[_cat])})" if _cat in UNITS else " (C, F, K, R)") + ".", "units", "value:float, from_unit:str, to_unit:str",
        (lambda cat: lambda value, from_unit, to_unit: fmt(float(f"{convert(value, from_unit, to_unit, cat if cat != 'temperature' else None):.12g}")))(_cat))
reg("list_units", "List the units the converter knows.", "units", "", lambda: {c: list(t) for c, t in UNITS.items()} | {"temperature": ["C", "F", "K", "R"]})
reg("celsius_to_fahrenheit", "°C to °F.", "units", "c:float", lambda c: fmt(c * 9 / 5 + 32))
reg("fahrenheit_to_celsius", "°F to °C.", "units", "f:float", lambda f: fmt(round((f - 32) * 5 / 9, 10)))
reg("bmi", "Body-mass index from kg and metres.", "units", "kg:float, metres:float", lambda kg, metres: fmt(round(kg / metres ** 2, 2)))
reg("fuel_economy", "Convert mpg to litres per 100 km.", "units", "mpg:float", lambda mpg: fmt(round(235.215 / mpg, 3)))

CONSTANTS = {
    "speed_of_light": (299792458, "m/s"), "gravitational_constant": (6.67430e-11, "m^3 kg^-1 s^-2"), "planck_constant": (6.62607015e-34, "J s"),
    "reduced_planck_constant": (1.054571817e-34, "J s"), "elementary_charge": (1.602176634e-19, "C"), "boltzmann_constant": (1.380649e-23, "J/K"),
    "avogadro_number": (6.02214076e23, "1/mol"), "gas_constant": (8.314462618, "J/(mol K)"), "stefan_boltzmann": (5.670374419e-8, "W m^-2 K^-4"),
    "vacuum_permittivity": (8.8541878128e-12, "F/m"), "vacuum_permeability": (1.25663706212e-6, "N/A^2"), "electron_mass": (9.1093837015e-31, "kg"),
    "proton_mass": (1.67262192369e-27, "kg"), "neutron_mass": (1.67492749804e-27, "kg"), "atomic_mass_unit": (1.66053906660e-27, "kg"),
    "fine_structure_constant": (7.2973525693e-3, ""), "rydberg_constant": (10973731.568160, "1/m"), "standard_gravity": (9.80665, "m/s^2"),
    "standard_atmosphere": (101325, "Pa"), "astronomical_unit": (1.495978707e11, "m"), "light_year": (9.4607304725808e15, "m"), "parsec": (3.0856775814914e16, "m"),
    "earth_mass": (5.9722e24, "kg"), "earth_radius": (6.371e6, "m"), "sun_mass": (1.98847e30, "kg"), "sun_radius": (6.957e8, "m"), "sun_luminosity": (3.828e26, "W"),
    "moon_mass": (7.342e22, "kg"), "moon_radius": (1.7374e6, "m"), "earth_moon_distance": (3.844e8, "m"), "hubble_constant": (70, "km/s/Mpc"),
    "wien_constant": (2.897771955e-3, "m K"), "faraday_constant": (96485.33212, "C/mol"), "bohr_radius": (5.29177210903e-11, "m"), "electron_volt": (1.602176634e-19, "J"),
    "sound_speed_air": (343, "m/s"), "water_density": (1000, "kg/m^3"), "water_specific_heat": (4184, "J/(kg K)"), "absolute_zero": (-273.15, "C"),
    "mars_mass": (6.4171e23, "kg"), "jupiter_mass": (1.8982e27, "kg"), "mars_radius": (3.3895e6, "m"), "jupiter_radius": (6.9911e7, "m"),
}
reg("physical_constant", "Look up a physical or astronomical constant by name.", "science", "name:str", lambda name: (lambda k: {"name": k, "value": CONSTANTS[k][0], "unit": CONSTANTS[k][1]})(_closest(name, CONSTANTS)))
reg("list_constants", "List all known physical constants.", "science", "", lambda: {k: f"{v[0]} {v[1]}".strip() for k, v in CONSTANTS.items()})

_EL = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
_NAMES = ("Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson").split()
_MASS = [1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798,
         85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209, 210, 222, 223, 226, 227, 232.04, 231.04, 238.03]
assert len(_EL) == len(_NAMES) == 118 and len(_MASS) == 92
MASS = {s: m for s, m in zip(_EL, _MASS)}


def element(q: str) -> dict:
    q = q.strip()
    for i, (s, n) in enumerate(zip(_EL, _NAMES), 1):
        if q.lower() in (s.lower(), n.lower()) or q == str(i):
            return {"number": i, "symbol": s, "name": n, "atomic_mass": MASS.get(s), "period": next(p for p, e in enumerate((2, 10, 18, 36, 54, 86, 118), 1) if i <= e), "block": "s" if s in ("H", "He") or i in (3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88) else "d" if 21 <= i <= 30 or 39 <= i <= 48 or 72 <= i <= 80 or 104 <= i <= 112 else "f" if 57 <= i <= 71 or 89 <= i <= 103 else "p"}
    raise ToolError(f"no element matches '{q}'")


reg("element_info", "Look up a chemical element by symbol, name or atomic number.", "chemistry", "query:str", element)
reg("list_elements", "All 118 elements: number, symbol, name.", "chemistry", "", lambda: [f"{i} {s} {n}" for i, (s, n) in enumerate(zip(_EL, _NAMES), 1)])
reg("molar_mass", "Molar mass of a chemical formula such as C6H12O6 or Ca(OH)2.", "chemistry", "formula:str", lambda formula: round(_molar(formula), 4))
reg("formula_composition", "Percentage composition by mass of a formula.", "chemistry", "formula:str", lambda formula: {k: round(v / _molar(formula) * 100, 3) for k, v in _count(formula).items() for v in [MASS[k] * v]})
reg("moles_from_mass", "Moles from mass (g) and formula.", "chemistry", "mass_g:float, formula:str", lambda mass_g, formula: fmt(mass_g / _molar(formula)))
reg("mass_from_moles", "Mass (g) from moles and formula.", "chemistry", "moles:float, formula:str", lambda moles, formula: fmt(moles * _molar(formula)))
reg("molarity", "Molarity from moles and litres.", "chemistry", "moles:float, litres:float", lambda moles, litres: fmt(moles / litres))
reg("dilution", "Solve M1·V1 = M2·V2 for the unknown (give three, leave one empty).", "chemistry", "m1:float?, v1:float?, m2:float?, v2:float?",
    lambda m1, v1, m2, v2: (lambda d: {k: v for k, v in zip(("m1", "v1", "m2", "v2"), d)})(_dil(m1, v1, m2, v2)))
reg("ph_from_h", "pH from [H+] in mol/L.", "chemistry", "h_concentration:float", lambda h_concentration: fmt(round(-math.log10(h_concentration), 4)))
reg("h_from_ph", "[H+] from pH.", "chemistry", "ph:float", lambda ph: 10 ** -ph)
reg("ideal_gas", "Solve PV = nRT (kPa, litres, mol, K). Leave exactly one empty.", "chemistry", "p_kpa:float?, v_litres:float?, n_mol:float?, t_kelvin:float?", lambda p_kpa, v_litres, n_mol, t_kelvin: _gas(p_kpa, v_litres, n_mol, t_kelvin))
reg("half_life_remaining", "Fraction remaining after a time given the half-life.", "chemistry", "elapsed:float, half_life:float", lambda elapsed, half_life: 0.5 ** (elapsed / half_life))
reg("empirical_ratio", "Mole ratio from element masses, e.g. {\"C\":12,\"H\":2}.", "chemistry", "masses:dict", lambda masses: (lambda r: {k: round(v / min(r.values()), 3) for k, v in r.items()})({k: v / MASS[k] for k, v in masses.items()}))
reg("balance_equation_check", "Check whether a reaction is balanced: 'H2 + O2 -> H2O' with coefficients, e.g. '2H2 + O2 -> 2H2O'.", "chemistry", "equation:str", lambda equation: _balanced(equation))

# ---- physics
Y = "physics"
reg("kinematics_speed", "Average speed from distance and time.", Y, "distance:float, time:float", lambda distance, time: fmt(distance / time))
reg("kinematics_suvat", "Final velocity, distance and time from u, a and t: v = u + at; s = ut + ½at².", Y, "u:float, a:float, t:float", lambda u, a, t: {"v": u + a * t, "s": u * t + a * t * t / 2})
reg("projectile_motion", "Range, max height and flight time for a launch (m/s, degrees).", Y, "speed:float, angle:float, height:float=0, g:float=9.80665", lambda speed, angle, height, g: _proj(speed, angle, height, g))
reg("force_newton", "F = m·a.", Y, "mass:float, acceleration:float", lambda mass, acceleration: fmt(mass * acceleration))
reg("kinetic_energy", "½mv².", Y, "mass:float, velocity:float", lambda mass, velocity: fmt(mass * velocity ** 2 / 2))
reg("potential_energy", "mgh.", Y, "mass:float, height:float, g:float=9.80665", lambda mass, height, g: fmt(mass * g * height))
reg("momentum", "p = m·v.", Y, "mass:float, velocity:float", lambda mass, velocity: fmt(mass * velocity))
reg("work_power", "Work (J) and power (W) from force, distance and time.", Y, "force:float, distance:float, time:float", lambda force, distance, time: {"work": force * distance, "power": force * distance / time})
reg("ohms_law", "Solve V = I·R (leave one empty).", Y, "v:float?, i:float?, r:float?", lambda v, i, r: _ohm(v, i, r))
reg("electric_power", "P = V·I and I²R.", Y, "voltage:float, current:float", lambda voltage, current: {"power_w": fmt(voltage * current), "resistance_ohm": fmt(voltage / current)})
reg("resistors_series", "Total resistance in series.", Y, "resistors:list", lambda resistors: fmt(sum(num_list(resistors))))
reg("resistors_parallel", "Total resistance in parallel.", Y, "resistors:list", lambda resistors: fmt(round(1 / sum(1 / r for r in num_list(resistors)), 10)))
reg("capacitors_series", "Total capacitance in series.", Y, "capacitors:list", lambda capacitors: fmt(1 / sum(1 / c for c in num_list(capacitors))))
reg("rc_time_constant", "τ = R·C and the cutoff frequency.", Y, "resistance:float, capacitance:float", lambda resistance, capacitance: {"tau": resistance * capacitance, "cutoff_hz": 1 / (2 * math.pi * resistance * capacitance)})
reg("wave_speed", "v = f·λ.", Y, "frequency:float, wavelength:float", lambda frequency, wavelength: fmt(frequency * wavelength))
reg("pendulum_period", "Period of a simple pendulum.", Y, "length:float, g:float=9.80665", lambda length, g: fmt(2 * math.pi * math.sqrt(length / g)))
reg("spring_period", "Period of a mass on a spring.", Y, "mass:float, k:float", lambda mass, k: fmt(2 * math.pi * math.sqrt(mass / k)))
reg("gravitational_force", "Newton's law of gravitation.", Y, "m1:float, m2:float, distance:float", lambda m1, m2, distance: 6.6743e-11 * m1 * m2 / distance ** 2)
reg("escape_velocity", "Escape velocity from mass (kg) and radius (m).", Y, "mass:float, radius:float", lambda mass, radius: math.sqrt(2 * 6.6743e-11 * mass / radius))
reg("orbital_period", "Kepler's third law for a circular orbit around a mass.", Y, "central_mass:float, radius:float", lambda central_mass, radius: 2 * math.pi * math.sqrt(radius ** 3 / (6.6743e-11 * central_mass)))
reg("photon_energy", "Photon energy (J and eV) from wavelength in nm.", Y, "wavelength_nm:float", lambda wavelength_nm: {"joules": 6.62607015e-34 * 299792458 / (wavelength_nm * 1e-9), "ev": 6.62607015e-34 * 299792458 / (wavelength_nm * 1e-9) / 1.602176634e-19})
reg("lorentz_factor", "γ for a speed as a fraction of c.", Y, "beta:float", lambda beta: 1 / math.sqrt(1 - beta ** 2))
reg("doppler_shift", "Observed frequency for a moving source (m/s, sound speed 343).", Y, "frequency:float, source_speed:float, wave_speed:float=343", lambda frequency, source_speed, wave_speed: frequency * wave_speed / (wave_speed - source_speed))
reg("heat_energy", "Q = m·c·ΔT.", Y, "mass:float, specific_heat:float, delta_t:float", lambda mass, specific_heat, delta_t: fmt(mass * specific_heat * delta_t))
reg("carnot_efficiency", "Maximum engine efficiency between two temperatures (K).", Y, "t_hot:float, t_cold:float", lambda t_hot, t_cold: fmt(1 - t_cold / t_hot))
reg("lens_equation", "Image distance from focal length and object distance.", Y, "focal:float, object_distance:float", lambda focal, object_distance: fmt(1 / (1 / focal - 1 / object_distance)))
reg("snell_law", "Refraction angle (degrees) from n1, n2 and the incident angle.", Y, "n1:float, n2:float, angle:float", lambda n1, n2, angle: fmt(math.degrees(math.asin(n1 * math.sin(math.radians(angle)) / n2))))
reg("terminal_velocity", "Terminal velocity of a falling object.", Y, "mass:float, drag_coefficient:float, area:float, air_density:float=1.225", lambda mass, drag_coefficient, area, air_density: math.sqrt(2 * mass * 9.80665 / (air_density * area * drag_coefficient)))
reg("sound_level", "Sound intensity level in dB from intensity in W/m².", Y, "intensity:float", lambda intensity: 10 * math.log10(intensity / 1e-12))
reg("decibel_ratio", "Decibels from a power or amplitude ratio.", Y, "ratio:float, kind:str=power", lambda ratio, kind: (10 if kind == "power" else 20) * math.log10(ratio))
reg("blackbody_peak", "Wien peak wavelength (nm) for a temperature in K.", Y, "temperature:float", lambda temperature: 2.897771955e-3 / temperature * 1e9)

# ---- finance
F = "finance"
reg("compound_interest", "Future value with compound interest.", F, "principal:float, rate_percent:float, years:float, per_year:int=1", lambda principal, rate_percent, years, per_year: fmt(round(principal * (1 + rate_percent / 100 / per_year) ** (per_year * years), 2)))
reg("simple_interest", "Simple interest earned.", F, "principal:float, rate_percent:float, years:float", lambda principal, rate_percent, years: fmt(principal * rate_percent / 100 * years))
reg("loan_payment", "Monthly payment for a loan.", F, "principal:float, annual_rate_percent:float, years:float", lambda principal, annual_rate_percent, years: fmt(round(_pmt(principal, annual_rate_percent / 1200, years * 12), 2)))
reg("loan_schedule", "Amortisation schedule (first rows) for a loan.", F, "principal:float, annual_rate_percent:float, years:float, rows:int=12", lambda principal, annual_rate_percent, years, rows: _sched(principal, annual_rate_percent / 1200, int(years * 12), rows))
reg("present_value", "Present value of a future amount.", F, "future:float, rate_percent:float, years:float", lambda future, rate_percent, years: fmt(round(future / (1 + rate_percent / 100) ** years, 2)))
reg("future_value_annuity", "Future value of regular deposits.", F, "payment:float, rate_percent:float, periods:int", lambda payment, rate_percent, periods: fmt(round(payment * (((1 + rate_percent / 100) ** periods - 1) / (rate_percent / 100)), 2)))
reg("npv", "Net present value of cash flows (first is time 0).", F, "rate_percent:float, cash_flows:list", lambda rate_percent, cash_flows: fmt(round(sum(c / (1 + rate_percent / 100) ** i for i, c in enumerate(num_list(cash_flows))), 2)))
reg("irr", "Internal rate of return (%) of cash flows.", F, "cash_flows:list", lambda cash_flows: round(_irr(num_list(cash_flows)) * 100, 4))
reg("cagr", "Compound annual growth rate (%).", F, "start:float, end:float, years:float", lambda start, end, years: round(((end / start) ** (1 / years) - 1) * 100, 4))
reg("rule_of_72", "Years to double at a rate.", F, "rate_percent:float", lambda rate_percent: fmt(72 / rate_percent))
reg("break_even", "Break-even units.", F, "fixed_costs:float, price:float, variable_cost:float", lambda fixed_costs, price, variable_cost: math.ceil(fixed_costs / (price - variable_cost)))
reg("roi", "Return on investment (%).", F, "gain:float, cost:float", lambda gain, cost: fmt((gain - cost) / cost * 100))
reg("depreciation_straight_line", "Annual straight-line depreciation.", F, "cost:float, salvage:float, life_years:float", lambda cost, salvage, life_years: fmt((cost - salvage) / life_years))
reg("tax_calculator", "Add tax to a price and show the tax amount.", F, "price:float, tax_percent:float", lambda price, tax_percent: {"tax": round(price * tax_percent / 100, 2), "total": round(price * (1 + tax_percent / 100), 2)})
reg("tip_split", "Tip and per-person share of a bill.", F, "bill:float, tip_percent:float=15, people:int=1", lambda bill, tip_percent, people: {"tip": round(bill * tip_percent / 100, 2), "total": round(bill * (1 + tip_percent / 100), 2), "per_person": round(bill * (1 + tip_percent / 100) / people, 2)})
reg("inflation_adjust", "Value of money after inflation.", F, "amount:float, inflation_percent:float, years:float", lambda amount, inflation_percent, years: fmt(round(amount / (1 + inflation_percent / 100) ** years, 2)))
reg("black_scholes_call", "Black–Scholes European call price.", F, "spot:float, strike:float, years:float, rate_percent:float, volatility_percent:float", lambda spot, strike, years, rate_percent, volatility_percent: _bs(spot, strike, years, rate_percent / 100, volatility_percent / 100))


# ------------------------------------------------------------------ helpers
def _fib(n):
    if not 0 <= n <= 100000:
        raise ToolError("n must be 0..100000")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _isprime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _sieve(n):
    if n > 1_000_000:
        raise ToolError("n must be at most 1,000,000")
    s = bytearray([1]) * (n + 1)
    s[:2] = b"\x00\x00"[: min(2, n + 1)]
    for i in range(2, math.isqrt(n) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i, v in enumerate(s) if v]


def _pf(n):
    if n < 2 or n > 10 ** 15:
        raise ToolError("n must be between 2 and 10^15")
    out, p = [], 2
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1 if p == 2 else 2
    return out + ([n] if n > 1 else [])


def _collatz(n):
    seq = [n]
    while n != 1 and len(seq) < 10000:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq


def _base(v, a, b):
    n = int(v.strip(), a)
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out, neg = "", n < 0
    n = abs(n)
    while n:
        out = digits[n % b] + out
        n //= b
    return ("-" if neg else "") + out


_RN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _roman(n):
    if not 1 <= n <= 3999:
        raise ToolError("1..3999 only")
    out = ""
    for v, s in _RN:
        while n >= v:
            out += s
            n -= v
    return out


def _unroman(s):
    val = dict(zip("IVXLCDM", (1, 5, 10, 50, 100, 500, 1000)))
    s = s.upper()
    if not s or any(c not in val for c in s):
        raise ToolError("not a Roman numeral")
    return sum(-val[c] if i + 1 < len(s) and val[c] < val[s[i + 1]] else val[c] for i, c in enumerate(s))


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n):
    if n < 0:
        return "minus " + _words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" and " + _words(n % 100) if n % 100 else "")
    for size, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            return _words(n // size) + " " + name + (" " + _words(n % size) if n % size else "")
    return str(n)


def _quad(a, b, c):
    if a == 0:
        raise ToolError("a must not be 0 (use solve_linear)")
    d = b * b - 4 * a * c
    if d >= 0:
        r = math.sqrt(d)
        return [(-b + r) / (2 * a), (-b - r) / (2 * a)]
    r = cmath.sqrt(d)
    return [(-b + r) / (2 * a), (-b - r) / (2 * a)]


def _horner(co, x):
    r = 0
    for c in co:
        r = r * x + c
    return r


def _poly_roots(co):
    while co and co[0] == 0:
        co = co[1:]
    if len(co) < 2:
        raise ToolError("need at least a degree-1 polynomial")
    n = len(co) - 1
    co = [c / co[0] for c in co]
    z = [complex(0.4, 0.9) ** k for k in range(n)]  # Durand–Kerner
    for _ in range(500):
        new = []
        for i, zi in enumerate(z):
            den = 1
            for j, zj in enumerate(z):
                if i != j:
                    den *= zi - zj
            new.append(zi - _horner(co, zi) / den)
        if max(abs(a - b) for a, b in zip(z, new)) < 1e-13:
            z = new
            break
        z = new
    return [fmt(round(r.real, 9)) if abs(r.imag) < 1e-8 else f"{r.real:.9g}{r.imag:+.9g}j" for r in z]


def _pder(co):
    n = len(co) - 1
    return [fmt(c * (n - i)) for i, c in enumerate(co[:-1])] or [0]


def _pint(co):
    n = len(co)
    return [fmt(round(c / (n - i), 12)) for i, c in enumerate(co)] + [0]


def _sys2(a1, b1, c1, a2, b2, c2):
    d = a1 * b2 - a2 * b1
    if d == 0:
        raise ToolError("no unique solution (the lines are parallel or identical)")
    return {"x": fmt((c1 * b2 - c2 * b1) / d), "y": fmt((a1 * c2 - a2 * c1) / d)}


def _gauss(A, b):
    n = len(A)
    M = [[Fraction(x).limit_denominator(10 ** 9) for x in row] + [Fraction(bi).limit_denominator(10 ** 9)] for row, bi in zip(A, b)]
    for i in range(n):
        p = next((r for r in range(i, n) if M[r][i] != 0), None)
        if p is None:
            raise ToolError("the system has no unique solution")
        M[i], M[p] = M[p], M[i]
        M[i] = [x / M[i][i] for x in M[i]]
        for r in range(n):
            if r != i:
                M[r] = [x - M[r][i] * y for x, y in zip(M[r], M[i])]
    return [float(M[i][n]) for i in range(n)]


def _root(expr, guess):
    x = guess
    for _ in range(100):
        fx = evaluate(expr, x=x)
        d = (evaluate(expr, x=x + 1e-7) - evaluate(expr, x=x - 1e-7)) / 2e-7
        if d == 0:
            break
        nx = x - fx / d
        if abs(nx - x) < 1e-12:
            return nx
        x = nx
    lo, hi = -1e6, 1e6
    for a in (-1000, -100, -10, -1, 0, 1, 10, 100):
        for b in (a + 1, a + 10, a + 100):
            try:
                if evaluate(expr, x=a) * evaluate(expr, x=b) < 0:
                    lo, hi = a, b
                    break
            except ToolError:
                pass
    if evaluate(expr, x=lo) * evaluate(expr, x=hi) > 0:
        raise ToolError("could not find a root; try another starting guess")
    for _ in range(200):
        mid = (lo + hi) / 2
        if evaluate(expr, x=lo) * evaluate(expr, x=mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _simpson(expr, a, b, n):
    n = max(2, min(n, 200000)) // 2 * 2
    h = (b - a) / n
    s = evaluate(expr, x=a) + evaluate(expr, x=b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * evaluate(expr, x=a + i * h)
    return s * h / 3


def _cross(a, b):
    if len(a) != 3 or len(b) != 3:
        raise ToolError("cross product needs two 3-D vectors")
    return [fmt(a[1] * b[2] - a[2] * b[1]), fmt(a[2] * b[0] - a[0] * b[2]), fmt(a[0] * b[1] - a[1] * b[0])]


def _det(M):
    n = len(M)
    if any(len(r) != n for r in M):
        raise ToolError("matrix must be square")
    M = [r[:] for r in M]
    d = Fraction(1)
    for i in range(n):
        p = next((r for r in range(i, n) if M[r][i] != 0), None)
        if p is None:
            return Fraction(0)
        if p != i:
            M[i], M[p] = M[p], M[i]
            d = -d
        d *= M[i][i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return float(d)


def _inv(A):
    n = len(A)
    M = [[Fraction(x).limit_denominator(10 ** 9) for x in row] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        p = next((r for r in range(i, n) if M[r][i] != 0), None)
        if p is None:
            raise ToolError("matrix is singular (no inverse)")
        M[i], M[p] = M[p], M[i]
        M[i] = [x / M[i][i] for x in M[i]]
        for r in range(n):
            if r != i:
                M[r] = [x - M[r][i] * y for x, y in zip(M[r], M[i])]
    return [[fmt(round(float(x), 10)) for x in row[n:]] for row in M]


def _rank(A):
    M = [[Fraction(x).limit_denominator(10 ** 9) for x in r] for r in A]
    rank, rows, cols = 0, len(M), len(M[0])
    for c in range(cols):
        p = next((r for r in range(rank, rows) if M[r][c] != 0), None)
        if p is None:
            continue
        M[rank], M[p] = M[p], M[rank]
        for r in range(rows):
            if r != rank and M[r][c] != 0:
                f = M[r][c] / M[rank][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[rank])]
        rank += 1
    return rank


def _mpow(A, n):
    r = [[int(i == j) for j in range(len(A))] for i in range(len(A))]
    for _ in range(min(n, 1000)):
        r = [[sum(a * b for a, b in zip(row, col)) for col in zip(*A)] for row in r]
    return r


def _pctl(xs, p):
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def _hist(xs, bins):
    lo, hi = min(xs), max(xs)
    w = (hi - lo) / bins or 1
    out = [0] * bins
    for x in xs:
        out[min(bins - 1, int((x - lo) / w))] += 1
    return [{"from": round(lo + i * w, 6), "to": round(lo + (i + 1) * w, 6), "count": c} for i, c in enumerate(out)]


def _dice(d, target, s):
    ways = {0: 1}
    for _ in range(d):
        nw = {}
        for k, v in ways.items():
            for f in range(1, s + 1):
                nw[k + f] = nw.get(k + f, 0) + v
        ways = nw
    return ways.get(target, 0) / s ** d


def _circ(p1, p2, p3):
    (ax, ay), (bx, by), (cx, cy) = p1, p2, p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        raise ToolError("the points are in a straight line")
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return {"center": [fmt(ux), fmt(uy)], "radius": fmt(math.hypot(ax - ux, ay - uy))}


def _closest(name, table):
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in table:
        return key
    near = [k for k in table if key in k or k in key]
    if near:
        return near[0]
    raise ToolError(f"unknown constant '{name}'. Try list_constants.")


def _count(formula):
    import re as _re

    def parse(s, i=0):
        out: dict[str, float] = {}
        while i < len(s):
            if s[i] == "(":
                inner, i = parse(s, i + 1)
                m = _re.match(r"\d+", s[i:])
                mult = int(m.group()) if m else 1
                i += len(m.group()) if m else 0
                for k, v in inner.items():
                    out[k] = out.get(k, 0) + v * mult
            elif s[i] == ")":
                return out, i + 1
            else:
                m = _re.match(r"([A-Z][a-z]?)(\d*)", s[i:])
                if not m or m.group(1) not in MASS:
                    raise ToolError(f"cannot read the formula near '{s[i:i + 4]}'")
                out[m.group(1)] = out.get(m.group(1), 0) + (int(m.group(2)) if m.group(2) else 1)
                i += len(m.group())
        return out, i

    return parse(formula.replace(" ", ""))[0]


def _molar(formula):
    return sum(MASS[k] * v for k, v in _count(formula).items())


def _balanced(eq):
    import re as _re
    left, _, right = eq.replace("→", "->").replace("=", "->").partition("->")

    def side(s):
        tot: dict[str, float] = {}
        for term in s.split("+"):
            term = term.strip()
            m = _re.match(r"(\d*)\s*(.+)", term)
            k = int(m.group(1)) if m.group(1) else 1
            for el, v in _count(m.group(2)).items():
                tot[el] = tot.get(el, 0) + k * v
        return tot

    l, r = side(left), side(right)
    return {"balanced": l == r, "left": l, "right": r}


def _dil(m1, v1, m2, v2):
    vals = [m1, v1, m2, v2]
    if vals.count(None) != 1:
        raise ToolError("give exactly three of m1, v1, m2, v2")
    i = vals.index(None)
    a, b, c = [v for v in vals if v is not None]
    vals[i] = {0: lambda: v2 * m2 / v1, 1: lambda: m2 * v2 / m1, 2: lambda: m1 * v1 / v2, 3: lambda: m1 * v1 / m2}[i]()
    return [fmt(round(v, 10)) for v in vals]


def _gas(p, v, n, t):
    R = 8.314462618
    vals = [p, v, n, t]
    if vals.count(None) != 1:
        raise ToolError("leave exactly one of p_kpa, v_litres, n_mol, t_kelvin empty")
    if p is None:
        return {"p_kpa": n * R * t / v}
    if v is None:
        return {"v_litres": n * R * t / p}
    if n is None:
        return {"n_mol": p * v / (R * t)}
    return {"t_kelvin": p * v / (n * R)}


def _proj(v, ang, h, g):
    a = math.radians(ang)
    vy, vx = v * math.sin(a), v * math.cos(a)
    t = (vy + math.sqrt(vy * vy + 2 * g * h)) / g
    return {"flight_time_s": fmt(round(t, 6)), "range_m": fmt(round(vx * t, 6)), "max_height_m": fmt(round(h + vy * vy / (2 * g), 6))}


def _ohm(v, i, r):
    if [v, i, r].count(None) != 1:
        raise ToolError("leave exactly one of v, i, r empty")
    return {"v": fmt(i * r)} if v is None else {"i": fmt(v / r)} if i is None else {"r": fmt(v / i)}


def _pmt(p, r, n):
    return p / n if r == 0 else p * r / (1 - (1 + r) ** -n)


def _sched(p, r, n, rows):
    pay, bal, out = _pmt(p, r, n), p, []
    for k in range(1, min(rows, n) + 1):
        interest = bal * r
        bal -= pay - interest
        out.append({"month": k, "payment": round(pay, 2), "interest": round(interest, 2), "principal": round(pay - interest, 2), "balance": round(max(bal, 0), 2)})
    return out


def _irr(cf):
    lo, hi = -0.99, 10.0
    f = lambda r: sum(c / (1 + r) ** i for i, c in enumerate(cf))  # noqa: E731
    if f(lo) * f(hi) > 0:
        raise ToolError("no IRR found (cash flows must change sign)")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
    return (lo + hi) / 2


def _bs(s, k, t, r, sig):
    nd = st.NormalDist().cdf
    d1 = (math.log(s / k) + (r + sig * sig / 2) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    return round(s * nd(d1) - k * math.exp(-r * t) * nd(d2), 4)
