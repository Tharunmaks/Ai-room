"""Statistics, probability, geometry, linear algebra, calculus extras, number theory, discrete maths."""

from __future__ import annotations

import math
import statistics as st
from fractions import Fraction

from .gen_math import MAT_NOTE, mat, sub
from .generators import Q, family, fmt, scale, yesno


def _data(rng, n, lo=1, hi=20):
    return [rng.randint(lo, hi) for _ in range(n)]


# ---------------------------------------------------------------- statistics

@family("stats.descriptive", "Mean, median, mode and spread", {"statistics": (1, 5), "math": (6, 8), "data_science": (1, 3)},
        "Mean = sum ÷ count. Median = middle value of the sorted data (average the two middle values if the count is even). Mode = most frequent value. "
        "Range = max − min. Population standard deviation = √(mean of squared deviations from the mean).",
        "Forgetting to sort before finding the median.")
def descriptive(rng, t):
    n = rng.randint(5, scale(t, 6, 9))
    d = _data(rng, n)
    if t > .5 and rng.random() < .4:
        sd = st.pstdev(d)
        return Q("numeric", f"What is the population standard deviation of {d}? (2 decimal places)", f"{sd:.2f}", f"mean = {st.mean(d):.2f}; sd = {sd:.2f}")
    kind = rng.choice(["mean", "median", "range"] + (["mode"] if n >= 7 else []))
    if kind == "mean":
        return Q("numeric", f"What is the mean of {d}? (2 decimal places)", f"{sum(d) / n:.2f}", f"{sum(d)} ÷ {n}")
    if kind == "median":
        return Q("numeric", f"What is the median of {d}?", fmt(st.median(d)), f"sorted: {sorted(d)}")
    if kind == "range":
        return Q("numeric", f"What is the range of {d}?", max(d) - min(d), f"{max(d)} − {min(d)}")
    d = d + [d[0], d[0]]
    return Q("numeric", f"What is the mode of {d}? (If several values tie, give the smallest.)", min(st.multimode(d)), "most frequent value")


@family("stats.zscore", "z-scores and the normal distribution", {"statistics": (3, 6)},
        "z = (x − μ)/σ says how many standard deviations x is from the mean. About 68% / 95% / 99.7% of a normal distribution lies within 1 / 2 / 3 standard deviations.",
        "Dividing by the variance instead of the standard deviation.")
def zscore(rng, t):
    mu, sigma = rng.randint(50, 100), rng.randint(2, 15)
    z = rng.randint(-3, 3)
    x = mu + z * sigma
    if rng.random() < .6:
        return Q("numeric", f"Scores have mean {mu} and standard deviation {sigma}. What is the z-score of {x}?", z, f"({x} − {mu})/{sigma} = {z}")
    return Q("numeric", f"Scores are normal with mean {mu} and standard deviation {sigma}. What score has z = {z}?", x, f"{mu} + {z}×{sigma}")


@family("stats.inference", "Confidence intervals and tests", {"statistics": (6, 12), "data_science": (6, 12), "research": (6, 12)},
        "Standard error = s/√n. A 95% confidence interval is x̄ ± 1.96·SE (large samples). Test statistic for a mean: t = (x̄ − μ₀)/(s/√n). "
        "χ² = Σ (observed − expected)²/expected.",
        "Using the standard deviation instead of the standard error.")
def inference(rng, t):
    k = rng.random()
    n, s = rng.choice([16, 25, 36, 49, 64, 100]), rng.randint(4, 20)
    if k < .4:
        return Q("numeric", f"A sample of n = {n} has standard deviation {s}. What is the margin of error of a 95% confidence interval for the mean (use 1.96)? (2 decimals)",
                 f"{1.96 * s / math.sqrt(n):.2f}", f"1.96 × {s}/√{n}")
    if k < .75:
        mu0, diff = rng.randint(20, 80), rng.choice([-6, -3, 3, 6, 9])
        return Q("numeric", f"A sample of n = {n} has mean {mu0 + diff} and standard deviation {s}. What is the test statistic t for H₀: μ = {mu0}? (2 decimals)",
                 f"{diff / (s / math.sqrt(n)):.2f}", f"({mu0 + diff} − {mu0}) / ({s}/√{n})")
    o = [rng.randint(8, 30) for _ in range(4)]
    e = sum(o) / 4
    chi = sum((x - e) ** 2 / e for x in o)
    return Q("numeric", f"Observed counts {o} across 4 equally likely categories. What is the χ² statistic? (2 decimals)", f"{chi:.2f}", f"expected {e} each")


@family("stats.regression", "Regression and correlation", {"statistics": (7, 12), "data_science": (4, 12), "machine_learning": (1, 3), "research": (7, 12)},
        "Least-squares line y = a + bx: slope b = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)², intercept a = ȳ − b·x̄. Correlation r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² Σ(y−ȳ)²), between −1 and 1.",
        "Reading correlation as causation.")
def regression(rng, t):
    n = 5
    xs = sorted(rng.sample(range(1, 15), n))
    ys = [rng.randint(1, 20) for _ in range(n)]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    pts = ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))
    if rng.random() < .5:
        return Q("numeric", f"Find the least-squares slope for the points {pts}. (3 decimals)", f"{sxy / sxx:.3f}", f"Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² = {sxy / sxx:.3f}")
    if syy == 0:
        ys[0] += 1
        return regression(rng, t)
    return Q("numeric", f"Find the correlation coefficient r for the points {pts}. (3 decimals)", f"{sxy / math.sqrt(sxx * syy):.3f}", "Pearson r")


# ---------------------------------------------------------------- probability

@family("prob.basic", "Basic probability", {"probability": (1, 4), "math": (5, 7)},
        "Probability = favourable outcomes ÷ total equally-likely outcomes, between 0 and 1. For independent events multiply; for draws without replacement the totals shrink.",
        "Forgetting that the total changes when drawing without replacement.")
def prob_basic(rng, t):
    r, b = rng.randint(2, 9), rng.randint(2, 9)
    k = rng.random()
    if k < .4:
        return Q("numeric", f"A bag has {r} red and {b} blue marbles. What is the probability of drawing a red one? (fraction)", Fraction(r, r + b), f"{r}/{r + b}")
    if k < .7:
        return Q("numeric", f"A bag has {r} red and {b} blue marbles. Two are drawn without replacement. What is the probability both are red? (fraction)",
                 Fraction(r, r + b) * Fraction(r - 1, r + b - 1), f"{r}/{r + b} × {r - 1}/{r + b - 1}")
    s = rng.randint(2, 12)
    ways = sum(1 for a in range(1, 7) for c in range(1, 7) if a + c == s)
    return Q("numeric", f"Two fair dice are rolled. What is the probability their sum is {s}? (fraction)", Fraction(ways, 36), f"{ways} ways out of 36")


@family("prob.conditional", "Conditional probability and Bayes' rule", {"probability": (4, 7), "logic": (8, 10), "ai_safety": (4, 12), "public_health": (4, 12), "medicine": (4, 12)},
        "P(A|B) = P(A∩B)/P(B). Bayes: P(A|B) = P(B|A)P(A)/P(B), where P(B) = P(B|A)P(A) + P(B|not A)P(not A).",
        "Confusing P(A|B) with P(B|A) (the base-rate fallacy).")
def conditional(rng, t):
    prior = rng.choice([0.01, 0.02, 0.05, 0.1])
    sens, spec = rng.choice([0.9, 0.95, 0.99]), rng.choice([0.9, 0.95])
    pos = sens * prior + (1 - spec) * (1 - prior)
    post = sens * prior / pos
    return Q("numeric", f"A condition affects {prior:.0%} of people. A test detects it {sens:.0%} of the time when present and wrongly flags {1 - spec:.0%} of healthy people. "
             f"If someone tests positive, what is the probability they have the condition? (as a percentage with 1 decimal, number only)", f"{post * 100:.1f}",
             f"P(pos) = {pos:.4f}; P(cond|pos) = {sens}×{prior}/{pos:.4f}")


@family("prob.distributions", "Random variables and distributions", {"probability": (3, 10)},
        "Expected value E[X] = Σ x·P(x). Binomial(n, p): P(X=k) = C(n,k)pᵏ(1−p)ⁿ⁻ᵏ, mean np, variance np(1−p). Poisson(λ): P(X=k) = e^(−λ)λᵏ/k!. Geometric(p): expected trials 1/p.",
        "Using the binomial when trials are not independent.")
def distributions(rng, t):
    k = rng.random()
    if k < .3:
        vals = [rng.randint(0, 10) for _ in range(3)]
        ps = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]
        return Q("numeric", f"X takes values {vals[0]}, {vals[1]}, {vals[2]} with probabilities 1/2, 1/3, 1/6. What is E[X]? (fraction or whole number)",
                 sum(v * p for v, p in zip(vals, ps)), "Σ x·P(x)")
    if k < .65:
        n, p, j = rng.randint(4, 10), rng.choice([0.2, 0.3, 0.5, 0.6]), 0
        j = rng.randint(1, n - 1)
        v = math.comb(n, j) * p ** j * (1 - p) ** (n - j)
        return Q("numeric", f"X ~ Binomial(n = {n}, p = {p}). What is P(X = {j})? (4 decimals)", f"{v:.4f}", f"C({n},{j})·{p}^{j}·{1 - p:.1f}^{n - j}")
    if k < .85:
        lam, j = rng.choice([1, 2, 3, 4]), rng.randint(0, 4)
        v = math.exp(-lam) * lam ** j / math.factorial(j)
        return Q("numeric", f"X ~ Poisson(λ = {lam}). What is P(X = {j})? (4 decimals)", f"{v:.4f}", "e^(−λ)λ^k/k!")
    p = rng.choice([Fraction(1, 4), Fraction(1, 5), Fraction(1, 6), Fraction(1, 10)])
    return Q("numeric", f"Each trial succeeds with probability {p}. What is the expected number of trials until the first success?", 1 / p, "1/p")


@family("prob.markov", "Markov chains", {"probability": (10, 12)},
        "A Markov chain moves between states with fixed probabilities. For two states A→B with probability p and B→A with probability q, the long-run share in A is q/(p+q).",
        "Using the starting state instead of the stationary distribution.")
def markov(rng, t):
    p, q = Fraction(rng.randint(1, 5), 10), Fraction(rng.randint(1, 5), 10)
    return Q("numeric", f"A two-state Markov chain moves A→B with probability {p} and B→A with probability {q} each step. What fraction of time does it spend in A in the long run? (fraction)",
             q / (p + q), "stationary π_A = q/(p+q)")


# ---------------------------------------------------------------- geometry

@family("geo.area", "Area and perimeter", {"geometry": (1, 4), "math": (5, 7)},
        "Rectangle: A = l×w, P = 2(l+w). Triangle: A = ½bh. Parallelogram: A = bh. Trapezoid: A = ½(a+b)h.",
        "Using the slanted side instead of the perpendicular height.")
def area(rng, t):
    a, b, h = rng.randint(2, scale(t, 10, 30)), rng.randint(2, scale(t, 10, 30)), 2 * rng.randint(1, scale(t, 6, 15))
    k = rng.choice(["rect", "tri", "trap", "perim"])
    if k == "rect":
        return Q("numeric", f"A rectangle is {a} by {b}. What is its area?", a * b, f"{a}×{b}")
    if k == "perim":
        return Q("numeric", f"A rectangle is {a} by {b}. What is its perimeter?", 2 * (a + b), f"2({a}+{b})")
    if k == "tri":
        return Q("numeric", f"A triangle has base {a} and height {h}. What is its area?", a * h // 2, f"½×{a}×{h}")
    return Q("numeric", f"A trapezoid has parallel sides {a} and {b} and height {h}. What is its area?", (a + b) * h // 2, f"½({a}+{b})×{h}")


@family("geo.circle", "Circles", {"geometry": (3, 6), "math": (5, 7)},
        "Circumference C = 2πr; area A = πr². An arc of angle θ° has length (θ/360)·2πr and the matching sector has area (θ/360)·πr².",
        "Using the diameter where the formula needs the radius.")
def circle(rng, t):
    r = rng.randint(2, scale(t, 10, 25))
    k = rng.random()
    if k < .35:
        return Q("numeric", f"What is the area of a circle of radius {r}? (use π = 3.14159, 2 decimals)", f"{math.pi * r * r:.2f}", f"πr² = π×{r}²")
    if k < .7:
        return Q("numeric", f"What is the circumference of a circle of radius {r}? (use π = 3.14159, 2 decimals)", f"{2 * math.pi * r:.2f}", "2πr")
    ang = rng.choice([30, 45, 60, 90, 120, 180])
    return Q("numeric", f"A sector of angle {ang}° is cut from a circle of radius {r}. What is the arc length? (π = 3.14159, 2 decimals)",
             f"{ang / 360 * 2 * math.pi * r:.2f}", f"({ang}/360)·2π·{r}")


@family("geo.angles", "Angles and polygons", {"geometry": (1, 4)},
        "Angles in a triangle add to 180°; on a straight line to 180°; around a point to 360°. A polygon with n sides has interior angles summing to (n − 2)×180°.",
        "Using (n − 1) instead of (n − 2).")
def angles(rng, t):
    k = rng.random()
    if k < .4:
        a, b = rng.randint(20, 80), rng.randint(20, 80)
        return Q("numeric", f"Two angles of a triangle are {a}° and {b}°. What is the third angle?", 180 - a - b, f"180 − {a} − {b}")
    if k < .7:
        n = rng.randint(5, 12)
        return Q("numeric", f"What is the sum of the interior angles of a polygon with {n} sides (in degrees)?", (n - 2) * 180, f"({n}−2)×180")
    n = rng.choice([5, 6, 8, 9, 10, 12])
    return Q("numeric", f"What is each interior angle of a regular polygon with {n} sides (degrees)?", fmt((n - 2) * 180 / n), f"({n}−2)×180/{n}")


@family("geo.volume", "Volume and surface area", {"geometry": (4, 9)},
        "Box: V = lwh. Cylinder: V = πr²h. Cone: V = ⅓πr²h. Sphere: V = (4/3)πr³, surface area 4πr². Cube surface area 6s².",
        "Forgetting the ⅓ for cones.")
def volume(rng, t):
    r, h = rng.randint(2, 12), rng.randint(2, 15)
    k = rng.choice(["box", "cyl", "cone", "sph", "cube"] if t > .3 else ["box", "cube"])
    if k == "box":
        a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 12)
        return Q("numeric", f"What is the volume of a box {a} × {b} × {c}?", a * b * c, f"{a}×{b}×{c}")
    if k == "cube":
        return Q("numeric", f"What is the surface area of a cube with side {r}?", 6 * r * r, f"6×{r}²")
    v = {"cyl": math.pi * r * r * h, "cone": math.pi * r * r * h / 3, "sph": 4 / 3 * math.pi * r ** 3}[k]
    name = {"cyl": f"a cylinder with radius {r} and height {h}", "cone": f"a cone with radius {r} and height {h}", "sph": f"a sphere of radius {r}"}[k]
    return Q("numeric", f"What is the volume of {name}? (π = 3.14159, 1 decimal)", f"{v:.1f}", "formula from the card")


@family("geo.coordinate", "Coordinate geometry", {"geometry": (5, 8), "math": (5, 7)},
        "Distance between (x₁,y₁) and (x₂,y₂) is √((x₂−x₁)² + (y₂−y₁)²). Midpoint = average of the coordinates. Slope = (y₂−y₁)/(x₂−x₁).",
        "Swapping x and y differences in the slope.")
def coordinate(rng, t):
    x1, y1 = rng.randint(-8, 8), rng.randint(-8, 8)
    dx, dy = rng.choice([(3, 4), (5, 12), (6, 8), (8, 15)])
    x2, y2 = x1 + dx * rng.choice([1, -1]), y1 + dy * rng.choice([1, -1])
    k = rng.random()
    if k < .4:
        return Q("numeric", f"What is the distance between ({x1}, {y1}) and ({x2}, {y2})?", fmt(math.hypot(dx, dy)), "Pythagoras")
    if k < .7:
        return Q("numeric", f"What is the slope of the line through ({x1}, {y1}) and ({x2}, {y2})? (fraction or decimal)", Fraction(y2 - y1, x2 - x1), "rise/run")
    return Q("numeric", f"What is the x-coordinate of the midpoint of ({x1}, {y1}) and ({x2}, {y2})? (decimal allowed)", fmt((x1 + x2) / 2), "average of x's")


@family("geo.triangles", "Triangles: similarity, Heron, cosine rule", {"geometry": (4, 11)},
        "Similar triangles have equal angles and proportional sides. Heron: A = √(s(s−a)(s−b)(s−c)) with s = (a+b+c)/2. Cosine rule: c² = a² + b² − 2ab·cos C.",
        "Forgetting to halve the perimeter in Heron's formula.")
def triangles(rng, t):
    k = rng.random()
    if k < .35:
        a, b, c = rng.choice([(3, 4, 5), (5, 12, 13), (13, 14, 15), (7, 15, 20), (9, 10, 17)])
        s = (a + b + c) / 2
        return Q("numeric", f"What is the area of a triangle with sides {a}, {b}, {c}? (Heron's formula)", fmt(math.sqrt(s * (s - a) * (s - b) * (s - c))), f"s = {s}")
    if k < .65:
        a, b = rng.randint(3, 15), rng.randint(3, 15)
        c = math.sqrt(a * a + b * b - a * b)
        return Q("numeric", f"Two sides of a triangle are {a} and {b} with a 60° angle between them. How long is the third side? (2 decimals)", f"{c:.2f}", "c² = a² + b² − 2ab cos 60°")
    m = rng.randint(2, 4)
    a, b = rng.randint(3, 9), rng.randint(3, 9)
    return Q("numeric", f"Triangle ABC has two sides of length {a} and {b}. A similar triangle has the side matching {a} equal to {a * m}. How long is the side matching {b}?", b * m, f"scale factor {m}")


@family("geo.polygon", "Polygon area by coordinates (shoelace)", {"geometry": (9, 12)},
        "Shoelace formula: for vertices (x₁,y₁)…(xₙ,yₙ) in order, area = ½|Σ(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)|.",
        "Listing the vertices out of order.")
def polygon(rng, t):
    pts = [(0, 0), (rng.randint(3, 9), 0), (rng.randint(3, 9), rng.randint(3, 8)), (rng.randint(0, 2), rng.randint(4, 9))]
    s = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
    return Q("numeric", f"Find the area of the quadrilateral with vertices {pts} in order. (decimal allowed)", fmt(abs(s) / 2), "shoelace")


# ---------------------------------------------------------------- linear algebra extras

@family("la.inverse", "Matrix inverse", {"linear_algebra": (4, 7)},
        "For A = [a b; c d] with det = ad − bc ≠ 0, A⁻¹ = (1/det)·[d −b; −c a]. A·A⁻¹ = I.",
        "Forgetting to swap a and d and negate b and c.")
def inverse(rng, t):
    k, m = rng.randint(1, 4), rng.randint(1, 4)
    A = [[1 + k * m, k], [m, 1]]
    inv = [[1, -k], [-m, 1 + k * m]]
    i, j = rng.randint(0, 1), rng.randint(0, 1)
    return Q("numeric", f"A = {mat(A)} has determinant 1. What is entry (row {i + 1}, column {j + 1}) of A⁻¹? {MAT_NOTE}", inv[i][j], "swap diagonal, negate off-diagonal")


@family("la.rank", "Rank and linear systems", {"linear_algebra": (5, 9)},
        "The rank is the number of linearly independent rows (= columns). If a row is a combination of the others it adds nothing. A square system has a unique solution when det ≠ 0 (rank = n).",
        "Counting rows instead of independent rows.")
def rank(rng, t):
    r1 = [rng.randint(-3, 4) for _ in range(3)]
    r2 = [rng.randint(-3, 4) for _ in range(3)]
    if rng.random() < .5:
        r3 = [a + b for a, b in zip(r1, r2)]
    else:
        r3 = [rng.randint(-3, 4) for _ in range(3)]
    M = [[Fraction(x) for x in r] for r in (r1, r2, r3)]
    rk, row = 0, 0
    for col in range(3):
        piv = next((i for i in range(row, 3) if M[i][col] != 0), None)
        if piv is None:
            continue
        M[row], M[piv] = M[piv], M[row]
        for i in range(row + 1, 3):
            f = M[i][col] / M[row][col]
            M[i] = [a - f * b for a, b in zip(M[i], M[row])]
        row += 1
        rk += 1
    return Q("numeric", f"What is the rank of the matrix {mat([r1, r2, r3])}? {MAT_NOTE}", rk, "Gaussian elimination")


@family("la.geometry", "Projections and cross products", {"linear_algebra": (6, 12)},
        "Scalar projection of v onto u is (v·u)/|u|. Vector projection = ((v·u)/(u·u))·u. Cross product u × v = (u₂v₃ − u₃v₂, u₃v₁ − u₁v₃, u₁v₂ − u₂v₁), perpendicular to both.",
        "Using |u| instead of u·u in the vector projection.")
def la_geometry(rng, t):
    u = [rng.randint(-4, 5) for _ in range(3)]
    while not any(u):
        u = [rng.randint(-4, 5) for _ in range(3)]
    v = [rng.randint(-4, 5) for _ in range(3)]
    if rng.random() < .5:
        i = rng.randint(0, 2)
        c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        return Q("numeric", f"What is component {i + 1} of the cross product of ({', '.join(map(str, u))}) and ({', '.join(map(str, v))})?", c[i], "cross product formula")
    uu = sum(a * a for a in u)
    return Q("numeric", f"What is (v·u)/(u·u) for u = ({', '.join(map(str, u))}) and v = ({', '.join(map(str, v))})? (fraction or decimal)",
             Fraction(sum(a * b for a, b in zip(u, v)), uu), "projection coefficient")


@family("la.solve3", "3×3 linear systems", {"linear_algebra": (3, 7)},
        "Use elimination: remove one variable from two equations to get a 2×2 system, solve it, then back-substitute.",
        "Arithmetic slips while eliminating — check by plugging back in.")
def solve3(rng, t):
    while True:
        xs = [rng.randint(-4, 6) for _ in range(3)]
        A = [[rng.choice([-3, -2, -1, 1, 2, 3, 4]) for _ in range(3)] for _ in range(3)]
        d = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
             + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
        if d != 0:
            break
    b = [sum(a * x for a, x in zip(row, xs)) for row in A]
    eqs = "; ".join(f"{r[0]}x + {r[1]}y + {r[2]}z = {bb}".replace("+ -", "− ") for r, bb in zip(A, b))
    idx = rng.randint(0, 2)
    return Q("numeric", f"Solve {eqs}. What is {'xyz'[idx]}?", xs[idx], f"solution (x, y, z) = {tuple(xs)}")


# ---------------------------------------------------------------- calculus extras

@family("calc.multivariable", "Partial derivatives and double integrals", {"calculus": (10, 12)},
        "Partial derivative ∂f/∂x treats y as a constant (and vice versa). A double integral over a rectangle can be done one variable at a time: ∫∫(x + y) dy dx.",
        "Differentiating with respect to both variables at once.")
def multivariable(rng, t):
    a, b = rng.randint(1, 4), rng.randint(1, 4)
    x0, y0 = rng.randint(-2, 3), rng.randint(-2, 3)
    if rng.random() < .6:
        return Q("numeric", f"If f(x, y) = {a}x²y + {b}xy², what is ∂f/∂x at ({x0}, {y0})?", 2 * a * x0 * y0 + b * y0 * y0, f"∂f/∂x = {2 * a}xy + {b}y²")
    p, q = rng.randint(1, 4), rng.randint(1, 4)
    return Q("numeric", f"Evaluate the double integral of (x + y) over the rectangle 0 ≤ x ≤ {p}, 0 ≤ y ≤ {q}. (fraction or whole number)",
             Fraction(p * p * q, 2) + Fraction(q * q * p, 2), "∫₀^q∫₀^p (x+y) dx dy")


# ---------------------------------------------------------------- number theory

@family("nt.divisibility", "Divisibility rules", {"number_theory": (1, 3), "math": (3, 5)},
        "By 2: last digit even. By 3: digit sum divisible by 3. By 5: ends in 0 or 5. By 9: digit sum divisible by 9. By 11: alternating digit sum divisible by 11.",
        "Applying the rule for 3 to the number's last digit.")
def divisibility(rng, t):
    d = rng.choice([2, 3, 5, 9, 11])
    n = rng.randint(100, scale(t, 999, 99999))
    return yesno(rng, f"Is {n} divisible by {d}?", n % d == 0, f"{n} mod {d} = {n % d}")


@family("nt.congruence", "Linear congruences", {"number_theory": (5, 8)},
        "To solve ax ≡ b (mod m) when gcd(a, m) = 1, multiply both sides by the inverse of a mod m: x ≡ a⁻¹b.",
        "Dividing both sides by a as if it were ordinary arithmetic.")
def congruence(rng, t):
    m = rng.choice([7, 9, 11, 13, 15, 17])
    a = rng.choice([x for x in range(2, m) if math.gcd(x, m) == 1])
    b = rng.randint(1, m - 1)
    return Q("numeric", f"Find the smallest non-negative x with {a}x ≡ {b} (mod {m}).", pow(a, -1, m) * b % m, f"x = {a}⁻¹·{b} mod {m}")


@family("nt.totient", "Euler's totient and Fermat's little theorem", {"number_theory": (6, 9)},
        "φ(n) counts integers in 1..n coprime to n. For prime p, φ(p) = p − 1. φ is multiplicative: φ(pq) = (p−1)(q−1). Fermat: a^(p−1) ≡ 1 (mod p).",
        "Forgetting φ(pᵏ) = pᵏ − pᵏ⁻¹.")
def totient(rng, t):
    n = rng.randint(10, scale(t, 40, 200))
    phi = sum(1 for i in range(1, n + 1) if math.gcd(i, n) == 1)
    if rng.random() < .6:
        return Q("numeric", f"What is φ({n}), Euler's totient?", phi, "count of numbers coprime to n")
    p = rng.choice([7, 11, 13, 17])
    a = rng.randint(2, p - 1)
    return Q("numeric", f"What is {a}^{p - 1 + p * rng.randint(1, 4) * (p - 1)} mod {p}? (Fermat's little theorem)", 1, f"a^(p−1) ≡ 1, so any multiple of p − 1 in the exponent gives 1")


@family("nt.crt", "Chinese remainder theorem and Diophantine equations", {"number_theory": (8, 12)},
        "If moduli are coprime, x ≡ a (mod m) and x ≡ b (mod n) have a unique solution mod mn. For ax + by = c, integer solutions exist iff gcd(a, b) divides c.",
        "Using the theorem when the moduli share a factor.")
def crt(rng, t):
    if rng.random() < .6:
        m, n = rng.choice([(3, 5), (5, 7), (4, 9), (7, 8), (5, 11)])
        x = rng.randint(1, m * n - 1)
        return Q("numeric", f"Find the smallest positive x with x ≡ {x % m} (mod {m}) and x ≡ {x % n} (mod {n}).", x if x % (m * n) else m * n, "Chinese remainder theorem")
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    c = rng.randint(30, 80)
    cnt = sum(1 for x in range(1, c) for y in range(1, c) if a * x + b * y == c)
    return Q("numeric", f"How many positive integer solutions (x, y) does {a}x + {b}y = {c} have?", cnt, "enumerate x")


@family("nt.residues", "Quadratic residues", {"number_theory": (9, 12)},
        "a is a quadratic residue mod p if x² ≡ a (mod p) has a solution. Euler's criterion: a^((p−1)/2) ≡ 1 (mod p) for residues, ≡ −1 for non-residues.",
        "Checking only a few x values.")
def residues(rng, t):
    p = rng.choice([7, 11, 13, 17, 19, 23])
    a = rng.randint(2, p - 1)
    return yesno(rng, f"Is {a} a quadratic residue modulo {p}?", any(x * x % p == a for x in range(1, p)), f"squares mod {p}: {sorted({x * x % p for x in range(1, p)})}")


# ---------------------------------------------------------------- discrete maths

@family("dm.sets", "Sets and counting principles", {"discrete_math": (1, 4), "logic": (3, 6)},
        "Inclusion–exclusion: |A ∪ B| = |A| + |B| − |A ∩ B|. Pigeonhole: putting more than n·k items into n boxes forces some box to hold more than k. Number of subsets of an n-element set: 2ⁿ.",
        "Double-counting the overlap.")
def sets(rng, t):
    a, b, c = rng.randint(10, 40), rng.randint(10, 40), rng.randint(2, 9)
    k = rng.random()
    if k < .4:
        return Q("numeric", f"In a class, {a} students play chess, {b} play football and {c} play both. How many play at least one?", a + b - c, f"{a} + {b} − {c}")
    if k < .7:
        n, kk = rng.randint(3, 9), rng.randint(2, 5)
        return Q("numeric", f"How many items guarantee that some box among {n} holds at least {kk + 1}? (minimum)", n * kk + 1, "pigeonhole: n·k + 1")
    n = rng.randint(3, 12)
    return Q("numeric", f"How many subsets does a set with {n} elements have?", 2 ** n, "2ⁿ")


@family("dm.recurrence", "Recurrences and induction", {"discrete_math": (5, 9)},
        "A recurrence defines each term from earlier ones: aₙ = p·aₙ₋₁ + q·aₙ₋₂. Induction proves a statement for all n: show the base case, then show 'true for n' implies 'true for n+1'. "
        "Sum of the first n odd numbers = n²; sum of first n cubes = (n(n+1)/2)².",
        "Skipping the base case.")
def recurrence(rng, t):
    if rng.random() < .6:
        p, q, a0, a1 = rng.randint(1, 3), rng.randint(0, 2), rng.randint(0, 3), rng.randint(1, 4)
        n = rng.randint(5, 9)
        seq = [a0, a1]
        for _ in range(n - 1):
            seq.append(p * seq[-1] + q * seq[-2])
        parts = [f"{'' if p == 1 else p}aₙ₋₁"] + ([f"{'' if q == 1 else q}aₙ₋₂"] if q else [])
        return Q("numeric", f"a₀ = {a0}, a₁ = {a1}, aₙ = {' + '.join(parts)}. What is a{sub(n)}?", seq[n], f"sequence: {seq[:n + 1]}")
    n = rng.randint(4, 12)
    return Q("numeric", f"What is 1³ + 2³ + … + {n}³?", (n * (n + 1) // 2) ** 2, "(n(n+1)/2)²")


@family("dm.graphs", "Graph theory", {"discrete_math": (4, 10)},
        "Handshake lemma: the degrees in a graph add up to twice the number of edges. K_n (complete graph) has n(n−1)/2 edges; a tree on n vertices has n − 1; K_{m,n} has mn.",
        "Forgetting each edge contributes to two vertices' degrees.")
def graphs(rng, t):
    k = rng.random()
    n = rng.randint(4, 15)
    if k < .3:
        return Q("numeric", f"How many edges does the complete graph K{sub(n)} have?", n * (n - 1) // 2, "n(n−1)/2")
    if k < .55:
        return Q("numeric", f"How many edges does a tree with {n} vertices have?", n - 1, "n − 1")
    if k < .8:
        deg = [rng.randint(1, 5) for _ in range(n)]
        if sum(deg) % 2:
            deg[0] += 1
        return Q("numeric", f"A graph has {n} vertices with degrees {deg}. How many edges does it have?", sum(deg) // 2, "sum of degrees / 2")
    return Q("numeric", f"What is the maximum number of edges in a triangle-free graph on {n * 2} vertices? (Turán/Mantel)", (2 * n) ** 2 // 4, "⌊n²/4⌋")


@family("dm.generating", "Generating functions and special numbers", {"discrete_math": (8, 12)},
        "The coefficient of xᵏ in (1 + x)ⁿ is C(n,k). In 1/(1−x)ᵐ it is C(k+m−1, k). Derangements Dₙ = n!·Σ(−1)ⁱ/i!; Catalan Cₙ = C(2n,n)/(n+1).",
        "Off-by-one in the binomial for 1/(1−x)ᵐ.")
def generating(rng, t):
    k = rng.random()
    if k < .35:
        n, j = rng.randint(5, 12), rng.randint(2, 4)
        return Q("numeric", f"What is the coefficient of x^{j} in (1 + x)^{n}?", math.comb(n, j), f"C({n},{j})")
    if k < .65:
        m, j = rng.randint(2, 4), rng.randint(2, 6)
        return Q("numeric", f"What is the coefficient of x^{j} in 1/(1 − x)^{m}?", math.comb(j + m - 1, j), f"C({j}+{m}−1,{j})")
    if k < .85:
        n = rng.randint(3, 8)
        d = round(math.factorial(n) * sum((-1) ** i / math.factorial(i) for i in range(n + 1)))
        return Q("numeric", f"How many derangements (permutations with no fixed point) of {n} objects are there?", d, "inclusion–exclusion")
    n = rng.randint(3, 9)
    return Q("numeric", f"What is the Catalan number C{sub(n)}?", math.comb(2 * n, n) // (n + 1), "C(2n,n)/(n+1)")
