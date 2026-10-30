"""Finance, accounting, economics and business-metrics families."""

from __future__ import annotations

import math

from .generators import Q, family, fmt, scale

SIG = "(3 significant figures)"


# ===================================================================== FINANCE

@family("fin.interest", "Simple and compound interest", {"finance": (1, 5), "accounting": (1, 3), "economics": (1, 3), "business": (1, 3)},
        "Simple interest I = P·r·t. Compound interest A = P(1 + r/n)^(nt). The rule of 72: money doubles in about 72 ÷ (interest rate in %) years.",
        "Entering the rate as 5 instead of 0.05.")
def interest(rng, t):
    p, r, yrs = rng.choice([500, 1000, 2000, 5000]), rng.choice([2, 4, 5, 8, 10]), rng.randint(2, 10)
    k = rng.random()
    if k < .4:
        return Q("numeric", f"What simple interest does ${p} earn at {r}% per year over {yrs} years ($)?", fmt(p * r / 100 * yrs), "I = Prt")
    if k < .8:
        return Q("numeric", f"What is ${p} worth after {yrs} years at {r}% interest compounded annually ($)? {SIG}", fmt(p * (1 + r / 100) ** yrs), "A = P(1+r)^t")
    return Q("numeric", f"Using the rule of 72, about how many years does money take to double at {r}% per year? (1 decimal)", f"{72 / r:.1f}", "72/r")


@family("fin.timevalue", "Time value of money and loans", {"finance": (4, 8)},
        "Present value PV = FV/(1 + r)ⁿ. Annuity future value FV = C·((1+r)ⁿ − 1)/r; present value PV = C·(1 − (1+r)⁻ⁿ)/r. Loan payment = P·r/(1 − (1+r)⁻ⁿ). A perpetuity pays C forever: PV = C/r.",
        "Using the annual rate with monthly periods.")
def timevalue(rng, t):
    k = rng.random()
    r, n = rng.choice([0.04, 0.05, 0.08, 0.1]), rng.randint(3, 12)
    if k < .3:
        fv = rng.choice([1000, 5000, 10000])
        return Q("numeric", f"What is the present value of ${fv} received in {n} years at {r:.0%} per year ($)? {SIG}", fmt(fv / (1 + r) ** n), "PV = FV/(1+r)ⁿ")
    if k < .55:
        c = rng.choice([100, 500, 1000])
        return Q("numeric", f"What is the future value of ${c} deposited at the end of each year for {n} years at {r:.0%} ($)? {SIG}", fmt(c * ((1 + r) ** n - 1) / r), "annuity FV")
    if k < .8:
        p, rm, m = rng.choice([10000, 20000, 200000]), rng.choice([0.06, 0.09, 0.12]) / 12, rng.choice([12, 24, 36, 60])
        return Q("numeric", f"A ${p} loan has a monthly interest rate of {rm * 100:.2f}% and {m} equal monthly payments. What is each payment ($)? {SIG}", fmt(p * rm / (1 - (1 + rm) ** -m)), "P·r/(1−(1+r)⁻ⁿ)")
    c = rng.choice([50, 100, 250])
    return Q("numeric", f"A perpetuity pays ${c} a year forever. At a {r:.0%} discount rate, what is its present value ($)?", fmt(c / r), "C/r")


@family("fin.valuation", "Valuation: NPV, bonds and stocks", {"finance": (6, 10)},
        "NPV = Σ CFₜ/(1+r)ᵗ (invest if NPV > 0). Bond price = Σ coupons discounted + face value discounted. Gordon growth stock price P = D₁/(r − g). P/E = price ÷ earnings per share.",
        "Forgetting to discount the face value of a bond.")
def valuation(rng, t):
    k = rng.random()
    r = rng.choice([0.05, 0.08, 0.1])
    if k < .4:
        i0, c1, c2 = rng.choice([100, 500, 1000]), rng.choice([50, 200, 600]), rng.choice([60, 300, 700])
        return Q("numeric", f"A project costs ${i0} now and returns ${c1} in year 1 and ${c2} in year 2. At {r:.0%}, what is the NPV ($)? {SIG}", fmt(-i0 + c1 / (1 + r) + c2 / (1 + r) ** 2), "NPV")
    if k < .7:
        face, cpn, n = 1000, rng.choice([40, 50, 60]), rng.randint(3, 8)
        price = sum(cpn / (1 + r) ** i for i in range(1, n + 1)) + face / (1 + r) ** n
        return Q("numeric", f"A bond pays ${cpn} a year for {n} years plus ${face} at maturity. At a {r:.0%} yield, what is its price ($)? {SIG}", fmt(price), "discount every cash flow")
    d, g = rng.choice([2, 3, 5]), rng.choice([0.02, 0.03])
    r = max(r, g + 0.03)
    return Q("numeric", f"A stock's next dividend is ${d}, growing {g:.0%} a year forever. With a required return of {r:.0%}, what is its price ($)? {SIG}", fmt(d / (r - g)), "D₁/(r−g)")


@family("fin.risk", "Risk and return", {"finance": (7, 11), "data_science": (6, 8)},
        "Portfolio expected return = Σ wᵢrᵢ. CAPM: E[r] = r_f + β(E[r_m] − r_f). Sharpe ratio = (return − r_f)/σ. Two-asset variance = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂.",
        "Using the market return instead of the market risk premium.")
def risk(rng, t):
    k = rng.random()
    if k < .35:
        rf, b, rm = rng.choice([0.02, 0.03]), rng.choice([0.8, 1.2, 1.5]), rng.choice([0.08, 0.1])
        return Q("numeric", f"r_f = {rf:.0%}, β = {b}, expected market return {rm:.0%}. What is the CAPM expected return (%)? {SIG}", fmt((rf + b * (rm - rf)) * 100), "r_f + β(r_m − r_f)")
    if k < .65:
        w, r1, r2 = rng.choice([0.3, 0.4, 0.6]), rng.choice([0.06, 0.1]), rng.choice([0.03, 0.05])
        return Q("numeric", f"A portfolio holds {w:.0%} in an asset returning {r1:.0%} and the rest in one returning {r2:.0%}. What is its expected return (%)? {SIG}", fmt((w * r1 + (1 - w) * r2) * 100), "weighted average")
    r_, rf, s = rng.choice([0.10, 0.12, 0.15]), 0.03, rng.choice([0.1, 0.15, 0.2])
    return Q("numeric", f"A fund returns {r_:.0%} with volatility {s:.0%}; the risk-free rate is 3%. What is its Sharpe ratio? {SIG}", fmt((r_ - rf) / s), "(r − r_f)/σ")


@family("fin.derivatives", "Options and forwards", {"finance": (9, 12)},
        "Call payoff max(S − K, 0); put payoff max(K − S, 0). Forward price F = S·e^(rT). Put–call parity: C − P = S − K·e^(−rT). One-step binomial: price = e^(−rT)[q·Cᵤ + (1−q)·C_d] with q = (e^(rT) − d)/(u − d).",
        "Forgetting that an option payoff cannot be negative.")
def derivatives(rng, t):
    k = rng.random()
    s, strike = rng.choice([80, 100, 120]), rng.choice([90, 100, 110])
    if k < .3:
        kind = rng.choice(["call", "put"])
        return Q("numeric", f"A {kind} option has strike ${strike}. The stock ends at ${s}. What is the payoff ($)?", max(s - strike, 0) if kind == "call" else max(strike - s, 0), "max(...,0)")
    if k < .6:
        r, tt = rng.choice([0.03, 0.05]), rng.choice([0.5, 1, 2])
        return Q("numeric", f"A stock trades at ${s} and the continuously compounded rate is {r:.0%}. What is the {tt}-year forward price ($)? {SIG}", fmt(s * math.exp(r * tt)), "S·e^(rT)")
    u, d, r = 1.2, 0.8, 0.05
    q = (math.exp(r) - d) / (u - d)
    cu, cd = max(100 * u - strike, 0), max(100 * d - strike, 0)
    return Q("numeric", f"S = $100 can go to $120 or $80 in one year (r = 5% continuous). What is a 1-year call with strike ${strike} worth ($)? {SIG}", fmt(math.exp(-r) * (q * cu + (1 - q) * cd)), "risk-neutral pricing")


# ===================================================================== ACCOUNTING

@family("acct.basics", "The accounting equation", {"accounting": (1, 4), "business": (1, 3)},
        "Assets = Liabilities + Owner's equity. Net income = revenues − expenses. Every transaction keeps the equation balanced (double-entry bookkeeping).",
        "Counting a loan received as revenue.")
def acct_basics(rng, t):
    a, l = rng.randint(50, 500) * 10, rng.randint(10, 200) * 10
    if rng.random() < .5:
        return Q("numeric", f"A firm has assets of ${a + l} and liabilities of ${l}. What is the owner's equity ($)?", a, "A = L + E")
    rev, ex = rng.randint(50, 300) * 100, rng.randint(10, 200) * 100
    return Q("numeric", f"Revenues were ${rev} and expenses ${ex}. What is net income ($)?", rev - ex, "revenue − expenses")


@family("acct.depreciation", "Depreciation", {"accounting": (3, 7)},
        "Straight-line depreciation per year = (cost − salvage value) ÷ useful life. Double-declining balance: rate = 2 ÷ life applied to the opening book value each year.",
        "Subtracting salvage value in the double-declining method (it is not subtracted until the end).")
def depreciation(rng, t):
    cost, salv, life = rng.choice([10000, 25000, 60000]), rng.choice([0, 1000, 5000]), rng.choice([4, 5, 8, 10])
    if rng.random() < .6:
        return Q("numeric", f"An asset costs ${cost}, has a salvage value of ${salv} and a {life}-year life. What is the annual straight-line depreciation ($)?", fmt((cost - salv) / life), "(cost − salvage)/life")
    return Q("numeric", f"An asset costs ${cost} with a {life}-year life. Using double-declining balance, what is the first-year depreciation ($)?", fmt(cost * 2 / life), "cost × 2/life")


@family("acct.ratios", "Financial ratios", {"accounting": (4, 9), "business": (4, 8), "finance": (1, 3)},
        "Current ratio = current assets ÷ current liabilities. Gross margin = (revenue − COGS) ÷ revenue. ROE = net income ÷ equity. Debt-to-equity = liabilities ÷ equity. Inventory turnover = COGS ÷ average inventory.",
        "Using total assets where the ratio calls for current assets.")
def ratios(rng, t):
    k = rng.random()
    ca, cl = rng.randint(50, 400) * 10, rng.randint(20, 200) * 10
    if k < .3:
        return Q("numeric", f"Current assets are ${ca} and current liabilities ${cl}. What is the current ratio? {SIG}", fmt(ca / cl), "CA/CL")
    if k < .6:
        rev, cogs = rng.randint(100, 900) * 10, rng.randint(20, 90) * 10
        cogs = min(cogs, rev - 10)
        return Q("numeric", f"Revenue is ${rev} and cost of goods sold ${cogs}. What is the gross margin (%)? {SIG}", fmt((rev - cogs) / rev * 100), "(rev − COGS)/rev")
    ni, eq = rng.randint(10, 100) * 10, rng.randint(100, 900) * 10
    return Q("numeric", f"Net income is ${ni} and shareholders' equity ${eq}. What is the return on equity (%)? {SIG}", fmt(ni / eq * 100), "NI/equity")


@family("acct.costing", "Cost behaviour and inventory", {"accounting": (5, 12), "business": (5, 10), "entrepreneurship": (4, 9), "marketing": (5, 8)},
        "Contribution margin per unit = price − variable cost per unit. Break-even units = fixed costs ÷ contribution margin. Units for a target profit = (fixed + profit) ÷ margin. FIFO assigns the oldest purchase costs to units sold first.",
        "Including fixed costs per unit in the contribution margin.")
def costing(rng, t):
    k = rng.random()
    price, var, fixed = rng.choice([20, 25, 40]), rng.choice([8, 10, 15]), rng.choice([2400, 6000, 10000, 15000])
    cm = price - var
    if k < .4:
        fixed = cm * rng.randint(100, 600)
        return Q("numeric", f"Price ${price}, variable cost ${var} per unit, fixed costs ${fixed}. How many units to break even?", fixed // cm, "fixed ÷ contribution margin")
    if k < .65:
        fixed = cm * rng.randint(100, 400)
        target = cm * rng.randint(50, 200)
        return Q("numeric", f"Price ${price}, variable cost ${var}, fixed costs ${fixed}. How many units give a profit of ${target}?", (fixed + target) // cm, "(fixed + profit)/margin")
    u1, c1, u2, c2, sold = rng.choice([100, 200]), rng.choice([5, 6, 8]), rng.choice([100, 200]), rng.choice([9, 10, 12]), rng.randint(50, 250)
    sold = min(sold, u1 + u2)
    cogs = min(sold, u1) * c1 + max(0, sold - u1) * c2
    return Q("numeric", f"Inventory: {u1} units bought at ${c1}, then {u2} units at ${c2}. Using FIFO, what is the cost of goods sold for {sold} units sold ($)?", cogs, "oldest costs first")


# ===================================================================== ECONOMICS

@family("econ.basics", "Opportunity cost and trade", {"economics": (1, 3), "microeconomics": (1, 2), "macroeconomics": (1, 1)},
        "Opportunity cost is what you give up to get something. If a country can make 20 cars or 40 trucks, one car costs 2 trucks. Comparative advantage: specialise where the opportunity cost is lower.",
        "Confusing absolute with comparative advantage.")
def econ_basics(rng, t):
    a, b = rng.choice([10, 20, 30]), rng.choice([20, 40, 60, 90])
    return Q("numeric", f"With all its resources an economy can make {a} cars or {b} trucks. How many trucks must it give up to make one more car?", fmt(b / a), "b/a")


@family("econ.market", "Supply, demand and elasticity", {"economics": (3, 7), "microeconomics": (1, 5), "business": (3, 6), "marketing": (3, 6)},
        "Equilibrium is where quantity demanded = quantity supplied. With Q_d = a − bP and Q_s = c + dP: P* = (a − c)/(b + d). Price elasticity of demand = %ΔQ / %ΔP; |e| > 1 is elastic (cutting price raises revenue).",
        "Using the starting values instead of the midpoint for elasticity.")
def market(rng, t):
    k = rng.random()
    if k < .5:
        p = rng.randint(3, 20)
        b, d, c = rng.randint(1, 5), rng.randint(1, 5), rng.randint(0, 20)
        a = (b + d) * p + c
        return Q("numeric", f"Demand is Q = {a} − {b}P and supply is Q = {c} + {d}P. What is the equilibrium price? (decimal allowed)", p, f"{a} − {b}P = {c} + {d}P → P = {p}")
    p1, p2, q1, q2 = rng.choice([(10, 12, 100, 90), (5, 6, 200, 160), (20, 22, 50, 48)])
    e = ((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2))
    return Q("numeric", f"Price rises from ${p1} to ${p2} and quantity demanded falls from {q1} to {q2}. What is the price elasticity (midpoint method, include the sign)? {SIG}", fmt(e), "%ΔQ/%ΔP with midpoints")


@family("econ.welfare", "Surplus, costs and market power", {"microeconomics": (5, 12), "economics": (6, 10), "business": (7, 10)},
        "Consumer surplus = ½(maximum price − market price)·Q for a linear demand curve. Marginal cost MC = dTC/dQ. A monopolist with demand P = a − bQ and constant marginal cost c maximises profit at MR = MC: Q* = (a − c)/(2b). HHI = Σ(market share %)².",
        "Setting price (not marginal revenue) equal to marginal cost.")
def welfare(rng, t):
    k = rng.random()
    if k < .3:
        pmax, p, q = rng.choice([20, 30, 50]), rng.choice([10, 12]), rng.choice([10, 20, 40])
        return Q("numeric", f"Demand is linear with a choke price of ${pmax}. At the market price ${p} the quantity is {q}. What is the consumer surplus ($)?", fmt(0.5 * (pmax - p) * q), "triangle area")
    if k < .55:
        a, b = rng.choice([1, 2, 3]), rng.randint(2, 6)
        q = rng.randint(2, 10)
        return Q("numeric", f"Total cost is TC = {a}Q² + {b}Q + 50. What is marginal cost at Q = {q}?", 2 * a * q + b, "MC = 2aQ + b")
    if k < .8:
        a, b, c = rng.choice([100, 200]), rng.choice([2, 4]), rng.choice([20, 40])
        return Q("numeric", f"A monopolist faces P = {a} − {b}Q with constant marginal cost {c}. What output maximises profit?", fmt((a - c) / (2 * b)), "MR = MC")
    sh = rng.choice([[50, 30, 20], [40, 30, 20, 10], [25, 25, 25, 25], [60, 20, 10, 10]])
    return Q("numeric", f"Firms have market shares {sh}%. What is the HHI?", sum(s * s for s in sh), "Σ share²")


@family("macro.accounts", "GDP, inflation and unemployment", {"macroeconomics": (1, 5), "economics": (2, 6)},
        "GDP = C + I + G + (X − M). Inflation rate = (CPI₁ − CPI₀)/CPI₀. Real GDP = nominal GDP ÷ deflator × 100. Unemployment rate = unemployed ÷ (employed + unemployed).",
        "Including people outside the labour force in the unemployment rate.")
def macro_accounts(rng, t):
    k = rng.random()
    if k < .35:
        c, i, g, x, m = rng.randint(50, 100) * 10, rng.randint(10, 30) * 10, rng.randint(10, 40) * 10, rng.randint(5, 20) * 10, rng.randint(5, 20) * 10
        return Q("numeric", f"C = {c}, I = {i}, G = {g}, exports = {x}, imports = {m} (billions). What is GDP?", c + i + g + x - m, "C + I + G + X − M")
    if k < .65:
        c0, c1 = rng.choice([100, 200]), rng.choice([104, 106, 110, 215, 224])
        c1 = c1 if c1 > c0 else c0 + 8
        return Q("numeric", f"The CPI rises from {c0} to {c1}. What is the inflation rate (%)? {SIG}", fmt((c1 - c0) / c0 * 100), "(CPI₁−CPI₀)/CPI₀")
    e, u = rng.choice([90, 95, 140]), rng.choice([5, 10, 10])
    return Q("numeric", f"There are {e} million employed and {u} million unemployed. What is the unemployment rate (%)? {SIG}", fmt(u / (e + u) * 100), "U/(E+U)")


@family("macro.policy", "Multipliers, growth and policy rules", {"macroeconomics": (5, 12), "economics": (6, 12)},
        "Spending multiplier = 1/(1 − MPC). Money multiplier = 1/reserve ratio. Fisher: real rate ≈ nominal rate − inflation. Rule of 70: doubling time ≈ 70/growth rate (%). Solow steady state: k* = (s/(δ+n))^(1/(1−α)). Taylor rule: i = π + r* + 0.5(π − π*) + 0.5·output gap.",
        "Using MPS instead of 1 − MPC in the multiplier.")
def macro_policy(rng, t):
    k = rng.random()
    if k < .3:
        mpc, dg = rng.choice([0.5, 0.75, 0.8, 0.9]), rng.choice([10, 20, 50])
        return Q("numeric", f"MPC = {mpc}. Government spending rises by ${dg} billion. By how much does GDP rise at most ($ billion)? {SIG}", fmt(dg / (1 - mpc)), "ΔG × 1/(1−MPC)")
    if k < .5:
        rr, dep = rng.choice([0.1, 0.2, 0.05]), rng.choice([100, 500])
        return Q("numeric", f"The reserve ratio is {rr:.0%}. A new deposit of ${dep} is made. What is the maximum total money created ($)? {SIG}", fmt(dep / rr), "deposit × 1/reserve ratio")
    if k < .7:
        i, pi = rng.choice([0.05, 0.08, 0.12]), rng.choice([0.02, 0.03])
        return Q("numeric", f"The nominal interest rate is {i:.0%} and inflation is {pi:.0%}. What is the real interest rate (%)?", fmt((i - pi) * 100), "i − π")
    if k < .85:
        g = rng.choice([2, 3.5, 5, 7])
        return Q("numeric", f"An economy grows {g}% a year. About how many years until output doubles (rule of 70)? {SIG}", fmt(70 / g), "70/g")
    s, d, n, a = rng.choice([0.2, 0.3]), rng.choice([0.05, 0.1]), 0.01, rng.choice([0.33, 0.5])
    return Q("numeric", f"In a Solow model s = {s}, δ = {d}, n = {n}, α = {a}. What is the steady-state capital per worker k* = (s/(δ+n))^(1/(1−α))? {SIG}", fmt((s / (d + n)) ** (1 / (1 - a))), "Solow steady state")


@family("biz.metrics", "Business and marketing metrics", {"business": (2, 12), "marketing": (1, 12), "entrepreneurship": (1, 12)},
        "Markup = (price − cost)/cost; margin = (price − cost)/price. Conversion rate = conversions ÷ visitors. Customer acquisition cost CAC = marketing spend ÷ new customers. Lifetime value LTV ≈ monthly margin × lifetime in months. Runway = cash ÷ monthly burn.",
        "Mixing up markup and margin.")
def biz_metrics(rng, t):
    k = rng.random()
    if k < .25:
        cost, price = rng.choice([20, 40, 60]), rng.choice([30, 50, 80, 100])
        price = max(price, cost + 5)
        return Q("numeric", f"A product costs ${cost} to make and sells for ${price}. What is the markup on cost (%)? {SIG}", fmt((price - cost) / cost * 100), "(price − cost)/cost")
    if k < .5:
        vis, conv = rng.choice([2000, 5000, 10000]), rng.choice([40, 100, 250])
        return Q("numeric", f"{vis} people visit a website and {conv} buy. What is the conversion rate (%)? {SIG}", fmt(conv / vis * 100), "conversions/visitors")
    if k < .75:
        spend, new, margin, life = rng.choice([5000, 10000]), rng.choice([50, 100, 200]), rng.choice([10, 20, 30]), rng.choice([12, 24, 36])
        return Q("numeric", f"A campaign costs ${spend} and wins {new} customers, each worth ${margin} margin a month for {life} months. What is the LTV/CAC ratio? {SIG}", fmt(margin * life / (spend / new)), "LTV ÷ CAC")
    cash, burn = rng.choice([120000, 300000, 900000]), rng.choice([10000, 20000, 50000])
    return Q("numeric", f"A startup has ${cash} in the bank and burns ${burn} a month. How many months of runway does it have? {SIG}", fmt(cash / burn), "cash ÷ burn")
