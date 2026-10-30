"""Life, earth, health and social-science calculation families, plus music maths."""

from __future__ import annotations

import math
from fractions import Fraction

from .generators import Q, family, fmt, scale

SIG = "(3 significant figures)"


@family("bio.genetics", "Genetics and inheritance", {"genetics": (1, 12), "biology": (4, 10)},
        "Mendel: a cross Aa × Aa gives AA : Aa : aa = 1 : 2 : 1, so a recessive trait appears in ¼. Dihybrid (AaBb × AaBb) phenotypes are 9 : 3 : 3 : 1. Hardy–Weinberg: p + q = 1 and p² + 2pq + q² = 1. "
        "PCR doubles the DNA every cycle: 2ⁿ copies after n cycles. Map distance (cM) = recombinants ÷ total offspring × 100.",
        "Confusing allele frequency (q) with genotype frequency (q²).")
def genetics(rng, t):
    k = rng.random()
    if k < .25:
        return Q("numeric", "Two heterozygous parents (Aa × Aa) have a child. What is the probability the child shows the recessive trait? (fraction)", Fraction(1, 4), "aa = ¼")
    if k < .45:
        return Q("numeric", "In an AaBb × AaBb cross, what fraction of offspring show both dominant traits? (fraction)", Fraction(9, 16), "9:3:3:1")
    if k < .7:
        q2 = rng.choice([0.04, 0.09, 0.16, 0.25, 0.01])
        q = math.sqrt(q2)
        return Q("numeric", f"In a population at Hardy–Weinberg equilibrium {q2:.0%} of individuals show a recessive trait. What fraction are carriers (heterozygotes)? (2 decimals)", f"{2 * q * (1 - q):.2f}", "q = √q², carriers = 2pq")
    if k < .85:
        n = rng.randint(5, 30)
        return Q("numeric", f"How many copies of a DNA segment exist after {n} PCR cycles starting from 1 copy?", 2 ** n, "2ⁿ")
    rec, tot = rng.choice([(12, 100), (30, 200), (9, 300), (45, 500)])
    return Q("numeric", f"{rec} of {tot} offspring are recombinant. What is the map distance in centimorgans? {SIG}", fmt(rec / tot * 100), "recombinants ÷ total × 100")


@family("bio.growth", "Growth and diversity", {"microbiology": (1, 12), "ecology": (1, 12), "biology": (3, 9), "zoology": (6, 12), "botany": (6, 12)},
        "Doubling growth: N = N₀·2^(t/g) with generation time g. Exponential growth: N = N₀e^(rt) with doubling time ln 2/r. Logistic growth: dN/dt = rN(1 − N/K). Simpson's diversity index = 1 − Σpᵢ².",
        "Using the logistic formula when the population is far below K (growth is then almost exponential).")
def growth(rng, t):
    k = rng.random()
    if k < .3:
        n0, g, tm = rng.choice([100, 500, 1000]), rng.choice([20, 30, 60]), rng.choice([60, 120, 180])
        return Q("numeric", f"A bacterial culture starts with {n0} cells and doubles every {g} minutes. How many cells are there after {tm} minutes?", n0 * 2 ** (tm // g), "N₀·2^(t/g)")
    if k < .55:
        r = rng.choice([0.01, 0.02, 0.05, 0.1])
        return Q("numeric", f"A population grows exponentially at r = {r} per year. What is its doubling time in years? {SIG}", fmt(math.log(2) / r), "ln 2 / r")
    if k < .8:
        r, n, K = rng.choice([0.2, 0.5]), rng.choice([100, 250, 500]), rng.choice([1000, 2000])
        return Q("numeric", f"A logistic population has r = {r}, N = {n} and carrying capacity K = {K}. What is dN/dt (individuals per time)? {SIG}", fmt(r * n * (1 - n / K)), "rN(1 − N/K)")
    ps = rng.choice([[0.5, 0.5], [0.7, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], [0.25] * 4])
    return Q("numeric", f"Species make up {ps} of a community. What is Simpson's diversity index 1 − Σp²? (3 decimals)", f"{1 - sum(p * p for p in ps):.3f}", "1 − Σp²")


@family("bio.neuro", "Neurons and signalling", {"neuroscience": (1, 12), "psychology": (4, 9), "biomedical_eng": (3, 8)},
        "The Nernst equation gives an ion's equilibrium potential: E = (61.5 mV/z)·log₁₀([out]/[in]) at body temperature. Conduction velocity = distance ÷ latency. Membrane time constant τ = R·C. Firing rate = spikes ÷ time.",
        "Forgetting the ion's charge z (and its sign) in the Nernst equation.")
def neuro(rng, t):
    k = rng.random()
    if k < .4:
        out, inn, z = rng.choice([(5, 140, 1), (145, 12, 1), (110, 10, -1), (2, 0.0002, 2)])
        return Q("numeric", f"An ion with charge z = {z:+d} has {out} mM outside and {inn} mM inside the cell. What is its Nernst potential (mV, 37 °C: E = 61.5/z · log₁₀(out/in))? {SIG}", fmt(61.5 / z * math.log10(out / inn)), "Nernst equation")
    if k < .7:
        d, lat = rng.choice([(0.5, 5), (1.2, 12), (0.1, 2), (0.9, 30)])
        return Q("numeric", f"A nerve signal travels {d} m in {lat} ms. What is the conduction velocity (m/s)? {SIG}", fmt(d / (lat / 1000)), "distance ÷ time")
    r, c = rng.choice([(100e6, 100e-12), (50e6, 200e-12), (20e6, 500e-12)])
    return Q("numeric", f"A membrane has R = {r / 1e6:g} MΩ and C = {c * 1e12:g} pF. What is its time constant (ms)? {SIG}", fmt(r * c * 1000), "τ = RC")


@family("bio.biochem", "Enzymes and organic structure", {"biochemistry": (1, 12), "organic_chemistry": (1, 12), "biology": (7, 12), "pharmacology": (3, 8)},
        "Michaelis–Menten: v = V_max[S]/(K_m + [S]); at [S] = K_m the rate is V_max/2. Degrees of unsaturation for C_aH_bN_cX_d: DoU = (2a + 2 + c − b − d)/2 (each ring or double bond counts 1, a triple bond 2).",
        "Forgetting that nitrogen adds to the numerator and halogens subtract.")
def biochem(rng, t):
    if rng.random() < .55:
        vmax, km, s = rng.choice([100, 200, 50]), rng.choice([2, 5, 10]), rng.choice([1, 5, 10, 20, 50])
        return Q("numeric", f"An enzyme has V_max = {vmax} μmol/min and K_m = {km} mM. What is the rate at [S] = {s} mM (μmol/min)? {SIG}", fmt(vmax * s / (km + s)), "V_max[S]/(K_m+[S])")
    a, c, d, rings = rng.randint(2, 10), rng.randint(0, 2), rng.randint(0, 2), rng.randint(0, 4)
    b = 2 * a + 2 + c - d - 2 * rings
    dou = rings
    formula = f"C{a}H{b}" + (f"N{c}" if c else "") + (f"Cl{d}" if d else "")
    return Q("numeric", f"What is the degree of unsaturation of {formula}?", dou, "(2C + 2 + N − H − X)/2")


@family("bio.epi", "Epidemiology and clinical numbers", {"public_health": (1, 12), "medicine": (2, 12), "ai_safety": (4, 12), "pharmacology": (1, 6)},
        "Herd immunity threshold = 1 − 1/R₀. Relative risk = risk in exposed ÷ risk in unexposed. Absolute risk reduction ARR = control risk − treated risk; number needed to treat NNT = 1/ARR. Prevalence = existing cases ÷ population. Dose = weight × (mg/kg).",
        "Treating relative risk reduction as if it were the absolute change.")
def epi(rng, t):
    k = rng.random()
    if k < .25:
        r0 = rng.choice([1.5, 2, 3, 4, 5, 10, 15])
        return Q("numeric", f"A disease has R₀ = {r0}. What percentage of the population must be immune for herd immunity? {SIG}", fmt((1 - 1 / r0) * 100), "1 − 1/R₀")
    if k < .5:
        e, ne = rng.choice([(30, 10), (12, 4), (8, 16), (50, 25)])
        return Q("numeric", f"{e} of 100 exposed people and {ne} of 100 unexposed people get ill. What is the relative risk? {SIG}", fmt(e / ne), "risk ratio")
    if k < .75:
        c, tr = rng.choice([(0.20, 0.15), (0.10, 0.06), (0.30, 0.25), (0.08, 0.04)])
        return Q("numeric", f"Event rates are {c:.0%} on control and {tr:.0%} on treatment. What is the number needed to treat (NNT)? {SIG}", fmt(1 / (c - tr)), "1/ARR")
    w, dose = rng.choice([20, 35, 60, 80]), rng.choice([5, 10, 15, 25])
    return Q("numeric", f"A patient weighs {w} kg and the dose is {dose} mg/kg. How many mg are given?", w * dose, "weight × mg/kg")


@family("bio.pharm", "Pharmacokinetics", {"pharmacology": (3, 12), "medicine": (6, 12), "biomedical_eng": (4, 8)},
        "Elimination with half-life t½: C(t) = C₀(½)^(t/t½). Loading dose = C_target × V_d. Steady state is reached after about 4–5 half-lives; average steady-state concentration C_ss = (F·dose/τ)/CL.",
        "Using the dosing interval instead of the half-life in the decay exponent.")
def pharm(rng, t):
    k = rng.random()
    if k < .45:
        c0, th, tm = rng.choice([80, 40, 100]), rng.choice([2, 4, 6]), rng.randint(1, 4)
        return Q("numeric", f"A drug has half-life {th} h and an initial concentration of {c0} mg/L. What is the concentration after {th * tm} hours (mg/L)? {SIG}", fmt(c0 / 2 ** tm), "C₀(½)^n")
    if k < .75:
        c, v = rng.choice([10, 20, 5]), rng.choice([40, 70, 100])
        return Q("numeric", f"A target concentration of {c} mg/L is wanted in a volume of distribution of {v} L. What loading dose (mg) is needed?", c * v, "C × V_d")
    f, dose, tau, cl = rng.choice([1, 0.8, 0.5]), rng.choice([200, 500, 1000]), rng.choice([6, 8, 12]), rng.choice([2, 5, 10])
    return Q("numeric", f"Bioavailability {f}, dose {dose} mg every {tau} h, clearance {cl} L/h. What is the average steady-state concentration (mg/L)? {SIG}", fmt(f * dose / tau / cl), "(F·dose/τ)/CL")


@family("bio.nutrition", "Energy and nutrition", {"nutrition": (1, 12), "public_health": (1, 5), "biology": (1, 4)},
        "Energy: protein and carbohydrate give 4 kcal per gram, fat 9 kcal per gram, alcohol 7. BMI = weight (kg) ÷ height² (m²). Mifflin–St Jeor (male) resting energy = 10·kg + 6.25·cm − 5·age + 5 kcal/day.",
        "Using pounds or inches in the metric formulas.")
def nutrition(rng, t):
    k = rng.random()
    if k < .4:
        p, c, f = rng.randint(10, 40), rng.randint(20, 90), rng.randint(5, 30)
        return Q("numeric", f"A food has {p} g protein, {c} g carbohydrate and {f} g fat. How many kilocalories does it provide?", 4 * p + 4 * c + 9 * f, "4P + 4C + 9F")
    if k < .7:
        w, h = rng.choice([50, 65, 80, 95]), rng.choice([1.55, 1.65, 1.75, 1.85])
        return Q("numeric", f"A person weighs {w} kg and is {h} m tall. What is their BMI? {SIG}", fmt(w / h ** 2), "kg / m²")
    w, h, a = rng.choice([60, 75, 90]), rng.choice([160, 175, 185]), rng.choice([20, 30, 45])
    return Q("numeric", f"Using Mifflin–St Jeor for a man ({w} kg, {h} cm, {a} years), what is the resting energy need (kcal/day)? {SIG}", fmt(10 * w + 6.25 * h - 5 * a + 5), "10w + 6.25h − 5a + 5")


@family("earth.calc", "Earth, ocean, weather and environment calculations", {"earth_science": (1, 12), "meteorology": (1, 12), "oceanography": (1, 12), "environmental_eng": (1, 12), "archaeology": (3, 12), "civil_eng": (1, 3)},
        "Radiometric dating: age = t½·log₂(N₀/N). Typical temperature lapse rate: 6.5 °C drop per 1000 m. Dew point ≈ T − (100 − RH)/5 (°C, RH in %). Shallow-water wave speed = √(g·depth). Plate speed = distance ÷ time. Mixing: C = (C₁Q₁ + C₂Q₂)/(Q₁ + Q₂).",
        "Forgetting that each half-life halves the amount remaining.")
def earth(rng, t):
    k = rng.random()
    if k < .2:
        th, ratio = rng.choice([(5730, 2), (5730, 4), (5730, 8), (1.25e9, 2), (4.5e9, 2)])
        return Q("numeric", f"A sample has 1/{ratio} of its original radioactive isotope (half-life {th:g} years). How old is it (years)? {SIG}", fmt(th * math.log2(ratio)), "t½ · log₂(N₀/N)")
    if k < .4:
        t0, h = rng.choice([20, 25, 30]), rng.choice([1000, 2000, 3000, 4500])
        return Q("numeric", f"The air temperature at sea level is {t0} °C. Using a lapse rate of 6.5 °C per 1000 m, what is it at {h} m? {SIG}", fmt(t0 - 6.5 * h / 1000), "T₀ − 6.5·h/1000")
    if k < .55:
        T, rh = rng.choice([25, 30, 20]), rng.choice([50, 60, 80, 90])
        return Q("numeric", f"Temperature {T} °C and relative humidity {rh}%. Estimate the dew point (°C) with Td ≈ T − (100 − RH)/5. {SIG}", fmt(T - (100 - rh) / 5), "rule of thumb")
    if k < .7:
        d = rng.choice([10, 100, 1000, 4000])
        return Q("numeric", f"What is the speed (m/s) of a shallow-water wave in {d} m of water? (g = 9.8) {SIG}", fmt(math.sqrt(9.8 * d)), "√(g d)")
    if k < .85:
        km, my = rng.choice([100, 200, 500]), rng.choice([5, 10, 20])
        return Q("numeric", f"A tectonic plate moves {km} km in {my} million years. What is its speed (cm per year)? {SIG}", fmt(km * 1e5 / (my * 1e6)), "distance ÷ time")
    c1, q1, c2, q2 = rng.choice([(2, 10, 0, 40), (10, 5, 1, 20), (50, 2, 0, 18)])
    return Q("numeric", f"A river (flow {q2} m³/s, concentration {c2} mg/L) receives a discharge ({q1} m³/s at {c1} mg/L). What is the mixed concentration (mg/L)? {SIG}", fmt((c1 * q1 + c2 * q2) / (q1 + q2)), "flow-weighted average")


@family("soc.research", "Research methods and social-science numbers", {"research": (3, 12), "psychology": (1, 12), "sociology": (1, 12), "education": (3, 12),
                                                                        "political_science": (3, 12), "geography": (1, 12), "anthropology": (6, 12), "linguistics": (6, 12), "law": (6, 12)},
        "Sample size for margin of error E: n = (z·σ/E)² (z = 1.96 for 95%). Effect size d = (mean₁ − mean₂)/pooled sd. Running k independent tests at level α gives a family-wise false-positive chance 1 − (1 − α)ᵏ; Bonferroni uses α/k. "
        "Poll margin of error ≈ 1.96·√(p(1 − p)/n). Population density = people ÷ area. Time zones: 15° of longitude = 1 hour.",
        "Running many tests and reporting only the 'significant' one (p-hacking).")
def soc_research(rng, t):
    k = rng.random()
    if k < .2:
        s, e = rng.choice([(10, 2), (15, 3), (20, 5), (12, 1.5)])
        return Q("numeric", f"What sample size gives a 95% margin of error of ±{e} when the standard deviation is {s}? (z = 1.96, round up to a whole number)", math.ceil((1.96 * s / e) ** 2), "n = (zσ/E)²")
    if k < .4:
        m1, m2, sd = rng.choice([(75, 70, 10), (82, 70, 15), (50, 45, 5)])
        return Q("numeric", f"Group means are {m1} and {m2} with a pooled standard deviation of {sd}. What is Cohen's d? {SIG}", fmt((m1 - m2) / sd), "difference ÷ sd")
    if k < .6:
        n, a = rng.choice([5, 10, 20]), 0.05
        return Q("numeric", f"You run {n} independent tests each at α = 0.05. What is the probability of at least one false positive when every null hypothesis is true? (3 decimals)", f"{1 - (1 - a) ** n:.3f}", "1 − (1−α)ⁿ")
    if k < .75:
        p, n = rng.choice([0.5, 0.4, 0.3]), rng.choice([400, 625, 1000, 2500])
        return Q("numeric", f"A poll of {n} people finds {p:.0%} support. What is the 95% margin of error in percentage points? {SIG}", fmt(1.96 * math.sqrt(p * (1 - p) / n) * 100), "1.96√(p(1−p)/n)")
    if k < .9:
        pop, area = rng.choice([(8000000, 800), (50000, 25), (1400000000, 9600000)])
        return Q("numeric", f"A region has {pop:,} people in {area:,} km². What is its population density (people per km²)? {SIG}", fmt(pop / area), "people ÷ area")
    lon = rng.choice([30, 45, 75, 120, 150])
    return Q("numeric", f"Two cities differ by {lon}° of longitude. What is their solar-time difference in hours?", fmt(lon / 15), "15° per hour")


@family("music.math", "Music maths", {"music_theory": (1, 12), "signal_processing": (1, 2), "physics": (1, 2)},
        "Equal temperament: each semitone multiplies the frequency by 2^(1/12); an octave (12 semitones) doubles it. f = 440·2^(n/12) Hz for a note n semitones above A4. A tempo of B beats per minute gives 60/B seconds per beat. "
        "Interval sizes in semitones: minor 3rd = 3, major 3rd = 4, perfect 5th = 7, octave = 12.",
        "Adding frequencies instead of multiplying by ratios.")
def music(rng, t):
    k = rng.random()
    if k < .35:
        n = rng.choice([3, 4, 7, 12, -12, -5, 9])
        return Q("numeric", f"What is the frequency (Hz) of the note {n} semitones from A4 = 440 Hz in equal temperament? {SIG}", fmt(440 * 2 ** (n / 12)), "440·2^(n/12)")
    if k < .65:
        b = rng.choice([60, 90, 120, 150])
        return Q("numeric", f"At {b} beats per minute, how many seconds does one beat last? {SIG}", fmt(60 / b), "60/BPM")
    note = rng.choice([("C", "E", 4), ("C", "G", 7), ("A", "C", 3), ("D", "F", 3), ("C", "C", 12)])
    return Q("numeric", f"How many semitones separate {note[0]} and the {note[1]} above it?" if note[0] != note[1] else "How many semitones are in an octave?", note[2], "count the half-steps")
