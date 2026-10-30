"""Physics, chemistry, astronomy, electronics and engineering families."""

from __future__ import annotations

import math
import re

from .gen_math import sub
from .generators import Q, family, fmt, mcq, sci, scale

G_ACC = 9.8
G_CONST = 6.674e-11
C_LIGHT = 3.00e8
K_COULOMB = 8.99e9
H_PLANCK = 6.626e-34
EV = 1.602e-19
R_GAS = 8.314
SIG = "(3 significant figures)"


def _num(rng, lo, hi):
    return rng.randint(lo, hi)


# ===================================================================== PHYSICS

@family("phys.units", "Units and conversions", {"physics": (1, 3), "chemistry": (1, 2)},
        "Convert by multiplying by a ratio equal to 1: 1 m = 100 cm, 1 km = 1000 m, 1 kg = 1000 g, 1 h = 60 min = 3600 s. Always write the unit with the number.",
        "Multiplying when you should divide (km → m multiplies by 1000).")
def units(rng, t):
    k = rng.choice([("cm", "m", 1 / 100), ("km", "m", 1000), ("g", "kg", 1 / 1000), ("h", "min", 60), ("min", "s", 60), ("m", "cm", 100)])
    v = rng.choice([2, 5, 25, 40, 120, 250, 450, 3000]) if k[2] < 1 else rng.randint(2, 40)
    return Q("numeric", f"Convert {v} {k[0]} to {k[1]}.", fmt(v * k[2]), f"{v} × {fmt(k[2])}")


@family("phys.speed", "Speed, distance and time", {"physics": (1, 4)},
        "speed = distance ÷ time, so distance = speed × time and time = distance ÷ speed. Keep the units consistent (km and h, or m and s).",
        "Mixing minutes with hours.")
def speed(rng, t):
    v, tm = rng.randint(3, scale(t, 20, 90)), rng.randint(2, scale(t, 5, 12))
    k = rng.random()
    if k < .4:
        return Q("numeric", f"A car travels {v * tm} km in {tm} hours. What is its average speed in km/h?", v, f"{v * tm} ÷ {tm}")
    if k < .7:
        return Q("numeric", f"A cyclist rides at {v} km/h for {tm} hours. How far does she go (km)?", v * tm, f"{v} × {tm}")
    return Q("numeric", f"How many hours does it take to travel {v * tm} km at {v} km/h?", tm, f"{v * tm} ÷ {v}")


@family("phys.newton", "Newton's laws", {"physics": (2, 5), "mechanical_eng": (1, 3), "aerospace_eng": (1, 2)},
        "Newton's second law: F = ma (force in newtons, mass in kg, acceleration in m/s²). Weight W = mg with g = 9.8 m/s². With no net force an object keeps its velocity (first law); forces come in equal and opposite pairs (third law).",
        "Confusing mass (kg) with weight (N).")
def newton(rng, t):
    m, a = rng.randint(2, scale(t, 20, 100)), rng.randint(1, scale(t, 6, 15))
    k = rng.random()
    if k < .35:
        return Q("numeric", f"What net force gives a {m} kg object an acceleration of {a} m/s²?", m * a, f"F = ma = {m}×{a}")
    if k < .65:
        return Q("numeric", f"A net force of {m * a} N acts on a {m} kg object. What is its acceleration (m/s²)?", a, f"a = F/m")
    if k < .85:
        return Q("numeric", f"What is the weight of a {m} kg object? (g = 9.8 m/s²) {SIG}", fmt(m * G_ACC), f"W = mg = {m}×9.8")
    fr = rng.randint(5, 30)
    return Q("numeric", f"A {m} kg block is pushed with {m * a + fr} N against {fr} N of friction. What is its acceleration (m/s²)?", a, "net force = push − friction")


@family("phys.density", "Density and pressure", {"physics": (3, 5), "civil_eng": (1, 2)},
        "Density ρ = m/V (kg/m³). Pressure P = F/A (pascals = N/m²). In a fluid, pressure at depth h is P = ρgh.",
        "Using the wrong area unit (cm² vs m²).")
def density(rng, t):
    k = rng.random()
    if k < .35:
        m, v = rng.randint(2, 50), rng.choice([2, 4, 5, 10])
        return Q("numeric", f"A block has mass {m * v} kg and volume {v} m³. What is its density (kg/m³)?", m, "ρ = m/V")
    if k < .65:
        f, a = rng.randint(20, 400), rng.choice([2, 4, 5, 10])
        return Q("numeric", f"A force of {f * a} N acts evenly on an area of {a} m². What is the pressure (Pa)?", f, "P = F/A")
    h = rng.randint(2, 40)
    return Q("numeric", f"What is the water pressure due to the water alone at a depth of {h} m? (ρ = 1000 kg/m³, g = 9.8 m/s², in Pa) {SIG}", fmt(1000 * G_ACC * h), "P = ρgh")


@family("phys.energy", "Work, energy and power", {"physics": (3, 6), "mechanical_eng": (2, 4)},
        "Work W = F·d. Kinetic energy KE = ½mv². Gravitational potential energy PE = mgh. Power P = W/t (watts). Energy is conserved: KE + PE stays constant without friction.",
        "Forgetting to square the speed in KE.")
def energy(rng, t):
    m, v, h = rng.randint(1, 20), rng.randint(2, 20), rng.randint(2, 50)
    k = rng.random()
    if k < .3:
        return Q("numeric", f"What is the kinetic energy of a {m} kg object moving at {v} m/s (J)?", fmt(0.5 * m * v * v), "½mv²")
    if k < .55:
        return Q("numeric", f"What is the potential energy gained by lifting {m} kg through {h} m? (g = 9.8, J) {SIG}", fmt(m * G_ACC * h), "mgh")
    if k < .75:
        f, d = rng.randint(5, 80), rng.randint(2, 30)
        return Q("numeric", f"How much work does a force of {f} N do moving an object {d} m in its direction (J)?", f * d, "W = F·d")
    return Q("numeric", f"A ball falls from rest through {h} m. How fast is it going just before landing (m/s)? (g = 9.8) {SIG}", fmt(math.sqrt(2 * G_ACC * h)), "mgh = ½mv²")


@family("phys.heat", "Heat and temperature", {"physics": (4, 7), "chemistry": (7, 9)},
        "Heat needed to change temperature: Q = mcΔT. Water: c = 4186 J/(kg·K). Latent heat of melting ice: 3.34×10⁵ J/kg.",
        "Using grams with a per-kg specific heat.")
def heat(rng, t):
    m, dt = rng.choice([0.5, 1, 2, 3, 5]), rng.randint(5, 80)
    if rng.random() < .7:
        return Q("numeric", f"How much heat raises {m} kg of water by {dt} K? (c = 4186 J/kg·K) {SIG}", fmt(m * 4186 * dt), "Q = mcΔT")
    return Q("numeric", f"How much heat melts {m} kg of ice at 0 °C? (L = 3.34×10⁵ J/kg) {SIG}", fmt(m * 3.34e5), "Q = mL")


@family("phys.kinematics", "Motion with constant acceleration", {"physics": (4, 7), "aerospace_eng": (1, 3)},
        "Equations for constant acceleration a: v = u + at; s = ut + ½at²; v² = u² + 2as (u = initial velocity, v = final).",
        "Using the equations when acceleration is not constant.")
def kinematics(rng, t):
    u, a, tm = rng.randint(0, 20), rng.randint(1, 9), rng.randint(2, 12)
    k = rng.random()
    if k < .35:
        return Q("numeric", f"A car starts at {u} m/s and accelerates at {a} m/s² for {tm} s. What is its final speed (m/s)?", u + a * tm, "v = u + at")
    if k < .7:
        return Q("numeric", f"A car starts at {u} m/s and accelerates at {a} m/s² for {tm} s. How far does it travel (m)?", fmt(u * tm + 0.5 * a * tm * tm), "s = ut + ½at²")
    s = rng.randint(20, 200)
    return Q("numeric", f"A car accelerates from {u} m/s at {a} m/s² over {s} m. What is its final speed (m/s)? {SIG}", fmt(math.sqrt(u * u + 2 * a * s)), "v² = u² + 2as")


@family("phys.projectile", "Projectile motion", {"physics": (5, 8), "aerospace_eng": (2, 4)},
        "Launch speed v at angle θ: time of flight T = 2v sinθ/g; maximum height H = v² sin²θ/(2g); range R = v² sin 2θ/g (level ground, no air resistance).",
        "Using degrees mode wrong in sin; forgetting the vertical and horizontal motions are independent.")
def projectile(rng, t):
    v, ang = rng.randint(10, 60), rng.choice([30, 45, 60])
    th = math.radians(ang)
    k = rng.random()
    if k < .4:
        return Q("numeric", f"A ball is launched at {v} m/s at {ang}° above horizontal on level ground. What is its range (m)? (g = 9.8) {SIG}", fmt(v * v * math.sin(2 * th) / G_ACC), "R = v² sin2θ / g")
    if k < .7:
        return Q("numeric", f"A ball is launched at {v} m/s at {ang}°. What is its maximum height (m)? (g = 9.8) {SIG}", fmt((v * math.sin(th)) ** 2 / (2 * G_ACC)), "H = v²sin²θ/2g")
    return Q("numeric", f"A ball is launched at {v} m/s at {ang}°. How long is it in the air (s)? (g = 9.8) {SIG}", fmt(2 * v * math.sin(th) / G_ACC), "T = 2v sinθ / g")


@family("phys.momentum", "Momentum and collisions", {"physics": (5, 7)},
        "Momentum p = mv. Impulse = FΔt = Δp. In any collision total momentum is conserved. A perfectly inelastic collision: (m₁+m₂)v = m₁v₁ + m₂v₂.",
        "Forgetting that velocity has a sign (direction).")
def momentum(rng, t):
    m1, m2, v1 = rng.randint(1, 10), rng.randint(1, 10), rng.randint(2, 20)
    if rng.random() < .5:
        return Q("numeric", f"A {m1} kg cart moving at {v1} m/s sticks to a {m2} kg cart at rest. What is their common speed (m/s)? {SIG}", fmt(m1 * v1 / (m1 + m2)), "momentum conservation")
    return Q("numeric", f"What is the momentum of a {m1} kg object moving at {v1} m/s (kg·m/s)?", m1 * v1, "p = mv")


@family("phys.circular", "Circular motion", {"physics": (6, 8)},
        "Centripetal acceleration a = v²/r points to the centre; the force needed is F = mv²/r. Period T = 2πr/v.",
        "Thinking there is an outward 'centrifugal' force acting on the object.")
def circular(rng, t):
    m, v, r = rng.randint(1, 20), rng.randint(2, 30), rng.randint(2, 50)
    if rng.random() < .5:
        return Q("numeric", f"What is the centripetal acceleration of an object moving at {v} m/s in a circle of radius {r} m (m/s²)? {SIG}", fmt(v * v / r), "v²/r")
    return Q("numeric", f"What centripetal force keeps a {m} kg mass moving at {v} m/s on a circle of radius {r} m (N)? {SIG}", fmt(m * v * v / r), "mv²/r")


@family("phys.gravity", "Gravitation", {"physics": (6, 9), "astronomy": (5, 8)},
        "Newton's law of gravitation: F = G m₁m₂/r², G = 6.674×10⁻¹¹ N·m²/kg². Surface gravity g = GM/R². Escape speed v = √(2GM/R).",
        "Using the diameter instead of the radius.")
def gravity(rng, t):
    bodies = {"Earth": (5.97e24, 6.371e6), "Moon": (7.35e22, 1.737e6), "Mars": (6.42e23, 3.390e6), "Jupiter": (1.90e27, 6.99e7)}
    name = rng.choice(list(bodies))
    M, R = bodies[name]
    if rng.random() < .5:
        return Q("numeric", f"What is the surface gravity of {name} (mass {sci(M)} kg, radius {sci(R)} m) in m/s²? (G = 6.674×10⁻¹¹) {SIG}", fmt(G_CONST * M / R ** 2), "g = GM/R²")
    return Q("numeric", f"What is the escape speed from {name} (mass {sci(M)} kg, radius {sci(R)} m) in m/s? (G = 6.674×10⁻¹¹) {SIG}", fmt(math.sqrt(2 * G_CONST * M / R)), "v = √(2GM/R)")


@family("phys.waves", "Waves and sound", {"physics": (5, 8), "signal_processing": (1, 3)},
        "Wave speed v = fλ. Frequency f = 1/T. Sound travels at about 343 m/s in air. Doppler: a source approaching at speed vₛ is heard at f′ = f·v/(v − vₛ).",
        "Mixing up period and frequency.")
def waves(rng, t):
    f, lam = rng.choice([220, 440, 500, 1000]), rng.choice([0.2, 0.5, 0.78, 1.5, 3])
    k = rng.random()
    if k < .4:
        return Q("numeric", f"A wave has frequency {f} Hz and wavelength {lam} m. What is its speed (m/s)? {SIG}", fmt(f * lam), "v = fλ")
    if k < .7:
        return Q("numeric", f"What is the period of a {f} Hz wave (s)? {SIG}", fmt(1 / f), "T = 1/f")
    vs = rng.choice([10, 20, 30])
    return Q("numeric", f"An ambulance siren of {f} Hz approaches at {vs} m/s. What frequency do you hear (Hz)? (sound speed 343 m/s) {SIG}", fmt(f * 343 / (343 - vs)), "Doppler effect")


@family("phys.shm", "Oscillations", {"physics": (6, 9)},
        "Mass on a spring: T = 2π√(m/k), energy ½kx². Pendulum (small angles): T = 2π√(L/g). Frequency f = 1/T.",
        "Thinking the period depends on the amplitude (for small oscillations it doesn't).")
def shm(rng, t):
    if rng.random() < .5:
        L = rng.choice([0.25, 0.5, 1, 2, 4])
        return Q("numeric", f"What is the period of a pendulum {L} m long (s)? (g = 9.8) {SIG}", fmt(2 * math.pi * math.sqrt(L / G_ACC)), "T = 2π√(L/g)")
    m, k = rng.choice([0.5, 1, 2, 4]), rng.choice([50, 100, 200])
    return Q("numeric", f"A {m} kg mass hangs on a spring with k = {k} N/m. What is its oscillation period (s)? {SIG}", fmt(2 * math.pi * math.sqrt(m / k)), "T = 2π√(m/k)")


@family("phys.electrostatics", "Electric force and potential", {"physics": (7, 9), "electrical_eng": (3, 6)},
        "Coulomb's law F = kq₁q₂/r² with k = 8.99×10⁹ N·m²/C². Field of a point charge E = kq/r²; potential V = kq/r.",
        "Forgetting to convert μC or nC to coulombs.")
def electrostatics(rng, t):
    q1, q2, r = rng.randint(1, 9), rng.randint(1, 9), rng.choice([0.1, 0.2, 0.5, 1])
    if rng.random() < .6:
        return Q("numeric", f"Two charges of {q1} μC and {q2} μC are {r} m apart. What is the force between them (N)? (k = 8.99×10⁹) {SIG}", fmt(K_COULOMB * q1 * q2 * 1e-12 / r ** 2), "F = kq₁q₂/r²")
    return Q("numeric", f"What is the electric field {r} m from a {q1} μC point charge (N/C)? (k = 8.99×10⁹) {SIG}", fmt(K_COULOMB * q1 * 1e-6 / r ** 2), "E = kq/r²")


@family("elec.ohm", "Ohm's law and power", {"electronics": (1, 4), "electrical_eng": (1, 3), "physics": (5, 8)},
        "Ohm's law: V = IR (volts, amps, ohms). Power P = VI = I²R = V²/R (watts).",
        "Mixing mA with A.")
def ohm(rng, t):
    i, r = rng.randint(1, 9), rng.choice([2, 4, 5, 10, 20, 50, 100])
    k = rng.random()
    if k < .35:
        return Q("numeric", f"A {r} Ω resistor carries {i} A. What is the voltage across it (V)?", i * r, "V = IR")
    if k < .6:
        return Q("numeric", f"A {r * i} V supply drives {i} A through a resistor. What is its resistance (Ω)?", r, "R = V/I")
    if k < .85:
        return Q("numeric", f"A {r} Ω resistor carries {i} A. How much power does it dissipate (W)?", i * i * r, "P = I²R")
    return Q("numeric", f"A {r} Ω resistor is connected to {r * i} V. How much current flows (A)?", i, "I = V/R")


@family("elec.networks", "Series and parallel circuits", {"electronics": (2, 6), "electrical_eng": (2, 6), "physics": (7, 9)},
        "Series: R = R₁ + R₂ + …. Parallel: 1/R = 1/R₁ + 1/R₂ + …; two in parallel: R = R₁R₂/(R₁+R₂). Voltage divider: V_out = V_in · R₂/(R₁ + R₂).",
        "Adding resistors directly when they are in parallel.")
def networks(rng, t):
    r1, r2 = rng.choice([100, 200, 300, 600, 1000, 1200]), rng.choice([100, 300, 400, 600, 1000, 1800])
    k = rng.random()
    if k < .3:
        return Q("numeric", f"What is the total resistance of {r1} Ω and {r2} Ω in series (Ω)?", r1 + r2, "add")
    if k < .6:
        return Q("numeric", f"What is the total resistance of {r1} Ω and {r2} Ω in parallel (Ω)? {SIG}", fmt(r1 * r2 / (r1 + r2)), "R₁R₂/(R₁+R₂)")
    v = rng.choice([5, 9, 12, 24])
    return Q("numeric", f"A voltage divider has {v} V in, R₁ = {r1} Ω on top and R₂ = {r2} Ω to ground. What is the output voltage (V)? {SIG}", fmt(v * r2 / (r1 + r2)), "Vout = Vin·R₂/(R₁+R₂)")


@family("elec.rc", "Capacitors, RC circuits and filters", {"electronics": (5, 9), "electrical_eng": (4, 8), "signal_processing": (3, 7), "control_systems": (1, 3)},
        "Capacitor energy E = ½CV²; charge Q = CV. RC time constant τ = RC; charging: V(t) = V₀(1 − e^(−t/τ)). A first-order RC low-pass filter has cutoff f_c = 1/(2πRC).",
        "Forgetting the 2π in the cutoff frequency.")
def rc(rng, t):
    r, c = rng.choice([1e3, 2.2e3, 10e3, 47e3, 100e3]), rng.choice([1e-9, 10e-9, 100e-9, 1e-6, 10e-6])
    k = rng.random()
    if k < .35:
        return Q("numeric", f"What is the time constant of R = {r / 1e3:g} kΩ and C = {c * 1e6:g} μF, in milliseconds? {SIG}", fmt(r * c * 1e3), "τ = RC")
    if k < .7:
        return Q("numeric", f"What is the cutoff frequency (Hz) of an RC low-pass filter with R = {r / 1e3:g} kΩ and C = {c * 1e6:g} μF? {SIG}", fmt(1 / (2 * math.pi * r * c)), "f_c = 1/2πRC")
    v = rng.choice([3.3, 5, 9, 12])
    return Q("numeric", f"How much energy (in μJ) is stored in a {c * 1e6:g} μF capacitor charged to {v} V? {SIG}", fmt(0.5 * c * v * v * 1e6), "½CV²")


@family("elec.devices", "Transistors, op-amps and converters", {"electronics": (6, 10), "electrical_eng": (5, 9)},
        "BJT: collector current I_C = β·I_B. Inverting op-amp gain = −R_f/R_in; non-inverting gain = 1 + R_f/R_g. An n-bit ADC with reference V_ref has step size V_ref/2ⁿ. Decibels: dB = 20 log₁₀(V_out/V_in).",
        "Forgetting the minus sign for the inverting amplifier.")
def devices(rng, t):
    k = rng.random()
    if k < .3:
        beta, ib = rng.choice([50, 100, 200]), rng.choice([10, 20, 50])
        return Q("numeric", f"A transistor with β = {beta} has base current {ib} μA. What is the collector current (mA)?", fmt(beta * ib / 1000), "I_C = βI_B")
    if k < .55:
        rf, rin = rng.choice([10, 20, 47, 100]), rng.choice([1, 2, 5, 10])
        return Q("numeric", f"An inverting op-amp has R_f = {rf} kΩ and R_in = {rin} kΩ. What is its voltage gain? (include the sign) {SIG}", fmt(-rf / rin), "−R_f/R_in")
    if k < .8:
        n, vref = rng.choice([8, 10, 12, 16]), rng.choice([3.3, 5])
        return Q("numeric", f"A {n}-bit ADC has a {vref} V reference. What is its step size in millivolts? {SIG}", fmt(vref / 2 ** n * 1000), "V_ref/2ⁿ")
    g = rng.choice([2, 10, 100, 1000])
    return Q("numeric", f"What is a voltage gain of {g} in decibels? {SIG}", fmt(20 * math.log10(g)), "20 log₁₀(gain)")


@family("elec.ac", "AC circuits and resonance", {"electronics": (8, 12), "electrical_eng": (6, 12), "physics": (8, 10)},
        "Reactances: X_L = 2πfL, X_C = 1/(2πfC). Series RLC impedance |Z| = √(R² + (X_L − X_C)²). Resonance f₀ = 1/(2π√(LC)). Power factor cosφ = R/|Z|; real power P = V_rms I_rms cosφ.",
        "Adding R, X_L and X_C directly instead of in quadrature.")
def ac(rng, t):
    k = rng.random()
    if k < .4:
        L, C = rng.choice([1e-3, 10e-3, 100e-3]), rng.choice([1e-6, 10e-6, 100e-9])
        return Q("numeric", f"What is the resonant frequency (Hz) of L = {L * 1e3:g} mH and C = {C * 1e6:g} μF? {SIG}", fmt(1 / (2 * math.pi * math.sqrt(L * C))), "f₀ = 1/(2π√LC)")
    if k < .75:
        r, xl, xc = rng.choice([30, 60, 80]), rng.choice([50, 90, 120]), rng.choice([10, 30, 60])
        return Q("numeric", f"A series RLC circuit has R = {r} Ω, X_L = {xl} Ω and X_C = {xc} Ω. What is |Z| (Ω)? {SIG}", fmt(math.hypot(r, xl - xc)), "√(R² + (X_L−X_C)²)")
    v, i, pf = rng.choice([120, 230]), rng.choice([2, 5, 10]), rng.choice([0.6, 0.8, 0.9])
    return Q("numeric", f"A load draws {i} A (rms) from {v} V (rms) with power factor {pf}. What is the real power (W)? {SIG}", fmt(v * i * pf), "P = V I cosφ")


@family("phys.magnetism", "Magnetism", {"physics": (8, 10), "electrical_eng": (7, 10)},
        "Force on a charge: F = qvB sinθ. Force on a wire: F = BIL. Field of a long straight wire: B = μ₀I/(2πr), μ₀ = 4π×10⁻⁷ T·m/A. Induced EMF = −dΦ/dt.",
        "Forgetting the angle factor.")
def magnetism(rng, t):
    k = rng.random()
    if k < .35:
        b, i, l = rng.choice([0.1, 0.5, 1]), rng.randint(1, 10), rng.choice([0.2, 0.5, 1])
        return Q("numeric", f"A {l} m wire carries {i} A perpendicular to a {b} T field. What is the force (N)? {SIG}", fmt(b * i * l), "F = BIL")
    if k < .7:
        i, r = rng.randint(1, 20), rng.choice([0.01, 0.05, 0.1, 0.5])
        return Q("numeric", f"What is the magnetic field (μT) at {r} m from a long straight wire carrying {i} A? {SIG}", fmt(4e-7 * math.pi * i / (2 * math.pi * r) * 1e6), "B = μ₀I/2πr")
    n, dphi, dt = rng.randint(10, 200), rng.choice([0.01, 0.02, 0.05]), rng.choice([0.1, 0.2, 0.5])
    return Q("numeric", f"A coil of {n} turns sees flux change by {dphi} Wb in {dt} s. What is the average induced EMF magnitude (V)? {SIG}", fmt(n * dphi / dt), "EMF = N ΔΦ/Δt")


@family("phys.optics", "Optics", {"physics": (7, 10)},
        "Thin lens: 1/f = 1/dₒ + 1/dᵢ; magnification m = −dᵢ/dₒ. Snell's law n₁ sinθ₁ = n₂ sinθ₂. Critical angle sinθ_c = n₂/n₁.",
        "Measuring angles from the surface instead of the normal.")
def optics(rng, t):
    if rng.random() < .55:
        f, do = rng.choice([10, 15, 20]), rng.choice([30, 40, 60])
        di = 1 / (1 / f - 1 / do)
        return Q("numeric", f"An object is {do} cm from a lens of focal length {f} cm. Where is the image (cm from the lens)? {SIG}", fmt(di), "1/dᵢ = 1/f − 1/dₒ")
    n2, th = rng.choice([1.33, 1.5, 1.52]), rng.choice([20, 30, 45, 60])
    v = math.degrees(math.asin(math.sin(math.radians(th)) / n2))
    return Q("numeric", f"Light in air hits a surface of refractive index {n2} at {th}° to the normal. What is the refraction angle (degrees)? {SIG}", fmt(v), "Snell's law")


@family("phys.gas", "Ideal gases", {"physics": (8, 10), "chemistry": (7, 10), "chemical_eng": (1, 12)},
        "Ideal gas law PV = nRT (R = 8.314 J/mol·K when P in pascals and V in m³; T in kelvin = °C + 273.15). Boyle: P₁V₁ = P₂V₂ at fixed T and n.",
        "Using °C instead of kelvin.")
def gas(rng, t):
    n, T = rng.choice([1, 2, 0.5]), rng.choice([273, 300, 350, 400])
    if rng.random() < .5:
        P = rng.choice([1.0e5, 2.0e5, 1.5e5])
        return Q("numeric", f"What volume (m³) does {n} mol of ideal gas occupy at {P:.0f} Pa and {T} K? (R = 8.314) {SIG}", fmt(n * R_GAS * T / P), "V = nRT/P")
    p1, v1, v2 = rng.choice([100, 200]), rng.choice([2, 4, 6]), rng.choice([1, 3, 8])
    return Q("numeric", f"A gas at {p1} kPa occupies {v1} L. It is compressed isothermally to {v2} L. What is the new pressure (kPa)? {SIG}", fmt(p1 * v1 / v2), "P₁V₁ = P₂V₂")


@family("phys.thermo", "Thermodynamics", {"physics": (8, 11), "mechanical_eng": (5, 12)},
        "First law ΔU = Q − W. Engine efficiency η = W/Q_h = 1 − Q_c/Q_h. Carnot limit η = 1 − T_c/T_h (kelvin). Entropy change ΔS = Q/T.",
        "Using Celsius temperatures in the Carnot formula.")
def thermo(rng, t):
    k = rng.random()
    th, tc = rng.choice([500, 600, 800, 1000]), rng.choice([300, 350])
    if k < .4:
        return Q("numeric", f"What is the maximum (Carnot) efficiency of an engine between {th} K and {tc} K, as a percentage? (number only) {SIG}", fmt((1 - tc / th) * 100), "1 − T_c/T_h")
    if k < .7:
        qh, w = rng.choice([1000, 2000, 5000]), rng.choice([200, 300, 400])
        return Q("numeric", f"An engine takes {qh} J from the hot reservoir and does {w} J of work. How much heat (J) does it reject?", qh - w, "Q_c = Q_h − W")
    q, T = rng.choice([1000, 3000, 6000]), rng.choice([300, 400, 500])
    return Q("numeric", f"{q} J of heat flows into a reservoir at {T} K. What is its entropy change (J/K)? {SIG}", fmt(q / T), "ΔS = Q/T")


@family("phys.rotation", "Rotational dynamics", {"physics": (9, 11), "mechanical_eng": (4, 8)},
        "Torque τ = rF sinθ. Moment of inertia: solid disc ½MR², point mass mr², rod about its end ⅓ML². Angular momentum L = Iω; rotational KE = ½Iω²; τ = Iα.",
        "Using the mass instead of the moment of inertia.")
def rotation(rng, t):
    m, r = rng.randint(1, 10), rng.choice([0.2, 0.5, 1])
    k = rng.random()
    if k < .35:
        return Q("numeric", f"What is the moment of inertia (kg·m²) of a solid disc of mass {m} kg and radius {r} m? {SIG}", fmt(0.5 * m * r * r), "½MR²")
    if k < .7:
        f = rng.randint(5, 50)
        return Q("numeric", f"A force of {f} N is applied perpendicular to a wrench {r} m long. What is the torque (N·m)? {SIG}", fmt(f * r), "τ = rF")
    w = rng.randint(2, 20)
    return Q("numeric", f"A disc of mass {m} kg and radius {r} m spins at {w} rad/s. What is its rotational kinetic energy (J)? {SIG}", fmt(0.5 * (0.5 * m * r * r) * w * w), "½Iω²")


@family("phys.relativity", "Special relativity", {"physics": (10, 12)},
        "Lorentz factor γ = 1/√(1 − v²/c²). Moving clocks run slow: Δt = γΔt₀. Lengths contract: L = L₀/γ. Rest energy E = mc², c = 3.00×10⁸ m/s.",
        "Applying γ the wrong way round.")
def relativity(rng, t):
    beta = rng.choice([0.6, 0.8, 0.5, 0.9])
    g = 1 / math.sqrt(1 - beta ** 2)
    k = rng.random()
    if k < .35:
        return Q("numeric", f"What is the Lorentz factor γ at v = {beta}c? {SIG}", fmt(g), "γ = 1/√(1−β²)")
    if k < .65:
        t0 = rng.choice([1, 2, 5, 10])
        return Q("numeric", f"A clock moving at {beta}c ticks off {t0} s of its own time. How much time passes for a stationary observer (s)? {SIG}", fmt(g * t0), "Δt = γΔt₀")
    m = rng.choice([1e-3, 1e-6, 1e-9])
    return Q("numeric", f"What is the rest energy of {sci(m, 1)} kg of matter (J)? (c = 3.00×10⁸) {SIG}", fmt(m * C_LIGHT ** 2), "E = mc²")


@family("phys.quantum", "Quantum physics", {"physics": (10, 12), "quantum_computing": (1, 4), "chemistry": (11, 12)},
        "Photon energy E = hf = hc/λ (h = 6.626×10⁻³⁴ J·s). de Broglie wavelength λ = h/p. Hydrogen levels Eₙ = −13.6 eV/n²; a transition releases the energy difference. 1 eV = 1.602×10⁻¹⁹ J.",
        "Using the wrong sign for transition energies.")
def quantum(rng, t):
    k = rng.random()
    if k < .3:
        lam = rng.choice([400, 500, 600, 700])
        return Q("numeric", f"What is the energy (in eV) of a photon of wavelength {lam} nm? (h = 6.626×10⁻³⁴, c = 3.00×10⁸) {SIG}", fmt(H_PLANCK * C_LIGHT / (lam * 1e-9) / EV), "E = hc/λ")
    if k < .6:
        n1, n2 = rng.choice([(3, 2), (4, 2), (2, 1), (3, 1), (5, 2)])
        return Q("numeric", f"A hydrogen atom drops from level n = {n1} to n = {n2}. What energy (eV) is released? (Eₙ = −13.6/n²) {SIG}", fmt(13.6 * (1 / n2 ** 2 - 1 / n1 ** 2)), "ΔE = 13.6(1/n₂² − 1/n₁²)")
    if k < .8:
        m, v = rng.choice([(9.11e-31, 1e6), (9.11e-31, 3e6), (1.67e-27, 1e4)])
        return Q("numeric", f"What is the de Broglie wavelength (in pm) of a particle of mass {sci(m)} kg at {sci(v, 1)} m/s? (h = 6.626×10⁻³⁴) {SIG}", fmt(H_PLANCK / (m * v) * 1e12), "λ = h/mv")
    n = rng.randint(1, 4)
    return Q("numeric", f"A qubit starts in |0⟩ and a fair measurement is made after a Hadamard gate. What is the probability of measuring 1? (decimal)", 0.5, "H|0⟩ = (|0⟩+|1⟩)/√2")


@family("phys.nuclear", "Nuclear and gravitational extremes", {"physics": (11, 12), "astrophysics": (8, 12)},
        "Radioactive decay: N = N₀(½)^(t/T½). Mass–energy: E = Δm c². Schwarzschild radius r_s = 2GM/c².",
        "Forgetting that the decay exponent counts half-lives.")
def nuclear(rng, t):
    if rng.random() < .5:
        n0, th, k = rng.choice([800, 1600, 640]), rng.choice([2, 5, 10]), rng.randint(1, 4)
        return Q("numeric", f"A sample has {n0} nuclei and a half-life of {th} days. How many remain after {th * k} days?", fmt(n0 / 2 ** k), "N₀ (½)^k")
    msun = rng.choice([1, 3, 10, 20])
    return Q("numeric", f"What is the Schwarzschild radius (km) of a {msun} solar-mass object? (M☉ = 1.989×10³⁰ kg, G = 6.674×10⁻¹¹, c = 3.00×10⁸) {SIG}",
             fmt(2 * G_CONST * msun * 1.989e30 / C_LIGHT ** 2 / 1000), "r_s = 2GM/c²")


# ===================================================================== CHEMISTRY

MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "S": 32.06, "Cl": 35.45,
        "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38}
ELEMENTS = [("H", 1), ("He", 2), ("Li", 3), ("C", 6), ("N", 7), ("O", 8), ("F", 9), ("Ne", 10), ("Na", 11), ("Mg", 12), ("Al", 13), ("Si", 14), ("S", 16), ("Cl", 17), ("K", 19), ("Ca", 20)]
COMPOUNDS = ["H2O", "CO2", "NaCl", "CaCO3", "H2SO4", "NaOH", "Mg(OH)2", "Al2O3", "C6H12O6", "NH3", "CH4", "HCl", "Fe2O3", "CuSO4", "Ca(OH)2", "KCl", "C2H5OH", "Al(OH)3"]


def molar_mass(formula: str) -> float:
    def parse(s, i=0):
        total, n = 0.0, len(s)
        while i < n:
            if s[i] == "(":
                inner, i = parse(s, i + 1)
                m = re.match(r"\d+", s[i:])
                mult = int(m.group()) if m else 1
                i += len(m.group()) if m else 0
                total += inner * mult
            elif s[i] == ")":
                return total, i + 1
            else:
                m = re.match(r"([A-Z][a-z]?)(\d*)", s[i:])
                total += MASS[m.group(1)] * (int(m.group(2)) if m.group(2) else 1)
                i += len(m.group())
        return total, i
    return parse(formula)[0]


def pretty(formula: str) -> str:
    return re.sub(r"(\d+)", lambda m: "".join("₀₁₂₃₄₅₆₇₈₉"[int(c)] for c in m.group()), formula)


@family("chem.atoms", "Atoms and ions", {"chemistry": (2, 4)},
        "An atom has Z protons (the atomic number), N neutrons and Z electrons when neutral. Mass number A = Z + N. An ion has gained (negative) or lost (positive) electrons.",
        "Changing the number of protons when forming an ion.")
def atoms(rng, t):
    sym, z = rng.choice(ELEMENTS[3:])
    a = 2 * z + rng.randint(0, 2) if z > 2 else z + 1
    if rng.random() < .5:
        return Q("numeric", f"An atom of {sym} has atomic number {z} and mass number {a}. How many neutrons does it have?", a - z, "N = A − Z")
    ch = rng.choice([-2, -1, 1, 2])
    return Q("numeric", f"An ion of {sym} has {z} protons and a charge of {ch:+d}. How many electrons does it have?", z - ch, "electrons = protons − charge")


@family("chem.molarmass", "Molar mass", {"chemistry": (4, 7), "chemical_eng": (1, 2)},
        "Molar mass (g/mol) is the sum of the atomic masses in the formula, counting subscripts and brackets: Mg(OH)₂ = 24.305 + 2(15.999 + 1.008).",
        "Forgetting to multiply the contents of brackets.")
def molarmass(rng, t):
    f = rng.choice(COMPOUNDS[:8] if t < .4 else COMPOUNDS)
    return Q("numeric", f"What is the molar mass of {pretty(f)} in g/mol? (atomic masses: " + ", ".join(f"{k} {v}" for k, v in MASS.items() if k in re.findall(r"[A-Z][a-z]?", f)) + f") {SIG}",
             fmt(molar_mass(f)), "sum of atomic masses")


@family("chem.mole", "The mole", {"chemistry": (5, 8), "chemical_eng": (1, 3)},
        "n = m/M (moles = mass ÷ molar mass). One mole contains 6.022×10²³ particles (Avogadro's number).",
        "Using grams where moles are needed.")
def mole(rng, t):
    f = rng.choice(COMPOUNDS[:8])
    m = rng.choice([5, 10, 18, 36, 50, 100])
    if rng.random() < .6:
        return Q("numeric", f"How many moles are in {m} g of {pretty(f)}? (M = {molar_mass(f):.3f} g/mol) {SIG}", fmt(m / molar_mass(f)), "n = m/M")
    n = rng.choice([0.5, 2, 3])
    return Q("numeric", f"How many molecules are in {n} mol of a substance? (Nₐ = 6.022×10²³) {SIG}", fmt(n * 6.022e23), "N = n·Nₐ")


REACTIONS = [
    ("2H₂ + O₂ → 2H₂O", [2, 1, 2], ("H2", "H2O"), (2, 2)),
    ("N₂ + 3H₂ → 2NH₃", [1, 3, 2], ("H2", "NH3"), (3, 2)),
    ("CH₄ + 2O₂ → CO₂ + 2H₂O", [1, 2, 1, 2], ("CH4", "CO2"), (1, 1)),
    ("2Mg + O₂ → 2MgO", [2, 1, 2], None, None),
    ("C₃H₈ + 5O₂ → 3CO₂ + 4H₂O", [1, 5, 3, 4], ("C3H8", "CO2"), None),
    ("4Al + 3O₂ → 2Al₂O₃", [4, 3, 2], None, None),
    ("2NaOH + H₂SO₄ → Na₂SO₄ + 2H₂O", [2, 1, 1, 2], None, None),
]


@family("chem.balancing", "Balancing equations", {"chemistry": (5, 7)},
        "A balanced equation has the same number of each kind of atom on both sides — adjust the coefficients (never the subscripts). Example: 2H₂ + O₂ → 2H₂O.",
        "Changing subscripts instead of coefficients.")
def balancing(rng, t):
    eq, co, _, _ = rng.choice(REACTIONS)
    unbal = re.sub(r"(?<![\w₀-₉])\d+(?=[A-Z])", "", eq)
    return Q("numeric", f"Balance {unbal} using the smallest whole-number coefficients. What is the sum of all the coefficients?", sum(co), f"balanced: {eq}")


@family("chem.stoichiometry", "Stoichiometry", {"chemistry": (6, 10), "chemical_eng": (3, 12)},
        "Use the balanced equation's coefficients as mole ratios: moles of product = moles of reactant × (product coeff / reactant coeff). Convert grams → moles → moles → grams. The reactant that runs out first is the limiting reagent.",
        "Using mass ratios instead of mole ratios.")
def stoich(rng, t):
    rx = rng.choice([r for r in REACTIONS if r[2] and r[3]])
    eq, co, (a, b), (ca, cb) = rx
    mass = rng.choice([4, 8, 12, 20, 40])
    n = mass / molar_mass(a) * cb / ca
    return Q("numeric", f"For {eq}: how many grams of {pretty(b)} form from {mass} g of {pretty(a)} (excess of the other reactant)? (M({pretty(a)}) = {molar_mass(a):.3f}, M({pretty(b)}) = {molar_mass(b):.3f} g/mol) {SIG}",
             fmt(n * molar_mass(b)), "grams → moles → mole ratio → grams")


@family("chem.solutions", "Solutions and dilution", {"chemistry": (6, 9)},
        "Molarity M = moles of solute ÷ litres of solution. Dilution: M₁V₁ = M₂V₂.",
        "Using millilitres without converting to litres.")
def solutions(rng, t):
    if rng.random() < .5:
        n, v = rng.choice([0.5, 1, 2, 0.25]), rng.choice([0.5, 1, 2, 4])
        return Q("numeric", f"What is the molarity of a solution with {n} mol of solute in {v} L? (mol/L) {SIG}", fmt(n / v), "M = n/V")
    m1, v1, v2 = rng.choice([2, 4, 6, 12]), rng.choice([10, 25, 50]), rng.choice([100, 250, 500])
    return Q("numeric", f"{v1} mL of {m1} M solution is diluted to {v2} mL. What is the new molarity (mol/L)? {SIG}", fmt(m1 * v1 / v2), "M₁V₁ = M₂V₂")


@family("chem.acids", "Acids, bases and pH", {"chemistry": (7, 10)},
        "pH = −log₁₀[H⁺]; pOH = −log₁₀[OH⁻]; pH + pOH = 14. Strong acids dissociate fully. Weak acid: [H⁺] ≈ √(Kₐ·C). Buffer: pH = pKₐ + log([A⁻]/[HA]).",
        "Forgetting the negative sign in the definition of pH.")
def acids(rng, t):
    k = rng.random()
    if k < .35:
        e = rng.randint(1, 6)
        return Q("numeric", f"What is the pH of a solution with [H⁺] = 1×10^-{e} mol/L?", e, "pH = −log[H⁺]")
    if k < .6:
        c = rng.choice([0.1, 0.01, 0.001])
        return Q("numeric", f"What is the pH of {c} M NaOH (a strong base)? {SIG}", fmt(14 + math.log10(c)), "pOH = −log[OH⁻]; pH = 14 − pOH")
    if k < .85:
        ka, c = rng.choice([1.8e-5, 1.0e-5, 4.0e-6]), rng.choice([0.1, 0.01])
        return Q("numeric", f"What is the pH of a {c} M weak acid with Kₐ = {sci(ka, 2)}? (use [H⁺] ≈ √(KₐC)) {SIG}", fmt(-math.log10(math.sqrt(ka * c))), "pH = −log√(KₐC)")
    pka, r = rng.choice([4.76, 7.2, 9.25]), rng.choice([10, 100, 0.1])
    return Q("numeric", f"A buffer has pKₐ = {pka} and [A⁻]/[HA] = {r}. What is its pH? {SIG}", fmt(pka + math.log10(r)), "Henderson–Hasselbalch")


@family("chem.thermochem", "Thermochemistry", {"chemistry": (8, 11)},
        "Heat absorbed by water q = mcΔT (c = 4.184 J/g·°C). Reaction enthalpy ΔH = ΣΔH_f(products) − ΣΔH_f(reactants). Hess's law: ΔH for a sum of reactions is the sum of their ΔH values.",
        "Subtracting in the wrong order (products − reactants).")
def thermochem(rng, t):
    if rng.random() < .5:
        m, dt = rng.choice([50, 100, 200]), rng.randint(3, 25)
        return Q("numeric", f"How much heat (J) raises {m} g of water by {dt} °C? (c = 4.184 J/g·°C) {SIG}", fmt(m * 4.184 * dt), "q = mcΔT")
    p1, p2, r1 = rng.randint(-400, -100), rng.randint(-300, -50), rng.randint(-200, 0)
    return Q("numeric", f"Products have ΔH_f values {p1} and {p2} kJ/mol; the single reactant has {r1} kJ/mol (all coefficients 1). What is ΔH of the reaction (kJ)?", p1 + p2 - r1, "ΣΔH_f(products) − ΣΔH_f(reactants)")


@family("chem.kinetics", "Reaction rates", {"chemistry": (8, 11)},
        "For a first-order reaction: rate = k[A], half-life t½ = ln 2/k = 0.693/k, [A] = [A]₀e^(−kt). Rate law rate = k[A]^m[B]^n; doubling [A] multiplies the rate by 2^m.",
        "Thinking half-life depends on concentration for first-order reactions.")
def kinetics(rng, t):
    k = rng.random()
    if k < .4:
        kk = rng.choice([0.01, 0.05, 0.1, 0.0231, 0.0693])
        return Q("numeric", f"A first-order reaction has k = {kk} s⁻¹. What is its half-life (s)? {SIG}", fmt(math.log(2) / kk), "t½ = ln2/k")
    if k < .7:
        c0, n = rng.choice([1.0, 0.8, 0.5]), rng.randint(1, 5)
        return Q("numeric", f"A first-order reactant starts at {c0} mol/L. What remains after {n} half-lives? (mol/L) {SIG}", fmt(c0 / 2 ** n), "halve each half-life")
    m = rng.choice([1, 2, 3])
    return Q("numeric", f"A reaction is order {m} in A. If [A] is tripled, by what factor does the rate change?", 3 ** m, "3^m")


@family("chem.equilibrium", "Chemical equilibrium", {"chemistry": (9, 11)},
        "For aA + bB ⇌ cC + dD, K = [C]^c[D]^d / ([A]^a[B]^b) at equilibrium. If Q < K the reaction goes forward; if Q > K it goes back. K is unchanged by concentration; it changes only with temperature.",
        "Including solids and pure liquids in K.")
def equilibrium(rng, t):
    a, b, c = rng.choice([2, 3, 4]), rng.choice([1, 2]), rng.choice([2, 4, 6])
    return Q("numeric", f"For A + B ⇌ C at equilibrium, [A] = {a} M, [B] = {b} M and [C] = {c} M. What is K? {SIG}", fmt(c / (a * b)), "K = [C]/([A][B])")


@family("chem.electro", "Electrochemistry", {"chemistry": (9, 11)},
        "E°cell = E°cathode − E°anode (a positive value means spontaneous). Faraday's law: mass = (I·t·M)/(n·F), F = 96485 C/mol e⁻.",
        "Flipping the sign of the anode potential twice.")
def electro(rng, t):
    cat, an = rng.choice([(0.34, -0.76), (0.80, 0.34), (0.34, -0.44), (0.80, -0.76)])
    if rng.random() < .5:
        return Q("numeric", f"A cell has cathode E° = {cat} V and anode E° = {an} V. What is E°cell (V)? {SIG}", fmt(cat - an), "E°cathode − E°anode")
    i, tm, n, mm = rng.choice([2, 5]), rng.choice([600, 1800, 3600]), 2, 63.55
    return Q("numeric", f"A current of {i} A runs for {tm} s depositing copper (Cu²⁺ + 2e⁻ → Cu, M = 63.55 g/mol). What mass (g) is deposited? (F = 96485) {SIG}", fmt(i * tm * mm / (n * 96485)), "m = ItM/nF")


@family("chem.gibbs", "Free energy", {"chemistry": (10, 12)},
        "ΔG = ΔH − TΔS. A reaction is spontaneous when ΔG < 0. At equilibrium ΔG = 0, so T = ΔH/ΔS. ΔG° = −RT ln K.",
        "Mixing kJ with J when ΔS is given in J/K.")
def gibbs(rng, t):
    dh, ds, T = rng.choice([-100, -50, 40, 80]), rng.choice([-0.1, 0.05, 0.2, 0.3]), rng.choice([298, 350, 500])
    if rng.random() < .6:
        return Q("numeric", f"ΔH = {dh} kJ/mol, ΔS = {ds} kJ/(mol·K), T = {T} K. What is ΔG (kJ/mol)? {SIG}", fmt(dh - T * ds), "ΔG = ΔH − TΔS")
    k = rng.choice([10, 100, 1000])
    return Q("numeric", f"At 298 K an equilibrium has K = {k}. What is ΔG° (kJ/mol)? (R = 8.314 J/mol·K) {SIG}", fmt(-R_GAS * 298 * math.log(k) / 1000), "ΔG° = −RT ln K")


# ===================================================================== ASTRONOMY

@family("astro.light", "Distances and light travel", {"astronomy": (1, 4), "astrophysics": (1, 3)},
        "Light travels 3.00×10⁸ m/s. A light-year is the distance light covers in a year (9.46×10¹⁵ m). 1 AU = 1.496×10¹¹ m is the Earth–Sun distance.",
        "Treating a light-year as a unit of time.")
def astro_light(rng, t):
    if rng.random() < .5:
        d = rng.choice([("the Sun", 1.496e11), ("Mars at its closest", 5.6e10), ("the Moon", 3.84e8), ("Jupiter", 6.2e11)])
        return Q("numeric", f"Light takes how many seconds to travel from Earth to {d[0]} (distance {sci(d[1])} m)? (c = 3.00×10⁸) {SIG}", fmt(d[1] / C_LIGHT), "t = d/c")
    ly = rng.choice([4.2, 8.6, 25, 100, 2500])
    return Q("numeric", f"A star is {ly} light-years away. How many years does its light take to reach us?", fmt(ly), "1 ly = 1 year of light travel")


@family("astro.kepler", "Kepler's third law and orbits", {"astronomy": (3, 8), "astrophysics": (2, 5), "aerospace_eng": (4, 12)},
        "Kepler's third law: T² = a³ when T is in years and a in AU (for orbits around the Sun). Circular orbital speed v = √(GM/r).",
        "Forgetting to take the square root at the end.")
def kepler(rng, t):
    a = rng.choice([0.387, 1.524, 5.203, 9.537, 19.19, 30.07, 4.0, 9.0])
    if rng.random() < .6:
        return Q("numeric", f"A planet orbits the Sun at {a} AU. What is its orbital period in years? {SIG}", fmt(a ** 1.5), "T = a^(3/2)")
    T = rng.choice([8, 27, 64, 125])
    return Q("numeric", f"A body orbits the Sun with period {T} years. What is its semi-major axis in AU?", fmt(T ** (2 / 3)), "a = T^(2/3)")


@family("astro.stars", "Stellar properties", {"astronomy": (4, 10), "astrophysics": (3, 8)},
        "Wien's law: λ_max = b/T with b = 2.898×10⁻³ m·K. Stellar luminosity L = 4πR²σT⁴, so L/L☉ = (R/R☉)²(T/T☉)⁴ with T☉ = 5772 K. Parallax: d (parsecs) = 1/p (arcseconds).",
        "Forgetting to square the radius and use the fourth power of temperature.")
def stars(rng, t):
    k = rng.random()
    if k < .3:
        T = rng.choice([3000, 5772, 10000, 20000])
        return Q("numeric", f"At what wavelength (nm) does a {T} K blackbody emit most strongly? (b = 2.898×10⁻³ m·K) {SIG}", fmt(2.898e-3 / T * 1e9), "λ = b/T")
    if k < .55:
        p = rng.choice([0.5, 0.25, 0.1, 0.05, 0.2])
        return Q("numeric", f"A star has a parallax of {p} arcseconds. What is its distance in parsecs?", fmt(1 / p), "d = 1/p")
    r, tt = rng.choice([1, 2, 3, 10]), rng.choice([1, 2])
    return Q("numeric", f"A star has radius {r} R☉ and temperature {tt}× the Sun's. What is its luminosity in L☉? {SIG}", fmt(r ** 2 * tt ** 4), "L ∝ R²T⁴")


@family("astro.cosmo", "Distances, magnitudes and the expanding universe", {"astronomy": (8, 12), "astrophysics": (6, 12)},
        "Hubble's law v = H₀d with H₀ ≈ 70 km/s/Mpc. Distance modulus m − M = 5 log₁₀(d/10 pc). A difference of 5 magnitudes is a factor of 100 in brightness. Age of the universe ≈ 1/H₀ ≈ 977.8/H₀ Gyr (H₀ in km/s/Mpc).",
        "Mixing parsecs with megaparsecs.")
def cosmo(rng, t):
    k = rng.random()
    if k < .35:
        d = rng.choice([10, 50, 100, 300, 1000])
        return Q("numeric", f"A galaxy is {d} Mpc away. What is its recession speed (km/s) with H₀ = 70 km/s/Mpc?", d * 70, "v = H₀d")
    if k < .65:
        d = rng.choice([10, 100, 1000, 10000])
        return Q("numeric", f"A star is {d} parsecs away. What is its distance modulus m − M? {SIG}", fmt(5 * math.log10(d / 10)), "5 log₁₀(d/10)")
    if k < .85:
        dm = rng.choice([1, 2, 5, 10])
        return Q("numeric", f"Star A is {dm} magnitudes brighter than star B. How many times brighter is it? {SIG}", fmt(100 ** (dm / 5)), "100^(Δm/5)")
    h = rng.choice([50, 70, 100])
    return Q("numeric", f"With H₀ = {h} km/s/Mpc, what is 1/H₀ in billions of years? {SIG}", fmt(977.8 / h), "977.8/H₀")


# ===================================================================== ENGINEERING

@family("eng.materials", "Stress, strain and materials", {"mechanical_eng": (3, 12), "civil_eng": (2, 12), "materials_science": (1, 12), "biomedical_eng": (1, 12)},
        "Stress σ = F/A (Pa). Strain ε = ΔL/L (no units). Young's modulus E = σ/ε, so ΔL = FL/(AE). A material fails when stress exceeds its yield strength. FCC crystals: 4 atoms per cell, packing fraction 0.74.",
        "Using diameter instead of area for stress.")
def materials(rng, t):
    k = rng.random()
    f, a = rng.choice([1000, 5000, 20000]), rng.choice([1e-4, 5e-4, 1e-3])
    if k < .35:
        return Q("numeric", f"A {f} N load pulls on a bar of cross-section {sci(a, 1)} m². What is the stress (MPa)? {SIG}", fmt(f / a / 1e6), "σ = F/A")
    if k < .7:
        e, l = rng.choice([200e9, 70e9, 110e9]), rng.choice([1, 2, 3])
        return Q("numeric", f"A {l} m bar (E = {sci(e)} Pa, area {sci(a, 1)} m²) carries a tensile load of {f} N. How much does it stretch (mm)? {SIG}", fmt(f * l / (a * e) * 1000), "ΔL = FL/(AE)")
    return Q("numeric", f"How many atoms are in one face-centred cubic (FCC) unit cell?", 4, "8 corners × 1/8 + 6 faces × 1/2 = 4")


@family("eng.statics", "Statics and beams", {"mechanical_eng": (1, 12), "civil_eng": (1, 12), "aerospace_eng": (1, 3)},
        "For equilibrium: ΣF = 0 and ΣM = 0 about any point. A simply supported beam of length L with a point load P at distance a from the left support: right reaction R_B = P·a/L, left reaction R_A = P − R_B.",
        "Taking moments about the wrong point or forgetting the lever arm.")
def statics(rng, t):
    L, P = rng.choice([4, 6, 8, 10]), rng.choice([10, 20, 40, 60])
    a = rng.choice([1, 2, 3])
    if rng.random() < .5:
        return Q("numeric", f"A {L} m simply supported beam carries a {P} kN point load {a} m from the left support. What is the right support reaction (kN)? {SIG}", fmt(P * a / L), "ΣM about the left support")
    w = rng.choice([2, 5, 10])
    return Q("numeric", f"A {L} m simply supported beam carries a uniform load of {w} kN/m. What is each support reaction (kN)?", fmt(w * L / 2), "wL/2")


@family("eng.machines", "Machines, gears and efficiency", {"mechanical_eng": (1, 5), "robotics": (1, 4)},
        "Gear ratio = driven teeth ÷ driver teeth; speed falls and torque rises by that ratio. Efficiency η = useful output ÷ input. Mechanical advantage = load ÷ effort.",
        "Applying the ratio the wrong way round.")
def machines(rng, t):
    k = rng.random()
    if k < .5:
        t1, t2, w = rng.choice([10, 12, 15, 20]), rng.choice([30, 40, 60]), rng.choice([600, 1200, 3000])
        return Q("numeric", f"A {t1}-tooth gear drives a {t2}-tooth gear turning at {w} rpm. What is the speed of the driven gear (rpm)? {SIG}", fmt(w * t1 / t2), "ω₂ = ω₁ T₁/T₂")
    p_in, eta = rng.choice([100, 500, 1000]), rng.choice([0.6, 0.8, 0.9])
    return Q("numeric", f"A machine takes in {p_in} W with efficiency {eta}. What is its useful output (W)? {SIG}", fmt(p_in * eta), "η × input")


@family("eng.fluids", "Fluids and heat transfer", {"mechanical_eng": (5, 12), "civil_eng": (3, 12), "chemical_eng": (4, 12), "aerospace_eng": (3, 12)},
        "Continuity: A₁v₁ = A₂v₂. Bernoulli: P + ½ρv² + ρgh = constant. Conduction: Q/t = kAΔT/L. Lift L = ½ρv²S·C_L.",
        "Using diameters instead of areas in the continuity equation.")
def fluids(rng, t):
    k = rng.random()
    if k < .3:
        a1, v1, a2 = rng.choice([0.04, 0.02]), rng.choice([1, 2, 3]), rng.choice([0.01, 0.005])
        return Q("numeric", f"Water flows at {v1} m/s through a pipe of area {a1} m² into a section of area {a2} m². What is the new speed (m/s)? {SIG}", fmt(a1 * v1 / a2), "A₁v₁ = A₂v₂")
    if k < .55:
        kk, a, dT, L = rng.choice([0.8, 205, 0.04]), rng.choice([1, 2, 10]), rng.choice([10, 20, 40]), rng.choice([0.1, 0.2, 0.5])
        return Q("numeric", f"A wall of conductivity {kk} W/m·K, area {a} m² and thickness {L} m has a {dT} K temperature difference across it. What is the heat flow (W)? {SIG}", fmt(kk * a * dT / L), "Q/t = kAΔT/L")
    if k < .8:
        v, s, cl = rng.choice([50, 70, 100]), rng.choice([10, 16, 30]), rng.choice([0.4, 0.8, 1.2])
        return Q("numeric", f"What lift (kN) does a wing of area {s} m², C_L = {cl} produce at {v} m/s? (ρ = 1.225 kg/m³) {SIG}", fmt(0.5 * 1.225 * v * v * s * cl / 1000), "L = ½ρv²SC_L")
    ve, m0, mf = rng.choice([2500, 3000, 4500]), rng.choice([100, 200, 500]), rng.choice([20, 50, 100])
    mf = min(mf, m0 // 2)
    return Q("numeric", f"A rocket with exhaust speed {ve} m/s goes from {m0} t to {mf} t. What is its Δv (m/s)? (Tsiolkovsky) {SIG}", fmt(ve * math.log(m0 / mf)), "Δv = vₑ ln(m₀/m_f)")


@family("eng.control", "Control systems", {"control_systems": (1, 12), "robotics": (5, 9)},
        "First-order system G = K/(τs + 1): time constant τ, DC gain K, step response y = K(1 − e^(−t/τ)) reaches 63% at t = τ. Second-order s² + 2ζωₙs + ωₙ²: natural frequency ωₙ = √(constant term), damping ζ = (coefficient of s)/(2ωₙ). Poles are the roots of the denominator; stable if all have negative real part.",
        "Calling a system stable when a pole has positive real part.")
def control(rng, t):
    k = rng.random()
    if k < .3:
        K, tau, tm = rng.choice([1, 2, 5]), rng.choice([0.5, 1, 2, 4]), rng.choice([1, 2, 3])
        return Q("numeric", f"A system has G(s) = {K}/({tau}s + 1). What is its step response value at t = {tm * tau} s? {SIG}", fmt(K * (1 - math.exp(-tm))), "K(1 − e^(−t/τ))")
    if k < .6:
        wn = rng.choice([2, 3, 4, 5, 10])
        z = rng.choice([0.2, 0.5, 0.7, 1.0])
        return Q("numeric", f"For s² + {2 * z * wn:g}s + {wn * wn}, what is the damping ratio ζ? {SIG}", fmt(z), "ζ = b/(2ωₙ)")
    p, q = rng.sample(range(1, 9), 2)
    return Q("numeric", f"The characteristic polynomial is s² + {p + q}s + {p * q}. What is the pole closest to zero (give the negative number)?", -min(p, q), "factor (s + p)(s + q)")


@family("eng.signals", "Sampling and spectra", {"signal_processing": (2, 12), "control_systems": (2, 4)},
        "Nyquist: sample at more than twice the highest frequency, f_s > 2f_max. A DFT of N samples at rate f_s has frequency resolution Δf = f_s/N. Decibels for power: 10 log₁₀(P₂/P₁).",
        "Sampling exactly at the Nyquist rate (it must be strictly higher).")
def signals(rng, t):
    k = rng.random()
    if k < .4:
        f = rng.choice([100, 440, 4000, 20000])
        return Q("numeric", f"What is the minimum sampling rate (Hz) needed to capture a signal whose highest frequency is {f} Hz? (the Nyquist rate)", 2 * f, "f_s = 2 f_max")
    if k < .75:
        fs, n = rng.choice([8000, 44100, 48000]), rng.choice([256, 1024, 4096])
        return Q("numeric", f"A DFT uses N = {n} samples at {fs} Hz. What is the frequency resolution (Hz)? {SIG}", fmt(fs / n), "Δf = f_s/N")
    p = rng.choice([2, 10, 100, 1000])
    return Q("numeric", f"A system's output power is {p}× its input power. What is the gain in dB? {SIG}", fmt(10 * math.log10(p)), "10 log₁₀")
