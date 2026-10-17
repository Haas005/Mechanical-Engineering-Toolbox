"""Checks the vibration solver against numerical integration and textbook results.
Run: python test_vibration_solver.py"""
import math

import numpy as np
from scipy.integrate import solve_ivp

from vibration_solver import (Load, Params, damping_for_ratio, magnification, phase_lag,
                              properties, resonance_peak, simulate, suggest_duration,
                              suggest_samples, transmissibility)

ok = True


def report(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")


def check(name, got, want, tol=1e-9):
    report(name, abs(got - want) <= tol * max(abs(want), 1e-12), f"got {got:.8g}, expected {want:.8g}")


def check_small(name, got, tol):
    report(name, abs(got) <= tol, f"got {got:.3g}, must be within {tol:g} of 0")


def reference(p, load, x0, v0, t):
    """Independent answer from a numerical ODE solver."""
    def f(tt, y):
        if load.kind == "harmonic":
            F = load.F0 * math.cos(load.omega * tt)
        elif load.kind == "step":
            F = load.F0
        else:
            F = 0.0
        return [y[1], (F - p.c * y[1] - p.k * y[0]) / p.m]
    s = solve_ivp(f, (0, t[-1]), [x0, v0], method="DOP853", t_eval=t, rtol=1e-12, atol=1e-14)
    return s.y[0], s.y[1]


m, k = 2.0, 800.0                      # wn = 20 rad/s
wn = math.sqrt(k / m)

# 1. Derived properties
pr = properties(Params(m, damping_for_ratio(m, k, 0.2), k))
check("Natural frequency wn = sqrt(k/m)", pr.wn, 20.0)
check("Natural frequency in Hz", pr.fn, 20.0 / (2 * math.pi))
check("Damping ratio from c", pr.zeta, 0.2)
check("Critical damping c_cc = 2 sqrt(k m)", pr.c_crit, 2 * math.sqrt(k * m))
check("Damped frequency wd = wn sqrt(1 - z^2)", pr.wd, 20 * math.sqrt(1 - 0.04))
check("Log decrement = 2 pi z / sqrt(1 - z^2)", pr.log_decrement, 2 * math.pi * 0.2 / math.sqrt(0.96))
check("Quality factor Q = 1 / (2 z)", pr.Q, 2.5)
for z, name in ((0.0, "Undamped"), (0.5, "Underdamped"), (1.0, "Critically damped"), (2.0, "Overdamped")):
    got = properties(Params(m, damping_for_ratio(m, k, z), k)).regime
    report(f"Regime for zeta = {z}", got == name, got)

# 2. Free response against numerical integration, all four regimes
rng = np.random.default_rng(5)
for z in (0.0, 0.05, 0.3, 0.9, 0.999999, 1.0, 1.000001, 1.5, 4.0):
    worst = 0.0
    for _ in range(5):
        p = Params(rng.uniform(0.5, 5), 0.0, rng.uniform(100, 2000))
        p.c = damping_for_ratio(p.m, p.k, z)
        x0, v0 = rng.uniform(-0.1, 0.1), rng.uniform(-2, 2)
        T = 6 * 2 * math.pi / math.sqrt(p.k / p.m)
        r = simulate(p, Load("free"), x0, v0, T, 1500)
        xr, vr = reference(p, Load("free"), x0, v0, r.t)
        scale = max(abs(x0), abs(v0) / math.sqrt(p.k / p.m))
        worst = max(worst, np.max(np.abs(r.x - xr)) / scale, np.max(np.abs(r.v - vr)) / (scale * math.sqrt(p.k / p.m)))
    check_small(f"Free response matches ODE solver, zeta = {z} (5 random systems)", worst, 1e-7)

# 3. Initial conditions are honoured
p = Params(m, damping_for_ratio(m, k, 0.3), k)
for load in (Load("free"), Load("step", 50.0), Load("harmonic", 30.0, 15.0)):
    r = simulate(p, load, 0.04, -0.7, 1.0, 500)
    check_small(f"x(0) = x0 and v(0) = v0 ({load.kind})", abs(r.x[0] - 0.04) + abs(r.v[0] + 0.7), 1e-12)

# 4. Undamped period and energy
p0 = Params(m, 0.0, k)
r = simulate(p0, Load("free"), 0.05, 0.0, 5.0, 100001)
up = np.where((r.x[:-1] < 0) & (r.x[1:] >= 0))[0]
t_up = r.t[up] - r.x[up] * (r.t[up + 1] - r.t[up]) / (r.x[up + 1] - r.x[up])
check("Undamped period = 2 pi / wn", float(np.mean(np.diff(t_up))), 2 * math.pi / wn, 1e-6)
tot = r.kinetic + r.potential
check_small("Undamped: total energy is constant", float(np.ptp(tot) / tot[0]), 1e-12)

# 5. Log decrement and exact decay per cycle (a property of the solution, checked directly)
for z in (0.02, 0.1, 0.4):
    p = Params(m, damping_for_ratio(m, k, z), k)
    pp = properties(p)
    t = np.array([0.13, 0.13 + pp.Td])
    r = simulate(p, Load("free"), 0.05, 0.3, 1.0, 2)   # only to exercise the call
    from vibration_solver import _homogeneous
    xa, _ = _homogeneous(t, pp.wn, pp.zeta, 0.05, 0.3)
    check(f"Amplitude ratio one damped period apart = exp(delta), zeta = {z}", math.log(xa[0] / xa[1]),
          pp.log_decrement, 1e-9)

# 6. Critical and over damping do not oscillate
for z, max_cross in ((1.0, 0), (2.5, 0)):
    p = Params(m, damping_for_ratio(m, k, z), k)
    r = simulate(p, Load("free"), 0.05, 0.0, 2.0, 5000)
    crossings = int(np.sum(np.diff(np.sign(r.x)) != 0))
    report(f"Released from rest, zeta = {z}: no oscillation and never crosses zero",
           crossings == max_cross and np.all(np.diff(r.x) <= 1e-15), f"{crossings} crossings")
p = Params(m, damping_for_ratio(m, k, 1.0), k)
t_c = simulate(p, Load("free"), 0.05, 0.0, 4.0, 40001)
slow = simulate(Params(m, damping_for_ratio(m, k, 3.0), k), Load("free"), 0.05, 0.0, 4.0, 40001)
tol_x = 0.05 * 0.02
reached = lambda r_: r_.t[np.argmax(np.abs(r_.x) < tol_x)] if np.any(np.abs(r_.x) < tol_x) else math.inf
report("Critical damping returns to rest faster than heavy overdamping",
       reached(t_c) < reached(slow) < math.inf, f"{reached(t_c):.3f} s versus {reached(slow):.3f} s")

# 7. Energy lost equals the work done by the damper: E(0) - E(T) = integral of c v^2 dt
p = Params(m, damping_for_ratio(m, k, 0.15), k)
r = simulate(p, Load("free"), 0.05, 0.4, 3.0, 60001)
E = r.kinetic + r.potential
err = lambda n_: (lambda q: float(np.max(np.abs((q.kinetic + q.potential)[0] - (q.kinetic + q.potential) - q.dissipated))
                                      / (q.kinetic + q.potential)[0]))(simulate(p, Load("free"), 0.05, 0.4, 3.0, n_))
e1, e2 = err(30001), err(60001)
check_small("Free: E(0) - E(t) = energy dissipated by the damper (max over time, relative)", e2, 1e-6)
report("   ... and the mismatch shrinks about 4x when the time step is halved (it is integration error only)",
       3.0 < e1 / e2 < 5.0, f"{e1:.2e} -> {e2:.2e}")

# 8. Step response: final value, overshoot, peak time, settling
for z in (0.1, 0.25, 0.5, 0.8):
    p = Params(m, damping_for_ratio(m, k, z), k)
    pp = properties(p)
    F0 = 100.0
    xs = F0 / k
    r = simulate(p, Load("step", F0), 0.0, 0.0, 30.0, 120001)
    check(f"Step: final value = F0 / k, zeta = {z}", r.x[-1], xs, 1e-6)
    check(f"Step: overshoot = exp(-pi z / sqrt(1 - z^2)), zeta = {z}", (r.x.max() - xs) / xs,
          math.exp(-math.pi * z / math.sqrt(1 - z * z)), 2e-5)
    short = simulate(p, Load("step", F0), 0.0, 0.0, 1.0, 400001)
    check(f"Step: time of first peak = pi / wd, zeta = {z}", float(short.t[np.argmax(short.x)]), math.pi / pp.wd, 1e-4)
    check(f"Step: overshoot property agrees, zeta = {z}", pp.overshoot, (r.x.max() - xs) / xs, 2e-5)
for z in (0.05, 0.2, 0.4):
    p = Params(m, damping_for_ratio(m, k, z), k)
    pp = properties(p)
    r = simulate(p, Load("step", 100.0), 0.0, 0.0, 6 * pp.settling_time, 60001)
    late = r.t >= pp.settling_time
    report(f"Step: stays within 2 % after 4 / (z wn), zeta = {z}",
           np.max(np.abs(r.x[late] - 100.0 / k)) <= 0.02 * 100.0 / k + 1e-12,
           f"worst {np.max(np.abs(r.x[late] - 100.0 / k)) / (100.0 / k) * 100:.2f} %")


# 8b. Settling time for critical and over damping (found from the exact solution)
for z in (1.0, 1.5, 3.0):
    p = Params(m, damping_for_ratio(m, k, z), k)
    pp = properties(p)
    r = simulate(p, Load("step", 100.0), 0.0, 0.0, 3 * pp.settling_time, 200001)
    err_ = np.abs(r.x - 100.0 / k) / (100.0 / k)
    after = err_[r.t >= pp.settling_time]
    before = err_[(r.t >= pp.settling_time - 0.01 * pp.settling_time) & (r.t < pp.settling_time)]
    report(f"Settling time: within 2 % afterwards, and just outside 2 % just before, zeta = {z}",
           after.max() <= 0.0201 and before.max() >= 0.0199, f"{pp.settling_time:.4f} s")
check("Critical damping settles in about 5.83 / wn (not 4 / wn)",
      properties(Params(m, damping_for_ratio(m, k, 1.0), k)).settling_time, 5.8335 / wn, 1e-3)

# 9. Harmonic force: full response (transient + steady state) against the ODE solver
for z, r_ in ((0.0, 0.6), (0.05, 0.9), (0.2, 1.0), (0.5, 1.7), (1.0, 0.4), (2.0, 3.0)):
    worst = 0.0
    for _ in range(3):
        p = Params(rng.uniform(0.5, 4), 0.0, rng.uniform(200, 1500))
        p.c = damping_for_ratio(p.m, p.k, z)
        w = r_ * math.sqrt(p.k / p.m)
        load = Load("harmonic", rng.uniform(10, 80), w)
        x0, v0 = rng.uniform(-0.05, 0.05), rng.uniform(-1, 1)
        T = 10 * 2 * math.pi / min(w, math.sqrt(p.k / p.m))
        rr = simulate(p, load, x0, v0, T, 4001)
        xr, vr = reference(p, load, x0, v0, rr.t)
        scale = max(np.max(np.abs(xr)), 1e-9)
        worst = max(worst, np.max(np.abs(rr.x - xr)) / scale, np.max(np.abs(rr.v - vr)) / (scale * w))
    check_small(f"Harmonic response matches ODE solver, zeta = {z}, r = {r_}", worst, 1e-7)

# 10. Undamped resonance: linear growth, exactly at r = 1
p = Params(m, 0.0, k)
load = Load("harmonic", 40.0, wn)
rr = simulate(p, load, 0.0, 0.0, 3.0, 6001)
xr, _ = reference(p, load, 0.0, 0.0, rr.t)
check_small("Undamped resonance matches the ODE solver (relative to the final amplitude)",
            float(np.max(np.abs(rr.x - xr)) / np.max(np.abs(xr))), 1e-7)
env = 40.0 / (2 * m * wn) * rr.t
last_cycle = rr.t > rr.t[-1] - 2 * math.pi / wn
ratio = float(np.max(np.abs(rr.x[last_cycle])) / env[-1])
report("Undamped resonance: |x| never exceeds F0 t / (2 m wn) and reaches it within the last cycle",
       bool(np.all(np.abs(rr.x) <= env + 1e-12)) and 1 - 2 * math.pi / wn / rr.t[-1] - 1e-3 <= ratio <= 1 + 1e-9,
       f"last-cycle peak / envelope = {ratio:.4f}")

# 11. Steady state: amplitude, resonance and phase, measured from a long run
for z, r_ in ((0.1, 0.5), (0.1, 1.0), (0.3, 1.4), (0.7, 0.8)):
    p = Params(m, damping_for_ratio(m, k, z), k)
    w = r_ * wn
    F0 = 50.0
    T = 60 / (z * wn)
    rr = simulate(p, Load("harmonic", F0, w), 0.0, 0.0, T, int(min(600000, 500 * T * max(w, wn) / (2 * math.pi))))
    late = rr.t > 0.8 * T
    X = F0 / k * float(magnification(r_, z))
    check(f"Steady-state amplitude = (F0 / k) / sqrt((1 - r^2)^2 + (2 z r)^2), zeta = {z}, r = {r_}",
          float(np.max(np.abs(rr.x[late]))), X, 1e-4)
    # phase: x lags F by phi, i.e. x = X cos(w t - phi); fit cos and sin parts by least squares
    A = np.column_stack([np.cos(w * rr.t[late]), np.sin(w * rr.t[late])])
    a, b = np.linalg.lstsq(A, rr.x[late], rcond=None)[0]
    phi = math.degrees(math.atan2(b, a))
    check(f"Phase lag matches atan2(2 z r, 1 - r^2), zeta = {z}, r = {r_}", phi, float(phase_lag(r_, z)), 1e-3)
check("Resonance amplitude: X / x_static = 1 / (2 z) at r = 1", float(magnification(1.0, 0.1)), 5.0)
check("Phase lag is 90 degrees at resonance", float(phase_lag(1.0, 0.1)), 90.0)

# 12. Peak of the magnification curve against a brute-force search
for z in (0.05, 0.15, 0.3, 0.6):
    r = np.linspace(0.01, 3, 600001)
    mag = magnification(r, z)
    rp, mp = resonance_peak(z)
    check(f"Peak frequency ratio sqrt(1 - 2 z^2), zeta = {z}", float(r[np.argmax(mag)]), rp, 1e-5)
    check(f"Peak magnification 1 / (2 z sqrt(1 - z^2)), zeta = {z}", float(mag.max()), mp, 1e-8)
rp, mp = resonance_peak(0.8)
report("No resonance peak when zeta >= 1/sqrt(2)", rp == 0.0 and mp == 1.0, f"{rp}, {mp}")
z = 0.02
r = np.linspace(0.5, 1.5, 2000001)
half = r[magnification(r, z) >= magnification(1.0, z) / math.sqrt(2)]
check("Half-power bandwidth is about 2 z (small damping)", float(half[-1] - half[0]), 2 * z, 2e-3)

# 13. Power balance in steady state: average input power equals average damper dissipation
for z, r_ in ((0.1, 0.7), (0.3, 1.3)):
    X = float(magnification(r_, z))     # F0 = k = 1
    phi = math.radians(float(phase_lag(r_, z)))
    # per unit F0 = 1, k = 1, m = 1: c = 2 z wn... only the dimensionless identity is tested:
    # c w X^2 / 2  =  F0 X sin(phi) / 2   with  c w / k = 2 z r
    check(f"Steady-state power balance: 2 z r X = sin(phi), zeta = {z}, r = {r_}", 2 * z * r_ * X, math.sin(phi), 1e-12)

# 14. Energy bookkeeping with a force: W_in = dE + dissipated
p = Params(m, damping_for_ratio(m, k, 0.2), k)
rr = simulate(p, Load("harmonic", 60.0, 0.8 * wn), 0.02, 0.1, 4.0, 80001)
E = rr.kinetic + rr.potential
check_small("Forced: work in = change in energy + dissipated (max error over time, relative)",
            float(np.max(np.abs(rr.work_in - (E - E[0]) - rr.dissipated)) / np.max(E)), 1e-7)

# 15. Transmissibility: 1 at r = sqrt 2 for every damping, below 1 beyond, 1 at r = 0
for z in (0.0, 0.1, 0.5, 1.5):
    check(f"Transmissibility = 1 at r = sqrt 2, zeta = {z}", float(transmissibility(math.sqrt(2), z)), 1.0, 1e-12)
check("Transmissibility = 1 at r = 0", float(transmissibility(0.0, 0.3)), 1.0, 1e-12)
report("Isolation: lightly damped is better than heavily damped beyond r = sqrt 2",
       float(transmissibility(3.0, 0.05)) < float(transmissibility(3.0, 0.5)) < 1.0)

# 16. Helpers
p = Params(m, damping_for_ratio(m, k, 0.1), k)
d = suggest_duration(p, Load("free"))
report("Suggested duration covers the settling time", d >= properties(p).settling_time, f"{d:.2f} s")
n = suggest_samples(p, Load("harmonic", 10.0, 10 * wn), 5.0)
report("Samples resolve the fastest oscillation (at least 40 per cycle)", n >= 40 * 5.0 * 10 * wn / (2 * math.pi) - 1, f"{n}")
report("Fast forcing frequency stays within the sample cap", n <= 40000)


# 17. Bad input
def rejects(label, fn):
    try:
        fn()
        got = "no error"
    except ValueError:
        got = "caught"
    report(f"Rejected: {label}", got == "caught", got)


rejects("zero mass", lambda: Params(0.0, 1.0, 1.0))
rejects("negative stiffness", lambda: Params(1.0, 1.0, -5.0))
rejects("negative damping", lambda: Params(1.0, -0.1, 5.0))
rejects("NaN stiffness", lambda: Params(1.0, 1.0, float("nan")))
rejects("harmonic load with zero frequency", lambda: Load("harmonic", 10.0, 0.0))
rejects("unknown load type", lambda: Load("ramp"))
rejects("zero duration", lambda: simulate(Params(1, 1, 1), Load("free"), 0.0, 0.0, 0.0))

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
