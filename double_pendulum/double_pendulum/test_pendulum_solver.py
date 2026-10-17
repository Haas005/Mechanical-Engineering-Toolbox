"""Physics checks for the double pendulum. Run: python test_pendulum_solver.py"""
import math

import numpy as np
import sympy as sp

from pendulum_solver import (Params, accelerations, energy, flip_time_grid, positions,
                             simulate)

ok = True


def report(name, good, detail):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")


def check(name, got, want, tol):
    report(name, abs(got - want) <= tol * max(abs(want), 1e-12),
           f"got {got:.8g}, expected {want:.8g}")


def check_small(name, got, tol):
    report(name, abs(got) <= tol, f"got {got:.3g}, must be within {tol:g} of 0")


def period_from_zero_crossings(t, y):
    """Average period from upward zero crossings (linear interpolation)."""
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    times = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    return float(np.mean(np.diff(times)))


# 1. Independent derivation: Lagrange's equations with SymPy must match the closed form
t = sp.symbols("t")
m1, m2, l1, l2, g = sp.symbols("m1 m2 l1 l2 g", positive=True)
q1, q2 = sp.Function("q1")(t), sp.Function("q2")(t)
x1, y1 = l1 * sp.sin(q1), -l1 * sp.cos(q1)
x2, y2 = x1 + l2 * sp.sin(q2), y1 - l2 * sp.cos(q2)
T = sp.Rational(1, 2) * m1 * (x1.diff(t) ** 2 + y1.diff(t) ** 2) \
    + sp.Rational(1, 2) * m2 * (x2.diff(t) ** 2 + y2.diff(t) ** 2)
V = m1 * g * y1 + m2 * g * y2
Lag = T - V
eqs = [sp.diff(Lag.diff(q.diff(t)), t) - Lag.diff(q) for q in (q1, q2)]
sol = sp.solve(eqs, [q1.diff(t, 2), q2.diff(t, 2)], dict=True)[0]
th1, th2, w1, w2 = sp.symbols("th1 th2 w1 w2")
subs = {q1.diff(t): w1, q2.diff(t): w2}
a1_sym = sol[q1.diff(t, 2)].subs(subs).subs({q1: th1, q2: th2})
a2_sym = sol[q2.diff(t, 2)].subs(subs).subs({q1: th1, q2: th2})
f_sym = sp.lambdify((th1, th2, w1, w2, m1, m2, l1, l2, g), (a1_sym, a2_sym), "numpy")

rng = np.random.default_rng(7)
worst = 0.0
for _ in range(200):
    pr = Params(*rng.uniform(0.3, 3.0, 4), rng.uniform(1.0, 20.0))
    s = rng.uniform(-3.0, 3.0, 4)
    ref = np.array(f_sym(*s, pr.m1, pr.m2, pr.l1, pr.l2, pr.g))
    got = np.array(accelerations(*s, pr))
    worst = max(worst, float(np.max(np.abs(got - ref) / (1 + np.abs(ref)))))
check_small("SymPy Lagrangian vs closed-form accelerations (200 random states)", worst, 1e-9)

# 2. Energy conservation on a violently chaotic run
p = Params(1.0, 1.0, 1.0, 1.0, 9.81)
traj = simulate(p, math.radians(120), math.radians(-10), 0.0, 0.0, 30.0)
check_small("Energy drift over 30 s of chaotic motion (relative)", traj.relative_energy_drift, 1e-8)

# 3. Equilibrium: hanging straight down stays put
traj = simulate(p, 0.0, 0.0, 0.0, 0.0, 5.0)
check_small("Rest at the bottom stays at rest", float(np.max(np.abs(traj.theta1)) +
                                                      np.max(np.abs(traj.theta2))), 1e-12)

# 4. Small-angle normal modes (equal masses and lengths): w^2 = (g/l)(2 -/+ sqrt 2)
amp = 0.002
for label, ratio, w2_expect in (("slow mode (swing together)", math.sqrt(2), 2 - math.sqrt(2)),
                                ("fast mode (swing opposed)", -math.sqrt(2), 2 + math.sqrt(2))):
    w = math.sqrt(w2_expect * p.g / p.l1)
    traj = simulate(p, amp * 1.0, amp * ratio, 0.0, 0.0, 12 * 2 * math.pi / w)
    # start at an extreme (cosine); shift by a quarter period of data via derivative sign
    per = period_from_zero_crossings(traj.t, traj.omega1)
    check(f"Normal mode period, {label}", per, 2 * math.pi / w, 1e-4)
    # in a pure mode the ratio theta2/theta1 stays constant for all time
    # (compare only where theta1 is well away from zero, otherwise dividing by a number
    # near zero turns tiny numerical noise into a large apparent error)
    far = np.abs(traj.theta1) > 0.5 * amp
    r = traj.theta2[far] / traj.theta1[far]
    check_small(f"Mode shape holds, {label}", float(np.max(np.abs(r - ratio))), 1e-3)

# 5. Light second bob: upper pendulum reduces to a simple pendulum of length l1
pl = Params(2.0, 1e-9, 1.3, 0.7, 9.81)
traj = simulate(pl, 1.0, 0.0, 0.0, 0.0, 6.0)


def simple(t_end, theta0, L, gg, n):
    from scipy.integrate import solve_ivp
    s = solve_ivp(lambda _t, y: [y[1], -gg / L * math.sin(y[0])], (0, t_end), [theta0, 0.0],
                  method="DOP853", t_eval=np.linspace(0, t_end, n), rtol=1e-11, atol=1e-11)
    return s.y[0]


ref = simple(6.0, 1.0, pl.l1, pl.g, len(traj.t))
check_small("m2 -> 0 limit matches a simple pendulum of length l1 (large amplitude)",
            float(np.max(np.abs(traj.theta1 - ref))), 1e-6)

# 6. Mirror symmetry: negating the initial state negates the whole trajectory
a = simulate(p, 1.0, 2.0, 0.3, -0.5, 10.0)
b = simulate(p, -1.0, -2.0, -0.3, 0.5, 10.0)
check_small("Mirror symmetry", float(np.max(np.abs(a.theta1 + b.theta1)) +
                                      np.max(np.abs(a.theta2 + b.theta2))), 1e-9)

# 7. Sensitive dependence on initial conditions (chaos) versus regular small motion
delta = 1e-8
c1 = simulate(p, math.radians(120), math.radians(-10), 0, 0, 30.0)
c2 = simulate(p, math.radians(120) + delta, math.radians(-10), 0, 0, 30.0)
sep_chaos = float(np.max(np.abs(c1.theta2 - c2.theta2)))
report("Chaotic start: a 1e-8 rad difference grows beyond 1 rad within 30 s", sep_chaos > 1.0,
       f"max separation {sep_chaos:.3g} rad")
s1 = simulate(p, 0.05, 0.05, 0, 0, 30.0)
s2 = simulate(p, 0.05 + delta, 0.05, 0, 0, 30.0)
sep_small = float(np.max(np.abs(s1.theta2 - s2.theta2)))
report("Small-angle start: the same 1e-8 rad difference stays tiny", sep_small < 1e-6,
       f"max separation {sep_small:.3g} rad")

# 8. Geometry and energy helper sanity
x1_, y1_, x2_, y2_ = positions(math.pi / 2, math.pi / 2, Params(1, 1, 1.0, 2.0, 9.81))
check_small("Horizontal rods: bob 1 at (l1, 0)", abs(x1_ - 1.0) + abs(y1_), 1e-12)
check_small("Horizontal rods: bob 2 at (l1 + l2, 0)", abs(x2_ - 3.0) + abs(y2_), 1e-12)
_, V0, _ = energy(0.0, 0.0, 0.0, 0.0, Params(1, 2, 1.0, 1.0, 10.0))
check("Potential energy hanging at rest = -(m1 + 2 m2) g l", V0, -(1 + 2 * 2) * 10.0, 1e-12)

# 9. Flip-time grid: cross-check against the adaptive integrator
angles, grid = flip_time_grid(p, n=41, t_end=3.0, dt=0.002)
center = grid[20, 20]
check("Flip grid: starting at rest at the bottom never flips", center, 3.0, 1e-12)
idx = np.random.default_rng(3).integers(0, 41, size=(40, 2))
bad = 0
compared = 0
for i, j in idx:
    tr = simulate(p, angles[j], angles[i], 0.0, 0.0, 3.0, n_points=3001)
    flipped = np.where((np.abs(tr.theta1) > math.pi) | (np.abs(tr.theta2) > math.pi))[0]
    t_ref = tr.t[flipped[0]] if len(flipped) else 3.0
    compared += 1
    if abs(t_ref - grid[i, j]) > 0.02:
        bad += 1
report("Flip grid agrees with the adaptive integrator (40 random cells, 3 s)", bad == 0,
       f"{compared - bad}/{compared} within 0.02 s")

# 10. Bad input is reported clearly
try:
    simulate(Params(-1, 1, 1, 1, 9.81), 0, 0, 0, 0, 1.0)
    got = "no error"
except ValueError:
    got = "caught"
report("Negative mass is rejected", got == "caught", got)

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
