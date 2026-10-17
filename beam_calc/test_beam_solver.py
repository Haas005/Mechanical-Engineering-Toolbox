"""Checks the solver against closed-form textbook results. Run: python test_beam_solver.py"""
import math

import numpy as np
from beam_solver import solve_beam, PointLoad, UniformLoad, PointMoment

L, E, I = 5.0, 200e9, 8e-5      # m, Pa, m^4
EI = E * I
w, P = 2000.0, 10000.0
ok = True

def check_zero(name, got, tol):
    """Quantities that should be zero (or a difference that should vanish): absolute tolerance."""
    global ok
    good = abs(got) <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:.3g}, must be within {tol:g} of 0")


def check(name, got, want, tol=1e-3):
    global ok
    good = abs(got - want) <= tol * max(abs(want), 1e-12)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:.6g}, expected {want:.6g}")

# 1. Simply supported, full UDL
r = solve_beam(L, E, I, "simple", [UniformLoad(w, 0, L)])
check("SS UDL  max |V| = wL/2", abs(r.extreme("shear")[0]), w*L/2)
check("SS UDL  max M  = wL^2/8", r.extreme("moment")[0], w*L**2/8)
check("SS UDL  max y  = -5wL^4/384EI", r.extreme("deflection")[0], -5*w*L**4/(384*EI))

# 2. Simply supported, centre point load
r = solve_beam(L, E, I, "simple", [PointLoad(P, L/2)])
check("SS point  max M = PL/4", r.extreme("moment")[0], P*L/4)
check("SS point  max y = -PL^3/48EI", r.extreme("deflection")[0], -P*L**3/(48*EI))

# 3. Cantilever, tip point load
r = solve_beam(L, E, I, "cantilever", [PointLoad(P, L)])
check("Cant point  max |M| = PL", abs(r.extreme("moment")[0]), P*L)
check("Cant point  tip y = -PL^3/3EI", r.deflection[-1], -P*L**3/(3*EI))
check("Cant point  fixed-end moment reaction = +PL (CCW)",
      r.reactions["Moment at x = 0 m"], P*L)

# 4. Cantilever, full UDL
r = solve_beam(L, E, I, "cantilever", [UniformLoad(w, 0, L)])
check("Cant UDL  tip y = -wL^4/8EI", r.deflection[-1], -w*L**4/(8*EI))

# 5. Fixed-fixed, UDL
r = solve_beam(L, E, I, "fixed_fixed", [UniformLoad(w, 0, L)])
check("FF UDL  centre y = -wL^4/384EI", r.deflection[len(r.x)//2], -w*L**4/(384*EI))
check("FF UDL  end moment magnitude = wL^2/12", abs(r.extreme("moment")[0]), w*L**2/12)

# 6. Propped cantilever, UDL: R at prop = 3wL/8
r = solve_beam(L, E, I, "propped", [UniformLoad(w, 0, L)])
check("Propped UDL  prop reaction = 3wL/8", r.reactions[f"Force at x = {L:g} m"], 3*w*L/8)

# 7. Equilibrium on an off-centre mixed load case
r = solve_beam(L, E, I, "simple",
               [PointLoad(P, 1.5), UniformLoad(w, 2.0, 4.0), PointMoment(3000, 3.0)])
Rs = sum(v for k, v in r.reactions.items() if k.startswith("Force"))
check("Mixed  sum of vertical reactions = applied load", Rs, P + w*2.0)
check("Mixed  V just left of right support = -R_right", r.shear[-2], -r.reactions[f"Force at x = {L:g} m"])

# 8. Applied CCW moment at midspan of a simply supported beam: R_left = +M/L, R_right = -M/L
M0 = 3000.0
r = solve_beam(L, E, I, "simple", [PointMoment(M0, L/2)])
check("CCW moment  left reaction = +M/L", r.reactions["Force at x = 0 m"], M0/L)
check("CCW moment  right reaction = -M/L", r.reactions[f"Force at x = {L:g} m"], -M0/L)


# 9. Linearly varying loads (triangular and trapezoidal)
from beam_solver import LinearLoad

w0 = 6000.0
r = solve_beam(L, E, I, "simple", [LinearLoad(0.0, w0, 0.0, L)])      # 0 at left -> w0 at right
xs = r.x
M_exact = w0 * xs / (6 * L) * (L**2 - xs**2)
y_exact = -w0 * (7 * L**4 * xs - 10 * L**2 * xs**3 + 3 * xs**5) / (360 * L * EI)
check("Triangle (SS)  left reaction = wL/6", r.reactions["Force at x = 0 m"], w0 * L / 6)
check("Triangle (SS)  right reaction = wL/3", r.reactions[f"Force at x = {L:g} m"], w0 * L / 3)
check_zero("Triangle (SS)  whole moment curve matches closed form (N m)",
           float(np.max(np.abs(r.moment - M_exact))), 1e-9 * w0 * L**2)
check("Triangle (SS)  max moment = wL^2/(9 sqrt 3)", r.extreme("moment")[0],
      w0 * L**2 / (9 * math.sqrt(3)))
check("Triangle (SS)  max moment at x = L/sqrt 3", r.extreme("moment")[1], L / math.sqrt(3), 2e-3)
check_zero("Triangle (SS)  whole deflection curve matches closed form (m)",
           float(np.max(np.abs(r.deflection - y_exact))), 1e-8 * float(np.abs(y_exact).max()))

# decreasing triangle (w0 at left -> 0 at right) mirrors it
r2 = solve_beam(L, E, I, "simple", [LinearLoad(w0, 0.0, 0.0, L)])
check("Mirrored triangle  left reaction = wL/3", r2.reactions["Force at x = 0 m"], w0 * L / 3)
check("Mirrored triangle  max deflection equals the original",
      r2.extreme("deflection")[0], r.extreme("deflection")[0], 1e-9)

# cantilever: load w0 at the wall tapering to 0 at the free end
r = solve_beam(L, E, I, "cantilever", [LinearLoad(w0, 0.0, 0.0, L)])
check("Cantilever, taper to tip  wall moment = wL^2/6", r.reactions["Moment at x = 0 m"], w0 * L**2 / 6)
check("Cantilever, taper to tip  wall force = wL/2", r.reactions["Force at x = 0 m"], w0 * L / 2)
check("Cantilever, taper to tip  tip deflection = -wL^4/(30EI)", r.deflection[-1], -w0 * L**4 / (30 * EI))
# cantilever: load 0 at the wall growing to w0 at the free end
r = solve_beam(L, E, I, "cantilever", [LinearLoad(0.0, w0, 0.0, L)])
check("Cantilever, growing to tip  wall moment = wL^2/3", r.reactions["Moment at x = 0 m"], w0 * L**2 / 3)
check("Cantilever, growing to tip  tip deflection = -11wL^4/(120EI)", r.deflection[-1],
      -11 * w0 * L**4 / (120 * EI))

# fixed-fixed beam, load 0 at left -> w0 at right
r = solve_beam(L, E, I, "fixed_fixed", [LinearLoad(0.0, w0, 0.0, L)])
check("Fixed-fixed triangle  left force = 3wL/20", r.reactions["Force at x = 0 m"], 3 * w0 * L / 20)
check("Fixed-fixed triangle  right force = 7wL/20", r.reactions[f"Force at x = {L:g} m"], 7 * w0 * L / 20)
check("Fixed-fixed triangle  left moment = wL^2/30", abs(r.reactions["Moment at x = 0 m"]), w0 * L**2 / 30)
check("Fixed-fixed triangle  right moment = wL^2/20", abs(r.reactions[f"Moment at x = {L:g} m"]), w0 * L**2 / 20)

# trapezoid = uniform + triangle (superposition), partial span
trap = solve_beam(L, E, I, "simple", [LinearLoad(2000.0, 6000.0, 1.0, 4.0)])
sup = solve_beam(L, E, I, "simple", [UniformLoad(2000.0, 1.0, 4.0), LinearLoad(0.0, 4000.0, 1.0, 4.0)])
check_zero("Trapezoid = uniform + triangle  (deflection, m)", float(np.max(np.abs(trap.deflection - sup.deflection))), 1e-12)
check_zero("Trapezoid = uniform + triangle  (moment, N m)", float(np.max(np.abs(trap.moment - sup.moment))), 1e-6)
xq = np.linspace(1.0, 4.0, 200001)
wq = 2000.0 + (6000.0 - 2000.0) * (xq - 1.0) / 3.0
force = np.trapezoid(wq, xq)
moment_left = np.trapezoid(wq * xq, xq)
check("Trapezoid  total load = area", sum(v for k, v in trap.reactions.items() if k.startswith("Force")), force, 1e-9)
check("Trapezoid  right reaction from moments about the left support",
      trap.reactions[f"Force at x = {L:g} m"], moment_left / L, 1e-6)

# equal end intensities is just a uniform load
a = solve_beam(L, E, I, "cantilever", [LinearLoad(3000.0, 3000.0, 0.0, L)])
b = solve_beam(L, E, I, "cantilever", [UniformLoad(3000.0, 0.0, L)])
check_zero("Constant LinearLoad = UniformLoad  (deflection, m)", float(np.max(np.abs(a.deflection - b.deflection))), 1e-12)

# a load that changes sign along the span (down to up) still balances
mix = solve_beam(L, E, I, "simple", [LinearLoad(4000.0, -4000.0, 0.0, L)])
check_zero("Sign-changing load  net vertical reaction = 0", sum(v for k, v in mix.reactions.items() if k.startswith("Force")), 1e-6)

# bad input
try:
    solve_beam(L, E, I, "simple", [LinearLoad(1000.0, 2000.0, 3.0, 6.0)])
    got = "no error"
except ValueError:
    got = "caught"
report_ok = got == "caught"
ok &= report_ok
print(f"{'PASS' if report_ok else 'FAIL'}  Linear load beyond the end of the beam is rejected ({got})")

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
