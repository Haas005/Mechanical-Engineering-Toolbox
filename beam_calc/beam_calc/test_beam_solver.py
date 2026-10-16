"""Checks the solver against closed-form textbook results. Run: python test_beam_solver.py"""
import numpy as np
from beam_solver import solve_beam, PointLoad, UniformLoad, PointMoment

L, E, I = 5.0, 200e9, 8e-5      # m, Pa, m^4
EI = E * I
w, P = 2000.0, 10000.0
ok = True

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

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
