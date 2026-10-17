"""Checks the truss solver against hand calculations. Run: python test_truss_solver.py"""
import math

from truss_solver import Member, NodalLoad, Node, Support, solve_truss

ok = True


def check(name, got, want, tol=1e-6):
    global ok
    good = abs(got - want) <= tol * max(abs(want), 1e-9)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:.6g}, expected {want:.6g}")


def check_zero(name, got, tol=1e-6):
    """Quantities that should be exactly zero are compared with an absolute tolerance."""
    global ok
    good = abs(got) <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:.3g}, expected 0")


def report(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")


# 1. Single axial bar: pinned at node 1, roller (free in x) at node 2, pulled by P.
#    Force = P, elongation = P*L/(E*A)
E, A, L, P = 200.0, 10.0, 2.0, 50.0          # GPa, cm^2, m, kN
nodes = [Node(0, 0), Node(L, 0)]
r = solve_truss(nodes, [Member(0, 1, E, A)],
                [Support(0, True, True), Support(1, False, True)],
                [NodalLoad(1, fx=P)])
check("Bar  axial force = P", r.member_force[0], P)
check("Bar  tip displacement = PL/EA (mm)", r.displacements[1][0],
      P * 1e3 * L / (E * 1e9 * A * 1e-4) * 1e3)
check("Bar  stress = P/A (MPa)", r.member_stress[0], P * 1e3 / (A * 1e-4) / 1e6)

# 2. Symmetric triangle: A(0,0) pin, B(4,0) roller, C(2,3) loaded down by P.
#    sin(theta) = 3/sqrt(13); leg force = P / (2 sin) compression; chord = leg*cos tension.
P = 10.0
nodes = [Node(0, 0), Node(4, 0), Node(2, 3)]
members = [Member(0, 1), Member(0, 2), Member(1, 2)]     # AB, AC, BC
r = solve_truss(nodes, members,
                [Support(0, True, True), Support(1, False, True)],
                [NodalLoad(2, fy=-P)])
sin_t, cos_t = 3 / math.sqrt(13), 2 / math.sqrt(13)
leg = P / (2 * sin_t)
check("Triangle  AC = -P/(2 sin) (compression)", r.member_force[1], -leg)
check("Triangle  BC = -P/(2 sin) (compression)", r.member_force[2], -leg)
check("Triangle  AB = +leg*cos (tension)", r.member_force[0], leg * cos_t)
check("Triangle  reaction at A (y) = P/2", r.reactions[(0, "y")], P / 2)
check("Triangle  reaction at B (y) = P/2", r.reactions[(1, "y")], P / 2)
check_zero("Triangle  reaction at A (x) = 0", r.reactions[(0, "x")])
check_zero("Triangle  determinate: m + r - 2j = 0", r.indeterminacy)

# 3. Global equilibrium on an asymmetric loading
r = solve_truss(nodes, members,
                [Support(0, True, True), Support(1, False, True)],
                [NodalLoad(2, fx=7.0, fy=-12.0), NodalLoad(1, fy=-3.0)])
check_zero("Equilibrium  sum Fx", r.equilibrium_error["sum_Fx_kN"])
check_zero("Equilibrium  sum Fy", r.equilibrium_error["sum_Fy_kN"])
check_zero("Equilibrium  sum M",  r.equilibrium_error["sum_M_origin_kNm"])

# 4. Joint equilibrium at the loaded joint C: member forces must balance the load
#    (members pull on the joint along their axes).
fx = fy = 0.0
for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
    if 2 in (i, j):
        other = i if j == 2 else j
        dx, dy = nodes[other].x - nodes[2].x, nodes[other].y - nodes[2].y
        Lk = math.hypot(dx, dy)
        fx += r.member_force[k] * dx / Lk
        fy += r.member_force[k] * dy / Lk
check_zero("Joint C  member forces + load, x", fx + 7.0)
check_zero("Joint C  member forces + load, y", fy - 12.0)

# 5. Statically indeterminate: a 4th member (extra diagonal) makes m + r - 2j = 1,
#    and equilibrium must still hold.
nodes = [Node(0, 0), Node(4, 0), Node(4, 3), Node(0, 3)]
members = [Member(0, 1), Member(1, 2), Member(2, 3), Member(3, 0), Member(0, 2)]
r = solve_truss(nodes, members,
                [Support(0, True, True), Support(1, False, True)],
                [NodalLoad(2, fx=20.0)])
check_zero("Braced rectangle  indeterminacy = 0 (5 members + 3 - 8)", r.indeterminacy)
check_zero("Braced rectangle  sum Fx", r.equilibrium_error["sum_Fx_kN"])
# diagonal of a 4x3 rectangle, horizontal load P at the top-right: F_diag = P*5/4
check("Braced rectangle  diagonal = 5P/4 (tension)", r.member_force[4], 20.0 * 5 / 4)

members6 = members + [Member(1, 3)]   # second diagonal -> 1 degree indeterminate
r = solve_truss(nodes, members6,
                [Support(0, True, True), Support(1, False, True)],
                [NodalLoad(2, fx=20.0)])
check("Double-braced rectangle  indeterminacy = 1", r.indeterminacy, 1.0)
check_zero("Double-braced rectangle  sum Fx", r.equilibrium_error["sum_Fx_kN"])
check_zero("Double-braced rectangle  sum Fy", r.equilibrium_error["sum_Fy_kN"])

# 6. Mechanism detection: unbraced rectangle must be rejected
try:
    solve_truss(nodes, members[:4],
                [Support(0, True, True), Support(1, False, True)],
                [NodalLoad(2, fx=20.0)])
    got = "no error"
except ValueError as e:
    got = "unstable" if "unstable" in str(e) else str(e)
good = got == "unstable"
ok &= good
print(f"{'PASS' if good else 'FAIL'}  Unbraced rectangle rejected as unstable ({got})")

# 7. Bad input is reported clearly
try:
    solve_truss([Node(0, 0), Node(1, 0)], [Member(0, 5)], [Support(0, True, True)], [])
    got = "no error"
except ValueError as e:
    got = "caught"
good = got == "caught"
ok &= good
print(f"{'PASS' if good else 'FAIL'}  Member to a missing node is rejected")


# 11. Buckling check
from truss_solver import check_buckling

# single compressed bar: pin at node 0, roller (free in x) at node 1, pushed with P
def bar(P, L=3.0, E=200.0, A=10.0, I=50.0):
    nodes = [Node(0, 0), Node(L, 0)]
    members = [Member(0, 1, E, A, I)]
    res = solve_truss(nodes, members, [Support(0, True, True), Support(1, False, True)],
                      [NodalLoad(1, fx=-P)])
    return res, members

res, mem = bar(20.0)
bk = check_buckling(res, mem, yield_strength=250.0, K=1.0, required_sf=2.0)
pcr_euler = math.pi ** 2 * 200e9 * 50e-8 / 3.0 ** 2 / 1e3          # kN
check("Euler: Pcr = pi^2 E I / L^2 (kN)", bk.critical_load[0], pcr_euler)
check("Euler: safety factor = Pcr / P", bk.safety_factor[0], pcr_euler / 20.0)
report("Slender bar is in the Euler regime", bk.regime[0] == "Euler", str(bk.regime[0]))
r_cm = math.sqrt(50.0 / 10.0)
check("Slenderness = L / r", bk.slenderness[0], 300.0 / r_cm)
check("Transition slenderness = pi sqrt(2E / sigma_y)", bk.transition_slenderness[0],
      math.pi * math.sqrt(2 * 200e9 / 250e6))

# K scaling is exact only while the bar stays slender (Euler); use a 6 m bar so K = 0.5 still is
res6, mem6 = bar(20.0, L=6.0)
base6 = check_buckling(res6, mem6, 250.0).critical_load[0]
for K, f in ((0.5, 4.0), (2.0, 0.25)):
    b2 = check_buckling(res6, mem6, 250.0, K=K)
    check(f"Effective length K = {K}: Pcr scales by {f}", b2.critical_load[0], base6 * f)

# stocky bar: Euler would exceed the squash load; Johnson must be used
res, mem = bar(100.0, L=0.5, I=100.0)
bk = check_buckling(res, mem, 250.0)
lam = bk.slenderness[0]
johnson = (250e6 - 250e6 ** 2 * lam ** 2 / (4 * math.pi ** 2 * 200e9)) * 10e-4 / 1e3
report("Stocky bar uses the Johnson parabola", bk.regime[0] == "Johnson", str(bk.regime[0]))
check("Johnson: Pcr = A [sigma_y - sigma_y^2 lam^2 / (4 pi^2 E)] (kN)", bk.critical_load[0], johnson)
report("Johnson capacity never exceeds the squash load A * sigma_y",
       bk.critical_load[0] <= 250e6 * 10e-4 / 1e3, f"{bk.critical_load[0]:.2f} kN <= 250 kN")

# the two regimes meet at lambda_c, where sigma_cr = sigma_y / 2
res, _ = bar(10.0)
vals = []
for factor in (0.999, 1.001):
    lam_c = math.pi * math.sqrt(2 * 200e9 / 250e6)
    Lc = factor * lam_c * (r_cm / 100.0)
    res_c, mem_c = bar(10.0, L=Lc)
    vals.append(check_buckling(res_c, mem_c, 250.0).critical_stress[0])
check("Regimes meet at the transition (just below)", vals[0], 125.0, 5e-3)
check("Regimes meet at the transition (just above)", vals[1], 125.0, 5e-3)

# safety-factor status thresholds (Pcr = 109.66 kN for the 3 m bar)
res_bar, mem_bar = bar(20.0)
pcr = check_buckling(res_bar, mem_bar, 250.0).critical_load[0]
for load, want in ((pcr / 0.9, "Buckles"), (pcr / 1.5, "Below required"), (pcr / 3.0, "OK")):
    resx, memx = bar(load)
    got = check_buckling(resx, memx, 250.0, required_sf=2.0).status[0]
    report(f"Status at SF = {pcr / load:.1f} (required 2.0)", got == want, f"{got}")

# triangle truss: legs in compression, chord in tension
nodes = [Node(0, 0), Node(4, 0), Node(2, 3)]
members = [Member(0, 1, 200.0, 10.0, 20.0), Member(0, 2, 200.0, 10.0, 20.0),
           Member(1, 2, 200.0, 10.0, 20.0)]
res = solve_truss(nodes, members, [Support(0, True, True), Support(1, False, True)],
                  [NodalLoad(2, fy=-10.0)])
bk = check_buckling(res, members, 250.0)
leg_len = math.sqrt(13.0)
pcr_leg = math.pi ** 2 * 200e9 * 20e-8 / leg_len ** 2 / 1e3
check("Triangle leg critical load (kN)", bk.critical_load[1], pcr_leg)
check("Triangle leg safety factor = Pcr / 6.009 kN", bk.safety_factor[1], pcr_leg / 6.00925, 1e-5)
report("Tension chord is not checked", bk.status[0] == "Tension" and math.isnan(bk.safety_factor[0]),
       f"{bk.status[0]}, SF {bk.safety_factor[0]}")

# a zero-force member is reported as such
nodes = [Node(0, 0), Node(2, 0), Node(2, 2)]
members = [Member(0, 1), Member(1, 2), Member(0, 2)]
res = solve_truss(nodes, members, [Support(0, True, True), Support(1, False, True)],
                  [NodalLoad(1, fy=-10.0)])
bk = check_buckling(res, members, 250.0)
report("Zero-force member is labelled", "Zero force" in set(bk.status), str(list(bk.status)))

# bad input
for kwargs, label in (({"yield_strength": 0.0}, "zero yield strength"), ({"K": -1.0}, "negative K"),
                      ({"required_sf": 0.5}, "required safety factor below 1")):
    try:
        check_buckling(res, members, **kwargs)
        got = "no error"
    except ValueError:
        got = "caught"
    report(f"Rejected: {label}", got == "caught", got)
try:
    solve_truss([Node(0, 0), Node(1, 0)], [Member(0, 1, 200.0, 10.0, 0.0)],
                [Support(0, True, True)], [])
    got = "no error"
except ValueError:
    got = "caught"
report("Rejected: member with I = 0", got == "caught", got)

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
