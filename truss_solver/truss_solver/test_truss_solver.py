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

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
