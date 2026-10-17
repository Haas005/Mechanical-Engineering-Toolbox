"""Checks the Mohr's circle solver against independent methods. Run: python test_mohr_solver.py"""
import math

import numpy as np

from mohr_solver import analyze, failure_check, strains, transform

ok = True


def report(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")


def check(name, got, want, tol=1e-9):
    """Relative comparison (absolute when the expected value is about zero)."""
    report(name, abs(got - want) <= tol * max(abs(want), 1.0), f"got {got:.8g}, expected {want:.8g}")


def angle_diff(a, b):
    """Difference between two angles that repeat every 180 degrees."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


rng = np.random.default_rng(11)
states = [tuple(rng.uniform(-200, 200, 3)) for _ in range(300)]

# 1. Textbook special cases
r = analyze(100.0, 0.0, 0.0)
check("Uniaxial: s1 = sigma", r.s1, 100.0)
check("Uniaxial: s2 = 0", r.s2, 0.0)
check("Uniaxial: tau_max = sigma / 2", r.radius, 50.0)
check("Uniaxial: principal direction is the x axis (deg)", r.theta_p, 0.0)
check("Uniaxial: max shear plane at -45 deg", r.theta_s, -45.0)

r = analyze(0.0, 0.0, 40.0)
check("Pure shear: s1 = +tau", r.s1, 40.0)
check("Pure shear: s2 = -tau", r.s2, -40.0)
check("Pure shear: principal direction 45 deg", r.theta_p, 45.0)

r = analyze(60.0, 60.0, 0.0)
check("Equal biaxial: radius 0 (every plane is principal)", r.radius, 0.0)

# Worked example: sx = -20, sy = 90, txy = 60  ->  C = 35, R = sqrt(55^2 + 60^2)
r = analyze(-20.0, 90.0, 60.0)
check("Example: centre", r.center, 35.0)
check("Example: radius", r.radius, math.sqrt(55 ** 2 + 60 ** 2))
check("Example: s1 = 35 + R", r.s1, 35 + math.sqrt(55 ** 2 + 60 ** 2))

# 2. Invariants on 300 random states
worst_trace = max(abs(r_.s1 + r_.s2 - (r_.sx + r_.sy)) for r_ in (analyze(*s) for s in states))
worst_det = max(abs(r_.s1 * r_.s2 - (r_.sx * r_.sy - r_.txy ** 2)) /
                (1 + abs(r_.s1 * r_.s2)) for r_ in (analyze(*s) for s in states))
report("Invariant: s1 + s2 = sx + sy (300 random states)", worst_trace < 1e-9, f"worst {worst_trace:.2e}")
report("Invariant: s1 * s2 = sx sy - txy^2 (300 random states)", worst_det < 1e-9, f"worst {worst_det:.2e}")

# 3. Independent check against the eigenvalues and eigenvectors of the stress tensor
worst_val = worst_dir = 0.0
for sx, sy, txy in states:
    r = analyze(sx, sy, txy)
    w, v = np.linalg.eigh(np.array([[sx, txy], [txy, sy]]))
    worst_val = max(worst_val, abs(w[1] - r.s1), abs(w[0] - r.s2))
    ang = math.degrees(math.atan2(v[1, 1], v[0, 1]))      # direction of the larger eigenvalue
    worst_dir = max(worst_dir, angle_diff(ang, r.theta_p))
report("Principal stresses match eigenvalues of the stress tensor", worst_val < 1e-9, f"worst {worst_val:.2e} MPa")
report("Principal direction matches the eigenvector", worst_dir < 1e-7, f"worst {worst_dir:.2e} deg")

# 4. Transformation equations versus rotating the tensor: S' = Q S Q^T
worst = 0.0
for sx, sy, txy in states:
    th = rng.uniform(-180, 180)
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    Q = np.array([[c, s], [-s, c]])
    S = Q @ np.array([[sx, txy], [txy, sy]]) @ Q.T
    got = transform(sx, sy, txy, th)
    worst = max(worst, abs(got[0] - S[0, 0]), abs(got[1] - S[1, 1]), abs(got[2] - S[0, 1]))
report("Transformation equations match Q S Q^T (300 random states and angles)", worst < 1e-9, f"worst {worst:.2e}")

# 5. The principal and shear planes really are where the formulas say
worst = 0.0
for s in states:
    r = analyze(*s)
    a = transform(*s, r.theta_p)
    b = transform(*s, r.theta_s)
    worst = max(worst, abs(a[0] - r.s1), abs(a[1] - r.s2), abs(a[2]),
                abs(b[2] - r.radius), abs(b[0] - r.center))
report("At theta_p: sigma_x' = s1, tau = 0.  At theta_s: tau = +R, sigma = centre", worst < 1e-9,
       f"worst {worst:.2e}")

# 6. Brute force sweep: the extremes over all angles are the principal stresses and R
for s in states[:20]:
    r = analyze(*s)
    sxp, _, txyp = transform(*s, np.linspace(-90, 90, 36001))
    good = (abs(sxp.max() - r.s1) < 1e-4 * (1 + abs(r.s1)) and abs(sxp.min() - r.s2) < 1e-4 * (1 + abs(r.s2))
            and abs(np.abs(txyp).max() - r.radius) < 1e-4 * (1 + r.radius))
    if not good:
        break
report("Sweep over 36001 angles: max/min sigma and max |tau| equal s1, s2 and R (20 states)", good)

# 7. Points on Mohr's circle: every transformed state lies at distance R from the centre
worst = 0.0
for s in states[:50]:
    r = analyze(*s)
    for th in (-70, -20, 0, 33, 80):
        sxp, _, txyp = transform(*s, th)
        worst = max(worst, abs(math.hypot(sxp - r.center, txyp) - r.radius))
report("Transformed states lie on the circle (distance R from the centre)", worst < 1e-9, f"worst {worst:.2e}")

# 8. Rotating the element by theta turns the circle point by 2 theta, counter-clockwise when
#    tau is plotted downward (the app plots it that way)
s = (-20.0, 90.0, 60.0)
r = analyze(*s)
worst = 0.0
for th in (10.0, 25.0, 40.0):
    sxp, _, txyp = transform(*s, th)
    a0 = math.atan2(-s[2], s[0] - r.center)      # display coords: y = -tau
    a1 = math.atan2(-txyp, sxp - r.center)
    turn = math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)
    worst = max(worst, abs(turn - 2 * th))
report("Circle point turns by 2 theta counter-clockwise when tau is plotted down", worst < 1e-9, f"worst {worst:.2e} deg")

# 9. Absolute maximum shear includes the zero out-of-plane stress
r = analyze(100.0, 50.0, 0.0)
check("Same-sign principal stresses: absolute max shear = s_max / 2", r.tau_abs_max, 50.0)
check("   ... larger than the in-plane maximum shear (25)", r.radius, 25.0)
r = analyze(100.0, -50.0, 0.0)
check("Opposite-sign principal stresses: absolute = in-plane R", r.tau_abs_max, 75.0)

# 10. Failure criteria
fv = failure_check(analyze(150.0, 0.0, 0.0), 250.0)
check("Von Mises, uniaxial = sigma", fv.von_mises, 150.0)
check("Tresca, uniaxial = sigma", fv.tresca, 150.0)
check("Safety factor = yield / equivalent stress", fv.sf_von_mises, 250.0 / 150.0)
fp = failure_check(analyze(0.0, 0.0, 80.0), 250.0)
check("Von Mises, pure shear = sqrt(3) tau", fp.von_mises, math.sqrt(3) * 80.0)
check("Tresca, pure shear = 2 tau", fp.tresca, 160.0)
check("Equal biaxial tension: Tresca = sigma (z-direction stress is zero)", failure_check(
      analyze(100.0, 100.0, 0.0), 250.0).tresca, 100.0)
check("Equal biaxial tension: von Mises = sigma", failure_check(
      analyze(100.0, 100.0, 0.0), 250.0).von_mises, 100.0)
worst = 0.0
for sx, sy, txy in states:
    f = failure_check(analyze(sx, sy, txy), 250.0)
    worst = max(worst, abs(f.von_mises - math.sqrt(sx * sx - sx * sy + sy * sy + 3 * txy * txy)))
report("Von Mises from s1, s2 equals the form using sx, sy, txy (300 random states)", worst < 1e-9, f"worst {worst:.2e}")
report("Tresca is never below von Mises (it is the more conservative criterion)",
       all(failure_check(analyze(*s), 250.0).tresca >= failure_check(analyze(*s), 250.0).von_mises - 1e-9
           for s in states))
report("Unstressed state has infinite safety factor",
       math.isinf(failure_check(analyze(0, 0, 0), 250.0).sf_von_mises))

# 11. Strains (Hooke's law, plane stress)
E, nu = 200.0, 0.3
e = strains(100.0, 0.0, 0.0, E, nu)
check("Uniaxial: ex = sigma / E", e.ex, 100.0 / 200e3)
check("Uniaxial: ey = -nu sigma / E", e.ey, -0.3 * 100.0 / 200e3)
check("Uniaxial: ez = -nu sigma / E", e.ez, -0.3 * 100.0 / 200e3)
e = strains(0.0, 0.0, 50.0, E, nu)
check("Pure shear: gamma = tau / G, with G = E / (2 (1 + nu))", e.gxy, 50.0 / (200e3 / (2 * 1.3)))
worst_dir = worst_val = 0.0
for sx, sy, txy in states:
    st = strains(sx, sy, txy, E, nu)
    T = np.array([[st.ex, st.gxy / 2], [st.gxy / 2, st.ey]])
    w, v = np.linalg.eigh(T)
    worst_val = max(worst_val, abs(w[1] - st.e1), abs(w[0] - st.e2))
    # for an isotropic material the principal strain and stress directions coincide
    worst_dir = max(worst_dir, angle_diff(st.theta_p, analyze(sx, sy, txy).theta_p))
report("Principal strains match eigenvalues of the strain tensor", worst_val < 1e-12, f"worst {worst_val:.2e}")
report("Principal strain direction = principal stress direction (isotropic)", worst_dir < 1e-6, f"worst {worst_dir:.2e} deg")

# 12. Bad input is rejected
def rejects(label, fn):
    try:
        fn()
        got = "no error"
    except ValueError:
        got = "caught"
    report(f"Rejected: {label}", got == "caught", got)


rejects("NaN stress", lambda: analyze(float("nan"), 0, 0))
rejects("infinite stress", lambda: analyze(0, float("inf"), 0))
rejects("zero yield strength", lambda: failure_check(analyze(1, 0, 0), 0.0))
rejects("negative E", lambda: strains(1, 0, 0, -200.0, 0.3))
rejects("Poisson's ratio 0.7", lambda: strains(1, 0, 0, 200.0, 0.7))

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
