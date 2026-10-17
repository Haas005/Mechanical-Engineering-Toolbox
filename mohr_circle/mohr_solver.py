"""Plane-stress transformation, Mohr's circle, failure criteria and strains.

Units: stresses in MPa, E in GPa. Strains are returned dimensionless (multiply by 1e6 for
microstrain). Angles in degrees.

Sign conventions (standard mechanics of materials):
    * Normal stress is positive in tension.
    * Shear stress tau_xy is positive when, on the +x face, it points in the +y direction
      (and therefore on the +y face it points in the +x direction).
    * theta is measured counter-clockwise from the x axis to the new x' axis.
    * Plane stress: sigma_z = tau_xz = tau_yz = 0.
"""
import math
from dataclasses import dataclass

import numpy as np


def _finite(**values):
    for name, v in values.items():
        if not math.isfinite(v):
            raise ValueError(f"{name} must be a finite number (got {v}).")


def _wrap_half_turn(angle_deg):
    """Wrap an angle to (-90, 90]: stress transformation repeats every 180 degrees."""
    w = (angle_deg + 90.0) % 180.0 - 90.0
    return 90.0 if w == -90.0 else w


def transform(sx, sy, txy, theta_deg):
    """Stresses on the element rotated counter-clockwise by theta_deg.

    Returns (sigma_x', sigma_y', tau_x'y'). Works with numbers or numpy arrays of angles.
    """
    th = np.radians(theta_deg)
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    avg, dif = (sx + sy) / 2, (sx - sy) / 2
    return (avg + dif * c2 + txy * s2,
            avg - dif * c2 - txy * s2,
            -dif * s2 + txy * c2)


@dataclass
class MohrResult:
    sx: float
    sy: float
    txy: float
    center: float        # average normal stress
    radius: float        # maximum in-plane shear stress
    s1: float            # larger in-plane principal stress
    s2: float            # smaller in-plane principal stress
    theta_p: float       # deg, x axis -> direction of s1 (counter-clockwise), in (-90, 90]
    theta_s: float       # deg, x axis -> plane of maximum positive in-plane shear
    tau_abs_max: float   # absolute maximum shear, including the zero stress out of plane
    principal_3d: tuple  # (largest, middle, smallest) of (s1, s2, 0)


def analyze(sx, sy, txy):
    """Principal stresses, maximum shear and their orientations for a plane-stress state."""
    _finite(sx=sx, sy=sy, txy=txy)
    center = (sx + sy) / 2
    dif = (sx - sy) / 2
    radius = math.hypot(dif, txy)
    theta_p = _wrap_half_turn(0.5 * math.degrees(math.atan2(txy, dif)))
    theta_s = _wrap_half_turn(theta_p - 45.0)
    s1, s2 = center + radius, center - radius
    three = tuple(sorted((s1, s2, 0.0), reverse=True))
    return MohrResult(sx, sy, txy, center, radius, s1, s2, theta_p, theta_s,
                      (three[0] - three[2]) / 2, three)


@dataclass
class FailureResult:
    von_mises: float      # equivalent stress, MPa
    tresca: float         # equivalent stress (largest principal difference), MPa
    sf_von_mises: float   # yield strength / equivalent stress (inf when unstressed)
    sf_tresca: float


def failure_check(res, yield_strength):
    """Von Mises and Tresca yield checks for a plane-stress state (sigma_z = 0)."""
    _finite(yield_strength=yield_strength)
    if yield_strength <= 0:
        raise ValueError("Yield strength must be positive.")
    s1, s2 = res.s1, res.s2
    vm = math.sqrt(max(s1 * s1 - s1 * s2 + s2 * s2, 0.0))
    tr = res.principal_3d[0] - res.principal_3d[2]   # Tresca compares max - min of (s1, s2, 0)
    div = lambda a, b: math.inf if b == 0 else a / b
    return FailureResult(vm, tr, div(yield_strength, vm), div(yield_strength, tr))


@dataclass
class StrainResult:
    ex: float
    ey: float
    gxy: float            # engineering shear strain
    ez: float             # out-of-plane normal strain (Poisson effect)
    e1: float             # principal in-plane strains
    e2: float
    theta_p: float        # deg, direction of e1
    gamma_max: float      # maximum in-plane shear strain


def strains(sx, sy, txy, E_GPa, nu):
    """In-plane strains from Hooke's law for plane stress, plus principal strains."""
    _finite(sx=sx, sy=sy, txy=txy, E=E_GPa, nu=nu)
    if E_GPa <= 0:
        raise ValueError("E must be positive.")
    if not 0.0 <= nu <= 0.5:
        raise ValueError("Poisson's ratio must be between 0 and 0.5.")
    E = E_GPa * 1e3   # MPa
    ex = (sx - nu * sy) / E
    ey = (sy - nu * sx) / E
    gxy = 2 * (1 + nu) * txy / E
    ez = -nu * (sx + sy) / E
    centre = (ex + ey) / 2
    dif = (ex - ey) / 2
    r = math.hypot(dif, gxy / 2)
    theta = _wrap_half_turn(0.5 * math.degrees(math.atan2(gxy / 2, dif)))
    return StrainResult(ex, ey, gxy, ez, centre + r, centre - r, theta, 2 * r)
