"""Damped single-degree-of-freedom spring-mass-damper: closed-form responses.

    m x'' + c x' + k x = F(t)

Units are SI (kg, N/m, N s/m, N, m, s); the app converts for display.

Sign conventions:
    * x is the displacement from the static equilibrium position, positive to the right.
    * The force F(t) acts on the mass, positive to the right.
    * Harmonic forcing is F(t) = F0 cos(omega t).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    m: float   # kg
    c: float   # N s/m
    k: float   # N/m

    def __post_init__(self):
        for name in ("m", "c", "k"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number.")
        if self.m <= 0:
            raise ValueError("The mass must be positive.")
        if self.k <= 0:
            raise ValueError("The spring stiffness must be positive.")
        if self.c < 0:
            raise ValueError("The damping coefficient cannot be negative.")


def damping_for_ratio(m, k, zeta):
    """Damping coefficient c that gives the damping ratio zeta."""
    return 2.0 * zeta * math.sqrt(k * m)


@dataclass
class Load:
    kind: str = "free"      # "free", "step" or "harmonic"
    F0: float = 0.0         # N (step height, or harmonic amplitude)
    omega: float = 0.0      # rad/s, harmonic only

    def __post_init__(self):
        if self.kind not in ("free", "step", "harmonic"):
            raise ValueError(f"Unknown load type '{self.kind}'.")
        if not (math.isfinite(self.F0) and math.isfinite(self.omega)):
            raise ValueError("The force and frequency must be finite numbers.")
        if self.kind == "harmonic" and self.omega <= 0:
            raise ValueError("The forcing frequency must be positive.")


@dataclass
class Properties:
    wn: float            # natural frequency, rad/s
    fn: float            # natural frequency, Hz
    zeta: float          # damping ratio
    c_crit: float        # critical damping coefficient, N s/m
    regime: str          # Undamped / Underdamped / Critically damped / Overdamped
    wd: float            # damped frequency, rad/s (nan unless underdamped)
    Td: float            # damped period, s (nan unless underdamped)
    log_decrement: float  # ln(x_n / x_n+1) (nan unless underdamped)
    Q: float             # quality factor 1 / (2 zeta)  (inf if undamped)
    settling_time: float  # 2 % settling time of a step response, about 4 / (zeta wn)
    overshoot: float     # step-response overshoot, fraction of the final value
    peak_time: float     # time of the first step-response peak (nan unless underdamped)


def properties(p):
    wn = math.sqrt(p.k / p.m)
    cc = 2.0 * math.sqrt(p.k * p.m)
    z = p.c / cc
    if z < 1e-12:
        regime = "Undamped"
    elif z < 1 - 1e-7:
        regime = "Underdamped"
    elif z <= 1 + 1e-7:
        regime = "Critically damped"
    else:
        regime = "Overdamped"
    under = regime == "Underdamped"
    wd = wn * math.sqrt(1 - z * z) if under else (wn if regime == "Undamped" else math.nan)
    if regime == "Undamped":
        settle = math.inf
    elif z >= 1 - 1e-7:
        settle = _settling_time_no_oscillation(wn, z)
    else:
        settle = 4.0 / (z * wn)      # standard approximation for oscillatory decay
    return Properties(
        wn=wn, fn=wn / (2 * math.pi), zeta=z, c_crit=cc, regime=regime,
        wd=wd if under else math.nan,
        Td=2 * math.pi / wd if under else math.nan,
        log_decrement=2 * math.pi * z / math.sqrt(1 - z * z) if under else math.nan,
        Q=math.inf if z < 1e-12 else 1.0 / (2 * z),
        settling_time=settle,
        overshoot=math.exp(-math.pi * z / math.sqrt(1 - z * z)) if under else
        (1.0 if regime == "Undamped" else 0.0),
        peak_time=math.pi / wd if under else math.nan)


def _settling_time_no_oscillation(wn, zeta):
    """Time after which a step response (from rest) stays within 2 % of its final value, for
    critical and over damping. The textbook 4 / (zeta wn) does not apply here (for critical
    damping the exact answer is about 5.83 / wn), so find it from the exact solution."""
    q = math.sqrt(max(zeta * zeta - 1.0, 0.0))
    slow = wn * (zeta - q) if zeta > 1 + 1e-7 else wn          # slowest decay rate, 1/s
    t = np.linspace(0.0, 15.0 / slow, 30001)
    y, _ = _homogeneous(t, wn, zeta, -1.0, 0.0)                 # x - x_final; final value = 1
    outside = np.where(np.abs(y) > 0.02)[0]
    return float(t[outside[-1] + 1]) if len(outside) else 0.0


# ---------------------------------------------------------------------------------------
# Frequency response (steady state, r = omega / omega_n)
# ---------------------------------------------------------------------------------------
def magnification(r, zeta):
    """Dynamic magnification factor X / (F0 / k) for a harmonic force."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)


def phase_lag(r, zeta):
    """Phase lag of the displacement behind the force, in degrees (0 to 180)."""
    r = np.asarray(r, dtype=float)
    return np.degrees(np.arctan2(2 * zeta * r, 1 - r ** 2))


def transmissibility(r, zeta):
    """Force (or base-motion) transmissibility, for vibration isolation."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(1 + (2 * zeta * r) ** 2) * magnification(r, zeta)


def resonance_peak(zeta):
    """(r at the peak, peak magnification). Returns (0, 1) when there is no peak (zeta >= 1/sqrt 2)."""
    if zeta < 1e-12:
        return 1.0, math.inf
    if zeta >= 1 / math.sqrt(2):
        return 0.0, 1.0
    return math.sqrt(1 - 2 * zeta ** 2), 1.0 / (2 * zeta * math.sqrt(1 - zeta ** 2))


# ---------------------------------------------------------------------------------------
# Time response
# ---------------------------------------------------------------------------------------
def _homogeneous(t, wn, zeta, x0, v0):
    """Free response (no force) from x0, v0. Returns (x, v)."""
    if zeta < 1e-12:
        return (x0 * np.cos(wn * t) + v0 / wn * np.sin(wn * t),
                -x0 * wn * np.sin(wn * t) + v0 * np.cos(wn * t))
    if zeta < 1 - 1e-7:
        sig = zeta * wn
        wd = wn * math.sqrt(1 - zeta ** 2)
        e = np.exp(-sig * t)
        cs, sn = np.cos(wd * t), np.sin(wd * t)
        x = e * (x0 * cs + (v0 + sig * x0) / wd * sn)
        v = e * (v0 * cs - (sig * v0 + wn ** 2 * x0) / wd * sn)
        return x, v
    if zeta <= 1 + 1e-7:
        e = np.exp(-wn * t)
        b = v0 + wn * x0
        return e * (x0 + b * t), e * (v0 - wn * b * t)
    q = math.sqrt(zeta ** 2 - 1)
    s1, s2 = -wn * (zeta - q), -wn * (zeta + q)
    c1 = (v0 - s2 * x0) / (s1 - s2)
    c2 = (s1 * x0 - v0) / (s1 - s2)
    e1, e2 = np.exp(s1 * t), np.exp(s2 * t)
    return c1 * e1 + c2 * e2, c1 * s1 * e1 + c2 * s2 * e2


def _particular(t, p, wn, zeta, load):
    """Particular (forced) solution and its velocity. Returns (x, v)."""
    zero = np.zeros_like(t)
    if load.kind == "free" or load.F0 == 0:
        return zero, zero
    if load.kind == "step":
        return zero + load.F0 / p.k, zero
    w = load.omega
    r = w / wn
    if zeta < 1e-12 and abs(1 - r) < 1e-9:
        # undamped resonance: the amplitude grows linearly with time
        a = load.F0 / (2 * p.m * wn)
        return a * t * np.sin(wn * t), a * (np.sin(wn * t) + wn * t * np.cos(wn * t))
    amp = load.F0 / p.k * float(magnification(r, zeta))
    phi = math.atan2(2 * zeta * r, 1 - r * r)
    return amp * np.cos(w * t - phi), -amp * w * np.sin(w * t - phi)


@dataclass
class Response:
    t: np.ndarray
    x: np.ndarray             # m
    v: np.ndarray             # m/s
    force: np.ndarray         # N
    kinetic: np.ndarray       # J
    potential: np.ndarray     # J
    dissipated: np.ndarray    # J, cumulative energy removed by the damper
    work_in: np.ndarray       # J, cumulative work done by the applied force


def _cumtrapz(y, t):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def simulate(p, load, x0, v0, t_end, n=4000):
    """Exact response of m x'' + c x' + k x = F(t) from x(0) = x0, x'(0) = v0."""
    if not (math.isfinite(x0) and math.isfinite(v0)):
        raise ValueError("The initial conditions must be finite numbers.")
    if t_end <= 0 or n < 2:
        raise ValueError("The duration must be positive and at least two samples are needed.")
    pr = properties(p)
    t = np.linspace(0.0, t_end, int(n))
    xp0, vp0 = (float(a[0]) for a in _particular(np.zeros(1), p, pr.wn, pr.zeta, load))
    xh, vh = _homogeneous(t, pr.wn, pr.zeta, x0 - xp0, v0 - vp0)
    xp, vp = _particular(t, p, pr.wn, pr.zeta, load)
    x, v = xh + xp, vh + vp
    if load.kind == "harmonic":
        f = load.F0 * np.cos(load.omega * t)
    elif load.kind == "step":
        f = np.full_like(t, load.F0)
    else:
        f = np.zeros_like(t)
    return Response(t, x, v, f, 0.5 * p.m * v ** 2, 0.5 * p.k * x ** 2,
                    _cumtrapz(p.c * v ** 2, t), _cumtrapz(f * v, t))


def suggest_duration(p, load, x_scale_periods=8):
    """A sensible plot duration: long enough for the transient to die away and to show
    several cycles, but not so long that the oscillation is unreadable."""
    pr = properties(p)
    Tn = 2 * math.pi / pr.wn
    t = x_scale_periods * Tn
    if math.isfinite(pr.settling_time):
        t = max(t, 1.2 * pr.settling_time)
    if load.kind == "harmonic":
        t = max(t, 8 * 2 * math.pi / load.omega)
    return min(t, 60 * Tn)


def suggest_samples(p, load, t_end):
    """Enough samples to resolve the fastest oscillation (at least 40 per cycle)."""
    pr = properties(p)
    fastest = max(pr.wn, load.omega if load.kind == "harmonic" else 0.0)
    return int(min(40000, max(2000, 40 * t_end * fastest / (2 * math.pi))))
