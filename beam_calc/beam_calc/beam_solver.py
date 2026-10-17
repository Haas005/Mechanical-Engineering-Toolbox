"""Beam solver: shear, moment, slope and deflection using SymPy's Beam class.

Sign convention (what the UI uses):
  * Loads are entered positive DOWNWARD (point loads in N, distributed in N/m).
  * Linearly varying (triangular / trapezoidal) loads give the intensity at each end.
  * Applied moments are positive COUNTER-CLOCKWISE.
  * Deflection is returned positive UPWARD (so a sagging beam has negative y).
  * Moment is positive when it causes sagging (tension on the bottom fibre).
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam

SUPPORT_PRESETS = {
    "Simply supported": "simple",
    "Cantilever (fixed at left end)": "cantilever",
    "Propped cantilever (fixed left, roller right)": "propped",
    "Fixed-fixed": "fixed_fixed",
}


@dataclass
class PointLoad:
    magnitude: float  # N, positive downward
    position: float   # m


@dataclass
class UniformLoad:
    intensity: float  # N/m, positive downward
    start: float      # m
    end: float        # m


@dataclass
class LinearLoad:
    """Load that varies linearly from one intensity to another (triangle or trapezoid).

    start_intensity applies at x = start and end_intensity at x = end, both in N/m and
    positive downward. A triangle has one intensity equal to zero.
    """
    start_intensity: float  # N/m at x = start
    end_intensity: float    # N/m at x = end
    start: float            # m
    end: float              # m


@dataclass
class PointMoment:
    magnitude: float  # N*m, positive counter-clockwise
    position: float   # m


@dataclass
class BeamResult:
    length: float
    reactions: dict
    x: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    slope: np.ndarray
    deflection: np.ndarray
    expressions: dict = field(default_factory=dict)

    def extreme(self, name):
        arr = getattr(self, name)
        i = int(np.argmax(np.abs(arr)))
        return float(arr[i]), float(self.x[i])


def _validate(L, E, I, loads):
    if L <= 0:
        raise ValueError("Beam length must be positive.")
    if E <= 0 or I <= 0:
        raise ValueError("E and I must be positive.")
    for ld in loads:
        if isinstance(ld, (UniformLoad, LinearLoad)):
            if not (0 <= ld.start < ld.end <= L):
                raise ValueError(
                    f"Distributed load must satisfy 0 <= start < end <= L "
                    f"(got {ld.start} to {ld.end}).")
        elif not (0 <= ld.position <= L):
            raise ValueError(
                f"Load position {ld.position} m is outside the beam (0 to {L} m).")


def solve_beam(L, E, I, support, loads, n_points=801):
    """Solve a statically determinate or indeterminate single-span beam.

    support: one of 'simple', 'cantilever', 'propped', 'fixed_fixed'.
    loads: list of PointLoad / UniformLoad / LinearLoad / PointMoment.
    """
    _validate(L, E, I, loads)
    x = sp.symbols("x")
    beam = Beam(sp.Float(L), sp.Float(E), sp.Float(I))

    # Supports. Reaction symbols are named by location (R_0, M_0, R_5, ...),
    # so we collect whichever symbols apply_support created.
    Lq = sp.Float(L)
    if support == "simple":
        beam.apply_support(0, "pin")
        beam.apply_support(Lq, "roller")
    elif support == "cantilever":
        beam.apply_support(0, "fixed")
    elif support == "propped":
        beam.apply_support(0, "fixed")
        beam.apply_support(Lq, "roller")
    elif support == "fixed_fixed":
        beam.apply_support(0, "fixed")
        beam.apply_support(Lq, "fixed")
    else:
        raise ValueError(f"Unknown support type: {support}")

    unknowns = [ld[0] for ld in beam.applied_loads if isinstance(ld[0], sp.Symbol)]

    # Loads: SymPy's Beam treats UPWARD loads as positive, so flip signs.
    for ld in loads:
        if isinstance(ld, PointLoad):
            beam.apply_load(-sp.Float(ld.magnitude), sp.Float(ld.position), -1)
        elif isinstance(ld, UniformLoad):
            beam.apply_load(-sp.Float(ld.intensity), sp.Float(ld.start), 0,
                            end=sp.Float(ld.end))
        elif isinstance(ld, LinearLoad):
            # A trapezoid is a uniform load (the start intensity) plus a ramp (the change).
            slope = (ld.end_intensity - ld.start_intensity) / (ld.end - ld.start)
            if ld.start_intensity != 0:
                beam.apply_load(-sp.Float(ld.start_intensity), sp.Float(ld.start), 0,
                                end=sp.Float(ld.end))
            if slope != 0:
                beam.apply_load(-sp.Float(slope), sp.Float(ld.start), 1,
                                end=sp.Float(ld.end))
        elif isinstance(ld, PointMoment):
            beam.apply_load(-sp.Float(ld.magnitude), sp.Float(ld.position), -2)  # SymPy: +ve = clockwise

    beam.solve_for_reaction_loads(*unknowns)
    reactions = {}
    for sym, loc, order, _ in beam.applied_loads:
        if sym in beam.reaction_loads:
            kind = "Force" if order == -1 else "Moment"
            val = float(beam.reaction_loads[sym])
            if order == -2:
                val = -val  # report counter-clockwise positive
            reactions[f"{kind} at x = {float(loc):g} m"] = val

    x = sp.symbols("x")
    # SymPy's V and M have the opposite sign to the textbook convention
    # (V>0 when the left side pushes up, M>0 when sagging), so flip them.
    V = -beam.shear_force()
    M = -beam.bending_moment()

    # Integrate EI*y'' = M ourselves. (Beam.slope() raises an error for
    # fixed-fixed beams; the reactions it solves are still correct.)
    EIq = sp.Float(E) * sp.Float(I)
    C1, C2 = sp.symbols("C1 C2")
    slope = sp.integrate(M / EIq, x) + C1
    defl = sp.integrate(slope, x) + C2
    if support == "simple":
        sol = sp.solve([defl.subs(x, 0), defl.subs(x, Lq)], [C1, C2], dict=True)[0]
    else:  # fixed at x=0: y(0)=0 and slope(0)=0
        sol = {C1: 0, C2: 0}
    exprs = {
        "shear": V,
        "moment": M,
        "slope": slope.subs(sol),
        "deflection": defl.subs(sol),
    }
    xs = np.linspace(0, L, n_points)
    out = {}
    for name, e in exprs.items():
        f = sp.lambdify(x, e, modules=[{"SingularityFunction": _sing}, "numpy"])
        out[name] = np.broadcast_to(np.asarray(f(xs), dtype=float), xs.shape).copy()

    return BeamResult(
        length=L, reactions=reactions, x=xs,
        shear=out["shear"], moment=out["moment"],
        slope=out["slope"], deflection=out["deflection"],
        expressions=exprs,
    )


def _sing(x, a, n):
    """Numeric SingularityFunction <x - a>^n for the orders SymPy emits."""
    x = np.asarray(x, dtype=float)
    d = x - float(a)
    n = int(n)
    if n >= 0:
        return np.where(d >= 0, d ** n, 0.0)
    return np.zeros_like(d)  # point loads/moments (n<0) vanish away from a
