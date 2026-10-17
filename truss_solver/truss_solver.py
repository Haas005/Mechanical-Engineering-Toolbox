"""2D truss solver using the direct stiffness method.

Units used by the public API (what the UI works in):
    coordinates in m, E in GPa, A in cm^2, I in cm^4, loads in kN, yield strength in MPa.
Results are returned in: displacements mm, forces kN, stress MPa.

Sign conventions:
    * x positive right, y positive up.
    * Applied nodal loads: Fx positive right, Fy positive UP.
    * Member axial force: positive = TENSION, negative = COMPRESSION.
    * Reactions are the forces the supports exert on the truss (same axes).
"""
from dataclasses import dataclass, field

import numpy as np

SUPPORT_TYPES = {
    "Pin (fixed x and y)": (True, True),
    "Roller (free in x, fixed y)": (False, True),
    "Roller (fixed x, free in y)": (True, False),
}


@dataclass
class Node:
    x: float  # m
    y: float  # m


@dataclass
class Member:
    i: int           # start node index (0-based)
    j: int           # end node index (0-based)
    E: float = 200.0  # GPa
    A: float = 10.0   # cm^2
    I: float = 20.0   # cm^4, smallest (weak-axis) second moment of area; used for buckling only


@dataclass
class Support:
    node: int         # 0-based
    fix_x: bool
    fix_y: bool


@dataclass
class NodalLoad:
    node: int         # 0-based
    fx: float = 0.0   # kN, + right
    fy: float = 0.0   # kN, + up


@dataclass
class TrussResult:
    displacements: np.ndarray        # (n_nodes, 2) in mm
    member_force: np.ndarray         # kN, + tension
    member_stress: np.ndarray        # MPa
    member_elongation: np.ndarray    # mm, + = lengthening
    member_length: np.ndarray        # m
    reactions: dict = field(default_factory=dict)  # {(node, 'x'|'y'): kN}
    equilibrium_error: dict = field(default_factory=dict)
    n_members: int = 0
    n_reactions: int = 0
    n_joints: int = 0

    @property
    def indeterminacy(self):
        """m + r - 2j: 0 = statically determinate, >0 = indeterminate."""
        return self.n_members + self.n_reactions - 2 * self.n_joints


def _validate(nodes, members, supports, loads):
    n = len(nodes)
    if n < 2:
        raise ValueError("Add at least two nodes.")
    if not members:
        raise ValueError("Add at least one member.")
    if not supports:
        raise ValueError("Add at least one support, otherwise the truss is free to move.")
    for k, m in enumerate(members, start=1):
        if not (0 <= m.i < n and 0 <= m.j < n):
            raise ValueError(f"Member {k} refers to a node number that does not exist.")
        if m.i == m.j:
            raise ValueError(f"Member {k} connects node {m.i + 1} to itself.")
        if m.E <= 0 or m.A <= 0 or m.I <= 0:
            raise ValueError(f"Member {k}: E, A and I must be positive.")
        if np.hypot(nodes[m.j].x - nodes[m.i].x, nodes[m.j].y - nodes[m.i].y) < 1e-9:
            raise ValueError(
                f"Member {k} has zero length (nodes {m.i + 1} and {m.j + 1} coincide).")
    for s in supports:
        if not 0 <= s.node < n:
            raise ValueError(f"A support refers to node {s.node + 1}, which does not exist.")
    for ld in loads:
        if not 0 <= ld.node < n:
            raise ValueError(f"A load refers to node {ld.node + 1}, which does not exist.")


def solve_truss(nodes, members, supports, loads):
    """Solve for displacements, reactions and member forces."""
    _validate(nodes, members, supports, loads)
    n = len(nodes)
    ndof = 2 * n

    K = np.zeros((ndof, ndof))
    lengths = np.zeros(len(members))
    cs = np.zeros((len(members), 2))
    EA = np.zeros(len(members))  # N (SI)

    for k, m in enumerate(members):
        dx = nodes[m.j].x - nodes[m.i].x
        dy = nodes[m.j].y - nodes[m.i].y
        L = float(np.hypot(dx, dy))
        c, s = dx / L, dy / L
        ea = (m.E * 1e9) * (m.A * 1e-4)          # Pa * m^2 = N
        lengths[k], cs[k], EA[k] = L, (c, s), ea
        kk = ea / L * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ])
        dofs = [2 * m.i, 2 * m.i + 1, 2 * m.j, 2 * m.j + 1]
        K[np.ix_(dofs, dofs)] += kk

    F = np.zeros(ndof)  # N
    for ld in loads:
        F[2 * ld.node] += ld.fx * 1e3
        F[2 * ld.node + 1] += ld.fy * 1e3

    fixed = set()
    for s in supports:
        if s.fix_x:
            fixed.add(2 * s.node)
        if s.fix_y:
            fixed.add(2 * s.node + 1)
    fixed = sorted(fixed)
    free = [d for d in range(ndof) if d not in fixed]
    if not fixed:
        raise ValueError("The supports do not restrain any displacement.")

    u = np.zeros(ndof)
    if free:
        Kff = K[np.ix_(free, free)]
        # Mechanism check: a stable truss has a full-rank reduced stiffness matrix.
        rank = np.linalg.matrix_rank(Kff, tol=1e-9 * np.max(np.abs(Kff)))
        if rank < len(free):
            raise ValueError(
                "The truss is unstable (a mechanism): it can move or fold without "
                "stretching any member. Check that the supports remove all rigid-body "
                "motion (at least 3 reaction components, not all parallel or concurrent), "
                "that every node is connected by members, and that the members form triangles.")
        u[free] = np.linalg.solve(Kff, F[free])

    R = K @ u - F  # support reactions (N), nonzero only at fixed dofs
    reactions = {}
    for d in fixed:
        reactions[(d // 2, "x" if d % 2 == 0 else "y")] = float(R[d]) / 1e3  # kN

    force = np.zeros(len(members))
    elong = np.zeros(len(members))
    for k, m in enumerate(members):
        c, s = cs[k]
        d_i = u[[2 * m.i, 2 * m.i + 1]]
        d_j = u[[2 * m.j, 2 * m.j + 1]]
        elong[k] = (d_j - d_i) @ np.array([c, s])      # m
        force[k] = EA[k] / lengths[k] * elong[k]       # N

    stress = np.array([force[k] / (members[k].A * 1e-4) / 1e6 for k in range(len(members))])

    # Global equilibrium of applied loads + reactions (should be ~0).
    sum_fx = F[0::2].sum() + sum(v for (nd, a), v in reactions.items() if a == "x") * 1e3
    sum_fy = F[1::2].sum() + sum(v for (nd, a), v in reactions.items() if a == "y") * 1e3
    mom = 0.0
    for nd in range(n):
        fx = F[2 * nd] + reactions.get((nd, "x"), 0.0) * 1e3
        fy = F[2 * nd + 1] + reactions.get((nd, "y"), 0.0) * 1e3
        mom += nodes[nd].x * fy - nodes[nd].y * fx
    eq = {"sum_Fx_kN": sum_fx / 1e3, "sum_Fy_kN": sum_fy / 1e3, "sum_M_origin_kNm": mom / 1e3}

    return TrussResult(
        displacements=u.reshape(n, 2) * 1e3,
        member_force=force / 1e3,
        member_stress=stress,
        member_elongation=elong * 1e3,
        member_length=lengths,
        reactions=reactions,
        equilibrium_error=eq,
        n_members=len(members),
        n_reactions=len(fixed),
        n_joints=n,
    )


# ----------------------------------------------------------------------------------------
# Buckling check
# ----------------------------------------------------------------------------------------
@dataclass
class BucklingResult:
    slenderness: np.ndarray             # K L / r for every member (dimensionless)
    transition_slenderness: np.ndarray  # lambda_c = pi sqrt(2E / sigma_y) per member
    critical_stress: np.ndarray         # MPa
    critical_load: np.ndarray           # kN
    safety_factor: np.ndarray           # Pcr / |N| for compression members, nan otherwise
    status: np.ndarray                  # Buckles / Below required / OK / Tension / Zero force
    regime: np.ndarray                  # Euler / Johnson


def check_buckling(result, members, yield_strength=250.0, K=1.0, required_sf=2.0):
    """Check every compression member against column buckling.

    Slender members (K L / r >= lambda_c, with lambda_c = pi sqrt(2 E / sigma_y)) use Euler:
        sigma_cr = pi^2 E / (K L / r)^2
    Stocky members use the Johnson parabola, which meets Euler at lambda_c where sigma_cr =
    sigma_y / 2 and rises to sigma_y as the slenderness goes to zero:
        sigma_cr = sigma_y - sigma_y^2 (K L / r)^2 / (4 pi^2 E)

    The critical load is sigma_cr * A, and the safety factor is critical load / compressive force.
    Tension and zero-force members are reported as not applicable (nan).

    yield_strength in MPa, K the effective-length factor (1.0 for pin-ended truss members),
    required_sf the safety factor you want to see (members below it are flagged).
    """
    if yield_strength <= 0:
        raise ValueError("Yield strength must be positive.")
    if K <= 0:
        raise ValueError("The effective length factor K must be positive.")
    if required_sf < 1:
        raise ValueError("The required safety factor must be at least 1.")

    n = len(members)
    L = np.asarray(result.member_length, dtype=float)
    A = np.array([m.A for m in members]) * 1e-4            # m^2
    I = np.array([m.I for m in members]) * 1e-8            # m^4
    E = np.array([m.E for m in members]) * 1e9             # Pa
    sy = yield_strength * 1e6                              # Pa
    r = np.sqrt(I / A)                                     # radius of gyration, m
    lam = K * L / r
    lam_c = np.pi * np.sqrt(2 * E / sy)

    euler = lam >= lam_c
    sigma_cr = np.where(euler, np.pi ** 2 * E / lam ** 2,
                        sy - sy ** 2 * lam ** 2 / (4 * np.pi ** 2 * E))
    p_cr = sigma_cr * A                                    # N

    force = np.asarray(result.member_force, dtype=float)   # kN, + tension
    tol = 1e-6 * max(1.0, float(np.max(np.abs(force))))
    compressive = np.where(force < -tol, -force, np.nan)   # kN
    sf = (p_cr / 1e3) / compressive

    status = np.full(n, "OK", dtype=object)
    status[force > tol] = "Tension"
    status[np.abs(force) <= tol] = "Zero force"
    comp = force < -tol
    status[comp & (sf < required_sf)] = "Below required"
    status[comp & (sf < 1.0)] = "Buckles"

    return BucklingResult(
        slenderness=lam, transition_slenderness=lam_c,
        critical_stress=sigma_cr / 1e6, critical_load=p_cr / 1e3,
        safety_factor=sf, status=status,
        regime=np.where(euler, "Euler", "Johnson").astype(object))
