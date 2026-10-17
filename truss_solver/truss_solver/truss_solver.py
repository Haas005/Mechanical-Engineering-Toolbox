"""2D truss solver using the direct stiffness method.

Units used by the public API (what the UI works in):
    coordinates in m, E in GPa, A in cm^2, loads in kN.
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
        if m.E <= 0 or m.A <= 0:
            raise ValueError(f"Member {k}: E and A must be positive.")
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
