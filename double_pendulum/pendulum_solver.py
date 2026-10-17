"""Double pendulum physics: equations of motion, integration, energy, chaos map.

Model: two point masses on massless rigid rods, frictionless pivots, uniform gravity.

Coordinates:
    theta1, theta2  rod angles measured from the DOWNWARD vertical, in radians.
                    Positive angles swing the rod toward +x (to the right), i.e.
                    counter-clockwise. Both angles are absolute, not relative to rod 1.
    omega1, omega2  angular velocities (rad/s).

Everything in the public API is SI: kg, m, s, rad.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81

    def validate(self):
        for name in ("m1", "m2", "l1", "l2", "g"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive.")


def accelerations(theta1, theta2, omega1, omega2, p: Params):
    """Angular accelerations from the Lagrangian equations of motion.

    Works on scalars or numpy arrays.
    """
    d = theta1 - theta2
    den = 2 * p.m1 + p.m2 - p.m2 * np.cos(2 * d)
    a1 = (-p.g * (2 * p.m1 + p.m2) * np.sin(theta1)
          - p.m2 * p.g * np.sin(theta1 - 2 * theta2)
          - 2 * np.sin(d) * p.m2 * (omega2 ** 2 * p.l2 + omega1 ** 2 * p.l1 * np.cos(d))
          ) / (p.l1 * den)
    a2 = (2 * np.sin(d) * (omega1 ** 2 * p.l1 * (p.m1 + p.m2)
                           + p.g * (p.m1 + p.m2) * np.cos(theta1)
                           + omega2 ** 2 * p.l2 * p.m2 * np.cos(d))
          ) / (p.l2 * den)
    return a1, a2


def positions(theta1, theta2, p: Params):
    """Cartesian positions of the two bobs. Pivot at the origin, y up, x right.

    theta = 0 hangs straight down; positive theta swings the bob to the right (+x).
    """
    x1 = p.l1 * np.sin(theta1)
    y1 = -p.l1 * np.cos(theta1)
    x2 = x1 + p.l2 * np.sin(theta2)
    y2 = y1 - p.l2 * np.cos(theta2)
    return x1, y1, x2, y2


def energy(theta1, theta2, omega1, omega2, p: Params):
    """Return (kinetic, potential, total) energy in joules. Potential is zero at the pivot."""
    d = theta1 - theta2
    T = (0.5 * p.m1 * p.l1 ** 2 * omega1 ** 2
         + 0.5 * p.m2 * (p.l1 ** 2 * omega1 ** 2 + p.l2 ** 2 * omega2 ** 2
                         + 2 * p.l1 * p.l2 * omega1 * omega2 * np.cos(d)))
    V = -(p.m1 + p.m2) * p.g * p.l1 * np.cos(theta1) - p.m2 * p.g * p.l2 * np.cos(theta2)
    return T, V, T + V


@dataclass
class Trajectory:
    t: np.ndarray
    theta1: np.ndarray
    theta2: np.ndarray
    omega1: np.ndarray
    omega2: np.ndarray
    params: Params

    @property
    def xy(self):
        return positions(self.theta1, self.theta2, self.params)

    @property
    def energy(self):
        return energy(self.theta1, self.theta2, self.omega1, self.omega2, self.params)

    @property
    def relative_energy_drift(self):
        """Max |E(t) - E(0)| divided by a natural energy scale, a unit-free accuracy check."""
        _, _, E = self.energy
        scale = (self.params.m1 + self.params.m2) * self.params.g * self.params.l1 \
            + self.params.m2 * self.params.g * self.params.l2
        return float(np.max(np.abs(E - E[0])) / scale)


def simulate(p: Params, theta1, theta2, omega1, omega2, t_end, n_points=None,
             rtol=1e-10, atol=1e-10):
    """Integrate with an 8th-order Runge-Kutta method (DOP853) at tight tolerances.

    Returns a Trajectory sampled at n_points evenly spaced times (default 100 per second).
    """
    p.validate()
    if t_end <= 0:
        raise ValueError("Duration must be positive.")
    if n_points is None:
        n_points = max(2, int(round(100 * t_end)) + 1)

    def rhs(_t, y):
        a1, a2 = accelerations(y[0], y[1], y[2], y[3], p)
        return [y[2], y[3], a1, a2]

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), [theta1, theta2, omega1, omega2],
                    method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return Trajectory(sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], p)


def flip_time_grid(p: Params, n=80, t_end=10.0, dt=0.005, omega1=0.0, omega2=0.0):
    """Time until either rod first swings past the upright position (|theta| > pi).

    Evaluates an n-by-n grid of starting angles in [-pi, pi]^2 at once with a fixed-step
    fourth-order Runge-Kutta scheme on whole arrays. Cells that never flip within t_end
    are returned as t_end. This is a visualization of chaos, not a precision calculation.

    Returns (angles, flip_time) with flip_time[i, j] for theta1 = angles[j], theta2 = angles[i].
    """
    p.validate()
    angles = np.linspace(-np.pi, np.pi, n)
    T1, T2 = np.meshgrid(angles, angles)
    th1, th2 = T1.copy(), T2.copy()
    w1 = np.full_like(th1, omega1)
    w2 = np.full_like(th2, omega2)
    flip = np.full(th1.shape, t_end)
    done = (np.abs(th1) > np.pi) | (np.abs(th2) > np.pi)

    def f(a, b, c, d):
        acc1, acc2 = accelerations(a, b, c, d, p)
        return c, d, acc1, acc2

    steps = int(round(t_end / dt))
    for k in range(steps):
        k1 = f(th1, th2, w1, w2)
        k2 = f(th1 + 0.5 * dt * k1[0], th2 + 0.5 * dt * k1[1],
               w1 + 0.5 * dt * k1[2], w2 + 0.5 * dt * k1[3])
        k3 = f(th1 + 0.5 * dt * k2[0], th2 + 0.5 * dt * k2[1],
               w1 + 0.5 * dt * k2[2], w2 + 0.5 * dt * k2[3])
        k4 = f(th1 + dt * k3[0], th2 + dt * k3[1], w1 + dt * k3[2], w2 + dt * k3[3])
        th1 = th1 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        th2 = th2 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        w1 = w1 + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        w2 = w2 + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        newly = (~done) & ((np.abs(th1) > np.pi) | (np.abs(th2) > np.pi))
        flip[newly] = (k + 1) * dt
        done |= newly
        if done.all():
            break
    return angles, flip
