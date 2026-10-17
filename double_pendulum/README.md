# Double pendulum

An interactive double pendulum with a smooth animation, motion plots, a sensitivity demonstration, and a map of where the chaos lives. The physics is derived from Lagrange's equations and integrated with an 8th-order Runge–Kutta solver, with an energy-drift check to show the numbers can be trusted.

## Run it

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Streamlit apps have to be started with the `streamlit` command; running `app.py` directly does nothing.

## What's in the app

- **Animation:** play, pause and scrub through the motion, with an adjustable trail and playback speed
- **Motion plots:** angles, angular velocities, phase portraits and kinetic, potential and total energy, plus a CSV download
- **Sensitivity (chaos):** runs the same pendulum twice, with the second start nudged by a tiny amount, and shows how fast the two paths separate on a log scale
- **Chaos map:** for every pair of starting angles, the time until a rod first flips over the top. Smooth regions are predictable and speckled regions are chaotic
- **How it works:** the Lagrangian and the equations of motion
- Presets: chaotic swing, horizontal release, small swings, both normal modes, almost upright, heavy upper bob
- Works in light and dark mode, with chart colors checked for color-blind readability

## Files

- `pendulum_solver.py` the physics (no UI code): equations of motion, integrator, energy, chaos-map grid
- `app.py` the Streamlit interface
- `test_pendulum_solver.py` physics checks (`python test_pendulum_solver.py`)

## The physics

Two point masses m₁ and m₂ hang from rigid massless rods of length l₁ and l₂. The angles θ₁ and θ₂ are measured from straight down, and both are absolute (θ₂ is not relative to rod 1).

The Lagrangian is

```
L = ½(m₁+m₂) l₁² θ̇₁² + ½ m₂ l₂² θ̇₂² + m₂ l₁ l₂ θ̇₁ θ̇₂ cos(θ₁−θ₂)
    + (m₁+m₂) g l₁ cos θ₁ + m₂ g l₂ cos θ₂
```

Applying d/dt(∂L/∂θ̇) − ∂L/∂θ = 0 for each angle gives two coupled second-order equations, solved for θ̈₁ and θ̈₂ and integrated numerically. The chaos map uses a faster fixed-step fourth-order Runge–Kutta scheme applied to a whole grid of starting points at once.

## How it is checked

`test_pendulum_solver.py` verifies:

- The closed-form accelerations match an independent derivation from the Lagrangian using SymPy, at 200 random states and random masses, lengths and gravity
- Total energy is conserved over 30 s of violently chaotic motion
- In the small-angle limit, the two normal-mode periods match the theory, ω² = (g/l)(2 ∓ √2), and the mode shapes hold
- With a very light second bob, the upper pendulum behaves as a simple pendulum, even at large amplitude
- Mirror symmetry holds
- A 1e-8 rad difference grows beyond 1 rad in a chaotic run but stays tiny for small swings
- The chaos-map grid agrees with the adaptive integrator on random cells

The tests use SymPy, which the app itself does not need.

## Conventions and limits

- SI units in the solver; the app shows angles in degrees
- Angle 0 hangs straight down; positive angles swing to the right (counter-clockwise)
- Frictionless pivots, rigid massless rods, point masses and uniform gravity. No air resistance, so energy is conserved
- The chaos map is a picture of the chaos, not a precision calculation
