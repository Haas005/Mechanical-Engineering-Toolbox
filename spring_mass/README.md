# Damped spring-mass system

An interactive mass-spring-damper explorer with a live animation, time response, frequency response and vibration-isolation plots. The response is computed from the exact closed-form solution (no time-stepping error) and checked against a numerical ODE solver in every damping regime.

## Run it

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Streamlit apps have to be started with the `streamlit` command; running `app.py` directly does nothing.

## What's in the app

- **Animation:** the mass, spring and damper moving, with the displacement trace building underneath. Play, pause or scrub. An orange arrow shows the applied force
- **Time response:** displacement (with the decay envelope, static deflection or steady-state amplitude marked), velocity, phase portrait and energy (kinetic, potential, total, dissipated and work done by the force), plus a CSV download
- **Frequency response:** magnification, phase lag and transmissibility against the frequency ratio r = ω/ωn, with your damping ratio compared to reference values and a marker at your forcing frequency
- **How it works:** the equations
- Inputs: mass, stiffness, damping (as ζ or as c), starting displacement and velocity, and the applied force: none, a step, or a harmonic force at any frequency ratio
- Presets: underdamped, undamped, critically damped, overdamped, step force, below resonance, at resonance, vibration isolation, undamped resonance (grows without limit)
- Works in light and dark mode

## Quantities reported

Natural frequency, damping ratio, damped period, log decrement, quality factor, 2 % settling time, step overshoot, resonance peak and half-power bandwidth, steady-state amplitude, phase lag, and transmitted force.

## Files

- `vibration_solver.py` the physics (no UI code): free, step and harmonic responses, frequency response, energy bookkeeping
- `app.py` the Streamlit interface
- `test_vibration_solver.py` checks (`python test_vibration_solver.py`)

## The physics

```
m x'' + c x' + k x = F(t)        wn = √(k/m)      ζ = c / (2√(km))
```

The free response depends on ζ: oscillating decay for ζ < 1, the fastest non-oscillating return at ζ = 1, and slow creep for ζ > 1. A force adds a particular solution (a constant F₀/k for a step, a steady oscillation for a harmonic force), and the free response is fitted to the starting conditions.

For a harmonic force F₀ cos ωt, with r = ω/ωn, the steady-state amplitude and phase lag are

```
X = (F₀/k) / √((1 − r²)² + (2ζr)²)        tan φ = 2ζr / (1 − r²)
```

## How it is checked

`test_vibration_solver.py` runs about 100 checks:

- The free response matches a numerical ODE solver (DOP853, tight tolerances) for ζ from 0 to 4, including right at critical damping, on random systems
- The harmonic response, including the start-up transient, matches the ODE solver, and so does undamped resonance with its linear growth
- Starting conditions are honoured, and the undamped period and energy are right
- The log decrement equals the measured amplitude ratio one damped period apart
- Critical and over damping never oscillate
- The energy removed equals the integral of c v², with the error shown to shrink 4× when the time step is halved
- The work done by the force equals the change in energy plus the energy dissipated
- Step response: final value F₀/k, overshoot e^(−πζ/√(1−ζ²)), peak time π/ωd, and 2 % settling time (including 5.83/ωn for critical damping)
- Steady-state amplitude and phase from a long run match the formulas, resonance gives X = F₀/(cωn) and 90°, and the peak of the magnification curve is where theory says
- Transmissibility equals 1 at r = √2 for every damping ratio
- Bad input (zero mass, negative stiffness, zero forcing frequency, and so on) is rejected

The tests use SciPy, which the app also needs.

## Conventions and limits

- SI units in the solver. The app shows displacement in mm and velocity in mm/s
- x is measured from the static equilibrium position, so gravity does not appear. Positive is to the right
- Linear spring, viscous damping, single degree of freedom, force applied directly to the mass
- The animation is not to scale: the motion is magnified so it can be seen. For fast systems it plays in slow motion (the caption says by how much)
