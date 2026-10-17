# Mechanical Engineering Toolbox

Interactive Python tools for solid mechanics and dynamics. Each one is a small web app you run locally, built on a solver that is tested against hand calculations, closed-form results, or an independent numerical method.

I built these while taking Dynamics and Solid Mechanics, to understand the theory by making it computable and to check my own homework.

## The tools

| Tool | Course | What it does |
| --- | --- | --- |
| [`beam_calc`](beam_calc) | Solid mechanics | Shear, bending moment, slope and deflection of a beam with any mix of supports and loads |
| [`truss_solver`](truss_solver) | Solid mechanics | 2D pin-jointed truss: member forces, reactions, deformed shape, and a column buckling check |
| [`mohr_circle`](mohr_circle) | Solid mechanics | Plane stress: Mohr's circle, principal stresses, rotated stress elements, yield criteria, strains |
| [`double_pendulum`](double_pendulum) | Dynamics | A chaotic double pendulum, animated, with a map of where the chaos lives |
| [`spring_mass`](spring_mass) | Dynamics | Damped spring-mass system: free, step and forced response, resonance and vibration isolation |

## Run any of them

You need Python 3.10 or newer. Open a terminal, go into a tool's folder, and run:

```
cd beam_calc
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Use `python -m streamlit`, not `python app.py`: Streamlit apps have to be started with the `streamlit` command. Press `Ctrl+C` in the terminal to stop an app. To run two at once, give the second a different port: `python -m streamlit run app.py --server.port 8502`.

Each folder also has a test file, for example `python test_beam_solver.py`. A passing run ends with `ALL PASSED`.

## What's in each tool

### Beam calculator
Pin, roller and fixed supports in any combination (so simply supported, cantilever, overhanging and fixed-fixed beams all work), with point loads, uniform loads, triangular and trapezoidal loads, and applied moments. It draws the load diagram, solves the reactions, and plots shear, moment, slope and deflection. Statically indeterminate beams are solved with SymPy's singularity functions.

### Truss solver
Enter nodes, members, supports and loads in editable tables, or start from a preset (simple triangle, Pratt bridge, cantilever truss). The direct stiffness method gives member forces (blue for tension, red for compression), stresses, reactions and the deformed shape. It reports static determinacy, detects mechanisms, and checks every compression member for buckling with Euler's formula or the Johnson parabola, flagging any below the safety factor you set.

### Mohr's circle
Enter σx, σy and τxy. The app shows the circle with the original and rotated element, principal stresses and angles, maximum shear (including the out-of-plane value), four stress elements side by side, von Mises and Tresca safety factors on the yield envelope, and the strains from Hooke's law.

### Double pendulum
The equations of motion come from Lagrange's method and are integrated with an 8th-order Runge-Kutta solver, with an energy-drift check to show the numbers can be trusted. The app has a smooth animation, motion plots, a demonstration of sensitive dependence on starting conditions, and a map of the time until the first flip for every pair of starting angles.

### Damped spring-mass system
An animated mass, spring and damper with free vibration, a step force, or a harmonic force. It shows the displacement, velocity, phase portrait and energy balance, plus magnification, phase lag and transmissibility against frequency. It also reports natural frequency, damping ratio, log decrement, overshoot and settling time.

## How the tools are checked

A solver that looks right is not the same as one that is right, so every tool is tested against something independent:

- **Closed-form results:** textbook beam deflections, truss member forces, Euler buckling loads, step-response overshoot
- **Independent methods:** Mohr's circle against the eigenvalues of the stress tensor, spring-mass against a numerical ODE solver, the pendulum's equations against a SymPy derivation from the Lagrangian
- **Conservation and balance checks:** equilibrium of every truss solution, energy conservation for the pendulum, work in equals energy stored plus energy dissipated for the spring-mass system
- **Bad input:** unstable structures, zero-length members, non-physical values and similar are rejected with a plain message

The apps were also run in a real browser and checked in light and dark mode.

## Project layout

```
Mechanical-Engineering-Toolbox/
  beam_calc/         beam_solver.py, app.py, test_beam_solver.py, README.md, requirements.txt
  truss_solver/      truss_solver.py, app.py, test_truss_solver.py, README.md, requirements.txt
  mohr_circle/       mohr_solver.py, app.py, test_mohr_solver.py, README.md, requirements.txt
  double_pendulum/   pendulum_solver.py, app.py, test_pendulum_solver.py, README.md, requirements.txt
  spring_mass/       vibration_solver.py, app.py, test_vibration_solver.py, README.md, requirements.txt
```

In every tool the physics lives in one `*_solver.py` file with no user-interface code, so it can be imported and tested on its own. The `app.py` file is only the Streamlit interface.

## Built with

Python, [Streamlit](https://streamlit.io) for the interfaces, NumPy and SciPy for the numerics, SymPy for the beam solver and for test derivations, and Matplotlib and Plotly for the plots.
