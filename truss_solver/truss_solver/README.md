# 2D truss solver

Analyzes statically determinate and indeterminate 2D pin-jointed trusses with the direct stiffness method, and shows the result in a Streamlit app: member forces coloured by tension and compression, reactions, node displacements, and the deformed shape.

## Run it

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Streamlit apps have to be started with the `streamlit` command; running `app.py` directly does nothing.

## What it does

- Takes nodes, members, supports and nodal loads from editable tables
- Includes presets: simple triangle, Pratt bridge truss, cantilever truss, blank
- Reports member axial forces (tension positive), stresses, elongations, reactions and node displacements
- Draws the truss with tension in blue, compression in red and zero-force members in grey, with force values on every member
- Overlays the deformed shape at an adjustable scale
- Reports the degree of static indeterminacy (m + r − 2j) and detects unstable structures
- Checks global equilibrium (sum of forces and moments) on every solve
- Exports the member results as CSV

## Files

- `truss_solver.py` the solver (no UI code); `solve_truss()` returns a `TrussResult`
- `app.py` the Streamlit interface
- `test_truss_solver.py` checks the solver against hand calculations (`python test_truss_solver.py`)

## How it works

For each member the 4×4 element stiffness matrix in global coordinates is

```
k = (EA / L) * [  c²   cs  -c²  -cs
                  cs   s²  -cs  -s²
                 -c²  -cs   c²   cs
                 -cs  -s²   cs   s² ]
```

where `c` and `s` are the cosine and sine of the member angle. The element matrices are assembled into the global matrix `K`, the supported degrees of freedom are removed, and the remaining system `K_ff u_f = F_f` is solved for the displacements. Reactions come from `R = K u − F`, and each member's axial force from its elongation, `N = (EA / L) * Δ`.

If the reduced stiffness matrix is rank deficient, the truss is a mechanism and the solver says so instead of returning meaningless numbers.

## Conventions

- Units: coordinates in m, E in GPa, A in cm², loads in kN. Results are in kN, MPa and mm.
- x positive to the right, y positive up; Fy is positive upward.
- Member force is positive in tension, negative in compression.
- Nodes are numbered by row in the Nodes table (first row is node 1).
- Support types: pin (fixes x and y), roller free in x (fixes y), roller free in y (fixes x).

## Limits

- Pin-jointed members carrying axial load only (no bending)
- Linear elastic material and small displacements
- Loads applied at nodes only
- Buckling of compression members is not checked
