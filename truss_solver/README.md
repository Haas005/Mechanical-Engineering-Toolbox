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
- Checks every compression member for buckling (Euler for slender members, Johnson parabola for stocky ones), reports the safety factor, and flags members below the required value on the plot
- Exports the member results as CSV

## Files

- `truss_solver.py` the solver (no UI code); `solve_truss()` returns a `TrussResult`
- `app.py` the Streamlit interface
- `test_truss_solver.py` checks the solver and the buckling formulas against hand calculations (`python test_truss_solver.py`)

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

## Buckling check

For a compression member of length `L`, area `A` and second moment `I`, the radius of gyration is `r = √(I/A)` and the slenderness is `λ = K·L / r`. The transition slenderness is `λc = π√(2E/σy)`.

```
λ ≥ λc  (slender):  σcr = π² E / λ²                      (Euler)
λ <  λc (stocky):   σcr = σy − σy² λ² / (4 π² E)         (Johnson parabola)
```

The two curves meet at `λc`, where `σcr = σy / 2`. The critical load is `Pcr = σcr · A` and the safety factor is `Pcr / |N|`. Tension and zero-force members are not checked. Members under the required safety factor are marked on the plot and in the table; one below 1.0 is predicted to buckle.

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
- The buckling check treats each member as an isolated pin-ended column (adjust K if you want something else) and uses the I you enter, which should be the smallest second moment of area of the cross-section. Local buckling and connection effects are not modelled
