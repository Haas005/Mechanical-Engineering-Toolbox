# Beam calculator

Shear, moment, slope and deflection for single-span beams, built on SymPy's
`Beam` class with a Streamlit interface.

## Run it

    pip install -r requirements.txt
    streamlit run app.py

## Files

- `beam_solver.py`      the engine (no UI code); `solve_beam()` returns a `BeamResult`
- `app.py`              the Streamlit interface
- `test_beam_solver.py` checks the solver against textbook formulas, including whole moment and deflection curves for triangular loads (`python test_beam_solver.py`)

## Supports
Simply supported, cantilever, propped cantilever, fixed-fixed.

## Loads
Point loads, uniform distributed loads over any part of the span, triangular and trapezoidal loads (the intensity varies in a straight line between the start and end), and applied moments. A load diagram shows what you entered, so you can check directions before reading the results.

## Sign convention
- Loads positive downward; applied moments positive counter-clockwise
- Shear positive when the left segment is pushed up; moment positive when sagging
- Deflection positive upward (a sagging beam has negative deflection)

## Limits
Prismatic beam (constant EI), small deflections, linear elastic, no axial load.
Distributed loads are linear (uniform, triangular, trapezoidal); curved load shapes are not supported.
