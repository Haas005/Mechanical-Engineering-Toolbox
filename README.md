# Mechanical Engineering Tools

A collection of Python tools and simulations built alongside my mechanical engineering coursework, currently dynamics and solid mechanics. The goal is to use code to check hand calculations, explore how systems behave, and build intuition for the underlying mechanics.

## Projects

| Project | Course topic | What it does | Tech |
|---|---|---|---|
| [beam_calc](beam_calc) | Solid mechanics | Shear, moment, slope and deflection for single-span beams, with a web interface | Python, SymPy, NumPy, Matplotlib, Streamlit |

### beam_calc

- Supports simply supported, cantilever, propped cantilever and fixed-fixed beams
- Handles point loads, partial-span distributed loads and applied moments
- Reports support reactions, maximum shear, moment and deflection, and where each occurs
- Plots shear, moment, slope and deflection along the beam
- Shows the exact symbolic expressions and exports results to CSV
- Verified against textbook closed-form results (for example 5wL⁴/384EI for a simply supported beam under a uniform load); run `python test_beam_solver.py` inside the folder to repeat the checks

See [beam_calc/README.md](beam_calc/README.md) for sign conventions, limitations and details.

## Getting started

Requirements: Python 3.9 or newer.

Each project is self-contained, with its own `requirements.txt` and README. To run one:

```
cd beam_calc
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Streamlit apps must be launched with the `streamlit` command, not by running the file directly.

## Repository layout

```
Mechanical-Engineering-Tools/
├── README.md              this overview
└── beam_calc/
    ├── app.py             Streamlit interface
    ├── beam_solver.py     solver (no UI code)
    ├── test_beam_solver.py  checks against textbook formulas
    ├── requirements.txt
    └── README.md
```

## Notes

These tools assume idealized models (linear elastic material, small deflections, constant cross-section for beams). They are for learning and for checking hand calculations, not for design work.
