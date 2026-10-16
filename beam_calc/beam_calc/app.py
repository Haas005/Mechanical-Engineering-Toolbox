"""Beam calculator UI. Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from beam_solver import (SUPPORT_PRESETS, PointLoad, PointMoment, UniformLoad,
                         solve_beam)

st.set_page_config(page_title="Beam Calculator", layout="wide")
st.title("Beam shear / moment / deflection calculator")
st.caption("Single-span beams, SI units. Loads are positive downward; "
           "applied moments are positive counter-clockwise.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Beam")
    L = st.number_input("Length L (m)", min_value=0.1, value=5.0, step=0.5)
    support_label = st.selectbox("Supports", list(SUPPORT_PRESETS))
    support = SUPPORT_PRESETS[support_label]

    st.subheader("Material and section")
    E_GPa = st.number_input("Young's modulus E (GPa)", min_value=0.1,
                            value=200.0, step=10.0,
                            help="Steel ~200, aluminium ~69, timber ~11")
    I_unit = st.selectbox("Second moment of area I in", ["cm^4", "mm^4", "m^4"])
    I_val = st.number_input(f"I ({I_unit})", min_value=1e-6, value=8000.0,
                            format="%.4f")
    I_m4 = I_val * {"cm^4": 1e-8, "mm^4": 1e-12, "m^4": 1.0}[I_unit]
    E_Pa = E_GPa * 1e9

# ---------------------------------------------------------------- loads
st.subheader("Loads")
st.write("Edit the tables below. Add rows with the + at the bottom of each table.")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Point loads**")
    pl_df = st.data_editor(
        pd.DataFrame({"Load (N)": [10000.0], "Position (m)": [2.5]}),
        num_rows="dynamic", key="pl", width="stretch")
with c2:
    st.markdown("**Distributed loads**")
    ul_df = st.data_editor(
        pd.DataFrame({"Intensity (N/m)": [0.0], "Start (m)": [0.0], "End (m)": [5.0]}),
        num_rows="dynamic", key="ul", width="stretch")
with c3:
    st.markdown("**Applied moments**")
    pm_df = st.data_editor(
        pd.DataFrame({"Moment (N·m)": pd.Series(dtype=float),
                      "Position (m)": pd.Series(dtype=float)}),
        num_rows="dynamic", key="pm", width="stretch")

loads = []
for _, r in pl_df.dropna().iterrows():
    if r.iloc[0] != 0:
        loads.append(PointLoad(float(r.iloc[0]), float(r.iloc[1])))
for _, r in ul_df.dropna().iterrows():
    if r.iloc[0] != 0:
        loads.append(UniformLoad(float(r.iloc[0]), float(r.iloc[1]), float(r.iloc[2])))
for _, r in pm_df.dropna().iterrows():
    if r.iloc[0] != 0:
        loads.append(PointMoment(float(r.iloc[0]), float(r.iloc[1])))

if not loads:
    st.info("Add at least one non-zero load to see results.")
    st.stop()

# ---------------------------------------------------------------- solve
try:
    with st.spinner("Solving..."):
        res = solve_beam(L, E_Pa, I_m4, support, loads)
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # SymPy can fail on odd load combinations
    st.error(f"The solver could not handle this configuration: {err}")
    st.stop()

# ---------------------------------------------------------------- results
st.subheader("Results")
m1, m2, m3 = st.columns(3)
v, xv = res.extreme("shear")
m, xm = res.extreme("moment")
d, xd = res.extreme("deflection")
m1.metric("Max |shear|", f"{abs(v):,.1f} N", f"at x = {xv:.2f} m", delta_color="off")
m2.metric("Max |moment|", f"{abs(m):,.1f} N·m", f"at x = {xm:.2f} m", delta_color="off")
m3.metric("Max |deflection|", f"{abs(d) * 1000:.3f} mm", f"at x = {xd:.2f} m",
          delta_color="off")

with st.expander("Support reactions", expanded=True):
    st.table(pd.DataFrame(
        {"Value": [f"{val:,.2f} {'N' if k.startswith('Force') else 'N·m'}"
                   for k, val in res.reactions.items()]},
        index=list(res.reactions)))

# Plots
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
series = [
    ("Shear force V (N)", res.shear, "tab:blue", 1.0),
    ("Bending moment M (N·m)", res.moment, "tab:red", 1.0),
    ("Slope (rad)", res.slope, "tab:green", 1.0),
    ("Deflection (mm, + up)", res.deflection, "tab:purple", 1000.0),
]
for ax, (label, y, colour, scale) in zip(axes, series):
    ax.plot(res.x, y * scale, color=colour, lw=2)
    ax.fill_between(res.x, 0, y * scale, color=colour, alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Position x (m)")
fig.tight_layout()
st.pyplot(fig)

# Exact expressions
with st.expander("Symbolic expressions (SymPy)"):
    st.caption("<x - a>^n is a singularity function: zero for x < a.")
    for name in ("shear", "moment", "slope", "deflection"):
        st.markdown(f"**{name.capitalize()}**")
        st.code(str(res.expressions[name]), language="python")

# Data export
df = pd.DataFrame({"x_m": res.x, "shear_N": res.shear, "moment_Nm": res.moment,
                   "slope_rad": res.slope, "deflection_m": res.deflection})
st.download_button("Download results as CSV", df.to_csv(index=False),
                   file_name="beam_results.csv", mime="text/csv")
