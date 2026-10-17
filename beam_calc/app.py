"""Beam calculator UI. Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from beam_solver import (SUPPORT_PRESETS, LinearLoad, PointLoad, PointMoment,
                         UniformLoad, solve_beam)

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
st.write("Edit the tables below. Add rows with the + at the bottom of each table. "
         "Loads are positive downward, so enter a negative value for an upward load.")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Point loads**")
    pl_df = st.data_editor(
        pd.DataFrame({"Load (N)": [10000.0], "Position (m)": [2.5]}),
        num_rows="dynamic", key="pl", width="stretch")
    st.markdown("**Distributed loads (uniform)**")
    ul_df = st.data_editor(
        pd.DataFrame({"Intensity (N/m)": [0.0], "Start (m)": [0.0], "End (m)": [5.0]}),
        num_rows="dynamic", key="ul", width="stretch")
with c2:
    st.markdown("**Triangular / trapezoidal loads**")
    ll_df = st.data_editor(
        pd.DataFrame({"Start intensity (N/m)": pd.Series(dtype=float),
                      "End intensity (N/m)": pd.Series(dtype=float),
                      "Start (m)": pd.Series(dtype=float),
                      "End (m)": pd.Series(dtype=float)}),
        num_rows="dynamic", key="ll", width="stretch")
    st.caption("The intensity varies in a straight line between the start and end. "
               "For a triangle set one intensity to 0; for a trapezoid use two different "
               "non-zero values.")
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
for _, r in ll_df.dropna().iterrows():
    if r.iloc[0] != 0 or r.iloc[1] != 0:
        loads.append(LinearLoad(float(r.iloc[0]), float(r.iloc[1]),
                                float(r.iloc[2]), float(r.iloc[3])))
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


# ---------------------------------------------------------------- load diagram
def draw_load_diagram():
    fig, ax = plt.subplots(figsize=(9, 2.3))
    h = 0.16 * L                                   # height of the biggest load symbol
    dist = [ld for ld in loads if isinstance(ld, (UniformLoad, LinearLoad))]
    peak = max([max(abs(getattr(ld, "intensity", 0.0)), abs(getattr(ld, "start_intensity", 0.0)),
                    abs(getattr(ld, "end_intensity", 0.0))) for ld in dist], default=1.0) or 1.0
    ax.plot([0, L], [0, 0], color="black", lw=5, solid_capstyle="butt", zorder=3)

    wall = {"cantilever": [0], "propped": [0], "fixed_fixed": [0, L]}.get(support, [])
    for xw in wall:
        side = -1 if xw == 0 else 1
        ax.add_patch(plt.Rectangle((xw if side > 0 else xw - 0.025 * L, -0.12 * L),
                                   0.025 * L, 0.24 * L, fc="#888888", ec="black", hatch="///",
                                   zorder=2))
    marks = []
    if support == "simple":
        marks = [(0, "^"), (L, "o")]
    elif support == "propped":
        marks = [(L, "o")]
    for xm, mk in marks:
        ax.plot(xm, -0.06 * L, marker=mk, ms=13, mfc="white", mec="black", mew=1.5, zorder=4)

    ext = [0.0, -0.2 * L, 0.2 * L]                 # vertical extent actually drawn

    def arrow(xa, y0, y1, color):
        ax.annotate("", xy=(xa, y1), xytext=(xa, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=5)

    for ld in loads:
        if isinstance(ld, PointLoad):
            s = 1 if ld.magnitude > 0 else -1
            arrow(ld.position, s * h * 1.15, s * 0.02 * L, "#C0392B")
            ext.append(s * h * 1.55)
            ax.text(ld.position, s * h * 1.3, f"{abs(ld.magnitude):,.0f} N", ha="center",
                    va="bottom" if s > 0 else "top", fontsize=9, color="#C0392B")
        elif isinstance(ld, (UniformLoad, LinearLoad)):
            wa = ld.intensity if isinstance(ld, UniformLoad) else ld.start_intensity
            wb = ld.intensity if isinstance(ld, UniformLoad) else ld.end_intensity
            ha, hb = wa / peak * h, wb / peak * h
            ax.fill([ld.start, ld.start, ld.end, ld.end], [0, ha, hb, 0],
                    color="#E67E22", alpha=0.25, zorder=1)
            ax.plot([ld.start, ld.start, ld.end, ld.end], [0, ha, hb, 0], color="#E67E22", lw=1.5)
            for xa in np.linspace(ld.start, ld.end, 8):
                hh = ha + (hb - ha) * (xa - ld.start) / (ld.end - ld.start)
                if abs(hh) > 0.02 * L:
                    arrow(xa, hh, np.sign(hh) * 0.02 * L, "#E67E22")
            label = (f"{abs(wa):,.0f} N/m" if wa == wb else f"{wa:,.0f} to {wb:,.0f} N/m")
            top = max(ha, hb) if max(ha, hb) > 0 else min(ha, hb)
            ext.append(top + np.sign(top) * 0.12 * L)
            ax.text((ld.start + ld.end) / 2, top + np.sign(top) * 0.03 * L, label, ha="center",
                    va="bottom" if top > 0 else "top", fontsize=9, color="#A04000")
        elif isinstance(ld, PointMoment):
            ccw = ld.magnitude > 0
            th = np.linspace(np.radians(30), np.radians(300), 60)
            r = 0.07 * L
            ext.extend([-r * 1.3, r * 1.9])
            ax.plot(ld.position + r * np.cos(th), r * np.sin(th), color="#2471A3", lw=1.8)
            end = th[-1] if ccw else th[0]
            prev = th[-4] if ccw else th[3]
            ax.annotate("", xy=(ld.position + r * np.cos(end), r * np.sin(end)),
                        xytext=(ld.position + r * np.cos(prev), r * np.sin(prev)),
                        arrowprops=dict(arrowstyle="-|>", color="#2471A3", lw=1.8))
            ax.text(ld.position, r * 1.15 + 0.01 * L, f"{abs(ld.magnitude):,.0f} N·m "
                    f"{'CCW' if ccw else 'CW'}", ha="center", va="bottom", fontsize=9,
                    color="#2471A3")
    ax.set_xlim(-0.08 * L, 1.08 * L)
    ax.set_ylim(min(ext), max(ext))
    ax.set_aspect("auto")
    ax.axis("off")
    fig.tight_layout()
    return fig


dfig = draw_load_diagram()
st.pyplot(dfig)
plt.close(dfig)

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
