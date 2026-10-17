"""2D truss analysis UI. Run with:  python -m streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from truss_solver import (SUPPORT_TYPES, Member, NodalLoad, Node, Support,
                          check_buckling, solve_truss)

st.set_page_config(page_title="Truss Solver", layout="wide")
st.title("2D truss solver")
st.caption("Direct stiffness method. Coordinates in m, E in GPa, A in cm², I in cm⁴, loads in kN. "
           "Fx is positive to the right and Fy is positive up. "
           "Member force is positive in tension, negative in compression.")

# Colours: tension / compression is a two-sided (diverging) quantity, so it gets two
# hues plus a neutral for zero-force members. The sign is also printed on every label,
# so colour is never the only cue.
TENSION, COMPRESSION, ZERO, INK, GRID = "#2166AC", "#B2182B", "#8A8F98", "#1B1F24", "#E3E6EA"

PIN, ROLLER_X, ROLLER_Y = list(SUPPORT_TYPES)

# ----------------------------------------------------------------- presets
PRESETS = {
    "Simple triangle": dict(
        nodes=[(0, 0), (4, 0), (2, 3)],
        members=[(1, 2), (1, 3), (2, 3)],
        supports=[(1, PIN), (2, ROLLER_X)],
        loads=[(3, 0.0, -10.0)]),
    "Pratt bridge truss": dict(
        nodes=[(0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (2, 2), (4, 2), (6, 2)],
        members=[(1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (1, 6), (8, 5),
                 (2, 6), (3, 7), (4, 8), (6, 3), (8, 3)],
        supports=[(1, PIN), (5, ROLLER_X)],
        loads=[(2, 0.0, -20.0), (3, 0.0, -20.0), (4, 0.0, -20.0)]),
    "Cantilever truss": dict(
        nodes=[(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)],
        members=[(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (3, 4), (5, 6), (2, 3), (4, 5)],
        supports=[(1, PIN), (2, ROLLER_Y)],
        loads=[(5, 0.0, -15.0)]),
    "Blank (start your own)": dict(
        nodes=[(0, 0), (1, 0)], members=[(1, 2)],
        supports=[(1, PIN), (2, ROLLER_X)], loads=[]),
}


def preset_frames(name):
    p = PRESETS[name]
    nodes = pd.DataFrame(p["nodes"], columns=["X (m)", "Y (m)"], dtype=float)
    members = pd.DataFrame([(a, b, 200.0, 10.0, 20.0) for a, b in p["members"]],
                           columns=["From node", "To node", "E (GPa)", "A (cm²)", "I (cm⁴)"])
    supports = pd.DataFrame(p["supports"], columns=["Node", "Type"])
    loads = pd.DataFrame(p["loads"], columns=["Node", "Fx (kN)", "Fy (kN)"])
    if loads.empty:
        loads = pd.DataFrame({"Node": pd.Series(dtype=float),
                              "Fx (kN)": pd.Series(dtype=float),
                              "Fy (kN)": pd.Series(dtype=float)})
    return nodes, members, supports, loads


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Setup")
    preset = st.selectbox("Start from", list(PRESETS))
    st.caption("Picking a preset replaces the tables. Edit them freely afterwards; "
               "nodes are numbered by row (first row = node 1).")
    st.subheader("Plot options")
    show_deformed = st.checkbox("Show deformed shape", value=True)
    auto_scale = st.checkbox("Auto deformation scale", value=True)
    manual_scale = st.number_input("Manual scale (×)", min_value=1.0, value=100.0,
                                   step=50.0, disabled=auto_scale)
    show_member_ids = st.checkbox("Show member numbers", value=False)
    st.subheader("Buckling check")
    st.caption("Compression members are checked as pin-ended columns using Euler's formula "
               "(slender) or the Johnson parabola (stocky). I is the smallest second moment "
               "of area of the cross-section, since a column buckles about its weak axis.")
    yield_strength = st.number_input("Yield strength (MPa)", min_value=1.0, value=250.0,
                                     step=10.0)
    k_factor = st.number_input("Effective length factor K", min_value=0.1, value=1.0, step=0.1,
                               help="1.0 for pin-ended truss members.")
    required_sf = st.number_input("Required safety factor", min_value=1.0, value=2.0, step=0.1)

n0, m0, s0, l0 = preset_frames(preset)

# ----------------------------------------------------------------- inputs
col_a, col_b = st.columns([2, 3])
with col_a:
    st.markdown("**Nodes**")
    nodes_df = st.data_editor(n0, num_rows="dynamic", hide_index=True,
                              key=f"nodes_{preset}", width="stretch")
    st.markdown("**Supports**")
    supports_df = st.data_editor(
        s0, num_rows="dynamic", hide_index=True, key=f"sup_{preset}", width="stretch",
        column_config={
            "Node": st.column_config.NumberColumn(step=1, format="%d"),
            "Type": st.column_config.SelectboxColumn(options=list(SUPPORT_TYPES),
                                                     required=True)})
with col_b:
    st.markdown("**Members**")
    members_df = st.data_editor(
        m0, num_rows="dynamic", hide_index=True, key=f"mem_{preset}", width="stretch",
        column_config={
            "From node": st.column_config.NumberColumn(step=1, format="%d"),
            "To node": st.column_config.NumberColumn(step=1, format="%d")})
    st.markdown("**Nodal loads**")
    loads_df = st.data_editor(
        l0, num_rows="dynamic", hide_index=True, key=f"load_{preset}", width="stretch",
        column_config={"Node": st.column_config.NumberColumn(step=1, format="%d")})


def as_node_index(value, what):
    if abs(value - round(value)) > 1e-9:
        raise ValueError(f"{what}: node numbers must be whole numbers (got {value}).")
    return int(round(value)) - 1


def parse_inputs():
    nd = nodes_df.dropna()
    nodes = [Node(float(r.iloc[0]), float(r.iloc[1])) for _, r in nd.iterrows()]
    members = []
    for k, (_, r) in enumerate(members_df.dropna().iterrows(), start=1):
        members.append(Member(as_node_index(r.iloc[0], f"Member {k}"),
                              as_node_index(r.iloc[1], f"Member {k}"),
                              float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4])))
    supports = []
    for _, r in supports_df.dropna().iterrows():
        fx, fy = SUPPORT_TYPES[r.iloc[1]]
        supports.append(Support(as_node_index(r.iloc[0], "Support"), fx, fy))
    loads = []
    for _, r in loads_df.dropna().iterrows():
        if r.iloc[1] != 0 or r.iloc[2] != 0:
            loads.append(NodalLoad(as_node_index(r.iloc[0], "Load"),
                                   float(r.iloc[1]), float(r.iloc[2])))
    return nodes, members, supports, loads


try:
    nodes, members, supports, loads = parse_inputs()
    res = solve_truss(nodes, members, supports, loads)
    bk = check_buckling(res, members, yield_strength, k_factor, required_sf)
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as err:
    st.error(f"The solver could not handle this configuration: {err}")
    st.stop()

if not loads:
    st.info("No loads yet, so every force is zero. Add a nodal load to see results.")

# ----------------------------------------------------------------- results
F = res.member_force
tol = 1e-6 * max(1.0, float(np.max(np.abs(F))))
kind = np.where(F > tol, "Tension", np.where(F < -tol, "Compression", "Zero force"))

st.subheader("Results")
ind = res.indeterminacy
if ind < 0:
    st.error(f"Mechanism: m + r − 2j = {ind}. There are too few members or reactions.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Determinacy (m + r − 2j)", f"{ind}",
          "statically determinate" if ind == 0 else
          f"{ind}° indeterminate" if ind > 0 else "unstable", delta_color="off")
if (F > tol).any():
    k = int(np.argmax(F))
    c2.metric("Max tension", f"{F[k]:,.2f} kN", f"member {k + 1}", delta_color="off")
else:
    c2.metric("Max tension", "none")
if (F < -tol).any():
    k = int(np.argmin(F))
    c3.metric("Max compression", f"{-F[k]:,.2f} kN", f"member {k + 1}", delta_color="off")
else:
    c3.metric("Max compression", "none")
mag = np.hypot(res.displacements[:, 0], res.displacements[:, 1])
c4.metric("Max displacement", f"{mag.max():.3f} mm", f"node {int(np.argmax(mag)) + 1}",
          delta_color="off")
sf_all = bk.safety_factor
if np.isfinite(sf_all).any():
    k = int(np.nanargmin(sf_all))
    c5.metric("Min buckling safety factor", f"{sf_all[k]:.2f}", f"member {k + 1}",
              delta_color="off")
else:
    c5.metric("Min buckling safety factor", "n/a", "no compression members", delta_color="off")
flag = np.isin(bk.status, ["Buckles", "Below required"])
if flag.any():
    items = ", ".join(f"{k + 1} (SF {sf_all[k]:.2f})" for k in np.where(flag)[0])
    msg = f"Buckling check: members {items} are below the required safety factor of {required_sf:g}."
    (st.error if (bk.status == "Buckles").any() else st.warning)(msg)


# ----------------------------------------------------------------- plot
def draw_truss():
    xs = np.array([n.x for n in nodes])
    ys = np.array([n.y for n in nodes])
    span = max(np.ptp(xs), np.ptp(ys), 1e-9)
    disp_m = res.displacements / 1e3
    dmax = float(np.max(np.hypot(disp_m[:, 0], disp_m[:, 1])))
    if auto_scale:
        scale = min(1e6, 0.06 * span / dmax) if dmax > 0 else 1.0
    else:
        scale = manual_scale

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")
    fmax = max(float(np.max(np.abs(F))), 1e-12)

    for k, m in enumerate(members):
        x0, y0, x1, y1 = nodes[m.i].x, nodes[m.i].y, nodes[m.j].x, nodes[m.j].y
        colour = TENSION if kind[k] == "Tension" else COMPRESSION if kind[k] == "Compression" else ZERO
        ax.plot([x0, x1], [y0, y1], color=colour, lw=2.0 + 4.0 * abs(F[k]) / fmax,
                solid_capstyle="round", zorder=2)
        if show_deformed:
            ax.plot([x0 + disp_m[m.i, 0] * scale, x1 + disp_m[m.j, 0] * scale],
                    [y0 + disp_m[m.i, 1] * scale, y1 + disp_m[m.j, 1] * scale],
                    color=INK, lw=1.2, ls=(0, (5, 4)), alpha=0.65, zorder=1)
        label = f"{F[k]:+.2f}" if kind[k] != "Zero force" else "0"
        if show_member_ids:
            label = f"{k + 1}: {label}"
        flagged = bool(flag[k])
        if flagged:
            label += f"\n⚠ SF {sf_all[k]:.2f}"
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=9,
                color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour,
                          lw=2.5 if flagged else 1))

    extra = []  # points (support symbols, load labels) that the view must include

    # supports: pin and y-restraining roller sit below the node; an x-restraining
    # roller (free to move in y) sits to its left, like a wall support
    for s in supports:
        nd = nodes[s.node]
        if s.fix_x and s.fix_y:
            marker, dx, dy = "^", 0.0, -0.045
        elif s.fix_y:
            marker, dx, dy = "o", 0.0, -0.045
        else:
            marker, dx, dy = "s", -0.045, 0.0
        ax.plot(nd.x + dx * span, nd.y + dy * span, marker=marker, ms=17, mfc="white",
                mec=INK, mew=1.6, zorder=3)
        extra.append((nd.x + dx * span * 1.6, nd.y + dy * span * 1.6))

    # loads: the arrow shows the force direction. Draw it ending at the node, unless
    # that would put it inside the truss; then start it at the node and point it away.
    lmax = max([np.hypot(l.fx, l.fy) for l in loads], default=0.0)
    cx, cy = xs.mean(), ys.mean()
    for l in loads:
        mag_l = np.hypot(l.fx, l.fy)
        if mag_l == 0:
            continue
        length = (0.10 + 0.08 * mag_l / lmax) * span
        ux, uy = l.fx / mag_l, l.fy / mag_l
        nd = nodes[l.node]
        tail_in = (nd.x - ux * length, nd.y - uy * length)   # arrow ends at the node
        tip_out = (nd.x + ux * length, nd.y + uy * length)   # arrow starts at the node
        dist = lambda p: np.hypot(p[0] - cx, p[1] - cy)
        if dist(tip_out) > dist(tail_in):
            start, end, label_at = (nd.x, nd.y), tip_out, (nd.x + ux * length * 1.15,
                                                          nd.y + uy * length * 1.15)
        else:
            start, end, label_at = tail_in, (nd.x, nd.y), (nd.x - ux * length * 1.15,
                                                          nd.y - uy * length * 1.15)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#B26A00", lw=2.4,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.text(*label_at, f"{mag_l:.3g} kN", color="#8A5200", fontsize=9, ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        extra.append((label_at[0], label_at[1] + np.sign(label_at[1] - nd.y) * 0.04 * span))

    ax.scatter(xs, ys, s=40, color=INK, zorder=6)
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        ax.text(x + 0.015 * span, y + 0.025 * span, str(i), fontsize=10, fontweight="bold",
                color=INK, zorder=7)

    all_x = np.concatenate([xs, [p[0] for p in extra]])
    all_y = np.concatenate([ys, [p[1] for p in extra]])
    pad = 0.08 * span
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)

    handles = [plt.Line2D([], [], color=TENSION, lw=4, label="Tension (+)"),
               plt.Line2D([], [], color=COMPRESSION, lw=4, label="Compression (−)"),
               plt.Line2D([], [], color=ZERO, lw=4, label="Zero force")]
    if show_deformed:
        handles.append(plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (5, 4)),
                                  label=f"Deformed (×{scale:,.0f})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(handles), frameon=False)
    fig.tight_layout()
    return fig


fig = draw_truss()
st.pyplot(fig)
plt.close(fig)

# ----------------------------------------------------------------- tables
t1, t2 = st.columns([3, 2])
with t1:
    st.markdown("**Member forces**")
    mem_table = pd.DataFrame({
        "Member": np.arange(1, len(members) + 1),
        "From": [m.i + 1 for m in members],
        "To": [m.j + 1 for m in members],
        "Length (m)": res.member_length,
        "Force (kN)": F,
        "Type": kind,
        "Stress (MPa)": res.member_stress,
        "Elongation (mm)": res.member_elongation,
        "Slenderness KL/r": bk.slenderness,
        "Buckling model": bk.regime,
        "Pcr (kN)": bk.critical_load,
        "Buckling SF": bk.safety_factor,
        "Buckling status": bk.status,
    })
    st.dataframe(mem_table.style.format({
        "Length (m)": "{:.3f}", "Force (kN)": "{:+.3f}", "Stress (MPa)": "{:+.2f}",
        "Elongation (mm)": "{:+.4f}", "Slenderness KL/r": "{:.0f}", "Pcr (kN)": "{:.2f}",
        "Buckling SF": "{:.2f}"}, na_rep="–"), hide_index=True, width="stretch")
with t2:
    st.markdown("**Support reactions**")
    rx = pd.DataFrame([(nd + 1, ax_.upper(), v) for (nd, ax_), v in sorted(res.reactions.items())],
                      columns=["Node", "Direction", "Reaction (kN)"])
    st.dataframe(rx.style.format({"Reaction (kN)": "{:+.3f}"}), hide_index=True,
                 width="stretch")
    st.markdown("**Node displacements**")
    dsp = pd.DataFrame({"Node": np.arange(1, len(nodes) + 1),
                        "ux (mm)": res.displacements[:, 0],
                        "uy (mm)": res.displacements[:, 1]})
    st.dataframe(dsp.style.format({"ux (mm)": "{:+.4f}", "uy (mm)": "{:+.4f}"}),
                 hide_index=True, width="stretch")

with st.expander("Equilibrium check"):
    st.write("Applied loads plus support reactions should sum to zero. "
             "Anything beyond rounding error means something is wrong with the model.")
    st.table(pd.DataFrame({
        "Check": ["Sum of Fx (kN)", "Sum of Fy (kN)", "Sum of moments about origin (kN·m)"],
        "Value": [f"{v:.2e}" for v in res.equilibrium_error.values()]}).set_index("Check"))

st.download_button("Download member results as CSV", mem_table.to_csv(index=False),
                   file_name="truss_members.csv", mime="text/csv")
