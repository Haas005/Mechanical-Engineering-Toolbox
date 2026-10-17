"""Mohr's circle and plane-stress calculator. Run with:  python -m streamlit run app.py"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc

import streamlit as st

from mohr_solver import analyze, failure_check, strains, transform

st.set_page_config(page_title="Mohr's Circle", layout="wide")
st.title("Mohr's circle and plane stress")
st.caption("Stresses in MPa. Normal stress is positive in tension. Shear stress τxy is positive "
           "when it points up (+y) on the right-hand face (+x). Angles are counter-clockwise "
           "from the x axis. Plane stress: the stress through the thickness is zero.")

INK, GRID, GREY = "#1B1F24", "#E3E6EA", "#8A8F98"
TENSION, COMPRESSION, SHEAR = "#2166AC", "#B2182B", "#B26A00"
ROTATED = "#7B3294"

PRESETS = {
    "Textbook example": (-20.0, 90.0, 60.0),
    "Uniaxial tension": (100.0, 0.0, 0.0),
    "Pure shear": (0.0, 0.0, 50.0),
    "Equal biaxial tension": (80.0, 80.0, 0.0),
    "Thin-walled cylinder (axial and hoop)": (50.0, 100.0, 0.0),
    "Shaft in bending and torsion": (80.0, 0.0, 40.0),
}

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Setup")
    preset = st.selectbox("Start from", list(PRESETS))
    st.caption("Picking a preset fills in the stresses. Edit them freely afterwards.")
    st.subheader("Rotated element")
    theta = st.slider("Element angle θ (°)", -90, 90, 30, step=1,
                      help="Counter-clockwise from the x axis. Mohr's circle turns by 2θ.")
    st.subheader("Material")
    yield_strength = st.number_input("Yield strength (MPa)", min_value=1.0, value=250.0, step=10.0)
    E_gpa = st.number_input("Young's modulus E (GPa)", min_value=1.0, value=200.0, step=5.0)
    nu = st.number_input("Poisson's ratio ν", min_value=0.0, max_value=0.5, value=0.30, step=0.01)

# ----------------------------------------------------------------- inputs
sx0, sy0, t0 = PRESETS[preset]
st.markdown("**Stress state**")
ca, cb, cc = st.columns(3)
sx = ca.number_input("σx (MPa)", value=sx0, step=5.0, key=f"sx_{preset}")
sy = cb.number_input("σy (MPa)", value=sy0, step=5.0, key=f"sy_{preset}")
txy = cc.number_input("τxy (MPa)", value=t0, step=5.0, key=f"txy_{preset}")

try:
    res = analyze(sx, sy, txy)
    fail = failure_check(res, yield_strength)
    eps = strains(sx, sy, txy, E_gpa, nu)
except ValueError as err:
    st.error(str(err))
    st.stop()

sxp, syp, txyp = transform(sx, sy, txy, theta)
sx_p, sy_p, tp = float(sxp), float(syp), float(txyp)

# ----------------------------------------------------------------- results
st.subheader("Results")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Principal stress σ1", f"{res.s1:,.2f} MPa")
m2.metric("Principal stress σ2", f"{res.s2:,.2f} MPa")
m3.metric("Max in-plane shear τmax", f"{res.radius:,.2f} MPa", f"at θs = {res.theta_s:.2f}°",
          delta_color="off")
m4.metric("Principal angle θp", f"{res.theta_p:.2f}°", "x axis to σ1", delta_color="off")
m5.metric("Absolute max shear", f"{res.tau_abs_max:,.2f} MPa",
          "includes the zero out-of-plane stress" if res.tau_abs_max > res.radius + 1e-9
          else "same as in-plane", delta_color="off")


# ----------------------------------------------------------------- Mohr's circle plot
def label_points(ax, points, centre):
    """Label points outward from the circle centre. Points at the same spot share a label."""
    merged = {}
    for x, y, name, colour, marker in points:
        key = (round(x, 6), round(y, 6))
        if key in merged:
            merged[key][2] += " = " + name
        else:
            merged[key] = [x, y, name, colour, marker]
    for x, y, name, colour, marker in merged.values():
        ax.plot(x, y, marker=marker, ms=9, color=colour, mec="white", mew=1.2, zorder=6, ls="none")
        dx, dy = x - centre[0], y - centre[1]
        norm = math.hypot(dx, dy)
        ux, uy = (dx / norm, dy / norm) if norm > 1e-9 else (0.7, 0.7)
        lift = 13 if abs(y) < 1e-9 else 0   # labels on the sigma axis sit just above it
        ax.annotate(name, (x, y), xytext=(ux * 16, uy * 14 + lift), textcoords="offset points",
                    ha="center", va="center", fontsize=9.5, color=colour, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def draw_circle():
    C, R = res.center, res.radius
    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor="white")
    ax.set_facecolor("white")
    # Display y is -tau, so a positive shear plots downward (the textbook convention) and
    # turning the element counter-clockwise turns the circle counter-clockwise too.
    t = np.linspace(0, 2 * np.pi, 500)
    ax.plot(C + R * np.cos(t), R * np.sin(t), color=INK, lw=1.8, zorder=2)
    ax.axhline(0, color=GREY, lw=1.0, zorder=1)
    ax.axvline(0, color=GREY, lw=1.0, zorder=1)

    ax.plot([sx, sy], [-txy, txy], color=INK, lw=1.2, ls=(0, (5, 4)), zorder=3)
    ax.plot([sx_p, sy_p], [-tp, tp], color=ROTATED, lw=1.6, ls=(0, (2, 3)), zorder=3)

    pts = [(sx, -txy, "X", INK, "o"), (sy, txy, "Y", INK, "o"),
           (res.s1, 0.0, f"σ1 = {res.s1:.1f}", TENSION, "s"),
           (res.s2, 0.0, f"σ2 = {res.s2:.1f}", TENSION, "s"),
           (C, -R, f"τmax = {R:.1f}", SHEAR, "D"), (C, R, "", SHEAR, "D"),
           (sx_p, -tp, "X′", ROTATED, "o"), (sy_p, tp, "Y′", ROTATED, "o")]
    # the second max-shear point (tau = -R) needs no label of its own
    named = [p for p in pts if p[2]]
    ax.plot(C, R, marker="D", ms=9, color=SHEAR, mec="white", mew=1.2, zorder=6, ls="none")
    label_points(ax, named, (C, 0.0))
    ax.plot(C, 0, marker="+", ms=11, color=INK, zorder=6)
    ax.annotate(f"C = {C:.1f}", (C, 0), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=9, color=INK, zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    if R > 1e-9 and abs(theta) > 0:
        a0 = math.degrees(math.atan2(-txy, sx - C))
        a_end = a0 + 2 * theta
        lo, hi = sorted((a0, a_end))
        ax.add_patch(Arc((C, 0), 0.62 * R, 0.62 * R, theta1=lo, theta2=hi, color=ROTATED,
                         lw=2.0, zorder=4))
        mid = math.radians((a0 + a_end) / 2)
        ax.text(C + 0.47 * R * math.cos(mid), 0.47 * R * math.sin(mid), f"2θ = {2 * theta}°",
                color=ROTATED, fontsize=9.5, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    pad = 0.5 * R if R > 0 else max(abs(C), 1.0) * 0.5
    ax.set_xlim(min(C - R, 0.0) - pad, max(C + R, 0.0) + pad)
    ax.set_ylim(-R - pad, R + pad)
    ax.set_aspect("equal")
    ax.yaxis.set_major_formatter(lambda v, _: f"{-v:g}" if abs(v) > 1e-9 else "0")
    ax.set_xlabel("Normal stress σ (MPa)")
    ax.set_ylabel("Shear stress τ (MPa)   positive downward")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    handles = [plt.Line2D([], [], color=INK, lw=1.8, label="Mohr's circle"),
               plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (5, 4)), label="X–Y: original element"),
               plt.Line2D([], [], color=ROTATED, lw=1.6, ls=(0, (2, 3)), label=f"X′–Y′: element at θ = {theta}°"),
               plt.Line2D([], [], color=TENSION, marker="s", ls="none", label="Principal stresses"),
               plt.Line2D([], [], color=SHEAR, marker="D", ls="none", label="Max in-plane shear")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- stress element drawing
def draw_element(ax, angle, s_x, s_y, tau, title, scale):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rot = lambda v: (c * v[0] - s * v[1], s * v[0] + c * v[1])   # local (x', y') -> global
    corners = np.array([rot(p) for p in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
    ax.fill(corners[:, 0], corners[:, 1], fc="#F2F4F7", ec=INK, lw=1.6, zorder=2)
    tol = 1e-6 * max(scale, 1e-9)

    def arrow(p0, p1, colour, lw=2.4):
        ax.annotate("", xy=p1, xytext=p0, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=14))

    for normal, val in (((1, 0), s_x), ((-1, 0), s_x), ((0, 1), s_y), ((0, -1), s_y)):
        if abs(val) <= tol:
            continue
        face = rot(normal)
        tail, tip = (face, tuple(f * 1.75 for f in face))
        colour = TENSION
        if val < 0:
            tail, tip, colour = tip, tail, COMPRESSION
        arrow(tail, tip, colour)
    if abs(tau) > tol:
        sign = 1.0 if tau > 0 else -1.0
        # +x' face: toward +y'; +y' face: toward +x'; opposite faces point the other way
        for face_n, direction in (((1, 0), (0, 1)), ((-1, 0), (0, -1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0))):
            mid = rot(face_n)
            d = rot((direction[0] * sign, direction[1] * sign))
            nrm = rot(face_n)
            p0 = (mid[0] - 0.42 * d[0] + 0.16 * nrm[0], mid[1] - 0.42 * d[1] + 0.16 * nrm[1])
            p1 = (mid[0] + 0.42 * d[0] + 0.16 * nrm[0], mid[1] + 0.42 * d[1] + 0.16 * nrm[1])
            arrow(p0, p1, SHEAR, 2.0)
    # the element's own x' and y' directions
    for label, vec in (("x′", (0.62, 0)), ("y′", (0, 0.62))):
        end = rot(vec)
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
        ax.text(*(1.28 * np.array(end)), label, color=GREY,
                fontsize=9, ha="center", va="center")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=INK)
    f1 = lambda v: f"{round(v, 1) + 0.0:.1f}"      # avoids printing -0.0
    ax.text(0, -2.1, f"σx′ = {f1(s_x)}   σy′ = {f1(s_y)}\nτx′y′ = {f1(tau)} MPa", ha="center",
            va="top", fontsize=9.5, color=INK)


def draw_elements():
    scale = max(abs(sx), abs(sy), abs(txy), res.radius, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.9), facecolor="white")
    sets = [(0.0, sx, sy, txy, "Original (θ = 0°)")]
    sets.append((float(theta), sx_p, sy_p, tp, f"Rotated (θ = {theta}°)"))
    a, b, cc_ = transform(sx, sy, txy, res.theta_p)
    sets.append((res.theta_p, float(a), float(b), float(cc_), f"Principal (θp = {res.theta_p:.1f}°)"))
    a, b, cc_ = transform(sx, sy, txy, res.theta_s)
    sets.append((res.theta_s, float(a), float(b), float(cc_), f"Max shear (θs = {res.theta_s:.1f}°)"))
    for ax, (ang, s_x, s_y, tau, title) in zip(axes, sets):
        draw_element(ax, ang, s_x, s_y, tau, title, scale)
    handles = [plt.Line2D([], [], color=TENSION, lw=3, label="Tension (pulls away)"),
               plt.Line2D([], [], color=COMPRESSION, lw=3, label="Compression (pushes in)"),
               plt.Line2D([], [], color=SHEAR, lw=3, label="Shear")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.2, wspace=0.05)
    return fig


# ----------------------------------------------------------------- failure envelope plot
def draw_envelope():
    sy_ = yield_strength
    phi = np.linspace(0, 2 * np.pi, 600)
    rho = sy_ / np.sqrt(np.cos(phi) ** 2 - np.cos(phi) * np.sin(phi) + np.sin(phi) ** 2)
    hexa = np.array([(sy_, 0), (sy_, sy_), (0, sy_), (-sy_, 0), (-sy_, -sy_), (0, -sy_), (sy_, 0)])
    fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor="white")
    ax.plot(rho * np.cos(phi), rho * np.sin(phi), color=TENSION, lw=2.0, label="Von Mises limit")
    ax.plot(hexa[:, 0], hexa[:, 1], color=SHEAR, lw=2.0, ls=(0, (6, 3)), label="Tresca limit")
    ax.axhline(0, color=GREY, lw=1.0)
    ax.axvline(0, color=GREY, lw=1.0)
    ax.plot(res.s1, res.s2, "o", ms=10, color=INK, mec="white", mew=1.2, zorder=6,
            label="Your stress state")
    if math.isfinite(fail.sf_von_mises):
        ax.plot([0, res.s1 * fail.sf_von_mises], [0, res.s2 * fail.sf_von_mises], color=INK,
                lw=1.1, ls=":", zorder=3, label="Load scaled until yield (von Mises)")
    lim = max(1.4 * sy_, 1.25 * max(abs(res.s1), abs(res.s2)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("σ1 (MPa)")
    ax.set_ylabel("σ2 (MPa)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


tab_circle, tab_elem, tab_fail, tab_strain, tab_how = st.tabs(
    ["Mohr's circle", "Stress elements", "Failure check", "Strains", "How it works"])

with tab_circle:
    fig = draw_circle()
    st.pyplot(fig)
    plt.close(fig)
    pts = pd.DataFrame([
        ("X (original x face)", sx, txy),
        ("Y (original y face)", sy, -txy),
        (f"X′ (rotated by {theta}°)", sx_p, tp),
        ("Principal plane 1", res.s1, 0.0),
        ("Principal plane 2", res.s2, 0.0),
        ("Max shear plane", res.center, res.radius),
    ], columns=["Point", "σ (MPa)", "τ (MPa)"])
    st.dataframe(pts.style.format({"σ (MPa)": "{:+.2f}", "τ (MPa)": "{:+.2f}"}),
                 hide_index=True, width="stretch")
    angles = np.arange(-90, 91, 15)
    a, b, c_ = transform(sx, sy, txy, angles)
    sweep = pd.DataFrame({"θ (°)": angles, "σx′ (MPa)": a, "σy′ (MPa)": b, "τx′y′ (MPa)": c_})
    with st.expander("Stress on every 15° rotation"):
        st.dataframe(sweep.style.format({"σx′ (MPa)": "{:+.2f}", "σy′ (MPa)": "{:+.2f}",
                                         "τx′y′ (MPa)": "{:+.2f}"}), hide_index=True,
                     width="stretch")
        st.download_button("Download as CSV", sweep.to_csv(index=False),
                           file_name="stress_transformation.csv", mime="text/csv")

with tab_elem:
    st.write("The same state of stress seen on four differently turned elements. On the "
             "principal element the shear is zero, and on the max-shear element the normal "
             "stresses are equal (both are the average stress).")
    fig = draw_elements()
    st.pyplot(fig)
    plt.close(fig)

with tab_fail:
    left, right = st.columns([3, 2])
    with left:
        fig = draw_envelope()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fmt = lambda v: "∞" if math.isinf(v) else f"{v:.2f}"
        st.metric("Von Mises equivalent stress", f"{fail.von_mises:,.2f} MPa")
        st.metric("Von Mises safety factor", fmt(fail.sf_von_mises))
        st.metric("Tresca equivalent stress", f"{fail.tresca:,.2f} MPa")
        st.metric("Tresca safety factor", fmt(fail.sf_tresca))
        if min(fail.sf_von_mises, fail.sf_tresca) < 1.0:
            st.error("Predicted to yield: the stress state is outside the limit of at least one criterion.")
        elif min(fail.sf_von_mises, fail.sf_tresca) < 1.5:
            st.warning("Elastic, but with a thin margin (safety factor below 1.5).")
        else:
            st.success("Predicted to stay elastic under both criteria.")
        st.caption("The safety factor is how much the whole stress state could be scaled up "
                   "before reaching the yield strength. Tresca is never less conservative than von Mises.")

with tab_strain:
    micro = lambda v: f"{v * 1e6:,.1f} µε"
    s1c, s2c, s3c = st.columns(3)
    s1c.metric("εx", micro(eps.ex))
    s2c.metric("εy", micro(eps.ey))
    s3c.metric("γxy", micro(eps.gxy))
    s4c, s5c, s6c = st.columns(3)
    s4c.metric("εz (thickness)", micro(eps.ez), "Poisson effect", delta_color="off")
    s5c.metric("Principal ε1", micro(eps.e1))
    s6c.metric("Principal ε2", micro(eps.e2))
    st.caption(f"Hooke's law for plane stress with E = {E_gpa:g} GPa and ν = {nu:g}. µε means "
               f"microstrain (10⁻⁶). The principal strain directions coincide with the principal "
               f"stress directions (θp = {res.theta_p:.2f}°) because the material is isotropic. "
               f"Maximum in-plane shear strain: {eps.gamma_max * 1e6:,.1f} µε.")

with tab_how:
    st.markdown("**Stress on a plane rotated by θ**")
    st.latex(r"\sigma_{x'}=\frac{\sigma_x+\sigma_y}{2}+\frac{\sigma_x-\sigma_y}{2}\cos2\theta+\tau_{xy}\sin2\theta")
    st.latex(r"\tau_{x'y'}=-\frac{\sigma_x-\sigma_y}{2}\sin2\theta+\tau_{xy}\cos2\theta")
    st.markdown("**Mohr's circle.** These equations trace a circle as θ changes, with")
    st.latex(r"C=\frac{\sigma_x+\sigma_y}{2},\qquad R=\sqrt{\left(\frac{\sigma_x-\sigma_y}{2}\right)^2+\tau_{xy}^2}")
    st.markdown("The principal stresses are the two ends of the horizontal diameter, "
                "and the maximum in-plane shear is the radius:")
    st.latex(r"\sigma_{1,2}=C\pm R,\qquad \tan2\theta_p=\frac{2\tau_{xy}}{\sigma_x-\sigma_y},\qquad \theta_s=\theta_p-45^\circ")
    st.markdown("Turning the element by θ turns the point on the circle by **2θ** in the same "
                "direction. That is why the angles on the circle are double the real angles.")
    st.markdown("**Absolute maximum shear.** In plane stress the third principal stress is zero. "
                "The largest shear is half the difference between the largest and smallest of "
                "(σ1, σ2, 0). When σ1 and σ2 have the same sign this is bigger than the in-plane "
                "radius R.")
    st.markdown("**Failure criteria** (yield strength σy)")
    st.latex(r"\sigma_{vM}=\sqrt{\sigma_1^2-\sigma_1\sigma_2+\sigma_2^2},\qquad \sigma_{Tresca}=\max(\lvert\sigma_1-\sigma_2\rvert,\lvert\sigma_1\rvert,\lvert\sigma_2\rvert)")
    st.latex(r"\mathrm{SF}=\sigma_y/\sigma_{eq}")
