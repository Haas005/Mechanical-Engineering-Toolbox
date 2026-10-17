"""Double pendulum explorer. Run with:  python -m streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pendulum_solver import Params, flip_time_grid, simulate

st.set_page_config(page_title="Double Pendulum", layout="wide")


# ------------------------------------------------------------------ theme and colours
def theme_is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = theme_is_dark()
# Two series colours, validated for colour-blind separation and contrast on each surface.
C1, C2 = ("#3B82D6", "#D9702A") if DARK else ("#2166AC", "#D95F02")
INK = "#E6E8EB" if DARK else "#1B1F24"
GRID = "rgba(255,255,255,0.12)" if DARK else "rgba(0,0,0,0.10)"
BTN_BG = "#2B313A" if DARK else "#F0F2F6"       # animation buttons
BTN_BORDER = "#4A5361" if DARK else "#C5CBD6"   # button outline and slider rail
# One-hue sequential ramp (low -> high), validated against the matching surface.
RAMP = (["#2D6CB0", "#4F8BC6", "#80B0DD", "#B6D3EE", "#E6F0FA"] if DARK
        else ["#8DB8E0", "#5E97CF", "#3A7BBA", "#1F5C9A", "#0B3A6B"])
COLORSCALE = [[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)]

# ------------------------------------------------------------------ presets
S2 = 2 ** 0.5
BASE = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81, w1=0.0, w2=0.0, T=20)
PRESETS = {
    "Chaotic swing": dict(BASE, th1=120.0, th2=-10.0),
    "Horizontal release": dict(BASE, th1=90.0, th2=90.0),
    "Small swings (regular motion)": dict(BASE, th1=10.0, th2=10.0),
    "Slow normal mode": dict(BASE, th1=5.0, th2=round(5.0 * S2, 3)),
    "Fast normal mode": dict(BASE, th1=5.0, th2=-round(5.0 * S2, 3)),
    "Almost upright": dict(BASE, th1=175.0, th2=175.0),
    "Heavy upper bob": dict(BASE, m1=5.0, m2=1.0, th1=100.0, th2=60.0),
}
DEFAULT = "Chaotic swing"


def apply_preset():
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value


if "m1" not in st.session_state:
    st.session_state["preset"] = DEFAULT
    for k, v in PRESETS[DEFAULT].items():
        st.session_state[k] = v

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Setup")
    st.selectbox("Starting point", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Picking a preset fills in the fields below. You can edit any of them.")

    st.subheader("Masses and rods")
    a, b = st.columns(2)
    a.number_input("Mass 1 (kg)", 0.01, 100.0, step=0.1, key="m1")
    b.number_input("Mass 2 (kg)", 0.01, 100.0, step=0.1, key="m2")
    a.number_input("Rod 1 (m)", 0.05, 10.0, step=0.1, key="l1")
    b.number_input("Rod 2 (m)", 0.05, 10.0, step=0.1, key="l2")
    st.number_input("Gravity (m/s²)", 0.1, 30.0, step=0.1, key="g",
                    help="Earth 9.81, Moon 1.62, Mars 3.71")

    st.subheader("Starting state")
    st.caption("Angles are measured from straight down; positive swings to the right.")
    a, b = st.columns(2)
    a.number_input("Angle 1 (°)", -360.0, 360.0, step=1.0, format="%.2f", key="th1")
    b.number_input("Angle 2 (°)", -360.0, 360.0, step=1.0, format="%.2f", key="th2")
    a.number_input("Speed 1 (°/s)", -720.0, 720.0, step=5.0, format="%.1f", key="w1")
    b.number_input("Speed 2 (°/s)", -720.0, 720.0, step=5.0, format="%.1f", key="w2")
    st.slider("Duration (s)", 2, 40, key="T")

    st.subheader("Animation")
    trail = st.slider("Trail length (frames)", 0, 240, 90,
                      help="30 frames = 1 second of motion")
    speed = st.select_slider("Playback speed", [0.25, 0.5, 1.0, 2.0], value=1.0,
                             format_func=lambda s: f"{s}×")
    show_trail1 = st.checkbox("Also draw the trail of bob 1", value=False)

# ------------------------------------------------------------------ simulation
S = st.session_state
params = (float(S.m1), float(S.m2), float(S.l1), float(S.l2), float(S.g))


@st.cache_data(show_spinner=False)
def run(p_tuple, th1_deg, th2_deg, w1_deg, w2_deg, t_end):
    return simulate(Params(*p_tuple), np.radians(th1_deg), np.radians(th2_deg),
                    np.radians(w1_deg), np.radians(w2_deg), float(t_end))


try:
    traj = run(params, float(S.th1), float(S.th2), float(S.w1), float(S.w2), S.T)
except (ValueError, RuntimeError) as err:
    st.error(str(err))
    st.stop()

p = traj.params
R = p.l1 + p.l2
KE, PE, E = traj.energy
x1, y1, x2, y2 = traj.xy
t = traj.t

st.title("Double pendulum")
st.caption("Two masses on rigid rods, frictionless, uniform gravity. The equations of "
           "motion come from Lagrange's method and are integrated with an 8th-order "
           "Runge–Kutta solver at tight tolerances.")

m1_, m2_, m3_ = st.columns(3)
m1_.metric("Total energy", f"{E[0]:.3f} J", help="Should stay constant: nothing removes energy.")
drift = traj.relative_energy_drift
m2_.metric("Energy drift (accuracy check)", f"{drift:.1e}",
           help="Largest change in total energy as a fraction of a natural energy scale. "
                "Values below about 1e-6 mean the numerical solution can be trusted.")
m3_.metric("Highest point of bob 2", f"{y2.max():+.2f} m", help="Relative to the pivot.")

tab_anim, tab_plots, tab_sens, tab_map, tab_how = st.tabs(
    ["Animation", "Motion plots", "Sensitivity (chaos)", "Chaos map", "How it works"])

# ------------------------------------------------------------------ animation
FPS = 30


def build_animation():
    n = int(round(t[-1] * FPS)) + 1
    idx = np.round(np.linspace(0, len(t) - 1, n)).astype(int)
    X1, Y1, X2, Y2 = x1[idx], y1[idx], x2[idx], y2[idx]
    tt = t[idx]
    ref = max(p.m1, p.m2)
    size1 = 12 + 14 * np.sqrt(p.m1 / ref)
    size2 = 12 + 14 * np.sqrt(p.m2 / ref)

    def xy(k):
        lo = max(0, k - int(trail))
        return [
            dict(x=[0, X1[k], X2[k]], y=[0, Y1[k], Y2[k]]),
            dict(x=X1[lo:k + 1] if show_trail1 else [], y=Y1[lo:k + 1] if show_trail1 else []),
            dict(x=X2[lo:k + 1], y=Y2[lo:k + 1]),
            dict(x=[X1[k]], y=[Y1[k]]),
            dict(x=[X2[k]], y=[Y2[k]]),
        ]

    first = xy(0)
    fig = go.Figure(
        data=[
            go.Scatter(**first[0], mode="lines", line=dict(color=INK, width=3),
                       hoverinfo="skip", showlegend=False),
            go.Scatter(**first[1], mode="lines", line=dict(color=C1, width=2),
                       opacity=0.55, hoverinfo="skip", showlegend=False, legendgroup="b1"),
            go.Scatter(**first[2], mode="lines", line=dict(color=C2, width=2),
                       opacity=0.7, hoverinfo="skip", showlegend=False, legendgroup="b2"),
            go.Scatter(**first[3], mode="markers", name=f"Bob 1 ({p.m1:g} kg)",
                       marker=dict(color=C1, size=size1, line=dict(color=INK, width=1.5)),
                       hoverinfo="skip", legendgroup="b1"),
            go.Scatter(**first[4], mode="markers", name=f"Bob 2 ({p.m2:g} kg)",
                       marker=dict(color=C2, size=size2, line=dict(color=INK, width=1.5)),
                       hoverinfo="skip", legendgroup="b2"),
            go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color=INK, size=9,
                       symbol="square"), hoverinfo="skip", showlegend=False),
        ],
        frames=[go.Frame(data=[go.Scatter(**d) for d in xy(k)], traces=[0, 1, 2, 3, 4],
                         name=str(k)) for k in range(n)],
    )
    dur = 1000.0 / FPS / speed
    anim_args = dict(frame=dict(duration=dur, redraw=False), transition=dict(duration=0),
                     mode="immediate")
    fig.update_layout(
        height=700, margin=dict(l=10, r=10, t=40, b=120),
        xaxis=dict(range=[-R * 1.1, R * 1.1], title="x (m)", gridcolor=GRID, zeroline=False,
                   constrain="domain"),
        yaxis=dict(range=[-R * 1.1, R * 1.1], title="y (m)", gridcolor=GRID,
                   zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain"),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.14, xanchor="left", yanchor="top",
            pad=dict(t=12, r=8), showactive=False, bgcolor=BTN_BG, bordercolor=BTN_BORDER,
            font=dict(color=INK, size=13),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(anim_args, fromcurrent=True)]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ])],
        sliders=[dict(
            active=0, x=0.17, len=0.83, y=-0.14, yanchor="top", pad=dict(t=8, b=0),
            bgcolor=BTN_BORDER, activebgcolor=C1, bordercolor="rgba(0,0,0,0)",
            currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=15, color=INK)),
            ticklen=0, tickcolor="rgba(0,0,0,0)", font=dict(color="rgba(0,0,0,0)", size=1),
            steps=[dict(method="animate", label=f"{tt[k]:.2f}",
                        args=[[str(k)], dict(anim_args, frame=dict(duration=0, redraw=False))])
                   for k in range(n)])],
    )
    return fig


with tab_anim:
    st.plotly_chart(build_animation(), theme="streamlit", width="stretch",
                    config=dict(displaylogo=False, displayModeBar=False))
    st.caption("Press Play. Drag the slider to scrub through time. "
               "Marker size grows with mass. The dark square is the pivot.")

# ------------------------------------------------------------------ plots
def style(fig, ytitle, xtitle="Time (s)", height=340, log=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False,
                     type="log" if log else "linear")
    return fig


def line(x, y, name, color, dash="solid", width=2):
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash))


def wrap_with_gaps(theta, omega):
    th = (theta + np.pi) % (2 * np.pi) - np.pi
    cuts = np.where(np.abs(np.diff(th)) > np.pi)[0] + 1
    return np.degrees(np.insert(th, cuts, np.nan)), np.degrees(np.insert(omega, cuts, np.nan))


PLOT_CFG = dict(displaylogo=False)
with tab_plots:
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Angles over time**")
        f = go.Figure([line(t, np.degrees(traj.theta1), "Angle 1", C1),
                       line(t, np.degrees(traj.theta2), "Angle 2", C2)])
        st.plotly_chart(style(f, "Angle from vertical (°)"), theme="streamlit",
                        width="stretch", config=PLOT_CFG)
        st.caption("Angles keep counting past ±180° when a rod loops over the top.")
    with c_right:
        st.markdown("**Angular velocities**")
        f = go.Figure([line(t, np.degrees(traj.omega1), "Rod 1", C1),
                       line(t, np.degrees(traj.omega2), "Rod 2", C2)])
        st.plotly_chart(style(f, "Angular velocity (°/s)"), theme="streamlit",
                        width="stretch", config=PLOT_CFG)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Phase portrait**")
        a1, w1_ = wrap_with_gaps(traj.theta1, traj.omega1)
        a2, w2_ = wrap_with_gaps(traj.theta2, traj.omega2)
        f = go.Figure([
            go.Scatter(x=a1, y=w1_, name="Rod 1", mode="lines", line=dict(color=C1, width=1.5)),
            go.Scatter(x=a2, y=w2_, name="Rod 2", mode="lines", line=dict(color=C2, width=1.5)),
        ])
        style(f, "Angular velocity (°/s)", "Angle, wrapped to ±180° (°)")
        f.update_layout(hovermode="closest")
        st.plotly_chart(f, theme="streamlit", width="stretch", config=PLOT_CFG)
        st.caption("Regular motion traces closed loops. Chaotic motion fills a region.")
    with c_right:
        st.markdown("**Energy**")
        f = go.Figure([line(t, KE, "Kinetic", C1), line(t, PE, "Potential", C2),
                       line(t, E, "Total", INK, dash="dash", width=2.5)])
        st.plotly_chart(style(f, "Energy (J)"), theme="streamlit", width="stretch",
                        config=PLOT_CFG)
        st.caption("Energy swaps between kinetic and potential, but the total stays flat.")

    df = pd.DataFrame({
        "t_s": t, "theta1_deg": np.degrees(traj.theta1), "theta2_deg": np.degrees(traj.theta2),
        "omega1_deg_s": np.degrees(traj.omega1), "omega2_deg_s": np.degrees(traj.omega2),
        "x1_m": x1, "y1_m": y1, "x2_m": x2, "y2_m": y2,
        "kinetic_J": KE, "potential_J": PE, "total_J": E})
    st.download_button("Download the motion as CSV", df.to_csv(index=False),
                       file_name="double_pendulum.csv", mime="text/csv")

# ------------------------------------------------------------------ sensitivity
with tab_sens:
    st.markdown(
        "Run the same pendulum twice. The second run starts with angle 1 nudged by a tiny "
        "amount. In regular motion the two runs stay together. In chaotic motion the tiny "
        "difference grows exponentially until the paths have nothing to do with each other.")
    nudge = st.select_slider("Nudge angle 1 by", [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                             value=1e-6, format_func=lambda v: f"{v:.0e} rad "
                             f"({np.degrees(v):.1e}°)")
    try:
        other = run(params, float(S.th1) + float(np.degrees(nudge)), float(S.th2),
                    float(S.w1), float(S.w2), S.T)
    except (ValueError, RuntimeError) as err:
        st.error(str(err))
        st.stop()
    ox1, oy1, ox2, oy2 = other.xy
    sep = np.hypot(x2 - ox2, y2 - oy2)
    limit = 0.1 * R
    crossed = np.where(sep > limit)[0]
    s1, s2 = st.columns(2)
    s1.metric("Starting difference (bob 2 position)", f"{sep[1] if sep[0] == 0 else sep[0]:.1e} m")
    s2.metric(f"Time until bob 2 paths differ by {limit:.2f} m (10% of reach)",
              f"{t[crossed[0]]:.1f} s" if len(crossed) else f"not within {S.T} s")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Bob 2 paths**")
        f = go.Figure([
            go.Scatter(x=x2, y=y2, name="Original", mode="lines", line=dict(color=C1, width=2)),
            go.Scatter(x=ox2, y=oy2, name="Nudged", mode="lines",
                       line=dict(color=C2, width=1.5, dash="dot")),
        ])
        f.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
                        xaxis=dict(title="x (m)", gridcolor=GRID, zeroline=False,
                                   range=[-R * 1.1, R * 1.1], constrain="domain"),
                        yaxis=dict(title="y (m)", gridcolor=GRID, zeroline=False,
                                   range=[-R * 1.1, R * 1.1], scaleanchor="x", scaleratio=1,
                                   constrain="domain"))
        st.plotly_chart(f, theme="streamlit", width="stretch", config=PLOT_CFG)
    with c_right:
        st.markdown("**How far apart the two bob 2 positions are**")
        f = go.Figure([
            line(t, np.maximum(sep, 1e-16), "Separation", INK, width=2.5),
            go.Scatter(x=[t[0], t[-1]], y=[limit, limit], mode="lines", name="10% of reach",
                       line=dict(color=C2, width=1.5, dash="dash")),
        ])
        st.plotly_chart(style(f, "Separation (m, log scale)", height=420, log=True),
                        theme="streamlit", width="stretch", config=PLOT_CFG)
        st.caption("A straight rising line on a log scale means exponential growth. "
                   "Try the 'Small swings' preset to see it stay flat instead.")

# ------------------------------------------------------------------ chaos map
@st.cache_data(show_spinner=False)
def cached_map(p_tuple, n, t_end):
    return flip_time_grid(Params(*p_tuple), n=n, t_end=float(t_end))


with tab_map:
    st.markdown(
        "Every pixel is a different starting point: both rods released from rest at the "
        "angles on the axes, using the masses, rods and gravity from the sidebar. The "
        "color shows how long it takes before either rod first swings over the top "
        "(past straight up). Smooth regions are predictable, and the speckled regions are "
        "chaos, where neighbouring starts end up flipping at very different times.")
    g1, g2 = st.columns(2)
    n_grid = g1.select_slider("Grid size", [40, 60, 80, 100], value=80)
    look = g2.slider("Look ahead (s)", 3, 20, 10)
    if st.button("Compute the map", type="primary"):
        st.session_state["map_on"] = True
    if st.session_state.get("map_on"):
        with st.spinner("Simulating thousands of pendulums at once..."):
            angles, grid = cached_map(params, n_grid, look)
        deg = np.degrees(angles)
        never = grid >= look - 1e-9
        hover = np.where(never, f"no flip within {look} s",
                         np.char.add(np.char.mod("%.2f", grid), " s"))
        f = go.Figure([
            go.Heatmap(x=deg, y=deg, z=grid, colorscale=COLORSCALE, zmin=0, zmax=look,
                       customdata=hover, hovertemplate="angle 1 %{x:.0f}°<br>angle 2 "
                       "%{y:.0f}°<br>%{customdata}<extra></extra>",
                       colorbar=dict(title="Time to first flip (s)")),
            go.Scatter(x=[(float(S.th1) + 180) % 360 - 180], y=[(float(S.th2) + 180) % 360 - 180],
                       mode="markers", name="Your starting point",
                       marker=dict(symbol="x", size=13, color=INK if not DARK else "#FFFFFF",
                                   line=dict(color=C2, width=2))),
        ])
        f.update_layout(height=640, margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
                        xaxis=dict(title="Starting angle 1 (°)", range=[-180, 180],
                                   constrain="domain"),
                        yaxis=dict(title="Starting angle 2 (°)", range=[-180, 180],
                                   scaleanchor="x", scaleratio=1, constrain="domain"))
        st.plotly_chart(f, theme="streamlit", width="stretch", config=PLOT_CFG)
        st.caption(f"{100 * (~never).mean():.0f}% of these starting points flip within "
                   f"{look} s. Cells at the top of the color bar never flip in that time "
                   "(hover to check). This map uses a fast fixed-step solver, so treat it "
                   "as a picture of the chaos, not a precise measurement.")
    else:
        st.info("Press the button to compute the map. It takes a few seconds.")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("**Setup.** Two point masses m₁ and m₂ hang from rigid massless rods of "
                "length l₁ and l₂. The angles θ₁ and θ₂ are measured from straight down.")
    st.markdown("**Positions** of the two masses (pivot at the origin):")
    st.latex(r"x_1 = l_1\sin\theta_1,\quad y_1 = -l_1\cos\theta_1")
    st.latex(r"x_2 = x_1 + l_2\sin\theta_2,\quad y_2 = y_1 - l_2\cos\theta_2")
    st.markdown("**Lagrangian** (kinetic energy minus potential energy):")
    st.latex(r"L = \tfrac12 (m_1+m_2) l_1^2\dot\theta_1^2 + \tfrac12 m_2 l_2^2\dot\theta_2^2"
             r" + m_2 l_1 l_2\dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2)"
             r" + (m_1+m_2) g l_1\cos\theta_1 + m_2 g l_2\cos\theta_2")
    st.markdown("Applying Lagrange's equations, d/dt(∂L/∂θ̇) − ∂L/∂θ = 0, for each angle "
                "gives two coupled second-order equations, which are solved for the "
                "angular accelerations and integrated numerically. With "
                "δ = θ₁ − θ₂ and D = 2m₁ + m₂ − m₂cos 2δ:")
    st.latex(r"\ddot\theta_1 = \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g\sin(\theta_1-2\theta_2)"
             r" - 2\sin\delta\, m_2(\dot\theta_2^2 l_2 + \dot\theta_1^2 l_1\cos\delta)}{l_1 D}")
    st.latex(r"\ddot\theta_2 = \frac{2\sin\delta\,\big(\dot\theta_1^2 l_1(m_1+m_2) + "
             r"g(m_1+m_2)\cos\theta_1 + \dot\theta_2^2 l_2 m_2\cos\delta\big)}{l_2 D}")
    st.markdown(
        "**Why it is chaotic.** The equations are nonlinear and the two angles are coupled. "
        "For small angles they simplify to two normal modes with frequencies "
        "ω² = (g/l)(2 ∓ √2) (equal masses and rods), and the motion is regular. At large "
        "angles the nonlinear terms dominate, and nearby starting points separate "
        "exponentially fast. Total energy is still conserved exactly, so the energy "
        "drift metric is a good check that the numerical solution is accurate.")
