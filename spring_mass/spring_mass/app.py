"""Damped spring-mass system explorer. Run with:  python -m streamlit run app.py"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from vibration_solver import (Load, Params, damping_for_ratio, magnification, phase_lag,
                              properties, resonance_peak, simulate, suggest_duration,
                              suggest_samples, transmissibility)

st.set_page_config(page_title="Damped Spring-Mass System", layout="wide")


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
MUTED = "#9AA3AF" if DARK else "#6B7280"
GRID = "rgba(255,255,255,0.12)" if DARK else "rgba(0,0,0,0.10)"
BTN_BG = "#2B313A" if DARK else "#F0F2F6"
BTN_BORDER = "#4A5361" if DARK else "#C5CBD6"
# One-hue sequential ramp, weakest to strongest contrast against the surface.
RAMP = (["#2D6CB0", "#4F8BC6", "#80B0DD", "#B6D3EE", "#E6F0FA"] if DARK
        else ["#8DB8E0", "#5E97CF", "#3A7BBA", "#1F5C9A", "#0B3A6B"])


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ------------------------------------------------------------------ presets
ZETA_MODE, C_MODE = "Damping ratio ζ", "Damping coefficient c"
FREE, STEP, HARM = "Free vibration", "Step force", "Harmonic force"
BASE = dict(m=2.0, k=800.0, dmode=ZETA_MODE, zeta=0.1, kind=FREE, x0=50.0, v0=0.0, F0=100.0,
            r=1.0, auto=True, T=5.0)
PRESETS = {
    "Underdamped (ζ = 0.1)": dict(BASE),
    "Undamped (ζ = 0)": dict(BASE, zeta=0.0),
    "Critically damped (ζ = 1)": dict(BASE, zeta=1.0),
    "Overdamped (ζ = 2)": dict(BASE, zeta=2.0),
    "Step force (ζ = 0.2)": dict(BASE, zeta=0.2, kind=STEP, x0=0.0, F0=100.0),
    "Below resonance (r = 0.5)": dict(BASE, kind=HARM, x0=0.0, F0=50.0, r=0.5),
    "Resonance (r = 1)": dict(BASE, kind=HARM, x0=0.0, F0=50.0, r=1.0),
    "Vibration isolation (r = 3)": dict(BASE, kind=HARM, x0=0.0, F0=50.0, r=3.0),
    "Undamped resonance": dict(BASE, zeta=0.0, kind=HARM, x0=0.0,
                                                    F0=50.0, r=1.0),
}
DEFAULT = "Underdamped (ζ = 0.1)"


def apply_preset():
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value
    st.session_state["c_in"] = damping_for_ratio(st.session_state["m"], st.session_state["k"],
                                                 st.session_state["zeta"])


if "m" not in st.session_state:
    st.session_state["preset"] = DEFAULT
    apply_preset()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Setup")
    st.selectbox("Starting point", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Picking a preset fills in the fields below. You can edit any of them.")

    st.subheader("System")
    a, b = st.columns(2)
    a.number_input("Mass m (kg)", 0.01, 10000.0, step=0.5, key="m", format="%.3g")
    b.number_input("Stiffness k (N/m)", 1.0, 1e7, step=50.0, key="k", format="%.6g")
    st.radio("Specify damping as", [ZETA_MODE, C_MODE], key="dmode", horizontal=True)
    if st.session_state["dmode"] == ZETA_MODE:
        st.number_input("Damping ratio ζ", 0.0, 20.0, step=0.05, key="zeta", format="%.3f",
                        help="0 = no damping, below 1 = oscillates, 1 = critical, above 1 = overdamped")
        c_now = damping_for_ratio(st.session_state["m"], st.session_state["k"],
                                  st.session_state["zeta"])
        st.caption(f"c = {c_now:.4g} N·s/m")
    else:
        st.number_input("Damping c (N·s/m)", 0.0, 1e6, step=1.0, key="c_in", format="%.4g")
        c_now = st.session_state["c_in"]

    st.subheader("Starting state")
    a, b = st.columns(2)
    a.number_input("Displacement x₀ (mm)", -1000.0, 1000.0, step=5.0, key="x0", format="%.1f")
    b.number_input("Velocity v₀ (mm/s)", -50000.0, 50000.0, step=50.0, key="v0", format="%.1f")

    st.subheader("Applied force")
    st.radio("Type", [FREE, STEP, HARM], key="kind", label_visibility="collapsed")
    if st.session_state["kind"] != FREE:
        st.number_input("Force amplitude F₀ (N)", -1e6, 1e6, step=10.0, key="F0", format="%.4g")
    if st.session_state["kind"] == HARM:
        st.slider("Frequency ratio r = ω / ωn", 0.05, 4.0, step=0.01, key="r")

    st.subheader("Time")
    st.checkbox("Choose the duration automatically", key="auto")
    if not st.session_state["auto"]:
        st.number_input("Duration (s)", 0.1, 600.0, step=1.0, key="T")

    st.subheader("Animation")
    speed = st.select_slider("Playback speed", [0.1, 0.25, 0.5, 1.0, 2.0], value=1.0,
                             format_func=lambda s: f"{s}×")

# ------------------------------------------------------------------ model
S = st.session_state
try:
    params = Params(float(S.m), float(c_now), float(S.k))
    pr = properties(params)
    if S.kind == FREE:
        load = Load("free")
    elif S.kind == STEP:
        load = Load("step", float(S.F0))
    else:
        load = Load("harmonic", float(S.F0), float(S.r) * pr.wn)
    x0, v0 = float(S.x0) / 1e3, float(S.v0) / 1e3
    T = float(suggest_duration(params, load)) if S.auto else float(S.T)
    n = suggest_samples(params, load, T)
except ValueError as err:
    st.error(str(err))
    st.stop()


@st.cache_data(show_spinner=False)
def run(m, c, k, kind, F0, omega, x0_, v0_, t_end, n_):
    return simulate(Params(m, c, k), Load(kind, F0, omega), x0_, v0_, t_end, n_)


resp = run(params.m, params.c, params.k, load.kind, load.F0, load.omega, x0, v0, T, n)
t = resp.t
x_mm, v_mm = resp.x * 1e3, resp.v * 1e3

st.title("Damped spring-mass system")
st.caption("A mass on a spring with a viscous damper:  m x″ + c x′ + k x = F(t).  The response is "
           "computed from the exact closed-form solution and checked against numerical "
           "integration. Displacement x is measured from the static equilibrium position.")


def fmt(v, unit="", digits=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    return f"{v:.{digits}g} {unit}".strip()


m1_, m2_, m3_, m4_, m5_ = st.columns(5)
m1_.metric("Natural frequency", fmt(pr.fn, "Hz"), f"ωn = {pr.wn:.4g} rad/s", delta_color="off")
m2_.metric("Damping ratio ζ", f"{pr.zeta:.4g}", pr.regime, delta_color="off")
m3_.metric("Damped period", fmt(pr.Td, "s"),
           f"ωd = {pr.wd:.4g} rad/s" if not math.isnan(pr.wd) else "does not oscillate",
           delta_color="off")
m4_.metric("Log decrement δ", fmt(pr.log_decrement),
           f"Q = {pr.Q:.3g}" if math.isfinite(pr.Q) else "Q = ∞", delta_color="off")
m5_.metric("Settling time (2 %)", fmt(pr.settling_time, "s"),
           f"step overshoot {pr.overshoot * 100:.1f} %" if pr.regime != "Undamped" else "never settles",
           delta_color="off")

if load.kind == "harmonic" and pr.zeta < 1e-12 and abs(float(S.r) - 1.0) < 1e-6:
    st.warning("Undamped resonance: with no damping at r = 1 the amplitude grows forever "
               "(linearly with time). Real systems always have some damping.")
elif (S.kind == FREE and x0 == 0 and v0 == 0) or (S.kind != FREE and load.F0 == 0 and x0 == 0 and v0 == 0):
    st.info("Nothing is moving yet: give the mass a starting displacement or velocity, or apply a force.")

tab_anim, tab_time, tab_freq, tab_how = st.tabs(
    ["Animation", "Time response", "Frequency response", "How it works"])

# ------------------------------------------------------------------ animation
FPS = 30
LS, AMP_VIS, WM, HM, ROD = 3.2, 0.5, 0.9, 0.9, 1.2     # drawing units


def spring_xy(xm, ys=0.2, coils=9, lead=0.25, amp=0.11):
    b0, b1 = lead, xm - lead
    xs, yv = [0.0, b0], [ys, ys]
    for i in range(2 * coils):
        xs.append(b0 + (i + 0.5) * (b1 - b0) / (2 * coils))
        yv.append(ys + (amp if i % 2 == 0 else -amp))
    return xs + [b1, xm], yv + [ys, ys]


def damper_xy(xm, yd=-0.2):
    if params.c <= 0:
        return [], []
    cyl0, cyl1, xp = 0.3, 2.6, xm - ROD
    return ([0, cyl0, None, cyl1, cyl0, cyl0, cyl1, None, xm, xp, None, xp, xp],
            [yd, yd, None, yd + .1, yd + .1, yd - .1, yd - .1, None, yd, yd, None, yd - .07, yd + .07])


def arrow_xy(xm, f, fmax):
    if fmax == 0 or abs(f) < 1e-9 * fmax:
        return [], []
    s = 1.0 if f > 0 else -1.0
    x0_ = xm + WM + 0.05 if s > 0 else xm - 0.05
    length = 0.9 * abs(f) / fmax
    tip = x0_ + s * length
    return ([x0_, tip, None, tip - s * .14, tip, tip - s * .14],
            [0, 0, None, .09, 0, -.09])


def build_animation():
    Tn_eff = 2 * math.pi / (max(pr.wn, load.omega) if load.kind == "harmonic" else pr.wn)
    dt = min(1.0 / FPS, Tn_eff / 20)
    nf = int(min(900, math.floor(T / dt) + 1))
    t_anim = (nf - 1) * dt
    r_ = run(params.m, params.c, params.k, load.kind, load.F0, load.omega, x0, v0, t_anim, nf)
    xs_ = r_.x
    xmax = max(float(np.max(np.abs(xs_))), 1e-12)
    vis = AMP_VIS / xmax
    xmm = xs_ * 1e3
    ylim = 1.18 * max(float(np.max(np.abs(xmm))), 1e-9)
    fmax = abs(load.F0) if load.kind != "free" else 0.0

    def parts(k):
        xm = LS + vis * xs_[k]
        sx, sy = spring_xy(xm)
        dx, dy = damper_xy(xm)
        ax_, ay_ = arrow_xy(xm, r_.force[k], fmax)
        return (dict(x=sx, y=sy), dict(x=dx, y=dy),
                dict(x=[xm, xm + WM, xm + WM, xm, xm], y=[-HM / 2, -HM / 2, HM / 2, HM / 2, -HM / 2]),
                dict(x=ax_, y=ay_),
                dict(x=r_.t[:k + 1], y=xmm[:k + 1]),
                dict(x=[r_.t[k]], y=[xmm[k]]))

    first = parts(0)
    hatch_x, hatch_y = [0, 0], [-0.6, 0.6]
    for yy in np.linspace(-0.5, 0.6, 8):
        hatch_x += [None, 0, -0.2]
        hatch_y += [None, yy, yy - 0.12]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.4, 0.6], vertical_spacing=0.08)
    add = lambda tr, row: fig.add_trace(tr, row=row, col=1)
    add(go.Scatter(**first[0], mode="lines", line=dict(color=INK, width=2.5),
                   hoverinfo="skip", showlegend=False), 1)                                  # 0 spring
    add(go.Scatter(**first[1], mode="lines", line=dict(color=MUTED, width=2.5),
                   hoverinfo="skip", showlegend=False), 1)                                  # 1 damper
    add(go.Scatter(**first[2], mode="lines", fill="toself", fillcolor=rgba(C1, 0.45),
                   line=dict(color=INK, width=2), hoverinfo="skip", showlegend=False), 1)   # 2 mass
    add(go.Scatter(**first[3], mode="lines", line=dict(color=C2, width=5),
                   hoverinfo="skip", showlegend=False), 1)                                  # 3 force
    add(go.Scatter(x=hatch_x + [None, 0, 5.7], y=hatch_y + [None, -0.5, -0.5], mode="lines",
                   line=dict(color=INK, width=2), hoverinfo="skip", showlegend=False), 1)    # 4 wall, floor
    add(go.Scatter(x=[LS, LS], y=[-0.7, 0.7], mode="lines+text", text=["", "equilibrium"],
                   textposition="top center", textfont=dict(color=MUTED, size=12),
                   line=dict(color=MUTED, width=1.5, dash="dash"), hoverinfo="skip",
                   showlegend=False), 1)                                                    # 5 reference
    add(go.Scatter(x=r_.t, y=xmm, mode="lines", line=dict(color=C1, width=1.5), opacity=0.25,
                   hoverinfo="skip", showlegend=False), 2)                                  # 6 ghost
    add(go.Scatter(**first[4], mode="lines", line=dict(color=C1, width=3),
                   hovertemplate="t = %{x:.3f} s<br>x = %{y:.2f} mm<extra></extra>",
                   showlegend=False), 2)                                                    # 7 progress
    add(go.Scatter(**first[5], mode="markers", marker=dict(color=C1, size=12,
                   line=dict(color=INK, width=1.5)), hoverinfo="skip", showlegend=False), 2)  # 8 dot
    frames = [go.Frame(data=[go.Scatter(**d) for d in (parts(k)[0], parts(k)[1], parts(k)[2],
                                                       parts(k)[3], parts(k)[4], parts(k)[5])],
                       traces=[0, 1, 2, 3, 7, 8], name=str(k)) for k in range(nf)]
    fig.frames = frames
    anim = dict(frame=dict(duration=1000.0 / FPS / speed, redraw=False), transition=dict(duration=0),
                mode="immediate")
    fig.update_layout(
        height=760, margin=dict(l=10, r=10, t=20, b=120), showlegend=False,
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top",
            pad=dict(t=12, r=8), showactive=False, bgcolor=BTN_BG, bordercolor=BTN_BORDER,
            font=dict(color=INK, size=13),
            buttons=[dict(label="▶ Play", method="animate", args=[None, dict(anim, fromcurrent=True)]),
                     dict(label="❚❚ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                             transition=dict(duration=0))])])],
        sliders=[dict(
            active=0, x=0.17, len=0.83, y=-0.12, yanchor="top", pad=dict(t=8, b=0),
            bgcolor=BTN_BORDER, activebgcolor=C1, bordercolor="rgba(0,0,0,0)",
            currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=15, color=INK)),
            ticklen=0, tickcolor="rgba(0,0,0,0)", font=dict(color="rgba(0,0,0,0)", size=1),
            steps=[dict(method="animate", label=f"{r_.t[k]:.2f}",
                        args=[[str(k)], dict(anim, frame=dict(duration=0, redraw=False))])
                   for k in range(nf)])])
    fig.update_xaxes(range=[-0.5, 5.7], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-0.85, 1.0], visible=False, scaleanchor="x", scaleratio=1,
                     constrain="domain", row=1, col=1)
    fig.update_xaxes(range=[0, t_anim], title="Time (s)", gridcolor=GRID, zeroline=False, row=2, col=1)
    fig.update_yaxes(range=[-ylim, ylim], title="Displacement (mm)", gridcolor=GRID,
                     zeroline=True, zerolinecolor=MUTED, row=2, col=1)
    return fig, dt, nf, t_anim


PLOT_CFG = dict(displaylogo=False)
with tab_anim:
    fig_a, dt_a, nf_a, t_anim = build_animation()
    st.plotly_chart(fig_a, theme="streamlit", width="stretch",
                    config=dict(displaylogo=False, displayModeBar=False))
    ratio = dt_a * FPS * speed
    st.caption(
        "Press Play. Drag the slider to scrub through time. The spring is on top and the damper "
        "is below it; the orange arrow is the applied force. The motion is magnified so you "
        "can see it, so the drawing is not to scale. "
        + (f"Shows the first {t_anim:.2f} s of {T:.2f} s. " if t_anim < T - 1e-9 else "")
        + (f"Playback runs at {ratio:.2g}× real time." if abs(ratio - 1) > 0.02 else "Playback runs in real time."))


# ------------------------------------------------------------------ time response
def style(fig, ytitle, xtitle="Time (s)", height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    return fig


def line(x, y, name, color, dash="solid", width=2.2):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash))


with tab_time:
    cl, cr = st.columns(2)
    with cl:
        f = go.Figure(line(t, x_mm, "Displacement", C1))
        if load.kind == "free" and pr.regime == "Underdamped":
            sig = pr.zeta * pr.wn
            amp = 1e3 * math.hypot(x0, (v0 + sig * x0) / pr.wd)
            env = amp * np.exp(-sig * t)
            f.add_trace(line(t, env, "Decay envelope", MUTED, "dash", 1.6))
            f.add_trace(line(t, -env, "Decay envelope ", MUTED, "dash", 1.6))
            f.data[-1].showlegend = False
        elif load.kind == "step":
            f.add_trace(line(t, np.full_like(t, 1e3 * load.F0 / params.k), "Static deflection F₀/k", MUTED, "dash", 1.6))
        elif load.kind == "harmonic":
            X = 1e3 * abs(load.F0) / params.k * float(magnification(load.omega / pr.wn, pr.zeta))
            if math.isfinite(X):
                f.add_trace(line(t, np.full_like(t, X), "Steady-state amplitude", MUTED, "dash", 1.6))
                f.add_trace(line(t, np.full_like(t, -X), "Steady-state amplitude ", MUTED, "dash", 1.6))
                f.data[-1].showlegend = False
        st.plotly_chart(style(f, "Displacement x (mm)"), theme="streamlit", width="stretch", config=PLOT_CFG)
    with cr:
        f = go.Figure(line(t, v_mm, "Velocity", C2))
        st.plotly_chart(style(f, "Velocity v (mm/s)"), theme="streamlit", width="stretch", config=PLOT_CFG)
    cl, cr = st.columns(2)
    with cl:
        f = go.Figure()
        f.add_trace(go.Scatter(x=x_mm, y=v_mm, mode="lines", name="Path", line=dict(color=C1, width=2.2)))
        f.add_trace(go.Scatter(x=[x_mm[0]], y=[v_mm[0]], mode="markers", name="Start",
                               marker=dict(color=C2, size=11, line=dict(color=INK, width=1.5))))
        f.add_trace(go.Scatter(x=[x_mm[-1]], y=[v_mm[-1]], mode="markers", name="End",
                               marker=dict(color=INK, size=10, symbol="square")))
        f.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
        f.update_xaxes(title="Displacement x (mm)", gridcolor=GRID, zeroline=False)
        f.update_yaxes(title="Velocity v (mm/s)", gridcolor=GRID, zeroline=False)
        st.plotly_chart(f, theme="streamlit", width="stretch", config=PLOT_CFG)
        st.caption("Phase portrait. Damping makes the path spiral in toward the origin (rest).")
    with cr:
        f = go.Figure()
        total = resp.kinetic + resp.potential
        f.add_trace(line(t, resp.kinetic, "Kinetic", C1))
        f.add_trace(line(t, resp.potential, "Potential (spring)", C2))
        f.add_trace(line(t, total, "Total mechanical", INK, "dash", 1.8))
        f.add_trace(line(t, resp.dissipated, "Dissipated by damper", MUTED, "dot", 2.2))
        if load.kind != "free":
            f.add_trace(line(t, resp.work_in, "Work done by force", C2, "dashdot", 1.8))
        st.plotly_chart(style(f, "Energy (J)"), theme="streamlit", width="stretch", config=PLOT_CFG)
        st.caption("Work in = change in total energy + energy dissipated.")
    if load.kind != "free":
        f = go.Figure(line(t, resp.force, "Applied force", C2))
        st.plotly_chart(style(f, "Force F (N)", height=260), theme="streamlit", width="stretch", config=PLOT_CFG)
    out = pd.DataFrame({"t (s)": t, "x (mm)": x_mm, "v (mm/s)": v_mm, "F (N)": resp.force,
                        "KE (J)": resp.kinetic, "PE (J)": resp.potential,
                        "Dissipated (J)": resp.dissipated})
    st.download_button("Download the response as CSV", out.to_csv(index=False),
                       file_name="spring_mass_response.csv", mime="text/csv")

# ------------------------------------------------------------------ frequency response
REF_ZETAS = [0.05, 0.1, 0.2, 0.5, 1.0]
with tab_freq:
    st.write("Steady-state response to a harmonic force F₀ cos(ωt), as a function of the frequency "
             "ratio r = ω / ωn. The thick orange line is your damping ratio; the blue lines (darkest = least damping) are for comparison.")
    r_axis = np.linspace(0.0, 3.0, 1201)
    zeta_user = pr.zeta
    peak_r, peak_m = resonance_peak(zeta_user)
    y_top = 12.0 if not math.isfinite(peak_m) else float(min(12.0, max(3.0, 1.15 * peak_m)))
    show_marker = load.kind == "harmonic"
    r_now = float(S.r) if show_marker else None

    def ramp_colour(z):   # low damping = strongest contrast
        return RAMP[len(RAMP) - 1 - REF_ZETAS.index(z)]

    def curves(fn, ytitle, yrange=None, extra=None):
        f = go.Figure()
        for z in REF_ZETAS:
            f.add_trace(line(r_axis, fn(r_axis, z), f"ζ = {z:g}", ramp_colour(z), width=1.8))
        f.add_trace(line(r_axis, fn(r_axis, zeta_user), f"Yours, ζ = {zeta_user:.3g}", C2, width=4.5))
        if show_marker:
            f.add_trace(go.Scatter(x=[r_now], y=[float(fn(r_now, zeta_user))], mode="markers",
                                   name=f"Now, r = {r_now:.2f}",
                                   marker=dict(color=C2, size=13, line=dict(color=INK, width=1.5))))
        if extra:
            extra(f)
        f.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
        f.update_xaxes(title="Frequency ratio r = ω / ωn", gridcolor=GRID, zeroline=False, range=[0, 3])
        f.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, range=yrange)
        return f

    c1_, c2_ = st.columns(2)
    with c1_:
        st.markdown("**Magnification** X / (F₀/k)")
        st.plotly_chart(curves(magnification, "Amplitude ratio", [0, y_top]), theme="streamlit",
                        width="stretch", config=PLOT_CFG)
    with c2_:
        st.markdown("**Phase lag** of the displacement behind the force")
        f = curves(phase_lag, "Phase lag (°)", [0, 180])
        f.update_yaxes(tickvals=[0, 45, 90, 135, 180])
        st.plotly_chart(f, theme="streamlit", width="stretch", config=PLOT_CFG)
    c3_, c4_ = st.columns(2)
    with c3_:
        st.markdown("**Transmissibility** (vibration isolation)")

        def iso_marks(f):
            f.add_vline(x=math.sqrt(2), line=dict(color=MUTED, width=1.5, dash="dash"),
                        annotation_text="r = √2", annotation_position="top right",
                        annotation_font=dict(color=MUTED))
            f.add_hline(y=1.0, line=dict(color=MUTED, width=1.5, dash="dot"))
        st.plotly_chart(curves(transmissibility, "Transmissibility", [0, min(y_top, 6.0)], iso_marks),
                        theme="streamlit", width="stretch", config=PLOT_CFG)
        st.caption("Above r = √2 the transmitted force is smaller than the applied force "
                   "(isolation), and more damping makes it worse there.")
    with c4_:
        st.markdown("**Key numbers**")
        rows = [("Resonance peak at r", fmt(peak_r) if peak_r > 0 else "no peak (ζ ≥ 0.707)"),
                ("Peak magnification", fmt(peak_m)),
                ("Quality factor Q = 1 / (2ζ)", fmt(pr.Q)),
                ("Half-power bandwidth (≈ 2ζ ωn)", fmt(2 * zeta_user * pr.wn, "rad/s") if zeta_user < 0.3 else "–")]
        if show_marker:
            Mg = float(magnification(r_now, zeta_user))
            rows += [("— at your frequency —", ""),
                     ("Forcing frequency", f"{load.omega:.4g} rad/s ({load.omega / 2 / math.pi:.4g} Hz)"),
                     ("Steady-state amplitude X", fmt(1e3 * abs(load.F0) / params.k * Mg, "mm")),
                     ("Phase lag", fmt(float(phase_lag(r_now, zeta_user)), "°")),
                     ("Transmitted force amplitude", fmt(abs(load.F0) * float(transmissibility(r_now, zeta_user)), "N"))]
        else:
            rows.append(("Tip", "Choose “Harmonic force” in the sidebar to mark your own frequency."))
        st.table(pd.DataFrame(rows, columns=["Quantity", "Value"]).set_index("Quantity"))

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("**Equation of motion** (Newton's second law for the mass)")
    st.latex(r"m\ddot{x}+c\dot{x}+kx=F(t)")
    st.markdown("Dividing by m gives the standard form, with the natural frequency and damping ratio:")
    st.latex(r"\ddot{x}+2\zeta\omega_n\dot{x}+\omega_n^2x=\frac{F(t)}{m},\qquad "
             r"\omega_n=\sqrt{\frac{k}{m}},\quad \zeta=\frac{c}{2\sqrt{km}}=\frac{c}{c_{cr}}")
    st.markdown("**Free response** depends on ζ:")
    st.latex(r"\zeta<1:\; x=e^{-\zeta\omega_nt}\left[x_0\cos\omega_dt+\frac{v_0+\zeta\omega_nx_0}{\omega_d}\sin\omega_dt\right],"
             r"\quad \omega_d=\omega_n\sqrt{1-\zeta^2}")
    st.latex(r"\zeta=1:\; x=e^{-\omega_nt}\left[x_0+(v_0+\omega_nx_0)t\right]")
    st.latex(r"\zeta>1:\; x=C_1e^{s_1t}+C_2e^{s_2t},\quad s_{1,2}=-\omega_n\left(\zeta\mp\sqrt{\zeta^2-1}\right)")
    st.markdown("**Log decrement.** Each cycle the amplitude shrinks by the same factor, so "
                "a measured decay gives the damping:")
    st.latex(r"\delta=\ln\frac{x_n}{x_{n+1}}=\frac{2\pi\zeta}{\sqrt{1-\zeta^2}}\approx2\pi\zeta\ (\text{small }\zeta)")
    st.markdown("**Step force.** The mass settles at F₀/k. With ζ < 1 it overshoots by")
    st.latex(r"M_p=e^{-\pi\zeta/\sqrt{1-\zeta^2}},\qquad t_p=\frac{\pi}{\omega_d},\qquad t_s\approx\frac{4}{\zeta\omega_n}\ (2\%)")
    st.markdown("**Harmonic force** F₀ cos ωt, with r = ω / ωn. After the transient dies away, "
                "the mass moves at the forcing frequency:")
    st.latex(r"x_{ss}=X\cos(\omega t-\phi),\quad X=\frac{F_0/k}{\sqrt{(1-r^2)^2+(2\zeta r)^2}},\quad "
             r"\tan\phi=\frac{2\zeta r}{1-r^2}")
    st.markdown("The amplitude peaks near r = √(1 − 2ζ²) at 1 / (2ζ√(1 − ζ²)). At r = 1 the "
                "phase lag is exactly 90° and X = F₀/(cωn). With no damping, X grows without limit.")
    st.markdown("**Transmissibility** is the fraction of the force that reaches the support (or of "
                "base motion that reaches the mass):")
    st.latex(r"TR=\sqrt{\frac{1+(2\zeta r)^2}{(1-r^2)^2+(2\zeta r)^2}}")
    st.markdown("It equals 1 at r = √2 for every ζ, and isolation only works above that.")
    st.markdown("**How it is solved.** The app uses the exact closed-form solution, so there is no "
                "time-stepping error. The full response is the particular solution plus a free "
                "response fitted to the starting conditions. `test_vibration_solver.py` checks it "
                "against a high-accuracy numerical ODE solver in every damping regime.")
