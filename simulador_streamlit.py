# ============================================================
# SSR SIMULATOR - STREAMLIT
# ============================================================

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go

from scipy.stats import gamma as gamma_dist
from scipy.special import gamma, gammainc
from scipy.integrate import quad
from scipy.optimize import minimize

# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="SSR Simulator", layout="wide")

st.title("Stress-Strength Reliability Simulator")

# ============================================================
# DISTRIBUTION
# ============================================================


def dS(x, mu, gam, alpha):

    den = gamma(mu) * (alpha ** (-mu) + (2 * gam + alpha) ** (-mu))

    num = x ** (mu - 1) * (np.exp(-alpha * x) + np.exp(-(2 * gam + alpha) * x))

    return num / den


def pS(x, mu, gam, alpha):

    den = alpha ** (-mu) + (2 * gam + alpha) ** (-mu)

    t1 = gammainc(mu, alpha * x) * alpha ** (-mu)

    t2 = gammainc(mu, (2 * gam + alpha) * x) * (2 * gam + alpha) ** (-mu)

    return (t1 + t2) / den


# ============================================================
# MOMENTS
# ============================================================


def moment_S(r, mu, gam, alpha):

    den = alpha ** (-mu) + (2 * gam + alpha) ** (-mu)

    num = (
        gamma(mu + r)
        / gamma(mu)
        * (alpha ** (-(mu + r)) + (2 * gam + alpha) ** (-(mu + r)))
    )

    return num / den


def mean_S(mu, gam, alpha):

    return moment_S(1, mu, gam, alpha)


def var_S(mu, gam, alpha):

    EX = moment_S(1, mu, gam, alpha)
    EX2 = moment_S(2, mu, gam, alpha)

    return EX2 - EX**2


# ============================================================
# RANDOM GENERATION
# ============================================================


def mixture_weights(mu, gam, alpha):

    w1 = alpha ** (-mu)

    w2 = (2 * gam + alpha) ** (-mu)

    den = w1 + w2

    return w1 / den, w2 / den


def rS(n, mu, gam, alpha):

    pi1, pi2 = mixture_weights(mu, gam, alpha)

    z = np.random.binomial(1, pi1, size=n)

    x = np.empty(n)

    n1 = np.sum(z == 1)
    n2 = np.sum(z == 0)

    x[z == 1] = gamma_dist.rvs(a=mu, scale=1 / alpha, size=n1)

    x[z == 0] = gamma_dist.rvs(a=mu, scale=1 / (2 * gam + alpha), size=n2)

    return x


# ============================================================
# SSR
# ============================================================


@st.cache_data
def reliability(mu_x, gam_x, alpha_x, mu_y, gam_y, alpha_y):

    integrand = lambda y: (pS(y, mu_x, gam_x, alpha_x) * dS(y, mu_y, gam_y, alpha_y))

    val, _ = quad(integrand, 0, np.inf, limit=200)

    return val


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Stress Distribution")

mu_x = st.sidebar.number_input(
    "μ_X", min_value=0.0001, value=2.0, step=0.1, format="%.4f"
)

gam_x = st.sidebar.number_input(
    "γ_X", min_value=0.0001, value=0.5, step=0.1, format="%.4f"
)

alpha_x = st.sidebar.number_input(
    "α_X", min_value=0.0001, value=1.0, step=0.1, format="%.4f"
)

st.sidebar.divider()

st.sidebar.header("Strength Distribution")

mu_y = st.sidebar.number_input(
    "μ_Y", min_value=0.0001, value=3.0, step=0.1, format="%.4f"
)

gam_y = st.sidebar.number_input(
    "γ_Y", min_value=0.0001, value=0.5, step=0.1, format="%.4f"
)

alpha_y = st.sidebar.number_input(
    "α_Y", min_value=0.0001, value=0.8, step=0.1, format="%.4f"
)

st.sidebar.divider()

n = st.sidebar.number_input("Sample Size", min_value=100, value=2000, step=100)

# ============================================================
# COMPUTING
# ============================================================

with st.spinner("Computing SSR..."):

    R = reliability(mu_x, gam_x, alpha_x, mu_y, gam_y, alpha_y)

    X = rS(n, mu_x, gam_x, alpha_x)

    Y = rS(n, mu_y, gam_y, alpha_y)

    R_emp = np.mean(X < Y)

# ============================================================
# MOMENTS
# ============================================================

mean_x = mean_S(mu_x, gam_x, alpha_x)

mean_y = mean_S(mu_y, gam_y, alpha_y)

sd_x = np.sqrt(var_S(mu_x, gam_x, alpha_x))

sd_y = np.sqrt(var_S(mu_y, gam_y, alpha_y))

# ============================================================
# PLOT
# ============================================================

xmax = np.quantile(np.concatenate([X, Y]), 0.995)

grid = np.linspace(0, xmax, 1000)

fx = dS(grid, mu_x, gam_x, alpha_x)

fy = dS(grid, mu_y, gam_y, alpha_y)

fig = go.Figure()

# ============================================================
# STRESS
# ============================================================

fig.add_trace(
    go.Scatter(
        x=grid, y=fx, mode="lines", name="Stress", line=dict(color="firebrick", width=3)
    )
)

# ============================================================
# STRENGTH
# ============================================================

fig.add_trace(
    go.Scatter(
        x=grid,
        y=fy,
        mode="lines",
        name="Strength",
        line=dict(color="royalblue", width=3),
    )
)

# ============================================================
# OVERLAP
# ============================================================

fig.add_trace(
    go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate([np.minimum(fx, fy), np.zeros_like(grid)]),
        fill="toself",
        fillcolor="rgba(128,0,128,0.20)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Overlap",
    )
)

# ============================================================
# LAYOUT
# ============================================================

graph_col, metrics_col = st.columns([4.5, 1.5])

# ============================================================
# GRAPH COLUMN
# ============================================================

with graph_col:

    xmax = np.quantile(np.concatenate([X, Y]), 0.995)

    grid = np.linspace(0, xmax, 1000)

    fx = dS(grid, mu_x, gam_x, alpha_x)

    fy = dS(grid, mu_y, gam_y, alpha_y)

    fig = go.Figure()

    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=fx,
            mode="lines",
            name="Stress",
            line=dict(color="firebrick", width=4),
        )
    )

    # --------------------------------------------------------
    # STRENGTH
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=fy,
            mode="lines",
            name="Strength",
            line=dict(color="royalblue", width=4),
        )
    )

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([np.minimum(fx, fy), np.zeros_like(grid)]),
            fill="toself",
            fillcolor="rgba(128,0,128,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Overlap",
        )
    )

    # ============================================================
    # MEAN LINES
    # ============================================================

    fig.add_vline(
        x=mean_x,
        line_width=2,
        line_dash="dash",
        line_color="firebrick",
        annotation_text="E[X]",
        annotation_position="top left",
    )

    fig.add_vline(
        x=mean_y,
        line_width=2,
        line_dash="dash",
        line_color="royalblue",
        annotation_text="E[Y]",
        annotation_position="top right",
    )

    # ============================================================
    # STANDARD DEVIATION REGIONS
    # ============================================================

    fig.add_vrect(
        x0=mean_x - sd_x, x1=mean_x + sd_x, fillcolor="red", opacity=0.08, line_width=0
    )

    fig.add_vrect(
        x0=mean_y - sd_y, x1=mean_y + sd_y, fillcolor="blue", opacity=0.08, line_width=0
    )

    # ============================================================
    # PARAMETER ANNOTATIONS
    # ============================================================

    fig.add_annotation(
        x=mean_x,
        y=max(fx),
        text=(
            f"Stress<br>" f"μ={mu_x:.2f}<br>" f"γ={gam_x:.2f}<br>" f"α={alpha_x:.2f}"
        ),
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(255,240,240,0.9)",
        bordercolor="firebrick",
    )

    fig.add_annotation(
        x=mean_y,
        y=max(fy),
        text=(
            f"Strength<br>" f"μ={mu_y:.2f}<br>" f"γ={gam_y:.2f}<br>" f"α={alpha_y:.2f}"
        ),
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(240,245,255,0.9)",
        bordercolor="royalblue",
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.update_layout(
        title=(f"Stress-Strength Reliability " f"| R = {R:.4f}"),
        template="plotly_white",
        height=720,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="x",
        yaxis_title="Density",
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# METRICS COLUMN
# ============================================================

with metrics_col:

    st.markdown(
        """
    <style>
    .small-title {
        font-size:18px !important;
        font-weight:700 !important;
        margin-bottom:10px !important;
    }

    .metric-box {
        background-color:#f7f7f7;
        padding:10px;
        border-radius:10px;
        margin-bottom:10px;
    }

    .metric-label {
        font-size:11px;
        color:#666;
    }

    .metric-value {
        font-size:20px;
        font-weight:700;
    }

    .section-title {
        font-size:15px;
        font-weight:700;
        margin-top:15px;
        margin-bottom:8px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # SSR
    # ========================================================

    st.markdown('<div class="small-title">SSR Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    # --------------------------------------------------------
    # ANALYTICAL
    # --------------------------------------------------------

    with m1:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    Analytical
                </div>
                <div class="metric-value">
                    {R:.6f}
                </div>
            </div> 
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # EMPIRICAL
    # --------------------------------------------------------

    with m2:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    Empirical
                </div>
                <div class="metric-value">
                    {R_emp:.6f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # ABS ERROR
    # --------------------------------------------------------

    with m3:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    Abs Error
                </div>
                <div class="metric-value">
                    {abs(R - R_emp):.6f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">Distributions</div>', unsafe_allow_html=True
    )

    colA, colB = st.columns(2)

    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    with colA:

        st.markdown(
            """
        <div style="
            background-color:#ffe5e5;
            padding:10px;
            border-radius:10px;
        ">
        <div style="
            font-size:14px;
            font-weight:700;
            color:firebrick;
            margin-bottom:6px;
        ">
        Stress
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-label">E[X]</div>
        <div class="metric-value">
            {mean_S(mu_x, gam_x, alpha_x):.4f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-label">Var[X]</div>
        <div class="metric-value">
            {var_S(mu_x, gam_x, alpha_x):.4f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # PARAMETER EFFECTS
    # ============================================================

    def parameter_effects(mu, gam, alpha):

        effects = []

        # --------------------------------------------------------
        # MU
        # --------------------------------------------------------

        if mu < 1:

            effects.append(
                "• Low μ → strong right-skewness and high concentration near zero."
            )

        elif mu < 3:

            effects.append("• Moderate μ → smoother density with visible asymmetry.")

        else:

            effects.append(
                "• Large μ → density becomes more concentrated and bell-shaped."
            )

        # --------------------------------------------------------
        # ALPHA
        # --------------------------------------------------------

        if alpha < 0.5:

            effects.append("• Small α → slower exponential decay and heavier tail.")

        elif alpha < 2:

            effects.append("• Moderate α → balanced decay behavior.")

        else:

            effects.append("• Large α → faster decay and shorter tail.")

        # --------------------------------------------------------
        # GAMMA
        # --------------------------------------------------------

        if gam < 0.5:

            effects.append("• Small γ → stronger overlap between mixture components.")

        elif gam < 2:

            effects.append("• Moderate γ → moderate separation between components.")

        else:

            effects.append(
                "• Large γ → second mixture component becomes more localized."
            )

        return effects

    # --------------------------------------------------------
    # STRENGTH
    # --------------------------------------------------------

    with colB:

        st.markdown(
            """
        <div style="
            background-color:#e8f0ff;
            padding:10px;
            border-radius:10px;
        ">
        <div style="
            font-size:14px;
            font-weight:700;
            color:royalblue;
            margin-bottom:6px;
        ">
        Strength
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-label">E[Y]</div>
        <div class="metric-value">
            {mean_S(mu_y, gam_y, alpha_y):.4f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-label">Var[Y]</div>
        <div class="metric-value">
            {var_S(mu_y, gam_y, alpha_y):.4f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # PARAMETER EFFECTS
    # ============================================================

    st.markdown(
        '<div class="section-title">Parameter Effects</div>', unsafe_allow_html=True
    )

    tabs = st.tabs(["Stress", "Strength"])

    # ------------------------------------------------------------
    # STRESS
    # ------------------------------------------------------------

    with tabs[0]:

        effects_x = parameter_effects(mu_x, gam_x, alpha_x)

        for eff in effects_x:

            st.markdown(
                f"<span style='font-size:13px'>{eff}</span>", unsafe_allow_html=True
            )

    # ------------------------------------------------------------
    # STRENGTH
    # ------------------------------------------------------------

    with tabs[1]:

        effects_y = parameter_effects(mu_y, gam_y, alpha_y)

        for eff in effects_y:

            st.markdown(
                f"<span style='font-size:13px'>{eff}</span>", unsafe_allow_html=True
            )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Interpretation</div>', unsafe_allow_html=True
    )

    if R > 0.9:

        st.success("Very high reliability")

    elif R > 0.7:

        st.info("Moderate reliability")

    elif R > 0.5:

        st.warning("Marginal reliability")

    else:

        st.error("Low reliability")
