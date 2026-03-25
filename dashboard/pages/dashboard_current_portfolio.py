"""
dashboard_current_portfolio.py
=================
Streamlit Portfolio-Dashboard.
Ablegen in: dashboard/pages/dashboard_current_portfolio.py

Starten:
    streamlit run dashboard/pages/dashboard_current_portfolio.py

Pakete:
    pip install streamlit plotly pandas numpy openpyxl
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_portfolio, ASSET_META

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #C8D0DC;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #E8EDF5;
    letter-spacing: -0.02em;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0F1622 0%, #111827 100%);
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
}
[data-testid="metric-container"] label {
    color: #6B7E96 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #E8EDF5 !important;
    font-size: 1.55rem !important;
    font-weight: 500;
}
.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3D6090;
    margin-bottom: 0.3rem;
    font-weight: 500;
}
.lambda-box {
    background: linear-gradient(135deg, #0F1E35 0%, #0A1525 100%);
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin: 0.5rem 0;
}
.lambda-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #3B82F6;
    line-height: 1;
    margin: 0.4rem 0;
}
.lambda-interp {
    font-size: 0.85rem;
    color: #6B9ED4;
    margin-top: 0.3rem;
}
hr { border-color: #1A2332 !important; margin: 2rem 0; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
div[data-testid="stToggle"] { margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444",
          "#8B5CF6", "#06B6D4", "#F97316", "#84CC16"]

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0B1120",
    font=dict(family="DM Sans", color="#8A9BB0", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#141E2D", linecolor="#1A2332", zeroline=False),
    yaxis=dict(gridcolor="#141E2D", linecolor="#1A2332", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1A2332",
                font=dict(size=10, color="#8A9BB0")),
)

RF = 0.02


def layout(fig, title="", height=340):
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title, font=dict(family="DM Serif Display", size=14, color="#C8D0DC")),
        height=height,
    )
    return fig


# ─────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Portfoliodaten …")
def get_data():
    return load_portfolio(PROJECT_ROOT / "data" / "Old Portfolio")


data = get_data()

prices_daily = data["prices_daily"]
prices_monthly = data["prices_monthly"]
weights        = data["weights"]
summary_df     = data["summary_df"]
total_value    = data["total_value"]
start_date     = data["start_date"]

asset_colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(prices_daily.columns)}


# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────
def compute_metrics(returns: pd.DataFrame, weights: pd.Series, freq: str) -> pd.DataFrame:
    """Compute performance metrics for the given return frequency."""
    scale = 252 if freq == "daily" else 12

    rows = []
    cols = weights.index.tolist()

    for col in returns.columns:
        r      = returns[col].dropna()
        cum    = (1 + r).cumprod()
        total  = cum.iloc[-1] - 1
        n      = len(r)
        ann_r  = (1 + total) ** (scale / n) - 1
        ann_v  = r.std() * np.sqrt(scale)
        sharpe = ann_r / ann_v if ann_v > 0 else np.nan
        neg    = r[r < 0]
        ds     = neg.std() * np.sqrt(scale) if len(neg) > 0 else np.nan
        sortino = ann_r / ds if ds and ds > 0 else np.nan
        roll_max = cum.cummax()
        max_dd   = ((cum - roll_max) / roll_max).min()
        cvar     = r[r <= r.quantile(0.05)].mean()
        rows.append({
            "asset":        col,
            "name":         ASSET_META.get(col, {}).get("name", col),
            "ann_return":   round(ann_r  * 100, 2),
            "ann_vol":      round(ann_v  * 100, 2),
            "sharpe":       round(sharpe, 3),
            "sortino":      round(sortino, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "cvar_95":      round(cvar   * 100, 2),
            "total_return": round(total  * 100, 2),
        })

    # Portfolio
    w       = weights.reindex(returns.columns).dropna()
    w       = w / w.sum()
    port_r  = returns[w.index].mul(w, axis=1).sum(axis=1)
    cum     = (1 + port_r).cumprod()
    total   = cum.iloc[-1] - 1
    n       = len(port_r)
    ann_r   = (1 + total) ** (scale / n) - 1
    ann_v   = port_r.std() * np.sqrt(scale)
    sharpe  = ann_r / ann_v if ann_v > 0 else np.nan
    neg     = port_r[port_r < 0]
    ds      = neg.std() * np.sqrt(scale) if len(neg) > 0 else np.nan
    sortino = ann_r / ds if ds and ds > 0 else np.nan
    roll_max = cum.cummax()
    max_dd   = ((cum - roll_max) / roll_max).min()
    cvar     = port_r[port_r <= port_r.quantile(0.05)].mean()
    rows.append({
        "asset":        "PORTFOLIO",
        "name":         "Portfolio Gesamt",
        "ann_return":   round(ann_r  * 100, 2),
        "ann_vol":      round(ann_v  * 100, 2),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "cvar_95":      round(cvar   * 100, 2),
        "total_return": round(total  * 100, 2),
    })

    return pd.DataFrame(rows).set_index("asset")


def compute_lambda(returns: pd.DataFrame, weights: pd.Series, freq: str, rf: float = RF):
    scale = 252 if freq == "daily" else 12
    cols = weights.index.tolist()
    r    = returns[cols].dropna()
    w    = weights.reindex(cols).dropna().values

    total_r = (1 + r).prod() - 1
    ann_r   = (1 + total_r) ** (scale / len(r)) - 1
    er      = ann_r.values
    cov_ann = r.cov().values * scale
    excess  = er - rf

    try:
        cov_inv = np.linalg.inv(cov_ann)
        denom   = float(w @ cov_inv @ excess)
        lam     = 1.0 / denom if abs(denom) > 1e-12 else np.nan
    except np.linalg.LinAlgError:
        lam, denom = np.nan, np.nan

    sens = []
    for rf_t in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        ex = er - rf_t
        try:
            d  = float(w @ cov_inv @ ex)
            l  = round(1.0 / d, 4) if abs(d) > 1e-12 else np.nan
        except Exception:
            l = np.nan
        sens.append({"r_f": f"{rf_t:.0%}", "λ": l})

    asset_df = pd.DataFrame({
        "Asset":    cols,
        "E(r)":     [f"{x:.2%}" for x in er],
        "E(r)-r_f": [f"{x:.2%}" for x in excess],
        "Weight":   [f"{v:.2%}" for v in w],
    })

    return {"lambda": lam, "asset_df": asset_df, "sens_df": pd.DataFrame(sens)}


def interp_lambda(lam):  # (float) -> Tuple[str, str]
    if np.isnan(lam):  return "Not computable", "#EF4444"
    if lam < 0:        return "Negative – implies irrational behavior", "#EF4444"
    if lam < 1:        return "Very low risk aversion (near risk-neutral)", "#F59E0B"
    if lam < 3:        return "Low to moderate risk aversion", "#84CC16"
    if lam < 6:        return "Moderate risk aversion (institutional)", "#10B981"
    if lam < 10:       return "High risk aversion", "#F59E0B"
    return                    "Very high risk aversion", "#EF4444"


# ─────────────────────────────────────────────
# HEADER + FREQUENZ-SCHALTER
# ─────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])

with h_left:
    st.markdown(f"""
    <div style="padding: 1.4rem 0 0.2rem 0;">
        <p class="section-label">Portfolio Intelligence</p>
        <h1 style="margin:0; font-size:2.1rem;">Current Portfolio — Analysis</h1>
        <p style="color:#4A6080; font-size:0.82rem; margin-top:0.3rem;">
            since {start_date} &nbsp;·&nbsp; {len(prices_daily.columns)} Assets &nbsp;·&nbsp;
            {prices_daily.index[0].strftime("%d.%m.%Y")} – {prices_daily.index[-1].strftime("%d.%m.%Y")}
        </p>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    st.markdown("<div style='padding-top:1.8rem;'>", unsafe_allow_html=True)
    use_monthly = st.toggle("📅  Monthly Returns", value=True,
                            help="On = monthly returns (×12) · Off = daily returns (×252)")
    st.markdown("</div>", unsafe_allow_html=True)


# ── German Bund YTM override ────────────────────────────────────────────
_BUND_TICKER = "BO221256 Corp"
_BUND_YTM_PA = 0.03074

def _override_bund(ret):
    """Shift Bund return series so that the GEOMETRIC annualised return
    equals the YTM (3.074 % p.a.) while preserving historical volatility.

    Uses scipy.optimize.brentq to find the exact additive shift s such that:
        geo_annualised(r_hist + s) = 3.074 %
    Volatility, correlations and distribution shape are fully preserved.
    """
    if _BUND_TICKER not in ret.columns:
        return ret
    from scipy.optimize import brentq
    ret  = ret.copy()
    r    = ret[_BUND_TICKER].dropna()
    n    = len(r)
    if n < 2:
        return ret

    def _geo_ann(series):
        total = (1 + series).prod() - 1
        return (1 + total) ** (12 / n) - 1

    def _objective(s):
        return _geo_ann(r + s) - _BUND_YTM_PA

    try:
        s_opt = brentq(_objective, -0.20, 0.20, xtol=1e-10)
    except ValueError:
        # fallback: arithmetic mean-shift if brentq bracket fails
        s_opt = _BUND_YTM_PA / 12 - r.mean()

    ret[_BUND_TICKER] = ret[_BUND_TICKER] + s_opt
    return ret

if use_monthly:
    returns  = _override_bund(prices_monthly.pct_change().dropna())
    freq     = "monthly"
    freq_lbl = "monthly returns"
    scale    = 12
else:
    returns  = _override_bund(prices_daily.pct_change().dropna())
    freq     = "daily"
    freq_lbl = "daily returns"
    scale    = 252

metrics_df = compute_metrics(returns, weights, freq)
port_m     = metrics_df.loc["PORTFOLIO"]

w_norm       = weights.reindex(returns.columns).dropna()
w_norm       = w_norm / w_norm.sum()
port_returns = returns[w_norm.index].mul(w_norm, axis=1).sum(axis=1)
port_cum     = (1 + port_returns).cumprod()
port_dd      = (port_cum - port_cum.cummax()) / port_cum.cummax()

st.caption(f"KPIs based on **{freq_lbl}** · r_f = {RF:.0%} · annualisation ×{scale}")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI ZEILE
# ─────────────────────────────────────────────
# Row 1: value metrics
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Portfolio Value (€ Mio.)",    f"{total_value/1e6:.2f}")
r1c2.metric("Ann. Return (%)",              f"{port_m['ann_return']:.2f}")
r1c3.metric("Ann. Volatility (%)",          f"{port_m['ann_vol']:.2f}")
r1c4.metric("Sharpe Ratio",                 f"{port_m['sharpe']:.3f}")

st.markdown("<div style='margin-top:.6rem;'>", unsafe_allow_html=True)

# Row 2: risk metrics
r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Sortino Ratio",                f"{port_m['sortino']:.3f}")
r2c2.metric("Max Drawdown (%)",             f"{port_m['max_drawdown']:.2f}")
r2c3.metric(
    "CVaR 95% (%, per period)",
    f"{port_m['cvar_95']:.2f}",
    help="Expected loss in the worst 5% of months. Expressed as % return per period."
)
r2c4.metric("Total Return (%)",             f"{port_m['total_return']:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# PERFORMANCE & DRAWDOWN
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    fig = go.Figure()
    for name in prices_daily.columns:
        r   = returns[name].dropna()
        cum = (1 + r).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=name,
            line=dict(color=asset_colors[name], width=1.4),
            hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values, name="Portfolio",
        line=dict(color="white", width=2.4, dash="dot"),
        hovertemplate="%{y:.3f}<extra>Portfolio</extra>",
    ))
    layout(fig, "Cumulative Returns (base = 1)", height=360)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    for name in prices_daily.columns:
        r   = returns[name].dropna()
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name,
            line=dict(color=asset_colors[name], width=1.2),
            hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=(port_dd * 100).index, y=(port_dd * 100).values, name="Portfolio",
        line=dict(color="white", width=2.4, dash="dot"),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
    ))
    layout(fig, "Drawdown (%)", height=360)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# KENNZAHLEN + KORRELATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Risk & Correlation</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1.1, 1])

with c1:
    disp = (metrics_df.reset_index()
            .rename(columns={
                "asset": "Ticker", "name": "Name",
                "ann_return": "Return %", "ann_vol": "Vola %",
                "sharpe": "Sharpe", "sortino": "Sortino",
                "max_drawdown": "Max DD %", "cvar_95": "CVaR 95%",
                "total_return": "Total %",
            }))
    mask  = disp["Ticker"] == "PORTFOLIO"
    disp  = pd.concat([disp[~mask], disp[mask]], ignore_index=True)
    st.markdown(f"**Key Metrics per Asset** *(basis: {freq_lbl})*")
    st.dataframe(
        disp[["Name", "Return %", "Vola %", "Sharpe", "Sortino", "Max DD %", "CVaR 95%"]],
        use_container_width=True, hide_index=True, height=340,
    )

with c2:
    corr  = returns.corr()
    short = [c.replace(" Equity", "").replace(" Corp", "") for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    layout(fig, f"Correlation Matrix ({freq_lbl})", height=380)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# LAMBDA
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Risk Aversion</p>', unsafe_allow_html=True)

lam_data  = compute_lambda(returns, weights, freq, RF)
lam       = lam_data["lambda"]
interp, lam_color = interp_lambda(lam)

c1, c2, c3 = st.columns([1, 1.3, 1.2])

with c1:
    lam_str = f"{lam:.3f}" if not np.isnan(lam) else "n/a"
    st.markdown(f"""
    <div class="lambda-box">
        <p style="font-size:0.7rem; letter-spacing:0.1em; color:#3D6090; text-transform:uppercase; margin:0;">
            Risk Aversion Coefficient
        </p>
        <div class="lambda-value" style="color:{lam_color};">{lam_str}</div>
        <p class="lambda-interp">{interp}</p>
        <hr style="border-color:#1E3A5F; margin:0.8rem 0;">
        <p style="font-size:0.72rem; color:#4A6080; margin:0;">
            λ = 1 / (w<sup>T</sup> Σ<sup>-1</sup> (E(r) − r<sub>f</sub>))
            <br>r<sub>f</sub> = {RF:.0%} &nbsp;·&nbsp; basis: {freq_lbl}
        </p>
        <p style="font-size:0.72rem; color:#4A6080; margin-top:0.4rem;">
            Benchmarks: λ 2–4 aggressive · λ 4–8 conservative
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"**Expected Returns & Excess Returns** *(annualised)*")
    st.dataframe(lam_data["asset_df"], use_container_width=True, hide_index=True, height=300)

with c3:
    sens     = lam_data["sens_df"]
    is_base  = sens["r_f"] == f"{RF:.0%}"
    bar_cols = [lam_color if b else "#1E3A5F" for b in is_base]
    fig = go.Figure(go.Bar(
        x=sens["r_f"], y=sens["λ"],
        marker=dict(color=bar_cols, line=dict(color="#080C14", width=1)),
        text=sens["λ"].round(2), textposition="outside",
        textfont=dict(color="#8A9BB0", size=10),
        hovertemplate="r_f = %{x}<br>λ = %{y:.4f}<extra></extra>",
    ))
    layout(fig, "Sensitivity of λ to r_f", height=300)
    fig.update_layout(xaxis_title="Risk-free rate r_f", yaxis_title="λ", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# ALLOKATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Allocation</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

def donut(labels, values, colors, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colors, line=dict(color="#080C14", width=2)),
        textinfo="percent+label", textfont=dict(size=9, color="white"),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    return layout(fig, title, height=300)

with c1:
    labels = [summary_df.loc[t, "name"] for t in weights.index]
    st.plotly_chart(donut(labels, weights.values, COLORS[:len(weights)],
                          "Portfolio Weights"), use_container_width=True)

with c2:
    cls = summary_df.groupby("class")["eur"].sum().reset_index()
    st.plotly_chart(donut(cls["class"].tolist(), cls["eur"].tolist(),
                          ["#3B82F6", "#10B981", "#F59E0B"], "Asset Classes"),
                    use_container_width=True)

with c3:
    reg = summary_df.groupby("region")["eur"].sum().reset_index()
    st.plotly_chart(donut(reg["region"].tolist(), reg["eur"].tolist(),
                          ["#EF4444", "#8B5CF6", "#06B6D4", "#F97316"], "Regions"),
                    use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SEKTOREN + KONZENTRATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Sectors & Concentration</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    sec_df = (summary_df.groupby("sector")["eur"].sum()
              .reset_index().sort_values("eur", ascending=True))
    sec_df["pct"] = sec_df["eur"] / total_value * 100
    fig = go.Figure(go.Bar(
        x=sec_df["pct"], y=sec_df["sector"], orientation="h",
        marker=dict(color=sec_df["pct"],
                    colorscale=[[0, "#1A2D4A"], [1, "#3B82F6"]],
                    line=dict(color="#080C14", width=0.5)),
        text=sec_df["pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="#8A9BB0", size=10),
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    ))
    layout(fig, "Sector Allocation", height=300)
    fig.update_layout(xaxis_title="Weight (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    conc        = summary_df.copy()
    conc["pct"] = conc["eur"] / total_value * 100

    def kpi_row(label, val, warn=None, msg=""):
        color = "#EF4444" if warn and val > warn else "#10B981"
        st.markdown(
            f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.5rem 0.9rem;margin:0.25rem 0;background:#0F1622;
            border-radius:8px;border:1px solid #1A2332;">
            <span style="color:#8A9BB0;font-size:0.83rem;">{label}</span>
            <span style="color:{color};font-weight:500;font-size:1rem;">{val:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        if warn and val > warn:
            st.caption(f"⚠️ {msg}")

    st.markdown("**Concentration Overview**")
    kpi_row("Top-3 Concentration",   conc["pct"].nlargest(3).sum(), 50, "High concentration")
    kpi_row("Germany Exposure",  conc[conc["region"]=="Germany"]["pct"].sum(), 50, "Strong home bias")
    kpi_row("Automotive Sector",     conc[conc["sector"]=="Automotive"]["pct"].sum(), 30, "Elevated sector risk")
    kpi_row("Equity Share",         conc[conc["class"]=="Equity"]["pct"].sum())
    kpi_row("Fixed Income Share",   conc[conc["class"]=="Fixed Income"]["pct"].sum())

st.markdown("---")

# ─────────────────────────────────────────────
# ESG ANALYSE
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">ESG Analyse</p>', unsafe_allow_html=True)

# ESG scores for old portfolio assets (LSEG/Refinitiv legacy ÷ 10, higher = better)
# DAX UCITS ETF  → proxy DAXEX GY Equity  (identical DAX index, same ETF family)
# S&P500 GY Equity → proxy CSPX LN Equity (identical S&P 500 index)
# BO221256 Corp  → German Bund, no issuer-level score available
ESG_OLD = {
    "BMW GY Equity":    5.83,
    "MBG GY Equity":    6.24,
    "DAX UCITS ETF":    7.80,   # proxy: DAXEX GY Equity
    "IWDA LN Equity":   7.113,
    "HIGH LN Equity":   7.30,
    "IBCI IM Equity":   7.50,
    "S&P500 GY Equity": 7.041,  # proxy: CSPX LN Equity
    "BO221256 Corp":    None,
}

ESG_LABEL_COLORS = {
    "AAA": "#10B981", "AA": "#84CC16", "A": "#F59E0B",
    "BBB": "#FB923C", "BB": "#EF4444", "B": "#DC2626",
    "CCC": "#7F1D1D", "n/a": "#374151",
}

def _esg_label(s):
    if s is None: return "n/a"
    if s >= 7.5:  return "AAA"
    if s >= 6.5:  return "AA"
    if s >= 5.5:  return "A"
    if s >= 4.5:  return "BBB"
    if s >= 3.5:  return "BB"
    if s >= 2.5:  return "B"
    return "CCC"

# Weighted average (scored assets only)
esg_pairs = [(s, weights.get(t, 0.)) for t, s in ESG_OLD.items() if s is not None]
esg_tw    = sum(w for _, w in esg_pairs)
esg_avg   = sum(s * w for s, w in esg_pairs) / esg_tw if esg_tw > 0 else None

# ── Score card + bar chart ──────────────────
ev_c1, ev_c2 = st.columns([1, 2])

with ev_c1:
    if esg_avg:
        lbl   = _esg_label(esg_avg)
        color = ESG_LABEL_COLORS.get(lbl, "#64748B")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0F1622 0%,#111827 100%);
             border:1px solid #1E2A3A;border-radius:14px;padding:1.8rem 2rem;
             text-align:center;">
          <div style="font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;
                      color:#3D6090;margin-bottom:.6rem;">
            Weighted Avg ESG Score
          </div>
          <div style="font-size:3.2rem;font-weight:700;color:{color};line-height:1;">
            {esg_avg:.2f}
          </div>
          <div style="font-size:1rem;font-weight:600;color:{color};margin:.3rem 0;">
            {lbl}
          </div>
          <hr style="border-color:#1E3A5F;margin:.8rem 0;">
          <div style="font-size:.7rem;color:#4A6080;line-height:1.6;">
            LSEG/Refinitiv · higher = better<br>
            {esg_tw*100:.0f}% of portfolio scored<br>
            <em>German Bund excluded</em>
          </div>
        </div>""", unsafe_allow_html=True)

with ev_c2:
    tickers_sorted = sorted(
        [t for t in ESG_OLD],
        key=lambda t: ESG_OLD[t] if ESG_OLD[t] is not None else -1
    )
    scores  = [ESG_OLD[t] if ESG_OLD[t] is not None else 0 for t in tickers_sorted]
    bcolors = [ESG_LABEL_COLORS.get(_esg_label(ESG_OLD[t]), "#1E2A3A")
               for t in tickers_sorted]
    w_pct   = [weights.get(t, 0.) * 100 for t in tickers_sorted]
    texts   = [f"{ESG_OLD[t]:.2f} ({_esg_label(ESG_OLD[t])})  w={w:.1f}%"
               if ESG_OLD[t] is not None else "n/a"
               for t, w in zip(tickers_sorted, w_pct)]

    fig = go.Figure(go.Bar(
        y=[t.replace(" Equity","").replace(" Corp","") for t in tickers_sorted],
        x=scores, orientation="h",
        marker=dict(color=bcolors, line=dict(color="#080C14", width=0.5)),
        text=texts, textposition="outside",
        textfont=dict(size=9, color="#8A9BB0"),
        hovertemplate="%{y}<br>ESG: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="ESG Score per Asset  (higher = better)",
                   font=dict(family="DM Serif Display", size=13, color="#C8D0DC")),
        height=max(280, len(tickers_sorted) * 38),
        xaxis=dict(gridcolor="#141E2D", linecolor="#1A2332", zeroline=False,
                   range=[0, 9.8], title="ESG Score"),
        yaxis=dict(gridcolor="#141E2D", linecolor="#1A2332"),
        showlegend=False,
    )
    for xv, lt, lc in [(4.5,"BBB","#FB923C"),(5.5,"A","#F59E0B"),
                        (6.5,"AA","#84CC16"),(7.5,"AAA","#10B981")]:
        fig.add_vline(x=xv, line=dict(color=lc, width=1, dash="dash"),
                      annotation_text=lt, annotation_position="top",
                      annotation_font=dict(color=lc, size=9))
    if esg_avg:
        fig.add_vline(x=esg_avg,
                      line=dict(color="#3B82F6", width=1.5, dash="dot"),
                      annotation_text=f"Avg {esg_avg:.2f}",
                      annotation_position="bottom right",
                      annotation_font=dict(color="#3B82F6", size=10))
    st.plotly_chart(fig, use_container_width=True)

# ── Detail table ───────────────────────────
st.markdown("**ESG Detail**")
esg_tbl = []
for t in sorted(ESG_OLD, key=lambda x: ESG_OLD[x] if ESG_OLD[x] else -1, reverse=True):
    s = ESG_OLD[t]
    esg_tbl.append({
        "Ticker":     t,
        "Name":       ASSET_META.get(t, {}).get("name", t),
        "Weight":     f"{weights.get(t, 0.)*100:.1f}%",
        "ESG Score":  f"{s:.2f}" if s is not None else "n/a",
        "Label":      _esg_label(s),
        "Note":       ("Proxy: DAXEX GY Equity"  if t == "DAX UCITS ETF" else
                       "Proxy: CSPX LN Equity"   if t == "S&P500 GY Equity" else
                       "No issuer-level score"    if s is None else ""),
    })
st.dataframe(pd.DataFrame(esg_tbl), use_container_width=True,
             hide_index=True, height=310)

st.markdown("---")

# ─────────────────────────────────────────────
# DETAIL-TABELLE
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Positions in Detail</p>', unsafe_allow_html=True)

detail = summary_df.copy().reset_index()
detail["Market Value (€)"] = detail["eur"].map(lambda x: f"€ {x:,.0f}")
detail["Weight"]   = detail["weight"].map(lambda x: f"{x*100:.2f}%")
detail = detail.rename(columns={
    "ticker": "Ticker", "name": "Name",
    "class": "Class", "region": "Region", "sector": "Sector",
})
st.dataframe(
    detail[["Ticker", "Name", "Class", "Region", "Sector", "Market Value (€)", "Weight"]],
    use_container_width=True, hide_index=True,
)

st.markdown("---")
st.markdown(
    f'<p style="color:#2A3A50;font-size:0.72rem;text-align:center;">'
    f'Data source: Bloomberg Excel Export &nbsp;·&nbsp; '
    f'Zeitraum: {start_date} – {prices_daily.index[-1].strftime("%d.%m.%Y")} &nbsp;·&nbsp; '
    f'KPI basis: {freq_lbl} &nbsp;·&nbsp; All figures in EUR</p>',
    unsafe_allow_html=True,
)
