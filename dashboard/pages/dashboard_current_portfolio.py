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

RF = 0.02   # risikofreier Zinssatz für Lambda


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

prices_daily = data["prices"]["daily"]
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
    """Berechnet Kennzahlen für gegebene Renditefrequenz."""
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
    """
    λ = 1 / (w^T · Σ⁻¹ · (E(r) - r_f))

    E(r)  = geometrisch annualisierte Rendite je Asset
    Σ     = annualisierte Kovarianzmatrix
    rf    = jährlicher risikofreier Zinssatz
    """
    scale = 252 if freq == "daily" else 12

    cols = weights.index.tolist()
    r    = returns[cols].dropna()
    w    = weights.reindex(cols).dropna().values

    # E(r) geometrisch annualisiert
    total_r = (1 + r).prod() - 1
    ann_r   = (1 + total_r) ** (scale / len(r)) - 1
    er      = ann_r.values

    # Kovarianzmatrix annualisiert
    cov_ann = r.cov().values * scale

    # Lambda
    excess  = er - rf
    try:
        cov_inv = np.linalg.inv(cov_ann)
        denom   = float(w @ cov_inv @ excess)
        lam     = 1.0 / denom if abs(denom) > 1e-12 else np.nan
    except np.linalg.LinAlgError:
        lam, denom = np.nan, np.nan

    # Sensitivität
    sens = []
    for rf_t in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        ex = er - rf_t
        try:
            d  = float(w @ cov_inv @ ex)
            l  = round(1.0 / d, 4) if abs(d) > 1e-12 else np.nan
        except Exception:
            l = np.nan
        sens.append({"r_f": f"{rf_t:.0%}", "λ": l})

    # Per-Asset Excess Return Tabelle
    asset_df = pd.DataFrame({
        "Asset":    cols,
        "E(r)":     [f"{x:.2%}" for x in er],
        "E(r)-r_f": [f"{x:.2%}" for x in excess],
        "Gewicht":  [f"{v:.2%}" for v in w],
    })

    return {
        "lambda":   lam,
        "denom":    denom,
        "er":       er,
        "excess":   excess,
        "asset_df": asset_df,
        "sens_df":  pd.DataFrame(sens),
    }


def interp_lambda(lam: float) -> tuple[str, str]:
    """Gibt (Beschreibung, Farbe) zurück."""
    if np.isnan(lam):
        return "Nicht berechenbar", "#EF4444"
    if lam < 0:
        return "Negativ – impliziert irrationales Verhalten", "#EF4444"
    if lam < 1:
        return "Sehr geringe Risikoaversion (nahezu risikoneutral)", "#F59E0B"
    if lam < 3:
        return "Geringe bis moderate Risikoaversion", "#84CC16"
    if lam < 6:
        return "Moderate Risikoaversion – typisch für institutionelle Anleger", "#10B981"
    if lam < 10:
        return "Hohe Risikoaversion", "#F59E0B"
    return "Sehr hohe Risikoaversion", "#EF4444"


# ─────────────────────────────────────────────
# HEADER + FREQUENZ-SCHALTER
# ─────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])

with h_left:
    st.markdown(f"""
    <div style="padding: 1.4rem 0 0.2rem 0;">
        <p class="section-label">Portfolio Intelligence</p>
        <h1 style="margin:0; font-size:2.1rem;">Old Portfolio — Analyse</h1>
        <p style="color:#4A6080; font-size:0.82rem; margin-top:0.3rem;">
            ab {start_date} &nbsp;·&nbsp; {len(prices_daily.columns)} Assets &nbsp;·&nbsp;
            {prices_daily.index[0].strftime("%d.%m.%Y")} – {prices_daily.index[-1].strftime("%d.%m.%Y")}
        </p>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    st.markdown("<div style='padding-top:1.8rem;'>", unsafe_allow_html=True)
    use_monthly = st.toggle("📅  Monatsrenditen", value=True,
                            help="Ein = Monatsrenditen (×12) · Aus = Tagesrenditen (×252)")
    st.markdown("</div>", unsafe_allow_html=True)

# Renditen & Frequenz je nach Schalter
if use_monthly:
    returns  = prices_monthly.pct_change().dropna()
    freq     = "monthly"
    freq_lbl = "Monatsrenditen"
    scale    = 12
else:
    returns  = prices_daily.pct_change().dropna()
    freq     = "daily"
    freq_lbl = "Tagesrenditen"
    scale    = 252

# Metriken neu berechnen
metrics_df = compute_metrics(returns, weights, freq)
port_m     = metrics_df.loc["PORTFOLIO"]

# Portfolio-Zeitreihen
w_norm       = weights.reindex(returns.columns).dropna()
w_norm       = w_norm / w_norm.sum()
port_returns = returns[w_norm.index].mul(w_norm, axis=1).sum(axis=1)
port_cum     = (1 + port_returns).cumprod()
port_dd      = (port_cum - port_cum.cummax()) / port_cum.cummax()

st.caption(f"KPIs berechnet auf Basis von **{freq_lbl}** · r_f = {RF:.0%} · Annualisierung ×{scale}")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI ZEILE
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("Portfoliowert",     f"€ {total_value/1e6:.1f} Mio.")
k2.metric("Ann. Rendite",      f"{port_m['ann_return']:.2f} %")
k3.metric("Ann. Volatilität",  f"{port_m['ann_vol']:.2f} %")
k4.metric("Sharpe Ratio",      f"{port_m['sharpe']:.2f}")
k5.metric("Sortino Ratio",     f"{port_m['sortino']:.2f}")
k6.metric("Max Drawdown",      f"{port_m['max_drawdown']:.2f} %")
k7.metric("CVaR 95 %",         f"{port_m['cvar_95']:.2f} %")

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
    layout(fig, "Kumulierte Renditen (Basis = 1)", height=360)
    st.plotly_chart(fig, width='stretch')

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
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# KENNZAHLEN + KORRELATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Risiko & Korrelation</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1.1, 1])

with c1:
    disp = (metrics_df.reset_index()
            .rename(columns={
                "asset": "Ticker", "name": "Name",
                "ann_return": "Rendite %", "ann_vol": "Vola %",
                "sharpe": "Sharpe", "sortino": "Sortino",
                "max_drawdown": "Max DD %", "cvar_95": "CVaR 95%",
                "total_return": "Total %",
            }))
    # Portfolio ans Ende
    mask  = disp["Ticker"] == "PORTFOLIO"
    disp  = pd.concat([disp[~mask], disp[mask]], ignore_index=True)
    st.markdown(f"**Kennzahlen je Asset** *(Basis: {freq_lbl})*")
    st.dataframe(
        disp[["Name", "Rendite %", "Vola %", "Sharpe", "Sortino", "Max DD %", "CVaR 95%"]],
        width='stretch', hide_index=True, height=340,
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
    layout(fig, f"Korrelationsmatrix ({freq_lbl})", height=380)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# LAMBDA – RISIKOAVERSIONSKOEFFIZIENT
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Risikoaversion</p>', unsafe_allow_html=True)

lam_data  = compute_lambda(returns, weights, freq, RF)
lam       = lam_data["lambda"]
interp, lam_color = interp_lambda(lam)

c1, c2, c3 = st.columns([1, 1.3, 1.2])

with c1:
    lam_str = f"{lam:.3f}" if not np.isnan(lam) else "n/a"
    st.markdown(f"""
    <div class="lambda-box">
        <p style="font-size:0.7rem; letter-spacing:0.1em; color:#3D6090; text-transform:uppercase; margin:0;">
            Risikoaversionskoeffizient
        </p>
        <div class="lambda-value" style="color:{lam_color};">{lam_str}</div>
        <p class="lambda-interp">{interp}</p>
        <hr style="border-color:#1E3A5F; margin:0.8rem 0;">
        <p style="font-size:0.72rem; color:#4A6080; margin:0;">
            λ = 1 / (w<sup>T</sup> Σ<sup>-1</sup> (E(r) − r<sub>f</sub>))
            <br>r<sub>f</sub> = {RF:.0%} &nbsp;·&nbsp; Basis: {freq_lbl}
        </p>
        <p style="font-size:0.72rem; color:#4A6080; margin-top:0.4rem;">
            Richtwerte: λ 2–4 aggressiv · λ 4–8 konservativ
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"**Erwartete Renditen & Excess Returns** *(annualisiert)*")
    st.dataframe(lam_data["asset_df"], width='stretch', hide_index=True, height=300)

with c3:
    # Sensitivitäts-Balken
    sens     = lam_data["sens_df"]
    is_base  = sens["r_f"] == f"{RF:.0%}"
    bar_cols = [lam_color if b else "#1E3A5F" for b in is_base]

    fig = go.Figure(go.Bar(
        x=sens["r_f"],
        y=sens["λ"],
        marker=dict(color=bar_cols, line=dict(color="#080C14", width=1)),
        text=sens["λ"].round(2),
        textposition="outside",
        textfont=dict(color="#8A9BB0", size=10),
        hovertemplate="r_f = %{x}<br>λ = %{y:.4f}<extra></extra>",
    ))
    layout(fig, "Sensitivität λ nach r_f", height=300)
    fig.update_layout(
        xaxis_title="Risikofreier Zinssatz r_f",
        yaxis_title="λ",
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# ALLOKATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Allokation</p>', unsafe_allow_html=True)

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
                          "Portfoliogewichte"), width='stretch')

with c2:
    cls = summary_df.groupby("class")["eur"].sum().reset_index()
    st.plotly_chart(donut(cls["class"].tolist(), cls["eur"].tolist(),
                          ["#3B82F6", "#10B981", "#F59E0B"], "Asset-Klassen"),
                    width='stretch')

with c3:
    reg = summary_df.groupby("region")["eur"].sum().reset_index()
    st.plotly_chart(donut(reg["region"].tolist(), reg["eur"].tolist(),
                          ["#EF4444", "#8B5CF6", "#06B6D4", "#F97316"], "Regionen"),
                    width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# SEKTOREN + KONZENTRATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Sektoren & Konzentration</p>', unsafe_allow_html=True)

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
    layout(fig, "Sektorallokation", height=300)
    fig.update_layout(xaxis_title="Gewicht (%)", yaxis_title="")
    st.plotly_chart(fig, width='stretch')

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

    st.markdown("**Konzentrationsübersicht**")
    kpi_row("Top-3 Konzentration",   conc["pct"].nlargest(3).sum(), 50, "Hohe Konzentration")
    kpi_row("Deutschland-Exposure",  conc[conc["region"]=="Germany"]["pct"].sum(), 50, "Starker Home Bias")
    kpi_row("Automotive-Sektor",     conc[conc["sector"]=="Automotive"]["pct"].sum(), 30, "Erhöhtes Sektorrisiko")
    kpi_row("Equity-Anteil",         conc[conc["class"]=="Equity"]["pct"].sum())
    kpi_row("Fixed Income-Anteil",   conc[conc["class"]=="Fixed Income"]["pct"].sum())

st.markdown("---")

# ─────────────────────────────────────────────
# DETAIL-TABELLE
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Positionen im Detail</p>', unsafe_allow_html=True)

detail = summary_df.copy().reset_index()
detail["Marktwert"] = detail["eur"].map(lambda x: f"€ {x:,.0f}")
detail["Gewicht"]   = detail["weight"].map(lambda x: f"{x*100:.2f}%")
detail = detail.rename(columns={
    "ticker": "Ticker", "name": "Name",
    "class": "Klasse", "region": "Region", "sector": "Sektor",
})
st.dataframe(
    detail[["Ticker", "Name", "Klasse", "Region", "Sektor", "Marktwert", "Gewicht"]],
    width='stretch', hide_index=True,
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#2A3A50;font-size:0.72rem;text-align:center;">'
    f'Datenquelle: Bloomberg Excel Export &nbsp;·&nbsp; '
    f'Zeitraum: {start_date} – {prices_daily.index[-1].strftime("%d.%m.%Y")} &nbsp;·&nbsp; '
    f'KPI-Basis: {freq_lbl} &nbsp;·&nbsp; Alle Angaben in EUR</p>',
    unsafe_allow_html=True,
)
