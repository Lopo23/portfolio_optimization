"""
dashboard_home.py
=================
Streamlit Portfolio-Dashboard.
Legt diese Datei in: dashboard/pages/dashboard_home.py

Starten:
    streamlit run dashboard/pages/dashboard_home.py

Pakete:
    pip install streamlit plotly pandas numpy openpyxl
"""

import sys
from pathlib import Path

# Projektroot in Pythonpfad aufnehmen, damit data_loader importierbar ist
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_portfolio, ASSET_META

# ─────────────────────────────────────────────
# SEITE KONFIGURIEREN
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

/* KPI Cards */
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

/* Section headers */
.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3D6090;
    margin-bottom: 0.3rem;
    font-weight: 500;
}

/* Divider */
hr { border-color: #1A2332 !important; margin: 2rem 0; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Warning/info boxes */
.stAlert { border-radius: 8px; border-left: 3px solid; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0B1120",
    font=dict(family="DM Sans", color="#8A9BB0", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="#141E2D", linecolor="#1A2332", zeroline=False),
    yaxis=dict(gridcolor="#141E2D", linecolor="#1A2332", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1A2332",
                font=dict(size=10, color="#8A9BB0")),
)

COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444",
          "#8B5CF6", "#06B6D4", "#F97316", "#84CC16"]


def apply_layout(fig, title="", height=340):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(
        family="DM Serif Display", size=14, color="#C8D0DC")), height=height)
    return fig


# ─────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Portfoliodaten …")
def get_data():
    return load_portfolio(PROJECT_ROOT / "data" / "Old Portfolio")


data = get_data()

prices        = data["prices"]
returns       = data["returns"]
weights       = data["weights"]
port_returns  = data["port_returns"]
port_cum      = data["port_cum"]
port_drawdown = data["port_drawdown"]
metrics_df    = data["metrics_df"]
summary_df    = data["summary_df"]
total_value   = data["total_value"]
start_date    = data["start_date"]

asset_colors  = {name: COLORS[i % len(COLORS)] for i, name in enumerate(prices.columns)}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 1.6rem 0 0.4rem 0;">
    <p class="section-label">Portfolio Intelligence</p>
    <h1 style="margin:0; font-size:2.1rem;">Old Portfolio — Analyse</h1>
    <p style="color:#4A6080; font-size:0.82rem; margin-top:0.4rem;">
        Zeitraum ab {start_date} &nbsp;·&nbsp; {len(prices.columns)} Assets &nbsp;·&nbsp;
        {prices.index[0].strftime("%d.%m.%Y")} – {prices.index[-1].strftime("%d.%m.%Y")}
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# KPI ZEILE
# ─────────────────────────────────────────────
port_m = metrics_df.loc["PORTFOLIO"]

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Portfoliowert",        f"€ {total_value/1e6:.1f} Mio.")
k2.metric("Ann. Rendite",         f"{port_m['ann_return']:.2f} %")
k3.metric("Ann. Volatilität",     f"{port_m['ann_vol']:.2f} %")
k4.metric("Sharpe Ratio",         f"{port_m['sharpe']:.2f}")
k5.metric("Max Drawdown",         f"{port_m['max_drawdown']:.2f} %")
k6.metric("CVaR 95 %",            f"{port_m['cvar_95']:.2f} %")

st.markdown("---")

# ─────────────────────────────────────────────
# ZEILE 1 – Performance & Drawdown
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    fig = go.Figure()
    for name in prices.columns:
        cum = (1 + returns[name].dropna()).cumprod()
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
    apply_layout(fig, "Kumulierte Renditen (Basis = 1)", height=360)
    st.plotly_chart(fig, width='stretch')

with c2:
    fig = go.Figure()
    for name in prices.columns:
        r   = returns[name].dropna()
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name,
            line=dict(color=asset_colors[name], width=1.2),
            hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    port_dd_pct = port_drawdown * 100
    fig.add_trace(go.Scatter(
        x=port_dd_pct.index, y=port_dd_pct.values, name="Portfolio",
        line=dict(color="white", width=2.4, dash="dot"),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
    ))
    apply_layout(fig, "Drawdown (%)", height=360)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# ZEILE 2 – Allokation
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Allokation</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    # Donut – Asset-Gewichte
    fig = go.Figure(go.Pie(
        labels=[summary_df.loc[t, "name"] for t in weights.index],
        values=weights.values,
        hole=0.58,
        marker=dict(colors=COLORS[:len(weights)], line=dict(color="#080C14", width=2)),
        textinfo="percent",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    apply_layout(fig, "Portfoliogewichte", height=320)
    st.plotly_chart(fig, width='stretch')

with c2:
    # Donut – Asset-Klasse
    class_df = (summary_df.groupby("class")["eur"].sum()
                .reset_index().rename(columns={"class": "Klasse", "eur": "EUR"}))
    fig = go.Figure(go.Pie(
        labels=class_df["Klasse"], values=class_df["EUR"],
        hole=0.58,
        marker=dict(colors=["#3B82F6", "#10B981", "#F59E0B"],
                    line=dict(color="#080C14", width=2)),
        textinfo="percent+label",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    apply_layout(fig, "Asset-Klassen", height=320)
    st.plotly_chart(fig, width='stretch')

with c3:
    # Donut – Region
    reg_df = (summary_df.groupby("region")["eur"].sum()
              .reset_index().rename(columns={"region": "Region", "eur": "EUR"}))
    fig = go.Figure(go.Pie(
        labels=reg_df["Region"], values=reg_df["EUR"],
        hole=0.58,
        marker=dict(colors=["#EF4444", "#8B5CF6", "#06B6D4", "#F97316"],
                    line=dict(color="#080C14", width=2)),
        textinfo="percent+label",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    apply_layout(fig, "Regionen", height=320)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# ZEILE 3 – Kennzahlen-Tabelle + Korrelation
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Risiko & Korrelation</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1.1, 1])

with c1:
    display = (metrics_df
               .drop(index="PORTFOLIO", errors="ignore")
               .reset_index()
               .rename(columns={
                   "asset":       "Ticker",
                   "name":        "Name",
                   "ann_return":  "Rendite %",
                   "ann_vol":     "Vola %",
                   "sharpe":      "Sharpe",
                   "sortino":     "Sortino",
                   "max_drawdown":"Max DD %",
                   "cvar_95":     "CVaR 95%",
                   "total_return":"Total %",
               }))
    # Portfolio-Zeile unten anhängen
    port_row = metrics_df.loc[["PORTFOLIO"]].reset_index().rename(columns={
        "asset": "Ticker", "name": "Name",
        "ann_return": "Rendite %", "ann_vol": "Vola %",
        "sharpe": "Sharpe", "sortino": "Sortino",
        "max_drawdown": "Max DD %", "cvar_95": "CVaR 95%", "total_return": "Total %",
    })
    display = pd.concat([display, port_row], ignore_index=True)

    st.markdown("**Kennzahlen je Asset**")
    st.dataframe(
        display[["Name", "Rendite %", "Vola %", "Sharpe", "Max DD %", "CVaR 95%"]],
        width='stretch',
        hide_index=True,
        height=320,
    )

with c2:
    corr  = returns.corr()
    # Kurze Labels
    short = [c.replace(" Equity", "").replace(" Corp", "") for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=short, y=short,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} / %{y}<br>Korrelation: %{z:.2f}<extra></extra>",
    ))
    apply_layout(fig, "Korrelationsmatrix", height=360)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ─────────────────────────────────────────────
# ZEILE 4 – Sektor + Konzentration
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Sektoren & Konzentration</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    sec_df = (summary_df.groupby("sector")["eur"].sum()
              .reset_index()
              .sort_values("eur", ascending=True)
              .rename(columns={"sector": "Sektor", "eur": "EUR"}))
    sec_df["Pct"] = sec_df["EUR"] / total_value * 100

    fig = go.Figure(go.Bar(
        x=sec_df["Pct"], y=sec_df["Sektor"],
        orientation="h",
        marker=dict(
            color=sec_df["Pct"],
            colorscale=[[0, "#1A2D4A"], [1, "#3B82F6"]],
            line=dict(color="#080C14", width=0.5),
        ),
        text=sec_df["Pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="#8A9BB0", size=10),
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    ))
    apply_layout(fig, "Sektorallokation", height=320)
    fig.update_layout(xaxis_title="Gewicht (%)", yaxis_title="")
    st.plotly_chart(fig, width='stretch')

with c2:
    # Konzentrations-Analyse – sortiert nach Gewicht
    conc_df = summary_df.copy()
    conc_df["pct"] = conc_df["eur"] / total_value * 100
    conc_df = conc_df.sort_values("pct", ascending=False)

    top3_pct     = conc_df["pct"].nlargest(3).sum()
    auto_pct     = conc_df[conc_df["sector"] == "Automotive"]["pct"].sum()
    germany_pct  = conc_df[conc_df["region"] == "Germany"]["pct"].sum()
    equity_pct   = conc_df[conc_df["class"] == "Equity"]["pct"].sum()
    fi_pct       = conc_df[conc_df["class"] == "Fixed Income"]["pct"].sum()
    single_pct   = conc_df[conc_df["sector"] == "Automotive"]["pct"].sum()

    st.markdown("**Konzentrationsübersicht**")

    def kpi_row(label, val, warn_threshold=None, warn_msg=""):
        color = "#EF4444" if warn_threshold and val > warn_threshold else "#10B981"
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:center;
            padding:0.55rem 0.8rem; margin:0.3rem 0;
            background:#0F1622; border-radius:8px; border:1px solid #1A2332;">
            <span style="color:#8A9BB0; font-size:0.83rem;">{label}</span>
            <span style="color:{color}; font-weight:500; font-size:1rem;">{val:.1f}%</span>
            </div>""",
            unsafe_allow_html=True,
        )
        if warn_threshold and val > warn_threshold:
            st.caption(f"⚠️ {warn_msg}")

    kpi_row("Top-3 Konzentration",      top3_pct,    50, "Hohe Konzentration in wenigen Positionen")
    kpi_row("Deutschland-Exposure",     germany_pct, 50, "Starker Home Bias")
    kpi_row("Automotive-Sektor",        auto_pct,    30, "Erhöhtes Sektorrisiko")
    kpi_row("Equity-Anteil",            equity_pct)
    kpi_row("Fixed Income-Anteil",      fi_pct)

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
    width='stretch',
    hide_index=True,
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#2A3A50; font-size:0.72rem; text-align:center;">'
    f'Datenquelle: Bloomberg Excel Export &nbsp;·&nbsp; '
    f'Zeitraum: {start_date} – {prices.index[-1].strftime("%d.%m.%Y")} &nbsp;·&nbsp; '
    f'Alle Angaben in EUR</p>',
    unsafe_allow_html=True,
)
