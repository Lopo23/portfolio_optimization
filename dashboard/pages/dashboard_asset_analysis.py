"""
asset_explorer.py  –  Asset Explorer
=====================================
Zeigt alle verfügbaren Assets aus New Portfolio und vergleicht sie
konsistent auf Basis MONATLICHER Daten über die letzten 5 Jahre.

Ablegen: dashboard/pages/asset_explorer.py
Starten: streamlit run dashboard/app.py

Pakete: pip install streamlit plotly pandas numpy openpyxl
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_chatgpt import load_new_portfolio


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Asset Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #F8FAFC;
    color: #1E293B;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; color: #0F172A; font-weight: 700; }

[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * {
    font-family: 'Syne', sans-serif !important;
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p {
    color: #CBD5E1 !important;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: .65rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #0F172A !important;
    font-size: 1.3rem !important;
    font-weight: 700;
}

.chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #3B82F6;
    border: 1px solid #BFDBFE;
    background: #EFF6FF;
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: .5rem;
}
.asset-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: .5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
hr {
    border-color: #E2E8F0 !important;
    margin: 1.5rem 0;
}
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
COLORS = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#06B6D4", "#F97316", "#84CC16", "#EC4899", "#14B8A6",
    "#A78BFA", "#FB923C", "#34D399", "#60A5FA", "#FBBF24",
    "#E879F9", "#4ADE80", "#FCA5A5", "#93C5FD", "#6EE7B7",
]

BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F8FAFC",
    font=dict(family="Syne", color="#64748B", size=11),
    margin=dict(l=10, r=10, t=44, b=10),
    xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
    yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
    legend=dict(
        bgcolor="rgba(255,255,255,.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=10, color="#475569"),
    ),
)

def make_fig(title="", h=400, extra=None):
    f = go.Figure()
    layout = {
        **BASE,
        "height": h,
        "title": dict(
            text=title,
            font=dict(family="Syne", size=13, color="#0F172A"),
        ),
    }
    if extra:
        layout.update(extra)
    f.update_layout(**layout)
    return f


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Asset-Daten …")
def get_data():
    return load_new_portfolio(PROJECT_ROOT / "data" / "New Portfolio")


data = get_data()
prices_monthly = data["prices_monthly"]
asset_meta = data["asset_meta"]
asset_classes = data["asset_classes"]
loader_metrics = data["metrics_df"]

ALL_ASSETS = sorted(prices_monthly.columns.tolist())
ASSET_COLORS = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ALL_ASSETS)}

# Konsistentes 5-Jahres-Fenster auf Monatsbasis
end_date = prices_monthly.index[-1]
start_5y = end_date - pd.DateOffset(years=5)
prices_5y = prices_monthly[prices_monthly.index >= start_5y].copy()

if prices_5y.empty:
    st.error("Kein 5-Jahres-Fenster in den Monatsdaten verfügbar.")
    st.stop()


# ─────────────────────────────────────────────
# KENNZAHLEN (5 Jahre, monatlich)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calc_metrics_5y(prices_5y_input: pd.DataFrame, asset_meta_input: dict) -> pd.DataFrame:
    ret = prices_5y_input.pct_change().dropna(how="any")
    rows = []

    for col in ret.columns:
        r = ret[col].dropna()
        if len(r) < 3:
            continue

        cum = (1 + r).cumprod()
        total = cum.iloc[-1] - 1
        ann_r = (1 + total) ** (12 / len(r)) - 1
        ann_v = r.std() * np.sqrt(12)
        sh = ann_r / ann_v if ann_v > 0 else np.nan

        neg = r[r < 0]
        ds = neg.std() * np.sqrt(12) if len(neg) > 0 else np.nan
        so = ann_r / ds if pd.notna(ds) and ds > 0 else np.nan

        roll = cum.cummax()
        mdd = ((cum - roll) / roll).min()

        tail = r[r <= r.quantile(0.05)]
        cvar = tail.mean() if not tail.empty else np.nan

        rows.append({
            "Asset": col,
            "Klasse": asset_meta_input.get(col, {}).get("asset_class", "–"),
            "Ann. Rendite": ann_r,
            "Ann. Vola": ann_v,
            "Sharpe": sh,
            "Sortino": so,
            "Max Drawdown": mdd,
            "CVaR 95%": cvar,
            "Total Return": total,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Klasse", "Ann. Rendite", "Ann. Vola", "Sharpe",
            "Sortino", "Max Drawdown", "CVaR 95%", "Total Return"
        ])

    return pd.DataFrame(rows).set_index("Asset").sort_index()


metrics = calc_metrics_5y(prices_5y, asset_meta)

if metrics.empty:
    st.error("Für das 5-Jahres-Fenster konnten keine Kennzahlen berechnet werden.")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR – FILTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Asset Explorer")
    st.markdown("---")

    st.markdown(
        '<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        'color:#64748B;margin-bottom:.3rem;">Asset-Klassen</p>',
        unsafe_allow_html=True,
    )
    all_classes = sorted(asset_classes.keys())
    sel_classes = st.multiselect(
        "Asset-Klassen",
        all_classes,
        default=all_classes,
        label_visibility="collapsed",
    )

    st.markdown(
        '<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        'color:#64748B;margin:.8rem 0 .3rem;">Asset-Auswahl</p>',
        unsafe_allow_html=True,
    )
    assets_in_class = [
        a for a in ALL_ASSETS
        if asset_meta.get(a, {}).get("asset_class", "Other") in sel_classes
        and a in metrics.index
    ]

    default_assets = assets_in_class[:]
    if len(default_assets) > 12:
        default_assets = default_assets[:12]

    sel_assets = st.multiselect(
        "Assets",
        assets_in_class,
        default=default_assets,
        label_visibility="collapsed",
    )

    if not sel_assets and assets_in_class:
        sel_assets = assets_in_class[:5]

    st.markdown("---")

    st.markdown(
        '<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        'color:#64748B;margin-bottom:.3rem;">Sortierung (Tabelle)</p>',
        unsafe_allow_html=True,
    )
    sort_col = st.selectbox(
        "Sortierung",
        ["Ann. Rendite", "Sharpe", "Ann. Vola", "Max Drawdown", "Total Return"],
        label_visibility="collapsed",
    )
    sort_asc = st.toggle("Aufsteigend", value=False)

    st.markdown("---")

    st.markdown(
        '<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        'color:#64748B;margin-bottom:.3rem;">Normierung</p>',
        unsafe_allow_html=True,
    )
    normalize = st.toggle(
        "Auf 100 normieren",
        value=True,
        help="Alle Assets starten bei 100 – besser vergleichbar",
    )


if not sel_assets:
    st.warning("Keine Assets ausgewählt.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding:.6rem 0 .4rem;">
    <span class="chip">New Portfolio · Asset Explorer</span>
    <h1 style="margin:.3rem 0 0;font-size:1.9rem;font-weight:800;color:#0F172A;">
        Asset Vergleich
    </h1>
    <p style="color:#94A3B8;font-size:.75rem;margin-top:.25rem;
              font-family:'JetBrains Mono',monospace;">
        {len(sel_assets)} Assets ausgewählt &nbsp;·&nbsp;
        5-Jahres-Fenster &nbsp;·&nbsp;
        {prices_5y.index[0].strftime("%b %Y")} – {prices_5y.index[-1].strftime("%b %Y")} &nbsp;·&nbsp;
        Monatsdaten / Monatsrenditen
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI SUMMARY
# ─────────────────────────────────────────────
selected_metrics = metrics.loc[[a for a in sel_assets if a in metrics.index]].copy()

if not selected_metrics.empty:
    best_ret = selected_metrics["Ann. Rendite"].idxmax()
    best_sh = selected_metrics["Sharpe"].idxmax()
    worst_dd = selected_metrics["Max Drawdown"].idxmin()
    low_vol = selected_metrics["Ann. Vola"].idxmin()

    st.markdown('<span class="chip">Highlights (Auswahl)</span>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏆 Beste Rendite", best_ret, f"{selected_metrics.loc[best_ret, 'Ann. Rendite'] * 100:.1f}% p.a.")
    k2.metric("⚡ Bester Sharpe", best_sh, f"{selected_metrics.loc[best_sh, 'Sharpe']:.2f}")
    k3.metric("🛡️ Geringste Vola", low_vol, f"{selected_metrics.loc[low_vol, 'Ann. Vola'] * 100:.1f}% p.a.")
    k4.metric("📉 Max Drawdown", worst_dd, f"{selected_metrics.loc[worst_dd, 'Max Drawdown'] * 100:.1f}%")
    st.markdown("---")


# ─────────────────────────────────────────────
# KUMULIERTE ENTWICKLUNG
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Kumulierte Entwicklung – 5 Jahre</span>', unsafe_allow_html=True)

f = make_fig("", h=480)
for asset in sel_assets:
    if asset not in prices_5y.columns:
        continue

    series = prices_5y[asset].dropna()
    if series.empty:
        continue

    if normalize:
        series = series / series.iloc[0] * 100

    f.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name=asset,
        line=dict(color=ASSET_COLORS[asset], width=1.8),
        hovertemplate=(
            f"{asset}<br>"
            "%{x|%d.%m.%Y}<br>"
            f"{'Index' if normalize else 'Preis'}: " + "%{y:.2f}<extra></extra>"
        ),
    ))

if normalize:
    f.add_hline(y=100, line=dict(color="#CBD5E1", width=1, dash="dash"))
    f.update_layout(yaxis_title="Index (Start = 100)")
else:
    f.update_layout(yaxis_title="Preis")

st.plotly_chart(f, width="stretch")


# ─────────────────────────────────────────────
# DRAWDOWN
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Drawdown – 5 Jahre</span>', unsafe_allow_html=True)

f2 = make_fig("", h=340)
for asset in sel_assets:
    if asset not in prices_5y.columns:
        continue

    s = prices_5y[asset].dropna()
    if s.empty:
        continue

    cum = s / s.iloc[0]
    dd = (cum - cum.cummax()) / cum.cummax() * 100

    f2.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        name=asset,
        line=dict(color=ASSET_COLORS[asset], width=1.5),
        hovertemplate=f"{asset}<br>DD: " + "%{y:.2f}%<extra></extra>",
    ))

f2.update_layout(yaxis_title="Drawdown (%)")
st.plotly_chart(f2, width="stretch")

st.markdown("---")


# ─────────────────────────────────────────────
# RISIKO-RENDITE + KORRELATION
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Risiko-Rendite Diagramm</span>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])

with c1:
    f3 = make_fig("Risiko vs. Rendite (5 Jahre, annualisiert)", h=420)

    for asset in sel_assets:
        if asset not in metrics.index:
            continue

        row = metrics.loc[asset]
        f3.add_trace(go.Scatter(
            x=[row["Ann. Vola"] * 100],
            y=[row["Ann. Rendite"] * 100],
            mode="markers+text",
            name=asset,
            text=[asset],
            textposition="top center",
            textfont=dict(size=9, color=ASSET_COLORS[asset]),
            marker=dict(
                size=12,
                color=ASSET_COLORS[asset],
                line=dict(color="white", width=1.5),
            ),
            hovertemplate=(
                f"{asset}<br>"
                "Rendite: %{y:.2f}%<br>"
                "Vola: %{x:.2f}%<extra></extra>"
            ),
            showlegend=False,
        ))

    f3.update_layout(
        xaxis_title="Ann. Volatilität (%)",
        yaxis_title="Ann. Rendite (%)",
    )
    st.plotly_chart(f3, width="stretch")

with c2:
    ret_sel = prices_5y[[a for a in sel_assets if a in prices_5y.columns]].pct_change().dropna(how="any")

    if len(ret_sel.columns) >= 2:
        corr = ret_sel.corr()
        short = [a[:14] for a in corr.columns]

        f4 = go.Figure(go.Heatmap(
            z=corr.values,
            x=short,
            y=short,
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=8),
            hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>",
        ))
        f4.update_layout(
            **BASE,
            height=420,
            title=dict(
                text="Korrelationsmatrix (5J, monatlich)",
                font=dict(family="Syne", size=13, color="#0F172A"),
            ),
        )
        st.plotly_chart(f4, width="stretch")
    else:
        st.info("Für die Korrelationsmatrix bitte mindestens 2 Assets auswählen.")

st.markdown("---")


# ─────────────────────────────────────────────
# KENNZAHLEN TABELLE
# ─────────────────────────────────────────────
st.markdown(
    '<span class="chip">Kennzahlen – 5 Jahre (annualisiert, monatliche Basis)</span>',
    unsafe_allow_html=True,
)

tbl = (
    metrics.loc[[a for a in sel_assets if a in metrics.index]]
    .sort_values(sort_col, ascending=sort_asc)
    .copy()
)

display = pd.DataFrame(index=tbl.index)
display["Klasse"] = tbl["Klasse"]
display["Ann. Rendite"] = tbl["Ann. Rendite"].map(lambda x: f"{x * 100:+.2f}%")
display["Ann. Vola"] = tbl["Ann. Vola"].map(lambda x: f"{x * 100:.2f}%")
display["Sharpe"] = tbl["Sharpe"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
display["Sortino"] = tbl["Sortino"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
display["Max Drawdown"] = tbl["Max Drawdown"].map(lambda x: f"{x * 100:.2f}%")
display["CVaR 95%"] = tbl["CVaR 95%"].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "–")
display["Total Return"] = tbl["Total Return"].map(lambda x: f"{x * 100:+.1f}%")

st.dataframe(display, width="stretch", height=min(60 + len(display) * 36, 520))

st.markdown("---")


# ─────────────────────────────────────────────
# MONATSRENDITEN HEATMAP
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Monatsrenditen Heatmap</span>', unsafe_allow_html=True)

hm_candidates = [a for a in sel_assets if a in prices_5y.columns]
hm_asset = st.selectbox(
    "Asset für Heatmap",
    hm_candidates,
    label_visibility="visible",
)

if hm_asset:
    r_hm = prices_5y[hm_asset].pct_change().dropna()
    r_hm = r_hm[r_hm.index >= start_5y]

    if not r_hm.empty:
        pivot = pd.DataFrame({
            "Jahr": r_hm.index.year,
            "Monat": r_hm.index.month,
            "Ret": r_hm.values * 100,
        }).pivot(index="Jahr", columns="Monat", values="Ret")

        month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        col_labels = [month_labels[c - 1] for c in pivot.columns]

        text_vals = np.empty(pivot.shape, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                text_vals[i, j] = "" if pd.isna(v) else f"{v:.1f}%"

        f5 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=col_labels,
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Jahr: %{y}<br>Monat: %{x}<br>Rendite: %{z:.2f}%<extra></extra>",
        ))
        f5.update_layout(
            **BASE,
            height=280,
            title=dict(
                text=f"Monatsrenditen: {hm_asset}",
                font=dict(family="Syne", size=13, color="#0F172A"),
            ),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(f5, width="stretch")
    else:
        st.info("Für dieses Asset sind im 5-Jahres-Fenster keine Monatsrenditen verfügbar.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'New Portfolio · Asset Explorer · '
    f'{prices_5y.index[0].strftime("%b %Y")} – {prices_5y.index[-1].strftime("%b %Y")} · '
    f'{len(ALL_ASSETS)} Assets total · Monatsbasis konsistent</p>',
    unsafe_allow_html=True,
)