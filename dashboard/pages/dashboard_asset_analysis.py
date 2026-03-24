"""
asset_explorer.py  –  Asset Explorer
=====================================
Überblick über alle Assets aus dem New Portfolio.
Kein Optimierer – nur Visualisierung und Vergleich.

Ablegen: dashboard/pages/asset_explorer.py
Starten: streamlit run dashboard/app.py
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

from new_portfolio_loader import load_new_portfolio
from asset_metadata import get_df as get_meta_df, esg_label


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Asset Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');
html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#F8FAFC; color:#1E293B; }
h1,h2,h3 { font-family:'Syne',sans-serif; color:#0F172A; font-weight:700; }
[data-testid="stSidebar"] { background:#0F172A !important; }
[data-testid="stSidebar"] * { font-family:'Syne',sans-serif !important; color:#94A3B8 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] .stMarkdown p { color:#CBD5E1 !important; }
[data-testid="metric-container"] {
    background:white; border:1px solid #E2E8F0; border-radius:10px;
    padding:1rem 1.2rem; box-shadow:0 1px 3px rgba(0,0,0,.06);
}
[data-testid="metric-container"] label {
    color:#64748B !important; font-size:.65rem !important;
    letter-spacing:.1em; text-transform:uppercase;
    font-family:'JetBrains Mono',monospace !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color:#0F172A !important; font-size:1.3rem !important; font-weight:700;
}
.chip { display:inline-block; font-family:'JetBrains Mono',monospace;
        font-size:.58rem; letter-spacing:.15em; text-transform:uppercase;
        color:#3B82F6; border:1px solid #BFDBFE; background:#EFF6FF;
        border-radius:4px; padding:2px 8px; margin-bottom:.5rem; }
hr { border-color:#E2E8F0 !important; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
COLORS = [
    "#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6",
    "#06B6D4","#F97316","#84CC16","#EC4899","#14B8A6",
    "#A78BFA","#FB923C","#34D399","#60A5FA","#FBBF24",
    "#E879F9","#4ADE80","#FCA5A5","#93C5FD","#6EE7B7",
]
MONTHS = ["Jan","Feb","Mär","Apr","Mai","Jun",
          "Jul","Aug","Sep","Okt","Nov","Dez"]

def _layout(title="", h=420, extra=None):
    d = dict(
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font=dict(family="Syne", color="#64748B", size=11),
        margin=dict(l=10, r=10, t=44, b=10), height=h,
        title=dict(text=title, font=dict(family="Syne", size=13, color="#0F172A")),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#E2E8F0",
                    borderwidth=1, font=dict(size=10, color="#475569")),
    )
    if extra:
        d.update(extra)
    return d

def make_fig(title="", h=420, extra=None):
    f = go.Figure()
    f.update_layout(**_layout(title, h, extra))
    return f


# ─────────────────────────────────────────────
# DATA  (neuer Loader → nur prices_monthly)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Asset-Daten …")
def _load():
    return load_new_portfolio(PROJECT_ROOT / "data" / "New Portfolio")

data           = _load()
PM             = data["prices_monthly"]   # monatliche Preismatrix
asset_meta_ldr = data["asset_meta"]
asset_classes  = data["asset_classes"]
META_DF        = get_meta_df()

if PM.empty:
    st.error("Keine Preisdaten verfügbar.")
    st.stop()

ALL_ASSETS   = sorted(PM.columns.tolist())
ALL_CLASSES  = sorted(asset_classes.keys())
A_COLORS     = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ALL_ASSETS)}

def meta(a: str) -> dict:
    """Kombiniert Loader-Meta + asset_metadata.py.
    Unterstützt beide Feldnamen-Konventionen:
      - Deutsch (alt):  klasse / sektor
      - Englisch (neu): asset_class / sector
    """
    m = dict(asset_meta_ldr.get(a, {}))
    if a in META_DF.index:
        row = META_DF.loc[a]
        m["klasse"]    = (row.get("klasse") or row.get("asset_class")
                          or m.get("asset_class", "–"))
        m["sektor"]    = row.get("sektor") or row.get("sector") or "–"
        m["region"]    = row.get("region",     "–")
        m["esg_score"] = row.get("esg_score",  None)
        m["name"]      = row.get("name",       a)
        m["isin"]      = row.get("isin",       None)
        m["asset_class"] = m["klasse"]
    else:
        m.setdefault("klasse",    m.get("asset_class", "–"))
        m.setdefault("sektor",    m.get("sector", "–"))
        m.setdefault("region",    "–")
        m.setdefault("esg_score", None)
        m.setdefault("name",      a)
        m.setdefault("isin",      None)
        m.setdefault("asset_class", m.get("klasse", "–"))
    return m

# 5-Jahres-Fenster (monatlich)
end_date  = PM.index[-1]
start_5y  = end_date - pd.DateOffset(years=5)
PM_5y     = PM[PM.index >= start_5y]


# ─────────────────────────────────────────────
# KENNZAHLEN (5 Jahre, monatlich)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calc_metrics():
    ret  = PM_5y.pct_change().dropna()
    rows = []
    for col in ret.columns:
        r = ret[col].dropna()
        if len(r) < 3:
            continue
        cum   = (1 + r).cumprod()
        total = cum.iloc[-1] - 1
        ann_r = (1 + total) ** (12 / len(r)) - 1
        ann_v = r.std() * np.sqrt(12)
        sh    = ann_r / ann_v if ann_v > 0 else np.nan
        neg   = r[r < 0]
        ds    = neg.std() * np.sqrt(12) if len(neg) > 0 else np.nan
        so    = ann_r / ds if ds and ds > 0 else np.nan
        roll  = cum.cummax()
        mdd   = ((cum - roll) / roll).min()
        cvar  = r[r <= r.quantile(0.05)].mean()
        m     = meta(col)
        rows.append({
            "Asset":        col,
            "Name":         m.get("name",      col),
            "ISIN":         m.get("isin",      None),
            "Klasse":       m.get("klasse",    "–"),
            "Sektor":       m.get("sektor",    "–"),
            "Region":       m.get("region",    "–"),
            "ESG":          m.get("esg_score", None),
            "Ann. Rendite": ann_r,
            "Ann. Vola":    ann_v,
            "Sharpe":       sh,
            "Sortino":      so,
            "Max Drawdown": mdd,
            "CVaR 95%":     cvar,
            "Total Return": total,
        })
    return pd.DataFrame(rows).set_index("Asset")

metrics = calc_metrics()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def _lbl(t):
    st.markdown(
        f'<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        f'color:#94A3B8;margin-bottom:.3rem;">{t}</p>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## 🔍 Asset Explorer")
    st.markdown("---")

    _lbl("Asset-Klassen")
    sel_classes = st.multiselect(
        "kl", ALL_CLASSES, default=ALL_CLASSES, label_visibility="collapsed"
    )

    assets_in_class = [
        a for a in ALL_ASSETS
        if asset_meta_ldr.get(a, {}).get("asset_class", "Other") in sel_classes
        and a in metrics.index
    ]

    _lbl("Assets")
    sel_assets = st.multiselect(
        "as", assets_in_class, default=assets_in_class, label_visibility="collapsed"
    )
    if not sel_assets:
        sel_assets = assets_in_class[:5]

    st.markdown("---")

    _lbl("Sortierung (Tabelle)")
    sort_col = st.selectbox(
        "sc", ["Ann. Rendite","Sharpe","Sortino","Ann. Vola","Max Drawdown","Total Return"],
        label_visibility="collapsed",
    )
    sort_asc = st.toggle("Aufsteigend", value=False)

    st.markdown("---")

    _lbl("Darstellung")
    normalize = st.toggle("Auf 100 normieren", value=True,
                           help="Alle Assets starten bei 100")


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
        {len(sel_assets)} Assets &nbsp;·&nbsp;
        5-Jahres-Fenster &nbsp;·&nbsp;
        {PM_5y.index[0].strftime("%b %Y")} – {PM_5y.index[-1].strftime("%b %Y")} &nbsp;·&nbsp;
        Monatsdaten
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if not sel_assets:
    st.warning("Bitte mindestens ein Asset auswählen.")
    st.stop()


# ─────────────────────────────────────────────
# KPI HIGHLIGHTS
# ─────────────────────────────────────────────
m_sel = metrics.loc[[a for a in sel_assets if a in metrics.index]]

if not m_sel.empty:
    st.markdown('<span class="chip">Highlights</span>', unsafe_allow_html=True)
    best_ret  = m_sel["Ann. Rendite"].idxmax()
    best_sh   = m_sel["Sharpe"].idxmax()
    worst_dd  = m_sel["Max Drawdown"].idxmin()
    low_vol   = m_sel["Ann. Vola"].idxmin()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("🏆 Beste Rendite",  best_ret,  f"{m_sel.loc[best_ret,'Ann. Rendite']*100:.1f}% p.a.")
    k2.metric("⚡ Bester Sharpe",  best_sh,   f"{m_sel.loc[best_sh,'Sharpe']:.2f}")
    k3.metric("🛡️ Geringste Vola", low_vol,   f"{m_sel.loc[low_vol,'Ann. Vola']*100:.1f}% p.a.")
    k4.metric("📉 Max Drawdown",   worst_dd,  f"{m_sel.loc[worst_dd,'Max Drawdown']*100:.1f}%")
    st.markdown("---")


# ─────────────────────────────────────────────
# KUMULIERTE RENDITEN
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Kumulierte Renditen – 5 Jahre</span>', unsafe_allow_html=True)

f = make_fig(h=480)
for a in sel_assets:
    if a not in PM_5y.columns:
        continue
    s = PM_5y[a].dropna()
    if s.empty:
        continue
    y = s / s.iloc[0] * 100 if normalize else s
    f.add_trace(go.Scatter(
        x=s.index, y=y.values, name=a,
        line=dict(color=A_COLORS[a], width=1.8),
        hovertemplate=f"{a}<br>%{{x|%b %Y}}<br>{'Index' if normalize else 'Preis'}: %{{y:.2f}}<extra></extra>",
    ))

if normalize:
    f.add_hline(y=100, line=dict(color="#CBD5E1", width=1, dash="dash"))
    f.update_layout(yaxis_title="Index (Start = 100)")
else:
    f.update_layout(yaxis_title="Preis")

st.plotly_chart(f, use_container_width=True)


# ─────────────────────────────────────────────
# DRAWDOWN
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Drawdown – 5 Jahre</span>', unsafe_allow_html=True)

f2 = make_fig(h=320)
for a in sel_assets:
    if a not in PM_5y.columns:
        continue
    s   = PM_5y[a].dropna()
    if s.empty:
        continue
    cum = s / s.iloc[0]
    dd  = (cum - cum.cummax()) / cum.cummax() * 100
    f2.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name=a,
        line=dict(color=A_COLORS[a], width=1.5),
        hovertemplate=f"{a}<br>DD: %{{y:.2f}}%<extra></extra>",
    ))
f2.update_layout(yaxis_title="Drawdown (%)")
st.plotly_chart(f2, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────
# RISIKO-RENDITE + KORRELATION
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Risiko-Rendite & Korrelation</span>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])

with c1:
    f3 = make_fig("Risiko vs. Rendite (5 Jahre, annualisiert)", h=440)
    for a in sel_assets:
        if a not in metrics.index:
            continue
        row = metrics.loc[a]
        f3.add_trace(go.Scatter(
            x=[row["Ann. Vola"] * 100],
            y=[row["Ann. Rendite"] * 100],
            mode="markers+text",
            name=a,
            text=[a],
            textposition="top center",
            textfont=dict(size=9, color=A_COLORS[a]),
            marker=dict(size=12, color=A_COLORS[a], line=dict(color="white", width=1.5)),
            hovertemplate=(f"{a}<br>Rendite: %{{y:.2f}}%<br>Vola: %{{x:.2f}}%<extra></extra>"),
            showlegend=False,
        ))
    f3.update_layout(xaxis_title="Ann. Volatilität (%)", yaxis_title="Ann. Rendite (%)")
    st.plotly_chart(f3, use_container_width=True)

with c2:
    avail = [a for a in sel_assets if a in PM_5y.columns]
    if len(avail) >= 2:
        ret_sel = PM_5y[avail].pct_change().dropna()
        corr    = ret_sel.corr()
        short   = [a[:14] for a in corr.columns]
        f4 = go.Figure(go.Heatmap(
            z=corr.values, x=short, y=short,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=8),
            hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>",
        ))
        f4.update_layout(**_layout("Korrelationsmatrix (5J, monatlich)", h=440))
        st.plotly_chart(f4, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────
# KENNZAHLEN TABELLE
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Kennzahlen – 5 Jahre (annualisiert)</span>',
            unsafe_allow_html=True)

tbl = (metrics
       .loc[[a for a in sel_assets if a in metrics.index]]
       .sort_values(sort_col, ascending=sort_asc)
       .copy())

disp = pd.DataFrame(index=tbl.index)
disp["Name"]         = tbl["Name"]
disp["ISIN"]         = tbl["ISIN"].fillna("–")
disp["Klasse"]       = tbl["Klasse"]
disp["Sektor"]       = tbl["Sektor"]
disp["Region"]       = tbl["Region"]
disp["ESG"]          = tbl["ESG"].apply(lambda x: f"{x:.2f} ({esg_label(x)})" if pd.notna(x) and x is not None else "–")
disp["Ann. Rendite"] = tbl["Ann. Rendite"].map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "–")
disp["Ann. Vola"]    = tbl["Ann. Vola"].map(lambda x: f"{x*100:.2f}%"     if pd.notna(x) else "–")
disp["Sharpe"]       = tbl["Sharpe"].map(lambda x: f"{x:.3f}"             if pd.notna(x) else "–")
disp["Sortino"]      = tbl["Sortino"].map(lambda x: f"{x:.3f}"            if pd.notna(x) else "–")
disp["Max DD"]       = tbl["Max Drawdown"].map(lambda x: f"{x*100:.2f}%"  if pd.notna(x) else "–")
disp["CVaR 95%"]     = tbl["CVaR 95%"].map(lambda x: f"{x*100:.2f}%"     if pd.notna(x) else "–")
disp["Total Return"] = tbl["Total Return"].map(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "–")

st.dataframe(disp, use_container_width=True, height=min(60 + len(disp) * 36, 540))

st.markdown("---")


# ─────────────────────────────────────────────
# MONATSRENDITEN HEATMAP
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Monatsrenditen Heatmap</span>', unsafe_allow_html=True)

hm_options = [a for a in sel_assets if a in PM_5y.columns]
if hm_options:
    hm_asset = st.selectbox("Asset für Heatmap", hm_options)

    r_hm  = PM_5y[hm_asset].pct_change().dropna()
    pivot = pd.DataFrame({
        "Jahr":  r_hm.index.year,
        "Monat": r_hm.index.month,
        "Ret":   r_hm.values * 100,
    }).pivot(index="Jahr", columns="Monat", values="Ret")

    col_labels = [MONTHS[c - 1] for c in pivot.columns]
    z_vals     = pivot.values.astype(float)
    txt        = np.where(
        np.isnan(z_vals), "",
        [[f"{v:.1f}%" for v in row] for row in z_vals]
    )

    f5 = go.Figure(go.Heatmap(
        z=z_vals, x=col_labels, y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=txt, texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="Jahr: %{y}<br>Monat: %{x}<br>Rendite: %{z:.2f}%<extra></extra>",
    ))
    f5.update_layout(**_layout(f"Monatsrenditen: {hm_asset}", h=280,
                                extra=dict(yaxis=dict(autorange="reversed"))))
    st.plotly_chart(f5, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'New Portfolio · Asset Explorer · '
    f'{PM_5y.index[0].strftime("%b %Y")} – {PM_5y.index[-1].strftime("%b %Y")} · '
    f'{len(ALL_ASSETS)} Assets total</p>',
    unsafe_allow_html=True,
)
