"""
dashboard_data_availability.py
====================
Übersicht über Datenverfügbarkeit aller Assets im New Portfolio.

Zeigt:
- Start- und Enddatum je Asset
- Anzahl vorhandener / fehlender Monatswerte
- Heatmap der Datenverfügbarkeit über die Zeit

Ablegen: dashboard/pages/data_availability.py
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

# scan_folder kommt direkt aus dem neuen Loader
from new_portfolio_loader import scan_folder


st.set_page_config(
    page_title="Data Availability",
    page_icon="🗂️",
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
    color:#0F172A !important; font-size:1.25rem !important; font-weight:700;
}
hr { border-color:#E2E8F0 !important; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def build_monthly_raw_matrix(raw_data: dict) -> pd.DataFrame:
    """
    Monatliche Matrix OHNE gemeinsamen Zuschnitt und OHNE dropna(axis=0).
    Genau das brauchen wir für die Verfügbarkeitsanalyse –
    wir wollen sehen wo Lücken sind, nicht sie verstecken.
    """
    series = {}
    for name, df in raw_data.items():
        s = (
            df[["Date", "PX_MID"]]
            .dropna()
            .drop_duplicates(subset=["Date"], keep="last")
            .sort_values("Date")
            .set_index("Date")["PX_MID"]
            .resample("ME")
            .last()
        )
        series[name] = s
    return pd.DataFrame(series).sort_index()


def build_availability_summary(prices_raw: pd.DataFrame, asset_meta: dict) -> pd.DataFrame:
    rows = []
    for asset in prices_raw.columns:
        s      = prices_raw[asset]
        non_na = s.dropna()

        first_valid = non_na.index.min() if not non_na.empty else pd.NaT
        last_valid  = non_na.index.max() if not non_na.empty else pd.NaT

        n_present = int(s.notna().sum())
        n_missing = int(s.isna().sum())

        if pd.notna(first_valid) and pd.notna(last_valid):
            in_range          = s.loc[first_valid:last_valid]
            internal_missing  = int(in_range.isna().sum())
            possible_months   = len(in_range)
        else:
            internal_missing = 0
            possible_months  = 0

        rows.append({
            "Asset":            asset,
            "Klasse":           asset_meta.get(asset, {}).get("asset_class", "Other"),
            "Erstes Datum":     first_valid.date() if pd.notna(first_valid) else None,
            "Letztes Datum":    last_valid.date()  if pd.notna(last_valid)  else None,
            "Monate vorhanden": n_present,
            "Fehlend gesamt":   n_missing,
            "Interne Lücken":   internal_missing,
            "Aktives Fenster":  possible_months,
            "Start >= 2025":    "Ja" if (pd.notna(first_valid)
                                         and first_valid >= pd.Timestamp("2025-01-01"))
                                     else "Nein",
        })

    return (pd.DataFrame(rows)
            .set_index("Asset")
            .sort_values(["Erstes Datum", "Klasse"], ascending=[True, True]))


# ─────────────────────────────────────────────
# DATA  (gecacht)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Datenverfügbarkeit …")
def get_data():
    root               = PROJECT_ROOT / "data" / "New Portfolio"
    raw_data, asset_meta = scan_folder(root)
    prices_raw         = build_monthly_raw_matrix(raw_data)
    summary            = build_availability_summary(prices_raw, asset_meta)
    return asset_meta, prices_raw, summary


asset_meta, prices_raw, summary = get_data()
all_classes = sorted(summary["Klasse"].dropna().unique().tolist())


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ Data Availability")
    st.markdown("---")

    sel_classes = st.multiselect(
        "Asset-Klassen", all_classes, default=all_classes,
    )

    only_2025  = st.toggle("Nur Assets mit Start ab 2025",               value=False)
    only_gaps  = st.toggle("Nur Assets mit internen Lücken",             value=False)

    sort_col = st.selectbox("Sortierung", [
        "Erstes Datum", "Letztes Datum",
        "Monate vorhanden", "Fehlend gesamt", "Interne Lücken",
    ])
    sort_asc = st.toggle("Aufsteigend", value=True)


# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
df = summary[summary["Klasse"].isin(sel_classes)].copy()

if only_2025:
    df = df[df["Start >= 2025"] == "Ja"]
if only_gaps:
    df = df[df["Interne Lücken"].fillna(0) > 0]

df = df.sort_values(sort_col, ascending=sort_asc)

sel_assets = df.index.tolist()

if not sel_assets:
    st.warning("Keine Assets für die aktuelle Filterauswahl gefunden.")
    st.stop()

avail = prices_raw[sel_assets].notna().astype(int)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:.6rem 0 .4rem;">
    <h1 style="margin:.3rem 0 0;font-size:1.9rem;font-weight:800;color:#0F172A;">
        Datenverfügbarkeit
    </h1>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"{len(sel_assets)} Assets  ·  "
    f"{avail.index.min().strftime('%b %Y')} – {avail.index.max().strftime('%b %Y')}  ·  "
    "Monatliche Verfügbarkeit"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets (gefiltert)",   len(sel_assets))
c2.metric("Assets gesamt",        len(summary))
c3.metric("Start ab 2025",        int((df["Start >= 2025"] == "Ja").sum()))
c4.metric("Mit internen Lücken",  int((df["Interne Lücken"].fillna(0) > 0).sum()))

st.markdown("---")


# ─────────────────────────────────────────────
# TABELLE
# ─────────────────────────────────────────────
st.subheader("Übersichtstabelle")
st.dataframe(df, use_container_width=True,
             height=min(60 + len(df) * 35, 700))
st.markdown("---")


# ─────────────────────────────────────────────
# HEATMAP: VERFÜGBARKEIT
# ─────────────────────────────────────────────
st.subheader("Heatmap: Daten vorhanden / fehlend")

heat_y = sel_assets[::-1]
heat_z = avail[heat_y].T.values

fig = go.Figure(go.Heatmap(
    z=heat_z,
    x=avail.index,
    y=heat_y,
    colorscale=[
        [0.0,    "#F1F5F9"],   # fehlt  → hellgrau
        [0.4999, "#F1F5F9"],
        [0.5,    "#2563EB"],   # vorhanden → blau
        [1.0,    "#2563EB"],
    ],
    zmin=0, zmax=1,
    hovertemplate="Asset: %{y}<br>Monat: %{x|%b %Y}<br>"
                  + "vorhanden" + "<extra></extra>",
    showscale=False,
))
fig.update_layout(
    height=max(500, len(heat_y) * 22),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Syne", size=11, color="#64748B"),
    margin=dict(l=10, r=10, t=44, b=10),
    title=dict(text="Blau = vorhanden · Grau = fehlend",
               font=dict(family="Syne", size=13, color="#0F172A")),
    xaxis=dict(title="Zeit",  gridcolor="#E2E8F0"),
    yaxis=dict(title="Asset", gridcolor="#E2E8F0"),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")


# ─────────────────────────────────────────────
# ASSETS MIT START AB 2025
# ─────────────────────────────────────────────
st.subheader("Assets mit Start ab 2025")
late = df[df["Start >= 2025"] == "Ja"]

if late.empty:
    st.info("Keine Assets mit erstem gültigen Monatswert ab 2025 in der aktuellen Auswahl.")
else:
    st.dataframe(
        late[["Klasse","Erstes Datum","Letztes Datum","Monate vorhanden","Interne Lücken"]],
        use_container_width=True,
        height=min(60 + len(late) * 35, 400),
    )
st.markdown("---")


# ─────────────────────────────────────────────
# ASSETS MIT INTERNEN LÜCKEN
# ─────────────────────────────────────────────
st.subheader("Assets mit Lücken zwischen erstem und letztem Wert")
gaps = df[df["Interne Lücken"].fillna(0) > 0]

if gaps.empty:
    st.info("Keine internen Lücken in der aktuellen Auswahl.")
else:
    st.dataframe(
        gaps[["Klasse","Erstes Datum","Letztes Datum","Interne Lücken","Monate vorhanden"]],
        use_container_width=True,
        height=min(60 + len(gaps) * 35, 400),
    )