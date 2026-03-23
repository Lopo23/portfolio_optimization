"""
data_availability.py
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

from data_chatgpt import scan_folder


st.set_page_config(
    page_title="Data Availability",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def build_monthly_raw_matrix(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Baut eine monatliche Matrix OHNE gemeinsamen Zuschnitt
    und OHNE dropna(axis=0, how='any').
    Genau das brauchen wir für die Verfügbarkeitsanalyse.
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


def build_availability_summary(prices_monthly_raw: pd.DataFrame, asset_meta: dict) -> pd.DataFrame:
    rows = []

    full_index = prices_monthly_raw.index

    for asset in prices_monthly_raw.columns:
        s = prices_monthly_raw[asset]
        non_na = s.dropna()

        first_valid = non_na.index.min() if not non_na.empty else pd.NaT
        last_valid = non_na.index.max() if not non_na.empty else pd.NaT

        n_present = int(s.notna().sum())
        n_missing_total = int(s.isna().sum())

        if pd.notna(first_valid) and pd.notna(last_valid):
            in_range = s.loc[first_valid:last_valid]
            internal_missing = int(in_range.isna().sum())
            possible_months = len(in_range)
        else:
            internal_missing = np.nan
            possible_months = 0

        starts_2025_or_later = (
            pd.notna(first_valid) and first_valid >= pd.Timestamp("2025-01-01")
        )

        rows.append({
            "Asset": asset,
            "Klasse": asset_meta.get(asset, {}).get("asset_class", "Other"),
            "Erstes Datum": first_valid.date() if pd.notna(first_valid) else None,
            "Letztes Datum": last_valid.date() if pd.notna(last_valid) else None,
            "Monate vorhanden": n_present,
            "Monate fehlend gesamt": n_missing_total,
            "Fehlende Monate zwischen erstem und letztem Wert": internal_missing,
            "Monate im aktiven Fenster": possible_months,
            "Start >= 2025": "Ja" if starts_2025_or_later else "Nein",
        })

    df = pd.DataFrame(rows).set_index("Asset").sort_values(
        ["Erstes Datum", "Klasse"], ascending=[True, True]
    )
    return df


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Datenverfügbarkeit …")
def get_data():
    root = PROJECT_ROOT / "data" / "New Portfolio"
    raw_data, asset_meta = scan_folder(root)
    prices_monthly_raw = build_monthly_raw_matrix(raw_data)
    summary = build_availability_summary(prices_monthly_raw, asset_meta)
    return raw_data, asset_meta, prices_monthly_raw, summary


raw_data, asset_meta, prices_monthly_raw, summary = get_data()

all_classes = sorted(summary["Klasse"].dropna().unique().tolist())


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ Data Availability")
    st.markdown("---")

    sel_classes = st.multiselect(
        "Asset-Klassen",
        options=all_classes,
        default=all_classes,
    )

    only_2025 = st.toggle("Nur Assets mit Start ab 2025", value=False)
    only_internal_gaps = st.toggle("Nur Assets mit Lücken zwischen Start und Ende", value=False)

    sort_col = st.selectbox(
        "Sortierung",
        [
            "Erstes Datum",
            "Letztes Datum",
            "Monate vorhanden",
            "Monate fehlend gesamt",
            "Fehlende Monate zwischen erstem und letztem Wert",
        ],
    )

    sort_asc = st.toggle("Aufsteigend", value=True)


# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
filtered_summary = summary[summary["Klasse"].isin(sel_classes)].copy()

if only_2025:
    filtered_summary = filtered_summary[filtered_summary["Start >= 2025"] == "Ja"]

if only_internal_gaps:
    filtered_summary = filtered_summary[
        filtered_summary["Fehlende Monate zwischen erstem und letztem Wert"].fillna(0) > 0
    ]

filtered_summary = filtered_summary.sort_values(sort_col, ascending=sort_asc)

selected_assets = filtered_summary.index.tolist()

if not selected_assets:
    st.warning("Keine Assets für die aktuelle Filterauswahl gefunden.")
    st.stop()

avail = prices_monthly_raw[selected_assets].notna().astype(int)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("Datenverfügbarkeit aller Assets")
st.caption(
    f"{len(selected_assets)} Assets · "
    f"{avail.index.min().strftime('%b %Y')} – {avail.index.max().strftime('%b %Y')} · "
    "Monatliche Verfügbarkeit"
)

c1, c2, c3 = st.columns(3)
c1.metric("Assets gesamt", len(selected_assets))
c2.metric("Start ab 2025", int((filtered_summary["Start >= 2025"] == "Ja").sum()))
c3.metric(
    "Mit internen Lücken",
    int((filtered_summary["Fehlende Monate zwischen erstem und letztem Wert"].fillna(0) > 0).sum())
)

st.markdown("---")


# ─────────────────────────────────────────────
# TABELLE
# ─────────────────────────────────────────────
st.subheader("Übersichtstabelle")

st.dataframe(
    filtered_summary,
    width="stretch",
    height=min(60 + len(filtered_summary) * 35, 700),
)

st.markdown("---")


# ─────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────
st.subheader("Heatmap: Daten vorhanden / fehlend")

heat_y = selected_assets[::-1]
heat_z = avail[heat_y].T.values

fig = go.Figure(
    go.Heatmap(
        z=heat_z,
        x=avail.index,
        y=heat_y,
        colorscale=[
            [0.0, "#F1F5F9"],   # fehlt
            [0.4999, "#F1F5F9"],
            [0.5, "#2563EB"],   # vorhanden
            [1.0, "#2563EB"],
        ],
        zmin=0,
        zmax=1,
        hovertemplate="Asset: %{y}<br>Monat: %{x|%b %Y}<br>%{z}<extra></extra>",
        showscale=False,
    )
)

fig.update_layout(
    height=max(500, len(heat_y) * 22),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=10, r=10, t=40, b=10),
    title="Blau = Daten vorhanden · Grau = fehlt",
    xaxis=dict(title="Zeit"),
    yaxis=dict(title="Asset"),
)

st.plotly_chart(fig, width="stretch")

st.markdown("---")


# ─────────────────────────────────────────────
# STARTDATEN AB 2025
# ─────────────────────────────────────────────
late_starters = filtered_summary[filtered_summary["Start >= 2025"] == "Ja"]

st.subheader("Assets mit Start ab 2025")

if late_starters.empty:
    st.info("Keine Assets mit erstem gültigen Monatswert ab 2025 in der aktuellen Auswahl.")
else:
    st.dataframe(
        late_starters[[
            "Klasse",
            "Erstes Datum",
            "Letztes Datum",
            "Monate vorhanden",
            "Fehlende Monate zwischen erstem und letztem Wert",
        ]],
        width="stretch",
        height=min(60 + len(late_starters) * 35, 400),
    )

st.markdown("---")


# ─────────────────────────────────────────────
# ASSETS MIT INTERNEN LÜCKEN
# ─────────────────────────────────────────────
gaps = filtered_summary[
    filtered_summary["Fehlende Monate zwischen erstem und letztem Wert"].fillna(0) > 0
]

st.subheader("Assets mit Lücken zwischen erstem und letztem Wert")

if gaps.empty:
    st.info("Keine internen Lücken in der aktuellen Auswahl.")
else:
    st.dataframe(
        gaps[[
            "Klasse",
            "Erstes Datum",
            "Letztes Datum",
            "Fehlende Monate zwischen erstem und letztem Wert",
            "Monate vorhanden",
        ]],
        width="stretch",
        height=min(60 + len(gaps) * 35, 400),
    )