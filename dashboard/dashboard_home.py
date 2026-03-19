import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Portfolio Dashboard – Markus Rzenkowski",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Pfade
# =========================================================

BASE_DIR = Path(__file__).resolve().parent          # .../dashboard
DATA_DIR = BASE_DIR.parent / "data" / "portfolio"   # .../data/portfolio

# =========================================================
# Hilfsfunktionen
# =========================================================

@st.cache_data
def load_data():
    files = {
        "asset_prices": DATA_DIR / "asset_prices.csv",
        "asset_returns": DATA_DIR / "asset_returns.csv",
        "portfolio_position_values": DATA_DIR / "portfolio_position_values.csv",
        "portfolio_timeseries": DATA_DIR / "portfolio_timeseries.csv",
        "portfolio_weights_over_time": DATA_DIR / "portfolio_weights_over_time.csv",
        "portfolio_weights_summary": DATA_DIR / "portfolio_weights_summary.csv",
    }

    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Folgende Dateien fehlen in {DATA_DIR}: {', '.join(missing)}"
        )

    asset_prices = pd.read_csv(files["asset_prices"], index_col=0, parse_dates=True)
    asset_returns = pd.read_csv(files["asset_returns"], index_col=0, parse_dates=True)
    portfolio_position_values = pd.read_csv(
        files["portfolio_position_values"], index_col=0, parse_dates=True
    )
    portfolio_timeseries = pd.read_csv(
        files["portfolio_timeseries"], index_col=0, parse_dates=True
    )
    portfolio_weights_over_time = pd.read_csv(
        files["portfolio_weights_over_time"], index_col=0, parse_dates=True
    )
    portfolio_weights_summary = pd.read_csv(files["portfolio_weights_summary"], index_col=0)

    return {
        "asset_prices": asset_prices,
        "asset_returns": asset_returns,
        "portfolio_position_values": portfolio_position_values,
        "portfolio_timeseries": portfolio_timeseries,
        "portfolio_weights_over_time": portfolio_weights_over_time,
        "portfolio_weights_summary": portfolio_weights_summary,
    }


def make_display_names():
    return {
        "MBG.DE": "Mercedes-Benz Group AG",
        "BMW.DE": "BMW AG",
        "EXS1.DE": "iShares Core DAX UCITS ETF",
        "SXR8.DE": "iShares Core S&P 500 UCITS ETF",
        "EUNL.DE": "iShares Core MSCI World UCITS ETF",
        "IHYG.L": "High Yield Corp Bond UCITS ETF",
        "IBCI.DE": "iShares € Inflation Linked Govt Bond UCITS ETF",
        "GERMAN_BUND_2036": "German Bund 15 May 2036",
        "TOTAL_PORTFOLIO": "Total Portfolio"
    }


def classify_asset(ticker: str):
    if ticker in ["MBG.DE", "BMW.DE", "EXS1.DE", "SXR8.DE", "EUNL.DE"]:
        return "Equity"
    elif ticker in ["IHYG.L", "IBCI.DE", "GERMAN_BUND_2036"]:
        return "Fixed Income"
    return "Other"


def classify_region(ticker: str):
    if ticker in ["MBG.DE", "BMW.DE", "EXS1.DE", "GERMAN_BUND_2036"]:
        return "Germany"
    elif ticker == "SXR8.DE":
        return "USA"
    elif ticker in ["IHYG.L", "EUNL.DE"]:
        return "Global"
    elif ticker == "IBCI.DE":
        return "Europe"
    return "Other"


# =========================================================
# Daten laden
# =========================================================

st.title("📊 Portfolio Dashboard – Markus Rzenkowski")

try:
    data = load_data()
except Exception as e:
    st.error(f"Fehler beim Laden der Portfoliodaten: {e}")
    st.stop()

asset_prices = data["asset_prices"]
asset_returns = data["asset_returns"]
portfolio_position_values = data["portfolio_position_values"]
portfolio_timeseries = data["portfolio_timeseries"]
portfolio_weights_over_time = data["portfolio_weights_over_time"]
portfolio_weights_summary = data["portfolio_weights_summary"]

display_names = make_display_names()

# =========================================================
# Metadaten / Aufbereitung
# =========================================================

summary_df = portfolio_weights_summary.reset_index().rename(columns={"index": "Ticker"})
summary_df["Name"] = summary_df["Ticker"].map(display_names).fillna(summary_df["Ticker"])
summary_df["Asset Class"] = summary_df["Ticker"].apply(classify_asset)
summary_df["Region"] = summary_df["Ticker"].apply(classify_region)

summary_df["initial_amount_eur"] = pd.to_numeric(summary_df["initial_amount_eur"], errors="coerce")
summary_df["weight_total_portfolio"] = pd.to_numeric(summary_df["weight_total_portfolio"], errors="coerce")
summary_df["weight_yahoo_only"] = pd.to_numeric(summary_df["weight_yahoo_only"], errors="coerce")

summary_df["Weight_Pct"] = summary_df["weight_total_portfolio"] * 100

positions_only_df = summary_df[summary_df["Ticker"] != "TOTAL_PORTFOLIO"].copy()
positions_only_df = positions_only_df.sort_values("initial_amount_eur", ascending=False)

total_portfolio_value = positions_only_df["initial_amount_eur"].sum()
num_positions = len(positions_only_df)

largest_position = positions_only_df.loc[positions_only_df["Weight_Pct"].idxmax()]
top_3_weight = positions_only_df.nlargest(3, "Weight_Pct")["Weight_Pct"].sum()
equity_weight = positions_only_df.loc[positions_only_df["Asset Class"] == "Equity", "Weight_Pct"].sum()
fixed_income_weight = positions_only_df.loc[positions_only_df["Asset Class"] == "Fixed Income", "Weight_Pct"].sum()
germany_weight = positions_only_df.loc[positions_only_df["Region"] == "Germany", "Weight_Pct"].sum()
single_stock_weight = positions_only_df.loc[
    positions_only_df["Ticker"].isin(["MBG.DE", "BMW.DE"]), "Weight_Pct"
].sum()

# Performance-Zeitreihe
portfolio_index_col = None
portfolio_return_col = None

for col in portfolio_timeseries.columns:
    if "portfolio_index_total" in col.lower():
        portfolio_index_col = col
    if "portfolio_return_total" in col.lower():
        portfolio_return_col = col

if portfolio_index_col is None:
    possible = [c for c in portfolio_timeseries.columns if "index" in c.lower()]
    if possible:
        portfolio_index_col = possible[0]

if portfolio_return_col is None:
    possible = [c for c in portfolio_timeseries.columns if "return" in c.lower()]
    if possible:
        portfolio_return_col = possible[0]

portfolio_index = portfolio_timeseries[portfolio_index_col].dropna() if portfolio_index_col else None
portfolio_returns = portfolio_timeseries[portfolio_return_col].dropna() if portfolio_return_col else None

# Drawdown berechnen
drawdown = None
max_drawdown = None
if portfolio_index is not None and not portfolio_index.empty:
    running_max = portfolio_index.cummax()
    drawdown = (portfolio_index / running_max) - 1
    max_drawdown = drawdown.min() * 100

# Annualisierte Kennzahlen
annual_return = None
annual_vol = None
if portfolio_returns is not None and not portfolio_returns.empty:
    annual_return = portfolio_returns.mean() * 252 * 100
    annual_vol = portfolio_returns.std() * (252 ** 0.5) * 100

# =========================================================
# Intro
# =========================================================

st.markdown("""
Dieses Dashboard greift direkt auf die **bereits heruntergeladenen Portfoliodaten** aus `data/portfolio/` zu
und visualisiert die **aktuelle Struktur** des Vermögens von Markus Rzenkowski.

Im Fokus stehen:
- aktuelle Gewichte und Positionsgrößen
- historische Entwicklung des Portfolios
- Konzentrations- und Regionalrisiken
- erste Einordnung für die spätere Restrukturierung
""")

st.caption(f"Datenpfad: `{DATA_DIR}`")

st.markdown("---")

# =========================================================
# KPIs
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Gesamtvermögen", f"{total_portfolio_value/1e6:.1f} Mio. EUR")
col2.metric("Anzahl Positionen", f"{num_positions}")
col3.metric("Größte Position", f"{largest_position['Weight_Pct']:.1f}%")
col4.metric("Top-3 Konzentration", f"{top_3_weight:.1f}%")
col5.metric("Max Drawdown", f"{max_drawdown:.1f}%" if max_drawdown is not None else "n/a")

if annual_return is not None and annual_vol is not None:
    col6, col7 = st.columns(2)
    col6.metric("Ø Jahresrendite", f"{annual_return:.2f}%")
    col7.metric("Annualisierte Volatilität", f"{annual_vol:.2f}%")

st.markdown("---")

# =========================================================
# Historische Entwicklung
# =========================================================

st.subheader("Historische Entwicklung des aktuellen Portfolios")

col_left, col_right = st.columns(2)

with col_left:
    if portfolio_index is not None and not portfolio_index.empty:
        index_df = portfolio_index.reset_index()
        index_df.columns = ["Date", "Portfolio Index"]

        fig_index = px.line(
            index_df,
            x="Date",
            y="Portfolio Index",
            title="Portfolio Index (Start = 100)"
        )
        fig_index.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="",
            yaxis_title="Index"
        )
        st.plotly_chart(fig_index, use_container_width=True)
    else:
        st.info("Keine Portfolio-Index-Zeitreihe gefunden.")

with col_right:
    if drawdown is not None and not drawdown.empty:
        dd_df = drawdown.reset_index()
        dd_df.columns = ["Date", "Drawdown"]

        fig_dd = px.area(
            dd_df,
            x="Date",
            y="Drawdown",
            title="Drawdown des Portfolios"
        )
        fig_dd.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="",
            yaxis_title="Drawdown"
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("Keine Drawdown-Daten verfügbar.")

st.markdown("---")

# =========================================================
# Aktuelle Allokation
# =========================================================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Aktuelle Portfolio-Gewichte")

    donut_df = positions_only_df[["Name", "initial_amount_eur"]].copy()

    fig_donut = px.pie(
        donut_df,
        values="initial_amount_eur",
        names="Name",
        hole=0.5
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent")
    fig_donut.update_layout(
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    st.subheader("Positionsgrößen")

    bar_df = positions_only_df.sort_values("initial_amount_eur", ascending=True).copy()
    bar_df["Amount_Million_EUR"] = bar_df["initial_amount_eur"] / 1e6

    fig_bar = px.bar(
        bar_df,
        x="Amount_Million_EUR",
        y="Name",
        orientation="h",
        color="Asset Class",
        text="Amount_Million_EUR"
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_bar.update_layout(
        xaxis_title="Mio. EUR",
        yaxis_title="",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# =========================================================
# Asset Class / Region
# =========================================================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Allokation nach Asset Class")

    asset_class_df = (
        positions_only_df.groupby("Asset Class", as_index=False)["initial_amount_eur"]
        .sum()
        .sort_values("initial_amount_eur", ascending=False)
    )
    asset_class_df["Weight_Pct"] = asset_class_df["initial_amount_eur"] / asset_class_df["initial_amount_eur"].sum() * 100

    fig_asset = px.bar(
        asset_class_df,
        x="Asset Class",
        y="Weight_Pct",
        text="Weight_Pct",
        color="Asset Class"
    )
    fig_asset.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_asset.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="Gewicht in %"
    )
    st.plotly_chart(fig_asset, use_container_width=True)

with col_right:
    st.subheader("Regionale Verteilung")

    region_df = (
        positions_only_df.groupby("Region", as_index=False)["initial_amount_eur"]
        .sum()
        .sort_values("initial_amount_eur", ascending=False)
    )

    fig_region = px.pie(
        region_df,
        values="initial_amount_eur",
        names="Region",
        hole=0.45
    )
    fig_region.update_traces(textposition="inside", textinfo="percent+label")
    fig_region.update_layout(
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_region, use_container_width=True)

st.markdown("---")

# =========================================================
# Gewichte über die Zeit
# =========================================================

st.subheader("Entwicklung der Portfolio-Gewichte über die Zeit")

weights_plot_df = portfolio_weights_over_time.copy()
weights_plot_df = weights_plot_df.rename(columns=display_names)

if not weights_plot_df.empty:
    fig_weights = px.area(
        weights_plot_df,
        x=weights_plot_df.index,
        y=weights_plot_df.columns,
        title="Gewichtsverschiebung der Positionen"
    )
    fig_weights.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="",
        yaxis_title="Gewicht"
    )
    st.plotly_chart(fig_weights, use_container_width=True)
else:
    st.info("Keine Zeitreihen für Gewichte verfügbar.")

st.markdown("---")

# =========================================================
# Konzentrationsanalyse
# =========================================================

st.subheader("Konzentrationsanalyse")

col1, col2 = st.columns([1.2, 1])

with col1:
    conc_df = positions_only_df[["Name", "Weight_Pct"]].sort_values("Weight_Pct", ascending=False)

    fig_conc = go.Figure()
    fig_conc.add_trace(
        go.Bar(
            x=conc_df["Name"],
            y=conc_df["Weight_Pct"],
            text=conc_df["Weight_Pct"].round(1).astype(str) + "%"
        )
    )
    fig_conc.update_traces(textposition="outside")
    fig_conc.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="Gewicht in %"
    )
    st.plotly_chart(fig_conc, use_container_width=True)

with col2:
    st.markdown("#### Wesentliche Beobachtungen")
    st.info(
        f"""
- Die größte Position ist **{largest_position['Name']}** mit **{largest_position['Weight_Pct']:.1f}%**.
- Die **Top-3-Positionen** machen **{top_3_weight:.1f}%** des Portfolios aus.
- Der Anteil an **Einzelaktien** beträgt **{single_stock_weight:.1f}%**.
- Der Anteil an **Equity** beträgt **{equity_weight:.1f}%**.
- Der Anteil an **Fixed Income** beträgt **{fixed_income_weight:.1f}%**.
- Das Exposure zu **Deutschland** beträgt **{germany_weight:.1f}%**.
"""
    )

    if top_3_weight > 50:
        st.warning("Das Portfolio ist stark auf wenige Positionen konzentriert.")
    if germany_weight > 50:
        st.warning("Das Portfolio weist einen deutlichen Deutschland-Schwerpunkt auf.")
    if single_stock_weight > 30:
        st.warning("Der Anteil einzelner Aktien erhöht das idiosynkratische Risiko spürbar.")

st.markdown("---")

# =========================================================
# Detailtabellen
# =========================================================

st.subheader("Detailübersicht")

table_df = positions_only_df.copy()
table_df["Initialbetrag"] = table_df["initial_amount_eur"].map(lambda x: f"{x:,.0f} EUR")
table_df["Gewicht Gesamtportfolio"] = table_df["weight_total_portfolio"].map(lambda x: f"{x*100:.2f}%")
table_df["Gewicht Yahoo-only"] = table_df["weight_yahoo_only"].apply(
    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-"
)

st.dataframe(
    table_df[
        [
            "Ticker",
            "Name",
            "Asset Class",
            "Region",
            "Initialbetrag",
            "Gewicht Gesamtportfolio",
            "Gewicht Yahoo-only",
        ]
    ],
    use_container_width=True,
    hide_index=True
)

with st.expander("Rohdaten anzeigen"):
    st.markdown("**portfolio_weights_summary.csv**")
    st.dataframe(portfolio_weights_summary, use_container_width=True)

    st.markdown("**portfolio_timeseries.csv**")
    st.dataframe(portfolio_timeseries.tail(20), use_container_width=True)

st.markdown("---")

# =========================================================
# Interpretation
# =========================================================

st.subheader("Interpretation")

st.markdown("""
Die aktuelle Portfoliostruktur zeigt bereits eine Mischung aus Aktien- und Rentenbausteinen,
ist aber klar von mehreren Schwerpunkten geprägt:

- hoher Anteil an **deutschen Assets**
- erhebliche Konzentration auf wenige große Positionen
- direkte Einzelaktienrisiken bei **Mercedes-Benz** und **BMW**
- defensive Bausteine vorhanden, aber noch nicht dominant

Damit eignet sich dieses Dashboard gut als Ausgangspunkt für die nächsten Schritte:
**Risikobewertung, Diversifikationsanalyse, Zielallokation und Restrukturierungsvorschläge**.
""")