"""
data_loader.py
==============
Liest alle Bloomberg-Excel-Dateien aus data/Old Portfolio ein
und bereitet alle Daten für das Dashboard auf.

Verwendung:
    from data_loader import load_portfolio
    data = load_portfolio()
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# KONFIGURATION  – hier anpassen
# ─────────────────────────────────────────────

# Namen müssen EXAKT dem Security-Feld in den Excel-Dateien entsprechen (Zeile 1, Spalte B).
# Starte load_portfolio() einmal – gefundene Namen werden ausgegeben.
POSITIONS_EUR: dict[str, float] = {
    "BMW GY Equity":         8_400_000,
    "MBG GY Equity":        11_500_000,
    "DAXEX GY Equity":         9_800_000,
    "IWDA LN Equity":        3_500_000,
    "HIGH LN Equity":        4_900_000,
    "IBCI IM Equity":        2_900_000,
    "CSPX LN Equity":      6_600_000,
    "BO221256 Corp":         2_800_000,   # German Bund
}

# Metadaten je Asset  (für Dashboard-Anzeige)
ASSET_META: dict[str, dict] = {
    "BMW GY Equity":    {"name": "BMW AG",                              "class": "Equity",       "region": "Germany", "sector": "Automotive"},
    "MBG GY Equity":   {"name": "Mercedes-Benz Group AG",              "class": "Equity",       "region": "Germany", "sector": "Automotive"},
    "DAXEX GY Equity":   {"name": "iShares Core DAX UCITS ETF",          "class": "Equity",       "region": "Germany", "sector": "Broad Market"},
    "IWDA LN Equity":  {"name": "iShares Core MSCI World UCITS ETF",   "class": "Equity",       "region": "Global",  "sector": "Broad Market"},
    "HIGH LN Equity":  {"name": "iShares € High Yield Corp Bond ETF",  "class": "Fixed Income", "region": "Europe",  "sector": "Corporate Credit"},
    "IBCI IM Equity":  {"name": "iShares € Inflation Linked Bond ETF", "class": "Fixed Income", "region": "Europe",  "sector": "Inflation-Linked"},
    "CSPX LN Equity":{"name": "iShares Core S&P 500 UCITS ETF",      "class": "Equity",       "region": "USA",     "sector": "Broad Market"},
    "BO221256 Corp":   {"name": "German Bund 2036",                     "class": "Fixed Income", "region": "Germany", "sector": "Sovereign Bond"},
}

START_DATE   = "2021-03-19"
TRADING_DAYS = 252


# ─────────────────────────────────────────────
# EXCEL EINLESEN
# ─────────────────────────────────────────────
def _read_bloomberg_excel(filepath: Path) -> tuple[str, pd.DataFrame]:
    meta     = pd.read_excel(filepath, header=None, nrows=5, usecols=[0, 1])
    security = str(meta.iloc[0, 1]).strip()

    df = pd.read_excel(filepath, skiprows=6, usecols=[0, 1, 2], parse_dates=[0])
    df.columns = ["Date", "PX_MID", "PX_BID"]
    df = (df
          .dropna(subset=["Date"])
          .assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.normalize())
          .sort_values("Date")
          .reset_index(drop=True))
    df["PX_MID"] = pd.to_numeric(df["PX_MID"], errors="coerce")
    df["PX_BID"] = pd.to_numeric(df["PX_BID"], errors="coerce")
    return security, df


def _load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw = {}
    for f in sorted(data_dir.glob("*.xlsx")):
        try:
            security, df = _read_bloomberg_excel(f)
            raw[security] = df
        except Exception as e:
            print(f"  ❌  {f.name}: {e}")
    return raw


# ─────────────────────────────────────────────
# KENNZAHLEN
# ─────────────────────────────────────────────
def _metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    cum    = (1 + r).cumprod()
    total  = cum.iloc[-1] - 1
    ann_r  = (1 + total) ** (TRADING_DAYS / len(r)) - 1
    ann_v  = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_r / ann_v if ann_v > 0 else np.nan
    neg    = r[r < 0]
    ds     = neg.std() * np.sqrt(TRADING_DAYS) if len(neg) > 0 else np.nan
    sortino = ann_r / ds if ds and ds > 0 else np.nan
    roll_max = cum.cummax()
    max_dd   = ((cum - roll_max) / roll_max).min()
    cvar     = r[r <= r.quantile(0.05)].mean()
    return {
        "ann_return":  round(ann_r  * 100, 2),
        "ann_vol":     round(ann_v  * 100, 2),
        "sharpe":      round(sharpe, 3),
        "sortino":     round(sortino, 3),
        "max_drawdown":round(max_dd * 100, 2),
        "cvar_95":     round(cvar   * 100, 2),
        "total_return":round(total  * 100, 2),
    }


# ─────────────────────────────────────────────
# HAUPTFUNKTION
# ─────────────────────────────────────────────
def load_portfolio(data_dir: str | Path = None) -> dict:
    """
    Lädt alle Portfoliodaten und gibt ein dict zurück mit:
        prices        – DataFrame: tägliche Schlusskurse je Asset
        returns       – DataFrame: tägliche Renditen je Asset
        weights       – Series:   Portfoliogewichte (normiert)
        port_returns  – Series:   tägliche Portfoliorenditen
        port_cum      – Series:   kumulierte Portfoliorendite (Basis 1)
        port_drawdown – Series:   Drawdown-Zeitreihe
        metrics_df    – DataFrame: Kennzahlen je Asset + Portfolio
        summary_df    – DataFrame: Gewichte + Metadaten je Asset
        positions_eur – dict:     Marktwerte in EUR
    """
    # Pfad ermitteln
    if data_dir is None:
        here     = Path(__file__).resolve().parent
        data_dir = here / "data" / "Old Portfolio"
    data_dir = Path(data_dir)

    # Rohdaten laden
    raw = _load_raw(data_dir)

    # Preismatrix
    frames = {}
    for name in POSITIONS_EUR:
        if name in raw:
            frames[name] = raw[name].set_index("Date")["PX_MID"]
        else:
            print(f"  ⚠️  '{name}' nicht gefunden. Verfügbare Namen: {list(raw.keys())}")

    prices = (pd.DataFrame(frames)
              .sort_index()
              .loc[START_DATE:]
              .ffill()
              .dropna())

    # Renditen
    returns = prices.pct_change().dropna()

    # Gewichte (nur verfügbare Assets, renormiert)
    avail   = {k: v for k, v in POSITIONS_EUR.items() if k in prices.columns}
    total_v = sum(avail.values())
    weights = pd.Series({k: v / total_v for k, v in avail.items()})
    weights = weights.reindex(prices.columns).dropna()

    # Portfolio-Zeitreihen
    port_returns  = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    port_cum      = (1 + port_returns).cumprod()
    roll_max      = port_cum.cummax()
    port_drawdown = (port_cum - roll_max) / roll_max

    # Kennzahlen je Asset
    rows = []
    for col in returns.columns:
        m = _metrics(returns[col])
        m["asset"] = col
        m["name"]  = ASSET_META.get(col, {}).get("name", col)
        rows.append(m)
    # Portfolio-Kennzahlen
    pm = _metrics(port_returns)
    pm["asset"] = "PORTFOLIO"
    pm["name"]  = "Portfolio Gesamt"
    rows.append(pm)
    metrics_df = pd.DataFrame(rows).set_index("asset")

    # Summary-Tabelle
    summary_rows = []
    for ticker, eur in avail.items():
        meta = ASSET_META.get(ticker, {})
        summary_rows.append({
            "ticker":  ticker,
            "name":    meta.get("name", ticker),
            "class":   meta.get("class", "–"),
            "region":  meta.get("region", "–"),
            "sector":  meta.get("sector", "–"),
            "eur":     eur,
            "weight":  weights[ticker],
        })
    summary_df = pd.DataFrame(summary_rows).set_index("ticker")

    return {
        "prices":        prices,
        "returns":       returns,
        "weights":       weights,
        "port_returns":  port_returns,
        "port_cum":      port_cum,
        "port_drawdown": port_drawdown,
        "metrics_df":    metrics_df,
        "summary_df":    summary_df,
        "positions_eur": avail,
        "total_value":   total_v,
        "start_date":    START_DATE,
    }
