"""
data_loader.py
==============
Liest alle Bloomberg-Excel-Dateien aus data/Old Portfolio ein
und bereitet alle Daten für das Dashboard auf.

Renditen: monatlich (letzter Handelstag je Monat → pct_change)
Annualisierung: × 12 für Renditen/Varianz, × √12 für Volatilität

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
POSITIONS_EUR = {  # type: Dict[str, float]
    "BMW GY Equity":         8_400_000,
    "MBG GY Equity":        11_500_000,
    "DAXEX GY Equity":         9_800_000,   # ← ggf. anpassen
    "IWDA LN Equity":        3_500_000,
    "HIGH LN Equity":        4_900_000,
    "IBCI IM Equity":        2_900_000,
    "CSPX LN Equity":      6_600_000,
    "BO221256 Corp":         2_800_000,   # German Bund
}

# Metadaten je Asset (für Dashboard-Anzeige)
ASSET_META = {  # type: Dict[str, dict]
    "BMW GY Equity":    {"name": "BMW AG",                              "class": "Equity",       "region": "Germany", "sector": "Automotive"},
    "MBG GY Equity":   {"name": "Mercedes-Benz Group AG",              "class": "Equity",       "region": "Germany", "sector": "Automotive"},
    "DAXEX GY Equity":   {"name": "iShares Core DAX UCITS ETF",          "class": "Equity",       "region": "Germany", "sector": "Broad Market"},
    "IWDA LN Equity":  {"name": "iShares Core MSCI World UCITS ETF",   "class": "Equity",       "region": "Global",  "sector": "Broad Market"},
    "HIGH LN Equity":  {"name": "iShares € High Yield Corp Bond ETF",  "class": "Fixed Income", "region": "Europe",  "sector": "Corporate Credit"},
    "IBCI IM Equity":  {"name": "iShares € Inflation Linked Bond ETF", "class": "Fixed Income", "region": "Europe",  "sector": "Inflation-Linked"},
    "CSPX LN Equity":{"name": "iShares Core S&P 500 UCITS ETF",      "class": "Equity",       "region": "USA",     "sector": "Broad Market"},
    "BO221256 Corp":   {"name": "German Bund 2036",                     "class": "Fixed Income", "region": "Germany", "sector": "Sovereign Bond"},
}

START_DATE     = "2021-03-19"
MONTHS_PER_YEAR = 12


# ─────────────────────────────────────────────
# EXCEL EINLESEN
# ─────────────────────────────────────────────
def _read_bloomberg_excel(filepath):  # (Path) -> Tuple[str, pd.DataFrame]
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


def _load_raw(data_dir):  # (Path) -> Dict[str, pd.DataFrame]
    raw = {}
    for f in sorted(data_dir.glob("*.xlsx")):
        try:
            security, df = _read_bloomberg_excel(f)
            raw[security] = df
        except Exception as e:
            print(f"  ❌  {f.name}: {e}")
    return raw


# ─────────────────────────────────────────────
# TÄGLICHE → MONATLICHE PREISE
# ─────────────────────────────────────────────
def _to_monthly(prices_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resamplet tägliche Kurse auf den letzten Handelstag je Monat (MS = month-end).
    """
    return prices_daily.resample("ME").last().dropna()


# ─────────────────────────────────────────────
# KENNZAHLEN  (monatliche Renditen als Input)
# ─────────────────────────────────────────────
def _metrics(r: pd.Series) -> dict:
    """
    r: monatliche Renditen
    Annualisierung: Rendite × 12, Volatilität × √12
    """
    r = r.dropna()
    if len(r) == 0:
        return {}

    cum     = (1 + r).cumprod()
    total   = cum.iloc[-1] - 1
    n_years = len(r) / MONTHS_PER_YEAR
    ann_r   = (1 + total) ** (1 / n_years) - 1          # geometrisch annualisiert
    ann_v   = r.std() * np.sqrt(MONTHS_PER_YEAR)         # √12-Skalierung

    sharpe  = ann_r / ann_v if ann_v > 0 else np.nan

    neg      = r[r < 0]
    ds       = neg.std() * np.sqrt(MONTHS_PER_YEAR) if len(neg) > 0 else np.nan
    sortino  = ann_r / ds if ds and ds > 0 else np.nan

    roll_max = cum.cummax()
    max_dd   = ((cum - roll_max) / roll_max).min()

    # CVaR auf monatlicher Basis (5%-Quantil der schlechtesten Monate)
    cvar     = r[r <= r.quantile(0.05)].mean()

    return {
        "ann_return":   round(ann_r  * 100, 2),
        "ann_vol":      round(ann_v  * 100, 2),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "cvar_95":      round(cvar   * 100, 2),
        "total_return": round(total  * 100, 2),
    }


# ─────────────────────────────────────────────
# HAUPTFUNKTION
# ─────────────────────────────────────────────
def load_portfolio(data_dir=None):  # accepts str or Path, returns dict
    """
    Lädt alle Portfoliodaten und gibt ein dict zurück mit:

        prices_daily   – DataFrame: tägliche Schlusskurse  (für Charts)
        prices_monthly – DataFrame: monatliche Schlusskurse (letzter Handelstag)
        returns        – DataFrame: monatliche Renditen     (für Optimierung & Kennzahlen)
        weights        – Series:   Portfoliogewichte (normiert)
        port_returns   – Series:   monatliche Portfoliorenditen
        port_cum       – Series:   kumulierte Portfoliorendite (Basis 1, monatlich)
        port_drawdown  – Series:   Drawdown-Zeitreihe (monatlich)
        metrics_df     – DataFrame: Kennzahlen je Asset + Portfolio
        summary_df     – DataFrame: Gewichte + Metadaten je Asset
        positions_eur  – dict:     Marktwerte in EUR
        total_value    – float:    Summe Marktwerte
        start_date     – str:      Konfigurierter Startzeitpunkt
        freq           – str:      "monthly"
    """
    if data_dir is None:
        here     = Path(__file__).resolve().parent
        data_dir = here / "data" / "Old Portfolio"
    data_dir = Path(data_dir)

    # ── Rohdaten laden ────────────────────────────────────────────────────
    raw = _load_raw(data_dir)

    # ── Tägliche Preismatrix ──────────────────────────────────────────────
    frames = {}
    for name in POSITIONS_EUR:
        if name in raw:
            frames[name] = raw[name].set_index("Date")["PX_MID"]
        else:
            print(f"  ⚠️  '{name}' nicht gefunden. Verfügbare Namen: {list(raw.keys())}")

    prices_daily = (pd.DataFrame(frames)
                    .sort_index()
                    .loc[START_DATE:]
                    .ffill()
                    .dropna())

    # ── Monatliche Preise & Renditen ──────────────────────────────────────
    prices_monthly = _to_monthly(prices_daily)
    returns        = prices_monthly.pct_change().dropna()

    # German Bund (Zero Coupon 2036): override with annualised YTM of 3.074 %
    # converted to monthly frequency.
    _BUND_TICKER = "BO221256 Corp"
    _BUND_YTM_PA = 0.03074
    if _BUND_TICKER in returns.columns:
        from scipy.optimize import brentq
        _r  = returns[_BUND_TICKER].dropna()
        _n  = len(_r)
        def _geo_ann(s):
            total = (1 + _r + s).prod() - 1
            return (1 + total) ** (12 / _n) - 1
        try:
            _s = brentq(lambda s: _geo_ann(s) - _BUND_YTM_PA, -0.20, 0.20, xtol=1e-10)
        except ValueError:
            _s = _BUND_YTM_PA / 12 - _r.mean()
        returns[_BUND_TICKER] = returns[_BUND_TICKER] + _s

    print(f"\n  📅  Monatliche Renditen: {len(returns)} Monate "
          f"({returns.index[0].strftime('%b %Y')} – {returns.index[-1].strftime('%b %Y')})")

    # ── Gewichte ──────────────────────────────────────────────────────────
    avail   = {k: v for k, v in POSITIONS_EUR.items() if k in prices_daily.columns}
    total_v = sum(avail.values())
    weights = pd.Series({k: v / total_v for k, v in avail.items()})
    weights = weights.reindex(returns.columns).dropna()

    # ── Portfolio-Zeitreihen (monatlich) ──────────────────────────────────
    port_returns  = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    port_cum      = (1 + port_returns).cumprod()
    roll_max      = port_cum.cummax()
    port_drawdown = (port_cum - roll_max) / roll_max

    # ── Kennzahlen ────────────────────────────────────────────────────────
    rows = []
    for col in returns.columns:
        m          = _metrics(returns[col])
        m["asset"] = col
        m["name"]  = ASSET_META.get(col, {}).get("name", col)
        rows.append(m)

    pm          = _metrics(port_returns)
    pm["asset"] = "PORTFOLIO"
    pm["name"]  = "Portfolio Gesamt"
    rows.append(pm)
    metrics_df = pd.DataFrame(rows).set_index("asset")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_rows = []
    for ticker, eur in avail.items():
        meta = ASSET_META.get(ticker, {})
        summary_rows.append({
            "ticker": ticker,
            "name":   meta.get("name", ticker),
            "class":  meta.get("class", "–"),
            "region": meta.get("region", "–"),
            "sector": meta.get("sector", "–"),
            "eur":    eur,
            "weight": weights.get(ticker, np.nan),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("ticker")

    return {
        "prices_daily":   prices_daily,
        "prices_monthly": prices_monthly,
        "returns":        returns,          # monatlich
        "weights":        weights,
        "port_returns":   port_returns,     # monatlich
        "port_cum":       port_cum,         # monatlich
        "port_drawdown":  port_drawdown,    # monatlich
        "metrics_df":     metrics_df,
        "summary_df":     summary_df,
        "positions_eur":  avail,
        "total_value":    total_v,
        "start_date":     START_DATE,
        "freq":           "monthly",
    }
