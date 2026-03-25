"""
new_portfolio_loader.py
=======================
Liest alle Preisdaten aus data/New Portfolio (Unterordner) ein.
Unterstützte Formate: .xlsx (Bloomberg & Simple), .json (f5/crypto), .csv

Format-Erkennung:
    .json  → slug > name > Dateiname  (crypto: "slug":"bitcoin" statt "name":"hist-data")
    .csv   → auto-sep, auto-Spalten
    .xlsx  → Bloomberg (Security-Header) oder Simple (date|close)

Verwendung:
    from new_portfolio_loader import load_new_portfolio
    data = load_new_portfolio()

Pakete: pip install pandas numpy openpyxl
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ═════════════════════════════════════════════
# KONFIGURATION
# ═════════════════════════════════════════════

START_DATE     = "2021-03-19"
MONTHS_PER_YEAR = 12

# Optionale manuelle Metadaten – werden mit Scan-Daten gemergt
MANUAL_META = {  # type: Dict[str, dict]
    # "bitcoin": {"display_name": "Bitcoin", "region": "Global"},
}


# ═════════════════════════════════════════════
# READER: Bloomberg Excel
# ═════════════════════════════════════════════

def read_bloomberg_excel(filepath, name=None):  # (Path, str) -> Tuple[str, pd.DataFrame]
    """
    Bloomberg-Format:
        Zeile 1:  Security | <name>
        Zeile 2–5: Metadaten
        Zeile 7:  Header (Date | PX_MID | PX_BID)
        Zeile 8+: Daten
    """
    meta     = pd.read_excel(filepath, header=None, nrows=5, usecols=[0, 1])
    security = name or str(meta.iloc[0, 1]).strip()

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


# ═════════════════════════════════════════════
# READER: Simple Excel (Catholic Index etc.)
# ═════════════════════════════════════════════

def read_simple_excel(filepath: Path, name: str = None,
                       header_row: int = 2,
                       date_col: str = "date",
                       price_col="close"):  # str -> Tuple[str, pd.DataFrame]
    """
    Einfaches Excel-Format:
        Zeile 1–2: leer / Metadaten
        Zeile 3:   Header → date | close
        Zeile 4+:  Daten  → 20/3/2026 | 1955,19  (europäisches Dezimalformat OK)
    """
    security = name or filepath.stem

    df = pd.read_excel(filepath, header=header_row)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if date_col not in df.columns:
        raise ValueError(f"Spalte '{date_col}' nicht gefunden. Verfügbar: {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"Spalte '{price_col}' nicht gefunden. Verfügbar: {list(df.columns)}")

    price_raw = df[price_col]
    if price_raw.dtype == object:
        price_raw = (price_raw.astype(str)
                     .str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False))

    df["Date"]   = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["PX_MID"] = pd.to_numeric(price_raw, errors="coerce")

    df = (df[["Date", "PX_MID"]]
          .dropna(subset=["Date", "PX_MID"])
          .assign(Date=lambda x: x["Date"].dt.normalize())
          .sort_values("Date")
          .reset_index(drop=True))
    return security, df


# ═════════════════════════════════════════════
# READER: JSON  (f5-Fonds & Crypto)
# ═════════════════════════════════════════════

def read_json_fund(filepath, name=None):  # (Path, str) -> Tuple[str, pd.DataFrame]
    """
    JSON-Format (f5-Fonds & Crypto CMC):
        {
          "slug":   "bitcoin",          ← wird als Security-Name bevorzugt
          "name":   "hist-data",        ← oft generisch, daher zweite Wahl
          "values": [{"date": "...", "close": ...}, ...]
        }

    Security-Name Priorität: explizit > slug > name > symbol > Dateiname
    Daten-Array:  values > data > prices > history > records > ohlcv > direktes Array
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # slug zuerst – bei Crypto-JSONs ist "name" oft generisch ("hist-data")
        security = (name
                    or data.get("slug")
                    or data.get("symbol")
                    or data.get("name")
                    or filepath.stem)

        # Daten-Array suchen
        values = None
        for key in ("values", "data", "prices", "history", "records", "ohlcv"):
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                values = data[key]
                break
        if values is None:
            raise ValueError(f"Kein Daten-Array in: {filepath.name}. Keys: {list(data.keys())}")

    elif isinstance(data, list):
        security = name or filepath.stem
        values   = data
    else:
        raise ValueError(f"Unbekannte JSON-Struktur: {filepath.name}")

    df = pd.DataFrame(values)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Datumsspalte
    date_col = next(
        (c for c in df.columns if c in ("date", "datetime", "timestamp", "time")),
        df.columns[0]
    )
    # Preisspalte
    price_col = next(
        (c for c in df.columns if c in ("close", "price", "last", "value", "adj_close")),
        None
    )
    if price_col is None:
        num_cols = [c for c in df.columns if c != date_col
                    and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if not num_cols:
            raise ValueError(f"Keine Preisspalte in: {filepath.name}. Spalten: {list(df.columns)}")
        price_col = num_cols[0]

    df["Date"]   = pd.to_datetime(df[date_col], errors="coerce")
    df["PX_MID"] = pd.to_numeric(df[price_col], errors="coerce")

    df = (df[["Date", "PX_MID"]]
          .dropna(subset=["Date", "PX_MID"])
          .assign(Date=lambda x: x["Date"].dt.normalize())
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .reset_index(drop=True))
    return security, df


# ═════════════════════════════════════════════
# READER: CSV
# ═════════════════════════════════════════════

def read_csv_fund(filepath, name=None):  # (Path, str) -> Tuple[str, pd.DataFrame]
    """
    CSV mit Preiszeitreihen – auto-erkennt Trennzeichen und Spalten.
    Unterstützt OHLCV (nimmt 'close') und Semikolon-Separator.
    """
    security = name or filepath.stem

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(filepath, sep=sep)
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c in ("date", "datetime", "timestamp", "time", "day")),
        df.columns[0]
    )
    price_col = next(
        (c for c in df.columns if c in ("close", "price", "last", "adj close", "adj_close", "value")),
        None
    )
    if price_col is None:
        num_cols = [c for c in df.columns if c != date_col
                    and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if not num_cols:
            raise ValueError(f"Keine Preisspalte in: {filepath.name}. Spalten: {list(df.columns)}")
        price_col = num_cols[0]

    df["Date"]   = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df["PX_MID"] = pd.to_numeric(df[price_col], errors="coerce")

    df = (df[["Date", "PX_MID"]]
          .dropna(subset=["Date", "PX_MID"])
          .assign(Date=lambda x: x["Date"].dt.normalize())
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .reset_index(drop=True))
    return security, df


# ═════════════════════════════════════════════
# FORMAT-ERKENNUNG
# ═════════════════════════════════════════════


# ═════════════════════════════════════════════
# READER: LSEG / Refinitiv Zeitreihen-Export
# ═════════════════════════════════════════════

def read_lseg_excel(filepath, name=None):  # (Path, str) -> Tuple[str, pd.DataFrame]
    """
    LSEG / Refinitiv Workspace Zeitreihen-Export:

        Zeile 1:  "Hist. Time Series – <Name>"
        Zeile 2:  Ric: | <RIC>
        Zeile 3:  Period: | <von> – <bis>
        Zeile 4:  Periodicity: | Daily / Monthly
        Zeile 5:  (leer)
        Zeile 6:  Header → Date | Open | High | Low | Close | Yield | Bid | Ask | ...
        Zeile 7+: Daten

    Security-Name: aus Zeile 2 (Ric), z.B. "SYBC.DE".
    Preis:  Close-Spalte (Index 4 in Standard-Export, Spalte E).
    """
    meta = pd.read_excel(filepath, header=None, nrows=5, usecols=[0, 1])

    # Prefer RIC from row 2 col B as security name
    ric = str(meta.iloc[1, 1]).strip() if not pd.isna(meta.iloc[1, 1]) else None
    security = name or ric or filepath.stem

    df = pd.read_excel(filepath, skiprows=5, header=0)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Date column: first column
    date_col = df.columns[0]
    df["Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    # Close column (required)
    if "close" not in df.columns:
        raise ValueError(
            f"LSEG reader: 'close' column not found in {filepath.name}. "
            f"Available: {list(df.columns)}"
        )
    df["PX_MID"] = pd.to_numeric(df["close"], errors="coerce")

    df = (df[["Date", "PX_MID"]]
          .dropna(subset=["Date", "PX_MID"])
          .assign(Date=lambda x: x["Date"].dt.normalize())
          .sort_values("Date")
          .reset_index(drop=True))
    return security, df


def detect_and_read(filepath, name=None):  # (Path, str) -> Tuple[str, pd.DataFrame]
    """
    Erkennt das Dateiformat automatisch und wählt den passenden Reader:

        .json  → read_json_fund()     (slug > name für Security-Name)
        .csv   → read_csv_fund()
        .xlsx  → Bloomberg (hat Security/Start Date Header) oder Simple (date|close)
    """
    suffix = filepath.suffix.lower()

    if suffix == ".json":
        return read_json_fund(filepath, name)

    if suffix == ".csv":
        return read_csv_fund(filepath, name)

    # .xlsx: Bloomberg vs. LSEG vs. Simple unterscheiden
    try:
        probe     = pd.read_excel(filepath, header=None, nrows=6, usecols=[0, 1])
        first_col = [str(v).strip().lower() for v in probe.iloc[:, 0].fillna("")]

        # Bloomberg: has "security" / "start date" / "end date" in first 3 rows
        is_bloomberg = any(
            kw in val
            for val in first_col[:3]
            for kw in ("security", "start date", "end date")
        )
        if is_bloomberg:
            return read_bloomberg_excel(filepath, name)

        # LSEG/Refinitiv: has "ric:" or "period:" or "periodicity:" in first 5 rows
        is_lseg = any(
            kw in val
            for val in first_col[:5]
            for kw in ("ric:", "period:", "periodicity:", "hist. time series")
        )
        if is_lseg:
            return read_lseg_excel(filepath, name)

        # Fallback: Simple (date | close)
        return read_simple_excel(filepath, name)
    except Exception:
        return read_bloomberg_excel(filepath, name)


# ═════════════════════════════════════════════
# FOLDER SCAN
# ═════════════════════════════════════════════

def scan_folder(root):  # (Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]
    """
    Durchsucht root + alle Unterordner nach .xlsx / .json / .csv.
    Unterordner-Name wird als Asset-Klasse verwendet.

    Returns:
        raw_data  – { security_name: DataFrame[Date, PX_MID] }
        meta      – { security_name: { asset_class, file } }
    """
    files = sorted(
        list(root.rglob("*.xlsx")) +
        list(root.rglob("*.json")) +
        list(root.rglob("*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"Keine Dateien (.xlsx/.json/.csv) in: {root}")

    print(f"\n📂  Scanne: {root}")
    print(f"    {len(files)} Dateien gefunden (.xlsx / .json / .csv)\n")

    raw_data = {}  # type: Dict[str, pd.DataFrame]
    meta:     dict[str, dict]         = {}

    for filepath in files:
        subfolder = filepath.parent.name if filepath.parent != root else "Other"
        try:
            security, df = detect_and_read(filepath)

            # Manuelle Metadaten mergen falls vorhanden
            if security in MANUAL_META:
                display = MANUAL_META[security].get("display_name", security)
            else:
                display = security

            raw_data[security] = df
            meta[security] = {"asset_class": subfolder, "file": filepath.name}
            if security in MANUAL_META:
                meta[security].update(MANUAL_META[security])

            print(f"  ✅  [{subfolder:12s}]  {security:40s}  {len(df):>5} Datenpunkte")

        except Exception as e:
            print(f"  ❌  [{subfolder:12s}]  {filepath.name}: {e}")

    print(f"\n  → {len(raw_data)} Assets erfolgreich geladen")
    return raw_data, meta


# ═════════════════════════════════════════════
# PREISMATRIX BAUEN
# ═════════════════════════════════════════════

def build_price_matrix(raw_data,
                        start_date=START_DATE):  # -> Tuple[pd.DataFrame, pd.DataFrame]
    """
    Baut tägliche und monatliche Preismatrizen.
    Gibt Diagnose aus wenn Assets Lücken haben.
    """
    frames = {name: df.set_index("Date")["PX_MID"] for name, df in raw_data.items()}

    all_prices = (pd.DataFrame(frames)
                  .sort_index()
                  .loc[start_date:]
                  .ffill())

    # Diagnose
    missing    = all_prices.columns[all_prices.isna().all()].tolist()
    incomplete = all_prices.columns[all_prices.isna().any()].tolist()

    if missing:
        print(f"\n  ⚠️  Keine Daten ab {start_date} → entfernt:")
        for m in missing:
            print(f"       – {m}")
    if incomplete:
        print(f"\n  ℹ️  Frühe Lücken (ffill angewendet):")
        for m in incomplete:
            first = all_prices[m].first_valid_index()
            print(f"       – {m}  (ab {first.date() if first else 'n/a'})")

    prices_daily = all_prices.dropna(axis=1, how="all").dropna(axis=0, how="any")

    # Fallback: gemeinsamen Startzeitpunkt suchen
    if prices_daily.empty:
        first_valids = {c: all_prices[c].first_valid_index() for c in all_prices.columns}
        common_start = max(v for v in first_valids.values() if v is not None)
        print(f"\n  ℹ️  Kein Zeitraum ab {start_date} → verwende {common_start.date()}")
        prices_daily = (all_prices
                        .loc[common_start:]
                        .dropna(axis=1, how="all")
                        .dropna(axis=0, how="any"))

    prices_monthly = prices_daily.resample("ME").last().dropna()
    return prices_daily, prices_monthly


# ═════════════════════════════════════════════
# KENNZAHLEN
# ═════════════════════════════════════════════

def compute_metrics(returns: pd.DataFrame, freq: str = "monthly") -> pd.DataFrame:
    """Annualisierte Kennzahlen je Asset. freq: 'monthly' (×12) oder 'daily' (×252)."""
    scale = 12 if freq == "monthly" else 252
    rows  = []

    for col in returns.columns:
        r       = returns[col].dropna()
        cum     = (1 + r).cumprod()
        total   = cum.iloc[-1] - 1
        ann_r   = (1 + total) ** (scale / len(r)) - 1
        ann_v   = r.std() * np.sqrt(scale)
        sharpe  = (ann_r / ann_v) if ann_v > 0 else np.nan
        neg     = r[r < 0]
        ds      = neg.std() * np.sqrt(scale) if len(neg) > 0 else np.nan
        sortino = (ann_r / ds) if ds and ds > 0 else np.nan
        roll    = cum.cummax()
        max_dd  = ((cum - roll) / roll).min()
        cvar    = r[r <= r.quantile(0.05)].mean()

        rows.append({
            "asset":        col,
            "ann_return":   round(ann_r  * 100, 2),
            "ann_vol":      round(ann_v  * 100, 2),
            "sharpe":       round(sharpe, 3),
            "sortino":      round(sortino, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "cvar_95":      round(cvar   * 100, 2),
            "total_return": round(total  * 100, 2),
        })

    return pd.DataFrame(rows).set_index("asset")


# ═════════════════════════════════════════════
# HAUPTFUNKTION
# ═════════════════════════════════════════════

def load_new_portfolio(
    data_dir=None,  # str or Path
    start_date: str = START_DATE,
    freq: str = "monthly",
) -> dict:
    """
    Lädt alle Assets aus data/New Portfolio und gibt zurück:

        prices_daily   – tägliche Kurse (für Charts)
        prices_monthly – monatliche Kurse (letzter Handelstag je Monat)
        returns        – Renditen in gewählter Frequenz
        metrics_df     – Kennzahlen je Asset
        asset_meta     – { security: { asset_class, file } }
        asset_classes  – { asset_class: [securities] }
        start_date     – verwendetes Startdatum
        freq           – "monthly" oder "daily"
    """
    if data_dir is None:
        here     = Path(__file__).resolve().parent
        data_dir = here / "data" / "New Portfolio"
    data_dir = Path(data_dir)

    # 1. Alle Dateien scannen
    raw_data, asset_meta = scan_folder(data_dir)

    # 2. Preismatrizen
    prices_daily, prices_monthly = build_price_matrix(raw_data, start_date)

    print(f"\n📅  Täglich:   {prices_daily.index[0].date()} → {prices_daily.index[-1].date()}"
          f"  ({len(prices_daily)} Tage)")
    print(f"📅  Monatlich: {prices_monthly.index[0].date()} → {prices_monthly.index[-1].date()}"
          f"  ({len(prices_monthly)} Monate)")
    print(f"📈  {len(prices_daily.columns)} Assets im gemeinsamen Zeitraum\n")

    # 3. Renditen & Kennzahlen
    prices  = prices_monthly if freq == "monthly" else prices_daily
    returns = prices.pct_change().dropna()

    # German Bund: shift return series so geometric annualised return = YTM.
    _BUND_TICKER = "BO221256 Corp"
    _BUND_YTM_PA = 0.03074
    if _BUND_TICKER in returns.columns:
        from scipy.optimize import brentq
        _r   = returns[_BUND_TICKER].dropna()
        _n   = len(_r)
        _scale = 12 if freq == "monthly" else 252
        def _geo_ann(s):
            total = (1 + _r + s).prod() - 1
            return (1 + total) ** (_scale / _n) - 1
        try:
            _s = brentq(lambda s: _geo_ann(s) - _BUND_YTM_PA, -0.20, 0.20, xtol=1e-10)
        except ValueError:
            _s = _BUND_YTM_PA / _scale - _r.mean()
        returns[_BUND_TICKER] = returns[_BUND_TICKER] + _s

    metrics_df = compute_metrics(returns, freq)

    # 4. Asset-Klassen-Mapping
    asset_classes = {}  # type: Dict[str, list]
    for sec, m in asset_meta.items():
        if sec not in prices_daily.columns:
            continue
        cls = m.get("asset_class", "Other")
        asset_classes.setdefault(cls, []).append(sec)

    print("📊  Asset-Klassen:")
    for cls, assets in sorted(asset_classes.items()):
        print(f"    {cls:15s}  {len(assets):>3} Assets")

    return {
        "prices_daily":   prices_daily,
        "prices_monthly": prices_monthly,
        "returns":        returns,
        "metrics_df":     metrics_df,
        "asset_meta":     asset_meta,
        "asset_classes":  asset_classes,
        "start_date":     start_date,
        "freq":           freq,
    }


# ═════════════════════════════════════════════
# DIREKTER AUFRUF
# ═════════════════════════════════════════════

if __name__ == "__main__":
    data = load_new_portfolio()

    print("\n" + "─" * 65)
    print("KENNZAHLEN (annualisiert)")
    print("─" * 65)
    print(data["metrics_df"].to_string())
