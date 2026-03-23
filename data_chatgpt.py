"""
new_portfolio_loader.py
=======================
Liest alle Preisdaten aus data/New Portfolio (inkl. Unterordner) ein
und harmonisiert ALLE Assets auf MONATSDATEN.

Unterstützte Formate:
    - .xlsx  (Bloomberg & Simple)
    - .json  (f5 / Crypto / allgemeine Preiszeitreihen)
    - .csv   (auto-sep, auto-Spalten)

Grundprinzip:
    - Rohdaten dürfen täglich, wöchentlich oder monatlich sein
    - Für die Analyse werden ALLE Reihen auf Monatsultimo resampled
    - Danach wird der gemeinsame Zeitraum über alle Assets bestimmt
    - Renditen und Kennzahlen basieren ausschließlich auf Monatsdaten

Verwendung:
    from new_portfolio_loader import load_new_portfolio
    data = load_new_portfolio()

Benötigte Pakete:
    pip install pandas numpy openpyxl
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════
# KONFIGURATION
# ═════════════════════════════════════════════

# Optional: wenn gesetzt, wird der Start NICHT früher als dieses Datum erlaubt.
# Für rein datengetriebenen Start einfach auf None lassen.
MIN_START_DATE = None   # z. B. "2021-03-31"

# Optional: Analyseende hart begrenzen; sonst automatisch gemeinsames Ende
MAX_END_DATE = None

MONTHS_PER_YEAR = 12

# Optionale manuelle Metadaten – werden mit Scan-Daten gemergt
MANUAL_META: dict[str, dict] = {
    # "bitcoin": {"display_name": "Bitcoin", "region": "Global"},
}


# ═════════════════════════════════════════════
# READER: Bloomberg Excel
# ═════════════════════════════════════════════

def read_bloomberg_excel(filepath: Path, name: str = None) -> tuple[str, pd.DataFrame]:
    """
    Bloomberg-Format:
        Zeile 1:  Security | <name>
        Zeile 2–5: Metadaten
        Zeile 7:  Header (Date | PX_MID | PX_BID)
        Zeile 8+: Daten
    """
    meta = pd.read_excel(filepath, header=None, nrows=5, usecols=[0, 1])
    security = name or str(meta.iloc[0, 1]).strip()

    df = pd.read_excel(filepath, skiprows=6, usecols=[0, 1, 2], parse_dates=[0])
    df.columns = ["Date", "PX_MID", "PX_BID"]

    df = (
        df.dropna(subset=["Date"])
          .assign(Date=lambda x: pd.to_datetime(x["Date"], errors="coerce").dt.normalize())
          .sort_values("Date")
          .reset_index(drop=True)
    )

    df["PX_MID"] = pd.to_numeric(df["PX_MID"], errors="coerce")
    df["PX_BID"] = pd.to_numeric(df["PX_BID"], errors="coerce")

    df = df[["Date", "PX_MID"]].dropna(subset=["Date", "PX_MID"]).reset_index(drop=True)
    return security, df


# ═════════════════════════════════════════════
# READER: Simple Excel
# ═════════════════════════════════════════════

def read_simple_excel(
    filepath: Path,
    name: str = None,
    header_row: int = 2,
    date_col: str = "date",
    price_col: str = "close",
) -> tuple[str, pd.DataFrame]:
    """
    Einfaches Excel-Format:
        Zeile 1–2: leer / Metadaten
        Zeile 3:   Header → date | close
        Zeile 4+:  Daten  → 20/3/2026 | 1955,19
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
        price_raw = (
            price_raw.astype(str)
                     .str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False)
        )

    df["Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["PX_MID"] = pd.to_numeric(price_raw, errors="coerce")

    df = (
        df[["Date", "PX_MID"]]
        .dropna(subset=["Date", "PX_MID"])
        .assign(Date=lambda x: x["Date"].dt.normalize())
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return security, df


# ═════════════════════════════════════════════
# READER: JSON
# ═════════════════════════════════════════════

def read_json_fund(filepath: Path, name: str = None) -> tuple[str, pd.DataFrame]:
    """
    JSON-Format:
        {
          "slug":   "bitcoin",
          "name":   "hist-data",
          "values": [{"date": "...", "close": ...}, ...]
        }

    Security-Name Priorität:
        explizit > slug > symbol > name > Dateiname

    Daten-Array:
        values > data > prices > history > records > ohlcv > direktes Array
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        security = (
            name
            or data.get("slug")
            or data.get("symbol")
            or data.get("name")
            or filepath.stem
        )

        values = None
        for key in ("values", "data", "prices", "history", "records", "ohlcv"):
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                values = data[key]
                break

        if values is None:
            raise ValueError(f"Kein Daten-Array in: {filepath.name}. Keys: {list(data.keys())}")

    elif isinstance(data, list):
        security = name or filepath.stem
        values = data
    else:
        raise ValueError(f"Unbekannte JSON-Struktur: {filepath.name}")

    df = pd.DataFrame(values)
    if df.empty:
        raise ValueError(f"Leeres Daten-Array in: {filepath.name}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c in ("date", "datetime", "timestamp", "time")),
        df.columns[0]
    )

    price_col = next(
        (c for c in df.columns if c in ("close", "price", "last", "value", "adj_close")),
        None
    )

    if price_col is None:
        num_cols = [
            c for c in df.columns
            if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
        ]
        if not num_cols:
            raise ValueError(f"Keine Preisspalte in: {filepath.name}. Spalten: {list(df.columns)}")
        price_col = num_cols[0]

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["PX_MID"] = pd.to_numeric(df[price_col], errors="coerce")

    df = (
        df[["Date", "PX_MID"]]
        .dropna(subset=["Date", "PX_MID"])
        .assign(Date=lambda x: x["Date"].dt.normalize())
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return security, df


# ═════════════════════════════════════════════
# READER: CSV
# ═════════════════════════════════════════════

def read_csv_fund(filepath: Path, name: str = None) -> tuple[str, pd.DataFrame]:
    """
    CSV mit Preiszeitreihen – auto-erkennt Trennzeichen und Spalten.
    Unterstützt OHLCV (nimmt 'close') und Semikolon-Separator.
    """
    security = name or filepath.stem

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(filepath, sep=sep)
    if df.empty:
        raise ValueError(f"Leere CSV-Datei: {filepath.name}")

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
        num_cols = [
            c for c in df.columns
            if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
        ]
        if not num_cols:
            raise ValueError(f"Keine Preisspalte in: {filepath.name}. Spalten: {list(df.columns)}")
        price_col = num_cols[0]

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df["PX_MID"] = pd.to_numeric(df[price_col], errors="coerce")

    df = (
        df[["Date", "PX_MID"]]
        .dropna(subset=["Date", "PX_MID"])
        .assign(Date=lambda x: x["Date"].dt.normalize())
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return security, df


# ═════════════════════════════════════════════
# FORMAT-ERKENNUNG
# ═════════════════════════════════════════════

def detect_and_read(filepath: Path, name: str = None) -> tuple[str, pd.DataFrame]:
    """
    Erkennt das Dateiformat automatisch und wählt den passenden Reader:

        .json  → read_json_fund()
        .csv   → read_csv_fund()
        .xlsx  → Bloomberg oder Simple
    """
    suffix = filepath.suffix.lower()

    if suffix == ".json":
        return read_json_fund(filepath, name)

    if suffix == ".csv":
        return read_csv_fund(filepath, name)

    if suffix == ".xlsx":
        try:
            probe = pd.read_excel(filepath, header=None, nrows=5, usecols=[0])
            first_col = [str(v).strip().lower() for v in probe.iloc[:, 0].fillna("")]
            is_bloomberg = any(
                kw in val
                for val in first_col
                for kw in ("security", "start date", "end date", "period", "currency")
            )
            return read_bloomberg_excel(filepath, name) if is_bloomberg else read_simple_excel(filepath, name)
        except Exception:
            return read_bloomberg_excel(filepath, name)

    raise ValueError(f"Nicht unterstütztes Dateiformat: {filepath.suffix}")


# ═════════════════════════════════════════════
# DIAGNOSE: ROH-FREQUENZ
# ═════════════════════════════════════════════

def detect_series_spacing(df: pd.DataFrame) -> str:
    """
    Erkennt grob die Roh-Frequenz anhand typischer Datumsabstände.
    Nur zu Diagnosezwecken – die Analyse wird trotzdem immer monatlich gemacht.
    """
    d = (
        df["Date"]
        .sort_values()
        .drop_duplicates()
        .diff()
        .dropna()
        .dt.days
    )

    if d.empty:
        return "unknown"

    median_gap = d.median()

    if median_gap <= 3:
        return "daily"
    if median_gap <= 10:
        return "weekly"
    return "monthly_or_lower"


# ═════════════════════════════════════════════
# FOLDER SCAN
# ═════════════════════════════════════════════

def scan_folder(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    Durchsucht root + alle Unterordner nach .xlsx / .json / .csv.
    Unterordnername wird als Asset-Klasse verwendet.

    Returns:
        raw_data  – { security_name: DataFrame[Date, PX_MID] }
        meta      – { security_name: { asset_class, file, ... } }
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

    raw_data: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}

    for filepath in files:
        subfolder = filepath.parent.name if filepath.parent != root else "Other"

        try:
            security, df = detect_and_read(filepath)

            raw_data[security] = df
            meta[security] = {
                "asset_class": subfolder,
                "file": filepath.name,
                "display_name": MANUAL_META.get(security, {}).get("display_name", security),
            }

            if security in MANUAL_META:
                meta[security].update(MANUAL_META[security])

            freq_label = detect_series_spacing(df)

            print(
                f"  ✅  [{subfolder:12s}]  "
                f"{security:40s}  "
                f"{len(df):>5} Datenpunkte  "
                f"({freq_label})"
            )

        except Exception as e:
            print(f"  ❌  [{subfolder:12s}]  {filepath.name}: {e}")

    print(f"\n  → {len(raw_data)} Assets erfolgreich geladen")

    if not raw_data:
        raise ValueError("Es konnte kein Asset erfolgreich geladen werden.")

    return raw_data, meta


# ═════════════════════════════════════════════
# MONATLICHE PREISMATRIX
# ═════════════════════════════════════════════

def build_monthly_price_matrix(
    raw_data: dict[str, pd.DataFrame],
    start_date: str | None = MIN_START_DATE,
    end_date: str | None = MAX_END_DATE,
) -> pd.DataFrame:
    """
    Harmonisiert alle Rohreihen auf Monatsultimo und bestimmt danach
    den gemeinsamen Zeitraum über alle Assets.

    Vorgehen:
        1. Jedes Asset auf Monatsultimo resamplen (letzter verfügbarer Wert im Monat)
        2. Alle Assets zu einer Matrix zusammenführen
        3. Optionalen externen Start-/End-Cut anwenden
        4. Gemeinsamen Start = spätestes erstes gültiges Monatsdatum
        5. Gemeinsames Ende = frühestes letztes gültiges Monatsdatum
        6. Nur vollständige Beobachtungen behalten
    """
    monthly_series = {}

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
        monthly_series[name] = s

    prices_monthly = pd.DataFrame(monthly_series).sort_index()

    if start_date is not None:
        prices_monthly = prices_monthly.loc[pd.to_datetime(start_date):]

    if end_date is not None:
        prices_monthly = prices_monthly.loc[:pd.to_datetime(end_date)]

    first_valids = {c: prices_monthly[c].first_valid_index() for c in prices_monthly.columns}
    last_valids = {c: prices_monthly[c].last_valid_index() for c in prices_monthly.columns}

    valid_starts = [v for v in first_valids.values() if v is not None]
    valid_ends = [v for v in last_valids.values() if v is not None]

    if not valid_starts or not valid_ends:
        return pd.DataFrame()

    common_start = max(valid_starts)
    common_end = min(valid_ends)

    if common_start > common_end:
        return pd.DataFrame()

    prices_monthly = prices_monthly.loc[common_start:common_end]

    # Diagnose vor finalem Drop
    missing_assets = prices_monthly.columns[prices_monthly.isna().all()].tolist()
    partially_missing_assets = prices_monthly.columns[prices_monthly.isna().any()].tolist()

    if missing_assets:
        print("\n⚠️  Komplett fehlend im gemeinsamen Zeitraum:")
        for asset in missing_assets:
            print(f"    – {asset}")

    if partially_missing_assets:
        print("\nℹ️  Unvollständige Monatswerte im gemeinsamen Zeitraum:")
        for asset in partially_missing_assets:
            first = prices_monthly[asset].first_valid_index()
            last = prices_monthly[asset].last_valid_index()
            print(f"    – {asset}  (erste gültige: {first.date() if first is not None else 'n/a'}, letzte gültige: {last.date() if last is not None else 'n/a'})")

    prices_monthly = (
        prices_monthly
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="any")
    )

    return prices_monthly


# ═════════════════════════════════════════════
# KENNZAHLEN
# ═════════════════════════════════════════════

def compute_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Annualisierte Kennzahlen je Asset auf Basis MONATLICHER Renditen.
    """
    rows = []

    for col in returns.columns:
        r = returns[col].dropna()

        if r.empty:
            continue

        cum = (1 + r).cumprod()
        total = cum.iloc[-1] - 1
        ann_r = (1 + total) ** (MONTHS_PER_YEAR / len(r)) - 1
        ann_v = r.std() * np.sqrt(MONTHS_PER_YEAR)
        sharpe = (ann_r / ann_v) if ann_v > 0 else np.nan

        neg = r[r < 0]
        ds = neg.std() * np.sqrt(MONTHS_PER_YEAR) if len(neg) > 0 else np.nan
        sortino = (ann_r / ds) if pd.notna(ds) and ds > 0 else np.nan

        roll = cum.cummax()
        max_dd = ((cum - roll) / roll).min()

        tail = r[r <= r.quantile(0.05)]
        cvar = tail.mean() if not tail.empty else np.nan

        rows.append({
            "asset": col,
            "ann_return": round(ann_r * 100, 2),
            "ann_vol": round(ann_v * 100, 2),
            "sharpe": round(sharpe, 3) if pd.notna(sharpe) else np.nan,
            "sortino": round(sortino, 3) if pd.notna(sortino) else np.nan,
            "max_drawdown": round(max_dd * 100, 2),
            "cvar_95": round(cvar * 100, 2) if pd.notna(cvar) else np.nan,
            "total_return": round(total * 100, 2),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ann_return", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "cvar_95", "total_return"
        ])

    return pd.DataFrame(rows).set_index("asset").sort_index()


# ═════════════════════════════════════════════
# HAUPTFUNKTION
# ═════════════════════════════════════════════

def load_new_portfolio(
    data_dir: str | Path = None,
    start_date: str | None = MIN_START_DATE,
    end_date: str | None = MAX_END_DATE,
) -> dict:
    """
    Lädt alle Assets aus data/New Portfolio und gibt zurück:

        prices_monthly – monatliche Preismatrix (gemeinsamer Zeitraum)
        returns        – monatliche Renditen
        metrics_df     – annualisierte Kennzahlen je Asset
        asset_meta     – { security: { asset_class, file, ... } }
        asset_classes  – { asset_class: [securities] }
        start_date     – verwendetes Startdatum
        end_date       – verwendetes Enddatum
        freq           – immer "monthly"
    """
    if data_dir is None:
        here = Path(__file__).resolve().parent
        data_dir = here / "data" / "New Portfolio"

    data_dir = Path(data_dir)

    # 1) Dateien scannen
    raw_data, asset_meta = scan_folder(data_dir)

    # 2) Monatliche konsistente Matrix
    prices_monthly = build_monthly_price_matrix(
        raw_data=raw_data,
        start_date=start_date,
        end_date=end_date,
    )

    if prices_monthly.empty:
        raise ValueError(
            "Kein gemeinsamer monatlicher Zeitraum über alle Assets gefunden. "
            "Bitte Start-/Enddatum prüfen oder Assets mit zu kurzer Historie entfernen."
        )

    # 3) Renditen & Kennzahlen
    returns = prices_monthly.pct_change().dropna(how="any")
    metrics_df = compute_metrics(returns)

    # 4) Asset-Klassen-Mapping nur für tatsächlich enthaltene Assets
    asset_classes: dict[str, list[str]] = {}
    for sec, m in asset_meta.items():
        if sec not in prices_monthly.columns:
            continue
        cls = m.get("asset_class", "Other")
        asset_classes.setdefault(cls, []).append(sec)

    # Sortierung
    asset_classes = {k: sorted(v) for k, v in sorted(asset_classes.items())}

    print(
        f"\n📅  Monatlicher gemeinsamer Zeitraum: "
        f"{prices_monthly.index[0].date()} → {prices_monthly.index[-1].date()}  "
        f"({len(prices_monthly)} Monate)"
    )
    print(f"📈  {len(prices_monthly.columns)} Assets im gemeinsamen Zeitraum\n")

    print("📊  Asset-Klassen:")
    for cls, assets in asset_classes.items():
        print(f"    {cls:15s}  {len(assets):>3} Assets")

    return {
        "prices_monthly": prices_monthly,
        "returns": returns,
        "metrics_df": metrics_df,
        "asset_meta": asset_meta,
        "asset_classes": asset_classes,
        "start_date": str(prices_monthly.index[0].date()),
        "end_date": str(prices_monthly.index[-1].date()),
        "freq": "monthly",
    }


# ═════════════════════════════════════════════
# DIREKTER AUFRUF
# ═════════════════════════════════════════════

if __name__ == "__main__":
    data = load_new_portfolio()

    print("\n" + "─" * 72)
    print("KENNZAHLEN (annualisiert, auf Basis monatlicher Renditen)")
    print("─" * 72)
    print(data["metrics_df"].to_string())