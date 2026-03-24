"""
optimization_dashboard.py  –  New Portfolio Optimizer
======================================================
Kompatibel mit new_portfolio_loader.py (nur prices_monthly).

Portfolios:  Utility · Max Sortino · Min CVaR

Ablegen: dashboard/pages/optimization_dashboard.py
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
from scipy.optimize import minimize

from data_chatgpt import load_new_portfolio
from asset_metadaten import get_df as get_meta_df, esg_label


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONSTRAINT DEFAULTS  –  hier zentral anpassen                             ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Alle Werte sind Brüche (0.05 = 5 %).                                      ║
# ║  Die Sidebar-Slider starten mit diesen Werten, können aber zur Laufzeit    ║
# ║  überschrieben werden.                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DEFAULT_MIN_W           = 0.00   # Mindestgewicht je Asset           (0 % = kein Mindestgewicht)
DEFAULT_MAX_W           = 0.35   # Maximalgewicht je Asset           (35 %)
DEFAULT_CRYPTO_MAX      = 0.05   # Bitcoin / Crypto gesamt           ≤  5 %
DEFAULT_SINGLE_STOCK_MAX= 0.10   # Einzelner Aktientitel             ≤ 10 %
DEFAULT_STOCKS_TOTAL_MAX= 0.60   # Alle Stocks zusammen             ≤ 40 %
DEFAULT_SECTOR_MAX      = 0.70   # Eine einzelne Branche            ≤ 70 %
DEFAULT_REGION_MAX      = 0.80   # Eine einzelne Region             ≤ 80 %
DEFAULT_ESG_MIN         = 0      # ESG-Mindest-Score  (0 = kein Filter)

# ── Lookback & Split ──────────────────────────────────────────────────────────
DEFAULT_LOOKBACK_MONTHS = 60     # Lookback-Fenster in Monaten
DEFAULT_TEST_PCT        = 0.30   # Anteil Out-of-Sample (30 %)
DEFAULT_LAMBDA          = 4.0    # Risikoaversionsparameter λ


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="⚡",
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
[data-testid="stSidebar"] { background: #0F172A !important; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; color: #94A3B8 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] .stMarkdown p { color: #CBD5E1 !important; }
[data-testid="metric-container"] {
    background: white; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748B !important; font-size: .65rem !important;
    letter-spacing: .1em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #0F172A !important; font-size: 1.25rem !important; font-weight: 700;
}
.chip {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.58rem; letter-spacing:.15em; text-transform:uppercase;
    color:#3B82F6; border:1px solid #BFDBFE; background:#EFF6FF;
    border-radius:4px; padding:2px 8px; margin-bottom:.5rem;
}
.chip-warn {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.58rem; letter-spacing:.15em; text-transform:uppercase;
    color:#D97706; border:1px solid #FDE68A; background:#FFFBEB;
    border-radius:4px; padding:2px 8px; margin:0 4px 4px 0;
}
.chip-green {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.58rem; letter-spacing:.15em; text-transform:uppercase;
    color:#059669; border:1px solid #A7F3D0; background:#ECFDF5;
    border-radius:4px; padding:2px 8px; margin:0 4px 4px 0;
}
hr { border-color: #E2E8F0 !important; margin: 1.5rem 0; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RF_ANNUAL       = 0.02017
MONTHS_PER_YEAR = 12
ALPHA           = 0.95

COLORS = [
    "#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
    "#06B6D4","#F97316","#84CC16","#EC4899","#14B8A6",
    "#A78BFA","#FB923C","#34D399","#60A5FA","#FBBF24",
]

PORTFOLIO_COLORS = {
    "Equal Weight": "#94A3B8",
    "Utility":      "#2563EB",
    "Max Sortino":  "#10B981",
    "Min CVaR":     "#F59E0B",
}

def _base_layout(title="", h=360, extra=None):
    lo = dict(
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
        lo.update(extra)
    return lo

def make_fig(title="", h=360, extra=None):
    f = go.Figure()
    f.update_layout(**_base_layout(title, h, extra))
    return f


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Portfolio-Daten …")
def get_data():
    return load_new_portfolio(
        PROJECT_ROOT / "data" / "New Portfolio",
    )

data           = get_data()
prices_monthly = data["prices_monthly"]   # DataFrame[Date × Asset]
asset_meta_ldr = data["asset_meta"]       # {asset: {asset_class, file}}
asset_classes  = data["asset_classes"]    # {class: [assets]}
ALL_ASSETS     = sorted(prices_monthly.columns.tolist())
ASSET_COLORS   = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ALL_ASSETS)}

# Metadaten aus asset_metadata.py
META_DF = get_meta_df()   # Index = Asset-Name

def get_full_meta(asset: str) -> dict:
    """Kombiniert Loader-Meta (asset_class) + asset_metadata.py (sektor/region/esg)."""
    m = dict(asset_meta_ldr.get(asset, {}))
    if asset in META_DF.index:
        row = META_DF.loc[asset]
        m["sektor"]    = row.get("sektor",    "–")
        m["region"]    = row.get("region",    "–")
        m["esg_score"] = row.get("esg_score", None)
    else:
        m.setdefault("sektor",    "–")
        m.setdefault("region",    "–")
        m.setdefault("esg_score", None)
    return m

if prices_monthly.empty:
    st.error("Keine monatlichen Preisdaten verfügbar.")
    st.stop()

ALL_REGIONS = sorted({get_full_meta(a).get("region", "–") for a in ALL_ASSETS})
ALL_SECTORS = sorted({get_full_meta(a).get("sektor", "–") for a in ALL_ASSETS})
ALL_CLASSES = sorted(asset_classes.keys())


# ─────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────
def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty: return np.nan
    return (1 + r).prod() ** (MONTHS_PER_YEAR / len(r)) - 1

def ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    return np.nan if r.empty else r.std() * np.sqrt(MONTHS_PER_YEAR)

def cvar_loss(r: pd.Series, alpha: float = ALPHA) -> float:
    r = r.dropna()
    if r.empty: return np.nan
    tail = r[r <= r.quantile(1 - alpha)]
    return np.nan if tail.empty else -tail.mean()

def sortino(r: pd.Series, rf: float = RF_ANNUAL) -> float:
    r = r.dropna()
    if r.empty: return np.nan
    rf_m     = (1 + rf) ** (1 / MONTHS_PER_YEAR) - 1
    excess   = r - rf_m
    down     = excess[excess < 0]
    if down.empty: return np.nan
    dd = down.std() * np.sqrt(MONTHS_PER_YEAR)
    return np.nan if dd <= 0 else (ann_return(r) - rf) / dd

def port_ret(w: np.ndarray, X: pd.DataFrame) -> pd.Series:
    return pd.Series(X.values @ w, index=X.index)

def portfolio_metrics(r: pd.Series, weights: dict, label: str) -> dict:
    r = r.dropna()
    if r.empty:
        return dict(label=label, ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
                    sortino=np.nan, max_dd=np.nan, cvar_95=np.nan, total_return=np.nan,
                    weights=weights, returns=r, cumulative=pd.Series(dtype=float))
    cum  = (1 + r).cumprod()
    ar   = ann_return(r); av = ann_vol(r)
    roll = cum.cummax()
    return dict(
        label=label, ann_return=ar, ann_vol=av,
        sharpe=(ar - RF_ANNUAL) / av if av and av > 0 else np.nan,
        sortino=sortino(r),
        max_dd=((cum - roll) / roll).min(),
        cvar_95=-cvar_loss(r),
        total_return=cum.iloc[-1] - 1,
        weights=weights, returns=r, cumulative=cum,
    )

def dd_series(cum: pd.Series) -> pd.Series:
    return (cum - cum.cummax()) / cum.cummax() * 100

def split(X: pd.DataFrame, test_pct: float):
    idx = max(int(len(X) * (1 - test_pct)), 12)
    idx = min(idx, len(X) - 3)
    return X.iloc[:idx].copy(), X.iloc[idx:].copy()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONSTRAINT BUILDER                                                         ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Jeder Constraint ist ein einzelner Eintrag in der cons-Liste.             ║
# ║  Alle "ineq"-Constraints müssen ≥ 0 sein (scipy-Konvention):              ║
# ║      cap - sum(w[i] for i in gruppe) ≥ 0   ↔   sum ≤ cap                 ║
# ║                                                                             ║
# ║  Harte Ausschlüsse (ESG, Region, Sektor) werden über bounds gelöst:       ║
# ║      bounds[i] = (0, 0)  →  Solver darf diesem Asset kein Gewicht geben   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def build_constraints(
    assets: list,
    min_w: float, max_w: float,
    crypto_max: float,        # Bitcoin / Crypto gesamt
    single_stock_max: float,  # ein einzelner Aktientitel
    stocks_total_max: float,  # alle Stocks zusammen
    sector_max: float,        # eine einzelne Branche
    region_max: float,        # eine einzelne Region
    esg_min: int,             # ESG-Mindest-Score (harter Ausschluss)
) -> tuple[list, list, set]:
    """
    Gibt (bounds, cons, excluded) zurück.

    bounds  – scipy-Bounds: [(lo, hi)] je Asset
    cons    – scipy-Constraint-Liste
    excluded – Set der hart ausgeschlossenen Asset-Namen (ESG < esg_min)
    """
    n        = len(assets)
    bounds   = [(min_w, max_w)] * n   # Standardbounds für alle Assets
    cons     = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]   # sum(w) == 1
    excluded = set()

    # ── Gruppen-Indizes ──────────────────────────────────────────────────────
    crypto_idx: list[int]         = []   # Crypto-Assets
    stock_idx:  list[int]         = []   # einzelne Aktien
    by_sector:  dict[str, list]   = {}   # Sektor → [Asset-Indizes]
    by_region:  dict[str, list]   = {}   # Region → [Asset-Indizes]

    for i, a in enumerate(assets):
        m      = get_full_meta(a)
        cls    = m.get("asset_class", "Other").lower()
        region = m.get("region",    "–")
        sektor = m.get("sektor",    "–")
        esg    = m.get("esg_score", None)

        # ── Harter Ausschluss via ESG-Score ─────────────────────────────────
        if esg is not None and esg < esg_min:
            excluded.add(a)
            bounds[i] = (0.0, 0.0)
            continue

        # ── Gruppen befüllen ─────────────────────────────────────────────────
        if cls == "crypto" or a.lower() == "bitcoin":
            crypto_idx.append(i)

        if cls == "stocks":
            stock_idx.append(i)

        by_sector.setdefault(sektor, []).append(i)
        by_region.setdefault(region, []).append(i)

    # ── C1: Bitcoin / Crypto gesamt ≤ crypto_max ────────────────────────────
    if crypto_idx:
        def _crypto(w, idx=crypto_idx, cap=crypto_max):
            return cap - sum(w[j] for j in idx)
        cons.append({"type": "ineq", "fun": _crypto})

    # ── C2: Einzelner Aktientitel ≤ single_stock_max  (je Asset separat) ────
    for i in stock_idx:
        def _single(w, ii=i, cap=single_stock_max):
            return cap - w[ii]
        cons.append({"type": "ineq", "fun": _single})

    # ── C3: Alle Stocks zusammen ≤ stocks_total_max ──────────────────────────
    if stock_idx:
        def _stocks_total(w, idx=stock_idx, cap=stocks_total_max):
            return cap - sum(w[j] for j in idx)
        cons.append({"type": "ineq", "fun": _stocks_total})

    # ── C4: Einzelne Branche ≤ sector_max  (je Sektor separat) ──────────────
    for sektor, idx in by_sector.items():
        def _sector(w, idx=idx, cap=sector_max):
            return cap - sum(w[j] for j in idx)
        cons.append({"type": "ineq", "fun": _sector})

    # ── C5: Einzelne Region ≤ region_max  (je Region separat) ───────────────
    for region, idx in by_region.items():
        def _region(w, idx=idx, cap=region_max):
            return cap - sum(w[j] for j in idx)
        cons.append({"type": "ineq", "fun": _region})

    return bounds, cons, excluded


# ─────────────────────────────────────────────
# OPTIMIZERS
# ─────────────────────────────────────────────

def _feasible_w0(n: int, bounds: list, cons: list) -> np.ndarray:
    """
    Berechnet einen zulässigen Startpunkt der alle Constraints erfüllt.
    Strategie: gleichgewichtet, dann bounds clippen, dann renormalisieren.
    Mehrere Versuche mit leicht gestörten Startpunkten falls nötig.
    """
    # Versuch 1: gleichgewichtet clippen + normieren
    w = np.ones(n) / n
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    w  = np.clip(w, lo, hi)
    s  = w.sum()
    if s > 1e-9:
        w = w / s
        w = np.clip(w, lo, hi)
        s = w.sum()
        if s > 1e-9:
            w = w / s

    # Versuch 2: falls immer noch infeasible, nur freie Assets (hi > 0) gleichgewichten
    free = [i for i in range(n) if hi[i] > 1e-9]
    if not free:
        return np.ones(n) / n   # Fallback – sollte nicht vorkommen
    w2 = np.zeros(n)
    w2[free] = 1.0 / len(free)
    w2 = np.clip(w2, lo, hi)
    s2 = w2.sum()
    if s2 > 1e-9:
        w2 = w2 / s2

    # Nimm den Startpunkt der näher an 1 summt
    return w if abs(w.sum() - 1) < abs(w2.sum() - 1) else w2


def _run_solver(obj_fn, bounds, cons, n_assets, n_restarts=5):
    """
    Führt SLSQP mit mehreren Startpunkten aus und gibt das beste Ergebnis zurück.
    Gibt immer ein gültiges Gewichtsvektor zurück (nie silent equal-weight Fallback).
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    best_x   = None
    best_val = np.inf

    for seed in range(n_restarts):
        if seed == 0:
            w0 = _feasible_w0(n_assets, bounds, cons)
        else:
            # Leicht gestörter Startpunkt
            rng = np.random.default_rng(seed)
            noise = rng.uniform(0, 0.02, n_assets)
            w0 = _feasible_w0(n_assets, bounds, cons) + noise
            w0 = np.clip(w0, lo, hi)
            s  = w0.sum()
            if s > 1e-9:
                w0 = w0 / s

        res = minimize(
            obj_fn, w0, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": 3000, "ftol": 1e-10},
        )

        if res.success:
            val = obj_fn(res.x)
            if val < best_val:
                best_val = val
                best_x   = res.x.copy()

    if best_x is not None:
        return best_x

    # Wenn kein Restart konvergiert: besten infeasible Punkt zurückgeben
    # (besser als silent equal-weight)
    print(f"  ⚠️  Solver nicht konvergiert nach {n_restarts} Versuchen – nehme letzten Versuch")
    return res.x


def opt_utility(X, lam, bounds, cons):
    mu = X.mean().values; cov = X.cov().values
    def obj(w):
        return -(float(w @ mu) - lam * float(w @ cov @ w))
    return _run_solver(obj, bounds, cons, X.shape[1])


def opt_sortino(X, bounds, cons):
    rf_m = (1 + RF_ANNUAL) ** (1/MONTHS_PER_YEAR) - 1
    def obj(w):
        pr = port_ret(w, X); ex = pr - rf_m; dn = ex[ex < 0]
        if dn.empty or dn.std() <= 1e-12: return 1e6
        return -ex.mean() / dn.std()
    return _run_solver(obj, bounds, cons, X.shape[1])


def opt_cvar(X, bounds, cons, alpha=ALPHA):
    def obj(w):
        l = cvar_loss(port_ret(w, X), alpha)
        return 1e6 if pd.isna(l) else l
    return _run_solver(obj, bounds, cons, X.shape[1])

def calc_frontier(X, bounds, n_pts=22):
    n = X.shape[1]; mu = X.mean().values; cov = X.cov().values
    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    vols, rets = [], []
    for t in np.linspace(mu.min(), mu.max(), n_pts):
        c = cons + [{"type": "ineq", "fun": lambda w, t=t: float(w @ mu) - t}]
        r = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP",
                     bounds=bounds, constraints=c,
                     options={"maxiter": 1000, "ftol": 1e-9})
        if not r.success: continue
        pr = port_ret(r.x, X)
        vols.append(ann_vol(pr) * 100)
        rets.append(ann_return(pr) * 100)
    return vols, rets


# ─────────────────────────────────────────────
# MAIN OPTIMIZATION (cached per parameter set)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Optimiere Portfolios …")
def run_optimization(
    assets: tuple,
    lookback_months: int,
    lambda_utility: float,
    min_w: float, max_w: float, test_pct: float,
    # ── Constraints (entsprechen den DEFAULT_* oben) ──────────────────────
    crypto_max: float,
    single_stock_max: float,
    stocks_total_max: float,
    sector_max: float,
    region_max: float,
    esg_min: int,
) -> dict:
    assets = list(assets)

    # Zeitfenster aus monatlichen Preisen ausschneiden
    prices = prices_monthly[assets].copy()
    prices = prices.iloc[-lookback_months:].copy()

    if len(prices) < 18:
        raise ValueError(
            f"Nur {len(prices)} Monate verfügbar nach Lookback-Filter. "
            "Lookback erhöhen oder weniger Assets wählen."
        )

    returns = prices.pct_change().dropna(how="any")
    if len(returns) < 15:
        raise ValueError("Zu wenige Renditebeobachtungen nach pct_change.")

    X_train, X_test = split(returns, test_pct)
    if len(X_train) < 12 or len(X_test) < 3:
        raise ValueError("Train/Test-Split zu klein – Test-Anteil oder Lookback anpassen.")

    bounds, cons, excluded = build_constraints(
        assets, min_w, max_w,
        crypto_max=crypto_max,
        single_stock_max=single_stock_max,
        stocks_total_max=stocks_total_max,
        sector_max=sector_max,
        region_max=region_max,
        esg_min=esg_min,
    )

    n  = len(assets)

    # ── Feasibility-Check: zulässigen Startpunkt prüfen ──────────────────────
    # Wenn zu viele Constraints aktiv sind, schlägt der Solver fehl.
    # Hier wird ein feasibler Startpunkt berechnet und geprüft.
    w_feas = _feasible_w0(n, bounds, cons)
    sum_check = w_feas.sum()
    if abs(sum_check - 1.0) > 0.05:
        raise ValueError(
            f"Constraints sind nicht erfüllbar: zulässiger Startpunkt summiert zu "
            f"{sum_check:.2f} ≠ 1.0. Bitte Constraints lockern "
            f"(z.B. Region/Sektor-Max erhöhen oder weniger Assets ausschließen)."
        )

    weights_map = {
        "Equal Weight": w_feas.copy(),   # Equal Weight = feasibler Startpunkt
        "Utility":      opt_utility(X_train, lambda_utility, bounds, cons),
        "Max Sortino":  opt_sortino(X_train, bounds, cons),
        "Min CVaR":     opt_cvar(X_train, bounds, cons),
    }

    portfolios = {
        label: portfolio_metrics(
            port_ret(w, X_test), dict(zip(assets, w)), label
        )
        for label, w in weights_map.items()
    }

    # Erwartete Renditen + Gewichtsübersicht
    er_annual = ((1 + X_train.mean()) ** MONTHS_PER_YEAR) - 1
    er_df = pd.DataFrame({
        "Asset":           assets,
        "Klasse":          [get_full_meta(a).get("asset_class","–") for a in assets],
        "Region":          [get_full_meta(a).get("region","–") for a in assets],
        "Sektor":          [get_full_meta(a).get("sektor","–") for a in assets],
        "ESG":             [get_full_meta(a).get("esg_score", None) for a in assets],
        "E(r) ann.":       er_annual.values,
        "w Utility":       [portfolios["Utility"]["weights"].get(a,0.) for a in assets],
        "w Sortino":       [portfolios["Max Sortino"]["weights"].get(a,0.) for a in assets],
        "w CVaR":          [portfolios["Min CVaR"]["weights"].get(a,0.) for a in assets],
        "Ausgeschlossen":  [a in excluded for a in assets],
    }).sort_values("E(r) ann.", ascending=False)

    ef_v, ef_r = calc_frontier(X_train, bounds, n_pts=22)

    return dict(
        assets=assets, excluded=excluded,
        prices_window=prices, returns_all=returns,
        X_train=X_train, X_test=X_test,
        portfolios=portfolios, er_df=er_df,
        ef_vols=ef_v, ef_rets=ef_r,
        window_start=prices.index[0], window_end=prices.index[-1],
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def _sb_label(text: str):
    st.markdown(
        f'<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        f'color:#94A3B8;margin-bottom:.3rem;">{text}</p>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## ⚡ Optimizer")
    st.markdown(
        '<p style="font-size:.75rem;color:#CBD5E1;margin-top:-.4rem;">'
        "Utility · Max Sortino · Min CVaR</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Asset-Auswahl ────────────────────────
    _sb_label("Asset-Klassen")
    sel_classes = st.multiselect(
        "k", ALL_CLASSES, default=ALL_CLASSES, label_visibility="collapsed"
    )
    filtered = [
        a for a in ALL_ASSETS
        if asset_meta_ldr.get(a, {}).get("asset_class", "Other") in sel_classes
    ]

    _sb_label("Assets")
    default_sel = filtered[:12] if len(filtered) > 12 else filtered
    sel_assets = st.multiselect(
        "a", filtered, default=default_sel, label_visibility="collapsed"
    )
    st.markdown("---")

    # ── Optimierung ──────────────────────────
    _sb_label("Optimierung")
    lookback_months = st.slider(
        "Lookback (Monate)", 24, 120, 60, 6,
        help=f"Verfügbare Monatsdaten: {len(prices_monthly)}"
    )
    lambda_utility = st.slider("λ Utility",   0.0, 20.0, 4.0, 0.5)
    test_pct       = st.slider("Test-Anteil", 0.10, 0.40, 0.30, 0.05)
    st.markdown("---")

    # ── Constraints ──────────────────────────
    _sb_label("Constraints")

    min_w            = st.slider("Min. Gewicht je Asset",       0.00, 0.10,
                                  DEFAULT_MIN_W,           0.01)
    max_w            = st.slider("Max. Gewicht je Asset",       0.05, 1.00,
                                  DEFAULT_MAX_W,           0.05)
    crypto_max       = st.slider("Crypto gesamt ≤",             0.00, 0.30,
                                  DEFAULT_CRYPTO_MAX,      0.01,
                                  help="C1 · Summe aller Crypto-Assets")
    single_stock_max = st.slider("Einzelaktie ≤",               0.05, 0.30,
                                  DEFAULT_SINGLE_STOCK_MAX, 0.01,
                                  help="C2 · Kein einzelner Aktientitel über X")
    stocks_total_max = st.slider("Aktien gesamt ≤",             0.10, 1.00,
                                  DEFAULT_STOCKS_TOTAL_MAX, 0.05,
                                  help="C3 · Summe aller Stocks")
    sector_max       = st.slider("Einzelne Branche ≤",          0.10, 1.00,
                                  DEFAULT_SECTOR_MAX,      0.05,
                                  help="C4 · Max. Gewicht je Sektor aus asset_metadata.py")
    region_max       = st.slider("Einzelne Region ≤",           0.10, 1.00,
                                  DEFAULT_REGION_MAX,      0.05,
                                  help="C5 · Max. Gewicht je Region aus asset_metadata.py")
    esg_min          = st.slider("ESG-Mindest-Score",           0, 80,
                                  DEFAULT_ESG_MIN,         5,
                                  help="Assets unter Schwelle → Gewicht = 0")

    if esg_min > 0:
        n_excl_esg = sum(
            1 for a in sel_assets
            if a in META_DF.index
            and META_DF.loc[a, "esg_score"] is not None
            and META_DF.loc[a, "esg_score"] < esg_min
        )
        if n_excl_esg:
            st.markdown(
                f'<span style="font-size:.65rem;color:#F59E0B;">'
                f"⚠ {n_excl_esg} Asset(s) unter ESG-Schwelle</span>",
                unsafe_allow_html=True,
            )
    st.markdown("---")

    run_btn = st.button("▶ Optimierung starten", type="primary")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
if len(sel_assets) < 2:
    st.warning("Bitte mindestens 2 Assets auswählen.")
    st.stop()
if min_w * len(sel_assets) > 1:
    st.error(f"Min-Gewicht ({min_w:.0%}) × {len(sel_assets)} Assets > 100%.")
    st.stop()
if max_w * len(sel_assets) < 1:
    st.error(f"Max-Gewicht ({max_w:.0%}) × {len(sel_assets)} Assets < 100%.")
    st.stop()
if lookback_months > len(prices_monthly):
    st.error(f"Lookback ({lookback_months} Monate) > verfügbare Daten ({len(prices_monthly)} Monate).")
    st.stop()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_btn or "opt_res" not in st.session_state:
    try:
        st.session_state["opt_res"] = run_optimization(
            assets=tuple(sel_assets),
            lookback_months=lookback_months,
            lambda_utility=lambda_utility,
            min_w=min_w, max_w=max_w, test_pct=test_pct,
            crypto_max=crypto_max,
            single_stock_max=single_stock_max,
            stocks_total_max=stocks_total_max,
            sector_max=sector_max,
            region_max=region_max,
            esg_min=esg_min,
        )
        st.session_state["opt_params"] = dict(
            lookback_months=lookback_months, lambda_utility=lambda_utility,
            min_w=min_w, max_w=max_w, test_pct=test_pct,
            crypto_max=crypto_max,
            single_stock_max=single_stock_max,
            stocks_total_max=stocks_total_max,
            sector_max=sector_max,
            region_max=region_max,
            esg_min=esg_min,
            assets=sel_assets,
        )
    except Exception as e:
        st.error(f"Optimierung fehlgeschlagen: {e}")
        st.stop()

res    = st.session_state["opt_res"]
params = st.session_state["opt_params"]
pf     = res["portfolios"]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
chips = []
if crypto_max < 1.0:
    chips.append(f'<span class="chip-warn">Crypto ≤ {crypto_max:.0%}</span>')
if single_stock_max < 0.30:
    chips.append(f'<span class="chip-warn">Aktie ≤ {single_stock_max:.0%}</span>')
if stocks_total_max < 1.0:
    chips.append(f'<span class="chip-warn">Stocks ≤ {stocks_total_max:.0%}</span>')
if sector_max < 1.0:
    chips.append(f'<span class="chip-warn">Branche ≤ {sector_max:.0%}</span>')
if region_max < 1.0:
    chips.append(f'<span class="chip-warn">Region ≤ {region_max:.0%}</span>')
if esg_min > 0:
    chips.append(f'<span class="chip-green">ESG ≥ {esg_min} · {esg_label(esg_min)}</span>')
if res["excluded"]:
    chips.append(
        f'<span class="chip-warn">{len(res["excluded"])} ausgeschlossen</span>'
    )

st.markdown(f"""
<div style="padding:.6rem 0 .4rem;">
    <span class="chip">New Portfolio · Optimizer</span>
    <h1 style="margin:.3rem 0 0;font-size:1.9rem;font-weight:800;color:#0F172A;">
        Portfolio Optimierung
    </h1>
    <p style="color:#94A3B8;font-size:.75rem;margin-top:.25rem;
              font-family:'JetBrains Mono',monospace;">
        {len(res["assets"])} Assets &nbsp;·&nbsp;
        {params["lookback_months"]} Monate Lookback &nbsp;·&nbsp;
        {res["window_start"].strftime("%b %Y")} – {res["window_end"].strftime("%b %Y")} &nbsp;·&nbsp;
        λ = {params["lambda_utility"]:.1f} &nbsp;·&nbsp;
        Test = {params["test_pct"]:.0%} &nbsp;·&nbsp;
        r_f = {RF_ANNUAL:.1%}
    </p>
    <div style="margin-top:.4rem;">{"".join(chips)}</div>
</div>
""", unsafe_allow_html=True)

# Ausgeschlossene Assets
if res["excluded"]:
    with st.expander(
        f"ℹ️ {len(res['excluded'])} Assets ausgeschlossen (Gewicht = 0)", expanded=False
    ):
        excl_rows = []
        for a in sorted(res["excluded"]):
            m   = get_full_meta(a)
            esg = m.get("esg_score", None)
            if esg is not None and esg_min > 0 and esg < esg_min:
                grund = f"ESG {esg} < {esg_min}"
            else:
                grund = "ESG-Filter"
            excl_rows.append({
                "Asset":  a,
                "Klasse": m.get("asset_class","–"),
                "Region": m.get("region","–"),
                "Sektor": m.get("sektor","–"),
                "ESG":    f"{esg} ({esg_label(esg)})" if esg else "–",
                "Grund":  grund,
            })
        st.dataframe(pd.DataFrame(excl_rows), hide_index=True, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Highlights</span>', unsafe_allow_html=True)

_opt = ["Utility","Max Sortino","Min CVaR"]
best_ret  = max(_opt, key=lambda k: pf[k]["ann_return"])
best_sort = max(_opt, key=lambda k: pf[k]["sortino"])
best_cvar = min(_opt, key=lambda k: pf[k]["cvar_95"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Beste Ann. Rendite", best_ret,   f'{pf[best_ret]["ann_return"]:.2%}')
c2.metric("Bester Sortino",     best_sort,  f'{pf[best_sort]["sortino"]:.3f}')
c3.metric("Bestes CVaR",        best_cvar,  f'{pf[best_cvar]["cvar_95"]:.2%}')
c4.metric("λ Utility",          f'{params["lambda_utility"]:.1f}', None)

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance", "⚖️ Gewichte",
    "🎯 Efficient Frontier", "📊 Kennzahlen", "🧩 Struktur & ESG",
])


# ══════════════════════════════════════════════
# TAB 1 – PERFORMANCE
# ══════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = make_fig("Kumulierte Renditen – Out-of-Sample", h=380)
        for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
            s = pf[n]["cumulative"]
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name=n,
                line=dict(color=PORTFOLIO_COLORS[n], width=2.4),
                hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>",
            ))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = make_fig("Drawdown (%)", h=380)
        for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
            d = dd_series(pf[n]["cumulative"])
            fig.add_trace(go.Scatter(
                x=d.index, y=d.values, name=n,
                line=dict(color=PORTFOLIO_COLORS[n], width=2.2),
                hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
            ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<span class="chip">Rolling Analyse (6 Monate)</span>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    rf_m   = (1 + RF_ANNUAL) ** (1/MONTHS_PER_YEAR) - 1

    with c3:
        fig = make_fig("Rolling Sortino (6 Monate)", h=300)
        def _rs(x):
            x  = pd.Series(x); ex = x - rf_m; dn = ex[ex < 0]
            if len(dn) < 2: return np.nan
            dd = dn.std() * np.sqrt(MONTHS_PER_YEAR)
            if dd <= 0: return np.nan
            return ((1+x).prod()**(MONTHS_PER_YEAR/len(x)) - 1 - RF_ANNUAL) / dd
        for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
            rs = pf[n]["returns"].rolling(6).apply(_rs, raw=False)
            fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name=n,
                line=dict(color=PORTFOLIO_COLORS[n], width=1.8)))
        fig.add_hline(y=0, line=dict(color="#CBD5E1", width=1, dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = make_fig("Rolling CVaR 95% (6 Monate)", h=300)
        def _rc(x):
            l = cvar_loss(pd.Series(x), ALPHA)
            return l * 100 if pd.notna(l) else np.nan
        for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
            rc = pf[n]["returns"].rolling(6).apply(_rc, raw=False)
            fig.add_trace(go.Scatter(x=rc.index, y=rc.values, name=n,
                line=dict(color=PORTFOLIO_COLORS[n], width=1.8)))
        fig.update_layout(yaxis_title="CVaR 95% (%)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 – GEWICHTE
# ══════════════════════════════════════════════
with tab2:
    sel_pf = st.selectbox("Portfolio", ["Utility","Max Sortino","Min CVaR"], index=0)
    c1, c2 = st.columns(2)

    with c1:
        nz  = {k: v for k, v in pf[sel_pf]["weights"].items() if v > 0.001}
        fig = go.Figure(go.Pie(
            labels=list(nz.keys()), values=list(nz.values()), hole=0.55,
            marker=dict(colors=[ASSET_COLORS[a] for a in nz],
                        line=dict(color="white", width=2)),
            textinfo="percent+label", textfont=dict(size=9),
        ))
        fig.update_layout(**_base_layout(f"Gewichte · {sel_pf}", h=380))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        wdf = pd.DataFrame({
            "Asset":       res["assets"],
            "Equal Weight":[pf["Equal Weight"]["weights"].get(a,0.) for a in res["assets"]],
            "Utility":     [pf["Utility"]["weights"].get(a,0.) for a in res["assets"]],
            "Max Sortino": [pf["Max Sortino"]["weights"].get(a,0.) for a in res["assets"]],
            "Min CVaR":    [pf["Min CVaR"]["weights"].get(a,0.) for a in res["assets"]],
        }).sort_values(sel_pf, ascending=True)
        fig = go.Figure()
        for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
            fig.add_trace(go.Bar(
                y=wdf["Asset"], x=wdf[n]*100, name=n, orientation="h",
                marker=dict(color=PORTFOLIO_COLORS[n]),
                hovertemplate="%{y}: %{x:.1f}%<extra>" + n + "</extra>",
            ))
        fig.update_layout(**_base_layout(
            "Gewichtsvergleich", h=max(380, len(res["assets"]) * 22),
            extra=dict(barmode="group", xaxis_title="Gewicht (%)")
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<span class="chip">Vollständige Gewichtstabelle</span>',
                unsafe_allow_html=True)
    wtbl = pd.DataFrame({
        "Asset":       res["assets"],
        "Klasse":      [get_full_meta(a).get("asset_class","–") for a in res["assets"]],
        "Region":      [get_full_meta(a).get("region","–") for a in res["assets"]],
        "Sektor":      [get_full_meta(a).get("sektor","–") for a in res["assets"]],
        "ESG":         [
            f'{get_full_meta(a).get("esg_score","–")} '
            f'({esg_label(get_full_meta(a).get("esg_score",0))})'
            for a in res["assets"]
        ],
        "✗":           ["✗" if a in res["excluded"] else "" for a in res["assets"]],
        "Equal Weight":[pf["Equal Weight"]["weights"].get(a,0.) for a in res["assets"]],
        "Utility":     [pf["Utility"]["weights"].get(a,0.) for a in res["assets"]],
        "Max Sortino": [pf["Max Sortino"]["weights"].get(a,0.) for a in res["assets"]],
        "Min CVaR":    [pf["Min CVaR"]["weights"].get(a,0.) for a in res["assets"]],
    }).sort_values("Utility", ascending=False)
    for c in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
        wtbl[c] = wtbl[c].map(lambda x: f"{x:.2%}")
    st.dataframe(wtbl, use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 3 – EFFICIENT FRONTIER
# ══════════════════════════════════════════════
with tab3:
    fig = make_fig(
        "Efficient Frontier · Monatsbasis", h=560,
        extra=dict(xaxis_title="Ann. Volatilität (%)",
                   yaxis_title="Ann. Rendite (%)"),
    )
    if res["ef_vols"]:
        fig.add_trace(go.Scatter(
            x=res["ef_vols"], y=res["ef_rets"],
            mode="lines+markers", name="Efficient Frontier",
            line=dict(color="#2563EB", width=2.4), marker=dict(size=5),
            hovertemplate="σ: %{x:.2f}%<br>μ: %{y:.2f}%<extra></extra>",
        ))
    for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]:
        p = pf[n]
        fig.add_trace(go.Scatter(
            x=[p["ann_vol"]*100], y=[p["ann_return"]*100],
            mode="markers+text", text=[n], textposition="top center", name=n,
            marker=dict(size=14, color=PORTFOLIO_COLORS[n],
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"{n}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%<extra></extra>",
        ))
    for a in res["assets"]:
        if a in res["excluded"]: continue
        r = res["X_test"][a].dropna()
        fig.add_trace(go.Scatter(
            x=[ann_vol(r)*100], y=[ann_return(r)*100],
            mode="markers", name=a,
            marker=dict(size=7, color=ASSET_COLORS[a], opacity=0.6,
                        line=dict(color="white", width=0.5)),
            hovertemplate=f"{a}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%<extra></extra>",
            showlegend=False,
        ))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 – KENNZAHLEN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<span class="chip">Portfoliovergleich (Out-of-Sample)</span>',
                unsafe_allow_html=True)

    comp = pd.DataFrame([{
        "Portfolio":    n,
        "Ann. Rendite": pf[n]["ann_return"],
        "Ann. Vola":    pf[n]["ann_vol"],
        "Sharpe":       pf[n]["sharpe"],
        "Sortino":      pf[n]["sortino"],
        "Max Drawdown": pf[n]["max_dd"],
        "CVaR 95%":     pf[n]["cvar_95"],
        "Total Return": pf[n]["total_return"],
    } for n in ["Equal Weight","Utility","Max Sortino","Min CVaR"]]).set_index("Portfolio")

    disp = comp.copy()
    for c in ["Ann. Rendite","Ann. Vola","Max Drawdown","Total Return","CVaR 95%"]:
        disp[c] = comp[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "–")
    for c in ["Sharpe","Sortino"]:
        disp[c] = comp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    st.dataframe(disp, use_container_width=True, height=220)

    st.markdown("---")
    st.markdown('<span class="chip">Erwartete Renditen + Gewichte (Training)</span>',
                unsafe_allow_html=True)

    er = res["er_df"].copy()
    er["ESG"]            = er["ESG"].apply(lambda x: f"{x} ({esg_label(x)})" if pd.notna(x) and x is not None else "–")
    er["E(r) ann."]      = er["E(r) ann."].map(lambda x: f"{x:.2%}")
    er["w Utility"]      = er["w Utility"].map(lambda x: f"{x:.2%}")
    er["w Sortino"]      = er["w Sortino"].map(lambda x: f"{x:.2%}")
    er["w CVaR"]         = er["w CVaR"].map(lambda x: f"{x:.2%}")
    er["Ausgeschlossen"] = er["Ausgeschlossen"].map(lambda x: "✗" if x else "")
    st.dataframe(er, use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 5 – STRUKTUR & ESG
# ══════════════════════════════════════════════
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        corr  = res["returns_all"].corr()
        short = [a.replace(" Equity","").replace(" Corp","")[:14]
                 for a in corr.columns]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=short, y=short,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=7),
            hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(**_base_layout("Korrelationsmatrix (Monatsrenditen)", h=480))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cls_w = {}
        for a, w in pf["Utility"]["weights"].items():
            c = get_full_meta(a).get("asset_class","Other")
            cls_w[c] = cls_w.get(c,0.) + w
        fig = go.Figure(go.Pie(
            labels=list(cls_w.keys()), values=list(cls_w.values()), hole=0.55,
            marker=dict(colors=COLORS[:len(cls_w)], line=dict(color="white", width=2)),
            textinfo="percent+label", textfont=dict(size=10),
        ))
        fig.update_layout(**_base_layout("Asset-Klassen · Utility", h=480))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<span class="chip">ESG-Profil der Auswahl</span>',
                unsafe_allow_html=True)

    esg_rows = []
    for a in res["assets"]:
        m = get_full_meta(a); s = m.get("esg_score", None)
        esg_rows.append({
            "Asset":   a,
            "Klasse":  m.get("asset_class","–"),
            "Sektor":  m.get("sektor","–"),
            "Region":  m.get("region","–"),
            "ESG":     s,
            "Label":   esg_label(s) if s is not None else "–",
            "w Utility": f'{pf["Utility"]["weights"].get(a,0.):.2%}',
            "✗":       "✗" if a in res["excluded"] else "",
        })
    esg_df = pd.DataFrame(esg_rows).sort_values("ESG", ascending=False)
    st.dataframe(esg_df, use_container_width=True, hide_index=True, height=380)

    # ESG-Score des Utility Portfolios (gewichteter Durchschnitt)
    esg_scores = [
        (pf["Utility"]["weights"].get(a, 0.), get_full_meta(a).get("esg_score", None))
        for a in res["assets"]
    ]
    valid_esg = [(w, s) for w, s in esg_scores if s is not None]
    if valid_esg:
        total_w   = sum(w for w, _ in valid_esg)
        weighted_esg = sum(w * s for w, s in valid_esg) / total_w if total_w > 0 else None
        if weighted_esg:
            st.markdown(
                f'<p style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.8rem;color:#475569;margin-top:.5rem;">'
                f'Gewichteter Portfolio-ESG (Utility): '
                f'<strong style="color:#0F172A;">'
                f'{weighted_esg:.1f} ({esg_label(int(weighted_esg))})'
                f'</strong></p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<span class="chip">Asset-Klassen Übersicht</span>',
                unsafe_allow_html=True)
    cls_tbl = pd.DataFrame([{
        "Klasse":          c,
        "Gewicht Utility": f"{w:.2%}",
        "Anzahl":          sum(1 for a in res["assets"]
                               if get_full_meta(a).get("asset_class","–") == c),
    } for c, w in sorted(cls_w.items(), key=lambda x: -x[1])])
    st.dataframe(cls_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'New Portfolio · Monatsbasis · '
    f'{params["lookback_months"]} Monate Lookback · '
    f'λ = {params["lambda_utility"]:.1f} · '
    f'test = {params["test_pct"]:.0%} · '
    f'r_f = {RF_ANNUAL:.1%} · '
    f'Crypto ≤ {params["crypto_max"]:.0%} · '
    f'Stocks ≤ {params["stocks_total_max"]:.0%} · '
    f'Aktie ≤ {params["single_stock_max"]:.0%} · '
    f'Branche ≤ {params["sector_max"]:.0%} · '
    f'Region ≤ {params["region_max"]:.0%} · '
    f'ESG ≥ {params["esg_min"]}</p>',
    unsafe_allow_html=True,
)