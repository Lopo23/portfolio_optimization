"""
optimization_dashboard.py  –  New Portfolio Optimizer
======================================================
Ablegen: dashboard/pages/optimization_dashboard.py
Starten: streamlit run dashboard/app.py

Portfolios:
    1) Max Utility  (Mean-Variance, λ wählbar)
    2) Min CVaR 95%
    + Equal Weight als Benchmark

Constraints (alle als scipy ineq, sum(w)==1 als eq):
    C1 · Crypto/Bitcoin gesamt          ≤ X %
    C2 · Einzelner Aktientitel          ≤ X %
    C3 · Alle Stocks zusammen           ≤ X %
    C4 · Einzelne Branche               ≤ X %
    C5 · Einzelne Region                ≤ X %
    C6 · Gewichteter Ø-ESG (bewertete   ≥ X  (nur Titel mit esg_score)
         Titel normiert auf ihr Teilportfolio)
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
from scipy.optimize import minimize, LinearConstraint, Bounds

from new_portfolio_loader import load_new_portfolio
from asset_metadata import get_df as get_meta_df, esg_label


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONSTRAINT DEFAULTS  –  hier anpassen                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
DEFAULT_MIN_W            = 0.00   # Mindestgewicht je Asset
DEFAULT_MAX_W            = 0.35   # Maximalgewicht je Asset
# Constraints: [MIN, MAX] als Bereich – Solver hält beide Grenzen ein
DEFAULT_CRYPTO_MIN       = 0.00   # C1 · Crypto gesamt            ≥  0 %
DEFAULT_CRYPTO_MAX       = 0.05   # C1 · Crypto gesamt            ≤  5 %
DEFAULT_SINGLE_STOCK_MAX = 0.10   # C2 · Einzelaktie (nur Stocks) ≤ 10 %
DEFAULT_STOCKS_MIN       = 0.00   # C3 · Stocks gesamt            ≥  0 %
DEFAULT_STOCKS_MAX       = 0.40   # C3 · Stocks gesamt            ≤ 40 %
DEFAULT_SECTOR_MIN       = 0.00   # C4 · Branche                  ≥  0 %
DEFAULT_SECTOR_MAX       = 0.30   # C4 · Branche                  ≤ 30 %
DEFAULT_REGION_MIN       = 0.00   # C5 · Region (exkl. Global/DM/EM) ≥  0 %
DEFAULT_REGION_MAX       = 0.40   # C5 · Region (exkl. Global/DM/EM) ≤ 40 %
DEFAULT_ESG_AVG_MIN      = 0.0    # C6 · Ø-ESG (nur bew. Titel)  ≥  0 (0 = aus)
DEFAULT_ETF_MIN          = 0.00   # C7 · ETF gesamt               ≥  0 %
DEFAULT_ETF_MAX          = 1.00   # C7 · ETF gesamt               ≤ 100 %
DEFAULT_BOND_ETF_MIN     = 0.00   # C8 · Bond ETF gesamt          ≥  0 %
DEFAULT_BOND_ETF_MAX     = 1.00   # C8 · Bond ETF gesamt          ≤ 100 %
DEFAULT_CATHOLIC_MIN     = 0.00   # C9 · Catholic Index           ≥  0 %
DEFAULT_CATHOLIC_MAX     = 0.10   # C9 · Catholic Index           ≤ 10 %
DEFAULT_LOOKBACK_MONTHS  = 60
DEFAULT_TEST_PCT         = 0.30
DEFAULT_LAMBDA           = 4.0
# Regionen die NICHT unter den Region-Constraint fallen (geografisch undefiniert)
UNCONSTRAINED_REGIONS    = {"Global", "Developed Markets", "Emerging Markets"}


# ─────────────────────────────────────────────
# PAGE CONFIG + STYLING
# ─────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Optimizer", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#F8FAFC;color:#1E293B;}
h1,h2,h3{font-family:'Syne',sans-serif;color:#0F172A;font-weight:700;}
[data-testid="stSidebar"]{background:#0F172A !important;}
[data-testid="stSidebar"] *{font-family:'Syne',sans-serif !important;color:#94A3B8 !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] .stMarkdown p{color:#CBD5E1 !important;}
[data-testid="metric-container"]{background:white;border:1px solid #E2E8F0;border-radius:10px;
    padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.06);}
[data-testid="metric-container"] label{color:#64748B !important;font-size:.65rem !important;
    letter-spacing:.1em;text-transform:uppercase;font-family:'JetBrains Mono',monospace !important;}
[data-testid="metric-container"] [data-testid="metric-value"]{
    color:#0F172A !important;font-size:1.25rem !important;font-weight:700;}
.chip{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.58rem;
      letter-spacing:.15em;text-transform:uppercase;color:#3B82F6;border:1px solid #BFDBFE;
      background:#EFF6FF;border-radius:4px;padding:2px 8px;margin-bottom:.5rem;}
.chip-warn{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.58rem;
           letter-spacing:.15em;text-transform:uppercase;color:#D97706;border:1px solid #FDE68A;
           background:#FFFBEB;border-radius:4px;padding:2px 8px;margin:0 4px 4px 0;}
.chip-green{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.58rem;
            letter-spacing:.15em;text-transform:uppercase;color:#059669;border:1px solid #A7F3D0;
            background:#ECFDF5;border-radius:4px;padding:2px 8px;margin:0 4px 4px 0;}
hr{border-color:#E2E8F0 !important;margin:1.5rem 0;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RF_ANNUAL       = 0.02017
MONTHS_PER_YEAR = 12
ALPHA           = 0.95

COLORS = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
          "#06B6D4","#F97316","#84CC16","#EC4899","#14B8A6",
          "#A78BFA","#FB923C","#34D399","#60A5FA","#FBBF24"]

PF_COLORS = {
    "Equal Weight": "#94A3B8",
    "Max Utility":  "#2563EB",
    "Min CVaR":     "#F59E0B",
}
PF_NAMES = ["Equal Weight", "Max Utility", "Min CVaR"]

def _lo(title="", h=360, extra=None):
    d = dict(paper_bgcolor="white", plot_bgcolor="#F8FAFC",
             font=dict(family="Syne", color="#64748B", size=11),
             margin=dict(l=10, r=10, t=44, b=10), height=h,
             title=dict(text=title, font=dict(family="Syne", size=13, color="#0F172A")),
             xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
             yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
             legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#E2E8F0",
                         borderwidth=1, font=dict(size=10, color="#475569")))
    if extra: d.update(extra)
    return d

def mfig(title="", h=360, extra=None):
    f = go.Figure(); f.update_layout(**_lo(title, h, extra)); return f


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Daten …")
def _load():
    return load_new_portfolio(PROJECT_ROOT / "data" / "New Portfolio")

data           = _load()
PM             = data["prices_monthly"]
asset_meta_ldr = data["asset_meta"]
asset_classes  = data["asset_classes"]
META_DF        = get_meta_df()

if PM.empty:
    st.error("Keine Preisdaten."); st.stop()

ALL_ASSETS  = sorted(PM.columns.tolist())
ALL_CLASSES = sorted(asset_classes.keys())
A_COLORS    = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ALL_ASSETS)}

def ameta(a: str) -> dict:
    """Kombiniert Loader-Meta + asset_metadata.py.
    Unterstützt beide Feldnamen-Konventionen:
      - Deutsch (alt):  klasse / sektor
      - Englisch (neu): asset_class / sector
    """
    m = dict(asset_meta_ldr.get(a, {}))
    if a in META_DF.index:
        row = META_DF.loc[a]
        # Klasse: "klasse" (DE) oder "asset_class" (EN) → intern als "klasse"
        m["klasse"]    = (row.get("klasse") or row.get("asset_class")
                          or m.get("asset_class", "–"))
        # Sektor: "sektor" (DE) oder "sector" (EN) → intern als "sektor"
        m["sektor"]    = row.get("sektor") or row.get("sector") or "–"
        m["region"]    = row.get("region",     "–")
        m["esg_score"] = row.get("esg_score",  None)
        m["name"]      = row.get("name",       a)
        m["isin"]      = row.get("isin",       None)
        # Auch asset_class befüllen (für Loader-Kompatibilität)
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


# ─────────────────────────────────────────────
# MATH
# ─────────────────────────────────────────────
def ann_ret(r): r=r.dropna(); return np.nan if r.empty else (1+r).prod()**(MONTHS_PER_YEAR/len(r))-1
def ann_vol(r): r=r.dropna(); return np.nan if r.empty else r.std()*np.sqrt(MONTHS_PER_YEAR)

def cvar_loss(r, alpha=ALPHA):
    r = r.dropna()
    if r.empty: return np.nan
    tail = r[r <= r.quantile(1-alpha)]
    return np.nan if tail.empty else -tail.mean()

def port_rets(w, X): return pd.Series(X.values @ w, index=X.index)

def pf_metrics(r, weights, label):
    r = r.dropna()
    if r.empty:
        return dict(label=label, ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
                    max_dd=np.nan, cvar_95=np.nan, total_return=np.nan,
                    weights=weights, returns=r, cum=pd.Series(dtype=float))
    cum  = (1+r).cumprod(); ar=ann_ret(r); av=ann_vol(r); roll=cum.cummax()
    return dict(label=label, ann_return=ar, ann_vol=av,
                sharpe=(ar-RF_ANNUAL)/av if av and av>0 else np.nan,
                max_dd=((cum-roll)/roll).min(), cvar_95=-cvar_loss(r),
                total_return=cum.iloc[-1]-1,
                weights=weights, returns=r, cum=cum)

def split_tt(X, pct):
    idx = max(int(len(X)*(1-pct)), 12); idx = min(idx, len(X)-3)
    return X.iloc[:idx].copy(), X.iloc[idx:].copy()

def dd_s(cum): return (cum-cum.cummax())/cum.cummax()*100


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONSTRAINT BUILDER                                                         ║
# ║                                                                             ║
# ║  Rückgabe: bounds_arr (n×2), cons_list, eq_w0 (feasibler Startpunkt)      ║
# ║                                                                             ║
# ║  Strategie:                                                                 ║
# ║  • bounds_arr: lo/hi je Asset (0–max_w, oder 0–0 für harte Ausschlüsse)   ║
# ║  • cons_list:  [sum==1] + C1–C6 als ineq (cap - sum >= 0)                 ║
# ║  • eq_w0: gleichgewichteter Startpunkt der ALLE ineq-Constraints erfüllt   ║
# ║    → berechnet via LP-ähnlichem Iteriationsverfahren                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def build_constraints(assets, min_w, max_w,
                      crypto_min, crypto_max,
                      single_stock_max,
                      stocks_min, stocks_max,
                      sector_min, sector_max,
                      region_min, region_max,
                      etf_min, etf_max,
                      bond_etf_min, bond_etf_max,
                      catholic_min, catholic_max,
                      esg_avg_min):
    n   = len(assets)
    lo  = np.full(n, float(min_w))
    hi  = np.full(n, float(max_w))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Bounds als explizite ineq-Constraints  (doppelte Sicherheit –
    # SLSQP respektiert bounds bei sehr vielen Constraints manchmal nicht)
    for i in range(n):
        _hi = float(hi[i])
        cons.append({"type":"ineq","fun": lambda w, ii=i, c=_hi: c - w[ii]})
        if float(lo[i]) > 1e-9:
            _lo = float(lo[i])
            cons.append({"type":"ineq","fun": lambda w, ii=i, c=_lo: w[ii] - c})

    crypto_idx = []; stock_idx = []
    by_sector  = {}; by_region = {}
    esg_pairs  = []   # (index, esg_score) für Titel MIT Score

    for i, a in enumerate(assets):
        m   = ameta(a)
        cls = m.get("asset_class", "Other").lower()
        reg = m.get("region", "–")
        sec = m.get("sektor", "–")
        esg = m.get("esg_score", None)

        if cls == "crypto" or a.lower() == "bitcoin":
            crypto_idx.append(i)
        if cls == "stocks":
            stock_idx.append(i)
        by_sector.setdefault(sec, []).append(i)
        by_region.setdefault(reg, []).append(i)
        if esg is not None:
            esg_pairs.append((i, float(esg)))

    # C1 · Crypto: crypto_min ≤ sum ≤ crypto_max
    if crypto_idx:
        _idx = crypto_idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(crypto_max): c - sum(w[j] for j in idx)})
        if float(crypto_min) > 0:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(crypto_min): sum(w[j] for j in idx) - c})

    # C2 · Einzelaktie ≤ single_stock_max  (nur echte Stocks, keine ETFs)
    for i in stock_idx:
        cons.append({"type":"ineq",
                     "fun": lambda w, ii=i, c=float(single_stock_max): c - w[ii]})

    # C3 · Stocks gesamt: stocks_min ≤ sum ≤ stocks_max
    if stock_idx:
        _idx = stock_idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(stocks_max): c - sum(w[j] for j in idx)})
        if float(stocks_min) > 0:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(stocks_min): sum(w[j] for j in idx) - c})

    # C4 · Branche: sector_min ≤ sum ≤ sector_max  (je Sektor separat)
    for sec_name, idx in by_sector.items():
        _idx = idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(sector_max): c - sum(w[j] for j in idx)})
        if float(sector_min) > 0:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(sector_min): sum(w[j] for j in idx) - c})

    # C5 · Region: region_min ≤ sum ≤ region_max
    # EXKLUDIERT: Global, Developed Markets, Emerging Markets
    # → diese sind geografisch undefiniert und unterliegen keinem Region-Cap
    for reg_name, idx in by_region.items():
        if reg_name in UNCONSTRAINED_REGIONS:
            continue   # Global/DM/EM: kein Cap
        _idx = idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(region_max): c - sum(w[j] for j in idx)})
        if float(region_min) > 0:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(region_min): sum(w[j] for j in idx) - c})

    # C6 · Gewichteter Ø-ESG (nur Titel mit Score) ≥ esg_avg_min
    # Formel: Σ(w_i·esg_i) / Σ(w_i)  ≥  esg_avg_min
    # Umgeformt: Σ(w_i·esg_i) - esg_avg_min·Σ(w_i)  ≥  0
    if esg_avg_min > 0 and len(esg_pairs) >= 2:
        _ep = esg_pairs[:]
        _mn = float(esg_avg_min)
        cons.append({"type":"ineq",
                     "fun": lambda w, ep=_ep, mn=_mn:
                         sum(w[j]*s for j,s in ep) - mn * sum(w[j] for j,_ in ep)})

    # C7 · ETF gesamt: etf_min ≤ Σ w_i ≤ etf_max
    etf_idx = [i for i, a in enumerate(assets)
                if ameta(a).get("asset_class","").lower() == "etf"]
    if etf_idx:
        _idx = etf_idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(etf_max): c - sum(w[j] for j in idx)})
        if float(etf_min) > 1e-9:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(etf_min): sum(w[j] for j in idx) - c})

    # C8 · Bond ETF gesamt: bond_etf_min ≤ Σ w_i ≤ bond_etf_max
    bond_etf_idx = [i for i, a in enumerate(assets)
                    if ameta(a).get("asset_class","").lower() == "bond etf"]
    if bond_etf_idx:
        _idx = bond_etf_idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(bond_etf_max): c - sum(w[j] for j in idx)})
        if float(bond_etf_min) > 1e-9:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(bond_etf_min): sum(w[j] for j in idx) - c})

    # C9 · Catholic Index: catholic_min ≤ w ≤ catholic_max (einzelnes Asset)
    catholic_idx = [i for i, a in enumerate(assets) if a.lower() == "catholic index"]
    if catholic_idx:
        _idx = catholic_idx[:]
        cons.append({"type":"ineq",
                     "fun": lambda w, idx=_idx, c=float(catholic_max): c - sum(w[j] for j in idx)})
        if float(catholic_min) > 1e-9:
            cons.append({"type":"ineq",
                         "fun": lambda w, idx=_idx, c=float(catholic_min): sum(w[j] for j in idx) - c})

    # bounds_list: bleibt als Sicherheitsnetz, wird aber auch als ineq enforzt
    bounds_list = [(float(lo[i]), float(hi[i])) for i in range(n)]

    # ── Feasibler Startpunkt ─────────────────────────────────────────────────
    # Strategie: gleichgewichtet → clip auf bounds → normieren → prüfen ob ineq ok
    # Falls nicht: iterativ Weights von verletzenden Gruppen reduzieren
    w0 = np.clip(np.ones(n)/n, lo, hi)
    s  = w0.sum()
    if s > 1e-9: w0 = np.clip(w0/s, lo, hi)

    # Prüfe ob sum(w0) ≈ 1 erreichbar (= mind. 1 freies Asset)
    free = [i for i in range(n) if hi[i] > 1e-9]
    if not free:
        # Alle Assets ausgeschlossen – kann nicht passieren, aber Fallback
        w0 = np.zeros(n); w0[0] = 1.0
    else:
        w0 = np.zeros(n)
        w0[free] = 1.0 / len(free)
        w0 = np.clip(w0, lo, hi)
        s  = w0.sum()
        if s > 1e-9: w0 = w0/s

    return bounds_list, cons, w0, esg_pairs


# ─────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────
def _solve(obj_fn, bounds_list, cons, w0, n_restarts=6, seed=0):
    """
    SLSQP mit mehreren Starts. Gibt bestes konvergiertes Ergebnis zurück.
    Entscheidend: wir starten immer von einem feasiblen Punkt und
    prüfen ob das Ergebnis tatsächlich die Constraints erfüllt.
    """
    lo  = np.array([b[0] for b in bounds_list])
    hi  = np.array([b[1] for b in bounds_list])
    rng = np.random.default_rng(seed)

    best_x, best_f = None, np.inf
    n = len(w0)

    for k in range(n_restarts):
        # Startpunkt: feasibler Punkt + kleine Störung
        if k == 0:
            start = w0.copy()
        else:
            noise  = rng.uniform(-0.02, 0.02, n)
            start  = np.clip(w0 + noise, lo, hi)
            s      = start.sum()
            start  = start / s if s > 1e-9 else w0.copy()

        res = minimize(obj_fn, start, method="SLSQP",
                       bounds=bounds_list, constraints=cons,
                       options={"maxiter": 5000, "ftol": 1e-12, "eps": 1e-8})

        if not res.success:
            continue

        f = obj_fn(res.x)
        # Constraints nachprüfen
        constraint_ok = all(
            c["fun"](res.x) >= -1e-4   # Toleranz
            for c in cons
            if c["type"] == "ineq"
        )
        sum_ok = abs(res.x.sum() - 1.0) < 1e-4

        if constraint_ok and sum_ok and f < best_f:
            best_f = f
            best_x = res.x.copy()

    if best_x is not None:
        return best_x, True

    # Kein konvergiertes Ergebnis → feasiblen Startpunkt zurückgeben
    return w0.copy(), False


def opt_utility(X, lam, bounds_list, cons, w0):
    mu  = X.mean().values
    cov = X.cov().values
    def obj(w): return -(float(w @ mu) - lam * float(w @ cov @ w))
    x, ok = _solve(obj, bounds_list, cons, w0)
    return x, ok


def opt_cvar(X, bounds_list, cons, w0, alpha=ALPHA):
    def obj(w):
        l = cvar_loss(port_rets(w, X), alpha)
        return 1e6 if pd.isna(l) else l
    x, ok = _solve(obj, bounds_list, cons, w0)
    return x, ok


def eff_frontier(X, bounds_list, cons, w0, n_pts=22):
    """
    Efficient Frontier UNTER DEN GLEICHEN CONSTRAINTS wie die Optimierung.
    Nur so liegt das Utility-Portfolio auf der Frontier.
    Strategie: Min-Varianz für jedes Return-Ziel auf dem constrained Set.
    """
    n   = X.shape[1]; mu = X.mean().values; cov = X.cov().values
    # cons enthält bereits sum==1 + alle C1-C6
    # Wir fügen für jeden Punkt eine zusätzliche Return-Mindestbedingung hinzu
    vols, rets = [], []
    # Erreichbares Return-Spektrum: teste mit w0
    r_lo = float(w0 @ mu) * 0.5
    r_hi = float(mu.max()) * 0.95
    for t in np.linspace(r_lo, r_hi, n_pts):
        c = cons + [{"type":"ineq","fun": lambda w, t=t: float(w@mu)-t}]
        r = minimize(lambda w: float(w@cov@w), w0, method="SLSQP",
                     bounds=bounds_list, constraints=c,
                     options={"maxiter": 2000, "ftol": 1e-10})
        if not r.success: continue
        # Constraint-Check
        if any(con["fun"](r.x) < -1e-3 for con in cons if con["type"]=="ineq"):
            continue
        pr = port_rets(r.x, X)
        vols.append(ann_vol(pr)*100); rets.append(ann_ret(pr)*100)
    return vols, rets


# ─────────────────────────────────────────────
# MAIN OPTIMIZATION  (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Optimiere …")
def run_opt(assets, lookback_months, lam, min_w, max_w, test_pct,
            crypto_min, crypto_max,
            single_stock_max,
            stocks_min, stocks_max,
            sector_min, sector_max,
            region_min, region_max,
            etf_min, etf_max,
            bond_etf_min, bond_etf_max,
            catholic_min, catholic_max,
            esg_avg_min):

    assets  = list(assets)
    prices  = PM[assets].iloc[-lookback_months:].copy()
    returns = prices.pct_change().dropna(how="any")

    if len(returns) < 15:
        raise ValueError(f"Nur {len(returns)} Monate – Lookback erhöhen.")

    X_train, X_test = split_tt(returns, test_pct)
    if len(X_train) < 12 or len(X_test) < 3:
        raise ValueError("Train/Test zu klein.")

    bounds_list, cons, w0_feas, esg_pairs = build_constraints(
        assets, min_w, max_w,
        crypto_min, crypto_max,
        single_stock_max,
        stocks_min, stocks_max,
        sector_min, sector_max,
        region_min, region_max,
        etf_min, etf_max,
        bond_etf_min, bond_etf_max,
        catholic_min, catholic_max,
        esg_avg_min,
    )

    n = len(assets)

    # ── Feasibility-Check ────────────────────────────────────────────────────
    # Prüfe ob der feasible Startpunkt alle ineq-Constraints erfüllt
    violations = []
    for c in cons:
        if c["type"] == "ineq":
            val = c["fun"](w0_feas)
            if val < -1e-3:
                violations.append(val)
    if violations:
        raise ValueError(
            f"Constraints nicht erfüllbar mit aktueller Asset-Auswahl "
            f"({len(violations)} Verletzungen). "
            "Constraints lockern oder andere Assets wählen."
        )

    # ── Equal Weight: bounds-konform (kein echter 1/n wenn max_w < 1/n) ────────
    # z.B. bei 2 Assets und max_w=0.35 → 1/2=50% > 35% → clip auf 35%, renorm
    _lo_ew   = np.full(n, float(min_w))
    _hi_ew   = np.full(n, float(max_w))
    w_ew_raw = np.ones(n) / n
    w_ew     = np.clip(w_ew_raw, _lo_ew, _hi_ew)
    s_ew     = w_ew.sum()
    w_ew     = w_ew / s_ew if s_ew > 1e-9 else w_ew

    # ── Optimierte Portfolios ────────────────────────────────────────────────
    w_util, util_ok  = opt_utility(X_train, lam, bounds_list, cons, w0_feas)
    w_cvar, cvar_ok  = opt_cvar(X_train, bounds_list, cons, w0_feas)

    solver_warnings = []
    if not util_ok: solver_warnings.append("Utility: Solver nicht konvergiert (feasibler Startpunkt verwendet)")
    if not cvar_ok: solver_warnings.append("CVaR: Solver nicht konvergiert (feasibler Startpunkt verwendet)")

    weights_map = {
        "Equal Weight": w_ew,
        "Max Utility":  w_util,
        "Min CVaR":     w_cvar,
    }

    portfolios = {
        label: pf_metrics(port_rets(w, X_test), dict(zip(assets, w)), label)
        for label, w in weights_map.items()
    }

    # Constraint-Verletzungen im Ergebnis prüfen
    def check_violations(w, label):
        viols = []
        for c in cons:
            if c["type"] == "ineq":
                val = c["fun"](w)
                if val < -1e-3:
                    viols.append(f"{label}: constraint {val:.4f}")
        return viols

    all_viols = check_violations(w_util, "Utility") + check_violations(w_cvar, "CVaR")

    er = ((1 + X_train.mean()) ** MONTHS_PER_YEAR) - 1
    er_df = pd.DataFrame({
        "Asset":    assets,
        "Klasse":   [ameta(a).get("klasse","–")     for a in assets],
        "Region":   [ameta(a).get("region","–")      for a in assets],
        "Sektor":   [ameta(a).get("sektor","–")      for a in assets],
        "ESG":      [ameta(a).get("esg_score", None) for a in assets],
        "E(r)":     er.values,
        "w Utility":[portfolios["Max Utility"]["weights"].get(a,0.) for a in assets],
        "w CVaR":   [portfolios["Min CVaR"]["weights"].get(a,0.)    for a in assets],
    }).sort_values("E(r)", ascending=False)

    ef_v, ef_r = eff_frontier(X_train, bounds_list, cons, w0_feas)

    return dict(
        assets=assets, esg_pairs=esg_pairs,
        returns_all=returns, X_train=X_train, X_test=X_test,
        portfolios=portfolios, er_df=er_df,
        ef_vols=ef_v, ef_rets=ef_r,
        window_start=prices.index[0], window_end=prices.index[-1],
        solver_warnings=solver_warnings,
        constraint_violations=all_viols,
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def _lbl(t):
    st.markdown(f'<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
                f'color:#94A3B8;margin-bottom:.3rem;">{t}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚡ Optimizer")
    st.markdown('<p style="font-size:.75rem;color:#CBD5E1;margin-top:-.4rem;">'
                "Max Utility · Min CVaR</p>", unsafe_allow_html=True)
    st.markdown("---")

    _lbl("Asset-Klassen")
    sel_classes = st.multiselect("kl", ALL_CLASSES, default=ALL_CLASSES,
                                  label_visibility="collapsed")
    filtered = [a for a in ALL_ASSETS
                if asset_meta_ldr.get(a,{}).get("asset_class","Other") in sel_classes]

    _lbl("Assets")
    # Default: ALLE verfügbaren Assets (nicht nur [:12])
    # → Solver hat genug Spielraum, Constraints feasibel zu erfüllen
    sel_assets = st.multiselect("as", filtered, default=filtered,
                                 label_visibility="collapsed")
    st.markdown("---")

    _lbl("Optimierung")
    lookback_months = st.slider("Lookback (Monate)", 24, len(PM),
                                 DEFAULT_LOOKBACK_MONTHS, 6,
                                 help=f"Verfügbar: {len(PM)} Monate")
    lam      = st.slider("λ Utility",   0.0, 20.0, DEFAULT_LAMBDA,   0.5)
    test_pct = st.slider("Test-Anteil", 0.10, 0.40, DEFAULT_TEST_PCT, 0.05)
    st.markdown("---")

    _lbl("Constraints")
    min_w, max_w = st.slider(
        "Gewicht je Asset [min, max]", 0.00, 1.00,
        (DEFAULT_MIN_W, DEFAULT_MAX_W), 0.01)

    crypto_min, crypto_max = st.slider(
        "C1 · Crypto [min, max]", 0.00, 0.30,
        (DEFAULT_CRYPTO_MIN, DEFAULT_CRYPTO_MAX), 0.01,
        help="Crypto/Bitcoin gesamt – Bereich [min, max]")

    single_stock_max = st.slider(
        "C2 · Einzelaktie ≤  (nur Stocks, keine ETFs)", 0.05, 1.00,
        DEFAULT_SINGLE_STOCK_MAX, 0.01)

    stocks_min, stocks_max = st.slider(
        "C3 · Stocks gesamt [min, max]", 0.00, 1.00,
        (DEFAULT_STOCKS_MIN, DEFAULT_STOCKS_MAX), 0.05,
        help="Alle Einzelaktien (Stocks) zusammen")

    sector_min, sector_max = st.slider(
        "C4 · Branche [min, max]", 0.00, 1.00,
        (DEFAULT_SECTOR_MIN, DEFAULT_SECTOR_MAX), 0.05,
        help="Gilt je Sektor separat")

    region_min, region_max = st.slider(
        "C5 · Region [min, max]", 0.00, 1.00,
        (DEFAULT_REGION_MIN, DEFAULT_REGION_MAX), 0.05,
        help="Gilt je Region separat. Global/DM/EM sind exkludiert.")

    esg_avg_min = st.slider("C6 · Ø-ESG (bew. Titel) ≥", 0.0, 8.0,
                             DEFAULT_ESG_AVG_MIN, 0.1,
                             help="Nur Titel mit ESG-Score. Bitcoin/Commodities zählen nicht.")

    etf_min, etf_max = st.slider(
        "C7 · ETF gesamt [min, max]", 0.00, 1.00,
        (DEFAULT_ETF_MIN, DEFAULT_ETF_MAX), 0.05,
        help="Alle ETFs (Klasse=ETF) zusammen")

    bond_etf_min, bond_etf_max = st.slider(
        "C8 · Bond ETF [min, max]", 0.00, 1.00,
        (DEFAULT_BOND_ETF_MIN, DEFAULT_BOND_ETF_MAX), 0.05,
        help="Alle Bond ETFs zusammen")

    catholic_min, catholic_max = st.slider(
        "C9 · Catholic Index [min, max]", 0.00, 0.30,
        (DEFAULT_CATHOLIC_MIN, DEFAULT_CATHOLIC_MAX), 0.01,
        help="Catholic Values Index (einzelnes Asset)")

    if esg_avg_min > 0:
        esg_vals = [META_DF.loc[a,"esg_score"]
                    for a in sel_assets
                    if a in META_DF.index and META_DF.loc[a,"esg_score"] is not None]
        if esg_vals:
            ew_avg = sum(esg_vals) / len(esg_vals)
            col = "#10B981" if ew_avg >= esg_avg_min else "#F59E0B"
            st.markdown(f'<span style="font-size:.65rem;color:{col};">'
                        f'EW Ø-ESG: {ew_avg:.2f} (Ziel ≥ {esg_avg_min:.1f})</span>',
                        unsafe_allow_html=True)
    st.markdown("---")
    run_btn = st.button("▶ Optimierung starten", type="primary")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
if len(sel_assets) < 2:
    st.warning("Bitte mindestens 2 Assets auswählen."); st.stop()
if min_w * len(sel_assets) > 1.0:
    st.error(f"Min-Gewicht ({min_w:.0%}) × {len(sel_assets)} Assets > 100%."); st.stop()
if max_w * len(sel_assets) < 1.0:
    st.error(f"Max-Gewicht ({max_w:.0%}) × {len(sel_assets)} Assets < 100%."); st.stop()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_btn or "opt_res" not in st.session_state:
    try:
        st.session_state["opt_res"] = run_opt(
            tuple(sel_assets), lookback_months, lam,
            min_w, max_w, test_pct,
            crypto_min, crypto_max,
            single_stock_max,
            stocks_min, stocks_max,
            sector_min, sector_max,
            region_min, region_max,
            etf_min, etf_max,
            bond_etf_min, bond_etf_max,
            catholic_min, catholic_max,
            esg_avg_min,
        )
        st.session_state["opt_params"] = dict(
            lookback_months=lookback_months, lam=lam, test_pct=test_pct,
            min_w=min_w, max_w=max_w,
            crypto_min=crypto_min, crypto_max=crypto_max,
            single_stock_max=single_stock_max,
            stocks_min=stocks_min, stocks_max=stocks_max,
            sector_min=sector_min, sector_max=sector_max,
            region_min=region_min, region_max=region_max,
            etf_min=etf_min, etf_max=etf_max,
            bond_etf_min=bond_etf_min, bond_etf_max=bond_etf_max,
            catholic_min=catholic_min, catholic_max=catholic_max,
            esg_avg_min=esg_avg_min,
        )
    except Exception as e:
        st.error(f"Optimierung fehlgeschlagen: {e}"); st.stop()

res    = st.session_state["opt_res"]
params = st.session_state["opt_params"]
pf     = res["portfolios"]


# ─────────────────────────────────────────────
# SOLVER-WARNUNGEN
# ─────────────────────────────────────────────
for w in res.get("solver_warnings", []):
    st.warning(f"⚠️ {w}")

# Constraint-Verletzungen anzeigen (sollte leer sein)
viols = res.get("constraint_violations", [])
if viols:
    with st.expander(f"⚠️ {len(viols)} Constraint-Verletzung(en) im Ergebnis", expanded=True):
        for v in viols:
            st.caption(v)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
chips = []
if crypto_max        < 1.0: chips.append(f'<span class="chip-warn">Crypto {crypto_min:.0%}–{crypto_max:.0%}</span>')
if single_stock_max  < 1.0: chips.append(f'<span class="chip-warn">Aktie ≤ {single_stock_max:.0%}</span>')
if stocks_max        < 1.0: chips.append(f'<span class="chip-warn">Stocks {stocks_min:.0%}–{stocks_max:.0%}</span>')
if sector_max        < 1.0: chips.append(f'<span class="chip-warn">Sektor {sector_min:.0%}–{sector_max:.0%}</span>')
if region_max        < 1.0: chips.append(f'<span class="chip-warn">Region {region_min:.0%}–{region_max:.0%}</span>')
if esg_avg_min       > 0:   chips.append(f'<span class="chip-green">Ø-ESG ≥ {esg_avg_min:.1f}</span>')
if etf_max           < 1.0: chips.append(f'<span class="chip-warn">ETF {etf_min:.0%}–{etf_max:.0%}</span>')
if bond_etf_max      < 1.0: chips.append(f'<span class="chip-warn">BondETF {bond_etf_min:.0%}–{bond_etf_max:.0%}</span>')
if catholic_max      < 0.30:chips.append(f'<span class="chip-warn">Catholic {catholic_min:.0%}–{catholic_max:.0%}</span>')

st.markdown(f"""
<div style="padding:.6rem 0 .4rem;">
  <span class="chip">New Portfolio · Optimizer</span>
  <h1 style="margin:.3rem 0 0;font-size:1.9rem;font-weight:800;color:#0F172A;">Portfolio Optimierung</h1>
  <p style="color:#94A3B8;font-size:.75rem;margin-top:.25rem;font-family:'JetBrains Mono',monospace;">
    {len(res["assets"])} Assets &nbsp;·&nbsp;
    {params["lookback_months"]} Monate &nbsp;·&nbsp;
    {res["window_start"].strftime("%b %Y")} – {res["window_end"].strftime("%b %Y")} &nbsp;·&nbsp;
    λ = {params["lam"]:.1f} &nbsp;·&nbsp; Test = {params["test_pct"]:.0%}
  </p>
  <div style="margin-top:.4rem;">{"".join(chips)}</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Highlights</span>', unsafe_allow_html=True)
best_ret  = max(["Max Utility","Min CVaR"], key=lambda k: pf[k]["ann_return"] or -np.inf)
best_cvar = min(["Max Utility","Min CVaR"], key=lambda k: pf[k]["cvar_95"]    or  np.inf)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Beste Ann. Rendite", best_ret,  f'{pf[best_ret]["ann_return"]:.2%}')
c2.metric("Bestes CVaR",        best_cvar, f'{pf[best_cvar]["cvar_95"]:.2%}')
c3.metric("λ Utility",          f'{params["lam"]:.1f}')
c4.metric("Test-Periode",
          f'{res["X_test"].index[0].strftime("%b %Y")} – {res["X_test"].index[-1].strftime("%b %Y")}')
st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈 Performance","⚖️ Gewichte","🎯 Efficient Frontier","📊 Kennzahlen","🧩 Struktur & ESG"
])


# ══════════════════════════════════════════════
# TAB 1 · PERFORMANCE
# ══════════════════════════════════════════════
with tab1:
    c1,c2 = st.columns(2)
    with c1:
        f = mfig("Kumulierte Renditen – Out-of-Sample", h=380)
        for n in PF_NAMES:
            s = pf[n]["cum"]
            f.add_trace(go.Scatter(x=s.index, y=s.values, name=n,
                line=dict(color=PF_COLORS[n], width=2.4),
                hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>"))
        st.plotly_chart(f, use_container_width=True)
    with c2:
        f = mfig("Drawdown (%)", h=380)
        for n in PF_NAMES:
            d = dd_s(pf[n]["cum"])
            f.add_trace(go.Scatter(x=d.index, y=d.values, name=n,
                line=dict(color=PF_COLORS[n], width=2.2),
                hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>"))
        st.plotly_chart(f, use_container_width=True)

    st.markdown('<span class="chip">Rolling CVaR 95% (6 Monate)</span>', unsafe_allow_html=True)
    f = mfig(h=300)
    def _rc(x):
        l = cvar_loss(pd.Series(x), ALPHA); return l*100 if pd.notna(l) else np.nan
    for n in PF_NAMES:
        rc = pf[n]["returns"].rolling(6).apply(_rc, raw=False)
        f.add_trace(go.Scatter(x=rc.index, y=rc.values, name=n,
            line=dict(color=PF_COLORS[n], width=1.8)))
    f.update_layout(yaxis_title="CVaR 95% (%)")
    st.plotly_chart(f, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 · GEWICHTE
# ══════════════════════════════════════════════
with tab2:
    sel_pf = st.selectbox("Portfolio", ["Max Utility","Min CVaR"], index=0)
    c1,c2  = st.columns(2)

    with c1:
        nz = {k:v for k,v in pf[sel_pf]["weights"].items() if v > 0.001}
        f  = go.Figure(go.Pie(labels=list(nz.keys()), values=list(nz.values()), hole=0.55,
            marker=dict(colors=[A_COLORS[a] for a in nz], line=dict(color="white",width=2)),
            textinfo="percent+label", textfont=dict(size=9)))
        f.update_layout(**_lo(f"Gewichte · {sel_pf}", h=380))
        st.plotly_chart(f, use_container_width=True)

    with c2:
        wdf = pd.DataFrame({"Asset": res["assets"],
            **{n: [pf[n]["weights"].get(a,0.) for a in res["assets"]] for n in PF_NAMES}
        }).sort_values(sel_pf, ascending=True)
        f = go.Figure()
        for n in PF_NAMES:
            f.add_trace(go.Bar(y=wdf["Asset"], x=wdf[n]*100, name=n, orientation="h",
                marker=dict(color=PF_COLORS[n]),
                hovertemplate="%{y}: %{x:.1f}%<extra>"+n+"</extra>"))
        f.update_layout(**_lo("Gewichtsvergleich", h=max(380, len(res["assets"])*22),
                               extra=dict(barmode="group", xaxis_title="Gewicht (%)")))
        st.plotly_chart(f, use_container_width=True)

    # Constraint-Check Tabelle
    st.markdown('<span class="chip">Constraint-Check</span>', unsafe_allow_html=True)

    def check_row(label, w_dict):
        w = np.array([w_dict.get(a, 0.) for a in res["assets"]])
        # Crypto
        ci = [i for i,a in enumerate(res["assets"])
              if ameta(a).get("asset_class","").lower()=="crypto" or a.lower()=="bitcoin"]
        # Stocks
        si = [i for i,a in enumerate(res["assets"])
              if ameta(a).get("asset_class","").lower()=="stocks"]
        by_sec = {}; by_reg = {}
        for i,a in enumerate(res["assets"]):
            by_sec.setdefault(ameta(a).get("sektor","–"),[]).append(i)
            by_reg.setdefault(ameta(a).get("region","–"),[]).append(i)
        rows = []
        if ci:
            rows.append({"Constraint":"C1 Crypto", "Wert": f"{sum(w[j] for j in ci):.2%}",
                         "Limit": f"≤ {crypto_max:.0%}",
                         "OK": "✅" if sum(w[j] for j in ci) <= crypto_max+1e-4 else "❌"})
        if si:
            rows.append({"Constraint":"C2 Max Aktie", "Wert": f"{max(w[j] for j in si):.2%}",
                         "Limit": f"≤ {single_stock_max:.0%}",
                         "OK": "✅" if max(w[j] for j in si) <= single_stock_max+1e-4 else "❌"})
            sv = sum(w[j] for j in si)
            rows.append({"Constraint":"C3 Stocks Σ", "Wert": f"{sv:.2%}",
                         "Limit": f"{stocks_min:.0%}–{stocks_max:.0%}",
                         "OK": "✅" if stocks_min-1e-4 <= sv <= stocks_max+1e-4 else "❌"})
        max_sec = max(by_sec.items(), key=lambda x: sum(w[j] for j in x[1]))
        sv = sum(w[j] for j in max_sec[1])
        rows.append({"Constraint":f"C4 Branche ({max_sec[0][:15]})",
                     "Wert": f"{sv:.2%}",
                     "Limit": f"{sector_min:.0%}–{sector_max:.0%}",
                     "OK": "✅" if sector_min-1e-4 <= sv <= sector_max+1e-4 else "❌"})
        # C5: nur nicht-globale Regionen prüfen
        reg_check = {r: idx for r, idx in by_reg.items() if r not in UNCONSTRAINED_REGIONS}
        if reg_check:
            max_reg = max(reg_check.items(), key=lambda x: sum(w[j] for j in x[1]))
            rv = sum(w[j] for j in max_reg[1])
            rows.append({"Constraint":f"C5 Region ({max_reg[0][:15]})",
                         "Wert": f"{rv:.2%}",
                         "Limit": f"{region_min:.0%}–{region_max:.0%}",
                         "OK": "✅" if region_min-1e-4 <= rv <= region_max+1e-4 else "❌"})
        if esg_avg_min > 0 and res["esg_pairs"]:
            ep = res["esg_pairs"]
            tw = sum(w[j] for j,_ in ep)
            wl = sum(w[j]*s for j,s in ep)/tw if tw>0 else 0
            rows.append({"Constraint":"C6 Ø-ESG", "Wert": f"{wl:.2f}",
                         "Limit": f"≥ {esg_avg_min:.1f}",
                         "OK": "✅" if wl >= esg_avg_min-1e-3 else "❌"})
        return pd.DataFrame(rows)

    ct1,ct2 = st.columns(2)
    with ct1:
        st.markdown("**Max Utility**")
        st.dataframe(check_row("Max Utility", pf["Max Utility"]["weights"]),
                     hide_index=True, use_container_width=True)
    with ct2:
        st.markdown("**Min CVaR**")
        st.dataframe(check_row("Min CVaR", pf["Min CVaR"]["weights"]),
                     hide_index=True, use_container_width=True)

    st.markdown('<span class="chip">Vollständige Gewichtstabelle</span>', unsafe_allow_html=True)
    wtbl = pd.DataFrame({
        "Asset":   res["assets"],
        "Klasse":  [ameta(a).get("klasse","–")    for a in res["assets"]],
        "Region":  [ameta(a).get("region","–")     for a in res["assets"]],
        "Sektor":  [ameta(a).get("sektor","–")     for a in res["assets"]],
        "ESG":     [f'{ameta(a).get("esg_score","–")} ({esg_label(ameta(a).get("esg_score",0))})'
                    for a in res["assets"]],
        **{n: [f'{pf[n]["weights"].get(a,0.):.2%}' for a in res["assets"]] for n in PF_NAMES}
    }).sort_values("Max Utility", ascending=False)
    st.dataframe(wtbl, use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 3 · EFFICIENT FRONTIER
# ══════════════════════════════════════════════
with tab3:
    f = mfig("Efficient Frontier · Monatsbasis", h=560,
             extra=dict(xaxis_title="Ann. Vola (%)", yaxis_title="Ann. Rendite (%)"))
    if res["ef_vols"]:
        f.add_trace(go.Scatter(x=res["ef_vols"], y=res["ef_rets"],
            mode="lines+markers", name="Efficient Frontier",
            line=dict(color="#2563EB", width=2.4), marker=dict(size=5),
            hovertemplate="σ: %{x:.2f}%<br>μ: %{y:.2f}%<extra></extra>"))
    for n in PF_NAMES:
        p = pf[n]
        f.add_trace(go.Scatter(x=[p["ann_vol"]*100], y=[p["ann_return"]*100],
            mode="markers+text", text=[n], textposition="top center", name=n,
            marker=dict(size=14, color=PF_COLORS[n], line=dict(color="white",width=1.5)),
            hovertemplate=f"{n}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%<extra></extra>"))
    for a in res["assets"]:
        r = res["X_test"][a].dropna()
        f.add_trace(go.Scatter(x=[ann_vol(r)*100], y=[ann_ret(r)*100],
            mode="markers", name=a, marker=dict(size=7, color=A_COLORS[a], opacity=0.6,
            line=dict(color="white",width=0.5)),
            hovertemplate=f"{a}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%<extra></extra>",
            showlegend=False))
    st.plotly_chart(f, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 · KENNZAHLEN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<span class="chip">Portfoliovergleich (Out-of-Sample)</span>',
                unsafe_allow_html=True)
    comp = pd.DataFrame([{"Portfolio": n,
        "Ann. Rendite": pf[n]["ann_return"], "Ann. Vola": pf[n]["ann_vol"],
        "Sharpe": pf[n]["sharpe"], "Max DD": pf[n]["max_dd"],
        "CVaR 95%": pf[n]["cvar_95"], "Total Return": pf[n]["total_return"],
    } for n in PF_NAMES]).set_index("Portfolio")
    disp = comp.copy()
    for c in ["Ann. Rendite","Ann. Vola","Max DD","Total Return","CVaR 95%"]:
        disp[c] = comp[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "–")
    disp["Sharpe"] = comp["Sharpe"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    st.dataframe(disp, use_container_width=True, height=200)

    st.markdown("---")
    st.markdown('<span class="chip">Erwartete Renditen + Gewichte (Training)</span>',
                unsafe_allow_html=True)
    er = res["er_df"].copy()
    er["ESG"]      = er["ESG"].apply(lambda x: f"{x:.2f} ({esg_label(x)})"
                                     if pd.notna(x) and x is not None else "–")
    er["E(r)"]     = er["E(r)"].map(lambda x: f"{x:.2%}")
    er["w Utility"]= er["w Utility"].map(lambda x: f"{x:.2%}")
    er["w CVaR"]   = er["w CVaR"].map(lambda x: f"{x:.2%}")
    st.dataframe(er, use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 5 · STRUKTUR & ESG
# ══════════════════════════════════════════════
with tab5:
    c1,c2 = st.columns(2)
    with c1:
        corr  = res["returns_all"].corr()
        short = [a.replace(" Equity","").replace(" Corp","")[:14] for a in corr.columns]
        f = go.Figure(go.Heatmap(z=corr.values, x=short, y=short,
            colorscale="RdYlGn", zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=7),
            hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>"))
        f.update_layout(**_lo("Korrelationsmatrix", h=480))
        st.plotly_chart(f, use_container_width=True)
    with c2:
        cls_w = {}
        for a, w in pf["Max Utility"]["weights"].items():
            c = ameta(a).get("klasse","Other"); cls_w[c] = cls_w.get(c,0.)+w
        f = go.Figure(go.Pie(labels=list(cls_w.keys()), values=list(cls_w.values()),
            hole=0.55, marker=dict(colors=COLORS[:len(cls_w)],
            line=dict(color="white",width=2)),
            textinfo="percent+label", textfont=dict(size=10)))
        f.update_layout(**_lo("Asset-Klassen · Utility", h=480))
        st.plotly_chart(f, use_container_width=True)

    # ESG-Profil
    st.markdown('<span class="chip">ESG der bewerteten Titel</span>', unsafe_allow_html=True)

    def w_esg(w_dict):
        ep = res["esg_pairs"]
        if not ep: return None, 0.
        tw = sum(w_dict.get(res["assets"][j], 0.) for j,_ in ep)
        av = sum(w_dict.get(res["assets"][j], 0.)*s for j,s in ep) / tw if tw>0 else None
        return av, tw

    kc = st.columns(3)
    for col_ui, pf_n in zip(kc, PF_NAMES):
        av, tw = w_esg(pf[pf_n]["weights"])
        col_ui.metric(f"Ø-ESG · {pf_n}",
                      f"{av:.2f} ({esg_label(av)})" if av else "–",
                      f"{tw*100:.0f}% bewertet")

    st.markdown("---")
    st.markdown('<span class="chip">ESG-Scores je Titel</span>', unsafe_allow_html=True)

    esg_list = [(res["assets"][j], s, pf["Max Utility"]["weights"].get(res["assets"][j],0.))
                for j,s in res["esg_pairs"]]
    esg_list.sort(key=lambda x: x[1])

    if esg_list:
        tgt = params.get("esg_avg_min", 0.)
        av_u, _ = w_esg(pf["Max Utility"]["weights"])
        colors  = ["#10B981" if (tgt==0 or s<=tgt) else "#F59E0B" for _,s,_ in esg_list]
        f = mfig("ESG-Score · Utility-Gewicht als Annotation (niedriger = besser)",
                 h=max(320, len(esg_list)*28),
                 extra=dict(xaxis_title="ESG-Score", yaxis_title=""))
        f.add_trace(go.Bar(y=[a for a,_,_ in esg_list], x=[s for _,s,_ in esg_list],
            orientation="h", marker=dict(color=colors, line=dict(color="white",width=0.5)),
            text=[f"{s:.2f}  w={w*100:.1f}%" for _,s,w in esg_list],
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{y}<br>ESG: %{x:.2f}<extra></extra>"))
        if tgt > 0:
            f.add_vline(x=tgt, line=dict(color="#EF4444",width=1.5,dash="dash"),
                        annotation_text=f"Ziel ≥ {tgt:.1f}",
                        annotation_position="top right",
                        annotation_font=dict(color="#EF4444",size=10))
        if av_u:
            f.add_vline(x=av_u, line=dict(color="#2563EB",width=1.5,dash="dot"),
                        annotation_text=f"Ø Utility: {av_u:.2f}",
                        annotation_position="bottom right",
                        annotation_font=dict(color="#2563EB",size=10))
        st.plotly_chart(f, use_container_width=True)

    st.markdown('<span class="chip">ESG-Tabelle</span>', unsafe_allow_html=True)
    esg_rows = []
    for a in res["assets"]:
        m = ameta(a); s = m.get("esg_score", None)
        esg_rows.append({"Asset": a, "Klasse": m.get("klasse","–"),
                          "Sektor": m.get("sektor","–"), "Region": m.get("region","–"),
                          "ESG": f"{s:.2f}" if s else "–",
                          "Label": esg_label(s) if s else "kein Score",
                          "w Utility": f'{pf["Max Utility"]["weights"].get(a,0.):.2%}'})
    st.dataframe(pd.DataFrame(esg_rows).sort_values("ESG"),
                 use_container_width=True, hide_index=True, height=360)

    st.markdown("---")
    st.markdown('<span class="chip">Asset-Klassen</span>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Klasse": c, "Gewicht Utility": f"{w:.2%}",
        "Anzahl": sum(1 for a in res["assets"] if ameta(a).get("klasse","–")==c)}
        for c,w in sorted(cls_w.items(), key=lambda x:-x[1])]),
        use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'New Portfolio · {params["lookback_months"]} M · λ={params["lam"]:.1f} · '
    f'test={params["test_pct"]:.0%} · r_f={RF_ANNUAL:.1%} · '
    f'Crypto {params["crypto_min"]:.0%}–{params["crypto_max"]:.0%} · '
    f'Aktie≤{params["single_stock_max"]:.0%} · '
    f'Stocks {params["stocks_min"]:.0%}–{params["stocks_max"]:.0%} · '
    f'Sektor {params["sector_min"]:.0%}–{params["sector_max"]:.0%} · '
    f'Region {params["region_min"]:.0%}–{params["region_max"]:.0%} · '
    f'ETF {params["etf_min"]:.0%}–{params["etf_max"]:.0%} · '
    f'BondETF {params["bond_etf_min"]:.0%}–{params["bond_etf_max"]:.0%} · '
    f'Catholic {params["catholic_min"]:.0%}–{params["catholic_max"]:.0%} · '
    f'Ø-ESG≥{params["esg_avg_min"]:.1f}</p>',
    unsafe_allow_html=True,
)
