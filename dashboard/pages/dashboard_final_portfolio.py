"""
strategic_portfolio.py  –  Strategic Portfolio Dashboard
=========================================================
Ablegen: dashboard/pages/strategic_portfolio.py

Feste Ziel-Allokation:
  3 %  Cash / Money Market  (synthetisch @ 3.5 % p.a.)
  5 %  Bitcoin
  5 %  Catholic Values Index
 12 %  Gold  (PHAU LN Equity)
 40 %  Fixed Income & Fund
        └ 82.4 % → VNGA60 IM Equity
        └ 17.6 % → CSBGE3 IM Equity
 35 %  Equity (sub-weights normalised to 100 %)
        └ 35.0 % → CSPX LN Equity
        └ 16.8 % → IWVL LN Equity
        └ 10.0 % → 7203 JT Equity
        └ 10.0 % → EBS AV Equity
        └ 10.0 % → MSFT US Equity
        └ 10.0 % → NOVN SW Equity
        └ 10.0 % → AAPL US Equity
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

from new_portfolio_loader import load_new_portfolio
from asset_metadata import get_df as get_meta_df, esg_label


# ─────────────────────────────────────────────
# STRATEGIC WEIGHTS  (sum = 1.0)
# ─────────────────────────────────────────────
_EQ_RAW = {
    "CSPX LN Equity": .350, "IWVL LN Equity": .168,
    "7203 JT Equity":  .100, "EBS AV Equity":  .100,
    "MSFT US Equity":  .100, "NOVN SW Equity": .100,
    "AAPL US Equity":  .100,
}
_EQ_SUM = sum(_EQ_RAW.values())   # 1.018 → normalise

WEIGHTS: dict[str, float] = {
    "VNGA60 IM Equity":  0.40 * 0.824,                        # 32.96 %
    "CSBGE3 IM Equity":  0.40 * 0.176,                        #  7.04 %
    **{k: 0.35 * (v / _EQ_SUM) for k, v in _EQ_RAW.items()},
    "PHAU LN Equity":    0.12,                                 # 12.00 %
    "bitcoin":           0.05,                                 #  5.00 %
    "catholic index":    0.05,                                 #  5.00 %
}
CASH_WEIGHT   = 0.03
CASH_YIELD_PA = 0.035

# Assets without meaningful ESG scores (excluded from weighted avg ESG)
NO_ESG_ASSETS = {"_CASH", "bitcoin", "PHAU LN Equity",
                 "PHAG IM Equity", "PHPT LN Equity", "IGLN LN Equity"}

BUCKET_MAP = {
    "VNGA60 IM Equity": "Fixed Income & Fund",
    "CSBGE3 IM Equity": "Fixed Income & Fund",
    "CSPX LN Equity":   "Equity",
    "IWVL LN Equity":   "Equity",
    "7203 JT Equity":   "Equity",
    "EBS AV Equity":    "Equity",
    "MSFT US Equity":   "Equity",
    "NOVN SW Equity":   "Equity",
    "AAPL US Equity":   "Equity",
    "PHAU LN Equity":   "Gold",
    "bitcoin":          "Crypto",
    "catholic index":   "Values",
    "_CASH":            "Cash",
}
BUCKET_COLORS = {
    "Fixed Income & Fund": "#2563EB",
    "Equity":              "#10B981",
    "Gold":                "#F59E0B",
    "Crypto":              "#EF4444",
    "Values":              "#8B5CF6",
    "Cash":                "#94A3B8",
}

# PRICE_TICKERS: all assets that have actual price data in the loader.
# bitcoin and catholic index are included if present in PM_ALL,
# otherwise they fall back to zero-return proxies in build_returns().
PRICE_TICKERS = list(WEIGHTS.keys())  # resolved after PM_ALL is loaded

RF_ANNUAL       = 0.02017
MONTHS_PER_YEAR = 12
ALPHA           = 0.95

COLORS = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
          "#06B6D4","#F97316","#84CC16","#EC4899","#14B8A6",
          "#A78BFA","#FB923C","#34D399","#60A5FA","#FBBF24"]

ESG_BAND_COLORS = {
    "Severe": "#EF4444", "High": "#F59E0B",
    "Medium": "#FBBF24", "Low":  "#84CC16", "Negligible": "#10B981",
}


# ─────────────────────────────────────────────
# PAGE CONFIG + STYLING
# ─────────────────────────────────────────────
st.set_page_config(page_title="Strategic Portfolio", page_icon="📋",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#F8FAFC;color:#1E293B;}
h1,h2,h3{font-family:'Syne',sans-serif;color:#0F172A;font-weight:700;}
[data-testid="metric-container"]{
    background:white;border:1px solid #E2E8F0;border-radius:12px;
    padding:1.1rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,.07);}
[data-testid="metric-container"] label{
    color:#64748B !important;font-size:.65rem !important;
    letter-spacing:.1em;text-transform:uppercase;
    font-family:'JetBrains Mono',monospace !important;}
[data-testid="metric-container"] [data-testid="metric-value"]{
    color:#0F172A !important;font-size:1.35rem !important;font-weight:700;}
.chip{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.58rem;
      letter-spacing:.15em;text-transform:uppercase;color:#3B82F6;
      border:1px solid #BFDBFE;background:#EFF6FF;border-radius:4px;
      padding:2px 8px;margin:.2rem .3rem .2rem 0;}
hr{border-color:#E2E8F0 !important;margin:2rem 0;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def _lo(title="", h=420, extra=None):
    d = dict(
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font=dict(family="Syne", color="#64748B", size=12),
        margin=dict(l=20, r=20, t=52, b=20), height=h,
        title=dict(text=title, font=dict(family="Syne", size=14, color="#0F172A")),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,.95)", bordercolor="#E2E8F0",
                    borderwidth=1, font=dict(size=11, color="#475569")),
    )
    if extra:
        d.update(extra)
    return d



# ── German Bund YTM override ────────────────────────────────────────────
# Applied after every pct_change() to replace mark-to-market returns
# with the constant YTM-based monthly return (3.074 % p.a.)
_BUND_TICKER = "BO221256 Corp"
_BUND_YTM_PA = 0.03074

def _override_bund(ret):
    """Shift Bund return series so that the GEOMETRIC annualised return
    equals the YTM (3.074 % p.a.) while preserving historical volatility.

    Uses scipy.optimize.brentq to find the exact additive shift s such that:
        geo_annualised(r_hist + s) = 3.074 %
    Volatility, correlations and distribution shape are fully preserved.
    """
    if _BUND_TICKER not in ret.columns:
        return ret
    from scipy.optimize import brentq
    ret  = ret.copy()
    r    = ret[_BUND_TICKER].dropna()
    n    = len(r)
    if n < 2:
        return ret

    def _geo_ann(series):
        total = (1 + series).prod() - 1
        return (1 + total) ** (12 / n) - 1

    def _objective(s):
        return _geo_ann(r + s) - _BUND_YTM_PA

    try:
        s_opt = brentq(_objective, -0.20, 0.20, xtol=1e-10)
    except ValueError:
        # fallback: arithmetic mean-shift if brentq bracket fails
        s_opt = _BUND_YTM_PA / 12 - r.mean()

    ret[_BUND_TICKER] = ret[_BUND_TICKER] + s_opt
    return ret

def mfig(title="", h=420, extra=None):
    f = go.Figure()
    f.update_layout(**_lo(title, h, extra))
    return f


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def _load():
    return load_new_portfolio(PROJECT_ROOT / "data" / "New Portfolio")

data           = _load()
PM_ALL         = data["prices_monthly"]
asset_meta_ldr = data["asset_meta"]
META_DF        = get_meta_df()

def ameta(a: str) -> dict:
    m = dict(asset_meta_ldr.get(a, {}))
    if a in META_DF.index:
        row = META_DF.loc[a]
        m["klasse"]      = row.get("klasse") or row.get("asset_class") or m.get("asset_class","–")
        m["sektor"]      = row.get("sektor") or row.get("sector") or "–"
        m["region"]      = row.get("region","–")
        m["esg_score"]   = row.get("esg_score", None)
        m["name"]        = row.get("name", a)
        m["isin"]        = row.get("isin", None)
        m["asset_class"] = m["klasse"]
    else:
        m.setdefault("klasse",    m.get("asset_class","–"))
        m.setdefault("sektor",    m.get("sector","–"))
        m.setdefault("region",    "–")
        m.setdefault("esg_score", None)
        m.setdefault("name",      a)
        m.setdefault("isin",      None)
        m.setdefault("asset_class", m.get("klasse","–"))
    return m

# Resolve which tickers have actual price data
PRICE_TICKERS = [k for k in WEIGHTS if k in PM_ALL.columns]
PROXY_TICKERS = [k for k in WEIGHTS if k not in PM_ALL.columns]
if PROXY_TICKERS:
    st.info(f"No price data for {PROXY_TICKERS} → zero-return proxy used.")

# Investable tickers that also have price data (for frontier / metrics)
INVESTABLE_TICKERS = [t for t in PRICE_TICKERS
                      if t not in ("bitcoin", "catholic index")]

PM = PM_ALL[PRICE_TICKERS].copy()


# ─────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Computing returns …")
def build_returns():
    ret = _override_bund(PM.pct_change().dropna(how="all"))
    # Synthetic cash: constant monthly return
    ret["_CASH"] = (1 + CASH_YIELD_PA) ** (1/MONTHS_PER_YEAR) - 1
    # Zero-return proxy only for tickers with no price data
    for t in PROXY_TICKERS:
        ret[t] = 0.0
    return ret

RETURNS = build_returns()


# ─────────────────────────────────────────────
# PORTFOLIO RETURNS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Computing portfolio …")
def build_portfolio():
    all_a = list(WEIGHTS.keys()) + ["_CASH"]
    all_w = list(WEIGHTS.values()) + [CASH_WEIGHT]
    w_s   = pd.Series(dict(zip(all_a, all_w)))
    cols  = [c for c in RETURNS.columns if c in w_s.index]
    w_al  = w_s[cols]; w_al = w_al / w_al.sum()
    pr    = RETURNS[cols].mul(w_al, axis=1).sum(axis=1); pr.name = "Portfolio"
    cum   = (1+pr).cumprod(); roll = cum.cummax()
    return dict(returns=pr, cum=cum, drawdown=(cum-roll)/roll, w_aligned=w_al, cols=cols)

PORT = build_portfolio()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def metrics(r: pd.Series) -> dict:
    r = r.dropna(); n = len(r)
    if n < 3:
        return dict(ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
                    sortino=np.nan, max_dd=np.nan, cvar_95=np.nan,
                    total_return=np.nan, n_months=n)
    cum = (1+r).cumprod(); tot = cum.iloc[-1]-1
    ar  = (1+tot)**(MONTHS_PER_YEAR/n)-1
    av  = r.std()*np.sqrt(MONTHS_PER_YEAR)
    neg = r[r<0]; ds = neg.std()*np.sqrt(MONTHS_PER_YEAR) if len(neg)>0 else np.nan
    roll= cum.cummax(); tail = r[r<=r.quantile(1-ALPHA)]
    return dict(ann_return=ar, ann_vol=av,
                sharpe=(ar-RF_ANNUAL)/av if av>0 else np.nan,
                sortino=(ar-RF_ANNUAL)/ds if ds and ds>0 else np.nan,
                max_dd=((cum-roll)/roll).min(),
                cvar_95=-tail.mean() if not tail.empty else np.nan,
                total_return=tot, n_months=n)

PORT_M  = metrics(PORT["returns"])
ASSET_M = {a: metrics(RETURNS[a].dropna()) for a in PORT["cols"]}


# ─────────────────────────────────────────────
# ESG  –  weighted avg over SCORED assets only
# Normalised so scored sub-portfolio = 100 %
# Excludes: Gold, Bitcoin, Cash (no issuer-level score)
# ─────────────────────────────────────────────
def build_esg_contrib():
    contrib = []
    for a, w in list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)]:
        if a in NO_ESG_ASSETS:
            continue
        s = ameta(a).get("esg_score", None)
        if s is not None:
            contrib.append((a, w, float(s)))
    return contrib

ESG_CONTRIB = build_esg_contrib()
_ESG_TW     = sum(w for _, w, _ in ESG_CONTRIB)   # weight sum of scored sub-portfolio
ESG_AVG     = (sum(w*s for _, w, s in ESG_CONTRIB) / _ESG_TW
               if _ESG_TW > 0 else None)


# ─────────────────────────────────────────────
# EFFICIENT FRONTIER  (investable assets only)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Computing efficient frontier …")
def build_frontier(n_pts: int = 40):
    """
    Unconstrained mean-variance frontier over PRICE_TICKERS.
    Uses training set = full available history (no train/test split for display).
    """
    X   = RETURNS[INVESTABLE_TICKERS].dropna(how="any")
    mu  = X.mean().values
    cov = X.cov().values
    n   = len(mu)
    bnd = [(0.0, 1.0)] * n
    eq  = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    vols, rets = [], []
    for t in np.linspace(mu.min(), mu.max() * 0.98, n_pts):
        c = eq + [{"type": "ineq", "fun": lambda w, t=t: float(w @ mu) - t}]
        r = minimize(lambda w: float(w @ cov @ w),
                     np.ones(n) / n, method="SLSQP",
                     bounds=bnd, constraints=c,
                     options={"maxiter": 2000, "ftol": 1e-10})
        if not r.success:
            continue
        pr = pd.Series(X.values @ r.x)
        vols.append(pr.std() * np.sqrt(MONTHS_PER_YEAR) * 100)
        rets.append(((1 + pr).prod() ** (MONTHS_PER_YEAR / len(pr)) - 1) * 100)

    return vols, rets

EF_VOLS, EF_RETS = build_frontier()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding:.8rem 0 .4rem;">
  <span class="chip">Strategic Allocation</span>
  <span class="chip">Fixed Weights</span>
  <span class="chip">Monthly Returns</span>
  <h1 style="margin:.5rem 0 0;font-size:2.2rem;font-weight:800;color:#0F172A;">
    Strategic Portfolio
  </h1>
  <p style="color:#94A3B8;font-size:.8rem;margin-top:.4rem;
            font-family:'JetBrains Mono',monospace;line-height:1.6;">
    {len(WEIGHTS)+1} instruments &nbsp;·&nbsp;
    {RETURNS.index[0].strftime("%b %Y")} – {RETURNS.index[-1].strftime("%b %Y")} &nbsp;·&nbsp;
    {PORT_M["n_months"]} months &nbsp;·&nbsp; r_f = {RF_ANNUAL:.1%} &nbsp;·&nbsp;
    Cash = {CASH_YIELD_PA:.1%} p.a.
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# BUCKET CARDS
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Allocation by Bucket</span>', unsafe_allow_html=True)

bucket_totals: dict[str, float] = {}
for a, w in list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)]:
    b = BUCKET_MAP[a]
    bucket_totals[b] = bucket_totals.get(b, 0.) + w

cols = st.columns(len(bucket_totals))
for col, (bucket, total) in zip(cols, bucket_totals.items()):
    clr = BUCKET_COLORS[bucket]
    col.markdown(f"""
    <div style="background:white;border:1px solid #E2E8F0;border-left:4px solid {clr};
         border-radius:8px;padding:1rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,.05);">
      <div style="font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;
                  color:#64748B;font-family:'JetBrains Mono',monospace;">{bucket}</div>
      <div style="font-size:1.8rem;font-weight:700;color:{clr};line-height:1.15;
                  margin-top:.2rem;">{total:.0%}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI  –  2 rows × 4 cols
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Portfolio KPIs (full history)</span>',
            unsafe_allow_html=True)

r1, r2 = st.columns(4), st.columns(4)
r1[0].metric("Ann. Return (%)",        f"{PORT_M['ann_return']*100:.2f}")
r1[1].metric("Ann. Volatility (%)",    f"{PORT_M['ann_vol']*100:.2f}")
r1[2].metric("Sharpe Ratio",           f"{PORT_M['sharpe']:.3f}" if pd.notna(PORT_M['sharpe']) else "–")
r1[3].metric("Sortino Ratio",          f"{PORT_M['sortino']:.3f}" if pd.notna(PORT_M['sortino']) else "–")

st.markdown("<div style='margin-top:.6rem;'></div>", unsafe_allow_html=True)
r2b = st.columns(4)
r2b[0].metric("Max Drawdown (%)",      f"{PORT_M['max_dd']*100:.2f}")
r2b[1].metric("CVaR 95% (% / month)",  f"{PORT_M['cvar_95']*100:.2f}",
              help="Expected loss in the worst 5% of months (% per period).")
r2b[2].metric("Total Return (%)",      f"{PORT_M['total_return']*100:.1f}")
r2b[3].metric("Avg ESG Score",
              f"{ESG_AVG:.2f} ({esg_label(ESG_AVG)})" if ESG_AVG else "–",
              help=f"Weighted avg over scored sub-portfolio ({_ESG_TW*100:.0f}% of total).\n"
                   "Gold, Bitcoin and Cash are excluded (no issuer-level score).")

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "⚖️ Allocation",
    "📊 Asset Metrics",
    "🔗 Correlations",
    "🌿 ESG",
])


# ══════════════════════════════════════════════
# TAB 1 · PERFORMANCE
# ══════════════════════════════════════════════
with tab1:

    # ── 1a: Cumulative return ──────────────────
    st.markdown('<span class="chip">Cumulative Portfolio Return</span>',
                unsafe_allow_html=True)
    f = mfig("Cumulative Return (base = 1)", h=460)
    cum = PORT["cum"]
    f.add_trace(go.Scatter(
        x=cum.index, y=cum.values, name="Portfolio",
        line=dict(color="#2563EB", width=3),
        hovertemplate="%{x|%b %Y}<br>%{y:.4f}<extra>Portfolio</extra>",
    ))
    f.add_hline(y=1, line=dict(color="#CBD5E1", width=1, dash="dash"))
    f.update_layout(yaxis_title="Cumulative Return", showlegend=False)
    st.plotly_chart(f, use_container_width=True)

    # ── 1b: Drawdown ──────────────────────────
    st.markdown('<span class="chip">Drawdown</span>', unsafe_allow_html=True)
    f = mfig("Portfolio Drawdown (%)", h=360)
    dd = PORT["drawdown"] * 100
    f.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown",
        line=dict(color="#EF4444", width=1.8),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    f.update_layout(yaxis_title="Drawdown (%)", showlegend=False)
    st.plotly_chart(f, use_container_width=True)

    # ── 1c: Bucket contribution ───────────────
    st.markdown('<span class="chip">Return Contribution by Bucket</span>',
                unsafe_allow_html=True)
    f = mfig("Cumulative Return per Bucket (weighted)", h=440)
    bucket_rets: dict[str, pd.Series] = {}
    for a, w in list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)]:
        if a not in RETURNS.columns:
            continue
        b  = BUCKET_MAP[a]
        r  = RETURNS[a] * w
        bucket_rets[b] = bucket_rets.get(b, pd.Series(0., index=RETURNS.index)) + r
    for b, br in bucket_rets.items():
        cum_b = (1 + br.dropna()).cumprod()
        f.add_trace(go.Scatter(
            x=cum_b.index, y=cum_b.values, name=b,
            line=dict(color=BUCKET_COLORS[b], width=2.2),
            hovertemplate=f"{b}<br>%{{y:.4f}}<extra></extra>",
        ))
    st.plotly_chart(f, use_container_width=True)

    # ── 1d: Rolling 12m return ────────────────
    st.markdown('<span class="chip">Rolling 12-Month Return</span>',
                unsafe_allow_html=True)
    f = mfig("Rolling 12-Month Portfolio Return (%)", h=380)
    roll12 = PORT["returns"].rolling(12).apply(lambda x: (1+x).prod()-1, raw=False) * 100
    clrs   = ["#10B981" if v >= 0 else "#EF4444" for v in roll12.dropna()]
    f.add_trace(go.Bar(
        x=roll12.dropna().index, y=roll12.dropna().values,
        marker_color=clrs,
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    f.add_hline(y=0, line=dict(color="#CBD5E1", width=1))
    f.update_layout(yaxis_title="Return (%)", showlegend=False)
    st.plotly_chart(f, use_container_width=True)

    # ── 1e: Monthly heatmap ───────────────────
    st.markdown('<span class="chip">Monthly Return Heatmap</span>',
                unsafe_allow_html=True)
    port_r = PORT["returns"]
    pivot  = pd.DataFrame({
        "Year":  port_r.index.year,
        "Month": port_r.index.month,
        "Ret":   port_r.values * 100,
    }).pivot(index="Year", columns="Month", values="Ret")
    months     = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels = [months[c-1] for c in pivot.columns]
    z          = pivot.values.astype(float)
    txt        = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
    f = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=txt, texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="Year: %{y}<br>%{x}<br>%{z:.2f}%<extra></extra>",
    ))
    f.update_layout(**_lo("Portfolio Monthly Returns (%)",
                          h=max(280, len(pivot) * 30),
                          extra=dict(yaxis=dict(autorange="reversed"))))
    st.plotly_chart(f, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 · ALLOCATION
# ══════════════════════════════════════════════
with tab2:

    # ── 2a: Three donuts ──────────────────────
    st.markdown('<span class="chip">Portfolio Composition</span>',
                unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    all_labels  = [ameta(a).get("name", a)[:30] if a != "_CASH" else "Cash / Money Market"
                   for a in list(WEIGHTS.keys()) + ["_CASH"]]
    all_weights = list(WEIGHTS.values()) + [CASH_WEIGHT]
    all_colors  = [BUCKET_COLORS[BUCKET_MAP.get(a,"Cash")]
                   for a in list(WEIGHTS.keys()) + ["_CASH"]]

    with d1:
        f = go.Figure(go.Pie(
            labels=all_labels, values=all_weights, hole=0.55,
            marker=dict(colors=all_colors, line=dict(color="white", width=2)),
            textinfo="percent", textfont=dict(size=9),
            hovertemplate="%{label}<br>%{percent:.1%}<extra></extra>",
        ))
        f.update_layout(**_lo("All Positions", h=460))
        st.plotly_chart(f, use_container_width=True)

    with d2:
        bl = list(bucket_totals.keys()); bv = list(bucket_totals.values())
        bc = [BUCKET_COLORS[b] for b in bl]
        f = go.Figure(go.Pie(
            labels=bl, values=bv, hole=0.55,
            marker=dict(colors=bc, line=dict(color="white", width=2)),
            textinfo="percent+label", textfont=dict(size=11),
            hovertemplate="%{label}<br>%{percent:.1%}<extra></extra>",
        ))
        f.update_layout(**_lo("By Bucket", h=460))
        st.plotly_chart(f, use_container_width=True)

    with d3:
        reg_w: dict[str, float] = {}
        for a, w in list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)]:
            r = "Cash" if a == "_CASH" else ameta(a).get("region","–")
            reg_w[r] = reg_w.get(r, 0.) + w
        rl = list(reg_w.keys()); rv = list(reg_w.values())
        f = go.Figure(go.Pie(
            labels=rl, values=rv, hole=0.55,
            marker=dict(colors=COLORS[:len(rl)], line=dict(color="white", width=2)),
            textinfo="percent+label", textfont=dict(size=10),
            hovertemplate="%{label}<br>%{percent:.1%}<extra></extra>",
        ))
        f.update_layout(**_lo("By Region", h=460))
        st.plotly_chart(f, use_container_width=True)

    # ── 2b: Sector bar ────────────────────────
    st.markdown('<span class="chip">Sector Breakdown</span>', unsafe_allow_html=True)
    sec_w: dict[str, float] = {}
    for a, w in list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)]:
        s = "Cash" if a == "_CASH" else ameta(a).get("sektor","–")
        sec_w[s] = sec_w.get(s, 0.) + w
    sec_df = pd.DataFrame({"Sector": list(sec_w), "Weight": list(sec_w.values())})
    sec_df = sec_df.sort_values("Weight", ascending=True)
    f = go.Figure(go.Bar(
        y=sec_df["Sector"], x=sec_df["Weight"]*100, orientation="h",
        marker=dict(
            color=sec_df["Weight"],
            colorscale=[[0,"#EFF6FF"],[1,"#2563EB"]],
            line=dict(color="white", width=0.5),
        ),
        text=[f"{v:.1f}%" for v in sec_df["Weight"]*100],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ))
    f.update_layout(**_lo("Sector Allocation (%)",
                          h=max(340, len(sec_df) * 36),
                          extra=dict(xaxis_title="Weight (%)", yaxis_title="")))
    st.plotly_chart(f, use_container_width=True)

    # ── 2c: Detail table ──────────────────────
    st.markdown('<span class="chip">Position Detail</span>', unsafe_allow_html=True)
    rows = []
    for a, w in sorted(list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)],
                       key=lambda x: -x[1]):
        m = ameta(a) if a != "_CASH" else {}
        rows.append({
            "Ticker":  a if a != "_CASH" else "CASH",
            "Name":    m.get("name","Cash / Money Market")[:48],
            "ISIN":    m.get("isin","–") or "–",
            "Bucket":  BUCKET_MAP.get(a,"–"),
            "Sector":  m.get("sektor","Cash") if a != "_CASH" else "Cash",
            "Region":  m.get("region","–")   if a != "_CASH" else "–",
            "Weight":  f"{w:.2%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True, height=500)


# ══════════════════════════════════════════════
# TAB 3 · ASSET METRICS + EFFICIENT FRONTIER
# ══════════════════════════════════════════════
with tab3:

    # ── 3a: Individual cumulative returns ─────
    st.markdown('<span class="chip">Individual Asset Returns</span>',
                unsafe_allow_html=True)
    f = mfig("Cumulative Returns – All Investable Assets", h=500)
    for i, a in enumerate(PRICE_TICKERS):
        r   = RETURNS[a].dropna()
        cum = (1+r).cumprod()
        nm  = ameta(a).get("name",a)[:28]
        bkt = BUCKET_MAP.get(a,"–")
        f.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=a, legendgroup=bkt,
            line=dict(color=BUCKET_COLORS.get(bkt, COLORS[i%len(COLORS)]), width=1.6),
            hovertemplate=f"{a}<br>%{{y:.4f}}<extra>{nm}</extra>",
        ))
    f.add_trace(go.Scatter(
        x=PORT["cum"].index, y=PORT["cum"].values, name="Portfolio",
        line=dict(color="#0F172A", width=2.8, dash="dot"),
        hovertemplate="Portfolio<br>%{y:.4f}<extra></extra>",
    ))
    st.plotly_chart(f, use_container_width=True)

    # ── 3b: Efficient Frontier ────────────────
    st.markdown('<span class="chip">Efficient Frontier</span>',
                unsafe_allow_html=True)
    f = mfig("Efficient Frontier · Investable Assets (unconstrained, long-only)",
             h=560,
             extra=dict(xaxis_title="Ann. Volatility (%)",
                        yaxis_title="Ann. Return (%)"))

    if EF_VOLS:
        f.add_trace(go.Scatter(
            x=EF_VOLS, y=EF_RETS,
            mode="lines", name="Efficient Frontier",
            line=dict(color="#2563EB", width=2.6),
            hovertemplate="σ: %{x:.2f}%<br>μ: %{y:.2f}%<extra>Frontier</extra>",
        ))

    # Individual assets
    for i, a in enumerate(PRICE_TICKERS):
        m   = ASSET_M[a]
        bkt = BUCKET_MAP.get(a,"–")
        if pd.isna(m["ann_vol"]): continue
        f.add_trace(go.Scatter(
            x=[m["ann_vol"]*100], y=[m["ann_return"]*100],
            mode="markers+text",
            text=[a.replace(" Equity","").replace(" Corp","")],
            textposition="top center",
            textfont=dict(size=9, color=BUCKET_COLORS.get(bkt,"#94A3B8")),
            name=bkt, legendgroup=bkt,
            showlegend=i == next(
                (j for j, b in enumerate(PRICE_TICKERS)
                 if BUCKET_MAP.get(b,"–") == bkt), None),
            marker=dict(size=11, color=BUCKET_COLORS.get(bkt,"#94A3B8"),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"{a}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%"
                           f"<extra>{bkt}</extra>"),
        ))

    # Strategic portfolio star
    f.add_trace(go.Scatter(
        x=[PORT_M["ann_vol"]*100], y=[PORT_M["ann_return"]*100],
        mode="markers+text", text=["Strategic Portfolio"],
        textposition="top right",
        textfont=dict(size=12, color="#0F172A"),
        name="Strategic Portfolio",
        marker=dict(size=18, color="#0F172A", symbol="star",
                    line=dict(color="white", width=2)),
        hovertemplate=(f"Strategic Portfolio<br>"
                       f"σ: {PORT_M['ann_vol']*100:.2f}%<br>"
                       f"μ: {PORT_M['ann_return']*100:.2f}%<extra></extra>"),
    ))
    st.plotly_chart(f, use_container_width=True)
    st.caption(
        "The frontier is computed over investable price-data assets (long-only, "
        "no constraints). Assets without price data are excluded; ""bitcoin and catholic index use zero-return proxies if no price series is loaded. "
        "The strategic portfolio may lie inside the frontier because it includes "
        "unoptimised fixed-weight positions (Gold, Cash, Bitcoin, Catholic Index)."
    )

    # ── 3c: Risk-Return scatter ───────────────
    st.markdown('<span class="chip">Risk vs Return (annualised)</span>',
                unsafe_allow_html=True)
    f = mfig("Risk vs Return – Individual Assets",
             h=500,
             extra=dict(xaxis_title="Ann. Volatility (%)",
                        yaxis_title="Ann. Return (%)"))
    for i, a in enumerate(PRICE_TICKERS):
        m   = ASSET_M[a]
        bkt = BUCKET_MAP.get(a,"–")
        if pd.isna(m["ann_vol"]): continue
        f.add_trace(go.Scatter(
            x=[m["ann_vol"]*100], y=[m["ann_return"]*100],
            mode="markers+text",
            text=[a.replace(" Equity","").replace(" Corp","")],
            textposition="top center", textfont=dict(size=10),
            name=bkt, legendgroup=bkt,
            showlegend=i == next(
                (j for j, b in enumerate(PRICE_TICKERS)
                 if BUCKET_MAP.get(b,"–") == bkt), None),
            marker=dict(size=13, color=BUCKET_COLORS.get(bkt,"#94A3B8"),
                        line=dict(color="white", width=1.5)),
            hovertemplate=(f"{a}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%"
                           f"<extra>{bkt}</extra>"),
        ))
    f.add_trace(go.Scatter(
        x=[PORT_M["ann_vol"]*100], y=[PORT_M["ann_return"]*100],
        mode="markers+text", text=["Portfolio"],
        textposition="top right", textfont=dict(size=12, color="#0F172A"),
        name="Strategic Portfolio",
        marker=dict(size=18, color="#0F172A", symbol="star",
                    line=dict(color="white", width=2)),
    ))
    st.plotly_chart(f, use_container_width=True)

    # ── 3d: Metrics table ────────────────────
    st.markdown('<span class="chip">Key Metrics per Asset</span>',
                unsafe_allow_html=True)
    mrows = []
    for a in PRICE_TICKERS:
        m   = ASSET_M[a]
        bkt = BUCKET_MAP.get(a,"–")
        w   = WEIGHTS.get(a, 0.)
        mrows.append({
            "Ticker":       a,
            "Name":         ameta(a).get("name",a)[:38],
            "Bucket":       bkt,
            "Weight":       f"{w:.2%}",
            "Ann. Return":  f"{m['ann_return']*100:.2f}%"  if pd.notna(m['ann_return'])  else "–",
            "Ann. Vol":     f"{m['ann_vol']*100:.2f}%"     if pd.notna(m['ann_vol'])     else "–",
            "Sharpe":       f"{m['sharpe']:.3f}"           if pd.notna(m['sharpe'])      else "–",
            "Sortino":      f"{m['sortino']:.3f}"          if pd.notna(m['sortino'])     else "–",
            "Max DD":       f"{m['max_dd']*100:.2f}%"      if pd.notna(m['max_dd'])      else "–",
            "CVaR 95%":     f"{m['cvar_95']*100:.2f}%"     if pd.notna(m['cvar_95'])     else "–",
            "Total Return": f"{m['total_return']*100:.1f}%" if pd.notna(m['total_return']) else "–",
        })
    mrows.append({
        "Ticker": "PORTFOLIO", "Name": "Strategic Portfolio (all)", "Bucket": "–",
        "Weight": "100.00%",
        "Ann. Return":  f"{PORT_M['ann_return']*100:.2f}%",
        "Ann. Vol":     f"{PORT_M['ann_vol']*100:.2f}%",
        "Sharpe":       f"{PORT_M['sharpe']:.3f}" if pd.notna(PORT_M['sharpe']) else "–",
        "Sortino":      f"{PORT_M['sortino']:.3f}" if pd.notna(PORT_M['sortino']) else "–",
        "Max DD":       f"{PORT_M['max_dd']*100:.2f}%",
        "CVaR 95%":     f"{PORT_M['cvar_95']*100:.2f}%",
        "Total Return": f"{PORT_M['total_return']*100:.1f}%",
    })
    st.dataframe(pd.DataFrame(mrows), use_container_width=True,
                 hide_index=True, height=520)


# ══════════════════════════════════════════════
# TAB 4 · CORRELATIONS
# ══════════════════════════════════════════════
with tab4:

    # ── 4a: Correlation matrix ────────────────
    st.markdown('<span class="chip">Correlation Matrix</span>',
                unsafe_allow_html=True)
    corr  = RETURNS[PRICE_TICKERS].corr()
    short = [a.replace(" Equity","").replace(" Corp","")[:14] for a in corr.columns]
    f = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{x} / %{y}<br>ρ = %{z:.3f}<extra></extra>",
    ))
    f.update_layout(**_lo("Correlation Matrix – Monthly Returns (full history)", h=580))
    st.plotly_chart(f, use_container_width=True)

    # ── 4b: Rolling correlation vs portfolio ──
    st.markdown('<span class="chip">Rolling 12-Month Correlation vs Portfolio</span>',
                unsafe_allow_html=True)
    f = mfig("Rolling Correlation vs Strategic Portfolio (12-month window)", h=420)
    port_r = PORT["returns"]
    for i, a in enumerate(PRICE_TICKERS):
        bkt  = BUCKET_MAP.get(a,"–")
        rc   = RETURNS[a].rolling(12).corr(port_r)
        f.add_trace(go.Scatter(
            x=rc.index, y=rc.values, name=a, legendgroup=bkt,
            line=dict(color=BUCKET_COLORS.get(bkt, COLORS[i%len(COLORS)]), width=1.5),
            hovertemplate=f"{a}<br>ρ: %{{y:.3f}}<extra></extra>",
        ))
    f.add_hline(y=0, line=dict(color="#CBD5E1", width=1, dash="dash"))
    f.update_layout(yaxis_title="Correlation ρ", yaxis_range=[-1.05, 1.05])
    st.plotly_chart(f, use_container_width=True)

    # ── 4c: Pairwise table ────────────────────
    st.markdown('<span class="chip">Pairwise Correlation Summary</span>',
                unsafe_allow_html=True)
    corr_flat = [
        {"Asset A": PRICE_TICKERS[i], "Asset B": PRICE_TICKERS[j],
         "ρ": f"{corr.iloc[i,j]:.3f}", "_v": corr.iloc[i,j]}
        for i in range(len(PRICE_TICKERS))
        for j in range(i+1, len(PRICE_TICKERS))
    ]
    corr_tbl = pd.DataFrame(corr_flat).sort_values("_v").drop("_v", axis=1)
    c1, c2   = st.columns(2)
    with c1:
        st.markdown("**Lowest – best diversifiers**")
        st.dataframe(corr_tbl.head(10), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Highest – most similar pairs**")
        st.dataframe(corr_tbl.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5 · ESG
# ══════════════════════════════════════════════
with tab5:

    # ── ESG score card ────────────────────────
    st.markdown('<span class="chip">ESG Profile</span>', unsafe_allow_html=True)

    if ESG_CONTRIB:
        lbl_avg  = esg_label(ESG_AVG)
        clr_avg  = ESG_BAND_COLORS.get(lbl_avg, "#64748B")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div style="background:white;border:1px solid #E2E8F0;border-radius:14px;
                 padding:2rem 2rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.07);">
              <div style="font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;
                          color:#64748B;font-family:'JetBrains Mono',monospace;margin-bottom:.6rem;">
                Weighted Avg ESG Risk Score
              </div>
              <div style="font-size:3.2rem;font-weight:800;color:{clr_avg};
                          line-height:1;">{ESG_AVG:.2f}</div>
              <div style="font-size:1.1rem;color:{clr_avg};font-weight:600;
                          margin:.3rem 0;">{lbl_avg}</div>
              <hr style="border-color:#E2E8F0;margin:.8rem 0;">
              <div style="font-size:.72rem;color:#94A3B8;line-height:1.6;">
                Sustainalytics scale · lower = better<br>
                Scored sub-portfolio: <strong style="color:#475569;">
                  {_ESG_TW*100:.0f}%</strong> of total<br>
                <em>Gold, Bitcoin &amp; Cash excluded</em>
              </div>
            </div>""", unsafe_allow_html=True)

        with c2:
            # Bar chart: normalised weights within scored sub-portfolio
            scored_sorted = sorted(ESG_CONTRIB, key=lambda x: x[2])
            norm_w = [w / _ESG_TW for _, w, _ in scored_sorted]
            bar_c  = [ESG_BAND_COLORS.get(esg_label(s),"#94A3B8") for _,_,s in scored_sorted]
            f = mfig("ESG Risk Score per Asset  (bar = score, label = normalised weight within scored portfolio)",
                     h=max(320, len(scored_sorted) * 34),
                     extra=dict(xaxis_title="ESG Risk Score (Sustainalytics, lower = better)",
                                yaxis_title=""))
            f.add_trace(go.Bar(
                y=[a for a,_,_ in scored_sorted],
                x=[s for _,_,s in scored_sorted],
                orientation="h",
                marker=dict(color=bar_c, line=dict(color="white", width=0.5)),
                text=[f"{s:.2f} ({esg_label(s)})  |  {nw*100:.1f}% of scored"
                      for (_,_,s), nw in zip(scored_sorted, norm_w)],
                textposition="outside", textfont=dict(size=10),
                hovertemplate="%{y}<br>ESG: %{x:.2f}<extra></extra>",
            ))
            # Portfolio average line
            f.add_vline(x=ESG_AVG,
                        line=dict(color="#2563EB", width=2, dash="dot"),
                        annotation_text=f"Avg: {ESG_AVG:.2f}",
                        annotation_position="bottom right",
                        annotation_font=dict(color="#2563EB", size=11))
            # Sustainalytics band lines
            for xv, lt, lc in [(4.5,"Low","#84CC16"),
                                (5.5,"Medium","#FBBF24"),
                                (6.5,"High","#F59E0B"),
                                (7.5,"Severe","#EF4444")]:
                f.add_vline(x=xv,
                            line=dict(color=lc, width=1, dash="dash"),
                            annotation_text=lt,
                            annotation_position="top",
                            annotation_font=dict(color=lc, size=10))
            st.plotly_chart(f, use_container_width=True)

    st.markdown("---")

    # ── ESG detail table ──────────────────────
    st.markdown('<span class="chip">ESG Detail Table – All Positions</span>',
                unsafe_allow_html=True)
    esg_rows = []
    for a, w in sorted(list(WEIGHTS.items()) + [("_CASH", CASH_WEIGHT)],
                       key=lambda x: -x[1]):
        if a == "_CASH":
            esg_rows.append({
                "Ticker": "CASH", "Name": "Cash / Money Market",
                "Bucket": "Cash", "Weight": f"{w:.2%}",
                "ESG Score": "–", "ESG Label": "–",
                "Scored Weight": "–",
                "Note": "Synthetic position – no ESG score",
            })
            continue
        m  = ameta(a)
        s  = m.get("esg_score", None)
        scored_w = (w / _ESG_TW * 100) if (s is not None and a not in NO_ESG_ASSETS) else None
        esg_rows.append({
            "Ticker":         a,
            "Name":           m.get("name",a)[:45],
            "Bucket":         BUCKET_MAP.get(a,"–"),
            "Weight":         f"{w:.2%}",
            "ESG Score":      f"{s:.2f}" if s is not None else "n/a",
            "ESG Label":      esg_label(s) if s is not None else "n/a",
            "Scored Weight":  f"{scored_w:.1f}%" if scored_w else "excluded",
            "Note":           m.get("esg_note","")[:60],
        })
    st.dataframe(pd.DataFrame(esg_rows), use_container_width=True,
                 hide_index=True, height=520)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'Strategic Portfolio · Fixed Weights · Monthly Returns · '
    f'{RETURNS.index[0].strftime("%b %Y")} – {RETURNS.index[-1].strftime("%b %Y")} · '
    f'r_f = {RF_ANNUAL:.1%} · Cash = {CASH_YIELD_PA:.1%} p.a. · '
    f'ESG avg over {_ESG_TW*100:.0f}% scored sub-portfolio · '
    f'Bitcoin: {"real data" if "bitcoin" not in PROXY_TICKERS else "zero-return proxy"} · '
    f'Catholic Index: {"real data" if "catholic index" not in PROXY_TICKERS else "zero-return proxy"}</p>',
    unsafe_allow_html=True,
)
