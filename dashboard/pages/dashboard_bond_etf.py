"""
bond_etf_optimizer.py  –  Bond ETF Portfolio Optimizer
=======================================================
Ablegen: dashboard/pages/bond_etf_optimizer.py
Starten: streamlit run dashboard/app.py

Universum:  nur Bond ETFs (asset_class == "Bond ETF")
Portfolios: Equal Weight · Max Utility (λ) · Min CVaR 95%
Constraints: keine – freie Optimierung über alle Bond ETFs
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
# PAGE CONFIG + STYLING
# ─────────────────────────────────────────────
st.set_page_config(page_title="Bond ETF Optimizer", page_icon="🔵",
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

DEFAULT_LOOKBACK = 60
DEFAULT_TEST_PCT = 0.30
DEFAULT_LAMBDA   = 4.0

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
# DATA  –  nur Bond ETFs
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def _load():
    return load_new_portfolio(PROJECT_ROOT / "data" / "New Portfolio")

data           = _load()
PM_ALL         = data["prices_monthly"]
asset_meta_ldr = data["asset_meta"]
META_DF        = get_meta_df()

if PM_ALL.empty:
    st.error("No price data available."); st.stop()

def ameta(a: str) -> dict:
    m = dict(asset_meta_ldr.get(a, {}))
    if a in META_DF.index:
        row = META_DF.loc[a]
        m["klasse"]      = row.get("klasse") or row.get("asset_class") or m.get("asset_class","–")
        m["sektor"]      = row.get("sektor") or row.get("sector") or "–"
        m["region"]      = row.get("region", "–")
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

# Filter: nur Bond ETFs die auch Preisdaten haben
BOND_ETF_UNIVERSE = [
    a for a in PM_ALL.columns
    if ameta(a).get("asset_class","").lower() in ("bond etf", "fund")
]

if not BOND_ETF_UNIVERSE:
    st.error("No Bond ETFs found in price data."); st.stop()

A_COLORS = {a: COLORS[i % len(COLORS)] for i, a in enumerate(BOND_ETF_UNIVERSE)}


# ─────────────────────────────────────────────
# MATH
# ─────────────────────────────────────────────
def ann_ret(r):
    r = r.dropna()
    return np.nan if r.empty else (1+r).prod()**(MONTHS_PER_YEAR/len(r))-1

def ann_vol(r):
    r = r.dropna()
    return np.nan if r.empty else r.std()*np.sqrt(MONTHS_PER_YEAR)

def cvar_loss(r, alpha=ALPHA):
    r = r.dropna()
    if r.empty: return np.nan
    tail = r[r <= r.quantile(1-alpha)]
    return np.nan if tail.empty else -tail.mean()

def port_rets(w, X):
    return pd.Series(X.values @ w, index=X.index)

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
    idx = max(int(len(X)*(1-pct)), 12)
    idx = min(idx, len(X)-3)
    return X.iloc[:idx].copy(), X.iloc[idx:].copy()

def dd_s(cum):
    return (cum - cum.cummax()) / cum.cummax() * 100


# ─────────────────────────────────────────────
# OPTIMIZERS  (keine Constraints außer sum==1)
# ─────────────────────────────────────────────
CONS_SUM1 = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

def _solve(obj_fn, n, n_restarts=6):
    """SLSQP, mehrere Starts, nur sum==1 Constraint, keine bounds."""
    bounds = [(0.0, 1.0)] * n
    rng    = np.random.default_rng(42)
    best_x, best_f = None, np.inf

    for k in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n)) if k > 0 else np.ones(n)/n
        res = minimize(obj_fn, w0, method="SLSQP",
                       bounds=bounds, constraints=CONS_SUM1,
                       options={"maxiter": 5000, "ftol": 1e-12, "eps": 1e-8})
        if not res.success: continue
        f = obj_fn(res.x)
        if abs(res.x.sum()-1.0) < 1e-4 and f < best_f:
            best_f = f; best_x = res.x.copy()

    return (best_x if best_x is not None else np.ones(n)/n), best_x is not None

def opt_utility(X, lam):
    mu  = X.mean().values; cov = X.cov().values
    def obj(w): return -(float(w@mu) - lam*float(w@cov@w))
    return _solve(obj, X.shape[1])

def opt_cvar(X):
    def obj(w):
        l = cvar_loss(port_rets(w, X), ALPHA)
        return 1e6 if pd.isna(l) else l
    return _solve(obj, X.shape[1])

def eff_frontier(X, n_pts=30):
    """Unconstrained efficient frontier (only sum==1)."""
    n = X.shape[1]; mu = X.mean().values; cov = X.cov().values
    bounds = [(0.0, 1.0)] * n
    vols, rets = [], []
    for t in np.linspace(mu.min(), mu.max()*0.98, n_pts):
        c = CONS_SUM1 + [{"type":"ineq","fun": lambda w, t=t: float(w@mu)-t}]
        r = minimize(lambda w: float(w@cov@w), np.ones(n)/n, method="SLSQP",
                     bounds=bounds, constraints=c,
                     options={"maxiter": 2000, "ftol": 1e-10})
        if not r.success: continue
        pr = port_rets(r.x, X)
        vols.append(ann_vol(pr)*100); rets.append(ann_ret(pr)*100)
    return vols, rets


# ─────────────────────────────────────────────
# MAIN OPTIMIZATION  (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Optimising …")
def run_opt(assets: tuple, lookback_months: int, lam: float, test_pct: float):
    assets  = list(assets)
    prices  = PM_ALL[assets].iloc[-lookback_months:].copy()
    returns = prices.pct_change().dropna(how="any")

    if len(returns) < 12:
        raise ValueError(f"Only {len(returns)} months available – increase lookback.")

    X_train, X_test = split_tt(returns, test_pct)
    n = len(assets)

    w_ew              = np.ones(n) / n
    w_util, util_ok   = opt_utility(X_train, lam)
    w_cvar, cvar_ok   = opt_cvar(X_train)

    warnings_list = []
    if not util_ok: warnings_list.append("Max Utility: solver did not converge – using equal weight fallback.")
    if not cvar_ok: warnings_list.append("Min CVaR: solver did not converge – using equal weight fallback.")

    weights_map = {"Equal Weight": w_ew, "Max Utility": w_util, "Min CVaR": w_cvar}
    portfolios  = {
        label: pf_metrics(port_rets(w, X_test), dict(zip(assets, w)), label)
        for label, w in weights_map.items()
    }

    er = ((1 + X_train.mean()) ** MONTHS_PER_YEAR) - 1
    er_df = pd.DataFrame({
        "Asset":      assets,
        "Name":       [ameta(a).get("name", a)         for a in assets],
        "Sector":     [ameta(a).get("sektor","–")       for a in assets],
        "Region":     [ameta(a).get("region","–")       for a in assets],
        "ESG":        [ameta(a).get("esg_score", None)  for a in assets],
        "E(r)":       er.values,
        "w Utility":  [portfolios["Max Utility"]["weights"].get(a,0.) for a in assets],
        "w CVaR":     [portfolios["Min CVaR"]["weights"].get(a,0.)    for a in assets],
    }).sort_values("E(r)", ascending=False)

    ef_v, ef_r = eff_frontier(X_train)

    return dict(
        assets=assets, returns_all=returns,
        X_train=X_train, X_test=X_test,
        portfolios=portfolios, er_df=er_df,
        ef_vols=ef_v, ef_rets=ef_r,
        window_start=prices.index[0], window_end=prices.index[-1],
        solver_warnings=warnings_list,
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def _lbl(t):
    st.markdown(f'<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
                f'color:#94A3B8;margin-bottom:.3rem;">{t}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔵 Bond ETF Optimizer")
    st.markdown('<p style="font-size:.75rem;color:#CBD5E1;margin-top:-.4rem;">'
                "No constraints · free optimisation</p>", unsafe_allow_html=True)
    st.markdown("---")

    _lbl("Bond ETFs")
    # Show name + ticker in multiselect
    label_map = {a: f"{a}  –  {ameta(a).get('name','')[:35]}" for a in BOND_ETF_UNIVERSE}
    sel_labels = st.multiselect(
        "be", list(label_map.values()), default=list(label_map.values()),
        label_visibility="collapsed"
    )
    inv_map    = {v: k for k, v in label_map.items()}
    sel_assets = [inv_map[l] for l in sel_labels]
    st.markdown("---")

    _lbl("Optimisation")
    lookback_months = st.slider("Lookback (months)", 12, len(PM_ALL),
                                DEFAULT_LOOKBACK, 6,
                                help=f"Available: {len(PM_ALL)} months")
    lam      = st.slider("λ Utility",  0.0, 20.0, DEFAULT_LAMBDA,   0.5)
    test_pct = st.slider("Test share", 0.10, 0.40, DEFAULT_TEST_PCT, 0.05)
    st.markdown("---")
    run_btn = st.button("▶  Run optimisation", type="primary")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
if len(sel_assets) < 2:
    st.warning("Please select at least 2 Bond ETFs."); st.stop()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_btn or "bond_res" not in st.session_state:
    try:
        st.session_state["bond_res"] = run_opt(
            tuple(sel_assets), lookback_months, lam, test_pct
        )
        st.session_state["bond_params"] = dict(
            lookback_months=lookback_months, lam=lam, test_pct=test_pct
        )
    except Exception as e:
        st.error(f"Optimisation failed: {e}"); st.stop()

res    = st.session_state["bond_res"]
params = st.session_state["bond_params"]
pf     = res["portfolios"]

for w in res.get("solver_warnings", []):
    st.warning(f"⚠️ {w}")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding:.6rem 0 .4rem;">
  <span class="chip">Bond ETF Universe · Unconstrained</span>
  <h1 style="margin:.3rem 0 0;font-size:1.9rem;font-weight:800;color:#0F172A;">
    Bond ETF Portfolio Optimisation
  </h1>
  <p style="color:#94A3B8;font-size:.75rem;margin-top:.25rem;font-family:'JetBrains Mono',monospace;">
    {len(res["assets"])} Bond ETFs &nbsp;·&nbsp;
    {params["lookback_months"]} months lookback &nbsp;·&nbsp;
    {res["window_start"].strftime("%b %Y")} – {res["window_end"].strftime("%b %Y")} &nbsp;·&nbsp;
    λ = {params["lam"]:.1f} &nbsp;·&nbsp; test = {params["test_pct"]:.0%} &nbsp;·&nbsp;
    r_f = {RF_ANNUAL:.1%}
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Highlights (out-of-sample)</span>', unsafe_allow_html=True)

best_ret  = max(["Max Utility","Min CVaR"], key=lambda k: pf[k]["ann_return"] or -np.inf)
best_cvar = min(["Max Utility","Min CVaR"], key=lambda k: pf[k]["cvar_95"]    or  np.inf)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Best Ann. Return",   best_ret,  f'{pf[best_ret]["ann_return"]:.2%}')
c2.metric("Best CVaR 95%",      best_cvar, f'{pf[best_cvar]["cvar_95"]:.2%}')
c3.metric("λ (Utility)",        f'{params["lam"]:.1f}')
c4.metric("Test period",
          f'{res["X_test"].index[0].strftime("%b %Y")} – {res["X_test"].index[-1].strftime("%b %Y")}')
st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Performance", "⚖️ Weights", "🎯 Efficient Frontier", "📊 Metrics"
])


# ══════════════════════════════════════════════
# TAB 1 · PERFORMANCE
# ══════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        f = mfig("Cumulative Returns – Out-of-Sample", h=380)
        for nm in PF_NAMES:
            s = pf[nm]["cum"]
            if s.empty: continue
            f.add_trace(go.Scatter(x=s.index, y=s.values, name=nm,
                line=dict(color=PF_COLORS[nm], width=2.4),
                hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>"))
        st.plotly_chart(f, use_container_width=True)

    with c2:
        f = mfig("Drawdown (%)", h=380)
        for nm in PF_NAMES:
            d = dd_s(pf[nm]["cum"])
            if d.empty: continue
            f.add_trace(go.Scatter(x=d.index, y=d.values, name=nm,
                line=dict(color=PF_COLORS[nm], width=2.2),
                hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>"))
        st.plotly_chart(f, use_container_width=True)

    st.markdown('<span class="chip">Individual Bond ETF Performance</span>',
                unsafe_allow_html=True)

    f = mfig("Cumulative Returns – all Bond ETFs", h=420)
    for i, a in enumerate(res["assets"]):
        r   = res["returns_all"][a].dropna()
        cum = (1+r).cumprod()
        f.add_trace(go.Scatter(x=cum.index, y=cum.values,
            name=a, line=dict(color=A_COLORS[a], width=1.4),
            hovertemplate=f"{a}<br>%{{y:.4f}}<extra></extra>"))
    st.plotly_chart(f, use_container_width=True)

    st.markdown('<span class="chip">Rolling CVaR 95% (6 months)</span>',
                unsafe_allow_html=True)
    f = mfig(h=280)
    def _rc(x):
        l = cvar_loss(pd.Series(x), ALPHA)
        return l*100 if pd.notna(l) else np.nan
    for nm in PF_NAMES:
        rc = pf[nm]["returns"].rolling(6).apply(_rc, raw=False)
        f.add_trace(go.Scatter(x=rc.index, y=rc.values, name=nm,
            line=dict(color=PF_COLORS[nm], width=1.8)))
    f.update_layout(yaxis_title="CVaR 95% (%)")
    st.plotly_chart(f, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 · WEIGHTS
# ══════════════════════════════════════════════
with tab2:
    sel_pf = st.selectbox("Portfolio", ["Max Utility", "Min CVaR"], index=0)
    c1, c2 = st.columns(2)

    with c1:
        nz = {k: v for k, v in pf[sel_pf]["weights"].items() if v > 0.001}
        f  = go.Figure(go.Pie(
            labels=list(nz.keys()), values=list(nz.values()), hole=0.55,
            marker=dict(colors=[A_COLORS.get(a,"#999") for a in nz],
                        line=dict(color="white", width=2)),
            textinfo="percent+label", textfont=dict(size=9),
        ))
        f.update_layout(**_lo(f"Weights · {sel_pf}", h=400))
        st.plotly_chart(f, use_container_width=True)

    with c2:
        wdf = pd.DataFrame({
            "Asset": res["assets"],
            **{nm: [pf[nm]["weights"].get(a, 0.) for a in res["assets"]] for nm in PF_NAMES}
        }).sort_values(sel_pf, ascending=True)

        f = go.Figure()
        for nm in PF_NAMES:
            f.add_trace(go.Bar(
                y=wdf["Asset"], x=wdf[nm]*100, name=nm, orientation="h",
                marker=dict(color=PF_COLORS[nm]),
                hovertemplate="%{y}: %{x:.1f}%<extra>"+nm+"</extra>",
            ))
        f.update_layout(**_lo("Weight comparison", h=max(380, len(res["assets"])*28),
                              extra=dict(barmode="group", xaxis_title="Weight (%)")))
        st.plotly_chart(f, use_container_width=True)

    st.markdown('<span class="chip">Full weight table</span>', unsafe_allow_html=True)
    wtbl = pd.DataFrame({
        "Ticker":    res["assets"],
        "Name":      [ameta(a).get("name","")[:45]        for a in res["assets"]],
        "Sector":    [ameta(a).get("sektor","–")           for a in res["assets"]],
        "Region":    [ameta(a).get("region","–")           for a in res["assets"]],
        "ESG":       [f'{ameta(a).get("esg_score","–"):.2f} ({esg_label(ameta(a).get("esg_score"))})'
                      if ameta(a).get("esg_score") else "–"
                      for a in res["assets"]],
        **{nm: [f'{pf[nm]["weights"].get(a,0.):.2%}' for a in res["assets"]] for nm in PF_NAMES}
    }).sort_values("Max Utility", ascending=False)
    st.dataframe(wtbl, use_container_width=True, hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 3 · EFFICIENT FRONTIER
# ══════════════════════════════════════════════
with tab3:
    f = mfig("Efficient Frontier – Bond ETF Universe (unconstrained)", h=540,
             extra=dict(xaxis_title="Ann. Volatility (%)", yaxis_title="Ann. Return (%)"))

    if res["ef_vols"]:
        f.add_trace(go.Scatter(
            x=res["ef_vols"], y=res["ef_rets"],
            mode="lines+markers", name="Efficient Frontier",
            line=dict(color="#2563EB", width=2.4), marker=dict(size=5),
            hovertemplate="σ: %{x:.3f}%<br>μ: %{y:.3f}%<extra></extra>",
        ))

    for nm in PF_NAMES:
        p = pf[nm]
        if pd.isna(p["ann_vol"]): continue
        f.add_trace(go.Scatter(
            x=[p["ann_vol"]*100], y=[p["ann_return"]*100],
            mode="markers+text", text=[nm], textposition="top center", name=nm,
            marker=dict(size=14, color=PF_COLORS[nm], line=dict(color="white", width=1.5)),
            hovertemplate=f"{nm}<br>σ: %{{x:.3f}}%<br>μ: %{{y:.3f}}%<extra></extra>",
        ))

    for a in res["assets"]:
        r = res["X_test"][a].dropna()
        f.add_trace(go.Scatter(
            x=[ann_vol(r)*100], y=[ann_ret(r)*100],
            mode="markers", name=a,
            marker=dict(size=8, color=A_COLORS[a], opacity=0.65,
                        line=dict(color="white", width=0.5)),
            hovertemplate=f"{a}<br>σ: %{{x:.3f}}%<br>μ: %{{y:.3f}}%<extra></extra>",
            showlegend=True,
        ))

    st.plotly_chart(f, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 · METRICS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<span class="chip">Portfolio comparison (out-of-sample)</span>',
                unsafe_allow_html=True)

    comp = pd.DataFrame([{
        "Portfolio":    nm,
        "Ann. Return":  pf[nm]["ann_return"],
        "Ann. Vol":     pf[nm]["ann_vol"],
        "Sharpe":       pf[nm]["sharpe"],
        "Max DD":       pf[nm]["max_dd"],
        "CVaR 95%":     pf[nm]["cvar_95"],
        "Total Return": pf[nm]["total_return"],
    } for nm in PF_NAMES]).set_index("Portfolio")

    disp = comp.copy()
    for col in ["Ann. Return","Ann. Vol","Max DD","Total Return","CVaR 95%"]:
        disp[col] = comp[col].map(lambda x: f"{x:.3%}" if pd.notna(x) else "–")
    disp["Sharpe"] = comp["Sharpe"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    st.dataframe(disp, use_container_width=True, height=200)

    st.markdown("---")
    st.markdown('<span class="chip">Correlation matrix (full history)</span>',
                unsafe_allow_html=True)

    corr  = res["returns_all"].corr()
    short = [a.replace(" Equity","").replace(" Corp","")[:16] for a in corr.columns]
    f = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale="RdYlGn_r", zmin=-1, zmax=1,   # reversed: low corr = green for bonds
        text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    f.update_layout(**_lo("Correlation Matrix – Bond ETFs (monthly returns)", h=460))
    st.plotly_chart(f, use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="chip">Expected returns + weights (training set)</span>',
                unsafe_allow_html=True)
    er = res["er_df"].copy()
    er["ESG"]       = er["ESG"].apply(lambda x: f"{x:.2f} ({esg_label(x)})"
                                      if pd.notna(x) and x is not None else "–")
    er["E(r)"]      = er["E(r)"].map(lambda x: f"{x:.3%}")
    er["w Utility"] = er["w Utility"].map(lambda x: f"{x:.2%}")
    er["w CVaR"]    = er["w CVaR"].map(lambda x: f"{x:.2%}")
    st.dataframe(er, use_container_width=True, hide_index=True, height=420)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'Bond ETF Universe · {params["lookback_months"]} M lookback · '
    f'λ={params["lam"]:.1f} · test={params["test_pct"]:.0%} · '
    f'r_f={RF_ANNUAL:.1%} · No constraints</p>',
    unsafe_allow_html=True,
)
