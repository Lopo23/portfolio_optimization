"""
optimization_dashboard.py  –  New Portfolio Optimizer
======================================================
Konsistente MONATLICHE Portfolio-Optimierung für 3 Portfolio-Typen:

1) Utility / Mean-Variance   (mit wählbarem Lambda)
2) Max Sortino Ratio
3) Min CVaR

Starten:
    streamlit run dashboard/pages/optimization_dashboard.py

Pakete:
    pip install streamlit plotly pandas numpy scipy openpyxl
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

[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * {
    font-family: 'Syne', sans-serif !important;
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p {
    color: #CBD5E1 !important;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 0.65rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #0F172A !important;
    font-size: 1.25rem !important;
    font-weight: 700;
}

.chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #3B82F6;
    border: 1px solid #BFDBFE;
    background: #EFF6FF;
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: .5rem;
}

hr {
    border-color: #E2E8F0 !important;
    margin: 1.5rem 0;
}

[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RF_ANNUAL = 0.02017
MONTHS_PER_YEAR = 12
ALPHA = 0.95

COLORS = [
    "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#06B6D4", "#F97316", "#84CC16", "#EC4899", "#14B8A6",
    "#A78BFA", "#FB923C", "#34D399", "#60A5FA", "#FBBF24",
]

PORTFOLIO_COLORS = {
    "Equal Weight": "#94A3B8",
    "Utility": "#2563EB",
    "Max Sortino": "#10B981",
    "Min CVaR": "#F59E0B",
}

def _base_layout(title="", h=360, extra=None):
    layout = dict(
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Syne", color="#64748B", size=11),
        margin=dict(l=10, r=10, t=44, b=10),
        height=h,
        title=dict(text=title, font=dict(family="Syne", size=13, color="#0F172A")),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zeroline=False),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#475569"),
        ),
    )
    if extra:
        layout.update(extra)
    return layout

def make_fig(title="", h=360, extra=None):
    fig = go.Figure()
    fig.update_layout(**_base_layout(title, h, extra))
    return fig


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Lade Portfolio-Daten …")
def get_data():
    return load_new_portfolio(PROJECT_ROOT / "data" / "New Portfolio")

data = get_data()
prices_monthly = data["prices_monthly"]
asset_meta = data["asset_meta"]
asset_classes = data["asset_classes"]

ALL_ASSETS = sorted(prices_monthly.columns.tolist())
ASSET_COLORS = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ALL_ASSETS)}

if prices_monthly.empty:
    st.error("Keine monatlichen Preisdaten verfügbar.")
    st.stop()


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def annualize_return(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan
    total = (1 + r).prod() - 1
    return (1 + total) ** (MONTHS_PER_YEAR / len(r)) - 1

def annualize_vol(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan
    return r.std() * np.sqrt(MONTHS_PER_YEAR)

def cvar_loss(monthly_returns: pd.Series, alpha: float = ALPHA) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan
    threshold = r.quantile(1 - alpha)
    tail = r[r <= threshold]
    if tail.empty:
        return np.nan
    return -tail.mean()

def sortino_ratio(monthly_returns: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return np.nan

    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
    excess = r - rf_monthly

    ann_r = annualize_return(r)
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan

    downside_dev = downside.std() * np.sqrt(MONTHS_PER_YEAR)
    if downside_dev <= 0 or pd.isna(downside_dev):
        return np.nan

    return (ann_r - rf_annual) / downside_dev

def compute_portfolio_metrics(returns: pd.Series, weights: dict, label: str) -> dict:
    r = returns.dropna()
    if r.empty:
        return {
            "label": label,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_dd": np.nan,
            "cvar_95": np.nan,
            "total_return": np.nan,
            "weights": weights,
            "returns": r,
            "cumulative": pd.Series(dtype=float),
        }

    cum = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    ann_r = annualize_return(r)
    ann_v = annualize_vol(r)
    sharpe = (ann_r - RF_ANNUAL) / ann_v if ann_v and ann_v > 0 else np.nan
    sortino = sortino_ratio(r, RF_ANNUAL)

    roll = cum.cummax()
    max_dd = ((cum - roll) / roll).min()

    cvar95 = cvar_loss(r, ALPHA)

    return {
        "label": label,
        "ann_return": ann_r,
        "ann_vol": ann_v,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cvar_95": -cvar95 if pd.notna(cvar95) else np.nan,  # als negative Return-Zahl anzeigen
        "total_return": total,
        "weights": weights,
        "returns": r,
        "cumulative": cum,
    }

def build_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")

def project_to_full_weights(asset_names, w):
    return dict(zip(asset_names, w))

def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n

def portfolio_returns(weights: np.ndarray, X: pd.DataFrame) -> pd.Series:
    return pd.Series(X.values @ weights, index=X.index)

def weight_constraints(n_assets: int, min_w: float, max_w: float):
    bounds = [(min_w, max_w) for _ in range(n_assets)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, cons

def optimize_utility(X_train: pd.DataFrame, lam: float, min_w: float, max_w: float) -> np.ndarray:
    n = X_train.shape[1]
    mu = X_train.mean().values
    cov = X_train.cov().values

    bounds, cons = weight_constraints(n, min_w, max_w)
    w0 = equal_weight(n)

    def obj(w):
        mean_m = float(w @ mu)
        var_m = float(w @ cov @ w)
        return -(mean_m - lam * var_m)

    res = minimize(
        obj, w0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return res.x if res.success else w0

def optimize_sortino(X_train: pd.DataFrame, min_w: float, max_w: float) -> np.ndarray:
    n = X_train.shape[1]
    rf_monthly = (1 + RF_ANNUAL) ** (1 / MONTHS_PER_YEAR) - 1

    bounds, cons = weight_constraints(n, min_w, max_w)
    w0 = equal_weight(n)

    def obj(w):
        pr = portfolio_returns(w, X_train)
        excess = pr - rf_monthly
        downside = excess[excess < 0]
        if downside.empty:
            return 1e6
        downside_dev = downside.std()
        if downside_dev <= 1e-12 or pd.isna(downside_dev):
            return 1e6
        mean_excess = excess.mean()
        ratio = mean_excess / downside_dev
        return -ratio

    res = minimize(
        obj, w0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return res.x if res.success else w0

def optimize_cvar(X_train: pd.DataFrame, min_w: float, max_w: float, alpha: float = ALPHA) -> np.ndarray:
    n = X_train.shape[1]
    bounds, cons = weight_constraints(n, min_w, max_w)
    w0 = equal_weight(n)

    def obj(w):
        pr = portfolio_returns(w, X_train)
        loss = cvar_loss(pr, alpha)
        if pd.isna(loss):
            return 1e6
        return loss

    res = minimize(
        obj, w0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return res.x if res.success else w0

def split_train_test(X: pd.DataFrame, test_pct: float):
    split_idx = int(len(X) * (1 - test_pct))
    split_idx = max(split_idx, 12)
    split_idx = min(split_idx, len(X) - 3)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    return X_train, X_test

def max_drawdown_series(cumulative: pd.Series) -> pd.Series:
    roll = cumulative.cummax()
    return (cumulative - roll) / roll * 100

def calc_efficient_frontier_points(X_train: pd.DataFrame, min_w: float, max_w: float, n_points: int = 20):
    n = X_train.shape[1]
    mu = X_train.mean().values
    cov = X_train.cov().values
    bounds, cons_base = weight_constraints(n, min_w, max_w)
    w0 = equal_weight(n)

    def solve_min_var(target_return):
        cons = cons_base + [
            {"type": "ineq", "fun": lambda w, tr=target_return: float(w @ mu) - tr}
        ]

        def obj(w):
            return float(w @ cov @ w)

        res = minimize(
            obj, w0, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        return res

    mu_min = mu.min()
    mu_max = mu.max()
    targets = np.linspace(mu_min, mu_max, n_points)

    vols = []
    rets = []

    for tr in targets:
        res = solve_min_var(tr)
        if not res.success:
            continue
        w = res.x
        port_r = portfolio_returns(w, X_train)
        vols.append(annualize_vol(port_r) * 100)
        rets.append(annualize_return(port_r) * 100)

    return vols, rets


# ─────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Optimiere Portfolios …")
def run_optimization(
    assets: tuple,
    lookback_years: int,
    lambda_utility: float,
    min_w: float,
    max_w: float,
    test_pct: float,
):
    assets = list(assets)

    prices = prices_monthly[assets].copy()

    end_date = prices.index[-1]
    start_date = end_date - pd.DateOffset(years=lookback_years)
    prices = prices[prices.index >= start_date].copy()

    if prices.empty:
        raise ValueError("Kein Preisfenster nach Lookback-Filter verfügbar.")

    returns = build_monthly_returns(prices)

    if len(returns) < 18:
        raise ValueError("Zu wenige Monatsrenditen für Optimierung. Bitte mehr Historie oder weniger Assets wählen.")

    X_train, X_test = split_train_test(returns, test_pct)

    if len(X_train) < 12 or len(X_test) < 3:
        raise ValueError("Train/Test-Split ergibt zu wenige Beobachtungen.")

    n = len(assets)

    # Benchmark
    w_eq = equal_weight(n)

    # Optimierte Portfolios
    w_util = optimize_utility(X_train, lambda_utility, min_w, max_w)
    w_sort = optimize_sortino(X_train, min_w, max_w)
    w_cvar = optimize_cvar(X_train, min_w, max_w, ALPHA)

    portfolios = {}

    for label, w in [
        ("Equal Weight", w_eq),
        ("Utility", w_util),
        ("Max Sortino", w_sort),
        ("Min CVaR", w_cvar),
    ]:
        r_test = portfolio_returns(w, X_test)
        portfolios[label] = compute_portfolio_metrics(
            returns=r_test,
            weights=project_to_full_weights(assets, w),
            label=label,
        )

    # Erwartete Renditen im Training
    er_annual = ((1 + X_train.mean()) ** MONTHS_PER_YEAR) - 1
    er_df = pd.DataFrame({
        "Asset": assets,
        "Klasse": [asset_meta.get(a, {}).get("asset_class", "Other") for a in assets],
        "E(r) ann.": er_annual.values,
        "Gewicht Utility": [portfolios["Utility"]["weights"].get(a, 0.0) for a in assets],
        "Gewicht Sortino": [portfolios["Max Sortino"]["weights"].get(a, 0.0) for a in assets],
        "Gewicht CVaR": [portfolios["Min CVaR"]["weights"].get(a, 0.0) for a in assets],
    }).sort_values("E(r) ann.", ascending=False)

    ef_vols, ef_rets = calc_efficient_frontier_points(X_train, min_w, max_w, n_points=22)

    return {
        "prices_window": prices,
        "returns_all": returns,
        "X_train": X_train,
        "X_test": X_test,
        "assets": assets,
        "portfolios": portfolios,
        "er_df": er_df,
        "ef_vols": ef_vols,
        "ef_rets": ef_rets,
        "window_start": prices.index[0],
        "window_end": prices.index[-1],
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Optimizer")
    st.markdown(
        '<p style="font-size:.75rem;color:#CBD5E1;margin-top:-.4rem;">'
        'Monatliche Optimierung für 3 Portfolios.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        '<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        'color:#94A3B8;margin-bottom:.3rem;">Asset-Klassen</p>',
        unsafe_allow_html=True,
    )
    all_classes = sorted(asset_classes.keys())
    sel_classes = st.multiselect(
        "Asset-Klassen",
        all_classes,
        default=all_classes,
        label_visibility="collapsed",
    )

    filtered_assets = [
        a for a in ALL_ASSETS
        if asset_meta.get(a, {}).get("asset_class", "Other") in sel_classes
    ]

    st.markdown(
        '<p style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;'
        'color:#94A3B8;margin:.8rem 0 .3rem;">Asset-Auswahl</p>',
        unsafe_allow_html=True,
    )
    default_assets = filtered_assets[:]
    if len(default_assets) > 12:
        default_assets = default_assets[:12]

    sel_assets = st.multiselect(
        "Assets",
        filtered_assets,
        default=default_assets,
        label_visibility="collapsed",
    )

    st.markdown("---")

    lookback_years = st.slider("Lookback (Jahre)", 3, 10, 5, 1)
    lambda_utility = st.slider("λ Utility", 0.0, 20.0, 4.0, 0.5)
    max_w = st.slider("Max. Gewicht je Asset", 0.05, 1.00, 0.35, 0.05)
    min_w = st.slider("Min. Gewicht je Asset", 0.00, 0.10, 0.00, 0.01)
    test_pct = st.slider("Test-Anteil", 0.10, 0.40, 0.30, 0.05)

    st.markdown("---")
    st.caption(
        "Portfolios: Utility (λ), Max Sortino, Min CVaR\n\n"
        "Frequenz: monatlich\n\n"
        "Benchmark: Equal Weight"
    )

    run_clicked = st.button("▶ Optimierung starten", type="primary")

if len(sel_assets) < 2:
    st.warning("Bitte mindestens 2 Assets auswählen.")
    st.stop()

if min_w * len(sel_assets) > 1:
    st.error("Min-Gewicht ist zu hoch für die Anzahl der Assets.")
    st.stop()

if max_w * len(sel_assets) < 1:
    st.error("Max-Gewicht ist zu niedrig für die Anzahl der Assets.")
    st.stop()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_clicked or "opt_res" not in st.session_state:
    try:
        st.session_state["opt_res"] = run_optimization(
            assets=tuple(sel_assets),
            lookback_years=lookback_years,
            lambda_utility=lambda_utility,
            min_w=min_w,
            max_w=max_w,
            test_pct=test_pct,
        )
        st.session_state["opt_params"] = {
            "lookback_years": lookback_years,
            "lambda_utility": lambda_utility,
            "min_w": min_w,
            "max_w": max_w,
            "test_pct": test_pct,
            "assets": sel_assets,
        }
    except Exception as e:
        st.error(f"Optimierung fehlgeschlagen: {e}")
        st.stop()

res = st.session_state["opt_res"]
params = st.session_state["opt_params"]
portfolios = res["portfolios"]

eq = portfolios["Equal Weight"]
util = portfolios["Utility"]
sortino_p = portfolios["Max Sortino"]
cvar_p = portfolios["Min CVaR"]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding:.6rem 0 .4rem;">
    <span class="chip">New Portfolio · Optimizer</span>
    <h1 style="margin:.3rem 0 0;font-size:1.9rem;font-weight:800;color:#0F172A;">
        Portfolio Optimierung
    </h1>
    <p style="color:#94A3B8;font-size:.75rem;margin-top:.25rem;
              font-family:'JetBrains Mono',monospace;">
        {len(res["assets"])} Assets &nbsp;·&nbsp;
        Monatsdaten &nbsp;·&nbsp;
        Lookback: {params["lookback_years"]} Jahre &nbsp;·&nbsp;
        λ Utility = {params["lambda_utility"]:.1f} &nbsp;·&nbsp;
        Test = {params["test_pct"]:.0%} &nbsp;·&nbsp;
        r_f = {RF_ANNUAL:.0%} p.a.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<span class="chip">Highlights</span>', unsafe_allow_html=True)

best_return_name = max(
    ["Utility", "Max Sortino", "Min CVaR"],
    key=lambda k: portfolios[k]["ann_return"]
)
best_sortino_name = max(
    ["Utility", "Max Sortino", "Min CVaR"],
    key=lambda k: portfolios[k]["sortino"]
)
best_cvar_name = min(
    ["Utility", "Max Sortino", "Min CVaR"],
    key=lambda k: portfolios[k]["cvar_95"]
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Beste Ann. Rendite", best_return_name, f'{portfolios[best_return_name]["ann_return"]:.2%}')
c2.metric("Beste Sortino", best_sortino_name, f'{portfolios[best_sortino_name]["sortino"]:.3f}')
c3.metric("Bestes CVaR", best_cvar_name, f'{portfolios[best_cvar_name]["cvar_95"]:.2%}')
c4.metric("Utility λ", f'{params["lambda_utility"]:.1f}', None)

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "⚖️ Gewichte",
    "🎯 Efficient Frontier",
    "📊 Kennzahlen",
    "🧩 Struktur",
])


# ══════════════════════════════════════════════
# TAB 1 – PERFORMANCE
# ══════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = make_fig("Kumulierte Renditen (Out-of-Sample)", h=360)

        for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
            s = portfolios[name]
            fig.add_trace(go.Scatter(
                x=s["cumulative"].index,
                y=s["cumulative"].values,
                name=name,
                line=dict(color=PORTFOLIO_COLORS[name], width=2.4),
                hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>",
            ))

        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = make_fig("Drawdown (%)", h=360)

        for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
            s = portfolios[name]
            dd = max_drawdown_series(s["cumulative"])
            fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                name=name,
                line=dict(color=PORTFOLIO_COLORS[name], width=2.2),
                hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
            ))

        st.plotly_chart(fig, width="stretch")

    st.markdown('<span class="chip">Rolling Analyse</span>', unsafe_allow_html=True)

    window = 6
    c3, c4 = st.columns(2)

    with c3:
        fig = make_fig(f"Rolling Sortino (Fenster: {window} Monate)", h=300)
        rf_monthly = (1 + RF_ANNUAL) ** (1 / MONTHS_PER_YEAR) - 1

        for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
            r = portfolios[name]["returns"]

            def _roll_sortino(x):
                x = pd.Series(x)
                excess = x - rf_monthly
                downside = excess[excess < 0]
                if len(downside) < 2:
                    return np.nan
                downside_dev = downside.std() * np.sqrt(MONTHS_PER_YEAR)
                if downside_dev <= 0 or pd.isna(downside_dev):
                    return np.nan
                ann_r = (1 + x).prod() ** (MONTHS_PER_YEAR / len(x)) - 1
                return (ann_r - RF_ANNUAL) / downside_dev

            rs = r.rolling(window).apply(_roll_sortino, raw=False)

            fig.add_trace(go.Scatter(
                x=rs.index,
                y=rs.values,
                name=name,
                line=dict(color=PORTFOLIO_COLORS[name], width=1.8),
            ))

        fig.add_hline(y=0, line=dict(color="#CBD5E1", width=1, dash="dash"))
        st.plotly_chart(fig, width="stretch")

    with c4:
        fig = make_fig(f"Rolling CVaR 95% (Fenster: {window} Monate)", h=300)

        for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
            r = portfolios[name]["returns"]

            def _roll_cvar(x):
                x = pd.Series(x)
                loss = cvar_loss(x, ALPHA)
                return loss * 100 if pd.notna(loss) else np.nan

            rc = r.rolling(window).apply(_roll_cvar, raw=False)

            fig.add_trace(go.Scatter(
                x=rc.index,
                y=rc.values,
                name=name,
                line=dict(color=PORTFOLIO_COLORS[name], width=1.8),
            ))

        fig.update_layout(yaxis_title="CVaR 95% (%)")
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════
# TAB 2 – GEWICHTE
# ══════════════════════════════════════════════
with tab2:
    selected_portfolio = st.selectbox(
        "Portfolio für Detailansicht",
        ["Utility", "Max Sortino", "Min CVaR"],
        index=0,
    )

    c1, c2 = st.columns(2)

    with c1:
        nz = {k: v for k, v in portfolios[selected_portfolio]["weights"].items() if v > 0.001}

        fig = go.Figure(go.Pie(
            labels=list(nz.keys()),
            values=list(nz.values()),
            hole=0.55,
            marker=dict(
                colors=[ASSET_COLORS[a] for a in nz.keys()],
                line=dict(color="white", width=2),
            ),
            textinfo="percent+label",
            textfont=dict(size=9),
        ))
        fig.update_layout(**_base_layout(f"Gewichte · {selected_portfolio}", h=360))
        st.plotly_chart(fig, width="stretch")

    with c2:
        wdf = pd.DataFrame({
            "Asset": res["assets"],
            "Equal Weight": [eq["weights"].get(a, 0.0) for a in res["assets"]],
            "Utility": [util["weights"].get(a, 0.0) for a in res["assets"]],
            "Max Sortino": [sortino_p["weights"].get(a, 0.0) for a in res["assets"]],
            "Min CVaR": [cvar_p["weights"].get(a, 0.0) for a in res["assets"]],
        }).sort_values(selected_portfolio, ascending=True)

        fig = go.Figure()
        for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
            fig.add_trace(go.Bar(
                y=wdf["Asset"],
                x=wdf[name] * 100,
                name=name,
                orientation="h",
                marker=dict(color=PORTFOLIO_COLORS[name]),
                hovertemplate="%{y}: %{x:.1f}%<extra>" + name + "</extra>",
            ))

        fig.update_layout(**_base_layout(
            "Gewichtsvergleich",
            h=420,
            extra=dict(barmode="group", xaxis_title="Gewicht (%)"),
        ))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<span class="chip">Vollständige Gewichtstabelle</span>', unsafe_allow_html=True)

    weight_tbl = pd.DataFrame({
        "Asset": res["assets"],
        "Klasse": [asset_meta.get(a, {}).get("asset_class", "Other") for a in res["assets"]],
        "Equal Weight": [eq["weights"].get(a, 0.0) for a in res["assets"]],
        "Utility": [util["weights"].get(a, 0.0) for a in res["assets"]],
        "Max Sortino": [sortino_p["weights"].get(a, 0.0) for a in res["assets"]],
        "Min CVaR": [cvar_p["weights"].get(a, 0.0) for a in res["assets"]],
    }).sort_values("Utility", ascending=False)

    for col in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
        weight_tbl[col] = weight_tbl[col].map(lambda x: f"{x:.2%}")

    st.dataframe(weight_tbl, width="stretch", hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 3 – EFFICIENT FRONTIER
# ══════════════════════════════════════════════
with tab3:
    fig = go.Figure()

    if res["ef_vols"]:
        fig.add_trace(go.Scatter(
            x=res["ef_vols"],
            y=res["ef_rets"],
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(color="#2563EB", width=2.4),
            marker=dict(size=5, color="#2563EB"),
            hovertemplate="σ: %{x:.2f}%<br>μ: %{y:.2f}%<extra>Efficient Frontier</extra>",
        ))

    for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]:
        p = portfolios[name]
        fig.add_trace(go.Scatter(
            x=[p["ann_vol"] * 100],
            y=[p["ann_return"] * 100],
            mode="markers+text",
            text=[name],
            textposition="top center",
            name=name,
            marker=dict(
                size=14,
                color=PORTFOLIO_COLORS[name],
                line=dict(color="white", width=1.5),
            ),
            hovertemplate=f"{name}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%<extra></extra>",
            showlegend=True,
        ))

    # Assets auf Basis Testfenster
    X_test = res["X_test"]
    for a in res["assets"]:
        r = X_test[a].dropna()
        av = annualize_vol(r) * 100
        ar = annualize_return(r) * 100
        fig.add_trace(go.Scatter(
            x=[av],
            y=[ar],
            mode="markers",
            name=a,
            marker=dict(
                size=7,
                color=ASSET_COLORS[a],
                opacity=0.65,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate=f"{a}<br>σ: %{{x:.2f}}%<br>μ: %{{y:.2f}}%<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(**_base_layout(
        "Efficient Frontier · Monatsbasis",
        h=540,
        extra=dict(
            xaxis_title="Annualisierte Volatilität (%)",
            yaxis_title="Annualisierte Rendite (%)",
        ),
    ))
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════
# TAB 4 – KENNZAHLEN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<span class="chip">Portfoliovergleich</span>', unsafe_allow_html=True)

    comp = pd.DataFrame([
        {
            "Portfolio": name,
            "Ann. Rendite": portfolios[name]["ann_return"],
            "Ann. Vola": portfolios[name]["ann_vol"],
            "Sharpe": portfolios[name]["sharpe"],
            "Sortino": portfolios[name]["sortino"],
            "Max Drawdown": portfolios[name]["max_dd"],
            "CVaR 95%": portfolios[name]["cvar_95"],
            "Total Return": portfolios[name]["total_return"],
        }
        for name in ["Equal Weight", "Utility", "Max Sortino", "Min CVaR"]
    ]).set_index("Portfolio")

    display = pd.DataFrame(index=comp.index)
    display["Ann. Rendite"] = comp["Ann. Rendite"].map(lambda x: f"{x:.2%}")
    display["Ann. Vola"] = comp["Ann. Vola"].map(lambda x: f"{x:.2%}")
    display["Sharpe"] = comp["Sharpe"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    display["Sortino"] = comp["Sortino"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    display["Max Drawdown"] = comp["Max Drawdown"].map(lambda x: f"{x:.2%}")
    display["CVaR 95%"] = comp["CVaR 95%"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "–")
    display["Total Return"] = comp["Total Return"].map(lambda x: f"{x:.2%}")

    st.dataframe(display, width="stretch", height=250)

    st.markdown("---")
    st.markdown('<span class="chip">Erwartete Renditen im Training</span>', unsafe_allow_html=True)

    er_df = res["er_df"].copy()
    er_df["E(r) ann."] = er_df["E(r) ann."].map(lambda x: f"{x:.2%}")
    er_df["Gewicht Utility"] = er_df["Gewicht Utility"].map(lambda x: f"{x:.2%}")
    er_df["Gewicht Sortino"] = er_df["Gewicht Sortino"].map(lambda x: f"{x:.2%}")
    er_df["Gewicht CVaR"] = er_df["Gewicht CVaR"].map(lambda x: f"{x:.2%}")

    st.dataframe(er_df, width="stretch", hide_index=True, height=420)


# ══════════════════════════════════════════════
# TAB 5 – STRUKTUR
# ══════════════════════════════════════════════
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        corr = res["returns_all"].corr()
        short = [a.replace(" Equity", "").replace(" Corp", "")[:14] for a in corr.columns]

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=short,
            y=short,
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=7),
            hovertemplate="%{x} / %{y}<br>ρ = %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(**_base_layout("Korrelationsmatrix (Monatsrenditen)", h=460))
        st.plotly_chart(fig, width="stretch")

    with c2:
        cls_w = {}
        for a, w in util["weights"].items():
            cls = asset_meta.get(a, {}).get("asset_class", "Other")
            cls_w[cls] = cls_w.get(cls, 0.0) + w

        fig = go.Figure(go.Pie(
            labels=list(cls_w.keys()),
            values=list(cls_w.values()),
            hole=0.55,
            marker=dict(
                colors=COLORS[:len(cls_w)],
                line=dict(color="white", width=2),
            ),
            textinfo="percent+label",
            textfont=dict(size=10),
        ))
        fig.update_layout(**_base_layout("Asset-Klassen · Utility Portfolio", h=460))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<span class="chip">Asset-Klassen Übersicht</span>', unsafe_allow_html=True)

    cls_tbl = pd.DataFrame([
        {
            "Klasse": cls,
            "Gewicht Utility": w,
            "Anzahl Assets": sum(
                1 for a in res["assets"]
                if asset_meta.get(a, {}).get("asset_class", "Other") == cls
            ),
        }
        for cls, w in sorted(cls_w.items(), key=lambda x: -x[1])
    ])

    if not cls_tbl.empty:
        cls_tbl["Gewicht Utility"] = cls_tbl["Gewicht Utility"].map(lambda x: f"{x:.2%}")
        st.dataframe(cls_tbl, width="stretch", hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="color:#CBD5E1;font-size:.62rem;text-align:center;'
    f'font-family:\'JetBrains Mono\',monospace;">'
    f'New Portfolio · Monatsbasis konsistent · '
    f'{params["lookback_years"]}Y Lookback · '
    f'λ = {params["lambda_utility"]:.1f} · '
    f'test = {params["test_pct"]:.0%} · '
    f'r_f = {RF_ANNUAL:.0%}</p>',
    unsafe_allow_html=True,
)


# constraints
# bitcoin btc <5
# equity single stock < 10
# equity < 40
