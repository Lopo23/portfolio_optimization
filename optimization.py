"""
portfolio_optimization_new.py
==============================
Portfolio-Optimierung für data/New Portfolio mit skfolio.

Optimierungsmodelle:
    1. Equally Weighted          – Benchmark
    2. Mean-Variance             – klassischer Markowitz
    3. CVaR                      – Conditional Value at Risk
    4. CDaR                      – Conditional Drawdown at Risk
    5. Semi-Varianz              – nur Downside-Risiko
    6. Minimum Variance          – reine Risikominimierung
    7. Maximum Sharpe (MVO)      – bestes Rendite/Risiko-Verhältnis

Constraints (alle Modelle):
    - Keine Leerverkäufe (min_weights = 0)
    - Max. 40% je Asset
    - Optionale Asset-Klassen-Constraints

Pakete: pip install skfolio pandas numpy matplotlib seaborn openpyxl
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split

from skfolio import Portfolio
from skfolio.optimization import (
    MeanRisk,
    EqualWeighted,
    InverseVolatility,
)
from skfolio.measures import RiskMeasure
from skfolio.preprocessing import prices_to_returns

from data_loader_new_portfolio import load_new_portfolio

# ═════════════════════════════════════════════
# KONFIGURATION
# ═════════════════════════════════════════════

TEST_SIZE   = 0.30      # 30% Testperiode, 70% Training
MAX_WEIGHT  = 0.40      # max. 40% je Asset
MIN_WEIGHT  = 0.00      # keine Leerverkäufe
RF = 0.02017            # risikofreier Zinssatz p.a.

# Zu verwendende Modelle: (Label, skfolio-Modell)
MODELS = {
    "Equally Weighted":   EqualWeighted(),
    "Min Variance":       MeanRisk(risk_measure=RiskMeasure.VARIANCE,
                                   min_weights=MIN_WEIGHT, max_weights=MAX_WEIGHT),
    "Max Sharpe":         MeanRisk(risk_measure=RiskMeasure.VARIANCE,
                                   min_weights=MIN_WEIGHT, max_weights=MAX_WEIGHT,
                                   objective_function="maximize_ratio"),
    "CVaR (95%)":         MeanRisk(risk_measure=RiskMeasure.CVAR,
                                   min_weights=MIN_WEIGHT, max_weights=MAX_WEIGHT),
    "CDaR (95%)":         MeanRisk(risk_measure=RiskMeasure.CDAR,
                                   min_weights=MIN_WEIGHT, max_weights=MAX_WEIGHT),
    "Semi-Varianz":       MeanRisk(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                   min_weights=MIN_WEIGHT, max_weights=MAX_WEIGHT),
    "Inv. Volatilität":   InverseVolatility(),
}

# Plot-Farben
COLORS = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF5722",
          "#9C27B0", "#FF9800", "#E91E63"]

BG, PANEL, GRID = "#0F1117", "#1A1D27", "#2A2D3A"
WHITE, MUTED    = "#E8E8E8", "#888888"


# ═════════════════════════════════════════════
# DATEN LADEN & VORBEREITEN
# ═════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lädt New Portfolio Daten, gibt (X_train, X_test, asset_meta) zurück."""
    data = load_new_portfolio(freq="monthly")

    # skfolio erwartet Preise → prices_to_returns intern
    # Wir übergeben monatliche Preise direkt
    prices  = data["prices_monthly"]
    returns = prices_to_returns(prices)     # skfolio-Format

    X_train, X_test = train_test_split(returns, test_size=TEST_SIZE, shuffle=False)

    print(f"\n{'─'*55}")
    print(f"Train: {X_train.index[0].date()} → {X_train.index[-1].date()}  ({len(X_train)} Monate)")
    print(f"Test:  {X_test.index[0].date()}  → {X_test.index[-1].date()}   ({len(X_test)} Monate)")
    print(f"{'─'*55}\n")

    return X_train, X_test, data["asset_meta"], data["asset_classes"]


# ═════════════════════════════════════════════
# OPTIMIERUNG
# ═════════════════════════════════════════════

def run_optimization(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> dict[str, Portfolio]:
    """
    Trainiert alle Modelle auf X_train, wertet auf X_test aus.
    Gibt dict { Modell-Name: Portfolio } zurück.
    """
    portfolios: dict[str, Portfolio] = {}

    for name, model in MODELS.items():
        try:
            model.fit(X_train)
            port = model.predict(X_test)
            portfolios[name] = port

            w_str = ", ".join(
                f"{col}: {w:.1%}"
                for col, w in zip(X_train.columns, model.weights_)
                if w > 0.001
            )
            print(f"  ✅  {name:22s}  → {w_str}")

        except Exception as e:
            print(f"  ❌  {name}: {e}")

    return portfolios


# ═════════════════════════════════════════════
# KENNZAHLEN VERGLEICH
# ═════════════════════════════════════════════

def collect_metrics(portfolios: dict[str, Portfolio]) -> pd.DataFrame:
    rows = []
    for name, port in portfolios.items():
        rows.append({
            "Modell":           name,
            "Ann. Rendite (%)": round(port.annualized_mean * 100 * 12, 2),  # monatlich → jährlich
            "Ann. Vola (%)":    round(port.annualized_standard_deviation * 100 * 12**0.5, 2),
            "Sharpe Ratio":     round(port.sharpe_ratio, 3),
            "Sortino Ratio":    round(port.sortino_ratio, 3),
            "Max Drawdown (%)": round(port.max_drawdown * 100, 2),
            "CVaR 95% (%)":     round(port.cvar * 100, 2),
            "Calmar Ratio":     round(port.calmar_ratio, 3),
        })
    return pd.DataFrame(rows).set_index("Modell")


# ═════════════════════════════════════════════
# GEWICHTSTABELLE
# ═════════════════════════════════════════════

def collect_weights(
    portfolios: dict[str, Portfolio],
    X_train:    pd.DataFrame,
    asset_meta: dict,
) -> pd.DataFrame:
    """Erstellt eine Gewichtstabelle [Asset × Modell]."""
    rows = {}
    for name, model in MODELS.items():
        if name in portfolios and hasattr(model, "weights_"):
            rows[name] = dict(zip(X_train.columns, model.weights_))

    wdf = pd.DataFrame(rows, index=X_train.columns).fillna(0)

    # Asset-Klasse als zusätzliche Spalte
    wdf.insert(0, "Asset-Klasse",
               [asset_meta.get(a, {}).get("asset_class", "?") for a in wdf.index])
    return wdf


# ═════════════════════════════════════════════
# VISUALISIERUNG
# ═════════════════════════════════════════════

def plot_results(
    portfolios:  dict[str, Portfolio],
    metrics:     pd.DataFrame,
    weights_df:  pd.DataFrame,
    X_train:     pd.DataFrame,
    X_test: pd.DataFrame,
    asset_meta:  dict,
    asset_classes: dict,
) -> None:

    fig = plt.figure(figsize=(20, 24), facecolor=BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30)

    def style(ax):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(axis="y", color=GRID, lw=0.5, ls="--")

    tkw = dict(color=WHITE, fontsize=12, fontweight="bold", pad=10)
    col_map = {n: COLORS[i % len(COLORS)] for i, n in enumerate(portfolios)}

    # ── 1. Kumulierte Renditen ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for name, port in portfolios.items():
        cum = (1 + port.returns_df).cumprod()
        lw  = 2.5 if name == "Equally Weighted" else 1.6
        ls  = "--" if name == "Equally Weighted" else "-"
        r = pd.Series(port.returns, index=X_test.index)
        cum = (1 + r).cumprod()
        ax1.plot(cum.index, cum.values, label=name, color=col_map[name], lw=lw, ls=ls)
    style(ax1)
    ax1.set_title("Kumulierte Renditen (Out-of-Sample)", **tkw)
    ax1.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE,
               facecolor=PANEL, edgecolor=GRID, ncol=4)
    ax1.set_ylabel("Wachstum (1 = Start)")

    # ── 2. Drawdown ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    for name, port in portfolios.items():
        lw = 2.5 if name == "Equally Weighted" else 1.4
        ls = "--" if name == "Equally Weighted" else "-"
        r = pd.Series(port.returns, index=X_test.index)
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        ax2.plot(dd.index, dd.values * 100, label=name, color=col_map[name], lw=lw, ls=ls)
    style(ax2)
    ax2.set_title("Drawdown (%)", **tkw)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE,
               facecolor=PANEL, edgecolor=GRID, ncol=4)

    # ── 3. Kennzahlen-Tabelle ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor(PANEL)
    ax3.axis("off")
    ax3.set_title("Performance-Metriken (Testperiode)", **tkw)

    tbl = ax3.table(
        cellText=metrics.values.tolist(),
        rowLabels=metrics.index.tolist(),
        colLabels=metrics.columns.tolist(),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        elif c == -1:
            cell.set_facecolor("#2A2D3A" if r % 2 == 0 else "#1F2230")
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#2A2D3A" if r % 2 == 0 else "#1F2230")
            cell.get_text().set_color("#CCCCCC")

    # ── 4. Gewichte-Heatmap ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    wplot = weights_df.drop(columns=["Asset-Klasse"], errors="ignore").astype(float)

    # Zeilenbeschriftung: Asset-Klasse + Name
    ylabels = [
        f"[{asset_meta.get(a,{}).get('asset_class','?')[:3]}] {a}"
        for a in wplot.index
    ]
    im = ax4.imshow(wplot.values, cmap="YlOrRd", aspect="auto",
                    vmin=0, vmax=wplot.values.max())
    ax4.set_xticks(range(len(wplot.columns)))
    ax4.set_xticklabels(wplot.columns, rotation=35, ha="right",
                        color=MUTED, fontsize=8)
    ax4.set_yticks(range(len(ylabels)))
    ax4.set_yticklabels(ylabels, color=MUTED, fontsize=7.5)
    for i in range(len(wplot.index)):
        for j in range(len(wplot.columns)):
            v = wplot.values[i, j]
            if v > 0.005:
                ax4.text(j, i, f"{v:.0%}", ha="center", va="center",
                         color="black" if v > 0.2 else "white", fontsize=7)
    ax4.set_facecolor(PANEL)
    ax4.set_title("Portfoliogewichte je Modell", **tkw)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # ── 5. Risiko-Rendite-Scatter ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    style(ax5)
    ax5.set_title("Risiko-Rendite-Diagramm", **tkw)
    for name, port in portfolios.items():
        x = port.annualized_standard_deviation * 100
        y = port.annualized_mean * 100
        m = "*" if name == "Equally Weighted" else "o"
        s = 200 if name == "Equally Weighted" else 110
        ax5.scatter(x, y, color=col_map[name], s=s, marker=m,
                    zorder=5, edgecolors="white", lw=0.5)
        ax5.annotate(name, (x, y), textcoords="offset points",
                     xytext=(6, 4), fontsize=8, color=col_map[name])
    ax5.set_xlabel("Annualisierte Volatilität (%)")
    ax5.set_ylabel("Annualisierte Rendite (%)")

    fig.suptitle("Portfolio-Optimierung — New Portfolio  |  skfolio",
                 color=WHITE, fontsize=16, fontweight="bold", y=0.998)

    plt.savefig("optimization_results.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("\n📊  Plot gespeichert: optimization_results.png")
    plt.show()


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print("  PORTFOLIO-OPTIMIERUNG — NEW PORTFOLIO")
    print("═" * 55)

    # 1. Daten
    X_train, X_test, asset_meta, asset_classes = load_data()

    # 2. Optimierung
    print("\nOptimiere Portfolios …")
    portfolios = run_optimization(X_train, X_test)

    # 3. Kennzahlen
    metrics = collect_metrics(portfolios)
    print("\n" + "─" * 55)
    print("PERFORMANCE-VERGLEICH (Out-of-Sample Testperiode)")
    print("─" * 55)
    print(metrics.to_string())

    # 4. Gewichte
    weights_df = collect_weights(portfolios, X_train, asset_meta)
    print("\n" + "─" * 55)
    print("OPTIMALE GEWICHTE (Trainingsperiode)")
    print("─" * 55)
    print(weights_df.to_string())

    # 5. Exports
    metrics.to_csv("optimization_metrics.csv")
    weights_df.to_csv("optimization_weights.csv")
    print("\n📄  Gespeichert: optimization_metrics.csv, optimization_weights.csv")

    # 6. Plot
    plot_results(portfolios, metrics, weights_df, X_train, X_test, asset_meta, asset_classes)
    print("\n✅  Fertig!")


if __name__ == "__main__":
    main()
