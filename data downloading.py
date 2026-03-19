import yfinance as yf
import pandas as pd
import numpy as np
import os

# =========================================================
# 1) Portfolio-Definition
# =========================================================

positions_eur = {
    "MBG.DE": 11_500_000,   # Mercedes-Benz Group AG
    "BMW.DE": 8_400_000,    # BMW AG
    "EXS1.DE": 9_800_000,   # iShares Core DAX UCITS ETF
    "SXR8.DE": 6_600_000,   # iShares Core S&P 500 UCITS ETF
    "EUNL.DE": 3_500_000,   # iShares Core MSCI World UCITS ETF
    "IHYG.L": 4_900_000,    # iShares € High Yield Corp Bond UCITS ETF
    "IBCI.DE": 2_900_000,   # iShares € Inflation Linked Govt Bond UCITS ETF
}

# Bund ist auf Yahoo i.d.R. nicht sauber verfügbar -> separat behandeln
bund_name = "GERMAN_BUND_2036"
bund_value_eur = 2_800_000

start_date = "2020-01-01"
end_date = "2025-09-10"

# =========================================================
# 2) Daten laden
# =========================================================

tickers = list(positions_eur.keys())

data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=True,
    group_by="column"
)

prices = data["Close"].copy()
prices = prices.dropna(axis=1, how="all")

print("Verfügbare Ticker:")
print(prices.columns.tolist())
print()

# =========================================================
# 3) Nur tatsächlich geladene Positionen verwenden
# =========================================================

available_positions = {
    ticker: value
    for ticker, value in positions_eur.items()
    if ticker in prices.columns
}

missing_positions = {
    ticker: value
    for ticker, value in positions_eur.items()
    if ticker not in prices.columns
}

print("Geladene Positionen:")
for k, v in available_positions.items():
    print(f"  {k}: EUR {v:,.0f}")

if missing_positions:
    print("\nNicht geladene Positionen:")
    for k, v in missing_positions.items():
        print(f"  {k}: EUR {v:,.0f}")

print(f"\nSeparat behandelter Bund:")
print(f"  {bund_name}: EUR {bund_value_eur:,.0f}")
print()

# =========================================================
# 4) Gewichte berechnen
#    a) Gewichte relativ zum Gesamtportfolio inkl. Bund
#    b) investierbare Yahoo-Gewichte ex Bund
# =========================================================

total_portfolio_value = sum(positions_eur.values()) + bund_value_eur
yahoo_portfolio_value = sum(available_positions.values())

weights_total = pd.Series(
    {ticker: value / total_portfolio_value for ticker, value in available_positions.items()},
    name="weight_total_portfolio"
)

weights_yahoo_only = pd.Series(
    {ticker: value / yahoo_portfolio_value for ticker, value in available_positions.items()},
    name="weight_yahoo_only"
)

bund_weight_total = bund_value_eur / total_portfolio_value
cash_like_weight = bund_weight_total  # wir behandeln den Bund in der Zeitreihe konservativ als 0%-Tagesrendite

print("Gewichte relativ zum Gesamtportfolio:")
print(weights_total.sort_values(ascending=False))
print(f"\n{bund_name}: {bund_weight_total:.4%}")

print("\nGewichte relativ zu den Yahoo-verfügbaren Assets:")
print(weights_yahoo_only.sort_values(ascending=False))
print()

# =========================================================
# 5) Renditen berechnen
# =========================================================

prices = prices[list(available_positions.keys())].copy()
returns = prices.pct_change().dropna()

# Sicherstellen, dass Gewichte in gleicher Reihenfolge wie Returns-Spalten stehen
weights_total = weights_total.reindex(returns.columns)
weights_yahoo_only = weights_yahoo_only.reindex(returns.columns)

# =========================================================
# 6) Portfolio-Renditen berechnen
# =========================================================

# A) Nur Yahoo-Assets, vollständig investiert
portfolio_return_yahoo_only = returns.mul(weights_yahoo_only, axis=1).sum(axis=1)
portfolio_index_yahoo_only = (1 + portfolio_return_yahoo_only).cumprod() * 100

# B) Gesamtportfolio inkl. Bund
#    Bund wird konservativ als "0%-Tagesrendite-Komponente" modelliert
portfolio_return_total = returns.mul(weights_total, axis=1).sum(axis=1)
portfolio_index_total = (1 + portfolio_return_total).cumprod() * 100

# =========================================================
# 7) Positionswerte im Zeitverlauf
# =========================================================

# Normierte Preisreihe je Asset ab 1
normalized_prices = prices / prices.iloc[0]

# Zeitreihe der Marktwerte je Yahoo-Asset
position_values = normalized_prices.mul(pd.Series(available_positions), axis=1)

# Bund als konstante Linie (vereinfachend)
position_values[bund_name] = bund_value_eur

# Gesamtportfolio-Marktwert
position_values["TOTAL_PORTFOLIO"] = position_values.sum(axis=1)

# =========================================================
# 8) Gewichtsentwicklung im Zeitverlauf
# =========================================================

weights_over_time = position_values.drop(columns=["TOTAL_PORTFOLIO"]).div(
    position_values["TOTAL_PORTFOLIO"], axis=0
)

# =========================================================
# 9) Kennzahlen
# =========================================================

summary = pd.DataFrame({
    "initial_amount_eur": pd.Series(available_positions),
    "weight_total_portfolio": weights_total,
    "weight_yahoo_only": weights_yahoo_only,
})

summary.loc[bund_name, "initial_amount_eur"] = bund_value_eur
summary.loc[bund_name, "weight_total_portfolio"] = bund_weight_total
summary.loc[bund_name, "weight_yahoo_only"] = np.nan

summary = summary.sort_values("initial_amount_eur", ascending=False)

print("Zusammenfassung:")
print(summary)
print()

# =========================================================
# 10) Speichern
# =========================================================

output_path = os.path.join("data", "portfolio")
os.makedirs(output_path, exist_ok=True)

prices.to_csv(os.path.join(output_path, "asset_prices.csv"))
returns.to_csv(os.path.join(output_path, "asset_returns.csv"))
summary.to_csv(os.path.join(output_path, "portfolio_weights_summary.csv"))
position_values.to_csv(os.path.join(output_path, "portfolio_position_values.csv"))
weights_over_time.to_csv(os.path.join(output_path, "portfolio_weights_over_time.csv"))

pd.DataFrame({
    "portfolio_return_yahoo_only": portfolio_return_yahoo_only,
    "portfolio_index_yahoo_only": portfolio_index_yahoo_only,
    "portfolio_return_total_incl_bund_proxy0": portfolio_return_total,
    "portfolio_index_total_incl_bund_proxy0": portfolio_index_total,
}).to_csv(os.path.join(output_path, "portfolio_timeseries.csv"))

print("Dateien gespeichert in:", output_path)
print(" - asset_prices.csv")
print(" - asset_returns.csv")
print(" - portfolio_weights_summary.csv")
print(" - portfolio_position_values.csv")
print(" - portfolio_weights_over_time.csv")
print(" - portfolio_timeseries.csv")