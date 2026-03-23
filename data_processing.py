import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/Old Portfolio")

def read_bloomberg_excel(filepath):
    meta = pd.read_excel(filepath, header=None, nrows=5, usecols=[0, 1])
    security = str(meta.iloc[0, 1]).strip()

    df = pd.read_excel(filepath, skiprows=6, usecols=[0, 1, 2], parse_dates=[0])
    df.columns = ["Date", "PX_MID", "PX_BID"]
    df = (df
          .dropna(subset=["Date"])
          .assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.normalize())
          .sort_values("Date")
          .reset_index(drop=True))
    return security, df

# Alle Excel-Dateien auf einmal einlesen
all_data = {}
for file in DATA_DIR.glob("*.xlsx"):
    security, df = read_bloomberg_excel(file)
    all_data[security] = df
    print(f"✅ {security}: {len(df)} Datenpunkte")
# Einzeln abrufen z.B.:
# df_bmw = all_data["BMW GY Equity"]df mit Spalten: Date | PX_MID | PX_BID