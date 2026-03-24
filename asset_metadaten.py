"""
asset_metadata.py
=================
Metadaten für alle Assets aus dem New Portfolio.
Enthält: Klasse, Sektor (15 Hauptsektoren), Region, ESG-Score (0–100, höher = besser)

Regionen:
- Einzeltitel: Länder (z. B. USA, Deutschland, Österreich)
- ETFs/Fonds/Indizes: geografische Exponierung (z. B. Global, Europa, Emerging Markets)

Sektoren (15 Hauptsektoren):
1. Technologie
2. Halbleiter
3. Industrie
4. Automobil
5. Energie
6. Versorger
7. Gesundheit
8. Finanzen
9. Konsum defensiv
10. Konsum zyklisch
11. Rohstoffe
12. Diversifiziert Aktien
13. Anleihen
14. Edelmetalle
15. Krypto / Alternativ

Verwendung:
    from asset_metadata import ASSET_META, get_meta, get_df

    meta = get_meta("AAPL US Equity")
    df   = get_df()
"""

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# ASSET_META
# Felder: klasse, sektor, region, esg_score (0–100)
# ─────────────────────────────────────────────────────────────────────────────
ASSET_META: dict[str, dict] = {

    # ── STOCKS ──────────────────────────────────────────────────────────────

    "XOM US Equity": {
        "klasse": "Stocks", "sektor": "Energie", "region": "USA",
        "esg_score": 5.45,
        "esg_note": "Größter US-Ölkonzern, hohe CO2-Emissionen, schwaches ESG-Profil",
    },
    "VOW3 GY Equity": {
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 5.20,
        "esg_note": "Dieselgate-Nachwirkungen, aber starke E-Strategie (ID-Serie)",
    },
    "VOE AV Equity": {
        "klasse": "Stocks", "sektor": "Rohstoffe", "region": "Österreich",
        "esg_score": 4.95,
        "esg_note": "Voestalpine – Dekarbonisierungsprogramm (greentec steel), mittleres ESG",
    },
    "VER AV Equity": {
        "klasse": "Stocks", "sektor": "Versorger", "region": "Österreich",
        "esg_score": 5.34,
        "esg_note": "Verbund – fast 100% Wasserkraft, eines der grünsten Energieunternehmen Europas",
    },
    "UNP US Equity": {
        "klasse": "Stocks", "sektor": "Industrie", "region": "USA",
        "esg_score": 4.52,
        "esg_note": "Union Pacific – Schiene effizienter als LKW, solide Governance",
    },
    "UNH US Equity": {
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "USA",
        "esg_score": 6.31,
        "esg_note": "UnitedHealth – starke Governance, aber Kritik an Versicherungsverweigerungen",
    },
    "ULVR LN Equity": {
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "Großbritannien",
        "esg_score": 5.28,
        "esg_note": "Unilever – Vorreiter bei Nachhaltigkeitszielen, MSCI AA",
    },
    "TSM US Equity": {
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "Taiwan",
        "esg_score": 4.79,
        "esg_note": "TSMC – hoher Wasserverbrauch, aber starke Governance und 100%-RE-Ziel",
    },
    "SIE GY Equity": {
        "klasse": "Stocks", "sektor": "Industrie", "region": "Deutschland",
        "esg_score": 6.59,
        "esg_note": "Siemens – DEGREE-Nachhaltigkeitsprogramm, MSCI A",
    },
    "SHEL LN Equity": {
        "klasse": "Stocks", "sektor": "Energie", "region": "Großbritannien",
        "esg_score": 6.74,
        "esg_note": "Shell – Öl & Gas, Net-Zero-Ziel 2050, aber weiterhin fossile Expansion",
    },
    "ROG SW Equity": {
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "Schweiz",
        "esg_score": 4.53,
        "esg_note": "Roche – starkes ESG-Programm, Zugang zu Medikamenten, MSCI AA",
    },
    "RIO LN Equity": {
        "klasse": "Stocks", "sektor": "Rohstoffe", "region": "Großbritannien",
        "esg_score": 6.92,
        "esg_note": "Rio Tinto – Juukan-Gorge-Skandal, aber Klimaziele und Governance-Reformen",
    },
    "RBI AV Equity": {
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Österreich",
        "esg_score": 5.83,
        "esg_note": "Raiffeisen Bank International – Russland-Exposure belastet ESG-Rating",
    },
    "PG US Equity": {
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "USA",
        "esg_score": 4.96,
        "esg_note": "Procter & Gamble – Ambition 2030, MSCI AA, starke Supply-Chain-Standards",
    },
    "PEP US Equity": {
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "PepsiCo – pep+ Nachhaltigkeitsplan, MSCI A",
    },
    "OMV AV Equity": {
        "klasse": "Stocks", "sektor": "Energie", "region": "Österreich",
        "esg_score": 6.14,
        "esg_note": "OMV – Transformation zu Chemicals, aber weiterhin fossile Kernaktivitäten",
    },
    "NVDA US Equity": {
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "USA",
        "esg_score": 6.94,
        "esg_note": "Nvidia – hoher Energieverbrauch durch KI-Chips, gute Governance",
    },
    "NOVN SW Equity": {
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "Schweiz",
        "esg_score": 5.11,
        "esg_note": "Novartis – Patient-Access-Programme, Carbon-neutral seit 2020",
    },
    "NKE US Equity": {
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "USA",
        "esg_score": 4.99,
        "esg_note": "Nike – Move to Zero, aber Supply-Chain-Arbeitsrechtsprobleme",
    },
    "NESN SW Equity": {
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "Schweiz",
        "esg_score": 5.44,
        "esg_note": "Nestlé – Net-Zero-Commitment, Wasserrechte-Kritik, MSCI A",
    },
    "NEE US Equity": {
        "klasse": "Stocks", "sektor": "Versorger", "region": "USA",
        "esg_score": 6.52,
        "esg_note": "NextEra Energy – weltgrößter Erzeuger Wind-/Solarenergie, MSCI AA",
    },
    "MSFT US Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.18,
        "esg_note": "Microsoft – carbon negative by 2030, MSCI AAA",
    },
    "MCD US Equity": {
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "USA",
        "esg_score": 4.25,
        "esg_note": "McDonald's – Scale for Good, aber Fleischproduktion & Verpackung problematisch",
    },
    "MC FP Equity": {
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "Frankreich",
        "esg_score": 4.86,
        "esg_note": "LVMH – LIFE 360 Programm, aber Luxuskonsum-Nachhaltigkeitsfrage",
    },
    "MBG GY Equity": {
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 6.24,
        "esg_note": "Mercedes-Benz – EV-Umstieg, aber weiterhin starke Verbrenner-Abhängigkeit",
    },
    "KO US Equity": {
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "USA",
        "esg_score": 5.14,
        "esg_note": "Coca-Cola – World Without Waste, Wasser-Stewardship, MSCI A",
    },
    "JPM US Equity": {
        "klasse": "Stocks", "sektor": "Finanzen", "region": "USA",
        "esg_score": 4.65,
        "esg_note": "JPMorgan – größter US-Kreditgeber für fossile Energie, Paris-Alignment fraglich",
    },
    "JNJ US Equity": {
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "USA",
        "esg_score": 5.84,
        "esg_note": "J&J – ESG-Leader im Pharmabereich, Talc-Rechtsstreit belastet",
    },
    "ISRG US Equity": {
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "USA",
        "esg_score": 3.64,
        "esg_note": "Intuitive Surgical – Medizintechnik, geringer direkter CO2-Fußabdruck",
    },
    "INFY US Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "Indien",
        "esg_score": 6.72,
        "esg_note": "Infosys – carbon neutral seit 2020, MSCI AA",
    },
    "IBN US Equity": {
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Indien",
        "esg_score": 3.94,
        "esg_note": "ICICI Bank – wachsende ESG-Rahmenwerke, Emerging Market Standard",
    },
    "HSBA LN Equity": {
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Großbritannien",
        "esg_score": 4.26,
        "esg_note": "HSBC – Greenwashing-Vorwürfe 2022, aber Net-Zero-Commitment",
    },
    "HON US Equity": {
        "klasse": "Stocks", "sektor": "Industrie", "region": "USA",
        "esg_score": 5.88,
        "esg_note": "Honeywell – ESG-Technologien (Building-Automation), MSCI A",
    },
    "GS US Equity": {
        "klasse": "Stocks", "sektor": "Finanzen", "region": "USA",
        "esg_score": 5.15,
        "esg_note": "Goldman Sachs – Sustainable Finance 750 Mrd. Commitment, aber Governance-Fragen",
    },
    "EBS AV Equity": {
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Österreich",
        "esg_score": 5.59,
        "esg_note": "Erste Group Bank – solides ESG-Profil im mitteleuropäischen Bankensektor",
    },
    "DUK US Equity": {
        "klasse": "Stocks", "sektor": "Versorger", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Duke Energy – Transition von Kohle zu erneuerbaren Energien, Net-Zero 2050",
    },
    "CVX US Equity": {
        "klasse": "Stocks", "sektor": "Energie", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Chevron – Öl & Gas, schwaches Klimabekenntnis vs. Peers",
    },
    "CRM US Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.52,
        "esg_note": "Salesforce – Net-Zero-Unternehmen, 1-1-1-Philanthropie-Modell, MSCI AAA",
    },
    "CON GY Equity": {
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 5.20,
        "esg_note": "Continental – Transformation zu Elektronik/Software, mittleres ESG",
    },
    "BMW GY Equity": {
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 5.83,
        "esg_note": "BMW – MSCI AA, starke EV-Roadmap (Neue Klasse), Kreislaufwirtschaft",
    },
    "BHP AU Equity": {
        "klasse": "Stocks", "sektor": "Rohstoffe", "region": "Australien",
        "esg_score": 7.82,
        "esg_note": "BHP – Samarco-Katastrophe Altlast, aber Klimaziele und Dekarbonisierung",
    },
    "BABA US Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "China",
        "esg_score": 5.17,
        "esg_note": "Alibaba – Regulatorisches Risiko China, ESG-Transparenz eingeschränkt",
    },
    "ASML NA Equity": {
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "Niederlande",
        "esg_score": 6.78,
        "esg_note": "ASML – enabling chips für grüne Technologien, starke ESG-Governance",
    },
    "AMZN US Equity": {
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "USA",
        "esg_score": 4.32,
        "esg_note": "Amazon – Climate Pledge, aber Arbeitnehmerrechte und Logistik-CO2 kritisiert",
    },
    "ADBE US Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.96,
        "esg_note": "Adobe – carbon neutral, MSCI AA, D&I-Leader",
    },
    "AAPL US Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "Apple – 100% RE, carbon neutral bis 2030, MSCI AAA",
    },
    "7203 JT Equity": {
        "klasse": "Stocks", "sektor": "Automobil", "region": "Japan",
        "esg_score": 6.19,
        "esg_note": "Toyota – Hybrid-Pionier, aber langsamere BEV-Transition vs. Peers",
    },
    "700 HK Equity": {
        "klasse": "Stocks", "sektor": "Technologie", "region": "China",
        "esg_score": 4.79,
        "esg_note": "Tencent – Gaming-Regulierung, ESG-Transparenz eingeschränkt, MSCI BBB",
    },
    "005930 KS Equity": {
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "Südkorea",
        "esg_score": 4.94,
        "esg_note": "Samsung Electronics – RE100-Mitglied, aber Governance-Konzern-Risiken",
    },

    # ── ETF ─────────────────────────────────────────────────────────────────

    "XMEU GY Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Europa",
        "esg_score": None,
        "esg_note": "Xtrackers MSCI Europe – breite Europa-Diversifikation, keine explizite ESG-Filterung",
    },
    "VWRL LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "Vanguard FTSE All-World – vollständige Marktabdeckung, kein ESG-Filter",
    },
    "VUSA LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "USA",
        "esg_score": None,
        "esg_note": "Vanguard S&P 500 – kein ESG-Filter, schwergewichtet in Öl/Gas",
    },
    "VFEM LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Emerging Markets",
        "esg_score": None,
        "esg_note": "Vanguard FTSE EM – inkl. China/Staatsunternehmen, niedrigere ESG-Standards",
    },
    "RENW IM Equity": {
        "klasse": "ETF", "sektor": "Energie", "region": "Global",
        "esg_score": None,
        "esg_note": "iShares Global Clean Energy – dezidierter ESG/Cleantech-ETF, sehr hoch",
    },
    "RBOT LN Equity": {
        "klasse": "ETF", "sektor": "Technologie", "region": "Global",
        "esg_score": None,
        "esg_note": "iShares Automation & Robotics – Zukunftstechnologien, mittleres ESG",
    },
    "IWDA LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Developed Markets",
        "esg_score": None,
        "esg_note": "iShares MSCI World – kein expliziter ESG-Filter, breite Diversifikation",
    },
    "IWVL LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Developed Markets",
        "esg_score": None,
        "esg_note": "iShares Edge MSCI World Value – Value-Tilt oft mit höherem Energie-Anteil",
    },
    "IUSN GY Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "iShares MSCI World Small Cap – geringere ESG-Transparenz bei kleineren Firmen",
    },
    "ISAC LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "iShares MSCI ACWI – All-Country, inkl. EM, kein ESG-Filter",
    },
    "IMEU NA Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Europa",
        "esg_score": None,
        "esg_note": "iShares Core MSCI Europe – breite Europa-Abdeckung, kein expliziter ESG-Filter",
    },
    "EIMI LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Emerging Markets",
        "esg_score": None,
        "esg_note": "iShares Core MSCI EM IMI – breite EM-Abdeckung, niedrigere ESG-Standards",
    },
    "CSPX LN Equity": {
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "USA",
        "esg_score": None,
        "esg_note": "iShares Core S&P 500 – kein ESG-Filter, US-Large-Cap-Abdeckung",
    },

    # ── BOND ETF ─────────────────────────────────────────────────────────────

    "XGLE GY Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "Xtrackers EUR Govt Bond – EU-Staatsanleihen, höhere ESG-Standards vs. EM",
    },
    "XBLC GY Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "Xtrackers EUR Corporate Bond – Investment Grade, kein ESG-Filter",
    },
    "VECP GY Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "Vanguard EUR Corporate Bond – IG-Diversifikation, kein ESG-Ausschluss",
    },
    "VDCP LN Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "USA",
        "esg_score": None,
        "esg_note": "Vanguard USD Corporate Bond – US-IG, kein ESG-Filter",
    },
    "MTD FP Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Global",
        "esg_score": None,
        "esg_note": "Amundi Global Govt Bond – Staatsanleihen-Diversifikation, mittleres ESG",
    },
    "OM3F GY Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "iShares EUR Corp Bond – IG-EUR-Unternehmensanleihen",
    },
    "IEAC LN Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "iShares EUR Corp Bond – Investment Grade, breite Sektordiversifikation",
    },
    "EUNH GY Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Eurozone",
        "esg_score": None,
        "esg_note": "iShares Core EUR Govt Bond – Eurozone-Staatsanleihen, solides ESG",
    },
    "CSBGE3 IM Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "Credit Suisse/UBS Green Bond – explizit auf Green Bonds fokussiert",
    },
    "CORP LN Equity": {
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "USA",
        "esg_score": None,
        "esg_note": "iShares USD Corp Bond – US-Investment-Grade, kein ESG-Filter",
    },

    # ── COMMODITIES ──────────────────────────────────────────────────────────

    "PHPT LN Equity": {
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Platin ETC – Bergbau-Lieferkette in SA, aber Industrie-/H2-Nachfrage",
    },
    "PHAU LN Equity": {
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Gold ETC – Bergbau umweltintensiv, aber Wertspeicher ohne Unternehmensrisiko",
    },
    "PHAG IM Equity": {
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Silber ETC – ähnlich Gold, Bergbau-ESG-Risiken",
    },
    "IGLN LN Equity": {
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "iShares Physical Gold – physisch gedecktes Gold ETC",
    },

    # ── FUND ─────────────────────────────────────────────────────────────────

    "VNGA60 IM Equity": {
        "klasse": "Fund", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "Vanguard LifeStrategy 60% Equity – kein ESG-Filter, breite Diversifikation",
    },

    # ── INDEX ────────────────────────────────────────────────────────────────

    "catholic index": {
        "klasse": "Index", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "Katholischer Ethik-Index – explizite Ausschlüsse, hohe ESG-Standards",
    },

    # ── CRYPTO ───────────────────────────────────────────────────────────────

    "bitcoin": {
        "klasse": "Crypto", "sektor": "Krypto / Alternativ", "region": "Global",
        "esg_score": None,
        "esg_note": "Bitcoin – Proof-of-Work, hoher Energieverbrauch, kein ESG-Rahmen",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

def get_meta(asset: str) -> dict:
    """Gibt Metadaten für ein Asset zurück. Leeres Dict falls nicht gefunden."""
    return ASSET_META.get(asset, {})


def get_df() -> pd.DataFrame:
    """Gibt alle Metadaten als pandas DataFrame zurück."""
    rows = []
    for asset, meta in ASSET_META.items():
        rows.append({
            "asset":     asset,
            "klasse":    meta.get("klasse", "–"),
            "sektor":    meta.get("sektor", "–"),
            "region":    meta.get("region", "–"),
            "esg_score": meta.get("esg_score", None),
            "esg_note":  meta.get("esg_note", ""),
        })
    return pd.DataFrame(rows).set_index("asset")


def esg_label(score: int) -> str:
    """Gibt ein ESG-Label basierend auf dem Score zurück."""
    if score is None:
        return "n/a"
    if score >= 7.5:
        return "AAA"
    elif score >= 6.5:
        return "AA"
    elif score >= 5.5:
        return "A"
    elif score >= 4.5:
        return "BBB"
    elif score >= 3.5:
        return "BB"
    elif score >= 2.5:
        return "B"
    else:
        return "CCC"


# ─────────────────────────────────────────────────────────────────────────────
# DIREKTER AUFRUF – Übersicht
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_df()
    df["esg_label"] = df["esg_score"].apply(esg_label)

    print(f"\n{'─'*90}")
    print(f"  ASSET METADATEN  –  {len(df)} Assets")
    print(f"{'─'*90}")
    print(df[["klasse", "sektor", "region", "esg_score", "esg_label"]].to_string())

    print(f"\n{'─'*40}")
    print("  ESG-Verteilung")
    print(f"{'─'*40}")
    bins   = [0, 35, 45, 55, 65, 75, 101]
    labels = ["CCC/B (<35)", "BB (35-45)", "BBB (45-55)", "A (55-65)", "AA (65-75)", "AAA (75+)"]
    df["esg_band"] = pd.cut(df["esg_score"], bins=bins, labels=labels, right=False)
    print(df["esg_band"].value_counts().sort_index().to_string())

    print(f"\n{'─'*40}")
    print("  ESG Top 5")
    print(f"{'─'*40}")
    print(df.nlargest(5, "esg_score")[["sektor", "region", "esg_score"]].to_string())

    print(f"\n{'─'*40}")
    print("  ESG Bottom 5")
    print(f"{'─'*40}")
    print(df.nsmallest(5, "esg_score")[["sektor", "region", "esg_score"]].to_string())

    print(f"\n{'─'*40}")
    print("  Verteilung nach Sektor")
    print(f"{'─'*40}")
    print(df["sektor"].value_counts().sort_values(ascending=False).to_string())

    print(f"\n{'─'*40}")
    print("  Verteilung nach Region")
    print(f"{'─'*40}")
    print(df["region"].value_counts().sort_values(ascending=False).to_string())