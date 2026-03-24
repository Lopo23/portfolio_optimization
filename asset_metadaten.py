"""
asset_metadata.py
=================
Metadaten für alle Assets aus dem New Portfolio.

Felder je Asset:
    name       – Vollständiger Name / Produktbezeichnung
    isin       – ISIN (International Securities Identification Number)
    klasse     – Asset-Klasse (Stocks, ETF, Bond ETF, Commodities, Fund, Index, Crypto)
    sektor     – Einer der 15 Hauptsektoren (siehe unten)
    region     – Land (Einzeltitel) oder geografische Exponierung (ETFs/Fonds)
    esg_score  – Sustainalytics-ähnlicher Score 0–10 (höher = riskanter, None = nicht verfügbar)
    esg_note   – Kurze ESG-Begründung

Sektoren (15):
    Technologie · Halbleiter · Industrie · Automobil · Energie · Versorger
    Gesundheit · Finanzen · Konsum defensiv · Konsum zyklisch · Rohstoffe
    Diversifiziert Aktien · Anleihen · Edelmetalle · Krypto / Alternativ

Verwendung:
    from asset_metadata import ASSET_META, get_meta, get_df, esg_label

    meta = get_meta("AAPL US Equity")
    df   = get_df()   # → DataFrame mit allen Feldern
"""

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# ASSET_META
# ─────────────────────────────────────────────────────────────────────────────
ASSET_META: dict[str, dict] = {

    # ── STOCKS ──────────────────────────────────────────────────────────────

    "XOM US Equity": {
        "name": "Exxon Mobil Corporation",
        "isin": "US30231G1022",
        "klasse": "Stocks", "sektor": "Energie", "region": "USA",
        "esg_score": 5.45,
        "esg_note": "Größter US-Ölkonzern, hohe CO2-Emissionen, schwaches ESG-Profil",
    },
    "VOW3 GY Equity": {
        "name": "Volkswagen AG (Vorzugsaktie)",
        "isin": "DE0007664039",
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 5.20,
        "esg_note": "Dieselgate-Nachwirkungen, aber starke E-Strategie (ID-Serie)",
    },
    "VOE AV Equity": {
        "name": "voestalpine AG",
        "isin": "AT0000937503",
        "klasse": "Stocks", "sektor": "Rohstoffe", "region": "Österreich",
        "esg_score": 4.95,
        "esg_note": "Dekarbonisierungsprogramm greentec steel, mittleres ESG",
    },
    "VER AV Equity": {
        "name": "Verbund AG",
        "isin": "AT0000746409",
        "klasse": "Stocks", "sektor": "Versorger", "region": "Österreich",
        "esg_score": 5.34,
        "esg_note": "Fast 100% Wasserkraft, eines der grünsten Energieunternehmen Europas",
    },
    "UNP US Equity": {
        "name": "Union Pacific Corporation",
        "isin": "US9078181081",
        "klasse": "Stocks", "sektor": "Industrie", "region": "USA",
        "esg_score": 4.52,
        "esg_note": "Schiene effizienter als LKW, solide Governance",
    },
    "UNH US Equity": {
        "name": "UnitedHealth Group Incorporated",
        "isin": "US91324P1021",
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "USA",
        "esg_score": 6.31,
        "esg_note": "Starke Governance, aber Kritik an Versicherungsverweigerungen",
    },
    "ULVR LN Equity": {
        "name": "Unilever PLC",
        "isin": "GB00B10RZP78",
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "Großbritannien",
        "esg_score": 5.28,
        "esg_note": "Vorreiter bei Nachhaltigkeitszielen, MSCI AA",
    },
    "TSM US Equity": {
        "name": "Taiwan Semiconductor Manufacturing Co. Ltd. (ADR)",
        "isin": "US8740391003",
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "Taiwan",
        "esg_score": 4.79,
        "esg_note": "Hoher Wasserverbrauch, aber starke Governance und 100%-RE-Ziel",
    },
    "SIE GY Equity": {
        "name": "Siemens AG",
        "isin": "DE0007236101",
        "klasse": "Stocks", "sektor": "Industrie", "region": "Deutschland",
        "esg_score": 6.59,
        "esg_note": "DEGREE-Nachhaltigkeitsprogramm, MSCI A",
    },
    "SHEL LN Equity": {
        "name": "Shell PLC",
        "isin": "GB00BP6MXD84",
        "klasse": "Stocks", "sektor": "Energie", "region": "Großbritannien",
        "esg_score": 6.74,
        "esg_note": "Net-Zero-Ziel 2050, aber weiterhin fossile Expansion",
    },
    "ROG SW Equity": {
        "name": "Roche Holding AG",
        "isin": "CH0012221716",
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "Schweiz",
        "esg_score": 4.53,
        "esg_note": "Starkes ESG-Programm, Zugang zu Medikamenten, MSCI AA",
    },
    "RIO LN Equity": {
        "name": "Rio Tinto PLC",
        "isin": "GB0007188757",
        "klasse": "Stocks", "sektor": "Rohstoffe", "region": "Großbritannien",
        "esg_score": 6.92,
        "esg_note": "Juukan-Gorge-Skandal, aber Klimaziele und Governance-Reformen",
    },
    "RBI AV Equity": {
        "name": "Raiffeisen Bank International AG",
        "isin": "AT0000606306",
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Österreich",
        "esg_score": 5.83,
        "esg_note": "Russland-Exposure belastet ESG-Rating",
    },
    "PG US Equity": {
        "name": "Procter & Gamble Company",
        "isin": "US7427181091",
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "USA",
        "esg_score": 4.96,
        "esg_note": "Ambition 2030, MSCI AA, starke Supply-Chain-Standards",
    },
    "PEP US Equity": {
        "name": "PepsiCo Inc.",
        "isin": "US7134481081",
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "pep+ Nachhaltigkeitsplan, MSCI A",
    },
    "OMV AV Equity": {
        "name": "OMV AG",
        "isin": "AT0000743059",
        "klasse": "Stocks", "sektor": "Energie", "region": "Österreich",
        "esg_score": 6.14,
        "esg_note": "Transformation zu Chemicals, aber weiterhin fossile Kernaktivitäten",
    },
    "NVDA US Equity": {
        "name": "NVIDIA Corporation",
        "isin": "US67066G1040",
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "USA",
        "esg_score": 6.94,
        "esg_note": "Hoher Energieverbrauch durch KI-Chips, gute Governance",
    },
    "NOVN SW Equity": {
        "name": "Novartis AG",
        "isin": "CH0012221802",
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "Schweiz",
        "esg_score": 5.11,
        "esg_note": "Patient-Access-Programme, Carbon-neutral seit 2020",
    },
    "NKE US Equity": {
        "name": "NIKE Inc.",
        "isin": "US6541061031",
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "USA",
        "esg_score": 4.99,
        "esg_note": "Move to Zero, aber Supply-Chain-Arbeitsrechtsprobleme",
    },
    "NESN SW Equity": {
        "name": "Nestlé S.A.",
        "isin": "CH0038863350",
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "Schweiz",
        "esg_score": 5.44,
        "esg_note": "Net-Zero-Commitment, Wasserrechte-Kritik, MSCI A",
    },
    "NEE US Equity": {
        "name": "NextEra Energy Inc.",
        "isin": "US65339F1012",
        "klasse": "Stocks", "sektor": "Versorger", "region": "USA",
        "esg_score": 6.52,
        "esg_note": "Weltgrößter Erzeuger Wind-/Solarenergie, MSCI AA",
    },
    "MSFT US Equity": {
        "name": "Microsoft Corporation",
        "isin": "US5949181045",
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.18,
        "esg_note": "Carbon negative by 2030, MSCI AAA",
    },
    "MCD US Equity": {
        "name": "McDonald's Corporation",
        "isin": "US5801351017",
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "USA",
        "esg_score": 4.25,
        "esg_note": "Scale for Good, aber Fleischproduktion & Verpackung problematisch",
    },
    "MC FP Equity": {
        "name": "LVMH Moët Hennessy Louis Vuitton SE",
        "isin": "FR0000121014",
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "Frankreich",
        "esg_score": 4.86,
        "esg_note": "LIFE 360 Programm, aber Luxuskonsum-Nachhaltigkeitsfrage",
    },
    "MBG GY Equity": {
        "name": "Mercedes-Benz Group AG",
        "isin": "DE0007100000",
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 6.24,
        "esg_note": "EV-Umstieg, aber weiterhin starke Verbrenner-Abhängigkeit",
    },
    "KO US Equity": {
        "name": "The Coca-Cola Company",
        "isin": "US1912161007",
        "klasse": "Stocks", "sektor": "Konsum defensiv", "region": "USA",
        "esg_score": 5.14,
        "esg_note": "World Without Waste, Wasser-Stewardship, MSCI A",
    },
    "JPM US Equity": {
        "name": "JPMorgan Chase & Co.",
        "isin": "US46625H1005",
        "klasse": "Stocks", "sektor": "Finanzen", "region": "USA",
        "esg_score": 4.65,
        "esg_note": "Größter US-Kreditgeber für fossile Energie, Paris-Alignment fraglich",
    },
    "JNJ US Equity": {
        "name": "Johnson & Johnson",
        "isin": "US4781601046",
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "USA",
        "esg_score": 5.84,
        "esg_note": "ESG-Leader im Pharmabereich, Talc-Rechtsstreit belastet",
    },
    "ISRG US Equity": {
        "name": "Intuitive Surgical Inc.",
        "isin": "US46120E6023",
        "klasse": "Stocks", "sektor": "Gesundheit", "region": "USA",
        "esg_score": 3.64,
        "esg_note": "Medizintechnik, geringer direkter CO2-Fußabdruck",
    },
    "INFY US Equity": {
        "name": "Infosys Limited (ADR)",
        "isin": "US4567881085",
        "klasse": "Stocks", "sektor": "Technologie", "region": "Indien",
        "esg_score": 6.72,
        "esg_note": "Carbon neutral seit 2020, MSCI AA",
    },
    "IBN US Equity": {
        "name": "ICICI Bank Limited (ADR)",
        "isin": "US45104G1040",
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Indien",
        "esg_score": 3.94,
        "esg_note": "Wachsende ESG-Rahmenwerke, Emerging Market Standard",
    },
    "HSBA LN Equity": {
        "name": "HSBC Holdings PLC",
        "isin": "GB0005405286",
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Großbritannien",
        "esg_score": 4.26,
        "esg_note": "Greenwashing-Vorwürfe 2022, aber Net-Zero-Commitment",
    },
    "HON US Equity": {
        "name": "Honeywell International Inc.",
        "isin": "US4385161066",
        "klasse": "Stocks", "sektor": "Industrie", "region": "USA",
        "esg_score": 5.88,
        "esg_note": "ESG-Technologien (Building-Automation), MSCI A",
    },
    "GS US Equity": {
        "name": "The Goldman Sachs Group Inc.",
        "isin": "US38141G1040",
        "klasse": "Stocks", "sektor": "Finanzen", "region": "USA",
        "esg_score": 5.15,
        "esg_note": "Sustainable Finance 750 Mrd. Commitment, aber Governance-Fragen",
    },
    "EBS AV Equity": {
        "name": "Erste Group Bank AG",
        "isin": "AT0000652011",
        "klasse": "Stocks", "sektor": "Finanzen", "region": "Österreich",
        "esg_score": 5.59,
        "esg_note": "Solides ESG-Profil im mitteleuropäischen Bankensektor",
    },
    "DUK US Equity": {
        "name": "Duke Energy Corporation",
        "isin": "US26441C2044",
        "klasse": "Stocks", "sektor": "Versorger", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Transition von Kohle zu erneuerbaren Energien, Net-Zero 2050",
    },
    "CVX US Equity": {
        "name": "Chevron Corporation",
        "isin": "US1667641005",
        "klasse": "Stocks", "sektor": "Energie", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Öl & Gas, schwaches Klimabekenntnis vs. Peers",
    },
    "CRM US Equity": {
        "name": "Salesforce Inc.",
        "isin": "US79466L3024",
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.52,
        "esg_note": "Net-Zero-Unternehmen, 1-1-1-Philanthropie-Modell, MSCI AAA",
    },
    "CON GY Equity": {
        "name": "Continental AG",
        "isin": "DE0005439004",
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 5.20,
        "esg_note": "Transformation zu Elektronik/Software, mittleres ESG",
    },
    "BMW GY Equity": {
        "name": "Bayerische Motoren Werke AG",
        "isin": "DE0005190003",
        "klasse": "Stocks", "sektor": "Automobil", "region": "Deutschland",
        "esg_score": 5.83,
        "esg_note": "MSCI AA, starke EV-Roadmap (Neue Klasse), Kreislaufwirtschaft",
    },
    "BHP AU Equity": {
        "name": "BHP Group Limited",
        "isin": "AU000000BHP4",
        "klasse": "Stocks", "sektor": "Rohstoffe", "region": "Australien",
        "esg_score": 7.82,
        "esg_note": "Samarco-Katastrophe Altlast, aber Klimaziele und Dekarbonisierung",
    },
    "BABA US Equity": {
        "name": "Alibaba Group Holding Limited (ADR)",
        "isin": "US01609W1027",
        "klasse": "Stocks", "sektor": "Technologie", "region": "China",
        "esg_score": 5.17,
        "esg_note": "Regulatorisches Risiko China, ESG-Transparenz eingeschränkt",
    },
    "ASML NA Equity": {
        "name": "ASML Holding N.V.",
        "isin": "NL0010273215",
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "Niederlande",
        "esg_score": 6.78,
        "esg_note": "Enabling chips für grüne Technologien, starke ESG-Governance",
    },
    "AMZN US Equity": {
        "name": "Amazon.com Inc.",
        "isin": "US0231351067",
        "klasse": "Stocks", "sektor": "Konsum zyklisch", "region": "USA",
        "esg_score": 4.32,
        "esg_note": "Climate Pledge, aber Arbeitnehmerrechte und Logistik-CO2 kritisiert",
    },
    "ADBE US Equity": {
        "name": "Adobe Inc.",
        "isin": "US00724F1012",
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.96,
        "esg_note": "Carbon neutral, MSCI AA, D&I-Leader",
    },
    "AAPL US Equity": {
        "name": "Apple Inc.",
        "isin": "US0378331005",
        "klasse": "Stocks", "sektor": "Technologie", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "100% RE, carbon neutral bis 2030, MSCI AAA",
    },
    "7203 JT Equity": {
        "name": "Toyota Motor Corporation",
        "isin": "JP3633400001",
        "klasse": "Stocks", "sektor": "Automobil", "region": "Japan",
        "esg_score": 6.19,
        "esg_note": "Hybrid-Pionier, aber langsamere BEV-Transition vs. Peers",
    },
    "700 HK Equity": {
        "name": "Tencent Holdings Limited",
        "isin": "KYG875721634",
        "klasse": "Stocks", "sektor": "Technologie", "region": "China",
        "esg_score": 4.79,
        "esg_note": "Gaming-Regulierung, ESG-Transparenz eingeschränkt, MSCI BBB",
    },
    "005930 KS Equity": {
        "name": "Samsung Electronics Co., Ltd.",
        "isin": "KR7005930003",
        "klasse": "Stocks", "sektor": "Halbleiter", "region": "Südkorea",
        "esg_score": 4.94,
        "esg_note": "RE100-Mitglied, aber Governance-Konzern-Risiken",
    },

    # ── ETF ─────────────────────────────────────────────────────────────────

    "XMEU GY Equity": {
        "name": "Xtrackers MSCI Europe UCITS ETF 1C",
        "isin": "LU0274209237",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Europa",
        "esg_score": 7.783,
        "esg_note": "Breite Europa-Diversifikation, keine explizite ESG-Filterung",
    },
    "VWRL LN Equity": {
        "name": "Vanguard FTSE All-World UCITS ETF (USD) Distributing",
        "isin": "IE00B3RBWM25",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": 7.082,
        "esg_note": "Vollständige Marktabdeckung, kein ESG-Filter",
    },
    "VUSA LN Equity": {
        "name": "Vanguard S&P 500 UCITS ETF",
        "isin": "IE00B3XXRP09",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "USA",
        "esg_score": 7.040,
        "esg_note": "Kein ESG-Filter, schwergewichtet in Öl/Gas",
    },
    "VFEM LN Equity": {
        "name": "Vanguard FTSE Emerging Markets UCITS ETF",
        "isin": "IE00B3VVMM84",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Emerging Markets",
        "esg_score": 7.032,
        "esg_note": "Inkl. China/Staatsunternehmen, niedrigere ESG-Standards",
    },
    "RENW IM Equity": {
        "name": "iShares Global Clean Energy UCITS ETF USD (Acc)",
        "isin": "IE00B1XNHC34",
        "klasse": "ETF", "sektor": "Energie", "region": "Global",
        "esg_score": 5.926,
        "esg_note": "Dezidierter ESG/Cleantech-ETF, sehr hoch",
    },
    "RBOT LN Equity": {
        "name": "iShares Automation & Robotics UCITS ETF",
        "isin": "IE00BYZK4552",
        "klasse": "ETF", "sektor": "Technologie", "region": "Global",
        "esg_score": 6.269,
        "esg_note": "Zukunftstechnologien, mittleres ESG",
    },
    "IWDA LN Equity": {
        "name": "iShares Core MSCI World UCITS ETF USD (Acc)",
        "isin": "IE00B4L5Y983",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Developed Markets",
        "esg_score": 7.113,
        "esg_note": "Kein expliziter ESG-Filter, breite Diversifikation",
    },
    "IWVL LN Equity": {
        "name": "iShares Edge MSCI World Value Factor UCITS ETF",
        "isin": "IE00BP3QZB59",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Developed Markets",
        "esg_score": 7.294,
        "esg_note": "Value-Tilt oft mit höherem Energie-Anteil",
    },
    "IUSN GY Equity": {
        "name": "iShares MSCI World Small Cap UCITS ETF",
        "isin": "IE00BF4RFH31",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": 5.219,
        "esg_note": "Geringere ESG-Transparenz bei kleineren Firmen",
    },
    "ISAC LN Equity": {
        "name": "iShares MSCI ACWI UCITS ETF USD (Acc)",
        "isin": "IE00B6R52259",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": 7.131,
        "esg_note": "All-Country inkl. EM, kein ESG-Filter",
    },
    "IMEU NA Equity": {
        "name": "iShares Core MSCI Europe UCITS ETF EUR (Acc)",
        "isin": "IE00B4K48X80",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Europa",
        "esg_score": 7.784,
        "esg_note": "Breite Europa-Abdeckung, kein expliziter ESG-Filter",
    },
    "EIMI LN Equity": {
        "name": "iShares Core MSCI Emerging Markets IMI UCITS ETF",
        "isin": "IE00BKM4GZ66",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "Emerging Markets",
        "esg_score": 7.093,
        "esg_note": "Breite EM-Abdeckung, niedrigere ESG-Standards",
    },
    "CSPX LN Equity": {
        "name": "iShares Core S&P 500 UCITS ETF USD (Acc)",
        "isin": "IE00B5BMR087",
        "klasse": "ETF", "sektor": "Diversifiziert Aktien", "region": "USA",
        "esg_score": 7.041,
        "esg_note": "Kein ESG-Filter, US-Large-Cap-Abdeckung",
    },

    # ── BOND ETF ─────────────────────────────────────────────────────────────

    "XGLE GY Equity": {
        "name": "Xtrackers II EUR Government Bond UCITS ETF 1C",
        "isin": "LU0290355717",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": None,
        "esg_note": "EU-Staatsanleihen, höhere ESG-Standards vs. EM",
    },
    "XBLC GY Equity": {
        "name": "Xtrackers II EUR Corporate Bond UCITS ETF 1C",
        "isin": "LU0478205379",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": 7.331,
        "esg_note": "Investment Grade EUR Corporate, kein ESG-Filter",
    },
    "VECP GY Equity": {
        "name": "Vanguard EUR Corporate Bond UCITS ETF",
        "isin": "IE00BZ163G84",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": 7.431,
        "esg_note": "IG-Diversifikation EUR, kein ESG-Ausschluss",
    },
    "VDCP LN Equity": {
        "name": "Vanguard USD Corporate Bond UCITS ETF",
        "isin": "IE00BZ163H91",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "USA",
        "esg_score": None,
        "esg_note": "US-IG Corporate, kein ESG-Filter",
    },
    "MTD FP Equity": {
        "name": "AMUNDI EURO GOVERNMENT INFLATION-LINKED BOND UCITS ETF ACC",
        "isin": "LU1650491282",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Global",
        "esg_score": None,
        "esg_note": "Globale Staatsanleihen-Diversifikation",
    },
    "OM3F GY Equity": {
        "name": "iShares EUR Corporate Bond ESG SRI UCITS ETF EUR",
        "isin": "IE00BYZTVT56",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": 7.453,
        "esg_note": "IG-EUR-Unternehmensanleihen",
    },
    "IEAC LN Equity": {
        "name": "iShares Core EUR Corp Bond UCITS ETF",
        "isin": "IE00B3F81R35",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": 7.332,
        "esg_note": "Investment Grade EUR, breite Sektordiversifikation",
    },
    "EUNH GY Equity": {
        "name": "iShares Core EUR Govt Bond UCITS ETF",
        "isin": "IE00B4WXJJ64",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Eurozone",
        "esg_score": None,
        "esg_note": "Eurozone-Staatsanleihen, solides ESG",
    },
    "CSBGE3 IM Equity": {
        "name": "UBS ETF (LU) J.P. Morgan USD EM Diversified Bond 1-5 UCITS ETF",
        "isin": "LU0879399441",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "Europa",
        "esg_score": 7.600,
        "esg_note": "Green Bond fokussiert",
    },
    "CORP LN Equity": {
        "name": "iShares USD Corp Bond UCITS ETF",
        "isin": "IE00B3F81G20",
        "klasse": "Bond ETF", "sektor": "Anleihen", "region": "USA",
        "esg_score": None,
        "esg_note": "US-Investment-Grade, kein ESG-Filter",
    },

    # ── COMMODITIES ──────────────────────────────────────────────────────────

    "PHPT LN Equity": {
        "name": "WisdomTree Physical Platinum",
        "isin": "JE00B1VS2W53",
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Bergbau-Lieferkette in SA, aber Industrie-/H2-Nachfrage",
    },
    "PHAU LN Equity": {
        "name": "WisdomTree Physical Gold",
        "isin": "JE00B1VS3770",
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Bergbau umweltintensiv, aber Wertspeicher ohne Unternehmensrisiko",
    },
    "PHAG IM Equity": {
        "name": "WisdomTree Physical Silver",
        "isin": "JE00B1VS3333",
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Ähnlich Gold, Bergbau-ESG-Risiken",
    },
    "IGLN LN Equity": {
        "name": "iShares Physical Gold ETC",
        "isin": "IE00B4ND3602",
        "klasse": "Commodities", "sektor": "Edelmetalle", "region": "Global",
        "esg_score": None,
        "esg_note": "Physisch gedecktes Gold ETC",
    },

    # ── FUND ─────────────────────────────────────────────────────────────────

    "VNGA60 IM Equity": {
        "name": "Vanguard LifeStrategy 60% Equity UCITS ETF (EUR) Accumulating",
        "isin": "IE00BMVB5Q45",
        "klasse": "Fund", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "60/40 Multi-Asset, kein ESG-Filter, breite Diversifikation",
    },

    # ── INDEX ────────────────────────────────────────────────────────────────

    "catholic index": {
        "name": "Catholic Values Index (intern)",
        "isin": None,
        "klasse": "Index", "sektor": "Diversifiziert Aktien", "region": "Global",
        "esg_score": None,
        "esg_note": "Explizite Ausschlüsse (Waffen, Abtreibung), hohe ESG-Standards",
    },

    # ── CRYPTO ───────────────────────────────────────────────────────────────

    "bitcoin": {
        "name": "Bitcoin",
        "isin": None,
        "klasse": "Crypto", "sektor": "Krypto / Alternativ", "region": "Global",
        "esg_score": None,
        "esg_note": "Proof-of-Work, hoher Energieverbrauch, kein ESG-Rahmen",
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
            "name":      meta.get("name",      "–"),
            "isin":      meta.get("isin",       None),
            "klasse":    meta.get("klasse",     "–"),
            "sektor":    meta.get("sektor",     "–"),
            "region":    meta.get("region",     "–"),
            "esg_score": meta.get("esg_score",  None),
            "esg_note":  meta.get("esg_note",   ""),
        })
    return pd.DataFrame(rows).set_index("asset")


def esg_label(score) -> str:
    """Gibt ein ESG-Risikolabel zurück (Sustainalytics-Skala: niedriger = besser)."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return "n/a"
    if score >= 7.5:  return "Severe"
    if score >= 6.5:  return "High"
    if score >= 5.5:  return "Medium"
    if score >= 4.5:  return "Low"
    return                   "Negligible"


# ─────────────────────────────────────────────────────────────────────────────
# DIREKTER AUFRUF
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_df()
    df["label"] = df["esg_score"].apply(esg_label)

    print(f"\n{'─'*100}")
    print(f"  ASSET METADATEN  –  {len(df)} Assets")
    print(f"{'─'*100}")
    print(df[["name", "isin", "klasse", "sektor", "region", "esg_score", "label"]].to_string())

    print(f"\n{'─'*50}")
    print("  Verteilung nach Klasse")
    print(f"{'─'*50}")
    print(df["klasse"].value_counts().to_string())

    print(f"\n{'─'*50}")
    print("  Verteilung nach Sektor")
    print(f"{'─'*50}")
    print(df["sektor"].value_counts().to_string())

    print(f"\n{'─'*50}")
    print("  Assets ohne ISIN")
    print(f"{'─'*50}")
    no_isin = df[df["isin"].isna()]
    print(no_isin[["name", "klasse"]].to_string() if not no_isin.empty else "  Alle haben ISIN")
