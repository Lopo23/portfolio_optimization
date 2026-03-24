"""
asset_metadata.py
=================
Metadata for all assets – merged from old (79-asset) and new (53-asset) versions.
Total: 83 assets (79 old + BO221256, DAXEX, HIGH, IBCI from new).

Fields per asset:
    name         – Full name / product name
    isin         – ISIN
    asset_class  – Stocks | ETF | Bond ETF | Bond | Commodities | Fund | Index | Crypto
    sector       – see SECTOR TAXONOMY below
    region       – country (single stocks) or geographic exposure (ETFs/Funds)
    esg_score    – Sustainalytics risk score 0–10 (higher = more risk, None = n/a)
    esg_note     – short ESG rationale

──────────────────────────────────────────────────────────────────────────────
SECTOR TAXONOMY  (13 sectors, designed to avoid concentration > 30 %)
──────────────────────────────────────────────────────────────────────────────
Single stocks / thematic ETFs:
    Technology              – semiconductors, software, hardware, IT services
    Industrials & Automotive – capital goods, transport, autos, machinery
    Energy & Utilities      – oil & gas, renewables, power utilities
    Healthcare              – pharma, biotech, medtech, health insurance
    Financials              – banks, insurers, asset managers
    Consumer                – staples + discretionary (food, retail, luxury, apparel)
    Materials               – mining, steel, chemicals, commodities producers

Broad ETFs (split by geography):
    Equity – Americas       – US / North American broad equity ETFs
    Equity – Europe         – European broad equity ETFs
    Equity – Global/EM      – global or emerging-market broad equity ETFs

Bond ETFs / fixed instruments:
    Fixed Income – Govts    – government bond ETFs and sovereign bonds
    Fixed Income – Credit   – corporate bond ETFs (IG and HY)

Real assets & crypto:
    Alternatives            – physical commodities, crypto

──────────────────────────────────────────────────────────────────────────────
ESG NOTE  (Sustainalytics scale: 0–10, lower = better)
    Old file used 0–100 (higher = better).
    Conversion applied: new_score = (100 - old_score) / 10
    Scores from the new file are kept as-is (already on 0–10 scale).
──────────────────────────────────────────────────────────────────────────────

Compatibility:
    ameta() in dashboards reads both "sector"/"asset_class" (this file)
    and "sektor"/"klasse" (legacy) via fallback – both versions work.

Usage:
    from asset_metadata import ASSET_META, get_meta, get_df, esg_label
"""

import pandas as pd


ASSET_META: dict[str, dict] = {

    # ── STOCKS ──────────────────────────────────────────────────────────────

    "AAPL US Equity": {
        "name": "Apple Inc.",
        "isin": "US0378331005",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "100% renewable electricity, carbon neutral by 2030, MSCI AAA",
    },
    "ADBE US Equity": {
        "name": "Adobe Inc.",
        "isin": "US00724F1012",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 4.60,
        "esg_note": "Carbon neutral, MSCI AA, D&I leader in tech",
    },
    "AMZN US Equity": {
        "name": "Amazon.com Inc.",
        "isin": "US0231351067",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 6.20,
        "esg_note": "Climate Pledge signatory, but labour rights and logistics CO2 criticised",
    },
    "ASML NA Equity": {
        "name": "ASML Holding N.V.",
        "isin": "NL0010273215",
        "asset_class": "Stocks", "sector": "Technology", "region": "Netherlands",
        "esg_score": 6.78,
        "esg_note": "Enables chips for green technologies, strong ESG governance",
    },
    "BABA US Equity": {
        "name": "Alibaba Group Holding Ltd. (ADR)",
        "isin": "US01609W1027",
        "asset_class": "Stocks", "sector": "Consumer", "region": "China",
        "esg_score": 7.10,
        "esg_note": "Regulatory risk in China, limited ESG transparency",
    },
    "BHP AU Equity": {
        "name": "BHP Group Limited",
        "isin": "AU000000BHP4",
        "asset_class": "Stocks", "sector": "Materials", "region": "Australia",
        "esg_score": 7.82,
        "esg_note": "Legacy impact from Samarco disaster, but with climate targets and decarbonisation plans",
    },
    "BMW GY Equity": {
        "name": "Bayerische Motoren Werke AG",
        "isin": "DE0005190003",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 5.83,
        "esg_note": "MSCI AA, strong EV roadmap (Neue Klasse), circular economy focus",
    },
    "CON GY Equity": {
        "name": "Continental AG",
        "isin": "DE0005439004",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 6.60,
        "esg_note": "Transformation to electronics/software, medium ESG profile",
    },
    "CRM US Equity": {
        "name": "Salesforce Inc.",
        "isin": "US79466L3024",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 4.40,
        "esg_note": "Net-zero company, 1-1-1 philanthropy model, MSCI AAA",
    },
    "CVX US Equity": {
        "name": "Chevron Corporation",
        "isin": "US1667641005",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 8.30,
        "esg_note": "Oil & gas, weak climate commitment vs. peers",
    },
    "DUK US Equity": {
        "name": "Duke Energy Corporation",
        "isin": "US26441C2044",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Transition from coal to renewables, net-zero by 2050",
    },
    "EBS AV Equity": {
        "name": "Erste Group Bank AG",
        "isin": "AT0000652011",
        "asset_class": "Stocks", "sector": "Financials", "region": "Austria",
        "esg_score": 5.59,
        "esg_note": "Solid ESG profile within the Central European banking sector",
    },
    "GS US Equity": {
        "name": "The Goldman Sachs Group Inc.",
        "isin": "US38141G1040",
        "asset_class": "Stocks", "sector": "Financials", "region": "USA",
        "esg_score": 5.15,
        "esg_note": "USD 750bn sustainable finance commitment, but governance questions remain",
    },
    "HON US Equity": {
        "name": "Honeywell International Inc.",
        "isin": "US4385161066",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "USA",
        "esg_score": 5.60,
        "esg_note": "ESG technologies (building automation), MSCI A",
    },
    "HSBA LN Equity": {
        "name": "HSBC Holdings plc",
        "isin": "GB0005405286",
        "asset_class": "Stocks", "sector": "Financials", "region": "United Kingdom",
        "esg_score": 6.10,
        "esg_note": "Greenwashing allegations 2022, but net-zero commitment",
    },
    "IBN US Equity": {
        "name": "ICICI Bank Ltd. (ADR)",
        "isin": "US45104G1040",
        "asset_class": "Stocks", "sector": "Financials", "region": "India",
        "esg_score": 6.50,
        "esg_note": "Growing ESG frameworks, emerging market standard",
    },
    "INFY US Equity": {
        "name": "Infosys Ltd. (ADR)",
        "isin": "US4567881085",
        "asset_class": "Stocks", "sector": "Technology", "region": "India",
        "esg_score": 4.80,
        "esg_note": "Carbon neutral since 2020, MSCI AA",
    },
    "ISRG US Equity": {
        "name": "Intuitive Surgical Inc.",
        "isin": "US46120E6023",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "USA",
        "esg_score": 5.80,
        "esg_note": "Medical technology, low direct CO2 footprint",
    },
    "JNJ US Equity": {
        "name": "Johnson & Johnson",
        "isin": "US4781601046",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "USA",
        "esg_score": 5.60,
        "esg_note": "ESG leader in pharma, talc litigation weighs on rating",
    },
    "JPM US Equity": {
        "name": "JPMorgan Chase & Co.",
        "isin": "US46625H1005",
        "asset_class": "Stocks", "sector": "Financials", "region": "USA",
        "esg_score": 4.65,
        "esg_note": "Largest US fossil-fuel lender, Paris alignment remains questionable",
    },
    "KO US Equity": {
        "name": "The Coca-Cola Company",
        "isin": "US1912161007",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 5.80,
        "esg_note": "World Without Waste, water stewardship, MSCI A",
    },
    "MBG GY Equity": {
        "name": "Mercedes-Benz Group AG",
        "isin": "DE0007100000",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 6.24,
        "esg_note": "EV transition underway, but still strongly dependent on combustion engines",
    },
    "MC FP Equity": {
        "name": "LVMH Moët Hennessy Louis Vuitton SE",
        "isin": "FR0000121014",
        "asset_class": "Stocks", "sector": "Consumer", "region": "France",
        "esg_score": 6.00,
        "esg_note": "LIFE 360 programme, but luxury consumption sustainability question",
    },
    "MCD US Equity": {
        "name": "McDonald's Corporation",
        "isin": "US5801351017",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 6.50,
        "esg_note": "Scale for Good, but meat production and packaging remain problematic",
    },
    "MSFT US Equity": {
        "name": "Microsoft Corporation",
        "isin": "US5949181045",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 5.18,
        "esg_note": "Carbon negative by 2030, MSCI AAA",
    },
    "NEE US Equity": {
        "name": "NextEra Energy Inc.",
        "isin": "US65339F1012",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 6.52,
        "esg_note": "World's largest producer of wind and solar energy, MSCI AA",
    },
    "NESN SW Equity": {
        "name": "Nestlé S.A.",
        "isin": "CH0012221716",
        "asset_class": "Stocks", "sector": "Consumer", "region": "Switzerland",
        "esg_score": 5.60,
        "esg_note": "Net-zero commitment, water rights criticism, MSCI A",
    },
    "NKE US Equity": {
        "name": "NIKE Inc.",
        "isin": "US6541061031",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 5.80,
        "esg_note": "Move to Zero, but supply-chain labour rights issues",
    },
    "NOVN SW Equity": {
        "name": "Novartis AG",
        "isin": "CH0012221802",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "Switzerland",
        "esg_score": 5.11,
        "esg_note": "Patient access programs, carbon neutral since 2020",
    },
    "NVDA US Equity": {
        "name": "NVIDIA Corporation",
        "isin": "US67066G1040",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 6.94,
        "esg_note": "High energy demand from AI chips, good governance",
    },
    "OMV AV Equity": {
        "name": "OMV AG",
        "isin": "AT0000743059",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "Austria",
        "esg_score": 6.14,
        "esg_note": "Transformation toward chemicals, but fossil businesses remain core",
    },
    "PEP US Equity": {
        "name": "PepsiCo Inc.",
        "isin": "US7134481081",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 5.60,
        "esg_note": "pep+ sustainability plan, MSCI A",
    },
    "PG US Equity": {
        "name": "Procter & Gamble Co.",
        "isin": "US7427181091",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 5.00,
        "esg_note": "Ambition 2030, MSCI AA, strong supply-chain standards",
    },
    "RBI AV Equity": {
        "name": "Raiffeisen Bank International AG",
        "isin": "AT0000606306",
        "asset_class": "Stocks", "sector": "Financials", "region": "Austria",
        "esg_score": 5.83,
        "esg_note": "Russia exposure weighs on ESG rating",
    },
    "RIO LN Equity": {
        "name": "Rio Tinto plc",
        "isin": "GB0007188757",
        "asset_class": "Stocks", "sector": "Materials", "region": "United Kingdom",
        "esg_score": 7.20,
        "esg_note": "Juukan Gorge scandal, but climate targets and governance reforms",
    },
    "ROG SW Equity": {
        "name": "Roche Holding AG",
        "isin": "CH0012032048",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "Switzerland",
        "esg_score": 4.60,
        "esg_note": "Strong ESG programme, access to medicines, MSCI AA",
    },
    "SHEL LN Equity": {
        "name": "Shell plc",
        "isin": "GB00BP6MXD84",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "United Kingdom",
        "esg_score": 8.10,
        "esg_note": "Oil & gas, net-zero target 2050, but continued fossil expansion",
    },
    "SIE GY Equity": {
        "name": "Siemens AG",
        "isin": "DE0007236101",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 6.59,
        "esg_note": "DEGREE sustainability program, MSCI A",
    },
    "TSM US Equity": {
        "name": "Taiwan Semiconductor Manufacturing Co. Ltd. (ADR)",
        "isin": "US8740391003",
        "asset_class": "Stocks", "sector": "Technology", "region": "Taiwan",
        "esg_score": 5.69,
        "esg_note": "High water consumption, but strong governance and 100% renewable target",
    },
    "ULVR LN Equity": {
        "name": "Unilever PLC",
        "isin": "GB00B10RZP78",
        "asset_class": "Stocks", "sector": "Consumer", "region": "United Kingdom",
        "esg_score": 5.28,
        "esg_note": "Leader in sustainability targets, MSCI AA",
    },
    "UNH US Equity": {
        "name": "UnitedHealth Group Inc.",
        "isin": "US91324P1021",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "USA",
        "esg_score": 6.50,
        "esg_note": "Strong governance, but criticism over insurance denials",
    },
    "UNP US Equity": {
        "name": "Union Pacific Corporation",
        "isin": "US9078181081",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "USA",
        "esg_score": 5.80,
        "esg_note": "Rail more efficient than trucking, solid governance",
    },
    "VER AV Equity": {
        "name": "Verbund AG",
        "isin": "AT0000746409",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "Austria",
        "esg_score": 4.60,
        "esg_note": "Nearly 100% hydropower, one of the greenest energy companies in Europe",
    },
    "VOE AV Equity": {
        "name": "voestalpine AG",
        "isin": "AT0000937503",
        "asset_class": "Stocks", "sector": "Materials", "region": "Austria",
        "esg_score": 6.80,
        "esg_note": "Decarbonisation programme (greentec steel), medium ESG profile",
    },
    "VOW3 GY Equity": {
        "name": "Volkswagen AG (Vz.)",
        "isin": "DE0007664039",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 7.10,
        "esg_note": "Dieselgate aftermath, but strong EV strategy (ID series)",
    },
    "XOM US Equity": {
        "name": "Exxon Mobil Corporation",
        "isin": "US30231G1022",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 8.80,
        "esg_note": "Largest US oil major, high CO2 emissions, weak ESG profile",
    },
    "7203 JT Equity": {
        "name": "Toyota Motor Corporation",
        "isin": "JP3633400001",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Japan",
        "esg_score": 6.19,
        "esg_note": "Hybrid pioneer, but slower BEV transition than peers",
    },
    "700 HK Equity": {
        "name": "Tencent Holdings Ltd.",
        "isin": "KYG875721634",
        "asset_class": "Stocks", "sector": "Technology", "region": "China",
        "esg_score": 6.80,
        "esg_note": "Gaming regulation, limited ESG transparency, MSCI BBB",
    },
    "005930 KS Equity": {
        "name": "Samsung Electronics Co. Ltd.",
        "isin": "KR7005930003",
        "asset_class": "Stocks", "sector": "Technology", "region": "South Korea",
        "esg_score": 5.80,
        "esg_note": "RE100 member, but conglomerate governance risks",
    },


    # ── ETF – Americas ───────────────────────────────────────────────────────

    "CSPX LN Equity": {
        "name": "iShares Core S&P 500 UCITS ETF USD (Acc)",
        "isin": "IE00B5BMR087",
        "asset_class": "ETF", "sector": "Equity – Americas", "region": "USA",
        "esg_score": 7.041,
        "esg_note": "No ESG filter, US large-cap exposure",
    },
    "VUSA LN Equity": {
        "name": "Vanguard S&P 500 UCITS ETF",
        "isin": "IE00B3XXRP09",
        "asset_class": "ETF", "sector": "Equity – Americas", "region": "USA",
        "esg_score": 7.040,
        "esg_note": "No ESG filter, heavily weighted toward oil & gas exposure",
    },


    # ── ETF – Europe ─────────────────────────────────────────────────────────

    "DAXEX GY Equity": {
        "name": "iShares Core DAX UCITS ETF (DE)",
        "isin": "DE0005933931",
        "asset_class": "ETF", "sector": "Equity – Europe", "region": "Germany",
        "esg_score": 7.80,
        "esg_note": "Broad German large-cap exposure, no ESG screening",
    },
    "IMEU NA Equity": {
        "name": "iShares Core MSCI Europe UCITS ETF EUR (Acc)",
        "isin": "IE00B4K48X80",
        "asset_class": "ETF", "sector": "Equity – Europe", "region": "Europe",
        "esg_score": 7.784,
        "esg_note": "Broad Europe exposure, no explicit ESG filter",
    },
    "XMEU GY Equity": {
        "name": "Xtrackers MSCI Europe UCITS ETF 1C",
        "isin": "LU0274209237",
        "asset_class": "ETF", "sector": "Equity – Europe", "region": "Europe",
        "esg_score": 7.783,
        "esg_note": "Broad European diversification, no explicit ESG screening",
    },


    # ── ETF – Global / EM ────────────────────────────────────────────────────

    "EIMI LN Equity": {
        "name": "iShares Core MSCI Emerging Markets IMI UCITS ETF",
        "isin": "IE00BKM4GZ66",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Emerging Markets",
        "esg_score": 7.093,
        "esg_note": "Broad EM exposure, lower ESG standards",
    },
    "ISAC LN Equity": {
        "name": "iShares MSCI ACWI UCITS ETF USD (Acc)",
        "isin": "IE00B6R52259",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Global",
        "esg_score": 7.131,
        "esg_note": "All-country exposure incl. EM, no ESG filter",
    },
    "IUSN GY Equity": {
        "name": "iShares MSCI World Small Cap UCITS ETF",
        "isin": "IE00BF4RFH31",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Global",
        "esg_score": 5.219,
        "esg_note": "Lower ESG transparency among smaller companies",
    },
    "IWDA LN Equity": {
        "name": "iShares Core MSCI World UCITS ETF USD (Acc)",
        "isin": "IE00B4L5Y983",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Developed Markets",
        "esg_score": 7.113,
        "esg_note": "No explicit ESG filter, broad developed-market diversification",
    },
    "IWVL LN Equity": {
        "name": "iShares Edge MSCI World Value Factor UCITS ETF",
        "isin": "IE00BP3QZB59",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Developed Markets",
        "esg_score": 7.294,
        "esg_note": "Value tilt often comes with higher energy exposure",
    },
    "VFEM LN Equity": {
        "name": "Vanguard FTSE Emerging Markets UCITS ETF",
        "isin": "IE00B3VVMM84",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Emerging Markets",
        "esg_score": 7.032,
        "esg_note": "Includes China and state-owned companies, lower ESG standards",
    },
    "VWRL LN Equity": {
        "name": "Vanguard FTSE All-World UCITS ETF (USD) Distributing",
        "isin": "IE00B3RBWM25",
        "asset_class": "ETF", "sector": "Equity – Global/EM", "region": "Global",
        "esg_score": 7.082,
        "esg_note": "Full market coverage, no ESG filter",
    },


    # ── ETF – Thematic ───────────────────────────────────────────────────────

    "RBOT LN Equity": {
        "name": "iShares Automation & Robotics UCITS ETF",
        "isin": "IE00BYZK4552",
        "asset_class": "ETF", "sector": "Technology", "region": "Global",
        "esg_score": 6.269,
        "esg_note": "Future technologies, medium ESG profile",
    },
    "RENW IM Equity": {
        "name": "iShares Global Clean Energy UCITS ETF USD (Acc)",
        "isin": "IE00B1XNHC34",
        "asset_class": "ETF", "sector": "Energy & Utilities", "region": "Global",
        "esg_score": 5.926,
        "esg_note": "Dedicated ESG/cleantech ETF, strong thematic sustainability angle",
    },


    # ── BOND ETF – Govts ─────────────────────────────────────────────────────

    "EUNH GY Equity": {
        "name": "iShares Core EUR Govt Bond UCITS ETF",
        "isin": "IE00B4WXJJ64",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Govts", "region": "Europe",
        "esg_score": 7.50,
        "esg_note": "Eurozone government bonds, solid ESG profile",
    },
    "IBCI IM Equity": {
        "name": "iShares € Inflation Linked Govt Bond UCITS ETF",
        "isin": "IE00B0M62X26",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Govts", "region": "Europe",
        "esg_score": 7.50,
        "esg_note": "Eurozone inflation-linked government bonds",
    },
    "MTD FP Equity": {
        "name": "Amundi Euro Govt Inflation-Linked Bond UCITS ETF",
        "isin": "LU1650491282",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Govts", "region": "Global",
        "esg_score": 7.40,
        "esg_note": "Global sovereign bond diversification",
    },
    "XGLE GY Equity": {
        "name": "Xtrackers II EUR Government Bond UCITS ETF 1C",
        "isin": "LU0290355717",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Govts", "region": "Europe",
        "esg_score": 7.450,
        "esg_note": "EU sovereign bonds, higher ESG standards versus EM",
    },


    # ── BOND ETF – Credit ────────────────────────────────────────────────────

    "CORP LN Equity": {
        "name": "iShares USD Corp Bond UCITS ETF",
        "isin": "IE00B3F81G20",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "USA",
        "esg_score": 7.20,
        "esg_note": "US investment-grade bonds, no ESG filter",
    },
    "CSBGE3 IM Equity": {
        "name": "UBS ETF J.P. Morgan USD EM Diversified Bond 1-5 UCITS ETF",
        "isin": "LU0879399441",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "Europe",
        "esg_score": 7.600,
        "esg_note": "Focused on green bond exposure",
    },
    "HIGH LN Equity": {
        "name": "iShares € High Yield Corp Bond UCITS ETF",
        "isin": "IE00BF3N7094",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "Europe",
        "esg_score": 7.30,
        "esg_note": "Sub-investment-grade EUR corporate bonds, no ESG filter",
    },
    "IEAC LN Equity": {
        "name": "iShares Core EUR Corp Bond UCITS ETF",
        "isin": "IE00B3F81R35",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "Europe",
        "esg_score": 7.332,
        "esg_note": "Investment-grade EUR bonds, broad sector diversification",
    },
    "OM3F GY Equity": {
        "name": "iShares EUR Corporate Bond ESG SRI UCITS ETF EUR",
        "isin": "IE00BYZTVT56",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "Europe",
        "esg_score": 7.453,
        "esg_note": "Investment-grade EUR corporate bond exposure with ESG/SRI tilt",
    },
    "VDCP LN Equity": {
        "name": "Vanguard USD Corporate Bond UCITS ETF",
        "isin": "IE00BZ163H91",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "USA",
        "esg_score": 7.250,
        "esg_note": "US investment-grade corporate bonds, no ESG filter",
    },
    "VECP GY Equity": {
        "name": "Vanguard EUR Corporate Bond UCITS ETF",
        "isin": "IE00BZ163G84",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "Europe",
        "esg_score": 7.431,
        "esg_note": "Diversified EUR IG bond exposure, no ESG exclusions",
    },
    "XBLC GY Equity": {
        "name": "Xtrackers II EUR Corporate Bond UCITS ETF 1C",
        "isin": "LU0478205379",
        "asset_class": "Bond ETF", "sector": "Fixed Income – Credit", "region": "Europe",
        "esg_score": 7.331,
        "esg_note": "Investment-grade EUR corporate bonds, no ESG filter",
    },


    # ── BOND (single instruments) ────────────────────────────────────────────

    "BO221256 Corp": {
        "name": "German Bund 15 May 2036 (Zero Coupon)",
        "isin": "DE0001102549",
        "asset_class": "Bond", "sector": "Fixed Income – Govts", "region": "Germany",
        "esg_score": None,
        "esg_note": "German sovereign bond – no Sustainalytics issuer-level score applicable",
    },


    # ── COMMODITIES ──────────────────────────────────────────────────────────

    "IGLN LN Equity": {
        "name": "iShares Physical Gold ETC",
        "isin": "IE00B4ND3602",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Physically backed gold ETC",
    },
    "PHAG IM Equity": {
        "name": "WisdomTree Physical Silver",
        "isin": "JE00B1VS3333",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Similar to gold, with mining-related ESG risks",
    },
    "PHAU LN Equity": {
        "name": "WisdomTree Physical Gold",
        "isin": "JE00B1VS3770",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Mining is environmentally intensive, but gold acts as a store of value without company-specific risk",
    },
    "PHPT LN Equity": {
        "name": "WisdomTree Physical Platinum",
        "isin": "JE00B1VS2W53",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Mining supply chain concentrated in South Africa, but supported by industrial/hydrogen demand",
    },


    # ── FUND ─────────────────────────────────────────────────────────────────

    "VNGA60 IM Equity": {
        "name": "Vanguard LifeStrategy 60% Equity UCITS ETF Accumulating",
        "isin": "IE00BMVB5P51",
        "asset_class": "Fund", "sector": "Equity – Global/EM", "region": "Global",
        "esg_score": 7.35,
        "esg_note": "60/40 multi-asset strategy, no ESG filter, broad diversification",
    },


    # ── INDEX ────────────────────────────────────────────────────────────────

    "catholic index": {
        "name": "Catholic Values Index (internal)",
        "isin": "",
        "asset_class": "Index", "sector": "Equity – Global/EM", "region": "Global",
        "esg_score": 3.826,
        "esg_note": "Explicit exclusions (weapons, abortion), high ESG standards",
    },


    # ── CRYPTO ───────────────────────────────────────────────────────────────

    "bitcoin": {
        "name": "Bitcoin",
        "isin": None,
        "asset_class": "Crypto", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Proof-of-Work, high energy consumption, no ESG framework",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_meta(asset: str) -> dict:
    """Return metadata for an asset. Empty dict if not found."""
    return ASSET_META.get(asset, {})


def get_df() -> pd.DataFrame:
    """Return all metadata as a pandas DataFrame."""
    rows = []
    for asset, meta in ASSET_META.items():
        rows.append({
            "asset":       asset,
            "name":        meta.get("name",        "–"),
            "isin":        meta.get("isin",        None),
            "asset_class": meta.get("asset_class", "–"),
            "sector":      meta.get("sector",      "–"),
            "region":      meta.get("region",      "–"),
            "esg_score":   meta.get("esg_score",   None),
            "esg_note":    meta.get("esg_note",    ""),
        })
    return pd.DataFrame(rows).set_index("asset")


def esg_label(score) -> str:
    """Return an ESG quality label.
    Scale: 0–10, HIGHER = better (LSEG/Refinitiv legacy score divided by 10).
    Bands:
        AAA  >= 7.5   – outstanding
        AA   >= 6.5   – strong
        A    >= 5.5   – above average
        BBB  >= 4.5   – average
        BB   >= 3.5   – below average
        B    >= 2.5   – weak
        CCC   < 2.5   – very weak
    """
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return "n/a"
    if score >= 7.5:  return "AAA"
    if score >= 6.5:  return "AA"
    if score >= 5.5:  return "A"
    if score >= 4.5:  return "BBB"
    if score >= 3.5:  return "BB"
    if score >= 2.5:  return "B"
    return "CCC"


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_df()
    df["label"] = df["esg_score"].apply(esg_label)

    print(f"\n{'─'*100}")
    print(f"  ASSET METADATA  –  {len(df)} assets")
    print(f"{'─'*100}")
    print(df[["name", "asset_class", "sector", "region", "esg_score", "label"]].to_string())

    print(f"\n{'─'*55}")
    print("  BY ASSET CLASS")
    print(f"{'─'*55}")
    for cls, grp in df.groupby("asset_class"):
        print(f"  {cls:15s}  {len(grp):3d} assets")

    print(f"\n{'─'*55}")
    print("  SECTOR DISTRIBUTION  (constraint target: ≤ 30%)")
    print(f"{'─'*55}")
    n = len(df)
    for s, c in df["sector"].value_counts().items():
        flag = "⚠️ " if c/n > 0.30 else "✅"
        print(f"  {flag}  {s:35s}  {c:2d}  ({c/n:.0%})")

    print(f"\n{'─'*55}")
    print("  REGION DISTRIBUTION  (constraint target: ≤ 40%)")
    print(f"{'─'*55}")
    for r, c in df["region"].value_counts().items():
        flag = "⚠️ " if c/n > 0.40 else "✅"
        print(f"  {flag}  {r:30s}  {c:2d}  ({c/n:.0%})")

    print(f"\n{'─'*55}")
    print("  STRATEGIC PORTFOLIO COVERAGE")
    print(f"{'─'*55}")
    strategic = ["VNGA60 IM Equity", "CSBGE3 IM Equity", "CSPX LN Equity",
                 "IWVL LN Equity", "7203 JT Equity", "EBS AV Equity",
                 "MSFT US Equity", "NOVN SW Equity", "AAPL US Equity",
                 "PHAU LN Equity", "bitcoin", "catholic index"]
    for a in strategic:
        ok  = a in df.index
        row = df.loc[a] if ok else None
        print(f"  {'✅' if ok else '❌'}  {a:25s}"
              + (f"  {row['asset_class']:12s}  {row['sector']}" if ok else "  MISSING"))
