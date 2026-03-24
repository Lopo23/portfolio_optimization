"""
asset_metadata.py
=================
Metadata for all assets in the new portfolio.

Fields per asset:
    name         – Full name / product name
    isin         – ISIN (International Securities Identification Number)
    asset_class  – Asset class (Stocks, ETF, Bond ETF, Commodities, Fund, Index, Crypto)
    sector       – One of the 10 main sectors (see below)
    region       – Country (single stocks) or geographic exposure (ETFs/Funds)
    esg_score    – Sustainalytics-like score 0–10 (higher = riskier, None = not available)
    esg_note     – Short ESG rationale

Sectors (10):
    Technology
    Industrials & Automotive
    Energy & Utilities
    Healthcare
    Financials
    Consumer
    Materials
    Diversified Equity
    Fixed Income
    Alternatives

Usage:
    from asset_metadata import ASSET_META, get_meta, get_df, esg_label

    meta = get_meta("AAPL US Equity")
    df   = get_df()   # -> DataFrame with all fields
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
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 5.45,
        "esg_note": "Largest US oil major, high CO2 emissions, weak ESG profile",
    },
    "VOW3 GY Equity": {
        "name": "Volkswagen AG (Preferred Share)",
        "isin": "DE0007664039",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 5.20,
        "esg_note": "Dieselgate after-effects, but strong EV strategy (ID series)",
    },
    "VOE AV Equity": {
        "name": "voestalpine AG",
        "isin": "AT0000937503",
        "asset_class": "Stocks", "sector": "Materials", "region": "Austria",
        "esg_score": 4.95,
        "esg_note": "Decarbonization program greentec steel, medium ESG profile",
    },
    "VER AV Equity": {
        "name": "Verbund AG",
        "isin": "AT0000746409",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "Austria",
        "esg_score": 5.34,
        "esg_note": "Almost 100% hydropower, one of Europe's greenest energy companies",
    },
    "UNP US Equity": {
        "name": "Union Pacific Corporation",
        "isin": "US9078181081",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "USA",
        "esg_score": 4.52,
        "esg_note": "Rail is more efficient than trucking, solid governance",
    },
    "UNH US Equity": {
        "name": "UnitedHealth Group Incorporated",
        "isin": "US91324P1021",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "USA",
        "esg_score": 6.31,
        "esg_note": "Strong governance, but criticized for insurance claim denials",
    },
    "ULVR LN Equity": {
        "name": "Unilever PLC",
        "isin": "GB00B10RZP78",
        "asset_class": "Stocks", "sector": "Consumer", "region": "United Kingdom",
        "esg_score": 5.28,
        "esg_note": "Leader in sustainability targets, MSCI AA",
    },
    "TSM US Equity": {
        "name": "Taiwan Semiconductor Manufacturing Co. Ltd. (ADR)",
        "isin": "US8740391003",
        "asset_class": "Stocks", "sector": "Technology", "region": "Taiwan",
        "esg_score": 5.69,
        "esg_note": "High water consumption, but strong governance and 100% renewable target",
    },
    "SIE GY Equity": {
        "name": "Siemens AG",
        "isin": "DE0007236101",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 6.59,
        "esg_note": "DEGREE sustainability program, MSCI A",
    },
    "SHEL LN Equity": {
        "name": "Shell PLC",
        "isin": "GB00BP6MXD84",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "United Kingdom",
        "esg_score": 6.74,
        "esg_note": "Net-zero target by 2050, but ongoing fossil fuel expansion",
    },
    "ROG SW Equity": {
        "name": "Roche Holding AG",
        "isin": "CH0012221716",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "Switzerland",
        "esg_score": 4.53,
        "esg_note": "Strong ESG program, access to medicine, MSCI AA",
    },
    "RIO LN Equity": {
        "name": "Rio Tinto PLC",
        "isin": "GB0007188757",
        "asset_class": "Stocks", "sector": "Materials", "region": "United Kingdom",
        "esg_score": 6.92,
        "esg_note": "Juukan Gorge scandal, but climate targets and governance reforms",
    },
    "RBI AV Equity": {
        "name": "Raiffeisen Bank International AG",
        "isin": "AT0000606306",
        "asset_class": "Stocks", "sector": "Financials", "region": "Austria",
        "esg_score": 5.83,
        "esg_note": "Russia exposure weighs on ESG rating",
    },
    "PG US Equity": {
        "name": "Procter & Gamble Company",
        "isin": "US7427181091",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 4.96,
        "esg_note": "Ambition 2030, MSCI AA, strong supply-chain standards",
    },
    "PEP US Equity": {
        "name": "PepsiCo Inc.",
        "isin": "US7134481081",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "pep+ sustainability plan, MSCI A",
    },
    "OMV AV Equity": {
        "name": "OMV AG",
        "isin": "AT0000743059",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "Austria",
        "esg_score": 6.14,
        "esg_note": "Transformation toward chemicals, but fossil businesses remain core",
    },
    "NVDA US Equity": {
        "name": "NVIDIA Corporation",
        "isin": "US67066G1040",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 6.94,
        "esg_note": "High energy demand from AI chips, good governance",
    },
    "NOVN SW Equity": {
        "name": "Novartis AG",
        "isin": "CH0012221802",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "Switzerland",
        "esg_score": 5.11,
        "esg_note": "Patient access programs, carbon neutral since 2020",
    },
    "NKE US Equity": {
        "name": "NIKE Inc.",
        "isin": "US6541061031",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 4.99,
        "esg_note": "Move to Zero, but labor-rights issues in the supply chain",
    },
    "NESN SW Equity": {
        "name": "Nestlé S.A.",
        "isin": "CH0038863350",
        "asset_class": "Stocks", "sector": "Consumer", "region": "Switzerland",
        "esg_score": 5.44,
        "esg_note": "Net-zero commitment, criticism around water rights, MSCI A",
    },
    "NEE US Equity": {
        "name": "NextEra Energy Inc.",
        "isin": "US65339F1012",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 6.52,
        "esg_note": "World's largest producer of wind and solar energy, MSCI AA",
    },
    "MSFT US Equity": {
        "name": "Microsoft Corporation",
        "isin": "US5949181045",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 5.18,
        "esg_note": "Carbon negative by 2030, MSCI AAA",
    },
    "MCD US Equity": {
        "name": "McDonald's Corporation",
        "isin": "US5801351017",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 4.25,
        "esg_note": "Scale for Good, but meat production and packaging remain problematic",
    },
    "MC FP Equity": {
        "name": "LVMH Moët Hennessy Louis Vuitton SE",
        "isin": "FR0000121014",
        "asset_class": "Stocks", "sector": "Consumer", "region": "France",
        "esg_score": 4.86,
        "esg_note": "LIFE 360 program, but sustainability of luxury consumption is debated",
    },
    "MBG GY Equity": {
        "name": "Mercedes-Benz Group AG",
        "isin": "DE0007100000",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 6.24,
        "esg_note": "EV transition underway, but still strongly dependent on combustion engines",
    },
    "KO US Equity": {
        "name": "The Coca-Cola Company",
        "isin": "US1912161007",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 5.14,
        "esg_note": "World Without Waste, water stewardship, MSCI A",
    },
    "JPM US Equity": {
        "name": "JPMorgan Chase & Co.",
        "isin": "US46625H1005",
        "asset_class": "Stocks", "sector": "Financials", "region": "USA",
        "esg_score": 4.65,
        "esg_note": "Largest US fossil-fuel lender, Paris alignment remains questionable",
    },
    "JNJ US Equity": {
        "name": "Johnson & Johnson",
        "isin": "US4781601046",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "USA",
        "esg_score": 5.84,
        "esg_note": "ESG leader in pharma, but talc litigation weighs on profile",
    },
    "ISRG US Equity": {
        "name": "Intuitive Surgical Inc.",
        "isin": "US46120E6023",
        "asset_class": "Stocks", "sector": "Healthcare", "region": "USA",
        "esg_score": 3.64,
        "esg_note": "Medical technology, low direct carbon footprint",
    },
    "INFY US Equity": {
        "name": "Infosys Limited (ADR)",
        "isin": "US4567881085",
        "asset_class": "Stocks", "sector": "Technology", "region": "India",
        "esg_score": 6.72,
        "esg_note": "Carbon neutral since 2020, MSCI AA",
    },
    "IBN US Equity": {
        "name": "ICICI Bank Limited (ADR)",
        "isin": "US45104G1040",
        "asset_class": "Stocks", "sector": "Financials", "region": "India",
        "esg_score": 3.94,
        "esg_note": "Growing ESG frameworks, in line with emerging-market standards",
    },
    "HSBA LN Equity": {
        "name": "HSBC Holdings PLC",
        "isin": "GB0005405286",
        "asset_class": "Stocks", "sector": "Financials", "region": "United Kingdom",
        "esg_score": 4.26,
        "esg_note": "Greenwashing accusations in 2022, but net-zero commitment remains",
    },
    "HON US Equity": {
        "name": "Honeywell International Inc.",
        "isin": "US4385161066",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "USA",
        "esg_score": 5.88,
        "esg_note": "ESG technologies (building automation), MSCI A",
    },
    "GS US Equity": {
        "name": "The Goldman Sachs Group Inc.",
        "isin": "US38141G1040",
        "asset_class": "Stocks", "sector": "Financials", "region": "USA",
        "esg_score": 5.15,
        "esg_note": "USD 750bn sustainable finance commitment, but governance questions remain",
    },
    "EBS AV Equity": {
        "name": "Erste Group Bank AG",
        "isin": "AT0000652011",
        "asset_class": "Stocks", "sector": "Financials", "region": "Austria",
        "esg_score": 5.59,
        "esg_note": "Solid ESG profile within the Central European banking sector",
    },
    "DUK US Equity": {
        "name": "Duke Energy Corporation",
        "isin": "US26441C2044",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Transition from coal to renewables, net-zero by 2050",
    },
    "CVX US Equity": {
        "name": "Chevron Corporation",
        "isin": "US1667641005",
        "asset_class": "Stocks", "sector": "Energy & Utilities", "region": "USA",
        "esg_score": 5.33,
        "esg_note": "Oil & gas exposure, weaker climate commitment versus peers",
    },
    "CRM US Equity": {
        "name": "Salesforce Inc.",
        "isin": "US79466L3024",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 5.52,
        "esg_note": "Net-zero company, 1-1-1 philanthropy model, MSCI AAA",
    },
    "CON GY Equity": {
        "name": "Continental AG",
        "isin": "DE0005439004",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 5.20,
        "esg_note": "Transformation toward electronics/software, medium ESG profile",
    },
    "BMW GY Equity": {
        "name": "Bayerische Motoren Werke AG",
        "isin": "DE0005190003",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Germany",
        "esg_score": 5.83,
        "esg_note": "MSCI AA, strong EV roadmap (Neue Klasse), circular economy focus",
    },
    "BHP AU Equity": {
        "name": "BHP Group Limited",
        "isin": "AU000000BHP4",
        "asset_class": "Stocks", "sector": "Materials", "region": "Australia",
        "esg_score": 7.82,
        "esg_note": "Legacy impact from Samarco disaster, but with climate targets and decarbonization plans",
    },
    "BABA US Equity": {
        "name": "Alibaba Group Holding Limited (ADR)",
        "isin": "US01609W1027",
        "asset_class": "Stocks", "sector": "Technology", "region": "China",
        "esg_score": 5.17,
        "esg_note": "China regulatory risk, limited ESG transparency",
    },
    "ASML NA Equity": {
        "name": "ASML Holding N.V.",
        "isin": "NL0010273215",
        "asset_class": "Stocks", "sector": "Technology", "region": "Netherlands",
        "esg_score": 6.78,
        "esg_note": "Enables chips for green technologies, strong ESG governance",
    },
    "AMZN US Equity": {
        "name": "Amazon.com Inc.",
        "isin": "US0231351067",
        "asset_class": "Stocks", "sector": "Consumer", "region": "USA",
        "esg_score": 4.32,
        "esg_note": "Climate Pledge, but criticized for labor rights and logistics emissions",
    },
    "ADBE US Equity": {
        "name": "Adobe Inc.",
        "isin": "US00724F1012",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 5.96,
        "esg_note": "Carbon neutral, MSCI AA, diversity & inclusion leader",
    },
    "AAPL US Equity": {
        "name": "Apple Inc.",
        "isin": "US0378331005",
        "asset_class": "Stocks", "sector": "Technology", "region": "USA",
        "esg_score": 5.83,
        "esg_note": "100% renewable electricity, carbon neutral by 2030, MSCI AAA",
    },
    "7203 JT Equity": {
        "name": "Toyota Motor Corporation",
        "isin": "JP3633400001",
        "asset_class": "Stocks", "sector": "Industrials & Automotive", "region": "Japan",
        "esg_score": 6.19,
        "esg_note": "Hybrid pioneer, but slower BEV transition than peers",
    },
    "700 HK Equity": {
        "name": "Tencent Holdings Limited",
        "isin": "KYG875721634",
        "asset_class": "Stocks", "sector": "Technology", "region": "China",
        "esg_score": 4.79,
        "esg_note": "Gaming regulation risk, limited ESG transparency, MSCI BBB",
    },
    "005930 KS Equity": {
        "name": "Samsung Electronics Co., Ltd.",
        "isin": "KR7005930003",
        "asset_class": "Stocks", "sector": "Technology", "region": "South Korea",
        "esg_score": 4.94,
        "esg_note": "RE100 member, but conglomerate governance risks remain",
    },

    # ── ETF ─────────────────────────────────────────────────────────────────

    "XMEU GY Equity": {
        "name": "Xtrackers MSCI Europe UCITS ETF 1C",
        "isin": "LU0274209237",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Europe",
        "esg_score": 7.783,
        "esg_note": "Broad European diversification, no explicit ESG screening",
    },
    "VWRL LN Equity": {
        "name": "Vanguard FTSE All-World UCITS ETF (USD) Distributing",
        "isin": "IE00B3RBWM25",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Global",
        "esg_score": 7.082,
        "esg_note": "Full market coverage, no ESG filter",
    },
    "VUSA LN Equity": {
        "name": "Vanguard S&P 500 UCITS ETF",
        "isin": "IE00B3XXRP09",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "USA",
        "esg_score": 7.040,
        "esg_note": "No ESG filter, heavily weighted toward oil & gas exposure",
    },
    "VFEM LN Equity": {
        "name": "Vanguard FTSE Emerging Markets UCITS ETF",
        "isin": "IE00B3VVMM84",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Emerging Markets",
        "esg_score": 7.032,
        "esg_note": "Includes China and state-owned companies, lower ESG standards",
    },
    "RENW IM Equity": {
        "name": "iShares Global Clean Energy UCITS ETF USD (Acc)",
        "isin": "IE00B1XNHC34",
        "asset_class": "ETF", "sector": "Energy & Utilities", "region": "Global",
        "esg_score": 5.926,
        "esg_note": "Dedicated ESG/cleantech ETF, strong thematic sustainability angle",
    },
    "RBOT LN Equity": {
        "name": "iShares Automation & Robotics UCITS ETF",
        "isin": "IE00BYZK4552",
        "asset_class": "ETF", "sector": "Technology", "region": "Global",
        "esg_score": 6.269,
        "esg_note": "Future technologies, medium ESG profile",
    },
    "IWDA LN Equity": {
        "name": "iShares Core MSCI World UCITS ETF USD (Acc)",
        "isin": "IE00B4L5Y983",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Developed Markets",
        "esg_score": 7.113,
        "esg_note": "No explicit ESG filter, broad diversification",
    },
    "IWVL LN Equity": {
        "name": "iShares Edge MSCI World Value Factor UCITS ETF",
        "isin": "IE00BP3QZB59",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Developed Markets",
        "esg_score": 7.294,
        "esg_note": "Value tilt often comes with higher energy exposure",
    },
    "IUSN GY Equity": {
        "name": "iShares MSCI World Small Cap UCITS ETF",
        "isin": "IE00BF4RFH31",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Global",
        "esg_score": 5.219,
        "esg_note": "Lower ESG transparency among smaller companies",
    },
    "ISAC LN Equity": {
        "name": "iShares MSCI ACWI UCITS ETF USD (Acc)",
        "isin": "IE00B6R52259",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Global",
        "esg_score": 7.131,
        "esg_note": "All-country exposure incl. EM, no ESG filter",
    },
    "IMEU NA Equity": {
        "name": "iShares Core MSCI Europe UCITS ETF EUR (Acc)",
        "isin": "IE00B4K48X80",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Europe",
        "esg_score": 7.784,
        "esg_note": "Broad Europe exposure, no explicit ESG filter",
    },
    "EIMI LN Equity": {
        "name": "iShares Core MSCI Emerging Markets IMI UCITS ETF",
        "isin": "IE00BKM4GZ66",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "Emerging Markets",
        "esg_score": 7.093,
        "esg_note": "Broad EM exposure, lower ESG standards",
    },
    "CSPX LN Equity": {
        "name": "iShares Core S&P 500 UCITS ETF USD (Acc)",
        "isin": "IE00B5BMR087",
        "asset_class": "ETF", "sector": "Diversified Equity", "region": "USA",
        "esg_score": 7.041,
        "esg_note": "No ESG filter, US large-cap exposure",
    },

    # ── BOND ETF ─────────────────────────────────────────────────────────────

    "XGLE GY Equity": {
        "name": "Xtrackers II EUR Government Bond UCITS ETF 1C",
        "isin": "LU0290355717",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.450,
        "esg_note": "EU sovereign bonds, higher ESG standards versus EM",
    },
    "XBLC GY Equity": {
        "name": "Xtrackers II EUR Corporate Bond UCITS ETF 1C",
        "isin": "LU0478205379",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.331,
        "esg_note": "Investment-grade EUR corporate bonds, no ESG filter",
    },
    "VECP GY Equity": {
        "name": "Vanguard EUR Corporate Bond UCITS ETF",
        "isin": "IE00BZ163G84",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.431,
        "esg_note": "Diversified EUR IG bond exposure, no ESG exclusions",
    },
    "VDCP LN Equity": {
        "name": "Vanguard USD Corporate Bond UCITS ETF",
        "isin": "IE00BZ163H91",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "USA",
        "esg_score": 7.250,
        "esg_note": "US investment-grade corporate bonds, no ESG filter",
    },
    "MTD FP Equity": {
        "name": "AMUNDI EURO GOVERNMENT INFLATION-LINKED BOND UCITS ETF ACC",
        "isin": "LU1650491282",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Global",
        "esg_score": 7.40,
        "esg_note": "Global sovereign bond diversification",
    },
    "OM3F GY Equity": {
        "name": "iShares EUR Corporate Bond ESG SRI UCITS ETF EUR",
        "isin": "IE00BYZTVT56",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.453,
        "esg_note": "Investment-grade EUR corporate bond exposure with ESG/SRI tilt",
    },
    "IEAC LN Equity": {
        "name": "iShares Core EUR Corp Bond UCITS ETF",
        "isin": "IE00B3F81R35",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.332,
        "esg_note": "Investment-grade EUR bonds, broad sector diversification",
    },
    "EUNH GY Equity": {
        "name": "iShares Core EUR Govt Bond UCITS ETF",
        "isin": "IE00B4WXJJ64",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.50,
        "esg_note": "Eurozone government bonds, solid ESG profile",
    },
    "CSBGE3 IM Equity": {
        "name": "UBS ETF (LU) J.P. Morgan USD EM Diversified Bond 1-5 UCITS ETF",
        "isin": "LU0879399441",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "Europe",
        "esg_score": 7.600,
        "esg_note": "Focused on green bond exposure",
    },
    "CORP LN Equity": {
        "name": "iShares USD Corp Bond UCITS ETF",
        "isin": "IE00B3F81G20",
        "asset_class": "Bond ETF", "sector": "Fixed Income", "region": "USA",
        "esg_score": 7.20,
        "esg_note": "US investment-grade bonds, no ESG filter",
    },

    # ── COMMODITIES ──────────────────────────────────────────────────────────

    "PHPT LN Equity": {
        "name": "WisdomTree Physical Platinum",
        "isin": "JE00B1VS2W53",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Mining supply chain concentrated in South Africa, but supported by industrial/hydrogen demand",
    },
    "PHAU LN Equity": {
        "name": "WisdomTree Physical Gold",
        "isin": "JE00B1VS3770",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Mining is environmentally intensive, but gold acts as a store of value without company-specific risk",
    },
    "PHAG IM Equity": {
        "name": "WisdomTree Physical Silver",
        "isin": "JE00B1VS3333",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Similar to gold, with mining-related ESG risks",
    },
    "IGLN LN Equity": {
        "name": "iShares Physical Gold ETC",
        "isin": "IE00B4ND3602",
        "asset_class": "Commodities", "sector": "Alternatives", "region": "Global",
        "esg_score": None,
        "esg_note": "Physically backed gold ETC",
    },

    # ── FUND ─────────────────────────────────────────────────────────────────

    "VNGA60 IM Equity": {
        "name": "Vanguard LifeStrategy 60% Equity UCITS ETF Accumulating",
        "isin": "IE00BMVB5P51",
        "asset_class": "Fund", "sector": "Diversified Equity", "region": "Global",
        "esg_score": 7.35,
        "esg_note": "60/40 multi-asset strategy, no ESG filter, broad diversification",
    },

    # ── INDEX ────────────────────────────────────────────────────────────────

    "catholic index": {
        "name": "Catholic Values Index (internal)",
        "isin": "",
        "asset_class": "Index", "sector": "Diversified Equity", "region": "Global",
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
    """Return an ESG risk label (Sustainalytics scale: lower = better)."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return "n/a"
    if score >= 7.5:
        return "Severe"
    if score >= 6.5:
        return "High"
    if score >= 5.5:
        return "Medium"
    if score >= 4.5:
        return "Low"
    return "Negligible"


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_df()
    df["label"] = df["esg_score"].apply(esg_label)

    print(f"\n{'─'*100}")
    print(f"  ASSET METADATA  –  {len(df)} Assets")
    print(f"{'─'*100}")
    print(df[["name", "isin", "asset_class", "sector", "region", "esg_score", "label"]].to_string())

    print(f"\n{'─'*50}")
    print("  Distribution by Asset Class")
    print(f"{'─'*50}")
    print(df["asset_class"].value_counts().to_string())

    print(f"\n{'─'*50}")
    print("  Distribution by Sector")
    print(f"{'─'*50}")
    print(df["sector"].value_counts().to_string())

    print(f"\n{'─'*50}")
    print("  Assets without ISIN")
    print(f"{'─'*50}")
    no_isin = df[df["isin"].isna()]
    print(no_isin[["name", "asset_class"]].to_string() if not no_isin.empty else "  All assets have an ISIN")