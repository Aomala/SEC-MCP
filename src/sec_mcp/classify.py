"""Canonical company → sector → industry classification (GICS-aligned).

Single source of truth for market-sector classification, replacing three older,
incompatible schemes (company_search._SIC_SECTORS 10-bucket, chat_app.SECTOR_UNIVERSE
curated, xbrl_mappings._SIC_RANGES 6-class IndustryClass). Used by the market-overview
sector tiles, company search, the chatbot/narrator knowledge base, and persisted per
company in the company_directory.

Two-layer resolution in `classify(sic_code, ticker)`:
  1. Curated ticker override (TICKER_TO_CLASS) — authoritative for large caps and for the
     GICS↔SIC mismatches that pure SIC can't get right (AMZN is SIC-retail but GICS
     Consumer Discretionary; GOOGL/META are SIC-software but GICS Communication Services).
  2. SIC-range base map (_SIC_CLASS) — GICS-correct Consumer Staples vs Discretionary and
     Communication Services splits; covers the long tail beyond the curated universe.
  3. Fallback ("Other"/"Unknown") — never returns None (the old map silently dropped
     unlisted SICs from the sector rollup).

Sector names match the 11 SPDR/GICS tiles in the app; each carries its sector-ETF ticker
so the frontend can match a tile unambiguously by ETF.
"""

from __future__ import annotations

from typing import NamedTuple

# ═══════════════════════════════════════════════════════════════════════════
#  The 11 GICS sectors → SPDR sector-ETF ticker (matches the Markets tab tiles)
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Consumer Discretionary": "XLY",
    "Financials": "XLF",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Energy": "XLE",
}
SECTORS: tuple[str, ...] = tuple(SECTOR_ETF.keys())

# ═══════════════════════════════════════════════════════════════════════════
#  Curated industries → parent sector (~30, reusing the SECTOR_UNIVERSE buckets)
# ═══════════════════════════════════════════════════════════════════════════

INDUSTRY_SECTOR: dict[str, str] = {
    # Technology
    "Semiconductors": "Technology",
    "Software": "Technology",
    "Cybersecurity": "Technology",
    "IT Services": "Technology",
    "Hardware & Equipment": "Technology",
    "Consumer Electronics": "Technology",
    # Communication Services
    "Telecom": "Communication Services",
    "Media & Entertainment": "Communication Services",
    "Interactive Media": "Communication Services",
    # Consumer Discretionary
    "Broadline Retail": "Consumer Discretionary",
    "Specialty Retail": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Automobiles": "Consumer Discretionary",
    "Apparel & Luxury": "Consumer Discretionary",
    "Hotels & Leisure": "Consumer Discretionary",
    # Consumer Staples
    "Food & Beverage": "Consumer Staples",
    "Household & Personal Products": "Consumer Staples",
    "Staples Retail": "Consumer Staples",
    # Financials
    "Banks": "Financials",
    "Regional Banks": "Financials",
    "Insurance": "Financials",
    "Fintech & Payments": "Financials",
    "Capital Markets": "Financials",
    "Crypto & Digital Assets": "Financials",
    # Health Care
    "Pharmaceuticals": "Health Care",
    "Biotechnology": "Health Care",
    "Medical Devices": "Health Care",
    "Healthcare Services": "Health Care",
    # Industrials
    "Aerospace & Defense": "Industrials",
    "Machinery & Equipment": "Industrials",
    "Airlines": "Industrials",
    "Transportation & Logistics": "Industrials",
    "Construction & Engineering": "Industrials",
    # Real Estate
    "REITs": "Real Estate",
    "Real Estate Services": "Real Estate",
    # Materials
    "Metals & Mining": "Materials",
    "Chemicals": "Materials",
    # Energy
    "Oil & Gas Majors": "Energy",
    "Oil & Gas E&P": "Energy",
    "Oil & Gas Services": "Energy",
    # Utilities
    "Utilities": "Utilities",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Curated industry → tickers (the classification knowledge base). Derived from
#  chat_app.SECTOR_UNIVERSE with GICS corrections applied per ticker below.
# ═══════════════════════════════════════════════════════════════════════════

_INDUSTRY_TICKERS: dict[str, list[str]] = {
    "Consumer Electronics": ["AAPL"],
    "Software": [
        "MSFT", "ORCL", "SAP", "CRM", "ADBE", "IBM", "INTU", "NOW", "WDAY",
        "SNPS", "CDNS", "ANSS", "PLTR", "TEAM", "HUBS", "DDOG", "MDB",
        "SNOW", "ZS", "NET", "VEEV", "BILL", "TTD", "ESTC", "DOCN",
    ],
    "Semiconductors": [
        "NVDA", "TSM", "AVGO", "ASML", "AMD", "QCOM", "TXN", "INTC",
        "AMAT", "LRCX", "KLAC", "MRVL", "ADI", "NXPI", "MU", "ON",
        "MCHP", "SWKS", "MPWR", "TER", "ENTG", "WOLF",
    ],
    "Cybersecurity": ["PANW", "CRWD", "FTNT", "OKTA", "S", "QLYS", "TENB", "RPD"],
    "IT Services": ["ACN", "CSCO", "HPE", "HPQ", "DELL", "CDW", "LDOS", "SAIC"],
    "Interactive Media": ["GOOG", "GOOGL", "META", "NFLX", "SPOT", "ROKU", "PINS", "SNAP"],
    "Media & Entertainment": [
        "DIS", "WBD", "PARA", "CMCSA", "FOX", "FOXA", "NWSA", "LYV", "IMAX", "MSGS",
    ],
    "Telecom": ["T", "VZ", "TMUS", "CHTR", "LUMN"],
    "Broadline Retail": ["AMZN", "TGT", "EBAY", "ETSY", "W", "CHWY"],
    "Specialty Retail": [
        "HD", "LOW", "TJX", "ROST", "BURL", "ULTA", "BBY",
        "FIVE", "ORLY", "AZO", "AAP", "WSM", "RH",
    ],
    "Restaurants": ["MCD", "SBUX", "CMG", "YUM", "DPZ", "QSR", "DINE", "SHAK", "WING", "CAVA"],
    "Automobiles": ["TSLA", "TM", "GM", "F", "STLA", "HMC", "RIVN", "LCID", "NIO", "LI", "XPEV"],
    "Staples Retail": ["WMT", "COST", "KR", "DG", "DLTR", "BJ"],
    "Food & Beverage": [
        "KO", "PEP", "MDLZ", "KHC", "GIS", "SJM", "HSY", "CPB",
        "CAG", "KDP", "MNST", "STZ", "TAP", "K",
    ],
    "Household & Personal Products": ["PG", "CL", "CLX", "KMB", "UL", "EL", "CHD", "KVUE"],
    "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Regional Banks": [
        "USB", "PNC", "TFC", "BK", "STT", "FITB",
        "KEY", "MTB", "HBAN", "RF", "CFG", "ZION", "CMA",
    ],
    "Capital Markets": ["SCHW", "BLK", "SPGI", "MCO", "ICE", "CME", "MSCI", "BX", "KKR", "APO"],
    "Insurance": [
        "BRK-B", "BRK-A", "ALL", "PGR", "MET", "AIG", "PRU", "AFL",
        "TRV", "HIG", "CB", "CINF", "GL", "RGA", "EG", "WRB",
    ],
    "Fintech & Payments": [
        "V", "MA", "PYPL", "AXP", "SQ", "FIS", "FI", "GPN",
        "AFRM", "TOST", "FOUR", "RPAY", "COF", "DFS", "SYF", "ALLY",
    ],
    "Crypto & Digital Assets": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "WULF", "CIFR"],
    "Pharmaceuticals": [
        "LLY", "JNJ", "ABBV", "MRK", "PFE", "AZN", "NVO", "BMY",
        "AMGN", "GILD", "SNY", "GSK", "TAK",
    ],
    "Biotechnology": [
        "MRNA", "BNTX", "BIIB", "ALNY", "BMRN", "INCY", "REGN", "VRTX",
        "IONS", "PCVX", "EXEL", "RARE", "HALO",
    ],
    "Healthcare Services": [
        "UNH", "ELV", "CI", "HUM", "CNC", "MOH", "CVS", "WBA", "HCA", "THC", "UHS",
    ],
    "Medical Devices": [
        "ABT", "MDT", "SYK", "BSX", "ISRG", "EW", "ZBH",
        "DXCM", "ALGN", "HOLX", "BAX", "BDX",
    ],
    "Oil & Gas Majors": ["XOM", "CVX", "SHEL", "BP", "TTE", "COP", "EOG"],
    "Oil & Gas E&P": ["DVN", "FANG", "OXY", "HES", "CTRA", "MRO", "APA"],
    "Oil & Gas Services": ["SLB", "HAL", "BKR", "MPC", "VLO", "PSX"],
    "Utilities": [
        "NEE", "DUK", "SO", "AEP", "D", "SRE", "EXC", "XEL",
        "WEC", "ED", "ES", "DTE", "PPL", "FE", "CMS", "AES", "PEG", "EIX",
    ],
    "Aerospace & Defense": ["BA", "LMT", "RTX", "NOC", "GD", "LHX", "HII", "TXT", "HWM"],
    "Machinery & Equipment": [
        "GE", "HON", "MMM", "CAT", "DE", "EMR", "ETN", "ROK",
        "ITW", "PH", "CMI", "DOV", "IR", "AME", "PCAR",
    ],
    "Airlines": ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "SAVE", "HA"],
    "Transportation & Logistics": ["UPS", "FDX", "XPO", "JBHT", "ODFL", "CHRW", "EXPD", "SAIA", "UNP", "CSX", "NSC"],
    "REITs": [
        "PLD", "AMT", "EQIX", "CCI", "SPG", "O", "DLR", "PSA",
        "WELL", "AVB", "EQR", "VTR", "ARE", "SUI", "MAA", "WPC",
        "SBAC", "IRM", "VICI", "INVH", "GLPI",
    ],
    "Metals & Mining": [
        "NEM", "GOLD", "AEM", "FNV", "WPM", "FCX", "BHP", "RIO",
        "NUE", "STLD", "CLF", "X", "AA",
    ],
    "Chemicals": ["LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "NEM", "CTVA", "LYB", "ALB", "FMC"],
}

# Build reverse map ticker → (sector, industry). Later entries win, so order the
# rare dual-listings intentionally (none currently conflict).
_TICKER_TO_CLASS: dict[str, tuple[str, str]] = {}
for _industry, _tickers in _INDUSTRY_TICKERS.items():
    _sector = INDUSTRY_SECTOR[_industry]
    for _t in _tickers:
        _TICKER_TO_CLASS[_t] = (_sector, _industry)


def _norm_ticker(t: str | None) -> str:
    if not t:
        return ""
    return t.strip().upper().replace(".", "-")


# ═══════════════════════════════════════════════════════════════════════════
#  SIC-range base map → (sector, industry). GICS-correct: Consumer Staples vs
#  Discretionary split, Communication Services split out. First match wins.
# ═══════════════════════════════════════════════════════════════════════════

_SIC_CLASS: list[tuple[int, int, str, str]] = [
    # Health Care
    (2833, 2836, "Health Care", "Pharmaceuticals"),
    (2830, 2831, "Health Care", "Pharmaceuticals"),
    (3826, 3829, "Health Care", "Medical Devices"),
    (3840, 3851, "Health Care", "Medical Devices"),
    (8000, 8099, "Health Care", "Healthcare Services"),
    (5912, 5912, "Consumer Staples", "Staples Retail"),   # drug stores → staples
    # Technology
    (3570, 3579, "Technology", "Hardware & Equipment"),
    (3670, 3674, "Technology", "Semiconductors"),
    (3661, 3669, "Technology", "Hardware & Equipment"),
    (3675, 3699, "Technology", "Hardware & Equipment"),
    (7370, 7372, "Technology", "Software"),
    (7373, 7379, "Technology", "IT Services"),
    # Communication Services (telecom + media/publishing/entertainment)
    (4800, 4899, "Communication Services", "Telecom"),
    (2700, 2799, "Communication Services", "Media & Entertainment"),
    (7800, 7841, "Communication Services", "Media & Entertainment"),
    (7900, 7999, "Communication Services", "Media & Entertainment"),
    # Utilities
    (4900, 4999, "Utilities", "Utilities"),
    # Energy
    (1300, 1399, "Energy", "Oil & Gas E&P"),
    (2900, 2999, "Energy", "Oil & Gas Services"),
    # Financials
    (6000, 6199, "Financials", "Banks"),
    (6200, 6299, "Financials", "Capital Markets"),
    (6300, 6411, "Financials", "Insurance"),
    (6412, 6499, "Financials", "Insurance"),
    (6500, 6599, "Real Estate", "REITs"),
    (6798, 6798, "Real Estate", "REITs"),
    (6700, 6799, "Financials", "Capital Markets"),
    (6600, 6699, "Financials", "Capital Markets"),
    # Consumer Staples (food/bev/tobacco/household + staples retail)
    (2000, 2079, "Consumer Staples", "Food & Beverage"),
    (2080, 2085, "Consumer Staples", "Food & Beverage"),
    (2086, 2199, "Consumer Staples", "Food & Beverage"),
    (2840, 2844, "Consumer Staples", "Household & Personal Products"),
    (5140, 5149, "Consumer Staples", "Food & Beverage"),
    (5400, 5412, "Consumer Staples", "Staples Retail"),
    # Materials (chemicals, metals, mining, agriculture)
    (2800, 2829, "Materials", "Chemicals"),
    (2845, 2899, "Materials", "Chemicals"),
    (1000, 1299, "Materials", "Metals & Mining"),
    (1400, 1499, "Materials", "Metals & Mining"),
    (3300, 3399, "Materials", "Metals & Mining"),
    (100, 999, "Materials", "Metals & Mining"),
    # Consumer Discretionary (autos, apparel, specialty retail, restaurants, leisure)
    (2300, 2399, "Consumer Discretionary", "Apparel & Luxury"),
    (3100, 3199, "Consumer Discretionary", "Apparel & Luxury"),
    (3700, 3716, "Consumer Discretionary", "Automobiles"),
    (2500, 2599, "Consumer Discretionary", "Specialty Retail"),
    (3630, 3639, "Consumer Discretionary", "Specialty Retail"),
    (5800, 5899, "Consumer Discretionary", "Restaurants"),   # before broad retail range
    (5200, 5399, "Consumer Discretionary", "Specialty Retail"),
    (5413, 5999, "Consumer Discretionary", "Specialty Retail"),
    (7000, 7299, "Consumer Discretionary", "Hotels & Leisure"),
    (7500, 7699, "Consumer Discretionary", "Specialty Retail"),
    # Industrials (construction, machinery, transport, defense)
    (1500, 1799, "Industrials", "Construction & Engineering"),
    (3400, 3599, "Industrials", "Machinery & Equipment"),
    (3710, 3799, "Industrials", "Aerospace & Defense"),
    (3800, 3825, "Industrials", "Machinery & Equipment"),
    (4000, 4599, "Industrials", "Transportation & Logistics"),
    (4600, 4799, "Industrials", "Transportation & Logistics"),
    (2200, 2299, "Industrials", "Machinery & Equipment"),
    (2400, 2499, "Industrials", "Machinery & Equipment"),
    (2600, 2699, "Materials", "Chemicals"),                # paper/pulp → materials
    (3000, 3099, "Materials", "Chemicals"),                # rubber/plastics → materials
    (3200, 3299, "Materials", "Metals & Mining"),          # stone/clay/glass
    (3900, 3999, "Industrials", "Machinery & Equipment"),
]


class Classification(NamedTuple):
    sector: str
    industry: str
    source: str   # "ticker" | "sic" | "fallback"


def _sic_int(sic_code: str | int | None) -> int | None:
    if sic_code is None:
        return None
    try:
        return int(str(sic_code).strip())
    except (ValueError, TypeError):
        return None


def classify(sic_code: str | int | None = None,
             ticker: str | None = None) -> Classification:
    """Classify a company into a GICS sector + curated industry.

    Ticker override wins (handles GICS↔SIC mismatches and large caps); otherwise
    the SIC-range base map; otherwise ("Other", "Unknown"). Never returns None.
    """
    tk = _norm_ticker(ticker)
    if tk and tk in _TICKER_TO_CLASS:
        sector, industry = _TICKER_TO_CLASS[tk]
        return Classification(sector, industry, "ticker")

    sic = _sic_int(sic_code)
    if sic is not None:
        for lo, hi, sector, industry in _SIC_CLASS:
            if lo <= sic <= hi:
                return Classification(sector, industry, "sic")

    return Classification("Other", "Unknown", "fallback")


def sector_for(sic_code: str | int | None = None,
               ticker: str | None = None) -> str:
    """Convenience: just the sector name."""
    return classify(sic_code, ticker).sector


def sector_etf(sector: str) -> str | None:
    """SPDR sector-ETF ticker for a sector name (for matching UI tiles)."""
    return SECTOR_ETF.get(sector)
