"""ETF/fund profile layer — seed registry + Polygon enrichment.

SEC XBRL has no fund-level data (ETFs don't file 10-K income statements the
app can use), so ETF profiles are assembled from two sources:

  1. ETF_SEED — curated static facts for the majors (issuer, expense ratio,
     asset class, inception, holdings count, peer list). Expense ratios are
     stable published facts; tagged `source: "seed"`.
  2. Polygon /v3/reference/tickers/{t} — live name, market_cap (≈ AUM for an
     ETF), share counts, and the `type` field used for ETF detection.

Real AUM/holdings from N-PORT ingestion is a planned follow-up; until then
`aumSource` tells the consumer which approximation it is getting.

Expense ratios are in PERCENT units (0.0945 means 0.0945%/yr).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# Polygon `type` values that mean "fund-like, not an operating company".
FUND_TYPES = {"ETF", "ETN", "ETV", "ETS", "FUND", "UNIT"}

_EQUITY_DEFAULT_PEERS = ["VOO", "IVV", "VTI", "QQQ", "DIA"]
_FIXED_INCOME_DEFAULT_PEERS = ["BND", "AGG", "TLT", "LQD", "HYG"]
_COMMODITY_DEFAULT_PEERS = ["GLD", "SLV"]

ETF_SEED: dict[str, dict[str, Any]] = {
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "issuer": "State Street Global Advisors",
            "expense_ratio": 0.0945, "asset_class": "Equity", "inception_date": "1993-01-22",
            "holdings_count": 503, "peers": ["VOO", "IVV", "VTI", "QQQ", "DIA"]},
    "VOO": {"name": "Vanguard S&P 500 ETF", "issuer": "Vanguard",
            "expense_ratio": 0.03, "asset_class": "Equity", "inception_date": "2010-09-07",
            "holdings_count": 505, "peers": ["SPY", "IVV", "VTI", "VUG", "VTV"]},
    "IVV": {"name": "iShares Core S&P 500 ETF", "issuer": "BlackRock",
            "expense_ratio": 0.03, "asset_class": "Equity", "inception_date": "2000-05-15",
            "holdings_count": 503, "peers": ["SPY", "VOO", "VTI", "QQQ", "DIA"]},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "issuer": "Vanguard",
            "expense_ratio": 0.03, "asset_class": "Equity", "inception_date": "2001-05-24",
            "holdings_count": 3600, "peers": ["SPY", "VOO", "IVV", "IWM", "SCHD"]},
    "QQQ": {"name": "Invesco QQQ Trust", "issuer": "Invesco",
            "expense_ratio": 0.20, "asset_class": "Equity", "inception_date": "1999-03-10",
            "holdings_count": 100, "peers": ["SPY", "VOO", "VUG", "VTI", "IVV"]},
    "DIA": {"name": "SPDR Dow Jones Industrial Average ETF Trust", "issuer": "State Street Global Advisors",
            "expense_ratio": 0.16, "asset_class": "Equity", "inception_date": "1998-01-14",
            "holdings_count": 30, "peers": ["SPY", "VOO", "IVV", "VTV", "VTI"]},
    "IWM": {"name": "iShares Russell 2000 ETF", "issuer": "BlackRock",
            "expense_ratio": 0.19, "asset_class": "Equity", "inception_date": "2000-05-22",
            "holdings_count": 1950, "peers": ["VTI", "SPY", "SCHD", "VTV", "VOO"]},
    "EFA": {"name": "iShares MSCI EAFE ETF", "issuer": "BlackRock",
            "expense_ratio": 0.32, "asset_class": "Equity", "inception_date": "2001-08-14",
            "holdings_count": 700, "peers": ["EEM", "VTI", "SPY", "VOO", "IVV"]},
    "EEM": {"name": "iShares MSCI Emerging Markets ETF", "issuer": "BlackRock",
            "expense_ratio": 0.70, "asset_class": "Equity", "inception_date": "2003-04-07",
            "holdings_count": 1200, "peers": ["EFA", "VTI", "SPY", "VOO", "IVV"]},
    "VUG": {"name": "Vanguard Growth ETF", "issuer": "Vanguard",
            "expense_ratio": 0.04, "asset_class": "Equity", "inception_date": "2004-01-26",
            "holdings_count": 180, "peers": ["QQQ", "VTV", "VOO", "SPY", "VTI"]},
    "VTV": {"name": "Vanguard Value ETF", "issuer": "Vanguard",
            "expense_ratio": 0.04, "asset_class": "Equity", "inception_date": "2004-01-26",
            "holdings_count": 340, "peers": ["VUG", "SCHD", "VOO", "SPY", "DIA"]},
    "SCHD": {"name": "Schwab U.S. Dividend Equity ETF", "issuer": "Charles Schwab",
             "expense_ratio": 0.06, "asset_class": "Equity", "inception_date": "2011-10-20",
             "holdings_count": 100, "peers": ["VTV", "VOO", "DIA", "VTI", "SPY"]},
    "AGG": {"name": "iShares Core U.S. Aggregate Bond ETF", "issuer": "BlackRock",
            "expense_ratio": 0.03, "asset_class": "Fixed Income", "inception_date": "2003-09-22",
            "holdings_count": 12000, "peers": ["BND", "TLT", "LQD", "HYG"]},
    "BND": {"name": "Vanguard Total Bond Market ETF", "issuer": "Vanguard",
            "expense_ratio": 0.03, "asset_class": "Fixed Income", "inception_date": "2007-04-03",
            "holdings_count": 11300, "peers": ["AGG", "TLT", "LQD", "HYG"]},
    "TLT": {"name": "iShares 20+ Year Treasury Bond ETF", "issuer": "BlackRock",
            "expense_ratio": 0.15, "asset_class": "Fixed Income", "inception_date": "2002-07-22",
            "holdings_count": 45, "peers": ["AGG", "BND", "LQD", "HYG"]},
    "LQD": {"name": "iShares iBoxx $ Investment Grade Corporate Bond ETF", "issuer": "BlackRock",
            "expense_ratio": 0.14, "asset_class": "Fixed Income", "inception_date": "2002-07-22",
            "holdings_count": 2800, "peers": ["AGG", "BND", "TLT", "HYG"]},
    "HYG": {"name": "iShares iBoxx $ High Yield Corporate Bond ETF", "issuer": "BlackRock",
            "expense_ratio": 0.49, "asset_class": "Fixed Income", "inception_date": "2007-04-04",
            "holdings_count": 1200, "peers": ["AGG", "BND", "TLT", "LQD"]},
    "GLD": {"name": "SPDR Gold Shares", "issuer": "State Street Global Advisors",
            "expense_ratio": 0.40, "asset_class": "Commodity", "inception_date": "2004-11-18",
            "holdings_count": 1, "peers": ["SLV", "IAU"]},
    "SLV": {"name": "iShares Silver Trust", "issuer": "BlackRock",
            "expense_ratio": 0.50, "asset_class": "Commodity", "inception_date": "2006-04-21",
            "holdings_count": 1, "peers": ["GLD", "IAU"]},
    "VNQ": {"name": "Vanguard Real Estate ETF", "issuer": "Vanguard",
            "expense_ratio": 0.13, "asset_class": "Real Estate", "inception_date": "2004-09-23",
            "holdings_count": 160, "peers": ["SPY", "VTI", "VOO", "SCHD"]},
}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and v == v else None


def default_peers(asset_class: str | None, exclude: str) -> list[str]:
    """Curated fallback peers by asset class for non-seeded ETFs."""
    if asset_class == "Fixed Income":
        pool = _FIXED_INCOME_DEFAULT_PEERS
    elif asset_class == "Commodity":
        pool = _COMMODITY_DEFAULT_PEERS
    else:
        pool = _EQUITY_DEFAULT_PEERS
    return [p for p in pool if p != exclude.upper()][:5]


def merge_etf_profile(ticker: str, details: dict | None, price: float | None = None) -> dict | None:
    """Pure merge: Polygon ticker-details + ETF_SEED → one profile dict.

    Returns None only when the ticker is neither seeded nor fund-typed on
    Polygon (i.e. not an ETF as far as we can tell).
    `details` is the Polygon /v3/reference/tickers `results` object (or None).
    """
    tk = ticker.upper()
    seed = ETF_SEED.get(tk)
    ptype = str((details or {}).get("type", "")).upper()
    is_fund = bool(seed) or ptype in FUND_TYPES
    if not is_fund:
        return None

    aum = _num((details or {}).get("market_cap"))
    aum_source = "polygon_market_cap" if aum else None
    if aum is None:
        shares = _num((details or {}).get("weighted_shares_outstanding")) or _num(
            (details or {}).get("share_class_shares_outstanding"))
        if shares and price:
            aum = shares * price
            aum_source = "shares_x_price"

    asset_class = (seed or {}).get("asset_class")
    peers = list((seed or {}).get("peers") or default_peers(asset_class, tk))

    return {
        "ticker": tk,
        "name": (details or {}).get("name") or (seed or {}).get("name") or tk,
        "issuer": (seed or {}).get("issuer"),
        "aum": aum,
        "aumSource": aum_source,
        "expenseRatio": (seed or {}).get("expense_ratio"),  # percent units
        "assetClass": asset_class or "Equity",
        "inceptionDate": (seed or {}).get("inception_date") or (details or {}).get("list_date"),
        "holdingsCount": (seed or {}).get("holdings_count"),
        "peers": peers,
        "profileSource": "seed+polygon" if (seed and details) else ("seed" if seed else "polygon"),
    }


def get_etf_profile(ticker: str) -> dict:
    """Full /api/etf/{ticker} response: detection + merged profile."""
    from sec_mcp import polygon_client

    tk = ticker.upper().strip()
    details = None
    try:
        details = polygon_client.get_ticker_details(tk)
    except Exception as exc:  # never let enrichment kill detection
        log.warning("Polygon details failed for %s: %s", tk, exc)

    ptype = str((details or {}).get("type", "")).upper()
    is_fund = tk in ETF_SEED or ptype in FUND_TYPES
    market_cap_missing = _num((details or {}).get("market_cap")) is None

    price = None
    if is_fund and market_cap_missing:
        try:
            from sec_mcp.core.realtime_price import get_realtime_price
            snap = get_realtime_price(tk)
            price = _num(snap.get("price"))
        except Exception as exc:
            log.debug("Price snapshot failed for %s: %s", tk, exc)

    profile = merge_etf_profile(tk, details, price=price)
    meta = {
        "source": "seed+polygon" if details is not None else "seed",
        "asOf": datetime.now(timezone.utc).isoformat(),
    }
    if profile is None:
        return {"isEtf": False, "ticker": tk, "meta": meta}
    return {"isEtf": True, "ticker": tk, "profile": profile, "meta": meta}


def get_etf_comps(tickers: list[str]) -> dict:
    """Profiles for a target + peer list in one call (POST /api/etf/comps)."""
    clean: list[str] = []
    for t in tickers:
        tk = str(t).upper().strip()
        if tk and tk not in clean:
            clean.append(tk)
    results = [get_etf_profile(tk) for tk in clean[:12]]
    return {
        "tool": "etf_comps",
        "results": results,
        "meta": {"asOf": datetime.now(timezone.utc).isoformat(), "count": len(results)},
    }
