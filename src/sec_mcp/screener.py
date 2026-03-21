"""Stock screener — filter companies from SECTOR_UNIVERSE using cached Supabase data.

Screens 300+ tickers by financial metrics without hitting SEC EDGAR.
Only uses cached data from Supabase for speed.

Usage:
    from sec_mcp.screener import screen, get_available_sectors, get_cached_ticker_count

    results = screen({"gross_margin_min": 0.3, "revenue_min": 1e9, "sector": "semiconductors"})
"""

from __future__ import annotations

import logging
from typing import Any

from sec_mcp.chat_app import SECTOR_UNIVERSE, _TICKER_TO_SECTOR

log = logging.getLogger(__name__)


def get_available_sectors() -> list[str]:
    """Return list of sector names from SECTOR_UNIVERSE."""
    return sorted(SECTOR_UNIVERSE.keys())


def get_cached_ticker_count() -> int:
    """Count how many tickers have cached data in Supabase."""
    from sec_mcp import supabase_cache

    if not supabase_cache.is_available():
        return 0

    count = 0
    for tickers in SECTOR_UNIVERSE.values():
        for ticker in tickers:
            cached = supabase_cache.get_cached(ticker, "financials")
            if cached and isinstance(cached, dict) and cached.get("metrics"):
                count += 1
    return count


def _extract_screening_metrics(ticker: str, data: dict, sector: str) -> dict | None:
    """Extract standardized screening metrics from cached financial data."""
    metrics = data.get("metrics", {})
    ratios = data.get("ratios", {})

    if not metrics:
        return None

    revenue = metrics.get("revenue") or metrics.get("total_revenue")
    net_income = metrics.get("net_income")
    gross_profit = metrics.get("gross_profit")
    total_assets = metrics.get("total_assets")
    total_liabilities = metrics.get("total_liabilities")
    total_equity = metrics.get("stockholders_equity") or metrics.get("total_equity")
    eps = metrics.get("eps") or metrics.get("earnings_per_share") or metrics.get("diluted_eps")

    # Calculate margins
    gross_margin = None
    if gross_profit and revenue and revenue != 0:
        gross_margin = gross_profit / revenue
    elif ratios.get("gross_margin") is not None:
        gross_margin = ratios["gross_margin"]

    net_margin = None
    if net_income and revenue and revenue != 0:
        net_margin = net_income / revenue
    elif ratios.get("net_margin") is not None:
        net_margin = ratios["net_margin"]

    # D/E ratio
    de_ratio = None
    if total_liabilities and total_equity and total_equity != 0:
        de_ratio = total_liabilities / total_equity
    elif ratios.get("debt_to_equity") is not None:
        de_ratio = ratios["debt_to_equity"]

    return {
        "ticker": ticker,
        "company": data.get("company_name", ticker),
        "sector": sector,
        "revenue": revenue,
        "net_income": net_income,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "eps": eps,
        "de_ratio": de_ratio,
    }


def screen(filters: dict, limit: int = 50) -> list[dict]:
    """Screen companies by financial metrics.

    filters is a dict like:
    {
        "gross_margin_min": 0.3,      # > 30%
        "net_margin_min": 0.1,        # > 10%
        "revenue_min": 1e9,           # > $1B
        "de_ratio_max": 2.0,          # < 2x
        "sector": "semiconductors",   # optional sector filter
    }

    Process:
    1. Get all tickers from SECTOR_UNIVERSE
    2. For each, try Supabase cache first (get_cached(ticker, "financials", ...))
    3. If not cached, skip (don't hit SEC — too slow for screening)
    4. Apply filters to metrics
    5. Return matching companies sorted by revenue desc

    Returns list of dicts with: ticker, company, sector, revenue, net_income,
    gross_margin, net_margin, eps, de_ratio.
    """
    from sec_mcp import supabase_cache

    sector_filter = filters.get("sector", "").lower().strip()
    matches: list[dict] = []

    # Determine which sectors to scan
    if sector_filter:
        sectors_to_scan = {
            k: v for k, v in SECTOR_UNIVERSE.items()
            if sector_filter in k.lower()
        }
    else:
        sectors_to_scan = SECTOR_UNIVERSE

    for sector_name, tickers in sectors_to_scan.items():
        for ticker in tickers:
            # Try Supabase cache — skip if not cached
            cached = supabase_cache.get_cached(ticker, "financials")
            if not cached or not isinstance(cached, dict) or not cached.get("metrics"):
                # Also try with common period keys
                for period_key in ("10-K|latest", "10-K|None", "annual"):
                    cached = supabase_cache.get_cached(ticker, "financials", period_key)
                    if cached and isinstance(cached, dict) and cached.get("metrics"):
                        break
                else:
                    continue

            row = _extract_screening_metrics(ticker, cached, sector_name)
            if not row:
                continue

            # Apply filters
            if not _passes_filters(row, filters):
                continue

            matches.append(row)

    # Sort by revenue descending (None values at the end)
    matches.sort(key=lambda x: x.get("revenue") or 0, reverse=True)

    return matches[:limit]


def _passes_filters(row: dict, filters: dict) -> bool:
    """Check if a company row passes all filter criteria."""
    # Revenue minimum
    rev_min = filters.get("revenue_min")
    if rev_min is not None:
        if row.get("revenue") is None or row["revenue"] < rev_min:
            return False

    # Revenue maximum
    rev_max = filters.get("revenue_max")
    if rev_max is not None:
        if row.get("revenue") is None or row["revenue"] > rev_max:
            return False

    # Gross margin minimum
    gm_min = filters.get("gross_margin_min")
    if gm_min is not None:
        if row.get("gross_margin") is None or row["gross_margin"] < gm_min:
            return False

    # Gross margin maximum
    gm_max = filters.get("gross_margin_max")
    if gm_max is not None:
        if row.get("gross_margin") is None or row["gross_margin"] > gm_max:
            return False

    # Net margin minimum
    nm_min = filters.get("net_margin_min")
    if nm_min is not None:
        if row.get("net_margin") is None or row["net_margin"] < nm_min:
            return False

    # Net margin maximum
    nm_max = filters.get("net_margin_max")
    if nm_max is not None:
        if row.get("net_margin") is None or row["net_margin"] > nm_max:
            return False

    # D/E ratio maximum
    de_max = filters.get("de_ratio_max")
    if de_max is not None:
        if row.get("de_ratio") is None or row["de_ratio"] > de_max:
            return False

    # D/E ratio minimum
    de_min = filters.get("de_ratio_min")
    if de_min is not None:
        if row.get("de_ratio") is None or row["de_ratio"] < de_min:
            return False

    # EPS minimum
    eps_min = filters.get("eps_min")
    if eps_min is not None:
        if row.get("eps") is None or row["eps"] < eps_min:
            return False

    # Net income minimum
    ni_min = filters.get("net_income_min")
    if ni_min is not None:
        if row.get("net_income") is None or row["net_income"] < ni_min:
            return False

    return True
