"""Batch cache quarterly (10-Q) data for all companies with cached 10-K data.

Also caches the prior year's 10-K for YoY comparison.

Run:
  python -m sec_mcp.batch_quarterly                    # All cached tickers
  python -m sec_mcp.batch_quarterly --limit 100        # First 100
  python -m sec_mcp.batch_quarterly --ticker AAPL      # Single ticker
"""

from __future__ import annotations

import argparse
import logging
import time

from sec_mcp.config import get_config
from sec_mcp.financials import extract_financials
from sec_mcp import supabase_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_quarterly")


def get_cached_tickers() -> list[str]:
    """Get all tickers that have 10-K data cached."""
    client = supabase_cache._get_client()
    if not client:
        return []

    result = client.table("company_directory").select("ticker").eq("cached", True).execute()
    return [r["ticker"] for r in (result.data or []) if r.get("ticker")]


def cache_period(ticker: str, form_type: str, year: int | None = None) -> bool:
    """Extract and cache a specific form type/year for a ticker."""
    period_key = f"{form_type}|{'latest' if not year else year}"

    # Skip if already cached
    cached = supabase_cache.get_cached(ticker, "financials", period_key)
    if cached and isinstance(cached, dict) and cached.get("metrics"):
        return True  # Already cached

    try:
        data = extract_financials(
            ticker, year=year, form_type=form_type,
            include_statements=True, include_segments=True,
        )
        if data and data.get("metrics"):
            supabase_cache.set_cached(ticker, "financials", data, period_key)
            return True
    except Exception as exc:
        log.debug("  %s %s: %s", ticker, form_type, exc)

    return False


def run(limit: int = 0, ticker: str | None = None):
    """Cache 10-Q (latest) + prior 10-K for all cached tickers."""
    if not supabase_cache.is_available():
        log.error("Supabase not available")
        return

    if ticker:
        tickers = [ticker.upper()]
    else:
        tickers = get_cached_tickers()
        if limit > 0:
            tickers = tickers[:limit]

    log.info("Caching quarterly + prior annual for %d tickers", len(tickers))

    q_cached = 0
    q_failed = 0
    prior_cached = 0
    t0 = time.time()

    for i, tk in enumerate(tickers):
        prefix = f"[{i+1}/{len(tickers)}]"

        # 1. Latest 10-Q
        if cache_period(tk, "10-Q"):
            q_cached += 1
            log.info("%s %s 10-Q — cached", prefix, tk)
        else:
            # Try 6-K for FPIs
            if cache_period(tk, "6-K"):
                q_cached += 1
                log.info("%s %s 6-K (FPI quarterly) — cached", prefix, tk)
            else:
                q_failed += 1
                log.debug("%s %s 10-Q — no data", prefix, tk)

        # 2. Prior year 10-K (for YoY in screener)
        from datetime import datetime
        current_year = datetime.now().year
        if cache_period(tk, "10-K", year=current_year - 1):
            prior_cached += 1

        time.sleep(0.3)

        if (i + 1) % 50 == 0:
            log.info("  Progress: %d/%d — 10-Q: %d, prior 10-K: %d",
                     i + 1, len(tickers), q_cached, prior_cached)

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("QUARTERLY BATCH COMPLETE")
    log.info("  Tickers processed: %d", len(tickers))
    log.info("  10-Q/6-K cached: %d", q_cached)
    log.info("  10-Q failed: %d", q_failed)
    log.info("  Prior 10-K cached: %d", prior_cached)
    log.info("  Time: %.1f min", elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch cache quarterly data")
    parser.add_argument("--limit", type=int, default=0, help="Limit tickers to process")
    parser.add_argument("--ticker", type=str, help="Single ticker to cache")
    args = parser.parse_args()
    run(limit=args.limit, ticker=args.ticker)
