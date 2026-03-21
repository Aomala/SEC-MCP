"""Batch cache job — extract and cache financials for all tickers in SECTOR_UNIVERSE.

Run:  python -m sec_mcp.batch_cache
      python -m sec_mcp.batch_cache --sector semiconductors
      python -m sec_mcp.batch_cache --resume

Respects SEC rate limit (8 req/sec) with 0.5s delay between tickers.
Saves to Supabase financial_cache table with 24h TTL.
Skips tickers that are already cached (use --force to override).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_cache")


def get_all_tickers(sector: str | None = None) -> list[tuple[str, str]]:
    """Return list of (ticker, sector) pairs."""
    from sec_mcp.chat_app import SECTOR_UNIVERSE
    pairs = []
    for sect, tickers in SECTOR_UNIVERSE.items():
        if sector and sect != sector:
            continue
        for tk in tickers:
            pairs.append((tk, sect))
    return pairs


def is_cached(ticker: str) -> bool:
    """Check if ticker has valid cached data in Supabase."""
    from sec_mcp import supabase_cache
    cached = supabase_cache.get_cached(ticker, "financials", "10-K|latest")
    return cached is not None and isinstance(cached, dict) and bool(cached.get("metrics"))


def extract_and_cache(ticker: str) -> dict | None:
    """Extract financials from SEC EDGAR and cache in Supabase."""
    from sec_mcp.financials import extract_financials
    from sec_mcp import supabase_cache

    try:
        data = extract_financials(
            ticker,
            include_statements=True,
            include_segments=True,
        )
        if data and data.get("metrics"):
            supabase_cache.set_cached(ticker, "financials", data, "10-K|latest")
            return data
        return None
    except Exception as exc:
        log.warning("  %s: extraction failed — %s", ticker, exc)
        return None


def run(sector: str | None = None, force: bool = False, delay: float = 0.5):
    """Run the batch cache job."""
    from sec_mcp import supabase_cache

    if not supabase_cache.is_available():
        log.error("Supabase not available — cannot cache. Check SUPABASE_URL and SUPABASE_KEY in .env")
        sys.exit(1)

    pairs = get_all_tickers(sector)
    total = len(pairs)
    log.info("Batch cache: %d tickers%s", total, f" (sector: {sector})" if sector else "")

    cached = 0
    extracted = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    for i, (tk, sect) in enumerate(pairs):
        prefix = f"[{i+1}/{total}]"

        # Skip if already cached (unless --force)
        if not force and is_cached(tk):
            cached += 1
            log.debug("%s %s — already cached, skipping", prefix, tk)
            continue

        log.info("%s %s (%s) — extracting...", prefix, tk, sect.replace("_", " "))

        data = extract_and_cache(tk)
        if data:
            rev = data.get("metrics", {}).get("revenue")
            company = data.get("company_name", "")
            log.info("%s %s — OK (%s, rev=%s)", prefix, tk, company[:30],
                     f"${rev/1e9:.1f}B" if rev and rev > 1e9 else f"${rev/1e6:.0f}M" if rev else "N/A")
            extracted += 1
        else:
            log.warning("%s %s — no data returned", prefix, tk)
            failed += 1

        # Rate limit: SEC allows 10 req/sec, we use 0.5s delay to be safe
        time.sleep(delay)

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("BATCH CACHE COMPLETE")
    log.info("  Total tickers: %d", total)
    log.info("  Already cached: %d", cached)
    log.info("  Freshly extracted: %d", extracted)
    log.info("  Failed: %d", failed)
    log.info("  Time: %.1f min", elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch cache SEC financials")
    parser.add_argument("--sector", help="Only cache a specific sector")
    parser.add_argument("--force", action="store_true", help="Re-cache even if already cached")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between tickers (seconds)")
    args = parser.parse_args()

    run(sector=args.sector, force=args.force, delay=args.delay)
