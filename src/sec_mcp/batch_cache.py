"""Batch cache job — warm financials, history, and TTM for the ticker universe.

Run:  python -m sec_mcp.batch_cache                       # curated sector universe
      python -m sec_mcp.batch_cache --universe top1000    # sectors ∪ top-1000 by revenue
      python -m sec_mcp.batch_cache --sector semiconductors
      python -m sec_mcp.batch_cache --artifacts annual,ttm
      python -m sec_mcp.batch_cache --refresh-older-than 6
      python -m sec_mcp.batch_cache --tickers-file my_list.txt

Per ticker (companyfacts frame stays hot across all four artifacts):
  annual     → financial_cache 'v2|10-K|latest'  (+ legacy '10-K|latest' dual-write)
  quarterly  → financial_cache 'v2|10-Q|latest'
  history    → sec_income_history_v2, 12 periods annual + quarter
  ttm        → ttm_metrics 'ttm' (derived from the quarters just fetched)

Writes go to Supabase financial_cache — the only cache that survives Railway
deploys (container disk is ephemeral). Run this LOCALLY, never as a Railway
daemon thread. Resumable: the per-artifact freshness skip means a crashed run
re-skips finished work. Respects the process-wide SEC 8 req/s limiter.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_cache")

ARTIFACTS = ("annual", "quarterly", "history", "ttm")


def get_all_tickers(sector: str | None = None) -> list[tuple[str, str]]:
    """Return list of (ticker, sector) pairs from the curated universe."""
    from sec_mcp.chat_app import SECTOR_UNIVERSE
    pairs = []
    for sect, tickers in SECTOR_UNIVERSE.items():
        if sector and sect != sector:
            continue
        for tk in tickers:
            pairs.append((tk, sect))
    return pairs


def get_top_revenue_tickers(limit: int = 1000) -> list[tuple[str, str]]:
    """Top-N company_directory tickers by revenue (populated by ingest_all)."""
    from sec_mcp import supabase_cache
    client = supabase_cache._get_client()
    if not client:
        return []
    try:
        rows = (
            client.table("company_directory")
            .select("ticker, sector, revenue")
            .not_.is_("ticker", "null")
            .not_.is_("revenue", "null")
            .order("revenue", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )
    except Exception as exc:
        log.warning("company_directory top-revenue query failed: %s", exc)
        return []
    return [(r["ticker"].upper(), r.get("sector") or "directory")
            for r in rows if r.get("ticker")]


def build_universe(universe: str, sector: str | None,
                   tickers_file: str | None) -> list[tuple[str, str]]:
    if tickers_file:
        with open(tickers_file) as fh:
            tks = [ln.strip().upper() for ln in fh if ln.strip() and not ln.startswith("#")]
        return [(tk, "file") for tk in tks]
    pairs = get_all_tickers(sector)
    if universe == "top1000" and not sector:
        seen = {t for t, _ in pairs}
        extra = 0
        for tk, sect in get_top_revenue_tickers(1000):
            if tk not in seen:
                seen.add(tk)
                pairs.append((tk, sect))
                extra += 1
        log.info("Universe top1000: %d curated + %d from company_directory",
                 len(pairs) - extra, extra)
        if extra == 0:
            log.warning("company_directory contributed 0 tickers — revenue column "
                        "may be unpopulated (run ingest_all.py enrichment)")
    return pairs


def artifact_fresh(ticker: str, artifact: str,
                   max_age_days: float | None) -> bool:
    """True when the artifact's cache entries are live and young enough.

    max_age_days=None → any unexpired entry counts (resume semantics);
    a number → entries older than that are re-extracted (weekly refresh);
    negative → never fresh (--force).
    """
    from sec_mcp import supabase_cache
    from sec_mcp.core.cache_keys import TTM_DATA_TYPE, TTM_PERIOD, financials_period_key

    def ok(age: float | None) -> bool:
        if age is None:
            return False
        return max_age_days is None or age <= max_age_days

    if max_age_days is not None and max_age_days < 0:
        return False
    if artifact == "history":
        return all(ok(supabase_cache.cached_age_days(ticker, "sec_income_history_v2", p))
                   for p in ("annual", "quarter"))
    if artifact == "ttm":
        from sec_mcp.core.cache_keys import QUARTER_METRICS_DATA_TYPE, QUARTER_METRICS_PERIOD
        return (ok(supabase_cache.cached_age_days(ticker, TTM_DATA_TYPE, TTM_PERIOD))
                and ok(supabase_cache.cached_age_days(
                    ticker, QUARTER_METRICS_DATA_TYPE, QUARTER_METRICS_PERIOD)))
    form = "10-K" if artifact == "annual" else "10-Q"
    return ok(supabase_cache.cached_age_days(
        ticker, "financials", financials_period_key(form)))


def cache_annual(ticker: str) -> dict | None:
    """Annual extraction → v2 key + legacy dual-write (transition only)."""
    from sec_mcp import supabase_cache
    from sec_mcp.core.cache_keys import financials_period_key, legacy_financials_period_key
    from sec_mcp.financials import extract_financials

    try:
        data = extract_financials(ticker, include_statements=True, include_segments=True)
    except Exception as exc:
        log.warning("  %s: annual extraction failed — %s", ticker, exc)
        return None
    if data and data.get("metrics"):
        supabase_cache.set_cached(ticker, "financials", data, financials_period_key("10-K"))
        # Dual-write the pre-versioning key so an already-deployed Railway
        # reader never goes cold. Remove one release after v2-everywhere.
        supabase_cache.set_cached(ticker, "financials", data,
                                  legacy_financials_period_key("10-K"))
        return data
    return None


def cache_quarterly(ticker: str) -> dict | None:
    from sec_mcp import supabase_cache
    from sec_mcp.core.cache_keys import financials_period_key
    from sec_mcp.financials import extract_financials

    try:
        data = extract_financials(ticker, form_type="10-Q",
                                  include_statements=True, include_segments=False)
    except Exception as exc:
        log.warning("  %s: quarterly extraction failed — %s", ticker, exc)
        return None
    if data and data.get("metrics"):
        supabase_cache.set_cached(ticker, "financials", data,
                                  financials_period_key("10-Q"),
                                  ttl=supabase_cache.QUARTERLY_FINANCIALS_TTL)
        return data
    return None


def prewarm_history(ticker: str) -> None:
    """Pre-compute full-depth (12-period) shaped history, annual + quarterly,
    into the same cache the /api/financials-history endpoint reads
    (sec_income_history_v2). This makes deep/older history instant: the
    marginal cost is small here — the ticker's companyfacts frame is already
    in memory from cache_annual — but it saves users a 10-60s cold
    extraction per (ticker, period)."""
    import time as _time

    from sec_mcp import supabase_cache
    from sec_mcp.financials import get_fmp_shaped_history

    for period in ("annual", "quarter"):
        try:
            full = get_fmp_shaped_history(ticker, period=period, limit=12)
            if full.get("income"):
                full["_cached_at"] = _time.time()  # SWR freshness marker
                supabase_cache.set_cached(ticker, "sec_income_history_v2", full, period)
        except Exception as exc:
            # Logged, never fatal — one bad period must not stop the sweep.
            log.warning("  %s: history pre-warm (%s) failed — %s", ticker, period, exc)


def cache_ttm(ticker: str) -> bool:
    """Derive + cache the TTM block AND the latest-standalone-quarter block
    from one quarterly-periods fetch. Runs after history so the quarterly
    accessions are disk-warm — near-zero extra EDGAR calls."""
    from sec_mcp import supabase_cache
    from sec_mcp.core.cache_keys import (
        QUARTER_METRICS_DATA_TYPE,
        QUARTER_METRICS_PERIOD,
        TTM_DATA_TYPE,
        TTM_PERIOD,
    )
    from sec_mcp.core.ttm import build_latest_quarter_metrics, build_ttm_metrics
    from sec_mcp.surface.fundamentals import _build_periods

    try:
        periods = _build_periods(ticker, "ttm", 8)
    except Exception as exc:
        log.info("  %s: no quarterly periods — %s", ticker, exc)
        return False

    q_block = build_latest_quarter_metrics(ticker, periods=periods)
    if not q_block.get("error"):
        supabase_cache.set_cached(ticker, QUARTER_METRICS_DATA_TYPE, q_block,
                                  QUARTER_METRICS_PERIOD)

    block = build_ttm_metrics(ticker, periods=periods)
    if block.get("error"):
        log.info("  %s: TTM unavailable — %s", ticker, block["error"])
        return False
    supabase_cache.set_cached(ticker, TTM_DATA_TYPE, block, TTM_PERIOD)
    return True


def run(
    sector: str | None = None,
    force: bool = False,
    delay: float = 0.5,
    universe: str = "sectors",
    artifacts: tuple[str, ...] = ARTIFACTS,
    refresh_older_than: float | None = None,
    tickers_file: str | None = None,
):
    """Run the batch cache job."""
    from sec_mcp import supabase_cache

    if not supabase_cache.is_available():
        log.error("Supabase not available — cannot cache. Check SUPABASE_URL and SUPABASE_KEY in .env")
        sys.exit(1)

    max_age = -1.0 if force else refresh_older_than
    pairs = build_universe(universe, sector, tickers_file)
    total = len(pairs)
    log.info("Batch cache: %d tickers, artifacts=%s%s", total, ",".join(artifacts),
             f" (sector: {sector})" if sector else "")

    done: dict[str, int] = {a: 0 for a in artifacts}
    skipped: dict[str, int] = {a: 0 for a in artifacts}
    failed: dict[str, int] = {a: 0 for a in artifacts}
    t0 = time.time()

    for i, (tk, sect) in enumerate(pairs):
        prefix = f"[{i+1}/{total}]"
        todo = [a for a in ARTIFACTS if a in artifacts and not artifact_fresh(tk, a, max_age)]
        for a in artifacts:
            if a not in todo:
                skipped[a] += 1
        if not todo:
            log.debug("%s %s — all artifacts fresh, skipping", prefix, tk)
            continue

        log.info("%s %s (%s) — %s...", prefix, tk, sect.replace("_", " "), "+".join(todo))
        annual_data = None
        for a in todo:
            ok = False
            if a == "annual":
                annual_data = cache_annual(tk)
                ok = annual_data is not None
            elif a == "quarterly":
                ok = cache_quarterly(tk) is not None
            elif a == "history":
                prewarm_history(tk)
                ok = True  # best-effort; failures logged inside
            elif a == "ttm":
                ok = cache_ttm(tk)
            done[a] += ok
            failed[a] += not ok

        if annual_data:
            rev = annual_data.get("metrics", {}).get("revenue")
            company = annual_data.get("company_name", "")
            log.info("%s %s — OK (%s, rev=%s)", prefix, tk, company[:30],
                     f"${rev/1e9:.1f}B" if rev and rev > 1e9 else f"${rev/1e6:.0f}M" if rev else "n/a")

        # The sec_client limiter (8 req/s) is the hard cap; this spacing just
        # keeps the sweep polite between tickers.
        time.sleep(delay)

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("BATCH CACHE COMPLETE — %d tickers in %.1f min", total, elapsed / 60)
    for a in artifacts:
        log.info("  %-9s done=%d skipped=%d failed=%d", a, done[a], skipped[a], failed[a])
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch cache SEC financials")
    parser.add_argument("--sector", help="Only cache a specific sector")
    parser.add_argument("--universe", choices=("sectors", "top1000"), default="sectors",
                        help="sectors = curated SECTOR_UNIVERSE; top1000 adds company_directory top-1000 by revenue")
    parser.add_argument("--artifacts", default=",".join(ARTIFACTS),
                        help=f"Comma list of {ARTIFACTS}")
    parser.add_argument("--refresh-older-than", type=float, default=None, metavar="DAYS",
                        help="Re-extract artifacts older than DAYS (default: skip any live entry)")
    parser.add_argument("--tickers-file", help="File with one ticker per line (overrides universe)")
    parser.add_argument("--force", action="store_true", help="Re-cache even if already cached")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between tickers (seconds)")
    args = parser.parse_args()

    arts = tuple(a.strip() for a in args.artifacts.split(",") if a.strip())
    bad = [a for a in arts if a not in ARTIFACTS]
    if bad:
        parser.error(f"Unknown artifacts: {bad} (valid: {ARTIFACTS})")

    run(sector=args.sector, force=args.force, delay=args.delay,
        universe=args.universe, artifacts=arts,
        refresh_older_than=args.refresh_older_than, tickers_file=args.tickers_file)
