"""Bulk delivery — the thing data buyers actually pay for.

Nobody buys a fundamentals API to fetch one ticker at a time; they buy coverage.
So the bulk path has one hard rule: **it must never trigger a cold extraction.**

A 1,000-ticker request where each miss costs 20-45s would run for hours, hammer
SEC EDGAR's 8 req/s budget, and starve the interactive API the whole time. So
bulk reads the warm cache ONLY. A ticker with no cached row is emitted as an
explicit `{"ticker": ..., "error": "not_cached"}` line rather than being
silently skipped — a buyer must be able to tell "we don't have it" from "the
stream ended early".

Rows come out of a single batched Supabase read, not one query per ticker.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

log = logging.getLogger(__name__)

# Supabase caps how much a single `in.()` filter can carry; chunk to stay well
# under any URL-length limit while keeping the round-trip count low.
_CHUNK = 100

_DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "AMZN", "NVDA",
]


def _client():
    from sec_mcp.supabase_cache import _get_client
    return _get_client()


def resolve_universe(name: str) -> list[str]:
    """Ticker list for a named universe. Never raises — unknown → empty."""
    name = (name or "").lower().strip()

    if name == "dow30":
        return list(_DOW30)

    if name == "sp500":
        try:
            from sec_mcp.surface.company_search import _load_sp500
            symbols = sorted(_load_sp500())
            if symbols:
                return symbols
        except Exception as exc:
            log.warning("sp500 universe unavailable: %s", exc)
        return []

    if name == "nasdaq100":
        # No curated NDX list in the repo; approximate from the covered set's
        # tech-heavy sectors rather than shipping a stale hardcoded list.
        try:
            from sec_mcp.chat_app import SECTOR_UNIVERSE
            tech = ("mega_tech", "semiconductors", "enterprise_software",
                    "cybersecurity", "it_services", "biotech")
            out: list[str] = []
            for sect in tech:
                out.extend(SECTOR_UNIVERSE.get(sect, []))
            return sorted(set(out))
        except Exception as exc:
            log.warning("nasdaq100 approximation failed: %s", exc)
            return []

    if name == "covered":
        return covered_tickers()

    return []


def covered_tickers(limit: int = 5000) -> list[str]:
    """Every ticker with a cached fundamentals row — our true coverage.

    Deliberately counts stale rows too: since stale-while-revalidate serves
    them instantly, they are real coverage, not dead weight.
    """
    client = _client()
    if not client:
        return []
    out: set[str] = set()
    page_size = 1000
    offset = 0
    try:
        # PostgREST silently truncates at 1000 rows, so page explicitly.
        while offset < limit:
            res = (
                client.table("financial_cache")
                .select("ticker")
                .eq("data_type", "financials")
                .order("ticker")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            rows = res.data or []
            if not rows:
                break
            out.update(r["ticker"] for r in rows if r.get("ticker"))
            if len(rows) < page_size:
                break
            offset += page_size
    except Exception as exc:
        log.warning("covered_tickers failed: %s", exc)
    return sorted(out)


def _fetch_cached_rows(symbols: list[str], data_type: str,
                       period: str) -> dict[str, dict]:
    """Batched cache read: {TICKER: data}. One query per _CHUNK tickers."""
    client = _client()
    if not client:
        return {}
    found: dict[str, dict] = {}
    for i in range(0, len(symbols), _CHUNK):
        chunk = symbols[i:i + _CHUNK]
        keys = [f"{t}|{data_type}|{period}||" for t in chunk]
        try:
            res = (
                client.table("financial_cache")
                .select("ticker, data")
                .in_("cache_key", keys)
                .execute()
            )
            for row in res.data or []:
                if row.get("ticker") and row.get("data"):
                    found[row["ticker"].upper()] = row["data"]
        except Exception as exc:
            log.warning("bulk chunk read failed (%d tickers): %s", len(chunk), exc)
    return found


def iter_bulk_fundamentals(symbols: list[str], period: str = "annual") -> Iterator[dict]:
    """Yield one row per requested ticker, warm cache only.

    Emits in the order requested so a consumer can reconcile against its input
    list without sorting.
    """
    from sec_mcp.core.cache_keys import (
        QUARTER_METRICS_DATA_TYPE,
        QUARTER_METRICS_PERIOD,
        TTM_DATA_TYPE,
        TTM_PERIOD,
        financials_period_key,
    )
    from sec_mcp.core.ratios import compute_ratios

    symbols = [s.upper() for s in symbols]

    # Annual extraction always loads: it carries identity, industry class, and
    # the filing provenance every row needs regardless of the period requested.
    annual = _fetch_cached_rows(symbols, "financials", financials_period_key())

    overlay: dict[str, dict] = {}
    if period == "ttm":
        overlay = _fetch_cached_rows(symbols, TTM_DATA_TYPE, TTM_PERIOD)
    elif period == "quarterly":
        overlay = _fetch_cached_rows(symbols, QUARTER_METRICS_DATA_TYPE,
                                     QUARTER_METRICS_PERIOD)

    for tk in symbols:
        base = annual.get(tk)
        if not (base and base.get("metrics")):
            yield {"ticker": tk, "error": "not_cached",
                   "hint": "Not yet in the warm universe. Request it via "
                           "/v1/fundamentals/{ticker} to trigger extraction."}
            continue

        block = overlay.get(tk)
        metrics = base.get("metrics") or {}
        prior = base.get("prior_metrics")
        basis = "annual"
        as_of = (base.get("filing_info") or {}).get("period_end")

        if block and not block.get("error") and block.get("metrics"):
            metrics = block["metrics"]
            prior = block.get("prior")
            basis = period
            as_of = block.get("as_of") or as_of
        elif period in ("ttm", "quarterly"):
            basis = "annual_fallback"

        industry = base.get("industry_class")
        # Ratios are recomputed at serve time, never read from the cached blob —
        # same rule the interactive endpoints follow, so bulk and single-ticker
        # responses can never disagree.
        yield {
            "ticker": tk,
            "companyName": base.get("company_name"),
            "industryClass": industry,
            "fiscalYear": base.get("fiscal_year"),
            "period": period,
            "periodBasis": basis,
            "asOf": as_of,
            "metrics": metrics,
            "ratios": compute_ratios(metrics, prior=prior, industry=industry),
            "filing": base.get("filing_info"),
        }


def coverage_report() -> dict:
    """Honest answer to 'what do you actually have?' — the first question a
    data buyer asks, and the one a vague answer loses the deal on."""
    client = _client()
    if not client:
        return {"available": False,
                "hint": "Cache backend unavailable — coverage unknown."}

    def _count(data_type: str, period: str) -> int:
        try:
            res = (
                client.table("financial_cache")
                .select("cache_key", count="exact")
                .eq("data_type", data_type)
                .eq("period", period)
                .limit(1)
                .execute()
            )
            return int(res.count or 0)
        except Exception as exc:
            log.debug("coverage count failed for %s: %s", data_type, exc)
            return 0

    from sec_mcp.core.cache_keys import (
        QUARTER_METRICS_DATA_TYPE,
        QUARTER_METRICS_PERIOD,
        TTM_DATA_TYPE,
        TTM_PERIOD,
        financials_period_key,
    )
    return {
        "available": True,
        "coverage": {
            "annualFundamentals": _count("financials", financials_period_key()),
            "ttmMetrics": _count(TTM_DATA_TYPE, TTM_PERIOD),
            "quarterMetrics": _count(QUARTER_METRICS_DATA_TYPE, QUARTER_METRICS_PERIOD),
        },
        "source": "SEC EDGAR XBRL companyfacts (as-reported)",
        "note": ("Cached rows serve immediately and refresh in the background; "
                 "a ticker outside the warm set is extracted on first request."),
    }
