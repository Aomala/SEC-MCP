"""Supabase caching layer for financial data.

Caches extraction results, FMP data, and segment data in Supabase Postgres.
Graceful fallback — app works fine without Supabase configured.

Tables expected (create via Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS financial_cache (
        id BIGSERIAL PRIMARY KEY,
        cache_key TEXT UNIQUE NOT NULL,
        ticker TEXT NOT NULL,
        data_type TEXT NOT NULL,          -- 'financials', 'geo_segments', 'product_segments', 'income_history'
        data JSONB NOT NULL,
        period TEXT DEFAULT 'annual',
        date_from TEXT,                   -- date range filter
        date_to TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX idx_cache_key ON financial_cache(cache_key);
    CREATE INDEX idx_cache_ticker ON financial_cache(ticker, data_type);
    CREATE INDEX idx_cache_expires ON financial_cache(expires_at);
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

_client = None
_available: bool | None = None

# Cache TTLs by data type (seconds). Freshness comes from the weekly batch
# rewrite cadence (batch_cache --refresh-older-than), not TTL expiry — TTLs
# are the backstop that keeps abandoned entries from serving forever.
CACHE_TTLS = {
    "financials": 1209600,      # 14 days — a filed document is immutable but the
                                #   '|latest' key REBINDS at each new filing, so a
                                #   long TTL serves a superseded fiscal period to
                                #   out-of-universe tickers the batch never rewrites
    "cross_check": 604800,      # 7 days — Polygon validation
    "geo_segments": 604800,     # 7 days — segments are annual
    "product_segments": 604800, # 7 days
    "income_history": 604800,   # 7 days
    "balance_history": 604800,
    "cashflow_history": 604800,
    "sec_income_history_v2": 1209600,  # 14 days — was silently falling to the
                                       #   1h default and re-extracting hourly
    "ttm_metrics": 1209600,     # 14 days — refreshed by the weekly batch
    "quarter_metrics": 1209600, # 14 days — latest standalone quarter block
}
# Quarterly financial_cache entries refresh faster than annual ones.
QUARTERLY_FINANCIALS_TTL = 1209600  # 14 days
# Section text gets 7 day TTL (matched by prefix)
_SECTION_TTL = 604800


def _get_client():
    """Lazy-init Supabase client."""
    global _client, _available
    if _available is False:
        return None
    if _client is not None:
        return _client

    cfg = get_config()
    if not cfg.supabase_url or not cfg.supabase_key:
        _available = False
        return None

    try:
        from supabase import create_client
        _client = create_client(cfg.supabase_url, cfg.supabase_key)
        _available = True
        log.info("Supabase cache connected")
        return _client
    except ImportError:
        log.info("supabase-py not installed — caching disabled")
        _available = False
        return None
    except Exception as exc:
        log.warning("Supabase connection failed: %s", exc)
        _available = False
        return None


def is_available() -> bool:
    """Check if Supabase caching is available."""
    _get_client()
    return _available is True


def _cache_key(ticker: str, data_type: str, period: str = "annual",
               date_from: str = "", date_to: str = "") -> str:
    """Build a unique cache key."""
    return f"{ticker.upper()}|{data_type}|{period}|{date_from}|{date_to}"


def get_cached(
    ticker: str,
    data_type: str,
    period: str = "annual",
    date_from: str = "",
    date_to: str = "",
) -> Any | None:
    """Retrieve cached data if not expired."""
    client = _get_client()
    if not client:
        return None

    key = _cache_key(ticker, data_type, period, date_from, date_to)
    try:
        result = (
            client.table("financial_cache")
            .select("data, expires_at")
            .eq("cache_key", key)
            .gte("expires_at", datetime.now(timezone.utc).isoformat())
            .limit(1)
            .execute()
        )
        if result.data:
            log.debug("Supabase cache hit: %s", key)
            return result.data[0]["data"]
    except Exception as exc:
        log.debug("Supabase cache read failed: %s", exc)

    return None


def set_cached(
    ticker: str,
    data_type: str,
    data: Any,
    period: str = "annual",
    date_from: str = "",
    date_to: str = "",
    ttl: int | None = None,
) -> bool:
    """Store data in cache with TTL-based expiration.

    `ttl` overrides the per-type default — use a short TTL for fallback-quality
    results so they retry the primary source sooner instead of squatting for
    the full window.
    """
    client = _get_client()
    if not client:
        return False

    key = _cache_key(ticker, data_type, period, date_from, date_to)
    if ttl is None:
        ttl = CACHE_TTLS.get(data_type, _SECTION_TTL if data_type.startswith("section_") else 3600)
    expires = datetime.now(timezone.utc) + timedelta(seconds=ttl)

    try:
        client.table("financial_cache").upsert({
            "cache_key": key,
            "ticker": ticker.upper(),
            "data_type": data_type,
            "data": data,
            "period": period,
            "date_from": date_from or None,
            "date_to": date_to or None,
            # Explicit so upserts refresh it — batch freshness checks read this
            # as "last written at" (Postgres DEFAULT only fires on insert).
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires.isoformat(),
        }, on_conflict="cache_key").execute()
        log.debug("Supabase cache set: %s (TTL %ds)", key, ttl)
        return True
    except Exception as exc:
        log.debug("Supabase cache write failed: %s", exc)
        return False


def cached_age_days(
    ticker: str,
    data_type: str,
    period: str = "annual",
    date_from: str = "",
    date_to: str = "",
) -> float | None:
    """Days since a live (unexpired) entry was last written; None when absent.

    Rows written before created_at became upsert-refreshed report their
    original insert time — safe direction: at worst we refresh early.
    """
    client = _get_client()
    if not client:
        return None
    key = _cache_key(ticker, data_type, period, date_from, date_to)
    try:
        result = (
            client.table("financial_cache")
            .select("created_at")
            .eq("cache_key", key)
            .gte("expires_at", datetime.now(timezone.utc).isoformat())
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        created = result.data[0].get("created_at")
        if not created:
            return 0.0
        dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        return max((datetime.now(timezone.utc) - dt).total_seconds() / 86400, 0.0)
    except Exception as exc:
        log.debug("Supabase cache age read failed: %s", exc)
        return None


def cleanup_expired():
    """Delete expired cache entries. Call periodically."""
    client = _get_client()
    if not client:
        return

    try:
        client.table("financial_cache").delete().lt(
            "expires_at", datetime.now(timezone.utc).isoformat()
        ).execute()
    except Exception as exc:
        log.debug("Supabase cache cleanup failed: %s", exc)
