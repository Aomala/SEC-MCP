"""Perplexity API client for real-time financial search.

Uses the Perplexity sonar model for web-grounded financial research.
Results are cached in Supabase `search_cache` table with 30-minute TTL.
Works without Supabase — caching is optional.

Expected table (create via Supabase SQL editor):

    CREATE TABLE IF NOT EXISTS search_cache (
        id BIGSERIAL PRIMARY KEY,
        cache_key TEXT UNIQUE NOT NULL,
        query TEXT NOT NULL,
        ticker TEXT,
        data JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX idx_search_cache_key ON search_cache(cache_key);
    CREATE INDEX idx_search_cache_expires ON search_cache(expires_at);
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

_BASE = "https://api.perplexity.ai/chat/completions"
_MODEL = "sonar"
_CACHE_TTL = 1800  # 30 minutes

_SYSTEM_PROMPT = (
    "You are a financial research assistant. Provide accurate, data-driven "
    "answers with specific numbers and sources. Focus on the most recent "
    "publicly available data. Be concise and factual."
)

_VALIDATION_PROMPT = (
    "You are a financial data verification assistant. Compare the given "
    "metric value against publicly available sources. Respond ONLY with "
    "valid JSON in this exact format: "
    '{"verified": true/false, "source_value": "value from sources", '
    '"explanation": "brief explanation"}. '
    "No markdown, no code fences, just raw JSON."
)

# Supabase client (lazy init, separate from supabase_cache module)
_sb_client = None
_sb_available: bool | None = None
_quota_exhausted_until: float = 0  # Circuit breaker timestamp


def _get_sb_client():
    """Lazy-init Supabase client for search cache."""
    global _sb_client, _sb_available
    if _sb_available is False:
        return None
    if _sb_client is not None:
        return _sb_client

    cfg = get_config()
    if not cfg.supabase_url or not cfg.supabase_key:
        _sb_available = False
        return None

    try:
        from supabase import create_client
        _sb_client = create_client(cfg.supabase_url, cfg.supabase_key)
        _sb_available = True
        return _sb_client
    except ImportError:
        _sb_available = False
        return None
    except Exception as exc:
        log.warning("Supabase connection failed (perplexity cache): %s", exc)
        _sb_available = False
        return None


def _cache_key(query: str) -> str:
    """SHA-256 hash of the query for cache key."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _cache_get(query: str) -> dict | None:
    """Read from search_cache if not expired."""
    client = _get_sb_client()
    if not client:
        return None

    key = _cache_key(query)
    try:
        result = (
            client.table("search_cache")
            .select("data")
            .eq("cache_key", key)
            .gte("expires_at", datetime.now(timezone.utc).isoformat())
            .limit(1)
            .execute()
        )
        if result.data:
            log.debug("Perplexity cache hit: %s", key[:12])
            return result.data[0]["data"]
    except Exception as exc:
        log.debug("Perplexity cache read failed: %s", exc)

    return None


def _cache_set(query: str, data: dict, ticker: str | None = None) -> None:
    """Write result to search_cache."""
    client = _get_sb_client()
    if not client:
        return

    key = _cache_key(query)
    expires = datetime.now(timezone.utc) + timedelta(seconds=_CACHE_TTL)

    try:
        client.table("search_cache").upsert({
            "cache_key": key,
            "query": query[:500],
            "ticker": ticker.upper() if ticker else None,
            "data": data,
            "expires_at": expires.isoformat(),
        }, on_conflict="cache_key").execute()
        log.debug("Perplexity cache set: %s", key[:12])
    except Exception as exc:
        log.debug("Perplexity cache write failed: %s", exc)


def _api_key() -> str:
    return get_config().perplexity_api_key


def _call(system: str, user_message: str) -> dict | None:
    """Call Perplexity chat completions API.

    Returns {"content": str, "citations": list[str]} or None on failure.
    """
    key = _api_key()
    if not key:
        log.debug("Perplexity API key not configured — skipping request")
        return None

    # Circuit breaker: skip if we've hit quota recently (avoid 30s timeouts)
    global _quota_exhausted_until
    if _quota_exhausted_until and time.time() < _quota_exhausted_until:
        return None

    try:
        resp = requests.post(
            _BASE,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])

        _quota_exhausted_until = 0  # Reset on success
        return {"content": content, "citations": citations}
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 429):
            # Quota exhausted or rate limited — back off for 10 minutes
            _quota_exhausted_until = time.time() + 600
            log.warning("Perplexity quota exhausted — disabling for 10 min")
        else:
            log.warning("Perplexity API request failed: %s", exc)
        return None
    except Exception as exc:
        log.warning("Perplexity API request failed: %s", exc)
        return None


def search(query: str, ticker: str | None = None) -> dict | None:
    """Search for real-time financial information via Perplexity.

    Args:
        query: The search query.
        ticker: Optional ticker to prepend for context.

    Returns:
        {"content": str, "citations": list[str]} or None.
    """
    full_query = f"{ticker.upper()}: {query}" if ticker else query

    # Check cache first
    cached = _cache_get(full_query)
    if cached:
        return cached

    result = _call(_SYSTEM_PROMPT, full_query)
    if result:
        _cache_set(full_query, result, ticker=ticker)

    return result


def search_financial_news(ticker: str) -> dict | None:
    """Search for recent financial news and analyst opinions for a ticker.

    Returns:
        {"content": str, "citations": list[str]} or None.
    """
    query = (
        f"What are the most recent financial news, analyst ratings, and "
        f"market opinions for {ticker.upper()}? Include price targets, "
        f"earnings estimates, and any notable developments from the past week."
    )
    return search(query, ticker=ticker)


def validate_metric(
    ticker: str,
    metric_name: str,
    sec_value: float,
) -> dict | None:
    """Verify a financial metric against public sources via Perplexity.

    Args:
        ticker: Stock ticker symbol.
        metric_name: Name of the metric (e.g. "revenue", "net_income").
        sec_value: The value extracted from SEC filings.

    Returns:
        {"verified": bool, "source_value": str, "explanation": str,
         "citations": list[str]} or None.
    """
    user_msg = (
        f"For {ticker.upper()}, the SEC filing reports {metric_name} as "
        f"${sec_value:,.0f}. Verify this value against the most recent "
        f"publicly available financial data. Is this accurate?"
    )

    result = _call(_VALIDATION_PROMPT, user_msg)
    if not result:
        return None

    # Parse the structured response from Perplexity
    try:
        parsed = json.loads(result["content"])
        return {
            "verified": bool(parsed.get("verified", False)),
            "source_value": str(parsed.get("source_value", "unknown")),
            "explanation": str(parsed.get("explanation", "")),
            "citations": result.get("citations", []),
        }
    except (json.JSONDecodeError, KeyError):
        # LLM didn't return clean JSON — return raw content as explanation
        return {
            "verified": False,
            "source_value": "parse_error",
            "explanation": result["content"][:500],
            "citations": result.get("citations", []),
        }


def is_available() -> bool:
    """Check if Perplexity API key is configured."""
    return bool(_api_key())
