"""Real-time price service with multi-provider fallback.

Tries providers in order:
  1. Polygon snapshot (paid, lowest latency, includes prev close)
  2. yfinance via MarketDataProvider (free, occasional throttling)
  3. FMP quote (paid, broad coverage, good intl support)

Returns a normalized dict with price/change/change_pct/volume/market_cap
plus `source` (which provider answered) and `timestamp`. Each call also
includes `cached_age_seconds` so the UI can display freshness.

Caching is shared across providers — first successful fetch lives 30s.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from sec_mcp.config import get_config
from sec_mcp.core.market_data import get_market_data_provider, YFINANCE_AVAILABLE

log = logging.getLogger(__name__)

# 30-second TTL — short enough to feel real-time, long enough that a tab open
# for an hour with /v1/price polling every 30s only generates 120 upstream calls
_TTL_SECONDS = 30
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _normalize_ticker(ticker: str) -> str:
    """yfinance + Polygon use '-' for class shares (BRK-B), not '.'."""
    return ticker.strip().upper().replace(".", "-")


def _from_polygon(ticker: str) -> dict | None:
    """Snapshot endpoint — last trade + day stats + prev close in one call."""
    key = get_config().polygon_api_key
    if not key:
        return None
    try:
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        resp = requests.get(url, params={"apiKey": key}, timeout=4)
        resp.raise_for_status()
        body = resp.json() or {}
        snap = body.get("ticker") or {}
        if not snap:
            return None
        day = snap.get("day") or {}
        prev_day = snap.get("prevDay") or {}
        last_trade = snap.get("lastTrade") or {}
        last_quote = snap.get("lastQuote") or {}
        # Best-available "current" price — last trade beats day close
        price = (
            last_trade.get("p")
            or day.get("c")
            or last_quote.get("p")
        )
        prev_close = prev_day.get("c")
        if price is None:
            return None
        change = (price - prev_close) if prev_close else None
        change_pct = (change / prev_close * 100.0) if (change is not None and prev_close) else None
        return {
            "price": float(price),
            "change": float(change) if change is not None else None,
            "change_pct": float(change_pct) if change_pct is not None else None,
            "volume": int(day.get("v") or 0),
            "high_52w": None,  # not provided by snapshot endpoint
            "low_52w": None,
            "pe_ratio": None,
            "market_cap": None,
        }
    except Exception as exc:
        log.debug("polygon snapshot failed for %s: %s", ticker, exc)
        return None


def _from_yfinance(ticker: str) -> dict | None:
    """MarketDataProvider already handles fallback chain + 5-min internal cache."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        return get_market_data_provider().get_price(ticker)
    except Exception as exc:
        log.debug("yfinance fallback failed for %s: %s", ticker, exc)
        return None


def _from_fmp(ticker: str) -> dict | None:
    """FMP /quote endpoint — broad coverage, good for intl tickers."""
    key = get_config().fmp_api_key
    if not key:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}"
        resp = requests.get(url, params={"apikey": key}, timeout=4)
        resp.raise_for_status()
        rows = resp.json() or []
        if not rows or not isinstance(rows, list):
            return None
        q = rows[0]
        price = q.get("price")
        if price is None:
            return None
        return {
            "price": float(price),
            "change": float(q.get("change")) if q.get("change") is not None else None,
            "change_pct": float(q.get("changesPercentage")) if q.get("changesPercentage") is not None else None,
            "volume": int(q.get("volume") or 0),
            "high_52w": float(q.get("yearHigh")) if q.get("yearHigh") is not None else None,
            "low_52w": float(q.get("yearLow")) if q.get("yearLow") is not None else None,
            "pe_ratio": float(q.get("pe")) if q.get("pe") is not None else None,
            "market_cap": int(q.get("marketCap")) if q.get("marketCap") is not None else None,
        }
    except Exception as exc:
        log.debug("FMP quote failed for %s: %s", ticker, exc)
        return None


def get_realtime_price(ticker: str) -> dict:
    """Get a current price snapshot, trying Polygon → yfinance → FMP.

    Returns a dict with normalized price fields plus `source`, `timestamp`,
    `cached_age_seconds`, and `error` (only if every provider failed).
    """
    norm = _normalize_ticker(ticker)
    now = time.time()

    with _lock:
        hit = _cache.get(norm)
        if hit and (now - hit[0]) < _TTL_SECONDS:
            payload = dict(hit[1])
            payload["cached_age_seconds"] = round(now - hit[0], 1)
            return payload

    # Try providers in order — first successful response wins
    for name, fn in (("polygon", _from_polygon), ("yfinance", _from_yfinance), ("fmp", _from_fmp)):
        data = fn(norm)
        if data is None or data.get("price") is None:
            continue
        result = {
            "ticker": norm,
            **data,
            "source": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cached_age_seconds": 0.0,
        }
        with _lock:
            _cache[norm] = (now, result)
        return result

    return {
        "ticker": norm,
        "error": "All providers failed (Polygon, yfinance, FMP).",
        "source": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
