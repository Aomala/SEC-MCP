"""Polygon.io API client for cross-validating SEC XBRL financial data.

Fetches company details and standardized financials from Polygon, then
compares against SEC-extracted values to flag discrepancies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

_BASE = "https://api.polygon.io"
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _api_key() -> str:
    return get_config().polygon_api_key


def _get(path: str, params: dict | None = None) -> Any:
    """Make a cached GET request to Polygon API."""
    key = _api_key()
    if not key:
        log.debug("Polygon API key not configured — skipping request")
        return None

    p = {"apiKey": key, **(params or {})}
    cache_key = f"{path}|{sorted(p.items())}"

    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    try:
        url = f"{_BASE}{path}"
        resp = requests.get(url, params=p, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _CACHE[cache_key] = (time.time(), data)
        return data
    except Exception as exc:
        log.warning("Polygon API request failed: %s — %s", path, exc)
        return None


def get_ticker_details(ticker: str) -> dict | None:
    """Fetch company overview from Polygon (name, market cap, SIC, description, etc.)."""
    data = _get(f"/v3/reference/tickers/{ticker.upper()}")
    if not data or not isinstance(data, dict):
        return None
    return data.get("results")


def get_ticker_news(ticker: str, limit: int = 12) -> list[dict]:
    """Recent per-ticker news from Polygon (reliable, no Perplexity). Each item:
    title, article_url, publisher{name,...}, published_utc, tickers[], image_url,
    description, insights[] (per-ticker sentiment)."""
    data = _get(
        "/v2/reference/news",
        {"ticker": ticker.upper(), "limit": limit, "order": "desc", "sort": "published_utc"},
    )
    if not isinstance(data, dict):
        return []
    return data.get("results") or []


def get_index_snapshot(symbols: list[str]) -> dict[str, dict] | None:
    """Live snapshot for one or more index instruments (I:SPX, I:VIX, …).

    Returns a dict keyed by the requested symbol → {value, change, changePct,
    name}. Polygon's Indices add-on backs this; None when the key lacks the
    entitlement or the request fails (caller degrades gracefully).
    """
    if not symbols:
        return None
    # Polygon wants comma-joined symbols under ticker.any_of
    joined = ",".join(s.upper() for s in symbols)
    data = _get("/v3/snapshot/indices", {"ticker.any_of": joined, "limit": len(symbols)})
    if not data or not isinstance(data, dict):
        return None
    out: dict[str, dict] = {}
    for row in data.get("results") or []:
        sym = (row.get("ticker") or "").upper()
        if not sym:
            continue
        sess = row.get("session") or {}
        out[sym] = {
            "value": row.get("value"),
            "change": sess.get("change"),
            "changePct": sess.get("change_percent"),
            "name": row.get("name"),
        }
    return out or None


def get_index_aggs(symbol: str, date_from: str, date_to: str,
                   timespan: str = "day") -> list[dict] | None:
    """Daily (or other timespan) index-level history between two ISO dates.

    Each row is {t: epoch_ms, c: close_value, …}. Index aggs carry the level
    in `c`, same as equity closes.
    """
    sym = symbol.upper()
    data = _get(
        f"/v2/aggs/ticker/{sym}/range/1/{timespan}/{date_from}/{date_to}",
        {"adjusted": "true", "sort": "asc", "limit": 5000},
    )
    if not data or not isinstance(data, dict):
        return None
    results = data.get("results")
    return results if isinstance(results, list) else None


def get_grouped_daily(date: str) -> dict[str, dict] | None:
    """Every US stock's OHLC for one trading date in a single request.

    The breadth + cap-weight input: one call returns ~10k rows keyed by
    ticker → {c, o, h, l, v}. `date` is ISO 'YYYY-MM-DD'.
    """
    data = _get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
        {"adjusted": "true"},
    )
    if not data or not isinstance(data, dict):
        return None
    out: dict[str, dict] = {}
    for row in data.get("results") or []:
        sym = (row.get("T") or "").upper()
        if sym:
            out[sym] = row
    return out or None


def get_financials(ticker: str, limit: int = 4) -> list[dict] | None:
    """Fetch standardized financials from Polygon.

    Returns list of financial report dicts (income_statement, balance_sheet,
    cash_flow_statement, comprehensive_income), newest first.
    """
    data = _get("/vX/reference/financials", {
        "ticker": ticker.upper(),
        "limit": limit,
    })
    if not data or not isinstance(data, dict):
        return None
    results = data.get("results")
    if not results or not isinstance(results, list):
        return None
    return results


def _extract_polygon_value(financials: list[dict], statement: str, tag: str) -> float | None:
    """Pull a single value from the most recent Polygon financial report."""
    if not financials:
        return None
    latest = financials[0]
    stmt = latest.get("financials", {}).get(statement, {})
    entry = stmt.get(tag, {})
    return entry.get("value") if entry else None


def _pct_diff(a: float, b: float) -> float:
    """Percentage difference between two values. Returns 0.0 if base is zero."""
    if b == 0:
        return 0.0 if a == 0 else 100.0
    return abs(a - b) / abs(b) * 100.0


_MATCH_TOLERANCE = 5.0  # percent


def cross_check(ticker: str, sec_data: dict) -> dict:
    """Compare SEC XBRL extracted values against Polygon standardized financials.

    Args:
        ticker: Stock ticker symbol.
        sec_data: Dict of SEC-extracted metrics. Expected keys:
            revenue, net_income, total_assets, eps (values as floats).

    Returns:
        Dict keyed by metric name, each containing:
            sec: SEC-extracted value
            polygon: Polygon value (or None)
            diff_pct: percentage difference
            match: True if within 5% tolerance
    """
    financials = get_financials(ticker, limit=1)

    # Polygon tag mappings: (statement, tag)
    metric_map: dict[str, tuple[str, str]] = {
        "revenue": ("income_statement", "revenues"),
        "net_income": ("income_statement", "net_income_loss"),
        "total_assets": ("balance_sheet", "assets"),
        "eps": ("income_statement", "basic_earnings_per_share"),
    }

    result: dict[str, dict] = {}
    for metric, (statement, tag) in metric_map.items():
        sec_val = sec_data.get(metric)
        poly_val = _extract_polygon_value(financials, statement, tag) if financials else None

        if sec_val is not None and poly_val is not None:
            diff = _pct_diff(float(sec_val), float(poly_val))
            match = diff <= _MATCH_TOLERANCE
        else:
            diff = 0.0
            match = sec_val is None and poly_val is None

        result[metric] = {
            "sec": sec_val,
            "polygon": poly_val,
            "diff_pct": round(diff, 2),
            "match": match,
        }

    return result


def is_available() -> bool:
    """Check if Polygon API key is configured."""
    return bool(_api_key())
