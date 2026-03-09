"""Financial Modeling Prep API client.

Provides geographic revenue, product segment revenue, and income statements
with multi-year history. Used as primary data source for segments/geo when
XBRL extraction is incomplete.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/stable"
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 600  # 10 minutes


def _api_key() -> str:
    return get_config().fmp_api_key


def _get(endpoint: str, params: dict | None = None) -> Any:
    """Make a cached GET request to FMP API."""
    key = _api_key()
    if not key:
        return None

    p = {"apikey": key, **(params or {})}
    cache_key = f"{endpoint}|{sorted(p.items())}"

    # Check cache
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    try:
        url = f"{_BASE}/{endpoint}"
        resp = requests.get(url, params=p, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _CACHE[cache_key] = (time.time(), data)
        return data
    except Exception as exc:
        log.warning("FMP API request failed: %s %s — %s", endpoint, params, exc)
        return None


def get_geo_segments(
    symbol: str,
    period: str = "annual",
    limit: int = 5,
) -> list[dict]:
    """Get geographic revenue segmentation.

    Returns list of {date, fiscal_year, period, segments: [{name, value}]}
    """
    data = _get("revenue-geographic-segmentation", {
        "symbol": symbol.upper(),
        "structure": "flat",
        "period": period,
    })
    if not data or not isinstance(data, list):
        return []

    results = []
    for entry in data[:limit]:
        seg_data = entry.get("data", {})
        if not seg_data:
            continue
        segments = [
            {"name": _clean_segment_name(k), "value": v}
            for k, v in seg_data.items()
            if isinstance(v, (int, float)) and v > 0
        ]
        if segments:
            results.append({
                "date": entry.get("date", ""),
                "fiscal_year": entry.get("fiscalYear"),
                "period": entry.get("period", "FY"),
                "segments": sorted(segments, key=lambda s: s["value"], reverse=True),
            })
    return results


def get_product_segments(
    symbol: str,
    period: str = "annual",
    limit: int = 5,
) -> list[dict]:
    """Get product/business segment revenue.

    Returns list of {date, fiscal_year, period, segments: [{name, value}]}
    """
    data = _get("revenue-product-segmentation", {
        "symbol": symbol.upper(),
        "structure": "flat",
        "period": period,
    })
    if not data or not isinstance(data, list):
        return []

    results = []
    for entry in data[:limit]:
        seg_data = entry.get("data", {})
        if not seg_data:
            continue
        segments = [
            {"name": _clean_segment_name(k), "value": v}
            for k, v in seg_data.items()
            if isinstance(v, (int, float)) and v > 0
        ]
        if segments:
            results.append({
                "date": entry.get("date", ""),
                "fiscal_year": entry.get("fiscalYear"),
                "period": entry.get("period", "FY"),
                "segments": sorted(segments, key=lambda s: s["value"], reverse=True),
            })
    return results


def get_income_statements(
    symbol: str,
    period: str = "annual",
    limit: int = 10,
) -> list[dict]:
    """Get income statements with multi-year history.

    Returns list of dicts with standardized field names, sorted newest-first.
    """
    data = _get("income-statement", {
        "symbol": symbol.upper(),
        "period": period,
        "limit": limit,
    })
    if not data or not isinstance(data, list):
        return []

    return data


def get_balance_sheet(
    symbol: str,
    period: str = "annual",
    limit: int = 10,
) -> list[dict]:
    """Get balance sheet statements."""
    data = _get("balance-sheet-statement", {
        "symbol": symbol.upper(),
        "period": period,
        "limit": limit,
    })
    if not data or not isinstance(data, list):
        return []
    return data


def get_cash_flow(
    symbol: str,
    period: str = "annual",
    limit: int = 10,
) -> list[dict]:
    """Get cash flow statements."""
    data = _get("cash-flow-statement", {
        "symbol": symbol.upper(),
        "period": period,
        "limit": limit,
    })
    if not data or not isinstance(data, list):
        return []
    return data


def _clean_segment_name(name: str) -> str:
    """Clean up FMP segment names (remove ' Segment' suffix, etc.)."""
    name = name.strip()
    for suffix in (" Segment", " segment", " Revenue", " revenue"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def is_available() -> bool:
    """Check if FMP API key is configured."""
    return bool(_api_key())
