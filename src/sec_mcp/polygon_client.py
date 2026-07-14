"""Polygon.io API client for cross-validating SEC XBRL financial data.

Fetches company details and standardized financials from Polygon, then
compares against SEC-extracted values to flag discrepancies.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

_BASE = "https://api.polygon.io"
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes

# SEC/yfinance class-share notation ("BRK-B", "BF-B") → Polygon's dot
# notation ("BRK.B"). Polygon 400s the dash form, which nulled every
# market-cap/shares/financials lookup for dash-class tickers.
_CLASS_SHARE_RE = re.compile(r"^([A-Z]{1,6})-([A-Z]{1,2})$")


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker to Polygon's notation (class shares use dots).

    Lives in the request layer so EVERY caller (details, news, financials,
    cross-check) benefits. Index symbols ("I:SPX") and plain tickers pass
    through untouched.
    """
    t = (ticker or "").strip().upper()
    m = _CLASS_SHARE_RE.match(t)
    return f"{m.group(1)}.{m.group(2)}" if m else t


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
    data = _get(f"/v3/reference/tickers/{normalize_ticker(ticker)}")
    if not data or not isinstance(data, dict):
        return None
    return data.get("results")


def get_ticker_news(ticker: str, limit: int = 12) -> list[dict]:
    """Recent per-ticker news from Polygon (reliable, no Perplexity). Each item:
    title, article_url, publisher{name,...}, published_utc, tickers[], image_url,
    description, insights[] (per-ticker sentiment)."""
    data = _get(
        "/v2/reference/news",
        {"ticker": normalize_ticker(ticker), "limit": limit,
         "order": "desc", "sort": "published_utc"},
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
        "ticker": normalize_ticker(ticker),
        "limit": limit,
    })
    if not data or not isinstance(data, dict):
        return None
    results = data.get("results")
    if not results or not isinstance(results, list):
        return None
    return results


def _report_value(report: dict | None, statement: str, tag: str) -> float | None:
    """Pull a single value from one Polygon financial report."""
    if not report:
        return None
    stmt = report.get("financials", {}).get(statement, {})
    entry = stmt.get(tag, {})
    return entry.get("value") if isinstance(entry, dict) else None


def _pct_diff(a: float, b: float) -> float:
    """Percentage difference between two values. Returns 0.0 if base is zero."""
    if b == 0:
        return 0.0 if a == 0 else 100.0
    return abs(a - b) / abs(b) * 100.0


_MATCH_TOLERANCE = 5.0  # percent
_ALIGN_WINDOW_DAYS = 14  # SEC vs Polygon period-end alignment tolerance


def _pick_aligned_report(rows: list[dict], sec_period_end: str | None,
                         timeframe: str) -> tuple[dict | None, bool]:
    """Pick the Polygon report on the same basis as our SEC period.

    Returns (report, aligned). Alignment = requested timeframe AND period end
    within ±14 days of the SEC period end (mirrors the MCP surface layer's
    fundamentals._cross_check). Non-December-FYE filers made the old
    "newest row" pick (a TTM row) flag 8-19% false mismatches on every
    annual comparison.
    """
    from datetime import date as _date
    want_end = str(sec_period_end or "")[:10]
    tf_rows = [r for r in rows if (r.get("timeframe") or "").lower() == timeframe]
    if want_end:
        for r in tf_rows:
            r_end = str(r.get("end_date") or "")[:10]
            try:
                delta = abs((_date.fromisoformat(want_end)
                             - _date.fromisoformat(r_end)).days)
            except ValueError:
                continue
            if delta <= _ALIGN_WINDOW_DAYS:
                return r, True
    # No aligned row — the newest row of the right TIMEFRAME still beats the
    # legacy any-row pick (which grabbed TTM against annual SEC data).
    if tf_rows:
        return tf_rows[0], False
    return (rows[0], False) if rows else (None, False)


def cross_check(ticker: str, sec_data: dict, *,
                sec_period_end: str | None = None,
                sec_fiscal_year: int | str | None = None,
                timeframe: str = "annual") -> dict:
    """Compare SEC XBRL extracted values against Polygon standardized financials.

    Args:
        ticker: Stock ticker symbol.
        sec_data: Dict of SEC-extracted metrics. Expected keys:
            revenue, net_income, total_assets (floats), plus
            eps / eps_diluted / eps_basic for the EPS comparison.
        sec_period_end: SEC period end date (ISO) — enables period ALIGNMENT:
            the Polygon report with the requested timeframe whose end date is
            within ±14 days is compared. Without it, the newest report of the
            requested timeframe is used.
        sec_fiscal_year: SEC fiscal year, echoed into the basis label.
        timeframe: Polygon timeframe to compare against ("annual"/"quarterly"/"ttm").

    Returns:
        Dict keyed by metric name, each containing:
            sec: SEC-extracted value
            polygon: Polygon value (or None)
            diff_pct: percentage difference
            match: True if within 5% tolerance
        Plus an additive "basis" key (str) naming what was compared, e.g.
        "annual_fy2025_vs_polygon_annual_2025-09-27". Existing consumers doing
        keyed metric lookups are unaffected.
    """
    rows = get_financials(ticker, limit=8) or []
    report, aligned = _pick_aligned_report(rows, sec_period_end, timeframe)

    # SEC EPS was historically never populated (extraction emits
    # eps_diluted/eps_basic, not "eps") — pick diluted first and compare it
    # against Polygon's matching diluted tag.
    sec_eps = sec_data.get("eps")
    eps_tag = "basic_earnings_per_share"
    if sec_eps is None:
        if sec_data.get("eps_diluted") is not None:
            sec_eps, eps_tag = sec_data.get("eps_diluted"), "diluted_earnings_per_share"
        else:
            sec_eps = sec_data.get("eps_basic")

    # Polygon tag mappings: (statement, tag, sec value)
    metric_map: dict[str, tuple[str, str, float | None]] = {
        "revenue": ("income_statement", "revenues", sec_data.get("revenue")),
        "net_income": ("income_statement", "net_income_loss", sec_data.get("net_income")),
        "total_assets": ("balance_sheet", "assets", sec_data.get("total_assets")),
        "eps": ("income_statement", eps_tag, sec_eps),
    }

    result: dict[str, Any] = {}
    for metric, (statement, tag, sec_val) in metric_map.items():
        poly_val = _report_value(report, statement, tag)

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

    # Additive basis label — names the two periods actually compared.
    sec_side = timeframe + (f"_fy{sec_fiscal_year}" if sec_fiscal_year else "")
    if report:
        poly_side = (f"polygon_{(report.get('timeframe') or 'unknown')}"
                     f"_{report.get('end_date') or 'unknown'}")
        result["basis"] = (f"{sec_side}_vs_{poly_side}"
                           + ("" if aligned else "_unaligned"))
    else:
        result["basis"] = f"{sec_side}_vs_polygon_none"

    return result


def is_available() -> bool:
    """Check if Polygon API key is configured."""
    return bool(_api_key())
