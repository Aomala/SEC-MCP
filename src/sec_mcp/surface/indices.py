"""get_index / get_market_overview — real index instruments, never proxies.

Until now "the market" was faked from tracking ETFs (SPY≈S&P 500). These tools
collect the actual index levels (I:SPX, I:NDX, I:DJI, I:RUT) and volatility
(I:VIX) from Polygon's Indices feed, plus market breadth and a cap-weighted
sector rollup of the S&P 500 constituents.

Same 24/7 contract as get_quote: a closed market is NEVER an error — the tool
returns the last level labeled session="closed". Every response carries the
mandatory meta block via tool_guard.

The heavy datasets (breadth across ~500 stocks, constituent cap-weights) are NOT
computed on the request path. `ingest_indices.py` recomputes them on a schedule
and writes a blob to disk; get_market_overview reads that blob and merges it with
a fresh, cheap index snapshot.
"""

from __future__ import annotations

# stdlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sec_mcp import polygon_client

# response contract helpers
from sec_mcp.surface.meta import (
    UPSTREAM_ERROR,
    ToolError,
    build_meta,
)

# session classification + session-aware TTLs (indices track the equity session)
from sec_mcp.surface.session import market_session, quote_ttl

# ── Index registry ──────────────────────────────────────────────────────────
# The five instruments the dashboard shows. Keys are Polygon index tickers.
INDEX_NAMES: dict[str, str] = {
    "I:SPX": "S&P 500",
    "I:NDX": "Nasdaq 100",
    "I:DJI": "Dow Jones Industrial Average",
    "I:RUT": "Russell 2000",
    "I:VIX": "CBOE Volatility Index",
}
DASHBOARD_INDICES: list[str] = ["I:SPX", "I:NDX", "I:DJI", "I:RUT", "I:VIX"]

# Canonical Polygon symbol → yfinance symbol (the free fallback when the
# Polygon Indices add-on isn't entitled). yfinance carries real index levels
# AND VIX, so the tape always answers even on a Stocks-only Polygon plan.
_YF: dict[str, str] = {
    "I:SPX": "^GSPC", "I:NDX": "^NDX", "I:DJI": "^DJI",
    "I:RUT": "^RUT", "I:VIX": "^VIX",
}

# Common aliases → canonical Polygon symbol, so callers can say "SPX" / "^GSPC".
_ALIASES: dict[str, str] = {
    "SPX": "I:SPX", "^GSPC": "I:SPX", "GSPC": "I:SPX", "SP500": "I:SPX",
    "NDX": "I:NDX", "^NDX": "I:NDX",
    "DJI": "I:DJI", "^DJI": "I:DJI", "DOW": "I:DJI",
    "RUT": "I:RUT", "^RUT": "I:RUT",
    "VIX": "I:VIX", "^VIX": "I:VIX",
}

# Where ingest_indices.py drops the breadth + constituent blob.
MARKET_CACHE = Path.home() / ".sec_mcp_cache" / "_market" / "overview.json"
MARKET_CACHE_TTL = 6 * 3600  # a stale blob (up to 6h) still beats none

# Surface-level snapshot cache: {symbol: (fetched_unix, payload)}, session-aware.
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()

_WINDOW_DAYS = {"5D": 7, "1M": 32, "3M": 95, "6M": 190, "1Y": 370, "5Y": 1830}


def normalize_index_symbol(symbol: str | None) -> str:
    """Resolve an alias or bare ticker to a canonical Polygon index symbol."""
    if not symbol or not isinstance(symbol, str) or not symbol.strip():
        raise ToolError(
            "INVALID_INPUT", "'symbol' is required.",
            "Pass an index like 'I:SPX', 'SPX', or '^VIX'.",
        )
    raw = symbol.strip().upper()
    if raw in INDEX_NAMES:
        return raw
    if raw in _ALIASES:
        return _ALIASES[raw]
    # Accept any I:* the caller supplies even if not in our named registry.
    if raw.startswith("I:"):
        return raw
    raise ToolError(
        "INVALID_INPUT", f"'{symbol}' is not a recognized index.",
        f"Supported: {', '.join(sorted(INDEX_NAMES))} (aliases like 'SPX'/'VIX' work).",
    )


_YF_PERIOD = {"5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}


def _yf_snapshot(symbols: list[str]) -> dict[str, dict]:
    """Fallback index snapshot via yfinance (real levels + VIX, no add-on needed)."""
    out: dict[str, dict] = {}
    try:
        import yfinance as yf
    except Exception:
        return out
    for sym in symbols:
        yf_sym = _YF.get(sym)
        if not yf_sym:
            continue
        try:
            fi = yf.Ticker(yf_sym).fast_info
            last, prev = fi.last_price, fi.previous_close
            if last is None:
                continue
            change = (last - prev) if prev else None
            out[sym] = {
                "value": last,
                "change": round(change, 2) if change is not None else None,
                "changePct": round((change / prev) * 100, 2) if prev else None,
                "name": INDEX_NAMES.get(sym),
            }
        except Exception:
            continue
    return out


def _index_snapshot(symbols: list[str]) -> tuple[dict[str, dict], str]:
    """Polygon Indices first; yfinance fallback. Returns (data, provider)."""
    snap = polygon_client.get_index_snapshot(symbols)
    if snap:
        return snap, "polygon"
    return _yf_snapshot(symbols), "yfinance"


def _history(symbol: str, window: str) -> dict | None:
    """chartSeries {labels, levels} for an index over the window (oldest→newest).

    Polygon index aggs first; yfinance history fallback when the add-on is
    absent — the dashboard sparkline always has data.
    """
    days = _WINDOW_DAYS.get(window.upper(), 32)
    to_d = datetime.now(timezone.utc).date()
    from_d = to_d - timedelta(days=days)
    rows = polygon_client.get_index_aggs(symbol, from_d.isoformat(), to_d.isoformat())
    if rows:
        labels: list[str] = []
        levels: list[float] = []
        for r in rows:
            ts, close = r.get("t"), r.get("c")
            if ts is None or close is None:
                continue
            labels.append(datetime.fromtimestamp(ts / 1000, timezone.utc).date().isoformat())
            levels.append(close)
        if levels:
            return {"labels": labels, "levels": levels}

    # yfinance fallback
    yf_sym = _YF.get(symbol)
    if not yf_sym:
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(yf_sym).history(period=_YF_PERIOD.get(window.upper(), "1mo"))
        if hist is None or hist.empty:
            return None
        labels = [d.date().isoformat() for d in hist.index]
        levels = [round(float(c), 2) for c in hist["Close"].tolist()]
        return {"labels": labels, "levels": levels} if levels else None
    except Exception:
        return None


def get_index_impl(symbol, include_history: bool = True,
                   history_window: str = "1M") -> dict:
    """Core implementation shared by the MCP tool and the REST route."""
    t0 = time.time()
    sym = normalize_index_symbol(symbol)
    session = market_session()
    ttl = quote_ttl(session)

    # ── cache read (session-aware, snapshot only; history is cheap-ish) ──────
    with _lock:
        hit = _cache.get(sym)
    if hit and (time.time() - hit[0]) < ttl and not include_history:
        fetched_at, payload = hit
        out = dict(payload)
        out["session"] = session
        out["ageSeconds"] = round(time.time() - fetched_at, 1)
        out["meta"] = build_meta(f"{payload['provider']}:indices", t0,
                                 cache_hit=True, as_of=payload["asOf"])
        return out

    # ── live snapshot (Polygon Indices → yfinance) ───────────────────────────
    snap, provider = _index_snapshot([sym])
    row = (snap or {}).get(sym)
    if not row or row.get("value") is None:
        raise ToolError(
            UPSTREAM_ERROR,
            f"No level available for {sym} from Polygon Indices or yfinance.",
            "Check the symbol (e.g. 'I:SPX', 'I:VIX'). If you expect Polygon "
            "index data, confirm the plan includes the Indices add-on.",
        )
    as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "symbol": sym,
        "name": INDEX_NAMES.get(sym, row.get("name") or sym),
        "level": row.get("value"),
        "change": row.get("change"),
        "changePct": row.get("changePct"),
        "asOf": as_of,
        "session": session,
        "provider": provider,
        "priceBasis": "last_close" if session == "closed" else "last_level",
    }
    with _lock:
        _cache[sym] = (time.time(), payload)

    out = dict(payload)
    if include_history:
        out["chartSeries"] = _history(sym, history_window)
    out["meta"] = build_meta(f"{provider}:indices", t0, cache_hit=False, as_of=as_of)
    return out


def _read_market_cache() -> dict | None:
    """Load the ingest worker's breadth + constituent blob, if fresh enough."""
    if not MARKET_CACHE.exists():
        return None
    try:
        blob = json.loads(MARKET_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - blob.get("cached_at", 0) > MARKET_CACHE_TTL:
        return None
    return blob


def get_market_overview_impl() -> dict:
    """One-call dashboard payload: indices + breadth + cap-weighted sectors.

    Indices are fetched live (one cheap Polygon call). Breadth and the
    constituent cap-weight rollup are read from the ingest worker's cached blob
    — never recomputed on the request path — so this stays fast and never blocks
    on ~500 quotes. Missing blob → those blocks are null (still a valid payload).
    """
    t0 = time.time()
    session = market_session()

    snap, provider = _index_snapshot(DASHBOARD_INDICES)
    snap = snap or {}
    indices = []
    for sym in DASHBOARD_INDICES:
        row = snap.get(sym) or {}
        indices.append({
            "symbol": sym,
            "name": INDEX_NAMES[sym],
            "level": row.get("value"),
            "change": row.get("change"),
            "changePct": row.get("changePct"),
        })

    blob = _read_market_cache() or {}
    as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "indices": indices,
        "breadth": blob.get("breadth"),          # {advancers, decliners, advDecRatio, newHighs, newLows, pctAbove50dma}
        "sectors": blob.get("sectors"),          # [{sector, weightPct, count, avgChangePct}]
        "session": session,
        "provider": provider,
        "constituentsAsOf": blob.get("cached_at_iso"),
        "asOf": as_of,
        "meta": build_meta(f"{provider}:indices", t0, cache_hit=bool(blob), as_of=as_of),
    }
