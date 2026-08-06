"""get_quote — session-aware price quotes that are NEVER silently stale.

Off-hours is not an error: when markets are closed the tool returns the
last available price labeled session="closed". Mandatory metadata on every
response: {asOf, session, provider, ageSeconds}.
"""

from __future__ import annotations

# stdlib
import threading
import time
from datetime import datetime, timezone

# provider chain (Polygon → yfinance → FMP) with its own 30s micro-cache
from sec_mcp.core.realtime_price import get_realtime_price

# response contract helpers
from sec_mcp.surface.meta import (
    UPSTREAM_ERROR,
    ToolError,
    build_meta,
    require_ticker,
)

# session classification + session-aware TTLs
from sec_mcp.surface.session import market_session, quote_ttl

# Daily context cache: {ticker: (fetched_unix, {marketCap, high52w, low52w})}.
# The Polygon snapshot path answers fastest but never carries market cap or
# the 52-week range — without this fill, those fields were null on every
# Polygon-served quote. Reference data moves slowly; one fetch a day is fine.
_CTX_TTL = 86400.0
_ctx_cache: dict[str, tuple[float, dict]] = {}


def _quote_context(ticker: str) -> dict:
    """Best-effort {marketCap, high52w, low52w} — never raises, never blocks
    the quote on failure."""
    with _lock:
        hit = _ctx_cache.get(ticker)
    if hit and (time.time() - hit[0]) < _CTX_TTL:
        return hit[1]
    ctx: dict = {}
    try:
        from sec_mcp import polygon_client
        details = polygon_client.get_ticker_details(ticker)
        if details and details.get("market_cap"):
            ctx["marketCap"] = details["market_cap"]
    except Exception:
        pass
    try:
        from sec_mcp.surface.history import get_price_history_impl
        hist = get_price_history_impl(ticker, "1Y", "day")["summary"]
        ctx["high52w"], ctx["low52w"] = hist.get("high"), hist.get("low")
    except Exception:
        pass
    with _lock:
        _ctx_cache[ticker] = (time.time(), ctx)
    return ctx

# Surface-level quote cache: {ticker: (fetched_unix, payload)} — TTL depends
# on the session at READ time, so a quote cached during "regular" expires in
# 30s but the same entry read while "closed" can serve for an hour.
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _fetch_quote(ticker: str) -> dict:
    """Hit the provider chain; raise structured error if every provider fails."""
    # get_realtime_price already normalizes BRK.B → BRK-B and tries 3 providers
    raw = get_realtime_price(ticker)
    # All-providers-failed → structured upstream error (NOT "market closed")
    if raw.get("error") or raw.get("price") is None:
        raise ToolError(
            UPSTREAM_ERROR,
            f"No price available for {ticker} from any provider (Polygon, yfinance, FMP).",
            "Check the ticker spelling; delisted tickers have no live quote — "
            "use get_filings/get_fundamentals for historical data instead.",
        )
    return raw


def get_quote_impl(ticker) -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()                                       # latency clock
    tk = require_ticker(ticker, "ticker")                  # validate input
    session = market_session()                             # pre/regular/after/closed
    ttl = quote_ttl(session)                               # session-aware freshness budget

    # ── cache read (session-aware TTL) ──────────────────────────────────
    with _lock:
        hit = _cache.get(tk)
    if hit and (time.time() - hit[0]) < ttl:
        fetched_at, payload = hit                          # unpack cache entry
        age = round(time.time() - fetched_at, 1)           # honest staleness
        out = dict(payload)                                # copy — never mutate cache
        out["session"] = session                           # session at READ time
        out["ageSeconds"] = age                            # mandatory: never silent
        out["meta"] = build_meta(payload["provider"], t0, cache_hit=True,
                                 as_of=payload["asOf"])    # meta reflects cache hit
        return out

    # ── live fetch ──────────────────────────────────────────────────────
    raw = _fetch_quote(tk)                                 # raises ToolError on failure
    as_of = raw.get("timestamp") or datetime.now(timezone.utc).isoformat()
    payload = {
        "ticker": raw.get("ticker", tk),                   # normalized symbol
        "price": raw.get("price"),                         # last trade / last close
        "change": raw.get("change"),                       # vs previous close
        "changePct": raw.get("change_pct"),                # percent move
        "volume": raw.get("volume"),                       # day volume
        "marketCap": raw.get("market_cap"),                # when provider supplies it
        "high52w": raw.get("high_52w"),                    # 52-week range context
        "low52w": raw.get("low_52w"),
    }
    # Fill provider gaps from reference data (Polygon snapshot path carries
    # neither market cap nor 52w range) — cached daily, never fatal
    if any(payload[k] is None for k in ("marketCap", "high52w", "low52w")):
        ctx = _quote_context(tk)
        for k in ("marketCap", "high52w", "low52w"):
            if payload[k] is None:
                payload[k] = ctx.get(k)
    payload |= {
        "asOf": as_of,                                     # mandatory metadata
        "session": session,                                # "closed" off-hours — never an error
        "provider": raw.get("source"),                     # which provider answered
        "ageSeconds": round(float(raw.get("cached_age_seconds") or 0.0), 1),
        # Closed-session quotes are last available trade/close — say so explicitly
        "priceBasis": "last_close" if session == "closed" else "last_trade",
    }
    # store for session-aware reuse
    with _lock:
        _cache[tk] = (time.time(), payload)
    out = dict(payload)
    out["meta"] = build_meta(payload["provider"] or "providers", t0,
                             cache_hit=False, as_of=as_of)
    return out
