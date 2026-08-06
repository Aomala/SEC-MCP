"""get_price_history — OHLCV history for stocks/ETFs, chart-ready.

Polygon aggs first (adjusted), yfinance fallback — the chart always has
data when the ticker trades. Same window vocabulary as get_index.
"""

from __future__ import annotations

# stdlib
import time
from datetime import datetime, timedelta, timezone

# Polygon aggs — /v2/aggs works identically for stocks and indices
from sec_mcp import polygon_client

# response contract helpers
from sec_mcp.surface.meta import (
    INVALID_INPUT,
    NOT_FOUND,
    ToolError,
    build_meta,
    require_ticker,
)

# window → calendar days to request (buffer over trading days)
_WINDOW_DAYS = {"5D": 8, "1M": 32, "3M": 95, "6M": 185, "1Y": 366, "5Y": 1830}

# window → yfinance period string for the fallback path
_YF_PERIOD = {"5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}

# yfinance uses dashes for class shares (BRK-B), Polygon uses dots (BRK.B)
def _yf_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def _from_polygon(ticker: str, window: str, timespan: str) -> list[dict] | None:
    """Adjusted OHLCV rows from Polygon aggs, oldest→newest."""
    days = _WINDOW_DAYS[window]
    to_d = datetime.now(timezone.utc).date()
    from_d = to_d - timedelta(days=days)
    rows = polygon_client.get_index_aggs(
        polygon_client.normalize_ticker(ticker),
        from_d.isoformat(), to_d.isoformat(), timespan=timespan,
    )
    if not rows:
        return None
    out = []
    for r in rows:
        ts = r.get("t")
        if ts is None or r.get("c") is None:
            continue
        out.append({
            "date": datetime.fromtimestamp(ts / 1000, timezone.utc).date().isoformat(),
            "open": r.get("o"), "high": r.get("h"), "low": r.get("l"),
            "close": r.get("c"), "volume": r.get("v"),
        })
    return out or None


def _from_yfinance(ticker: str, window: str) -> list[dict] | None:
    """Fallback path — daily bars only."""
    try:
        import yfinance as yf
        hist = yf.Ticker(_yf_symbol(ticker)).history(period=_YF_PERIOD[window])
        if hist is None or hist.empty:
            return None
        out = []
        for idx, row in hist.iterrows():
            out.append({
                "date": idx.date().isoformat(),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })
        return out or None
    except Exception:
        return None


def get_price_history_impl(ticker, window: str = "1Y",
                           timespan: str = "day") -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")
    win = str(window or "1Y").upper()
    if win not in _WINDOW_DAYS:
        raise ToolError(INVALID_INPUT, f"Unknown window '{window}'.",
                        f"Use one of: {', '.join(_WINDOW_DAYS)}.")
    span = str(timespan or "day").lower()
    if span not in ("day", "week", "month"):
        raise ToolError(INVALID_INPUT, f"Unknown timespan '{timespan}'.",
                        "Use day, week, or month.")

    rows = _from_polygon(tk, win, span)
    provider = "polygon:aggs"
    if rows is None and span == "day":
        rows = _from_yfinance(tk, win)
        provider = "yfinance"
    if not rows:
        raise ToolError(NOT_FOUND, f"No price history for {tk} over {win}.",
                        "Check the ticker; delisted or OTC symbols may have no bars.")

    closes = [r["close"] for r in rows]
    first, last = closes[0], closes[-1]
    return {
        "ticker": tk,
        "window": win,
        "timespan": span,
        "bars": rows,                                       # oldest → newest
        "chartSeries": {                                    # plot-ready, matches get_index shape
            "labels": [r["date"] for r in rows],
            "closes": closes,
        },
        "summary": {
            "start": rows[0]["date"], "end": rows[-1]["date"],
            "startClose": first, "endClose": last,
            "changePct": round((last - first) / first * 100, 2) if first else None,
            "high": max(r["high"] for r in rows if r["high"] is not None),
            "low": min(r["low"] for r in rows if r["low"] is not None),
            "bars": len(rows),
        },
        "adjusted": True,                                   # splits/dividends adjusted
        "meta": build_meta(provider, t0, cache_hit=False,
                           as_of=rows[-1]["date"]),
    }
