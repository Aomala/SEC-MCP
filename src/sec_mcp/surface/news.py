"""get_news — recent per-ticker news with publisher provenance.

Backed by Polygon's news reference feed. Every item carries the publisher
name and URL so downstream consumers (Fineas chat, reports) can cite the
source instead of presenting headlines as unattributed facts.
"""

from __future__ import annotations

# stdlib
import time

# Polygon news feed
from sec_mcp import polygon_client

# response contract helpers
from sec_mcp.surface.meta import (
    NOT_FOUND,
    UNAVAILABLE,
    ToolError,
    build_meta,
    require_ticker,
)


def get_news_impl(ticker, limit: int = 10) -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")
    n = max(1, min(int(limit or 10), 50))

    if not polygon_client.is_available():
        raise ToolError(UNAVAILABLE, "News requires a Polygon API key.",
                        "Set POLYGON_API_KEY in the environment.")

    raw = polygon_client.get_ticker_news(tk, limit=n)
    if not raw:
        raise ToolError(NOT_FOUND, f"No recent news for {tk}.",
                        "Check the ticker spelling; thinly covered tickers may have no articles.")

    articles = []
    for a in raw[:n]:
        pub = a.get("publisher") or {}
        # per-ticker sentiment insight when Polygon provides one
        sentiment = None
        for ins in a.get("insights") or []:
            if (ins.get("ticker") or "").upper() == tk:
                sentiment = ins.get("sentiment")
                break
        articles.append({
            "title": a.get("title"),
            "publisher": pub.get("name"),                   # provenance — always cite
            "publishedUtc": a.get("published_utc"),
            "url": a.get("article_url"),
            "description": a.get("description"),
            "tickers": a.get("tickers"),
            "sentiment": sentiment,                         # positive/neutral/negative or null
        })
    newest = articles[0]["publishedUtc"] if articles else None
    return {
        "ticker": tk,
        "articles": articles,                               # newest first
        "count": len(articles),
        "meta": build_meta("polygon:news", t0, cache_hit=False, as_of=newest),
    }
