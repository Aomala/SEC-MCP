"""get_valuation — P/E, EV/EBITDA, ROE/ROIC and friends for one ticker.

Delegates to the same engine as /api/metrics/{ticker} (chat_app) so the MCP
surface and the REST dashboard can never disagree: TTM-first multiples,
cross-listing-safe P/E basis, bank/REIT not-applicable handling.
"""

from __future__ import annotations

# stdlib
import asyncio
import time

# response contract helpers
from sec_mcp.surface.meta import (
    INTERNAL,
    INVALID_INPUT,
    NOT_FOUND,
    ToolError,
    build_meta,
    require_choice,
    require_ticker,
)

_ERROR_CODE_MAP = {
    "BAD_PERIOD": INVALID_INPUT,
    "NOT_FOUND": NOT_FOUND,
}


def get_valuation_impl(ticker, period: str = "ttm") -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")
    period = require_choice(period, "period", ("annual", "quarterly", "ttm"),
                            default="ttm")

    # chat_app's handler is async-declared but does sync work — run it here
    from sec_mcp.chat_app import get_ticker_metrics
    result = asyncio.run(get_ticker_metrics(tk, period=period))

    if isinstance(result, dict) and result.get("error"):
        code = _ERROR_CODE_MAP.get(result.get("code"), INTERNAL)
        raise ToolError(code, str(result["error"]),
                        result.get("hint") or "Funds/ETFs have no XBRL "
                        "valuation — use get_etf_profile for those.")

    result["meta"] = build_meta("edgar:xbrl+price", t0, cache_hit=False)
    return result
