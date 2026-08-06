"""get_corporate_actions — dividend and split history from Polygon reference data.

Dividends carry the full date chain (declaration → ex-date → record → pay)
plus an annualized-yield estimate when a live quote is available. Splits
carry from/to ratios (SMCI's 2024 10:1 shows as split_from=1, split_to=10).
"""

from __future__ import annotations

# stdlib
import time

# Polygon reference endpoints
from sec_mcp import polygon_client

# response contract helpers
from sec_mcp.surface.meta import (
    UNAVAILABLE,
    ToolError,
    build_meta,
    require_ticker,
)


def _annualized_dividend(dividends: list[dict]) -> float | None:
    """Latest cash amount × frequency — the standard forward-dividend estimate."""
    if not dividends:
        return None
    latest = dividends[0]
    amount, freq = latest.get("cashAmount"), latest.get("frequency")
    if not amount or not freq:
        return None
    return round(float(amount) * int(freq), 4)


def get_corporate_actions_impl(ticker, limit: int = 20) -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")
    n = max(1, min(int(limit or 20), 100))

    if not polygon_client.is_available():
        raise ToolError(UNAVAILABLE, "Corporate actions require a Polygon API key.",
                        "Set POLYGON_API_KEY in the environment.")

    divs = polygon_client.get_dividends(tk, limit=n)
    splits = polygon_client.get_splits(tk, limit=n)

    dividends = [{
        "cashAmount": d.get("cash_amount"),
        "currency": d.get("currency"),
        "declarationDate": d.get("declaration_date"),
        "exDividendDate": d.get("ex_dividend_date"),
        "recordDate": d.get("record_date"),
        "payDate": d.get("pay_date"),
        "frequency": d.get("frequency"),                    # 4 = quarterly, 12 = monthly
        "type": d.get("dividend_type"),                     # CD regular / SC special
    } for d in divs]

    split_rows = [{
        "executionDate": s.get("execution_date"),
        "splitFrom": s.get("split_from"),
        "splitTo": s.get("split_to"),
        "ratio": f"{s.get('split_to')}-for-{s.get('split_from')}",
    } for s in splits]

    # forward yield needs a live price — best effort, never fatal
    fwd_dividend = _annualized_dividend(dividends)
    fwd_yield_pct = None
    if fwd_dividend:
        try:
            from sec_mcp.core.realtime_price import get_realtime_price
            price = (get_realtime_price(tk) or {}).get("price")
            if price:
                fwd_yield_pct = round(fwd_dividend / float(price) * 100, 2)
        except Exception:
            pass

    newest = (dividends[0]["exDividendDate"] if dividends
              else split_rows[0]["executionDate"] if split_rows else None)
    return {
        "ticker": tk,
        "dividends": dividends,                             # newest first
        "splits": split_rows,                               # newest first
        "forwardAnnualDividend": fwd_dividend,
        "forwardYieldPct": fwd_yield_pct,
        "paysDividend": bool(dividends),
        "meta": build_meta("polygon:reference", t0, cache_hit=False, as_of=newest),
    }
