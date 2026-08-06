"""diff_fundamentals — year-over-year metric deltas with significance flags.

Wraps the FilingDiffEngine metrics path (no Claude dependency — the
narrative diff lives in server_legacy). Each metric carries both years,
absolute + percent change, and a minor/moderate/major significance label.
"""

from __future__ import annotations

# stdlib
import time
from datetime import datetime, timezone

# stateless metric comparison over extracted financials
from sec_mcp.core.filing_diff import FilingDiffEngine

# response contract helpers
from sec_mcp.surface.meta import (
    INVALID_INPUT,
    NOT_FOUND,
    ToolError,
    build_meta,
    require_ticker,
)


def diff_fundamentals_impl(ticker, year1: int, year2: int,
                           form_type: str = "10-K") -> dict:
    """Core implementation shared by the MCP tool and the test suite."""
    t0 = time.time()
    tk = require_ticker(ticker, "ticker")
    try:
        y1, y2 = int(year1), int(year2)
    except (TypeError, ValueError):
        raise ToolError(INVALID_INPUT, "year1 and year2 must be integers.",
                        "Example: diff_fundamentals('AAPL', 2023, 2024).") from None
    this_year = datetime.now(timezone.utc).year
    if not (1994 <= y1 <= this_year and 1994 <= y2 <= this_year) or y1 == y2:
        raise ToolError(INVALID_INPUT, f"Invalid year pair ({y1}, {y2}).",
                        f"Use two distinct fiscal years between 1994 and {this_year}.")
    if y1 > y2:
        y1, y2 = y2, y1                                     # always older → newer
    ft = str(form_type or "10-K").upper()
    if ft not in ("10-K", "10-Q"):
        raise ToolError(INVALID_INPUT, f"Unsupported form_type '{form_type}'.",
                        "Use 10-K (annual) or 10-Q (quarterly).")

    result = FilingDiffEngine().diff_metrics(tk, y1, y2, form_type=ft)
    if not result or not result.get("metrics"):
        raise ToolError(NOT_FOUND,
                        f"Could not extract comparable {ft} metrics for {tk} "
                        f"in both {y1} and {y2}.",
                        "Recent IPOs lack older filings; try a later year pair.")
    result["meta"] = build_meta("edgar:xbrl_companyfacts", t0, cache_hit=False)
    return result
