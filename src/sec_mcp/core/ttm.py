"""TTM metrics builder for the HTTP API.

Sums the newest four STANDALONE quarters from get_fmp_shaped_history into a
snake_case metrics dict that feeds core.ratios.compute_ratios and
core.ticker_metrics.compute_ticker_metrics unmodified. Flows sum (any gap →
honest None), balance items take the latest quarter, and the year-ago quarter
supplies the prior balances for average-balance ROE/ROA.

The MCP surface has its own camelCase TTM path (surface/fundamentals.py
_ttm_from_quarters, a frozen response contract) — same summing rule, different
key domain; keep them in sync if the rule ever changes.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# camelCase history keys → snake_case extraction keys (flows sum for TTM)
_FLOW_MAP = {
    "revenue": "revenue",
    "grossProfit": "gross_profit",
    "operatingIncome": "operating_income",
    "netIncome": "net_income",
    "ebitda": "ebitda",
    "operatingCashFlow": "operating_cash_flow",
    "capex": "capital_expenditures",
    "freeCashFlow": "free_cash_flow",
    # Tax/interest lines feed effective_tax_rate() and the ROIC EBIT proxy.
    "incomeTaxExpense": "income_tax_expense",
    "incomeBeforeTax": "income_before_tax",
    "interestExpense": "interest_expense",
}
# Balance/point-in-time keys → latest quarter wins
_STOCK_MAP = {
    "totalAssets": "total_assets",
    "totalLiabilities": "total_liabilities",
    "totalEquity": "stockholders_equity",
    "cashAndEquivalents": "cash_and_equivalents",
    "totalDebt": "total_debt",
    "sharesOutstanding": "shares_outstanding",
}

# YTD-quality rows are cumulative — summing them double-counts. Only these
# per-quarter qualities are safe TTM addends.
_SUMMABLE_QUALITIES = {"standalone", "standalone_decumulated", "q4_synthesized", None, ""}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _days_between(newer: Any, older: Any) -> int | None:
    from datetime import date
    try:
        return (date.fromisoformat(str(newer)[:10])
                - date.fromisoformat(str(older)[:10])).days
    except (ValueError, TypeError):
        return None


def _clean_quarters(ticker: str,
                    periods: list[dict] | None = None) -> list[dict] | dict:
    """Newest-first standalone quarters, or an {error, hint} dict."""
    if periods is None:
        from sec_mcp.surface.fundamentals import _build_periods
        try:
            periods = _build_periods(ticker, "ttm", 8)
        except Exception as exc:
            # _build_periods raises the surface ToolError on no-data; normalize.
            return {"error": f"No quarterly fundamentals for {ticker}: {exc}",
                    "hint": "Use period=annual for this company."}
    return [p for p in periods if p.get("quality") in _SUMMABLE_QUALITIES]


def _year_ago_prior(clean: list[dict], as_of: Any) -> dict[str, float | None] | None:
    """Balances from the quarter ~4 back, only when it truly sits one year
    before as_of (330-400d) — misaligned rows would poison avg-balance ROE."""
    if len(clean) < 5:
        return None
    ya = clean[4]
    gap = _days_between(as_of, ya.get("endDate"))
    if gap is None or not (330 <= gap <= 400):
        return None
    m = ya["metrics"]
    return {
        "stockholders_equity": _num(m.get("totalEquity")),
        "total_assets": _num(m.get("totalAssets")),
    }


def _add_aliases(metrics: dict[str, float | None]) -> dict[str, float | None]:
    """Frozen-shape consumers (fineasai comps mapper) read total_equity and
    net_debt off the annual extraction dict — emit them here too so a TTM/
    quarter overlay doesn't silently null EV, EV/EBITDA and P/B downstream."""
    if metrics.get("total_equity") is None:
        metrics["total_equity"] = metrics.get("stockholders_equity")
    td = metrics.get("total_debt")
    if metrics.get("net_debt") is None and td is not None:
        metrics["net_debt"] = td - (metrics.get("cash_and_equivalents") or 0.0)
    return metrics


def _snake_metrics(q: dict) -> dict[str, float | None]:
    """One period's camelCase metrics → snake_case extraction keys."""
    src = q["metrics"]
    out = {snake: _num(src.get(camel)) for camel, snake in _FLOW_MAP.items()}
    for camel, snake in _STOCK_MAP.items():
        out[snake] = _num(src.get(camel))
    if out.get("capital_expenditures") is not None:
        out["capital_expenditures"] = abs(out["capital_expenditures"])
    return _add_aliases(out)


def build_latest_quarter_metrics(ticker: str,
                                 periods: list[dict] | None = None) -> dict[str, Any]:
    """Latest STANDALONE quarter as a snake_case metrics block.

    The as-reported 10-Q path serves YTD flows for Q2/Q3 (six/nine months) —
    fine for statements, wrong for a "Quarterly" metrics view. This uses the
    same decumulated standalone quarters as the TTM builder. `prior` = same
    quarter last year (average-balance ROE + honest YoY basis).
    """
    clean = _clean_quarters(ticker, periods)
    if isinstance(clean, dict):
        return clean
    if not clean:
        return {"error": f"No standalone quarters for {ticker}",
                "hint": "Use period=annual for this company."}
    q = clean[0]
    # Cadence guard: a real quarter sits ~91 days after its neighbour. FPI
    # 6-K rows (SAP-style) carry annual figures stamped "standalone" with
    # filing-date spacing — the gap test rejects them. A lone row is only
    # trusted from a 10-Q (quarterly by construction).
    if len(clean) >= 2:
        gap = _days_between(q.get("endDate"), clean[1].get("endDate"))
        if gap is None or not (75 <= gap <= 110):
            return {"error": f"Quarter cadence check failed for {ticker} "
                             f"(gap {gap}d between newest periods)",
                    "hint": "Semiannual/irregular filers have no standalone "
                            "quarters; fall back to period=annual."}
    elif (q.get("formType") or "") != "10-Q":
        return {"error": f"Single non-10-Q period for {ticker} — cannot "
                         "verify it is a standalone quarter",
                "hint": "Fall back to period=annual."}
    metrics = _snake_metrics(q)
    # Row epsDiluted is only trustworthy on plain standalone rows — the eps
    # column skips decumulation/Q4-synthesis, so decumulated rows carry
    # YTD/full-year EPS. Otherwise derive from NI / share count.
    ni, shares = metrics.get("net_income"), metrics.get("shares_outstanding")
    if (q.get("quality") or "standalone") == "standalone":
        eps = _num(q["metrics"].get("epsDiluted"))
    else:
        eps = None
    if eps is None and ni is not None and shares:
        eps = ni / shares
    metrics["eps_diluted"] = eps

    return {
        "metrics": metrics,
        "prior": _year_ago_prior(clean, q.get("endDate")),
        "as_of": q.get("endDate"),
        "fiscal_year": q.get("fiscalYear"),
        "fiscal_period": q.get("fiscalPeriod"),
        "form_type": q.get("formType"),
        "quality": q.get("quality") or "standalone",
    }


def build_ttm_metrics(ticker: str,
                      periods: list[dict] | None = None) -> dict[str, Any]:
    """Build a TTM metrics block. Returns
    {metrics, prior, as_of, fiscal_year, form_type, quality, quarters_used}
    or {error, hint} when fewer than 4 clean standalone quarters exist
    (semiannual FPIs) — callers fall back to annual, never 500.
    """
    clean = _clean_quarters(ticker, periods)
    if isinstance(clean, dict):
        return clean
    if len(clean) < 4:
        return {"error": f"Fewer than 4 standalone quarters for {ticker} "
                         f"({len(clean)} usable)",
                "hint": "Semiannual filers (20-F/6-K) cannot produce TTM; "
                        "fall back to period=annual."}
    qs = clean[:4]

    # TTM window guard: four real quarter-ends span ~273 days (3 quarters).
    # FPI 6-K rows (SAP-style) carry the same annual figure stamped
    # "standalone Q4" with filing-date spacing — summing four of those
    # fabricates 4× annual revenue. Reject anything outside the window.
    span = _days_between(qs[0].get("endDate"), qs[3].get("endDate"))
    if span is None or not (240 <= span <= 300):
        return {"error": f"TTM window check failed for {ticker} — the 4 "
                         f"newest 'quarters' span {span} days, not ~273",
                "hint": "Semiannual/irregular filers cannot produce TTM; "
                        "fall back to period=annual."}

    metrics: dict[str, float | None] = {}
    for camel, snake in _FLOW_MAP.items():
        vals = [_num(q["metrics"].get(camel)) for q in qs]
        metrics[snake] = sum(vals) if all(v is not None for v in vals) else None
    for camel, snake in _STOCK_MAP.items():
        metrics[snake] = _num(qs[0]["metrics"].get(camel))

    # capex arrives negative (FMP sign convention); extraction keys are magnitudes
    if metrics.get("capital_expenditures") is not None:
        metrics["capital_expenditures"] = abs(metrics["capital_expenditures"])
    _add_aliases(metrics)

    # TTM EPS: NI_ttm / latest share count. Summing quarterly epsDiluted rows
    # is only safe when every row is plain "standalone" — decumulated and
    # q4_synthesized rows carry YTD/full-year EPS (the eps column skips
    # decumulation), which inflated TTM P/E 2-3x across the panel. Callers
    # backfill from NI / live share count when this stays None.
    ni = metrics.get("net_income")
    shares = metrics.get("shares_outstanding")
    eps = ni / shares if (ni is not None and shares) else None
    if eps is None and all((q.get("quality") or "standalone") == "standalone" for q in qs):
        q_eps = [_num(q["metrics"].get("epsDiluted")) for q in qs]
        eps = sum(q_eps) if all(v is not None for v in q_eps) else None
    metrics["eps_diluted"] = eps

    return {
        "metrics": metrics,
        # Year-ago quarter balances → average-balance ROE/ROA denominators
        "prior": _year_ago_prior(clean, qs[0].get("endDate")),
        "as_of": qs[0].get("endDate"),
        "fiscal_year": qs[0].get("fiscalYear"),
        "form_type": qs[0].get("formType"),
        "quality": "ttm_sum_4q",
        "quarters_used": [
            {"endDate": q.get("endDate"), "fiscalPeriod": q.get("fiscalPeriod"),
             "quality": q.get("quality")} for q in qs
        ],
    }
