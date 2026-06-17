"""Query + chart endpoints for the SEC data engine.

GET /api/v1/metrics/{ticker}/{metric}   — one canonical metric as a time series
                                          with full provenance (tag, filing,
                                          confidence, quality flags)
GET /api/v1/chart-data/{ticker}         — chart-ready {labels, series[]} for
                                          one or more metrics
GET /api/v1/concepts/{ticker}/{accession} — the filing's calculation /
                                          presentation trees (the concept
                                          graph behind every number)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from sec_mcp.financials import get_fmp_shaped_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["metrics"])

# canonical metric key → (statement, FMP field) in the shaped history rows
_METRIC_FIELDS: dict[str, tuple[str, str]] = {
    "revenue": ("income", "revenue"),
    "cost_of_revenue": ("income", "costOfRevenue"),
    "gross_profit": ("income", "grossProfit"),
    "operating_income": ("income", "operatingIncome"),
    "net_income": ("income", "netIncome"),
    "ebitda": ("income", "ebitda"),
    "eps": ("income", "eps"),
    "eps_diluted": ("income", "epsDiluted"),
    "rd_expense": ("income", "researchAndDevelopmentExpenses"),
    "sga_expense": ("income", "sellingGeneralAndAdministrativeExpenses"),
    "total_assets": ("balance", "totalAssets"),
    "total_liabilities": ("balance", "totalLiabilities"),
    "stockholders_equity": ("balance", "totalStockholdersEquity"),
    "cash_and_equivalents": ("balance", "cashAndCashEquivalents"),
    "total_debt": ("balance", "totalDebt"),
    "net_debt": ("balance", "netDebt"),
    "inventory": ("balance", "inventory"),
    "goodwill": ("balance", "goodwill"),
    "operating_cash_flow": ("cashflow", "operatingCashFlow"),
    "capital_expenditures": ("cashflow", "capitalExpenditure"),
    "free_cash_flow": ("cashflow", "freeCashFlow"),
    "dividends_paid": ("cashflow", "commonDividendsPaid"),
    "shares_repurchased": ("cashflow", "commonStockRepurchased"),
    "stock_based_compensation": ("cashflow", "stockBasedCompensation"),
}


def _series_for(ticker: str, metrics: list[str], period: str, limit: int) -> dict:
    """Shared series builder for /metrics and /chart-data."""
    unknown = [m for m in metrics if m not in _METRIC_FIELDS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric(s): {unknown}. Available: {sorted(_METRIC_FIELDS)}",
        )
    hist = get_fmp_shaped_history(
        ticker.upper(),
        period="quarter" if period in ("quarter", "quarterly") else "annual",
        limit=limit,
    )
    by_stmt = {
        "income": {r.get("date"): r for r in hist.get("income") or []},
        "balance": {r.get("date"): r for r in hist.get("balance") or []},
        "cashflow": {r.get("date"): r for r in hist.get("cashflow") or []},
    }
    dates = sorted({d for rows in by_stmt.values() for d in rows if d})
    series = []
    for metric in metrics:
        stmt, field = _METRIC_FIELDS[metric]
        points = []
        for d in dates:
            row = by_stmt[stmt].get(d) or {}
            meta = row.get("_meta") or {}
            points.append({
                "date": d,
                "value": row.get(field),
                "period": row.get("period"),
                "fiscalYear": row.get("fiscalYear"),
                "quality": meta.get("quality"),
                "isYtd": meta.get("isYtd", False),
                "source": (meta.get("sources") or {}).get(metric),
                "confidence": (meta.get("confidence") or {}).get(metric),
                "accession": meta.get("accession"),
            })
        series.append({"metric": metric, "statement": stmt, "points": points})
    return {"ticker": ticker.upper(), "period": period, "dates": dates,
            "series": series, "source": "sec"}


@router.get("/metrics/{ticker}/{metric}")
def get_metric_series(
    ticker: str,
    metric: str,
    period: str = Query("annual", pattern="^(annual|quarter|quarterly)$"),
    limit: int = Query(8, ge=1, le=12),
):
    """One canonical metric over time, with provenance on every point."""
    data = _series_for(ticker, [metric], period, limit)
    s = data["series"][0]
    return {"ticker": data["ticker"], "metric": metric, "period": period,
            "points": s["points"], "statement": s["statement"], "source": "sec"}


@router.get("/chart-data/{ticker}")
def get_chart_data(
    ticker: str,
    metrics: str = Query("revenue,net_income", description="comma-separated metric keys"),
    period: str = Query("annual", pattern="^(annual|quarter|quarterly)$"),
    limit: int = Query(8, ge=1, le=12),
):
    """Chart-ready shape: labels (dates) + one series per metric."""
    keys = [m.strip() for m in metrics.split(",") if m.strip()]
    if not keys:
        raise HTTPException(status_code=400, detail="No metrics requested")
    data = _series_for(ticker, keys, period, limit)
    return {
        "ticker": data["ticker"],
        "period": period,
        "labels": data["dates"],
        "series": [
            {"name": s["metric"],
             "data": [p["value"] for p in s["points"]],
             "quality": [p["quality"] for p in s["points"]]}
            for s in data["series"]
        ],
        "source": "sec",
    }


@router.get("/concepts/{ticker}/{accession}")
def get_filing_concepts(ticker: str, accession: str):
    """The filing's calculation + presentation trees — the concept graph
    behind the numbers. This is the 'show me WHY this value' endpoint and
    the debugging surface for bad matches."""
    from sec_mcp.graph import store
    from sec_mcp.graph.filing_parser import parse_filing_graph

    fg = store.load_graph(accession)
    if fg is None:
        fg = parse_filing_graph(ticker.upper(), accession)
        if fg is not None:
            store.save_graph(fg)
    if fg is None:
        raise HTTPException(
            status_code=404,
            detail=f"No XBRL linkbases parseable for {accession}",
        )
    return {"ticker": ticker.upper(), "accession": accession,
            "calc": fg.calc, "pres": fg.pres,
            "parser_version": fg.parser_version}


# ═══════════════════════════════════════════════════════════════════════════
#  /api/v2 — overview endpoint backing the explorer's animated dashboard
# ═══════════════════════════════════════════════════════════════════════════

# Separate router so v2 surface-backed endpoints don't mix with the v1
# shaped-history endpoints above (different data contracts).
router_v2 = APIRouter(prefix="/api/v2", tags=["overview"])


@router_v2.get("/overview/{ticker}")
def get_overview_v2(
    ticker: str,
    periods: int = Query(4, ge=2, le=12),
    period: str = Query("annual", pattern="^(annual|quarterly|ttm)$"),
):
    """Everything the animated company dashboard needs in ONE call:

    quote (session-aware), latest-period metrics, chartSeries (plot-ready),
    geographic + product segments (pie/choropleth-ready), and the Polygon
    cross-check. All shapes come verbatim from the MCP v2 surface — the
    explorer renders them without any reshaping (Skill 20 contract).

    `period` (annual|quarterly|ttm) drives which filing the figures and
    segments come from, so the dashboard can switch annual↔quarterly.

    Sync `def` on purpose: FastAPI runs it in a worker thread, so the
    blocking EDGAR/provider I/O never stalls the event loop.
    """
    # Surface implementations already return structured {error, code, hint}
    # dicts on failure — pass them through rather than 500ing.
    from sec_mcp.surface.fundamentals import get_fundamentals_impl
    from sec_mcp.surface.meta import ToolError
    from sec_mcp.surface.quotes import get_quote_impl

    tk = ticker.strip().upper()

    # Quote is best-effort — a delisted ticker still deserves fundamentals
    try:
        quote = get_quote_impl(tk)
    except ToolError as exc:
        quote = {"error": exc.message, "code": exc.code, "hint": exc.hint}

    # Fundamentals + segments are the core payload; structured errors bubble
    # up as a 404/400 with the surface's own hint text
    try:
        fund = get_fundamentals_impl(tk, period=period, periods_back=periods,
                                     include_segments=True)
    except ToolError as exc:
        code = 404 if exc.code in ("NOT_FOUND", "UNKNOWN_TICKER") else 400
        raise HTTPException(status_code=code,
                            detail=f"{exc.message} {exc.hint}") from None

    return {
        "ticker": tk,
        "quote": quote,                          # incl. session / ageSeconds / priceBasis
        "latest": fund["periods"][0],            # newest period + metrics + accession
        "chartSeries": fund["chartSeries"],      # oldest→newest arrays, plot as-is
        "segments": fund.get("segments") or {},  # geographic / product, name+value+pct
        "crossCheck": fund.get("crossCheck"),    # provenance pill feeds off this
        "meta": fund.get("meta"),
    }
