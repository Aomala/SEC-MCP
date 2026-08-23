"""The public, sellable REST API — `/v1`.

Design rule: this file contains NO business logic. Every route is a thin HTTP
shell over the `surface/*_impl` functions that already back the 18 MCP tools.
That is deliberate — it means the REST API and the MCP server cannot drift
apart, are fixed by the same bug fix, and carry the identical response contract
(`meta` on success, `{error, code, hint}` on failure) that surface/meta.py
enforces. One engine, two storefronts.

What this file DOES own:
  - the URL vocabulary a customer sees and pays for
  - which plan scope each endpoint belongs to
  - plan-dependent shaping (history depth, bulk access)
  - the bulk endpoints, which have no MCP equivalent

Versioning promise: `/v1` response keys are additive-only. New fields may
appear; existing ones never change type or disappear. Breaking changes get a
`/v2`.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sec_mcp.commerce import plans as plan_catalogue
from sec_mcp.commerce import usage
from sec_mcp.commerce.auth import Caller, require

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


def _err(message: str, code: str, hint: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": message, "code": code, "hint": hint})


def _status_for(payload: dict) -> int:
    """Map the surface layer's error codes onto HTTP status codes.

    The impls return a 200-shaped dict with an `error` key rather than raising,
    so without this every failure would leave as HTTP 200 — which breaks every
    generated client and every retry policy a customer writes.
    """
    if not isinstance(payload, dict) or "error" not in payload:
        return 200
    return {
        "NOT_FOUND": 404,
        "INVALID_INPUT": 422,
        "RATE_LIMITED": 429,
        "UPSTREAM_ERROR": 502,
        "INTERNAL": 500,
    }.get(str(payload.get("code", "")).upper(), 400)


def _respond(payload: dict) -> JSONResponse | dict:
    status = _status_for(payload)
    return payload if status == 200 else JSONResponse(status_code=status, content=payload)


def _trim_history(payload: dict, caller: Caller) -> dict:
    """Enforce the plan's history depth.

    History depth is the cleanest upgrade lever we have: every tier gets the
    same accurate data, cheaper tiers just see fewer years of it. Applied here
    rather than in the impls so the engine stays billing-unaware.
    """
    years = caller.history_years
    if not years or not isinstance(payload, dict):
        return payload
    periods = payload.get("periods")
    if isinstance(periods, list) and len(periods) > years:
        payload = {**payload, "periods": periods[:years]}
        payload.setdefault("truncated", {})
        payload["truncated"] = {
            "reason": "plan_history_limit",
            "historyYears": years,
            "plan": caller.plan_name,
            "hint": "Upgrade for full history — see /v1/plans.",
        }
    return payload


# ═══════════════════════════════════════════════════════════════════════════
#  Discovery — what the API is, what it costs, who you are
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/", summary="API root — capabilities and links")
def v1_root() -> dict:
    return {
        "name": "Fineas Financial Data API",
        "version": "1.0",
        "description": (
            "As-reported SEC XBRL fundamentals with provenance, plus market "
            "data, ownership, and segment breakdowns."
        ),
        "docs": "/docs",
        "openapi": "/openapi.json",
        "plans": "/v1/plans",
        "mcp": "Same data is available over MCP — see /docs#mcp",
        "auth": "Send your key as 'X-API-Key: <key>' or 'Authorization: Bearer <key>'.",
    }


@router.get("/plans", summary="Price ladder and per-tier limits")
def v1_plans() -> dict:
    return {"plans": plan_catalogue.public_catalogue(),
            "scopes": list(plan_catalogue.ALL_SCOPES)}


@router.get("/me", summary="Your key, plan, and current usage")
def v1_me(request: Request, caller: Caller = Depends(require("search"))) -> dict:
    body = {
        "authenticated": caller.authenticated,
        "plan": caller.plan_name,
        "keyPrefix": caller.key_prefix or None,
        "limits": {
            "dailyCalls": caller.plan.get("daily_calls"),
            "requestsPerMinute": caller.plan.get("rpm"),
            "bulkAccess": bool(caller.plan.get("bulk")),
            "historyYears": caller.plan.get("history_years"),
        },
        "remaining": {
            "today": caller.verdict.get("remaining"),
            "thisMinute": caller.verdict.get("rpm_remaining"),
            "resetsAt": caller.verdict.get("reset_epoch"),
        },
    }
    if caller.authenticated:
        body["usage"] = usage.summary_for_key(caller.key_id)
    return body


# ═══════════════════════════════════════════════════════════════════════════
#  Companies + filings
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/companies/search", summary="Search companies by name or ticker")
def v1_company_search(
    q: str = Query(..., description="Name or ticker fragment"),
    limit: int = Query(10, ge=1, le=100),
    sector: str | None = None,
    exchange: str | None = None,
    country: str | None = None,
    sp500: bool | None = None,
    caller: Caller = Depends(require("search")),
):
    from sec_mcp.surface.company_search import search_companies_impl
    filters = {k: v for k, v in
               {"sector": sector, "exchange": exchange, "country": country,
                "sp500": sp500}.items() if v is not None}
    return _respond(search_companies_impl(q, filters or None, limit))


@router.get("/filings/{ticker}", summary="Filing index for a company")
def v1_filings(
    ticker: str,
    form_type: str | None = Query(None, description="10-K, 10-Q, 8-K, 4, 13F …"),
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = Query(20, ge=1, le=200),
    caller: Caller = Depends(require("filings")),
):
    from sec_mcp.surface.filings import get_filings_impl
    return _respond(get_filings_impl(ticker, form_type, date_from, date_to, limit=limit))


@router.get("/filings/{ticker}/sections/{section}",
            summary="Extracted section text from a filing")
def v1_filing_section(
    ticker: str,
    section: str,
    accession: str = Query(..., description="Accession number of the filing"),
    max_length: int = Query(50_000, ge=1_000, le=500_000),
    caller: Caller = Depends(require("filings")),
):
    from sec_mcp.surface.filings import get_filing_section_impl
    return _respond(get_filing_section_impl(accession, section, ticker, max_length))


# ═══════════════════════════════════════════════════════════════════════════
#  Fundamentals — the core product
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/fundamentals/{ticker}",
            summary="As-reported financials with provenance and cross-check")
def v1_fundamentals(
    ticker: str,
    period: str = Query("annual", pattern="^(annual|quarterly|ttm)$"),
    periods_back: int = Query(4, ge=1, le=12),
    metrics: str | None = Query(None, description="Comma-separated metric subset"),
    include_segments: bool = True,
    caller: Caller = Depends(require("fundamentals")),
):
    from sec_mcp.surface.fundamentals import get_fundamentals_impl
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
    payload = get_fundamentals_impl(ticker, period, metric_list, periods_back,
                                    include_segments)
    return _respond(_trim_history(payload, caller))


@router.get("/metrics/{ticker}", summary="Valuation multiples and return ratios")
def v1_metrics(
    ticker: str,
    period: str = Query("ttm", pattern="^(annual|quarterly|ttm)$"),
    caller: Caller = Depends(require("fundamentals")),
):
    from sec_mcp.surface.valuation import get_valuation_impl
    return _respond(get_valuation_impl(ticker, period))


@router.get("/fundamentals/{ticker}/diff", summary="Year-over-year fundamental deltas")
def v1_diff(
    ticker: str,
    year1: int = Query(..., description="Earlier fiscal year"),
    year2: int = Query(..., description="Later fiscal year"),
    form_type: str = "10-K",
    caller: Caller = Depends(require("fundamentals")),
):
    from sec_mcp.surface.diff import diff_fundamentals_impl
    return _respond(diff_fundamentals_impl(ticker, year1, year2, form_type))


@router.get("/compare", summary="Normalized side-by-side across 2-8 companies")
def v1_compare(
    tickers: str = Query(..., description="Comma-separated, 2-8 symbols"),
    metrics: str | None = None,
    period: str = Query("annual", pattern="^(annual|ttm)$"),
    caller: Caller = Depends(require("fundamentals")),
):
    from sec_mcp.surface.fundamentals import compare_impl
    tk = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not (2 <= len(tk) <= 8):
        return _err("Provide 2-8 comma-separated tickers.", "INVALID_INPUT",
                    "Example: /v1/compare?tickers=AAPL,MSFT,GOOGL", 422)
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
    return _respond(compare_impl(tk, metric_list, period))


@router.get("/segments/{ticker}", summary="Revenue by product/service and geography")
def v1_segments(
    ticker: str,
    period: str = Query("annual", pattern="^(annual|quarterly)$"),
    caller: Caller = Depends(require("segments")),
):
    from sec_mcp.surface.fundamentals import _segments_block
    return _respond({"ticker": ticker.upper(), "period": period,
                     "segments": _segments_block(ticker.upper(), period=period)})


# ═══════════════════════════════════════════════════════════════════════════
#  Market data
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/quotes/{ticker}", summary="Latest quote (session-aware)")
def v1_quote(ticker: str, caller: Caller = Depends(require("market"))):
    from sec_mcp.surface.quotes import get_quote_impl
    return _respond(get_quote_impl(ticker))


@router.get("/prices/{ticker}", summary="OHLCV price history")
def v1_prices(
    ticker: str,
    window: str = Query("1Y", description="1D, 5D, 1M, 6M, 1Y, 5Y, MAX"),
    timespan: str = Query("day", description="minute, hour, day, week, month"),
    caller: Caller = Depends(require("market")),
):
    from sec_mcp.surface.history import get_price_history_impl
    return _respond(get_price_history_impl(ticker, window, timespan))


@router.get("/news/{ticker}", summary="Company news with publisher provenance")
def v1_news(ticker: str, limit: int = Query(10, ge=1, le=50),
            caller: Caller = Depends(require("market"))):
    from sec_mcp.surface.news import get_news_impl
    return _respond(get_news_impl(ticker, limit))


@router.get("/actions/{ticker}", summary="Dividends, splits, and forward yield")
def v1_actions(ticker: str, limit: int = Query(20, ge=1, le=100),
               caller: Caller = Depends(require("market"))):
    from sec_mcp.surface.actions import get_corporate_actions_impl
    return _respond(get_corporate_actions_impl(ticker, limit))


@router.get("/indices/{symbol}", summary="Index level and history")
def v1_index(symbol: str, include_history: bool = True,
             window: str = "1Y", caller: Caller = Depends(require("market"))):
    from sec_mcp.surface.indices import get_index_impl
    return _respond(get_index_impl(symbol, include_history, window))


@router.get("/market/overview", summary="Breadth and sector performance")
def v1_market_overview(caller: Caller = Depends(require("market"))):
    from sec_mcp.surface.indices import get_market_overview_impl
    return _respond(get_market_overview_impl())


@router.get("/etf/{ticker}", summary="ETF profile — expense ratio, AUM, holdings")
def v1_etf(ticker: str, caller: Caller = Depends(require("market"))):
    from sec_mcp.surface.etf import get_etf_profile_impl
    return _respond(get_etf_profile_impl(ticker))


# ═══════════════════════════════════════════════════════════════════════════
#  Ownership
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/ownership/{ticker}/insiders", summary="Form 4 insider transactions")
def v1_insiders(
    ticker: str,
    date_from: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    caller: Caller = Depends(require("ownership")),
):
    from sec_mcp.surface.ownership import get_insider_activity_impl
    return _respond(get_insider_activity_impl(ticker, date_from, limit))


@router.get("/ownership/{ticker}/institutional",
            summary="13F holders, activist 13D/G, and holder families")
def v1_institutional(ticker: str, caller: Caller = Depends(require("ownership"))):
    from sec_mcp.surface.ownership import get_ownership_impl
    return _respond(get_ownership_impl(ticker))


# ═══════════════════════════════════════════════════════════════════════════
#  Discovery / screening
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/screen", summary="Composable screener over the covered universe")
async def v1_screen(request: Request, caller: Caller = Depends(require("screen"))):
    from starlette.concurrency import run_in_threadpool

    from sec_mcp.surface.screen import screen_impl
    try:
        body = await request.json()
    except (ValueError, TypeError):
        return _err("Body must be JSON.", "INVALID_INPUT",
                    'Example: {"filters": {"sector": "Technology"}, "limit": 20}', 422)
    filters = body.get("filters") or {}
    limit = int(body.get("limit") or 20)
    return _respond(await run_in_threadpool(screen_impl, filters, limit))


@router.get("/peers/{ticker}", summary="Comparable companies")
def v1_peers(ticker: str, max_peers: int = Query(5, ge=1, le=25),
             caller: Caller = Depends(require("screen"))):
    from sec_mcp.surface.peers import find_peers_impl
    return _respond(find_peers_impl(ticker, max_peers))


# ═══════════════════════════════════════════════════════════════════════════
#  Bulk — the paid differentiator
# ═══════════════════════════════════════════════════════════════════════════
#  Data buyers pay for coverage, not for one ticker at a time. These stream
#  NDJSON so a client can start writing rows immediately and memory stays flat
#  regardless of universe size — a 1,000-ticker JSON array would have to be
#  fully materialized on both ends.

_UNIVERSES = ("sp500", "nasdaq100", "dow30", "covered")


@router.get("/bulk/fundamentals", summary="Stream fundamentals for a whole universe (NDJSON)")
def v1_bulk_fundamentals(
    universe: str = Query("sp500", description=f"One of {', '.join(_UNIVERSES)}"),
    period: str = Query("annual", pattern="^(annual|quarterly|ttm)$"),
    tickers: str | None = Query(None, description="Explicit comma-separated list (overrides universe)"),
    limit: int = Query(1000, ge=1, le=5000),
    caller: Caller = Depends(require("bulk")),
):
    if not caller.plan.get("bulk"):
        return _err("Bulk download requires a Pro plan or higher.", "plan_required",
                    "See /v1/plans — Pro includes full-universe bulk access.", 403)

    from sec_mcp.commerce.bulk import iter_bulk_fundamentals, resolve_universe

    if tickers:
        symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()][:limit]
    else:
        if universe not in _UNIVERSES:
            return _err(f"Unknown universe '{universe}'.", "INVALID_INPUT",
                        f"Use one of: {', '.join(_UNIVERSES)}.", 422)
        symbols = resolve_universe(universe)[:limit]

    if not symbols:
        return _err("No tickers matched that universe.", "NOT_FOUND",
                    "Try universe=sp500 or pass tickers=AAPL,MSFT.", 404)

    started = time.time()

    def _stream():
        # Header line first so a consumer knows what it is reading before the
        # rows arrive — matters for a stream that may run for a while.
        yield json.dumps({
            "_meta": {
                "universe": None if tickers else universe,
                "period": period,
                "count": len(symbols),
                "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "format": "ndjson",
                "note": "One JSON object per line. Rows with an 'error' key were unavailable.",
            }
        }) + "\n"
        served = 0
        for row in iter_bulk_fundamentals(symbols, period):
            served += 1
            yield json.dumps(row, default=str) + "\n"
        log.info("bulk fundamentals: %d/%d rows in %.1fs (plan=%s)",
                 served, len(symbols), time.time() - started, caller.plan_name)

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="fundamentals_{universe}_{period}.ndjson"',
            "X-Bulk-Count": str(len(symbols)),
            "Cache-Control": "no-store",
        },
    )


@router.get("/bulk/universe", summary="The full covered ticker universe")
def v1_bulk_universe(
    universe: str = Query("covered", description=f"One of {', '.join(_UNIVERSES)}"),
    caller: Caller = Depends(require("bulk")),
):
    if not caller.plan.get("bulk"):
        return _err("Bulk download requires a Pro plan or higher.", "plan_required",
                    "See /v1/plans — Pro includes full-universe bulk access.", 403)
    from sec_mcp.commerce.bulk import resolve_universe
    symbols = resolve_universe(universe)
    return {"universe": universe, "count": len(symbols), "tickers": symbols}


@router.get("/coverage", summary="What this API currently has warm")
def v1_coverage(caller: Caller = Depends(require("search"))):
    """Honest coverage reporting — a buyer's first question is always 'how many
    tickers do you actually have, and how fresh is it?'"""
    from sec_mcp.commerce.bulk import coverage_report
    return coverage_report()
