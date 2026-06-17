"""get_fundamentals + compare — XBRL fundamentals, chart-ready, cross-checked.

Source of truth is SEC XBRL companyfacts (via the existing extraction
engine). When a Polygon key is configured, headline metrics are
cross-checked against Polygon's standardized financials and any >2%
disagreement is flagged — never silently merged.

Chart enrichment (per the Fineas charting requirement):
  - `chartSeries`: per-period arrays for revenue / netIncome / freeCashFlow /
    margins — drop straight into a line/bar chart.
  - `segments.geographic` / `segments.product`: latest-period revenue
    breakdowns with percentages — drop straight into a pie/treemap.
  - `segmentSeries.geographic`: multi-period geographic revenue (FMP,
    when available) — stacked-area-ready.
"""

from __future__ import annotations

# stdlib
import logging
import time

# optional cross-check + segment-series providers
from sec_mcp import fmp_client, polygon_client
from sec_mcp.config import get_config

# XBRL extraction engine + multi-period history
from sec_mcp.financials import (
    extract_financials,
    get_fmp_shaped_history,
)

# response contract
from sec_mcp.surface.meta import (
    INVALID_INPUT,
    NOT_FOUND,
    ToolError,
    build_meta,
    require_choice,
    require_pos_int,
    require_ticker,
)

log = logging.getLogger(__name__)

# Canonical metric names callers may request — keys of the per-period dict.
_KNOWN_METRICS = (
    "revenue", "grossProfit", "operatingIncome", "netIncome", "ebitda",
    "eps", "epsDiluted", "totalAssets", "totalLiabilities", "totalEquity",
    "cashAndEquivalents", "totalDebt", "operatingCashFlow", "capex",
    "freeCashFlow", "grossMargin", "operatingMargin", "netMargin",
    "sharesOutstanding",
)

# Flow metrics sum across quarters for TTM; everything else takes latest.
_FLOW_METRICS = {
    "revenue", "grossProfit", "operatingIncome", "netIncome", "ebitda",
    "operatingCashFlow", "capex", "freeCashFlow",
}


def _row_to_metrics(income: dict, balance: dict, cashflow: dict) -> dict:
    """Collapse one period's FMP-shaped statement rows into canonical metrics."""
    # Local helper: first non-None across candidate keys
    def g(row, *keys):
        for k in keys:
            v = row.get(k)
            if v is not None:
                return v
        return None
    rev = g(income, "revenue")
    gp = g(income, "grossProfit")
    oi = g(income, "operatingIncome")
    ni = g(income, "netIncome")
    ocf = g(cashflow, "operatingCashFlow", "netCashProvidedByOperatingActivities")
    capex = g(cashflow, "capitalExpenditure", "investmentsInPropertyPlantAndEquipment")
    # FCF = OCF − |capex| (capex stored negative per FMP convention)
    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else g(cashflow, "freeCashFlow")
    debt_lt = g(balance, "longTermDebt") or 0
    debt_st = g(balance, "shortTermDebt") or 0
    return {
        "revenue": rev,
        "grossProfit": gp,
        "operatingIncome": oi,
        "netIncome": ni,
        "ebitda": g(income, "ebitda"),
        "eps": g(income, "eps"),
        "epsDiluted": g(income, "epsDiluted"),
        "totalAssets": g(balance, "totalAssets"),
        "totalLiabilities": g(balance, "totalLiabilities"),
        "totalEquity": g(balance, "totalStockholdersEquity", "totalEquity"),
        "cashAndEquivalents": g(balance, "cashAndCashEquivalents"),
        "totalDebt": (debt_lt + debt_st) or g(balance, "totalDebt"),
        "operatingCashFlow": ocf,
        "capex": capex,
        "freeCashFlow": fcf,
        # Margins derived in-place so charts get ready-made ratio series
        "grossMargin": round(gp / rev, 4) if (gp is not None and rev) else None,
        "operatingMargin": round(oi / rev, 4) if (oi is not None and rev) else None,
        "netMargin": round(ni / rev, 4) if (ni is not None and rev) else None,
        "sharesOutstanding": g(income, "weightedAverageShsOut"),
    }


def _build_periods(ticker: str, period: str, periods_back: int) -> list[dict]:
    """Fetch N periods of fundamentals, newest first, with provenance."""
    # TTM needs 4 standalone quarters regardless of periods_back
    fetch_kind = "quarter" if period in ("quarterly", "ttm") else "annual"
    n = max(periods_back, 4) if period == "ttm" else periods_back
    hist = get_fmp_shaped_history(ticker, period=fetch_kind, limit=n)
    income, balance, cashflow = hist["income"], hist["balance"], hist["cashflow"]
    if not income:
        raise ToolError(NOT_FOUND, f"No XBRL fundamentals found for {ticker}.",
                        "Recent IPOs may have no periodic filings yet; check "
                        "get_filings(ticker) — fundamentals appear after the first 10-Q/10-K.")
    periods: list[dict] = []
    for i, inc in enumerate(income):
        bal = balance[i] if i < len(balance) else {}
        cf = cashflow[i] if i < len(cashflow) else {}
        periods.append({
            "endDate": inc.get("date"),                    # period end (report date)
            "fiscalYear": inc.get("fiscalYear"),
            "fiscalPeriod": inc.get("period"),             # FY / Q1..Q4
            "currency": inc.get("reportedCurrency") or "USD",  # filing currency (FPIs ≠ USD)
            "accession": (inc.get("_meta") or {}).get("accession"),
            # Source filing form (10-K/10-Q/20-F/6-K) so callers can label the period
            "formType": (inc.get("_meta") or {}).get("formType"),
            "quality": (inc.get("_meta") or {}).get("quality"),
            "metrics": _row_to_metrics(inc, bal, cf),
        })
    return periods


def _ttm_from_quarters(periods: list[dict]) -> dict:
    """Collapse the latest 4 standalone quarters into one TTM period."""
    qs = periods[:4]                                       # newest-first already
    if len(qs) < 4:
        raise ToolError(NOT_FOUND, "Fewer than 4 quarters available — cannot compute TTM.",
                        "Use period='annual' or period='quarterly' for this company.")
    ttm_metrics: dict = {}
    for name in _KNOWN_METRICS:
        vals = [q["metrics"].get(name) for q in qs]
        if name in _FLOW_METRICS:
            # Flows sum across the four quarters; any gap → honest None
            ttm_metrics[name] = round(sum(vals), 2) if all(v is not None for v in vals) else None
        else:
            # Stocks (balance items, shares) take the most recent quarter
            ttm_metrics[name] = vals[0]
    # Re-derive margins from the summed flows for consistency
    rev = ttm_metrics.get("revenue")
    for num, key in (("grossProfit", "grossMargin"), ("operatingIncome", "operatingMargin"),
                     ("netIncome", "netMargin")):
        v = ttm_metrics.get(num)
        ttm_metrics[key] = round(v / rev, 4) if (v is not None and rev) else None
    return {
        "endDate": qs[0]["endDate"],                       # TTM window ends at latest quarter
        "fiscalYear": qs[0]["fiscalYear"],
        "fiscalPeriod": "TTM",
        "currency": qs[0].get("currency") or "USD",        # carry filing currency
        "accession": qs[0]["accession"],
        # TTM is summed from quarterlies — surface the underlying form (10-Q/6-K)
        "formType": qs[0].get("formType"),
        "quality": "ttm_sum_4q",                           # provenance flag
        "metrics": ttm_metrics,
    }


def _cross_check(ticker: str, latest: dict) -> dict:
    """Compare headline metrics against Polygon's standardized financials.

    Periods are ALIGNED before comparing: we pick the Polygon report whose
    end date sits within 10 days of our period end — comparing our FY
    against Polygon's TTM would flag false mismatches.
    """
    # No key → explicitly say so rather than pretending we verified
    if not get_config().polygon_api_key:
        return {"provider": "polygon", "status": "unavailable",
                "detail": "POLYGON_API_KEY not configured."}
    try:
        rows = polygon_client.get_financials(ticker, limit=8) or []
        if not rows:
            return {"provider": "polygon", "status": "unavailable",
                    "detail": "Polygon returned no financials."}
        # Align by period: same end date (±10 days) and an annual timeframe
        # when our period is a fiscal year
        from datetime import date as _date
        our_end = str(latest.get("endDate") or "")[:10]
        want_annual = str(latest.get("fiscalPeriod") or "").upper() in ("FY", "TTM")
        match = None
        for r in rows:
            tf = (r.get("timeframe") or "").lower()
            if want_annual and tf not in ("annual", "ttm"):
                continue                                   # skip quarterly rows
            r_end = str(r.get("end_date") or "")[:10]
            if our_end and r_end:
                try:
                    delta = abs((_date.fromisoformat(our_end)
                                 - _date.fromisoformat(r_end)).days)
                except ValueError:
                    continue
                if delta <= 10:                            # same reporting period
                    match = r
                    break
        if match is None:
            return {"provider": "polygon", "status": "unavailable",
                    "detail": f"No Polygon report aligned with period ending {our_end}."}
        fin = match.get("financials") or {}
        inc = fin.get("income_statement") or {}
        checks: list[dict] = []
        # Compare revenue + net income (the two metrics charts hang on)
        for ours_key, theirs_key in (("revenue", "revenues"),
                                     ("netIncome", "net_income_loss")):
            ours = (latest.get("metrics") or {}).get(ours_key)
            theirs = ((inc.get(theirs_key) or {}).get("value")
                      if isinstance(inc.get(theirs_key), dict) else inc.get(theirs_key))
            if ours is None or theirs in (None, 0):
                continue                                   # nothing to compare
            delta_pct = abs(ours - theirs) / abs(theirs) * 100
            checks.append({"metric": ours_key, "sec": ours, "polygon": theirs,
                           "deltaPct": round(delta_pct, 2),
                           "ok": delta_pct <= 2.0})        # 2% tolerance
        status = "ok" if checks and all(c["ok"] for c in checks) else \
                 ("mismatch" if checks else "unavailable")
        return {"provider": "polygon", "status": status, "checks": checks}
    except Exception as exc:                               # cross-check never breaks the tool
        log.debug("polygon cross-check failed for %s: %s", ticker, exc)
        return {"provider": "polygon", "status": "unavailable", "detail": str(exc)[:120]}


def _segments_block(ticker: str, period: str = "annual") -> dict:
    """Latest geographic + product revenue split, chart-ready, plus a
    multi-period geographic series when FMP is configured.

    Primary source: dimensional XBRL parsed from the filing itself
    (graph/segments.py) — companyfacts strips dimensions, so that's the
    only authoritative segment source. `period` selects the source filing:
    annual → 10-K/20-F, quarterly/ttm → 10-Q/6-K. `sourceMeta` echoes which
    filing the breakdown came from (formType/fiscalPeriod/reportDate/currency).
    """
    # Map the requested period to the segment source filing's form
    seg_form = "10-Q" if period in ("quarterly", "ttm") else "10-K"
    seg_fmp_period = "quarter" if period in ("quarterly", "ttm") else "annual"
    out: dict = {"geographic": [], "product": [], "segmentSeries": None,
                 "sourceMeta": None}
    # Total revenue gives each slice its percentage (chart labels)
    import contextlib
    total = None
    with contextlib.suppress(Exception):                   # pct is best-effort
        total = (extract_financials(ticker) or {}).get("metrics", {}).get("revenue")

    def _shape(rows: list[dict]) -> list[dict]:
        """[{segment, value}] → [{name, value, pct}] sorted big → small."""
        shaped = [{"name": r.get("segment") or r.get("name"),
                   "value": r.get("value"),
                   "pct": round(r["value"] / total * 100, 1)
                          if (total and r.get("value") is not None) else None}
                  for r in rows if r.get("value") is not None]
        shaped.sort(key=lambda s: -(s["value"] or 0))
        return shaped

    try:
        # Dimensional XBRL from the period's filing (authoritative)
        from sec_mcp.graph.segments import get_dimensional_segments
        dims = get_dimensional_segments(ticker, form_type=seg_form) or {}
        out["product"] = _shape(dims.get("segments") or [])
        out["geographic"] = _shape(dims.get("geographic_segments") or [])
        out["sourceMeta"] = dims.get("source_meta")        # which filing it came from
    except Exception as exc:
        log.debug("dimensional segments failed for %s: %s", ticker, exc)
    try:
        # FMP fallback fills whichever breakdown XBRL didn't provide
        if (not out["geographic"] or not out["product"]) and get_config().fmp_api_key:
            if not out["geographic"]:
                geo = fmp_client.get_geo_segments(ticker, period=seg_fmp_period, limit=1) or []
                if geo:
                    out["geographic"] = _shape(
                        [{"name": s.get("name"), "value": s.get("value")}
                         for s in (geo[0].get("segments") or [])])
            if not out["product"]:
                prod = fmp_client.get_product_segments(ticker, period=seg_fmp_period, limit=1) or []
                if prod:
                    out["product"] = _shape(
                        [{"name": s.get("name"), "value": s.get("value")}
                         for s in (prod[0].get("segments") or [])])
    except Exception as exc:
        log.debug("FMP segment fallback failed for %s: %s", ticker, exc)
    try:
        # Multi-period geographic series (FMP) — feeds stacked-area charts
        if get_config().fmp_api_key:
            series = fmp_client.get_geo_segments(ticker, period=seg_fmp_period, limit=5) or []
            if series:
                out["segmentSeries"] = {"geographic": series, "source": "fmp"}
    except Exception as exc:
        log.debug("FMP geo series failed for %s: %s", ticker, exc)
    return out


def _chart_series(periods: list[dict]) -> dict:
    """Per-metric arrays (oldest → newest) ready for direct charting."""
    ordered = list(reversed(periods))                      # charts read left-to-right
    labels = [f"{p.get('fiscalPeriod')} {p.get('fiscalYear')}" for p in ordered]
    series = {}
    for name in ("revenue", "netIncome", "freeCashFlow", "grossMargin",
                 "operatingMargin", "netMargin"):
        series[name] = [p["metrics"].get(name) for p in ordered]
    return {"labels": labels, "endDates": [p.get("endDate") for p in ordered], **series}


def get_fundamentals_impl(ticker, period=None, metrics=None, periods_back=None,
                          include_segments=True) -> dict:
    """Core implementation for the get_fundamentals tool."""
    t0 = time.time()                                       # latency clock
    tk = require_ticker(ticker, "ticker")
    period = require_choice(period, "period", ("annual", "quarterly", "ttm"),
                            default="annual")
    periods_back = require_pos_int(periods_back, "periods_back", default=4, hi=12)
    # Validate the requested metric names before doing any work
    if metrics is not None:
        if not isinstance(metrics, (list, tuple)) or not all(isinstance(m, str) for m in metrics):
            raise ToolError(INVALID_INPUT, "'metrics' must be a list of metric names.",
                            f"Known metrics: {list(_KNOWN_METRICS)}.")
        bad = [m for m in metrics if m not in _KNOWN_METRICS]
        if bad:
            raise ToolError(INVALID_INPUT, f"Unknown metric(s): {bad}.",
                            f"Known metrics: {list(_KNOWN_METRICS)}.")

    periods = _build_periods(tk, period, periods_back)     # raises NOT_FOUND if empty
    if period == "ttm":
        periods = [_ttm_from_quarters(periods)]            # single synthetic TTM period
    # Trim each period to the requested metric subset (after TTM math)
    if metrics:
        for p in periods:
            p["metrics"] = {k: p["metrics"].get(k) for k in metrics}

    out = {
        "ticker": tk,
        "period": period,
        "periods": periods,                                # newest first
        "crossCheck": _cross_check(tk, periods[0]),        # provider verification
        "chartSeries": _chart_series(periods),             # chart-ready arrays
        "meta": build_meta("edgar:xbrl_companyfacts", t0, cache_hit=False,
                           as_of=periods[0].get("endDate")),
    }
    # Segment breakdowns are chart gold but cost an extra extraction — optional.
    # Pass period so quarterly views pull the latest 10-Q/6-K segments.
    if include_segments:
        out["segments"] = _segments_block(tk, period=period)
    return out


def compare_impl(tickers, metrics=None, period=None) -> dict:
    """Core implementation for the compare tool — normalized side-by-side."""
    t0 = time.time()                                       # latency clock
    # Validate the ticker list shape before any network work
    if not isinstance(tickers, (list, tuple)) or not (2 <= len(tickers) <= 8):
        raise ToolError(INVALID_INPUT, "'tickers' must be a list of 2-8 symbols.",
                        "Example: compare(['AAPL', 'MSFT'], ['revenue', 'netMargin']).")
    tks = [require_ticker(t, "tickers[]") for t in tickers]
    period = require_choice(period, "period", ("annual", "ttm"), default="annual")
    metrics = metrics or ["revenue", "netIncome", "grossMargin", "netMargin",
                          "freeCashFlow", "totalDebt"]
    bad = [m for m in metrics if m not in _KNOWN_METRICS]
    if bad:
        raise ToolError(INVALID_INPUT, f"Unknown metric(s): {bad}.",
                        f"Known metrics: {list(_KNOWN_METRICS)}.")

    rows: list[dict] = []
    failures: list[dict] = []
    for tk in tks:                                         # serial keeps EDGAR happy;
        try:                                               # extraction is cached anyway
            fund = get_fundamentals_impl(tk, period=period, metrics=metrics,
                                         periods_back=4, include_segments=False)
            latest = fund["periods"][0]
            rows.append({
                "ticker": tk,
                "endDate": latest.get("endDate"),          # so callers see period alignment
                "fiscalPeriod": f"{latest.get('fiscalPeriod')} {latest.get('fiscalYear')}",
                "values": latest["metrics"],
            })
        except ToolError as exc:                           # partial results beat all-or-nothing
            failures.append({"ticker": tk, "error": exc.message, "code": exc.code})
    if not rows:
        raise ToolError(NOT_FOUND, "No fundamentals available for any requested ticker.",
                        f"Failures: {failures}.")
    return {
        "period": period,
        "metrics": metrics,
        # Normalization note: each company's latest completed period — fiscal
        # calendars differ, so endDate is surfaced per row for honesty
        "normalization": "latest completed fiscal period per company (see endDate per row)",
        "rows": rows,
        "failures": failures or None,
        "meta": build_meta("edgar:xbrl_companyfacts", t0, cache_hit=False),
    }
