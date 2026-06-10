"""Write-through persistence of resolved metrics (audit trail + query store).

Every successful extraction records one row per metric in
`metric_observations`: value, source XBRL tag, resolution method, confidence,
quality flag, and the accession it came from — so every number the API serves
can answer "which tag, which filing, what confidence".

All writes are best-effort: without Supabase (or before the concept-graph
migration is applied) they no-op silently.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_OBSERVED_METRICS = (
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "net_income", "ebitda", "eps_basic", "eps_diluted",
    "total_assets", "total_liabilities", "stockholders_equity",
    "minority_interest", "cash_and_equivalents", "long_term_debt",
    "operating_cash_flow", "capital_expenditures", "free_cash_flow",
)


def _period_type_of(result: dict) -> str:
    if result.get("period_type") != "quarterly":
        return "FY"
    label = (result.get("quarter_label") or "").split()
    return label[0].upper() if label else "Q?"


def record_observations(result: dict) -> None:
    """Persist resolved metrics from one extract_financials() result."""
    try:
        from sec_mcp.supabase_cache import _get_client
        client = _get_client()
    except Exception:
        return
    if client is None:
        return

    fi = result.get("filing_info") or {}
    period_end = fi.get("report_date") or ""
    cik = result.get("cik")
    ticker = str(result.get("ticker_or_cik") or "").upper()
    if not (cik and ticker and period_end):
        return

    m = result.get("metrics") or {}
    sources = result.get("metrics_sourced") or {}
    confidence = result.get("confidence_scores") or {}
    ptype = _period_type_of(result)
    fy = result.get("fiscal_year")
    rows = []
    for key in _OBSERVED_METRICS:
        if key not in m:
            continue
        rows.append({
            "cik": cik, "ticker": ticker, "canonical_key": key,
            "period_end": period_end, "period_type": ptype,
            "fiscal_year": fy if isinstance(fy, int) else None,
            "value": m.get(key),
            "currency": result.get("reporting_currency") or "USD",
            "source_concept": sources.get(key),
            "method": None,
            "confidence": confidence.get(key),
            "quality": result.get("quality"),
            "accession": fi.get("accession_number"),
        })
    if not rows:
        return
    try:
        client.table("metric_observations").upsert(
            rows, on_conflict="cik,canonical_key,period_type,period_end"
        ).execute()
    except Exception as exc:
        # Table absent (migration not applied) or transient — never break extraction
        log.debug("metric_observations upsert skipped: %s", exc)
