"""Multi-year filing retrieval and metric extraction engine.

Fetches 10-K, 10-Q, and 8-K filings spanning 10+ years, extracts all
standardised metrics, builds direct EDGAR source URLs for each data point,
optionally generates AI summaries, and persists everything in MongoDB.

Now uses direct SEC EDGAR APIs (no edgartools dependency).
Data flow:
  1. SECClient.get_company_info() → company metadata
  2. SECClient.get_filings() → list of filings to process
  3. SECClient.get_facts_dataframe() → XBRL facts for extraction
  4. _extract_filing_metrics() → metrics + ratios for each filing
  5. MongoDB upsert → persist for instant future retrieval
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from sec_mcp.config import get_config
from sec_mcp.sec_client import get_sec_client
from sec_mcp.db import (
    count_filings,
    get_filing,
    get_filings,
    get_job,
    set_job,
    upsert_company,
    upsert_filing,
)
from sec_mcp.financials import (
    _apply_quality_filters,
    _build_statement_from_facts,
    _compute_ratios,
    _resolve_metric,
    _safe,
    _validate,
    _INCOME_CONCEPTS,
    _BALANCE_CONCEPTS,
    _CASHFLOW_CONCEPTS,
    _lookup_fact,
)
from sec_mcp.xbrl_mappings import (
    CONCEPT_MAP,
    EBITDA_COMPONENTS,
    IndustryClass,
    detect_industry_class,
    get_revenue_concepts,
)

log = logging.getLogger(__name__)

# Maximum filings per form type to process in historical extraction
# Includes FPI equivalents: 20-F (annual), 6-K (interim)
_MAX_PER_FORM = {"10-K": 12, "20-F": 12, "10-Q": 44, "6-K": 44, "8-K": 120}


# ═══════════════════════════════════════════════════════════════════════════
#  Source URL builders
# ═══════════════════════════════════════════════════════════════════════════

def _filing_urls(cik: int | str, accession: str, form_type: str = "") -> dict:
    """Build direct EDGAR URLs for a filing (used for citations in the UI)."""
    cik_raw = str(int(cik))
    cik_pad = cik_raw.zfill(10)
    acc_clean = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_clean}"

    return {
        "filing_index": f"{base}/{accession}-index.htm",
        "company_page": (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik_pad}&type={form_type}&dateb=&owner=include&count=40"
        ),
    }


def _metric_source_url(cik: int | str, accession: str) -> str:
    """Build a link directly to the inline XBRL viewer for the filing."""
    cik_raw = str(int(cik))
    acc_clean = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_clean}/{accession}-index.htm"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  AI summary generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_summary(
    company_name: str,
    form_type: str,
    filing_date: str,
    metrics: dict,
    ratios: dict,
) -> str | None:
    """Auto-generate a brief AI summary for a filing using Anthropic.

    Only called for the most recent filings to conserve API tokens.
    Returns None if ANTHROPIC_API_KEY is not configured.
    """
    try:
        config = get_config()
        if not config.anthropic_api_key:
            return None

        import anthropic

        # Build a compact metric string for the prompt
        lines = []
        for k, v in metrics.items():
            if v is not None and isinstance(v, dict) and v.get("value") is not None:
                lines.append(f"- {k}: ${v['value']:,.0f}")
        if not lines:
            return None

        ratio_lines = []
        for k, v in ratios.items():
            if v is not None:
                ratio_lines.append(f"- {k}: {v:.4f}")

        prompt = (
            f"Summarize this {form_type} filing for {company_name} (filed {filing_date}) "
            f"in 2-3 concise sentences highlighting key trends:\n\n"
            f"Metrics:\n" + "\n".join(lines[:15]) + "\n\n"
            + (f"Ratios:\n" + "\n".join(ratio_lines[:8]) + "\n\n" if ratio_lines else "")
            + "Be specific with numbers. No markdown."
        )

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        log.debug("Summary generation failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Single-filing metric extraction (from companyfacts data)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_filing_metrics(
    filing_meta: dict,
    facts_df: pd.DataFrame | None,
    industry: IndustryClass,
    cik: int | str,
) -> dict:
    """Extract all metrics + statements from one filing's XBRL data.

    Args:
        filing_meta: {accession_number, form_type, filing_date, description}
        facts_df: XBRL facts DataFrame filtered to this filing's accession
        industry: Detected industry class for revenue strategy
        cik: Company CIK for building source URLs
    """
    accession = filing_meta.get("accession_number", "")
    form_type = filing_meta.get("form_type", "")
    filing_date = filing_meta.get("filing_date", "")
    source_url = _metric_source_url(cik, accession)

    record: dict[str, Any] = {
        "accession": accession,
        "form_type": form_type,
        "filing_date": filing_date,
        "source_urls": _filing_urls(cik, accession, form_type),
        "metrics": {},
        "ratios": {},
        "statements": {},
        "summary": None,
    }

    # 8-K: no XBRL metrics — just metadata + description
    if "8-K" in form_type:
        desc = filing_meta.get("description") or ""
        record["description"] = desc
        # Try to extract item types from the description
        items = re.findall(r"Item\s+(\d+\.\d+)", desc)
        record["items_reported"] = items
        return record

    # 10-K / 10-Q: full XBRL extraction from companyfacts data
    if facts_df is None or facts_df.empty:
        return record

    clean_df = _apply_quality_filters(facts_df)

    # ── Revenue (industry-aware) ────
    rev_concepts = get_revenue_concepts(industry)
    resolved = _resolve_metric(clean_df, rev_concepts, period_index=0, industry=industry)
    record["metrics"]["revenue"] = {
        "value": resolved.value,
        "source": resolved.source,
        "confidence": resolved.confidence,
        "source_url": source_url,
    }

    # ── All other metrics from CONCEPT_MAP ────
    for metric_name, concepts in CONCEPT_MAP.items():
        resolved = _resolve_metric(clean_df, concepts, period_index=0, industry=industry)
        record["metrics"][metric_name] = {
            "value": resolved.value,
            "source": resolved.source,
            "confidence": resolved.confidence,
            "source_url": source_url,
        }

    # ── Derived: Free Cash Flow (OCF - |capex|) ────
    ocf_v = (record["metrics"].get("operating_cash_flow") or {}).get("value")
    capex_v = (record["metrics"].get("capital_expenditures") or {}).get("value")
    if ocf_v is not None and capex_v is not None:
        record["metrics"]["free_cash_flow"] = {
            "value": ocf_v - abs(capex_v),
            "source": "derived: OCF - |capex|",
            "confidence": 0.80,
            "source_url": source_url,
        }

    # ── Derived: EBITDA (Operating Income + D&A) ────
    oi_v = (record["metrics"].get("operating_income") or {}).get("value")
    if oi_v is not None:
        da = 0.0
        for entry in EBITDA_COMPONENTS:
            if entry.aggregate:
                v = _lookup_fact(clean_df, entry.xbrl_concept, 0, match_mode="exact")
                if v is None:
                    v = _lookup_fact(clean_df, entry.xbrl_concept, 0, match_mode="contains")
                if v is not None:
                    da += abs(v)
        if da > 0:
            record["metrics"]["ebitda"] = {
                "value": oi_v + da,
                "source": "derived: OI + D&A",
                "confidence": 0.70,
                "source_url": source_url,
            }

    # ── Ratios ────
    flat = {k: (v or {}).get("value") for k, v in record["metrics"].items()}
    record["ratios"] = _compute_ratios(flat)

    # ── Statements (built from companyfacts data) ────
    record["statements"] = {
        "income_statement": _build_statement_from_facts(facts_df, _INCOME_CONCEPTS),
        "balance_sheet": _build_statement_from_facts(facts_df, _BALANCE_CONCEPTS),
        "cash_flow_statement": _build_statement_from_facts(facts_df, _CASHFLOW_CONCEPTS),
    }

    return record


# ═══════════════════════════════════════════════════════════════════════════
#  Full historical extraction (runs as background task)
# ═══════════════════════════════════════════════════════════════════════════

def run_historical_extraction(ticker: str, years: int = 10):
    """Fetch all filings for *ticker* spanning *years*, extract metrics, store in MongoDB.

    Designed to be called as a FastAPI background task.

    Process:
      1. Look up company via SEC submissions API
      2. Get all filings (10-K, 10-Q, 8-K) within the year range
      3. Fetch companyfacts data once (all XBRL facts)
      4. For each filing, filter facts by accession and extract metrics
      5. Generate AI summaries for the most recent filings
      6. Upsert everything into MongoDB
    """
    client = get_sec_client()
    set_job(ticker, "processing", 0, 0, "Loading company info")

    # ── Get company info ──
    try:
        company = client.get_company_info(ticker)
    except Exception as exc:
        set_job(ticker, "error", detail=f"Company not found: {exc}")
        return

    cik = company.cik
    sic = company.sic_code
    ticker_hint = company.ticker or ticker
    industry = detect_industry_class(sic, ticker=ticker_hint)

    upsert_company(ticker, {
        "name": company.name or ticker,
        "cik": cik,
        "sic": sic,
        "industry_class": industry.value,
    })

    # ── Collect filing metadata for all form types ──
    cutoff_year = datetime.now().year - years
    all_filings: list[dict] = []  # list of {accession_number, form_type, filing_date, ...}

    for form_type, max_count in _MAX_PER_FORM.items():
        try:
            filings = client.get_filings(ticker, form_type=form_type, limit=max_count)
            for f in filings:
                # Filter by year (only process filings within our window)
                if f.filing_date:
                    try:
                        fdate = datetime.strptime(f.filing_date[:10], "%Y-%m-%d")
                        if fdate.year < cutoff_year:
                            continue
                    except ValueError:
                        pass

                # Skip if already in MongoDB
                if get_filing(ticker, f.accession_number) is not None:
                    continue

                all_filings.append({
                    "accession_number": f.accession_number,
                    "form_type": f.form_type,
                    "filing_date": f.filing_date,
                    "description": f.description,
                })
        except Exception as exc:
            log.warning("Could not list %s filings for %s: %s", form_type, ticker, exc)

    total = len(all_filings)
    set_job(ticker, "processing", 0, total, f"Extracting {total} filings")
    log.info("Processing %d new filings for %s", total, ticker)

    # ── Fetch companyfacts once (contains ALL XBRL data) ──
    # We'll filter by accession for each filing
    full_facts_df = None
    try:
        full_facts_df = client.get_facts_dataframe(ticker)
    except Exception as exc:
        log.warning("Could not fetch companyfacts for %s: %s", ticker, exc)

    # ── Process each filing ──
    for idx, filing_meta in enumerate(all_filings):
        accession = filing_meta["accession_number"]
        ft = filing_meta["form_type"]
        set_job(ticker, "processing", idx + 1, total, f"{ft} {accession[:20]}")

        try:
            # Filter facts to this specific filing's accession number
            filing_facts = None
            if full_facts_df is not None and not full_facts_df.empty and "accn" in full_facts_df.columns:
                filing_facts = full_facts_df[full_facts_df["accn"] == accession]

            record = _extract_filing_metrics(filing_meta, filing_facts, industry, cik)

            # AI summary for the most recent 15 filings (to save API tokens)
            if idx < 15:
                record["summary"] = _generate_summary(
                    company.name or ticker, ft,
                    record.get("filing_date", ""),
                    record.get("metrics", {}),
                    record.get("ratios", {}),
                )

            upsert_filing(ticker, accession, record)
        except Exception as exc:
            log.warning("Failed to process %s %s: %s", ft, accession, exc)
            continue

    set_job(ticker, "complete", total, total, "Done")
    log.info("Historical extraction complete for %s: %d filings", ticker, total)


def get_historical_data(ticker: str, form_type: str | None = None) -> dict:
    """Return the historical dataset for *ticker* from MongoDB.

    Called by the API to return cached data instantly.

    Returns:
        {company, filings, total_filings, job}
    """
    from sec_mcp.db import get_company as db_get_company

    company = db_get_company(ticker)
    filings = get_filings(ticker, form_type=form_type)
    job = get_job(ticker)
    total = count_filings(ticker)

    return {
        "company": company,
        "filings": filings,
        "total_filings": total,
        "job": job,
    }
