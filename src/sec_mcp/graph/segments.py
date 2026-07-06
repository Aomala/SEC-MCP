"""Dimensional segment extraction from per-filing XBRL.

companyfacts strips ALL dimensional facts by design, so true segment
breakdowns (business segments, product lines, geography) only exist in the
filing's own XBRL instance. edgartools exposes them via dimension queries —
this is the authoritative source the filing-text scraper was approximating
(and sometimes getting badly wrong: income-statement rows served as
"segments").

Results are cached in-process per (cik, form) for the session; the heavy
XBRL parse is also reused by the calc-tree graph (edgartools caches the
filing internally).
"""

from __future__ import annotations

import logging
import re
import threading

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

# Revenue concepts to look for on each axis, in preference order
_REVENUE_CONCEPTS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
)

# Axes that express each segment kind
_BUSINESS_AXES = ("us-gaap_StatementBusinessSegmentsAxis",)
_PRODUCT_AXES = ("srt_ProductOrServiceAxis",)
_GEO_AXES = (
    "srt_StatementGeographicalAxis",
    "us-gaap_StatementGeographicalAxis",
)

# Labels that signal a breakdown is GEOGRAPHIC (some filers' reportable
# segments ARE regions — Apple's are Americas/Europe/Greater China/etc.)
_GEO_WORDS = ("americas", "europe", "china", "japan", "asia", "united states",
              "international", "emea", "apac", "canada", "rest of", "domestic",
              "foreign", "other countries", "latin america", "middle east")


def _looks_geographic(rows: list[dict]) -> bool:
    if not rows:
        return False
    hits = sum(1 for r in rows
               if any(w in r["segment"].lower() for w in _GEO_WORDS))
    return hits >= max(2, len(rows) // 2)

# Member labels that are totals/eliminations, not real segments
_EXCLUDE_LABELS = ("total", "elimination", "consolidat", "reconcil",
                   "intersegment", "corporate and other unallocated")

# Generic product-vs-service members (us-gaap ProductMember/ServiceMember and
# friends). Some filers (MSFT) tag this coarse split on ProductOrServiceAxis
# ALONGSIDE their detailed product lines — two complete partitions of the same
# revenue, which double-counts the total and halves every pct.
_GENERIC_TYPE_LABELS = frozenset({
    "product", "products", "service", "services", "service and other",
    "services and other", "product and other", "products and services",
})


def _drop_generic_type_partition(rows: list[dict]) -> list[dict]:
    """Drop the coarse Product/Service partition when it duplicates the
    detailed members (both partitions sum to the same total). Sum equality is
    the guard: a genuinely generic-labeled segment (Apple's 'Services') won't
    match the rest of the rows' sum, so it survives.
    """
    generic = [r for r in rows
               if r["segment"].strip().lower() in _GENERIC_TYPE_LABELS]
    detail = [r for r in rows if r not in generic]
    if generic and len(detail) >= 2:
        gsum = sum(r["value"] for r in generic)
        dsum = sum(r["value"] for r in detail)
        if gsum and abs(gsum - dsum) / gsum < 0.02:
            return detail
    return rows

_cache: dict[tuple, dict | None] = {}
_lock = threading.Lock()


def _xbrl_for(ticker: str, form_type: str):
    """Return (filing, xbrl) for the latest filing of form_type, or (None, None).

    Callers need the filing object for its accession/form and the xbrl for
    facts + entity_info (fiscal period, report date).
    """
    import edgar
    edgar.set_identity(get_config().edgar_identity)
    company = edgar.Company(ticker)
    filing = company.get_filings(form=form_type).latest(1)
    if filing is None:
        return None, None
    return filing, filing.xbrl()


def _axis_breakdown(facts, axes: tuple[str, ...]) -> tuple[list[dict], str | None]:
    """Latest-period revenue breakdown along the first axis that has data.

    Returns (rows, currency) where currency is the ISO code from the facts'
    unit_ref (e.g. "USD", "EUR" for FPIs) or None if undetectable.
    """
    for axis in axes:
        for concept in _REVENUE_CONCEPTS:
            try:
                df = (facts.query().by_dimension(axis)
                      .by_concept(concept).to_dataframe())
            except Exception:
                continue
            if df is None or df.empty or "period_end" not in df.columns:
                continue
            latest = df[df["period_end"] == df["period_end"].max()]
            # Currency rides on the fact's unit_ref; normalize to an ISO code.
            # The ref format varies by edgartools version ("usd" vs "u_usd" vs
            # "U_iso4217:USD") — keep only the token after the last _ or :
            currency = None
            if "unit_ref" in latest.columns:
                units = latest["unit_ref"].dropna()
                if not units.empty:
                    raw = re.split(r"[_:]", str(units.iloc[0]))[-1].upper()
                    currency = raw if re.fullmatch(r"[A-Z]{3}", raw) else None
            rows = []
            for _, r in latest.iterrows():
                label = str(r.get("label") or "").strip()
                val = r.get("numeric_value")
                if not label or val is None or val <= 0:
                    continue
                if any(x in label.lower() for x in _EXCLUDE_LABELS):
                    continue
                rows.append({"segment": label, "value": float(val)})
            # Dedup labels (same member can appear in multiple contexts)
            seen: dict[str, dict] = {}
            for row in rows:
                cur = seen.get(row["segment"])
                if cur is None or row["value"] > cur["value"]:
                    seen[row["segment"]] = row
            rows = sorted(seen.values(), key=lambda x: -x["value"])
            rows = _drop_subtotals(rows)
            rows = _drop_generic_type_partition(rows)
            if len(rows) >= 2:
                total = sum(r["value"] for r in rows)
                for r in rows:
                    r["pct"] = round(100 * r["value"] / total, 1)
                return rows, currency
    return [], None


def _drop_subtotals(rows: list[dict]) -> list[dict]:
    """Remove parent subtotal members that double-count their children.

    Apple's ProductOrServiceAxis carries 'Products' (= iPhone + Mac + iPad +
    Wearables) alongside the detail members. A member is a subtotal when some
    subset (>=2) of the OTHER members sums to its value within 1%; checked
    largest-first with brute-force subset sums (axes have <=10 members).
    """
    from itertools import combinations

    rows = list(rows)
    i = 0
    while i < len(rows) and len(rows) > 2:
        target = rows[i]["value"]
        others = [r["value"] for j, r in enumerate(rows) if j != i]
        if len(others) <= 10:
            found = False
            for k in range(2, len(others) + 1):
                for combo in combinations(others, k):
                    s = sum(combo)
                    if s and abs(s - target) / target < 0.01:
                        found = True
                        break
                if found:
                    break
            if found:
                rows.pop(i)
                continue
        i += 1
    return rows


def get_dimensional_segments(ticker: str, form_type: str = "10-K") -> dict | None:
    """{'segments': [...], 'geographic_segments': [...], 'source_meta': {...}}
    from the latest filing's dimensional XBRL, or None when unavailable.

    `segments` is the product/service-type revenue split
    (srt_ProductOrServiceAxis), falling back to reportable business segments
    only when no product axis is tagged; `geographic_segments` is the
    geographic split. `source_meta.segmentsAxis`/`geographyAxis` record which
    axis each actually came from.

    `form_type` defaults to "10-K" (the annual path is unchanged). Pass "10-Q"
    for the latest quarterly breakdown. Foreign private issuers are handled via
    get_form_alternatives (10-K→20-F, 10-Q→6-K), so an FPI request resolves to
    the right filing and `source_meta.formType` reports what was actually used.

    `source_meta` describes the SOURCE filing the segments came from:
    {formType, fiscalPeriod, fiscalYear, reportDate, accession, currency,
    segmentsAxis, geographyAxis}.
    """
    from sec_mcp.sec_client import get_form_alternatives

    key = (ticker.upper(), form_type)
    with _lock:
        if key in _cache:
            return _cache[key]
    result: dict | None = None
    try:
        # Try the requested form first, then its FPI equivalent (20-F/6-K).
        filing = x = None
        for ft in get_form_alternatives(form_type):
            filing, x = _xbrl_for(ticker.upper(), ft)
            if x is not None:
                break
        if x is not None:
            biz, ccy_biz = _axis_breakdown(x.facts, _BUSINESS_AXES)
            prod, ccy_prod = _axis_breakdown(x.facts, _PRODUCT_AXES)
            geo, ccy_geo = _axis_breakdown(x.facts, _GEO_AXES)

            # The two breakdowns we serve are product/service type and
            # geography. srt_ProductOrServiceAxis is the primary "segments"
            # answer; the business axis only stands in when no product axis
            # is tagged AND it isn't just regions (some filers' reportable
            # segments ARE regions — Apple's are Americas/Europe/Greater
            # China/… — those belong under geography, not product).
            biz_is_geo = _looks_geographic(biz)
            geo_axis = "geographic" if geo else None
            if not geo and biz_is_geo:
                geo = biz
                geo_axis = "business"
            segs = prod or ([] if biz_is_geo else biz)
            segs_axis = "product" if prod else ("business" if segs else None)

            if segs or geo:
                # Build provenance from the filing actually used. entity_info
                # carries fiscal period/year; the filing object the accession.
                ei = getattr(x, "entity_info", None) or {}
                result = {
                    "segments": segs,
                    "geographic_segments": geo,
                    "source": "sec_xbrl_dimensions",
                    "source_meta": {
                        "formType": getattr(x, "document_type", None)
                                    or getattr(filing, "form", None),
                        "fiscalPeriod": ei.get("fiscal_period"),
                        "fiscalYear": ei.get("fiscal_year"),
                        "reportDate": str(getattr(x, "period_of_report", "")
                                          or getattr(filing, "report_date", "")) or None,
                        "accession": getattr(filing, "accession_no", None),
                        # currency rides on the segment facts (USD/EUR/…)
                        "currency": ccy_prod or ccy_geo or ccy_biz,
                        # which XBRL axis each breakdown actually came from
                        "segmentsAxis": segs_axis,
                        "geographyAxis": geo_axis,
                    },
                }
    except Exception as exc:
        log.warning("dimensional segments failed for %s: %s", ticker, exc)
    with _lock:
        _cache[key] = result
        if len(_cache) > 64:
            _cache.clear()
    return result
