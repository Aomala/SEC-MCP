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

_cache: dict[tuple, dict | None] = {}
_lock = threading.Lock()


def _xbrl_for(ticker: str, form_type: str):
    import edgar
    edgar.set_identity(get_config().edgar_identity)
    company = edgar.Company(ticker)
    filings = company.get_filings(form=form_type).latest(1)
    if filings is None:
        return None
    return filings.xbrl()


def _axis_breakdown(facts, axes: tuple[str, ...]) -> list[dict]:
    """Latest-period revenue breakdown along the first axis that has data."""
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
            if len(rows) >= 2:
                total = sum(r["value"] for r in rows)
                for r in rows:
                    r["pct"] = round(100 * r["value"] / total, 1)
                return rows
    return []


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
    """{'segments': [...], 'geographic_segments': [...]} from the latest
    filing's dimensional XBRL, or None when unavailable."""
    key = (ticker.upper(), form_type)
    with _lock:
        if key in _cache:
            return _cache[key]
    result: dict | None = None
    try:
        x = _xbrl_for(ticker.upper(), form_type)
        if x is not None:
            biz = _axis_breakdown(x.facts, _BUSINESS_AXES)
            prod = _axis_breakdown(x.facts, _PRODUCT_AXES)
            geo = _axis_breakdown(x.facts, _GEO_AXES)

            # Some filers' reportable segments ARE regions (Apple). When the
            # business axis looks geographic, the product axis is the real
            # "segments" answer — and the regional split can stand in for
            # geography if no geographic axis is tagged.
            if _looks_geographic(biz) and prod:
                segs = prod
                if not geo:
                    geo = biz
            else:
                segs = biz or prod

            if segs or geo:
                result = {"segments": segs, "geographic_segments": geo,
                          "source": "sec_xbrl_dimensions"}
    except Exception as exc:
        log.warning("dimensional segments failed for %s: %s", ticker, exc)
    with _lock:
        _cache[key] = result
        if len(_cache) > 64:
            _cache.clear()
    return result
