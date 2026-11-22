"""Production-grade financial data extraction engine.

Three-layer architecture:
  Layer 1 — Deterministic XBRL filtering (dimension/abstract/level)
  Layer 2 — Canonical dictionary matching (ordered concept lists)
  Layer 3 — LLM disambiguation (only when ambiguous)

Uses SEC EDGAR's companyfacts API directly (no edgartools).
Data flow:
  1. SECClient.get_company_info() → company metadata + SIC code
  2. SECClient.get_filings() → find the target filing (by year/form)
  3. SECClient.get_facts_dataframe() → XBRL facts as a DataFrame
  4. _resolve_metric() → match concepts to standardized metrics
  5. _compute_ratios() → derive financial ratios
  6. _validate() → flag data quality issues

Features:
  1. Industry detection (bank/standard/insurance/crypto ...)
  2. Year-constrained filing selection
  3. 4-pass concept resolution with quality filtering
  4. Custom extension detection (ms_NetRevenues, gs_TotalNetRevenues ...)
  5. Multi-concept aggregation (bank revenue = NII + non-interest + trading)
  6. Revenue/geographic segmentation
  7. Validation rules with confidence scoring
  8. SEC EDGAR filing links for citation
  9. LLM disambiguation for ambiguous cases
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from sec_mcp import disk_cache
from sec_mcp.config import get_config
from sec_mcp.facts.interim import fiscal_quarter, has_release_format, release_quarters
from sec_mcp.facts.periods import PeriodType, annotate_periods, dedupe_facts
from sec_mcp.models import FilingMetadata
from sec_mcp.sec_client import get_sec_client
from sec_mcp.xbrl_mappings import (
    ABSTRACT_COLUMNS,
    BANK_NONINTEREST_SUBDRIVERS,
    CONCEPT_MAP,
    DIMENSION_COLUMNS,
    LEVEL_COLUMNS,
    MAX_ACCEPTABLE_LEVEL,
    ConceptEntry,
    IndustryClass,
    detect_industry_class,
    get_bank_revenue_drivers,
    get_operating_income_concepts,
    get_revenue_concepts,
    is_custom_net_revenue,
    is_custom_revenue,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Safe numeric helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe(v: Any) -> float | None:
    """Convert a value to float, returning None for invalid/missing values."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None) -> str:
    """Format a number for display (e.g., $1.23B, $456M)."""
    if v is None:
        return "N/A"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e12:
        return f"{sign}${av / 1e12:,.2f}T"
    if av >= 1e9:
        return f"{sign}${av / 1e9:,.2f}B"
    if av >= 1e6:
        return f"{sign}${av / 1e6:,.2f}M"
    return f"{sign}${av:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════
#  Year-based filing selection (uses SEC submissions API)
# ═══════════════════════════════════════════════════════════════════════════

def _find_filing_accession(
    ticker_or_cik: str,
    form_type: str = "10-K",
    year: int | None = None,
) -> dict | None:
    """Find the best matching filing accession for a company/year/form.

    Searches the SEC submissions API (always up-to-date) for the target
    filing. Returns a dict with accession_number, filing_date, form_type
    or None if no matching filing found.
    """
    client = get_sec_client()
    # Periodic-filing index derived from companyfacts (full history, even for
    # heavy filers whose submissions `recent` window only spans months), with
    # automatic FPI fallback (20-F for 10-K, 6-K for 10-Q).
    filings = client.get_periodic_filings_smart(ticker_or_cik, form_type=form_type, limit=40)

    if not filings:
        return None

    if year is not None:
        # Match on the REPORT period-end (fiscal) year — the authoritative basis
        # for "which fiscal year is this". filing_date lags the period end (a
        # Sept-FY company files FY2024 in Nov 2024; a Dec-FY company files FY2024
        # in Feb 2025), so matching filing_date mislabels periods. Prefer the
        # filing whose reportDate falls in the requested year.
        by_report = [
            f for f in filings
            if getattr(f, "report_date", "") and f.report_date.startswith(str(year))
        ]
        if by_report:
            filings = by_report
        else:
            # Fallback for filings missing reportDate: keep the old filing_date
            # window (year or the following calendar year for Jan/Feb filers).
            matching = [
                f for f in filings
                if f.filing_date and any(
                    f.filing_date.startswith(str(y)) for y in (year, year + 1)
                )
            ]
            if matching:
                filings = matching

    if not filings:
        return None

    # Return the most recent match
    best = filings[0]
    return {
        "accession_number": best.accession_number,
        "filing_date": best.filing_date,
        "form_type": best.form_type,
        "report_date": getattr(best, "report_date", ""),
    }


def _get_facts_for_filing(
    ticker_or_cik: str,
    accession: str | None = None,
    form_type: str = "10-K",
) -> pd.DataFrame:
    """Get XBRL facts as a DataFrame, optionally filtered to a specific filing.

    Uses the SEC companyfacts API which returns ALL XBRL data for a company.
    We filter by accession number to get facts for a specific filing,
    or by form_type to get the most recent facts of that type.

    Returns an empty DataFrame if XBRL data is unavailable (e.g. 404).
    """
    client = get_sec_client()
    try:
        return client.get_facts_dataframe(
            ticker_or_cik,
            form_type=form_type,
            accession=accession,
            taxonomy="all" if form_type in ("6-K", "6-K/A") else "us-gaap",
        )
    except Exception as exc:
        log.warning("Failed to get XBRL facts for %s: %s", ticker_or_cik, exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  Row quality filtering
# ═══════════════════════════════════════════════════════════════════════════

def _apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out segment-level, abstract, and deeply-nested rows.

    This is the critical step that prevents the resolver from picking
    segment breakdowns or intersegment eliminations instead of totals.
    The companyfacts API data is generally cleaner than raw XBRL,
    but we still apply filters for safety.
    """
    if df is None or df.empty:
        return df

    filtered = df.copy()

    # --- Exclude rows with dimensional / segment data ---
    for col in DIMENSION_COLUMNS:
        if col in filtered.columns:
            mask = filtered[col].isna() | (filtered[col].astype(str).str.strip() == "")
            filtered = filtered[mask]
            break  # only need one dimension column

    # --- Exclude abstract rows ---
    for col in ABSTRACT_COLUMNS:
        if col in filtered.columns:
            mask = ~filtered[col].astype(str).str.lower().isin(["true", "1", "yes"])
            filtered = filtered[mask]
            break

    # --- Exclude dimension/text remnants by concept name ---
    # companyfacts strips dimensional facts, but custom taxonomies can still
    # surface Axis/Member concepts, and TextBlock facts are narrative HTML.
    concept_col = _find_column(filtered, ("concept", "Concept", "tag", "Tag"))
    if concept_col is not None:
        names = filtered[concept_col].astype(str)
        bad = (names.str.endswith("Axis") | names.str.endswith("Member")
               | names.str.contains("TextBlock", na=False))
        filtered = filtered[~bad]

    # --- Prefer rows with level <= MAX_ACCEPTABLE_LEVEL ---
    for col in LEVEL_COLUMNS:
        if col in filtered.columns:
            try:
                levels = pd.to_numeric(filtered[col], errors="coerce")
                level_mask = levels.isna() | (levels <= MAX_ACCEPTABLE_LEVEL)
                if level_mask.any():
                    filtered = filtered[level_mask]
            except Exception as exc:
                log.debug("Level filtering failed on column %s: %s", col, exc)
            break

    # If filtering removed everything, fall back to the original
    if filtered.empty:
        return df

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Concept resolver (production-grade)
# ═══════════════════════════════════════════════════════════════════════════

class ResolvedMetric:
    """Result of resolving a metric with confidence scoring."""

    __slots__ = ("value", "source", "confidence", "method")

    def __init__(
        self,
        value: float | None,
        source: str | None,
        confidence: float = 0.0,
        method: str = "none",
    ):
        self.value = value
        self.source = source
        self.confidence = confidence  # 0.0–1.0
        self.method = method          # "exact", "contains", "custom_ext", "aggregate"


def _resolve_metric(
    facts_df: pd.DataFrame | None,
    concepts: list[ConceptEntry],
    *,
    period_index: int = 0,
    industry: IndustryClass = IndustryClass.STANDARD,
    prefer_quarterly: bool = False,
    duration_pref: str | None = None,
    target_fp: str | None = None,
    target_fy: int | None = None,
    is_revenue: bool = False,
    canonical_key: str | None = None,
    ticker_or_cik: str | None = None,
    accession: str | None = None,
) -> ResolvedMetric:
    """Dispatcher: graph resolver (per the GRAPH_RESOLVER flag) with the
    legacy 4-pass resolver as the always-available fallback.

    Modes (config.graph_resolver):
      off    — legacy only (default)
      shadow — run both, SERVE legacy, log disagreements for adjudication
      on     — serve graph result when it resolves; legacy otherwise
    """
    mode = "off"
    if canonical_key and accession and ticker_or_cik:
        try:
            mode = (get_config().graph_resolver or "off").lower()
        except Exception:
            mode = "off"

    graph_result: ResolvedMetric | None = None
    if mode in ("shadow", "on"):
        try:
            from sec_mcp.graph import get_graph_resolver
            graph_result = get_graph_resolver().resolve(
                facts_df, canonical_key,
                ticker_or_cik=ticker_or_cik, accession=accession,
                industry=industry, period_index=period_index,
                duration_pref=duration_pref,
                target_fp=target_fp, target_fy=target_fy,
            )
        except Exception as exc:
            log.warning("graph resolver failed for %s/%s: %s",
                        ticker_or_cik, canonical_key, exc)

    if mode == "on" and graph_result is not None:
        return graph_result

    legacy = _resolve_metric_legacy(
        facts_df, concepts, period_index=period_index, industry=industry,
        prefer_quarterly=prefer_quarterly, duration_pref=duration_pref,
        target_fp=target_fp, target_fy=target_fy, is_revenue=is_revenue,
    )

    if mode == "shadow" and graph_result is not None:
        _log_resolver_diff(ticker_or_cik, accession, canonical_key,
                           legacy, graph_result)
    return legacy


def _log_resolver_diff(ticker, accession, key, legacy, graph) -> None:
    """Record legacy-vs-graph disagreements (shadow mode adjudication)."""
    lv, gv = legacy.value, graph.value
    if lv is not None and gv is not None and lv != 0:
        rel = abs(lv - gv) / max(abs(lv), 1e-9)
        if rel < 0.001:
            return  # agreement
    elif lv is None and gv is None:
        return
    else:
        rel = None
    log.warning("RESOLVER_DIFF %s %s %s: legacy=%s (%s) graph=%s (%s) rel=%s",
                ticker, accession, key, lv, legacy.source, gv, graph.source,
                f"{rel:.2%}" if rel is not None else "n/a")
    try:
        from sec_mcp.supabase_cache import _get_client
        client = _get_client()
        if client is not None:
            client.table("resolver_diffs").insert({
                "ticker": str(ticker), "accession": accession,
                "canonical_key": key, "legacy_value": lv, "graph_value": gv,
                "legacy_source": legacy.source, "graph_source": graph.source,
                "rel_diff": rel,
            }).execute()
    except Exception:
        pass  # diff logging must never break extraction


def _resolve_metric_legacy(
    facts_df: pd.DataFrame | None,
    concepts: list[ConceptEntry],
    *,
    period_index: int = 0,
    industry: IndustryClass = IndustryClass.STANDARD,
    prefer_quarterly: bool = False,
    duration_pref: str | None = None,
    target_fp: str | None = None,
    target_fy: int | None = None,
    is_revenue: bool = False,
) -> ResolvedMetric:
    """Resolve a metric from a facts DataFrame using ordered concept list.

    Resolution order (4-pass):
      1. Exact match on non-aggregate us-gaap concepts (highest confidence)
      2. Contains match on non-aggregate concepts (opt-in via allow_fuzzy)
      3. Custom extension pattern match (revenue resolution only)
      4. Aggregate fallback (sum of components, duration-targeted)

    duration_pref:
      "quarterly" — YTD-through-quarter durations (for 10-Q flow metrics)
      "annual"    — ~12-month durations (for 10-K flow metrics)
      None        — no filtering (for instant/balance-sheet items)

    prefer_quarterly (legacy): equivalent to duration_pref="quarterly"
    """
    if facts_df is None or facts_df.empty:
        return ResolvedMetric(None, None)

    # Resolve legacy flag
    if duration_pref is None and prefer_quarterly:
        duration_pref = "quarterly"

    # Apply quality filters to exclude segment/abstract/nested rows
    clean_df = _apply_quality_filters(facts_df)

    # --- Pass 1: Non-aggregate exact match ---
    for entry in concepts:
        if entry.aggregate:
            continue
        # For balance sheet (instant) concepts, never filter by duration
        is_instant = entry.xbrl_concept in _INSTANT_CONCEPTS
        dp = duration_pref if not is_instant else None
        tfp = target_fp if not is_instant else None
        tfy = target_fy if not is_instant else None
        val = _lookup_fact(clean_df, entry.xbrl_concept, period_index,
                           match_mode="exact", duration_pref=dp,
                           target_fp=tfp, target_fy=tfy)
        if val is not None:
            return ResolvedMetric(val, entry.display_name, confidence=0.99, method="exact")

    # --- Pass 2: Non-aggregate contains match (opt-in only) ---
    # Substring matching is how guidance/segment/wrong tags leak in (e.g.
    # "Revenues" contains-matches "DeferredRevenues"). Entries must opt in
    # via ConceptEntry.allow_fuzzy.
    for entry in concepts:
        if entry.aggregate or not entry.allow_fuzzy:
            continue
        is_instant = entry.xbrl_concept in _INSTANT_CONCEPTS
        dp = duration_pref if not is_instant else None
        tfp = target_fp if not is_instant else None
        tfy = target_fy if not is_instant else None
        val = _lookup_fact(clean_df, entry.xbrl_concept, period_index,
                           match_mode="contains", duration_pref=dp,
                           target_fp=tfp, target_fy=tfy)
        if val is not None:
            return ResolvedMetric(val, entry.display_name, confidence=0.90, method="contains")

    # --- Pass 3: Custom extension patterns (banks, large filers) ---
    # Revenue-shaped patterns ONLY apply when we're resolving revenue;
    # without the gate a missing metric (e.g. deferred_revenue) could be
    # silently filled with the filer's custom *revenue* value.
    if is_revenue:
        if industry == IndustryClass.BANK:
            val, src = _resolve_custom_extension(
                clean_df, is_custom_net_revenue, "Custom Net Revenue", period_index,
                duration_pref=duration_pref, target_fp=target_fp, target_fy=target_fy,
            )
            if val is not None:
                return ResolvedMetric(val, src, confidence=0.85, method="custom_ext")
        else:
            val, src = _resolve_custom_extension(
                clean_df, is_custom_revenue, "Custom Revenue", period_index,
                duration_pref=duration_pref, target_fp=target_fp, target_fy=target_fy,
            )
            if val is not None:
                return ResolvedMetric(val, src, confidence=0.80, method="custom_ext")

    # --- Pass 4: Aggregate fallback ---
    total = 0.0
    matched_any = False
    components: list[str] = []
    for entry in concepts:
        if not entry.aggregate:
            continue
        # Exact match only, with the same duration targeting as Pass 1 —
        # summing components from mismatched periods is worse than missing.
        is_instant = entry.xbrl_concept in _INSTANT_CONCEPTS
        dp = duration_pref if not is_instant else None
        tfp = target_fp if not is_instant else None
        tfy = target_fy if not is_instant else None
        val = _lookup_fact(clean_df, entry.xbrl_concept, period_index,
                           match_mode="exact", duration_pref=dp,
                           target_fp=tfp, target_fy=tfy)
        if val is not None:
            total += val
            matched_any = True
            components.append(entry.display_name)

    if matched_any:
        label = " + ".join(components)
        conf = min(0.85, 0.60 + 0.05 * len(components))
        return ResolvedMetric(total, label, confidence=conf, method="aggregate")

    return ResolvedMetric(None, None)


_CUSTOM_EXT_BLOCKLIST = ("guidance", "forecast", "estimated", "expected",
                         "projected", "outlook")


def _resolve_custom_extension(
    facts_df: pd.DataFrame,
    pattern_fn,
    label_prefix: str,
    period_index: int,
    *,
    duration_pref: str | None = None,
    target_fp: str | None = None,
    target_fy: int | None = None,
) -> tuple[float | None, str | None]:
    """Resolve a value from custom-extension concepts matching `pattern_fn`.

    Candidate concepts are collected first, then each is resolved through
    `_lookup_fact` with full duration/period targeting — the legacy version
    returned the first matching ROW, i.e. a value from an arbitrary period.
    When several custom concepts match, the largest absolute value wins
    (a total is at least as large as any of its components).
    """
    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    if concept_col is None:
        return None, None

    candidates = [
        str(c) for c in facts_df[concept_col].dropna().unique()
        if pattern_fn(str(c))
        and not any(kw in str(c).lower() for kw in _CUSTOM_EXT_BLOCKLIST)
    ]

    best_val: float | None = None
    best_src: str | None = None
    for name in candidates:
        val = _lookup_fact(facts_df, name, period_index,
                           match_mode="exact", duration_pref=duration_pref,
                           target_fp=target_fp, target_fy=target_fy)
        if val is not None and (best_val is None or abs(val) > abs(best_val)):
            best_val = val
            best_src = f"{label_prefix} ({name})"

    return best_val, best_src


def _lookup_fact(
    facts_df: pd.DataFrame | None,
    concept: str,
    period_index: int = 0,
    *,
    match_mode: str = "exact",
    prefer_quarterly: bool = False,
    duration_pref: str | None = None,
    target_fp: str | None = None,
    target_fy: int | None = None,
) -> float | None:
    """Find a fact value for the given concept and period.

    match_mode:
      "exact"    — concept column matches the tag name directly
      "contains" — concept column contains the tag name anywhere

    duration_pref:
      "quarterly" — prefer 3-month durations (for 10-Q flow metrics)
      "annual"    — prefer 12-month durations (for 10-K flow metrics)
      None        — no filtering (for instant/balance-sheet items)

    prefer_quarterly (legacy):
      Equivalent to duration_pref="quarterly". Kept for backwards compatibility.
    """
    if facts_df is None or facts_df.empty:
        return None

    # Resolve legacy flag into duration_pref
    if duration_pref is None and prefer_quarterly:
        duration_pref = "quarterly"

    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(facts_df, ("value", "Value", "val"))
    if concept_col is None or value_col is None:
        return None

    if match_mode == "exact":
        direct_mask = (facts_df[concept_col] == concept)
        direct_matches = facts_df.loc[direct_mask]

        if not direct_matches.empty:
            matches = direct_matches
        else:
            mask = (
                facts_df[concept_col].str.endswith(f":{concept}", na=False)
                | facts_df[concept_col].str.endswith(f"_{concept}", na=False)
            )
            matches = facts_df.loc[mask]
    else:
        mask = facts_df[concept_col].str.contains(concept, case=False, na=False)
        matches = facts_df.loc[mask]

    if matches.empty:
        return None

    # Collapse duplicate reports of the same (concept, unit, start, end) —
    # companyfacts repeats a fact once per filing that reported it (original,
    # comparative, amendment); the latest `filed` value is the current truth.
    # Without this, period_index walks an arbitrary mix of duplicates.
    matches = dedupe_facts(matches)

    # Sort by period (newest first) — companyfacts has 'end' column
    end_col = None
    for pcol in ("end", "period", "Period", "endDate"):
        if pcol in matches.columns:
            end_col = pcol
            try:
                matches = matches.sort_values(pcol, ascending=False)
            except Exception as exc:
                log.debug("Sort by %s failed: %s", pcol, exc)
            break

    # Duration-aware filtering for flow metrics. A fact's period comes from
    # its own start/end dates — NEVER fy/fp, which describe the filing that
    # reported it (a 10-K stamps its comparative-year facts with the same
    # fy/fp, so fp filtering collapses years).
    if duration_pref and end_col and "start" in matches.columns:
        if duration_pref == "standalone_q":
            matches = _filter_by_duration(
                matches, end_col, prefer=duration_pref,
                target_fp=target_fp, target_fy=target_fy,
            )
        else:
            matches = _filter_by_period_type(matches, end_col, duration_pref)

    if period_index >= len(matches):
        return None

    raw = matches.iloc[period_index][value_col]
    return _safe(raw)


def _filter_by_period_type(
    df: pd.DataFrame,
    end_col: str,
    prefer: str,
) -> pd.DataFrame:
    """Keep flow rows whose duration class matches the preferred period.

    prefer="annual"    → ~12-month durations (FY band, incl. 52/53-week years)
    prefer="quarterly" → the YTD-through-quarter duration per period end
                         (Q1 = 3mo, Q2 = 6mo YTD, Q3 = 9mo YTD)

    "quarterly" deliberately returns YTD, not the standalone 3-month tag:
    income statements carry both durations but cash flows are YTD-only, and
    the in-memory decumulation downstream subtracts the prior quarter from
    EVERY flow metric — that arithmetic is only correct when income and cash
    flow share the same YTD basis. (extract_financials flags these rows
    is_ytd so nothing YTD ever masquerades as standalone.)

    Instant rows (no start) pass through untouched. If the preferred class is
    absent (stub periods), falls back to the closest available duration per
    period-end instead of returning nothing.
    """
    work = annotate_periods(df)
    instant_mask = work["_ptype"] == PeriodType.INSTANT.value
    instants = work[instant_mask]
    flows = work[~instant_mask]
    if flows.empty:
        return df

    if prefer == "annual":
        kept = flows[flows["_ptype"] == PeriodType.YEAR.value]
        if kept.empty:
            # Transition periods / stubs: longest duration per period end —
            # but never a single quarter. ABBV and HON tag GrossProfit ONLY in
            # the 10-K's quarterly-data note (Q4, 92 days); taking it as the
            # "annual" value served FY25 gross profit of $12.1B (real ~$43B)
            # and a −$18.8B synthesized Q4. With no annual-length fact the
            # metric stays None and the Revenue − COGS rollup takes over.
            idx = flows.groupby(end_col)["_dur_days"].idxmax()
            kept = flows.loc[idx]
            kept = kept[kept["_dur_days"] >= 150]
    else:  # quarterly — YTD-through-quarter per period end
        sub = flows[flows["_dur_days"] <= 295]  # exclude full-year rows
        if sub.empty:
            sub = flows
        idx = sub.groupby(end_col)["_dur_days"].idxmax()
        kept = sub.loc[idx]

    result = pd.concat([instants, kept])
    try:
        result = result.sort_values(end_col, ascending=False)
    except Exception as exc:
        log.debug("Sort by %s failed: %s", end_col, exc)
    return result.drop(columns=["_ptype", "_dur_days"], errors="ignore")


def _concept_for_source(concepts: list[ConceptEntry], source: str | None) -> str | None:
    """The XBRL tag behind a resolver `source` label (exact, non-aggregate passes)."""
    for e in concepts or ():
        if not e.aggregate and e.display_name == source:
            return e.xbrl_concept
    return None


def _three_month_fact(facts_df: pd.DataFrame, concept: str, end: pd.Timestamp) -> float | None:
    """The filer's own ~3-month value of `concept` ending at `end`, or None.

    Ambiguity (two different values for the same 3-month context, e.g. a
    segment leak the quality filter missed) returns None — never guess.
    """
    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(facts_df, ("value", "Value", "val"))
    if concept_col is None or value_col is None or "start" not in facts_df.columns:
        return None
    col = facts_df[concept_col].astype(str)
    m = facts_df[(col == concept) | col.str.endswith(f":{concept}") | col.str.endswith(f"_{concept}")]
    if m.empty:
        return None
    m = annotate_periods(m)
    end_dt = pd.to_datetime(m["end"], errors="coerce")
    m = m[((end_dt - end).abs() <= pd.Timedelta(days=4))
          & (m["_dur_days"] >= 80) & (m["_dur_days"] <= 100)]
    if m.empty:
        return None
    vals = pd.to_numeric(dedupe_facts(m)[value_col], errors="coerce").dropna().unique()
    return float(vals[0]) if len(vals) == 1 else None


def _basis_check(facts_df: pd.DataFrame, sourced: dict, metrics: dict,
                 industry: IndustryClass) -> dict | None:
    """This filing's own prior-year comparative for revenue (same tag, same
    duration, one year earlier). Comparing it with how the prior-year filing
    originally reported that period says whether this filing recast history —
    the signal that two filings sit on different bases (spin-off/discontinued
    operations), so subtracting one from the other is meaningless."""
    rev = metrics.get("revenue")
    concept = _concept_for_source(get_revenue_concepts(industry), sourced.get("revenue"))
    if rev is None or not concept:
        return None
    clean = _apply_quality_filters(facts_df)
    concept_col = _find_column(clean, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(clean, ("value", "Value", "val"))
    if concept_col is None or value_col is None or "start" not in clean.columns:
        return None
    col = clean[concept_col].astype(str)
    m = annotate_periods(clean[(col == concept) | col.str.endswith(f":{concept}")
                               | col.str.endswith(f"_{concept}")])
    m = m[m["_dur_days"].notna()]
    if m.empty:
        return None
    m = m.assign(_end=pd.to_datetime(m["end"], errors="coerce"),
                 _val=pd.to_numeric(m[value_col], errors="coerce"))
    cur = m[m["_val"] == rev]
    if cur.empty:
        return None
    end, dur = cur["_end"].max(), cur.loc[cur["_end"].idxmax(), "_dur_days"]
    py = m[((m["_end"] - (end - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=10))
           & ((m["_dur_days"] - dur).abs() <= 20)]
    vals = py["_val"].dropna().unique()
    out = {"period_end": end.date().isoformat(), "duration_days": int(dur)}
    if len(vals) == 1:
        out["revenue_prior_year"] = float(vals[0])
    return out


# Income-statement flows + per-share columns a 10-Q tags at the 3-month context.
_THREE_MONTH_KEYS = (
    "revenue", "cost_of_revenue", "gross_profit", "operating_income", "operating_expenses",
    "sga_expense", "ga_expense", "rd_expense", "interest_expense", "income_tax_expense",
    "other_income_expense", "income_before_tax", "net_income", "provision_for_credit_losses",
    "eps_basic", "eps_diluted", "weighted_shares_basic", "weighted_shares_diluted",
)


def _capture_three_month(
    facts_df: pd.DataFrame,
    metrics: dict,
    sourced: dict,
    industry: IndustryClass,
) -> dict[str, float]:
    """Q2/Q3 10-Qs tag BOTH the YTD and the 3-month context. History rows are
    extracted YTD and decumulated by subtracting the prior 10-Q's YTD, which is
    wrong whenever the two filings sit on different bases — a spin-off or
    divestiture recast between them (WDC Q3 FY25 revenue −$1.47B; APP Q2'25
    $934M vs $1,259M filed) — and it forces EPS to be re-derived instead of
    read (190 of 879 audited quarterly EPS values were off). Read the filer's
    own 3-month figure from the SAME tag that produced each YTD value; anything
    composite or auto-corrected is left to the subtraction path."""
    clean = _apply_quality_filters(facts_df)
    if clean is None or clean.empty or "start" not in clean.columns:
        return {}
    flows = clean[pd.to_datetime(clean["start"], errors="coerce").notna()]
    ends = pd.to_datetime(flows["end"], errors="coerce").dropna()
    if ends.empty:
        return {}
    end = ends.max()
    out: dict[str, float] = {}
    for k in _THREE_MONTH_KEYS:
        if metrics.get(k) is None:
            continue
        if k == "revenue":
            concepts = get_revenue_concepts(industry)
        elif k == "operating_income":
            concepts = get_operating_income_concepts(industry)
        elif k == "ga_expense":
            concepts = [ConceptEntry("GeneralAndAdministrativeExpense", "General & Administrative")]
        else:
            concepts = CONCEPT_MAP.get(k) or []
        concept = _concept_for_source(concepts, sourced.get(k))
        if concept:
            v = _three_month_fact(clean, concept, end)
            if v is not None:
                out[k] = v
    if ("gross_profit" not in out and "revenue" in out and "cost_of_revenue" in out
            and str(sourced.get("gross_profit") or "").startswith("Computed: Revenue")):
        out["gross_profit"] = out["revenue"] - abs(out["cost_of_revenue"])
    return out


# Balance sheet concepts are point-in-time, not flow — no duration filtering needed
_INSTANT_CONCEPTS = {
    "Assets", "Liabilities", "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "LiabilitiesAndStockholdersEquity", "AssetsCurrent", "NoncurrentAssets",
    "LiabilitiesCurrent", "NoncurrentLiabilities",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    "ShortTermInvestments", "MarketableSecuritiesCurrent",
    "LongTermDebt", "LongTermDebtNoncurrent", "ShortTermBorrowings", "DebtCurrent",
    "CommonStockSharesOutstanding", "SharesOutstanding",
    "RetainedEarningsAccumulatedDeficit", "TotalAssets", "TotalLiabilities",
    "AccountsReceivableNetCurrent", "AccountsReceivableNet",
    "InventoryNet", "Inventory", "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",
    "PropertyPlantAndEquipmentNet",
    "AccountsPayableCurrent", "AccountsPayable",
    "OperatingLeaseRightOfUseAsset",
    "OperatingLeaseLiabilityCurrent", "OperatingLeaseLiabilityNoncurrent",
    "CommonStockValue", "AdditionalPaidInCapital", "TreasuryStockValue",
    "MinorityInterest", "Equity", "CommonStockholdersEquity",
    # Bank-specific
    "Deposits", "LoansAndLeasesReceivableNetReportedAmount",
    "AvailableForSaleSecurities", "HeldToMaturitySecurities",
    "FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase",
    "OtherAssetsNoncurrent", "OtherAssetsCurrent",
    "OtherLiabilitiesNoncurrent", "OtherLiabilitiesCurrent",
    "AccruedLiabilitiesCurrent", "DeferredRevenueCurrent",
    "LongTermDebtCurrent", "CommercialPaper",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
    "PrepaidExpenseAndOtherAssetsCurrent",
    "DeferredIncomeTaxAssetsNet", "DeferredIncomeTaxLiabilitiesNet",
    "LongTermInvestments", "InvestmentsAndAdvances",
    # IFRS instant concepts
    "CurrentAssets", "CurrentLiabilities",
    "EquityAttributableToOwnersOfParent",
    "CashAndCashEquivalents",
    "NoncurrentPortionOfNoncurrentBorrowings", "LongTermBorrowings",
    "CurrentPortionOfNoncurrentBorrowings", "CurrentBorrowings",
    "Cash",
}


# SEC fp field → valid duration ranges (days).  Research source: XBRL US / EDGAR EFM.
# fp="Q3" means the 10-Q covers 9 months YTD — do NOT expect a 90-day standalone value.
_FP_DURATION_RANGES: dict[str, tuple[int, int]] = {
    "Q1": (65, 115),    # ~91 days standalone first quarter
    "Q2": (155, 205),   # ~182 days YTD through second quarter
    "H1": (155, 205),   # synonym for Q2 period in some filings
    "Q3": (245, 295),   # ~273 days YTD through third quarter
    "M9": (245, 295),   # synonym for Q3 period
    "FY": (340, 395),   # full fiscal year (52/53-week filers: 363-371 days)
    "CY": (340, 395),   # calendar year
    "H2": (155, 210),   # second half (rare)
    "Q4": (65, 115),    # standalone Q4 (rarely tagged separately)
}

# fp values that correspond to 10-Q filings (duration > 1 quarter)
_FP_QUARTERLY_FORMS = {"Q1", "Q2", "H1", "Q3", "M9"}
_FP_ANNUAL_FORMS = {"FY", "CY"}


def _filter_by_fp(
    df: pd.DataFrame,
    target_fp: str,
    target_fy: int | None = None,
) -> pd.DataFrame:
    """Filter facts using the authoritative `fp` (fiscal period) field.

    This is the primary period-selection method — more reliable than duration
    calculation because fp is directly declared by the filer in DEI tags.

    Args:
        target_fp: The fiscal period to target (e.g. "Q3", "FY").
        target_fy:  The fiscal year to target (company-defined integer).
    """
    if "fp" not in df.columns:
        return df  # fp column not present — caller should fall back to duration

    fp_col = "fp"
    filtered = df[df[fp_col].astype(str).str.upper() == target_fp.upper()]

    if target_fy is not None and "fy" in df.columns:
        filtered = filtered[pd.to_numeric(filtered["fy"], errors="coerce") == target_fy]

    # Keep instant rows (no start date — balance sheet) unconditionally
    if "start" in filtered.columns:
        instant_mask = filtered["start"].isna()
        duration_rows = filtered[~instant_mask]
        instant_rows = filtered[instant_mask]
        # Validate duration is consistent with fp to catch mislabelled facts
        if not duration_rows.empty and "end" in duration_rows.columns:
            lo, hi = _FP_DURATION_RANGES.get(target_fp.upper(), (0, 999))
            dur = (
                pd.to_datetime(duration_rows["end"], errors="coerce")
                - pd.to_datetime(duration_rows["start"], errors="coerce")
            ).dt.days
            valid_dur = dur.between(lo, hi) | dur.isna()
            duration_rows = duration_rows[valid_dur]
        filtered = pd.concat([instant_rows, duration_rows], ignore_index=True)

    return filtered if not filtered.empty else df


def _filter_by_duration(
    df: pd.DataFrame,
    end_col: str,
    prefer: str = "quarterly",
    target_fp: str | None = None,
    target_fy: int | None = None,
) -> pd.DataFrame:
    """Filter facts to prefer quarterly or annual durations.

    Primary method: if target_fp is provided, use the SEC fp field directly —
    this is authoritative (declared by the filer) and avoids duration math.

    Fallback: duration-day calculation for when fp is unavailable.

    Note: For 10-Q filings:
     - Q1 cash flows are ~91-day standalone values
     - Q2/Q3 cash flows are YTD cumulative (~182/273-day) — this is correct GAAP
       behaviour; the SEC does NOT require standalone quarter cash flows
    """
    if "start" not in df.columns:
        return df

    # ── Standalone 3-month quarter selection (10-Q flow metrics) ────────────
    # 10-Qs tag BOTH the YTD duration (e.g. Q3 = 273 days) AND the standalone
    # 3-month quarter (~90 days) under the SAME fp — so fp-based selection keeps
    # the YTD. For Fineas we want the standalone quarter: among facts at the
    # latest period-end, take the shortest (~3-month) duration. The filer has
    # already done the decumulation; we just pick the right context. If a filer
    # only tagged YTD (no <=100-day fact), we fall through to the YTD value.
    if prefer == "standalone_q":
        work = df.copy()
        start_dt = pd.to_datetime(work["start"], errors="coerce")
        end_dt = pd.to_datetime(work[end_col], errors="coerce")
        work["_dur"] = (end_dt - start_dt).dt.days
        is_instant = work["start"].isna() | work["_dur"].isna()
        instant_rows = work[is_instant]
        dur_rows = work[~is_instant]
        if not dur_rows.empty:
            if target_fy is not None and "fy" in dur_rows.columns:
                fy_match = dur_rows[pd.to_numeric(dur_rows["fy"], errors="coerce") == target_fy]
                if not fy_match.empty:
                    dur_rows = fy_match
            latest_end = pd.to_datetime(dur_rows[end_col], errors="coerce").max()
            at_latest = dur_rows[pd.to_datetime(dur_rows[end_col], errors="coerce") == latest_end]
            # Take the YTD-through-this-quarter = the LONGEST duration ending at
            # the current period end (uniform for income AND cash-flow, whether or
            # not the filer also tagged a standalone 3-month context). The caller
            # then decumulates (currentYTD - priorYTD) to get the standalone
            # quarter — never double-counting a directly-tagged 3-month value.
            picked = at_latest
            concept_col = _find_column(at_latest, ("concept", "Concept", "tag", "Tag"))
            if concept_col is not None:
                idx = at_latest.groupby(concept_col)["_dur"].idxmax()
                picked = at_latest.loc[idx]
            dur_rows = picked
        result = pd.concat([instant_rows, dur_rows], ignore_index=True)
        result.drop(columns=["_dur"], inplace=True, errors="ignore")
        if end_col in result.columns:
            result = result.sort_values(end_col, ascending=False)
        return result

    # ── Primary: fp-based selection ─────────────────────────────────────────
    if target_fp and "fp" in df.columns:
        result = _filter_by_fp(df, target_fp, target_fy)
        if not result.empty and len(result) < len(df):
            if end_col in result.columns:
                result = result.sort_values(end_col, ascending=False)
            return result

    # ── Fallback: duration-day heuristic ────────────────────────────────────
    df = df.copy()
    start_dt = pd.to_datetime(df["start"], errors="coerce")
    end_dt = pd.to_datetime(df[end_col], errors="coerce")
    df["_duration_days"] = (end_dt - start_dt).dt.days

    instant_mask = df["start"].isna() | df["_duration_days"].isna()
    instant_rows = df[instant_mask]
    duration_rows = df[~instant_mask]

    if duration_rows.empty:
        df.drop(columns=["_duration_days"], inplace=True, errors="ignore")
        return df

    if prefer == "quarterly":
        # Keep rows with duration consistent with Q1 (65-115 days) if available.
        # For Q2/Q3 filings the only available facts are YTD — that's correct.
        short = duration_rows[duration_rows["_duration_days"] <= 115]
        if not short.empty:
            duration_rows = short
        else:
            # No standalone-quarter data — keep shortest available (YTD is fine)
            concept_col = _find_column(duration_rows, ("concept", "Concept", "tag", "Tag"))
            if concept_col and end_col:
                idx = duration_rows.groupby([concept_col, end_col])["_duration_days"].idxmin()
                duration_rows = duration_rows.loc[idx]
    else:
        # Annual: keep the longest duration (full-year, fp=FY)
        long = duration_rows[duration_rows["_duration_days"] > 300]
        if not long.empty:
            duration_rows = long

    result = pd.concat([instant_rows, duration_rows], ignore_index=True)
    result.drop(columns=["_duration_days"], inplace=True, errors="ignore")
    if end_col in result.columns:
        result = result.sort_values(end_col, ascending=False)
    return result


def _filter_to_currency(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    """Keep monetary facts in `currency` only; non-monetary units (shares,
    pure, ratios) pass through. Currency-per-share units (e.g. 'TWD/shares')
    are kept when their currency part matches."""
    if df is None or df.empty or "units" not in df.columns:
        return df

    def _keep(u: str) -> bool:
        u = str(u).replace("iso4217:", "")
        cur = u.split("/")[0]
        if not (len(cur) == 3 and cur.isalpha() and cur.isupper()):
            return True  # shares, pure, etc.
        return cur == currency

    mask = df["units"].map(_keep)
    filtered = df[mask]
    return filtered if not filtered.empty else df


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Segment extraction (revenue by product/service + geographic)
# ═══════════════════════════════════════════════════════════════════════════

def _clean_segment_label(concept: str) -> str:
    """Convert a CamelCase XBRL concept name to a readable label.

    Strips common revenue prefixes and inserts spaces between words.
    E.g. "AdvertisingRevenue" → "Advertising Revenue"
         "OnlineStoresRevenue" → "Online Stores"
    """
    import re
    # Strip long common prefixes first
    prefixes = [
        r"^RevenueFromContractWithCustomerExcludingAssessedTax",
        r"^RevenueFromContractWithCustomer",
        r"^NetRevenueFrom",
        r"^ProductsAndServicesRevenue",
        r"^SalesRevenue",
    ]
    label = concept
    for p in prefixes:
        cleaned = re.sub(p, "", label, flags=re.IGNORECASE)
        if cleaned and len(cleaned) < len(label):
            label = cleaned
            break
    # Strip trailing "Revenue"/"Revenues"/"Net" if there's still content
    label = re.sub(r"Revenue[s]?$", "", label).strip()
    # Insert spaces before uppercase letter that follows lowercase (CamelCase split)
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    # Handle sequences of caps followed by lower (e.g. "USRevenue" → "US Revenue")
    label = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", label)
    label = label.strip()
    return label if label else concept


def _extract_segments(
    facts_df: pd.DataFrame | None,
    target_fp: str | None = None,
    target_fy: int | None = None,
) -> dict:
    """Extract revenue and geographic segments from XBRL facts.

    The SEC companyfacts API does not include XBRL dimensional context
    (segment axes), so true segment breakdowns are unavailable for most
    large-cap filers. However, some companies tag separate business/geo
    revenue as distinct top-level XBRL concepts — those are captured here.

    Returns a dict with:
      revenue_segments:    [{segment, value, pct}, ...]  — product/business breakdown
      geographic_segments: [{segment, value, pct}, ...]  — geographic breakdown
      available:           bool — True if any non-trivial segment data found
      note:                str  — human-readable explanation for UI
    """
    result: dict = {
        "revenue_segments": [],
        "geographic_segments": [],
        "available": False,
        "note": (
            "Segment data requires XBRL dimensional context which is not "
            "available in the SEC companyfacts API. View the filing directly "
            "in the EDGAR Interactive Viewer for segment breakdowns."
        ),
    }

    if facts_df is None or facts_df.empty:
        return result

    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(facts_df, ("value", "Value", "val"))
    end_col = _find_column(facts_df, ("end", "period"))
    if concept_col is None or value_col is None:
        return result

    # Filter to current period facts when fp/fy are known
    working = facts_df.copy()
    if target_fp and "fp" in working.columns:
        fp_rows = working[working["fp"].astype(str).str.upper() == target_fp.upper()]
        if not fp_rows.empty:
            working = fp_rows
    if target_fy and "fy" in working.columns:
        fy_rows = working[pd.to_numeric(working["fy"], errors="coerce") == target_fy]
        if not fy_rows.empty:
            working = fy_rows

    # Sort newest first
    if end_col and end_col in working.columns:
        working = working.sort_values(end_col, ascending=False)

    # ── Geographic concept name → display label mapping ────────────────────
    GEO_MAP: list[tuple[str, str]] = [
        # Americas
        ("UnitedStates",           "United States"),
        ("Domestic",               "United States"),
        ("NorthAmerica",           "North America"),
        ("Americas",               "Americas"),
        ("LatinAmerica",           "Latin America"),
        ("SouthAmerica",           "South America"),
        ("Brazil",                 "Brazil"),
        ("Mexico",                 "Mexico"),
        ("Canada",                 "Canada"),
        ("Argentina",              "Argentina"),
        ("Colombia",               "Colombia"),
        ("Chile",                  "Chile"),
        # Europe
        ("Europe",                 "Europe"),
        ("EMEA",                   "EMEA"),
        ("UnitedKingdom",          "United Kingdom"),
        ("UK",                     "United Kingdom"),
        ("Germany",                "Germany"),
        ("France",                 "France"),
        ("Netherlands",            "Netherlands"),
        ("Ireland",                "Ireland"),
        ("Switzerland",            "Switzerland"),
        ("Italy",                  "Italy"),
        ("Spain",                  "Spain"),
        ("Sweden",                 "Sweden"),
        ("Norway",                 "Norway"),
        ("Denmark",                "Denmark"),
        ("Finland",                "Finland"),
        ("Belgium",                "Belgium"),
        ("Austria",                "Austria"),
        ("Poland",                 "Poland"),
        ("EuropeanUnion",          "European Union"),
        ("WesternEurope",          "Western Europe"),
        ("EasternEurope",          "Eastern Europe"),
        # Asia Pacific
        ("GreaterChina",           "Greater China"),
        ("ChinaMainland",          "China"),
        ("China",                  "China"),
        ("HongKong",               "Hong Kong"),
        ("Taiwan",                 "Taiwan"),
        ("Japan",                  "Japan"),
        ("Korea",                  "South Korea"),
        ("SouthKorea",             "South Korea"),
        ("India",                  "India"),
        ("AsiaPacific",            "Asia Pacific"),
        ("APAC",                   "Asia Pacific"),
        ("Australia",              "Australia"),
        ("NewZealand",             "New Zealand"),
        ("Singapore",              "Singapore"),
        ("Malaysia",               "Malaysia"),
        ("Thailand",               "Thailand"),
        ("Indonesia",              "Indonesia"),
        ("Vietnam",                "Vietnam"),
        ("Philippines",            "Philippines"),
        ("SoutheastAsia",          "Southeast Asia"),
        ("RestOfAsia",             "Rest of Asia"),
        # Middle East & Africa
        ("MiddleEast",             "Middle East"),
        ("MiddleEastAndAfrica",    "Middle East & Africa"),
        ("Africa",                 "Africa"),
        ("SouthAfrica",            "South Africa"),
        ("Israel",                 "Israel"),
        ("SaudiArabia",            "Saudi Arabia"),
        ("UAE",                    "UAE"),
        ("UnitedArabEmirates",     "UAE"),
        ("Turkey",                 "Turkey"),
        ("Egypt",                  "Egypt"),
        ("Nigeria",                "Nigeria"),
        # Generic
        ("International",          "International"),
        ("RestOfWorld",            "Rest of World"),
        ("AllOtherCountries",      "Other Countries"),
        ("OtherCountries",         "Other Countries"),
        ("OtherGeographic",        "Other"),
        ("ForeignCountries",       "International"),
        ("OutsideUnitedStates",    "International"),
    ]
    GEO_KEYS = {k.lower() for k, _ in GEO_MAP}

    # ── Revenue segment concept patterns (product/business) ───────────────
    # These match standalone segment revenue tags some companies use.
    # IMPORTANT: concept must also contain "Revenue" or "Sales" or "Segment"
    # to avoid matching expense/fee/liability concepts.
    REV_SEG_PATTERNS = [
        # Tech product lines
        "DataCenter", "ComputeAndNetworking", "Gaming", "Automotive",
        "Visualization", "CloudRevenue",
        "OnlineStore", "PhysicalStore", "ThirdPartySeller",
        "MacRevenue", "iPhoneRevenue", "iPadRevenue", "WearableRevenue",
        "ProductRevenue", "ServiceRevenue",
        # Media / ad
        "AdvertisingRevenue", "SubscriptionRevenue",
        # Broad segments
        "Retail", "Wholesale",
        # Banking / financial
        "InstitutionalSecurities", "WealthManagement", "InvestmentManagement",
        "InvestmentBanking", "TradingRevenue", "AssetManagement",
        "CommercialBanking", "ConsumerBanking", "CorporateBanking",
        "GlobalBanking", "GlobalMarkets", "CardServices",
        "NetInterestIncome", "NonInterestRevenue",
        # Insurance
        "PremiumRevenue", "Underwriting",
        # Generic
        "OtherRevenue", "OtherSalesRevenue",
    ]

    # Words that MUST appear in the concept for it to be a revenue segment
    _REV_QUALIFIERS = {"revenue", "sales", "income", "segment", "net"}

    # Words that disqualify a concept from being a revenue segment
    _REV_BLOCKERS = {
        "expense", "cost", "fee", "loss", "liability", "payable",
        "depreciation", "amortization", "provision", "impairment",
        "accrued", "deferred", "allowance", "reserve", "tax",
        "receivable", "asset", "inventory", "equity", "dividend",
        "compensation", "stock", "share", "warrant", "option",
    }

    # Concept names that are almost certainly the TOTAL (not a sub-segment)
    TOTAL_CONCEPTS = {
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "Revenue", "NetRevenues", "TotalRevenues",
        "SalesRevenueNet", "OperatingRevenue",
    }

    seen_rev: dict[str, float] = {}     # label → value (keep highest val for dedup)
    seen_geo: dict[str, float] = {}

    for _, row in working.iterrows():
        concept_name = str(row[concept_col])
        val = _safe(row[value_col])
        if val is None or val <= 0:
            continue
        # Skip obvious total concepts
        if concept_name in TOTAL_CONCEPTS:
            continue

        concept_lo = concept_name.lower()

        # Quick blocker check — skip expense/liability concepts entirely
        if any(b in concept_lo for b in _REV_BLOCKERS):
            continue

        # ── Geographic match ──────────────────────────────────────────────
        # Require the concept to also contain a revenue-like word
        matched_geo = False
        has_rev_word = any(q in concept_lo for q in _REV_QUALIFIERS)
        if has_rev_word:
            for key, display in GEO_MAP:
                if key.lower() in concept_lo:
                    if display not in seen_geo or val > seen_geo[display]:
                        seen_geo[display] = val
                    matched_geo = True
                    break

        if not matched_geo:
            # ── Product/business segment match ────────────────────────────
            # Concept must contain a revenue qualifier OR be a known segment pattern
            for pat in REV_SEG_PATTERNS:
                if pat.lower() in concept_lo:
                    # Require revenue qualifier for generic patterns
                    if not has_rev_word and pat.lower() not in concept_lo:
                        continue
                    label = _clean_segment_label(concept_name) or pat
                    if label not in seen_rev or val > seen_rev[label]:
                        seen_rev[label] = val
                    break

    def _pct_list(d: dict[str, float]) -> list[dict]:
        total = sum(d.values()) or 1
        return sorted(
            [{"segment": k, "value": v, "pct": round(v / total * 100, 1)}
             for k, v in d.items()],
            key=lambda x: x["value"],
            reverse=True,
        )

    rev_list = _pct_list(seen_rev)
    geo_list = _pct_list(seen_geo)

    result["revenue_segments"] = rev_list
    result["geographic_segments"] = geo_list

    has_data = len(rev_list) >= 2 or len(geo_list) >= 2
    result["available"] = has_data
    if has_data:
        result["note"] = ""

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Filing-text geographic revenue parser (fallback when XBRL has no geo data)
# ═══════════════════════════════════════════════════════════════════════════

# Known region/country names to match in filing text tables
_GEO_TEXT_LABELS: list[tuple[re.Pattern, str]] = [
    # Order matters — more specific patterns first
    (re.compile(r"\bUnited\s+States\b", re.I),               "United States"),
    (re.compile(r"\bU\.?S\.?A?\.?\b"),                        "United States"),
    (re.compile(r"\bNorth\s+America\b", re.I),                "North America"),
    (re.compile(r"\bLatin\s+America\b", re.I),                "Latin America"),
    (re.compile(r"\bSouth\s+America\b", re.I),                "South America"),
    (re.compile(r"\bAmericas\b", re.I),                       "Americas"),
    (re.compile(r"\bCanada\b", re.I),                         "Canada"),
    (re.compile(r"\bMexico\b", re.I),                         "Mexico"),
    (re.compile(r"\bBrazil\b", re.I),                         "Brazil"),
    (re.compile(r"\bArgentina\b", re.I),                      "Argentina"),
    (re.compile(r"\bColombia\b", re.I),                       "Colombia"),
    (re.compile(r"\bChile\b", re.I),                          "Chile"),
    (re.compile(r"\bUnited\s+Kingdom\b", re.I),               "United Kingdom"),
    (re.compile(r"\bU\.?K\.?\b"),                              "United Kingdom"),
    (re.compile(r"\bGermany\b", re.I),                        "Germany"),
    (re.compile(r"\bFrance\b", re.I),                         "France"),
    (re.compile(r"\bNetherlands\b", re.I),                    "Netherlands"),
    (re.compile(r"\bIreland\b", re.I),                        "Ireland"),
    (re.compile(r"\bSwitzerland\b", re.I),                    "Switzerland"),
    (re.compile(r"\bItaly\b", re.I),                          "Italy"),
    (re.compile(r"\bSpain\b", re.I),                          "Spain"),
    (re.compile(r"\bSweden\b", re.I),                         "Sweden"),
    (re.compile(r"\bNorway\b", re.I),                         "Norway"),
    (re.compile(r"\bDenmark\b", re.I),                        "Denmark"),
    (re.compile(r"\bFinland\b", re.I),                        "Finland"),
    (re.compile(r"\bBelgium\b", re.I),                        "Belgium"),
    (re.compile(r"\bAustria\b", re.I),                        "Austria"),
    (re.compile(r"\bPoland\b", re.I),                         "Poland"),
    (re.compile(r"\bWestern\s+Europe\b", re.I),               "Western Europe"),
    (re.compile(r"\bEastern\s+Europe\b", re.I),               "Eastern Europe"),
    (re.compile(r"\bEMEA\b"),                                  "EMEA"),
    (re.compile(r"\bEurope,?\s*Middle\s+East", re.I),         "EMEA"),
    (re.compile(r"\bEurope\b", re.I),                         "Europe"),
    (re.compile(r"\bGreater\s*China\b", re.I),                "Greater China"),
    (re.compile(r"\bMainland\s+China\b", re.I),               "China"),
    (re.compile(r"\bChina\b", re.I),                          "China"),
    (re.compile(r"\bHong\s+Kong\b", re.I),                    "Hong Kong"),
    (re.compile(r"\bTaiwan\b", re.I),                         "Taiwan"),
    (re.compile(r"\bJapan\b", re.I),                          "Japan"),
    (re.compile(r"\bSouth\s+Korea\b", re.I),                  "South Korea"),
    (re.compile(r"\bKorea\b", re.I),                          "South Korea"),
    (re.compile(r"\bIndia\b", re.I),                          "India"),
    (re.compile(r"\bAsia\s*[/-]?\s*Pacific\b", re.I),         "Asia Pacific"),
    (re.compile(r"\bAPAC\b"),                                  "Asia Pacific"),
    (re.compile(r"\bAustralia\b", re.I),                      "Australia"),
    (re.compile(r"\bNew\s+Zealand\b", re.I),                  "New Zealand"),
    (re.compile(r"\bSingapore\b", re.I),                      "Singapore"),
    (re.compile(r"\bMalaysia\b", re.I),                       "Malaysia"),
    (re.compile(r"\bThailand\b", re.I),                       "Thailand"),
    (re.compile(r"\bIndonesia\b", re.I),                      "Indonesia"),
    (re.compile(r"\bVietnam\b", re.I),                        "Vietnam"),
    (re.compile(r"\bPhilippines\b", re.I),                    "Philippines"),
    (re.compile(r"\bSoutheast\s+Asia\b", re.I),               "Southeast Asia"),
    (re.compile(r"\bMiddle\s+East\s*(and|&)\s*Africa\b", re.I), "Middle East & Africa"),
    (re.compile(r"\bMiddle\s+East\b", re.I),                  "Middle East"),
    (re.compile(r"\bAfrica\b", re.I),                         "Africa"),
    (re.compile(r"\bSouth\s+Africa\b", re.I),                 "South Africa"),
    (re.compile(r"\bIsrael\b", re.I),                         "Israel"),
    (re.compile(r"\bSaudi\s+Arabia\b", re.I),                 "Saudi Arabia"),
    (re.compile(r"\bUnited\s+Arab\s+Emirates\b", re.I),       "UAE"),
    (re.compile(r"\bUAE\b"),                                   "UAE"),
    (re.compile(r"\bTurkey\b", re.I),                         "Turkey"),
    (re.compile(r"\bTürkiye\b", re.I),                        "Turkey"),
    (re.compile(r"\bEgypt\b", re.I),                          "Egypt"),
    (re.compile(r"\bNigeria\b", re.I),                        "Nigeria"),
    (re.compile(r"\bRest\s+of\s+(?:the\s+)?World\b", re.I),  "Rest of World"),
    (re.compile(r"\bInternational\b", re.I),                  "International"),
    (re.compile(r"\bOther\s+Countries\b", re.I),              "Other Countries"),
    (re.compile(r"\bAll\s+Other\b", re.I),                    "Other"),
    (re.compile(r"\bRest\s*of\s*Asia\b", re.I),              "Rest of Asia"),
    (re.compile(r"\bRest\s+of\s+Europe\b", re.I),            "Rest of Europe"),
    (re.compile(r"\bRest\s+of\s+Americas\b", re.I),          "Rest of Americas"),
]

# Section headers that signal a geographic revenue table
_GEO_SECTION_RE = re.compile(
    r"(?:geographic|geographical)\s+(?:information|areas?|breakdown|data)"
    r"|revenue\s+by\s+(?:geography|region|country|area)"
    r"|(?:revenue|net\s+sales|net\s+revenues)\s+(?:by|from)\s+(?:geographic|geographical)"
    r"|disaggregation\s+of\s+revenue"
    r"|revenue\s+(?:from|earned\s+in)\s+(?:external\s+)?customers?\s+by\s+geography"
    r"|information\s+by\s+geographic\s+area"
    r"|(?:net\s+)?(?:revenue|sales)\s+by\s+reportable\s+segment\s+and\s+geograph",
    re.IGNORECASE,
)

# Dollar amount pattern: $12,345 or 12,345 or $12.3 (context-dependent)
_DOLLAR_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"   # e.g. $12,345.6 or 12,345
    r"|\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|m|k)"  # e.g. $12.3 billion
    , re.IGNORECASE,
)


def _parse_dollar_amount(match: re.Match) -> float | None:
    """Convert a dollar regex match to a numeric value."""
    if match.group(1):
        # Plain number like 12,345 or 12,345.6
        return float(match.group(1).replace(",", ""))
    if match.group(2) and match.group(3):
        base = float(match.group(2))
        suffix = match.group(3).lower()
        multipliers = {
            "billion": 1_000_000_000, "bn": 1_000_000_000,
            "million": 1_000_000, "mm": 1_000_000, "m": 1_000_000,
            "thousand": 1_000, "k": 1_000,
        }
        return base * multipliers.get(suffix, 1)
    return None


def _parse_geo_from_filing_text(
    text: str,
    total_revenue: float | None = None,
) -> list[dict]:
    """Extract geographic revenue breakdown from 10-K filing text.

    Handles two common table formats:
    1. Row-based: "Americas ... $178,353" (label + value on same line)
    2. Columnar: headers on one line, "Net sales $X $Y $Z" on next line
       (common in AAPL-style segment tables)

    Returns list of {segment, value, pct} dicts sorted by value descending.
    """
    if not text or len(text) < 200:
        return []

    results: dict[str, float] = {}  # display_label → dollar value

    # Find all potential geo-revenue sections
    section_starts: list[int] = []
    for m in _GEO_SECTION_RE.finditer(text):
        section_starts.append(m.start())

    if not section_starts:
        return []

    # ── Detect scale from the BEST section (one with actual table data) ──
    scale = 1.0
    # We'll scan broader windows for scale markers
    for start in section_starts:
        header_window = text[max(0, start - 300):start + 1500].lower()
        if "in millions" in header_window or "(millions)" in header_window:
            scale = 1_000_000
            break
        elif "in thousands" in header_window or "(thousands)" in header_window:
            scale = 1_000
            break
        elif "in billions" in header_window or "(billions)" in header_window:
            scale = 1_000_000_000
            break

    # ── Parse each section ──
    columnar_found = False
    for start in section_starts:
        window = text[start:start + 4000]
        lines = window.split("\n")

        # ── Strategy 1: Columnar table ──
        # Look for a header line with 2+ geo labels, then a "Net sales" or
        # revenue line with corresponding dollar values
        if _try_columnar_parse(lines, results):
            columnar_found = True

    # ── Strategy 2: Row-based (label + value on same line) ──
    # Only try this if columnar parsing didn't find anything
    if not columnar_found:
        for start in section_starts:
            window = text[start:start + 4000]
            lines = window.split("\n")
            _try_row_parse(lines, results)

    if not results:
        return []

    # If no explicit scale found, infer from total_revenue comparison
    if scale == 1.0 and total_revenue is not None and total_revenue > 0:
        raw_total = sum(results.values())
        if raw_total > 0:
            ratio = total_revenue / raw_total
            if ratio > 500_000:
                scale = 1_000_000
            elif ratio > 500:
                scale = 1_000

    # Apply scale
    if scale != 1.0:
        results = {k: v * scale for k, v in results.items()}

    # Remove entries that are subsets of other entries (e.g. "China" when
    # "Greater China" exists, or "Asia" when "Rest of Asia" exists)
    _SUBSET_PAIRS = [
        ("China", "Greater China"),
        ("Asia", "Rest of Asia"),
        ("Asia", "Asia Pacific"),
        ("Asia", "Southeast Asia"),
        ("Europe", "Rest of Europe"),
        ("Americas", "Rest of Americas"),
    ]
    for child, parent in _SUBSET_PAIRS:
        if child in results and parent in results:
            del results[child]

    # Build output with percentages
    total = sum(results.values()) or 1
    segments = sorted(
        [{"segment": k, "value": v, "pct": round(v / total * 100, 1)}
         for k, v in results.items()],
        key=lambda x: x["value"],
        reverse=True,
    )

    return segments


def _try_columnar_parse(lines: list[str], results: dict[str, float]) -> bool:
    """Parse columnar geo tables (headers on one line, values on the next).

    Example (AAPL format):
        Americas  Europe  GreaterChina  Japan  Rest of Asia Pacific  Total
        Net sales $ 178,353  $ 111,032  $ 64,377  $ 28,703  $ 33,696  $ 416,161
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line has 2+ geo labels and NO dollar amounts
        # (candidate column header line)
        geo_matches_on_line: list[str] = []
        for pat, display in _GEO_TEXT_LABELS:
            if pat.search(stripped):
                geo_matches_on_line.append(display)

        if len(geo_matches_on_line) < 2:
            continue

        # This line has multiple geo labels — look at subsequent lines for dollar values
        dollar_line = None
        for j in range(i + 1, min(i + 6, len(lines))):
            candidate = lines[j].strip()
            if not candidate:
                continue
            # Must have dollar amounts and mention revenue/sales or just have $
            dollars = list(_DOLLAR_RE.finditer(candidate))
            if len(dollars) >= 2 and (
                re.search(r"net\s+sales|revenue|total", candidate, re.I) or
                "$" in candidate
            ):
                dollar_line = candidate
                break

        if not dollar_line:
            continue

        # Extract dollar values from the value line
        dollars = list(_DOLLAR_RE.finditer(dollar_line))

        # Filter out "Total" from geo labels — it's not a region
        geo_labels = [g for g in geo_matches_on_line
                      if g.lower() not in ("other", "all other")]

        # Try to match each geo label position to dollar values.
        # The trick: find positions of labels on the header line, and positions
        # of dollar amounts on the value line, then pair by column position.

        # Get character positions of each geo label on the header line
        label_positions: list[tuple[int, str]] = []
        for pat, display in _GEO_TEXT_LABELS:
            m = pat.search(stripped)
            if m and display in geo_labels:
                label_positions.append((m.start(), display))
        label_positions.sort(key=lambda x: x[0])

        # Get character positions of dollar amounts on the value line
        dollar_positions: list[tuple[int, float]] = []
        for dm in dollars:
            val = _parse_dollar_amount(dm)
            if val is not None and val > 0:
                dollar_positions.append((dm.start(), val))

        # Skip the last dollar amount if it looks like a total
        # (usually "Total" is the rightmost column)
        if (len(dollar_positions) == len(label_positions) + 1 and
                re.search(r"\btotal\b", stripped, re.I)):
            dollar_positions = dollar_positions[:-1]

        # Pair labels to values by position order
        n_pairs = min(len(label_positions), len(dollar_positions))
        if n_pairs >= 2:
            for idx in range(n_pairs):
                label = label_positions[idx][1]
                val = dollar_positions[idx][1]
                if label not in results or val > results[label]:
                    results[label] = val
            return True  # Columnar parse succeeded

    return False


def _try_row_parse(lines: list[str], results: dict[str, float]) -> None:
    """Parse row-based geo tables (label + value on same line).

    Only matches lines where the geo label appears near the START of the line
    (within first ~60 chars) to avoid matching random mentions of countries
    in descriptive text like investment tables.

    Example:
        Americas    $178,353
        Europe      $111,032
    """
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) > 300:
            continue

        # Skip total/header lines
        if re.match(r"^\s*Total\b", stripped, re.I):
            continue

        # Try to match a known geo label — require it near the START of the line
        matched_label: str | None = None
        for pat, display in _GEO_TEXT_LABELS:
            m = pat.search(stripped)
            if m and m.start() < 60:
                matched_label = display
                break

        if not matched_label:
            continue

        # Find dollar amounts on the same line
        dollar_matches = list(_DOLLAR_RE.finditer(stripped))
        if not dollar_matches:
            continue

        # Take the FIRST dollar amount (most recent year usually leftmost)
        val = _parse_dollar_amount(dollar_matches[0])
        if val is None or val <= 0:
            continue

        # Reject suspiciously small values or values that look like years
        if 1900 <= val <= 2100:
            continue

        # Only keep if the label and the dollar amount are "close" —
        # skip lines where there's a lot of unrelated text between them
        if matched_label not in results or val > results[matched_label]:
            results[matched_label] = val


# ═══════════════════════════════════════════════════════════════════════════
#  Filing-text revenue segment parser (fallback when XBRL has < 2 segments)
# ═══════════════════════════════════════════════════════════════════════════

_SEG_SECTION_RE = re.compile(
    r"(?:net\s+)?(?:revenue|sales|net\s+revenues)\s+by\s+(?:segment|business|reportable\s+segment|operating\s+segment)"
    r"|reportable\s+segment\s+(?:information|data|results)"
    r"|segment\s+(?:information|data|results|reporting|net\s+revenues)"
    r"|(?:results\s+of\s+)?operations\s+by\s+(?:segment|business\s+segment)"
    r"|(?:business|operating)\s+segment\s+(?:net\s+)?(?:revenue|sales)"
    r"|business\s+segments\s*[-—–:]\s*"
    r"|(?:net\s+)?revenues?\s+(?:by|from)\s+(?:business|operating)\s+segment"
    r"|selected\s+financial\s+(?:data|information)\s+by\s+segment"
    r"|segment\s+(?:net\s+)?revenues"
    r"|(?:the\s+)?following\s+table.*?(?:net\s+)?revenues?\s+.*?segment",
    re.IGNORECASE,
)

# Blocklist: lines that look like segment names but aren't
_SEG_BLOCKLIST = re.compile(
    r"\b(?:total|consolidat|eliminat|intersegment|corporate|reconcil|"
    r"unallocated|adjustment|other\s+reconcil|interest\s+expense)\b",
    re.IGNORECASE,
)


def _parse_segments_from_filing_text(
    text: str,
    total_revenue: float | None = None,
) -> list[dict]:
    """Extract revenue segment breakdown from 10-K filing text.

    Looks for segment revenue tables and parses segment names + dollar values.
    Returns list of {segment, value, pct} dicts sorted by value descending.
    """
    if not text or len(text) < 200:
        return []

    results: dict[str, float] = {}

    section_starts: list[int] = []
    for m in _SEG_SECTION_RE.finditer(text):
        section_starts.append(m.start())

    if not section_starts:
        # No explicit segment table headers — try the header-based approach
        _try_segment_header_parse(text, results)

    # Detect scale — check section headers first, then broader text
    scale = 1.0
    scale_windows = [text[max(0, s - 300):s + 1500].lower() for s in section_starts]
    if not scale_windows:
        # Search the first 5000 chars for scale indicators (often in MDA header)
        scale_windows = [text[:5000].lower()]
    for hw in scale_windows:
        if "in millions" in hw or "(millions)" in hw or "$ in millions" in hw:
            scale = 1_000_000
            break
        elif "in thousands" in hw or "(thousands)" in hw:
            scale = 1_000
            break
        elif "in billions" in hw or "(billions)" in hw:
            scale = 1_000_000_000
            break

    for start in section_starts:
        window = text[start:start + 3000]
        lines = window.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped or len(stripped) > 200 or len(stripped) < 5:
                continue

            # Skip blocklisted lines
            if _SEG_BLOCKLIST.search(stripped):
                continue

            # Must have dollar amounts
            dollar_matches = list(_DOLLAR_RE.finditer(stripped))
            if not dollar_matches:
                continue

            # Extract the text before the first dollar amount as the segment name
            first_dollar_pos = dollar_matches[0].start()
            label_text = stripped[:first_dollar_pos].strip().rstrip("$").strip()

            # Clean up label — remove trailing dots, dashes, whitespace
            label_text = re.sub(r"[.\-–—:]+$", "", label_text).strip()

            # Skip if label is too short or too long
            if len(label_text) < 3 or len(label_text) > 80:
                continue

            # Skip if label looks like a header/date
            if re.match(r"^\d{4}$", label_text):
                continue
            if re.match(r"^(for|year|three|six|nine|twelve|quarter)", label_text, re.I):
                continue

            val = _parse_dollar_amount(dollar_matches[0])
            if val is None or val <= 0:
                continue

            # Skip values that look like years
            if 1900 <= val <= 2100:
                continue

            if label_text not in results or val > results[label_text]:
                results[label_text] = val

    # ── Strategy 2: Segment-header pattern ──
    # Some companies (banks, diversified) have each segment as its own section
    # with "Net revenues" as a line item. Pattern: {SegmentName} ... Net revenues {$X}
    if len(results) < 2:
        _try_segment_header_parse(text, results)

    if len(results) < 2:
        return []

    # Infer scale if not found
    if scale == 1.0 and total_revenue is not None and total_revenue > 0:
        raw_total = sum(results.values())
        if raw_total > 0:
            ratio = total_revenue / raw_total
            if ratio > 500_000:
                scale = 1_000_000
            elif ratio > 500:
                scale = 1_000

    if scale != 1.0:
        results = {k: v * scale for k, v in results.items()}

    # Filter out geographic regions — these belong in geo revenue, not segments
    _GEO_FILTER = {
        "americas", "europe", "greater china", "japan", "asia pacific",
        "rest of asia pacific", "rest of asia", "united states", "emea",
        "north america", "latin america", "middle east", "africa",
        "international", "rest of world", "asia", "apac", "china",
        "hong kong", "taiwan", "uk", "united kingdom", "canada",
        "brazil", "mexico", "india", "south korea", "korea",
        "australia", "singapore", "germany", "france",
    }
    results = {k: v for k, v in results.items()
               if k.lower() not in _GEO_FILTER}

    # Filter out labels that look like sentences or contain junk characters
    results = {k: v for k, v in results.items()
               if (len(k) < 50
                   and not re.search(r"\b(increased|decreased|during|for|the|was|were|by)\b", k, re.I)
                   and not re.search(r"[|(){}\[\]]", k))}  # No special chars

    # Filter out parent categories that double-count children
    # e.g. "Products" = iPhone + Mac + iPad + Wearables
    if len(results) > 2:
        total_check = sum(results.values())
        to_remove = []
        for k, v in results.items():
            others_total = total_check - v
            # If this segment's value ≈ sum of remaining segments, it's a parent
            if total_check > 0 and abs(v - others_total) / total_check < 0.05:
                to_remove.append(k)
        for k in to_remove:
            results.pop(k, None)

    if len(results) < 2:
        return []

    total = sum(results.values()) or 1
    segments = sorted(
        [{"segment": k, "value": v, "pct": round(v / total * 100, 1)}
         for k, v in results.items()],
        key=lambda x: x["value"],
        reverse=True,
    )

    # Drop segments < 1% of total (noise)
    segments = [s for s in segments if s["pct"] >= 1.0]

    # Limit to top 8 segments
    return segments[:8]


def _try_segment_header_parse(text: str, results: dict[str, float]) -> None:
    """Find segment revenue via section headers.

    Pattern: Each business segment has its own section header, followed by
    an income statement table with a "Net revenues" line. Example:

        Institutional Securities
        Income Statement Information
        ...
        Net revenues  33,080  28,080  23,060
    """
    # Find candidate segment headers — short lines (< 60 chars) that appear
    # before an "Income Statement" or "Financial Data" sub-header
    _header_re = re.compile(
        r"^([A-Z][A-Za-z &/,\-]{3,55})$",
        re.MULTILINE,
    )

    # Non-segment headers to skip
    _skip_headers = re.compile(
        r"(?i)^(?:table\s+of\s+contents|income\s+tax|provision|non.interest|"
        r"other\s+net|compensation|expenses?|revenues?|net\s+income|"
        r"consolidated|new\s+york|overview|forward.looking|risk|"
        r"liquidity|capital|regulation|selected\s+financial|"
        r"management|executive|legal|accounting|auditor|balance\s+sheet|"
        r"cash\s+flow|comprehensive|fair\s+value|goodwill|item\s+\d)",
    )

    # Geographic region names to filter out of revenue segments
    _geo_labels = {
        "americas", "europe", "greater china", "japan", "asia pacific",
        "rest of asia pacific", "rest of asia", "united states", "emea",
        "north america", "latin america", "middle east", "africa",
        "international", "rest of world", "asia", "apac",
    }

    candidates: list[tuple[str, int]] = []  # (label, position)
    for m in _header_re.finditer(text):
        label = m.group(1).strip()
        # Skip non-segment headers
        if _skip_headers.match(label):
            continue
        # Skip geographic regions
        if label.lower() in _geo_labels:
            continue
        # Verify this is followed by something that looks like financial data
        after = text[m.end():m.end() + 2000]
        if re.search(r"(?i)(?:income\s+statement|financial\s+(?:data|information)|statement\s+of\s+(?:income|operations))", after):
            candidates.append((label, m.end()))

    for label, pos in candidates:
        # Look for "Net revenues" line within next 2000 chars
        after = text[pos:pos + 2000]
        rev_match = re.search(
            r"(?i)(?:Net\s+revenues?|Total\s+(?:net\s+)?revenues?)\s+"
            r"(?:\$\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
            after,
        )
        if rev_match:
            val_str = rev_match.group(1).replace(",", "")
            try:
                val = float(val_str)
                if val > 0 and not (1900 <= val <= 2100):
                    if label not in results or val > results[label]:
                        results[label] = val
            except ValueError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  SEC EDGAR links for citation
# ═══════════════════════════════════════════════════════════════════════════

def _build_sec_links(cik: int | str | None, accession: str | None = None) -> dict:
    """Build direct links to SEC EDGAR pages for citation."""
    links: dict[str, str] = {}
    if cik is None:
        return links

    cik_str = str(cik).zfill(10)
    cik_raw = str(int(cik))

    links["company_page"] = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik_str}&type=10-K&dateb=&owner=include&count=10"
    )
    links["filings_page"] = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik_str}&type=&dateb=&owner=include&count=40"
    )
    links["edgar_full_text_search"] = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{cik_raw}%22&dateRange=custom"
    )

    if accession:
        acc_clean = accession.replace("-", "")
        links["filing_index"] = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_clean}/{accession}-index.htm"
        )

    return links


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Disambiguation (Layer 3 — only when ambiguous)
# ═══════════════════════════════════════════════════════════════════════════

def _llm_disambiguate(
    candidates: list[dict],
    company_name: str,
    metric: str,
) -> int | None:
    """Ask Claude to pick the correct value when multiple candidates exist.

    This is Layer 3 of the resolution architecture:
      Layer 1: Deterministic XBRL filtering
      Layer 2: Canonical dictionary matching
      Layer 3: LLM disambiguation (this function)

    Returns the 0-based index of the best candidate, or None if LLM unavailable.
    """
    try:
        config = get_config()
        if not config.anthropic_api_key:
            return None

        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        prompt = (
            f"Given these candidate XBRL values for '{metric}' of {company_name}:\n\n"
            f"{json.dumps(candidates, indent=2, default=str)}\n\n"
            f"Which value most likely represents the consolidated total {metric}? "
            f"Respond with ONLY the index number (0-based) of the correct candidate. "
            f"If none are correct, respond with -1."
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        idx = int(response.content[0].text.strip())
        if 0 <= idx < len(candidates):
            return idx
    except Exception as exc:
        log.debug("Claude disambiguation failed: %s", exc)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Statement DataFrame extraction
# ═══════════════════════════════════════════════════════════════════════════

def _stmt_to_records(stmt) -> list[dict]:
    """Convert a statement object or DataFrame to a list of record dicts.

    Works with both edgartools statement objects and raw DataFrames.
    """
    if stmt is None:
        return []
    try:
        df = stmt.to_dataframe() if hasattr(stmt, "to_dataframe") else None
        if df is None and isinstance(stmt, pd.DataFrame):
            df = stmt
        if df is None or df.empty:
            return []
        return df.where(df.notna(), None).to_dict(orient="records")
    except Exception as exc:
        log.debug("Statement DataFrame conversion failed: %s", exc)
        return []


def _build_statement_from_facts(
    facts_df: pd.DataFrame,
    statement_concepts: list[str],
    *,
    duration_pref: str | None = None,
    is_instant_statement: bool = False,
    target_fp: str | None = None,
    target_fy: int | None = None,
) -> list[dict]:
    """Build a statement table from companyfacts data.

    Groups facts by concept, shows the most recent value for each.
    Uses exact matching to avoid duplicate rows from partial matches.
    Applies quality filters and deduplication.

    duration_pref: "annual" or "quarterly" — filter flow metrics by duration.
    is_instant_statement: True for balance sheet (instant context, no duration filter).
    """
    if facts_df is None or facts_df.empty:
        return []

    records: list[dict] = []
    seen_labels: set[str] = set()

    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(facts_df, ("value", "Value", "val"))
    label_col = _find_column(facts_df, ("label", "Label"))
    end_col = _find_column(facts_df, ("end", "period", "Period"))

    if concept_col is None or value_col is None:
        return []

    # Apply quality filters to get consolidated (non-segment) data
    clean_df = _apply_quality_filters(facts_df)

    for concept_name in statement_concepts:
        # Exact match first (preferred — avoids partial matches)
        exact_mask = (clean_df[concept_col] == concept_name)
        matches = clean_df.loc[exact_mask]

        # Fallback: endswith match for prefixed concepts (us-gaap:ConceptName)
        if matches.empty:
            suffix_mask = (
                clean_df[concept_col].str.endswith(f":{concept_name}", na=False)
                | clean_df[concept_col].str.endswith(f"_{concept_name}", na=False)
            )
            matches = clean_df.loc[suffix_mask]

        if matches.empty:
            continue

        # Sort by date, take most recent
        if end_col and end_col in matches.columns:
            matches = matches.sort_values(end_col, ascending=False)

        # Apply duration filter per-concept (not globally) so that concepts with
        # only YTD values (e.g. cash flows in 10-Qs) are still included.
        # Income items get Q3 standalone; CF items get YTD — both correct.
        if duration_pref and not is_instant_statement and end_col and "start" in matches.columns:
            filtered = _filter_by_duration(
                matches.copy(), end_col, prefer=duration_pref,
                target_fp=target_fp, target_fy=target_fy,
            )
            if not filtered.empty:
                matches = filtered

        row = matches.iloc[0]
        label_raw = row[label_col] if label_col and label_col in matches.columns else None
        label = str(label_raw) if label_raw is not None and pd.notna(label_raw) else concept_name
        val = _safe(row[value_col])
        end_date = str(row[end_col]) if end_col and end_col in matches.columns else ""

        # Prior period value (second most recent)
        prior_val = None
        if len(matches) >= 2:
            prior_val = _safe(matches.iloc[1][value_col])

        # Deduplicate by label (prevents "Intangible Assets" appearing twice)
        if label in seen_labels:
            continue
        seen_labels.add(label)

        if val is not None:
            records.append({
                "label": label,
                "concept": concept_name,
                "value": val,
                "prior_value": prior_val,
                "end_date": end_date,
            })

    return records


# Income statement, balance sheet, and cash flow concept lists
# Used to build statement tables from companyfacts data
# Order matters — this is the display order in the UI
_INCOME_CONCEPTS = [
    # Top-line revenue
    "Revenues", "Revenue", "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetRevenues", "TotalRevenues",
    # Cost of revenue
    "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    # Gross profit
    "GrossProfit",
    # Operating expenses breakdown
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "GeneralAndAdministrativeExpense",
    "SellingAndMarketingExpense",
    "OperatingExpenses", "CostsAndExpenses",
    # Operating income
    "OperatingIncomeLoss",
    # Below-the-line
    "InterestExpense", "InterestExpenseDebt",
    "InterestIncomeExpenseNonoperatingNet",
    "OtherNonoperatingIncomeExpense",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeTaxExpenseBenefit",
    # Net income
    "NetIncomeLoss", "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    # Per-share
    "EarningsPerShareBasic", "EarningsPerShareDiluted",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    # Bank-specific
    "InterestIncomeExpenseNet", "NetInterestIncome",
    "NoninterestIncome", "NoninterestExpense",
    "ProvisionForLoanLeaseAndOtherLosses",
]

_BALANCE_CONCEPTS = [
    # Current assets
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    "ShortTermInvestments", "MarketableSecuritiesCurrent",
    "AccountsReceivableNetCurrent", "AccountsReceivableNet",
    "InventoryNet", "Inventory",
    "PrepaidExpenseAndOtherAssetsCurrent", "PrepaidExpenseCurrent",
    "OtherAssetsCurrent",
    "AssetsCurrent",
    # Non-current assets
    "PropertyPlantAndEquipmentNet",
    "OperatingLeaseRightOfUseAsset",
    "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",
    "LongTermInvestments", "InvestmentsAndAdvances",
    "DeferredIncomeTaxAssetsNet",
    "OtherAssetsNoncurrent",
    "Assets",
    # Current liabilities
    "AccountsPayableCurrent", "AccountsPayable",
    "AccruedLiabilitiesCurrent",
    "ShortTermBorrowings", "DebtCurrent", "CommercialPaper",
    "LongTermDebtCurrent",
    "DeferredRevenueCurrent",
    "OperatingLeaseLiabilityCurrent",
    "OtherLiabilitiesCurrent",
    "LiabilitiesCurrent",
    # Non-current liabilities
    "LongTermDebt", "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "OperatingLeaseLiabilityNoncurrent",
    "DeferredIncomeTaxLiabilitiesNet",
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "OtherLiabilitiesNoncurrent",
    "Liabilities",
    # Equity
    "CommonStockValue", "CommonStocksIncludingAdditionalPaidInCapital",
    "AdditionalPaidInCapital",
    "RetainedEarningsAccumulatedDeficit",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
    "TreasuryStockValue",
    "StockholdersEquity",
    "MinorityInterest",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "LiabilitiesAndStockholdersEquity",
    # Bank-specific
    "LoansAndLeasesReceivableNetReportedAmount",
    "AvailableForSaleSecurities", "HeldToMaturitySecurities",
    "FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase",
    "Deposits",
]

def _build_sector_income_statement(
    industry_class: IndustryClass | None,
    metrics: dict,
    prior_metrics: dict,
) -> list[dict]:
    """Build a sector-templated, ordered income statement from extracted metrics.

    Returns rows even when filer-specific XBRL concept names don't match the
    standard list — this is the path that fixes banks (MS/JPM/GS) where
    the XBRL-driven build comes back empty.
    """
    pm = prior_metrics or {}
    rows: list[dict] = []

    def add(label: str, key: str, *, bold: bool = False, fmt: str | None = None) -> None:
        cur = metrics.get(key)
        prior = pm.get(key)
        if cur is None and prior is None:
            return
        rows.append({
            "label": label,
            "concept": key,
            "value": cur,
            "prior_value": prior,
            "bold": bold,
            "fmt": fmt,
        })

    ic = industry_class

    if ic == IndustryClass.BANK:
        add("Total Revenue (Net Revenues)", "revenue", bold=True)
        add("Interest Expense", "interest_expense")
        add("Provision for Credit Losses", "provision_for_credit_losses")
        add("Operating Income", "operating_income", bold=True)
        add("Other Income / (Expense)", "other_income_expense")
        add("Income Before Tax", "income_before_tax", bold=True)
        add("Income Tax Expense", "income_tax_expense")
        add("Net Income", "net_income", bold=True)
        add("EPS — Basic", "eps_basic", fmt="eps")
        add("EPS — Diluted", "eps_diluted", fmt="eps")
        add("Shares Outstanding", "shares_outstanding", fmt="shares")
    elif ic == IndustryClass.INSURANCE:
        add("Total Revenue (Premiums + Investment Income)", "revenue", bold=True)
        add("Operating Expenses (Losses + Underwriting)", "operating_expenses")
        add("Operating Income", "operating_income", bold=True)
        add("Interest Expense", "interest_expense")
        add("Income Before Tax", "income_before_tax", bold=True)
        add("Income Tax Expense", "income_tax_expense")
        add("Net Income", "net_income", bold=True)
        add("EPS — Diluted", "eps_diluted", fmt="eps")
    elif ic == IndustryClass.REIT:
        add("Total Revenue (Rental + Other)", "revenue", bold=True)
        add("Operating Expenses", "operating_expenses")
        add("Depreciation & Amortization", "depreciation_amortization")
        add("Operating Income", "operating_income", bold=True)
        add("Interest Expense", "interest_expense")
        add("Net Income", "net_income", bold=True)
        # FFO reconciliation section header — analysts expect this for REITs
        rows.append({
            "label": "FFO Reconciliation",
            "concept": "_section",
            "value": None,
            "prior_value": None,
            "is_section": True,
        })
        add("Net Income (FFO start)", "net_income")
        add("(+) Real Estate D&A", "depreciation_amortization")
        ffo = metrics.get("reit_ffo")
        if ffo is not None:
            rows.append({
                "label": "FFO (Funds From Operations)",
                "concept": "reit_ffo",
                "value": ffo,
                "prior_value": pm.get("reit_ffo"),
                "bold": True,
            })
    else:
        # Standard / industrial template — used as fallback when XBRL build is empty
        add("Revenue", "revenue", bold=True)
        add("Cost of Revenue", "cost_of_revenue")
        add("Gross Profit", "gross_profit", bold=True)
        add("Research & Development", "rd_expense")
        add("Selling, General & Administrative", "sga_expense")
        add("Operating Expenses", "operating_expenses")
        add("Operating Income (EBIT)", "operating_income", bold=True)
        add("Interest Expense", "interest_expense")
        add("Other Income / (Expense)", "other_income_expense")
        add("Pre-Tax Income", "income_before_tax", bold=True)
        add("Income Tax Expense", "income_tax_expense")
        add("Net Income", "net_income", bold=True)
        add("EPS — Basic", "eps_basic", fmt="eps")
        add("EPS — Diluted", "eps_diluted", fmt="eps")
        add("Shares Outstanding", "shares_outstanding", fmt="shares")

    return rows


def _insert_gross_margin_row(
    rows: list[dict],
    metrics: dict,
    prior_metrics: dict,
) -> list[dict]:
    """Insert a 'Gross Margin %' row directly after Gross Profit when both
    revenue and gross_profit are present.
    """
    if not rows:
        return rows
    rev = metrics.get("revenue")
    gp = metrics.get("gross_profit")
    if not (rev and gp):
        return rows
    cur_gm = (gp / rev) * 100.0
    prev_rev = (prior_metrics or {}).get("revenue")
    prev_gp = (prior_metrics or {}).get("gross_profit")
    prev_gm = (prev_gp / prev_rev * 100.0) if (prev_rev and prev_gp) else None
    out: list[dict] = []
    inserted = False
    for r in rows:
        out.append(r)
        if not inserted and (r.get("label") or "").lower().startswith("gross profit"):
            out.append({
                "label": "Gross Margin %",
                "concept": "_gross_margin_pct",
                "value": cur_gm,
                "prior_value": prev_gm,
                "fmt": "pct",
            })
            inserted = True
    return out


_CASHFLOW_CONCEPTS = [
    # Operating activities
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByOperatingActivities",
    # Key operating components
    "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
    "ShareBasedCompensation",
    "DeferredIncomeTaxExpenseBenefit",
    "ProvisionForDoubtfulAccounts",
    # Investing activities
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireBusinessesNetOfCashAcquired",
    "PaymentsToAcquireInvestments",
    "ProceedsFromSaleOfInvestments",
    "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
    "NetCashProvidedByUsedInInvestingActivities",
    # Financing activities
    "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
    "PaymentsForRepurchaseOfCommonStock",
    "ProceedsFromIssuanceOfLongTermDebt",
    "RepaymentsOfLongTermDebt",
    "ProceedsFromStockOptionExercises",
    "NetCashProvidedByUsedInFinancingActivities",
    # Net change
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
    "CashAndCashEquivalentsPeriodIncreaseDecrease",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Core extraction
# ═══════════════════════════════════════════════════════════════════════════

# FLOW metrics (income statement + cash flow). In standalone-quarter mode these
# are taken as the YTD-through-this-quarter, then decumulated (currentYTD -
# priorYTD) to the 3-month quarter. Balance-sheet (instant) items and per-share
# values are NOT in this set — they are point-in-time, not cumulative.
_FLOW_DECUM_METRICS = (
    # Income statement
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "operating_expenses", "sga_expense", "ga_expense", "rd_expense", "interest_expense",
    "income_tax_expense", "depreciation_amortization", "provision_for_credit_losses",
    "other_income_expense", "income_before_tax", "share_based_compensation",
    "net_income", "ebitda",
    # Cash flow
    "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
    "capital_expenditures", "dividends_paid", "shares_repurchased",
    "sbc_cash_flow", "acquisitions", "proceeds_from_debt", "repayment_of_debt",
    "free_cash_flow",
)


def _decumulate_cash_flow(
    result: dict, ticker_or_cik: str, filing_meta: dict | None, target_fy: int | None
) -> None:
    """Turn YTD cash-flow metrics into the standalone 3-month quarter by
    subtracting the prior 10-Q's YTD (Q2 = H1 - Q1, Q3 = 9M - H1). Income-
    statement metrics are left alone (already taken at the 3-month tag). Only
    runs for Q2/Q3-type periods; Q1 YTD already equals the standalone quarter.
    """
    try:
        cur_acc = (filing_meta or {}).get("accession_number")
        cm = result.get("metrics") or {}
        if not cur_acc or not cm:
            return
        client = get_sec_client()
        filings = client.get_periodic_filings_smart(ticker_or_cik, form_type="10-Q", limit=8)
        accs = [f.accession_number for f in (filings or [])]
        if cur_acc not in accs:
            return
        idx = accs.index(cur_acc)
        if idx + 1 >= len(accs):
            return  # no prior 10-Q to subtract
        prior = extract_financials(
            ticker_or_cik, accession=accs[idx + 1], standalone_quarter=False
        )
        if not prior or not prior.get("metrics"):
            return
        # Only subtract when the prior quarter is the SAME fiscal year (so we're
        # genuinely peeling one quarter off a YTD, not crossing a year boundary).
        if target_fy is not None and prior.get("fiscal_year") not in (target_fy, None):
            return
        pm = prior["metrics"]
        direct = result.get("metrics_3m") or {}
        for k in _FLOW_DECUM_METRICS:
            if k in direct:  # the filer's own 3-month figure beats subtraction
                cm[k] = direct[k]
                continue
            cv, pv = cm.get(k), pm.get(k)
            if cv is not None and pv is not None:
                cm[k] = cv - pv
            elif cv is not None:
                # Prior YTD never reported this column — a cumulative value
                # must not survive in a standalone-quarter result.
                cm[k] = None
        for k in ("weighted_shares_basic", "weighted_shares_diluted"):
            if k in direct:
                cm[k] = direct[k]
        # Recompute FCF + per-share columns from the decumulated basis
        # (EPS: the 10-Q's own 3-month figure, else YTD-neighbour diff, else
        # None — never the cumulative YTD value).
        result["eps_source"] = _recompute_derived_quarter_metrics(
            cm, prior_eps={k: pm.get(k) for k in _PER_SHARE_METRICS},
            direct=direct, industry=result.get("industry_class"))
        _guard_impossible(cm)
    except Exception as exc:  # never let decumulation break extraction
        log.debug("cash-flow decumulation failed for %s: %s", ticker_or_cik, exc)


def resolve_bank_revenue_drivers(
    facts_df: pd.DataFrame | None,
    *,
    total_net_revenue: float | None = None,
    period_index: int = 0,
    target_fp: str | None = None,
    target_fy: int | None = None,
    duration_pref: str | None = "annual",
) -> dict | None:
    """Decompose bank revenue into canonical income-type drivers.

    Resolves each driver PICK-FIRST from BANK_REVENUE_DRIVERS (first tag the
    filer reports wins — no summing, so interest-variant double-counting can't
    happen). Builds a breakdown that reconciles to total net revenue:

      Tier 1: net_interest_income + noninterest_income  (≈ total; a hard XBRL
              identity for banks that tag both).
      Tier 2: the noninterest sub-drivers (investment banking, trading,
              brokerage, asset/wealth mgmt, card, deposits, fiduciary, mortgage)
              that resolve via a live standard tag, plus a balancing
              "Other fee income" plug = noninterest − Σ(resolved sub-drivers) so
              custom-tagged fees invisible in companyfacts are bucketed, not dropped.

    Returns None when neither NII nor noninterest income can be resolved (e.g. a
    filer misclassified as a bank). Otherwise:
      {
        "income_type": [{driver, label, value, pct_of_total, source_tag}, ...],
        "net_interest_income": float|None,
        "noninterest_income": float|None,
        "total_net_revenue": float|None,
        "reconciles": bool,       # displayed rows sum to total (via residual plug)
        "tier1_identity": bool,   # NII + noninterest ≈ total held (clean 2-tier)
      }
    Values are in the facts' native currency (same scale as metrics["revenue"]);
    the caller applies FX. Business-segment rows are added separately by the caller.
    """
    if facts_df is None or getattr(facts_df, "empty", True):
        return None

    drivers = get_bank_revenue_drivers()

    def _resolve_driver(key: str) -> tuple[float | None, str | None]:
        for entry in drivers[key]:
            val = _lookup_fact(
                facts_df, entry.xbrl_concept, period_index,
                match_mode="exact", duration_pref=duration_pref,
                target_fp=target_fp, target_fy=target_fy,
            )
            if val is not None:
                return val, entry.xbrl_concept
        return None, None

    resolved: dict[str, tuple[float | None, str | None]] = {
        key: _resolve_driver(key) for key in drivers
    }

    nii, nii_src = resolved["net_interest_income"]
    non, non_src = resolved["noninterest_income"]
    if nii is None and non is None:
        return None  # not enough to call this a bank breakdown

    total = total_net_revenue
    if total is None and nii is not None and non is not None:
        total = nii + non

    def _pct(v: float | None) -> float | None:
        if v is None or not total:
            return None
        return round(v / total * 100, 1)

    rows: list[dict] = []
    accounted = 0.0
    if nii is not None:
        accounted += nii
        rows.append({"driver": "net_interest_income", "label": "Net Interest Income",
                     "value": nii, "pct_of_total": _pct(nii), "source_tag": nii_src})

    # Tier-2 sub-drivers (components OF noninterest income).
    sub_found = False
    for key in BANK_NONINTEREST_SUBDRIVERS:
        val, src = resolved[key]
        if val is None:
            continue
        sub_found = True
        accounted += val
        rows.append({"driver": key, "label": drivers[key][0].display_name,
                     "value": val, "pct_of_total": _pct(val), "source_tag": src})

    # Balancing residual so the rows sum to the reported total. Covers the fees
    # we cannot decompose (custom-tagged tier-3 lines invisible in companyfacts,
    # e.g. SCHW's asset-mgmt/trading which aren't even under NoninterestIncome).
    tier1_identity = (
        nii is not None and non is not None and total is not None
        and abs((nii + non) - total) <= abs(total) * 0.01
    )
    if total is not None:
        residual = total - accounted
        if residual > abs(total) * 0.005:
            # "Other Fee Income" when the residual is inside a decomposed
            # noninterest total; "Other Revenue" when noninterest itself is
            # untagged (SCHW) so the remainder spans fees we can't see.
            label = ("Other Fee Income"
                     if (non is not None and sub_found) else "Other Revenue")
            rows.append({"driver": "other_revenue", "label": label,
                         "value": residual, "pct_of_total": _pct(residual),
                         "source_tag": non_src})
            accounted += residual
    elif non is not None and not sub_found:
        # No total available — surface noninterest income as its own row.
        rows.append({"driver": "noninterest_income", "label": "Noninterest Income",
                     "value": non, "pct_of_total": _pct(non), "source_tag": non_src})
        accounted += non

    # Reconciles = the displayed rows sum to the reported total (always true when
    # a total is present, by construction of the residual plug).
    reconciles = bool(
        total is not None and abs(accounted - total) <= abs(total) * 0.01
    )

    return {
        "income_type": rows,
        "net_interest_income": nii,
        "noninterest_income": non,
        "total_net_revenue": total,
        "reconciles": reconciles,          # displayed rows sum to total
        "tier1_identity": tier1_identity,  # NII + noninterest ≈ total held
    }


def _bank_business_segments(ticker: str, form_type: str = "10-K") -> list[dict]:
    """Revenue by reportable business division (CIB/CCB/AWM …) from the filing's
    dimensional data. Uses the authoritative per-filing segment parser (companyfacts
    strips dimensions). Returns [] unless the filing tags the *business* axis."""
    try:
        from sec_mcp.graph.segments import get_dimensional_segments
        dims = get_dimensional_segments(ticker, form_type) or {}
        if (dims.get("source_meta") or {}).get("segmentsAxis") != "business":
            return []
        return [
            {"segment": r.get("segment"), "value": r.get("value"), "pct": r.get("pct")}
            for r in (dims.get("segments") or [])
        ]
    except Exception as exc:
        log.debug("bank business-segment lookup failed for %s: %s", ticker, exc)
        return []


def _scale_bank_drivers(block: dict, fx_rate: float) -> None:
    """Convert a native-currency driver block to USD in place (native × fx_rate)."""
    for k in ("net_interest_income", "noninterest_income", "total_net_revenue"):
        if block.get(k) is not None:
            block[k] = block[k] * fx_rate
    for row in block.get("income_type", []):
        if row.get("value") is not None:
            row["value"] = row["value"] * fx_rate


def extract_financials(
    ticker_or_cik: str,
    *,
    year: int | None = None,
    form_type: str = "10-K",
    accession: str | None = None,
    include_statements: bool = False,
    include_segments: bool = False,
    period_index: int = 0,
    standalone_quarter: bool = False,
    filing: FilingMetadata | None = None,
) -> dict | None:
    """Extract standardized financials for one company.

    This is the main entry point. It:
      1. Looks up the company via SEC submissions API
      2. Finds the target filing (by year/form OR by accession)
      3. Fetches XBRL facts from companyfacts API
      4. Resolves each metric using the 4-pass concept resolver
      5. Computes financial ratios
      6. Validates the results

    Args:
        ticker_or_cik: Ticker symbol (e.g., "AAPL") or CIK number
        year: Fiscal year to target (None = most recent)
        form_type: "10-K" (annual) or "10-Q" (quarterly)
        accession: Specific filing accession number (bypasses year/form search)
        include_statements: Include full income/balance/cashflow tables
        include_segments: Include revenue/geographic segment data
        period_index: 0 = most recent period, 1 = previous, etc.

    Returns a dict with:
      company info, metrics, metrics_sourced, confidence_scores,
      ratios, validation, segments, statements
    """
    client = get_sec_client()

    # ── Get company info ──────────────────────────────────────────────
    try:
        company = client.get_company_info(ticker_or_cik)
    except Exception as exc:
        return {"ticker_or_cik": ticker_or_cik, "error": f"Company not found: {exc}"}

    sic = company.sic_code
    ticker_hint = company.ticker or ticker_or_cik
    industry = detect_industry_class(sic, ticker=ticker_hint)
    cik_val = company.cik

    # Build IR link from company website
    company_website = company.website or ""
    ir_link = ""
    if company_website:
        base = company_website.rstrip("/")
        if not base.startswith("http"):
            base = "https://" + base
        ir_link = base + "/investor-relations"

    result: dict = {
        "ticker_or_cik": ticker_or_cik,
        "company_name": company.name or ticker_or_cik,
        "cik": cik_val,
        "sic_code": sic,
        "industry_class": industry.value,
        "fiscal_year": year or "latest",
        "filing_info": None,   # populated with citation data below
        "sec_links": _build_sec_links(cik_val),
        "company_website": company_website,
        "ir_link": ir_link,
        "metrics": {},
        "metrics_sourced": {},
        "confidence_scores": {},
        "ratios": {},
        "validation": [],
    }

    # ── Find the target filing ────────────────────────────────────────
    filing_meta = None

    if filing is not None:
        filing_meta = filing.model_dump()
    elif accession:
        # Specific filing requested — look up its metadata from submissions
        filing_meta = {
            "accession_number": accession,
            "form_type": form_type,
            "filing_date": "",
        }
        try:
            # The facts-derived periodic index covers the FULL filing history;
            # the submissions `recent` window only spans months for heavy
            # filers, which left older accessions with report_date="" (and
            # garbage period labels downstream).
            found = False
            for ft_search in ["10-K", "10-Q", "20-F", "6-K"]:
                for f in client.get_periodic_filings(
                        ticker_or_cik, form_type=ft_search, limit=80):
                    if f.accession_number == accession:
                        filing_meta["filing_date"] = f.filing_date
                        filing_meta["form_type"] = f.form_type
                        filing_meta["report_date"] = f.report_date
                        found = True
                        break
                if found:
                    break
            if not found:
                # Non-periodic form (8-K etc.) — fall back to submissions
                for ft_search in [form_type, None]:
                    search_filings = client.get_filings(
                        ticker_or_cik, form_type=ft_search, limit=50
                    )
                    for f in search_filings:
                        if f.accession_number == accession:
                            filing_meta["filing_date"] = f.filing_date
                            filing_meta["form_type"] = f.form_type
                            filing_meta["report_date"] = getattr(f, "report_date", "")
                            found = True
                            break
                    if found:
                        break
        except Exception as exc:
            log.warning("Filing metadata lookup failed for %s: %s", ticker_or_cik, exc)

        # Derive fiscal_year + period_type from the filing's report period-end
        # (authoritative) so accession-targeted extractions are labelled the same
        # way as year-targeted ones.
        rd = (filing_meta or {}).get("report_date") or ""
        ftl = (filing_meta or {}).get("form_type") or form_type or ""
        if rd and len(rd) >= 4 and rd[:4].isdigit():
            result["fiscal_year"] = int(rd[:4])
        result["period_type"] = "quarterly" if ftl in ("10-Q", "6-K") else "annual"
    else:
        # Search by year/form_type — _find_filing_accession uses get_filings_smart
        # which auto-tries FPI alternatives (20-F for 10-K, 6-K for 10-Q)
        forms_to_try = [form_type]
        if form_type in ("10-K", "20-F"):
            forms_to_try.append("10-Q")  # fallback to quarterly if no annual
        elif form_type not in ("10-Q", "6-K", "6-K/A"):
            forms_to_try.append("10-K")
        for ft in forms_to_try:
            filing_meta = _find_filing_accession(ticker_or_cik, form_type=ft, year=year)
            if filing_meta:
                break

    if not filing_meta and form_type in ("10-Q", "6-K", "6-K/A"):
        result["error"] = "No quarterly filing available"
        return result

    if filing_meta:
        result["filing_info"] = filing_meta
        acc = filing_meta.get("accession_number")
        if acc and cik_val:
            result["sec_links"].update(_build_sec_links(cik_val, accession=acc))

        # Derive fiscal_year + period_type from the ACTUAL filing selected.
        # Authoritative source = the report period-end (reportDate). The fiscal
        # year is the calendar year of the period END (AAPL FY2024 ends
        # 2024-09-28 -> FY2024), which is correct for any fiscal-year-end month
        # and any filing lag. Only fall back to the filing_date heuristic when a
        # filing somehow lacks a reportDate.
        ft_for_label = filing_meta.get("form_type") or form_type or ""
        report_date = filing_meta.get("report_date") or ""
        if report_date and len(report_date) >= 4 and report_date[:4].isdigit():
            result["fiscal_year"] = int(report_date[:4])
        else:
            fd = filing_meta.get("filing_date") or ""
            try:
                filing_year = int(fd[:4]) if fd else None
                filing_month = int(fd[5:7]) if fd and len(fd) >= 7 else 0
            except (ValueError, TypeError):
                filing_year, filing_month = None, 0
            if filing_year:
                is_annual = ft_for_label in ("10-K", "20-F", "40-F")
                if is_annual and filing_month and filing_month <= 3:
                    result["fiscal_year"] = filing_year - 1
                else:
                    result["fiscal_year"] = filing_year
        result["period_type"] = (
            "quarterly" if ft_for_label in ("10-Q", "6-K") else "annual"
        )

    # ── Check disk cache ─────────────────────────────────────────────
    # Key by accession + standalone flag so 3-month and YTD quarterly results
    # never collide in the cache. Graph-resolver-on results get their own key
    # — values can legitimately differ from legacy and must not cross-pollute.
    cache_acc = (filing_meta or {}).get("accession_number", "")
    if cache_acc:
        # Extraction-output version: v2 adds weighted shares, metrics_3m and
        # basis_check (Nov 2026 audit) — a 7-day-old v1 entry must not mask them.
        cache_acc += "#x2"
    is_interim = (filing_meta or {}).get("form_type") in ("6-K", "6-K/A")
    if cache_acc and is_interim:
        cache_acc += "#interim-v1"
    if cache_acc and standalone_quarter:
        cache_acc = f"{cache_acc}#sq"
    try:
        if cache_acc and (get_config().graph_resolver or "off").lower() == "on":
            cache_acc = f"{cache_acc}#g"
    except Exception:
        pass
    if filing_meta and cache_acc:
        cached = disk_cache.get(ticker_or_cik, cache_acc)
        if cached and "data" in cached:
            data = cached["data"]
            # A slim cached result (no statements/segments) must NOT satisfy a
            # request that asked for them — the cache key doesn't encode these
            # flags, so verify the payload actually covers what was requested.
            wants_more = (
                (include_statements and not data.get("income_statement"))
                or (include_segments and not data.get("segments"))
            )
            if not wants_more:
                log.info("Disk cache hit for %s / %s", ticker_or_cik, cache_acc)
                return data
            log.info("Disk cache hit for %s / %s is slim (statements=%s segments=%s requested) — re-extracting",
                     ticker_or_cik, cache_acc, include_statements, include_segments)

    # ── Get XBRL facts ────────────────────────────────────────────────
    # Use companyfacts API — gets ALL facts for this company, then filter.
    # Fallback chain handles FPI companies (6-K often lacks XBRL):
    #   1. Try exact accession + form_type
    #   2. Try accession only (no form filter)
    #   3. Try form_type only (no accession)
    #   4. Try FPI alternative forms (6-K→20-F, 10-Q→10-K)
    #   5. No filters at all (gets ALL facts, relies on duration filtering)
    accession = filing_meta.get("accession_number") if filing_meta else None
    actual_form = filing_meta.get("form_type", form_type) if filing_meta else form_type

    try:
        facts_df = _get_facts_for_filing(
            ticker_or_cik,
            accession=accession,
            form_type=actual_form,
        )
    except Exception as exc:
        log.warning("XBRL facts extraction failed for %s: %s", ticker_or_cik, exc)
        facts_df = pd.DataFrame()

    if is_interim:
        # Foreign issuers file events, interim YTD results and standalone quarters
        # on the same form. Require this accession's own three-month facts.
        scoped = annotate_periods(facts_df)
        if scoped is not None and not scoped.empty:
            ends = scoped.loc[scoped["_ptype"] != PeriodType.INSTANT.value, "end"]
            end = pd.to_datetime(ends, errors="coerce").max()
            scoped = scoped[(scoped["end"] == end) & scoped["_ptype"].isin(
                [PeriodType.QUARTER.value, PeriodType.INSTANT.value])]
            fiscal = fiscal_quarter(end.date(), company.fiscal_year_end) if pd.notna(end) else None
        else:
            fiscal = None
        if not fiscal or not (scoped["_ptype"] == PeriodType.QUARTER.value).any():
            result["error"] = "This 6-K has no identifiable tagged standalone quarter. See the filing exhibits."
            return result
        result["interimVersion"] = 1
        interim_year, interim_quarter = fiscal
        filing_meta["report_date"] = end.date().isoformat()
        facts_df = scoped.copy()
        facts_df["fp"], facts_df["fy"] = interim_quarter, interim_year

    if accession and (facts_df is None or facts_df.empty):
        result["error"] = "No tagged financials for this accession. See the filing exhibits."
        return result

    # Fallback 3: try FPI alternative form types
    # 6-K filings often lack XBRL — fall back to 20-F data
    # 10-Q might not exist for FPIs — try 10-K/20-F with quarterly duration
    if (facts_df is None or facts_df.empty):
        from sec_mcp.sec_client import get_form_alternatives
        alt_forms = get_form_alternatives(actual_form)
        for alt_ft in alt_forms:
            if alt_ft == actual_form:
                continue
            log.info("Trying FPI alternative form %s for %s", alt_ft, ticker_or_cik)
            try:
                facts_df = _get_facts_for_filing(
                    ticker_or_cik, accession=None, form_type=alt_ft,
                )
                if facts_df is not None and not facts_df.empty:
                    break
            except Exception:
                continue

    # Fallback 4: no filters at all — gets ALL facts across all filings.
    # Duration filtering in _resolve_metric() will pick quarterly vs annual.
    if (facts_df is None or facts_df.empty):
        log.info("Retrying XBRL facts for %s with no filters", ticker_or_cik)
        try:
            facts_df = _get_facts_for_filing(
                ticker_or_cik,
                accession=None,
                form_type=None,
            )
        except Exception:
            facts_df = pd.DataFrame()

    if facts_df is None or facts_df.empty:
        result["error"] = (
            f"No XBRL financials available"
            f"{f' for year {year}' if year else ''} (no {form_type} found)"
        )
        return result

    # ── Single-currency view of the facts ─────────────────────────────
    # Dual-currency filers (TSM tags Revenue in BOTH USD and TWD) must be
    # resolved in ONE currency — the detected reporting currency — and
    # converted exactly once at the end. Without this filter the resolver
    # could pick the USD-tagged fact and the FX step would convert it again
    # (TSM revenue came out 32x low).
    from sec_mcp.core.fx import detect_currency
    reporting_currency = detect_currency(facts_df)
    if reporting_currency != "USD":
        facts_df = _filter_to_currency(facts_df, reporting_currency)

    # ── Extract metrics ───────────────────────────────────────────────
    metrics: dict[str, float | None] = {}
    sourced: dict[str, str | None] = {}
    confidence: dict[str, float] = {}

    # Duration preference: quarterly for 10-Q/6-K, annual for 10-K/20-F.
    # standalone_quarter=True asks for the 3-month standalone quarter (Fineas),
    # otherwise quarterly flow metrics return the SEC-native YTD (dashboard default).
    is_quarterly = is_interim or actual_form.upper() == "10-Q"
    if is_quarterly:
        dur_pref = "standalone_q" if standalone_quarter else "quarterly"
    else:
        dur_pref = "annual"
    pq = is_quarterly  # legacy compat
    result["period_type"] = "quarterly" if is_quarterly else "annual"

    # Extract the authoritative fp/fy from the filing's facts.
    # fp (fiscal period focus) is declared by the filer and is more reliable
    # than duration-day calculations for period selection.
    target_fp: str | None = None
    target_fy: int | None = None
    if "fp" in facts_df.columns and not facts_df.empty:
        fp_counts = facts_df["fp"].dropna().astype(str).value_counts()
        if not fp_counts.empty:
            target_fp = fp_counts.index[0]
            log.debug("Filing fp (authoritative): %s", target_fp)
    if "fy" in facts_df.columns and not facts_df.empty:
        fy_vals = pd.to_numeric(facts_df["fy"], errors="coerce").dropna()
        if not fy_vals.empty:
            target_fy = int(fy_vals.mode().iloc[0])
            log.debug("Filing fy (authoritative): %s", target_fy)
            # DEI fy is the filer-declared fiscal year — authoritative for BOTH
            # annual and quarterly. report_date-year is wrong for quarters whose
            # period end falls in a different calendar year than the fiscal-year
            # end (e.g. AAPL Q1 FY2026 ends Dec 2025). Override the provisional
            # report_date label so fiscal_year + the decumulation FY-match are right.
            result["fiscal_year"] = target_fy

    # Flag YTD vs standalone for 10-Q data
    if is_quarterly and target_fp:
        fp_upper = target_fp.upper()
        if standalone_quarter or is_interim:
            # Flow metrics are decumulated to the 3-month standalone quarter.
            result["is_ytd"] = False
            result["quarter_label"] = f"{fp_upper} (Standalone)"
        else:
            result["is_ytd"] = fp_upper in ("Q2", "Q3", "H1", "M9")
            result["quarter_label"] = {
                "Q1": "Q1 (Standalone)", "Q2": "Q2 (6-month YTD)",
                "Q3": "Q3 (9-month YTD)", "H1": "H1 (6-month)", "M9": "M9 (9-month)"
            }.get(fp_upper, fp_upper)

    # Revenue (industry-aware, with confidence)
    rev_concepts = get_revenue_concepts(industry)
    resolved = _resolve_metric(
        facts_df, rev_concepts, period_index=period_index,
        industry=industry, duration_pref=dur_pref,
        target_fp=target_fp, target_fy=target_fy, is_revenue=True,
        canonical_key="revenue", ticker_or_cik=ticker_or_cik,
        accession=accession,
    )
    metrics["revenue"] = resolved.value
    sourced["revenue"] = resolved.source
    confidence["revenue"] = resolved.confidence

    # All other metrics from the CONCEPT_MAP
    for metric_name, concepts in CONCEPT_MAP.items():
        # operating_income is industry-aware: banks/insurers use pretax income as
        # their operating-income proxy; standard filers use a list with no pretax
        # fallback (so pretax is never mislabeled as operating income).
        if metric_name == "operating_income":
            concepts = get_operating_income_concepts(industry)
        resolved = _resolve_metric(
            facts_df, concepts, period_index=period_index,
            industry=industry, duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy,
            canonical_key=metric_name, ticker_or_cik=ticker_or_cik,
            accession=accession,
        )
        metrics[metric_name] = resolved.value
        sourced[metric_name] = resolved.source
        confidence[metric_name] = resolved.confidence

    # ── Layer 2: Deterministic rollup fallbacks ─────────────────────
    # Gross Profit = Revenue - Cost of Revenue (if GP missing)
    if metrics.get("gross_profit") is None:
        rev_val = metrics.get("revenue")
        cor_val = metrics.get("cost_of_revenue")
        # If cost_of_revenue is missing, try direct COGS lookup
        if cor_val is None:
            for cogs_tag in ("CostOfGoodsSold", "CostOfGoodsAndServicesSold",
                             "CostOfSales", "CostOfServices"):
                cor_val = _lookup_fact(facts_df, cogs_tag, period_index,
                                       match_mode="exact", duration_pref=dur_pref,
                                       target_fp=target_fp, target_fy=target_fy)
                if cor_val is not None:
                    break
        if rev_val is not None and cor_val is not None:
            metrics["gross_profit"] = rev_val - abs(cor_val)
            sourced["gross_profit"] = "Computed: Revenue - Cost of Revenue"
            confidence["gross_profit"] = min(
                confidence.get("revenue", 0), confidence.get("cost_of_revenue", 0.5)
            ) * 0.9
            log.info("Gross Profit computed via rollup: %s", _fmt(metrics["gross_profit"]))

    # Current Assets = sum of current asset components (if total missing)
    if metrics.get("current_assets") is None:
        _ca_components = [
            "CashAndCashEquivalentsAtCarryingValue",
            "ShortTermInvestments",
            "AccountsReceivableNetCurrent",
            "InventoryNet",
            "PrepaidExpenseAndOtherAssetsCurrent",
            "OtherAssetsCurrent",
        ]
        ca_total = 0.0
        ca_found = 0
        for tag in _ca_components:
            v = _lookup_fact(facts_df, tag, period_index, match_mode="exact")
            if v is not None:
                ca_total += v
                ca_found += 1
        if ca_found >= 2:
            metrics["current_assets"] = ca_total
            sourced["current_assets"] = f"Computed: sum of {ca_found} current asset components"
            confidence["current_assets"] = 0.70
            log.info("Current Assets computed via rollup (%d components): %s",
                      ca_found, _fmt(ca_total))

    # Operating Income = Gross Profit - Operating Expenses (if missing).
    # Guard: only valid when operating_expenses is a TRUE operating-expense line
    # (excludes COGS). When it resolved to a total-costs concept
    # ("Costs and Expenses" / "Operating Costs and Expenses", which already
    # include cost of revenue), gross_profit - total_costs double-subtracts COGS
    # and yields garbage (e.g. CVX: 80.8B - 169.3B = -88.5B). Skip rather than
    # emit an obviously-invalid value.
    if metrics.get("operating_income") is None:
        gp_val = metrics.get("gross_profit")
        opex_val = metrics.get("operating_expenses")
        opex_src = (sourced.get("operating_expenses") or "")
        _opex_is_total = any(t in opex_src for t in
                             ("Costs and Expenses", "Operating Costs and Expenses"))
        if gp_val is not None and opex_val is not None and not _opex_is_total:
            metrics["operating_income"] = gp_val - abs(opex_val)
            sourced["operating_income"] = "Computed: Gross Profit - OpEx"
            confidence["operating_income"] = 0.70

    # CapEx: add equipment bought to lease OUT (equipment lessors report two
    # capex lines — own-use PP&E + rental/leased fleet — and every data provider
    # reports headline capex as the sum, e.g. CAT: 2.82B PP&E + 1.47B on-lease =
    # 4.29B). Distinct, non-overlapping concepts, so summing can't double-count.
    capex_src = sourced.get("capital_expenditures") or ""
    if metrics.get("capital_expenditures") is not None and "Capital Expenditures" in capex_src:
        lease_capex = _lookup_fact(
            facts_df, "PaymentsToAcquireEquipmentOnLease", period_index,
            match_mode="exact", duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy)
        if lease_capex is not None and lease_capex > 0:
            metrics["capital_expenditures"] = metrics["capital_expenditures"] + lease_capex
            sourced["capital_expenditures"] = f"{capex_src} + Equipment on Lease"
    if (metrics.get("capital_expenditures") is not None
            and capex_src == "Capital Expenditures (other productive assets)"):
        owned_re = _lookup_fact(
            facts_df, "PaymentsToAcquireAndDevelopRealEstate", period_index,
            match_mode="exact", duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy)
        if owned_re is not None and owned_re > 0:
            metrics["capital_expenditures"] = abs(metrics["capital_expenditures"]) + owned_re
            sourced["capital_expenditures"] = f"{capex_src} + Owned Real Estate"

    # Total liabilities = Assets − Equity − NCI (when no Liabilities subtotal
    # is tagged — utilities often jump straight to "capitalization and
    # liabilities"; serving LiabilitiesAndStockholdersEquity here was the old
    # failure mode)
    if metrics.get("total_liabilities") is None:
        ta_v = metrics.get("total_assets")
        eq_v = metrics.get("stockholders_equity")
        if ta_v is not None and eq_v is not None:
            nci_v = metrics.get("minority_interest") or 0
            metrics["total_liabilities"] = ta_v - eq_v - nci_v
            sourced["total_liabilities"] = "Computed: Assets − Equity − NCI"
            confidence["total_liabilities"] = 0.80

    # Net debt
    ltd = metrics.get("long_term_debt")
    std = metrics.get("short_term_debt")
    cash = metrics.get("cash_and_equivalents")
    if ltd is not None or std is not None:
        total_debt = (ltd or 0) + (std or 0)
        metrics["total_debt"] = total_debt
        if cash is not None:
            metrics["net_debt"] = total_debt - cash
        else:
            metrics["net_debt"] = total_debt

    # Free cash flow (derived: operating cash flow - |capex|)
    ocf = metrics.get("operating_cash_flow")
    capex = metrics.get("capital_expenditures")
    if ocf is not None and capex is not None:
        metrics["free_cash_flow"] = ocf - abs(capex)
        confidence["free_cash_flow"] = min(
            confidence.get("operating_cash_flow", 0),
            confidence.get("capital_expenditures", 0),
        )
    else:
        metrics["free_cash_flow"] = None
        confidence["free_cash_flow"] = 0.0

    # EBITDA = Net Income + Interest Expense + Income Tax + D&A
    # Fallback: Operating Income + D&A (EBIT + D&A approximation)
    oi = metrics.get("operating_income")
    ni_val = metrics.get("net_income")
    int_exp = metrics.get("interest_expense")
    tax_exp = metrics.get("income_tax_expense")

    # D&A lookup: take the LARGEST broad total tagged for the period. Filers
    # can tag several "totals" where one is actually partial — CRM tags
    # DepreciationDepletionAndAmortization (property-only, 1.0B) NEXT TO the
    # true total DepreciationAndAmortization (3.5B); SO's utility total lives
    # under DepreciationAmortizationAndAccretionNet (6.0B) while its only
    # mapped concept was a 32M intangibles line. The genuine total is always a
    # superset of a partial line, so max() picks it.
    da_candidates: list[float] = []
    for tag in ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                "DepreciationAmortizationAndAccretionNet"):
        v = _lookup_fact(facts_df, tag, period_index,
                         match_mode="exact", duration_pref=dur_pref,
                         target_fp=target_fp, target_fy=target_fy)
        if v is not None:
            da_candidates.append(abs(v))
    da_val = max(da_candidates) if da_candidates else None
    if da_val is None:
        # Component fallback: Depreciation + Amortization. Depreciation must be
        # present — an amortization-of-intangibles line on its own is NOT a
        # D&A total (SO's 32M) and must not seed the EBITDA add-back.
        dep = _lookup_fact(facts_df, "Depreciation", period_index,
                           match_mode="exact", duration_pref=dur_pref,
                           target_fp=target_fp, target_fy=target_fy)
        amort = _lookup_fact(facts_df, "AmortizationOfIntangibleAssets",
                             period_index, match_mode="exact",
                             duration_pref=dur_pref,
                             target_fp=target_fp, target_fy=target_fy)
        if dep is not None:
            da_val = abs(dep) + abs(amort or 0)
    # Keep the served D&A metric consistent with the total used for EBITDA —
    # otherwise a partially-resolved D&A (CRM annual = 1.0B vs quarters ≈
    # 0.87B each) makes synthesized Q4 D&A go NEGATIVE (FY − ΣQ1-3).
    _da_metric = metrics.get("depreciation_amortization")
    if da_val is not None and (_da_metric is None or abs(_da_metric) < da_val):
        metrics["depreciation_amortization"] = da_val
        sourced["depreciation_amortization"] = "D&A (largest tagged total)"

    ebitda_val, ebitda_src, ebitda_conf = _build_ebitda(
        oi, ni_val, int_exp, tax_exp, da_val, industry)
    metrics["ebitda"] = ebitda_val
    confidence["ebitda"] = ebitda_conf
    if ebitda_src:
        sourced["ebitda"] = ebitda_src

    # ── Post-extraction sanity checks + auto-correction ────────────
    # If NI > Revenue, we likely picked the wrong revenue tag.
    # Try broader revenue concepts or sum components.
    rev = metrics.get("revenue")
    ni = metrics.get("net_income")
    if rev is not None and ni is not None and rev > 0 and ni > 0 and ni > rev:
        log.warning(
            "NI (%.0f) > Revenue (%.0f) — attempting broader revenue lookup",
            ni, rev,
        )
        _BROAD_REVENUE_TAGS = [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "Revenue",
            "SalesRevenueNet",
            "NetRevenues",
            "TotalRevenues",
            "OperatingRevenue",
            "RealEstateRevenueNet",
            "InterestAndDividendIncomeOperating",
        ]
        best_rev = rev
        best_src = sourced.get("revenue")
        for tag in _BROAD_REVENUE_TAGS:
            v = _lookup_fact(facts_df, tag, period_index,
                             match_mode="exact", duration_pref=dur_pref,
                             target_fp=target_fp, target_fy=target_fy)
            if v is not None and v > best_rev:
                best_rev = v
                best_src = f"{tag} (auto-corrected)"
        if best_rev > rev:
            log.info("Auto-corrected revenue: %.0f → %.0f (%s)", rev, best_rev, best_src)
            metrics["revenue"] = best_rev
            sourced["revenue"] = best_src
            confidence["revenue"] = 0.80

    # ── SG&A from components when no total is tagged ──
    # AMAT reports "Marketing and selling" and "General and administrative"
    # with no SG&A total; the resolver took G&A alone (≈half the real line),
    # and FY − ΣQ1-3 on mixed tags then served Q4 SG&A at 2.4×.
    def _flow(tag: str) -> float | None:
        return _lookup_fact(facts_df, tag, period_index, match_mode="exact",
                            duration_pref=dur_pref, target_fp=target_fp, target_fy=target_fy)
    ga_tagged = _flow("GeneralAndAdministrativeExpense")
    metrics["ga_expense"] = ga_tagged
    sourced["ga_expense"] = "General & Administrative" if ga_tagged is not None else None
    sga_concept = _concept_for_source(CONCEPT_MAP["sga_expense"], sourced.get("sga_expense"))
    if metrics.get("sga_expense") is not None and sga_concept not in (None, "SellingGeneralAndAdministrativeExpense"):
        selling = next((v for v in (_flow(t) for t in (
            "SellingAndMarketingExpense", "SellingExpense", "MarketingAndAdvertisingExpense"))
            if v is not None), None)
        if ga_tagged is not None and selling is not None:
            metrics["sga_expense"] = abs(ga_tagged) + abs(selling)
            sourced["sga_expense"] = "Computed: Selling/Marketing + G&A"
            confidence["sga_expense"] = 0.85

    # ── Intangibles never include goodwill ──
    # MAR's 10-Qs tag only IntangibleAssetsNetIncludingGoodwill: quarterly
    # intangibles served $19.2B against $10.3B in the 10-K, and goodwill was
    # counted twice in any tangible-book calculation.
    if (metrics.get("intangible_assets") is not None
            and _concept_for_source(CONCEPT_MAP["intangible_assets"], sourced.get("intangible_assets"))
            == "IntangibleAssetsNetIncludingGoodwill"):
        gw = metrics.get("goodwill")
        metrics["intangible_assets"] = (metrics["intangible_assets"] - gw) if gw is not None else None
        sourced["intangible_assets"] = ("Computed: Intangibles incl. Goodwill − Goodwill"
                                        if gw is not None else None)

    # ── Operating expenses must tie GP − OpEx = OI ──
    # OperatingExpenses / CostsAndExpenses / OperatingCostsAndExpenses mean
    # different things per filer (AMZN and GOOGL tag total costs including
    # cost of revenue), so "operatingExpenses" mixed two definitions across
    # 23 tickers. Where GP and OI are both known, OpEx is defined by them.
    gp_v, oi_v, opex_v = metrics.get("gross_profit"), metrics.get("operating_income"), metrics.get("operating_expenses")
    if gp_v is not None and oi_v is not None and opex_v is not None:
        scale = max(abs(metrics.get("revenue") or 0), abs(gp_v), 1.0)
        if abs(gp_v - abs(opex_v) - oi_v) > 0.02 * scale:
            metrics["operating_expenses"] = gp_v - oi_v
            sourced["operating_expenses"] = "Computed: Gross Profit − Operating Income"
            confidence["operating_expenses"] = 0.80

    # ── Filer-tagged 3-month values for Q2/Q3 YTD extractions ──
    metrics_3m: dict[str, float] = {}
    if (is_quarterly and not is_interim
            and (target_fp or "").upper() in ("Q2", "Q3", "H1", "M9")):
        try:
            metrics_3m = _capture_three_month(facts_df, metrics, sourced, industry)
        except Exception as exc:  # never let the garnish break the extraction
            log.debug("3-month capture failed for %s: %s", ticker_or_cik, exc)
    basis = None
    if not is_interim:
        try:
            basis = _basis_check(facts_df, sourced, metrics, industry)
        except Exception as exc:
            log.debug("basis check failed for %s: %s", ticker_or_cik, exc)

    # ── Bank revenue-driver breakdown (income-type composition; additive) ──
    # Banks only. Computed from the already-loaded native facts_df (cheap), so
    # metrics["revenue"] here is still the native total. Business-division rows
    # need a separate filing fetch, so they're gated behind include_segments.
    bank_drivers = None
    if industry == IndustryClass.BANK:
        bank_drivers = resolve_bank_revenue_drivers(
            facts_df, total_net_revenue=metrics.get("revenue"),
            period_index=period_index, target_fp=target_fp,
            target_fy=target_fy, duration_pref=dur_pref,
        )

    # ── Currency conversion to USD (currency detected pre-extraction) ──
    from sec_mcp.core.fx import convert_metrics_to_usd
    result["reporting_currency"] = reporting_currency
    if reporting_currency != "USD":
        metrics, fx_rate = convert_metrics_to_usd(metrics, reporting_currency)
        if metrics_3m:
            metrics_3m, _ = convert_metrics_to_usd(metrics_3m, reporting_currency)
        if basis and fx_rate and basis.get("revenue_prior_year") is not None:
            basis["revenue_prior_year"] *= fx_rate
        result["fx_rate"] = fx_rate
        result["fx_note"] = f"Converted from {reporting_currency} to USD at {fx_rate:.4f}" if fx_rate else f"Reported in {reporting_currency} (conversion unavailable)"
        log.info("Foreign filer %s reports in %s, converted to USD", ticker_or_cik, reporting_currency)
        if bank_drivers and fx_rate:
            _scale_bank_drivers(bank_drivers, fx_rate)

    if bank_drivers is not None:
        if include_segments:
            bank_drivers["business_segments"] = _bank_business_segments(
                ticker_or_cik, actual_form)
        result["bank_revenue_drivers"] = bank_drivers

    result["metrics"] = metrics
    result["metrics_sourced"] = sourced
    result["confidence_scores"] = confidence
    result["validation"] = _validate(metrics, industry, confidence)
    if metrics_3m:
        result["metrics_3m"] = metrics_3m
    if basis:
        result["basis_check"] = basis

    # ── Prior-period metrics for YoY comparison ─────────────────────
    # For annual (10-K): compare to prior fiscal year (period_index+1)
    # For quarterly (10-Q): compare to same quarter last year (period_index+4)
    #   and also provide QoQ (period_index+1)
    if is_quarterly:
        yoy_idx = period_index + 4   # Same quarter, prior year (Q3 2024 vs Q3 2023)
        qoq_idx = period_index + 1   # Prior quarter (Q3 vs Q2)
    else:
        yoy_idx = period_index + 1   # Prior fiscal year
        qoq_idx = None

    prior_metrics: dict[str, float | None] = {}
    qoq_metrics: dict[str, float | None] = {}
    try:
        prev_rev = _resolve_metric(
            facts_df, rev_concepts, period_index=yoy_idx,
            industry=industry, duration_pref=dur_pref, is_revenue=True,
        )
        prior_metrics["revenue"] = prev_rev.value
        for metric_name, concepts in CONCEPT_MAP.items():
            if metric_name == "operating_income":
                concepts = get_operating_income_concepts(industry)
            prev = _resolve_metric(
                facts_df, concepts, period_index=yoy_idx,
                industry=industry, duration_pref=dur_pref,
            )
            prior_metrics[metric_name] = prev.value
        prev_ocf = prior_metrics.get("operating_cash_flow")
        prev_capex = prior_metrics.get("capital_expenditures")
        if prev_ocf is not None and prev_capex is not None:
            prior_metrics["free_cash_flow"] = prev_ocf - abs(prev_capex)
        else:
            prior_metrics["free_cash_flow"] = None
    except Exception:
        pass

    # QoQ for quarterly
    if qoq_idx is not None:
        try:
            q_rev = _resolve_metric(
                facts_df, rev_concepts, period_index=qoq_idx,
                industry=industry, duration_pref=dur_pref, is_revenue=True,
            )
            qoq_metrics["revenue"] = q_rev.value
            for metric_name, concepts in CONCEPT_MAP.items():
                if metric_name == "operating_income":
                    concepts = get_operating_income_concepts(industry)
                q_prev = _resolve_metric(
                    facts_df, concepts, period_index=qoq_idx,
                    industry=industry, duration_pref=dur_pref,
                )
                qoq_metrics[metric_name] = q_prev.value
            q_ocf = qoq_metrics.get("operating_cash_flow")
            q_capex = qoq_metrics.get("capital_expenditures")
            if q_ocf is not None and q_capex is not None:
                qoq_metrics["free_cash_flow"] = q_ocf - abs(q_capex)
            else:
                qoq_metrics["free_cash_flow"] = None
        except Exception:
            pass

    # Convert prior/qoq metrics to USD if foreign filer
    if reporting_currency != "USD":
        prior_metrics, _ = convert_metrics_to_usd(prior_metrics, reporting_currency)
        if qoq_metrics:
            qoq_metrics, _ = convert_metrics_to_usd(qoq_metrics, reporting_currency)

    result["prior_metrics"] = prior_metrics
    result["qoq_metrics"] = qoq_metrics
    result["comparison_label"] = "QoQ" if is_quarterly else "YoY"
    result["yoy_label"] = "vs Same Quarter Last Year" if is_quarterly else "vs Prior Year"

    # Ratios computed after prior_metrics (and its USD conversion) so ROE/ROA
    # can use average balances; nothing between here and the metrics block
    # reads result["ratios"].
    result["ratios"] = _compute_ratios(
        metrics, prior=prior_metrics, industry=result.get("industry_class"))

    # ── Segments ──────────────────────────────────────────────────────
    if include_segments:
        result["segments"] = _extract_segments(
            facts_df, target_fp=target_fp, target_fy=target_fy
        )

    # ── Statements ────────────────────────────────────────────────────
    # Build statement tables from companyfacts data
    if include_statements:
        xbrl_income = _build_statement_from_facts(
            facts_df, _INCOME_CONCEPTS, duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy)
        # Sector template uses already-extracted metrics so banks/insurers/REITs
        # always get an ordered, populated statement even when filer-specific
        # XBRL concept names don't match the standard list.
        sector_income = _build_sector_income_statement(
            result.get("industry_class"),
            result.get("metrics") or {},
            result.get("prior_metrics") or {},
        )
        # For financial-template sectors prefer the deterministic sector layout;
        # for industrials prefer the richer XBRL build (more line items) and
        # fall back to the template only when XBRL returns nothing.
        ic = result.get("industry_class")
        if ic in (IndustryClass.BANK, IndustryClass.INSURANCE, IndustryClass.REIT):
            result["income_statement"] = sector_income
        else:
            result["income_statement"] = xbrl_income or sector_income
        # Always insert a Gross Margin % row right after Gross Profit when both
        # revenue and gross_profit are present (one-line readout analysts expect).
        result["income_statement"] = _insert_gross_margin_row(
            result["income_statement"],
            result.get("metrics") or {},
            result.get("prior_metrics") or {},
        )
        result["balance_sheet"] = _build_statement_from_facts(
            facts_df, _BALANCE_CONCEPTS, is_instant_statement=True,
            target_fp=target_fp, target_fy=target_fy)
        result["cash_flow_statement"] = _build_statement_from_facts(
            facts_df, _CASHFLOW_CONCEPTS, duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy)

    # ── Standalone-quarter cash-flow decumulation (YTD-only metrics) ──
    if (standalone_quarter and is_quarterly and not is_interim and result.get("metrics")
            and (target_fp or "").upper() in ("Q2", "Q3", "H1", "M9")):
        _decumulate_cash_flow(result, ticker_or_cik, filing_meta, target_fy)

    # ── Write to disk cache ──────────────────────────────────────────
    if filing_meta and filing_meta.get("accession_number"):
        try:
            summary = generate_local_summary(result)
            disk_cache.put(ticker_or_cik, cache_acc, result, summary)
        except Exception as exc:
            log.warning("Disk cache write failed for %s: %s", ticker_or_cik, exc)

    # ── Audit trail: record resolved metrics (best-effort, never blocks) ──
    try:
        from sec_mcp.facts.ingest import record_observations
        record_observations(result)
    except Exception as exc:
        log.debug("observation write-through skipped: %s", exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Ratios
# ═══════════════════════════════════════════════════════════════════════════

def _div(a: float | None, b: float | None) -> float | None:
    """Safe division — returns None if either operand is None or divisor is zero."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def _compute_ratios(
    m: dict[str, float | None],
    prior: dict[str, float | None] | None = None,
    industry: str | None = None,
) -> dict[str, float | None]:
    """Compute financial ratios — delegates to core.ratios, the canonical engine.

    Kept as a thin wrapper so every existing import/caller keeps working.
    Same fraction units and key names as before, plus roic/roic_basis,
    effective_tax_rate, liabilities_to_equity, roe_basis.
    """
    from sec_mcp.core.ratios import compute_ratios
    return compute_ratios(m, prior=prior, industry=industry)


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

def _build_ebitda(
    oi: float | None,
    ni: float | None,
    int_exp: float | None,
    tax_exp: float | None,
    da: float | None,
    industry: IndustryClass | str | None = None,
) -> tuple[float | None, str | None, float]:
    """EBITDA = Operating Income + D&A (the CapIQ/Bloomberg definition).

    The old primary build, NI + interest + tax + D&A, drags every non-operating
    line into "EBITDA": equity-investment marks, divestiture gains, FX. GOOGL
    Q2'26 served EBITDA $146B on $40.8B of operating income, and EV/EBITDA
    inherited it. The bottom-up build survives only where operating income is
    a pretax proxy (banks, insurers) or is missing. Tax is added back SIGNED —
    a net benefit is negative (QCOM Q2 FY26: −5.1B), abs() would double it.
    """
    if da is None:
        return None, None, 0.0
    financial = str(getattr(industry, "value", industry) or "").lower() in ("bank", "insurance")
    if oi is not None and not financial:
        return oi + abs(da), "Operating Income + D&A", 0.85
    if ni is not None:
        conf = 0.85 if (int_exp is not None and tax_exp is not None) else 0.70
        return (ni + abs(int_exp or 0) + (tax_exp or 0) + abs(da),
                "Net Income + Interest + Tax + D&A", conf)
    if oi is not None:
        return oi + abs(da), "Operating Income + D&A", 0.65
    return None, None, 0.0


def _try_repair_balance_sheet(
    m: dict[str, float | None],
    ta: float,
    tl: float,
    eq: float,
    confidence: dict[str, float],
) -> dict | None:
    """Try to fix A != L + E by finding alternative XBRL concepts.

    Common reasons it doesn't balance:
    1. Equity = StockholdersEquity but should be StockholdersEquityIncludingNCI
    2. Total liabilities = Liabilities but the tag actually includes equity (LiabilitiesAndStockholdersEquity)
    3. Minority interest / NCI not included in equity
    4. Total assets uses a sub-total (current only) instead of full

    Returns dict of corrected metric values, or None if no fix found.
    """
    repairs = {}

    # A directly-tagged total is the filer's own number. When A ≠ L + E it is
    # almost always a line we don't model — temporary/mezzanine equity,
    # redeemable or untagged NCI — not a wrong tag. Overwriting a tagged
    # StockholdersEquity with A − L served BX ~2.5×, SPGI +10-16%, MSTR
    # +11-25% and CRWV +61% (Nov 2026 audit). Only weakly-sourced values
    # (rollups, fuzzy matches) may be "repaired".
    def _weak(key: str) -> bool:
        return (confidence or {}).get(key, 0.0) < 0.90

    # Strategy 1: If L+E > A, equity might include NCI that should be separate
    # Try: Equity = Assets - Liabilities (derive equity from the equation)
    if tl is not None and ta is not None and _weak("stockholders_equity"):
        derived_eq = ta - tl
        if derived_eq > 0 and abs(derived_eq - eq) / max(abs(eq), 1) > 0.05:
            # The derived equity is different — check if it's more reasonable
            # If derived equity is between 10% and 200% of total assets, it's plausible
            eq_ratio = derived_eq / ta if ta > 0 else 0
            if 0.01 < eq_ratio < 0.9:
                repairs["stockholders_equity"] = derived_eq
                log.info("Balance sheet repair: derived equity %s from A-L (was %s)",
                         _fmt(derived_eq), _fmt(eq))
                return repairs

    # Strategy 2: If total_liabilities was actually LiabilitiesAndStockholdersEquity
    # (i.e., L = L+E), then real liabilities = total - equity
    if tl is not None and ta is not None and eq is not None:
        if abs(tl - ta) / max(abs(ta), 1) < 0.02:  # L ≈ A: a mis-tag, not a gap
            # "Liabilities" value is suspiciously close to "Assets" — it's probably L+E
            real_liab = ta - eq
            if real_liab > 0:
                repairs["total_liabilities"] = real_liab
                log.info("Balance sheet repair: liabilities was L+E, corrected to %s",
                         _fmt(real_liab))
                return repairs

    # Strategy 3: Minority interest missing from equity
    # Some companies tag equity as parent-only but assets/liabilities are consolidated
    nci = m.get("minority_interest") or m.get("noncontrolling_interest") or m.get("redeemable_noncontrolling_interest")
    if nci is not None and nci > 0 and _weak("stockholders_equity"):
        eq_with_nci = eq + nci
        new_diff = abs(ta - (tl + eq_with_nci)) / max(abs(ta), 1)
        if new_diff < 0.05:
            repairs["stockholders_equity"] = eq_with_nci
            log.info("Balance sheet repair: added NCI %s to equity", _fmt(nci))
            return repairs

    # Strategy 4: Try computing total_liabilities = total_assets - equity
    if ta is not None and eq is not None and _weak("total_liabilities"):
        derived_tl = ta - eq
        old_diff = abs(ta - (tl + eq)) / max(abs(ta), 1)
        new_diff = abs(ta - (derived_tl + eq)) / max(abs(ta), 1)
        if new_diff < old_diff and derived_tl > 0:
            repairs["total_liabilities"] = derived_tl
            log.info("Balance sheet repair: derived liabilities %s from A-E (was %s)",
                     _fmt(derived_tl), _fmt(tl))
            return repairs

    return None


def _validate(
    m: dict[str, float | None],
    industry: IndustryClass,
    confidence: dict[str, float],
) -> list[dict]:
    """Run validation rules on extracted metrics to flag data quality issues."""
    warnings: list[dict] = []
    rev = m.get("revenue")
    ni = m.get("net_income")
    ta = m.get("total_assets")
    tl = m.get("total_liabilities")
    eq = m.get("stockholders_equity")

    # Rule 1: Revenue must exceed net income for profitable companies
    if rev is not None and ni is not None and rev > 0 and ni > 0:
        if ni > rev:
            warnings.append({
                "rule": "revenue_gt_net_income",
                "severity": "error",
                "message": (
                    f"Net income ({_fmt(ni)}) exceeds revenue ({_fmt(rev)}). "
                    f"Revenue concept may be wrong — check for segment vs total."
                ),
            })

    # Rule 2: Accounting equation: A = L + E_parent + NCI (within 5%)
    # stockholders_equity is deliberately PARENT-ONLY; consolidated balance
    # sheets only tie once noncontrolling interests are added back. Without
    # the NCI term the "repair" below overwrites a correct parent equity with
    # Assets − Liabilities (parent + NCI + temp equity) — for NEE FY2024
    # that inflated equity from $50.1B to $60.9B.
    if ta is not None and tl is not None and eq is not None:
        nci = m.get("minority_interest") or 0
        expected = tl + eq + nci
        diff_pct = abs(ta - expected) / ta if ta != 0 else 0
        if diff_pct > 0.05:
            # ── Auto-repair: try alternative concept resolutions ──
            repaired = _try_repair_balance_sheet(m, ta, tl, eq, confidence)
            if repaired:
                # Update metrics with repaired values
                for k, v in repaired.items():
                    m[k] = v
                ta = m.get("total_assets")
                tl = m.get("total_liabilities")
                eq = m.get("stockholders_equity")
                new_expected = (tl or 0) + (eq or 0) + (m.get("minority_interest") or 0)
                new_diff = abs((ta or 0) - new_expected) / (ta or 1)
                if new_diff <= 0.05:
                    warnings.append({
                        "rule": "accounting_equation_repaired",
                        "severity": "info",
                        "message": (
                            f"Accounting equation repaired. "
                            f"Assets ({_fmt(ta)}) ≈ Liabilities ({_fmt(tl)}) + "
                            f"Equity ({_fmt(eq)}) = {_fmt(new_expected)}. "
                            f"Diff: {new_diff:.1%}"
                        ),
                    })
                else:
                    warnings.append({
                        "rule": "accounting_equation",
                        "severity": "warning",
                        "message": (
                            f"Assets ({_fmt(ta)}) != Liabilities ({_fmt(tl)}) + "
                            f"Equity ({_fmt(eq)}) = {_fmt(new_expected)}. "
                            f"Difference: {new_diff:.1%} (auto-repair attempted)"
                        ),
                    })
            else:
                warnings.append({
                    "rule": "accounting_equation",
                    "severity": "warning",
                    "message": (
                        f"Assets ({_fmt(ta)}) != Liabilities ({_fmt(tl)}) + "
                        f"Equity ({_fmt(eq)}) = {_fmt(expected)}. "
                        f"Difference: {diff_pct:.1%}"
                    ),
                })

    # Rule 3: Revenue missing
    if rev is None:
        warnings.append({
            "rule": "revenue_missing",
            "severity": "warning",
            "message": "Could not resolve total revenue from XBRL concepts.",
        })

    # Rule 4: Net income missing
    if ni is None:
        warnings.append({
            "rule": "net_income_missing",
            "severity": "warning",
            "message": "Could not resolve net income from XBRL concepts.",
        })

    # Rule 5: Bank-specific — revenue too low (segment capture)
    if industry == IndustryClass.BANK and rev is not None and ni is not None:
        if rev > 0 and ni > 0 and rev < ni * 0.8:
            warnings.append({
                "rule": "bank_revenue_segment",
                "severity": "error",
                "message": (
                    f"Revenue ({_fmt(rev)}) is below net income ({_fmt(ni)}). "
                    f"Likely only a segment was captured. "
                    f"Bank revenue should include NII + non-interest income + trading."
                ),
            })

    # Rule 6: Gross margin range (standard companies)
    if industry == IndustryClass.STANDARD:
        gp = m.get("gross_profit")
        if gp is not None and rev is not None and rev > 0:
            gm = gp / rev
            if gm < 0 or gm > 1.0:
                warnings.append({
                    "rule": "gross_margin_range",
                    "severity": "warning",
                    "message": f"Gross margin {gm:.1%} outside 0-100% range.",
                })

    # Rule 7: Low confidence on critical metrics
    for metric_name in ("revenue", "net_income"):
        conf = confidence.get(metric_name, 0.0)
        if conf > 0 and conf < 0.50:
            warnings.append({
                "rule": f"low_confidence_{metric_name}",
                "severity": "warning",
                "message": (
                    f"{metric_name.replace('_', ' ').title()} resolved with low "
                    f"confidence ({conf:.0%}). Value may be unreliable."
                ),
            })

    # Rule 8: Net margin sanity (> 100% or < -100%)
    if rev is not None and ni is not None and rev > 0:
        margin = ni / rev
        if abs(margin) > 1.0:
            warnings.append({
                "rule": "net_margin_extreme",
                "severity": "warning",
                "message": (
                    f"Net margin is {margin:.0%} — outside normal range. "
                    f"Verify revenue ({_fmt(rev)}) and net income ({_fmt(ni)}) tags."
                ),
            })

    # Rule 9 (FAC identity): GrossProfit = Revenue − CostOfRevenue
    gp = m.get("gross_profit")
    cor = m.get("cost_of_revenue")
    if None not in (gp, rev, cor) and rev:
        expected_gp = rev - abs(cor)
        if abs(gp - expected_gp) / max(abs(rev), 1) > 0.02:
            warnings.append({
                "rule": "gross_profit_identity",
                "severity": "warning",
                "message": (
                    f"GrossProfit ({_fmt(gp)}) != Revenue − CostOfRevenue "
                    f"({_fmt(expected_gp)}) — one of the three tags is off."
                ),
            })

    # Rule 10 (FAC identity): NetIncome = IncomeBeforeTax − IncomeTaxExpense
    ibt = m.get("income_before_tax")
    tax = m.get("income_tax_expense")
    if None not in (ni, ibt, tax) and ibt:
        expected_ni = ibt - tax
        if abs(ni - expected_ni) / max(abs(ibt), 1) > 0.05:
            warnings.append({
                "rule": "pretax_tax_tie",
                "severity": "info",
                "message": (
                    f"NetIncome ({_fmt(ni)}) != IncomeBeforeTax − Tax "
                    f"({_fmt(expected_ni)}) — NCI/discontinued ops or a wrong tag."
                ),
            })

    return warnings


# ═══════════════════════════════════════════════════════════════════════════
#  Local narrative generator (no Claude needed)
# ═══════════════════════════════════════════════════════════════════════════

def generate_local_summary(data: dict) -> str:
    """Generate a readable natural-language summary from extraction results.

    Works without any external API — pure template-based generation.
    Covers revenue, profitability, balance sheet, cash flow, segments.
    """
    name = data.get("company_name", "Unknown")
    ticker = data.get("ticker_or_cik", "?")
    m = data.get("metrics", {})
    r = data.get("ratios", {})
    v = data.get("validation", [])
    fi = data.get("filing_info")
    links = data.get("sec_links", {})
    segs = data.get("segments", {})

    parts: list[str] = []

    # ── Header ────────────────────────────────────────────────────────
    fy = data.get("fiscal_year", "latest")
    parts.append(f"## {name} ({ticker}) — FY {fy}")
    parts.append("")

    # ── Revenue & profitability ───────────────────────────────────────
    rev = m.get("revenue")
    ni = m.get("net_income")
    gp = m.get("gross_profit")
    oi = m.get("operating_income")

    if rev is not None:
        line = f"**Revenue** of {_fmt(rev)}"
        if ni is not None:
            margin = r.get("net_margin")
            margin_str = f" ({margin:.1%} net margin)" if margin is not None else ""
            line += f" with **net income** of {_fmt(ni)}{margin_str}"
        line += "."
        parts.append(line)

        if gp is not None:
            gm = r.get("gross_margin")
            gm_str = f" ({gm:.1%})" if gm is not None else ""
            parts.append(f"Gross profit of {_fmt(gp)}{gm_str}.")
        if oi is not None:
            om = r.get("operating_margin")
            om_str = f" ({om:.1%} operating margin)" if om is not None else ""
            parts.append(f"Operating income of {_fmt(oi)}{om_str}.")
    elif ni is not None:
        parts.append(f"**Net income** of {_fmt(ni)}. Revenue data not available from XBRL.")
    else:
        parts.append("Limited financial data available from XBRL for this filing.")

    # ── Expense breakdown ─────────────────────────────────────────────
    cor = m.get("cost_of_revenue")
    sga = m.get("sga_expense")
    rd = m.get("rd_expense")
    expense_parts = []
    if cor is not None:
        expense_parts.append(f"COGS {_fmt(cor)}")
    if sga is not None:
        expense_parts.append(f"SG&A {_fmt(sga)}")
    if rd is not None:
        expense_parts.append(f"R&D {_fmt(rd)}")
    if expense_parts:
        parts.append(f"Expense breakdown: {', '.join(expense_parts)}.")

    parts.append("")

    # ── Balance sheet ─────────────────────────────────────────────────
    ta = m.get("total_assets")
    eq = m.get("stockholders_equity")
    cash = m.get("cash_and_equivalents")
    cr = r.get("current_ratio")

    if ta is not None:
        line = f"**Balance sheet**: {_fmt(ta)} total assets"
        if eq is not None:
            line += f", {_fmt(eq)} stockholders' equity"
        line += "."
        parts.append(line)
    if cash is not None:
        parts.append(f"Cash and equivalents of {_fmt(cash)}.")
    if cr is not None:
        health = "healthy" if cr > 1.2 else "tight" if cr > 0.8 else "concerning"
        parts.append(f"Current ratio of {cr:.2f}x ({health} liquidity).")

    # ── Debt ──────────────────────────────────────────────────────────
    ltd = m.get("long_term_debt")
    dte = r.get("debt_to_equity")
    if ltd is not None:
        line = f"Long-term debt of {_fmt(ltd)}"
        if dte is not None:
            line += f" (D/E ratio: {dte:.2f}x)"
        line += "."
        parts.append(line)

    parts.append("")

    # ── Cash flow ─────────────────────────────────────────────────────
    ocf = m.get("operating_cash_flow")
    capex = m.get("capital_expenditures")
    fcf = m.get("free_cash_flow")

    if ocf is not None:
        line = f"**Cash flow**: {_fmt(ocf)} from operations"
        if capex is not None:
            line += f", {_fmt(abs(capex))} in capex"
        if fcf is not None:
            line += f", yielding **{_fmt(fcf)} free cash flow**"
        line += "."
        parts.append(line)

    parts.append("")

    # ── Segments ──────────────────────────────────────────────────────
    rev_segs = (segs or {}).get("revenue_segments", [])
    geo_segs = (segs or {}).get("geographic_segments", [])
    if rev_segs:
        parts.append("**Revenue segments**: " + ", ".join(
            f"{s['segment']} ({_fmt(s['value'])})" for s in rev_segs[:6]
        ) + ".")
    if geo_segs:
        parts.append("**Geographic breakdown**: " + ", ".join(
            f"{s['segment']} ({_fmt(s['value'])})" for s in geo_segs[:6]
        ) + ".")

    if rev_segs or geo_segs:
        parts.append("")

    # ── Validation warnings ───────────────────────────────────────────
    errors = [w for w in v if w.get("severity") == "error"]
    warns = [w for w in v if w.get("severity") == "warning"]
    if errors:
        parts.append("**Data quality issues**:")
        for w in errors:
            parts.append(f"- {w['message']}")
    if warns:
        if not errors:
            parts.append("**Notes**:")
        for w in warns:
            parts.append(f"- {w['message']}")

    if errors or warns:
        parts.append("")

    # ── Citation ──────────────────────────────────────────────────────
    parts.append("---")
    if fi:
        acc = fi.get("accession_number", "")
        fdate = fi.get("filing_date", "")
        form = fi.get("form_type", "10-K")
        parts.append(f"*Source: SEC {form} filed {fdate} (Accession: {acc})*")
    filing_link = links.get("filing_index") or links.get("company_page")
    if filing_link:
        parts.append(f"*[View on EDGAR]({filing_link})*")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  FMP-compatible statement shaping (drop-in for downstream FMP consumers)
# ═══════════════════════════════════════════════════════════════════════════

def _period_label(result: dict) -> tuple[str, str, str]:
    """(date, fiscalYear, period) for an FMP-shaped row, from SEC fields."""
    fi = result.get("filing_info") or {}
    date = fi.get("report_date") or fi.get("filing_date") or ""
    fy = result.get("fiscal_year")
    fy_s = str(fy) if isinstance(fy, int) else ""
    if result.get("period_type") == "quarterly":
        ql = (result.get("quarter_label") or "").split()
        period = ql[0] if ql else "Q"
    else:
        period = "FY"
    return date, fy_s, period


def sec_to_fmp_statements(result: dict) -> dict:
    """Convert one extract_financials() result into FMP-shaped income /
    balance / cash-flow row dicts. Field names mirror Financial Modeling Prep's
    `stable` API so existing FMP consumers (e.g. Fineas) parse it unchanged.
    Outflows (capex, dividends, buybacks) use FMP's negative-sign convention.
    """
    m = result.get("metrics") or {}
    fi = result.get("filing_info") or {}
    date, fy, period = _period_label(result)
    sym = result.get("ticker_or_cik", "")
    filing_ccy = result.get("reporting_currency") or "USD"
    # reportedCurrency must describe the VALUES: when fx conversion ran the
    # metrics are USD even though the filing was EUR/GBP/… — labeling them
    # with the filing currency overstated ASML by the full EURUSD rate.
    ccy = "USD" if result.get("fx_rate") else filing_ccy
    cik = result.get("cik")
    fd = fi.get("filing_date", "")
    base = {"date": date, "symbol": sym, "reportedCurrency": ccy, "cik": cik,
            "filingDate": fd, "acceptedDate": fd, "fiscalYear": fy, "period": period,
            # Additive provenance block — existing FMP consumers ignore it.
            "_meta": {
                "quality": result.get("quality"),
                "isYtd": bool(result.get("is_ytd")),
                "accession": fi.get("accession_number"),
                # Source filing form (10-K/10-Q/20-F/6-K) — additive, lets the
                # frontend label each period by the filing it came from.
                "formType": fi.get("form_type"),
                "confidence": result.get("confidence_scores") or {},
                "sources": result.get("metrics_sourced") or {},
                "validation": result.get("validation") or [],
                # FPI provenance: the filing's native currency + the rate used
                # to express the values in USD (null for US filers).
                "filingCurrency": filing_ccy,
                "fxRate": result.get("fx_rate"),
                # Additive: where the per-share columns came from
                # ("reported" = the filing's own figure, "derived" = arithmetic)
                # and the split factor applied to bring an older row onto the
                # newest share basis (1 = none).
                "epsSource": result.get("eps_source") or "reported",
                "splitFactor": result.get("split_factor") or 1,
            }}

    def neg(v):
        return -abs(v) if v is not None else None

    income = {**base,
        "revenue": m.get("revenue"), "costOfRevenue": m.get("cost_of_revenue"),
        "grossProfit": m.get("gross_profit"),
        "researchAndDevelopmentExpenses": m.get("rd_expense"),
        # G&A only when the filer reports it as its own line — it used to be
        # a copy of SG&A for every filer (132 of 158 audited tickers).
        "generalAndAdministrativeExpenses": m.get("ga_expense"),
        "sellingGeneralAndAdministrativeExpenses": m.get("sga_expense"),
        "operatingExpenses": m.get("operating_expenses"),
        "depreciationAndAmortization": m.get("depreciation_amortization"),
        "ebitda": m.get("ebitda"), "ebit": m.get("operating_income"),
        "operatingIncome": m.get("operating_income"),
        "totalOtherIncomeExpensesNet": m.get("other_income_expense"),
        "interestExpense": m.get("interest_expense"),
        "incomeBeforeTax": m.get("income_before_tax"),
        "incomeTaxExpense": m.get("income_tax_expense"),
        "netIncome": m.get("net_income"), "bottomLineNetIncome": m.get("net_income"),
        "eps": m.get("eps_basic"), "epsDiluted": m.get("eps_diluted"),
        # Weighted averages are the EPS denominators. Period-end shares only
        # stand in when the filer tags no weighted count; diluted falls back
        # to basic only when no diluted count exists (net-loss filers often
        # tag one BasicAndDiluted figure, where equal IS correct).
        "weightedAverageShsOut": m.get("weighted_shares_basic") or m.get("shares_outstanding"),
        "weightedAverageShsOutDil": (m.get("weighted_shares_diluted")
                                     or m.get("weighted_shares_basic")
                                     or m.get("shares_outstanding")),
        # Additive: period-end common shares (balance-sheet/cover count).
        "sharesOutstanding": m.get("shares_outstanding")}

    balance = {**base,
        "cashAndCashEquivalents": m.get("cash_and_equivalents"),
        "netReceivables": m.get("accounts_receivable"),
        "accountsReceivables": m.get("accounts_receivable"),
        "inventory": m.get("inventory"),
        "totalCurrentAssets": m.get("current_assets"),
        "propertyPlantEquipmentNet": m.get("property_plant_equipment"),
        "goodwill": m.get("goodwill"), "intangibleAssets": m.get("intangible_assets"),
        "totalAssets": m.get("total_assets"),
        "accountPayables": m.get("accounts_payable"),
        "deferredRevenue": m.get("deferred_revenue"),
        "shortTermDebt": m.get("short_term_debt"),
        "totalCurrentLiabilities": m.get("current_liabilities"),
        "longTermDebt": m.get("long_term_debt"),
        "totalLiabilities": m.get("total_liabilities"),
        "totalDebt": m.get("total_debt"), "netDebt": m.get("net_debt"),
        "retainedEarnings": m.get("retained_earnings"),
        "totalStockholdersEquity": m.get("stockholders_equity"),
        "totalEquity": m.get("total_equity") or m.get("stockholders_equity")}

    cashflow = {**base,
        "netIncome": m.get("net_income"),
        "depreciationAndAmortization": m.get("depreciation_amortization"),
        "stockBasedCompensation": m.get("share_based_compensation") or m.get("sbc_cash_flow"),
        "operatingCashFlow": m.get("operating_cash_flow"),
        "netCashProvidedByOperatingActivities": m.get("operating_cash_flow"),
        "capitalExpenditure": neg(m.get("capital_expenditures")),
        "freeCashFlow": m.get("free_cash_flow"),
        "netCashProvidedByInvestingActivities": m.get("investing_cash_flow"),
        "netCashProvidedByFinancingActivities": m.get("financing_cash_flow"),
        "commonDividendsPaid": neg(m.get("dividends_paid")),
        "netDividendsPaid": neg(m.get("dividends_paid")),
        "commonStockRepurchased": neg(m.get("shares_repurchased"))}

    return {"income": income, "balance": balance, "cashflow": cashflow}


_FP_RANK = {"Q1": 1, "Q2": 2, "H1": 2, "Q3": 3, "M9": 3, "Q4": 4, "FY": 4}


def _quarter_rank(result: dict) -> int | None:
    fp = (result.get("quarter_label") or "").split()
    return _FP_RANK.get(fp[0].upper()) if fp else None


# Per-share flow columns. Cumulative like the flow metrics, but they can't be
# decumulated by plain subtraction of the underlying dollars — they must be
# read from the filing's 3-month context (10-Q) or differenced as per-share
# figures (YTD − prior YTD, FY − ΣQ1-3).
_PER_SHARE_METRICS = ("eps_basic", "eps_diluted")
_SHARE_COUNT_METRICS = ("weighted_shares_basic", "weighted_shares_diluted", "shares_outstanding")

# Lines that cannot be negative in a real quarter. A negative value here is
# the fingerprint of subtracting figures reported on different bases.
_NONNEG_QUARTER_METRICS = ("revenue", "cost_of_revenue", "sga_expense", "ga_expense",
                           "rd_expense", "depreciation_amortization")
_IS_FLOW_KEYS = frozenset((
    "revenue", "cost_of_revenue", "gross_profit", "operating_income", "operating_expenses",
    "sga_expense", "ga_expense", "rd_expense", "interest_expense", "income_tax_expense",
    "depreciation_amortization", "provision_for_credit_losses", "other_income_expense",
    "income_before_tax", "net_income", "ebitda",
))

# Common split ratios (forward). Reverse splits use the reciprocals.
_SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)


def _report_date(result: dict) -> pd.Timestamp:
    return pd.to_datetime((result.get("filing_info") or {}).get("report_date") or "",
                          errors="coerce")


def _days(newer: pd.Timestamp, older: pd.Timestamp) -> int | None:
    if pd.isna(newer) or pd.isna(older):
        return None
    return int((newer - older).days)


def _guard_impossible(cm: dict) -> list[str]:
    """Null quarter lines that cannot be negative; return what was nulled.

    If revenue itself was impossible, every income-statement flow in the row
    shares the broken basis — null them all rather than serve half a quarter.
    """
    bad = [k for k in _NONNEG_QUARTER_METRICS if (cm.get(k) is not None and cm[k] < 0)]
    for k in bad:
        cm[k] = None
    if "revenue" in bad:
        for k in _IS_FLOW_KEYS | set(_PER_SHARE_METRICS):
            cm[k] = None
    return bad


def _note(result: dict, rule: str, message: str) -> None:
    result.setdefault("validation", [])
    result["validation"] = list(result["validation"]) + [
        {"rule": rule, "severity": "warning", "message": message}]


def _split_ratio(newer: float | None, older: float | None) -> float | None:
    """Forward/reverse split factor implied by two adjacent share counts."""
    if not newer or not older or newer <= 0 or older <= 0:
        return None
    r = newer / older
    if r >= 1.8:
        k = min(_SPLIT_RATIOS, key=lambda s: abs(r / s - 1))
        return float(k) if abs(r / k - 1) <= 0.06 else None
    if r <= 0.55:
        k = min(_SPLIT_RATIOS, key=lambda s: abs((1 / r) / s - 1))
        return 1.0 / k if abs((1 / r) / k - 1) <= 0.06 else None
    return None


def _share_basis(result: dict) -> float | None:
    m = result.get("metrics") or {}
    return m.get("weighted_shares_diluted") or m.get("weighted_shares_basic") or m.get("shares_outstanding")


def _scale_per_share(result: dict, factor: float) -> None:
    """Bring a row onto a newer share basis: per-share ÷ f, share counts × f."""
    for block in ("metrics", "metrics_3m"):
        m = result.get(block) or {}
        for k in _PER_SHARE_METRICS:
            if m.get(k) is not None:
                m[k] = m[k] / factor
        for k in _SHARE_COUNT_METRICS:
            if m.get(k) is not None:
                m[k] = m[k] * factor
    result["split_factor"] = (result.get("split_factor") or 1) * factor


def _split_adjust(results: list[dict]) -> list[dict]:
    """Restate older rows' per-share figures onto the newest share basis.

    Filings report per-share history as originally reported, so a series that
    spans a split jumps by the split ratio (NVDA FY24 EPS 11.93 → FY25 2.94;
    ORLY 15:1, NFLX 10:1, CTAS 4:1 …), and any Q4 = FY − ΣQ1-3 that straddles
    one is garbage (CTAS Q4 FY24 4.09 vs ~1.00). A split shows up as adjacent
    share counts differing by a clean ratio (2, 3, 4, 10, 15 …, ±6%); ordinary
    issuance and buybacks never do. Detected from the rows themselves, so no
    external corporate-actions feed is needed.
    """
    dated = sorted((r for r in results if pd.notna(_report_date(r)) and _share_basis(r)),
                   key=_report_date, reverse=True)
    raw = [_share_basis(r) for r in dated]  # unadjusted — read before any scaling
    factor = 1.0
    for i in range(1, len(dated)):
        k = _split_ratio(raw[i - 1], raw[i])
        if k:
            factor *= k
        if factor != 1.0:
            _scale_per_share(dated[i], factor)
    return results


def _align_to(result: dict, reference: dict | None) -> None:
    """Put a stand-alone row (the 10-K used for Q4) on the reference row's basis."""
    if reference is None:
        return
    k = _split_ratio(_share_basis(reference), _share_basis(result))
    if k:
        _scale_per_share(result, k)


def _recompute_derived_quarter_metrics(cm: dict, prior_eps: dict | None = None,
                                       direct: dict | None = None,
                                       industry: Any = None) -> str:
    """Re-derive FCF, per-share columns and EBITDA after a quarter changes basis.

    EPS priority (a decumulated/synthesized row must NEVER keep the
    cumulative YTD/FY per-share value it was extracted with):
      1. the filing's own 3-month EPS (`direct`, 10-Q Q2/Q3)       → "reported"
      2. cumulative EPS minus the YTD-neighbour / ΣQ1-3 EPS          → "derived"
      3. None — better an honest gap than a YTD/FY value
    Net income ÷ period-end shares is NOT a fallback: it ignores weighting,
    dilution and preferred dividends (GS/BAC quarterly EPS were 7-13% high).
    Returns the EPS provenance.
    """
    ocf, capex = cm.get("operating_cash_flow"), cm.get("capital_expenditures")
    if ocf is not None and capex is not None:
        cm["free_cash_flow"] = ocf - abs(capex)
    ni = cm.get("net_income")
    source = "reported"
    for k in _PER_SHARE_METRICS:
        if direct and direct.get(k) is not None:
            cm[k] = direct[k]
            continue
        cur, prior = cm.get(k), (prior_eps or {}).get(k)
        if cur is not None and prior is not None:
            v = round(cur - prior, 2)
            # Sign cross-check against the row's own NI — a residual split or
            # basis mismatch shows up as an EPS of the wrong sign.
            cm[k] = None if (ni is not None and ni != 0 and v != 0 and (v < 0) != (ni < 0)) else v
            if cm[k] is not None:
                source = "derived"
        else:
            cm[k] = None
    # EBITDA: rebuild from the row's OWN (decumulated/synthesized) components —
    # never by subtracting cumulative EBITDAs of possibly different bases
    # (JPM Q2'24: 1.1B vs ~25B). Same definition as extraction time.
    da_q = cm.get("depreciation_amortization")
    rebuilt, _, _ = _build_ebitda(cm.get("operating_income"), ni, cm.get("interest_expense"),
                                  cm.get("income_tax_expense"), da_q, industry)
    cm["ebitda"] = rebuilt
    return source


def _recast(result: dict, results: list[dict]) -> bool | None:
    """Did this filing restate its prior-year comparative? None = can't tell.

    Compares the filing's own prior-year revenue comparative with how the
    prior-year filing of the same rank originally reported that period (its
    pre-decumulation snapshot in `_ytd_metrics`).
    """
    py = (result.get("basis_check") or {}).get("revenue_prior_year")
    d, rk = _report_date(result), _quarter_rank(result)
    if py is None or pd.isna(d) or not rk:
        return None
    for other in results:
        if other is result or _quarter_rank(other) != rk:
            continue
        gap = _days(d, _report_date(other))
        if gap is not None and 350 <= gap <= 380:
            orig = (other.get("_ytd_metrics") or other.get("metrics") or {}).get("revenue")
            if orig:
                return abs(py - orig) / abs(orig) > 0.005
    return None


def _basis_break(a: dict, b: dict, results: list[dict]) -> bool:
    """True when exactly one of two filings recast prior periods — i.e. a
    spin-off/discontinued-ops restatement landed between them."""
    fa, fb = _recast(a, results), _recast(b, results)
    return fa is not None and fb is not None and fa != fb


def _neighbour(result: dict, results: list[dict], rank: int) -> dict | None:
    """The rank-`rank` row ~one quarter (75-105 days) before `result`.

    Paired by REPORT DATE, never by fiscal-year label: WDAY's FY26 10-Qs and
    its FY25 10-Qs both carried fiscal_year 2025, so a (fy, rank) key paired
    Q3 FY26 with Q2 FY25 and served revenue 21% high.
    """
    d = _report_date(result)
    for other in results:
        if other is result or _quarter_rank(other) != rank:
            continue
        gap = _days(d, _report_date(other))
        if gap is not None and 75 <= gap <= 105:
            return other
    return None


_DURATION_RANK = ((65, 115, 1), (155, 205, 2), (245, 295, 3))


def _normalize_quarter_period(result: dict) -> None:
    """Trust the revenue fact's own period over the filing's labels.

    DEI fp and "latest fact end" describe the filing, not the period: BX's
    Q2'24 10-Q carried fp=Q1 (its six-month YTD served as a standalone Q1),
    and a subsequent-event fact dated its Q3'26 10-Q 2026-10-31, so neither
    found its neighbour and both stayed YTD.
    """
    bc = result.get("basis_check") or {}
    if result.get("period_type") != "quarterly" or not bc.get("duration_days"):
        return
    fi = result.setdefault("filing_info", {})
    end = bc.get("period_end")
    rd = _report_date(result)
    if end and (pd.isna(rd) or abs((pd.Timestamp(end) - rd).days) > 7):
        fi["report_date"] = end
    rank = next((k for lo, hi, k in _DURATION_RANK if lo <= bc["duration_days"] <= hi), None)
    if rank and rank != _quarter_rank(result):
        result["quarter_label"] = {1: "Q1 (Standalone)", 2: "Q2 (6-month YTD)",
                                   3: "Q3 (9-month YTD)"}[rank]
        result["is_ytd"] = rank > 1


def _decumulate_quarters_inmemory(results: list[dict]) -> list[dict]:
    """Turn per-quarter YTD extractions into standalone 3-month quarters.

    Income statement: the filing's own 3-month figures (`metrics_3m`) when
    tagged, else YTD − the neighbouring quarter's YTD. Cash flow is YTD-only in
    10-Qs, so it is always differenced. Each row gets a `quality` flag:
      - Q1 / already-standalone rows pass through       → "standalone"
      - YTD row with its neighbour in hand               → "standalone_decumulated"
      - YTD row whose neighbour is MISSING stays YTD     → "ytd_fallback"
    The YTD snapshots are taken before any mutation, so decumulation order
    can never subtract an already-decumulated neighbour.
    """
    for r in results:
        _normalize_quarter_period(r)
        r["_ytd_metrics"] = dict(r.get("metrics") or {})

    for r in results:
        rk = _quarter_rank(r)
        cm = r.get("metrics") or {}
        if not (rk and cm):
            r["quality"] = "unknown_period"
            continue

        if rk == 1 or not r.get("is_ytd"):
            r["is_ytd"] = False
            r["quarter_label"] = f"Q{min(rk, 4)} (Standalone)"
            r["quality"] = "standalone"
            r["eps_source"] = "reported"
            continue

        nb = _neighbour(r, results, rk - 1)
        if nb is None:
            r["is_ytd"] = True
            r["quality"] = "ytd_fallback"
            continue

        pm = nb["_ytd_metrics"]
        r["_was_ytd"] = True  # `_ytd_metrics` is this filing's cumulative YTD
        direct = r.get("metrics_3m") or {}
        broken = _basis_break(r, nb, results)
        for k in _FLOW_DECUM_METRICS:
            if k in direct:
                cm[k] = direct[k]
                continue
            cv, pv = cm.get(k), pm.get(k)
            # A YTD value whose neighbour never reported the column cannot be
            # decumulated and must NOT survive on a standalone row; neither can
            # one whose neighbour sits on a different (recast) basis.
            cm[k] = cv - pv if (cv is not None and pv is not None and not broken) else None
        for k in ("weighted_shares_basic", "weighted_shares_diluted"):
            if k in direct:
                cm[k] = direct[k]
        r["eps_source"] = _recompute_derived_quarter_metrics(
            cm, prior_eps=None if broken else {k: pm.get(k) for k in _PER_SHARE_METRICS},
            direct=direct, industry=r.get("industry_class"))
        if broken:
            _note(r, "basis_break", "Prior quarter was reported on a different basis "
                  "(restated comparatives); differenced lines are left empty.")
        nulled = _guard_impossible(cm)
        if nulled:
            _note(r, "impossible_quarter_value", f"Negative {', '.join(nulled)} nulled.")
        r["is_ytd"] = False
        r["quarter_label"] = f"Q{min(rk, 4)} (Standalone)"
        r["quality"] = "standalone_decumulated"
    return results


def _synthesize_q4(ticker: str, results: list[dict]) -> list[dict]:
    """Append synthesized Q4 rows: Q4 = FY − (Q1+Q2+Q3) for flow metrics.

    Q4 has no 10-Q filing, so quarterly histories built from 10-Qs have a
    hole at every fiscal year end. For each Q3 with its Q2 and Q1 in hand
    (chained by report date, ~one quarter apart), extract the 10-K whose
    period end lands 60-130 days after Q3 and derive the fourth. Balance-sheet
    (instant) metrics come from the 10-K directly.

    Refused (flows left empty, row kept for its balance sheet) when the 10-K
    and the Q3 10-Q sit on different bases: HON's FY2025 10-K restated for the
    Solstice spin while its Q3 10-Q predates it, which served Q4 revenue
    $6.86B and gross profit −$7.9B.
    """
    standalone = [r for r in results if _quarter_rank(r) and (r.get("quality") or "").startswith("standalone")]
    annual_filings: list | None = None  # lazy — one fetch for all groups
    new_rows: list[dict] = []
    for q3 in sorted((r for r in standalone if _quarter_rank(r) == 3), key=_report_date):
        q2 = _neighbour(q3, standalone, 2)
        q1 = _neighbour(q2, standalone, 1) if q2 else None
        q3_end = _report_date(q3)
        if not (q1 and q2) or pd.isna(q3_end):
            continue
        if any(_quarter_rank(r) == 4 and 60 <= (_days(_report_date(r), q3_end) or -1) <= 130
               for r in results + new_rows):
            continue
        if annual_filings is None:
            try:
                annual_filings = get_sec_client().get_periodic_filings_smart(
                    ticker, form_type="10-K", limit=15) or []
            except Exception as exc:
                log.debug("Q4 synthesis: 10-K listing failed for %s: %s", ticker, exc)
                annual_filings = []
        acc = None
        for f in annual_filings:
            rd = pd.to_datetime(getattr(f, "report_date", "") or "", errors="coerce")
            if pd.notna(rd) and 60 <= (rd - q3_end).days <= 130:
                acc = f.accession_number
                break
        if not acc:
            continue
        try:
            annual = extract_financials(ticker, accession=acc, form_type="10-K")
        except Exception as exc:
            log.debug("Q4 synthesis: annual extract failed for %s: %s", ticker, exc)
            continue
        if not annual or annual.get("error") or not annual.get("metrics"):
            continue
        annual = {**annual, "metrics": dict(annual["metrics"])}
        _align_to(annual, q3)
        am = annual["metrics"]
        quarters = (q1, q2, q3)

        # Basis: did the 10-K restate its prior year relative to the four
        # quarters we already hold for that year, and did the Q3 10-Q?
        broken = False
        py_fy = (annual.get("basis_check") or {}).get("revenue_prior_year")
        prior = [r for r in results + new_rows
                 if _quarter_rank(r) and (r.get("quality") or "").startswith(("standalone", "q4_"))
                 and 260 <= (_days(q3_end, _report_date(r)) or -1) <= 560]
        prior_rev = [(r.get("metrics") or {}).get("revenue") for r in prior]
        k_flag = (abs(py_fy - sum(prior_rev)) / abs(sum(prior_rev)) > 0.005
                  if py_fy and len(prior) == 4 and None not in prior_rev and sum(prior_rev) else None)
        q_flag = _recast(q3, results)
        broken = k_flag is not None and q_flag is not None and k_flag != q_flag

        # Subtract the Q3 10-Q's OWN nine-month YTD, not Σ of the standalone
        # quarters: once Q2/Q3 carry their filed 3-month figures the three
        # quarters no longer telescope to the 9M total when the basis moved
        # mid-year (JNJ's Kenvue exit: ΣQ1-3 on mixed bases served Q4 revenue
        # $13.5B vs $21.4B filed; APP's apps sale likewise). FY − 9M uses the
        # two filings the basis check above vouches for.
        nine = q3.get("_ytd_metrics") if q3.get("_was_ytd") else None
        q4m = dict(am)  # instants (balance sheet) carry over from the 10-K
        for k in _FLOW_DECUM_METRICS:
            fyv = am.get(k)
            if nine is not None:
                ytd = nine.get(k)
                q4m[k] = fyv - ytd if (fyv is not None and ytd is not None and not broken) else None
                continue
            qs = [(q.get("metrics") or {}).get(k) for q in quarters]
            q4m[k] = fyv - sum(qs) if (fyv is not None and None not in qs and not broken) else None
        eps_sum = {}
        for k in _PER_SHARE_METRICS:
            if nine is not None:
                eps_sum[k] = nine.get(k) if not broken else None
                continue
            eps_qs = [(q.get("metrics") or {}).get(k) for q in quarters]
            eps_sum[k] = sum(eps_qs) if (None not in eps_qs and not broken) else None
        _recompute_derived_quarter_metrics(
            q4m, prior_eps=eps_sum, industry=annual.get("industry_class"))
        nulled = _guard_impossible(q4m)
        if not broken and (q4m.get("revenue") is None or q4m["revenue"] <= 0):
            # No revenue, or the matched 10-K doesn't cover these quarters.
            continue

        q4 = {**annual,
              "fiscal_year": q3.get("fiscal_year"),  # group's label (10-K DEI fy can disagree — CRM)
              "metrics": q4m,
              "period_type": "quarterly",
              "quarter_label": "Q4 (Synthesized)",
              "is_ytd": False,
              "quality": "q4_synthesized",
              "eps_source": "derived",
              "ratios": _compute_ratios(q4m)}
        q4.pop("metrics_3m", None)
        if broken:
            _note(q4, "basis_break", "The 10-K restated prior periods (e.g. a spin-off) "
                  "but the Q3 10-Q did not; Q4 flows cannot be derived as FY − ΣQ1-3.")
        if nulled:
            _note(q4, "impossible_quarter_value", f"Negative {', '.join(nulled)} nulled.")
        new_rows.append(q4)

    results = results + new_rows
    results.sort(key=lambda r: (r.get("filing_info") or {}).get("report_date") or "",
                 reverse=True)
    return results


HISTORY_VERSION = 3  # 3: ga_expense decumulated (v2 served AAPL's YTD G&A as quarters)


def get_fmp_shaped_history(ticker: str, *, period: str = "annual", limit: int = 8) -> dict:
    """SEC-native replacement for FMP's multi-period statements. Returns
    {ticker, period, income[], balance[], cashflow[], source:"sec"} newest-first
    — a drop-in for the old FMP-backed /api/financials-history. Quarters are
    standalone. FMP is NOT used. Periods are extracted in PARALLEL as YTD, then
    quarters are decumulated in-memory (avoids a 2nd extraction per quarter).
    """
    tk = ticker.upper()
    is_q = period == "quarter"
    form = "10-Q" if is_q else "10-K"
    want = min(max(limit, 1), 12)
    # Pull one extra quarter so the oldest returned has a decumulation neighbour.
    fetch_n = want + 1 if is_q else want
    try:
        client = get_sec_client()
        filings = client.get_periodic_filings_smart(tk, form_type=form, limit=max(fetch_n, 12)) or []
        by_accession = {}
        for filing in filings:
            if filing.accession_number:
                by_accession.setdefault(filing.accession_number, filing)
        selected = list(by_accession.values())[:fetch_n]

        def _ex(filing: FilingMetadata) -> dict | None:
            try:
                return extract_financials(tk, filing=filing, standalone_quarter=False)
            except Exception as exc:
                log.debug("history extract failed for %s/%s: %s", tk, filing.accession_number, exc)
                return None

        # Fanning this out is COUNTERPRODUCTIVE, which is unintuitive enough to
        # be worth writing down. Each worker calls extract_financials, which
        # calls get_company_info -> sec_client._get_submissions — a lockless
        # 120s dict cache. Cold, all N workers miss it simultaneously and then
        # serialize behind the 8 req/s rate lock, so the fan-out buys N
        # duplicate SEC round trips and no parallelism. Measured on a genuinely
        # cold build (disk cache stubbed): 1 worker 9.3-10.0s, 8 workers
        # 12.5-14.5s — 25-35% SLOWER.
        #
        # Default 1. Raise SEC_HISTORY_WORKERS only after adding a single-flight
        # lock to the submissions cache, which is what would make >1 pay off.
        workers = max(1, int(os.environ.get("SEC_HISTORY_WORKERS", "1")))
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(selected)))) as ex:
            results = [r for r in ex.map(_ex, selected)
                       if r and (r.get("metrics") or {}).get("revenue") is not None]

        # newest-first by report period end
        results.sort(key=lambda r: (r.get("filing_info") or {}).get("report_date") or "", reverse=True)
        # Per-share history onto the newest share basis BEFORE any quarter
        # arithmetic — a Q4 or decumulated quarter that straddles a split
        # otherwise mixes pre- and post-split EPS.
        results = _split_adjust(results)
        if is_q:
            is_foreign = has_release_format(tk) or any(f.form_type in ("6-K", "6-K/A") for f in selected)
            if is_foreign:
                # Amended accessions have precedence over the original. Limit
                # after deduplicating and merging verified exhibits.
                unique = {}
                for row in sorted(results, key=lambda r: (
                    r["filing_info"]["filing_date"], r["filing_info"]["accession_number"]
                ), reverse=True):
                    rank = _quarter_rank(row)
                    if rank:
                        unique.setdefault((row["fiscal_year"], f"Q{rank}"), row)
                results = _decumulate_quarters_inmemory(list(unique.values()))
                try:
                    results.extend(release_quarters(tk, want, set(unique)))
                except Exception as exc:
                    log.warning("6-K exhibit lookup failed for %s: %s", tk, exc)
                results.sort(key=lambda r: r["filing_info"]["report_date"], reverse=True)
            else:
                results = _synthesize_q4(tk, _decumulate_quarters_inmemory(results))
        results = results[:want]

        income, balance, cashflow = [], [], []
        for data in results:
            rows = sec_to_fmp_statements(data)
            income.append(rows["income"])
            balance.append(rows["balance"])
            cashflow.append(rows["cashflow"])
        # historyVersion 2 (Nov 2026 audit): weighted basic/diluted shares,
        # filed 3-month EPS and income lines, date-paired decumulation,
        # basis-break guards, split-adjusted per-share history, OI-based EBITDA.
        # Cached blobs without it are rebuilt in the background on read.
        return {"ticker": tk, "period": period, "interimVersion": 1,
                "historyVersion": HISTORY_VERSION, "income": income,
                "balance": balance, "cashflow": cashflow, "source": "sec"}
    except Exception as exc:
        log.warning("get_fmp_shaped_history failed for %s: %s", tk, exc)
        return {"ticker": tk, "period": period, "income": [],
                "balance": [], "cashflow": [], "source": "sec"}


# ═══════════════════════════════════════════════════════════════════════════
#  Batch extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_financials_batch(
    tickers: list[str],
    *,
    year: int | None = None,
    form_type: str = "10-K",
    include_statements: bool = False,
    include_segments: bool = False,
    period_index: int = 0,
    max_workers: int = 5,
) -> list[dict]:
    """Extract financials for multiple companies in parallel.

    Uses a thread pool to fetch data concurrently while respecting
    SEC's rate limits (handled by SECClient's rate limiter).
    """
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_financials, t,
                year=year,
                form_type=form_type,
                include_statements=include_statements,
                include_segments=include_segments,
                period_index=period_index,
            ): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                data = future.result()
                if data is not None:
                    results.append(data)
                else:
                    results.append({
                        "ticker_or_cik": ticker,
                        "error": "No financials available",
                    })
            except Exception as exc:
                results.append({
                    "ticker_or_cik": ticker,
                    "error": str(exc),
                })
    return results
