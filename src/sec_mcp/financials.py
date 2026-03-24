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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from sec_mcp import disk_cache
from sec_mcp.config import get_config
from sec_mcp.sec_client import get_sec_client
from sec_mcp.xbrl_mappings import (
    ABSTRACT_COLUMNS,
    CONCEPT_MAP,
    ConceptEntry,
    DIMENSION_COLUMNS,
    EBITDA_COMPONENTS,
    IndustryClass,
    LEVEL_COLUMNS,
    MAX_ACCEPTABLE_LEVEL,
    detect_industry_class,
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
    # Use smart filing search with automatic FPI fallback (20-F for 10-K, 6-K for 10-Q)
    filings = client.get_filings_smart(ticker_or_cik, form_type=form_type, limit=40)

    if not filings:
        return None

    if year is not None:
        # Filter to filings from the target year or year+1
        # (companies often file annual reports in Jan/Feb of the following year)
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
) -> ResolvedMetric:
    """Resolve a metric from a facts DataFrame using ordered concept list.

    Resolution order (4-pass):
      1. Exact match on non-aggregate us-gaap concepts (highest confidence)
      2. Contains match on non-aggregate concepts (broader)
      3. Custom extension pattern match (bank: NetRevenue patterns)
      4. Aggregate fallback (sum of components)

    duration_pref:
      "quarterly" — prefer 3-month durations (for 10-Q flow metrics)
      "annual"    — prefer 12-month durations (for 10-K flow metrics)
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

    # --- Pass 2: Non-aggregate contains match (broader) ---
    for entry in concepts:
        if entry.aggregate:
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
    if industry == IndustryClass.BANK:
        val, src = _resolve_custom_extension(
            clean_df, is_custom_net_revenue, "Custom Net Revenue", period_index
        )
        if val is not None:
            return ResolvedMetric(val, src, confidence=0.85, method="custom_ext")
    else:
        val, src = _resolve_custom_extension(
            clean_df, is_custom_revenue, "Custom Revenue", period_index
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
        # For aggregates, try exact first, then contains
        val = _lookup_fact(clean_df, entry.xbrl_concept, period_index, match_mode="exact")
        if val is None:
            val = _lookup_fact(clean_df, entry.xbrl_concept, period_index, match_mode="contains")
        if val is not None:
            total += val
            matched_any = True
            components.append(entry.display_name)

    if matched_any:
        label = " + ".join(components)
        conf = min(0.85, 0.60 + 0.05 * len(components))
        return ResolvedMetric(total, label, confidence=conf, method="aggregate")

    return ResolvedMetric(None, None)


def _resolve_custom_extension(
    facts_df: pd.DataFrame,
    pattern_fn,
    label_prefix: str,
    period_index: int,
) -> tuple[float | None, str | None]:
    """Scan all concepts in the DataFrame for custom extension matches."""
    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(facts_df, ("value", "Value", "val"))
    if concept_col is None or value_col is None:
        return None, None

    for _, row in facts_df.iterrows():
        concept_name = str(row[concept_col])
        if pattern_fn(concept_name):
            # Skip guidance/forecast/estimate concepts
            concept_lo = concept_name.lower()
            if any(kw in concept_lo for kw in ("guidance", "forecast", "estimated",
                                                 "expected", "projected", "outlook")):
                continue
            val = _safe(row[value_col])
            if val is not None:
                return val, f"{label_prefix} ({concept_name})"

    return None, None


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

    # Duration-aware filtering for flow metrics
    # SEC companyfacts reports both quarterly and annual/YTD values
    # with the same end date but different start dates.
    # Balance sheet items are instant (no start date) — unaffected.
    if duration_pref and end_col and "start" in matches.columns:
        matches = _filter_by_duration(
            matches, end_col, prefer=duration_pref,
            target_fp=target_fp, target_fy=target_fy,
        )

    if period_index >= len(matches):
        return None

    raw = matches.iloc[period_index][value_col]
    return _safe(raw)


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
    "Cash", "Equity",
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

def extract_financials(
    ticker_or_cik: str,
    *,
    year: int | None = None,
    form_type: str = "10-K",
    accession: str | None = None,
    include_statements: bool = False,
    include_segments: bool = False,
    period_index: int = 0,
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

    if accession:
        # Specific filing requested — look up its metadata from submissions
        filing_meta = {
            "accession_number": accession,
            "form_type": form_type,
            "filing_date": "",
        }
        try:
            # Search by form_type first (faster for prolific filers like MS)
            for ft_search in [form_type, "10-K", "20-F", "10-Q", "6-K", None]:
                found = False
                search_filings = client.get_filings(
                    ticker_or_cik, form_type=ft_search, limit=50
                )
                for f in search_filings:
                    if f.accession_number == accession:
                        filing_meta["filing_date"] = f.filing_date
                        filing_meta["form_type"] = f.form_type
                        found = True
                        break
                if found:
                    break
        except Exception as exc:
            log.warning("Filing metadata lookup failed for %s: %s", ticker_or_cik, exc)
    else:
        # Search by year/form_type — _find_filing_accession uses get_filings_smart
        # which auto-tries FPI alternatives (20-F for 10-K, 6-K for 10-Q)
        forms_to_try = [form_type]
        if form_type in ("10-K", "20-F"):
            forms_to_try.append("10-Q")  # fallback to quarterly if no annual
        else:
            forms_to_try.append("10-K")  # fallback to annual if no quarterly
        for ft in forms_to_try:
            filing_meta = _find_filing_accession(ticker_or_cik, form_type=ft, year=year)
            if filing_meta:
                break

    if filing_meta:
        result["filing_info"] = filing_meta
        acc = filing_meta.get("accession_number")
        if acc and cik_val:
            result["sec_links"].update(_build_sec_links(cik_val, accession=acc))

    # ── Check disk cache ─────────────────────────────────────────────
    if filing_meta and filing_meta.get("accession_number"):
        cached = disk_cache.get(ticker_or_cik, filing_meta["accession_number"])
        if cached and "data" in cached:
            log.info("Disk cache hit for %s / %s", ticker_or_cik, filing_meta["accession_number"])
            return cached["data"]

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

    # Fallback 2: drop accession filter (accession format might not match)
    if (facts_df is None or facts_df.empty) and accession:
        log.info("Retrying XBRL facts for %s without accession filter", ticker_or_cik)
        try:
            facts_df = _get_facts_for_filing(
                ticker_or_cik,
                accession=None,
                form_type=actual_form,
            )
        except Exception:
            facts_df = pd.DataFrame()

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

    # ── Extract metrics ───────────────────────────────────────────────
    metrics: dict[str, float | None] = {}
    sourced: dict[str, str | None] = {}
    confidence: dict[str, float] = {}

    # Duration preference: quarterly for 10-Q/6-K, annual for 10-K/20-F
    is_quarterly = actual_form in ("10-Q", "10-q", "6-K", "6-k")
    dur_pref = "quarterly" if is_quarterly else "annual"
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

    # Flag YTD vs standalone for 10-Q data
    if is_quarterly and target_fp:
        fp_upper = target_fp.upper()
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
        target_fp=target_fp, target_fy=target_fy,
    )
    metrics["revenue"] = resolved.value
    sourced["revenue"] = resolved.source
    confidence["revenue"] = resolved.confidence

    # All other metrics from the CONCEPT_MAP
    for metric_name, concepts in CONCEPT_MAP.items():
        resolved = _resolve_metric(
            facts_df, concepts, period_index=period_index,
            industry=industry, duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy,
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

    # Operating Income = Gross Profit - Operating Expenses (if missing)
    if metrics.get("operating_income") is None:
        gp_val = metrics.get("gross_profit")
        opex_val = metrics.get("operating_expenses")
        if gp_val is not None and opex_val is not None:
            metrics["operating_income"] = gp_val - abs(opex_val)
            sourced["operating_income"] = "Computed: Gross Profit - OpEx"
            confidence["operating_income"] = 0.70

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

    # D&A lookup: try broad totals first, then sum components
    da_val = None
    for tag in ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"):
        da_val = _lookup_fact(facts_df, tag, period_index,
                              match_mode="exact", duration_pref=dur_pref,
                              target_fp=target_fp, target_fy=target_fy)
        if da_val is not None:
            break
    if da_val is None:
        dep = _lookup_fact(facts_df, "Depreciation", period_index,
                           match_mode="exact", duration_pref=dur_pref,
                           target_fp=target_fp, target_fy=target_fy)
        amort = _lookup_fact(facts_df, "AmortizationOfIntangibleAssets",
                             period_index, match_mode="exact",
                             duration_pref=dur_pref,
                             target_fp=target_fp, target_fy=target_fy)
        if dep is not None or amort is not None:
            da_val = abs(dep or 0) + abs(amort or 0)

    if ni_val is not None and da_val is not None:
        # Proper EBITDA: NI + Interest + Tax + D&A
        ebitda_val = ni_val + abs(int_exp or 0) + abs(tax_exp or 0) + abs(da_val)
        metrics["ebitda"] = ebitda_val
        confidence["ebitda"] = 0.85 if (int_exp is not None and tax_exp is not None) else 0.70
    elif oi is not None and da_val is not None:
        # Fallback: EBIT + D&A ≈ Operating Income + D&A
        metrics["ebitda"] = oi + abs(da_val)
        confidence["ebitda"] = 0.65
        sourced["ebitda"] = "Approximated: Operating Income + D&A"
    else:
        metrics["ebitda"] = None
        confidence["ebitda"] = 0.0

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

    # ── Currency detection & conversion to USD ─────────────────────
    from sec_mcp.core.fx import detect_currency, convert_metrics_to_usd
    reporting_currency = detect_currency(facts_df)
    result["reporting_currency"] = reporting_currency
    if reporting_currency != "USD":
        metrics, fx_rate = convert_metrics_to_usd(metrics, reporting_currency)
        result["fx_rate"] = fx_rate
        result["fx_note"] = f"Converted from {reporting_currency} to USD at {fx_rate:.4f}" if fx_rate else f"Reported in {reporting_currency} (conversion unavailable)"
        log.info("Foreign filer %s reports in %s, converted to USD", ticker_or_cik, reporting_currency)

    result["metrics"] = metrics
    result["metrics_sourced"] = sourced
    result["confidence_scores"] = confidence
    result["ratios"] = _compute_ratios(metrics)
    result["validation"] = _validate(metrics, industry, confidence)

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
            industry=industry, duration_pref=dur_pref,
        )
        prior_metrics["revenue"] = prev_rev.value
        for metric_name, concepts in CONCEPT_MAP.items():
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
                industry=industry, duration_pref=dur_pref,
            )
            qoq_metrics["revenue"] = q_rev.value
            for metric_name, concepts in CONCEPT_MAP.items():
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

    # ── Segments ──────────────────────────────────────────────────────
    if include_segments:
        result["segments"] = _extract_segments(
            facts_df, target_fp=target_fp, target_fy=target_fy
        )

    # ── Statements ────────────────────────────────────────────────────
    # Build statement tables from companyfacts data
    if include_statements:
        result["income_statement"] = _build_statement_from_facts(
            facts_df, _INCOME_CONCEPTS, duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy)
        result["balance_sheet"] = _build_statement_from_facts(
            facts_df, _BALANCE_CONCEPTS, is_instant_statement=True,
            target_fp=target_fp, target_fy=target_fy)
        result["cash_flow_statement"] = _build_statement_from_facts(
            facts_df, _CASHFLOW_CONCEPTS, duration_pref=dur_pref,
            target_fp=target_fp, target_fy=target_fy)

    # ── Write to disk cache ──────────────────────────────────────────
    if filing_meta and filing_meta.get("accession_number"):
        try:
            summary = generate_local_summary(result)
            disk_cache.put(ticker_or_cik, filing_meta["accession_number"], result, summary)
        except Exception as exc:
            log.warning("Disk cache write failed for %s: %s", ticker_or_cik, exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Ratios
# ═══════════════════════════════════════════════════════════════════════════

def _div(a: float | None, b: float | None) -> float | None:
    """Safe division — returns None if either operand is None or divisor is zero."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def _compute_ratios(m: dict[str, float | None]) -> dict[str, float | None]:
    """Compute financial ratios from extracted metrics."""
    rev = m.get("revenue")
    ni = m.get("net_income")
    gp = m.get("gross_profit")
    oi = m.get("operating_income")
    ta = m.get("total_assets")
    ca = m.get("current_assets")
    cl = m.get("current_liabilities")
    eq = m.get("stockholders_equity")
    ocf = m.get("operating_cash_flow")
    ebitda = m.get("ebitda")
    ltd = m.get("long_term_debt")
    std = m.get("short_term_debt")
    fcf = m.get("free_cash_flow")

    total_debt = None
    if ltd is not None or std is not None:
        total_debt = (ltd or 0) + (std or 0)

    return {
        "gross_margin": _div(gp, rev),
        "operating_margin": _div(oi, rev),
        "net_margin": _div(ni, rev),
        "return_on_assets": _div(ni, ta),
        "return_on_equity": _div(ni, eq),
        "roe": _div(ni, eq),  # alias used by the UI
        "current_ratio": _div(ca, cl),
        "debt_to_equity": _div(total_debt, eq),
        "debt_to_assets": _div(total_debt, ta),
        "ebitda_margin": _div(ebitda, rev),
        "fcf_margin": _div(fcf, rev),
        "ocf_to_net_income": _div(ocf, ni),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

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

    # Strategy 1: If L+E > A, equity might include NCI that should be separate
    # Try: Equity = Assets - Liabilities (derive equity from the equation)
    if tl is not None and ta is not None:
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
        if abs(tl - ta) / max(abs(ta), 1) < 0.02:
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
    if nci is not None and nci > 0:
        eq_with_nci = eq + nci
        new_diff = abs(ta - (tl + eq_with_nci)) / max(abs(ta), 1)
        if new_diff < 0.05:
            repairs["stockholders_equity"] = eq_with_nci
            log.info("Balance sheet repair: added NCI %s to equity", _fmt(nci))
            return repairs

    # Strategy 4: Try computing total_liabilities = total_assets - equity
    if ta is not None and eq is not None:
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

    # Rule 2: Accounting equation: A = L + E (within 5% tolerance)
    # If it doesn't balance, try to auto-repair from the raw XBRL data
    if ta is not None and tl is not None and eq is not None:
        expected = tl + eq
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
                new_expected = (tl or 0) + (eq or 0)
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
