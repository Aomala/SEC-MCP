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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

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
            except Exception:
                pass
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
        val = _lookup_fact(clean_df, entry.xbrl_concept, period_index,
                           match_mode="exact", duration_pref=dp)
        if val is not None:
            return ResolvedMetric(val, entry.display_name, confidence=0.95, method="exact")

    # --- Pass 2: Non-aggregate contains match (broader) ---
    for entry in concepts:
        if entry.aggregate:
            continue
        is_instant = entry.xbrl_concept in _INSTANT_CONCEPTS
        dp = duration_pref if not is_instant else None
        val = _lookup_fact(clean_df, entry.xbrl_concept, period_index,
                           match_mode="contains", duration_pref=dp)
        if val is not None:
            return ResolvedMetric(val, entry.display_name, confidence=0.80, method="contains")

    # --- Pass 3: Custom extension patterns (banks, large filers) ---
    if industry == IndustryClass.BANK:
        val, src = _resolve_custom_extension(
            clean_df, is_custom_net_revenue, "Custom Net Revenue", period_index
        )
        if val is not None:
            return ResolvedMetric(val, src, confidence=0.70, method="custom_ext")
    else:
        val, src = _resolve_custom_extension(
            clean_df, is_custom_revenue, "Custom Revenue", period_index
        )
        if val is not None:
            return ResolvedMetric(val, src, confidence=0.65, method="custom_ext")

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
        conf = min(0.60, 0.30 + 0.10 * len(components))
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
            except Exception:
                pass
            break

    # Duration-aware filtering for flow metrics
    # SEC companyfacts reports both quarterly and annual/YTD values
    # with the same end date but different start dates.
    # Balance sheet items are instant (no start date) — unaffected.
    if duration_pref and end_col and "start" in matches.columns:
        matches = _filter_by_duration(matches, end_col, prefer=duration_pref)

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


def _filter_by_duration(
    df: pd.DataFrame,
    end_col: str,
    prefer: str = "quarterly",
) -> pd.DataFrame:
    """Filter facts to prefer quarterly (3-month) or annual (12-month) durations.

    For each unique end date, if multiple durations exist, keep only the
    shortest one (quarterly) or longest (annual/YTD) depending on `prefer`.
    Rows without a start date (instant/balance sheet items) are kept as-is.
    """
    if "start" not in df.columns:
        return df

    df = df.copy()

    # Calculate duration in days
    start_dt = pd.to_datetime(df["start"], errors="coerce")
    end_dt = pd.to_datetime(df[end_col], errors="coerce")
    df["_duration_days"] = (end_dt - start_dt).dt.days

    # Separate instant (no start) vs duration-based rows
    instant_mask = df["start"].isna() | df["_duration_days"].isna()
    instant_rows = df[instant_mask]
    duration_rows = df[~instant_mask]

    if duration_rows.empty:
        df.drop(columns=["_duration_days"], inplace=True, errors="ignore")
        return df

    if prefer == "quarterly":
        # For each (concept, end_date) group, keep the shortest duration
        # Quarterly ≈ 80-100 days, YTD ≈ 170-370 days
        # Keep rows with duration < 120 days if available, else keep all
        short = duration_rows[duration_rows["_duration_days"] <= 120]
        if not short.empty:
            duration_rows = short
        # If no short durations found, keep the shortest available per end date
        elif not duration_rows.empty:
            concept_col = _find_column(duration_rows, ("concept", "Concept", "tag", "Tag"))
            if concept_col and end_col:
                idx = duration_rows.groupby([concept_col, end_col])["_duration_days"].idxmin()
                duration_rows = duration_rows.loc[idx]
    else:
        # Keep the longest duration (annual/YTD)
        long = duration_rows[duration_rows["_duration_days"] > 300]
        if not long.empty:
            duration_rows = long

    result = pd.concat([instant_rows, duration_rows], ignore_index=True)
    result.drop(columns=["_duration_days"], inplace=True, errors="ignore")

    # Re-sort by end date
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

def _extract_segments(facts_df: pd.DataFrame | None) -> dict:
    """Extract revenue and geographic segments from XBRL facts.

    The companyfacts API doesn't include dimensional data directly,
    so segment extraction is limited. We look for segment-related
    concepts in the fact names themselves.
    """
    result: dict = {"revenue_segments": [], "geographic_segments": []}

    if facts_df is None or facts_df.empty:
        return result

    concept_col = _find_column(facts_df, ("concept", "Concept", "tag", "Tag"))
    value_col = _find_column(facts_df, ("value", "Value", "val"))
    if concept_col is None or value_col is None:
        return result

    # Look for segment-related revenue concepts
    rev_segment_patterns = [
        "RevenueFrom", "SalesRevenue", "NetRevenue",
        "ProductRevenue", "ServiceRevenue", "SubscriptionRevenue",
    ]
    geo_patterns = [
        "Americas", "Europe", "Asia", "International", "Domestic",
        "UnitedStates", "China", "Japan",
    ]

    for _, row in facts_df.head(500).iterrows():  # Limit to avoid slow iteration
        concept_name = str(row[concept_col])
        val = _safe(row[value_col])
        if val is None:
            continue

        # Check for revenue segments
        for pattern in rev_segment_patterns:
            if pattern.lower() in concept_name.lower():
                result["revenue_segments"].append({
                    "segment": concept_name,
                    "value": val,
                })
                break

        # Check for geographic segments
        for pattern in geo_patterns:
            if pattern.lower() in concept_name.lower():
                result["geographic_segments"].append({
                    "segment": concept_name,
                    "value": val,
                })
                break

    return result


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
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        idx = int(response.content[0].text.strip())
        if 0 <= idx < len(candidates):
            return idx
    except Exception:
        pass
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
    except Exception:
        return []


def _build_statement_from_facts(
    facts_df: pd.DataFrame,
    statement_concepts: list[str],
    *,
    duration_pref: str | None = None,
    is_instant_statement: bool = False,
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

    # Pre-filter by duration for flow statements (income, cash flow)
    # This prevents picking Q4-only values for annual filings
    if duration_pref and not is_instant_statement and end_col and "start" in clean_df.columns:
        clean_df = _filter_by_duration(clean_df, end_col, prefer=duration_pref)

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

        row = matches.iloc[0]
        label_raw = row[label_col] if label_col and label_col in matches.columns else None
        label = str(label_raw) if label_raw is not None and pd.notna(label_raw) else concept_name
        val = _safe(row[value_col])
        end_date = str(row[end_col]) if end_col and end_col in matches.columns else ""

        # Deduplicate by label (prevents "Intangible Assets" appearing twice)
        if label in seen_labels:
            continue
        seen_labels.add(label)

        if val is not None:
            records.append({
                "label": label,
                "concept": concept_name,
                end_date: val,
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
        except Exception:
            pass
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

    # Revenue (industry-aware, with confidence)
    rev_concepts = get_revenue_concepts(industry)
    resolved = _resolve_metric(
        facts_df, rev_concepts, period_index=period_index,
        industry=industry, duration_pref=dur_pref,
    )
    metrics["revenue"] = resolved.value
    sourced["revenue"] = resolved.source
    confidence["revenue"] = resolved.confidence

    # All other metrics from the CONCEPT_MAP
    for metric_name, concepts in CONCEPT_MAP.items():
        resolved = _resolve_metric(
            facts_df, concepts, period_index=period_index,
            industry=industry, duration_pref=dur_pref,
        )
        metrics[metric_name] = resolved.value
        sourced[metric_name] = resolved.source
        confidence[metric_name] = resolved.confidence

    # ── Layer 2: Deterministic rollup fallbacks ─────────────────────
    # Gross Profit = Revenue - Cost of Revenue (if GP missing)
    if metrics.get("gross_profit") is None:
        rev_val = metrics.get("revenue")
        cor_val = metrics.get("cost_of_revenue")
        if rev_val is not None and cor_val is not None:
            metrics["gross_profit"] = rev_val - abs(cor_val)
            sourced["gross_profit"] = "Computed: Revenue - Cost of Revenue"
            confidence["gross_profit"] = min(
                confidence.get("revenue", 0), confidence.get("cost_of_revenue", 0)
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

    # EBITDA (derived: operating income + depreciation & amortization)
    # D&A tags are ordered broadest→narrowest; take the FIRST match
    # to avoid double-counting (DDA already includes D + A)
    oi = metrics.get("operating_income")
    if oi is not None:
        da_val = None
        # First try the broad totals (exact)
        for tag in ("DepreciationDepletionAndAmortization",
                     "DepreciationAndAmortization"):
            da_val = _lookup_fact(facts_df, tag, period_index,
                                  match_mode="exact", duration_pref=dur_pref)
            if da_val is not None:
                break
        # Fallback: sum Depreciation + Amortization separately
        if da_val is None:
            dep = _lookup_fact(facts_df, "Depreciation", period_index,
                               match_mode="exact", duration_pref=dur_pref)
            amort = _lookup_fact(facts_df, "AmortizationOfIntangibleAssets",
                                 period_index, match_mode="exact",
                                 duration_pref=dur_pref)
            if dep is not None or amort is not None:
                da_val = abs(dep or 0) + abs(amort or 0)
        if da_val is not None:
            metrics["ebitda"] = oi + abs(da_val)
            confidence["ebitda"] = 0.70
        else:
            metrics["ebitda"] = None
            confidence["ebitda"] = 0.0
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
                             match_mode="exact", duration_pref=dur_pref)
            if v is not None and v > best_rev:
                best_rev = v
                best_src = f"{tag} (auto-corrected)"
        if best_rev > rev:
            log.info("Auto-corrected revenue: %.0f → %.0f (%s)", rev, best_rev, best_src)
            metrics["revenue"] = best_rev
            sourced["revenue"] = best_src
            confidence["revenue"] = 0.80

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

    result["prior_metrics"] = prior_metrics
    result["qoq_metrics"] = qoq_metrics
    result["comparison_label"] = "QoQ" if is_quarterly else "YoY"
    result["yoy_label"] = "vs Same Quarter Last Year" if is_quarterly else "vs Prior Year"

    # ── Segments ──────────────────────────────────────────────────────
    if include_segments:
        result["segments"] = _extract_segments(facts_df)

    # ── Statements ────────────────────────────────────────────────────
    # Build statement tables from companyfacts data
    if include_statements:
        result["income_statement"] = _build_statement_from_facts(
            facts_df, _INCOME_CONCEPTS, duration_pref=dur_pref)
        result["balance_sheet"] = _build_statement_from_facts(
            facts_df, _BALANCE_CONCEPTS, is_instant_statement=True)
        result["cash_flow_statement"] = _build_statement_from_facts(
            facts_df, _CASHFLOW_CONCEPTS, duration_pref=dur_pref)

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
    if ta is not None and tl is not None and eq is not None:
        expected = tl + eq
        diff_pct = abs(ta - expected) / ta if ta != 0 else 0
        if diff_pct > 0.05:
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
