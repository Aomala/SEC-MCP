"""Period classification and fact deduplication.

The core rules (XBRL US / DERA guidance):

1. A fact's period is defined by its (start, end) dates — NEVER by the fy/fp
   columns. fy/fp describe the FILING that reported the fact, so a FY2024 10-K
   stamps fy=2024 on its FY2023 comparative facts too. Filtering on fp
   collapses comparative years into the target year.

2. companyfacts retains every fact from every filing. The same
   (concept, unit, start, end) appears once per filing that reported it —
   originals, comparatives, and amendments. The current-best value is the one
   with the latest `filed` date (amendments and restated comparatives
   supersede the original).

3. Duration classifies a flow fact: ~91d = one quarter, ~182d = half-year YTD,
   ~273d = nine-month YTD, ~365d = fiscal year. Instant facts (no start) are
   balance-sheet snapshots.
"""

from __future__ import annotations

import enum

import pandas as pd


class PeriodType(enum.Enum):
    INSTANT = "instant"
    QUARTER = "Q"        # ~91 days (standalone quarter)
    HALF_YTD = "H"       # ~182 days (6-month YTD)
    NINE_MONTH_YTD = "9M"  # ~273 days (9-month YTD)
    YEAR = "FY"          # ~365 days (full fiscal year)
    OTHER = "other"      # stub periods, 53-week oddities outside all bands


# Duration bands in days. 52/53-week filers put FY at 363-371; transition
# periods and stubs land in OTHER deliberately so they never masquerade as a
# quarter or a year.
_BANDS: tuple[tuple[PeriodType, int, int], ...] = (
    (PeriodType.QUARTER, 65, 115),
    (PeriodType.HALF_YTD, 155, 205),
    (PeriodType.NINE_MONTH_YTD, 245, 295),
    (PeriodType.YEAR, 340, 400),
)


def classify_duration_days(days: float | None) -> PeriodType:
    if days is None or pd.isna(days):
        return PeriodType.INSTANT
    for ptype, lo, hi in _BANDS:
        if lo <= days <= hi:
            return ptype
    return PeriodType.OTHER


def classify_period(start, end) -> PeriodType:
    """Classify a fact period from its start/end dates (str or Timestamp)."""
    if start is None or (isinstance(start, float) and pd.isna(start)) or start is pd.NaT:
        return PeriodType.INSTANT
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return PeriodType.INSTANT if pd.isna(s) else PeriodType.OTHER
    return classify_duration_days((e - s).days)


# YTD period types that need decumulation to produce a standalone quarter
YTD_TYPES = {PeriodType.HALF_YTD, PeriodType.NINE_MONTH_YTD}
FLOW_TYPES = {PeriodType.QUARTER, PeriodType.HALF_YTD,
              PeriodType.NINE_MONTH_YTD, PeriodType.YEAR}


def annotate_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_ptype` (PeriodType.value) and `_dur_days` columns to a facts frame.

    Expects companyfacts-shaped columns: start, end. Idempotent and cheap;
    callers may reuse the annotated frame.
    """
    if df is None or df.empty or "_ptype" in df.columns:
        return df
    out = df.copy()
    start_dt = pd.to_datetime(out.get("start"), errors="coerce")
    end_dt = pd.to_datetime(out.get("end"), errors="coerce")
    dur = (end_dt - start_dt).dt.days
    out["_dur_days"] = dur
    out["_ptype"] = [classify_duration_days(d) .value for d in dur]
    return out


def dedupe_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated reports of the same fact — latest `filed` wins.

    Key: (concept, taxonomy, units, start, end). companyfacts repeats a fact
    once per filing that reported it; without this dedupe, sorting by `end`
    and indexing (the legacy `period_index` pattern) silently picks an
    arbitrary report — original or restated, whichever sorted first.
    """
    if df is None or df.empty:
        return df
    key_cols = [c for c in ("concept", "taxonomy", "units", "start", "end")
                if c in df.columns]
    if not key_cols or "filed" not in df.columns:
        return df
    out = df.copy()
    filed = pd.to_datetime(out["filed"], errors="coerce")
    out["_filed_sort"] = filed
    # NaT-started instants must group together — fillna a sentinel for the key
    out = out.sort_values("_filed_sort", ascending=True, na_position="first")
    out = out.drop_duplicates(subset=key_cols, keep="last")
    return out.drop(columns=["_filed_sort"])


def select_period_facts(
    df: pd.DataFrame,
    *,
    period_type: PeriodType,
    period_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Facts matching an explicit (period_type, period_end) target.

    This is the replacement for fp-based filtering: the caller decides which
    period it wants (e.g. YEAR ending 2024-09-28) and gets exactly those
    facts, deduped so the latest-filed value wins.
    """
    if df is None or df.empty:
        return df
    work = annotate_periods(df)
    mask = work["_ptype"] == period_type.value
    if period_end is not None:
        end_dt = pd.to_datetime(period_end, errors="coerce")
        mask &= pd.to_datetime(work["end"], errors="coerce") == end_dt
    return dedupe_facts(work[mask])
