"""Shared Supabase financial_cache period-key builders.

Every reader/writer of the 'financials' data_type must build its period key
here. A silent mismatch between _handle_financials ("v2|10-K|latest") and
batch_cache ("10-K|latest") meant the batch sweep never warmed /api/metrics.
"""

from __future__ import annotations

FIN_CACHE_VERSION = "v2"

# TTM-derived metrics block (core/ttm.py output) — its own data_type so the
# annual/quarterly 'financials' entries stay untouched.
TTM_DATA_TYPE = "ttm_metrics"
TTM_PERIOD = "ttm"

# Latest STANDALONE quarter metrics block (core/ttm.py) — distinct from the
# as-reported 10-Q 'financials' entry, whose Q2/Q3 flows are YTD.
QUARTER_METRICS_DATA_TYPE = "quarter_metrics"
QUARTER_METRICS_PERIOD = "latest"


def financials_period_key(form_type: str = "10-K", year: int | None = None) -> str:
    """Versioned period key for the 'financials' data_type: 'v2|10-K|latest'."""
    return f"{FIN_CACHE_VERSION}|{form_type}|{year or 'latest'}"


def legacy_financials_period_key(form_type: str = "10-K", year: int | None = None) -> str:
    """Pre-versioning key ('10-K|latest') — dual-written by the batch during the
    transition so an already-deployed reader never goes cold. Remove one release
    after the v2-everywhere deploy is verified."""
    return f"{form_type}|{year or 'latest'}"
