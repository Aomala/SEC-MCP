"""Regression tests for the July 2026 debt-extraction + provider-linkage fixes.

Covers three confirmed production bugs (45-ticker live validation):

  BUG 1 — long_term_debt under-extraction for filers whose balance-sheet debt
          lives outside the LongTermDebt tag family:
            O    (REIT):      NotesPayable + LoansPayable + SecuredDebt lines
            MET  (insurance): LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities
            ORCL (FY2026):    LongTermNotesPayable (vintage tag change)
            CVX  (10-Q):      ...IncludingCurrentMaturities (quarterly variant)
  BUG 2 — Polygon rejects SEC dash-class tickers (BRK-B → must query BRK.B)
  BUG 3 — /api/cross-check compared SEC FY against Polygon TTM (false
          mismatches for non-Dec FYE filers) and never populated SEC EPS.

All tests here are OFFLINE (synthetic facts frames / stubbed Polygon rows).
"""

from __future__ import annotations

import pandas as pd
import pytest

from sec_mcp import polygon_client
from sec_mcp.financials import _resolve_metric_legacy
from sec_mcp.polygon_client import _pick_aligned_report, cross_check, normalize_ticker
from sec_mcp.xbrl_mappings import LONG_TERM_DEBT, SHORT_TERM_DEBT, ConceptEntry

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _facts(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """Build a minimal instant-fact frame: (concept, value, end)."""
    return pd.DataFrame([
        {"concept": c, "label": c, "value": v, "start": None, "end": e,
         "filed": "2026-02-25", "form": "10-K", "accn": "0000000000-26-000001",
         "fy": 2025, "fp": "FY", "units": "USD", "taxonomy": "us-gaap"}
        for c, v, e in rows
    ])


def _ltd(df: pd.DataFrame):
    return _resolve_metric_legacy(df, LONG_TERM_DEBT, duration_pref="annual")


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 1 — concept-list shape and resolution order
# ═══════════════════════════════════════════════════════════════════════════

class TestLongTermDebtConceptList:
    def _names(self):
        return [c.xbrl_concept for c in LONG_TERM_DEBT]

    def test_new_filer_variants_present(self):
        names = self._names()
        for tag in (
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
            "LongTermNotesPayable",
            "DebtAndCapitalLeaseObligations",
            "NotesPayable",
            "LoansPayable",
            "SecuredDebt",
        ):
            assert tag in names, f"missing {tag}"

    def test_noncurrent_tags_win_before_incl_current_variants(self):
        """Order guards double counting: a true noncurrent tag must resolve
        before any tag that includes the current portion / ST line."""
        names = self._names()
        assert names.index("LongTermDebtNoncurrent") \
            < names.index("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")
        assert names.index("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities") \
            < names.index("DebtAndCapitalLeaseObligations")

    def test_reit_debt_lines_are_aggregate_fallback_only(self):
        """NotesPayable/LoansPayable/SecuredDebt must be aggregate entries —
        they are face LINES (O/Realty Income), never a pick-first total."""
        by_name = {c.xbrl_concept: c for c in LONG_TERM_DEBT}
        for tag in ("NotesPayable", "LoansPayable", "SecuredDebt"):
            assert by_name[tag].aggregate is True

    def test_non_aggregate_entries_unchanged_for_existing_filers(self):
        """The original pick-first prefix is intact — filers that resolved
        before this change resolve identically (AAPL/SPG/CVX-10-K...)."""
        assert self._names()[:3] == [
            "LongTermDebt", "LongTermDebtNoncurrent",
            "LongTermDebtAndCapitalLeaseObligations",
        ]

    def test_no_fuzzy_matching_on_debt(self):
        """Substring matching is how wrong tags leak — debt entries must not
        opt into Pass-2 fuzzy resolution."""
        for entry in (*LONG_TERM_DEBT, *SHORT_TERM_DEBT):
            assert isinstance(entry, ConceptEntry)
            assert entry.allow_fuzzy is False


class TestDebtResolution:
    """Synthetic per-filer facts frames — mirrors each filer's actual tagging."""

    def test_orcl_fy2026_long_term_notes_payable(self):
        df = _facts([
            ("LongTermNotesPayable", 122_342e6, "2026-05-31"),
            ("NotesPayableCurrent", 7_199e6, "2026-05-31"),
            ("DebtCurrent", 7_199e6, "2026-05-31"),
        ])
        r = _ltd(df)
        assert r.value == pytest.approx(122_342e6)
        assert r.method == "exact"

    def test_cvx_10q_including_current_maturities(self):
        df = _facts([
            ("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
             39_600e6, "2026-03-31"),
            ("ShortTermBorrowings", 5_828e6, "2026-03-31"),
        ])
        r = _ltd(df)
        assert r.value == pytest.approx(39_600e6)

    def test_met_prefers_lt_incl_current_over_combined_total(self):
        """MET's 10-K tags BOTH the LT-incl-current line (14,467) and the
        combined ST+LT total (14,822). Picking the total would double-count
        ShortTermBorrowings (355) in total_debt."""
        df = _facts([
            ("DebtAndCapitalLeaseObligations", 14_822e6, "2025-12-31"),
            ("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
             14_467e6, "2025-12-31"),
            ("ShortTermBorrowings", 355e6, "2025-12-31"),
        ])
        r = _ltd(df)
        assert r.value == pytest.approx(14_467e6)

    def test_reit_aggregates_sum_face_debt_lines(self):
        """O (Realty Income): no LongTermDebt-family tag at all — sum the
        notes + term-loan + mortgage lines."""
        df = _facts([
            ("NotesPayable", 25_031_947_000, "2025-12-31"),
            ("LoansPayable", 1_701_615_000, "2025-12-31"),
            ("SecuredDebt", 37_761_000, "2025-12-31"),
            ("CommercialPaper", 516_800_000, "2025-12-31"),
        ])
        r = _ltd(df)
        assert r.method == "aggregate"
        assert r.value == pytest.approx(26_771_323_000)

    def test_aggregates_never_fire_when_total_tag_exists(self):
        """A filer tagging LongTermDebt AND footnote NotesPayable must serve
        the total tag only — aggregates are strictly a fallback."""
        df = _facts([
            ("LongTermDebt", 90_678e6, "2025-09-27"),
            ("NotesPayable", 5_000e6, "2025-09-27"),
        ])
        r = _ltd(df)
        assert r.value == pytest.approx(90_678e6)
        assert r.method == "exact"


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 2 — Polygon ticker normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeTicker:
    def test_class_shares_dash_to_dot(self):
        assert normalize_ticker("BRK-B") == "BRK.B"
        assert normalize_ticker("BF-B") == "BF.B"
        assert normalize_ticker("brk-b") == "BRK.B"

    def test_plain_tickers_untouched(self):
        assert normalize_ticker("AAPL") == "AAPL"
        assert normalize_ticker(" msft ") == "MSFT"

    def test_index_symbols_untouched(self):
        assert normalize_ticker("I:SPX") == "I:SPX"

    def test_empty_safe(self):
        assert normalize_ticker("") == ""


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 3 — cross-check period alignment + SEC EPS
# ═══════════════════════════════════════════════════════════════════════════

_POLYGON_ROWS = [
    {  # newest row is TTM — the OLD code compared SEC FY against this
        "timeframe": "ttm", "fiscal_period": "TTM", "fiscal_year": "",
        "end_date": "2026-03-28",
        "financials": {
            "income_statement": {
                "revenues": {"value": 451_442e6},
                "net_income_loss": {"value": 122_575e6},
                "basic_earnings_per_share": {"value": 8.29},
                "diluted_earnings_per_share": {"value": 8.26},
            },
            "balance_sheet": {"assets": {"value": 371_082e6}},
        },
    },
    {
        "timeframe": "quarterly", "fiscal_period": "Q2", "fiscal_year": "2026",
        "end_date": "2026-03-28",
        "financials": {"income_statement": {"revenues": {"value": 111_184e6}}},
    },
    {  # the aligned annual row (same period end as the SEC 10-K)
        "timeframe": "annual", "fiscal_period": "FY", "fiscal_year": "2025",
        "end_date": "2025-09-27",
        "financials": {
            "income_statement": {
                "revenues": {"value": 416_161e6},
                "net_income_loss": {"value": 112_010e6},
                "basic_earnings_per_share": {"value": 7.49},
                "diluted_earnings_per_share": {"value": 7.46},
            },
            "balance_sheet": {"assets": {"value": 359_241e6}},
        },
    },
]

_SEC_METRICS = {
    "revenue": 416_161e6,
    "net_income": 112_010e6,
    "total_assets": 359_241e6,
    "eps_basic": 7.49,
    "eps_diluted": 7.46,
}


@pytest.fixture
def stub_polygon(monkeypatch):
    monkeypatch.setattr(polygon_client, "get_financials",
                        lambda ticker, limit=4: list(_POLYGON_ROWS))


class TestPickAlignedReport:
    def test_picks_annual_row_matching_sec_period_end(self):
        report, aligned = _pick_aligned_report(_POLYGON_ROWS, "2025-09-27", "annual")
        assert aligned is True
        assert report["timeframe"] == "annual"
        assert report["end_date"] == "2025-09-27"

    def test_alignment_window_tolerates_small_end_date_drift(self):
        report, aligned = _pick_aligned_report(_POLYGON_ROWS, "2025-09-30", "annual")
        assert aligned is True and report["end_date"] == "2025-09-27"

    def test_unaligned_falls_back_to_newest_annual_not_ttm(self):
        report, aligned = _pick_aligned_report(_POLYGON_ROWS, "2020-01-01", "annual")
        assert aligned is False
        assert report["timeframe"] == "annual"  # never the TTM row

    def test_no_period_end_prefers_requested_timeframe(self):
        report, aligned = _pick_aligned_report(_POLYGON_ROWS, None, "annual")
        assert report["timeframe"] == "annual"


class TestCrossCheckAligned:
    def test_non_december_fye_no_false_mismatch(self, stub_polygon):
        """AAPL-shaped: SEC FY2025 must compare against Polygon FY2025, not
        the TTM row (which produced 7.8%/8.6% false revenue/NI mismatches)."""
        out = cross_check("AAPL", _SEC_METRICS,
                          sec_period_end="2025-09-27", sec_fiscal_year=2025)
        assert out["revenue"]["match"] is True
        assert out["revenue"]["diff_pct"] == 0.0
        assert out["net_income"]["match"] is True
        assert out["total_assets"]["match"] is True

    def test_sec_eps_populated_from_diluted(self, stub_polygon):
        out = cross_check("AAPL", _SEC_METRICS,
                          sec_period_end="2025-09-27", sec_fiscal_year=2025)
        assert out["eps"]["sec"] == pytest.approx(7.46)
        assert out["eps"]["polygon"] == pytest.approx(7.46)  # diluted vs diluted
        assert out["eps"]["match"] is True

    def test_sec_eps_falls_back_to_basic(self, stub_polygon):
        metrics = {k: v for k, v in _SEC_METRICS.items() if k != "eps_diluted"}
        out = cross_check("AAPL", metrics,
                          sec_period_end="2025-09-27", sec_fiscal_year=2025)
        assert out["eps"]["sec"] == pytest.approx(7.49)
        assert out["eps"]["polygon"] == pytest.approx(7.49)  # basic vs basic

    def test_basis_names_compared_periods(self, stub_polygon):
        out = cross_check("AAPL", _SEC_METRICS,
                          sec_period_end="2025-09-27", sec_fiscal_year=2025)
        assert out["basis"] == "annual_fy2025_vs_polygon_annual_2025-09-27"

    def test_unaligned_basis_is_labelled(self, stub_polygon):
        out = cross_check("AAPL", _SEC_METRICS,
                          sec_period_end="2020-01-01", sec_fiscal_year=2019)
        assert out["basis"].endswith("_unaligned")

    def test_response_keys_additive(self, stub_polygon):
        """Existing consumers key into the four metric entries — those shapes
        must be exactly as before; `basis` is the only new key."""
        out = cross_check("AAPL", _SEC_METRICS,
                          sec_period_end="2025-09-27", sec_fiscal_year=2025)
        assert set(out) == {"revenue", "net_income", "total_assets", "eps", "basis"}
        for metric in ("revenue", "net_income", "total_assets", "eps"):
            assert set(out[metric]) == {"sec", "polygon", "diff_pct", "match"}

    def test_legacy_call_without_period_still_works(self, stub_polygon):
        """The enrich path calls cross_check(ticker, metrics) with no period —
        it must keep working (and now compares the newest ANNUAL row)."""
        out = cross_check("AAPL", _SEC_METRICS)
        assert out["revenue"]["sec"] == pytest.approx(416_161e6)
        assert out["revenue"]["polygon"] == pytest.approx(416_161e6)
