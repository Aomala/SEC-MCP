"""Regression tests: decumulation / Q4-synthesis honesty for per-share columns.

Guards the July 2026 extraction fixes:
  - a standalone_decumulated row must never carry the YTD eps it was
    extracted with (JPM Q3'25 served 15.41 = 9-month YTD; standalone ~5.07)
  - a q4_synthesized row must never carry the FULL-YEAR eps
    (META Q4'25 served 23.49 = FY25 annual; standalone ~8.87)
  - eps priority: NI/shares → YTD-neighbour diff → None (never cumulative)
  - the YTD-diff is sign-checked against the row's own NI (stock splits:
    WMT 3:1 Feb'24 made FY-minus-Σquarters = −1.8 on a +5.5B-NI quarter)
  - flow columns whose YTD neighbour is missing are nulled, not served YTD
  - Q4 synthesis joins the 10-K by report-date window, not fiscal-year label
    (Jan/Feb-FYE filers: HD/CRM never got a Q4 and lost TTM eligibility)

All fixtures are synthetic — no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sec_mcp import financials
from sec_mcp.financials import (
    _decumulate_quarters_inmemory,
    _recompute_derived_quarter_metrics,
    _synthesize_q4,
)


def _row(fy, label, report_date, *, is_ytd, **metrics):
    return {
        "fiscal_year": fy,
        "quarter_label": label,
        "is_ytd": is_ytd,
        "period_type": "quarterly",
        "metrics": dict(metrics),
        "filing_info": {"report_date": report_date, "accession_number": f"acc-{report_date}"},
        "ticker_or_cik": "TEST",
    }


# ─── _recompute_derived_quarter_metrics ──────────────────────────────────────

class TestRecomputePerShare:
    def test_shares_available_uses_ni_over_shares(self):
        cm = {"net_income": 1000.0, "shares_outstanding": 400.0,
              "eps_basic": 7.5, "eps_diluted": 7.4}  # stale YTD values
        _recompute_derived_quarter_metrics(cm)
        assert cm["eps_basic"] == 2.5
        assert cm["eps_diluted"] == 2.5

    def test_no_shares_falls_back_to_ytd_diff(self):
        cm = {"net_income": 1000.0, "eps_basic": 7.5, "eps_diluted": 7.4}
        _recompute_derived_quarter_metrics(
            cm, prior_eps={"eps_basic": 5.0, "eps_diluted": 4.95})
        assert cm["eps_basic"] == 2.5
        assert cm["eps_diluted"] == pytest.approx(2.45)

    def test_no_shares_no_neighbour_nulls_eps(self):
        """The YTD eps must NOT survive when there is nothing to rebuild from."""
        cm = {"net_income": 1000.0, "eps_basic": 7.5, "eps_diluted": 7.4}
        _recompute_derived_quarter_metrics(cm)
        assert cm["eps_basic"] is None
        assert cm["eps_diluted"] is None

    def test_split_boundary_sign_guard_nulls_eps(self):
        """WMT 3:1 split: FY eps split-adjusted, quarter eps not → diff goes
        negative on a positive-NI quarter → serve None, not −1.8."""
        cm = {"net_income": 5_494e6, "eps_basic": 1.91, "eps_diluted": 1.91}
        _recompute_derived_quarter_metrics(
            cm, prior_eps={"eps_basic": 3.71, "eps_diluted": 3.71})
        assert cm["eps_basic"] is None
        assert cm["eps_diluted"] is None

    def test_negative_ni_negative_diff_is_kept(self):
        cm = {"net_income": -500.0, "eps_basic": -1.0, "eps_diluted": -1.0}
        _recompute_derived_quarter_metrics(
            cm, prior_eps={"eps_basic": 0.25, "eps_diluted": 0.25})
        assert cm["eps_basic"] == -1.25
        assert cm["eps_diluted"] == -1.25

    def test_ebitda_rebuilt_from_row_components_signed_tax(self):
        """QCOM Q2 FY26: −5.1B tax benefit — abs(tax) had EBITDA at 113% of
        revenue. Rebuild must use the SIGNED tax expense."""
        cm = {"net_income": 7_371.0, "interest_expense": 172.0,
              "income_tax_expense": -5_139.0, "depreciation_amortization": 413.0,
              "ebitda": 12_009.0}  # garbage from cumulative subtraction
        _recompute_derived_quarter_metrics(cm)
        assert cm["ebitda"] == pytest.approx(7_371 + 172 - 5_139 + 413)

    def test_ebitda_without_components_is_nulled(self):
        """A subtracted cumulative EBITDA with no components to verify against
        must not be served on a standalone row."""
        cm = {"net_income": 1000.0, "ebitda": 999_999.0}
        _recompute_derived_quarter_metrics(cm)
        assert cm["ebitda"] is None

    def test_ebitda_oi_fallback(self):
        cm = {"operating_income": 2_310.0, "depreciation_amortization": 413.0,
              "ebitda": 12_009.0}
        _recompute_derived_quarter_metrics(cm)
        assert cm["ebitda"] == pytest.approx(2_723.0)


# ─── _decumulate_quarters_inmemory ───────────────────────────────────────────

def _three_quarters():
    q1 = _row(2025, "Q1 (Standalone)", "2025-03-31", is_ytd=False,
              revenue=100.0, net_income=10.0, eps_basic=1.0, eps_diluted=0.98)
    q2 = _row(2025, "Q2 (6-month YTD)", "2025-06-30", is_ytd=True,
              revenue=220.0, net_income=25.0, eps_basic=2.5, eps_diluted=2.45)
    q3 = _row(2025, "Q3 (9-month YTD)", "2025-09-30", is_ytd=True,
              revenue=360.0, net_income=45.0, eps_basic=4.5, eps_diluted=4.41)
    return [q3, q2, q1]  # newest-first, as get_fmp_shaped_history sorts


class TestDecumulation:
    def test_flow_metrics_decumulated(self):
        rows = _decumulate_quarters_inmemory(_three_quarters())
        q3, q2, q1 = rows
        assert q1["quality"] == "standalone"
        assert q2["quality"] == "standalone_decumulated"
        assert q2["metrics"]["revenue"] == 120.0
        assert q3["metrics"]["revenue"] == 140.0
        assert q3["metrics"]["net_income"] == 20.0

    def test_decumulated_rows_never_keep_ytd_eps(self):
        """The core BUG-1 regression: eps on a standalone_decumulated row must
        be the standalone value (here via YTD diff), never the YTD input."""
        rows = _decumulate_quarters_inmemory(_three_quarters())
        q3, q2, _ = rows
        assert q2["metrics"]["eps_basic"] == pytest.approx(1.5)   # 2.5 − 1.0
        assert q2["metrics"]["eps_diluted"] == pytest.approx(1.47)
        assert q3["metrics"]["eps_basic"] == pytest.approx(2.0)   # 4.5 − 2.5
        assert q3["metrics"]["eps_diluted"] == pytest.approx(1.96)
        for r in (q2, q3):
            assert r["metrics"]["eps_basic"] not in (2.5, 4.5), "YTD eps leaked"

    def test_eps_nulled_when_no_method(self):
        rows = _three_quarters()
        del rows[1]["metrics"]["eps_basic"]  # Q2 filing didn't tag eps
        del rows[1]["metrics"]["eps_diluted"]
        out = _decumulate_quarters_inmemory(rows)
        q3 = out[0]
        assert q3["quality"] == "standalone_decumulated"
        # Q3 diff needs the Q2 YTD eps — gone → None, never the 4.5 YTD value
        assert q3["metrics"]["eps_basic"] is None
        assert q3["metrics"]["eps_diluted"] is None

    def test_missing_neighbour_column_is_nulled_not_served_ytd(self):
        rows = _three_quarters()
        rows[1]["metrics"]["revenue"] = 220.0
        del rows[1]["metrics"]["net_income"]  # Q2 snapshot lacks NI
        out = _decumulate_quarters_inmemory(rows)
        q3 = out[0]
        assert q3["quality"] == "standalone_decumulated"
        assert q3["metrics"]["revenue"] == 140.0        # decumulable → decumulated
        assert q3["metrics"]["net_income"] is None      # not decumulable → nulled

    def test_missing_neighbour_row_is_ytd_fallback(self):
        rows = _three_quarters()[:2]  # Q3, Q2 — no Q1 in hand
        out = _decumulate_quarters_inmemory(rows)
        q2 = out[1]
        assert q2["quality"] == "ytd_fallback"
        assert q2["is_ytd"] is True
        assert q2["metrics"]["revenue"] == 220.0  # stays YTD, honestly labelled


# ─── _synthesize_q4 ──────────────────────────────────────────────────────────

def _fake_annual(report_date, **metrics):
    return {
        "ticker_or_cik": "TEST",
        "fiscal_year": 1999,  # deliberately wrong label — CRM's 10-K disagrees
        "period_type": "annual",
        "metrics": dict(metrics),
        "filing_info": {"report_date": report_date, "accession_number": "acc-10K"},
    }


@pytest.fixture
def patched_annual(monkeypatch):
    """Patch the 10-K listing + extraction that _synthesize_q4 performs."""
    state = {}

    def fake_client():
        return SimpleNamespace(get_periodic_filings_smart=lambda *a, **k: [
            SimpleNamespace(accession_number="acc-10K",
                            report_date=state["report_date"],
                            filing_date=state["report_date"], form_type="10-K"),
        ])

    monkeypatch.setattr(financials, "get_sec_client", fake_client)
    monkeypatch.setattr(financials, "extract_financials",
                        lambda *a, **k: state["annual"])

    def configure(report_date, **metrics):
        state["report_date"] = report_date
        state["annual"] = _fake_annual(report_date, **metrics)

    return configure


class TestQ4Synthesis:
    def test_q4_row_synthesized_and_eps_is_not_fy(self, patched_annual):
        rows = _decumulate_quarters_inmemory(_three_quarters())
        # Jan-FYE calendar: FY ends 2026-01-31, one quarter after Q3 (HD/CRM
        # case) — and the 10-K's own fiscal_year label disagrees on purpose.
        patched_annual("2026-01-31", revenue=500.0, net_income=65.0,
                       eps_basic=6.5, eps_diluted=6.37)
        out = _synthesize_q4("TEST", rows)
        q4 = [r for r in out if r.get("quality") == "q4_synthesized"]
        assert len(q4) == 1, "report-date window join must fire for Jan-FYE"
        q4 = q4[0]
        assert q4["fiscal_year"] == 2025  # the quarters' label, not the 10-K's
        assert q4["metrics"]["revenue"] == pytest.approx(140.0)  # 500 − 360
        assert q4["metrics"]["net_income"] == pytest.approx(20.0)
        # eps = FY − Σ(standalone Q1-3) = FY − Q3-YTD, never the FY value
        assert q4["metrics"]["eps_basic"] == pytest.approx(2.0)   # 6.5 − 4.5
        assert q4["metrics"]["eps_diluted"] == pytest.approx(1.96)
        assert q4["metrics"]["eps_basic"] != 6.5, "FY eps leaked into Q4 row"

    def test_q4_eps_prefers_ni_over_shares(self, patched_annual):
        rows = _decumulate_quarters_inmemory(_three_quarters())
        patched_annual("2026-01-31", revenue=500.0, net_income=65.0,
                       shares_outstanding=10.0, eps_basic=6.5, eps_diluted=6.37)
        out = _synthesize_q4("TEST", rows)
        q4 = [r for r in out if r.get("quality") == "q4_synthesized"][0]
        assert q4["metrics"]["eps_basic"] == pytest.approx(2.0)  # 20 / 10

    def test_no_10k_in_window_no_q4(self, patched_annual):
        rows = _decumulate_quarters_inmemory(_three_quarters())
        # 10-K ends the same day as Q3 — not one quarter after → no join
        patched_annual("2025-09-30", revenue=500.0, net_income=65.0)
        out = _synthesize_q4("TEST", rows)
        assert not [r for r in out if r.get("quality") == "q4_synthesized"]

    def test_mismatched_10k_negative_revenue_skipped(self, patched_annual):
        rows = _decumulate_quarters_inmemory(_three_quarters())
        # Annual revenue below ΣQ1-3 → synthesized Q4 revenue would be negative
        patched_annual("2026-01-31", revenue=300.0, net_income=65.0)
        out = _synthesize_q4("TEST", rows)
        assert not [r for r in out if r.get("quality") == "q4_synthesized"]
