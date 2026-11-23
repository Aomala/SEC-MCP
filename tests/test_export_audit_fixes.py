"""Regressions for the Nov 2026 Fineas Top-100 export audit (offline, synthetic).

Each test pins one defect the audit found against SEC XBRL:
  - diluted weighted shares were a copy of basic (590 rows, 101 tickers)
  - quarterly EPS re-derived from period-end shares instead of read (190/879)
  - decumulation paired quarters by fiscal-year LABEL (WDAY +21% revenue)
  - subtracting filings on different bases after a spin-off (WDC −$1.47B
    revenue, HON Q4 revenue $6.86B / gross profit −$7.9B)
  - per-share history not split-adjusted (NVDA, ORLY, CTAS …)
  - a Q4-only GrossProfit fact served as the annual value (ABBV, HON)
  - tagged StockholdersEquity overwritten with Assets − Liabilities (BX ~2.5×)
  - EBITDA built from net income (GOOGL Q2'26 $146B on $40.8B op income)
  - segment pct always summed to 100% (AMD covered 42% of revenue)
  - TTM never carried tax/current-asset lines (effectiveTaxRate, currentRatio null)
  - MNST market cap halved by a split-lagging share count
  - XOM frozen at Q1 2026 by a legacy-CIK pin
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from sec_mcp import financials
from sec_mcp.financials import (
    _build_ebitda,
    _decumulate_quarters_inmemory,
    _filter_by_period_type,
    _split_adjust,
    _synthesize_q4,
    _try_repair_balance_sheet,
    sec_to_fmp_statements,
)


def _row(label, report_date, *, is_ytd, fy=2025, metrics_3m=None, basis=None, **metrics):
    r = {
        "fiscal_year": fy,
        "quarter_label": label,
        "is_ytd": is_ytd,
        "period_type": "quarterly",
        "metrics": dict(metrics),
        "filing_info": {"report_date": report_date, "accession_number": f"acc-{report_date}"},
        "ticker_or_cik": "TEST",
    }
    if metrics_3m:
        r["metrics_3m"] = dict(metrics_3m)
    if basis is not None:
        r["basis_check"] = {"revenue_prior_year": basis}
    return r


# ─── shares / statements ─────────────────────────────────────────────────────

class TestStatementShares:
    def _res(self, **m):
        return {"metrics": m, "filing_info": {"report_date": "2026-09-26"},
                "period_type": "quarterly", "quarter_label": "Q4 (Synthesized)"}

    def test_diluted_is_not_a_copy_of_basic(self):
        inc = sec_to_fmp_statements(self._res(
            weighted_shares_basic=14_532e6, weighted_shares_diluted=14_602e6,
            shares_outstanding=14_480e6))["income"]
        assert inc["weightedAverageShsOut"] == 14_532e6
        assert inc["weightedAverageShsOutDil"] == 14_602e6
        assert inc["sharesOutstanding"] == 14_480e6

    def test_fallbacks_when_filer_tags_no_weighted_counts(self):
        inc = sec_to_fmp_statements(self._res(shares_outstanding=100.0))["income"]
        assert inc["weightedAverageShsOut"] == inc["weightedAverageShsOutDil"] == 100.0
        inc = sec_to_fmp_statements(self._res(weighted_shares_basic=90.0,
                                              shares_outstanding=100.0))["income"]
        assert inc["weightedAverageShsOutDil"] == 90.0  # no dilution data → basic

    def test_ga_is_not_a_copy_of_sga(self):
        inc = sec_to_fmp_statements(self._res(sga_expense=7_459e6))["income"]
        assert inc["sellingGeneralAndAdministrativeExpenses"] == 7_459e6
        assert inc["generalAndAdministrativeExpenses"] is None
        inc = sec_to_fmp_statements(self._res(sga_expense=462e6, ga_expense=256e6))["income"]
        assert inc["generalAndAdministrativeExpenses"] == 256e6


# ─── decumulation ────────────────────────────────────────────────────────────

class TestDecumulationPairing:
    def test_pairs_by_report_date_not_fiscal_year_label(self):
        """WDAY: FY26 and FY25 10-Qs both labelled fy=2025. Q3 FY26 must
        subtract Q2 FY26 (one quarter earlier), not Q2 FY25."""
        rows = [
            _row("Q3 (9-month YTD)", "2025-10-31", is_ytd=True, revenue=7_020.0),
            _row("Q2 (6-month YTD)", "2025-07-31", is_ytd=True, revenue=4_588.0),
            _row("Q1 (Standalone)", "2025-04-30", is_ytd=False, revenue=2_240.0),
            _row("Q2 (6-month YTD)", "2024-07-31", is_ytd=True, revenue=4_075.0),
            _row("Q1 (Standalone)", "2024-04-30", is_ytd=False, revenue=1_990.0),
        ]
        out = _decumulate_quarters_inmemory(rows)
        assert out[0]["metrics"]["revenue"] == pytest.approx(2_432.0)  # not 2,945
        assert out[1]["metrics"]["revenue"] == pytest.approx(2_348.0)

    def test_filed_three_month_values_beat_subtraction(self):
        """WDC Q3 FY25: 9M on the post-spin basis minus H1 on the pre-spin
        basis gave revenue −$1.47B. The 10-Q's own 3-month figure is $2.29B."""
        rows = [
            _row("Q3 (9-month YTD)", "2025-03-28", is_ytd=True,
                 revenue=6_915.0, eps_diluted=4.6,
                 metrics_3m={"revenue": 2_294.0, "eps_diluted": 1.36,
                             "weighted_shares_diluted": 351e6}),
            _row("Q2 (6-month YTD)", "2024-12-27", is_ytd=True, revenue=8_380.0, eps_diluted=3.06),
            _row("Q1 (Standalone)", "2024-09-27", is_ytd=False, revenue=4_095.0),
        ]
        q3 = _decumulate_quarters_inmemory(rows)[0]
        assert q3["metrics"]["revenue"] == 2_294.0
        assert q3["metrics"]["eps_diluted"] == 1.36
        assert q3["metrics"]["weighted_shares_diluted"] == 351e6
        assert q3["eps_source"] == "reported"

    def test_negative_revenue_is_nulled_not_served(self):
        rows = [
            _row("Q3 (9-month YTD)", "2025-03-28", is_ytd=True, revenue=6_915.0, net_income=1_600.0),
            _row("Q2 (6-month YTD)", "2024-12-27", is_ytd=True, revenue=8_380.0, net_income=1_080.0),
            _row("Q1 (Standalone)", "2024-09-27", is_ytd=False, revenue=4_095.0),
        ]
        q3 = _decumulate_quarters_inmemory(rows)[0]
        assert q3["metrics"]["revenue"] is None
        assert q3["metrics"]["net_income"] is None  # same broken basis
        assert any(v["rule"] == "impossible_quarter_value" for v in q3["validation"])

    def test_basis_break_blocks_cash_flow_subtraction(self):
        """Q3 10-Q restated its prior-year comparative (spin-off), the Q2 10-Q
        didn't: differencing their YTD cash flows mixes two companies."""
        rows = [
            _row("Q3 (9-month YTD)", "2025-09-30", is_ytd=True, revenue=270.0,
                 operating_cash_flow=90.0, basis=240.0),   # PY 9M restated 300 → 240
            _row("Q2 (6-month YTD)", "2025-06-30", is_ytd=True, revenue=200.0,
                 operating_cash_flow=70.0, basis=200.0),   # PY 6M as originally filed
            _row("Q1 (Standalone)", "2025-03-31", is_ytd=False, revenue=100.0),
            _row("Q3 (9-month YTD)", "2024-09-30", is_ytd=True, revenue=300.0),
            _row("Q2 (6-month YTD)", "2024-06-30", is_ytd=True, revenue=200.0),
            _row("Q1 (Standalone)", "2024-03-31", is_ytd=False, revenue=100.0),
        ]
        q3 = _decumulate_quarters_inmemory(rows)[0]
        assert q3["metrics"]["operating_cash_flow"] is None
        assert any(v["rule"] == "basis_break" for v in q3["validation"])

    def test_period_comes_from_the_revenue_fact_not_filing_labels(self):
        """BX Q2'24 10-Q carried fp=Q1 (6-month YTD served as a standalone
        Q1); BX Q3'26 was dated 2026-10-31 by a subsequent-event fact."""
        q2 = _row("Q1 (Standalone)", "2024-06-30", is_ytd=False, revenue=6_484.0)
        q2["basis_check"] = {"period_end": "2024-06-30", "duration_days": 182}
        q3 = _row("Q3 (9-month YTD)", "2024-10-31", is_ytd=True, revenue=10_147.0)
        q3["basis_check"] = {"period_end": "2024-09-30", "duration_days": 273}
        q1 = _row("Q1 (Standalone)", "2024-03-31", is_ytd=False, revenue=3_688.0)
        out = _decumulate_quarters_inmemory([q3, q2, q1])
        assert out[1]["metrics"]["revenue"] == pytest.approx(2_796.0)
        assert out[0]["filing_info"]["report_date"] == "2024-09-30"
        assert out[0]["metrics"]["revenue"] == pytest.approx(3_663.0)
        assert out[0]["quality"] == "standalone_decumulated"

    def test_same_basis_still_decumulates(self):
        rows = [
            _row("Q3 (9-month YTD)", "2025-09-30", is_ytd=True, revenue=330.0,
                 operating_cash_flow=90.0, basis=300.0),
            _row("Q2 (6-month YTD)", "2025-06-30", is_ytd=True, revenue=220.0,
                 operating_cash_flow=70.0, basis=200.0),
            _row("Q1 (Standalone)", "2025-03-31", is_ytd=False, revenue=100.0),
            _row("Q3 (9-month YTD)", "2024-09-30", is_ytd=True, revenue=300.0),
            _row("Q2 (6-month YTD)", "2024-06-30", is_ytd=True, revenue=200.0),
            _row("Q1 (Standalone)", "2024-03-31", is_ytd=False, revenue=100.0),
        ]
        q3 = _decumulate_quarters_inmemory(rows)[0]
        assert q3["metrics"]["operating_cash_flow"] == pytest.approx(20.0)


# ─── Q4 synthesis basis break ────────────────────────────────────────────────

@pytest.fixture
def annual_10k(monkeypatch):
    state = {}

    def fake_client():
        return SimpleNamespace(get_periodic_filings_smart=lambda *a, **k: [
            SimpleNamespace(accession_number="acc-10K", report_date=state["rd"],
                            filing_date=state["rd"], form_type="10-K")])

    monkeypatch.setattr(financials, "get_sec_client", fake_client)
    monkeypatch.setattr(financials, "extract_financials", lambda *a, **k: state["annual"])

    def configure(rd, basis=None, **metrics):
        state["rd"] = rd
        state["annual"] = {"ticker_or_cik": "TEST", "period_type": "annual",
                           "metrics": dict(metrics),
                           "filing_info": {"report_date": rd, "accession_number": "acc-10K"},
                           **({"basis_check": {"revenue_prior_year": basis}} if basis else {})}
    return configure


def _two_years():
    return [
        _row("Q3 (9-month YTD)", "2025-09-30", is_ytd=True, revenue=30.6, basis=29.0),
        _row("Q2 (6-month YTD)", "2025-06-30", is_ytd=True, revenue=20.2),
        _row("Q1 (Standalone)", "2025-03-31", is_ytd=False, revenue=9.8),
        _row("Q4 (Synthesized)", "2024-12-31", is_ytd=False, revenue=10.1),
        _row("Q3 (9-month YTD)", "2024-09-30", is_ytd=True, revenue=29.0),
        _row("Q2 (6-month YTD)", "2024-06-30", is_ytd=True, revenue=19.3),
        _row("Q1 (Standalone)", "2024-03-31", is_ytd=False, revenue=9.4),
    ]


class TestQ4BasisBreak:
    def test_restated_10k_vs_unrestated_q3_refuses_q4_flows(self, annual_10k):
        rows = _decumulate_quarters_inmemory(_two_years())
        rows[3]["quality"] = "q4_synthesized"
        # HON FY2025 10-K: FY24 restated to 33.0 (ex-Solstice) vs the 39.1 the
        # prior year's quarters add up to; its Q3 10-Q still says 29.0 (orig).
        annual_10k("2025-12-31", basis=33.0, revenue=37.4, total_assets=75.0)
        out = _synthesize_q4("TEST", rows)
        q4 = [r for r in out if r["filing_info"]["report_date"] == "2025-12-31"][0]
        assert q4["metrics"]["revenue"] is None          # not 37.4 − 30.6 = 6.8
        assert q4["metrics"]["total_assets"] == 75.0     # balance sheet kept
        assert any(v["rule"] == "basis_break" for v in q4["validation"])

    def test_q4_uses_q3_nine_month_ytd_not_sum_of_filed_quarters(self, annual_10k):
        """JNJ 2023: Q1/Q2 as filed include Kenvue, Q3's 3-month figure and
        its 9M YTD exclude it. ΣQ1-3 ≠ 9M, so Q4 must be FY − 9M (21.4),
        not FY − Σ (13.5)."""
        rows = [
            _row("Q3 (9-month YTD)", "2023-10-01", is_ytd=True, revenue=63.8,
                 metrics_3m={"revenue": 21.4}),
            _row("Q2 (6-month YTD)", "2023-07-02", is_ytd=True, revenue=50.2),
            _row("Q1 (Standalone)", "2023-04-02", is_ytd=False, revenue=24.7),
        ]
        rows = _decumulate_quarters_inmemory(rows)
        annual_10k("2023-12-31", revenue=85.2)
        q4 = [r for r in _synthesize_q4("TEST", rows) if r.get("quality") == "q4_synthesized"][0]
        assert q4["metrics"]["revenue"] == pytest.approx(21.4)

    def test_consistent_bases_synthesize_normally(self, annual_10k):
        rows = _decumulate_quarters_inmemory(_two_years())
        rows[3]["quality"] = "q4_synthesized"
        annual_10k("2025-12-31", basis=39.1, revenue=40.8)
        out = _synthesize_q4("TEST", rows)
        q4 = [r for r in out if r["filing_info"]["report_date"] == "2025-12-31"][0]
        assert q4["metrics"]["revenue"] == pytest.approx(10.2)


# ─── splits ──────────────────────────────────────────────────────────────────

class TestSplitAdjust:
    def test_older_rows_restated_to_newest_share_basis(self):
        new = {"metrics": {"eps_diluted": 2.94, "weighted_shares_diluted": 24_804e6},
               "filing_info": {"report_date": "2025-01-26"}}
        old = {"metrics": {"eps_diluted": 11.93, "weighted_shares_diluted": 2_494e6},
               "filing_info": {"report_date": "2024-01-28"}}
        _split_adjust([new, old])
        assert old["metrics"]["eps_diluted"] == pytest.approx(1.193)
        assert old["metrics"]["weighted_shares_diluted"] == pytest.approx(24_940e6)
        assert old["split_factor"] == 10
        assert new["metrics"]["eps_diluted"] == 2.94

    def test_ordinary_issuance_is_not_a_split(self):
        new = {"metrics": {"eps_diluted": 3.0, "weighted_shares_diluted": 291e6},
               "filing_info": {"report_date": "2024-12-31"}}
        old = {"metrics": {"eps_diluted": 5.0, "weighted_shares_diluted": 179e6},
               "filing_info": {"report_date": "2023-12-31"}}  # FANG +63% (merger)
        _split_adjust([new, old])
        assert old["metrics"]["eps_diluted"] == 5.0


# ─── extraction-time pieces ──────────────────────────────────────────────────

def test_annual_filter_never_takes_a_single_quarter():
    """ABBV tags GrossProfit only in the 10-K quarterly note (Q4, 92 days)."""
    df = pd.DataFrame({"concept": ["GrossProfit"] * 2, "value": [12_066e6, 10_706e6],
                       "start": ["2025-10-01", "2024-10-01"], "end": ["2025-12-31", "2024-12-31"]})
    assert _filter_by_period_type(df, "end", "annual").empty


def test_tagged_equity_is_not_overwritten_with_assets_minus_liabilities():
    """BX: SE 8.7B tagged; A − L = 21.9B includes NCI and redeemable NCI."""
    m = {"total_assets": 46.0e9, "total_liabilities": 24.1e9, "stockholders_equity": 8.67e9}
    assert _try_repair_balance_sheet(m, 46.0e9, 24.1e9, 8.67e9,
                                     {"stockholders_equity": 0.99, "total_liabilities": 0.99}) is None
    weak = _try_repair_balance_sheet(m, 46.0e9, 24.1e9, 8.67e9,
                                     {"stockholders_equity": 0.6, "total_liabilities": 0.99})
    assert weak == {"stockholders_equity": pytest.approx(21.9e9)}


class TestEbitda:
    def test_operating_income_plus_da(self):
        # GOOGL Q2'26: $40.8B op income, $112B NI (equity-stake marks)
        v, src, _ = _build_ebitda(40.77e9, 112.19e9, None, 26.56e9, 7.47e9, "standard")
        assert v == pytest.approx(48.24e9)
        assert src == "Operating Income + D&A"

    def test_banks_keep_bottom_up_build(self):
        v, src, _ = _build_ebitda(20e9, 15e9, 2e9, 4e9, 1e9, "bank")
        assert v == pytest.approx(22e9)
        assert src.startswith("Net Income")

    def test_signed_tax_benefit(self):
        v, _, _ = _build_ebitda(None, 7_371.0, 172.0, -5_139.0, 413.0, "standard")
        assert v == pytest.approx(7_371 + 172 - 5_139 + 413)


# ─── segments ────────────────────────────────────────────────────────────────

def test_segment_pct_measured_against_consolidated_revenue():
    from sec_mcp.graph.segments import _apply_coverage
    rows = [{"segment": "Data Center", "value": 9.0}, {"segment": "Client", "value": 5.55}]
    meta = _apply_coverage(rows, 34.64)  # AMD: two of four segments
    assert meta["sumCheck"] == "partial"
    assert meta["coverage"] == pytest.approx(0.42, abs=0.001)
    assert sum(r["pct"] for r in rows) == pytest.approx(42.0, abs=0.2)
    over = [{"segment": "A", "value": 15.0}, {"segment": "B", "value": 7.9}]
    assert _apply_coverage(over, 14.09)["sumCheck"] == "exceeds_total"


# ─── TTM ─────────────────────────────────────────────────────────────────────

def test_ttm_carries_tax_and_current_lines(monkeypatch):
    from sec_mcp.core import ttm

    def q(end, eps):
        return {"endDate": end, "fiscalYear": end[:4], "fiscalPeriod": "Q", "quality": "standalone_decumulated",
                "epsSource": "reported", "formType": "10-Q",
                "metrics": {"revenue": 100.0, "netIncome": 20.0, "epsDiluted": eps, "epsBasic": eps + 0.01,
                            "incomeTaxExpense": 5.0, "incomeBeforeTax": 25.0,
                            "totalCurrentAssets": 300.0, "totalCurrentLiabilities": 150.0,
                            "sharesDiluted": 10.0, "sharesOutstanding": 9.5}}
    periods = [q(e, 2.0 + i / 10) for i, e in enumerate(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"])]
    block = ttm.build_ttm_metrics("TEST", periods=periods)
    m = block["metrics"]
    assert m["income_tax_expense"] == 20.0 and m["income_before_tax"] == 100.0
    assert m["current_assets"] == 300.0 and m["current_liabilities"] == 150.0
    assert m["eps_diluted"] == pytest.approx(8.6)       # Σ filed EPS, not NI/shares
    assert m["eps_basic"] == pytest.approx(8.64)


# ─── sec_client ──────────────────────────────────────────────────────────────

def _client_with(facts):
    from sec_mcp.sec_client import SECClient
    c = SECClient.__new__(SECClient)
    c.get_company_facts = lambda *_a, **_k: facts
    return c


def test_cover_shares_sum_classes_and_skip_20f():
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"val": 979_525_882, "end": "2026-07-31", "accn": "a1", "filed": "2026-08-07", "form": "10-Q"},
        {"val": 1_957_511_922, "end": "2026-10-30", "accn": "a2", "filed": "2026-11-06", "form": "10-Q"},
        {"val": 5, "end": "2026-10-30", "accn": "a2", "filed": "2026-11-06", "form": "10-Q"},
        {"val": 9_999_999_999, "end": "2026-11-15", "accn": "a3", "filed": "2026-11-20", "form": "20-F"},
    ]}}}}}
    got = _client_with(facts).get_latest_cover_shares("MNST")
    assert got["shares"] == 1_957_511_927 and got["as_of"] == "2026-10-30"


def test_predecessor_facts_merge():
    from sec_mcp.sec_client import _merge_company_facts
    succ = {"cik": 2115436, "facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"accn": "new", "val": 1}]}}}}}
    pred = {"cik": 34088, "facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"accn": "old", "val": 2}]}},
                                                "Assets": {"units": {"USD": [{"accn": "old", "val": 3}]}}}}}
    m = _merge_company_facts(succ, pred)
    assert [x["accn"] for x in m["facts"]["us-gaap"]["Revenues"]["units"]["USD"]] == ["new", "old"]
    assert "Assets" in m["facts"]["us-gaap"] and m["cik"] == 2115436


def test_ga_expense_is_a_flow_like_sga():
    """historyVersion 2 left ga_expense out of the decumulated flows, so AAPL's
    'quarterly' G&A was its 6M/9M/FY cumulative figure (4.37B, 6.68B, 8.64B)."""
    rows = [
        _row("Q3 (9-month YTD)", "2026-06-27", is_ytd=True, ga_expense=6_683.0,
             metrics_3m={"ga_expense": 2_312.0}),
        _row("Q2 (6-month YTD)", "2026-03-28", is_ytd=True, ga_expense=4_371.0),
        _row("Q1 (Standalone)", "2025-12-27", is_ytd=False, ga_expense=2_095.0),
    ]
    out = _decumulate_quarters_inmemory(rows)
    assert out[0]["metrics"]["ga_expense"] == 2_312.0      # filed 3-month figure
    assert out[1]["metrics"]["ga_expense"] == pytest.approx(2_276.0)  # 4,371 − 2,095


class _FakeQuery:
    """Just enough of supabase-py's builder chain for cached_age_days."""

    def __init__(self, rows, seen):
        self._rows, self._seen = rows, seen

    def select(self, cols):
        self._seen.append(cols)
        return self

    def eq(self, *_a):
        return self

    gte = limit = eq

    def execute(self):
        return type("R", (), {"data": self._rows})()


def _fake_client(rows, seen):
    return type("C", (), {"table": lambda self, _name: _FakeQuery(rows, seen)})()


def test_batch_age_check_treats_outdated_rows_as_absent(monkeypatch):
    # The sweep skips rows younger than --refresh-older-than; a row rebuilt
    # yesterday by pre-fix code must not hide behind its age.
    from datetime import datetime, timezone

    from sec_mcp import supabase_cache as sc

    now = datetime.now(timezone.utc).isoformat()
    seen: list[str] = []
    for dv, expect_fresh in ((None, False), (sc.DATA_VERSION - 1, False),
                             ("junk", False), (sc.DATA_VERSION, True)):
        monkeypatch.setattr(sc, "_get_client",
                            lambda dv=dv: _fake_client([{"created_at": now, "dv": dv}], seen))
        age = sc.cached_age_days("AAPL", "sec_income_history_v2", "quarter")
        assert (age is not None) is expect_fresh, dv
    assert "_dataVersion" in seen[-1]

    # Unversioned types never read the blob and keep age-only semantics.
    monkeypatch.setattr(sc, "_get_client", lambda: _fake_client([{"created_at": now}], seen))
    assert sc.cached_age_days("AAPL", "geo_segments", "annual") is not None
    assert seen[-1] == "created_at"
