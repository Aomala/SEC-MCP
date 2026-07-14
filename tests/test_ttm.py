"""Tests for the TTM metrics builder (core/ttm.py) — pure, _build_periods mocked."""

import pytest

from sec_mcp.core.ttm import build_ttm_metrics


def _q(end, rev, ni, equity=800.0, assets=2000.0, quality="standalone", **over):
    metrics = {
        "revenue": rev,
        "grossProfit": rev * 0.5,
        "operatingIncome": rev * 0.3,
        "netIncome": ni,
        "ebitda": rev * 0.4,
        "operatingCashFlow": ni * 1.5,
        "capex": -(rev * 0.05),          # FMP negative-sign convention
        "freeCashFlow": ni * 1.2,
        "totalAssets": assets,
        "totalLiabilities": assets - equity,
        "totalEquity": equity,
        "cashAndEquivalents": 150.0,
        "totalDebt": 600.0,
        "sharesOutstanding": 100.0,
        "epsDiluted": ni / 100.0,
    }
    metrics.update(over)
    return {"endDate": end, "fiscalYear": end[:4], "fiscalPeriod": "Q",
            "quality": quality, "formType": "10-Q", "metrics": metrics}


def _eight_quarters():
    # Newest first, equity grows 50/quarter so year-ago balance differs
    return [
        _q("2026-03-31", 250.0, 50.0, equity=800.0, assets=2000.0),
        _q("2025-12-31", 240.0, 48.0, equity=750.0, assets=1950.0),
        _q("2025-09-30", 230.0, 46.0, equity=700.0, assets=1900.0),
        _q("2025-06-30", 220.0, 44.0, equity=650.0, assets=1850.0),
        _q("2025-03-31", 210.0, 42.0, equity=600.0, assets=1800.0),
        _q("2024-12-31", 200.0, 40.0, equity=550.0, assets=1750.0),
        _q("2024-09-30", 190.0, 38.0, equity=500.0, assets=1700.0),
        _q("2024-06-30", 180.0, 36.0, equity=450.0, assets=1650.0),
    ]


@pytest.fixture
def mock_periods(monkeypatch):
    def _install(periods):
        monkeypatch.setattr(
            "sec_mcp.surface.fundamentals._build_periods",
            lambda ticker, period, n: periods,
        )
    return _install


def test_frozen_shape_aliases_and_tax_keys(mock_periods):
    qs = _eight_quarters()
    for q in qs:
        rev = q["metrics"]["revenue"]
        q["metrics"]["incomeTaxExpense"] = rev * 0.05
        q["metrics"]["incomeBeforeTax"] = rev * 0.25
        q["metrics"]["interestExpense"] = rev * 0.01
    mock_periods(qs)
    from sec_mcp.core.ratios import compute_ratios
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    out = build_ttm_metrics("TEST")
    m = out["metrics"]
    # comps-mapper contract keys (annual metrics dict carries these)
    assert m["total_equity"] == m["stockholders_equity"] == 800.0
    assert m["net_debt"] == pytest.approx(600.0 - 150.0)
    # tax/interest flows summed → effective tax + roic on effective basis
    assert m["income_tax_expense"] == pytest.approx(940.0 * 0.05)
    assert m["income_before_tax"] == pytest.approx(940.0 * 0.25)
    r = compute_ratios(m, prior=out["prior"])
    assert r["effective_tax_rate"] == pytest.approx(0.20)
    assert r["roic"] is not None
    assert r["roic_basis"] == "oi_effective_tax"
    q_out = build_latest_quarter_metrics("TEST", periods=qs)
    assert q_out["metrics"]["total_equity"] == 800.0
    assert q_out["metrics"]["net_debt"] == pytest.approx(450.0)


def test_flows_sum_balances_latest(mock_periods):
    mock_periods(_eight_quarters())
    out = build_ttm_metrics("TEST")
    m = out["metrics"]
    assert m["revenue"] == pytest.approx(250 + 240 + 230 + 220)
    assert m["net_income"] == pytest.approx(50 + 48 + 46 + 44)
    assert m["gross_profit"] == pytest.approx(m["revenue"] * 0.5)
    # Balance items from the newest quarter only
    assert m["stockholders_equity"] == 800.0
    assert m["total_assets"] == 2000.0
    assert m["total_debt"] == 600.0
    # capex normalized back to magnitude
    assert m["capital_expenditures"] == pytest.approx(m["revenue"] * 0.05)
    assert out["quality"] == "ttm_sum_4q"
    assert out["as_of"] == "2026-03-31"
    assert len(out["quarters_used"]) == 4


def test_missing_flow_value_gives_honest_none(mock_periods):
    qs = _eight_quarters()
    qs[2]["metrics"]["ebitda"] = None
    mock_periods(qs)
    m = build_ttm_metrics("TEST")["metrics"]
    assert m["ebitda"] is None
    assert m["revenue"] is not None  # other flows unaffected


def test_eps_from_ni_over_latest_shares(mock_periods):
    mock_periods(_eight_quarters())
    m = build_ttm_metrics("TEST")["metrics"]
    assert m["eps_diluted"] == pytest.approx(188.0 / 100.0)


def test_eps_falls_back_to_summed_quarterly_eps(mock_periods):
    qs = _eight_quarters()
    for q in qs:
        q["metrics"]["sharesOutstanding"] = None
    mock_periods(qs)
    m = build_ttm_metrics("TEST")["metrics"]
    assert m["eps_diluted"] == pytest.approx((50 + 48 + 46 + 44) / 100.0)


def test_eps_never_summed_from_decumulated_rows(mock_periods):
    # Decumulated/synthesized rows carry YTD or full-year EPS — summing them
    # inflated TTM P/E 2-3x (JPM, META, WMT, LLY...). Without shares, honest None.
    qs = _eight_quarters()
    qs[1]["quality"] = "standalone_decumulated"
    for q in qs:
        q["metrics"]["sharesOutstanding"] = None
    mock_periods(qs)
    m = build_ttm_metrics("TEST")["metrics"]
    assert m["eps_diluted"] is None
    assert m["net_income"] is not None  # NI itself is decumulated-correct


def test_quarter_eps_ignores_row_eps_on_decumulated_rows(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    qs = _eight_quarters()
    qs[0]["quality"] = "standalone_decumulated"
    qs[0]["metrics"]["epsDiluted"] = 1.5   # YTD value — must be ignored
    mock_periods(qs)
    m = build_latest_quarter_metrics("TEST")["metrics"]
    assert m["eps_diluted"] == pytest.approx(50.0 / 100.0)  # NI / shares instead


def test_prior_balances_from_year_ago_quarter(mock_periods):
    mock_periods(_eight_quarters())
    out = build_ttm_metrics("TEST")
    assert out["prior"] == {"stockholders_equity": 600.0, "total_assets": 1800.0}


def test_fewer_than_four_quarters_errors_not_raises(mock_periods):
    mock_periods(_eight_quarters()[:3])
    out = build_ttm_metrics("SEMIANNUAL")
    assert out["error"]
    assert "annual" in out["hint"]


def test_ytd_fallback_quarters_are_excluded(mock_periods):
    qs = _eight_quarters()
    for q in qs[:5]:
        q["quality"] = "ytd_fallback"  # cumulative — summing would double-count
    mock_periods(qs)
    out = build_ttm_metrics("TEST")
    assert out["error"]  # only 3 clean quarters remain


def test_decumulated_and_synthesized_quarters_are_summable(mock_periods):
    qs = _eight_quarters()
    qs[0]["quality"] = "standalone_decumulated"
    qs[1]["quality"] = "q4_synthesized"
    mock_periods(qs)
    out = build_ttm_metrics("TEST")
    assert "error" not in out
    assert out["metrics"]["revenue"] == pytest.approx(940.0)


def test_build_periods_exception_becomes_error_dict(monkeypatch):
    def boom(ticker, period, n):
        raise RuntimeError("no data")
    monkeypatch.setattr("sec_mcp.surface.fundamentals._build_periods", boom)
    out = build_ttm_metrics("NEWIPO")
    assert out["error"]
    assert out["hint"]


def test_latest_quarter_is_standalone_not_ytd(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    mock_periods(_eight_quarters())
    out = build_latest_quarter_metrics("TEST")
    m = out["metrics"]
    assert m["revenue"] == 250.0            # single quarter, not a YTD sum
    assert m["net_income"] == 50.0
    assert m["stockholders_equity"] == 800.0
    assert m["eps_diluted"] == pytest.approx(0.5)
    assert out["as_of"] == "2026-03-31"
    assert out["prior"] == {"stockholders_equity": 600.0, "total_assets": 1800.0}


def test_latest_quarter_skips_ytd_fallback_rows(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    qs = _eight_quarters()
    qs[0]["quality"] = "ytd_fallback"       # newest row is cumulative — skip it
    mock_periods(qs)
    out = build_latest_quarter_metrics("TEST")
    assert out["metrics"]["revenue"] == 240.0
    assert out["as_of"] == "2025-12-31"


def test_latest_quarter_no_clean_quarters_errors(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    qs = _eight_quarters()[:2]
    for q in qs:
        q["quality"] = "ytd_fallback"
    mock_periods(qs)
    out = build_latest_quarter_metrics("TEST")
    assert out["error"]


def test_shared_periods_param_avoids_refetch(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    periods = _eight_quarters()
    # No mock installed for these two calls — passing periods must not fetch
    out_q = build_latest_quarter_metrics("TEST", periods=periods)
    out_t = build_ttm_metrics("TEST", periods=periods)
    assert out_q["metrics"]["revenue"] == 250.0
    assert out_t["metrics"]["revenue"] == pytest.approx(940.0)


def _sap_style_6k_rows():
    """FPI 6-K pathology: annual figures stamped 'standalone Q4', end dates
    spaced by filing cadence (days apart), not quarter cadence."""
    rows = []
    for end in ("2026-05-08", "2026-04-28", "2026-04-21", "2026-03-06",
                "2026-02-03", "2025-10-27", "2025-07-25", "2025-05-21"):
        q = _q(end, 41_900.0, 6_000.0)
        q["fiscalPeriod"] = "Q4"
        q["formType"] = "6-K"
        rows.append(q)
    return rows


def test_ttm_window_guard_rejects_6k_annual_reruns(mock_periods):
    mock_periods(_sap_style_6k_rows())
    out = build_ttm_metrics("SAP")
    assert out["error"]
    assert "annual" in out["hint"]


def test_quarter_cadence_guard_rejects_6k_annual_reruns(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    mock_periods(_sap_style_6k_rows())
    out = build_latest_quarter_metrics("SAP")
    assert out["error"]


def test_single_10q_row_is_trusted_but_single_6k_is_not(mock_periods):
    from sec_mcp.core.ttm import build_latest_quarter_metrics
    lone = _eight_quarters()[:1]
    mock_periods(lone)
    assert build_latest_quarter_metrics("IPO")["metrics"]["revenue"] == 250.0
    lone[0]["formType"] = "6-K"
    mock_periods(lone)
    assert build_latest_quarter_metrics("FPI").get("error")


def test_misaligned_year_ago_quarter_drops_prior(mock_periods):
    qs = _eight_quarters()
    qs[4]["endDate"] = "2025-08-15"  # not ~365d before 2026-03-31
    mock_periods(qs)
    out = build_ttm_metrics("TEST")
    assert out["prior"] is None
    assert out["metrics"]["revenue"] == pytest.approx(940.0)


def test_ratios_pipeline_integration(mock_periods):
    """TTM metrics feed compute_ratios/compute_ticker_metrics unmodified."""
    from sec_mcp.core.ratios import compute_ratios
    from sec_mcp.core.ticker_metrics import compute_ticker_metrics

    mock_periods(_eight_quarters())
    out = build_ttm_metrics("TEST")
    ratios = compute_ratios(out["metrics"], prior=out["prior"])
    assert ratios["net_margin"] == pytest.approx(188.0 / 940.0)
    assert ratios["roe"] == pytest.approx(188.0 / 700.0)  # avg(800, 600)
    assert ratios["roe_basis"] == "avg"
    block = compute_ticker_metrics("TEST", out["metrics"], ratios, price=10.0)
    assert block["peRatio"] == pytest.approx(10.0 / 1.88)
    assert block["totalDebt"] == 600.0
