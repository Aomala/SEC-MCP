"""Tests for compute_ticker_metrics (core/ticker_metrics.py) — pure, no network."""

import pytest

from sec_mcp.core.ticker_metrics import compute_ticker_metrics


def test_full_computation(sample_financial_metrics):
    ratios = {
        "gross_margin": 0.433, "operating_margin": 0.303, "net_margin": 0.253,
        "return_on_equity": 1.543, "return_on_assets": 0.283,
        "current_ratio": 0.98, "debt_to_equity": 1.53, "debt_to_assets": 0.28,
    }
    out = compute_ticker_metrics("aapl", sample_financial_metrics, ratios, price=200.0)

    assert out["ticker"] == "AAPL"
    # marketCap = price × shares
    assert out["marketCap"] == 200.0 * 15_943_425_000
    # netDebt = LTD − cash (no STD in fixture)
    assert out["netDebt"] == 98_959_000_000 - 29_965_000_000
    assert out["enterpriseValue"] == out["marketCap"] + out["netDebt"]
    # P/E = price / diluted EPS
    assert abs(out["peRatio"] - 200.0 / 6.13) < 1e-9
    assert out["psRatio"] == out["marketCap"] / 394_328_000_000
    assert out["pbRatio"] == out["marketCap"] / 64_671_000_000
    # margins normalized to percent
    assert abs(out["netMargin"] - 25.3) < 1e-9
    assert abs(out["roe"] - 154.3) < 1e-9


def test_no_price_no_multiples(sample_financial_metrics):
    out = compute_ticker_metrics("AAPL", sample_financial_metrics, {})
    assert out["marketCap"] is None
    assert out["peRatio"] is None
    assert out["psRatio"] is None
    # raw financials still present
    assert out["revenue"] == 394_328_000_000


def test_polygon_overrides_win(sample_financial_metrics):
    out = compute_ticker_metrics(
        "AAPL", sample_financial_metrics, {}, price=200.0,
        market_cap_override=3.1e12, shares_override=15.0e9,
    )
    assert out["marketCap"] == 3.1e12
    assert out["sharesOutstanding"] == 15.0e9


def test_empty_inputs_are_all_none_not_nan():
    out = compute_ticker_metrics("XXXX", {}, {}, price=None)
    for key, val in out.items():
        if key == "ticker":
            continue
        assert val is None, f"{key} should be None, got {val}"


def test_negative_eps_yields_no_pe():
    out = compute_ticker_metrics("LOSS", {"eps_diluted": -2.5}, {}, price=50.0)
    assert out["peRatio"] is None


def test_fractions_convert_to_percent_even_above_one():
    out = compute_ticker_metrics("T", {}, {"net_margin": 0.243, "return_on_equity": 1.543})
    assert out["netMargin"] == 24.3
    assert abs(out["roe"] - 154.3) < 1e-9


def test_roic_and_tax_rate_surface_as_percent():
    out = compute_ticker_metrics(
        "T", {}, {"roic": 0.192, "roic_basis": "oi_effective_tax",
                  "effective_tax_rate": 0.20})
    assert out["roic"] == pytest.approx(19.2)
    assert out["roicBasis"] == "oi_effective_tax"
    assert out["effectiveTaxRate"] == pytest.approx(20.0)


def test_net_debt_basis_markers():
    # debt known, cash missing → debt_only
    out = compute_ticker_metrics("T", {"long_term_debt": 100.0}, {})
    assert out["netDebt"] == 100.0
    assert out["netDebtBasis"] == "debt_only"
    # cash known, debt missing → cash_only (negative net debt, EV can compute)
    out = compute_ticker_metrics(
        "T", {"cash_and_equivalents": 40.0}, {}, market_cap_override=1000.0)
    assert out["netDebt"] == -40.0
    assert out["netDebtBasis"] == "cash_only"
    assert out["enterpriseValue"] == 960.0
    # both known → debt_and_cash
    out = compute_ticker_metrics(
        "T", {"long_term_debt": 100.0, "cash_and_equivalents": 40.0}, {})
    assert out["netDebtBasis"] == "debt_and_cash"


def test_precombined_total_debt_wins():
    out = compute_ticker_metrics("T", {"total_debt": 600.0}, {})
    assert out["totalDebt"] == 600.0
    assert out["netDebt"] == 600.0
