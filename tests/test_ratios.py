"""Tests for the canonical ratio engine (core/ratios.py) — pure, no network."""

import pytest

from sec_mcp.core.ratios import (
    NOT_APPLICABLE,
    PERCENT_KEYS,
    compute_ratios,
    compute_roic,
    effective_tax_rate,
    not_applicable_for,
    to_percent,
)

BASE = {
    "revenue": 1000.0,
    "net_income": 200.0,
    "gross_profit": 500.0,
    "operating_income": 300.0,
    "income_before_tax": 280.0,
    "income_tax_expense": 56.0,       # 20% effective rate
    "interest_expense": 20.0,
    "total_assets": 2000.0,
    "total_liabilities": 1200.0,
    "current_assets": 600.0,
    "current_liabilities": 400.0,
    "stockholders_equity": 800.0,
    "operating_cash_flow": 350.0,
    "ebitda": 400.0,
    "long_term_debt": 500.0,
    "short_term_debt": 100.0,
    "cash_and_equivalents": 150.0,
    "free_cash_flow": 250.0,
}


# ── effective_tax_rate ──────────────────────────────────────────────────────

def test_effective_tax_rate_happy():
    assert effective_tax_rate(BASE) == pytest.approx(0.20)


def test_effective_tax_rate_clamped_above_50pct():
    m = {**BASE, "income_tax_expense": 200.0}  # 71% — one-off noise
    assert effective_tax_rate(m) is None


def test_effective_tax_rate_pretax_loss_or_benefit():
    assert effective_tax_rate({**BASE, "income_before_tax": -50.0}) is None
    assert effective_tax_rate({**BASE, "income_tax_expense": -10.0}) is None
    assert effective_tax_rate({}) is None


# ── compute_roic ────────────────────────────────────────────────────────────

def test_roic_happy_path_effective_tax():
    roic, basis = compute_roic(BASE)
    # NOPAT = 300 × 0.8 = 240; IC = 600 + 800 − 150 = 1250
    assert roic == pytest.approx(240.0 / 1250.0)
    assert basis == "oi_effective_tax"


def test_roic_statutory_fallback_when_rate_unusable():
    m = {**BASE, "income_tax_expense": None}
    roic, basis = compute_roic(m)
    assert roic == pytest.approx(300.0 * 0.79 / 1250.0)
    assert basis == "oi_statutory_tax"


def test_roic_ebit_proxy_when_no_operating_income():
    m = {**BASE, "operating_income": None}
    roic, basis = compute_roic(m)
    # EBIT proxy = 280 + 20 = 300 → NOPAT = 300 × 0.8
    assert roic == pytest.approx(240.0 / 1250.0)
    assert basis == "ebit_proxy"


def test_roic_none_without_debt_or_equity():
    assert compute_roic({**BASE, "long_term_debt": None, "short_term_debt": None}) == (None, None)
    assert compute_roic({**BASE, "stockholders_equity": None}) == (None, None)


def test_roic_accepts_precombined_total_debt():
    # TTM/quarter blocks carry only total_debt (no ltd/std legs)
    m = {**BASE, "long_term_debt": None, "short_term_debt": None, "total_debt": 600.0}
    roic, basis = compute_roic(m)
    assert roic == pytest.approx(240.0 / 1250.0)
    assert basis == "oi_effective_tax"


def test_roic_none_when_invested_capital_nonpositive():
    m = {**BASE, "stockholders_equity": 50.0, "long_term_debt": 10.0,
         "short_term_debt": None, "cash_and_equivalents": 200.0}
    assert compute_roic(m) == (None, None)


# ── compute_ratios ──────────────────────────────────────────────────────────

def test_ratios_superset_of_legacy_keys_same_units():
    r = compute_ratios(BASE)
    assert r["gross_margin"] == pytest.approx(0.5)
    assert r["operating_margin"] == pytest.approx(0.3)
    assert r["net_margin"] == pytest.approx(0.2)
    assert r["current_ratio"] == pytest.approx(1.5)
    assert r["debt_to_equity"] == pytest.approx(600.0 / 800.0)
    assert r["debt_to_assets"] == pytest.approx(0.3)
    assert r["ebitda_margin"] == pytest.approx(0.4)
    assert r["fcf_margin"] == pytest.approx(0.25)
    assert r["ocf_to_net_income"] == pytest.approx(1.75)
    # new keys
    assert r["roic"] is not None
    assert r["liabilities_to_equity"] == pytest.approx(1.5)
    assert r["effective_tax_rate"] == pytest.approx(0.20)


def test_roe_ending_balance_without_prior():
    r = compute_ratios(BASE)
    assert r["roe"] == pytest.approx(0.25)
    assert r["return_on_equity"] == r["roe"]
    assert r["roe_basis"] == "ending"
    assert r["return_on_assets"] == pytest.approx(0.10)


def test_roe_average_balance_with_prior():
    prior = {"stockholders_equity": 600.0, "total_assets": 1800.0}
    r = compute_ratios(BASE, prior=prior)
    assert r["roe"] == pytest.approx(200.0 / 700.0)
    assert r["roe_basis"] == "avg"
    assert r["return_on_assets"] == pytest.approx(200.0 / 1900.0)


def test_negative_prior_equity_falls_back_to_ending():
    r = compute_ratios(BASE, prior={"stockholders_equity": -100.0})
    assert r["roe"] == pytest.approx(0.25)
    assert r["roe_basis"] == "ending"


def test_negative_equity_nulls_equity_ratios():
    m = {**BASE, "stockholders_equity": -500.0}
    r = compute_ratios(m)
    assert r["roe"] is None
    assert r["return_on_equity"] is None
    assert r["roe_basis"] is None
    assert r["debt_to_equity"] is None
    assert r["liabilities_to_equity"] is None
    # non-equity ratios unaffected
    assert r["net_margin"] == pytest.approx(0.2)


def test_precombined_total_debt_wins():
    m = {**BASE, "long_term_debt": None, "short_term_debt": None, "total_debt": 900.0}
    r = compute_ratios(m)
    assert r["debt_to_equity"] == pytest.approx(900.0 / 800.0)


def test_empty_metrics_total():
    r = compute_ratios({})
    assert all(v is None for v in r.values())


# ── to_percent ──────────────────────────────────────────────────────────────

def test_to_percent_scales_exactly_percent_keys():
    r = compute_ratios(BASE)
    p = to_percent(r)
    assert p["gross_margin"] == pytest.approx(50.0)
    assert p["roe"] == pytest.approx(25.0)
    assert p["effective_tax_rate"] == pytest.approx(20.0)
    # multiples/coverage stay raw
    assert p["current_ratio"] == pytest.approx(1.5)
    assert p["debt_to_equity"] == pytest.approx(0.75)
    assert p["ocf_to_net_income"] == pytest.approx(1.75)
    assert p["liabilities_to_equity"] == pytest.approx(1.5)
    # non-numeric passthrough, original dict untouched
    assert p["roe_basis"] == "ending"
    assert r["gross_margin"] == pytest.approx(0.5)


def test_percent_keys_are_known_ratio_keys():
    r = set(compute_ratios(BASE))
    assert r >= PERCENT_KEYS


# ── applicability map ───────────────────────────────────────────────────────

def test_not_applicable_bank():
    snake = not_applicable_for("bank")
    assert "gross_margin" in snake
    assert "current_ratio" in snake
    assert "roic" in snake
    camel = not_applicable_for("bank", camel=True)
    assert "grossMargin" in camel
    assert "currentRatio" in camel
    assert "evEbitda" in camel
    assert "priceToFcf" in camel
    assert "freeCashFlow" in camel


def test_not_applicable_standard_and_unknown_empty():
    assert not_applicable_for("standard") == []
    assert not_applicable_for(None) == []
    assert not_applicable_for("crypto") == []


def test_not_applicable_classes_cover_expected():
    assert set(NOT_APPLICABLE) == {"bank", "insurance", "reit", "utility"}
    assert "gross_margin" in NOT_APPLICABLE["reit"]
    assert NOT_APPLICABLE["utility"] == frozenset({"current_ratio"})
