"""Canonical ratio engine — the ONE place ratio math lives.

financials._compute_ratios delegates here; /api/metrics and /api/comps
recompute from cached metrics at serve time so new ratio fields appear
without invalidating warm cache entries.

Unit convention: every ratio is a FRACTION (gross_margin 0.43 == 43%).
`to_percent()` is the single fraction→percent conversion point; endpoints
that speak percent (/api/metrics via core.ticker_metrics) go through it.
"""

from __future__ import annotations

from typing import Any

# US federal statutory corporate rate — ROIC fallback only, never reported
# as the company's effective rate.
STATUTORY_TAX_RATE = 0.21

# Ratio keys that are percent-like fractions (×100 in to_percent). Pure
# multiples/coverage ratios (current_ratio, debt_to_equity, …) stay raw.
PERCENT_KEYS: frozenset[str] = frozenset({
    "gross_margin", "operating_margin", "net_margin",
    "return_on_assets", "return_on_equity", "roe", "roic",
    "ebitda_margin", "fcf_margin", "effective_tax_rate",
})

# Metrics that are structurally meaningless for an industry class — not
# "missing data" but "wrong question" (banks have no COGS; unclassified
# balance sheets have no current ratio; invested-capital ROIC is ill-defined
# for financials). Frontends drop these rows instead of rendering N/A.
NOT_APPLICABLE: dict[str, frozenset[str]] = {
    "bank": frozenset({
        "gross_margin", "current_ratio", "ebitda_margin", "ev_ebitda",
        "ev_to_revenue", "fcf_margin", "price_to_fcf", "free_cash_flow",
        "roic",
    }),
    "insurance": frozenset({
        "gross_margin", "current_ratio", "ebitda_margin", "ev_ebitda",
        "roic",
    }),
    "reit": frozenset({"gross_margin", "current_ratio"}),
    "utility": frozenset({"current_ratio"}),
}

# snake_case ratio/metric names → the camelCase names /api/metrics speaks.
SNAKE_TO_API: dict[str, str] = {
    "gross_margin": "grossMargin",
    "operating_margin": "operatingMargin",
    "net_margin": "netMargin",
    "ebitda_margin": "ebitdaMargin",
    "fcf_margin": "fcfMargin",
    "return_on_assets": "roa",
    "return_on_equity": "roe",
    "roe": "roe",
    "roic": "roic",
    "current_ratio": "currentRatio",
    "debt_to_equity": "debtToEquity",
    "debt_to_assets": "debtToAssets",
    "liabilities_to_equity": "liabilitiesToEquity",
    "ev_ebitda": "evEbitda",
    "ev_to_revenue": "evToRevenue",
    "price_to_fcf": "priceToFcf",
    "free_cash_flow": "freeCashFlow",
    "effective_tax_rate": "effectiveTaxRate",
}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _div(a: float | None, b: float | None) -> float | None:
    """Safe division — returns None if either operand is None or divisor is zero."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def effective_tax_rate(m: dict[str, Any]) -> float | None:
    """income_tax_expense / income_before_tax, honest None when not computable.

    Guards: pretax must be positive (a rate against a pretax loss is
    meaningless) and tax expense non-negative (benefit years distort).
    Clamped to [0, 0.50] — outside that band the inputs are one-off noise
    (audit settlements, valuation-allowance releases), not a usable rate.
    """
    tax = _num(m.get("income_tax_expense"))
    pretax = _num(m.get("income_before_tax"))
    if tax is None or pretax is None or pretax <= 0 or tax < 0:
        return None
    rate = tax / pretax
    if rate > 0.50:
        return None
    return rate


def compute_roic(m: dict[str, Any]) -> tuple[float | None, str | None]:
    """ROIC = NOPAT / invested capital. Returns (roic_fraction, basis).

    NOPAT = operating_income × (1 − tax rate); effective rate when computable,
    else the 21% statutory fallback (basis records which). When operating
    income is unavailable, EBIT proxy = income_before_tax + interest_expense
    (both required). Invested capital = total_debt + equity − cash; cash
    missing counts as 0, but equity and at least one debt leg must exist.
    """
    eq = _num(m.get("stockholders_equity")) or _num(m.get("total_equity"))
    cash = _num(m.get("cash_and_equivalents")) or 0.0
    # Pre-combined total_debt (TTM/quarter builders) wins; else the legs.
    total_debt = _num(m.get("total_debt"))
    if total_debt is None:
        ltd = _num(m.get("long_term_debt"))
        std = _num(m.get("short_term_debt"))
        if ltd is None and std is None:
            return None, None
        total_debt = (ltd or 0.0) + (std or 0.0)
    if eq is None:
        return None, None
    invested_capital = total_debt + eq - cash
    if invested_capital <= 0:
        return None, None

    rate = effective_tax_rate(m)
    oi = _num(m.get("operating_income"))
    if oi is not None:
        basis = "oi_effective_tax" if rate is not None else "oi_statutory_tax"
        nopat = oi * (1 - (rate if rate is not None else STATUTORY_TAX_RATE))
    else:
        pretax = _num(m.get("income_before_tax"))
        interest = _num(m.get("interest_expense"))
        if pretax is None or interest is None:
            return None, None
        basis = "ebit_proxy"
        ebit = pretax + abs(interest)
        nopat = ebit * (1 - (rate if rate is not None else STATUTORY_TAX_RATE))
    return nopat / invested_capital, basis


def compute_ratios(
    m: dict[str, Any],
    prior: dict[str, Any] | None = None,
    industry: str | None = None,
) -> dict[str, Any]:
    """Compute financial ratios from extracted metrics (fractions).

    Superset of the historical _compute_ratios keys plus roic/roic_basis,
    effective_tax_rate, liabilities_to_equity, roe_basis. ROE/ROA use average
    balances when `prior` carries the year-ago equity/assets (basis marked);
    negative equity nulls ROE and the equity-denominator leverage ratios
    instead of emitting sign-flipped nonsense.
    """
    m = m or {}
    rev = _num(m.get("revenue"))
    ni = _num(m.get("net_income"))
    gp = _num(m.get("gross_profit"))
    oi = _num(m.get("operating_income"))
    ta = _num(m.get("total_assets"))
    tl = _num(m.get("total_liabilities"))
    ca = _num(m.get("current_assets"))
    cl = _num(m.get("current_liabilities"))
    eq = _num(m.get("stockholders_equity")) or _num(m.get("total_equity"))
    ocf = _num(m.get("operating_cash_flow"))
    ebitda = _num(m.get("ebitda"))
    ltd = _num(m.get("long_term_debt"))
    std = _num(m.get("short_term_debt"))
    fcf = _num(m.get("free_cash_flow"))

    # Pre-combined total_debt (TTM builder) wins; else derive from the legs.
    total_debt = _num(m.get("total_debt"))
    if total_debt is None and (ltd is not None or std is not None):
        total_debt = (ltd or 0) + (std or 0)

    # Average balances for return ratios when the prior period is known.
    prior = prior or {}
    prior_eq = _num(prior.get("stockholders_equity")) or _num(prior.get("total_equity"))
    prior_ta = _num(prior.get("total_assets"))
    roe_basis = "ending"
    eq_denom = eq
    if eq is not None and prior_eq is not None and eq > 0 and prior_eq > 0:
        eq_denom = (eq + prior_eq) / 2
        roe_basis = "avg"
    ta_denom = ta
    if ta is not None and prior_ta is not None and ta > 0 and prior_ta > 0:
        ta_denom = (ta + prior_ta) / 2

    # Negative/zero equity → equity-denominator ratios are meaningless.
    eq_ok = eq is not None and eq > 0
    roe = _div(ni, eq_denom) if eq_ok else None
    roic, roic_basis = compute_roic(m)

    return {
        "gross_margin": _div(gp, rev),
        "operating_margin": _div(oi, rev),
        "net_margin": _div(ni, rev),
        "return_on_assets": _div(ni, ta_denom),
        "return_on_equity": roe,
        "roe": roe,  # alias used by the UI
        "roe_basis": roe_basis if roe is not None else None,
        "roic": roic,
        "roic_basis": roic_basis,
        "effective_tax_rate": effective_tax_rate(m),
        "current_ratio": _div(ca, cl),
        "debt_to_equity": _div(total_debt, eq) if eq_ok else None,
        "debt_to_assets": _div(total_debt, ta),
        "liabilities_to_equity": _div(tl, eq) if eq_ok else None,
        "ebitda_margin": _div(ebitda, rev),
        "fcf_margin": _div(fcf, rev),
        "ocf_to_net_income": _div(ocf, ni),
    }


def to_percent(ratios: dict[str, Any]) -> dict[str, Any]:
    """New dict with PERCENT_KEYS ×100; non-numeric and other keys passthrough."""
    out: dict[str, Any] = {}
    for k, v in (ratios or {}).items():
        n = _num(v)
        out[k] = n * 100 if (k in PERCENT_KEYS and n is not None) else v
    return out


def not_applicable_for(industry: str | None, camel: bool = False) -> list[str]:
    """Metrics that are structurally not-applicable for an industry class."""
    keys = NOT_APPLICABLE.get((industry or "").lower(), frozenset())
    if camel:
        return sorted(SNAKE_TO_API.get(k, k) for k in keys)
    return sorted(keys)
