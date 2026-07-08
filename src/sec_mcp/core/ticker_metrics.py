"""Single-ticker valuation metrics — /api/metrics/{ticker}.

Combines the SEC-XBRL fundamentals already produced by extract_financials
(metrics + ratios dicts) with a live price to compute the valuation multiples
the Fineas Metrics tab needs (P/E, P/S, P/B, EV/EBITDA, EV/Revenue).

Conventions (match /api/comps consumers):
  - margins / returns are PERCENT units (netMargin 24.3 == 24.3%)
  - raw financials (revenue, netIncome, ebitda, freeCashFlow, netDebt,
    marketCap, enterpriseValue) are RAW DOLLARS
"""

from __future__ import annotations

from typing import Any


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _pct(v: Any) -> float | None:
    """Fraction → percent (0.243 → 24.3).

    Inputs come from financials._compute_ratios which always emits fractions,
    so convert unconditionally — a ≤1 heuristic would misread legitimate
    >100% ratios (e.g. Apple's ROE ≈ 1.54).
    """
    n = _num(v)
    return None if n is None else n * 100


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_ticker_metrics(
    ticker: str,
    metrics: dict[str, Any],
    ratios: dict[str, Any],
    price: float | None = None,
    shares_override: float | None = None,
    market_cap_override: float | None = None,
) -> dict[str, Any]:
    """Pure computation: SEC metrics/ratios + price context → valuation block."""
    m, r = metrics or {}, ratios or {}

    revenue = _num(m.get("revenue"))
    net_income = _num(m.get("net_income"))
    ebitda = _num(m.get("ebitda"))
    fcf = _num(m.get("free_cash_flow"))
    equity = _num(m.get("stockholders_equity")) or _num(m.get("total_equity"))
    cash = _num(m.get("cash_and_equivalents"))
    ltd = _num(m.get("long_term_debt"))
    std = _num(m.get("short_term_debt"))
    eps = _num(m.get("eps_diluted")) or _num(m.get("eps_basic"))
    shares = shares_override or _num(m.get("shares_outstanding"))

    total_debt = None
    if ltd is not None or std is not None:
        total_debt = (ltd or 0) + (std or 0)
    net_debt = None
    if total_debt is not None:
        net_debt = total_debt - (cash or 0)

    market_cap = market_cap_override
    if market_cap is None and price and shares:
        market_cap = price * shares

    ev = None
    if market_cap is not None and net_debt is not None:
        ev = market_cap + net_debt

    out: dict[str, Any] = {
        "ticker": ticker.upper(),
        "price": price,
        "sharesOutstanding": shares,
        "marketCap": market_cap,
        "enterpriseValue": ev,
        "netDebt": net_debt,
        "totalDebt": total_debt,
        # Valuation multiples
        "peRatio": _div(price, eps) if (eps or 0) > 0 else None,
        "psRatio": _div(market_cap, revenue) if (revenue or 0) > 0 else None,
        "pbRatio": _div(market_cap, equity) if (equity or 0) > 0 else None,
        "evEbitda": _div(ev, ebitda) if (ebitda or 0) > 0 else None,
        "evToRevenue": _div(ev, revenue) if (revenue or 0) > 0 else None,
        "priceToFcf": _div(market_cap, fcf) if (fcf or 0) > 0 else None,
        # Margins / returns (percent units)
        "grossMargin": _pct(r.get("gross_margin")),
        "operatingMargin": _pct(r.get("operating_margin")),
        "netMargin": _pct(r.get("net_margin")),
        "ebitdaMargin": _pct(r.get("ebitda_margin")),
        "fcfMargin": _pct(r.get("fcf_margin")),
        "roe": _pct(r.get("roe") if r.get("roe") is not None else r.get("return_on_equity")),
        "roa": _pct(r.get("return_on_assets")),
        # Balance ratios (raw)
        "currentRatio": _num(r.get("current_ratio")),
        "debtToEquity": _num(r.get("debt_to_equity")),
        "debtToAssets": _num(r.get("debt_to_assets")),
        # Raw financials (dollars)
        "revenue": revenue,
        "netIncome": net_income,
        "ebitda": ebitda,
        "freeCashFlow": fcf,
        "eps": eps,
    }
    return out
