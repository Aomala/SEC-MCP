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

    # Pre-combined total_debt (TTM builder) wins; else derive from the legs.
    total_debt = _num(m.get("total_debt"))
    if total_debt is None and (ltd is not None or std is not None):
        total_debt = (ltd or 0) + (std or 0)
    # Net debt with either side known (missing side = 0, basis marked) — a
    # missing debt line was nulling EV and every EV multiple even when
    # EV ≈ marketCap. Both missing → still None.
    net_debt = None
    net_debt_basis = None
    if total_debt is not None:
        net_debt = total_debt - (cash or 0)
        net_debt_basis = "debt_and_cash" if cash is not None else "debt_only"
    elif cash is not None:
        net_debt = -cash
        net_debt_basis = "cash_only"

    market_cap = market_cap_override
    if market_cap is None and price and shares:
        market_cap = price * shares

    ev = None
    if market_cap is not None and net_debt is not None:
        ev = market_cap + net_debt

    # P/E: price/EPS when the two are on the same share basis; market-cap /
    # net-income otherwise. Cross-listed filers break the eps path — TSM's
    # XBRL EPS is per ordinary Taiwan share while price/shares are ADR-basis
    # (5:1), serving P/E 5x too high — and the two bases disagreeing by >1.8x
    # is the tell (normal diluted-vs-weighted drift is far smaller).
    pe = _div(price, eps) if (eps or 0) > 0 else None
    pe_basis = "eps" if pe is not None else None
    pe_mcap = _div(market_cap, net_income) if (net_income or 0) > 0 else None
    if pe_mcap is not None and (pe is None or pe / pe_mcap > 1.8 or pe_mcap / pe > 1.8):
        pe, pe_basis = pe_mcap, "mcap_ni"

    out: dict[str, Any] = {
        "ticker": ticker.upper(),
        "price": price,
        "sharesOutstanding": shares,
        "marketCap": market_cap,
        "enterpriseValue": ev,
        "netDebt": net_debt,
        "netDebtBasis": net_debt_basis,
        "totalDebt": total_debt,
        # Valuation multiples
        "peRatio": pe,
        "peBasis": pe_basis,
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
        "roic": _pct(r.get("roic")),
        "roicBasis": r.get("roic_basis"),
        "effectiveTaxRate": _pct(r.get("effective_tax_rate")),
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
        "epsDiluted": _num(m.get("eps_diluted")),
    }
    return out
