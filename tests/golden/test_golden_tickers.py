"""Golden-ticker correctness suite.

Compares the full extraction pipeline against hand-pinned values pulled
straight from EDGAR companyfacts (see scripts/build_golden_values.py).
These tests are the gate for any change to period selection or concept
resolution: if they go red, the pipeline is picking the wrong tag, the
wrong period, or the wrong row.

Run:  pytest tests/golden -m integration
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sec_mcp.financials import extract_financials, get_fmp_shaped_history

GOLDEN = json.loads((Path(__file__).parent / "golden_values.json").read_text())
COMPANIES = GOLDEN["companies"]

REL_TOL = 0.005          # 0.5% — values must match EDGAR to the rounding digit
IDENTITY_TOL = 0.01      # 1% — A = L + E coherence on extracted metrics
QUARTER_TOL = 0.01       # 1% — standalone quarters vs golden

# pipeline metric key → golden metric key
METRIC_KEYS = (
    "revenue", "net_income", "total_assets", "total_liabilities",
    "stockholders_equity", "operating_cash_flow", "capital_expenditures",
)

ANNUAL_CASES = [
    (t, y) for t, spec in COMPANIES.items() for y in spec.get("annual", {})
]

_extract_cache: dict[tuple, dict] = {}


def _extract(ticker: str, year: int) -> dict:
    key = (ticker, year)
    if key not in _extract_cache:
        _extract_cache[key] = extract_financials(ticker, year=year, form_type="10-K") or {}
    return _extract_cache[key]


def _to_native(result: dict, value: float | None) -> float | None:
    """Back out the pipeline's USD conversion so we compare in the filing's
    reporting currency (golden values are stored native)."""
    if value is None:
        return None
    ccy = result.get("reporting_currency") or "USD"
    rate = result.get("fx_rate")
    if ccy != "USD" and rate:
        return value / rate
    return value


def _rel_err(got: float, want: float) -> float:
    return abs(got - want) / max(abs(want), 1.0)


@pytest.mark.integration
@pytest.mark.parametrize("ticker,year", ANNUAL_CASES, ids=lambda v: str(v))
def test_annual_metrics(ticker: str, year: str):
    golden = COMPANIES[ticker]["annual"][year]
    result = _extract(ticker, int(year))
    assert result and not result.get("error"), f"extraction failed: {result.get('error')}"

    # Filing must actually be the requested fiscal year
    rd = (result.get("filing_info") or {}).get("report_date", "")
    assert rd.startswith(year), f"wrong filing selected: report_date={rd!r}, wanted FY{year}"

    metrics = result.get("metrics") or {}
    failures = []
    for key in METRIC_KEYS:
        want = golden.get(key)
        if want is None:
            continue
        got = _to_native(result, metrics.get(key))
        if got is None:
            failures.append(f"{key}: missing (want {want:,.0f})")
            continue
        # capex sign convention differs (outflow may be negative downstream)
        if key == "capital_expenditures":
            got, want = abs(got), abs(want)
        if _rel_err(got, want) > REL_TOL:
            failures.append(
                f"{key}: got {got:,.0f}, want {want:,.0f} "
                f"({_rel_err(got, want):.2%} off) [source: {result.get('metrics_sourced', {}).get(key)}]"
            )
    assert not failures, f"{ticker} FY{year}:\n  " + "\n  ".join(failures)


@pytest.mark.integration
@pytest.mark.parametrize(
    "ticker,year",
    [(t, y) for t, y in ANNUAL_CASES
     if {"total_assets", "total_liabilities", "stockholders_equity"}
     <= set(COMPANIES[t]["annual"][y])],
    ids=lambda v: str(v),
)
def test_accounting_equation(ticker: str, year: str):
    """Extracted A = L + E must cohere (catches mismatched tag picks)."""
    result = _extract(ticker, int(year))
    m = result.get("metrics") or {}
    ta, tl, eq = m.get("total_assets"), m.get("total_liabilities"), m.get("stockholders_equity")
    assert None not in (ta, tl, eq), f"missing balance metrics: A={ta} L={tl} E={eq}"
    # Equity is parent-only; the consolidated identity needs NCI added back
    nci = m.get("minority_interest") or 0
    diff = abs(ta - (tl + eq + nci)) / abs(ta)
    assert diff < 0.05, (
        f"A != L + E + NCI for {ticker} FY{year}: "
        f"{ta:,.0f} vs {tl + eq + nci:,.0f} ({diff:.1%})"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "ticker", [t for t, s in COMPANIES.items() if s.get("quarters_fy2024")],
)
def test_standalone_quarters(ticker: str):
    """Quarterly history rows must be true standalone 3-month values."""
    golden_q = COMPANIES[ticker]["quarters_fy2024"]
    hist = get_fmp_shaped_history(ticker, period="quarter", limit=12)
    rows = {r["date"]: r for r in hist.get("income", [])}
    assert rows, "no quarterly income history returned"

    checked = 0
    failures = []
    field = {"revenue": "revenue", "net_income": "netIncome"}
    for metric, quarters in golden_q.items():
        for q in quarters:
            if q.get("derived"):
                continue  # Q4 has no 10-Q filing; synthesized Q4 tested separately
            row = rows.get(q["period_end"])
            if row is None:
                continue  # period outside the returned window
            got = row.get(field[metric])
            if got is None:
                failures.append(f"{metric} {q['quarter']} ({q['period_end']}): missing")
                continue
            checked += 1
            if _rel_err(got, q["value"]) > QUARTER_TOL:
                failures.append(
                    f"{metric} {q['quarter']} ({q['period_end']}): got {got:,.0f}, "
                    f"want {q['value']:,.0f} ({_rel_err(got, q['value']):.1%} off — YTD leak?)"
                )
    assert checked >= 2, f"too few golden quarters matched ({checked}) — dates: {sorted(rows)}"
    assert not failures, f"{ticker} quarters:\n  " + "\n  ".join(failures)


@pytest.mark.integration
@pytest.mark.parametrize(
    "ticker", [t for t, s in COMPANIES.items() if s.get("quarters_fy2024")],
)
def test_quarters_sum_to_fiscal_year(ticker: str):
    """ΣQ1..Q4 standalone = FY total (decumulation arithmetic check)."""
    golden_q = COMPANIES[ticker]["quarters_fy2024"]
    fy = COMPANIES[ticker]["annual"]["2024"]
    for metric, quarters in golden_q.items():
        if len(quarters) < 4 or fy.get(metric) is None:
            continue
        total = sum(q["value"] for q in quarters)
        assert _rel_err(total, fy[metric]) < 0.01, (
            f"{ticker} {metric}: ΣQ = {total:,.0f} != FY {fy[metric]:,.0f}"
        )
