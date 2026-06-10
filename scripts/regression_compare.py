"""Regression snapshot/diff for the extraction pipeline.

Snapshot the current outputs of extract_financials + get_fmp_shaped_history
for a ticker set, then diff a later run against the snapshot to prove that a
refactor only changed what it intended to change.

Usage:
    python scripts/regression_compare.py snapshot                 # write baseline
    python scripts/regression_compare.py diff                     # compare vs baseline
    python scripts/regression_compare.py snapshot --tickers AAPL,MSFT
    python scripts/regression_compare.py diff --tol 0.001

Baseline file: tests/golden/regression_snapshot.json (committed to git).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sec_mcp.financials import extract_financials, get_fmp_shaped_history  # noqa: E402

SNAPSHOT = ROOT / "tests" / "golden" / "regression_snapshot.json"

# Liquid, sector-diverse set: tech, banks, insurance, REIT, utility, crypto,
# energy, healthcare, retail, industrials, FPIs.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "JPM", "BAC", "GS", "WFC", "BRK-B", "PGR", "MET",
    "O", "SPG", "PLD", "NEE", "DUK", "SO",
    "COIN", "MSTR", "XOM", "CVX", "UNH", "JNJ", "LLY",
    "WMT", "COST", "HD", "CAT", "BA", "GE",
    "ASML", "SAP", "TSM", "SHOP", "ORCL", "CRM",
]

ANNUAL_FIELDS = (
    "revenue", "net_income", "gross_profit", "operating_income",
    "total_assets", "total_liabilities", "stockholders_equity",
    "operating_cash_flow", "capital_expenditures", "free_cash_flow",
    "eps_diluted", "ebitda",
)
HIST_FIELDS = ("revenue", "netIncome", "operatingCashFlow", "freeCashFlow")


def _capture_one(ticker: str) -> dict:
    out: dict = {"ticker": ticker}
    try:
        annual = extract_financials(ticker, form_type="10-K") or {}
        m = annual.get("metrics") or {}
        out["annual"] = {k: m.get(k) for k in ANNUAL_FIELDS}
        out["annual_meta"] = {
            "fiscal_year": annual.get("fiscal_year"),
            "report_date": (annual.get("filing_info") or {}).get("report_date"),
            "revenue_source": (annual.get("metrics_sourced") or {}).get("revenue"),
            "industry_class": annual.get("industry_class"),
        }
    except Exception as exc:
        out["annual_error"] = str(exc)
    try:
        hist = get_fmp_shaped_history(ticker, period="quarter", limit=4)
        out["quarters"] = [
            {"date": r.get("date"), "period": r.get("period"),
             **{f: r.get(f) for f in HIST_FIELDS if f in r}}
            for r in (hist.get("income") or [])
        ]
        cash = {r.get("date"): r for r in (hist.get("cashflow") or [])}
        for q in out["quarters"]:
            c = cash.get(q["date"]) or {}
            q["operatingCashFlow"] = c.get("operatingCashFlow")
            q["freeCashFlow"] = c.get("freeCashFlow")
    except Exception as exc:
        out["quarters_error"] = str(exc)
    return out


def capture(tickers: list[str]) -> dict:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(_capture_one, tickers):
            results[r["ticker"]] = r
            sys.stderr.write(f"  {r['ticker']}: "
                             f"{'ok' if 'annual' in r else r.get('annual_error', '?')}\n")
    return results


def _close(a, b, tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)
    except TypeError:
        return a == b


def diff(tickers: list[str], tol: float) -> int:
    if not SNAPSHOT.exists():
        sys.exit("No snapshot found — run `snapshot` first.")
    base = json.loads(SNAPSHOT.read_text())
    cur = capture(tickers)
    changes = 0
    for t in tickers:
        b, c = base.get(t), cur.get(t)
        if not b:
            print(f"~ {t}: not in baseline (new ticker)")
            continue
        for k in ANNUAL_FIELDS:
            bv, cv = (b.get("annual") or {}).get(k), (c.get("annual") or {}).get(k)
            if not _close(bv, cv, tol):
                changes += 1
                print(f"! {t}.annual.{k}: {bv} -> {cv}")
        bm, cm = b.get("annual_meta") or {}, c.get("annual_meta") or {}
        for k in ("fiscal_year", "report_date", "revenue_source"):
            if str(bm.get(k)) != str(cm.get(k)):
                changes += 1
                print(f"! {t}.meta.{k}: {bm.get(k)} -> {cm.get(k)}")
        bq = {q.get("date"): q for q in b.get("quarters") or []}
        for q in c.get("quarters") or []:
            old = bq.get(q.get("date"))
            if not old:
                continue
            for k in ("revenue", "netIncome", "operatingCashFlow", "freeCashFlow"):
                if not _close(old.get(k), q.get(k), tol):
                    changes += 1
                    print(f"! {t}.q[{q['date']}].{k}: {old.get(k)} -> {q.get(k)}")
    print(f"\n{changes} change(s) vs baseline" if changes else "\nclean — no changes vs baseline")
    return 1 if changes else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["snapshot", "diff"])
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance for numeric comparisons")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.mode == "snapshot":
        results = capture(tickers)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(results, indent=1, default=str))
        print(f"wrote {SNAPSHOT} ({len(results)} tickers)")
    else:
        sys.exit(diff(tickers, args.tol))


if __name__ == "__main__":
    main()
