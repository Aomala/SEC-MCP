"""Build tests/golden/golden_values.json from EDGAR companyfacts directly.

This deliberately does NOT use the sec_mcp extraction pipeline. Each metric is
pinned to a hand-chosen (taxonomy, tag) per company — the tag the company's
actual statements use for that line item (verified against the EDGAR Financial
Report viewer). Period selection here is intentionally simple and strict:

  - annual flows:   duration 340-400 days, period ends in the target year
  - annual instants: end date == that fiscal year's period end
  - quarters:       duration 75-105 days inside the fiscal year (Q1-Q3 only;
                    Q4 is derived as FY - (Q1+Q2+Q3) and marked "derived")

Latest `filed` always wins (amendments supersede). Values are stored in the
company's reporting currency (`currency` field) so FX conversion noise in the
pipeline can be backed out in the test.

Usage:
    python scripts/build_golden_values.py            # writes the JSON
    python scripts/build_golden_values.py --print    # dump to stdout only
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "golden" / "golden_values.json"

UA = os.environ.get("EDGAR_IDENTITY", "")
if not UA:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("EDGAR_IDENTITY="):
                UA = line.split("=", 1)[1].strip().strip('"').strip("'")
if not UA:
    sys.exit("EDGAR_IDENTITY not set")

YEARS = [2023, 2024]
QUARTER_TICKERS = {"AAPL", "MSFT", "JPM"}  # quarters built for FY2024 only

# Per-company pinned tags. unit is the reporting-currency unit key in
# companyfacts (USD for domestic filers, EUR for ASML/SAP).
SPEC: dict[str, dict] = {
    "AAPL": {
        "cik": 320193, "fy_end_month": 9, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
            "capital_expenditures": "PaymentsToAcquirePropertyPlantAndEquipment",
        },
    },
    "MSFT": {
        "cik": 789019, "fy_end_month": 6, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
            "capital_expenditures": "PaymentsToAcquirePropertyPlantAndEquipment",
        },
    },
    "JPM": {
        "cik": 19617, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "Revenues",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "BRK-B": {
        "cik": 1067983, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "Revenues",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "O": {
        "cik": 726728, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "Revenues",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "SPG": {
        "cik": 1063761, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "Revenues",
            # SPG headline NI = attributable to common stockholders (large NCI
            # from the operating partnership; plain NetIncomeLoss isn't tagged)
            "net_income": "NetIncomeLossAvailableToCommonStockholdersBasic",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "NEE": {
        "cik": 753308, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            # NEE total operating revenues = RegulatedAndUnregulatedOperatingRevenue
            # (the 606 incl-tax tag is only the contract-with-customer subset)
            "revenue": ["RegulatedAndUnregulatedOperatingRevenue",
                        "RevenueFromContractWithCustomerIncludingAssessedTax", "Revenues"],
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "COIN": {
        "cik": 1679788, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "USD",
        "tags": {
            "revenue": "Revenues",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "ASML": {
        # ASML reports US GAAP in EUR (NASDAQ listing) — not IFRS
        "cik": 937966, "fy_end_month": 12, "taxonomy": "us-gaap", "unit": "EUR",
        "tags": {
            "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "net_income": "NetIncomeLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "StockholdersEquity",
            "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "SAP": {
        "cik": 1000184, "fy_end_month": 12, "taxonomy": "ifrs-full", "unit": "EUR",
        "tags": {
            "revenue": "Revenue",
            "net_income": "ProfitLoss",
            "total_assets": "Assets",
            "total_liabilities": "Liabilities",
            "stockholders_equity": "Equity",
            "operating_cash_flow": "CashFlowsFromUsedInOperatingActivities",
        },
    },
}

_session = requests.Session()
_session.headers["User-Agent"] = UA


def fetch_companyfacts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    r = _session.get(url, timeout=60)
    r.raise_for_status()
    time.sleep(0.15)  # stay well under 10 req/s
    return r.json()


def _facts_for(cf: dict, taxonomy: str, tag: str | list, unit: str) -> tuple[list[dict], str]:
    """Facts for a tag (or first non-empty tag from a fallback list)."""
    tags = tag if isinstance(tag, list) else [tag]
    for t in tags:
        node = cf.get("facts", {}).get(taxonomy, {}).get(t)
        facts = (node or {}).get("units", {}).get(unit, [])
        if facts:
            return facts, t
    return [], tags[0]


def _days(f: dict) -> int | None:
    if not f.get("start"):
        return None
    try:
        return (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days
    except ValueError:
        return None


def _latest_filed(facts: list[dict]) -> dict | None:
    return max(facts, key=lambda f: f.get("filed", ""), default=None)


def _as_reported(facts: list[dict]) -> list[dict]:
    """Keep facts filed within ~400 days of their period end.

    This pins the golden to AS-ORIGINALLY-REPORTED values (the filing's own
    10-K plus quick amendments) — matching the pipeline's per-accession
    semantics. Without the window, later filings' re-presented comparatives
    win (e.g. COIN's 2025 rescission of safeguarding gross-up restated total
    assets from $207B to $15B for the same period end).
    """
    out = []
    for f in facts:
        try:
            gap = (date.fromisoformat(f["filed"]) - date.fromisoformat(f["end"])).days
        except (KeyError, ValueError):
            continue
        if 0 <= gap <= 400:
            out.append(f)
    return out or facts


def annual_value(facts: list[dict], year: int, fy_end_month: int) -> tuple[float, str] | None:
    """(value, period_end) for the FY whose period ends in `year`."""
    instants = [f for f in facts if not f.get("start") and f["end"].startswith(str(year))
                and int(f["end"][5:7]) == fy_end_month]
    if instants:
        instants = _as_reported(instants)
        ends = sorted({f["end"] for f in instants})
        end = ends[-1]
        best = _latest_filed([f for f in instants if f["end"] == end])
        return (float(best["val"]), end) if best else None
    flows = [f for f in facts
             if f["end"].startswith(str(year)) and int(f["end"][5:7]) == fy_end_month
             and (_days(f) or 0) >= 340 and (_days(f) or 999) <= 400]
    if flows:
        flows = _as_reported(flows)
        ends = sorted({f["end"] for f in flows})
        end = ends[-1]
        best = _latest_filed([f for f in flows if f["end"] == end])
        return (float(best["val"]), end) if best else None
    return None


def quarter_values(facts: list[dict], fy_end: str) -> list[dict]:
    """Standalone Q1-Q3 facts (75-105 day durations) for the FY ending at fy_end."""
    fy_end_d = date.fromisoformat(fy_end)
    fy_start_d = date(fy_end_d.year - 1, fy_end_d.month, fy_end_d.day)
    out: dict[str, dict] = {}
    for f in facts:
        d = _days(f)
        if d is None or not (75 <= d <= 105):
            continue
        end_d = date.fromisoformat(f["end"])
        if not (fy_start_d < end_d < fy_end_d):
            continue
        cur = out.get(f["end"])
        if cur is None or f.get("filed", "") > cur.get("filed", ""):
            out[f["end"]] = f
    rows = [{"period_end": k, "value": float(v["val"])} for k, v in sorted(out.items())]
    for i, r in enumerate(rows):
        r["quarter"] = f"Q{i + 1}"
    return rows


def main() -> None:
    golden: dict = {
        "_meta": {
            "built": date.today().isoformat(),
            "source": "EDGAR companyfacts API, hand-pinned tags per company",
            "note": "values in reporting currency; quarters are standalone 3-month",
        },
        "companies": {},
    }
    for ticker, spec in SPEC.items():
        print(f"── {ticker}", flush=True)
        cf = fetch_companyfacts(spec["cik"])
        entry: dict = {
            "cik": spec["cik"], "currency": spec["unit"],
            "taxonomy": spec["taxonomy"], "tags": spec["tags"], "annual": {},
        }
        for year in YEARS:
            yvals: dict = {}
            for metric, tag in spec["tags"].items():
                facts, tag = _facts_for(cf, spec["taxonomy"], tag, spec["unit"])
                if not facts:
                    print(f"   !! {metric}: tag {tag} has no {spec['unit']} facts")
                    available = sorted(
                        t for t in cf.get("facts", {}).get(spec["taxonomy"], {})
                        if metric.split("_")[0].lower() in t.lower()
                    )[:12]
                    print(f"      candidates: {available}")
                    continue
                got = annual_value(facts, year, spec["fy_end_month"])
                if got is None:
                    print(f"   !! {metric}: no FY{year} value for {tag}")
                    continue
                val, end = got
                yvals[metric] = val
                yvals.setdefault("_period_end", end)
            if yvals:
                entry["annual"][str(year)] = yvals
                print(f"   FY{year} ({yvals.get('_period_end')}): "
                      f"rev={yvals.get('revenue', 0) / 1e9:,.3f}B "
                      f"ni={yvals.get('net_income', 0) / 1e9:,.3f}B "
                      f"assets={yvals.get('total_assets', 0) / 1e9:,.3f}B")
        if ticker in QUARTER_TICKERS:
            fy_end = entry["annual"].get("2024", {}).get("_period_end")
            if fy_end:
                quarters: dict = {}
                for metric in ("revenue", "net_income"):
                    facts, _tag = _facts_for(
                        cf, spec["taxonomy"], spec["tags"][metric], spec["unit"])
                    qs = quarter_values(facts, fy_end)
                    fy_total = entry["annual"]["2024"].get(metric)
                    if len(qs) == 3 and fy_total is not None:
                        qs.append({
                            "quarter": "Q4", "period_end": fy_end,
                            "value": fy_total - sum(q["value"] for q in qs),
                            "derived": True,
                        })
                    quarters[metric] = qs
                    print(f"   FY2024 {metric} quarters: "
                          + ", ".join(f"{q['quarter']}={q['value'] / 1e9:,.2f}B" for q in qs))
                entry["quarters_fy2024"] = quarters
        golden["companies"][ticker] = entry

    if "--print" in sys.argv:
        print(json.dumps(golden, indent=2))
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
