"""Live audit of every MCP tool in sec_mcp.server against 5 representative tickers.

Writes MCP_AUDIT.md at repo root with: tool name, inputs, output shape,
data source, pass/fail per ticker, exact error + reproduction for failures.

Ticker panel:
  AAPL — mega-cap, standard filer
  CAVA — mid-cap (~$10B), 2023 IPO, short history
  CRWV — recent IPO (Mar 2025), minimal filing history
  ASML — foreign private issuer (20-F/6-K filer)
  SMCI — recent corporate action (10-for-1 split Oct 2024) + filing delays

Run:  .venv/bin/python scripts/audit_tools.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Import the server module — tools are FastMCP FunctionTool objects
from sec_mcp import server as S

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "MCP_AUDIT.md"

TICKERS = ["AAPL", "CAVA", "CRWV", "ASML", "SMCI"]
TIMEOUT = 120  # seconds per call — extraction on a cold cache is slow


def fn(tool):
    """Unwrap a FastMCP tool to its underlying python function."""
    return getattr(tool, "fn", tool)


def shape(result) -> str:
    """Describe the result shape compactly for the audit table."""
    try:
        if isinstance(result, dict):
            keys = list(result.keys())[:8]
            return f"dict keys={keys}"
        if isinstance(result, list):
            n = len(result)
            inner = type(result[0]).__name__ if n else "—"
            return f"list[{inner}] n={n}"
        if isinstance(result, str):
            return f"str len={len(result)}"
        return type(result).__name__
    except Exception:
        return "?"


def looks_like_error(result) -> str | None:
    """Detect soft failures: error dicts, empty payloads."""
    if result is None:
        return "returned None"
    if isinstance(result, dict) and result.get("error"):
        return f"error dict: {result['error']!s:.200}"
    if isinstance(result, list):
        if len(result) == 0:
            return "empty list"
        if isinstance(result[0], dict) and result[0].get("error"):
            return f"error dict: {result[0]['error']!s:.200}"
    if isinstance(result, str) and result.startswith(("No filings", "Could not")):
        return f"soft error string: {result[:200]}"
    return None


def run_case(label, callable_, kwargs):
    """Run one tool call with a hard timeout; capture outcome."""
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(callable_, **kwargs)
        try:
            result = fut.result(timeout=TIMEOUT)
            ms = int((time.time() - t0) * 1000)
            soft = looks_like_error(result)
            if soft:
                return {"label": label, "status": "SOFT-FAIL", "ms": ms,
                        "detail": soft, "kwargs": kwargs}
            return {"label": label, "status": "PASS", "ms": ms,
                    "detail": shape(result), "kwargs": kwargs}
        except cf.TimeoutError:
            return {"label": label, "status": "TIMEOUT", "ms": TIMEOUT * 1000,
                    "detail": f">{TIMEOUT}s", "kwargs": kwargs}
        except Exception as exc:
            ms = int((time.time() - t0) * 1000)
            tb = traceback.format_exc().strip().splitlines()[-1]
            return {"label": label, "status": "CRASH", "ms": ms,
                    "detail": f"{type(exc).__name__}: {exc} | {tb}"[:400],
                    "kwargs": kwargs}


# ── Tool matrix ──────────────────────────────────────────────────────────────
# (tool_name, source, cases) — cases is list of kwargs dicts.
PER_TICKER = [t for t in TICKERS]

MATRIX = [
    ("search_company", "EDGAR company_tickers.json",
     [{"query": t} for t in PER_TICKER]),
    ("get_filing_list", "EDGAR submissions API",
     [{"ticker_or_cik": t, "form_type": "10-K", "limit": 3} for t in PER_TICKER]),
    ("get_financials", "EDGAR XBRL companyfacts",
     [{"ticker_or_cik": t} for t in PER_TICKER]),
    ("get_financials_batch", "EDGAR XBRL companyfacts",
     [{"tickers": ["AAPL", "MSFT"]}]),
    ("get_income_statement", "EDGAR XBRL companyfacts",
     [{"ticker_or_cik": "AAPL"}, {"ticker_or_cik": "ASML"}]),
    ("get_balance_sheet", "EDGAR XBRL companyfacts",
     [{"ticker_or_cik": "AAPL"}, {"ticker_or_cik": "CRWV"}]),
    ("get_cash_flow", "EDGAR XBRL companyfacts",
     [{"ticker_or_cik": "AAPL"}, {"ticker_or_cik": "SMCI"}]),
    ("get_financial_ratios", "EDGAR XBRL companyfacts",
     [{"ticker_or_cik": t} for t in PER_TICKER]),
    ("get_revenue_segments", "XBRL segments + FMP fallback",
     [{"ticker_or_cik": t} for t in PER_TICKER]),
    ("compare_companies", "EDGAR XBRL companyfacts",
     [{"tickers": ["AAPL", "ASML"]}]),
    ("explain_financials", "Claude API + XBRL",
     [{"ticker_or_cik": "AAPL"}]),
    ("explain_comparison", "Claude API + XBRL",
     []),  # skipped: covered by explain_financials; saves API cost
    ("get_filing_text", "EDGAR filing documents",
     [{"ticker_or_cik": t, "section": "risk_factors"} for t in PER_TICKER]),
    ("analyze_sentiment", "FinBERT or Claude fallback",
     [{"text": "Revenue grew 25% year over year with expanding margins."}]),
    ("summarize_filing", "BART or Claude fallback",
     [{"text": ("The company reported strong quarterly results. " * 30)}]),
    ("extract_entities", "BERT NER or Claude fallback",
     [{"text": "Apple Inc. paid $3.5 billion to Qualcomm in March 2019."}]),
    ("analyze_filing", "NLP combo",
     []),  # skipped: components covered above
    ("get_stock_price", "yfinance",
     [{"ticker": t} for t in PER_TICKER]),
    ("get_valuation_metrics", "yfinance + XBRL",
     [{"ticker": t} for t in PER_TICKER]),
    ("diff_financials", "EDGAR XBRL companyfacts",
     [{"ticker": "AAPL", "year1": 2023, "year2": 2024}]),
    ("diff_filing_section", "EDGAR + Claude",
     [{"ticker": "AAPL", "section": "risk_factors", "year1": 2023, "year2": 2024}]),
    ("find_peers", "SIC map + curated peers",
     [{"ticker": "AAPL"}, {"ticker": "CAVA"}]),
    ("screen_companies", "cached XBRL universe",
     [{"filters": [{"metric": "net_margin", "operator": ">", "value": 25.0}],
       "limit": 5}]),
    ("export_financials", "EDGAR XBRL companyfacts",
     [{"ticker": "AAPL", "format": "csv"}]),
]


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for name, source, cases in MATRIX:
        tool = getattr(S, name, None)
        if tool is None:
            rows.append({"tool": name, "source": source, "results": [
                {"label": name, "status": "MISSING", "ms": 0,
                 "detail": "tool not found in server.py", "kwargs": {}}]})
            continue
        callable_ = fn(tool)
        results = []
        if not cases:
            results.append({"label": name, "status": "SKIPPED", "ms": 0,
                            "detail": "components covered by other tools (API cost)",
                            "kwargs": {}})
        for kwargs in cases:
            tick = kwargs.get("ticker") or kwargs.get("ticker_or_cik") or \
                   ",".join(kwargs.get("tickers", [])) or "—"
            label = f"{name}({tick})"
            print(f"→ {label} ...", flush=True)
            r = run_case(label, callable_, kwargs)
            print(f"   {r['status']} {r['ms']}ms {r['detail'][:120]}", flush=True)
            results.append(r)
        rows.append({"tool": name, "source": source, "results": results})

    # ── Render MCP_AUDIT.md ──────────────────────────────────────────────
    lines = [
        "# MCP_AUDIT.md — fineasmcp (sec-mcp) live tool audit",
        "",
        f"- **Run started (UTC):** {started}",
        "- **Ticker panel:** AAPL (mega-cap) · CAVA (mid-cap, 2023 IPO) · "
        "CRWV (recent IPO Mar-2025) · ASML (foreign private issuer, 20-F) · "
        "SMCI (recent 10:1 split + filing delays)",
        f"- **Timeout per call:** {TIMEOUT}s",
        "",
        "## Summary",
        "",
        "| Tool | Source | Pass | Soft-fail | Crash/Timeout |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        rs = row["results"]
        p = sum(1 for r in rs if r["status"] == "PASS")
        s = sum(1 for r in rs if r["status"] == "SOFT-FAIL")
        c = sum(1 for r in rs if r["status"] in ("CRASH", "TIMEOUT", "MISSING"))
        lines.append(f"| `{row['tool']}` | {row['source']} | {p} | {s} | {c} |")

    lines += ["", "## Failures (exact error + reproduction)", ""]
    any_fail = False
    for row in rows:
        for r in row["results"]:
            if r["status"] in ("SOFT-FAIL", "CRASH", "TIMEOUT", "MISSING"):
                any_fail = True
                repro = (f".venv/bin/python -c \"from sec_mcp import server as S; "
                         f"print(S.{row['tool']}.fn(**{json.dumps(r['kwargs'])}))\"")
                lines += [
                    f"### {r['label']} — {r['status']}",
                    f"- **Error:** `{r['detail']}`",
                    f"- **Repro:** `{repro}`",
                    "",
                ]
    if not any_fail:
        lines.append("_No failures._")

    lines += ["", "## Full results", "",
              "| Call | Status | Latency | Detail |", "|---|---|---|---|"]
    for row in rows:
        for r in row["results"]:
            detail = r["detail"][:160].replace("|", "\\|")  # escape pipes for the md table
            lines.append(
                f"| `{r['label']}` | {r['status']} | {r['ms']}ms | {detail} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
