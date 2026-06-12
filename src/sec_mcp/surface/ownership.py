"""get_insider_activity + get_ownership — who's buying, who holds, who's agitating.

Insider activity: parsed Form 4s straight from EDGAR (existing parser),
filtered by date, normalized to buy/sell sides, with post-transaction
holdings.

Ownership: 13F institutional holders (yfinance's aggregated 13F dataset —
inverting raw 13Fs across every institution is not feasible per-call) plus
SC 13D/13G activist/blockholder filings from the subject company's own
EDGAR index, with filing dates and direct URLs.
"""

from __future__ import annotations

# stdlib
import logging
import time

# existing rate-limited Form 4 parser
from sec_mcp.insider_tracker import get_insider_transactions

# EDGAR client for 13D/G discovery
from sec_mcp.sec_client import get_sec_client

# response contract
from sec_mcp.surface.meta import (
    NOT_FOUND,
    UNKNOWN_TICKER,
    ToolError,
    build_meta,
    parse_iso_date,
    require_pos_int,
    require_ticker,
)

log = logging.getLogger(__name__)

# Form 4 human-readable transaction types → normalized trade side
_BUY_TYPES = {"Purchase"}
_SELL_TYPES = {"Sale", "Disposition pursuant to tender offer"}


def _side(txn_type: str) -> str:
    """Normalize a Form 4 transaction label to buy / sell / other."""
    if txn_type in _BUY_TYPES:
        return "buy"
    if txn_type in _SELL_TYPES:
        return "sell"
    return "other"                                         # grants, exercises, gifts…


def get_insider_activity_impl(ticker, date_from=None, limit=None) -> dict:
    """Core implementation for the get_insider_activity tool."""
    t0 = time.time()                                       # latency clock
    tk = require_ticker(ticker, "ticker")
    d_from = parse_iso_date(date_from, "date_from")        # optional cutoff
    limit = require_pos_int(limit, "limit", default=50, hi=200)

    # Fetch extra rows when date-filtering so the cutoff doesn't starve output
    raw = get_insider_transactions(tk, limit=limit * 2 if d_from else limit)
    if raw is None:
        raw = []
    txns: list[dict] = []
    for r in raw:
        # Apply the date_from filter on the transaction date (ISO strings sort)
        if d_from and (r.get("transaction_date") or "") < d_from.isoformat():
            continue
        txns.append({
            "insiderName": r.get("insider_name"),
            "role": r.get("title") or None,                # CEO / CFO / Director / 10% Owner
            "side": _side(r.get("transaction_type", "")),  # buy / sell / other
            "transactionType": r.get("transaction_type"),  # original Form 4 label
            "date": r.get("transaction_date"),
            "shares": r.get("shares"),
            "price": r.get("price_per_share"),
            "value": r.get("total_value"),
            "sharesOwnedAfter": r.get("shares_owned_after"),  # post-transaction holdings
            "filingDate": r.get("filing_date"),
            "accession": r.get("accession"),
        })
        if len(txns) >= limit:
            break
    # Aggregate the window so callers get a signal, not just rows
    buys = [t for t in txns if t["side"] == "buy"]
    sells = [t for t in txns if t["side"] == "sell"]
    return {
        "ticker": tk,
        "dateFrom": d_from.isoformat() if d_from else None,
        "count": len(txns),
        "summary": {
            "buyCount": len(buys),
            "sellCount": len(sells),
            "netShares": sum(t["shares"] or 0 for t in buys)
                       - sum(t["shares"] or 0 for t in sells),
        },
        "transactions": txns,
        # Empty is a valid answer (no insider trades in window) — not an error
        "note": None if txns else "No Form 4 transactions in the requested window.",
        "meta": build_meta("edgar:form4", t0, cache_hit=False),
    }


def _institutional_13f(ticker: str) -> tuple[list[dict], str]:
    """Top 13F holders via yfinance's aggregated institutional dataset."""
    try:
        import yfinance as yf  # optional dependency
        # yfinance wants '-' for class shares
        df = yf.Ticker(ticker.replace(".", "-")).institutional_holders
        if df is None or df.empty:
            return [], "unavailable"
        holders: list[dict] = []
        for _, row in df.iterrows():
            holders.append({
                "institution": str(row.get("Holder", "")),
                # Date Reported = the 13F period the position is as-of
                "reportDate": str(row.get("Date Reported", ""))[:10] or None,
                "shares": int(row["Shares"]) if row.get("Shares") == row.get("Shares") else None,
                "value": float(row["Value"]) if row.get("Value") == row.get("Value") else None,
                "pctHeld": round(float(row["pctHeld"]) * 100, 2)
                           if "pctHeld" in row and row["pctHeld"] == row["pctHeld"] else None,
            })
        return holders, "ok"
    except Exception as exc:
        log.debug("13F holders via yfinance failed for %s: %s", ticker, exc)
        return [], "unavailable"


def _activist_13dg(cik: str, limit: int = 20) -> list[dict]:
    """SC 13D/13G filings from the subject company's own EDGAR index.

    EDGAR cross-indexes beneficial-ownership filings under the SUBJECT
    company, so its submissions feed lists every 13D/G filed about it.
    """
    client = get_sec_client()
    subs = client._get_submissions(cik)                    # cached + rate-limited
    recent = (subs.get("filings") or {}).get("recent") or {}
    accs = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    docs = recent.get("primaryDocument") or []
    out: list[dict] = []
    cik_raw = str(int(cik))
    for i in range(len(accs)):
        form = forms[i] if i < len(forms) else ""
        # Only beneficial-ownership forms (13D = activist intent, 13G = passive)
        if not form.startswith(("SC 13D", "SC 13G", "SCHEDULE 13D", "SCHEDULE 13G")):
            continue
        acc_nodash = accs[i].replace("-", "")
        doc = docs[i] if i < len(docs) and docs[i] else ""
        out.append({
            "form": form,                                  # SC 13D / SC 13G (+ /A amendments)
            "kind": "activist" if "13D" in form else "passive",
            "filingDate": dates[i] if i < len(dates) else None,
            "accession": accs[i],
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_nodash}/{doc}"
                   if doc else f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc_nodash}/",
        })
        if len(out) >= limit:
            break
    return out


def get_ownership_impl(ticker) -> dict:
    """Core implementation for the get_ownership tool."""
    t0 = time.time()                                       # latency clock
    tk = require_ticker(ticker, "ticker")
    try:
        cik = get_sec_client().resolve_cik(tk)             # raises on unknown ticker
    except Exception:
        raise ToolError(UNKNOWN_TICKER, f"Could not resolve {tk!r} to a CIK.",
                        "Use search_companies to find the right symbol.") from None

    holders, holders_status = _institutional_13f(tk)       # 13F aggregate
    activists = _activist_13dg(cik)                        # 13D/G from EDGAR

    # Both sources empty AND holders provider down → tell the caller honestly
    if not holders and not activists and holders_status != "ok":
        raise ToolError(NOT_FOUND,
                        f"No ownership data available for {tk} right now.",
                        "13F data comes from an aggregated provider that may be "
                        "throttling; retry in a minute. 13D/G filings appear only "
                        "if a >5% blockholder has filed.")
    return {
        "ticker": tk,
        "cik": cik.zfill(10),
        "institutionalHolders": {                          # latest 13F snapshot
            "status": holders_status,
            "source": "yfinance(13F aggregate)",
            "holders": holders,
        },
        "beneficialOwners": {                              # 13D/G blockholders
            "source": "edgar:subject-company SC 13D/G index",
            "filings": activists,
            "note": None if activists else
                    "No SC 13D/13G filings on record — no >5% outside blockholder has filed.",
        },
        "meta": build_meta("yfinance(13F)+edgar(13D/G)", t0, cache_hit=False),
    }
