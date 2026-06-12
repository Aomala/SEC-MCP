"""screen — composable screener over the curated ticker universe.

Filter groups (all AND-ed together):
  valuation: pe_max, ev_ebitda_max
  growth:    rev_growth_min            (decimal: 0.15 = 15% YoY)
  quality:   fcf_positive, net_debt_ebitda_max
  events:    filed_8k_last_7d, insider_buying_last_30d
  scoping:   sector, market_cap_min, market_cap_max

Evaluation is staged cheapest-first so expensive lookups only run on
survivors: cached XBRL fundamentals → live price/valuation → EDGAR events.
The response always reports how much of the universe was evaluated —
no silent truncation.
"""

from __future__ import annotations

# stdlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

# curated 300+ ticker universe grouped by sector
from sec_mcp.chat_app import SECTOR_UNIVERSE

# live prices for valuation filters
from sec_mcp.core.realtime_price import get_realtime_price

# XBRL extraction (disk-cached per accession — warm universe screens fast)
from sec_mcp.financials import extract_financials

# Form 4 parser for the insider-buying event filter
from sec_mcp.insider_tracker import get_insider_transactions

# EDGAR submissions for the 8-K event filter
from sec_mcp.sec_client import get_sec_client

# response contract
from sec_mcp.surface.meta import (
    INVALID_INPUT,
    ToolError,
    build_meta,
    require_number,
    require_pos_int,
)

log = logging.getLogger(__name__)

# Every filter key the tool accepts — anything else is a structured error
_ALLOWED = {"pe_max", "ev_ebitda_max", "rev_growth_min", "fcf_positive",
            "net_debt_ebitda_max", "filed_8k_last_7d", "insider_buying_last_30d",
            "sector", "market_cap_min", "market_cap_max"}
# Hard ceilings keeping one screen call inside SEC rate limits
_MAX_CANDIDATES = 80                                       # fundamentals evaluations
_MAX_EVENT_CHECKS = 40                                     # EDGAR event evaluations


def _fundamentals_row(ticker: str) -> dict | None:
    """One ticker's screening metrics from cached XBRL (None on failure)."""
    try:
        # include_statements gives prior_value rows → YoY revenue growth
        data = extract_financials(ticker, include_statements=True)
        if not data or data.get("error"):
            return None
        m = data.get("metrics") or {}
        # YoY revenue growth from the income statement's revenue row
        growth = None
        for row in data.get("income_statement") or []:
            label = (row.get("concept") or row.get("label") or "").lower()
            if "revenue" in label and row.get("value") and row.get("prior_value"):
                growth = (row["value"] - row["prior_value"]) / abs(row["prior_value"])
                break
        debt = (m.get("long_term_debt") or 0) + (m.get("short_term_debt") or 0)
        cash = m.get("cash_and_equivalents") or 0
        return {
            "ticker": ticker,
            "name": data.get("company_name"),
            "fiscalYear": data.get("fiscal_year"),
            "revenue": m.get("revenue"),
            "netIncome": m.get("net_income"),
            "ebitda": m.get("ebitda"),
            "fcf": m.get("free_cash_flow"),
            "revGrowth": round(growth, 4) if growth is not None else None,
            "netDebt": debt - cash,                        # net of cash on hand
            # shares let us derive market cap when the quote provider omits it
            "shares": m.get("shares_outstanding"),
        }
    except Exception as exc:                               # one bad ticker never kills a screen
        log.debug("screen fundamentals failed for %s: %s", ticker, exc)
        return None


def _passes_fundamentals(row: dict, f: dict) -> bool:
    """Apply the cached-data filter group. Missing data → conservative fail."""
    # Unknown FCF can't prove positivity → excluded
    if f.get("fcf_positive") and (row["fcf"] is None or row["fcf"] <= 0):
        return False
    if f.get("rev_growth_min") is not None and (
            row["revGrowth"] is None or row["revGrowth"] < f["rev_growth_min"]):
        return False
    if f.get("net_debt_ebitda_max") is not None:
        ebitda = row["ebitda"]
        # Leverage undefined without positive EBITDA → excluded
        if not ebitda or ebitda <= 0:
            return False
        if row["netDebt"] / ebitda > f["net_debt_ebitda_max"]:
            return False
    return True


def _passes_valuation(row: dict, f: dict) -> bool:
    """Apply price-dependent filters; fetches one quote per survivor."""
    needs = any(f.get(k) is not None for k in
                ("pe_max", "ev_ebitda_max", "market_cap_min", "market_cap_max"))
    if not needs:
        return True                                        # skip the quote entirely
    q = get_realtime_price(row["ticker"])
    cap = q.get("market_cap")
    # Polygon snapshots omit market cap — derive it from price × shares
    # (XBRL weighted-average shares) so valuation filters still work
    if cap is None and q.get("price") and row.get("shares"):
        cap = q["price"] * row["shares"]
    row["marketCap"] = cap                                 # surface it in results
    if f.get("market_cap_min") is not None and (cap is None or cap < f["market_cap_min"]):
        return False
    if f.get("market_cap_max") is not None and (cap is None or cap > f["market_cap_max"]):
        return False
    if f.get("pe_max") is not None:
        ni = row["netIncome"]
        # P/E needs positive earnings AND a market cap
        if not cap or not ni or ni <= 0:
            return False
        pe = cap / ni
        row["pe"] = round(pe, 1)
        if pe > f["pe_max"]:
            return False
    if f.get("ev_ebitda_max") is not None:
        ebitda = row["ebitda"]
        if not cap or not ebitda or ebitda <= 0:
            return False
        ev = cap + (row["netDebt"] or 0)                   # EV = cap + net debt
        row["evEbitda"] = round(ev / ebitda, 1)
        if row["evEbitda"] > f["ev_ebitda_max"]:
            return False
    return True


def _passes_events(row: dict, f: dict) -> bool:
    """Apply EDGAR event filters — the most expensive group, run last."""
    now = datetime.now(timezone.utc).date()
    if f.get("filed_8k_last_7d"):
        cutoff = (now - timedelta(days=7)).isoformat()
        try:
            subs = get_sec_client()._get_submissions(
                get_sec_client().resolve_cik(row["ticker"]))
            recent = (subs.get("filings") or {}).get("recent") or {}
            forms = recent.get("form") or []
            dates = recent.get("filingDate") or []
            # Any 8-K with a filing date inside the window passes
            hit = any(fm.startswith("8-K") and dt >= cutoff
                      for fm, dt in zip(forms, dates, strict=False))
            if not hit:
                return False
            row["recent8K"] = True
        except Exception:
            return False                                   # can't verify → exclude
    if f.get("insider_buying_last_30d"):
        cutoff = (now - timedelta(days=30)).isoformat()
        try:
            txns = get_insider_transactions(row["ticker"], limit=20) or []
            buys = [t for t in txns
                    if t.get("transaction_type") == "Purchase"
                    and (t.get("transaction_date") or "") >= cutoff]
            if not buys:
                return False
            row["insiderBuys30d"] = len(buys)
        except Exception:
            return False
    return True


def screen_impl(filters, limit=None) -> dict:
    """Core implementation for the screen tool."""
    t0 = time.time()                                       # latency clock
    limit = require_pos_int(limit, "limit", default=20, hi=50)
    # filters must be a dict — a list (old screener style) is a clear error
    if not isinstance(filters, dict) or not filters:
        raise ToolError(INVALID_INPUT, "'filters' must be a non-empty object.",
                        "Example: screen({'pe_max': 25, 'fcf_positive': True, "
                        "'sector': 'semiconductors'}).")
    unknown = set(filters) - _ALLOWED
    if unknown:
        raise ToolError(INVALID_INPUT, f"Unknown filter(s): {sorted(unknown)}.",
                        f"Supported: {sorted(_ALLOWED)}.")
    # Validate every numeric threshold up front (malformed → structured error)
    f = {
        "pe_max": require_number(filters.get("pe_max"), "pe_max"),
        "ev_ebitda_max": require_number(filters.get("ev_ebitda_max"), "ev_ebitda_max"),
        "rev_growth_min": require_number(filters.get("rev_growth_min"), "rev_growth_min"),
        "net_debt_ebitda_max": require_number(filters.get("net_debt_ebitda_max"), "net_debt_ebitda_max"),
        "market_cap_min": require_number(filters.get("market_cap_min"), "market_cap_min"),
        "market_cap_max": require_number(filters.get("market_cap_max"), "market_cap_max"),
        "fcf_positive": bool(filters.get("fcf_positive")),
        "filed_8k_last_7d": bool(filters.get("filed_8k_last_7d")),
        "insider_buying_last_30d": bool(filters.get("insider_buying_last_30d")),
    }

    # ── build the candidate universe (sector scoping first) ─────────────
    sector = filters.get("sector")
    if sector is not None:
        sector_key = str(sector).strip().lower()
        if sector_key not in SECTOR_UNIVERSE:
            raise ToolError(INVALID_INPUT, f"Unknown sector {sector!r}.",
                            f"Available sectors: {sorted(SECTOR_UNIVERSE)}.")
        candidates = list(SECTOR_UNIVERSE[sector_key])
    else:
        # Flatten all sectors, dedupe, keep deterministic order
        seen: set[str] = set()
        candidates = []
        for ticks in SECTOR_UNIVERSE.values():
            for tk in ticks:
                if tk not in seen:
                    seen.add(tk)
                    candidates.append(tk)
    universe_size = len(candidates)
    candidates = candidates[:_MAX_CANDIDATES]              # bounded evaluation

    # ── stage 1: cached fundamentals in parallel ─────────────────────────
    with ThreadPoolExecutor(max_workers=8) as ex:          # SEC limiter serializes I/O
        rows = [r for r in ex.map(_fundamentals_row, candidates) if r]
    rows = [r for r in rows if _passes_fundamentals(r, f)]

    # ── stage 2: valuation (one quote per survivor) ──────────────────────
    rows = [r for r in rows if _passes_valuation(r, f)]

    # ── stage 3: EDGAR events (bounded, survivors only) ──────────────────
    has_events = f["filed_8k_last_7d"] or f["insider_buying_last_30d"]
    event_checked = 0
    if has_events:
        survivors = []
        for r in rows:
            if event_checked >= _MAX_EVENT_CHECKS:
                break                                      # honest cap, reported below
            event_checked += 1
            if _passes_events(r, f):
                survivors.append(r)
            if len(survivors) >= limit:
                break
        rows = survivors

    rows = rows[:limit]
    return {
        "filters": filters,                                # echo for traceability
        "count": len(rows),
        "matches": rows,
        # Coverage block — no silent caps, ever
        "coverage": {
            "universeSize": universe_size,
            "candidatesEvaluated": min(universe_size, _MAX_CANDIDATES),
            "eventChecksRun": event_checked if has_events else None,
            "note": (f"Universe capped at {_MAX_CANDIDATES} candidates per call; "
                     f"scope with 'sector' to cover a specific slice fully.")
                    if universe_size > _MAX_CANDIDATES else None,
        },
        "meta": build_meta("edgar:xbrl(+price providers, edgar events)", t0,
                           cache_hit=False),
    }
