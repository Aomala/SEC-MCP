"""search_companies — ranked company search with rich filters.

Universe: SEC company_tickers_exchange.json (every public filer, ~10k rows).
Filters: sector, industry, market_cap_min/max, exchange, country,
ipo_date_after, is_sp500. Filters needing per-company enrichment (sector,
country, market cap, IPO date) are applied lazily to the top-ranked
candidates only, so the tool stays fast and SEC-rate-limit friendly.
"""

from __future__ import annotations

# stdlib
import json
import logging
import re
import time
from pathlib import Path

import requests

from sec_mcp.config import get_config

# market data for market-cap enrichment
from sec_mcp.core.realtime_price import get_realtime_price

# shared EDGAR client (rate-limited, cached)
from sec_mcp.sec_client import get_sec_client

# response contract
from sec_mcp.surface.meta import (
    INVALID_INPUT,
    UNAVAILABLE,
    ToolError,
    build_meta,
    parse_iso_date,
    require_number,
    require_pos_int,
)

log = logging.getLogger(__name__)

# How many ranked candidates we enrich (submissions/price fetches) at most —
# keeps worst-case EDGAR traffic bounded well under the 10 req/s limit.
_MAX_ENRICH = 40

# Coarse SIC-range → sector buckets. Specific ranges are checked before broad
# ones (first match wins), so pharma resolves to healthcare not manufacturing.
_SIC_SECTORS: list[tuple[int, int, str]] = [
    (2833, 2836, "healthcare"),       # pharma preparations/diagnostics
    (3826, 3851, "healthcare"),       # lab instruments, medical devices
    (8000, 8099, "healthcare"),       # health services
    (3570, 3579, "technology"),       # computers & office equipment
    (3661, 3699, "technology"),       # comms equipment, semiconductors (3674)
    (7370, 7379, "technology"),       # software & data processing
    (4800, 4899, "communications"),   # telephone, broadcasting, media
    (4900, 4999, "utilities"),        # electric, gas, water
    (1300, 1399, "energy"),           # oil & gas extraction
    (2900, 2999, "energy"),           # petroleum refining
    (6000, 6199, "financials"),       # banks & credit
    (6200, 6299, "financials"),       # brokers & exchanges
    (6300, 6499, "financials"),       # insurance
    (6500, 6599, "real_estate"),      # real estate
    (6798, 6798, "real_estate"),      # REITs
    (6600, 6999, "financials"),       # other finance/holding companies
    (5200, 5999, "consumer"),         # retail
    (2000, 2199, "consumer"),         # food & tobacco
    (5800, 5899, "consumer"),         # restaurants (inside retail range, kept for clarity)
    (1000, 1499, "materials"),        # mining & metals
    (2800, 2899, "materials"),        # chemicals
    (3300, 3399, "materials"),        # primary metals
    (1500, 1799, "industrials"),      # construction
    (3400, 3999, "industrials"),      # remaining manufacturing
    (4000, 4799, "industrials"),      # transportation
    (2200, 3299, "industrials"),      # remaining light manufacturing
    (100, 999, "materials"),          # agriculture/forestry
    (7000, 8999, "consumer"),         # services (hotels, leisure, misc)
]

# Allowed values surfaced to the caller when validation fails
_KNOWN_SECTORS = sorted({s for _, _, s in _SIC_SECTORS})

# Wikipedia constituent list (parsed by regex) backs the is_sp500 filter;
# refreshed weekly, cached on disk so off-line runs still answer.
_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP500_CACHE = Path.home() / ".sec_mcp_cache" / "sp500_constituents.json"
_SP500_TTL = 7 * 86400  # constituents change a few times a year — weekly is plenty


def _sic_to_sector(sic: str | None) -> str | None:
    """Map a 4-digit SIC code to one of our coarse sector buckets."""
    if not sic:
        return None
    try:
        code = int(str(sic)[:4])                          # normalize "3674.0" etc.
    except ValueError:
        return None
    # First matching range wins — list is ordered specific → broad
    for lo, hi, sector in _SIC_SECTORS:
        if lo <= code <= hi:
            return sector
    return None


def _load_sp500() -> set[str]:
    """S&P 500 ticker set: disk cache first, Wikipedia refresh when stale."""
    # Serve the cached set while fresh
    if _SP500_CACHE.exists():
        try:
            blob = json.loads(_SP500_CACHE.read_text())
            if time.time() - blob.get("fetched_at", 0) < _SP500_TTL:
                return set(blob.get("tickers", []))
        except Exception:                                  # corrupt cache → refetch
            pass
    # Refresh from Wikipedia's constituents table
    try:
        resp = requests.get(_SP500_URL, timeout=15,
                            headers={"User-Agent": get_config().edgar_identity})
        resp.raise_for_status()
        # Symbols appear as links like ".../quote/AAPL" or bare <td>AAPL</td>
        # in the first wikitable; this regex targets the symbol column links.
        symbols = set(re.findall(
            r'href="https://www\.nyse\.com/quote/[^"]*?([A-Z][A-Z0-9.\-]{0,6})"|'
            r'href="https://www\.nasdaq\.com/market-activity/stocks/([a-z0-9.\-]+)"',
            resp.text))
        tickers = {(a or b).upper().replace("-", ".") for a, b in symbols if (a or b)}
        # Sanity check: a real constituents page yields ~500 symbols
        if len(tickers) >= 400:
            _SP500_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _SP500_CACHE.write_text(json.dumps(
                {"fetched_at": time.time(), "tickers": sorted(tickers)}))
            return tickers
    except Exception as exc:
        log.warning("S&P 500 list refresh failed: %s", exc)
    # Stale cache beats no data — fall back to whatever we have on disk
    if _SP500_CACHE.exists():
        try:
            return set(json.loads(_SP500_CACHE.read_text()).get("tickers", []))
        except Exception:
            pass
    return set()                                           # caller raises UNAVAILABLE


def _enrich(record: dict, need_profile: bool, need_cap: bool) -> dict:
    """Attach sector/industry/country/IPO-proxy and market cap to a candidate."""
    out = dict(record)                                     # never mutate the universe map
    if need_profile:
        try:
            client = get_sec_client()
            cik = str(record["cik_str"]).zfill(10)         # submissions wants 10 digits
            subs = client._get_submissions(cik)            # cached + rate-limited
            sic = subs.get("sic")                          # 4-digit SIC string
            out["sic"] = sic
            out["industry"] = subs.get("sicDescription")   # EDGAR's industry label
            out["sector"] = _sic_to_sector(sic)            # coarse bucket
            # Country: HQ address description beats state-of-incorporation
            addr = (subs.get("addresses") or {}).get("business") or {}
            out["country"] = (addr.get("stateOrCountryDescription")
                              or subs.get("stateOfIncorporationDescription") or "")
            # IPO proxy: EDGAR first-filing date — the oldest pagination file's
            # filingFrom, else the oldest date in the recent window.
            pages = (subs.get("filings") or {}).get("files") or []
            if pages:
                out["first_filed"] = min(p.get("filingFrom", "9999") for p in pages)
            else:
                dates = ((subs.get("filings") or {}).get("recent") or {}).get("filingDate") or []
                out["first_filed"] = min(dates) if dates else None
        except Exception as exc:                           # enrichment is best-effort
            log.debug("profile enrichment failed for %s: %s", record.get("ticker"), exc)
    if need_cap:
        try:
            # Provider chain returns market_cap from yfinance/FMP when known
            q = get_realtime_price(record["ticker"])
            out["market_cap"] = q.get("market_cap")
        except Exception:
            out["market_cap"] = None
    return out


def search_companies_impl(query, filters=None, limit=None) -> dict:
    """Core implementation: rank the universe, then filter the leaders."""
    t0 = time.time()                                       # latency clock
    limit = require_pos_int(limit, "limit", default=10, hi=50)
    f = dict(filters or {})                                # tolerate None

    # ── validate filters up front (malformed input → clear error) ───────
    allowed = {"sector", "industry", "market_cap_min", "market_cap_max",
               "exchange", "country", "ipo_date_after", "is_sp500"}
    unknown = set(f) - allowed
    if unknown:
        raise ToolError(INVALID_INPUT, f"Unknown filter(s): {sorted(unknown)}.",
                        f"Supported filters: {sorted(allowed)}.")
    cap_min = require_number(f.get("market_cap_min"), "market_cap_min")
    cap_max = require_number(f.get("market_cap_max"), "market_cap_max")
    ipo_after = parse_iso_date(f.get("ipo_date_after"), "ipo_date_after")
    sector = f.get("sector")
    if sector is not None and str(sector).lower() not in _KNOWN_SECTORS:
        raise ToolError(INVALID_INPUT, f"Unknown sector {sector!r}.",
                        f"Supported sectors: {_KNOWN_SECTORS}.")
    # Query may be empty ONLY when at least one filter narrows the universe
    q = (query or "").strip()
    if not q and not f:
        raise ToolError(INVALID_INPUT, "Provide a query and/or at least one filter.",
                        "Example: search_companies('nvidia') or "
                        "search_companies('', {'sector': 'technology', 'is_sp500': True}).")

    # ── rank candidates from the full filer universe ─────────────────────
    client = get_sec_client()
    universe = client._get_tickers_map()                   # ~10k filers, 30-min cached
    qu = q.upper()
    scored: list[tuple[float, dict]] = []
    for key, rec in universe.items():
        if key.startswith("CIK:"):                         # skip reverse-lookup keys
            continue
        if not q:                                          # filter-only mode: everyone scores equal
            scored.append((1.0, rec))
            continue
        tick = rec.get("ticker", "")
        title = rec.get("title", "").upper()
        # Simple deterministic ranking: exact ticker ≫ ticker prefix ≫
        # name word-prefix ≫ name substring
        if tick == qu:
            scored.append((100.0, rec))
        elif tick.startswith(qu):
            scored.append((60.0 - len(tick), rec))
        elif any(w.startswith(qu) for w in title.split()):
            scored.append((40.0, rec))
        elif qu in title:
            scored.append((20.0, rec))
    scored.sort(key=lambda s: -s[0])                       # best first

    # ── cheap filters first (no network): exchange, is_sp500 ────────────
    if f.get("exchange") is not None:
        want_exch = str(f["exchange"]).strip().lower()
        scored = [(s, r) for s, r in scored
                  if (r.get("exchange") or "").lower() == want_exch]
    if f.get("is_sp500"):
        sp500 = _load_sp500()
        if not sp500:                                      # no list + no cache → honest error
            raise ToolError(UNAVAILABLE, "S&P 500 constituent list is unavailable (fetch failed, no cache).",
                            "Retry later, or drop the is_sp500 filter.")
        scored = [(s, r) for s, r in scored
                  if r.get("ticker", "").replace("-", ".") in sp500]

    # ── expensive filters on the leaders only ────────────────────────────
    need_profile = any(k in f for k in ("sector", "industry", "country", "ipo_date_after"))
    need_cap = cap_min is not None or cap_max is not None
    results: list[dict] = []
    enriched_count = 0
    for score, rec in scored:
        # Bound total enrichment work regardless of universe size
        if (need_profile or need_cap) and enriched_count >= _MAX_ENRICH:
            break
        row = _enrich(rec, need_profile, need_cap) if (need_profile or need_cap) else dict(rec)
        if need_profile or need_cap:
            enriched_count += 1
        # Apply enrichment-dependent filters
        if sector is not None and (row.get("sector") or "") != str(sector).lower():
            continue
        if f.get("industry") is not None and \
                str(f["industry"]).lower() not in (row.get("industry") or "").lower():
            continue
        if f.get("country") is not None and \
                str(f["country"]).lower() not in (row.get("country") or "").lower():
            continue
        if ipo_after is not None:
            first = row.get("first_filed")
            # Unknown first-filing date → cannot prove it's recent → exclude
            if not first or parse_iso_date(first[:10], "first_filed") <= ipo_after:
                continue
        mc = row.get("market_cap")
        if cap_min is not None and (mc is None or mc < cap_min):
            continue
        if cap_max is not None and (mc is None or mc > cap_max):
            continue
        # Shape the output row — CIK + ticker always present per the contract
        results.append({
            "ticker": row.get("ticker"),
            "name": row.get("title"),
            "cik": str(row.get("cik_str", "")).zfill(10),
            "exchange": row.get("exchange") or None,
            "score": round(score, 1),
            # enrichment fields are present only when a filter requested them
            **({"sector": row.get("sector"), "industry": row.get("industry"),
                "country": row.get("country")} if need_profile else {}),
            **({"marketCap": row.get("market_cap")} if need_cap else {}),
            **({"firstFiledDate": row.get("first_filed")} if ipo_after else {}),
        })
        if len(results) >= limit:
            break

    return {
        "query": q,                                        # echo for traceability
        "filters": f,
        "count": len(results),
        "results": results,
        # Honest coverage note: enrichment-bounded searches are not exhaustive
        "note": (f"Enriched filters evaluated on the top {_MAX_ENRICH} ranked candidates."
                 if (need_profile or need_cap) else None),
        "meta": build_meta("edgar:company_tickers_exchange", t0, cache_hit=False),
    }
