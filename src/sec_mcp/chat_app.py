"""SEC Terminal — split-panel chat + analysis dashboard.

Modern fintech-style UI with:
  - Chat panel (left 35%) — natural language queries with thinking/tool transparency
  - Analysis panel (right 65%) — financial data, charts, tables
  - Chart.js for interactive visualizations
  - MongoDB-backed historical data
  - Anthropic Claude-powered Q&A over financial data
  - Entity profile from SEC submissions
  - MD&A section extraction + display

Run:  python -m sec_mcp.chat_app
Open: http://localhost:{PORT}  (default 8877)
"""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sec_mcp.edgar_client import get_filing_content, list_filings, search_companies
from sec_mcp.financials import (
    extract_financials,
    extract_financials_batch,
    generate_local_summary,
)
from sec_mcp.historical import get_historical_data, run_historical_extraction
from sec_mcp.db import get_job, is_available as db_available
from sec_mcp.intent_parser import learn_company, parse_intent, resolve_name
from sec_mcp import disk_cache

log = logging.getLogger(__name__)

# ── Server-side result cache ──────────────────────────────────────────────────
# Keyed by (ticker_upper, accession) → {data, summary, ts}
# Prevents re-processing XBRL on every period switch for the same company.
_result_cache: dict[str, dict] = {}
_RESULT_CACHE_TTL = 600  # 10 minutes

def _rcache_key(ticker: str, accession: str) -> str:
    return f"{ticker.upper()}|{accession}"

def _rcache_get(ticker: str, accession: str) -> dict | None:
    entry = _result_cache.get(_rcache_key(ticker, accession))
    if entry and (time.time() - entry["ts"]) < _RESULT_CACHE_TTL:
        return entry
    return None

def _rcache_put(ticker: str, accession: str, data: dict, summary: str) -> None:
    _result_cache[_rcache_key(ticker, accession)] = {
        "data": data, "summary": summary, "ts": time.time()
    }

app = FastAPI(title="SEC Terminal", docs_url="/swagger", redoc_url="/redoc")

# CORS for cross-origin access (needed for Railway deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    """Non-blocking startup: init DB connection in background thread."""
    import threading

    def _warmup():
        try:
            db_available()
            log.info("DB warmup complete")
        except Exception as exc:
            log.warning("DB warmup failed (non-fatal): %s", exc)

    threading.Thread(target=_warmup, daemon=True).start()
    log.info("SEC Terminal ready (DB warming up in background)")

# Serve static assets (CSS, JS)
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str


class LoadFilingRequest(BaseModel):
    ticker: str
    accession: str
    form_type: str = "10-K"


class ChatbotRequest(BaseModel):
    message: str
    ticker: str = ""
    context: dict = {}
    history: list = []  # conversation history: [{role: "user"/"assistant", content: "..."}]


class CompsRequest(BaseModel):
    tickers: list[str]
    year: int | None = None
    form_type: str = "10-K"


# ── Peer comparison map ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
#  Rules-Based Comparable Company Engine
#
#  Matching rules (applied in order):
#    1. Sector match — same SECTOR_UNIVERSE bucket
#    2. Size filter — revenue within 0.2x–5x of target
#    3. Rank by revenue proximity (closest size = best comp)
#    4. Return top 5 comps
#
#  If target ticker is unknown or has no revenue data,
#  falls back to the static PEER_MAP.
# ═══════════════════════════════════════════════════════════════════════════

# Sector universe: sector_id → list of tickers (ordered roughly by size)
# Each ticker appears in exactly one sector.
SECTOR_UNIVERSE: dict[str, list[str]] = {
    "mega_tech": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NFLX"],
    "semiconductors": [
        "NVDA", "TSM", "AVGO", "ASML", "AMD", "QCOM", "TXN", "INTC",
        "AMAT", "LRCX", "KLAC", "MRVL", "ADI", "NXPI", "MU", "ON",
        "MCHP", "SWKS", "MPWR", "TER", "ENTG", "WOLF",
    ],
    "enterprise_software": [
        "ORCL", "SAP", "CRM", "ADBE", "IBM", "INTU", "NOW", "WDAY",
        "SNPS", "CDNS", "ANSS", "PLTR", "TEAM", "HUBS", "DDOG", "MDB",
        "SNOW", "ZS", "NET", "VEEV", "BILL", "TTD", "ESTC", "DOCN",
    ],
    "cybersecurity": ["PANW", "CRWD", "FTNT", "ZS", "OKTA", "S", "QLYS", "TENB", "RPD"],
    "it_services": ["ACN", "CSCO", "HPE", "HPQ", "DELL", "CDW", "LDOS", "SAIC"],
    "fintech_payments": [
        "V", "MA", "PYPL", "AXP", "SQ", "FIS", "FISV", "GPN",
        "AFRM", "COIN", "TOST", "BILL", "FOUR", "RPAY",
    ],
    "banks_mega": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "banks_regional": [
        "USB", "PNC", "TFC", "COF", "SCHW", "BK", "STT", "FITB",
        "KEY", "MTB", "HBAN", "RF", "CFG", "ZION", "CMA", "ALLY",
    ],
    "insurance": [
        "BRK-B", "BRK-A", "ALL", "PGR", "MET", "AIG", "PRU", "AFL",
        "TRV", "HIG", "CB", "CINF", "GL", "RGA", "EG", "WRB",
    ],
    "pharma_large": [
        "LLY", "JNJ", "ABBV", "MRK", "PFE", "AZN", "NVO", "BMY",
        "AMGN", "GILD", "REGN", "VRTX", "SNY", "GSK", "TAK",
    ],
    "biotech": [
        "MRNA", "BNTX", "BIIB", "SGEN", "ALNY", "BMRN", "INCY",
        "SAREPTA", "IONS", "PCVX", "EXEL", "RARE", "HALO",
    ],
    "healthcare_services": [
        "UNH", "ELV", "CI", "HUM", "CNC", "MOH",
        "CVS", "WBA", "HCA", "THC", "UHS",
    ],
    "medtech": [
        "ABT", "MDT", "SYK", "BSX", "ISRG", "EW", "ZBH",
        "DXCM", "ALGN", "HOLX", "BAX", "BDX",
    ],
    "energy_majors": ["XOM", "CVX", "SHEL", "BP", "TTE", "COP", "EOG", "SLB"],
    "energy_ep": [
        "PXD", "DVN", "FANG", "MPC", "VLO", "PSX", "HES",
        "OXY", "HAL", "BKR", "CTRA",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "AEP", "D", "SRE", "EXC", "XEL",
        "WEC", "ED", "ES", "DTE", "PPL", "FE", "CMS", "AES",
    ],
    "telecom": ["T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN"],
    "retail_broadline": ["WMT", "AMZN", "COST", "TGT", "DG", "DLTR", "BJ", "KR"],
    "retail_specialty": [
        "HD", "LOW", "TJX", "ROST", "BURL", "ULTA", "BBY",
        "FIVE", "ORLY", "AZO", "AAP", "WSM", "RH",
    ],
    "restaurants": ["MCD", "SBUX", "CMG", "YUM", "DPZ", "QSR", "DINE", "SHAK", "WING", "CAVA"],
    "consumer_staples": [
        "PG", "KO", "PEP", "UL", "CL", "MDLZ", "KHC", "GIS",
        "SJM", "HSY", "CPB", "CAG", "KDP", "MNST", "CLX", "KMB",
    ],
    "auto": ["TSLA", "TM", "GM", "F", "STLA", "HMC", "RIVN", "LCID", "NIO", "LI", "XPEV"],
    "aerospace_defense": ["BA", "LMT", "RTX", "NOC", "GD", "LHX", "HII", "TXT", "HWM"],
    "industrials": [
        "GE", "HON", "MMM", "CAT", "DE", "EMR", "ETN", "ROK",
        "ITW", "PH", "CMI", "DOV", "IR", "AME",
    ],
    "reits": [
        "PLD", "AMT", "EQIX", "CCI", "SPG", "O", "DLR", "PSA",
        "WELL", "AVB", "EQR", "VTR", "ARE", "SUI", "MAA", "WPC",
        "SBAC", "IRM", "VICI", "INVH", "GLPI",
    ],
    "media_entertainment": [
        "DIS", "WBD", "PARA", "CMCSA", "FOX", "NWSA", "LYV",
        "SPOT", "ROKU", "IMAX", "MSGS",
    ],
    "airlines": ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "SAVE", "HA"],
    "crypto": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "WULF", "CIFR"],
    "materials_mining": [
        "NEM", "GOLD", "AEM", "FNV", "WPM", "FCX", "BHP", "RIO",
        "NUE", "STLD", "CLF", "X", "AA",
    ],
    "logistics": ["UPS", "FDX", "XPO", "JBHT", "ODFL", "CHRW", "EXPD", "SAIA"],
}

# Build reverse lookup: ticker → sector_id
_TICKER_TO_SECTOR: dict[str, str] = {}
for _sect, _tickers in SECTOR_UNIVERSE.items():
    for _t in _tickers:
        _TICKER_TO_SECTOR[_t] = _sect

# Revenue estimates (in billions) for size-tier matching
# These are approximate and used ONLY for sizing comps when we can't fetch live data.
# Updated periodically. If a ticker isn't here, we fetch from XBRL.
_REVENUE_ESTIMATES_B: dict[str, float] = {
    # Mega tech
    "AAPL": 390, "MSFT": 245, "GOOG": 340, "GOOGL": 340, "AMZN": 640, "META": 160, "NFLX": 39,
    # Semis
    "NVDA": 130, "TSM": 90, "AVGO": 50, "ASML": 30, "AMD": 24, "QCOM": 38, "TXN": 18,
    "INTC": 54, "AMAT": 27, "LRCX": 15, "KLAC": 11, "MRVL": 6, "ADI": 12, "NXPI": 13,
    "MU": 25, "ON": 8, "MCHP": 8, "SWKS": 5, "MPWR": 2, "TER": 3,
    # Enterprise SW
    "ORCL": 53, "SAP": 35, "CRM": 35, "ADBE": 21, "IBM": 62, "INTU": 16, "NOW": 10,
    "WDAY": 8, "SNPS": 6, "CDNS": 4, "PLTR": 3, "TEAM": 4, "HUBS": 2, "DDOG": 2,
    "SNOW": 3, "ZS": 2, "NET": 2, "VEEV": 2, "MDB": 2, "TTD": 2,
    # Cybersecurity
    "PANW": 8, "CRWD": 4, "FTNT": 6, "OKTA": 2, "ZS": 2, "S": 1, "QLYS": 0.6,
    # IT Services
    "ACN": 65, "CSCO": 57, "HPE": 30, "HPQ": 54, "DELL": 102, "CDW": 21,
    # Fintech
    "V": 35, "MA": 27, "PYPL": 30, "AXP": 60, "SQ": 22, "FIS": 15, "FISV": 20, "GPN": 10,
    "COIN": 5, "AFRM": 2, "TOST": 4,
    # Banks
    "JPM": 170, "BAC": 100, "WFC": 82, "C": 78, "GS": 50, "MS": 55,
    "USB": 24, "PNC": 22, "TFC": 23, "COF": 37, "SCHW": 20, "BK": 18, "STT": 12,
    # Insurance
    "BRK-B": 370, "BRK-A": 370, "ALL": 57, "PGR": 62, "MET": 70, "AIG": 46,
    "PRU": 60, "AFL": 20, "TRV": 42, "CB": 45,
    # Pharma
    "LLY": 42, "JNJ": 85, "ABBV": 58, "MRK": 60, "PFE": 58, "AZN": 46, "NVO": 33,
    "BMY": 45, "AMGN": 28, "GILD": 27, "REGN": 14, "VRTX": 10,
    # Biotech
    "MRNA": 7, "BNTX": 4, "BIIB": 10,
    # Healthcare Services
    "UNH": 370, "ELV": 170, "CI": 230, "HUM": 110, "CNC": 154,
    "CVS": 360, "WBA": 140, "HCA": 65,
    # Energy
    "XOM": 350, "CVX": 200, "SHEL": 380, "BP": 220, "TTE": 220, "COP": 60, "EOG": 23,
    "SLB": 36, "PXD": 20, "DVN": 15, "MPC": 150, "VLO": 145, "PSX": 150,
    # Utilities
    "NEE": 28, "DUK": 29, "SO": 25, "AEP": 19, "D": 15, "SRE": 17, "EXC": 22,
    # Telecom
    "T": 122, "VZ": 134, "TMUS": 80, "CMCSA": 122, "CHTR": 55,
    # Retail
    "WMT": 650, "COST": 250, "TGT": 107, "DG": 40, "DLTR": 30, "KR": 150,
    "HD": 157, "LOW": 87, "TJX": 54, "BBY": 43,
    # Restaurants
    "MCD": 26, "SBUX": 36, "CMG": 10, "YUM": 7, "DPZ": 4.5,
    # Consumer Staples
    "PG": 84, "KO": 46, "PEP": 91, "UL": 62, "CL": 20, "MDLZ": 36,
    "KHC": 26, "GIS": 20, "CLX": 7, "KMB": 20,
    # Auto
    "TSLA": 97, "TM": 310, "GM": 172, "F": 176, "STLA": 190, "RIVN": 4, "NIO": 8,
    # Aero/Defense
    "BA": 78, "LMT": 68, "RTX": 69, "NOC": 40, "GD": 43,
    # Industrials
    "GE": 68, "HON": 37, "MMM": 33, "CAT": 67, "DE": 55, "EMR": 17, "ETN": 24,
    # REITs
    "PLD": 8, "AMT": 12, "EQIX": 8, "CCI": 7, "SPG": 6, "O": 4, "DLR": 6, "PSA": 4,
    # Media
    "DIS": 90, "WBD": 40, "PARA": 30, "SPOT": 16, "ROKU": 4,
    # Airlines
    "DAL": 58, "UAL": 55, "AAL": 53, "LUV": 27, "ALK": 11,
    # Crypto
    "MSTR": 0.5, "MARA": 0.4, "RIOT": 0.3, "CLSK": 0.2,
    # Materials
    "NEM": 12, "GOLD": 5, "FCX": 23, "BHP": 55, "NUE": 35, "STLD": 18,
    # Logistics
    "UPS": 91, "FDX": 88, "XPO": 8, "ODFL": 6,
}


def _find_comps(
    ticker: str,
    *,
    max_comps: int = 5,
    size_range: tuple[float, float] = (0.2, 5.0),
) -> list[str]:
    """Find comparable companies using sector + size rules.

    Rules:
      1. Find the ticker's sector from SECTOR_UNIVERSE
      2. Get all tickers in the same sector (excluding the target)
      3. Filter by revenue size: keep those within size_range of target
      4. Sort by revenue proximity (closest match first)
      5. Return top max_comps

    If ticker is not in any sector, returns empty list.
    If we can't determine revenue, returns sector peers by position (assumes size ordering).
    """
    tk = ticker.upper()
    sector = _TICKER_TO_SECTOR.get(tk)
    if not sector:
        return []

    sector_peers = [t for t in SECTOR_UNIVERSE[sector] if t != tk]
    if not sector_peers:
        return []

    # Get target revenue for size matching
    target_rev = _REVENUE_ESTIMATES_B.get(tk)

    # If we don't know the target's revenue, return the closest peers by list position
    # (SECTOR_UNIVERSE lists are ordered roughly by size)
    if target_rev is None or target_rev <= 0:
        # Find target's position in the sector list
        full_list = SECTOR_UNIVERSE[sector]
        try:
            idx = full_list.index(tk)
        except ValueError:
            idx = 0
        # Take neighbors: 2 above, 3 below (or adjust if at edges)
        start = max(0, idx - 2)
        end = min(len(full_list), idx + 4)
        neighbors = [t for t in full_list[start:end] if t != tk]
        return neighbors[:max_comps]

    # Size-filtered matching
    lo, hi = target_rev * size_range[0], target_rev * size_range[1]
    scored: list[tuple[str, float]] = []

    for peer in sector_peers:
        peer_rev = _REVENUE_ESTIMATES_B.get(peer)
        if peer_rev is None:
            # Unknown size — include with low priority (distance = large)
            scored.append((peer, 1000.0))
            continue
        if lo <= peer_rev <= hi:
            # Revenue ratio distance: how many "x" away from 1:1
            ratio = max(peer_rev / target_rev, target_rev / peer_rev)
            scored.append((peer, ratio))

    # Sort by distance (closest match first)
    scored.sort(key=lambda x: x[1])

    # If size filtering is too restrictive (< 3 results), widen to full sector
    if len(scored) < 3:
        for peer in sector_peers:
            if peer not in [s[0] for s in scored]:
                peer_rev = _REVENUE_ESTIMATES_B.get(peer, 0)
                ratio = max(peer_rev / target_rev, target_rev / peer_rev) if peer_rev > 0 else 100.0
                scored.append((peer, ratio))
        scored.sort(key=lambda x: x[1])

    return [t for t, _ in scored[:max_comps]]


# Legacy static fallback (used when ticker isn't in SECTOR_UNIVERSE)
PEER_MAP: dict[str, list[str]] = {
    tk: _find_comps(tk, max_comps=5) or peers
    for tk, peers in {
        "AAPL": ["MSFT", "GOOG", "AMZN", "META"],
        "MSFT": ["AAPL", "GOOG", "AMZN", "ORCL"],
    }.items()
}


# ═══════════════════════════════════════════════════════════════════════════
#  Health check (Railway uses this to verify the service is running)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check endpoint for Railway deployment.

    MUST respond instantly — never block on DB/network calls.
    Railway kills containers that don't pass health checks quickly.
    """
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
#  Tool handlers — each returns a dict for the frontend
# ═══════════════════════════════════════════════════════════════════════════

def _handle_financials(ticker: str, year: int | None, form_type: str = "10-K") -> dict:
    """Handle a financial data request for a single company.

    Cache hierarchy:
      1. Memory (_result_cache) — 10 min TTL, fastest
      2. Supabase (financial_cache) — 1 hour TTL, persists across deploys
      3. Disk (disk_cache) — 24 hour TTL, local only
      4. SEC EDGAR — authoritative source, slowest
    """
    from sec_mcp import supabase_cache

    # ── Try Supabase cache first (before hitting SEC EDGAR) ──
    period_key = f"{form_type}|{year or 'latest'}"
    sb_cached = supabase_cache.get_cached(ticker.upper(), "financials", period_key)
    if sb_cached and isinstance(sb_cached, dict) and sb_cached.get("metrics"):
        log.info("Supabase cache hit for %s financials", ticker)
        data = sb_cached
        summary = generate_local_summary(data)
        # Warm the memory cache too
        acc = (data.get("filing_info") or {}).get("accession_number", "cached")
        _rcache_put(ticker, acc, data, summary)
    else:
        # ── Extract from SEC EDGAR ──
        try:
            data = extract_financials(
                ticker, year=year, form_type=form_type,
                include_statements=True, include_segments=True,
            )
        except Exception as exc:
            log.exception("extract_financials crashed for %s", ticker)
            return {
                "tool": "financials",
                "data": {"ticker_or_cik": ticker, "error": str(exc), "metrics": {}},
                "summary": f"Error loading data: {exc}",
            }
        if data:
            learn_company(ticker, data.get("company_name", ""))
            acc = (data.get("filing_info") or {}).get("accession_number")
            summary = generate_local_summary(data)
            if acc:
                _rcache_put(ticker, acc, data, summary)
                disk_cache.put(ticker, acc, data, summary)
            # ── Persist to Supabase for cross-deploy caching ──
            supabase_cache.set_cached(ticker.upper(), "financials", data, period_key)
        else:
            summary = "No data available."

    # ── Data provenance: cross-validate SEC data with external sources ──
    sources = {"sec_edgar": True, "polygon_validated": False, "web_context": None, "web_citations": []}
    cross_check_results = {}

    # Polygon.io cross-check (non-blocking, cached in Supabase)
    if data and data.get("metrics"):
        try:
            from sec_mcp.polygon_client import cross_check, is_available as poly_avail
            if poly_avail():
                # Check Supabase for cached cross-check
                cached_xcheck = supabase_cache.get_cached(ticker.upper(), "cross_check", period_key)
                if cached_xcheck:
                    cross_check_results = cached_xcheck
                    sources["polygon_validated"] = True
                else:
                    validation = cross_check(ticker.upper(), data["metrics"])
                    if validation:
                        sources["polygon_validated"] = True
                        cross_check_results = validation
                        # Cache cross-check results (24h TTL via supabase_cache defaults)
                        supabase_cache.set_cached(ticker.upper(), "cross_check", validation, period_key)
        except Exception as exc:
            log.debug("Polygon cross-check failed for %s: %s", ticker, exc)

    # Perplexity web context (non-blocking, already cached by perplexity_client)
    try:
        from sec_mcp.perplexity_client import search_financial_news, is_available as pplx_avail
        if pplx_avail():
            news = search_financial_news(ticker.upper())
            if news and news.get("content"):
                sources["web_context"] = news["content"][:3000]
                sources["web_citations"] = news.get("citations", [])[:8]
    except Exception as exc:
        log.debug("Perplexity news fetch failed for %s: %s", ticker, exc)

    result = {"tool": "financials", "data": data, "summary": summary}
    result["sources"] = sources
    result["cross_check"] = cross_check_results
    return result


def _handle_compare(tickers: list[str], year: int | None, form_type: str = "10-K") -> dict:
    """Handle a comparison request for multiple companies.

    Performance: check _rcache and disk_cache for each ticker before falling
    back to the full extract_financials_batch().  Cached hits are spliced in
    so only uncached tickers hit SEC EDGAR.
    """
    from sec_mcp import supabase_cache

    # Try to pull each ticker from cache: memory → Supabase → SEC EDGAR
    cached_results: dict[str, dict] = {}  # ticker -> data
    uncached: list[str] = []
    for t in tickers:
        tk = t.upper()
        # 1. Memory cache
        hit = None
        for key, entry in _result_cache.items():
            if key.startswith(tk + "|") and (time.time() - entry["ts"]) < _RESULT_CACHE_TTL:
                hit = entry
                break
        if hit:
            cached_results[tk] = hit["data"]
            continue
        # 2. Supabase cache
        sb_data = supabase_cache.get_cached(tk, "financials", f"{form_type}|latest")
        if sb_data and isinstance(sb_data, dict) and sb_data.get("metrics"):
            cached_results[tk] = sb_data
            continue
        # 3. Need fresh extraction
        uncached.append(t)

    try:
        fresh = extract_financials_batch(
            uncached, year=year, form_type=form_type,
            include_statements=True, include_segments=True,
        ) if uncached else []
    except Exception as e:
        log.exception("Compare batch failed")
        return {"tool": "compare", "error": str(e), "results": []}

    # Merge cached + fresh in original ticker order
    fresh_iter = iter(fresh)
    results = []
    for t in tickers:
        if t.upper() in cached_results:
            results.append(cached_results[t.upper()])
        else:
            results.append(next(fresh_iter, None))

    summaries = []
    for d in results:
        if d and not d.get("error"):
            learn_company(d.get("ticker_or_cik", ""), d.get("company_name", ""))
        summaries.append(generate_local_summary(d) if d else "No data")
    valid = [r for r in results if r and not r.get("error")]

    narrative = None
    if len(valid) >= 1:
        try:
            from sec_mcp.narrator import explain_comparison
            narrative = explain_comparison(valid, focus="comparison")
        except Exception as e:
            log.warning("Comparison narrative failed: %s", e)

    return {
        "tool": "compare",
        "results": [{"data": d, "summary": s} for d, s in zip(results, summaries)],
        "comparison_narrative": narrative,
    }


def _handle_filing_text(
    ticker: str,
    section: str | None,
    form_type: str = "10-K",
) -> dict:
    """Handle a filing text request (full or section).

    Tries the requested form_type first, then falls back to alternatives.
    For section requests, returns generous text to let the LLM sort it.
    """
    try:
        # Try requested form type first, then FPI alternatives, then cross-type
        # This ensures ASML (20-F), BABA (20-F), etc. are found automatically
        from sec_mcp.sec_client import get_form_alternatives
        forms_to_try = get_form_alternatives(form_type)
        # Also try the other period type as last resort
        if form_type in ("10-K", "20-F"):
            forms_to_try.extend(get_form_alternatives("10-Q"))
        else:
            forms_to_try.extend(get_form_alternatives("10-K"))
        # Deduplicate while preserving order
        seen_ft = set()
        forms_to_try = [f for f in forms_to_try if not (f in seen_ft or seen_ft.add(f))]

        filing = None
        for ft in forms_to_try:
            fils = list_filings(ticker, form_type=ft, limit=1)
            if fils:
                filing = fils[0]
                break

        if not filing:
            return {"tool": "filing_text", "error": f"No filings found for {ticker}"}

        # Use generous limits — section extraction will narrow down
        max_len = 120000 if section else 80000
        text = get_filing_content(
            ticker, filing.accession_number,
            section=section, max_length=max_len,
        )

        # If section text references the proxy statement, auto-fetch DEF 14A
        _PROXY_PHRASES = (
            "proxy statement", "definitive proxy", "def 14a",
            "incorporated by reference", "annual meeting",
        )
        if text and len(text.strip()) < 3000:
            text_lower = text.lower()
            if any(p in text_lower for p in _PROXY_PHRASES):
                log.info("%s section '%s' references proxy — auto-fetching DEF 14A", ticker, section)
                try:
                    proxy_filings = list_filings(ticker, form_type="DEF 14A", limit=1)
                    if proxy_filings:
                        proxy_text = get_filing_content(
                            ticker, proxy_filings[0].accession_number,
                            section=section, max_length=max_len,
                        )
                        if proxy_text and len(proxy_text.strip()) > len(text.strip()):
                            text = (
                                f"[Auto-fetched from DEF 14A proxy statement filed "
                                f"{proxy_filings[0].filing_date}]\n\n{proxy_text}"
                            )
                except Exception as pexc:
                    log.warning("Proxy auto-fetch failed for %s: %s", ticker, pexc)

        # If section was requested but we got very short or no text, report it
        if section and (not text or len(text.strip()) < 200):
            # Try without section filter and explain
            full_text = get_filing_content(
                ticker, filing.accession_number,
                section=None, max_length=max_len,
            )
            section_label = {
                "risk_factors": "Risk Factors (Item 1A)",
                "mda": "MD&A (Item 7)",
                "business": "Business (Item 1)",
                "financial_statements": "Financial Statements (Item 8)",
                "legal": "Legal Proceedings (Item 3)",
                "controls": "Controls and Procedures (Item 9A)",
                "executive_compensation": "Executive Compensation (Item 11)",
            }.get(section, section)
            # Return the full text with a note
            text = (
                f"[Note: Could not isolate '{section_label}' section automatically. "
                f"Returning full filing text from {filing.form_type} filed {filing.filing_date}. "
                f"Search for '{section_label}' within the text below.]\n\n"
                + full_text
            )

        # Display limit: 50k for sections (they're focused), 30k for full filings
        display_limit = 50000 if section else 30000
        return {
            "tool": "filing_text", "ticker": ticker,
            "section": section or "full filing",
            "filing_date": filing.filing_date,
            "accession": filing.accession_number,
            "form_type": filing.form_type,
            "text_length": len(text),
            "text": text[:display_limit],
        }
    except Exception as e:
        return {"tool": "filing_text", "error": str(e)}


def _handle_explain(ticker: str, year: int | None) -> dict:
    """Handle an explain request — extract data + Claude narrative."""
    data = extract_financials(ticker, year=year, include_statements=True, include_segments=True)
    if data:
        learn_company(ticker, data.get("company_name", ""))
    summary = generate_local_summary(data) if data else ""
    narrative = None
    try:
        from sec_mcp.narrator import explain_financials
        narrative = explain_financials(data)
    except Exception as e:
        log.warning("Narrative generation failed: %s", e)
        narrative = f"_Narrative generation unavailable: {e}_"
    return {"tool": "explain", "data": data, "summary": summary, "narrative": narrative}


def _handle_entity(ticker: str) -> dict:
    """Handle an entity profile request — company info from SEC submissions."""
    try:
        from sec_mcp.sec_client import get_sec_client
        client = get_sec_client()
        info = client.get_company_info(ticker)

        # Get the raw submissions data for extra fields
        cik = client.resolve_cik(ticker)
        subs = client._get_submissions(cik)

        # Extract address info
        addresses = subs.get("addresses", {})
        mailing = addresses.get("mailing", {}) or addresses.get("business", {})
        business_addr = addresses.get("business", {}) or mailing

        address_parts = []
        for key in ("street1", "street2", "city", "stateOrCountry", "zipCode"):
            v = business_addr.get(key)
            if v:
                address_parts.append(v)
        address = ", ".join(address_parts)

        # Extract officers if available (sometimes in older filings)
        # The SEC submissions JSON has limited officer data, but we can try
        # to get it from the latest 10-K filing
        officers = []
        # The submissions data doesn't reliably have officers
        # We note this and suggest checking the proxy/10-K

        # Get most recent filings for context (auto-tries 20-F/6-K for foreign filers)
        recent_10k = client.get_filings_smart(ticker, form_type="10-K", limit=1)
        recent_10q = client.get_filings_smart(ticker, form_type="10-Q", limit=1)
        latest_filing = None
        if recent_10k:
            latest_filing = recent_10k[0]
        elif recent_10q:
            latest_filing = recent_10q[0]

        # Phone
        phone = subs.get("phone") or business_addr.get("phone") or ""

        # Fiscal year end
        fy_end = subs.get("fiscalYearEnd", "")
        if fy_end and len(fy_end) == 4:
            fy_end = f"{fy_end[:2]}/{fy_end[2:]}"

        # State of incorporation
        state_inc = subs.get("stateOfIncorporation", "")

        profile = {
            "name": info.name,
            "ticker": info.ticker or ticker.upper(),
            "cik": str(info.cik).zfill(10),
            "sic_code": info.sic_code,
            "industry": info.industry,
            "website": info.website or "",
            "address": address,
            "phone": phone,
            "fiscal_year_end": fy_end,
            "state_of_incorporation": state_inc,
            "ein": subs.get("ein", ""),
            "exchange": subs.get("exchanges", [""])[0] if subs.get("exchanges") else "",
            "latest_filing": {
                "form_type": latest_filing.form_type if latest_filing else None,
                "filing_date": latest_filing.filing_date if latest_filing else None,
                "accession": latest_filing.accession_number if latest_filing else None,
            } if latest_filing else None,
        }

        # Try to extract CEO from latest 10-K filing text (first 2000 chars often has signatures)
        ceo_name = None
        ceo_source = None
        if recent_10k:
            try:
                filing_text = get_filing_content(
                    ticker, recent_10k[0].accession_number,
                    section="executive_compensation", max_length=30000,
                )
                if filing_text and len(filing_text) > 100:
                    # Look for CEO/Chief Executive Officer patterns
                    ceo_patterns = [
                        r"Chief Executive Officer[,\s\-\u2014]+([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)",
                        r"([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)[,\s\-\u2014]+Chief Executive Officer",
                        r"CEO[,\s\-\u2014]+([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)",
                        r"([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)[,\s\-\u2014]+CEO",
                    ]
                    for pat in ceo_patterns:
                        m = re.search(pat, filing_text)
                        if m:
                            ceo_name = m.group(1).strip()
                            ceo_source = f"10-K filed {recent_10k[0].filing_date}"
                            break
            except Exception as exc:
                log.debug("CEO extraction from filing text failed: %s", exc)

        # Fallback: try Claude to extract CEO if we have the API key
        if not ceo_name:
            try:
                from sec_mcp.config import get_config
                config = get_config()
                if config.anthropic_api_key:
                    import anthropic
                    ac = anthropic.Anthropic(api_key=config.anthropic_api_key)
                    resp = ac.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=150,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"Who is the current CEO of {info.name} ({info.ticker})? "
                                f"Reply with ONLY the name, nothing else. "
                                f"If you're unsure, reply 'Unknown'."
                            ),
                        }],
                    )
                    name_text = resp.content[0].text.strip()
                    if name_text and name_text.lower() != "unknown" and len(name_text) < 60:
                        ceo_name = name_text
                        ceo_source = "AI lookup (may not be current)"
            except Exception as exc:
                log.debug("Claude CEO lookup failed: %s", exc)

        profile["ceo"] = ceo_name
        profile["ceo_source"] = ceo_source

        return {"tool": "entity", "profile": profile}
    except Exception as e:
        log.exception("Entity profile failed")
        return {"tool": "entity", "error": str(e)}


def _handle_qa(ticker: str, question: str, year: int | None = None, form_type: str = "10-K") -> dict:
    """Handle a Q&A request — uses Anthropic to answer questions about financial data."""
    try:
        from sec_mcp.config import get_config
        config = get_config()
        if not config.anthropic_api_key:
            return {
                "tool": "qa",
                "answer": "Anthropic API key not configured. Add ANTHROPIC_API_KEY to your .env file.",
                "citations": [],
            }

        # Build context from available financial data
        context_parts = []
        data = None
        try:
            data = extract_financials(
                ticker, year=year, form_type=form_type,
                include_statements=True, include_segments=True,
            )
            if data:
                learn_company(ticker, data.get("company_name", ""))
                context_parts.append(f"Company: {data.get('company_name', ticker)}")
                context_parts.append(f"Ticker: {ticker}")
                fi = data.get("filing_info", {})
                context_parts.append(f"Filing: {fi.get('form_type', form_type)} filed {fi.get('filing_date', 'N/A')}")

                # Metrics
                metrics = data.get("metrics", {})
                if metrics:
                    context_parts.append("\nKEY METRICS:")
                    for k, v in metrics.items():
                        if v is not None:
                            context_parts.append(f"  {k}: {v}")

                # Ratios
                ratios = data.get("ratios", {})
                if ratios:
                    context_parts.append("\nFINANCIAL RATIOS:")
                    for k, v in ratios.items():
                        if v is not None:
                            context_parts.append(f"  {k}: {v}")

                # Prior metrics for comparison
                pm = data.get("prior_metrics", {})
                if pm:
                    context_parts.append(f"\nPRIOR PERIOD METRICS ({data.get('yoy_label', 'vs Prior')}):")
                    for k, v in pm.items():
                        if v is not None:
                            context_parts.append(f"  {k}: {v}")

                # Validation
                val = data.get("validation", [])
                if val:
                    context_parts.append("\nDATA WARNINGS:")
                    for w in val:
                        context_parts.append(f"  [{w.get('severity')}] {w.get('message')}")
        except Exception as e:
            context_parts.append(f"Error loading financial data: {e}")

        context_text = "\n".join(context_parts) if context_parts else "No financial data available."

        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        system = (
            "You are a senior financial analyst answering questions about SEC filing data. "
            "You have access to the structured financial data below. "
            "Rules:\n"
            "- Use actual numbers from the data (format: $B, $M, %).\n"
            "- Cite where data comes from (e.g., 'from the 10-K filed 2025-02-15').\n"
            "- If data is missing, say 'I don't have that data' and suggest fetching it.\n"
            "- Be concise but thorough. Use bullet points for clarity.\n"
            "- If asked about trends, use prior period data if available.\n"
            "- Flag any data quality warnings you see.\n"
        )

        user_msg = f"FINANCIAL DATA CONTEXT:\n{context_text}\n\nQUESTION: {question}"

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )

        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text

        # Build citations from the data context
        citations = []
        fi = (data or {}).get("filing_info", {})
        if fi:
            citations.append({
                "source": f"{fi.get('form_type', '10-K')} filed {fi.get('filing_date', 'N/A')}",
                "accession": fi.get("accession_number"),
            })

        return {
            "tool": "qa", "answer": answer, "citations": citations,
            "data": data, "ticker": ticker,
        }
    except Exception as e:
        log.exception("Q&A failed")
        return {"tool": "qa", "answer": f"Error: {e}", "citations": []}


# ═══════════════════════════════════════════════════════════════════════════
#  API endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(req: ChatRequest, bg: BackgroundTasks) -> dict:
    """Main chat endpoint — parses intent and dispatches to the right handler.

    Returns intent metadata (tool, reasoning) alongside results so the
    frontend can show the thinking/routing process.
    """
    t0 = time.time()
    intent = parse_intent(req.message)

    # Always include intent metadata for transparency
    meta = {
        "intent_tool": intent["tool"],
        "intent_reasoning": intent.get("reasoning", []),
        "intent_tickers": intent["tickers"],
        "intent_section": intent.get("section"),
        "intent_form": intent["form_type"],
    }

    if not intent["tickers"]:
        return {
            "type": "info",
            "message": (
                "I couldn't identify a company. Try:\n\n"
                "- **Apple** or **AAPL** — financials\n"
                "- **Morgan Stanley 10-Q** — quarterly\n"
                "- **compare MSFT vs GOOG**\n"
                "- **risk factors Tesla**\n"
                "- **NVDA history** — 10-year analysis\n"
                "- **Who is Apple's CEO?** — entity profile\n"
                "- **TSLA md&a** — management discussion"
            ),
            **meta,
        }

    tickers = intent["tickers"]
    year = intent["year"]
    tool = intent["tool"]
    form_type = intent["form_type"]

    try:
        if tool == "entity":
            result = _handle_entity(tickers[0])
        elif tool == "qa":
            result = _handle_qa(tickers[0], req.message, year, form_type)
        elif tool == "compare":
            result = _handle_compare(tickers, year, form_type)
        elif tool == "filing_text":
            result = _handle_filing_text(tickers[0], intent["section"], form_type)
        elif tool == "explain":
            result = _handle_explain(tickers[0], year)
        elif tool == "historical":
            if db_available():
                bg.add_task(run_historical_extraction, tickers[0])
                result = {"tool": "historical", "ticker": tickers[0],
                          "message": f"Historical extraction started for {tickers[0]}. Dashboard loading..."}
            else:
                result = _handle_financials(tickers[0], year, form_type)
                result["tool"] = "financials"
        else:
            result = _handle_financials(tickers[0], year, form_type)

        result["resolved_tickers"] = tickers
        result["query"] = req.message
        result["elapsed_ms"] = int((time.time() - t0) * 1000)

        # Auto-trigger historical extraction in background for financial queries
        if tool in ("financials", "explain") and db_available():
            job = get_job(tickers[0])
            if job is None or job.get("status") != "complete":
                bg.add_task(run_historical_extraction, tickers[0])

        return {"type": "result", **meta, **result}
    except Exception as exc:
        log.exception("Chat request failed")
        return {"type": "error", "message": traceback.format_exc(), **meta}


@app.post("/api/ask")
async def ask_question(req: ChatRequest) -> dict:
    """Direct Q&A endpoint — takes a question + uses current context.

    This allows the frontend to send follow-up questions about on-screen data.
    """
    t0 = time.time()
    intent = parse_intent(req.message)
    tickers = intent["tickers"]

    if not tickers:
        return {"answer": "Please specify a company (e.g., 'AAPL revenue yoy').", "citations": []}

    result = _handle_qa(tickers[0], req.message, intent["year"], intent["form_type"])
    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


@app.get("/api/historical/{ticker}")
async def historical_data(ticker: str, form_type: str | None = None):
    """Return cached historical data from MongoDB."""
    data = get_historical_data(ticker.upper(), form_type=form_type)
    return data


@app.post("/api/historical/{ticker}/fetch")
async def trigger_fetch(ticker: str, bg: BackgroundTasks):
    """Manually trigger historical extraction as a background task."""
    if not db_available():
        return {"error": "MongoDB not configured. Add MONGODB_URI to .env"}
    bg.add_task(run_historical_extraction, ticker.upper())
    return {"status": "started", "ticker": ticker.upper()}


@app.get("/api/historical/{ticker}/status")
async def fetch_status(ticker: str):
    """Poll historical extraction job progress."""
    job = get_job(ticker.upper())
    return job or {"status": "none"}


@app.get("/api/filings/{ticker}")
async def list_available_filings(ticker: str, limit: int = 24):
    """List available filings for the filing selector toolbar.

    Tries both US (10-K/10-Q) and FPI (20-F/6-K) form types automatically.
    """
    from sec_mcp.sec_client import get_sec_client
    client = get_sec_client()
    # Try US annual, then FPI annual; same for quarterly
    filings_k = client.get_filings_smart(ticker, form_type="10-K", limit=12)
    filings_q = client.get_filings_smart(ticker, form_type="10-Q", limit=16)
    combined = sorted(
        [{"accession": f.accession_number, "form_type": f.form_type,
          "filing_date": f.filing_date, "description": f.description}
         for f in filings_k + filings_q],
        key=lambda x: x.get("filing_date", ""), reverse=True,
    )
    return {"ticker": ticker.upper(), "filings": combined[:limit]}


@app.post("/api/load-filing")
async def load_specific_filing(req: LoadFilingRequest):
    """Load financials for one specific filing (by accession number).

    Checks memory cache then disk cache before hitting SEC EDGAR.
    Preserves the original ticker symbol so the UI never loses context.
    """
    ticker = req.ticker.upper()

    # Memory cache hit → instant return
    cached = _rcache_get(ticker, req.accession)
    if cached:
        log.info("Memory cache hit for %s / %s", ticker, req.accession)
        if cached["data"]:
            cached["data"]["ticker_or_cik"] = ticker
        return {"data": cached["data"], "summary": cached["summary"], "cached": True, "cache_source": "memory"}

    # Disk cache hit → fast return, repopulate memory cache
    disk_hit = disk_cache.get(ticker, req.accession)
    if disk_hit:
        log.info("Disk cache hit for %s / %s", ticker, req.accession)
        data = disk_hit["data"]
        summary = disk_hit["summary"]
        if data:
            data["ticker_or_cik"] = ticker
            _rcache_put(ticker, req.accession, data, summary)
        return {"data": data, "summary": summary, "cached": True, "cache_source": "disk"}

    data = extract_financials(
        ticker, accession=req.accession, form_type=req.form_type,
        include_statements=True, include_segments=True,
    )
    if data:
        data["ticker_or_cik"] = ticker
        learn_company(ticker, data.get("company_name", ""))
        summary = generate_local_summary(data)
        _rcache_put(ticker, req.accession, data, summary)
        disk_cache.put(ticker, req.accession, data, summary)
    else:
        summary = ""
    return {"data": data, "summary": summary}


@app.get("/api/search")
async def search_assets(q: str = "", limit: int = 12):
    """Search for companies by ticker or name. Returns ticker, CIK, name, and exchange."""
    if len(q.strip()) < 1:
        return {"results": []}
    try:
        results = search_companies(q.strip())
    except Exception as exc:
        log.warning("Search failed: %s", exc)
        return {"results": []}
    seen: set[int] = set()
    deduped = []
    for r in results:
        if r.cik not in seen:
            seen.add(r.cik)
            deduped.append({
                "ticker": r.ticker or "",
                "cik": str(r.cik).zfill(10),
                "name": r.name,
                "exchange": r.exchange or "",
            })
        if len(deduped) >= limit:
            break
    return {"results": deduped}


@app.get("/api/cache/stats")
async def cache_stats():
    """Return cache statistics (memory + disk)."""
    mem_entries = len(_result_cache)
    return {
        "memory": {"entries": mem_entries},
        "disk": disk_cache.stats(),
        "tickers": disk_cache.list_tickers(),
    }

@app.get("/api/peers/{ticker}")
async def get_peers(ticker: str):
    """Return suggested peer companies using rules-based sector + size matching.

    Rules:
      1. Sector match — same SECTOR_UNIVERSE bucket
      2. Size filter — revenue within 0.2x–5x of target
      3. Ranked by revenue proximity
    """
    tk = ticker.upper()
    # Dynamic comp engine
    comps = _find_comps(tk, max_comps=5)
    # Fallback to legacy map
    if not comps:
        comps = PEER_MAP.get(tk, [])
    sector = _TICKER_TO_SECTOR.get(tk, "unknown")
    target_rev = _REVENUE_ESTIMATES_B.get(tk)
    return {
        "ticker": tk,
        "peers": comps,
        "sector": sector,
        "target_revenue_est": target_rev,
        "match_method": "sector_size" if comps else "fallback",
    }


@app.get("/api/geo-revenue/{ticker}")
async def get_geo_revenue(ticker: str, period: str = "annual"):
    """Geographic revenue breakdown — FMP first, then XBRL, then filing text."""
    tk = ticker.upper()

    # Check Supabase cache
    from sec_mcp import supabase_cache
    cached = supabase_cache.get_cached(tk, "geo_segments", period)
    if cached:
        return cached

    # 1. Try FMP API (best: structured, multi-year)
    try:
        from sec_mcp.fmp_client import get_geo_segments, is_available as fmp_ok
        if fmp_ok():
            fmp_data = get_geo_segments(tk, period=period)
            if fmp_data and fmp_data[0].get("segments"):
                # Return current + historical
                current = fmp_data[0]["segments"]
                geo_formatted = [
                    {"segment": s["name"], "value": s["value"]}
                    for s in current
                ]
                result = {
                    "ticker": tk,
                    "geographic_segments": geo_formatted,
                    "history": fmp_data,
                    "source": "fmp",
                }
                supabase_cache.set_cached(tk, "geo_segments", result, period)
                return result
    except Exception as exc:
        log.debug("FMP geo failed for %s: %s", tk, exc)

    # 2. Try XBRL segment data
    try:
        data = extract_financials(tk, include_segments=True)
        if data:
            geo = (data.get("segments") or {}).get("geographic_segments", [])
            if len(geo) >= 2:
                result = {"ticker": tk, "geographic_segments": geo, "source": "xbrl"}
                supabase_cache.set_cached(tk, "geo_segments", result, period)
                return result
    except Exception as exc:
        log.warning("XBRL geo extraction failed for %s: %s", tk, exc)

    # 3. Fall back to filing text parsing
    try:
        from sec_mcp.financials import _parse_geo_from_filing_text

        filing_result = _handle_filing_text(tk, section="financial_statements", form_type="10-K")
        text = filing_result.get("text", "")
        total_rev = None
        try:
            data = extract_financials(tk)
            total_rev = (data or {}).get("metrics", {}).get("revenue")
        except Exception as exc:
            log.debug("Revenue fetch for geo scale failed for %s: %s", tk, exc)

        geo = _parse_geo_from_filing_text(text, total_revenue=total_rev)
        if geo:
            result = {"ticker": tk, "geographic_segments": geo, "source": "filing_text"}
            supabase_cache.set_cached(tk, "geo_segments", result, period)
            return result
    except Exception as e:
        log.warning("Geo revenue parsing failed for %s: %s", tk, e)

    return {"ticker": tk, "geographic_segments": [], "source": "none"}


@app.get("/api/segments/{ticker}")
async def get_segments(ticker: str, period: str = "annual"):
    """Revenue segments — FMP first, then XBRL, then filing text."""
    tk = ticker.upper()

    # Check Supabase cache
    from sec_mcp import supabase_cache
    cached = supabase_cache.get_cached(tk, "product_segments", period)
    if cached:
        return cached

    # 1. Try FMP API (best: structured product/business segments)
    try:
        from sec_mcp.fmp_client import get_product_segments, is_available as fmp_ok
        if fmp_ok():
            fmp_data = get_product_segments(tk, period=period)
            if fmp_data and fmp_data[0].get("segments"):
                current = fmp_data[0]["segments"]
                seg_formatted = [
                    {"segment": s["name"], "value": s["value"]}
                    for s in current
                ]
                result = {
                    "ticker": tk,
                    "segments": seg_formatted,
                    "history": fmp_data,
                    "source": "fmp",
                }
                supabase_cache.set_cached(tk, "product_segments", result, period)
                return result
    except Exception as exc:
        log.debug("FMP segments failed for %s: %s", tk, exc)

    # 2. Try XBRL segments
    try:
        data = extract_financials(tk, include_segments=True)
        if data:
            segs = (data.get("segments") or {}).get("revenue_segments", [])
            if len(segs) >= 2:
                result = {"ticker": tk, "segments": segs, "source": "xbrl"}
                supabase_cache.set_cached(tk, "product_segments", result, period)
                return result
    except Exception as exc:
        log.warning("XBRL segment extraction failed for %s: %s", tk, exc)

    # 3. Fall back to filing text parsing
    try:
        from sec_mcp.financials import _parse_segments_from_filing_text

        total_rev = None
        try:
            data = extract_financials(tk)
            total_rev = (data or {}).get("metrics", {}).get("revenue")
        except Exception as exc:
            log.debug("Revenue fetch for segment scale failed for %s: %s", tk, exc)

        for section in ("mda", "financial_statements"):
            filing_result = _handle_filing_text(tk, section=section, form_type="10-K")
            text = filing_result.get("text", "")
            segs = _parse_segments_from_filing_text(text, total_revenue=total_rev)
            if len(segs) >= 2:
                result = {"ticker": tk, "segments": segs, "source": "filing_text"}
                supabase_cache.set_cached(tk, "product_segments", result, period)
                return result
    except Exception as e:
        log.warning("Segment parsing failed for %s: %s", tk, e)

    return {"ticker": tk, "segments": [], "source": "none"}


@app.get("/api/financials-history/{ticker}")
async def get_financials_history(ticker: str, period: str = "annual",
                                  date_from: str = "", date_to: str = "",
                                  limit: int = 10):
    """Multi-year income statement, balance sheet, and cash flow data from FMP.

    Used by the frontend for date-range chart rendering.
    """
    tk = ticker.upper()

    from sec_mcp import supabase_cache
    cache_type = "income_history"
    cached = supabase_cache.get_cached(tk, cache_type, period, date_from, date_to)
    if cached:
        return cached

    try:
        from sec_mcp.fmp_client import (
            get_income_statements,
            get_balance_sheet,
            get_cash_flow,
            is_available as fmp_ok,
        )
        if not fmp_ok():
            return {"ticker": tk, "error": "FMP API not configured", "income": [], "balance": [], "cashflow": []}

        income = get_income_statements(tk, period=period, limit=limit)
        balance = get_balance_sheet(tk, period=period, limit=limit)
        cashflow = get_cash_flow(tk, period=period, limit=limit)

        # Apply date range filter if provided
        if date_from:
            income = [r for r in income if r.get("date", "") >= date_from]
            balance = [r for r in balance if r.get("date", "") >= date_from]
            cashflow = [r for r in cashflow if r.get("date", "") >= date_from]
        if date_to:
            income = [r for r in income if r.get("date", "") <= date_to]
            balance = [r for r in balance if r.get("date", "") <= date_to]
            cashflow = [r for r in cashflow if r.get("date", "") <= date_to]

        result = {
            "ticker": tk,
            "period": period,
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "source": "fmp",
        }
        supabase_cache.set_cached(tk, cache_type, result, period, date_from, date_to)
        return result
    except Exception as exc:
        log.warning("FMP history fetch failed for %s: %s", tk, exc)
        return {"ticker": tk, "error": str(exc), "income": [], "balance": [], "cashflow": []}


@app.post("/api/comps")
async def get_comps(req: CompsRequest):
    """Fetch financial data for multiple tickers for side-by-side comparison."""
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:8]
    if not tickers:
        return {"error": "No tickers provided", "results": [], "resolved_tickers": []}
    result = _handle_compare(tickers, year=req.year, form_type=req.form_type)
    result["resolved_tickers"] = tickers
    return result


@app.delete("/api/cache/clear")
async def cache_clear(ticker: str | None = None):
    """Clear disk cache. Optional ticker query param to clear just one company."""
    count = disk_cache.clear(ticker)
    # Also clear memory cache entries for this ticker
    if ticker:
        keys_to_del = [k for k in _result_cache if k.startswith(ticker.upper() + "|")]
        for k in keys_to_del:
            del _result_cache[k]
    else:
        _result_cache.clear()
    return {"cleared": count, "ticker": ticker}


@app.post("/api/chatbot")
async def chatbot_qa(req: ChatbotRequest) -> dict:
    """Right-panel chatbot — answers questions using on-screen data as context.

    Accepts the currently loaded financial data so it doesn't need to re-fetch.
    """
    try:
        from sec_mcp.config import get_config
        config = get_config()
        if not config.anthropic_api_key:
            return {
                "answer": "Add ANTHROPIC_API_KEY to your .env file to enable the AI assistant.",
                "citations": [],
            }

        context_parts: list[str] = []
        ctx = req.context
        if req.ticker:
            context_parts.append(f"Ticker: {req.ticker}")
        if ctx:
            if ctx.get("company_name"):
                context_parts.append(f"Company: {ctx['company_name']}")
            fi = ctx.get("filing_info", {})
            if fi:
                context_parts.append(
                    f"Filing: {fi.get('form_type', '?')} filed {fi.get('filing_date', '?')}"
                )
                context_parts.append(f"Period type: {ctx.get('period_type', '?')}")
            metrics = ctx.get("metrics", {})
            if metrics:
                context_parts.append("\nKEY METRICS:")
                for k, v in metrics.items():
                    if v is not None:
                        context_parts.append(f"  {k}: {v}")
            ratios = ctx.get("ratios", {})
            if ratios:
                context_parts.append("\nFINANCIAL RATIOS:")
                for k, v in ratios.items():
                    if v is not None:
                        context_parts.append(f"  {k}: {v}")
            pm = ctx.get("prior_metrics", {})
            if pm:
                context_parts.append(
                    f"\nPRIOR PERIOD ({ctx.get('yoy_label', 'vs Prior')}):"
                )
                for k, v in pm.items():
                    if v is not None:
                        context_parts.append(f"  {k}: {v}")
            segs = ctx.get("segments", {})
            if segs:
                for seg_type in ("revenue_segments", "geographic_segments"):
                    seg_list = segs.get(seg_type, [])
                    if seg_list:
                        context_parts.append(f"\n{seg_type.upper().replace('_', ' ')}:")
                        for s in seg_list:
                            context_parts.append(
                                f"  {s.get('segment', '?')}: {s.get('value')}"
                            )
            val = ctx.get("validation", [])
            if val:
                context_parts.append("\nDATA WARNINGS:")
                for w in val:
                    context_parts.append(
                        f"  [{w.get('severity')}] {w.get('message')}"
                    )

            # Include filing text sections (MD&A, Risk Factors, etc.) from frontend cache
            filing_sections = ctx.get("_filing_sections", {})
            if filing_sections:
                for sec_name, sec_text in filing_sections.items():
                    if sec_text and len(sec_text) > 100:
                        context_parts.append(f"\n{sec_name.upper()} SECTION:")
                        context_parts.append(sec_text[:8000])

            # Include statement data for richer context
            for stmt_key in ("income_statement", "balance_sheet", "cash_flow_statement"):
                stmt = ctx.get(stmt_key, [])
                if stmt:
                    context_parts.append(f"\n{stmt_key.upper().replace('_', ' ')}:")
                    for row in stmt[:30]:
                        label = row.get("label", "")
                        vals = {k: v for k, v in row.items()
                                if k not in ("label", "concept", "standard_concept", "level",
                                             "is_abstract", "is_total", "abstract", "units", "decimals")
                                and v is not None}
                        if vals:
                            context_parts.append(f"  {label}: {vals}")

        # Enrich with Perplexity real-time search for news/current data queries
        _realtime_keywords = ("news", "recent", "latest", "today", "analyst", "price target",
                              "upgrade", "downgrade", "earnings", "guidance", "forecast",
                              "compare to", "vs market", "industry average", "verify", "validate")
        if any(kw in req.message.lower() for kw in _realtime_keywords):
            try:
                from sec_mcp.perplexity_client import search, is_available as pplx_avail
                if pplx_avail():
                    pplx = search(req.message, ticker=req.ticker or None)
                    if pplx and pplx.get("content"):
                        context_parts.append("\nREAL-TIME WEB SEARCH RESULTS (via Perplexity):")
                        context_parts.append(pplx["content"][:4000])
                        if pplx.get("citations"):
                            context_parts.append("\nSources: " + ", ".join(pplx["citations"][:5]))
            except Exception as exc:
                log.debug("Perplexity enrichment failed: %s", exc)

        # Fallback: fetch data if no context provided
        if not context_parts and req.ticker:
            try:
                data = extract_financials(
                    req.ticker, include_statements=True, include_segments=True,
                )
                if data:
                    context_parts.append(f"Company: {data.get('company_name', req.ticker)}")
                    for k, v in data.get("metrics", {}).items():
                        if v is not None:
                            context_parts.append(f"  {k}: {v}")
            except Exception as exc:
                log.debug("Financial context fetch failed for chatbot: %s", exc)

        context_text = (
            "\n".join(context_parts) if context_parts
            else "No financial data loaded."
        )

        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        system = (
            "You are a senior financial analyst AI assistant embedded in an SEC filings dashboard. "
            "You have DIRECT ACCESS to the company's financial data currently displayed on the user's screen. "
            "This data comes from real SEC EDGAR XBRL filings — it is authoritative.\n\n"
            "ADAPTIVE RESPONSE STYLE — match depth to the question:\n"
            "- Simple factual questions (\"what's revenue?\") → 1-2 sentences with the number and source.\n"
            "- Analytical questions (\"how are margins trending?\") → 2-3 paragraphs with bullets.\n"
            "- Deep-dive requests (\"executive summary\", \"analyze\") → full structured breakdown with ## headers.\n\n"
            "FORMAT — Your response is rendered as full Markdown (GFM). Use rich formatting:\n"
            "- **Bold** for key numbers, terms, and company names.\n"
            "- Use `## Headers` and `### Subheaders` to organize longer responses.\n"
            "- Use bullet points (`- item`) for lists of 3+ items.\n"
            "- Use **Markdown tables** for side-by-side comparisons or multi-metric summaries:\n"
            "  ```\n"
            "  | Metric | Current | Prior | Change |\n"
            "  |--------|---------|-------|--------|\n"
            "  | Revenue | $100B | $90B | +11.1% |\n"
            "  ```\n"
            "- Reference specific numbers (format: **$1.23B**, **$456M**, **12.3%**).\n"
            "- Start with a direct answer on the first line — no preamble.\n"
            "- Cite the filing source (e.g., 'per the 10-K filed 2024-11-01').\n"
            "- Compare to prior periods when data is available — calculate % changes.\n"
            "- If data is missing, say exactly what's missing and suggest how to get it.\n\n"
            "SOURCE ATTRIBUTION — always tag where data comes from:\n"
            "- Prefix SEC-sourced numbers with [SEC EDGAR] inline.\n"
            "- Prefix web-sourced context with [Web Search] inline.\n"
            "- Prefix cross-validated data (SEC matched by Polygon) with [Verified ✓] inline.\n"
            "- Keep these tags inline with the data, not as separate sections.\n\n"
            "FOR DEEP ANALYSIS:\n"
            "- Use ## headers to organize sections.\n"
            "- Provide actionable insights: 'This suggests...', 'Key risk:...'.\n"
            "- End with a suggested follow-up question.\n"
            "- For exec summaries, organize by: Overview, Strengths, Risks, Outlook.\n"
        )

        # Build messages with conversation history for multi-turn context
        messages = []
        # Include recent history (last 6 turns to stay within token budget)
        for turn in (req.history or [])[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Current message with full data context
        user_msg = (
            f"FINANCIAL DATA ON SCREEN:\n{context_text}\n\n"
            f"USER QUESTION: {req.message}"
        )
        messages.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system,
            messages=messages,
        )

        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text

        citations = []
        fi = (ctx or {}).get("filing_info", {})
        if fi and fi.get("form_type"):
            citations.append({
                "source": f"{fi.get('form_type', '?')} filed {fi.get('filing_date', '?')}",
                "accession": fi.get("accession_number"),
            })

        return {"answer": answer, "citations": citations}
    except Exception as e:
        log.exception("Chatbot Q&A failed")
        return {"answer": f"Error: {e}", "citations": []}


class ExecSummaryRequest(BaseModel):
    ticker: str = ""
    context: dict = {}
    sections: dict = {}


@app.post("/api/exec-summaries")
async def exec_summaries(req: ExecSummaryRequest) -> dict:
    """Generate executive summaries via Claude — single call for speed.

    Instead of 4 sequential API calls (~20-30s), uses ONE call that generates
    all 4 summaries in a structured format (~5-8s). Context is trimmed to
    only metrics+ratios+prior (no full statements or filing text) to keep
    the prompt small and fast.
    """
    try:
        from sec_mcp.config import get_config
        config = get_config()
        if not config.anthropic_api_key:
            return {k: "Add ANTHROPIC_API_KEY to enable exec summaries." for k in ["overall", "income", "balance_sheet", "cash_flow"]}

        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        ctx = req.context
        metrics = ctx.get("metrics", {})
        ratios = ctx.get("ratios", {})
        prior = ctx.get("prior_metrics", {})
        fi = ctx.get("filing_info", {})

        # Build a FOCUSED data block — only key metrics, ratios, and prior period
        # NO full statements or filing text (too large, makes Claude slow/fail)
        data_block = f"Company: {ctx.get('company_name', req.ticker)}\n"
        data_block += f"Filing: {fi.get('form_type', '?')} filed {fi.get('filing_date', '?')}\n"
        data_block += f"Period: {ctx.get('period_type', 'annual')}\n"
        if ctx.get("industry_class"):
            data_block += f"Industry: {ctx['industry_class']}\n"

        if metrics:
            data_block += "\nKEY METRICS:\n"
            for k, v in metrics.items():
                if v is not None:
                    data_block += f"  {k}: {v}\n"
        if ratios:
            data_block += "\nRATIOS:\n"
            for k, v in ratios.items():
                if v is not None:
                    data_block += f"  {k}: {v}\n"
        if prior:
            label = ctx.get("yoy_label", "vs Prior")
            data_block += f"\nPRIOR PERIOD ({label}):\n"
            for k, v in prior.items():
                if v is not None:
                    data_block += f"  {k}: {v}\n"

        # Include only the top 15 rows from each statement (enough for context)
        for stmt_key in ["income_statement", "balance_sheet", "cash_flow_statement"]:
            stmt = ctx.get(stmt_key, [])
            if stmt:
                data_block += f"\n{stmt_key.upper().replace('_', ' ')}:\n"
                for row in stmt[:15]:
                    label = row.get("label", "")
                    vals = {k: v for k, v in row.items()
                            if k not in ("label", "concept", "standard_concept", "level",
                                         "is_abstract", "is_total", "abstract", "units", "decimals")
                            and v is not None}
                    if vals:
                        data_block += f"  {label}: {vals}\n"

        # Single Claude call that returns all 4 summaries at once
        system = (
            "You are a senior financial analyst. Generate executive summaries using "
            "specific numbers ($B, $M, %). Use **bold** for emphasis and bullet points. "
            "Be concise — each section should be 2-3 short paragraphs."
        )
        prompt = (
            f"DATA:\n{data_block}\n\n"
            "Generate 4 executive summaries in this EXACT format (use these exact headers):\n\n"
            "## OVERALL\n[3-4 paragraph executive summary of financial health, strengths, risks, outlook]\n\n"
            "## INCOME\n[2-3 paragraphs on revenue, margins, cost structure, profitability trends]\n\n"
            "## BALANCE_SHEET\n[2-3 paragraphs on asset quality, leverage, liquidity, equity]\n\n"
            "## CASH_FLOW\n[2-3 paragraphs on operating cash, capex, FCF, capital allocation]"
        )

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        full_text = resp.content[0].text.strip()

        # Parse the structured response into 4 sections
        results = _parse_exec_sections(full_text)
        return results

    except Exception as e:
        log.exception("Exec summaries failed")
        return {k: f"Error: {e}" for k in ["overall", "income", "balance_sheet", "cash_flow"]}


def _parse_exec_sections(text: str) -> dict:
    """Parse Claude's structured exec summary response into 4 sections."""
    sections = {"overall": "", "income": "", "balance_sheet": "", "cash_flow": ""}

    # Split by ## headers
    parts = re.split(r"##\s*", text)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        lower = part[:30].lower()
        if lower.startswith("overall"):
            sections["overall"] = part[part.index("\n"):].strip() if "\n" in part else part
        elif lower.startswith("income"):
            sections["income"] = part[part.index("\n"):].strip() if "\n" in part else part
        elif lower.startswith("balance"):
            sections["balance_sheet"] = part[part.index("\n"):].strip() if "\n" in part else part
        elif lower.startswith("cash"):
            sections["cash_flow"] = part[part.index("\n"):].strip() if "\n" in part else part

    # If parsing failed (no headers found), put everything in "overall"
    if not any(sections.values()):
        sections["overall"] = text

    return sections


@app.get("/api/section/{ticker}/{section}")
async def get_section(ticker: str, section: str) -> dict:
    """Fetch a specific filing text section for background loading."""
    try:
        # Try US and FPI form types for section extraction
        from sec_mcp.sec_client import get_sec_client
        _sc = get_sec_client()
        filings = _sc.get_filings_smart(ticker, form_type="10-K", limit=1)
        if not filings:
            filings = _sc.get_filings_smart(ticker, form_type="10-Q", limit=1)
        if not filings:
            return {"text": "", "error": "No filings found"}

        f = filings[0]
        text = get_filing_content(
            ticker, f.accession_number,
            section=section, max_length=50000,
        )
        return {
            "text": text or "",
            "filing": f.form_type,
            "date": f.filing_date,
            "accession": f.accession_number,
        }
    except Exception as exc:
        log.warning("Section fetch failed for %s/%s: %s", ticker, section, exc)
        return {"text": "", "error": str(exc)}


@app.get("/api/entity/{ticker}")
async def get_entity(ticker: str) -> dict:
    """Direct entity profile endpoint — no chat routing needed."""
    try:
        result = _handle_entity(ticker)
        return result
    except Exception as exc:
        log.warning("Entity fetch failed for %s: %s", ticker, exc)
        return {"error": str(exc)}


@app.get("/api/filing-text/{ticker}/{section}")
async def get_filing_text(
    ticker: str,
    section: str,
    accession: str | None = None,
    form_type: str | None = None,
) -> dict:
    """Direct filing text endpoint — uses the selected period's filing.

    If accession is provided, fetches that specific filing's section.
    Otherwise falls back to the latest 10-K/20-F filing.
    """
    try:
        from sec_mcp import supabase_cache

        # ── Check Supabase cache for section text ──
        cache_key = f"{form_type or '10-K'}|{accession or 'latest'}"
        section_key = section or "full"
        sb_cached = supabase_cache.get_cached(ticker.upper(), f"section_{section_key}", cache_key)
        if sb_cached and isinstance(sb_cached, dict) and sb_cached.get("text"):
            log.info("Supabase cache hit for %s/%s", ticker, section_key)
            return sb_cached

        if accession:
            # Use the specific filing the user has selected
            text = get_filing_content(
                ticker, accession,
                section=section, max_length=120000 if section else 80000,
            )
            display_limit = 50000 if section else 30000
            result = {
                "tool": "filing_text", "ticker": ticker,
                "section": section or "full filing",
                "filing_date": "",
                "accession": accession,
                "form_type": form_type or "",
                "text_length": len(text) if text else 0,
                "text": (text or "")[:display_limit],
            }
            # Cache in Supabase
            if result.get("text"):
                supabase_cache.set_cached(ticker.upper(), f"section_{section_key}", result, cache_key)
            return result

        # No accession — fall back to standard handler with form type fallback
        primary_form = form_type or "10-K"
        result = _handle_filing_text(ticker, section, form_type=primary_form)

        # If the primary form failed (no text), try FPI alternatives
        if result.get("error") or not result.get("text"):
            from sec_mcp.sec_client import get_form_alternatives
            alternatives = get_form_alternatives(primary_form)
            for alt_form in alternatives[1:]:
                alt_result = _handle_filing_text(ticker, section, form_type=alt_form)
                if alt_result.get("text"):
                    result = alt_result
                    break
            else:
                cross_forms = ["20-F", "10-K", "40-F"] if primary_form in ("10-Q", "6-K") else ["10-Q", "6-K"]
                for cf in cross_forms:
                    alt_result = _handle_filing_text(ticker, section, form_type=cf)
                    if alt_result.get("text"):
                        result = alt_result
                        break

        # Cache successful result in Supabase
        if result.get("text"):
            supabase_cache.set_cached(ticker.upper(), f"section_{section_key}", result, cache_key)

        return result
    except Exception as exc:
        log.warning("Filing text fetch failed for %s/%s: %s", ticker, section, exc)
        return {"tool": "filing_text", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
#  Perplexity real-time search endpoint
# ═══════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    ticker: str = ""


@app.post("/api/search-realtime")
async def search_realtime(req: SearchRequest):
    """Real-time web search via Perplexity for financial data validation."""
    try:
        from sec_mcp.perplexity_client import search, is_available
        if not is_available():
            return {"error": "Perplexity API not configured", "available": False}

        result = search(req.query, ticker=req.ticker or None)
        if not result:
            return {"error": "Search failed"}

        return {
            "content": result["content"],
            "citations": result.get("citations", []),
            "ticker": req.ticker,
            "query": req.query,
        }
    except Exception as e:
        log.exception("Real-time search failed")
        return {"error": str(e)}


@app.get("/api/news/{ticker}")
async def get_news(ticker: str):
    """Get latest financial news for a ticker via Perplexity."""
    try:
        from sec_mcp.perplexity_client import search_financial_news, is_available
        if not is_available():
            return {"error": "Perplexity API not configured"}

        result = search_financial_news(ticker.upper())
        if not result:
            return {"error": "No news found"}

        return {
            "ticker": ticker.upper(),
            "content": result["content"],
            "citations": result.get("citations", []),
        }
    except Exception as e:
        log.exception("News fetch failed for %s", ticker)
        return {"error": str(e)}


@app.post("/api/validate-metric")
async def validate_metric_endpoint(request: Request):
    """Validate a specific financial metric against public sources."""
    try:
        body = await request.json()
        ticker = body.get("ticker", "")
        metric = body.get("metric", "")
        value = body.get("value", 0)

        from sec_mcp.perplexity_client import validate_metric, is_available
        if not is_available():
            return {"error": "Perplexity API not configured"}

        result = validate_metric(ticker.upper(), metric, float(value))
        return result or {"error": "Validation failed"}
    except Exception as e:
        log.exception("Metric validation failed")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  Polygon.io cross-validation endpoint
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/cross-check/{ticker}")
async def cross_check_ticker(ticker: str):
    """Cross-check SEC XBRL data against Polygon.io standardized financials."""
    try:
        from sec_mcp.polygon_client import cross_check, get_ticker_details, is_available
        if not is_available():
            return {"error": "Polygon API not configured", "available": False}

        # Get SEC data first
        sec_result = _handle_financials(ticker.upper(), None)
        sec_data = sec_result.get("data") or {}
        sec_metrics = sec_data.get("metrics") or {}

        # Cross-check against Polygon
        validation = cross_check(ticker.upper(), sec_metrics)
        details = get_ticker_details(ticker.upper())

        return {
            "ticker": ticker.upper(),
            "validation": validation,
            "polygon_details": details,
            "sec_filing": (sec_data.get("filing_info") or {}).get("accession_number"),
        }
    except Exception as e:
        log.exception("Cross-check failed for %s", ticker)
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  Public API (v1) — rate-limited, key-authenticated
# ═══════════════════════════════════════════════════════════════════════════

# Simple in-memory API key store (replace with DB lookup for production)
_API_KEYS: dict[str, dict] = {}  # key -> {tier: "free"|"pro"|"lifetime", calls: int, limit: int}


def _check_api_key(request) -> dict | None:
    """Validate API key from X-API-Key header. Returns key info or None."""
    key = request.headers.get("x-api-key", "")
    if not key:
        return None
    return _API_KEYS.get(key)


@app.get("/v1/financials/{ticker}")
async def api_v1_financials(request: Request, ticker: str, year: int | None = None, form_type: str = "10-K"):
    """Public API: Get financial data for a ticker.

    Requires X-API-Key header. Returns JSON with metrics, statements, segments.
    """
    # For now, allow unauthenticated access (paywall will be added via Stripe)
    # key_info = _check_api_key(request)
    # if not key_info:
    #     return JSONResponse({"error": "API key required. Get one at /docs"}, status_code=401)

    result = _handle_financials(ticker.upper(), year, form_type)
    data = result.get("data") or {}
    return {
        "ticker": ticker.upper(),
        "company": data.get("company_name"),
        "period": data.get("fiscal_period"),
        "year": data.get("fiscal_year"),
        "metrics": data.get("metrics", {}),
        "filing": data.get("filing_info"),
        "summary": result.get("summary"),
    }


@app.get("/v1/compare")
async def api_v1_compare(request: Request, tickers: str, year: int | None = None, form_type: str = "10-K"):
    """Public API: Compare multiple tickers. Pass comma-separated tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list or len(ticker_list) > 10:
        return {"error": "Provide 1-10 comma-separated tickers"}

    result = _handle_compare(ticker_list, year, form_type)
    return {
        "tickers": ticker_list,
        "results": [
            {
                "ticker": (r["data"] or {}).get("ticker_or_cik"),
                "company": (r["data"] or {}).get("company_name"),
                "metrics": (r["data"] or {}).get("metrics", {}),
                "summary": r.get("summary"),
            }
            for r in result.get("results", [])
        ],
        "narrative": result.get("comparison_narrative"),
    }


@app.get("/v1/search")
async def api_v1_search(request: Request, q: str):
    """Public API: Search companies by name or ticker."""
    results = search_companies(q, limit=10)
    return {"query": q, "results": results}


@app.get("/v1/cross-check/{ticker}")
async def api_v1_cross_check(request: Request, ticker: str):
    """Public API: Cross-check SEC data against Polygon.io."""
    return await cross_check_ticker(ticker)


# ═══════════════════════════════════════════════════════════════════════════
#  API Documentation page
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/docs")
async def api_docs():
    """Interactive API documentation page."""
    html = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Fineas API — Documentation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0a0f1e; --bg2: #111827; --bg3: #1e293b;
      --text: #f1f5f9; --text2: #94a3b8; --text3: #64748b;
      --brand: #6366f1; --brand-light: #818cf8;
      --success: #10b981; --danger: #ef4444; --warning: #f59e0b;
      --border: rgba(148,163,184,0.1);
      --radius: 12px;
    }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
    .container { max-width: 900px; margin: 0 auto; padding: 40px 24px; }
    .hero { text-align: center; padding: 60px 0 40px; }
    .hero h1 { font-size: 42px; font-weight: 800; letter-spacing: -0.03em; }
    .hero h1 span { background: linear-gradient(135deg, var(--brand), var(--brand-light)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { color: var(--text2); font-size: 18px; margin-top: 12px; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }
    .badge-free { background: rgba(16,185,129,0.15); color: var(--success); }
    .badge-pro { background: rgba(99,102,241,0.15); color: var(--brand-light); }

    /* Pricing */
    .pricing { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 40px 0; }
    .price-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; text-align: center; transition: all 0.2s; }
    .price-card:hover { border-color: var(--brand); transform: translateY(-2px); box-shadow: 0 8px 32px rgba(99,102,241,0.15); }
    .price-card.featured { border-color: var(--brand); background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(99,102,241,0.02)); }
    .price-card h3 { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
    .price-amount { font-size: 36px; font-weight: 800; letter-spacing: -0.03em; }
    .price-amount span { font-size: 14px; color: var(--text2); font-weight: 400; }
    .price-card ul { list-style: none; margin: 20px 0; text-align: left; }
    .price-card li { padding: 6px 0; font-size: 14px; color: var(--text2); }
    .price-card li::before { content: "✓ "; color: var(--success); font-weight: 700; }
    .price-btn { width: 100%; padding: 10px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    .price-btn-primary { background: var(--brand); color: white; }
    .price-btn-primary:hover { background: var(--brand-light); }
    .price-btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .price-btn-outline:hover { border-color: var(--brand); }

    /* Endpoints */
    .endpoint { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); margin: 16px 0; overflow: hidden; }
    .endpoint-header { display: flex; align-items: center; gap: 12px; padding: 16px 20px; cursor: pointer; }
    .endpoint-header:hover { background: rgba(99,102,241,0.04); }
    .method { padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; }
    .method-get { background: rgba(16,185,129,0.15); color: var(--success); }
    .method-post { background: rgba(59,130,246,0.15); color: #3b82f6; }
    .endpoint-path { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 500; }
    .endpoint-desc { color: var(--text2); font-size: 13px; margin-left: auto; }
    .endpoint-body { padding: 0 20px 20px; display: none; }
    .endpoint.open .endpoint-body { display: block; }
    .code-block { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; font-family: 'JetBrains Mono', monospace; font-size: 13px; overflow-x: auto; margin: 12px 0; white-space: pre; color: var(--text2); }
    .param-table { width: 100%; border-collapse: collapse; margin: 12px 0; }
    .param-table th { text-align: left; padding: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text3); border-bottom: 1px solid var(--border); }
    .param-table td { padding: 8px; font-size: 13px; border-bottom: 1px solid var(--border); }
    .param-table td:first-child { font-family: 'JetBrains Mono', monospace; color: var(--brand-light); }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .tag-required { background: rgba(239,68,68,0.15); color: var(--danger); }
    .tag-optional { background: rgba(148,163,184,0.1); color: var(--text3); }
    h2 { font-size: 24px; font-weight: 700; margin: 40px 0 16px; letter-spacing: -0.02em; }
    .section-desc { color: var(--text2); font-size: 14px; margin-bottom: 20px; }
    .back-link { color: var(--brand-light); text-decoration: none; font-size: 14px; }
    .back-link:hover { text-decoration: underline; }
    footer { text-align: center; padding: 40px 0; color: var(--text3); font-size: 13px; border-top: 1px solid var(--border); margin-top: 60px; }

    @media (max-width: 768px) { .pricing { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="container">
  <a href="/" class="back-link">← Back to Dashboard</a>

  <div class="hero">
    <h1>Fineas <span>API</span></h1>
    <p>SEC filing data, financial metrics, and AI analysis — one API call away</p>
  </div>

  <!-- Pricing -->
  <h2>Pricing</h2>
  <div class="pricing">
    <div class="price-card">
      <h3>Starter</h3>
      <div class="price-amount">$5<span>/mo</span></div>
      <ul>
        <li>100 API calls/day</li>
        <li>SEC XBRL financials</li>
        <li>Company search</li>
        <li>JSON response format</li>
      </ul>
      <button class="price-btn price-btn-outline" onclick="alert('Coming soon — Stripe integration')">Get Started</button>
    </div>
    <div class="price-card featured">
      <h3>Pro <span class="badge badge-pro">POPULAR</span></h3>
      <div class="price-amount">$100<span>/yr</span></div>
      <ul>
        <li>Unlimited API calls</li>
        <li>Cross-validation (Polygon)</li>
        <li>AI-powered insights</li>
        <li>Comparables engine</li>
        <li>Priority support</li>
      </ul>
      <button class="price-btn price-btn-primary" onclick="alert('Coming soon — Stripe integration')">Subscribe</button>
    </div>
    <div class="price-card">
      <h3>Lifetime</h3>
      <div class="price-amount">$10k</div>
      <ul>
        <li>Everything in Pro</li>
        <li>Lifetime access</li>
        <li>Custom integrations</li>
        <li>Direct support line</li>
        <li>Early access to features</li>
      </ul>
      <button class="price-btn price-btn-outline" onclick="alert('Contact austin@fineas.ai')">Contact Us</button>
    </div>
  </div>

  <!-- Authentication -->
  <h2>Authentication</h2>
  <p class="section-desc">Include your API key in the <code>X-API-Key</code> header with every request.</p>
  <div class="code-block">curl -H "X-API-Key: YOUR_KEY" \\
  https://sec-mcp-production.up.railway.app/v1/financials/AAPL</div>

  <!-- Endpoints -->
  <h2>Endpoints</h2>
  <p class="section-desc">All endpoints return JSON. Base URL: <code>https://sec-mcp-production.up.railway.app</code></p>

  <div class="endpoint open">
    <div class="endpoint-header" onclick="this.parentElement.classList.toggle('open')">
      <span class="method method-get">GET</span>
      <span class="endpoint-path">/v1/financials/{ticker}</span>
      <span class="endpoint-desc">Get financial data from SEC filings</span>
    </div>
    <div class="endpoint-body">
      <table class="param-table">
        <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td>ticker</td><td>string <span class="tag tag-required">required</span></td><td>Stock ticker (e.g. AAPL, MSFT)</td></tr>
          <tr><td>year</td><td>int <span class="tag tag-optional">optional</span></td><td>Fiscal year (default: latest)</td></tr>
          <tr><td>form_type</td><td>string <span class="tag tag-optional">optional</span></td><td>10-K, 10-Q, 20-F, 6-K (default: 10-K)</td></tr>
        </tbody>
      </table>
      <strong>Response</strong>
      <div class="code-block">{
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "period": "FY",
  "year": 2024,
  "metrics": {
    "revenue": 391035000000,
    "net_income": 93736000000,
    "gross_margin": 0.462,
    "eps_diluted": 6.13,
    ...
  },
  "filing": { "accession_number": "...", "filing_date": "..." },
  "summary": "Apple reported $391B revenue..."
}</div>
    </div>
  </div>

  <div class="endpoint">
    <div class="endpoint-header" onclick="this.parentElement.classList.toggle('open')">
      <span class="method method-get">GET</span>
      <span class="endpoint-path">/v1/compare?tickers=AAPL,MSFT,GOOG</span>
      <span class="endpoint-desc">Compare multiple companies</span>
    </div>
    <div class="endpoint-body">
      <table class="param-table">
        <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td>tickers</td><td>string <span class="tag tag-required">required</span></td><td>Comma-separated tickers (max 10)</td></tr>
          <tr><td>year</td><td>int <span class="tag tag-optional">optional</span></td><td>Fiscal year</td></tr>
          <tr><td>form_type</td><td>string <span class="tag tag-optional">optional</span></td><td>Filing form type</td></tr>
        </tbody>
      </table>
      <strong>Response</strong>
      <div class="code-block">{
  "tickers": ["AAPL", "MSFT", "GOOG"],
  "results": [
    { "ticker": "AAPL", "company": "Apple Inc.", "metrics": {...}, "summary": "..." },
    ...
  ],
  "narrative": "AI comparison analysis..."
}</div>
    </div>
  </div>

  <div class="endpoint">
    <div class="endpoint-header" onclick="this.parentElement.classList.toggle('open')">
      <span class="method method-get">GET</span>
      <span class="endpoint-path">/v1/search?q=apple</span>
      <span class="endpoint-desc">Search companies by name or ticker</span>
    </div>
    <div class="endpoint-body">
      <table class="param-table">
        <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td>q</td><td>string <span class="tag tag-required">required</span></td><td>Search query</td></tr>
        </tbody>
      </table>
      <strong>Response</strong>
      <div class="code-block">{
  "query": "apple",
  "results": [
    { "ticker": "AAPL", "name": "Apple Inc.", "cik": "0000320193" },
    ...
  ]
}</div>
    </div>
  </div>

  <div class="endpoint">
    <div class="endpoint-header" onclick="this.parentElement.classList.toggle('open')">
      <span class="method method-get">GET</span>
      <span class="endpoint-path">/v1/cross-check/{ticker}</span>
      <span class="endpoint-desc">Validate SEC data against Polygon.io</span>
    </div>
    <div class="endpoint-body">
      <p style="color:var(--text2);font-size:13px;margin-bottom:12px">Cross-checks SEC XBRL extraction against Polygon's standardized financial data. Flags discrepancies >5%.</p>
      <table class="param-table">
        <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td>ticker</td><td>string <span class="tag tag-required">required</span></td><td>Stock ticker</td></tr>
        </tbody>
      </table>
      <strong>Response</strong>
      <div class="code-block">{
  "ticker": "AAPL",
  "validation": {
    "revenue": { "sec": 391035000000, "polygon": 391035000000, "diff_pct": 0.0, "match": true },
    "net_income": { "sec": 93736000000, "polygon": 93736000000, "diff_pct": 0.0, "match": true },
    ...
  },
  "polygon_details": { "name": "Apple Inc.", "market_cap": 3200000000000, ... }
}</div>
    </div>
  </div>

  <!-- Try It -->
  <h2>Try It Live</h2>
  <p class="section-desc">Test the API directly from your browser. No API key required during beta.</p>
  <div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px">
    <div style="display:flex;gap:8px;margin-bottom:12px">
      <select id="try-endpoint" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:13px;flex-shrink:0">
        <option value="financials">GET /v1/financials/{ticker}</option>
        <option value="compare">GET /v1/compare?tickers=</option>
        <option value="search">GET /v1/search?q=</option>
        <option value="cross-check">GET /v1/cross-check/{ticker}</option>
      </select>
      <input id="try-input" placeholder="AAPL" value="AAPL" style="flex:1;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:13px;font-family:'JetBrains Mono',monospace"/>
      <button onclick="tryApi()" style="background:var(--brand);color:white;border:none;border-radius:8px;padding:8px 20px;font-size:13px;font-weight:600;cursor:pointer">Send</button>
    </div>
    <pre id="try-result" style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:16px;font-family:'JetBrains Mono',monospace;font-size:12px;overflow-x:auto;max-height:400px;overflow-y:auto;color:var(--text2);white-space:pre-wrap">Click Send to test the API...</pre>
  </div>
  <script>
  async function tryApi() {
    const ep = document.getElementById('try-endpoint').value;
    const val = document.getElementById('try-input').value.trim();
    const out = document.getElementById('try-result');
    if (!val) { out.textContent = 'Enter a value'; return; }
    out.textContent = 'Loading...';
    let url;
    if (ep === 'financials') url = '/v1/financials/' + encodeURIComponent(val);
    else if (ep === 'compare') url = '/v1/compare?tickers=' + encodeURIComponent(val);
    else if (ep === 'search') url = '/v1/search?q=' + encodeURIComponent(val);
    else if (ep === 'cross-check') url = '/v1/cross-check/' + encodeURIComponent(val);
    try {
      const r = await fetch(url);
      const j = await r.json();
      out.textContent = JSON.stringify(j, null, 2);
    } catch(e) { out.textContent = 'Error: ' + e.message; }
  }
  </script>

  <!-- Rate Limits -->
  <h2>Rate Limits</h2>
  <div class="endpoint">
    <div class="endpoint-body" style="display:block">
      <table class="param-table">
        <thead><tr><th>Tier</th><th>Rate Limit</th><th>Data Sources</th></tr></thead>
        <tbody>
          <tr><td>Starter ($5/mo)</td><td>100 calls/day</td><td>SEC EDGAR XBRL</td></tr>
          <tr><td>Pro ($100/yr)</td><td>Unlimited</td><td>SEC + Polygon + AI Insights</td></tr>
          <tr><td>Lifetime ($10k)</td><td>Unlimited</td><td>Everything + Custom integrations</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <footer>
    Fineas.ai — SEC Financial Intelligence API<br/>
    Data sourced from SEC EDGAR (XBRL) · Cross-validated with Polygon.io · AI powered by Claude
  </footer>
</div>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/")
async def index():
    """Serve the main dashboard HTML."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    import uvicorn

    # Read PORT from environment (Railway sets this) or default to 8877
    port = int(os.environ.get("PORT", "8877"))
    print(f"\n  SEC Terminal \u2192 http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
