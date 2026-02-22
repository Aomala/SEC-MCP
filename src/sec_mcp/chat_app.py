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

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sec_mcp.edgar_client import get_filing_content, list_filings, search_companies
from sec_mcp.financials import (
    extract_financials,
    extract_financials_batch,
    generate_local_summary,
)
from sec_mcp.historical import get_historical_data, run_historical_extraction
from sec_mcp.db import get_job, is_available as mongo_available
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

app = FastAPI(title="SEC Terminal")

# CORS for cross-origin access (needed for Railway deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ═══════════════════════════════════════════════════════════════════════════
#  Health check (Railway uses this to verify the service is running)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check endpoint for Railway deployment."""
    return {
        "status": "ok",
        "mongodb": "connected" if mongo_available() else "unavailable",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Tool handlers — each returns a dict for the frontend
# ═══════════════════════════════════════════════════════════════════════════

def _handle_financials(ticker: str, year: int | None, form_type: str = "10-K") -> dict:
    """Handle a financial data request for a single company."""
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
        # Cache by accession so period switches are instant
        acc = (data.get("filing_info") or {}).get("accession_number")
        if acc:
            summary = generate_local_summary(data)
            _rcache_put(ticker, acc, data, summary)
            disk_cache.put(ticker, acc, data, summary)
        else:
            summary = generate_local_summary(data)
    else:
        summary = "No data available."
    return {"tool": "financials", "data": data, "summary": summary}


def _handle_compare(tickers: list[str], year: int | None, form_type: str = "10-K") -> dict:
    """Handle a comparison request for multiple companies."""
    try:
        results = extract_financials_batch(
            tickers, year=year, form_type=form_type,
            include_statements=True, include_segments=True,
        )
    except Exception as e:
        log.exception("Compare batch failed")
        return {"tool": "compare", "error": str(e), "results": []}

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
            except Exception:
                pass

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
            except Exception:
                pass

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
            if mongo_available():
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
        if tool in ("financials", "explain") and mongo_available():
            job = get_job(tickers[0])
            if job is None or job.get("status") != "complete":
                bg.add_task(run_historical_extraction, tickers[0])

        return {"type": "result", **meta, **result}
    except Exception:
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
    if not mongo_available():
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
            except Exception:
                pass

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
            "ALWAYS:\n"
            "- Reference specific numbers from the data (format: **$1.23B**, **$456M**, **12.3%**).\n"
            "- Start with a direct answer on the first line — no preamble.\n"
            "- Cite the filing source (e.g., 'per the 10-K filed 2024-11-01').\n"
            "- Use **bold** for key numbers, terms, and company names.\n"
            "- Use bullet points (- item) for lists of 3+ items.\n"
            "- Compare to prior periods when data is available — calculate % changes.\n"
            "- If data is missing, say exactly what's missing and suggest how to get it.\n\n"
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
        if accession:
            # Use the specific filing the user has selected
            text = get_filing_content(
                ticker, accession,
                section=section, max_length=120000 if section else 80000,
            )
            display_limit = 50000 if section else 30000
            return {
                "tool": "filing_text", "ticker": ticker,
                "section": section or "full filing",
                "filing_date": "",
                "accession": accession,
                "form_type": form_type or "",
                "text_length": len(text) if text else 0,
                "text": (text or "")[:display_limit],
            }
        # No accession — fall back to standard handler (latest filing)
        result = _handle_filing_text(ticker, section, form_type=form_type or "10-K")
        return result
    except Exception as exc:
        log.warning("Filing text fetch failed for %s/%s: %s", ticker, section, exc)
        return {"tool": "filing_text", "error": str(exc)}


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
