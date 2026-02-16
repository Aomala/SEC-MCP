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

import json
import logging
import os
import re
import time
import traceback
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sec_mcp.edgar_client import get_filing_content, list_filings, search_companies
from sec_mcp.financials import (
    extract_financials,
    extract_financials_batch,
    generate_local_summary,
)
from sec_mcp.historical import get_historical_data, run_historical_extraction
from sec_mcp.db import get_job, is_available as mongo_available

log = logging.getLogger(__name__)

app = FastAPI(title="SEC Terminal")

# CORS for cross-origin access (needed for Railway deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
#  Company alias resolver — maps common names to tickers
# ═══════════════════════════════════════════════════════════════════════════

_ALIASES_PATH = Path(__file__).parent / "company_aliases.json"


def _load_aliases() -> dict[str, str]:
    """Load company name → ticker aliases from JSON file."""
    if _ALIASES_PATH.exists():
        try:
            return json.loads(_ALIASES_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_alias(name: str, ticker: str):
    """Persist a new company alias for future lookups."""
    data = _load_aliases()
    data[name.lower().strip()] = ticker.upper().strip()
    try:
        _ALIASES_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def resolve_name(text: str) -> str | None:
    """Resolve a company name/alias to a ticker symbol."""
    aliases = _load_aliases()
    low = text.lower().strip()
    if low in aliases:
        return aliases[low]
    # Try partial match (longest alias first for specificity)
    for name in sorted(aliases, key=len, reverse=True):
        if name in low:
            return aliases[name]
    return None


def learn_company(ticker: str, name: str):
    """Auto-learn a new company alias from API responses."""
    if name and ticker and len(name) > 2:
        clean = name.lower().strip()
        aliases = _load_aliases()
        if clean not in aliases:
            _save_alias(clean, ticker)


# ═══════════════════════════════════════════════════════════════════════════
#  Intent parser — extract tool, tickers, year, section from user message
# ═══════════════════════════════════════════════════════════════════════════

# Stop words that should not be treated as ticker symbols
_STOP = {
    "GET", "SHOW", "FIND", "FOR", "THE", "INFO", "DATA", "AND", "WITH",
    "YEAR", "FISCAL", "GIVE", "PLEASE", "CAN", "YOU", "ME", "ABOUT",
    "WHAT", "HOW", "MUCH", "DOES", "MAKE", "LATEST", "RECENT", "THEIR",
    "ALL", "FULL", "DETAILS", "DETAILED", "FROM", "SEC", "EDGAR",
    "VS", "VERSUS", "COMPARE", "EXPLAIN", "ANALYZE", "SUMMARIZE",
    "RISK", "FACTORS", "FILING", "SENTIMENT", "ENTITIES",
    "WHO", "CEO", "CFO", "TELL", "WHERE", "WHEN", "WHY", "WHICH",
    "HEADQUARTERED", "FOUNDED", "INDUSTRY", "SECTOR",
}

# Entity-related trigger phrases
_ENTITY_TRIGGERS = [
    "who is", "who's the", "ceo of", "cfo of", "chief executive",
    "chief financial", "headquarters", "headquartered",
    "founded", "when was", "where is", "company info", "company profile",
    "tell me about", "what does.*do", "sector", "industry of",
    "website", "address", "phone", "officers",
]

# Q&A trigger phrases — freeform questions about data
_QA_TRIGGERS = [
    "why", "how come", "what caused", "explain why", "reason for",
    "what do you think", "interpret", "insight", "opinion",
    "what should i", "is it good", "is it bad", "healthy",
    "concern", "red flag", "strength", "weakness",
    "revenue yoy", "margin trend", "cash flow analysis",
]


def _parse_intent(msg: str) -> dict:
    """Parse a user message to determine which tool to invoke and with what parameters.

    Returns:
        {tool, tickers, year, section, form_type, raw, reasoning}
    """
    low = msg.lower().strip()
    upper = msg.upper().strip()
    reasoning_parts = []

    # Extract year (2010-2029)
    ym = re.search(r"\b(20[12]\d)\b", msg)
    year = int(ym.group(1)) if ym else None
    if year:
        reasoning_parts.append(f"Detected year: {year}")

    # Detect form type
    form_type = "10-K"
    if re.search(r"\b(10-?[Qq]|quarterly|quarter)\b", low):
        form_type = "10-Q"
        reasoning_parts.append("Detected quarterly (10-Q) request")
    if re.search(r"\b[Qq]([1-4])\b", msg):
        form_type = "10-Q"
        reasoning_parts.append("Detected specific quarter reference")

    # Check for entity profile queries first (CEO, company info, etc.)
    tool = "financials"
    is_entity_query = any(re.search(pat, low) for pat in _ENTITY_TRIGGERS)
    is_qa_query = any(kw in low for kw in _QA_TRIGGERS)

    if is_entity_query:
        tool = "entity"
        reasoning_parts.append("Routing to Entity Profile (detected entity/company info question)")
    elif is_qa_query:
        tool = "qa"
        reasoning_parts.append("Routing to AI Q&A (detected analytical/interpretive question)")
    elif any(w in low for w in ("sentiment", "mood", "positive", "negative")):
        tool = "sentiment"
        reasoning_parts.append("Routing to Sentiment Analysis")
    elif any(w in low for w in ("summarize", "summary", "tldr")):
        tool = "summary"
        reasoning_parts.append("Routing to Summary tool")
    elif any(w in low for w in ("compare", " vs ", "versus")):
        tool = "compare"
        reasoning_parts.append("Routing to Compare tool")
    elif any(w in low for w in ("explain", "analyze", "analysis")):
        tool = "explain"
        reasoning_parts.append("Routing to Explain tool (Claude narrative)")
    elif any(w in low for w in ("filing", "10-k text", "section")):
        tool = "filing_text"
        reasoning_parts.append("Routing to Filing Text extraction")
    elif any(w in low for w in ("history", "historical", "10 year", "decade", "all filings", "timeline")):
        tool = "historical"
        reasoning_parts.append("Routing to Historical data")

    # Detect section requests (risk factors, md&a, etc.)
    section = None
    _SEC_MAP = {
        "risk factor": "risk_factors", "risk factors": "risk_factors",
        "business overview": "business", "business description": "business",
        "md&a": "mda", "mda": "mda",
        "management discussion": "mda", "management's discussion": "mda",
        "discussion and analysis": "mda",
        "financial statement": "financial_statements",
        "legal proceeding": "legal", "legal proceedings": "legal",
        "properties": "properties",
        "controls and procedures": "controls", "internal controls": "controls",
        "executive compensation": "executive_compensation",
    }
    for phrase, sec in _SEC_MAP.items():
        if phrase in low:
            tool = "filing_text"
            section = sec
            reasoning_parts.append(f"Detected section request: {phrase} → {sec}")
            break

    # "business" alone is too generic — only trigger if it looks intentional
    if section is None and re.search(r"\bbusiness\b", low) and any(
        w in low for w in ("item 1", "section", "describe", "overview", "10-k")
    ):
        tool = "filing_text"
        section = "business"
        reasoning_parts.append("Detected business section request")

    if not reasoning_parts:
        reasoning_parts.append("Default routing to Financials tool")

    # Resolve company tickers from the message
    tickers: list[str] = []

    # Clean the query by removing known keywords
    clean_q = low
    for kw in ("compare", "financials", "risk factors", "summarize", "summary",
               "sentiment", "explain", "analyze", "segments", "filing",
               "10-k", "10-q", "10q", "10k", "entities", "quarterly",
               "quarter", "annual", "history", "historical", "timeline",
               "decade", "all filings", "who is", "what is", "ceo of",
               "cfo of", "tell me about", "company info", "md&a", "mda",
               "management discussion"):
        clean_q = clean_q.replace(kw, " ")
    clean_q = re.sub(r"\b(20[12]\d|q[1-4])\b", " ", clean_q).strip()

    # Try full-text alias resolution first
    resolved = resolve_name(clean_q)
    if resolved:
        tickers.append(resolved)

    # Split on "vs", "versus", "and", comma for multi-company queries
    split_q = low
    for kw in ("compare", "financials", "risk factors", "summarize", "summary",
               "sentiment", "explain", "analyze", "filing", "10-k", "10-q", "entities",
               "who is", "ceo of", "cfo of", "tell me about", "md&a"):
        split_q = split_q.replace(kw, " ")
    split_q = re.sub(r"\b(20[12]\d|q[1-4])\b", " ", split_q).strip()
    parts = re.split(r"\bvs\b|\bversus\b|\band\b|,", split_q, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        r = resolve_name(part)
        if r and r not in tickers:
            tickers.append(r)

    # Fallback: extract uppercase tokens that look like tickers
    if not tickers:
        tokens = re.findall(r"\b[A-Z]{1,5}\b", upper)
        tickers = [t for t in tokens if t not in _STOP and len(t) >= 2]

    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    tickers = unique[:8]

    if tickers:
        reasoning_parts.append(f"Resolved tickers: {', '.join(tickers)}")

    # Auto-upgrade to "compare" if multiple tickers detected
    if len(tickers) > 1 and tool == "financials":
        tool = "compare"
        reasoning_parts.append("Multiple tickers → upgraded to Compare")

    return {
        "tool": tool, "tickers": tickers, "year": year,
        "section": section, "form_type": form_type, "raw": msg,
        "reasoning": reasoning_parts,
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
    summary = generate_local_summary(data) if data else "No data available."
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
                        model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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
    intent = _parse_intent(req.message)

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
    intent = _parse_intent(req.message)
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
    """Load financials for one specific filing (by accession number)."""
    data = extract_financials(
        req.ticker, accession=req.accession, form_type=req.form_type,
        include_statements=True, include_segments=True,
    )
    if data:
        learn_company(req.ticker, data.get("company_name", ""))
    summary = generate_local_summary(data) if data else ""
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
            model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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
    return HTMLResponse(HTML)


# ═══════════════════════════════════════════════════════════════════════════
#  Frontend — Crypto Tech Noir Dashboard
#  Split-panel: Chat (left 32%) + Analysis (right 68%)
#  Filing selector toolbar, fixed charts, citation popouts
# ═══════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SEC Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;scrollbar-width:thin;scrollbar-color:rgba(100,120,140,.18) transparent}
:root{
  --bg0:#0c0f14;--bg1:#13161d;--bg2:#1a1e27;--bg3:#232830;--bg4:#2e343e;
  --card:#1a1e27;--glass:rgba(26,30,39,.80);
  --bdr:rgba(255,255,255,.06);--bdr2:rgba(255,255,255,.09);--bdr3:rgba(255,255,255,.16);
  --t1:#f0f2f5;--t2:#a0a8b8;--t3:#6b7585;--t4:#454d5c;
  --acc:#4a7aff;--acc2:#3d6ae8;--deep:#2952cc;--deep2:#1e3fa6;
  --green:#34d399;--red:#ff5266;--amber:#ffb347;--purple:#a78bfa;
  --inc:#34d399;--exp:#ff6b7f;
  --glow:none;
  --ff:'Inter',system-ui,-apple-system,sans-serif;--fm:'Inter',system-ui,sans-serif;--r:16px;
  --shadow:0 2px 12px rgba(0,0,0,.25);
}
body.light{
  --bg0:#f7f8fa;--bg1:#ffffff;--bg2:#f0f2f5;--bg3:#e4e7ec;--bg4:#d1d5dc;
  --card:#ffffff;--glass:rgba(255,255,255,.85);
  --bdr:rgba(0,0,0,.06);--bdr2:rgba(0,0,0,.09);--bdr3:rgba(0,0,0,.16);
  --t1:#1a1d23;--t2:#4a5060;--t3:#7a8294;--t4:#a8b0c0;
  --acc:#3b68d9;--acc2:#2e55bf;--deep:#2448a8;--deep2:#1a3580;
  --green:#22a870;--red:#e0324e;--amber:#d48a00;
  --inc:#22a870;--exp:#e0324e;
  --shadow:0 2px 12px rgba(0,0,0,.06);
}
html,body{height:100%;background:var(--bg0);color:var(--t1);font:15px/1.55 var(--ff);
  overflow:hidden;-webkit-font-smoothing:antialiased}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(100,120,140,.18);border-radius:6px}

.app{display:flex;height:100vh}

/* ═══ CHAT PANEL (left 32%) ═══ */
.cp{width:32%;min-width:290px;max-width:420px;flex-shrink:0;display:flex;flex-direction:column;
  background:var(--bg1);border-right:1px solid var(--bdr);position:relative;z-index:2}
.cp-hdr{padding:18px 20px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:10px}
.cp-logo{width:36px;height:36px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  background:var(--acc);font:700 14px var(--ff);color:#fff}
.cp-hdr h1{font:700 18px var(--ff);letter-spacing:-.4px}
.cp-hdr .tag{font:500 9px/1 var(--ff);color:var(--acc);padding:3px 10px;
  background:rgba(74,122,255,.08);border-radius:20px;letter-spacing:.5px;text-transform:uppercase}
.cp-hdr .spacer{margin-left:auto}
.theme-btn{width:32px;height:32px;border-radius:50%;border:1px solid var(--bdr2);background:var(--bg2);
  color:var(--t3);font-size:15px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:.2s}
.theme-btn:hover{border-color:var(--bdr3);color:var(--t1);background:var(--bg3);transform:scale(1.05)}

.chat{flex:1;overflow-y:auto;padding:16px 18px;display:flex;flex-direction:column;gap:14px}

.welcome{margin:auto;text-align:center;max-width:320px;animation:fu .5s ease-out}
.w-ico{width:56px;height:56px;border-radius:16px;margin:0 auto 16px;display:flex;align-items:center;justify-content:center;
  background:var(--acc);font-size:22px;color:#fff}
.welcome h2{font:800 22px var(--ff);margin-bottom:6px;letter-spacing:-.5px}
.welcome p{color:var(--t3);font-size:14px;margin-bottom:18px;line-height:1.5}
.pills{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.pill{background:var(--bg2);border:1px solid var(--bdr2);border-radius:24px;padding:8px 16px;
  font:500 13px var(--ff);color:var(--t2);cursor:pointer;transition:.2s}
.pill:hover{border-color:var(--acc);color:var(--t1);background:var(--bg3);transform:translateY(-2px);
  box-shadow:var(--shadow)}

.u-row{display:flex;justify-content:flex-end}
.u-bub{max-width:82%;background:var(--bg2);border:1px solid var(--bdr2);
  border-radius:20px 20px 4px 20px;padding:12px 18px;font-size:14px;color:var(--t1);animation:fu .15s}
.a-row{display:flex;gap:10px;align-items:flex-start;animation:fu .2s}
.a-av{width:28px;height:28px;border-radius:50%;flex-shrink:0;margin-top:2px;
  background:var(--acc);display:flex;align-items:center;justify-content:center;
  font:700 11px var(--ff);color:#fff}
.a-body{flex:1;min-width:0}
.a-card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--r);
  padding:14px 18px;font-size:14px;color:var(--t2);line-height:1.65}
.a-card strong{color:var(--t1);font-weight:600}
.a-card a{color:var(--acc);text-decoration:none}

.ld-wrap{display:flex;gap:6px;padding:6px 0}
.ld{width:7px;height:7px;border-radius:50%;background:var(--acc);animation:pls 1s infinite}
.ld:nth-child(2){animation-delay:.15s}.ld:nth-child(3){animation-delay:.3s}

.inp-row{display:flex;gap:10px;padding:14px 18px 18px;border-top:1px solid var(--bdr)}
.inp-row input{flex:1;background:var(--bg2);border:1px solid var(--bdr2);border-radius:24px;
  padding:13px 20px;color:var(--t1);font:14px var(--ff);outline:none;transition:.2s}
.inp-row input:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(74,122,255,.12)}
.inp-row input::placeholder{color:var(--t4)}
.snd{width:44px;height:44px;border-radius:50%;border:none;color:#fff;font-size:17px;cursor:pointer;
  background:var(--acc);display:flex;align-items:center;justify-content:center;
  transition:.2s}
.snd:hover{transform:scale(1.06);background:var(--deep)}
.snd:disabled{opacity:.2;cursor:default;transform:none}

/* ═══ ANALYSIS PANEL (right 68%) ═══ */
.ap{flex:1;display:flex;flex-direction:column;overflow:hidden;background:var(--bg0)}

.ap-empty{flex:1;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--t4)}
.ap-empty .ico{font-size:52px;opacity:.06;margin-bottom:14px}
.ap-empty p{font-size:15px}
.ap-ld{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;color:var(--t4)}
.sp{width:40px;height:40px;border:3px solid var(--bg3);border-top-color:var(--acc);border-radius:50%;animation:spin .8s linear infinite}
.ap-ld .tm{font:600 26px var(--ff);color:var(--t2)}
.ap-ld .sub{font-size:13px;color:var(--t4)}

/* ── Toolbar: search + dropdowns ── */
.ap-toolbar{display:flex;align-items:center;gap:10px;padding:10px 18px;border-bottom:1px solid var(--bdr);
  background:var(--bg1);flex-shrink:0;flex-wrap:wrap}
.srch{position:relative;flex:1;min-width:180px;max-width:340px}
.srch input{width:100%;background:var(--bg2);border:1px solid var(--bdr2);border-radius:12px;
  padding:9px 14px 9px 34px;color:var(--t1);font:13px var(--ff);outline:none;transition:.2s}
.srch input:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(74,122,255,.10)}
.srch input::placeholder{color:var(--t4)}
.srch .mag{position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--t4);font-size:13px;pointer-events:none}
.sr-drop{position:absolute;top:100%;left:0;right:0;margin-top:4px;background:var(--card);border:1px solid var(--bdr3);
  border-radius:12px;max-height:280px;overflow-y:auto;z-index:50;box-shadow:var(--shadow);display:none}
.sr-item{display:flex;align-items:center;gap:10px;padding:10px 14px;cursor:pointer;transition:.15s;font:13px var(--ff)}
.sr-item:hover{background:rgba(74,122,255,.06)}
.sr-tk{color:var(--acc);font-weight:700;min-width:50px}
.sr-cik{color:var(--t4);min-width:80px;font-size:11px}
.sr-nm{color:var(--t2);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:13px}

.sel{background:var(--bg2);border:1px solid var(--bdr2);border-radius:12px;padding:8px 12px;
  color:var(--t2);font:13px var(--ff);outline:none;cursor:pointer;transition:.15s;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%236b7585'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center;padding-right:28px}
.sel:focus{border-color:var(--acc)}
.sel option{background:var(--card);color:var(--t2)}
.tb-sep{width:1px;height:22px;background:var(--bdr2);flex-shrink:0}
.tb-label{font:600 11px var(--ff);color:var(--t4);text-transform:uppercase;letter-spacing:.3px;flex-shrink:0}

/* ═══ Two-column split ═══ */
.split{display:flex;flex:1;overflow:hidden}
.col-l{width:40%;min-width:300px;overflow-y:auto;border-right:1px solid var(--bdr);display:flex;flex-direction:column;background:var(--bg1)}
.col-r{flex:1;display:flex;flex-direction:column;overflow:hidden;background:var(--bg1)}

/* ── Company header (left col) ── */
.dh{padding:18px 20px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:10px;flex-wrap:wrap;flex-shrink:0}
.dh-tk{font:700 16px var(--ff);color:var(--acc);background:rgba(74,122,255,.08);padding:5px 14px;
  border-radius:20px}
.dh-name{font:700 17px var(--ff);letter-spacing:-.3px}
.dh-tag{font:500 11px var(--ff);color:var(--t3);padding:4px 10px;background:var(--bg2);border-radius:20px}
.dh-link{font:500 12px var(--ff);color:var(--acc);padding:4px 12px;background:rgba(74,122,255,.06);
  border-radius:20px;text-decoration:none;transition:.15s}
.dh-link:hover{background:rgba(74,122,255,.12)}

/* ── Key Metrics grid (2x3 in left col) ── */
.km-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--bdr);flex-shrink:0}
.km{padding:16px 20px;background:var(--bg1);transition:.15s;position:relative}
.km:hover{background:var(--bg2)}
.km-label{font:500 11px var(--ff);text-transform:uppercase;letter-spacing:.3px;color:var(--t4);margin-bottom:6px;
  display:flex;align-items:center;justify-content:space-between}
.km-val{font:700 24px var(--ff);color:var(--t1);letter-spacing:-.5px}
.km-val.neg{color:var(--red)}.km-val.pos{color:var(--green)}
.km-sub{font:500 12px var(--ff);color:var(--t3);margin-top:3px}

/* ── Charts — FIXED height ── */
.ch-area{display:flex;gap:1px;background:var(--bdr);height:210px;flex-shrink:0;overflow:hidden}
.ch-box{flex:1;padding:14px 16px 10px;background:var(--bg1);position:relative;overflow:hidden;display:flex;flex-direction:column}
.ch-hdr{display:flex;align-items:center;justify-content:space-between;flex-shrink:0;margin-bottom:6px}
.ch-hdr h4{font:600 11px var(--ff);text-transform:uppercase;letter-spacing:.2px;color:var(--t3)}
.ch-box canvas{flex:1;min-height:0}

/* ── Narrative (left col) ── */
.narr-box{padding:16px 20px;border-bottom:1px solid var(--bdr);max-height:200px;overflow-y:auto}
.narr-box h3{font:700 12px var(--ff);text-transform:uppercase;letter-spacing:.3px;color:var(--t3);margin-bottom:8px}
.narr-box p{font-size:14px;color:var(--t2);line-height:1.7}

/* Inline Exec Summary (auto-generated with data) */
.exec-inline{padding:20px;border-bottom:1px solid var(--bdr);background:var(--bg1);max-height:400px;overflow-y:auto}
.exec-inline-hdr{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.exec-inline-hdr h3{font:700 13px var(--ff);text-transform:uppercase;letter-spacing:.4px;color:var(--t3)}
.exec-badge{font:600 10px var(--ff);padding:3px 10px;border-radius:12px;
  background:rgba(74,122,255,.08);color:var(--acc);letter-spacing:.3px}
.exec-badge.done{background:rgba(52,211,153,.08);color:var(--green)}
.exec-inline-body{font:400 14px/1.8 var(--ff);color:var(--t2)}
.exec-inline-body strong{color:var(--t1);font-weight:600}
.exec-inline-body ul{margin:8px 0;padding-left:20px}
.exec-inline-body li{margin:4px 0}
.exec-inline-body h4{font:700 13px var(--ff);color:var(--t1);margin:16px 0 8px}
.exec-inline-loading{display:flex;align-items:center;gap:8px;padding:8px 0}

/* ── Validation (left col) ── */
.vw{font:12px var(--ff);padding:8px 14px;border-radius:12px;margin-bottom:4px;display:flex;align-items:center;gap:8px}
.vw.err{background:rgba(255,82,100,.05);border:1px solid rgba(255,82,100,.12);color:var(--red)}
.vw.wrn{background:rgba(255,179,71,.05);border:1px solid rgba(255,179,71,.12);color:var(--amber)}

/* ═══ Main tabs (Full Data / Exec Summaries) ═══ */
.main-tabs{display:flex;gap:0;border-bottom:2px solid var(--bdr);padding:0 12px;flex-shrink:0;background:var(--bg2)}
.main-tab{padding:12px 24px;font:700 13px var(--ff);color:var(--t4);cursor:pointer;
  border-bottom:3px solid transparent;transition:.15s;text-transform:uppercase;letter-spacing:.5px}
.main-tab:hover{color:var(--t2)}.main-tab.act{color:var(--acc);border-bottom-color:var(--acc)}
.main-pane{flex:1;display:flex;flex-direction:column;overflow-y:auto}

/* ═══ Statement sub-tabs + table (right col) ═══ */
.st-tabs{display:flex;gap:0;border-bottom:1px solid var(--bdr);padding:0 12px;flex-shrink:0}
.st-tab{padding:12px 18px;font:600 13px var(--ff);color:var(--t4);cursor:pointer;
  border-bottom:2px solid transparent;transition:.15s}
.st-tab:hover{color:var(--t2)}.st-tab.act{color:var(--acc);border-bottom-color:var(--acc)}
.st-wrap{flex:1;overflow-y:auto}

/* ═══ Exec Summaries ═══ */
.exec-loading{display:flex;align-items:center;gap:12px;padding:40px 24px;color:var(--t3);font:500 14px var(--ff)}
.exec-card{padding:20px 24px;border-bottom:1px solid var(--bdr)}
.exec-card h4{font:700 14px var(--ff);color:var(--t1);margin:0 0 12px;display:flex;align-items:center;gap:8px;text-transform:uppercase;letter-spacing:.3px}
.exec-card h4 .edot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.exec-card h4 .edot.inc{background:#4a7aff}.exec-card h4 .edot.bs{background:#7c5cfc}
.exec-card h4 .edot.cf{background:#22c997}.exec-card h4 .edot.ov{background:#f59e0b}
.exec-card .exec-body{font:400 14px/1.8 var(--ff);color:var(--t2)}
.exec-card .exec-body strong{color:var(--t1);font-weight:600}
.exec-card .exec-body ul{margin:8px 0;padding-left:20px}
.exec-card .exec-body li{margin:4px 0}

.st{width:100%;border-collapse:collapse;font:13px var(--ff)}
.st th{position:sticky;top:0;z-index:1;background:var(--bg2);text-align:right;padding:10px 14px;
  color:var(--t4);font-weight:600;text-transform:uppercase;letter-spacing:.3px;border-bottom:1px solid var(--bdr);font-size:11px}
.st th:first-child{text-align:left;padding-left:20px}
.st td{padding:8px 14px;border-bottom:1px solid var(--bdr);color:var(--t2);text-align:right;position:relative}
.st td:first-child{text-align:left;color:var(--t1);font-weight:500;padding-left:20px}
.st tr:hover td{background:var(--bg2)}
.stmt-hdr{display:flex;align-items:center;gap:10px;padding:14px 20px;border-bottom:2px solid var(--bdr)}
.stmt-hdr h3{font:700 15px var(--ff);color:var(--t1);margin:0;text-transform:uppercase;letter-spacing:.5px}
.stmt-hdr .stmt-tk{font:600 12px var(--ff);color:var(--acc);background:rgba(74,122,255,.08);padding:3px 10px;border-radius:8px}
.st tr.sub td{color:var(--t1);font-weight:700;background:rgba(74,122,255,.04);border-top:2px solid var(--bdr);border-bottom:2px solid var(--bdr)}
.st tr.sec-hdr td{color:var(--t3);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.4px;
  padding:14px 20px 6px;border-bottom:1px solid var(--bdr);background:transparent}
.st .neg{color:var(--red)}
.st tr.row-exp td:first-child{color:var(--exp)}
.st tr.sub.row-exp td{color:var(--exp)}

/* ── Overview tab ── */
.ov-sec{padding:18px 20px;border-bottom:1px solid var(--bdr)}
.ov-sec h4{font:700 12px var(--ff);text-transform:uppercase;letter-spacing:.3px;color:var(--t3);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.ov-sec h4 .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.ov-sec h4 .dot.g{background:var(--acc)}.ov-sec h4 .dot.r{background:var(--exp)}
.ov-sec h4 .dot.b{background:var(--acc)}
.ov-row{display:flex;justify-content:space-between;align-items:center;padding:7px 12px;border-radius:10px;font:14px var(--ff);transition:.15s}
.ov-row:hover{background:var(--bg2)}.ov-row .lbl{color:var(--t2)}.ov-row .val{font-weight:600;color:var(--t1)}
.ov-row .val.neg{color:var(--red)}.ov-row .val.pos{color:var(--acc)}
.ov-row.total{font-weight:700;border-top:1px solid var(--bdr);padding-top:8px;margin-top:4px}
.ov-row.total .lbl{color:var(--t1)}.ov-row.total .val{font-size:16px}

/* ── YoY writeup ── */
.yoy-sec{padding:18px 20px;border-bottom:1px solid var(--bdr)}
.yoy-sec h3{font:700 12px var(--ff);text-transform:uppercase;letter-spacing:.3px;color:var(--t3);margin-bottom:10px}
.yoy-row{display:flex;align-items:center;gap:10px;padding:6px 0;font:14px var(--ff)}
.yoy-row .lbl{color:var(--t2);min-width:110px}
.yoy-row .val{color:var(--t1);font-weight:600;min-width:85px}
.yoy-row .delta{font-weight:600;font-size:12px;padding:3px 8px;border-radius:8px}
.yoy-row .delta.up{color:var(--acc);background:rgba(74,122,255,.08)}.yoy-row .delta.dn{color:var(--exp);background:rgba(255,82,127,.08)}
.yoy-row .prev{color:var(--t4);font-size:12px;margin-left:4px}

/* ── Fact-check icon ── */
.fc-btn{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:6px;
  background:transparent;border:1px solid transparent;color:var(--t4);cursor:pointer;font-size:10px;
  transition:.15s;margin-left:4px;vertical-align:middle;flex-shrink:0;opacity:0}
.km:hover .fc-btn,.st tr:hover .fc-btn,.ch-hdr:hover .fc-btn{opacity:1}
.fc-btn:hover{background:rgba(74,122,255,.10);border-color:var(--bdr3);color:var(--acc)}
.fc-active{opacity:1;background:rgba(74,122,255,.10);border-color:var(--bdr3);color:var(--acc)}

/* ── Fact-check popup ── */
.fc-popup{position:fixed;z-index:100;background:var(--card);border:1px solid var(--bdr3);border-radius:14px;
  padding:8px;box-shadow:var(--shadow);display:none;min-width:170px}
.fc-popup .fc-title{font:600 11px var(--ff);color:var(--t3);text-transform:uppercase;letter-spacing:.3px;
  padding:6px 8px;margin-bottom:2px}
.fc-popup a{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;
  font:500 13px var(--ff);color:var(--t2);text-decoration:none;transition:.15s}
.fc-popup a:hover{background:var(--bg2);color:var(--t1)}
.fc-popup a .fc-ico{font-size:15px;width:20px;text-align:center}

/* ── Toast notification ── */
.toast{position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(60px);
  background:var(--card);border:1px solid var(--bdr3);
  border-radius:14px;padding:12px 24px;font:500 14px var(--ff);color:var(--acc);z-index:300;
  box-shadow:var(--shadow);transition:transform .25s ease,opacity .25s ease;opacity:0;pointer-events:none}
.toast.show{transform:translateX(-50%) translateY(0);opacity:1}

/* ═══ Historical section (full-width, collapsible) ═══ */
.hist-sec{border-top:1px solid var(--bdr);flex-shrink:0}
.hist-toggle{display:flex;align-items:center;gap:8px;padding:8px 16px;cursor:pointer;transition:.12s;user-select:none}
.hist-toggle:hover{background:rgba(74,122,255,.02)}
.hist-toggle .lab{font:700 9px var(--fm);color:var(--t4);text-transform:uppercase;letter-spacing:.8px}
.hist-toggle .arr{color:var(--t4);font-size:10px;transition:transform .2s}
.hist-toggle.open .arr{transform:rotate(90deg)}
.hist-body{display:none;overflow:hidden}
.hist-body.open{display:block}

.fbar{display:flex;align-items:center;gap:6px;padding:8px 16px;border-bottom:1px solid var(--bdr);flex-wrap:wrap;flex-shrink:0}
.fbar .lab{font:700 9px var(--fm);color:var(--t4);text-transform:uppercase;letter-spacing:.8px}
.fbtn{font:500 12px var(--ff);padding:6px 14px;border-radius:20px;border:1px solid var(--bdr2);
  background:transparent;color:var(--t3);cursor:pointer;transition:.15s}
.fbtn:hover{border-color:var(--bdr3);color:var(--t2)}
.fbtn.act{background:rgba(74,122,255,.08);border-color:rgba(74,122,255,.25);color:var(--acc)}
.fsep{width:1px;height:14px;background:var(--bdr2);margin:0 2px}
.fright{margin-left:auto;display:flex;gap:4px;align-items:center}
.fcnt{font:10px var(--fm);color:var(--t4)}
.fref{font:500 12px var(--ff);padding:6px 14px;border-radius:20px;border:1px solid var(--bdr2);
  background:transparent;color:var(--t3);cursor:pointer;transition:.15s}
.fref:hover{border-color:var(--acc);color:var(--acc)}

.prog{height:2px;background:var(--bg2);overflow:hidden;flex-shrink:0}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--acc),var(--green));transition:width .3s;box-shadow:0 0 6px rgba(74,122,255,.25)}

.tbl-area{overflow:auto;padding:0;max-height:300px}
.dt{width:100%;border-collapse:collapse;font:13px var(--ff)}
.dt th{position:sticky;top:0;z-index:2;background:var(--bg2);text-align:right;padding:10px 14px;
  color:var(--t3);font-weight:600;text-transform:uppercase;letter-spacing:.3px;border-bottom:1px solid var(--bdr);font-size:11px;white-space:nowrap}
.dt th:first-child{text-align:left;position:sticky;left:0;z-index:3;padding-left:20px}
.dt td{padding:8px 14px;border-bottom:1px solid var(--bdr);color:var(--t2);text-align:right;white-space:nowrap}
.dt td:first-child{text-align:left;position:sticky;left:0;background:var(--bg1);z-index:1;color:var(--t1);font-weight:500;padding-left:20px}
.dt tr:hover td{background:var(--bg2)}
.dt .neg{color:var(--red)}
.dt td a.src{color:var(--t4);font-size:9px;text-decoration:none;margin-left:3px;transition:.12s}
.dt td a.src:hover{color:var(--acc2)}

.tl{padding:0 18px 18px}
.tl-title{font:700 11px var(--ff);color:var(--t1);margin:14px 0 8px;display:flex;align-items:center;gap:6px}
.tl-title .dot{width:7px;height:7px;border-radius:50%;background:var(--amber);box-shadow:0 0 6px rgba(255,166,43,.25)}
.tl-item{display:flex;gap:10px;padding:8px 12px;border-left:2px solid var(--bg3);margin-left:3px;transition:.12s}
.tl-item:hover{border-color:var(--acc);background:rgba(74,122,255,.025);border-radius:0 var(--r) var(--r) 0}
.tl-date{font:10px var(--fm);color:var(--t3);flex-shrink:0;width:70px}
.tl-body{flex:1;font-size:12px;color:var(--t2)}
.tl-body .items{font:10px var(--fm);color:var(--t4);margin-top:2px}
.tl-body a{color:var(--acc2);text-decoration:none;font-size:10px}

.sum-sec{padding:14px 18px;border-top:1px solid var(--bdr)}
.sum-sec h3{font:700 10px var(--fm);color:var(--t3);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.sum-sec p{font-size:14px;color:var(--t2);line-height:1.7}

/* Modal */
.m-bg{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:200;display:flex;align-items:center;
  justify-content:center;animation:fi .15s;backdrop-filter:blur(20px)}
.m-box{background:var(--bg1);border:1px solid var(--bdr2);border-radius:20px;width:94vw;max-width:1200px;
  max-height:90vh;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.5)}
.m-hdr{display:flex;align-items:center;justify-content:space-between;padding:18px 22px;border-bottom:1px solid var(--bdr)}
.m-hdr h3{font:700 16px var(--ff)}
.m-x{background:none;border:none;color:var(--t3);font-size:17px;cursor:pointer;width:32px;height:32px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;transition:.15s}
.m-x:hover{background:var(--bg3);color:var(--t1)}
.m-body{flex:1;overflow:auto}

/* ═══ Thinking/tool routing card ═══ */
.think-av{background:var(--bg3)!important;color:var(--t3)!important;font-size:13px!important}
.think-card{background:var(--bg2);border:1px solid var(--bdr);border-radius:12px;padding:10px 14px;font-size:12px;color:var(--t3);animation:fu .2s}
.think-hdr{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.think-tool{font:700 10px var(--ff);color:var(--acc);background:rgba(74,122,255,.08);padding:3px 8px;border-radius:6px;text-transform:uppercase;letter-spacing:.3px}
.think-time{font:400 10px var(--ff);color:var(--t4);margin-left:auto}
.think-steps{display:flex;flex-direction:column;gap:2px}
.think-step{font:400 11px var(--ff);color:var(--t4);padding-left:4px}

/* ═══ Filing text view ═══ */
.filing-view{display:flex;flex-direction:column;height:100%}
.filing-hdr{padding:16px 20px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:10px;flex-wrap:wrap;flex-shrink:0;background:var(--bg1)}
.filing-sec{font:700 16px var(--ff);color:var(--t1);letter-spacing:-.3px}
.filing-summary{padding:14px 20px;border-bottom:1px solid var(--bdr);background:var(--bg1)}
.filing-summary-hdr{font:700 11px var(--ff);text-transform:uppercase;letter-spacing:.5px;color:var(--t4);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.filing-summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.filing-summary-item{padding:8px 12px;background:var(--bg2);border-radius:8px;border:1px solid var(--bdr)}
.filing-summary-label{display:block;font:500 10px var(--ff);color:var(--t4);text-transform:uppercase;letter-spacing:.3px;margin-bottom:2px}
.filing-summary-val{font:700 15px var(--mn);color:var(--t1)}.filing-summary-val.neg{color:var(--red)}
.filing-body{flex:1;overflow-y:auto;padding:24px 28px;font-size:14px;color:var(--t2);line-height:1.8}
.filing-heading{font:700 16px var(--ff);color:var(--acc);margin:24px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--bdr)}
.filing-subhead{font:600 14px var(--ff);color:var(--t1);margin:18px 0 6px}
.filing-para{margin:4px 0;color:var(--t2)}
.filing-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;text-align:center;padding:40px;color:var(--t3)}
.filing-empty-ico{font-size:48px;margin-bottom:16px;opacity:.3}
.filing-empty h3{font:700 18px var(--ff);color:var(--t2);margin-bottom:8px}
.filing-empty p{font-size:14px;color:var(--t3);max-width:420px;line-height:1.6;margin-bottom:4px}
.filing-empty ul{text-align:left;font-size:13px;color:var(--t4);margin:8px 0;padding-left:20px}
.filing-empty li{margin-bottom:4px}

/* ═══ Entity profile view ═══ */
.entity-view{padding:24px;overflow-y:auto;flex:1}
.entity-hdr{display:flex;align-items:center;gap:16px;margin-bottom:24px}
.entity-avatar{width:56px;height:56px;border-radius:16px;background:var(--acc);display:flex;align-items:center;justify-content:center;
  font:800 22px var(--ff);color:#fff;flex-shrink:0}
.entity-title h2{font:800 22px var(--ff);letter-spacing:-.5px;margin-bottom:6px}
.entity-meta{display:flex;gap:8px;flex-wrap:wrap}
.entity-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--bdr);border-radius:var(--r);overflow:hidden;margin-bottom:20px}
.entity-field{background:var(--bg1);padding:16px 18px;display:flex;flex-direction:column;gap:4px}
.entity-label{font:600 11px var(--ff);text-transform:uppercase;letter-spacing:.3px;color:var(--t4)}
.entity-val{font:500 15px var(--ff);color:var(--t1)}
.entity-val.link{color:var(--acc);text-decoration:none}
.entity-val.link:hover{text-decoration:underline}
.entity-sub{font:400 11px var(--ff);color:var(--t4)}
.entity-filing{background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r);padding:16px 18px;display:flex;align-items:center;gap:12px;margin-bottom:16px}
.entity-filing h4{font:700 13px var(--ff);color:var(--t2)}
.entity-load{margin-left:auto;background:var(--acc);color:#fff;border:none;border-radius:12px;padding:8px 18px;
  font:600 13px var(--ff);cursor:pointer;transition:.15s}
.entity-load:hover{background:var(--deep);transform:translateY(-1px)}
.entity-note{background:rgba(74,122,255,.04);border:1px solid rgba(74,122,255,.10);border-radius:12px;padding:12px 16px;
  font:13px var(--ff);color:var(--t3);line-height:1.5}

/* ═══ Compare view ═══ */
.compare-view{padding:20px;overflow-y:auto;flex:1}
.compare-hdr{display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap}
.compare-hdr h2{font:800 20px var(--ff);letter-spacing:-.5px}
.compare-pills{display:flex;gap:8px}
.compare-pill{background:var(--bg2);border:1px solid var(--bdr2);border-radius:20px;padding:8px 18px;
  font:600 14px var(--ff);color:var(--t2);cursor:pointer;transition:.15s}
.compare-pill:hover{border-color:var(--acc);color:var(--acc);background:rgba(74,122,255,.06)}
.compare-tbl{width:100%;border-collapse:collapse;font:14px var(--ff);margin-bottom:24px}
.compare-tbl th{text-align:left;padding:12px 16px;background:var(--bg2);color:var(--t3);font-weight:600;border-bottom:1px solid var(--bdr)}
.compare-tbl td{padding:10px 16px;border-bottom:1px solid var(--bdr);color:var(--t2)}
.compare-tbl td.lbl{color:var(--t1);font-weight:500}
.compare-tbl tr:hover td{background:var(--bg2)}
.compare-narr{margin-top:16px}
.compare-narr h3{font:700 13px var(--ff);color:var(--t3);margin-bottom:8px;text-transform:uppercase;letter-spacing:.3px}

/* ═══ Q&A answer citations ═══ */
.qa-cits{margin-top:10px;padding-top:8px;border-top:1px solid var(--bdr);font:12px var(--ff);color:var(--t4)}
.qa-cit{display:inline-block;background:rgba(74,122,255,.06);border:1px solid rgba(74,122,255,.12);
  border-radius:8px;padding:3px 8px;font:500 11px var(--ff);color:var(--acc);margin:2px 4px 2px 0}

/* ═══ Chatbot panel (right side) ═══ */
.cb{width:0;max-width:0;overflow:hidden;display:flex;flex-direction:column;background:var(--bg1);
  border-left:1px solid var(--bdr);transition:width .25s ease,max-width .25s ease,opacity .25s;flex-shrink:0;opacity:0}
.cb.open{width:340px;max-width:340px;opacity:1}
.cb-hdr{padding:14px 16px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:8px;flex-shrink:0}
.cb-hdr h3{font:700 15px var(--ff);letter-spacing:-.3px;flex:1}
.cb-hdr .cb-badge{font:600 9px/1 var(--ff);color:var(--acc);padding:3px 8px;
  background:rgba(74,122,255,.08);border-radius:12px;letter-spacing:.3px}
.cb-hdr .cb-close{width:28px;height:28px;border-radius:50%;border:1px solid var(--bdr2);background:transparent;
  color:var(--t3);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;transition:.15s}
.cb-hdr .cb-close:hover{background:var(--bg3);color:var(--t1)}
.cb-msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
.cb-welcome{text-align:center;padding:24px 12px;color:var(--t4);font:13px var(--ff)}
.cb-welcome .cb-wico{font-size:36px;margin-bottom:10px;opacity:.25}
.cb-welcome p{line-height:1.5}
.cb-chips{display:flex;flex-wrap:wrap;gap:5px;justify-content:center;margin-top:10px}
.cb-chip{background:var(--bg2);border:1px solid var(--bdr2);border-radius:16px;padding:6px 12px;
  font:500 11px var(--ff);color:var(--t3);cursor:pointer;transition:.15s}
.cb-chip:hover{border-color:var(--acc);color:var(--acc);background:rgba(74,122,255,.04)}
.cb-msg{display:flex;gap:8px;animation:fu .15s}
.cb-msg.user{justify-content:flex-end}
.cb-msg.user .cb-bubble{background:var(--acc);color:#fff;border-radius:16px 16px 4px 16px;max-width:85%}
.cb-msg.ai .cb-bubble{background:var(--bg2);border:1px solid var(--bdr);border-radius:4px 16px 16px 16px;max-width:92%;color:var(--t2)}
.cb-bubble{padding:10px 14px;font:13.5px/1.65 var(--ff);word-break:break-word}
.cb-bubble strong{color:var(--t1);font-weight:600}
.cb-cits{margin-top:6px;padding-top:4px;border-top:1px solid var(--bdr)}
.cb-cit{display:inline-block;background:rgba(74,122,255,.06);border:1px solid rgba(74,122,255,.12);
  border-radius:6px;padding:2px 6px;margin:2px;font:500 10px var(--ff);color:var(--acc)}
.cb-typing{display:flex;gap:4px;padding:6px 0}
.cb-typing span{width:5px;height:5px;border-radius:50%;background:var(--t4);animation:pls .8s infinite}
.cb-typing span:nth-child(2){animation-delay:.12s}
.cb-typing span:nth-child(3){animation-delay:.24s}
.cb-inp-row{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--bdr);flex-shrink:0;align-items:flex-end}
.cb-inp{flex:1;background:var(--bg2);border:1px solid var(--bdr2);border-radius:18px;
  padding:10px 14px;color:var(--t1);font:13px var(--ff);outline:none;transition:.2s;resize:none;
  max-height:80px;min-height:36px}
.cb-inp:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(74,122,255,.10)}
.cb-inp::placeholder{color:var(--t4)}
.cb-send{width:36px;height:36px;border-radius:50%;border:none;background:var(--acc);color:#fff;
  cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;transition:.15s;flex-shrink:0}
.cb-send:hover{background:var(--deep);transform:scale(1.05)}
.cb-send:disabled{opacity:.3;cursor:default;transform:none}
/* Chatbot toggle in toolbar */
.cb-toggle{display:flex;align-items:center;gap:6px;padding:7px 14px;border-radius:20px;
  border:1px solid var(--bdr2);background:transparent;color:var(--t3);font:500 12px var(--ff);
  cursor:pointer;transition:.2s;flex-shrink:0;white-space:nowrap}
.cb-toggle:hover{border-color:var(--acc);color:var(--acc);background:rgba(74,122,255,.04)}
.cb-toggle.active{background:rgba(74,122,255,.08);border-color:rgba(74,122,255,.25);color:var(--acc)}
.cb-toggle .cb-dot{width:6px;height:6px;border-radius:50%;background:var(--t4);transition:.2s}
.cb-toggle.active .cb-dot{background:var(--acc);box-shadow:0 0 6px rgba(74,122,255,.4)}

/* ═══ Floating chat FAB (bottom-right of analysis) ═══ */
.chat-fab{position:absolute;bottom:20px;right:20px;width:50px;height:50px;border-radius:50%;
  background:var(--acc);color:#fff;border:none;font-size:22px;cursor:pointer;z-index:10;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 4px 20px rgba(74,122,255,.35);transition:.2s}
.chat-fab:hover{transform:scale(1.08);box-shadow:0 6px 28px rgba(74,122,255,.45);background:var(--deep)}
.chat-fab.active{background:var(--deep2);box-shadow:0 4px 20px rgba(74,122,255,.5)}
/* Content area below permanent toolbar */
.ap-content{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}

@keyframes fu{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes fi{from{opacity:0}to{opacity:1}}
@keyframes pls{0%,100%{opacity:.15;transform:scale(.7)}50%{opacity:1;transform:scale(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head>
<body>
<div class="app">
  <div class="cp">
    <div class="cp-hdr">
      <div class="cp-logo">S</div>
      <h1>SEC Terminal</h1>
      <div class="spacer"></div>
      <span class="tag">LIVE</span>
      <button class="theme-btn" id="theme-btn" onclick="toggleTheme()" title="Toggle light/dark mode">&#x263E;</button>
    </div>
    <div class="chat" id="chat">
      <div class="welcome" id="welcome">
        <div class="w-ico">&#9826;</div>
        <h2>Financial Intelligence</h2>
        <p>Query any public company. Analysis renders on the right panel.</p>
        <div class="pills">
          <span class="pill" onclick="q('Apple')">Apple</span>
          <span class="pill" onclick="q('NVDA 10-Q')">NVDA Quarterly</span>
          <span class="pill" onclick="q('compare AAPL vs MSFT')">AAPL vs MSFT</span>
          <span class="pill" onclick="q('Who is Apple\'s CEO?')">Apple CEO</span>
          <span class="pill" onclick="q('TSLA md&a')">Tesla MD&A</span>
          <span class="pill" onclick="q('risk factors NVDA')">NVDA Risks</span>
        </div>
      </div>
    </div>
    <div class="inp-row">
      <input id="inp" placeholder="Company name or ticker..." autocomplete="off"/>
      <button class="snd" id="btn" onclick="send()">&#8594;</button>
    </div>
  </div>

  <div class="ap" id="apanel">
    <div class="ap-toolbar" id="ap-toolbar">
      <div class="srch"><span class="mag">&#x1F50E;</span>
        <input id="asrch" placeholder="Search ticker or company..." oninput="debSearch(this.value)" autocomplete="off"/>
        <div class="sr-drop" id="sr-drop"></div>
      </div>
      <div class="tb-sep"></div>
      <span class="tb-label">View</span>
      <select id="sel-tool" class="sel" onchange="onToolSel()">
        <option value="dashboard">Dashboard</option>
        <option value="mda">MD&amp;A</option>
        <option value="risk_factors">Risk Factors</option>
        <option value="business">Business Overview</option>
        <option value="entity">Entity Profile</option>
        <option value="executive_compensation">Exec Compensation</option>
      </select>
      <div class="tb-sep"></div>
      <span class="tb-label">Form</span>
      <select id="sel-form" class="sel" onchange="onFormSel()">
        <option value="">All</option><option value="10-K">10-K Annual</option><option value="10-Q">10-Q Quarterly</option><option value="20-F">20-F Annual (FPI)</option><option value="6-K">6-K Interim (FPI)</option>
      </select>
      <span class="tb-label">Period</span>
      <select id="sel-period" class="sel" onchange="onPeriodSel()"><option value="">Select Period</option></select>
      <button class="cb-toggle" id="cb-btn" onclick="toggleCb()"><span class="cb-dot"></span>AI Assistant</button>
    </div>
    <div class="ap-content" id="ap-content">
      <div class="ap-empty" id="ap-empty">
        <div><div class="ico">&#x25C8;</div><p>Query a company to begin analysis</p></div>
      </div>
    </div>
    <button class="chat-fab" id="chat-fab" onclick="toggleCb()" title="AI Assistant">&#x1F4AC;</button>
  </div>

  <div class="cb" id="cbpanel">
    <div class="cb-hdr">
      <h3>AI Assistant</h3>
      <span class="cb-badge">CLAUDE</span>
      <button class="cb-close" onclick="toggleCb()" title="Close panel">&#x2715;</button>
    </div>
    <div class="cb-msgs" id="cb-msgs">
      <div class="cb-welcome" id="cb-welcome">
        <div class="cb-wico">&#x1F4CA;</div>
        <p><strong>Ask anything</strong> about the data on screen</p>
        <p style="font-size:11px;margin-top:6px">Executive summaries, comparisons, insights</p>
        <div class="cb-chips">
          <span class="cb-chip" onclick="cbQ('Give me an executive summary')">Executive Summary</span>
          <span class="cb-chip" onclick="cbQ('What are the key risks?')">Key Risks</span>
          <span class="cb-chip" onclick="cbQ('How is revenue trending?')">Revenue Trend</span>
          <span class="cb-chip" onclick="cbQ('Analyze the balance sheet')">Balance Sheet</span>
        </div>
      </div>
    </div>
    <div class="cb-inp-row">
      <input class="cb-inp" id="cb-inp" placeholder="Ask about this company..." autocomplete="off"/>
      <button class="cb-send" id="cb-send" onclick="sendCb()">&#8594;</button>
    </div>
  </div>
</div>

<div class="m-bg" id="modal" style="display:none" onclick="closeModal(event)">
  <div class="m-box" onclick="event.stopPropagation()">
    <div class="m-hdr"><h3 id="m-title">Detail</h3><button class="m-x" onclick="closeModal()">&#x2715;</button></div>
    <div class="m-body" id="m-body"></div>
  </div>
</div>

<div class="fc-popup" id="fc-popup">
  <div class="fc-title">Verify with</div>
  <a href="#" onclick="return fcGo('chatgpt')"><span class="fc-ico">&#x1F4AC;</span>ChatGPT</a>
  <a href="#" onclick="return fcGo('claude')"><span class="fc-ico">&#x1F9E0;</span>Claude</a>
  <a href="#" onclick="return fcGo('perplexity')"><span class="fc-ico">&#x1F50D;</span>Perplexity</a>
  <a href="#" onclick="return fcGo('google')"><span class="fc-ico">&#x1F310;</span>Google Search</a>
</div>
<div class="toast" id="toast">Copied to clipboard!</div>

<script>
const CH=document.getElementById('chat'),I=document.getElementById('inp'),BT=document.getElementById('btn'),AP=document.getElementById('apanel'),APC=document.getElementById('ap-content');
let _tk=null,_filings=[],_company=null,_avail=[],_curAcc=null,_curData=null;
let _vm='metric_rows',_ff=null,_poll=null,_ci={},_lt=null,_fcPrompt='',_histOpen=true;
let _bgSections={};  /* cached filing text sections: {mda:'...', risk_factors:'...', ...} */
let _execSummaries=null; /* cached exec summaries from Claude */
let _chatHistory=[];  /* conversation history for context-aware follow-ups: [{role,content},...] */

/* ═══ Browser-side data cache ═══ */
/* Keyed by "TICKER|accession" — stores {data, summary} so switching periods is instant */
const _dataCache={};
function cacheKey(tk,acc){return(tk||'')+'|'+(acc||'')}
function cacheGet(tk,acc){return _dataCache[cacheKey(tk,acc)]||null}
function cachePut(tk,acc,data,summary){_dataCache[cacheKey(tk,acc)]={data,summary,ts:Date.now()}}
I.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();send()}});
function q(t){I.value=t;send()}

/* ═══ Search bar autocomplete ═══ */
let _searchTimeout=null;
function debSearch(val){
  clearTimeout(_searchTimeout);
  const drop=document.getElementById('sr-drop');
  if(!val||val.length<1){if(drop)drop.style.display='none';return}
  _searchTimeout=setTimeout(async()=>{
    try{
      const r=await fetch('/api/search?q='+encodeURIComponent(val));
      const j=await r.json();
      if(!j.results||!j.results.length){if(drop)drop.style.display='none';return}
      let h='';
      for(const x of j.results){
        h+='<div class="sr-item" onclick="pickAsset(\''+esc(x.ticker)+'\')"><span class="sr-tk">'+esc(x.ticker)+'</span>';
        if(x.exchange)h+='<span class="sr-cik">'+esc(x.exchange)+'</span>';
        else h+='<span class="sr-cik">CIK '+esc(x.cik)+'</span>';
        h+='<span class="sr-nm">'+esc(x.name)+'</span></div>';
      }
      if(drop){drop.innerHTML=h;drop.style.display='block'}
    }catch(e){if(drop)drop.style.display='none'}
  },250);
}
function pickAsset(tk){
  const drop=document.getElementById('sr-drop');if(drop)drop.style.display='none';
  const si=document.getElementById('asrch');if(si)si.value='';
  _tk=tk.toUpperCase();_execSummaries=null;_bgSections={};_chatHistory=[];
  I.value=tk;send();
}
document.addEventListener('click',e=>{
  const drop=document.getElementById('sr-drop');
  if(drop&&!e.target.closest('.srch'))drop.style.display='none';
  const fp=document.getElementById('fc-popup');
  if(fp&&fp.style.display==='block'&&!e.target.closest('.fc-popup')&&!e.target.closest('.fc-btn'))fp.style.display='none';
});

/* ═══ Follow-up detection ═══ */
/* If data is loaded and the message is a question about it (not a new company query), use context-aware chat */
function isFollowUp(msg){
  if(!_curData)return false;
  const low=msg.toLowerCase().trim();
  /* Check for new tickers: 2-5 uppercase letters that aren't common words or finance acronyms.
     Also exclude the currently loaded ticker (asking about it IS a follow-up). */
  const commonWords='THE|FOR|AND|BUT|NOT|ARE|WAS|HAS|HAD|ITS|ALL|CAN|DID|GET|HAS|HER|HIM|HIS|HOW|LET|MAY|NEW|NOW|OLD|OUR|OWN|SAY|SHE|TOO|USE|HER|WAY|WHO|BOY|DAD|MOM';
  const financeAcronyms='CEO|CFO|COO|CTO|ROE|ROA|EPS|IPO|USA|SEC|MDA|DCF|YOY|QOQ|GDP|FCF|ETF|SGA|OCF|TTM|YTD|NAV|AUM|IRR|NPV|PE|PB';
  const excludeRe=new RegExp('\\b('+commonWords+'|'+financeAcronyms+'|'+(_tk||'ZZZZZ')+')\\b');
  const tickerCandidates=msg.match(/\b[A-Z]{2,5}\b/g)||[];
  const realTickers=tickerCandidates.filter(t=>!excludeRe.test(t));
  const hasNewTicker=realTickers.length>0;
  const hasCompanyKeyword=/\b(compare|load|show me|pull up|get|fetch|switch to|look up)\b.*\b[A-Z]{2,5}\b/i.test(msg);
  if(hasCompanyKeyword)return false;
  /* If the message starts with a question word or analysis keyword, it's a follow-up */
  const followUpStarters=['what','why','how','is','are','does','do','can','should','tell','explain',
    'analyze','summarize','break down','walk me through','give me','revenue','margin','profit',
    'cash flow','balance sheet','income','debt','assets','growth','risk','trend','outlook',
    'compare to','year over year','quarter','dividend','eps','roe','valuation','healthy',
    'concern','strength','weakness','insight','opinion','interpret','executive summary'];
  for(const s of followUpStarters){if(low.startsWith(s)||low.includes(s))return !hasNewTicker}
  /* Short questions without tickers are likely follow-ups */
  if(low.length<80&&!hasNewTicker&&(low.includes('?')||low.startsWith('is ')||low.startsWith('are ')))return true;
  return false;
}

/* ═══ Chat send ═══ */
async function send(){
  const m=I.value.trim();if(!m)return;
  const w=document.getElementById('welcome');if(w)w.remove();
  addU(m);I.value='';BT.disabled=true;

  /* Smart routing: if data is loaded and this is a follow-up question, use context-aware AI */
  if(isFollowUp(m)){
    const lid=addLd();
    /* Show thinking indicator */
    const thinkEl=document.createElement('div');thinkEl.className='a-row';
    thinkEl.innerHTML='<div class="a-av think-av">\u2699</div><div class="a-body"><div class="think-card">'
      +'<div class="think-hdr"><span class="think-tool">AI Q&A</span></div>'
      +'<div class="think-steps"><div class="think-step">\u2192 Using loaded '+esc(_tk||'')+' data as context</div>'
      +'<div class="think-step">\u2192 Asking Claude for analysis</div></div></div></div>';
    CH.appendChild(thinkEl);scr();
    try{
      const ctx=Object.assign({},_curData||{});
      if(_bgSections&&Object.keys(_bgSections).length)ctx._filing_sections=_bgSections;
      const body={message:m,ticker:_tk||'',context:ctx,history:_chatHistory.slice(-6)};
      const r=await fetch('/api/chatbot',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body)});
      const j=await r.json();rm(lid);
      let answer=j.answer||'No response.';
      if(j.citations&&j.citations.length){
        answer+='\n\n_Source: '+j.citations.map(c=>c.source).join(', ')+'_';
      }
      addA(answer);
      /* Track conversation history for multi-turn context */
      _chatHistory.push({role:'user',content:m});
      _chatHistory.push({role:'assistant',content:answer});
      if(_chatHistory.length>20)_chatHistory=_chatHistory.slice(-12);
    }catch(e){rm(lid);addA('Error: '+e.message,1)}
    BT.disabled=false;I.focus();return;
  }

  /* Standard routing: new company query — fetch fresh data */
  const lid=addLd();showLd();
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});
    const j=await r.json();rm(lid);
    /* Show thinking/routing first */
    if(j.intent_reasoning&&j.intent_reasoning.length){showThinking(j)}
    if(j.type==='error'){stopLd();addA(j.message,1)}
    else if(j.type==='info'){stopLd();addA(j.message)}
    else if(j.type==='result')handleRes(j);
  }catch(e){rm(lid);stopLd();addA(e.message,1)}
  BT.disabled=false;I.focus();
}

/* ═══ Thinking/tool routing display ═══ */
function showThinking(j){
  const tool=j.intent_tool||'?';
  const reasons=j.intent_reasoning||[];
  const elapsed=j.elapsed_ms?(' \u00b7 '+j.elapsed_ms+'ms'):'';
  const d=document.createElement('div');d.className='a-row';
  let inner='<div class="a-av think-av">\u2699</div><div class="a-body"><div class="think-card">';
  inner+='<div class="think-hdr"><span class="think-tool">'+esc(tool.toUpperCase())+'</span>';
  inner+='<span class="think-time">'+esc(elapsed)+'</span></div>';
  inner+='<div class="think-steps">';
  for(const r of reasons)inner+='<div class="think-step">\u2192 '+esc(r)+'</div>';
  inner+='</div></div></div>';
  d.innerHTML=inner;CH.appendChild(d);scr();
}
function addU(t){const d=document.createElement('div');d.className='u-row';
  d.innerHTML='<div class="u-bub">'+esc(t)+'</div>';CH.appendChild(d);scr()}
function addA(t,err){const d=mkA();const c=d.querySelector('.a-card');
  if(err)c.style.color='var(--red)';c.innerHTML=md(t);CH.appendChild(d);scr()}
function addLd(){const d=mkA();const c=d.querySelector('.a-card');
  c.innerHTML='<div class="ld-wrap"><div class="ld"></div><div class="ld"></div><div class="ld"></div></div>';
  const id='l'+Date.now();d.id=id;CH.appendChild(d);scr();return id}
function rm(id){const e=document.getElementById(id);if(e)e.remove()}
function scr(){CH.scrollTop=CH.scrollHeight}
function esc(s){if(s==null)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML}
function mkA(){const r=document.createElement('div');r.className='a-row';
  r.innerHTML='<div class="a-av">S</div><div class="a-body"><div class="a-card"></div></div>';return r}
function md(s){if(!s)return'';
  let h=esc(s);
  h=h.replace(/### (.+)/g,'<strong style="display:block;margin:10px 0 4px;font-size:13px;color:var(--t1)">$1</strong>');
  h=h.replace(/## (.+)/g,'<strong style="display:block;margin:12px 0 6px;font-size:14px;color:var(--t1)">$1</strong>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  h=h.replace(/\*(.+?)\*/g,'<em style="color:var(--t3)">$1</em>');
  h=h.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  h=h.replace(/^- /gm,'<li>').replace(/\n- /g,'</li><li>');
  if(h.includes('<li>'))h='<ul style="margin:6px 0;padding-left:18px">'+h+'</ul>';
  h=h.replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
  return h}

/* ═══ Handle result ═══ */
function handleRes(j){
  const tk=(j.resolved_tickers||[])[0]||'';
  if(j.tool==='financials'||j.tool==='explain'){
    const d=j.data||{};const name=d.company_name||tk||'Unknown';
    const m=d.metrics||{};const fi=d.filing_info||{};
    /* Build a rich summary message for the chat */
    let msg='**'+name+'** loaded\n\n';
    if(fi.form_type)msg+='- Filing: **'+fi.form_type+'** filed '+fi.filing_date+'\n';
    if(m.revenue!=null)msg+='- Revenue: **'+fmtN(m.revenue)+'**\n';
    if(m.net_income!=null)msg+='- Net Income: **'+fmtN(m.net_income)+'**\n';
    if(m.operating_cash_flow!=null)msg+='- Op. Cash Flow: **'+fmtN(m.operating_cash_flow)+'**\n';
    if(m.total_assets!=null)msg+='- Total Assets: **'+fmtN(m.total_assets)+'**\n';
    msg+='\n_Ask me anything about this data — margins, trends, risks, etc._';
    addA(msg);
    _execSummaries=null;_chatHistory=[];
    /* Cache this data for instant switching later */
    if(j.data&&j.data.filing_info){cachePut(tk||_tk,j.data.filing_info.accession_number,j.data,j.narrative||j.summary||'')}
    render(j.data,j.narrative||j.summary||'');
    if(tk){fetchAvail(tk);bgLoadSections(tk);}
  }else if(j.tool==='compare'){
    if(j.error){addA(j.error,1);return}
    const res=j.results||[];
    const valid=res.filter(r=>r&&r.data&&!r.data.error);
    if(valid.length===0){addA('Could not load comparison data for '+(j.resolved_tickers||[]).join(', '),1);return}
    addA('**Comparison** loaded: '+valid.map(r=>r.data.company_name||r.data.ticker_or_cik).join(' vs '));
    renderCompare(j);if((j.resolved_tickers||[])[0])fetchAvail((j.resolved_tickers||[])[0])
  }else if(j.tool==='filing_text'){
    if(j.error)addA(j.error,1);
    else{addA('**'+esc(j.ticker||'')+'** filing section loaded \u2192 analysis panel');renderFiling(j)}
  }else if(j.tool==='historical'){
    addA(j.message||'Historical extraction started.');if(tk){_tk=tk;fetchHist(tk)}
  }else if(j.tool==='entity'){
    if(j.error)addA(j.error,1);
    else{renderEntity(j.profile);addA('**'+esc((j.profile||{}).name||tk)+'** entity profile loaded')}
  }else if(j.tool==='qa'){
    if(j.error)addA(j.error,1);
    else{renderQA(j);if(j.data){render(j.data,'')}}
  }else addA(JSON.stringify(j,null,2));
}

/* ═══ Loading ═══ */
function showLd(){
  if(_lt)clearInterval(_lt);const s=Date.now();
  APC.innerHTML='<div class="ap-ld"><div class="sp"></div><div class="tm" id="ltm">0.0s</div>'
    +'<div class="sub">Fetching from SEC EDGAR\u2026</div></div>';
  _lt=setInterval(()=>{const e=document.getElementById('ltm');if(e)e.textContent=((Date.now()-s)/1000).toFixed(1)+'s'},100);
}
function stopLd(){if(_lt){clearInterval(_lt);_lt=null}}

/* ═══ Fetch available filings for dropdowns ═══ */
async function fetchAvail(tk){
  try{
    const r=await fetch('/api/filings/'+encodeURIComponent(tk));
    const j=await r.json();_avail=j.filings||[];populatePeriodDropdown()
  }catch(e){_avail=[]}
}

function populatePeriodDropdown(){
  const sel=document.getElementById('sel-period');if(!sel)return;
  const formSel=document.getElementById('sel-form');
  const formFilter=formSel?formSel.value:'';
  let filtered=_avail;
  if(formFilter){
    const altMap={'10-K':['10-K','20-F'],'10-Q':['10-Q','6-K'],'20-F':['20-F','10-K'],'6-K':['6-K','10-Q']};
    const alts=altMap[formFilter]||[formFilter];
    filtered=_avail.filter(f=>alts.includes(f.form_type));
  }
  let h='<option value="">Select Period</option>';
  for(const f of filtered){
    const sel2=_curAcc===f.accession?' selected':'';
    const label=f.form_type+' \u00b7 '+shortDate(f.filing_date);
    h+='<option value="'+esc(f.accession)+'|'+esc(f.form_type)+'"'+sel2+'>'+label+'</option>';
  }
  sel.innerHTML=h;
}
function onFormSel(){populatePeriodDropdown()}
function onPeriodSel(){
  const sel=document.getElementById('sel-period');if(!sel||!sel.value)return;
  const parts=sel.value.split('|');if(parts.length===2)loadFiling(parts[0],parts[1]);
}

async function loadFiling(acc,ft){
  if(!_tk)return;
  _curAcc=acc;_execSummaries=null;
  populatePeriodDropdown();

  /* Check browser cache first — instant switch for previously loaded periods */
  const cached=cacheGet(_tk,acc);
  if(cached){
    addA('Loaded **'+ft+'** filed '+(cached.data?.filing_info?.filing_date||'')+' *(cached)*');
    render(cached.data,cached.summary||'');
    return;
  }

  /* Light loading overlay — preserves toolbar instead of destroying entire panel */
  const overlay=document.createElement('div');
  overlay.id='filing-overlay';
  overlay.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;background:rgba(17,17,21,0.85);border-radius:12px"><div class="sp"></div><span style="color:var(--t2);margin-left:12px">Loading '+esc(ft)+'...</span></div>';
  overlay.style.cssText='position:absolute;inset:0;z-index:100;pointer-events:all';
  APC.style.position='relative';APC.appendChild(overlay);
  try{
    const r=await fetch('/api/load-filing',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker:_tk,accession:acc,form_type:ft})});
    const j=await r.json();
    const ov=document.getElementById('filing-overlay');if(ov)ov.remove();
    /* Cache for instant switching later */
    if(j.data)cachePut(_tk,acc,j.data,j.summary||'');
    addA('Loaded **'+ft+'** filed '+(j.data?.filing_info?.filing_date||''));
    render(j.data,j.summary||'');
  }catch(e){
    const ov=document.getElementById('filing-overlay');if(ov)ov.remove();
    addA('Error loading filing: '+e.message,1);
  }
}

/* ═══ Fact-check feature ═══ */
function factCheck(metric,value,e){
  if(!_curData)return;
  const fi=_curData.filing_info||{};
  const prompt='Verify this SEC filing data for '+(_tk||'?')+': '+metric+' = '+value
    +', from '+(fi.form_type||'10-K')+' filed '+(fi.filing_date||'?')
    +', accession '+(fi.accession_number||'?')+'. Cross-reference the actual EDGAR filing.';
  _fcPrompt=prompt;
  navigator.clipboard.writeText(prompt).then(()=>showToast('Copied to clipboard!'));
  const popup=document.getElementById('fc-popup');
  if(popup){
    const rect=e.target.getBoundingClientRect();
    popup.style.left=Math.min(rect.left,window.innerWidth-180)+'px';
    popup.style.top=(rect.bottom+4)+'px';
    popup.style.display='block';
  }
  if(e)e.stopPropagation();
}
function factCheckChart(chartLabel,e){
  if(!_curData)return;
  const m=_curData.metrics||{},r=_curData.ratios||{},fi=_curData.filing_info||{};
  let data='';
  if(chartLabel==='revenue'){
    data='Revenue='+fmtN(m.revenue)+', Gross Profit='+fmtN(m.gross_profit)+', Op Income='+fmtN(m.operating_income)+', Net Income='+fmtN(m.net_income);
  }else{
    data='Gross Margin='+(r.gross_margin!=null?(r.gross_margin*100).toFixed(1)+'%':'?')
      +', Op Margin='+(r.operating_margin!=null?(r.operating_margin*100).toFixed(1)+'%':'?')
      +', Net Margin='+(r.net_margin!=null?(r.net_margin*100).toFixed(1)+'%':'?');
  }
  const prompt='Verify SEC filing data for '+(_tk||'?')+': '+data
    +'. Source: '+(fi.form_type||'10-K')+' filed '+(fi.filing_date||'?')
    +', accession '+(fi.accession_number||'?')+'. Cross-reference EDGAR.';
  _fcPrompt=prompt;
  navigator.clipboard.writeText(prompt).then(()=>showToast('Copied to clipboard!'));
  const popup=document.getElementById('fc-popup');
  if(popup){
    const rect=e.target.getBoundingClientRect();
    popup.style.left=Math.min(rect.left,window.innerWidth-180)+'px';
    popup.style.top=(rect.bottom+4)+'px';
    popup.style.display='block';
  }
  if(e)e.stopPropagation();
}
function fcGo(target){
  const urls={chatgpt:'https://chat.openai.com/',claude:'https://claude.ai/new',
    perplexity:'https://www.perplexity.ai/',google:'https://www.google.com/search?q='+encodeURIComponent(_fcPrompt)};
  window.open(urls[target]||'#','_blank');
  document.getElementById('fc-popup').style.display='none';
  return false;
}
function showToast(msg){
  const t=document.getElementById('toast');if(!t)return;t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2000);
}

/* ═══ Main Renderer ═══ */
function render(d,narr){
  stopLd();if(!d){APC.innerHTML='<div class="ap-empty"><div><p>No data</p></div></div>';return}
  Object.values(_ci).forEach(c=>{try{c.destroy()}catch(e){}});_ci={};
  _tk=(d.ticker_or_cik||'').toUpperCase();
  /* Always reset exec summaries when rendering new data (new asset or new period) */
  _execSummaries=null;
  _curData=d;
  if(d.filing_info)_curAcc=d.filing_info.accession_number||null;
  const m=d.metrics||{},r=d.ratios||{},fi=d.filing_info||{},lk=d.sec_links||{};
  let h='';

  /* Reset the View dropdown to dashboard when rendering financial data */
  const toolSel=document.getElementById('sel-tool');
  if(toolSel)toolSel.value='dashboard';

  /* ── Two-column split ── */
  h+='<div class="split">';

  /* ── Left column (40%) ── */
  h+='<div class="col-l" id="col-l">';

  /* Company header */
  h+='<div class="dh"><span class="dh-tk">'+esc(_tk)+'</span>';
  h+='<span class="dh-name">'+esc(d.company_name||_tk)+'</span>';
  if(d.industry_class)h+='<span class="dh-tag">'+esc(d.industry_class.toUpperCase())+'</span>';
  h+='<span class="dh-tag">'+(fi.form_type||'10-K')+' \u00b7 Filed '+(fi.filing_date||d.fiscal_year||'latest')+'</span>';
  if(d.period_type==='quarterly')h+='<span class="dh-tag" style="color:var(--acc)">Q Only</span>';
  if(lk.filing_index)h+='<a href="'+lk.filing_index+'" target="_blank" class="dh-link">EDGAR \u2197</a>';
  if(d.ir_link)h+='<a href="'+esc(d.ir_link)+'" target="_blank" class="dh-link">IR Page \u2197</a>';
  h+='</div>';

  /* Key metrics grid (2x4) */
  const kms=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'gross_profit',l:'Gross Profit'},
    {k:'operating_income',l:'Op. Income'},{k:'ebitda',l:'EBITDA'},{k:'total_assets',l:'Total Assets'},
    {k:'free_cash_flow',l:'Free Cash Flow'},{k:'operating_cash_flow',l:'Op. Cash Flow'}];
  h+='<div class="km-grid">';
  for(const km of kms){
    const v=m[km.k];if(v==null)continue;const neg=v<0;
    h+='<div class="km"><div class="km-label">'+km.l;
    h+='<button class="fc-btn" onclick="factCheck(\''+km.l+'\',\''+fmtN(v)+'\',event)" title="Verify">&#x2713;</button>';
    h+='</div>';
    h+='<div class="km-val'+(neg?' neg':'')+'">'+fmtN(v)+'</div>';
    if(km.k==='net_income'&&r.net_margin!=null)h+='<div class="km-sub">Margin '+(r.net_margin*100).toFixed(1)+'%</div>';
    if(km.k==='gross_profit'&&r.gross_margin!=null)h+='<div class="km-sub">'+(r.gross_margin*100).toFixed(1)+'% margin</div>';
    if(km.k==='total_assets'&&r.roe!=null)h+='<div class="km-sub">ROE '+(r.roe*100).toFixed(1)+'%</div>';
    h+='</div>';
  }
  h+='</div>';

  /* Charts — FIXED HEIGHT */
  h+='<div class="ch-area">';
  h+='<div class="ch-box"><div class="ch-hdr"><h4>Revenue ($B)</h4>';
  h+='<button class="fc-btn" onclick="factCheckChart(\'revenue\',event)" title="Verify chart">&#x2713;</button></div>';
  h+='<canvas id="c-rev"></canvas></div>';
  h+='<div class="ch-box"><div class="ch-hdr"><h4>Margins (%)</h4>';
  h+='<button class="fc-btn" onclick="factCheckChart(\'margins\',event)" title="Verify chart">&#x2713;</button></div>';
  h+='<canvas id="c-mar"></canvas></div>';
  h+='</div>';

  /* Narrative */
  if(narr)h+='<div class="narr-box"><h3>Analysis</h3><p>'+md(narr)+'</p></div>';

  /* Inline exec summary (auto-generated) */
  h+='<div class="exec-inline" id="exec-inline">';
  h+='<div class="exec-inline-hdr"><h3>Executive Summary</h3>';
  h+='<span class="exec-badge" id="exec-badge">Generating...</span></div>';
  h+='<div class="exec-inline-body" id="exec-inline-body">';
  h+='<div class="exec-inline-loading"><div class="ld-wrap"><div class="ld"></div><div class="ld"></div><div class="ld"></div></div>';
  h+='<span style="color:var(--t3);font-size:13px;margin-left:8px">Claude is analyzing the data...</span></div>';
  h+='</div></div>';

  /* Validation */
  const val=d.validation||[];
  if(val.length){h+='<div style="padding:8px 16px">';
    for(const w of val){const c=w.severity==='error'?'err':'wrn';
      h+='<div class="vw '+c+'">'+(c==='err'?'\u26D4':'\u26A0')+' '+esc(w.message)+'</div>';}h+='</div>';}

  h+='</div>'; /* end col-l */

  /* ── Right column (60%) — statements with Overview tab ── */
  h+='<div class="col-r" id="col-r">';
  const ss=[{k:'income_statement',l:'Income'},{k:'balance_sheet',l:'Balance Sheet'},{k:'cash_flow_statement',l:'Cash Flow'}];
  const hasStmts=ss.some(s=>(d[s.k]||[]).length);
  const hasMetrics=Object.values(m).some(v=>v!=null);

  if(hasStmts||hasMetrics){
    /* Main sub-tabs: Full Data | Exec Summaries */
    h+='<div class="main-tabs" id="main-tabs">';
    h+='<div class="main-tab act" onclick="swMainTab(this,\'mt-data\')">Full Data</div>';
    h+='<div class="main-tab" onclick="swMainTab(this,\'mt-exec\')">Exec Summaries</div>';
    h+='</div>';

    /* Full Data pane with statement sub-tabs */
    h+='<div class="main-pane" id="mt-data">';
    h+='<div class="st-tabs" id="st-tabs">';
    h+='<div class="st-tab act" onclick="swTab(this,\'sw-ov\')">Overview</div>';
    ss.forEach((s,i)=>h+='<div class="st-tab" onclick="swTab(this,\'sw'+i+'\')">'+s.l+'</div>');
    h+='</div>';
    h+='<div class="st-wrap" id="sw-ov">'+bOverview(d)+'</div>';
    ss.forEach((s,i)=>h+='<div class="st-wrap" id="sw'+i+'" style="display:none">'+bStmt(d[s.k]||[],s.l)+'</div>');
    h+='</div>';

    /* Exec Summaries pane (populated async via Claude) */
    h+='<div class="main-pane" id="mt-exec" style="display:none">';
    h+='<div class="exec-loading" id="exec-loading"><div class="sp"></div><span>Generating executive summaries with Claude\u2026</span></div>';
    h+='<div id="exec-content"></div>';
    h+='</div>';
  }else{
    h+='<div style="flex:1;display:flex;align-items:center;justify-content:center;color:var(--t4);font-size:13px">';
    h+='No financial statement data available</div>';
  }
  h+='</div>'; /* end col-r */
  h+='</div>'; /* end split */

  /* ── Historical section (full-width, collapsible) ── */
  h+='<div class="hist-sec" id="hist-sec">';
  h+='<div class="hist-toggle'+((_histOpen)?' open':'')+'" id="hist-toggle" onclick="toggleHist()">';
  h+='<span class="arr">&#x25B6;</span><span class="lab">Historical Data</span>';
  h+='<span class="fcnt" id="fcnt" style="margin-left:auto"></span></div>';
  h+='<div class="hist-body'+((_histOpen)?' open':'')+'" id="hist-body">';
  h+='<div class="fbar">';
  h+='<button class="fbtn act" onclick="setFf(null,this)">All</button>';
  h+='<button class="fbtn" onclick="setFf(\'10-K\',this)">10-K/20-F</button>';
  h+='<button class="fbtn" onclick="setFf(\'10-Q\',this)">10-Q/6-K</button>';
  h+='<button class="fbtn" onclick="setFf(\'8-K\',this)">8-K</button>';
  h+='<div class="fsep"></div>';
  h+='<button class="fbtn act" onclick="setVm(\'metric_rows\',this)">Metrics\u00d7Time</button>';
  h+='<button class="fbtn" onclick="setVm(\'period_rows\',this)">Time\u00d7Metrics</button>';
  h+='<div class="fright">';
  h+='<button class="fref" onclick="trigRef()">\u21BB Refresh</button>';
  h+='<button class="fref" onclick="popTbl()">\u2922 Expand</button></div></div>';
  h+='<div class="prog" id="prog"><div class="prog-fill" id="pf" style="width:0"></div></div>';
  h+='<div class="tbl-area" id="tbl"></div><div id="extra"></div>';
  h+='</div></div>'; /* end hist-body, hist-sec */

  APC.innerHTML=h;_ff=null;_vm='metric_rows';
  setTimeout(()=>{rCharts(m,r);populatePeriodDropdown()},50);
  fetchHist(_tk);
  /* Auto-generate exec summaries in background after a short delay */
  if(!_execSummaries&&_curData){setTimeout(()=>genExecSummaries(),500)}
}

function toggleHist(){
  const toggle=document.getElementById('hist-toggle');
  const body=document.getElementById('hist-body');
  if(!toggle||!body)return;
  _histOpen=!_histOpen;
  toggle.classList.toggle('open',_histOpen);
  body.classList.toggle('open',_histOpen);
}

/* ═══ Charts (fixed container) ═══ */
function rCharts(m,r){
  const rc=document.getElementById('c-rev');
  if(rc){
    const vals=[m.revenue,m.gross_profit,m.operating_income,m.net_income];
    const labs=['Revenue','Gross','Op. Inc','Net Inc'];
    const cols=['rgba(74,122,255,.60)','rgba(59,104,217,.60)','rgba(2,62,138,.60)','rgba(99,179,237,.60)'];
    const bcols=['rgb(0,180,216)','rgb(0,119,182)','rgb(2,62,138)','rgb(0,245,196)'];
    const fv=[],fl=[],fc=[],fb=[];
    vals.forEach((v,i)=>{if(v!=null){fv.push(v/1e9);fl.push(labs[i]);fc.push(cols[i]);fb.push(bcols[i])}});
    if(fv.length)_ci.rev=new Chart(rc,{type:'bar',
      data:{labels:fl,datasets:[{data:fv,backgroundColor:fc,borderColor:fb,borderWidth:1,borderRadius:4}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
        scales:{y:{ticks:{callback:v=>'$'+v.toFixed(0)+'B',color:'#4d6478',font:{family:'JetBrains Mono',size:9}},grid:{color:'rgba(74,122,255,.03)'}},
          x:{ticks:{color:'#4d6478',font:{family:'Inter',size:9}},grid:{display:false}}}}});
  }
  const mc=document.getElementById('c-mar');
  if(mc){
    const ml=[],mdata=[],mco=[];
    if(r.gross_margin!=null){ml.push('Gross');mdata.push((r.gross_margin*100).toFixed(1));mco.push('rgba(74,122,255,.65)')}
    if(r.operating_margin!=null){ml.push('Op.');mdata.push((r.operating_margin*100).toFixed(1));mco.push('rgba(59,104,217,.65)')}
    if(r.net_margin!=null){ml.push('Net');mdata.push((r.net_margin*100).toFixed(1));mco.push('rgba(99,179,237,.65)')}
    if(r.ebitda_margin!=null){ml.push('EBITDA');mdata.push((r.ebitda_margin*100).toFixed(1));mco.push('rgba(255,166,43,.65)')}
    if(ml.length)_ci.mar=new Chart(mc,{type:'bar',
      data:{labels:ml,datasets:[{data:mdata,backgroundColor:mco,borderRadius:4,borderSkipped:false}]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
        scales:{x:{ticks:{callback:v=>v+'%',color:'#4d6478',font:{family:'JetBrains Mono',size:9}},grid:{color:'rgba(74,122,255,.03)'}},
          y:{ticks:{color:'#8298b4',font:{family:'Inter',size:10}},grid:{display:false}}}}});
  }
}

/* ═══ Filing text view (MD&A, Risk Factors, etc.) ═══ */
function renderFiling(j){
  stopLd();
  const sec=j.section||'full filing';
  const secLabel={'mda':'MD&A (Management Discussion & Analysis)','risk_factors':'Risk Factors',
    'business':'Business Overview','financial_statements':'Financial Statements',
    'legal':'Legal Proceedings','controls':'Controls & Procedures',
    'executive_compensation':'Executive Compensation','full filing':'Full Filing'}[sec]||sec;
  const txt=j.text||'';
  const hasContent=txt.length>200;
  let h='<div class="filing-view">';

  /* Header with ticker + section label + filing info */
  h+='<div class="filing-hdr">';
  h+='<span class="dh-tk">'+esc(j.ticker||_tk||'')+'</span>';
  h+='<span class="filing-sec">'+esc(secLabel)+'</span>';
  h+='<span class="dh-tag">'+esc(j.form_type||'10-K')+' \u00b7 Filed '+esc(j.filing_date||'')+'</span>';
  if(j.text_length)h+='<span class="dh-tag">'+Math.round(j.text_length/1000)+'K chars</span>';
  h+='</div>';

  /* Summary card — shows key metrics from currently loaded data for context */
  if(_curData&&_curData.metrics){
    const m=_curData.metrics,r=_curData.ratios||{},fi=_curData.filing_info||{};
    h+='<div class="filing-summary">';
    h+='<div class="filing-summary-hdr">Financial Context <span class="dh-tag">'+esc(fi.form_type||'')+' '+esc(fi.filing_date||'')+'</span></div>';
    h+='<div class="filing-summary-grid">';
    const kms=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'operating_income',l:'Op. Income'},
      {k:'total_assets',l:'Assets'},{k:'free_cash_flow',l:'FCF'},{k:'operating_cash_flow',l:'Op. Cash Flow'}];
    for(const km of kms){
      const v=m[km.k];if(v==null)continue;
      h+='<div class="filing-summary-item"><span class="filing-summary-label">'+km.l+'</span>';
      h+='<span class="filing-summary-val'+(v<0?' neg':'')+'">'+fmtN(v)+'</span></div>';
    }
    /* Add key ratios */
    if(r.net_margin!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Net Margin</span><span class="filing-summary-val">'+(r.net_margin*100).toFixed(1)+'%</span></div>';
    if(r.current_ratio!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Current Ratio</span><span class="filing-summary-val">'+r.current_ratio.toFixed(2)+'x</span></div>';
    h+='</div></div>';
  }

  if(!hasContent){
    h+='<div class="filing-empty"><div class="filing-empty-ico">\u26A0</div>';
    h+='<h3>Section Not Available</h3>';
    h+='<p>Could not extract <strong>'+esc(secLabel)+'</strong> from this filing.</p>';
    h+='<p>This may be because:</p>';
    h+='<ul><li>The section markers were not found in the filing HTML</li>';
    h+='<li>The filing uses non-standard formatting</li>';
    h+='<li>The section does not exist in this filing type</li></ul>';
    h+='<p style="margin-top:12px">Try searching on <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company='+esc(j.ticker||_tk||'')+'&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany" target="_blank" style="color:var(--acc)">EDGAR</a> directly.</p>';
    h+='</div>';
  }else{
    h+='<div class="filing-body">'+formatFilingText(txt)+'</div>';
  }
  h+='</div>';
  APC.innerHTML=h;
}

function formatFilingText(txt){
  let lines=esc(txt).split('\\n');
  if(lines.length<3)lines=esc(txt).split('\n');
  let out='';
  for(const line of lines){
    const trimmed=line.trim();
    if(!trimmed){out+='<br>';continue}
    if(trimmed.match(/^(Item\s+\d|ITEM\s+\d|Part\s+[IVX])/i)){
      out+='<h3 class="filing-heading">'+trimmed+'</h3>';
    }else if(trimmed.length<80&&trimmed===trimmed.toUpperCase()&&trimmed.length>3){
      out+='<h4 class="filing-subhead">'+trimmed+'</h4>';
    }else{
      out+='<p class="filing-para">'+trimmed+'</p>';
    }
  }
  return out;
}

/* ═══ Entity Profile view ═══ */
function renderEntity(p){
  if(!p)return;
  stopLd();
  /* Don't override _tk — the ticker is already locked from the initial load.
     Only update if _tk is null (first load via entity). */
  if(!_tk)_tk=(p.ticker||'').toUpperCase();
  let h='<div class="entity-view">';
  h+='<div class="entity-hdr">';
  h+='<div class="entity-avatar">'+esc((p.ticker||'?').charAt(0))+'</div>';
  h+='<div class="entity-title">';
  h+='<h2>'+esc(p.name||'')+'</h2>';
  h+='<div class="entity-meta">';
  h+='<span class="dh-tk">'+esc(p.ticker||'')+'</span>';
  if(p.exchange)h+='<span class="dh-tag">'+esc(p.exchange)+'</span>';
  if(p.industry)h+='<span class="dh-tag">'+esc(p.industry)+'</span>';
  h+='</div></div></div>';
  h+='<div class="entity-grid">';
  const fields=[
    {l:'CEO',v:p.ceo,sub:p.ceo_source},
    {l:'CIK',v:p.cik},
    {l:'SIC Code',v:p.sic_code},
    {l:'Website',v:p.website,link:true},
    {l:'Address',v:p.address},
    {l:'Phone',v:p.phone},
    {l:'Fiscal Year End',v:p.fiscal_year_end},
    {l:'State of Incorp.',v:p.state_of_incorporation},
    {l:'EIN',v:p.ein},
  ];
  for(const f of fields){
    if(!f.v)continue;
    h+='<div class="entity-field"><span class="entity-label">'+esc(f.l)+'</span>';
    if(f.link&&f.v){
      const url=f.v.startsWith('http')?f.v:'https://'+f.v;
      h+='<a href="'+esc(url)+'" target="_blank" class="entity-val link">'+esc(f.v)+'</a>';
    }else{
      h+='<span class="entity-val">'+esc(f.v||'\u2014')+'</span>';
    }
    if(f.sub)h+='<span class="entity-sub">'+esc(f.sub)+'</span>';
    h+='</div>';
  }
  h+='</div>';
  /* Financial summary card if data is loaded */
  if(_curData&&_curData.metrics){
    const m=_curData.metrics,r=_curData.ratios||{},fi=_curData.filing_info||{};
    h+='<div class="filing-summary" style="margin-top:16px">';
    h+='<div class="filing-summary-hdr">Loaded Financials <span class="dh-tag">'+esc(fi.form_type||'')+' '+esc(fi.filing_date||'')+'</span></div>';
    h+='<div class="filing-summary-grid">';
    const kms=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'operating_income',l:'Op. Income'},
      {k:'total_assets',l:'Assets'},{k:'free_cash_flow',l:'FCF'},{k:'operating_cash_flow',l:'Op. Cash Flow'}];
    for(const km of kms){
      const v=m[km.k];if(v==null)continue;
      h+='<div class="filing-summary-item"><span class="filing-summary-label">'+km.l+'</span>';
      h+='<span class="filing-summary-val'+(v<0?' neg':'')+'">'+fmtN(v)+'</span></div>';
    }
    if(r.net_margin!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Net Margin</span><span class="filing-summary-val">'+(r.net_margin*100).toFixed(1)+'%</span></div>';
    if(r.current_ratio!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Current Ratio</span><span class="filing-summary-val">'+r.current_ratio.toFixed(2)+'x</span></div>';
    h+='</div></div>';
  }

  if(p.latest_filing){
    const lf=p.latest_filing;
    h+='<div class="entity-filing"><h4>Latest Filing</h4>';
    h+='<span class="dh-tag">'+esc(lf.form_type||'')+' \u00b7 Filed '+esc(lf.filing_date||'')+'</span>';
    if(!_curData){
      h+='<button class="entity-load" onclick="I.value=\''+esc(p.ticker)+'\';send()">Load Financials \u2192</button>';
    }else{
      h+='<button class="entity-load" onclick="document.getElementById(\'sel-tool\').value=\'dashboard\';onToolSel()">View Dashboard \u2192</button>';
    }
    h+='</div>';
  }
  if(!p.ceo){
    h+='<div class="entity-note">\u2139 CEO data is not directly available from SEC filings. ';
    h+='Check the company\'s latest <strong>DEF 14A (Proxy Statement)</strong> or investor relations page.</div>';
  }
  h+='</div>';
  APC.innerHTML=h;
  if(!_avail||!_avail.length)fetchAvail(p.ticker);
}

/* ═══ Compare view ═══ */
let _compareRes=[];
function renderCompare(j){
  stopLd();
  const res=(j.results||[]).filter(r=>r&&r.data&&!r.data.error);
  if(!res.length)return;
  _compareRes=res;
  const tickers=(j.resolved_tickers||[]);
  _tk=tickers[0]||(res[0].data.ticker_or_cik||'').toUpperCase();
  _curData=res[0].data;

  const METRICS=['revenue','net_income','gross_profit','operating_income','ebitda',
    'total_assets','total_liabilities','stockholders_equity','operating_cash_flow',
    'free_cash_flow','capital_expenditures','cash_and_equivalents','long_term_debt'];
  const LABELS={revenue:'Revenue',net_income:'Net Income',gross_profit:'Gross Profit',
    operating_income:'Op. Income',ebitda:'EBITDA',total_assets:'Total Assets',
    total_liabilities:'Total Liab.',stockholders_equity:'Equity',
    operating_cash_flow:'Op. Cash Flow',free_cash_flow:'Free Cash Flow',
    capital_expenditures:'CapEx',cash_and_equivalents:'Cash',
    long_term_debt:'LT Debt'};

  let h='<div class="compare-view">';
  h+='<div class="compare-hdr"><h2>Comparison</h2>';
  h+='<div class="compare-pills">';
  for(let i=0;i<res.length;i++){
    const tk=(res[i].data.ticker_or_cik||'').toUpperCase();
    h+='<button class="compare-pill" onclick="loadCompareIdx('+i+')" title="Load full view">'+esc(tk)+'</button>';
  }
  h+='</div></div>';

  h+='<table class="compare-tbl"><thead><tr><th>Metric</th>';
  for(const r of res)h+='<th>'+esc((r.data.company_name||r.data.ticker_or_cik||'?').substring(0,20))+'</th>';
  h+='</tr></thead><tbody>';
  for(const mk of METRICS){
    const hasAny=res.some(r=>(r.data.metrics||{})[mk]!=null);
    if(!hasAny)continue;
    h+='<tr><td class="lbl">'+esc(LABELS[mk]||mk)+'</td>';
    for(const r of res){
      const v=(r.data.metrics||{})[mk];
      h+='<td>'+(v!=null?fmtN(v):'\u2014')+'</td>';
    }
    h+='</tr>';
  }
  h+='</tbody></table>';

  if(j.comparison_narrative){
    h+='<div class="compare-narr"><h3>AI Comparison</h3><div class="narr-box"><p>'+md(j.comparison_narrative)+'</p></div></div>';
  }
  h+='</div>';
  APC.innerHTML=h;
  fetchAvail(tickers[0]);
}
function loadCompareIdx(i){
  const r=_compareRes[i];
  if(!r)return;
  render(r.data,r.summary||'');
}

/* ═══ Q&A answer view ═══ */
function renderQA(j){
  const answer=j.answer||'No answer available.';
  const cits=j.citations||[];
  let citHtml='';
  if(cits.length){
    citHtml='<div class="qa-cits"><strong>Sources:</strong> ';
    for(const c of cits)citHtml+='<span class="qa-cit">'+esc(c.source||'')+'</span> ';
    citHtml+='</div>';
  }
  addA(md(answer)+citHtml);
}

/* ═══ Historical ═══ */
async function fetchHist(tk){
  try{
    const r=await fetch('/api/historical/'+encodeURIComponent(tk));
    const j=await r.json();_company=j.company;_filings=j.filings||[];
    rTbl();rExtra();
    if(j.job&&j.job.status==='processing'){uProg(j.job.progress,j.job.total);sPoll(tk)}else hProg();
    const fc=document.getElementById('fcnt');if(fc)fc.textContent=_filings.length+' filings';
  }catch(e){const t=document.getElementById('tbl');
    if(t)t.innerHTML='<div style="padding:18px;color:var(--t4);font-size:12px">Historical data unavailable</div>'}
}
function sPoll(tk){if(_poll)clearInterval(_poll);
  _poll=setInterval(async()=>{try{const r=await fetch('/api/historical/'+encodeURIComponent(tk)+'/status');
    const j=await r.json();if(j.status==='processing')uProg(j.progress,j.total);
    else{clearInterval(_poll);_poll=null;hProg();fetchHist(tk)}}catch(e){clearInterval(_poll);_poll=null}},3000)}
function uProg(p,t){const e=document.getElementById('pf');if(e)e.style.width=(t>0?p/t*100:0)+'%'}
function hProg(){const e=document.getElementById('pf');if(e)e.style.width='0'}
async function trigRef(){if(!_tk)return;await fetch('/api/historical/'+_tk+'/fetch',{method:'POST'});sPoll(_tk);addA('Refreshing **'+_tk+'** historical data...')}

function setFf(v,el){_ff=v;el.parentElement.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('act'));el.classList.add('act');rTbl();rExtra()}
function setVm(v,el){_vm=v;el.parentElement.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('act'));el.classList.add('act');rTbl()}

const KM=['revenue','net_income','gross_profit','operating_income','ebitda','total_assets','total_liabilities','stockholders_equity',
  'operating_cash_flow','free_cash_flow','capital_expenditures','eps_basic','eps_diluted','cost_of_revenue','operating_expenses',
  'sga_expense','rd_expense','interest_expense','income_tax_expense','current_assets','current_liabilities','long_term_debt',
  'short_term_debt','depreciation_amortization','shares_outstanding'];
const ML={revenue:'Revenue',net_income:'Net Income',gross_profit:'Gross Profit',operating_income:'Op. Income',ebitda:'EBITDA',
  total_assets:'Total Assets',total_liabilities:'Total Liab.',stockholders_equity:'Equity',operating_cash_flow:'Op. CF',
  free_cash_flow:'Free CF',capital_expenditures:'CapEx',eps_basic:'EPS Basic',eps_diluted:'EPS Diluted',
  cost_of_revenue:'Cost of Rev.',operating_expenses:'OpEx',sga_expense:'SG&A',rd_expense:'R&D',interest_expense:'Interest',
  income_tax_expense:'Income Tax',current_assets:'Cur. Assets',current_liabilities:'Cur. Liab.',long_term_debt:'LT Debt',
  short_term_debt:'ST Debt',depreciation_amortization:'D&A',shares_outstanding:'Shares Out.'};

function gFilt(){let f=_filings;
  if(_ff){
    /* Include FPI equivalents: 10-K includes 20-F, 10-Q includes 6-K */
    const altMap={'10-K':['10-K','20-F'],'10-Q':['10-Q','6-K']};
    const alts=altMap[_ff]||[_ff];
    f=f.filter(x=>alts.includes(x.form_type));
  }
  return f.sort((a,b)=>(b.filing_date||'').localeCompare(a.filing_date||''))}
function rTbl(){
  const a=document.getElementById('tbl');if(!a)return;
  const fl=gFilt().filter(f=>f.form_type!=='8-K');
  if(!fl.length){a.innerHTML='<div style="padding:18px;text-align:center;color:var(--t4);font-size:12px">'+
    (_ff==='8-K'?'8-K events shown in timeline below':'No historical filings cached. Click Refresh.')+'</div>';return}
  a.innerHTML=_vm==='metric_rows'?bMR(fl):bPR(fl);
  const fc=document.getElementById('fcnt');if(fc)fc.textContent=_filings.length+' filings';
}
function bMR(fl){
  let h='<table class="dt"><thead><tr><th>Metric</th>';
  for(const f of fl)h+='<th>'+esc(shortDate(f.filing_date))+(['10-Q','6-K'].includes(f.form_type)?' Q':'')+'</th>';
  h+='</tr></thead><tbody>';
  for(const mk of KM){
    if(!fl.some(f=>{const m=(f.metrics||{})[mk];return m&&m.value!=null}))continue;
    h+='<tr><td>'+esc(ML[mk]||mk)+'</td>';
    for(const f of fl){const m=(f.metrics||{})[mk];
      if(!m||m.value==null){h+='<td style="color:var(--t4)">\u2014</td>';continue}
      const v=m.value,neg=v<0,url=m.source_url||(f.source_urls||{}).filing_index||'';
      h+='<td'+(neg?' class="neg"':'')+'>'+fmtN(v);
      if(url)h+='<a class="src" href="'+url+'" target="_blank" title="EDGAR source">\u2197</a>';
      h+='</td>'}h+='</tr>'}
  return h+'</tbody></table>';
}
function bPR(fl){
  const am=KM.filter(mk=>fl.some(f=>{const m=(f.metrics||{})[mk];return m&&m.value!=null}));
  let h='<table class="dt"><thead><tr><th>Period</th><th>Type</th>';
  for(const mk of am)h+='<th>'+esc(ML[mk]||mk)+'</th>';h+='</tr></thead><tbody>';
  for(const f of fl){h+='<tr><td>'+esc(shortDate(f.filing_date))+'</td><td>'+esc(f.form_type)+'</td>';
    for(const mk of am){const m=(f.metrics||{})[mk];
      if(!m||m.value==null){h+='<td style="color:var(--t4)">\u2014</td>';continue}
      const v=m.value,neg=v<0,url=m.source_url||(f.source_urls||{}).filing_index||'';
      h+='<td'+(neg?' class="neg"':'')+'>'+fmtN(v);
      if(url)h+='<a class="src" href="'+url+'" target="_blank">\u2197</a>';
      h+='</td>'}h+='</tr>'}
  return h+'</tbody></table>';
}

function rExtra(){
  const sec=document.getElementById('extra');if(!sec)return;let h='';
  const ws=_filings.filter(f=>f.summary);
  if(ws.length)h+='<div class="sum-sec"><h3>AI Summary</h3><p>'+esc(ws[0].summary)+'</p></div>';
  const ek=_filings.filter(f=>f.form_type==='8-K');
  if(ek.length&&(_ff===null||_ff==='8-K')){
    h+='<div class="tl"><div class="tl-title"><span class="dot"></span>8-K Events ('+ek.length+')</div>';
    for(const f of ek.slice(0,40)){const u=f.source_urls||{};
      h+='<div class="tl-item"><span class="tl-date">'+esc(shortDate(f.filing_date))+'</span>';
      h+='<div class="tl-body">'+esc(f.description||'8-K');
      if(f.items_reported&&f.items_reported.length)h+='<div class="items">Items: '+f.items_reported.join(', ')+'</div>';
      if(f.summary)h+='<div style="margin-top:2px;font-size:11px;color:var(--t3)">'+esc(f.summary)+'</div>';
      if(u.filing_index)h+=' <a href="'+u.filing_index+'" target="_blank">View \u2197</a>';
      h+='</div></div>'}h+='</div>'}
  sec.innerHTML=h;
}

/* ═══ Overview tab builder ═══ */
function bOverview(d){
  const m=d.metrics||{},r=d.ratios||{},pm=d.prior_metrics||{},conf=d.confidence_scores||{};
  let h='';

  /* Data coverage indicator */
  const reqKeys=['revenue','net_income','total_assets','current_assets','operating_cash_flow',
    'capital_expenditures','free_cash_flow','stockholders_equity','total_liabilities','cash_and_equivalents'];
  const filled=reqKeys.filter(k=>m[k]!=null).length;
  const pct=Math.round(filled/reqKeys.length*100);
  const barColor=pct>=80?'var(--acc)':pct>=50?'var(--amber)':'var(--red)';
  h+='<div style="padding:12px 20px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:10px">';
  h+='<span style="font:600 11px var(--ff);color:var(--t4);text-transform:uppercase;letter-spacing:.3px">Data Coverage</span>';
  h+='<div style="flex:1;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden">';
  h+='<div style="width:'+pct+'%;height:100%;background:'+barColor+';border-radius:2px"></div></div>';
  h+='<span style="font:600 12px var(--ff);color:var(--t2)">'+pct+'%</span>';
  h+='<span style="font:400 11px var(--ff);color:var(--t4)">('+filled+'/'+reqKeys.length+' required fields)</span>';
  h+='</div>';

  function ovRow(label,val,opts){
    opts=opts||{};const isTotal=opts.total;const isNeg=val!=null&&val<0;
    const cls='ov-row'+(isTotal?' total':'');
    let vc='val';if(isNeg)vc+=' neg';else if(val!=null&&val>0&&opts.green)vc+=' pos';
    let valStr='\u2014';
    if(val!=null)valStr=fmtN(val);
    else if(opts.required)valStr='<span style="color:var(--t4);font-size:12px;font-style:italic">Not disclosed</span>';
    h+='<div class="'+cls+'"><span class="lbl">'+esc(label)+'</span>';
    h+='<span class="'+vc+'">'+valStr+'</span></div>';
    /* Show source/confidence for computed values */
    const src=(d.metrics_sourced||{})[opts.key];
    const c=(d.confidence_scores||{})[opts.key];
    if(src&&c!=null&&c<0.9){
      h+='<div style="padding:0 12px;font:400 10px var(--ff);color:var(--t4)">'+esc(src)+' ('+Math.round(c*100)+'% conf.)</div>';
    }
  }

  h+='<div class="ov-sec"><h4><span class="dot g"></span>Income Summary</h4>';
  ovRow('Revenue',m.revenue,{green:1,required:1,key:'revenue'});
  ovRow('Cost of Revenue',m.cost_of_revenue,{key:'cost_of_revenue'});
  ovRow('Gross Profit',m.gross_profit,{green:1,total:1,required:1,key:'gross_profit'});
  if(r.gross_margin!=null)h+='<div style="padding:0 12px;font:500 11px var(--ff);color:var(--t3)">Margin: '+(r.gross_margin*100).toFixed(1)+'%</div>';
  ovRow('Operating Expenses',m.operating_expenses,{key:'operating_expenses'});
  if(m.sga_expense!=null)ovRow('  SG&A',m.sga_expense,{key:'sga_expense'});
  if(m.rd_expense!=null)ovRow('  R&D',m.rd_expense,{key:'rd_expense'});
  ovRow('Operating Income',m.operating_income,{green:1,key:'operating_income'});
  if(r.operating_margin!=null)h+='<div style="padding:0 12px;font:500 11px var(--ff);color:var(--t3)">Margin: '+(r.operating_margin*100).toFixed(1)+'%</div>';
  ovRow('Interest Expense',m.interest_expense,{key:'interest_expense'});
  ovRow('Income Tax',m.income_tax_expense,{key:'income_tax_expense'});
  ovRow('Net Income',m.net_income,{green:1,total:1,required:1,key:'net_income'});
  if(r.net_margin!=null)h+='<div style="padding:0 12px;font:500 11px var(--ff);color:var(--t3)">Margin: '+(r.net_margin*100).toFixed(1)+'%</div>';
  if(m.eps_basic!=null)h+='<div class="ov-row"><span class="lbl">EPS (Basic / Diluted)</span><span class="val">$'+(m.eps_basic||0).toFixed(2)+' / $'+(m.eps_diluted||m.eps_basic||0).toFixed(2)+'</span></div>';
  h+='</div>';

  h+='<div class="ov-sec"><h4><span class="dot b"></span>Balance Sheet Summary</h4>';
  ovRow('Cash & Equivalents',m.cash_and_equivalents,{green:1,required:1,key:'cash_and_equivalents'});
  ovRow('Current Assets',m.current_assets,{green:1,required:1,key:'current_assets'});
  ovRow('Total Assets',m.total_assets,{green:1,total:1,required:1,key:'total_assets'});
  ovRow('Current Liabilities',m.current_liabilities,{key:'current_liabilities'});
  ovRow('Total Liabilities',m.total_liabilities,{required:1,key:'total_liabilities'});
  ovRow('Stockholders\u2019 Equity',m.stockholders_equity,{green:1,total:1,required:1,key:'stockholders_equity'});
  h+='</div>';

  h+='<div class="ov-sec"><h4><span class="dot b"></span>Debt Summary</h4>';
  ovRow('Short-term Debt',m.short_term_debt,{key:'short_term_debt'});
  ovRow('Long-term Debt',m.long_term_debt,{key:'long_term_debt'});
  if(m.total_debt!=null)ovRow('Total Debt',m.total_debt,{total:1});
  if(m.net_debt!=null)ovRow('Net Debt',m.net_debt,{total:1});
  if(r.debt_to_equity!=null)h+='<div style="padding:0 12px;font:500 11px var(--ff);color:var(--t3)">Debt/Equity: '+(r.debt_to_equity).toFixed(2)+'x</div>';
  h+='</div>';

  h+='<div class="ov-sec"><h4><span class="dot g"></span>Cash Flow Summary</h4>';
  ovRow('Operating Cash Flow',m.operating_cash_flow,{green:1,required:1,key:'operating_cash_flow'});
  ovRow('Capital Expenditures',m.capital_expenditures,{required:1,key:'capital_expenditures'});
  ovRow('Free Cash Flow',m.free_cash_flow,{green:1,total:1,required:1,key:'free_cash_flow'});
  ovRow('Dividends Paid',m.dividends_paid,{key:'dividends_paid'});
  ovRow('Share Repurchases',m.shares_repurchased,{key:'shares_repurchased'});
  if(m.investing_cash_flow!=null)ovRow('Investing Cash Flow',m.investing_cash_flow,{key:'investing_cash_flow'});
  if(m.financing_cash_flow!=null)ovRow('Financing Cash Flow',m.financing_cash_flow,{key:'financing_cash_flow'});
  h+='</div>';

  /* Period comparison — uses correct labels from backend */
  const yoyLabel=d.yoy_label||'vs Prior Year';
  const compLabel=d.comparison_label||'YoY';
  const qm=d.qoq_metrics||{};
  if(pm&&Object.keys(pm).length>0){
    const yoys=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'gross_profit',l:'Gross Profit'},
      {k:'operating_income',l:'Op. Income'},{k:'ebitda',l:'EBITDA'},{k:'total_assets',l:'Total Assets'},
      {k:'operating_cash_flow',l:'Op. Cash Flow'},{k:'free_cash_flow',l:'Free CF'}];
    let hasYoy=yoys.some(y=>m[y.k]!=null&&pm[y.k]!=null);
    if(hasYoy){
      h+='<div class="yoy-sec"><h3>'+esc(yoyLabel)+'</h3>';
      for(const y of yoys){
        const cur=m[y.k],prev=pm[y.k];if(cur==null||prev==null)continue;
        const delta=prev!==0?((cur-prev)/Math.abs(prev)*100):0;
        const up=delta>=0;
        h+='<div class="yoy-row"><span class="lbl">'+y.l+'</span>';
        h+='<span class="val">'+fmtN(cur)+'</span>';
        h+='<span class="delta '+(up?'up':'dn')+'">'+(up?'\u25B2':'\u25BC')+' '+Math.abs(delta).toFixed(1)+'%</span>';
        h+='<span class="prev">vs '+fmtN(prev)+'</span></div>';
      }
      h+='</div>';
    }
    /* QoQ comparison for quarterly */
    let hasQoq=Object.keys(qm).length>0&&yoys.some(y=>m[y.k]!=null&&qm[y.k]!=null);
    if(hasQoq){
      h+='<div class="yoy-sec"><h3>vs Prior Quarter (QoQ)</h3>';
      for(const y of yoys){
        const cur=m[y.k],prev=qm[y.k];if(cur==null||prev==null)continue;
        const delta=prev!==0?((cur-prev)/Math.abs(prev)*100):0;
        const up=delta>=0;
        h+='<div class="yoy-row"><span class="lbl">'+y.l+'</span>';
        h+='<span class="val">'+fmtN(cur)+'</span>';
        h+='<span class="delta '+(up?'up':'dn')+'">'+(up?'\u25B2':'\u25BC')+' '+Math.abs(delta).toFixed(1)+'%</span>';
        h+='<span class="prev">vs '+fmtN(prev)+'</span></div>';
      }
      h+='</div>';
    }
  }
  return h;
}

/* ═══ Semantic row color classification — expenses only ═══ */
const EXP_KW=['expense','cost of','loss','depreciation','amortization','impairment','write-off','write off','restructuring'];
function rowClass(label){
  const ll=label.toLowerCase();
  if(ll.includes('total')||ll.includes('net income')||ll.includes('gross profit'))return'';
  for(const w of EXP_KW)if(ll.includes(w))return' row-exp';
  return'';
}

function bStmt(recs,stmtLabel){
  if(!recs||!recs.length)return'<div style="padding:18px;color:var(--t4);font-size:12px">No statement data</div>';
  const skip=new Set(['concept','standard_concept','level','is_abstract','is_total','abstract','units','decimals']);
  const cols=Object.keys(recs[0]).filter(k=>!skip.has(k));
  const lc=cols.find(c=>c==='label'||c==='Label')||cols[0];const vc=cols.filter(c=>c!==lc);
  let h='<div class="stmt-hdr"><h3>'+esc(stmtLabel)+'</h3>';
  if(_tk)h+='<span class="stmt-tk">'+esc(_tk)+'</span>';
  h+='</div>';
  h+='<table class="st"><thead><tr><th>'+esc(lc)+'</th>';
  for(const c of vc)h+='<th>'+esc(sCol(c))+'</th>';h+='</tr></thead><tbody>';
  for(let ri=0;ri<recs.length;ri++){const rec=recs[ri];const lab=String(rec[lc]||'');
    const ll=lab.toLowerCase();
    const isTotal=ll.includes('total')||ll.includes('net income')||ll.includes('gross profit')||ll.includes('earnings per share')||rec.is_total;
    const isSection=rec.is_abstract||(!rec[vc[0]]&&!rec[vc[1]]&&lab&&!isTotal);
    const rc=rowClass(lab);
    let cls='';
    if(isSection)cls='sec-hdr';
    else if(isTotal)cls='sub';
    cls+=rc;
    h+='<tr class="'+cls+'">';
    if(isSection){h+='<td colspan="'+(vc.length+1)+'">'+esc(lab)+'</td></tr>';continue}
    h+='<td>'+esc(lab)+'</td>';
    for(const c of vc){const v=rec[c];
      if(v==null||v==='')h+='<td style="color:var(--t4)">\u2014</td>';
      else if(typeof v==='number'||!isNaN(Number(v))){const n=Number(v);
        h+='<td'+(n<0?' class="neg"':'')+'>'+fmtN(n);
        h+='<button class="fc-btn" onclick="factCheck(\''+esc(lab)+'\',\''+fmtN(n)+'\',event)" title="Verify">&#x2713;</button></td>'}
      else h+='<td>'+esc(String(v))+'</td>'}h+='</tr>'}
  return h+'</tbody></table>';
}
function sCol(c){if(/^\d{4}-\d{2}/.test(c))try{return new Date(c).toLocaleDateString('en-US',{month:'short',year:'numeric'})}catch(e){}return c.length>14?c.slice(0,12)+'..':c}
function swMainTab(el,id){
  el.parentElement.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('act'));el.classList.add('act');
  const parent=el.closest('.col-r');if(!parent)return;
  parent.querySelectorAll('.main-pane').forEach(p=>p.style.display='none');
  const pane=document.getElementById(id);if(pane)pane.style.display='';
  /* Auto-generate exec summaries on first click */
  if(id==='mt-exec'&&!_execSummaries&&_curData)genExecSummaries();
}
function swTab(el,id){
  const tabs=el.closest('.main-pane')||el.closest('.col-r')||el.parentElement.parentElement;
  el.parentElement.querySelectorAll('.st-tab').forEach(t=>t.classList.remove('act'));el.classList.add('act');
  tabs.querySelectorAll('.st-wrap').forEach(w=>w.style.display='none');document.getElementById(id).style.display=''}

function popTbl(){const a=document.getElementById('tbl');if(!a)return;
  document.getElementById('m-title').textContent=(_tk||'')+' \u2014 '+(_ff||'All')+' Filings';
  document.getElementById('m-body').innerHTML=a.innerHTML;document.getElementById('modal').style.display='flex'}
function closeModal(e){if(e&&e.target!==document.getElementById('modal'))return;document.getElementById('modal').style.display='none'}
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.getElementById('modal').style.display='none';document.getElementById('fc-popup').style.display='none'}});

function fmtN(n){if(n==null||isNaN(n))return'\u2014';const s=n<0?'-':'',a=Math.abs(n);
  if(a>=1e12)return s+'$'+(a/1e12).toFixed(1)+'T';if(a>=1e9)return s+'$'+(a/1e9).toFixed(1)+'B';
  if(a>=1e6)return s+'$'+(a/1e6).toFixed(0)+'M';if(a>=1e3)return s+'$'+(a/1e3).toFixed(0)+'K';
  if(a>0&&a<1)return(n*100).toFixed(1)+'%';if(a===0)return'\u2014';return s+'$'+a.toLocaleString()}
function shortDate(d){if(!d)return'?';if(d.length>=10)try{return new Date(d).toLocaleDateString('en-US',{month:'short',year:'numeric'})}catch(e){}return d.slice(0,7)}

/* ═══ Tool selector ═══ */
async function onToolSel(){
  const sel=document.getElementById('sel-tool');
  if(!sel||!sel.value||!_tk)return;
  const tool=sel.value;

  /* Dashboard — re-render cached data (no API call) */
  if(tool==='dashboard'){if(_curData)render(_curData,'');return}

  /* Direct API calls — no chat routing, no company misrouting.
     Each view calls its specific endpoint with the locked ticker. */
  showLd();
  try{
    if(tool==='entity'){
      /* Entity profile — direct endpoint */
      const r=await fetch('/api/entity/'+encodeURIComponent(_tk));
      const j=await r.json();
      stopLd();
      if(j.error){addA(j.error,1);return}
      renderEntity(j.profile||j);
    }else if(tool==='mda'||tool==='risk_factors'||tool==='business'||tool==='executive_compensation'){
      /* Filing sections — direct endpoint, uses the currently selected period's filing */
      let url='/api/filing-text/'+encodeURIComponent(_tk)+'/'+encodeURIComponent(tool);
      const params=[];
      if(_curAcc)params.push('accession='+encodeURIComponent(_curAcc));
      if(_curData&&_curData.filing_info)params.push('form_type='+encodeURIComponent(_curData.filing_info.form_type||''));
      if(params.length)url+='?'+params.join('&');
      const r=await fetch(url);
      const j=await r.json();
      stopLd();
      if(j.error){addA(j.error,1);return}
      /* Fill in filing_date from current data if the endpoint didn't return it */
      if(!j.filing_date&&_curData&&_curData.filing_info)j.filing_date=_curData.filing_info.filing_date;
      if(!j.form_type&&_curData&&_curData.filing_info)j.form_type=_curData.filing_info.form_type;
      renderFiling(j);
    }else{
      /* Unknown view — fall back to chat (shouldn't happen) */
      const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:_tk+' '+tool})});
      const j=await r.json();
      if(j.type==='error'){stopLd();addA(j.message,1)}
      else if(j.type==='result')handleRes(j);
      else{stopLd();addA(j.message||'Loaded')}
    }
  }catch(e){stopLd();addA('Error: '+e.message,1)}
}

/* ═══ Exec Summary Generation ═══ */
async function genExecSummaries(){
  if(!_curData)return;
  if(_execSummaries)return; /* already generated */
  /* Send only focused context: metrics, ratios, prior, and top statement rows.
     Do NOT send _bgSections (filing text) — too large, makes Claude slow/fail. */
  const focused={
    company_name:_curData.company_name,
    ticker_or_cik:_curData.ticker_or_cik,
    filing_info:_curData.filing_info,
    period_type:_curData.period_type,
    industry_class:_curData.industry_class,
    metrics:_curData.metrics,
    ratios:_curData.ratios,
    prior_metrics:_curData.prior_metrics,
    yoy_label:_curData.yoy_label,
    validation:_curData.validation,
    income_statement:(_curData.income_statement||[]).slice(0,15),
    balance_sheet:(_curData.balance_sheet||[]).slice(0,15),
    cash_flow_statement:(_curData.cash_flow_statement||[]).slice(0,15),
  };
  try{
    const ctrl=new AbortController();
    const timeout=setTimeout(()=>ctrl.abort(),30000); /* 30s timeout */
    const body={ticker:_tk||'',context:focused,sections:{}};
    const r=await fetch('/api/exec-summaries',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
    clearTimeout(timeout);
    const j=await r.json();
    _execSummaries=j;

    /* Update inline exec summary in main view */
    const inlineBody=document.getElementById('exec-inline-body');
    const inlineBadge=document.getElementById('exec-badge');
    if(inlineBody){
      let ih='';
      ih+='<div style="margin-bottom:16px">'+mdCb(j.overall||'')+'</div>';
      if(j.income)ih+='<h4>Income Statement</h4><div style="margin-bottom:12px">'+mdCb(j.income)+'</div>';
      if(j.balance_sheet)ih+='<h4>Balance Sheet</h4><div style="margin-bottom:12px">'+mdCb(j.balance_sheet)+'</div>';
      if(j.cash_flow)ih+='<h4>Cash Flow</h4><div>'+mdCb(j.cash_flow)+'</div>';
      inlineBody.innerHTML=ih;
    }
    if(inlineBadge){inlineBadge.textContent='AI Generated';inlineBadge.classList.add('done')}

    /* Also update the tab pane version */
    const el=document.getElementById('exec-content');
    const ld=document.getElementById('exec-loading');
    if(ld)ld.style.display='none';
    if(el){
      let h='';
      const cards=[
        {key:'overall',title:'Overall Executive Summary',dot:'ov'},
        {key:'income',title:'Income Statement Analysis',dot:'inc'},
        {key:'balance_sheet',title:'Balance Sheet Analysis',dot:'bs'},
        {key:'cash_flow',title:'Cash Flow Analysis',dot:'cf'}
      ];
      for(const c of cards){
        const txt=j[c.key]||'Not available';
        h+='<div class="exec-card"><h4><span class="edot '+c.dot+'"></span>'+c.title+'</h4>';
        h+='<div class="exec-body">'+mdCb(txt)+'</div></div>';
      }
      el.innerHTML=h;
    }
  }catch(e){
    const inlineBody=document.getElementById('exec-inline-body');
    const inlineBadge=document.getElementById('exec-badge');
    const msg=e.name==='AbortError'?'Timed out — try again or check API key':'Failed: '+esc(e.message);
    if(inlineBody)inlineBody.innerHTML='<span style="color:var(--t3);font-size:13px">'+msg+'</span>';
    if(inlineBadge){inlineBadge.textContent='Retry';inlineBadge.style.color='var(--acc)';inlineBadge.style.cursor='pointer';inlineBadge.onclick=()=>{_execSummaries=null;genExecSummaries()}}
    const ld=document.getElementById('exec-loading');
    if(ld)ld.innerHTML='<span style="color:var(--red)">Error: '+esc(e.message)+'</span>';
  }
}

/* ═══ Background section loading ═══ */
async function bgLoadSections(tk){
  if(!tk)return;
  _bgSections={};
  const secs=['mda','risk_factors','business'];
  for(const sec of secs){
    try{
      const r=await fetch('/api/section/'+encodeURIComponent(tk)+'/'+sec);
      const j=await r.json();
      if(j.text&&j.text.length>100)_bgSections[sec]=j.text.slice(0,15000);
    }catch(e){}
  }
}

/* ═══ Right-panel chatbot (AI Assistant) ═══ */
let _cbOpen=false;
function toggleCb(){
  _cbOpen=!_cbOpen;
  const p=document.getElementById('cbpanel');const b=document.getElementById('cb-btn');const fab=document.getElementById('chat-fab');
  if(p)p.classList.toggle('open',_cbOpen);
  if(b)b.classList.toggle('active',_cbOpen);
  if(fab)fab.classList.toggle('active',_cbOpen);
  if(_cbOpen){setTimeout(()=>{const inp=document.getElementById('cb-inp');if(inp)inp.focus()},100)}
}
/* Open chatbot panel and show data-aware welcome */
function openCbWithData(){
  if(!_cbOpen)toggleCb();
  /* Clear previous messages and show data-aware welcome */
  const msgs=document.getElementById('cb-msgs');if(!msgs)return;
  const w=document.getElementById('cb-welcome');if(w)w.remove();
  const d=_curData;if(!d)return;
  const name=d.company_name||_tk||'Company';
  const m=d.metrics||{};
  /* Add data-aware suggestion chips */
  let chips='<div class="cb-chips">';
  chips+='<span class="cb-chip" onclick="cbQ(\'Give me an executive summary of '+esc(name)+'\')">Executive Summary</span>';
  if(m.revenue!=null)chips+='<span class="cb-chip" onclick="cbQ(\'Analyze revenue and margins\')">Revenue Analysis</span>';
  chips+='<span class="cb-chip" onclick="cbQ(\'What are the key risks?\')">Key Risks</span>';
  if(m.operating_cash_flow!=null)chips+='<span class="cb-chip" onclick="cbQ(\'How is cash flow generation?\')">Cash Flow</span>';
  chips+='</div>';
  addCbMsg('**'+name+'** data loaded. '+chips,'ai');
}
function cbQ(t){const inp=document.getElementById('cb-inp');if(inp){inp.value=t;sendCb()}}
document.addEventListener('keydown',e=>{
  if(e.target&&e.target.id==='cb-inp'&&e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendCb()}
});
async function sendCb(){
  const inp=document.getElementById('cb-inp');const btn=document.getElementById('cb-send');
  if(!inp||!inp.value.trim())return;
  const msg=inp.value.trim();inp.value='';if(btn)btn.disabled=true;
  const w=document.getElementById('cb-welcome');if(w)w.remove();
  addCbMsg(msg,'user');
  const tid='cbt'+Date.now();addCbTyping(tid);
  try{
    const ctx=Object.assign({},_curData||{});
    if(_bgSections&&Object.keys(_bgSections).length)ctx._filing_sections=_bgSections;
    const body={message:msg,ticker:_tk||'',context:ctx,history:_chatHistory.slice(-6)};
    const r=await fetch('/api/chatbot',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    const j=await r.json();rmCb(tid);
    const ans=j.answer||'No response.';
    addCbMsg(ans,'ai',j.citations);
    _chatHistory.push({role:'user',content:msg});
    _chatHistory.push({role:'assistant',content:ans});
    if(_chatHistory.length>20)_chatHistory=_chatHistory.slice(-12);
  }catch(e){rmCb(tid);addCbMsg('Error: '+e.message,'ai')}
  if(btn)btn.disabled=false;if(inp)inp.focus();
}
function addCbMsg(text,role,cits){
  const msgs=document.getElementById('cb-msgs');if(!msgs)return;
  const d=document.createElement('div');d.className='cb-msg '+role;
  let inner='<div class="cb-bubble">';
  if(role==='ai'){
    inner+=mdCb(text);
    if(cits&&cits.length){inner+='<div class="cb-cits">';
      for(const c of cits)inner+='<span class="cb-cit">'+esc(c.source||'')+'</span>';
      inner+='</div>'}
  }else{inner+=esc(text)}
  inner+='</div>';d.innerHTML=inner;msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}
function addCbTyping(id){
  const msgs=document.getElementById('cb-msgs');if(!msgs)return;
  const d=document.createElement('div');d.id=id;d.className='cb-msg ai';
  d.innerHTML='<div class="cb-bubble"><div class="cb-typing"><span></span><span></span><span></span></div></div>';
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}
function rmCb(id){const e=document.getElementById(id);if(e)e.remove()}
function mdCb(s){if(!s)return'';
  let h=esc(s);
  h=h.replace(/### (.+)/g,'<h4 style="margin:12px 0 6px;font-size:14px;color:var(--t1)">$1</h4>');
  h=h.replace(/## (.+)/g,'<h3 style="margin:14px 0 8px;font-size:15px;color:var(--t1)">$1</h3>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  h=h.replace(/\*(.+?)\*/g,'<em>$1</em>');
  h=h.replace(/^- /gm,'<li>').replace(/\n- /g,'</li><li>');
  if(h.includes('<li>'))h='<ul>'+h+'</ul>';
  h=h.replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
  return h;
}

/* ═══ Theme toggle (light/dark) ═══ */
function toggleTheme(){
  const isLight=document.body.classList.toggle('light');
  localStorage.setItem('sec-theme',isLight?'light':'dark');
  const btn=document.getElementById('theme-btn');
  if(btn)btn.innerHTML=isLight?'\u2600':'\u263E';
}
(function initTheme(){
  const saved=localStorage.getItem('sec-theme');
  if(saved==='light'){document.body.classList.add('light');
    const btn=document.getElementById('theme-btn');if(btn)btn.innerHTML='\u2600';}
})();
</script>
</body></html>
"""

if __name__ == "__main__":
    import uvicorn

    # Read PORT from environment (Railway sets this) or default to 8877
    port = int(os.environ.get("PORT", "8877"))
    print(f"\n  SEC Terminal \u2192 http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
