"""Intent parser — extract tool, tickers, year, section from user messages.

Also handles company alias resolution (common names → tickers).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
#  Company alias resolver — maps common names to tickers
# ═══════════════════════════════════════════════════════════════════════════

_ALIASES_PATH = Path(__file__).parent / "company_aliases.json"


def _load_aliases() -> dict[str, str]:
    """Load company name → ticker aliases from JSON file."""
    if _ALIASES_PATH.exists():
        try:
            return json.loads(_ALIASES_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_alias(name: str, ticker: str):
    """Persist a new company alias for future lookups."""
    data = _load_aliases()
    data[name.lower().strip()] = ticker.upper().strip()
    try:
        _ALIASES_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def resolve_name(text: str) -> str | None:
    """Resolve a company name/alias to a ticker symbol.

    Uses exact match first, then word-boundary partial match (longest first).
    Rejects aliases shorter than 3 chars for partial matching to avoid
    false positives like 'ms' matching 'terms'.
    """
    aliases = _load_aliases()
    low = text.lower().strip()
    if low in aliases:
        return aliases[low]
    # Try word-boundary partial match (longest alias first for specificity)
    # Skip very short aliases (< 3 chars) for partial matching — too ambiguous
    for name in sorted(aliases, key=len, reverse=True):
        if len(name) < 3:
            continue
        # Require word boundaries around the alias to prevent substring matches
        # e.g. "apple" should NOT match inside "pineapple"
        if re.search(r'\b' + re.escape(name) + r'\b', low):
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
    # Statement / document fragments that appear in common queries
    "MD", "MDA", "KPI", "YOY", "QOQ", "TTM", "YTD", "FCF", "OCF",
    "DCF", "EPS", "NAV", "ROE", "ROA", "IRR", "NPV", "EBIT", "SGA",
    "ETF", "IPO", "IS", "BS", "CF", "IR", "FY", "PY", "QQ", "AUM",
    "10K", "10Q", "20F", "6K", "40F", "FPI", "DEF",
    # Common English words that look like tickers
    "DO", "SO", "IT", "AT", "ON", "BY", "OR", "AS", "IF", "GO", "UP",
    "HAS", "HAD", "ARE", "WAS", "HIS", "HER", "ITS", "OUR", "ALSO",
    "BUT", "NOT", "NOR", "YET", "HE", "SHE", "WE", "THEY", "THIS",
    "THAT", "ANY", "NEW", "OLD", "BIG", "TOP", "LOW", "NET", "TAX",
    "LOOK", "LIKE", "BEEN", "MORE", "MOST", "OVER", "INTO", "EACH",
    "SOME", "THAN", "VERY", "JUST", "ONLY", "MADE", "GOOD", "BEST",
    "HIGH", "LAST", "LONG", "WELL", "LIST", "KEEP", "MANY", "SAME",
    "STOCK", "SHARE", "PRICE", "VALUE", "DEBT", "CASH", "FUND",
    "BANK", "RATE", "BOND", "LOAN", "COST", "LOSS", "GAIN", "SELL",
    "HOLD", "PLAN", "TERM", "NOTE", "FORM", "ITEM", "PART", "TYPE",
    "VIEW", "HELP", "NEED", "WANT", "KNOW", "THINK", "SEE", "SAY",
    "LET", "PUT", "RUN", "USE", "TRY", "ASK", "SET", "OWN", "PAY",
    "ADD", "GOT", "HIT", "CUT", "BUY",
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


def parse_intent(msg: str) -> dict:
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

    # Fallback: extract uppercase tokens from the CLEANED query (not raw) to avoid
    # picking up fragments from keywords like MD&A → ["MD", "A"]
    if not tickers:
        clean_upper = re.sub(r"[^A-Z\s]", " ", clean_q.upper())
        # Require 2-5 chars (6-char tickers are very rare and cause false positives)
        tokens = re.findall(r"\b[A-Z]{2,5}\b", clean_upper)
        tickers = [t for t in tokens if t not in _STOP]
        # If still empty, fall back to original message but with stricter filtering
        if not tickers:
            tokens = re.findall(r"\b[A-Z]{2,5}\b", upper)
            tickers = [t for t in tokens if t not in _STOP]
        # Cap at 3 tickers from regex fallback to avoid noise
        tickers = tickers[:3]

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
