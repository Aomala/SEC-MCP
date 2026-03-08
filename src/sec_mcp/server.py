"""SEC-MCP: MCP server for SEC filing analysis.

Tool hierarchy
──────────────
  Base / discovery
    1. search_company           — ticker/name → CIK + metadata
    2. get_filing_list          — list filings (10-K, 10-Q, 8-K …)

  Financials (standardized, industry-aware, year-constrained)
    3. get_financials           — metrics + ratios + validation + opt segments/statements
    4. get_financials_batch     — N companies in parallel
    5. get_income_statement     — just the income statement rows
    6. get_balance_sheet        — just the balance sheet rows
    7. get_cash_flow            — just the cash flow rows
    8. get_financial_ratios     — just computed ratios
    9. get_revenue_segments     — product/service + geographic revenue breakdowns
   10. compare_companies        — side-by-side metrics for multiple tickers

  Narrative (Claude-powered)
   11. explain_financials       — readable narrative explanation of one company
   12. explain_comparison       — readable comparison of multiple companies

  Filing text
   13. get_filing_text          — full or section text from a filing

  NLP analysis
   14. analyze_sentiment        — FinBERT sentiment
   15. summarize_filing         — BART summarization
   16. extract_entities         — NER
   17. analyze_filing           — combined sentiment + summary + entities
"""

from __future__ import annotations

from fastmcp import FastMCP

from sec_mcp.edgar_client import (
    search_companies,
    list_filings,
    get_filing_content,
)
from sec_mcp.financials import extract_financials, extract_financials_batch
from sec_mcp.models import CombinedAnalysis
import logging

mcp = FastMCP(name="SEC-MCP")
log = logging.getLogger(__name__)

# Lazy singletons — NLP models load on first use, with Claude fallback
_sentiment = None
_summarizer = None
_ner = None
_peer_engine = None
_screener = None


def _get_sentiment():
    global _sentiment
    if _sentiment is None:
        try:
            from sec_mcp.nlp.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            analyzer._load()  # Force model download/load to detect ImportError early
            _sentiment = analyzer
        except (ImportError, OSError) as exc:
            log.info("Local sentiment model unavailable (%s), using Claude fallback", exc)
            from sec_mcp.nlp.claude_fallback import ClaudeSentimentAnalyzer
            _sentiment = ClaudeSentimentAnalyzer()
    return _sentiment


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from sec_mcp.nlp.summarizer import FilingSummarizer
            summarizer = FilingSummarizer()
            summarizer._load()
            _summarizer = summarizer
        except (ImportError, OSError) as exc:
            log.info("Local summarizer unavailable (%s), using Claude fallback", exc)
            from sec_mcp.nlp.claude_fallback import ClaudeFilingSummarizer
            _summarizer = ClaudeFilingSummarizer()
    return _summarizer


def _get_ner():
    global _ner
    if _ner is None:
        try:
            from sec_mcp.nlp.ner import EntityExtractor
            extractor = EntityExtractor()
            extractor._load()
            _ner = extractor
        except (ImportError, OSError) as exc:
            log.info("Local NER model unavailable (%s), using Claude fallback", exc)
            from sec_mcp.nlp.claude_fallback import ClaudeEntityExtractor
            _ner = ClaudeEntityExtractor()
    return _ner


def _get_peer_engine():
    """Lazy-load PeerEngine singleton for peer discovery."""
    global _peer_engine
    if _peer_engine is None:
        try:
            from sec_mcp.core.peer_engine import PeerEngine
            _peer_engine = PeerEngine()
        except ImportError as exc:
            log.error("PeerEngine module unavailable: %s", exc)
            raise
    return _peer_engine


def _get_screener():
    """Lazy-load Screener singleton for company screening."""
    global _screener
    if _screener is None:
        try:
            from sec_mcp.core.screener import Screener
            _screener = Screener()
        except ImportError as exc:
            log.error("Screener module unavailable: %s", exc)
            raise
    return _screener


def _resolve_text(
    text: str | None,
    ticker_or_cik: str | None,
    accession_number: str | None,
    section: str | None,
) -> str:
    """Resolve text from direct input or by fetching a filing.

    If ticker is provided without accession, auto-finds the latest 10-K/10-Q.
    """
    if text:
        return text
    if not ticker_or_cik:
        raise ValueError(
            "Provide either 'text' directly, or 'ticker_or_cik' "
            "(+ optional 'accession_number') to fetch from a filing."
        )
    if not accession_number:
        from sec_mcp.sec_client import get_sec_client
        _client = get_sec_client()
        # Use smart search that auto-tries FPI forms (20-F/6-K)
        filings = _client.get_filings_smart(ticker_or_cik, form_type="10-K", limit=1)
        if not filings:
            filings = _client.get_filings_smart(ticker_or_cik, form_type="10-Q", limit=1)
        if not filings:
            raise ValueError(f"No filings found for {ticker_or_cik}")
        accession_number = filings[0].accession_number

    return get_filing_content(ticker_or_cik, accession_number, section)


# ═══════════════════════════════════════════════════════════════════════════
#  BASE / DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def search_company(query: str) -> list[dict]:
    """Search for a company by ticker symbol or name.

    Returns matching companies with CIK, ticker, SIC code, and industry info.
    Use a ticker like 'AAPL' for direct lookup, or a name like 'Apple' for search.
    """
    results = search_companies(query)
    return [r.model_dump() for r in results]


@mcp.tool()
def get_filing_list(
    ticker_or_cik: str,
    form_type: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """List available SEC filings for a company.

    Filter by form_type: '10-K' (annual), '10-Q' (quarterly), '8-K' (current events).
    Returns filing date, accession number, and description.
    """
    results = list_filings(ticker_or_cik, form_type=form_type, limit=limit)
    return [r.model_dump() for r in results]


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIALS (standardized, industry-aware, year-constrained)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_financials(
    ticker_or_cik: str,
    year: int | None = None,
    include_statements: bool = False,
    include_segments: bool = False,
) -> dict:
    """Get standardized SEC XBRL financials for a company.

    Args:
        ticker_or_cik: ticker (e.g. 'AAPL') or CIK number
        year: fiscal year to retrieve (e.g. 2024, 2023). None = latest available.
        include_statements: also return full income_statement, balance_sheet,
            and cash_flow_statement as row-level tables
        include_segments: also return revenue_segments (product/service breakdown)
            and geographic_segments

    Returns metrics, ratios, validation warnings, and industry classification.
    For banks, revenue is properly aggregated (NII + non-interest + trading).
    """
    result = extract_financials(
        ticker_or_cik,
        year=year,
        include_statements=include_statements,
        include_segments=include_segments,
    )
    return result or {"ticker_or_cik": ticker_or_cik, "error": "No data"}


@mcp.tool()
def get_financials_batch(
    tickers: list[str],
    year: int | None = None,
    include_statements: bool = False,
    include_segments: bool = False,
    max_workers: int = 5,
) -> list[dict]:
    """Get standardized financials for multiple companies in parallel.

    Args:
        tickers: list of ticker symbols (e.g. ['AAPL', 'MSFT', 'GOOGL'])
        year: fiscal year (None = latest)
        include_statements: include full statement tables
        include_segments: include revenue + geographic segments
        max_workers: parallel threads (default 5)
    """
    return extract_financials_batch(
        tickers,
        year=year,
        include_statements=include_statements,
        include_segments=include_segments,
        max_workers=max_workers,
    )


@mcp.tool()
def get_income_statement(ticker_or_cik: str, year: int | None = None) -> list[dict]:
    """Get the income statement for a company as a list of row dicts.

    Args:
        ticker_or_cik: ticker or CIK
        year: fiscal year (None = latest)

    Each row has a 'label' and one or more period columns with values.
    """
    result = extract_financials(
        ticker_or_cik, year=year, include_statements=True
    )
    if result and result.get("income_statement"):
        return result["income_statement"]
    return []


@mcp.tool()
def get_balance_sheet(ticker_or_cik: str, year: int | None = None) -> list[dict]:
    """Get the balance sheet for a company.

    Args:
        ticker_or_cik: ticker or CIK
        year: fiscal year (None = latest)
    """
    result = extract_financials(
        ticker_or_cik, year=year, include_statements=True
    )
    if result and result.get("balance_sheet"):
        return result["balance_sheet"]
    return []


@mcp.tool()
def get_cash_flow(ticker_or_cik: str, year: int | None = None) -> list[dict]:
    """Get the cash flow statement for a company.

    Args:
        ticker_or_cik: ticker or CIK
        year: fiscal year (None = latest)
    """
    result = extract_financials(
        ticker_or_cik, year=year, include_statements=True
    )
    if result and result.get("cash_flow_statement"):
        return result["cash_flow_statement"]
    return []


@mcp.tool()
def get_financial_ratios(ticker_or_cik: str, year: int | None = None) -> dict:
    """Get computed financial ratios for a company.

    Args:
        ticker_or_cik: ticker or CIK
        year: fiscal year (None = latest)

    Returns: gross_margin, operating_margin, net_margin, ROA, ROE,
    current_ratio, debt_to_equity, debt_to_assets, ebitda_margin,
    fcf_margin, ocf_to_net_income.
    """
    result = extract_financials(ticker_or_cik, year=year)
    if result is None:
        return {"ticker_or_cik": ticker_or_cik, "error": "No data"}
    return {
        "ticker_or_cik": result.get("ticker_or_cik"),
        "company_name": result.get("company_name"),
        "fiscal_year": result.get("fiscal_year"),
        "industry_class": result.get("industry_class"),
        "ratios": result.get("ratios", {}),
        "metrics": result.get("metrics", {}),
        "validation": result.get("validation", []),
    }


@mcp.tool()
def get_revenue_segments(ticker_or_cik: str, year: int | None = None) -> dict:
    """Get revenue segmentation for a company.

    Args:
        ticker_or_cik: ticker or CIK
        year: fiscal year (None = latest)

    Returns:
      - revenue_segments: product/service breakdown (e.g. iPhone, Services, Mac)
      - geographic_segments: regional breakdown (e.g. Americas, Europe, China)
    """
    result = extract_financials(
        ticker_or_cik, year=year, include_segments=True
    )
    if result is None:
        return {"ticker_or_cik": ticker_or_cik, "error": "No data"}
    return {
        "ticker_or_cik": result.get("ticker_or_cik"),
        "company_name": result.get("company_name"),
        "fiscal_year": result.get("fiscal_year"),
        "segments": result.get("segments", {}),
        "total_revenue": result.get("metrics", {}).get("revenue"),
    }


@mcp.tool()
def compare_companies(
    tickers: list[str],
    year: int | None = None,
    metrics_only: bool = True,
) -> list[dict]:
    """Side-by-side financial comparison of multiple companies.

    Args:
        tickers: list of ticker symbols
        year: fiscal year (None = latest)
        metrics_only: True = slim output (no full statements). False = everything.

    Returns a list where each entry has company_name, metrics, ratios, validation.
    """
    results = extract_financials_batch(
        tickers,
        year=year,
        include_statements=not metrics_only,
        include_segments=True,
    )
    if metrics_only:
        for r in results:
            for key in ("income_statement", "balance_sheet",
                        "cash_flow_statement", "metrics_sourced"):
                r.pop(key, None)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  NARRATIVE (Claude-powered readable explanations)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def explain_financials(
    ticker_or_cik: str,
    year: int | None = None,
    focus: str | None = None,
) -> str:
    """Get a readable, analyst-quality narrative explanation of a company's financials.

    Uses Claude to transform raw financial data into clear prose with:
    - Executive summary
    - Revenue analysis (with segments if available)
    - Profitability discussion
    - Balance sheet health
    - Cash flow analysis
    - Key concerns or strengths

    Args:
        ticker_or_cik: ticker or CIK
        year: fiscal year (None = latest)
        focus: optional focus area (e.g. 'profitability', 'cash flow', 'segments',
               'balance sheet', 'growth')

    Requires ANTHROPIC_API_KEY in .env.
    """
    from sec_mcp.narrator import explain_financials as _narrate

    data = extract_financials(
        ticker_or_cik,
        year=year,
        include_segments=True,
    )
    if data is None or data.get("error"):
        err = (data or {}).get("error", "No data available")
        return f"Could not generate narrative: {err}"

    return _narrate(data, focus=focus)


@mcp.tool()
def explain_comparison(
    tickers: list[str],
    year: int | None = None,
    focus: str | None = None,
) -> str:
    """Get a readable comparative analysis of multiple companies' financials.

    Uses Claude to produce a side-by-side narrative with tables comparing
    key metrics, strengths, and weaknesses across companies.

    Args:
        tickers: list of ticker symbols to compare
        year: fiscal year (None = latest)
        focus: optional focus area

    Requires ANTHROPIC_API_KEY in .env.
    """
    from sec_mcp.narrator import explain_comparison as _narrate_compare

    results = extract_financials_batch(
        tickers,
        year=year,
        include_segments=True,
    )
    # Filter out total failures
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return "Could not generate comparison: no valid financial data found."

    return _narrate_compare(valid, focus=focus)


# ═══════════════════════════════════════════════════════════════════════════
#  FILING TEXT
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_filing_text(
    ticker_or_cik: str,
    accession_number: str | None = None,
    section: str | None = None,
    max_length: int = 100000,
) -> str:
    """Fetch the text content of an SEC filing or a specific section.

    Section names for 10-K: 'Item 1' (Business), 'Item 1A' (Risk Factors),
    'Item 7' (MD&A), 'Item 8' (Financial Statements).
    For 10-Q: 'Part I, Item 1', 'Part I, Item 2', etc.
    Also accepts aliases: 'risk factors', 'risk_factors', 'mda', 'md&a',
    'management discussion', 'business', 'controls', 'legal', etc.

    If accession_number is not provided, the latest 10-K is used automatically.

    Returns the extracted section text, or the full filing text if section
    extraction fails (with a note at the top).
    """
    if not accession_number:
        from sec_mcp.sec_client import get_sec_client
        _sc = get_sec_client()
        filings = _sc.get_filings_smart(ticker_or_cik, form_type="10-K", limit=1)
        if not filings:
            filings = _sc.get_filings_smart(ticker_or_cik, form_type="10-Q", limit=1)
        if not filings:
            return f"No filings found for {ticker_or_cik}"
        accession_number = filings[0].accession_number

    text = get_filing_content(ticker_or_cik, accession_number, section, max_length)

    if section and (not text or len(text.strip()) < 200):
        full = get_filing_content(ticker_or_cik, accession_number, None, max_length)
        section_label = {
            "risk_factors": "Risk Factors (Item 1A)", "mda": "MD&A (Item 7)",
            "business": "Business (Item 1)", "financial_statements": "Financial Statements (Item 8)",
            "legal": "Legal Proceedings (Item 3)", "controls": "Controls (Item 9A)",
        }.get(section, section)
        return (
            f"[Could not isolate '{section_label}' section. "
            f"Returning full filing text.]\n\n" + full
        )
    return text


# ═══════════════════════════════════════════════════════════════════════════
#  NLP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def analyze_sentiment(
    text: str | None = None,
    ticker_or_cik: str | None = None,
    accession_number: str | None = None,
    section: str | None = None,
) -> dict:
    """Run FinBERT financial sentiment analysis on text or a filing section.

    Provide either 'text' directly, or 'ticker_or_cik' + 'accession_number'.
    Returns positive/negative/neutral sentiment with confidence scores.
    """
    resolved = _resolve_text(text, ticker_or_cik, accession_number, section)
    result = _get_sentiment().analyze(resolved)
    return result.model_dump()


@mcp.tool()
def summarize_filing(
    text: str | None = None,
    ticker_or_cik: str | None = None,
    accession_number: str | None = None,
    section: str | None = None,
    max_summary_length: int = 300,
) -> dict:
    """Summarize SEC filing text or a specific section using BART.

    Provide either 'text' directly, or 'ticker_or_cik' + 'accession_number'.
    Uses hierarchical summarization for long documents.
    """
    resolved = _resolve_text(text, ticker_or_cik, accession_number, section)
    result = _get_summarizer().summarize(resolved, max_summary_length=max_summary_length)
    return result.model_dump()


@mcp.tool()
def extract_entities(
    text: str | None = None,
    ticker_or_cik: str | None = None,
    accession_number: str | None = None,
    section: str | None = None,
) -> dict:
    """Extract named entities from SEC filing text.

    Finds companies (ORG), people (PER), locations (LOC),
    monetary values (MONEY), dates (DATE), and percentages (PERCENT).
    """
    resolved = _resolve_text(text, ticker_or_cik, accession_number, section)
    result = _get_ner().extract(resolved)
    return result.model_dump()


@mcp.tool()
def analyze_filing(
    ticker_or_cik: str,
    accession_number: str,
    section: str | None = None,
) -> dict:
    """Run full analysis on a filing section: sentiment + summary + entity extraction.

    This convenience tool combines all three NLP analyses in one call.
    """
    text = get_filing_content(ticker_or_cik, accession_number, section)

    sentiment = _get_sentiment().analyze(text)
    summary = _get_summarizer().summarize(text)
    entities = _get_ner().extract(text)

    result = CombinedAnalysis(
        sentiment=sentiment,
        summary=summary,
        entities=entities,
    )
    return result.model_dump()



# ═══════════════════════════════════════════════════════════════════════════
#  V2: MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Get current stock price, change, volume, and market cap for a company.
    
    Returns real-time(ish) price data including 52-week range and P/E ratio.
    Requires yfinance package. Returns error dict if unavailable.
    """
    try:
        # Try to import and use market data provider
        import yfinance as yf
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Build response with available fields
        result = {
            "ticker": ticker,
            "price": info.get("currentPrice"),
            "change": info.get("regularMarketChange"),
            "change_pct": info.get("regularMarketChangePercent"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "source": "yfinance",
            "timestamp": None,
        }
        return result
    except ImportError:
        # yfinance not available
        return {
            "error": "Market data module (yfinance) not available. Install with: pip install yfinance"
        }
    except Exception as e:
        # API error or ticker not found
        return {"error": f"Failed to fetch stock price for {ticker}: {str(e)}"}


@mcp.tool()
def get_valuation_metrics(ticker: str, year: int | None = None) -> dict:
    """Get valuation metrics combining market price with XBRL fundamentals.
    
    Computes P/E, P/S, P/B, EV/EBITDA, EV/Revenue by combining
    live stock price with SEC filing data.
    """
    try:
        # Get XBRL metrics via extract_financials
        financials = extract_financials(ticker, year=year)
        if "error" in financials:
            return {"error": f"Could not fetch financials for {ticker}: {financials.get('error')}"}
        
        # Get market data via yfinance
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract needed values
        market_cap = info.get("marketCap")
        current_price = info.get("currentPrice")
        shares_outstanding = info.get("sharesOutstanding")
        
        # Extract XBRL metrics
        metrics = financials.get("metrics", {})
        revenue = metrics.get("revenue")
        net_income = metrics.get("net_income")
        total_assets = metrics.get("total_assets")
        stockholders_equity = metrics.get("stockholders_equity")
        ebitda = metrics.get("ebitda")
        total_debt = (metrics.get("long_term_debt", 0) or 0) + (metrics.get("short_term_debt", 0) or 0)
        
        # Compute valuation metrics
        pe_ratio = None
        ps_ratio = None
        pb_ratio = None
        ev_ebitda = None
        ev_revenue = None
        
        if current_price and net_income and shares_outstanding:
            pe_ratio = (current_price * shares_outstanding) / net_income if net_income > 0 else None
        
        if current_price and revenue and shares_outstanding:
            ps_ratio = (current_price * shares_outstanding) / revenue if revenue > 0 else None
        
        if current_price and stockholders_equity and shares_outstanding:
            pb_ratio = (current_price * shares_outstanding) / stockholders_equity if stockholders_equity > 0 else None
        
        if market_cap and ebitda:
            ev_ebitda = market_cap / ebitda if ebitda > 0 else None
        
        if market_cap and revenue:
            ev_revenue = market_cap / revenue if revenue > 0 else None
        
        result = {
            "ticker": ticker,
            "market_cap": market_cap,
            "enterprise_value": (market_cap + total_debt) if market_cap else None,
            "pe_ratio": pe_ratio,
            "ps_ratio": ps_ratio,
            "pb_ratio": pb_ratio,
            "ev_ebitda": ev_ebitda,
            "ev_revenue": ev_revenue,
            "dividend_yield": info.get("dividendYield"),
        }
        return result
    except ImportError:
        return {"error": "yfinance not available. Install with: pip install yfinance"}
    except Exception as e:
        return {"error": f"Failed to compute valuation metrics: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
#  V2: FILING DIFF
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def diff_financials(ticker: str, year1: int, year2: int) -> dict:
    """Compare financial metrics between two fiscal years.
    
    Shows what changed: revenue growth, margin expansion/compression,
    debt changes, etc. Each metric gets a significance rating.
    """
    try:
        # Fetch financials for both years
        fin1 = extract_financials(ticker, year=year1)
        fin2 = extract_financials(ticker, year=year2)
        
        # Check for errors
        if "error" in fin1:
            return {"error": f"Could not fetch {year1} financials: {fin1.get('error')}"}
        if "error" in fin2:
            return {"error": f"Could not fetch {year2} financials: {fin2.get('error')}"}
        
        # Extract metrics from both years
        metrics1 = fin1.get("metrics", {})
        metrics2 = fin2.get("metrics", {})
        ratios1 = fin1.get("ratios", {})
        ratios2 = fin2.get("ratios", {})
        
        # Build list of changes
        changes = []
        
        # Key metrics to compare
        key_metrics = [
            "revenue", "net_income", "gross_profit", "operating_income", "ebitda",
            "total_assets", "total_liabilities", "stockholders_equity",
            "operating_cash_flow", "free_cash_flow", "long_term_debt"
        ]
        
        for metric_name in key_metrics:
            old_val = metrics1.get(metric_name)
            new_val = metrics2.get(metric_name)
            
            # Skip if both are None
            if old_val is None and new_val is None:
                continue
            
            # Calculate change
            change = None
            change_pct = None
            if old_val is not None and new_val is not None and old_val != 0:
                change = new_val - old_val
                change_pct = (change / abs(old_val)) * 100
            
            # Determine significance
            significance = "minor"
            if change_pct is not None:
                abs_pct = abs(change_pct)
                if abs_pct > 20:
                    significance = "major"
                elif abs_pct > 10:
                    significance = "moderate"
            
            changes.append({
                "metric": metric_name,
                "old_value": old_val,
                "new_value": new_val,
                "change": change,
                "change_pct": change_pct,
                "significance": significance,
            })
        
        # Also compare key ratios
        ratio_keys = [
            "gross_margin", "operating_margin", "net_margin",
            "return_on_assets", "return_on_equity", "current_ratio",
            "debt_to_equity"
        ]
        
        for ratio_name in ratio_keys:
            old_val = ratios1.get(ratio_name)
            new_val = ratios2.get(ratio_name)
            
            if old_val is None and new_val is None:
                continue
            
            change = None
            change_pct = None
            if old_val is not None and new_val is not None and old_val != 0:
                change = new_val - old_val
                change_pct = (change / abs(old_val)) * 100
            
            significance = "minor"
            if change_pct is not None and abs(change_pct) > 15:
                significance = "major" if abs(change_pct) > 30 else "moderate"
            
            changes.append({
                "metric": ratio_name,
                "old_value": old_val,
                "new_value": new_val,
                "change": change,
                "change_pct": change_pct,
                "significance": significance,
            })
        
        result = {
            "ticker": ticker,
            "year1": year1,
            "year2": year2,
            "changes": changes,
            "summary": None,
        }
        return result
    except Exception as e:
        return {"error": f"Failed to compare financials: {str(e)}"}


@mcp.tool()
def diff_filing_section(ticker: str, section: str, year1: int, year2: int) -> dict:
    """Compare a specific filing section between two years.
    
    Useful for tracking changes in Risk Factors, MD&A, etc.
    Returns a summary of key changes (Claude-powered if available).
    
    Section names: 'risk_factors', 'mda', 'business', 'controls', 'legal'
    """
    try:
        # Map section names to form sections
        section_map = {
            "risk_factors": "Item 1A. Risk Factors",
            "mda": "Item 7. Management's Discussion and Analysis",
            "business": "Item 1. Business",
            "controls": "Item 9A. Changes in and Disagreements with Accountants",
            "legal": "Item 3. Legal Proceedings",
        }
        
        mapped_section = section_map.get(section.lower(), section)
        
        # Get filings for both years
        filings1 = list_filings(ticker, form="10-K", limit=5)
        filings2 = list_filings(ticker, form="10-K", limit=5)
        
        # Find the filings closest to each year
        text1 = None
        text2 = None
        
        for filing in filings1:
            filing_year = int(filing.get("filing_date", "")[:4])
            if filing_year == year1:
                text1 = get_filing_content(ticker, filing.get("accession_number"), mapped_section)
                break
        
        for filing in filings2:
            filing_year = int(filing.get("filing_date", "")[:4])
            if filing_year == year2:
                text2 = get_filing_content(ticker, filing.get("accession_number"), mapped_section)
                break
        
        if not text1 or not text2:
            return {"error": f"Could not retrieve section '{section}' for both years"}
        
        # Try to use Claude to summarize the changes
        summary = None
        try:
            from sec_mcp.narrator import get_narrator
            narrator = get_narrator()
            if narrator:
                summary = narrator.explain_section_change(section, text1, text2)
        except Exception:
            # Fall back to no Claude summary
            pass
        
        result = {
            "ticker": ticker,
            "section": section,
            "year1": year1,
            "year2": year2,
            "text1_snippet": text1[:500] if text1 else None,
            "text2_snippet": text2[:500] if text2 else None,
            "summary": summary,
        }
        return result
    except Exception as e:
        return {"error": f"Failed to compare filing sections: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
#  V2: PEER DISCOVERY & SCREENING
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def find_peers(ticker: str, max_peers: int = 5) -> list[dict]:
    """Find peer companies for comparison.

    Uses industry classification (SIC code) and a curated peer map
    to find the most relevant comparable companies.

    Args:
        ticker: Company ticker (e.g., "AAPL")
        max_peers: Maximum number of peers to return (default: 5)

    Returns:
        List of peer company dicts with fields: ticker, name, sic,
        relevance_score (0-1), reason
    """
    try:
        # Get PeerEngine singleton and find peers
        engine = _get_peer_engine()
        results = engine.find_peers(ticker.upper(), max_peers=max_peers)

        # If no peers found, return empty list (not an error)
        if not results:
            return []

        return results
    except ImportError:
        # Fallback: use search_companies for basic validation
        try:
            company = search_companies(ticker)
            if not company or len(company) == 0:
                return [{"error": f"Company {ticker} not found"}]
            return [{"error": "Peer discovery engine not available, but company found"}]
        except Exception as e:
            return [{"error": f"Failed to find peers: {str(e)}"}]
    except Exception as e:
        return [{"error": f"Failed to find peers: {str(e)}"}]


@mcp.tool()
def screen_companies(filters: list[dict], limit: int = 20) -> list[dict]:
    """Screen SEC-filing companies by financial criteria.

    Each filter: {"metric": "net_margin", "operator": ">", "value": 25.0}
    Supported metrics: revenue, net_income, gross_margin, operating_margin,
    net_margin, roa, roe, total_assets, market_cap
    Operators: >, <, >=, <=, ==, between (between requires "value_max" key)

    Args:
        filters: List of filter dicts with keys: metric, operator, value
                 and optional value_max for "between" operator
        limit: Maximum number of results to return (default: 20)

    Returns:
        List of matching company dicts with fields: ticker, company_name,
        metrics, ratios, matched

    Note: Screening is compute-intensive. Results are cached for 30 minutes.
    """
    try:
        # Get Screener singleton and run screening
        screener = _get_screener()
        results = screener.screen(filters, limit=limit)

        # If no matches found, return empty list (not an error)
        if not results:
            return []

        return results
    except ValueError as e:
        # Filter validation error from screener
        return [{"error": f"Invalid filter: {str(e)}"}]
    except ImportError:
        return [{"error": "Screener module not available"}]
    except Exception as e:
        return [{"error": f"Failed to screen companies: {str(e)}"}]


@mcp.tool()
def export_financials(ticker: str, format: str = "json", year: int | None = None) -> dict:
    """Export financial data in a specified format.
    
    Args:
        ticker: company ticker
        format: 'json' or 'csv'
        year: fiscal year (None = latest)
    
    Returns the data as a string in the requested format,
    or as a structured dict for JSON.
    """
    try:
        # Get financials
        financials = extract_financials(ticker, year=year)
        
        if "error" in financials:
            return {"error": f"Could not fetch financials: {financials.get('error')}"}
        
        if format.lower() == "json":
            # Return as dict
            return financials
        elif format.lower() == "csv":
            # Convert to CSV format
            import csv
            from io import StringIO
            
            metrics = financials.get("metrics", {})
            ratios = financials.get("ratios", {})
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write metrics
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                if value is not None:
                    writer.writerow([key, value])
            
            writer.writerow([])
            
            # Write ratios
            writer.writerow(["Ratio", "Value"])
            for key, value in ratios.items():
                if value is not None:
                    writer.writerow([key, value])
            
            csv_string = output.getvalue()
            return {
                "ticker": ticker,
                "format": "csv",
                "data": csv_string,
            }
        else:
            return {"error": f"Unsupported format: {format}. Use 'json' or 'csv'"}
    except Exception as e:
        return {"error": f"Failed to export financials: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Support SSE transport for Railway/remote hosting:
    #   python -m sec_mcp.server --sse
    # Default is STDIO (for Claude Desktop / Cursor / local MCP clients)
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run()
