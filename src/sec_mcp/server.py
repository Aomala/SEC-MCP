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
from sec_mcp.nlp.sentiment import SentimentAnalyzer
from sec_mcp.nlp.summarizer import FilingSummarizer
from sec_mcp.nlp.ner import EntityExtractor

mcp = FastMCP(name="SEC-MCP")

# Lazy singletons — NLP models load on first use
_sentiment: SentimentAnalyzer | None = None
_summarizer: FilingSummarizer | None = None
_ner: EntityExtractor | None = None


def _get_sentiment() -> SentimentAnalyzer:
    global _sentiment
    if _sentiment is None:
        _sentiment = SentimentAnalyzer()
    return _sentiment


def _get_summarizer() -> FilingSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = FilingSummarizer()
    return _summarizer


def _get_ner() -> EntityExtractor:
    global _ner
    if _ner is None:
        _ner = EntityExtractor()
    return _ner


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
