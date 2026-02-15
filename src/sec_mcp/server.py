"""SEC-MCP: MCP server for SEC filing analysis with BERT-based NLP."""

from __future__ import annotations

from fastmcp import FastMCP

from sec_mcp.config import get_config
from sec_mcp.edgar_client import (
    search_companies,
    get_company,
    list_filings,
    get_filing_content,
)
from sec_mcp.models import (
    CompanyInfo,
    FilingMetadata,
    SentimentAnalysis,
    SummaryResult,
    EntityExtractionResult,
    CombinedAnalysis,
)
from sec_mcp.nlp.sentiment import SentimentAnalyzer
from sec_mcp.nlp.summarizer import FilingSummarizer
from sec_mcp.nlp.ner import EntityExtractor

mcp = FastMCP(name="SEC-MCP")

# Lazy singletons — models load on first use
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
    """Get text from either direct input or filing reference."""
    if text:
        return text
    if not ticker_or_cik or not accession_number:
        raise ValueError(
            "Provide either 'text' directly, or 'ticker_or_cik' + 'accession_number' to fetch from a filing."
        )
    return get_filing_content(ticker_or_cik, accession_number, section)


# ---------------------------------------------------------------------------
# Tool 1: Search Company
# ---------------------------------------------------------------------------


@mcp.tool()
def search_company(query: str) -> list[dict]:
    """Search for a company by ticker symbol or name.

    Returns matching companies with CIK, ticker, and industry info.
    Use a ticker like 'AAPL' for direct lookup, or a name like 'Apple' for search.
    """
    results = search_companies(query)
    return [r.model_dump() for r in results]


# ---------------------------------------------------------------------------
# Tool 2: Get Filing List
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool 3: Get Filing Text
# ---------------------------------------------------------------------------


@mcp.tool()
def get_filing_text(
    ticker_or_cik: str,
    accession_number: str,
    section: str | None = None,
    max_length: int = 50000,
) -> str:
    """Fetch the text content of an SEC filing or a specific section.

    Section names for 10-K: 'Item 1' (Business), 'Item 1A' (Risk Factors),
    'Item 7' (MD&A), 'Item 8' (Financial Statements).
    For 10-Q: 'Part I, Item 1', 'Part I, Item 2', etc.
    For 8-K: 'Item 1.01', 'Item 2.01', etc.
    Also accepts aliases: 'risk factors', 'mda', 'business', etc.
    """
    return get_filing_content(ticker_or_cik, accession_number, section, max_length)


# ---------------------------------------------------------------------------
# Tool 4: Analyze Sentiment
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_sentiment(
    text: str | None = None,
    ticker_or_cik: str | None = None,
    accession_number: str | None = None,
    section: str | None = None,
) -> dict:
    """Run FinBERT financial sentiment analysis on text or a filing section.

    Provide either:
    - 'text': raw text to analyze
    - 'ticker_or_cik' + 'accession_number': fetch and analyze a filing (optionally a specific section)

    Returns positive/negative/neutral sentiment with confidence scores.
    """
    resolved = _resolve_text(text, ticker_or_cik, accession_number, section)
    result = _get_sentiment().analyze(resolved)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Tool 5: Summarize Filing
# ---------------------------------------------------------------------------


@mcp.tool()
def summarize_filing(
    text: str | None = None,
    ticker_or_cik: str | None = None,
    accession_number: str | None = None,
    section: str | None = None,
    max_summary_length: int = 300,
) -> dict:
    """Summarize SEC filing text or a specific section using BART.

    Provide either:
    - 'text': raw text to summarize
    - 'ticker_or_cik' + 'accession_number': fetch and summarize a filing

    Uses hierarchical summarization for long documents.
    """
    resolved = _resolve_text(text, ticker_or_cik, accession_number, section)
    result = _get_summarizer().summarize(resolved, max_summary_length=max_summary_length)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Tool 6: Extract Entities
# ---------------------------------------------------------------------------


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

    Provide either 'text' or 'ticker_or_cik' + 'accession_number'.
    """
    resolved = _resolve_text(text, ticker_or_cik, accession_number, section)
    result = _get_ner().extract(resolved)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Tool 7: Analyze Filing (Combined)
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_filing(
    ticker_or_cik: str,
    accession_number: str,
    section: str | None = None,
) -> dict:
    """Run full analysis on a filing section: sentiment + summary + entity extraction.

    This convenience tool combines all three analyses in one call.
    Provide a section name for focused analysis (e.g., 'Item 1A' for risk factors).
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
