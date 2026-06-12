"""fineasmcp — always-on financial research MCP server (v2 surface).

The single source of truth for company info: SEC filings, fundamentals,
prices, ownership, insider activity — queryable 24/7 with rich filters.

Tool surface (exactly 9 tools)
──────────────────────────────
  1. search_companies      — ranked company search with rich filters
  2. get_filings           — filings index + EDGAR full-text search
  3. get_filing_section    — clean section text (risk_factors, mdna, item_X…)
  4. get_fundamentals      — XBRL fundamentals, chart-ready, cross-checked
  5. get_quote             — session-aware price, never silently stale
  6. get_insider_activity  — parsed Form 4s with post-transaction holdings
  7. get_ownership         — 13F holders + 13D/G blockholders
  8. screen                — composable valuation/growth/quality/event screener
  9. compare               — side-by-side fundamentals, normalized periods

Contract (enforced by sec_mcp.surface.meta.tool_guard on every tool):
  - every response carries meta = {source, asOf, cacheHit, latencyMs}
  - every error is {error, code, hint} — raw stack traces never leave the server
  - markets being closed is NEVER an error: quotes return the last close
    labeled session="closed"; filings/fundamentals answer at any hour

The previous 22-tool surface lives in server_legacy.py (not served).
"""

from __future__ import annotations

# MCP framework
from fastmcp import FastMCP

# one implementation module per tool (all logic lives there, not here)
from sec_mcp.surface.company_search import search_companies_impl
from sec_mcp.surface.filings import get_filing_section_impl, get_filings_impl
from sec_mcp.surface.fundamentals import compare_impl, get_fundamentals_impl

# response contract wrapper — meta injection + structured errors
from sec_mcp.surface.meta import tool_guard
from sec_mcp.surface.ownership import get_insider_activity_impl, get_ownership_impl
from sec_mcp.surface.quotes import get_quote_impl
from sec_mcp.surface.screen import screen_impl

# The MCP server instance Claude Desktop / Cursor / Fineas connect to
mcp = FastMCP(name="fineasmcp")


@mcp.tool()
@tool_guard("edgar:company_tickers_exchange")
def search_companies(query: str = "", filters: dict | None = None,
                     limit: int = 10) -> dict:
    """Search SEC-registered companies by ticker or name, with rich filters.

    Args:
        query: ticker ('NVDA') or name fragment ('nvidia'). May be empty
            when at least one filter is provided.
        filters: optional dict with any of:
            sector (technology|healthcare|financials|energy|utilities|consumer|
                    industrials|materials|real_estate|communications),
            industry (substring match on EDGAR's SIC description),
            market_cap_min / market_cap_max (USD),
            exchange ('Nasdaq', 'NYSE', …),
            country (substring, e.g. 'Netherlands'),
            ipo_date_after ('YYYY-MM-DD' — proxied by first EDGAR filing date),
            is_sp500 (bool).
        limit: max results (default 10, max 50).

    Returns ranked matches, each with cik + ticker + name + exchange.
    """
    return search_companies_impl(query, filters, limit)


@mcp.tool()
@tool_guard("edgar:submissions/efts")
def get_filings(ticker_or_cik: str | None = None, form_type: str | None = None,
                date_from: str | None = None, date_to: str | None = None,
                full_text_query: str | None = None, limit: int = 20) -> dict:
    """List SEC filings for a company, or full-text search across filings.

    Args:
        ticker_or_cik: 'AAPL' or '0000320193'. Optional when full_text_query
            is set (then the search spans all filers).
        form_type: 10-K, 10-Q, 8-K, S-1, 13F, 13D, 13G, 13D/G, DEF 14A, 4 —
            FPI equivalents (20-F, 6-K, F-1) are included automatically.
        date_from / date_to: ISO dates bounding the filing date.
        full_text_query: phrase to search inside filing documents
            (EDGAR full-text search; coverage starts 2001).
        limit: max rows (default 20, max 100).

    Each row carries accession number, acceptance timestamp, and a direct
    EDGAR URL.
    """
    return get_filings_impl(ticker_or_cik, form_type, date_from, date_to,
                            full_text_query, limit)


@mcp.tool()
@tool_guard("edgar:archives")
def get_filing_section(accession: str, section: str,
                       ticker_or_cik: str | None = None,
                       max_length: int = 80000) -> dict:
    """Extract one section of a filing as clean text (no HTML artifacts).

    Args:
        accession: e.g. '0000320193-24-000123' (from get_filings).
        section: risk_factors | mdna | business | financial_statements,
            or item_X for 8-Ks (e.g. 'item_2.02', 'item_8.01').
        ticker_or_cik: optional — speeds up and disambiguates filer resolution.
        max_length: character cap on returned text (default 80k).
    """
    return get_filing_section_impl(accession, section, ticker_or_cik, max_length)


@mcp.tool()
@tool_guard("edgar:xbrl_companyfacts")
def get_fundamentals(ticker: str, period: str = "annual",
                     metrics: list[str] | None = None, periods_back: int = 4,
                     include_segments: bool = True) -> dict:
    """Standardized fundamentals from SEC XBRL, with provider cross-check.

    Args:
        ticker: company ticker.
        period: 'annual' | 'quarterly' | 'ttm' (ttm = trailing-4-quarter sums).
        metrics: optional subset, e.g. ['revenue', 'netMargin', 'freeCashFlow'].
        periods_back: how many periods (default 4, max 12).
        include_segments: include geographic + product revenue breakdowns
            (chart-ready, with percentages) and a multi-year geo series.

    Returns periods (newest first) with metrics + provenance, a crossCheck
    block (Polygon, 2% tolerance), and chartSeries arrays ready to plot.
    """
    return get_fundamentals_impl(ticker, period, metrics, periods_back,
                                 include_segments)


@mcp.tool()
@tool_guard("price-providers")
def get_quote(ticker: str) -> dict:
    """Current price quote with mandatory freshness metadata.

    Always returns { price, asOf, session, provider, ageSeconds, … }.
    When markets are closed the last close is returned with
    session='closed' — never an error. Quotes are never silently stale:
    ageSeconds and session always say exactly what you're looking at.
    """
    return get_quote_impl(ticker)


@mcp.tool()
@tool_guard("edgar:form4")
def get_insider_activity(ticker: str, date_from: str | None = None,
                         limit: int = 50) -> dict:
    """Insider transactions parsed from SEC Form 4 filings.

    Args:
        ticker: company ticker.
        date_from: optional ISO date — only transactions on/after this date.
        limit: max transactions (default 50, max 200).

    Each row: insider name, role, buy/sell side, shares, price, value,
    post-transaction holdings, and the source accession. A summary block
    aggregates buy/sell counts and net shares for the window.
    """
    return get_insider_activity_impl(ticker, date_from, limit)


@mcp.tool()
@tool_guard("yfinance(13F)+edgar(13D/G)")
def get_ownership(ticker: str) -> dict:
    """Institutional and beneficial ownership for a company.

    Returns the latest 13F institutional holders (aggregated dataset, with
    report dates) plus every SC 13D (activist) / SC 13G (passive >5%)
    filing from the company's EDGAR index, with filing dates and URLs.
    """
    return get_ownership_impl(ticker)


@mcp.tool()
@tool_guard("edgar:xbrl+price+events")
def screen(filters: dict, limit: int = 20) -> dict:
    """Screen the curated universe (300+ tickers) with composable filters.

    Filters (AND-ed):
        valuation: pe_max, ev_ebitda_max
        growth:    rev_growth_min (0.15 = 15% YoY)
        quality:   fcf_positive (bool), net_debt_ebitda_max
        events:    filed_8k_last_7d (bool), insider_buying_last_30d (bool)
        scoping:   sector, market_cap_min, market_cap_max

    The coverage block reports exactly how much of the universe was
    evaluated — results are never silently truncated.
    """
    return screen_impl(filters, limit)


@mcp.tool()
@tool_guard("edgar:xbrl_companyfacts")
def compare(tickers: list[str], metrics: list[str] | None = None,
            period: str = "annual") -> dict:
    """Side-by-side fundamentals for 2-8 companies, normalized periods.

    Args:
        tickers: 2-8 symbols, e.g. ['AAPL', 'MSFT', 'GOOGL'].
        metrics: optional subset (defaults to revenue, netIncome,
            grossMargin, netMargin, freeCashFlow, totalDebt).
        period: 'annual' | 'ttm'.

    Each row carries its fiscal period end so cross-company alignment is
    explicit; failures for individual tickers are reported, not fatal.
    """
    return compare_impl(tickers, metrics, period)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # SSE transport for remote hosting (Railway): python -m sec_mcp.server --sse
    # Default is STDIO for Claude Desktop / Cursor / local MCP clients.
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run()
