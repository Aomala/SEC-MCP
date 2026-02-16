"""EDGAR API wrapper — now backed by direct SEC HTTP APIs.

This module provides the same public interface as before but uses
sec_client.SECClient instead of the edgartools library. This ensures
we always get the freshest filing data directly from SEC EDGAR.

Public functions:
    search_companies(query) -> list[CompanyInfo]
    get_company(ticker_or_cik) -> CompanyInfo
    list_filings(ticker_or_cik, form_type, limit) -> list[FilingMetadata]
    get_filing_content(ticker_or_cik, accession, section, max_length) -> str
"""

from __future__ import annotations

from sec_mcp.models import CompanyInfo, FilingMetadata
from sec_mcp.sec_client import get_sec_client

# Re-export aliases from the canonical source for backward compatibility
from sec_mcp.section_segmenter import SECTION_ALIASES as _SEG_ALIASES

# Map item-ids back to "Item N" format for legacy callers
SECTION_ALIASES: dict[str, str] = {
    name: f"Item {item_id}" for name, item_id in _SEG_ALIASES.items()
}


def search_companies(query: str) -> list[CompanyInfo]:
    """Search for companies by ticker or name.

    Queries the SEC company_tickers.json for matches.
    Returns exact ticker matches first, then partial name matches.
    """
    client = get_sec_client()
    return client.search_companies(query, limit=10)


def get_company(ticker_or_cik: str) -> CompanyInfo:
    """Direct company lookup by ticker or CIK.

    Fetches full company metadata from the SEC submissions endpoint.
    """
    client = get_sec_client()
    return client.get_company_info(ticker_or_cik)


def list_filings(
    ticker_or_cik: str,
    form_type: str | None = None,
    limit: int = 10,
) -> list[FilingMetadata]:
    """List available filings for a company.

    Uses the SEC submissions API which is updated in real-time,
    ensuring we always have the most recent filings.
    """
    client = get_sec_client()
    return client.get_filings(ticker_or_cik, form_type=form_type, limit=limit)


def get_filing_content(
    ticker_or_cik: str,
    accession_number: str,
    section: str | None = None,
    max_length: int = 50000,
) -> str:
    """Fetch filing text, optionally a specific section.

    Downloads the filing document from SEC Archives, strips HTML,
    and extracts the requested section if specified.

    Section aliases: "risk_factors", "business", "mda", etc.
    """
    client = get_sec_client()
    return client.get_filing_document(
        ticker_or_cik,
        accession_number,
        section=section,
        max_length=max_length,
    )
