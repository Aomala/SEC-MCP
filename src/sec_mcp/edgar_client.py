"""EDGAR API wrapper using edgartools."""

from __future__ import annotations

from edgar import Company, set_identity, find

from sec_mcp.config import get_config
from sec_mcp.models import CompanyInfo, FilingMetadata

# Section aliases for friendly name lookup
SECTION_ALIASES: dict[str, str] = {
    # 10-K
    "business": "Item 1",
    "risk factors": "Item 1A",
    "risk_factors": "Item 1A",
    "properties": "Item 2",
    "legal": "Item 3",
    "legal proceedings": "Item 3",
    "mda": "Item 7",
    "md&a": "Item 7",
    "management discussion": "Item 7",
    "financial statements": "Item 8",
    "controls": "Item 9A",
    # 10-Q
    "financial_statements_q": "Part I, Item 1",
    "mda_q": "Part I, Item 2",
    "risk_factors_q": "Part I, Item 1A",
}

_identity_set = False


def _ensure_identity():
    global _identity_set
    if not _identity_set:
        config = get_config()
        set_identity(config.edgar_identity)
        _identity_set = True


def _resolve_section(section: str | None) -> str | None:
    if section is None:
        return None
    return SECTION_ALIASES.get(section.lower().strip(), section)


def search_companies(query: str) -> list[CompanyInfo]:
    """Search for companies by ticker or name."""
    _ensure_identity()
    results = []

    # Try direct ticker lookup first
    try:
        company = Company(query)
        results.append(CompanyInfo(
            name=company.name,
            cik=company.cik,
            ticker=getattr(company, "tickers", [None])[0] if getattr(company, "tickers", None) else None,
            industry=getattr(company, "industry", None),
            sic_code=getattr(company, "sic", None),
        ))
        return results
    except Exception:
        pass

    # Fall back to search
    try:
        matches = find(query)
        if matches is not None:
            for match in matches:
                cik = getattr(match, "cik", None) or getattr(match, "cik_number", None)
                if cik is None:
                    continue
                results.append(CompanyInfo(
                    name=getattr(match, "name", str(match)),
                    cik=int(cik),
                    ticker=getattr(match, "ticker", None),
                    industry=getattr(match, "industry", None),
                    sic_code=getattr(match, "sic", None),
                ))
    except Exception:
        pass

    return results


def get_company(ticker_or_cik: str) -> CompanyInfo:
    """Direct company lookup by ticker or CIK."""
    _ensure_identity()
    company = Company(ticker_or_cik)
    return CompanyInfo(
        name=company.name,
        cik=company.cik,
        ticker=getattr(company, "tickers", [None])[0] if getattr(company, "tickers", None) else None,
        industry=getattr(company, "industry", None),
        sic_code=getattr(company, "sic", None),
    )


def list_filings(
    ticker_or_cik: str,
    form_type: str | None = None,
    limit: int = 10,
) -> list[FilingMetadata]:
    """List available filings for a company."""
    _ensure_identity()
    company = Company(ticker_or_cik)
    filings = company.get_filings(form=form_type) if form_type else company.get_filings()

    results = []
    for filing in filings[:limit]:
        results.append(FilingMetadata(
            accession_number=filing.accession_no,
            form_type=filing.form,
            filing_date=str(filing.filing_date),
            description=getattr(filing, "description", None),
        ))
    return results


def get_filing_content(
    ticker_or_cik: str,
    accession_number: str,
    section: str | None = None,
    max_length: int = 50000,
) -> str:
    """Fetch filing text, optionally a specific section."""
    _ensure_identity()
    company = Company(ticker_or_cik)
    filings = company.get_filings()

    # Find the specific filing by accession number
    filing = None
    for f in filings:
        if f.accession_no == accession_number:
            filing = f
            break

    if filing is None:
        raise ValueError(f"Filing {accession_number} not found for {ticker_or_cik}")

    resolved_section = _resolve_section(section)

    if resolved_section:
        try:
            obj = filing.obj()
            # Try dictionary-style access for the section
            section_text = obj[resolved_section]
            if section_text:
                text = str(section_text)
                return text[:max_length]
        except (KeyError, TypeError, AttributeError):
            pass

    # Fall back to full text
    try:
        text = filing.text()
        if text:
            return text[:max_length]
    except Exception:
        pass

    # Last resort: try HTML
    try:
        html = filing.html()
        if html:
            # Basic HTML tag stripping
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_length]
    except Exception:
        pass

    raise ValueError(f"Could not extract text from filing {accession_number}")
