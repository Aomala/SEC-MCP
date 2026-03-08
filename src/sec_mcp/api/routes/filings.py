"""
FastAPI router for filing endpoints.
Provides endpoints for listing filings and retrieving section content.
GET /api/v1/filings/{ticker} -> list of FilingMetadata
GET /api/v1/filings/{ticker}/section -> section text
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

# Import Edgar client for filing operations
from sec_mcp.edgar_client import list_filings, get_filing_content
from sec_mcp.section_segmenter import segment_filing

# Create logger for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1/filings", tags=["filings"])


# Pydantic model for filing metadata
class FilingMetadata(BaseModel):
    """Metadata for a single SEC filing."""
    # Accession number (unique filing identifier)
    accession: str = Field(..., description="SEC accession number")
    # Form type (10-K, 10-Q, 8-K, etc.)
    form_type: str = Field(..., description="SEC form type")
    # Date filing was submitted
    filing_date: str = Field(..., description="Filing submission date (YYYY-MM-DD)")
    # End date of reporting period
    period_end: str = Field(..., description="Period end date (YYYY-MM-DD)")
    # Fiscal year end date
    fiscal_year_end: Optional[str] = Field(None, description="Fiscal year end date (YYYY-MM-DD)")
    # Company name at time of filing
    company_name: str = Field(..., description="Company name as filed")
    # Central Index Key (company identifier)
    cik: str = Field(..., description="SEC Central Index Key")
    # URL to access filing on EDGAR
    edgar_url: str = Field(..., description="Direct EDGAR URL")


# Pydantic model for filing section content
class FilingSection(BaseModel):
    """Content for a specific section of a filing."""
    # Accession number of parent filing
    accession: str = Field(..., description="SEC accession number")
    # Form type of parent filing
    form_type: str = Field(..., description="SEC form type")
    # Section identifier (e.g., "1A", "7", "8")
    section: str = Field(..., description="Section identifier")
    # Human-readable section title
    section_title: str = Field(..., description="Section title")
    # Raw HTML or text content
    content: str = Field(..., description="Section content (may be HTML or text)")
    # Content format type
    content_type: str = Field(default="text", description="Content type: text, html, or xbrl")
    # Character count for content
    length: int = Field(..., description="Content length in characters")


@router.get("/{ticker}", response_model=List[FilingMetadata])
async def list_company_filings(
    ticker: str = Path(..., description="Stock ticker symbol"),
    form_type: Optional[str] = Query(None, description="Filter by form type (10-K, 10-Q, 8-K, etc.)"),
    limit: int = Query(20, description="Maximum number of filings to return (1-100)"),
    offset: int = Query(0, description="Number of filings to skip (for pagination)"),
) -> List[FilingMetadata]:
    """
    List SEC filings for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        form_type: Optional filter by form type (10-K, 10-Q, 8-K, etc.)
        limit: Maximum number of filings (default 20, max 100)
        offset: Number of filings to skip for pagination (default 0)
    
    Returns:
        List of FilingMetadata sorted by filing date descending
    
    Raises:
        HTTPException: 404 if company not found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate limit parameter (1-100 range)
        if limit < 1 or limit > 100:
            # Return 400 for invalid limit
            raise HTTPException(
                status_code=400,
                detail="limit parameter must be between 1 and 100"
            )
        
        # Validate offset parameter (non-negative)
        if offset < 0:
            # Return 400 for negative offset
            raise HTTPException(
                status_code=400,
                detail="offset parameter must be non-negative"
            )
        
        # Log request with parameters
        logger.info(f"Listing filings for {ticker_upper}, form={form_type}, limit={limit}, offset={offset}")
        
        # Call edgar_client to retrieve filings list
        filings_data = list_filings(
            ticker=ticker_upper,
            form_type=form_type,
            limit=limit + offset,  # Add offset to get correct slice
        )
        
        # Verify that filings were returned
        if not filings_data:
            # Log warning and return 404
            logger.warning(f"No filings found for {ticker_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No SEC filings found for {ticker}"
            )
        
        # Apply offset to slice results correctly
        filings_data = filings_data[offset:offset + limit]
        
        # Transform raw filing data to response format
        filings_metadata = []
        for filing in filings_data:
            # Create FilingMetadata for each filing
            metadata = FilingMetadata(
                accession=filing.get("accession", ""),
                form_type=filing.get("form", ""),
                filing_date=filing.get("filed", ""),
                period_end=filing.get("period_end", ""),
                fiscal_year_end=filing.get("fiscal_year_end"),
                company_name=filing.get("company_name", ""),
                cik=filing.get("cik", ""),
                edgar_url=filing.get("url", ""),
            )
            # Add to filings list
            filings_metadata.append(metadata)
        
        # Return list of filing metadata
        return filings_metadata
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Error listing filings for {ticker}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{ticker}/section", response_model=FilingSection)
async def get_filing_section(
    ticker: str = Path(..., description="Stock ticker symbol"),
    accession: str = Query(..., description="SEC accession number"),
    section: str = Query(..., description="Section identifier (e.g., '1A', '7', 'MD&A')"),
) -> FilingSection:
    """
    Get content for a specific section of a filing.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        accession: SEC accession number (e.g., "0001193125-24-001234")
        section: Section identifier (e.g., "1A", "7", "MD&A")
    
    Returns:
        FilingSection with section content
    
    Raises:
        HTTPException: 404 if filing/section not found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate accession number format (basic check)
        if not accession or len(accession) < 10:
            # Return 400 for invalid accession
            raise HTTPException(
                status_code=400,
                detail="Invalid accession number format"
            )
        
        # Validate section identifier (non-empty)
        if not section:
            # Return 400 for empty section
            raise HTTPException(
                status_code=400,
                detail="section parameter is required"
            )
        
        # Log request for debugging
        logger.info(f"Fetching section {section} from filing {accession} for {ticker_upper}")
        
        # Attempt to retrieve full filing content
        filing_content = get_filing_content(accession=accession)
        
        # Verify that filing was retrieved
        if not filing_content:
            # Log warning and return 404
            logger.warning(f"Filing not found: {accession}")
            raise HTTPException(
                status_code=404,
                detail=f"Filing {accession} not found"
            )
        
        # Attempt to segment filing into sections
        try:
            # Call section segmenter to identify section boundaries
            sections = segment_filing(filing_content, form_type=filing_content.get("form_type", "10-K"))
        except Exception as e:
            # Log error but continue with unsegmented content (graceful degradation)
            logger.warning(f"Error segmenting filing {accession}: {e}")
            sections = {}
        
        # Look up requested section in segmented content
        section_content = sections.get(section, "")
        
        # Verify that section was found
        if not section_content:
            # Log warning and return 404
            logger.warning(f"Section {section} not found in filing {accession}")
            raise HTTPException(
                status_code=404,
                detail=f"Section {section} not found in filing {accession}"
            )
        
        # Determine content type (text, html, or xbrl)
        content_type = "text"
        if "<html" in section_content.lower() or "<body" in section_content.lower():
            # Content is HTML formatted
            content_type = "html"
        elif "<xbrl" in section_content.lower():
            # Content is XBRL formatted
            content_type = "xbrl"
        
        # Build and return section response
        return FilingSection(
            accession=accession,
            form_type=filing_content.get("form_type", ""),
            section=section,
            section_title=sections.get(f"{section}_title", section),
            content=section_content,
            content_type=content_type,
            length=len(section_content),
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error fetching section from {accession}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")
