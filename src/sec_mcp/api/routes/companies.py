"""
FastAPI router for company endpoints.
Provides endpoints for company profile, metadata, and price data.
GET /api/v1/companies/{ticker} -> CompanyProfile (search + metadata)
GET /api/v1/companies/{ticker}/price -> PriceData (from core.market_data)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Import SEC client for company searches
from sec_mcp.sec_client import get_sec_client
from sec_mcp.edgar_client import search_companies

# Create a logger instance for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1/companies", tags=["companies"])


# Pydantic model for company profile response
class CompanyProfile(BaseModel):
    """Response model for company profile endpoint."""
    # Ticker symbol (e.g., AAPL, MSFT)
    ticker: str = Field(..., description="Stock ticker symbol")
    # Company legal name
    name: str = Field(..., description="Official company name")
    # Central Index Key (unique SEC identifier)
    cik: str = Field(..., description="SEC Central Index Key")
    # Industry classification
    industry: Optional[str] = Field(None, description="SIC industry classification")
    # Company description or business summary
    description: Optional[str] = Field(None, description="Company business description")
    # Company website URL
    website: Optional[str] = Field(None, description="Company website")
    # Count of available SEC filings
    filing_count: int = Field(0, description="Number of available SEC filings")
    # Timestamp when data was retrieved
    data_retrieved_at: str = Field(..., description="ISO 8601 timestamp")


# Pydantic model for price data response
class PriceData(BaseModel):
    """Response model for current price data."""
    # Ticker symbol
    ticker: str = Field(..., description="Stock ticker symbol")
    # Current price per share
    current_price: Optional[float] = Field(None, description="Current stock price")
    # Previous close price
    previous_close: Optional[float] = Field(None, description="Previous close price")
    # Day's high price
    day_high: Optional[float] = Field(None, description="Day high price")
    # Day's low price
    day_low: Optional[float] = Field(None, description="Day low price")
    # Year 52-week high price
    year_high: Optional[float] = Field(None, description="52-week high price")
    # Year 52-week low price
    year_low: Optional[float] = Field(None, description="52-week low price")
    # Market capitalization
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    # Data availability status message
    status: str = Field(default="partial", description="Data availability status")


@router.get("/{ticker}", response_model=CompanyProfile)
async def get_company(
    ticker: str = Field(..., description="Stock ticker symbol"),
) -> CompanyProfile:
    """
    Get company profile and metadata.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
    
    Returns:
        CompanyProfile with company details and filing information
    
    Raises:
        HTTPException: 404 if company not found in SEC database
    """
    try:
        # Normalize ticker to uppercase for consistent lookups
        ticker_upper = ticker.upper()
        
        # Search for company in SEC EDGAR database
        logger.info(f"Searching for company profile: {ticker_upper}")
        results = search_companies(ticker_upper)
        
        # Verify that search returned results
        if not results:
            logger.warning(f"Company not found: {ticker_upper}")
            raise HTTPException(status_code=404, detail=f"Company {ticker} not found in SEC database")
        
        # Extract first result as primary match
        company_data = results[0]
        
        # Attempt to get SEC client for filing count
        try:
            sec_client = get_sec_client()
            # Query SEC for filing count (cached, rate-limited)
            filing_count = len(sec_client.get_company_tickers().get(ticker_upper, []))
        except Exception as e:
            # Log error but continue with filing_count=0 (graceful degradation)
            logger.warning(f"Could not retrieve filing count for {ticker_upper}: {e}")
            filing_count = 0
        
        # Build and return company profile response
        return CompanyProfile(
            ticker=ticker_upper,
            name=company_data.get("name", ""),
            cik=company_data.get("cik", ""),
            industry=company_data.get("industry"),
            description=company_data.get("description"),
            website=company_data.get("website"),
            filing_count=filing_count,
            data_retrieved_at=company_data.get("timestamp", ""),
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise
    except Exception as e:
        # Log unexpected errors with full context
        logger.error(f"Unexpected error fetching company {ticker}: {e}", exc_info=True)
        # Return 500 error with generic message
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{ticker}/price", response_model=PriceData)
async def get_company_price(
    ticker: str = Field(..., description="Stock ticker symbol"),
) -> PriceData:
    """
    Get current price data for company.
    
    Note: This endpoint attempts to fetch from market_data module.
    If unavailable, returns partial data with fundamentals only.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
    
    Returns:
        PriceData with current and historical price information
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Log request for debugging
        logger.info(f"Fetching price data for: {ticker_upper}")
        
        # Attempt to import and use market data module (may not be available)
        try:
            # Lazy import to avoid circular dependencies
            from sec_mcp.core.market_data import get_market_data
            
            # Fetch price data from market data engine
            price_data = get_market_data(ticker_upper)
            # If data found, return with success status
            status = "complete"
        except (ImportError, AttributeError, Exception) as e:
            # Market data module unavailable (expected in some deployments)
            logger.warning(f"Market data module unavailable for {ticker_upper}: {e}")
            # Return minimal response with graceful degradation
            price_data = {}
            status = "unavailable"
        
        # Build response with available data or placeholder values
        return PriceData(
            ticker=ticker_upper,
            current_price=price_data.get("current_price"),
            previous_close=price_data.get("previous_close"),
            day_high=price_data.get("day_high"),
            day_low=price_data.get("day_low"),
            year_high=price_data.get("year_high"),
            year_low=price_data.get("year_low"),
            market_cap=price_data.get("market_cap"),
            status=status,
        )
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error fetching price data for {ticker}: {e}", exc_info=True)
        # Return HTTP 500 with error message
        raise HTTPException(status_code=500, detail="Could not retrieve price data")
