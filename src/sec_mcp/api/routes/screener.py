"""
FastAPI router for financial screener endpoints.
Provides endpoints for filtering and screening companies based on financial metrics.
POST /api/v1/screener -> list of matching companies
"""

import logging
from typing import Optional, List, Literal, Dict, Any

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

# Create logger for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1/screener", tags=["screener"])


# Pydantic model for individual screening filter
class ScreenerFilter(BaseModel):
    """Single filter criterion for screener."""
    # Metric name (e.g., "revenue", "pe_ratio", "debt_to_equity")
    metric: str = Field(..., description="Metric name")
    # Comparison operator (gt, gte, lt, lte, eq)
    operator: Literal["gt", "gte", "lt", "lte", "eq"] = Field(
        ...,
        description="Comparison operator: gt (>), gte (>=), lt (<), lte (<=), eq (==)"
    )
    # Threshold value for comparison
    value: float = Field(..., description="Threshold value")
    # Optional logical operator for combining filters (and/or)
    logic: Literal["and", "or"] = Field(default="and", description="Logic for combining with next filter")


# Pydantic model for screener request
class ScreenerRequest(BaseModel):
    """Request body for stock screener."""
    # List of filter criteria
    filters: List[ScreenerFilter] = Field(
        ...,
        description="List of filter criteria"
    )
    # Maximum number of results to return
    limit: int = Field(default=50, description="Maximum results to return (1-1000)")
    # Fiscal year to use for metrics
    year: Optional[int] = Field(None, description="Fiscal year (latest if not specified)")
    # Form type to use (10-K or 10-Q)
    form_type: str = Field(default="10-K", description="Form type: 10-K or 10-Q")
    # Sector to filter by (optional)
    sector: Optional[str] = Field(None, description="Sector filter (Technology, Healthcare, etc.)")
    # Minimum market cap filter (in millions USD)
    min_market_cap: Optional[float] = Field(None, description="Minimum market cap in millions USD")
    # Maximum market cap filter (in millions USD)
    max_market_cap: Optional[float] = Field(None, description="Maximum market cap in millions USD")


# Pydantic model for screener result
class ScreenerResult(BaseModel):
    """Single company result from screener."""
    # Ticker symbol
    ticker: str = Field(..., description="Stock ticker symbol")
    # Company name
    name: str = Field(..., description="Company name")
    # Industry classification
    industry: Optional[str] = Field(None, description="Industry classification")
    # Current market cap
    market_cap: Optional[float] = Field(None, description="Market capitalization in USD")
    # Metric values that matched filters
    metric_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Metric values that matched filters"
    )
    # Match score (0.0-1.0, higher is better match)
    match_score: float = Field(..., description="Match quality score (0.0-1.0)")


# Pydantic model for screener response
class ScreenerResponse(BaseModel):
    """Response for screener request."""
    # Number of matching results
    total_matches: int = Field(..., description="Total number of matching companies")
    # List of results returned
    results: List[ScreenerResult] = Field(..., description="List of matching companies")
    # Filters applied
    filters_applied: List[ScreenerFilter] = Field(..., description="Filters that were applied")
    # Summary statistics
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


@router.post("/", response_model=ScreenerResponse)
async def screen_companies(
    request: ScreenerRequest = Body(..., description="Screener request"),
) -> ScreenerResponse:
    """
    Screen companies based on financial metrics and filters.
    
    Args:
        request: ScreenerRequest with filters, year, form_type, etc.
    
    Returns:
        ScreenerResponse with matching companies
    
    Raises:
        HTTPException: 400 if invalid parameters, 500 if processing error
    """
    try:
        # Validate filters list (at least 1 filter required)
        if not request.filters or len(request.filters) == 0:
            # Return 400 for empty filters
            raise HTTPException(
                status_code=400,
                detail="At least one filter is required"
            )
        
        # Validate limit parameter (1-1000)
        if request.limit < 1 or request.limit > 1000:
            # Return 400 for invalid limit
            raise HTTPException(
                status_code=400,
                detail="limit parameter must be between 1 and 1000"
            )
        
        # Validate form_type parameter
        if request.form_type not in ["10-K", "10-Q", "20-F"]:
            # Return 400 for invalid form type
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form_type: {request.form_type}"
            )
        
        # Validate market cap filters (if provided)
        if (request.min_market_cap is not None and request.max_market_cap is not None):
            # Check that min is less than max
            if request.min_market_cap > request.max_market_cap:
                # Return 400 for invalid range
                raise HTTPException(
                    status_code=400,
                    detail="min_market_cap must be less than max_market_cap"
                )
        
        # Log request for debugging
        logger.info(f"Running screener with {len(request.filters)} filters, limit={request.limit}")
        
        # Attempt to use screener engine if available
        try:
            # Lazy import to avoid circular dependencies
            from sec_mcp.core.screener import Screener
            
            # Create screener instance
            screener = Screener()
            
            # Run screening with provided filters
            matching_companies = screener.screen(
                filters=request.filters,
                year=request.year,
                form_type=request.form_type,
                sector=request.sector,
                min_market_cap=request.min_market_cap,
                max_market_cap=request.max_market_cap,
                limit=request.limit,
            )
        except (ImportError, AttributeError, Exception) as e:
            # Screener engine not available
            logger.warning(f"Screener engine unavailable: {e}")
            # Return empty results instead of error (graceful degradation)
            matching_companies = []
        
        # Transform matching companies to response format
        results = []
        for company in matching_companies:
            # Create ScreenerResult for each matching company
            result = ScreenerResult(
                ticker=company.get("ticker", ""),
                name=company.get("name", ""),
                industry=company.get("industry"),
                market_cap=company.get("market_cap"),
                metric_values=company.get("metric_values", {}),
                match_score=company.get("match_score", 0.5),
            )
            # Add to results list
            results.append(result)
        
        # Sort results by match score descending (best matches first)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        # Apply limit to results
        results = results[:request.limit]
        
        # Build summary statistics
        summary = {
            "filters_applied": len(request.filters),
            "results_returned": len(results),
            "year": request.year or "latest",
            "form_type": request.form_type,
        }
        
        # Add sector info if provided
        if request.sector:
            summary["sector_filter"] = request.sector
        
        # Add market cap range if provided
        if request.min_market_cap or request.max_market_cap:
            summary["market_cap_range"] = {
                "min": request.min_market_cap,
                "max": request.max_market_cap,
            }
        
        # Return screener response
        return ScreenerResponse(
            total_matches=len(matching_companies),
            results=results,
            filters_applied=request.filters,
            summary=summary,
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error in screener: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Error running screener: {e}", exc_info=True)
        # Return 500 error with generic message
        raise HTTPException(status_code=500, detail="Internal server error")
