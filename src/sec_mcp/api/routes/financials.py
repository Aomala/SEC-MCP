"""
FastAPI router for financial data endpoints.
Provides endpoints for standardized financial statements, historical data, and comparisons.
GET /api/v1/financials/{ticker} -> StandardizedFinancials
GET /api/v1/financials/{ticker}/history -> list of multi-year financials
GET /api/v1/financials/{ticker}/diff -> MetricDiff (year1 vs year2 comparison)
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

# Import financial extraction engine
from sec_mcp.financials import extract_financials, extract_financials_batch
from sec_mcp.models import StandardizedFinancials

# Create logger for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1/financials", tags=["financials"])


# Pydantic model for financial history item
class FinancialHistoryItem(BaseModel):
    """Single year financial snapshot in history response."""
    # Fiscal year end date
    fiscal_year: int = Field(..., description="Fiscal year")
    # Form type (10-K, 10-Q)
    form_type: str = Field(..., description="SEC form type (10-K, 10-Q, etc.)")
    # Filing date in YYYY-MM-DD format
    filing_date: str = Field(..., description="Date filing was submitted")
    # Period end date in YYYY-MM-DD format
    period_end: str = Field(..., description="End date of reporting period")
    # Financial data for this period
    financials: StandardizedFinancials = Field(..., description="Standardized financials")
    # Confidence score for extracted data (0.0-1.0)
    confidence: float = Field(0.9, description="Data extraction confidence score")


# Pydantic model for metric comparison (year1 vs year2)
class MetricDiff(BaseModel):
    """Comparison of a single metric between two periods."""
    # Metric name (e.g., "revenue")
    metric: str = Field(..., description="Metric name")
    # Value in first period
    year1_value: Optional[float] = Field(None, description="Value in first period")
    # Year of first period
    year1: int = Field(..., description="First fiscal year")
    # Value in second period
    year2_value: Optional[float] = Field(None, description="Value in second period")
    # Year of second period
    year2: int = Field(..., description="Second fiscal year")
    # Absolute change (year2 - year1)
    change_absolute: Optional[float] = Field(None, description="Absolute change")
    # Percentage change ((year2 - year1) / year1 * 100)
    change_percent: Optional[float] = Field(None, description="Percentage change")
    # Unit of measurement (e.g., "USD", "shares")
    unit: str = Field(default="USD", description="Unit of measurement")


# Pydantic model for metric comparison response
class ComparisonResult(BaseModel):
    """Response with multiple metric comparisons."""
    # Ticker being analyzed
    ticker: str = Field(..., description="Stock ticker symbol")
    # List of metric comparisons
    metrics: List[MetricDiff] = Field(..., description="List of metric comparisons")
    # Key metrics summary (revenue, net income, etc.)
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary metrics")


@router.get("/{ticker}", response_model=StandardizedFinancials)
async def get_financials(
    ticker: str = Path(..., description="Stock ticker symbol"),
    year: Optional[int] = Query(None, description="Fiscal year (latest if not specified)"),
    form_type: Optional[str] = Query("10-K", description="Form type: 10-K or 10-Q"),
    include_statements: bool = Query(True, description="Include P&L, balance sheet, cash flow"),
    include_segments: bool = Query(False, description="Include segment data if available"),
) -> StandardizedFinancials:
    """
    Get standardized financial statements for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        year: Fiscal year (if not provided, returns latest available)
        form_type: "10-K" for annual, "10-Q" for quarterly
        include_statements: Whether to include P&L, balance sheet, cash flow
        include_segments: Whether to include segment data
    
    Returns:
        StandardizedFinancials with extracted and normalized data
    
    Raises:
        HTTPException: 404 if no financials found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate form_type parameter
        if form_type not in ["10-K", "10-Q", "20-F"]:
            # Return 400 error for invalid form type
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form_type: {form_type}. Must be 10-K, 10-Q, or 20-F"
            )
        
        # Log request with parameters
        logger.info(f"Fetching financials for {ticker_upper}, year={year}, form={form_type}")
        
        # Call financial extraction engine with specified parameters
        financials = extract_financials(
            ticker=ticker_upper,
            year=year,
            form_type=form_type,
            include_statements=include_statements,
            include_segments=include_segments,
        )
        
        # Verify that extraction returned valid data
        if not financials:
            # Log warning and return 404
            logger.warning(f"No financials found for {ticker_upper}, year={year}, form={form_type}")
            raise HTTPException(
                status_code=404,
                detail=f"No {form_type} filings found for {ticker} in year {year or 'latest'}"
            )
        
        # Return extracted and normalized financials
        return financials
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors (e.g., invalid ticker format)
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Error extracting financials for {ticker}: {e}", exc_info=True)
        # Return 500 error with generic message
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{ticker}/history", response_model=List[FinancialHistoryItem])
async def get_financials_history(
    ticker: str = Path(..., description="Stock ticker symbol"),
    years: int = Query(5, description="Number of years to retrieve (default 5)"),
    form_type: Optional[str] = Query("10-K", description="Form type: 10-K or 10-Q"),
) -> List[FinancialHistoryItem]:
    """
    Get multi-year financial history for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        years: Number of years to retrieve (default 5)
        form_type: "10-K" for annual, "10-Q" for quarterly
    
    Returns:
        List of FinancialHistoryItem sorted by fiscal year descending
    
    Raises:
        HTTPException: 404 if no financials found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate years parameter (must be 1-20)
        if years < 1 or years > 20:
            # Return 400 for invalid years
            raise HTTPException(
                status_code=400,
                detail="years parameter must be between 1 and 20"
            )
        
        # Validate form_type parameter
        if form_type not in ["10-K", "10-Q", "20-F"]:
            # Return 400 for invalid form
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form_type: {form_type}. Must be 10-K, 10-Q, or 20-F"
            )
        
        # Log request for debugging
        logger.info(f"Fetching {years} years of {form_type} financials for {ticker_upper}")
        
        # Call batch financial extraction for multiple years
        history_data = extract_financials_batch(
            ticker=ticker_upper,
            years=years,
            form_type=form_type,
        )
        
        # Verify that extraction returned some data
        if not history_data:
            # Log warning and return 404
            logger.warning(f"No financial history found for {ticker_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No {form_type} filings found for {ticker}"
            )
        
        # Transform extracted data to response format
        history_items = []
        for entry in history_data:
            # Create FinancialHistoryItem for each year
            item = FinancialHistoryItem(
                fiscal_year=entry.get("fiscal_year"),
                form_type=entry.get("form_type", form_type),
                filing_date=entry.get("filing_date", ""),
                period_end=entry.get("period_end", ""),
                financials=entry.get("financials"),
                confidence=entry.get("confidence", 0.9),
            )
            # Append to history list
            history_items.append(item)
        
        # Return history sorted by fiscal year (descending)
        return sorted(history_items, key=lambda x: x.fiscal_year, reverse=True)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error fetching financial history for {ticker}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{ticker}/diff", response_model=ComparisonResult)
async def get_financials_diff(
    ticker: str = Path(..., description="Stock ticker symbol"),
    year1: int = Query(..., description="First fiscal year"),
    year2: int = Query(..., description="Second fiscal year"),
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics to compare"),
) -> ComparisonResult:
    """
    Compare financial metrics between two periods.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        year1: First fiscal year
        year2: Second fiscal year
        metrics: Comma-separated metric names (e.g., "revenue,net_income,total_assets")
                If not provided, compares all key metrics
    
    Returns:
        ComparisonResult with metric-by-metric diffs
    
    Raises:
        HTTPException: 404 if data not found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate that year1 and year2 are different
        if year1 == year2:
            # Return 400 for same year comparison
            raise HTTPException(
                status_code=400,
                detail="year1 and year2 must be different"
            )
        
        # Log request for debugging
        logger.info(f"Comparing financials for {ticker_upper}: {year1} vs {year2}")
        
        # Extract financials for both years (10-K annual filings)
        financials_year1 = extract_financials(ticker_upper, year=year1, form_type="10-K")
        financials_year2 = extract_financials(ticker_upper, year=year2, form_type="10-K")
        
        # Verify that both years have data
        if not financials_year1 or not financials_year2:
            # Log warning and return 404
            logger.warning(f"Missing data for {ticker_upper} in year {year1} or {year2}")
            raise HTTPException(
                status_code=404,
                detail=f"Financial data not found for {ticker} in year {year1} or {year2}"
            )
        
        # Parse requested metrics list (if provided, else use defaults)
        if metrics:
            # Split comma-separated metric names
            metric_list = [m.strip() for m in metrics.split(",")]
        else:
            # Use default key metrics for comparison
            metric_list = ["revenue", "net_income", "total_assets", "total_liabilities", "stockholders_equity"]
        
        # Build comparison results for each metric
        comparison_metrics = []
        for metric_name in metric_list:
            # Get metric value from year1 financials
            value_year1 = getattr(financials_year1, metric_name, None) if hasattr(financials_year1, metric_name) else None
            # Get metric value from year2 financials
            value_year2 = getattr(financials_year2, metric_name, None) if hasattr(financials_year2, metric_name) else None
            
            # Calculate absolute change (only if both values exist)
            change_absolute = None
            change_percent = None
            if value_year1 is not None and value_year2 is not None:
                # Compute year2 - year1
                change_absolute = value_year2 - value_year1
                # Compute ((year2 - year1) / year1 * 100), avoid division by zero
                if value_year1 != 0:
                    change_percent = (change_absolute / value_year1) * 100
            
            # Create metric diff record
            diff = MetricDiff(
                metric=metric_name,
                year1_value=value_year1,
                year1=year1,
                year2_value=value_year2,
                year2=year2,
                change_absolute=change_absolute,
                change_percent=change_percent,
                unit="USD",
            )
            # Add to comparison results
            comparison_metrics.append(diff)
        
        # Build summary with key metrics
        summary = {
            "comparison_period": f"{year1} vs {year2}",
            "metrics_compared": len(comparison_metrics),
            "metrics_with_data": len([m for m in comparison_metrics if m.change_absolute is not None]),
        }
        
        # Return comparison result
        return ComparisonResult(
            ticker=ticker_upper,
            metrics=comparison_metrics,
            summary=summary,
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error comparing financials for {ticker}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")
