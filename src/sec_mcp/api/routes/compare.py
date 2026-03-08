"""
FastAPI router for comparison endpoints.
Provides endpoints for comparing multiple companies and finding peer companies.
POST /api/v1/compare -> ComparisonResult (multiple companies side-by-side)
GET /api/v1/peers/{ticker} -> list of peer tickers with relevance scores
"""

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

# Import financial extraction and comparison engines
from sec_mcp.financials import extract_financials_batch
from sec_mcp.models import StandardizedFinancials

# Create logger for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1", tags=["comparison"])


# Pydantic model for comparison request body
class ComparisonRequest(BaseModel):
    """Request body for multi-company comparison."""
    # List of stock tickers to compare
    tickers: List[str] = Field(..., description="List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])")
    # Fiscal year to compare (if None, uses latest available)
    year: Optional[int] = Field(None, description="Fiscal year (latest if not specified)")
    # Form type (10-K or 10-Q)
    form_type: str = Field(default="10-K", description="Form type: 10-K or 10-Q")
    # Metrics to include in comparison
    metrics: Optional[List[str]] = Field(
        None,
        description="List of metrics to include (e.g., ['revenue', 'net_income', 'roe'])"
    )


# Pydantic model for peer company with relevance score
class PeerCompany(BaseModel):
    """A peer company with relevance scoring."""
    # Ticker symbol
    ticker: str = Field(..., description="Stock ticker symbol")
    # Company name
    name: str = Field(..., description="Company name")
    # Industry classification
    industry: Optional[str] = Field(None, description="Industry classification")
    # Relevance score (0.0-1.0, where 1.0 is most relevant)
    relevance_score: float = Field(..., description="Relevance score (0.0-1.0)")
    # Reason for peer selection
    reason: str = Field(..., description="Why this company is a peer")
    # Market cap (for sizing comparison)
    market_cap: Optional[float] = Field(None, description="Market capitalization")


# Pydantic model for comparison result
class ComparisonMetric(BaseModel):
    """Single metric comparison across companies."""
    # Metric name (e.g., "revenue")
    metric: str = Field(..., description="Metric name")
    # Unit of measurement (e.g., "USD", "shares", "%")
    unit: str = Field(default="USD", description="Unit of measurement")
    # Dictionary of ticker -> value for each company
    values: Dict[str, Optional[float]] = Field(..., description="Ticker -> metric value mapping")
    # Order of companies for display
    tickers_in_order: List[str] = Field(..., description="Tickers in display order")


# Pydantic model for multi-company comparison response
class MultiComparisonResult(BaseModel):
    """Response for multi-company financial comparison."""
    # Tickers compared
    tickers: List[str] = Field(..., description="List of compared tickers")
    # Fiscal year compared
    year: int = Field(..., description="Fiscal year")
    # Form type (10-K, 10-Q)
    form_type: str = Field(..., description="Form type used")
    # List of metric comparisons
    metrics: List[ComparisonMetric] = Field(..., description="List of metric comparisons")
    # Summary statistics
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


@router.post("/compare", response_model=MultiComparisonResult)
async def compare_companies(
    request: ComparisonRequest = Body(..., description="Comparison request"),
) -> MultiComparisonResult:
    """
    Compare financial metrics across multiple companies.
    
    Args:
        request: ComparisonRequest with tickers, year, form_type, and optional metrics
    
    Returns:
        MultiComparisonResult with side-by-side metric comparison
    
    Raises:
        HTTPException: 404 if data not found, 400 if invalid parameters
    """
    try:
        # Normalize tickers to uppercase
        tickers_upper = [t.upper() for t in request.tickers]
        
        # Validate ticker list (1-10 companies for performance)
        if len(tickers_upper) < 1 or len(tickers_upper) > 10:
            # Return 400 for invalid ticker count
            raise HTTPException(
                status_code=400,
                detail="Must provide 1 to 10 tickers for comparison"
            )
        
        # Remove duplicates from ticker list
        tickers_upper = list(set(tickers_upper))
        
        # Validate form_type parameter
        if request.form_type not in ["10-K", "10-Q", "20-F"]:
            # Return 400 for invalid form type
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form_type: {request.form_type}"
            )
        
        # Log request for debugging
        logger.info(f"Comparing companies: {tickers_upper}, year={request.year}, form={request.form_type}")
        
        # Extract financials for all tickers using batch operation
        financials_batch = extract_financials_batch(
            tickers=tickers_upper,
            year=request.year,
            form_type=request.form_type,
        )
        
        # Verify that we got data for at least one company
        if not financials_batch:
            # Log warning and return 404
            logger.warning(f"No financial data found for any of: {tickers_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No financial data found for {', '.join(tickers_upper)}"
            )
        
        # Determine which metrics to compare
        if request.metrics:
            # Use user-specified metrics
            metrics_to_compare = request.metrics
        else:
            # Use default key metrics for comparison
            metrics_to_compare = [
                "revenue",
                "net_income",
                "total_assets",
                "total_liabilities",
                "stockholders_equity",
                "operating_cash_flow",
            ]
        
        # Build comparison for each metric
        comparison_metrics = []
        for metric_name in metrics_to_compare:
            # Initialize values dict for this metric across all tickers
            metric_values = {}
            
            # Iterate through each company's financials
            for ticker, financials in financials_batch.items():
                # Get metric value from financials object
                value = None
                if financials and hasattr(financials, metric_name):
                    # Extract metric value from StandardizedFinancials
                    value = getattr(financials, metric_name, None)
                
                # Store value in dict (ticker -> value)
                metric_values[ticker] = value
            
            # Create ComparisonMetric for this metric
            comparison_metric = ComparisonMetric(
                metric=metric_name,
                unit="USD",
                values=metric_values,
                tickers_in_order=tickers_upper,
            )
            # Add to comparison results
            comparison_metrics.append(comparison_metric)
        
        # Build summary statistics
        summary = {
            "companies_compared": len(financials_batch),
            "metrics_compared": len(comparison_metrics),
            "metrics_with_data": len([m for m in comparison_metrics if any(m.values.values())]),
        }
        
        # Determine fiscal year from first available data
        fiscal_year = request.year or 2024  # Default to current year if not specified
        
        # Return comparison result
        return MultiComparisonResult(
            tickers=tickers_upper,
            year=fiscal_year,
            form_type=request.form_type,
            metrics=comparison_metrics,
            summary=summary,
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error in comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Error comparing companies: {e}", exc_info=True)
        # Return 500 error with generic message
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/peers/{ticker}", response_model=List[PeerCompany])
async def get_peer_companies(
    ticker: str = Path(..., description="Stock ticker symbol"),
    limit: int = Query(10, description="Maximum number of peers to return (1-50)"),
) -> List[PeerCompany]:
    """
    Get peer companies for a given ticker.
    
    Peer identification is based on:
    1. Industry classification matching
    2. Market cap proximity
    3. Business similarity
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        limit: Maximum number of peers to return (default 10, max 50)
    
    Returns:
        List of PeerCompany sorted by relevance_score descending
    
    Raises:
        HTTPException: 404 if company not found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate limit parameter (1-50)
        if limit < 1 or limit > 50:
            # Return 400 for invalid limit
            raise HTTPException(
                status_code=400,
                detail="limit parameter must be between 1 and 50"
            )
        
        # Log request for debugging
        logger.info(f"Finding peers for {ticker_upper}, limit={limit}")
        
        # Attempt to use peer engine if available
        try:
            # Lazy import to avoid circular dependencies
            from sec_mcp.core.peer_engine import PeerEngine
            
            # Create peer engine instance
            peer_engine = PeerEngine()
            
            # Get peer companies with relevance scoring
            peers_data = peer_engine.find_peers(
                ticker=ticker_upper,
                limit=limit,
            )
        except (ImportError, AttributeError, Exception) as e:
            # Peer engine not available, return basic industry-based peers
            logger.warning(f"Peer engine unavailable for {ticker_upper}: {e}")
            peers_data = []
        
        # Verify that peers were found
        if not peers_data:
            # Log warning and return 404
            logger.warning(f"No peers found for {ticker_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No peer companies found for {ticker}"
            )
        
        # Transform peer data to response format
        peer_companies = []
        for peer in peers_data:
            # Create PeerCompany for each peer
            peer_company = PeerCompany(
                ticker=peer.get("ticker", ""),
                name=peer.get("name", ""),
                industry=peer.get("industry"),
                relevance_score=peer.get("relevance_score", 0.5),
                reason=peer.get("reason", "Industry classification match"),
                market_cap=peer.get("market_cap"),
            )
            # Add to peer list
            peer_companies.append(peer_company)
        
        # Sort by relevance score descending
        peer_companies.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Return top N peers
        return peer_companies[:limit]
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error finding peers for {ticker}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")
