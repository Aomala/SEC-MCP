"""
FastAPI router for data export endpoints.
Provides endpoints for exporting financial data in various formats.
GET /api/v1/export/{ticker}/csv -> CSV file download
GET /api/v1/export/{ticker}/json -> JSON file download
"""

import logging
import json
import csv
from io import StringIO
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Import financial extraction engine
from sec_mcp.financials import extract_financials

# Create logger for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1/export", tags=["export"])


def _financials_to_csv(ticker: str, financials: any) -> str:
    """
    Convert StandardizedFinancials object to CSV format.
    
    Args:
        ticker: Stock ticker symbol
        financials: StandardizedFinancials object
    
    Returns:
        CSV formatted string with metrics and values
    """
    # Create StringIO buffer for CSV output
    output = StringIO()
    
    # Create CSV writer instance
    writer = csv.writer(output)
    
    # Write CSV header row (Metric, Value)
    writer.writerow(["Metric", "Value", "Unit"])
    
    # Iterate through financial attributes
    if financials:
        # Get all attributes from financials object
        for attr_name in dir(financials):
            # Skip private/magic attributes (start with _)
            if attr_name.startswith("_"):
                continue
            
            # Skip methods (only include properties/fields)
            attr_value = getattr(financials, attr_name, None)
            if callable(attr_value):
                continue
            
            # Skip None values (data not available)
            if attr_value is None:
                continue
            
            # Write row: metric name, value, unit
            writer.writerow([attr_name, attr_value, "USD"])
    
    # Get CSV string from buffer
    csv_string = output.getvalue()
    
    # Close buffer
    output.close()
    
    # Return CSV formatted string
    return csv_string


def _financials_to_json(ticker: str, financials: any) -> str:
    """
    Convert StandardizedFinancials object to JSON format.
    
    Args:
        ticker: Stock ticker symbol
        financials: StandardizedFinancials object
    
    Returns:
        JSON formatted string
    """
    # Build JSON-serializable dict from financials
    data_dict = {
        "ticker": ticker,
        "financials": {},
    }
    
    # Iterate through financial attributes
    if financials:
        # Get all attributes from financials object
        for attr_name in dir(financials):
            # Skip private/magic attributes (start with _)
            if attr_name.startswith("_"):
                continue
            
            # Skip methods (only include properties/fields)
            attr_value = getattr(financials, attr_name, None)
            if callable(attr_value):
                continue
            
            # Add attribute to dict
            data_dict["financials"][attr_name] = attr_value
    
    # Convert dict to JSON string with indentation for readability
    json_string = json.dumps(data_dict, indent=2, default=str)
    
    # Return JSON string
    return json_string


@router.get("/{ticker}/csv")
async def export_csv(
    ticker: str = Path(..., description="Stock ticker symbol"),
    year: Optional[int] = Query(None, description="Fiscal year (latest if not specified)"),
    form_type: str = Query("10-K", description="Form type: 10-K or 10-Q"),
) -> StreamingResponse:
    """
    Export financial data as CSV file.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        year: Fiscal year (if not provided, returns latest available)
        form_type: "10-K" for annual, "10-Q" for quarterly
    
    Returns:
        StreamingResponse with CSV file download
    
    Raises:
        HTTPException: 404 if financials not found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate form_type parameter
        if form_type not in ["10-K", "10-Q", "20-F"]:
            # Return 400 for invalid form type
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form_type: {form_type}"
            )
        
        # Log request for debugging
        logger.info(f"Exporting CSV for {ticker_upper}, year={year}, form={form_type}")
        
        # Extract financials from SEC filings
        financials = extract_financials(
            ticker=ticker_upper,
            year=year,
            form_type=form_type,
        )
        
        # Verify that extraction returned valid data
        if not financials:
            # Log warning and return 404
            logger.warning(f"No financials found for {ticker_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No {form_type} filings found for {ticker}"
            )
        
        # Convert financials to CSV format
        csv_content = _financials_to_csv(ticker_upper, financials)
        
        # Generate filename with ticker and year
        filename = f"{ticker_upper}_financials"
        if year:
            filename += f"_{year}"
        filename += ".csv"
        
        # Return streaming response with CSV content
        return StreamingResponse(
            # Create async generator that yields CSV content
            (chunk for chunk in [csv_content.encode()]),
            media_type="text/csv",
            headers={
                # Set Content-Disposition header for file download
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Error exporting CSV for {ticker}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{ticker}/json")
async def export_json(
    ticker: str = Path(..., description="Stock ticker symbol"),
    year: Optional[int] = Query(None, description="Fiscal year (latest if not specified)"),
    form_type: str = Query("10-K", description="Form type: 10-K or 10-Q"),
) -> StreamingResponse:
    """
    Export financial data as JSON file.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        year: Fiscal year (if not provided, returns latest available)
        form_type: "10-K" for annual, "10-Q" for quarterly
    
    Returns:
        StreamingResponse with JSON file download
    
    Raises:
        HTTPException: 404 if financials not found, 400 if invalid parameters
    """
    try:
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Validate form_type parameter
        if form_type not in ["10-K", "10-Q", "20-F"]:
            # Return 400 for invalid form type
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form_type: {form_type}"
            )
        
        # Log request for debugging
        logger.info(f"Exporting JSON for {ticker_upper}, year={year}, form={form_type}")
        
        # Extract financials from SEC filings
        financials = extract_financials(
            ticker=ticker_upper,
            year=year,
            form_type=form_type,
        )
        
        # Verify that extraction returned valid data
        if not financials:
            # Log warning and return 404
            logger.warning(f"No financials found for {ticker_upper}")
            raise HTTPException(
                status_code=404,
                detail=f"No {form_type} filings found for {ticker}"
            )
        
        # Convert financials to JSON format
        json_content = _financials_to_json(ticker_upper, financials)
        
        # Generate filename with ticker and year
        filename = f"{ticker_upper}_financials"
        if year:
            filename += f"_{year}"
        filename += ".json"
        
        # Return streaming response with JSON content
        return StreamingResponse(
            # Create async generator that yields JSON content
            (chunk for chunk in [json_content.encode()]),
            media_type="application/json",
            headers={
                # Set Content-Disposition header for file download
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Error exporting JSON for {ticker}: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")
