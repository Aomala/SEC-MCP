"""
FastAPI application factory for SEC-MCP REST API.
Mounts all V1 API routes and configures middleware for CORS, logging, etc.
Primary entry point: create_api_app() -> FastAPI instance
"""

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Import all route routers
from sec_mcp.api.routes import companies, financials, filings, compare, screener, chat, export

# Create logger for this module
logger = logging.getLogger(__name__)


def create_api_app(
    title: str = "Fineas.ai API",
    version: str = "2.0",
    description: Optional[str] = None,
    allow_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application with all V1 routes.
    
    Args:
        title: API title (displayed in OpenAPI docs)
        version: API version string
        description: API description
        allow_origins: List of allowed CORS origins (default: ["*"])
    
    Returns:
        Configured FastAPI application instance
    """
    
    # Set default CORS origins if not provided
    if allow_origins is None:
        # Allow all origins by default (update for production)
        allow_origins = ["*"]
    
    # Create FastAPI application instance with metadata
    app = FastAPI(
        title=title,
        version=version,
        description=description or "REST API for SEC filing analysis with XBRL extraction and financial metrics",
        # OpenAPI documentation settings
        docs_url="/api/docs",  # Swagger UI endpoint
        redoc_url="/api/redoc",  # ReDoc endpoint
        openapi_url="/api/openapi.json",  # OpenAPI schema endpoint
    )
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        # List of origins allowed to make cross-origin requests
        allow_origins=allow_origins,
        # Allow credentials (cookies, auth headers) in cross-origin requests
        allow_credentials=True,
        # HTTP methods allowed in cross-origin requests
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        # Headers allowed in cross-origin requests
        allow_headers=["*"],
        # Number of seconds to cache CORS preflight requests
        max_age=600,
    )
    
    # Add trusted host middleware to prevent host header attacks
    app.add_middleware(
        TrustedHostMiddleware,
        # List of trusted hostnames (localhost for development)
        allowed_hosts=["localhost", "127.0.0.1", "*.fineas.ai"],
    )
    
    # Exception handler for validation errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors with detailed error messages."""
        # Log validation error for debugging
        logger.warning(f"Validation error on {request.url.path}: {exc}")
        
        # Return 422 with formatted error details
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "errors": exc.errors(),
            },
        )
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        """
        Health check endpoint for monitoring.
        
        Returns:
            Dictionary with status indicator
        """
        # Return 200 OK with status message
        return {"status": "ok", "version": version}
    
    # Root endpoint with API information
    @app.get("/", tags=["info"])
    async def root() -> dict:
        """
        Root endpoint with API information.
        
        Returns:
            Dictionary with API metadata
        """
        # Return API information
        return {
            "api": title,
            "version": version,
            "docs": "/api/docs",
            "endpoints": {
                "companies": "/api/v1/companies/{ticker}",
                "financials": "/api/v1/financials/{ticker}",
                "filings": "/api/v1/filings/{ticker}",
                "compare": "/api/v1/compare",
                "screener": "/api/v1/screener",
                "chat": "/api/v1/chat",
                "export": "/api/v1/export/{ticker}/csv",
            },
        }
    
    # Log middleware configuration
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Middleware to log HTTP requests and responses.
        
        Args:
            request: HTTP request
            call_next: Callable to process request
        
        Returns:
            HTTP response
        """
        # Extract request details for logging
        method = request.method
        path = request.url.path
        
        # Log incoming request
        logger.info(f"Request: {method} {path}")
        
        # Process request and get response
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unexpected errors during request processing
            logger.error(f"Error processing {method} {path}: {e}", exc_info=True)
            # Re-raise exception for FastAPI to handle
            raise
        
        # Log response status code
        logger.info(f"Response: {method} {path} -> {response.status_code}")
        
        # Return response to client
        return response
    
    # Include companies router (endpoints for company profiles)
    app.include_router(companies.router)
    
    # Include financials router (endpoints for financial data)
    app.include_router(financials.router)
    
    # Include filings router (endpoints for filing metadata and content)
    app.include_router(filings.router)
    
    # Include compare router (endpoints for comparisons and peers)
    app.include_router(compare.router)
    
    # Include screener router (endpoints for financial screening)
    app.include_router(screener.router)
    
    # Include chat router (endpoints for streaming chat)
    app.include_router(chat.router)
    
    # Include export router (endpoints for data export)
    app.include_router(export.router)
    
    # Log that app is fully configured
    logger.info(f"FastAPI app created: {title} v{version} with 7 route groups")
    
    # Return configured application
    return app


# Create default application instance for direct import
# Usage: from sec_mcp.api.app import app
app = create_api_app()


if __name__ == "__main__":
    # This block runs when script is executed directly
    import uvicorn
    
    # Create application with default settings
    api_app = create_api_app()
    
    # Start development server on localhost:8877
    logger.info("Starting development server on http://localhost:8877")
    logger.info("API docs available at http://localhost:8877/api/docs")
    
    # Run uvicorn development server
    uvicorn.run(
        api_app,
        host="0.0.0.0",
        port=8877,
        log_level="info",
    )
