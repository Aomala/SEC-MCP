"""
FastAPI router for streaming chat endpoints.
Provides Server-Sent Events (SSE) endpoint for interactive financial analysis.
POST /api/v1/chat -> Server-Sent Events stream with thinking, tokens, and results
"""

import logging
import json
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import intent parser and financial extraction
# These would import from sec_mcp modules in production
logger = logging.getLogger(__name__)

# Create FastAPI router with API v1 prefix and tag for organization
router = APIRouter(prefix="/api/v1", tags=["chat"])


# Pydantic model for chat request
class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    # User's natural language question
    message: str = Field(..., description="User's question or request")
    # Optional company ticker for context
    ticker: str = Field(default="", description="Stock ticker for context (optional)")
    # Optional additional context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (e.g., {'year': 2024, 'form_type': '10-K'})"
    )
    # Optional conversation history for multi-turn
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history (list of {'role': 'user'/'assistant', 'content': '...'} dicts)"
    )


# Pydantic model for SSE chat event
class ChatEvent(BaseModel):
    """Single SSE event in chat stream."""
    # Event type (thinking, token, done, error)
    event_type: str = Field(..., description="Event type: thinking, token, done, or error")
    # Event content/data
    content: str = Field(default="", description="Event content")
    # Optional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., tool calls, token count)"
    )


def _format_sse_event(event: ChatEvent) -> str:
    """
    Format a ChatEvent as SSE (Server-Sent Events) format.
    
    Args:
        event: ChatEvent to format
    
    Returns:
        String in SSE format: "data: {json}\n\n"
    """
    # Convert event to JSON
    event_json = json.dumps({
        "type": event.event_type,
        "content": event.content,
        "metadata": event.metadata,
    })
    # Return in SSE format (data: prefix, double newline suffix)
    return f"data: {event_json}\n\n"


async def _stream_chat_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Generate streaming chat response as SSE events.
    
    Args:
        request: ChatRequest with message and context
    
    Yields:
        SSE formatted event strings
    """
    try:
        # Yield thinking event to indicate processing started
        thinking_event = ChatEvent(
            event_type="thinking",
            content="Parsing your question and gathering context...",
        )
        yield _format_sse_event(thinking_event)
        
        # Add small delay to allow client to receive thinking event
        await asyncio.sleep(0.1)
        
        # Attempt to import intent parser (may not be available)
        try:
            # Lazy import to avoid circular dependencies
            from sec_mcp.intent_parser import parse_intent
            
            # Parse user's intent (what are they asking?)
            intent_result = parse_intent(request.message)
        except (ImportError, AttributeError, Exception) as e:
            # Intent parser not available, assume generic financial query
            logger.warning(f"Intent parser unavailable: {e}")
            intent_result = {"intent": "general", "entities": []}
        
        # Yield thinking event with parsed intent
        thinking_event = ChatEvent(
            event_type="thinking",
            content=f"Intent detected: {intent_result.get('intent', 'general')}",
            metadata={"entities": intent_result.get("entities", [])},
        )
        yield _format_sse_event(thinking_event)
        
        # Add delay before processing
        await asyncio.sleep(0.1)
        
        # Handle different intents
        intent_type = intent_result.get("intent", "general")
        
        if intent_type == "financials" and request.ticker:
            # User is asking about financial data
            try:
                # Lazy import to avoid circular dependencies
                from sec_mcp.financials import extract_financials
                
                # Yield thinking event
                thinking_event = ChatEvent(
                    event_type="thinking",
                    content=f"Extracting financials for {request.ticker}...",
                )
                yield _format_sse_event(thinking_event)
                
                # Extract financials from SEC filings
                financials = extract_financials(
                    ticker=request.ticker,
                    year=request.context.get("year"),
                    form_type=request.context.get("form_type", "10-K"),
                )
                
                # Build response text from financials
                if financials:
                    # Format financials as readable text
                    response_text = f"Found financials for {request.ticker}:\n"
                    # Add key metrics to response
                    if hasattr(financials, "revenue"):
                        response_text += f"- Revenue: ${financials.revenue:,.0f}\n"
                    if hasattr(financials, "net_income"):
                        response_text += f"- Net Income: ${financials.net_income:,.0f}\n"
                    if hasattr(financials, "total_assets"):
                        response_text += f"- Total Assets: ${financials.total_assets:,.0f}\n"
                else:
                    # No financials found
                    response_text = f"No financials found for {request.ticker}"
                
                # Yield token events (split response into tokens for streaming)
                words = response_text.split()
                for i, word in enumerate(words):
                    # Yield token event for each word
                    token_event = ChatEvent(
                        event_type="token",
                        content=word + " ",
                        metadata={"token_index": i},
                    )
                    yield _format_sse_event(token_event)
                    # Add small delay for streaming effect
                    await asyncio.sleep(0.01)
            
            except Exception as e:
                # Log error during financial extraction
                logger.error(f"Error extracting financials: {e}", exc_info=True)
                # Yield error event
                error_event = ChatEvent(
                    event_type="error",
                    content=f"Error retrieving financials: {str(e)}",
                )
                yield _format_sse_event(error_event)
        
        else:
            # Generic response for non-financial queries
            # In production, would call Claude API for LLM response
            response_text = f"I received your question: '{request.message}'"
            
            # Check if Claude API is available
            try:
                # Lazy import to avoid circular dependency
                from sec_mcp.narrator import call_claude
                
                # Call Claude to generate response (non-streaming for now)
                claude_response = call_claude(
                    message=request.message,
                    context={"ticker": request.ticker} if request.ticker else {},
                    history=request.history,
                )
                
                # Use Claude's response if available
                response_text = claude_response or response_text
            
            except (ImportError, AttributeError, Exception) as e:
                # Claude API not available, use default response
                logger.warning(f"Claude API unavailable: {e}")
                # Keep response_text as is
            
            # Yield token events for response
            words = response_text.split()
            for i, word in enumerate(words):
                # Yield each word as token event
                token_event = ChatEvent(
                    event_type="token",
                    content=word + " ",
                    metadata={"token_index": i},
                )
                yield _format_sse_event(token_event)
                # Add small delay for streaming effect
                await asyncio.sleep(0.01)
        
        # Yield done event to signal completion
        done_event = ChatEvent(
            event_type="done",
            content="Response complete",
            metadata={"timestamp": str(__import__('datetime').datetime.now().isoformat())},
        )
        yield _format_sse_event(done_event)
    
    except Exception as e:
        # Log unexpected error
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        # Yield error event
        error_event = ChatEvent(
            event_type="error",
            content=f"Stream error: {str(e)}",
        )
        yield _format_sse_event(error_event)


@router.post("/chat")
async def chat_stream(
    request: ChatRequest = Body(..., description="Chat request"),
) -> StreamingResponse:
    """
    Stream chat response as Server-Sent Events.
    
    Args:
        request: ChatRequest with message and optional context
    
    Returns:
        StreamingResponse with SSE formatted events
    
    Raises:
        HTTPException: 400 if invalid request
    """
    try:
        # Validate message (non-empty)
        if not request.message or len(request.message.strip()) == 0:
            # Return 400 for empty message
            raise HTTPException(
                status_code=400,
                detail="message parameter is required and cannot be empty"
            )
        
        # Validate message length (max 1000 characters)
        if len(request.message) > 1000:
            # Return 400 for message too long
            raise HTTPException(
                status_code=400,
                detail="message exceeds maximum length of 1000 characters"
            )
        
        # Validate ticker if provided
        if request.ticker and len(request.ticker) > 10:
            # Return 400 for invalid ticker
            raise HTTPException(
                status_code=400,
                detail="ticker must be 10 characters or less"
            )
        
        # Log chat request
        logger.info(f"Chat request: {request.message[:50]}... (ticker: {request.ticker})")
        
        # Create streaming response with SSE media type
        return StreamingResponse(
            _stream_chat_response(request),
            media_type="text/event-stream",
            headers={
                # Required for SSE
                "Cache-Control": "no-cache",
                # Allow cross-origin requests for web clients
                "X-Accel-Buffering": "no",
            },
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Return 400 for validation errors
        logger.warning(f"Validation error in chat: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        # Return 500 error
        raise HTTPException(status_code=500, detail="Internal server error")
