"""Simple API endpoints for web search and deep search (non-streaming)."""

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.chat.service import ChatSearchService

router = APIRouter(prefix="/api/search", tags=["search"])
logger = structlog.get_logger(__name__)


class SearchRequest(BaseModel):
    """Request model for search endpoints."""
    query: str
    chat_history: list[dict] = []


class SearchResponse(BaseModel):
    """Response model for search endpoints."""
    answer: str
    query: str
    mode: str


@router.post("/web", response_model=SearchResponse)
async def web_search(request: SearchRequest, app_request: Request):
    """
    Simple web search endpoint (non-streaming).
    
    Returns search results as a single response without streaming.
    """
    chat_service: ChatSearchService = app_request.app.state.chat_service
    
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    
    try:
        logger.info("Web search API request", query=request.query[:100])
        
        # Create a no-op stream generator for compatibility
        class NoOpStream:
            def emit_status(self, *args, **kwargs):
                pass
            def emit_report_chunk(self, *args, **kwargs):
                pass
            def emit_final_report(self, *args, **kwargs):
                pass
        
        result = await chat_service.answer_web(
            query=request.query,
            stream=NoOpStream(),
            messages=request.chat_history,
        )
        
        logger.info("Web search completed", answer_length=len(result.answer) if result.answer else 0)
        
        return SearchResponse(
            answer=result.answer or "",
            query=request.query,
            mode="web_search"
        )
    except Exception as e:
        logger.error("Web search failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/deep", response_model=SearchResponse)
async def deep_search(request: SearchRequest, app_request: Request):
    """
    Simple deep search endpoint (non-streaming).
    
    Returns deep search results as a single response without streaming.
    """
    chat_service: ChatSearchService = app_request.app.state.chat_service
    
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    
    try:
        logger.info("Deep search API request", query=request.query[:100])
        
        # Create a no-op stream generator for compatibility
        class NoOpStream:
            def emit_status(self, *args, **kwargs):
                pass
            def emit_report_chunk(self, *args, **kwargs):
                pass
            def emit_final_report(self, *args, **kwargs):
                pass
        
        result = await chat_service.answer_deep(
            query=request.query,
            stream=NoOpStream(),
            messages=request.chat_history,
        )
        
        logger.info("Deep search completed", answer_length=len(result.answer) if result.answer else 0)
        
        return SearchResponse(
            answer=result.answer or "",
            query=request.query,
            mode="deep_search"
        )
    except Exception as e:
        logger.error("Deep search failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deep search failed: {str(e)}")
