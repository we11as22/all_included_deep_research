"""FastAPI application initialization and configuration."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import unquote, parse_qs, urlencode

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import get_settings
from src.config.logging_config import configure_logging
from src.chat.service import ChatSearchService
from src.chat.search import ChatMessageSearchEngine
from src.database.connection import create_database_engine, create_db_pool, create_session_factory
from src.embeddings.factory import create_embedding_provider
from src.memory.hybrid_search import HybridSearchEngine
from src.memory.manager import MemoryManager
from src.llm.factory import create_chat_model

# Import routers
from src.api.routes import (
    chat_router,
    chat_stream_router,
    chats_router,
    config_router,
    health_router,
    memory_router,
)

logger = structlog.get_logger(__name__)


class URLDecodeMiddleware(BaseHTTPMiddleware):
    """Middleware to decode URL-encoded query parameters for logging."""
    
    async def dispatch(self, request: Request, call_next):
        # Decode query parameters for logging
        if request.query_params:
            decoded_params = {}
            for key, value in request.query_params.multi_items():
                try:
                    decoded_key = unquote(key)
                    decoded_value = unquote(value)
                    decoded_params[decoded_key] = decoded_value
                except Exception:
                    decoded_params[key] = value
            
            # Log decoded URL if it contains non-ASCII characters
            if any(ord(c) > 127 for c in str(request.url)):
                decoded_path = unquote(str(request.url.path))
                decoded_query = "&".join([f"{k}={v}" for k, v in decoded_params.items()])
                logger.debug(
                    "Request URL decoded",
                    method=request.method,
                    path=decoded_path,
                    query=decoded_query if decoded_query else None,
                )
        
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()

    # Configure logging FIRST - this removes verbose tracebacks
    configure_logging(debug_mode=settings.debug_mode)

    logger.info("Starting up All-Included Deep Research API...")
    logger.info("Settings loaded", debug_mode=settings.debug_mode)

    # Initialize database
    logger.info("Initializing database connection...")
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)
    db_pool = await create_db_pool(settings)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.db_pool = db_pool

    # Initialize embedding provider
    logger.info("Initializing embedding provider...", provider=settings.embedding_provider)
    embedding_provider = create_embedding_provider(settings)
    embedding_dimension = embedding_provider.get_dimension()
    logger.info("Embedding dimension detected", dimension=embedding_dimension, provider=settings.embedding_provider)
    app.state.embedding_provider = embedding_provider
    app.state.embedding_dimension = embedding_dimension

    # Initialize hybrid search engine
    logger.info("Initializing hybrid search engine...")
    search_engine = HybridSearchEngine(
        db_pool=db_pool,
        embedding_provider=embedding_provider,
        rrf_k=settings.rrf_k,
    )
    app.state.search_engine = search_engine

    # Initialize chat message search engine
    logger.info("Initializing chat message search engine...")
    chat_message_search_engine = ChatMessageSearchEngine(
        db_pool=db_pool,
        embedding_provider=embedding_provider,
        rrf_k=settings.rrf_k,
    )
    app.state.chat_message_search_engine = chat_message_search_engine

    # Initialize memory manager
    logger.info("Initializing memory manager...")
    memory_manager = MemoryManager(
        memory_dir=settings.memory_dir,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
    )
    app.state.memory_manager = memory_manager

    # Initialize LLMs for research workflows
    logger.info("Initializing research LLMs...")
    research_llm = create_chat_model(
        settings.research_model,
        settings,
        max_tokens=settings.research_model_max_tokens,
        temperature=0.7,
    )
    compression_llm = create_chat_model(
        settings.compression_model,
        settings,
        max_tokens=settings.compression_model_max_tokens,
        temperature=0.3,
    )
    final_report_llm = create_chat_model(
        settings.final_report_model,
        settings,
        max_tokens=settings.final_report_model_max_tokens,
        temperature=0.7,
    )
    app.state.research_llm = research_llm
    app.state.compression_llm = compression_llm
    app.state.final_report_llm = final_report_llm

    # Initialize chat LLM for general chat operations
    logger.info("Initializing chat LLM...")
    chat_llm = create_chat_model(
        settings.chat_model,
        settings,
        max_tokens=settings.chat_model_max_tokens,
        temperature=0.7,
    )
    app.state.chat_llm = chat_llm

    # Initialize chat search service
    logger.info("Initializing chat search service...")
    chat_service = ChatSearchService(
        settings=settings,
        search_engine=search_engine,
        embedding_provider=embedding_provider,
    )
    app.state.chat_service = chat_service
    app.state.settings = settings
    
    # Initialize active tasks storage for cancellation
    app.state.active_tasks: dict[str, asyncio.Task] = {}
    # Initialize active streams storage for reconnection
    app.state.active_streams: dict[str, Any] = {}

    logger.info(
        "All-Included Deep Research API started successfully",
        available_modes=["chat", "search", "deep_search", "deep_research"],
    )

    yield

    # Shutdown
    logger.info("Shutting down All-Included Deep Research API...")

    # Cleanup database connections
    if hasattr(app.state, "engine"):
        await app.state.engine.dispose()
    if hasattr(app.state, "db_pool"):
        await app.state.db_pool.close()

    logger.info("All-Included Deep Research API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="All-Included Deep Research API",
        description="Comprehensive deep research system with memory integration",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add URL decode middleware for better logging
    app.add_middleware(URLDecodeMiddleware)

    # Include routers
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(chat_stream_router)
    app.include_router(chats_router)
    app.include_router(memory_router)
    app.include_router(config_router)

    logger.info("FastAPI app created")

    return app


# Create app instance
_fastapi_app = create_app()

# Wrap with Socket.IO
try:
    import socketio
    from src.api.socketio_server import get_sio

    sio = get_sio()

    # Create ASGI app that combines FastAPI and Socket.IO
    app = socketio.ASGIApp(
        socketio_server=sio,
        other_asgi_app=_fastapi_app,
        socketio_path='/socket.io',
    )

    logger.info("Socket.IO integration enabled")

    # Add helper functions to access app state
    def get_app_state():
        """Get FastAPI app state."""
        return _fastapi_app.state

    def get_session_factory():
        """Get database session factory."""
        return _fastapi_app.state.session_factory

    # Make these available globally
    import src.main
    src.main.get_app_state = get_app_state
    src.main.get_session_factory = get_session_factory

except ImportError as e:
    logger.warning("Socket.IO not available, running without Socket.IO support", error=str(e))
    app = _fastapi_app
