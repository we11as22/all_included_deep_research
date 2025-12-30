"""API routes."""

from src.api.routes.chat import router as chat_router
from src.api.routes.chat_stream import router as chat_stream_router
from src.api.routes.chats import router as chats_router
from src.api.routes.config import router as config_router
from src.api.routes.health import router as health_router
from src.api.routes.memory import router as memory_router
from src.api.routes.research import router as research_router

__all__ = [
    "health_router",
    "chat_router",
    "chats_router",
    "research_router",
    "memory_router",
    "config_router",
    "chat_stream_router",
]
