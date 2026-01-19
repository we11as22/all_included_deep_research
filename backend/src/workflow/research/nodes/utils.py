"""Utility functions for research nodes."""

import asyncio
import structlog
from typing import Any, Dict

from src.workflow.research.nodes import runtime_deps_context, _get_runtime_deps

logger = structlog.get_logger(__name__)


def _restore_runtime_deps(state: Dict[str, Any]) -> Dict[str, Any]:
    """Restore runtime dependencies to state from context variable."""
    deps = _get_runtime_deps()
    for key, value in deps.items():
        # CRITICAL: Always restore stream if it's in deps, even if already in state
        # This ensures stream is never lost
        if value is not None:
            if key == "stream" or key not in state:
                state[key] = value
                if key == "stream":
                    logger.debug("Stream restored to state", has_stream=value is not None)
    return state


async def _save_message_to_db_async(
    stream: Any,
    role: str,
    content: str,
    message_id: str,
    max_retries: int = 3,
) -> bool:
    """
    Save message to database asynchronously with retry logic.
    
    This ensures all assistant messages (deep search, clarification, etc.) are persisted
    even if stream fails or user switches chats.
    """
    if not stream or not hasattr(stream, "app_state"):
        logger.warning("Cannot save message to DB - stream or app_state missing")
        return False
    
    app_state = stream.app_state
    chat_id = app_state.get("chat_id")
    session_factory = app_state.get("session_factory")
    
    if not chat_id or not session_factory:
        logger.warning("Cannot save message to DB - chat_id or session_factory missing", 
                      has_chat_id=bool(chat_id), has_session_factory=bool(session_factory))
        return False
    
    for attempt in range(max_retries):
        try:
            from src.database.schema import ChatMessageModel, ChatModel
            from sqlalchemy import select
            from datetime import datetime
            
            async with session_factory() as session:
                # Verify chat exists
                result = await session.execute(
                    select(ChatModel).where(ChatModel.id == chat_id)
                )
                chat = result.scalar_one_or_none()
                
                if not chat:
                    logger.warning("Chat not found for message save", chat_id=chat_id)
                    return False
                
                # Check if message already exists
                existing_result = await session.execute(
                    select(ChatMessageModel).where(ChatMessageModel.message_id == message_id)
                )
                existing_message = existing_result.scalar_one_or_none()
                
                # CRITICAL: Generate embedding for search functionality
                # This ensures ALL messages (from all modes: chat, web_search, deep_search, deep_research) are searchable
                embedding = None
                if content.strip():
                    try:
                        embedding_provider = app_state.get("embedding_provider")
                        # Fallback: try to get from stream if not in app_state
                        if not embedding_provider and hasattr(stream, "app_state"):
                            stream_app_state = stream.app_state
                            if isinstance(stream_app_state, dict):
                                embedding_provider = stream_app_state.get("embedding_provider")
                            else:
                                embedding_provider = getattr(stream_app_state, "embedding_provider", None)
                        
                        if embedding_provider:
                            embedding_vector = await embedding_provider.embed_text(content)
                            from src.database.schema import EMBEDDING_DIMENSION
                            db_dimension = EMBEDDING_DIMENSION
                            if len(embedding_vector) < db_dimension:
                                embedding_vector = list(embedding_vector) + [0.0] * (db_dimension - len(embedding_vector))
                            elif len(embedding_vector) > db_dimension:
                                embedding_vector = embedding_vector[:db_dimension]
                            embedding = embedding_vector
                            logger.debug("Generated embedding for message", message_id=message_id, embedding_dim=len(embedding_vector))
                        else:
                            logger.warning("No embedding_provider available - message will not be searchable", message_id=message_id)
                    except Exception as e:
                        logger.warning("Failed to generate embedding for message", error=str(e), message_id=message_id, exc_info=True)
                
                if existing_message:
                    # Update existing message
                    existing_message.content = content
                    existing_message.role = role
                    if embedding is not None:
                        existing_message.embedding = embedding
                    chat.updated_at = datetime.now()
                    await session.commit()
                    logger.info("Message updated in DB", message_id=message_id, role=role, content_length=len(content), has_embedding=embedding is not None)
                    return True
                else:
                    # Create new message
                    message = ChatMessageModel(
                        chat_id=chat_id,
                        message_id=message_id,
                        role=role,
                        content=content,
                        embedding=embedding,
                    )
                    session.add(message)
                    chat.updated_at = datetime.now()
                    await session.commit()
                    logger.info("Message saved to DB", message_id=message_id, role=role, content_length=len(content), has_embedding=embedding is not None)
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to save message to DB (attempt {attempt + 1}/{max_retries})", 
                        error=str(e), message_id=message_id, exc_info=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error("Failed to save message to DB after all retries", message_id=message_id)
                return False
    
    return False
