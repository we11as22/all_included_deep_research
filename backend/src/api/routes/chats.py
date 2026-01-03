"""Chat management endpoints."""

from urllib.parse import unquote
from uuid import uuid4
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from sqlalchemy import select, func

from src.database.schema import ChatModel, ChatMessageModel
from src.utils.text import summarize_text

router = APIRouter(prefix="/api/chats", tags=["chats"])
logger = structlog.get_logger(__name__)


class ChatCreateRequest(BaseModel):
    """Request model for creating a chat."""
    title: str = "New Chat"


@router.get("")
async def list_chats(app_request: Request):
    """List all chats."""
    session_factory = app_request.app.state.session_factory
    
    async with session_factory() as session:
        result = await session.execute(
            select(ChatModel).order_by(ChatModel.updated_at.desc())
        )
        chats = result.scalars().all()
        
        return {
            "chats": [chat.to_dict() for chat in chats]
        }


@router.get("/search")
async def search_chats(
    app_request: Request,
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20, description="Maximum results (top N messages)"),
):
    """
    Search chat messages using hybrid search (vector + fulltext).
    Returns top N most relevant messages with their chat context.
    """
    if not q.strip():
        return {"messages": []}

    chat_message_search_engine = app_request.app.state.chat_message_search_engine

    logger.info("Searching chat messages", query=q, limit=limit)

    try:
        # Search messages using hybrid search
        results = await chat_message_search_engine.search(query=q, limit=limit)

        # Convert results to dict
        messages = [result.to_dict() for result in results]

        return {
            "messages": messages,
            "count": len(messages),
        }
    except Exception as e:
        logger.error("Chat message search failed", error=str(e), query=q)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("")
async def create_chat(chat_request: ChatCreateRequest, app_request: Request):
    """Create a new chat."""
    session_factory = app_request.app.state.session_factory
    
    chat_id = str(uuid4())
    
    try:
        async with session_factory() as session:
            chat = ChatModel(
                id=chat_id,
                title=chat_request.title,
            )
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            
            logger.info("Chat created", chat_id=chat_id, title=chat_request.title)
            return chat.to_dict()
    except Exception as e:
        logger.error("Failed to create chat", error=str(e), title=chat_request.title)
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")


@router.get("/{chat_id}")
async def get_chat(chat_id: str, app_request: Request):
    """Get chat with messages."""
    session_factory = app_request.app.state.session_factory
    
    async with session_factory() as session:
        # Get chat
        result = await session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Get messages
        result = await session.execute(
            select(ChatMessageModel)
            .where(ChatMessageModel.chat_id == chat_id)
            .order_by(ChatMessageModel.created_at.asc())
        )
        messages = result.scalars().all()
        
        return {
            "chat": chat.to_dict(),
            "messages": [msg.to_dict() for msg in messages]
        }


@router.patch("/{chat_id}/title")
async def update_chat_title(chat_id: str, title: str, app_request: Request):
    """Update chat title."""
    session_factory = app_request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        chat = result.scalar_one_or_none()

        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat.title = title
        chat.updated_at = datetime.utcnow()
        await session.commit()

        logger.info("Chat title updated", chat_id=chat_id, title=title)
        return {"id": chat.id, "title": chat.title}


@router.post("/{chat_id}/generate-title")
async def generate_chat_title(chat_id: str, app_request: Request):
    """Auto-generate chat title using LLM based on first messages."""
    session_factory = app_request.app.state.session_factory

    async with session_factory() as session:
        # Get chat with messages
        result = await session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        chat = result.scalar_one_or_none()

        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Get first few messages
        messages_result = await session.execute(
            select(ChatMessageModel)
            .where(ChatMessageModel.chat_id == chat_id)
            .order_by(ChatMessageModel.created_at)
            .limit(4)
        )
        messages = messages_result.scalars().all()

        if not messages:
            return {"id": chat.id, "title": "New Chat"}

        # Build conversation context
        conversation = "\n".join([
            f"{msg.role}: {msg.content[:200]}" for msg in messages
        ])

        # Generate title using LLM
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = app_request.app.state.chat_llm

            # Use structured output for title generation
            from src.models.schemas import ChatTitle
            
            structured_llm = llm.with_structured_output(ChatTitle, method="function_calling")
            title_response = await structured_llm.ainvoke([
                SystemMessage(content="Generate a concise, descriptive title (max 60 characters) for this conversation."),
                HumanMessage(content=conversation)
            ])

            if isinstance(title_response, ChatTitle):
                generated_title = title_response.title.strip().strip('"').strip("'")[:60]
            else:
                # Fallback
                generated_title = title_response.content.strip().strip('"').strip("'")[:60] if hasattr(title_response, "content") else "New Chat"

            # Update chat title
            chat.title = generated_title
            chat.updated_at = datetime.utcnow()
            await session.commit()

            logger.info("Chat title generated", chat_id=chat_id, title=generated_title)
            return {"id": chat.id, "title": chat.title}

        except Exception as e:
            logger.error("Failed to generate title", error=str(e))
            # Fallback: use first user message
            user_msg = next((m for m in messages if m.role == "user"), None)
            if user_msg:
                fallback_title = user_msg.content[:60]
                chat.title = fallback_title
                chat.updated_at = datetime.utcnow()
                await session.commit()
                return {"id": chat.id, "title": fallback_title}

            return {"id": chat.id, "title": "New Chat"}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, app_request: Request):
    """Delete a chat and all its messages."""
    session_factory = app_request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        chat = result.scalar_one_or_none()

        if not chat:
            # Return 200 OK if chat doesn't exist (idempotent delete)
            # This allows bulk delete to continue even if some chats are already deleted
            logger.info("Chat not found (already deleted or never existed)", chat_id=chat_id)
            return {"status": "deleted", "chat_id": chat_id, "note": "Chat not found"}

        # Count messages before deletion for logging
        messages_result = await session.execute(
            select(func.count(ChatMessageModel.id)).where(ChatMessageModel.chat_id == chat_id)
        )
        messages_count = messages_result.scalar() or 0

        # Delete chat from database (cascades to messages via ondelete="CASCADE" in ForeignKey)
        # SQLAlchemy's session.delete() is synchronous, but we need to await commit
        # The cascade="all, delete-orphan" in relationship ensures messages are deleted
        session.delete(chat)
        await session.commit()

        logger.info("Chat deleted successfully", chat_id=chat_id, messages_count=messages_count)
        return {"status": "deleted", "chat_id": chat_id, "messages_deleted": messages_count}


@router.post("/{chat_id}/messages")
async def add_message(
    chat_id: str,
    app_request: Request,
    role: str,
    content: str,
    message_id: str | None = None,
):
    """Add a message to a chat and index it for search."""
    # Decode content if it's URL-encoded
    if content and '%' in content:
        try:
            content = unquote(content)
        except Exception:
            pass
    
    session_factory = app_request.app.state.session_factory
    if role not in {"user", "assistant", "system"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    msg_id = message_id or str(uuid4())
    
    async with session_factory() as session:
        # Verify chat exists
        result = await session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Check if message with this message_id already exists (upsert logic)
        existing_message_result = await session.execute(
            select(ChatMessageModel).where(ChatMessageModel.message_id == msg_id)
        )
        existing_message = existing_message_result.scalar_one_or_none()
        
        if existing_message:
            # Update existing message (in case of retry or duplicate)
            logger.info("Message with message_id already exists, updating", message_id=msg_id, chat_id=chat_id)
            existing_message.content = content
            existing_message.role = role
            # Update embedding if content changed
            if content.strip():
                try:
                    embedding_provider = app_request.app.state.embedding_provider
                    embedding_vector = await embedding_provider.embed_text(content)
                    from src.database.schema import EMBEDDING_DIMENSION
                    db_dimension = EMBEDDING_DIMENSION
                    if len(embedding_vector) < db_dimension:
                        embedding_vector = list(embedding_vector) + [0.0] * (db_dimension - len(embedding_vector))
                    elif len(embedding_vector) > db_dimension:
                        embedding_vector = embedding_vector[:db_dimension]
                    existing_message.embedding = embedding_vector
                except Exception as e:
                    logger.warning("Failed to update embedding for existing message", error=str(e), chat_id=chat_id)
            message = existing_message
        else:
            # Generate embedding for new message content
            embedding = None
            if content.strip():
                try:
                    embedding_provider = app_request.app.state.embedding_provider
                    embedding_vector = await embedding_provider.embed_text(content)

                    # Normalize embedding to database schema dimension (not provider dimension!)
                    # Database schema uses EMBEDDING_DIMENSION from settings/environment
                    from src.database.schema import EMBEDDING_DIMENSION
                    db_dimension = EMBEDDING_DIMENSION
                    
                    if len(embedding_vector) < db_dimension:
                        embedding_vector = list(embedding_vector) + [0.0] * (db_dimension - len(embedding_vector))
                    elif len(embedding_vector) > db_dimension:
                        embedding_vector = embedding_vector[:db_dimension]

                    embedding = embedding_vector
                except Exception as e:
                    logger.warning("Failed to generate embedding for message", error=str(e), chat_id=chat_id)

            # Create new message
            message = ChatMessageModel(
                chat_id=chat_id,
                message_id=msg_id,
                role=role,
                content=content,
                embedding=embedding,
            )
            session.add(message)

        # Update chat updated_at
        chat.updated_at = datetime.now()

        await session.commit()
        await session.refresh(message)

        # Log message addition (content already decoded above)
        logger.info(
            "Message added",
            chat_id=chat_id,
            message_id=msg_id,
            role=role,
            content_preview=content[:100] if content else "",
            has_embedding=embedding is not None
        )
        return message.to_dict()
