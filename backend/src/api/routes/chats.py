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
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Delete chat from database (cascades to messages)
        await session.delete(chat)
        await session.commit()
        
        logger.info("Chat deleted", chat_id=chat_id)
        return {"status": "deleted", "chat_id": chat_id}


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

        # Generate embedding for message content
        embedding = None
        if content.strip():
            try:
                embedding_provider = app_request.app.state.embedding_provider
                embedding_vector = await embedding_provider.embed_text(content)

                # Normalize embedding to 1536 dimensions (database schema requirement)
                if len(embedding_vector) < 1536:
                    embedding_vector = list(embedding_vector) + [0.0] * (1536 - len(embedding_vector))
                elif len(embedding_vector) > 1536:
                    embedding_vector = embedding_vector[:1536]

                embedding = embedding_vector
            except Exception as e:
                logger.warning("Failed to generate embedding for message", error=str(e), chat_id=chat_id)

        # Create message
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
