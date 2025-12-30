"""Chat management endpoints."""

from uuid import uuid4
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.schema import ChatModel, ChatMessageModel

router = APIRouter(prefix="/api/chats", tags=["chats"])
logger = structlog.get_logger(__name__)


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


@router.post("")
async def create_chat(app_request: Request, title: str = "New Chat"):
    """Create a new chat."""
    session_factory = app_request.app.state.session_factory
    
    chat_id = str(uuid4())
    
    async with session_factory() as session:
        chat = ChatModel(
            id=chat_id,
            title=title,
        )
        session.add(chat)
        await session.commit()
        await session.refresh(chat)
        
        logger.info("Chat created", chat_id=chat_id, title=title)
        return chat.to_dict()


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
    """Add a message to a chat."""
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
        
        # Create message
        message = ChatMessageModel(
            chat_id=chat_id,
            message_id=msg_id,
            role=role,
            content=content,
        )
        session.add(message)
        
        # Update chat updated_at
        chat.updated_at = datetime.now()
        
        await session.commit()
        await session.refresh(message)
        
        logger.info("Message added", chat_id=chat_id, message_id=msg_id, role=role)
        return message.to_dict()

