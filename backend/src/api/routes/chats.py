"""Chat management endpoints."""

from urllib.parse import unquote
from uuid import uuid4
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.schema import ChatModel, ChatMessageModel, MemoryFileModel
from src.memory.models.search import SearchMode

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
    """Delete a chat and all its messages, including from memory."""
    session_factory = app_request.app.state.session_factory
    memory_manager = app_request.app.state.memory_manager
    
    async with session_factory() as session:
        result = await session.execute(
            select(ChatModel).where(ChatModel.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Delete chat messages from memory
        try:
            # Search for memory files with path pattern: conversations/chat_{chat_id}/...
            result = await session.execute(
                select(MemoryFileModel).where(
                    MemoryFileModel.file_path.like(f"conversations/chat_{chat_id}/%")
                )
            )
            memory_files = result.scalars().all()
            
            # Delete each memory file (this will cascade to chunks)
            for memory_file in memory_files:
                await memory_manager.delete_file(memory_file.file_path)
                logger.info("Deleted chat message from memory", file_path=memory_file.file_path, chat_id=chat_id)
        except Exception as e:
            logger.warning("Failed to delete chat messages from memory", error=str(e), chat_id=chat_id)
            # Continue with database deletion even if memory deletion fails
        
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
    memory_manager = app_request.app.state.memory_manager
    embedding_dimension = app_request.app.state.embedding_dimension
    
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
        
        # Index message for search
        try:
            # Use conversations folder for chat messages to match category detection
            file_path = f"conversations/chat_{chat_id}/message_{msg_id}.md"
            # Format message as markdown
            message_content = f"# {role.capitalize()} Message\n\n{content}"
            await memory_manager.create_file(
                file_path=file_path,
                title=f"{chat.title} - {role} message",
                content=message_content,
            )
            await memory_manager.sync_file_to_db(
                file_path, 
                embedding_dimension=embedding_dimension
            )
            logger.info("Message indexed for search", chat_id=chat_id, message_id=msg_id)
        except Exception as e:
            logger.warning("Failed to index message", chat_id=chat_id, message_id=msg_id, error=str(e))
            # Don't fail the request if indexing fails
        
        # Log message addition (content already decoded above)
        logger.info(
            "Message added", 
            chat_id=chat_id, 
            message_id=msg_id, 
            role=role,
            content_preview=content[:100] if content else ""
        )
        return message.to_dict()


@router.get("/search")
async def search_chats(
    app_request: Request,
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
):
    """Search chats by message content using hybrid search."""
    if not q.strip():
        return {"chats": []}
    
    session_factory = app_request.app.state.session_factory
    search_engine = app_request.app.state.search_engine
    
    logger.info("Searching chats", query=q, limit=limit)
    
    async with session_factory() as session:
        # Use hybrid search to find relevant message chunks
        # Filter by category="chat" to only search chat messages
        search_results = await search_engine.search(
            query=q,
            search_mode=SearchMode.HYBRID,
            limit=limit * 3,  # Get more results to filter by chat
            category_filter="chat",
        )
        
        logger.info("Search results", count=len(search_results), query=q)
        
        # Extract unique chat IDs from search results
        chat_ids = set()
        chat_scores = {}
        chat_previews = {}
        
        for result in search_results:
            # Check if result is from a chat message
            # File path format: conversations/chat_{chat_id}/message_{message_id}.md
            if result.file_path:
                logger.debug("Search result", file_path=result.file_path, score=result.score)
                # Handle both old format (chat_{id}/...) and new format (conversations/chat_{id}/...)
                if "chat_" in result.file_path:
                    parts = result.file_path.split("/")
                    # Find the part that starts with chat_
                    for part in parts:
                        if part.startswith("chat_"):
                            chat_id = part.replace("chat_", "")
                            if chat_id:
                                chat_ids.add(chat_id)
                                # Track best score for each chat
                                if chat_id not in chat_scores or result.score > chat_scores[chat_id]:
                                    chat_scores[chat_id] = result.score
                                # Store preview from best matching message
                                if chat_id not in chat_previews:
                                    preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
                                    chat_previews[chat_id] = preview
                            break
        
        if not chat_ids:
            return {"chats": []}
        
        # Get chats and order by relevance score
        result = await session.execute(
            select(ChatModel).where(ChatModel.id.in_(chat_ids))
        )
        chats = result.scalars().all()
        
        # Sort by score and get preview
        sorted_chats = []
        for chat in chats:
            score = chat_scores.get(chat.id, 0.0)
            preview = chat_previews.get(chat.id, None)
            
            chat_dict = chat.to_dict()
            chat_dict["metadata"] = {"score": score, "preview": preview}
            sorted_chats.append((score, chat_dict))
        
        # Sort by score descending
        sorted_chats.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        return {
            "chats": [chat_dict for _, chat_dict in sorted_chats[:limit]]
        }

