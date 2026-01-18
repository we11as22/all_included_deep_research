"""Socket.IO streaming generator (replaces SSE)."""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional
import socketio
import structlog

logger = structlog.get_logger(__name__)


class SocketIOStreamingGenerator:
    """Socket.IO streaming generator for real-time progress updates."""

    def __init__(
        self,
        sid: str,
        sio: socketio.AsyncServer,
        message_id: str | None = None,
        chat_id: str | None = None,
        session_id: str | None = None,
        app_state: dict[str, Any] | None = None,
    ):
        """Initialize Socket.IO streaming generator.

        Args:
            sid: Socket.IO session ID
            sio: Socket.IO server instance
            message_id: Assistant message ID for this stream
            chat_id: Chat ID for this stream
            app_state: Optional app state for DB access
        """
        self.sid = sid
        self.sio = sio
        self.message_id = message_id
        self.chat_id = chat_id
        self.session_id = session_id or sid
        self.app_state = app_state or {}
        self._event_count = 0

    def _schedule(self, coro: asyncio.Future) -> asyncio.Task | None:
        try:
            return asyncio.create_task(coro)
        except RuntimeError:
            logger.warning("No running event loop for Socket.IO emit", sid=self.sid)
            return None

    def _with_meta(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data) if data else {}
        if self.message_id:
            payload.setdefault("messageId", self.message_id)
            payload.setdefault("message_id", self.message_id)
        if self.chat_id:
            payload.setdefault("chatId", self.chat_id)
            payload.setdefault("chat_id", self.chat_id)
        if self.session_id:
            payload.setdefault("sessionId", self.session_id)
        return payload

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to client.

        Args:
            event_type: Event type (e.g., 'stream:init', 'stream:status')
            data: Event data
        """
        try:
            self._event_count += 1
            await self.sio.emit(event_type, self._with_meta(data), room=self.sid)
            logger.debug(
                "Emitted Socket.IO event",
                sid=self.sid,
                event_type=event_type,
                event_count=self._event_count,
            )
        except Exception as e:
            logger.error(
                "Failed to emit Socket.IO event",
                sid=self.sid,
                event_type=event_type,
                error=str(e),
            )

    def emit_init(self, mode: str) -> asyncio.Task | None:
        """Emit initialization event."""
        return self._schedule(self._emit('stream:init', {'mode': mode, 'sessionId': self.session_id}))

    def emit_status(self, message: str, step: Optional[str] = None) -> asyncio.Task | None:
        """Emit status update."""
        data = {'message': message}
        if step:
            data['step'] = step
        return self._schedule(self._emit('stream:status', data))

    def emit_memory_search(
        self,
        context_count: Optional[int] = None,
        preview: Optional[List[Dict[str, Any]]] = None,
    ) -> asyncio.Task | None:
        """Emit memory search results."""
        data = {}
        if context_count is not None:
            data['context_count'] = context_count
        if preview:
            data['preview'] = preview
        return self._schedule(self._emit('stream:memory_search', data))

    def emit_search_queries(
        self,
        queries: List[str],
        count: Optional[int] = None,
        label: Optional[str] = None,
    ) -> asyncio.Task | None:
        """Emit generated search queries."""
        data = {'queries': queries}
        if count:
            data['count'] = count
        if label:
            data['label'] = label
        return self._schedule(self._emit('stream:search_queries', data))

    def emit_planning(
        self,
        plan: str | Dict[str, Any],
        topics: Optional[List[str]] = None,
        topic_count: Optional[int] = None,
    ) -> asyncio.Task | None:
        """Emit research plan."""
        if isinstance(plan, dict):
            data = {
                "plan": plan.get("plan") or plan.get("reasoning") or "",
                "topics": plan.get("topics") or [],
                "topic_count": plan.get("topic_count") or plan.get("topicCount"),
            }
        else:
            data = {'plan': plan, 'topics': topics or []}
            if topic_count is not None:
                data['topic_count'] = topic_count
        return self._schedule(self._emit('stream:planning', data))

    def emit_research_start(
        self,
        researcher_id: str | Dict[str, Any],
        topic: Optional[str] = None,
    ) -> asyncio.Task | None:
        """Emit research start event."""
        if isinstance(researcher_id, dict):
            data = {
                "researcher_id": researcher_id.get("researcher_id"),
                "topic": researcher_id.get("topic"),
            }
        else:
            data = {'researcher_id': researcher_id, 'topic': topic}
        return self._schedule(self._emit('stream:research_start', data))

    def emit_source_found(
        self,
        url: Optional[str | Dict[str, Any]] = None,
        title: Optional[str] = None,
        researcher_id: Optional[str] = None,
    ) -> asyncio.Task | None:
        """Emit source found event."""
        if isinstance(url, dict):
            data = {
                "url": url.get("url"),
                "title": url.get("title"),
                "researcher_id": url.get("researcher_id"),
            }
        else:
            data = {}
            if url:
                data['url'] = url
            if title:
                data['title'] = title
            if researcher_id:
                data['researcher_id'] = researcher_id
        return self._schedule(self._emit('stream:source_found', data))

    def emit_source(self, researcher_id: str, source: Dict[str, Any]) -> asyncio.Task | None:
        """Emit source found event (alias for compatibility)."""
        return self.emit_source_found(
            {
                "researcher_id": researcher_id,
                "url": source.get("url"),
                "title": source.get("title"),
            }
        )

    def emit_finding(
        self,
        topic: Optional[str | Dict[str, Any]] = None,
        summary: Optional[str] = None,
        researcher_id: Optional[str] = None,
        findings_count: Optional[int] = None,
    ) -> asyncio.Task | None:
        """Emit research finding."""
        if isinstance(topic, dict):
            data = {
                "topic": topic.get("topic"),
                "summary": topic.get("summary") or topic.get("summary_preview"),
                "researcher_id": topic.get("researcher_id"),
                "findings_count": topic.get("findings_count"),
            }
        else:
            data = {}
            if topic:
                data['topic'] = topic
            if summary:
                data['summary'] = summary
            if researcher_id:
                data['researcher_id'] = researcher_id
            if findings_count:
                data['findings_count'] = findings_count
        return self._schedule(self._emit('stream:finding', data))

    def emit_agent_todo(
        self,
        researcher_id: str,
        todos: List[Dict[str, Any]],
        pending: Optional[int] = None,
        completed: Optional[int] = None,
    ) -> asyncio.Task | None:
        """Emit agent TODO list."""
        data = {'researcher_id': researcher_id, 'todos': todos}
        if pending is not None:
            data['pending'] = pending
        if completed is not None:
            data['completed'] = completed
        return self._schedule(self._emit('stream:agent_todo', data))

    def emit_agent_note(
        self,
        researcher_id: str,
        note: Dict[str, Any],
    ) -> asyncio.Task | None:
        """Emit agent note."""
        data = {'researcher_id': researcher_id, 'note': note}
        return self._schedule(self._emit('stream:agent_note', data))

    def emit_compression(self, message: Optional[str] = None) -> asyncio.Task | None:
        """Emit compression event."""
        data = {}
        if message:
            data['message'] = message
        return self._schedule(self._emit('stream:compression', data))

    def emit_report_chunk(self, chunk: str) -> asyncio.Task | None:
        """Emit report chunk."""
        return self._schedule(self._emit('stream:report_chunk', {'content': chunk}))

    def emit_final_report(self, report: str) -> asyncio.Task | None:
        """Emit final report and save to DB.
        
        CRITICAL: Preserves all formatting including newlines - content is saved as-is.
        """
        # CRITICAL: Preserve all formatting including newlines - don't modify content!
        # Report should already be in markdown format with proper \n characters
        import re
        has_newlines = "\n" in report
        newline_count = report.count("\n")
        has_markdown = bool(re.search(r'^#{2,}\s+', report, re.MULTILINE))
        
        logger.info("Emitting final report via SocketIO", 
                   report_length=len(report),
                   newline_count=newline_count,
                   has_newlines=has_newlines,
                   has_markdown_headings=has_markdown,
                   message_id=self.message_id,
                   chat_id=self.chat_id)
        
        # Emit to client
        emit_task = self._schedule(self._emit('stream:final_report', {
            'report': report,  # CRITICAL: Preserve original content with all \n
            'length': len(report),
            'has_newlines': has_newlines,
            'newline_count': newline_count,
        }))
        
        # CRITICAL: Save final report to DB immediately (same as SSE version)
        # This ensures all assistant messages are persisted for ALL modes (chat, search, deep_search, deep_research)
        asyncio.create_task(self._save_final_message_to_db(report))
        
        return emit_task
    
    async def _save_final_message_to_db(self, content: str) -> None:
        """
        Save final message to database asynchronously.
        
        This is called for all final reports (chat, search, deep_search, deep_research).
        Works for ALL modes that use SocketIOStreamingGenerator.
        
        CRITICAL: Preserves all formatting including newlines - content is saved as-is.
        """
        if not content or not content.strip():
            logger.warning("Cannot save final message to DB - content is empty")
            return
        
        # Get chat_id and session_factory from app_state
        chat_id = self.chat_id or self.app_state.get("chat_id")
        session_factory = self.app_state.get("session_factory")
        
        if not chat_id or not session_factory:
            logger.warning("Cannot save final message to DB - chat_id or session_factory missing",
                          has_chat_id=bool(chat_id), 
                          has_session_factory=bool(session_factory),
                          app_state_keys=list(self.app_state.keys()) if self.app_state else [])
            return
        
        # CRITICAL: Don't strip or modify content - preserve all formatting including \n
        final_content = content
        if not final_content.strip():
            return
        
        # CRITICAL: Log formatting preservation
        import re
        has_newlines = "\n" in final_content
        newline_count = final_content.count("\n")
        has_markdown = bool(re.search(r'^#{2,}\s+', final_content, re.MULTILINE))
        logger.debug("Saving final message with formatting via SocketIO", 
                    content_length=len(final_content),
                    newline_count=newline_count,
                    has_newlines=has_newlines,
                    has_markdown_headings=has_markdown,
                    message_id=self.message_id,
                    chat_id=chat_id)
        
        # Generate unique message_id if not set
        import time
        message_id = self.message_id or f"assistant_{self.session_id}_{int(time.time() * 1000)}"
        
        # Save with retry logic
        max_retries = 3
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
                        logger.warning("Chat not found for final message save", chat_id=chat_id)
                        return
                    
                    # Check if message already exists
                    existing_result = await session.execute(
                        select(ChatMessageModel).where(ChatMessageModel.message_id == message_id)
                    )
                    existing_message = existing_result.scalar_one_or_none()
                    
                    # CRITICAL: Generate embedding for search functionality
                    # This ensures ALL final messages (from all modes) are searchable
                    embedding = None
                    if final_content.strip():
                        try:
                            embedding_provider = self.app_state.get("embedding_provider")
                            if embedding_provider:
                                embedding_vector = await embedding_provider.embed_text(final_content)
                                from src.database.schema import EMBEDDING_DIMENSION
                                db_dimension = EMBEDDING_DIMENSION
                                if len(embedding_vector) < db_dimension:
                                    embedding_vector = list(embedding_vector) + [0.0] * (db_dimension - len(embedding_vector))
                                elif len(embedding_vector) > db_dimension:
                                    embedding_vector = embedding_vector[:db_dimension]
                                embedding = embedding_vector
                                logger.debug("Generated embedding for final message via SocketIO", message_id=message_id, embedding_dim=len(embedding_vector))
                            else:
                                logger.warning("No embedding_provider available - final message will not be searchable", message_id=message_id)
                        except Exception as e:
                            logger.warning("Failed to generate embedding for final message via SocketIO", error=str(e), message_id=message_id, exc_info=True)
                    
                    if existing_message:
                        # Update existing message
                        existing_message.content = final_content  # CRITICAL: Preserve all formatting including \n
                        existing_message.role = "assistant"
                        if embedding is not None:
                            existing_message.embedding = embedding
                        chat.updated_at = datetime.now()
                        await session.commit()
                        logger.info("Final message updated in DB via SocketIO", message_id=message_id, content_length=len(final_content), has_embedding=embedding is not None)
                        return
                    else:
                        # Create new message
                        message = ChatMessageModel(
                            chat_id=chat_id,
                            message_id=message_id,
                            role="assistant",
                            content=final_content,  # CRITICAL: Preserve all formatting including \n
                            embedding=embedding,
                        )
                        session.add(message)
                        chat.updated_at = datetime.now()
                        await session.commit()
                        logger.info("Final message saved to DB via SocketIO", message_id=message_id, content_length=len(final_content), has_embedding=embedding is not None)
                        return
                        
            except Exception as e:
                logger.error(f"Failed to save final message to DB via SocketIO (attempt {attempt + 1}/{max_retries})",
                            error=str(e), message_id=message_id, exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error("Failed to save final message to DB via SocketIO after all retries", message_id=message_id)

    def emit_error(self, error: str, details: Optional[str] = None) -> asyncio.Task | None:
        """Emit error event."""
        data = {'error': error}
        if details:
            data['details'] = details
        return self._schedule(self._emit('stream:error', data))

    def emit_done(self) -> asyncio.Task | None:
        """Emit done event."""
        task = self._schedule(self._emit('stream:done', {}))
        logger.info("Stream completed", sid=self.sid, total_events=self._event_count)
        return task
