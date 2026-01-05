"""Socket.IO streaming generator (replaces SSE)."""

import asyncio
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
        return self._schedule(self._emit('stream:init', {'mode': mode, 'sessionId': self.sid}))

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
        """Emit final report."""
        return self._schedule(self._emit('stream:final_report', {'report': report}))

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
