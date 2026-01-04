"""SSE streaming for deep research workflows."""

import asyncio
import json
import time
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _debug_payload(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > 2000:
            return {"length": len(value), "preview": value[:500]}
        return value
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            output[key] = _debug_payload(item)
        return output
    if isinstance(value, list):
        return [_debug_payload(item) for item in value[:20]]
    return value


class StreamEventType(str, Enum):
    """Types of events that can be streamed."""

    INIT = "init"
    STATUS = "status"
    MEMORY_SEARCH = "memory_search"
    SEARCH_QUERIES = "search_queries"
    PLANNING = "planning"
    RESEARCH_START = "research_start"
    RESEARCH_TOPIC = "research_topic"
    SOURCE_FOUND = "source_found"
    FINDING = "finding"
    AGENT_TODO = "agent_todo"
    AGENT_NOTE = "agent_note"
    COMPRESSION = "compression"
    REPORT_CHUNK = "report_chunk"
    FINAL_REPORT = "final_report"
    ERROR = "error"
    DONE = "done"

    # NEW: LangGraph deep research events
    GRAPH_STATE_UPDATE = "graph_state_update"
    SUPERVISOR_REACT = "supervisor_react"
    SUPERVISOR_DIRECTIVE = "supervisor_directive"
    AGENT_ACTION = "agent_action"
    AGENT_REASONING = "agent_reasoning"
    REPLAN = "replan"
    GAP_IDENTIFIED = "gap_identified"
    DEBUG = "debug"


class StreamingGenerator:
    """Base streaming generator with async queue."""

    def __init__(self, app_state: dict[str, Any] | None = None):
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._finished = False
        self.app_state = app_state or {}
        # Store all sent events for reconnection (keep last 1000 events)
        self._event_history: list[str] = []
        self._max_history = 1000

    def add(self, data: str) -> None:
        """Add data to stream."""
        if not self._finished:
            try:
                self.queue.put_nowait(data)
                # Store in history for reconnection
                self._event_history.append(data)
                # Keep only last N events to avoid memory issues
                if len(self._event_history) > self._max_history:
                    self._event_history = self._event_history[-self._max_history:]
            except asyncio.QueueFull:
                # Queue is full - this shouldn't happen with unbounded queue, but handle it
                logger.warning("Stream queue is full, dropping event")

    def finish(self) -> None:
        """Signal stream completion."""
        if not self._finished:
            self._finished = True
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                # Shouldn't happen, but handle it
                pass

    async def stream(self, replay_history: bool = False):
        """
        Async generator for streaming data.
        
        Args:
            replay_history: If True, replay all stored events first (for reconnection)
        """
        try:
            # If reconnecting, replay all stored events first
            if replay_history and self._event_history:
                logger.info("Replaying event history for reconnection", events_count=len(self._event_history))
                for event in self._event_history:
                    yield event
            
            # Then stream new events
            while True:
                data = await self.queue.get()
                if data is None:
                    break
                yield data
        except Exception as e:
            logger.error("Stream generator error", error=str(e), exc_info=True)
            # Try to send error event before closing (only if ResearchStreamingGenerator)
            try:
                if hasattr(self, '_create_event'):
                    error_event = self._create_event(StreamEventType.ERROR, {"error": f"Stream error: {str(e)}", "details": "Stream generator encountered an error"})
                    yield error_event
            except:
                pass


class OpenAIStreamingGenerator(StreamingGenerator):
    """OpenAI-compatible streaming generator."""

    def __init__(self, model: str = "all-included-deep-research"):
        super().__init__()
        self.model = model
        self.fingerprint = f"fp_{hex(hash(model))[-8:]}"
        self.id = f"chatcmpl-{int(time.time())}{hash(str(time.time()))}"[:29]
        self.created = int(time.time())
        self.choice_index = 0

    def add_chunk_from_str(self, content: str) -> None:
        """Add text chunk in OpenAI format."""
        response = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint,
            "choices": [
                {
                    "delta": {"content": content, "role": "assistant", "tool_calls": None},
                    "index": self.choice_index,
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "usage": None,
        }
        self.add(f"data: {json.dumps(response)}\n\n")

    def finish(self, content: str | None = None, finish_reason: str = "stop") -> None:
        """Finish stream with final chunk."""
        final_response = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint,
            "choices": [
                {
                    "index": self.choice_index,
                    "delta": {"content": content or "", "role": "assistant", "tool_calls": None},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self.add(f"data: {json.dumps(final_response)}\n\n")
        self.add("data: [DONE]\n\n")
        super().finish()


class ResearchStreamingGenerator(StreamingGenerator):
    """Research-specific streaming with structured events."""

    def __init__(self, session_id: str | None = None, app_state: dict[str, Any] | None = None):
        super().__init__(app_state=app_state)
        self.session_id = session_id or f"research_{int(time.time())}"

    def _create_event(self, event_type: StreamEventType, data: Any) -> str:
        """Create SSE event with metadata."""
        event = {
            "session_id": self.session_id,
            "type": event_type.value,
            "timestamp": time.time(),
            "data": data,
        }
        if self.app_state.get("debug_mode"):
            logger.info(
                "stream_event",
                session_id=self.session_id,
                event_type=event_type.value,
                data=_debug_payload(data),
            )
        return f"data: {json.dumps(event)}\n\n"

    def emit_init(self, mode: str) -> None:
        """Emit initialization event."""
        self.add(self._create_event(StreamEventType.INIT, {"mode": mode, "session_id": self.session_id}))

    def emit_status(self, message: str, step: str | None = None) -> None:
        """Emit status update."""
        self.add(self._create_event(StreamEventType.STATUS, {"message": message, "step": step}))

    def emit_memory_context(self, context: list[dict]) -> None:
        """Emit memory search results."""
        self.add(
            self._create_event(
                StreamEventType.MEMORY_SEARCH, {"context_count": len(context), "preview": context[:3] if context else []}
            )
        )

    def emit_search_queries(self, queries: list[str], label: str | None = None) -> None:
        """Emit search queries."""
        payload = {"queries": queries, "count": len(queries)}
        if label:
            payload["label"] = label
        self.add(self._create_event(StreamEventType.SEARCH_QUERIES, payload))

    def emit_research_plan(self, plan: str, topics: list[str]) -> None:
        """Emit research plan."""
        self.add(
            self._create_event(StreamEventType.PLANNING, {"plan": plan, "topics": topics, "topic_count": len(topics)})
        )

    def emit_planning(self, data: dict) -> None:
        """Emit planning event (alias for backward compatibility)."""
        topics = data.get("topics", [])
        reasoning = data.get("reasoning", "")
        self.add(
            self._create_event(StreamEventType.PLANNING, {"reasoning": reasoning, "topics": topics, "topic_count": len(topics)})
        )

    def emit_research_start(self, data: dict) -> None:
        """Emit researcher start event."""
        researcher_id = data.get("researcher_id", "")
        topic = data.get("topic", "")
        self.add(self._create_event(StreamEventType.RESEARCH_START, {"researcher_id": researcher_id, "topic": topic}))

    def emit_source(self, researcher_id: str, source: dict) -> None:
        """Emit source found event."""
        self.add(
            self._create_event(
                StreamEventType.SOURCE_FOUND,
                {"researcher_id": researcher_id, "url": source.get("url"), "title": source.get("title")},
            )
        )

    def emit_source_found(self, data: dict) -> None:
        """Emit source found event (alias for backward compatibility)."""
        researcher_id = data.get("researcher_id", "")
        self.add(
            self._create_event(
                StreamEventType.SOURCE_FOUND,
                {"researcher_id": researcher_id, "url": data.get("url"), "title": data.get("title")},
            )
        )

    def emit_finding(self, data: dict) -> None:
        """Emit research finding."""
        researcher_id = data.get("researcher_id", "")
        topic = data.get("topic", "")
        summary = data.get("summary", "")
        key_findings = data.get("key_findings", [])
        summary_preview = summary[:240] + "..." if isinstance(summary, str) and len(summary) > 240 else summary
        self.add(
            self._create_event(
                StreamEventType.FINDING,
                {
                    "researcher_id": researcher_id,
                    "topic": topic,
                    "summary": summary,
                    "summary_preview": summary_preview,
                    "findings_count": len(key_findings),
                },
            )
        )

    def emit_supervisor_react(self, data: dict) -> None:
        """Emit supervisor reaction event."""
        self.add(
            self._create_event(
                StreamEventType.SUPERVISOR_REACT,
                {
                    "reasoning": data.get("reasoning", ""),
                    "should_continue": data.get("should_continue", False),
                    "gaps": data.get("gaps", []),
                },
            )
        )

    def emit_agent_todo(self, researcher_id: str, todos: list[dict]) -> None:
        """Emit agent todo list update."""
        pending = sum(1 for item in todos if item.get("status") != "done")
        completed = sum(1 for item in todos if item.get("status") == "done")
        self.add(
            self._create_event(
                StreamEventType.AGENT_TODO,
                {
                    "researcher_id": researcher_id,
                    "todos": todos,
                    "pending": pending,
                    "completed": completed,
                },
            )
        )

    def emit_agent_note(self, researcher_id: str, note: dict) -> None:
        """Emit agent note update."""
        summary = note.get("summary", "")
        summary_preview = summary[:240] + "..." if isinstance(summary, str) and len(summary) > 240 else summary
        payload = {**note, "summary_preview": summary_preview}
        self.add(
            self._create_event(
                StreamEventType.AGENT_NOTE,
                {"researcher_id": researcher_id, "note": payload},
            )
        )

    def emit_compression(self, data: dict | str) -> None:
        """Emit compression event."""
        if isinstance(data, str):
            compressed_text = data
        else:
            compressed_text = data.get("message", "")
        self.add(
            self._create_event(
                StreamEventType.COMPRESSION,
                {"preview": compressed_text[:300] + "..." if len(compressed_text) > 300 else compressed_text},
            )
        )

    def emit_report_chunk(self, chunk: str) -> None:
        """Emit report generation chunk."""
        self.add(self._create_event(StreamEventType.REPORT_CHUNK, {"content": chunk}))

    def emit_final_report(self, report: str) -> None:
        """Emit final report."""
        self.add(
            self._create_event(
                StreamEventType.FINAL_REPORT,
                {
                    "report": report,
                    "length": len(report),
                    "preview": report[:500] + "..." if len(report) > 500 else report,
                },
            )
        )

    def emit_error(self, error: str, details: str | None = None) -> None:
        """Emit error event."""
        logger.error("Research stream error", error=error, details=details)
        self.add(self._create_event(StreamEventType.ERROR, {"error": error, "details": details}))

    def emit_done(self) -> None:
        """Emit completion and finish stream."""
        self.add(self._create_event(StreamEventType.DONE, {}))
        self.finish()
