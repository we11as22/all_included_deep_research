"""LangGraph nodes for deep research workflow with structured outputs.

Each node is an async function that takes state and returns state updates.
All enhanced nodes use structured outputs with reasoning fields.
"""

import asyncio
import contextvars
import structlog
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

# IMPORTANT: Import runtime_deps_context from nodes/__init__.py to ensure we use THE SAME context variable
# Previously this file had its own context variable which was never set, causing llm=None errors!
from src.workflow.research.nodes import runtime_deps_context


def _get_runtime_deps() -> Dict[str, Any]:
    """Get runtime dependencies from context variable."""
    deps = runtime_deps_context.get()
    return deps or {}


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

from src.workflow.research.state import (
    ResearchState,
    CompressedFindings,
)
from src.workflow.research.models import (
    QueryAnalysis,
    ResearchPlan,
    ResearchTopic,
    AgentCharacteristics,
    AgentCharacteristic,
    AgentTodo,
    SupervisorAssessment,
    AgentDirective,
    FinalReport,
    ReportSection,
    ReportValidation,
    ClarificationNeeds,
)
from src.workflow.research.supervisor_queue import SupervisorQueue
from src.workflow.research.researcher import run_researcher_agent_enhanced
from src.workflow.research.supervisor_agent import run_supervisor_agent
from src.models.agent_models import AgentTodoItem

logger = structlog.get_logger(__name__)


# ==================== Helper Functions ====================

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


# ==================== Memory Search Node ====================

async def search_memory_node(state: ResearchState) -> Dict:
    """Search vector memory for relevant context."""
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state["query"]
    stream = state.get("stream")

    if stream:
        stream.emit_status("Searching memory...", step="memory")

    # TODO: Implement actual memory search when vector store is integrated
    # For now, return empty context
    memory_results = []

    logger.info("Memory search completed", results=len(memory_results))

    return {
        "memory_context": {"type": "override", "value": memory_results},
    }


# ==================== Deep Search Node ====================

async def run_deep_search_node(state: ResearchState) -> Dict:
    """
    Run deep search to gather initial context before planning.
    
    CRITICAL: If deep_search_result already exists and user has answered clarification,
    skip deep search and return existing result. This prevents re-running deep search
    after user answers clarification questions.
    """
    query = state["query"]
    chat_history = state.get("chat_history", [])
    
    # CRITICAL: Check if deep search was already done and user answered clarification
    # Proper sequence: deep search -> clarification questions -> user answer -> skip deep search
    deep_search_done = False
    deep_search_index = -1
    clarification_index = -1
    has_user_answer = False
    found_clarification = False
    
    # Step 1: Find deep search result in chat history
    for i, msg in enumerate(chat_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            # Check if this message contains deep search result
            if "initial deep search context" in content or "deep search completed" in content:
                deep_search_done = True
                deep_search_index = i
                logger.info("Found deep search result in chat history", message_index=i)
                break
    
    # Step 2: Find clarification questions (should come AFTER deep search OR in the same message)
    if deep_search_done:
        # CRITICAL: Check the same message first (in case deep search and clarification are combined)
        if deep_search_index >= 0:
            msg = chat_history[deep_search_index]
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                # Check if clarification is also in the same message (combined message)
                if "clarification" in content or "🔍" in content or "clarify" in content:
                    found_clarification = True
                    clarification_index = deep_search_index
                    logger.info("Found clarification questions in same message as deep search (combined message)", 
                               message_index=deep_search_index, 
                               note="Deep search and clarification are in one message")
                    # Step 3: Check if user answered (next message should be from user)
                    if deep_search_index + 1 < len(chat_history) and chat_history[deep_search_index + 1].get("role") == "user":
                        has_user_answer = True
                        logger.info("User answered clarification questions", 
                                   answer_preview=chat_history[deep_search_index + 1].get("content", "")[:100])
        
        # Also check messages AFTER deep search (in case they were sent separately)
        if not found_clarification:
            for i in range(deep_search_index + 1, len(chat_history)):
                msg = chat_history[i]
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    if "clarification" in content or "🔍" in content or "clarify" in content:
                        found_clarification = True
                        clarification_index = i
                        logger.info("Found clarification questions in chat history", message_index=i, after_deep_search=True)
                        # Step 3: Check if user answered (next message should be from user)
                        if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                            has_user_answer = True
                            logger.info("User answered clarification questions", answer_preview=chat_history[i + 1].get("content", "")[:100])
                            break
    
    # If deep search was done AND clarification was asked AND user answered, skip deep search
    if deep_search_done and found_clarification and has_user_answer:
        logger.info("CRITICAL: Skipping deep search - already completed, clarification asked, and user answered. Proceeding directly to research.")
        # CRITICAL: Return the existing deep_search_result from state if available, otherwise return empty
        # This ensures the result is preserved for the rest of the workflow
        existing_result = state.get("deep_search_result", "")
        if existing_result and not (isinstance(existing_result, dict) and existing_result.get("type") == "override" and existing_result.get("value") == ""):
            logger.info("Using existing deep_search_result from state for continuation")
            return {"deep_search_result": existing_result}
        else:
            # Return empty result - clarify node will use existing context, then proceed to research
            return {"deep_search_result": {"type": "override", "value": ""}}
    
    # Also check if deep_search_result already exists in state (from checkpoint)
    # This is CRITICAL for continuation after clarification
    deep_search_result_raw = state.get("deep_search_result", "")
    if deep_search_result_raw:
        # Extract actual value if it's a dict with "type": "override"
        if isinstance(deep_search_result_raw, dict) and deep_search_result_raw.get("type") == "override":
            actual_result = deep_search_result_raw.get("value", "")
        else:
            actual_result = deep_search_result_raw
        
        # CRITICAL: If deep_search_result exists in state (from checkpoint), check if user answered clarification
        # ONLY skip if user actually answered - otherwise we might loop
        if actual_result or (isinstance(deep_search_result_raw, dict) and deep_search_result_raw.get("type") == "override"):
            # CRITICAL: Only skip if user answered clarification - check both chat_history and state
            # If user hasn't answered, we should NOT skip - let the graph wait for user input
            if has_user_answer:
                logger.info("CRITICAL: Skipping deep search - result exists in state (from checkpoint) and user answered clarification. Proceeding directly to research.")
                return {"deep_search_result": deep_search_result_raw}
            elif clarification_index >= 0:
                # Clarification was asked but user hasn't answered yet
                # CRITICAL: Return existing result to avoid re-running deep search, but don't skip node
                # The graph will continue to clarify node, which will stop and wait for user input
                logger.info("CRITICAL: Deep search result exists but user hasn't answered clarification yet. Returning existing result - graph will stop at clarify node and wait for user.")
                # Return existing result - this prevents re-running deep search
                # Graph will proceed to clarify node, which will detect no answer and stop (END)
                # CRITICAL: Return empty override to signal that we should NOT re-run deep search
                # but also NOT proceed to research (wait for user answer)
                return {"deep_search_result": {"type": "override", "value": ""}}
    
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    stream = state.get("stream")

    # CRITICAL: Always try to restore stream from context if missing
    if not stream:
        logger.warning("Stream missing in state, restoring from runtime deps", state_keys=list(state.keys()))
        deps = _get_runtime_deps()
        stream = deps.get("stream")
        if stream:
            state["stream"] = stream
            logger.info("Stream restored from runtime deps context")
        else:
            logger.error("CRITICAL: stream is None even after restoring from context!", deps_keys=list(deps.keys()) if deps else "no deps")

    if stream:
        try:
            stream.emit_status("Running deep search for initial context...", step="deep_search")
            logger.info("✅ Progress emitted: Running deep search")
        except Exception as e:
            logger.error("Failed to emit progress", error=str(e), exc_info=True)
    else:
        logger.error("❌ Cannot emit progress - stream is None!", node="run_deep_search_node")

    logger.info("Running deep search before agent spawning")

    try:
        # Import search service
        from src.workflow.search.service import SearchService

        # Get dependencies from state
        llm = state.get("llm")
        search_provider = state.get("search_provider")
        scraper = state.get("scraper")

        if not all([llm, search_provider, scraper]):
            logger.warning("Missing dependencies for deep search, skipping")
            return {"deep_search_result": {"type": "override", "value": None}}

        # CRITICAL: Log max_tokens to verify it's correct
        max_tokens_value = None
        if hasattr(llm, "max_tokens"):
            max_tokens_value = llm.max_tokens
        logger.info(
            "Creating SearchService for deep search",
            llm_type=type(llm).__name__,
            max_tokens=max_tokens_value,
            has_max_tokens=hasattr(llm, "max_tokens"),
        )

        # Create search service
        search_service = SearchService(
            classifier_llm=llm,
            research_llm=llm,
            writer_llm=llm,
            search_provider=search_provider,
            scraper=scraper,
        )

        # Run deep search
        result = await search_service._answer_deep_search(
            query=query,
            classification=None,  # Will be created internally
            stream=stream,
            chat_history=chat_history,
        )

        logger.info("Deep search completed", answer_length=len(result) if result else 0)

        # Emit deep search result to frontend via stream - FULL RESULT, NO TRUNCATION
        if stream and result:
            try:
                stream.emit_status(f"Deep search completed. Found {len(result)} characters of context.", step="deep_search")
                # Send FULL result - no truncation! User needs to see complete deep search answer
                # Send in smaller chunks to ensure smooth streaming and avoid blocking
                # CRITICAL: End with \n\n\n\n (4 newlines = 2 empty lines) to ensure proper separation when clarification questions are sent separately
                # This ensures that when clarification is sent as next chunk, there will be 2 empty lines between them
                # Deep search ends: \n\n\n\n, clarification starts: (no prefix needed), result: 2 empty lines separation
                normalized_result = result.rstrip()  # Remove trailing whitespace from result
                full_message = f"## Initial Deep Search Context\n\n{normalized_result}\n\n---\n\n*This context will be used to guide the research agents.*\n\n\n\n"
                
                # Always send in chunks for smooth streaming (chunk size 10000 chars)
                chunk_size = 10000
                chunks = [full_message[i:i+chunk_size] for i in range(0, len(full_message), chunk_size)]
                logger.info("Sending deep search result in chunks", total_length=len(full_message), chunks_count=len(chunks))
                
                # CRITICAL: Ensure last chunk ends with proper separation (2 empty lines = 4 newlines)
                # If last chunk doesn't end with \n\n\n\n, add it to ensure proper separation with clarification
                # This is ONLY for deep research mode - other modes don't need this
                if chunks:
                    last_chunk = chunks[-1]
                    # Check if last chunk ends with exactly 4 newlines (2 empty lines)
                    trailing_newlines = len(last_chunk) - len(last_chunk.rstrip('\n'))
                    if trailing_newlines < 4:
                        # Add enough newlines to make it 4 total (2 empty lines)
                        chunks[-1] = last_chunk.rstrip('\n') + '\n' * 4
                        logger.info("Added trailing newlines to last deep search chunk for proper separation",
                                   added_newlines=4-trailing_newlines,
                                   note="Deep research mode only - ensures 2 empty lines before clarification")
                    elif trailing_newlines > 4:
                        # Too many newlines - normalize to exactly 4
                        chunks[-1] = last_chunk.rstrip('\n') + '\n' * 4
                        logger.info("Normalized trailing newlines in last deep search chunk",
                                   original_newlines=trailing_newlines,
                                   note="Normalized to exactly 4 newlines (2 empty lines)")
                
                for i, chunk in enumerate(chunks):
                    stream.emit_report_chunk(chunk)
                    # Small delay between chunks to ensure smooth streaming
                    if i < len(chunks) - 1:  # Don't sleep after last chunk
                        await asyncio.sleep(0.03)  # Small delay between chunks
                
                # CRITICAL: Add extra delay after last deep search chunk to ensure frontend processes it
                # before clarification chunk arrives - this helps with proper markdown rendering
                await asyncio.sleep(0.5)
                
                # CRITICAL: Verify last chunk ends with proper separation
                if chunks:
                    last_chunk = chunks[-1]
                    trailing_newlines_count = len(last_chunk) - len(last_chunk.rstrip('\n'))
                    last_chars = last_chunk[-30:] if len(last_chunk) >= 30 else last_chunk
                    logger.info("Deep search FULL result emitted to stream", 
                               result_length=len(result), 
                               chunks_sent=len(chunks),
                               last_chunk_length=len(last_chunk),
                               trailing_newlines=trailing_newlines_count,
                               last_chars_preview=repr(last_chars),
                               note="Last chunk should end with \\n\\n\\n\\n (4 newlines) for proper separation")
                else:
                    logger.warning("No chunks to send for deep search!", result_length=len(result))
                
                # CRITICAL: Do NOT save deep search separately - it will be saved together with clarification questions
                # This ensures deep search and questions are in the same message and won't be lost on page reload
                logger.info("Deep search result emitted - will be saved together with clarification questions",
                           result_length=len(result),
                           note="Deep search will be combined with clarification in one message")
                
                # CRITICAL: Longer delay to ensure all chunks are fully processed by frontend before sending clarification
                # This ensures proper separation between deep search and clarification
                # CRITICAL: This delay is ONLY for deep research mode - other modes don't need it
                # CRITICAL: Increased delay to 1.0 seconds to ensure frontend fully processes and renders all deep search chunks
                await asyncio.sleep(1.0)  # Increased delay to ensure frontend processes all chunks and renders them
                logger.info("Delay completed before sending clarification questions",
                           delay_seconds=1.0,
                           note="Ensures frontend has fully processed and rendered all deep search chunks before clarification")
            except Exception as e:
                logger.error("Failed to emit deep search result to stream", error=str(e), exc_info=True)
                # Try to emit at least a status message
                try:
                    stream.emit_status(f"Deep search completed ({len(result)} chars). Proceeding with research...", step="deep_search")
                    await asyncio.sleep(0.1)
                except Exception as e2:
                    logger.error("Failed to emit status after deep search", error=str(e2))

        return {
            "deep_search_result": {"type": "override", "value": result},
        }

    except Exception as e:
        logger.error("Deep search failed", error=str(e), exc_info=True)
        return {"deep_search_result": {"type": "override", "value": None}}


# ==================== Clarifying Questions Node ====================

async def clarify_with_user_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask clarifying questions to user if needed before starting research.
    
    This helps narrow down research scope and ensure we understand the query correctly.
    """
    query = state.get("query", "")
    # Get deep_search_result - handle both dict and string formats
    # CRITICAL: We need the actual deep search result text to combine with clarification
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        # If it's a dict with "type": "override", extract the value
        deep_search_result_text = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result_text = deep_search_result_raw or ""
    
    # Also try to get from research_session if available
    session_id = state.get("session_id")
    # CRITICAL: Get stream BEFORE trying to access it in session retrieval
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    settings = state.get("settings")
    
    # CRITICAL: Now that stream is available, try to get deep search from session
    if not deep_search_result_text and session_id and stream:
        try:
            from src.workflow.research.session.manager import SessionManager
            session_factory = stream.app_state.get("session_factory") if stream else None
            if session_factory:
                session_manager = SessionManager(session_factory)
                session_data = await session_manager.get_session(session_id)
                if session_data and session_data.get("deep_search_result"):
                    deep_search_result_text = session_data.get("deep_search_result")
                    logger.info("Retrieved deep search result from session", 
                               session_id=session_id,
                               result_length=len(deep_search_result_text) if deep_search_result_text else 0)
        except Exception as e:
            logger.warning("Failed to get deep search from session", error=str(e))
    
    chat_history = state.get("chat_history", [])
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    settings = state.get("settings")
    
    # CRITICAL: Check if LLM is available
    if not llm:
        logger.error("CRITICAL: llm is None in clarify_with_user_node!", state_keys=list(state.keys()))
        deps = _get_runtime_deps()
        llm = deps.get("llm")
        if llm:
            state["llm"] = llm
            logger.info("LLM restored from runtime deps context")
        else:
            logger.error("CRITICAL: llm is None even after restoring from context! Will use fallback questions immediately")
    
    # CRITICAL: Log if stream is missing and try to restore from context
    if not stream:
        logger.error("CRITICAL: stream is None in clarify_with_user_node!", state_keys=list(state.keys()))
        deps = _get_runtime_deps()
        stream = deps.get("stream")
        if stream:
            state["stream"] = stream
            logger.info("Stream restored from runtime deps context")
        else:
            logger.error("CRITICAL: stream is None even after restoring from context!")
    
    # ALWAYS generate clarifying questions about the research topic and approach
    # This helps ensure we understand what the user wants to research
    # Only skip if user has already answered in recent chat history
    has_recent_clarification = False
    if len(chat_history) >= 2:
        # Check if last assistant message contains clarification questions
        last_assistant_msg = None
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break
        if last_assistant_msg and ("clarification" in last_assistant_msg.lower() or "clarify" in last_assistant_msg.lower() or "🔍" in last_assistant_msg):
            # Check if user responded (next message is from user)
            if len(chat_history) >= 3:
                # Check if there's a user message after the clarification
                found_clarification = False
                for i, msg in enumerate(chat_history):
                    if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                        # Check if next message is from user
                        if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                            has_recent_clarification = True
                            logger.info("User has already answered clarification questions")
                            break
    
    if has_recent_clarification:
        logger.info("Clarification already provided, skipping")
        return {"clarification_needed": False}
    
    if stream:
        stream.emit_status("Analyzing if clarification is needed...", step="clarification")
        logger.info("Progress emitted: Analyzing clarification")
    else:
        logger.warning("Cannot emit progress - stream is None!", node="clarify_with_user_node")
        logger.info("Clarification node: deep_search_result available", 
                   has_result=bool(deep_search_result), 
                   result_length=len(deep_search_result) if deep_search_result else 0,
                   result_preview=deep_search_result[:200] if deep_search_result else "")
    
    # CRITICAL: Use query from state - it already contains the correct original query
    # The query in state is set in chat_stream.py with proper logic to find the first user message
    # in the current deep research session (before any deep search/clarification markers)
    # This ensures we always use the correct original query, not a clarification answer or message from other modes
    
    # Verify that query is not empty and log it for debugging
    if not query:
        logger.error("CRITICAL: query is empty in state! Cannot generate clarification questions.")
        return {"clarification_needed": False}
    
    user_message_for_context = query
    logger.info(
        "Using query from state for clarification questions",
        query_preview=user_message_for_context[:100],
        query_length=len(user_message_for_context),
        note="This query is the original user message from the start of current deep research session (set in chat_stream.py)",
        state_query=query[:100] if query else "EMPTY"
    )
    
    # Use full deep search result (not summary) so LLM sees everything
    deep_search_context = f"\n\n**DEEP SEARCH RESULT:**\n{deep_search_result}" if deep_search_result else ""
    logger.info("Deep search context prepared", context_length=len(deep_search_context))
    
    # CRITICAL: Include current date and time for context
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    prompt = f"""Generate 2-3 clarifying questions about this research topic.

Current date: {current_date}
Current time: {current_time}

Query: {user_message_for_context}
{deep_search_context}

Requirements:
- The message above is the ORIGINAL user query that started this deep research session
- Generate clarifying questions STRICTLY about THIS specific query and topic
- Questions MUST be directly related to the original query - do NOT generate questions about unrelated topics
- If the deep search context mentions specific aspects, generate questions about THOSE aspects within the context of the original query
- Do NOT generate questions about topics that are NOT mentioned in the original query or deep search context

You MUST always generate 2-3 clarifying questions about:
1. The specific aspect or focus of the research related to the original query (what exactly should be researched about THIS topic)
2. The depth and scope of the research (how detailed should it be for THIS topic)
3. The type of information needed for THIS topic (technical details, historical context, comparisons, etc.)

Guidelines:
- Questions must be about the SPECIFIC TOPIC from the original query
- Do NOT generate questions about topics that are NOT in the original query
- Do NOT misinterpret the query - questions must be directly related to the original topic
- These questions help guide the research direction and ensure comprehensive coverage of THE ORIGINAL TOPIC
- Even if the query seems clear, ask questions to refine the research approach FOR THIS SPECIFIC TOPIC

Language Requirement:
- Write all questions in the SAME LANGUAGE as the original user message
- If the user wrote in Russian, write questions in Russian
- If the user wrote in English, write questions in English
- Match the language exactly

Output Requirements:
- You MUST always return at least 2 questions in the questions list, even if the query seems clear
- Set needs_clarification=True and provide meaningful questions that help improve research quality
- Format questions with: question (the actual question), why_needed (why this clarification helps), default_assumption (what we'll assume if not answered)
- All fields must be in the same language as the user's message

IMPORTANT: You MUST always return at least 2 questions in the questions list, even if the query seems clear.
Set needs_clarification=True and provide meaningful questions that help improve research quality.

Format questions with:
- question: The actual question (in the same language as the user's message)
- why_needed: Why this clarification helps improve research (in the same language as the user's message)
- default_assumption: What we'll assume if not answered (in the same language as the user's message)
"""
    
    try:
        # CRITICAL: Don't use user_language from state - let LLM detect language from query itself
        # This is simpler and more reliable
        
        system_prompt = """You are a research planning expert. Generate clarifying questions to help improve research quality.

CRITICAL REQUIREMENTS:
1. Write all questions in the SAME LANGUAGE as the user's query - detect the language from the query and match it exactly
2. Generate questions STRICTLY about the SPECIFIC TOPIC from the original user message
3. Do NOT generate questions about topics that are NOT mentioned in the original query or deep search context
4. Questions should help refine the research approach FOR THE ORIGINAL TOPIC
5. You MUST return at least 2-3 questions in the questions list - NEVER return empty list
6. Make questions specific to the query topic, not generic

MANDATORY: The questions field MUST contain 2-3 questions. The model enforces min_length=2, so you MUST provide questions."""
        
        # Check if LLM is available
        if not llm:
            logger.error("LLM is None - cannot generate clarification questions")
            clarification = None
        else:
            import asyncio
            if stream:
                stream.emit_status("Generating clarification questions...", step="clarification")
            
            try:
                # Set timeout to 30 seconds for clarification generation
                clarification = await asyncio.wait_for(
                    llm.with_structured_output(ClarificationNeeds).ainvoke([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]),
                    timeout=30.0
                )
                logger.info("Clarification questions generated successfully", 
                           questions_count=len(clarification.questions) if clarification and clarification.questions else 0)
            except asyncio.TimeoutError:
                logger.error("Clarification generation TIMED OUT - using fallback questions")
                clarification = None
                if stream:
                    stream.emit_status("Using default questions (LLM timeout)", step="clarification")
            except Exception as e:
                logger.error("Failed to generate clarification questions", error=str(e), exc_info=True)
                clarification = None
                if stream:
                    stream.emit_status("Using default questions (LLM error)", step="clarification")
        
        # Handle case when clarification generation failed
        if clarification is None:
            logger.warning("Clarification generation failed - using fallback questions")
            questions_to_send = []
        else:
            questions_count = len(clarification.questions) if clarification.questions else 0
            logger.info(
                "Clarification analysis",
                needs_clarification=clarification.needs_clarification,
                questions_count=questions_count,
                can_proceed=clarification.can_proceed_without,
                original_query=user_message_for_context[:100],
                questions_preview=[q.question[:100] if hasattr(q, 'question') else str(q)[:100] for q in clarification.questions[:3]] if clarification.questions else [],
                has_questions=bool(clarification.questions)
            )
            
            questions_to_send = clarification.questions if clarification.questions else []
        
        # If LLM didn't generate questions, create simple default ones
        # CRITICAL: Don't use user_language - just create universal questions that work for any language
        if not questions_to_send or len(questions_to_send) == 0:
            logger.warning("LLM didn't generate questions, creating simple default ones",
                         query_preview=user_message_for_context[:100])
            from src.workflow.research.models import ClarifyingQuestion
            
            # Simple universal questions - LLM will handle language in actual generation
            questions_to_send = [
                ClarifyingQuestion(
                    question=f"What specific aspect of '{user_message_for_context[:50]}...' should be the primary focus?",
                    why_needed="This helps narrow down the research scope.",
                    default_assumption="We'll research all major aspects comprehensively."
                ),
                ClarifyingQuestion(
                    question="What level of detail do you need?",
                    why_needed="This determines the depth of information we'll gather.",
                    default_assumption="We'll provide a comprehensive overview."
                )
            ]
        
        # ALWAYS send questions to user - this is mandatory after deep search
        # CRITICAL: Log before sending to track execution flow
        logger.info("About to send clarification questions",
                   questions_count=len(questions_to_send) if questions_to_send else 0,
                   has_stream=bool(stream),
                   deep_search_result_length=len(deep_search_result_text) if deep_search_result_text else 0,
                   note="This log confirms we reached the sending stage")
        
        if questions_to_send and len(questions_to_send) > 0:
            # CRITICAL: Get deep search result from state to combine with clarification questions
            # This ensures both are in the same message and won't be lost on page reload
            # Use the deep_search_result_text we extracted earlier
            deep_search_result = deep_search_result_text
            
            logger.info("Preparing to send clarification questions",
                       questions_count=len(questions_to_send),
                       has_deep_search=bool(deep_search_result),
                       note="Combining deep search with clarification questions")
            
            # CRITICAL: If deep search was already sent separately (for streaming), we still need to include it
            # in the combined message for DB persistence. But we should NOT send it again via stream.
            # Check if deep search was already emitted by checking if it's in the current assistant message
            # For now, we'll include it in combined message and send only the clarification part via stream
            # OR send the full combined message (which will append to existing deep search in the same message)
            
            # Build combined message: deep search + double line break + clarification questions
            questions_text = "\n\n".join([
                f"**Q{i+1}:** {q.question}\n\n*Why needed:* {q.why_needed}\n\n*Default assumption if not answered:* {q.default_assumption}"
                for i, q in enumerate(questions_to_send)
            ])
            
            # Combine deep search and clarification in one message
            combined_message_parts = []
            
            # Add deep search if available
            if deep_search_result:
                # Normalize deep_search_result to ensure it ends properly
                normalized_deep_search = deep_search_result.rstrip()
                # Build deep search section ending with 2 empty lines (4 newlines)
                # CRITICAL: End with \n\n\n\n to ensure 2 empty lines before questions
                deep_search_section = f"## Initial Deep Search Context\n\n{normalized_deep_search}\n\n---\n\n*This context will be used to guide the research agents.*\n\n\n\n"
                combined_message_parts.append(deep_search_section)
            
            # Add clarification questions (starts directly, no extra newlines since deep search already has 2 empty lines)
            combined_message_parts.append(f"""## 🔍 Clarification Needed

Before starting the research, I need to clarify a few points:

{questions_text}

---

*Note: Please answer these questions to help guide the research direction. 
Research will proceed after you provide your answers.*""")
            
            # Join parts - deep_search_section already ends with \n\n\n\n, so just join with empty string
            # This ensures proper markdown separation: deep_search ends with 2 empty lines, then clarification starts
            combined_message = "".join(combined_message_parts)
            
            # Send clarifying questions to user via stream
            if stream:
                # CRITICAL: If deep search was already sent, only send clarification part
                # Otherwise send full combined message
                # CRITICAL: Check if deep search was already emitted by looking at recent chat history AND database
                # This ensures we catch deep search even if it was sent in a previous session
                deep_search_already_sent = False
                existing_deep_search_message_id = None
                
                # First check chat_history (from current session)
                if chat_history:
                    for msg in reversed(chat_history[-5:]):  # Check last 5 messages
                        if msg.get("role") == "assistant" and "Initial Deep Search Context" in msg.get("content", ""):
                            deep_search_already_sent = True
                            logger.info("Deep search found in chat_history - already sent separately",
                                       note="Will update existing message in DB with combined content")
                            break
                
                # Also check database directly (for page reloads and cross-session scenarios)
                # CRITICAL: Check for BOTH separate deep search AND combined messages
                if not deep_search_already_sent and stream:
                    try:
                        from src.database.schema import ChatMessageModel
                        from sqlalchemy import select
                        
                        app_state = stream.app_state if stream else {}
                        chat_id = app_state.get("chat_id")
                        session_factory = app_state.get("session_factory")
                        
                        if chat_id and session_factory:
                            async with session_factory() as session:
                                # CRITICAL: First check for combined message (deep search + clarification)
                                # This is the preferred format - both in one message
                                combined_result = await session.execute(
                                    select(ChatMessageModel)
                                    .where(
                                        ChatMessageModel.chat_id == chat_id,
                                        ChatMessageModel.role == "assistant",
                                        ChatMessageModel.content.like("%Initial Deep Search Context%"),
                                        ChatMessageModel.content.like("%Clarification Needed%")
                                    )
                                    .order_by(ChatMessageModel.created_at.desc())
                                    .limit(1)
                                )
                                combined_msg = combined_result.scalar_one_or_none()
                                
                                if combined_msg:
                                    # Combined message already exists - don't update, it's already correct
                                    deep_search_already_sent = True
                                    logger.info("Combined deep search + clarification already exists in database",
                                              existing_message_id=combined_msg.message_id,
                                              note="Message already contains both parts, no update needed")
                                else:
                                    # Check for separate deep search message (without clarification)
                                    separate_result = await session.execute(
                                        select(ChatMessageModel)
                                        .where(
                                            ChatMessageModel.chat_id == chat_id,
                                            ChatMessageModel.role == "assistant",
                                            ChatMessageModel.content.like("%Initial Deep Search Context%"),
                                            ~ChatMessageModel.content.like("%Clarification Needed%")  # NOT contains clarification
                                        )
                                        .order_by(ChatMessageModel.created_at.desc())
                                        .limit(1)
                                    )
                                    existing_msg = separate_result.scalar_one_or_none()
                                    
                                    if existing_msg:
                                        deep_search_already_sent = True
                                        existing_deep_search_message_id = existing_msg.message_id
                                        logger.info("Separate deep search found in database - will update with combined content",
                                                  existing_message_id=existing_deep_search_message_id,
                                                  existing_content_length=len(existing_msg.content),
                                                  note="Will update existing message to include clarification")
                    except Exception as e:
                        logger.warning("Failed to check database for existing deep search message", error=str(e))
                
                try:
                    if deep_search_already_sent and deep_search_result:
                        # Deep search already sent - only send clarification part via stream
                        # But save full combined message to DB
                        # CRITICAL: Deep search ends with \n\n\n\n (4 newlines = 2 empty lines) - see line 340
                        # Clarification should start directly without extra newlines since deep search already has 2 empty lines
                        # Result: \n\n\n\n (end of deep search) + (start of clarification) = 2 empty lines separation
                        # CRITICAL: Clarification starts directly without extra newlines
                        # Deep search already ends with \n\n\n\n (4 newlines = 2 empty lines)
                        # When frontend concatenates: existing (ends with \n\n\n\n) + clarification_only (starts with ##)
                        # Result: proper 2 empty lines separation in markdown
                        clarification_only = f"""## 🔍 Clarification Needed

Before starting the research, I need to clarify a few points:

{questions_text}

---

*Note: Please answer these questions to help guide the research direction. 
Research will proceed after you provide your answers.*"""
                        
                        # CRITICAL: Verify that clarification starts correctly (no leading newlines)
                        if clarification_only.startswith('\n'):
                            logger.warning("Clarification has leading newlines - removing them for proper separation",
                                         clarification_preview=clarification_only[:100])
                            clarification_only = clarification_only.lstrip('\n')
                        
                        # CRITICAL: Ensure proper separation between deep search and clarification
                        # Deep search already ends with \n\n\n\n (4 newlines = 2 empty lines)
                        # When sending clarification separately, we need to ensure frontend sees the separation
                        # Markdown requires 2 empty lines (4 newlines) for proper section separation
                        # Since deep search ends with \n\n\n\n, clarification should start directly
                        # But to ensure frontend renders it correctly, add explicit markdown separator
                        # Use markdown horizontal rule or ensure proper spacing
                        clarification_with_separator = "\n\n---\n\n" + clarification_only
                        
                        # CRITICAL: Add delay before sending clarification to ensure deep search chunk is fully processed
                        # This gives frontend time to render the deep search content before clarification arrives
                        await asyncio.sleep(1.0)
                        
                        stream.emit_report_chunk(clarification_with_separator)
                        logger.info("Clarification chunk sent", 
                                   clarification_length=len(clarification_only),
                                   starts_with=clarification_only[:50],
                                   note="Should start with '## 🔍' without leading newlines")
                        logger.info("Clarification questions emitted (deep search already sent separately)",
                                   clarification_length=len(clarification_only))
                    else:
                        # Send full combined message
                        chunk_size = 10000
                        chunks = [combined_message[i:i+chunk_size] for i in range(0, len(combined_message), chunk_size)]
                        logger.info("Sending combined deep search + clarification in chunks", 
                                   total_length=len(combined_message), 
                                   chunks_count=len(chunks),
                                   has_deep_search=bool(deep_search_result))
                        
                        for i, chunk in enumerate(chunks):
                            stream.emit_report_chunk(chunk)
                            if i < len(chunks) - 1:
                                await asyncio.sleep(0.03)
                        
                        logger.info("Combined deep search + clarification questions emitted", 
                                   message_length=len(combined_message),
                                   has_deep_search=bool(deep_search_result))
                    
                    # CRITICAL: Save combined message (deep search + clarification) to DB
                    # This ensures both are persisted together and won't be lost on page reload
                    # IMPORTANT: If deep search was already sent separately, update existing message
                    # Otherwise create new message with combined content
                    from uuid import uuid4
                    import time
                    session_id = state.get("session_id", "unknown")
                    
                    # CRITICAL: Always use existing message_id if found, otherwise create new
                    # This ensures we update the same message instead of creating duplicates
                    message_id = existing_deep_search_message_id
                    if not message_id:
                        # Check if combined message already exists (shouldn't happen, but handle it)
                        try:
                            from src.database.schema import ChatMessageModel
                            from sqlalchemy import select
                            
                            app_state = stream.app_state if stream else {}
                            chat_id = app_state.get("chat_id")
                            session_factory = app_state.get("session_factory")
                            
                            if chat_id and session_factory:
                                async with session_factory() as session:
                                    # Check for existing combined message
                                    result = await session.execute(
                                        select(ChatMessageModel)
                                        .where(
                                            ChatMessageModel.chat_id == chat_id,
                                            ChatMessageModel.role == "assistant",
                                            ChatMessageModel.content.like("%Initial Deep Search Context%"),
                                            ChatMessageModel.content.like("%Clarification Needed%")
                                        )
                                        .order_by(ChatMessageModel.created_at.desc())
                                        .limit(1)
                                    )
                                    existing_combined = result.scalar_one_or_none()
                                    if existing_combined:
                                        message_id = existing_combined.message_id
                                        logger.info("Found existing combined message - will update it",
                                                  message_id=message_id,
                                                  note="Updating existing combined message instead of creating duplicate")
                        except Exception as e:
                            logger.warning("Failed to check for existing combined message", error=str(e))
                    
                    # Create new message_id if still not found
                    if not message_id:
                        message_id = f"deep_search_clarification_{session_id}_{int(time.time() * 1000)}"
                    
                    # CRITICAL: Verify combined_message has proper separation before saving
                    # Ensure deep search ends with \n\n\n\n (4 newlines) and clarification starts with ##
                    if "## Initial Deep Search Context" in combined_message and "## 🔍 Clarification Needed" in combined_message:
                        # Verify separation is correct
                        deep_search_end = combined_message.find("## 🔍 Clarification Needed")
                        if deep_search_end > 0:
                            before_clarification = combined_message[:deep_search_end]
                            trailing_newlines = len(before_clarification) - len(before_clarification.rstrip('\n'))
                            if trailing_newlines < 4:
                                # Fix separation - ensure 4 newlines before clarification
                                combined_message = before_clarification.rstrip('\n') + '\n' * 4 + combined_message[deep_search_end:]
                                logger.info("Fixed separation in combined message before saving",
                                           added_newlines=4-trailing_newlines,
                                           note="Ensured 4 newlines (2 empty lines) between deep search and clarification")
                    
                    # Save or update message in DB (update if message_id exists, create if new)
                    await _save_message_to_db_async(
                        stream=stream,
                        role="assistant",
                        content=combined_message,
                        message_id=message_id,
                    )
                    logger.info("Combined deep search + clarification saved to DB",
                               message_id=message_id,
                               content_length=len(combined_message),
                               has_deep_search="## Initial Deep Search Context" in combined_message,
                               has_clarification="## 🔍 Clarification Needed" in combined_message,
                               was_update=bool(existing_deep_search_message_id),
                               note="Message will contain both deep search and clarification for proper page reload")
                except Exception as e:
                    logger.error("Failed to emit clarification questions as report chunk", error=str(e), exc_info=True)
                
                # CRITICAL: Update status to indicate questions were sent
                try:
                    stream.emit_status("✅ Clarification questions sent - please answer them", step="clarification")
                    logger.info("Status updated: Clarification questions sent",
                               questions_count=len(questions_to_send),
                               combined_message_length=len(combined_message),
                               note="Questions should now be visible on frontend")
                except Exception as e:
                    logger.error("Failed to emit clarification status", error=str(e))
                
                # Small delay to ensure all events are sent before graph stops
                await asyncio.sleep(0.2)
                
                # Also emit as a separate message event to ensure it's visible
                logger.info("MANDATORY: Combined deep search + clarification sent to user", 
                           questions_count=len(questions_to_send),
                           message_length=len(combined_message),
                           deep_search_used=bool(deep_search_result),
                           has_deep_search=bool(deep_search_result),
                           chunks_sent=len(chunks) if 'chunks' in locals() else 0,
                           clarification_only_sent=deep_search_already_sent if 'deep_search_already_sent' in locals() else False,
                           note="Questions MUST be visible to user - check frontend if not showing")
            
            # Always proceed with assumptions (user can answer later, but research continues)
            logger.info("Clarifying questions shown to user, proceeding with default assumptions. User can provide answers in chat.")
            
            return {
                "clarification_needed": True,  # Questions were sent
                "clarification_questions": [q.dict() if hasattr(q, "dict") else {"question": q.question, "why_needed": q.why_needed, "default_assumption": q.default_assumption} for q in questions_to_send]  # Store for reference
            }
        else:
            # This should never happen after our fixes, but log if it does
            logger.error("CRITICAL: questions_to_send is empty after all fallbacks! This should not happen!",
                        clarification_was_none=clarification is None,
                        query_preview=user_message_for_context[:100])
            # Even in this case, try to create emergency questions
            from src.workflow.research.models import ClarifyingQuestion
            emergency_questions = [
                ClarifyingQuestion(
                    question="What specific aspect should be the primary focus?",
                    why_needed="To narrow down research scope",
                    default_assumption="We'll research all major aspects"
                ),
                ClarifyingQuestion(
                    question="What level of detail do you need?",
                    why_needed="To determine research depth",
                    default_assumption="We'll provide comprehensive overview"
                )
            ]
            # Try to send emergency questions if stream is available
            if stream:
                try:
                    emergency_message = "## 🔍 Clarification Needed\n\n**Q1:** What specific aspect should be the primary focus?\n\n**Q2:** What level of detail do you need?"
                    stream.emit_report_chunk(emergency_message)
                    await _save_message_to_db_async(stream=stream, role="assistant", content=emergency_message, message_id=f"emergency_clarification_{state.get('session_id', 'unknown')}")
                    logger.warning("Sent emergency clarification questions")
                except Exception as e:
                    logger.error("Failed to send emergency questions", error=str(e))
            return {"clarification_needed": True, "clarification_questions": [{"question": q.question, "why_needed": q.why_needed, "default_assumption": q.default_assumption} for q in emergency_questions]}
        
    except Exception as e:
        logger.error("Clarification analysis failed", error=str(e), exc_info=True)
        # CRITICAL: Even if everything fails, try to send emergency questions
        # This ensures user always sees questions, not just "Analyzing..."
        if stream:
            try:
                from src.workflow.research.models import ClarifyingQuestion
                emergency_questions = [
                    ClarifyingQuestion(
                        question="What specific aspect should be the primary focus?",
                        why_needed="To narrow down research scope",
                        default_assumption="We'll research all major aspects"
                    ),
                    ClarifyingQuestion(
                        question="What level of detail do you need?",
                        why_needed="To determine research depth",
                        default_assumption="We'll provide comprehensive overview"
                    )
                ]
                emergency_message = "## 🔍 Clarification Needed\n\n**Q1:** What specific aspect should be the primary focus?\n\n**Q2:** What level of detail do you need?"
                stream.emit_report_chunk(emergency_message)
                await _save_message_to_db_async(
                    stream=stream,
                    role="assistant",
                    content=emergency_message,
                    message_id=f"emergency_clarification_{state.get('session_id', 'unknown')}"
                )
                stream.emit_status("✅ Emergency clarification questions sent", step="clarification")
                logger.warning("Sent emergency clarification questions after error",
                             error=str(e)[:200],
                             note="User should see questions even after error")
                return {
                    "clarification_needed": True,
                    "clarification_questions": [
                        {"question": q.question, "why_needed": q.why_needed, "default_assumption": q.default_assumption}
                        for q in emergency_questions
                    ]
                }
            except Exception as e2:
                logger.error("CRITICAL: Failed to send emergency questions after error", error=str(e2), exc_info=True)
        return {"clarification_needed": False}


# ==================== Analysis Node (Enhanced) ====================

async def analyze_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze query and prepare for research planning.
    
    CRITICAL: If clarification was needed but user hasn't answered yet, stop here.
    """
    # Check if clarification was needed and user hasn't answered
    clarification_needed = state.get("clarification_needed", False)
    chat_history = state.get("chat_history", [])
    
    if clarification_needed:
        # Check if user has answered
        has_user_answer = False
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if "clarification" in content or "🔍" in content or "clarify" in content:
                    # Check if next message is from user (user answered)
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        has_user_answer = True
                        logger.info("User answered clarification, proceeding with analysis")
                        break
        
        if not has_user_answer:
            # User hasn't answered yet - stop graph execution
            logger.info("CRITICAL: Clarification needed but user hasn't answered - STOPPING GRAPH")
            stream = state.get("stream")
            if stream:
                stream.emit_status("⏸️ Waiting for your clarification answers before proceeding...", step="clarification")
            # Return state that will stop the graph
            return {
                "clarification_waiting": True,
                "should_stop": True
            }
    """
    Analyze query to determine research approach.

    Uses structured output to assess query complexity and plan.
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)

    llm = state.get("llm")
    stream = state.get("stream")

    # Get user language from state (already detected in create_research_state)
    user_language = state.get("user_language", "English")
    logger.info(f"Using user language from state: {user_language}", query_preview=query[:100])

    if stream:
        stream.emit_status("Analyzing query complexity...", step="analysis")

    # Get deep_search_result for context
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    
    deep_search_context = f"\n\n**Initial Deep Search Context:**\n{deep_search_result[:1500] if deep_search_result else 'No initial deep search context available.'}\n" if deep_search_result else ""
    
    # CRITICAL: Extract user clarification answers from chat_history
    clarification_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    user_answer = chat_history[i + 1].get("content", "")
                    clarification_context = f"\n\n**USER CLARIFICATION ANSWERS (CRITICAL - MUST BE CONSIDERED):**\n{user_answer}\n\nThese answers refine the research scope. Use them when analyzing the query."
                    break
    
    prompt = f"""Analyze this research query to determine the best research approach.

Query: {query}
{clarification_context}
{deep_search_context}

Assess:
1. What are the main topics and subtopics?
2. How complex is this query (simple/moderate/complex)?
3. How many specialized research agents would be optimal?
4. What different research angles should agents cover?

If user provided clarification answers, use them to refine your analysis.
"""

    try:
        analysis = await llm.with_structured_output(QueryAnalysis).ainvoke([
            {"role": "system", "content": "You are an expert research planner. Analyze queries to determine complexity, topics, and optimal research strategy."},
            {"role": "user", "content": prompt}
        ])

        logger.info(
            "Query analyzed",
            complexity=analysis.complexity,
            agent_count=analysis.estimated_agent_count,
            topics=analysis.topics
        )

        return {
            "query_analysis": analysis.dict(),
            "requires_deep_search": analysis.requires_deep_search,
            "estimated_agent_count": analysis.estimated_agent_count
        }

    except Exception as e:
        logger.error("Query analysis failed", error=str(e))
        # Fallback
        return {
            "query_analysis": {
                "reasoning": "Fallback analysis due to error",
                "topics": [query],
                "complexity": "moderate",
                "requires_deep_search": True,
                "estimated_agent_count": 4
            },
            "requires_deep_search": True,
            "estimated_agent_count": 4
        }


# ==================== Planning Node (Enhanced) ====================

async def plan_research_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create detailed research plan with structured output.

    Generates research topics, priorities, and coordination strategy.
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    query_analysis = state.get("query_analysis", {})
    # Get deep_search_result - handle both dict and string formats
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        # If it's a dict with "type": "override", extract the value
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")

    if stream:
        stream.emit_status("Creating research plan...", step="planning")

    context_info = f"\n\nDeep search context:\n{deep_search_result}" if deep_search_result else ""
    
    # CRITICAL: Extract clarification questions AND user answers from chat_history
    clarification_context = ""
    clarification_questions_text = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                # Extract the questions themselves from assistant message
                assistant_content = msg.get("content", "")
                if "Clarification Needed" in assistant_content or "🔍" in assistant_content:
                    # Extract questions section (everything from "## 🔍 Clarification Needed" to "---" or end)
                    questions_start = assistant_content.find("## 🔍 Clarification Needed")
                    if questions_start != -1:
                        # Find the end of questions section (before user answers or at "---")
                        questions_end = assistant_content.find("\n---\n", questions_start)
                        if questions_end == -1:
                            questions_end = assistant_content.find("\n\n*Note:", questions_start)
                        if questions_end == -1:
                            questions_end = len(assistant_content)
                        
                        clarification_questions_text = assistant_content[questions_start:questions_end].strip()
                        logger.info("Extracted clarification questions for planning", 
                                  questions_preview=clarification_questions_text[:300],
                                  clarification_index=i)
                
                # Extract user answers
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    user_answer = chat_history[i + 1].get("content", "")
                    # Build comprehensive clarification context with both questions and answers
                    questions_section = f"\n\n**CLARIFICATION QUESTIONS ASKED:**\n{clarification_questions_text}\n" if clarification_questions_text else ""
                    clarification_context = f"{questions_section}\n\n**USER CLARIFICATION ANSWERS (CRITICAL - MUST BE CONSIDERED):**\n{user_answer}\n\nThese answers refine the research scope and priorities. Use them to focus the research plan."
                    logger.info("Extracted clarification questions and answers for planning",
                              questions_preview=clarification_questions_text[:200] if clarification_questions_text else "None",
                              answer_preview=user_answer[:200] if user_answer else "None")
                    break

    prompt = f"""Create a comprehensive research plan for this query.

Query: {query}
{clarification_context}

Query Analysis:
- Reasoning: {query_analysis.get('reasoning', '')}
- Identified topics: {', '.join(query_analysis.get('topics', []))}
- Complexity: {query_analysis.get('complexity', 'moderate')}
{context_info}

Requirements:
- Create a detailed research plan with specific topics for different agents to investigate
- If user provided clarification answers, use them to refine and focus the research plan
- Each topic should be specific and actionable
- Topics should cover different aspects of the query to ensure comprehensive research
- Plan should align with the query complexity level

Create specific research topics that agents can investigate to build a complete understanding of the query.
"""

    try:
        plan = await llm.with_structured_output(ResearchPlan).ainvoke([
            {"role": "system", "content": "You are a research strategy expert."},
            {"role": "user", "content": prompt}
        ])

        session_id = state.get("session_id")
        if not session_id:
            logger.warning("session_id not found in state - using 'unknown' for logging", state_keys=list(state.keys())[:10])
            session_id = "unknown"
        logger.info("Research plan created", 
                   topics_count=len(plan.topics),
                   session_id=session_id)

        research_plan_dict = {
            "reasoning": plan.reasoning,
            "research_depth": plan.research_depth,
            "coordination_strategy": plan.coordination_strategy
        }
        
        # CRITICAL: Save research plan to main.md for persistence and supervisor editing
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        if agent_memory_service:
            try:
                from datetime import datetime
                # Read current main.md
                try:
                    main_content = await agent_memory_service.file_manager.read_file("main.md")
                except FileNotFoundError:
                    main_content = ""
                
                # Format research plan for main.md
                topics_text = "\n".join([
                    f"- **{topic.topic}**: {topic.description} (Priority: {topic.priority}, Estimated sources: {topic.estimated_sources})"
                    for topic in plan.topics
                ])
                
                research_plan_section = f"""## Research Plan

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Research Depth:** {plan.research_depth}
**Coordination Strategy:** {plan.coordination_strategy}

### Strategy

{plan.reasoning}

### Research Topics

{topics_text}

---
**Note:** This research plan can be updated by the supervisor as research progresses.
"""
                
                # Append research plan to main.md (or create if empty)
                if main_content:
                    # Check if research plan section already exists
                    if "## Research Plan" in main_content:
                        # Replace existing research plan section
                        import re
                        pattern = r"## Research Plan.*?(?=\n## |\Z)"
                        main_content = re.sub(pattern, research_plan_section.strip(), main_content, flags=re.DOTALL)
                    else:
                        # Append research plan section
                        main_content = main_content + "\n\n" + research_plan_section
                else:
                    # Create new main.md with research plan
                    main_content = f"""# Research Session - Main Index

**Query:** {query}
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{research_plan_section}

## Key Insights

<!-- Supervisor will add key insights here as research progresses -->

## Notes

<!-- Additional notes and context -->
"""
                
                await agent_memory_service.file_manager.write_file("main.md", main_content)
                logger.info("Research plan saved to main.md", 
                           topics_count=len(plan.topics),
                           session_id=session_id)
            except Exception as e:
                logger.warning("Failed to save research plan to main.md", error=str(e), exc_info=True)
        
        return {
            "research_plan": research_plan_dict,
            "research_topics": [topic.dict() for topic in plan.topics]
        }

    except Exception as e:
        logger.error("Research planning failed", error=str(e))
        # Fallback with all required fields
        fallback_topic = ResearchTopic(
            topic=query,
            description=f"Research: {query}",
            priority="high",
            estimated_sources=5  # Default estimate for fallback
        )
        return {
            "research_plan": {
                "reasoning": "Fallback plan due to planning error",
                "research_depth": "standard",
                "coordination_strategy": "Parallel research"
            },
            "research_topics": [fallback_topic.dict()]
        }


# ==================== Agent Characteristics Creation (Enhanced) ====================

async def create_agent_characteristics_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create specialized agent characteristics with initial todos.

    Each agent gets:
    - Unique role and expertise
    - Personality
    - Initial structured todos
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    research_plan = state.get("research_plan", {})
    research_topics = state.get("research_topics", [])
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    settings = state.get("settings")

    # Get agent count from settings or state
    # CRITICAL: Use deep_research_num_agents from settings, not max_concurrent_agents
    # CRITICAL: LLM may suggest more agents, but we MUST respect settings limit
    if settings:
        max_agent_count = getattr(settings, "deep_research_num_agents", 3)
    else:
        from src.config.settings import get_settings
        settings_obj = get_settings()
        max_agent_count = getattr(settings_obj, "deep_research_num_agents", 3)
    
    # Get LLM's estimate, but cap it at settings limit
    estimated_from_llm = state.get("estimated_agent_count", max_agent_count)
    agent_count = min(estimated_from_llm, max_agent_count)  # CRITICAL: Cap at settings limit
    
    session_id = state.get("session_id")
    if not session_id:
        logger.warning("session_id not found in state - using 'unknown' for logging", state_keys=list(state.keys())[:10])
        session_id = "unknown"
    logger.info("Creating agent characteristics", 
               agent_count=agent_count, 
               estimated_from_llm=estimated_from_llm,
               max_from_settings=max_agent_count,
               session_id=session_id,
               note=f"Using {agent_count} agents (capped from LLM estimate {estimated_from_llm} by settings limit {max_agent_count})")

    # Don't emit status here - we'll emit after we know the actual agent count (after fallback)

    # Get deep search result and user clarification answers
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    
    clarification_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    user_answer = chat_history[i + 1].get("content", "")
                    clarification_context = f"\n\n**USER CLARIFICATION ANSWERS:**\n{user_answer}\n"
                    break

    # Get user language from state (already detected in create_research_state)
    user_language = state.get("user_language", "English")
    language_instruction = ""
    if user_language == "Russian":
        language_instruction = "\n\n**CRITICAL LANGUAGE REQUIREMENT:** The user's query is in RUSSIAN. All agent task descriptions, findings, and final reports MUST be in RUSSIAN language. Agents will conduct research using English sources but MUST write all outputs in RUSSIAN."
    else:
        language_instruction = f"\n\n**LANGUAGE REQUIREMENT:** All research outputs should be in {user_language}."

    prompt = f"""Create a team of {agent_count} specialized research agents for this project.

Query: {query}{language_instruction}

Initial Context:
{deep_search_result[:2000] if deep_search_result else "No initial context available."}
{clarification_context}

Research Topics:
{chr(10).join([f"- {t.get('topic')}: {t.get('description')}" for t in research_topics])}

Requirements:
- Create exactly {agent_count} agents, each with 2-3 initial tasks
- Each agent must cover a different research angle to build a complete picture
- All tasks must be unique across agents - no duplicate or similar task titles
- Each task must be specific and include the query in the objective
- Tasks must be self-contained (agents only see their task description, not the full query)

Task Creation Guidelines:
- Every task objective MUST include the original user query
- Every task MUST be specific to the user's query - not generic
- Task format: Start each task objective with "The user asked: '[query]'. Research [specific aspect related to query]..."
- If clarification answers are provided, interpret them IN THE CONTEXT of the original query
  * Clarification specifies WHAT ASPECT of the original topic to focus on, NOT a new topic
  * Include clarification answers in task descriptions, but ALWAYS in context of original query
- Do NOT create generic tasks - be SPECIFIC
- Do NOT interpret clarification as a standalone query - it's ALWAYS about the original query topic
- Each task must be self-contained - the agent will NOT see the original query, only the task description

For each agent, create:
1. Unique role (e.g., "Aviation Historian", "Technical Analyst", "Case Study Researcher")
2. Specific expertise area - ensure different angles:
   - Historical development and evolution
   - Technical specifications and details
   - Expert analysis and critical perspectives
   - Real-world applications and case studies
   - Industry trends and current state
   - Comparative analysis
   - Impact and implications
   - Challenges and limitations
3. Personality traits (thorough, analytical, critical, etc.)
4. 2-3 research tasks with unique titles and objectives based on agent's expertise
   - Each agent's tasks MUST be UNIQUE and based on their specific expertise
   - Tasks must reflect the agent's unique expertise angle
   - Do NOT create identical or similar tasks for different agents
   - Each task title and objective must be DISTINCT and reflect the agent's unique expertise angle

Distribution Requirements:
- You MUST create EXACTLY {agent_count} agents
- EACH agent MUST have 2-3 initial tasks (NOT just 1 task per agent!)
- Each agent's tasks MUST be UNIQUE - NO duplicate or similar tasks across agents
- Ensure agents cover DIFFERENT angles - avoid overlap
- Each agent should contribute unique insights to build comprehensive understanding
- All angles must relate to the user's query - do NOT research unrelated topics
- If clarification was provided, interpret it IN CONTEXT of the original query

Verification: Before responding, check:
1. Do you have exactly {agent_count} agents?
2. Does each agent have 2-3 tasks?
3. Are ALL task titles UNIQUE across all agents? (NO duplicates!)
4. Do tasks reflect each agent's unique expertise angle?
If any answer is NO, adjust your response!
"""

    try:
        characteristics = await llm.with_structured_output(AgentCharacteristics).ainvoke([
            {"role": "system", "content": f"You are an expert at designing research teams. Create exactly {agent_count} agents, each with 2-3 unique tasks. Each agent's tasks must be unique and reflect their specific expertise angle."},
            {"role": "user", "content": prompt}
        ])

        # CRITICAL: Validate that each agent has 2-3 tasks
        agents_with_insufficient_tasks = []
        for i, agent_char in enumerate(characteristics.agents):
            todos_count = len(agent_char.initial_todos)
            if todos_count < 2:
                agents_with_insufficient_tasks.append((i, agent_char.role, todos_count))
                logger.warning(f"Agent {i+1} ({agent_char.role}) has only {todos_count} tasks, expected 2-3")
        
        logger.info("Agent characteristics created by LLM",
                   agent_count=len(characteristics.agents),
                   expected_count=agent_count,
                   shortfall=max(0, agent_count - len(characteristics.agents)),
                   agents_with_insufficient_tasks=len(agents_with_insufficient_tasks))

        # CRITICAL: Validate that each agent has 2-3 tasks AND that tasks are UNIQUE across agents
        # The structured output model now enforces min_length=2, so LLM should create 2-3 tasks per agent
        from src.workflow.research.models import AgentTodo
        
        # Collect all task titles to check for duplicates
        all_task_titles = []
        all_task_objectives = []
        
        for agent_char in characteristics.agents:
            initial_count = len(agent_char.initial_todos)
            task_titles = [t.title for t in agent_char.initial_todos]
            task_objectives = [t.objective for t in agent_char.initial_todos]
            
            logger.info(f"Validating agent {agent_char.role} tasks",
                       initial_count=initial_count,
                       todos_titles=task_titles,
                       expertise=agent_char.expertise)
            
            all_task_titles.extend(task_titles)
            all_task_objectives.extend(task_objectives)
            
            # Model should enforce min_length=2, but double-check
            if initial_count < 2:
                logger.error(f"CRITICAL: Agent {agent_char.role} has only {initial_count} tasks - structured output model should enforce min_length=2!",
                           initial_count=initial_count,
                           role=agent_char.role,
                           expertise=agent_char.expertise,
                           note="This should not happen - model has min_length=2 constraint")
            elif initial_count == 1:
                # Fallback: if somehow only 1 task, add 1-2 more to reach 2-3
                logger.warning(f"Agent {agent_char.role} has only 1 task (model should prevent this), adding tasks to reach 2-3")
                target_tasks = 3  # Aim for 3 tasks
                tasks_added = 0
                while len(agent_char.initial_todos) < target_tasks:
                    task_num = len(agent_char.initial_todos) + 1
                    new_task = AgentTodo(
                        reasoning=f"Additional research task {task_num} for comprehensive coverage of {agent_char.expertise}",
                        title=f"{agent_char.role}: {agent_char.expertise} - Additional research task {task_num}",
                        objective=f"The user asked: '{query}'. Conduct additional research on {agent_char.expertise} to ensure comprehensive coverage. Focus on {agent_char.expertise} aspects not yet fully covered.",
                        expected_output=f"Additional findings about {agent_char.expertise}",
                        sources_needed=[],
                        guidance=f"Focus on {agent_char.expertise}. Use web search to find authoritative sources. Ensure comprehensive coverage."
                    )
                    agent_char.initial_todos.append(new_task)
                    tasks_added += 1
                
                final_count = len(agent_char.initial_todos)
                logger.info(f"Added {tasks_added} tasks to agent {agent_char.role}, now has {final_count} tasks",
                           initial_count=initial_count,
                           final_count=final_count,
                           todos_titles=[t.title for t in agent_char.initial_todos])
            else:
                logger.info(f"Agent {agent_char.role} has sufficient tasks", count=initial_count)
        
        # CRITICAL: Check for duplicate tasks across agents
        from collections import Counter
        title_counts = Counter(all_task_titles)
        objective_counts = Counter([obj[:100] for obj in all_task_objectives])  # Check first 100 chars for similarity
        
        duplicates = {title: count for title, count in title_counts.items() if count > 1}
        similar_objectives = {obj: count for obj, count in objective_counts.items() if count > 1}
        
        if duplicates:
            logger.error(f"CRITICAL: Found duplicate task titles across agents!",
                       duplicates=duplicates,
                       note="Tasks must be unique per agent based on their expertise")
            
            # Fix duplicates by making tasks unique based on agent expertise and role
            for agent_char in characteristics.agents:
                for todo in agent_char.initial_todos:
                    if todo.title in duplicates:
                        # Make title and objective unique by incorporating agent's specific expertise
                        original_title = todo.title
                        original_objective = todo.objective
                        
                        # Extract the core task from title (remove generic parts)
                        core_task = todo.title
                        if ":" in core_task:
                            core_task = core_task.split(":", 1)[1].strip()
                        
                        # Create unique title based on agent's expertise
                        todo.title = f"{agent_char.role}: {agent_char.expertise} - {core_task}"
                        
                        # Update objective to be specific to agent's expertise
                        if agent_char.expertise.lower() not in todo.objective.lower():
                            todo.objective = f"The user asked: '{query}'. Research {core_task} with focus on {agent_char.expertise}. {agent_char.expertise} aspects: {todo.objective}"
                        
                        logger.warning(f"Made task unique for agent {agent_char.role}",
                                     original_title=original_title,
                                     new_title=todo.title,
                                     expertise=agent_char.expertise)
        
        if similar_objectives:
            logger.warning(f"Found similar task objectives across agents",
                         similar_count=len(similar_objectives),
                         note="This may indicate agents have overlapping tasks - consider reviewing")
        
        # CRITICAL: Fallback if LLM returned fewer agents than requested
        # This happens with weaker models (GPT-4-mini, etc.) that struggle with large lists in structured output
        if len(characteristics.agents) < agent_count:
            logger.warning(f"LLM returned {len(characteristics.agents)} agents but {agent_count} were requested. Creating {agent_count - len(characteristics.agents)} additional agents with fallback logic.")

            # Get topics that weren't covered yet
            covered_topics = [agent.expertise.lower() for agent in characteristics.agents]
            remaining_topics = [
                topic for topic in research_topics
                if not any(covered in topic.get('topic', '').lower() or covered in topic.get('description', '').lower()
                          for covered in covered_topics)
            ]

            # Import AgentCharacteristic and AgentTodo for creating fallback agents
            from src.workflow.research.models import AgentCharacteristic, AgentTodo

            # Create fallback agents for remaining topics
            # Each fallback agent should have 2-3 tasks to match the expected distribution
            for i in range(len(characteristics.agents), agent_count):
                agent_num = i + 1
                if i - len(characteristics.agents) < len(remaining_topics):
                    topic = remaining_topics[i - len(characteristics.agents)]
                    topic_name = topic.get('topic', f'Research Area {agent_num}')
                    topic_desc = topic.get('description', '')

                    fallback_agent = AgentCharacteristic(
                        role=f"Research Specialist {agent_num}",
                        expertise=topic_name,
                        personality="Thorough, analytical, detail-oriented",
                        initial_todos=[
                            AgentTodo(
                                reasoning=f"Research {topic_name} as specified in the research plan",
                                title=f"Research: {topic_name}",
                                objective=f"The user asked: '{query}'. Research {topic_name}: {topic_desc}",
                                expected_output=f"Comprehensive findings about {topic_name} with verified sources",
                                sources_needed=[],
                                guidance="Use web search to find authoritative sources. Focus on accuracy and depth."
                            ),
                            AgentTodo(
                                reasoning=f"Analyze findings and identify key insights about {topic_name}",
                                title=f"Analyze {topic_name} findings",
                                objective=f"Synthesize research on {topic_name} and extract key insights relevant to the user's query: '{query}'",
                                expected_output=f"Key insights and analysis of {topic_name}",
                                sources_needed=[],
                                guidance="Focus on answering the user's original question with your findings."
                            ),
                            AgentTodo(
                                reasoning=f"Verify and cross-reference key claims about {topic_name}",
                                title=f"Verify {topic_name} findings",
                                objective=f"The user asked: '{query}'. Verify important claims about {topic_name} by finding multiple independent sources and cross-referencing information.",
                                expected_output=f"Verified and cross-referenced findings about {topic_name}",
                                sources_needed=[],
                                guidance="Find multiple independent sources to verify key claims. Cross-reference information for accuracy."
                            )
                        ]
                    )
                else:
                    # If we ran out of topics, create a generic research agent with multiple tasks
                    fallback_agent = AgentCharacteristic(
                        role=f"General Research Agent {agent_num}",
                        expertise=f"General research and analysis",
                        personality="Thorough, analytical, detail-oriented",
                        initial_todos=[
                            AgentTodo(
                                reasoning=f"Provide additional research coverage for the user's query",
                                title=f"Additional research for: {query}",
                                objective=f"The user asked: '{query}'. Conduct supplementary research to fill any gaps not covered by other agents.",
                                expected_output="Additional relevant findings that complement other agents' work",
                                sources_needed=[],
                                guidance="Focus on aspects not fully covered by other agents. Use web search to find authoritative sources."
                            ),
                            AgentTodo(
                                reasoning=f"Analyze and synthesize findings from multiple sources",
                                title=f"Synthesize findings for: {query}",
                                objective=f"The user asked: '{query}'. Analyze and synthesize findings from multiple sources to provide comprehensive coverage.",
                                expected_output="Synthesized analysis combining multiple perspectives",
                                sources_needed=[],
                                guidance="Combine findings from different sources to provide a comprehensive view."
                            ),
                            AgentTodo(
                                reasoning=f"Identify and investigate related aspects not yet covered",
                                title=f"Explore related aspects for: {query}",
                                objective=f"The user asked: '{query}'. Identify and investigate related aspects, connections, or implications that haven't been fully explored by other agents.",
                                expected_output="Findings about related aspects and connections",
                                sources_needed=[],
                                guidance="Look for related topics, connections, or implications that add depth to the research."
                            )
                        ]
                    )

                characteristics.agents.append(fallback_agent)
                logger.info(f"Created fallback agent {i+1}", role=fallback_agent.role, expertise=fallback_agent.expertise)

        logger.info("Final agent team ready", agent_count=len(characteristics.agents))

        # Emit status with ACTUAL agent count (after fallback)
        if stream:
            stream.emit_status(f"Creating {len(characteristics.agents)} specialized research agents...", step="agent_characteristics")

        # Get memory services from stream
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        agent_file_service = stream.app_state.get("agent_file_service") if stream else None

        # Create supervisor file if services available
        if agent_file_service:
            try:
                await agent_file_service.write_agent_file(
                    agent_id="supervisor",
                    todos=[],
                    notes=[],
                    character="""**Role**: Research Supervisor
**Expertise**: Coordinating research teams, synthesizing findings, identifying gaps
**Personality**: Analytical, strategic, thorough
""",
                    preferences="Focus on comprehensive, diverse research coverage. Keep main.md minimal with only essential shared information."
                )
                logger.info("Supervisor file created")
            except Exception as e:
                logger.error("Failed to create supervisor file", error=str(e))

        # Create agent files with initial todos
        agent_chars = {}
        for i, agent_char in enumerate(characteristics.agents):
            agent_id = f"agent_{i+1}"

            # CRITICAL: Log todos count BEFORE conversion to verify fallback worked
            initial_todos_count = len(agent_char.initial_todos)
            logger.info(f"Creating agent file for {agent_id}",
                       role=agent_char.role,
                       initial_todos_count=initial_todos_count,
                       todos_titles=[t.title for t in agent_char.initial_todos])

            # Convert todos to AgentTodoItem
            agent_todos = [
                AgentTodoItem(
                    reasoning=todo.reasoning,
                    title=todo.title,
                    objective=todo.objective,
                    expected_output=todo.expected_output,
                    sources_needed=todo.sources_needed,
                    status="pending",
                    note=todo.guidance if hasattr(todo, 'guidance') and todo.guidance else ""
                )
                for todo in agent_char.initial_todos
            ]
            
            # CRITICAL: Verify todos count after conversion
            if len(agent_todos) != initial_todos_count:
                logger.error(f"CRITICAL: Todos count mismatch for {agent_id}",
                           before_conversion=initial_todos_count,
                           after_conversion=len(agent_todos))
            if len(agent_todos) < 2:
                logger.error(f"CRITICAL: Agent {agent_id} has only {len(agent_todos)} tasks after fallback - this should not happen!",
                           role=agent_char.role,
                           todos_titles=[t.title for t in agent_todos])

            # Create agent file if services available
            if agent_file_service:
                try:
                    await agent_file_service.write_agent_file(
                        agent_id=agent_id,
                        todos=agent_todos,
                        character=f"""**Role**: {agent_char.role}
**Expertise**: {agent_char.expertise}
**Personality**: {agent_char.personality}
""",
                        preferences=f"Focus on: {agent_char.expertise}"
                    )
                    logger.info(f"Agent file created", agent_id=agent_id, todos=len(agent_todos))

                    # Emit todos to frontend so user can see agent progress immediately
                    if stream and agent_todos:
                        todos_dict = [
                            {
                                "title": t.title,
                                "status": t.status,
                                "objective": t.objective,
                                "expected_output": t.expected_output,
                            }
                            for t in agent_todos
                        ]
                        stream.emit_agent_todo(agent_id, todos_dict)
                        logger.info(f"Agent todos emitted to frontend", agent_id=agent_id, todos_count=len(todos_dict))
                except Exception as e:
                    logger.error(f"Failed to create agent file", agent_id=agent_id, error=str(e))

            agent_chars[agent_id] = {
                "role": agent_char.role,
                "expertise": agent_char.expertise,
                "personality": agent_char.personality,
                "initial_todos": [todo.dict() for todo in agent_char.initial_todos]
            }

        return {
            "agent_characteristics": agent_chars,
            "agent_count": len(agent_chars),
            "coordination_notes": characteristics.coordination_notes
        }

    except Exception as e:
        logger.error("Agent characteristics creation failed", error=str(e))
        # Fallback: create simple agents
        fallback_chars = {}
        for i in range(agent_count):
            agent_id = f"agent_{i+1}"
            fallback_chars[agent_id] = {
                "role": f"Research Agent {i+1}",
                "expertise": "general research",
                "personality": "thorough and analytical",
                "initial_todos": []
            }
        return {
            "agent_characteristics": fallback_chars,
            "agent_count": agent_count,
            "coordination_notes": "Parallel research with supervisor coordination"
        }


# ==================== Execute Agents Node (Enhanced with Queue) ====================

async def execute_agents_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute research agents in parallel with supervisor queue coordination.
    
    Agents work in parallel, each on ONE task at a time.
    When an agent completes a task, it queues for supervisor review.
    Agents continue working on next tasks until all todos are done.
    
    This implements the requirement: "МНОГО АГЕНТОВ ПОЧТИ ОДНОВРЕМЕННО 
    ВЫПОЛНЯЮТ ЗАДАЧУ И ВЫЗЫВАЮТ ГЛАВНОГО!"
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    agent_characteristics = state.get("agent_characteristics", {})
    agent_count = state.get("agent_count", 4)
    llm = state.get("llm")
    search_provider = state.get("search_provider")
    scraper = state.get("scraper")
    stream = state.get("stream")
    settings = state.get("settings")
    # Get max_iterations from settings (centralized config)
    if settings:
        max_iterations = state.get("max_iterations", settings.deep_research_default_max_iterations)
    else:
        from src.config.settings import get_settings
        settings_obj = get_settings()
        max_iterations = state.get("max_iterations", settings_obj.deep_research_default_max_iterations)
    current_iteration = state.get("iteration", 0)

    # Don't emit status here - we'll emit after discovering agents from files

    # Create supervisor queue
    supervisor_queue = SupervisorQueue()

    # Store in state for agents to access
    state["supervisor_queue"] = supervisor_queue

    # All collected findings from all agent iterations
    all_findings = []
    
    # Track supervisor calls (not ReAct iterations, but actual supervisor invocations)
    supervisor_call_count = state.get("supervisor_call_count", 0)
    # Get max_supervisor_calls from settings (centralized config)
    settings = state.get("settings")
    if settings:
        max_supervisor_calls = settings.deep_research_max_supervisor_calls
    else:
        from src.config.settings import get_settings
        settings_obj = get_settings()
        max_supervisor_calls = settings_obj.deep_research_max_supervisor_calls
    
    # Run agents in continuous mode until all todos complete or max iterations
    agents_active = True
    iteration_count = 0
    
    # CRITICAL: Hard limit to prevent infinite loops
    # If max_iterations reached, MUST stop and generate report
    while agents_active and iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"Agent execution cycle {iteration_count}")
        
        if stream:
            stream.emit_status(f"🔄 Agent execution cycle {iteration_count}/{max_iterations} (Supervisor calls: {supervisor_call_count}/{max_supervisor_calls})", step="agents")
            logger.info(f"Emitting progress: cycle {iteration_count}/{max_iterations}, supervisor calls {supervisor_call_count}/{max_supervisor_calls}")
        
        # Launch all agents in parallel for this iteration
        # Get max_steps from settings (centralized config)
        if settings:
            agent_max_steps = settings.deep_research_agent_max_steps
        else:
            from src.config.settings import get_settings
            settings_obj = get_settings()
            agent_max_steps = settings_obj.deep_research_agent_max_steps
        
        # CRITICAL FIX (Bug #26): Load agents from files, not agent_characteristics
        # Supervisor can create new agents via create_agent_todo, but they won't be in agent_characteristics
        # So we need to load ALL agents from files to include agent_2, agent_3, etc.
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        agent_file_service = stream.app_state.get("agent_file_service") if stream else None

        if agent_file_service:
            try:
                # Get list of all agent files
                file_manager = agent_file_service.file_manager
                agent_files = await file_manager.list_files("agents/agent_*.md")

                # Extract agent IDs from filenames (e.g., "agents/agent_1.md" -> "agent_1")
                discovered_agents = []
                for file_path in agent_files:
                    agent_id = file_path.replace("agents/", "").replace(".md", "")
                    if agent_id.startswith("agent_"):
                        discovered_agents.append(agent_id)

                logger.info(f"Discovered {len(discovered_agents)} agents from files",
                           agents=discovered_agents,
                           note="Loading agents from files instead of agent_characteristics to include supervisor-created agents")

                # Use discovered agents if found, otherwise fallback to agent_characteristics
                agents_to_run = discovered_agents if discovered_agents else list(agent_characteristics.keys())
            except Exception as e:
                logger.warning(f"Failed to discover agents from files, falling back to agent_characteristics", error=str(e))
                agents_to_run = list(agent_characteristics.keys())
        else:
            # No file service, use agent_characteristics
            agents_to_run = list(agent_characteristics.keys())

        # Emit status with ACTUAL agent count
        if stream:
            stream.emit_status(f"Executing {len(agents_to_run)} research agents in parallel...", step="agents")

        agent_tasks = []
        # CRITICAL: Launch ALL agents simultaneously, not sequentially
        # Create all tasks first, then they all run in parallel
        for agent_id in agents_to_run:
            task = asyncio.create_task(
                run_researcher_agent_enhanced(
                    agent_id=agent_id,
                    state=state,
                    llm=llm,
                    search_provider=search_provider,
                    scraper=scraper,
                    stream=stream,
                    supervisor_queue=supervisor_queue,
                    max_steps=agent_max_steps
                )
            )
            agent_tasks.append((agent_id, task))
            logger.info(f"Created task for agent {agent_id} in cycle {iteration_count}",
                       note="Task created, will run in parallel with other agents")

        # CRITICAL: All tasks are now created and running in parallel
        # asyncio.create_task immediately schedules them for execution
        logger.info(f"Launched {len(agent_tasks)} agents in parallel for cycle {iteration_count}",
                   agents=agents_to_run,
                   note="All agents run concurrently via asyncio.create_task - they execute simultaneously, not sequentially")
        
        if stream:
            stream.emit_status(f"🚀 Launched {len(agent_tasks)} agents in parallel for cycle {iteration_count}", step="agents")

        # Collect results from this cycle - process completions as they arrive
        # CRITICAL: Agents complete tasks independently and queue for supervisor immediately
        # We process supervisor queue as agents complete, not waiting for all
        cycle_findings = []
        no_tasks_count = 0
        
        # Use asyncio.as_completed to process agents as they finish
        # This allows supervisor to review immediately when agent completes
        logger.info(f"Waiting for {len(agent_tasks)} agents to complete (processing supervisor queue as agents finish)...")
        
        # Create a mapping of tasks to agent_ids for result processing
        # CRITICAL: Use a list to track which agents have been processed
        tasks_by_agent = {agent_id: task for agent_id, task in agent_tasks}
        processed_agents = set()  # Track which agents we've already processed
        
        # Process agents as they complete using as_completed
        # This processes each agent immediately when it finishes
        completed_count = 0
        pending_tasks_list = [task for _, task in agent_tasks]
        
        # CRITICAL: Track all agent tasks to ensure we process all of them
        logger.info(f"Processing {len(pending_tasks_list)} agent tasks as they complete",
                   agent_ids=agents_to_run)
        
        for completed_coro in asyncio.as_completed(pending_tasks_list):
            try:
                result = await completed_coro
                # Find which agent this result belongs to
                agent_id = None
                
                # First try: get agent_id from result dict
                if isinstance(result, dict):
                    agent_id = result.get("agent_id")
                
                # Second try: find by checking which task is done and not yet processed
                if not agent_id:
                    for agent_id_check, task in tasks_by_agent.items():
                        if agent_id_check in processed_agents:
                            continue  # Skip already processed agents
                        if task.done():
                            try:
                                task_result = task.result()
                                # If task_result has agent_id matching agent_id_check, this is the one
                                if isinstance(task_result, dict):
                                    if task_result.get("agent_id") == agent_id_check:
                                        agent_id = agent_id_check
                                        processed_agents.add(agent_id_check)
                                        break
                            except Exception as e:
                                logger.debug(f"Error checking task result for {agent_id_check}", error=str(e))
                                pass
                
                # Third try: if still not found, check all done tasks and pick first unprocessed
                if not agent_id:
                    for agent_id_check, task in tasks_by_agent.items():
                        if agent_id_check in processed_agents:
                            continue
                        if task.done():
                            agent_id = agent_id_check
                            processed_agents.add(agent_id_check)
                            logger.info(f"Matched completed task to agent {agent_id} by process of elimination")
                            break
                
                if agent_id:
                    completed_count += 1
                    # Mark this agent as processed
                    processed_agents.add(agent_id)
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent_id} failed", error=str(result), exc_info=result)
                    elif result:
                        if result.get("topic") == "no_tasks":
                            no_tasks_count += 1
                            logger.info(f"Agent {agent_id} has no tasks")
                        else:
                            cycle_findings.append(result)
                            all_findings.append(result)
                            logger.info(f"Agent {agent_id} completed task", 
                                      task=result.get("topic", "unknown"),
                                      sources=len(result.get("sources", [])),
                                      completed_agents=f"{completed_count}/{len(agent_tasks)}",
                                      note="Result queued for supervisor, supervisor will review immediately")
                            
                            # CRITICAL: Process supervisor review immediately when agent completes
                            # Don't wait for all agents - supervisor should review EACH agent's work as soon as they complete
                            # This ensures supervisor updates draft_report and manages tasks in real-time
                            # CRITICAL: Supervisor is ALWAYS called for findings processing and draft_report writing
                            # Limit applies ONLY to TODO operations (create_agent_todo, update_agent_todo), NOT to findings processing
                            # CRITICAL: Other agents continue working in parallel - supervisor review doesn't block them
                            
                            # Track call count for TODO operations limit, but ALWAYS call supervisor for findings
                            is_todo_operations_available = supervisor_call_count < max_supervisor_calls
                            
                            if is_todo_operations_available:
                                # Increment counter only for TODO operations tracking
                                supervisor_call_count += 1
                                state["supervisor_call_count"] = supervisor_call_count
                                logger.info(f"Agent {agent_id} completed task - calling supervisor for review (call {supervisor_call_count}/{max_supervisor_calls}, TODO operations available)",
                                          note="Other agents continue working in parallel during supervisor review")
                            else:
                                # Don't increment counter, but STILL call supervisor for findings processing
                                logger.info(f"Agent {agent_id} completed task - calling supervisor for findings processing (TODO limit reached: {supervisor_call_count}/{max_supervisor_calls})",
                                          note="Supervisor will process findings and write to draft_report, but TODO operations are disabled")
                            
                            try:
                                if settings:
                                    supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                                else:
                                    from src.config.settings import get_settings
                                    settings_obj = get_settings()
                                    supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                                
                                if stream:
                                    if is_todo_operations_available:
                                        stream.emit_status(f"👔 Supervisor reviewing findings from {agent_id} (call {supervisor_call_count}/{max_supervisor_calls})", step="supervisor")
                                    else:
                                        stream.emit_status(f"👔 Supervisor processing findings from {agent_id} (TODO limit reached, writing to draft_report)", step="supervisor")
                                
                                # CRITICAL: Supervisor review happens while other agents continue working
                                # This is non-blocking - agents run in parallel via asyncio.create_task
                                # CRITICAL: Ensure deep_search_result is in state before calling supervisor
                                deep_search_result_in_state = state.get("deep_search_result", "")
                                if not deep_search_result_in_state:
                                    logger.warning("deep_search_result missing from state before supervisor call",
                                                 state_keys=list(state.keys())[:20],
                                                 note="Supervisor may not have access to initial deep search context")
                                else:
                                    logger.debug("deep_search_result available in state for supervisor",
                                                has_deep_search=bool(deep_search_result_in_state),
                                                deep_search_type=type(deep_search_result_in_state).__name__)
                                
                                # CRITICAL: Add current finding to state so supervisor can see it
                                # Supervisor reads findings from state, not from cycle_findings
                                if result:
                                    existing_findings = state.get("findings", state.get("agent_findings", []))
                                    # Check if this finding is already in state (avoid duplicates)
                                    finding_already_in_state = any(
                                        f.get("topic") == result.get("topic") and f.get("agent_id") == result.get("agent_id")
                                        for f in existing_findings
                                    )
                                    if not finding_already_in_state:
                                        existing_findings.append(result)
                                        state["findings"] = existing_findings
                                        state["agent_findings"] = existing_findings
                                        logger.info(f"Added finding to state for supervisor",
                                                   finding_topic=result.get("topic", "unknown"),
                                                   total_findings_in_state=len(existing_findings),
                                                   note="Supervisor will now see this finding in state")
                                
                                from src.workflow.research.supervisor_agent import run_supervisor_agent
                                decision = await run_supervisor_agent(
                                    state=state,
                                    llm=llm,
                                    stream=stream,
                                    supervisor_queue=supervisor_queue,
                                    max_iterations=supervisor_max_iterations
                                )
                                
                                state["should_continue"] = decision.get("should_continue", False)
                                state["replanning_needed"] = decision.get("replanning_needed", False)
                                
                                # CRITICAL: After supervisor review, check if supervisor created new tasks
                                # Even if supervisor says stop, we must check for new tasks before stopping
                                # Supervisor might have created new tasks before deciding to stop
                                new_pending_tasks = 0  # Initialize before use
                                if agent_file_service:
                                    try:
                                        # Check if any agents have new pending tasks after supervisor review
                                        agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                                        all_agent_ids = []
                                        for file_path in agent_files:
                                            agent_id = file_path.replace("agents/", "").replace(".md", "")
                                            if agent_id.startswith("agent_") and agent_id != "supervisor":
                                                all_agent_ids.append(agent_id)
                                        
                                        new_pending_tasks = 0
                                        agents_with_new_tasks = []
                                        for agent_id in all_agent_ids:
                                            agent_file = await agent_file_service.read_agent_file(agent_id)
                                            todos = agent_file.get("todos", [])
                                            pending_tasks = [t for t in todos if t.status == "pending"]
                                            in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                                            if pending_tasks or in_progress_tasks:
                                                agents_with_new_tasks.append(agent_id)
                                                new_pending_tasks += len(pending_tasks) + len(in_progress_tasks)
                                        
                                        if new_pending_tasks > 0:
                                            logger.info(f"After supervisor review: {len(agents_with_new_tasks)} agents have {new_pending_tasks} pending/in_progress tasks",
                                                       agents_with_tasks=agents_with_new_tasks,
                                                       total_pending=new_pending_tasks,
                                                       note="Supervisor created new tasks - agents will continue working in next cycle")
                                            # CRITICAL: If supervisor created new tasks, we MUST continue
                                            # Override supervisor's decision to stop if there are new tasks
                                            state["should_continue"] = True
                                            logger.info("Overriding supervisor's stop decision because new tasks were created",
                                                       note="Agents must complete all tasks before finalization")
                                    except Exception as e:
                                        logger.error("Error checking for new tasks after supervisor review", error=str(e), exc_info=True)
                                
                                # CRITICAL: Update status after supervisor review completes
                                # If supervisor decided to continue or created new tasks, show that agents are working
                                if stream:
                                    if state.get("should_continue", False) or new_pending_tasks > 0:
                                        # Supervisor decided to continue or created new tasks - agents are working
                                        if new_pending_tasks > 0:
                                            stream.emit_status(f"🚀 Agents continuing work ({new_pending_tasks} tasks remaining)", step="agents")
                                        else:
                                            stream.emit_status("🚀 Agents continuing research...", step="agents")
                                    elif not decision.get("should_continue", False):
                                        # Supervisor decided to stop
                                        stream.emit_status("✅ Supervisor decided research is complete", step="supervisor")
                                
                                # CRITICAL: Even if supervisor says stop, check if there are pending tasks
                                # Supervisor might have created new tasks before deciding to stop
                                # We need to check pending tasks before actually stopping
                                if not decision.get("should_continue", False) and not state.get("should_continue", False):
                                    logger.info("Supervisor decided to stop", decision_reasoning=decision.get("reasoning", "")[:200])
                                    # Don't break immediately - check for pending tasks first
                                    # The check below will verify if there are pending tasks
                                    # If there are pending tasks, we'll continue despite supervisor's decision
                            except Exception as e:
                                logger.error("Supervisor review failed", error=str(e), exc_info=True)
                else:
                    logger.warning(f"Could not identify agent for completed task", 
                                 result_type=type(result).__name__,
                                 result_keys=list(result.keys()) if isinstance(result, dict) else None,
                                 processed_agents=list(processed_agents),
                                 total_agents=len(agent_tasks),
                                 note="This may indicate an issue with agent task completion tracking")
            except Exception as e:
                logger.error(f"Error processing agent completion", error=str(e), exc_info=e)
                # CRITICAL: Log which agents haven't completed yet
                incomplete_agents = [aid for aid, task in tasks_by_agent.items() if aid not in processed_agents and not task.done()]
                if incomplete_agents:
                    logger.warning(f"Agents not yet completed: {incomplete_agents}", 
                                 total_agents=len(agent_tasks),
                                 completed=len(processed_agents),
                                 incomplete=len(incomplete_agents))
        
        # CRITICAL: Verify all agents were processed
        unprocessed_agents = [aid for aid in agents_to_run if aid not in processed_agents]
        if unprocessed_agents:
            logger.warning(f"Cycle {iteration_count}: Some agents were not processed",
                         unprocessed_agents=unprocessed_agents,
                         processed_count=len(processed_agents),
                         total_agents=len(agent_tasks),
                         note="This may indicate agents are hanging or not completing")
            # Try to get results from unprocessed agents
            for agent_id in unprocessed_agents:
                if agent_id in tasks_by_agent:
                    task = tasks_by_agent[agent_id]
                    if task.done():
                        try:
                            result = task.result()
                            if isinstance(result, dict):
                                agent_id_from_result = result.get("agent_id")
                                if agent_id_from_result == agent_id:
                                    logger.info(f"Found unprocessed result for agent {agent_id}, processing now")
                                    if result.get("topic") == "no_tasks":
                                        no_tasks_count += 1
                                    else:
                                        cycle_findings.append(result)
                                        all_findings.append(result)
                                    processed_agents.add(agent_id)
                                    completed_count += 1
                        except Exception as e:
                            logger.error(f"Error processing unprocessed agent {agent_id}", error=str(e))
        
        logger.info(f"Cycle {iteration_count} complete: {len(cycle_findings)} tasks completed, {no_tasks_count} agents with no tasks, {completed_count}/{len(agent_tasks)} agents processed",
                   processed_agents=list(processed_agents),
                   unprocessed_agents=unprocessed_agents if unprocessed_agents else None)
        
        # CRITICAL: Check if agents have pending tasks after cycle completes
        # This ensures agents continue working if supervisor assigned new tasks
        # IMPORTANT: Check even if agents_active is False - supervisor might have created new tasks before stopping
        if agent_file_service:
            try:
                # Reload agents list in case supervisor created new agents
                agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                all_agent_ids = []
                for file_path in agent_files:
                    agent_id = file_path.replace("agents/", "").replace(".md", "")
                    if agent_id.startswith("agent_") and agent_id != "supervisor":
                        all_agent_ids.append(agent_id)
                
                # Check if any agents have pending tasks
                agents_with_pending_tasks = []
                total_pending = 0
                for agent_id in all_agent_ids:
                    agent_file = await agent_file_service.read_agent_file(agent_id)
                    todos = agent_file.get("todos", [])
                    pending_tasks = [t for t in todos if t.status == "pending"]
                    in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                    if pending_tasks or in_progress_tasks:
                        agents_with_pending_tasks.append(agent_id)
                        total_pending += len(pending_tasks) + len(in_progress_tasks)
                
                if agents_with_pending_tasks:
                    logger.info(f"After cycle {iteration_count}: {len(agents_with_pending_tasks)} agents have {total_pending} pending/in_progress tasks, continuing to next cycle",
                               agents_with_tasks=agents_with_pending_tasks,
                               total_pending_tasks=total_pending,
                               note="Agents will continue working in next cycle - supervisor may have assigned new tasks")
                    # CRITICAL: Even if supervisor decided to stop, if there are pending tasks, we must continue
                    # Supervisor might have created new tasks before deciding to stop
                    agents_active = True
                    logger.info("Reactivating agents because pending tasks found", 
                               pending_tasks=total_pending,
                               agents=agents_with_pending_tasks,
                               note="Supervisor may have created new tasks before stopping - agents must complete them")
                    # Continue to next iteration - agents will pick up their pending tasks
                    # The while loop will continue because agents_active is now True
                else:
                    logger.info(f"After cycle {iteration_count}: no agents have pending tasks")
                    # Check if we should stop
                    if no_tasks_count == len(agent_tasks):
                        logger.info("All agents have no tasks, stopping agent execution")
                        if stream:
                            stream.emit_status("✅ All agents completed their tasks", step="agents")
                        
                        # CRITICAL: When all tasks are done, FORCE supervisor to finalize report
                        # This ensures final report is generated even if supervisor didn't call make_final_decision
                        # MANDATORY: Call supervisor EVEN IF limit reached - this is finalization call
                        logger.info("MANDATORY: All tasks completed - forcing supervisor to finalize report (bypassing call limit if needed)")
                        if stream:
                            stream.emit_status("👔 Supervisor finalizing report...", step="supervisor")
                        
                        # CRITICAL: Always call supervisor for finalization, even if limit reached
                        # This is a special finalization call that bypasses the normal limit
                        try:
                            # Increment counter but don't check limit - this is mandatory finalization
                            supervisor_call_count += 1
                            state["supervisor_call_count"] = supervisor_call_count
                            
                            if settings:
                                supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                            else:
                                from src.config.settings import get_settings
                                settings_obj = get_settings()
                                supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                            
                            logger.info("Calling supervisor for MANDATORY finalization (bypassing call limit)",
                                       call_count=supervisor_call_count,
                                       max_calls=max_supervisor_calls,
                                       note="This is a special finalization call when all tasks are done")
                            
                            from src.workflow.research.supervisor_agent import run_supervisor_agent
                            decision = await run_supervisor_agent(
                                state=state,
                                llm=llm,
                                stream=stream,
                                supervisor_queue=supervisor_queue,
                                max_iterations=supervisor_max_iterations
                            )
                            
                            # Force should_continue to False to trigger report generation
                            state["should_continue"] = False
                            state["replanning_needed"] = False
                            
                            # CRITICAL: Update status after supervisor finalization
                            if stream:
                                stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                            
                            logger.info("Forced supervisor finalization completed", 
                                      decision=decision.get("should_continue"),
                                      note="Research will proceed to report generation")
                        except Exception as e:
                            logger.error("Failed to force supervisor finalization", error=str(e), exc_info=True)
                            # Even if supervisor fails, set should_continue to False to proceed to report
                            state["should_continue"] = False
                            state["replanning_needed"] = False
                        
                        agents_active = False
            except Exception as e:
                logger.error("Error checking agent tasks after cycle", error=str(e), exc_info=True)
                # Continue anyway - don't stop on error
        
        # CRITICAL: Findings are NOT automatically added to draft_report
        # Findings are stored separately and supervisor receives them one-by-one via supervisor_queue
        # Supervisor adds findings to draft_report as structured chapters (one chapter = one finding)
        # This ensures draft_report is a clean research draft, not a dump of raw findings
        if cycle_findings and len(cycle_findings) > 0:
            logger.info("Cycle findings collected - supervisor will process them individually via queue",
                       cycle_findings_count=len(cycle_findings),
                       note="Findings stored separately, supervisor will add them to draft_report as chapters")

        # If all agents report no tasks, stop and force supervisor finalization
        if no_tasks_count == len(agent_tasks):
            logger.info("All agents report no tasks, stopping agent execution", 
                       no_tasks_count=no_tasks_count, total_agents=len(agent_tasks))
            if stream:
                stream.emit_status("✅ All agents completed their tasks", step="agents")
            
            # CRITICAL: When all tasks are done, FORCE supervisor to finalize report
            # MANDATORY: Call supervisor EVEN IF limit reached - this is finalization call
            logger.info("MANDATORY: All tasks completed - forcing supervisor to finalize report (bypassing call limit if needed)")
            if stream:
                stream.emit_status("👔 Supervisor finalizing report...", step="supervisor")
            
            # CRITICAL: Always call supervisor for finalization, even if limit reached
            # This is a special finalization call that bypasses the normal limit
            try:
                # Increment counter but don't check limit - this is mandatory finalization
                supervisor_call_count += 1
                state["supervisor_call_count"] = supervisor_call_count
                
                if settings:
                    supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                else:
                    from src.config.settings import get_settings
                    settings_obj = get_settings()
                    supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                
                logger.info("Calling supervisor for MANDATORY finalization (bypassing call limit)",
                           call_count=supervisor_call_count,
                           max_calls=max_supervisor_calls,
                           note="This is a special finalization call when all tasks are done")
                
                # CRITICAL: Extract ALL findings from supervisor_queue and add to state BEFORE finalization
                # This ensures supervisor sees all findings when finalizing the report
                if supervisor_queue and supervisor_queue.size() > 0:
                    findings_from_queue = []
                    temp_events = []
                    queue_size = supervisor_queue.size()
                    for _ in range(queue_size):
                        try:
                            event = supervisor_queue.queue.get_nowait()
                            temp_events.append(event)
                            if event.result:
                                findings_from_queue.append(event.result)
                        except:
                            break
                    
                    # Put events back in queue (they'll be processed properly by supervisor)
                    for event in temp_events:
                        await supervisor_queue.queue.put(event)
                    
                    # Add findings to state so supervisor can see them
                    if findings_from_queue:
                        existing_findings = state.get("findings", state.get("agent_findings", []))
                        # Combine existing findings with queue findings (avoid duplicates)
                        for new_finding in findings_from_queue:
                            finding_already_exists = any(
                                f.get("topic") == new_finding.get("topic") and 
                                f.get("agent_id") == new_finding.get("agent_id")
                                for f in existing_findings
                            )
                            if not finding_already_exists:
                                existing_findings.append(new_finding)
                        
                        state["findings"] = existing_findings
                        state["agent_findings"] = existing_findings
                        logger.info(f"Extracted {len(findings_from_queue)} findings from supervisor_queue before finalization",
                                   total_findings=len(existing_findings),
                                   note="Supervisor will now see ALL findings when finalizing report")
                
                # Also extract findings from cycle_findings if available
                if cycle_findings and len(cycle_findings) > 0:
                    existing_findings = state.get("findings", state.get("agent_findings", []))
                    for new_finding in cycle_findings:
                        finding_already_exists = any(
                            f.get("topic") == new_finding.get("topic") and 
                            f.get("agent_id") == new_finding.get("agent_id")
                            for f in existing_findings
                        )
                        if not finding_already_exists:
                            existing_findings.append(new_finding)
                    
                    state["findings"] = existing_findings
                    state["agent_findings"] = existing_findings
                    logger.info(f"Added {len(cycle_findings)} findings from cycle_findings to state before finalization",
                               total_findings=len(existing_findings),
                               note="Supervisor will now see ALL findings when finalizing report")
                
                from src.workflow.research.supervisor_agent import run_supervisor_agent
                decision = await run_supervisor_agent(
                    state=state,
                    llm=llm,
                    stream=stream,
                    supervisor_queue=supervisor_queue,
                    max_iterations=supervisor_max_iterations
                )
                
                # Force should_continue to False to trigger report generation
                state["should_continue"] = False
                state["replanning_needed"] = False
                
                # CRITICAL: Update status after supervisor finalization
                if stream:
                    stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                
                logger.info("Forced supervisor finalization completed", 
                          decision=decision.get("should_continue"),
                          total_findings=len(state.get("findings", state.get("agent_findings", []))),
                          note="Research will proceed to report generation")
            except Exception as e:
                logger.error("Failed to force supervisor finalization", error=str(e), exc_info=True)
                # Even if supervisor fails, set should_continue to False to proceed to report
                state["should_continue"] = False
                state["replanning_needed"] = False
            
            agents_active = False
            # CRITICAL: After finalization, we should NOT process queue again - supervisor already finalized
            # Just break and return state with should_continue=False to trigger report generation
            logger.info("All tasks completed and supervisor finalized - breaking to trigger report generation",
                       note="should_continue=False will route to compress_findings -> generate_report")
            break
        
        # After cycle completes, check if there are any remaining items in supervisor queue
        # (Most items should have been processed during the cycle as agents completed)
        queue_size = supervisor_queue.size()
        logger.info(f"After cycle {iteration_count}: supervisor queue size = {queue_size}, cycle findings = {len(cycle_findings)}")
        
        # Process any remaining items in queue (should be rare, as we process during cycle)
        # CRITICAL: Only process queue if agents are still active (not finalized)
        if queue_size > 0 and agents_active:
            # CRITICAL: Supervisor is ALWAYS called for findings processing and draft_report writing
            # Limit applies ONLY to TODO operations, NOT to findings processing
            is_todo_operations_available = supervisor_call_count < max_supervisor_calls
            
            if is_todo_operations_available:
                # Increment counter only for TODO operations tracking
                supervisor_call_count += 1
                state["supervisor_call_count"] = supervisor_call_count
                logger.info(f"Processing {supervisor_queue.size()} agent completions in supervisor queue (call {supervisor_call_count}/{max_supervisor_calls}, TODO operations available)")
            else:
                # Don't increment counter, but STILL call supervisor for findings processing
                logger.info(f"Processing {supervisor_queue.size()} agent completions in supervisor queue (TODO limit reached: {supervisor_call_count}/{max_supervisor_calls})",
                          note="Supervisor will process findings and write to draft_report, but TODO operations are disabled")
            
            if stream:
                if is_todo_operations_available:
                    stream.emit_status(f"👔 Supervisor reviewing findings (call {supervisor_call_count}/{max_supervisor_calls})", step="supervisor")
                else:
                    stream.emit_status(f"👔 Supervisor processing findings (TODO limit reached, writing to draft_report)", step="supervisor")
            
            # CRITICAL: Call supervisor agent to review and process findings
            # Supervisor is ALWAYS called - limit only blocks TODO operations, NOT findings processing
            from src.workflow.research.supervisor_agent import run_supervisor_agent
            
            try:
                
                # Get max_iterations from settings (centralized config)
                if settings:
                    supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
                else:
                    from src.config.settings import get_settings
                    settings_obj = get_settings()
                    supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
                
                # CRITICAL: Extract findings from supervisor_queue and add to state
                # Supervisor needs findings in state to process them
                if supervisor_queue and supervisor_queue.size() > 0:
                    findings_from_queue = []
                    # Get all pending findings from queue (peek without removing)
                    temp_events = []
                    queue_size = supervisor_queue.size()
                    for _ in range(queue_size):
                        try:
                            event = supervisor_queue.queue.get_nowait()
                            temp_events.append(event)
                            if event.result:
                                findings_from_queue.append(event.result)
                        except:
                            break
                    
                    # Put events back in queue (they'll be processed properly by supervisor)
                    for event in temp_events:
                        await supervisor_queue.queue.put(event)
                    
                    # Add findings to state so supervisor can see them
                    if findings_from_queue:
                        existing_findings = state.get("findings", state.get("agent_findings", []))
                        # Combine existing findings with queue findings (avoid duplicates)
                        for new_finding in findings_from_queue:
                            finding_already_exists = any(
                                f.get("topic") == new_finding.get("topic") and 
                                f.get("agent_id") == new_finding.get("agent_id")
                                for f in existing_findings
                            )
                            if not finding_already_exists:
                                existing_findings.append(new_finding)
                        
                        state["findings"] = existing_findings
                        state["agent_findings"] = existing_findings
                        logger.info(f"Extracted {len(findings_from_queue)} findings from supervisor_queue and added to state",
                                   total_findings=len(existing_findings),
                                   note="Supervisor will now see these findings in state")
                
                decision = await run_supervisor_agent(
                    state=state,
                    llm=llm,
                    stream=stream,
                    supervisor_queue=supervisor_queue,  # Pass supervisor_queue
                    max_iterations=supervisor_max_iterations
                )
                
                # Update state with supervisor decision
                state["should_continue"] = decision.get("should_continue", False)
                state["replanning_needed"] = decision.get("replanning_needed", False)
                
                # CRITICAL: Update status after supervisor review completes
                if stream:
                    if not decision.get("should_continue", False):
                        stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
                    else:
                        # Check if there are pending tasks
                        if agent_file_service:
                            try:
                                agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                                all_agent_ids = []
                                for file_path in agent_files:
                                    agent_id = file_path.replace("agents/", "").replace(".md", "")
                                    if agent_id.startswith("agent_") and agent_id != "supervisor":
                                        all_agent_ids.append(agent_id)
                                
                                total_pending = 0
                                for agent_id in all_agent_ids:
                                    agent_file = await agent_file_service.read_agent_file(agent_id)
                                    todos = agent_file.get("todos", [])
                                    pending_tasks = [t for t in todos if t.status == "pending"]
                                    in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                                    total_pending += len(pending_tasks) + len(in_progress_tasks)
                                
                                if total_pending > 0:
                                    stream.emit_status(f"🚀 Agents continuing work ({total_pending} tasks remaining)", step="agents")
                                else:
                                    stream.emit_status("🚀 Research continuing...", step="agents")
                            except Exception as e:
                                logger.warning("Failed to check pending tasks for status update", error=str(e))
                                stream.emit_status("🚀 Research continuing...", step="agents")
                
                # If supervisor says stop, break the loop
                if not decision.get("should_continue", False):
                    logger.info("Supervisor decided to stop, breaking agent execution loop", 
                               decision_reasoning=decision.get("reasoning", "")[:200])
                    agents_active = False
                    break
                
                # Check if we've reached the limit after this call
                if supervisor_call_count >= max_supervisor_calls:
                    logger.warning(f"Supervisor call limit reached after decision ({supervisor_call_count}/{max_supervisor_calls}), agents will complete tasks without supervisor")
                    # Don't break - let agents complete their tasks
                    # Clear queue and continue
                    while not supervisor_queue.queue.empty():
                        try:
                            supervisor_queue.queue.get_nowait()
                            supervisor_queue.queue.task_done()
                        except:
                            break
                    # Continue loop - agents will finish their tasks
                    continue  # Skip further supervisor calls but continue agent execution
                
                # Clear the queue after processing
                while not supervisor_queue.queue.empty():
                    try:
                        supervisor_queue.queue.get_nowait()
                        supervisor_queue.queue.task_done()
                    except:
                        break
                        
                logger.info("Supervisor queue processed", 
                           decision=decision.get("should_continue"),
                           note="Agents will continue working in next cycle if they have pending tasks")
                
                # CRITICAL: After supervisor review, agents should continue working if they have pending tasks
                # The while loop will continue and agents will be launched again in next iteration
                
            except Exception as e:
                logger.error("Supervisor processing failed", error=str(e))
        
        # CRITICAL: DO NOT automatically add findings to draft_report - supervisor should add them as chapters
        # Automatic addition creates duplicate sections ("New Findings") and messes up the structure
        # Supervisor uses write_draft_report to add findings as proper chapters
        # Only finalize draft_report with ALL findings if supervisor limit reached (see below)
        # REMOVED: Automatic findings addition - it was creating duplicate sections and mess

    # CRITICAL: Check if max_iterations reached - if so, MUST stop and force finalization
    if iteration_count >= max_iterations:
        logger.warning(f"MANDATORY: Max iterations reached ({iteration_count}/{max_iterations}) - forcing stop and finalization",
                      note="Research will proceed to report generation even if tasks incomplete")
        agents_active = False
        # Force should_continue to False to trigger report generation
        state["should_continue"] = False
        state["replanning_needed"] = False
        # CRITICAL: Force supervisor finalization even if limit reached
        state["_force_supervisor_finalization"] = True
        if stream:
            stream.emit_status(f"⚠️ Max iterations reached ({iteration_count}/{max_iterations}) - finalizing report", step="agents")
    
    # Add findings to agent_findings (using reducer)
    # Update iteration in state
    new_iteration = current_iteration + iteration_count
    logger.info(f"Agent execution completed", cycles=iteration_count, total_iteration=new_iteration, max_iterations=max_iterations, supervisor_calls=supervisor_call_count, max_reached=(iteration_count >= max_iterations))
    
    # CRITICAL: Supervisor continues to be called even after TODO limit is reached
    # Supervisor can still process findings and write chapters to draft_report
    # Finalization happens only when all tasks are completed, NOT when limit is reached
    agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
    agent_file_service = stream.app_state.get("agent_file_service") if stream else None
    
    # NOTE: We do NOT finalize draft_report here when limit is reached
    # Supervisor will continue to be called for findings processing and will write chapters
    # Finalization happens in generate_final_report_enhanced_node when all tasks are done
    if False:  # Disabled - supervisor continues working after limit
        logger.info("MANDATORY: Supervisor call limit reached - finalizing draft_report with ALL findings", 
                   supervisor_calls=supervisor_call_count, max_calls=max_supervisor_calls, findings_count=len(all_findings))
        
        # CRITICAL: MUST have agent_memory_service to save draft_report
        if agent_memory_service:
            try:
                # Read current draft_report
                try:
                    draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                except FileNotFoundError:
                    draft_content = ""
                
                # Get ALL findings with FULL content (no truncation)
                from datetime import datetime
                query = state.get("query", "")
                
                # Build comprehensive findings section with ALL findings (no truncation)
                findings_sections = []
                for f in all_findings:
                    # Get FULL summary (no truncation)
                    full_summary = f.get('summary', 'No summary')
                    # Get ALL key findings (no truncation)
                    all_key_findings = f.get('key_findings', [])
                    # Get ALL sources info
                    sources = f.get('sources', [])
                    
                    findings_sections.append(f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Confidence:** {f.get('confidence', 'unknown')}

### Summary

{full_summary}

### Key Findings

{chr(10).join([f"- {kf}" for kf in all_key_findings]) if all_key_findings else "No key findings"}

### Sources ({len(sources)})

{chr(10).join([f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources[:20]]) if sources else "No sources"}
""")
                
                findings_text = "\n\n".join(findings_sections)
                
                # CRITICAL: Check if draft_report already has chapters from supervisor
                # If supervisor wrote chapters, DO NOT overwrite them - just append synthesis if needed
                import re
                has_chapters = bool(re.search(r'##\s+Chapter\s+\d+:', draft_content))
                
                if has_chapters:
                    # Supervisor already wrote chapters - DO NOT overwrite!
                    # Only append final synthesis if draft is substantial
                    logger.info("Draft report already contains chapters from supervisor - preserving them",
                              draft_length=len(draft_content),
                              chapters_detected=True,
                              note="Will NOT overwrite supervisor's chapters")
                    # Don't overwrite - supervisor's chapters are already there
                    final_draft = draft_content
                elif len(draft_content) > 500:
                    # Draft exists but no chapters - append synthesis
                    final_draft = f"""{draft_content}

---

## Final Synthesis (After Supervisor Limit)

**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(all_findings)}
**Supervisor Calls:** {supervisor_call_count}/{max_supervisor_calls}

### Complete Findings from All Agents

{findings_text}

### Final Conclusion

Research completed with {len(all_findings)} comprehensive findings from {len(agent_characteristics)} research agents. All agents have completed their tasks and provided full results.
"""
                else:
                    # Draft is empty or too short - create comprehensive draft
                    # BUT: Use Chapter format to match supervisor's format
                    logger.warning("Draft report is empty or too short - creating comprehensive draft with Chapter format",
                                 draft_length=len(draft_content),
                                 note="Using Chapter format to match supervisor's structure")
                    
                    # Create chapters from findings (matching supervisor's format)
                    chapters_text = []
                    for i, f in enumerate(all_findings, 1):
                        full_summary = f.get('summary', 'No summary')
                        all_key_findings = f.get('key_findings', [])
                        sources = f.get('sources', [])
                        topic = f.get('topic', 'Unknown Topic')
                        
                        chapter_content = f"{full_summary}\n\n"
                        if all_key_findings:
                            chapter_content += "### Key Findings\n\n" + "\n".join([f"- {kf}" for kf in all_key_findings]) + "\n\n"
                        if sources:
                            sources_list = [f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources[:30]]
                            chapter_content += "### Sources\n\n" + "\n".join(sources_list) + "\n"
                        
                        chapters_text.append(f"## Chapter {i}: {topic}\n\n{chapter_content}")
                    
                    final_draft = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(all_findings)}
**Supervisor Calls:** {supervisor_call_count}/{max_supervisor_calls}

{chr(10).join(chapters_text)}

## Conclusion

Research completed with {len(all_findings)} findings from multiple agents covering various aspects of the topic. All agents have completed their tasks and provided comprehensive results.
"""
                
                await agent_memory_service.file_manager.write_file("draft_report.md", final_draft)
                logger.info("MANDATORY: Draft report finalized with ALL findings after supervisor limit (no truncation)", 
                          draft_length=len(final_draft), findings_count=len(all_findings), supervisor_calls=supervisor_call_count)
                if stream:
                    stream.emit_status(f"📝 Draft report finalized after supervisor limit ({len(final_draft)} chars, {len(all_findings)} findings)", step="supervisor")
            except Exception as e:
                logger.error("CRITICAL: Failed to finalize draft_report after supervisor limit - this must not happen!", 
                           error=str(e), findings_count=len(all_findings), supervisor_calls=supervisor_call_count)
            # Try to create at least a minimal draft report as last resort
            try:
                minimal_draft = f"""# Research Report

**Query:** {state.get('query', 'Unknown')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(all_findings)}
**Note:** Draft report creation failed, but findings are available.

## Findings Summary

{len(all_findings)} findings collected from research agents.
"""
                await agent_memory_service.file_manager.write_file("draft_report.md", minimal_draft)
                logger.warning("Created minimal draft report as fallback after error", draft_length=len(minimal_draft))
            except Exception as e2:
                logger.error("CRITICAL: Even minimal draft report creation failed! Report generation will use raw findings.", error=str(e2))
        else:
            logger.error("CRITICAL: Cannot finalize draft_report - agent_memory_service not available!", 
                        findings_count=len(all_findings), supervisor_calls=supervisor_call_count)
    
    # CRITICAL: Also ensure draft_report is finalized when all agents complete tasks, even if supervisor limit not reached
    # This ensures all findings are in draft_report regardless of supervisor status
    if all_findings and len(all_findings) > 0 and agent_memory_service:
        try:
            # Check if draft_report exists and contains all findings
            try:
                draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
            except FileNotFoundError:
                draft_content = ""
            
            # Extract topics from draft to check coverage
            existing_topics = set()
            if draft_content:
                import re
                topic_matches = re.findall(r'##\s+([^\n]+)', draft_content)
                existing_topics = {t.strip().lower() for t in topic_matches}
            
            # Check if all findings are in draft
            missing_findings = []
            for f in all_findings:
                topic = f.get('topic', '').lower()
                if topic not in existing_topics:
                    missing_findings.append(f)
            
            # If there are missing findings, supervisor should have added them as chapters
            # CRITICAL: Missing findings should be added as chapters by supervisor, not automatically
            # This block is kept only as fallback if supervisor didn't add them
            if missing_findings:
                logger.warning("Found findings not in draft_report - supervisor should have added them as chapters", 
                           missing_count=len(missing_findings), total_findings=len(all_findings),
                           note="These findings should be added by supervisor using write_draft_report, not automatically")
                # DO NOT add raw findings sections - supervisor should add them as chapters
                # Just log the warning and let supervisor handle it
                pass
        except Exception as e:
            logger.warning("Failed to add missing findings to draft_report", error=str(e), findings_count=len(all_findings))

    # CRITICAL: Multiple safety checks to ensure research ALWAYS completes and generates result
    # 1. Check if max_iterations reached
    # 2. Check if supervisor call limit reached
    # 3. Check if all tasks done
    # ANY of these conditions MUST trigger report generation
    
    final_should_continue = state.get("should_continue", True)
    
    # Safety check 1: Max iterations reached
    if iteration_count >= max_iterations:
        logger.warning(f"MANDATORY: Max iterations reached ({iteration_count}/{max_iterations}) - forcing should_continue=False")
        final_should_continue = False
    
    # Safety check 2: Supervisor call limit reached
    if supervisor_call_count >= max_supervisor_calls:
        logger.warning(f"MANDATORY: Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}) - forcing should_continue=False")
        final_should_continue = False
    
    # Safety check 3: All tasks done
    if not final_should_continue:
        logger.info("should_continue is False - research will proceed to report generation")
    else:
        # Check if all agents really have no tasks
        if agent_file_service:
            try:
                agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
                all_agent_ids = []
                for file_path in agent_files:
                    agent_id = file_path.replace("agents/", "").replace(".md", "")
                    if agent_id.startswith("agent_") and agent_id != "supervisor":
                        all_agent_ids.append(agent_id)
                
                all_agents_have_no_tasks = True
                for agent_id in all_agent_ids:
                    agent_file = await agent_file_service.read_agent_file(agent_id)
                    todos = agent_file.get("todos", [])
                    pending_tasks = [t for t in todos if t.status == "pending"]
                    in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                    if pending_tasks or in_progress_tasks:
                        all_agents_have_no_tasks = False
                        break
                
                if all_agents_have_no_tasks:
                    logger.info("MANDATORY: All agents have no tasks - forcing should_continue=False to trigger report generation")
                    final_should_continue = False
            except Exception as e:
                logger.warning("Could not verify agent tasks status", error=str(e))
    
    # CRITICAL: Final guarantee - if we have findings, we MUST generate report
    # Even if should_continue is True, if we have findings and limits reached, force completion
    if all_findings and len(all_findings) > 0:
        if iteration_count >= max_iterations or supervisor_call_count >= max_supervisor_calls:
            logger.warning(f"MANDATORY: Limits reached but findings exist - forcing completion to generate report",
                          findings_count=len(all_findings),
                          iteration_count=iteration_count,
                          max_iterations=max_iterations,
                          supervisor_calls=supervisor_call_count,
                          max_supervisor_calls=max_supervisor_calls)
            final_should_continue = False
    
    # CRITICAL: If no findings and limits reached, still generate report (even if empty)
    # This ensures user always gets a result, not infinite loop
    if not all_findings or len(all_findings) == 0:
        if iteration_count >= max_iterations or supervisor_call_count >= max_supervisor_calls:
            logger.warning(f"MANDATORY: Limits reached with no findings - forcing completion to generate report (may be empty)",
                          iteration_count=iteration_count,
                          max_iterations=max_iterations,
                          supervisor_calls=supervisor_call_count,
                          max_supervisor_calls=max_supervisor_calls)
            final_should_continue = False
    
    logger.info("Final should_continue decision",
              should_continue=final_should_continue,
              iteration_count=iteration_count,
              max_iterations=max_iterations,
              supervisor_calls=supervisor_call_count,
              max_supervisor_calls=max_supervisor_calls,
              findings_count=len(all_findings),
              note="If False, research will proceed to report generation")
    
    return {
        "agent_findings": all_findings,
        "findings": all_findings,  # Keep for supervisor review
        "findings_count": len(all_findings),
        "iteration": new_iteration,
        "supervisor_call_count": supervisor_call_count,
        "should_continue": final_should_continue,  # CRITICAL: Ensure this is False when limits reached or tasks done
        "replanning_needed": False  # CRITICAL: Don't replan when limits reached or tasks done
    }


# ==================== Supervisor Review (Enhanced) ====================

async def supervisor_review_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor reviews all findings using LangGraph agent with ReAct format.
    
    The supervisor is now a full agent with tools:
    - read_main_document
    - write_main_document  
    - review_agent_progress
    - create_agent_todo
    - make_final_decision
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    agent_file_service = stream.app_state.get("agent_file_service") if stream else None

    # CRITICAL: Check if all agents have no tasks - if so, FORCE supervisor to finalize
    if agent_file_service:
        try:
            agent_files = await agent_file_service.file_manager.list_files("agents/agent_*.md")
            all_agent_ids = []
            for file_path in agent_files:
                agent_id = file_path.replace("agents/", "").replace(".md", "")
                if agent_id.startswith("agent_") and agent_id != "supervisor":
                    all_agent_ids.append(agent_id)
            
            all_agents_have_no_tasks = True
            total_pending = 0
            total_in_progress = 0
            for agent_id in all_agent_ids:
                agent_file = await agent_file_service.read_agent_file(agent_id)
                todos = agent_file.get("todos", [])
                pending_tasks = [t for t in todos if t.status == "pending"]
                in_progress_tasks = [t for t in todos if t.status == "in_progress"]
                if pending_tasks or in_progress_tasks:
                    all_agents_have_no_tasks = False
                    total_pending += len(pending_tasks)
                    total_in_progress += len(in_progress_tasks)
            
            if all_agents_have_no_tasks:
                logger.info("MANDATORY: All agents have no tasks - forcing supervisor to finalize report (bypassing call limit)",
                           agents_checked=len(all_agent_ids),
                           note="Supervisor will be instructed to synthesize all findings and finish - this call bypasses limit")
                if stream:
                    stream.emit_status("👔 All tasks completed - supervisor finalizing report...", step="supervisor")
                # Force should_continue to False before calling supervisor
                state["should_continue"] = False
                state["replanning_needed"] = False
                # CRITICAL: Set flag to bypass supervisor call limit for this finalization call
                state["_force_supervisor_finalization"] = True
            else:
                logger.info("Some agents still have tasks", 
                           agents_with_tasks=len(all_agent_ids) - sum(1 for aid in all_agent_ids if not any(t.status in ["pending", "in_progress"] for t in (await agent_file_service.read_agent_file(aid)).get("todos", []))),
                           total_pending=total_pending,
                           total_in_progress=total_in_progress)
        except Exception as e:
            logger.warning("Could not check agent tasks status in supervisor_review", error=str(e))

    # Use new supervisor agent with ReAct format
    try:
        # Get max_iterations from settings (centralized config)
        settings = state.get("settings")
        if settings:
            supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
        else:
            from src.config.settings import get_settings
            settings_obj = get_settings()
            supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
        
        decision = await run_supervisor_agent(
            state=state,
            llm=llm,
            stream=stream,
            supervisor_queue=None,  # No queue for finalization call
            max_iterations=supervisor_max_iterations
        )
        
        logger.info("Supervisor agent completed", decision=decision)
        
        # CRITICAL: Update status after supervisor review completes
        # If supervisor decided to finish, show finalization status
        if stream:
            if not decision.get("should_continue", False):
                stream.emit_status("✅ Supervisor finalized report - generating final result...", step="supervisor")
            else:
                # Supervisor decided to continue (shouldn't happen in finalization, but handle it)
                stream.emit_status("🚀 Research continuing...", step="agents")
        
        return decision

    except Exception as e:
        logger.error("Supervisor agent failed", error=str(e), exc_info=True)
        
        # CRITICAL: Even if supervisor fails, ensure draft_report is updated with findings
        # This ensures frontend always gets results even if supervisor crashes
        agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
        if agent_memory_service:
            try:
                findings = state.get("findings", state.get("agent_findings", []))
                if findings:
                    # Read current draft_report
                    try:
                        draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                    except FileNotFoundError:
                        draft_content = ""
                    
                    # CRITICAL: Check if draft_report already has chapters from supervisor
                    # If supervisor wrote chapters, DO NOT overwrite them!
                    import re
                    has_chapters = bool(re.search(r'##\s+Chapter\s+\d+:', draft_content))
                    
                    if has_chapters:
                        # Supervisor already wrote chapters - DO NOT overwrite!
                        logger.info("Draft report already contains chapters from supervisor - preserving them after error",
                                  draft_length=len(draft_content),
                                  chapters_detected=True,
                                  note="Will NOT overwrite supervisor's chapters even after error")
                        # Don't overwrite - supervisor's chapters are already there
                    elif len(draft_content) < 500:
                        # Draft is empty or too short AND no chapters - create with Chapter format
                        from datetime import datetime
                        query = state.get("query", "")
                        
                        # Create chapters from findings (matching supervisor's format)
                        chapters_text = []
                        for i, f in enumerate(findings, 1):
                            topic = f.get('topic', 'Unknown Topic')
                            summary = f.get('summary', 'No summary')
                            key_findings = f.get('key_findings', [])
                            
                            chapter_content = f"{summary}\n\n"
                            if key_findings:
                                chapter_content += "### Key Findings\n\n" + "\n".join([f"- {kf}" for kf in key_findings[:10]]) + "\n"
                            
                            chapters_text.append(f"## Chapter {i}: {topic}\n\n{chapter_content}")
                        
                        draft_content = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Note:** Supervisor encountered an error, but findings are available below.

{chr(10).join(chapters_text)}
"""
                        await agent_memory_service.file_manager.write_file("draft_report.md", draft_content)
                        logger.info("Updated draft_report with findings in Chapter format after supervisor error", 
                                  findings_count=len(findings),
                                  note="Used Chapter format to match supervisor's structure")
            except Exception as e2:
                logger.warning("Failed to update draft_report after supervisor error", error=str(e2))
        
        # Fallback: stop research but return findings
        return {
            "should_continue": False,
            "replanning_needed": False,
            "gaps_identified": [],
            "iteration": state.get("iteration", 0) + 1,
            "completion_criteria_met": True
        }


# ==================== Compress Findings Node ====================

async def compress_findings_node(state: ResearchState) -> Dict:
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    """Compress all findings before final report."""
    findings = state.get("agent_findings", [])
    stream = state.get("stream")

    if not findings:
        return {"compressed_research": {"type": "override", "value": ""}}

    if stream:
        stream.emit_status("Compressing findings...", step="compression")

    findings_text = "\n\n".join([
        f"### {f.get('topic')}\n{f.get('summary', '')}"
        for f in findings if f
    ])

    system_prompt = """Compress research findings into a coherent synthesis.

Aim for 800-1200 words. Identify key themes and important sources.
Return JSON with: reasoning, compressed_summary, key_themes, important_sources
"""

    try:
        llm = state.get("llm")
        structured_llm = llm.with_structured_output(CompressedFindings, method="function_calling")

        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=findings_text)
        ])

        logger.info("Findings compressed", length=len(result.compressed_summary))

        if stream:
            stream.emit_compression({"message": "Findings compressed"})

        return {
            "compressed_research": {"type": "override", "value": result.compressed_summary},
        }

    except Exception as e:
        logger.error("Compression failed", error=str(e))
        return {"compressed_research": {"type": "override", "value": findings_text}}


# ==================== Final Report Generation (Enhanced) ====================

async def generate_final_report_enhanced_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final research report with validation.

    Uses structured output for well-organized report.
    """
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    query = state.get("query", "")
    # Use findings from execute_agents, fallback to agent_findings
    findings = state.get("findings", state.get("agent_findings", []))
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")

    if stream:
        stream.emit_status("📄 Generating final report from draft...", step="report")
        logger.info("Starting final report generation", findings_count=len(findings))

    # Get memory service to read draft report (supervisor's working document)
    agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
    draft_report = ""
    main_document = ""
    
    if agent_memory_service:
        try:
            # Read draft_report.md - this is what supervisor wrote
            draft_report_raw = await agent_memory_service.file_manager.read_file("draft_report.md")
            logger.info("Read draft report", length=len(draft_report_raw))
            
            # CRITICAL: Clean up draft_report - remove duplicates, "New Findings" sections, and create proper structure
            # Extract unique chapters and create clean structure
            import re
            from collections import OrderedDict
            
            # Extract all chapters (## Chapter N: Title)
            # CRITICAL: Also match single # Chapter format to catch duplicates
            chapter_pattern = r'##\s+Chapter\s+(\d+):\s+([^\n]+)'
            single_hash_pattern = r'#\s+Chapter\s+(\d+):\s+([^\n]+)'
            chapters = OrderedDict()
            current_chapter = None
            current_content = []
            
            lines = draft_report_raw.split('\n')
            in_new_findings = False
            seen_chapter_titles = set()  # Track seen chapter titles to prevent duplicates
            
            for i, line in enumerate(lines):
                # Skip "New Findings" sections
                if "## New Findings" in line or "## 🔍 RAW FINDINGS" in line:
                    in_new_findings = True
                    continue
                if in_new_findings and (line.startswith("## ") or line.startswith("# ")):
                    in_new_findings = False
                
                if in_new_findings:
                    continue
                
                # Check if this is a chapter header (## Chapter N: Title or # Chapter N: Title)
                match = re.match(chapter_pattern, line)
                if not match:
                    match = re.match(single_hash_pattern, line)
                
                if match:
                    chapter_num = int(match.group(1))
                    chapter_title = match.group(2).strip()
                    chapter_key = f"{chapter_num}:{chapter_title.lower()}"
                    
                    # CRITICAL: Check for duplicates - if we've seen this chapter before, skip it
                    if chapter_key in chapters or chapter_title.lower() in seen_chapter_titles:
                        logger.warning("Skipping duplicate chapter",
                                     chapter_number=chapter_num,
                                     chapter_title=chapter_title,
                                     note="This chapter was already added - skipping duplicate")
                        # Reset current_chapter to skip this duplicate
                        current_chapter = None
                        current_content = []
                        continue
                    
                    # Save previous chapter if exists
                    if current_chapter:
                        prev_chapter_key = f"{current_chapter['number']}:{current_chapter['title'].lower()}"
                        if prev_chapter_key not in chapters:
                            chapters[prev_chapter_key] = {
                                'number': current_chapter['number'],
                                'title': current_chapter['title'],
                                'content': '\n'.join(current_content).strip()
                            }
                            seen_chapter_titles.add(current_chapter['title'].lower())
                    
                    # Start new chapter
                    current_chapter = {
                        'number': chapter_num,
                        'title': chapter_title
                    }
                    current_content = []
                elif current_chapter:
                    # Skip metadata lines like "**Added:**", "**Agent:**", "**Confidence:**", "**Updated:**", "**Finding ID:**"
                    # Also skip raw findings sections like "### Summary", "### Key Findings" (these are from raw findings, not chapters)
                    if line.strip().startswith("**Added:**") or \
                       line.strip().startswith("**Agent:**") or \
                       line.strip().startswith("**Confidence:**") or \
                       line.strip().startswith("**Updated:**") or \
                       line.strip().startswith("**Finding ID:**") or \
                       "Finding ID:" in line or \
                       "agent_1.done" in line or \
                       "agent_2.done" in line or \
                       (line.strip().startswith("### Summary") and "Detailed Research Findings" in '\n'.join(current_content[-5:])) or \
                       (line.strip().startswith("### Key Findings") and "Detailed Research Findings" in '\n'.join(current_content[-5:])) or \
                       (line.strip().startswith("### Sources") and "Detailed Research Findings" in '\n'.join(current_content[-5:])) or \
                       "Detailed Research Findings:" in line:
                        # Skip these metadata and raw findings lines
                        continue
                    # Skip empty lines that are just separators
                    if line.strip() == "" and len(current_content) > 0 and current_content[-1].strip() == "":
                        continue
                    current_content.append(line)
            
            # Save last chapter
            if current_chapter:
                chapter_key = f"{current_chapter['number']}:{current_chapter['title'].lower()}"
                if chapter_key not in chapters and current_chapter['title'].lower() not in seen_chapter_titles:
                    chapters[chapter_key] = {
                        'number': current_chapter['number'],
                        'title': current_chapter['title'],
                        'content': '\n'.join(current_content).strip()
                    }
                    seen_chapter_titles.add(current_chapter['title'].lower())
            
            # Build clean draft report with unique chapters
            if chapters:
                query = state.get("query", "Unknown query")
                from datetime import datetime
                
                chapters_text = []
                chapter_num = 1
                # CRITICAL: Sort chapters by original number to preserve order, but renumber sequentially
                sorted_chapters = sorted(chapters.items(), key=lambda x: x[1]['number'])
                for chapter_key, chapter_data in sorted_chapters:
                    # CRITICAL: Clean up sources sections within chapter content - remove duplicate "## Sources" sections
                    chapter_content = chapter_data['content']
                    
                    # Find all "## Sources" sections in this chapter
                    import re
                    sources_pattern = r'##\s+Sources\s*\n\n(.*?)(?=\n\n##\s+Chapter|\Z)'
                    sources_matches = list(re.finditer(sources_pattern, chapter_content, re.DOTALL))
                    
                    if len(sources_matches) > 1:
                        # Multiple sources sections found - deduplicate and keep only the last one
                        logger.warning("Found multiple Sources sections in chapter - deduplicating",
                                     chapter_title=chapter_data['title'],
                                     sources_sections_count=len(sources_matches))
                        
                        # Extract all unique sources from all sections
                        seen_source_urls = set()
                        all_sources = []
                        
                        for match in sources_matches:
                            sources_text = match.group(1)
                            # Extract individual source lines
                            source_lines = [line.strip() for line in sources_text.split('\n') if line.strip() and line.strip().startswith('-')]
                            for line in source_lines:
                                # Extract URL from markdown link format: - [Title](URL)
                                url_match = re.search(r'\(([^)]+)\)', line)
                                if url_match:
                                    url = url_match.group(1).lower().rstrip('/')
                                    if url not in seen_source_urls:
                                        seen_source_urls.add(url)
                                        all_sources.append(line)
                                else:
                                    # No URL, check by title
                                    title_match = re.search(r'\[([^\]]+)\]', line)
                                    if title_match:
                                        title = title_match.group(1).lower()
                                        if title not in [s.split(']')[0].replace('- [', '').lower() for s in all_sources]:
                                            all_sources.append(line)
                                    else:
                                        # Plain text source
                                        if line not in all_sources:
                                            all_sources.append(line)
                        
                        # Remove all sources sections and add single deduplicated one at the end
                        for match in reversed(sources_matches):
                            chapter_content = chapter_content[:match.start()] + chapter_content[match.end():]
                        
                        # Add single deduplicated sources section at the end
                        if all_sources:
                            chapter_content = chapter_content.rstrip() + f"\n\n## Sources\n\n" + "\n".join(all_sources) + "\n"
                    
                    chapters_text.append(f"""## Chapter {chapter_num}: {chapter_data['title']}

{chapter_content}
""")
                    chapter_num += 1
                
                # CRITICAL: Simple, clean structure - just chapters, no metadata, no overview, no "Research Report Draft" header
                # This will be used directly as final report, so no metadata needed
                draft_report = f"""{''.join(chapters_text)}
"""
                logger.info("Cleaned draft report", 
                          original_length=len(draft_report_raw),
                          cleaned_length=len(draft_report),
                          chapters_count=len(chapters),
                          removed_duplicates=True)
            else:
                # No chapters found - check if original has substantial content
                # CRITICAL: If original is only initial structure or very short, it means supervisor didn't write chapters
                # But if original has substantial content (even without "Chapter" format), use it
                initial_structure_indicators = [
                    "## Overview",
                    "This is the working draft",
                    "Status: In Progress",
                    "**Started:**"
                ]
                has_only_initial_structure = any(indicator in draft_report_raw for indicator in initial_structure_indicators) and len(draft_report_raw) < 500
                
                if has_only_initial_structure:
                    # Only initial structure found - supervisor didn't write chapters
                    logger.warning("No chapters found in draft_report and only initial structure present - supervisor may not have called write_draft_report",
                                 original_length=len(draft_report_raw),
                                 findings_count=len(findings),
                                 note="Will create draft from findings if available")
                    draft_report = ""  # Mark as empty to trigger fallback
                else:
                    # Original has content even without "Chapter" format - use it
                    draft_report = draft_report_raw
                    logger.warning("No chapters found in draft_report but original has content, using original", 
                                 original_length=len(draft_report_raw),
                                 note="Draft may not follow Chapter format but has content")
            
            # CRITICAL: Check if draft report is properly filled
            # If draft report is too short or empty, create it from all findings
            # BUT: Only do fallback if draft_report is truly empty or very short
            # If supervisor wrote content but it's just under 1000 chars, that's still valid
            if not draft_report or len(draft_report.strip()) < 500:
                # CRITICAL: This is a fallback - supervisor should have written to draft_report
                logger.error("CRITICAL: Draft report is too short or empty - FALLBACK TRIGGERED!", 
                           draft_length=len(draft_report) if draft_report else 0,
                           findings_count=len(findings),
                           original_length=len(draft_report_raw) if 'draft_report_raw' in locals() else 0,
                           chapters_found=len(chapters) if 'chapters' in locals() else 0,
                           note="This indicates supervisor may not have called write_draft_report or wrote insufficient content")
                logger.warning("Creating comprehensive draft from all findings as fallback", 
                             draft_length=len(draft_report) if draft_report else 0, 
                             findings_count=len(findings))
                
                # Create comprehensive draft report from ALL findings (no truncation)
                from datetime import datetime
                query = state.get("query", "Unknown query")
                
                findings_sections = []
                for f in findings:
                    # CRITICAL: Extract ALL information from finding - no truncation, no loss
                    full_summary = f.get('summary', 'No summary')
                    all_key_findings = f.get('key_findings', [])
                    sources = f.get('sources', [])
                    # Get additional fields that might contain important information
                    detailed_analysis = f.get('detailed_analysis', '')
                    methodology = f.get('methodology', '')
                    conclusions = f.get('conclusions', '')
                    topic = f.get('topic', 'Unknown Topic')
                    agent_id = f.get('agent_id', 'unknown')
                    confidence = f.get('confidence', 'unknown')
                    
                    # Build comprehensive section with ALL available information
                    section_parts = [f"## {topic}"]
                    
                    # Add metadata
                    section_parts.append(f"\n**Agent:** {agent_id}")
                    section_parts.append(f"**Confidence:** {confidence}")
                    
                    # Add summary (full, not truncated)
                    section_parts.append(f"\n### Summary\n\n{full_summary}")
                    
                    # Add detailed analysis if available
                    if detailed_analysis and detailed_analysis.strip():
                        section_parts.append(f"\n### Detailed Analysis\n\n{detailed_analysis}")
                    
                    # Add methodology if available
                    if methodology and methodology.strip():
                        section_parts.append(f"\n### Methodology\n\n{methodology}")
                    
                    # Add key findings (all of them, not truncated)
                    if all_key_findings:
                        # CRITICAL: Extract list comprehension outside f-string to avoid syntax errors
                        key_findings_list = [f"- {kf}" for kf in all_key_findings]
                        section_parts.append(f"\n### Key Findings\n\n{chr(10).join(key_findings_list)}")
                    else:
                        section_parts.append("\n### Key Findings\n\nNo key findings listed")
                    
                    # Add conclusions if available
                    if conclusions and conclusions.strip():
                        section_parts.append(f"\n### Conclusions\n\n{conclusions}")
                    
                    # Add sources (all of them, not just first 20)
                    if sources:
                        # CRITICAL: Extract list comprehension outside f-string to avoid syntax errors
                        sources_list = [f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources]
                        section_parts.append(f"\n### Sources ({len(sources)})\n\n{chr(10).join(sources_list)}")
                    else:
                        section_parts.append("\n### Sources\n\nNo sources listed")
                    
                    findings_sections.append("\n".join(section_parts) + "\n")
                
                findings_text = "\n\n".join(findings_sections)
                
                # Create comprehensive draft
                draft_report = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(findings)}

## Executive Summary

This report synthesizes findings from research agents working on: {query}

## Detailed Findings (Complete, No Truncation)

{findings_text}

## Conclusion

Research completed with {len(findings)} findings from multiple agents covering various aspects of the topic. All agents have completed their tasks and provided comprehensive results.
"""
                
                # Save the created draft report
                await agent_memory_service.file_manager.write_file("draft_report.md", draft_report)
                logger.info("Created comprehensive draft report from all findings", 
                          draft_length=len(draft_report), findings_count=len(findings))
            else:
                logger.info("Draft report contains comprehensive research", draft_length=len(draft_report))
        except FileNotFoundError:
            # CRITICAL: Draft report doesn't exist - this means supervisor NEVER called write_draft_report
            # This is a serious error - supervisor should have written to draft_report
            logger.error("CRITICAL: Draft report not found - supervisor never created draft_report.md!", 
                       findings_count=len(findings),
                       note="This indicates supervisor did not call write_draft_report - this should not happen!")
            logger.warning("MANDATORY: creating comprehensive draft from all findings as fallback", findings_count=len(findings))
            if stream:
                stream.emit_status("Creating draft report from all findings...", step="report")
            
            # MANDATORY: Create comprehensive draft report from ALL findings (no truncation)
            from datetime import datetime
            query = state.get("query", "Unknown query")
            
            findings_sections = []
            for f in findings:
                # CRITICAL: Extract ALL information from finding - no truncation, no loss
                full_summary = f.get('summary', 'No summary')
                all_key_findings = f.get('key_findings', [])
                sources = f.get('sources', [])
                # Get additional fields that might contain important information
                detailed_analysis = f.get('detailed_analysis', '')
                methodology = f.get('methodology', '')
                conclusions = f.get('conclusions', '')
                topic = f.get('topic', 'Unknown Topic')
                agent_id = f.get('agent_id', 'unknown')
                confidence = f.get('confidence', 'unknown')
                
                # Build comprehensive section with ALL available information
                section_parts = [f"## {topic}"]
                
                # Add metadata
                section_parts.append(f"\n**Agent:** {agent_id}")
                section_parts.append(f"**Confidence:** {confidence}")
                
                # Add summary (full, not truncated)
                section_parts.append(f"\n### Summary\n\n{full_summary}")
                
                # Add detailed analysis if available
                if detailed_analysis and detailed_analysis.strip():
                    section_parts.append(f"\n### Detailed Analysis\n\n{detailed_analysis}")
                
                # Add methodology if available
                if methodology and methodology.strip():
                    section_parts.append(f"\n### Methodology\n\n{methodology}")
                
                # Add key findings (all of them, not truncated)
                if all_key_findings:
                    # CRITICAL: Extract list comprehension outside f-string to avoid syntax errors
                    key_findings_list = [f"- {kf}" for kf in all_key_findings]
                    section_parts.append(f"\n### Key Findings\n\n{chr(10).join(key_findings_list)}")
                else:
                    section_parts.append("\n### Key Findings\n\nNo key findings listed")
                
                # Add conclusions if available
                if conclusions and conclusions.strip():
                    section_parts.append(f"\n### Conclusions\n\n{conclusions}")
                
                # Add sources (all of them, not just first 20)
                if sources:
                    # CRITICAL: Extract list comprehension outside f-string to avoid syntax errors
                    sources_list = [f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources]
                    section_parts.append(f"\n### Sources ({len(sources)})\n\n{chr(10).join(sources_list)}")
                else:
                    section_parts.append("\n### Sources\n\nNo sources listed")
                
                findings_sections.append("\n".join(section_parts) + "\n")
            
            findings_text = "\n\n".join(findings_sections)
            
            # Create comprehensive draft
            draft_report = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(findings)}

## Executive Summary

This report synthesizes findings from research agents working on: {query}

## Detailed Findings (Complete, No Truncation)

{findings_text}

## Conclusion

Research completed with {len(findings)} findings from multiple agents covering various aspects of the topic. All agents have completed their tasks and provided comprehensive results.
"""
            
            # Save the created draft report
            await agent_memory_service.file_manager.write_file("draft_report.md", draft_report)
            logger.info("MANDATORY: Created comprehensive draft report from all findings (file was missing)", 
                      draft_length=len(draft_report), findings_count=len(findings))
        except Exception as e:
            logger.warning("Could not read draft report", error=str(e))
        
        # Also read main.md for additional context (key insights only)
        try:
            main_document = await agent_memory_service.read_main_file()
            logger.info("Read main document", length=len(main_document))
        except Exception as e:
            logger.warning("Could not read main document", error=str(e))

    # Compile all findings (NO TRUNCATION - use ALL key findings)
    findings_text = "\n\n".join([
        f"### {f.get('topic')}\n{f.get('summary', '')}\n\nKey findings:\n" +
        "\n".join([f"- {kf}" for kf in f.get('key_findings', [])])  # NO [:5] truncation - use ALL findings
        for f in findings
    ])

    # CRITICAL: Check if draft_report is incomplete (too short or missing chapters)
    # If so, use chapter summaries from session_metadata as fallback
    use_chapter_summaries = False
    chapter_summaries_text = ""
    
    if (not draft_report or len(draft_report) < 2000) and stream:
        try:
            session_id = state.get("session_id")
            session_factory = stream.app_state.get("session_factory")
            if session_id and session_factory:
                from src.workflow.research.session.manager import SessionManager
                session_manager = SessionManager(session_factory)
                session_data = await session_manager.get_session(session_id)
                
                if session_data:
                    metadata = session_data.get("session_metadata", {})
                    chapter_summaries = metadata.get("chapter_summaries", [])
                    
                    if chapter_summaries and len(chapter_summaries) > 0:
                        logger.info("Draft report incomplete - using chapter summaries from session_metadata",
                                   draft_length=len(draft_report) if draft_report else 0,
                                   chapters_count=len(chapter_summaries))
                        
                        # Build comprehensive synthesis from chapter summaries
                        chapter_parts = []
                        for ch in chapter_summaries:
                            chapter_parts.append(f"""## Chapter {ch.get('chapter_number', '?')}: {ch.get('chapter_title', 'Unknown')}

**Agent:** {ch.get('agent_id', 'unknown')}
**Topic:** {ch.get('topic', 'Unknown')}
**Added:** {ch.get('timestamp', 'Unknown')}

### Summary

{ch.get('summary', 'No summary')}

### Key Findings

{chr(10).join([f"- {kf}" for kf in ch.get('key_findings', [])]) if ch.get('key_findings') else "No key findings"}

### Content Preview

{ch.get('content_preview', '')}

**Sources:** {ch.get('sources_count', 0)}
""")
                        
                        chapter_summaries_text = "\n\n".join(chapter_parts)
                        use_chapter_summaries = True
                        
                        logger.info("Chapter summaries prepared for fallback synthesis",
                                   chapters_count=len(chapter_summaries),
                                   total_length=len(chapter_summaries_text))
        except Exception as e:
            logger.warning("Failed to get chapter summaries from session_metadata", error=str(e))
    
    # Use draft_report as primary source, fallback to chapter summaries, then main_document, then findings
    primary_source = draft_report if draft_report else None
    if not primary_source and use_chapter_summaries and chapter_summaries_text:
        # Fallback to chapter summaries - create unified synthesis
        from datetime import datetime
        primary_source = f"""# Research Report Draft - Synthesized from Chapter Summaries

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Note:** Draft report was incomplete. Using comprehensive synthesis from chapter summaries.

## Executive Summary

This report synthesizes findings from {len(chapter_summaries) if 'chapter_summaries' in locals() else 'multiple'} research chapters covering various aspects of: {query}

## Chapters

{chapter_summaries_text}

## Conclusion

Research completed with {len(chapter_summaries) if 'chapter_summaries' in locals() else 'multiple'} chapters from multiple agents covering various aspects of the topic.
"""
        logger.info("Created draft report from chapter summaries", 
                   chapters_count=len(chapter_summaries) if 'chapter_summaries' in locals() else 0,
                   draft_length=len(primary_source))
    elif not primary_source:
        primary_source = main_document if main_document else findings_text
        if not primary_source:
            logger.warning("CRITICAL: No draft report, chapter summaries, main document, or findings - creating minimal draft")
            from datetime import datetime
            primary_source = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Note:** Draft report was not created by supervisor. Using findings directly.

## Findings

{findings_text if findings_text else "No findings available - research may have failed or been incomplete."}
"""
    
    # Build main document section separately (avoid backslash in f-string expression)
    main_doc_section = ""
    if main_document and main_document != primary_source:
        # Use more of main document (up to 10000 chars for context)
        main_doc_preview = main_document[:10000] if len(main_document) > 10000 else main_document
        if len(main_document) > 10000:
            main_doc_preview += f"\n\n[... {len(main_document) - 10000} more characters in main document]"
        main_doc_section = f"\nMain document (key insights):\n{main_doc_preview}\n\n"

    # CRITICAL: Use FULL draft_report (no truncation) - it's the supervisor's comprehensive synthesis
    # CRITICAL: For final report generation, we MUST use the ENTIRE draft_report to ensure no information is lost
    # Even if draft_report is very large, we'll use it fully (or intelligent summarization if absolutely necessary)
    draft_report_for_prompt = primary_source
    
    # Log draft report size for monitoring
    logger.info("Draft report for final report generation",
               draft_length=len(primary_source),
               has_synthesized_section="SUPERVISOR SYNTHESIZED REPORT" in primary_source,
               has_raw_findings="RAW FINDINGS" in primary_source,
               note="Using FULL draft_report to ensure comprehensive final report")
    
    # Only truncate if absolutely necessary (very large >100k chars)
    # For most cases, use full draft_report to ensure comprehensive final report
    if len(primary_source) > 100000:
        # For extremely large draft reports, use first 80000 chars + summary of rest
        from src.utils.text import summarize_text
        draft_report_for_prompt = primary_source[:80000] + "\n\n[... additional content summarized below (use this summary to ensure nothing is missed) ...]\n\n" + summarize_text(primary_source[80000:], 20000)
        logger.warning("Draft report is extremely large - using first 80k chars + detailed summary for prompt", 
                   total_length=len(primary_source), prompt_length=len(draft_report_for_prompt),
                   note="This is rare - most draft reports should be under 100k chars")
    elif len(primary_source) > 50000:
        # For very large draft reports, use first 45000 chars + detailed summary of rest
        from src.utils.text import summarize_text
        draft_report_for_prompt = primary_source[:45000] + "\n\n[... additional content summarized below (use this summary to ensure nothing is missed) ...]\n\n" + summarize_text(primary_source[45000:], 15000)
        logger.info("Draft report is very large - using first 45k chars + detailed summary for prompt", 
                   total_length=len(primary_source), prompt_length=len(draft_report_for_prompt))
    else:
        # For normal/large draft reports, use FULL content - no truncation
        logger.info("Using FULL draft_report in prompt (no truncation)", length=len(primary_source))

    # Get user language from state (detected in create_research_state)
    user_language = state.get("user_language", "English")
    logger.info(f"Using user language for final report", language=user_language)
    
    prompt = f"""Generate a comprehensive, detailed final research report. This should be a full, substantial research document, not a brief summary.

Query: {query}

Language Requirement:
- Write the entire report in {user_language}
- Match the user's query language exactly
- This applies to ALL sections: executive summary, all report sections, and conclusion

Primary Source:
- Use the Draft Research Report as the PRIMARY source - it contains the supervisor's comprehensive synthesis of all agent findings
- Generate a full, detailed report with extensive analysis, multiple sections, and comprehensive coverage
- The report should be substantial and complete (aim for 3000-8000+ words total)

Draft Research Report (supervisor's working document - PRIMARY SOURCE):
{draft_report_for_prompt}
{main_doc_section}
Additional findings (for reference if draft is incomplete - use ALL of this):
{findings_text[:20000] if len(findings_text) > 20000 else findings_text if findings_text else "None"}
{("... (truncated " + str(len(findings_text) - 20000) + " more characters of findings - use draft report as primary source)") if len(findings_text) > 20000 else ""}

Report Structure:
1. Executive summary (400-600 words) - synthesize the draft report's main points, should be substantial, not brief
2. Multiple detailed sections (500-1200 words each) - use ALL sections and findings from draft report, each section should have specific facts, data, and evidence
3. Evidence-based conclusions (400-600 words) - based on draft report's analysis, should be substantial, not brief
4. Citations to sources - include ALL sources mentioned in draft report, list ALL sources, not just a few

Content Requirements:
- Extract and include ALL detailed information from the draft report
- If draft report has "SUPERVISOR SYNTHESIZED REPORT" section, use that as the PRIMARY content source
- If draft report has "RAW FINDINGS" sections, synthesize them into structured sections
- Include ALL technical details, specifications, dates, numbers, and facts from the draft report
- Do NOT summarize or condense - expand and elaborate on the draft report content
- The final report should be MORE comprehensive than the draft, not less
- Create at least 4-7 major sections covering different aspects of the topic
- Use ALL information from the draft report - don't skip or summarize too much
- Each section should be detailed with specific facts, data, and analysis (300-1000 words per section)
- Include ALL key findings from all agents - use EVERY finding from the draft report
- The draft report should be your PRIMARY source - it contains the supervisor's comprehensive synthesis
- If draft report is incomplete or missing, use fallback sources but note this in the report
- Do NOT create a brief summary - create a FULL research document with extensive detail
- Include ALL information from the SUPERVISOR SYNTHESIZED REPORT section if present
- Include ALL RAW FINDINGS if synthesized report is incomplete
- The final report should be LONGER and MORE DETAILED than the draft report, not shorter
"""

    try:
        report = await llm.with_structured_output(FinalReport).ainvoke([
            {"role": "system", "content": "You are an expert research report writer."},
            {"role": "user", "content": prompt}
        ])

        # Validate report
        # Use summarize_text for intelligent truncation, not hard cut
        from src.utils.text import summarize_text
        exec_summary_preview = summarize_text(report.executive_summary, 500)
        conclusion_preview = summarize_text(report.conclusion, 300)
        
        validation_prompt = f"""Validate this research report for quality.

Query: {query}

Report title: {report.title}
Executive summary: {exec_summary_preview}
Sections: {len(report.sections)}
Conclusion: {conclusion_preview}

Check for completeness, accuracy, and quality.
"""

        validation = await llm.with_structured_output(ReportValidation).ainvoke([
            {"role": "system", "content": "You are a research quality auditor."},
            {"role": "user", "content": validation_prompt}
        ])

        # Check if report is comprehensive enough BEFORE formatting
        total_content_length = len(report.executive_summary) + sum(len(s.content) for s in report.sections) + len(report.conclusion)
        logger.info(
            "Report generated and validated",
            is_complete=validation.is_complete,
            quality_score=validation.quality_score,
            sections=len(report.sections),
            total_length=total_content_length,
            executive_summary_length=len(report.executive_summary),
            conclusion_length=len(report.conclusion)
        )
        
        if total_content_length < 1500:
            logger.warning("CRITICAL: Final report is too short! Report may be incomplete.", 
                         total_length=total_content_length, sections=len(report.sections))
            if stream:
                stream.emit_status("⚠️ Warning: Report may be incomplete - check draft_report.md for full content", step="report")

        # Format as markdown - include ALL sections with FULL content (no truncation)
        report_markdown = f"""# {report.title}

## Executive Summary

{report.executive_summary}

"""
        for section in report.sections:
            report_markdown += f"""## {section.title}

{section.content}

"""

        report_markdown += f"""## Conclusion

{report.conclusion}

---

*Confidence Level: {report.confidence_level}*
*Research Quality Score: {validation.quality_score}/10*
"""

        # Stream report in chunks
        if stream:
            logger.info("Streaming final report", report_length=len(report_markdown))
            chunk_size = 500
            for i in range(0, len(report_markdown), chunk_size):
                chunk = report_markdown[i:i+chunk_size]
                stream.emit_report_chunk(chunk)
                await asyncio.sleep(0.02)  # Small delay for smooth streaming
            logger.info("Final report chunks streamed", total_chunks=(len(report_markdown) + chunk_size - 1) // chunk_size)

        logger.info("Returning final report from node", report_length=len(report_markdown))
        return {
            "final_report": report_markdown,
            "confidence": report.confidence_level
        }

    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        # Fallback - use draft_report if available, otherwise construct from main_document and findings
        if draft_report and len(draft_report) > 1000:
            # Use draft_report directly as it contains supervisor's synthesis
            fallback_report = draft_report
            logger.info("Using draft_report.md directly as final report (generation failed)")
        elif main_document and len(main_document) > 500:
            fallback_report = f"""# Research Report: {query}

## Overview

This report is based on the main research document due to report generation failure.

{main_document}

## Additional Findings

{findings_text if findings_text else "No additional findings"}
"""
        else:
            # Last resort: use findings only - CRITICAL: Always return something
            from datetime import datetime
            fallback_report = f"""# Research Report: {query}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Note:** Report generation failed. This is a fallback report from findings.

## Findings

{findings_text if findings_text else "No findings available - research may have failed or been incomplete."}

## Summary

This report was generated as a fallback due to report generation failure. Please check draft_report.md for supervisor's synthesis if available.
"""
        
        logger.warning("Using fallback report", source="draft_report" if draft_report else "main_document" if main_document else "findings")
        
        # CRITICAL: Ensure fallback report is never empty
        if not fallback_report or len(fallback_report.strip()) < 50:
            logger.error("CRITICAL: Fallback report is empty - creating minimal report")
            fallback_report = f"""# Research Report: {query}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** Report generation encountered errors

## Note

Research was completed but report generation failed. Please check draft_report.md or findings for research results.
"""
        
        # Stream fallback report in chunks
        if stream:
            logger.info("Streaming fallback report", report_length=len(fallback_report))
            chunk_size = 500
            for i in range(0, len(fallback_report), chunk_size):
                chunk = fallback_report[i:i+chunk_size]
                stream.emit_report_chunk(chunk)
                await asyncio.sleep(0.02)  # Small delay for smooth streaming
            logger.info("Fallback report chunks streamed", total_chunks=(len(fallback_report) + chunk_size - 1) // chunk_size)
        
        # CRITICAL: Always return a report, even if minimal
        return {
            "final_report": fallback_report,
            "confidence": "medium" if draft_report else "low"
        }


# ==================== Aliases for Backward Compatibility ====================

# Old names point to enhanced versions
plan_research_node = plan_research_enhanced_node
spawn_agents_node = create_agent_characteristics_enhanced_node
execute_agents_node = execute_agents_enhanced_node
supervisor_react_node = supervisor_review_enhanced_node
generate_report_node = generate_final_report_enhanced_node
