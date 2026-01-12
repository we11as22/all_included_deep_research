"""LangGraph nodes for deep research workflow with structured outputs.

Each node is an async function that takes state and returns state updates.
All enhanced nodes use structured outputs with reasoning fields.
"""

import asyncio
import contextvars
import structlog
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

# Context variable for runtime dependencies (not serialized in state)
runtime_deps_context = contextvars.ContextVar('runtime_deps', default=None)


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
                
                if existing_message:
                    # Update existing message
                    existing_message.content = content
                    existing_message.role = role
                    chat.updated_at = datetime.now()
                    await session.commit()
                    logger.info("Message updated in DB", message_id=message_id, role=role, content_length=len(content))
                    return True
                else:
                    # Create new message
                    message = ChatMessageModel(
                        chat_id=chat_id,
                        message_id=message_id,
                        role=role,
                        content=content,
                    )
                    session.add(message)
                    chat.updated_at = datetime.now()
                    await session.commit()
                    logger.info("Message saved to DB", message_id=message_id, role=role, content_length=len(content))
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
    
    # Step 2: Find clarification questions (should come AFTER deep search)
    if deep_search_done:
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
                full_message = f"## Initial Deep Search Context\n\n{result}\n\n---\n\n*This context will be used to guide the research agents.*\n\n"
                
                # Always send in chunks for smooth streaming (chunk size 10000 chars)
                chunk_size = 10000
                chunks = [full_message[i:i+chunk_size] for i in range(0, len(full_message), chunk_size)]
                logger.info("Sending deep search result in chunks", total_length=len(full_message), chunks_count=len(chunks))
                
                for i, chunk in enumerate(chunks):
                    stream.emit_report_chunk(chunk)
                    # Small delay between chunks to ensure smooth streaming
                    if i < len(chunks) - 1:  # Don't sleep after last chunk
                        await asyncio.sleep(0.03)  # Small delay between chunks
                
                logger.info("Deep search FULL result emitted to stream", result_length=len(result), chunks_sent=len(chunks))
                
                # CRITICAL: Save deep search result to DB immediately after emitting
                # This ensures it's persisted even if stream fails or user switches chats
                # Use unique message_id based on session_id and timestamp
                from uuid import uuid4
                import time
                session_id = state.get("session_id", "unknown")
                message_id = f"deep_search_{session_id}_{int(time.time() * 1000)}"
                await _save_message_to_db_async(
                    stream=stream,
                    role="assistant",
                    content=full_message,
                    message_id=message_id,
                )
                
                # Small delay to ensure all chunks are sent before continuing
                await asyncio.sleep(0.15)
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
    deep_search_result_raw = state.get("deep_search_result", "")
    if isinstance(deep_search_result_raw, dict):
        # If it's a dict with "type": "override", extract the value
        deep_search_result = deep_search_result_raw.get("value", "") if isinstance(deep_search_result_raw, dict) else ""
    else:
        deep_search_result = deep_search_result_raw or ""
    chat_history = state.get("chat_history", [])
    # Restore runtime dependencies if not in state
    state = _restore_runtime_deps(state)
    
    llm = state.get("llm")
    stream = state.get("stream")
    settings = state.get("settings")
    
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
    
    prompt = f"""Generate clarifying questions about the research topic and approach.

**ORIGINAL USER MESSAGE (from the start of current deep research session):**
{user_message_for_context}
{deep_search_context}

**CRITICAL REQUIREMENTS:**
1. The message above is the ORIGINAL user query that started this deep research session
2. Generate clarifying questions STRICTLY about THIS specific query and topic
3. Questions MUST be directly related to the original query - do NOT generate questions about unrelated topics
4. If the deep search context mentions specific aspects, generate questions about THOSE aspects within the context of the original query
5. Do NOT generate questions about topics that are NOT mentioned in the original query or deep search context

**MANDATORY**: Your questions MUST be about the SPECIFIC TOPIC from the original query: "{user_message_for_context}"

You MUST always generate 2-3 clarifying questions about:
1. The specific aspect or focus of the research related to the original query (what exactly should be researched about THIS topic)
2. The depth and scope of the research (how detailed should it be for THIS topic)
3. The type of information needed for THIS topic (technical details, historical context, comparisons, etc.)

**FORBIDDEN**: Do NOT generate questions about topics that are NOT in the original query!
**FORBIDDEN**: Do NOT misinterpret the query - if the user asked about "оформление работников", do NOT ask about "ИП" (individual entrepreneurs) unless it's clearly related!

These questions help guide the research direction and ensure comprehensive coverage of THE ORIGINAL TOPIC.
Even if the query seems clear, ask questions to refine the research approach FOR THIS SPECIFIC TOPIC.

**CRITICAL: LANGUAGE REQUIREMENT**
- You MUST write all questions in the SAME LANGUAGE as the original user message above
- If the user wrote in Russian, write questions in Russian
- If the user wrote in English, write questions in English
- Match the user's language exactly

IMPORTANT: You MUST always return at least 2 questions in the questions list, even if the query seems clear.
Set needs_clarification=True and provide meaningful questions that help improve research quality.

Format questions with:
- question: The actual question (in the same language as the user's message)
- why_needed: Why this clarification helps improve research (in the same language as the user's message)
- default_assumption: What we'll assume if not answered (in the same language as the user's message)
"""
    
    try:
        system_prompt = """You are a research planning expert. You MUST always generate clarifying questions to help improve research quality. 

CRITICAL REQUIREMENTS:
1. Write all questions in the SAME LANGUAGE as the user's original message - match their language exactly
2. Generate questions STRICTLY about the SPECIFIC TOPIC from the original user message
3. Do NOT generate questions about topics that are NOT mentioned in the original query or deep search context
4. If the original query is about "оформление работников", do NOT ask about "ИП" (individual entrepreneurs) unless it's clearly related to employee registration
5. If the original query is about a specific topic, all questions MUST be about aspects of THAT topic, not unrelated topics
6. Questions should help refine the research approach FOR THE ORIGINAL TOPIC, not introduce new topics"""
        clarification = await llm.with_structured_output(ClarificationNeeds).ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        logger.info(
            "Clarification analysis",
            needs_clarification=clarification.needs_clarification,
            questions_count=len(clarification.questions),
            can_proceed=clarification.can_proceed_without,
            original_query=user_message_for_context[:100],
            questions_preview=[q.question[:100] if hasattr(q, 'question') else str(q)[:100] for q in clarification.questions[:3]] if clarification.questions else []
        )
        
        # ALWAYS send questions if they exist, regardless of needs_clarification flag
        # If no questions generated, create default ones
        questions_to_send = clarification.questions if clarification.questions else []
        
        # If LLM didn't generate questions, create default ones based on query and deep search
        if not questions_to_send:
            logger.warning("LLM didn't generate questions, creating default ones")
            from src.workflow.research.models import ClarifyingQuestion
            
            # Create default questions based on query and deep search context
            questions_to_send = [
                ClarifyingQuestion(
                    question="What specific aspect of this topic should be the primary focus of the research?",
                    why_needed="This helps narrow down the research scope and ensure we cover the most important aspects.",
                    default_assumption="We'll research all major aspects comprehensively."
                ),
                ClarifyingQuestion(
                    question="What level of detail do you need? (e.g., overview, technical deep-dive, historical context, practical applications)",
                    why_needed="This determines the depth and type of information we'll gather.",
                    default_assumption="We'll provide a comprehensive overview with key technical details."
                ),
                ClarifyingQuestion(
                    question="Are there any specific sources, perspectives, or angles you want us to prioritize?",
                    why_needed="This helps us focus on the most relevant information sources.",
                    default_assumption="We'll use diverse sources and perspectives for balanced coverage."
                )
            ]
        
        # ALWAYS send questions to user - this is mandatory after deep search
        if questions_to_send:
            # Send clarifying questions to user via stream
            if stream:
                questions_text = "\n\n".join([
                    f"**Q{i+1}:** {q.question}\n\n*Why needed:* {q.why_needed}\n\n*Default assumption if not answered:* {q.default_assumption}"
                    for i, q in enumerate(questions_to_send)
                ])
                
                clarification_message = f"""## 🔍 Clarification Needed

Before starting the research, I need to clarify a few points:

{questions_text}

---

*Note: Please answer these questions to help guide the research direction. 
Research will proceed after you provide your answers.*
"""
                
                # CRITICAL: Emit as report chunk FIRST so it appears in the assistant message
                # This ensures questions are visible to user before research continues
                try:
                    stream.emit_report_chunk(clarification_message)
                    logger.info("Clarification questions emitted as report chunk", message_length=len(clarification_message))
                    
                    # CRITICAL: Save clarification questions to DB immediately after emitting
                    # This ensures they're persisted even if stream fails or user switches chats
                    # Use unique message_id based on session_id and timestamp
                    from uuid import uuid4
                    import time
                    session_id = state.get("session_id", "unknown")
                    message_id = f"clarification_{session_id}_{int(time.time() * 1000)}"
                    await _save_message_to_db_async(
                        stream=stream,
                        role="assistant",
                        content=clarification_message,
                        message_id=message_id,
                    )
                except Exception as e:
                    logger.error("Failed to emit clarification questions as report chunk", error=str(e), exc_info=True)
                
                # Also emit as status for progress tracking
                try:
                    stream.emit_status("Waiting for your clarification answers...", step="clarification")
                except Exception as e:
                    logger.error("Failed to emit clarification status", error=str(e))
                
                # Small delay to ensure all events are sent before graph stops
                await asyncio.sleep(0.2)
                
                # Also emit as a separate message event to ensure it's visible
                logger.info("MANDATORY: Clarifying questions sent to user after deep search", 
                           questions_count=len(questions_to_send),
                           message_length=len(clarification_message),
                           deep_search_used=bool(deep_search_result))
            
            # Always proceed with assumptions (user can answer later, but research continues)
            logger.info("Clarifying questions shown to user, proceeding with default assumptions. User can provide answers in chat.")
            
            return {
                "clarification_needed": True,  # Questions were sent
                "clarification_questions": [q.dict() if hasattr(q, "dict") else {"question": q.question, "why_needed": q.why_needed, "default_assumption": q.default_assumption} for q in questions_to_send]  # Store for reference
            }
        else:
            # This should never happen, but log if it does
            logger.error("CRITICAL: No questions generated and no fallback questions created!")
            return {"clarification_needed": False}
        
    except Exception as e:
        logger.error("Clarification analysis failed", error=str(e))
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
    
    prompt = f"""Analyze this research query to determine the best approach:

Query: {query}
{clarification_context}
{deep_search_context}
Assess:
1. What are the main topics/subtopics?
2. How complex is this query?
3. How many specialized research agents would be optimal?
4. What research angles should different agents cover?

IMPORTANT: If user provided clarification answers, use them to refine your analysis.
"""

    try:
        analysis = await llm.with_structured_output(QueryAnalysis).ainvoke([
            {"role": "system", "content": "You are an expert research planner."},
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
    
    # CRITICAL: Extract user clarification answers from chat_history
    clarification_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant" and ("clarification" in msg.get("content", "").lower() or "🔍" in msg.get("content", "")):
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                    user_answer = chat_history[i + 1].get("content", "")
                    clarification_context = f"\n\n**USER CLARIFICATION ANSWERS (CRITICAL - MUST BE CONSIDERED):**\n{user_answer}\n\nThese answers refine the research scope and priorities. Use them to focus the research plan."
                    break

    prompt = f"""Create a comprehensive research plan for this query:

Query: {query}
{clarification_context}
Query analysis: {query_analysis.get('reasoning', '')}
Identified topics: {', '.join(query_analysis.get('topics', []))}
Complexity: {query_analysis.get('complexity', 'moderate')}{context_info}

Create a detailed research plan with specific topics for different agents to investigate.
IMPORTANT: If user provided clarification answers, use them to refine and focus the research plan.
"""

    try:
        plan = await llm.with_structured_output(ResearchPlan).ainvoke([
            {"role": "system", "content": "You are a research strategy expert."},
            {"role": "user", "content": prompt}
        ])

        logger.info("Research plan created", topics=len(plan.topics))

        return {
            "research_plan": {
                "reasoning": plan.reasoning,
                "research_depth": plan.research_depth,
                "coordination_strategy": plan.coordination_strategy
            },
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
    agent_count = state.get("estimated_agent_count", getattr(settings, "max_concurrent_agents", 4) if settings else 4)
    agent_count = min(agent_count, len(research_topics) + 1)  # At least one per topic

    if stream:
        stream.emit_status(f"Creating {agent_count} specialized research agents...", step="agent_characteristics")

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

    prompt = f"""Create a team of {agent_count} specialized research agents for this project:

**ORIGINAL USER QUERY:** {query}

**INITIAL DEEP SEARCH CONTEXT:**
{deep_search_result[:2000] if deep_search_result else "No initial deep search context available."}
{clarification_context}

Research topics:
{chr(10).join([f"- {t.get('topic')}: {t.get('description')}" for t in research_topics])}

CRITICAL: Each agent must research DIFFERENT aspects to build a complete picture!
**MANDATORY: Use the original user query "{query}", deep search context, and user clarification answers above to create relevant agent characteristics and tasks that directly address what the user asked for.**

**CRITICAL: TASK CREATION REQUIREMENTS - MANDATORY:**
- **EVERY task MUST include the original user query "{query}" in the objective**
- **EVERY task MUST be SPECIFIC to the user's query - not generic!**
- **MANDATORY format**: Start each task objective with "The user asked: '{query}'. Research [specific aspect related to query]..."
- **CRITICAL: Clarification interpretation**: If clarification answers are provided, interpret them IN THE CONTEXT of the original query
  * Clarification specifies WHAT ASPECT of the original topic to focus on, NOT a new topic
  * Example: Query "оформление работников", Clarification "все режимы" → means "режимы оформления работников", NOT "режимы" in general
  * Example: Query "обучение qwen", Clarification "технические тонкости" → means "технические тонкости обучения qwen", NOT "технические тонкости" in general
- **MANDATORY**: Include user's clarification answers (if provided) in task descriptions, but ALWAYS in context of original query
- **FORBIDDEN**: Do NOT create generic tasks like "Identify and clarify the specific event" - be SPECIFIC!
- **FORBIDDEN**: Do NOT interpret clarification as a standalone query - it's ALWAYS about the original query topic!
- **EXAMPLE**: If user asked "расскажи про хронологию боёв за курскую область на СВО" and wants "технический аспект точная хронология и инфа об участвовавших подразделениях", create tasks like:
  * "The user asked: 'расскажи про хронологию боёв за курскую область на СВО'. Research the precise chronological timeline of battles for Kursk Oblast in SMO, including exact dates, locations, and participating military units."
  * NOT: "Identify and clarify the specific event or operation" (too generic!)
- **MANDATORY**: Each task must be self-contained - the agent will NOT see the original query, only the task description!

For each agent, create:
1. A unique role (e.g., "Senior Aviation Historian", "Economic Policy Analyst", "Technical Specifications Expert", "Case Study Researcher")
2. Specific expertise area - ensure DIFFERENT angles:
   - Historical development and evolution
   - Technical specifications and technical details
   - Expert opinions, analysis, and critical perspectives
   - Real-world applications, case studies, and practical examples
   - Industry trends, current state, and future prospects
   - Comparative analysis with alternatives/competitors
   - Economic, social, or cultural impact
   - Challenges, limitations, and controversies
3. Personality traits (thorough, analytical, critical, etc.)
4. Initial todo list with 2-3 specific research tasks - each agent should cover DIFFERENT aspects

DIVERSITY REQUIREMENT:
- Ensure agents cover DIFFERENT angles - avoid overlap!
- Each agent should contribute unique insights to build comprehensive understanding
- From diverse agent findings, the supervisor will assemble a COMPLETE picture
- Examples: one agent focuses on history, another on technical specs, another on expert views, another on applications, etc.
- **BUT**: All angles must relate to the user's query "{query}" - do NOT research unrelated topics!
- **CRITICAL**: If clarification was provided, interpret it IN CONTEXT of the original query - clarification specifies aspects of the original topic, NOT a new topic!{query}"!
"""

    try:
        characteristics = await llm.with_structured_output(AgentCharacteristics).ainvoke([
            {"role": "system", "content": "You are an expert at designing research teams."},
            {"role": "user", "content": prompt}
        ])

        logger.info("Agent characteristics created", agent_count=len(characteristics.agents))

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

    if stream:
        stream.emit_status(f"Executing {agent_count} research agents in parallel...", step="agents")

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
        
        agent_tasks = []
        for agent_id in agent_characteristics.keys():
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

        logger.info(f"Launched {len(agent_tasks)} agents for cycle {iteration_count}")
        
        if stream:
            stream.emit_status(f"🚀 Launched {len(agent_tasks)} agents for cycle {iteration_count}", step="agents")

        # Collect results from this cycle - wait for all agents in parallel
        cycle_findings = []
        no_tasks_count = 0
        
        # Gather all agent tasks in parallel (not sequentially)
        agent_results = await asyncio.gather(
            *[task for _, task in agent_tasks],
            return_exceptions=True
        )
        
        # Process results
        for (agent_id, _), result in zip(agent_tasks, agent_results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_id} failed", error=str(result), exc_info=result)
            elif result:
                if result.get("topic") == "no_tasks":
                    no_tasks_count += 1
                    logger.info(f"Agent {agent_id} has no tasks")
                else:
                    cycle_findings.append(result)
                    all_findings.append(result)
                    logger.info(f"Agent {agent_id} completed task", sources=len(result.get("sources", [])))
        
        # CRITICAL: Automatically update draft_report with new findings after each agent cycle
        # This ensures draft_report is always up-to-date with latest findings, even if supervisor doesn't call write_draft_report
        if cycle_findings and len(cycle_findings) > 0:
            agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
            if agent_memory_service:
                try:
                    # Read current draft_report
                    try:
                        draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                    except FileNotFoundError:
                        draft_content = ""
                    
                    # Get new findings from this cycle
                    from datetime import datetime
                    query = state.get("query", "")
                    
                    # Build findings sections for new cycle findings
                    findings_sections = []
                    for f in cycle_findings:
                        full_summary = f.get('summary', 'No summary')
                        all_key_findings = f.get('key_findings', [])
                        sources = f.get('sources', [])
                        
                        findings_sections.append(f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Confidence:** {f.get('confidence', 'unknown')}
**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Summary

{full_summary}

### Key Findings

{chr(10).join([f"- {kf}" for kf in all_key_findings]) if all_key_findings else "No key findings"}

### Sources ({len(sources)})

{chr(10).join([f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources[:10]]) if sources else "No sources"}
""")
                    
                    findings_text = "\n\n".join(findings_sections)
                    
                    # Append to existing draft or create new
                    if len(draft_content) > 500:
                        # Append new findings section
                        update = f"""

---

## New Findings - Cycle {iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{findings_text}
"""
                        updated_draft = draft_content + update
                    else:
                        # Create new draft with all findings so far
                        all_findings_text = "\n\n".join([
                            f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Confidence:** {f.get('confidence', 'unknown')}

### Summary

{f.get('summary', 'No summary')}

### Key Findings

{chr(10).join([f"- {kf}" for kf in f.get('key_findings', [])]) if f.get('key_findings') else "No key findings"}

### Sources ({len(f.get('sources', []))})

{chr(10).join([f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in f.get('sources', [])[:10]]) if f.get('sources') else "No sources"}
""" for f in all_findings
                        ])
                        
                        updated_draft = f"""# Research Report Draft

**Query:** {query}
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(all_findings)}

## Overview

This is the working draft of the research report. Findings are automatically updated as agents complete their tasks.

## Detailed Findings

{all_findings_text}

---
"""
                    
                    await agent_memory_service.file_manager.write_file("draft_report.md", updated_draft)
                    logger.info("Draft report automatically updated with cycle findings", 
                              cycle_findings=len(cycle_findings), total_findings=len(all_findings), draft_length=len(updated_draft))
                except Exception as e:
                    logger.warning("Failed to automatically update draft_report with cycle findings", error=str(e))

        # If all agents report no tasks, stop
        if no_tasks_count == len(agent_tasks):
            logger.info("All agents report no tasks, stopping agent execution", 
                       no_tasks_count=no_tasks_count, total_agents=len(agent_tasks))
            if stream:
                stream.emit_status("✅ All agents completed their tasks", step="agents")
            agents_active = False
            break
        
        # After each cycle, process supervisor queue if there are completions
        if supervisor_queue.size() > 0:
            # Check supervisor call limit BEFORE calling
            if supervisor_call_count >= max_supervisor_calls:
                logger.warning(f"Supervisor call limit reached ({supervisor_call_count}/{max_supervisor_calls}), agents will complete tasks without supervisor")
                # Don't break - let agents complete their tasks without supervisor
                # Just clear the queue and continue
                while not supervisor_queue.queue.empty():
                    try:
                        supervisor_queue.queue.get_nowait()
                        supervisor_queue.queue.task_done()
                    except:
                        break
                # Continue loop - agents will finish their tasks
                continue  # Skip supervisor call but continue agent execution
            
            logger.info(f"Processing {supervisor_queue.size()} agent completions in supervisor queue (supervisor call {supervisor_call_count + 1}/{max_supervisor_calls})")
            
            if stream:
                stream.emit_status(f"👔 Supervisor reviewing findings (call {supervisor_call_count + 1}/{max_supervisor_calls})", step="supervisor")
            
            # Call supervisor agent to review and assign new tasks
            from src.workflow.research.supervisor_agent import run_supervisor_agent
            
            try:
                supervisor_call_count += 1
                state["supervisor_call_count"] = supervisor_call_count
                
                # Get max_iterations from settings (centralized config)
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
                    supervisor_queue=supervisor_queue,  # Pass supervisor_queue
                    max_iterations=supervisor_max_iterations
                )
                
                # Update state with supervisor decision
                state["should_continue"] = decision.get("should_continue", False)
                state["replanning_needed"] = decision.get("replanning_needed", False)
                
                # If supervisor says stop, break the loop
                if not decision.get("should_continue", False):
                    logger.info("Supervisor decided to stop, breaking agent execution loop", 
                               decision_reasoning=decision.get("reasoning", "")[:200])
                    if stream:
                        stream.emit_status("✅ Supervisor decided research is complete", step="supervisor")
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
                        
                logger.info("Supervisor queue processed", decision=decision.get("should_continue"))
                
            except Exception as e:
                logger.error("Supervisor processing failed", error=str(e))
        
        # CRITICAL: Automatically update draft_report with new findings after each agent cycle
        # This ensures draft_report is always up-to-date, even if supervisor doesn't call write_draft_report
        if all_findings and len(all_findings) > 0:
            agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
            if agent_memory_service:
                try:
                    # Read current draft_report
                    try:
                        draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                    except FileNotFoundError:
                        draft_content = ""
                    
                    # Get new findings from this cycle (findings that aren't already in draft)
                    # Extract topics already in draft to avoid duplicates
                    existing_topics = set()
                    if draft_content:
                        import re
                        topic_matches = re.findall(r'##\s+([^\n]+)', draft_content)
                        existing_topics = {t.strip().lower() for t in topic_matches}
                    
                    # Filter out findings already in draft
                    new_findings = []
                    for f in all_findings:
                        topic = f.get('topic', '').lower()
                        if topic not in existing_topics:
                            new_findings.append(f)
                    
                    # Only update if there are new findings
                    if new_findings:
                        from datetime import datetime
                        query = state.get("query", "")
                        
                        # Build findings sections for new findings only
                        findings_sections = []
                        for f in new_findings:
                            full_summary = f.get('summary', 'No summary')
                            all_key_findings = f.get('key_findings', [])
                            sources = f.get('sources', [])
                            
                            findings_sections.append(f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Confidence:** {f.get('confidence', 'unknown')}
**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Summary

{full_summary}

### Key Findings

{chr(10).join([f"- {kf}" for kf in all_key_findings]) if all_key_findings else "No key findings"}

### Sources ({len(sources)})

{chr(10).join([f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in sources[:10]]) if sources else "No sources"}
""")
                        
                        findings_text = "\n\n".join(findings_sections)
                        
                        # Append to existing draft or create new
                        if len(draft_content) > 500:
                            # Append new findings section
                            update = f"""

---

## New Findings - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{findings_text}
"""
                            updated_draft = draft_content + update
                        else:
                            # Create new draft with all findings
                            all_findings_text = "\n\n".join([
                                f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Confidence:** {f.get('confidence', 'unknown')}

### Summary

{f.get('summary', 'No summary')}

### Key Findings

{chr(10).join([f"- {kf}" for kf in f.get('key_findings', [])]) if f.get('key_findings') else "No key findings"}

### Sources ({len(f.get('sources', []))})

{chr(10).join([f"- {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}" for s in f.get('sources', [])[:10]]) if f.get('sources') else "No sources"}
""" for f in all_findings
                            ])
                            
                            updated_draft = f"""# Research Report Draft

**Query:** {query}
**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(all_findings)}

## Overview

This is the working draft of the research report. Findings are automatically updated as agents complete their tasks.

## Detailed Findings

{all_findings_text}

---
"""
                        
                        await agent_memory_service.file_manager.write_file("draft_report.md", updated_draft)
                        logger.info("Draft report automatically updated with new findings", 
                                  new_findings=len(new_findings), total_findings=len(all_findings), draft_length=len(updated_draft))
                except Exception as e:
                    logger.warning("Failed to automatically update draft_report", error=str(e))

    # Add findings to agent_findings (using reducer)
    # Update iteration in state
    new_iteration = current_iteration + iteration_count
    logger.info(f"Agent execution completed", cycles=iteration_count, total_iteration=new_iteration, max_iterations=max_iterations, supervisor_calls=supervisor_call_count)
    
    # CRITICAL: If supervisor call limit reached, MUST finalize draft_report with ALL findings (no truncation)
    # This is MANDATORY to ensure a comprehensive report is always available even if supervisor couldn't complete it
    if supervisor_call_count >= max_supervisor_calls:
        logger.info("MANDATORY: Supervisor call limit reached - finalizing draft_report with ALL findings", 
                   supervisor_calls=supervisor_call_count, max_calls=max_supervisor_calls, findings_count=len(all_findings))
    agent_memory_service = stream.app_state.get("agent_memory_service") if stream else None
    agent_file_service = stream.app_state.get("agent_file_service") if stream else None

    if agent_memory_service and agent_file_service:
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
            
            # Append to existing draft or create new
            if len(draft_content) > 500:
                # Append to existing draft
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
                # Create new comprehensive draft
                final_draft = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Findings:** {len(all_findings)}
**Supervisor Calls:** {supervisor_call_count}/{max_supervisor_calls}

## Executive Summary

This report synthesizes findings from {len(agent_characteristics)} research agents working on: {query}

## Detailed Findings (Complete, No Truncation)

{findings_text}

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

    return {
        "agent_findings": all_findings,
        "findings": all_findings,  # Keep for supervisor review
        "findings_count": len(all_findings),
        "iteration": new_iteration,
        "supervisor_call_count": supervisor_call_count
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
            max_iterations=supervisor_max_iterations
        )
        
        logger.info("Supervisor agent completed", decision=decision)
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
                    
                    # If draft_report is empty or short, create/update it with findings
                    if len(draft_content) < 500:
                        from datetime import datetime
                        query = state.get("query", "")
                        
                        findings_sections = []
                        for f in findings:
                            findings_sections.append(f"""## {f.get('topic', 'Unknown Topic')}

**Agent:** {f.get('agent_id', 'unknown')}
**Summary:** {f.get('summary', 'No summary')}
**Key Findings:** {', '.join(f.get('key_findings', [])[:5])}
""")
                        
                        findings_text = "\n\n".join(findings_sections)
                        draft_content = f"""# Research Report Draft

**Query:** {query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Note:** Supervisor encountered an error, but findings are available below.

## Findings

{findings_text}
"""
                        await agent_memory_service.file_manager.write_file("draft_report.md", draft_content)
                        logger.info("Updated draft_report with findings after supervisor error", findings_count=len(findings))
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
            draft_report = await agent_memory_service.file_manager.read_file("draft_report.md")
            logger.info("Read draft report", length=len(draft_report))
            
            # CRITICAL: Check if draft report is properly filled
            # If draft report is too short or empty, create it from all findings
            if len(draft_report) < 1000:
                logger.warning("Draft report is too short or empty - creating comprehensive draft from all findings", 
                             draft_length=len(draft_report), findings_count=len(findings))
                
                # Create comprehensive draft report from ALL findings (no truncation)
                from datetime import datetime
                query = state.get("query", "Unknown query")
                
                findings_sections = []
                for f in findings:
                    full_summary = f.get('summary', 'No summary')
                    all_key_findings = f.get('key_findings', [])
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
            logger.warning("Draft report not found - MANDATORY: creating comprehensive draft from all findings", findings_count=len(findings))
            if stream:
                stream.emit_status("Creating draft report from all findings...", step="report")
            
            # MANDATORY: Create comprehensive draft report from ALL findings (no truncation)
            from datetime import datetime
            query = state.get("query", "Unknown query")
            
            findings_sections = []
            for f in findings:
                full_summary = f.get('summary', 'No summary')
                all_key_findings = f.get('key_findings', [])
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

    # Use draft_report as primary source, fallback to main_document and findings
    primary_source = draft_report if draft_report else main_document
    if not primary_source:
        primary_source = findings_text
        logger.error("CRITICAL: No draft report or main document - supervisor should have written draft_report.md")
    
    # Build main document section separately (avoid backslash in f-string expression)
    main_doc_section = ""
    if main_document and main_document != primary_source:
        # Use more of main document (up to 10000 chars for context)
        main_doc_preview = main_document[:10000] if len(main_document) > 10000 else main_document
        if len(main_document) > 10000:
            main_doc_preview += f"\n\n[... {len(main_document) - 10000} more characters in main document]"
        main_doc_section = f"\nMain document (key insights):\n{main_doc_preview}\n\n"

    # CRITICAL: Use FULL draft_report (no truncation) - it's the supervisor's comprehensive synthesis
    # If draft_report is very large (>50000 chars), we'll use intelligent summarization for the prompt
    # but still instruct LLM to generate a COMPREHENSIVE report
    draft_report_for_prompt = primary_source
    if len(primary_source) > 50000:
        # For very large draft reports, use first 40000 chars + summary of rest
        from src.utils.text import summarize_text
        draft_report_for_prompt = primary_source[:40000] + "\n\n[... additional content summarized below ...]\n\n" + summarize_text(primary_source[40000:], 10000)
        logger.info("Draft report is very large - using first 40k chars + summary for prompt", 
                   total_length=len(primary_source), prompt_length=len(draft_report_for_prompt))
    elif len(primary_source) > 30000:
        # For large draft reports, use full content but note it
        logger.info("Draft report is large - using full content in prompt", length=len(primary_source))

    # Detect user language from query
    def _detect_user_language(text: str) -> str:
        """Detect user language from query text."""
        if not text:
            return "English"
        # Check for Cyrillic (Russian, Ukrainian, etc.)
        if any('\u0400' <= char <= '\u04FF' for char in text):
            return "Russian"
        return "English"
    
    user_language = _detect_user_language(query)
    
    prompt = f"""Generate a COMPREHENSIVE, DETAILED final research report. This should be a FULL, SUBSTANTIAL research document, not a brief summary.

Query: {query}

**CRITICAL: LANGUAGE REQUIREMENT**
- **MANDATORY**: You MUST write the entire report in {user_language}
- Match the user's query language exactly - if the user asked in {user_language}, write the report in {user_language}
- This applies to ALL sections: executive summary, all report sections, and conclusion

**CRITICAL**: Use the Draft Research Report as the PRIMARY source - it contains the supervisor's comprehensive synthesis of all agent findings.
**IMPORTANT**: Generate a FULL, DETAILED report with extensive analysis, multiple sections, and comprehensive coverage. The report should be SUBSTANTIAL and COMPLETE.

Draft Research Report (supervisor's working document - PRIMARY SOURCE):
{draft_report_for_prompt}
{main_doc_section}
Additional findings (for reference if draft is incomplete):
{findings_text[:10000] if len(findings_text) > 10000 else findings_text if findings_text else "None"}

Create a well-structured, COMPREHENSIVE, DETAILED report with:
1. Executive summary (synthesize the draft report's main points - should be substantial, 200-400 words, not brief)
2. Multiple detailed sections with extensive analysis (use ALL sections and findings from draft report - each section should be 300-800 words with specific facts, data, and evidence)
3. Evidence-based conclusions (based on draft report's analysis - should be 200-400 words, not brief)
4. Citations to sources (include ALL sources mentioned in draft report)

**CRITICAL REQUIREMENTS:**
- The report MUST be COMPREHENSIVE and SUBSTANTIAL - aim for 2000-5000+ words total
- Use ALL information from the draft report - don't skip or summarize too much
- Each section should be DETAILED with specific facts, data, and analysis
- Include ALL key findings from all agents
- Create at least 3-5 major sections covering different aspects of the topic
- The draft report should be your PRIMARY source - it contains the supervisor's comprehensive synthesis
- If draft report is incomplete or missing, use fallback sources but note this in the report
- DO NOT create a brief summary - create a FULL research document with extensive detail
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
            # Last resort: use findings only
            fallback_report = f"""# Research Report: {query}

## Findings

{findings_text if findings_text else "No findings available"}
"""
        
        logger.warning("Using fallback report", source="draft_report" if draft_report else "main_document" if main_document else "findings")
        
        # Stream fallback report in chunks
        if stream:
            logger.info("Streaming fallback report", report_length=len(fallback_report))
            chunk_size = 500
            for i in range(0, len(fallback_report), chunk_size):
                chunk = fallback_report[i:i+chunk_size]
                stream.emit_report_chunk(chunk)
                await asyncio.sleep(0.02)  # Small delay for smooth streaming
            logger.info("Fallback report chunks streamed", total_chunks=(len(fallback_report) + chunk_size - 1) // chunk_size)
        
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
