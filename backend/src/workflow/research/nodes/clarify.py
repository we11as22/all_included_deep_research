"""Clarification node for asking user questions."""

import asyncio
import structlog
from typing import Dict, Any
from uuid import uuid4
import time

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import ClarificationNeeds, ClarifyingQuestion
from src.workflow.research.prompts.clarify import ClarificationPromptBuilder

logger = structlog.get_logger(__name__)


class ClarifyNode(ResearchNode):
    """Ask clarifying questions to user before starting research.

    This helps narrow down research scope and ensure we understand the query correctly.
    Uses session_status to avoid text marker searches.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute clarification node.

        Args:
            state: Current research state

        Returns:
            State updates with clarification_needed flag
        """
        query = state.get("query", "")
        original_query = state.get("original_query", query)
        chat_history = state.get("chat_history", [])

        # Get deep_search_result - handle both dict and string formats
        deep_search_result_raw = state.get("deep_search_result", "")
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "")
        else:
            deep_search_result = deep_search_result_raw or ""

        session_status = state.get("session_status", "active")
        session_id = state.get("session_id", "unknown")

        # Access dependencies
        llm = self.deps.llm
        stream = self.deps.stream
        session_manager = self.deps.session_manager
        settings = self.deps.settings

        # CRITICAL: Use session state as source of truth, not chat_history
        # Method 1: Check if session_status is already "researching" (user answered, status updated)
        if session_status == "researching":
            logger.info("Session status is 'researching' - user already answered clarification, proceeding with research",
                       session_id=session_id)
            return {"clarification_needed": False}
        
        # Method 2: Check clarification_answers from session state (loaded from DB in create_initial_state)
        clarification_answers_from_state = state.get("clarification_answers", "")
        if clarification_answers_from_state and clarification_answers_from_state.strip():
            logger.info("Clarification already answered (from session state) - proceeding with research",
                       session_id=session_id,
                       answers_length=len(clarification_answers_from_state),
                       note="Using clarification_answers from session state, not chat_history")
            # Ensure session status is updated
            if session_manager and session_status != "researching":
                try:
                    await session_manager.update_status(session_id, "researching")
                    logger.info("Updated session status to 'researching' after detecting answers in state",
                               session_id=session_id)
                except Exception as e:
                    logger.warning("Failed to update session status", error=str(e), exc_info=True)
            return {"clarification_needed": False, "session_status": "researching"}

        # CRITICAL: Check session_status to determine if clarification already sent
        # Use session_status from DB session, NOT chat_history!
        if session_status == "waiting_clarification":
            # CRITICAL: Work with session from DB, not chat_history!
            # Check if user has already answered by checking session.clarification_answers
            existing_answers = None
            if session_manager:
                try:
                    session = await session_manager.get_session(session_id)
                    if session:
                        existing_answers = session.clarification_answers
                        logger.info("Checked session from DB for clarification answers",
                                   session_id=session_id,
                                   has_existing_answers=bool(existing_answers),
                                   existing_answers_preview=existing_answers[:100] if existing_answers else None)
                except Exception as e:
                    logger.warning("Failed to get session from DB", error=str(e), exc_info=True)
            
            # CRITICAL: Check if user has ANSWERED the clarification questions
            # Use chat_history only as fallback to detect NEW answers (when user just sent message)
            # But use session.clarification_answers as source of truth
            # User answered if:
            # 1. Last message in chat_history is from user (new message just received)
            # 2. AND it's different from original_query (not the original query repeated)
            # 3. AND either:
            #    a) clarification_answers is empty/None in session (user just answered)
            #    b) OR last_user_message is different from existing_answers (user updated answer)
            
            # Check for new user message in chat_history (fallback detection)
            if chat_history and chat_history[-1].get("role") == "user":
                last_user_message = chat_history[-1].get("content", "").strip()
                
                # Check if this is a new answer (different from original_query and substantial)
                is_new_answer = (
                    last_user_message and
                    len(last_user_message) > 10 and  # Substantial answer
                    last_user_message != original_query.strip()  # Not the original query
                )
                
                # User answered if:
                # 1. It's a new answer (different from original_query)
                # 2. AND either no existing answers OR answer is different from existing
                user_answered = False
                if is_new_answer:
                    if not existing_answers:
                        # No existing answers - this is the first answer
                        user_answered = True
                        logger.info("✅ User answered clarification (first answer, no existing answers in session)",
                                   session_id=session_id,
                                   answer_preview=last_user_message[:100],
                                   note="Detected from chat_history, saving to session")
                    elif last_user_message != existing_answers.strip():
                        # Answer is different from existing - user updated/answered
                        user_answered = True
                        logger.info("✅ User answered clarification (new/different answer)",
                                   session_id=session_id,
                                   answer_preview=last_user_message[:100],
                                   existing_answer_preview=existing_answers[:100] if existing_answers else None,
                                   note="Detected from chat_history, updating session")
                
                if user_answered:
                    # CRITICAL: Save answers to session - this is the source of truth
                    if session_manager:
                        try:
                            # Save clarification answers to session
                            await session_manager.save_clarification_answers(session_id, last_user_message)
                            logger.info("✅ Saved clarification answers to session", 
                                       session_id=session_id,
                                       note="clarification_answers is now source of truth, not chat_history")
                            
                            # Update session status to researching
                            await session_manager.update_status(session_id, "researching")
                            logger.info("✅ Session status updated to 'researching'", session_id=session_id)
                        except Exception as e:
                            logger.error("Failed to save answers/update status", error=str(e), exc_info=True)

                    # Return with updated status and clarification_needed=False to proceed
                    # CRITICAL: Return clarification_answers so it's available in state
                    return {
                        "clarification_needed": False,
                        "session_status": "researching",
                        "clarification_just_sent": False,
                        "clarification_answers": last_user_message  # Save the answers to state
                    }

            # If we reach here, user has NOT answered yet
            # CRITICAL: Clarification was already sent, combined message already exists in DB
            # Do NOT send it again - it's already loaded from chat_history
            logger.info("⏸️ Clarification already sent, still waiting for user answers - will interrupt before analyze_query",
                       session_id=session_id,
                       session_status=session_status,
                       has_existing_answers=bool(existing_answers),
                       last_message_role=chat_history[-1].get("role") if chat_history else None,
                       chat_history_length=len(chat_history),
                       note="Combined message already exists in DB, not sending again")
            # CRITICAL: Return state with flags so interrupt_before=["analyze_query"] stops the graph
            return {
                "clarification_needed": True,
                "session_status": "waiting_clarification",
                "clarification_just_sent": False  # CRITICAL: Set to False - clarification was sent in previous run, not now
            }

        if stream:
            stream.emit_status("Analyzing if clarification is needed...", step="clarification")

        # Verify query is not empty
        if not query:
            logger.error("CRITICAL: query is empty in state! Cannot generate clarification questions.")
            return {"clarification_needed": False}

        # Use original_query for clarification
        user_message_for_context = original_query
        logger.info("Using original_query for clarification questions",
                   query_preview=user_message_for_context[:100],
                   session_id=session_id)

        # Check if clarification questions are enabled in settings
        enable_clarifying_questions = getattr(settings, "deep_research_enable_clarifying_questions", True)
        if not enable_clarifying_questions:
            logger.info("Clarifying questions disabled in settings, skipping")
            return {"clarification_needed": False}

        # Build prompt using prompt builder
        prompt_builder = ClarificationPromptBuilder()

        # Get user language from state (detected in create_initial_state)
        user_language = state.get("user_language", "English")

        # Get query analysis if available
        query_analysis = state.get("query_analysis", {})

        prompt = prompt_builder.build_clarification_prompt(
            query=user_message_for_context,
            deep_search_result=deep_search_result,
            query_analysis=query_analysis,
            user_language=user_language
        )

        try:
            system_prompt = f"""You are a research planning expert. Generate clarifying questions to help improve research quality.

CRITICAL REQUIREMENTS:
1. Write all questions in {user_language}
2. Generate questions STRICTLY about the SPECIFIC TOPIC from the original user message
3. Questions should help refine the research approach FOR THE ORIGINAL TOPIC
4. Always generate 2-3 meaningful questions"""

            clarification = await llm.with_structured_output(ClarificationNeeds).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

            logger.info("Clarification analysis completed",
                       needs_clarification=clarification.needs_clarification,
                       questions_count=len(clarification.questions))

            # Get questions or create default ones
            questions_to_send = clarification.questions if clarification.questions else []

            # If LLM didn't generate questions, create default ones
            if not questions_to_send:
                logger.warning("LLM didn't generate questions, creating default ones")
                questions_to_send = self._create_default_questions(user_language)

            # Send questions to user via stream
            if questions_to_send and stream:
                clarification_message = self._format_clarification_message(questions_to_send)
                
                # CRITICAL: Check if combined message already exists in DB (session-based check)
                # This prevents double sending on continuation/reload
                # Use SESSION-BASED check, NOT chat_history (chat_history is unreliable for deep research sessions)
                # Also check session_status - if already "waiting_clarification", combined message was already sent
                combined_message_exists_in_db = False
                if session_status == "waiting_clarification":
                    # Session status indicates clarification was already sent
                    # Check DB to confirm combined message exists
                    if session_manager and session_id:
                        try:
                            from src.database.schema import ChatMessageModel
                            from sqlalchemy import select
                            
                            app_state = stream.app_state if stream else {}
                            chat_id = app_state.get("chat_id")
                            session_factory = app_state.get("session_factory")
                            
                            if chat_id and session_factory:
                                async with session_factory() as db:
                                    # Check for combined message (deep_search + clarification) in DB
                                    result = await db.execute(
                                        select(ChatMessageModel)
                                        .where(
                                            ChatMessageModel.chat_id == chat_id,
                                            ChatMessageModel.role == "assistant",
                                            ChatMessageModel.content.like("%🔍 Initial Deep Search%"),
                                            ChatMessageModel.content.like("%Clarification Needed%")
                                        )
                                        .order_by(ChatMessageModel.created_at.desc())
                                        .limit(1)
                                    )
                                    existing_combined = result.scalar_one_or_none()
                                    if existing_combined:
                                        combined_message_exists_in_db = True
                                        logger.info("Combined deep_search + clarification already exists in DB (session_status=waiting_clarification)",
                                                   session_id=session_id,
                                                   message_id=existing_combined.message_id,
                                                   note="Combined message exists and was already sent, skipping send to frontend")
                                    else:
                                        logger.warning("Session status is waiting_clarification but combined message not found in DB",
                                                      session_id=session_id,
                                                      note="Will send combined message anyway to ensure it's saved")
                        except Exception as e:
                            logger.warning("Failed to check DB for existing combined message",
                                          session_id=session_id,
                                          error=str(e))
                else:
                    # First time sending - check if combined message exists (shouldn't, but check anyway)
                    if session_manager and session_id:
                        try:
                            from src.database.schema import ChatMessageModel
                            from sqlalchemy import select
                            
                            app_state = stream.app_state if stream else {}
                            chat_id = app_state.get("chat_id")
                            session_factory = app_state.get("session_factory")
                            
                            if chat_id and session_factory:
                                async with session_factory() as db:
                                    # Check for combined message (deep_search + clarification) in DB
                                    result = await db.execute(
                                        select(ChatMessageModel)
                                        .where(
                                            ChatMessageModel.chat_id == chat_id,
                                            ChatMessageModel.role == "assistant",
                                            ChatMessageModel.content.like("%🔍 Initial Deep Search%"),
                                            ChatMessageModel.content.like("%Clarification Needed%")
                                        )
                                        .order_by(ChatMessageModel.created_at.desc())
                                        .limit(1)
                                    )
                                    existing_combined = result.scalar_one_or_none()
                                    if existing_combined:
                                        combined_message_exists_in_db = True
                                        logger.info("Combined deep_search + clarification already exists in DB (unexpected - first time)",
                                                   session_id=session_id,
                                                   message_id=existing_combined.message_id,
                                                   note="Combined message exists, skipping send to frontend")
                        except Exception as e:
                            logger.warning("Failed to check DB for existing combined message",
                                          session_id=session_id,
                                          error=str(e))
                
                # CRITICAL: Send COMBINED message (deep_search + clarification) to frontend
                # In workflow logic they are separate entities, but on frontend/DB they are combined
                # Build combined message for frontend and DB
                combined_message = ""
                if deep_search_result and len(deep_search_result.strip()) > 0:
                    normalized_result = deep_search_result.rstrip()
                    combined_message = f"## 🔍 Initial Deep Search\n\n{normalized_result}\n\n---\n\n"
                combined_message += clarification_message
                
                try:
                    # CRITICAL: Only send to frontend if:
                    # 1. Combined message doesn't exist in DB
                    # 2. AND session_status is NOT "waiting_clarification" (if waiting_clarification, it was already sent)
                    # This prevents double sending on continuation
                    should_send_to_frontend = not combined_message_exists_in_db and session_status != "waiting_clarification"
                    
                    if should_send_to_frontend:
                        # Send COMBINED message to frontend
                        chunk_size = 10000
                        chunks = [combined_message[i:i+chunk_size] for i in range(0, len(combined_message), chunk_size)]
                        for i, chunk in enumerate(chunks):
                            stream.emit_report_chunk(chunk)
                            if i < len(chunks) - 1:
                                await asyncio.sleep(0.03)
                        
                        logger.info("Combined deep_search + clarification sent to frontend",
                                   combined_length=len(combined_message),
                                   deep_search_length=len(deep_search_result) if deep_search_result else 0,
                                   clarification_length=len(clarification_message),
                                   questions_count=len(questions_to_send),
                                   chunks_count=len(chunks),
                                   session_status=session_status,
                                   note="Sent as unified message to frontend (workflow logic: separate, frontend/DB: combined)")
                    else:
                        skip_reason = "combined message exists in DB" if combined_message_exists_in_db else f"session_status={session_status} (already sent)"
                        logger.info("Skipping send to frontend",
                                   session_id=session_id,
                                   session_status=session_status,
                                   combined_message_exists_in_db=combined_message_exists_in_db,
                                   skip_reason=skip_reason,
                                   note="Combined message already sent or exists in DB, will be loaded from chat_history on page reload")
                    
                    # CRITICAL: Always save/update COMBINED message (deep_search + clarification) to DB
                    # This ensures both are persisted together and won't be lost on page reload
                    # Check if combined message already exists - update or create
                    message_id = None
                    if combined_message_exists_in_db and chat_id and session_factory:
                        try:
                            async with session_factory() as db:
                                result = await db.execute(
                                    select(ChatMessageModel)
                                    .where(
                                        ChatMessageModel.chat_id == chat_id,
                                        ChatMessageModel.role == "assistant",
                                        ChatMessageModel.content.like("%🔍 Initial Deep Search%"),
                                        ChatMessageModel.content.like("%Clarification Needed%")
                                    )
                                    .order_by(ChatMessageModel.created_at.desc())
                                    .limit(1)
                                )
                                existing = result.scalar_one_or_none()
                                if existing:
                                    message_id = existing.message_id
                                    logger.info("Updating existing combined message in DB",
                                               message_id=message_id,
                                               session_id=session_id,
                                               note="Updating existing combined message with latest clarification")
                        except Exception as e:
                            logger.warning("Failed to find existing combined message for update", error=str(e))
                    
                    if not message_id:
                        # CRITICAL: message_id must be <= 64 chars (DB constraint)
                        # Use short prefix + session_id hash + timestamp
                        import hashlib
                        session_hash = hashlib.md5(session_id.encode()).hexdigest()[:8] if session_id else "unknown"
                        timestamp = int(time.time() * 1000) % 1000000000  # 9 digits max
                        message_id = f"ds_clr_{session_hash}_{timestamp}"
                        # Ensure it's <= 64 chars: "ds_clr_" (7) + hash (8) + "_" (1) + timestamp (9) = 25 chars
                        if len(message_id) > 64:
                            message_id = message_id[:64]
                    
                    # Save or update combined message in DB
                    # CRITICAL: Save message BEFORE updating status to ensure consistency
                    message_saved = await self._save_message_to_db(
                        stream=stream,
                        role="assistant",
                        content=combined_message,  # Save combined message to DB
                        message_id=message_id,
                    )
                    if message_saved:
                        logger.info("Combined message saved/updated in DB",
                                   message_id=message_id,
                                   combined_length=len(combined_message),
                                   deep_search_included=bool(deep_search_result),
                                   was_update=combined_message_exists_in_db,
                                   note="Combined message saved to DB for persistence and page reload")
                    else:
                        logger.error("CRITICAL: Failed to save combined message to DB",
                                   message_id=message_id,
                                   note="Message not saved - status will not be updated to prevent inconsistency")
                        # Don't update status if message save failed - this prevents inconsistent state
                        # The error is logged, but workflow continues (message might be saved on retry)

                    # CRITICAL: Update session status to waiting_clarification ONLY if message was saved
                    # This ensures consistency: if message is in DB, status is waiting_clarification
                    if message_saved and session_manager:
                        try:
                            await session_manager.update_status(session_id, "waiting_clarification")
                            logger.info("Session status updated to waiting_clarification",
                                       session_id=session_id,
                                       message_saved=message_saved,
                                       note="Status updated after successful message save")
                        except Exception as status_error:
                            logger.error("Failed to update session status after message save",
                                       session_id=session_id,
                                       error=str(status_error),
                                       exc_info=True,
                                       note="Message saved but status update failed - may cause inconsistency")

                except Exception as e:
                    logger.error("Failed to emit/save clarification questions",
                                error=str(e), exc_info=True)

                # Emit status
                try:
                    stream.emit_status("Waiting for your clarification answers...", step="clarification")
                except Exception as e:
                    logger.error("Failed to emit clarification status", error=str(e))

                # Small delay to ensure all events are sent
                await asyncio.sleep(0.2)

                logger.info("Clarifying questions sent to user - graph will interrupt before analyze_query",
                           questions_count=len(questions_to_send),
                           session_id=session_id)

                # CRITICAL: Return state with flags so interrupt_before=["analyze_query"] stops the graph
                # Graph will stop BEFORE analyze_query, save checkpoint
                # When user answers and graph is resumed, clarify will re-execute
                # and detect that user has answered, then proceed to analyze_query
                return {
                    "clarification_needed": True,
                    "session_status": "waiting_clarification",
                    "clarification_just_sent": True,
                    "clarification_questions": [
                        q.dict() if hasattr(q, "dict") else {
                            "question": q.question,
                            "why_needed": q.why_needed,
                            "default_assumption": q.default_assumption
                        } for q in questions_to_send
                    ]
                }
            else:
                return {"clarification_needed": False}

        except Exception as e:
            logger.error("Clarification analysis failed", error=str(e), exc_info=True)
            return {"clarification_needed": False}

    def _create_default_questions(self, user_language: str) -> list:
        """Create default clarification questions.

        Args:
            user_language: Language for questions

        Returns:
            List of ClarifyingQuestion objects
        """
        if user_language == "Russian":
            return [
                ClarifyingQuestion(
                    question="Какой аспект этой темы должен быть в фокусе исследования?",
                    why_needed="Это помогает сузить область исследования и убедиться, что мы покрываем самые важные аспекты.",
                    default_assumption="Мы исследуем все основные аспекты комплексно."
                ),
                ClarifyingQuestion(
                    question="Какой уровень детализации вам нужен? (например, обзор, технический deep-dive, исторический контекст, практическое применение)",
                    why_needed="Это определяет глубину и тип информации, которую мы будем собирать.",
                    default_assumption="Мы предоставим комплексный обзор с ключевыми техническими деталями."
                ),
                ClarifyingQuestion(
                    question="Есть ли какие-то конкретные источники, перспективы или углы зрения, которые вы хотите приоритизировать?",
                    why_needed="Это помогает нам сосредоточиться на наиболее релевантных источниках информации.",
                    default_assumption="Мы будем использовать разнообразные источники и перспективы для сбалансированного покрытия."
                )
            ]
        else:
            return [
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

    def _format_clarification_message(self, questions: list) -> str:
        """Format clarification questions into message.

        Args:
            questions: List of ClarifyingQuestion objects

        Returns:
            Formatted message string
        """
        questions_text = "\n\n".join([
            f"**Q{i+1}:** {q.question}\n\n*Why needed:* {q.why_needed}\n\n*Default assumption if not answered:* {q.default_assumption}"
            for i, q in enumerate(questions)
        ])

        return f"""## 🔍 Clarification Needed

Before starting the research, I need to clarify a few points:

{questions_text}

---

*Note: Please answer these questions to help guide the research direction.
Research will proceed after you provide your answers.*
"""

    async def _save_message_to_db(self, stream: Any, role: str, content: str, message_id: str) -> bool:
        """Save or update message in database.

        Args:
            stream: Stream object with app_state
            role: Message role (user/assistant)
            content: Message content
            message_id: Unique message ID

        Returns:
            True if message was saved successfully, False otherwise
        """
        try:
            from src.database.schema import ChatMessageModel
            from sqlalchemy import select

            app_state = getattr(stream, "app_state", {})
            chat_id = app_state.get("chat_id")
            session_factory = app_state.get("session_factory")

            if not chat_id or not session_factory:
                logger.warning("Cannot save message - missing chat_id or session_factory",
                              has_chat_id=bool(chat_id),
                              has_session_factory=bool(session_factory))
                return False

            async with session_factory() as session:
                # CRITICAL: Check if message already exists (by message_id)
                result = await session.execute(
                    select(ChatMessageModel).where(ChatMessageModel.message_id == message_id)
                )
                existing_message = result.scalar_one_or_none()
                
                if existing_message:
                    # Update existing message
                    existing_message.content = content
                    existing_message.role = role
                    # Ensure chat_id is correct (in case it changed)
                    if existing_message.chat_id != chat_id:
                        existing_message.chat_id = chat_id
                    await session.commit()
                    logger.info("Message updated in DB",
                               message_id=message_id,
                               chat_id=chat_id,
                               role=role,
                               content_length=len(content),
                               note="Updated existing message")
                    return True
                else:
                    # Create new message
                    new_message = ChatMessageModel(
                        chat_id=chat_id,
                        message_id=message_id,  # This is the message_id column, NOT id (which is auto-generated)
                        role=role,
                        content=content,
                    )
                    session.add(new_message)
                    await session.commit()
                    logger.info("Message saved to DB",
                               message_id=message_id,
                               chat_id=chat_id,
                               role=role,
                               content_length=len(content),
                               note="Created new message")
                    return True
        except Exception as e:
            logger.error("Failed to save message to DB", 
                        message_id=message_id,
                        chat_id=chat_id,
                        role=role,
                        error=str(e), 
                        exc_info=True)
            return False


# Legacy function wrapper for backward compatibility
async def clarify_with_user_node(state: ResearchState) -> Dict:
    """Legacy wrapper for ClarifyNode.

    This function maintains backward compatibility with existing code
    that imports clarify_with_user_node directly.

    TODO: Update imports to use ClarifyNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"clarification_needed": False}

    # Create dependencies container
    from src.workflow.research.dependencies import ResearchDependencies

    deps = ResearchDependencies(
        llm=runtime_deps.get("llm"),
        search_provider=runtime_deps.get("search_provider"),
        scraper=runtime_deps.get("scraper"),
        stream=runtime_deps.get("stream"),
        agent_memory_service=runtime_deps.get("agent_memory_service"),
        agent_file_service=runtime_deps.get("agent_file_service"),
        session_factory=runtime_deps.get("session_factory"),
        session_manager=runtime_deps.get("session_manager"),
        settings=runtime_deps.get("settings"),
    )

    # Execute node
    node = ClarifyNode(deps)
    return await node.execute(state)
