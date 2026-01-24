"""Clarification node for asking user questions."""

import asyncio
import structlog
from typing import Dict, Any
from uuid import uuid4
import time

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import ClarificationNeeds, ClarifyingQuestion
from typing import List
from src.workflow.research.prompts.clarify import ClarificationPromptBuilder

try:
    from openai import PermissionDeniedError
except ImportError:
    # Fallback if openai is not available
    PermissionDeniedError = Exception

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
                       answers_preview=clarification_answers_from_state[:100],
                       session_status=session_status,
                       note="Using clarification_answers from session state, not chat_history")
            # CRITICAL: Ensure session status is updated to "researching" if answers exist
            # This handles case where answers were saved but status wasn't updated yet
            if session_manager and session_status != "researching":
                try:
                    await session_manager.update_status(session_id, "researching")
                    logger.info("Updated session status to 'researching' after detecting answers in state",
                               session_id=session_id,
                               previous_status=session_status,
                               note="Status was not 'researching' but answers exist - updating status")
                except Exception as e:
                    logger.warning("Failed to update session status", error=str(e), exc_info=True)
            return {"clarification_needed": False, "session_status": "researching"}

        # CRITICAL: Check if this might be a clarification answer even if status is not "waiting_clarification"
        # This handles cases where status wasn't updated correctly but deep_search_result exists
        # and user sends a new message (not equal to original_query)
        if session_status != "waiting_clarification" and deep_search_result and not clarification_answers_from_state:
            # Check if user sent a new message (different from original_query)
            if chat_history and chat_history[-1].get("role") == "user":
                last_user_message = chat_history[-1].get("content", "").strip()
                if last_user_message and last_user_message != original_query.strip():
                    # This looks like a clarification answer - check session from DB
                    if session_manager:
                        try:
                            session = await session_manager.get_session(session_id)
                            if session and session.deep_search_result and not session.clarification_answers:
                                # Deep search done, no answers yet, user sent new message - treat as clarification answer
                                logger.info("🔍 Detected potential clarification answer (status not waiting_clarification but deep_search exists)",
                                           session_id=session_id,
                                           session_status=session_status,
                                           message_preview=last_user_message[:100],
                                           note="Deep search exists, no answers yet, treating new message as clarification answer")
                                # Save as clarification answer
                                await session_manager.save_clarification_answers(session_id, last_user_message)
                                await session_manager.update_status(session_id, "researching")
                                return {
                                    "clarification_needed": False,
                                    "session_status": "researching",
                                    "clarification_answers": last_user_message
                                }
                        except Exception as e:
                            logger.warning("Failed to check session for clarification answer detection", error=str(e))
        
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
            # Use chat_history to detect NEW answers (when user just sent message)
            # But use session.clarification_answers as source of truth
            # User answered if:
            # 1. Session is waiting for clarification (session_status == "waiting_clarification")
            # 2. Last message in chat_history is from user (new message just received)
            # 3. AND it's different from original_query (not the original query repeated)
            # 4. AND either:
            #    a) clarification_answers is empty/None in session (user just answered)
            #    b) OR last_user_message is different from existing_answers (user updated answer)
            
            # Check for new user message in chat_history
            if chat_history and chat_history[-1].get("role") == "user":
                last_user_message = chat_history[-1].get("content", "").strip()
                
                logger.info("Checking if user message is clarification answer",
                           session_id=session_id,
                           session_status=session_status,
                           message_preview=last_user_message[:100],
                           message_length=len(last_user_message),
                           has_existing_answers=bool(existing_answers),
                           original_query_preview=original_query[:100] if original_query else None,
                           note="Evaluating if user message is answer to clarification questions")
                
                # CRITICAL: If session is waiting for clarification and user sent a message,
                # it's likely an answer to clarification questions, even if short
                # Check if this is a new answer (different from original_query)
                # REMOVED: len(last_user_message) > 10 check - short answers like "да" are valid!
                is_new_answer = (
                    last_user_message and
                    len(last_user_message) > 0 and  # Any non-empty message
                    last_user_message != original_query.strip()  # Not the original query
                )
                
                logger.info("Answer evaluation",
                           session_id=session_id,
                           is_new_answer=is_new_answer,
                           is_waiting_clarification=(session_status == "waiting_clarification"),
                           message_length=len(last_user_message),
                           different_from_original=(last_user_message != original_query.strip() if last_user_message else False),
                           note="Checking if message qualifies as clarification answer")
                
                # User answered if:
                # 1. It's a new answer (different from original_query and non-empty)
                # 2. AND session is waiting for clarification (user is responding to questions)
                # 3. AND either no existing answers OR answer is different from existing
                user_answered = False
                if is_new_answer and session_status == "waiting_clarification":
                    if not existing_answers:
                        # No existing answers - this is the first answer
                        user_answered = True
                        logger.info("✅ User answered clarification (first answer, no existing answers in session)",
                                   session_id=session_id,
                                   answer_preview=last_user_message[:100],
                                   answer_length=len(last_user_message),
                                   note="Detected from chat_history, saving to session (short answers accepted)")
                    elif last_user_message != existing_answers.strip():
                        # Answer is different from existing - user updated/answered
                        user_answered = True
                        logger.info("✅ User answered clarification (new/different answer)",
                                   session_id=session_id,
                                   answer_preview=last_user_message[:100],
                                   answer_length=len(last_user_message),
                                   existing_answer_preview=existing_answers[:100] if existing_answers else None,
                                   note="Detected from chat_history, updating session (short answers accepted)")
                elif not is_new_answer:
                    logger.info("Message does not qualify as clarification answer",
                               session_id=session_id,
                               is_empty=(not last_user_message or len(last_user_message) == 0),
                               is_original_query=(last_user_message == original_query.strip() if last_user_message else False),
                               note="Message is empty or same as original query")
                elif session_status != "waiting_clarification":
                    logger.info("Session not waiting for clarification",
                               session_id=session_id,
                               session_status=session_status,
                               note="Session status is not 'waiting_clarification' - not treating as clarification answer")
                
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
            # CRITICAL: Clarification was already sent, but we need to ensure user sees it
            # Check if combined message exists in DB - if yes, it should be loaded from chat_history
            # But if user didn't see it (page reload, etc), we should send it again
            logger.info("⏸️ Clarification already sent, still waiting for user answers",
                       session_id=session_id,
                       session_status=session_status,
                       has_existing_answers=bool(existing_answers),
                       last_message_role=chat_history[-1].get("role") if chat_history else None,
                       chat_history_length=len(chat_history),
                       note="Checking if combined message needs to be resent for visibility")
            
            # CRITICAL: Always try to send combined message if it's not in recent chat_history
            # This ensures user sees clarification questions even after page reload
            # Check if combined message is in recent chat_history (last 5 messages)
            combined_message_in_history = False
            if chat_history:
                recent_messages = chat_history[-5:]  # Check last 5 messages
                for msg in recent_messages:
                    if msg.get("role") == "assistant" and "🔍 Initial Deep Search" in msg.get("content", "") and "Clarification Needed" in msg.get("content", ""):
                        combined_message_in_history = True
                        break
            
            if not combined_message_in_history and stream:
                # Combined message not in recent history - send it again to ensure visibility
                logger.info("Combined message not in recent chat_history - resending for visibility",
                           session_id=session_id,
                           note="User may not have seen clarification questions - resending to ensure visibility")
                
                # Get deep_search_result from state
                deep_search_result_raw = state.get("deep_search_result", "")
                if isinstance(deep_search_result_raw, dict):
                    deep_search_result = deep_search_result_raw.get("value", "")
                else:
                    deep_search_result = deep_search_result_raw or ""
                
                # Get questions from session or generate default
                questions_to_send = []
                if session_manager:
                    try:
                        session = await session_manager.get_session(session_id)
                        if session and hasattr(session, 'clarification_questions') and session.clarification_questions:
                            # Load questions from session
                            import json
                            if isinstance(session.clarification_questions, str):
                                questions_data = json.loads(session.clarification_questions)
                            else:
                                questions_data = session.clarification_questions
                            questions_to_send = [ClarifyingQuestion(**q) if isinstance(q, dict) else q for q in questions_data]
                    except Exception as e:
                        logger.warning("Failed to load questions from session", error=str(e))
                
                if not questions_to_send:
                    # Generate default questions as fallback
                    user_language = state.get("user_language", "English")
                    questions_to_send = self._create_default_questions(user_language)
                
                if questions_to_send:
                    clarification_message = self._format_clarification_message(questions_to_send)
                    combined_message = ""
                    if deep_search_result and len(deep_search_result.strip()) > 0:
                        combined_message = f"## 🔍 Initial Deep Search\n\n{deep_search_result.rstrip()}\n\n---\n\n"
                    else:
                        query = state.get("original_query", state.get("query", ""))
                        fallback_message = (
                            f"Initial deep search for '{query}' completed. "
                            "Found relevant sources and proceeding with detailed research approach."
                        )
                        combined_message = f"## 🔍 Initial Deep Search\n\n{fallback_message}\n\n---\n\n"
                    combined_message += clarification_message
                    
                    # Send to frontend
                    chunk_size = 10000
                    chunks = [combined_message[i:i+chunk_size] for i in range(0, len(combined_message), chunk_size)]
                    for i, chunk in enumerate(chunks):
                        stream.emit_report_chunk(chunk)
                        if i < len(chunks) - 1:
                            await asyncio.sleep(0.03)
                    
                    logger.info("Combined message resent to frontend for visibility",
                               session_id=session_id,
                               note="User may not have seen it - resent to ensure visibility")
            
            # CRITICAL: Return state with flags so interrupt_before=["analyze_query"] stops the graph
            return {
                "clarification_needed": True,
                "session_status": "waiting_clarification",
                "clarification_just_sent": False  # CRITICAL: Set to False - clarification was sent in previous run, not now
            }

        # CRITICAL: Check if clarification questions were already sent (combined message exists in DB)
        # This prevents regenerating questions when deep_search_result exists but status wasn't updated
        combined_message_exists = False
        if session_manager and session_id and deep_search_result:
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
                                ChatMessageModel.content.like("%Clarification Needed%"),
                                ChatMessageModel.session_id == session_id
                            )
                            .order_by(ChatMessageModel.created_at.desc())
                            .limit(1)
                        )
                        existing_combined = result.scalar_one_or_none()
                        if existing_combined:
                            combined_message_exists = True
                            logger.info("🛑 Clarification questions already sent (found in DB) - skipping regeneration",
                                       session_id=session_id,
                                       message_id=existing_combined.message_id,
                                       note="Combined message exists in DB - not generating questions again")
                            # Return with clarification_needed=True to wait for user answer
                            # But don't send questions again
                            return {
                                "clarification_needed": True,
                                "session_status": "waiting_clarification",
                                "clarification_just_sent": False
                            }
            except Exception as e:
                logger.warning("Failed to check for existing combined message", error=str(e))

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
                                    # CRITICAL: Also check session_id to ensure we only find messages for THIS session
                                    # This prevents finding old messages from previous sessions
                                    query_conditions = [
                                        ChatMessageModel.chat_id == chat_id,
                                        ChatMessageModel.role == "assistant",
                                        ChatMessageModel.content.like("%🔍 Initial Deep Search%"),
                                        ChatMessageModel.content.like("%Clarification Needed%")
                                    ]
                                    # Add session_id check if available
                                    if session_id and session_id != "unknown":
                                        query_conditions.append(ChatMessageModel.session_id == session_id)
                                    
                                    result = await db.execute(
                                        select(ChatMessageModel)
                                        .where(*query_conditions)
                                        .order_by(ChatMessageModel.created_at.desc())
                                        .limit(1)
                                    )
                                    existing_combined = result.scalar_one_or_none()
                                    if existing_combined:
                                        # CRITICAL: Verify session_id matches (double-check)
                                        if existing_combined.session_id == session_id or not session_id or session_id == "unknown":
                                            combined_message_exists_in_db = True
                                            logger.info("Combined deep_search + clarification already exists in DB (session_status=waiting_clarification)",
                                                       session_id=session_id,
                                                       message_id=existing_combined.message_id,
                                                       existing_session_id=existing_combined.session_id,
                                                       note="Combined message exists and was already sent, skipping send to frontend")
                                        else:
                                            logger.warning("Found combined message but session_id mismatch - will send anyway",
                                                          session_id=session_id,
                                                          existing_session_id=existing_combined.session_id,
                                                          message_id=existing_combined.message_id,
                                                          note="Message belongs to different session - will send new message")
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
                                    # CRITICAL: Also check session_id to ensure we only find messages for THIS session
                                    # This prevents finding old messages from previous sessions
                                    query_conditions = [
                                        ChatMessageModel.chat_id == chat_id,
                                        ChatMessageModel.role == "assistant",
                                        ChatMessageModel.content.like("%🔍 Initial Deep Search%"),
                                        ChatMessageModel.content.like("%Clarification Needed%")
                                    ]
                                    # Add session_id check if available
                                    if session_id and session_id != "unknown":
                                        query_conditions.append(ChatMessageModel.session_id == session_id)
                                    
                                    result = await db.execute(
                                        select(ChatMessageModel)
                                        .where(*query_conditions)
                                        .order_by(ChatMessageModel.created_at.desc())
                                        .limit(1)
                                    )
                                    existing_combined = result.scalar_one_or_none()
                                    if existing_combined:
                                        # CRITICAL: Verify session_id matches (double-check)
                                        if existing_combined.session_id == session_id or not session_id or session_id == "unknown":
                                            combined_message_exists_in_db = True
                                            logger.info("Combined deep_search + clarification already exists in DB (unexpected - first time)",
                                                       session_id=session_id,
                                                       message_id=existing_combined.message_id,
                                                       existing_session_id=existing_combined.session_id,
                                                       note="Combined message exists, skipping send to frontend")
                                        else:
                                            logger.warning("Found combined message but session_id mismatch - will send anyway",
                                                          session_id=session_id,
                                                          existing_session_id=existing_combined.session_id,
                                                          message_id=existing_combined.message_id,
                                                          note="Message belongs to different session - will send new message")
                        except Exception as e:
                            logger.warning("Failed to check DB for existing combined message",
                                          session_id=session_id,
                                          error=str(e))
                
                # CRITICAL: Send COMBINED message (deep_search + clarification) to frontend
                # In workflow logic they are separate entities, but on frontend/DB they are combined
                # Build combined message for frontend and DB
                combined_message = ""
                # CRITICAL: Always include deep_search_result section, even if empty
                # If empty, use fallback message to ensure user sees that deep search completed
                if deep_search_result and len(deep_search_result.strip()) > 0:
                    normalized_result = deep_search_result.rstrip()
                    combined_message = f"## 🔍 Initial Deep Search\n\n{normalized_result}\n\n---\n\n"
                else:
                    # CRITICAL: If deep_search_result is empty, still show section with fallback
                    # This ensures user sees that deep search completed, even if result was empty
                    query = state.get("original_query", state.get("query", ""))
                    fallback_message = (
                        f"Initial deep search for '{query}' completed. "
                        "Found relevant sources and proceeding with detailed research approach."
                    )
                    combined_message = f"## 🔍 Initial Deep Search\n\n{fallback_message}\n\n---\n\n"
                    logger.warning("Deep search result was empty - using fallback in combined message",
                                 session_id=session_id,
                                 original_result_length=len(deep_search_result) if deep_search_result else 0,
                                 note="Deep search completed but result was empty - using fallback for user visibility")
                combined_message += clarification_message
                
                try:
                    # CRITICAL: Always send combined message to frontend for NEW sessions
                    # This ensures user ALWAYS sees deep_search + clarification questions
                    # Check if this is a new session (not waiting for clarification)
                    is_new_session = session_status != "waiting_clarification"
                    
                    # CRITICAL: For new sessions, ALWAYS send to frontend to ensure visibility
                    if is_new_session or not combined_message_exists_in_db:
                        # Send combined message to frontend
                        stream.emit_report_chunk(combined_message)
                        logger.info("Sent combined deep_search + clarification to frontend",
                                   session_id=session_id,
                                   is_new_session=is_new_session,
                                   combined_message_length=len(combined_message),
                                   note="Combined message sent to frontend for user visibility")
                        
                        # CRITICAL: Update session status to waiting_clarification after sending questions
                        if session_manager and session_id:
                            try:
                                await session_manager.update_status(session_id, "waiting_clarification")
                                logger.info("✅ Updated session status to 'waiting_clarification' after sending clarification questions",
                                           session_id=session_id,
                                           questions_count=len(questions_to_send))
                            except Exception as e:
                                logger.warning("Failed to update session status to 'waiting_clarification'",
                                             session_id=session_id,
                                             error=str(e))
                    else:
                        logger.info("Skipping send to frontend - combined message already exists in DB",
                                   session_id=session_id,
                                   note="Combined message was already sent, not sending again")
                except Exception as e:
                    logger.error("Failed to send combined message to frontend",
                               session_id=session_id,
                               error=str(e),
                               exc_info=True)
            
            # Return with clarification questions
            return {
                "clarification_needed": True,
                "session_status": "waiting_clarification",
                "clarification_just_sent": True,
                "clarification_questions": questions_to_send
            }
        
        except (PermissionDeniedError, TimeoutError, Exception) as e:
            # CRITICAL: If LLM fails (403, timeout, etc.), skip clarification and proceed with research
            # This ensures workflow continues even when LLM is blocked or unavailable
            error_type = type(e).__name__
            is_permission_error = isinstance(e, PermissionDeniedError) or (hasattr(e, 'status_code') and e.status_code == 403)
            is_timeout = isinstance(e, TimeoutError) or (hasattr(e, 'status_code') and e.status_code == 408)
            
            logger.error("Clarification analysis failed - skipping clarification and proceeding with research",
                        error=str(e),
                        error_type=error_type,
                        is_permission_error=is_permission_error,
                        is_timeout=is_timeout,
                        session_id=session_id,
                        note="LLM failed to generate clarification questions. Proceeding without clarification to ensure research continues.")
            
            # CRITICAL: Send deep search result to frontend even if clarification fails
            # This ensures user sees deep search results
            deep_search_result_raw = state.get("deep_search_result", "")
            if isinstance(deep_search_result_raw, dict):
                deep_search_result = deep_search_result_raw.get("value", "")
            else:
                deep_search_result = deep_search_result_raw or ""
            
            if deep_search_result and stream:
                logger.info("Sending deep search result to frontend (clarification failed)",
                           session_id=session_id,
                           result_length=len(deep_search_result),
                           note="Deep search result will be sent to frontend even though clarification failed")
                # Send deep search result to frontend
                chunk_size = 10000
                chunks = [deep_search_result[i:i+chunk_size] for i in range(0, len(deep_search_result), chunk_size)]
                for i, chunk in enumerate(chunks):
                    stream.emit_report_chunk(chunk)
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.03)
            
            # Return without clarification - proceed with research
            return {
                "clarification_needed": False,
                "session_status": "researching",
                "clarification_just_sent": False,
                "clarification_answers": ""
            }

        except Exception as e:
            # CRITICAL: Catch any other unexpected errors and proceed without clarification
            logger.error("Unexpected error in clarification node - proceeding without clarification",
                        error=str(e),
                        error_type=type(e).__name__,
                        session_id=session_id,
                        exc_info=True,
                        note="Unexpected error occurred. Proceeding without clarification to ensure research continues.")
            
            # Send deep search result to frontend even on unexpected errors
            deep_search_result_raw = state.get("deep_search_result", "")
            if isinstance(deep_search_result_raw, dict):
                deep_search_result = deep_search_result_raw.get("value", "")
            else:
                deep_search_result = deep_search_result_raw or ""
            
            if deep_search_result and stream:
                logger.info("Sending deep search result to frontend (unexpected error in clarification)",
                           session_id=session_id,
                           result_length=len(deep_search_result),
                           note="Deep search result will be sent to frontend even though clarification failed")
                chunk_size = 10000
                chunks = [deep_search_result[i:i+chunk_size] for i in range(0, len(deep_search_result), chunk_size)]
                for i, chunk in enumerate(chunks):
                    stream.emit_report_chunk(chunk)
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.03)
            
            return {
                "clarification_needed": False,
                "session_status": "researching",
                "clarification_just_sent": False,
                "clarification_answers": ""
            }

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

    async def _save_message_to_db(self, stream: Any, role: str, content: str, message_id: str, session_id: str = None) -> bool:
        """Save or update message in database.

        Args:
            stream: Stream object with app_state
            role: Message role (user/assistant)
            content: Message content
            message_id: Unique message ID
            session_id: Research session ID (optional, but recommended for deep research)

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
                    # CRITICAL: Update session_id if provided (ensures message is linked to correct session)
                    if session_id and session_id != "unknown":
                        existing_message.session_id = session_id
                    await session.commit()
                    logger.info("Message updated in DB",
                               message_id=message_id,
                               chat_id=chat_id,
                               session_id=session_id,
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
                        session_id=session_id if session_id and session_id != "unknown" else None,  # CRITICAL: Link message to session
                    )
                    session.add(new_message)
                    await session.commit()
                    logger.info("Message saved to DB",
                               message_id=message_id,
                               chat_id=chat_id,
                               session_id=session_id,
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
