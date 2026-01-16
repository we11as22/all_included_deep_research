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

        # CRITICAL: Check session_status to determine if clarification already sent
        # Use session_status instead of text markers!
        if session_status == "waiting_clarification":
            # Check if user has ANSWERED the clarification questions
            # User answered if:
            # 1. Last message in chat_history is from user
            # 2. AND it's NOT the original query (meaning it's a NEW message - the answer)
            if chat_history and chat_history[-1].get("role") == "user":
                last_user_message = chat_history[-1].get("content", "").strip()
                # If last user message is different from original query, user has answered
                if last_user_message and last_user_message != original_query.strip():
                    logger.info("User answered clarification questions, proceeding with research",
                               session_id=session_id,
                               answer_preview=last_user_message[:100])

                    # Update session status to researching
                    if session_manager:
                        try:
                            await session_manager.update_status(session_id, "researching")
                        except Exception as e:
                            logger.warning("Failed to update session status", error=str(e))

                    # Return with updated status and clarification_needed=False to proceed
                    return {
                        "clarification_needed": False,
                        "session_status": "researching",
                        "clarification_just_sent": False,
                        "clarification_answers": last_user_message  # Save the answers
                    }

            # If we reach here, user has NOT answered yet
            logger.info("Clarification already sent, still waiting for user answers - will interrupt before analyze_query",
                       session_id=session_id,
                       session_status=session_status)
            # CRITICAL: Return state with flags so interrupt_before=["analyze_query"] stops the graph
            return {
                "clarification_needed": True,
                "session_status": "waiting_clarification",
                "clarification_just_sent": True
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

                # Emit as report chunk
                try:
                    stream.emit_report_chunk(clarification_message)
                    logger.info("Clarification questions emitted",
                               message_length=len(clarification_message),
                               questions_count=len(questions_to_send))

                    # Save clarification questions to DB
                    message_id = f"clarification_{session_id}_{int(time.time() * 1000)}"
                    await self._save_message_to_db(
                        stream=stream,
                        role="assistant",
                        content=clarification_message,
                        message_id=message_id,
                    )

                    # Update session status to waiting_clarification
                    if session_manager:
                        await session_manager.update_status(session_id, "waiting_clarification")
                        logger.info("Session status updated to waiting_clarification",
                                   session_id=session_id)

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

    async def _save_message_to_db(self, stream: Any, role: str, content: str, message_id: str) -> None:
        """Save message to database.

        Args:
            stream: Stream object with app_state
            role: Message role (user/assistant)
            content: Message content
            message_id: Unique message ID
        """
        try:
            from src.database.schema import ChatMessageModel

            app_state = getattr(stream, "app_state", {})
            chat_id = app_state.get("chat_id")
            session_factory = app_state.get("session_factory")

            if not chat_id or not session_factory:
                logger.warning("Cannot save message - missing chat_id or session_factory")
                return

            async with session_factory() as session:
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
                           content_length=len(content))
        except Exception as e:
            logger.error("Failed to save message to DB", error=str(e), exc_info=True)


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
