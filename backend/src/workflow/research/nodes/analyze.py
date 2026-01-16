"""Analysis node for query analysis."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import QueryAnalysis
from src.workflow.research.prompts.analysis import AnalysisPromptBuilder

logger = structlog.get_logger(__name__)


class AnalyzeQueryNode(ResearchNode):
    """Analyze query to determine research approach.

    Uses structured output to assess query complexity and plan.
    Uses session_status to check for clarification answers.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute query analysis node.

        Args:
            state: Current research state

        Returns:
            State updates with query_analysis
        """
        query = state.get("query", "")
        original_query = state.get("original_query", query)
        session_status = state.get("session_status", "active")
        session_id = state.get("session_id", "unknown")
        chat_history = state.get("chat_history", [])
        mode = state.get("mode", "quality")

        # Access dependencies
        llm = self.deps.llm
        stream = self.deps.stream

        # Check if clarification was sent but user hasn't answered
        # Use session_status instead of text markers!
        clarification_needed = state.get("clarification_needed", False)

        if clarification_needed and session_status == "waiting_clarification":
            # Check if user has answered (new user message after clarification)
            # Look at chat_history to see if there's a new user message
            has_user_answer = len(chat_history) > 0 and chat_history[-1].get("role") == "user"

            if not has_user_answer:
                # User hasn't answered yet - stop graph execution
                logger.info("Clarification needed but user hasn't answered - STOPPING GRAPH",
                           session_id=session_id,
                           session_status=session_status)
                if stream:
                    stream.emit_status("⏸️ Waiting for your clarification answers before proceeding...",
                                     step="clarification")

                return {
                    "clarification_waiting": True,
                    "should_stop": True
                }

            logger.info("User answered clarification, proceeding with analysis",
                       session_id=session_id)

        if stream:
            stream.emit_status("Analyzing query complexity...", step="analysis")

        # Get deep_search_result for context
        deep_search_result_raw = state.get("deep_search_result", "")
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "")
        else:
            deep_search_result = deep_search_result_raw or ""

        # Extract clarification answers from chat history if available
        clarification_context = self._extract_clarification_answers(chat_history)

        # Build prompt using prompt builder
        prompt_builder = AnalysisPromptBuilder()
        prompt = prompt_builder.build_analysis_prompt(
            query=original_query,
            deep_search_result=deep_search_result,
            chat_history=chat_history,
            mode=mode
        )

        # Add clarification context if available
        if clarification_context:
            prompt += f"\n\n**USER CLARIFICATION ANSWERS (CRITICAL - MUST BE CONSIDERED):**\n{clarification_context}\n\nThese answers refine the research scope. Use them when analyzing the query."

        try:
            system_prompt = "You are an expert research planner. Analyze queries to determine the best research approach."

            analysis = await llm.with_structured_output(QueryAnalysis).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

            logger.info("Query analyzed successfully",
                       complexity=analysis.complexity.get("level") if isinstance(analysis.complexity, dict) else analysis.complexity,
                       estimated_agents=analysis.estimated_agent_count if hasattr(analysis, "estimated_agent_count") else None,
                       session_id=session_id)

            # Extract agent count safely
            estimated_agent_count = 4  # Default
            if hasattr(analysis, "estimated_agent_count"):
                estimated_agent_count = analysis.estimated_agent_count
            elif isinstance(analysis.complexity, dict):
                estimated_agent_count = analysis.complexity.get("estimated_agents", 4)

            return {
                "query_analysis": analysis.dict() if hasattr(analysis, "dict") else analysis,
                "estimated_agent_count": estimated_agent_count
            }

        except Exception as e:
            logger.error("Query analysis failed", error=str(e), exc_info=True,
                        session_id=session_id)

            # Fallback
            return {
                "query_analysis": {
                    "reasoning": f"Fallback analysis due to error: {str(e)}",
                    "key_aspects": [original_query],
                    "complexity": {"level": "medium", "reasoning": "Default complexity", "estimated_agents": 4},
                    "research_strategy": "Comprehensive research approach",
                    "requires_clarification": False
                },
                "estimated_agent_count": 4
            }

    def _extract_clarification_answers(self, chat_history: list) -> str:
        """Extract user clarification answers from chat history.

        Args:
            chat_history: Chat history

        Returns:
            Clarification answers or empty string
        """
        if not chat_history:
            return ""

        # Look for clarification message followed by user answer
        for i, msg in enumerate(chat_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if "clarification" in content or "🔍" in content or "clarify" in content:
                    # Check if next message is from user
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        user_answer = chat_history[i + 1].get("content", "")
                        logger.info("Extracted clarification answers from chat history",
                                   answer_length=len(user_answer))
                        return user_answer

        return ""


# Legacy function wrapper for backward compatibility
async def analyze_query_node(state: ResearchState) -> Dict:
    """Legacy wrapper for AnalyzeQueryNode.

    This function maintains backward compatibility with existing code
    that imports analyze_query_node directly.

    TODO: Update imports to use AnalyzeQueryNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {
            "query_analysis": {
                "reasoning": "No runtime dependencies",
                "key_aspects": [state.get("query", "")],
                "complexity": {"level": "medium", "reasoning": "Default", "estimated_agents": 4},
                "research_strategy": "Default approach",
                "requires_clarification": False
            },
            "estimated_agent_count": 4
        }

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
    node = AnalyzeQueryNode(deps)
    return await node.execute(state)
