"""Planning node for research planning."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import ResearchPlan
from src.workflow.research.prompts.planning import PlanningPromptBuilder

logger = structlog.get_logger(__name__)


class PlanResearchNode(ResearchNode):
    """Create detailed research plan with structured output.

    Generates research topics, priorities, and coordination strategy.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute research planning node.

        Args:
            state: Current research state

        Returns:
            State updates with research_plan
        """
        query = state.get("query", "")
        original_query = state.get("original_query", query)
        query_analysis = state.get("query_analysis", {})
        mode = state.get("mode", "quality")
        session_id = state.get("session_id", "unknown")
        chat_history = state.get("chat_history", [])

        # Access dependencies
        llm = self.deps.llm
        stream = self.deps.stream

        if stream:
            stream.emit_status("Creating research plan...", step="planning")

        # Get deep_search_result for context
        deep_search_result_raw = state.get("deep_search_result", "")
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "")
        else:
            deep_search_result = deep_search_result_raw or ""

        # Extract clarification answers
        clarification_answers = self._extract_clarification_answers(chat_history)

        # Build prompt using prompt builder
        prompt_builder = PlanningPromptBuilder()
        prompt = prompt_builder.build_planning_prompt(
            query=original_query,
            query_analysis=query_analysis,
            deep_search_result=deep_search_result,
            clarification_answers=clarification_answers,
            mode=mode
        )

        try:
            system_prompt = """You are an expert research planner. Create comprehensive research plans with clear topics and priorities.

CRITICAL: All topics must relate to the original query. Include query context in topic descriptions."""

            plan = await llm.with_structured_output(ResearchPlan).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

            logger.info("Research plan created",
                       topics_count=len(plan.topics) if hasattr(plan, "topics") else 0,
                       session_id=session_id)

            # Extract topics safely
            topics = []
            if hasattr(plan, "topics") and plan.topics:
                topics = plan.topics

            return {
                "research_plan": plan.dict() if hasattr(plan, "dict") else plan,
                "topics": [t.dict() if hasattr(t, "dict") else t for t in topics],
                "coordination_notes": plan.reasoning if hasattr(plan, "reasoning") else ""
            }

        except Exception as e:
            logger.error("Research planning failed", error=str(e), exc_info=True,
                        session_id=session_id)

            # Fallback: create basic plan
            fallback_topics = self._create_fallback_topics(original_query, query_analysis)

            return {
                "research_plan": {
                    "reasoning": f"Fallback plan due to error: {str(e)}",
                    "topics": fallback_topics,
                    "stop": False
                },
                "topics": fallback_topics,
                "coordination_notes": "Basic fallback plan"
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
                if "clarification" in content or "🔍" in content:
                    # Check if next message is from user
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "user":
                        return chat_history[i + 1].get("content", "")

        return ""

    def _create_fallback_topics(self, query: str, query_analysis: dict) -> list:
        """Create fallback research topics.

        Args:
            query: Original query
            query_analysis: Query analysis results

        Returns:
            List of fallback topics
        """
        key_aspects = query_analysis.get("key_aspects", [])

        if not key_aspects:
            # Create basic topics from query
            return [
                {
                    "reasoning": "Overview and background",
                    "topic": f"Overview of {query}",
                    "description": f"Research the basics and background of {query}",
                    "priority": "high"
                },
                {
                    "reasoning": "Key details and specifics",
                    "topic": f"Key aspects of {query}",
                    "description": f"Investigate specific details and important aspects of {query}",
                    "priority": "high"
                },
                {
                    "reasoning": "Current state and trends",
                    "topic": f"Current state and trends for {query}",
                    "description": f"Explore current developments and trends related to {query}",
                    "priority": "medium"
                }
            ]

        # Create topics from key aspects
        topics = []
        for i, aspect in enumerate(key_aspects[:6]):  # Max 6 topics
            priority = "high" if i < 2 else "medium" if i < 4 else "low"
            topics.append({
                "reasoning": f"Investigate {aspect}",
                "topic": aspect,
                "description": f"Research and analyze {aspect} in the context of {query}",
                "priority": priority
            })

        return topics


# Legacy function wrapper for backward compatibility
async def plan_research_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for PlanResearchNode.

    This function maintains backward compatibility with existing code
    that imports plan_research_enhanced_node directly.

    TODO: Update imports to use PlanResearchNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {
            "research_plan": {
                "reasoning": "No runtime dependencies",
                "topics": [],
                "stop": False
            },
            "topics": [],
            "coordination_notes": ""
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
    node = PlanResearchNode(deps)
    return await node.execute(state)
