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
        session_id = state.get("session_id")
        if not session_id:
            logger.warning("session_id not found in state - using 'unknown' for logging", state_keys=list(state.keys())[:10])
            session_id = "unknown"
        chat_history = state.get("chat_history", [])

        # Access dependencies
        llm = self.deps.llm
        stream = self.deps.stream

        # Log LLM configuration for debugging
        llm_info = {}
        if hasattr(llm, "_client"):
            client = llm._client
            if hasattr(client, "model_name"):
                llm_info["model"] = client.model_name
            if hasattr(client, "openai_api_base"):
                llm_info["base_url"] = client.openai_api_base
            if hasattr(client, "openai_api_key"):
                # Only log first 10 chars of API key for security
                api_key_preview = client.openai_api_key[:10] + "..." if client.openai_api_key else None
                llm_info["api_key_preview"] = api_key_preview
        
        logger.info("Research planning - LLM configuration",
                   llm_info=llm_info,
                   session_id=session_id,
                   note="This helps identify which provider/model is causing 'Blocked by Google' errors")
        
        # CRITICAL: Check if LLM has max_retries configured
        if hasattr(llm, "_client"):
            client = llm._client
            if hasattr(client, "max_retries"):
                logger.info("LLM retry configuration",
                           max_retries=client.max_retries,
                           session_id=session_id,
                           note="Retry is configured on LLM client")
            else:
                logger.warning("LLM client missing max_retries",
                             session_id=session_id,
                             note="Retry may not be configured - this could cause issues with transient errors")

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
        
        # CRITICAL: Log prompt length for debugging "Blocked by Google" errors
        # Large prompts or specific content might trigger provider's content filter
        prompt_length = len(prompt)
        logger.info("Planning prompt built",
                   prompt_length=prompt_length,
                   query_length=len(original_query),
                   deep_search_length=len(deep_search_result),
                   clarification_length=len(clarification_answers),
                   session_id=session_id,
                   note="Large prompts (>50k chars) or specific content might trigger provider filters")

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
            error_str = str(e)
            error_type = type(e).__name__
            
            # Check for specific API errors
            is_permission_error = "PermissionDeniedError" in error_type or "403" in error_str or "Blocked by Google" in error_str
            is_rate_limit = "RateLimitError" in error_type or "429" in error_str or "rate limit" in error_str.lower()
            is_timeout = "TimeoutError" in error_type or "timeout" in error_str.lower()
            
            # Log error with context including LLM info
            logger.error("Research planning failed", 
                        error=error_str[:500],  # Limit error message length
                        error_type=error_type,
                        is_permission_error=is_permission_error,
                        is_rate_limit=is_rate_limit,
                        is_timeout=is_timeout,
                        llm_info=llm_info,
                        session_id=session_id,
                        exc_info=True,
                        note="'Blocked by Google' usually means: 1) Using Google Gemini model via OpenRouter/other provider, 2) Provider's content policy blocking, 3) Region/IP restrictions")
            
            # Emit user-friendly error message via stream
            if stream:
                if is_permission_error:
                    stream.emit_status("⚠️ API access denied - using fallback plan", step="planning")
                elif is_rate_limit:
                    stream.emit_status("⚠️ API rate limit reached - using fallback plan", step="planning")
                elif is_timeout:
                    stream.emit_status("⚠️ API timeout - using fallback plan", step="planning")
                else:
                    stream.emit_status("⚠️ Planning error - using fallback plan", step="planning")
            
            # Fallback: create basic plan
            fallback_topics = self._create_fallback_topics(original_query, query_analysis)
            
            # Create fallback reasoning based on error type
            if is_permission_error:
                fallback_reasoning = "Fallback plan created due to API access restrictions. Research will continue with basic topics."
            elif is_rate_limit:
                fallback_reasoning = "Fallback plan created due to API rate limiting. Research will continue with basic topics."
            elif is_timeout:
                fallback_reasoning = "Fallback plan created due to API timeout. Research will continue with basic topics."
            else:
                fallback_reasoning = f"Fallback plan created due to error: {error_type}. Research will continue with basic topics."

            logger.info("Using fallback research plan",
                       topics_count=len(fallback_topics),
                       session_id=session_id,
                       note="Research will continue despite planning error")

            return {
                "research_plan": {
                    "reasoning": fallback_reasoning,
                    "topics": fallback_topics,
                    "stop": False  # CRITICAL: Don't stop research, continue with fallback plan
                },
                "topics": fallback_topics,
                "coordination_notes": "Basic fallback plan - research will continue"
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
