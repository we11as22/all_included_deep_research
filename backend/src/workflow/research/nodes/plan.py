"""Planning node for research planning."""

import asyncio
import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import ResearchPlan, ResearchTopic
from src.workflow.research.prompts.planning import PlanningPromptBuilder

try:
    from openai import PermissionDeniedError
except ImportError:
    # Fallback if openai is not available
    PermissionDeniedError = Exception

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
        # CRITICAL: Use original_query for planning, not current query which might be clarification answer!
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

        # CRITICAL: Log all context data before planning (лизонинг)
        logger.info("🔍 PLANNING: Context data check",
                   session_id=session_id,
                   original_query=original_query[:100] if original_query else None,
                   query=query[:100] if query else None,
                   has_deep_search_result="deep_search_result" in state,
                   chat_history_length=len(chat_history),
                   note="Verifying all context data is available for planning")
        
        # Get deep_search_result for context
        deep_search_result_raw = state.get("deep_search_result", "")
        if isinstance(deep_search_result_raw, dict):
            deep_search_result = deep_search_result_raw.get("value", "")
        else:
            deep_search_result = deep_search_result_raw or ""
        
        logger.info("🔍 PLANNING: Deep search result",
                   session_id=session_id,
                   result_length=len(deep_search_result) if deep_search_result else 0,
                   result_preview=deep_search_result[:200] if deep_search_result else None,
                   note="Deep search result loaded for planning")

        # CRITICAL: Check if clarification was needed but not answered
        # If clarification_needed=True but clarification_answers is empty, we should NOT create plan
        clarification_needed = state.get("clarification_needed", False)
        session_status = state.get("session_status", "active")
        
        # CRITICAL: Use clarification_answers from session state (loaded from DB), not chat_history
        # clarification_answers is the source of truth, loaded from session in create_initial_state
        clarification_answers = state.get("clarification_answers", "")
        
        # Fallback: if not in state, try to extract from chat_history (for backward compatibility)
        if not clarification_answers:
            clarification_answers = self._extract_clarification_answers(chat_history)
            if clarification_answers:
                logger.warning("Using clarification_answers from chat_history (fallback) - should be in session state",
                             session_id=session_id,
                             note="This should not happen in normal flow - clarification_answers should be in session")
        
        # CRITICAL: If clarification was needed but not answered, STOP and wait
        # This prevents creating plan before user answers clarification questions
        if clarification_needed and session_status == "waiting_clarification" and not clarification_answers:
            logger.error("❌ CRITICAL: Cannot create research plan - clarification needed but not answered",
                       session_id=session_id,
                       clarification_needed=clarification_needed,
                       session_status=session_status,
                       has_clarification_answers=bool(clarification_answers),
                       note="Research plan should NOT be created before user answers clarification questions")
            if stream:
                stream.emit_status("⏸️ Waiting for clarification answers before creating research plan...",
                                 step="planning")
            return {
                "planning_waiting": True,
                "should_stop": True,
                "error": "clarification_answers_required"
            }
        
        logger.info("🔍 PLANNING: Clarification answers",
                   session_id=session_id,
                   has_clarification_answers=bool(clarification_answers),
                   clarification_preview=clarification_answers[:200] if clarification_answers else None,
                   source="session_state" if state.get("clarification_answers") else "chat_history_fallback",
                   clarification_needed=clarification_needed,
                   session_status=session_status,
                   note="Clarification answers from session state (source of truth)")

        # Build prompt using prompt builder
        # CRITICAL: Use original_query for planning, not query (which might be clarification answer)
        # Planning should be about the ORIGINAL research topic, not the clarification answers
        prompt_builder = PlanningPromptBuilder()
        prompt = prompt_builder.build_planning_prompt(
            query=original_query,  # CRITICAL: Use original_query - research is about the original topic
            query_analysis=query_analysis,
            deep_search_result=deep_search_result,
            clarification_answers=clarification_answers,  # Clarification answers refine the approach, but topic is original_query
            mode=mode
        )
        
        # CRITICAL: Log prompt length for debugging "Blocked by Google" errors
        # Large prompts or specific content might trigger provider's content filter
        prompt_length = len(prompt)
        logger.info("Planning prompt built",
                   prompt_length=prompt_length,
                   query_length=len(query),
                   deep_search_length=len(deep_search_result),
                   clarification_length=len(clarification_answers),
                   session_id=session_id,
                   note="Large prompts (>50k chars) or specific content might trigger provider filters")

        try:
            system_prompt = """You are an expert research planner. Create comprehensive research plans with clear topics and priorities.

CRITICAL: All topics must relate to the original query. Include query context in topic descriptions."""

            # CRITICAL: Add timeout to prevent hanging (120 seconds for planning)
            # This prevents the workflow from hanging indefinitely if LLM is slow or unresponsive
            logger.info("Calling LLM for research planning", 
                       prompt_length=prompt_length,
                       session_id=session_id,
                       note="Using 120s timeout to prevent hanging")
            
            try:
                plan = await asyncio.wait_for(
                    llm.with_structured_output(ResearchPlan).ainvoke([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]),
                    timeout=120.0  # 120 seconds timeout
                )
            except asyncio.TimeoutError:
                logger.error("LLM call timed out after 120 seconds",
                           session_id=session_id,
                           prompt_length=prompt_length,
                           note="Planning LLM call exceeded timeout - creating fallback plan")
                # Create fallback plan instead of raising error
                plan = ResearchPlan(
                    reasoning=f"Fallback research plan created due to LLM timeout. Research will focus on: {query}",
                    research_depth="comprehensive",
                    coordination_strategy="parallel",
                    topics=[
                        ResearchTopic(
                            topic=query,
                            description=f"Comprehensive research on: {query}",
                            priority="high",
                            estimated_sources=10
                        )
                    ]
                )
            except PermissionDeniedError as e:
                # CRITICAL: If LLM is blocked (403), create fallback plan instead of failing
                logger.error("LLM call blocked by provider (403) - creating fallback plan",
                           session_id=session_id,
                           prompt_length=prompt_length,
                           error=str(e),
                           note="LLM provider blocked request. Creating fallback plan to ensure research continues.")
                plan = ResearchPlan(
                    reasoning=f"Fallback research plan created due to LLM provider blocking. Research will focus on: {query}",
                    research_depth="comprehensive",
                    coordination_strategy="parallel",
                    topics=[
                        ResearchTopic(
                            topic=query,
                            description=f"Comprehensive research on: {query}",
                            priority="high",
                            estimated_sources=10
                        )
                    ]
                )
            except Exception as e:
                # CRITICAL: Catch any other LLM errors and create fallback plan
                error_type = type(e).__name__
                logger.error("LLM call failed - creating fallback plan",
                           session_id=session_id,
                           prompt_length=prompt_length,
                           error=str(e),
                           error_type=error_type,
                           note="LLM call failed. Creating fallback plan to ensure research continues.")
                plan = ResearchPlan(
                    reasoning=f"Fallback research plan created due to LLM error. Research will focus on: {query}",
                    research_depth="comprehensive",
                    coordination_strategy="parallel",
                    topics=[
                        ResearchTopic(
                            topic=query,
                            description=f"Comprehensive research on: {query}",
                            priority="high",
                            estimated_sources=10
                        )
                    ]
                )

            logger.info("Research plan created",
                       topics_count=len(plan.topics) if hasattr(plan, "topics") else 0,
                       session_id=session_id)

            # Extract topics safely
            topics = []
            if hasattr(plan, "topics") and plan.topics:
                topics = plan.topics

            # CRITICAL: Build research_plan_dict matching original format
            # Original returned: {"reasoning": ..., "research_depth": ..., "coordination_strategy": ...}
            # ResearchPlan model has research_depth and coordination_strategy fields, so access them directly
            research_plan_dict = {
                "reasoning": plan.reasoning,
                "research_depth": plan.research_depth,  # Direct access as in original backup
                "coordination_strategy": plan.coordination_strategy  # Direct access as in original backup
            }
            
            # CRITICAL: Save research plan to main.md for persistence and supervisor editing
            # This matches the original implementation
            # Original uses stream.app_state.get("agent_memory_service") directly
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
                        for topic in topics
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
                               topics_count=len(topics),
                               session_id=session_id)
                except Exception as e:
                    logger.warning("Failed to save research plan to main.md", error=str(e), exc_info=True)

            # CRITICAL: Return format must match original - use "research_topics" not "topics"
            return {
                "research_plan": research_plan_dict,
                "research_topics": [t.dict() if hasattr(t, "dict") else t for t in topics]
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
            
            # Fallback: create basic plan matching original format
            # CRITICAL: Original creates ResearchTopic object, not dict
            fallback_topic = ResearchTopic(
                topic=query,
                description=f"Research: {query}",
                priority="high",
                estimated_sources=5  # Default estimate for fallback
            )
            
            # Create fallback reasoning based on error type
            if is_permission_error:
                fallback_reasoning = "Fallback plan due to planning error"
            elif is_rate_limit:
                fallback_reasoning = "Fallback plan due to planning error"
            elif is_timeout:
                fallback_reasoning = "Fallback plan due to planning error"
            else:
                fallback_reasoning = "Fallback plan due to planning error"

            logger.info("Using fallback research plan",
                       session_id=session_id,
                       note="Research will continue despite planning error")

            # CRITICAL: Return format must match original - use "research_topics" not "topics"
            # Original returns ResearchTopic.dict(), not list of dicts
            return {
                "research_plan": {
                    "reasoning": fallback_reasoning,
                    "research_depth": "standard",
                    "coordination_strategy": "Parallel research"
                },
                "research_topics": [fallback_topic.dict()]
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
