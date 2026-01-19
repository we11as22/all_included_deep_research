"""Supervisor review node for coordinating research."""

import structlog
import re
from typing import Dict, Any
from datetime import datetime

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.nodes.utils import _restore_runtime_deps
from src.workflow.research.supervisor_agent import run_supervisor_agent

logger = structlog.get_logger(__name__)


class SupervisorReviewNode(ResearchNode):
    """Supervisor reviews agent progress and coordinates research."""

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute supervisor review node.

        Args:
            state: Current research state

        Returns:
            State updates with supervisor decisions
        """
        # Convert ResearchState to dict for compatibility
        if isinstance(state, dict):
            state_dict = state
        else:
            state_dict = dict(state)
        
        # Restore runtime dependencies if not in state
        state_dict = _restore_runtime_deps(state_dict)
        
        llm = state_dict.get("llm")
        stream = state_dict.get("stream")
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
                    state_dict["should_continue"] = False
                    state_dict["replanning_needed"] = False
                    # CRITICAL: Set flag to bypass supervisor call limit for this finalization call
                    state_dict["_force_supervisor_finalization"] = True
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
            settings = state_dict.get("settings")
            if settings:
                supervisor_max_iterations = settings.deep_research_supervisor_max_iterations
            else:
                from src.config.settings import get_settings
                settings_obj = get_settings()
                supervisor_max_iterations = settings_obj.deep_research_supervisor_max_iterations
            
            decision = await run_supervisor_agent(
                state=state_dict,
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
                    findings = state_dict.get("findings", state_dict.get("agent_findings", []))
                    if findings:
                        # Read current draft_report
                        try:
                            draft_content = await agent_memory_service.file_manager.read_file("draft_report.md")
                        except FileNotFoundError:
                            draft_content = ""
                        
                        # CRITICAL: Check if draft_report already has chapters from supervisor
                        # If supervisor wrote chapters, DO NOT overwrite them!
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
                            query = state_dict.get("query", "")
                            
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
                "iteration": state_dict.get("iteration", 0) + 1,
                "completion_criteria_met": True
            }


# Legacy function wrapper for backward compatibility
async def supervisor_review_enhanced_node(state: ResearchState) -> Dict:
    """Legacy wrapper for SupervisorReviewNode.

    This function maintains backward compatibility with existing code
    that imports supervisor_review_enhanced_node directly.

    TODO: Update imports to use SupervisorReviewNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"should_continue": False}

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
    node = SupervisorReviewNode(deps)
    return await node.execute(state)
