"""Compress findings node for consolidating research results."""

import structlog
from typing import Dict, Any

from src.workflow.research.state import ResearchState
from src.workflow.research.nodes.base import ResearchNode
from src.workflow.research.models import CompressedFindings

logger = structlog.get_logger(__name__)


class CompressFindingsNode(ResearchNode):
    """Compress and consolidate all research findings.

    This node takes all agent findings and compresses them into a structured summary
    for the final report generation.
    """

    async def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute findings compression node.

        Args:
            state: Current research state

        Returns:
            State updates with compressed_research
        """
        findings = state.get("findings", [])
        query = state.get("query", "")
        original_query = state.get("original_query", query)
        session_id = state.get("session_id", "unknown")

        # Access dependencies
        llm = self.deps.llm
        stream = self.deps.stream

        if stream:
            stream.emit_status("Compressing research findings...", step="compression")

        logger.info("Compressing findings",
                   findings_count=len(findings),
                   session_id=session_id)

        if not findings:
            logger.warning("No findings to compress", session_id=session_id)
            return {
                "compressed_research": f"No research findings available for: {original_query}"
            }

        # Concatenate all findings
        all_findings_text = "\n\n---\n\n".join([
            f"**Finding {i+1}:**\n{finding.get('content', finding.get('note', str(finding)))}"
            for i, finding in enumerate(findings)
        ])

        # Truncate if too long (LLM context limit)
        max_length = 15000  # characters
        if len(all_findings_text) > max_length:
            logger.warning("Findings too long, truncating",
                          original_length=len(all_findings_text),
                          truncated_length=max_length)
            all_findings_text = all_findings_text[:max_length] + "\n\n[... truncated ...]"

        prompt = f"""Compress and synthesize these research findings into a structured summary.

**Original Query:** {original_query}

**Research Findings:**
{all_findings_text}

Provide a comprehensive synthesis that:
1. Organizes findings by theme/topic
2. Highlights key insights and discoveries
3. Removes redundancy and noise
4. Maintains important details and sources
5. Structures information logically

Focus on creating a well-organized summary that will be used for the final report.
"""

        try:
            system_prompt = """You are an expert research synthesizer. Compress and organize research findings into clear, structured summaries.

Provide structured output with:
- summary: Overall synthesis of findings
- key_insights: List of most important discoveries
- themes: Major themes/topics covered
- sources_count: Approximate number of unique sources"""

            result = await llm.with_structured_output(CompressedFindings).ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

            logger.info("Findings compressed successfully",
                       summary_length=len(result.summary) if hasattr(result, "summary") else 0,
                       key_insights_count=len(result.key_insights) if hasattr(result, "key_insights") else 0,
                       session_id=session_id)

            # Format compressed research
            compressed = self._format_compressed_findings(result)

            return {"compressed_research": compressed}

        except Exception as e:
            logger.error("Findings compression failed", error=str(e), exc_info=True,
                        session_id=session_id)

            # Fallback: simple concatenation
            fallback = f"""# Research Findings for: {original_query}

## All Findings

{all_findings_text}

---

*Note: This is a fallback summary due to compression error: {str(e)}*
"""
            return {"compressed_research": fallback}

    def _format_compressed_findings(self, result: CompressedFindings) -> str:
        """Format compressed findings into markdown.

        Args:
            result: CompressedFindings object

        Returns:
            Formatted markdown string
        """
        sections = []

        # Summary
        if hasattr(result, "summary") and result.summary:
            sections.append(f"## Summary\n\n{result.summary}")

        # Key insights
        if hasattr(result, "key_insights") and result.key_insights:
            insights_text = "\n".join([f"- {insight}" for insight in result.key_insights])
            sections.append(f"## Key Insights\n\n{insights_text}")

        # Themes
        if hasattr(result, "themes") and result.themes:
            themes_text = "\n".join([f"- {theme}" for theme in result.themes])
            sections.append(f"## Major Themes\n\n{themes_text}")

        # Sources count
        if hasattr(result, "sources_count") and result.sources_count:
            sections.append(f"## Sources\n\nAnalyzed approximately {result.sources_count} unique sources.")

        return "\n\n".join(sections)


# Legacy function wrapper for backward compatibility
async def compress_findings_node(state: ResearchState) -> Dict:
    """Legacy wrapper for CompressFindingsNode.

    This function maintains backward compatibility with existing code
    that imports compress_findings_node directly.

    TODO: Update imports to use CompressFindingsNode class directly,
    then remove this wrapper.
    """
    from src.workflow.research.nodes import runtime_deps_context

    runtime_deps = runtime_deps_context.get()
    if not runtime_deps:
        logger.warning("Runtime dependencies not found in context")
        return {"compressed_research": "No runtime dependencies available"}

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
    node = CompressFindingsNode(deps)
    return await node.execute(state)
