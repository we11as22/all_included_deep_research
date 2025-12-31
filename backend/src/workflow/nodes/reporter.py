"""Final report generation node."""

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.workflow.state import ResearchFinding, ResearchState
from src.utils.text import summarize_text
from src.workflow.agentic.schemas import FinalReport

logger = structlog.get_logger(__name__)


REPORTER_SYSTEM_PROMPT = """You are a research report writer creating comprehensive, well-structured reports.

Your task is to synthesize research findings into a clear, informative report that:
- Directly answers the research question
- Integrates insights from all research findings
- Provides a cohesive narrative with proper flow
- Includes relevant citations and sources
- Highlights key takeaways and conclusions

Report Quality Standards:
- **Clarity**: Use clear, accessible language
- **Structure**: Organize information logically with sections
- **Evidence**: Support claims with specific findings
- **Completeness**: Address all aspects of the query
- **Citations**: Reference sources appropriately

Citation rules:
- Use inline citations like [1], [2] that match the numbered source catalog.
- Number sources sequentially without gaps."""


def _format_findings_for_report(findings: list[ResearchFinding]) -> str:
    """Format research findings for report generation."""

    if not findings:
        return "No research findings available."

    formatted = ""

    for idx, finding in enumerate(findings, 1):
        formatted += f"\n### Research Finding {idx}: {finding.topic}\n\n"
        formatted += f"**Summary:** {finding.summary}\n\n"

        if finding.key_findings:
            formatted += "**Key Insights:**\n"
            for insight in finding.key_findings:
                formatted += f"- {insight}\n"
            formatted += "\n"

        if finding.sources:
            formatted += "**Sources:**\n"
            for source in finding.sources[:5]:  # Limit to top 5
                formatted += f"- [{source.title}]({source.url})\n"
            formatted += "\n"

        formatted += f"**Confidence:** {finding.confidence}\n\n"
        formatted += "---\n"

    return formatted.strip()


def _format_sources_catalog(findings: list[ResearchFinding]) -> str:
    sources = []
    seen = set()
    for finding in findings:
        for source in finding.sources:
            if source.url in seen:
                continue
            seen.add(source.url)
            sources.append((source.title, source.url))

    if not sources:
        return "No sources available."

    lines = ["Sources Catalog:"]
    for idx, (title, url) in enumerate(sources, 1):
        lines.append(f"[{idx}] {title}: {url}")
    return "\n".join(lines)


async def generate_final_report_node(
    state: ResearchState,
    llm: any,  # LangChain LLM for report generation
) -> dict:
    """
    Generate final research report from all findings.

    Args:
        state: Current research state with findings
        llm: LangChain LLM for report generation

    Returns:
        State update with final_report
    """
    query = state.get("clarified_query") or state.get("query", "")
    findings = state.get("findings", [])
    mode = state.get("mode", "balanced")
    memory_context = state.get("memory_context", [])
    stream = state.get("stream")

    logger.info(
        "Generating final report",
        query=query,
        findings_count=len(findings),
        mode=mode,
    )

    # Format findings
    findings_text = _format_findings_for_report(findings)
    sources_catalog = _format_sources_catalog(findings)

    memory_block = ""
    if memory_context:
        memory_block = "\n## Memory Context\n" + "\n".join(
            [
                f"- {ctx.file_title} ({ctx.file_path}): {summarize_text(ctx.content, 3000)}"
                for ctx in memory_context[:3]
            ]
        )

    # Create report prompt
    user_prompt = f"""Research Question: {query}

Research Mode: {mode}

# Research Findings

{findings_text}
{memory_block}

{sources_catalog}

---

Please create a comprehensive research report that:
1. Provides a clear executive summary
2. Addresses the research question thoroughly
3. Integrates insights from all findings
4. Organizes information into logical sections
5. Includes citations to sources
6. Ends with key takeaways

Structure your report with these sections:
- **Executive Summary** (2-3 paragraphs)
- **Detailed Findings** (organized by theme/topic)
- **Key Takeaways** (3-5 bullet points)
- **Sources** (list of all sources cited)

Write in a clear, informative style appropriate for {mode} mode and include inline citations like [1].
Return JSON with fields reasoning and report."""

    messages = [
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        structured_llm = llm.with_structured_output(FinalReport, method="function_calling")
        response = await structured_llm.ainvoke(messages)
        if not isinstance(response, FinalReport):
            raise ValueError("FinalReport response was not structured")
        report = response.report

        logger.info(
            "Final report generated",
            report_length=len(report),
        )

        if stream and report:
            for i in range(0, len(report), 200):
                stream.emit_report_chunk(report[i : i + 200])
            stream.emit_final_report(report)

        return {
            "final_report": {"type": "override", "value": report}
        }

    except Exception as e:
        logger.error("Report generation failed", error=str(e))

        # Fallback: create simple report
        fallback_report = _create_fallback_report(query, findings, memory_context)

        if stream:
            for i in range(0, len(fallback_report), 200):
                stream.emit_report_chunk(fallback_report[i : i + 200])
            stream.emit_final_report(fallback_report)

        return {
            "final_report": {"type": "override", "value": fallback_report}
        }


def _create_fallback_report(
    query: str, findings: list[ResearchFinding], memory_context: list = None
) -> str:
    """Create a simple fallback report if generation fails."""

    report = f"# Research Report: {query}\n\n"
    report += "## Summary\n\n"

    if findings:
        report += "Based on research findings:\n\n"

        for finding in findings:
            report += f"### {finding.topic}\n\n"
            report += f"{finding.summary}\n\n"

            if finding.key_findings:
                for kf in finding.key_findings:
                    report += f"- {kf}\n"
                report += "\n"

        # Add sources
        report += "## Sources\n\n"
        report += _format_sources_catalog(findings).replace("Sources Catalog:", "").strip()
        report += "\n"

        if memory_context:
            report += "\n## Memory Sources\n\n"
            for ctx in memory_context[:5]:
                report += f"- {ctx.file_title} ({ctx.file_path})\n"

    else:
        report += "No research findings were generated.\n"

    return report
