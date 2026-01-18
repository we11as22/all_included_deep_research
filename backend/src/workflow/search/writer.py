"""Writer agent for final answer synthesis with citations.

Second stage of the Perplexica two-stage architecture.
Synthesizes research results into cited answers.
"""

import re
import structlog
from typing import Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.workflow.search.classifier import get_current_date, format_chat_history

logger = structlog.get_logger(__name__)


# ==================== Schemas ====================


class CitedAnswer(BaseModel):
    """Answer with inline citations."""

    reasoning: str = Field(description="Why the sources support this answer")
    answer: str = Field(
        description="Final answer with inline citations [1], [2], etc. in markdown format. MUST use proper markdown: ## for main sections (NOT #), ### for subsections, **bold**, *italic*, lists, links. Do NOT use plain text with large letters - use markdown headings!"
    )
    citations: list[str] = Field(
        description="List of source URLs as strings: ['https://example.com', 'https://example2.com', ...]"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in answer based on source quality and coverage"
    )


# ==================== Writer Prompts ====================


def get_writer_prompt(mode: str) -> str:
    """Get mode-specific writer prompt."""

    current_date = get_current_date()

    base_prompt = f"""You are an expert writer synthesizing research into comprehensive answers.

Current date: {current_date}

Your role:
1. Read the research sources provided
2. Synthesize information into a clear, well-structured answer
3. CITE EVERY FACT with inline citations [1], [2], etc.
4. Provide a list of all sources at the end
5. Be accurate and truthful

CRITICAL CITATION RULES:
- Every factual statement MUST have a citation
- Use inline citations: "According to [1], ..."
- Multiple sources for important claims: "Studies show [1][2] that..."
- Never make claims without sources
- If sources conflict, mention both: "While [1] suggests X, [2] indicates Y"

CRITICAL SOURCES SECTION FORMAT:
- At the END of your answer, you MUST add a "## Sources" section
- Format: Each source on a new line as "- [Title](URL)" (NOT "[1] [Title](URL)")
- Example Sources section:
  ## Sources

  - [Source Title 1](https://example.com/1)
  - [Source Title 2](https://example.com/2)
- Include the Sources section directly in your answer field - it will be part of the markdown
- The Sources section format matches deep_research draft_report format

CRITICAL FORMATTING REQUIREMENTS:
- You MUST use markdown formatting throughout your entire answer
- Use ## for main sections, ### for subsections (NOT # for main title - start with ##)
- Use **bold** for emphasis, *italic* for subtle emphasis
- Use proper markdown lists (- for unordered, 1. for ordered)
- Use proper markdown links: [text](url)
- Structure with clear sections using markdown headings (##, ###)
- Write like a knowledgeable blog post with proper markdown formatting
- Be engaging but factual
- CRITICAL: Your answer field MUST be valid markdown - use markdown syntax, not plain text!
- **CRITICAL NEWLINE FORMATTING**: Use TWO newlines (blank line) between paragraphs for proper markdown rendering!
- Each paragraph must be separated by a blank line (two newlines: \n\n)!
- Sections must be separated by blank lines!
- This ensures proper markdown rendering on the frontend - without blank lines, paragraphs will merge together!
- **EXAMPLE OF CORRECT FORMATTING**:
  ```
  ## Heading 1

  This is paragraph 1. It has multiple sentences. Each sentence ends with punctuation.

  This is paragraph 2. Notice the blank line (two newlines) between paragraphs.

  ### Subheading

  This is paragraph 3 after subheading.
  ```
- **WRONG FORMATTING** (DO NOT DO THIS):
  ```
  ## Heading 1
  This is paragraph 1. It has multiple sentences.
  This is paragraph 2. No blank line - paragraphs will merge!
  ```
"""

    if mode == "speed":
        return base_prompt + """
MODE: SPEED

Provide a complete, informative answer:
- 600-1000 words minimum (NOT 400-600 - be comprehensive!)
- Focus on key points but cover them FULLY and in detail
- Use ALL provided sources - each source has valuable information
- Clear structure with markdown sections (## for main sections, ### for subsections)
- Still cite everything!
- CRITICAL: Use proper markdown formatting - ## for sections, **bold**, *italic*, lists

IMPORTANT: Don't just summarize snippets - synthesize information from ALL sources into a comprehensive answer with proper markdown formatting.
"""

    elif mode == "balanced":
        return base_prompt + """
MODE: BALANCED

Provide thorough, comprehensive coverage:
- 1200-2000 words minimum (NOT 800-1200 - be comprehensive and detailed!)
- Well-organized sections with clear markdown structure (## for main sections, ### for subsections)
- Use ALL provided sources - synthesize information from each
- Cover main aspects of the topic in depth with full details
- Include specific details, data, and examples from sources
- Compare different perspectives if sources provide them
- CRITICAL: Use proper markdown formatting throughout - ## for sections, **bold**, *italic*, lists, links

IMPORTANT: You have many sources available - use them all! Don't just pick a few.
Each source adds value - synthesize them into a complete picture with proper markdown formatting.
"""

    else:  # quality
        return base_prompt + """
MODE: QUALITY

Create an in-depth, comprehensive response:
- Minimum 1500-3000 words
- Multiple sections with clear structure (Introduction, Main sections, Conclusion)
- Cover the topic from ALL angles
- Include background, detailed analysis, implications, examples
- Like a comprehensive research report or detailed article
- Deep analysis with specific data and evidence from sources

CRITICAL: Use EVERY source provided! You have extensive research - leverage all of it!
- Synthesize information from all sources into a coherent narrative
- Include specific quotes, data, and facts from sources
- Compare and contrast different perspectives
- Build a complete, authoritative answer
"""


# ==================== Writer Agent ====================


async def writer_agent(
    query: str,
    research_results: dict[str, Any],
    llm: Any,
    stream: Any,
    mode: str = "balanced",
    chat_history: list[dict] = None,
) -> str:
    """
    Synthesize final answer with citations from research.

    Args:
        query: Original user query
        research_results: Results from research_agent (sources, scraped_content, reasoning)
        llm: LLM instance
        stream: Streaming generator
        mode: Research mode (affects answer length/depth)
        chat_history: Chat history for context

    Returns:
        Formatted answer with inline citations and sources section
    """
    chat_history = chat_history or []

    sources = research_results.get("sources", [])
    scraped = research_results.get("scraped_content", [])

    if not sources and not scraped:
        logger.warning("No sources available for writer agent")
        return "I couldn't find enough information to answer this question reliably."

    # Prepare citation-ready sources
    all_sources = []

    # Add search results (use full snippet - it's already short from search engine)
    for source in sources:
        all_sources.append({
            "title": source.get("title", "Untitled"),
            "url": source.get("url", ""),
            "content": source.get("snippet", "")  # Full snippet, no truncation
        })

    # Add scraped content (richer information with summaries!)
    for scraped_item in scraped:
        if "error" not in scraped_item:
            # Prefer summary (comprehensive 800-token context) over content
            # Content field already contains summary from scrape_url_handler
            content = scraped_item.get("content", "")
            
            # CRITICAL: Include markdown if available - helps writer preserve formatting
            markdown = scraped_item.get("markdown")
            if markdown:
                # Include markdown in content for better context (writer can use it)
                # Summary is already in content, but markdown provides original structure
                logger.debug(f"Including markdown for source", url=scraped_item.get("url"), markdown_length=len(markdown))

            all_sources.append({
                "title": scraped_item.get("title", "Untitled"),
                "url": scraped_item.get("url", ""),
                "content": content,  # Already summarized by LLM in scrape_url_handler
                "markdown": markdown  # CRITICAL: Preserve markdown for proper formatting context
            })

    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for source in all_sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)

    # Use all available sources - don't limit artificially
    # LLM can handle context and decide which to use
    # In speed mode: typically 3-5 sources
    # In balanced mode: typically 8-12 sources
    # In quality mode: typically 15-20 sources
    logger.info(f"Writer synthesizing from {len(unique_sources)} sources")

    # Build source context for LLM
    # CRITICAL: Include markdown if available to preserve formatting structure
    source_context_parts = []
    for i, src in enumerate(unique_sources):
        source_text = f"[{i+1}] **{src['title']}** ({src['url']})\n"
        
        # Prefer markdown if available (preserves structure), otherwise use content (summary)
        if src.get("markdown"):
            source_text += f"Original markdown content:\n{src['markdown']}\n\n"
            source_text += f"Summary:\n{src['content']}"
            logger.debug(f"Including markdown for source {i+1}", url=src.get("url"), has_markdown=True)
        else:
            source_text += src['content']
        
        source_context_parts.append(source_text)
    
        # CRITICAL: Use \n\n between sources to preserve markdown formatting (same as deep research)
        # This ensures proper paragraph separation in markdown
        source_context = "\n\n".join(source_context_parts)

    # CRITICAL: Detect user language from query
    def _detect_user_language(text: str) -> str:
        """Detect user language from query text."""
        if not text:
            return "English"
        try:
            from langdetect import detect
            detected = detect(text)
            if detected == "ru":
                return "Russian"
            elif detected == "en":
                return "English"
            elif detected == "es":
                return "Spanish"
            elif detected == "fr":
                return "French"
            elif detected == "de":
                return "German"
            elif detected == "zh-cn" or detected == "zh-tw":
                return "Chinese"
            # Check for Cyrillic (Russian, Ukrainian, etc.)
            if any('\u0400' <= char <= '\u04FF' for char in text):
                return "Russian"
            return "English"
        except Exception:
            # Fallback: check for Cyrillic
            if any('\u0400' <= char <= '\u04FF' for char in text):
                return "Russian"
            return "English"
    
    user_language = _detect_user_language(query)
    logger.info("Detected user language for writer", language=user_language, query_preview=query[:50])

    # Get writer prompt
    system_prompt = get_writer_prompt(mode)
    
    # CRITICAL: Add language instruction to system prompt
    system_prompt += f"\n\nCRITICAL LANGUAGE REQUIREMENT:\n"
    system_prompt += f"- Write your ENTIRE answer in {user_language}\n"
    system_prompt += f"- All text, headings, citations, and sources section must be in {user_language}\n"
    system_prompt += f"- Detect the language from the user's query and match it exactly\n"
    system_prompt += f"- Do NOT mix languages - use ONLY {user_language} throughout the entire answer\n"

    # Build user prompt
    sources_count = len(unique_sources)
    user_prompt = f"""Query: {query}

Research sources ({sources_count} sources provided - USE ALL OF THEM):
{source_context}

Write a comprehensive answer with inline citations [1], [2], etc. in the text.

CRITICAL MARKDOWN FORMATTING REQUIREMENT (MANDATORY):
- Your answer MUST be valid markdown with proper formatting - NOT plain text!
- Use ## for main sections (NOT # - start with ##)
- Use ### for subsections
- Use **bold** for emphasis, *italic* for subtle emphasis
- Use proper markdown lists (- for unordered, 1. for ordered)
- Use proper markdown links: [text](url)
- Structure with clear markdown sections - do NOT use plain text with large letters!
- CRITICAL: Format your answer as markdown, not plain text! Use markdown syntax!
- **CRITICAL NEWLINE FORMATTING**: Use TWO newlines (\n\n) between paragraphs and sections for proper markdown rendering!
- Each paragraph should be separated by a blank line (two newlines)!
- Sections should be separated by blank lines!
- This ensures proper markdown rendering on the frontend!

CRITICAL SOURCES FORMATTING REQUIREMENT:
- You MUST include inline citations [1], [2], [3], etc. in your answer text for every fact or claim
- At the END of your answer, you MUST add a "## Sources" section with ALL sources used
- Sources section format: Each source on a new line as "- [Title](URL)" (NOT "[1] [Title](URL)")
- Example Sources section:
  ## Sources

  - [Source Title 1](https://example.com/1)
  - [Source Title 2](https://example.com/2)
- Include the Sources section directly in your answer field - it will be part of the markdown
- The Sources section format matches deep_research draft_report format

CRITICAL LANGUAGE REQUIREMENT:
- Write your ENTIRE answer in {user_language} (the same language as the query)
- All text, headings, citations, and sources section must be in {user_language}
- Do NOT mix languages - use ONLY {user_language}

CRITICAL MARKDOWN FORMATTING REQUIREMENT (MANDATORY - NO EXCEPTIONS):
- Your answer field MUST be valid markdown with proper formatting - NOT plain text!
- You MUST use markdown syntax throughout the entire answer
- Use ## for main sections (NOT # - start with ##)
- Use ### for subsections
- Use **bold** for emphasis, *italic* for subtle emphasis
- Use proper markdown lists (- for unordered, 1. for ordered)
- Use proper markdown links: [text](url)
- Structure with clear markdown sections - do NOT use plain text!
- CRITICAL: Format your answer as markdown, not plain text with large letters!
- FORBIDDEN: Do NOT write plain text without markdown formatting!
- FORBIDDEN: Do NOT use large letters or bold text without markdown syntax!
- EXAMPLE CORRECT FORMAT:
  ## Main Section Title

  This is a paragraph with **bold text** and *italic text*.

  ### Subsection

  - List item 1
  - List item 2

IMPORTANT INSTRUCTIONS:
- Use information from ALL {sources_count} sources - each one has valuable content
- Synthesize information from all sources into a coherent, complete answer
- Don't just use the first few sources - leverage ALL available research
- Include specific details, data, and examples from different sources
- If sources provide different perspectives, present them all with citations
- Be comprehensive and detailed - don't write brief summaries!

Remember: CITE EVERY FACT. Return ONLY your answer in markdown format. Do NOT return JSON or any other format. Just the markdown answer.
"""

    try:
        if stream:
            stream.emit_status("Synthesizing answer...", step="synthesis")

        # CRITICAL: Use simple LLM call WITHOUT structured output to preserve markdown formatting!
        # Structured output (JSON) loses \n characters during JSON parsing
        # Deep Research uses simple llm.ainvoke() and preserves formatting - we should do the same!
        
        # CRITICAL: Log max_tokens to verify it's correct
        max_tokens_value = None
        if hasattr(llm, "max_tokens"):
            max_tokens_value = llm.max_tokens
        
        logger.info(
            "Writer agent calling LLM (simple call, no structured output)",
            mode=mode,
            sources_count=len(unique_sources),
            prompt_length=len(user_prompt),
            max_tokens=max_tokens_value,
            note="Using simple llm.ainvoke() like DeepSearchNode to preserve markdown formatting"
        )

        result = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # CRITICAL: Get answer directly from LLM response (like DeepSearchNode)
        final_answer = result.content if hasattr(result, 'content') else str(result)
        
        # CRITICAL: Log EXACTLY what LLM returned - BEFORE any processing
        if final_answer:
            import re
            raw_newline_count = final_answer.count('\n')
            raw_double_newline_count = final_answer.count('\n\n')
            raw_triple_newline_count = final_answer.count('\n\n\n')
            has_markdown_headings = bool(re.search(r'^#{2,}\s+', final_answer, re.MULTILINE))
            
            logger.info(
                "RAW LLM RESPONSE (before any processing)",
                answer_length=len(final_answer),
                newline_count=raw_newline_count,
                double_newline_count=raw_double_newline_count,
                triple_newline_count=raw_triple_newline_count,
                has_markdown_headings=has_markdown_headings,
                first_200_chars=repr(final_answer[:200]),  # Use repr to see actual \n characters
                last_200_chars=repr(final_answer[-200:]) if len(final_answer) > 200 else repr(final_answer),
                note="This is EXACTLY what LLM returned - no modifications yet"
            )
            
            # Show sample of actual content with newlines visible
            sample_lines = final_answer.split('\n')[:10]
            logger.debug(
                "First 10 lines of raw LLM response (showing actual newlines)",
                lines=[repr(line) for line in sample_lines],
                note="Use repr() to see actual \\n characters"
            )
            original_newline_count = final_answer.count('\n')
            original_double_newline_count = final_answer.count('\n\n')
            
            # Step 1: Always ensure headings have \n\n after them (if not already present)
            # Match: heading followed by single \n and non-heading content
            final_answer = re.sub(r'(#{2,}\s+[^\n]+)\n([^\n#\s])', r'\1\n\n\2', final_answer)
            
            # Step 2: Add \n\n between paragraphs (CRITICAL for markdown rendering!)
            # Strategy: Split by single \n, then add \n\n between paragraphs
            # This ensures proper markdown paragraph separation
            lines = final_answer.split('\n')
            processed_lines = []
            in_code_block = False
            in_list = False
            
            for i, line in enumerate(lines):
                current_line_stripped = line.strip()
                
                # Track code blocks
                if current_line_stripped.startswith('```'):
                    in_code_block = not in_code_block
                    processed_lines.append(line)
                    continue
                
                # Inside code block - preserve as-is
                if in_code_block:
                    processed_lines.append(line)
                    continue
                
                # Track lists
                is_list_item = bool(re.match(r'^[-*+]\s+', current_line_stripped) or 
                                   re.match(r'^\d+\.\s+', current_line_stripped))
                
                # If this is a list item and previous wasn't, we might need spacing
                if is_list_item and not in_list and processed_lines:
                    # Check if previous line needs spacing before list
                    prev_line = processed_lines[-1] if processed_lines else ''
                    if prev_line.strip() and not prev_line.strip().endswith(':'):
                        # Add spacing before list if previous line is not empty and doesn't end with colon
                        processed_lines.append('')
                
                in_list = is_list_item
                
                # Add current line
                processed_lines.append(line)
                
                # Check if we need to add spacing after this line
                if i < len(lines) - 1:  # Not last line
                    next_line = lines[i + 1]
                    next_line_stripped = next_line.strip()
                    
                    # Skip if next line is empty (already has spacing)
                    if not next_line_stripped:
                        continue
                    
                    # Skip if next line starts with markdown syntax that doesn't need spacing
                    if re.match(r'^#{1,6}\s+', next_line_stripped):
                        # Headings - spacing already handled in Step 1
                        continue
                    if re.match(r'^[-*+]\s+', next_line_stripped):
                        # List items - spacing handled above
                        continue
                    if re.match(r'^\d+\.\s+', next_line_stripped):
                        # Numbered lists - spacing handled above
                        continue
                    if next_line_stripped.startswith('```'):
                        # Code blocks - add spacing before
                        if current_line_stripped:
                            processed_lines.append('')
                        continue
                    
                    # CRITICAL: Add \n\n between paragraphs for proper markdown rendering
                    # If current line is a paragraph (ends with punctuation) and next is also a paragraph
                    # This is CRITICAL - markdown requires blank lines between paragraphs!
                    if (current_line_stripped and 
                        current_line_stripped[-1] in '.!?' and
                        next_line_stripped and
                        len(next_line_stripped) > 10 and  # Next line is substantial (not just a word)
                        not next_line_stripped[0].islower()):  # Next starts with capital (new sentence/paragraph)
                        # Check if we already have spacing (look ahead)
                        has_spacing = False
                        if i < len(lines) - 2:
                            # Check if next line is empty (already has spacing)
                            if not lines[i + 1].strip():
                                has_spacing = True
                        # Also check if current line already ends with double newline (look back)
                        if not has_spacing and len(processed_lines) > 0:
                            # Check if last added line was empty (spacing already present)
                            if processed_lines[-1].strip() == '':
                                has_spacing = True
                        
                        if not has_spacing:
                            # Add spacing between paragraphs - CRITICAL for markdown!
                            processed_lines.append('')
                            logger.debug("Added spacing between paragraphs", 
                                       current_line_preview=current_line_stripped[:50],
                                       next_line_preview=next_line_stripped[:50])
            
            final_answer = '\n'.join(processed_lines)
            
            # Step 3: Normalize multiple consecutive newlines (more than 2) to exactly 2
            # This prevents excessive spacing while preserving paragraph breaks
            final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)
            
            # Step 4: CRITICAL - Ensure proper spacing after all headings (like Deep Research does)
            # Deep Research uses "\n\n".join() between sections - we should do the same
            # Add \n\n after every heading if not already present
            final_answer = re.sub(r'(#{2,}\s+[^\n]+)\n([^\n#\s])', r'\1\n\n\2', final_answer)
            
            # Step 5: Final aggressive fix - if answer still has very few double newlines
            # This handles cases where LLM completely ignored formatting instructions
            double_newline_ratio = final_answer.count('\n\n') / max(final_answer.count('\n'), 1)
            if double_newline_ratio < 0.15 and len(final_answer) > 500:
                # Very few double newlines - LLM likely didn't format properly
                logger.warning("Answer has very few double newlines - applying aggressive formatting fix",
                             double_newline_ratio=round(double_newline_ratio, 3),
                             double_newline_count=final_answer.count('\n\n'),
                             total_newline_count=final_answer.count('\n'),
                             answer_length=len(final_answer))
                
                # CRITICAL: More aggressive approach - add \n\n after every sentence ending with punctuation
                # But preserve existing structure (headings, lists, code blocks)
                # Strategy: Find sentences ending with .!? followed by capital letter (new sentence)
                # and ensure they have \n\n between them
                lines = final_answer.split('\n')
                fixed_lines = []
                for i, line in enumerate(lines):
                    fixed_lines.append(line)
                    if i < len(lines) - 1:
                        next_line = lines[i + 1]
                        current_stripped = line.strip()
                        next_stripped = next_line.strip()
                        
                        # Skip if next line is empty or markdown syntax
                        if (not next_stripped or 
                            re.match(r'^#{1,6}\s+', next_stripped) or
                            re.match(r'^[-*+]\s+', next_stripped) or
                            re.match(r'^\d+\.\s+', next_stripped) or
                            next_stripped.startswith('```')):
                            continue
                        
                        # If current line ends with sentence punctuation and next starts with capital
                        if (current_stripped and 
                            current_stripped[-1] in '.!?' and
                            next_stripped and
                            len(next_stripped) > 5 and  # Substantial next line
                            next_stripped[0].isupper() and  # Starts with capital
                            not next_stripped[0].islower()):  # Not lowercase
                            # Check if we already added spacing
                            if len(fixed_lines) > 0 and fixed_lines[-1].strip() == '':
                                continue  # Already has spacing
                            # Add spacing
                            fixed_lines.append('')
                
                final_answer = '\n'.join(fixed_lines)
                # Normalize again
                final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)
            
            new_newline_count = final_answer.count('\n')
            new_double_newline_count = final_answer.count('\n\n')
            
            if original_double_newline_count != new_double_newline_count or original_newline_count != new_newline_count:
                logger.info("Post-processed answer to add missing newlines", 
                           original_newline_count=original_newline_count,
                           new_newline_count=new_newline_count,
                           original_double_newline_count=original_double_newline_count,
                           new_double_newline_count=new_double_newline_count,
                           double_newline_ratio_before=original_double_newline_count / max(original_newline_count, 1),
                           double_newline_ratio_after=new_double_newline_count / max(new_newline_count, 1),
                           note="Added \\n\\n between paragraphs for proper markdown rendering")
        
        # CRITICAL: Log formatting preservation
        newline_count = final_answer.count("\n") if final_answer else 0
        has_newlines = "\n" in (final_answer or "")
        logger.debug("Writer received answer from LLM", 
                    answer_length=len(final_answer) if final_answer else 0,
                    newline_count=newline_count,
                    has_newlines=has_newlines,
                    answer_preview=final_answer[:200] if final_answer else "")
        
        # CRITICAL: Check if answer contains markdown formatting
        has_markdown_headings = bool(re.search(r'^#{2,}\s+', final_answer, re.MULTILINE))
        has_markdown_bold = bool(re.search(r'\*\*.*?\*\*', final_answer))
        has_markdown_lists = bool(re.search(r'^[-*+]\s+', final_answer, re.MULTILINE))
        has_markdown = has_markdown_headings or has_markdown_bold or has_markdown_lists
        
        if not has_markdown:
            logger.warning(
                "Writer answer appears to lack markdown formatting - LLM ignored instructions!",
                answer_preview=final_answer[:500],
                has_headings=has_markdown_headings,
                has_bold=has_markdown_bold,
                has_lists=has_markdown_lists,
            )
            # CRITICAL: Force markdown formatting if LLM didn't use it
            # Wrap the answer in markdown structure
            # CRITICAL: Don't use .strip() - check if answer starts with ## without stripping
            answer_starts_with_heading = final_answer.lstrip().startswith('##') if final_answer else False
            if not answer_starts_with_heading:
                # Add main heading if missing
                # CRITICAL: Preserve any leading newlines in original answer
                leading_newlines = len(final_answer) - len(final_answer.lstrip('\n')) if final_answer else 0
                if leading_newlines > 0:
                    final_answer = final_answer.lstrip('\n')  # Remove only leading newlines, preserve rest
                final_answer = f"## {query}\n\n{final_answer}"
                logger.info("Added markdown heading to answer")
        else:
            logger.info(
                "Writer answer contains markdown formatting",
                has_headings=has_markdown_headings,
                has_bold=has_markdown_bold,
                has_lists=has_markdown_lists,
            )

        # CRITICAL: Ensure Sources section is present and formatted correctly with numbered sources
        # Format: [N] [Title](URL) where N matches the citation number in text [N]
        import re
        has_sources_section = "## Sources" in final_answer
        sources_section_pattern = r'(##\s+Sources.*?)(?=\n\n##|\Z)'
        
        # Extract citation numbers from text (e.g., [1], [2], [3])
        # Match [N] but not markdown links [text](url) or LaTeX formulas
        citation_pattern = r'(?<!\[)\[(\d+)\](?!\()'
        citations_found = re.findall(citation_pattern, final_answer)
        citation_numbers = sorted(set(int(c) for c in citations_found))
        
        # Create mapping: citation number -> source (by order of appearance in text)
        source_map = {}
        source_index = 1
        seen_urls = set()
        
        # CRITICAL: Always create source_map from all unique_sources, regardless of citations
        for source in unique_sources:
            title = source.get('title', 'Unknown')
            url = source.get('url', '')
            # Deduplicate by URL
            url_normalized = url.lower().rstrip('/') if url else ""
            if url_normalized and url_normalized not in seen_urls:
                seen_urls.add(url_normalized)
                # Always add to source_map with sequential numbering
                source_map[source_index] = {'title': title, 'url': url}
                source_index += 1
            elif not url_normalized:  # If no URL, still add it (might be a document or book)
                source_map[source_index] = {'title': title, 'url': url}
                source_index += 1
        
        # If citations were found, ensure all cited sources are in the map
        if citation_numbers:
            # Fill in missing sources for cited numbers
            for num in citation_numbers:
                if num not in source_map and num <= len(unique_sources):
                    # Try to find source by index
                    if num - 1 < len(unique_sources):
                        source = unique_sources[num - 1]
                        title = source.get('title', 'Unknown')
                        url = source.get('url', '')
                        source_map[num] = {'title': title, 'url': url}
        
        if not has_sources_section:
            # CRITICAL: Add Sources section if LLM didn't include it
            # ALWAYS add sources, even if LLM didn't include them!
            logger.info("LLM did not include Sources section - adding it automatically", 
                       sources_count=len(unique_sources),
                       source_map_size=len(source_map),
                       citation_numbers=citation_numbers)
            final_answer += "\n\n## Sources\n\n"
            # Use citation numbers from text, or sequential if none found
            numbers_to_use = citation_numbers if citation_numbers else sorted(source_map.keys())
            
            # CRITICAL: If source_map is empty but we have unique_sources, create it
            if not source_map and unique_sources:
                logger.warning("source_map is empty but unique_sources exist - creating mapping", sources_count=len(unique_sources))
                for idx, source in enumerate(unique_sources, 1):
                    source_map[idx] = {
                        'title': source.get('title', 'Unknown'),
                        'url': source.get('url', '')
                    }
                numbers_to_use = list(range(1, len(unique_sources) + 1))
            
            # CRITICAL: Always add ALL sources, not just cited ones!
            # Use all sources from source_map (which contains all unique sources)
            if source_map:
                # Use ALL sources from source_map, not just citation_numbers
                all_source_numbers = sorted(source_map.keys())
                logger.info("Adding ALL sources to Sources section", 
                           total_sources=len(all_source_numbers),
                           citation_numbers=citation_numbers,
                           note="Including all sources, not just cited ones")
                for num in all_source_numbers:
                    source_info = source_map[num]
                    if source_info.get('url'):
                        final_answer += f"[{num}] [{source_info['title']}]({source_info['url']})\n"
                    else:
                        final_answer += f"[{num}] {source_info['title']}\n"
            elif unique_sources:
                # Fallback: add all sources sequentially
                logger.warning("source_map empty - adding all sources sequentially", sources_count=len(unique_sources))
                for idx, source in enumerate(unique_sources, 1):
                    title = source.get('title', 'Unknown')
                    url = source.get('url', '')
                    if url:
                        final_answer += f"[{idx}] [{title}]({url})\n"
                    else:
                        final_answer += f"[{idx}] {title}\n"
            else:
                logger.error("No sources available to add to Sources section!", sources_count=len(unique_sources))
                final_answer += "*No sources available*\n"
        else:
            # LLM included Sources section - check if it has content and fix format if needed
            match = re.search(sources_section_pattern, final_answer, re.DOTALL | re.IGNORECASE)
            if match:
                sources_section = match.group(1)
                # Check if Sources section is empty or has no actual sources (only header)
                sources_content = sources_section.replace("## Sources", "").strip()
                has_sources_content = bool(re.search(r'\[.*?\]\(.*?\)|-\s+\[.*?\]', sources_content, re.IGNORECASE))
                
                if not has_sources_content:
                    # Sources section exists but is empty - replace it with proper numbered sources
                    # CRITICAL: Always include ALL sources, not just cited ones!
                    logger.info("Sources section is empty - adding ALL numbered sources", 
                               sources_count=len(unique_sources), 
                               citation_numbers=citation_numbers,
                               source_map_size=len(source_map))
                    
                    # CRITICAL: Use ALL sources from source_map, not just citation_numbers!
                    # source_map contains all unique sources with sequential numbering
                    all_source_numbers = sorted(source_map.keys()) if source_map else list(range(1, len(unique_sources) + 1))
                    
                    # If source_map is empty, create it from unique_sources
                    if not source_map and unique_sources:
                        logger.warning("source_map is empty in Sources section replacement - creating from unique_sources", sources_count=len(unique_sources))
                        for idx, source in enumerate(unique_sources, 1):
                            source_map[idx] = {
                                'title': source.get('title', 'Unknown'),
                                'url': source.get('url', '')
                            }
                        all_source_numbers = sorted(source_map.keys())
                    
                    sources_list = []
                    for num in all_source_numbers:
                        if num in source_map:
                            source_info = source_map[num]
                            if source_info.get('url'):
                                sources_list.append(f"[{num}] [{source_info['title']}]({source_info['url']})")
                            else:
                                sources_list.append(f"[{num}] {source_info['title']}")
                    
                    # CRITICAL: Use \n\n between sources for proper markdown formatting (same as deep research)
                    new_sources_section = f"## Sources\n\n" + "\n".join(sources_list) + "\n"
                    final_answer = final_answer.replace(sources_section, new_sources_section)
                    logger.info("Replaced empty Sources section with all sources", 
                               sources_added=len(sources_list),
                               total_sources=len(unique_sources))
                else:
                    # Sources section has content - verify it uses numbered format [N] [Title](URL)
                    # If it uses old format without numbers, add numbers based on citations
                    if not re.search(r'\[\d+\]\s+\[', sources_section):
                        # No numbered format found - add numbers based on citations
                        logger.info("Sources section missing numbers - adding based on citations", citation_numbers=citation_numbers)
                        # This is complex, so we'll leave it as-is if it has content
                        pass
            else:
                # Sources section header exists but regex didn't match - add numbered sources after it
                # CRITICAL: Always include ALL sources, not just cited ones!
                logger.info("Sources section header found but no content - adding ALL numbered sources", 
                           sources_count=len(unique_sources), 
                           citation_numbers=citation_numbers,
                           source_map_size=len(source_map))
                
                # CRITICAL: Use ALL sources from source_map, not just citation_numbers!
                all_source_numbers = sorted(source_map.keys()) if source_map else list(range(1, len(unique_sources) + 1))
                
                # If source_map is empty, create it from unique_sources
                if not source_map and unique_sources:
                    logger.warning("source_map is empty in Sources section addition - creating from unique_sources", sources_count=len(unique_sources))
                    for idx, source in enumerate(unique_sources, 1):
                        source_map[idx] = {
                            'title': source.get('title', 'Unknown'),
                            'url': source.get('url', '')
                        }
                    all_source_numbers = sorted(source_map.keys())
                
                sources_list = []
                for num in all_source_numbers:
                    if num in source_map:
                        source_info = source_map[num]
                        if source_info.get('url'):
                            sources_list.append(f"[{num}] [{source_info['title']}]({source_info['url']})")
                        else:
                            sources_list.append(f"[{num}] {source_info['title']}")
                
                # Find position of "## Sources" and add sources after it
                sources_header_pos = final_answer.rfind("## Sources")
                if sources_header_pos != -1:
                    # Find end of line after "## Sources"
                    end_of_line = final_answer.find("\n", sources_header_pos)
                    if end_of_line != -1:
                        # CRITICAL: Use \n\n between sources for proper markdown formatting
                        final_answer = final_answer[:end_of_line+1] + "\n" + "\n".join(sources_list) + "\n" + final_answer[end_of_line+1:]
                    else:
                        final_answer += "\n\n" + "\n".join(sources_list) + "\n"
                logger.info("Added all sources to Sources section", sources_added=len(sources_list), total_sources=len(unique_sources))

        # Add confidence indicator (optional, for debugging)
        # final_answer += f"\n\n*Confidence: {result.confidence}*"

        # CRITICAL: Log final answer formatting before returning
        final_newline_count = final_answer.count("\n") if final_answer else 0
        final_has_newlines = "\n" in (final_answer or "")
        logger.info(
            "Answer synthesized",
            length=len(final_answer),
            sources=len(unique_sources),
            confidence=result.confidence,
            newline_count=final_newline_count,
            has_newlines=final_has_newlines,
            has_markdown_headings=bool(re.search(r'^#{2,}\s+', final_answer, re.MULTILINE)),
            answer_preview=final_answer[:200] if final_answer else ""
        )

        # CRITICAL: Return answer as-is - preserve all formatting including \n
        return final_answer

    except Exception as e:
        logger.error("Writer agent failed", error=str(e), exc_info=True)

        # Fallback: Simple source listing with markdown formatting
        # CRITICAL: Use markdown format even in fallback!
        fallback = f"## {query}\n\n"
        fallback += f"Based on the research:\n\n{source_context}\n\n"
        fallback += "## Sources\n\n"
        
        if unique_sources:
            for idx, source in enumerate(unique_sources, 1):
                title = source.get('title', 'Unknown')
                url = source.get('url', '')
                if url:
                    fallback += f"[{idx}] [{title}]({url})\n"
                else:
                    fallback += f"[{idx}] {title}\n"
        else:
            fallback += "*No sources available*\n"

        return fallback
