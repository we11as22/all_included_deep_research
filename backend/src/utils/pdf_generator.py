"""PDF generation utility for research reports."""

import re
import os
from io import BytesIO
from typing import Any

import markdown
import structlog
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate, PageBreak, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = structlog.get_logger(__name__)

# Register Unicode-compatible fonts
# Try to use system fonts or fallback to built-in fonts
_UNICODE_FONT_REGISTERED = False

def _register_unicode_fonts():
    """Register Unicode-compatible fonts for PDF generation."""
    global _UNICODE_FONT_REGISTERED
    if _UNICODE_FONT_REGISTERED:
        return
    
    # Try to find and register DejaVu Sans (common Unicode font)
    font_paths = [
        # Linux common paths
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/TTF/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        # macOS common paths
        '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        # Windows common paths
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/arialuni.ttf',
    ]
    
    bold_font_paths = [
        # Linux common paths
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
        # macOS common paths
        '/Library/Fonts/Arial Unicode.ttf',  # Arial Unicode supports bold
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        # Windows common paths
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/arialuni.ttf',
    ]
    
    font_registered = False
    base_font_path = None
    
    # Find base font
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('UnicodeFont', font_path))
                base_font_path = font_path
                font_registered = True
                logger.info("Registered Unicode font", path=font_path)
                break
            except Exception as e:
                logger.warning("Failed to register font", path=font_path, error=str(e))
                continue
    
    # Find bold font
    if font_registered and base_font_path:
        bold_registered = False
        for bold_path in bold_font_paths:
            if os.path.exists(bold_path):
                try:
                    pdfmetrics.registerFont(TTFont('UnicodeFont-Bold', bold_path))
                    bold_registered = True
                    logger.info("Registered Unicode bold font", path=bold_path)
                    break
                except Exception as e:
                    logger.warning("Failed to register bold font", path=bold_path, error=str(e))
                    continue
        
        # If bold font not found, use base font for bold (will be rendered as regular)
        if not bold_registered:
            try:
                pdfmetrics.registerFont(TTFont('UnicodeFont-Bold', base_font_path))
                logger.info("Using base font for bold variant")
            except Exception:
                pass
    
    # If no system font found, try to use ReportLab's built-in Unicode support
    if not font_registered:
        try:
            # Use ReportLab's built-in Unicode font support via CID fonts
            # This should work for most Unicode characters including Cyrillic
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            # Try different CID fonts that support Unicode
            # CID fonts support many Unicode ranges including Cyrillic
            cid_fonts = ['SimSun', 'STSong-Light', 'HeiseiMin-W3', 'HeiseiKakuGo-W5']
            for cid_font in cid_fonts:
                try:
                    pdfmetrics.registerFont(UnicodeCIDFont(cid_font))
                    font_registered = True
                    logger.info("Using ReportLab built-in Unicode CID font", font=cid_font)
                    break
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Failed to register built-in Unicode font", error=str(e))
            # Last resort: will use Helvetica (may show squares for unsupported characters)
    
    _UNICODE_FONT_REGISTERED = True


def _extract_sources_from_report(report: str) -> dict[int, tuple[str, str]]:
    """
    Extract sources from report text.
    
    Looks for patterns like:
    [1] Title: URL
    or in Sources section
    
    Returns:
        Dictionary mapping citation number to (title, url) tuple
    """
    sources = {}
    
    from urllib.parse import unquote
    
    # Pattern 1: [1] Title: URL
    pattern1 = r'\[(\d+)\]\s+([^:]+):\s+(https?://[^\s\)]+)'
    matches = re.finditer(pattern1, report)
    for match in matches:
        num = int(match.group(1))
        title = match.group(2).strip()
        url = match.group(3).strip()
        
        # CRITICAL: Decode URL-encoded title and URL
        try:
            if '%' in title:
                title = unquote(title, encoding='utf-8')
            if '%' in url:
                url = unquote(url, encoding='utf-8')
        except Exception as e:
            logger.warning("Failed to decode source", title_preview=title[:50], url_preview=url[:50], error=str(e))
        
        sources[num] = (title, url)
    
    # Pattern 2: Sources sections with markdown links: - [Title](url)
    # CRITICAL: Find ALL Sources sections (including the last one in the last chapter)
    # Use findall to get all Sources sections, not just the first one
    sources_sections = re.finditer(r'##\s+Sources\s+(.*?)(?=##|$)', report, re.DOTALL | re.IGNORECASE)
    
    for sources_section in sources_sections:
        section_text = sources_section.group(1)
        # Match markdown links: - [Title](url)
        pattern2 = re.compile(r'-\s*\[([^\]]+)\]\(([^)]+)\)')
        for match in pattern2.finditer(section_text):
            num = len(sources) + 1
            title = match.group(1).strip()
            url = match.group(2).strip()
            
            # CRITICAL: Decode URL-encoded title and URL
            try:
                if '%' in title:
                    title = unquote(title, encoding='utf-8')
                if '%' in url:
                    url = unquote(url, encoding='utf-8')
            except Exception as e:
                logger.warning("Failed to decode source", title_preview=title[:50], url_preview=url[:50], error=str(e))
            
            sources[num] = (title, url)
        
        # Also match numbered format: [1] Title: URL or 1. Title: URL (fallback)
        # Only if no markdown links were found in this section
        # Use search() result directly, not any(), to avoid TypeError
        if not re.search(r'-\s*\[([^\]]+)\]\(([^)]+)\)', section_text):
            pattern2_fallback = r'(?:\[(\d+)\]|(\d+)\.)\s+([^:]+):\s+(https?://[^\s\)]+)'
            matches = re.finditer(pattern2_fallback, section_text)
            for match in matches:
                num = int(match.group(1) or match.group(2))
                title = match.group(3).strip()
                url = match.group(4).strip()
                
                # CRITICAL: Decode URL-encoded title and URL
                try:
                    if '%' in title:
                        title = unquote(title, encoding='utf-8')
                    if '%' in url:
                        url = unquote(url, encoding='utf-8')
                except Exception as e:
                    logger.warning("Failed to decode source", title_preview=title[:50], url_preview=url[:50], error=str(e))
                
                sources[num] = (title, url)
    
    logger.info("Extracted sources from report",
               total_sources=len(sources),
               sources_sections_found=len(list(re.finditer(r'##\s+Sources\s+', report, re.IGNORECASE))),
               note="All Sources sections (including last chapter) should be processed. Sources are used for clickable citations, and Sources sections are rendered as part of HTML content.")
    
    return sources


def _make_citations_clickable(text: str, sources: dict[int, tuple[str, str]]) -> str:
    """
    Convert citation markers [1], [2] to clickable links in markdown.
    
    Args:
        text: Report text with citations like [1], [2]
        sources: Dictionary mapping citation number to (title, url) tuple
    
    Returns:
        Text with citations converted to markdown links
    """
    def replace_citation(match):
        num = int(match.group(1))
        if num in sources:
            title, url = sources[num]
            # Create clickable link
            return f'<a href="{url}" color="blue">[{num}]</a>'
        return match.group(0)
    
    # Replace [1], [2], etc. with clickable links
    pattern = r'\[(\d+)\]'
    result = re.sub(pattern, replace_citation, text)
    
    return result


def markdown_to_pdf(report: str, title: str = "Research Report") -> BytesIO:
    """
    Convert markdown report to PDF with clickable links.
    
    Args:
        report: Markdown formatted report text
        title: PDF document title
    
    Returns:
        BytesIO buffer containing PDF data
    """
    # Register Unicode fonts first
    _register_unicode_fonts()
    
    buffer = BytesIO()
    
    # Extract sources for citation linking
    sources = _extract_sources_from_report(report)
    
    # Make citations clickable
    report_with_links = _make_citations_clickable(report, sources)
    
    # Convert markdown to HTML
    # CRITICAL: Use 'extra' extension which includes link processing
    # This ensures all markdown links [text](url) are converted to <a> tags
    html = markdown.markdown(
        report_with_links,
        extensions=['extra', 'nl2br', 'sane_lists', 'tables', 'fenced_code'],
    )
    
    # CRITICAL: Log Sources sections in markdown before conversion
    sources_sections_md = list(re.finditer(r'##\s+Sources\s+', report, re.IGNORECASE))
    logger.info("PDF: Found Sources sections in markdown",
               sources_sections_count=len(sources_sections_md),
               note="All Sources sections (including last chapter) should be converted to HTML and rendered")
    
    # Parse HTML and extract text with links
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # CRITICAL: Verify Sources sections are in HTML after conversion
    sources_h2_in_html = soup.find_all('h2', string=re.compile(r'^Sources$', re.IGNORECASE))
    logger.info("PDF: Sources sections in HTML after markdown conversion",
               sources_sections_count=len(sources_h2_in_html),
               expected_count=len(sources_sections_md),
               note="All Sources sections from markdown should be present in HTML. If count differs, some Sources sections may be missing.")
    
    # Create PDF document with UTF-8 encoding support
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    
    # Ensure UTF-8 encoding for all text processing
    import sys
    if sys.version_info < (3, 7):
        # Python < 3.7: ensure UTF-8 encoding
        import codecs
        report = report.encode('utf-8').decode('utf-8')
    else:
        # Python >= 3.7: UTF-8 is default
        pass
    
    # Determine which font to use
    # Try to use registered Unicode font, fallback to Helvetica
    registered_fonts = pdfmetrics.getRegisteredFontNames()
    
    # Check for TTF Unicode fonts first
    if 'UnicodeFont' in registered_fonts:
        unicode_font_name = 'UnicodeFont'
        unicode_bold_font_name = 'UnicodeFont-Bold' if 'UnicodeFont-Bold' in registered_fonts else 'UnicodeFont'
    # Check for CID fonts (they register with their own names)
    elif any(font in registered_fonts for font in ['SimSun', 'STSong-Light', 'HeiseiMin-W3', 'HeiseiKakuGo-W5']):
        # Use the first CID font found
        cid_font = next((f for f in ['SimSun', 'STSong-Light', 'HeiseiMin-W3', 'HeiseiKakuGo-W5'] if f in registered_fonts), 'SimSun')
        unicode_font_name = cid_font
        unicode_bold_font_name = cid_font  # CID fonts don't have separate bold variants
    else:
        # Fallback to Helvetica (may show squares for unsupported characters)
        unicode_font_name = 'Helvetica'
        unicode_bold_font_name = 'Helvetica-Bold'
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles with Unicode font support
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=unicode_font_name,
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=1,  # Center
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontName=unicode_bold_font_name,
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontName=unicode_bold_font_name,
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=10,
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontName=unicode_font_name,
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        leading=14,
    )
    
    link_style = ParagraphStyle(
        'LinkStyle',
        parent=body_style,
        fontName=unicode_font_name,
        textColor=colors.HexColor('#0066cc'),
        underline=True,
    )
    
    # Build PDF content
    story = []
    
    # CRITICAL: Check if first h1 in HTML matches the title to avoid duplication
    # Extract first h1 from HTML
    first_h1 = None
    for element in soup.children:
        if hasattr(element, 'name') and element.name == 'h1':
            first_h1 = element.get_text().strip()
            break
    
    # Only add title if it doesn't match first h1 (to avoid duplication)
    if first_h1 and first_h1.strip().lower() == title.strip().lower():
        logger.info("PDF: Skipping duplicate title - first h1 matches title",
                   title=title[:50],
                   first_h1=first_h1[:50],
                   note="Title will be added once from h1 element, not duplicated")
    else:
        # Title doesn't match first h1 or no h1 found - add title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3 * inch))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 0.3 * inch))
    
    # Process HTML elements
    def process_element(element):
        """Recursively process HTML elements."""
        if element.name is None:  # Text node
            text = str(element).strip()
            # Ensure UTF-8 encoding for text
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            if text:
                # Check for links in text - if parent is <a>, create clickable link
                if element.parent and element.parent.name == 'a':
                    href = element.parent.get('href', '')
                    link_text = text
                    # Format as clickable link using ReportLab's link format
                    story.append(
                        Paragraph(
                            f'<link href="{href}" color="blue"><u>{link_text}</u></link>',
                            link_style,
                        )
                    )
                else:
                    # Regular text - but check if it contains any links that weren't processed
                    # This shouldn't happen if markdown conversion worked correctly
                    story.append(Paragraph(text, body_style))
            return
        
        tag = element.name
        
        if tag == 'h1':
            text = element.get_text().strip()
            # Ensure UTF-8 encoding
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            if text:
                story.append(Paragraph(text, heading1_style))
                story.append(Spacer(1, 0.1 * inch))
        elif tag == 'h2':
            text = element.get_text().strip()
            # Ensure UTF-8 encoding
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            if text:
                # CRITICAL: Log when processing Sources section to verify it's rendered
                if text.strip().lower() == 'sources':
                    logger.info("PDF: Processing Sources section",
                               section_title=text,
                               note="Sources section found and will be rendered in PDF")
                story.append(Paragraph(text, heading2_style))
                story.append(Spacer(1, 0.08 * inch))
        elif tag == 'h3':
            text = element.get_text().strip()
            # Ensure UTF-8 encoding
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            if text:
                story.append(Paragraph(f'<b>{text}</b>', body_style))
                story.append(Spacer(1, 0.06 * inch))
        elif tag == 'p':
            # Process paragraph with all its children, preserving links
            para_html = str(element)
            # Ensure UTF-8 for HTML
            if isinstance(para_html, bytes):
                para_html = para_html.decode('utf-8', errors='replace')
            
            # CRITICAL: Replace ALL <a> tags with ReportLab clickable link format
            # Handle both simple links and links with attributes
            # Pattern 1: <a href="url">text</a>
            para_html = re.sub(
                r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
                r'<link href="\1" color="blue"><u>\2</u></link>',
                para_html,
                flags=re.IGNORECASE | re.DOTALL
            )
            # Pattern 2: <a href="url" color="blue">[1]</a> (citations)
            para_html = re.sub(
                r'<a\s+href="([^"]+)"[^>]*color="blue"[^>]*>\[(\d+)\]</a>',
                r'<link href="\1" color="blue"><u>[\2]</u></link>',
                para_html,
                flags=re.IGNORECASE
            )
            # Pattern 3: Nested links or complex content
            para_html = re.sub(
                r'<a\s+href="([^"]+)"[^>]*>([^<]*(?:<[^>]+>[^<]*)*)</a>',
                lambda m: f'<link href="{m.group(1)}" color="blue"><u>{re.sub(r"<[^>]+>", "", m.group(2))}</u></link>',
                para_html,
                flags=re.IGNORECASE | re.DOTALL
            )
            
            # Remove HTML tags except ReportLab tags (<link>, <u>, <b>, <i>, etc.)
            # But keep the link tags we just created
            # ReportLab supports: <link>, <u>, <b>, <i>, <font>, etc.
            
            text = element.get_text().strip()
            if text or para_html.strip():
                story.append(Paragraph(para_html, body_style))
                story.append(Spacer(1, 0.06 * inch))
        elif tag == 'ul' or tag == 'ol':
            # CRITICAL: Process all list items, including those in Sources sections
            # Use find_all with recursive=False to get direct children only
            list_items = element.find_all('li', recursive=False)
            
            # If no direct children, try recursive search (for nested lists)
            if not list_items:
                list_items = element.find_all('li', recursive=True)
            
            for li in list_items:
                # Process list item with all its children, preserving links
                li_html = str(li)
                # Ensure UTF-8 for HTML
                if isinstance(li_html, bytes):
                    li_html = li_html.decode('utf-8', errors='replace')
                
                # CRITICAL: Replace ALL <a> tags with ReportLab clickable link format
                # Handle both simple links and links with attributes
                # Pattern 1: Simple links <a href="url">text</a>
                li_html = re.sub(
                    r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
                    r'<link href="\1" color="blue"><u>\2</u></link>',
                    li_html,
                    flags=re.IGNORECASE | re.DOTALL
                )
                # Pattern 2: Links with nested content
                li_html = re.sub(
                    r'<a\s+href="([^"]+)"[^>]*>([^<]*(?:<[^>]+>[^<]*)*)</a>',
                    lambda m: f'<link href="{m.group(1)}" color="blue"><u>{re.sub(r"<[^>]+>", "", m.group(2))}</u></link>',
                    li_html,
                    flags=re.IGNORECASE | re.DOTALL
                )
                
                # Remove <li> tags, keep content
                li_html = re.sub(r'</?li[^>]*>', '', li_html)
                
                # Also remove <ul> and <ol> tags if nested
                li_html = re.sub(r'</?[uo]l[^>]*>', '', li_html)
                
                text = li.get_text().strip()
                if text or li_html.strip():
                    # Check if this is in a Sources section (parent h2 contains "Sources")
                    # This helps with debugging
                    is_sources_section = False
                    parent = element.parent
                    while parent:
                        if hasattr(parent, 'name') and parent.name == 'h2':
                            if parent.get_text().strip().lower() == 'sources':
                                is_sources_section = True
                                break
                        parent = getattr(parent, 'parent', None)
                    
                    story.append(Paragraph(f'• {li_html}', body_style))
                    story.append(Spacer(1, 0.04 * inch))
            story.append(Spacer(1, 0.1 * inch))
        elif tag == 'li':
            # Handled in ul/ol
            pass
        elif tag == 'a':
            # CRITICAL: Handle standalone links (not inside paragraphs/lists)
            # This ensures all links are clickable, even if they're standalone elements
            href = element.get('href', '')
            text = element.get_text().strip()
            if text and href:
                # Format as clickable link using ReportLab's link format
                story.append(
                    Paragraph(
                        f'<link href="{href}" color="blue"><u>{text}</u></link>',
                        link_style,
                    )
                )
                story.append(Spacer(1, 0.04 * inch))
        elif tag == 'strong' or tag == 'b':
            text = element.get_text().strip()
            if text:
                story.append(Paragraph(f'<b>{text}</b>', body_style))
        elif tag == 'em' or tag == 'i':
            text = element.get_text().strip()
            if text:
                story.append(Paragraph(f'<i>{text}</i>', body_style))
        elif tag == 'hr':
            story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.HexColor('#cccccc')))
            story.append(Spacer(1, 0.2 * inch))
        elif tag == 'br':
            story.append(Spacer(1, 0.1 * inch))
        else:
            # Process children
            for child in element.children:
                if hasattr(child, 'name'):
                    process_element(child)
    
    # Process main content
    # CRITICAL: process_element processes elements recursively, so all nested elements (including Sources sections)
    # should be processed. However, we need to ensure we process ALL top-level elements, including those
    # that might be at the end of the document (like Sources sections in the last chapter).
    
    # CRITICAL: Process all top-level elements, including those at the end (like Sources in last chapter)
    # Use soup.find_all() to ensure we get all elements, not just direct children
    # But we want to process in order, so we'll iterate through soup.children first
    # and also check for any elements that might be missed
    
    # Count Sources sections in HTML to verify they're all processed
    sources_h2_elements = soup.find_all('h2', string=re.compile(r'^Sources$', re.IGNORECASE))
    logger.info("PDF: Found Sources sections in HTML",
               sources_sections_count=len(sources_h2_elements),
               note="All Sources sections (including last chapter) should be processed and rendered in PDF")
    
    # Process all top-level elements
    processed_elements = 0
    for element in soup.children:
        if hasattr(element, 'name'):
            processed_elements += 1
            process_element(element)
    
    # CRITICAL: Also process any elements that might be in body tag but not in direct children
    # This ensures we don't miss any content, especially at the end of the document
    body = soup.find('body')
    if body:
        # If body exists, process its children (in case soup.children didn't catch everything)
        for element in body.children:
            if hasattr(element, 'name') and element.name:
                # Only process if not already processed (avoid duplicates)
                # We can't easily track this, but since we process soup.children first,
                # this should only catch elements that were missed
                process_element(element)
    
    logger.info("PDF: Processed HTML elements",
               processed_count=processed_elements,
               sources_sections_found=len(sources_h2_elements),
               note="All elements including Sources sections should be rendered in PDF")
    
    # CRITICAL: Do NOT add Sources section at the end of PDF
    # Sources are already included in each chapter of draft_report (added automatically by supervisor)
    # Adding Sources section here would duplicate sources that are already in chapters
    # Sources are processed as part of the main content (each chapter has its own Sources section)
    # 
    # NOTE: The `sources` dict extracted by `_extract_sources_from_report` is only used
    # for making citations [1], [2] clickable in the text, NOT for adding a Sources section
    # 
    # Removed code that was adding Sources section at the end:
    # - Checking for "## Sources" section in HTML and processing it separately
    # - Adding extracted sources as a table at the end
    # 
    # Sources in chapters are already processed as part of the main content flow above
    
    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        logger.info("PDF generated successfully", title=title, size=len(buffer.getvalue()))
        return buffer
    except Exception as e:
        logger.error("PDF generation failed", error=str(e))
        raise

