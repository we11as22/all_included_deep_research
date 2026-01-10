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
    
    # Pattern 1: [1] Title: URL
    pattern1 = r'\[(\d+)\]\s+([^:]+):\s+(https?://[^\s\)]+)'
    matches = re.finditer(pattern1, report)
    for match in matches:
        num = int(match.group(1))
        title = match.group(2).strip()
        url = match.group(3).strip()
        sources[num] = (title, url)
    
    # Pattern 2: Sources section with numbered list
    sources_section = re.search(r'##\s+Sources\s+(.*?)(?=##|$)', report, re.DOTALL | re.IGNORECASE)
    if sources_section:
        section_text = sources_section.group(1)
        # Match [1] Title: URL or 1. Title: URL
        pattern2 = r'(?:\[(\d+)\]|(\d+)\.)\s+([^:]+):\s+(https?://[^\s\)]+)'
        matches = re.finditer(pattern2, section_text)
        for match in matches:
            num = int(match.group(1) or match.group(2))
            title = match.group(3).strip()
            url = match.group(4).strip()
            sources[num] = (title, url)
    
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
    html = markdown.markdown(
        report_with_links,
        extensions=['extra', 'nl2br', 'sane_lists'],
    )
    
    # Parse HTML and extract text with links
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
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
    
    # Title
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
                # Check for links in text
                if element.parent and element.parent.name == 'a':
                    href = element.parent.get('href', '')
                    link_text = text
                    # Format as clickable link
                    story.append(
                        Paragraph(
                            f'<link href="{href}" color="blue"><u>{link_text}</u></link>',
                            link_style,
                        )
                    )
                else:
                    # Regular text
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
            text = element.get_text().strip()
            # Ensure UTF-8 encoding
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            if text:
                # Check for links in paragraph
                para_html = str(element)
                # Ensure UTF-8 for HTML
                if isinstance(para_html, bytes):
                    para_html = para_html.decode('utf-8', errors='replace')
                # Replace <a> tags with reportlab link format
                para_html = re.sub(
                    r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
                    r'<link href="\1" color="blue"><u>\2</u></link>',
                    para_html,
                )
                # Also handle citations [1] that are now links
                para_html = re.sub(
                    r'<a\s+href="([^"]+)"[^>]*color="blue">\[(\d+)\]</a>',
                    r'<link href="\1" color="blue"><u>[\2]</u></link>',
                    para_html,
                )
                story.append(Paragraph(para_html, body_style))
                story.append(Spacer(1, 0.06 * inch))
        elif tag == 'ul' or tag == 'ol':
            for li in element.find_all('li', recursive=False):
                text = li.get_text().strip()
                # Ensure UTF-8 encoding
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='replace')
                if text:
                    # Check for links in list item
                    li_html = str(li)
                    # Ensure UTF-8 for HTML
                    if isinstance(li_html, bytes):
                        li_html = li_html.decode('utf-8', errors='replace')
                    li_html = re.sub(
                        r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
                        r'<link href="\1" color="blue"><u>\2</u></link>',
                        li_html,
                    )
                    # Remove <li> tags, keep content
                    li_html = re.sub(r'</?li[^>]*>', '', li_html)
                    story.append(Paragraph(f'• {li_html}', body_style))
                    story.append(Spacer(1, 0.04 * inch))
            story.append(Spacer(1, 0.1 * inch))
        elif tag == 'li':
            # Handled in ul/ol
            pass
        elif tag == 'a':
            # Links handled in parent elements
            href = element.get('href', '')
            text = element.get_text().strip()
            if text and href:
                story.append(
                    Paragraph(
                        f'<link href="{href}" color="blue"><u>{text}</u></link>',
                        link_style,
                    )
                )
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
    for element in soup.children:
        if hasattr(element, 'name'):
            process_element(element)
    
    # Add sources section at the end if available
    if sources:
        story.append(PageBreak())
        story.append(Paragraph("Sources", heading1_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Create table for sources
        source_data = [['#', 'Title', 'URL']]
        for num in sorted(sources.keys()):
            title, url = sources[num]
            source_data.append([str(num), title[:50], url[:60]])
        
        source_table = Table(source_data, colWidths=[0.5 * inch, 3 * inch, 3.5 * inch])
        source_table.setStyle(
            TableStyle(
                [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#333333')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), unicode_bold_font_name),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
                    ('FONTNAME', (0, 1), (-1, -1), unicode_font_name),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ]
            )
        )
        story.append(source_table)
    
    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        logger.info("PDF generated successfully", title=title, size=len(buffer.getvalue()))
        return buffer
    except Exception as e:
        logger.error("PDF generation failed", error=str(e))
        raise

