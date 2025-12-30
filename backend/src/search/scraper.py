"""Advanced web scraping utilities."""

import re
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import structlog
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.search.models import ScrapedContent

logger = structlog.get_logger(__name__)


class WebScraper:
    """Advanced web scraper with content extraction and cleaning."""

    def __init__(self, timeout: int = 30, user_agent: str | None = None):
        """
        Initialize web scraper.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; DeepResearchBot/1.0; +https://github.com)"
        )
        self.headers = {"User-Agent": self.user_agent}

    async def scrape(self, url: str, extract_markdown: bool = True) -> ScrapedContent:
        """
        Scrape and extract content from URL.

        Args:
            url: URL to scrape
            extract_markdown: Convert content to markdown

        Returns:
            ScrapedContent with extracted data
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    # Check if it's a PDF file
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                        return await self._scrape_pdf(url, response)
                    
                    # Try to read as text, but handle encoding errors
                    try:
                        html = await response.text()
                    except UnicodeDecodeError as e:
                        # If text decoding fails, log and return empty content
                        logger.warning("Failed to decode response as text", url=url, error=str(e))
                        return ScrapedContent(
                            url=url,
                            title="Unable to decode content",
                            content="",
                            markdown=None,
                            html=None,
                            images=[],
                            links=[],
                        )

            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title = self._extract_title(soup)

            # Extract main content
            content = self._extract_main_content(soup)

            # Extract images
            images = self._extract_images(soup, url)

            # Extract links
            links = self._extract_links(soup, url)

            # Convert to markdown if requested
            markdown = None
            if extract_markdown:
                markdown = self._html_to_markdown(content)

            # Get clean text content
            text_content = soup.get_text(separator=" ", strip=True)
            text_content = self._clean_text(text_content)

            scraped = ScrapedContent(
                url=url,
                title=title,
                content=text_content,
                markdown=markdown,
                html=html,
                images=images,
                links=links,
            )

            logger.info(
                "Web scraping completed",
                url=url,
                content_length=len(text_content),
                images_count=len(images),
                links_count=len(links),
            )

            return scraped

        except aiohttp.ClientError as e:
            logger.error("Web scraping failed - connection error", error=str(e), url=url)
            raise
        except Exception as e:
            logger.error("Web scraping failed", error=str(e), url=url)
            raise

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try multiple methods
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        # Try meta og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return "No title"

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page."""
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            element.decompose()

        # Try to find main content container
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile("content|article|post|entry", re.I))
            or soup.find("body")
        )

        if main_content:
            return str(main_content)

        return str(soup)

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract image URLs."""
        images = []

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, src)
                images.append(absolute_url)

        return images[:20]  # Limit to 20 images

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract outbound links."""
        links = []
        base_domain = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute_url = urljoin(base_url, href)

            # Only include http/https links
            parsed = urlparse(absolute_url)
            if parsed.scheme in ["http", "https"]:
                # Filter out links to same domain (optional)
                if parsed.netloc != base_domain:
                    links.append(absolute_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links[:50]  # Limit to 50 links

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        try:
            markdown = md(
                html,
                heading_style="ATX",  # Use # for headings
                bullets="-",  # Use - for bullets
                strip=["script", "style"],
            )
            return markdown.strip()
        except Exception as e:
            logger.warning("Markdown conversion failed", error=str(e))
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        return text.strip()

    async def _scrape_pdf(self, url: str, response: aiohttp.ClientResponse) -> ScrapedContent:
        """
        Scrape PDF file content.
        
        Args:
            url: PDF URL
            response: HTTP response
            
        Returns:
            ScrapedContent with PDF text
        """
        try:
            # Read PDF as bytes
            pdf_bytes = await response.read()
            
            # Try to extract text from PDF using PyPDF2 or pypdf
            try:
                import io
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    try:
                        from pypdf import PdfReader
                    except ImportError:
                        logger.warning("PDF libraries not available, skipping PDF extraction", url=url)
                        return ScrapedContent(
                            url=url,
                            title=url.split("/")[-1],
                            content="[PDF file - text extraction not available]",
                            markdown=None,
                            html=None,
                            images=[],
                            links=[],
                        )
                
                pdf_file = io.BytesIO(pdf_bytes)
                reader = PdfReader(pdf_file)
                
                # Extract text from all pages
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                
                # Extract title from first page or filename
                title = url.split("/")[-1].replace(".pdf", "")
                if reader.metadata and reader.metadata.title:
                    title = reader.metadata.title
                
                logger.info("PDF scraping completed", url=url, pages=len(reader.pages), content_length=len(text_content))
                
                return ScrapedContent(
                    url=url,
                    title=title,
                    content=text_content.strip(),
                    markdown=None,
                    html=None,
                    images=[],
                    links=[],
                )
            except Exception as e:
                logger.warning("PDF text extraction failed", url=url, error=str(e))
                return ScrapedContent(
                    url=url,
                    title=url.split("/")[-1],
                    content="[PDF file - text extraction failed]",
                    markdown=None,
                    html=None,
                    images=[],
                    links=[],
                )
        except Exception as e:
            logger.error("PDF scraping failed", url=url, error=str(e))
            raise


class ChunkedScraper(WebScraper):
    """Web scraper with content chunking for large pages."""

    def __init__(self, timeout: int = 30, user_agent: str | None = None, chunk_size: int = 2000):
        """
        Initialize chunked scraper.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
            chunk_size: Maximum characters per chunk
        """
        super().__init__(timeout, user_agent)
        self.chunk_size = chunk_size

    def chunk_content(self, content: str) -> list[str]:
        """
        Split content into chunks.

        Args:
            content: Text content to chunk

        Returns:
            List of content chunks
        """
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    # Paragraph itself is too long, split by sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
