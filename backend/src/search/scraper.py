"""Advanced web scraping utilities."""

import asyncio
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import structlog
from aiohttp import TCPConnector
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.search.models import ScrapedContent

logger = structlog.get_logger(__name__)

# Pool of realistic User-Agent strings for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class WebScraper:
    """Advanced web scraper with content extraction and cleaning."""

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str | None = None,
        use_playwright: bool = False,
        scroll_enabled: bool = False,
        scroll_pause: float = 1.0,
        max_scrolls: int = 5,
        rate_limit_per_domain: float = 2.0,  # requests per second per domain
        max_concurrent: int = 10,  # max concurrent requests
    ):
        """
        Initialize web scraper.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
            use_playwright: Use Playwright for JavaScript rendering (default: False, auto-fallback on 403)
            scroll_enabled: Enable automatic scrolling to load dynamic content
            scroll_pause: Pause between scrolls in seconds
            max_scrolls: Maximum number of scroll operations
            rate_limit_per_domain: Requests per second per domain (default: 2.0)
            max_concurrent: Maximum concurrent requests (default: 10)
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        # Use realistic browser User-Agent to avoid 403 Forbidden errors
        # Rotate User-Agent for better anti-detection
        self.user_agent = user_agent or random.choice(USER_AGENTS)
        # Use realistic browser headers to avoid bot detection
        # CRITICAL: Keep "br" in Accept-Encoding - brotli is in dependencies
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",  # Keep br - brotli is installed
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "Referer": "https://www.google.com/",  # Add referer to appear more legitimate
        }
        self.use_playwright = use_playwright
        self.scroll_enabled = scroll_enabled
        self.scroll_pause = scroll_pause
        self.max_scrolls = max_scrolls
        self.rate_limit_per_domain = rate_limit_per_domain
        self.max_concurrent = max_concurrent
        
        # CRITICAL: Single session for all requests (reuses connections, cookies, etc.)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[TCPConnector] = None
        
        # Rate limiting: track last request time per domain
        self._domain_last_request: dict[str, datetime] = defaultdict(lambda: datetime.min)
        
        # Semaphore for concurrent request limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Check if Playwright is available (for auto-fallback on 403)
        self._playwright_available = None

    async def _ensure_session(self):
        """Ensure aiohttp session is created (lazy initialization)."""
        if self._session is None or self._session.closed:
            # Create connector with connection pooling
            self._connector = TCPConnector(
                limit=100,  # Max connections
                limit_per_host=10,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                force_close=False,  # Reuse connections
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers,
                connector=self._connector,
                cookie_jar=aiohttp.CookieJar(),  # Enable cookie storage
            )
            logger.debug("Created new aiohttp session", connector_limit=self._connector.limit)

    async def _check_playwright_available(self) -> bool:
        """Check if Playwright is available (cached check)."""
        if self._playwright_available is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright_available = True
            except ImportError:
                self._playwright_available = False
        return self._playwright_available

    async def _rate_limit(self, url: str):
        """Apply rate limiting per domain."""
        domain = urlparse(url).netloc
        now = datetime.now()
        last_request = self._domain_last_request[domain]
        
        # Calculate time since last request
        time_since_last = (now - last_request).total_seconds()
        min_interval = 1.0 / self.rate_limit_per_domain
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            # Add small random jitter (0-200ms) to avoid synchronized requests
            jitter = random.uniform(0, 0.2)
            wait_time += jitter
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for domain {domain}", url=url)
            await asyncio.sleep(wait_time)
        
        self._domain_last_request[domain] = datetime.now()

    async def scrape(
        self, url: str, extract_markdown: bool = True, scroll: bool | None = None, max_retries: int = 2
    ) -> ScrapedContent:
        """
        Scrape and extract content from URL with retry logic.

        Args:
            url: URL to scrape
            extract_markdown: Convert content to markdown
            scroll: Override scroll_enabled setting for this scrape
            max_retries: Maximum number of retry attempts (with exponential backoff + Retry-After)

        Returns:
            ScrapedContent with extracted data
        """
        should_scroll = scroll if scroll is not None else self.scroll_enabled
        
        # Apply rate limiting
        await self._rate_limit(url)
        
        # Apply semaphore for concurrent request limiting
        async with self._semaphore:
            last_error = None
            playwright_tried = False  # Track if Playwright was already tried (to prevent infinite loop)
            for attempt in range(max_retries + 1):
                try:
                    # Rotate User-Agent on retry for better anti-detection
                    if attempt > 0:
                        self.user_agent = random.choice(USER_AGENTS)
                        self.headers["User-Agent"] = self.user_agent
                        # Update session headers if session exists
                        if self._session:
                            self._session.headers.update({"User-Agent": self.user_agent})

                    # Use Playwright if enabled or if scrolling is requested
                    if self.use_playwright or should_scroll:
                        playwright_tried = True
                        return await self._scrape_with_playwright(url, extract_markdown, should_scroll)

                    # Fallback to standard HTTP scraping
                    return await self._scrape_with_http(url, extract_markdown, playwright_tried=playwright_tried)
                    
                except aiohttp.ClientResponseError as e:
                    # CRITICAL: Extended retry logic for 429, 5xx errors
                    retryable_statuses = {429, 500, 502, 503, 504}
                    is_retryable = e.status in retryable_statuses
                    
                    last_error = e
                    if is_retryable and attempt < max_retries:
                        # Check for Retry-After header
                        retry_after = e.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                # If Retry-After is a date, calculate difference
                                try:
                                    retry_date = datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S %Z")
                                    wait_time = max(0, (retry_date - datetime.now()).total_seconds())
                                except:
                                    wait_time = 2 ** attempt  # Fallback to exponential
                        else:
                            # Exponential backoff with jitter
                            wait_time = 2 ** attempt + random.uniform(0, 1)
                        
                        logger.warning(
                            f"Retryable error {e.status}, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})",
                            url=url,
                            status=e.status,
                            retry_after=retry_after
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    elif e.status == 403:
                        # CRITICAL: Auto-fallback to Playwright on 403 even if use_playwright=False
                        # But only if Playwright wasn't already tried (to prevent infinite loop)
                        if not playwright_tried and await self._check_playwright_available():
                            playwright_tried = True
                            logger.info("403 Forbidden - auto-fallback to Playwright", url=url)
                            try:
                                return await self._scrape_with_playwright(url, extract_markdown, should_scroll)
                            except Exception as playwright_error:
                                logger.warning("Playwright fallback failed", url=url, error=str(playwright_error))
                                # Continue to retry logic or raise
                        # If Playwright not available, already tried, or failed, raise
                        logger.error("403 Forbidden and Playwright not available/already tried/failed", url=url, playwright_tried=playwright_tried)
                        raise
                    else:
                        # Non-retryable error or last attempt
                        logger.error(
                            f"HTTP error {e.status} (non-retryable or max retries reached)",
                            url=url,
                            status=e.status,
                            attempt=attempt + 1
                        )
                        raise
                        
                except asyncio.TimeoutError as e:
                    last_error = e
                    if attempt < max_retries:
                        # Exponential backoff for timeouts
                        wait_time = 2 ** attempt + random.uniform(0, 1)
                        logger.warning(
                            f"Timeout, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})",
                            url=url
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error("All retry attempts failed (timeout)", url=url)
                        raise
                except Exception as e:
                    # For other errors, don't retry
                    logger.error("Scraping failed with non-retryable error", url=url, error_type=type(e).__name__, error=str(e))
                    raise
            
            # Should never reach here, but just in case
            if last_error:
                raise last_error

    async def _scrape_with_playwright(
        self, url: str, extract_markdown: bool = True, scroll: bool = False
    ) -> ScrapedContent:
        """Scrape using Playwright for JavaScript rendering and scrolling."""
        # CRITICAL: Check if URL is a PDF file BEFORE trying Playwright
        # PDF files cannot be loaded with Playwright (they trigger download, not page load)
        if url.lower().endswith(".pdf") or "/.pdf" in url.lower():
            logger.info("PDF file detected, skipping Playwright and using HTTP scraping", url=url)
            return await self._scrape_with_http(url, extract_markdown, playwright_tried=True)
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed, falling back to HTTP scraping", url=url)
            return await self._scrape_with_http(url, extract_markdown, playwright_tried=True)

        try:
            async with async_playwright() as p:
                # Use chromium headless shell for better performance in Docker
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-extensions',
                    ]
                )
                context = await browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1920, "height": 1080},
                    # Add realistic browser headers to avoid bot detection
                    extra_http_headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                    }
                )
                # CRITICAL: Set navigation timeout AFTER context creation (navigation_timeout is not a parameter of new_context)
                context.set_default_navigation_timeout(60000)  # 60 seconds in milliseconds
                page = await context.new_page()

                logger.info("Loading page with Playwright", url=url)
                # CRITICAL: Use domcontentloaded instead of networkidle (networkidle is discouraged)
                await page.goto(url, wait_until="domcontentloaded", timeout=int(self.timeout.total * 1000))

                # CRITICAL: Wait for main content selectors instead of networkidle
                # This is more reliable and doesn't hang on pages with constant background requests
                try:
                    # Try to wait for common content selectors
                    selectors = ["main", "article", "h1", "[role='main']", ".content", "#content"]
                    for selector in selectors:
                        try:
                            await page.wait_for_selector(selector, timeout=3000)
                            logger.debug(f"Found content selector: {selector}", url=url)
                            break
                        except:
                            continue
                except Exception:
                    # If no selector found, just continue (page may still have content)
                    pass
                
                # Small stabilization sleep (200-500ms) instead of networkidle
                stabilization_sleep = random.uniform(0.2, 0.5)
                await asyncio.sleep(stabilization_sleep)

                # Scroll down to load dynamic content if enabled
                if scroll:
                    await self._scroll_and_load(page)

                # Get final HTML content
                html = await page.content()
                await browser.close()

                # Parse with BeautifulSoup
                return self._parse_html(html, url, extract_markdown)

        except Exception as e:
            error_msg = str(e) if e else "Unknown Playwright error"
            error_type = type(e).__name__ if e else "UnknownError"
            
            # CRITICAL: Check if error is "Download is starting" - this means it's a PDF or binary file
            # Don't retry Playwright in this case, go directly to HTTP scraping
            is_download_error = "download is starting" in error_msg.lower() or "download" in error_msg.lower()
            
            logger.error(
                "Playwright scraping failed",
                error=error_msg,
                error_type=error_type,
                url=url,
                is_download_error=is_download_error
            )
            # Fallback to HTTP scraping
            # CRITICAL: Pass playwright_tried=True to prevent infinite loop (403 -> Playwright -> 403 -> ...)
            logger.info("Falling back to HTTP scraping", url=url, playwright_tried=True)
            return await self._scrape_with_http(url, extract_markdown, playwright_tried=True)

    async def _scroll_and_load(self, page) -> None:
        """Scroll page down to trigger dynamic content loading."""
        logger.info("Starting scroll sequence", max_scrolls=self.max_scrolls)

        previous_height = 0
        scroll_count = 0

        for i in range(self.max_scrolls):
            # Get current page height
            current_height = await page.evaluate("document.body.scrollHeight")

            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for new content to load
            await asyncio.sleep(self.scroll_pause)

            # CRITICAL: Don't use networkidle - just wait for content selectors or use small sleep
            # Wait for potential new content (small sleep instead of networkidle)
            await asyncio.sleep(0.3)

            # Check if new content was loaded
            new_height = await page.evaluate("document.body.scrollHeight")

            if new_height == previous_height:
                logger.info("No new content loaded, stopping scroll", scroll_count=i + 1)
                break

            previous_height = new_height
            scroll_count += 1
            logger.debug("Scrolled and loaded content", scroll_count=scroll_count, height=new_height)

        # Scroll back to top to ensure we have all content
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.5)

        logger.info("Scroll sequence completed", total_scrolls=scroll_count)

    async def _scrape_with_http(self, url: str, extract_markdown: bool = True, playwright_tried: bool = False) -> ScrapedContent:
        """Scrape using standard HTTP request with persistent session."""
        await self._ensure_session()
        
        try:
            async with self._session.get(url) as response:
                # Check for 403 Forbidden - auto-fallback to Playwright
                # CRITICAL: Only try Playwright if it wasn't already tried (to prevent infinite loop)
                if response.status == 403 and not playwright_tried:
                    logger.warning(
                        "Received 403 Forbidden, attempting Playwright fallback",
                        url=url,
                        status=response.status
                    )
                    # Auto-fallback to Playwright if available (even if use_playwright=False)
                    if await self._check_playwright_available():
                        try:
                            logger.info("Auto-fallback to Playwright for 403 error", url=url)
                            return await self._scrape_with_playwright(url, extract_markdown, scroll=self.scroll_enabled)
                        except Exception as playwright_error:
                            logger.warning(
                                "Playwright fallback also failed",
                                url=url,
                                error=str(playwright_error)
                            )
                    # If no Playwright or it failed, raise the 403 error
                    response.raise_for_status()
                elif response.status == 403:
                    # Playwright already tried, just raise
                    response.raise_for_status()
                
                response.raise_for_status()

                # Check if it's a PDF file
                content_type = response.headers.get("Content-Type", "").lower()
                if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                    return await self._scrape_pdf(url, response)

                # Try to read as text with proper encoding handling
                # Many sites (especially Russian) don't specify encoding correctly
                html = None
                try:
                    # Read raw bytes first (we'll decode them ourselves)
                    raw_bytes = await response.read()
                    
                    # First, try to get encoding from Content-Type header
                    content_type = response.headers.get("Content-Type", "").lower()
                    encoding = None
                    if "charset=" in content_type:
                        encoding = content_type.split("charset=")[1].split(";")[0].strip()
                    
                    # Try to detect encoding from HTML meta tag if not in header
                    if not encoding:
                        try:
                            # Try UTF-8 first to check for meta tag
                            sample = raw_bytes[:8192].decode('utf-8', errors='ignore')
                            # Look for charset in meta tag
                            charset_match = re.search(r'<meta[^>]*charset=["\']?([^"\'>\s]+)', sample, re.IGNORECASE)
                            if charset_match:
                                encoding = charset_match.group(1).lower()
                        except:
                            pass
                    
                    # Try detected encoding first
                    if encoding:
                        try:
                            html = raw_bytes.decode(encoding)
                            logger.debug("Successfully decoded with detected encoding", url=url, encoding=encoding)
                        except (UnicodeDecodeError, LookupError) as e:
                            logger.debug("Failed to decode with detected encoding, trying common encodings", 
                                       url=url, encoding=encoding, error=str(e))
                            encoding = None
                    
                    # If no encoding detected or failed, try common encodings
                    if not html:
                        # Common encodings for web content (especially Russian sites)
                        common_encodings = ['utf-8', 'windows-1251', 'cp1251', 'iso-8859-1', 'latin1', 'cp866', 'koi8-r']
                        for enc in common_encodings:
                            try:
                                html = raw_bytes.decode(enc)
                                logger.debug("Successfully decoded with encoding", url=url, encoding=enc)
                                break
                            except (UnicodeDecodeError, LookupError):
                                continue
                    
                    # If still no success, try with errors='replace' to at least get some content
                    if not html:
                        try:
                            html = raw_bytes.decode('utf-8', errors='replace')
                            logger.warning("Decoded with UTF-8 and error replacement (some characters may be lost)", url=url)
                        except Exception as e:
                            logger.error("Failed to decode response with all methods", url=url, error=str(e))
                            return ScrapedContent(
                                url=url,
                                title="Unable to decode content",
                                content="",
                                markdown=None,
                                html=None,
                                images=[],
                                links=[],
                            )
                            
                except Exception as e:
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

            return self._parse_html(html, url, extract_markdown)

        except aiohttp.ClientResponseError as e:
            # Handle 403 and other HTTP errors
            # Note: Playwright fallback is handled in scrape() method to prevent infinite loops
            error_msg = str(e) if e else "Unknown connection error"
            error_type = type(e).__name__
            logger.error(
                "Web scraping failed - HTTP error",
                error=error_msg,
                error_type=error_type,
                status=getattr(e, 'status', None),
                url=url
            )
            raise
        except asyncio.TimeoutError as e:
            # Handle timeout errors gracefully
            error_msg = str(e) if e else "Request timeout"
            logger.warning(
                "Web scraping timeout",
                error=error_msg,
                error_type="TimeoutError",
                url=url,
                timeout_seconds=self.timeout.total if hasattr(self.timeout, 'total') else None
            )
            raise
        except aiohttp.ClientError as e:
            error_msg = str(e) if e else "Unknown connection error"
            error_type = type(e).__name__
            logger.error(
                "Web scraping failed - connection error",
                error=error_msg,
                error_type=error_type,
                url=url
            )
            raise
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            error_type = type(e).__name__
            logger.error(
                "Web scraping failed",
                error=error_msg,
                error_type=error_type,
                url=url,
                exc_info=True  # Include full traceback
            )
            raise

    async def close(self):
        """Close the aiohttp session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed aiohttp session")
        if self._connector:
            await self._connector.close()
            logger.debug("Closed TCP connector")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _parse_html(self, html: str, url: str, extract_markdown: bool = True) -> ScrapedContent:
        """Parse HTML and extract content."""
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
            # CRITICAL: Use longer timeout for PDF reading (PDFs can be large)
            # Read PDF as bytes with extended timeout (60 seconds instead of default 30)
            try:
                pdf_bytes = await asyncio.wait_for(response.read(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "PDF download timeout - file may be too large",
                    url=url,
                    timeout_seconds=60
                )
                # Return partial result with error message
                return ScrapedContent(
                    url=url,
                    title=url.split("/")[-1],
                    content="[PDF file - download timeout, file may be too large]",
                    markdown=None,
                    html=None,
                    images=[],
                    links=[],
                )
            
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
                # CRITICAL: Handle NullObject errors (some PDFs contain NullObject references)
                text_content = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                    except Exception as page_error:
                        # Handle NullObject and other page extraction errors
                        error_type = type(page_error).__name__
                        if "NullObject" in str(page_error) or "NullObject" in error_type:
                            logger.warning(f"PDF page {page_num + 1} contains NullObject - skipping", 
                                         url=url, page=page_num + 1, error=str(page_error))
                            text_content += f"[Page {page_num + 1}: Content extraction failed - NullObject reference]\n"
                        else:
                            logger.warning(f"PDF page {page_num + 1} extraction failed", 
                                         url=url, page=page_num + 1, error=str(page_error))
                            text_content += f"[Page {page_num + 1}: Content extraction failed]\n"
                
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
        except asyncio.TimeoutError:
            logger.warning(
                "PDF scraping timeout",
                url=url,
                timeout_seconds=60
            )
            return ScrapedContent(
                url=url,
                title=url.split("/")[-1],
                content="[PDF file - download timeout, file may be too large]",
                markdown=None,
                html=None,
                images=[],
                links=[],
            )
        except Exception as e:
            error_msg = str(e) if e else "Unknown PDF scraping error"
            error_type = type(e).__name__ if e else "UnknownError"
            logger.error(
                "PDF scraping failed",
                url=url,
                error=error_msg,
                error_type=error_type,
                exc_info=True
            )
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
