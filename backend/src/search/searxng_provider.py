"""SearXNG search provider implementation."""

import aiohttp
import structlog

from src.search.base import SearchProvider
from src.search.models import ScrapedContent, SearchResponse, SearchResult

logger = structlog.get_logger(__name__)


class SearXNGSearchProvider(SearchProvider):
    """SearXNG metasearch engine provider."""

    def __init__(
        self,
        instance_url: str,
        timeout: int = 30,
        language: str = "en",
        categories: str = "",
        engines: str = "",
        safesearch: int = 0,
    ):
        """
        Initialize SearXNG provider.

        Args:
            instance_url: SearXNG instance URL (e.g., http://localhost:8080)
            timeout: Request timeout in seconds
        """
        self.instance_url = instance_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.language = language
        self.categories = self._split_list(categories)
        self.engines = self._split_list(engines)
        self.safesearch = safesearch
        logger.info("SearXNGSearchProvider initialized", instance_url=self.instance_url)

    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """
        Search using SearXNG API.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            SearchResponse with results
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # SearXNG search API endpoint
                url = f"{self.instance_url}/search"
                params = {
                    "q": query,
                    "format": "json",
                    "language": self.language,
                    "safesearch": self.safesearch,
                }

                if self.categories:
                    params["categories"] = ",".join(self.categories)
                if self.engines:
                    params["engines"] = ",".join(self.engines)

                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                results = []
                for item in data.get("results", [])[:max_results]:
                    # SearXNG returns various result types, we only want web results
                    if item.get("template") not in [None, "default"]:
                        continue

                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("content", ""),
                            score=item.get("score", 0.0),
                            published_date=item.get("publishedDate"),
                        )
                    )

                logger.info("SearXNG search completed", query=query, results_count=len(results))

                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=data.get("number_of_results", len(results)),
                )

        except aiohttp.ClientError as e:
            logger.error("SearXNG search failed - connection error", error=str(e), query=query)
            return SearchResponse(query=query, results=[], total_results=0)
        except Exception as e:
            logger.error("SearXNG search failed", error=str(e), query=query)
            return SearchResponse(query=query, results=[], total_results=0)

    async def scrape(self, url: str) -> ScrapedContent:
        """
        Scrape URL content.

        Note: SearXNG doesn't provide built-in scraping,
        so we use a generic HTTP fetch and basic HTML cleaning.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent with extracted data
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()

                # Basic HTML cleaning - will be improved by scraper.py
                from html.parser import HTMLParser
                from io import StringIO

                class HTMLToText(HTMLParser):
                    """Simple HTML to text converter."""

                    def __init__(self):
                        super().__init__()
                        self.text = StringIO()
                        self.skip_tags = {"script", "style", "noscript"}
                        self.in_skip_tag = False

                    def handle_starttag(self, tag, attrs):
                        if tag in self.skip_tags:
                            self.in_skip_tag = True

                    def handle_endtag(self, tag):
                        if tag in self.skip_tags:
                            self.in_skip_tag = False

                    def handle_data(self, data):
                        if not self.in_skip_tag:
                            self.text.write(data)
                            self.text.write(" ")

                    def get_text(self):
                        return self.text.getvalue().strip()

                parser = HTMLToText()
                parser.feed(html)
                content = parser.get_text()

                # Extract title from HTML
                title = ""
                title_start = html.find("<title>")
                title_end = html.find("</title>")
                if title_start != -1 and title_end != -1:
                    title = html[title_start + 7 : title_end].strip()

                scraped = ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    markdown=None,  # Will be added by scraper.py
                    html=html,
                    images=[],  # Will be extracted by scraper.py
                    links=[],  # Will be extracted by scraper.py
                )

                logger.info("SearXNG scrape completed", url=url, content_length=len(content))

                return scraped

        except aiohttp.ClientError as e:
            logger.error("SearXNG scrape failed - connection error", error=str(e), url=url)
            raise
        except Exception as e:
            logger.error("SearXNG scrape failed", error=str(e), url=url)
            raise

    def _split_list(self, raw: str) -> list[str]:
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
