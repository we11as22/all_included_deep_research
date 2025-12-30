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

                logger.info("SearXNG search request", url=url, params=params, query=query)
                
                try:
                    async with session.get(url, params=params) as response:
                        response_text = await response.text()
                        
                        logger.debug(
                            "SearXNG raw response",
                            status=response.status,
                            content_length=len(response_text),
                            content_preview=response_text[:200] if response_text else "empty",
                        )
                        
                        if response.status != 200:
                            logger.warning(
                                "SearXNG returned non-200 status",
                                status=response.status,
                                error=response_text[:500],
                                query=query,
                                url=url,
                            )
                            return SearchResponse(query=query, results=[], total_results=0)
                        
                        try:
                            data = await response.json()
                            logger.info(
                                "SearXNG response received",
                                query=query,
                                results_count=len(data.get("results", [])),
                                number_of_results=data.get("number_of_results", 0),
                                has_results=bool(data.get("results")),
                                response_keys=list(data.keys()) if isinstance(data, dict) else "not_dict",
                            )
                            
                            # Log first result structure for debugging
                            if data.get("results") and len(data.get("results", [])) > 0:
                                first_result = data["results"][0]
                                logger.debug(
                                    "SearXNG first result structure",
                                    result_keys=list(first_result.keys()) if isinstance(first_result, dict) else "not_dict",
                                    has_url="url" in first_result if isinstance(first_result, dict) else False,
                                    has_title="title" in first_result if isinstance(first_result, dict) else False,
                                    has_content="content" in first_result if isinstance(first_result, dict) else False,
                                )
                        except Exception as e:
                            logger.error(
                                "SearXNG response parse failed",
                                error=str(e),
                                response_preview=response_text[:1000],
                                query=query,
                                status=response.status,
                                content_type=response.headers.get("content-type", "unknown"),
                            )
                            return SearchResponse(query=query, results=[], total_results=0)
                        
                        results = []
                        raw_results = data.get("results", [])
                        
                        logger.info(
                            "Processing SearXNG results",
                            query=query,
                            raw_results_count=len(raw_results),
                            max_results=max_results,
                        )
                        
                        # Perplexica approach: simple mapping without strict template filtering
                        for idx, item in enumerate(raw_results):
                            # Only require URL - title can be empty, we'll use URL as fallback
                            if not item.get("url"):
                                logger.debug("Skipping result without URL", index=idx, item_keys=list(item.keys()))
                                continue
                            
                            # Use content or title (like Perplexica does)
                            content = item.get("content") or item.get("snippet") or ""
                            title = item.get("title") or item.get("url", "")[:100]  # Use URL as fallback for title

                            results.append(
                                SearchResult(
                                    title=title,
                                    url=item.get("url", ""),
                                    snippet=content,
                                    score=item.get("score", 0.0),
                                    published_date=item.get("publishedDate"),
                                )
                            )
                            
                            if len(results) >= max_results:
                                break

                        # Use actual results count if number_of_results is 0 or missing
                        total_results = data.get("number_of_results", 0)
                        if total_results == 0 and len(results) > 0:
                            total_results = len(results)
                        
                        logger.info(
                            "SearXNG search completed",
                            query=query,
                            results_count=len(results),
                            total_results=total_results,
                            raw_results_processed=len(raw_results),
                            number_of_results_from_api=data.get("number_of_results", 0),
                        )
                        
                        if len(results) == 0 and len(raw_results) > 0:
                            logger.warning(
                                "SearXNG returned raw results but none were valid",
                                query=query,
                                raw_results_count=len(raw_results),
                                sample_result=raw_results[0] if raw_results else None,
                            )

                        return SearchResponse(
                            query=query,
                            results=results,
                            total_results=total_results,
                        )
                except aiohttp.ClientError as e:
                    logger.error("SearXNG search failed - connection error", error=str(e), query=query, url=url)
                    return SearchResponse(query=query, results=[], total_results=0)

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
