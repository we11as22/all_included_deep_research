"""Tavily search provider implementation."""

import structlog
from tavily import TavilyClient

from src.search.base import SearchProvider
from src.search.models import ScrapedContent, SearchResponse, SearchResult

logger = structlog.get_logger(__name__)


class TavilySearchProvider(SearchProvider):
    """Tavily API search provider."""

    def __init__(self, api_key: str):
        """
        Initialize Tavily provider.

        Args:
            api_key: Tavily API key
        """
        self.client = TavilyClient(api_key=api_key)
        logger.info("TavilySearchProvider initialized")

    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """
        Search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            SearchResponse with results
        """
        try:
            # Tavily search (synchronous API)
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # Use advanced search for better quality
                include_answer=False,  # We'll generate our own answers
                include_raw_content=False,  # Don't need raw HTML in search results
            )

            results = [
                SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    score=result.get("score", 0.0),
                    published_date=result.get("published_date"),
                )
                for result in response.get("results", [])
            ]

            logger.info("Tavily search completed", query=query, results_count=len(results))

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
            )

        except Exception as e:
            logger.error("Tavily search failed", error=str(e), query=query)
            return SearchResponse(query=query, results=[], total_results=0)

    async def scrape(self, url: str) -> ScrapedContent:
        """
        Scrape URL using Tavily extract API.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent with extracted data
        """
        try:
            # Tavily extract (synchronous API)
            result = self.client.extract(urls=[url])

            if not result or not result.get("results"):
                raise ValueError(f"No content extracted from {url}")

            extracted = result["results"][0]

            content = ScrapedContent(
                url=url,
                title=extracted.get("title", ""),
                content=extracted.get("raw_content", ""),
                markdown=None,  # Tavily doesn't provide markdown
                html=None,  # Not included by default
                images=[],  # Not extracted by Tavily
                links=[],  # Not extracted by Tavily
            )

            logger.info("Tavily scrape completed", url=url, content_length=len(content.content))

            return content

        except Exception as e:
            logger.error("Tavily scrape failed", error=str(e), url=url)
            raise
