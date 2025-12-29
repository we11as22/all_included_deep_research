"""Base search provider interface."""

from abc import ABC, abstractmethod

from src.search.models import ScrapedContent, SearchResponse


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """
        Search the web for a query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            SearchResponse with results
        """
        pass

    @abstractmethod
    async def scrape(self, url: str) -> ScrapedContent:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent with extracted data
        """
        pass

    async def search_and_scrape(
        self, query: str, max_results: int = 10, scrape_top_n: int = 3
    ) -> tuple[SearchResponse, list[ScrapedContent]]:
        """
        Search and scrape top results.

        Args:
            query: Search query
            max_results: Maximum search results
            scrape_top_n: Number of top results to scrape

        Returns:
            Tuple of (SearchResponse, list of ScrapedContent)
        """
        search_response = await self.search(query, max_results)

        # Scrape top N results
        scraped = []
        for result in search_response.results[:scrape_top_n]:
            try:
                content = await self.scrape(result.url)
                scraped.append(content)
            except Exception:
                # Continue if scraping fails for a specific URL
                continue

        return search_response, scraped
