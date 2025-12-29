"""Mock search provider for offline testing."""

from __future__ import annotations

from urllib.parse import quote_plus

from src.search.base import SearchProvider
from src.search.models import ScrapedContent, SearchResponse, SearchResult


class MockSearchProvider(SearchProvider):
    """Return deterministic mock search results."""

    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        safe_query = quote_plus(query.strip() or "query")
        results = []

        for idx in range(max_results):
            results.append(
                SearchResult(
                    title=f"Mock Result {idx + 1} for {query}",
                    url=f"https://example.com/{safe_query}/{idx + 1}",
                    snippet=f"Mock snippet {idx + 1} about {query}.",
                    score=1.0 - (idx * 0.05),
                    published_date=None,
                )
            )

        return SearchResponse(query=query, results=results, total_results=len(results))

    async def scrape(self, url: str) -> ScrapedContent:
        return ScrapedContent(
            url=url,
            title="Mock Source",
            content=f"Mock content extracted from {url}.",
            markdown=None,
            html=None,
            images=[],
            links=[],
        )
