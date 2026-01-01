"""Mock search provider for testing."""

from typing import List


class MockSearchResult:
    """Mock search result."""

    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content
        self.snippet = content[:200]


class MockSearchResponse:
    """Mock search response."""

    def __init__(self, results: List[MockSearchResult]):
        self.results = results


class MockSearchProvider:
    """Mock search provider for testing."""

    def __init__(self, fixed_results: List[dict] | None = None):
        """
        Args:
            fixed_results: List of dicts with {title, url, content} to return
        """
        self.fixed_results = fixed_results or [
            {
                "title": "Python Official Documentation",
                "url": "https://docs.python.org",
                "content": "Python is a high-level, interpreted programming language with dynamic semantics. "
                          "Its high-level built-in data structures, combined with dynamic typing and binding, "
                          "make it very attractive for Rapid Application Development.",
            },
            {
                "title": "Python Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "content": "Python is an interpreted, high-level and general-purpose programming language. "
                          "Python's design philosophy emphasizes code readability with its notable use of "
                          "significant indentation.",
            },
            {
                "title": "Learn Python - Beginner's Guide",
                "url": "https://www.learnpython.org",
                "content": "Python is a popular programming language. It was created by Guido van Rossum, "
                          "and released in 1991. It is used for web development, software development, "
                          "mathematics, system scripting.",
            },
        ]
        self.search_count = 0

    async def search(self, query: str, max_results: int = 5) -> MockSearchResponse:
        """Mock search operation."""
        self.search_count += 1

        # Return fixed results (limited by max_results)
        results = [
            MockSearchResult(
                title=r["title"],
                url=r["url"],
                content=r["content"]
            )
            for r in self.fixed_results[:max_results]
        ]

        return MockSearchResponse(results)


class MockTavilySearch:
    """Mock Tavily search (matches actual API interface)."""

    def __init__(self, api_key: str = "mock"):
        self.api_key = api_key
        self.provider = MockSearchProvider()

    async def search(self, query: str, max_results: int = 5, **kwargs) -> dict:
        """Mock Tavily search."""
        response = await self.provider.search(query, max_results)

        return {
            "query": query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                    "score": 0.9,
                }
                for r in response.results
            ]
        }
