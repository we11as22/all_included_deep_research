"""SearXNG search provider implementation."""

import aiohttp
import structlog
import re
from urllib.parse import urlparse

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

    def _build_params(self, query: str, engines_override: list[str] | None) -> dict[str, str | int]:
        """
        Build search parameters for SearXNG.
        
        Like Perplexica: minimal params - just query and format.
        Let SearXNG handle language detection, safesearch defaults, etc.
        Only specify engines if explicitly provided (for specialized searches).
        """
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
        }
        
        # Only add safesearch if explicitly configured (not default)
        if self.safesearch != 0:
            params["safesearch"] = self.safesearch
        
        # Like Perplexica: NEVER specify language - always let SearXNG auto-detect
        # SearXNG is smart enough to detect language from query content
        
        # Only add categories if explicitly configured
        if self.categories:
            params["categories"] = ",".join(self.categories)
        
        # Handle engines: only specify if explicitly provided (like Perplexica)
        # For web_search: don't specify engines - let SearXNG auto-select
        # For specialized searches (academic, social): specify engines
        if engines_override is not None:
            # engines_override explicitly provided
            if engines_override:  # Non-empty list
                params["engines"] = ",".join(engines_override)
            # If empty list, don't add engines param - SearXNG auto-selects
        elif self.engines:
            # Use configured engines from settings (if any)
            params["engines"] = ",".join(self.engines)
        # If no engines specified at all, SearXNG will auto-select (like Perplexica)
        return params

    def _select_language(self, query: str) -> str:
        """
        Select language for SearXNG query.
        
        Like Perplexica: ALWAYS return empty string for auto-detect.
        SearXNG is smart enough to detect language from query content automatically.
        This works for all languages (English, Russian, Chinese, Japanese, Arabic, etc.)
        
        Returns:
            Always empty string (auto-detect)
        """
        # Like Perplexica: never specify language, always auto-detect
        return ""

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        return bool(re.search(r"[\u0400-\u04FF]", text))

    async def _search_once(
        self,
        session: aiohttp.ClientSession,
        query: str,
        max_results: int,
        engines_override: list[str] | None,
        label: str,
    ) -> SearchResponse:
        url = f"{self.instance_url}/search"
        params = self._build_params(query, engines_override)

        logger.info("SearXNG search request", url=url, params=params, query=query, label=label)

        try:
            async with session.get(url, params=params) as response:
                response_text = await response.text()

                logger.debug(
                    "SearXNG raw response",
                    status=response.status,
                    content_length=len(response_text),
                    content_preview=response_text[:200] if response_text else "empty",
                    label=label,
                )

                if response.status != 200:
                    logger.warning(
                        "SearXNG returned non-200 status",
                        status=response.status,
                        error=response_text[:500],
                        query=query,
                        url=url,
                        label=label,
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
                        label=label,
                    )

                    if data.get("results") and len(data.get("results", [])) > 0:
                        first_result = data["results"][0]
                        # Log first few results to debug relevance
                        top_3_results = []
                        for r in data.get("results", [])[:3]:
                            top_3_results.append({
                                "title": r.get("title", "")[:80] if isinstance(r, dict) else "",
                                "url": r.get("url", "")[:100] if isinstance(r, dict) else "",
                                "content_preview": (r.get("content", "") or r.get("snippet", ""))[:100] if isinstance(r, dict) else ""
                            })
                        logger.info(
                            "SearXNG top results preview",
                            query=query[:100],
                            top_3_results=top_3_results,
                            label=label,
                        )
                        logger.debug(
                            "SearXNG first result structure",
                            result_keys=list(first_result.keys()) if isinstance(first_result, dict) else "not_dict",
                            has_url="url" in first_result if isinstance(first_result, dict) else False,
                            has_title="title" in first_result if isinstance(first_result, dict) else False,
                            has_content="content" in first_result if isinstance(first_result, dict) else False,
                            label=label,
                        )
                except Exception as e:
                    logger.error(
                        "SearXNG response parse failed",
                        error=str(e),
                        response_preview=response_text[:1000],
                        query=query,
                        status=response.status,
                        content_type=response.headers.get("content-type", "unknown"),
                        label=label,
                    )
                    return SearchResponse(query=query, results=[], total_results=0)

                results = []
                raw_results = data.get("results", [])

                logger.info(
                    "Processing SearXNG results",
                    query=query,
                    raw_results_count=len(raw_results),
                    max_results=max_results,
                    label=label,
                )

                for idx, item in enumerate(raw_results):
                    if not item.get("url"):
                        logger.debug("Skipping result without URL", index=idx, item_keys=list(item.keys()), label=label)
                        continue

                    # Like Perplexica: use content || title (not snippet)
                    content = item.get("content") or item.get("title") or ""
                    title = item.get("title") or item.get("url", "")[:100]

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

                total_results = data.get("number_of_results", 0)
                if total_results == 0 and len(results) > 0:
                    total_results = len(results)

                # No filtering - just return results as-is
                # Semantic reranking (embeddings) will handle proper ranking by query similarity
                filtered_results = self._filter_results_by_query(query, results, max_results)

                # Log top results for debugging relevance
                top_results_info = []
                for r in filtered_results[:5]:
                    domain = ""
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(r.url).netloc if r.url else ""
                    except:
                        pass
                    top_results_info.append({
                        "title": r.title[:60] if r.title else "",
                        "domain": domain,
                        "url": r.url[:80] if r.url else ""
                    })
                
                logger.info(
                    "SearXNG search completed",
                    query=query[:100],
                    results_count=len(filtered_results),
                    total_results=total_results,
                    raw_results_processed=len(raw_results),
                    number_of_results_from_api=data.get("number_of_results", 0),
                    top_results=top_results_info,
                    label=label,
                )

                return SearchResponse(
                    query=query,
                    results=filtered_results,
                    total_results=total_results,
                )
        except aiohttp.ClientError as e:
            logger.error("SearXNG search failed - connection error", error=str(e), query=query, url=url, label=label)
            return SearchResponse(query=query, results=[], total_results=0)

    def _result_diversity(self, results: list[SearchResult]) -> tuple[int, int]:
        domains = []
        for item in results:
            if not item.url:
                continue
            try:
                domain = urlparse(item.url).netloc.lower()
            except Exception:
                continue
            if domain:
                domains.append(domain)
        return len(set(domains)), len(domains)

    def _should_fallback(self, results: list[SearchResult], max_results: int) -> bool:
        if not results:
            return True
        unique_domains, total = self._result_diversity(results)
        if total == 0:
            return True
        if total >= 3 and unique_domains <= 1:
            return True
        if total >= 6 and (unique_domains / total) < 0.34:
            return True
        if total < max(3, max_results // 2):
            return True
        return False

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text supporting all Unicode languages."""
        # Use Unicode word boundaries to support all languages (Latin, Cyrillic, CJK, etc.)
        # \w+ matches word characters in any language when used with re.UNICODE
        tokens = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
        # Filter out very short tokens (less than 2 chars)
        return [token for token in tokens if len(token) >= 2]


    def _filter_results_by_query(
        self, query: str, results: list[SearchResult], max_results: int
    ) -> list[SearchResult]:
        """
        Return results as-is from SearXNG.
        
        No filtering - we rely on semantic reranking (embeddings) in chat service
        to properly rank results by query similarity. SearXNG already does
        language-aware filtering and initial ranking.
        """
        if not results:
            return results
        
        # Just return top results as-is - semantic reranking will handle ranking
        return results[:max_results]


    def _prefer_fallback(
        self,
        primary_results: list[SearchResult],
        fallback_results: list[SearchResult],
    ) -> bool:
        if not fallback_results:
            return False
        primary_domains, primary_total = self._result_diversity(primary_results)
        fallback_domains, fallback_total = self._result_diversity(fallback_results)
        return (fallback_domains, fallback_total) > (primary_domains, primary_total)

    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """
        Search using SearXNG API.
        
        Like Perplexica: simple, single request. No fallback logic.
        Let SearXNG handle engine selection automatically.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            SearchResponse with results
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Simple search like Perplexica: one request, no fallback
                # SearXNG will automatically choose working engines
                return await self._search_once(
                    session,
                    query,
                    max_results=max_results,
                    engines_override=[],  # Empty = SearXNG auto-selects engines
                    label="search",
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
