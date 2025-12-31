"""Application settings with environment variable support."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")

    # Database Settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="deep_research", description="PostgreSQL database")
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")
    postgres_password: str = Field(..., description="PostgreSQL password")
    database_pool_size: int = Field(default=10, description="Database connection pool size")

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Construct sync database URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Memory Settings
    memory_dir: str = Field(default="./memory_files", description="Memory files directory")
    chunk_size: int = Field(default=800, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")

    # Embedding Settings
    embedding_provider: Literal["openai", "ollama", "cohere", "huggingface", "mock"] = Field(
        default="openai", description="Embedding provider"
    )
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")

    # OpenAI Embeddings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(
        default=None, description="OpenAI API base URL (for OpenRouter, 302.AI, or any OpenAI-compatible API)"
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    
    # OpenAI-compatible API headers (for OpenRouter, etc.)
    openai_api_http_referer: Optional[str] = Field(
        default=None, description="HTTP-Referer header for OpenAI-compatible APIs (e.g., OpenRouter)"
    )
    openai_api_x_title: Optional[str] = Field(
        default=None, description="X-Title header for OpenAI-compatible APIs (e.g., OpenRouter)"
    )

    # Ollama Embeddings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", description="Ollama embedding model")

    # Cohere Embeddings
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    cohere_embedding_model: str = Field(default="embed-english-v3.0", description="Cohere embedding model")
    cohere_input_type: str = Field(default="search_document", description="Cohere input type")

    # HuggingFace Embeddings
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    huggingface_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace model"
    )
    huggingface_use_local: bool = Field(default=True, description="Use local HuggingFace model")

    # Search Settings
    search_provider: Literal["tavily", "searxng", "mock"] = Field(default="tavily", description="Search provider")
    
    # Web Scraper Settings
    scraper_timeout: int = Field(default=30, description="Web scraper timeout in seconds")
    scraper_use_playwright: bool = Field(default=False, description="Use Playwright for JavaScript rendering")
    scraper_scroll_enabled: bool = Field(default=False, description="Enable automatic scrolling to load dynamic content")
    scraper_scroll_pause: float = Field(default=1.0, description="Pause between scrolls in seconds")
    scraper_max_scrolls: int = Field(default=5, description="Maximum number of scroll operations")

    # Tavily
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")
    tavily_max_results: int = Field(default=8, description="Tavily max results")

    # SearXNG
    searxng_instance_url: Optional[str] = Field(default="http://localhost:8080", description="SearXNG instance URL")
    searxng_api_key: Optional[str] = Field(default=None, description="SearXNG API key")
    searxng_max_results: int = Field(default=8, description="SearXNG max results")
    searxng_language: str = Field(default="en", description="SearXNG search language")
    searxng_categories: str = Field(default="", description="SearXNG categories (comma-separated)")
    searxng_engines: str = Field(default="", description="SearXNG engines (comma-separated)")
    searxng_safesearch: int = Field(default=0, description="SearXNG safe search level")

    # LLM Settings
    llm_mode: Literal["live", "mock"] = Field(default="live", description="LLM mode: live or mock")
    chat_model: str = Field(default="openai:gpt-4o-mini", description="Chat model for search answers")
    chat_model_max_tokens: int = Field(default=2048, description="Chat model max tokens")

    search_summarization_model: str = Field(
        default="openai:gpt-4o-mini", description="Model for summarizing scraped sources"
    )
    search_summarization_model_max_tokens: int = Field(default=1024, description="Summarization model max tokens")

    research_model: str = Field(default="openai:gpt-4o", description="Research model")
    research_model_max_tokens: int = Field(default=4096, description="Research model max tokens")

    compression_model: str = Field(default="openai:gpt-4o-mini", description="Compression model")
    compression_model_max_tokens: int = Field(default=2048, description="Compression model max tokens")

    final_report_model: str = Field(default="openai:gpt-4o", description="Final report model")
    final_report_model_max_tokens: int = Field(default=8192, description="Final report model max tokens")

    # Anthropic (for Claude models)
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Research Mode Configuration
    speed_max_iterations: int = Field(default=2, description="Speed mode max iterations")
    speed_max_concurrent: int = Field(default=1, description="Speed mode max concurrent researchers")

    balanced_max_iterations: int = Field(default=6, description="Balanced mode max iterations")
    balanced_max_concurrent: int = Field(default=3, description="Balanced mode max concurrent researchers")

    quality_max_iterations: int = Field(default=25, description="Quality mode max iterations")
    quality_max_concurrent: int = Field(default=4, description="Quality mode max concurrent researchers")

    # Advanced Settings
    memory_context_limit: int = Field(default=6, description="Max memory snippets for chat prompts")
    sources_limit: int = Field(default=8, description="Max sources to include in prompts")
    search_content_max_chars: int = Field(default=6000, description="Max chars per source before summarization")
    chat_history_limit: int = Field(default=6, description="Chat messages to include in prompts")
    simple_search_max_results: int = Field(default=5, description="Max results for simple search")
    simple_search_scrape_top_n: int = Field(default=2, description="Top results to scrape in simple search")
    deep_search_max_results: int = Field(default=8, description="Max results per query in deep search")
    deep_search_queries: int = Field(default=3, description="Number of queries for deep search")
    deep_search_scrape_top_n: int = Field(default=4, description="Top results to scrape in deep search")
    deep_search_rerank_top_k: int = Field(default=6, description="Reranked results to keep in deep search")
    deep_search_iterations: int = Field(default=6, description="Search refinement iterations for deep search")
    deep_search_quality_max_results: int = Field(
        default=12, description="Max results per query in quality deep search"
    )
    deep_search_quality_queries: int = Field(default=3, description="Number of queries for quality deep search")
    deep_search_quality_scrape_top_n: int = Field(default=6, description="Top results to scrape in quality deep search")
    deep_search_quality_rerank_top_k: int = Field(
        default=10, description="Reranked results to keep in quality deep search"
    )
    deep_search_quality_iterations: int = Field(
        default=25, description="Search refinement iterations for quality deep search"
    )

    max_retries: int = Field(default=3, description="Max retries for API calls")
    max_structured_output_retries: int = Field(default=3, description="Max retries for structured output")
    rrf_k: int = Field(default=60, description="RRF K parameter")
    embedding_batch_size: int = Field(default=100, description="Embedding batch size")
    allow_clarification: bool = Field(default=True, description="Allow clarification questions in quality mode")
    debug_mode: bool = Field(default=False, description="Enable debug logging for streams and frontend sync")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
