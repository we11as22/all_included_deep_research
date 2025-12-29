"""Configuration endpoint."""

from fastapi import APIRouter, Request

from src.api.models.config import ConfigResponse
from src.config.settings import Settings

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    """Get application configuration."""
    settings: Settings = request.app.state.settings

    return ConfigResponse(
        search_provider=settings.search_provider,
        embedding_provider=settings.embedding_provider,
        speed_max_iterations=settings.speed_max_iterations,
        balanced_max_iterations=settings.balanced_max_iterations,
        quality_max_iterations=settings.quality_max_iterations,
        memory_enabled=True,
    )

