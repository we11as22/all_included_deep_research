"""Main entry point for running the backend server."""

import uvicorn

from src.config.settings import get_settings


def main():
    """Start the FastAPI server."""
    settings = get_settings()

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

