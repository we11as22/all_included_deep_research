#!/usr/bin/env python3
"""Simple OpenRouter test."""

import os
from pathlib import Path

import pytest
from openai import AsyncOpenAI

from src.config.settings import Settings


def _load_settings() -> Settings:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        return Settings(_env_file=env_path)
    return Settings()


@pytest.mark.asyncio
async def test_openrouter():
    """Test OpenRouter API directly."""
    settings = _load_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    base_url = settings.openai_base_url or "https://openrouter.ai/api/v1"
    headers = {}
    if settings.openai_api_http_referer:
        headers["HTTP-Referer"] = settings.openai_api_http_referer
    if settings.openai_api_x_title:
        headers["X-Title"] = settings.openai_api_x_title

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=headers or None,
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello from OpenRouter!' in one sentence."}
        ],
        max_tokens=50,
    )

    content = response.choices[0].message.content
    assert content
