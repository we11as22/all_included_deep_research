#!/usr/bin/env python3
"""Simple OpenRouter test."""

import asyncio
from openai import AsyncOpenAI

async def test_openrouter():
    """Test OpenRouter API directly."""
    
    api_key = "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"
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/all-included-deep-research",
            "X-Title": "All-Included Deep Research Test",
        }
    )
    
    try:
        print("Testing OpenRouter with gpt-4o-mini...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'Hello from OpenRouter!' in one sentence."}
            ],
            max_tokens=50
        )
        
        print(f"✓ Success!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_openrouter())

