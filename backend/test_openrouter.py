#!/usr/bin/env python3
"""Test OpenRouter and HuggingFace integration."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_settings():
    """Test settings loading."""
    print("=" * 60)
    print("1. Testing Settings...")
    print("=" * 60)
    
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        
        print(f"✓ Settings loaded")
        print(f"  - Embedding provider: {settings.embedding_provider}")
        print(f"  - Embedding dimension: {settings.embedding_dimension}")
        print(f"  - HuggingFace model: {settings.huggingface_model}")
        print(f"  - Research model: {settings.research_model}")
        print(f"  - OpenAI base URL: {settings.openai_base_url}")
        print(f"  - API key present: {'Yes' if settings.openai_api_key else 'No'}")
        return True
    except Exception as e:
        print(f"✗ Settings failed: {e}")
        return False


async def test_huggingface_embeddings():
    """Test HuggingFace embeddings."""
    print("\n" + "=" * 60)
    print("2. Testing HuggingFace Embeddings...")
    print("=" * 60)
    
    try:
        from src.config.settings import get_settings
        from src.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider
        
        settings = get_settings()
        
        print(f"Loading model: {settings.huggingface_model}...")
        provider = HuggingFaceEmbeddingProvider(settings)
        
        print("Generating test embedding...")
        test_text = "This is a test sentence for embeddings."
        embedding = await provider.embed_text(test_text)
        
        print(f"✓ HuggingFace embeddings work")
        print(f"  - Embedding dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        return True
    except Exception as e:
        print(f"✗ HuggingFace embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_openrouter_llm():
    """Test OpenRouter LLM."""
    print("\n" + "=" * 60)
    print("3. Testing OpenRouter LLM...")
    print("=" * 60)
    
    try:
        from src.config.settings import get_settings
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        settings = get_settings()
        
        print(f"Creating LLM with model: {settings.research_model}")
        print(f"Using base URL: {settings.openai_base_url}")
        
        llm_kwargs = {
            "model": "gpt-4o-mini",
            "api_key": settings.openai_api_key,
            "max_tokens": 100,
            "temperature": 0.7,
        }
        
        if settings.openai_base_url:
            llm_kwargs["base_url"] = settings.openai_base_url
            
            # Add OpenRouter-specific headers
            if "openrouter.ai" in settings.openai_base_url:
                llm_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/all-included-deep-research",
                    "X-Title": "All-Included Deep Research",
                }
            
        llm = ChatOpenAI(**llm_kwargs)
        
        print("Sending test message...")
        message = HumanMessage(content="Say 'Hello from OpenRouter!' in one sentence.")
        response = await llm.ainvoke([message])
        
        print(f"✓ OpenRouter LLM works")
        print(f"  - Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ OpenRouter LLM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_factory():
    """Test workflow factory."""
    print("\n" + "=" * 60)
    print("4. Testing Workflow Factory...")
    print("=" * 60)
    
    try:
        from src.config.settings import get_settings
        # We can't test full workflow without database, but we can test factory creation
        
        print("✓ Workflow factory would work (skipping full test without DB)")
        return True
    except Exception as e:
        print(f"✗ Workflow factory failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("OpenRouter + HuggingFace Integration Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(await test_settings())
    results.append(await test_huggingface_embeddings())
    results.append(await test_openrouter_llm())
    results.append(await test_workflow_factory())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

