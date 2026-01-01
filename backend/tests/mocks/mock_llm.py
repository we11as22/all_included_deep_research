"""Mock LLM for testing."""

from typing import Any, Dict, List
from pydantic import BaseModel


class MockChatModel:
    """Mock chat model for testing."""

    def __init__(self, responses: List[str] | None = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.structured_schema = None

    async def ainvoke(self, messages: List[Any]) -> Any:
        """Mock async invocation."""
        self.call_count += 1

        # If structured output schema is set, return instance
        if self.structured_schema:
            # Return a mock instance of the schema
            return self._create_mock_instance()

        # Otherwise return simple message
        class MockMessage:
            def __init__(self, content: str):
                self.content = content
                self.tool_calls = []

        response_idx = min(self.call_count - 1, len(self.responses) - 1)
        return MockMessage(self.responses[response_idx])

    def with_structured_output(self, schema: type[BaseModel], method: str = "function_calling"):
        """Mock structured output."""
        self.structured_schema = schema

        class StructuredMockLLM:
            def __init__(self, parent):
                self.parent = parent

            async def ainvoke(self, messages: List[Any]):
                return self.parent._create_mock_instance()

        return StructuredMockLLM(self)

    def _create_mock_instance(self) -> BaseModel:
        """Create a mock instance of the structured schema."""
        if not self.structured_schema:
            raise ValueError("No schema set")

        # Import common schemas
        schema_name = self.structured_schema.__name__

        if schema_name == "QueryClassification":
            from src.workflow.search.classifier import QueryClassification
            return QueryClassification(
                reasoning="Mock classification reasoning",
                query_type="factual",
                standalone_query="What is Python?",
                suggested_mode="web",
                requires_sources=True,
                time_sensitive=False
            )

        elif schema_name == "CitedAnswer":
            from src.workflow.search.writer import CitedAnswer
            return CitedAnswer(
                reasoning="Mock writer reasoning",
                answer="Python is a high-level programming language [1].",
                citations=[{"number": "1", "url": "https://python.org", "title": "Python Docs"}],
                confidence="high"
            )

        elif schema_name == "ResearchPlan":
            return self.structured_schema(
                reasoning="Mock research planning",
                topics=["Topic 1", "Topic 2", "Topic 3"],
                num_topics=3
            )

        elif schema_name == "SupervisorReActOutput":
            return self.structured_schema(
                reasoning="Mock supervisor reasoning",
                should_continue=True,
                replanning_needed=False,
                directives=[],
                new_topics=[],
                gaps_identified=[]
            )

        # Generic mock for other schemas
        # Try to instantiate with default/minimal values
        try:
            fields = {}
            for field_name, field_info in self.structured_schema.model_fields.items():
                if field_name == "reasoning":
                    fields[field_name] = "Mock reasoning"
                elif field_info.annotation == str:
                    fields[field_name] = f"Mock {field_name}"
                elif field_info.annotation == bool:
                    fields[field_name] = False
                elif field_info.annotation == int:
                    fields[field_name] = 0
                elif field_info.annotation == list or "list" in str(field_info.annotation).lower():
                    fields[field_name] = []
                elif field_info.annotation == dict or "dict" in str(field_info.annotation).lower():
                    fields[field_name] = {}

            return self.structured_schema(**fields)
        except Exception:
            # Fallback: return None
            raise ValueError(f"Cannot create mock for schema: {schema_name}")


class MockLLM:
    """High-level mock LLM wrapper."""

    def __init__(self, provider: str = "mock", model: str = "mock-model", **kwargs):
        self.provider = provider
        self.model = model
        self._client = MockChatModel()

    async def ainvoke(self, messages: List[Any]) -> Any:
        return await self._client.ainvoke(messages)

    def with_structured_output(self, schema: type[BaseModel], method: str = "function_calling"):
        return self._client.with_structured_output(schema, method)
