"""Memory tools for searching and saving to memory system."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class MemorySearchInput(BaseModel):
    """Input for memory search tool."""

    query: str = Field(..., description="Search query for memory system")
    category: str | None = Field(
        default=None,
        description="Optional category filter: project, concept, conversation, preference, main, other",
    )
    max_results: int = Field(default=5, description="Maximum number of results to return", ge=1, le=20)


class MemorySaveInput(BaseModel):
    """Input for memory save tool."""

    title: str = Field(..., description="Title for the memory entry")
    content: str = Field(..., description="Content to save to memory")
    category: str = Field(
        default="concept",
        description="Category: project, concept, conversation, preference, main, other",
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


@tool
def search_memory_tool(
    query: Annotated[str, "Search query for memory system"],
    category: Annotated[
        str | None,
        "Optional category filter: project, concept, conversation, preference, main, other",
    ] = None,
    max_results: Annotated[int, "Maximum number of results to return (1-20)"] = 5,
) -> str:
    """
    Search the memory system for relevant past research and knowledge.

    Use this tool to:
    - Find previous research on similar topics
    - Retrieve stored knowledge about a subject
    - Check if we've already answered similar questions
    - Access project-specific information
    - Review past conversations and preferences

    The memory system uses hybrid search (semantic + keyword) for best results.

    Args:
        query: What to search for in memory
        category: Filter by category (optional)
        max_results: How many results to return (1-20)

    Returns:
        Formatted search results with relevant memory chunks
    """
    # This is a placeholder - actual memory search happens in the workflow
    # The tool call signals to use the HybridSearchEngine
    return f"Searching memory for: {query} (category: {category}, max_results: {max_results})"


@tool
def save_to_memory_tool(
    title: Annotated[str, "Title for the memory entry"],
    content: Annotated[str, "Content to save to memory"],
    category: Annotated[
        str, "Category: project, concept, conversation, preference, main, other"
    ] = "concept",
    tags: Annotated[list[str], "Tags for categorization"] = [],
) -> str:
    """
    Save research findings or knowledge to the memory system.

    Use this tool to:
    - Store important research findings for future reference
    - Save key insights and conclusions
    - Document project-specific information
    - Record user preferences and requirements
    - Create reusable knowledge entries

    The content will be saved as a markdown file and indexed for future retrieval.

    Args:
        title: Descriptive title for the memory entry
        content: The actual content to save (markdown format recommended)
        category: Category for organization
        tags: Optional tags for better categorization

    Returns:
        Confirmation that content has been saved
    """
    # This is a placeholder - actual saving happens in the workflow
    # The tool call signals to use FileSyncService
    return f"Saved to memory: '{title}' (category: {category}, tags: {tags})"


@tool
def list_memory_categories_tool() -> str:
    """
    List available memory categories and their descriptions.

    Returns:
        List of memory categories with descriptions
    """
    categories = """
Available Memory Categories:

- **project**: Project-specific information, requirements, architecture
- **concept**: Technical concepts, explanations, definitions
- **conversation**: Important past conversations and discussions
- **preference**: User preferences, coding style, workflow preferences
- **main**: General knowledge and reference material
- **other**: Miscellaneous entries that don't fit other categories

Use the appropriate category when saving to memory for better organization.
    """
    return categories.strip()
