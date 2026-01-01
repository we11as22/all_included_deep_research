"""Mock web scraper for testing."""


class MockScrapedContent:
    """Mock scraped content."""

    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content
        self.metadata = {"scraped": True}


class MockScraper:
    """Mock web scraper for testing."""

    def __init__(self, fixed_content: dict | None = None):
        """
        Args:
            fixed_content: Dict mapping URLs to {title, content} to return
        """
        self.fixed_content = fixed_content or {
            "https://docs.python.org": {
                "title": "Python Documentation",
                "content": """
# Python Documentation

Python is a powerful programming language that lets you work quickly and integrate systems more effectively.

## Features

- **Easy to Learn**: Python has a simple syntax that makes it easy to learn.
- **Versatile**: Can be used for web development, data science, automation, and more.
- **Large Community**: Huge ecosystem of libraries and frameworks.
- **Cross-platform**: Runs on Windows, Mac, Linux, and more.

## Getting Started

1. Install Python from python.org
2. Write your first program: `print("Hello, World!")`
3. Explore the standard library
4. Learn about packages with pip

## Use Cases

Python is used by companies like Google, Netflix, Instagram, and many more for:
- Web applications (Django, Flask)
- Data analysis (Pandas, NumPy)
- Machine learning (TensorFlow, PyTorch)
- Automation and scripting
- Scientific computing

For more information, visit the official documentation at docs.python.org.
                """.strip(),
            },
            "https://en.wikipedia.org/wiki/Python_(programming_language)": {
                "title": "Python (programming language) - Wikipedia",
                "content": """
# Python (programming language)

Python is an interpreted, high-level and general-purpose programming language.

## History

Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant indentation.

## Design Philosophy

The language's core philosophy is summarized in "The Zen of Python":
- Beautiful is better than ugly
- Explicit is better than implicit
- Simple is better than complex
- Readability counts

## Features

- Dynamic typing and automatic memory management
- Supports multiple programming paradigms: object-oriented, imperative, functional, procedural
- Comprehensive standard library
- Large ecosystem of third-party packages

## Implementations

- CPython (reference implementation)
- PyPy (JIT compiler)
- Jython (Java platform)
- IronPython (.NET platform)

## Applications

Python is widely used in:
- Web development
- Scientific computing and data analysis
- Artificial intelligence and machine learning
- Education
- Software testing and quality assurance
                """.strip(),
            },
            "https://www.learnpython.org": {
                "title": "Learn Python - Free Interactive Python Tutorial",
                "content": """
# Learn Python

Welcome to the LearnPython.org interactive Python tutorial!

## Why Learn Python?

Python is one of the most popular programming languages in 2024:
- Easy to learn for beginners
- Powerful for experts
- Used in industry and academia

## Course Outline

### 1. Basics
- Variables and Types
- Lists
- Basic Operators
- String Formatting

### 2. Advanced Topics
- Conditions
- Loops
- Functions
- Classes and Objects
- Dictionaries
- Modules and Packages

### 3. Practical Applications
- File I/O
- Working with CSV
- JSON processing
- HTTP requests
- Database operations

## Start Learning

Click on any topic to start learning Python interactively!
                """.strip(),
            },
        }
        self.scrape_count = 0

    async def scrape(self, url: str) -> MockScrapedContent:
        """Mock scraping operation."""
        self.scrape_count += 1

        # Check if we have fixed content for this URL
        if url in self.fixed_content:
            data = self.fixed_content[url]
            return MockScrapedContent(
                title=data["title"],
                url=url,
                content=data["content"]
            )

        # Default content for unknown URLs
        return MockScrapedContent(
            title=f"Mock Page: {url}",
            url=url,
            content=f"""
# Mock Page Content

This is mock content scraped from {url}.

## About

This page contains sample information for testing purposes.

## Details

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

For more information, visit the source URL.
            """.strip()
        )


class MockWebScraper:
    """Alternative mock scraper interface."""

    def __init__(self):
        self._scraper = MockScraper()

    async def scrape_url(self, url: str, **kwargs) -> dict:
        """Scrape URL and return dict."""
        result = await self._scraper.scrape(url)
        return {
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "success": True,
        }
