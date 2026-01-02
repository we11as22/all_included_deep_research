# All-Included Deep Research: Complete Workflow Documentation

**Version**: 2.0
**Last Updated**: 2026-01-02
**Project**: Deep Research Console with Multi-Mode Search & Research

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Mode Comparison](#3-mode-comparison)
4. [Chat Mode Workflow](#4-chat-mode-workflow)
5. [Search/Web Mode Workflow](#5-searchweb-mode-workflow)
6. [Deep Search Mode Workflow](#6-deep-search-mode-workflow)
7. [Deep Research Mode Workflow](#7-deep-research-mode-workflow)
8. [Component Details](#8-component-details)
9. [Prompt Engineering](#9-prompt-engineering)
10. [Configuration & Tuning](#10-configuration--tuning)
11. [Best Practices](#11-best-practices)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Overview

### Purpose

The All-Included Deep Research system is a multi-mode AI research assistant that adapts its workflow based on query complexity and requirements. It provides:

- **Simple Q&A** for general knowledge questions
- **Web Search** for quick fact-finding
- **Deep Search** for thorough investigations
- **Deep Research** for comprehensive, multi-agent research projects

### Key Features

- ✅ **Intelligent Query Classification**: Automatically routes queries to appropriate mode
- ✅ **Multi-Agent Coordination**: Parallel research agents with supervisor oversight
- ✅ **Hybrid Search**: Vector similarity + full-text search for chat history
- ✅ **Perplexica-Style Architecture**: Two-stage researcher → writer pattern
- ✅ **Citation Management**: Every fact cited with inline references
- ✅ **Streaming Progress**: Real-time updates on research progress
- ✅ **Memory Integration**: Vector memory for context retrieval

### Technology Stack

**Backend**:
- Python 3.11+
- FastAPI (API server)
- LangChain (LLM orchestration)
- LangGraph (multi-agent workflows)
- PostgreSQL + pgvector (vector storage)
- AsyncPG (async database driver)
- Structlog (structured logging)

**Frontend**:
- Next.js 14.1.0
- React 18
- TypeScript
- TailwindCSS

**AI Models**:
- OpenAI GPT-4 / GPT-3.5-turbo (configurable)
- OpenAI text-embedding-ada-002 (embeddings)

---

## 2. Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Home   │  │  Search  │  │ Sidebar  │  │  Chat UI │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/SSE
┌────────────────────────┴────────────────────────────────────┐
│                  API LAYER (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /v1/chat/completions (streaming)                    │   │
│  │  /api/chats/* (CRUD)                                 │   │
│  │  /api/chats/search (hybrid search)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│               WORKFLOW ORCHESTRATION                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Chat Service │  │Search Service│  │Research Graph│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   AGENT LAYER                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Classifier│  │Researcher│  │  Writer  │  │Supervisor│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                 SERVICES & STORAGE                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Search  │  │  Scraper │  │   Memory │  │ Database │   │
│  │ Provider │  │          │  │  Service │  │ (Postgres)│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Query** → Frontend
2. **API Request** → Backend `/v1/chat/completions`
3. **Mode Detection** → Classifier determines route
4. **Workflow Execution** → Appropriate service handles request
5. **Streaming Response** → SSE events stream to frontend
6. **Storage** → Messages & embeddings saved to PostgreSQL
7. **UI Update** → Frontend displays streaming response

---

## 3. Mode Comparison

### Feature Matrix

| Feature | Chat | Search/Web | Deep Search | Deep Research |
|---------|------|------------|-------------|---------------|
| **Sources Required** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Web Search** | ❌ | ✅ | ✅ | ✅✅ |
| **Iterations** | N/A | 2 | 6 | Variable (6-25) |
| **Agents** | 1 (LLM only) | 1 (Researcher+Writer) | 1 (Researcher+Writer) | 4+ (Multi-agent) |
| **Supervisor** | ❌ | ❌ | ❌ | ✅ |
| **Citations** | ❌ | ✅ | ✅ | ✅✅ |
| **Memory Search** | ✅ | ✅ | ✅ | ✅✅ |
| **Typical Length** | 100-300 words | 200-400 words | 500-800 words | 1000-2000+ words |
| **Response Time** | Fast (<5s) | Medium (~15-30s) | Slow (~30-60s) | Very Slow (1-5 min) |
| **Best For** | Greetings, simple Q&A | Quick facts | Explanations | Comprehensive analysis |

### Mode Selection Guide

```
User Query Type                    →  Suggested Mode
─────────────────────────────────────────────────────────────
"Hello" / "Hi"                     →  chat
"What is 2+2?"                     →  chat
"Define photosynthesis"            →  chat (if pre-2024 knowledge)

"Who is CEO of Apple?"             →  web
"Current stock price of TSLA?"     →  web
"Latest news about SpaceX?"        →  web

"How does solar energy work?"      →  deep_search
"Compare React vs Vue"             →  deep_search
"Benefits of meditation?"          →  deep_search

"Analyze AI impact on healthcare"  →  deep_research (balanced)
"Comprehensive guide to blockchain" →  deep_research (quality)
"Compare renewable energy options" →  deep_research (balanced)
```

---

## 4. Chat Mode Workflow

### Overview

**Purpose**: Answer simple questions without web sources using LLM's training data.

**When to use**:
- Greetings and small talk
- Basic math calculations
- General knowledge (pre-2024)
- Definitions of common terms

### Workflow Diagram

```
User Query
    ↓
Classifier → query_type="simple" → suggested_mode="chat"
    ↓
ChatService.answer_simple()
    ↓
Memory Search (optional context)
    ↓
LLM Direct Response (no tools)
    ↓
Stream Response to User
    ↓
Save to Database (with embedding)
```

### Step-by-Step Process

1. **Query Reception**
   - User sends message
   - Frontend → `/v1/chat/completions` with mode="chat"

2. **Optional Memory Search**
   - Vector search in message history
   - Retrieve top-k relevant past messages
   - Add to context (optional)

3. **LLM Invocation**
   - Simple prompt: "You are a helpful assistant. Answer from your training data."
   - No tool calling
   - No web search
   - Direct streaming response

4. **Response Streaming**
   - Stream text chunks to frontend via SSE
   - No citations needed
   - No sources section

5. **Storage**
   - Save user message to `chat_messages` table
   - Generate embedding vector
   - Save assistant response with embedding
   - Update chat `updated_at` timestamp

### Code Example

```python
# backend/src/chat/service.py

async def answer_simple(self, query: str, chat_history: list, stream: StreamingGenerator):
    """Answer without web sources."""

    # Optional: Memory search for context
    memory_context = await self.memory_service.search(query, limit=5)

    # Build prompt
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        *chat_history,
        HumanMessage(content=query)
    ]

    # Stream response
    async for chunk in self.llm.astream(messages):
        stream.emit_report_chunk(chunk.content)
```

### Configuration

```python
# backend/src/config/settings.py

class Settings:
    chat_model: str = "gpt-3.5-turbo"  # Faster, cheaper model for chat
    chat_temperature: float = 0.7
    chat_max_tokens: int = 500
```

---

## 5. Search/Web Mode Workflow

### Overview

**Purpose**: Quick web search for factual questions requiring verification.

**Architecture**: Perplexica-style two-stage (Researcher → Writer)

**Iterations**: 2 (speed mode)

### Workflow Diagram

```
User Query
    ↓
Classifier → query_type="factual" → suggested_mode="web"
    ↓
SearchService.answer_web_search()
    ↓
┌──────────────────────────────────────────┐
│        STAGE 1: Research Agent          │
│  Iteration 1:                            │
│    1. __reasoning_preamble (optional)    │
│    2. web_search: 1-2 queries            │
│    3. scrape_url: 1-2 URLs               │
│  Iteration 2:                            │
│    4. web_search: verification query     │
│    5. done                                │
│                                           │
│  Output: sources[], scraped_content[]    │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         STAGE 2: Writer Agent            │
│  Input: Research results                 │
│  Process:                                 │
│    1. Read all sources                   │
│    2. Synthesize answer (200-400 words)  │
│    3. Add inline citations [1][2]        │
│    4. Create Sources section             │
│                                           │
│  Output: Cited answer in markdown        │
└──────────────────────────────────────────┘
    ↓
Stream Final Answer to User
    ↓
Save to Database
```

### Stage 1: Research Agent

**Objective**: Gather relevant sources quickly.

**Available Actions**:
- `web_search`: Search the web with queries
- `scrape_url`: Scrape full content from URLs
- `done`: Signal completion

**Typical Flow**:

```
Iteration 1:
  - web_search(queries=["specific factual query"])
    → Returns: search results with titles, URLs, snippets
  - scrape_url(urls=["most_relevant_url"])
    → Returns: full page content

Iteration 2 (if needed):
  - web_search(queries=["verification query"])
  - done(summary="Found sufficient information")
```

**Streaming Events**:
- `search_queries`: Show queries being searched
- `source_found`: Show each source URL discovered
- `status`: Update on current action

### Stage 2: Writer Agent

**Objective**: Synthesize cited answer from research.

**Input**:
- `sources[]`: Search results (title, URL, snippet)
- `scraped_content[]`: Scraped pages (title, URL, full content)

**Process**:

1. **Deduplicate Sources**
   - Remove duplicate URLs
   - Limit to top 10 sources

2. **Build Context**
   - Format sources with [1], [2] numbering
   - Include snippets and scraped content

3. **LLM Synthesis**
   - Structured output: `CitedAnswer`
   - Fields: `reasoning`, `answer`, `citations`, `confidence`
   - Speed mode: 200-400 words

4. **Format Output**
   - Inline citations in answer text
   - Sources section at end with markdown links

**Example Output**:

```markdown
According to recent data [1], the population of Tokyo is approximately 14 million in the city proper and 37 million in the Greater Tokyo Area [2]. This makes it the largest metropolitan area in the world [3].

The city's population density is around 6,000 people per square kilometer [1]. However, population growth has slowed in recent years due to Japan's aging demographic trends [4].

## Sources

[1] [Tokyo Population 2024](https://example.com/tokyo-population)
[2] [Greater Tokyo Statistics](https://example.com/greater-tokyo)
[3] [World's Largest Cities](https://example.com/largest-cities)
[4] [Japan Demographics](https://example.com/japan-demographics)
```

### Configuration

```python
# backend/src/config/settings.py

# Web search mode (speed)
web_search_max_results: int = 5
web_search_queries: int = 2  # Max queries per iteration
web_search_iterations: int = 2
web_scrape_top_n: int = 2  # URLs to scrape
```

---

## 6. Deep Search Mode Workflow

### Overview

**Purpose**: Thorough investigation with multiple iterations and comprehensive coverage.

**Architecture**: Enhanced Perplexica-style (Researcher → Writer)

**Iterations**: 6 (balanced mode)

**Key Differences from Web Mode**:
- More iterations (6 vs 2)
- **Mandatory** reasoning preamble
- Multiple search angles
- More sources scraped
- Longer, more detailed answers

### Workflow Diagram

```
User Query
    ↓
Classifier → suggested_mode="deep"
    ↓
SearchService.answer_deep_search()
    ↓
┌──────────────────────────────────────────┐
│        STAGE 1: Research Agent          │
│  (Balanced Mode - 6 Iterations)          │
│                                           │
│  Iteration 1:                            │
│    1. __reasoning_preamble (MANDATORY)   │
│       "Need to explore X. Gaps: Y."      │
│    2. web_search: 2-3 diverse queries    │
│    3. scrape_url: 2-3 top URLs           │
│                                           │
│  Iteration 2:                            │
│    1. __reasoning_preamble               │
│       "Found A, B. Need C for completeness" │
│    2. web_search: 2-3 new angles         │
│    3. scrape_url: 2-3 URLs               │
│                                           │
│  Iterations 3-5:                         │
│    - Continue exploring gaps             │
│    - Verify important claims             │
│    - Add depth and perspectives          │
│                                           │
│  Iteration 6:                            │
│    1. __reasoning_preamble               │
│       "Comprehensive coverage achieved"  │
│    2. done                                │
│                                           │
│  Output: 6-10 sources, comprehensive data│
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         STAGE 2: Writer Agent            │
│  (Balanced Mode)                         │
│                                           │
│  Input: Research results                 │
│  Process:                                 │
│    1. Synthesize 500-800 word answer     │
│    2. Structure with sections (##, ###)  │
│    3. Add citations for every claim      │
│    4. Include multiple perspectives      │
│    5. Create comprehensive Sources list  │
│                                           │
│  Output: Detailed cited answer           │
└──────────────────────────────────────────┘
    ↓
Stream Final Answer
```

### Reasoning Preamble

**Purpose**: Chain-of-thought reasoning before each action set.

**Format**:

```json
{
  "action": "__reasoning_preamble",
  "args": {
    "reasoning": "Current understanding: I've found X and Y. Gaps identified: still need Z for complete picture. Next action: search for Z from authoritative sources."
  }
}
```

**Benefits**:
- Forces agent to think before acting
- Helps identify research gaps
- Improves action quality
- Provides transparency to users

**Streaming**: Reasoning is streamed to frontend as "agent_reasoning" event

### Research Strategy (Balanced Mode)

**Iteration Planning**:

1. **Iterations 1-2**: Discovery and broad coverage
   - Multiple search angles
   - Scrape overview sources
   - Identify key sub-topics

2. **Iterations 3-4**: Depth and verification
   - Dig deeper into sub-topics
   - Verify key claims from multiple sources
   - Find expert opinions

3. **Iterations 5-6**: Gap filling and completion
   - Address identified gaps
   - Add final perspectives
   - Complete comprehensive coverage

**Example Progression**:

```python
# Query: "How does solar energy work?"

# Iteration 1
reasoning: "Need fundamental understanding of solar energy"
web_search: ["solar energy how it works", "photovoltaic cells explanation"]
scrape_url: ["https://energy.gov/solar-basics", "https://science.org/pv-cells"]

# Iteration 2
reasoning: "Have basics. Need efficiency data and comparisons."
web_search: ["solar panel efficiency 2024", "solar vs other renewables"]
scrape_url: ["efficiency study", "comparison article"]

# Iteration 3
reasoning: "Need practical applications and real-world examples."
web_search: ["solar energy applications", "solar power case studies"]
scrape_url: ["applications overview", "case study"]

# Iteration 4
reasoning: "Need limitations and challenges for balanced view."
web_search: ["solar energy limitations", "solar power challenges"]
scrape_url: ["limitations article"]

# Iteration 5
reasoning: "Need recent developments and future outlook."
web_search: ["solar energy innovations 2024", "solar future trends"]
scrape_url: ["innovations article"]

# Iteration 6
reasoning: "Comprehensive coverage: basics, efficiency, applications, limitations, future. Ready to finish."
done: "Research complete"
```

### Writer Output (Balanced Mode)

**Structure**:

```markdown
## Overview
[Introduction with citations]

## How Solar Energy Works
### Photovoltaic Process
[Detailed explanation with citations]

### Conversion Efficiency
[Data and comparisons with citations]

## Applications
- Residential use [1]
- Commercial installations [2]
- Utility-scale solar farms [3]

## Advantages and Limitations
### Benefits
[Points with citations]

### Challenges
[Points with citations]

## Recent Developments
[2024 updates with citations]

## Conclusion
[Synthesis]

## Sources
[1-10 numbered sources]
```

**Length**: 500-800 words

**Quality Criteria**:
- ✅ Multiple sections with subsections
- ✅ 6-10 cited sources
- ✅ Both benefits and limitations
- ✅ Recent developments included
- ✅ Expert perspectives if applicable

### Configuration

```python
# Deep search mode (balanced)
deep_search_queries: int = 3  # Per iteration
deep_search_iterations: int = 6
deep_scrape_top_n: int = 4  # URLs to scrape per iteration
deep_rerank_top_k: int = 6  # Reranking parameter
```

---

## 7. Deep Research Mode Workflow

### Overview

**Purpose**: Comprehensive, multi-agent research with supervisor coordination for complex, multi-dimensional questions.

**Architecture**: LangGraph-based multi-agent system

**Agents**: 4+ parallel researchers + 1 supervisor

**Modes**:
- **Speed**: 2-3 agents, fast research
- **Balanced**: 3-4 agents, thorough coverage
- **Quality**: 4+ agents, exhaustive research

### Workflow Diagram

```
User Query
    ↓
Classifier → suggested_mode="research_balanced"
    ↓
ResearchGraph Execution (LangGraph)
    ↓
┌──────────────────────────────────────────┐
│         1. MEMORY SEARCH                 │
│  - Vector search in past conversations   │
│  - Retrieve relevant context (top-6)     │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         2. OPTIONAL: DEEP SEARCH         │
│  - If enabled: run deep search first     │
│  - Provides initial context for planning │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         3. RESEARCH PLANNING             │
│  Supervisor creates plan:                │
│    - Analyzes query complexity           │
│    - Breaks into 3-5 topics (balanced)   │
│    - Each topic = 1 agent assignment     │
│                                           │
│  Example for "AI in healthcare":         │
│    1. AI diagnostic tools accuracy       │
│    2. AI drug discovery applications     │
│    3. AI patient care automation         │
│    4. Healthcare AI regulatory landscape │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         4. SPAWN AGENTS                  │
│  Create agent instance per topic:        │
│    Agent 1 → Topic 1                     │
│    Agent 2 → Topic 2                     │
│    Agent 3 → Topic 3                     │
│    Agent 4 → Topic 4                     │
│                                           │
│  Each agent initialized with:            │
│    - AgenticResearcher instance          │
│    - Personal todo list                  │
│    - Personal note storage               │
│    - Access to shared memory             │
│    - Access to supervisor directives     │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         5. EXECUTE AGENTS (PARALLEL)     │
│                                           │
│  ┌────────────┐  ┌────────────┐         │
│  │  Agent 1   │  │  Agent 2   │         │
│  │  ReAct     │  │  ReAct     │         │
│  │  Loop      │  │  Loop      │         │
│  │  (6 steps) │  │  (6 steps) │         │
│  └────────────┘  └────────────┘         │
│       ↕                 ↕                │
│  ┌────────────┐  ┌────────────┐         │
│  │ Supervisor │  │ Supervisor │         │
│  │  Wakes up  │  │  Wakes up  │         │
│  │ after each │  │ after each │         │
│  │   action   │  │   action   │         │
│  └────────────┘  └────────────┘         │
│       ↕                 ↕                │
│  ┌────────────┐  ┌────────────┐         │
│  │  Updates   │  │  Updates   │         │
│  │   todos    │  │   todos    │         │
│  │  actively  │  │  actively  │         │
│  └────────────┘  └────────────┘         │
│                                           │
│  Shared Memory:                          │
│    - Notes visible to all agents         │
│    - Todo directives from supervisor     │
│    - Findings accumulating               │
│                                           │
│  Supervisor monitors EVERY action:       │
│    - Are agents digging deep enough?     │
│    - Do todos need updating?             │
│    - What gaps exist?                    │
│    - Should create more tasks?           │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         6. SUPERVISOR REACT              │
│  After each agent action:                │
│    1. Evaluate depth of research         │
│    2. Check todo progress                │
│    3. Identify gaps                      │
│    4. Decide intervention:               │
│       - Update agent todos               │
│       - Create new tasks                 │
│       - Write gaps to main.md            │
│       - Continue or finish               │
│                                           │
│  Routing decision:                       │
│    "continue" → execute_agents (loop)    │
│    "replan" → plan_research (new topics) │
│    "compress" → compress_findings (done) │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         7. COMPRESS FINDINGS             │
│  Synthesize all agent findings:          │
│    - Combine notes from all agents       │
│    - Extract key themes                  │
│    - Identify important sources          │
│    - Prepare for final report            │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│         8. GENERATE REPORT               │
│  Create comprehensive markdown report:   │
│    - Executive summary                   │
│    - Findings by topic                   │
│    - Direct quotes from sources          │
│    - Inline citations [1][2]...          │
│    - Sources section (numbered)          │
│    - 1000-2000+ words (quality mode)     │
└──────────────────────────────────────────┘
    ↓
Stream Final Report to User
    ↓
Save to Database
```

### Agent ReAct Loop

Each agent runs a **ReAct (Reasoning + Acting) loop** for up to 6 steps:

```python
for step in range(6):  # max_steps
    # 1. Apply supervisor directives (if any)
    await agent._apply_supervisor_directives(agent_id)

    # 2. Build prompt with current state
    prompt = await agent._build_prompt(
        agent_id=agent_id,
        topic=topic,
        existing_findings=existing_findings,
        last_tool_result=last_tool_result
    )

    # 3. Get LLM decision (structured output)
    response = await llm.ainvoke([
        SystemMessage(content=get_agentic_system_prompt()),
        HumanMessage(content=prompt)
    ])
    # → Returns: {"reasoning": "...", "action": "web_search", "args": {...}}

    # 4. Execute action
    result = await agent._execute_action(action, args, agent_id)

    # 5. Supervisor wakes up and reacts
    supervisor_result = await supervisor.react_step(
        query=topic,
        agent_id=agent_id,
        agent_action=action,
        action_result=result,
        findings=existing_findings
    )
    # Supervisor may update this agent's todos or create new tasks

    # 6. Check for finish
    if action == "finish":
        break
```

### Agent Actions

**Search & Scraping**:
- `web_search`: Multi-query web search
- `scrape_urls`: Scrape full content from URLs
- `scroll_page`: Dynamic page scrolling for lazy-loaded content
- `summarize_content`: Summarize very long scraped content

**Memory & Notes**:
- `write_note`: Create note with direct quotes and citations
- `update_note`: Update existing note
- `read_note`: Read specific note file
- `read_shared_notes`: Read notes from other agents

**Todo Management**:
- `add_todo`: Create new todo items (structured)
- `update_todo`: Modify existing todo
- `complete_todo`: Mark todos as done

**Collaboration**:
- `read_agent_file`: Read another agent's file (todos, notes)
- `read_main`: Read main project file with status

**Completion**:
- `finish`: Signal ready to finish (supervisor may override)

### Supervisor Interventions

The supervisor **actively manages** research depth by:

#### 1. Depth Assessment

After each agent action, supervisor checks:

```
✅ Deep Research Indicators:
- Multiple sources per claim (3-5+)
- Sub-topics explored
- Claims verified from independent sources
- Multiple perspectives (pro, con, neutral)
- Primary sources consulted
- Todos evolving based on discoveries

❌ Shallow Research Indicators:
- Single source per claim
- No sub-topic exploration
- 1-2 searches total
- No verification
- Todos unchanged after multiple actions
- Finishing too quickly
```

#### 2. Todo Modification

**Example: Agent doing shallow research**

```json
// Agent completed "Research X" after only 1 search + 1 scrape
// Supervisor intervenes:

{
  "reasoning": "Agent completed research on X too quickly with insufficient sources. Adding deep-dive tasks.",
  "actions": [
    {
      "action": "update_agent_todo",
      "args": {
        "agent_id": "agent_r0_1",
        "todo_title": "Verify X claim from 4-5 independent sources",
        "reasoning": "Current finding based on single source. Need cross-verification.",
        "objective": "Find 4-5 independent sources confirming or refuting claim about X",
        "expected_output": "List of 4-5 sources with direct quotes",
        "sources_needed": ["academic papers", "official reports", "expert interviews"],
        "priority": "high",
        "status": "pending"
      }
    },
    {
      "action": "update_agent_todo",
      "args": {
        "agent_id": "agent_r0_1",
        "todo_title": "Investigate criticisms and limitations of X",
        "reasoning": "Only benefits covered. Need balanced perspective.",
        "objective": "Find expert critiques and limitations",
        "expected_output": "3-4 criticisms with sources + counter-arguments",
        "sources_needed": ["critique articles", "expert opinions"],
        "priority": "medium",
        "status": "pending"
      }
    }
  ]
}
```

#### 3. Gap Analysis

Supervisor tracks coverage across dimensions:

| Dimension | Check |
|-----------|-------|
| Temporal | Historical context? Current state? Future trends? |
| Perspectives | Advocates? Critics? Neutral analysts? |
| Evidence | Primary sources? Secondary? Data? Expert testimony? |
| Scope | Definitions? Features? Comparisons? Use cases? Limitations? |
| Technical | How it works? Architecture? Specifications? |
| Practical | Real-world applications? Case studies? |
| Contextual | Related topics? Dependencies? Controversies? |

**Gap Documentation**:

```json
{
  "action": "write_to_main",
  "args": {
    "section": "Gaps",
    "content": "## Research Gaps Identified\n\n- ❌ No expert opinions found yet\n- ❌ Historical context missing\n- ⚠️  Only 1 source for key claim (need verification)\n- ✅ Current state well-documented"
  }
}
```

#### 4. Stop Condition Checklist

Supervisor only allows finishing when:

- [ ] All major angles explored (breadth)
- [ ] Each angle investigated with 3-5+ sources (depth)
- [ ] Key claims verified from multiple independent sources
- [ ] Primary sources consulted
- [ ] Sub-topics identified and investigated
- [ ] Gaps addressed or documented
- [ ] Multiple perspectives represented
- [ ] Temporal coverage (history, current, future)
- [ ] Technical depth sufficient
- [ ] Real-world applications documented

**Minimum thresholds**:
- ≥ 15 unique sources across all agents
- ≥ 5 major dimensions covered
- ≥ 3 independent confirmations for key claims
- ≥ 8 completed todos per agent

### Shared Memory

**Purpose**: Enable agent collaboration and information sharing.

**Components**:

1. **Shared Notes**
   - Agents write high-signal notes with `share=true`
   - Visible to all other agents via `read_shared_notes`
   - Prevents duplicate work
   - Enables building on each other's findings

2. **Todo Directives**
   - Supervisor creates directives for specific agents
   - Queued per agent ID
   - Applied at start of each agent iteration
   - Allows supervisor to dynamically modify research plans

3. **Main File (`main.md`)**
   - Central knowledge repository
   - Project status section (updated by supervisor)
   - Contains: query, findings count, agent statuses
   - Agents can read via `read_main` action

4. **Agent Files** (persistent)
   - Each agent has a JSON file: `agent_r0_1.json`
   - Contains: todos, notes, character, preferences
   - Survives across iterations
   - Can be read by other agents or supervisor

### Quality Metrics (Deep Research)

**Excellence indicators**:

- 📚 **Sources**: 15-20+ unique authoritative sources
- 🔍 **Verification**: Key claims verified from 3-5+ independent sources
- 👥 **Perspectives**: Multiple expert viewpoints (academic, industry, critical)
- 📊 **Data**: Both quantitative data and qualitative insights
- ⚖️ **Balance**: Pros, cons, and neutral perspectives
- 🏛️ **Primary sources**: Official docs, research papers, archives
- 🌐 **Diversity**: Academic, industry, news, technical sources
- 📝 **Quotes**: 10-15+ direct quotes with proper attribution
- 🌳 **Sub-topics**: 2-4 sub-topics explored per main topic
- ⏱️ **Timeline**: Historical development documented

### Configuration

```python
# Deep research mode
deep_research_num_agents: int = 4  # Balanced mode
deep_research_max_rounds: int = 3  # Supervisor rounds
deep_research_max_concurrent: int = 4  # Parallel agents
deep_research_max_sources: int = 10  # Per agent
deep_research_agent_max_steps: int = 6  # Per agent ReAct loop

# Enable/disable features
enable_clarifying_questions: bool = False
run_deep_search_first: bool = False  # Run deep search before planning
```

---

## 8. Component Details

### Query Classifier

**File**: `backend/src/workflow/search/classifier.py`

**Purpose**: Intelligent routing of queries to appropriate modes.

**Process**:

1. **Input**: User query + chat history
2. **Analysis**: LLM analyzes with structured output
3. **Output**: `QueryClassification` object

**Fields**:
- `reasoning`: Why this classification
- `query_type`: simple | factual | research | opinion | comparison | news
- `standalone_query`: Rewritten query without context dependencies
- `suggested_mode`: chat | web | deep | research_speed | research_balanced | research_quality
- `requires_sources`: Boolean
- `time_sensitive`: Boolean

**Example**:

```python
# Query: "Who founded it?"
# History: "Tell me about Tesla"

classification = await classify_query(
    query="Who founded it?",
    chat_history=[{"role": "user", "content": "Tell me about Tesla"}],
    llm=llm
)

# Output:
{
  "reasoning": "Pronoun 'it' refers to Tesla from previous message. Factual question needing verification.",
  "query_type": "factual",
  "standalone_query": "Who founded Tesla?",
  "suggested_mode": "web",
  "requires_sources": true,
  "time_sensitive": false
}
```

### Search Provider

**File**: `backend/src/search/base.py`

**Implementation**: SearxNG or Brave Search

**Interface**:

```python
class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        pass

class SearchResponse:
    results: list[SearchResult]

class SearchResult:
    title: str
    url: str
    content: str  # snippet
    score: float
```

**Configuration**:

```python
# backend/src/config/settings.py

searxng_url: str = "http://localhost:8080"
search_blocked_domains: str = ""  # Comma-separated blocklist
search_blocked_keywords: str = ""  # Quality filtering
```

### Web Scraper

**File**: `backend/src/search/scraper.py`

**Technologies**:
- Playwright (for JS-rendered pages)
- BeautifulSoup (for HTML parsing)
- Fallback to Jina.ai reader API

**Interface**:

```python
class WebScraper:
    async def scrape(self, url: str, scroll: bool = False) -> ScrapedContent:
        """Scrape URL content."""
        pass

class ScrapedContent:
    url: str
    title: str
    content: str  # Cleaned text
    html: str | None
```

**Features**:
- JavaScript execution
- Dynamic scrolling for lazy-loaded content
- Content cleaning (remove ads, navigation, etc.)
- Timeout handling
- Fallback mechanisms

### Memory Service

**File**: `backend/src/memory/vector_memory.py`

**Purpose**: Vector storage and semantic search over past conversations.

**Storage**: PostgreSQL with pgvector extension

**Schema**:

```sql
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    chat_id VARCHAR(64) NOT NULL,
    message_id VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL,  -- 'user' | 'assistant'
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI embedding
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_chat_messages_embedding
ON chat_messages USING ivfflat (embedding vector_cosine_ops);
```

**Interface**:

```python
class VectorMemoryService:
    async def add_message(self, chat_id: str, role: str, content: str, embedding: list[float]):
        """Store message with embedding."""

    async def search(self, query: str, limit: int = 5, chat_id: str | None = None) -> list[Message]:
        """Semantic search in message history."""

    async def hybrid_search(self, query: str, limit: int = 5) -> list[Message]:
        """Hybrid vector + fulltext search."""
```

**Hybrid Search** (RRF - Reciprocal Rank Fusion):

```python
# Combine vector similarity and full-text search
vector_results = await vector_search(query_embedding, limit=20)
fts_results = await fulltext_search(query, limit=20)

# RRF scoring
final_results = reciprocal_rank_fusion(
    [vector_results, fts_results],
    k=60  # RRF constant
)[:limit]
```

### Streaming

**Protocol**: Server-Sent Events (SSE)

**Event Types**:

```python
# Status updates
stream.emit_status("Searching web...", step="search")

# Search queries
stream.emit_search_queries(["query1", "query2"])

# Sources found
stream.emit_source(agent_id, {"url": "...", "title": "..."})

# Agent reasoning
stream.emit_agent_reasoning(agent_id, "I need to verify...")

# Agent todos
stream.emit_agent_todo(agent_id, [{"title": "...", "status": "pending"}, ...])

# Findings
stream.emit_finding(agent_id, topic, summary, key_findings)

# Report chunks (final answer streaming)
stream.emit_report_chunk("partial text...")

# Planning
stream.emit_research_plan(plan_text, topics)

# Completion
stream.emit_done()
```

**Frontend Handling**:

```typescript
const response = await fetch('/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ messages, mode, stream: true })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));

      switch (event.type) {
        case 'report_chunk':
          appendToAnswer(event.data.content);
          break;
        case 'source_found':
          addSource(event.data);
          break;
        case 'agent_todo':
          updateAgentTodos(event.data);
          break;
        // ... handle other events
      }
    }
  }
}
```

---

## 9. Prompt Engineering

### Improved Prompts

All improved prompts are located in:
- `backend/src/workflow/legacy/agentic/prompts_improved.py` (Deep Research)
- `backend/src/workflow/search/prompts_improved.py` (Search modes)

### Key Improvements

**1. Supervisor Prompt**:
- ✅ Explicit depth assessment criteria
- ✅ Concrete intervention examples
- ✅ Gap analysis framework with table
- ✅ Stop condition checklist with metrics
- ✅ Todo update examples in JSON
- ✅ Quality thresholds (15+ sources, 8+ todos/agent)

**2. Agentic Researcher Prompt**:
- ✅ Comprehensive workflow examples
- ✅ Quote collection guidelines (3-5 quotes per finding)
- ✅ Tool usage best practices
- ✅ Todo management examples
- ✅ Deep research checklist
- ✅ Success metrics and warning signs

**3. Perplexica Researcher Prompt**:
- ✅ Mode-specific strategies (speed/balanced/quality)
- ✅ Iteration planning guides
- ✅ Research dimension checklist
- ✅ Reasoning preamble templates
- ✅ Example progressions for each mode

**4. Writer Prompt**:
- ✅ Citation format examples
- ✅ Structure templates
- ✅ Mode-specific length guidelines
- ✅ Quality criteria per mode
- ✅ Bad vs good examples

**5. Classifier Prompt**:
- ✅ Decision guidelines per mode
- ✅ Query type examples
- ✅ Standalone rewriting examples
- ✅ Time sensitivity detection

### Prompt Usage

To use improved prompts, update imports:

```python
# Old
from src.workflow.legacy.agentic.coordinator import get_supervisor_system_prompt

# New (improved)
from src.workflow.legacy.agentic.prompts_improved import get_supervisor_system_prompt_improved
```

---

## 10. Configuration & Tuning

### Settings File

**Location**: `backend/src/config/settings.py`

### Mode Settings

```python
class Settings:
    # Default mode
    default_mode: str = "search"

    # Speed mode (web search)
    speed_max_iterations: int = 2
    speed_max_concurrent: int = 1

    # Balanced mode (deep search)
    balanced_max_iterations: int = 6
    balanced_max_concurrent: int = 3

    # Quality mode (deep research)
    quality_max_iterations: int = 25
    quality_max_concurrent: int = 4
```

### Search Tuning

```python
# Web search (speed mode)
web_search_queries: int = 2
web_search_iterations: int = 2
web_scrape_top_n: int = 2
web_rerank_top_k: int = 6

# Deep search (balanced mode)
deep_search_queries: int = 3
deep_search_iterations: int = 6
deep_scrape_top_n: int = 4
deep_rerank_top_k: int = 6

# Quality search
deep_search_quality_queries: int = 3
deep_search_quality_iterations: int = 25
deep_search_quality_scrape_top_n: int = 6
```

### Deep Research Tuning

```python
# Agent configuration
deep_research_num_agents: int = 4
deep_research_max_rounds: int = 3  # Supervisor rounds
deep_research_max_concurrent: int = 4
deep_research_max_sources: int = 10  # Per agent

# Features
enable_clarifying_questions: bool = False
run_deep_search_first: bool = False

# Memory
memory_search_limit: int = 6
memory_enabled: bool = True
```

### LLM Configuration

```python
# Model selection
chat_model: str = "gpt-3.5-turbo"
research_model: str = "gpt-4"
embedding_model: str = "text-embedding-ada-002"

# Temperature
chat_temperature: float = 0.7
research_temperature: float = 0.3  # More focused
classifier_temperature: float = 0.1  # Very deterministic

# Tokens
chat_max_tokens: int = 500
research_max_tokens: int = 2000
```

### Performance Tuning

**Trade-offs**:

| Setting | ↑ Increase | ↓ Decrease |
|---------|-----------|------------|
| `iterations` | More thorough, slower | Faster, less thorough |
| `num_agents` | More parallel coverage, expensive | Cheaper, sequential |
| `max_sources` | More sources per agent | Less context per agent |
| `max_rounds` | Supervisor has more chances to intervene | Faster, less supervision |
| `scrape_top_n` | More detailed content | Faster searches |

**Recommended Profiles**:

```python
# Fast & Cheap
speed_iterations = 2
num_agents = 2
chat_model = "gpt-3.5-turbo"

# Balanced
balanced_iterations = 6
num_agents = 4
research_model = "gpt-4"

# Best Quality
quality_iterations = 25
num_agents = 6
research_model = "gpt-4"
enable_clarifying_questions = True
```

---

## 11. Best Practices

### For Users

**Crafting Effective Queries**:

1. **Be Specific**: "How does CRISPR gene editing work in treating cancer?" vs "Tell me about CRISPR"

2. **Indicate Depth**: Use keywords like:
   - "quick overview" → triggers speed mode
   - "detailed analysis" → triggers balanced mode
   - "comprehensive research" → triggers quality mode

3. **Provide Context**: Reference specific aspects, timeframes, or dimensions

4. **Follow-up Questions**: Use chat history for context (classifier resolves pronouns)

### For Developers

**Prompt Engineering**:

1. **Structured Outputs**: Always use `llm.with_structured_output(Schema, method="function_calling")`
   - More reliable than JSON parsing
   - Automatic validation
   - Better error messages

2. **System Prompts**: Keep separate from user prompts
   - Easier to version and test
   - Better prompt injection protection

3. **Examples in Prompts**: Include good/bad examples
   - Few-shot learning improves quality
   - Clarifies expectations

**Error Handling**:

```python
try:
    result = await llm.ainvoke(messages)
except Exception as e:
    logger.error("LLM invocation failed", error=str(e))
    # Fallback strategy
    return fallback_response()
```

**Logging**:

```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "Research completed",
    mode=mode,
    iterations=iteration_count,
    sources=len(sources),
    agents=len(agents)
)
```

**Testing**:

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test workflows end-to-end
3. **Prompt Testing**: Version prompts, track performance
4. **Mock LLM**: Use `llm_type="mock-chat"` for fast testing

### Performance Optimization

**Caching**:
- Cache search results (24hr TTL)
- Cache scraped content (1hr TTL)
- Cache embeddings (permanent)

**Parallel Execution**:
```python
# Execute multiple scrapes in parallel
results = await asyncio.gather(
    *[scraper.scrape(url) for url in urls],
    return_exceptions=True
)
```

**Database Optimization**:
- Use connection pooling (asyncpg)
- Create proper indexes
- Use batched inserts

**Streaming**:
- Stream immediately, don't buffer
- Use `async for` for streaming
- Close streams properly

---

## 12. Troubleshooting

### Common Issues

#### 1. "No sources found"

**Symptoms**: Empty search results or "I couldn't find enough information"

**Causes**:
- Blocked domains too restrictive
- Search provider down
- Query too specific/niche

**Solutions**:
```python
# Check search provider
await search_provider.search("test query")

# Review blocklists
settings.search_blocked_domains = ""
settings.search_blocked_keywords = ""

# Try broader query
classification.standalone_query = "broader version of query"
```

#### 2. "Agent loops without finishing"

**Symptoms**: Agent doesn't call `finish` action, uses all iterations

**Causes**:
- Supervisor keeps adding todos
- Agent doesn't meet finish criteria
- Stop condition too strict

**Solutions**:
```python
# Reduce supervisor intervention
deep_research_max_rounds = 2  # Instead of 3

# Relax finish criteria in prompt
# Or add max iteration hard stop
```

#### 3. "Slow response times"

**Symptoms**: Takes >2 minutes for deep research

**Causes**:
- Too many agents
- Too many iterations
- Slow scraping

**Solutions**:
```python
# Reduce parallelism
deep_research_max_concurrent = 2  # Instead of 4

# Reduce iterations
balanced_max_iterations = 4  # Instead of 6

# Skip scraping for some queries
# Or use faster model for research
research_model = "gpt-3.5-turbo"
```

#### 4. "Citations missing or incorrect"

**Symptoms**: Answer lacks [1][2] citations or links broken

**Causes**:
- Writer prompt not emphasizing citations
- Sources not properly formatted
- Structured output parsing failed

**Solutions**:
```python
# Verify writer prompt has citation rules
# Check CitedAnswer schema validation
# Review source formatting in context
```

#### 5. "Database errors"

**Symptoms**: `asyncpg.exceptions.UndefinedTable` or similar

**Causes**:
- Migrations not run
- PostgreSQL not configured
- Wrong database URL

**Solutions**:
```bash
# Run migrations
cd backend
alembic upgrade head

# Check database
docker-compose ps postgres
docker-compose logs postgres

# Verify connection string
echo $DATABASE_URL
```

### Debug Mode

Enable debug logging:

```python
# backend/src/config/settings.py
log_level: str = "DEBUG"

# Or environment variable
export LOG_LEVEL=DEBUG
```

View detailed logs:

```bash
# Backend logs
docker-compose logs -f backend

# Database queries
docker-compose logs postgres | grep "LOG:"
```

### Monitoring

**Key Metrics**:
- Query classification accuracy
- Average response times per mode
- Source count per query
- Agent iteration counts
- Supervisor intervention frequency
- Citation coverage (% of facts cited)

**Logging Queries**:

```python
logger.info(
    "Query processed",
    mode=mode,
    classification=classification.query_type,
    response_time_s=elapsed,
    sources_count=len(sources),
    tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None
)
```

---

## Appendix A: File Structure

```
all_included_deep_research/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── chat.py          # Main chat endpoint
│   │   │   │   └── chats.py         # CRUD + search endpoints
│   │   │   └── app.py                # FastAPI app initialization
│   │   ├── chat/
│   │   │   ├── service.py            # Chat service (legacy workflow)
│   │   │   └── search.py             # Message search engine
│   │   ├── workflow/
│   │   │   ├── search/               # Perplexica-style workflows
│   │   │   │   ├── classifier.py     # Query classifier
│   │   │   │   ├── researcher.py     # Research agent
│   │   │   │   ├── writer.py         # Writer agent
│   │   │   │   ├── actions.py        # Action registry
│   │   │   │   ├── service.py        # Search service orchestrator
│   │   │   │   └── prompts_improved.py  # ✨ IMPROVED PROMPTS
│   │   │   ├── research/             # LangGraph research workflows
│   │   │   │   ├── graph.py          # Research graph definition
│   │   │   │   ├── state.py          # State schemas
│   │   │   │   └── nodes/
│   │   │   │       ├── planner.py    # Planning node
│   │   │   │       ├── researcher.py # Researcher node
│   │   │   │       └── reporter.py   # Reporting node
│   │   │   └── legacy/
│   │   │       ├── agentic/
│   │   │       │   ├── coordinator.py    # Supervisor/coordinator
│   │   │       │   ├── researcher.py     # Agentic researcher
│   │   │       │   ├── schemas.py        # Structured schemas
│   │   │       │   ├── models.py         # Agent memory models
│   │   │       │   └── prompts_improved.py  # ✨ IMPROVED PROMPTS
│   │   │       └── nodes/
│   │   │           ├── planner.py        # Legacy planner
│   │   │           └── memory_search.py  # Memory search node
│   │   ├── search/
│   │   │   ├── base.py               # Search provider interface
│   │   │   ├── searxng.py            # SearxNG implementation
│   │   │   └── scraper.py            # Web scraper
│   │   ├── memory/
│   │   │   ├── vector_memory.py      # Vector memory service
│   │   │   ├── agent_memory_service.py
│   │   │   └── agent_file_service.py
│   │   ├── database/
│   │   │   └── schema.py             # SQLAlchemy models
│   │   └── config/
│   │       └── settings.py           # Configuration
│   └── alembic/
│       └── versions/
│           ├── 001_*.py              # Initial migration
│           └── 002_add_chat_message_embedding.py
├── frontend/
│   └── src/
│       ├── app/
│       │   └── page.tsx              # Main chat UI
│       ├── components/
│       │   ├── ChatSearch.tsx        # Search component
│       │   └── ChatSidebar.tsx       # Sidebar with chat list
│       └── lib/
│           └── api.ts                # API client functions
├── docker-compose.yml
├── CHAT_SEARCH_IMPLEMENTATION.md    # Chat search docs
├── FRONTEND_CHAT_SEARCH.md           # Frontend search docs
└── WORKFLOW_DOCUMENTATION.md         # ✨ THIS FILE
```

---

## Appendix B: API Reference

### Main Chat Endpoint

**Endpoint**: `POST /v1/chat/completions`

**Request**:
```json
{
  "messages": [
    {"role": "user", "content": "How does solar energy work?"}
  ],
  "mode": "deep_search",
  "stream": true
}
```

**Response**: SSE stream of events

**Events**:
- `init`: Session initialized
- `status`: Status update
- `search_queries`: Queries being searched
- `source_found`: Source discovered
- `agent_todo`: Agent todo list update
- `agent_note`: Agent note created
- `finding`: Research finding
- `report_chunk`: Partial answer text
- `final_report`: Complete answer
- `done`: Stream complete

### Chat CRUD Endpoints

- `GET /api/chats` - List all chats
- `POST /api/chats` - Create new chat
- `GET /api/chats/{chat_id}` - Get chat with messages
- `DELETE /api/chats/{chat_id}` - Delete chat
- `POST /api/chats/{chat_id}/messages` - Add message to chat

### Chat Search Endpoint

**Endpoint**: `GET /api/chats/search?q={query}&limit={limit}`

**Purpose**: Hybrid search in chat messages (vector + fulltext)

**Example**:
```bash
curl "http://localhost:8000/api/chats/search?q=solar%20energy&limit=5"
```

**Response**:
```json
{
  "messages": [
    {
      "message_id": 123,
      "chat_id": "abc123",
      "role": "assistant",
      "content": "Solar energy works by...",
      "created_at": "2024-01-15T10:30:00Z",
      "chat_title": "Solar Energy Discussion",
      "score": 0.85,
      "search_mode": "hybrid"
    }
  ],
  "count": 5
}
```

---

## Appendix C: Glossary

- **Agent**: Autonomous AI entity that performs research actions
- **Agentic**: Agent-based architecture with tool-calling loops
- **Citation**: Inline reference [1] linking to source
- **Classifier**: Component that routes queries to modes
- **Deep Research**: Comprehensive multi-agent research mode
- **Deep Search**: Iterative single-agent search mode (6 iterations)
- **Embedding**: Vector representation of text (1536 dimensions)
- **Hybrid Search**: Combination of vector similarity and full-text search
- **IVFFlat**: Vector index type for approximate nearest neighbor search
- **LangChain**: Framework for LLM applications
- **LangGraph**: Framework for multi-agent workflows with graphs
- **Memory Search**: Semantic search in conversation history
- **Mode**: Operating mode (chat, web, deep_search, deep_research)
- **Perplexica**: Two-stage architecture (researcher → writer)
- **pgvector**: PostgreSQL extension for vector operations
- **ReAct**: Reasoning + Acting loop pattern
- **RRF**: Reciprocal Rank Fusion (search ranking algorithm)
- **Scraping**: Extracting content from web pages
- **SSE**: Server-Sent Events (streaming protocol)
- **Structured Output**: LLM response with enforced schema
- **Supervisor**: Coordinator agent that manages other agents
- **Todo**: Task item in agent's personal checklist
- **Vector**: Numerical representation for semantic similarity
- **Web Search**: Mode with quick web search (2 iterations)

---

**End of Documentation**

For updates and questions, refer to project repository.
