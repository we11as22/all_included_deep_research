# Legacy Code Cleanup Report

**Date**: 2026-01-02
**Project**: All-Included Deep Research

---

## Executive Summary

The `backend/src/workflow/legacy/` directory contains **6,477 lines** of deprecated code that is no longer actively used. The current system uses a new LangGraph-based workflow for deep research mode, and only 5 Pydantic schemas from legacy are actively used.

**Actions Taken**:
1. ✅ Extracted 5 active schemas to `src/models/schemas.py`
2. ✅ Updated imports in `chat/service.py`
3. ✅ Identified deprecated `/api/research` endpoint
4. ✅ Verified frontend only uses `/api/chat/stream`

**Recommendations**:
- 🗑️ Delete entire `legacy/` directory (6,477 lines)
- 🗑️ Delete `ResearchStream.tsx` component (unused)
- 🗑️ Delete `streamResearch()` function from `api.ts`
- 🗑️ Delete `/api/research` endpoint routes
- 🗑️ Delete `memory/agent_memory_service.py` and `memory/agent_file_service.py` (only used by legacy)

---

## Active vs Legacy Code Analysis

### ✅ ACTIVE Code (Currently Used)

**Main Workflow** (deep_research mode):
- Location: `src/workflow/research/` (NEW LangGraph implementation)
- Entry: `/api/chat/stream` endpoint
- File: `backend/src/api/routes/chat_stream.py`
- Function: `run_research_graph()` from `src.workflow.research`

**Search Modes** (search, deep_search):
- Service: `ChatSearchService` (`src/chat/service.py`)
- Uses: 5 schemas from `src/models/schemas.py` (extracted from legacy)

**Frontend**:
- Main UI: `frontend/src/app/page.tsx`
- API Client: `streamChatProgress()` from `frontend/src/lib/api.ts`
- Endpoint: `POST /api/chat/stream`

### ❌ DEPRECATED Code (Not Used)

**Legacy Workflows** (0 active references):
- `SpeedResearchWorkflow` - 0 imports outside legacy
- `BalancedResearchWorkflow` - 0 imports outside legacy
- `QualityResearchWorkflow` - 0 imports outside legacy
- `WorkflowFactory` - Referenced in `/api/research` but never initialized in `app.state`

**Legacy Agentic System** (0 active references):
- `AgenticResearchCoordinator` - 0 imports outside legacy
- `AgenticResearcher` - 0 imports outside legacy
- `AgenticSupervisor` - 0 imports outside legacy
- Improved prompts file - not imported anywhere

**Legacy Nodes** (0 active references):
- `memory_search.py` - only used internally by legacy workflows
- `planner.py` - only used internally by legacy workflows
- `reporter.py` - only used internally by legacy workflows
- `researcher.py` - only used internally by legacy workflows
- `supervisor.py` - only used internally by legacy workflows

**Legacy Memory Services** (only used by legacy):
- `memory/agent_memory_service.py` - imports from legacy, only referenced in `/api/research`
- `memory/agent_file_service.py` - imports from legacy, only referenced in `/api/research`
- `memory/agent_session.py` - creates above services, only used in `/api/research`

**Deprecated API Endpoint** (not used by frontend):
- `POST /api/research` - References `workflow_factory` which is never initialized
- Would fail with `AttributeError: 'State' object has no attribute 'workflow_factory'`
- Frontend does NOT call this endpoint

**Unused Frontend Component**:
- `ResearchStream.tsx` - Not imported in any parent component
- Uses `streamResearch()` which calls deprecated `/api/research`

---

## Import Analysis

### Fixed Imports

**Before**:
```python
# backend/src/chat/service.py:26
from src.workflow.legacy.agentic.schemas import QueryRewrite, SearchQueries, ...
```

**After**:
```python
# backend/src/chat/service.py:26
from src.models.schemas import QueryRewrite, SearchQueries, ...
```

### Broken Imports (within legacy only)

Legacy code has internal broken imports referencing non-existent `src.workflow.agentic.*`:

```python
# These imports fail but only within legacy code:
from src.workflow.agentic.coordinator import ...  # Should be src.workflow.legacy.agentic.coordinator
from src.workflow.agentic.researcher import ...   # Should be src.workflow.legacy.agentic.researcher
from src.workflow.agentic.models import ...       # Should be src.workflow.legacy.agentic.models
from src.workflow.agentic.schemas import ...      # Should be src.workflow.legacy.agentic.schemas
```

These imports are only within legacy files, so deleting legacy/ resolves them.

---

## Frontend Integration Verification

### ✅ Active Frontend Flow

```typescript
// frontend/src/app/page.tsx
import { streamChatProgress } from '@/lib/api';

// User sends message with mode selection
for await (const event of streamChatProgress({
  messages: [...],
  mode: 'deep_research'  // or 'search', 'deep_search', 'chat'
})) {
  // Process streaming events
}
```

```typescript
// frontend/src/lib/api.ts
export async function* streamChatProgress(params) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    body: JSON.stringify({
      messages: params.messages,
      mode: params.mode,
      stream: true
    })
  });
  // ... SSE parsing
}
```

### ❌ Unused Frontend Code

```typescript
// frontend/src/lib/api.ts:401
export async function* streamResearch(params) {
  const response = await fetch('/api/research', {  // ← DEPRECATED ENDPOINT
    method: 'POST',
    // ...
  });
}
```

```typescript
// frontend/src/components/ResearchStream.tsx
// This component is NEVER imported in any parent component
// Uses streamResearch() which calls deprecated endpoint
```

**Search Results**:
- ❌ `streamResearch` imported: 1 file (only ResearchStream.tsx)
- ❌ `ResearchStream` imported: 0 files

---

## Backend API Routing

### ✅ Active API Endpoints

**Main Chat Streaming**:
```python
# backend/src/api/routes/chat_stream.py

@router.post("/api/chat/stream")
async def stream_chat(request: ChatStreamRequest, app_request: Request):
    mode = request.mode  # 'chat', 'search', 'deep_search', 'deep_research'

    if mode == "deep_research":
        # Uses NEW LangGraph workflow
        from src.workflow.research import run_research_graph
        final_state = await run_research_graph(...)
    elif mode in ("search", "deep_search"):
        # Uses ChatSearchService
        chat_service = app_request.app.state.chat_service
        result = await chat_service.answer_web(...) or answer_deep(...)
    else:
        # Chat mode
        ...
```

**Chat Management**:
```python
# backend/src/api/routes/chats.py

@router.get("/api/chats")  # List chats
@router.post("/api/chats")  # Create chat
@router.get("/api/chats/{chat_id}")  # Get chat with messages
@router.delete("/api/chats/{chat_id}")  # Delete chat
@router.get("/api/chats/search")  # Hybrid search (vector + fulltext)
@router.post("/api/chats/{chat_id}/messages")  # Add message
```

### ❌ Deprecated API Endpoint

```python
# backend/src/api/routes/research.py

@router.post("/api/research")  # ← NOT USED BY FRONTEND
async def start_research(research_request: ResearchRequest, app_request: Request):
    # Line 57: This fails because workflow_factory is never initialized
    workflow_factory = app_request.app.state.workflow_factory  # ← AttributeError
    workflow = workflow_factory.create_workflow(research_request.mode.value)
```

**Why it's broken**:
```python
# backend/src/api/app.py - app.state initialization
# workflow_factory is NEVER created or assigned to app.state
# This endpoint would fail if called
```

---

## Extracted Schemas

**New File**: `backend/src/models/schemas.py`

Extracted from `backend/src/workflow/legacy/agentic/schemas.py`:

1. **QueryRewrite** - Used for query rewriting in search pipeline
2. **SearchQueries** - Used for multi-query generation
3. **FollowupQueries** - Used for iterative search gap filling
4. **SummarizedContent** - Used for content summarization
5. **SynthesizedAnswer** - Used for answer synthesis

These are the ONLY schemas from legacy that are actively used (by `ChatSearchService`).

---

## File Size Analysis

**Legacy Directory**:
```
backend/src/workflow/legacy/
├── agentic/
│   ├── coordinator.py         44,531 bytes
│   ├── researcher.py          62,127 bytes
│   ├── supervisor.py           1,532 bytes
│   ├── prompts_improved.py    36,128 bytes
│   ├── schemas.py             13,412 bytes
│   ├── models.py               5,537 bytes
│   └── __init__.py               246 bytes
├── nodes/
│   ├── memory_search.py        2,891 bytes
│   ├── planner.py              8,142 bytes
│   ├── reporter.py             6,489 bytes
│   ├── researcher.py          16,234 bytes
│   └── supervisor.py          11,052 bytes
├── tools/
│   ├── conduct_research.py     3,156 bytes
│   ├── memory_tools.py         2,847 bytes
│   ├── research_complete.py    1,923 bytes
│   └── think.py                  876 bytes
├── balanced_research.py       21,334 bytes
├── quality_research.py        18,956 bytes
├── speed_research.py          15,432 bytes
├── factory.py                  2,145 bytes
├── state.py                    4,892 bytes
└── __init__.py                   312 bytes

Total: ~279KB, ~6,477 lines of code
```

**Memory Services** (only used by legacy):
```
backend/src/memory/
├── agent_memory_service.py    15,234 bytes
├── agent_file_service.py      12,456 bytes
└── agent_session.py            3,891 bytes

Total: ~31KB
```

**Deprecated Frontend**:
```
frontend/src/components/ResearchStream.tsx    8,234 bytes
```

**Total Legacy Code**: ~318KB, ~7,500+ lines

---

## Cleanup Plan

### Step 1: Delete Legacy Backend Code

```bash
# Delete legacy workflow directory
rm -rf backend/src/workflow/legacy/

# Delete legacy memory services (only used by legacy)
rm backend/src/memory/agent_memory_service.py
rm backend/src/memory/agent_file_service.py
rm backend/src/memory/agent_session.py

# Delete deprecated research endpoint
rm backend/src/api/routes/research.py

# Update API app.py to remove research routes
# (Remove: from src.api.routes import research)
# (Remove: app.include_router(research.router))
```

### Step 2: Delete Legacy Frontend Code

```bash
# Delete unused research stream component
rm frontend/src/components/ResearchStream.tsx

# Update api.ts to remove streamResearch function
# (Remove lines 401-478 in frontend/src/lib/api.ts)
```

### Step 3: Cleanup Imports

```bash
# No imports to fix - already updated chat/service.py
# All other legacy imports are within legacy code itself
```

### Step 4: Verification

```bash
# Check no remaining legacy imports outside legacy
cd backend/src
grep -r "from src\.workflow\.legacy" --include="*.py" . | grep -v "legacy/"
# Should return: no results

grep -r "from src\.workflow\.agentic" --include="*.py" . | grep -v "legacy/"
# Should return: no results

grep -r "agent_memory_service" --include="*.py" . | grep -v "legacy/"
# Should return: no results

# Frontend verification
cd frontend/src
grep -r "streamResearch" .
# Should return: no results after cleanup

grep -r "/api/research" .
# Should return: no results after cleanup
```

---

## Migration Completeness

### ✅ Fully Migrated

- **Deep Research Mode**: Uses new LangGraph workflow (`src/workflow/research/`)
- **Search Modes**: Use `ChatSearchService` with extracted schemas
- **Frontend**: Uses `streamChatProgress()` → `/api/chat/stream`
- **Chat Storage**: PostgreSQL with embeddings and hybrid search
- **Memory**: Vector memory service (non-agent-specific)

### ❌ Not Needed (Can Delete)

- Legacy workflows (3 files)
- Legacy agentic system (coordinator, researcher, supervisor)
- Legacy nodes (5 files)
- Legacy tools (4 files)
- Legacy memory services (agent_memory_service, agent_file_service)
- Deprecated `/api/research` endpoint
- Unused `ResearchStream` component

---

## Benefits of Cleanup

1. **Code Clarity**: Remove 7,500+ lines of unused code
2. **Reduced Confusion**: No more broken imports or duplicate workflows
3. **Faster Development**: Clearer codebase structure
4. **Less Maintenance**: Don't maintain deprecated code
5. **Smaller Build**: ~318KB less code to package
6. **Better Docs**: Documentation matches actual code

---

## Risks & Mitigation

**Risk**: Accidentally deleting used code
**Mitigation**:
- ✅ Verified all imports outside legacy
- ✅ Verified frontend doesn't call `/api/research`
- ✅ Verified new workflow is production-ready
- ✅ Extracted all active schemas to safe location

**Risk**: Breaking existing functionality
**Mitigation**:
- ✅ Frontend only uses `/api/chat/stream` (verified)
- ✅ Deep research uses new LangGraph workflow (verified)
- ✅ All active schemas extracted to `src/models/` (complete)

**Recommendation**:
- Create git branch before deletion
- Test all modes after cleanup (chat, search, deep_search, deep_research)
- Verify frontend can send messages and receive streams

---

## Summary

**Status**: ✅ Ready for Cleanup

The legacy code is fully isolated and unused. All active functionality has been migrated to:
- New LangGraph workflow for deep research
- ChatSearchService for search modes
- Extracted schemas in `src/models/schemas.py`

**Next Action**: Execute cleanup plan and verify all modes still work.

---

**End of Report**
