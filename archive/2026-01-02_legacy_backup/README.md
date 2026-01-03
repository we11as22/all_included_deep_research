# Legacy Code Archive - 2026-01-02

This directory contains archived legacy code that was replaced by the new LangGraph-based workflow system.

## What's Archived

### 1. Legacy Workflow System (`legacy/`)
- **Speed/Balanced/Quality Research Workflows** - Old multi-agent research implementation
- **Agentic Coordinator & Researcher** - Supervisor-based agentic research system
- **Legacy Nodes** - Planner, Researcher, Reporter, Supervisor nodes
- **Legacy Tools** - Research tools (conduct_research, memory_tools, etc.)
- **Legacy State** - Old state definitions

### 2. Agent Memory Services
- `agent_memory_service.py` - Persistent memory for agents (main.md, items/)
- `agent_file_service.py` - Per-agent file management (agent_r0_1.json, etc.)
- `agent_session.py` - Session management for agents

### 3. Improved Prompts (Created but not integrated)
- `prompts_improved.py` - Enhanced supervisor and researcher prompts for legacy agentic system
- `search_prompts_improved.py` - Enhanced prompts for Perplexica-style search workflow

## Why Archived

**Replaced by**: New LangGraph workflow in `backend/src/workflow/research/`

**Current System Uses**:
- `/api/chat/stream` endpoint (NOT `/api/research`)
- `run_research_graph()` from `src.workflow.research`
- Simplified researcher agents without full agentic capabilities

**Only Active Code from Legacy**:
- 5 Pydantic schemas extracted to `src/models/schemas.py`:
  - QueryRewrite
  - SearchQueries
  - FollowupQueries
  - SummarizedContent
  - SynthesizedAnswer

## Key Features Lost in Migration

The legacy agentic system had rich features that the new simplified workflow doesn't have:

❌ **Not in New System**:
1. **Agent Todos** - Structured task lists with status tracking (pending, in_progress, done)
2. **Agent Notes** - Research notes with direct quotes and source citations
3. **Supervisor Interventions** - Active monitoring and todo updates based on research depth
4. **Shared Memory** - Inter-agent communication via shared notes
5. **Deep Research Prompts** - Comprehensive prompts with quality metrics and examples
6. **Agent Files** - Persistent per-agent state across sessions

✅ **In New System**:
1. LangGraph state machine
2. Basic web search + scrape
3. Simple findings emission
4. Cleaner code structure

## Frontend Impact

The frontend has excellent UI for displaying:
- Agent todos with progress indicators
- Agent notes with citations
- Research plan and topics
- Findings and sources

**But**: New workflow doesn't emit `agent_todo` and `agent_note` events, so these UI components are not populated for deep_research mode.

## Potential Revival

If detailed agent progress tracking is needed again, consider:

1. **Option A**: Use legacy agentic system (from this archive)
   - Pros: Feature-complete, proven system
   - Cons: More complex codebase

2. **Option B**: Enhance new workflow with todo/note emissions
   - Pros: Keeps new clean structure
   - Cons: Need to reimplement todo/note logic

3. **Option C**: Hybrid approach
   - Use new LangGraph for orchestration
   - Add legacy agentic researcher as agent implementation
   - Best of both worlds

## File Inventory

Total archived: ~318KB, ~7,500+ lines of code

**Legacy Workflow** (~279KB, 6,477 lines):
- `legacy/agentic/` - Coordinator, Researcher, Supervisor, Prompts, Schemas, Models
- `legacy/nodes/` - Planner, Researcher, Reporter, Supervisor, Memory Search
- `legacy/tools/` - Conduct Research, Memory Tools, Research Complete, Think
- `legacy/*.py` - Speed/Balanced/Quality Workflows, Factory, State

**Memory Services** (~31KB):
- `agent_memory_service.py`
- `agent_file_service.py`
- `agent_session.py`

**Improved Prompts** (~40KB):
- `prompts_improved.py` - Agentic system prompts (supervisor, researcher)
- `search_prompts_improved.py` - Search workflow prompts (researcher, writer, classifier)

## References

For migration details, see:
- `/root/asudakov/projects/all_included_deep_research/LEGACY_CLEANUP_REPORT.md`
- `/root/asudakov/projects/all_included_deep_research/WORKFLOW_DOCUMENTATION.md`

## Restoration

To restore legacy code:
```bash
# Copy back to original location
cp -r archive/2026-01-02_legacy_backup/legacy backend/src/workflow/
cp archive/2026-01-02_legacy_backup/agent_*.py backend/src/memory/

# Update imports if needed
# Restore /api/research endpoint
# Initialize workflow_factory in app.py
```

**Note**: Before restoration, review LEGACY_CLEANUP_REPORT.md to understand what needs to be reconnected.

---

**Archived**: 2026-01-02
**Reason**: Replaced by new LangGraph workflow system
**Status**: Preserved for reference and potential feature revival
