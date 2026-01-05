"""Helpers for per-session agent memory services."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

import structlog

from src.memory.agent_file_service import AgentFileService
from src.memory.agent_memory_service import AgentMemoryService
from src.memory.file_manager import FileManager

logger = structlog.get_logger(__name__)


def create_agent_session_services(memory_root: Path, session_id: str) -> tuple[AgentMemoryService, AgentFileService, Path]:
    """
    Create agent session services for deep research.
    
    Creates session directory structure:
    - agent_sessions/{session_id}/
      - agents/ (agent personal files)
      - items/ (agent notes)
      - main.md (session main file - created by AgentMemoryService.read_main_file())
      - files_index.json (session index - created here)
    """
    session_dir = memory_root / "agent_sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "agents").mkdir(exist_ok=True)
    (session_dir / "items").mkdir(exist_ok=True)

    # Create files_index.json for this session (only for deep research)
    json_index = session_dir / "files_index.json"
    if not json_index.exists():
        json_index.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "files": [],
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        logger.info("Session files_index.json created", session_id=session_id, path=str(json_index))

    file_manager = FileManager(str(session_dir))
    return AgentMemoryService(file_manager), AgentFileService(file_manager), session_dir


def cleanup_agent_session_dir(memory_root: Path, session_dir: Path) -> None:
    try:
        root = memory_root.resolve()
        target = session_dir.resolve()
        if target == root or root not in target.parents:
            logger.warning("Refusing to cleanup session dir outside memory root", session_dir=str(target))
            return
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            logger.info("Agent session dir cleaned", session_dir=str(target))
    except Exception as exc:
        logger.warning("Agent session cleanup failed", session_dir=str(session_dir), error=str(exc))
