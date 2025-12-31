"""Manage main.md and JSON index for memory files."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class IndexManager:
    """Manage the File Index section in main.md."""

    def __init__(self, main_file_path: Path) -> None:
        self.main_file_path = main_file_path
        logger.info("index_manager_initialized", path=str(main_file_path))

    def read_main_file(self) -> str:
        if not self.main_file_path.exists():
            raise FileNotFoundError(f"Main file not found: {self.main_file_path}")
        return self.main_file_path.read_text(encoding="utf-8")

    def write_main_file(self, content: str) -> None:
        self.main_file_path.write_text(content, encoding="utf-8")
        logger.info("main_file_updated")

    def update_file_index(self, file_path: str, description: str, category: str) -> None:
        """
        Add or update file reference in the File Index section.

        Args:
            file_path: Relative path to the memory file
            description: Short description
            category: One of projects, concepts, conversations, preferences, other
        """
        content = self.read_main_file()
        category_map = {
            "project": "Projects",
            "projects": "Projects",
            "concept": "Concepts",
            "concepts": "Concepts",
            "conversation": "Conversations",
            "conversations": "Conversations",
            "preference": "Preferences",
            "preferences": "Preferences",
            "main": "Main",
            "other": "Other",
        }
        category_header = f"### {category_map.get(category, category.title())}"
        link = f"- [{Path(file_path).stem.replace('_', ' ').title()}](/memory_files/{file_path}) - {description}"

        if category_header not in content:
            logger.warning("category_not_found_in_index", category=category)
            return

        pattern = rf"({re.escape(category_header)}.*?)(\n###|\n---|\Z)"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            logger.warning("category_section_not_found", category=category)
            return

        section_content = match.group(1)
        file_pattern = rf"- \[.*?\]\(/memory_files/{re.escape(file_path)}\)"
        if re.search(file_pattern, section_content):
            new_section = re.sub(file_pattern, link, section_content)
        else:
            new_section = section_content.replace("<!-- Add", link + "\n<!-- Add")

        content = content.replace(section_content, new_section)
        self.write_main_file(content)
        logger.info("file_index_updated", file_path=file_path, category=category)

    def touch_updated_at(self) -> None:
        """Update the Last Updated line in main.md."""
        content = self.read_main_file()
        updated = re.sub(
            r"Last Updated: .*",
            f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}",
            content,
        )
        self.write_main_file(updated)


class JsonIndexManager:
    """Manage files_index.json metadata."""

    def __init__(self, json_index_path: Path) -> None:
        self.json_index_path = json_index_path
        logger.info("json_index_manager_initialized", path=str(json_index_path))

    def read_index(self) -> dict[str, Any]:
        if not self.json_index_path.exists():
            return {
                "version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "files": [],
            }

        try:
            content = self.json_index_path.read_text(encoding="utf-8")
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("json_index_read_failed", error=str(exc))
            return {
                "version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "files": [],
            }

    def write_index(self, data: dict[str, Any]) -> None:
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.json_index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        logger.info("json_index_updated", files_count=len(data.get("files", [])))

    def upsert_file(self, file_info: dict[str, Any]) -> None:
        data = self.read_index()
        files = data.get("files", [])

        existing_idx = next(
            (idx for idx, entry in enumerate(files) if entry.get("file_path") == file_info.get("file_path")),
            None,
        )
        if existing_idx is None:
            files.append(file_info)
        else:
            files[existing_idx] = file_info

        data["files"] = files
        self.write_index(data)

    def remove_file(self, file_path: str) -> None:
        data = self.read_index()
        files = [entry for entry in data.get("files", []) if entry.get("file_path") != file_path]
        data["files"] = files
        self.write_index(data)
