"""Session management for deep research workflows.

This module provides:
- SessionManager: For creating, retrieving, and managing research sessions
- Multi-chat support: Each chat can have one active deep_research session
- Session resume: Return to an incomplete session in the same chat
"""

from .manager import SessionManager

__all__ = ["SessionManager"]
