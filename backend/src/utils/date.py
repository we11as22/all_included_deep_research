"""Utility functions for date and time handling."""

from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_date() -> str:
    """
    Get current date in a readable format for prompts.
    
    Returns:
        Current date string in format: "December 29, 2024"
    """
    return datetime.now(ZoneInfo("UTC")).strftime("%B %d, %Y")


def get_current_datetime() -> str:
    """
    Get current date and time in ISO format.
    
    Returns:
        Current datetime string in ISO format
    """
    return datetime.now(ZoneInfo("UTC")).isoformat()

