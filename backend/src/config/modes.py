"""Research mode configurations."""

from enum import Enum


class ResearchMode(str, Enum):
    """Research mode enum with iteration limits."""

    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"

    @classmethod
    def from_string(cls, mode_str: str) -> "ResearchMode":
        """Convert string to ResearchMode, handling model names."""
        mode_lower = mode_str.lower()

        # Handle various formats
        if "speed" in mode_lower:
            return cls.SPEED
        elif "balanced" in mode_lower:
            return cls.BALANCED
        elif "quality" in mode_lower or "deep" in mode_lower:
            return cls.QUALITY

        # Default to balanced
        return cls.BALANCED

    def get_max_iterations(self) -> int:
        """Get max iterations for this mode."""
        from src.config.settings import get_settings

        settings = get_settings()

        if self == ResearchMode.SPEED:
            return settings.speed_max_iterations
        elif self == ResearchMode.BALANCED:
            return settings.balanced_max_iterations
        else:  # QUALITY
            return settings.quality_max_iterations

    def get_max_concurrent(self) -> int:
        """Get max concurrent researchers for this mode."""
        from src.config.settings import get_settings

        settings = get_settings()

        if self == ResearchMode.SPEED:
            return settings.speed_max_concurrent
        elif self == ResearchMode.BALANCED:
            return settings.balanced_max_concurrent
        else:  # QUALITY
            return settings.quality_max_concurrent
