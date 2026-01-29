"""World Settings Models - configuration for world-specific features.

Contains the WorldSettings model for storing world-level configuration
including calendar, timeline, and validation settings.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.memory.world_calendar import WorldCalendar

logger = logging.getLogger(__name__)


class WorldSettings(BaseModel):
    """World-level settings stored per world database.

    Contains configuration for:
    - Calendar system (if using custom calendar)
    - Timeline boundaries
    - Temporal validation preferences
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for these settings",
    )
    calendar: WorldCalendar | None = Field(
        default=None,
        description="Custom calendar system for this world (None = use simple years)",
    )
    timeline_start_year: int | None = Field(
        default=None,
        description="Earliest year in the world's history (for timeline bounds)",
    )
    timeline_end_year: int | None = Field(
        default=None,
        description="Latest year in the world's timeline (None = open-ended)",
    )
    validate_temporal_consistency: bool = Field(
        default=True,
        description="Whether to validate temporal consistency of entities",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(use_enum_values=True)

    @property
    def has_calendar(self) -> bool:
        """Check if a custom calendar is configured."""
        result = self.calendar is not None
        logger.debug(f"has_calendar check: {result}")
        return result

    @property
    def timeline_span(self) -> int | None:
        """Get the span of the timeline in years, if both bounds are set."""
        if self.timeline_start_year is not None and self.timeline_end_year is not None:
            span = self.timeline_end_year - self.timeline_start_year
            logger.debug(f"Timeline span: {span} years")
            return span
        return None

    def get_era_abbreviation(self) -> str:
        """Get the era abbreviation from calendar, or default 'Y' for year.

        Returns:
            Era abbreviation string.
        """
        if self.calendar:
            abbrev = self.calendar.era_abbreviation
            logger.debug(f"Era abbreviation from calendar: {abbrev}")
            return abbrev
        logger.debug("No calendar, using default era abbreviation 'Y'")
        return "Y"

    def format_year(self, year: int) -> str:
        """Format a year using calendar settings if available.

        Args:
            year: Year to format.

        Returns:
            Formatted year string (e.g., "Year 342 TE" or "342 Y").
        """
        if self.calendar:
            result = self.calendar.format_date(year)
            logger.debug(f"Formatted year {year} with calendar: {result}")
            return result
        era = self.get_era_abbreviation()
        result = f"Year {year} {era}"
        logger.debug(f"Formatted year {year} without calendar: {result}")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for storage.

        Returns:
            Dictionary representation suitable for JSON storage.
        """
        logger.debug("Converting WorldSettings to dict")
        # Pydantic's model_dump handles nested models recursively
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldSettings:
        """Create WorldSettings from dictionary data.

        Args:
            data: Dictionary with settings data.

        Returns:
            WorldSettings instance.
        """
        logger.debug("Creating WorldSettings from dict")
        # Handle calendar deserialization
        if "calendar" in data and data["calendar"] is not None:
            data["calendar"] = WorldCalendar.from_dict(data["calendar"])
        return cls.model_validate(data)


def create_default_world_settings() -> WorldSettings:
    """Create default world settings without a calendar.

    Returns:
        WorldSettings with default configuration.
    """
    logger.debug("Creating default world settings (no calendar)")
    return WorldSettings(
        validate_temporal_consistency=True,
    )
