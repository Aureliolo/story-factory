"""Timeline data types for event and entity history visualization.

These models define the structure for:
- Story timestamps (calendar and relative)
- Timeline items (entities, events)
- Lifecycle tracking for entities
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class StoryTimestamp(BaseModel):
    """A timestamp within the story's timeline.

    Supports both calendar-based (year/month/day) and relative ordering
    for non-calendar timelines (e.g., "before the war", "after the fall").
    """

    year: int | None = Field(default=None, description="Year in story calendar")
    month: int | None = Field(default=None, ge=1, le=12, description="Month (1-12)")
    day: int | None = Field(default=None, ge=1, le=31, description="Day (1-31)")
    relative_order: int | None = Field(
        default=None,
        description="Relative ordering for non-calendar timelines (lower = earlier)",
    )
    raw_text: str = Field(default="", description="Original timestamp string for display")

    model_config = ConfigDict(use_enum_values=True)

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """Generate a sortable key for ordering timestamps.

        Returns:
            Tuple of (has_calendar, year, month, day) or relative_order fallback.
            Calendar dates sort before relative-only timestamps.
        """
        if self.year is not None:
            # Calendar-based: sort by year, month, day
            return (0, self.year, self.month or 0, self.day or 0)
        elif self.relative_order is not None:
            # Relative only: sort after all calendar dates
            return (1, self.relative_order, 0, 0)
        else:
            # No ordering info: sort last
            return (2, 0, 0, 0)

    @property
    def display_text(self) -> str:
        """Get human-readable display text.

        Returns:
            Original raw_text if available, otherwise formatted date.
        """
        if self.raw_text:
            return self.raw_text

        parts = []
        if self.year is not None:
            parts.append(f"Year {self.year}")
        if self.month is not None:
            parts.append(f"Month {self.month}")
        if self.day is not None:
            parts.append(f"Day {self.day}")

        if parts:
            return ", ".join(parts)
        elif self.relative_order is not None:
            return f"Event #{self.relative_order}"
        else:
            return "Unknown time"

    @property
    def has_date(self) -> bool:
        """Check if this timestamp has any date information."""
        return (
            self.year is not None
            or self.month is not None
            or self.day is not None
            or self.relative_order is not None
        )


class EntityLifecycle(BaseModel):
    """Lifecycle information for an entity (birth, death, etc.)."""

    birth: StoryTimestamp | None = Field(
        default=None,
        description="When the entity came into existence",
    )
    death: StoryTimestamp | None = Field(
        default=None,
        description="When the entity ceased to exist (if applicable)",
    )
    first_appearance: StoryTimestamp | None = Field(
        default=None,
        description="First mention in the story",
    )
    last_appearance: StoryTimestamp | None = Field(
        default=None,
        description="Last mention in the story",
    )

    model_config = ConfigDict(use_enum_values=True)


class TimelineItem(BaseModel):
    """An item to display on the timeline.

    Represents either an entity lifespan or a point event.
    """

    id: str = Field(description="Unique identifier for the timeline item")
    entity_id: str | None = Field(
        default=None,
        description="ID of the associated entity (for entity lifespans)",
    )
    event_id: str | None = Field(
        default=None,
        description="ID of the associated event (for events)",
    )
    label: str = Field(description="Display label for the item")
    item_type: str = Field(description="Type: character, location, event, etc.")
    start: StoryTimestamp = Field(description="Start timestamp")
    end: StoryTimestamp | None = Field(
        default=None,
        description="End timestamp (None for point events)",
    )
    color: str = Field(description="Hex color for display")
    description: str = Field(default="", description="Tooltip description")
    group: str | None = Field(
        default=None,
        description="Optional grouping (e.g., by entity type or chapter)",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def is_range(self) -> bool:
        """Check if this is a range item (has start and end) vs point item."""
        return self.end is not None


def parse_timestamp(text: str) -> StoryTimestamp:
    """Parse a timestamp string into a StoryTimestamp.

    Supports various formats:
    - "Year 1042" or "1042" -> year=1042
    - "Year 1042, Month 3" -> year=1042, month=3
    - "Day 15 of Month 7, Year 1042" -> year=1042, month=7, day=15
    - "Day 3" or "Day 3 of the journey" -> relative_order based on day
    - "Before the war" -> raw_text only (no structured data)
    - "Chapter 5" -> relative_order=5

    Args:
        text: The timestamp string to parse.

    Returns:
        Parsed StoryTimestamp with raw_text preserved.
    """
    logger.debug(f"Parsing timestamp: {text!r}")

    result = StoryTimestamp(raw_text=text.strip())
    text_lower = text.lower().strip()

    # Try to extract year
    year_patterns = [
        r"year\s+(\d+)",
        r"(\d{3,4})\s*(?:ad|ce|bc|bce)?",  # 3-4 digit numbers
        r"in\s+(\d{3,4})",
    ]
    for pattern in year_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result.year = int(match.group(1))
            logger.debug(f"Extracted year: {result.year}")
            break

    # Try to extract month
    month_patterns = [
        r"month\s+(\d+)",
        r"(\d{1,2})(?:st|nd|rd|th)?\s+month",
    ]
    for pattern in month_patterns:
        match = re.search(pattern, text_lower)
        if match:
            month_val = int(match.group(1))
            if 1 <= month_val <= 12:
                result.month = month_val
                logger.debug(f"Extracted month: {result.month}")
            break

    # Try to extract day
    day_patterns = [
        r"day\s+(\d+)",
        r"(\d{1,2})(?:st|nd|rd|th)?\s+day",
    ]
    for pattern in day_patterns:
        match = re.search(pattern, text_lower)
        if match:
            day_val = int(match.group(1))
            if 1 <= day_val <= 31:
                result.day = day_val
                logger.debug(f"Extracted day: {result.day}")
            break

    # If no calendar date found, try relative ordering
    if result.year is None:
        # Try chapter number
        chapter_match = re.search(r"chapter\s+(\d+)", text_lower)
        if chapter_match:
            result.relative_order = int(chapter_match.group(1))
            logger.debug(f"Extracted chapter as relative_order: {result.relative_order}")
        # Try "event N" or "part N"
        elif event_match := re.search(r"(?:event|part|phase|act)\s+(\d+)", text_lower):
            result.relative_order = int(event_match.group(1))
            logger.debug(f"Extracted event/part as relative_order: {result.relative_order}")
        # If we only have day and nothing else, use it as relative order
        elif result.day is not None and result.year is None and result.month is None:
            result.relative_order = result.day
            result.day = None  # Clear day since we're using relative ordering
            logger.debug(f"Using standalone day as relative_order: {result.relative_order}")

    logger.debug(f"Parsed timestamp result: {result}")
    return result


def extract_lifecycle_from_attributes(attributes: dict[str, Any]) -> EntityLifecycle | None:
    """Extract lifecycle information from entity attributes.

    Looks for a 'lifecycle' key in attributes with birth/death timestamps.

    Args:
        attributes: Entity attributes dictionary.

    Returns:
        EntityLifecycle if found, None otherwise.
    """
    logger.debug(f"Extracting lifecycle from attributes: {list(attributes.keys())}")

    lifecycle_data = attributes.get("lifecycle")
    if lifecycle_data is None or not isinstance(lifecycle_data, dict):
        return None

    result = EntityLifecycle()

    if birth_data := lifecycle_data.get("birth"):
        if isinstance(birth_data, dict):
            result.birth = StoryTimestamp(**birth_data)
        elif isinstance(birth_data, str):
            result.birth = parse_timestamp(birth_data)

    if death_data := lifecycle_data.get("death"):
        if isinstance(death_data, dict):
            result.death = StoryTimestamp(**death_data)
        elif isinstance(death_data, str):
            result.death = parse_timestamp(death_data)

    if first_data := lifecycle_data.get("first_appearance"):
        if isinstance(first_data, dict):
            result.first_appearance = StoryTimestamp(**first_data)
        elif isinstance(first_data, str):
            result.first_appearance = parse_timestamp(first_data)

    if last_data := lifecycle_data.get("last_appearance"):
        if isinstance(last_data, dict):
            result.last_appearance = StoryTimestamp(**last_data)
        elif isinstance(last_data, str):
            result.last_appearance = parse_timestamp(last_data)

    logger.debug(f"Extracted lifecycle: birth={result.birth}, death={result.death}")
    return result
