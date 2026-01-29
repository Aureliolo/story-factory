"""Timeline data types for event and entity history visualization.

These models define the structure for:
- Story timestamps (calendar and relative)
- Timeline items (entities, events)
- Lifecycle tracking for entities
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.memory.world_calendar import WorldCalendar

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
    era_name: str | None = Field(default=None, description="Name of the era this date is in")
    calendar_id: str | None = Field(
        default=None, description="ID of the WorldCalendar this timestamp uses"
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """
        Provide a tuple key used to order StoryTimestamp values chronologically, with calendar dates before relative orders and unknown times last.

        Returns:
            tuple[int, int, int, int]: A sortable key:
              - (0, year, month_or_0, day_or_0) when a calendar year is present.
              - (1, relative_order, 0, 0) when only a relative order is present.
              - (2, 0, 0, 0) when no ordering information is available.
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
        """
        Produce a human-readable label for the timestamp.

        Returns:
            A string containing `raw_text` if present; otherwise a comma-separated date like
            "Year <year>, Month <month>, Day <day>" for available calendar fields, or
            "Event #<relative_order>" if only a relative order is available, or
            "Unknown time" when no date information exists.
        """
        if self.raw_text:
            return self.raw_text

        parts = []
        if self.year is not None:
            year_str = f"Year {self.year}"
            if self.era_name:
                year_str = f"{year_str} {self.era_name}"
            parts.append(year_str)
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

    def format_display(self, calendar: WorldCalendar | None = None) -> str:
        """Format timestamp using a WorldCalendar for custom display.

        Args:
            calendar: Optional WorldCalendar instance for custom formatting.

        Returns:
            Formatted date string using calendar conventions if available.
        """
        if calendar is not None and self.year is not None:
            # Use calendar's formatting
            result: str = calendar.format_date(
                year=self.year,
                month=self.month,
                day=self.day,
                era_name=self.era_name,
            )
            return result
        # Fall back to display_text
        return self.display_text

    @property
    def has_date(self) -> bool:
        """
        Determine whether the timestamp contains any calendar or relative-order information.

        Returns:
            True if `year`, `month`, `day`, or `relative_order` is set, False otherwise.
        """
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
    temporal_notes: str = Field(
        default="",
        description="Additional temporal context or notes about the entity's timeline",
    )
    founding_year: int | None = Field(
        default=None,
        description="Year of founding/establishment (for factions/locations)",
    )
    destruction_year: int | None = Field(
        default=None,
        description="Year of destruction/dissolution (for factions/locations)",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def lifespan(self) -> int | None:
        """Calculate lifespan in years if birth/death or founding/destruction known.

        Returns:
            Lifespan in years, or None if cannot be calculated.
        """
        # For characters: use birth/death
        if self.birth and self.birth.year is not None:
            if self.death and self.death.year is not None:
                return self.death.year - self.birth.year
        # For factions/locations: use founding/destruction
        if self.founding_year is not None:
            if self.destruction_year is not None:
                return self.destruction_year - self.founding_year
        return None

    @property
    def start_year(self) -> int | None:
        """Get the earliest year associated with this entity.

        Returns:
            Birth year, founding year, or None.
        """
        if self.birth and self.birth.year is not None:
            return self.birth.year
        return self.founding_year

    @property
    def end_year(self) -> int | None:
        """Get the latest year associated with this entity.

        Returns:
            Death year, destruction year, or None if still active.
        """
        if self.death and self.death.year is not None:
            return self.death.year
        return self.destruction_year


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
        """
        Determine whether the timeline item represents a range (has an end timestamp).

        Returns:
            True if the item has an `end` timestamp (range), False otherwise.
        """
        return self.end is not None


def parse_timestamp(text: str) -> StoryTimestamp:
    """
    Parse a free-form timestamp string into a StoryTimestamp model.

    Preserves the original input in `raw_text`. Extracts calendar fields (`year`, `month`, `day`) when present; if no year is found, attempts to derive `relative_order` from phrases like "chapter N", "event/part/phase/act N", or from a standalone day value. If no structured information can be determined, the result contains only `raw_text`.

    Parameters:
        text (str): The timestamp string to parse.

    Returns:
        StoryTimestamp: Parsed timestamp with `raw_text` preserved and any discovered `year`, `month`, `day`, or `relative_order` populated.
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
    """
    Build an EntityLifecycle from an attributes dictionary's "lifecycle" entry.

    If present, the "lifecycle" value must be a mapping that may contain any of
    "birth", "death", "first_appearance", and "last_appearance". Each field may be
    either a dict of StoryTimestamp fields or a string that will be parsed by
    parse_timestamp.

    Parameters:
        attributes (dict[str, Any]): Entity attributes; may include a "lifecycle"
            mapping describing timestamp data.

    Returns:
        EntityLifecycle constructed from the "lifecycle" data, or `None` if the
        "lifecycle" entry is missing or not a mapping.
    """
    logger.debug(f"Extracting lifecycle from attributes: {list(attributes.keys())}")

    lifecycle_data = attributes.get("lifecycle")
    if lifecycle_data is None:
        return None

    if not isinstance(lifecycle_data, dict):
        logger.warning(
            f"Malformed lifecycle data: expected dict, got {type(lifecycle_data).__name__} "
            f"(value: {lifecycle_data!r}). Ignoring lifecycle for this entity."
        )
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

    # Extract additional temporal fields
    if notes := lifecycle_data.get("temporal_notes"):
        result.temporal_notes = str(notes)

    # Use 'is not None' to handle year 0 correctly
    founding = lifecycle_data.get("founding_year")
    if founding is not None:
        if isinstance(founding, int):
            result.founding_year = founding
        elif isinstance(founding, str) and founding.isdigit():
            result.founding_year = int(founding)

    destruction = lifecycle_data.get("destruction_year")
    if destruction is not None:
        if isinstance(destruction, int):
            result.destruction_year = destruction
        elif isinstance(destruction, str) and destruction.isdigit():
            result.destruction_year = int(destruction)

    logger.debug(
        f"Extracted lifecycle: birth={result.birth}, death={result.death}, "
        f"founding={result.founding_year}, destruction={result.destruction_year}"
    )
    return result
