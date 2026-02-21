"""Timeline data types for event and entity history visualization.

These models define the structure for:
- Story timestamps (calendar and relative)
- Timeline items (entities, events)
- Lifecycle tracking for entities
"""

import json
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
                result = self.death.year - self.birth.year
                logger.debug(f"EntityLifecycle.lifespan (birth/death): {result}")
                return result
        # For factions/locations: use founding/destruction
        if self.founding_year is not None:
            if self.destruction_year is not None:
                result = self.destruction_year - self.founding_year
                logger.debug(f"EntityLifecycle.lifespan (founding/destruction): {result}")
                return result
        logger.debug("EntityLifecycle.lifespan: unavailable")
        return None

    @property
    def start_year(self) -> int | None:
        """Get the earliest year associated with this entity.

        Returns:
            Birth year, founding year, or None.
        """
        if self.birth and self.birth.year is not None:
            logger.debug(f"EntityLifecycle.start_year (birth): {self.birth.year}")
            return self.birth.year
        logger.debug(f"EntityLifecycle.start_year (founding): {self.founding_year}")
        return self.founding_year

    @property
    def end_year(self) -> int | None:
        """Get the latest year associated with this entity.

        Returns:
            Death year, destruction year, or None if still active.
        """
        if self.death and self.death.year is not None:
            logger.debug(f"EntityLifecycle.end_year (death): {self.death.year}")
            return self.death.year
        logger.debug(f"EntityLifecycle.end_year (destruction): {self.destruction_year}")
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


def _try_parse_json_timestamp(text: str, result: StoryTimestamp) -> bool:
    """Try to parse text as a JSON object containing timestamp fields.

    Handles LLM output that arrives as stringified JSON, e.g.
    ``'{"year": -2037, "era_name": "Dark Age"}'``.

    Parameters:
        text: The raw input string (stripped).
        result: StoryTimestamp to populate in-place.

    Returns:
        True if JSON was parsed successfully; False otherwise.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError, ValueError:
        return False

    if not isinstance(data, dict):
        return False

    parsed_any = False

    if "year" in data and data["year"] is not None:
        try:
            result.year = int(data["year"])
            parsed_any = True
        except ValueError, TypeError:
            logger.warning("JSON timestamp has non-integer year: %r", data["year"])

    if data.get("era_name"):
        result.era_name = str(data["era_name"])
        parsed_any = True

    if data.get("calendar_id"):
        result.calendar_id = str(data["calendar_id"])
        parsed_any = True

    if "month" in data and data["month"] is not None:
        try:
            month_val = int(data["month"])
            if 1 <= month_val <= 12:
                result.month = month_val
                parsed_any = True
        except ValueError, TypeError:
            pass

    if "day" in data and data["day"] is not None:
        try:
            day_val = int(data["day"])
            if 1 <= day_val <= 31:
                result.day = day_val
                parsed_any = True
        except ValueError, TypeError:
            pass

    if parsed_any:
        logger.debug("Parsed timestamp from JSON: %s", result)
    return parsed_any


def _extract_era_name_from_segments(text: str) -> str | None:
    """Extract era_name from comma-separated timestamp segments.

    After year/month/day extraction, any remaining segment that does not
    match a known keyword pattern (year, month, day, chapter, etc.) or
    start with digits is treated as an era name.

    Parameters:
        text: The original timestamp string.

    Returns:
        The era name string, or None if no candidate segment is found.
    """
    keyword_pattern = re.compile(
        r"^\s*(?:year|month|day|chapter|event|part|phase|act|in)\s", re.IGNORECASE
    )
    digit_start = re.compile(r"^\s*-?\d")

    for segment in text.split(","):
        stripped = segment.strip()
        if not stripped:
            continue
        if keyword_pattern.search(stripped):
            continue
        if digit_start.match(stripped):
            continue
        return stripped
    return None


def parse_timestamp(text: str) -> StoryTimestamp:
    """Parse a free-form timestamp string into a StoryTimestamp model.

    Preserves the original input in ``raw_text``. First attempts JSON parsing
    for structured LLM output (e.g. ``{"year": -2037, "era_name": "Dark Age"}``).
    Then falls back to regex extraction of calendar fields (``year``, ``month``,
    ``day``).  Supports negative years and BC/BCE suffixes. Extracts ``era_name``
    from comma-separated segments. If no year is found, attempts to derive
    ``relative_order`` from phrases like "chapter N", "event/part/phase/act N",
    or from a standalone day value.

    Parameters:
        text: The timestamp string to parse.

    Returns:
        Parsed timestamp with ``raw_text`` preserved and any discovered fields populated.
    """
    logger.debug("Parsing timestamp: %r", text)

    result = StoryTimestamp(raw_text=text.strip())

    # Strategy 1: Try JSON parsing first (handles stringified LLM output)
    if _try_parse_json_timestamp(text.strip(), result):
        return result

    text_lower = text.lower().strip()

    # Strategy 2: Regex extraction
    # Try to extract year (supports negative years and BC/BCE suffix)
    year_patterns = [
        (r"year\s+(-?\d+)", False),  # "Year 1042" or "Year -2037"
        (r"(-?\d{3,4})\s*(ad|ce|bc|bce)\b", True),  # "1042 BCE" — has era suffix
        (r"(-?\d{3,4})(?:\s|,|$)", False),  # bare 3-4 digit number
        (r"in\s+(-?\d{3,4})", False),  # "in 1042" or "in -1042"
    ]
    for pattern, has_suffix_group in year_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result.year = int(match.group(1))
            # Negate year for BC/BCE suffix if year is positive
            if has_suffix_group and match.lastindex is not None and match.lastindex >= 2:
                suffix = match.group(2)
                if suffix in ("bc", "bce") and result.year > 0:
                    result.year = -result.year
                    logger.debug("Negated year for %s suffix: %d", suffix, result.year)
            logger.debug("Extracted year: %d", result.year)
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
                logger.debug("Extracted month: %d", result.month)
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
                logger.debug("Extracted day: %d", result.day)
            break

    # Try to extract era_name from comma-separated segments
    if result.era_name is None:
        era_candidate = _extract_era_name_from_segments(text)
        if era_candidate:
            result.era_name = era_candidate
            logger.debug("Extracted era_name from segment: %s", result.era_name)

    # If no calendar date found, try relative ordering
    if result.year is None:
        # Try chapter number
        chapter_match = re.search(r"chapter\s+(\d+)", text_lower)
        if chapter_match:
            result.relative_order = int(chapter_match.group(1))
            logger.debug("Extracted chapter as relative_order: %d", result.relative_order)
        # Try "event N" or "part N"
        elif event_match := re.search(r"(?:event|part|phase|act)\s+(\d+)", text_lower):
            result.relative_order = int(event_match.group(1))
            logger.debug("Extracted event/part as relative_order: %d", result.relative_order)
        # If we only have day and nothing else, use it as relative order
        elif result.day is not None and result.year is None and result.month is None:
            result.relative_order = result.day
            result.day = None  # Clear day since we're using relative ordering
            logger.debug("Using standalone day as relative_order: %d", result.relative_order)

    logger.debug("Parsed timestamp result: %s", result)
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
        elif isinstance(founding, str):
            try:
                result.founding_year = int(founding)
            except ValueError:
                logger.warning("Could not parse founding_year string: %r", founding)

    destruction = lifecycle_data.get("destruction_year")
    if destruction is not None:
        if isinstance(destruction, int):
            result.destruction_year = destruction
        elif isinstance(destruction, str):
            try:
                result.destruction_year = int(destruction)
            except ValueError:
                logger.warning("Could not parse destruction_year string: %r", destruction)

    logger.debug(
        f"Extracted lifecycle: birth={result.birth}, death={result.death}, "
        f"founding={result.founding_year}, destruction={result.destruction_year}"
    )
    return result
