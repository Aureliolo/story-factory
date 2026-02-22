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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
        Provide a tuple key used to order StoryTimestamp values
        chronologically, with calendar dates before relative orders
        and unknown times last.

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
                logger.debug("EntityLifecycle.lifespan (birth/death): %s", result)
                return result
        # For factions/locations: use founding/destruction
        if self.founding_year is not None:
            if self.destruction_year is not None:
                result = self.destruction_year - self.founding_year
                logger.debug("EntityLifecycle.lifespan (founding/destruction): %s", result)
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
            logger.debug("EntityLifecycle.start_year (birth): %s", self.birth.year)
            return self.birth.year
        logger.debug("EntityLifecycle.start_year (founding): %s", self.founding_year)
        return self.founding_year

    @property
    def end_year(self) -> int | None:
        """Get the latest year associated with this entity.

        Returns:
            Death year, destruction year, or None if still active.
        """
        if self.death and self.death.year is not None:
            logger.debug("EntityLifecycle.end_year (death): %s", self.death.year)
            return self.death.year
        logger.debug("EntityLifecycle.end_year (destruction): %s", self.destruction_year)
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


def _try_parse_json_timestamp(text: str) -> StoryTimestamp | None:
    """Try to parse text as a JSON object containing timestamp fields.

    Handles LLM output that arrives as stringified JSON, e.g.
    ``'{"year": -2037, "era_name": "Dark Age"}'``.

    Parameters:
        text: The raw input string (stripped).

    Returns:
        A new StoryTimestamp with ``raw_text`` set to *text* and any extracted
        fields populated, or None if the text is not valid JSON, not a dict,
        or contains no usable timestamp fields (year, month, day, era_name,
        calendar_id).
    """
    try:
        data = json.loads(text)
    except ValueError:
        return None

    if not isinstance(data, dict):
        return None

    fields: dict[str, Any] = {}

    if "year" in data and data["year"] is not None:
        if isinstance(data["year"], bool):
            logger.warning("JSON timestamp has non-integer year: %r", data["year"])
        else:
            try:
                fields["year"] = int(data["year"])
            except ValueError, TypeError:
                logger.warning("JSON timestamp has non-integer year: %r", data["year"])

    raw_era = data.get("era_name")
    if raw_era:
        stripped_era = str(raw_era).strip()
        if stripped_era:
            fields["era_name"] = stripped_era

    raw_cid = data.get("calendar_id")
    if raw_cid:
        stripped_cid = str(raw_cid).strip()
        if stripped_cid:
            fields["calendar_id"] = stripped_cid

    if "month" in data and data["month"] is not None:
        try:
            month_val = int(data["month"])
            if 1 <= month_val <= 12:
                fields["month"] = month_val
            else:
                logger.warning("JSON timestamp month out of range: %d", month_val)
        except ValueError, TypeError:
            logger.warning("JSON timestamp has non-integer month: %r", data["month"])

    if "day" in data and data["day"] is not None:
        try:
            day_val = int(data["day"])
            if 1 <= day_val <= 31:
                fields["day"] = day_val
            else:
                logger.warning("JSON timestamp day out of range: %d", day_val)
        except ValueError, TypeError:
            logger.warning("JSON timestamp has non-integer day: %r", data["day"])

    if fields:
        logger.debug("Parsed timestamp from JSON: %s", fields)
        return StoryTimestamp(raw_text=text, **fields)
    return None


_CALENDAR_SUFFIXES = frozenset({"bc", "bce", "ad", "ce"})
_KEYWORD_PATTERN = re.compile(
    r"^\s*(?:year|month|day|chapter|event|part|phase|act|in)\b", re.IGNORECASE
)
_DIGIT_START = re.compile(r"^\s*-?\d")


def _extract_era_name_from_segments(text: str) -> str | None:
    """Extract era_name from comma-separated timestamp segments.

    After year/month/day extraction, any remaining segment that does not
    match a known keyword pattern (year, month, day, chapter, event, part,
    phase, act, in), start with a digit or minus sign, or equal a calendar
    era suffix (BC, BCE, AD, CE) is treated as an era name.

    Parameters:
        text: The original timestamp string.

    Returns:
        The era name string, or None if no candidate segment is found.
    """
    logger.debug("Extracting era name from segments: %r", text)

    for segment in text.split(","):
        stripped = segment.strip()
        if not stripped:
            continue
        if _KEYWORD_PATTERN.match(stripped):
            logger.debug("Skipping keyword segment: %r", stripped)
            continue
        if _DIGIT_START.match(stripped):
            logger.debug("Skipping digit-start segment: %r", stripped)
            continue
        if stripped.lower() in _CALENDAR_SUFFIXES:
            logger.debug("Skipping calendar suffix segment: %r", stripped)
            continue
        logger.debug("Found era name candidate: %r", stripped)
        return stripped
    logger.debug("No era name candidate found in: %r", text)
    return None


def parse_timestamp(text: str) -> StoryTimestamp:
    """Parse a free-form timestamp string into a StoryTimestamp model.

    Preserves the original input in ``raw_text``. First attempts JSON parsing
    for structured LLM output (e.g. ``{"year": -2037, "era_name": "Dark Age"}``).
    Then falls back to regex extraction of calendar fields (``year``, ``month``,
    ``day``).  Supports negative years and BC/BCE/AD/CE era suffixes (BC/BCE
    negate a positive year value). Extracts ``era_name`` from comma-separated
    segments. If no year is found, attempts to derive ``relative_order`` from
    phrases like "chapter N", "event/part/phase/act N", or from a standalone
    day value.

    Parameters:
        text: The timestamp string to parse.

    Returns:
        Parsed timestamp with ``raw_text`` preserved and any discovered fields populated.
    """
    logger.debug("Parsing timestamp: %r", text)
    stripped = text.strip()

    # Strategy 1: Try JSON parsing first (handles stringified LLM output)
    json_result = _try_parse_json_timestamp(stripped)
    if json_result is not None:
        return json_result

    # Strategy 2: Regex extraction — accumulate fields, construct once
    fields: dict[str, Any] = {"raw_text": stripped}
    text_lower = stripped.lower()

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
            year = int(match.group(1))
            # Negate year for BC/BCE suffix if year is positive
            if has_suffix_group and match.lastindex is not None and match.lastindex >= 2:
                suffix = match.group(2)
                if suffix in ("bc", "bce") and year > 0:
                    year = -year
                    logger.debug("Negated year for %s suffix: %d", suffix, year)
            fields["year"] = year
            logger.debug("Extracted year: %d", year)
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
                fields["month"] = month_val
                logger.debug("Extracted month: %d", month_val)
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
                fields["day"] = day_val
                logger.debug("Extracted day: %d", day_val)
                break

    # Try to extract era_name from comma-separated segments
    era_candidate = _extract_era_name_from_segments(stripped)
    if era_candidate:
        fields["era_name"] = era_candidate
        logger.debug("Extracted era_name from segment: %s", era_candidate)

    # If no calendar date found, try relative ordering
    if "year" not in fields:
        # Try chapter number
        chapter_match = re.search(r"chapter\s+(\d+)", text_lower)
        if chapter_match:
            fields["relative_order"] = int(chapter_match.group(1))
            logger.debug("Extracted chapter as relative_order: %d", fields["relative_order"])
        # Try "event N" or "part N"
        elif event_match := re.search(r"(?:event|part|phase|act)\s+(\d+)", text_lower):
            fields["relative_order"] = int(event_match.group(1))
            logger.debug(
                "Extracted event/part as relative_order: %d",
                fields["relative_order"],
            )
        # If we only have day and nothing else, use it as relative order
        elif "day" in fields and "month" not in fields:
            fields["relative_order"] = fields.pop("day")
            logger.debug(
                "Using standalone day as relative_order: %d",
                fields["relative_order"],
            )

    result = StoryTimestamp(**fields)
    logger.debug("Parsed timestamp result: %s", result)
    return result


SENTINEL_YEARS: frozenset[int] = frozenset({-1, 0, 9999})
"""Year values that LLMs use as placeholders/sentinels and must be rejected."""


def _check_sentinel(year: int, field_name: str) -> int | None:
    """Reject sentinel year values that LLMs use as placeholders.

    Parameters:
        year: The parsed year value to check.
        field_name: Field label for log messages.

    Returns:
        The year unchanged if valid, or None if it is a sentinel value.
    """
    if year in SENTINEL_YEARS:
        logger.warning("Rejected sentinel %s value: %d (known placeholder)", field_name, year)
        return None
    return year


def _parse_year(value: object, field_name: str) -> int | None:
    """Coerce a raw LLM year value to int, with type-aware logging.

    Handles: bool (rejected), int (pass-through), float (truncated to int),
    str (parsed via ``int()``), and anything else (logged and rejected).
    Sentinel values (-1, 0, 9999) are rejected after conversion.

    Parameters:
        value: The raw value from LLM output (any type).
        field_name: Field label for log messages (e.g. "founding_year").

    Returns:
        The year as an int, or None if the value cannot be converted.
    """
    if isinstance(value, bool):
        logger.warning("Unexpected %s type: bool (%r)", field_name, value)
        return None
    if isinstance(value, int):
        return _check_sentinel(value, field_name)
    if isinstance(value, float):
        logger.debug("Converted float %s to int: %s", field_name, value)
        return _check_sentinel(int(value), field_name)
    if isinstance(value, str):
        try:
            return _check_sentinel(int(value), field_name)
        except ValueError as exc:
            logger.warning("Could not parse %s string: %r — %s", field_name, value, exc)
            return None
    logger.warning("Unexpected %s type: %s (%r)", field_name, type(value).__name__, value)
    return None


def _filter_sentinel_year_in_dict(data: dict, field_prefix: str) -> dict:
    """Return a copy of *data* with any sentinel year value parsed through ``_parse_year``.

    Dict-constructed ``StoryTimestamp(**data)`` bypasses ``_parse_year``, so
    sentinel values (-1, 0, 9999) would slip through.  This helper applies
    the same sentinel rejection used by ``founding_year`` / ``destruction_year``
    to the ``year`` key in timestamp dicts (birth, death, first/last appearance).

    Returns a new dict — the original is never mutated.
    """
    if "year" not in data or data["year"] is None:
        return data
    parsed = _parse_year(data["year"], f"{field_prefix}.year")
    return {**data, "year": parsed}


def extract_lifecycle_from_attributes(attributes: dict[str, Any]) -> EntityLifecycle | None:
    """
    Build an EntityLifecycle from an attributes dictionary's "lifecycle" entry.

    If present, the "lifecycle" value must be a mapping that may contain any of
    "birth", "death", "first_appearance", "last_appearance", "temporal_notes",
    "founding_year", and "destruction_year". Timestamp fields may be either a
    dict of StoryTimestamp fields or a string parsed by ``parse_timestamp``.
    Year fields accept int, float (truncated to int), or string values.

    Parameters:
        attributes (dict[str, Any]): Entity attributes; may include a "lifecycle"
            mapping describing timestamp data.

    Returns:
        EntityLifecycle constructed from the "lifecycle" data, or `None` if the
        "lifecycle" entry is missing or not a mapping.
    """
    logger.debug("Extracting lifecycle from attributes: %s", list(attributes.keys()))

    lifecycle_data = attributes.get("lifecycle")
    if lifecycle_data is None:
        return None

    if not isinstance(lifecycle_data, dict):
        logger.warning(
            "Malformed lifecycle data: expected dict, got %s (value: %r). "
            "Ignoring lifecycle for this entity.",
            type(lifecycle_data).__name__,
            lifecycle_data,
        )
        return None

    result = EntityLifecycle()

    if birth_data := lifecycle_data.get("birth"):
        if isinstance(birth_data, dict):
            filtered = _filter_sentinel_year_in_dict(birth_data, "birth")
            try:
                result.birth = StoryTimestamp(**filtered)
            except ValidationError as exc:
                logger.warning("Malformed birth timestamp dict %r: %s", birth_data, exc)
        elif isinstance(birth_data, str):
            result.birth = parse_timestamp(birth_data)

    if death_data := lifecycle_data.get("death"):
        if isinstance(death_data, dict):
            filtered = _filter_sentinel_year_in_dict(death_data, "death")
            try:
                result.death = StoryTimestamp(**filtered)
            except ValidationError as exc:
                logger.warning("Malformed death timestamp dict %r: %s", death_data, exc)
        elif isinstance(death_data, str):
            result.death = parse_timestamp(death_data)

    if first_data := lifecycle_data.get("first_appearance"):
        if isinstance(first_data, dict):
            filtered = _filter_sentinel_year_in_dict(first_data, "first_appearance")
            try:
                result.first_appearance = StoryTimestamp(**filtered)
            except ValidationError as exc:
                logger.warning("Malformed first_appearance timestamp dict %r: %s", first_data, exc)
        elif isinstance(first_data, str):
            result.first_appearance = parse_timestamp(first_data)

    if last_data := lifecycle_data.get("last_appearance"):
        if isinstance(last_data, dict):
            filtered = _filter_sentinel_year_in_dict(last_data, "last_appearance")
            try:
                result.last_appearance = StoryTimestamp(**filtered)
            except ValidationError as exc:
                logger.warning("Malformed last_appearance timestamp dict %r: %s", last_data, exc)
        elif isinstance(last_data, str):
            result.last_appearance = parse_timestamp(last_data)

    # Extract additional temporal fields
    if notes := lifecycle_data.get("temporal_notes"):
        result.temporal_notes = str(notes)

    # Use 'is not None' so that sentinel values like 0 reach _parse_year for proper rejection
    founding = lifecycle_data.get("founding_year")
    if founding is not None:
        result.founding_year = _parse_year(founding, "founding_year")

    destruction = lifecycle_data.get("destruction_year")
    if destruction is not None:
        result.destruction_year = _parse_year(destruction, "destruction_year")

    logger.debug(
        "Extracted lifecycle: birth=%s, death=%s, founding=%s, destruction=%s",
        result.birth,
        result.death,
        result.founding_year,
        result.destruction_year,
    )
    return result
