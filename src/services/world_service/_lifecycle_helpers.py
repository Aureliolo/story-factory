"""Lifecycle attribute builders for world entity generation.

Convert temporal fields into lifecycle dicts compatible with
``extract_lifecycle_from_attributes()`` in timeline_types.py.
``build_character_lifecycle`` operates on a typed Character model;
``build_entity_lifecycle`` operates on raw entity dicts (location,
faction, item, concept).  The output is merged into entity attributes
during ``build_world()``.
"""

import logging
from typing import Any, Literal

from src.memory.story_state import Character
from src.memory.world_calendar import WorldCalendar

logger = logging.getLogger(__name__)


def _resolve_era_name(year: int, calendar: WorldCalendar) -> str | None:
    """Look up which era a year belongs to using the calendar.

    Args:
        year: Year to look up.
        calendar: WorldCalendar with era definitions.

    Returns:
        Era name if found, None if year doesn't fall within any defined era.
    """
    era = calendar.get_era_for_year(year)
    if era is None:
        logger.debug("Year %d does not fall within any defined era in calendar", year)
        return None
    return era.name


def _validate_and_resolve_era(
    year: int | None,
    claimed_era: str | None,
    calendar: WorldCalendar | None,
    entity_label: str,
    year_field: str,
) -> str | None:
    """Validate that a year falls within the claimed era; auto-resolve if mismatch.

    When a calendar is available and a year is provided:
    - If ``claimed_era`` is provided, validates it matches the era the year falls
      within.  On mismatch, logs a warning and returns the correct era.
    - If ``claimed_era`` is not provided, auto-resolves the era from the calendar
      and logs at DEBUG level.

    Args:
        year: The year value (may be None).
        claimed_era: The era name claimed by the LLM (may be None).
        calendar: Optional WorldCalendar for era lookups.
        entity_label: Human-readable label for log messages (e.g. "character 'Gandalf'").
        year_field: Field name for log messages (e.g. "birth_year", "founding_year").

    Returns:
        The validated/resolved era name, or the original ``claimed_era`` if no
        calendar is available or the year is None.
    """
    if calendar is None or year is None:
        return claimed_era

    resolved = _resolve_era_name(year, calendar)

    if claimed_era is not None:
        # Validate: claimed era should match the calendar's era for this year
        if resolved is not None and claimed_era != resolved:
            logger.warning(
                "%s: %s=%d claimed era '%s' but calendar says '%s' — auto-resolving to '%s'",
                entity_label,
                year_field,
                year,
                claimed_era,
                resolved,
                resolved,
            )
            return resolved
        # Claimed era matches or year is outside all eras — keep claimed
        return claimed_era

    # No claimed era but we have a year — auto-resolve from calendar
    if resolved is not None:
        logger.debug(
            "%s: auto-resolved %s=%d to era '%s' from calendar",
            entity_label,
            year_field,
            year,
            resolved,
        )
    return resolved


def build_character_lifecycle(
    char: Character,
    calendar: WorldCalendar | None = None,
) -> dict[str, Any]:
    """Build a lifecycle attributes dict from a Character's temporal fields.

    Produces a ``{"lifecycle": {...}}`` dict compatible with
    ``extract_lifecycle_from_attributes()``.  Returns an empty dict when no
    temporal data is present so callers can safely unpack with ``**``.

    When a ``calendar`` is provided:
    - If ``birth_year`` is set but ``birth_era`` is not, auto-resolves the era
      from the calendar.
    - If ``death_year`` is set but ``death_era`` is not, auto-resolves the era
      from the calendar.
    - If an era is provided but mismatches the calendar for the given year,
      logs a warning and corrects it.

    Args:
        char: Character with optional temporal fields.
        calendar: Optional WorldCalendar for era validation/resolution.

    Returns:
        Dict with ``"lifecycle"`` key, or empty dict if no temporal data.
    """
    has_data = (
        char.birth_year is not None
        or char.death_year is not None
        or char.birth_era is not None
        or char.death_era is not None
        or char.temporal_notes
    )
    if not has_data:
        logger.debug("No temporal data for character '%s', skipping lifecycle", char.name)
        return {}

    entity_label = f"character '{char.name}'"
    lifecycle: dict[str, Any] = {}

    if char.birth_year is not None or char.birth_era is not None:
        birth_era = _validate_and_resolve_era(
            char.birth_year, char.birth_era, calendar, entity_label, "birth_year"
        )
        birth: dict[str, Any] = {}
        if char.birth_year is not None:
            birth["year"] = char.birth_year
        if birth_era is not None:
            birth["era_name"] = birth_era
        lifecycle["birth"] = birth

    if char.death_year is not None or char.death_era is not None:
        death: dict[str, Any] = {}
        if char.death_year is not None and char.death_year < 0:
            # Negative death_year is an LLM sentinel for "alive" — skip entire death section
            logger.warning(
                "Character '%s' has negative death_year %d — treating as alive "
                "(LLM sentinel, dropping death data including era '%s')",
                char.name,
                char.death_year,
                char.death_era,
            )
        else:
            death_era = _validate_and_resolve_era(
                char.death_year, char.death_era, calendar, entity_label, "death_year"
            )
            if char.death_year is not None:
                death["year"] = char.death_year
            if death_era is not None:
                death["era_name"] = death_era
            elif char.death_year is not None:
                # M5: death era could not be resolved — log for visibility
                logger.debug(
                    "%s: death_year=%d but era_name is None (calendar %s, death_era claim=%r)",
                    entity_label,
                    char.death_year,
                    "available" if calendar else "absent",
                    char.death_era,
                )
            if death:
                lifecycle["death"] = death

    if char.temporal_notes:
        lifecycle["temporal_notes"] = char.temporal_notes

    if not lifecycle:
        logger.debug("No lifecycle data remaining for character '%s' after filtering", char.name)
        return {}

    logger.debug(
        "Built character lifecycle for '%s': birth_year=%s, death_year=%s",
        char.name,
        char.birth_year,
        char.death_year,
    )
    return {"lifecycle": lifecycle}


def _is_destruction_sentinel(value: Any) -> bool:
    """Check if a destruction/dissolution year value is the sentinel ``0``.

    LLMs sometimes emit ``destruction_year: 0`` or ``dissolution_year: 0`` to
    mean "not destroyed / still exists".  This function detects that sentinel
    so callers can drop the field.

    Args:
        value: The raw destruction/dissolution year value.

    Returns:
        True if the value is the integer 0 sentinel.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def build_entity_lifecycle(
    entity_dict: dict[str, Any],
    entity_type: Literal["location", "faction", "item", "concept"],
    calendar: WorldCalendar | None = None,
) -> dict[str, Any]:
    """Build a lifecycle attributes dict from an entity dict's temporal fields.

    Handles location, faction, item, and concept entity types.  Each type uses
    different field names for its temporal data (e.g. ``founding_year`` for
    locations/factions, ``creation_year`` for items, ``emergence_year`` for
    concepts).

    Note: faction ``dissolution_year`` is mapped to ``destruction_year`` in
    the lifecycle dict to match the ``EntityLifecycle`` schema.

    When a ``calendar`` is provided:
    - Validates that the origin year falls within the claimed era's year range.
      On mismatch, logs a warning and auto-resolves the era from the calendar.
    - If no era is claimed but a year is provided, auto-resolves the era.

    ``destruction_year=0`` and ``dissolution_year=0`` are treated as sentinels
    meaning "not destroyed" and are dropped from the lifecycle.

    Args:
        entity_dict: Entity data dict with optional temporal fields.
        entity_type: One of "location", "faction", "item", "concept".
        calendar: Optional WorldCalendar for era validation/resolution.

    Returns:
        Dict with ``"lifecycle"`` key, or empty dict if no temporal data.
    """
    entity_name = entity_dict.get("name", "?")
    entity_label = f"{entity_type} '{entity_name}'"
    lifecycle: dict[str, Any] = {}

    if entity_type == "location":
        founding = entity_dict.get("founding_year")
        destruction = entity_dict.get("destruction_year")
        era = entity_dict.get("founding_era")
        notes = entity_dict.get("temporal_notes", "")

        # Treat destruction_year=0 as sentinel for "not destroyed"
        if _is_destruction_sentinel(destruction):
            logger.warning(
                "%s: destruction_year=0 is a sentinel for 'not destroyed' — dropping",
                entity_label,
            )
            destruction = None

        era = _validate_and_resolve_era(founding, era, calendar, entity_label, "founding_year")

        if founding is not None:
            lifecycle["founding_year"] = founding
        birth_data = {k: v for k, v in (("year", founding), ("era_name", era)) if v is not None}
        if birth_data:
            lifecycle["birth"] = birth_data
        if destruction is not None:
            lifecycle["destruction_year"] = destruction
        if notes:
            lifecycle["temporal_notes"] = notes

    elif entity_type == "faction":
        founding = entity_dict.get("founding_year")
        dissolution = entity_dict.get("dissolution_year")
        era = entity_dict.get("founding_era")
        notes = entity_dict.get("temporal_notes", "")

        # Treat dissolution_year=0 as sentinel for "not dissolved"
        if _is_destruction_sentinel(dissolution):
            logger.warning(
                "%s: dissolution_year=0 is a sentinel for 'not dissolved' — dropping",
                entity_label,
            )
            dissolution = None

        era = _validate_and_resolve_era(founding, era, calendar, entity_label, "founding_year")

        if founding is not None:
            lifecycle["founding_year"] = founding
        birth_data = {k: v for k, v in (("year", founding), ("era_name", era)) if v is not None}
        if birth_data:
            lifecycle["birth"] = birth_data
        if dissolution is not None:
            lifecycle["destruction_year"] = dissolution
        if notes:
            lifecycle["temporal_notes"] = notes

    elif entity_type == "item":
        creation = entity_dict.get("creation_year")
        era = entity_dict.get("creation_era")
        notes = entity_dict.get("temporal_notes", "")

        era = _validate_and_resolve_era(creation, era, calendar, entity_label, "creation_year")

        birth_data = {k: v for k, v in (("year", creation), ("era_name", era)) if v is not None}
        if birth_data:
            lifecycle["birth"] = birth_data
        if notes:
            lifecycle["temporal_notes"] = notes

    elif entity_type == "concept":
        emergence = entity_dict.get("emergence_year")
        era = entity_dict.get("emergence_era")
        notes = entity_dict.get("temporal_notes", "")

        era = _validate_and_resolve_era(emergence, era, calendar, entity_label, "emergence_year")

        birth_data = {k: v for k, v in (("year", emergence), ("era_name", era)) if v is not None}
        if birth_data:
            lifecycle["birth"] = birth_data
        if notes:
            lifecycle["temporal_notes"] = notes

    if not lifecycle:
        logger.debug("No temporal data for %s '%s'", entity_type, entity_name)
        return {}

    logger.debug(
        "Built %s lifecycle for '%s': %s",
        entity_type,
        entity_name,
        lifecycle,
    )
    return {"lifecycle": lifecycle}
