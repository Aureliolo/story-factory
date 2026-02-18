"""Lifecycle attribute builders for world entity generation.

Convert temporal fields from Pydantic entity models into lifecycle dicts
compatible with `extract_lifecycle_from_attributes()` in timeline_types.py.
The output is merged into entity attributes during `build_world()`.
"""

import logging
from typing import Any

from src.memory.story_state import Character

logger = logging.getLogger(__name__)


def build_character_lifecycle(char: Character) -> dict[str, Any]:
    """Build a lifecycle attributes dict from a Character's temporal fields.

    Produces a ``{"lifecycle": {...}}`` dict compatible with
    ``extract_lifecycle_from_attributes()``.  Returns an empty dict when no
    temporal data is present so callers can safely unpack with ``**``.

    Args:
        char: Character with optional temporal fields.

    Returns:
        Dict with ``"lifecycle"`` key, or empty dict if no temporal data.
    """
    has_data = (
        char.birth_year is not None
        or char.death_year is not None
        or char.birth_era is not None
        or char.temporal_notes
    )
    if not has_data:
        logger.debug("No temporal data for character '%s', skipping lifecycle", char.name)
        return {}

    lifecycle: dict[str, Any] = {}

    if char.birth_year is not None or char.birth_era is not None:
        birth: dict[str, Any] = {}
        if char.birth_year is not None:
            birth["year"] = char.birth_year
        if char.birth_era is not None:
            birth["era_name"] = char.birth_era
        lifecycle["birth"] = birth

    if char.death_year is not None:
        lifecycle["death"] = {"year": char.death_year}

    if char.temporal_notes:
        lifecycle["temporal_notes"] = char.temporal_notes

    logger.debug(
        "Built character lifecycle for '%s': birth_year=%s, death_year=%s",
        char.name,
        char.birth_year,
        char.death_year,
    )
    return {"lifecycle": lifecycle}


def build_entity_lifecycle(entity_dict: dict[str, Any], entity_type: str) -> dict[str, Any]:
    """Build a lifecycle attributes dict from an entity dict's temporal fields.

    Handles location, faction, item, and concept entity types.  Each type uses
    different field names for its temporal data (e.g. ``founding_year`` for
    locations/factions, ``creation_year`` for items, ``emergence_year`` for
    concepts).

    Args:
        entity_dict: Entity data dict with optional temporal fields.
        entity_type: One of "location", "faction", "item", "concept".

    Returns:
        Dict with ``"lifecycle"`` key, or empty dict if no temporal data.
    """
    lifecycle: dict[str, Any] = {}

    if entity_type == "location":
        founding = entity_dict.get("founding_year")
        destruction = entity_dict.get("destruction_year")
        era = entity_dict.get("founding_era")
        notes = entity_dict.get("temporal_notes", "")

        if founding is not None:
            lifecycle["founding_year"] = founding
            if era:
                lifecycle["birth"] = {"year": founding, "era_name": era}
        if destruction is not None:
            lifecycle["destruction_year"] = destruction
        if notes:
            lifecycle["temporal_notes"] = notes

    elif entity_type == "faction":
        founding = entity_dict.get("founding_year")
        dissolution = entity_dict.get("dissolution_year")
        era = entity_dict.get("founding_era")
        notes = entity_dict.get("temporal_notes", "")

        if founding is not None:
            lifecycle["founding_year"] = founding
            if era:
                lifecycle["birth"] = {"year": founding, "era_name": era}
        if dissolution is not None:
            lifecycle["destruction_year"] = dissolution
        if notes:
            lifecycle["temporal_notes"] = notes

    elif entity_type == "item":
        creation = entity_dict.get("creation_year")
        era = entity_dict.get("creation_era")
        notes = entity_dict.get("temporal_notes", "")

        if creation is not None:
            birth: dict[str, Any] = {"year": creation}
            if era:
                birth["era_name"] = era
            lifecycle["birth"] = birth
        if notes:
            lifecycle["temporal_notes"] = notes

    elif entity_type == "concept":
        emergence = entity_dict.get("emergence_year")
        era = entity_dict.get("emergence_era")
        notes = entity_dict.get("temporal_notes", "")

        if emergence is not None:
            birth_data: dict[str, Any] = {"year": emergence}
            if era:
                birth_data["era_name"] = era
            lifecycle["birth"] = birth_data
        if notes:
            lifecycle["temporal_notes"] = notes

    else:
        logger.warning("Unknown entity type '%s' for lifecycle building", entity_type)
        return {}

    if not lifecycle:
        logger.debug("No temporal data for %s '%s'", entity_type, entity_dict.get("name", "?"))
        return {}

    logger.debug(
        "Built %s lifecycle for '%s': %s",
        entity_type,
        entity_dict.get("name", "?"),
        lifecycle,
    )
    return {"lifecycle": lifecycle}
