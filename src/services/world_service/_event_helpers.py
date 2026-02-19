"""Event helper functions for world building.

Contains utility functions for building event context, timestamps,
and resolving event participants during world generation.
"""

import json
import logging
from typing import Any

from src.memory.entities import Entity
from src.memory.world_database import WorldDatabase
from src.services.world_service._name_matching import _find_entity_by_name

logger = logging.getLogger(__name__)

# Lifecycle sub-dict keys that may contain temporal 'year' values
_LIFECYCLE_TEMPORAL_KEYS = ("birth", "death", "founding", "dissolution", "creation")


def _parse_lifecycle_sub(value: Any) -> dict[str, Any] | None:
    """Parse a lifecycle sub-entry that may be a dict or a JSON string.

    WorldDatabase flattens attributes at nesting depth 3, so lifecycle sub-dicts
    like ``{"year": 1200}`` become the JSON string ``'{"year": 1200}'`` after
    storage and retrieval.

    Args:
        value: Raw value from the lifecycle dict — either a dict or a JSON string.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError, TypeError:
            pass
    return None


def _extract_lifecycle_temporal(attrs: dict[str, Any]) -> list[str]:
    """Extract temporal annotations from entity lifecycle attributes.

    Lifecycle data is stored under attrs["lifecycle"][key]["year"], where key
    is one of birth, death, founding, dissolution, creation.  After WorldDatabase
    storage (max nesting depth 3), the inner dicts may be JSON strings instead of
    dicts — this function handles both forms.

    Args:
        attrs: Entity attributes dict.

    Returns:
        List of formatted temporal strings (e.g. "birth_year=1200").
    """
    lifecycle = attrs.get("lifecycle")
    if not lifecycle or not isinstance(lifecycle, dict):
        return []

    parts: list[str] = []
    for key in _LIFECYCLE_TEMPORAL_KEYS:
        sub = _parse_lifecycle_sub(lifecycle.get(key))
        if sub is not None and sub.get("year") is not None:
            parts.append(f"{key}_year={sub['year']}")

    logger.debug("Extracted %d temporal parts from lifecycle attributes", len(parts))
    return parts


def build_event_entity_context(world_db: WorldDatabase) -> str:
    """Build a text context of entities and relationships for event generation prompts.

    Produces a formatted string with entity names/types, lifecycle temporal data,
    and relationships suitable for inclusion in LLM prompts.

    Args:
        world_db: World database to read entities and relationships from.

    Returns:
        Formatted context string, or "No entities yet." if the database is empty.
    """
    all_entities = world_db.list_entities()
    all_relationships = world_db.list_relationships()
    entity_by_id = {e.id: e for e in all_entities}

    context_parts: list[str] = []

    if all_entities:
        entity_lines: list[str] = []
        for e in all_entities:
            line = f"  - {e.name} ({e.type})"
            attrs = e.attributes or {}
            temporal_parts = _extract_lifecycle_temporal(attrs)
            for part in temporal_parts:
                line += f", {part}"
            entity_lines.append(line)
        context_parts.append("ENTITIES:\n" + "\n".join(entity_lines))

    if all_relationships:
        rel_lines: list[str] = []
        for r in all_relationships:
            source = entity_by_id.get(r.source_id)
            target = entity_by_id.get(r.target_id)
            if source and target:
                rel_lines.append(f"  - {source.name} -[{r.relation_type}]-> {target.name}")
            else:
                logger.warning(
                    "Skipping relationship %s -> %s in event context: "
                    "source_found=%s, target_found=%s (dangling reference)",
                    r.source_id,
                    r.target_id,
                    source is not None,
                    target is not None,
                )
        if rel_lines:
            context_parts.append("RELATIONSHIPS:\n" + "\n".join(rel_lines))

    if not context_parts:
        logger.warning(
            "build_event_entity_context: no entities or relationships found in world database. "
            "Events will be generated without entity context."
        )
        return "No entities yet."
    return "\n\n".join(context_parts)


def build_event_timestamp(event: dict[str, Any]) -> str:
    """Build a human-readable timestamp string from an event's temporal fields.

    Args:
        event: Event dict with optional 'year', 'month', and 'era_name' keys.

    Returns:
        Comma-separated timestamp string (e.g. "Year 1200, Month 3, Dark Age"),
        or empty string if no temporal fields are present.
    """
    timestamp_parts: list[str] = []
    year = event.get("year")
    month = event.get("month")
    era_name = event.get("era_name", "")
    if year is not None:
        timestamp_parts.append(f"Year {year}")
    if month is not None:
        timestamp_parts.append(f"Month {month}")
    if era_name:
        timestamp_parts.append(era_name)
    result = ", ".join(timestamp_parts)
    logger.debug(
        "build_event_timestamp: year=%s month=%s era=%s -> '%s'", year, month, era_name, result
    )
    return result


def resolve_event_participants(
    event: dict[str, Any], all_entities: list[Entity]
) -> list[tuple[str, str]]:
    """Resolve event participant names to entity IDs.

    Each participant entry can be a dict with 'entity_name' and 'role',
    or a plain string (treated as entity_name with role 'affected').

    Args:
        event: Event dict with an optional 'participants' list.
        all_entities: List of entities to match names against.

    Returns:
        List of (entity_id, role) tuples for successfully resolved participants.
    """
    participants: list[tuple[str, str]] = []
    for p in event.get("participants", []):
        if isinstance(p, dict):
            entity_name = p.get("entity_name", "")
            role = p.get("role", "affected")
        else:
            entity_name = str(p)
            role = "affected"
            logger.warning(
                "Unexpected participant format (expected dict, got %s): %r",
                type(p).__name__,
                p,
            )
        if entity_name:
            matched = _find_entity_by_name(all_entities, entity_name)
            if matched:
                participants.append((matched.id, role))
            else:
                logger.warning(
                    "Could not resolve event participant '%s' to any of %d entities "
                    "-- participant will be dropped from event",
                    entity_name,
                    len(all_entities),
                )
    return participants
