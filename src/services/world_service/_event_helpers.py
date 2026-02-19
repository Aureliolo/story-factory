"""Event helper functions for world building.

Contains utility functions for building event context, timestamps,
and resolving event participants during world generation, plus the
top-level ``_generate_events`` orchestrator extracted from ``_build.py``.
"""

import json
import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.memory.entities import Entity
from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._name_matching import _find_entity_by_name

if TYPE_CHECKING:
    from src.services import ServiceContainer
    from src.services.world_service import WorldService

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


def _generate_events(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Generate and add world events to the database using quality refinement.

    Collects all entities, relationships, and lifecycle data to build context,
    then generates events that reference existing entities as participants.
    Existing event descriptions are collected for deduplication to avoid
    generating duplicate events.

    Args:
        svc: WorldService instance.
        state: Current story state with brief.
        world_db: World database to read entities from and persist events to.
        services: Service container providing the world quality service.
        cancel_check: Optional callable that returns True to stop generation.

    Returns:
        Number of events successfully added to the world database.
    """
    entity_context = build_event_entity_context(world_db)

    # Get existing event descriptions for dedup
    existing_events = world_db.list_events()
    existing_descriptions = [e.description for e in existing_events]

    # Determine event count (per-project overrides take precedence over settings)
    event_min = (
        state.target_events_min
        if state.target_events_min is not None
        else svc.settings.world_gen_events_min
    )
    event_max = (
        state.target_events_max
        if state.target_events_max is not None
        else svc.settings.world_gen_events_max
    )
    if event_min > event_max:
        logger.error("Invalid event count range: min=%d > max=%d, swapping", event_min, event_max)
        event_min, event_max = event_max, event_min
    event_count = random.randint(event_min, event_max)

    event_results = services.world_quality.generate_events_with_quality(
        state,
        existing_descriptions,
        entity_context,
        event_count,
        cancel_check=cancel_check,
    )

    all_entities = world_db.list_entities()
    added_count = 0
    for event, event_scores in event_results:
        if cancel_check and cancel_check():
            logger.info("Event processing cancelled after %d events", added_count)
            break

        description = event.get("description", "")
        if not description:
            logger.warning("Skipping event with empty description: %s", event)
            continue

        timestamp_in_story = build_event_timestamp(event)
        participants = resolve_event_participants(event, all_entities)

        consequences = event.get("consequences", [])

        world_db.add_event(
            description=description,
            participants=participants if participants else None,
            timestamp_in_story=timestamp_in_story,
            consequences=consequences if consequences else None,
        )
        added_count += 1
        logger.debug(
            "Added event '%s' (quality: %.1f, participants: %d)",
            description[:60],
            event_scores.average,
            len(participants),
        )

    logger.info(
        "Event generation complete: added %d/%d events to world database",
        added_count,
        event_count,
    )
    return added_count
