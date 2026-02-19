"""Event helper functions for world building.

Contains utility functions for building event context, timestamps,
and resolving event participants during world generation.
"""

import logging
from typing import Any

from src.memory.entities import Entity
from src.memory.world_database import WorldDatabase
from src.services.world_service._name_matching import _find_entity_by_name

logger = logging.getLogger(__name__)


def _extract_lifecycle_temporal(attrs: dict[str, Any]) -> list[str]:
    """Extract human-readable temporal annotations from lifecycle attributes.

    Lifecycle data is stored under ``attrs["lifecycle"]`` as nested dicts
    produced by ``build_character_lifecycle`` / ``build_entity_lifecycle``.
    This function reads the canonical keys from that nested structure.

    Args:
        attrs: Entity attributes dict (may or may not contain a "lifecycle" key).

    Returns:
        List of "key=value" strings for each temporal field found.
    """
    lifecycle = attrs.get("lifecycle")
    if not lifecycle or not isinstance(lifecycle, dict):
        return []

    parts: list[str] = []

    # Birth / founding / creation / emergence year (stored in lifecycle["birth"]["year"])
    birth = lifecycle.get("birth")
    if isinstance(birth, dict) and birth.get("year") is not None:
        parts.append(f"birth_year={birth['year']}")
        if birth.get("era_name"):
            parts.append(f"era={birth['era_name']}")

    # Death / destruction year (stored in lifecycle["death"]["year"])
    death = lifecycle.get("death")
    if isinstance(death, dict) and death.get("year") is not None:
        parts.append(f"death_year={death['year']}")

    # Founding year (locations/factions store this separately)
    founding = lifecycle.get("founding_year")
    if founding is not None:
        parts.append(f"founding_year={founding}")

    # Destruction / dissolution year
    destruction = lifecycle.get("destruction_year")
    if destruction is not None:
        parts.append(f"destruction_year={destruction}")

    # Temporal notes
    notes = lifecycle.get("temporal_notes")
    if notes:
        parts.append(f"notes={notes}")

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
            if temporal_parts:
                line += ", " + ", ".join(temporal_parts)
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

    return "\n\n".join(context_parts) if context_parts else "No entities yet."


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
    return ", ".join(timestamp_parts)


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
            # Plain string is a supported format (see docstring)
            entity_name = str(p)
            role = "affected"
            logger.debug(
                "Participant provided as plain string (converting to entity_name): %r",
                p,
            )
        if entity_name:
            matched = _find_entity_by_name(all_entities, entity_name)
            if matched:
                participants.append((matched.id, role))
            else:
                logger.warning(
                    "Could not resolve event participant '%s' to any entity",
                    entity_name,
                )
    return participants
