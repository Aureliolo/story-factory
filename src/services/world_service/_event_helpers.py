"""Event helper functions for world building.

Contains utility functions for building event context, timestamps,
and resolving event participants during world generation, plus the
top-level ``_generate_events`` orchestrator extracted from ``_build.py``.
"""

import json
import logging
import random
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.memory.entities import Entity
from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._name_matching import _find_entity_by_name
from src.utils.exceptions import DatabaseClosedError, GenerationCancelledError

if TYPE_CHECKING:
    from src.memory.world_calendar import WorldCalendar
    from src.services import ServiceContainer
    from src.services.world_service import WorldService

logger = logging.getLogger(__name__)

# Markdown/title-prefix cleanup regexes for LLM output sanitization (H1)
_MD_BOLD_RE = re.compile(r"\*{2}(.+?)\*{2}")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_TITLE_PREFIX_RE = re.compile(r"^(?:event\s+title|title)\s*:\s*", re.IGNORECASE)

# Lifecycle sub-dict keys that may contain temporal 'year' values
_LIFECYCLE_TEMPORAL_KEYS = ("birth", "death", "founding", "dissolution", "creation")


def _sanitize_event_text(text: str) -> str:
    """Strip Markdown artifacts and title prefixes from LLM event text.

    Removes ``**bold**``, ``*italic*``, and leading ``Title:`` / ``Event Title:``
    prefixes that LLMs sometimes inject into event descriptions and names.

    Args:
        text: Raw event text from LLM output.

    Returns:
        Cleaned text with Markdown formatting and title prefixes removed.
    """
    text = text.strip()
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _TITLE_PREFIX_RE.sub("", text)
    return text.strip()


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
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug(
                "Failed to parse lifecycle sub-entry as JSON: %r (%s)",
                value[:100],
                exc,
            )
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

    context_parts.append(
        "IMPORTANT: Use entity names EXACTLY as listed above. "
        "Do not abbreviate, modify, or invent new names."
    )
    return "\n\n".join(context_parts)


def build_event_timestamp(
    event: dict[str, Any],
    *,
    calendar: WorldCalendar | None = None,
) -> str:
    """Build a human-readable timestamp string from an event's temporal fields.

    Args:
        event: Event dict with optional 'year', 'month', and 'era_name' keys.
        calendar: Optional WorldCalendar for era auto-resolution when era_name
            is missing but year is present.

    Returns:
        Comma-separated timestamp string (e.g. "Year 1200, Month 3, Dark Age"),
        or empty string if no temporal fields are present.
    """
    timestamp_parts: list[str] = []
    year = event.get("year")
    month = event.get("month")
    era_name = event.get("era_name", "")

    # H2: auto-resolve era from calendar when year is present but era is missing
    if year is not None and not era_name and calendar is not None:
        try:
            era = calendar.get_era_for_year(int(year))
            if era is not None:
                era_name = era.name
                logger.debug(
                    "build_event_timestamp: auto-resolved era '%s' for year %s from calendar",
                    era_name,
                    year,
                )
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("build_event_timestamp: could not resolve era for year %s: %s", year, e)

    if year is not None:
        timestamp_parts.append(f"Year {year}")
    if month is not None:
        timestamp_parts.append(f"Month {month}")
    if era_name:
        timestamp_parts.append(era_name)

    if not timestamp_parts:
        logger.warning(
            "build_event_timestamp: event has no temporal fields (year/month/era all absent)"
        )

    result = ", ".join(timestamp_parts)
    logger.debug(
        "build_event_timestamp: year=%s month=%s era=%s -> '%s'", year, month, era_name, result
    )
    return result


def resolve_event_participants(
    event: dict[str, Any],
    all_entities: list[Entity],
    threshold: float,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Resolve event participant names to entity IDs.

    Each participant entry can be a dict with 'entity_name' and 'role',
    or a plain string (treated as entity_name with role 'affected').

    Args:
        event: Event dict with an optional 'participants' list.
        all_entities: List of entities to match names against.
        threshold: Minimum similarity score for fuzzy name matching fallback (required).

    Returns:
        Tuple of (resolved_participants, dropped_names) where resolved_participants
        is a list of (entity_id, role) tuples and dropped_names is a list of
        participant names that could not be matched.
    """
    participants: list[tuple[str, str]] = []
    dropped_names: list[str] = []
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
        # M6: strip bracket/markdown artifacts from participant names
        if entity_name:
            entity_name = entity_name.strip().strip("[]")
            entity_name = _sanitize_event_text(entity_name)
        if entity_name:
            matched = _find_entity_by_name(all_entities, entity_name, threshold=threshold)
            if matched:
                participants.append((matched.id, role))
            else:
                dropped_names.append(entity_name)
    if dropped_names:
        logger.warning(
            "Dropped %d unresolved event participant(s): %s (out of %d entities)",
            len(dropped_names),
            dropped_names,
            len(all_entities),
        )
    return participants, dropped_names


def _generate_events(
    svc: WorldService,
    state: StoryState,
    world_db: WorldDatabase,
    services: ServiceContainer,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[..., None] | None = None,
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
        progress_callback: Optional callback for per-event progress updates.

    Returns:
        Number of events successfully added to the world database.
    """
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
        logger.warning("Invalid event count range: min=%d > max=%d, swapping", event_min, event_max)
        event_min, event_max = event_max, event_min
    event_count = random.randint(event_min, event_max)

    event_results = services.world_quality.generate_events_with_quality(
        state,
        existing_descriptions,
        entity_context_provider=lambda: build_event_entity_context(world_db),
        count=event_count,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )

    # Retrieve calendar for era auto-resolution in timestamps (H2)
    from src.services.world_service._build_utils import get_calendar_from_world_db

    calendar = get_calendar_from_world_db(world_db)

    all_entities = world_db.list_entities()
    threshold = svc.settings.fuzzy_match_threshold
    added_count = 0
    for event, event_scores in event_results:
        if cancel_check and cancel_check():
            logger.info("Event processing cancelled after %d events", added_count)
            break

        try:
            description = _sanitize_event_text(event.get("description", ""))
            if not description:
                logger.warning("Skipping event with empty description: %s", event)
                continue

            timestamp_in_story = build_event_timestamp(event, calendar=calendar)
            participants, dropped = resolve_event_participants(
                event, all_entities, threshold=threshold
            )
            if dropped:
                logger.warning(
                    "Event '%s' lost %d participant(s): %s",
                    description[:60],
                    len(dropped),
                    dropped,
                )

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
        except GenerationCancelledError, DatabaseClosedError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to add event to database (non-fatal), skipping: %s",
                exc,
                exc_info=True,
            )

    logger.info(
        "Event generation complete: added %d/%d events to world database",
        added_count,
        event_count,
    )
    return added_count
