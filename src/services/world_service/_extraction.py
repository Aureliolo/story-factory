"""Entity extraction functions for WorldService."""

import logging
import re
from typing import TYPE_CHECKING

from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_positive,
    validate_type,
)

if TYPE_CHECKING:
    from src.services.world_service import WorldService

logger = logging.getLogger(__name__)


def extract_entities_from_structure(
    svc: WorldService, state: StoryState, world_db: WorldDatabase
) -> int:
    """Extract characters and locations from story structure to world database.

    Args:
        svc: WorldService instance.
        state: Story state with characters and world description.
        world_db: WorldDatabase to populate.

    Returns:
        Number of entities extracted.
    """
    validate_not_none(state, "state")
    validate_type(state, "state", StoryState)
    validate_not_none(world_db, "world_db")
    validate_type(world_db, "world_db", WorldDatabase)
    logger.debug(
        f"extract_entities_from_structure called: project_id={state.id}, "
        f"characters={len(state.characters)}"
    )
    count = 0

    try:
        # Extract characters
        for char in state.characters:
            existing = world_db.search_entities(char.name, entity_type="character")
            if existing:
                continue

            attributes = {
                "role": char.role,
                "personality_traits": char.trait_names,
                "goals": char.goals,
                "arc_notes": char.arc_notes,
            }

            entity_id = world_db.add_entity(
                entity_type="character",
                name=char.name,
                description=char.description,
                attributes=attributes,
            )
            count += 1

            # Add relationships from character data
            for related_name, relationship in char.relationships.items():
                related_entities = world_db.search_entities(related_name, entity_type="character")
                if related_entities:
                    world_db.add_relationship(
                        source_id=entity_id,
                        target_id=related_entities[0].id,
                        relation_type=relationship,
                    )

        # Extract locations from world description
        if state.world_description:
            locations = _extract_locations_from_text(svc, state.world_description)
            for loc_name, loc_desc in locations:
                existing = world_db.search_entities(loc_name, entity_type="location")
                if existing:
                    continue

                world_db.add_entity(
                    entity_type="location",
                    name=loc_name,
                    description=loc_desc,
                )
                count += 1

        logger.info(f"Extracted {count} entities from story structure for project {state.id}")
        return count
    except Exception as e:
        logger.error(f"Failed to extract entities for project {state.id}: {e}", exc_info=True)
        raise


def extract_from_chapter(
    svc: WorldService,
    content: str,
    world_db: WorldDatabase,
    chapter_number: int,
) -> dict[str, int]:
    """Extract new entities and events from chapter content.

    Args:
        svc: WorldService instance.
        content: Chapter text content.
        world_db: WorldDatabase to update.
        chapter_number: The chapter number for event tracking.

    Returns:
        Dictionary with counts of extracted items.
    """
    validate_not_empty(content, "content")
    validate_not_none(world_db, "world_db")
    validate_type(world_db, "world_db", WorldDatabase)
    validate_positive(chapter_number, "chapter_number")
    logger.debug(
        f"extract_from_chapter called: chapter={chapter_number}, content_length={len(content)}"
    )
    counts = {
        "entities": 0,
        "relationships": 0,
        "events": 0,
    }

    try:
        # Extract potential new locations mentioned
        locations = svc._extract_locations_from_text(content)
        for loc_name, loc_desc in locations:
            existing = world_db.search_entities(loc_name, entity_type="location")
            if not existing:
                world_db.add_entity(
                    entity_type="location",
                    name=loc_name,
                    description=loc_desc,
                )
                counts["entities"] += 1

        # Extract items mentioned
        items = _extract_items_from_text(svc, content)
        for item_name, item_desc in items:
            existing = world_db.search_entities(item_name, entity_type="item")
            if not existing:
                world_db.add_entity(
                    entity_type="item",
                    name=item_name,
                    description=item_desc,
                )
                counts["entities"] += 1

        # Extract key events
        events = _extract_events_from_text(svc, content, chapter_number)
        for event_desc in events:
            world_db.add_event(
                description=event_desc,
                chapter_number=chapter_number,
            )
            counts["events"] += 1

        logger.info(
            f"Chapter {chapter_number}: extracted {counts['entities']} entities, "
            f"{counts['events']} events"
        )
        return counts
    except Exception as e:
        logger.error(f"Failed to extract from chapter {chapter_number}: {e}", exc_info=True)
        raise


def _extract_locations_from_text(svc: WorldService, text: str) -> list[tuple[str, str]]:
    """Extract location names and descriptions from text.

    Uses heuristics to find capitalized place names.

    Args:
        svc: WorldService instance.
        text: Text to analyze.

    Returns:
        List of (name, description) tuples.
    """
    locations = []

    # Pattern for "the [Place Name]" or "[Place Name]"
    # Look for capitalized multi-word names that might be places
    patterns = [
        r"(?:in|at|to|from|near|within)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:forest|castle|city|town|village|mountain|river|valley|kingdom|empire|realm)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 2 and match not in ["The", "And", "But", "For"]:
                # Get context around the match for description
                idx = text.find(match)
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(match) + 100)
                    context = text[start:end].strip()
                    locations.append((match, context))

    # Deduplicate
    seen = set()
    unique_locations = []
    for name, desc in locations:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique_locations.append((name, desc))

    return unique_locations[: svc.settings.entity_extract_locations_max]


def _extract_items_from_text(svc: WorldService, text: str) -> list[tuple[str, str]]:
    """Extract significant items from text.

    Args:
        svc: WorldService instance.
        text: Text to analyze.

    Returns:
        List of (name, description) tuples.
    """
    items = []

    # Pattern for "the [Item]" with descriptive adjectives
    patterns = [
        r"(?:the|a|an)\s+((?:ancient|magical|enchanted|cursed|sacred|golden|silver)\s+[a-z]+)",
        r"([A-Z][a-z]+(?:'s)?\s+(?:sword|ring|amulet|staff|crown|book|scroll|key|orb|gem))",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 3:
                idx = text.find(match)
                if idx >= 0:
                    start = max(0, idx - 30)
                    end = min(len(text), idx + len(match) + 80)
                    context = text[start:end].strip()
                    items.append((match.title(), context))

    # Deduplicate
    seen = set()
    unique_items = []
    for name, desc in items:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique_items.append((name, desc))

    return unique_items[: svc.settings.entity_extract_items_max]


def _extract_events_from_text(svc: WorldService, text: str, chapter_number: int) -> list[str]:
    """Extract key events from chapter text.

    Args:
        svc: WorldService instance.
        text: Chapter text content.
        chapter_number: Chapter number.

    Returns:
        List of event descriptions.
    """
    events = []

    # Split into sentences and look for action-heavy ones
    sentences = re.split(r"[.!?]+", text)

    action_verbs = [
        "discovered",
        "found",
        "killed",
        "defeated",
        "escaped",
        "revealed",
        "betrayed",
        "married",
        "died",
        "born",
        "destroyed",
        "created",
        "saved",
        "captured",
        "freed",
        "declared",
        "attacked",
        "defended",
        "won",
        "lost",
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if (
            len(sentence) < svc.settings.event_sentence_min_length
            or len(sentence) > svc.settings.event_sentence_max_length
        ):
            continue

        # Check if sentence contains significant action
        sentence_lower = sentence.lower()
        for verb in action_verbs:
            if verb in sentence_lower:
                events.append(sentence)
                break

    return events[: svc.settings.entity_extract_events_max]
