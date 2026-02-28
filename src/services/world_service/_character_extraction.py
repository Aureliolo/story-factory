"""Character extraction from StoryState to WorldDatabase."""

import logging

from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._lifecycle_helpers import build_character_lifecycle

logger = logging.getLogger(__name__)


def _extract_characters_to_world(state: StoryState, world_db: WorldDatabase) -> tuple[int, int]:
    """Extract characters and their pre-defined relationships to world database.

    Uses a two-pass approach: first adds all character entities, then creates
    implicit relationships from Character.relationships (set by ArchitectAgent).

    Returns:
        Tuple of (characters_added, implicit_relationships_added).
    """
    added_count = 0
    char_id_map: dict[str, str] = {}
    newly_added: set[str] = set()

    # Pass 1: add all characters, building a name→ID map
    for char in state.characters:
        existing = world_db.get_entity_by_name(char.name, entity_type="character")
        if existing:
            logger.debug("Character already exists: %s", char.name)
            char_id_map[char.name] = existing.id
            continue

        entity_id = world_db.add_entity(
            entity_type="character",
            name=char.name,
            description=char.description,
            attributes={
                "role": char.role,
                "personality_traits": char.trait_names,
                "goals": char.goals,
                "arc_notes": char.arc_notes,
                **build_character_lifecycle(char),
            },
        )
        char_id_map[char.name] = entity_id
        newly_added.add(char.name)
        added_count += 1

    # Pass 2: create implicit relationships only for newly added characters
    # (skip pre-existing characters to avoid duplicates on incremental builds)
    implicit_rel_count = 0
    for char in state.characters:
        if char.name not in newly_added:
            continue
        source_id = char_id_map[char.name]  # guaranteed by pass 1

        for related_name, relationship in char.relationships.items():
            target_id = char_id_map.get(related_name)
            if not target_id:
                logger.debug(
                    "Skipping relationship %s -[%s]-> %s: target not in character list",
                    char.name,
                    relationship,
                    related_name,
                )
                continue

            world_db.add_relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relationship,
            )
            implicit_rel_count += 1
            logger.debug(
                "Created implicit character relationship: %s -[%s]-> %s",
                char.name,
                relationship,
                related_name,
            )

    if implicit_rel_count:
        logger.info(
            "Character extraction created %d implicit relationship(s)",
            implicit_rel_count,
        )
    return added_count, implicit_rel_count
