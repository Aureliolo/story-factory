"""Architecture/structure phase functions for StoryOrchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.agents import ResponseValidationError
from src.memory.story_state import Character, StoryState

if TYPE_CHECKING:
    from . import StoryOrchestrator

logger = logging.getLogger(__name__)


def build_story_structure(orc: StoryOrchestrator) -> StoryState:
    """Have the architect build the story structure.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        Updated StoryState with world, characters, plot, and chapters.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    logger.info("Building story structure...")
    orc._set_phase("architect")
    orc._emit("agent_start", "Architect", "Building world...")

    logger.info(f"Calling architect with model: {orc.architect.model}")
    orc.story_state = orc.architect.build_story_structure(orc.story_state)

    # Validate key outputs for language correctness
    try:
        if orc.story_state.world_description:
            orc._validate_response(orc.story_state.world_description, "World description")
        if orc.story_state.plot_summary:
            orc._validate_response(orc.story_state.plot_summary, "Plot summary")
    except ResponseValidationError as e:
        logger.warning(f"Validation warning during structure build: {e}")
        # Don't block on validation errors, just log them

    # Set total chapters for progress tracking
    orc._total_chapters = len(orc.story_state.chapters)

    logger.info(
        f"Structure built: {len(orc.story_state.chapters)} chapters, "
        f"{len(orc.story_state.characters)} characters"
    )
    orc._emit("agent_complete", "Architect", "Story structure complete!")
    return orc.story_state


def generate_more_characters(orc: StoryOrchestrator, count: int = 2) -> list[Character]:
    """Generate additional characters for the story.

    Args:
        orc: StoryOrchestrator instance.
        count: Number of characters to generate.

    Returns:
        List of new Character objects.
    """
    if not orc.story_state:
        raise ValueError("No story state. Create a story first.")

    logger.info(f"Generating {count} more characters...")
    orc._emit("agent_start", "Architect", f"Generating {count} new characters...")

    existing_names = [c.name for c in orc.story_state.characters]
    new_characters = orc.architect.generate_more_characters(orc.story_state, existing_names, count)

    # Add to story state
    orc.story_state.characters.extend(new_characters)

    orc._emit(
        "agent_complete",
        "Architect",
        f"Generated {len(new_characters)} new characters!",
    )
    return new_characters


def generate_locations(orc: StoryOrchestrator, count: int = 3) -> list[dict[str, Any]]:
    """Generate locations for the story world.

    Args:
        orc: StoryOrchestrator instance.
        count: Number of locations to generate.

    Returns:
        List of location dictionaries.
    """
    if not orc.story_state:
        raise ValueError("No story state. Create a story first.")

    logger.info(f"Generating {count} locations...")
    orc._emit("agent_start", "Architect", f"Generating {count} new locations...")

    # Get existing location names from world_description heuristic
    existing_locations: list[str] = []
    # Locations will be added to world database by the caller

    locations = orc.architect.generate_locations(orc.story_state, existing_locations, count)

    orc._emit(
        "agent_complete",
        "Architect",
        f"Generated {len(locations)} new locations!",
    )
    return locations


def generate_relationships(
    orc: StoryOrchestrator,
    entity_names: list[str],
    existing_rels: list[tuple[str, str]],
    count: int = 5,
) -> list[dict[str, Any]]:
    """Generate relationships between entities.

    Args:
        orc: StoryOrchestrator instance.
        entity_names: Names of all entities that can have relationships.
        existing_rels: List of (source, target) tuples to avoid duplicates.
        count: Number of relationships to generate.

    Returns:
        List of relationship dictionaries.
    """
    if not orc.story_state:
        raise ValueError("No story state. Create a story first.")

    logger.info(f"Generating {count} relationships...")
    orc._emit("agent_start", "Architect", f"Generating {count} new relationships...")

    relationships = orc.architect.generate_relationships(
        orc.story_state, entity_names, existing_rels, count
    )

    orc._emit(
        "agent_complete",
        "Architect",
        f"Generated {len(relationships)} new relationships!",
    )
    return relationships


def rebuild_world(orc: StoryOrchestrator) -> StoryState:
    """Rebuild the entire world from scratch.

    This regenerates world description, characters, plot, and chapters.
    Use with caution if chapters have already been written.

    Args:
        orc: StoryOrchestrator instance.

    Returns:
        Updated StoryState.
    """
    if not orc.story_state:
        raise ValueError("No story state. Create a story first.")

    logger.info("Rebuilding entire world...")
    orc._emit("agent_start", "Architect", "Rebuilding world from scratch...")

    # Clear existing content but keep the brief
    orc.story_state.world_description = ""
    orc.story_state.world_rules = []
    orc.story_state.characters = []
    orc.story_state.plot_summary = ""
    orc.story_state.plot_points = []
    orc.story_state.chapters = []

    # Rebuild everything
    orc.story_state = orc.architect.build_story_structure(orc.story_state)

    orc._emit("agent_complete", "Architect", "World rebuilt successfully!")
    return orc.story_state
