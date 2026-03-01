"""Utility functions for world building (progress calculation, DB cleanup, calendar)."""

import logging
from typing import TYPE_CHECKING

from src.memory.world_calendar import WorldCalendar
from src.memory.world_database import WorldDatabase

if TYPE_CHECKING:
    from src.services.world_service import WorldBuildOptions

logger = logging.getLogger(__name__)


def calculate_total_steps(
    options: WorldBuildOptions,
    *,
    generate_calendar: bool = False,
    validate_temporal: bool = False,
) -> int:
    """Calculate total number of steps for progress reporting."""
    steps = 3  # Character extraction + embedding + completion
    if options.clear_existing:
        steps += 1
    if generate_calendar:
        steps += 1
    if options.generate_structure:
        steps += 1
        # Quality review steps for Architect output (characters, plot, chapters)
        steps += 3
    if options.generate_locations:
        steps += 1
    if options.generate_factions:
        steps += 1
    if options.generate_items:
        steps += 1
    if options.generate_concepts:
        steps += 1
    if options.generate_relationships:
        steps += 1
        # +1 for orphan recovery step after relationship generation
        steps += 1
    if options.generate_events:
        steps += 1
    if validate_temporal:
        steps += 1
    logger.debug(
        "calculate_total_steps: %d steps (calendar=%s, temporal=%s)",
        steps,
        generate_calendar,
        validate_temporal,
    )
    return steps


def get_calendar_from_world_db(world_db: WorldDatabase) -> WorldCalendar | None:
    """Retrieve the WorldCalendar from world settings, if available.

    Args:
        world_db: WorldDatabase to query for settings.

    Returns:
        WorldCalendar if one exists in world settings, None otherwise.
    """
    world_settings = world_db.get_world_settings()
    if world_settings is None or world_settings.calendar is None:
        logger.debug("No calendar available in world settings for era resolution")
        return None
    logger.debug(
        "Using calendar '%s' for lifecycle era resolution",
        world_settings.calendar.current_era_name,
    )
    return world_settings.calendar


def clear_world_db(world_db: WorldDatabase) -> None:
    """Clear all entities, relationships, and events from world database."""
    # Delete events first (they reference entities via participants)
    world_db.clear_events()

    # Delete relationships (they reference entities)
    relationships = world_db.list_relationships()
    logger.info(f"Deleting {len(relationships)} existing relationships...")
    for rel in relationships:
        world_db.delete_relationship(rel.id)

    # Delete all entities
    entities = world_db.list_entities()
    logger.info(f"Deleting {len(entities)} existing entities...")
    for entity in entities:
        world_db.delete_entity(entity.id)
