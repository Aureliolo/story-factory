"""Graph analysis and context functions for WorldService."""

import logging
from typing import TYPE_CHECKING

from src.memory.entities import Entity, Relationship
from src.memory.world_database import WorldDatabase

if TYPE_CHECKING:
    from src.services.world_service import WorldService

logger = logging.getLogger(__name__)


def find_path(
    svc: WorldService,
    world_db: WorldDatabase,
    source_id: str,
    target_id: str,
) -> list[str] | None:
    """Find shortest path between two entities.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        source_id: Source entity ID.
        target_id: Target entity ID.

    Returns:
        List of entity IDs forming the path, or None if no path.
    """
    return world_db.find_path(source_id, target_id)


def get_connected_entities(
    svc: WorldService,
    world_db: WorldDatabase,
    entity_id: str,
    max_depth: int = 2,
) -> list[Entity]:
    """Get entities connected to a given entity.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        entity_id: Central entity ID.
        max_depth: Maximum relationship depth to traverse.

    Returns:
        List of connected Entity objects.
    """
    return world_db.get_connected_entities(entity_id, max_depth=max_depth)


def get_communities(svc: WorldService, world_db: WorldDatabase) -> list[list[str]]:
    """Detect communities/clusters in the entity graph.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.

    Returns:
        List of communities, each a list of entity IDs.
    """
    return world_db.get_communities()


def get_most_connected(
    svc: WorldService,
    world_db: WorldDatabase,
    limit: int = 10,
) -> list[tuple[Entity, int]]:
    """Get most connected entities (highest degree centrality).

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        limit: Maximum number to return.

    Returns:
        List of (Entity, connection_count) tuples.
    """
    return world_db.get_most_connected(limit=limit)


def get_entity_summary(svc: WorldService, world_db: WorldDatabase) -> dict[str, int]:
    """Get a summary count of entities by type.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.

    Returns:
        Dictionary mapping entity type to count.
    """
    logger.debug("get_entity_summary called")
    summary = {
        "character": world_db.count_entities("character"),
        "location": world_db.count_entities("location"),
        "item": world_db.count_entities("item"),
        "faction": world_db.count_entities("faction"),
        "concept": world_db.count_entities("concept"),
        "relationships": len(world_db.list_relationships()),
    }
    logger.debug(f"Entity summary: {summary}")
    return summary


def add_relationship(
    svc: WorldService,
    world_db: WorldDatabase,
    source_id: str,
    target_id: str,
    relation_type: str,
    description: str = "",
    bidirectional: bool = False,
) -> str:
    """Add a relationship between entities.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        source_id: Source entity ID.
        target_id: Target entity ID.
        relation_type: Type of relationship.
        description: Optional description.
        bidirectional: Whether relationship goes both ways.

    Returns:
        The new relationship's ID.
    """
    logger.info(f"Adding relationship: {source_id} -> {target_id} ({relation_type})")
    rel_id = world_db.add_relationship(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        description=description,
        bidirectional=bidirectional,
        validate=svc.settings.relationship_validation_enabled,
    )
    logger.debug(f"Created relationship {rel_id}")
    return rel_id


def delete_relationship(svc: WorldService, world_db: WorldDatabase, relationship_id: str) -> bool:
    """Delete a relationship.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        relationship_id: Relationship ID to delete.

    Returns:
        True if deleted, False if not found.
    """
    logger.info(f"Deleting relationship {relationship_id}")
    result = world_db.delete_relationship(relationship_id)
    if result:
        logger.debug(f"Relationship {relationship_id} deleted")
    else:
        logger.warning(f"Relationship {relationship_id} not found")
    return result


def get_relationships(
    svc: WorldService,
    world_db: WorldDatabase,
    entity_id: str | None = None,
) -> list[Relationship]:
    """Get relationships, optionally filtered by entity.

    Args:
        svc: WorldService instance.
        world_db: WorldDatabase instance.
        entity_id: Optional entity ID to filter by.

    Returns:
        List of Relationship objects.
    """
    if entity_id:
        return world_db.get_relationships(entity_id)
    return world_db.list_relationships()
