"""Graph analysis mixin for WorldService."""

from __future__ import annotations

import logging
from typing import Any

from src.memory.entities import Entity
from src.memory.world_database import WorldDatabase

from ._base import WorldServiceBase

logger = logging.getLogger(__name__)


class GraphMixin(WorldServiceBase):
    """Mixin providing graph analysis operations."""

    def find_path(
        self,
        world_db: WorldDatabase,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """Find shortest path between two entities.

        Args:
            world_db: WorldDatabase instance.
            source_id: Source entity ID.
            target_id: Target entity ID.

        Returns:
            List of entity IDs forming the path, or None if no path.
        """
        return world_db.find_path(source_id, target_id)

    def get_connected_entities(
        self,
        world_db: WorldDatabase,
        entity_id: str,
        max_depth: int = 2,
    ) -> list[Entity]:
        """Get entities connected to a given entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Central entity ID.
            max_depth: Maximum relationship depth to traverse.

        Returns:
            List of connected Entity objects.
        """
        return world_db.get_connected_entities(entity_id, max_depth=max_depth)

    def get_communities(self, world_db: WorldDatabase) -> list[list[str]]:
        """Detect communities/clusters in the entity graph.

        Args:
            world_db: WorldDatabase instance.

        Returns:
            List of communities, each a list of entity IDs.
        """
        return world_db.get_communities()

    def get_most_connected(
        self,
        world_db: WorldDatabase,
        limit: int = 10,
    ) -> list[tuple[Entity, int]]:
        """Get most connected entities (highest degree centrality).

        Args:
            world_db: WorldDatabase instance.
            limit: Maximum number to return.

        Returns:
            List of (Entity, connection_count) tuples.
        """
        return world_db.get_most_connected(limit=limit)

    def get_context_for_agents(
        self,
        world_db: WorldDatabase,
        max_entities: int = 50,
    ) -> dict[str, Any]:
        """Get world context formatted for AI agents.

        Args:
            world_db: WorldDatabase instance.
            max_entities: Maximum entities per type to include.

        Returns:
            Dictionary with world context (characters, locations, items, etc.).
        """
        return world_db.get_context_for_agents(max_entities=max_entities)

    def get_entity_summary(self, world_db: WorldDatabase) -> dict[str, int]:
        """Get a summary count of entities by type.

        Args:
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
