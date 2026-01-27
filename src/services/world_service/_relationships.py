"""Relationship management mixin for WorldService."""

from __future__ import annotations

import logging

from src.memory.entities import Relationship
from src.memory.world_database import WorldDatabase

from ._base import WorldServiceBase

logger = logging.getLogger(__name__)


class RelationshipMixin(WorldServiceBase):
    """Mixin providing relationship management operations."""

    def add_relationship(
        self,
        world_db: WorldDatabase,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        bidirectional: bool = False,
    ) -> str:
        """Add a relationship between entities.

        Args:
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
            validate=self.settings.relationship_validation_enabled,
        )
        logger.debug(f"Created relationship {rel_id}")
        return rel_id

    def delete_relationship(self, world_db: WorldDatabase, relationship_id: str) -> bool:
        """Delete a relationship.

        Args:
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
        self,
        world_db: WorldDatabase,
        entity_id: str | None = None,
    ) -> list[Relationship]:
        """Get relationships, optionally filtered by entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Optional entity ID to filter by.

        Returns:
            List of Relationship objects.
        """
        if entity_id:
            return world_db.get_relationships(entity_id)
        return world_db.list_relationships()
