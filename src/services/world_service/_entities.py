"""Entity CRUD mixin for WorldService."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from src.memory.entities import Entity
from src.memory.world_database import WorldDatabase

from ._base import WorldServiceBase

logger = logging.getLogger(__name__)


class EntityMixin(WorldServiceBase):
    """Mixin providing entity CRUD operations."""

    def add_entity(
        self,
        world_db: WorldDatabase,
        entity_type: str,
        name: str,
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add a new entity to the world.

        Args:
            world_db: WorldDatabase instance.
            entity_type: Type of entity.
            name: Entity name.
            description: Entity description.
            attributes: Optional attributes dictionary.

        Returns:
            The new entity's ID.
        """
        logger.info(f"Adding {entity_type} entity: {name}")
        entity_id = world_db.add_entity(
            entity_type=entity_type,
            name=name,
            description=description,
            attributes=attributes or {},
        )
        logger.debug(f"Created entity {entity_id}")
        return entity_id

    def update_entity(
        self,
        world_db: WorldDatabase,
        entity_id: str,
        name: str | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID to update.
            name: New name (optional).
            description: New description (optional).
            attributes: New attributes (optional, merged with existing).

        Returns:
            True if updated, False if not found.
        """
        logger.info(f"Updating entity {entity_id}" + (f" -> {name}" if name else ""))
        # Only pass non-None values to avoid overwriting with None
        update_kwargs: dict[str, Any] = {}
        if name is not None:
            update_kwargs["name"] = name
        if description is not None:
            update_kwargs["description"] = description
        if attributes is not None:
            update_kwargs["attributes"] = attributes
        result = world_db.update_entity(entity_id=entity_id, **update_kwargs)
        if result:
            logger.debug(f"Entity {entity_id} updated successfully")
        else:
            logger.warning(f"Entity {entity_id} not found for update")
        return result

    def delete_entity(self, world_db: WorldDatabase, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        logger.info(f"Deleting entity {entity_id}")
        result = world_db.delete_entity(entity_id)
        if result:
            logger.debug(f"Entity {entity_id} deleted")
        else:
            logger.warning(f"Entity {entity_id} not found for deletion")
        return result

    def get_entity(self, world_db: WorldDatabase, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.

        Returns:
            Entity or None if not found.
        """
        return world_db.get_entity(entity_id)

    def get_entity_versions(
        self, world_db: WorldDatabase, entity_id: str, limit: int | None = None
    ) -> list:
        """Get version history for an entity.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.
            limit: Maximum number of versions to return.

        Returns:
            List of EntityVersion objects, newest first.
        """
        logger.debug(f"get_entity_versions called: entity_id={entity_id}, limit={limit}")
        versions = world_db.get_entity_versions(entity_id, limit=limit)
        logger.debug(f"Found {len(versions)} versions for entity {entity_id}")
        return versions

    def revert_entity_to_version(
        self, world_db: WorldDatabase, entity_id: str, version_number: int
    ) -> bool:
        """Revert an entity to a previous version.

        Args:
            world_db: WorldDatabase instance.
            entity_id: Entity ID.
            version_number: Version number to revert to.

        Returns:
            True if reverted successfully.

        Raises:
            ValueError: If version not found or entity not found.
        """
        logger.info(f"Reverting entity {entity_id} to version {version_number}")
        result = world_db.revert_entity_to_version(entity_id, version_number)
        if result:
            logger.info(f"Entity {entity_id} reverted to version {version_number}")
        return result

    def list_entities(
        self,
        world_db: WorldDatabase,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """List all entities, optionally filtered by type.

        Args:
            world_db: WorldDatabase instance.
            entity_type: Optional type filter.

        Returns:
            List of Entity objects.
        """
        logger.debug(f"list_entities called: entity_type={entity_type}")
        entities = world_db.list_entities(entity_type=entity_type)
        logger.debug(
            f"Found {len(entities)} entities" + (f" of type {entity_type}" if entity_type else "")
        )
        return entities

    def search_entities(
        self,
        world_db: WorldDatabase,
        query: str,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Search entities by name or description.

        Args:
            world_db: WorldDatabase instance.
            query: Search query.
            entity_type: Optional type filter.

        Returns:
            List of matching Entity objects.
        """
        logger.debug(f"search_entities called: query={query}, entity_type={entity_type}")
        results = world_db.search_entities(query, entity_type=entity_type)
        logger.debug(f"Found {len(results)} entities matching '{query}'")
        return results

    def find_entity_by_name(
        self,
        world_db: WorldDatabase,
        name: str,
        entity_type: str | None = None,
        fuzzy_threshold: float = 0.8,
    ) -> Entity | None:
        """Find an entity by name with fuzzy matching support.

        Attempts exact match first, then case-insensitive, then fuzzy matching
        if threshold is met.

        Args:
            world_db: WorldDatabase instance.
            name: Entity name to find.
            entity_type: Optional type filter.
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0).
                Lower values allow more lenient matching. Default 0.8.

        Returns:
            Entity if found, None otherwise.
        """
        # Normalize and validate input early
        normalized_name = name.strip().lower() if name else ""
        if not normalized_name:
            logger.debug("find_entity_by_name called with empty/whitespace name, returning None")
            return None

        # Clamp fuzzy_threshold to valid range
        fuzzy_threshold = max(0.0, min(1.0, fuzzy_threshold))

        logger.debug(
            f"find_entity_by_name called: name='{normalized_name}', "
            f"type={entity_type}, threshold={fuzzy_threshold}"
        )

        # Exact case-insensitive match (uses world_db's get_entity_by_name)
        # Pass normalized name for consistent lookup
        entity = world_db.get_entity_by_name(normalized_name, entity_type=entity_type)
        if entity:
            logger.debug(f"Found exact match for '{normalized_name}': {entity.id}")
            return entity

        # Fuzzy matching - get all entities of the specified type
        all_entities = world_db.list_entities(entity_type=entity_type)
        if not all_entities:
            logger.debug("No entities found for fuzzy matching")
            return None

        best_match: Entity | None = None
        best_score = 0.0

        for entity in all_entities:
            entity_name = entity.name.lower().strip()

            # Calculate similarity score using sequence matcher
            score = self._calculate_name_similarity(normalized_name, entity_name)

            if score > best_score and score >= fuzzy_threshold:
                best_score = score
                best_match = entity

        if best_match:
            logger.debug(
                f"Found fuzzy match for '{normalized_name}': {best_match.name} "
                f"(score={best_score:.2f})"
            )
        else:
            logger.debug(
                f"No fuzzy match found for '{normalized_name}' above threshold {fuzzy_threshold}"
            )

        return best_match

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity score between two names.

        Uses a combination of:
        - Exact match: 1.0
        - Prefix/suffix match: 0.9
        - Substring match: 0.85
        - Character-based similarity: SequenceMatcher ratio

        Args:
            name1: First name (already lowercase).
            name2: Second name (already lowercase).

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Exact match
        if name1 == name2:
            return 1.0

        # Prefix or suffix match
        if name1.startswith(name2) or name2.startswith(name1):
            return 0.9
        if name1.endswith(name2) or name2.endswith(name1):
            return 0.9

        # Substring match
        if name1 in name2 or name2 in name1:
            return 0.85

        # Character-based similarity using difflib
        return SequenceMatcher(None, name1, name2).ratio()
