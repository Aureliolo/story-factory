"""Relationship CRUD operations mixin for WorldDatabase."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.memory.entities import Relationship
from src.utils.exceptions import RelationshipValidationError

if TYPE_CHECKING:
    from ._base import WorldDatabaseBase

    _RelationshipMixinBase = WorldDatabaseBase
else:
    _RelationshipMixinBase = object

logger = logging.getLogger(__name__)


class RelationshipMixin(_RelationshipMixinBase):
    """Mixin providing relationship CRUD operations."""

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        strength: float = 1.0,
        bidirectional: bool = False,
        attributes: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> str:
        """Add a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            description: Relationship description
            strength: Relationship strength (0.0-1.0)
            bidirectional: Whether relationship goes both ways
            attributes: Additional attributes
            validate: Whether to validate source/target entities exist (default True)

        Returns:
            Relationship ID

        Raises:
            RelationshipValidationError: If validation is enabled and:
                - source_id or target_id does not exist
                - source_id equals target_id (self-loop)
        """
        logger.debug(
            "add_relationship called: source=%s, target=%s, type=%s, validate=%s",
            source_id,
            target_id,
            relation_type,
            validate,
        )

        rel_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        attrs_json = json.dumps(attributes or {})

        # Use lock for entire operation (validation + insert) to prevent TOCTOU race condition
        with self._lock:
            # Validation logic - must be inside lock to prevent entity deletion between check and insert
            if validate:
                # Check for self-loop (no lock needed for this check)
                if source_id == target_id:
                    raise RelationshipValidationError(
                        f"Cannot create self-referential relationship: entity {source_id} cannot "
                        f"have a relationship with itself",
                        source_id=source_id,
                        target_id=target_id,
                        reason="self_loop",
                        suggestions=["Choose a different target entity"],
                    )

                # Check source entity exists (RLock allows reentrant acquisition)
                source_entity = self.get_entity(source_id)
                if source_entity is None:
                    raise RelationshipValidationError(
                        f"Source entity with ID '{source_id}' does not exist",
                        source_id=source_id,
                        target_id=target_id,
                        reason="source_not_found",
                        suggestions=["Verify the source entity ID is correct"],
                    )

                # Check target entity exists
                target_entity = self.get_entity(target_id)
                if target_entity is None:
                    raise RelationshipValidationError(
                        f"Target entity with ID '{target_id}' does not exist",
                        source_id=source_id,
                        target_id=target_id,
                        reason="target_not_found",
                        suggestions=["Verify the target entity ID is correct"],
                    )
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO relationships
                (id, source_id, target_id, relation_type, description, strength, bidirectional, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_id,
                    source_id,
                    target_id,
                    relation_type,
                    description,
                    strength,
                    1 if bidirectional else 0,
                    attrs_json,
                    now,
                ),
            )
            self.conn.commit()
            self._add_relationship_to_graph(
                rel_id, source_id, target_id, relation_type, description, strength, bidirectional
            )

        logger.debug(f"Added relationship: {source_id} --{relation_type}--> {target_id}")
        return rel_id

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationships
        """
        with self._lock:
            cursor = self.conn.cursor()

            if direction == "outgoing":
                cursor.execute("SELECT * FROM relationships WHERE source_id = ?", (entity_id,))
            elif direction == "incoming":
                cursor.execute("SELECT * FROM relationships WHERE target_id = ?", (entity_id,))
            else:
                cursor.execute(
                    "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
                    (entity_id, entity_id),
                )

            return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def get_relationship_between(self, source_id: str, target_id: str) -> Relationship | None:
        """Get relationship between two specific entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            Relationship or None
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM relationships
                WHERE (source_id = ? AND target_id = ?)
                   OR (bidirectional = 1 AND source_id = ? AND target_id = ?)
                """,
                (source_id, target_id, target_id, source_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_relationship(row)

    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship.

        Args:
            rel_id: Relationship ID

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            cursor = self.conn.cursor()
            # Get relationship info before deletion for graph update
            cursor.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,))
            row = cursor.fetchone()
            if row is None:
                return False

            source_id = row["source_id"]
            target_id = row["target_id"]
            bidirectional = bool(row["bidirectional"])

            cursor.execute("DELETE FROM relationships WHERE id = ?", (rel_id,))
            self.conn.commit()
            self._remove_relationship_from_graph(source_id, target_id, bidirectional)
            return True

    def update_relationship(
        self,
        relationship_id: str,
        relation_type: str | None = None,
        description: str | None = None,
        strength: float | None = None,
        bidirectional: bool | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing relationship.

        Args:
            relationship_id: Relationship ID to update.
            relation_type: New relationship type.
            description: New description.
            strength: New strength value.
            bidirectional: New bidirectional flag.
            attributes: New attributes (merged with existing).

        Returns:
            True if updated, False if not found.
        """
        with self._lock:
            cursor = self.conn.cursor()

            # Get current relationship
            cursor.execute("SELECT * FROM relationships WHERE id = ?", (relationship_id,))
            row = cursor.fetchone()
            if row is None:
                return False

            source_id = row["source_id"]
            target_id = row["target_id"]
            old_bidir = bool(row["bidirectional"])

            # Prepare updated values
            new_type = relation_type if relation_type is not None else row["relation_type"]
            new_desc = description if description is not None else row["description"]
            new_strength = strength if strength is not None else row["strength"]
            new_bidir = bidirectional if bidirectional is not None else old_bidir

            # Handle attributes merging
            current_attrs = json.loads(row["attributes"]) if row["attributes"] else {}
            if attributes is not None:
                current_attrs.update(attributes)

            cursor.execute(
                """
                UPDATE relationships
                SET relation_type = ?,
                    description = ?,
                    strength = ?,
                    bidirectional = ?,
                    attributes = ?
                WHERE id = ?
                """,
                (
                    new_type,
                    new_desc,
                    new_strength,
                    1 if new_bidir else 0,
                    json.dumps(current_attrs),
                    relationship_id,
                ),
            )
            self.conn.commit()

            # Update graph: remove old edges and add new ones
            self._remove_relationship_from_graph(source_id, target_id, old_bidir)
            self._add_relationship_to_graph(
                relationship_id, source_id, target_id, new_type, new_desc, new_strength, new_bidir
            )

            return cursor.rowcount > 0

    def list_relationships(self) -> list[Relationship]:
        """List all relationships.

        Returns:
            List of all relationships
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM relationships ORDER BY created_at")
            return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def _row_to_relationship(self, row) -> Relationship:
        """Convert a database row to a Relationship."""
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            description=row["description"],
            strength=row["strength"],
            bidirectional=bool(row["bidirectional"]),
            attributes=json.loads(row["attributes"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
