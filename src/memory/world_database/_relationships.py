"""Relationship CRUD operations and context extraction for WorldDatabase."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.memory.entities import Relationship
from src.utils.exceptions import RelationshipValidationError

if TYPE_CHECKING:
    from . import WorldDatabase

logger = logging.getLogger("src.memory.world_database._relationships")


def add_relationship(
    db: WorldDatabase,
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
        db: WorldDatabase instance.
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
    with db._lock:
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
            source_entity = db.get_entity(source_id)
            if source_entity is None:
                raise RelationshipValidationError(
                    f"Source entity with ID '{source_id}' does not exist",
                    source_id=source_id,
                    target_id=target_id,
                    reason="source_not_found",
                    suggestions=["Verify the source entity ID is correct"],
                )

            # Check target entity exists
            target_entity = db.get_entity(target_id)
            if target_entity is None:
                raise RelationshipValidationError(
                    f"Target entity with ID '{target_id}' does not exist",
                    source_id=source_id,
                    target_id=target_id,
                    reason="target_not_found",
                    suggestions=["Verify the target entity ID is correct"],
                )
        cursor = db.conn.cursor()
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
        db.conn.commit()
        from . import _graph

        _graph.add_relationship_to_graph(
            db, rel_id, source_id, target_id, relation_type, description, strength, bidirectional
        )

    logger.debug(f"Added relationship: {source_id} --{relation_type}--> {target_id}")
    return rel_id


def get_relationships(
    db: WorldDatabase, entity_id: str, direction: str = "both"
) -> list[Relationship]:
    """Get relationships for an entity.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID
        direction: "outgoing", "incoming", or "both"

    Returns:
        List of relationships
    """
    with db._lock:
        cursor = db.conn.cursor()

        if direction == "outgoing":
            cursor.execute("SELECT * FROM relationships WHERE source_id = ?", (entity_id,))
        elif direction == "incoming":
            cursor.execute("SELECT * FROM relationships WHERE target_id = ?", (entity_id,))
        else:
            cursor.execute(
                "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )

        return [row_to_relationship(row) for row in cursor.fetchall()]


def get_relationship_between(
    db: WorldDatabase, source_id: str, target_id: str
) -> Relationship | None:
    """Get relationship between two specific entities.

    Args:
        db: WorldDatabase instance.
        source_id: Source entity ID
        target_id: Target entity ID

    Returns:
        Relationship or None
    """
    with db._lock:
        cursor = db.conn.cursor()
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
        return row_to_relationship(row)


def delete_relationship(db: WorldDatabase, rel_id: str) -> bool:
    """Delete a relationship.

    Args:
        db: WorldDatabase instance.
        rel_id: Relationship ID

    Returns:
        True if deleted, False if not found
    """
    with db._lock:
        cursor = db.conn.cursor()
        # Get relationship info before deletion for graph update
        cursor.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,))
        row = cursor.fetchone()
        if row is None:
            return False

        source_id = row["source_id"]
        target_id = row["target_id"]
        bidirectional = bool(row["bidirectional"])

        cursor.execute("DELETE FROM relationships WHERE id = ?", (rel_id,))
        db.conn.commit()
        from . import _graph

        _graph.remove_relationship_from_graph(db, source_id, target_id, bidirectional)
        return True


def update_relationship(
    db: WorldDatabase,
    relationship_id: str,
    relation_type: str | None = None,
    description: str | None = None,
    strength: float | None = None,
    bidirectional: bool | None = None,
    attributes: dict[str, Any] | None = None,
) -> bool:
    """Update an existing relationship.

    Args:
        db: WorldDatabase instance.
        relationship_id: Relationship ID to update.
        relation_type: New relationship type.
        description: New description.
        strength: New strength value.
        bidirectional: New bidirectional flag.
        attributes: New attributes (merged with existing).

    Returns:
        True if updated, False if not found.
    """
    with db._lock:
        cursor = db.conn.cursor()

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
        db.conn.commit()

        # Update graph: remove old edges and add new ones
        from . import _graph

        _graph.remove_relationship_from_graph(db, source_id, target_id, old_bidir)
        _graph.add_relationship_to_graph(
            db, relationship_id, source_id, target_id, new_type, new_desc, new_strength, new_bidir
        )

        return cursor.rowcount > 0


def list_relationships(db: WorldDatabase) -> list[Relationship]:
    """List all relationships.

    Args:
        db: WorldDatabase instance.

    Returns:
        List of all relationships
    """
    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM relationships ORDER BY created_at")
        return [row_to_relationship(row) for row in cursor.fetchall()]


def row_to_relationship(row: sqlite3.Row) -> Relationship:
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


def get_important_relationships(db: WorldDatabase, limit: int = 30) -> list[dict[str, Any]]:
    """Get most important relationships for context.

    Args:
        db: WorldDatabase instance.
        limit: Maximum relationships to return

    Returns:
        List of relationship dicts
    """
    relationships = list_relationships(db)
    # Sort by strength descending
    relationships.sort(key=lambda r: r.strength, reverse=True)

    result = []
    for rel in relationships[:limit]:
        source = db.get_entity(rel.source_id)
        target = db.get_entity(rel.target_id)
        if source and target:
            result.append(
                {
                    "from": source.name,
                    "to": target.name,
                    "type": rel.relation_type,
                    "description": rel.description,
                }
            )
    return result


def get_context_for_agents(db: WorldDatabase, max_entities: int = 50) -> dict[str, Any]:
    """Get compressed world context for AI agents.

    Args:
        db: WorldDatabase instance.
        max_entities: Maximum entities to include per type

    Returns:
        Dict with world context
    """
    return {
        "characters": [
            {"name": e.name, "description": e.description, "attributes": e.attributes}
            for e in db.list_entities("character")[:20]
        ],
        "locations": [
            {"name": e.name, "description": e.description, "attributes": e.attributes}
            for e in db.list_entities("location")[:15]
        ],
        "items": [
            {"name": e.name, "description": e.description} for e in db.list_entities("item")[:10]
        ],
        "factions": [
            {"name": e.name, "description": e.description} for e in db.list_entities("faction")[:10]
        ],
        "key_relationships": get_important_relationships(db, limit=30),
        "recent_events": [
            {"description": e.description, "chapter": e.chapter_number}
            for e in db.list_events(limit=20)
        ],
        "entity_counts": {
            "characters": db.count_entities("character"),
            "locations": db.count_entities("location"),
            "items": db.count_entities("item"),
            "factions": db.count_entities("faction"),
            "concepts": db.count_entities("concept"),
        },
    }
