"""Entity CRUD operations for WorldDatabase."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.memory.entities import Entity

if TYPE_CHECKING:
    from . import WorldDatabase

logger = logging.getLogger("src.memory.world_database._entities")


def add_entity(
    db: WorldDatabase,
    entity_type: str,
    name: str,
    description: str = "",
    attributes: dict[str, Any] | None = None,
) -> str:
    """Add a new entity to the database.

    Args:
        db: WorldDatabase instance.
        entity_type: Type of entity (character, location, item, faction, concept)
        name: Entity name
        description: Entity description
        attributes: Additional attributes as key-value pairs

    Returns:
        Entity ID

    Raises:
        ValueError: If name, entity_type, or attributes are invalid
    """
    from . import VALID_ENTITY_TYPES, _validate_and_normalize_attributes

    logger.debug("add_entity called: type=%s, name=%s", entity_type, name)
    # Validate inputs
    name = name.strip()
    if not name:
        raise ValueError("Entity name cannot be empty")
    if len(name) > 200:
        raise ValueError("Entity name cannot exceed 200 characters")

    entity_type = entity_type.strip().lower()
    if not entity_type:
        raise ValueError("Entity type cannot be empty")
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity type '{entity_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"
        )

    # Strip whitespace from description for consistency
    description = description.strip() if description else ""
    if len(description) > 5000:
        raise ValueError("Entity description cannot exceed 5000 characters")

    # Validate and normalize attributes (flattens deep nesting)
    attrs = attributes or {}
    if attrs:
        attrs = _validate_and_normalize_attributes(attrs)

    entity_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    attrs_json = json.dumps(attrs)

    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO entities (id, type, name, description, attributes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entity_id, entity_type, name, description, attrs_json, now, now),
        )
        # Save initial version in the same transaction as entity insert
        from . import _versions

        _versions.save_entity_version_internal(cursor, db, entity_id, "created")
        db.conn.commit()
        # Update graph after commit
        from . import _graph

        _graph.add_entity_to_graph(db, entity_id, name, entity_type, description, attrs)

    logger.debug(f"Added entity: {name} ({entity_type}) id={entity_id}")
    return entity_id


def get_entity(db: WorldDatabase, entity_id: str) -> Entity | None:
    """Get an entity by ID.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID

    Returns:
        Entity or None if not found
    """
    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return row_to_entity(row)


def get_entity_by_name(
    db: WorldDatabase, name: str, entity_type: str | None = None
) -> Entity | None:
    """Get an entity by name (case-insensitive).

    Args:
        db: WorldDatabase instance.
        name: Entity name
        entity_type: Optional type filter

    Returns:
        Entity or None if not found
    """
    with db._lock:
        cursor = db.conn.cursor()
        if entity_type:
            cursor.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(?) AND type = ?",
                (name, entity_type),
            )
        else:
            cursor.execute("SELECT * FROM entities WHERE LOWER(name) = LOWER(?)", (name,))
        row = cursor.fetchone()
        if row is None:
            return None
        return row_to_entity(row)


def update_entity(db: WorldDatabase, entity_id: str, **updates: Any) -> bool:
    """Update an entity.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID
        **updates: Fields to update (name, description, attributes, type)

    Returns:
        True if updated, False if entity not found

    Raises:
        ValueError: If validation fails or field names are invalid
    """
    from . import ENTITY_UPDATE_FIELDS, VALID_ENTITY_TYPES, _validate_and_normalize_attributes

    logger.debug("update_entity called: entity_id=%s, fields=%s", entity_id, list(updates.keys()))
    # Filter and validate field names (SQL injection prevention)
    update_fields: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in ENTITY_UPDATE_FIELDS:
            logger.warning(f"Ignoring unknown field in update_entity: {key}")
            continue
        if key == "updated_at":
            continue  # We set this ourselves
        update_fields[key] = value

    if not update_fields:
        return False

    # Validate name if being updated
    if "name" in update_fields:
        name = update_fields["name"].strip()
        if not name:
            raise ValueError("Entity name cannot be empty")
        if len(name) > 200:
            raise ValueError("Entity name cannot exceed 200 characters")
        update_fields["name"] = name

    # Validate description if being updated
    if "description" in update_fields:
        description = update_fields["description"]
        if description is not None:
            description = str(description).strip()
            if len(description) > 5000:
                raise ValueError("Entity description cannot exceed 5000 characters")
            update_fields["description"] = description

    # Validate type if being updated
    if "type" in update_fields:
        entity_type = update_fields["type"].strip().lower()
        if not entity_type:
            raise ValueError("Entity type cannot be empty")
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity type '{entity_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"
            )
        update_fields["type"] = entity_type

    # Validate, normalize (flatten deep nesting), and serialize attributes
    if "attributes" in update_fields:
        attrs = update_fields["attributes"]
        if attrs:
            attrs = _validate_and_normalize_attributes(attrs)
        update_fields["attributes"] = json.dumps(attrs or {})

    update_fields["updated_at"] = datetime.now().isoformat()

    # Build SET clause with explicit field validation (defense in depth)
    set_parts = []
    values = []
    for field in update_fields:
        # Double-check field is in whitelist (already filtered, but defense in depth)
        if field not in ENTITY_UPDATE_FIELDS:  # pragma: no cover
            raise ValueError(f"Invalid field name: {field}")
        set_parts.append(f"{field} = ?")
        values.append(update_fields[field])

    set_clause = ", ".join(set_parts)
    values.append(entity_id)

    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute(
            f"UPDATE entities SET {set_clause} WHERE id = ?",
            values,
        )
        updated = cursor.rowcount > 0

        if updated:
            # Save version in the same transaction as the entity update
            from . import _versions

            _versions.save_entity_version_internal(cursor, db, entity_id, "edited")
        db.conn.commit()

        # Update graph incrementally after commit
        if updated:
            from . import _graph

            _graph.update_entity_in_graph(
                db,
                entity_id,
                update_fields.get("name"),
                update_fields.get("type"),
                update_fields.get("description"),
                json.loads(update_fields["attributes"]) if "attributes" in update_fields else None,
            )

    return updated


def delete_entity(db: WorldDatabase, entity_id: str) -> bool:
    """Delete an entity and its relationships.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID

    Returns:
        True if deleted, False if not found
    """
    logger.debug("delete_entity called: entity_id=%s", entity_id)
    with db._lock:
        cursor = db.conn.cursor()
        cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        db.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            from . import _graph

            _graph.remove_entity_from_graph(db, entity_id)
    return deleted


def list_entities(db: WorldDatabase, entity_type: str | None = None) -> list[Entity]:
    """List all entities, optionally filtered by type.

    Args:
        db: WorldDatabase instance.
        entity_type: Optional type filter

    Returns:
        List of entities
    """
    with db._lock:
        cursor = db.conn.cursor()
        if entity_type:
            cursor.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY name",
                (entity_type,),
            )
        else:
            cursor.execute("SELECT * FROM entities ORDER BY type, name")
        return [row_to_entity(row) for row in cursor.fetchall()]


def count_entities(db: WorldDatabase, entity_type: str | None = None) -> int:
    """Count entities, optionally by type.

    Args:
        db: WorldDatabase instance.
        entity_type: Optional type filter

    Returns:
        Count of entities
    """
    with db._lock:
        cursor = db.conn.cursor()
        if entity_type:
            cursor.execute("SELECT COUNT(*) FROM entities WHERE type = ?", (entity_type,))
        else:
            cursor.execute("SELECT COUNT(*) FROM entities")
        result = cursor.fetchone()
        return int(result[0]) if result else 0


def search_entities(db: WorldDatabase, query: str, entity_type: str | None = None) -> list[Entity]:
    """Search entities by name or description.

    Args:
        db: WorldDatabase instance.
        query: Search query
        entity_type: Optional type filter

    Returns:
        List of matching entities
    """
    with db._lock:
        cursor = db.conn.cursor()
        search_pattern = f"%{query}%"
        if entity_type:
            cursor.execute(
                """
                SELECT * FROM entities
                WHERE (name LIKE ? OR description LIKE ?) AND type = ?
                ORDER BY name
                """,
                (search_pattern, search_pattern, entity_type),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM entities
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY type, name
                """,
                (search_pattern, search_pattern),
            )
        return [row_to_entity(row) for row in cursor.fetchall()]


def row_to_entity(row: sqlite3.Row) -> Entity:
    """Convert a database row to an Entity."""
    return Entity(
        id=row["id"],
        type=row["type"],
        name=row["name"],
        description=row["description"],
        attributes=json.loads(row["attributes"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
