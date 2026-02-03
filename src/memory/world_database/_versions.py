"""Entity versioning operations for WorldDatabase."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime

from src.memory.entities import EntityVersion

logger = logging.getLogger(__name__)


def save_entity_version_internal(
    cursor: sqlite3.Cursor,
    db,
    entity_id: str,
    change_type: str,
    change_reason: str = "",
    quality_score: float | None = None,
) -> str | None:
    """Internal function to save entity version (called within existing transaction).

    Args:
        cursor: Database cursor from existing transaction.
        db: WorldDatabase instance.
        entity_id: Entity ID to version.
        change_type: Type of change (created, refined, edited, regenerated).
        change_reason: Reason for the change.
        quality_score: Optional quality score at time of change.

    Returns:
        Version ID if successful, None if entity not found.
    """
    # Get current entity state
    cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
    row = cursor.fetchone()
    if row is None:
        return None

    # Get next version number
    cursor.execute(
        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM entity_versions WHERE entity_id = ?",
        (entity_id,),
    )
    version_number = cursor.fetchone()[0]

    # Create version snapshot
    version_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    data_json = json.dumps(
        {
            "type": row["type"],
            "name": row["name"],
            "description": row["description"],
            "attributes": json.loads(row["attributes"]),
        }
    )

    cursor.execute(
        """
        INSERT INTO entity_versions
        (id, entity_id, version_number, data_json, created_at, change_type, change_reason, quality_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            entity_id,
            version_number,
            data_json,
            now,
            change_type,
            change_reason,
            quality_score,
        ),
    )

    # Apply retention policy
    apply_version_retention(entity_id, cursor)

    logger.debug(f"Saved version {version_number} for entity {entity_id} ({change_type})")
    return version_id


def save_entity_version(
    db,
    entity_id: str,
    change_type: str,
    change_reason: str = "",
    quality_score: float | None = None,
) -> str | None:
    """Save current entity state as a new version.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID to version.
        change_type: Type of change (created, refined, edited, regenerated).
        change_reason: Reason for the change.
        quality_score: Optional quality score at time of change.

    Returns:
        Version ID if successful, None if entity not found.

    Raises:
        ValueError: If change_type is invalid.
    """
    valid_change_types = {"created", "refined", "edited", "regenerated"}
    if change_type not in valid_change_types:
        raise ValueError(
            f"Invalid change_type '{change_type}'. "
            f"Must be one of: {', '.join(sorted(valid_change_types))}"
        )

    logger.debug(
        "save_entity_version called: entity_id=%s, change_type=%s",
        entity_id,
        change_type,
    )

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        version_id = save_entity_version_internal(
            cursor, db, entity_id, change_type, change_reason, quality_score
        )
        db.conn.commit()
        return version_id


def get_entity_versions(db, entity_id: str, limit: int | None = None) -> list[EntityVersion]:
    """Get version history for an entity.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID.
        limit: Maximum number of versions to return. Must be positive if provided.

    Returns:
        List of versions, newest first. Empty list if limit <= 0.
    """
    # Guard against invalid limit values
    if limit is not None and limit <= 0:
        logger.debug(f"get_entity_versions called with limit={limit}, returning empty list")
        return []

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()
        if limit is not None:
            cursor.execute(
                """
                SELECT * FROM entity_versions
                WHERE entity_id = ?
                ORDER BY version_number DESC
                LIMIT ?
                """,
                (entity_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM entity_versions
                WHERE entity_id = ?
                ORDER BY version_number DESC
                """,
                (entity_id,),
            )
        return [row_to_entity_version(row) for row in cursor.fetchall()]


def revert_entity_to_version(db, entity_id: str, version_number: int) -> bool:
    """Revert an entity to a previous version.

    Restores the entity to the specified version and creates a new
    'edited' version recording the revert.

    Args:
        db: WorldDatabase instance.
        entity_id: Entity ID.
        version_number: Version number to revert to.

    Returns:
        True if reverted successfully.

    Raises:
        ValueError: If version not found.
    """
    logger.info(f"Reverting entity {entity_id} to version {version_number}")

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

        # Get the target version
        cursor.execute(
            """
            SELECT * FROM entity_versions
            WHERE entity_id = ? AND version_number = ?
            """,
            (entity_id, version_number),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Version {version_number} not found for entity {entity_id}")

        # Parse version data
        version_data = json.loads(row["data_json"])

        # Update entity with version data
        cursor.execute(
            """
            UPDATE entities
            SET type = ?,
                name = ?,
                description = ?,
                attributes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                version_data["type"],
                version_data["name"],
                version_data["description"],
                json.dumps(version_data.get("attributes", {})),
                datetime.now().isoformat(),
                entity_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Entity {entity_id} not found")

        # Save new version recording the revert in the same transaction
        save_entity_version_internal(
            cursor,
            db,
            entity_id,
            "edited",
            f"Reverted to version {version_number}",
        )
        db.conn.commit()

    # Update graph after commit and outside the lock
    from . import _graph

    _graph.update_entity_in_graph(
        db,
        entity_id,
        version_data["name"],
        version_data["type"],
        version_data["description"],
        version_data.get("attributes"),
    )

    logger.info(f"Entity {entity_id} reverted to version {version_number}")
    return True


def apply_version_retention(entity_id: str, cursor: sqlite3.Cursor) -> None:
    """Apply version retention policy, keeping only last N versions plus initial.

    Always preserves version 1 (initial creation snapshot) for audit trail.

    Args:
        entity_id: Entity ID.
        cursor: Database cursor.

    Raises:
        Exception: If Settings.load() fails (fail-fast on misconfiguration).
    """
    # Import here to avoid circular import
    from src.settings import Settings

    # Fail fast if settings can't be loaded - don't silently use defaults
    settings = Settings.load()
    retention_limit = settings.entity_version_retention

    # Count versions (excluding version 1 which is always preserved)
    cursor.execute(
        "SELECT COUNT(*) FROM entity_versions WHERE entity_id = ? AND version_number != 1",
        (entity_id,),
    )
    count = cursor.fetchone()[0]

    if count > retention_limit:
        # Delete oldest versions beyond limit, but always preserve version 1
        cursor.execute(
            """
            DELETE FROM entity_versions
            WHERE entity_id = ?
            AND version_number != 1
            AND id NOT IN (
                SELECT id FROM entity_versions
                WHERE entity_id = ?
                AND version_number != 1
                ORDER BY version_number DESC
                LIMIT ?
            )
            """,
            (entity_id, entity_id, retention_limit),
        )
        deleted = cursor.rowcount
        if deleted > 0:
            logger.debug(
                f"Deleted {deleted} old versions for entity {entity_id} "
                f"(retention limit: {retention_limit}, version 1 preserved)"
            )


def row_to_entity_version(row: sqlite3.Row) -> EntityVersion:
    """Convert a database row to an EntityVersion."""
    return EntityVersion(
        id=row["id"],
        entity_id=row["entity_id"],
        version_number=row["version_number"],
        data_json=json.loads(row["data_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        change_type=row["change_type"],
        change_reason=row["change_reason"] or "",
        quality_score=row["quality_score"],
    )
