"""Base class and initialization for WorldDatabase.

Contains core database setup, schema management, and connection handling.
"""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Literal

from networkx import DiGraph

logger = logging.getLogger(__name__)

# Schema version for migration support
SCHEMA_VERSION = 2

# Valid entity types (whitelist)
VALID_ENTITY_TYPES = frozenset({"character", "location", "item", "faction", "concept"})

# Allowed fields for entity updates (SQL injection prevention)
ENTITY_UPDATE_FIELDS = frozenset({"name", "description", "attributes", "type", "updated_at"})

# Attributes constraints
MAX_ATTRIBUTES_DEPTH = 3
MAX_ATTRIBUTES_SIZE_BYTES = 10 * 1024  # 10KB


def _flatten_deep_attributes(
    obj: Any, max_depth: int = MAX_ATTRIBUTES_DEPTH, current_depth: int = 0
) -> Any:
    """Flatten attributes that exceed max nesting depth.

    Converts deeply nested structures to JSON string representations at the max depth
    to preserve data while meeting storage constraints.

    Args:
        obj: Object to potentially flatten.
        max_depth: Maximum nesting depth allowed.
        current_depth: Current depth in the recursion.

    Returns:
        Flattened object with nested structures converted to strings at max depth.
    """
    if current_depth >= max_depth:
        # At max depth, convert complex types to string representation
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Use deterministic JSON representation for nested structures
        logger.debug(f"Flattening nested {type(obj).__name__} at depth {current_depth}")
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            logger.warning(
                "Failed to JSON-serialize attributes at max depth; falling back to str()"
            )
            return str(obj)

    if isinstance(obj, dict):
        return {
            k: _flatten_deep_attributes(v, max_depth, current_depth + 1) for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_flatten_deep_attributes(item, max_depth, current_depth + 1) for item in obj]
    return obj


def _check_nesting_depth(obj: Any, max_depth: int, current_depth: int = 0) -> bool:
    """Check if object exceeds maximum nesting depth.

    Args:
        obj: Object to check.
        max_depth: Maximum allowed nesting depth.
        current_depth: Current depth in the recursion.

    Returns:
        True if object exceeds max depth, False otherwise.
    """
    if current_depth > max_depth:
        return True
    if isinstance(obj, dict):
        return any(_check_nesting_depth(v, max_depth, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        return any(_check_nesting_depth(item, max_depth, current_depth + 1) for item in obj)
    return False


def _validate_and_normalize_attributes(
    attrs: dict[str, Any], max_depth: int = MAX_ATTRIBUTES_DEPTH
) -> dict[str, Any]:
    """Validate and normalize attributes dict.

    Flattens deeply nested structures and validates size constraints.

    Args:
        attrs: Attributes dictionary to validate.
        max_depth: Maximum nesting depth allowed.

    Returns:
        Normalized attributes dict with deep nesting flattened.

    Raises:
        ValueError: If attributes exceed size limit.
    """
    # Check if flattening is needed and flatten
    if _check_nesting_depth(attrs, max_depth, current_depth=1):
        logger.warning(
            f"Attributes exceed maximum nesting depth of {max_depth}, flattening deep structures"
        )
        attrs = _flatten_deep_attributes(attrs, max_depth, current_depth=1)

    # Check size (after flattening)
    attrs_json = json.dumps(attrs)
    if len(attrs_json.encode("utf-8")) > MAX_ATTRIBUTES_SIZE_BYTES:
        raise ValueError(f"Attributes exceed maximum size of {MAX_ATTRIBUTES_SIZE_BYTES // 1024}KB")

    return attrs


class WorldDatabaseBase:
    """Base class for WorldDatabase with initialization and connection management.

    Thread-safe implementation using RLock for all database operations.
    """

    def __init__(self, db_path: Path | str):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety lock
        self._lock = threading.RLock()

        # Connection with WAL mode for better concurrency
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")

        self._init_schema()
        self._graph: DiGraph[Any] | None = None
        self._closed = False

    def __del__(self) -> None:
        """Safety net for resource cleanup."""
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception as e:
                # Log but don't raise during garbage collection
                logger.debug("Error during WorldDatabase cleanup in __del__: %s", e)

    def _init_schema(self) -> None:
        """Initialize database schema with versioning."""
        with self._lock:
            cursor = self.conn.cursor()

            # Schema version table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """
            )

            # Check current version
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            row = cursor.fetchone()
            current_version = row[0] if row else 0

            # Core entity table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    attributes TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Entity versions table for tracking changes
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_versions (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    change_type TEXT NOT NULL CHECK(change_type IN ('created', 'refined', 'edited', 'regenerated')),
                    change_reason TEXT DEFAULT '',
                    quality_score REAL DEFAULT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """
            )

            # Indexes for entity versions
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_versions_entity_id ON entity_versions(entity_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_versions_created_at ON entity_versions(created_at)"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_versions_entity_version "
                "ON entity_versions(entity_id, version_number)"
            )

            # Run migrations if needed
            if current_version < SCHEMA_VERSION:
                self._run_migrations(cursor, current_version, SCHEMA_VERSION)
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )

            # Relationships table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    strength REAL DEFAULT 1.0,
                    bidirectional INTEGER DEFAULT 0,
                    attributes TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """
            )

            # Events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    chapter_number INTEGER,
                    timestamp_in_story TEXT DEFAULT '',
                    consequences TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
            """
            )

            # Event participants (many-to-many)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS event_participants (
                    event_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    PRIMARY KEY (event_id, entity_id),
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """
            )

            # Indexes for fast queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")

            self.conn.commit()
            logger.debug(f"Database schema initialized: {self.db_path}")

    def _run_migrations(self, cursor: sqlite3.Cursor, from_version: int, to_version: int) -> None:
        """Run database migrations.

        Args:
            cursor: Database cursor.
            from_version: Current schema version.
            to_version: Target schema version.
        """
        logger.info(f"Migrating database schema from v{from_version} to v{to_version}")

        # Migration from v1 to v2: Add entity_versions table
        if from_version < 2:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_versions (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    change_type TEXT NOT NULL CHECK(change_type IN ('created', 'refined', 'edited', 'regenerated')),
                    change_reason TEXT DEFAULT '',
                    quality_score REAL DEFAULT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_versions_entity_id ON entity_versions(entity_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_versions_created_at ON entity_versions(created_at)"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_versions_entity_version "
                "ON entity_versions(entity_id, version_number)"
            )
            logger.info("Migration v1->v2: Added entity_versions table")

        logger.info("Database migration complete")

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self.conn and not self._closed:
                self.conn.close()
                self._closed = True
                logger.debug(f"Database connection closed: {self.db_path}")

    def __enter__(self) -> WorldDatabaseBase:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False  # Don't suppress exceptions
