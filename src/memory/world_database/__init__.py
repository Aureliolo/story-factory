"""SQLite-backed worldbuilding database with NetworkX integration."""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from networkx import DiGraph

if TYPE_CHECKING:
    from src.memory.world_settings import WorldSettings

from src.memory.entities import Entity, EntityVersion, EventParticipant, Relationship, WorldEvent
from src.utils.exceptions import DatabaseClosedError, RelationshipValidationError

from . import _entities, _events, _graph, _io, _relationships, _versions

logger = logging.getLogger(__name__)

# Schema version for migration support
SCHEMA_VERSION = 4

# Valid entity types (whitelist)
VALID_ENTITY_TYPES = frozenset({"character", "location", "item", "faction", "concept"})

# Allowed fields for entity updates (SQL injection prevention)
ENTITY_UPDATE_FIELDS = frozenset({"name", "description", "attributes", "type", "updated_at"})

# Attributes constraints
MAX_ATTRIBUTES_DEPTH = 3
MAX_ATTRIBUTES_SIZE_BYTES = 10 * 1024  # 10KB


def flatten_deep_attributes(
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
        except TypeError, ValueError:
            logger.warning(
                "Failed to JSON-serialize attributes at max depth; falling back to str()"
            )
            return str(obj)

    if isinstance(obj, dict):
        return {k: flatten_deep_attributes(v, max_depth, current_depth + 1) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [flatten_deep_attributes(item, max_depth, current_depth + 1) for item in obj]
    return obj


def check_nesting_depth(obj: Any, max_depth: int, current_depth: int = 0) -> bool:
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
        return any(check_nesting_depth(v, max_depth, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        return any(check_nesting_depth(item, max_depth, current_depth + 1) for item in obj)
    return False


def validate_and_normalize_attributes(
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
    if check_nesting_depth(attrs, max_depth, current_depth=1):
        logger.warning(
            f"Attributes exceed maximum nesting depth of {max_depth}, flattening deep structures"
        )
        attrs = flatten_deep_attributes(attrs, max_depth, current_depth=1)

    # Check size (after flattening)
    attrs_json = json.dumps(attrs)
    if len(attrs_json.encode("utf-8")) > MAX_ATTRIBUTES_SIZE_BYTES:
        raise ValueError(f"Attributes exceed maximum size of {MAX_ATTRIBUTES_SIZE_BYTES // 1024}KB")

    return attrs


class WorldDatabase:
    """SQLite-backed worldbuilding database with NetworkX integration.

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
        """
        Apply incremental schema migrations to bring the database from from_version up to to_version.

        Performs in-place schema changes and data migrations as needed:
        - v1 -> v2: creates the entity_versions table and indexes.
        - v2 -> v3: creates world_settings and historical_eras tables and indexes.
        - v3 -> v4: if a relationships table exists, normalizes legacy free-form relationship types (may update rows).

        Parameters:
            cursor (sqlite3.Cursor): Database cursor used to execute migration SQL statements.
            from_version (int): Current schema version stored in the database.
            to_version (int): Target schema version to migrate to.
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

        # Migration from v2 to v3: Add world_settings and historical_eras tables
        if from_version < 3:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS world_settings (
                    id TEXT PRIMARY KEY,
                    calendar_json TEXT,
                    timeline_start_year INTEGER,
                    timeline_end_year INTEGER,
                    validate_temporal_consistency INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_eras (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    start_year INTEGER NOT NULL,
                    end_year INTEGER,
                    description TEXT DEFAULT '',
                    display_order INTEGER DEFAULT 0
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_historical_eras_start ON historical_eras(start_year)"
            )
            logger.info("Migration v2->v3: Added world_settings and historical_eras tables")

        # Migration from v3 to v4: Normalize legacy free-form relationship types
        if from_version < 4:
            # Only normalize if the relationships table already exists
            # (new databases create it after migrations run)
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'"
            )
            if cursor.fetchone():
                from src.memory.conflict_types import normalize_relation_type

                cursor.execute("SELECT id, relation_type FROM relationships")
                rows = cursor.fetchall()
                updated = 0
                for row_id, raw_type in rows:
                    normalized = normalize_relation_type(raw_type)
                    if normalized != raw_type:
                        cursor.execute(
                            "UPDATE relationships SET relation_type = ? WHERE id = ?",
                            (normalized, row_id),
                        )
                        updated += 1
                logger.info(
                    "Migration v3->v4: Normalized %d/%d relationship types",
                    updated,
                    len(rows),
                )
            else:
                logger.info("Migration v3->v4: Skipped (relationships table not yet created)")

        logger.info("Database migration complete")

    def _ensure_open(self) -> None:
        """Check that the database connection is still open.

        Raises:
            DatabaseClosedError: If the database has been closed.
        """
        if self._closed:
            raise DatabaseClosedError(f"Database connection is closed: {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self.conn and not self._closed:
                self.conn.close()
                self._closed = True
                logger.debug(f"Database connection closed: {self.db_path}")

    def __enter__(self) -> WorldDatabase:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False  # Don't suppress exceptions

    # =========================================================================
    # Entity CRUD Operations (delegated to _entities)
    # =========================================================================

    def add_entity(
        self,
        entity_type: str,
        name: str,
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> str:
        return _entities.add_entity(self, entity_type, name, description, attributes)

    def get_entity(self, entity_id: str) -> Entity | None:
        return _entities.get_entity(self, entity_id)

    def get_entity_by_name(self, name: str, entity_type: str | None = None) -> Entity | None:
        return _entities.get_entity_by_name(self, name, entity_type)

    def update_entity(self, entity_id: str, **updates: Any) -> bool:
        return _entities.update_entity(self, entity_id, **updates)

    def delete_entity(self, entity_id: str) -> bool:
        return _entities.delete_entity(self, entity_id)

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        return _entities.list_entities(self, entity_type)

    def count_entities(self, entity_type: str | None = None) -> int:
        return _entities.count_entities(self, entity_type)

    def search_entities(self, query: str, entity_type: str | None = None) -> list[Entity]:
        return _entities.search_entities(self, query, entity_type)

    # =========================================================================
    # Entity Versioning Operations (delegated to _versions)
    # =========================================================================

    def save_entity_version(
        self,
        entity_id: str,
        change_type: str,
        change_reason: str = "",
        quality_score: float | None = None,
    ) -> str | None:
        return _versions.save_entity_version(
            self, entity_id, change_type, change_reason, quality_score
        )

    def get_entity_versions(self, entity_id: str, limit: int | None = None) -> list[EntityVersion]:
        return _versions.get_entity_versions(self, entity_id, limit)

    def revert_entity_to_version(self, entity_id: str, version_number: int) -> bool:
        return _versions.revert_entity_to_version(self, entity_id, version_number)

    # =========================================================================
    # Relationship CRUD Operations (delegated to _relationships)
    # =========================================================================

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
        return _relationships.add_relationship(
            self,
            source_id,
            target_id,
            relation_type,
            description,
            strength,
            bidirectional,
            attributes,
            validate,
        )

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[Relationship]:
        return _relationships.get_relationships(self, entity_id, direction)

    def get_relationship_between(self, source_id: str, target_id: str) -> Relationship | None:
        return _relationships.get_relationship_between(self, source_id, target_id)

    def delete_relationship(self, rel_id: str) -> bool:
        return _relationships.delete_relationship(self, rel_id)

    def update_relationship(
        self,
        relationship_id: str,
        relation_type: str | None = None,
        description: str | None = None,
        strength: float | None = None,
        bidirectional: bool | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        return _relationships.update_relationship(
            self, relationship_id, relation_type, description, strength, bidirectional, attributes
        )

    def list_relationships(self) -> list[Relationship]:
        return _relationships.list_relationships(self)

    # =========================================================================
    # Event Management (delegated to _events)
    # =========================================================================

    def add_event(
        self,
        description: str,
        participants: list[tuple[str, str]] | None = None,
        chapter_number: int | None = None,
        timestamp_in_story: str = "",
        consequences: list[str] | None = None,
    ) -> str:
        return _events.add_event(
            self, description, participants, chapter_number, timestamp_in_story, consequences
        )

    def get_events_for_entity(self, entity_id: str) -> list[WorldEvent]:
        return _events.get_events_for_entity(self, entity_id)

    def get_events_for_chapter(self, chapter_number: int) -> list[WorldEvent]:
        return _events.get_events_for_chapter(self, chapter_number)

    def get_event_participants(self, event_id: str) -> list[EventParticipant]:
        return _events.get_event_participants(self, event_id)

    def list_events(self, limit: int | None = None) -> list[WorldEvent]:
        return _events.list_events(self, limit)

    # =========================================================================
    # NetworkX Graph Operations (delegated to _graph)
    # =========================================================================

    def _invalidate_graph(self) -> None:
        _graph.invalidate_graph(self)

    def invalidate_graph_cache(self) -> None:
        _graph.invalidate_graph_cache(self)

    def _update_entity_in_graph(
        self,
        entity_id: str,
        name: str | None = None,
        entity_type: str | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        _graph.update_entity_in_graph(self, entity_id, name, entity_type, description, attributes)

    def get_graph(self) -> DiGraph[Any]:
        return _graph.get_graph(self)

    def find_path(self, source_id: str, target_id: str) -> list[str]:
        return _graph.find_path(self, source_id, target_id)

    def find_all_paths(
        self, source_id: str, target_id: str, max_length: int = 5
    ) -> list[list[str]]:
        return _graph.find_all_paths(self, source_id, target_id, max_length)

    def get_connected_entities(self, entity_id: str, max_depth: int = 2) -> list[Entity]:
        return _graph.get_connected_entities(self, entity_id, max_depth)

    def get_communities(self) -> list[list[str]]:
        return _graph.get_communities(self)

    def get_entity_centrality(self) -> dict[str, float]:
        return _graph.get_entity_centrality(self)

    def get_most_connected(self, limit: int = 10) -> list[tuple[Entity, int]]:
        return _graph.get_most_connected(self, limit)

    def find_orphans(self, entity_type: str | None = None) -> list[Entity]:
        return _graph.find_orphans(self, entity_type)

    def find_circular_relationships(
        self,
        relation_types: list[str] | None = None,
        max_cycle_length: int = 10,
    ) -> list[list[tuple[str, str, str]]]:
        return _graph.find_circular_relationships(self, relation_types, max_cycle_length)

    # =========================================================================
    # Context for Agents (delegated to _relationships)
    # =========================================================================

    def get_context_for_agents(self, max_entities: int = 50) -> dict[str, Any]:
        return _relationships.get_context_for_agents(self, max_entities)

    def _get_important_relationships(self, limit: int = 30) -> list[dict[str, Any]]:
        return _relationships.get_important_relationships(self, limit)

    # =========================================================================
    # Export/Import (delegated to _io)
    # =========================================================================

    def export_to_json(self) -> dict[str, Any]:
        return _io.export_to_json(self)

    def import_from_json(self, data: dict[str, Any]) -> None:
        _io.import_from_json(self, data)

    # =========================================================================
    # World Settings (calendar, timeline config)
    # =========================================================================

    def get_world_settings(self) -> WorldSettings | None:
        """Get world settings including calendar configuration.

        Returns:
            WorldSettings instance or None if not configured.
        """
        # Import here to avoid circular imports
        from src.memory.world_settings import WorldSettings

        logger.debug("Loading world settings from database")
        with self._lock:
            self._ensure_open()
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, calendar_json, timeline_start_year, timeline_end_year,
                       validate_temporal_consistency, created_at, updated_at
                FROM world_settings
                LIMIT 1
                """
            )
            row = cursor.fetchone()

        if not row:
            logger.debug("No world settings found in database")
            return None

        calendar_data = json.loads(row[1]) if row[1] else None
        settings_data = {
            "id": row[0],
            "calendar": calendar_data,
            "timeline_start_year": row[2],
            "timeline_end_year": row[3],
            "validate_temporal_consistency": bool(row[4]),
            "created_at": row[5],
            "updated_at": row[6],
        }
        logger.debug(
            f"Loaded world settings: id={row[0]}, has_calendar={calendar_data is not None}"
        )
        return WorldSettings.from_dict(settings_data)

    def save_world_settings(self, settings: WorldSettings) -> None:
        """Save or update world settings.

        Args:
            settings: WorldSettings instance to save.
        """
        from datetime import datetime

        logger.debug(f"Saving world settings: id={settings.id}")
        calendar_json = json.dumps(settings.calendar.to_dict()) if settings.calendar else None
        now = datetime.now().isoformat()

        with self._lock:
            self._ensure_open()
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO world_settings
                (id, calendar_json, timeline_start_year, timeline_end_year,
                 validate_temporal_consistency, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    settings.id,
                    calendar_json,
                    settings.timeline_start_year,
                    settings.timeline_end_year,
                    int(settings.validate_temporal_consistency),
                    settings.created_at.isoformat()
                    if hasattr(settings.created_at, "isoformat")
                    else now,
                    now,
                ),
            )
            self.conn.commit()
        logger.info(f"Saved world settings: id={settings.id}")


# Re-export for backward compatibility
__all__ = [
    "ENTITY_UPDATE_FIELDS",
    "MAX_ATTRIBUTES_DEPTH",
    "MAX_ATTRIBUTES_SIZE_BYTES",
    "SCHEMA_VERSION",
    "VALID_ENTITY_TYPES",
    "Entity",
    "EntityVersion",
    "EventParticipant",
    "Relationship",
    "RelationshipValidationError",
    "WorldDatabase",
    "WorldEvent",
    "check_nesting_depth",
    "flatten_deep_attributes",
    "validate_and_normalize_attributes",
]
