"""Database schema initialization and migration for WorldDatabase."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import WorldDatabase

logger = logging.getLogger(__name__)


def init_schema(db: WorldDatabase) -> None:
    """Initialize database schema with versioning.

    Creates all tables if they don't exist and handles version upgrades.
    Currently supports migration from v5 (no accepted_cycles) to v6.

    Args:
        db: WorldDatabase instance.
    """
    from . import SCHEMA_VERSION

    with db._lock:
        db._ensure_open()
        cursor = db.conn.cursor()

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

        # World settings table
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

        # Embedding metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                source_id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_metadata_type "
            "ON embedding_metadata(content_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_metadata_model "
            "ON embedding_metadata(embedding_model)"
        )

        # vec_embeddings virtual table (sqlite-vec is mandatory)
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[1024],
                content_type text partition key,
                +source_id text,
                +entity_type text,
                +chapter_number integer,
                +display_text text,
                +embedding_model text,
                +embedded_at text
            )
            """
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

        # Accepted cycles table (circular chains marked as intentional)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accepted_cycles (
                cycle_hash TEXT PRIMARY KEY,
                accepted_at TEXT NOT NULL
            )
        """
        )

        # Indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")

        # Set schema version (stamped last so crash-retries re-run migrations)
        if current_version < SCHEMA_VERSION:
            logger.info(
                "Upgrading database schema from version %d to %d: %s",
                current_version,
                SCHEMA_VERSION,
                db.db_path,
            )
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

        db.conn.commit()
        logger.debug(f"Database schema initialized: {db.db_path}")
