"""Tests for world health accepted cycles and cycle hash computation."""

from unittest.mock import MagicMock

import pytest

from src.memory.world_database import WorldDatabase


class TestCycleHashComputation:
    """Tests for deterministic cycle hash computation."""

    def test_hash_is_deterministic(self):
        """Same edges should produce the same hash."""
        edges = [("A", "leads", "B"), ("B", "leads", "C"), ("C", "leads", "A")]
        hash1 = WorldDatabase.compute_cycle_hash(edges)
        hash2 = WorldDatabase.compute_cycle_hash(edges)
        assert hash1 == hash2

    def test_hash_independent_of_traversal_start(self):
        """Different traversal orders of the same cycle should produce the same hash."""
        # Same cycle, different start points
        edges1 = [("A", "leads", "B"), ("B", "leads", "C"), ("C", "leads", "A")]
        edges2 = [("B", "leads", "C"), ("C", "leads", "A"), ("A", "leads", "B")]
        edges3 = [("C", "leads", "A"), ("A", "leads", "B"), ("B", "leads", "C")]
        hash1 = WorldDatabase.compute_cycle_hash(edges1)
        hash2 = WorldDatabase.compute_cycle_hash(edges2)
        hash3 = WorldDatabase.compute_cycle_hash(edges3)
        assert hash1 == hash2 == hash3

    def test_different_cycles_produce_different_hashes(self):
        """Different cycles should produce different hashes."""
        edges1 = [("A", "leads", "B"), ("B", "leads", "A")]
        edges2 = [("A", "leads", "C"), ("C", "leads", "A")]
        hash1 = WorldDatabase.compute_cycle_hash(edges1)
        hash2 = WorldDatabase.compute_cycle_hash(edges2)
        assert hash1 != hash2

    def test_hash_sensitive_to_relation_type(self):
        """Same entities but different relation types should produce different hashes."""
        edges_leads = [("A", "leads", "B"), ("B", "leads", "A")]
        edges_hates = [("A", "hates", "B"), ("B", "hates", "A")]
        hash_leads = WorldDatabase.compute_cycle_hash(edges_leads)
        hash_hates = WorldDatabase.compute_cycle_hash(edges_hates)
        assert hash_leads != hash_hates

    def test_hash_is_16_chars(self):
        """Hash should be exactly 16 hex characters."""
        edges = [("X", "knows", "Y"), ("Y", "knows", "X")]
        h = WorldDatabase.compute_cycle_hash(edges)
        assert len(h) == 16
        # Verify it's valid hex
        int(h, 16)

    def test_empty_cycle_produces_hash(self):
        """Empty edge list should still produce a valid hash."""
        h = WorldDatabase.compute_cycle_hash([])
        assert len(h) == 16


class TestAcceptedCyclesRoundTrip:
    """Tests for accept_cycle / get_accepted_cycles / remove_accepted_cycle."""

    @pytest.fixture
    def world_db(self, tmp_path):
        """Create and yield a WorldDatabase, closing it after the test."""
        db = WorldDatabase(tmp_path / "test.db")
        yield db
        db.close()

    def test_accept_and_get_cycle(self, world_db):
        """Accepted cycle should appear in get_accepted_cycles."""
        cycle_hash = "abcdef0123456789"
        world_db.accept_cycle(cycle_hash)

        accepted = world_db.get_accepted_cycles()
        assert cycle_hash in accepted

    def test_get_empty_accepted_cycles(self, world_db):
        """Empty database should return empty set."""
        accepted = world_db.get_accepted_cycles()
        assert accepted == set()

    def test_accept_duplicate_is_idempotent(self, world_db):
        """Accepting the same cycle twice should not cause errors."""
        cycle_hash = "1234567890abcdef"
        world_db.accept_cycle(cycle_hash)
        world_db.accept_cycle(cycle_hash)

        accepted = world_db.get_accepted_cycles()
        assert cycle_hash in accepted
        assert len(accepted) == 1

    def test_remove_accepted_cycle(self, world_db):
        """Removed cycle should no longer appear in accepted set."""
        cycle_hash = "fedcba9876543210"
        world_db.accept_cycle(cycle_hash)
        assert cycle_hash in world_db.get_accepted_cycles()

        removed = world_db.remove_accepted_cycle(cycle_hash)
        assert removed is True
        assert cycle_hash not in world_db.get_accepted_cycles()

    def test_remove_nonexistent_cycle_returns_false(self, world_db):
        """Removing a cycle that doesn't exist should return False."""
        removed = world_db.remove_accepted_cycle("0000000000000000")
        assert removed is False

    def test_multiple_accepted_cycles(self, world_db):
        """Multiple different cycles can be accepted."""
        hashes = ["aaaa000000000000", "bbbb000000000000", "cccc000000000000"]
        for h in hashes:
            world_db.accept_cycle(h)

        accepted = world_db.get_accepted_cycles()
        assert len(accepted) == 3
        for h in hashes:
            assert h in accepted

    def test_accept_rejects_invalid_hash(self, world_db):
        """accept_cycle should reject hashes that aren't 16 hex chars."""
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.accept_cycle("tooshort")

        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.accept_cycle("")

        # Right length but not hex
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.accept_cycle("ghijklmnopqrstuv")

    def test_remove_rejects_invalid_hash(self, world_db):
        """remove_accepted_cycle should reject hashes that aren't 16 hex chars."""
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.remove_accepted_cycle("tooshort")

        # Right length but not hex
        with pytest.raises(ValueError, match="Invalid cycle hash"):
            world_db.remove_accepted_cycle("ghijklmnopqrstuv")


class TestHealthMetricsFilterAcceptedCycles:
    """Tests for filtering accepted cycles from health metrics."""

    def test_accepted_cycles_excluded_from_health(self, tmp_path):
        """Accepted cycles should be excluded from health metrics computation."""
        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_filter.db")
        try:
            # Create entities forming a cycle: A -> B -> C -> A
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            id_c = db.add_entity("character", "Carol", "Character C")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_c, "leads", "B leads C")
            db.add_relationship(id_c, id_a, "leads", "C leads A")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            # First: get metrics without accepting the cycle
            metrics_before = get_world_health_metrics(svc, db)
            assert metrics_before.circular_count > 0, "Should detect circular chain"

            # Accept the cycle
            cycles = db.find_circular_relationships()
            assert len(cycles) > 0
            cycle_hash = db.compute_cycle_hash(cycles[0])
            db.accept_cycle(cycle_hash)

            # After accepting: cycle should be filtered out
            metrics_after = get_world_health_metrics(svc, db)
            assert metrics_after.circular_count == 0, "Accepted cycle should be excluded"

        finally:
            db.close()

    def test_unaccepted_cycles_still_shown(self, tmp_path):
        """Cycles that are not accepted should still appear in health metrics."""
        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_partial.db")
        try:
            # Create two separate cycles
            # Cycle 1: A -> B -> A
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            # Cycle 2: C -> D -> C
            id_c = db.add_entity("character", "Carol", "Character C")
            id_d = db.add_entity("character", "Dave", "Character D")
            db.add_relationship(id_c, id_d, "leads", "C leads D")
            db.add_relationship(id_d, id_c, "leads", "D leads C")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            metrics = get_world_health_metrics(svc, db)
            initial_count = metrics.circular_count
            assert initial_count >= 2, "Should detect at least 2 circular chains"

            # Accept only the first cycle
            cycles = db.find_circular_relationships()
            cycle_hash = db.compute_cycle_hash(cycles[0])
            db.accept_cycle(cycle_hash)

            metrics_after = get_world_health_metrics(svc, db)
            assert metrics_after.circular_count == initial_count - 1

        finally:
            db.close()


class TestAcceptHandlerHashConsistency:
    """Test that the accept handler builds hashes consistent with health metrics."""

    def test_hash_from_edge_dicts_matches_raw_tuples(self, tmp_path):
        """Hash computed from edge dicts (UI accept path) must match raw tuple hash
        (health metrics path) for the same cycle."""
        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_consistency.db")
        try:
            # Create a cycle
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            # Get health metrics (which returns dicts with edge data)
            metrics = get_world_health_metrics(svc, db)
            assert metrics.circular_count > 0

            # Simulate what handle_accept_circular does: build tuples from edge dicts
            cycle_dict = metrics.circular_relationships[0]
            edges = cycle_dict["edges"]
            accept_tuples = [(edge["source"], edge["type"], edge["target"]) for edge in edges]
            accept_hash = db.compute_cycle_hash(accept_tuples)

            # Get the raw cycle tuples (as health metrics filtering uses them)
            raw_cycles = db.find_circular_relationships()
            raw_hash = db.compute_cycle_hash(raw_cycles[0])

            # These must match for the accept → filter chain to work
            assert accept_hash == raw_hash, (
                f"Hash mismatch: accept handler produces {accept_hash}, "
                f"health metrics uses {raw_hash}"
            )

            # Verify it actually filters when accepted
            db.accept_cycle(accept_hash)
            metrics_after = get_world_health_metrics(svc, db)
            assert metrics_after.circular_count == 0
        finally:
            db.close()


class TestImportExportAcceptedCycles:
    """Tests for accepted_cycles round-trip through export/import."""

    def test_import_with_accepted_cycles(self, tmp_path):
        """Importing data with accepted_cycles should persist them."""
        db = WorldDatabase(tmp_path / "test_import.db")
        try:
            # Create entity data for a valid export structure
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            # Accept a cycle
            cycle_hash = db.compute_cycle_hash(
                [("Alice", "leads", "Bob"), ("Bob", "leads", "Alice")]
            )
            db.accept_cycle(cycle_hash)

            # Export
            data = db.export_to_json()
            assert cycle_hash in data["accepted_cycles"]

            # Import into fresh database
            new_db = WorldDatabase(tmp_path / "imported.db")
            try:
                new_db.import_from_json(data)
                accepted = new_db.get_accepted_cycles()
                assert cycle_hash in accepted
            finally:
                new_db.close()
        finally:
            db.close()

    def test_import_skips_invalid_cycle_hashes(self, tmp_path):
        """Import should skip invalid cycle hashes (wrong length, non-hex)."""
        db = WorldDatabase(tmp_path / "test_skip.db")
        try:
            valid_hash = "abcdef0123456789"
            data = {
                "entities": [],
                "relationships": [],
                "events": [],
                "accepted_cycles": [
                    valid_hash,
                    "tooshort",
                    "ghijklmnopqrstuv",
                    "",
                ],
            }
            db.import_from_json(data)
            accepted = db.get_accepted_cycles()
            assert accepted == {valid_hash}
        finally:
            db.close()


class TestHealthMetricsFallbackOnAcceptedCyclesError:
    """Tests for graceful fallback when accepted cycles lookup fails."""

    def test_health_metrics_shows_all_cycles_when_accepted_cycles_fails(self, tmp_path):
        """If get_accepted_cycles() raises, health metrics should still show all cycles."""
        from unittest.mock import patch

        from src.services.world_service import WorldService
        from src.services.world_service._health import get_world_health_metrics

        db = WorldDatabase(tmp_path / "test_fallback.db")
        try:
            # Create a cycle
            id_a = db.add_entity("character", "Alice", "Character A")
            id_b = db.add_entity("character", "Bob", "Character B")
            db.add_relationship(id_a, id_b, "leads", "A leads B")
            db.add_relationship(id_b, id_a, "leads", "B leads A")

            settings = MagicMock()
            settings.orphan_detection_enabled = True
            settings.circular_detection_enabled = True
            settings.circular_check_all_types = True
            settings.circular_relationship_types = []
            svc = WorldService(settings)

            # Patch get_accepted_cycles to raise
            with patch.object(db, "get_accepted_cycles", side_effect=RuntimeError("DB corrupted")):
                metrics = get_world_health_metrics(svc, db)
                # Cycles should still be reported (fallback to empty accepted set)
                assert metrics.circular_count > 0
        finally:
            db.close()


class TestSchemaV5ToV6Migration:
    """Tests for schema migration from v5 (no accepted_cycles) to v6."""

    def _create_v5_database(self, db_path):
        """Create a raw SQLite database with v5 schema (no accepted_cycles table).

        This manually creates all tables that existed before v6, setting the
        schema_version to 5. The WorldDatabase constructor should upgrade it to v6.
        """
        import sqlite3

        from src.utils.sqlite_vec_loader import load_vec_extension

        conn = sqlite3.connect(str(db_path))
        load_vec_extension(conn)

        conn.execute("PRAGMA journal_mode=WAL")

        # Schema version table with v5
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version (version) VALUES (5)")

        # Core entities table
        conn.execute(
            """
            CREATE TABLE entities (
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

        # Entity versions table
        conn.execute(
            """
            CREATE TABLE entity_versions (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                change_type TEXT NOT NULL,
                change_reason TEXT DEFAULT '',
                quality_score REAL DEFAULT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
            """
        )

        # Relationships table
        conn.execute(
            """
            CREATE TABLE relationships (
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
        conn.execute(
            """
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                chapter_number INTEGER,
                timestamp_in_story TEXT DEFAULT '',
                consequences TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
            """
        )

        # Event participants
        conn.execute(
            """
            CREATE TABLE event_participants (
                event_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY (event_id, entity_id),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
            """
        )

        # World settings table
        conn.execute(
            """
            CREATE TABLE world_settings (
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

        # Historical eras table
        conn.execute(
            """
            CREATE TABLE historical_eras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                start_year INTEGER NOT NULL,
                end_year INTEGER,
                description TEXT DEFAULT '',
                display_order INTEGER DEFAULT 0
            )
            """
        )

        # Embedding metadata table
        conn.execute(
            """
            CREATE TABLE embedding_metadata (
                source_id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
            """
        )

        # vec_embeddings virtual table
        conn.execute(
            """
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(
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

        conn.commit()
        conn.close()

    def test_v5_database_upgraded_to_v6(self, tmp_path):
        """Opening a v5 database should auto-upgrade to v6 with accepted_cycles table."""
        db_path = tmp_path / "v5_test.db"
        self._create_v5_database(db_path)

        # Open with WorldDatabase — triggers _init_schema() migration
        db = WorldDatabase(db_path)
        try:
            # Verify schema version upgraded to 6
            cursor = db.conn.cursor()
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            version = cursor.fetchone()[0]
            assert version == 6, f"Expected schema version 6, got {version}"

            # Verify accepted_cycles table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='accepted_cycles'"
            )
            table = cursor.fetchone()
            assert table is not None, "accepted_cycles table should exist after migration"

            # Verify the table is functional
            cycle_hash = "abcdef0123456789"
            db.accept_cycle(cycle_hash)
            accepted = db.get_accepted_cycles()
            assert cycle_hash in accepted
        finally:
            db.close()

    def test_v6_database_not_double_upgraded(self, tmp_path):
        """Re-running _init_schema on a v6 database should not lose accepted cycles."""
        db = WorldDatabase(tmp_path / "v6_test.db")
        try:
            # Accept a cycle
            cycle_hash = "1234567890abcdef"
            db.accept_cycle(cycle_hash)
            assert cycle_hash in db.get_accepted_cycles()

            # Re-run _init_schema (simulates reopening the database)
            db._init_schema()

            # Verify accepted cycle is still present
            accepted = db.get_accepted_cycles()
            assert cycle_hash in accepted, "Accepted cycle should survive re-init"

            # Verify schema version is still 6
            cursor = db.conn.cursor()
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            version = cursor.fetchone()[0]
            assert version == 6
        finally:
            db.close()
