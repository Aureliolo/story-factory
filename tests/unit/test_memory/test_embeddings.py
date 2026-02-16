"""Tests for WorldDatabase embedding operations (_embeddings module).

Tests the embedding storage, search, and management functions that use
sqlite-vec for vector similarity search, with graceful degradation when
the extension is unavailable.
"""

import hashlib
import threading
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from src.memory.world_database import WorldDatabase, _embeddings


@pytest.fixture
def db(tmp_path) -> Generator[WorldDatabase]:
    """Create a test database with vec_available left at its default.

    Since sqlite-vec may not be installed in the test environment,
    _vec_available will typically be False.
    """
    database = WorldDatabase(tmp_path / "test_embeddings.db")
    yield database
    database.close()


@pytest.fixture
def db_with_vec(tmp_path) -> Generator[WorldDatabase]:
    """Create a test database with vec_available forced to True.

    Creates a standard WorldDatabase, then forces _vec_available = True
    and manually creates a regular table mimicking the vec0 schema (since
    real vec0 requires the sqlite-vec extension which may not be installed).
    """
    database = WorldDatabase(tmp_path / "test_embeddings_vec.db")
    database._vec_available = True

    # Create a regular table mimicking the vec0 schema for tests that
    # interact with the database directly (upsert, delete).
    cursor = database.conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS vec_embeddings")
    cursor.execute(
        """
        CREATE TABLE vec_embeddings (
            embedding BLOB,
            content_type TEXT,
            source_id TEXT,
            entity_type TEXT,
            chapter_number INTEGER,
            display_text TEXT,
            embedding_model TEXT,
            embedded_at TEXT
        )
        """
    )
    database.conn.commit()

    yield database
    database.close()


def _make_embedding(dims: int = 4, value: float = 0.1) -> list[float]:
    """Create a simple embedding vector for testing.

    Args:
        dims: Number of dimensions.
        value: Fill value for each dimension.

    Returns:
        List of floats representing a fake embedding.
    """
    return [value] * dims


def _make_mock_db(vec_available: bool = True) -> MagicMock:
    """Create a mock WorldDatabase for tests that cannot use a real connection.

    This is used for search_similar tests where the real vec0 virtual table
    is not available, and for clear/recreate tests that issue vec0 DDL.

    Args:
        vec_available: Whether to simulate vec being available.

    Returns:
        MagicMock configured to behave like a WorldDatabase.
    """
    mock_db = MagicMock(spec=WorldDatabase)
    mock_db.vec_available = vec_available
    mock_db._vec_available = vec_available
    mock_db._lock = threading.RLock()
    mock_db._closed = False
    mock_db._ensure_open = MagicMock()

    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db.conn = mock_conn

    return mock_db


class TestUpsertEmbedding:
    """Tests for the upsert_embedding function."""

    def test_upsert_embedding_vec_unavailable(self, db):
        """Upsert returns False when sqlite-vec is not available."""
        db._vec_available = False

        result = _embeddings.upsert_embedding(
            db,
            source_id="entity-001",
            content_type="entity",
            text="A brave warrior",
            embedding=_make_embedding(),
            model="fake-embed:latest",
        )

        assert result is False

    def test_upsert_embedding_success(self, db_with_vec):
        """Upsert inserts a new embedding and returns True."""
        result = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text="A brave warrior",
            embedding=_make_embedding(dims=4, value=0.5),
            model="fake-embed:latest",
            entity_type="character",
            chapter_number=1,
        )

        assert result is True

        # Verify metadata was stored
        cursor = db_with_vec.conn.cursor()
        cursor.execute(
            "SELECT source_id, content_type, embedding_model, embedding_dim "
            "FROM embedding_metadata WHERE source_id = ?",
            ("entity-001",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "entity-001"
        assert row[1] == "entity"
        assert row[2] == "fake-embed:latest"
        assert row[3] == 4

        # Verify the vec_embeddings table has the row
        cursor.execute(
            "SELECT source_id, content_type, entity_type, chapter_number, display_text "
            "FROM vec_embeddings WHERE source_id = ?",
            ("entity-001",),
        )
        vec_row = cursor.fetchone()
        assert vec_row is not None
        assert vec_row[0] == "entity-001"
        assert vec_row[1] == "entity"
        assert vec_row[2] == "character"
        assert vec_row[3] == 1
        assert vec_row[4] == "A brave warrior"

    def test_upsert_embedding_skip_unchanged(self, db_with_vec):
        """Upsert returns False when content and model are unchanged."""
        text = "A brave warrior"
        embedding = _make_embedding(dims=4, value=0.5)

        # First upsert
        result1 = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text=text,
            embedding=embedding,
            model="fake-embed:latest",
        )
        assert result1 is True

        # Second upsert with same content and model
        result2 = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text=text,
            embedding=embedding,
            model="fake-embed:latest",
        )
        assert result2 is False

    def test_upsert_embedding_update_changed(self, db_with_vec):
        """Upsert returns True when content changes for the same source_id."""
        embedding = _make_embedding(dims=4, value=0.5)

        # First upsert
        result1 = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text="A brave warrior",
            embedding=embedding,
            model="fake-embed:latest",
        )
        assert result1 is True

        # Second upsert with different content
        result2 = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text="A cowardly thief",
            embedding=_make_embedding(dims=4, value=0.9),
            model="fake-embed:latest",
        )
        assert result2 is True

        # Verify the metadata was updated with new hash
        cursor = db_with_vec.conn.cursor()
        cursor.execute(
            "SELECT content_hash FROM embedding_metadata WHERE source_id = ?",
            ("entity-001",),
        )
        row = cursor.fetchone()
        expected_hash = hashlib.sha256(b"A cowardly thief").hexdigest()
        assert row[0] == expected_hash

    def test_upsert_embedding_update_model_changed(self, db_with_vec):
        """Upsert returns True when model changes even if content is the same."""
        text = "A brave warrior"
        embedding = _make_embedding(dims=4, value=0.5)

        # First upsert with model A
        result1 = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text=text,
            embedding=embedding,
            model="fake-embed-v1:latest",
        )
        assert result1 is True

        # Second upsert with model B (same content, different model)
        result2 = _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-001",
            content_type="entity",
            text=text,
            embedding=embedding,
            model="fake-embed-v2:latest",
        )
        assert result2 is True

    def test_upsert_embedding_truncates_display_text(self, db_with_vec):
        """Upsert truncates display_text to 500 characters in vec table."""
        long_text = "A" * 1000
        embedding = _make_embedding(dims=4, value=0.5)

        _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-long",
            content_type="entity",
            text=long_text,
            embedding=embedding,
            model="fake-embed:latest",
        )

        cursor = db_with_vec.conn.cursor()
        cursor.execute(
            "SELECT display_text FROM vec_embeddings WHERE source_id = ?",
            ("entity-long",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert len(row[0]) == 500


class TestDeleteEmbedding:
    """Tests for the delete_embedding function."""

    def test_delete_embedding_success(self, db_with_vec):
        """Delete removes an existing embedding and its metadata."""
        # Insert an embedding first
        _embeddings.upsert_embedding(
            db_with_vec,
            source_id="entity-del",
            content_type="entity",
            text="To be deleted",
            embedding=_make_embedding(),
            model="fake-embed:latest",
        )

        result = _embeddings.delete_embedding(db_with_vec, "entity-del")
        assert result is True

        # Verify metadata is gone
        cursor = db_with_vec.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM embedding_metadata WHERE source_id = ?",
            ("entity-del",),
        )
        assert cursor.fetchone()[0] == 0

        # Verify vec row is gone
        cursor.execute(
            "SELECT COUNT(*) FROM vec_embeddings WHERE source_id = ?",
            ("entity-del",),
        )
        assert cursor.fetchone()[0] == 0

    def test_delete_embedding_not_found(self, db_with_vec):
        """Delete returns False when the source_id does not exist."""
        result = _embeddings.delete_embedding(db_with_vec, "nonexistent-id")
        assert result is False

    def test_delete_embedding_metadata_only_when_vec_unavailable(self, db):
        """Delete removes metadata even when vec is unavailable."""
        db._vec_available = False

        # Manually insert metadata to test the metadata-only path
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("entity-meta", "entity", "abc123", "fake-embed:latest", "2026-01-01", 4),
        )
        db.conn.commit()

        result = _embeddings.delete_embedding(db, "entity-meta")
        assert result is True

        # Verify metadata is gone
        cursor.execute(
            "SELECT COUNT(*) FROM embedding_metadata WHERE source_id = ?",
            ("entity-meta",),
        )
        assert cursor.fetchone()[0] == 0


class TestSearchSimilar:
    """Tests for the search_similar function."""

    def test_search_similar_vec_unavailable(self, db):
        """Search returns empty list when sqlite-vec is not available."""
        db._vec_available = False

        results = _embeddings.search_similar(
            db,
            query_embedding=_make_embedding(),
        )

        assert results == []

    def test_search_similar_empty_query(self, db_with_vec):
        """Search returns empty list when query_embedding is empty."""
        results = _embeddings.search_similar(
            db_with_vec,
            query_embedding=[],
        )

        assert results == []

    def test_search_similar_with_content_type_filter(self):
        """Search with content_type filter executes the partition key query path."""
        mock_db = _make_mock_db(vec_available=True)
        mock_cursor = mock_db.conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            ("entity-001", "entity", "character", None, "Brave warrior", 0.15),
            ("entity-002", "entity", "location", None, "Dark forest", 0.45),
        ]

        results = _embeddings.search_similar(
            mock_db,
            query_embedding=_make_embedding(dims=4),
            k=5,
            content_type="entity",
        )

        assert len(results) == 2
        assert results[0]["source_id"] == "entity-001"
        assert results[0]["content_type"] == "entity"
        assert results[0]["entity_type"] == "character"
        assert results[0]["distance"] == 0.15
        assert results[1]["source_id"] == "entity-002"

        # Verify the content_type filter was passed to the query
        execute_calls = mock_cursor.execute.call_args_list
        last_query = execute_calls[-1][0][0]
        assert "content_type = ?" in last_query

    def test_search_similar_without_content_type(self):
        """Search without content_type filter uses the unfiltered query path."""
        mock_db = _make_mock_db(vec_available=True)
        mock_cursor = mock_db.conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            ("rel-001", "relationship", "", 2, "allies with", 0.2),
        ]

        results = _embeddings.search_similar(
            mock_db,
            query_embedding=_make_embedding(dims=4),
            k=3,
        )

        assert len(results) == 1
        assert results[0]["source_id"] == "rel-001"
        assert results[0]["content_type"] == "relationship"
        assert results[0]["chapter_number"] == 2

        # Verify no content_type filter in the query
        execute_calls = mock_cursor.execute.call_args_list
        last_query = execute_calls[-1][0][0]
        assert "content_type = ?" not in last_query

    def test_search_similar_entity_type_filter(self):
        """Search filters results by entity_type post-query."""
        mock_db = _make_mock_db(vec_available=True)
        mock_cursor = mock_db.conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            ("entity-001", "entity", "character", None, "Brave warrior", 0.1),
            ("entity-002", "entity", "location", None, "Dark forest", 0.3),
            ("entity-003", "entity", "character", None, "Cunning rogue", 0.5),
        ]

        results = _embeddings.search_similar(
            mock_db,
            query_embedding=_make_embedding(dims=4),
            entity_type="character",
        )

        # Only "character" entity_type rows should be returned
        assert len(results) == 2
        assert all(r["entity_type"] == "character" for r in results)

    def test_search_similar_chapter_number_filter(self):
        """Search filters results by chapter_number post-query."""
        mock_db = _make_mock_db(vec_available=True)
        mock_cursor = mock_db.conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            ("entity-001", "entity", "character", 1, "Brave warrior", 0.1),
            ("entity-002", "entity", "character", 2, "Cunning rogue", 0.3),
            ("entity-003", "entity", "character", 1, "Wise sage", 0.5),
        ]

        results = _embeddings.search_similar(
            mock_db,
            query_embedding=_make_embedding(dims=4),
            chapter_number=1,
        )

        assert len(results) == 2
        assert all(r["chapter_number"] == 1 for r in results)


class TestGetEmbeddingStats:
    """Tests for the get_embedding_stats function."""

    def test_get_embedding_stats_empty(self, db):
        """Stats return zeros and empty lists when no embeddings exist."""
        stats = _embeddings.get_embedding_stats(db)

        assert stats["total"] == 0
        assert stats["by_content_type"] == {}
        assert stats["models"] == []
        assert stats["dimensions"] == []
        assert "vec_available" in stats

    def test_get_embedding_stats_with_data(self, db):
        """Stats reflect correct counts, models, and dimensions from metadata."""
        cursor = db.conn.cursor()

        # Insert metadata rows (no vec needed for stats)
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e1", "entity", "hash1", "fake-embed-v1:latest", "2026-01-01", 384),
        )
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e2", "entity", "hash2", "fake-embed-v1:latest", "2026-01-01", 384),
        )
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("r1", "relationship", "hash3", "fake-embed-v2:latest", "2026-01-02", 768),
        )
        db.conn.commit()

        stats = _embeddings.get_embedding_stats(db)

        assert stats["total"] == 3
        assert stats["by_content_type"]["entity"] == 2
        assert stats["by_content_type"]["relationship"] == 1
        assert set(stats["models"]) == {"fake-embed-v1:latest", "fake-embed-v2:latest"}
        assert set(stats["dimensions"]) == {384, 768}


class TestNeedsReembedding:
    """Tests for the needs_reembedding function."""

    def test_needs_reembedding_false_when_empty(self, db):
        """No reembedding needed when there are no embeddings at all."""
        result = _embeddings.needs_reembedding(db, "fake-embed:latest")
        assert result is False

    def test_needs_reembedding_false_all_same_model(self, db):
        """No reembedding needed when all embeddings use the current model."""
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e1", "entity", "hash1", "fake-embed:latest", "2026-01-01", 384),
        )
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e2", "entity", "hash2", "fake-embed:latest", "2026-01-01", 384),
        )
        db.conn.commit()

        result = _embeddings.needs_reembedding(db, "fake-embed:latest")
        assert result is False

    def test_needs_reembedding_true_different_model(self, db):
        """Reembedding needed when some embeddings use a different model."""
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e1", "entity", "hash1", "fake-embed-v1:latest", "2026-01-01", 384),
        )
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e2", "entity", "hash2", "fake-embed-v2:latest", "2026-01-01", 384),
        )
        db.conn.commit()

        result = _embeddings.needs_reembedding(db, "fake-embed-v2:latest")
        assert result is True


class TestClearAllEmbeddings:
    """Tests for the clear_all_embeddings function."""

    def test_clear_all_embeddings(self, db_with_vec):
        """Clear removes all metadata and recreates the vec table."""
        # Insert some test data using the fake vec_embeddings table
        _embeddings.upsert_embedding(
            db_with_vec,
            source_id="e1",
            content_type="entity",
            text="Warrior",
            embedding=_make_embedding(),
            model="fake-embed:latest",
        )
        _embeddings.upsert_embedding(
            db_with_vec,
            source_id="e2",
            content_type="entity",
            text="Mage",
            embedding=_make_embedding(),
            model="fake-embed:latest",
        )

        # Verify data exists before clearing
        cursor = db_with_vec.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
        assert cursor.fetchone()[0] == 2

        # clear_all_embeddings tries to CREATE VIRTUAL TABLE USING vec0(...)
        # which requires the real sqlite-vec extension. We wrap the connection
        # to intercept vec0 DDL while keeping metadata deletion real.
        real_conn = db_with_vec.conn

        class CursorWrapper:
            """Wrapper that intercepts vec0 virtual table DDL statements."""

            def __init__(self, real_cursor):
                """Wrap a real cursor to intercept vec0 DDL.

                Args:
                    real_cursor: The actual sqlite3 cursor to wrap.
                """
                self._real = real_cursor

            def execute(self, sql, params=()):
                """Execute SQL, skipping vec0 CREATE VIRTUAL TABLE statements.

                Args:
                    sql: SQL statement to execute.
                    params: Query parameters.
                """
                if "CREATE VIRTUAL TABLE" in sql and "vec0" in sql:
                    # Simulate the vec0 table recreation with a regular table
                    regular_sql = """
                        CREATE TABLE IF NOT EXISTS vec_embeddings (
                            embedding BLOB, content_type TEXT, source_id TEXT,
                            entity_type TEXT, chapter_number INTEGER,
                            display_text TEXT, embedding_model TEXT, embedded_at TEXT
                        )
                    """
                    return self._real.execute(regular_sql)
                return self._real.execute(sql, params) if params else self._real.execute(sql)

            def __getattr__(self, name):
                """Delegate all other attribute access to the real cursor.

                Args:
                    name: Attribute name to look up.
                """
                return getattr(self._real, name)

        class ConnWrapper:
            """Wrapper that returns CursorWrapper from cursor() calls."""

            def __init__(self, real_conn):
                """Wrap a real connection to return CursorWrapper cursors.

                Args:
                    real_conn: The actual sqlite3 connection to wrap.
                """
                self._real = real_conn

            def cursor(self):
                """Return a wrapped cursor that intercepts vec0 DDL."""
                return CursorWrapper(self._real.cursor())

            def __getattr__(self, name):
                """Delegate all other attribute access to the real connection.

                Args:
                    name: Attribute name to look up.
                """
                return getattr(self._real, name)

        db_with_vec.conn = ConnWrapper(real_conn)

        _embeddings.clear_all_embeddings(db_with_vec)

        # Restore real connection for verification queries
        db_with_vec.conn = real_conn

        # Verify metadata is cleared
        cursor = real_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
        assert cursor.fetchone()[0] == 0

        # Verify vec_embeddings table still exists (recreated as regular table)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
        )
        assert cursor.fetchone() is not None

    def test_clear_all_embeddings_vec_unavailable(self, db):
        """Clear removes metadata but skips vec table when vec is unavailable."""
        db._vec_available = False

        # Insert metadata directly
        cursor = db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e1", "entity", "hash1", "fake-embed:latest", "2026-01-01", 384),
        )
        db.conn.commit()

        _embeddings.clear_all_embeddings(db)

        cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
        assert cursor.fetchone()[0] == 0


class TestRecreateVecTable:
    """Tests for the recreate_vec_table function."""

    def test_recreate_vec_table_vec_unavailable(self, db):
        """Recreate is a no-op when sqlite-vec is not available."""
        # Drop the vec_embeddings table if it was created during __init__
        # (happens when sqlite-vec is installed in the test environment)
        try:
            db.conn.execute("DROP TABLE IF EXISTS vec_embeddings")
        except Exception:
            pass
        db._vec_available = False

        # Should not raise any errors
        _embeddings.recreate_vec_table(db, dimensions=512)

        # Verify no vec_embeddings table was created
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE name='vec_embeddings'")
        assert cursor.fetchone() is None

    def test_recreate_vec_table_success(self):
        """Recreate drops and recreates the vec table with new dimensions."""
        mock_db = _make_mock_db(vec_available=True)
        mock_cursor = mock_db.conn.cursor.return_value

        _embeddings.recreate_vec_table(mock_db, dimensions=512)

        # Verify DROP was called
        execute_calls = mock_cursor.execute.call_args_list
        drop_calls = [c for c in execute_calls if "DROP TABLE" in str(c)]
        assert len(drop_calls) == 1

        # Verify CREATE was called with correct dimensions
        create_calls = [c for c in execute_calls if "CREATE" in str(c)]
        assert len(create_calls) == 1
        create_sql = create_calls[0][0][0]
        assert "float[512]" in create_sql

        # Verify metadata was cleared
        delete_calls = [c for c in execute_calls if "DELETE FROM embedding_metadata" in str(c)]
        assert len(delete_calls) == 1

        # Verify commit was called
        mock_db.conn.commit.assert_called_once()

    def test_recreate_vec_table_clears_metadata(self, db_with_vec):
        """Recreate clears embedding_metadata since old dims are incompatible."""
        # Insert metadata
        cursor = db_with_vec.conn.cursor()
        cursor.execute(
            """
            INSERT INTO embedding_metadata
            (source_id, content_type, content_hash, embedding_model, embedded_at, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("e1", "entity", "hash1", "fake-embed:latest", "2026-01-01", 4),
        )
        db_with_vec.conn.commit()

        # Use a mock db for the vec0 DDL but verify metadata clearing on real db
        real_conn = db_with_vec.conn

        class CursorWrapper:
            """Wrapper that intercepts vec0 DDL for recreate_vec_table tests."""

            def __init__(self, real_cursor):
                """Wrap a real cursor to intercept vec0 DDL.

                Args:
                    real_cursor: The actual sqlite3 cursor to wrap.
                """
                self._real = real_cursor

            def execute(self, sql, params=()):
                """Execute SQL, replacing vec0 CREATE with regular CREATE.

                Args:
                    sql: SQL statement to execute.
                    params: Query parameters.
                """
                if "CREATE VIRTUAL TABLE" in sql and "vec0" in sql:
                    regular_sql = """
                        CREATE TABLE IF NOT EXISTS vec_embeddings (
                            embedding BLOB, content_type TEXT, source_id TEXT,
                            entity_type TEXT, chapter_number INTEGER,
                            display_text TEXT, embedding_model TEXT, embedded_at TEXT
                        )
                    """
                    return self._real.execute(regular_sql)
                return self._real.execute(sql, params) if params else self._real.execute(sql)

            def __getattr__(self, name):
                """Delegate all other attribute access to the real cursor.

                Args:
                    name: Attribute name to look up.
                """
                return getattr(self._real, name)

        class ConnWrapper:
            """Wrapper that returns CursorWrapper cursors."""

            def __init__(self, real_conn):
                """Wrap a real connection.

                Args:
                    real_conn: The actual sqlite3 connection to wrap.
                """
                self._real = real_conn

            def cursor(self):
                """Return a wrapped cursor."""
                return CursorWrapper(self._real.cursor())

            def __getattr__(self, name):
                """Delegate to real connection.

                Args:
                    name: Attribute name to look up.
                """
                return getattr(self._real, name)

        db_with_vec.conn = ConnWrapper(real_conn)
        _embeddings.recreate_vec_table(db_with_vec, dimensions=512)
        db_with_vec.conn = real_conn

        # Metadata should be cleared
        cursor = real_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
        assert cursor.fetchone()[0] == 0


class TestContentHash:
    """Tests for the internal _content_hash helper."""

    def test_content_hash_deterministic(self):
        """Same input always produces the same hash."""
        hash1 = _embeddings._content_hash("hello world")
        hash2 = _embeddings._content_hash("hello world")
        assert hash1 == hash2

    def test_content_hash_different_inputs(self):
        """Different inputs produce different hashes."""
        hash1 = _embeddings._content_hash("hello world")
        hash2 = _embeddings._content_hash("goodbye world")
        assert hash1 != hash2

    def test_content_hash_matches_sha256(self):
        """Hash matches direct SHA-256 computation."""
        text = "test content"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert _embeddings._content_hash(text) == expected
