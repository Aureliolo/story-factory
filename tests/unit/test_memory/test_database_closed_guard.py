"""Tests for DatabaseClosedError guard on WorldDatabase."""

import pytest

from src.memory.world_database import WorldDatabase
from src.utils.exceptions import DatabaseClosedError


@pytest.fixture
def db(tmp_path):
    """Create a test database that auto-closes after each test."""
    database = WorldDatabase(tmp_path / "test.db")
    yield database
    if not database._closed:
        database.close()


class TestEnsureOpen:
    """Tests for _ensure_open guard method."""

    def test_ensure_open_passes_when_open(self, db):
        """Test that _ensure_open does not raise on an open database."""
        db._ensure_open()  # Should not raise

    def test_ensure_open_raises_when_closed(self, db):
        """Test that _ensure_open raises DatabaseClosedError on a closed database."""
        db.close()
        with pytest.raises(DatabaseClosedError):
            db._ensure_open()


class TestEntityOperationsAfterClose:
    """Tests that entity operations raise DatabaseClosedError after close."""

    def test_add_entity_after_close(self, db):
        """Test that add_entity raises DatabaseClosedError after close."""
        db.close()
        with pytest.raises(DatabaseClosedError):
            db.add_entity(
                entity_type="character",
                name="Test",
                description="Test description",
            )

    def test_list_entities_after_close(self, db):
        """Test that list_entities raises DatabaseClosedError after close."""
        db.close()
        with pytest.raises(DatabaseClosedError):
            db.list_entities()

    def test_get_entity_after_close(self, db):
        """Test that get_entity raises DatabaseClosedError after close."""
        entity_id = db.add_entity(
            entity_type="character",
            name="Test",
            description="Test description",
        )
        db.close()
        with pytest.raises(DatabaseClosedError):
            db.get_entity(entity_id)

    def test_count_entities_after_close(self, db):
        """Test that count_entities raises DatabaseClosedError after close."""
        db.close()
        with pytest.raises(DatabaseClosedError):
            db.count_entities()


class TestRelationshipOperationsAfterClose:
    """Tests that relationship operations raise DatabaseClosedError after close."""

    def test_add_relationship_after_close(self, db):
        """Test that add_relationship raises DatabaseClosedError after close."""
        # Create two entities first
        id1 = db.add_entity(entity_type="character", name="Alice", description="Character A")
        id2 = db.add_entity(entity_type="character", name="Bob", description="Character B")
        db.close()
        with pytest.raises(DatabaseClosedError):
            db.add_relationship(
                source_id=id1,
                target_id=id2,
                relation_type="friend",
                description="Friends",
            )

    def test_list_relationships_after_close(self, db):
        """Test that list_relationships raises DatabaseClosedError after close."""
        db.close()
        with pytest.raises(DatabaseClosedError):
            db.list_relationships()


class TestCloseIdempotent:
    """Test that close is idempotent."""

    def test_close_idempotent(self, db):
        """Test that calling close twice does not raise."""
        db.close()
        db.close()  # Should not raise
        assert db._closed
