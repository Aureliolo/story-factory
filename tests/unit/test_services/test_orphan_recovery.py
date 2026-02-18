"""Tests for orphan entity recovery module.

Tests cover:
- _recover_orphans: no orphans, successful recovery, retry on failure, cancellation
- MAX_RETRIES_PER_ORPHAN constant
"""

from unittest.mock import MagicMock

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._orphan_recovery import MAX_RETRIES_PER_ORPHAN, _recover_orphans
from src.utils.exceptions import WorldGenerationError


@pytest.fixture
def story_state():
    """Create story state with brief for testing."""
    state = StoryState(id="test-story-id")
    state.brief = StoryBrief(
        premise="A detective solves mysteries",
        genre="mystery",
        subgenres=["gothic"],
        tone="dark",
        themes=["truth"],
        setting_time="Victorian era",
        setting_place="England",
        target_length="novella",
        language="English",
        content_rating="mild",
    )
    return state


@pytest.fixture
def mock_world_db():
    """Create an in-memory WorldDatabase for testing."""
    db = WorldDatabase(":memory:")
    yield db
    db.close()


@pytest.fixture
def mock_services():
    """Create mock ServiceContainer."""
    return MagicMock()


@pytest.fixture
def mock_svc():
    """Create mock WorldService."""
    return MagicMock()


class TestMaxRetriesConstant:
    """Test MAX_RETRIES_PER_ORPHAN constant."""

    def test_max_retries_is_two(self):
        """Test constant value is 2 (3 total attempts per orphan)."""
        assert MAX_RETRIES_PER_ORPHAN == 2


class TestRecoverOrphansNoOrphans:
    """Test orphan recovery when no orphans exist."""

    def test_no_orphans_returns_zero(self, mock_svc, story_state, mock_world_db, mock_services):
        """Test returns 0 when no orphan entities exist."""
        # Empty database has no orphans
        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)
        assert count == 0


class TestRecoverOrphansSuccess:
    """Test successful orphan recovery."""

    def test_recovers_orphan_with_valid_relationship(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test orphan is connected via generated relationship."""
        # Add entities - Hero has no relationships (orphan), Villain connected to Castle
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "An evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")

        # Connect villain to castle so only hero is orphan
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "Villain",
                "relation_type": "enemies",
                "description": "Mortal enemies",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        # Hero should now have a relationship
        relationships = mock_world_db.list_relationships()
        rel_types = [r.relation_type for r in relationships]
        assert "enemies" in rel_types


class TestRecoverOrphansRetry:
    """Test orphan recovery retry behavior."""

    def test_retries_on_generation_error(self, mock_svc, story_state, mock_world_db, mock_services):
        """Test retries when generation fails, then succeeds."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("character", "Villain", "Evil")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        # First attempt fails, second succeeds
        mock_services.world_quality.generate_relationship_with_quality.side_effect = [
            WorldGenerationError("Temporary failure"),
            (
                {
                    "source": "Hero",
                    "target": "Villain",
                    "relation_type": "rivals",
                    "description": "Fierce rivals",
                },
                mock_quality_scores,
                1,
            ),
        ]

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        # Should have been called twice (first fail + second success)
        assert mock_services.world_quality.generate_relationship_with_quality.call_count == 2


class TestRecoverOrphansCancellation:
    """Test orphan recovery cancellation."""

    def test_cancellation_stops_recovery(self, mock_svc, story_state, mock_world_db, mock_services):
        """Test cancel_check stops orphan recovery."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("character", "Villain", "Evil")

        cancel_called = False

        def cancel_check():
            """Return True to cancel after first check."""
            nonlocal cancel_called
            if cancel_called:
                return True
            cancel_called = True
            return True  # Cancel immediately

        count = _recover_orphans(
            mock_svc, story_state, mock_world_db, mock_services, cancel_check=cancel_check
        )

        assert count == 0
