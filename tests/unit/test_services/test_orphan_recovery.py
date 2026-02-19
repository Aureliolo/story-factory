"""Tests for orphan entity recovery module.

Tests cover:
- _recover_orphans: no orphans, successful recovery, retry on failure, cancellation
- MAX_RETRIES_PER_ORPHAN constant
- Neither endpoint is orphan safety check
- Empty relation_type defaults to 'related_to'
- Single entity skips orphan recovery
- Orphan already connected by previous relationship
- GenerationCancelledError re-raise vs WorldGenerationError retry
"""

from unittest.mock import MagicMock

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._orphan_recovery import MAX_RETRIES_PER_ORPHAN, _recover_orphans
from src.utils.exceptions import GenerationCancelledError, WorldGenerationError


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


class TestNeitherEndpointIsOrphan:
    """Test safety check when neither relationship endpoint is an orphan."""

    def test_neither_endpoint_is_orphan(self, mock_svc, story_state, mock_world_db, mock_services):
        """Test relationship is skipped when LLM returns endpoints that are not orphans."""
        # Add three entities: Hero (orphan), Villain and Castle (connected)
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "An evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")

        # Connect villain to castle so only Hero is orphan
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.5

        # LLM returns relationship between Villain and Castle (neither is an orphan)
        # All MAX_RETRIES_PER_ORPHAN+1 attempts return non-orphan endpoints
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Villain",
                "target": "Castle",
                "relation_type": "owns",
                "description": "Villain owns the castle",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Relationship should be skipped, so no new connections added for Hero
        assert count == 0
        # Should have retried all attempts (each returns non-orphan endpoints)
        assert (
            mock_services.world_quality.generate_relationship_with_quality.call_count
            == MAX_RETRIES_PER_ORPHAN + 1
        )


class TestEmptyRelationTypeDefault:
    """Test relation_type defaults to 'related_to' when missing or falsy."""

    def test_empty_relation_type_default(self, mock_svc, story_state, mock_world_db, mock_services):
        """Test relation_type defaults to 'related_to' when not in relationship dict."""
        # Add two entities: Hero (orphan), Villain connected to Castle
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "An evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")

        # Connect villain to castle so only Hero is orphan
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        # LLM returns relationship dict without a relation_type key
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "Villain",
                "description": "They are connected somehow",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        # Verify the relationship was created with the default relation_type
        relationships = mock_world_db.list_relationships()
        orphan_rels = [r for r in relationships if r.relation_type == "related_to"]
        assert len(orphan_rels) == 1


class TestSingleEntitySkipsOrphan:
    """Test orphan recovery when orphan is the only entity in the database."""

    def test_single_entity_skips_orphan(self, mock_svc, story_state, mock_world_db, mock_services):
        """Test orphan is skipped when it is the only entity (no partners available)."""
        # Add only one entity - it has no partners to connect to
        mock_world_db.add_entity("character", "Lonely Hero", "The sole inhabitant")

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Should return 0 because the single orphan has no partners
        assert count == 0
        # Quality generation should never be called (skipped before reaching it)
        mock_services.world_quality.generate_relationship_with_quality.assert_not_called()


class TestOrphanAlreadyConnectedByPreviousRelationship:
    """Test orphan skipped when already connected as target of a previous orphan's relationship."""

    def test_orphan_already_connected_by_previous_relationship(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test orphan connected as target of another orphan's relationship is skipped."""
        # Add three entities: Hero and Sidekick are orphans, Villain is connected to Castle
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("character", "Sidekick", "Hero's loyal companion")
        villain_id = mock_world_db.add_entity("character", "Villain", "An evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")

        # Connect villain to castle so Hero and Sidekick are orphans
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        # When processing the first orphan (Hero), the LLM returns a relationship
        # connecting Hero to Sidekick. This connects Sidekick too (as a target).
        # When we get to Sidekick in the orphan loop, it should be skipped.
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "Sidekick",
                "relation_type": "allies",
                "description": "Trusted allies",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Only one relationship generated (Hero -> Sidekick), Sidekick skipped
        assert count == 1
        # generate_relationship_with_quality called only once (for Hero, not for Sidekick)
        assert mock_services.world_quality.generate_relationship_with_quality.call_count == 1


class TestGenerationCancelledErrorReRaise:
    """Test GenerationCancelledError is re-raised, not retried."""

    def test_generation_cancelled_error_re_raise(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test GenerationCancelledError propagates out, unlike WorldGenerationError which retries."""
        # Add two entities: Hero is orphan, Villain connected to Castle
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "An evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")

        # Connect villain to castle so only Hero is orphan
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        # Raise GenerationCancelledError on first attempt
        mock_services.world_quality.generate_relationship_with_quality.side_effect = (
            GenerationCancelledError("User cancelled")
        )

        with pytest.raises(GenerationCancelledError, match="User cancelled"):
            _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Should only be called once (no retry for GenerationCancelledError)
        assert mock_services.world_quality.generate_relationship_with_quality.call_count == 1
