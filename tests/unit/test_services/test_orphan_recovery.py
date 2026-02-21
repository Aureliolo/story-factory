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

        def cancel_check():
            """Return True to cancel immediately."""
            return True

        count = _recover_orphans(
            mock_svc, story_state, mock_world_db, mock_services, cancel_check=cancel_check
        )

        assert count == 0


class TestRecoverOrphansEmptyRelationship:
    """Test orphan recovery with empty/invalid relationship from quality service."""

    def test_empty_source_target_returns_zero(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that empty source/target in generated relationship adds nothing."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        # Return empty source/target for all attempts
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {"source": "", "target": ""},
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 0


class TestRecoverOrphansUnresolvableNames:
    """Test orphan recovery when generated entity names don't match any real entities."""

    def test_unresolvable_names_returns_zero(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that unresolvable entity names in relationship adds nothing."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "NonExistent1",
                "target": "NonExistent2",
                "relation_type": "knows",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 0


class TestRecoverOrphansNeitherOrphan:
    """Test orphan recovery safety check: neither endpoint is an orphan."""

    def test_neither_is_orphan_returns_zero(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that relationship between two non-orphans is skipped."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.0

        # Return a relationship between two already-connected entities (neither is an orphan)
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Villain",
                "target": "Castle",
                "relation_type": "resides_in",
                "description": "Lives there",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 0


class TestRecoverOrphansAlreadyConnected:
    """Test orphan recovery skips orphans already connected by previous relationships."""

    def test_already_connected_skips_second_orphan(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that second orphan is skipped when already connected by first orphan's rel."""
        # Hero and Sidekick are both orphans; Villain is connected to Castle
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("character", "Sidekick", "A loyal sidekick")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        # First call: connect Hero to Sidekick (connects both orphans at once)
        # Second call should not happen because Sidekick is already connected
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "Sidekick",
                "relation_type": "allies",
                "description": "Loyal allies",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Only one relationship needed — connecting Hero to Sidekick resolves both orphans
        assert count == 1


class TestRecoverOrphansEntityIdResolution:
    """Test orphan recovery resolves entities by ID when names collide across types."""

    def test_same_name_different_types_resolves_by_id(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that orphan with same name as another entity type is correctly identified by ID."""
        # "The Feathered Dominion" exists as both a faction and a concept
        faction_id = mock_world_db.add_entity(
            "faction", "The Feathered Dominion", "A powerful bird faction"
        )
        concept_id = mock_world_db.add_entity(
            "concept", "The Feathered Dominion", "Philosophical concept of avian rule"
        )
        hero_id = mock_world_db.add_entity("character", "Hero", "A brave hero")

        # Connect hero and faction so concept is the orphan
        mock_world_db.add_relationship(hero_id, faction_id, "member_of", "Belongs to faction")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        # Quality service returns a relationship with the orphan's name as source
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "The Feathered Dominion",
                "target": "Hero",
                "relation_type": "inspires",
                "description": "The concept inspires the hero",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        # Verify the relationship was created with the CONCEPT entity (the orphan),
        # not the faction entity (which shares the same name)
        relationships = mock_world_db.list_relationships()
        orphan_rels = [r for r in relationships if r.source_id == concept_id]
        assert len(orphan_rels) == 1
        assert orphan_rels[0].relation_type == "inspires"
        assert orphan_rels[0].target_id == hero_id

    def test_neither_endpoint_matches_orphan_name_falls_back_to_fuzzy(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test fallback to fuzzy name lookup when neither endpoint matches the orphan name."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        # Quality service returns a relationship where neither name matches orphan "Hero"
        # (LLM used a different spelling/variation)
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "The Hero",
                "target": "Villain",
                "relation_type": "rivals",
                "description": "They are rivals",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        # Both endpoints resolved via fuzzy matching (neither exactly matched orphan name)
        relationships = mock_world_db.list_relationships()
        rival_rels = [r for r in relationships if r.relation_type == "rivals"]
        assert len(rival_rels) == 1
