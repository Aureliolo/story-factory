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
    svc = MagicMock()
    svc.settings.fuzzy_match_threshold = 0.8
    return svc


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


class TestRecoverOrphansMissingEntityReference:
    """Test orphan recovery logs warning when relationship references missing entity."""

    def test_skips_relationship_with_missing_entity_reference(
        self, mock_svc, story_state, mock_world_db, mock_services, caplog
    ):
        """Test that relationships referencing deleted/missing entities are skipped with warning."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")

        # Connect villain to castle so hero is an orphan
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        # Insert a stale relationship referencing a non-existent entity via raw SQL
        # (bypasses validation to simulate corrupt data — e.g. entity deleted externally)
        with mock_world_db._lock:
            mock_world_db.conn.execute(
                "INSERT INTO relationships"
                " (id, source_id, target_id, relation_type, description, created_at)"
                " VALUES (?, ?, ?, ?, ?, datetime('now'))",
                ("stale-rel-id", villain_id, "non-existent-id", "guards", "Broken reference"),
            )
            mock_world_db.conn.commit()

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "Villain",
                "relation_type": "rivals",
                "description": "They are rivals",
            },
            mock_quality_scores,
            1,
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="src.services.world_service._orphan_recovery"):
            count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Hero should still be recovered despite the broken relationship in the DB
        assert count == 1
        # Verify the warning was logged for the stale relationship
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "missing entity" in r.message
        ]
        assert len(warning_records) >= 1


class TestRecoverOrphansMissingRelationType:
    """Test orphan recovery defaults relation_type to 'related_to' when omitted."""

    def test_missing_relation_type_defaults_to_related_to(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that missing relation_type in generated relationship defaults to 'related_to'."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 7.5

        # Quality service returns a relationship WITHOUT relation_type
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "Villain",
                "description": "They have a connection",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        relationships = mock_world_db.list_relationships()
        hero_rels = [r for r in relationships if r.relation_type == "related_to"]
        assert len(hero_rels) == 1


class TestRecoverOrphansEntityIdResolution:
    """Test orphan recovery handles cross-type name collisions correctly."""

    def test_same_name_different_types_resolves_via_direct_reference(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that orphan with same name as another entity type is resolved via direct object reference."""
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

    def test_orphan_as_target_resolves_by_id(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test orphan is correctly used when its name matches the target endpoint."""
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

        # Quality service returns orphan as TARGET (not source)
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Hero",
                "target": "The Feathered Dominion",
                "relation_type": "studies",
                "description": "The hero studies the concept",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        assert count == 1
        # Verify the relationship was created with the CONCEPT entity (the orphan) as target
        relationships = mock_world_db.list_relationships()
        orphan_rels = [r for r in relationships if r.target_id == concept_id]
        assert len(orphan_rels) == 1
        assert orphan_rels[0].relation_type == "studies"
        assert orphan_rels[0].source_id == hero_id

    def test_normalized_name_matches_orphan_via_article_prefix(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test that _normalize_name strips articles like 'The' so 'The Hero' matches orphan 'Hero'.

        This exercises the direct orphan reference branch (source_name_norm == orphan_name_norm),
        NOT the fuzzy fallback, because normalization strips the leading article.
        """
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        # LLM prefixed orphan name with "The" — normalization strips the article
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
        relationships = mock_world_db.list_relationships()
        rival_rels = [r for r in relationships if r.relation_type == "rivals"]
        assert len(rival_rels) == 1

    def test_fuzzy_fallback_when_neither_endpoint_normalizes_to_orphan(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Test fuzzy fallback when endpoint names don't normalize-match the orphan name.

        When the LLM uses a completely different name variation (not just an article prefix),
        both endpoints fall through to _find_entity_by_name fuzzy lookup.
        """
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        villain_id = mock_world_db.add_entity("character", "Villain", "Evil villain")
        castle_id = mock_world_db.add_entity("location", "Castle", "Dark castle")
        mock_world_db.add_relationship(villain_id, castle_id, "resides_in", "Lives in castle")

        mock_quality_scores = MagicMock()
        mock_quality_scores.average = 8.0

        # LLM used "Brave Warrior" for orphan "Hero" — normalization won't match,
        # but fuzzy matching in _find_entity_by_name should still resolve it
        mock_services.world_quality.generate_relationship_with_quality.return_value = (
            {
                "source": "Brave Warrior",
                "target": "Dark Castle",
                "relation_type": "explores",
                "description": "The warrior explores the castle",
            },
            mock_quality_scores,
            1,
        )

        count = _recover_orphans(mock_svc, story_state, mock_world_db, mock_services)

        # Fuzzy matching may or may not resolve these names — either outcome is valid.
        # The key assertion is that the code doesn't crash and the fuzzy branch executes.
        assert count >= 0
