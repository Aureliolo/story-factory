"""Tests for cross-entity name collision detection in world build.

Verifies that entity name lists passed to quality service generation functions
include ALL entity types (not just the type being generated), preventing
cross-type name collisions like "The Feathered Dominion" as both faction and concept.
"""

from unittest.mock import MagicMock

import pytest

from src.memory.story_state import StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.world_service._build import (
    _generate_concepts,
    _generate_factions,
    _generate_items,
    _generate_locations,
)


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
    """Create mock WorldService with settings."""
    svc = MagicMock()
    svc.settings.world_gen_locations_min = 1
    svc.settings.world_gen_locations_max = 1
    svc.settings.world_gen_factions_min = 1
    svc.settings.world_gen_factions_max = 1
    svc.settings.world_gen_items_min = 1
    svc.settings.world_gen_items_max = 1
    svc.settings.world_gen_concepts_min = 1
    svc.settings.world_gen_concepts_max = 1
    return svc


class TestCrossEntityNameCollision:
    """Verify existing_names passed to quality service includes all entity types."""

    def test_generate_locations_includes_all_entity_names(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Location generation should pass all entity names, not just location names."""
        # Add entities of various types
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("faction", "Dark Guild", "An evil guild")

        mock_services.world_quality.generate_locations_with_quality.return_value = []

        _generate_locations(mock_svc, story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_locations_with_quality.call_args
        existing_names = call_args.kwargs["name_provider"]()
        assert "Hero" in existing_names
        assert "Dark Guild" in existing_names

    def test_generate_factions_includes_all_entity_names(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Faction generation should pass all entity names, not just faction names."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("location", "Castle", "A dark castle")

        mock_services.world_quality.generate_factions_with_quality.return_value = []

        _generate_factions(mock_svc, story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_factions_with_quality.call_args
        existing_names = call_args.kwargs["name_provider"]()
        assert "Hero" in existing_names
        assert "Castle" in existing_names

    def test_generate_items_includes_all_entity_names(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Item generation should pass all entity names, not just item names."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("location", "Castle", "A dark castle")
        mock_world_db.add_entity("faction", "Dark Guild", "An evil guild")

        mock_services.world_quality.generate_items_with_quality.return_value = []

        _generate_items(mock_svc, story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_items_with_quality.call_args
        existing_names = call_args.kwargs["name_provider"]()
        assert "Hero" in existing_names
        assert "Castle" in existing_names
        assert "Dark Guild" in existing_names

    def test_generate_concepts_includes_all_entity_names(
        self, mock_svc, story_state, mock_world_db, mock_services
    ):
        """Concept generation should pass all entity names, not just concept names."""
        mock_world_db.add_entity("character", "Hero", "A brave hero")
        mock_world_db.add_entity("location", "Castle", "A dark castle")
        mock_world_db.add_entity("faction", "Dark Guild", "An evil guild")
        mock_world_db.add_entity("item", "Magic Sword", "A legendary sword")

        mock_services.world_quality.generate_concepts_with_quality.return_value = []

        _generate_concepts(mock_svc, story_state, mock_world_db, mock_services)

        call_args = mock_services.world_quality.generate_concepts_with_quality.call_args
        existing_names = call_args.kwargs["name_provider"]()
        assert "Hero" in existing_names
        assert "Castle" in existing_names
        assert "Dark Guild" in existing_names
        assert "Magic Sword" in existing_names
