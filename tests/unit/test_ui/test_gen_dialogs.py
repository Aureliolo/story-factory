"""Tests for world generation dialog helpers."""

from unittest.mock import MagicMock

import pytest

from src.ui.pages.world._gen_dialogs import get_all_entity_names, get_entity_names_by_type


def _make_entity(name: str, entity_type: str) -> MagicMock:
    """Create a mock entity with the given name and type."""
    entity = MagicMock(type=entity_type)
    entity.name = name
    return entity


@pytest.fixture
def page_no_world_db():
    """Page mock with no world database."""
    page = MagicMock()
    page.state.world_db = None
    return page


@pytest.fixture
def page_with_entities():
    """Page mock with a world database containing mixed entity types."""
    char = _make_entity("Alice", "character")
    char2 = _make_entity("Bob", "character")
    loc = _make_entity("Dungeon", "location")
    faction = _make_entity("Guild", "faction")

    page = MagicMock()
    page.state.world_db.list_entities.return_value = [char, char2, loc, faction]
    return page


class TestGetEntityNamesByType:
    """Tests for get_entity_names_by_type helper."""

    def test_returns_empty_when_no_world_db(self, page_no_world_db):
        """Should return empty list when world_db is None."""
        assert get_entity_names_by_type(page_no_world_db, "character") == []

    def test_filters_by_entity_type(self, page_with_entities):
        """Should return only names matching the requested type."""
        assert get_entity_names_by_type(page_with_entities, "character") == ["Alice", "Bob"]
        assert get_entity_names_by_type(page_with_entities, "location") == ["Dungeon"]
        assert get_entity_names_by_type(page_with_entities, "faction") == ["Guild"]
        assert get_entity_names_by_type(page_with_entities, "item") == []

    def test_returns_all_matching_names(self, page_with_entities):
        """Should return all names of the matching type."""
        assert get_entity_names_by_type(page_with_entities, "character") == ["Alice", "Bob"]


class TestGetAllEntityNames:
    """Tests for get_all_entity_names helper."""

    def test_returns_empty_when_no_world_db(self, page_no_world_db):
        """Should return empty list when world_db is None."""
        assert get_all_entity_names(page_no_world_db) == []

    def test_returns_all_names(self, page_with_entities):
        """Should return names from all entity types."""
        assert get_all_entity_names(page_with_entities) == ["Alice", "Bob", "Dungeon", "Guild"]
