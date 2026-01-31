"""Tests for world generation dialog helpers."""

from unittest.mock import MagicMock


class TestGetEntityNamesByType:
    """Tests for get_entity_names_by_type helper."""

    def test_returns_empty_when_no_world_db(self):
        """Should return empty list when world_db is None."""
        from src.ui.pages.world._gen_dialogs import get_entity_names_by_type

        page = MagicMock()
        page.state.world_db = None
        assert get_entity_names_by_type(page, "character") == []

    def test_filters_by_entity_type(self):
        """Should return only names matching the requested type."""
        from src.ui.pages.world._gen_dialogs import get_entity_names_by_type

        char = MagicMock(name="Alice", type="character")
        char.name = "Alice"
        loc = MagicMock(name="Dungeon", type="location")
        loc.name = "Dungeon"
        faction = MagicMock(name="Guild", type="faction")
        faction.name = "Guild"

        page = MagicMock()
        page.state.world_db.list_entities.return_value = [char, loc, faction]

        assert get_entity_names_by_type(page, "character") == ["Alice"]
        assert get_entity_names_by_type(page, "location") == ["Dungeon"]
        assert get_entity_names_by_type(page, "faction") == ["Guild"]
        assert get_entity_names_by_type(page, "item") == []

    def test_returns_all_matching_names(self):
        """Should return all names of the matching type."""
        from src.ui.pages.world._gen_dialogs import get_entity_names_by_type

        e1 = MagicMock(type="character")
        e1.name = "Alice"
        e2 = MagicMock(type="character")
        e2.name = "Bob"
        e3 = MagicMock(type="location")
        e3.name = "Castle"

        page = MagicMock()
        page.state.world_db.list_entities.return_value = [e1, e2, e3]

        assert get_entity_names_by_type(page, "character") == ["Alice", "Bob"]


class TestGetAllEntityNames:
    """Tests for get_all_entity_names helper."""

    def test_returns_empty_when_no_world_db(self):
        """Should return empty list when world_db is None."""
        from src.ui.pages.world._gen_dialogs import get_all_entity_names

        page = MagicMock()
        page.state.world_db = None
        assert get_all_entity_names(page) == []

    def test_returns_all_names(self):
        """Should return names from all entity types."""
        from src.ui.pages.world._gen_dialogs import get_all_entity_names

        e1 = MagicMock(type="character")
        e1.name = "Alice"
        e2 = MagicMock(type="location")
        e2.name = "Castle"

        page = MagicMock()
        page.state.world_db.list_entities.return_value = [e1, e2]

        assert get_all_entity_names(page) == ["Alice", "Castle"]
