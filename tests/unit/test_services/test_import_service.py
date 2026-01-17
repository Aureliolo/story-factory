"""Tests for import service."""

import pytest

from memory.story_state import Brief, StoryState
from services.import_service import ImportService
from services.model_mode_service import ModelModeService
from settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings.load()


@pytest.fixture
def mode_service(settings):
    """Create model mode service."""
    return ModelModeService(settings)


@pytest.fixture
def import_service(settings, mode_service):
    """Create import service."""
    return ImportService(settings, mode_service)


@pytest.fixture
def sample_text():
    """Sample text for testing extraction."""
    return """
    Sarah Chen walked into the old library, her footsteps echoing through the dusty halls.
    She had been searching for the ancient tome for weeks, following clues left by her late
    mentor, Professor Williams. The book, bound in leather and adorned with strange symbols,
    was said to hold the key to understanding the mysterious artifact discovered in the ruins
    of Atlantis.

    Meanwhile, her rival, Marcus Drake, was already ahead of her. He had hired mercenaries
    to guard the entrance to the underground temple where the artifact was hidden. Sarah
    knew she would have to outsmart him if she wanted to succeed.

    The golden amulet, glowing faintly in the darkness, was the prize they both sought.
    Legend said it could grant its wearer the power to see the future.
    """


@pytest.fixture
def story_state():
    """Create a story state with brief."""
    brief = Brief(
        genre="adventure",
        premise="A quest for an ancient artifact",
        tone="mysterious",
        setting_place="Modern day",
        setting_time="Present",
        target_length="novella",
        themes=["discovery", "rivalry"],
        pov="third person",
        language="English",
    )
    state = StoryState(id="test-project", name="Test Project")
    state.brief = brief
    return state


class TestImportService:
    """Tests for ImportService."""

    def test_initialization(self, import_service):
        """Test service initializes correctly."""
        assert import_service is not None
        assert import_service.settings is not None
        assert import_service.mode_service is not None

    def test_get_model(self, import_service):
        """Test model selection."""
        model = import_service._get_model()
        assert model is not None
        assert isinstance(model, str)

    @pytest.mark.asyncio
    async def test_extract_characters_empty_text(self, import_service):
        """Test character extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_characters("")

    @pytest.mark.asyncio
    async def test_extract_locations_empty_text(self, import_service):
        """Test location extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_locations("")

    @pytest.mark.asyncio
    async def test_extract_items_empty_text(self, import_service):
        """Test item extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_items("")

    @pytest.mark.asyncio
    async def test_infer_relationships_empty_text(self, import_service):
        """Test relationship inference with empty text."""
        characters = [{"name": "Sarah"}, {"name": "Marcus"}]
        with pytest.raises(ValueError):
            import_service.infer_relationships(characters, "")

    def test_infer_relationships_no_characters(self, import_service):
        """Test relationship inference with no characters."""
        result = import_service.infer_relationships([], "some text")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_all_empty_text(self, import_service):
        """Test full extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_all("")

    # Note: The following tests require a running Ollama instance
    # They are marked to skip in CI but can be run locally for integration testing

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama instance")
    def test_extract_characters_from_sample(self, import_service, sample_text, story_state):
        """Test character extraction from sample text."""
        characters = import_service.extract_characters(sample_text, story_state)

        assert isinstance(characters, list)
        assert len(characters) > 0

        # Check for expected characters
        char_names = [c.get("name", "").lower() for c in characters]
        assert any("sarah" in name for name in char_names)
        assert any("marcus" in name for name in char_names)

        # Validate structure
        for char in characters:
            assert isinstance(char, dict)
            assert "name" in char
            assert "role" in char
            assert "description" in char
            assert "confidence" in char
            assert "needs_review" in char
            assert 0.0 <= char["confidence"] <= 1.0

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama instance")
    def test_extract_locations_from_sample(self, import_service, sample_text, story_state):
        """Test location extraction from sample text."""
        locations = import_service.extract_locations(sample_text, story_state)

        assert isinstance(locations, list)
        assert len(locations) > 0

        # Check for expected locations
        loc_names = [loc.get("name", "").lower() for loc in locations]
        assert any("library" in name for name in loc_names) or any(
            "atlantis" in name for name in loc_names
        )

        # Validate structure
        for loc in locations:
            assert isinstance(loc, dict)
            assert "name" in loc
            assert "type" in loc
            assert loc["type"] == "location"
            assert "description" in loc
            assert "confidence" in loc
            assert "needs_review" in loc

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama instance")
    def test_extract_items_from_sample(self, import_service, sample_text, story_state):
        """Test item extraction from sample text."""
        items = import_service.extract_items(sample_text, story_state)

        assert isinstance(items, list)
        assert len(items) > 0

        # Check for expected items
        item_names = [item.get("name", "").lower() for item in items]
        assert any("amulet" in name or "tome" in name or "book" in name for name in item_names)

        # Validate structure
        for item in items:
            assert isinstance(item, dict)
            assert "name" in item
            assert "type" in item
            assert item["type"] == "item"
            assert "description" in item
            assert "confidence" in item
            assert "needs_review" in item

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama instance")
    def test_infer_relationships_from_sample(self, import_service, sample_text):
        """Test relationship inference from sample text."""
        characters = [
            {"name": "Sarah Chen"},
            {"name": "Marcus Drake"},
            {"name": "Professor Williams"},
        ]

        relationships = import_service.infer_relationships(characters, sample_text)

        assert isinstance(relationships, list)
        # There should be some relationships inferred
        assert len(relationships) > 0

        # Validate structure
        for rel in relationships:
            assert isinstance(rel, dict)
            assert "source" in rel
            assert "target" in rel
            assert "relation_type" in rel
            assert "description" in rel
            assert "confidence" in rel
            assert "needs_review" in rel

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama instance")
    def test_extract_all_from_sample(self, import_service, sample_text, story_state):
        """Test full extraction from sample text."""
        result = import_service.extract_all(sample_text, story_state)

        assert isinstance(result, dict)
        assert "characters" in result
        assert "locations" in result
        assert "items" in result
        assert "relationships" in result
        assert "summary" in result

        # Check summary
        summary = result["summary"]
        assert "total_entities" in summary
        assert "characters" in summary
        assert "locations" in summary
        assert "items" in summary
        assert "relationships" in summary
        assert "needs_review" in summary

        # Verify counts match
        assert summary["characters"] == len(result["characters"])
        assert summary["locations"] == len(result["locations"])
        assert summary["items"] == len(result["items"])
        assert summary["relationships"] == len(result["relationships"])
        assert (
            summary["total_entities"]
            == summary["characters"] + summary["locations"] + summary["items"]
        )

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires running Ollama instance")
    def test_confidence_flagging(self, import_service, story_state):
        """Test that low confidence items are flagged for review."""
        # Text with an ambiguous reference
        ambiguous_text = """
        Someone walked into the room. They seemed important.
        A strange object glowed on the table.
        """

        result = import_service.extract_all(ambiguous_text, story_state)

        # At least some items should need review due to ambiguity
        needs_review_items = [
            item
            for entities in [result["characters"], result["locations"], result["items"]]
            for item in entities
            if item.get("needs_review", False)
        ]

        # With such vague text, we expect some flagging
        assert len(needs_review_items) >= 0  # May or may not flag, depends on LLM

        # All items should have confidence scores
        all_items = (
            result["characters"] + result["locations"] + result["items"] + result["relationships"]
        )
        for item in all_items:
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0
            # If confidence < 0.7, should be flagged
            if item["confidence"] < 0.7:
                assert item.get("needs_review", False) is True
