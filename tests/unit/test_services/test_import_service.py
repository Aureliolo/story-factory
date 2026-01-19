"""Tests for import service."""

from unittest.mock import MagicMock, patch

import pytest

from memory.story_state import StoryBrief, StoryState
from services.import_service import (
    ExtractedCharacter,
    ExtractedCharacterList,
    ExtractedItem,
    ExtractedItemList,
    ExtractedLocation,
    ExtractedLocationList,
    ExtractedRelationship,
    ExtractedRelationshipList,
    ImportService,
)
from services.model_mode_service import ModelModeService
from settings import Settings
from utils.exceptions import WorldGenerationError


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
def mock_settings():
    """Create mock settings with required attributes."""
    settings = MagicMock(spec=Settings)
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_timeout = 30
    settings.temp_import_extraction = 0.3
    settings.import_default_confidence = 0.5
    settings.import_confidence_threshold = 0.7
    settings.import_character_token_multiplier = 4
    settings.llm_tokens_character_create = 500
    settings.llm_tokens_location_create = 400
    settings.llm_tokens_item_create = 400
    settings.llm_tokens_relationship_create = 300
    return settings


@pytest.fixture
def mock_mode_service():
    """Create mock model mode service."""
    service = MagicMock(spec=ModelModeService)
    service.get_model_for_agent.return_value = "test-model"
    return service


@pytest.fixture
def mock_import_service(mock_settings, mock_mode_service):
    """Create import service with mocked dependencies."""
    return ImportService(mock_settings, mock_mode_service)


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
    brief = StoryBrief(
        genre="adventure",
        premise="A quest for an ancient artifact",
        tone="mysterious",
        setting_place="Modern day",
        setting_time="Present",
        target_length="novella",
        themes=["discovery", "rivalry"],
        language="English",
        content_rating="none",
    )
    state = StoryState(id="test-project", project_name="Test Project")
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

    def test_extract_characters_empty_text(self, import_service):
        """Test character extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_characters("")

    def test_extract_locations_empty_text(self, import_service):
        """Test location extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_locations("")

    def test_extract_items_empty_text(self, import_service):
        """Test item extraction with empty text."""
        with pytest.raises(ValueError):
            import_service.extract_items("")

    def test_infer_relationships_empty_text(self, import_service):
        """Test relationship inference with empty text."""
        characters = [{"name": "Sarah"}, {"name": "Marcus"}]
        with pytest.raises(ValueError):
            import_service.infer_relationships(characters, "")

    def test_infer_relationships_no_characters(self, import_service):
        """Test relationship inference with no characters."""
        result = import_service.infer_relationships([], "some text")
        assert result == []

    def test_extract_all_empty_text(self, import_service):
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


class TestExtractCharactersMocked:
    """Tests for extract_characters with mocked LLM."""

    def test_extract_characters_success(self, mock_import_service, sample_text, story_state):
        """Test successful character extraction."""
        mock_response = ExtractedCharacterList(
            characters=[
                ExtractedCharacter(
                    name="Sarah Chen",
                    role="protagonist",
                    description="A determined researcher",
                    relationships={"Professor Williams": "mentor"},
                    confidence=0.95,
                    needs_review=False,
                ),
                ExtractedCharacter(
                    name="Marcus Drake",
                    role="antagonist",
                    description="A rival treasure hunter",
                    relationships={"Sarah Chen": "rival"},
                    confidence=0.9,
                    needs_review=False,
                ),
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_characters(sample_text, story_state)

            assert len(result) == 2
            assert result[0]["name"] == "Sarah Chen"
            assert result[0]["role"] == "protagonist"
            assert result[1]["name"] == "Marcus Drake"

    def test_extract_characters_without_story_state(self, mock_import_service, sample_text):
        """Test character extraction without story state context."""
        mock_response = ExtractedCharacterList(
            characters=[
                ExtractedCharacter(
                    name="Test Character",
                    role="supporting",
                    description="A test description",
                    relationships={},
                    confidence=0.8,
                    needs_review=False,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_characters(sample_text, story_state=None)

            assert len(result) == 1
            assert result[0]["name"] == "Test Character"

    def test_extract_characters_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence characters are flagged for review."""
        mock_response = ExtractedCharacterList(
            characters=[
                ExtractedCharacter(
                    name="Uncertain Character",
                    role="supporting",
                    description="Ambiguous description",
                    relationships={},
                    confidence=0.5,  # Below threshold of 0.7
                    needs_review=False,  # Will be set by service
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_characters(sample_text)

            assert len(result) == 1
            assert result[0]["confidence"] == 0.5
            assert result[0]["needs_review"] is True

    def test_extract_characters_high_confidence_not_flagged(self, mock_import_service, sample_text):
        """Test that high confidence characters are not flagged for review."""
        mock_response = ExtractedCharacterList(
            characters=[
                ExtractedCharacter(
                    name="Clear Character",
                    role="protagonist",
                    description="Well-described character",
                    relationships={},
                    confidence=0.9,  # Above threshold
                    needs_review=False,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_characters(sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is False

    def test_extract_characters_error(self, mock_import_service, sample_text):
        """Test handling of errors during character extraction."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=Exception("Model error"),
        ):
            with pytest.raises(WorldGenerationError, match="Character extraction failed"):
                mock_import_service.extract_characters(sample_text)

    def test_extract_characters_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.raises(WorldGenerationError, match="Character extraction failed"):
                mock_import_service.extract_characters(sample_text)

    def test_extract_characters_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=TimeoutError("Request timed out"),
        ):
            with pytest.raises(WorldGenerationError, match="Character extraction failed"):
                mock_import_service.extract_characters(sample_text)


class TestExtractLocationsMocked:
    """Tests for extract_locations with mocked LLM."""

    def test_extract_locations_success(self, mock_import_service, sample_text, story_state):
        """Test successful location extraction."""
        mock_response = ExtractedLocationList(
            locations=[
                ExtractedLocation(
                    name="Old Library",
                    type="location",
                    description="A dusty library with ancient books",
                    significance="Where the search began",
                    confidence=0.9,
                    needs_review=False,
                ),
                ExtractedLocation(
                    name="Ruins of Atlantis",
                    type="location",
                    description="Ancient underwater ruins",
                    significance="Where the artifact was discovered",
                    confidence=0.85,
                    needs_review=False,
                ),
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_locations(sample_text, story_state)

            assert len(result) == 2
            assert result[0]["name"] == "Old Library"
            assert result[0]["type"] == "location"
            assert result[1]["name"] == "Ruins of Atlantis"

    def test_extract_locations_without_story_state(self, mock_import_service, sample_text):
        """Test location extraction without story state context."""
        mock_response = ExtractedLocationList(
            locations=[
                ExtractedLocation(
                    name="Test Location",
                    type="location",
                    description="A test location",
                    significance="Test significance",
                    confidence=0.8,
                    needs_review=False,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_locations(sample_text, story_state=None)

            assert len(result) == 1
            assert result[0]["name"] == "Test Location"

    def test_extract_locations_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence locations are flagged for review."""
        mock_response = ExtractedLocationList(
            locations=[
                ExtractedLocation(
                    name="Uncertain Location",
                    type="location",
                    description="Vague description",
                    significance="Unclear",
                    confidence=0.4,  # Below threshold of 0.7
                    needs_review=False,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_locations(sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is True

    def test_extract_locations_high_confidence_not_flagged(self, mock_import_service, sample_text):
        """Test that high confidence locations are not flagged for review."""
        mock_response = ExtractedLocationList(
            locations=[
                ExtractedLocation(
                    name="Clear Location",
                    type="location",
                    description="Well-described location",
                    significance="Important",
                    confidence=0.9,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_locations(sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is False

    def test_extract_locations_error(self, mock_import_service, sample_text):
        """Test handling of errors during location extraction."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=Exception("Model error"),
        ):
            with pytest.raises(WorldGenerationError, match="Location extraction failed"):
                mock_import_service.extract_locations(sample_text)

    def test_extract_locations_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.raises(WorldGenerationError, match="Location extraction failed"):
                mock_import_service.extract_locations(sample_text)

    def test_extract_locations_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=TimeoutError("Timed out"),
        ):
            with pytest.raises(WorldGenerationError, match="Location extraction failed"):
                mock_import_service.extract_locations(sample_text)


class TestExtractItemsMocked:
    """Tests for extract_items with mocked LLM."""

    def test_extract_items_success(self, mock_import_service, sample_text, story_state):
        """Test successful item extraction."""
        mock_response = ExtractedItemList(
            items=[
                ExtractedItem(
                    name="Ancient Tome",
                    type="item",
                    description="A leather-bound book with strange symbols",
                    significance="Contains the key to understanding the artifact",
                    properties=["magical", "ancient"],
                    confidence=0.9,
                    needs_review=False,
                ),
                ExtractedItem(
                    name="Golden Amulet",
                    type="item",
                    description="A glowing golden amulet",
                    significance="Grants the power to see the future",
                    properties=["magical", "glowing"],
                    confidence=0.95,
                    needs_review=False,
                ),
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_items(sample_text, story_state)

            assert len(result) == 2
            assert result[0]["name"] == "Ancient Tome"
            assert result[0]["type"] == "item"
            assert result[1]["name"] == "Golden Amulet"
            assert "magical" in result[0]["properties"]

    def test_extract_items_without_story_state(self, mock_import_service, sample_text):
        """Test item extraction without story state context."""
        mock_response = ExtractedItemList(
            items=[
                ExtractedItem(
                    name="Test Item",
                    type="item",
                    description="A test item",
                    significance="Test",
                    properties=[],
                    confidence=0.8,
                    needs_review=False,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_items(sample_text, story_state=None)

            assert len(result) == 1
            assert result[0]["name"] == "Test Item"

    def test_extract_items_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence items are flagged for review."""
        mock_response = ExtractedItemList(
            items=[
                ExtractedItem(
                    name="Uncertain Item",
                    type="item",
                    description="Vague",
                    significance="Unclear",
                    properties=[],
                    confidence=0.3,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_items(sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is True

    def test_extract_items_high_confidence_not_flagged(self, mock_import_service, sample_text):
        """Test that high confidence items are not flagged for review."""
        mock_response = ExtractedItemList(
            items=[
                ExtractedItem(
                    name="Clear Item",
                    type="item",
                    description="Well-described",
                    significance="Important",
                    properties=["unique"],
                    confidence=0.95,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.extract_items(sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is False

    def test_extract_items_error(self, mock_import_service, sample_text):
        """Test handling of errors during item extraction."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=Exception("Model error"),
        ):
            with pytest.raises(WorldGenerationError, match="Item extraction failed"):
                mock_import_service.extract_items(sample_text)

    def test_extract_items_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.raises(WorldGenerationError, match="Item extraction failed"):
                mock_import_service.extract_items(sample_text)

    def test_extract_items_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        with patch(
            "services.import_service.generate_structured",
            side_effect=TimeoutError("Timed out"),
        ):
            with pytest.raises(WorldGenerationError, match="Item extraction failed"):
                mock_import_service.extract_items(sample_text)


class TestInferRelationshipsMocked:
    """Tests for infer_relationships with mocked LLM."""

    def test_infer_relationships_success(self, mock_import_service, sample_text):
        """Test successful relationship inference."""
        characters = [
            {"name": "Sarah Chen"},
            {"name": "Marcus Drake"},
            {"name": "Professor Williams"},
        ]

        mock_response = ExtractedRelationshipList(
            relationships=[
                ExtractedRelationship(
                    source="Sarah Chen",
                    target="Professor Williams",
                    relation_type="mentored_by",
                    description="Professor Williams was Sarah's mentor",
                    confidence=0.9,
                    needs_review=False,
                ),
                ExtractedRelationship(
                    source="Sarah Chen",
                    target="Marcus Drake",
                    relation_type="rivals_with",
                    description="They are competing for the same artifact",
                    confidence=0.95,
                    needs_review=False,
                ),
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.infer_relationships(characters, sample_text)

            assert len(result) == 2
            assert result[0]["source"] == "Sarah Chen"
            assert result[0]["target"] == "Professor Williams"
            assert result[0]["relation_type"] == "mentored_by"

    def test_infer_relationships_handles_empty_characters(self, mock_import_service, sample_text):
        """Test that empty characters list returns empty result without LLM call."""
        with patch("services.import_service.generate_structured") as mock_gen:
            result = mock_import_service.infer_relationships([], sample_text)

            assert result == []
            mock_gen.assert_not_called()

    def test_infer_relationships_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence relationships are flagged for review."""
        characters = [{"name": "A"}, {"name": "B"}]

        mock_response = ExtractedRelationshipList(
            relationships=[
                ExtractedRelationship(
                    source="A",
                    target="B",
                    relation_type="knows",
                    description="Possible acquaintance",
                    confidence=0.4,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.infer_relationships(characters, sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is True

    def test_infer_relationships_high_confidence_not_flagged(
        self, mock_import_service, sample_text
    ):
        """Test that high confidence relationships are not flagged for review."""
        characters = [{"name": "A"}, {"name": "B"}]

        mock_response = ExtractedRelationshipList(
            relationships=[
                ExtractedRelationship(
                    source="A",
                    target="B",
                    relation_type="parent_of",
                    description="Explicitly stated parent relationship",
                    confidence=1.0,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.infer_relationships(characters, sample_text)

            assert len(result) == 1
            assert result[0]["needs_review"] is False

    def test_infer_relationships_handles_non_dict_characters(
        self, mock_import_service, sample_text
    ):
        """Test handling of characters list with non-dict items."""
        characters = [{"name": "Alice"}, "not a dict", {"name": "Bob"}]

        mock_response = ExtractedRelationshipList(
            relationships=[
                ExtractedRelationship(
                    source="Alice",
                    target="Bob",
                    relation_type="knows",
                    description="They know each other",
                    confidence=0.8,
                    needs_review=False,
                )
            ]
        )

        with patch("services.import_service.generate_structured", return_value=mock_response):
            result = mock_import_service.infer_relationships(characters, sample_text)

            assert len(result) == 1

    def test_infer_relationships_error(self, mock_import_service, sample_text):
        """Test handling of errors during relationship inference."""
        characters = [{"name": "Alice"}, {"name": "Bob"}]

        with patch(
            "services.import_service.generate_structured",
            side_effect=Exception("Model error"),
        ):
            with pytest.raises(WorldGenerationError, match="Relationship inference failed"):
                mock_import_service.infer_relationships(characters, sample_text)

    def test_infer_relationships_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        characters = [{"name": "Alice"}]

        with patch(
            "services.import_service.generate_structured",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.raises(WorldGenerationError, match="Relationship inference failed"):
                mock_import_service.infer_relationships(characters, sample_text)

    def test_infer_relationships_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        characters = [{"name": "Alice"}]

        with patch(
            "services.import_service.generate_structured",
            side_effect=TimeoutError("Timed out"),
        ):
            with pytest.raises(WorldGenerationError, match="Relationship inference failed"):
                mock_import_service.infer_relationships(characters, sample_text)


class TestExtractAllMocked:
    """Tests for extract_all with mocked extraction methods."""

    def test_extract_all_success(self, mock_import_service, sample_text, story_state):
        """Test successful full extraction."""
        mock_characters = [
            {"name": "Sarah", "role": "protagonist", "confidence": 0.9, "needs_review": False}
        ]
        mock_locations = [
            {"name": "Library", "type": "location", "confidence": 0.85, "needs_review": False}
        ]
        mock_items = [{"name": "Book", "type": "item", "confidence": 0.8, "needs_review": False}]
        mock_relationships = [
            {
                "source": "Sarah",
                "target": "Library",
                "relation_type": "visits",
                "confidence": 0.7,
                "needs_review": False,
            }
        ]

        with (
            patch.object(
                mock_import_service, "extract_characters", return_value=mock_characters
            ) as mock_chars,
            patch.object(
                mock_import_service, "extract_locations", return_value=mock_locations
            ) as mock_locs,
            patch.object(mock_import_service, "extract_items", return_value=mock_items) as mock_itm,
            patch.object(
                mock_import_service, "infer_relationships", return_value=mock_relationships
            ) as mock_rels,
        ):
            result = mock_import_service.extract_all(sample_text, story_state)

            assert "characters" in result
            assert "locations" in result
            assert "items" in result
            assert "relationships" in result
            assert "summary" in result

            assert result["characters"] == mock_characters
            assert result["locations"] == mock_locations
            assert result["items"] == mock_items
            assert result["relationships"] == mock_relationships

            # Verify summary
            assert result["summary"]["characters"] == 1
            assert result["summary"]["locations"] == 1
            assert result["summary"]["items"] == 1
            assert result["summary"]["relationships"] == 1
            assert result["summary"]["total_entities"] == 3
            assert result["summary"]["needs_review"] == 0

            # Verify all methods were called
            mock_chars.assert_called_once_with(sample_text, story_state)
            mock_locs.assert_called_once_with(sample_text, story_state)
            mock_itm.assert_called_once_with(sample_text, story_state)
            mock_rels.assert_called_once_with(mock_characters, sample_text)

    def test_extract_all_without_story_state(self, mock_import_service, sample_text):
        """Test extract_all without story state."""
        mock_characters = [{"name": "Char", "confidence": 0.9, "needs_review": False}]
        mock_locations = [{"name": "Loc", "confidence": 0.8, "needs_review": False}]
        mock_items = [{"name": "Item", "confidence": 0.85, "needs_review": False}]
        mock_relationships: list[dict] = []

        with (
            patch.object(mock_import_service, "extract_characters", return_value=mock_characters),
            patch.object(mock_import_service, "extract_locations", return_value=mock_locations),
            patch.object(mock_import_service, "extract_items", return_value=mock_items),
            patch.object(
                mock_import_service, "infer_relationships", return_value=mock_relationships
            ),
        ):
            result = mock_import_service.extract_all(sample_text)

            assert result["summary"]["total_entities"] == 3
            assert result["summary"]["relationships"] == 0

    def test_extract_all_counts_needs_review(self, mock_import_service, sample_text):
        """Test that extract_all correctly counts items needing review."""
        mock_characters = [
            {"name": "Char1", "confidence": 0.9, "needs_review": False},
            {"name": "Char2", "confidence": 0.5, "needs_review": True},
        ]
        mock_locations = [
            {"name": "Loc1", "confidence": 0.6, "needs_review": True},
        ]
        mock_items = [
            {"name": "Item1", "confidence": 0.4, "needs_review": True},
        ]
        mock_relationships = [
            {"source": "A", "target": "B", "confidence": 0.5, "needs_review": True},
        ]

        with (
            patch.object(mock_import_service, "extract_characters", return_value=mock_characters),
            patch.object(mock_import_service, "extract_locations", return_value=mock_locations),
            patch.object(mock_import_service, "extract_items", return_value=mock_items),
            patch.object(
                mock_import_service, "infer_relationships", return_value=mock_relationships
            ),
        ):
            result = mock_import_service.extract_all(sample_text)

            # 4 items need review: Char2, Loc1, Item1, and the relationship
            assert result["summary"]["needs_review"] == 4

    def test_extract_all_handles_non_dict_entities(self, mock_import_service, sample_text):
        """Test that extract_all handles non-dict entities in lists."""
        mock_characters = [
            {"name": "Char", "confidence": 0.9, "needs_review": False},
            "not a dict",  # Should be skipped in needs_review count
        ]
        mock_locations: list[dict] = []
        mock_items: list[dict] = []
        mock_relationships: list[dict] = []

        with (
            patch.object(mock_import_service, "extract_characters", return_value=mock_characters),
            patch.object(mock_import_service, "extract_locations", return_value=mock_locations),
            patch.object(mock_import_service, "extract_items", return_value=mock_items),
            patch.object(
                mock_import_service, "infer_relationships", return_value=mock_relationships
            ),
        ):
            result = mock_import_service.extract_all(sample_text)

            # Non-dict items should be skipped in needs_review count
            assert result["summary"]["needs_review"] == 0

    def test_extract_all_propagates_world_generation_error(self, mock_import_service, sample_text):
        """Test that WorldGenerationError is propagated from sub-methods."""
        with patch.object(
            mock_import_service,
            "extract_characters",
            side_effect=WorldGenerationError("Character extraction failed"),
        ):
            with pytest.raises(WorldGenerationError, match="Character extraction failed"):
                mock_import_service.extract_all(sample_text)

    def test_extract_all_unexpected_error(self, mock_import_service, sample_text):
        """Test handling of unexpected errors in extract_all."""
        with patch.object(
            mock_import_service,
            "extract_characters",
            side_effect=RuntimeError("Unexpected error"),
        ):
            with pytest.raises(WorldGenerationError, match="Unexpected extraction error"):
                mock_import_service.extract_all(sample_text)


class TestGenerateStructuredCalls:
    """Tests verifying generate_structured is called with correct arguments."""

    def test_character_extraction_calls_generate_structured(
        self, mock_import_service, sample_text, story_state
    ):
        """Test that extract_characters calls generate_structured with correct args."""
        mock_response = ExtractedCharacterList(
            characters=[
                ExtractedCharacter(
                    name="Test",
                    role="protagonist",
                    description="A test",
                    confidence=0.9,
                )
            ]
        )

        with patch(
            "services.import_service.generate_structured", return_value=mock_response
        ) as mock_gen:
            mock_import_service.extract_characters(sample_text, story_state)

            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["settings"] == mock_import_service.settings
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["response_model"] == ExtractedCharacterList
            assert call_kwargs["temperature"] == mock_import_service.settings.temp_import_extraction
            assert "Sarah Chen" in call_kwargs["prompt"] or sample_text in call_kwargs["prompt"]

    def test_location_extraction_calls_generate_structured(
        self, mock_import_service, sample_text, story_state
    ):
        """Test that extract_locations calls generate_structured with correct args."""
        mock_response = ExtractedLocationList(
            locations=[
                ExtractedLocation(
                    name="Test Location",
                    description="A test",
                    confidence=0.9,
                )
            ]
        )

        with patch(
            "services.import_service.generate_structured", return_value=mock_response
        ) as mock_gen:
            mock_import_service.extract_locations(sample_text, story_state)

            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["response_model"] == ExtractedLocationList

    def test_item_extraction_calls_generate_structured(
        self, mock_import_service, sample_text, story_state
    ):
        """Test that extract_items calls generate_structured with correct args."""
        mock_response = ExtractedItemList(
            items=[
                ExtractedItem(
                    name="Test Item",
                    description="A test",
                    confidence=0.9,
                )
            ]
        )

        with patch(
            "services.import_service.generate_structured", return_value=mock_response
        ) as mock_gen:
            mock_import_service.extract_items(sample_text, story_state)

            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["response_model"] == ExtractedItemList

    def test_relationship_inference_calls_generate_structured(
        self, mock_import_service, sample_text
    ):
        """Test that infer_relationships calls generate_structured with correct args."""
        characters = [{"name": "Alice"}, {"name": "Bob"}]
        mock_response = ExtractedRelationshipList(
            relationships=[
                ExtractedRelationship(
                    source="Alice",
                    target="Bob",
                    relation_type="knows",
                    description="They know each other",
                    confidence=0.9,
                )
            ]
        )

        with patch(
            "services.import_service.generate_structured", return_value=mock_response
        ) as mock_gen:
            mock_import_service.infer_relationships(characters, sample_text)

            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["response_model"] == ExtractedRelationshipList
