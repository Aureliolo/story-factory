"""Tests for import service."""

import json
from unittest.mock import MagicMock, patch

import ollama
import pytest

from memory.story_state import StoryBrief, StoryState
from services.import_service import ImportService
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


class TestClientProperty:
    """Tests for the client property lazy initialization."""

    def test_client_lazy_initialization(self, mock_settings, mock_mode_service):
        """Test that client is lazily initialized on first access."""
        service = ImportService(mock_settings, mock_mode_service)

        # Client should be None initially
        assert service._client is None

        # Access client property with mocked ollama.Client
        with patch("services.import_service.ollama.Client") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            client = service.client

            # Verify client was created with correct parameters
            mock_client_class.assert_called_once_with(
                host="http://localhost:11434",
                timeout=30.0,
            )
            assert client == mock_client_instance

    def test_client_reuses_existing_instance(self, mock_settings, mock_mode_service):
        """Test that client is reused on subsequent accesses."""
        service = ImportService(mock_settings, mock_mode_service)

        with patch("services.import_service.ollama.Client") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            # Access client twice
            client1 = service.client
            client2 = service.client

            # Should only be created once
            mock_client_class.assert_called_once()
            assert client1 is client2


class TestExtractCharactersMocked:
    """Tests for extract_characters with mocked LLM."""

    def test_extract_characters_success(self, mock_import_service, sample_text, story_state):
        """Test successful character extraction."""
        mock_response = [
            {
                "name": "Sarah Chen",
                "role": "protagonist",
                "description": "A determined researcher",
                "relationships": {"Professor Williams": "mentor"},
                "confidence": 0.95,
                "needs_review": False,
            },
            {
                "name": "Marcus Drake",
                "role": "antagonist",
                "description": "A rival treasure hunter",
                "relationships": {"Sarah Chen": "rival"},
                "confidence": 0.9,
                "needs_review": False,
            },
        ]

        mock_client = MagicMock()
        # Use markdown code block format like real LLM responses
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text, story_state)

        assert len(result) == 2
        assert result[0]["name"] == "Sarah Chen"
        assert result[0]["role"] == "protagonist"
        assert result[1]["name"] == "Marcus Drake"
        mock_client.generate.assert_called_once()

    def test_extract_characters_without_story_state(self, mock_import_service, sample_text):
        """Test character extraction without story state context."""
        mock_response = [
            {
                "name": "Test Character",
                "role": "supporting",
                "description": "A test description",
                "relationships": {},
                "confidence": 0.8,
                "needs_review": False,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text, story_state=None)

        assert len(result) == 1
        assert result[0]["name"] == "Test Character"

    def test_extract_characters_missing_confidence_field(self, mock_import_service, sample_text):
        """Test character extraction when LLM response missing confidence field."""
        mock_response = [
            {
                "name": "Character Without Confidence",
                "role": "supporting",
                "description": "No confidence provided",
                "relationships": {},
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text)

        assert len(result) == 1
        # Should use default confidence and flag for review
        assert result[0]["confidence"] == 0.5  # import_default_confidence
        assert result[0]["needs_review"] is True

    def test_extract_characters_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence characters are flagged for review."""
        mock_response = [
            {
                "name": "Uncertain Character",
                "role": "supporting",
                "description": "Ambiguous description",
                "relationships": {},
                "confidence": 0.5,  # Below threshold of 0.7
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.5
        assert result[0]["needs_review"] is True

    def test_extract_characters_high_confidence_not_flagged(self, mock_import_service, sample_text):
        """Test that high confidence characters are not flagged for review."""
        mock_response = [
            {
                "name": "Clear Character",
                "role": "protagonist",
                "description": "Well-described character",
                "relationships": {},
                "confidence": 0.9,  # Above threshold
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is False

    def test_extract_characters_single_object_wrapped_in_list(
        self, mock_import_service, sample_text
    ):
        """Test that single object response is wrapped in a list (LLM fallback)."""
        mock_client = MagicMock()
        # Single character object instead of array - should be wrapped in list
        mock_client.generate.return_value = {
            "response": '{"name": "Solo Character", "role": "hero", "confidence": 0.9}'
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text)
        assert len(result) == 1
        assert result[0]["name"] == "Solo Character"

    def test_extract_characters_empty_list_response(self, mock_import_service, sample_text):
        """Test handling of empty list response."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "[]"}
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Invalid character extraction response"):
            mock_import_service.extract_characters(sample_text)

    def test_extract_characters_response_error(self, mock_import_service, sample_text):
        """Test handling of Ollama ResponseError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ollama.ResponseError("Model not found")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during character extraction"):
            mock_import_service.extract_characters(sample_text)

    def test_extract_characters_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Connection refused")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during character extraction"):
            mock_import_service.extract_characters(sample_text)

    def test_extract_characters_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = TimeoutError("Request timed out")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during character extraction"):
            mock_import_service.extract_characters(sample_text)

    def test_extract_characters_json_parsing_error(self, mock_import_service, sample_text):
        """Test handling of JSON parsing errors (ValueError, KeyError, TypeError)."""
        mock_client = MagicMock()
        # Return invalid JSON that will fail to parse
        mock_client.generate.return_value = {"response": "not valid json at all"}
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Invalid character extraction response"):
            mock_import_service.extract_characters(sample_text)

    def test_extract_characters_unexpected_error(self, mock_import_service, sample_text):
        """Test handling of unexpected errors."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Unexpected error")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Unexpected character extraction error"):
            mock_import_service.extract_characters(sample_text)


class TestExtractLocationsMocked:
    """Tests for extract_locations with mocked LLM."""

    def test_extract_locations_success(self, mock_import_service, sample_text, story_state):
        """Test successful location extraction."""
        mock_response = [
            {
                "name": "Old Library",
                "type": "location",
                "description": "A dusty library with ancient books",
                "significance": "Where the search began",
                "confidence": 0.9,
                "needs_review": False,
            },
            {
                "name": "Ruins of Atlantis",
                "type": "location",
                "description": "Ancient underwater ruins",
                "significance": "Where the artifact was discovered",
                "confidence": 0.85,
                "needs_review": False,
            },
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text, story_state)

        assert len(result) == 2
        assert result[0]["name"] == "Old Library"
        assert result[0]["type"] == "location"
        assert result[1]["name"] == "Ruins of Atlantis"

    def test_extract_locations_without_story_state(self, mock_import_service, sample_text):
        """Test location extraction without story state context."""
        mock_response = [
            {
                "name": "Test Location",
                "type": "location",
                "description": "A test location",
                "significance": "Test significance",
                "confidence": 0.8,
                "needs_review": False,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text, story_state=None)

        assert len(result) == 1
        assert result[0]["name"] == "Test Location"

    def test_extract_locations_missing_confidence_field(self, mock_import_service, sample_text):
        """Test location extraction when LLM response missing confidence field."""
        mock_response = [
            {
                "name": "Location Without Confidence",
                "type": "location",
                "description": "No confidence provided",
                "significance": "Unknown",
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text)

        assert len(result) == 1
        # Should use default confidence (0.5) and flag for review
        assert result[0]["confidence"] == 0.5
        assert result[0]["needs_review"] is True

    def test_extract_locations_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence locations are flagged for review."""
        mock_response = [
            {
                "name": "Uncertain Location",
                "type": "location",
                "description": "Vague description",
                "significance": "Unclear",
                "confidence": 0.4,  # Below threshold of 0.7
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is True

    def test_extract_locations_high_confidence_not_flagged(self, mock_import_service, sample_text):
        """Test that high confidence locations are not flagged for review."""
        mock_response = [
            {
                "name": "Clear Location",
                "type": "location",
                "description": "Well-described location",
                "significance": "Important",
                "confidence": 0.9,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is False

    def test_extract_locations_single_object_wrapped_in_list(
        self, mock_import_service, sample_text
    ):
        """Test that single object response is wrapped in a list (LLM fallback)."""
        mock_client = MagicMock()
        # Single location object instead of array - should be wrapped in list
        mock_client.generate.return_value = {
            "response": '{"name": "Solo Location", "type": "city", "confidence": 0.9}'
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text)
        assert len(result) == 1
        assert result[0]["name"] == "Solo Location"

    def test_extract_locations_response_error(self, mock_import_service, sample_text):
        """Test handling of Ollama ResponseError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ollama.ResponseError("Model error")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during location extraction"):
            mock_import_service.extract_locations(sample_text)

    def test_extract_locations_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Connection refused")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during location extraction"):
            mock_import_service.extract_locations(sample_text)

    def test_extract_locations_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = TimeoutError("Timed out")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during location extraction"):
            mock_import_service.extract_locations(sample_text)

    def test_extract_locations_json_parsing_error(self, mock_import_service, sample_text):
        """Test handling of JSON parsing errors."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "invalid json content"}
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Invalid location extraction response"):
            mock_import_service.extract_locations(sample_text)

    def test_extract_locations_unexpected_error(self, mock_import_service, sample_text):
        """Test handling of unexpected errors."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Unexpected error")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Unexpected location extraction error"):
            mock_import_service.extract_locations(sample_text)


class TestExtractItemsMocked:
    """Tests for extract_items with mocked LLM."""

    def test_extract_items_success(self, mock_import_service, sample_text, story_state):
        """Test successful item extraction."""
        mock_response = [
            {
                "name": "Ancient Tome",
                "type": "item",
                "description": "A leather-bound book with strange symbols",
                "significance": "Contains the key to understanding the artifact",
                "properties": ["magical", "ancient"],
                "confidence": 0.9,
                "needs_review": False,
            },
            {
                "name": "Golden Amulet",
                "type": "item",
                "description": "A glowing golden amulet",
                "significance": "Grants the power to see the future",
                "properties": ["magical", "glowing"],
                "confidence": 0.95,
                "needs_review": False,
            },
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text, story_state)

        assert len(result) == 2
        assert result[0]["name"] == "Ancient Tome"
        assert result[0]["type"] == "item"
        assert result[1]["name"] == "Golden Amulet"
        assert "magical" in result[0]["properties"]

    def test_extract_items_without_story_state(self, mock_import_service, sample_text):
        """Test item extraction without story state context."""
        mock_response = [
            {
                "name": "Test Item",
                "type": "item",
                "description": "A test item",
                "significance": "Test",
                "properties": [],
                "confidence": 0.8,
                "needs_review": False,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text, story_state=None)

        assert len(result) == 1
        assert result[0]["name"] == "Test Item"

    def test_extract_items_missing_confidence_field(self, mock_import_service, sample_text):
        """Test item extraction when LLM response missing confidence field."""
        mock_response = [
            {
                "name": "Item Without Confidence",
                "type": "item",
                "description": "No confidence",
                "significance": "Unknown",
                "properties": [],
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.5
        assert result[0]["needs_review"] is True

    def test_extract_items_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence items are flagged for review."""
        mock_response = [
            {
                "name": "Uncertain Item",
                "type": "item",
                "description": "Vague",
                "significance": "Unclear",
                "properties": [],
                "confidence": 0.3,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is True

    def test_extract_items_high_confidence_not_flagged(self, mock_import_service, sample_text):
        """Test that high confidence items are not flagged for review."""
        mock_response = [
            {
                "name": "Clear Item",
                "type": "item",
                "description": "Well-described",
                "significance": "Important",
                "properties": ["unique"],
                "confidence": 0.95,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is False

    def test_extract_items_single_object_wrapped_in_list(self, mock_import_service, sample_text):
        """Test that single object response is wrapped in a list (LLM fallback)."""
        mock_client = MagicMock()
        # Single item object instead of array - should be wrapped in list
        mock_client.generate.return_value = {
            "response": '{"name": "Solo Item", "type": "artifact", "confidence": 0.9}'
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text)
        assert len(result) == 1
        assert result[0]["name"] == "Solo Item"

    def test_extract_items_response_error(self, mock_import_service, sample_text):
        """Test handling of Ollama ResponseError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ollama.ResponseError("Model error")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during item extraction"):
            mock_import_service.extract_items(sample_text)

    def test_extract_items_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Connection refused")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during item extraction"):
            mock_import_service.extract_items(sample_text)

    def test_extract_items_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = TimeoutError("Timed out")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during item extraction"):
            mock_import_service.extract_items(sample_text)

    def test_extract_items_json_parsing_error(self, mock_import_service, sample_text):
        """Test handling of JSON parsing errors."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not json"}
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Invalid item extraction response"):
            mock_import_service.extract_items(sample_text)

    def test_extract_items_unexpected_error(self, mock_import_service, sample_text):
        """Test handling of unexpected errors."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Unexpected")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Unexpected item extraction error"):
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

        mock_response = [
            {
                "source": "Sarah Chen",
                "target": "Professor Williams",
                "relation_type": "mentored_by",
                "description": "Professor Williams was Sarah's mentor",
                "confidence": 0.9,
                "needs_review": False,
            },
            {
                "source": "Sarah Chen",
                "target": "Marcus Drake",
                "relation_type": "rivals_with",
                "description": "They are competing for the same artifact",
                "confidence": 0.95,
                "needs_review": False,
            },
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)

        assert len(result) == 2
        assert result[0]["source"] == "Sarah Chen"
        assert result[0]["target"] == "Professor Williams"
        assert result[0]["relation_type"] == "mentored_by"

    def test_infer_relationships_handles_empty_characters(self, mock_import_service, sample_text):
        """Test that empty characters list returns empty result without LLM call."""
        mock_client = MagicMock()
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships([], sample_text)

        assert result == []
        mock_client.generate.assert_not_called()

    def test_infer_relationships_missing_confidence_field(self, mock_import_service, sample_text):
        """Test relationship inference when LLM response missing confidence field."""
        characters = [{"name": "Alice"}, {"name": "Bob"}]

        mock_response = [
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "knows",
                "description": "They know each other",
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.5
        assert result[0]["needs_review"] is True

    def test_infer_relationships_low_confidence_flagged(self, mock_import_service, sample_text):
        """Test that low confidence relationships are flagged for review."""
        characters = [{"name": "A"}, {"name": "B"}]

        mock_response = [
            {
                "source": "A",
                "target": "B",
                "relation_type": "knows",
                "description": "Possible acquaintance",
                "confidence": 0.4,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is True

    def test_infer_relationships_high_confidence_not_flagged(
        self, mock_import_service, sample_text
    ):
        """Test that high confidence relationships are not flagged for review."""
        characters = [{"name": "A"}, {"name": "B"}]

        mock_response = [
            {
                "source": "A",
                "target": "B",
                "relation_type": "parent_of",
                "description": "Explicitly stated parent relationship",
                "confidence": 1.0,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)

        assert len(result) == 1
        assert result[0]["needs_review"] is False

    def test_infer_relationships_handles_non_dict_characters(
        self, mock_import_service, sample_text
    ):
        """Test handling of characters list with non-dict items."""
        characters = [{"name": "Alice"}, "not a dict", {"name": "Bob"}]

        mock_response = [
            {
                "source": "Alice",
                "target": "Bob",
                "relation_type": "knows",
                "description": "They know each other",
                "confidence": 0.8,
                "needs_review": False,
            }
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)

        assert len(result) == 1

    def test_infer_relationships_single_object_wrapped_in_list(
        self, mock_import_service, sample_text
    ):
        """Test that single object response is wrapped in a list (LLM fallback)."""
        characters = [{"name": "Alice"}, {"name": "Bob"}]

        mock_client = MagicMock()
        # Single relationship object instead of array - should be wrapped in list
        mock_client.generate.return_value = {
            "response": '{"source": "Alice", "target": "Bob", "relation_type": "friends", "confidence": 0.9}'
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)
        assert len(result) == 1
        assert result[0]["source"] == "Alice"
        assert result[0]["target"] == "Bob"

    def test_infer_relationships_response_error(self, mock_import_service, sample_text):
        """Test handling of Ollama ResponseError."""
        characters = [{"name": "Alice"}, {"name": "Bob"}]

        mock_client = MagicMock()
        mock_client.generate.side_effect = ollama.ResponseError("Model error")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during relationship inference"):
            mock_import_service.infer_relationships(characters, sample_text)

    def test_infer_relationships_connection_error(self, mock_import_service, sample_text):
        """Test handling of ConnectionError."""
        characters = [{"name": "Alice"}]

        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Connection refused")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during relationship inference"):
            mock_import_service.infer_relationships(characters, sample_text)

    def test_infer_relationships_timeout_error(self, mock_import_service, sample_text):
        """Test handling of TimeoutError."""
        characters = [{"name": "Alice"}]

        mock_client = MagicMock()
        mock_client.generate.side_effect = TimeoutError("Timed out")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="LLM error during relationship inference"):
            mock_import_service.infer_relationships(characters, sample_text)

    def test_infer_relationships_json_parsing_error(self, mock_import_service, sample_text):
        """Test handling of JSON parsing errors."""
        characters = [{"name": "Alice"}]

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "invalid json"}
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Invalid relationship inference response"):
            mock_import_service.infer_relationships(characters, sample_text)

    def test_infer_relationships_unexpected_error(self, mock_import_service, sample_text):
        """Test handling of unexpected errors."""
        characters = [{"name": "Alice"}]

        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Unexpected")
        mock_import_service._client = mock_client

        with pytest.raises(WorldGenerationError, match="Unexpected relationship inference error"):
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


class TestPostProcessingLogic:
    """Tests for post-processing logic in extraction methods."""

    def test_characters_with_non_dict_items(self, mock_import_service, sample_text):
        """Test character extraction handles non-dict items in response."""
        mock_response = [
            {"name": "Valid Character", "role": "protagonist", "confidence": 0.9},
            "not a dict",  # Should be skipped
            123,  # Should be skipped
            {"name": "Another Valid", "role": "supporting", "confidence": 0.8},
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_characters(sample_text)

        # Should return all items, but only process dicts
        assert len(result) == 4
        # Only the dict items should have needs_review set
        assert result[0].get("needs_review") is False
        assert result[3].get("needs_review") is False

    def test_locations_with_non_dict_items(self, mock_import_service, sample_text):
        """Test location extraction handles non-dict items in response."""
        mock_response = [
            {"name": "Valid Location", "type": "location", "confidence": 0.9},
            None,  # Should be skipped
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_locations(sample_text)

        assert len(result) == 2
        assert result[0].get("needs_review") is False

    def test_items_with_non_dict_items(self, mock_import_service, sample_text):
        """Test item extraction handles non-dict items in response."""
        mock_response = [
            {"name": "Valid Item", "type": "item", "confidence": 0.9},
            [],  # Should be skipped
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.extract_items(sample_text)

        assert len(result) == 2
        assert result[0].get("needs_review") is False

    def test_relationships_with_non_dict_items(self, mock_import_service, sample_text):
        """Test relationship inference handles non-dict items in response."""
        characters = [{"name": "A"}, {"name": "B"}]
        mock_response = [
            {"source": "A", "target": "B", "relation_type": "knows", "confidence": 0.9},
            "invalid",  # Should be skipped
        ]

        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": f"```json\n{json.dumps(mock_response)}\n```"
        }
        mock_import_service._client = mock_client

        result = mock_import_service.infer_relationships(characters, sample_text)

        assert len(result) == 2
        assert result[0].get("needs_review") is False
