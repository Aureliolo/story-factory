"""Tests for the import service."""

from unittest.mock import MagicMock

import pytest

from services.import_service import ImportService
from settings import Settings
from utils.exceptions import WorldGenerationError


class TestImportServiceJsonParsingErrors:
    """Tests for JSON parsing error handling in ImportService."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        settings = Settings()
        settings.temp_import_extraction = 0.3
        settings.llm_tokens_character_create = 500
        settings.llm_tokens_location_create = 400
        settings.llm_tokens_item_create = 400
        settings.llm_tokens_relationship_create = 300
        settings.import_character_token_multiplier = 4
        settings.import_confidence_threshold = 0.7
        settings.import_default_confidence = 0.5
        return settings

    @pytest.fixture
    def mock_mode_service(self):
        """Create a mock ModelModeService."""
        mock = MagicMock()
        mock.get_model_for_agent.return_value = "test-model:8b"
        return mock

    @pytest.fixture
    def import_service(self, settings, mock_mode_service):
        """Create an ImportService with mocked dependencies."""
        return ImportService(settings, mock_mode_service)

    def test_extract_characters_raises_on_json_parsing_error(
        self, import_service, monkeypatch, caplog
    ):
        """Test extract_characters raises WorldGenerationError on JSON parsing error."""
        # Lines 165-166: JSON parsing error handling in extract_characters
        import logging

        mock_client = MagicMock()
        # Return something that causes extract_json to fail (non-JSON response)
        mock_client.generate.return_value = {"response": "This is not JSON at all"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(WorldGenerationError, match="Invalid character extraction"):
                import_service.extract_characters("Some story text with characters.")

    def test_extract_locations_raises_on_json_parsing_error(
        self, import_service, monkeypatch, caplog
    ):
        """Test extract_locations raises WorldGenerationError on JSON parsing error."""
        # Lines 282-283: JSON parsing error handling in extract_locations
        import logging

        mock_client = MagicMock()
        # Return response that causes extract_json to return None or non-list
        mock_client.generate.return_value = {"response": "no json here"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(WorldGenerationError, match="Invalid location extraction"):
                import_service.extract_locations("Some story text with locations.")

    def test_extract_items_raises_on_json_parsing_error(self, import_service, monkeypatch, caplog):
        """Test extract_items raises WorldGenerationError on JSON parsing error."""
        # Lines 400-401: JSON parsing error handling in extract_items
        import logging

        mock_client = MagicMock()
        # Return response that causes extract_json to return None
        mock_client.generate.return_value = {"response": "not a json response"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(WorldGenerationError, match="Invalid item extraction"):
                import_service.extract_items("Some story text with items.")

    def test_infer_relationships_raises_on_json_parsing_error(
        self, import_service, monkeypatch, caplog
    ):
        """Test infer_relationships raises WorldGenerationError on JSON parsing error."""
        # Lines 516-517: JSON parsing error handling in infer_relationships
        import logging

        mock_client = MagicMock()
        # Return response that causes extract_json to return None
        mock_client.generate.return_value = {"response": "invalid response"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        characters = [{"name": "Alice"}, {"name": "Bob"}]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(WorldGenerationError, match="Invalid relationship inference"):
                import_service.infer_relationships(characters, "Alice and Bob are friends.")

    def test_extract_characters_logs_json_parsing_error(self, import_service, monkeypatch, caplog):
        """Test extract_characters logs the JSON parsing error."""
        import logging

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not json"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR), pytest.raises(WorldGenerationError):
            import_service.extract_characters("Test text")

        assert "Character extraction" in caplog.text or "Invalid character" in caplog.text

    def test_extract_locations_logs_json_parsing_error(self, import_service, monkeypatch, caplog):
        """Test extract_locations logs the JSON parsing error."""
        import logging

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not json"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR), pytest.raises(WorldGenerationError):
            import_service.extract_locations("Test text")

        assert "Location extraction" in caplog.text or "Invalid location" in caplog.text

    def test_extract_items_logs_json_parsing_error(self, import_service, monkeypatch, caplog):
        """Test extract_items logs the JSON parsing error."""
        import logging

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not json"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR), pytest.raises(WorldGenerationError):
            import_service.extract_items("Test text")

        assert "Item extraction" in caplog.text or "Invalid item" in caplog.text

    def test_infer_relationships_logs_json_parsing_error(self, import_service, monkeypatch, caplog):
        """Test infer_relationships logs the JSON parsing error."""
        import logging

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "not json"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        with caplog.at_level(logging.ERROR), pytest.raises(WorldGenerationError):
            import_service.infer_relationships([{"name": "Alice"}], "Test text")

        assert "Relationship inference" in caplog.text or "Invalid relationship" in caplog.text


class TestImportServiceValueErrors:
    """Tests for ValueError/KeyError/TypeError handling in extraction methods."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        settings = Settings()
        settings.temp_import_extraction = 0.3
        settings.llm_tokens_character_create = 500
        settings.llm_tokens_location_create = 400
        settings.llm_tokens_item_create = 400
        settings.llm_tokens_relationship_create = 300
        settings.import_character_token_multiplier = 4
        settings.import_confidence_threshold = 0.7
        settings.import_default_confidence = 0.5
        return settings

    @pytest.fixture
    def mock_mode_service(self):
        """Create a mock ModelModeService."""
        mock = MagicMock()
        mock.get_model_for_agent.return_value = "test-model:8b"
        return mock

    @pytest.fixture
    def import_service(self, settings, mock_mode_service):
        """Create an ImportService with mocked dependencies."""
        return ImportService(settings, mock_mode_service)

    def test_extract_characters_raises_on_value_error(self, import_service, monkeypatch):
        """Test extract_characters raises WorldGenerationError on ValueError."""
        # Lines 165-166: ValueError handler - mock extract_json_list to raise ValueError
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "valid response"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        # Mock extract_json_list to raise ValueError
        def mock_extract_json_list_raises(response, strict=True):
            raise ValueError("JSON parsing failed")

        monkeypatch.setattr(
            "services.import_service.extract_json_list", mock_extract_json_list_raises
        )

        with pytest.raises(WorldGenerationError, match="Invalid character extraction"):
            import_service.extract_characters("Test text")

    def test_extract_locations_raises_on_value_error(self, import_service, monkeypatch):
        """Test extract_locations raises WorldGenerationError on ValueError."""
        # Lines 282-283: ValueError handler
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "valid response"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        def mock_extract_json_list_raises(response, strict=True):
            raise KeyError("Missing required field")

        monkeypatch.setattr(
            "services.import_service.extract_json_list", mock_extract_json_list_raises
        )

        with pytest.raises(WorldGenerationError, match="Invalid location extraction"):
            import_service.extract_locations("Test text")

    def test_extract_items_raises_on_value_error(self, import_service, monkeypatch):
        """Test extract_items raises WorldGenerationError on ValueError."""
        # Lines 400-401: ValueError handler
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "valid response"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        def mock_extract_json_list_raises(response, strict=True):
            raise TypeError("Invalid type for processing")

        monkeypatch.setattr(
            "services.import_service.extract_json_list", mock_extract_json_list_raises
        )

        with pytest.raises(WorldGenerationError, match="Invalid item extraction"):
            import_service.extract_items("Test text")

    def test_infer_relationships_raises_on_value_error(self, import_service, monkeypatch):
        """Test infer_relationships raises WorldGenerationError on ValueError."""
        # Lines 516-517: ValueError handler
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "valid response"}
        monkeypatch.setattr(import_service, "_client", mock_client)

        def mock_extract_json_list_raises(response, strict=True):
            raise ValueError("Invalid JSON structure")

        monkeypatch.setattr(
            "services.import_service.extract_json_list", mock_extract_json_list_raises
        )

        with pytest.raises(WorldGenerationError, match="Invalid relationship inference"):
            import_service.infer_relationships([{"name": "Alice"}], "Test text")
