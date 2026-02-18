"""Tests for the import service - error handling paths.

These tests verify error handling when structured generation fails.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.import_service import ImportService
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError


class TestImportServiceStructuredGenerationErrors:
    """Tests for structured generation error handling in ImportService."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        settings = Settings()
        settings.temp_import_extraction = 0.3
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

    def test_extract_characters_raises_on_generation_error(self, import_service, caplog):
        """Test extract_characters raises WorldGenerationError on generation failure."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("LLM validation failed")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError, match="Character extraction failed"):
                    import_service.extract_characters("Some story text with characters.")

    def test_extract_locations_raises_on_generation_error(self, import_service, caplog):
        """Test extract_locations raises WorldGenerationError on generation failure."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("LLM validation failed")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError, match="Location extraction failed"):
                    import_service.extract_locations("Some story text with locations.")

    def test_extract_items_raises_on_generation_error(self, import_service, caplog):
        """Test extract_items raises WorldGenerationError on generation failure."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("LLM validation failed")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError, match="Item extraction failed"):
                    import_service.extract_items("Some story text with items.")

    def test_infer_relationships_raises_on_generation_error(self, import_service, caplog):
        """Test infer_relationships raises WorldGenerationError on generation failure."""
        import logging

        characters = [{"name": "Alice"}, {"name": "Bob"}]

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("LLM validation failed")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError, match="Relationship inference failed"):
                    import_service.infer_relationships(characters, "Alice and Bob are friends.")

    def test_extract_characters_logs_generation_error(self, import_service, caplog):
        """Test extract_characters logs the generation error."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("Validation error")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError):
                    import_service.extract_characters("Test text")

            assert "Character extraction" in caplog.text or "extraction failed" in caplog.text

    def test_extract_locations_logs_generation_error(self, import_service, caplog):
        """Test extract_locations logs the generation error."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("Validation error")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError):
                    import_service.extract_locations("Test text")

            assert "Location extraction" in caplog.text or "extraction failed" in caplog.text

    def test_extract_items_logs_generation_error(self, import_service, caplog):
        """Test extract_items logs the generation error."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("Validation error")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError):
                    import_service.extract_items("Test text")

            assert "Item extraction" in caplog.text or "extraction failed" in caplog.text

    def test_infer_relationships_logs_generation_error(self, import_service, caplog):
        """Test infer_relationships logs the generation error."""
        import logging

        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = Exception("Validation error")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(WorldGenerationError):
                    import_service.infer_relationships([{"name": "Alice"}], "Test text")

            assert "Relationship inference" in caplog.text or "inference failed" in caplog.text


class TestImportServiceConnectionErrors:
    """Tests for connection error handling in extraction methods."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        settings = Settings()
        settings.temp_import_extraction = 0.3
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

    def test_extract_characters_raises_on_connection_error(self, import_service):
        """Test extract_characters raises WorldGenerationError on ConnectionError."""
        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = ConnectionError("Connection refused")

            with pytest.raises(WorldGenerationError, match="Character extraction failed"):
                import_service.extract_characters("Test text")

    def test_extract_locations_raises_on_connection_error(self, import_service):
        """Test extract_locations raises WorldGenerationError on ConnectionError."""
        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = ConnectionError("Connection refused")

            with pytest.raises(WorldGenerationError, match="Location extraction failed"):
                import_service.extract_locations("Test text")

    def test_extract_items_raises_on_connection_error(self, import_service):
        """Test extract_items raises WorldGenerationError on ConnectionError."""
        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = ConnectionError("Connection refused")

            with pytest.raises(WorldGenerationError, match="Item extraction failed"):
                import_service.extract_items("Test text")

    def test_infer_relationships_raises_on_connection_error(self, import_service):
        """Test infer_relationships raises WorldGenerationError on ConnectionError."""
        with patch("src.services.import_service.generate_structured") as mock_gen:
            mock_gen.side_effect = ConnectionError("Connection refused")

            with pytest.raises(WorldGenerationError, match="Relationship inference failed"):
                import_service.infer_relationships([{"name": "Alice"}], "Test text")
