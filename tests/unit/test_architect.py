"""Tests for the Architect agent."""

import logging
from unittest.mock import patch

import pytest

from agents.architect import ArchitectAgent
from memory.story_state import StoryBrief
from settings import Settings


class TestArchitectParseVariationResponse:
    """Tests for _parse_variation_response error handling."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def architect(self, settings):
        """Create an ArchitectAgent for testing."""
        with patch("agents.base.ollama.Client"):
            agent = ArchitectAgent(model="test-model", settings=settings)
        return agent

    @pytest.fixture
    def brief(self):
        """Create a sample StoryBrief."""
        return StoryBrief(
            premise="A hero's journey",
            genre="Fantasy",
            subgenres=["Adventure"],
            tone="Epic",
            themes=["Courage"],
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novella",
            language="English",
            content_rating="none",
        )

    def test_parse_variation_response_logs_warning_on_character_parse_failure(
        self, architect, brief, caplog
    ):
        """Test _parse_variation_response logs warning when character parsing fails."""
        # Lines 436-437: Warning when character parsing fails
        # Create a response with invalid character JSON that will cause parse error
        response = """
        World description: A magical realm.

        Characters:
        ```json
        [{"invalid": "not a valid character"}]
        ```

        Plot summary: An epic adventure.

        Plot points:
        ```json
        []
        ```

        Chapters:
        ```json
        []
        ```
        """

        with caplog.at_level(logging.WARNING):
            variation = architect._parse_variation_response(response, 1, brief)

        # Should log warning about failed character parse
        assert "Failed to parse character" in caplog.text
        # Variation should still be created
        assert variation is not None
        assert variation.name == "Variation 1"

    def test_parse_variation_response_logs_warning_on_plot_point_parse_failure(
        self, architect, brief, caplog
    ):
        """Test _parse_variation_response logs warning when plot point parsing fails."""
        # Lines 464-465: Warning when plot point parsing fails
        response = """
        World description: A magical realm.

        Characters:
        ```json
        []
        ```

        Plot summary: An epic adventure.

        Plot points:
        ```json
        [{"invalid": "not a valid plot point"}]
        ```

        Chapters:
        ```json
        []
        ```
        """

        with caplog.at_level(logging.WARNING):
            variation = architect._parse_variation_response(response, 1, brief)

        # Should log warning about failed plot point parse
        assert "Failed to parse plot point" in caplog.text
        # Variation should still be created
        assert variation is not None

    def test_parse_variation_response_logs_warning_on_chapter_parse_failure(
        self, architect, brief, caplog
    ):
        """Test _parse_variation_response logs warning when chapter parsing fails."""
        # Lines 484-485: Warning when chapter parsing fails
        response = """
        World description: A magical realm.

        Characters:
        ```json
        []
        ```

        Plot summary: An epic adventure.

        Plot points:
        ```json
        []
        ```

        Chapters:
        ```json
        [{"invalid": "not a valid chapter"}]
        ```
        """

        with caplog.at_level(logging.WARNING):
            variation = architect._parse_variation_response(response, 1, brief)

        # Should log warning about failed chapter parse
        assert "Failed to parse chapter" in caplog.text
        # Variation should still be created
        assert variation is not None

    def test_parse_variation_response_handles_json_extraction_failure_for_characters(
        self, architect, brief, caplog, monkeypatch
    ):
        """Test _parse_variation_response handles complete JSON extraction failure for characters."""
        # Lines 436-437: Warning when extract_json_list raises exception
        import agents.architect as architect_module

        # Mock extract_json_list to raise an exception
        def mock_extract_json_list(text):
            raise ValueError("Invalid JSON format")

        monkeypatch.setattr(architect_module, "extract_json_list", mock_extract_json_list)

        response = """
        World description: A magical realm.

        Characters:
        ```json
        [{"name": "test"}]
        ```

        Plot summary: An epic adventure.
        """

        with caplog.at_level(logging.WARNING):
            variation = architect._parse_variation_response(response, 1, brief)

        # Should log warning about failed extraction
        assert "Failed to extract characters" in caplog.text
        # Variation should still be created with empty characters
        assert variation is not None
        assert len(variation.characters) == 0

    def test_parse_variation_response_handles_json_extraction_failure_for_plot_points(
        self, architect, brief, caplog, monkeypatch
    ):
        """Test _parse_variation_response handles complete JSON extraction failure for plot points."""
        # Lines 464-465: Warning when extract_json_list raises exception for plot points
        import agents.architect as architect_module

        call_count = [0]

        def mock_extract_json_list(text):
            call_count[0] += 1
            # Let first call (characters) pass, fail on second call (plot points)
            if call_count[0] == 1:
                return []  # Empty characters
            raise ValueError("Invalid JSON format for plot points")

        monkeypatch.setattr(architect_module, "extract_json_list", mock_extract_json_list)

        response = """
        World description: A magical realm.

        Characters:
        ```json
        []
        ```

        Plot summary: An epic adventure.

        Plot points:
        ```json
        [{"description": "test"}]
        ```

        Chapters:
        ```json
        []
        ```
        """

        with caplog.at_level(logging.WARNING):
            variation = architect._parse_variation_response(response, 1, brief)

        # Should log warning about failed extraction
        assert "Failed to extract plot points" in caplog.text
        # Variation should still be created with empty plot points
        assert variation is not None
        assert len(variation.plot_points) == 0

    def test_parse_variation_response_handles_json_extraction_failure_for_chapters(
        self, architect, brief, caplog, monkeypatch
    ):
        """Test _parse_variation_response handles complete JSON extraction failure for chapters."""
        # Lines 484-485: Warning when extract_json_list raises exception for chapters
        import agents.architect as architect_module

        call_count = [0]

        def mock_extract_json_list(text):
            call_count[0] += 1
            # Let first two calls pass, fail on third call (chapters)
            if call_count[0] <= 2:
                return []  # Empty for characters and plot points
            raise ValueError("Invalid JSON format for chapters")

        monkeypatch.setattr(architect_module, "extract_json_list", mock_extract_json_list)

        response = """
        World description: A magical realm.

        Characters:
        ```json
        []
        ```

        Plot summary: An epic adventure.

        Plot points:
        ```json
        []
        ```

        Chapters:
        ```json
        [{"number": 1}]
        ```
        """

        with caplog.at_level(logging.WARNING):
            variation = architect._parse_variation_response(response, 1, brief)

        # Should log warning about failed extraction
        assert "Failed to extract chapters" in caplog.text
        # Variation should still be created with empty chapters
        assert variation is not None
        assert len(variation.chapters) == 0

    def test_parse_variation_response_creates_variation_with_defaults_on_all_failures(
        self, architect, brief
    ):
        """Test _parse_variation_response creates variation even when all parsing fails."""
        # Response with no valid sections
        response = "Just some text without any proper structure or JSON"

        variation = architect._parse_variation_response(response, 2, brief)

        # Should still create a variation with defaults
        assert variation is not None
        assert variation.name == "Variation 2"
        assert "Variation 2" in variation.ai_rationale
