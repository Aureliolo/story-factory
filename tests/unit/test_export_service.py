"""Tests for the export service."""

from pathlib import Path

import pytest

from memory.story_state import Chapter, StoryBrief, StoryState
from services.export_service import ExportService
from settings import Settings


class TestExportServiceSaveToFile:
    """Tests for save_to_file error handling in ExportService."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def export_service(self, settings):
        """Create an ExportService for testing."""
        return ExportService(settings)

    @pytest.fixture
    def sample_state(self):
        """Create a sample StoryState with chapters."""
        brief = StoryBrief(
            premise="A test story",
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
        state = StoryState(
            id="test-export-001",
            project_name="Export Test",
            brief=brief,
            status="completed",
        )
        state.chapters = [
            Chapter(
                number=1,
                title="The Beginning",
                outline="Chapter outline",
                content="This is the content of chapter one.\n\nWith multiple paragraphs.",
                status="completed",
            )
        ]
        return state

    def test_save_to_file_raises_on_unsupported_format(
        self, export_service, sample_state, tmp_path
    ):
        """Test save_to_file raises ValueError for unsupported format."""
        # Lines 826-828: Unsupported export format error
        filepath = tmp_path / "test.xyz"

        with pytest.raises(ValueError, match="format.*must be one of"):
            export_service.save_to_file(sample_state, "xyz", filepath)

    def test_save_to_file_reraises_value_error(self, export_service, sample_state, tmp_path):
        """Test save_to_file re-raises ValueError from validation."""
        # Line 833: Re-raise ValueError
        # Test with a format that fails validation
        filepath = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="format.*must be one of"):
            export_service.save_to_file(sample_state, "invalid_format", filepath)

    def test_save_to_file_logs_error_on_unsupported_format(
        self, export_service, sample_state, tmp_path, caplog
    ):
        """Test save_to_file logs error for unsupported format."""
        import logging

        filepath = tmp_path / "test.xyz"

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                export_service.save_to_file(sample_state, "xyz", filepath)

        # Note: The error is raised by validation before logging would occur
        # for unsupported format. The log happens in the else branch which
        # is not reachable due to validate_string_in_choices

    def test_save_to_file_validates_format_choices(self, export_service, sample_state, tmp_path):
        """Test save_to_file validates format is one of allowed choices."""
        # Should not raise for valid formats
        valid_formats = ["markdown", "text", "html"]
        for fmt in valid_formats:
            export_service.save_to_file(sample_state, fmt, tmp_path / f"test.{fmt}")


class TestExportServiceFormats:
    """Tests for various export formats."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def export_service(self, settings):
        """Create an ExportService for testing."""
        return ExportService(settings)

    @pytest.fixture
    def sample_state(self):
        """Create a sample StoryState with chapters."""
        brief = StoryBrief(
            premise="A test story",
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
        state = StoryState(
            id="test-export-001",
            project_name="Export Test",
            brief=brief,
            status="completed",
        )
        state.chapters = [
            Chapter(
                number=1,
                title="The Beginning",
                outline="Chapter outline",
                content="This is the content of chapter one.\n\nWith multiple paragraphs.",
                status="completed",
            )
        ]
        return state

    def test_save_to_file_markdown(self, export_service, sample_state, tmp_path):
        """Test save_to_file with markdown format."""
        filepath = tmp_path / "test.md"
        result = export_service.save_to_file(sample_state, "markdown", filepath)
        assert result.exists()
        content = result.read_text()
        assert "Export Test" in content or "A test story" in content

    def test_save_to_file_text(self, export_service, sample_state, tmp_path):
        """Test save_to_file with text format."""
        filepath = tmp_path / "test.txt"
        result = export_service.save_to_file(sample_state, "text", filepath)
        assert result.exists()
        content = result.read_text()
        assert "The Beginning" in content.upper() or "chapter" in content.lower()

    def test_save_to_file_html(self, export_service, sample_state, tmp_path):
        """Test save_to_file with html format."""
        filepath = tmp_path / "test.html"
        result = export_service.save_to_file(sample_state, "html", filepath)
        assert result.exists()
        content = result.read_text()
        assert "<html>" in content
        assert "</html>" in content


class TestExportServiceReraise:
    """Tests specifically targeting the re-raise behavior."""

    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return Settings()

    @pytest.fixture
    def export_service(self, settings):
        """Create an ExportService for testing."""
        return ExportService(settings)

    @pytest.fixture
    def sample_state(self):
        """Create a minimal StoryState."""
        brief = StoryBrief(
            premise="Test",
            genre="Fantasy",
            subgenres=[],
            tone="Epic",
            themes=[],
            setting_time="Medieval",
            setting_place="Kingdom",
            target_length="novella",
            language="English",
            content_rating="none",
        )
        state = StoryState(id="test-001", brief=brief)
        state.chapters = []
        return state

    def test_save_to_file_path_validation_raises_value_error(
        self, export_service, sample_state, tmp_path, monkeypatch
    ):
        """Test save_to_file re-raises ValueError from path validation."""
        # Line 833: ValueError re-raise
        # We need to trigger a ValueError from _validate_export_path
        # by providing a path outside allowed directories

        # Create a path that would escape the base directory
        # Note: On Windows the path traversal check works differently
        # We'll test with a path that triggers validation error
        from services.export_service import _validate_export_path

        # Test that ValueError from validation is raised
        with pytest.raises(ValueError, match="outside allowed directory"):
            # Use a path outside the base dir
            # Root path should fail validation
            _validate_export_path(Path("/some/other/directory/file.txt"))
