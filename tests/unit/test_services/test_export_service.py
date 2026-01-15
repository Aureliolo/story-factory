"""Unit tests for export service."""

import tempfile
from pathlib import Path

import pytest

from memory.story_state import Chapter, StoryBrief, StoryState
from services.export_service import ExportService, _validate_export_path
from settings import STORIES_DIR, Settings


class TestExportServiceMarkdown:
    """Test markdown export functionality."""

    def test_export_to_markdown_basic(self, tmp_path):
        """Test basic markdown export."""
        settings = Settings()
        service = ExportService(settings)

        # Create test story state with required fields
        brief = StoryBrief(
            premise="Test story premise",
            genre="Fantasy",
            tone="Epic",
            setting_time="Modern",
            setting_place="Earth",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-1", project_name="Test Story", brief=brief)
        state.chapters = [
            Chapter(
                number=1, title="Chapter One", outline="Outline", content="Chapter 1 content here."
            )
        ]

        output_file = tmp_path / "test_story.md"
        markdown = service.to_markdown(state)
        output_file.write_text(markdown)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Test Story" in content
        assert "Chapter 1 content here." in content

    def test_export_to_markdown_empty_story(self, tmp_path):
        """Test markdown export with empty story content."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="Empty story",
            genre="Drama",
            tone="Somber",
            setting_time="Present",
            setting_place="City",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-2", project_name="Empty Story", brief=brief)

        output_file = tmp_path / "empty.md"
        markdown = service.to_markdown(state)
        output_file.write_text(markdown)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Empty Story" in content

    def test_export_to_markdown_multiple_chapters(self, tmp_path):
        """Test markdown export with multiple chapters."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="Multi-chapter story",
            genre="Adventure",
            tone="Lighthearted",
            setting_time="Future",
            setting_place="Space",
            target_length="novella",
            content_rating="none",
        )
        state = StoryState(id="test-3", project_name="Multi-Chapter Story", brief=brief)
        state.chapters = [
            Chapter(number=1, title="Chapter One", outline="First", content="First chapter."),
            Chapter(number=2, title="Chapter Two", outline="Second", content="Second chapter."),
            Chapter(number=3, title="Chapter Three", outline="Third", content="Third chapter."),
        ]

        output_file = tmp_path / "multi.md"
        markdown = service.to_markdown(state)
        output_file.write_text(markdown)

        assert output_file.exists()
        content = output_file.read_text()
        assert "First chapter." in content
        assert "Second chapter." in content
        assert "Third chapter." in content


class TestExportServiceText:
    """Test text export functionality."""

    def test_export_to_text(self, tmp_path):
        """Test basic text export."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="Text story",
            genre="Mystery",
            tone="Suspenseful",
            setting_time="1920s",
            setting_place="Chicago",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-4", project_name="Text Story", brief=brief)
        state.chapters = [
            Chapter(number=1, title="The Case", outline="Intro", content="Content in text format.")
        ]

        output_file = tmp_path / "story.txt"
        text = service.to_text(state)
        output_file.write_text(text)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Text Story" in content or "TEXT STORY" in content
        assert "Content in text format." in content

    def test_export_to_text_preserves_formatting(self, tmp_path):
        """Test text export preserves line breaks."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="Formatted text",
            genre="Literary",
            tone="Reflective",
            setting_time="Present",
            setting_place="Anywhere",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-5", project_name="Formatted Text", brief=brief)
        state.chapters = [
            Chapter(number=1, title="Lines", outline="Test", content="Line 1\n\nLine 2\n\nLine 3")
        ]

        output_file = tmp_path / "formatted.txt"
        text = service.to_text(state)
        output_file.write_text(text)

        content = output_file.read_text()
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content


class TestExportServicePDF:
    """Test PDF export functionality."""

    def test_export_to_pdf(self):
        """Test PDF export."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="PDF story",
            genre="Thriller",
            tone="Dark",
            setting_time="Modern",
            setting_place="City",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-6", project_name="PDF Story", brief=brief)
        state.chapters = [
            Chapter(number=1, title="Chapter", outline="Test", content="PDF content.")
        ]

        # Test that to_pdf method exists and returns bytes
        pdf_bytes = service.to_pdf(state)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0


class TestExportServiceEPUB:
    """Test EPUB export functionality."""

    def test_export_to_epub(self):
        """Test EPUB export."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="EPUB story",
            genre="Science Fiction",
            tone="Futuristic",
            setting_time="2250",
            setting_place="Mars Colony",
            target_length="novella",
            content_rating="none",
        )
        state = StoryState(id="test-7", project_name="EPUB Story", brief=brief)
        state.chapters = [
            Chapter(number=1, title="First Contact", outline="Intro", content="EPUB content.")
        ]

        # Test that to_epub method exists and returns bytes
        epub_bytes = service.to_epub(state)
        assert isinstance(epub_bytes, bytes)
        assert len(epub_bytes) > 0


class TestExportServiceMethods:
    """Test export service has required methods."""

    def test_service_has_export_methods(self):
        """Test that export service has all required methods."""
        settings = Settings()
        service = ExportService(settings)

        assert hasattr(service, "to_markdown")
        assert hasattr(service, "to_text")
        assert hasattr(service, "to_pdf")
        assert hasattr(service, "to_epub")
        assert callable(service.to_markdown)
        assert callable(service.to_text)
        assert callable(service.to_pdf)
        assert callable(service.to_epub)


class TestExportServiceEdgeCases:
    """Test edge cases in export service."""

    def test_export_with_no_chapters(self):
        """Test export with no chapters doesn't crash."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="No chapters",
            genre="Drama",
            tone="Calm",
            setting_time="Present",
            setting_place="Home",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-8", project_name="No Chapters", brief=brief)
        # No chapters added

        markdown = service.to_markdown(state)
        assert "No Chapters" in markdown or len(markdown) > 0

    def test_export_with_special_characters(self):
        """Test export with special characters in content."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="Unicode test",
            genre="Fantasy",
            tone="Whimsical",
            setting_time="Timeless",
            setting_place="Everywhere",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-9", project_name="Story with émojis 🎉", brief=brief)
        state.chapters = [
            Chapter(
                number=1,
                title="Unicode Chapter",
                outline="Test",
                content="Content with unicode: αβγ 中文 🎨",
            )
        ]

        markdown = service.to_markdown(state)
        # Should handle unicode without crashing
        assert len(markdown) > 0

    def test_export_to_docx(self, tmp_path):
        """Test DOCX export."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="DOCX test story",
            genre="Sci-Fi",
            tone="Tense",
            setting_time="Future",
            setting_place="Space Station",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-docx", project_name="DOCX Test", brief=brief)
        state.chapters = [
            Chapter(
                number=1,
                title="First Chapter",
                outline="Start",
                content="This is the first chapter.\n\nAnd a second paragraph.",
            )
        ]

        docx_bytes = service.to_docx(state)
        assert len(docx_bytes) > 0
        assert docx_bytes[:4] == b"PK\x03\x04"  # DOCX is a ZIP file

    def test_save_to_file_docx(self, tmp_path):
        """Test saving DOCX to file."""
        settings = Settings()
        service = ExportService(settings)

        brief = StoryBrief(
            premise="File test",
            genre="Mystery",
            tone="Suspenseful",
            setting_time="1920s",
            setting_place="London",
            target_length="short_story",
            content_rating="none",
        )
        state = StoryState(id="test-file", project_name="File Test", brief=brief)
        state.chapters = [
            Chapter(number=1, title="Opening", outline="Start", content="Content here.")
        ]

        output_file = tmp_path / "story.docx"
        result_path = service.save_to_file(state, "docx", output_file)

        assert result_path.exists()
        assert result_path.suffix == ".docx"
        assert result_path.stat().st_size > 0


class TestValidateExportPath:
    """Tests for _validate_export_path function (path traversal prevention)."""

    def test_valid_path_within_base(self, tmp_path):
        """Test that valid paths within base directory are accepted."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        valid_path = base_dir / "stories" / "story.md"

        result = _validate_export_path(valid_path, base_dir)
        assert result == valid_path.resolve()

    def test_valid_path_in_temp_directory(self):
        """Test that paths in temp directory are accepted (for testing)."""
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / "test_export.md"

        result = _validate_export_path(temp_path, STORIES_DIR.parent)
        assert result == temp_path.resolve()

    def test_rejects_path_traversal_unix(self):
        """Test that Unix-style path traversal is rejected."""
        # Use STORIES_DIR.parent as base, then traverse outside it
        base_dir = STORIES_DIR.parent
        malicious_path = base_dir / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match="outside"):
            _validate_export_path(malicious_path, base_dir)

    def test_rejects_path_traversal_windows(self):
        """Test that Windows-style path traversal is rejected."""
        # Use STORIES_DIR.parent as base, then traverse outside it
        base_dir = STORIES_DIR.parent
        malicious_path = base_dir / ".." / ".." / ".." / "windows" / "system32" / "file"

        with pytest.raises(ValueError, match="outside"):
            _validate_export_path(malicious_path, base_dir)

    def test_rejects_absolute_path_outside_base(self):
        """Test that absolute paths outside base are rejected."""
        base_dir = STORIES_DIR.parent
        # Use a path that's clearly outside both base_dir and temp
        outside_path = Path("/some/other/path/file.txt")

        with pytest.raises(ValueError, match="outside"):
            _validate_export_path(outside_path, base_dir)

    def test_returns_resolved_path(self, tmp_path):
        """Test that the function returns resolved absolute paths."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        relative_path = base_dir / "." / "sub" / ".." / "story.md"

        result = _validate_export_path(relative_path, base_dir)
        assert result.is_absolute()
        assert ".." not in str(result)

    def test_uses_default_base_dir(self, tmp_path):
        """Test that function uses default base_dir when not specified."""
        # Create a path inside the actual output directory
        valid_path = STORIES_DIR / "test_story.md"

        # Should not raise if path is within default base_dir
        result = _validate_export_path(valid_path)
        assert result.is_absolute()
