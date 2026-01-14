"""Tests for input validation utilities."""

import pytest
from pathlib import Path
import tempfile

from utils.validators import (
    ValidationError,
    validate_story_id,
    validate_file_path,
    validate_model_name,
    validate_temperature,
    validate_chapter_number,
    sanitize_filename,
)


class TestValidateStoryId:
    """Tests for story ID validation."""

    def test_valid_uuid(self):
        """Should accept valid UUID."""
        story_id = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_story_id(story_id) is True

    def test_rejects_empty(self):
        """Should reject empty string."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_story_id("")

    def test_rejects_path_traversal(self):
        """Should reject path traversal attempts."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_story_id("../etc/passwd")
        
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_story_id("story/../config")

    def test_rejects_invalid_format(self):
        """Should reject non-UUID format."""
        with pytest.raises(ValidationError, match="valid UUID format"):
            validate_story_id("not-a-uuid")
        
        with pytest.raises(ValidationError, match="valid UUID format"):
            validate_story_id("12345")


class TestValidateFilePath:
    """Tests for file path validation."""

    def test_valid_path(self):
        """Should accept valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            result = validate_file_path(str(path))
            assert isinstance(result, Path)

    def test_rejects_empty(self):
        """Should reject empty path."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_file_path("")

    def test_constrains_to_base_dir(self):
        """Should constrain path to base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            valid_path = base / "story.json"
            
            # Valid path within base
            result = validate_file_path(str(valid_path), base)
            assert result == valid_path.resolve()
            
            # Invalid path outside base
            with pytest.raises(ValidationError, match="outside allowed directory"):
                validate_file_path("/etc/passwd", base)

    def test_prevents_path_traversal(self):
        """Should prevent path traversal attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            # Attempt to escape base directory
            with pytest.raises(ValidationError, match="outside allowed directory"):
                validate_file_path(str(base / "../etc/passwd"), base)


class TestValidateModelName:
    """Tests for model name validation."""

    def test_valid_model_names(self):
        """Should accept valid model names."""
        assert validate_model_name("username/model") is True
        assert validate_model_name("username/model:tag") is True
        assert validate_model_name("user_name/model-name:1.0") is True
        assert validate_model_name("auto") is True

    def test_rejects_empty(self):
        """Should reject empty model name."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_model_name("")

    def test_rejects_invalid_format(self):
        """Should reject invalid format."""
        with pytest.raises(ValidationError, match="invalid"):
            validate_model_name("just-a-name")
        
        with pytest.raises(ValidationError, match="invalid"):
            validate_model_name("model@version")


class TestValidateTemperature:
    """Tests for temperature validation."""

    def test_valid_temperatures(self):
        """Should accept valid temperatures."""
        assert validate_temperature(0.0) is True
        assert validate_temperature(0.7) is True
        assert validate_temperature(1.0) is True
        assert validate_temperature(2.0) is True

    def test_rejects_out_of_range(self):
        """Should reject out of range values."""
        with pytest.raises(ValidationError, match="between 0.0 and 2.0"):
            validate_temperature(-0.1)
        
        with pytest.raises(ValidationError, match="between 0.0 and 2.0"):
            validate_temperature(2.1)

    def test_rejects_non_numeric(self):
        """Should reject non-numeric values."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_temperature("0.7")


class TestValidateChapterNumber:
    """Tests for chapter number validation."""

    def test_valid_chapter_numbers(self):
        """Should accept valid chapter numbers."""
        assert validate_chapter_number(1) is True
        assert validate_chapter_number(10) is True
        assert validate_chapter_number(5, max_chapters=10) is True

    def test_rejects_non_positive(self):
        """Should reject non-positive numbers."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_chapter_number(0)
        
        with pytest.raises(ValidationError, match="must be positive"):
            validate_chapter_number(-1)

    def test_rejects_exceeds_max(self):
        """Should reject numbers exceeding max."""
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_chapter_number(11, max_chapters=10)

    def test_rejects_non_integer(self):
        """Should reject non-integer values."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_chapter_number(1.5)


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_preserves_safe_names(self):
        """Should preserve safe filenames."""
        assert sanitize_filename("my_story.json") == "my_story.json"
        assert sanitize_filename("story-2024.md") == "story-2024.md"

    def test_removes_unsafe_characters(self):
        """Should remove unsafe characters."""
        assert sanitize_filename("my/story") == "mystory"
        assert sanitize_filename("story<>file") == "storyfile"
        assert sanitize_filename("file|name") == "filename"

    def test_handles_empty(self):
        """Should handle empty input."""
        assert sanitize_filename("") == "untitled"
        assert sanitize_filename("   ") == "untitled"

    def test_normalizes_whitespace(self):
        """Should normalize whitespace."""
        assert sanitize_filename("my   story") == "my_story"
        assert sanitize_filename("story  file") == "story_file"

    def test_truncates_long_names(self):
        """Should truncate very long names."""
        long_name = "a" * 250
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_handles_special_cases(self):
        """Should handle special edge cases."""
        assert sanitize_filename("...") == "untitled"
        assert sanitize_filename("///") == "untitled"
