"""Unit tests for prompts package."""

from pathlib import Path


class TestPromptsPackage:
    """Tests for prompts package initialization."""

    def test_templates_dir_exists(self):
        """Test that TEMPLATES_DIR is exported and points to correct path."""
        from src.prompts import TEMPLATES_DIR

        assert isinstance(TEMPLATES_DIR, Path)
        assert TEMPLATES_DIR.name == "templates"
        assert TEMPLATES_DIR.parent.name == "prompts"

    def test_templates_dir_is_real_directory(self):
        """Test that TEMPLATES_DIR points to an actual directory."""
        from src.prompts import TEMPLATES_DIR

        assert TEMPLATES_DIR.exists()
        assert TEMPLATES_DIR.is_dir()
