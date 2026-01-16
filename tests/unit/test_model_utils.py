"""Tests for model utility functions."""

import pytest

from utils.model_utils import extract_model_name


class TestExtractModelName:
    """Tests for extract_model_name function."""

    def test_simple_model_name(self):
        """Test extraction of simple model name without path."""
        assert extract_model_name("llama3.3") == "llama3.3"

    def test_huggingface_path(self):
        """Test extraction from HuggingFace-style path."""
        assert extract_model_name("qwen/qwen3-30b") == "qwen3-30b"

    def test_nested_path(self):
        """Test extraction from nested path."""
        assert extract_model_name("org/team/model-name") == "model-name"

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            extract_model_name("")

    def test_none_raises(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            extract_model_name(None)  # type: ignore[arg-type]

    def test_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            extract_model_name("   ")

    def test_model_with_special_characters(self):
        """Test extraction with special characters in name."""
        assert extract_model_name("org/model-v1.2_beta") == "model-v1.2_beta"

    def test_trailing_slash_raises(self):
        """Test that model_id ending with '/' raises ValueError."""
        with pytest.raises(ValueError, match="cannot end with '/'"):
            extract_model_name("org/")
