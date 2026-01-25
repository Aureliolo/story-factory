"""Unit tests for custom exceptions in exceptions.py."""

from src.utils.exceptions import DuplicateNameError, EmptyGenerationError


class TestEmptyGenerationError:
    """Tests for EmptyGenerationError exception."""

    def test_empty_generation_error_message(self) -> None:
        """
        Verify that EmptyGenerationError exposes the provided message as its string representation.
        """
        error = EmptyGenerationError("Generation returned empty content")
        assert str(error) == "Generation returned empty content"


class TestDuplicateNameError:
    """Tests for DuplicateNameError exception."""

    def test_duplicate_name_error_with_all_attributes(self) -> None:
        """Test DuplicateNameError stores all attributes."""
        error = DuplicateNameError(
            message="Duplicate faction name detected",
            generated_name="The Guild",
            existing_name="Guild",
            reason="prefix_match",
        )
        assert str(error) == "Duplicate faction name detected"
        assert error.generated_name == "The Guild"
        assert error.existing_name == "Guild"
        assert error.reason == "prefix_match"

    def test_duplicate_name_error_with_defaults(self) -> None:
        """Test DuplicateNameError with default None attributes."""
        error = DuplicateNameError("Duplicate detected")
        assert str(error) == "Duplicate detected"
        assert error.generated_name is None
        assert error.existing_name is None
        assert error.reason is None

    def test_duplicate_name_error_partial_attributes(self) -> None:
        """Test DuplicateNameError with partial attributes."""
        error = DuplicateNameError(
            message="Name conflict",
            generated_name="Shadow Council",
            reason="substring",
        )
        assert error.generated_name == "Shadow Council"
        assert error.existing_name is None
        assert error.reason == "substring"
