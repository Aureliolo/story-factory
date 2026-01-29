"""Unit tests for custom exceptions in exceptions.py."""

from src.utils.exceptions import (
    CircuitOpenError,
    DuplicateNameError,
    EmptyGenerationError,
    TemporalValidationError,
)


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


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_circuit_open_error_with_retry_time(self) -> None:
        """Test CircuitOpenError stores time_until_retry."""
        error = CircuitOpenError(
            message="Circuit breaker is open",
            time_until_retry=30.5,
        )
        assert str(error) == "Circuit breaker is open"
        assert error.time_until_retry == 30.5

    def test_circuit_open_error_without_retry_time(self) -> None:
        """Test CircuitOpenError with default None time_until_retry."""
        error = CircuitOpenError("Circuit is open")
        assert str(error) == "Circuit is open"
        assert error.time_until_retry is None


class TestTemporalValidationError:
    """Tests for TemporalValidationError exception."""

    def test_temporal_validation_error_with_all_attributes(self) -> None:
        """Test TemporalValidationError stores all attributes."""
        error = TemporalValidationError(
            message="Character born before faction founded",
            entity_id="char-1",
            entity_name="Young Knight",
            error_type="predates_dependency",
            related_entity_id="faction-1",
        )
        assert str(error) == "Character born before faction founded"
        assert error.entity_id == "char-1"
        assert error.entity_name == "Young Knight"
        assert error.error_type == "predates_dependency"
        assert error.related_entity_id == "faction-1"

    def test_temporal_validation_error_with_defaults(self) -> None:
        """Test TemporalValidationError with default None attributes."""
        error = TemporalValidationError("Temporal inconsistency detected")
        assert str(error) == "Temporal inconsistency detected"
        assert error.entity_id is None
        assert error.entity_name is None
        assert error.error_type is None
        assert error.related_entity_id is None

    def test_temporal_validation_error_partial_attributes(self) -> None:
        """Test TemporalValidationError with partial attributes."""
        error = TemporalValidationError(
            message="Invalid date",
            entity_id="char-2",
            error_type="invalid_date",
        )
        assert error.entity_id == "char-2"
        assert error.entity_name is None
        assert error.error_type == "invalid_date"
        assert error.related_entity_id is None
