"""Tests for input validation utilities."""

import pytest

from src.utils.validation import (
    validate_in_range,
    validate_non_negative,
    validate_not_empty,
    validate_not_empty_collection,
    validate_not_none,
    validate_positive,
    validate_string_in_choices,
    validate_type,
)


class TestValidateNotNone:
    """Tests for validate_not_none function."""

    def test_valid_value(self):
        """Test that valid non-None values pass."""
        validate_not_none("value", "param")
        validate_not_none(123, "param")
        validate_not_none([], "param")
        validate_not_none({}, "param")

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_not_none(None, "test_param")


class TestValidateNotEmpty:
    """Tests for validate_not_empty function."""

    def test_valid_string(self):
        """Test that non-empty strings pass."""
        validate_not_empty("hello", "param")
        validate_not_empty("  hello  ", "param")

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_not_empty(None, "test_param")

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be empty"):
            validate_not_empty("", "test_param")

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be empty"):
            validate_not_empty("   ", "test_param")

    def test_non_string_raises_type_error(self):
        """Test that non-string values raise TypeError."""
        with pytest.raises(TypeError, match="Parameter 'test_param' must be a string"):
            validate_not_empty(123, "test_param")  # type: ignore[arg-type]


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_valid_positive_int(self):
        """Test that positive integers pass."""
        validate_positive(1, "param")
        validate_positive(100, "param")

    def test_valid_positive_float(self):
        """Test that positive floats pass."""
        validate_positive(0.1, "param")
        validate_positive(99.9, "param")

    def test_zero_raises_error(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' must be positive"):
            validate_positive(0, "test_param")

    def test_negative_raises_error(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' must be positive"):
            validate_positive(-1, "test_param")

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_positive(None, "test_param")

    def test_non_numeric_raises_type_error(self):
        """Test that non-numeric values raise TypeError."""
        with pytest.raises(TypeError, match="Parameter 'test_param' must be numeric"):
            validate_positive("5", "test_param")  # type: ignore[arg-type]


class TestValidateNonNegative:
    """Tests for validate_non_negative function."""

    def test_valid_positive(self):
        """Test that positive values pass."""
        validate_non_negative(1, "param")
        validate_non_negative(0.5, "param")

    def test_zero_passes(self):
        """Test that zero passes."""
        validate_non_negative(0, "param")
        validate_non_negative(0.0, "param")

    def test_negative_raises_error(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' must be non-negative"):
            validate_non_negative(-1, "test_param")

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_non_negative(None, "test_param")

    def test_non_numeric_raises_type_error(self):
        """Test that non-numeric values raise TypeError."""
        with pytest.raises(TypeError, match="Parameter 'test_param' must be numeric"):
            validate_non_negative("5", "test_param")  # type: ignore[arg-type]


class TestValidateInRange:
    """Tests for validate_in_range function."""

    def test_value_in_range(self):
        """Test that values within range pass."""
        validate_in_range(5, "param", min_val=0, max_val=10)
        validate_in_range(0, "param", min_val=0, max_val=10)
        validate_in_range(10, "param", min_val=0, max_val=10)

    def test_value_only_min(self):
        """Test validation with only minimum."""
        validate_in_range(100, "param", min_val=0)
        validate_in_range(0, "param", min_val=0)

    def test_value_only_max(self):
        """Test validation with only maximum."""
        validate_in_range(5, "param", max_val=10)
        validate_in_range(-100, "param", max_val=10)

    def test_below_min_raises_error(self):
        """Test that values below minimum raise ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' must be >= 0"):
            validate_in_range(-1, "test_param", min_val=0)

    def test_above_max_raises_error(self):
        """Test that values above maximum raise ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' must be <= 10"):
            validate_in_range(11, "test_param", max_val=10)

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_in_range(None, "test_param", min_val=0, max_val=10)

    def test_non_numeric_raises_type_error(self):
        """Test that non-numeric values raise TypeError."""
        with pytest.raises(TypeError, match="Parameter 'test_param' must be numeric"):
            validate_in_range("5", "test_param", min_val=0, max_val=10)  # type: ignore[arg-type]


class TestValidateNotEmptyCollection:
    """Tests for validate_not_empty_collection function."""

    def test_valid_list(self):
        """Test that non-empty lists pass."""
        validate_not_empty_collection([1, 2, 3], "param")
        validate_not_empty_collection(["a"], "param")

    def test_valid_dict(self):
        """Test that non-empty dicts pass."""
        validate_not_empty_collection({"a": 1}, "param")
        validate_not_empty_collection({"x": "y", "z": "w"}, "param")

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be empty"):
            validate_not_empty_collection([], "test_param")

    def test_empty_dict_raises_error(self):
        """Test that empty dict raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be empty"):
            validate_not_empty_collection({}, "test_param")

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_not_empty_collection(None, "test_param")

    def test_non_collection_raises_type_error(self):
        """Test that non-collection values raise TypeError."""
        with pytest.raises(TypeError, match="Parameter 'test_param' must be a list or dict"):
            validate_not_empty_collection("string", "test_param")  # type: ignore[arg-type]


class TestValidateType:
    """Tests for validate_type function."""

    def test_correct_type(self):
        """Test that correct types pass."""
        validate_type("hello", "param", str)
        validate_type(123, "param", int)
        validate_type([1, 2], "param", list)

    def test_wrong_type_raises_error(self):
        """Test that wrong type raises TypeError."""
        with pytest.raises(TypeError, match="Parameter 'test_param' must be str, got int"):
            validate_type(123, "test_param", str)


class TestValidateStringInChoices:
    """Tests for validate_string_in_choices function."""

    def test_valid_choice(self):
        """Test that valid choices pass."""
        validate_string_in_choices("red", "color", ["red", "green", "blue"])
        validate_string_in_choices("green", "color", ["red", "green", "blue"])

    def test_invalid_choice_raises_error(self):
        """Test that invalid choice raises ValueError."""
        with pytest.raises(
            ValueError, match="Parameter 'test_param' must be one of \\['a', 'b'\\]"
        ):
            validate_string_in_choices("c", "test_param", ["a", "b"])

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be None"):
            validate_string_in_choices(None, "test_param", ["a", "b"])

    def test_empty_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'test_param' cannot be empty"):
            validate_string_in_choices("", "test_param", ["a", "b"])
