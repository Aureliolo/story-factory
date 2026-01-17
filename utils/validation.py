"""Input validation utilities for services and agents.

This module provides reusable validation functions that raise clear
ValueError or TypeError exceptions for invalid inputs.
"""


def validate_not_none(value, param_name: str):
    """Validate that a required parameter is not None.

    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")


def validate_not_empty(value: str | None, param_name: str) -> None:
    """Validate that a string parameter is not None or empty.

    Args:
        value: The string value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is None, empty string, or only whitespace
    """
    if value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")
    if not isinstance(value, str):
        raise TypeError(f"Parameter '{param_name}' must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"Parameter '{param_name}' cannot be empty")


def validate_positive(value: int | float | None, param_name: str) -> None:
    """Validate that a numeric parameter is positive.

    Args:
        value: The numeric value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is None, not a number, or not positive
        TypeError: If value is not int or float
    """
    if value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")
    if not isinstance(value, (int, float)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Parameter '{param_name}' must be positive, got {value}")


def validate_non_negative(value: int | float | None, param_name: str) -> None:
    """Validate that a numeric parameter is non-negative (>= 0).

    Args:
        value: The numeric value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is None, not a number, or negative
        TypeError: If value is not int or float
    """
    if value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")
    if not isinstance(value, (int, float)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"Parameter '{param_name}' must be non-negative, got {value}")


def validate_in_range(
    value: int | float | None,
    param_name: str,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
) -> None:
    """Validate that a numeric parameter is within a specified range.

    Args:
        value: The numeric value to validate
        param_name: Name of the parameter for error messages
        min_val: Minimum allowed value (inclusive), None for no minimum
        max_val: Maximum allowed value (inclusive), None for no maximum

    Raises:
        ValueError: If value is None, not in range, or not a number
        TypeError: If value is not int or float
    """
    if value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")
    if not isinstance(value, (int, float)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value).__name__}")

    if min_val is not None and value < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {value}")


def validate_not_empty_collection(value: list | dict | None, param_name: str) -> None:
    """Validate that a collection parameter is not None or empty.

    Args:
        value: The collection to validate (list or dict)
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is None or empty
        TypeError: If value is not a list or dict
    """
    if value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")
    if not isinstance(value, (list, dict)):
        raise TypeError(
            f"Parameter '{param_name}' must be a list or dict, got {type(value).__name__}"
        )
    if len(value) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")


def validate_type(value, param_name: str, expected_type: type) -> None:
    """Validate that a parameter is of the expected type.

    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages
        expected_type: The expected type

    Raises:
        TypeError: If value is not of expected_type
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Parameter '{param_name}' must be {expected_type.__name__}, got {type(value).__name__}"
        )


def validate_string_in_choices(value: str | None, param_name: str, choices: list[str]) -> None:
    """Validate that a string parameter is one of the allowed choices.

    Args:
        value: The string value to validate
        param_name: Name of the parameter for error messages
        choices: List of allowed values

    Raises:
        ValueError: If value is None, empty, or not in choices
    """
    validate_not_empty(value, param_name)
    if value not in choices:
        raise ValueError(f"Parameter '{param_name}' must be one of {choices}, got '{value}'")
