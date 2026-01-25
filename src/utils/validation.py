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
    """
    Ensure the string parameter is one of the allowed choices.

    Parameters:
        value (str | None): The string to validate.
        param_name (str): Parameter name used in error messages.
        choices (list[str]): Allowed string values.

    Raises:
        ValueError: If `value` is None, empty or only whitespace, or not one of `choices`.
    """
    validate_not_empty(value, param_name)
    if value not in choices:
        raise ValueError(f"Parameter '{param_name}' must be one of {choices}, got '{value}'")


# Common prefixes to strip when comparing names for duplicates
_NAME_PREFIXES = ("the ", "a ", "an ")


def _normalize_name(name: str) -> str:
    """
    Normalize a name for comparison by removing a leading common article, converting to lowercase, and trimming surrounding whitespace.

    Parameters:
        name (str): The input name to normalize.

    Returns:
        str: The name converted to lowercase, trimmed of surrounding whitespace, and with a leading common prefix ("the ", "a ", "an ") removed if present.
    """
    normalized = name.lower().strip()
    for prefix in _NAME_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    return normalized


def validate_unique_name(
    name: str,
    existing_names: list[str],
    check_substring: bool = True,
    min_substring_length: int = 4,
) -> tuple[bool, str | None, str | None]:
    """
    Determine whether a candidate name conflicts with any names in an existing list.

    Performs these checks (in order): exact match (case-sensitive), case-insensitive match,
    match after removing common leading articles (e.g., "the", "a", "an"), and optionally
    substring containment when both normalized names meet the minimum length.
    Empty or whitespace-only candidate names are treated as unique; empty existing entries are ignored.

    Parameters:
        name: Candidate name to validate.
        existing_names: Sequence of existing names to check against.
        check_substring: If true, check for substring containment in both directions.
        min_substring_length: Minimum normalized length required to perform substring checks.

    Returns:
        A tuple (is_unique, conflicting_name, reason):
        - is_unique: `True` if no conflict was found, `False` otherwise.
        - conflicting_name: The existing name that conflicts with `name`, or `None` if unique.
        - reason: One of `"exact"`, `"case_insensitive"`, `"prefix_match"`, or `"substring"`, or `None` if unique.
    """
    if not name or not name.strip():
        return True, None, None  # Empty names are handled elsewhere

    normalized_new = _normalize_name(name)

    for existing in existing_names:
        if not existing or not existing.strip():
            continue

        normalized_existing = _normalize_name(existing)

        # Check exact match (case-insensitive)
        if name.lower().strip() == existing.lower().strip():
            if name.strip() == existing.strip():
                return False, existing, "exact"
            return False, existing, "case_insensitive"

        # Check prefix-stripped match
        if normalized_new == normalized_existing:
            return False, existing, "prefix_match"

        # Check substring containment (both directions)
        if check_substring:
            # Only check if both names are long enough to avoid false positives
            if len(normalized_new) >= min_substring_length:
                if len(normalized_existing) >= min_substring_length:
                    if normalized_new in normalized_existing:
                        return False, existing, "substring"
                    if normalized_existing in normalized_new:
                        return False, existing, "substring"

    return True, None, None
