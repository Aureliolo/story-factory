"""Attribute validation and normalization utilities for WorldDatabase.

Handles depth-checking, size validation, and flattening of deeply nested
attribute dictionaries before they are stored in SQLite.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Attributes constraints
# Lifecycle dicts (entity -> lifecycle -> birth -> {year, era_name}) need depth 4
MAX_ATTRIBUTES_DEPTH = 5
MAX_ATTRIBUTES_SIZE_BYTES = 10 * 1024  # 10KB


def flatten_deep_attributes(
    obj: Any,
    max_depth: int = MAX_ATTRIBUTES_DEPTH,
    current_depth: int = 0,
    key_path: str = "",
) -> Any:
    """Flatten attributes that exceed max nesting depth.

    Converts deeply nested structures to JSON string representations at the max depth
    to preserve data while meeting storage constraints.

    Args:
        obj: Object to potentially flatten.
        max_depth: Maximum nesting depth allowed.
        current_depth: Current depth in the recursion.
        key_path: Dot-separated path to the current position for diagnostic logging.

    Returns:
        Flattened object with nested structures converted to strings at max depth.
    """
    if current_depth >= max_depth:
        # At max depth, convert complex types to string representation
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Use deterministic JSON representation for nested structures
        logger.debug(
            "Flattening nested %s at depth %d (path: %s)",
            type(obj).__name__,
            current_depth,
            key_path or "<root>",
        )
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to JSON-serialize %s at max depth; falling back to str(): %s",
                type(obj).__name__,
                exc,
            )
            return str(obj)

    if isinstance(obj, dict):
        return {
            k: flatten_deep_attributes(
                v, max_depth, current_depth + 1, f"{key_path}.{k}" if key_path else k
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            flatten_deep_attributes(item, max_depth, current_depth + 1, f"{key_path}[{i}]")
            for i, item in enumerate(obj)
        ]
    return obj


def check_nesting_depth(obj: Any, max_depth: int, current_depth: int = 0) -> bool:
    """Check if object exceeds maximum nesting depth.

    Args:
        obj: Object to check.
        max_depth: Maximum allowed nesting depth.
        current_depth: Current depth in the recursion.

    Returns:
        True if object exceeds max depth, False otherwise.
    """
    if current_depth > max_depth:
        return True
    if isinstance(obj, dict):
        return any(check_nesting_depth(v, max_depth, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        return any(check_nesting_depth(item, max_depth, current_depth + 1) for item in obj)
    return False


def validate_and_normalize_attributes(
    attrs: dict[str, Any], max_depth: int = MAX_ATTRIBUTES_DEPTH
) -> dict[str, Any]:
    """Validate and normalize attributes dict.

    Flattens deeply nested structures and validates size constraints.

    Args:
        attrs: Attributes dictionary to validate.
        max_depth: Maximum nesting depth allowed.

    Returns:
        Normalized attributes dict with deep nesting flattened.

    Raises:
        ValueError: If attributes exceed size limit.
    """
    # Check if flattening is needed and flatten
    if check_nesting_depth(attrs, max_depth, current_depth=1):
        deep_keys = [
            k for k, v in attrs.items() if check_nesting_depth(v, max_depth, current_depth=2)
        ]
        logger.warning(
            "Attributes exceed maximum nesting depth of %d, flattening deep structures "
            "(affected keys: %s)",
            max_depth,
            deep_keys or ["<root>"],
        )
        attrs = flatten_deep_attributes(attrs, max_depth, current_depth=1)

    # Check size (after flattening)
    attrs_json = json.dumps(attrs)
    if len(attrs_json.encode("utf-8")) > MAX_ATTRIBUTES_SIZE_BYTES:
        raise ValueError(f"Attributes exceed maximum size of {MAX_ATTRIBUTES_SIZE_BYTES // 1024}KB")

    return attrs
