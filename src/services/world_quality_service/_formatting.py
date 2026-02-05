"""Formatting utilities for WorldQualityService.

Contains helper functions for formatting properties, ETA calculations,
and prompt generation for entity generation.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def calculate_eta(
    completed_times: list[float],
    remaining_count: int,
) -> float | None:
    """Calculate ETA using exponential moving average.

    Uses EMA with alpha=0.3 to weight recent entity generation times more heavily,
    providing more accurate estimates as we learn the actual generation speed.

    Args:
        completed_times: List of completion times for previous entities (in seconds).
        remaining_count: Number of entities remaining to generate.

    Returns:
        Estimated remaining time in seconds, or None if no data available.
    """
    if not completed_times or remaining_count <= 0:
        logger.debug(
            "calculate_eta: no estimate (completed_times=%d, remaining=%d)",
            len(completed_times),
            remaining_count,
        )
        return None
    # EMA with alpha=0.3 to weight recent times more heavily
    alpha = 0.3
    avg = completed_times[0]
    for t in completed_times[1:]:
        avg = alpha * t + (1 - alpha) * avg
    eta = avg * remaining_count
    logger.debug(
        "calculate_eta: avg=%.2fs, remaining=%d, eta=%.2fs",
        avg,
        remaining_count,
        eta,
    )
    return eta


def format_properties(properties: list[Any] | Any | None) -> str:
    """Format a list of properties into a comma-separated string.

    Handles both string and dict properties (LLM sometimes returns dicts).
    Also handles None or non-list inputs gracefully.

    Args:
        properties: List of properties (strings or dicts), or None/single value.

    Returns:
        Comma-separated string of property names, or empty string if no properties.
    """
    if not properties:
        logger.debug(f"format_properties: early return on falsy input: {properties!r}")
        return ""
    if not isinstance(properties, list):
        logger.debug(
            f"format_properties: coercing non-list input to list: {type(properties).__name__}"
        )
        properties = [properties]

    result: list[str] = []
    for prop in properties:
        if isinstance(prop, str):
            result.append(prop)
        elif isinstance(prop, dict):
            # Try to extract a name or description from dict
            # Use key existence check to handle empty strings correctly
            # Coerce to string to handle non-string values (e.g., None, int)
            if "name" in prop:
                value = prop["name"]
                if value is not None:
                    result.append(str(value))
            elif "description" in prop:
                value = prop["description"]
                if value is not None:
                    result.append(str(value))
            else:
                result.append(str(prop))
        else:
            result.append(str(prop))
    # Filter empty strings to avoid ", , " in output
    return ", ".join(s for s in result if s)


def format_existing_names_warning(existing_names: list[str], entity_type: str) -> str:
    """Format existing names with explicit DO NOT examples for consistent duplicate prevention.

    Parameters:
        existing_names: Existing entity names to avoid.
        entity_type: Type of entity (concept, item, location, faction) for context.

    Returns:
        Formatted warning string with examples of what NOT to use.
    """
    if not existing_names:
        logger.debug("Formatting existing %s names: none provided", entity_type)
        return (
            f"EXISTING {entity_type.upper()}S: None yet - you are creating the first {entity_type}."
        )

    formatted_names = "\n".join(f"  - {name}" for name in existing_names)

    # Generate example DO NOT variations from the first name
    # (existing_names is guaranteed non-empty here due to early return above)
    example_name = existing_names[0]
    do_not_examples = [
        f'"{example_name}" (exact match)',
        f'"{example_name.upper()}" (case variation)',
        f'"The {example_name}" (prefix variation)',
    ]

    logger.debug("Formatted %d existing %s names for prompt", len(existing_names), entity_type)

    return f"""<existing-{entity_type}s>
EXISTING {entity_type.upper()}S (DO NOT DUPLICATE OR CREATE SIMILAR NAMES):
{formatted_names}

DO NOT USE names like:
{chr(10).join(f"  - {ex}" for ex in do_not_examples)}

Create a COMPLETELY DIFFERENT {entity_type} name.
</existing-{entity_type}s>"""
