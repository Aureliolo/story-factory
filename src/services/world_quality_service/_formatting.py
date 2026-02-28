"""Formatting utilities for WorldQualityService.

Contains helper functions for formatting properties, ETA calculations,
batch summary logging, and prompt generation for entity generation.
"""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.world_quality import BaseQualityScores

logger = logging.getLogger(__name__)


def calculate_eta(
    completed_times: list[float],
    remaining_count: int,
    initial_estimate_seconds: float | None = None,
) -> float | None:
    """
    Estimate remaining time for remaining entities using an exponential moving average of past completion times.

    Parameters:
        completed_times (list[float]): Sequence of past completion times in seconds (ordered from earliest to latest).
        remaining_count (int): Number of entities still to generate.
        initial_estimate_seconds (float | None): Fallback per-entity estimate when no
            completion data is available yet.  Used as a safety net for early ETA display.

    Returns:
        float | None: Estimated remaining time in seconds, or ``None`` if no estimate
            can be computed.
    """
    if remaining_count <= 0:
        logger.debug("calculate_eta: skipping (remaining=%d)", remaining_count)
        return None
    if not completed_times:
        if initial_estimate_seconds is not None:
            result = initial_estimate_seconds * remaining_count
            logger.debug(
                "calculate_eta: using initial estimate %.2fs * %d = %.2fs",
                initial_estimate_seconds,
                remaining_count,
                result,
            )
            return result
        logger.debug(
            "calculate_eta: skipping (no completed_times, no initial_estimate)",
        )
        return None
    # EMA with alpha=0.3 to weight recent times more heavily
    alpha = 0.3
    avg = completed_times[0]
    for t in completed_times[1:]:
        avg = alpha * t + (1 - alpha) * avg
    result = avg * remaining_count
    logger.debug("calculate_eta: avg=%.2fs, remaining=%d, eta=%.2fs", avg, remaining_count, result)
    return result


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
    """
    Constructs a prompt-friendly warning block that lists existing names for an entity type and provides explicit "DO NOT" example variations.

    Parameters:
        existing_names (list[str]): Existing names to include in the warning; if empty, the returned string indicates this is the first entity of the given type.
        entity_type (str): Noun describing the entity kind (e.g., "concept", "item", "location", "faction") used in headings and directive text.

    Returns:
        str: A multi-line string containing a structured prompt block with the existing names and example names to avoid, or a short message indicating no existing names.
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


# Common short words that are valid name endings (not truncation indicators)
_VALID_SHORT_ENDINGS = frozenset(
    {
        "the",
        "of",
        "in",
        "on",
        "at",
        "by",
        "to",
        "is",
        "it",
        "an",
        "as",
        "or",
        "no",
        "so",
        "do",
        "up",
        "if",
        "al",
        "el",
        "le",
        "la",
        "de",
        "du",
        "st",
        "ii",
        "iv",
        "vi",
    }
)


def check_name_completeness(name: str) -> bool:
    """Check whether a name appears truncated and log a warning if so.

    Detects names that end mid-word (last word < 3 chars and not a common word).

    Args:
        name: The entity name to check.

    Returns:
        True if the name appears complete, False if it looks truncated.
    """
    if not name or len(name) < 3:
        return True  # Too short to judge

    words = name.strip().split()
    if not words:
        return True

    last_word = words[-1].lower().rstrip(".,;:!?")
    if len(last_word) < 3 and last_word not in _VALID_SHORT_ENDINGS and len(words) > 1:
        logger.warning(
            "Entity name '%s' may be truncated (last word '%s' is < 3 chars)",
            name,
            last_word,
        )
        return False

    return True


def log_batch_summary(
    results: Sequence[tuple[Any, BaseQualityScores]],
    entity_type: str,
    quality_threshold: float,
    elapsed: float,
    get_name: Callable[[Any], str] | None = None,
) -> None:
    """Log an aggregate summary at the end of a batch generation or review.

    Args:
        results: List of (entity, scores) tuples produced by the batch.
        entity_type: Human-readable entity type (e.g., "character").
        quality_threshold: The configured quality threshold for pass/fail.
        elapsed: Total batch wall-clock time in seconds.
        get_name: Callable to extract display name from an entity. When provided,
            used for all entity types (chapters, relationships, etc.). Falls back
            to ``entity.get("name")`` / ``getattr(entity, "name")`` when ``None``.
    """
    if not results:
        logger.info(
            "Batch %s summary: 0 entities produced (%.1fs)",
            entity_type,
            elapsed,
        )
        return

    averages = [scores.average for _, scores in results]
    passed = sum(1 for avg in averages if round(avg, 1) >= quality_threshold)
    total = len(results)
    min_score = min(averages)
    max_score = max(averages)
    avg_score = sum(averages) / total

    failed_names: list[str] = []
    for entity, scores in results:
        if round(scores.average, 1) < quality_threshold:
            if get_name is not None:
                name = get_name(entity)
            elif isinstance(entity, dict):
                name = entity.get("name", "Unknown")
            else:
                name = getattr(entity, "name", "Unknown")
            failed_names.append(name)

    summary_parts = [
        f"passed={passed}/{total}",
        f"scores: min={min_score:.1f} max={max_score:.1f} avg={avg_score:.1f}",
        f"threshold={quality_threshold:.1f}",
        f"time={elapsed:.1f}s",
    ]
    if failed_names:
        summary_parts.append(f"below threshold: {', '.join(failed_names)}")

    if failed_names:
        logger.warning(
            "Batch %s summary: %s",
            entity_type,
            ", ".join(summary_parts),
        )
    else:
        logger.info(
            "Batch %s summary: %s",
            entity_type,
            ", ".join(summary_parts),
        )


def aggregate_errors(errors: list[str]) -> str:
    """Deduplicate and aggregate identical error messages.

    Instead of repeating ``"Failed to generate relationship after 3 attempts"``
    nine times, produces ``"Failed to generate relationship after 3 attempts (x9)"``.

    Args:
        errors: List of raw error messages (may contain duplicates).

    Returns:
        A single joined string with counts appended for repeated messages.
    """
    counts = Counter(errors)
    parts: list[str] = []
    for msg, count in counts.items():
        if count > 1:
            parts.append(f"{msg} (x{count})")
        else:
            parts.append(msg)
    return "; ".join(parts)
