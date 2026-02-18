"""Name-matching helpers for entity lookup in the world build pipeline.

Provides fuzzy name matching to handle LLM name variations like
added articles ("The"), case differences, and extra whitespace.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.entities import Entity

logger = logging.getLogger(__name__)

_LEADING_ARTICLES = ("the ", "a ", "an ")


def _normalize_name(name: str) -> str:
    """Normalize an entity name for fuzzy comparison.

    Collapses whitespace, lowercases, and strips common English articles
    ("The", "A", "An") that LLMs frequently prepend, causing mismatches
    (e.g., "The Echoes of the Network" vs "Echoes of the Network").
    """
    normalized = " ".join(name.split()).lower()
    for article in _LEADING_ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article) :]
            break
    return normalized


def _find_entity_by_name(entities: list[Entity], name: str) -> Entity | None:
    """Find an entity by name with fuzzy matching.

    Tries exact match first, then falls back to normalized comparison
    to handle LLM name variations like added "The" prefixes or
    case differences. If multiple entities match via fuzzy matching,
    logs a warning and returns None to avoid ambiguous assignment.

    Args:
        entities: List of entity objects with .name attribute.
        name: Name to search for.

    Returns:
        Matching entity, or None if not found or ambiguous.
    """
    # Exact match first (fast path)
    for e in entities:
        if e.name == name:
            return e

    # Fuzzy match: normalize both sides and collect all matches
    normalized_target = _normalize_name(name)
    matches = [e for e in entities if _normalize_name(e.name) == normalized_target]

    if len(matches) == 1:
        logger.debug(f"Fuzzy matched relationship entity: '{name}' -> '{matches[0].name}'")
        return matches[0]

    if len(matches) > 1:
        match_names = [e.name for e in matches]
        logger.warning(f"Ambiguous fuzzy match for '{name}': {match_names}. Skipping assignment.")
        return None

    return None
