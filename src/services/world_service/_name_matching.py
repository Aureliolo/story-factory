"""Name-matching helpers for entity lookup in the world build pipeline.

Provides fuzzy name matching to handle LLM name variations like
added articles ("The"), case differences, extra whitespace, possessives,
and abbreviation punctuation. Includes similarity-based fallback using
``calculate_name_similarity`` for near-misses that normalization alone
cannot resolve (e.g., "Glacier's Whisper" vs "Glacial Whisper").
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.entities import Entity

logger = logging.getLogger(__name__)

_LEADING_ARTICLES = ("the ", "a ", "an ")

# Matches possessive suffixes: "'s", "s'", "\u2019s" (curly apostrophe)
_POSSESSIVE_RE = re.compile(r"(?:'s|s'|\u2019s)(?=\s|$)", re.IGNORECASE)

# Matches dots preceded by a word character (abbreviation dots like "A.P." -> "AP")
_ABBREVIATION_DOT_RE = re.compile(r"(?<=\w)\.")


def _normalize_name(name: str) -> str:
    """Normalize an entity name for fuzzy comparison.

    Collapses whitespace, lowercases, strips surrounding square brackets
    (LLM artifact), and strips common English articles ("The", "A", "An")
    that LLMs frequently prepend, causing mismatches.
    """
    original = name
    name = name.strip()
    # Strip surrounding square brackets (LLM participant name artifact)
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]
    normalized = " ".join(name.split()).lower()
    for article in _LEADING_ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article) :]
            break
    if normalized != original.strip().lower():
        logger.debug("Normalized entity name from %r to %r", original, normalized)
    return normalized


def _deep_normalize(name: str) -> str:
    """Apply aggressive normalization for similarity comparison.

    Extends ``_normalize_name`` with additional transformations:
    - Strip possessives ("'s", "s'")
    - Normalize abbreviation punctuation ("A.P." -> "ap", "A. P." -> "a p")
    - Strip trailing punctuation
    - Collapse whitespace

    Args:
        name: Raw entity name.

    Returns:
        Deeply normalized name string.
    """
    result = _normalize_name(name)
    # Strip possessives
    result = _POSSESSIVE_RE.sub("", result)
    # Normalize abbreviation dots
    result = _ABBREVIATION_DOT_RE.sub("", result)
    # Strip trailing punctuation
    result = result.rstrip(".,;:!?-")
    # Collapse whitespace again after transformations
    result = " ".join(result.split())
    if result != name.strip().lower():
        logger.debug("Deep-normalized entity name from %r to %r", name, result)
    return result


def _find_entity_by_name(
    entities: list[Entity],
    name: str,
    threshold: float,
) -> Entity | None:
    """Find an entity by name with fuzzy matching and similarity fallback.

    Resolution order:
    1. Exact match (fast path).
    2. Normalized comparison (articles, case, whitespace).
    3. Similarity fallback using ``calculate_name_similarity`` with
       ``_deep_normalize`` on both sides. A single match above *threshold*
       is accepted; multiple matches (ambiguity) return None.

    Args:
        entities: List of entity objects with .name attribute.
        name: Name to search for.
        threshold: Minimum similarity score for the fallback (0.0 to 1.0).
            Must be explicitly provided from settings.

    Returns:
        Matching entity, or None if not found or ambiguous.

    Raises:
        ValueError: If threshold is outside [0.0, 1.0].
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

    # Exact match first (fast path)
    for e in entities:
        if e.name == name:
            return e

    # Fuzzy match: normalize both sides and collect all matches
    normalized_target = _normalize_name(name)
    matches = [e for e in entities if _normalize_name(e.name) == normalized_target]

    if len(matches) == 1:
        logger.debug("Fuzzy matched relationship entity: %r -> %r", name, matches[0].name)
        return matches[0]

    if len(matches) > 1:
        match_names = [e.name for e in matches]
        logger.warning("Ambiguous fuzzy match for %r: %s. Skipping assignment.", name, match_names)
        return None

    # Similarity fallback: deep-normalize both sides, compute similarity score
    from src.services.world_service._entities import calculate_name_similarity

    deep_target = _deep_normalize(name)
    if not deep_target:
        logger.warning(
            "Deep normalization of %r produced empty string, "
            "cannot perform similarity matching. Returning None.",
            name,
        )
        return None

    similarity_matches: list[tuple[Entity, float]] = []
    for e in entities:
        deep_candidate = _deep_normalize(e.name)
        if not deep_candidate:
            continue
        score = calculate_name_similarity(deep_target, deep_candidate)
        if score >= threshold:
            similarity_matches.append((e, score))

    if len(similarity_matches) == 1:
        matched_entity, score = similarity_matches[0]
        logger.debug(
            "Similarity matched entity: %r -> %r (score=%.3f, threshold=%.2f)",
            name,
            matched_entity.name,
            score,
            threshold,
        )
        return matched_entity

    if len(similarity_matches) > 1:
        match_info = [f"{e.name} ({s:.3f})" for e, s in similarity_matches]
        logger.warning(
            "Ambiguous similarity match for %r: %s. Skipping assignment.",
            name,
            match_info,
        )
        return None

    logger.debug(
        "No match found for entity name %r after exact, normalized, "
        "and similarity search (%d entities, threshold=%.2f)",
        name,
        len(entities),
        threshold,
    )
    return None
