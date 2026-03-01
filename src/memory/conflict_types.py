"""Conflict and tension mapping data types.

These models define the structure for:
- Conflict categories (alliance, rivalry, tension, neutral)
- Relationship classification mapping
- Conflict metrics and analysis results
"""

import functools
import logging
import threading
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ConflictCategory(StrEnum):
    """Categories for relationship conflict classification."""

    ALLIANCE = "alliance"  # Positive relationships
    RIVALRY = "rivalry"  # Active opposition
    TENSION = "tension"  # Potential conflict, distrust
    NEUTRAL = "neutral"  # Neither positive nor negative


# Mapping from relation_type strings to conflict categories
# This maps the existing relation types from src/memory/entities.py
RELATION_CONFLICT_MAPPING: dict[str, ConflictCategory] = {
    # Alliance - positive, cooperative relationships
    "loves": ConflictCategory.ALLIANCE,
    "ally_of": ConflictCategory.ALLIANCE,
    "allies_with": ConflictCategory.ALLIANCE,
    "protects": ConflictCategory.ALLIANCE,
    "member_of": ConflictCategory.ALLIANCE,
    "serves": ConflictCategory.ALLIANCE,
    "trusts": ConflictCategory.ALLIANCE,
    "supports": ConflictCategory.ALLIANCE,
    "befriends": ConflictCategory.ALLIANCE,
    "mentors": ConflictCategory.ALLIANCE,
    "parent_of": ConflictCategory.ALLIANCE,
    "child_of": ConflictCategory.ALLIANCE,
    "works_for": ConflictCategory.ALLIANCE,
    "follows": ConflictCategory.ALLIANCE,
    "leads": ConflictCategory.ALLIANCE,
    "leader_of": ConflictCategory.ALLIANCE,
    "admires": ConflictCategory.ALLIANCE,
    "collaborates_with": ConflictCategory.ALLIANCE,
    "friends_with": ConflictCategory.ALLIANCE,
    "friends": ConflictCategory.ALLIANCE,
    # Alliance - LLM-generated compound types
    "mutual_respect": ConflictCategory.ALLIANCE,
    "admiring_friendship": ConflictCategory.ALLIANCE,
    "mentor_like_bond": ConflictCategory.ALLIANCE,
    "bonded_with": ConflictCategory.ALLIANCE,
    "devoted_to": ConflictCategory.ALLIANCE,
    "loyal_to": ConflictCategory.ALLIANCE,
    "respects": ConflictCategory.ALLIANCE,
    "cares_for": ConflictCategory.ALLIANCE,
    "inspires": ConflictCategory.ALLIANCE,
    "guided_by": ConflictCategory.ALLIANCE,
    "sibling_of": ConflictCategory.ALLIANCE,
    "grateful_to": ConflictCategory.ALLIANCE,
    # Rivalry - active opposition, hostility
    "hates": ConflictCategory.RIVALRY,
    "enemy_of": ConflictCategory.RIVALRY,
    "enemies_with": ConflictCategory.RIVALRY,
    "rivals_with": ConflictCategory.RIVALRY,
    "rivals": ConflictCategory.RIVALRY,
    "opposes": ConflictCategory.RIVALRY,
    "betrayed": ConflictCategory.RIVALRY,
    "fights": ConflictCategory.RIVALRY,
    "hunts": ConflictCategory.RIVALRY,
    "destroys": ConflictCategory.RIVALRY,
    "attacks": ConflictCategory.RIVALRY,
    # Rivalry - LLM-generated compound types
    "sworn_enemy": ConflictCategory.RIVALRY,
    "bitter_rivals": ConflictCategory.RIVALRY,
    "nemesis_of": ConflictCategory.RIVALRY,
    "despises": ConflictCategory.RIVALRY,
    "undermines": ConflictCategory.RIVALRY,
    "seeks_revenge": ConflictCategory.RIVALRY,
    "betrays": ConflictCategory.RIVALRY,
    # Tension - potential conflict, negative but not active hostility
    "distrusts": ConflictCategory.TENSION,
    "competes_with": ConflictCategory.TENSION,
    "fears": ConflictCategory.TENSION,
    "resents": ConflictCategory.TENSION,
    "suspects": ConflictCategory.TENSION,
    "jealous_of": ConflictCategory.TENSION,
    "avoids": ConflictCategory.TENSION,
    "threatens": ConflictCategory.TENSION,
    "manipulates": ConflictCategory.TENSION,
    "envies": ConflictCategory.TENSION,
    "owes_debt_to": ConflictCategory.TENSION,
    # Tension - LLM-generated compound types
    "friendly_rivalry": ConflictCategory.TENSION,
    "uneasy_alliance": ConflictCategory.TENSION,
    "reluctant_allies": ConflictCategory.TENSION,
    "wary_of": ConflictCategory.TENSION,
    "indebted_to": ConflictCategory.TENSION,
    "conflicted_about": ConflictCategory.TENSION,
    "intimidated_by": ConflictCategory.TENSION,
    "obsessed_with": ConflictCategory.TENSION,
    "haunted_by": ConflictCategory.TENSION,
    "dependent_on": ConflictCategory.TENSION,
    "bound_to": ConflictCategory.TENSION,
    "conflicts_with": ConflictCategory.TENSION,
    "challenges": ConflictCategory.TENSION,
    # Neutral - informational, no inherent conflict
    "knows": ConflictCategory.NEUTRAL,
    "works_with": ConflictCategory.NEUTRAL,
    "related_to": ConflictCategory.NEUTRAL,
    "located_in": ConflictCategory.NEUTRAL,
    "owns": ConflictCategory.NEUTRAL,
    "created": ConflictCategory.NEUTRAL,
    "contains": ConflictCategory.NEUTRAL,
    "uses": ConflictCategory.NEUTRAL,
    "inhabits": ConflictCategory.NEUTRAL,
    "teaches": ConflictCategory.NEUTRAL,
    "studies": ConflictCategory.NEUTRAL,
    "develops": ConflictCategory.NEUTRAL,
    "interconnected": ConflictCategory.NEUTRAL,
    "consults_with": ConflictCategory.NEUTRAL,
    "rules": ConflictCategory.NEUTRAL,
    "enforces": ConflictCategory.NEUTRAL,
    "controls": ConflictCategory.NEUTRAL,
    "works_at": ConflictCategory.NEUTRAL,
    "based_in": ConflictCategory.NEUTRAL,
    "lives_in": ConflictCategory.NEUTRAL,
    "located_near": ConflictCategory.NEUTRAL,
    "connected_to": ConflictCategory.NEUTRAL,
    # Romantic / emotional bonds
    "romantic_interest": ConflictCategory.TENSION,
}

# Sorted list of all valid relationship types for constraining LLM prompts
VALID_RELATIONSHIP_TYPES: list[str] = sorted(RELATION_CONFLICT_MAPPING.keys())

# Hierarchical relationship types where circular references indicate real structural problems
# (e.g., A parent_of B parent_of A is clearly wrong, vs A knows B knows A which is fine)
HIERARCHICAL_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {
        "parent_of",
        "child_of",
        "mentors",
        "works_for",
        "leads",
        "leader_of",
        "serves",
        "follows",
        "rules",
        "controls",
        "enforces",
        "teaches",
        "studies",
        "created",
        "owns",
        "contains",
    }
)

# Keys sorted by length descending for substring matching (prefer longer matches first)
_SORTED_KEYS_BY_LENGTH: list[str] = sorted(RELATION_CONFLICT_MAPPING.keys(), key=len, reverse=True)

# Priority ordering for word-level conflict classification.
# Higher values win when multiple signal words appear in a compound type.
_CONFLICT_PRIORITY: dict[ConflictCategory, int] = {
    ConflictCategory.RIVALRY: 3,
    ConflictCategory.TENSION: 2,
    ConflictCategory.ALLIANCE: 1,
    ConflictCategory.NEUTRAL: 0,
}


def _validate_conflict_priority() -> None:
    """Validate that _CONFLICT_PRIORITY has an entry for every ConflictCategory."""
    missing = set(ConflictCategory) - set(_CONFLICT_PRIORITY)
    if missing:
        raise RuntimeError(f"_CONFLICT_PRIORITY missing categories: {missing}")


_validate_conflict_priority()

# Word-level lookup: individual words that strongly signal a known relationship type.
# Used as a last-resort fallback when substring matching fails on novel compound types.
# Maps a single word (found after splitting on underscores) to the canonical relation type.
_WORD_TO_RELATION: dict[str, str] = {
    # Alliance signals
    "mentor": "mentors",
    "friend": "friends",
    "friendship": "friends",
    "ally": "ally_of",
    "allies": "allies_with",
    "loyal": "loyal_to",
    "devoted": "devoted_to",
    "protects": "protects",
    "trust": "trusts",
    "trusts": "trusts",
    "love": "loves",
    "loves": "loves",
    "admire": "admires",
    "admires": "admires",
    "respects": "respects",
    "sibling": "sibling_of",
    "bonded": "bonded_with",
    "inspires": "inspires",
    "protect": "protects",
    "support": "supports",
    "supports": "supports",
    "guide": "guided_by",
    "cares": "cares_for",
    "colleague": "works_with",
    "follows": "follows",
    # Rivalry signals
    "hostile": "enemy_of",
    "contempt": "despises",
    "hatred": "hates",
    "enemy": "enemy_of",
    "enemies": "enemies_with",
    "rival": "rivals",
    "rivalry": "rivals",
    "rivals": "rivals",
    "nemesis": "nemesis_of",
    "revenge": "seeks_revenge",
    "despises": "despises",
    "betrayed": "betrayed",
    "hates": "hates",
    "conflict": "rivals",
    "conflicts": "rivals",
    "oppose": "opposes",
    "opposes": "opposes",
    "disagree": "opposes",
    "disagrees": "opposes",
    "betray": "betrays",
    "betrays": "betrays",
    "attack": "attacks",
    "attacks": "attacks",
    "destroy": "destroys",
    "fight": "fights",
    # Tension signals
    "distrustful": "distrusts",
    "annoyance": "resents",
    "suspicious": "suspects",
    "questioning": "challenges",
    "stealing": "manipulates",
    "grudge": "resents",
    "fears": "fears",
    "fear": "fears",
    "wary": "wary_of",
    "reluctant": "reluctant_allies",
    "uneasy": "uneasy_alliance",
    "obsessed": "obsessed_with",
    "haunted": "haunted_by",
    "intimidated": "intimidated_by",
    "conflicted": "conflicted_about",
    "jealous": "jealous_of",
    "envies": "envies",
    "distrusts": "distrusts",
    "suspects": "suspects",
    "challenge": "challenges",
    "challenges": "challenges",
    "dissent": "challenges",
    "objection": "challenges",
    "manipulate": "manipulates",
    "manipulates": "manipulates",
    "resent": "resents",
    "resents": "resents",
    "threaten": "threatens",
    "threatens": "threatens",
    # L1: Prose sentiment keywords (formerly in separate _PROSE_SENTIMENT_KEYWORDS dict).
    # Merged here to use priority-based word-level matching instead of substring search.
    "admiration": "admires",
    "hostility": "enemy_of",
    "partnership": "allies_with",
    "cooperation": "allies_with",
    "animosity": "enemy_of",
    "antagonism": "rivals",
    "camaraderie": "friends",
    "enmity": "enemy_of",
    "affection": "cares_for",
    "devotion": "devoted_to",
    "respect": "respects",
    "distrust": "distrusts",
    "suspicion": "suspects",
    "reverence": "admires",
    "alliance": "allies_with",
    "loyalty": "loyal_to",
    "tension": "conflicts_with",
}


@functools.lru_cache(maxsize=256)
def normalize_relation_type(raw_type: str) -> str:
    """Normalize a free-form relationship type to the controlled vocabulary.

    Applies the following normalization steps:
    1. Lowercase, replace hyphens/spaces with underscores.
    2. If the normalized string is a known type, return it.
    3. If pipe-delimited, take the first recognized part.
    4. If a known type appears as a substring, extract and return it
       (prefers longer matches to avoid partial hits).
    5. Split on underscores and check individual words against _WORD_TO_RELATION.
       When multiple words match, the highest-conflict-priority category wins
       (RIVALRY > TENSION > ALLIANCE > NEUTRAL).
    6. Otherwise return the normalized string unchanged.

    Args:
        raw_type: Raw relationship type string from the LLM or legacy data.

    Returns:
        Normalized relationship type string.
    """
    normalized = raw_type.lower().strip().replace("-", "_").replace(" ", "_")

    # Direct match
    if normalized in RELATION_CONFLICT_MAPPING:
        return normalized

    # Pipe-delimited: take first recognized part
    if "|" in normalized:
        parts = [p.strip() for p in normalized.split("|") if p.strip()]
        for part in parts:
            if part in RELATION_CONFLICT_MAPPING:
                logger.debug(
                    "Normalized pipe-delimited type '%s' -> '%s' (first recognized part)",
                    raw_type,
                    part,
                )
                return part
        # No recognized part — fall through to substring matching on the full string
        logger.debug(
            "No recognized part in pipe-delimited type '%s', trying substring match",
            raw_type,
        )

    # Substring match: check if any known key appears in the normalized input
    # Sort by length descending so "allies_with" matches before "allies"
    for key in _SORTED_KEYS_BY_LENGTH:
        if key in normalized and key != normalized:
            logger.debug(
                "Normalized prose type '%s' -> '%s' (substring match)",
                raw_type,
                key,
            )
            return key

    # Word-level match: split on underscores and check ALL words against
    # the lookup. Picks the highest-conflict-priority match so that word
    # ordering doesn't affect classification (e.g., "colleague_and_occasional_rival"
    # matches RIVALRY via "rival" even though "colleague" appears first).
    # Priority: RIVALRY > TENSION > ALLIANCE > NEUTRAL
    words = normalized.split("_")
    best_type: str | None = None
    best_priority = -1
    best_word = ""
    for word in words:
        if word in _WORD_TO_RELATION:
            candidate = _WORD_TO_RELATION[word]
            category = RELATION_CONFLICT_MAPPING.get(candidate, ConflictCategory.NEUTRAL)
            priority = _CONFLICT_PRIORITY[category]
            if priority > best_priority:
                best_priority = priority
                best_type = candidate
                best_word = word
    if best_type is not None:
        logger.debug(
            "Normalized novel type '%s' -> '%s' (word-level match on '%s')",
            raw_type,
            best_type,
            best_word,
        )
        return best_type

    logger.debug("No normalization match for '%s', returning as-is: '%s'", raw_type, normalized)
    return normalized


# Load-time validation: every _WORD_TO_RELATION value must be a key in RELATION_CONFLICT_MAPPING
def _validate_word_to_relation() -> None:
    """Validate that every _WORD_TO_RELATION value is a key in RELATION_CONFLICT_MAPPING."""
    invalid = set(_WORD_TO_RELATION.values()) - set(RELATION_CONFLICT_MAPPING)
    if invalid:
        raise RuntimeError(f"_WORD_TO_RELATION references unknown types: {invalid}")


_validate_word_to_relation()


# Colors for conflict visualization (consistent with theme.py patterns)
CONFLICT_COLORS: dict[str, str] = {
    "alliance": "#4CAF50",  # Green
    "rivalry": "#F44336",  # Red
    "tension": "#FFC107",  # Yellow/Amber
    "neutral": "#2196F3",  # Blue
}


_warned_types: set[str] = set()
_warned_types_lock = threading.Lock()


def _reset_warned_types() -> None:
    """Reset the warned-types dedup set (for test teardown)."""
    with _warned_types_lock:
        cleared_count = len(_warned_types)
        _warned_types.clear()
    logger.debug("Reset warned-types dedup set (cleared=%d)", cleared_count)


@functools.lru_cache(maxsize=256)
def classify_relationship(relation_type: str) -> ConflictCategory:
    """
    Map a relationship type string to its corresponding ConflictCategory.

    Handles pipe-delimited compound types (e.g., ``"created|consults_with"``) by
    splitting on ``|`` and returning the first non-NEUTRAL match (or NEUTRAL if
    all parts are neutral/unknown).

    Parameters:
        relation_type (str): Relationship label (e.g., "enemy_of", "trusts", "works with").

    Returns:
        ConflictCategory: The category for the given relationship; defaults to
        ``ConflictCategory.NEUTRAL`` if unrecognized.
    """
    # Handle pipe-delimited compound types the LLM sometimes generates
    if "|" in relation_type:
        parts = [p.strip() for p in relation_type.split("|") if p.strip()]
        if parts:
            for part in parts:
                part_category = classify_relationship(part)
                if part_category != ConflictCategory.NEUTRAL:
                    logger.debug(
                        "Compound type '%s': resolved to %s via part '%s'",
                        relation_type,
                        part_category.value,
                        part,
                    )
                    return part_category
            logger.debug("Compound type '%s': all parts are NEUTRAL", relation_type)
            return ConflictCategory.NEUTRAL

    # Apply the full 4-tier normalization chain (direct match, pipe-delimited,
    # substring, word-level) before looking up the conflict category. This
    # ensures compound types like "colleague_and_occasional_rival" are resolved
    # to a known type before classification.
    normalized = normalize_relation_type(relation_type)
    category = RELATION_CONFLICT_MAPPING.get(normalized)

    if category is None:
        # Note: lru_cache means this block only executes on cache misses.
        # The _warned_types guard provides a safety net if the cache is
        # cleared or evicted, ensuring at most one warning per unknown type.
        with _warned_types_lock:
            if normalized not in _warned_types:
                logger.warning(
                    "Unknown relationship type '%s' (normalized: '%s') - "
                    "defaulting to NEUTRAL. Consider adding to RELATION_CONFLICT_MAPPING.",
                    relation_type,
                    normalized,
                )
                _warned_types.add(normalized)
        category = ConflictCategory.NEUTRAL
    else:
        logger.debug("Classified relationship '%s' as %s", relation_type, category.value)

    return category


def get_conflict_color(category: ConflictCategory | str) -> str:
    """
    Return the hex color used to visualize a conflict category.

    Unknown or unrecognized categories default to the neutral color.
    Returns:
        Hex color string (e.g., "#4CAF50") for the given category; defaults to the neutral color if not found.
    """
    if isinstance(category, ConflictCategory):
        key = category.value
    else:
        key = category.lower()

    color = CONFLICT_COLORS.get(key)
    if color is None:
        logger.warning(
            f"Unknown conflict category '{category}' (key: '{key}') - "
            f"defaulting to neutral color. Valid categories: {list(CONFLICT_COLORS.keys())}"
        )
        return CONFLICT_COLORS["neutral"]

    logger.debug(f"Resolved color for '{category}': {color}")
    return color


class TensionPair(BaseModel):
    """A pair of entities with tension/conflict metrics."""

    entity_a_id: str = Field(description="First entity ID")
    entity_a_name: str = Field(description="First entity name")
    entity_b_id: str = Field(description="Second entity ID")
    entity_b_name: str = Field(description="Second entity name")
    score: float = Field(ge=0, le=1, description="Tension/alliance strength (0-1)")
    relationship_types: list[str] = Field(
        default_factory=list,
        description="Relationship types between these entities",
    )

    model_config = ConfigDict(use_enum_values=True)


class FactionCluster(BaseModel):
    """A cluster of allied entities (faction)."""

    id: str = Field(description="Cluster identifier")
    entity_ids: list[str] = Field(description="Entity IDs in this cluster")
    entity_names: list[str] = Field(description="Entity names in this cluster")
    internal_alliance_strength: float = Field(
        ge=0,
        le=1,
        description="Average alliance strength within the cluster",
    )

    model_config = ConfigDict(use_enum_values=True)


class ConflictMetrics(BaseModel):
    """Aggregated conflict metrics for the world.

    Provides analysis of tensions, alliances, and faction dynamics.
    """

    # Tension analysis
    highest_tension_pairs: list[TensionPair] = Field(
        default_factory=list,
        description="Top pairs with rivalry/tension relationships",
    )
    strongest_alliances: list[TensionPair] = Field(
        default_factory=list,
        description="Top pairs with alliance relationships",
    )

    # Entity analysis
    isolated_entities: list[str] = Field(
        default_factory=list,
        description="Entity IDs with no alliance connections",
    )
    most_connected_entities: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Entities with most relationships: (entity_id, count)",
    )

    # Faction analysis
    faction_clusters: list[FactionCluster] = Field(
        default_factory=list,
        description="Groups of allied entities",
    )

    # Overall metrics
    total_relationships: int = Field(default=0, description="Total relationship count")
    alliance_count: int = Field(default=0, description="Number of alliance relationships")
    rivalry_count: int = Field(default=0, description="Number of rivalry relationships")
    tension_count: int = Field(default=0, description="Number of tension relationships")
    neutral_count: int = Field(default=0, description="Number of neutral relationships")
    conflict_density: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Ratio of (rivalry + tension) / total relationships",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def has_conflicts(self) -> bool:
        """
        Determine whether any rivalry or tension relationships are present.

        Returns:
            bool: `True` if `rivalry_count` > 0 or `tension_count` > 0, `False` otherwise.
        """
        return self.rivalry_count > 0 or self.tension_count > 0


class ConflictGraphNode(BaseModel):
    """A node in the conflict graph visualization."""

    id: str = Field(description="Node ID (entity ID)")
    label: str = Field(description="Display label")
    entity_type: str = Field(description="Entity type (character, faction, etc.)")
    color: str = Field(description="Node color")
    size: int = Field(default=20, description="Node size based on connections")
    title: str = Field(default="", description="Tooltip content")

    model_config = ConfigDict(use_enum_values=True)


class ConflictGraphEdge(BaseModel):
    """An edge in the conflict graph visualization."""

    from_id: str = Field(description="Source node ID")
    to_id: str = Field(description="Target node ID")
    relation_type: str = Field(description="Original relationship type")
    category: ConflictCategory = Field(description="Conflict category")
    color: str = Field(description="Edge color based on category")
    width: int = Field(default=1, description="Edge width based on strength")
    title: str = Field(default="", description="Tooltip content")
    dashes: bool = Field(default=False, description="Use dashed line for tension")

    model_config = ConfigDict(use_enum_values=True)


class ConflictGraphData(BaseModel):
    """Complete data for conflict graph visualization."""

    nodes: list[ConflictGraphNode] = Field(default_factory=list)
    edges: list[ConflictGraphEdge] = Field(default_factory=list)
    metrics: ConflictMetrics = Field(default_factory=ConflictMetrics)

    model_config = ConfigDict(use_enum_values=True)
