"""Conflict and tension mapping data types.

These models define the structure for:
- Conflict categories (alliance, rivalry, tension, neutral)
- Relationship classification mapping
- Conflict metrics and analysis results
"""

import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ConflictCategory(str, Enum):
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
    # Rivalry - active opposition, hostility
    "hates": ConflictCategory.RIVALRY,
    "enemy_of": ConflictCategory.RIVALRY,
    "opposes": ConflictCategory.RIVALRY,
    "betrayed": ConflictCategory.RIVALRY,
    "fights": ConflictCategory.RIVALRY,
    "hunts": ConflictCategory.RIVALRY,
    "destroys": ConflictCategory.RIVALRY,
    "attacks": ConflictCategory.RIVALRY,
    # Tension - potential conflict, negative but not active hostility
    "distrusts": ConflictCategory.TENSION,
    "competes_with": ConflictCategory.TENSION,
    "fears": ConflictCategory.TENSION,
    "resents": ConflictCategory.TENSION,
    "suspects": ConflictCategory.TENSION,
    "jealous_of": ConflictCategory.TENSION,
    "avoids": ConflictCategory.TENSION,
    # Neutral - informational, no inherent conflict
    "knows": ConflictCategory.NEUTRAL,
    "works_with": ConflictCategory.NEUTRAL,
    "related_to": ConflictCategory.NEUTRAL,
    "located_in": ConflictCategory.NEUTRAL,
    "owns": ConflictCategory.NEUTRAL,
    "created": ConflictCategory.NEUTRAL,
    "contains": ConflictCategory.NEUTRAL,
    "uses": ConflictCategory.NEUTRAL,
}


# Colors for conflict visualization (consistent with theme.py patterns)
CONFLICT_COLORS: dict[str, str] = {
    "alliance": "#4CAF50",  # Green
    "rivalry": "#F44336",  # Red
    "tension": "#FFC107",  # Yellow/Amber
    "neutral": "#2196F3",  # Blue
}


def classify_relationship(relation_type: str) -> ConflictCategory:
    """
    Map a relationship type string to its corresponding ConflictCategory.

    Parameters:
        relation_type (str): Relationship label (e.g., "enemy_of", "trusts", "works with").

    Returns:
        ConflictCategory: The category for the given relationship; defaults to `ConflictCategory.NEUTRAL` if unrecognized.
    """
    # Normalize to lowercase for matching
    normalized = relation_type.lower().replace("-", "_").replace(" ", "_")
    category = RELATION_CONFLICT_MAPPING.get(normalized)

    if category is None:
        logger.warning(
            f"Unknown relationship type '{relation_type}' (normalized: '{normalized}') - "
            f"defaulting to NEUTRAL. Consider adding to RELATION_CONFLICT_MAPPING."
        )
        category = ConflictCategory.NEUTRAL
    else:
        logger.debug(f"Classified relationship '{relation_type}' as {category.value}")

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
