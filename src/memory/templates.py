"""Template models for story structures and presets."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def normalize_traits(v: Any) -> Any:
    """Normalize plain-string personality traits to PersonalityTrait dicts.

    YAML templates use plain strings like ["brave", "clever"]; this converts
    them to ``{"trait": "brave", "category": "core"}`` for Pydantic.
    """
    if not isinstance(v, list):
        return v
    return [{"trait": item, "category": "core"} if isinstance(item, str) else item for item in v]


class PersonalityTrait(BaseModel):
    """A categorized personality trait for richer character modeling.

    Categories help continuity checks (distinguishing core traits from flaws
    for voice consistency) and quality judges (the "flaws" dimension benefits
    from knowing which traits are flaws).
    """

    trait: str
    category: Literal["core", "flaw", "quirk"] = "core"


# Shared type for target length across templates and story briefs
type TargetLength = Literal["short_story", "novella", "novel"]


class EntityHints(BaseModel):
    """Hints for entity generation in a world template."""

    character_roles: list[str] = Field(
        default_factory=list, description="Suggested character roles for this genre"
    )
    location_types: list[str] = Field(
        default_factory=list, description="Types of locations typical for this genre"
    )
    faction_types: list[str] = Field(
        default_factory=list, description="Types of factions/organizations in this genre"
    )
    item_types: list[str] = Field(
        default_factory=list, description="Types of significant items in this genre"
    )
    concept_types: list[str] = Field(
        default_factory=list, description="Key concepts/themes for this genre"
    )


class WorldTemplate(BaseModel):
    """Template for a genre-specific world preset."""

    id: str = Field(description="Unique identifier for this template")
    name: str = Field(description="Display name of the template")
    description: str = Field(description="Brief description of the world style")
    is_builtin: bool = Field(default=False, description="Whether this is a built-in template")
    genre: str = Field(description="Primary genre this template supports")
    entity_hints: EntityHints = Field(
        default_factory=EntityHints, description="Hints for generating entities"
    )
    relationship_patterns: list[str] = Field(
        default_factory=list, description="Common relationship types in this genre"
    )
    naming_style: str = Field(default="", description="Naming conventions for this genre")
    recommended_counts: dict[str, tuple[int, int]] = Field(
        default_factory=dict, description="Recommended entity counts by type (min, max)"
    )
    atmosphere: str = Field(default="", description="Overall atmosphere and mood")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")

    @field_validator("recommended_counts")
    @classmethod
    def _validate_recommended_counts(
        cls, value: dict[str, tuple[int, int]]
    ) -> dict[str, tuple[int, int]]:
        """Validate that recommended_counts contains valid (min, max) pairs.

        Note: Pydantic's type annotation validates tuple format/length before this runs.
        This validator checks semantic constraints (non-negative, min <= max).
        """
        for key, (min_count, max_count) in value.items():
            if min_count < 0 or max_count < 0:
                raise ValueError(
                    f"Invalid recommended_counts for '{key}': negative values not allowed"
                )
            if min_count > max_count:
                raise ValueError(
                    f"Invalid recommended_counts for '{key}': min ({min_count}) > max ({max_count})"
                )
        return value


class CharacterTemplate(BaseModel):
    """Template for a character archetype."""

    name: str  # Placeholder name or role
    role: str  # protagonist, antagonist, mentor, etc.
    description: str
    personality_traits: list[PersonalityTrait] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    arc_notes: str = ""
    arc_type: str | None = None  # Reference to arc template ID (e.g., "hero_journey")

    @field_validator("personality_traits", mode="before")
    @classmethod
    def normalize_personality_traits(cls, v: Any) -> Any:
        """Normalize plain-string personality traits from YAML templates."""
        return normalize_traits(v)


class PlotPointTemplate(BaseModel):
    """Template for a plot point."""

    description: str
    act: int | None = None  # Which act this belongs to (1, 2, 3)
    percentage: int | None = None  # Approximate story position (0-100)


class StructurePreset(BaseModel):
    """Predefined story structure (three-act, hero's journey, etc.)."""

    id: str
    name: str
    description: str
    acts: list[str] = Field(default_factory=list)  # Names of acts/sections
    plot_points: list[PlotPointTemplate] = Field(default_factory=list)
    beats: list[str] = Field(default_factory=list)  # Story beats


class StoryTemplate(BaseModel):
    """Complete story template with genre, structure, and character archetypes."""

    id: str
    name: str
    description: str
    is_builtin: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Story brief defaults
    genre: str
    subgenres: list[str] = Field(default_factory=list)
    tone: str
    themes: list[str] = Field(default_factory=list)
    setting_time: str = ""
    setting_place: str = ""
    target_length: TargetLength = "novel"

    # Structure preset reference
    structure_preset_id: str | None = None

    # Template components
    world_description: str = ""
    world_rules: list[str] = Field(default_factory=list)
    characters: list[CharacterTemplate] = Field(default_factory=list)
    plot_points: list[PlotPointTemplate] = Field(default_factory=list)

    # User customization
    author: str = ""  # Creator of template
    tags: list[str] = Field(default_factory=list)  # For searching/filtering
