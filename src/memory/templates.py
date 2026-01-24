"""Template models for story structures and presets."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Shared type for target length across templates and story briefs
type TargetLength = Literal["short_story", "novella", "novel"]


class CharacterTemplate(BaseModel):
    """Template for a character archetype."""

    name: str  # Placeholder name or role
    role: str  # protagonist, antagonist, mentor, etc.
    description: str
    personality_traits: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    arc_notes: str = ""


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
