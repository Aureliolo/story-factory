"""Entity-specific quality score models.

These models define the quality scoring dimensions for world-building entities:
characters, locations, factions, items, concepts, events, and calendars.

Each subclass inherits generic ``average``, ``to_dict``, and ``weak_dimensions``
from :class:`BaseQualityScores`.  Subclasses that set ``_EXCLUDED_FROM_AVERAGE``
exclude inflated dimensions (e.g. ``temporal_plausibility``) from the average
while still validating them via the ``dimension_minimum`` floor check.
"""

import logging
from typing import ClassVar

from pydantic import Field

from src.memory.world_quality._models import BaseQualityScores

logger = logging.getLogger(__name__)


class CharacterQualityScores(BaseQualityScores):
    """Quality scores for a character (0-10 scale).

    All score fields are required - no defaults. This ensures that the LLM
    must provide explicit scores or the parsing will fail, preventing silent
    fallbacks to meaningless default values.
    """

    _EXCLUDED_FROM_AVERAGE: ClassVar[frozenset[str]] = frozenset({"temporal_plausibility"})

    depth: float = Field(ge=0.0, le=10.0, description="Psychological complexity")
    goals: float = Field(
        alias="goal_clarity", ge=0.0, le=10.0, description="Clarity and story relevance"
    )
    flaws: float = Field(ge=0.0, le=10.0, description="Meaningful vulnerabilities")
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctiveness")
    arc_potential: float = Field(ge=0.0, le=10.0, description="Room for transformation")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )


class LocationQualityScores(BaseQualityScores):
    """Quality scores for a location (0-10 scale).

    All score fields are required - no defaults.
    """

    _EXCLUDED_FROM_AVERAGE: ClassVar[frozenset[str]] = frozenset({"temporal_plausibility"})

    atmosphere: float = Field(ge=0.0, le=10.0, description="Sensory richness, mood")
    significance: float = Field(
        alias="narrative_significance", ge=0.0, le=10.0, description="Plot/symbolic meaning"
    )
    story_relevance: float = Field(ge=0.0, le=10.0, description="Theme/character links")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )


class FactionQualityScores(BaseQualityScores):
    """Quality scores for a faction (0-10 scale).

    All score fields are required - no defaults.
    """

    _EXCLUDED_FROM_AVERAGE: ClassVar[frozenset[str]] = frozenset({"temporal_plausibility"})

    coherence: float = Field(ge=0.0, le=10.0, description="Internal consistency")
    influence: float = Field(ge=0.0, le=10.0, description="World impact")
    conflict_potential: float = Field(ge=0.0, le=10.0, description="Story conflict potential")
    distinctiveness: float = Field(ge=0.0, le=10.0, description="Memorable qualities")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )


class ItemQualityScores(BaseQualityScores):
    """Quality scores for an item (0-10 scale).

    All score fields are required - no defaults.
    """

    _EXCLUDED_FROM_AVERAGE: ClassVar[frozenset[str]] = frozenset({"temporal_plausibility"})

    significance: float = Field(
        alias="story_significance", ge=0.0, le=10.0, description="Story importance"
    )
    uniqueness: float = Field(ge=0.0, le=10.0, description="Distinctive qualities")
    narrative_potential: float = Field(ge=0.0, le=10.0, description="Plot opportunities")
    integration: float = Field(ge=0.0, le=10.0, description="Fits world")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )


class ConceptQualityScores(BaseQualityScores):
    """Quality scores for a concept (0-10 scale).

    All score fields are required - no defaults.
    """

    _EXCLUDED_FROM_AVERAGE: ClassVar[frozenset[str]] = frozenset({"temporal_plausibility"})

    relevance: float = Field(ge=0.0, le=10.0, description="Theme alignment")
    depth: float = Field(ge=0.0, le=10.0, description="Philosophical richness")
    manifestation: float = Field(ge=0.0, le=10.0, description="How it appears in story")
    resonance: float = Field(ge=0.0, le=10.0, description="Emotional impact")
    temporal_plausibility: float = Field(
        ge=0.0, le=10.0, description="Timeline consistency and era-appropriate placement"
    )


class EventQualityScores(BaseQualityScores):
    """Quality scores for a world event (0-10 scale).

    All score fields are required - no defaults.
    """

    _EXCLUDED_FROM_AVERAGE: ClassVar[frozenset[str]] = frozenset({"temporal_plausibility"})

    significance: float = Field(ge=0.0, le=10.0, description="How world-shaping is this event")
    temporal_plausibility: float = Field(ge=0.0, le=10.0, description="Calendar timeline fit")
    causal_coherence: float = Field(ge=0.0, le=10.0, description="Logical causes and consequences")
    narrative_potential: float = Field(ge=0.0, le=10.0, description="Story opportunity creation")
    entity_integration: float = Field(ge=0.0, le=10.0, description="Participant roles make sense")


class CalendarQualityScores(BaseQualityScores):
    """Quality scores for a calendar system (0-10 scale).

    All score fields are required - no defaults.
    """

    internal_consistency: float = Field(
        ge=0.0, le=10.0, description="Internal logic and coherence of the calendar system"
    )
    thematic_fit: float = Field(
        ge=0.0, le=10.0, description="How well the calendar fits the story's genre and setting"
    )
    completeness: float = Field(
        ge=0.0, le=10.0, description="Thoroughness of months, eras, and day names"
    )
    uniqueness: float = Field(
        ge=0.0, le=10.0, description="Distinctiveness from real-world or generic calendars"
    )
