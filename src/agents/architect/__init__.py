"""Architect Agent package - Story structure design and world-building.

This package provides the ArchitectAgent class which is responsible for:
- World-building and setting creation
- Character design and development
- Plot outline generation
- Chapter structure planning
- Outline variation generation
"""

from ._agent import (
    ARCHITECT_SYSTEM_PROMPT,
    CHAPTER_SCHEMA,
    CHARACTER_SCHEMA,
    LOCATION_SCHEMA,
    PLOT_POINT_SCHEMA,
    RELATIONSHIP_SCHEMA,
    ArchitectAgentBase,
)
from ._structure import StructureMixin
from ._variations import VariationsMixin
from ._world import WorldMixin


class ArchitectAgent(WorldMixin, StructureMixin, VariationsMixin, ArchitectAgentBase):
    """Agent that designs story structure, characters, and outlines.

    This class combines all architect functionality through mixins:
    - ArchitectAgentBase: Core initialization and utilities
    - WorldMixin: World-building, characters, locations, relationships
    - StructureMixin: Plot outlines, chapter structures, story building
    - VariationsMixin: Outline variation generation and parsing
    """

    pass


__all__ = [
    "ARCHITECT_SYSTEM_PROMPT",
    "CHAPTER_SCHEMA",
    "CHARACTER_SCHEMA",
    "LOCATION_SCHEMA",
    "PLOT_POINT_SCHEMA",
    "RELATIONSHIP_SCHEMA",
    "ArchitectAgent",
    "ArchitectAgentBase",
    "StructureMixin",
    "VariationsMixin",
    "WorldMixin",
]
