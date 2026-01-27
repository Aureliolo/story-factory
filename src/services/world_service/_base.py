"""Base class and data classes for WorldService."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from src.memory.templates import WorldTemplate
from src.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class WorldBuildProgress:
    """Progress information for world building operations."""

    step: int
    total_steps: int
    message: str
    entity_type: str | None = None
    count: int = 0


@dataclass
class WorldBuildOptions:
    """Options for world building operations.

    Attributes:
        clear_existing: Whether to clear existing world data first.
        generate_structure: Whether to generate story structure (characters, chapters).
        generate_locations: Whether to generate location entities.
        generate_factions: Whether to generate faction entities.
        generate_items: Whether to generate item entities.
        generate_concepts: Whether to generate concept entities.
        generate_relationships: Whether to generate relationships between entities.
        cancellation_event: Optional threading.Event to signal cancellation.
        world_template: Optional world template for genre-specific hints.
    """

    clear_existing: bool = False
    generate_structure: bool = True
    generate_locations: bool = True
    generate_factions: bool = True
    generate_items: bool = True
    generate_concepts: bool = True
    generate_relationships: bool = True
    cancellation_event: threading.Event | None = field(default=None, repr=False)
    world_template: WorldTemplate | None = field(default=None, repr=False)

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self.cancellation_event is not None and self.cancellation_event.is_set()

    @classmethod
    def full(
        cls,
        cancellation_event: threading.Event | None = None,
        world_template: WorldTemplate | None = None,
    ) -> WorldBuildOptions:
        """Create options for full world build (everything, keeping existing data)."""
        return cls(
            clear_existing=False,
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_relationships=True,
            cancellation_event=cancellation_event,
            world_template=world_template,
        )

    @classmethod
    def full_rebuild(
        cls,
        cancellation_event: threading.Event | None = None,
        world_template: WorldTemplate | None = None,
    ) -> WorldBuildOptions:
        """Create options for full world rebuild (everything, clearing existing first)."""
        return cls(
            clear_existing=True,
            generate_structure=True,
            generate_locations=True,
            generate_factions=True,
            generate_items=True,
            generate_concepts=True,
            generate_relationships=True,
            cancellation_event=cancellation_event,
            world_template=world_template,
        )


class WorldServiceBase:
    """Base class for world service."""

    def __init__(self, settings: Settings | None = None):
        """Initialize WorldService.

        Args:
            settings: Application settings. If None, loads from src/settings.json.
        """
        logger.debug("Initializing WorldService")
        self.settings = settings or Settings.load()
        logger.debug("WorldService initialized successfully")
