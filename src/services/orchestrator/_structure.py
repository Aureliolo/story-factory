"""Architecture/structure phase mixin for StoryOrchestrator."""

import logging
from typing import Any

from src.agents import ResponseValidationError
from src.memory.story_state import Character, StoryState
from src.services.orchestrator._base import StoryOrchestratorBase

logger = logging.getLogger(__name__)


class StructureMixin(StoryOrchestratorBase):
    """Mixin providing architecture/structure phase functionality."""

    def build_story_structure(self) -> StoryState:
        """Have the architect build the story structure."""
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        logger.info("Building story structure...")
        self._set_phase("architect")
        self._emit("agent_start", "Architect", "Building world...")

        logger.info(f"Calling architect with model: {self.architect.model}")
        self.story_state = self.architect.build_story_structure(self.story_state)

        # Validate key outputs for language correctness
        try:
            if self.story_state.world_description:
                self._validate_response(self.story_state.world_description, "World description")
            if self.story_state.plot_summary:
                self._validate_response(self.story_state.plot_summary, "Plot summary")
        except ResponseValidationError as e:
            logger.warning(f"Validation warning during structure build: {e}")
            # Don't block on validation errors, just log them

        # Set total chapters for progress tracking
        self._total_chapters = len(self.story_state.chapters)

        logger.info(
            f"Structure built: {len(self.story_state.chapters)} chapters, {len(self.story_state.characters)} characters"
        )
        self._emit("agent_complete", "Architect", "Story structure complete!")
        return self.story_state

    def generate_more_characters(self, count: int = 2) -> list[Character]:
        """Generate additional characters for the story.

        Args:
            count: Number of characters to generate.

        Returns:
            List of new Character objects.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info(f"Generating {count} more characters...")
        self._emit("agent_start", "Architect", f"Generating {count} new characters...")

        existing_names = [c.name for c in self.story_state.characters]
        new_characters = self.architect.generate_more_characters(
            self.story_state, existing_names, count
        )

        # Add to story state
        self.story_state.characters.extend(new_characters)

        self._emit(
            "agent_complete",
            "Architect",
            f"Generated {len(new_characters)} new characters!",
        )
        return new_characters

    def generate_locations(self, count: int = 3) -> list[dict[str, Any]]:
        """Generate locations for the story world.

        Args:
            count: Number of locations to generate.

        Returns:
            List of location dictionaries.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info(f"Generating {count} locations...")
        self._emit("agent_start", "Architect", f"Generating {count} new locations...")

        # Get existing location names from world_description heuristic
        existing_locations: list[str] = []
        # Locations will be added to world database by the caller

        locations = self.architect.generate_locations(self.story_state, existing_locations, count)

        self._emit(
            "agent_complete",
            "Architect",
            f"Generated {len(locations)} new locations!",
        )
        return locations

    def generate_relationships(
        self, entity_names: list[str], existing_rels: list[tuple[str, str]], count: int = 5
    ) -> list[dict[str, Any]]:
        """Generate relationships between entities.

        Args:
            entity_names: Names of all entities that can have relationships.
            existing_rels: List of (source, target) tuples to avoid duplicates.
            count: Number of relationships to generate.

        Returns:
            List of relationship dictionaries.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info(f"Generating {count} relationships...")
        self._emit("agent_start", "Architect", f"Generating {count} new relationships...")

        relationships = self.architect.generate_relationships(
            self.story_state, entity_names, existing_rels, count
        )

        self._emit(
            "agent_complete",
            "Architect",
            f"Generated {len(relationships)} new relationships!",
        )
        return relationships

    def rebuild_world(self) -> StoryState:
        """Rebuild the entire world from scratch.

        This regenerates world description, characters, plot, and chapters.
        Use with caution if chapters have already been written.

        Returns:
            Updated StoryState.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info("Rebuilding entire world...")
        self._emit("agent_start", "Architect", "Rebuilding world from scratch...")

        # Clear existing content but keep the brief
        self.story_state.world_description = ""
        self.story_state.world_rules = []
        self.story_state.characters = []
        self.story_state.plot_summary = ""
        self.story_state.plot_points = []
        self.story_state.chapters = []

        # Rebuild everything
        self.story_state = self.architect.build_story_structure(self.story_state)

        self._emit("agent_complete", "Architect", "World rebuilt successfully!")
        return self.story_state

    def get_outline_summary(self) -> str:
        """Get a human-readable summary of the story outline."""
        if not self.story_state:
            raise ValueError("No story state available.")

        state = self.story_state
        summary_parts = [
            "=" * 50,
            "STORY OUTLINE",
            "=" * 50,
        ]

        # Handle projects created before brief feature was added
        if state.brief:
            summary_parts.extend(
                [
                    f"\nPREMISE: {state.brief.premise}",
                    f"GENRE: {state.brief.genre}",
                    f"TONE: {state.brief.tone}",
                    f"CONTENT RATING: {state.brief.content_rating}",
                ]
            )
        else:
            summary_parts.append("\n(No brief available)")

        if state.world_description:
            summary_parts.append(f"\nWORLD:\n{state.world_description[:500]}...")

        summary_parts.append("\nCHARACTERS:")

        for char in state.characters:
            summary_parts.append(f"  - {char.name} ({char.role}): {char.description}")

        summary_parts.append(f"\nPLOT SUMMARY:\n{state.plot_summary}")

        summary_parts.append(f"\nCHAPTER OUTLINE ({len(state.chapters)} chapters):")
        for ch in state.chapters:
            summary_parts.append(f"  {ch.number}. {ch.title}")
            summary_parts.append(f"     {ch.outline[:100]}...")

        return "\n".join(summary_parts)
