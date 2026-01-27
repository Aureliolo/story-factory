"""Generation mixin for StoryService - title and world generation."""

import logging
from typing import Any

from src.memory.story_state import Character, StoryState

from ._base import StoryServiceBase

logger = logging.getLogger(__name__)


class GenerationMixin(StoryServiceBase):
    """Mixin providing title and world generation functionality."""

    def generate_title_suggestions(self, state: StoryState) -> list[str]:
        """Generate AI-powered title suggestions.

        Args:
            state: The story state with brief.

        Returns:
            List of suggested titles.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.generate_title_suggestions()

    def generate_more_characters(self, state: StoryState, count: int = 2) -> list[Character]:
        """Generate additional characters for the story.

        Args:
            state: The story state.
            count: Number of characters to generate.

        Returns:
            List of new Character objects.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        new_chars = orchestrator.generate_more_characters(count)
        self._sync_state(orchestrator, state)
        return new_chars

    def generate_locations(self, state: StoryState, count: int = 3) -> list[Any]:
        """Generate locations for the story world.

        Args:
            state: The story state.
            count: Number of locations to generate.

        Returns:
            List of location dictionaries.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        locations = orchestrator.generate_locations(count)
        self._sync_state(orchestrator, state)
        return locations

    def generate_relationships(
        self,
        state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        count: int = 5,
    ) -> list[Any]:
        """Generate relationships between entities.

        Args:
            state: The story state.
            entity_names: Names of all entities.
            existing_rels: Existing (source, target) relationships.
            count: Number of relationships to generate.

        Returns:
            List of relationship dictionaries.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        relationships = orchestrator.generate_relationships(entity_names, existing_rels, count)
        self._sync_state(orchestrator, state)
        return relationships

    def rebuild_world(self, state: StoryState) -> StoryState:
        """Rebuild the entire world from scratch.

        Args:
            state: The story state.

        Returns:
            Updated StoryState.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        orchestrator.rebuild_world()
        self._sync_state(orchestrator, state)
        return state
