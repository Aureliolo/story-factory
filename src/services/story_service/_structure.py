"""Structure phase mixin for StoryService."""

import logging

from src.memory.story_state import StoryState
from src.memory.world_database import WorldDatabase
from src.utils.validation import (
    validate_not_none,
    validate_type,
)

from ._base import StoryServiceBase

logger = logging.getLogger(__name__)


class StructureMixin(StoryServiceBase):
    """Mixin providing structure/outline generation functionality."""

    def build_structure(self, state: StoryState, world_db: WorldDatabase) -> StoryState:
        """Build the story structure and extract entities to world database.

        Args:
            state: The story state with completed brief.
            world_db: WorldDatabase to populate with extracted entities.

        Returns:
            Updated StoryState with structure.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_not_none(world_db, "world_db")
        validate_type(world_db, "world_db", WorldDatabase)
        logger.debug(f"build_structure called: project_id={state.id}")
        if not state.brief:
            error_msg = "Cannot build structure - no brief exists."
            logger.error(f"build_structure failed for project {state.id}: {error_msg}")
            raise ValueError(error_msg)

        try:
            orchestrator = self._get_orchestrator(state)
            orchestrator.build_story_structure()
            self._sync_state(orchestrator, state)

            # Extract entities to world database
            self._extract_entities_to_world(state, world_db)

            logger.info(
                f"Story structure built for project {state.id}: {len(state.characters)} characters, {len(state.chapters)} chapters"
            )
            return state
        except Exception as e:
            logger.error(f"Failed to build structure for project {state.id}: {e}", exc_info=True)
            raise

    def _extract_entities_to_world(self, state: StoryState, world_db: WorldDatabase) -> None:
        """Extract characters and locations from story state to world database.

        Args:
            state: Story state with characters.
            world_db: WorldDatabase to populate.
        """
        # Extract characters
        for char in state.characters:
            # Check if already exists
            existing = world_db.search_entities(char.name, entity_type="character")
            if existing:
                continue

            attributes = {
                "role": char.role,
                "personality_traits": char.personality_traits,
                "goals": char.goals,
                "arc_notes": char.arc_notes,
            }

            entity_id = world_db.add_entity(
                entity_type="character",
                name=char.name,
                description=char.description,
                attributes=attributes,
            )

            # Add relationships from character data
            for related_name, relationship in char.relationships.items():
                # Find the related character
                related_entities = world_db.search_entities(related_name, entity_type="character")
                if related_entities:
                    world_db.add_relationship(
                        source_id=entity_id,
                        target_id=related_entities[0].id,
                        relation_type=relationship,
                    )

        logger.info(f"Extracted {len(state.characters)} characters to world database")

    def generate_outline_variations(
        self,
        state: StoryState,
        count: int = 3,
    ) -> list:
        """Generate multiple variations of the story outline.

        Args:
            state: Story state with completed brief.
            count: Number of variations to generate (3-5).

        Returns:
            List of OutlineVariation objects.
        """
        if not state.brief:
            raise ValueError("Cannot generate variations - no brief exists.")

        logger.info(f"Generating {count} outline variations for story {state.id}")

        # Get orchestrator and architect
        orchestrator = self._get_orchestrator(state)
        architect = orchestrator.architect

        # Generate variations
        variations = architect.generate_outline_variations(state, count=count)

        # Add to state
        for variation in variations:
            state.add_outline_variation(variation)

        logger.info(f"Generated {len(variations)} outline variations")
        return variations

    def select_variation(self, state: StoryState, variation_id: str) -> bool:
        """Select an outline variation as the canonical structure.

        Args:
            state: Story state.
            variation_id: ID of the variation to select.

        Returns:
            True if successful, False otherwise.
        """
        success = state.select_variation_as_canonical(variation_id)
        if success:
            logger.info(f"Selected variation {variation_id} as canonical for {state.id}")
        return success

    def rate_variation(
        self,
        state: StoryState,
        variation_id: str,
        rating: int,
        notes: str = "",
    ) -> bool:
        """Rate an outline variation.

        Args:
            state: Story state.
            variation_id: ID of the variation to rate.
            rating: Rating from 0-5.
            notes: Optional user notes.

        Returns:
            True if successful, False if variation not found.
        """
        variation = state.get_variation_by_id(variation_id)
        if not variation:
            return False

        variation.user_rating = max(0, min(5, rating))
        if notes:
            variation.user_notes = notes

        logger.debug(f"Rated variation {variation_id}: {rating}/5")
        return True

    def toggle_variation_favorite(self, state: StoryState, variation_id: str) -> bool:
        """Toggle favorite status on a variation.

        Args:
            state: Story state.
            variation_id: ID of the variation.

        Returns:
            True if successful, False if variation not found.
        """
        variation = state.get_variation_by_id(variation_id)
        if not variation:
            return False

        variation.is_favorite = not variation.is_favorite
        logger.debug(f"Toggled favorite for variation {variation_id}: {variation.is_favorite}")
        return True

    def create_merged_variation(
        self,
        state: StoryState,
        name: str,
        source_elements: dict[str, list[str]],
    ):
        """Create a merged variation from selected elements.

        Args:
            state: Story state.
            name: Name for merged variation.
            source_elements: Dict mapping variation_id to element types.

        Returns:
            The new merged OutlineVariation.
        """
        merged = state.create_merged_variation(name, source_elements)
        logger.info(f"Created merged variation: {name}")
        return merged

    def get_outline(self, state: StoryState) -> str:
        """Get a formatted story outline.

        Args:
            state: The story state.

        Returns:
            Human-readable outline string.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_outline_summary()
