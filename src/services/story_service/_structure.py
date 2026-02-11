"""Structure phase functions for StoryService."""

import logging
from typing import TYPE_CHECKING

from src.memory.story_state import StoryState

if TYPE_CHECKING:
    from src.services.story_service import StoryService

logger = logging.getLogger(__name__)


def generate_outline_variations(
    svc: StoryService,
    state: StoryState,
    count: int = 3,
) -> list:
    """Generate multiple variations of the story outline.

    Parameters:
        svc: The StoryService instance.
        state: Story state with completed brief.
        count: Number of variations to generate (3-5).

    Returns:
        List of OutlineVariation objects.
    """
    if not state.brief:
        raise ValueError("Cannot generate variations - no brief exists.")

    logger.info(f"Generating {count} outline variations for story {state.id}")

    # Get orchestrator and architect
    orchestrator = svc._get_orchestrator(state)
    architect = orchestrator.architect

    # Generate variations
    variations = architect.generate_outline_variations(state, count=count)

    # Add to state
    for variation in variations:
        state.add_outline_variation(variation)

    logger.info(f"Generated {len(variations)} outline variations")
    return variations


def select_variation(svc: StoryService, state: StoryState, variation_id: str) -> bool:
    """Select an outline variation as the canonical structure.

    Parameters:
        svc: The StoryService instance (unused but kept for consistency).
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
    svc: StoryService,
    state: StoryState,
    variation_id: str,
    rating: int,
    notes: str = "",
) -> bool:
    """Rate an outline variation.

    Parameters:
        svc: The StoryService instance (unused but kept for consistency).
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


def toggle_variation_favorite(svc: StoryService, state: StoryState, variation_id: str) -> bool:
    """Toggle favorite status on a variation.

    Parameters:
        svc: The StoryService instance (unused but kept for consistency).
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
    svc: StoryService,
    state: StoryState,
    name: str,
    source_elements: dict[str, list[str]],
):
    """Create a merged variation from selected elements.

    Parameters:
        svc: The StoryService instance (unused but kept for consistency).
        state: Story state.
        name: Name for merged variation.
        source_elements: Dict mapping variation_id to element types.

    Returns:
        The new merged OutlineVariation.
    """
    merged = state.create_merged_variation(name, source_elements)
    logger.info(f"Created merged variation: {name}")
    return merged
