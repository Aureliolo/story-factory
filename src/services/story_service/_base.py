"""Base class and core functionality for StoryService."""

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

import src.services.orchestrator as _orchestrator
from src.memory.story_state import StoryState
from src.settings import Settings
from src.utils.validation import (
    validate_not_none,
    validate_type,
)

logger = logging.getLogger(__name__)


class GenerationCancelled(Exception):
    """Exception raised when generation is cancelled by user.

    Attributes:
        message: Cancellation message
        chapter_num: Chapter number being generated when cancelled (if applicable)
        progress_state: Optional dict with progress information at cancellation
    """

    def __init__(
        self,
        message: str = "Generation cancelled",
        chapter_num: int | None = None,
        progress_state: dict[str, Any] | None = None,
    ):
        """Initialize GenerationCancelled exception.

        Args:
            message: Cancellation message
            chapter_num: Chapter number being generated (optional)
            progress_state: Progress information at cancellation (optional)
        """
        super().__init__(message)
        self.chapter_num = chapter_num
        self.progress_state = progress_state or {}


class StoryServiceBase:
    """Base class for story service with orchestrator management."""

    def __init__(self, settings: Settings, mode_service: ModelModeService | None = None):
        """Create a StoryService configured with application settings and an optional mode service.

        Parameters:
            settings (Settings): Application settings used to configure the service; must be a Settings instance.
            mode_service (ModelModeService | None): Optional service that provides adaptive learning hooks; stored on the instance for use by orchestrators.
        """
        validate_not_none(settings, "settings")
        validate_type(settings, "settings", Settings)
        logger.debug("Initializing StoryService")
        self.settings = settings
        self.mode_service = mode_service  # For learning hooks
        # Use OrderedDict for LRU cache behavior
        self._orchestrators: OrderedDict[str, _orchestrator.StoryOrchestrator] = OrderedDict()
        logger.debug("StoryService initialized successfully")

    def _get_orchestrator(self, state: StoryState) -> _orchestrator.StoryOrchestrator:
        """
        Retrieve the _orchestrator.StoryOrchestrator associated with the given StoryState, creating and caching one if absent.

        Creates a new orchestrator tied to the provided state and inserts it into an LRU cache; evicts the least-recently-used orchestrator when the cache exceeds settings.orchestrator_cache_size.

        Parameters:
            state (StoryState): The story state to obtain an orchestrator for.

        Returns:
            _orchestrator.StoryOrchestrator: The orchestrator instance for the story.
        """
        if state.id in self._orchestrators:
            # Move to end (most recently used)
            self._orchestrators.move_to_end(state.id)
            return self._orchestrators[state.id]

        # Create new orchestrator with mode service for learning hooks
        orchestrator = _orchestrator.StoryOrchestrator(
            settings=self.settings,
            mode_service=self.mode_service,
        )
        orchestrator.story_state = state
        self._orchestrators[state.id] = orchestrator

        # Evict oldest if over capacity
        if len(self._orchestrators) > self.settings.orchestrator_cache_size:
            evicted_id, _ = self._orchestrators.popitem(last=False)
            logger.debug(f"Evicted orchestrator {evicted_id} from cache (LRU)")

        return orchestrator

    def _sync_state(self, orchestrator: _orchestrator.StoryOrchestrator, state: StoryState) -> None:
        """Sync orchestrator state back to the provided state object.

        Args:
            orchestrator: The orchestrator with potentially updated state.
            state: The state object to update.
        """
        if orchestrator.story_state:
            # Copy relevant fields back
            state.brief = orchestrator.story_state.brief
            state.characters = orchestrator.story_state.characters
            state.chapters = orchestrator.story_state.chapters
            state.plot_summary = orchestrator.story_state.plot_summary
            state.plot_points = orchestrator.story_state.plot_points
            state.world_description = orchestrator.story_state.world_description
            state.world_rules = orchestrator.story_state.world_rules
            state.established_facts = orchestrator.story_state.established_facts
            state.timeline = orchestrator.story_state.timeline
            state.current_chapter = orchestrator.story_state.current_chapter
            state.status = orchestrator.story_state.status

    def _on_story_complete(self, state: StoryState) -> list[Any] | None:
        """
        Invoke learning hooks via the configured mode service when a story is completed.

        If a mode service is not configured, no action is taken. When configured, this method requests recommendations from the mode service and asks the mode service to handle them. It returns any recommendations that remain pending user approval.

        Parameters:
            state (StoryState): The completed story state (used for logging/context).

        Returns:
            list[Any] | None: A list of pending recommendations if any exist, `None` otherwise.
        """
        if not self.mode_service:
            return None

        try:
            recommendations = self.mode_service.on_project_complete()
            if recommendations:
                logger.info(
                    f"Generated {len(recommendations)} learning recommendations "
                    f"for project {state.id}"
                )
                # Handle recommendations based on autonomy level
                pending = self.mode_service.handle_recommendations(recommendations)
                if pending:
                    logger.info(f"{len(pending)} recommendations pending user approval")
                return list(pending)  # Explicit cast for type checker
            logger.debug(f"No recommendations generated for project {state.id}")
            return None
        except Exception as e:
            logger.warning(
                f"Failed to process project completion learning for project {state.id}: {e}",
                exc_info=True,
            )
            return None

    def complete_project(self, state: StoryState) -> dict[str, Any]:
        """Mark a project as complete and trigger learning.

        Call this when the user explicitly marks a story as finished.
        This triggers the learning system to generate recommendations
        based on the project's generation data.

        Args:
            state: The story state to complete.

        Returns:
            Dictionary with completion info and any pending recommendations.

        Raises:
            TypeError: If state is not a StoryState.
            ValueError: If state is None.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.info(f"Completing project {state.id}")
        state.status = "complete"

        # Trigger learning
        recommendations = self._on_story_complete(state)

        return {
            "project_id": state.id,
            "status": "complete",
            "pending_recommendations": recommendations or [],
        }

    def cleanup_orchestrator(self, state: StoryState) -> None:
        """Clean up orchestrator for a story (free memory).

        Args:
            state: The story state.
        """
        logger.debug(f"cleanup_orchestrator called: project_id={state.id}")
        if state.id in self._orchestrators:
            del self._orchestrators[state.id]
            logger.debug(f"Cleaned up orchestrator for story {state.id}")
