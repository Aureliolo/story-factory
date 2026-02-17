"""Story service - handles story generation workflow.

This package provides StoryService and GenerationCancelled, split from a single
module into logical sub-modules for maintainability.

Sub-modules:
    _interview  - Interview phase (start, process, finalize, continue)
    _structure  - Outline variations, selection, rating, merging
    _writing    - Chapter writing, regeneration, short story
    _editing    - Continuation, editing, review, world generation
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.world_database import WorldDatabase
    from src.services.context_retrieval_service import ContextRetrievalService
    from src.services.model_mode_service import ModelModeService

from src.agents.continuity import ContinuityIssue
from src.memory.story_state import Character, StoryBrief, StoryState
from src.services.orchestrator import StoryOrchestrator, WorkflowEvent
from src.services.story_service import _editing, _interview, _structure, _writing
from src.settings import Settings
from src.utils.validation import validate_not_none, validate_type

logger = logging.getLogger(__name__)

__all__ = ["GenerationCancelled", "StoryService"]


class StoryService:
    """Story generation workflow service.

    This service wraps the StoryOrchestrator to provide a clean interface
    for the UI layer. It handles interview, structure building, and
    chapter writing workflows.
    """

    def __init__(
        self,
        settings: Settings,
        mode_service: ModelModeService | None = None,
        context_retrieval: ContextRetrievalService | None = None,
    ):
        """Create a StoryService configured with application settings and an optional mode service.

        Parameters:
            settings (Settings): Application settings used to configure the service; must be a
                Settings instance.
            mode_service (ModelModeService | None): Optional service that provides adaptive
                learning hooks; stored on the instance for use by orchestrators.
            context_retrieval (ContextRetrievalService | None): Optional context retrieval service
                for RAG-based prompt enrichment in agent calls.
        """
        validate_not_none(settings, "settings")
        validate_type(settings, "settings", Settings)
        logger.debug("Initializing StoryService")
        self.settings = settings
        self.mode_service = mode_service  # For learning hooks
        self.context_retrieval = context_retrieval  # For RAG context injection
        # Use OrderedDict for LRU cache behavior
        self._orchestrators: OrderedDict[str, StoryOrchestrator] = OrderedDict()
        logger.debug("StoryService initialized successfully")

    def _get_orchestrator(self, state: StoryState) -> StoryOrchestrator:
        """Retrieve the StoryOrchestrator associated with the given StoryState, creating one if absent.

        Creates a new orchestrator tied to the provided state and inserts it into an LRU cache;
        evicts the least-recently-used orchestrator when the cache exceeds
        settings.orchestrator_cache_size.

        Parameters:
            state: The story state to obtain an orchestrator for.

        Returns:
            The orchestrator instance for the story.
        """
        if state.id in self._orchestrators:
            # Move to end (most recently used)
            self._orchestrators.move_to_end(state.id)
            return self._orchestrators[state.id]

        # Create new orchestrator with mode service for learning hooks
        orchestrator = StoryOrchestrator(
            settings=self.settings,
            mode_service=self.mode_service,
            context_retrieval=self.context_retrieval,
        )
        orchestrator.story_state = state
        self._orchestrators[state.id] = orchestrator

        # Evict oldest if over capacity
        if len(self._orchestrators) > self.settings.orchestrator_cache_size:
            evicted_id, _ = self._orchestrators.popitem(last=False)
            logger.debug(f"Evicted orchestrator {evicted_id} from cache (LRU)")

        return orchestrator

    def _sync_state(self, orchestrator: StoryOrchestrator, state: StoryState) -> None:
        """Sync orchestrator state back to the provided state object.

        Parameters:
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
        """Invoke learning hooks via the configured mode service when a story is completed.

        If a mode service is not configured, no action is taken. When configured, this method
        requests recommendations from the mode service and asks the mode service to handle them.
        It returns any recommendations that remain pending user approval.

        Parameters:
            state: The completed story state (used for logging/context).

        Returns:
            A list of pending recommendations if any exist, None otherwise.
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

        Parameters:
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

    # ========== INTERVIEW PHASE ==========

    def start_interview(self, state: StoryState) -> str:
        """Begin the interview for the given story."""
        return _interview.start_interview(self, state)

    def process_interview(self, state: StoryState, user_message: str) -> tuple[str, bool]:
        """Process a user response in the interview."""
        return _interview.process_interview(self, state, user_message)

    def finalize_interview(self, state: StoryState) -> StoryBrief:
        """Force finalize the interview with current information."""
        return _interview.finalize_interview(self, state)

    def continue_interview(self, state: StoryState, additional_info: str) -> str:
        """Continue an already-completed interview with additional information."""
        return _interview.continue_interview(self, state, additional_info)

    # ========== STRUCTURE PHASE ==========

    def generate_outline_variations(
        self,
        state: StoryState,
        count: int = 3,
    ) -> list:
        """Generate multiple variations of the story outline."""
        return _structure.generate_outline_variations(self, state, count)

    def select_variation(self, state: StoryState, variation_id: str) -> bool:
        """Select an outline variation as the canonical structure."""
        return _structure.select_variation(self, state, variation_id)

    def rate_variation(
        self,
        state: StoryState,
        variation_id: str,
        rating: int,
        notes: str = "",
    ) -> bool:
        """Rate an outline variation."""
        return _structure.rate_variation(self, state, variation_id, rating, notes)

    def toggle_variation_favorite(self, state: StoryState, variation_id: str) -> bool:
        """Toggle favorite status on a variation."""
        return _structure.toggle_variation_favorite(self, state, variation_id)

    def create_merged_variation(
        self,
        state: StoryState,
        name: str,
        source_elements: dict[str, list[str]],
    ):
        """Create a merged variation from selected elements."""
        return _structure.create_merged_variation(self, state, name, source_elements)

    def get_outline(self, state: StoryState) -> str:
        """Get a formatted story outline."""
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_outline_summary()

    # ========== WRITING PHASE ==========

    def write_chapter(
        self,
        state: StoryState,
        chapter_num: int,
        feedback: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        world_db: WorldDatabase | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Write a single chapter with streaming events."""
        return _writing.write_chapter(self, state, chapter_num, feedback, cancel_check, world_db)

    def write_all_chapters(
        self,
        state: StoryState,
        cancel_check: Callable[[], bool] | None = None,
        world_db: WorldDatabase | None = None,
    ) -> Generator[WorkflowEvent]:
        """Stream the generation of every chapter, yielding workflow events."""
        return _writing.write_all_chapters(self, state, cancel_check, world_db)

    def write_short_story(
        self, state: StoryState, world_db: WorldDatabase | None = None
    ) -> Generator[WorkflowEvent, None, str]:
        """Generate a single-chapter short story while streaming progress events."""
        return _writing.write_short_story(self, state, world_db)

    def get_full_story(self, state: StoryState) -> str:
        """Get the complete story text."""
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_full_story()

    def get_chapter_content(self, state: StoryState, chapter_num: int) -> str | None:
        """Get content of a specific chapter."""
        chapter = next((c for c in state.chapters if c.number == chapter_num), None)
        return chapter.content if chapter else None

    def get_statistics(self, state: StoryState) -> dict[str, Any]:
        """Get story statistics."""
        logger.debug(f"get_statistics called: project_id={state.id}")
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        stats = orchestrator.get_statistics()
        logger.debug(f"Statistics for project {state.id}: {stats}")
        return stats

    def regenerate_chapter_with_feedback(
        self,
        state: StoryState,
        chapter_num: int,
        feedback: str,
        cancel_check: Callable[[], bool] | None = None,
        world_db: WorldDatabase | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Regenerate a chapter incorporating user feedback."""
        return _writing.regenerate_chapter_with_feedback(
            self, state, chapter_num, feedback, cancel_check, world_db
        )

    # ========== CONTINUATION & EDITING ==========

    def continue_chapter(
        self,
        state: StoryState,
        chapter_num: int,
        direction: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Continue writing a chapter from where it left off."""
        return _editing.continue_chapter(self, state, chapter_num, direction)

    def edit_passage(
        self,
        state: StoryState,
        text: str,
        focus: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Edit a specific passage with optional focus area."""
        return _editing.edit_passage(self, state, text, focus)

    def get_edit_suggestions(
        self,
        state: StoryState,
        text: str,
    ) -> Generator[WorkflowEvent, None, str]:
        """Get editing suggestions without making changes."""
        return _editing.get_edit_suggestions(self, state, text)

    def review_full_story(
        self,
        state: StoryState,
    ) -> Generator[WorkflowEvent, None, list[ContinuityIssue]]:
        """Perform a full story continuity review."""
        return _editing.review_full_story(self, state)

    # ========== FEEDBACK & REVIEWS ==========

    def add_review(
        self, state: StoryState, review_type: str, content: str, chapter_num: int | None = None
    ) -> None:
        """Add a review or note to the story."""
        state.reviews.append(
            {
                "type": review_type,
                "content": content,
                "chapter": chapter_num,
                "timestamp": str(state.updated_at),
            }
        )

    def get_reviews(
        self, state: StoryState, chapter_num: int | None = None
    ) -> list[dict[str, Any]]:
        """Get reviews for a story or specific chapter."""
        if chapter_num is None:
            return state.reviews
        return [r for r in state.reviews if r.get("chapter") == chapter_num]

    # ========== TITLE GENERATION ==========

    def generate_title_suggestions(self, state: StoryState) -> list[str]:
        """Generate AI-powered title suggestions."""
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.generate_title_suggestions()

    # ========== WORLD GENERATION ==========

    def generate_more_characters(self, state: StoryState, count: int = 2) -> list[Character]:
        """Generate additional characters for the story."""
        return _editing.generate_more_characters(self, state, count)

    def generate_locations(self, state: StoryState, count: int = 3) -> list[Any]:
        """Generate locations for the story world."""
        return _editing.generate_locations(self, state, count)

    def generate_relationships(
        self,
        state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        count: int = 5,
    ) -> list[Any]:
        """Generate relationships between entities."""
        return _editing.generate_relationships(self, state, entity_names, existing_rels, count)

    def rebuild_world(self, state: StoryState) -> StoryState:
        """Rebuild the entire world from scratch."""
        return _editing.rebuild_world(self, state)

    # ========== CLEANUP ==========

    def cleanup_orchestrator(self, state: StoryState) -> None:
        """Clean up orchestrator for a story (free memory)."""
        logger.debug(f"cleanup_orchestrator called: project_id={state.id}")
        if state.id in self._orchestrators:
            del self._orchestrators[state.id]
            logger.debug(f"Cleaned up orchestrator for story {state.id}")


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

        Parameters:
            message: Cancellation message
            chapter_num: Chapter number being generated (optional)
            progress_state: Progress information at cancellation (optional)
        """
        super().__init__(message)
        self.chapter_num = chapter_num
        self.progress_state = progress_state or {}
