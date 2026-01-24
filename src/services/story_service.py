"""Story service - handles story generation workflow."""

import logging
from collections import OrderedDict
from collections.abc import Callable, Generator
from typing import Any

from src.agents.continuity import ContinuityIssue
from src.memory.story_state import Character, StoryBrief, StoryState
from src.memory.world_database import WorldDatabase
from src.services.orchestrator import StoryOrchestrator, WorkflowEvent
from src.settings import Settings
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_positive,
    validate_type,
)

logger = logging.getLogger(__name__)


class StoryService:
    """Story generation workflow service.

    This service wraps the StoryOrchestrator to provide a clean interface
    for the UI layer. It handles interview, structure building, and
    chapter writing workflows.
    """

    def __init__(self, settings: Settings):
        """Initialize story service.

        Args:
            settings: Application settings.
        """
        validate_not_none(settings, "settings")
        validate_type(settings, "settings", Settings)
        logger.debug("Initializing StoryService")
        self.settings = settings
        # Use OrderedDict for LRU cache behavior
        self._orchestrators: OrderedDict[str, StoryOrchestrator] = OrderedDict()
        logger.debug("StoryService initialized successfully")

    def _get_orchestrator(self, state: StoryState) -> StoryOrchestrator:
        """Get or create an orchestrator for a story state.

        Each story gets its own orchestrator instance to maintain
        conversation history and agent state. Uses LRU eviction to
        prevent unbounded memory growth.

        Args:
            state: The story state.

        Returns:
            StoryOrchestrator for this story.
        """
        if state.id in self._orchestrators:
            # Move to end (most recently used)
            self._orchestrators.move_to_end(state.id)
            return self._orchestrators[state.id]

        # Create new orchestrator
        orchestrator = StoryOrchestrator(settings=self.settings)
        orchestrator.story_state = state
        self._orchestrators[state.id] = orchestrator

        # Evict oldest if over capacity
        if len(self._orchestrators) > self.settings.orchestrator_cache_size:
            evicted_id, _ = self._orchestrators.popitem(last=False)
            logger.debug(f"Evicted orchestrator {evicted_id} from cache (LRU)")

        return orchestrator

    def _sync_state(self, orchestrator: StoryOrchestrator, state: StoryState) -> None:
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

    # ========== INTERVIEW PHASE ==========

    def start_interview(self, state: StoryState) -> str:
        """Start the interview process.

        Args:
            state: The story state.

        Returns:
            Initial interview questions from the interviewer agent.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"start_interview called: project_id={state.id}")
        try:
            orchestrator = self._get_orchestrator(state)
            questions = orchestrator.start_interview()

            # Store in interview history
            state.interview_history.append(
                {
                    "role": "assistant",
                    "content": questions,
                }
            )

            logger.info(f"Interview started for project {state.id}")
            return questions
        except Exception as e:
            logger.error(f"Failed to start interview for project {state.id}: {e}", exc_info=True)
            raise

    def process_interview(self, state: StoryState, user_message: str) -> tuple[str, bool]:
        """Process a user response in the interview.

        Args:
            state: The story state.
            user_message: User's response to interview questions.

        Returns:
            Tuple of (response_text, is_complete).
            is_complete is True when the brief has been generated.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_not_empty(user_message, "user_message")
        logger.debug(
            f"process_interview called: project_id={state.id}, message_length={len(user_message)}"
        )
        try:
            orchestrator = self._get_orchestrator(state)

            # Store user message in history
            state.interview_history.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )

            response, is_complete = orchestrator.process_interview_response(user_message)

            # Store assistant response
            state.interview_history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            # Sync state back
            self._sync_state(orchestrator, state)

            if is_complete:
                logger.info(f"Interview completed for project {state.id}")
            else:
                logger.debug(f"Interview in progress for project {state.id}")

            return response, is_complete
        except Exception as e:
            logger.error(f"Failed to process interview for project {state.id}: {e}", exc_info=True)
            raise

    def finalize_interview(self, state: StoryState) -> StoryBrief:
        """Force finalize the interview with current information.

        Args:
            state: The story state.

        Returns:
            The generated StoryBrief.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"finalize_interview called: project_id={state.id}")
        try:
            orchestrator = self._get_orchestrator(state)
            brief = orchestrator.finalize_interview()
            self._sync_state(orchestrator, state)
            logger.info(
                f"Interview finalized for project {state.id}: genre={brief.genre}, tone={brief.tone}"
            )
            return brief
        except Exception as e:
            logger.error(f"Failed to finalize interview for project {state.id}: {e}", exc_info=True)
            raise

    def continue_interview(self, state: StoryState, additional_info: str) -> str:
        """Continue an already-completed interview with additional information.

        This allows users to add clarifications or changes after the initial
        interview is complete. The interviewer should be apprehensive of big
        changes but allow small adjustments.

        Args:
            state: The story state (must have a brief already).
            additional_info: Additional information or changes from the user.

        Returns:
            Response from the interviewer acknowledging changes.
        """
        if not state.brief:
            raise ValueError("Cannot continue interview - no brief exists yet.")

        orchestrator = self._get_orchestrator(state)

        # Store user message
        state.interview_history.append(
            {
                "role": "user",
                "content": additional_info,
            }
        )

        # Process as a continuation
        response, _ = orchestrator.process_interview_response(additional_info)

        # Store response
        state.interview_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        self._sync_state(orchestrator, state)
        return response

    # ========== STRUCTURE PHASE ==========

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

    # ========== WRITING PHASE ==========

    def write_chapter(
        self,
        state: StoryState,
        chapter_num: int,
        feedback: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Write a single chapter with streaming events.

        Args:
            state: The story state.
            chapter_num: Chapter number to write.
            feedback: Optional feedback to incorporate.
            cancel_check: Optional callable that returns True if cancellation is requested.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The completed chapter content.

        Raises:
            GenerationCancelled: If cancellation is requested.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_positive(chapter_num, "chapter_num")
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        # Write the chapter
        content = ""
        for event in orchestrator.write_chapter(chapter_num):
            # Check for cancellation
            if cancel_check and cancel_check():
                logger.info(f"Chapter {chapter_num} generation cancelled by user")
                raise GenerationCancelled(
                    f"Chapter {chapter_num} generation cancelled", chapter_num=chapter_num
                )

            yield event
            # The generator returns the content at the end
            if event.event_type == "agent_complete" and event.agent_name == "System":
                # Get the chapter content
                chapter = next((c for c in state.chapters if c.number == chapter_num), None)
                if chapter:
                    content = chapter.content

        self._sync_state(orchestrator, state)
        return content

    def write_all_chapters(
        self, state: StoryState, cancel_check: Callable[[], bool] | None = None
    ) -> Generator[WorkflowEvent]:
        """Write all chapters with streaming events.

        Args:
            state: The story state.
            cancel_check: Optional callable that returns True if cancellation is requested.

        Yields:
            WorkflowEvent objects for progress updates.

        Raises:
            GenerationCancelled: If cancellation is requested.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        for event in orchestrator.write_all_chapters():
            # Check for cancellation
            if cancel_check and cancel_check():
                logger.info("Write all chapters cancelled by user")
                raise GenerationCancelled("Write all chapters cancelled")

            yield event

        self._sync_state(orchestrator, state)

    def write_short_story(self, state: StoryState) -> Generator[WorkflowEvent, None, str]:
        """Write a short story (single chapter).

        Args:
            state: The story state.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The completed story content.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        content = ""
        for event in orchestrator.write_short_story():
            yield event
            if event.event_type == "agent_complete" and event.agent_name == "System":
                if state.chapters:
                    content = state.chapters[0].content

        self._sync_state(orchestrator, state)
        return content

    def get_full_story(self, state: StoryState) -> str:
        """Get the complete story text.

        Args:
            state: The story state.

        Returns:
            Full story text with all chapters.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state
        return orchestrator.get_full_story()

    def get_chapter_content(self, state: StoryState, chapter_num: int) -> str | None:
        """Get content of a specific chapter.

        Args:
            state: The story state.
            chapter_num: Chapter number.

        Returns:
            Chapter content or None if not found/written.
        """
        chapter = next((c for c in state.chapters if c.number == chapter_num), None)
        return chapter.content if chapter else None

    def get_statistics(self, state: StoryState) -> dict[str, Any]:
        """Get story statistics.

        Args:
            state: The story state.

        Returns:
            Dictionary with statistics including word count, chapters, etc.
        """
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
    ) -> Generator[WorkflowEvent, None, str]:
        """Regenerate a chapter incorporating user feedback.

        This method:
        1. Saves the current chapter content as a version
        2. Regenerates the chapter with the provided feedback
        3. Saves the new content as the current version

        Args:
            state: The story state.
            chapter_num: Chapter number to regenerate.
            feedback: User feedback to incorporate into the regeneration.
            cancel_check: Optional callable that returns True if cancellation is requested.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The regenerated chapter content.

        Raises:
            GenerationCancelled: If cancellation is requested.
            ValueError: If chapter not found or no existing content.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_positive(chapter_num, "chapter_num")
        validate_not_empty(feedback, "feedback")

        logger.info(f"Regenerating chapter {chapter_num} with feedback: {feedback[:100]}...")

        # Find the chapter
        chapter = next((c for c in state.chapters if c.number == chapter_num), None)
        if not chapter:
            raise ValueError(f"Chapter {chapter_num} not found")

        if not chapter.content:
            raise ValueError(f"Chapter {chapter_num} has no content to regenerate. Write it first.")

        # Save current version before regenerating (without feedback - feedback applies to the NEW version)
        version_id = chapter.save_current_as_version(feedback="")
        logger.debug(f"Saved current content as version {version_id} before regeneration")

        # Get orchestrator and regenerate with feedback
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        # Write the chapter with feedback
        content = ""
        for event in orchestrator.write_chapter(chapter_num, feedback=feedback):
            # Check for cancellation
            if cancel_check and cancel_check():
                logger.info(f"Chapter {chapter_num} regeneration cancelled by user")
                # Rollback to previous version
                chapter.rollback_to_version(version_id)
                raise GenerationCancelled(
                    f"Chapter {chapter_num} regeneration cancelled", chapter_num=chapter_num
                )

            yield event
            # The generator returns the content at the end
            if event.event_type == "agent_complete" and event.agent_name == "System":
                content = chapter.content

        # Save the new content as a version with the feedback that was used to create it
        chapter.save_current_as_version(feedback=feedback)

        self._sync_state(orchestrator, state)
        logger.info(f"Chapter {chapter_num} regenerated successfully")
        return content

    # ========== CONTINUATION & EDITING ==========

    def continue_chapter(
        self,
        state: StoryState,
        chapter_num: int,
        direction: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Continue writing a chapter from where it left off.

        Args:
            state: The story state.
            chapter_num: Chapter number to continue.
            direction: Optional direction for where to take the scene.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The continuation text.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        continuation = ""
        for event in orchestrator.continue_chapter(chapter_num, direction):
            yield event
            if event.event_type == "agent_complete" and event.agent_name == "Writer":
                # Get updated chapter content
                chapter = next((c for c in state.chapters if c.number == chapter_num), None)
                if chapter:
                    continuation = event.data.get("continuation", "") if event.data else ""

        self._sync_state(orchestrator, state)
        return continuation

    def edit_passage(
        self,
        state: StoryState,
        text: str,
        focus: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Edit a specific passage with optional focus area.

        Args:
            state: The story state.
            text: The text passage to edit.
            focus: Optional focus area (e.g., "dialogue", "pacing").

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            The edited passage.
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        edited = ""
        for event in orchestrator.edit_passage(text, focus):
            yield event
            if event.event_type == "agent_complete":
                # The edited text is returned from the generator
                pass

        # The generator returns the edited text at the end
        # We need to call it fully and capture the return
        return edited

    def get_edit_suggestions(
        self,
        state: StoryState,
        text: str,
    ) -> Generator[WorkflowEvent, None, str]:
        """Get editing suggestions without making changes.

        Args:
            state: The story state.
            text: The text to review.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            Suggestions for improving the text.
        """
        orchestrator = self._get_orchestrator(state)

        suggestions = ""
        for event in orchestrator.get_edit_suggestions(text):
            yield event
            if event.event_type == "agent_complete" and event.data:
                suggestions = event.data.get("suggestions", "")

        return suggestions

    def review_full_story(
        self,
        state: StoryState,
    ) -> Generator[WorkflowEvent, None, list[ContinuityIssue]]:
        """Perform a full story continuity review.

        Args:
            state: The story state.

        Yields:
            WorkflowEvent objects for progress updates.

        Returns:
            List of ContinuityIssue objects (as dicts).
        """
        orchestrator = self._get_orchestrator(state)
        orchestrator.story_state = state

        issues = []
        for event in orchestrator.review_full_story():
            yield event
            if event.event_type == "agent_complete" and event.data:
                issues = event.data.get("issues", [])

        return issues

    # ========== FEEDBACK & REVIEWS ==========

    def add_review(
        self, state: StoryState, review_type: str, content: str, chapter_num: int | None = None
    ) -> None:
        """Add a review or note to the story.

        Args:
            state: The story state.
            review_type: Type of review ("user_note", "ai_suggestion", "marked_for_review").
            content: The review content.
            chapter_num: Optional chapter number this review applies to.
        """
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
        """Get reviews for a story or specific chapter.

        Args:
            state: The story state.
            chapter_num: Optional chapter number to filter by.

        Returns:
            List of review dictionaries.
        """
        if chapter_num is None:
            return state.reviews
        return [r for r in state.reviews if r.get("chapter") == chapter_num]

    # ========== TITLE GENERATION ==========

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

    # ========== WORLD GENERATION ==========

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

    # ========== CLEANUP ==========

    def cleanup_orchestrator(self, state: StoryState) -> None:
        """Clean up orchestrator for a story (free memory).

        Args:
            state: The story state.
        """
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

        Args:
            message: Cancellation message
            chapter_num: Chapter number being generated (optional)
            progress_state: Progress information at cancellation (optional)
        """
        super().__init__(message)
        self.chapter_num = chapter_num
        self.progress_state = progress_state or {}
