"""Writing phase mixin for StoryService."""

import logging
from collections.abc import Callable, Generator
from typing import Any

from src.memory.story_state import StoryState
from src.services.orchestrator import WorkflowEvent
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_positive,
    validate_type,
)

from ._base import GenerationCancelled, StoryServiceBase

logger = logging.getLogger(__name__)


class WritingMixin(StoryServiceBase):
    """Mixin providing writing phase functionality."""

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
        """
        Stream the generation of every chapter, yielding workflow events as progress is produced.

        Parameters:
            state (StoryState): The story state to write into; orchestrator state will be synchronized back to this object.
            cancel_check (Callable[[], bool] | None): Optional callable invoked between events; return `True` to request cancellation.

        Yields:
            WorkflowEvent: Progress events emitted during chapter generation.

        Raises:
            GenerationCancelled: If `cancel_check` returns `True` indicating the user requested cancellation.
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

        # Trigger project completion learning
        self._on_story_complete(state)

    def write_short_story(self, state: StoryState) -> Generator[WorkflowEvent, None, str]:
        """
        Generate a single-chapter short story while streaming progress events.

        Parameters:
            state (StoryState): Story state to use and update during generation.

        Yields:
            WorkflowEvent: Progress events emitted by the orchestrator during generation.

        Returns:
            str: The completed story content from chapter 1, or an empty string if no chapter was produced.
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
