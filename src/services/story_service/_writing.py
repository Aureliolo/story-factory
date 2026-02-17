"""Writing phase functions for StoryService."""

import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from src.memory.story_state import StoryState
from src.services.orchestrator import WorkflowEvent
from src.utils.validation import (
    validate_not_empty,
    validate_not_none,
    validate_positive,
    validate_type,
)

if TYPE_CHECKING:
    from src.memory.world_database import WorldDatabase
    from src.services.story_service import StoryService

logger = logging.getLogger(__name__)


def write_chapter(
    svc: StoryService,
    state: StoryState,
    chapter_num: int,
    feedback: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    world_db: WorldDatabase | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Write a single chapter with streaming events.

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        chapter_num: Chapter number to write.
        feedback: Optional feedback to incorporate.
        cancel_check: Optional callable that returns True if cancellation is requested.
        world_db: Optional world database for RAG context retrieval.

    Yields:
        WorkflowEvent objects for progress updates.

    Returns:
        The completed chapter content.

    Raises:
        GenerationCancelled: If cancellation is requested.
    """
    from src.services.story_service import GenerationCancelled

    validate_not_none(state, "state")
    validate_type(state, "state", StoryState)
    validate_positive(chapter_num, "chapter_num")
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    orchestrator.world_db = world_db

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

    svc._sync_state(orchestrator, state)
    return content


def write_all_chapters(
    svc: StoryService,
    state: StoryState,
    cancel_check: Callable[[], bool] | None = None,
    world_db: WorldDatabase | None = None,
) -> Generator[WorkflowEvent]:
    """Stream the generation of every chapter, yielding workflow events as progress is produced.

    Parameters:
        svc: The StoryService instance.
        state: The story state to write into.
        cancel_check: Optional callable invoked between events; return True to request cancellation.
        world_db: Optional world database for RAG context retrieval.

    Yields:
        WorkflowEvent: Progress events emitted during chapter generation.

    Raises:
        GenerationCancelled: If cancel_check returns True.
    """
    from src.services.story_service import GenerationCancelled

    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    orchestrator.world_db = world_db

    for event in orchestrator.write_all_chapters():
        # Check for cancellation
        if cancel_check and cancel_check():
            logger.info("Write all chapters cancelled by user")
            raise GenerationCancelled("Write all chapters cancelled")

        yield event

    svc._sync_state(orchestrator, state)

    # Trigger project completion learning
    svc._on_story_complete(state)


def write_short_story(
    svc: StoryService,
    state: StoryState,
    world_db: WorldDatabase | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Generate a single-chapter short story while streaming progress events.

    Parameters:
        svc: The StoryService instance.
        state: Story state to use and update during generation.
        world_db: Optional world database for RAG context retrieval.

    Yields:
        WorkflowEvent: Progress events emitted by the orchestrator during generation.

    Returns:
        The completed story content from chapter 1, or an empty string if no chapter was produced.
    """
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    orchestrator.world_db = world_db

    content = ""
    for event in orchestrator.write_short_story():
        yield event
        if event.event_type == "agent_complete" and event.agent_name == "System":
            if state.chapters:
                content = state.chapters[0].content

    svc._sync_state(orchestrator, state)
    return content


def regenerate_chapter_with_feedback(
    svc: StoryService,
    state: StoryState,
    chapter_num: int,
    feedback: str,
    cancel_check: Callable[[], bool] | None = None,
    world_db: WorldDatabase | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Regenerate a chapter incorporating user feedback.

    This method:
    1. Saves the current chapter content as a version
    2. Regenerates the chapter with the provided feedback
    3. Saves the new content as the current version

    Parameters:
        svc: The StoryService instance.
        state: The story state.
        chapter_num: Chapter number to regenerate.
        feedback: User feedback to incorporate into the regeneration.
        cancel_check: Optional callable that returns True if cancellation is requested.
        world_db: Optional world database for RAG context retrieval.

    Yields:
        WorkflowEvent objects for progress updates.

    Returns:
        The regenerated chapter content.

    Raises:
        GenerationCancelled: If cancellation is requested.
        ValueError: If chapter not found or no existing content.
    """
    from src.services.story_service import GenerationCancelled

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
    orchestrator = svc._get_orchestrator(state)
    orchestrator.story_state = state
    orchestrator.world_db = world_db

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

    svc._sync_state(orchestrator, state)
    logger.info(f"Chapter {chapter_num} regenerated successfully")
    return content
