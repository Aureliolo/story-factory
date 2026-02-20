"""Editing and review functions for StoryOrchestrator."""

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

from src.agents import ResponseValidationError
from src.agents.continuity import ContinuityIssue
from src.services.orchestrator._writing import (
    _combine_contexts,
    _retrieve_temporal_context,
    _retrieve_world_context,
    _warn_if_context_missing,
)
from src.utils.exceptions import ExportError

if TYPE_CHECKING:
    from . import StoryOrchestrator, WorkflowEvent

logger = logging.getLogger(__name__)


def continue_chapter(
    orc: StoryOrchestrator,
    chapter_number: int,
    direction: str | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Continue writing a chapter from where it left off.

    Args:
        orc: StoryOrchestrator instance.
        chapter_number: The chapter to continue.
        direction: Optional direction for where to take the scene.

    Yields:
        WorkflowEvent objects for UI progress updates.

    Returns:
        The continued content.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    chapter = next((c for c in orc.story_state.chapters if c.number == chapter_number), None)
    if not chapter:
        raise ValueError(f"Chapter {chapter_number} not found.")

    if not chapter.content:
        raise ValueError(f"Chapter {chapter_number} has no content to continue from.")

    orc._emit(
        "agent_start",
        "Writer",
        f"Continuing Chapter {chapter_number}...",
        {"direction": direction} if direction else None,
    )
    yield orc.events[-1]

    # Use WriterAgent.continue_scene() to continue from current text
    continuation = orc.writer.continue_scene(
        orc.story_state,
        chapter.content,
        direction=direction,
    )

    # Validate continuation
    try:
        orc._validate_response(continuation, f"Chapter {chapter_number} continuation")
    except ResponseValidationError as e:
        logger.warning(f"Validation warning for continuation: {e}")

    # Append continuation to chapter content
    chapter.content = chapter.content + "\n\n" + continuation
    chapter.word_count = len(chapter.content.split())

    # Auto-save
    try:
        orc.save_story()
        logger.debug(f"Auto-saved after continuing chapter {chapter_number}")
    except (OSError, PermissionError) as e:
        error_msg = f"Failed to save continued chapter {chapter_number}: {e}"
        logger.error(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e
    except (ValueError, TypeError) as e:
        error_msg = f"Failed to serialize continued chapter {chapter_number}: {e}"
        logger.error(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error saving continued chapter {chapter_number}: {e}"
        logger.exception(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e

    orc._emit(
        "agent_complete",
        "Writer",
        f"Chapter {chapter_number} continued (+{len(continuation.split())} words)",
    )
    yield orc.events[-1]

    return continuation


def edit_passage(
    orc: StoryOrchestrator,
    text: str,
    focus: str | None = None,
) -> Generator[WorkflowEvent, None, str]:
    """Edit a specific passage with optional focus area.

    Args:
        orc: StoryOrchestrator instance.
        text: The text passage to edit.
        focus: Optional focus area (e.g., "dialogue", "pacing", "description").

    Yields:
        WorkflowEvent objects for UI progress updates.

    Returns:
        The edited passage.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    brief = orc.story_state.brief
    language = brief.language if brief else "English"

    focus_msg = f" (focus: {focus})" if focus else ""
    orc._emit("agent_start", "Editor", f"Editing passage{focus_msg}...")
    yield orc.events[-1]

    # Use EditorAgent.edit_passage() for targeted editing
    edited = orc.editor.edit_passage(text, focus=focus, language=language)

    orc._emit(
        "agent_complete",
        "Editor",
        f"Passage edited ({len(edited.split())} words)",
    )
    yield orc.events[-1]

    return edited


def get_edit_suggestions(
    orc: StoryOrchestrator,
    text: str,
) -> Generator[WorkflowEvent, None, str]:
    """Get editing suggestions without making changes.

    Args:
        orc: StoryOrchestrator instance.
        text: The text to review.

    Yields:
        WorkflowEvent objects for UI progress updates.

    Returns:
        Suggestions for improving the text.
    """
    orc._emit("agent_start", "Editor", "Analyzing text for suggestions...")
    yield orc.events[-1]

    # Use EditorAgent.get_edit_suggestions() for review mode
    suggestions = orc.editor.get_edit_suggestions(text)

    orc._emit(
        "agent_complete",
        "Editor",
        "Edit suggestions ready",
        {"suggestions": suggestions},
    )
    yield orc.events[-1]

    return suggestions


def review_full_story(
    orc: StoryOrchestrator,
) -> Generator[WorkflowEvent, None, list[ContinuityIssue]]:
    """Perform a full story continuity review.

    Args:
        orc: StoryOrchestrator instance.

    Yields:
        WorkflowEvent objects for UI progress updates.

    Returns:
        List of ContinuityIssue objects.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    orc._emit("agent_start", "Continuity", "Reviewing complete story...")
    yield orc.events[-1]

    world_context = _retrieve_world_context(orc, "Full story review for continuity")
    temporal_context = _retrieve_temporal_context(orc)
    combined_context = _combine_contexts(world_context, temporal_context)
    _warn_if_context_missing(orc, combined_context, "Full story review")
    issues = orc.continuity.check_full_story(orc.story_state, world_context=combined_context)

    if issues:
        feedback = orc.continuity.format_revision_feedback(issues)
        critical_count = sum(1 for i in issues if i.severity == "critical")
        moderate_count = sum(1 for i in issues if i.severity == "moderate")
        minor_count = sum(1 for i in issues if i.severity == "minor")

        orc._emit(
            "agent_complete",
            "Continuity",
            f"Found {len(issues)} issues: "
            f"{critical_count} critical, {moderate_count} moderate, {minor_count} minor",
            {"issues": [vars(i) for i in issues], "feedback": feedback},
        )
    else:
        orc._emit(
            "agent_complete",
            "Continuity",
            "No continuity issues found!",
        )

    yield orc.events[-1]
    return issues
