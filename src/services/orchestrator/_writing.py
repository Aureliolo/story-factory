"""Writing phase functions for StoryOrchestrator."""

import logging
from collections.abc import Callable, Generator
from datetime import datetime
from typing import TYPE_CHECKING

from src.agents import ResponseValidationError
from src.memory.story_state import Chapter
from src.utils.exceptions import ExportError, GenerationCancelledError

if TYPE_CHECKING:
    from . import StoryOrchestrator, WorkflowEvent

logger = logging.getLogger(__name__)


def _retrieve_world_context(orc: StoryOrchestrator, task_description: str) -> str:
    """Retrieve RAG world context for prompt enrichment.

    Uses the orchestrator's context_retrieval service and world_db to perform
    vector-similarity-based retrieval. Returns empty string if RAG is not
    configured or retrieval fails.

    Args:
        orc: StoryOrchestrator instance with optional context_retrieval and world_db.
        task_description: Description of the current agent task for query embedding.

    Returns:
        Formatted world context string, or empty string if unavailable.
    """
    if not orc.context_retrieval or not orc.world_db or not orc.story_state:
        return ""

    try:
        retrieved = orc.context_retrieval.retrieve_context(
            task_description=task_description,
            world_db=orc.world_db,
            story_state=orc.story_state,
        )
        formatted = retrieved.format_for_prompt()
        if formatted:
            logger.debug(
                "Retrieved RAG context: %d items, ~%d tokens",
                len(retrieved.items),
                retrieved.total_tokens,
            )
        return formatted
    except Exception as e:
        logger.warning("RAG context retrieval failed (non-fatal): %s", e, exc_info=True)
        return ""


def _retrieve_temporal_context(orc: StoryOrchestrator) -> str:
    """Retrieve temporal context from timeline data for prompt enrichment.

    Uses the orchestrator's timeline service and world_db to build a readable
    temporal context string. Returns empty string if world_db is not set,
    temporal validation is disabled, timeline service is not configured,
    or retrieval fails.

    Args:
        orc: StoryOrchestrator instance with optional world_db and timeline.

    Returns:
        Formatted temporal context string, or empty string if unavailable.
    """
    if not orc.world_db:
        logger.debug("No world_db configured, skipping temporal context")
        return ""

    if not orc.settings.validate_temporal_consistency:
        logger.debug("Temporal validation disabled, skipping temporal context")
        return ""

    if not orc.timeline:
        logger.debug("No timeline service configured, skipping temporal context")
        return ""

    try:
        context = orc.timeline.build_temporal_context(orc.world_db)
        if context:
            logger.debug("Retrieved temporal context: %d chars", len(context))
        return context
    except GenerationCancelledError:
        raise
    except Exception as e:
        logger.warning("Temporal context retrieval failed (non-fatal): %s", e, exc_info=True)
        return ""


def _combine_contexts(world_context: str, temporal_context: str) -> str:
    """Merge RAG world context and temporal context into a single string.

    Order matters: world_context (RAG) must be first, temporal_context second.
    The combined string is injected into agent prompts with RAG data preceding
    temporal timeline data.

    Args:
        world_context: RAG-retrieved world context string (may be empty).
        temporal_context: Temporal timeline context string (may be empty).

    Returns:
        Combined context string, or empty string if both inputs are empty.
    """
    if not world_context and not temporal_context:
        return ""
    logger.debug(
        "Combining contexts: world=%d chars, temporal=%d chars",
        len(world_context),
        len(temporal_context),
    )
    if not temporal_context:
        return world_context
    result = f"{world_context}\n\n{temporal_context}" if world_context else temporal_context
    logger.debug("Combined context: %d chars", len(result))
    return result


def _warn_if_context_missing(orc: StoryOrchestrator, combined_context: str, task: str) -> None:
    """Log a warning if context sources are enabled but returned no data.

    Checks both RAG and temporal context source configuration against the
    actual combined context result. Only warns when sources are configured
    but produced empty output, indicating a retrieval failure.

    Args:
        orc: StoryOrchestrator instance.
        combined_context: The combined context string (may be empty).
        task: Human-readable label for the warning message (e.g. "Final story review").
    """
    if combined_context:
        return
    rag_enabled = bool(
        orc.context_retrieval
        and orc.world_db
        and orc.story_state
        and orc.settings.rag_context_enabled
    )
    temporal_enabled = bool(
        orc.world_db and orc.settings.validate_temporal_consistency and orc.timeline
    )
    if rag_enabled or temporal_enabled:
        logger.warning(
            "%s proceeding without any world/temporal context "
            "despite context sources being configured",
            task,
        )


def write_short_story(orc: StoryOrchestrator) -> Generator[WorkflowEvent, None, str]:
    """Write a short story with revision loop.

    Args:
        orc: StoryOrchestrator instance.

    Yields:
        WorkflowEvent objects for UI progress updates.

    Returns:
        The final short story content.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    # Create a proper Chapter for the short story
    short_story_chapter = Chapter(
        number=1,
        title="Complete Story",
        outline="Short story",
        status="drafting",
    )
    orc.story_state.chapters = [short_story_chapter]

    # Retrieve RAG context for writer.
    # Temporal context is intentionally omitted for short stories — they don't
    # go through the full continuity review that chapter-based stories use.
    world_context = _retrieve_world_context(orc, "Write a complete short story")

    # Write initial draft
    orc._emit("agent_start", "Writer", "Writing story...")
    yield orc.events[-1]

    content = orc.writer.write_short_story(orc.story_state, world_context=world_context)

    # Validate language/correctness
    try:
        orc._validate_response(content, "Short story content")
    except ResponseValidationError as e:
        logger.warning(f"Validation warning for short story: {e}")

    short_story_chapter.content = content

    # Edit
    orc._emit("agent_start", "Editor", "Editing story...")
    yield orc.events[-1]

    edited_content = orc.editor.edit_chapter(orc.story_state, content, world_context=world_context)
    short_story_chapter.content = edited_content
    short_story_chapter.status = "edited"

    # Revision loop (matching write_chapter pattern)
    orc._emit("agent_start", "Continuity", "Checking for issues...")
    yield orc.events[-1]

    revision_count = 0
    while revision_count < orc.settings.max_revision_iterations:
        # Check for continuity issues
        issues = orc.continuity.check_chapter(
            orc.story_state, short_story_chapter.content, 1, world_context=world_context
        )

        # Also validate against plot outline
        outline_issues = orc.continuity.validate_against_outline(
            orc.story_state,
            short_story_chapter.content,
            orc.story_state.plot_summary,
        )
        issues.extend(outline_issues)

        if not issues or not orc.continuity.should_revise(issues):
            break

        revision_count += 1
        feedback = orc.continuity.format_revision_feedback(issues)
        short_story_chapter.revision_notes.append(feedback)

        orc._emit(
            "progress",
            "Writer",
            f"Revision {revision_count}: Addressing {len(issues)} issues...",
        )
        yield orc.events[-1]

        # Pass revision feedback to writer
        revised = orc.writer.write_short_story(
            orc.story_state, revision_feedback=feedback, world_context=world_context
        )
        short_story_chapter.content = orc.editor.edit_chapter(
            orc.story_state, revised, world_context=world_context
        )

    # Extract new facts from the story
    new_facts = orc.continuity.extract_new_facts(short_story_chapter.content, orc.story_state)
    orc.story_state.established_facts.extend(new_facts)

    # Finalize
    short_story_chapter.status = "final"
    short_story_chapter.word_count = len(short_story_chapter.content.split())
    orc.story_state.status = "complete"

    # Auto-save completed story
    try:
        orc.save_story()
        logger.debug("Auto-saved completed short story")
    except (OSError, PermissionError) as e:
        error_msg = f"Failed to save short story: {e}"
        logger.error(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e
    except (ValueError, TypeError) as e:
        error_msg = f"Failed to serialize short story: {e}"
        logger.error(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error saving short story: {e}"
        logger.exception(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e

    orc._emit(
        "agent_complete", "System", f"Story complete! ({short_story_chapter.word_count} words)"
    )
    yield orc.events[-1]

    return short_story_chapter.content


def write_chapter(
    orc: StoryOrchestrator, chapter_number: int, feedback: str | None = None
) -> Generator[WorkflowEvent, None, str]:
    """Run the full write-edit-continuity pipeline for a single chapter.

    Yields WorkflowEvent objects at key stages to report progress to the UI.
    Writes the chapter and applies editor passes. Performs continuity checks
    and iterative revisions. Extracts new facts and character-arc updates.
    Marks plot points as completed. Updates story state and autosaves.
    Optionally reports learning/training metrics via the configured mode service.

    Args:
        orc: StoryOrchestrator instance.
        chapter_number: The chapter number to process (must exist in the current story structure).
        feedback: Optional feedback or revision guidance to incorporate into the initial generation.

    Yields:
        WorkflowEvent objects for UI progress updates.

    Returns:
        The final content of the chapter after editing and revisions.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    # Validate chapter number bounds
    if not orc.story_state.chapters:
        raise ValueError("No chapters defined. Build story structure first.")

    valid_numbers = [c.number for c in orc.story_state.chapters]
    min_chapter = min(valid_numbers)
    max_chapter = max(valid_numbers)

    if chapter_number < min_chapter or chapter_number > max_chapter:
        raise ValueError(
            f"Chapter {chapter_number} is out of bounds. "
            f"Valid chapter numbers are {min_chapter} to {max_chapter}."
        )

    chapter = next((c for c in orc.story_state.chapters if c.number == chapter_number), None)
    if not chapter:
        raise ValueError(
            f"Chapter {chapter_number} not found. Available chapters: {sorted(valid_numbers)}"
        )

    chapter.status = "drafting"

    # Track total chapters for progress
    if orc._total_chapters == 0:
        orc._total_chapters = len(orc.story_state.chapters)

    # Retrieve RAG context for this chapter
    world_context = _retrieve_world_context(
        orc,
        f"Write chapter {chapter_number}: {chapter.title}. Outline: {chapter.outline[:200]}",
    )

    # Write
    orc._set_phase("writer")
    write_msg = f"Writing Chapter {chapter_number}..."
    if feedback:
        write_msg = f"Regenerating Chapter {chapter_number} with feedback..."
    orc._emit("agent_start", "Writer", write_msg)
    yield orc.events[-1]

    # Start learning tracking
    write_start_time = datetime.now()
    if orc.mode_service and orc.story_state.id:
        try:
            # Get model_id using the same logic as BaseAgent._get_model
            model_id = orc.writer.model
            if not model_id:
                if orc.settings.use_mode_system:
                    model_id = orc.mode_service.get_model_for_agent("writer")
                else:
                    model_id = orc.settings.get_model_for_agent("writer")
            genre = orc.story_state.brief.genre if orc.story_state.brief else None
            orc._current_score_id = orc.mode_service.record_generation(
                project_id=orc.story_state.id,
                agent_role="writer",
                model_id=model_id,
                chapter_id=str(chapter_number),
                genre=genre,
            )
            logger.debug(f"Started tracking generation score {orc._current_score_id}")
        except Exception as e:
            logger.warning(
                "Failed to start generation tracking for story %s chapter %d: %s",
                orc.story_state.id,
                chapter_number,
                e,
            )
            orc._current_score_id = None

    content = orc.writer.write_chapter(
        orc.story_state, chapter, revision_feedback=feedback, world_context=world_context
    )

    # Validate language/correctness
    try:
        orc._validate_response(content, f"Chapter {chapter_number} content")
    except ResponseValidationError as e:
        logger.warning(f"Validation warning for chapter {chapter_number}: {e}")

    chapter.content = content

    # Edit
    orc._set_phase("editor")
    orc._emit("agent_start", "Editor", f"Editing Chapter {chapter_number}...")
    yield orc.events[-1]

    edited_content = orc.editor.edit_chapter(orc.story_state, content, world_context=world_context)

    # Ensure consistency with previous chapter
    if chapter_number > 1:
        prev_chapter = next(
            (c for c in orc.story_state.chapters if c.number == chapter_number - 1), None
        )
        if prev_chapter and prev_chapter.content:
            edited_content = orc.editor.ensure_consistency(
                edited_content, prev_chapter.content, orc.story_state
            )

    chapter.content = edited_content
    chapter.status = "edited"

    # Check continuity
    orc._set_phase("continuity")
    orc._emit("agent_start", "Continuity", f"Checking Chapter {chapter_number}...")
    yield orc.events[-1]

    revision_count = 0
    while revision_count < orc.settings.max_revision_iterations:
        # Check for continuity issues
        issues = orc.continuity.check_chapter(
            orc.story_state, chapter.content, chapter_number, world_context=world_context
        )

        # Also validate against outline
        outline_issues = orc.continuity.validate_against_outline(
            orc.story_state, chapter.content, chapter.outline
        )
        issues.extend(outline_issues)

        if not issues or not orc.continuity.should_revise(issues):
            break

        revision_count += 1
        feedback = orc.continuity.format_revision_feedback(issues)
        chapter.revision_notes.append(feedback)

        orc._emit(
            "progress",
            "Writer",
            f"Revision {revision_count}: Addressing {len(issues)} issues...",
        )
        yield orc.events[-1]

        content = orc.writer.write_chapter(
            orc.story_state, chapter, revision_feedback=feedback, world_context=world_context
        )
        edited_content = orc.editor.edit_chapter(
            orc.story_state, content, world_context=world_context
        )

        # Ensure consistency in revisions too
        if chapter_number > 1:
            prev_chapter = next(
                (c for c in orc.story_state.chapters if c.number == chapter_number - 1), None
            )
            if prev_chapter and prev_chapter.content:
                edited_content = orc.editor.ensure_consistency(
                    edited_content, prev_chapter.content, orc.story_state
                )

        chapter.content = edited_content

    # Extract new facts
    new_facts = orc.continuity.extract_new_facts(chapter.content, orc.story_state)
    orc.story_state.established_facts.extend(new_facts)

    # Track character arc progression
    arc_updates = orc.continuity.extract_character_arcs(
        chapter.content, orc.story_state, chapter_number
    )
    for char_name, arc_state in arc_updates.items():
        char = orc.story_state.get_character_by_name(char_name)
        if char:
            char.update_arc(chapter_number, arc_state)

    # Mark completed plot points
    completed_indices = orc.continuity.check_plot_points_completed(
        chapter.content, orc.story_state, chapter_number
    )
    for idx in completed_indices:
        if idx < len(orc.story_state.plot_points):
            orc.story_state.plot_points[idx].completed = True
            logger.debug(f"Plot point {idx} marked complete")

    chapter.status = "final"
    chapter.word_count = len(chapter.content.split())
    orc.story_state.current_chapter = chapter_number

    # Update completed chapters count for progress tracking
    orc._completed_chapters = chapter_number

    # Finish learning tracking
    if orc.mode_service:
        try:
            # Update performance metrics
            if orc._current_score_id:
                elapsed = (datetime.now() - write_start_time).total_seconds()
                tokens_estimated = chapter.word_count * 1.3  # Rough token estimate
                orc.mode_service.update_performance_metrics(
                    orc._current_score_id,
                    tokens_generated=int(tokens_estimated),
                    time_seconds=elapsed,
                )
                logger.debug(
                    f"Updated score {orc._current_score_id}: "
                    f"{tokens_estimated:.0f} tokens in {elapsed:.1f}s"
                )

            # Notify chapter complete for learning triggers
            orc.mode_service.on_chapter_complete()
            logger.debug("Notified mode service of chapter completion")
        except Exception as e:
            logger.warning(
                "Failed to complete generation tracking for story %s chapter %d: %s",
                orc.story_state.id,
                chapter_number,
                e,
            )

    # Auto-save after each chapter to prevent data loss
    try:
        orc.save_story()
        logger.debug(f"Auto-saved after chapter {chapter_number}")
    except (OSError, PermissionError) as e:
        error_msg = f"Failed to save chapter {chapter_number}: {e}"
        logger.error(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e
    except (ValueError, TypeError) as e:
        error_msg = f"Failed to serialize chapter {chapter_number}: {e}"
        logger.error(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error saving chapter {chapter_number}: {e}"
        logger.exception(error_msg)
        orc._emit("error", "System", error_msg)
        raise ExportError(error_msg) from e

    orc._emit(
        "agent_complete",
        "System",
        f"Chapter {chapter_number} complete ({chapter.word_count} words)",
    )
    yield orc.events[-1]

    return chapter.content


def write_all_chapters(
    orc: StoryOrchestrator,
    on_checkpoint: Callable[[int, str], bool] | None = None,
) -> Generator[WorkflowEvent]:
    """Write all chapters, with optional checkpoints for user feedback.

    Args:
        orc: StoryOrchestrator instance.
        on_checkpoint: Callback at checkpoints. Returns True to continue, False to pause.

    Yields:
        WorkflowEvent objects for UI progress updates.
    """
    if not orc.story_state:
        raise ValueError("No story state. Call create_new_story() first.")

    for chapter in orc.story_state.chapters:
        # Write the chapter
        yield from orc.write_chapter(chapter.number)

        # Check if we need a checkpoint
        if (
            orc.interaction_mode in ["checkpoint", "interactive"]
            and chapter.number % orc.settings.chapters_between_checkpoints == 0
            and on_checkpoint
        ):
            orc._emit(
                "user_input_needed",
                "System",
                f"Checkpoint: Chapter {chapter.number} complete. Review and continue?",
                {"chapter": chapter.number, "content": chapter.content},
            )
            yield orc.events[-1]

            # The UI will handle getting user input and calling continue

    # Final full-story review
    orc._emit("agent_start", "Continuity", "Performing final story review...")
    yield orc.events[-1]

    world_context = _retrieve_world_context(orc, "Final story review for continuity")
    temporal_context = _retrieve_temporal_context(orc)
    combined_context = _combine_contexts(world_context, temporal_context)
    _warn_if_context_missing(orc, combined_context, "Final story review")
    final_issues = orc.continuity.check_full_story(orc.story_state, world_context=combined_context)
    if final_issues:
        # Report any remaining issues but don't block completion
        issue_summary = orc.continuity.format_revision_feedback(final_issues)
        critical_count = sum(1 for i in final_issues if i.severity == "critical")
        moderate_count = sum(1 for i in final_issues if i.severity == "moderate")
        minor_count = sum(1 for i in final_issues if i.severity == "minor")

        orc._emit(
            "progress",
            "Continuity",
            f"Final review found {len(final_issues)} issues: "
            f"{critical_count} critical, {moderate_count} moderate, {minor_count} minor",
            {"issues": [vars(i) for i in final_issues], "feedback": issue_summary},
        )
        yield orc.events[-1]
        logger.info(f"Final story review found {len(final_issues)} issues")
    else:
        orc._emit("progress", "Continuity", "Final review: No issues found!")
        yield orc.events[-1]
        logger.info("Final story review: clean")

    orc.story_state.status = "complete"
    orc._emit("agent_complete", "System", "All chapters complete!")
    yield orc.events[-1]
