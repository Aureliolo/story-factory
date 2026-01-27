"""Generation logic - write chapter, write all, regenerate, learning recommendations."""

import logging
from typing import TYPE_CHECKING

from nicegui import run

from src.memory.mode_models import TuningRecommendation
from src.services.story_service import GenerationCancelled
from src.ui.components.recommendation_dialog import show_recommendations

if TYPE_CHECKING:
    from . import WritePage

logger = logging.getLogger("src.ui.pages.write._generation")


async def write_current_chapter(page: WritePage) -> None:
    """Write the current chapter with background processing and live updates.

    Args:
        page: The WritePage instance.
    """
    from . import _writing as writing_mod

    if not page.state.project or not page.state.current_chapter:
        page._notify("Select a chapter first", type="warning")
        return

    # Prevent multiple concurrent generations
    if page.state.is_writing:
        page._notify("Generation already in progress", type="warning")
        return

    # Capture for closure type narrowing
    project = page.state.project
    chapter_num = page.state.current_chapter

    try:
        # Reset generation flags
        page.state.reset_generation_flags()
        page.state.is_writing = True

        # Show generation status
        if page._generation_status:
            page._generation_status.show()
            page._generation_status.update_progress(f"Starting chapter {chapter_num}...")

        page._notify(f"Writing chapter {chapter_num}...", type="info")

        # Define cancellation check
        def should_cancel() -> bool:
            """Check if generation should be cancelled.

            Returns:
                True if cancellation was requested.
            """
            return page.state.generation_cancel_requested

        # Run generation in background with progressive updates
        async def background_generation():
            """Run generation in a background thread and yield events."""
            events = []
            try:
                # Run blocking generator in thread pool
                def write_chapter_blocking():
                    """Execute the blocking chapter generation and collect all events."""
                    return list(
                        page.services.story.write_chapter(
                            project, chapter_num, cancel_check=should_cancel
                        )
                    )

                events = await run.io_bound(write_chapter_blocking)
                return events, None
            except GenerationCancelled as e:
                logger.info(f"Chapter {chapter_num} generation cancelled")
                return [], e
            except Exception as e:
                logger.exception(f"Failed to write chapter {chapter_num}")
                return [], e

        # Process events with UI updates
        events, error = await background_generation()

        # Update UI based on results
        if error:
            if isinstance(error, GenerationCancelled):
                page._notify("Generation cancelled by user", type="warning")
            else:
                page._notify(f"Error: {error}", type="negative")
        else:
            # Process events for progress display
            for event in events:
                page.state.writing_progress = event.message
                if page._generation_status and page._client:
                    try:
                        with page._client:
                            page._generation_status.update_from_event(event)
                    except RuntimeError:
                        # Client context not available, skip UI update
                        logger.debug("Client context not available for progress update")

            writing_mod.refresh_writing_display(page)
            page.services.project.save_project(page.state.project)
            page._notify("Chapter complete!", type="positive")

            # Check for learning recommendations
            _check_learning_recommendations(page)

    finally:
        page.state.is_writing = False
        page.state.reset_generation_flags()
        if page._generation_status and page._client:
            try:
                with page._client:
                    page._generation_status.hide()
            except RuntimeError:
                logger.debug("Client context not available for hiding status")


async def write_all_chapters(page: WritePage) -> None:
    """Write all chapters with background processing and live updates.

    Args:
        page: The WritePage instance.
    """
    from . import _writing as writing_mod

    if not page.state.project:
        return

    # Prevent multiple concurrent generations
    if page.state.is_writing:
        page._notify("Generation already in progress", type="warning")
        return

    # Capture for closure type narrowing
    project = page.state.project

    try:
        # Reset generation flags
        page.state.reset_generation_flags()
        page.state.is_writing = True

        # Show generation status
        if page._generation_status:
            page._generation_status.show()
            page._generation_status.update_progress("Starting story generation...")

        page._notify("Writing all chapters...", type="info")

        # Define cancellation check
        def should_cancel() -> bool:
            """Check if the user has requested to cancel the generation."""
            return page.state.generation_cancel_requested

        # Run generation in background with progressive updates
        async def background_generation():
            """Run generation in a background thread and yield events."""
            events = []
            try:
                # Run blocking generator in thread pool
                def write_all_blocking():
                    """Execute the blocking generation for all chapters and collect events."""
                    return list(
                        page.services.story.write_all_chapters(project, cancel_check=should_cancel)
                    )

                events = await run.io_bound(write_all_blocking)
                return events, None
            except GenerationCancelled as e:
                logger.info("Write all chapters cancelled")
                return [], e
            except Exception as e:
                logger.exception("Failed to write all chapters")
                return [], e

        # Process events with UI updates
        events, error = await background_generation()

        # Update UI based on results
        if error:
            if isinstance(error, GenerationCancelled):
                page._notify("Generation cancelled by user", type="warning")
            else:
                page._notify(f"Error: {error}", type="negative")
        else:
            # Process events for progress display
            for event in events:
                page.state.writing_progress = event.message
                if page._generation_status and page._client:
                    try:
                        with page._client:
                            page._generation_status.update_from_event(event)
                    except RuntimeError:
                        # Client context not available, skip UI update
                        logger.debug("Client context not available for progress update")

            writing_mod.refresh_writing_display(page)
            page.services.project.save_project(page.state.project)
            page._notify("All chapters complete!", type="positive")

            # Check for learning recommendations (story complete = more likely to have recs)
            _check_learning_recommendations(page)

    finally:
        page.state.is_writing = False
        page.state.reset_generation_flags()
        if page._generation_status and page._client:
            try:
                with page._client:
                    page._generation_status.hide()
            except RuntimeError:
                logger.debug("Client context not available for hiding status")


def _check_learning_recommendations(page: WritePage) -> None:
    """Check for pending tuning recommendations and show them to the user if present.

    This is a no-op when no mode service is available or when learning is disabled in settings.
    Any errors encountered while retrieving or displaying recommendations are logged and not raised.

    Args:
        page: The WritePage instance.
    """
    if not page.services.mode:
        return

    # Check if learning is enabled
    settings = page.services.settings
    if "off" in settings.learning_triggers:
        return

    try:
        # Get pending recommendations (don't gate on should_tune() to allow AFTER_PROJECT)
        recommendations = page.services.mode.get_pending_recommendations()
        if not recommendations:
            logger.debug("No pending recommendations to show")
            return

        logger.info(f"Showing {len(recommendations)} pending recommendations")

        # Show recommendation dialog
        show_recommendations(
            recommendations=recommendations,
            on_apply=lambda recs: _apply_recommendations(page, recs),
            on_dismiss=lambda recs: _dismiss_recommendations(page, recs),
        )
    except Exception as e:
        logger.warning(f"Error checking learning recommendations: {e}")


def _apply_recommendations(page: WritePage, recommendations: list[TuningRecommendation]) -> None:
    """Apply a list of tuning recommendations using the configured mode service.

    Attempts to apply each recommendation and shows a user notification indicating
    how many recommendations were applied. If the mode service is not available,
    the call is a no-op.

    Args:
        page: The WritePage instance.
        recommendations: Recommendations to apply.
    """
    if not page.services.mode:
        return

    try:
        applied_count = 0
        for rec in recommendations:
            try:
                if page.services.mode.apply_recommendation(rec):
                    applied_count += 1
            except Exception as e:
                logger.error(f"Failed to apply recommendation {rec.id}: {e}")

        if applied_count > 0:
            page._notify(f"Applied {applied_count} recommendation(s)", type="positive")
            logger.info(f"Applied {applied_count} learning recommendations")
        else:
            page._notify("No recommendations applied", type="info")
    except Exception as e:
        logger.error(f"Error applying recommendations: {e}")
        page._notify("Failed to apply recommendations", type="negative")


def _dismiss_recommendations(page: WritePage, recommendations: list[TuningRecommendation]) -> None:
    """Persist dismissal of the given tuning recommendations.

    If a mode service is available, each recommendation will be marked as dismissed;
    failures for individual recommendations are logged and do not stop processing.

    Args:
        page: The WritePage instance.
        recommendations: Recommendations to dismiss.
    """
    if not page.services.mode:
        logger.debug("No mode service available to dismiss recommendations")
        return

    dismissed_count = 0
    for rec in recommendations:
        try:
            page.services.mode.dismiss_recommendation(rec)
            dismissed_count += 1
        except Exception as e:
            logger.error(f"Failed to dismiss recommendation: {e}")

    logger.info(f"Dismissed {dismissed_count} recommendation(s)")


def _record_regeneration_signal(page: WritePage, project_id: str, chapter_num: int) -> None:
    """Record regeneration as an implicit negative signal for learning.

    When a user regenerates a chapter, it indicates dissatisfaction with
    the previous output. This signal helps the learning system understand
    model performance.

    Args:
        page: The WritePage instance.
        project_id: The project ID.
        chapter_num: The chapter being regenerated.
    """
    if not page.services.mode:
        return

    try:
        # Mark the most recent score for this chapter as regenerated
        page.services.mode.on_regenerate(project_id, str(chapter_num))
        logger.debug(
            f"Recorded regeneration signal for project {project_id}, chapter {chapter_num}"
        )
    except Exception as e:
        # Don't fail the regeneration if signal recording fails
        logger.warning(f"Failed to record regeneration signal: {e}")


async def regenerate_with_feedback(page: WritePage) -> None:
    """Regenerate the currently selected chapter using the text entered in the feedback input.

    This method validates that a project, a selected chapter, and non-empty feedback exist,
    records the regeneration as a learning signal, prevents concurrent generation, shows progress
    and status updates, supports user cancellation, saves the updated chapter and version history,
    refreshes the writing UI, clears the feedback input on success, and notifies the user.

    Args:
        page: The WritePage instance.
    """
    from . import _writing as writing_mod

    if not page._regenerate_feedback_input or not page._regenerate_feedback_input.value:
        page._notify("Please enter feedback before regenerating", type="warning")
        return
    if not page.state.project or page.state.current_chapter is None:
        page._notify("No chapter selected", type="warning")
        return

    # Check if chapter has content
    chapter = next(
        (c for c in page.state.project.chapters if c.number == page.state.current_chapter),
        None,
    )
    if not chapter or not chapter.content:
        page._notify("Chapter has no content yet. Write it first.", type="warning")
        return

    # Prevent multiple concurrent generations
    if page.state.is_writing:
        page._notify("Generation already in progress", type="warning")
        return

    # Capture for closure type narrowing
    project = page.state.project
    chapter_num = page.state.current_chapter
    feedback = page._regenerate_feedback_input.value

    # Record regeneration as implicit negative signal for learning
    _record_regeneration_signal(page, project.id, chapter_num)

    try:
        # Reset generation flags
        page.state.reset_generation_flags()
        page.state.is_writing = True

        # Show generation status
        if page._generation_status:
            page._generation_status.show()
            page._generation_status.update_progress(
                f"Regenerating chapter {chapter_num} with your feedback..."
            )

        page._notify(f"Regenerating chapter {chapter_num}...", type="info")

        # Define cancellation check
        def should_cancel() -> bool:
            """Check if the user has requested to cancel the regeneration."""
            return page.state.generation_cancel_requested

        # Run regeneration in background with progressive updates
        async def background_regeneration():
            """Run regeneration in a background thread and yield events."""
            events = []
            try:
                # Run blocking generator in thread pool
                def regenerate_blocking():
                    """Execute the blocking chapter regeneration with feedback and collect events."""
                    return list(
                        page.services.story.regenerate_chapter_with_feedback(
                            project, chapter_num, feedback, cancel_check=should_cancel
                        )
                    )

                events = await run.io_bound(regenerate_blocking)
                return events, None
            except GenerationCancelled as e:
                logger.info(f"Chapter {chapter_num} regeneration cancelled")
                return [], e
            except Exception as e:
                logger.exception(f"Failed to regenerate chapter {chapter_num}")
                return [], e

        # Process events with UI updates
        events, error = await background_regeneration()

        # Update UI based on results
        if error:
            if isinstance(error, GenerationCancelled):
                page._notify("Regeneration cancelled by user", type="warning")
            else:
                page._notify(f"Error: {error}", type="negative")
        else:
            # Process events for progress display
            for event in events:
                page.state.writing_progress = event.message
                if page._generation_status and page._client:
                    try:
                        with page._client:
                            page._generation_status.update_from_event(event)
                    except RuntimeError:
                        logger.debug("Client context not available for progress update")

            writing_mod.refresh_writing_display(page)
            writing_mod.refresh_version_history(page)
            page.services.project.save_project(page.state.project)
            page._notify("Chapter regenerated successfully!", type="positive")
            # Clear feedback input
            page._regenerate_feedback_input.value = ""

    finally:
        page.state.is_writing = False
        page.state.reset_generation_flags()
        if page._generation_status and page._client:
            try:
                with page._client:
                    page._generation_status.hide()
            except RuntimeError:
                logger.debug("Client context not available for hiding status")
