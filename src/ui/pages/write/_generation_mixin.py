"""Write Story page - Generation and feedback mixin."""

import logging
from typing import TYPE_CHECKING

from nicegui import run

from src.ui.pages.write._page import WritePageBase

if TYPE_CHECKING:
    from src.memory.mode_models import TuningRecommendation

logger = logging.getLogger(__name__)


class GenerationMixin(WritePageBase):
    """Mixin providing chapter generation and feedback methods for WritePage.

    This mixin handles:
    - Writing individual chapters
    - Writing all chapters
    - Regenerating chapters with feedback
    - Learning recommendations
    """

    async def _write_current_chapter(self) -> None:
        """Write the current chapter with background processing and live updates."""
        from src.services.story_service import GenerationCancelled

        if not self.state.project or not self.state.current_chapter:
            self._notify("Select a chapter first", type="warning")
            return

        # Prevent multiple concurrent generations
        if self.state.is_writing:
            self._notify("Generation already in progress", type="warning")
            return

        # Capture for closure type narrowing
        project = self.state.project
        chapter_num = self.state.current_chapter

        try:
            # Reset generation flags
            self.state.reset_generation_flags()
            self.state.is_writing = True

            # Show generation status
            if self._generation_status:
                self._generation_status.show()
                self._generation_status.update_progress(f"Starting chapter {chapter_num}...")

            self._notify(f"Writing chapter {chapter_num}...", type="info")

            # Define cancellation check
            def should_cancel() -> bool:
                """Check if generation should be cancelled.

                Returns:
                    True if cancellation was requested.
                """
                return self.state.generation_cancel_requested

            # Run generation in background with progressive updates
            async def background_generation():
                """Run generation in a background thread and yield events."""
                events = []
                try:
                    # Run blocking generator in thread pool
                    def write_chapter_blocking():
                        """Execute the blocking chapter generation and collect all events."""
                        return list(
                            self.services.story.write_chapter(
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
                    self._notify("Generation cancelled by user", type="warning")
                else:
                    self._notify(f"Error: {error}", type="negative")
            else:
                # Process events for progress display
                for event in events:
                    self.state.writing_progress = event.message
                    if self._generation_status and self._client:
                        try:
                            with self._client:
                                self._generation_status.update_from_event(event)
                        except RuntimeError:
                            # Client context not available, skip UI update
                            logger.debug("Client context not available for progress update")

                self._refresh_writing_display()
                self.services.project.save_project(self.state.project)
                self._notify("Chapter complete!", type="positive")

                # Check for learning recommendations
                self._check_learning_recommendations()

        finally:
            self.state.is_writing = False
            self.state.reset_generation_flags()
            if self._generation_status and self._client:
                try:
                    with self._client:
                        self._generation_status.hide()
                except RuntimeError:
                    logger.debug("Client context not available for hiding status")

    async def _write_all_chapters(self) -> None:
        """Write all chapters with background processing and live updates."""
        from src.services.story_service import GenerationCancelled

        if not self.state.project:
            return

        # Prevent multiple concurrent generations
        if self.state.is_writing:
            self._notify("Generation already in progress", type="warning")
            return

        # Capture for closure type narrowing
        project = self.state.project

        try:
            # Reset generation flags
            self.state.reset_generation_flags()
            self.state.is_writing = True

            # Show generation status
            if self._generation_status:
                self._generation_status.show()
                self._generation_status.update_progress("Starting story generation...")

            self._notify("Writing all chapters...", type="info")

            # Define cancellation check
            def should_cancel() -> bool:
                """Check if the user has requested to cancel the generation."""
                return self.state.generation_cancel_requested

            # Run generation in background with progressive updates
            async def background_generation():
                """Run generation in a background thread and yield events."""
                events = []
                try:
                    # Run blocking generator in thread pool
                    def write_all_blocking():
                        """Execute the blocking generation for all chapters and collect events."""
                        return list(
                            self.services.story.write_all_chapters(
                                project, cancel_check=should_cancel
                            )
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
                    self._notify("Generation cancelled by user", type="warning")
                else:
                    self._notify(f"Error: {error}", type="negative")
            else:
                # Process events for progress display
                for event in events:
                    self.state.writing_progress = event.message
                    if self._generation_status and self._client:
                        try:
                            with self._client:
                                self._generation_status.update_from_event(event)
                        except RuntimeError:
                            # Client context not available, skip UI update
                            logger.debug("Client context not available for progress update")

                self._refresh_writing_display()
                self.services.project.save_project(self.state.project)
                self._notify("All chapters complete!", type="positive")

                # Check for learning recommendations (story complete = more likely to have recs)
                self._check_learning_recommendations()

        finally:
            self.state.is_writing = False
            self.state.reset_generation_flags()
            if self._generation_status and self._client:
                try:
                    with self._client:
                        self._generation_status.hide()
                except RuntimeError:
                    logger.debug("Client context not available for hiding status")

    def _check_learning_recommendations(self) -> None:
        """
        Check for pending tuning recommendations and show them to the user if present.

        This is a no-op when no mode service is available or when learning is disabled in settings.
        Any errors encountered while retrieving or displaying recommendations are logged and not raised.
        """
        from src.ui.components.recommendation_dialog import show_recommendations

        if not self.services.mode:
            return

        # Check if learning is enabled
        settings = self.services.settings
        if "off" in settings.learning_triggers:
            return

        try:
            # Get pending recommendations (don't gate on should_tune() to allow AFTER_PROJECT)
            recommendations = self.services.mode.get_pending_recommendations()
            if not recommendations:
                logger.debug("No pending recommendations to show")
                return

            logger.info(f"Showing {len(recommendations)} pending recommendations")

            # Show recommendation dialog
            show_recommendations(
                recommendations=recommendations,
                on_apply=self._apply_recommendations,
                on_dismiss=self._dismiss_recommendations,
            )
        except Exception as e:
            logger.warning(f"Error checking learning recommendations: {e}")

    def _apply_recommendations(self, recommendations: list[TuningRecommendation]) -> None:
        """
        Apply a list of tuning recommendations using the configured mode service.

        Attempts to apply each recommendation and shows a user notification indicating how many recommendations were applied. If the mode service is not available, the call is a no-op. Errors applying individual recommendations are logged and do not stop processing the remaining items.

        Parameters:
            recommendations (list[TuningRecommendation]): Recommendations to apply.
        """
        if not self.services.mode:
            return

        try:
            applied_count = 0
            for rec in recommendations:
                try:
                    if self.services.mode.apply_recommendation(rec):
                        applied_count += 1
                except Exception as e:
                    logger.error(f"Failed to apply recommendation {rec.id}: {e}")

            if applied_count > 0:
                self._notify(f"Applied {applied_count} recommendation(s)", type="positive")
                logger.info(f"Applied {applied_count} learning recommendations")
            else:
                self._notify("No recommendations applied", type="info")
        except Exception as e:
            logger.error(f"Error applying recommendations: {e}")
            self._notify("Failed to apply recommendations", type="negative")

    def _dismiss_recommendations(self, recommendations: list[TuningRecommendation]) -> None:
        """
        Persist dismissal of the given tuning recommendations.

        If a mode service is available, each recommendation will be marked as dismissed; failures for individual recommendations are logged and do not stop processing.

        Parameters:
            recommendations (list[TuningRecommendation]): Recommendations to dismiss.
        """
        if not self.services.mode:
            logger.debug("No mode service available to dismiss recommendations")
            return

        dismissed_count = 0
        for rec in recommendations:
            try:
                self.services.mode.dismiss_recommendation(rec)
                dismissed_count += 1
            except Exception as e:
                logger.error(f"Failed to dismiss recommendation: {e}")

        logger.info(f"Dismissed {dismissed_count} recommendation(s)")

    def _record_regeneration_signal(self, project_id: str, chapter_num: int) -> None:
        """Record regeneration as an implicit negative signal for learning.

        When a user regenerates a chapter, it indicates dissatisfaction with
        the previous output. This signal helps the learning system understand
        model performance.

        Args:
            project_id: The project ID.
            chapter_num: The chapter being regenerated.
        """
        if not self.services.mode:
            return

        try:
            # Mark the most recent score for this chapter as regenerated
            self.services.mode.on_regenerate(project_id, str(chapter_num))
            logger.debug(
                f"Recorded regeneration signal for project {project_id}, chapter {chapter_num}"
            )
        except Exception as e:
            # Don't fail the regeneration if signal recording fails
            logger.warning(f"Failed to record regeneration signal: {e}")

    async def _regenerate_with_feedback(self) -> None:
        """
        Regenerate the currently selected chapter using the text entered in the feedback input.

        This method:
        - Validates that a project, a selected chapter, and non-empty feedback exist
        - Records the regeneration as a learning signal
        - Prevents concurrent generation
        - Shows progress and status updates
        - Supports user cancellation
        - Saves the updated chapter and version history
        - Refreshes the writing UI
        - Clears the feedback input on success
        - Notifies the user of success, cancellation, or errors
        """
        from src.services.story_service import GenerationCancelled

        if not self._regenerate_feedback_input or not self._regenerate_feedback_input.value:
            self._notify("Please enter feedback before regenerating", type="warning")
            return
        if not self.state.project or self.state.current_chapter is None:
            self._notify("No chapter selected", type="warning")
            return

        # Check if chapter has content
        chapter = next(
            (c for c in self.state.project.chapters if c.number == self.state.current_chapter),
            None,
        )
        if not chapter or not chapter.content:
            self._notify("Chapter has no content yet. Write it first.", type="warning")
            return

        # Prevent multiple concurrent generations
        if self.state.is_writing:
            self._notify("Generation already in progress", type="warning")
            return

        # Capture for closure type narrowing
        project = self.state.project
        chapter_num = self.state.current_chapter
        feedback = self._regenerate_feedback_input.value

        # Record regeneration as implicit negative signal for learning
        self._record_regeneration_signal(project.id, chapter_num)

        try:
            # Reset generation flags
            self.state.reset_generation_flags()
            self.state.is_writing = True

            # Show generation status
            if self._generation_status:
                self._generation_status.show()
                self._generation_status.update_progress(
                    f"Regenerating chapter {chapter_num} with your feedback..."
                )

            self._notify(f"Regenerating chapter {chapter_num}...", type="info")

            # Define cancellation check
            def should_cancel() -> bool:
                """Check if the user has requested to cancel the regeneration."""
                return self.state.generation_cancel_requested

            # Run regeneration in background with progressive updates
            async def background_regeneration():
                """Run regeneration in a background thread and yield events."""
                events = []
                try:
                    # Run blocking generator in thread pool
                    def regenerate_blocking():
                        """Execute the blocking chapter regeneration with feedback and collect events."""
                        return list(
                            self.services.story.regenerate_chapter_with_feedback(
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
                    self._notify("Regeneration cancelled by user", type="warning")
                else:
                    self._notify(f"Error: {error}", type="negative")
            else:
                # Process events for progress display
                for event in events:
                    self.state.writing_progress = event.message
                    if self._generation_status and self._client:
                        try:
                            with self._client:
                                self._generation_status.update_from_event(event)
                        except RuntimeError:
                            logger.debug("Client context not available for progress update")

                self._refresh_writing_display()
                self._refresh_version_history()
                self.services.project.save_project(self.state.project)
                self._notify("Chapter regenerated successfully!", type="positive")
                # Clear feedback input
                self._regenerate_feedback_input.value = ""

        finally:
            self.state.is_writing = False
            self.state.reset_generation_flags()
            if self._generation_status and self._client:
                try:
                    with self._client:
                        self._generation_status.hide()
                except RuntimeError:
                    logger.debug("Client context not available for hiding status")
