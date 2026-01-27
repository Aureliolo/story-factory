"""Writing phase mixin for StoryOrchestrator."""

import logging
from collections.abc import Callable, Generator
from datetime import datetime

from src.agents import ResponseValidationError
from src.memory.story_state import Chapter
from src.services.orchestrator._base import StoryOrchestratorBase, WorkflowEvent
from src.utils.exceptions import ExportError

logger = logging.getLogger(__name__)


class WritingMixin(StoryOrchestratorBase):
    """Mixin providing writing phase functionality."""

    def write_short_story(self) -> Generator[WorkflowEvent, None, str]:
        """Write a short story with revision loop."""
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        # Create a proper Chapter for the short story
        short_story_chapter = Chapter(
            number=1,
            title="Complete Story",
            outline="Short story",
            status="drafting",
        )
        self.story_state.chapters = [short_story_chapter]

        # Write initial draft
        self._emit("agent_start", "Writer", "Writing story...")
        yield self.events[-1]

        content = self.writer.write_short_story(self.story_state)

        # Validate language/correctness
        try:
            self._validate_response(content, "Short story content")
        except ResponseValidationError as e:
            logger.warning(f"Validation warning for short story: {e}")

        short_story_chapter.content = content

        # Edit
        self._emit("agent_start", "Editor", "Editing story...")
        yield self.events[-1]

        edited_content = self.editor.edit_chapter(self.story_state, content)
        short_story_chapter.content = edited_content
        short_story_chapter.status = "edited"

        # Revision loop (matching write_chapter pattern)
        self._emit("agent_start", "Continuity", "Checking for issues...")
        yield self.events[-1]

        revision_count = 0
        while revision_count < self.settings.max_revision_iterations:
            # Check for continuity issues
            issues = self.continuity.check_chapter(self.story_state, short_story_chapter.content, 1)

            # Also validate against plot outline
            outline_issues = self.continuity.validate_against_outline(
                self.story_state,
                short_story_chapter.content,
                self.story_state.plot_summary,
            )
            issues.extend(outline_issues)

            if not issues or not self.continuity.should_revise(issues):
                break

            revision_count += 1
            feedback = self.continuity.format_revision_feedback(issues)
            short_story_chapter.revision_notes.append(feedback)

            self._emit(
                "progress",
                "Writer",
                f"Revision {revision_count}: Addressing {len(issues)} issues...",
            )
            yield self.events[-1]

            # Pass revision feedback to writer
            revised = self.writer.write_short_story(self.story_state, revision_feedback=feedback)
            short_story_chapter.content = self.editor.edit_chapter(self.story_state, revised)

        # Extract new facts from the story
        new_facts = self.continuity.extract_new_facts(short_story_chapter.content, self.story_state)
        self.story_state.established_facts.extend(new_facts)

        # Finalize
        short_story_chapter.status = "final"
        short_story_chapter.word_count = len(short_story_chapter.content.split())
        self.story_state.status = "complete"

        # Auto-save completed story
        try:
            self.save_story()
            logger.debug("Auto-saved completed short story")
        except (OSError, PermissionError) as e:
            error_msg = f"Failed to save short story: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e
        except (ValueError, TypeError) as e:
            error_msg = f"Failed to serialize short story: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error saving short story: {e}"
            logger.exception(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e

        self._emit(
            "agent_complete", "System", f"Story complete! ({short_story_chapter.word_count} words)"
        )
        yield self.events[-1]

        return short_story_chapter.content

    def write_chapter(
        self, chapter_number: int, feedback: str | None = None
    ) -> Generator[WorkflowEvent, None, str]:
        """
        Run the full write-edit-continuity pipeline for a single chapter, yielding workflow events during processing.

        Detailed behavior:
        - Yields WorkflowEvent objects at key stages to report progress to the UI.
        - Writes the chapter and applies editor passes.
        - Performs continuity checks and iterative revisions.
        - Extracts new facts and character-arc updates.
        - Marks plot points as completed.
        - Updates story state and autosaves.
        - Optionally reports learning/training metrics via the configured mode service.

        Parameters:
            chapter_number (int): The chapter number to process (must exist in the current story structure).
            feedback (str | None): Optional feedback or revision guidance to incorporate into the initial generation.

        Returns:
            chapter_content (str): The final content of the chapter after editing and revisions.
        """
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        # Validate chapter number bounds
        if not self.story_state.chapters:
            raise ValueError("No chapters defined. Build story structure first.")

        valid_numbers = [c.number for c in self.story_state.chapters]
        min_chapter = min(valid_numbers)
        max_chapter = max(valid_numbers)

        if chapter_number < min_chapter or chapter_number > max_chapter:
            raise ValueError(
                f"Chapter {chapter_number} is out of bounds. "
                f"Valid chapter numbers are {min_chapter} to {max_chapter}."
            )

        chapter = next((c for c in self.story_state.chapters if c.number == chapter_number), None)
        if not chapter:
            raise ValueError(
                f"Chapter {chapter_number} not found. Available chapters: {sorted(valid_numbers)}"
            )

        chapter.status = "drafting"

        # Track total chapters for progress
        if self._total_chapters == 0:
            self._total_chapters = len(self.story_state.chapters)

        # Write
        self._set_phase("writer")
        write_msg = f"Writing Chapter {chapter_number}..."
        if feedback:
            write_msg = f"Regenerating Chapter {chapter_number} with feedback..."
        self._emit("agent_start", "Writer", write_msg)
        yield self.events[-1]

        # Start learning tracking
        write_start_time = datetime.now()
        if self.mode_service and self.story_state.id:
            try:
                # Get model_id using the same logic as BaseAgent._get_model
                model_id = self.writer.model
                if not model_id:
                    if self.settings.use_mode_system:
                        model_id = self.mode_service.get_model_for_agent("writer")
                    else:
                        model_id = self.settings.get_model_for_agent("writer")
                genre = self.story_state.brief.genre if self.story_state.brief else None
                self._current_score_id = self.mode_service.record_generation(
                    project_id=self.story_state.id,
                    agent_role="writer",
                    model_id=model_id,
                    chapter_id=str(chapter_number),
                    genre=genre,
                )
                logger.debug(f"Started tracking generation score {self._current_score_id}")
            except Exception as e:
                logger.warning(
                    "Failed to start generation tracking for story %s chapter %d: %s",
                    self.story_state.id,
                    chapter_number,
                    e,
                )
                self._current_score_id = None

        content = self.writer.write_chapter(self.story_state, chapter, revision_feedback=feedback)

        # Validate language/correctness
        try:
            self._validate_response(content, f"Chapter {chapter_number} content")
        except ResponseValidationError as e:
            logger.warning(f"Validation warning for chapter {chapter_number}: {e}")

        chapter.content = content

        # Edit
        self._set_phase("editor")
        self._emit("agent_start", "Editor", f"Editing Chapter {chapter_number}...")
        yield self.events[-1]

        edited_content = self.editor.edit_chapter(self.story_state, content)

        # Ensure consistency with previous chapter
        if chapter_number > 1:
            prev_chapter = next(
                (c for c in self.story_state.chapters if c.number == chapter_number - 1), None
            )
            if prev_chapter and prev_chapter.content:
                edited_content = self.editor.ensure_consistency(
                    edited_content, prev_chapter.content, self.story_state
                )

        chapter.content = edited_content
        chapter.status = "edited"

        # Check continuity
        self._set_phase("continuity")
        self._emit("agent_start", "Continuity", f"Checking Chapter {chapter_number}...")
        yield self.events[-1]

        revision_count = 0
        while revision_count < self.settings.max_revision_iterations:
            # Check for continuity issues
            issues = self.continuity.check_chapter(
                self.story_state, chapter.content, chapter_number
            )

            # Also validate against outline
            outline_issues = self.continuity.validate_against_outline(
                self.story_state, chapter.content, chapter.outline
            )
            issues.extend(outline_issues)

            if not issues or not self.continuity.should_revise(issues):
                break

            revision_count += 1
            feedback = self.continuity.format_revision_feedback(issues)
            chapter.revision_notes.append(feedback)

            self._emit(
                "progress",
                "Writer",
                f"Revision {revision_count}: Addressing {len(issues)} issues...",
            )
            yield self.events[-1]

            content = self.writer.write_chapter(
                self.story_state, chapter, revision_feedback=feedback
            )
            edited_content = self.editor.edit_chapter(self.story_state, content)

            # Ensure consistency in revisions too
            if chapter_number > 1:
                prev_chapter = next(
                    (c for c in self.story_state.chapters if c.number == chapter_number - 1), None
                )
                if prev_chapter and prev_chapter.content:
                    edited_content = self.editor.ensure_consistency(
                        edited_content, prev_chapter.content, self.story_state
                    )

            chapter.content = edited_content

        # Extract new facts
        new_facts = self.continuity.extract_new_facts(chapter.content, self.story_state)
        self.story_state.established_facts.extend(new_facts)

        # Track character arc progression
        arc_updates = self.continuity.extract_character_arcs(
            chapter.content, self.story_state, chapter_number
        )
        for char_name, arc_state in arc_updates.items():
            char = self.story_state.get_character_by_name(char_name)
            if char:
                char.update_arc(chapter_number, arc_state)

        # Mark completed plot points
        completed_indices = self.continuity.check_plot_points_completed(
            chapter.content, self.story_state, chapter_number
        )
        for idx in completed_indices:
            if idx < len(self.story_state.plot_points):
                self.story_state.plot_points[idx].completed = True
                logger.debug(f"Plot point {idx} marked complete")

        chapter.status = "final"
        chapter.word_count = len(chapter.content.split())
        self.story_state.current_chapter = chapter_number

        # Update completed chapters count for progress tracking
        self._completed_chapters = chapter_number

        # Finish learning tracking
        if self.mode_service:
            try:
                # Update performance metrics
                if self._current_score_id:
                    elapsed = (datetime.now() - write_start_time).total_seconds()
                    tokens_estimated = chapter.word_count * 1.3  # Rough token estimate
                    self.mode_service.update_performance_metrics(
                        self._current_score_id,
                        tokens_generated=int(tokens_estimated),
                        time_seconds=elapsed,
                    )
                    logger.debug(
                        f"Updated score {self._current_score_id}: "
                        f"{tokens_estimated:.0f} tokens in {elapsed:.1f}s"
                    )

                # Notify chapter complete for learning triggers
                self.mode_service.on_chapter_complete()
                logger.debug("Notified mode service of chapter completion")
            except Exception as e:
                logger.warning(
                    "Failed to complete generation tracking for story %s chapter %d: %s",
                    self.story_state.id,
                    chapter_number,
                    e,
                )

        # Auto-save after each chapter to prevent data loss
        try:
            self.save_story()
            logger.debug(f"Auto-saved after chapter {chapter_number}")
        except (OSError, PermissionError) as e:
            error_msg = f"Failed to save chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e
        except (ValueError, TypeError) as e:
            error_msg = f"Failed to serialize chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error saving chapter {chapter_number}: {e}"
            logger.exception(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e

        self._emit(
            "agent_complete",
            "System",
            f"Chapter {chapter_number} complete ({chapter.word_count} words)",
        )
        yield self.events[-1]

        return chapter.content

    def write_all_chapters(
        self,
        on_checkpoint: Callable[[int, str], bool] | None = None,
    ) -> Generator[WorkflowEvent]:
        """Write all chapters, with optional checkpoints for user feedback.

        Args:
            on_checkpoint: Callback at checkpoints. Returns True to continue, False to pause.
        """
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        for chapter in self.story_state.chapters:
            # Write the chapter
            yield from self.write_chapter(chapter.number)

            # Check if we need a checkpoint
            if (
                self.interaction_mode in ["checkpoint", "interactive"]
                and chapter.number % self.settings.chapters_between_checkpoints == 0
                and on_checkpoint
            ):
                self._emit(
                    "user_input_needed",
                    "System",
                    f"Checkpoint: Chapter {chapter.number} complete. Review and continue?",
                    {"chapter": chapter.number, "content": chapter.content},
                )
                yield self.events[-1]

                # The UI will handle getting user input and calling continue

        # Final full-story review
        self._emit("agent_start", "Continuity", "Performing final story review...")
        yield self.events[-1]

        final_issues = self.continuity.check_full_story(self.story_state)
        if final_issues:
            # Report any remaining issues but don't block completion
            issue_summary = self.continuity.format_revision_feedback(final_issues)
            critical_count = sum(1 for i in final_issues if i.severity == "critical")
            moderate_count = sum(1 for i in final_issues if i.severity == "moderate")
            minor_count = sum(1 for i in final_issues if i.severity == "minor")

            self._emit(
                "progress",
                "Continuity",
                f"Final review found {len(final_issues)} issues: {critical_count} critical, {moderate_count} moderate, {minor_count} minor",
                {"issues": [vars(i) for i in final_issues], "feedback": issue_summary},
            )
            yield self.events[-1]
            logger.info(f"Final story review found {len(final_issues)} issues")
        else:
            self._emit("progress", "Continuity", "Final review: No issues found!")
            yield self.events[-1]
            logger.info("Final story review: clean")

        self.story_state.status = "complete"
        self._emit("agent_complete", "System", "All chapters complete!")
        yield self.events[-1]
