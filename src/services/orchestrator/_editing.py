"""Editing phase mixin for StoryOrchestrator."""

import logging
from collections.abc import Generator

from src.agents import ResponseValidationError
from src.agents.continuity import ContinuityIssue
from src.services.orchestrator._base import StoryOrchestratorBase, WorkflowEvent
from src.utils.exceptions import ExportError

logger = logging.getLogger(__name__)


class EditingMixin(StoryOrchestratorBase):
    """Mixin providing editing phase functionality."""

    def continue_chapter(
        self,
        chapter_number: int,
        direction: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Continue writing a chapter from where it left off.

        Args:
            chapter_number: The chapter to continue.
            direction: Optional direction for where to take the scene.

        Returns:
            The continued content.
        """
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        chapter = next((c for c in self.story_state.chapters if c.number == chapter_number), None)
        if not chapter:
            raise ValueError(f"Chapter {chapter_number} not found.")

        if not chapter.content:
            raise ValueError(f"Chapter {chapter_number} has no content to continue from.")

        self._emit(
            "agent_start",
            "Writer",
            f"Continuing Chapter {chapter_number}...",
            {"direction": direction} if direction else None,
        )
        yield self.events[-1]

        # Use WriterAgent.continue_scene() to continue from current text
        continuation = self.writer.continue_scene(
            self.story_state,
            chapter.content,
            direction=direction,
        )

        # Validate continuation
        try:
            self._validate_response(continuation, f"Chapter {chapter_number} continuation")
        except ResponseValidationError as e:
            logger.warning(f"Validation warning for continuation: {e}")

        # Append continuation to chapter content
        chapter.content = chapter.content + "\n\n" + continuation
        chapter.word_count = len(chapter.content.split())

        # Auto-save
        try:
            self.save_story()
            logger.debug(f"Auto-saved after continuing chapter {chapter_number}")
        except (OSError, PermissionError) as e:
            error_msg = f"Failed to save continued chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e
        except (ValueError, TypeError) as e:
            error_msg = f"Failed to serialize continued chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error saving continued chapter {chapter_number}: {e}"
            logger.exception(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(error_msg) from e

        self._emit(
            "agent_complete",
            "Writer",
            f"Chapter {chapter_number} continued (+{len(continuation.split())} words)",
        )
        yield self.events[-1]

        return continuation

    def edit_passage(
        self,
        text: str,
        focus: str | None = None,
    ) -> Generator[WorkflowEvent, None, str]:
        """Edit a specific passage with optional focus area.

        Args:
            text: The text passage to edit.
            focus: Optional focus area (e.g., "dialogue", "pacing", "description").

        Returns:
            The edited passage.
        """
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        brief = self.story_state.brief
        language = brief.language if brief else "English"

        focus_msg = f" (focus: {focus})" if focus else ""
        self._emit("agent_start", "Editor", f"Editing passage{focus_msg}...")
        yield self.events[-1]

        # Use EditorAgent.edit_passage() for targeted editing
        edited = self.editor.edit_passage(text, focus=focus, language=language)

        self._emit(
            "agent_complete",
            "Editor",
            f"Passage edited ({len(edited.split())} words)",
        )
        yield self.events[-1]

        return edited

    def get_edit_suggestions(
        self,
        text: str,
    ) -> Generator[WorkflowEvent, None, str]:
        """Get editing suggestions without making changes.

        Args:
            text: The text to review.

        Returns:
            Suggestions for improving the text.
        """
        self._emit("agent_start", "Editor", "Analyzing text for suggestions...")
        yield self.events[-1]

        # Use EditorAgent.get_edit_suggestions() for review mode
        suggestions = self.editor.get_edit_suggestions(text)

        self._emit(
            "agent_complete",
            "Editor",
            "Edit suggestions ready",
            {"suggestions": suggestions},
        )
        yield self.events[-1]

        return suggestions

    def review_full_story(
        self,
    ) -> Generator[WorkflowEvent, None, list[ContinuityIssue]]:
        """Perform a full story continuity review.

        Returns:
            List of ContinuityIssue objects.
        """
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        self._emit("agent_start", "Continuity", "Reviewing complete story...")
        yield self.events[-1]

        issues = self.continuity.check_full_story(self.story_state)

        if issues:
            feedback = self.continuity.format_revision_feedback(issues)
            critical_count = sum(1 for i in issues if i.severity == "critical")
            moderate_count = sum(1 for i in issues if i.severity == "moderate")
            minor_count = sum(1 for i in issues if i.severity == "minor")

            self._emit(
                "agent_complete",
                "Continuity",
                f"Found {len(issues)} issues: {critical_count} critical, {moderate_count} moderate, {minor_count} minor",
                {"issues": [vars(i) for i in issues], "feedback": feedback},
            )
        else:
            self._emit(
                "agent_complete",
                "Continuity",
                "No continuity issues found!",
            )

        yield self.events[-1]
        return issues
