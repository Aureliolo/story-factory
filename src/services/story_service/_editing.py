"""Editing and review mixin for StoryService."""

import logging
from collections.abc import Generator
from typing import Any

from src.agents.continuity import ContinuityIssue
from src.memory.story_state import StoryState
from src.services.orchestrator import WorkflowEvent

from ._base import StoryServiceBase

logger = logging.getLogger(__name__)


class EditingMixin(StoryServiceBase):
    """Mixin providing editing and review functionality."""

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
