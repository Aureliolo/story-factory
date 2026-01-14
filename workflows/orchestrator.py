"""Main orchestrator that coordinates all agents."""

import uuid
from datetime import datetime
from typing import Generator, Callable, Optional
from dataclasses import dataclass

from memory.story_state import StoryState, StoryBrief
from agents import (
    InterviewerAgent,
    ArchitectAgent,
    WriterAgent,
    EditorAgent,
    ContinuityAgent,
)
from settings import Settings


@dataclass
class WorkflowEvent:
    """An event in the workflow for UI updates."""
    event_type: str  # "agent_start", "agent_complete", "user_input_needed", "progress", "error"
    agent_name: str
    message: str
    data: dict = None


class StoryOrchestrator:
    """Orchestrates the story generation workflow."""

    def __init__(
        self,
        settings: Settings = None,
        model_override: str = None,  # Force specific model for all agents
    ):
        self.settings = settings or Settings.load()
        self.model_override = model_override

        # Initialize agents with settings
        self.interviewer = InterviewerAgent(model=model_override, settings=self.settings)
        self.architect = ArchitectAgent(model=model_override, settings=self.settings)
        self.writer = WriterAgent(model=model_override, settings=self.settings)
        self.editor = EditorAgent(model=model_override, settings=self.settings)
        self.continuity = ContinuityAgent(model=model_override, settings=self.settings)

        # State
        self.story_state: Optional[StoryState] = None
        self.events: list[WorkflowEvent] = []

    @property
    def interaction_mode(self):
        return self.settings.interaction_mode

    def create_new_story(self) -> StoryState:
        """Initialize a new story."""
        self.story_state = StoryState(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            status="interview",
        )
        return self.story_state

    def _emit(self, event_type: str, agent: str, message: str, data: dict = None):
        """Emit a workflow event."""
        event = WorkflowEvent(event_type, agent, message, data or {})
        self.events.append(event)
        return event

    # ========== INTERVIEW PHASE ==========

    def start_interview(self) -> str:
        """Start the interview process."""
        self._emit("agent_start", "Interviewer", "Starting interview...")
        questions = self.interviewer.get_initial_questions()
        self._emit("agent_complete", "Interviewer", "Initial questions ready")
        return questions

    def process_interview_response(self, user_response: str) -> tuple[str, bool]:
        """Process user response and return next questions or indicate completion.

        Returns: (response_text, is_complete)
        """
        self._emit("agent_start", "Interviewer", "Processing your response...")
        response = self.interviewer.process_response(user_response)

        # Check if a brief was generated
        brief = self.interviewer.extract_brief(response)
        if brief:
            self.story_state.brief = brief
            self.story_state.status = "outlining"
            self._emit("agent_complete", "Interviewer", "Story brief created!")
            return response, True

        self._emit("agent_complete", "Interviewer", "Follow-up questions ready")
        return response, False

    def finalize_interview(self) -> StoryBrief:
        """Force finalize the interview with current information."""
        history = "\n".join(
            f"{h['role']}: {h['content']}"
            for h in self.interviewer.conversation_history
        )
        brief = self.interviewer.finalize_brief(history)
        self.story_state.brief = brief
        self.story_state.status = "outlining"
        return brief

    # ========== ARCHITECTURE PHASE ==========

    def build_story_structure(self) -> StoryState:
        """Have the architect build the story structure."""
        self._emit("agent_start", "Architect", "Building world...")
        self.story_state = self.architect.build_story_structure(self.story_state)
        self._emit("agent_complete", "Architect", "Story structure complete!")
        return self.story_state

    def get_outline_summary(self) -> str:
        """Get a human-readable summary of the story outline."""
        state = self.story_state
        summary_parts = [
            "=" * 50,
            "STORY OUTLINE",
            "=" * 50,
            f"\nPREMISE: {state.brief.premise}",
            f"GENRE: {state.brief.genre}",
            f"TONE: {state.brief.tone}",
            f"NSFW LEVEL: {state.brief.nsfw_level}",
            f"\nWORLD:\n{state.world_description[:500]}...",
            f"\nCHARACTERS:",
        ]

        for char in state.characters:
            summary_parts.append(f"  - {char.name} ({char.role}): {char.description}")

        summary_parts.append(f"\nPLOT SUMMARY:\n{state.plot_summary}")

        summary_parts.append(f"\nCHAPTER OUTLINE ({len(state.chapters)} chapters):")
        for ch in state.chapters:
            summary_parts.append(f"  {ch.number}. {ch.title}")
            summary_parts.append(f"     {ch.outline[:100]}...")

        return "\n".join(summary_parts)

    # ========== WRITING PHASE ==========

    def write_short_story(self) -> Generator[WorkflowEvent, None, str]:
        """Write a short story (single pass)."""
        self._emit("agent_start", "Writer", "Writing story...")
        yield self.events[-1]

        content = self.writer.write_short_story(self.story_state)

        self._emit("agent_start", "Editor", "Editing story...")
        yield self.events[-1]

        edited_content = self.editor.edit_chapter(self.story_state, content)

        self._emit("agent_start", "Continuity", "Checking for issues...")
        yield self.events[-1]

        # Create a temporary chapter for checking
        self.story_state.chapters = [
            type(self.story_state.chapters[0] if self.story_state.chapters else object)(
                number=1, title="Complete Story", outline="", content=edited_content
            )
        ]

        issues = self.continuity.check_chapter(self.story_state, edited_content, 1)

        if issues and self.continuity.should_revise(issues):
            feedback = self.continuity.format_revision_feedback(issues)
            self._emit("progress", "Writer", f"Revising based on {len(issues)} issues...")
            yield self.events[-1]

            revised = self.writer.write_short_story(self.story_state)
            edited_content = self.editor.edit_chapter(self.story_state, revised)

        self.story_state.chapters[0].content = edited_content
        self.story_state.status = "complete"

        self._emit("agent_complete", "System", "Story complete!")
        yield self.events[-1]

        return edited_content

    def write_chapter(self, chapter_number: int) -> Generator[WorkflowEvent, None, str]:
        """Write a single chapter with the full pipeline."""
        chapter = next(
            (c for c in self.story_state.chapters if c.number == chapter_number),
            None
        )
        if not chapter:
            raise ValueError(f"Chapter {chapter_number} not found")

        chapter.status = "drafting"

        # Write
        self._emit("agent_start", "Writer", f"Writing Chapter {chapter_number}...")
        yield self.events[-1]

        content = self.writer.write_chapter(self.story_state, chapter)
        chapter.content = content

        # Edit
        self._emit("agent_start", "Editor", f"Editing Chapter {chapter_number}...")
        yield self.events[-1]

        edited_content = self.editor.edit_chapter(self.story_state, content)
        chapter.content = edited_content
        chapter.status = "edited"

        # Check continuity
        self._emit("agent_start", "Continuity", f"Checking Chapter {chapter_number}...")
        yield self.events[-1]

        revision_count = 0
        while revision_count < self.settings.max_revision_iterations:
            issues = self.continuity.check_chapter(
                self.story_state, chapter.content, chapter_number
            )

            if not issues or not self.continuity.should_revise(issues):
                break

            revision_count += 1
            feedback = self.continuity.format_revision_feedback(issues)
            chapter.revision_notes.append(feedback)

            self._emit(
                "progress",
                "Writer",
                f"Revision {revision_count}: Addressing {len(issues)} issues..."
            )
            yield self.events[-1]

            content = self.writer.write_chapter(
                self.story_state, chapter, revision_feedback=feedback
            )
            chapter.content = self.editor.edit_chapter(self.story_state, content)

        # Extract new facts
        new_facts = self.continuity.extract_new_facts(chapter.content, self.story_state)
        self.story_state.established_facts.extend(new_facts)

        chapter.status = "final"
        chapter.word_count = len(chapter.content.split())
        self.story_state.current_chapter = chapter_number

        self._emit(
            "agent_complete",
            "System",
            f"Chapter {chapter_number} complete ({chapter.word_count} words)"
        )
        yield self.events[-1]

        return chapter.content

    def write_all_chapters(
        self,
        on_checkpoint: Callable[[int, str], bool] = None,
    ) -> Generator[WorkflowEvent, None, None]:
        """Write all chapters, with optional checkpoints for user feedback.

        Args:
            on_checkpoint: Callback at checkpoints. Returns True to continue, False to pause.
        """
        for chapter in self.story_state.chapters:
            # Write the chapter
            for event in self.write_chapter(chapter.number):
                yield event

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
                    {"chapter": chapter.number, "content": chapter.content}
                )
                yield self.events[-1]

                # The UI will handle getting user input and calling continue

        self.story_state.status = "complete"
        self._emit("agent_complete", "System", "All chapters complete!")
        yield self.events[-1]

    # ========== OUTPUT ==========

    def get_full_story(self) -> str:
        """Get the complete story text."""
        parts = []
        for chapter in self.story_state.chapters:
            if chapter.content:
                parts.append(f"# Chapter {chapter.number}: {chapter.title}\n\n{chapter.content}")
        return "\n\n---\n\n".join(parts)

    def export_to_markdown(self) -> str:
        """Export the story as markdown."""
        brief = self.story_state.brief
        md_parts = [
            f"# {brief.premise[:50]}...\n",
            f"*Genre: {brief.genre} | Tone: {brief.tone}*\n",
            f"*Setting: {brief.setting_place}, {brief.setting_time}*\n",
            "---\n",
        ]

        for chapter in self.story_state.chapters:
            if chapter.content:
                md_parts.append(f"\n## Chapter {chapter.number}: {chapter.title}\n\n")
                md_parts.append(chapter.content)

        return "\n".join(md_parts)

    def get_statistics(self) -> dict:
        """Get story statistics."""
        total_words = sum(ch.word_count for ch in self.story_state.chapters)
        completed_chapters = sum(1 for ch in self.story_state.chapters if ch.status == "final")

        return {
            "total_words": total_words,
            "total_chapters": len(self.story_state.chapters),
            "completed_chapters": completed_chapters,
            "characters": len(self.story_state.characters),
            "established_facts": len(self.story_state.established_facts),
        }
