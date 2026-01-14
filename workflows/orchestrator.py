"""Main orchestrator that coordinates all agents."""

import json
import logging
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agents import (
    ArchitectAgent,
    ContinuityAgent,
    EditorAgent,
    InterviewerAgent,
    WriterAgent,
)
from memory.story_state import Chapter, StoryBrief, StoryState
from settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class WorkflowEvent:
    """An event in the workflow for UI updates."""

    event_type: str  # "agent_start", "agent_complete", "user_input_needed", "progress", "error"
    agent_name: str
    message: str
    data: dict = None


# Maximum events to keep in memory to prevent unbounded growth
MAX_EVENTS = 100


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
        self.story_state: StoryState | None = None
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
        # Trim old events to prevent memory leak
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]
        return event

    def clear_events(self):
        """Clear all events (call after story completion if needed)."""
        self.events.clear()

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
            f"{h['role']}: {h['content']}" for h in self.interviewer.conversation_history
        )
        brief = self.interviewer.finalize_brief(history)
        self.story_state.brief = brief
        self.story_state.status = "outlining"
        return brief

    # ========== ARCHITECTURE PHASE ==========

    def build_story_structure(self) -> StoryState:
        """Have the architect build the story structure."""
        logger.info("Building story structure...")
        self._emit("agent_start", "Architect", "Building world...")

        logger.info(f"Calling architect with model: {self.architect.model}")
        self.story_state = self.architect.build_story_structure(self.story_state)

        logger.info(
            f"Structure built: {len(self.story_state.chapters)} chapters, {len(self.story_state.characters)} characters"
        )
        self._emit("agent_complete", "Architect", "Story structure complete!")
        return self.story_state

    def get_outline_summary(self) -> str:
        """Get a human-readable summary of the story outline."""
        state = self.story_state
        summary_parts = [
            "=" * 50,
            "STORY OUTLINE",
            "=" * 50,
        ]

        # Handle missing brief (old saved stories may not have it)
        if state.brief:
            summary_parts.extend(
                [
                    f"\nPREMISE: {state.brief.premise}",
                    f"GENRE: {state.brief.genre}",
                    f"TONE: {state.brief.tone}",
                    f"NSFW LEVEL: {state.brief.nsfw_level}",
                ]
            )
        else:
            summary_parts.append("\n(No brief available)")

        if state.world_description:
            summary_parts.append(f"\nWORLD:\n{state.world_description[:500]}...")

        summary_parts.append("\nCHARACTERS:")

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
        """Write a short story with revision loop."""
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
        except Exception as e:
            logger.warning(f"Auto-save failed for short story: {e}")

        self._emit(
            "agent_complete", "System", f"Story complete! ({short_story_chapter.word_count} words)"
        )
        yield self.events[-1]

        return short_story_chapter.content

    def write_chapter(self, chapter_number: int) -> Generator[WorkflowEvent, None, str]:
        """Write a single chapter with the full pipeline."""
        chapter = next((c for c in self.story_state.chapters if c.number == chapter_number), None)
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

        # Auto-save after each chapter to prevent data loss
        try:
            self.save_story()
            logger.debug(f"Auto-saved after chapter {chapter_number}")
        except Exception as e:
            logger.warning(f"Auto-save failed after chapter {chapter_number}: {e}")

        self._emit(
            "agent_complete",
            "System",
            f"Chapter {chapter_number} complete ({chapter.word_count} words)",
        )
        yield self.events[-1]

        return chapter.content

    def write_all_chapters(
        self,
        on_checkpoint: Callable[[int, str], bool] = None,
    ) -> Generator[WorkflowEvent]:
        """Write all chapters, with optional checkpoints for user feedback.

        Args:
            on_checkpoint: Callback at checkpoints. Returns True to continue, False to pause.
        """
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

    def export_to_text(self) -> str:
        """Export the story as plain text."""
        brief = self.story_state.brief
        text_parts = [
            brief.premise[:80],
            f"Genre: {brief.genre} | Tone: {brief.tone}",
            f"Setting: {brief.setting_place}, {brief.setting_time}",
            "=" * 60,
            "",
        ]

        for chapter in self.story_state.chapters:
            if chapter.content:
                text_parts.append(f"CHAPTER {chapter.number}: {chapter.title.upper()}")
                text_parts.append("")
                text_parts.append(chapter.content)
                text_parts.append("")
                text_parts.append("-" * 40)
                text_parts.append("")

        return "\n".join(text_parts)

    def export_story_to_file(self, format: str = "markdown", filepath: str = None) -> str:
        """Export the story to a file.

        Args:
            format: Export format ('markdown', 'text', 'json')
            filepath: Optional custom path. Defaults to output/stories/<story_id>.<ext>

        Returns:
            The path where the story was exported.
        """
        if not self.story_state:
            raise ValueError("No story to export")

        # Determine file extension and content
        if format == "markdown":
            ext = ".md"
            content = self.export_to_markdown()
        elif format == "text":
            ext = ".txt"
            content = self.export_to_text()
        elif format == "json":
            ext = ".json"
            # JSON export is handled by save_story
            return self.save_story(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Default export location
        if not filepath:
            output_dir = Path(__file__).parent.parent / "output" / "stories"
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"{self.story_state.id}{ext}"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Story exported to {filepath} ({format} format)")
        return str(filepath)

    def get_statistics(self) -> dict:
        """Get story statistics including reading time estimate."""
        total_words = sum(ch.word_count for ch in self.story_state.chapters)
        completed_chapters = sum(1 for ch in self.story_state.chapters if ch.status == "final")
        completed_plot_points = sum(1 for p in self.story_state.plot_points if p.completed)

        # Average reading speed: 200-250 words per minute
        reading_time_min = total_words / 225

        return {
            "total_words": total_words,
            "total_chapters": len(self.story_state.chapters),
            "completed_chapters": completed_chapters,
            "characters": len(self.story_state.characters),
            "established_facts": len(self.story_state.established_facts),
            "plot_points_total": len(self.story_state.plot_points),
            "plot_points_completed": completed_plot_points,
            "reading_time_minutes": round(reading_time_min, 1),
        }

    # ========== PERSISTENCE ==========

    def save_story(self, filepath: str = None) -> str:
        """Save the current story state to a JSON file.

        Args:
            filepath: Optional custom path. Defaults to output/stories/<story_id>.json

        Returns:
            The path where the story was saved.
        """
        if not self.story_state:
            raise ValueError("No story to save")

        # Default save location
        if not filepath:
            output_dir = Path(__file__).parent.parent / "output" / "stories"
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"{self.story_state.id}.json"

        filepath = Path(filepath)

        # Convert to dict for JSON serialization
        story_data = self.story_state.model_dump(mode="json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, default=str)

        logger.info(f"Story saved to {filepath}")
        return str(filepath)

    def load_story(self, filepath: str) -> StoryState:
        """Load a story state from a JSON file.

        Args:
            filepath: Path to the story JSON file.

        Returns:
            The loaded StoryState.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Story file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            story_data = json.load(f)

        self.story_state = StoryState.model_validate(story_data)
        logger.info(f"Story loaded from {filepath}")
        return self.story_state

    @staticmethod
    def list_saved_stories() -> list[dict]:
        """List all saved stories in the output directory.

        Returns:
            List of dicts with story metadata (id, path, created_at, status, etc.)
        """
        output_dir = Path(__file__).parent.parent / "output" / "stories"
        stories = []

        if not output_dir.exists():
            return stories

        for filepath in output_dir.glob("*.json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                stories.append(
                    {
                        "id": data.get("id"),
                        "path": str(filepath),
                        "created_at": data.get("created_at"),
                        "status": data.get("status"),
                        "premise": (
                            data.get("brief", {}).get("premise", "")[:100]
                            if data.get("brief")
                            else ""
                        ),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not read story file {filepath}: {e}")

        return sorted(stories, key=lambda x: x.get("created_at", ""), reverse=True)

    def reset_state(self):
        """Reset the orchestrator state for a new story."""
        self.story_state = None
        self.events.clear()
        # Reset agent conversation histories
        self.interviewer.conversation_history = []
        logger.info("Orchestrator state reset")
