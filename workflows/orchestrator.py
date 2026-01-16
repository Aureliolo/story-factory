"""Main orchestrator that coordinates all agents."""

import json
import logging
import uuid
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import (
    ArchitectAgent,
    ContinuityAgent,
    EditorAgent,
    InterviewerAgent,
    ResponseValidationError,
    ValidatorAgent,
    WriterAgent,
)
from agents.continuity import ContinuityIssue
from memory.story_state import Chapter, Character, StoryBrief, StoryState
from settings import STORIES_DIR, Settings
from utils.constants import get_language_code
from utils.exceptions import ExportError
from utils.message_analyzer import analyze_message, format_inference_context

logger = logging.getLogger(__name__)


@dataclass
class WorkflowEvent:
    """An event in the workflow for UI updates."""

    event_type: str  # "agent_start", "agent_complete", "user_input_needed", "progress", "error"
    agent_name: str
    message: str
    data: dict[str, Any] | None = None
    timestamp: datetime | None = None  # When the event occurred
    correlation_id: str | None = None  # For tracking related events


class StoryOrchestrator:
    """Orchestrates the story generation workflow."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_override: str | None = None,  # Force specific model for all agents
    ):
        self.settings = settings or Settings.load()
        self.model_override = model_override

        # Initialize agents with settings
        self.interviewer = InterviewerAgent(model=model_override, settings=self.settings)
        self.architect = ArchitectAgent(model=model_override, settings=self.settings)
        self.writer = WriterAgent(model=model_override, settings=self.settings)
        self.editor = EditorAgent(model=model_override, settings=self.settings)
        self.continuity = ContinuityAgent(model=model_override, settings=self.settings)
        self.validator = ValidatorAgent(settings=self.settings)  # Uses small model

        # State
        self.story_state: StoryState | None = None
        # Use deque with maxlen to prevent unbounded memory growth
        self.events: deque[WorkflowEvent] = deque(maxlen=self.settings.workflow_max_events)
        self._correlation_id: str | None = None  # Current workflow correlation ID

    def _validate_response(self, response: str, task: str = "") -> str:
        """Validate an AI response for language and basic correctness.

        Args:
            response: The AI-generated response
            task: Description of what the response should contain

        Returns:
            The response if valid

        Raises:
            ResponseValidationError: If validation fails
        """
        if not self.story_state or not self.story_state.brief:
            return response  # Can't validate without knowing expected language

        language = self.story_state.brief.language
        try:
            self.validator.validate_response(response, language, task)
            return response
        except ResponseValidationError as e:
            logger.error(f"Response validation failed: {e}")
            raise

    @property
    def interaction_mode(self) -> str:
        return self.settings.interaction_mode

    def create_new_story(self) -> StoryState:
        """Initialize a new story with a default project name."""
        now = datetime.now()
        default_name = f"New Story - {now.strftime('%b %d, %Y %I:%M %p')}"

        story_id = str(uuid.uuid4())
        self.story_state = StoryState(
            id=story_id,
            created_at=now,
            project_name=default_name,
            status="interview",
        )
        # Set correlation ID for event tracking.
        # Use first 8 chars of the UUID story ID for readability while maintaining
        # sufficient uniqueness within a single workflow/session.
        self._correlation_id = story_id[:8]
        # Autosave immediately so it appears in project list
        self.autosave()
        return self.story_state

    def update_project_name(self, name: str) -> None:
        """Update the project name."""
        if self.story_state:
            self.story_state.project_name = name
            self.autosave()

    def generate_title_suggestions(self) -> list[str]:
        """Generate AI-powered title suggestions based on the story content."""
        logger.info("Generating title suggestions...")

        if not self.story_state:
            logger.warning("No story state for title generation")
            return []

        # Build context for title generation
        context_parts = []
        if self.story_state.brief:
            brief = self.story_state.brief
            context_parts.append(f"Premise: {brief.premise}")
            context_parts.append(f"Genre: {brief.genre}")
            context_parts.append(f"Tone: {brief.tone}")
            if brief.themes:
                context_parts.append(f"Themes: {', '.join(brief.themes)}")

        if not context_parts:
            logger.warning("No brief data for title generation, using fallbacks")
            return []

        context = "\n".join(context_parts)
        logger.debug(f"Title generation context: {context[:200]}...")

        prompt = f"""Based on this story concept, generate exactly 5 creative and evocative title suggestions.
Each title should be 2-6 words, memorable, and capture the essence of the story.

Story concept:
{context}

Return ONLY a JSON array of 5 title strings, nothing else.
Example format: ["Title One", "Title Two", "Title Three", "Title Four", "Title Five"]"""

        try:
            from utils.json_parser import extract_json_list

            logger.info("Calling AI for title suggestions...")
            response = self.interviewer.generate(prompt, "", temperature=0.9)
            logger.debug(f"Title generation response: {response[:200]}...")

            titles = extract_json_list(response)
            if titles and isinstance(titles, list):
                result = [str(t) for t in titles[:5]]
                logger.info(f"Generated {len(result)} title suggestions: {result}")
                return result
            else:
                logger.warning(f"Failed to parse titles from response: {response[:100]}...")
        except Exception as e:
            logger.exception(f"Failed to generate title suggestions: {e}")

        # Return empty list on failure - UI will handle fallback message
        return []

    def _emit(
        self, event_type: str, agent: str, message: str, data: dict[str, Any] | None = None
    ) -> WorkflowEvent:
        """Emit a workflow event with timestamp and correlation ID."""
        event = WorkflowEvent(
            event_type=event_type,
            agent_name=agent,
            message=message,
            data=data or {},
            timestamp=datetime.now(),
            correlation_id=self._correlation_id,
        )
        self.events.append(event)
        # Deque automatically trims old events when maxlen is reached
        return event

    def clear_events(self) -> None:
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
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        self._emit("agent_start", "Interviewer", "Processing your response...")

        # Analyze the user message to infer language and content rating
        analysis = analyze_message(user_response)
        context = format_inference_context(analysis)

        response = self.interviewer.process_response(user_response, context=context)

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
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

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
        if not self.story_state:
            raise ValueError("No story state. Call create_new_story() first.")

        logger.info("Building story structure...")
        self._emit("agent_start", "Architect", "Building world...")

        logger.info(f"Calling architect with model: {self.architect.model}")
        self.story_state = self.architect.build_story_structure(self.story_state)

        # Validate key outputs for language correctness
        try:
            if self.story_state.world_description:
                self._validate_response(self.story_state.world_description, "World description")
            if self.story_state.plot_summary:
                self._validate_response(self.story_state.plot_summary, "Plot summary")
        except ResponseValidationError as e:
            logger.warning(f"Validation warning during structure build: {e}")
            # Don't block on validation errors, just log them

        logger.info(
            f"Structure built: {len(self.story_state.chapters)} chapters, {len(self.story_state.characters)} characters"
        )
        self._emit("agent_complete", "Architect", "Story structure complete!")
        return self.story_state

    def generate_more_characters(self, count: int = 2) -> list[Character]:
        """Generate additional characters for the story.

        Args:
            count: Number of characters to generate.

        Returns:
            List of new Character objects.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info(f"Generating {count} more characters...")
        self._emit("agent_start", "Architect", f"Generating {count} new characters...")

        existing_names = [c.name for c in self.story_state.characters]
        new_characters = self.architect.generate_more_characters(
            self.story_state, existing_names, count
        )

        # Add to story state
        self.story_state.characters.extend(new_characters)

        self._emit(
            "agent_complete",
            "Architect",
            f"Generated {len(new_characters)} new characters!",
        )
        return new_characters

    def generate_locations(self, count: int = 3) -> list[dict[str, Any]]:
        """Generate locations for the story world.

        Args:
            count: Number of locations to generate.

        Returns:
            List of location dictionaries.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info(f"Generating {count} locations...")
        self._emit("agent_start", "Architect", f"Generating {count} new locations...")

        # Get existing location names from world_description heuristic
        existing_locations: list[str] = []
        # Locations will be added to world database by the caller

        locations = self.architect.generate_locations(self.story_state, existing_locations, count)

        self._emit(
            "agent_complete",
            "Architect",
            f"Generated {len(locations)} new locations!",
        )
        return locations

    def generate_relationships(
        self, entity_names: list[str], existing_rels: list[tuple[str, str]], count: int = 5
    ) -> list[dict[str, Any]]:
        """Generate relationships between entities.

        Args:
            entity_names: Names of all entities that can have relationships.
            existing_rels: List of (source, target) tuples to avoid duplicates.
            count: Number of relationships to generate.

        Returns:
            List of relationship dictionaries.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info(f"Generating {count} relationships...")
        self._emit("agent_start", "Architect", f"Generating {count} new relationships...")

        relationships = self.architect.generate_relationships(
            self.story_state, entity_names, existing_rels, count
        )

        self._emit(
            "agent_complete",
            "Architect",
            f"Generated {len(relationships)} new relationships!",
        )
        return relationships

    def rebuild_world(self) -> StoryState:
        """Rebuild the entire world from scratch.

        This regenerates world description, characters, plot, and chapters.
        Use with caution if chapters have already been written.

        Returns:
            Updated StoryState.
        """
        if not self.story_state:
            raise ValueError("No story state. Create a story first.")

        logger.info("Rebuilding entire world...")
        self._emit("agent_start", "Architect", "Rebuilding world from scratch...")

        # Clear existing content but keep the brief
        self.story_state.world_description = ""
        self.story_state.world_rules = []
        self.story_state.characters = []
        self.story_state.plot_summary = ""
        self.story_state.plot_points = []
        self.story_state.chapters = []

        # Rebuild everything
        self.story_state = self.architect.build_story_structure(self.story_state)

        self._emit("agent_complete", "Architect", "World rebuilt successfully!")
        return self.story_state

    def get_outline_summary(self) -> str:
        """Get a human-readable summary of the story outline."""
        if not self.story_state:
            raise ValueError("No story state available.")

        state = self.story_state
        summary_parts = [
            "=" * 50,
            "STORY OUTLINE",
            "=" * 50,
        ]

        # Handle projects created before brief feature was added
        if state.brief:
            summary_parts.extend(
                [
                    f"\nPREMISE: {state.brief.premise}",
                    f"GENRE: {state.brief.genre}",
                    f"TONE: {state.brief.tone}",
                    f"CONTENT RATING: {state.brief.content_rating}",
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
        except (IOError, OSError, PermissionError) as e:
            error_msg = f"Failed to save short story: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Cannot save story: {e}") from e
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_msg = f"Failed to serialize short story: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Cannot serialize story: {e}") from e
        except Exception as e:
            error_msg = f"Unexpected error saving short story: {e}"
            logger.exception(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Unexpected save error: {e}") from e

        self._emit(
            "agent_complete", "System", f"Story complete! ({short_story_chapter.word_count} words)"
        )
        yield self.events[-1]

        return short_story_chapter.content

    def write_chapter(self, chapter_number: int) -> Generator[WorkflowEvent, None, str]:
        """Write a single chapter with the full pipeline."""
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

        # Write
        self._emit("agent_start", "Writer", f"Writing Chapter {chapter_number}...")
        yield self.events[-1]

        content = self.writer.write_chapter(self.story_state, chapter)

        # Validate language/correctness
        try:
            self._validate_response(content, f"Chapter {chapter_number} content")
        except ResponseValidationError as e:
            logger.warning(f"Validation warning for chapter {chapter_number}: {e}")

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
        except (IOError, OSError, PermissionError) as e:
            error_msg = f"Failed to save chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Cannot save chapter: {e}") from e
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_msg = f"Failed to serialize chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Cannot serialize chapter: {e}") from e
        except Exception as e:
            error_msg = f"Unexpected error saving chapter {chapter_number}: {e}"
            logger.exception(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Unexpected save error: {e}") from e

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

    # ========== CONTINUATION & EDITING ==========

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
        except (IOError, OSError, PermissionError) as e:
            error_msg = f"Failed to save continued chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Cannot save chapter continuation: {e}") from e
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_msg = f"Failed to serialize continued chapter {chapter_number}: {e}"
            logger.error(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Cannot serialize chapter continuation: {e}") from e
        except Exception as e:
            error_msg = f"Unexpected error saving continued chapter {chapter_number}: {e}"
            logger.exception(error_msg)
            self._emit("error", "System", error_msg)
            raise ExportError(f"Unexpected save error: {e}") from e

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

    # ========== OUTPUT ==========

    def get_full_story(self) -> str:
        """Get the complete story text."""
        if not self.story_state:
            raise ValueError("No story state available.")

        parts = []
        for chapter in self.story_state.chapters:
            if chapter.content:
                parts.append(f"# Chapter {chapter.number}: {chapter.title}\n\n{chapter.content}")
        return "\n\n---\n\n".join(parts)

    def export_to_markdown(self) -> str:
        """Export the story as markdown."""
        if not self.story_state:
            raise ValueError("No story state available.")

        brief = self.story_state.brief
        md_parts = []

        if brief:
            md_parts.extend(
                [
                    f"# {brief.premise[:50]}...\n",
                    f"*Genre: {brief.genre} | Tone: {brief.tone}*\n",
                    f"*Setting: {brief.setting_place}, {brief.setting_time}*\n",
                    "---\n",
                ]
            )
        else:
            md_parts.append("# Untitled Story\n\n---\n")

        for chapter in self.story_state.chapters:
            if chapter.content:
                md_parts.append(f"\n## Chapter {chapter.number}: {chapter.title}\n\n")
                md_parts.append(chapter.content)

        return "\n".join(md_parts)

    def export_to_text(self) -> str:
        """Export the story as plain text."""
        if not self.story_state:
            raise ValueError("No story state available.")

        brief = self.story_state.brief
        text_parts = []

        if brief:
            text_parts.extend(
                [
                    brief.premise[:80],
                    f"Genre: {brief.genre} | Tone: {brief.tone}",
                    f"Setting: {brief.setting_place}, {brief.setting_time}",
                    "=" * 60,
                    "",
                ]
            )
        else:
            text_parts.extend(["Untitled Story", "=" * 60, ""])

        for chapter in self.story_state.chapters:
            if chapter.content:
                text_parts.append(f"CHAPTER {chapter.number}: {chapter.title.upper()}")
                text_parts.append("")
                text_parts.append(chapter.content)
                text_parts.append("")
                text_parts.append("-" * 40)
                text_parts.append("")

        return "\n".join(text_parts)

    def export_to_epub(self) -> bytes:
        """Export the story as EPUB e-book format."""
        if not self.story_state:
            raise ValueError("No story state available.")

        from ebooklib import epub

        book = epub.EpubBook()

        # Set metadata
        brief = self.story_state.brief
        title = self.story_state.project_name or (brief.premise[:50] if brief else "Untitled Story")
        lang_code = get_language_code(brief.language) if brief else "en"
        book.set_identifier(self.story_state.id)
        book.set_title(title)
        book.set_language(lang_code)

        if brief:
            book.add_metadata("DC", "description", brief.premise)
            book.add_metadata("DC", "subject", brief.genre)

        # Create chapters
        chapters = []
        for ch in self.story_state.chapters:
            if ch.content:
                epub_chapter = epub.EpubHtml(
                    title=f"Chapter {ch.number}: {ch.title}",
                    file_name=f"chapter_{ch.number}.xhtml",
                    lang=lang_code,
                )
                # Convert content to HTML (simple paragraph wrapping)
                html_content = "<br/><br/>".join(
                    f"<p>{para}</p>" for para in ch.content.split("\n\n") if para.strip()
                )
                epub_chapter.content = f"<h1>Chapter {ch.number}: {ch.title}</h1>{html_content}"
                book.add_item(epub_chapter)
                chapters.append(epub_chapter)

        # Add navigation
        book.toc = tuple(chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Define spine
        book.spine = ["nav"] + chapters

        # Write to bytes
        from io import BytesIO

        output = BytesIO()
        epub.write_epub(output, book)
        return output.getvalue()

    def export_to_pdf(self) -> bytes:
        """Export the story as PDF format."""
        if not self.story_state:
            raise ValueError("No story state available.")

        from io import BytesIO

        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
        )
        chapter_style = ParagraphStyle(
            "ChapterTitle",
            parent=styles["Heading2"],
            fontSize=18,
            spaceAfter=20,
            spaceBefore=30,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            spaceAfter=12,
        )

        story_elements: list[Flowable] = []

        # Title page
        brief = self.story_state.brief
        title = self.story_state.project_name or (brief.premise[:50] if brief else "Untitled Story")
        story_elements.append(Paragraph(title, title_style))

        if brief:
            story_elements.append(
                Paragraph(f"<i>{brief.genre} | {brief.tone}</i>", styles["Normal"])
            )
            story_elements.append(Spacer(1, 0.5 * inch))

        story_elements.append(PageBreak())

        # Chapters
        for ch in self.story_state.chapters:
            if ch.content:
                story_elements.append(Paragraph(f"Chapter {ch.number}: {ch.title}", chapter_style))

                # Split content into paragraphs
                for para in ch.content.split("\n\n"):
                    if para.strip():
                        # Escape special characters for reportlab
                        safe_para = (
                            para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        )
                        story_elements.append(Paragraph(safe_para, body_style))

                story_elements.append(PageBreak())

        doc.build(story_elements)
        return buffer.getvalue()

    def export_to_mobi(self) -> bytes:
        """Export the story as MOBI format (Kindle).

        Note: Amazon discontinued MOBI support in March 2025. EPUB is now the
        recommended format for Kindle devices.
        """
        raise RuntimeError(
            "MOBI format is no longer supported.\n\n"
            "Amazon discontinued MOBI in March 2025. Use EPUB instead:\n"
            "• Send EPUB to your Kindle via 'Send to Kindle' email (yourname@kindle.com)\n"
            "• Use the Kindle app on your phone/tablet - it supports EPUB directly\n"
            "• Use Calibre to convert EPUB to AZW3 if needed"
        )

    def export_story_to_file(self, format: str = "markdown", filepath: str | None = None) -> str:
        """Export the story to a file.

        Args:
            format: Export format ('markdown', 'text', 'json', 'epub', 'pdf')
            filepath: Optional custom path. Defaults to output/stories/<story_id>.<ext>

        Returns:
            The path where the story was exported.
        """
        if not self.story_state:
            raise ValueError("No story to export")

        # Determine file extension and content
        content: str | bytes
        if format == "markdown":
            ext = ".md"
            content = self.export_to_markdown()
        elif format == "text":
            ext = ".txt"
            content = self.export_to_text()
        elif format == "json":
            # JSON export is handled by save_story
            return self.save_story(filepath)
        elif format == "epub":
            ext = ".epub"
            content = self.export_to_epub()
        elif format == "pdf":
            ext = ".pdf"
            content = self.export_to_pdf()
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Default export location
        output_path: Path
        if not filepath:
            output_dir = STORIES_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.story_state.id}{ext}"
        else:
            output_path = Path(filepath)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use isinstance for proper type narrowing that mypy understands
        if isinstance(content, bytes):
            with open(output_path, "wb") as f:
                f.write(content)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        logger.info(f"Story exported to {output_path} ({format} format)")
        return str(output_path)

    def get_statistics(self) -> dict[str, int | float]:
        """Get story statistics including reading time estimate."""
        if not self.story_state:
            raise ValueError("No story state available.")

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

    def autosave(self) -> str | None:
        """Auto-save current state with timestamp update.

        Returns:
            The path where saved, or None if no story to save.
        """
        if not self.story_state:
            return None

        try:
            self.story_state.last_saved = datetime.now()
            self.story_state.updated_at = datetime.now()
            path = self.save_story()
            logger.debug(f"Autosaved story to {path}")
            return path
        except Exception as e:
            logger.warning(f"Autosave failed: {e}")
            return None

    def save_story(self, filepath: str | None = None) -> str:
        """Save the current story state to a JSON file.

        Args:
            filepath: Optional custom path. Defaults to output/stories/<story_id>.json

        Returns:
            The path where the story was saved.
        """
        if not self.story_state:
            raise ValueError("No story to save")

        # Update timestamps
        self.story_state.updated_at = datetime.now()
        if not self.story_state.last_saved:
            self.story_state.last_saved = datetime.now()

        # Default save location
        output_path: Path
        if not filepath:
            output_dir = STORIES_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.story_state.id}.json"
        else:
            output_path = Path(filepath)

        # Convert to dict for JSON serialization
        story_data = self.story_state.model_dump(mode="json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, default=str)

        logger.info(f"Story saved to {output_path}")
        return str(output_path)

    def load_story(self, filepath: str) -> StoryState:
        """Load a story state from a JSON file.

        Args:
            filepath: Path to the story JSON file.

        Returns:
            The loaded StoryState.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Story file not found: {path}")

        with open(path, encoding="utf-8") as f:
            story_data = json.load(f)

        self.story_state = StoryState.model_validate(story_data)
        # Set correlation ID for event tracking
        self._correlation_id = self.story_state.id[:8]
        logger.info(f"Story loaded from {path}")
        return self.story_state

    @staticmethod
    def list_saved_stories() -> list[dict[str, str | None]]:
        """List all saved stories in the output directory.

        Returns:
            List of dicts with story metadata (id, path, created_at, status, etc.)
        """
        output_dir = STORIES_DIR
        stories: list[dict[str, str | None]] = []

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

        return sorted(stories, key=lambda x: x.get("created_at") or "", reverse=True)

    def reset_state(self) -> None:
        """Reset the orchestrator state for a new story."""
        self.story_state = None
        self.events.clear()
        # Reset agent conversation histories
        self.interviewer.conversation_history = []
        logger.info("Orchestrator state reset")
