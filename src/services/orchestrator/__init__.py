"""Main orchestrator that coordinates all agents."""

import logging
import uuid
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.mode_database import ModeDatabase
    from src.memory.world_database import WorldDatabase
    from src.services.context_retrieval_service import ContextRetrievalService
    from src.services.model_mode_service import ModelModeService
    from src.services.timeline_service import TimelineService

from src.agents import (
    ArchitectAgent,
    ContinuityAgent,
    EditorAgent,
    InterviewerAgent,
    ResponseValidationError,
    ValidatorAgent,
    WriterAgent,
)
from src.agents.continuity import ContinuityIssue
from src.memory.story_state import Character, StoryBrief, StoryState
from src.settings import Settings

from . import _editing, _interview, _persistence, _structure, _writing

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
    # Progress tracking fields
    phase: str | None = (
        None  # Current phase: "interview", "architect", "writer", "editor", "continuity"
    )
    progress: float | None = None  # Overall progress 0.0-1.0
    chapter_number: int | None = None  # Current chapter being processed
    eta_seconds: float | None = None  # Estimated time remaining in seconds


class StoryOrchestrator:
    """Orchestrates the story generation workflow."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_override: str | None = None,  # Force specific model for all agents
        mode_service: ModelModeService | None = None,  # ModelModeService for learning hooks
        context_retrieval: ContextRetrievalService | None = None,
        timeline: TimelineService | None = None,
        mode_db: ModeDatabase | None = None,
    ):
        """Create a StoryOrchestrator and initialize agents, persistent state, and progress tracking.

        Parameters:
            settings (Settings | None): Application settings; when None the default settings are loaded.
            model_override (str | None): Model identifier to force for all agents; when None agents use their configured models.
            mode_service (ModelModeService | None): Optional ModelModeService instance for adaptive learning hooks.
            context_retrieval (ContextRetrievalService | None): Optional context retrieval service for RAG-based prompt enrichment.
            timeline (TimelineService | None): Optional timeline service for temporal context in prompts.
            mode_db (ModeDatabase | None): Shared ModeDatabase instance for ETA calculations. If None, creates one on demand.
        """
        self.settings = settings or Settings.load()
        self.model_override = model_override
        self.mode_service = mode_service  # For learning hooks
        self.context_retrieval = context_retrieval  # For RAG context injection
        self.timeline = timeline  # For temporal context in agent prompts
        self._mode_db = mode_db  # Shared ModeDatabase for ETA lookups
        self.world_db: WorldDatabase | None = None  # Set per-project by StoryService

        # Initialize agents with settings
        self.interviewer = InterviewerAgent(model=model_override, settings=self.settings)
        self.architect = ArchitectAgent(model=model_override, settings=self.settings)
        self.writer = WriterAgent(model=model_override, settings=self.settings)
        self.editor = EditorAgent(model=model_override, settings=self.settings)
        self.continuity = ContinuityAgent(model=model_override, settings=self.settings)
        self.validator = ValidatorAgent(settings=self.settings)  # Rule-based only

        # State
        self.story_state: StoryState | None = None
        # Use deque with maxlen to prevent unbounded memory growth
        self.events: deque[WorkflowEvent] = deque(maxlen=self.settings.workflow_max_events)
        self._correlation_id: str | None = None  # Current workflow correlation ID

        # Progress tracking
        self._current_phase: str = "interview"
        self._phase_start_time: datetime | None = None
        self._total_chapters: int = 0
        self._completed_chapters: int = 0
        self._current_score_id: int | None = None  # For learning tracking

    def set_project_context(
        self,
        world_db: WorldDatabase | None,
        story_state: StoryState | None,
    ) -> None:
        """Set project-specific context for the current writing session.

        Bundles the world database and story state assignment into a single
        explicit call.  Preferred over direct attribute mutation for callers
        that need to set both world_db and story_state.

        Args:
            world_db: WorldDatabase for RAG context retrieval (may be None).
            story_state: Story state to operate on (may be None when clearing).
        """
        self.world_db = world_db
        self.story_state = story_state
        logger.debug(
            "Project context set: world_db=%s, story_state=%s",
            "present" if world_db else "None",
            story_state.id[:8] if story_state else "None",
        )

    # ========== INTERNAL HELPERS (inline) ==========

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
            logger.error(
                "Response validation failed for task '%s' (language=%s, len=%d): %s",
                task,
                language,
                len(response),
                e,
            )
            raise

    @property
    def interaction_mode(self) -> str:
        """Get the current interaction mode from src.settings.

        Returns:
            The interaction mode (e.g., 'guided', 'autonomous').
        """
        return self.settings.interaction_mode

    def create_new_story(self) -> StoryState:
        """Initialize a new story with a default project name."""
        logger.debug("create_new_story called")
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
        logger.debug("update_project_name called: name=%s", name)
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
            from src.utils.json_parser import extract_json_list

            logger.info("Calling AI for title suggestions...")
            response = self.interviewer.generate(
                prompt, "", temperature=self.settings.temp_interviewer_override
            )
            logger.debug(f"Title generation response: {response[:200]}...")

            # Use strict=False since title suggestions are optional
            titles = extract_json_list(response, strict=False)
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
        """Emit a workflow event with timestamp, correlation ID, and progress info."""
        # Calculate progress and ETA
        progress = self._calculate_progress()
        eta = self._calculate_eta()
        chapter_num = self._completed_chapters + 1 if self._total_chapters > 0 else None

        event = WorkflowEvent(
            event_type=event_type,
            agent_name=agent,
            message=message,
            data=data or {},
            timestamp=datetime.now(),
            correlation_id=self._correlation_id,
            phase=self._current_phase,
            progress=progress,
            chapter_number=chapter_num,
            eta_seconds=eta,
        )
        self.events.append(event)
        # Deque automatically trims old events when maxlen is reached
        return event

    def _calculate_progress(self) -> float:
        """Calculate overall progress as a value between 0.0 and 1.0.

        Progress is calculated using weighted phases where the writer phase
        accounts for most of the work. For multi-chapter stories, the writer,
        editor, and continuity phases cycle per chapter.

        Returns:
            Progress value 0.0-1.0
        """
        if not self.story_state:
            logger.debug("No story state, returning 0.0 progress")
            return 0.0

        # Phase weights (sum to 1.0)
        # Note: writer/editor/continuity cycle per chapter, so their weights
        # are distributed across all chapters
        phase_weights = {
            "interview": 0.10,  # 10%
            "architect": 0.15,  # 15%
            "writer": 0.50,  # 50% (main work)
            "editor": 0.15,  # 15%
            "continuity": 0.10,  # 10%
        }

        # Calculate base progress for completed phases
        completed_phases = []
        if self.story_state.brief:
            completed_phases.append("interview")
        if self.story_state.chapters:
            completed_phases.append("architect")

        # If we're in a chapter-processing phase, interview and architect are definitely done
        if self._current_phase in ["writer", "editor", "continuity"]:
            if "interview" not in completed_phases:
                completed_phases.append("interview")
            if "architect" not in completed_phases:
                completed_phases.append("architect")

        base_progress = sum(phase_weights[p] for p in completed_phases)

        # Add progress within current phase
        current_weight = phase_weights.get(self._current_phase, 0.0)

        if self._current_phase in ["writer", "editor", "continuity"] and self._total_chapters > 0:
            # Progress through chapters - distribute phase weight across all chapters
            chapter_progress = self._completed_chapters / self._total_chapters
            base_progress += current_weight * chapter_progress
        elif self._current_phase not in completed_phases:
            # Phase in progress but not complete - estimate 50% done
            base_progress += current_weight * 0.5

        progress = min(1.0, base_progress)
        logger.debug(
            f"Progress: {progress:.1%} (phase={self._current_phase}, "
            f"chapters={self._completed_chapters}/{self._total_chapters})"
        )
        return progress

    def _calculate_eta(self) -> float | None:
        """Calculate estimated time remaining in seconds.

        Uses historical generation times from mode_database if available,
        otherwise returns None.

        Returns:
            Estimated seconds remaining, or None if cannot estimate
        """
        if not self.story_state or not self._phase_start_time:
            return None

        try:
            from src.memory.mode_database import ModeDatabase

            if self._mode_db is None:
                self._mode_db = ModeDatabase()
            db = self._mode_db

            # Get historical times for current model and role
            genre = self.story_state.brief.genre if self.story_state.brief else None

            # Estimate time per chapter based on agent role
            agent_role_map = {
                "writer": "writer",
                "editor": "editor",
                "continuity": "continuity",
            }

            if self._current_phase in agent_role_map and self._total_chapters > 0:
                role = agent_role_map[self._current_phase]

                # Get model ID from the appropriate agent
                model_id: str | None = None
                if role == "writer" and hasattr(self, "writer"):
                    model_id = getattr(self.writer, "model", None)
                elif role == "editor" and hasattr(self, "editor"):
                    model_id = getattr(self.editor, "model", None)
                elif role == "continuity" and hasattr(self, "continuity"):
                    model_id = getattr(self.continuity, "model", None)

                if not model_id:
                    logger.debug(f"No model found for role {role}, skipping historical ETA")
                    model_id = self.settings.default_model

                # Query historical performance
                perf_data = db.get_model_performance(
                    model_id=model_id,
                    agent_role=role,
                    genre=genre,
                )

                if perf_data:
                    # Use average time from historical data
                    avg_tokens_per_sec = float(perf_data[0].get("avg_tokens_per_second", 20.0))
                    # Estimate ~500-1000 tokens per chapter section
                    estimated_tokens_per_chapter = 750
                    time_per_chapter = estimated_tokens_per_chapter / avg_tokens_per_sec

                    remaining_chapters = self._total_chapters - self._completed_chapters
                    return float(remaining_chapters * time_per_chapter)

            # Fallback: estimate based on elapsed time
            elapsed = (datetime.now() - self._phase_start_time).total_seconds()
            if self._completed_chapters > 0:
                avg_time_per_chapter = elapsed / self._completed_chapters
                remaining_chapters = self._total_chapters - self._completed_chapters
                return remaining_chapters * avg_time_per_chapter

        except Exception as e:
            logger.debug(f"Could not calculate ETA: {e}")

        return None

    def _set_phase(self, phase: str) -> None:
        """Set the current workflow phase and reset phase timer.

        Args:
            phase: Phase name (interview, architect, writer, editor, continuity)
        """
        self._current_phase = phase
        self._phase_start_time = datetime.now()
        logger.debug(f"Phase transition: {phase}")

    def clear_events(self) -> None:
        """Clear all events (call after story completion if needed)."""
        self.events.clear()

    # ========== INTERVIEW PHASE (delegated) ==========

    def start_interview(self) -> str:
        """Start the interview process."""
        return _interview.start_interview(self)

    def process_interview_response(self, user_response: str) -> tuple[str, bool]:
        """Process user response and return next questions or indicate completion.

        Returns: (response_text, is_complete)
        """
        return _interview.process_interview_response(self, user_response)

    def finalize_interview(self) -> StoryBrief:
        """Force finalize the interview with current information."""
        return _interview.finalize_interview(self)

    # ========== ARCHITECTURE PHASE (delegated) ==========

    def build_story_structure(self) -> StoryState:
        """Have the architect build the story structure."""
        return _structure.build_story_structure(self)

    def generate_more_characters(self, count: int = 2) -> list[Character]:
        """Generate additional characters for the story.

        Args:
            count: Number of characters to generate.

        Returns:
            List of new Character objects.
        """
        return _structure.generate_more_characters(self, count)

    def generate_locations(self, count: int = 3) -> list[dict[str, Any]]:
        """Generate locations for the story world.

        Args:
            count: Number of locations to generate.

        Returns:
            List of location dictionaries.
        """
        return _structure.generate_locations(self, count)

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
        return _structure.generate_relationships(self, entity_names, existing_rels, count)

    def rebuild_world(self) -> StoryState:
        """Rebuild the entire world from scratch.

        Returns:
            Updated StoryState.
        """
        return _structure.rebuild_world(self)

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

    # ========== WRITING PHASE (delegated) ==========

    def write_short_story(self) -> Generator[WorkflowEvent, None, str]:
        """Write a short story with revision loop."""
        return (yield from _writing.write_short_story(self))

    def write_chapter(
        self, chapter_number: int, feedback: str | None = None
    ) -> Generator[WorkflowEvent, None, str]:
        """Run the full write-edit-continuity pipeline for a single chapter, yielding workflow events during processing.

        Parameters:
            chapter_number (int): The chapter number to process (must exist in the current story structure).
            feedback (str | None): Optional feedback or revision guidance to incorporate into the initial generation.

        Returns:
            chapter_content (str): The final content of the chapter after editing and revisions.
        """
        return (yield from _writing.write_chapter(self, chapter_number, feedback))

    def write_all_chapters(
        self,
        on_checkpoint: Callable[[int, str], bool] | None = None,
    ) -> Generator[WorkflowEvent]:
        """Write all chapters, with optional checkpoints for user feedback.

        Args:
            on_checkpoint: Callback at checkpoints. Returns True to continue, False to pause.
        """
        yield from _writing.write_all_chapters(self, on_checkpoint)

    # ========== CONTINUATION & EDITING (delegated) ==========

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
        return (yield from _editing.continue_chapter(self, chapter_number, direction))

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
        return (yield from _editing.edit_passage(self, text, focus))

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
        return (yield from _editing.get_edit_suggestions(self, text))

    def review_full_story(
        self,
    ) -> Generator[WorkflowEvent, None, list[ContinuityIssue]]:
        """Perform a full story continuity review.

        Returns:
            List of ContinuityIssue objects.
        """
        return (yield from _editing.review_full_story(self))

    # ========== OUTPUT (delegated) ==========

    def get_full_story(self) -> str:
        """Get the complete story text."""
        return _persistence.get_full_story(self)

    def export_to_markdown(self) -> str:
        """Export the story as markdown."""
        return _persistence.export_to_markdown(self)

    def export_to_text(self) -> str:
        """Export the story as plain text."""
        return _persistence.export_to_text(self)

    def export_to_epub(self) -> bytes:
        """Export the story as EPUB e-book.

        Returns:
            The EPUB file contents as bytes.
        """
        return _persistence.export_to_epub(self)

    def export_to_pdf(self) -> bytes:
        """Export the story as PDF format.

        Returns:
            The PDF file contents as bytes.
        """
        return _persistence.export_to_pdf(self)

    def export_to_mobi(self) -> bytes:
        """Export the story as MOBI format (Kindle).

        Note: Amazon discontinued MOBI support in March 2025. EPUB is now the
        recommended format for Kindle devices.
        """
        raise RuntimeError(
            "MOBI format is no longer supported.\n\n"
            "Amazon discontinued MOBI in March 2025. Use EPUB instead:\n"
            "- Send EPUB to your Kindle via 'Send to Kindle' email (yourname@kindle.com)\n"
            "- Use the Kindle app on your phone/tablet - it supports EPUB directly\n"
            "- Use Calibre to convert EPUB to AZW3 if needed"
        )

    def export_story_to_file(self, format: str = "markdown", filepath: str | None = None) -> str:
        """Export the story to a file.

        Args:
            format: Export format ('markdown', 'text', 'json', 'epub', 'pdf')
            filepath: Optional custom path. Defaults to output/stories/<story_id>.<ext>

        Returns:
            The path where the story was exported.
        """
        return _persistence.export_story_to_file(self, format, filepath)

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

    # ========== PERSISTENCE (delegated) ==========

    def autosave(self) -> str | None:
        """Auto-save current state with timestamp update.

        Returns:
            The path where saved, or None if no story to save.
        """
        return _persistence.autosave(self)

    def save_story(self, filepath: str | None = None) -> str:
        """Save the current story state to a JSON file.

        Args:
            filepath: Optional custom path. Defaults to output/stories/<story_id>.json

        Returns:
            The path where the story was saved.
        """
        return _persistence.save_story(self, filepath)

    def load_story(self, filepath: str) -> StoryState:
        """Load a story state from a JSON file.

        Args:
            filepath: Path to the story JSON file.

        Returns:
            The loaded StoryState.
        """
        return _persistence.load_story(self, filepath)

    @staticmethod
    def list_saved_stories() -> list[dict[str, str | None]]:
        """List all saved stories in the output directory.

        Returns:
            List of dicts with story metadata (id, path, created_at, status, etc.)
        """
        return _persistence.list_saved_stories()

    def reset_state(self) -> None:
        """Reset the orchestrator state for a new story."""
        self.story_state = None
        self.world_db = None
        self.events.clear()
        # Reset agent conversation histories
        self.interviewer.conversation_history = []
        logger.info("Orchestrator state reset")


__all__ = ["StoryOrchestrator", "WorkflowEvent"]
