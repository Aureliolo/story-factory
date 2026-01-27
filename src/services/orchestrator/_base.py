"""Base class and core functionality for StoryOrchestrator."""

import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

from src.agents import (
    ArchitectAgent,
    ContinuityAgent,
    EditorAgent,
    InterviewerAgent,
    ResponseValidationError,
    ValidatorAgent,
    WriterAgent,
)
from src.memory.story_state import StoryState
from src.settings import Settings

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


class StoryOrchestratorBase:
    """Base class for story orchestrator with initialization and core functionality."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_override: str | None = None,  # Force specific model for all agents
        mode_service: ModelModeService | None = None,  # ModelModeService for learning hooks
    ):
        """Create a StoryOrchestrator and initialize agents, persistent state, and progress tracking.

        Parameters:
            settings (Settings | None): Application settings; when None the default settings are loaded.
            model_override (str | None): Model identifier to force for all agents; when None agents use their configured models.
            mode_service (ModelModeService | None): Optional ModelModeService instance for adaptive learning hooks.
        """
        self.settings = settings or Settings.load()
        self.model_override = model_override
        self.mode_service = mode_service  # For learning hooks

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

        # Progress tracking
        self._current_phase: str = "interview"
        self._phase_start_time: datetime | None = None
        self._total_chapters: int = 0
        self._completed_chapters: int = 0
        self._current_score_id: int | None = None  # For learning tracking

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

            db = ModeDatabase()

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

    def reset_state(self) -> None:
        """Reset the orchestrator state for a new story."""
        self.story_state = None
        self.events.clear()
        # Reset agent conversation histories
        self.interviewer.conversation_history = []
        logger.info("Orchestrator state reset")
