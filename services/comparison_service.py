"""Comparison service - multi-model chapter comparison."""

import asyncio
import logging
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memory.story_state import StoryState
from settings import Settings
from utils.validation import validate_not_empty, validate_not_none, validate_type
from workflows.orchestrator import StoryOrchestrator, WorkflowEvent

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a single model's chapter generation."""

    model_id: str
    content: str
    word_count: int
    generation_time: float  # seconds
    events: list[WorkflowEvent] = field(default_factory=list)
    error: str | None = None


@dataclass
class ComparisonRecord:
    """Record of a comparison session."""

    id: str
    timestamp: datetime
    chapter_number: int
    models: list[str]
    results: dict[str, ComparisonResult]  # model_id -> result
    selected_model: str | None = None
    user_notes: str = ""


class ComparisonService:
    """Service for comparing chapter generation across multiple models.

    Handles:
    - Parallel generation with multiple models
    - Tracking comparison results
    - Recording user selections for analytics
    """

    def __init__(self, settings: Settings):
        """Initialize comparison service.

        Args:
            settings: Application settings.
        """
        validate_not_none(settings, "settings")
        validate_type(settings, "settings", Settings)
        logger.debug("Initializing ComparisonService")
        self.settings = settings
        self._comparison_history: list[ComparisonRecord] = []
        logger.debug("ComparisonService initialized successfully")

    def generate_chapter_comparison(
        self,
        state: StoryState,
        chapter_num: int,
        models: list[str],
        feedback: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Generator[dict[str, Any], None, ComparisonRecord]:
        """Generate a chapter with multiple models for comparison.

        Args:
            state: The story state.
            chapter_num: Chapter number to generate.
            models: List of model IDs to compare (2-4 models).
            feedback: Optional feedback to incorporate.
            cancel_check: Optional callable that returns True if cancellation requested.

        Yields:
            Progress events as dict with:
                - model_id: str
                - event: WorkflowEvent
                - progress: float (0.0 to 1.0)

        Returns:
            ComparisonRecord with all results.

        Raises:
            ValueError: If models list is invalid (empty, too many, or contains duplicates).
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        validate_not_empty(models, "models")

        if len(models) < 2:
            raise ValueError("At least 2 models required for comparison")
        if len(models) > 4:
            raise ValueError("Maximum 4 models allowed for comparison")
        if len(models) != len(set(models)):
            raise ValueError("Duplicate models in comparison list")

        logger.info(
            f"Starting comparison for chapter {chapter_num} with models: {', '.join(models)}"
        )

        # Create comparison record
        record = ComparisonRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            chapter_number=chapter_num,
            models=models.copy(),
            results={},
        )

        # Generate with each model sequentially
        # (Parallel would require thread-safe orchestrators, which is complex)
        for i, model_id in enumerate(models):
            if cancel_check and cancel_check():
                logger.info(f"Comparison cancelled at model {i + 1}/{len(models)}")
                break

            logger.debug(f"Generating chapter {chapter_num} with model: {model_id}")

            # Generate with this model
            result = self._generate_with_model(
                state=state,
                chapter_num=chapter_num,
                model_id=model_id,
                feedback=feedback,
                cancel_check=cancel_check,
                progress_callback=lambda event: {
                    "model_id": model_id,
                    "event": event,
                    "progress": (i + 0.5) / len(models),
                },
            )

            # Yield progress events
            for event_dict in result["events"]:
                yield event_dict

            # Store result
            record.results[model_id] = result["result"]

            # Yield completion for this model
            yield {
                "model_id": model_id,
                "event": None,
                "progress": (i + 1) / len(models),
                "completed": True,
            }

        # Add to history
        self._comparison_history.append(record)
        logger.info(
            f"Comparison complete: {len(record.results)}/{len(models)} models generated"
        )

        return record

    def _generate_with_model(
        self,
        state: StoryState,
        chapter_num: int,
        model_id: str,
        feedback: str | None,
        cancel_check: Callable[[], bool] | None,
        progress_callback: Callable[[WorkflowEvent], dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate chapter content with a specific model.

        Args:
            state: Story state.
            chapter_num: Chapter number.
            model_id: Model to use.
            feedback: Optional feedback.
            cancel_check: Cancellation check.
            progress_callback: Callback for progress events.

        Returns:
            Dict with:
                - result: ComparisonResult
                - events: List of event dicts for yielding
        """
        start_time = datetime.now()
        events_to_yield = []

        try:
            # Create a temporary orchestrator with this specific model
            orchestrator = StoryOrchestrator(settings=self.settings)
            orchestrator.story_state = state

            # Override model for writer agent
            original_model = self.settings.agent_models.get("writer", "auto")
            self.settings.agent_models["writer"] = model_id

            # Collect events
            collected_events = []
            content = ""

            try:
                for event in orchestrator.write_chapter(chapter_num):
                    if cancel_check and cancel_check():
                        break

                    collected_events.append(event)
                    events_to_yield.append(progress_callback(event))

                    # Extract content when chapter is complete
                    if event.event_type == "agent_complete" and event.agent_name == "System":
                        chapter = next(
                            (c for c in state.chapters if c.number == chapter_num), None
                        )
                        if chapter:
                            content = chapter.content

            finally:
                # Restore original model
                self.settings.agent_models["writer"] = original_model

            # Calculate metrics
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            word_count = len(content.split()) if content else 0

            result = ComparisonResult(
                model_id=model_id,
                content=content,
                word_count=word_count,
                generation_time=generation_time,
                events=collected_events,
            )

            logger.debug(
                f"Model {model_id} generated {word_count} words in {generation_time:.1f}s"
            )

            return {"result": result, "events": events_to_yield}

        except Exception as e:
            logger.error(f"Error generating with model {model_id}: {e}", exc_info=True)
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            result = ComparisonResult(
                model_id=model_id,
                content="",
                word_count=0,
                generation_time=generation_time,
                error=str(e),
            )

            return {"result": result, "events": events_to_yield}

    def select_winner(
        self, comparison_id: str, selected_model: str, user_notes: str = ""
    ) -> None:
        """Record user's selection from comparison.

        Args:
            comparison_id: ID of the comparison record.
            selected_model: Model ID selected as best.
            user_notes: Optional notes about selection.

        Raises:
            ValueError: If comparison not found or model not in comparison.
        """
        validate_not_empty(comparison_id, "comparison_id")
        validate_not_empty(selected_model, "selected_model")

        # Find comparison record
        record = next((r for r in self._comparison_history if r.id == comparison_id), None)
        if not record:
            raise ValueError(f"Comparison record not found: {comparison_id}")

        if selected_model not in record.results:
            raise ValueError(f"Model {selected_model} not in comparison results")

        record.selected_model = selected_model
        record.user_notes = user_notes

        logger.info(f"User selected model {selected_model} from comparison {comparison_id}")

    def get_comparison_history(self) -> list[ComparisonRecord]:
        """Get all comparison records.

        Returns:
            List of comparison records, newest first.
        """
        return list(reversed(self._comparison_history))

    def get_comparison(self, comparison_id: str) -> ComparisonRecord | None:
        """Get a specific comparison record.

        Args:
            comparison_id: ID of comparison to retrieve.

        Returns:
            ComparisonRecord if found, None otherwise.
        """
        return next((r for r in self._comparison_history if r.id == comparison_id), None)

    def get_model_win_rate(self, model_id: str) -> float:
        """Calculate how often a model was selected as best.

        Args:
            model_id: Model ID to check.

        Returns:
            Win rate as percentage (0.0 to 100.0).
        """
        comparisons_with_model = [
            r for r in self._comparison_history if model_id in r.results
        ]

        if not comparisons_with_model:
            return 0.0

        wins = sum(1 for r in comparisons_with_model if r.selected_model == model_id)
        return (wins / len(comparisons_with_model)) * 100.0

    def clear_history(self) -> None:
        """Clear all comparison history."""
        self._comparison_history.clear()
        logger.info("Cleared comparison history")
