"""Model mode service - manages generation modes, scoring, and adaptive learning.

This service handles:
- Mode management (presets and custom modes)
- VRAM-aware model loading strategies
- Score recording and aggregation
- Learning/tuning recommendations
"""

import logging
from pathlib import Path
from typing import Any

from src.memory.mode_database import ModeDatabase
from src.memory.mode_models import (
    GenerationMode,
    LearningSettings,
    QualityScores,
    TuningRecommendation,
    get_preset_mode,
    list_preset_modes,
)
from src.services.model_mode_service import _analytics, _learning, _modes, _scoring, _vram
from src.settings import Settings

logger = logging.getLogger(__name__)


class ModelModeService:
    """Service for managing generation modes and adaptive learning.

    This service coordinates:
    - Mode selection and customization
    - VRAM-aware model loading/unloading
    - Quality scoring via LLM judge
    - Performance tracking and aggregation
    - Tuning recommendations based on historical data
    """

    def __init__(
        self,
        settings: Settings,
        db_path: Path | str | None = None,
    ):
        """Initialize model mode service.

        Args:
            settings: Application settings.
            db_path: Path to scoring database. Defaults to output/model_scores.db
        """
        logger.debug(f"Initializing ModelModeService: db_path={db_path}")
        self.settings = settings
        # Default to output/model_scores.db at project root
        default_db = Path(__file__).parent.parent.parent / "output" / "model_scores.db"
        self._db_path = Path(db_path) if db_path else default_db
        self._db = ModeDatabase(self._db_path)

        # Current mode
        self._current_mode: GenerationMode | None = None

        # Learning settings
        self._learning_settings = LearningSettings()

        # Track chapters for periodic triggers
        self._chapters_since_analysis = 0

        # Loaded model tracking (for VRAM management)
        self._loaded_models: set[str] = set()
        logger.debug("ModelModeService initialized successfully")

    # === Mode Management ===

    def get_current_mode(self) -> GenerationMode:
        """Get the current generation mode.

        Returns preset 'balanced' if no mode is set.
        """
        if self._current_mode is None:
            self._current_mode = get_preset_mode("balanced") or list_preset_modes()[0]
        return self._current_mode

    def set_mode(self, mode_id: str) -> GenerationMode:
        """Activate the generation mode identified by `mode_id`.

        Args:
            mode_id: Identifier of a preset or custom generation mode.

        Returns:
            The activated generation mode object.

        Raises:
            ValueError: If no mode with `mode_id` exists or if a custom mode
                contains an invalid VRAM strategy.
        """
        return _modes.set_mode(self, mode_id)

    def list_modes(self) -> list[GenerationMode]:
        """Get all available generation modes.

        Returns:
            A list of GenerationMode instances for presets and custom modes.

        Raises:
            ValueError: If a custom mode has missing or invalid size_preference.
        """
        return _modes.list_modes(self)

    def save_custom_mode(self, mode: GenerationMode) -> None:
        """Persist a custom GenerationMode to the mode database.

        Args:
            mode: The custom mode to persist; must not be None.
        """
        _modes.save_custom_mode(self, mode)

    def delete_custom_mode(self, mode_id: str) -> bool:
        """Delete a custom mode.

        Args:
            mode_id: The mode ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        return _modes.delete_custom_mode(self, mode_id)

    def get_model_for_agent(self, agent_role: str) -> str:
        """Get the model ID for an agent based on current mode.

        Args:
            agent_role: The agent role (writer, architect, etc.)

        Returns:
            Model ID selected based on size preference or from user override.
        """
        return _modes.get_model_for_agent(self, agent_role)

    def _select_model_with_size_preference(
        self,
        agent_role: str,
        size_pref: Any,
        available_vram: int,
    ) -> str:
        """Select the best installed model ID for an agent role.

        Args:
            agent_role: Agent role tag to filter models.
            size_pref: Desired model size preference.
            available_vram: Available VRAM in gigabytes.

        Returns:
            The selected model ID.

        Raises:
            ValueError: If no installed model is tagged for the given agent_role.
        """
        return _modes.select_model_with_size_preference(self, agent_role, size_pref, available_vram)

    def _calculate_tier_score(self, size_gb: float, size_pref: Any) -> float:
        """Score how well a model size matches the given SizePreference.

        Args:
            size_gb: Model size in gigabytes.
            size_pref: Desired size preference enum.

        Returns:
            A score between 0.0 and 10.0.
        """
        return _modes.calculate_tier_score(size_gb, size_pref)

    def get_temperature_for_agent(self, agent_role: str) -> float:
        """Get the temperature for an agent role according to the active generation mode.

        Args:
            agent_role: The agent role.

        Returns:
            Temperature value for the specified agent role.
        """
        return _modes.get_temperature_for_agent(self, agent_role)

    # === VRAM Management ===

    def prepare_model(self, model_id: str) -> None:
        """Prepare a model for use according to the configured VRAM strategy.

        Args:
            model_id: Identifier of the model to prepare.
        """
        _vram.prepare_model(self, model_id)

    # === Score Recording ===

    def record_generation(
        self,
        project_id: str,
        agent_role: str,
        model_id: str,
        *,
        chapter_id: str | None = None,
        genre: str | None = None,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        prompt_text: str | None = None,
    ) -> int:
        """Record a generation event.

        Returns:
            The score ID for later updates.
        """
        return _scoring.record_generation(
            self,
            project_id,
            agent_role,
            model_id,
            chapter_id=chapter_id,
            genre=genre,
            tokens_generated=tokens_generated,
            time_seconds=time_seconds,
            prompt_text=prompt_text,
        )

    def update_quality_scores(
        self,
        score_id: int,
        quality: QualityScores,
    ) -> None:
        """Update a score record with quality scores."""
        _scoring.update_quality_scores(self, score_id, quality)

    def record_implicit_signal(
        self,
        score_id: int,
        *,
        was_regenerated: bool | None = None,
        edit_distance: int | None = None,
        user_rating: int | None = None,
    ) -> None:
        """Record an implicit quality signal."""
        _scoring.record_implicit_signal(
            self,
            score_id,
            was_regenerated=was_regenerated,
            edit_distance=edit_distance,
            user_rating=user_rating,
        )

    def update_performance_metrics(
        self,
        score_id: int,
        *,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        tokens_per_second: float | None = None,
        vram_used_gb: float | None = None,
    ) -> None:
        """Update a score record with performance metrics.

        Args:
            score_id: The score record ID.
            tokens_generated: Number of tokens generated.
            time_seconds: Generation time in seconds.
            tokens_per_second: Generation speed (calculated if not provided).
            vram_used_gb: VRAM used during generation.
        """
        _scoring.update_performance_metrics(
            self,
            score_id,
            tokens_generated=tokens_generated,
            time_seconds=time_seconds,
            tokens_per_second=tokens_per_second,
            vram_used_gb=vram_used_gb,
        )

    # === LLM Quality Judge ===

    def judge_quality(
        self,
        content: str,
        genre: str,
        tone: str,
        themes: list[str],
    ) -> QualityScores:
        """Use LLM to judge content quality.

        Args:
            content: The generated content to evaluate.
            genre: Story genre.
            tone: Story tone.
            themes: Story themes.

        Returns:
            QualityScores with prose_quality and instruction_following.
        """
        return _scoring.judge_quality(self, content, genre, tone, themes)

    def calculate_consistency_score(self, issues: list[dict[str, Any]]) -> float:
        """Calculate consistency score from continuity issues.

        Args:
            issues: List of ContinuityIssue-like dicts with 'severity'.

        Returns:
            Score from 0-10 (10 = no issues).
        """
        return _scoring.calculate_consistency_score(issues)

    # === Learning/Tuning ===

    def set_learning_settings(self, settings: LearningSettings) -> None:
        """Update learning settings."""
        _learning.set_learning_settings(self, settings)

    def get_learning_settings(self) -> LearningSettings:
        """Get current learning settings."""
        return _learning.get_learning_settings(self)

    def should_tune(self) -> bool:
        """Check if tuning analysis should run based on triggers."""
        return _learning.should_tune(self)

    def on_chapter_complete(self) -> None:
        """Called when a chapter is completed."""
        _learning.on_chapter_complete(self)

    def on_project_complete(self) -> list[TuningRecommendation]:
        """Called when a project is completed.

        Returns recommendations if after_project trigger is enabled.
        """
        return _learning.on_project_complete(self)

    def get_recommendations(self) -> list[TuningRecommendation]:
        """Generate tuning recommendations based on historical data.

        Returns:
            List of recommendations, may be empty if insufficient data.
        """
        return _learning.get_recommendations(self)

    def apply_recommendation(self, recommendation: TuningRecommendation) -> bool:
        """Apply a tuning recommendation to the current mode.

        Args:
            recommendation: The recommendation to apply.

        Returns:
            True if the recommendation was applied, False otherwise.
        """
        return _learning.apply_recommendation(self, recommendation)

    def handle_recommendations(
        self,
        recommendations: list[TuningRecommendation],
    ) -> list[TuningRecommendation]:
        """Handle recommendations based on autonomy level.

        Returns:
            Recommendations that were not auto-applied (need user approval).
        """
        return _learning.handle_recommendations(self, recommendations)

    # === Analytics ===

    def get_quality_vs_speed_data(
        self,
        agent_role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get data for quality vs speed scatter plot."""
        return _analytics.get_quality_vs_speed_data(self, agent_role)

    def get_model_performance(
        self,
        model_id: str | None = None,
        agent_role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated model performance."""
        return _analytics.get_model_performance(self, model_id, agent_role)

    def get_recommendation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recommendation history."""
        return _analytics.get_recommendation_history(self, limit)

    def export_scores_csv(self, output_path: Path | str) -> int:
        """Export all recorded scores to a CSV file.

        Args:
            output_path: Destination file path for the exported CSV.

        Returns:
            Number of score records written to the CSV.
        """
        return _analytics.export_scores_csv(self, output_path)

    def get_pending_recommendations(self) -> list[TuningRecommendation]:
        """Return pending tuning recommendations from the database.

        Returns:
            Parsed pending recommendations (may be empty).
        """
        return _analytics.get_pending_recommendations(self)

    def dismiss_recommendation(self, recommendation: TuningRecommendation) -> None:
        """Mark a tuning recommendation as dismissed.

        Args:
            recommendation: The recommendation to dismiss.
        """
        _analytics.dismiss_recommendation(self, recommendation)

    def on_regenerate(self, project_id: str, chapter_id: str) -> None:
        """Record regeneration as a negative implicit signal.

        Args:
            project_id: The project ID.
            chapter_id: The chapter being regenerated.
        """
        _analytics.on_regenerate(self, project_id, chapter_id)
