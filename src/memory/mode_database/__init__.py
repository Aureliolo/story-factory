"""Database for model scoring and mode management.

Stores generation scores, model performance metrics, and tuning recommendations
for the adaptive learning system.
"""

import logging
import threading
from pathlib import Path
from typing import Any

from src.memory.mode_models import (
    GenerationScore,
    ModelPerformanceSummary,
    TuningRecommendation,
)

from . import (
    _cost_tracking,
    _custom_modes,
    _performance,
    _prompt_metrics,
    _recommendations,
    _refinement,
    _schema,
    _scoring,
    _world_entity,
)

logger = logging.getLogger(__name__)


# Go up from memory/mode_database/ to memory/ to src/ to project root, then into output/
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "output" / "model_scores.db"


class ModeDatabase:
    """SQLite database for model scoring and learning data.

    Tables:
    - generation_scores: Per-generation metrics (quality, speed, implicit signals)
    - model_performance: Aggregated model performance by role and genre
    - recommendations: Tuning recommendation history
    - custom_modes: User-defined generation modes
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file. Defaults to output/model_scores.db.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Create and initialize the SQLite schema."""
        _schema.init_db(self)

    # === Generation Scores (delegated to _scoring) ===

    def record_score(
        self,
        project_id: str,
        agent_role: str,
        model_id: str,
        mode_name: str,
        *,
        chapter_id: str | None = None,
        genre: str | None = None,
        tokens_generated: int | None = None,
        time_seconds: float | None = None,
        tokens_per_second: float | None = None,
        vram_used_gb: float | None = None,
        prose_quality: float | None = None,
        instruction_following: float | None = None,
        consistency_score: float | None = None,
        was_regenerated: bool = False,
        edit_distance: int | None = None,
        user_rating: int | None = None,
        prompt_hash: str | None = None,
    ) -> int:
        """Record a single generation score row for a project and model."""
        return _scoring.record_score(
            self,
            project_id,
            agent_role,
            model_id,
            mode_name,
            chapter_id=chapter_id,
            genre=genre,
            tokens_generated=tokens_generated,
            time_seconds=time_seconds,
            tokens_per_second=tokens_per_second,
            vram_used_gb=vram_used_gb,
            prose_quality=prose_quality,
            instruction_following=instruction_following,
            consistency_score=consistency_score,
            was_regenerated=was_regenerated,
            edit_distance=edit_distance,
            user_rating=user_rating,
            prompt_hash=prompt_hash,
        )

    def update_score(
        self,
        score_id: int,
        *,
        prose_quality: float | None = None,
        instruction_following: float | None = None,
        consistency_score: float | None = None,
        was_regenerated: bool | None = None,
        edit_distance: int | None = None,
        user_rating: int | None = None,
    ) -> None:
        """Update an existing score with additional metrics."""
        return _scoring.update_score(
            self,
            score_id,
            prose_quality=prose_quality,
            instruction_following=instruction_following,
            consistency_score=consistency_score,
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
        """Update an existing score with performance metrics."""
        return _scoring.update_performance_metrics(
            self,
            score_id,
            tokens_generated=tokens_generated,
            time_seconds=time_seconds,
            tokens_per_second=tokens_per_second,
            vram_used_gb=vram_used_gb,
        )

    def get_scores_for_model(
        self,
        model_id: str,
        agent_role: str | None = None,
        genre: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent scores for a model."""
        return _scoring.get_scores_for_model(self, model_id, agent_role, genre, limit)

    def get_scores_for_project(self, project_id: str) -> list[dict[str, Any]]:
        """Retrieve all generation scores for a project ordered by most recent first."""
        return _scoring.get_scores_for_project(self, project_id)

    def get_latest_score_for_chapter(
        self, project_id: str, chapter_id: str
    ) -> dict[str, Any] | None:
        """Get the most recent score for a specific project chapter."""
        return _scoring.get_latest_score_for_chapter(self, project_id, chapter_id)

    def get_score_count(
        self,
        model_id: str | None = None,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> int:
        """Return count of generation score records."""
        return _scoring.get_score_count(self, model_id, agent_role, genre)

    def get_all_scores(
        self,
        agent_role: str | None = None,
        genre: str | None = None,
        limit: int = 1000,
    ) -> list[GenerationScore]:
        """Get all scores as GenerationScore objects."""
        return _scoring.get_all_scores(self, agent_role, genre, limit)

    def export_scores_csv(self, output_path: Path | str) -> int:
        """Export all scores to CSV."""
        return _scoring.export_scores_csv(self, output_path)

    def get_average_score(
        self,
        metric: str,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> float | None:
        """Get average of a specific metric."""
        return _scoring.get_average_score(self, metric, agent_role, genre)

    def get_content_statistics(
        self,
        project_id: str | None = None,
        agent_role: str | None = None,
    ) -> dict:
        """Get content statistics from generation scores."""
        return _scoring.get_content_statistics(self, project_id, agent_role)

    def get_quality_time_series(
        self,
        metric: str = "prose_quality",
        agent_role: str | None = None,
        genre: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get time series data for quality metrics."""
        return _scoring.get_quality_time_series(self, metric, agent_role, genre, limit)

    def get_daily_quality_averages(
        self,
        metric: str = "prose_quality",
        days: int = 30,
        agent_role: str | None = None,
    ) -> list[dict]:
        """Get daily average quality metrics."""
        return _scoring.get_daily_quality_averages(self, metric, days, agent_role)

    # === Model Performance Aggregates (delegated to _performance) ===

    def update_model_performance(
        self,
        model_id: str,
        agent_role: str,
        genre: str | None = None,
    ) -> None:
        """Recalculate aggregated performance for a model/role/genre combo."""
        return _performance.update_model_performance(self, model_id, agent_role, genre)

    def get_model_performance(
        self,
        model_id: str | None = None,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated model performance."""
        return _performance.get_model_performance(self, model_id, agent_role, genre)

    def get_top_models_for_role(
        self,
        agent_role: str,
        limit: int = 5,
        min_samples: int = 3,
    ) -> list[dict[str, Any]]:
        """Get top performing models for a role."""
        return _performance.get_top_models_for_role(self, agent_role, limit, min_samples)

    def get_model_summaries(
        self,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> list[ModelPerformanceSummary]:
        """Get model performance summaries."""
        return _performance.get_model_summaries(self, agent_role, genre)

    def get_quality_vs_speed_data(
        self,
        agent_role: str | None = None,
        min_samples: int = 3,
    ) -> list[dict[str, Any]]:
        """Get data for quality vs speed scatter plot."""
        return _performance.get_quality_vs_speed_data(self, agent_role, min_samples)

    def get_genre_breakdown(self, model_id: str) -> list[dict[str, Any]]:
        """Get performance breakdown by genre for a model."""
        return _performance.get_genre_breakdown(self, model_id)

    def get_unique_genres(self) -> list[str]:
        """Get list of unique genres from scores."""
        return _performance.get_unique_genres(self)

    # === Recommendations (delegated to _recommendations) ===

    def record_recommendation(
        self,
        recommendation_type: str,
        current_value: str,
        suggested_value: str,
        reason: str,
        confidence: float,
        evidence: dict[str, Any] | None = None,
        affected_role: str | None = None,
        expected_improvement: str | None = None,
    ) -> int:
        """Record a tuning recommendation."""
        return _recommendations.record_recommendation(
            self,
            recommendation_type,
            current_value,
            suggested_value,
            reason,
            confidence,
            evidence,
            affected_role,
            expected_improvement,
        )

    def update_recommendation_outcome(
        self,
        recommendation_id: int,
        was_applied: bool,
        user_feedback: str | None = None,
    ) -> None:
        """Record the user's outcome and optional feedback for a recommendation."""
        return _recommendations.update_recommendation_outcome(
            self, recommendation_id, was_applied, user_feedback
        )

    def get_pending_recommendations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recommendations not yet applied and without user feedback."""
        return _recommendations.get_pending_recommendations(self, limit)

    def get_recommendation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent recommendation history."""
        return _recommendations.get_recommendation_history(self, limit)

    def get_recent_recommendations(self, limit: int = 10) -> list[TuningRecommendation]:
        """Get recent recommendations as TuningRecommendation objects."""
        return _recommendations.get_recent_recommendations(self, limit)

    # === Custom Modes (delegated to _custom_modes) ===

    def save_custom_mode(
        self,
        mode_id: str,
        name: str,
        agent_models: dict[str, str],
        agent_temperatures: dict[str, float],
        size_preference: str = "medium",
        vram_strategy: str = "adaptive",
        description: str = "",
        is_experimental: bool = False,
    ) -> None:
        """Save or update a custom generation mode."""
        return _custom_modes.save_custom_mode(
            self,
            mode_id,
            name,
            agent_models,
            agent_temperatures,
            size_preference,
            vram_strategy,
            description,
            is_experimental,
        )

    def get_custom_mode(self, mode_id: str) -> dict[str, Any] | None:
        """Get a custom mode by ID."""
        return _custom_modes.get_custom_mode(self, mode_id)

    def list_custom_modes(self) -> list[dict[str, Any]]:
        """List all custom modes."""
        return _custom_modes.list_custom_modes(self)

    def delete_custom_mode(self, mode_id: str) -> bool:
        """Delete a custom mode."""
        return _custom_modes.delete_custom_mode(self, mode_id)

    # === World Entity Scores (delegated to _world_entity) ===

    def record_world_entity_score(
        self,
        project_id: str,
        entity_type: str,
        entity_name: str,
        model_id: str,
        scores: dict[str, float],
        *,
        entity_id: str | None = None,
        iterations_used: int | None = None,
        generation_time_seconds: float | None = None,
        feedback: str | None = None,
        early_stop_triggered: bool = False,
        threshold_met: bool = False,
        peak_score: float | None = None,
        final_score: float | None = None,
        score_progression: list[float] | None = None,
        consecutive_degradations: int = 0,
        best_iteration: int = 0,
        quality_threshold: float | None = None,
        max_iterations: int | None = None,
    ) -> int:
        """Record a world entity quality score with refinement effectiveness metrics."""
        return _world_entity.record_world_entity_score(
            self,
            project_id,
            entity_type,
            entity_name,
            model_id,
            scores,
            entity_id=entity_id,
            iterations_used=iterations_used,
            generation_time_seconds=generation_time_seconds,
            feedback=feedback,
            early_stop_triggered=early_stop_triggered,
            threshold_met=threshold_met,
            peak_score=peak_score,
            final_score=final_score,
            score_progression=score_progression,
            consecutive_degradations=consecutive_degradations,
            best_iteration=best_iteration,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )

    def get_world_entity_scores(
        self,
        project_id: str | None = None,
        entity_type: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get world entity scores with optional filters."""
        return _world_entity.get_world_entity_scores(self, project_id, entity_type, model_id, limit)

    def get_world_quality_summary(
        self,
        entity_type: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Get summary statistics for world entity quality scores."""
        return _world_entity.get_world_quality_summary(self, entity_type, model_id)

    def get_world_entity_count(
        self,
        entity_type: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Get count of world entity scores."""
        return _world_entity.get_world_entity_count(self, entity_type, model_id)

    # === Prompt Metrics (delegated to _prompt_metrics) ===

    def record_prompt_metrics(
        self,
        prompt_hash: str,
        agent_role: str,
        task: str,
        template_version: str,
        model_id: str,
        tokens_generated: int | None = None,
        generation_time_seconds: float | None = None,
        success: bool = True,
        project_id: str | None = None,
        error_message: str | None = None,
    ) -> int:
        """Record prompt template usage for analytics."""
        return _prompt_metrics.record_prompt_metrics(
            self,
            prompt_hash,
            agent_role,
            task,
            template_version,
            model_id,
            tokens_generated,
            generation_time_seconds,
            success,
            project_id,
            error_message,
        )

    def get_prompt_analytics(
        self,
        agent_role: str | None = None,
        task: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get prompt performance statistics."""
        return _prompt_metrics.get_prompt_analytics(self, agent_role, task, days)

    def get_prompt_metrics_summary(self) -> dict[str, Any]:
        """Get overall summary of prompt metrics."""
        return _prompt_metrics.get_prompt_metrics_summary(self)

    def get_prompt_metrics_by_hash(
        self,
        prompt_hash: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get metrics for a specific template hash."""
        return _prompt_metrics.get_prompt_metrics_by_hash(self, prompt_hash, limit)

    def get_prompt_error_summary(self, days: int = 7) -> list[dict[str, Any]]:
        """Get summary of prompt errors for debugging."""
        return _prompt_metrics.get_prompt_error_summary(self, days)

    # === Refinement Effectiveness (delegated to _refinement) ===

    def get_refinement_effectiveness_summary(
        self,
        entity_type: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Summarize refinement-loop effectiveness for world entities."""
        return _refinement.get_refinement_effectiveness_summary(self, entity_type, days)

    def get_refinement_progression_data(
        self,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve recent world-entity refinement progression records."""
        return _refinement.get_refinement_progression_data(self, entity_type, limit)

    # === Generation Cost Tracking (delegated to _cost_tracking) ===

    def start_generation_run(
        self,
        run_id: str,
        project_id: str,
        run_type: str,
    ) -> int:
        """Start tracking a new generation run."""
        return _cost_tracking.start_generation_run(self, run_id, project_id, run_type)

    def update_generation_run(
        self,
        run_id: str,
        *,
        total_tokens: int | None = None,
        total_time_seconds: float | None = None,
        total_calls: int | None = None,
        by_entity_type: dict[str, dict] | None = None,
        by_model: dict[str, dict] | None = None,
        total_iterations: int | None = None,
        wasted_iterations: int | None = None,
        completed: bool = False,
    ) -> None:
        """Update accumulated metrics for an existing generation run."""
        return _cost_tracking.update_generation_run(
            self,
            run_id,
            total_tokens=total_tokens,
            total_time_seconds=total_time_seconds,
            total_calls=total_calls,
            by_entity_type=by_entity_type,
            by_model=by_model,
            total_iterations=total_iterations,
            wasted_iterations=wasted_iterations,
            completed=completed,
        )

    def complete_generation_run(self, run_id: str) -> None:
        """Mark a generation run as completed."""
        return _cost_tracking.complete_generation_run(self, run_id)

    def get_generation_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a generation run record by run_id."""
        return _cost_tracking.get_generation_run(self, run_id)

    def get_generation_runs(
        self,
        project_id: str | None = None,
        run_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent generation runs."""
        return _cost_tracking.get_generation_runs(self, project_id, run_type, limit)

    def get_cost_summary(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Summarize generation-run cost and usage metrics."""
        return _cost_tracking.get_cost_summary(self, project_id, days)

    def get_model_cost_breakdown(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Produce a per-model cost and usage breakdown."""
        return _cost_tracking.get_model_cost_breakdown(self, project_id, days)

    def get_entity_type_cost_breakdown(
        self,
        project_id: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Produce cost breakdowns aggregated by world entity type."""
        return _cost_tracking.get_entity_type_cost_breakdown(self, project_id, days)
