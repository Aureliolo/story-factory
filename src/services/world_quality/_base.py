"""Base class for WorldQualityService.

Contains core initialization, config, model selection, and analytics.
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import ollama

from src.memory.mode_database import ModeDatabase
from src.memory.world_quality import RefinementConfig, RefinementHistory
from src.services.model_mode_service import ModelModeService
from src.settings import Settings
from src.utils.validation import validate_not_empty

logger = logging.getLogger(__name__)


@dataclass
class EntityGenerationProgress:
    """Progress information for entity generation operations."""

    current: int  # Current entity (1-indexed)
    total: int  # Total entities requested
    entity_type: str  # character, location, faction, etc.
    entity_name: str | None = None  # Name if known (after generation completes)
    phase: str = "generating"  # generating, refining, complete
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None

    @property
    def progress_fraction(self) -> float:
        """Progress as 0.0-1.0."""
        return (self.current - 1) / max(self.total, 1)


class WorldQualityServiceBase:
    """Base class for WorldQualityService with core functionality.

    Uses a multi-model iteration loop:
    1. Creator generates initial entity (high temperature)
    2. Judge evaluates quality (low temperature)
    3. If below threshold, refine with feedback and repeat
    4. Return entity with quality scores
    """

    # Map entity types to specialized agent roles for model selection.
    # Creator roles: Characters/locations/items need descriptive writing (writer),
    # factions/concepts need reasoning (architect), relationships need dynamics (editor).
    # Judge roles: All entities use validator for consistent quality assessment.
    ENTITY_CREATOR_ROLES: ClassVar[dict[str, str]] = {
        "character": "writer",  # Strong character development
        "faction": "architect",  # Political/organizational reasoning
        "location": "writer",  # Atmospheric/descriptive writing
        "item": "writer",  # Creative descriptions
        "concept": "architect",  # Abstract thinking
        "relationship": "editor",  # Understanding dynamics
    }

    ENTITY_JUDGE_ROLES: ClassVar[dict[str, str]] = {
        "character": "validator",  # Character consistency checking
        "faction": "validator",  # Faction coherence checking
        "location": "validator",  # Location plausibility checking
        "item": "validator",  # Item consistency checking
        "concept": "validator",  # Concept coherence checking
        "relationship": "validator",  # Relationship validity checking
    }

    @staticmethod
    def _calculate_eta(
        completed_times: list[float],
        remaining_count: int,
    ) -> float | None:
        """Calculate ETA using exponential moving average.

        Uses EMA with alpha=0.3 to weight recent entity generation times more heavily,
        providing more accurate estimates as we learn the actual generation speed.

        Args:
            completed_times: List of completion times for previous entities (in seconds).
            remaining_count: Number of entities remaining to generate.

        Returns:
            Estimated remaining time in seconds, or None if no data available.
        """
        if not completed_times or remaining_count <= 0:
            return None
        # EMA with alpha=0.3 to weight recent times more heavily
        alpha = 0.3
        avg = completed_times[0]
        for t in completed_times[1:]:
            avg = alpha * t + (1 - alpha) * avg
        return avg * remaining_count

    @staticmethod
    def _format_properties(properties: list[Any] | Any | None) -> str:
        """Format a list of properties into a comma-separated string.

        Handles both string and dict properties (LLM sometimes returns dicts).
        Also handles None or non-list inputs gracefully.

        Args:
            properties: List of properties (strings or dicts), or None/single value.

        Returns:
            Comma-separated string of property names, or empty string if no properties.
        """
        if not properties:
            logger.debug(f"_format_properties: early return on falsy input: {properties!r}")
            return ""
        if not isinstance(properties, list):
            logger.debug(
                f"_format_properties: coercing non-list input to list: {type(properties).__name__}"
            )
            properties = [properties]

        result: list[str] = []
        for prop in properties:
            if isinstance(prop, str):
                result.append(prop)
            elif isinstance(prop, dict):
                # Try to extract a name or description from dict
                # Use key existence check to handle empty strings correctly
                # Coerce to string to handle non-string values (e.g., None, int)
                if "name" in prop:
                    value = prop["name"]
                    result.append("" if value is None else str(value))
                elif "description" in prop:
                    value = prop["description"]
                    result.append("" if value is None else str(value))
                else:
                    result.append(str(prop))
            else:
                result.append(str(prop))
        return ", ".join(result)

    def __init__(self, settings: Settings, mode_service: ModelModeService):
        """Initialize WorldQualityService.

        Args:
            settings: Application settings.
            mode_service: Model mode service for model selection.
        """
        logger.debug("Initializing WorldQualityService")
        self.settings = settings
        self.mode_service = mode_service
        self._client: ollama.Client | None = None
        self._analytics_db: ModeDatabase | None = None
        logger.debug("WorldQualityService initialized successfully")

    @property
    def analytics_db(self) -> ModeDatabase:
        """Get or create analytics database instance."""
        if self._analytics_db is None:
            self._analytics_db = ModeDatabase()
        return self._analytics_db

    @property
    def client(self) -> ollama.Client:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = ollama.Client(
                host=self.settings.ollama_url,
                timeout=float(self.settings.ollama_timeout),
            )
        return self._client

    def get_config(self) -> RefinementConfig:
        """Get refinement configuration from src.settings."""
        return RefinementConfig.from_settings(self.settings)

    def _get_creator_model(self, entity_type: str | None = None) -> str:
        """Get the model to use for creative generation.

        Args:
            entity_type: Type of entity being created (character, faction, location, etc.).
                        If provided, uses entity-type-specific agent role for model selection.

        Returns:
            Model ID to use for generation.
        """
        agent_role = (
            self.ENTITY_CREATOR_ROLES.get(entity_type, "writer") if entity_type else "writer"
        )
        model = self.mode_service.get_model_for_agent(agent_role)
        logger.debug(
            f"Selected creator model '{model}' for entity_type={entity_type} (role={agent_role})"
        )
        return model

    def _get_judge_model(self, entity_type: str | None = None) -> str:
        """Get the model to use for quality judgment.

        Args:
            entity_type: Type of entity being judged. Currently all use validator,
                        but allows future differentiation per entity type.

        Returns:
            Model ID to use for judgment.
        """
        agent_role = (
            self.ENTITY_JUDGE_ROLES.get(entity_type, "validator") if entity_type else "validator"
        )
        model = self.mode_service.get_model_for_agent(agent_role)
        logger.debug(
            f"Selected judge model '{model}' for entity_type={entity_type} (role={agent_role})"
        )
        return model

    def record_entity_quality(
        self,
        project_id: str,
        entity_type: str,
        entity_name: str,
        scores: dict[str, Any],
        iterations: int,
        generation_time: float,
        model_id: str | None = None,
        *,
        early_stop_triggered: bool = False,
        threshold_met: bool = False,
        peak_score: float | None = None,
        final_score: float | None = None,
        score_progression: list[float] | None = None,
        consecutive_degradations: int = 0,
        best_iteration: int = 0,
        quality_threshold: float | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """
        Persist entity quality scores and refinement metrics to the analytics database.

        Parameters:
            project_id (str): Project identifier.
            entity_type (str): Entity category (e.g., "character", "location", "faction").
            entity_name (str): Name of the entity being recorded.
            scores (dict[str, Any]): Dictionary of quality metrics; may include keys like `"average"` and `"feedback"`.
            iterations (int): Number of refinement iterations performed.
            generation_time (float): Total generation time in seconds.
            model_id (str | None): Identifier of the model used; if `None`, a creator model is selected based on `entity_type`.
            early_stop_triggered (bool): Whether the refinement loop stopped early (e.g., due to stagnation or safety triggers).
            threshold_met (bool): Whether the configured quality threshold was reached during refinement.
            peak_score (float | None): Highest score observed across iterations.
            final_score (float | None): Score of the final returned entity.
            score_progression (list[float] | None): Sequence of scores recorded per iteration.
            consecutive_degradations (int): Count of consecutive iterations with decreasing scores.
            best_iteration (int): Iteration index that produced the best observed score.
            quality_threshold (float | None): Quality threshold used to judge success.
            max_iterations (int | None): Maximum allowed refinement iterations.
        """
        validate_not_empty(project_id, "project_id")
        validate_not_empty(entity_type, "entity_type")
        validate_not_empty(entity_name, "entity_name")

        # Determine model_id if not provided
        if model_id is None:
            model_id = self._get_creator_model(entity_type)

        try:
            self.analytics_db.record_world_entity_score(
                project_id=project_id,
                entity_type=entity_type,
                entity_name=entity_name,
                model_id=model_id,
                scores=scores,
                iterations_used=iterations,
                generation_time_seconds=generation_time,
                feedback=scores.get("feedback", ""),
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
            logger.debug(
                f"Recorded {entity_type} '{entity_name}' quality to analytics "
                f"(model={model_id}, avg: {scores.get('average', 0):.1f}, "
                f"threshold_met={threshold_met}, early_stop={early_stop_triggered})"
            )
        except Exception as e:
            # Don't fail generation if analytics recording fails
            logger.warning(f"Failed to record entity quality to analytics: {e}")

    def _log_refinement_analytics(
        self,
        history: RefinementHistory,
        project_id: str,
        *,
        early_stop_triggered: bool = False,
        threshold_met: bool = False,
        quality_threshold: float | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """
        Log and persist refinement iteration analytics for a completed refinement history.

        Analyzes the provided RefinementHistory, emits a concise info-level summary of iterations and score progression, and records extended refinement metrics to the analytics database via record_entity_quality.

        Parameters:
            history (RefinementHistory): The refinement history for the entity being reported.
            project_id (str): Project identifier under which analytics should be recorded.
            early_stop_triggered (bool): True if the refinement loop stopped early (e.g., due to consecutive degradations or an early-stop condition).
            threshold_met (bool): True if the configured quality threshold was reached during refinement.
            quality_threshold (float | None): The numeric quality threshold that was used for this refinement run, if any.
            max_iterations (int | None): The configured maximum number of refinement iterations for this run, if any.
        """
        analysis = history.analyze_improvement()

        logger.info(
            f"REFINEMENT ANALYTICS [{history.entity_type}] '{history.entity_name}':\n"
            f"  - Total iterations: {analysis['total_iterations']}\n"
            f"  - Score progression: {' -> '.join(f'{s:.1f}' for s in analysis['score_progression'])}\n"
            f"  - Best iteration: {analysis['best_iteration']} ({history.peak_score:.1f})\n"
            f"  - Final returned: iteration {history.final_iteration} ({history.final_score:.1f})\n"
            f"  - Improved over first: {analysis['improved']}\n"
            f"  - Worsened after peak: {analysis.get('worsened_after_peak', False)}\n"
            f"  - Threshold met: {threshold_met}\n"
            f"  - Early stop triggered: {early_stop_triggered}"
        )

        # Record to analytics database with extended scores info
        extended_scores = {
            "final_score": history.final_score,
            "peak_score": history.peak_score,
            "best_iteration": analysis["best_iteration"],
            "improved": analysis["improved"],
            "worsened_after_peak": analysis.get("worsened_after_peak", False),
            "average": history.final_score,  # For backwards compatibility
        }
        self.record_entity_quality(
            project_id=project_id,
            entity_type=history.entity_type,
            entity_name=history.entity_name,
            scores=extended_scores,
            iterations=analysis["total_iterations"],
            generation_time=0.0,  # Not tracked at this level
            early_stop_triggered=early_stop_triggered,
            threshold_met=threshold_met,
            peak_score=history.peak_score,
            final_score=history.final_score,
            score_progression=analysis["score_progression"],
            consecutive_degradations=history.consecutive_degradations,
            best_iteration=analysis["best_iteration"],
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
        )
