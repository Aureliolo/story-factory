"""World Quality Service - multi-model iteration for world building quality.

Implements a generate-judge-refine loop using:
- Creator model: High temperature (0.9) for creative generation
- Judge model: Low temperature (0.1) for consistent evaluation
- Refinement: Incorporates feedback to improve entities
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import ollama

from src.memory.entities import Entity
from src.memory.mode_database import ModeDatabase
from src.memory.story_state import (
    Chapter,
    Character,
    PlotOutline,
    StoryBrief,
    StoryState,
)
from src.memory.world_quality import (
    ChapterQualityScores,
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    JudgeConsistencyConfig,
    LocationQualityScores,
    PlotQualityScores,
    RefinementConfig,
    RefinementHistory,
    RelationshipQualityScores,
)
from src.services.model_mode_service import ModelModeService
from src.services.world_quality_service import (
    _chapter_quality,
    _character,
    _concept,
    _faction,
    _item,
    _location,
    _plot,
    _relationship,
)
from src.services.world_quality_service._batch import (
    generate_characters_with_quality as _generate_characters_with_quality,
)
from src.services.world_quality_service._batch import (
    generate_concepts_with_quality as _generate_concepts_with_quality,
)
from src.services.world_quality_service._batch import (
    generate_factions_with_quality as _generate_factions_with_quality,
)
from src.services.world_quality_service._batch import (
    generate_items_with_quality as _generate_items_with_quality,
)
from src.services.world_quality_service._batch import (
    generate_locations_with_quality as _generate_locations_with_quality,
)
from src.services.world_quality_service._batch import (
    generate_relationships_with_quality as _generate_relationships_with_quality,
)
from src.services.world_quality_service._batch import (
    review_chapters_batch as _review_chapters_batch,
)
from src.services.world_quality_service._batch import (
    review_characters_batch as _review_characters_batch,
)
from src.services.world_quality_service._chapter_quality import (
    review_chapter_quality as _review_chapter_quality,
)
from src.services.world_quality_service._character import (
    generate_character_with_quality as _generate_character_with_quality,
)
from src.services.world_quality_service._character import (
    review_character_quality as _review_character_quality,
)
from src.services.world_quality_service._concept import (
    generate_concept_with_quality as _generate_concept_with_quality,
)
from src.services.world_quality_service._faction import (
    FACTION_IDEOLOGY_HINTS,
    FACTION_NAMING_HINTS,
    FACTION_STRUCTURE_HINTS,
)
from src.services.world_quality_service._faction import (
    generate_faction_with_quality as _generate_faction_with_quality,
)
from src.services.world_quality_service._item import (
    generate_item_with_quality as _generate_item_with_quality,
)
from src.services.world_quality_service._location import (
    generate_location_with_quality as _generate_location_with_quality,
)
from src.services.world_quality_service._model_cache import ModelResolutionCache
from src.services.world_quality_service._plot import (
    review_plot_quality as _review_plot_quality,
)
from src.services.world_quality_service._relationship import (
    generate_relationship_with_quality as _generate_relationship_with_quality,
)
from src.services.world_quality_service._validation import (
    check_contradiction as _check_contradiction,
)
from src.services.world_quality_service._validation import (
    extract_entity_claims as _extract_entity_claims,
)
from src.services.world_quality_service._validation import (
    generate_mini_description as _generate_mini_description,
)
from src.services.world_quality_service._validation import (
    generate_mini_descriptions_batch as _generate_mini_descriptions_batch,
)
from src.services.world_quality_service._validation import (
    refine_entity as _refine_entity,
)
from src.services.world_quality_service._validation import (
    regenerate_entity as _regenerate_entity,
)
from src.services.world_quality_service._validation import (
    suggest_relationships_for_entity as _suggest_relationships_for_entity,
)
from src.services.world_quality_service._validation import (
    validate_entity_consistency as _validate_entity_consistency,
)
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


class WorldQualityService:
    """Service for quality-controlled world entity generation.

    Uses a multi-model iteration loop:
    1. Creator generates initial entity (high temperature)
    2. Judge evaluates quality (low temperature)
    3. If below threshold, refine with feedback and repeat
    4. Return entity with quality scores
    """

    # Map entity types to specialized agent roles for model selection.
    # Creator roles: Characters/locations/items need descriptive writing (writer),
    # factions/concepts need reasoning (architect), relationships need dynamics (editor).
    # Judge roles: All entities use judge for quality evaluation (needs reasoning capability).
    ENTITY_CREATOR_ROLES: ClassVar[dict[str, str]] = {
        "character": "writer",  # Strong character development
        "faction": "architect",  # Political/organizational reasoning
        "location": "writer",  # Atmospheric/descriptive writing
        "item": "writer",  # Creative descriptions
        "concept": "architect",  # Abstract thinking
        "relationship": "editor",  # Understanding dynamics
        "plot": "architect",  # Plot structure reasoning
        "chapter": "architect",  # Chapter outline structure
    }

    ENTITY_JUDGE_ROLES: ClassVar[dict[str, str]] = {
        "character": "judge",  # Character quality evaluation
        "faction": "judge",  # Faction quality evaluation
        "location": "judge",  # Location quality evaluation
        "item": "judge",  # Item quality evaluation
        "concept": "judge",  # Concept quality evaluation
        "relationship": "judge",  # Relationship quality evaluation
        "plot": "judge",  # Plot quality evaluation
        "chapter": "judge",  # Chapter quality evaluation
    }

    # Faction diversity hints (exposed as class vars for backward compat)
    FACTION_NAMING_HINTS: ClassVar[list[str]] = FACTION_NAMING_HINTS
    FACTION_STRUCTURE_HINTS: ClassVar[list[str]] = FACTION_STRUCTURE_HINTS
    FACTION_IDEOLOGY_HINTS: ClassVar[list[str]] = FACTION_IDEOLOGY_HINTS

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
        self._model_cache = ModelResolutionCache(settings, mode_service)
        logger.debug("WorldQualityService initialized successfully")

    @property
    def analytics_db(self) -> ModeDatabase:
        """Get or create analytics database instance."""
        if self._analytics_db is None:
            self._analytics_db = ModeDatabase()
        return self._analytics_db

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
            scores (dict[str, Any]): Dictionary of quality metrics.
            iterations (int): Number of refinement iterations performed.
            generation_time (float): Total generation time in seconds.
            model_id (str | None): Identifier of the model used.
            early_stop_triggered (bool): Whether the refinement loop stopped early.
            threshold_met (bool): Whether the configured quality threshold was reached.
            peak_score (float | None): Highest score observed across iterations.
            final_score (float | None): Score of the final returned entity.
            score_progression (list[float] | None): Sequence of scores per iteration.
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

    @property
    def client(self) -> ollama.Client:
        """Get or create Ollama client with scaled timeout based on model size."""
        if self._client is None:
            # Use writer model for timeout scaling since it's typically the largest
            writer_model = self.settings.get_model_for_agent("writer")
            self._client = ollama.Client(
                host=self.settings.ollama_url,
                timeout=self.settings.get_scaled_timeout(writer_model),
            )
        return self._client

    def get_config(self) -> RefinementConfig:
        """Get refinement configuration from src.settings."""
        return RefinementConfig.from_settings(self.settings)

    def get_judge_config(self) -> JudgeConsistencyConfig:
        """Get judge consistency configuration from settings."""
        return JudgeConsistencyConfig.from_settings(self.settings)

    def _resolve_model_for_role(self, agent_role: str) -> str:
        """Resolve the model for an agent role, respecting Settings hierarchy.

        The resolution order is:
        1. If use_per_agent_models is False and default_model is set → use default_model
        2. If use_per_agent_models is True and agent_models has an explicit model → use it
        3. Otherwise → delegate to ModelModeService auto-selection

        This ensures that Settings-level configuration (UI-visible) takes priority
        over the mode system's automatic selection, which was previously bypassed.

        Args:
            agent_role: The agent role to resolve (writer, validator, etc.).

        Returns:
            Model ID to use for this role.
        """
        # 1. Per-agent disabled + explicit default → use default for everything
        if not self.settings.use_per_agent_models:
            if self.settings.default_model != "auto":
                logger.debug(
                    "Using default_model '%s' for role '%s' (use_per_agent_models=False)",
                    self.settings.default_model,
                    agent_role,
                )
                return self.settings.default_model

        # 2. Per-agent enabled + explicit model for this role → use it
        if self.settings.use_per_agent_models:
            if agent_role not in self.settings.agent_models:
                raise ValueError(
                    f"Unknown agent role '{agent_role}'. "
                    f"Configured roles: {sorted(self.settings.agent_models.keys())}"
                )
            model_setting = self.settings.agent_models[agent_role]
            if model_setting != "auto":
                logger.debug(
                    "Using explicit agent model '%s' for role '%s'",
                    model_setting,
                    agent_role,
                )
                return model_setting

        # 3. Fall through to mode service auto-selection
        model = self.mode_service.get_model_for_agent(agent_role)
        logger.debug(
            "Auto-selected model '%s' for role '%s' via mode service",
            model,
            agent_role,
        )
        return model

    def invalidate_model_cache(self) -> None:
        """Invalidate resolved model storage.

        Call this when settings change to force re-resolution of models.
        """
        self._model_cache.invalidate()

    def _get_creator_model(self, entity_type: str | None = None) -> str:
        """Get the model to use for creative generation.

        Respects the Settings hierarchy: explicit default_model or per-agent model
        takes priority over ModelModeService auto-selection.

        Uses resolved model storage to avoid redundant tier score calculations
        when the same role is requested multiple times.

        Args:
            entity_type: Type of entity being created (character, faction, location, etc.).
                        If provided, uses entity-type-specific agent role for model selection.

        Returns:
            Model ID to use for generation.
        """
        agent_role = (
            self.ENTITY_CREATOR_ROLES.get(entity_type, "writer") if entity_type else "writer"
        )

        # Check cache (validates context automatically)
        cached = self._model_cache.get_creator_model(agent_role)
        if cached is not None:
            logger.debug("Using cached creator model '%s' for role=%s", cached, agent_role)
            return cached

        # Resolve and store the model
        model = self._resolve_model_for_role(agent_role)
        self._model_cache.store_creator_model(agent_role, model)
        logger.debug(
            "Resolved and cached creator model '%s' for entity_type=%s (role=%s)",
            model,
            entity_type,
            agent_role,
        )
        return model

    def _get_judge_model(self, entity_type: str | None = None) -> str:
        """Get the model to use for quality judgment.

        Respects the Settings hierarchy: explicit default_model or per-agent model
        takes priority over ModelModeService auto-selection.

        Uses resolved model storage to avoid redundant tier score calculations
        when the same role is requested multiple times.

        If the resolved judge model is the same as the creator model for the
        same entity type, attempts to pick a different model from the available
        judge-tagged models. Falls back to the same model with a throttled warning
        if no alternative is available.

        Args:
            entity_type: Type of entity being judged. If provided, checks that
                        the judge model differs from the creator model.

        Returns:
            Model ID to use for judgment.
        """
        agent_role = self.ENTITY_JUDGE_ROLES.get(entity_type, "judge") if entity_type else "judge"

        # Check cache (validates context automatically)
        cached = self._model_cache.get_judge_model(agent_role)
        if cached is not None:
            logger.debug("Using cached judge model '%s' for role=%s", cached, agent_role)
            return cached

        # Resolve the model
        model = self._resolve_model_for_role(agent_role)

        # Prefer a different model from the creator to avoid self-judging bias
        if entity_type:
            creator_model = self._get_creator_model(entity_type)
            if model == creator_model:
                # Try to find an alternative judge model
                alternatives = self.settings.get_models_for_role("judge")
                alternative_found = False
                for alt_model in alternatives:
                    if alt_model != creator_model:
                        logger.debug(
                            "Swapping judge model from '%s' to '%s' for entity_type=%s "
                            "to avoid self-judging bias (creator model is '%s')",
                            model,
                            alt_model,
                            entity_type,
                            creator_model,
                        )
                        model = alt_model
                        alternative_found = True
                        break

                if not alternative_found:
                    # Throttle warning: only warn once per entity_type:model combination
                    conflict_key = f"{entity_type}:{model}"
                    if not self._model_cache.has_warned_conflict(conflict_key):
                        self._model_cache.mark_conflict_warned(conflict_key)
                        logger.warning(
                            "Judge model '%s' is the same as creator model for entity_type=%s "
                            "and no alternative judge model is available. "
                            "A model judging its own output produces unreliable scores. "
                            "Configure a different model for the 'judge' role in Settings > Models.",
                            model,
                            entity_type,
                        )

        # Store the resolved model (including any swapped alternative)
        self._model_cache.store_judge_model(agent_role, model)
        logger.debug(
            "Resolved and cached judge model '%s' for entity_type=%s (role=%s)",
            model,
            entity_type,
            agent_role,
        )
        return model

    def _format_existing_names_warning(self, existing_names: list[str], entity_type: str) -> str:
        """
        Format existing names with explicit DO NOT examples for consistent duplicate prevention.

        Parameters:
            existing_names (list[str]): Existing entity names to avoid.
            entity_type (str): Type of entity (concept, item, location, faction) for context.

        Returns:
            str: Formatted warning string with examples of what NOT to use.
        """
        if not existing_names:
            logger.debug("Formatting existing %s names: none provided", entity_type)
            return f"EXISTING {entity_type.upper()}S: None yet - you are creating the first {entity_type}."

        formatted_names = "\n".join(f"  - {name}" for name in existing_names)

        # Generate example DO NOT variations from the first name
        example_name = existing_names[0] if existing_names else "Example"
        do_not_examples = [
            f'"{example_name}" (exact match)',
            f'"{example_name.upper()}" (case variation)',
            f'"The {example_name}" (prefix variation)',
        ]

        logger.debug("Formatted %d existing %s names for prompt", len(existing_names), entity_type)

        return f"""EXISTING {entity_type.upper()}S (DO NOT DUPLICATE OR CREATE SIMILAR NAMES):
{formatted_names}

DO NOT USE names like:
{chr(10).join(f"  - {ex}" for ex in do_not_examples)}

Create a COMPLETELY DIFFERENT {entity_type} name."""

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

        Parameters:
            history (RefinementHistory): The refinement history for the entity being reported.
            project_id (str): Project identifier under which analytics should be recorded.
            early_stop_triggered (bool): True if the refinement loop stopped early.
            threshold_met (bool): True if the configured quality threshold was reached.
            quality_threshold (float | None): The numeric quality threshold used.
            max_iterations (int | None): The configured maximum number of refinement iterations.
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

    # ========== DELEGATED METHODS ==========

    # -- Character --
    def generate_character_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        custom_instructions: str | None = None,
    ) -> tuple[Character, CharacterQualityScores, int]:
        return _generate_character_with_quality(
            self, story_state, existing_names, custom_instructions
        )

    def review_character_quality(
        self,
        character: Character,
        story_state: StoryState,
    ) -> tuple[Character, CharacterQualityScores, int]:
        return _review_character_quality(self, character, story_state)

    # -- Plot --
    def review_plot_quality(
        self,
        plot_outline: PlotOutline,
        story_state: StoryState,
    ) -> tuple[PlotOutline, PlotQualityScores, int]:
        return _review_plot_quality(self, plot_outline, story_state)

    # -- Chapter --
    def review_chapter_quality(
        self,
        chapter: Chapter,
        story_state: StoryState,
    ) -> tuple[Chapter, ChapterQualityScores, int]:
        return _review_chapter_quality(self, chapter, story_state)

    # -- Location --
    def generate_location_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], LocationQualityScores, int]:
        return _generate_location_with_quality(self, story_state, existing_names)

    # -- Faction --
    def generate_faction_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        existing_locations: list[str] | None = None,
    ) -> tuple[dict[str, Any], FactionQualityScores, int]:
        return _generate_faction_with_quality(self, story_state, existing_names, existing_locations)

    # -- Item --
    def generate_item_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], ItemQualityScores, int]:
        return _generate_item_with_quality(self, story_state, existing_names)

    # -- Concept --
    def generate_concept_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], ConceptQualityScores, int]:
        return _generate_concept_with_quality(self, story_state, existing_names)

    # -- Relationship --
    def generate_relationship_with_quality(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
    ) -> tuple[dict[str, Any], RelationshipQualityScores, int]:
        return _generate_relationship_with_quality(self, story_state, entity_names, existing_rels)

    # -- Batch operations --
    def generate_factions_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        existing_locations: list[str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], FactionQualityScores]]:
        return _generate_factions_with_quality(
            self,
            story_state,
            existing_names,
            count,
            existing_locations,
            cancel_check,
            progress_callback,
        )

    def generate_items_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 3,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], ItemQualityScores]]:
        return _generate_items_with_quality(
            self, story_state, existing_names, count, cancel_check, progress_callback
        )

    def generate_concepts_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], ConceptQualityScores]]:
        return _generate_concepts_with_quality(
            self, story_state, existing_names, count, cancel_check, progress_callback
        )

    def generate_characters_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        custom_instructions: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[Character, CharacterQualityScores]]:
        return _generate_characters_with_quality(
            self,
            story_state,
            existing_names,
            count,
            custom_instructions,
            cancel_check,
            progress_callback,
        )

    def generate_locations_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 3,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], LocationQualityScores]]:
        return _generate_locations_with_quality(
            self, story_state, existing_names, count, cancel_check, progress_callback
        )

    def generate_relationships_with_quality(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        count: int = 5,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], RelationshipQualityScores]]:
        return _generate_relationships_with_quality(
            self, story_state, entity_names, existing_rels, count, cancel_check, progress_callback
        )

    # -- Batch review operations (Architect output quality review) --
    def review_characters_batch(
        self,
        characters: list[Character],
        story_state: StoryState,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[Character, CharacterQualityScores]]:
        return _review_characters_batch(
            self, characters, story_state, cancel_check, progress_callback
        )

    def review_chapters_batch(
        self,
        chapters: list[Chapter],
        story_state: StoryState,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[Chapter, ChapterQualityScores]]:
        return _review_chapters_batch(self, chapters, story_state, cancel_check, progress_callback)

    # -- Private: Character helpers --
    def _create_character(self, story_state, existing_names, temperature, custom_instructions=None):
        return _character._create_character(
            self, story_state, existing_names, temperature, custom_instructions
        )

    def _judge_character_quality(self, character, story_state, temperature):
        return _character._judge_character_quality(self, character, story_state, temperature)

    def _refine_character(self, character, scores, story_state, temperature):
        return _character._refine_character(self, character, scores, story_state, temperature)

    # -- Private: Location helpers --
    def _create_location(self, story_state, existing_names, temperature):
        return _location._create_location(self, story_state, existing_names, temperature)

    def _judge_location_quality(self, location, story_state, temperature):
        return _location._judge_location_quality(self, location, story_state, temperature)

    def _refine_location(self, location, scores, story_state, temperature):
        return _location._refine_location(self, location, scores, story_state, temperature)

    # -- Private: Faction helpers --
    def _create_faction(self, story_state, existing_names, temperature, existing_locations=None):
        return _faction._create_faction(
            self, story_state, existing_names, temperature, existing_locations
        )

    def _judge_faction_quality(self, faction, story_state, temperature):
        return _faction._judge_faction_quality(self, faction, story_state, temperature)

    def _refine_faction(self, faction, scores, story_state, temperature):
        return _faction._refine_faction(self, faction, scores, story_state, temperature)

    # -- Private: Item helpers --
    def _create_item(self, story_state, existing_names, temperature):
        return _item._create_item(self, story_state, existing_names, temperature)

    def _judge_item_quality(self, item, story_state, temperature):
        return _item._judge_item_quality(self, item, story_state, temperature)

    def _refine_item(self, item, scores, story_state, temperature):
        return _item._refine_item(self, item, scores, story_state, temperature)

    # -- Private: Concept helpers --
    def _create_concept(self, story_state, existing_names, temperature):
        return _concept._create_concept(self, story_state, existing_names, temperature)

    def _judge_concept_quality(self, concept, story_state, temperature):
        return _concept._judge_concept_quality(self, concept, story_state, temperature)

    def _refine_concept(self, concept, scores, story_state, temperature):
        return _concept._refine_concept(self, concept, scores, story_state, temperature)

    # -- Private: Relationship helpers --
    @staticmethod
    def _is_duplicate_relationship(source_name, target_name, rel_type, existing_rels):
        return _relationship._is_duplicate_relationship(
            source_name, target_name, rel_type, existing_rels
        )

    def _create_relationship(self, story_state, entity_names, existing_rels, temperature):
        return _relationship._create_relationship(
            self, story_state, entity_names, existing_rels, temperature
        )

    def _judge_relationship_quality(self, relationship, story_state, temperature):
        return _relationship._judge_relationship_quality(
            self, relationship, story_state, temperature
        )

    def _refine_relationship(self, relationship, scores, story_state, temperature):
        return _relationship._refine_relationship(
            self, relationship, scores, story_state, temperature
        )

    # -- Private: Plot helpers --
    def _judge_plot_quality(self, plot_outline, story_state, temperature):
        return _plot._judge_plot_quality(self, plot_outline, story_state, temperature)

    def _refine_plot(self, plot_outline, scores, story_state, temperature):
        return _plot._refine_plot(self, plot_outline, scores, story_state, temperature)

    # -- Private: Chapter helpers --
    def _judge_chapter_quality(self, chapter, story_state, temperature):
        return _chapter_quality._judge_chapter_quality(self, chapter, story_state, temperature)

    def _refine_chapter_outline(self, chapter, scores, story_state, temperature):
        return _chapter_quality._refine_chapter_outline(
            self, chapter, scores, story_state, temperature
        )

    # -- Validation / Mini descriptions --
    def generate_mini_description(
        self,
        name: str,
        entity_type: str,
        full_description: str,
    ) -> str:
        return _generate_mini_description(self, name, entity_type, full_description)

    def generate_mini_descriptions_batch(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, str]:
        return _generate_mini_descriptions_batch(self, entities)

    async def refine_entity(
        self,
        entity: Entity | None,
        story_brief: StoryBrief | None,
    ) -> dict[str, Any] | None:
        return await _refine_entity(self, entity, story_brief)

    async def regenerate_entity(
        self,
        entity: Entity | None,
        story_brief: StoryBrief | None,
        custom_instructions: str | None = None,
    ) -> dict[str, Any] | None:
        return await _regenerate_entity(self, entity, story_brief, custom_instructions)

    async def suggest_relationships_for_entity(
        self,
        entity: Entity,
        available_entities: list[Entity],
        existing_relationships: list[dict[str, str]],
        story_brief: StoryBrief | None,
        num_suggestions: int = 3,
    ) -> list[dict[str, Any]]:
        return await _suggest_relationships_for_entity(
            self, entity, available_entities, existing_relationships, story_brief, num_suggestions
        )

    async def extract_entity_claims(
        self,
        entity: Entity,
    ) -> list[dict[str, str]]:
        return await _extract_entity_claims(self, entity)

    async def check_contradiction(
        self,
        claim_a: dict[str, str],
        claim_b: dict[str, str],
    ) -> dict[str, Any] | None:
        return await _check_contradiction(self, claim_a, claim_b)

    async def validate_entity_consistency(
        self,
        entities: list[Entity],
        max_comparisons: int = 50,
    ) -> list[dict[str, Any]]:
        return await _validate_entity_consistency(self, entities, max_comparisons)
