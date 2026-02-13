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
from src.services.world_quality_service._analytics import (
    log_refinement_analytics as _log_refinement_analytics,
)
from src.services.world_quality_service._analytics import (
    record_entity_quality as _record_entity_quality,
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
from src.services.world_quality_service._formatting import (
    calculate_eta as _calculate_eta,
)
from src.services.world_quality_service._formatting import (
    format_existing_names_warning as _format_existing_names_warning,
)
from src.services.world_quality_service._formatting import (
    format_properties as _format_properties,
)
from src.services.world_quality_service._item import (
    generate_item_with_quality as _generate_item_with_quality,
)
from src.services.world_quality_service._location import (
    generate_location_with_quality as _generate_location_with_quality,
)
from src.services.world_quality_service._model_cache import ModelResolutionCache
from src.services.world_quality_service._model_resolver import (
    get_creator_model as _get_creator_model,
)
from src.services.world_quality_service._model_resolver import (
    get_judge_model as _get_judge_model,
)
from src.services.world_quality_service._model_resolver import (
    resolve_model_for_role as _resolve_model_for_role,
)
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

    # Static method delegates
    _calculate_eta = staticmethod(_calculate_eta)
    _format_properties = staticmethod(_format_properties)

    def __init__(self, settings: Settings, mode_service: ModelModeService):
        """
        Create a WorldQualityService configured with application settings and a model mode service used for model resolution.

        Parameters:
            settings (Settings): Application configuration and feature settings used by the service.
            mode_service (ModelModeService): Service responsible for selecting and managing model modes.
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
        Persist entity quality scores and refinement metrics to the analytics database for later analysis.

        Parameters:
            project_id (str): Project identifier associated with the entity.
            entity_type (str): Type of the entity (e.g., "character", "faction").
            entity_name (str): Name or unique identifier of the entity instance.
            scores (dict[str, Any]): Raw quality scores or structured score breakdown for the entity.
            iterations (int): Number of refinement iterations performed.
            generation_time (float): Total generation/refinement time in seconds.
            model_id (str | None): Identifier of the model used for generation or refinement, if available.
            early_stop_triggered (bool): Whether the refinement loop stopped early due to a stopping condition.
            threshold_met (bool): Whether a configured quality threshold was reached.
            peak_score (float | None): Highest score observed during refinement.
            final_score (float | None): Score observed at the final iteration.
            score_progression (list[float] | None): Sequence of scores recorded across iterations.
            consecutive_degradations (int): Number of consecutive iterations where score decreased.
            best_iteration (int): Index of the iteration that produced the best score.
            quality_threshold (float | None): Configured quality threshold used to determine success.
            max_iterations (int | None): Maximum allowed iterations for the refinement loop.

        Notes:
            This method records metrics for analytics and does not return a value.
        """
        _record_entity_quality(
            self,
            project_id,
            entity_type,
            entity_name,
            scores,
            iterations,
            generation_time,
            model_id,
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

    @property
    def client(self) -> ollama.Client:
        """
        Provide an Ollama client configured with a timeout scaled to the writer model's size; creates and caches the client on first access.

        Returns:
            ollama.Client: The cached or newly created Ollama client configured with the service host and a timeout derived from the writer model.
        """
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
        """
        Provide the judge consistency configuration derived from the service settings.

        Returns:
            JudgeConsistencyConfig: Configuration for judge consistency constructed from the current settings.
        """
        return JudgeConsistencyConfig.from_settings(self.settings)

    def _resolve_model_for_role(self, agent_role: str) -> str:
        """
        Resolve which model should be used for the given agent role, honoring the Settings hierarchy.

        Parameters:
            agent_role (str): Agent role name (e.g., "writer", "judge") to resolve a model for.

        Returns:
            model (str): Identifier of the model to use for the specified role.
        """
        return _resolve_model_for_role(self, agent_role)

    def invalidate_model_cache(self) -> None:
        """
        Clear cached model resolution mappings.

        Forces subsequent model-resolution calls to recompute models instead of using cached results.
        """
        self._model_cache.invalidate()

    def _get_creator_model(self, entity_type: str | None = None) -> str:
        """
        Resolve the creator model identifier to use for a given entity type.

        Parameters:
            entity_type (str | None): Optional entity type (e.g., "character", "faction", "location") used to select a specialized creator model. If None, the default creator model is returned.

        Returns:
            str: The model name or identifier to use for creative generation.
        """
        return _get_creator_model(self, entity_type)

    def _get_judge_model(self, entity_type: str | None = None) -> str:
        """
        Resolve the judge model name to use for a given entity type.

        Parameters:
            entity_type (str | None): Optional entity type to select a specialized judge model; if None a default judge model is chosen.

        Returns:
            model_name (str): The resolved judge model identifier.
        """
        return _get_judge_model(self, entity_type)

    def _format_existing_names_warning(self, existing_names: list[str], entity_type: str) -> str:
        """
        Produce a warning string listing existing names and explicit "DO NOT" examples to discourage duplicate names for the given entity type.

        Parameters:
                existing_names (list[str]): Names already present for the entity type.
                entity_type (str): Human-readable entity type label (e.g., "character", "location") used in the warning text.

        Returns:
                warning (str): A formatted warning message that enumerates existing names and includes clear "DO NOT" duplicate examples.
        """
        return _format_existing_names_warning(existing_names, entity_type)

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
        Log and persist analytics about a completed refinement history.

        Parameters:
            history (RefinementHistory): The refinement history containing iteration records and metrics.
            project_id (str): Identifier of the project or story associated with the refinement.
            early_stop_triggered (bool, optional): Whether the refinement stopped early due to a stopping condition.
            threshold_met (bool, optional): Whether the target quality threshold was reached during refinement.
            quality_threshold (float | None, optional): The quality threshold that was used, if any.
            max_iterations (int | None, optional): The maximum number of refinement iterations allowed, if configured.
        """
        _log_refinement_analytics(
            self,
            history,
            project_id,
            early_stop_triggered=early_stop_triggered,
            threshold_met=threshold_met,
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
        existing_rels: list[tuple[str, str, str]],
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
        existing_rels: list[tuple[str, str, str]],
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
