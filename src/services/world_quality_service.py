"""World Quality Service - multi-model iteration for world building quality.

Implements a generate-judge-refine loop using:
- Creator model: High temperature (0.9) for creative generation
- Judge model: Low temperature (0.1) for consistent evaluation
- Refinement: Incorporates feedback to improve entities
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import ollama

from src.memory.entities import Entity
from src.memory.mode_database import ModeDatabase
from src.memory.story_state import (
    Character,
    Concept,
    Faction,
    Item,
    Location,
    StoryBrief,
    StoryState,
)
from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RefinementConfig,
    RefinementHistory,
    RelationshipQualityScores,
)
from src.services.llm_client import generate_structured
from src.services.model_mode_service import ModelModeService
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError
from src.utils.json_parser import extract_json
from src.utils.validation import validate_not_empty, validate_unique_name

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

    # ========== CHARACTER GENERATION WITH QUALITY ==========

    def generate_character_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        custom_instructions: str | None = None,
    ) -> tuple[Character, CharacterQualityScores, int]:
        """
        Generate a character and iteratively refine it until a quality threshold or stopping criteria is reached.

        Parameters:
            story_state (StoryState): Current story state containing the brief and identifiers used for generation and analytics.
            existing_names (list[str]): Names to avoid when generating a new character.
            custom_instructions (str | None): Optional additional instructions to influence creator model output.

        Returns:
            tuple[Character, CharacterQualityScores, int]: The selected character, its quality scores, and the number of iterations performed.

        Raises:
            WorldGenerationError: If generation fails and no valid iterations were produced.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for character generation")

        logger.info(f"Generating character with quality threshold {config.quality_threshold}")

        # Track all iterations for best-selection using RefinementHistory
        history = RefinementHistory(entity_type="character", entity_name="")
        iteration = 0
        character: Character | None = None
        scores: CharacterQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                # Create new character on first iteration OR if previous returned None
                # (e.g., validation failure or duplicate name)
                if iteration == 0 or character is None:
                    # Initial generation
                    character = self._create_character(
                        story_state, existing_names, config.creator_temperature, custom_instructions
                    )
                else:
                    # Refinement based on feedback
                    if character and scores:
                        # Use dynamic temperature that decreases over iterations
                        dynamic_temp = config.get_refinement_temperature(iteration + 1)
                        character = self._refine_character(
                            character,
                            scores,
                            story_state,
                            dynamic_temp,
                        )

                if character is None:
                    last_error = f"Character creation returned None on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                # Update history entity name
                if not history.entity_name:
                    history.entity_name = character.name

                # Judge quality - this can raise if parsing fails
                scores = self._judge_character_quality(
                    character, story_state, config.judge_temperature
                )

                # Track this iteration
                history.add_iteration(
                    iteration=iteration + 1,
                    entity_data=character.model_dump(),
                    scores=scores.to_dict(),
                    average_score=scores.average,
                    feedback=scores.feedback,
                )

                logger.info(
                    f"Character '{character.name}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                    f"at iteration {history.best_iteration})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Character '{character.name}' met quality threshold")
                    history.final_iteration = iteration + 1
                    history.final_score = scores.average
                    self._log_refinement_analytics(
                        history,
                        story_state.id,
                        threshold_met=True,
                        early_stop_triggered=False,
                        quality_threshold=config.quality_threshold,
                        max_iterations=config.max_iterations,
                    )
                    return character, scores, iteration + 1

                # Check for early stopping after tracking iteration (enhanced with variance tolerance)
                if history.should_stop_early(
                    config.early_stopping_patience,
                    min_iterations=config.early_stopping_min_iterations,
                    variance_tolerance=config.early_stopping_variance_tolerance,
                ):
                    logger.info(
                        f"Early stopping: Character '{character.name}' quality degraded "
                        f"for {history.consecutive_degradations} consecutive iterations "
                        f"(patience: {config.early_stopping_patience}). "
                        f"Stopping at iteration {iteration + 1}."
                    )
                    break

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Character generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # Didn't meet threshold - return BEST iteration, not last
        if not history.iterations:
            raise WorldGenerationError(
                f"Failed to generate character after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # Pick best iteration (not necessarily the last one)
        best_entity = history.get_best_entity()

        if best_entity and history.best_iteration != len(history.iterations):
            # We have a better iteration than the last one
            logger.warning(
                f"Character '{history.entity_name}' iterations got WORSE after peak. "
                f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
                f"Final: iteration {len(history.iterations)} "
                f"({history.iterations[-1].average_score:.1f}). "
                f"Returning best iteration."
            )

        # Return best entity or last one
        if best_entity:
            best_character = Character(**best_entity)
            # Reconstruct scores from best iteration
            for record in history.iterations:
                if record.iteration == history.best_iteration:
                    best_scores = CharacterQualityScores(**record.scores)
                    break
            if best_scores:
                history.final_iteration = history.best_iteration
                history.final_score = history.peak_score
                # Early stop = exited before max iterations due to degradation patience.
                # If we're in this block (returning best entity), we didn't meet
                # threshold early (that returns from within the loop), so
                # iterations < max means degradation-based early stop triggered.
                was_early_stop = len(history.iterations) < config.max_iterations
                self._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=history.peak_score >= config.quality_threshold,
                    early_stop_triggered=was_early_stop,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return best_character, best_scores, history.best_iteration

        # Fallback to last iteration
        if character and scores:
            logger.warning(
                f"Character '{character.name}' did not meet quality threshold "
                f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
            )
            history.final_iteration = len(history.iterations)
            history.final_score = scores.average
            self._log_refinement_analytics(
                history,
                story_state.id,
                threshold_met=False,
                early_stop_triggered=False,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            return character, scores, len(history.iterations)

        raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
            f"Failed to generate character after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    def _create_character(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
        custom_instructions: str | None = None,
    ) -> Character | None:
        """Create a new character using the creator model.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            temperature: Generation temperature.
            custom_instructions: Optional custom instructions to refine generation.

        Returns:
            New Character or None on failure.
        """
        brief = story_state.brief
        if not brief:
            return None

        # Random naming hint to encourage variety
        naming_styles = [
            "Use an unexpected, fresh name - avoid common fantasy names like Elara, Kael, Thorne, or Lyra.",
            "Draw inspiration from diverse cultures for a unique name.",
            "Create a memorable name that reflects the character's personality.",
            "Use a short, punchy name or a longer, elaborate one - be creative.",
        ]
        naming_hint = random.choice(naming_styles)

        # Build custom instructions section if provided
        custom_section = ""
        if custom_instructions:
            custom_section = f"\n\nSPECIFIC REQUIREMENTS:\n{custom_instructions}\n"

        prompt = f"""Create a compelling NEW character for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}
SETTING: {brief.setting_place}, {brief.setting_time}

EXISTING CHARACTERS IN THIS WORLD: {", ".join(existing_names) if existing_names else "None yet"}
(Create a NEW character with a different name that complements these existing ones)

NAMING: {naming_hint}
{custom_section}
Create a character with:
1. Deep psychological complexity - internal contradictions, layers
2. Clear goals - what they want vs what they need
3. Meaningful flaws that drive conflict
4. Uniqueness - not a genre archetype
5. Arc potential - room for transformation

Write all text in {brief.language}."""

        try:
            model = self._get_creator_model(entity_type="character")
            character = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Character,
                temperature=temperature,
            )
            return character
        except Exception as e:
            logger.error("Character creation error for story %s: %s", story_state.id, e)
            raise WorldGenerationError(f"Character creation failed: {e}") from e

    def _judge_character_quality(
        self,
        character: Character,
        story_state: StoryState,
        temperature: float,
    ) -> CharacterQualityScores:
        """Judge character quality using the validator model.

        Args:
            character: Character to evaluate.
            story_state: Current story state for context.
            temperature: Judge temperature (low for consistency).

        Returns:
            CharacterQualityScores with ratings and feedback.

        Raises:
            WorldGenerationError: If quality judgment fails or returns invalid data.
        """
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are a literary critic evaluating character quality for a {genre} story.

CHARACTER TO EVALUATE:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

Rate each dimension 0-10:
- depth: Psychological complexity, internal contradictions, layers
- goals: Clarity, story relevance, want vs need tension
- flaws: Meaningful vulnerabilities that drive conflict
- uniqueness: Distinctiveness from genre archetypes
- arc_potential: Room for transformation and growth

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"depth": <number>, "goals": <number>, "flaws": <number>, "uniqueness": <number>, "arc_potential": <number>, "feedback": "<string>"}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

        try:
            model = self._get_judge_model(entity_type="character")
            return generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=CharacterQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            logger.exception("Character quality judgment failed for '%s': %s", character.name, e)
            raise WorldGenerationError(f"Character quality judgment failed: {e}") from e

    def _refine_character(
        self,
        character: Character,
        scores: CharacterQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> Character:
        """Refine a character based on quality feedback.

        Args:
            character: Character to refine.
            scores: Quality scores with feedback.
            story_state: Current story state.
            temperature: Refinement temperature.

        Returns:
            Refined Character.
        """
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this character based on quality feedback.

ORIGINAL CHARACTER:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

QUALITY SCORES (0-10):
- Depth: {scores.depth}
- Goals: {scores.goals}
- Flaws: {scores.flaws}
- Uniqueness: {scores.uniqueness}
- Arc Potential: {scores.arc_potential}

FEEDBACK: {scores.feedback}

WEAK AREAS TO IMPROVE: {", ".join(weak) if weak else "None - minor improvements only"}

Keep the name "{character.name}" and role "{character.role}", but enhance the weak areas.
Make the character more compelling while maintaining consistency.
Write all text in {brief.language if brief else "English"}."""

        try:
            model = self._get_creator_model(entity_type="character")
            return generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Character,
                temperature=temperature,
            )
        except Exception as e:
            logger.exception("Character refinement failed for '%s': %s", character.name, e)
            raise WorldGenerationError(f"Character refinement failed: {e}") from e

    # ========== LOCATION GENERATION WITH QUALITY ==========

    def generate_location_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], LocationQualityScores, int]:
        """
        Generate a location and iteratively refine it until it meets the configured quality threshold or retries are exhausted.

        Parameters:
            story_state (StoryState): Current story state containing the brief and project id used for prompts and analytics.
            existing_names (list[str]): Names of existing locations to avoid producing duplicates.

        Returns:
            tuple: (location_dict, scores, iterations_used)
                location_dict (dict): Generated location fields (e.g., name, type, description, significance).
                scores (LocationQualityScores): Quality evaluation for the returned location.
                iterations_used (int): Number of create/refine iterations performed for the returned result.

        Raises:
            WorldGenerationError: If no valid location could be produced after all attempts.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for location generation")

        logger.info(f"Generating location with quality threshold {config.quality_threshold}")

        # Track all iterations for best-selection using RefinementHistory
        history = RefinementHistory(entity_type="location", entity_name="")
        iteration = 0
        location: dict[str, Any] = {}
        scores: LocationQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                # Create new location on first iteration OR if previous returned empty
                # (e.g., duplicate name detection returns {} to force retry)
                if iteration == 0 or not location.get("name"):
                    location = self._create_location(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if location and scores:
                        # Use dynamic temperature that decreases over iterations
                        dynamic_temp = config.get_refinement_temperature(iteration + 1)
                        location = self._refine_location(
                            location,
                            scores,
                            story_state,
                            dynamic_temp,
                        )

                if not location.get("name"):
                    last_error = f"Location creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                # Update history entity name
                if not history.entity_name:
                    history.entity_name = location.get("name", "Unknown")

                scores = self._judge_location_quality(
                    location, story_state, config.judge_temperature
                )

                # Track this iteration
                history.add_iteration(
                    iteration=iteration + 1,
                    entity_data=location.copy(),
                    scores=scores.to_dict(),
                    average_score=scores.average,
                    feedback=scores.feedback,
                )

                logger.info(
                    f"Location '{location.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                    f"at iteration {history.best_iteration})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Location '{location.get('name')}' met quality threshold")
                    history.final_iteration = iteration + 1
                    history.final_score = scores.average
                    self._log_refinement_analytics(
                        history,
                        story_state.id,
                        threshold_met=True,
                        early_stop_triggered=False,
                        quality_threshold=config.quality_threshold,
                        max_iterations=config.max_iterations,
                    )
                    return location, scores, iteration + 1

                # Check for early stopping after tracking iteration (enhanced with variance tolerance)
                if history.should_stop_early(
                    config.early_stopping_patience,
                    min_iterations=config.early_stopping_min_iterations,
                    variance_tolerance=config.early_stopping_variance_tolerance,
                ):
                    logger.info(
                        f"Early stopping: Location '{location.get('name')}' quality degraded "
                        f"for {history.consecutive_degradations} consecutive iterations "
                        f"(patience: {config.early_stopping_patience}). "
                        f"Stopping at iteration {iteration + 1}."
                    )
                    break

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Location generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # Didn't meet threshold - return BEST iteration, not last
        if not history.iterations:
            raise WorldGenerationError(
                f"Failed to generate location after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # Pick best iteration (not necessarily the last one)
        best_entity = history.get_best_entity()

        if best_entity and history.best_iteration != len(history.iterations):
            logger.warning(
                f"Location '{history.entity_name}' iterations got WORSE after peak. "
                f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
                f"Final: iteration {len(history.iterations)} "
                f"({history.iterations[-1].average_score:.1f}). "
                f"Returning best iteration."
            )

        # Return best entity or last one
        if best_entity:
            # Reconstruct scores from best iteration
            for record in history.iterations:
                if record.iteration == history.best_iteration:
                    best_scores = LocationQualityScores(**record.scores)
                    break
            if best_scores:
                history.final_iteration = history.best_iteration
                history.final_score = history.peak_score
                was_early_stop = len(history.iterations) < config.max_iterations
                self._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=history.peak_score >= config.quality_threshold,
                    early_stop_triggered=was_early_stop,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return best_entity, best_scores, history.best_iteration

        # Fallback to last iteration
        if location.get("name") and scores:
            logger.warning(
                f"Location '{location.get('name')}' did not meet quality threshold "
                f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
            )
            history.final_iteration = len(history.iterations)
            history.final_score = scores.average
            self._log_refinement_analytics(
                history,
                story_state.id,
                threshold_met=False,
                early_stop_triggered=False,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            return location, scores, len(history.iterations)

        raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
            f"Failed to generate location after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    def _create_location(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new location using the creator model with structured generation."""
        brief = story_state.brief
        if not brief:
            return {}

        # Format existing names with explicit warnings
        existing_names_formatted = self._format_existing_names_warning(existing_names, "location")

        prompt = f"""Create a compelling location for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}

=== CRITICAL: UNIQUENESS REQUIREMENTS ===
{existing_names_formatted}

STRICT RULES:
- DO NOT use any name from the list above
- DO NOT use case variations (e.g., "Forest" vs "FOREST")
- DO NOT use similar names (e.g., "Dark Woods" vs "The Dark Wood")
- Create something COMPLETELY DIFFERENT

Create a location with:
1. Rich atmosphere - sensory details, mood
2. Narrative significance - symbolic or plot meaning
3. Strong story relevance - connections to themes/characters
4. Distinctiveness - memorable unique qualities

Write all text in {brief.language}."""

        try:
            model = self._get_creator_model(entity_type="location")
            location = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Location,
                temperature=temperature,
            )

            # Comprehensive uniqueness validation (with optional semantic checking)
            if location.name:
                is_unique, conflicting_name, reason = validate_unique_name(
                    location.name,
                    existing_names,
                    check_semantic=self.settings.semantic_duplicate_enabled,
                    semantic_threshold=self.settings.semantic_duplicate_threshold,
                    ollama_url=self.settings.ollama_url,
                    embedding_model=self.settings.embedding_model,
                )
                if not is_unique:
                    logger.warning(
                        f"Location name '{location.name}' conflicts with '{conflicting_name}' "
                        f"(reason: {reason}), clearing to force retry"
                    )
                    return {}  # Return empty to trigger retry

            # Convert to dict for compatibility with existing code
            return location.model_dump()
        except Exception as e:
            logger.exception("Location creation failed for story %s: %s", story_state.id, e)
            raise WorldGenerationError(f"Location creation failed: {e}") from e

    def _judge_location_quality(
        self,
        location: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> LocationQualityScores:
        """Judge location quality using the validator model.

        Raises:
            WorldGenerationError: If quality judgment fails or returns invalid data.
        """
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a location for a {genre} story.

LOCATION TO EVALUATE:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}

Rate each dimension 0-10:
- atmosphere: Sensory richness, mood, immersion
- significance: Plot or symbolic meaning
- story_relevance: Connections to themes and characters
- distinctiveness: Memorable, unique qualities

Provide specific improvement feedback in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"atmosphere": <number>, "significance": <number>, "story_relevance": <number>, "distinctiveness": <number>, "feedback": "<string>"}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

        try:
            model = self._get_judge_model(entity_type="location")
            return generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=LocationQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            logger.exception(
                "Location quality judgment failed for '%s': %s", location.get("name", "Unknown"), e
            )
            raise WorldGenerationError(f"Location quality judgment failed: {e}") from e

    def _refine_location(
        self,
        location: dict[str, Any],
        scores: LocationQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a location based on quality feedback using structured generation."""
        brief = story_state.brief

        # Build specific improvement instructions from feedback
        improvement_focus = []
        if scores.atmosphere < 8:
            improvement_focus.append("Add richer sensory details and mood")
        if scores.significance < 8:
            improvement_focus.append("Deepen the plot or symbolic meaning")
        if scores.story_relevance < 8:
            improvement_focus.append("Strengthen connections to themes and characters")
        if scores.distinctiveness < 8:
            improvement_focus.append("Make more memorable with unique qualities")

        prompt = f"""TASK: Improve this location to score HIGHER on the weak dimensions.

ORIGINAL LOCATION:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}

CURRENT SCORES (need 9+ in all areas):
- Atmosphere: {scores.atmosphere}/10
- Significance: {scores.significance}/10
- Story Relevance: {scores.story_relevance}/10
- Distinctiveness: {scores.distinctiveness}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the exact name: "{location.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements to weak areas
3. Add concrete sensory details, not vague generalities
4. Output in {brief.language if brief else "English"}

Return ONLY the improved location."""

        try:
            model = self._get_creator_model(entity_type="location")
            refined = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Location,
                temperature=temperature,
            )

            # Ensure name is preserved from original location
            result = refined.model_dump()
            result["name"] = location.get("name", "Unknown")
            result["type"] = "location"
            return result
        except Exception as e:
            logger.exception(
                "Location refinement failed for '%s': %s", location.get("name") or "Unknown", e
            )
            raise WorldGenerationError(f"Location refinement failed: {e}") from e

    # ========== RELATIONSHIP GENERATION WITH QUALITY ==========

    def generate_relationship_with_quality(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
    ) -> tuple[dict[str, Any], RelationshipQualityScores, int]:
        """Generate a relationship with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            entity_names: Names of entities that can have relationships.
            existing_rels: Existing (source, target) pairs to avoid.

        Returns:
            Tuple of (relationship_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If relationship generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for relationship generation")

        if len(entity_names) < 2:
            raise ValueError("Need at least 2 entities for relationship generation")

        logger.info(f"Generating relationship with quality threshold {config.quality_threshold}")

        # Track all iterations for best-selection using RefinementHistory
        history = RefinementHistory(entity_type="relationship", entity_name="")
        iteration = 0
        relationship: dict[str, Any] = {}
        scores: RelationshipQualityScores | None = None
        last_error: str = ""
        needs_fresh_creation = True  # Track whether we need fresh creation vs refinement

        while iteration < config.max_iterations:
            try:
                # Create new relationship on first iteration OR if previous was invalid/duplicate
                if needs_fresh_creation:
                    relationship = self._create_relationship(
                        story_state, entity_names, existing_rels, config.creator_temperature
                    )
                else:
                    if relationship and scores:
                        # Use dynamic temperature that decreases over iterations
                        dynamic_temp = config.get_refinement_temperature(iteration + 1)
                        relationship = self._refine_relationship(
                            relationship,
                            scores,
                            story_state,
                            dynamic_temp,
                        )

                if not relationship.get("source") or not relationship.get("target"):
                    last_error = (
                        f"Relationship creation returned incomplete on iteration {iteration + 1}"
                    )
                    logger.error(last_error)
                    needs_fresh_creation = True  # Retry with fresh creation
                    iteration += 1
                    continue

                # Check for duplicate relationship
                source = relationship.get("source", "")
                target = relationship.get("target", "")
                rel_type = relationship.get("relation_type", "knows")
                if self._is_duplicate_relationship(source, target, rel_type, existing_rels):
                    last_error = f"Generated duplicate relationship {source} -> {target}"
                    logger.warning(last_error)
                    needs_fresh_creation = True  # Retry with fresh creation
                    iteration += 1
                    continue

                # Got a valid relationship - can proceed to refinement on next iteration
                needs_fresh_creation = False

                # Update history entity name
                if not history.entity_name:
                    history.entity_name = f"{source} -> {target}"

                scores = self._judge_relationship_quality(
                    relationship, story_state, config.judge_temperature
                )

                # Track this iteration
                history.add_iteration(
                    iteration=iteration + 1,
                    entity_data=relationship.copy(),
                    scores=scores.to_dict(),
                    average_score=scores.average,
                    feedback=scores.feedback,
                )

                logger.info(
                    f"Relationship '{source} -> {target}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                    f"at iteration {history.best_iteration})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info("Relationship met quality threshold")
                    history.final_iteration = iteration + 1
                    history.final_score = scores.average
                    self._log_refinement_analytics(
                        history,
                        story_state.id,
                        threshold_met=True,
                        early_stop_triggered=False,
                        quality_threshold=config.quality_threshold,
                        max_iterations=config.max_iterations,
                    )
                    return relationship, scores, iteration + 1

                # Check for early stopping after tracking iteration (enhanced with variance tolerance)
                if history.should_stop_early(
                    config.early_stopping_patience,
                    min_iterations=config.early_stopping_min_iterations,
                    variance_tolerance=config.early_stopping_variance_tolerance,
                ):
                    logger.info(
                        f"Early stopping: Relationship '{source} -> {target}' quality degraded "
                        f"for {history.consecutive_degradations} consecutive iterations "
                        f"(patience: {config.early_stopping_patience}). "
                        f"Stopping at iteration {iteration + 1}."
                    )
                    break

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Relationship generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # Didn't meet threshold - return BEST iteration, not last
        if not history.iterations:
            raise WorldGenerationError(
                f"Failed to generate relationship after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # Pick best iteration (not necessarily the last one)
        best_entity = history.get_best_entity()

        if best_entity and history.best_iteration != len(history.iterations):
            logger.warning(
                f"Relationship '{history.entity_name}' iterations got WORSE after peak. "
                f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
                f"Final: iteration {len(history.iterations)} "
                f"({history.iterations[-1].average_score:.1f}). "
                f"Returning best iteration."
            )

        # Return best entity or last one
        if best_entity:
            # Reconstruct scores from best iteration
            for record in history.iterations:
                if record.iteration == history.best_iteration:
                    best_scores = RelationshipQualityScores(**record.scores)
                    break
            if best_scores:
                history.final_iteration = history.best_iteration
                history.final_score = history.peak_score
                was_early_stop = len(history.iterations) < config.max_iterations
                self._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=history.peak_score >= config.quality_threshold,
                    early_stop_triggered=was_early_stop,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return best_entity, best_scores, history.best_iteration

        # Fallback to last iteration
        if relationship.get("source") and relationship.get("target") and scores:
            logger.warning(
                f"Relationship '{relationship.get('source')} -> {relationship.get('target')}' "
                f"did not meet quality threshold ({scores.average:.1f} < {config.quality_threshold}), "
                f"returning anyway"
            )
            history.final_iteration = len(history.iterations)
            history.final_score = scores.average
            self._log_refinement_analytics(
                history,
                story_state.id,
                threshold_met=False,
                early_stop_triggered=False,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            return relationship, scores, len(history.iterations)

        raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
            f"Failed to generate relationship after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    def _is_duplicate_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        existing_rels: list[tuple[str, str]],
    ) -> bool:
        """Check if a relationship already exists (in either direction for same type).

        Args:
            source: Source entity name.
            target: Target entity name.
            rel_type: Relationship type.
            existing_rels: List of existing (source, target) pairs.

        Returns:
            True if this relationship already exists.
        """
        for existing_source, existing_target in existing_rels:
            # Check both directions
            same_pair = (source == existing_source and target == existing_target) or (
                source == existing_target and target == existing_source
            )
            if same_pair:
                return True
        return False

    def _create_relationship(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new relationship using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        existing_rel_strs = [f"{s} -> {t}" for s, t in existing_rels]

        prompt = f"""Create a compelling relationship between entities for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

AVAILABLE ENTITIES: {", ".join(entity_names)}
EXISTING RELATIONSHIPS (avoid): {", ".join(existing_rel_strs[:10]) if existing_rel_strs else "None"}

Create a relationship with:
1. Tension - conflict potential
2. Complex dynamics - power balance, history
3. Story potential - opportunities for scenes
4. Authenticity - believable connection

Output ONLY valid JSON (all text in {brief.language}):
{{
    "source": "Entity Name 1",
    "target": "Entity Name 2",
    "relation_type": "knows|loves|hates|allies_with|enemies_with|located_in|owns|member_of",
    "description": "Description of the relationship with history and dynamics"
}}"""

        try:
            model = self._get_creator_model(entity_type="relationship")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_relationship_create,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                return data
            else:
                logger.error(f"Relationship creation returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid relationship JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error("Relationship creation LLM error for story %s: %s", story_state.id, e)
            raise WorldGenerationError(f"LLM error during relationship creation: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Relationship creation JSON parsing error for story %s: %s", story_state.id, e
            )
            raise WorldGenerationError(f"Invalid relationship response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error in relationship creation for story %s: %s", story_state.id, e
            )
            raise WorldGenerationError(f"Unexpected relationship creation error: {e}") from e

    def _judge_relationship_quality(
        self,
        relationship: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> RelationshipQualityScores:
        """Judge relationship quality using the validator model.

        Raises:
            WorldGenerationError: If quality judgment fails or returns invalid data.
        """
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a relationship for a {genre} story.

RELATIONSHIP TO EVALUATE:
Source: {relationship.get("source", "Unknown")}
Target: {relationship.get("target", "Unknown")}
Type: {relationship.get("relation_type", "unknown")}
Description: {relationship.get("description", "")}

Rate each dimension 0-10:
- TENSION: Conflict potential
- DYNAMICS: Complexity, power balance, history
- STORY_POTENTIAL: Opportunities for scenes and development
- AUTHENTICITY: Believability of the connection

Provide specific improvement feedback.

IMPORTANT: Evaluate honestly. Do NOT copy these example values - provide YOUR OWN assessment scores.

Output ONLY valid JSON:
{{"tension": <your_score>, "dynamics": <your_score>, "story_potential": <your_score>, "authenticity": <your_score>, "feedback": "<your_specific_feedback>"}}"""

        try:
            model = self._get_judge_model(entity_type="relationship")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_relationship_judge,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                # Validate all required fields are present
                required_fields = ["tension", "dynamics", "story_potential", "authenticity"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Relationship judge response missing required fields: {missing}"
                    )

                return RelationshipQualityScores(
                    tension=float(data["tension"]),
                    dynamics=float(data["dynamics"]),
                    story_potential=float(data["story_potential"]),
                    authenticity=float(data["authenticity"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from relationship judge: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(
                "Relationship quality judgment failed for %s->%s: %s",
                relationship.get("source") or "Unknown",
                relationship.get("target") or "Unknown",
                e,
            )
            raise WorldGenerationError(f"Relationship quality judgment failed: {e}") from e

    def _refine_relationship(
        self,
        relationship: dict[str, Any],
        scores: RelationshipQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a relationship based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this relationship based on quality feedback.

ORIGINAL RELATIONSHIP:
Source: {relationship.get("source", "Unknown")}
Target: {relationship.get("target", "Unknown")}
Type: {relationship.get("relation_type", "unknown")}
Description: {relationship.get("description", "")}

QUALITY SCORES (0-10):
- Tension: {scores.tension}
- Dynamics: {scores.dynamics}
- Story Potential: {scores.story_potential}
- Authenticity: {scores.authenticity}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep source/target/type, enhance the description and weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "source": "{relationship.get("source", "Unknown")}",
    "target": "{relationship.get("target", "Unknown")}",
    "relation_type": "{relationship.get("relation_type", "knows")}",
    "description": "Improved description with more depth"
}}"""

        try:
            model = self._get_creator_model(entity_type="relationship")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_relationship_refine,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                return data
            else:
                logger.error(f"Relationship refinement returned invalid JSON structure: {data}")
                raise WorldGenerationError(
                    f"Invalid relationship refinement JSON structure: {data}"
                )
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(
                "Relationship refinement LLM error for %s->%s: %s",
                relationship.get("source") or "Unknown",
                relationship.get("target") or "Unknown",
                e,
            )
            raise WorldGenerationError(f"LLM error during relationship refinement: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Relationship refinement JSON parsing error for %s->%s: %s",
                relationship.get("source") or "Unknown",
                relationship.get("target") or "Unknown",
                e,
            )
            raise WorldGenerationError(
                f"Invalid relationship refinement response format: {e}"
            ) from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error in relationship refinement for %s->%s: %s",
                relationship.get("source") or "Unknown",
                relationship.get("target") or "Unknown",
                e,
            )
            raise WorldGenerationError(f"Unexpected relationship refinement error: {e}") from e

    # ========== FACTION GENERATION WITH QUALITY ==========

    def generate_faction_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        existing_locations: list[str] | None = None,
    ) -> tuple[dict[str, Any], FactionQualityScores, int]:
        """
        Generate a faction using an iterative create-refine-judge loop and return the best iteration.

        Performs up to the configured maximum iterations, preserves the best-scoring iteration (not necessarily the last), and logs refinement analytics including peak and final scores.

        Parameters:
            story_state (StoryState): Current story state containing the brief and context used to ground faction generation.
            existing_names (list[str]): Existing faction names to avoid duplicates; used to enforce unique naming.
            existing_locations (list[str] | None): Optional list of location names for spatial grounding and contextual details.

        Returns:
            tuple[dict[str, Any], FactionQualityScores, int]: A tuple of (faction_dict, quality_scores, iterations_used), where `faction_dict` is the chosen faction representation, `quality_scores` are the evaluated scores for that faction, and `iterations_used` is the iteration number returned.

        Raises:
            WorldGenerationError: If faction generation fails to produce any valid iterations.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for faction generation")

        logger.info(f"Generating faction with quality threshold {config.quality_threshold}")

        # Track all iterations for best-selection
        history = RefinementHistory(entity_type="faction", entity_name="")
        iteration = 0
        faction: dict[str, Any] = {}
        scores: FactionQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                # Create new faction on first iteration OR if previous returned empty
                # (e.g., duplicate name detection returns {} to force retry)
                if iteration == 0 or not faction.get("name"):
                    faction = self._create_faction(
                        story_state, existing_names, config.creator_temperature, existing_locations
                    )
                else:
                    if faction and scores:
                        # Use dynamic temperature that decreases over iterations
                        dynamic_temp = config.get_refinement_temperature(iteration + 1)
                        faction = self._refine_faction(
                            faction,
                            scores,
                            story_state,
                            dynamic_temp,
                        )

                if not faction.get("name"):
                    last_error = f"Faction creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                # Update history entity name
                if not history.entity_name:
                    history.entity_name = faction.get("name", "Unknown")

                scores = self._judge_faction_quality(faction, story_state, config.judge_temperature)

                # Track this iteration
                history.add_iteration(
                    iteration=iteration + 1,
                    entity_data=faction.copy(),
                    scores=scores.to_dict(),
                    average_score=scores.average,
                    feedback=scores.feedback,
                )

                logger.info(
                    f"Faction '{faction.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                    f"at iteration {history.best_iteration})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Faction '{faction.get('name')}' met quality threshold")
                    history.final_iteration = iteration + 1
                    history.final_score = scores.average
                    # Log analytics
                    self._log_refinement_analytics(
                        history,
                        story_state.id,
                        threshold_met=True,
                        early_stop_triggered=False,
                        quality_threshold=config.quality_threshold,
                        max_iterations=config.max_iterations,
                    )
                    return faction, scores, iteration + 1

                # Check for early stopping after tracking iteration (enhanced with variance tolerance)
                if history.should_stop_early(
                    config.early_stopping_patience,
                    min_iterations=config.early_stopping_min_iterations,
                    variance_tolerance=config.early_stopping_variance_tolerance,
                ):
                    logger.info(
                        f"Early stopping: Faction '{faction.get('name')}' quality degraded "
                        f"for {history.consecutive_degradations} consecutive iterations "
                        f"(patience: {config.early_stopping_patience}). "
                        f"Stopping at iteration {iteration + 1}."
                    )
                    break  # Exit loop early

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Faction generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # Didn't meet threshold - return BEST iteration, not last
        if not history.iterations:
            raise WorldGenerationError(
                f"Failed to generate faction after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # Pick best iteration (not necessarily the last one)
        best_entity = history.get_best_entity()

        if best_entity and history.best_iteration != len(history.iterations):
            # We have a better iteration than the last one
            logger.warning(
                f"Faction '{history.entity_name}' iterations got WORSE after peak. "
                f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
                f"Final: iteration {len(history.iterations)} ({history.iterations[-1].average_score:.1f}). "
                f"Returning best iteration."
            )
            faction = best_entity
            # Find best iteration record by iteration number (not index)
            # This handles cases where some iterations failed and weren't added to the list
            best_record = next(
                (r for r in history.iterations if r.iteration == history.best_iteration),
                None,
            )
            if best_record is None:  # pragma: no cover
                logger.error(
                    f"Best iteration {history.best_iteration} not found in history. "
                    f"Available iterations: {[r.iteration for r in history.iterations]}"
                )
                # Fall back to last iteration
                best_record = history.iterations[-1]
            scores = FactionQualityScores(
                coherence=best_record.scores.get("coherence", 0),
                influence=best_record.scores.get("influence", 0),
                conflict_potential=best_record.scores.get("conflict_potential", 0),
                distinctiveness=best_record.scores.get("distinctiveness", 0),
                feedback=best_record.feedback,
            )
            history.final_iteration = history.best_iteration
            history.final_score = history.peak_score
        else:
            history.final_iteration = len(history.iterations)
            # Reconstruct scores from last iteration if not available
            # Note: In practice, scores should always be set if we have iterations,
            # since add_iteration is called after _judge_faction_quality succeeds.
            # This is defensive code for edge cases that may not occur in practice.
            if scores is None:  # pragma: no cover
                last_record = history.iterations[-1]
                scores = FactionQualityScores(
                    coherence=last_record.scores.get("coherence", 0),
                    influence=last_record.scores.get("influence", 0),
                    conflict_potential=last_record.scores.get("conflict_potential", 0),
                    distinctiveness=last_record.scores.get("distinctiveness", 0),
                    feedback=last_record.feedback,
                )
            history.final_score = scores.average

        logger.warning(
            f"Faction '{history.entity_name}' did not meet quality threshold "
            f"({history.final_score:.1f} < {config.quality_threshold}), "
            f"returning iteration {history.final_iteration}"
        )

        # Log analytics
        was_early_stop = len(history.iterations) < config.max_iterations
        self._log_refinement_analytics(
            history,
            story_state.id,
            threshold_met=history.final_score >= config.quality_threshold,
            early_stop_triggered=was_early_stop,
            quality_threshold=config.quality_threshold,
            max_iterations=config.max_iterations,
        )

        return faction, scores, history.final_iteration

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

    # Diversity hints for faction naming to avoid generic names
    FACTION_NAMING_HINTS: ClassVar[list[str]] = [
        "Use an evocative, specific name - avoid generic names like 'The Guild' or 'The Order'",
        "Draw from historical organizations for inspiration (e.g., Hanseatic League, Templars)",
        "Name could reflect their methods (e.g., 'The Whispered Accord', 'Iron Covenant')",
        "Name could be ironic or misleading (e.g., a violent group called 'The Peacemakers')",
        "Name could reference their origin or founding myth",
        "Name could be in a constructed language or archaic form",
        "Name should be memorable and distinct from common fantasy tropes",
    ]

    FACTION_STRUCTURE_HINTS: ClassVar[list[str]] = [
        "Consider a non-hierarchical structure (cells, councils, rotating leadership)",
        "Structure could mirror their beliefs (egalitarian values = flat structure)",
        "Consider secret inner circles or public-facing vs. true leadership",
        "Structure could be based on expertise, seniority, or divine mandate",
        "Consider how new members join and advance within the organization",
    ]

    FACTION_IDEOLOGY_HINTS: ClassVar[list[str]] = [
        "Core beliefs should create internal tensions or contradictions",
        "Ideology could be a corruption or evolution of older beliefs",
        "Consider what the faction fears or opposes, not just what they support",
        "Ideology should naturally conflict with at least one other group",
        "Consider the gap between stated ideals and actual practices",
    ]

    def _format_existing_names(self, existing_names: list[str]) -> str:
        """
        Format a list of existing faction names into a prompt-ready string that highlights names to avoid duplicating.

        Parameters:
            existing_names (list[str]): Existing faction names to include in the prompt.

        Returns:
            str: Newline-separated names each prefixed with "-". If `existing_names` is empty, returns
            "None yet - you are creating the first faction."
        """
        if not existing_names:
            logger.debug("Formatting existing faction names: none provided")
            return "None yet - you are creating the first faction."

        formatted = []
        for name in existing_names:
            # Also show variations that should be avoided
            formatted.append(f"- {name}")

        logger.debug("Formatted %d existing faction names for prompt", len(formatted))
        return "\n".join(formatted)

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

    def _create_faction(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
        existing_locations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a unique faction definition for the given story using the configured creator model.

        Parameters:
            story_state (StoryState): Story context and brief used to seed faction generation.
            existing_names (list[str]): Existing faction names to avoid; used for strict uniqueness checks.
            temperature (float): Sampling temperature for the creator model.
            existing_locations (list[str] | None): Optional list of world locations to ground the faction's base.

        Returns:
            dict[str, Any]: A faction dictionary with keys including `name`, `type`, `description`, `leader`, `goals`, `values`, and `base_location`. Returns an empty dict when generation should be retried (e.g., name conflict detected).

        Raises:
            WorldGenerationError: If faction generation fails due to unrecoverable model/validation errors.
        """
        brief = story_state.brief
        if not brief:
            return {}

        # Build location context
        location_context = ""
        if existing_locations:
            location_context = f"""
EXISTING LOCATIONS IN THIS WORLD: {", ".join(existing_locations)}
(If applicable, use one of these existing locations as the faction's base)
"""

        # Select random diversity hints for this generation
        naming_hint = random.choice(self.FACTION_NAMING_HINTS)
        structure_hint = random.choice(self.FACTION_STRUCTURE_HINTS)
        ideology_hint = random.choice(self.FACTION_IDEOLOGY_HINTS)

        # Format existing names with clear guidance
        existing_names_formatted = self._format_existing_names(existing_names)

        prompt = f"""Create a compelling faction/organization for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

=== CRITICAL: UNIQUENESS REQUIREMENTS ===
EXISTING FACTIONS (DO NOT DUPLICATE OR CREATE SIMILAR NAMES):
{existing_names_formatted}

STRICT RULES:
- Case variations (e.g., "The Guild" vs "THE GUILD") are NOT acceptable
- Similar names (e.g., "Shadow Council" vs "Council of Shadows") are NOT acceptable
- Names that contain existing faction names are NOT acceptable
- Prefix variations (e.g., "The Order" vs "Order") are NOT acceptable
- Create something COMPLETELY DIFFERENT from the above
{location_context}
=== DIVERSITY GUIDANCE (follow these for this faction) ===
NAMING: {naming_hint}
STRUCTURE: {structure_hint}
IDEOLOGY: {ideology_hint}

Create a faction with:
1. Internal coherence - clear structure, beliefs, and rules
2. World influence - meaningful impact on the setting
3. Conflict potential - natural tensions with other groups
4. Distinctiveness - unique identity and aesthetics
5. Spatial grounding - connection to a specific location (headquarters, base, territory)

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Faction Name",
    "type": "faction",
    "description": "Description of the faction, its history, and purpose (2-3 sentences)",
    "leader": "Name or title of leader (if any)",
    "goals": ["primary goal", "secondary goal"],
    "values": ["core value 1", "core value 2"],
    "base_location": "Name of their headquarters/territory (use one of the existing locations listed above if applicable)"
}}"""

        try:
            model = self._get_creator_model(entity_type="faction")
            # Use structured generation with Pydantic model for reliable output
            faction = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Faction,
                temperature=temperature,
            )

            # Comprehensive uniqueness validation (with optional semantic checking)
            if faction.name:
                is_unique, conflicting_name, reason = validate_unique_name(
                    faction.name,
                    existing_names,
                    check_semantic=self.settings.semantic_duplicate_enabled,
                    semantic_threshold=self.settings.semantic_duplicate_threshold,
                    ollama_url=self.settings.ollama_url,
                    embedding_model=self.settings.embedding_model,
                )
                if not is_unique:
                    logger.warning(
                        f"Faction name '{faction.name}' conflicts with '{conflicting_name}' "
                        f"(reason: {reason}), clearing to force retry"
                    )
                    return {}  # Return empty to trigger retry

            # Convert to dict for compatibility with existing code
            return faction.model_dump()
        except Exception as e:
            logger.exception("Faction creation failed for story %s: %s", story_state.id, e)
            raise WorldGenerationError(f"Faction creation failed: {e}") from e

    def _judge_faction_quality(
        self,
        faction: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> FactionQualityScores:
        """Judge faction quality using the validator model."""
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are a strict quality judge evaluating a faction for a {genre} story.

FACTION TO EVALUATE:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

SCORING GUIDE (use the FULL range):
- 9-10: EXCEPTIONAL - publishable quality, deeply compelling, no improvements needed
- 7-8: GOOD - solid foundation, minor improvements possible
- 5-6: ACCEPTABLE - functional but generic, needs more depth
- 3-4: WEAK - major issues, significant rework needed
- 1-2: POOR - fundamentally flawed, start over

Rate each dimension using this calibration:
- COHERENCE: Internal consistency, clear structure (9+ = seamless logic, no contradictions)
- INFLUENCE: World impact, power level (9+ = shapes entire world, major player)
- CONFLICT_POTENTIAL: Story conflict opportunities (9+ = multiple compelling story hooks)
- DISTINCTIVENESS: Memorable, unique qualities (9+ = unlike any other faction, iconic)

BE HONEST: Award 9-10 ONLY for truly exceptional work. Most factions score 6-8.
Provide SPECIFIC feedback on exactly what would raise each dimension's score.

Output ONLY valid JSON:
{{"coherence": <score>, "influence": <score>, "conflict_potential": <score>, "distinctiveness": <score>, "feedback": "<specific_actionable_feedback>"}}"""

        try:
            model = self._get_judge_model(entity_type="faction")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_faction_judge,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                required_fields = [
                    "coherence",
                    "influence",
                    "conflict_potential",
                    "distinctiveness",
                ]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Faction judge response missing required fields: {missing}"
                    )

                return FactionQualityScores(
                    coherence=float(data["coherence"]),
                    influence=float(data["influence"]),
                    conflict_potential=float(data["conflict_potential"]),
                    distinctiveness=float(data["distinctiveness"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from faction judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(
                "Faction quality judgment failed for '%s': %s",
                faction.get("name") or "Unknown",
                e,
            )
            raise WorldGenerationError(f"Faction quality judgment failed: {e}") from e

    def _refine_faction(
        self,
        faction: dict[str, Any],
        scores: FactionQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a faction based on quality feedback."""
        brief = story_state.brief

        # Build specific improvement instructions from feedback
        improvement_focus = []
        if scores.coherence < 8:
            improvement_focus.append("Make internal logic more consistent")
        if scores.influence < 8:
            improvement_focus.append("Increase world impact and power level")
        if scores.conflict_potential < 8:
            improvement_focus.append("Add more story conflict opportunities")
        if scores.distinctiveness < 8:
            improvement_focus.append("Make more unique and memorable")

        prompt = f"""TASK: Improve this faction to score HIGHER on the weak dimensions.

ORIGINAL FACTION:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

CURRENT SCORES (need 9+ in all areas):
- Coherence: {scores.coherence}/10
- Influence: {scores.influence}/10
- Conflict Potential: {scores.conflict_potential}/10
- Distinctiveness: {scores.distinctiveness}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the exact name: "{faction.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements to weak areas
3. Add concrete details, not vague generalities
4. Output in {brief.language if brief else "English"}

Return ONLY the improved faction."""

        try:
            model = self._get_creator_model(entity_type="faction")
            # Use structured generation with Pydantic model for reliable output
            refined = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Faction,
                temperature=temperature,
            )

            # Ensure name is preserved from original faction
            result = refined.model_dump()
            result["name"] = faction.get("name", "Unknown")
            result["type"] = "faction"
            return result
        except Exception as e:
            logger.exception(
                "Faction refinement failed for '%s': %s", faction.get("name") or "Unknown", e
            )
            raise WorldGenerationError(f"Faction refinement failed: {e}") from e

    # ========== ITEM GENERATION WITH QUALITY ==========

    def generate_item_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], ItemQualityScores, int]:
        """
        Generate an item using iterative creation and refinement until it meets quality criteria or retries are exhausted.

        Parameters:
            story_state (StoryState): Current story state containing the brief and identifiers used to guide generation.
            existing_names (list[str]): Existing item names to avoid duplicate naming.

        Returns:
            tuple[dict[str, Any], ItemQualityScores, int]: The chosen item dictionary, its quality scores, and the number of iterations used.

        Raises:
            WorldGenerationError: If item generation fails after all attempts.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for item generation")

        logger.info(f"Generating item with quality threshold {config.quality_threshold}")

        # Track all iterations for best-selection using RefinementHistory
        history = RefinementHistory(entity_type="item", entity_name="")
        iteration = 0
        item: dict[str, Any] = {}
        scores: ItemQualityScores | None = None
        last_error: str = ""

        while iteration < config.max_iterations:
            try:
                # Create new item on first iteration OR if previous returned empty
                # (e.g., duplicate name detection returns {} to force retry)
                if iteration == 0 or not item.get("name"):
                    item = self._create_item(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if item and scores:
                        # Use dynamic temperature that decreases over iterations
                        dynamic_temp = config.get_refinement_temperature(iteration + 1)
                        item = self._refine_item(
                            item,
                            scores,
                            story_state,
                            dynamic_temp,
                        )

                if not item.get("name"):
                    last_error = f"Item creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    iteration += 1
                    continue

                # Update history entity name
                if not history.entity_name:
                    history.entity_name = item.get("name", "Unknown")

                scores = self._judge_item_quality(item, story_state, config.judge_temperature)

                # Track this iteration
                history.add_iteration(
                    iteration=iteration + 1,
                    entity_data=item.copy(),
                    scores=scores.to_dict(),
                    average_score=scores.average,
                    feedback=scores.feedback,
                )

                logger.info(
                    f"Item '{item.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                    f"at iteration {history.best_iteration})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Item '{item.get('name')}' met quality threshold")
                    history.final_iteration = iteration + 1
                    history.final_score = scores.average
                    self._log_refinement_analytics(
                        history,
                        story_state.id,
                        threshold_met=True,
                        early_stop_triggered=False,
                        quality_threshold=config.quality_threshold,
                        max_iterations=config.max_iterations,
                    )
                    return item, scores, iteration + 1

                # Check for early stopping after tracking iteration (enhanced with variance tolerance)
                if history.should_stop_early(
                    config.early_stopping_patience,
                    min_iterations=config.early_stopping_min_iterations,
                    variance_tolerance=config.early_stopping_variance_tolerance,
                ):
                    logger.info(
                        f"Early stopping: Item '{item.get('name')}' quality degraded "
                        f"for {history.consecutive_degradations} consecutive iterations "
                        f"(patience: {config.early_stopping_patience}). "
                        f"Stopping at iteration {iteration + 1}."
                    )
                    break

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Item generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # Didn't meet threshold - return BEST iteration, not last
        if not history.iterations:
            raise WorldGenerationError(
                f"Failed to generate item after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # Pick best iteration (not necessarily the last one)
        best_entity = history.get_best_entity()

        if best_entity and history.best_iteration != len(history.iterations):
            logger.warning(
                f"Item '{history.entity_name}' iterations got WORSE after peak. "
                f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
                f"Final: iteration {len(history.iterations)} "
                f"({history.iterations[-1].average_score:.1f}). "
                f"Returning best iteration."
            )

        # Return best entity or last one
        if best_entity:
            # Reconstruct scores from best iteration
            for record in history.iterations:
                if record.iteration == history.best_iteration:
                    best_scores = ItemQualityScores(**record.scores)
                    break
            if best_scores:
                history.final_iteration = history.best_iteration
                history.final_score = history.peak_score
                was_early_stop = len(history.iterations) < config.max_iterations
                self._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=history.peak_score >= config.quality_threshold,
                    early_stop_triggered=was_early_stop,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return best_entity, best_scores, history.best_iteration

        # Fallback to last iteration
        if item.get("name") and scores:
            logger.warning(
                f"Item '{item.get('name')}' did not meet quality threshold "
                f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
            )
            history.final_iteration = len(history.iterations)
            history.final_score = scores.average
            self._log_refinement_analytics(
                history,
                story_state.id,
                threshold_met=False,
                early_stop_triggered=False,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            return item, scores, len(history.iterations)

        raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
            f"Failed to generate item after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    def _create_item(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new item using the creator model with structured generation."""
        brief = story_state.brief
        if not brief:
            return {}

        # Format existing names with explicit warnings
        existing_names_formatted = self._format_existing_names_warning(existing_names, "item")

        prompt = f"""Create a significant item/object for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

=== CRITICAL: UNIQUENESS REQUIREMENTS ===
{existing_names_formatted}

STRICT RULES:
- DO NOT use any name from the list above
- DO NOT use case variations (e.g., "Sword" vs "SWORD")
- DO NOT use similar names (e.g., "The Blade" vs "Blade of Destiny")
- Create something COMPLETELY DIFFERENT

Create an item with:
1. Significance - meaningful role in the plot or character development
2. Uniqueness - distinctive appearance or properties
3. Narrative potential - opportunities for scenes and conflict
4. Integration - fits naturally into the world

Write all text in {brief.language}."""

        try:
            model = self._get_creator_model(entity_type="item")
            item = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Item,
                temperature=temperature,
            )

            # Comprehensive uniqueness validation (with optional semantic checking)
            if item.name:
                is_unique, conflicting_name, reason = validate_unique_name(
                    item.name,
                    existing_names,
                    check_semantic=self.settings.semantic_duplicate_enabled,
                    semantic_threshold=self.settings.semantic_duplicate_threshold,
                    ollama_url=self.settings.ollama_url,
                    embedding_model=self.settings.embedding_model,
                )
                if not is_unique:
                    logger.warning(
                        f"Item name '{item.name}' conflicts with '{conflicting_name}' "
                        f"(reason: {reason}), clearing to force retry"
                    )
                    return {}  # Return empty to trigger retry

            # Convert to dict for compatibility with existing code
            return item.model_dump()
        except Exception as e:
            logger.exception("Item creation failed for story %s: %s", story_state.id, e)
            raise WorldGenerationError(f"Item creation failed: {e}") from e

    def _judge_item_quality(
        self,
        item: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> ItemQualityScores:
        """Judge item quality using the validator model."""
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating an item for a {genre} story.

ITEM TO EVALUATE:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {self._format_properties(item.get("properties", []))}

Rate each dimension 0-10:
- SIGNIFICANCE: Story importance, plot relevance
- UNIQUENESS: Distinctive qualities
- NARRATIVE_POTENTIAL: Opportunities for scenes
- INTEGRATION: How well it fits the world

Provide specific improvement feedback.

IMPORTANT: Evaluate honestly. Do NOT copy these example values - provide YOUR OWN assessment scores.

Output ONLY valid JSON:
{{"significance": <your_score>, "uniqueness": <your_score>, "narrative_potential": <your_score>, "integration": <your_score>, "feedback": "<your_specific_feedback>"}}"""

        try:
            model = self._get_judge_model(entity_type="item")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_item_judge,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                required_fields = [
                    "significance",
                    "uniqueness",
                    "narrative_potential",
                    "integration",
                ]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Item judge response missing required fields: {missing}"
                    )

                return ItemQualityScores(
                    significance=float(data["significance"]),
                    uniqueness=float(data["uniqueness"]),
                    narrative_potential=float(data["narrative_potential"]),
                    integration=float(data["integration"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from item judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(
                "Item quality judgment failed for '%s': %s", item.get("name") or "Unknown", e
            )
            raise WorldGenerationError(f"Item quality judgment failed: {e}") from e

    def _refine_item(
        self,
        item: dict[str, Any],
        scores: ItemQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine an item based on quality feedback using structured generation."""
        brief = story_state.brief

        # Build specific improvement instructions from feedback
        improvement_focus = []
        if scores.significance < 8:
            improvement_focus.append("Increase story importance and plot relevance")
        if scores.uniqueness < 8:
            improvement_focus.append("Make more distinctive with unique qualities")
        if scores.narrative_potential < 8:
            improvement_focus.append("Add more opportunities for scenes and conflict")
        if scores.integration < 8:
            improvement_focus.append("Improve how naturally it fits into the world")

        prompt = f"""TASK: Improve this item to score HIGHER on the weak dimensions.

ORIGINAL ITEM:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {self._format_properties(item.get("properties", []))}

CURRENT SCORES (need 9+ in all areas):
- Significance: {scores.significance}/10
- Uniqueness: {scores.uniqueness}/10
- Narrative Potential: {scores.narrative_potential}/10
- Integration: {scores.integration}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the exact name: "{item.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements to weak areas
3. Add concrete details, not vague generalities
4. Output in {brief.language if brief else "English"}

Return ONLY the improved item."""

        try:
            model = self._get_creator_model(entity_type="item")
            refined = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Item,
                temperature=temperature,
            )

            # Ensure name is preserved from original item
            result = refined.model_dump()
            result["name"] = item.get("name", "Unknown")
            result["type"] = "item"
            return result
        except Exception as e:
            logger.exception(
                "Item refinement failed for '%s': %s", item.get("name") or "Unknown", e
            )
            raise WorldGenerationError(f"Item refinement failed: {e}") from e

    # ========== CONCEPT GENERATION WITH QUALITY ==========

    def generate_concept_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
    ) -> tuple[dict[str, Any], ConceptQualityScores, int]:
        """Generate a concept with quality refinement loop.

        Args:
            story_state: Current story state with brief.
            existing_names: Names of existing concepts to avoid.

        Returns:
            Tuple of (concept_dict, QualityScores, iterations_used)

        Raises:
            WorldGenerationError: If concept generation fails after all retries.
        """
        config = self.get_config()
        brief = story_state.brief
        if not brief:
            raise ValueError("Story must have a brief for concept generation")

        logger.info(f"Generating concept with quality threshold {config.quality_threshold}")

        # Track all iterations for best-selection using RefinementHistory
        history = RefinementHistory(entity_type="concept", entity_name="")
        iteration = 0
        concept: dict[str, Any] = {}
        scores: ConceptQualityScores | None = None
        last_error: str = ""
        needs_fresh_creation = True  # Track whether we need fresh creation vs refinement

        while iteration < config.max_iterations:
            try:
                if needs_fresh_creation:
                    concept = self._create_concept(
                        story_state, existing_names, config.creator_temperature
                    )
                else:
                    if concept and scores:
                        # Use dynamic temperature that decreases over iterations
                        dynamic_temp = config.get_refinement_temperature(iteration + 1)
                        concept = self._refine_concept(
                            concept,
                            scores,
                            story_state,
                            dynamic_temp,
                        )

                if not concept.get("name"):
                    last_error = f"Concept creation returned empty on iteration {iteration + 1}"
                    logger.error(last_error)
                    needs_fresh_creation = True  # Retry with fresh creation
                    iteration += 1
                    continue

                # Update history entity name
                if not history.entity_name:
                    history.entity_name = concept.get("name", "Unknown")

                scores = self._judge_concept_quality(concept, story_state, config.judge_temperature)
                needs_fresh_creation = False  # Successfully created, now can refine

                # Track this iteration
                history.add_iteration(
                    iteration=iteration + 1,
                    entity_data=concept.copy(),
                    scores=scores.to_dict(),
                    average_score=scores.average,
                    feedback=scores.feedback,
                )

                logger.info(
                    f"Concept '{concept.get('name')}' iteration {iteration + 1}: "
                    f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                    f"at iteration {history.best_iteration})"
                )

                if scores.average >= config.quality_threshold:
                    logger.info(f"Concept '{concept.get('name')}' met quality threshold")
                    history.final_iteration = iteration + 1
                    history.final_score = scores.average
                    self._log_refinement_analytics(
                        history,
                        story_state.id,
                        threshold_met=True,
                        early_stop_triggered=False,
                        quality_threshold=config.quality_threshold,
                        max_iterations=config.max_iterations,
                    )
                    return concept, scores, iteration + 1

                # Check for early stopping after tracking iteration (enhanced with variance tolerance)
                if history.should_stop_early(
                    config.early_stopping_patience,
                    min_iterations=config.early_stopping_min_iterations,
                    variance_tolerance=config.early_stopping_variance_tolerance,
                ):
                    logger.info(
                        f"Early stopping: Concept '{concept.get('name')}' quality degraded "
                        f"for {history.consecutive_degradations} consecutive iterations "
                        f"(patience: {config.early_stopping_patience}). "
                        f"Stopping at iteration {iteration + 1}."
                    )
                    break

            except WorldGenerationError as e:
                last_error = str(e)
                logger.error(f"Concept generation error on iteration {iteration + 1}: {e}")

            iteration += 1

        # Didn't meet threshold - return BEST iteration, not last
        if not history.iterations:
            raise WorldGenerationError(
                f"Failed to generate concept after {config.max_iterations} attempts. "
                f"Last error: {last_error}"
            )

        # Pick best iteration (not necessarily the last one)
        best_entity = history.get_best_entity()

        if best_entity and history.best_iteration != len(history.iterations):
            logger.warning(
                f"Concept '{history.entity_name}' iterations got WORSE after peak. "
                f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
                f"Final: iteration {len(history.iterations)} "
                f"({history.iterations[-1].average_score:.1f}). "
                f"Returning best iteration."
            )

        # Return best entity or last one
        if best_entity:
            # Reconstruct scores from best iteration
            for record in history.iterations:
                if record.iteration == history.best_iteration:
                    best_scores = ConceptQualityScores(**record.scores)
                    break
            if best_scores:
                history.final_iteration = history.best_iteration
                history.final_score = history.peak_score
                was_early_stop = len(history.iterations) < config.max_iterations
                self._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=history.peak_score >= config.quality_threshold,
                    early_stop_triggered=was_early_stop,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return best_entity, best_scores, history.best_iteration

        # Fallback to last iteration
        if concept.get("name") and scores:
            logger.warning(
                f"Concept '{concept.get('name')}' did not meet quality threshold "
                f"({scores.average:.1f} < {config.quality_threshold}), returning anyway"
            )
            history.final_iteration = len(history.iterations)
            history.final_score = scores.average
            self._log_refinement_analytics(
                history,
                story_state.id,
                threshold_met=False,
                early_stop_triggered=False,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            return concept, scores, len(history.iterations)

        raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
            f"Failed to generate concept after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    def _create_concept(
        self,
        story_state: StoryState,
        existing_names: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        """Create a new concept using the creator model with structured generation."""
        brief = story_state.brief
        if not brief:
            return {}

        # Format existing names with explicit warnings
        existing_names_formatted = self._format_existing_names_warning(existing_names, "concept")

        prompt = f"""Create a thematic concept/idea for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

=== CRITICAL: UNIQUENESS REQUIREMENTS ===
{existing_names_formatted}

STRICT RULES:
- DO NOT use any name from the list above
- DO NOT use case variations (e.g., "Hope" vs "HOPE")
- DO NOT use similar names (e.g., "Redemption" vs "The Redemption")
- Create something COMPLETELY DIFFERENT

Create a concept that:
1. Is relevant to the story's themes
2. Has philosophical depth
3. Can manifest in concrete ways in the story
4. Resonates emotionally with readers

Write all text in {brief.language}."""

        try:
            model = self._get_creator_model(entity_type="concept")
            concept = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Concept,
                temperature=temperature,
            )

            # Comprehensive uniqueness validation (with optional semantic checking)
            if concept.name:
                is_unique, conflicting_name, reason = validate_unique_name(
                    concept.name,
                    existing_names,
                    check_semantic=self.settings.semantic_duplicate_enabled,
                    semantic_threshold=self.settings.semantic_duplicate_threshold,
                    ollama_url=self.settings.ollama_url,
                    embedding_model=self.settings.embedding_model,
                )
                if not is_unique:
                    logger.warning(
                        f"Concept name '{concept.name}' conflicts with '{conflicting_name}' "
                        f"(reason: {reason}), clearing to force retry"
                    )
                    return {}  # Return empty to trigger retry

            # Convert to dict for compatibility with existing code
            return concept.model_dump()
        except Exception as e:
            logger.exception("Concept creation failed for story %s: %s", story_state.id, e)
            raise WorldGenerationError(f"Concept creation failed: {e}") from e

    def _judge_concept_quality(
        self,
        concept: dict[str, Any],
        story_state: StoryState,
        temperature: float,
    ) -> ConceptQualityScores:
        """Judge concept quality using the validator model."""
        brief = story_state.brief
        genre = brief.genre if brief else "fiction"

        prompt = f"""You are evaluating a thematic concept for a {genre} story.

CONCEPT TO EVALUATE:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

Rate each dimension 0-10:
- RELEVANCE: Alignment with story themes
- DEPTH: Philosophical richness
- MANIFESTATION: How well it can appear in story
- RESONANCE: Emotional impact potential

Provide specific improvement feedback.

IMPORTANT: Evaluate honestly. Do NOT copy these example values - provide YOUR OWN assessment scores.

Output ONLY valid JSON:
{{"relevance": <your_score>, "depth": <your_score>, "manifestation": <your_score>, "resonance": <your_score>, "feedback": "<your_specific_feedback>"}}"""

        try:
            model = self._get_judge_model(entity_type="concept")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_concept_judge,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                required_fields = ["relevance", "depth", "manifestation", "resonance"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise WorldGenerationError(
                        f"Concept judge response missing required fields: {missing}"
                    )

                return ConceptQualityScores(
                    relevance=float(data["relevance"]),
                    depth=float(data["depth"]),
                    manifestation=float(data["manifestation"]),
                    resonance=float(data["resonance"]),
                    feedback=data.get("feedback", ""),
                )

            raise WorldGenerationError(
                f"Failed to extract JSON from concept judge response: {response['response'][:200]}..."
            )

        except WorldGenerationError:
            raise
        except Exception as e:
            logger.exception(
                "Concept quality judgment failed for '%s': %s",
                concept.get("name") or "Unknown",
                e,
            )
            raise WorldGenerationError(f"Concept quality judgment failed: {e}") from e

    def _refine_concept(
        self,
        concept: dict[str, Any],
        scores: ConceptQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a concept based on quality feedback using structured generation."""
        brief = story_state.brief

        # Build specific improvement instructions from feedback
        improvement_focus = []
        if scores.relevance < 8:
            improvement_focus.append("Strengthen alignment with story themes")
        if scores.depth < 8:
            improvement_focus.append("Add more philosophical richness and complexity")
        if scores.manifestation < 8:
            improvement_focus.append("Provide clearer ways the concept appears in the story")
        if scores.resonance < 8:
            improvement_focus.append("Increase emotional impact potential")

        prompt = f"""TASK: Improve this concept to score HIGHER on the weak dimensions.

ORIGINAL CONCEPT:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

CURRENT SCORES (need 9+ in all areas):
- Relevance: {scores.relevance}/10
- Depth: {scores.depth}/10
- Manifestation: {scores.manifestation}/10
- Resonance: {scores.resonance}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the exact name: "{concept.get("name", "Unknown")}"
2. Make SUBSTANTIAL improvements to weak areas
3. Add concrete details, not vague generalities
4. Output in {brief.language if brief else "English"}

Return ONLY the improved concept."""

        try:
            model = self._get_creator_model(entity_type="concept")
            refined = generate_structured(
                settings=self.settings,
                model=model,
                prompt=prompt,
                response_model=Concept,
                temperature=temperature,
            )

            # Ensure name is preserved from original concept
            result = refined.model_dump()
            result["name"] = concept.get("name", "Unknown")
            result["type"] = "concept"
            return result
        except Exception as e:
            logger.exception(
                "Concept refinement failed for '%s': %s", concept.get("name") or "Unknown", e
            )
            raise WorldGenerationError(f"Concept refinement failed: {e}") from e

    # ========== BATCH OPERATIONS ==========

    def generate_factions_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        existing_locations: list[str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], FactionQualityScores]]:
        """Generate multiple factions with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of factions to generate.
            existing_locations: Names of existing locations for spatial grounding.
            cancel_check: Optional callable that returns True to cancel generation.
            progress_callback: Optional callback to receive progress updates.

        Returns:
            List of (faction_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no factions could be generated.
        """
        results: list[tuple[dict[str, Any], FactionQualityScores]] = []
        names = existing_names.copy()
        errors: list[str] = []
        batch_start_time = time.time()
        completed_times: list[float] = []

        for i in range(count):
            # Check cancellation before starting entity
            if cancel_check and cancel_check():
                logger.info(f"Faction generation cancelled after {len(results)}/{count}")
                break

            # Report progress before generation
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="faction",
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=self._calculate_eta(completed_times, count - i),
                    )
                )

            entity_start = time.time()
            try:
                logger.info(f"Generating faction {i + 1}/{count} with quality refinement")
                faction, scores, iterations = self.generate_faction_with_quality(
                    story_state, names, existing_locations
                )
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                results.append((faction, scores))
                faction_name = faction.get("name", "Unknown")
                names.append(faction_name)
                logger.info(
                    f"Faction '{faction_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
                )

                # Report completion with entity name
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=i + 1,
                            total=count,
                            entity_type="faction",
                            entity_name=faction_name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=self._calculate_eta(
                                completed_times, count - i - 1
                            ),
                        )
                    )
                # Analytics already recorded via _log_refinement_analytics in generate_faction_with_quality
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate faction {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any factions. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} factions. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_items_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 3,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], ItemQualityScores]]:
        """Generate multiple items with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of items to generate.
            cancel_check: Optional callable that returns True to cancel generation.
            progress_callback: Optional callback to receive progress updates.

        Returns:
            List of (item_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no items could be generated.
        """
        results: list[tuple[dict[str, Any], ItemQualityScores]] = []
        names = existing_names.copy()
        errors: list[str] = []
        batch_start_time = time.time()
        completed_times: list[float] = []

        for i in range(count):
            # Check cancellation before starting entity
            if cancel_check and cancel_check():
                logger.info(f"Item generation cancelled after {len(results)}/{count}")
                break

            # Report progress before generation
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="item",
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=self._calculate_eta(completed_times, count - i),
                    )
                )

            entity_start = time.time()
            try:
                logger.info(f"Generating item {i + 1}/{count} with quality refinement")
                item, scores, iterations = self.generate_item_with_quality(story_state, names)
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                results.append((item, scores))
                item_name = item.get("name", "Unknown")
                names.append(item_name)
                logger.info(
                    f"Item '{item_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
                )

                # Report completion with entity name
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=i + 1,
                            total=count,
                            entity_type="item",
                            entity_name=item_name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=self._calculate_eta(
                                completed_times, count - i - 1
                            ),
                        )
                    )
                # Analytics already recorded via _log_refinement_analytics in generate_item_with_quality
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate item {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(f"Failed to generate any items. Errors: {'; '.join(errors)}")

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} items. {len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_concepts_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], ConceptQualityScores]]:
        """Generate multiple concepts with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of concepts to generate.
            cancel_check: Optional callable that returns True to cancel generation.
            progress_callback: Optional callback to receive progress updates.

        Returns:
            List of (concept_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no concepts could be generated.
        """
        results: list[tuple[dict[str, Any], ConceptQualityScores]] = []
        names = existing_names.copy()
        errors: list[str] = []
        batch_start_time = time.time()
        completed_times: list[float] = []

        for i in range(count):
            # Check cancellation before starting entity
            if cancel_check and cancel_check():
                logger.info(f"Concept generation cancelled after {len(results)}/{count}")
                break

            # Report progress before generation
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="concept",
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=self._calculate_eta(completed_times, count - i),
                    )
                )

            entity_start = time.time()
            try:
                logger.info(f"Generating concept {i + 1}/{count} with quality refinement")
                concept, scores, iterations = self.generate_concept_with_quality(story_state, names)
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                results.append((concept, scores))
                concept_name = concept.get("name", "Unknown")
                names.append(concept_name)
                logger.info(
                    f"Concept '{concept_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
                )

                # Report completion with entity name
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=i + 1,
                            total=count,
                            entity_type="concept",
                            entity_name=concept_name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=self._calculate_eta(
                                completed_times, count - i - 1
                            ),
                        )
                    )
                # Analytics already recorded via _log_refinement_analytics in generate_concept_with_quality
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate concept {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any concepts. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} concepts. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_characters_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        custom_instructions: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[Character, CharacterQualityScores]]:
        """Generate multiple characters with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of characters to generate.
            custom_instructions: Optional custom instructions to refine generation.
            cancel_check: Optional callable that returns True to cancel generation.
            progress_callback: Optional callback to receive progress updates.

        Returns:
            List of (Character, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no characters could be generated.
        """
        results: list[tuple[Character, CharacterQualityScores]] = []
        names = existing_names.copy()
        errors: list[str] = []
        batch_start_time = time.time()
        completed_times: list[float] = []

        for i in range(count):
            # Check cancellation before starting entity
            if cancel_check and cancel_check():
                logger.info(f"Character generation cancelled after {len(results)}/{count}")
                break

            # Report progress before generation
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="character",
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=self._calculate_eta(completed_times, count - i),
                    )
                )

            entity_start = time.time()
            try:
                logger.info(f"Generating character {i + 1}/{count} with quality refinement")
                char, scores, iterations = self.generate_character_with_quality(
                    story_state, names, custom_instructions
                )
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                results.append((char, scores))
                names.append(char.name)
                logger.info(
                    f"Character '{char.name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
                )

                # Report completion with entity name
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=i + 1,
                            total=count,
                            entity_type="character",
                            entity_name=char.name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=self._calculate_eta(
                                completed_times, count - i - 1
                            ),
                        )
                    )
                # Analytics already recorded via _log_refinement_analytics in generate_character_with_quality
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate character {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any characters. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} characters. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_locations_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 3,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], LocationQualityScores]]:
        """Generate multiple locations with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of locations to generate.
            cancel_check: Optional callable that returns True to cancel generation.
            progress_callback: Optional callback to receive progress updates.

        Returns:
            List of (location_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no locations could be generated.
        """
        results: list[tuple[dict[str, Any], LocationQualityScores]] = []
        names = existing_names.copy()
        errors: list[str] = []
        batch_start_time = time.time()
        completed_times: list[float] = []

        for i in range(count):
            # Check cancellation before starting entity
            if cancel_check and cancel_check():
                logger.info(f"Location generation cancelled after {len(results)}/{count}")
                break

            # Report progress before generation
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="location",
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=self._calculate_eta(completed_times, count - i),
                    )
                )

            entity_start = time.time()
            try:
                logger.info(f"Generating location {i + 1}/{count} with quality refinement")
                loc, scores, iterations = self.generate_location_with_quality(story_state, names)
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                results.append((loc, scores))
                loc_name = loc.get("name", "Unknown")
                names.append(loc_name)
                logger.info(
                    f"Location '{loc_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
                )

                # Report completion with entity name
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=i + 1,
                            total=count,
                            entity_type="location",
                            entity_name=loc_name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=self._calculate_eta(
                                completed_times, count - i - 1
                            ),
                        )
                    )
                # Analytics already recorded via _log_refinement_analytics in generate_location_with_quality
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate location {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any locations. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} locations. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    def generate_relationships_with_quality(
        self,
        story_state: StoryState,
        entity_names: list[str],
        existing_rels: list[tuple[str, str]],
        count: int = 5,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[EntityGenerationProgress], None] | None = None,
    ) -> list[tuple[dict[str, Any], RelationshipQualityScores]]:
        """Generate multiple relationships with quality refinement.

        Args:
            story_state: Current story state.
            entity_names: Entity names available for relationships.
            existing_rels: Existing relationships to avoid.
            count: Number of relationships to generate.
            cancel_check: Optional callable that returns True to cancel generation.
            progress_callback: Optional callback to receive progress updates.

        Returns:
            List of (relationship_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no relationships could be generated.
        """
        results: list[tuple[dict[str, Any], RelationshipQualityScores]] = []
        rels = existing_rels.copy()
        errors: list[str] = []
        batch_start_time = time.time()
        completed_times: list[float] = []

        for i in range(count):
            # Check cancellation before starting entity
            if cancel_check and cancel_check():
                logger.info(f"Relationship generation cancelled after {len(results)}/{count}")
                break

            # Report progress before generation
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="relationship",
                        phase="generating",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=self._calculate_eta(completed_times, count - i),
                    )
                )

            entity_start = time.time()
            try:
                logger.info(f"Generating relationship {i + 1}/{count} with quality refinement")
                rel, scores, iterations = self.generate_relationship_with_quality(
                    story_state, entity_names, rels
                )
                entity_elapsed = time.time() - entity_start
                completed_times.append(entity_elapsed)
                results.append((rel, scores))
                rels.append((rel.get("source", ""), rel.get("target", "")))
                rel_name = f"{rel.get('source', '')} -> {rel.get('target', '')}"
                logger.info(
                    f"Relationship '{rel_name}' complete "
                    f"after {iterations} iteration(s), quality: {scores.average:.1f}, "
                    f"generation_time: {entity_elapsed:.2f}s"
                )

                # Report completion with entity name
                if progress_callback:
                    progress_callback(
                        EntityGenerationProgress(
                            current=i + 1,
                            total=count,
                            entity_type="relationship",
                            entity_name=rel_name,
                            phase="complete",
                            elapsed_seconds=time.time() - batch_start_time,
                            estimated_remaining_seconds=self._calculate_eta(
                                completed_times, count - i - 1
                            ),
                        )
                    )
                # Analytics already recorded via _log_refinement_analytics in generate_relationship_with_quality
            except WorldGenerationError as e:
                errors.append(str(e))
                logger.error(f"Failed to generate relationship {i + 1}/{count}: {e}")

        if not results and errors:
            raise WorldGenerationError(
                f"Failed to generate any relationships. Errors: {'; '.join(errors)}"
            )

        if errors:
            logger.warning(
                f"Generated {len(results)}/{count} relationships. "
                f"{len(errors)} failed: {'; '.join(errors)}"
            )

        return results

    # ========== MINI DESCRIPTION GENERATION ==========

    def generate_mini_description(
        self,
        name: str,
        entity_type: str,
        full_description: str,
    ) -> str:
        """Generate a short 10-15 word mini description for hover tooltips.

        Uses a fast model with low temperature for consistent, concise output.

        Args:
            name: Entity name.
            entity_type: Type of entity (character, location, etc.).
            full_description: Full description to summarize.

        Returns:
            A short summary (configured word limit).
        """
        max_words = self.settings.mini_description_words_max
        if not full_description or len(full_description.split()) <= max_words:
            # Already short enough, just return trimmed version
            words = full_description.split()[:max_words]
            return " ".join(words)

        prompt = f"""Summarize in EXACTLY 10-{max_words} words for a tooltip preview.

ENTITY: {name} ({entity_type})
FULL DESCRIPTION: {full_description}

Write a punchy, informative summary. NO quotes, NO formatting, just the summary text.

SUMMARY:"""

        try:
            model = self._get_judge_model(entity_type=entity_type)  # Use fast validator model
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.settings.world_quality_judge_temp,
                    "num_predict": self.settings.llm_tokens_mini_description,
                },
            )
            summary: str = str(response["response"]).strip()
            # Clean up any quotes or formatting
            summary = summary.strip("\"'").strip()
            # Ensure it's not too long
            words = summary.split()
            if len(words) > max_words + 3:
                summary = " ".join(words[:max_words]) + "..."
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate mini description: {e}")
            # Fallback: truncate description
            words = full_description.split()[:max_words]
            return " ".join(words) + ("..." if len(full_description.split()) > max_words else "")

    def generate_mini_descriptions_batch(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Generate mini descriptions for a batch of entities.

        Args:
            entities: List of entity dicts with 'name', 'type', 'description'.

        Returns:
            Dict mapping entity names to mini descriptions.
        """
        # Filter entities that have descriptions
        entities_with_desc = [e for e in entities if e.get("description")]
        total_count = len(entities_with_desc)

        logger.info(f"Starting mini description generation for {total_count} entities")
        start_time = time.time()

        results = {}
        for i, entity in enumerate(entities_with_desc):
            name = entity.get("name", "Unknown")
            entity_type = entity.get("type", "entity")
            description = entity.get("description", "")

            logger.debug(
                f"Generating mini description {i + 1}/{total_count}: {entity_type} '{name}'"
            )

            mini_desc = self.generate_mini_description(name, entity_type, description)
            results[name] = mini_desc
            logger.debug(f"Generated mini description for {name}: {mini_desc[:50]}...")

        elapsed = time.time() - start_time
        avg_time = elapsed / max(len(results), 1)
        logger.info(
            f"Completed mini description generation: {len(results)} descriptions "
            f"in {elapsed:.2f}s ({avg_time:.2f}s avg)"
        )

        return results

    async def refine_entity(
        self,
        entity: Entity | None,
        story_brief: StoryBrief | None,
    ) -> dict[str, Any] | None:
        """Refine an existing entity to improve its quality.

        Uses the quality service's refinement loop with the existing entity as a base.

        Args:
            entity: Entity to refine.
            story_brief: Story brief for context.

        Returns:
            Dictionary with refined entity data (name, description, attributes), or None on failure.
        """
        if not entity or not story_brief:
            logger.warning("refine_entity called with missing entity or story_brief")
            return None

        entity_type = entity.type.lower()
        logger.info(f"Refining {entity_type}: {entity.name}")

        try:
            # Get quality scores to identify weak areas
            current_scores = (entity.attributes or {}).get("quality_scores")
            feedback = ""
            if current_scores:
                # Build feedback from low scores
                weak_areas = []
                for key, value in current_scores.items():
                    if key not in ("average", "feedback") and isinstance(value, (int, float)):
                        if value < 7:
                            weak_areas.append(f"{key.replace('_', ' ')} ({value:.1f})")
                if weak_areas:
                    feedback = f"Focus on improving: {', '.join(weak_areas)}"

            # Build refinement prompt
            refinement_prompt = f"""Refine this {entity_type} for a story. Keep the core identity but improve quality.

Current {entity_type}:
- Name: {entity.name}
- Description: {entity.description}
- Attributes: {entity.attributes}
- Role: {(entity.attributes or {}).get("role", "unknown")}

{f"Feedback: {feedback}" if feedback else "Improve overall quality and depth."}

Story context:
- Title: {getattr(story_brief, "title", "Unknown")}
- Genre: {getattr(story_brief, "genre", "Unknown")}
- Themes: {getattr(story_brief, "themes", "Unknown")}

Return a JSON object with:
- name: string (can refine slightly but keep recognizable)
- description: string (improved, more detailed)
- attributes: object (enhanced attributes for this {entity_type} type)"""

            # Use creator model for refinement
            model_id = self._get_creator_model(entity_type)
            config = RefinementConfig.from_settings(self.settings)

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=refinement_prompt,
                options={
                    "temperature": config.refinement_temperature,
                    "num_predict": 2048,
                },
            )

            response_text = str(response.get("response", ""))
            if not response_text:
                logger.warning(f"Empty response when refining {entity.name}")
                return None

            # Parse response
            result = extract_json(response_text)
            if not result or not isinstance(result, dict):
                logger.warning(f"Failed to parse refinement response for {entity.name}")
                return None

            logger.info(f"Successfully refined {entity_type}: {entity.name}")
            return result

        except Exception as e:
            logger.exception(f"Failed to refine entity {entity.name}: {e}")
            return None

    async def regenerate_entity(
        self,
        entity: Entity | None,
        story_brief: StoryBrief | None,
        custom_instructions: str | None = None,
    ) -> dict[str, Any] | None:
        """Fully regenerate an entity with AI.

        Creates a new version of the entity while preserving its role in the story.

        Args:
            entity: Entity to regenerate.
            story_brief: Story brief for context.
            custom_instructions: Optional user guidance for regeneration.

        Returns:
            Dictionary with regenerated entity data (name, description, attributes), or None on failure.
        """
        if not entity or not story_brief:
            logger.warning("regenerate_entity called with missing entity or story_brief")
            return None

        entity_type = entity.type.lower()
        logger.info(f"Regenerating {entity_type}: {entity.name}")

        try:
            # Build regeneration prompt
            instruction_text = custom_instructions or "Create a fresh, high-quality version."

            regeneration_prompt = f"""Create a new version of this {entity_type} for a story.

Original {entity_type} (for reference):
- Name: {entity.name}
- Type: {entity.type}
- Description: {entity.description}
- Role: {(entity.attributes or {}).get("role", "unknown")}

Instructions: {instruction_text}

Story context:
- Title: {getattr(story_brief, "title", "Unknown")}
- Genre: {getattr(story_brief, "genre", "Unknown")}
- Themes: {getattr(story_brief, "themes", "Unknown")}
- Setting: {getattr(story_brief, "setting", "Unknown")}

Return a JSON object with:
- name: string (new name, can be similar or different)
- description: string (detailed, engaging description)
- attributes: object (full attributes for this {entity_type} type including role, traits, etc.)"""

            # Use creator model
            model_id = self._get_creator_model(entity_type)
            config = RefinementConfig.from_settings(self.settings)

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=regeneration_prompt,
                options={
                    "temperature": config.creator_temperature,
                    "num_predict": 2048,
                },
            )

            response_text = str(response.get("response", ""))
            if not response_text:
                logger.warning(f"Empty response when regenerating {entity.name}")
                return None

            # Parse response
            result = extract_json(response_text)
            if not result or not isinstance(result, dict):
                logger.warning(f"Failed to parse regeneration response for {entity.name}")
                return None

            logger.info(
                f"Successfully regenerated {entity_type}: {entity.name} -> {result.get('name', 'unknown')}"
            )
            return result

        except Exception as e:
            logger.exception(f"Failed to regenerate entity {entity.name}: {e}")
            return None

    # ========== RELATIONSHIP SUGGESTIONS ==========

    async def suggest_relationships_for_entity(
        self,
        entity: Entity,
        available_entities: list[Entity],
        existing_relationships: list[dict[str, str]],
        story_brief: StoryBrief | None,
        num_suggestions: int = 3,
    ) -> list[dict[str, Any]]:
        """Suggest meaningful relationships for an entity based on narrative context.

        Uses LLM to propose relationships based on entity descriptions,
        story context, and existing world structure.

        Args:
            entity: The entity to suggest relationships for.
            available_entities: Other entities that could be relationship targets.
            existing_relationships: List of existing relationships (as dicts with
                source_name, target_name, relation_type).
            story_brief: Story brief for context (optional).
            num_suggestions: Number of suggestions to generate.

        Returns:
            List of relationship suggestion dicts with keys:
            - target_entity_name: Name of suggested target
            - target_entity_id: ID of suggested target
            - relation_type: Type of relationship
            - description: Relationship description
            - confidence: Confidence score (0.0-1.0)
            - bidirectional: Whether relationship is bidirectional
        """
        logger.info(f"Suggesting relationships for entity: {entity.name}")

        # Filter out the entity itself from available entities
        targets = [e for e in available_entities if e.id != entity.id]
        if not targets:
            logger.debug("No available target entities for relationship suggestions")
            return []

        # Get relationship minimums from settings - log warning if not configured
        entity_type = entity.type.lower()
        entity_role = (entity.attributes or {}).get("role", "default")
        minimums = self.settings.relationship_minimums.get(entity_type, {})

        if entity_type not in self.settings.relationship_minimums:
            logger.warning(
                f"No relationship_minimums configured for entity type '{entity_type}', "
                f"using fallback. Consider adding to settings.relationship_minimums."
            )

        if entity_role in minimums:
            min_rels = minimums[entity_role]
        elif "default" in minimums:
            min_rels = minimums["default"]
            logger.debug(f"Role '{entity_role}' not in minimums for '{entity_type}', using default")
        else:
            # No configuration found - log warning and use sensible default
            min_rels = 2
            logger.warning(
                f"No relationship minimum configured for {entity_type}/{entity_role}, "
                f"using fallback value of 2"
            )

        # Enforce max_relationships_per_entity by limiting suggestions
        max_rels = self.settings.max_relationships_per_entity
        current_rel_count = len(
            [
                r
                for r in existing_relationships
                if r.get("source_name") == entity.name or r.get("target_name") == entity.name
            ]
        )
        effective_suggestions = min(num_suggestions, max(0, max_rels - current_rel_count))

        if effective_suggestions <= 0:
            logger.debug(f"Entity {entity.name} already at max relationships ({max_rels})")
            return []

        # Build prompt with entity IDs for stable reference
        entities_context = "\n".join(
            f"- [{e.id}] {e.name} ({e.type}): {e.description[:100]}..."
            for e in targets[:20]  # Limit to avoid context overflow
        )

        existing_rels_context = "\n".join(
            f"- {r.get('relation_type', 'unknown')}: {r.get('source_name')} -> {r.get('target_name')}"
            for r in existing_relationships[:15]
        )

        prompt = f"""Suggest meaningful relationships for this entity.

TARGET ENTITY:
Name: {entity.name}
Type: {entity.type}
Description: {entity.description}

AVAILABLE ENTITIES FOR RELATIONSHIPS (use the bracketed ID when specifying targets):
{entities_context}

{"EXISTING RELATIONSHIPS (avoid duplicates):" + chr(10) + existing_rels_context if existing_relationships else ""}

STORY CONTEXT:
- Premise: {story_brief.premise[:100] if story_brief and story_brief.premise else "Unknown"}...
- Genre: {story_brief.genre if story_brief else "Unknown"}
- Themes: {story_brief.themes if story_brief else "Unknown"}

Suggest {effective_suggestions} new relationships for "{entity.name}".
Consider: allies, rivals, family, romance, professional ties, hidden connections.
Minimum recommended relationships for {entity_type}/{entity_role}: {min_rels}
Maximum relationships per entity: {max_rels} (current: {current_rel_count})

IMPORTANT: Respond with ONLY the JSON object below. No markdown code blocks, no explanations, no additional text.

{{
    "suggestions": [
        {{
            "target_entity_id": "uuid-from-brackets",
            "target_entity_name": "Entity Name",
            "relation_type": "...",
            "description": "...",
            "confidence": 0.85,
            "bidirectional": true
        }}
    ]
}}"""

        try:
            model_id = self._get_creator_model(entity_type="relationship")
            config = RefinementConfig.from_settings(self.settings)

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=prompt,
                options={
                    "temperature": config.creator_temperature,
                    "num_predict": 1024,
                },
            )

            response_text = str(response.get("response", ""))
            if not response_text:
                logger.warning(f"Empty response for relationship suggestions for {entity.name}")
                return []

            result = extract_json(response_text)
            if not result or not isinstance(result, dict):
                logger.warning(f"Failed to parse relationship suggestions for {entity.name}")
                return []

            suggestions = result.get("suggestions", [])

            # Enrich suggestions with target entity IDs using ID-first, then fuzzy matching
            enriched_suggestions = []
            fuzzy_threshold = self.settings.fuzzy_match_threshold
            for suggestion in suggestions:
                target_id = suggestion.get("target_entity_id", "")
                target_name = suggestion.get("target_entity_name", "")

                # First try: resolve by ID if provided
                target_entity = None
                if target_id:
                    target_entity = next((e for e in targets if e.id == target_id), None)

                # Second try: exact case-insensitive name match
                if not target_entity and target_name:
                    target_entity = next(
                        (e for e in targets if e.name.lower() == target_name.lower()),
                        None,
                    )

                # Third try: fuzzy name matching
                if not target_entity and target_name:
                    from difflib import SequenceMatcher

                    best_match = None
                    best_score = 0.0
                    search_name = target_name.lower().strip()
                    for e in targets:
                        entity_name = e.name.lower().strip()
                        score = SequenceMatcher(None, search_name, entity_name).ratio()
                        if score > best_score and score >= fuzzy_threshold:
                            best_score = score
                            best_match = e
                    if best_match:
                        logger.debug(
                            f"Fuzzy matched '{target_name}' -> '{best_match.name}' (score={best_score:.2f})"
                        )
                        target_entity = best_match

                if target_entity:
                    # Coerce confidence to float with fallback and clamping
                    raw_confidence = suggestion.get("confidence", 0.5)
                    try:
                        confidence = float(raw_confidence)
                    except (TypeError, ValueError):
                        confidence = 0.5
                    confidence = max(0.0, min(1.0, confidence))

                    # Coerce bidirectional to boolean
                    raw_bidirectional = suggestion.get("bidirectional", False)
                    if isinstance(raw_bidirectional, str):
                        bidirectional = raw_bidirectional.strip().lower() in ("true", "1", "yes")
                    elif isinstance(raw_bidirectional, (int, bool)):
                        bidirectional = bool(raw_bidirectional)
                    else:
                        bidirectional = False

                    enriched_suggestions.append(
                        {
                            "source_entity_id": entity.id,
                            "source_entity_name": entity.name,
                            "target_entity_id": target_entity.id,
                            "target_entity_name": target_entity.name,
                            "relation_type": suggestion.get("relation_type", "related_to"),
                            "description": suggestion.get("description", ""),
                            "confidence": confidence,
                            "bidirectional": bidirectional,
                        }
                    )
                else:
                    logger.warning(
                        f"Could not resolve target entity: id={target_id}, name={target_name}"
                    )

            logger.info(
                f"Generated {len(enriched_suggestions)} relationship suggestions for {entity.name}"
            )
            return enriched_suggestions

        except Exception as e:
            logger.exception(f"Failed to suggest relationships for {entity.name}: {e}")
            return []

    # ========== CONTRADICTION DETECTION ==========

    async def extract_entity_claims(
        self,
        entity: Entity,
    ) -> list[dict[str, str]]:
        """Extract factual claims from an entity's description and attributes.

        Args:
            entity: Entity to extract claims from.

        Returns:
            List of claim dicts with keys: entity_id, entity_name, entity_type, claim, source_text.
        """
        logger.debug(f"Extracting claims from entity: {entity.name}")

        prompt = f"""Extract key factual claims from this entity.

ENTITY:
Name: {entity.name}
Type: {entity.type}
Description: {entity.description}
Attributes: {entity.attributes}

Extract 3-5 factual claims about this entity (things that must be true about them).
Focus on: identity, relationships, abilities, history, location, appearance.

Return JSON:
{{
    "claims": [
        "Claim 1 (e.g., 'Character X is the king of Y')",
        "Claim 2 (e.g., 'Character X has blue eyes')",
        "Claim 3"
    ]
}}"""

        try:
            model_id = self._get_judge_model(entity_type=entity.type.lower())

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            )

            response_text = str(response.get("response", ""))
            result = extract_json(response_text)

            if not result or not isinstance(result, dict):
                return []

            claims_list = result.get("claims", [])
            return [
                {
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "entity_type": entity.type,
                    "claim": claim,
                    "source_text": entity.description[:200],
                }
                for claim in claims_list
                if isinstance(claim, str)
            ]

        except Exception as e:
            logger.exception(f"Failed to extract claims from {entity.name}: {e}")
            return []

    async def check_contradiction(
        self,
        claim_a: dict[str, str],
        claim_b: dict[str, str],
    ) -> dict[str, Any] | None:
        """Check if two claims contradict each other.

        Args:
            claim_a: First claim dict.
            claim_b: Second claim dict.

        Returns:
            Contradiction dict if found, None otherwise.
        """
        prompt = f"""Analyze whether these two claims contradict each other.

CLAIM A:
Entity: {claim_a.get("entity_name")} ({claim_a.get("entity_type")})
Claim: {claim_a.get("claim")}

CLAIM B:
Entity: {claim_b.get("entity_name")} ({claim_b.get("entity_type")})
Claim: {claim_b.get("claim")}

Analyze for contradictions considering:
1. Direct logical contradictions
2. Temporal inconsistencies
3. Character/attribute conflicts
4. Location/geography conflicts
5. Relationship contradictions

Return JSON:
{{
    "is_contradiction": true/false,
    "severity": "low" | "medium" | "high" | "critical",
    "explanation": "Why these claims do or don't contradict",
    "resolution_suggestion": "How to resolve if contradictory",
    "confidence": 0.0-1.0
}}"""

        try:
            model_id = self._get_judge_model(entity_type="validator")

            response = await asyncio.to_thread(
                self.client.generate,
                model=model_id,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            )

            response_text = str(response.get("response", ""))
            result = extract_json(response_text)

            if not result or not isinstance(result, dict):
                return None

            # Properly coerce is_contradiction to boolean (string "false" is truthy!)
            raw_is_contradiction = result.get("is_contradiction", False)
            if isinstance(raw_is_contradiction, str):
                is_contradiction = raw_is_contradiction.strip().lower() in ("true", "1", "yes")
            elif isinstance(raw_is_contradiction, (int, bool)):
                is_contradiction = bool(raw_is_contradiction)
            else:
                is_contradiction = False

            if is_contradiction:
                # Coerce confidence to float with fallback and clamping
                try:
                    confidence = float(result.get("confidence", 0.5))
                except (TypeError, ValueError):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))

                return {
                    "claim_a": claim_a,
                    "claim_b": claim_b,
                    "severity": result.get("severity", "medium"),
                    "explanation": result.get("explanation", ""),
                    "resolution_suggestion": result.get("resolution_suggestion", ""),
                    "confidence": confidence,
                }

            return None

        except Exception as e:
            logger.exception("Failed to check contradiction: %s", e)
            return None

    async def validate_entity_consistency(
        self,
        entities: list[Entity],
        max_comparisons: int = 50,
    ) -> list[dict[str, Any]]:
        """Validate consistency across entities by detecting contradictions.

        Extracts claims from each entity and cross-references them pairwise
        to detect contradictions.

        Args:
            entities: List of entities to validate.
            max_comparisons: Maximum number of claim pairs to compare (for performance).

        Returns:
            List of detected contradictions.
        """
        logger.info(f"Validating consistency across {len(entities)} entities")

        # Extract claims from all entities
        all_claims: list[dict[str, str]] = []
        for entity in entities:
            claims = await self.extract_entity_claims(entity)
            all_claims.extend(claims)

        logger.debug(f"Extracted {len(all_claims)} claims from {len(entities)} entities")

        if len(all_claims) < 2:
            return []

        # Compare claims pairwise (with limit for performance)
        contradictions: list[dict[str, Any]] = []
        comparisons_made = 0

        for i, claim_a in enumerate(all_claims):
            for claim_b in all_claims[i + 1 :]:
                if comparisons_made >= max_comparisons:
                    break

                # Skip comparing claims from the same entity
                if claim_a.get("entity_id") == claim_b.get("entity_id"):
                    continue

                contradiction = await self.check_contradiction(claim_a, claim_b)
                if contradiction:
                    contradictions.append(contradiction)

                comparisons_made += 1

            if comparisons_made >= max_comparisons:
                break

        logger.info(f"Found {len(contradictions)} contradictions in {comparisons_made} comparisons")
        return contradictions
