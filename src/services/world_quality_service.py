"""World Quality Service - multi-model iteration for world building quality.

Implements a generate-judge-refine loop using:
- Creator model: High temperature (0.9) for creative generation
- Judge model: Low temperature (0.1) for consistent evaluation
- Refinement: Incorporates feedback to improve entities
"""

import logging
import random
import time
from typing import Any, ClassVar

import ollama

from src.memory.mode_database import ModeDatabase
from src.memory.story_state import Character, Faction, StoryState
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
from src.utils.validation import validate_not_empty

logger = logging.getLogger(__name__)


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
                        character = self._refine_character(
                            character,
                            scores,
                            story_state,
                            config.refinement_temperature,
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

                # Check for early stopping after tracking iteration
                if history.should_stop_early(config.early_stopping_patience):
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
                # Check if this was an early stop (we didn't reach max iterations)
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
            logger.error(f"Character creation error: {e}")
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
            logger.exception(f"Character quality judgment failed: {e}")
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
            logger.exception(f"Character refinement failed: {e}")
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
                        location = self._refine_location(
                            location,
                            scores,
                            story_state,
                            config.refinement_temperature,
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

                # Check for early stopping after tracking iteration
                if history.should_stop_early(config.early_stopping_patience):
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
        """Create a new location using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        prompt = f"""Create a compelling location for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}

EXISTING LOCATIONS IN THIS WORLD: {", ".join(existing_names) if existing_names else "None yet"}
(Create a NEW location with a different name that complements these existing ones)

Create a location with:
1. Rich atmosphere - sensory details, mood
2. Narrative significance - symbolic or plot meaning
3. Strong story relevance - connections to themes/characters
4. Distinctiveness - memorable unique qualities

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Location Name",
    "type": "location",
    "description": "Detailed description with sensory details (2-3 sentences)",
    "significance": "Why this place matters to the story"
}}"""

        try:
            model = self._get_creator_model(entity_type="location")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_location_create,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                # Validate name is not a duplicate
                name = data.get("name", "")
                if name and name in existing_names:
                    logger.warning(f"Location name '{name}' is duplicate, clearing to force retry")
                    return {}  # Return empty to trigger retry
                return data
            else:
                logger.error(f"Location creation returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid location JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Location creation LLM error: {e}")
            raise WorldGenerationError(f"LLM error during location creation: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Location creation JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid location response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in location creation: {e}")
            raise WorldGenerationError(f"Unexpected location creation error: {e}") from e

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
            logger.exception(f"Location quality judgment failed: {e}")
            raise WorldGenerationError(f"Location quality judgment failed: {e}") from e

    def _refine_location(
        self,
        location: dict[str, Any],
        scores: LocationQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a location based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this location based on quality feedback.

ORIGINAL LOCATION:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}

QUALITY SCORES (0-10):
- Atmosphere: {scores.atmosphere}
- Significance: {scores.significance}
- Story Relevance: {scores.story_relevance}
- Distinctiveness: {scores.distinctiveness}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{location.get("name", "Unknown")}",
    "type": "location",
    "description": "Improved description",
    "significance": "Improved significance"
}}"""

        try:
            model = self._get_creator_model(entity_type="location")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_location_refine,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                return data
            else:
                logger.error(f"Location refinement returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid location refinement JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Location refinement LLM error: {e}")
            raise WorldGenerationError(f"LLM error during location refinement: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Location refinement JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid location refinement response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in location refinement: {e}")
            raise WorldGenerationError(f"Unexpected location refinement error: {e}") from e

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
                        relationship = self._refine_relationship(
                            relationship,
                            scores,
                            story_state,
                            config.refinement_temperature,
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

                # Check for early stopping after tracking iteration
                if history.should_stop_early(config.early_stopping_patience):
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
            logger.error(f"Relationship creation LLM error: {e}")
            raise WorldGenerationError(f"LLM error during relationship creation: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Relationship creation JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid relationship response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in relationship creation: {e}")
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
            logger.exception(f"Relationship quality judgment failed: {e}")
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
            logger.error(f"Relationship refinement LLM error: {e}")
            raise WorldGenerationError(f"LLM error during relationship refinement: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Relationship refinement JSON parsing error: {e}")
            raise WorldGenerationError(
                f"Invalid relationship refinement response format: {e}"
            ) from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in relationship refinement: {e}")
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
                        faction = self._refine_faction(
                            faction,
                            scores,
                            story_state,
                            config.refinement_temperature,
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

                # Check for early stopping after tracking iteration
                if history.should_stop_early(config.early_stopping_patience):
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
        from src.utils.validation import validate_unique_name

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

            # Comprehensive uniqueness validation
            if faction.name:
                is_unique, conflicting_name, reason = validate_unique_name(
                    faction.name, existing_names
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
            logger.exception(f"Faction creation failed: {e}")
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
            logger.exception(f"Faction quality judgment failed: {e}")
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
            logger.exception(f"Faction refinement failed: {e}")
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
                        item = self._refine_item(
                            item,
                            scores,
                            story_state,
                            config.refinement_temperature,
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

                # Check for early stopping after tracking iteration
                if history.should_stop_early(config.early_stopping_patience):
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
        """Create a new item using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        prompt = f"""Create a significant item/object for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

EXISTING ITEMS IN THIS WORLD: {", ".join(existing_names) if existing_names else "None yet"}
(Create a NEW item with a different name that complements these existing ones)

Create an item with:
1. Significance - meaningful role in the plot or character development
2. Uniqueness - distinctive appearance or properties
3. Narrative potential - opportunities for scenes and conflict
4. Integration - fits naturally into the world

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Item Name",
    "type": "item",
    "description": "Physical description and history (2-3 sentences)",
    "significance": "Why this item matters to the story",
    "properties": ["special property 1", "special property 2"]
}}"""

        try:
            model = self._get_creator_model(entity_type="item")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_item_create,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                # Validate name is not a duplicate
                name = data.get("name", "")
                if name and name in existing_names:
                    logger.warning(f"Item name '{name}' is duplicate, clearing to force retry")
                    return {}  # Return empty to trigger retry
                return data
            else:
                logger.error(f"Item creation returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid item JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Item creation LLM error: {e}")
            raise WorldGenerationError(f"LLM error during item creation: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Item creation JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid item response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in item creation: {e}")
            raise WorldGenerationError(f"Unexpected item creation error: {e}") from e

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
            logger.exception(f"Item quality judgment failed: {e}")
            raise WorldGenerationError(f"Item quality judgment failed: {e}") from e

    def _refine_item(
        self,
        item: dict[str, Any],
        scores: ItemQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine an item based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this item based on quality feedback.

ORIGINAL ITEM:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {self._format_properties(item.get("properties", []))}

QUALITY SCORES (0-10):
- Significance: {scores.significance}
- Uniqueness: {scores.uniqueness}
- Narrative Potential: {scores.narrative_potential}
- Integration: {scores.integration}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{item.get("name", "Unknown")}",
    "type": "item",
    "description": "Improved description",
    "significance": "Improved significance",
    "properties": ["improved properties"]
}}"""

        try:
            model = self._get_creator_model(entity_type="item")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_item_refine,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                return data
            else:
                logger.error(f"Item refinement returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid item refinement JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Item refinement LLM error: {e}")
            raise WorldGenerationError(f"LLM error during item refinement: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Item refinement JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid item refinement response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in item refinement: {e}")
            raise WorldGenerationError(f"Unexpected item refinement error: {e}") from e

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
                        concept = self._refine_concept(
                            concept,
                            scores,
                            story_state,
                            config.refinement_temperature,
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

                # Check for early stopping after tracking iteration
                if history.should_stop_early(config.early_stopping_patience):
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
        """Create a new concept using the creator model."""
        brief = story_state.brief
        if not brief:
            return {}

        existing_str = ", ".join(existing_names) if existing_names else "None yet"
        prompt = f"""Create a thematic concept/idea for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}

EXISTING CONCEPTS IN THIS STORY: {existing_str}
(Create a NEW concept with a different name that complements these existing ones)

Create a concept that:
1. Is relevant to the story's themes
2. Has philosophical depth
3. Can manifest in concrete ways in the story
4. Resonates emotionally with readers

Output ONLY valid JSON (all text in {brief.language}):
{{
    "name": "Concept Name",
    "type": "concept",
    "description": "What this concept means in the context of the story (2-3 sentences)",
    "manifestations": "How this concept appears in the story - through events, characters, or symbols"
}}"""

        try:
            model = self._get_creator_model(entity_type="concept")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_concept_create,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                # Validate name is not a duplicate
                name = data.get("name", "")
                if name and name in existing_names:
                    logger.warning(f"Concept name '{name}' is duplicate, clearing to force retry")
                    return {}  # Return empty to trigger retry
                return data
            else:
                logger.error(f"Concept creation returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid concept JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Concept creation LLM error: {e}")
            raise WorldGenerationError(f"LLM error during concept creation: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Concept creation JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid concept response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in concept creation: {e}")
            raise WorldGenerationError(f"Unexpected concept creation error: {e}") from e

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
            logger.exception(f"Concept quality judgment failed: {e}")
            raise WorldGenerationError(f"Concept quality judgment failed: {e}") from e

    def _refine_concept(
        self,
        concept: dict[str, Any],
        scores: ConceptQualityScores,
        story_state: StoryState,
        temperature: float,
    ) -> dict[str, Any]:
        """Refine a concept based on quality feedback."""
        brief = story_state.brief
        weak = scores.weak_dimensions(self.get_config().quality_threshold)

        prompt = f"""Improve this concept based on quality feedback.

ORIGINAL CONCEPT:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

QUALITY SCORES (0-10):
- Relevance: {scores.relevance}
- Depth: {scores.depth}
- Manifestation: {scores.manifestation}
- Resonance: {scores.resonance}

FEEDBACK: {scores.feedback}
WEAK AREAS: {", ".join(weak) if weak else "None"}

Keep the name, enhance the weak areas.

Output ONLY valid JSON (all text in {brief.language if brief else "English"}):
{{
    "name": "{concept.get("name", "Unknown")}",
    "type": "concept",
    "description": "Improved description",
    "manifestations": "Improved manifestations"
}}"""

        try:
            model = self._get_creator_model(entity_type="concept")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.settings.llm_tokens_concept_refine,
                },
            )

            data = extract_json(response["response"], strict=False)
            if data and isinstance(data, dict):
                return data
            else:
                logger.error(f"Concept refinement returned invalid JSON structure: {data}")
                raise WorldGenerationError(f"Invalid concept refinement JSON structure: {data}")
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Concept refinement LLM error: {e}")
            raise WorldGenerationError(f"LLM error during concept refinement: {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Concept refinement JSON parsing error: {e}")
            raise WorldGenerationError(f"Invalid concept refinement response format: {e}") from e
        except WorldGenerationError:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in concept refinement: {e}")
            raise WorldGenerationError(f"Unexpected concept refinement error: {e}") from e

    # ========== BATCH OPERATIONS ==========

    def generate_factions_with_quality(
        self,
        story_state: StoryState,
        existing_names: list[str],
        count: int = 2,
        existing_locations: list[str] | None = None,
    ) -> list[tuple[dict[str, Any], FactionQualityScores]]:
        """Generate multiple factions with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of factions to generate.
            existing_locations: Names of existing locations for spatial grounding.

        Returns:
            List of (faction_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no factions could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating faction {i + 1}/{count} with quality refinement")
                start_time = time.time()
                faction, scores, iterations = self.generate_faction_with_quality(
                    story_state, names, existing_locations
                )
                elapsed = time.time() - start_time
                results.append((faction, scores))
                faction_name = faction.get("name", "Unknown")
                names.append(faction_name)
                logger.info(
                    f"Faction '{faction_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {elapsed:.2f}s"
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
    ) -> list[tuple[dict[str, Any], ItemQualityScores]]:
        """Generate multiple items with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of items to generate.

        Returns:
            List of (item_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no items could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating item {i + 1}/{count} with quality refinement")
                start_time = time.time()
                item, scores, iterations = self.generate_item_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((item, scores))
                item_name = item.get("name", "Unknown")
                names.append(item_name)
                logger.info(
                    f"Item '{item_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {elapsed:.2f}s"
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
    ) -> list[tuple[dict[str, Any], ConceptQualityScores]]:
        """Generate multiple concepts with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of concepts to generate.

        Returns:
            List of (concept_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no concepts could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating concept {i + 1}/{count} with quality refinement")
                start_time = time.time()
                concept, scores, iterations = self.generate_concept_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((concept, scores))
                concept_name = concept.get("name", "Unknown")
                names.append(concept_name)
                logger.info(
                    f"Concept '{concept_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {elapsed:.2f}s"
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
    ) -> list[tuple[Character, CharacterQualityScores]]:
        """Generate multiple characters with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of characters to generate.
            custom_instructions: Optional custom instructions to refine generation.

        Returns:
            List of (Character, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no characters could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating character {i + 1}/{count} with quality refinement")
                start_time = time.time()
                char, scores, iterations = self.generate_character_with_quality(
                    story_state, names, custom_instructions
                )
                elapsed = time.time() - start_time
                results.append((char, scores))
                names.append(char.name)
                logger.info(
                    f"Character '{char.name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {elapsed:.2f}s"
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
    ) -> list[tuple[dict[str, Any], LocationQualityScores]]:
        """Generate multiple locations with quality refinement.

        Args:
            story_state: Current story state.
            existing_names: Names to avoid.
            count: Number of locations to generate.

        Returns:
            List of (location_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no locations could be generated.
        """
        results = []
        names = existing_names.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating location {i + 1}/{count} with quality refinement")
                start_time = time.time()
                loc, scores, iterations = self.generate_location_with_quality(story_state, names)
                elapsed = time.time() - start_time
                results.append((loc, scores))
                loc_name = loc.get("name", "Unknown")
                names.append(loc_name)
                logger.info(
                    f"Location '{loc_name}' complete after {iterations} iteration(s), "
                    f"quality: {scores.average:.1f}, generation_time: {elapsed:.2f}s"
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
    ) -> list[tuple[dict[str, Any], RelationshipQualityScores]]:
        """Generate multiple relationships with quality refinement.

        Args:
            story_state: Current story state.
            entity_names: Entity names available for relationships.
            existing_rels: Existing relationships to avoid.
            count: Number of relationships to generate.

        Returns:
            List of (relationship_dict, QualityScores) tuples.

        Raises:
            WorldGenerationError: If no relationships could be generated.
        """
        results = []
        rels = existing_rels.copy()
        errors = []

        for i in range(count):
            try:
                logger.info(f"Generating relationship {i + 1}/{count} with quality refinement")
                start_time = time.time()
                rel, scores, iterations = self.generate_relationship_with_quality(
                    story_state, entity_names, rels
                )
                elapsed = time.time() - start_time
                results.append((rel, scores))
                rels.append((rel.get("source", ""), rel.get("target", "")))
                rel_name = f"{rel.get('source', '')} -> {rel.get('target', '')}"
                logger.info(
                    f"Relationship '{rel_name}' complete "
                    f"after {iterations} iteration(s), quality: {scores.average:.1f}, "
                    f"generation_time: {elapsed:.2f}s"
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
        results = {}
        for entity in entities:
            name = entity.get("name", "Unknown")
            entity_type = entity.get("type", "entity")
            description = entity.get("description", "")

            if description:
                mini_desc = self.generate_mini_description(name, entity_type, description)
                results[name] = mini_desc
                logger.debug(f"Generated mini description for {name}: {mini_desc}")

        return results
