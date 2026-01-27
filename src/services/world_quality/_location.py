"""Location generation mixin for WorldQualityService."""

import logging
from typing import Any

import src.services.llm_client as _llm_client
from src.memory.story_state import Location, StoryState
from src.memory.world_quality import LocationQualityScores, RefinementHistory
from src.utils.exceptions import WorldGenerationError
from src.utils.validation import validate_unique_name

from ._base import WorldQualityServiceBase

logger = logging.getLogger(__name__)


class LocationMixin(WorldQualityServiceBase):
    """Mixin providing location generation with quality refinement."""

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
        best_scores: LocationQualityScores | None = None

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
            location = _llm_client.generate_structured(
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
            return _llm_client.generate_structured(
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
            refined = _llm_client.generate_structured(
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
