"""Item generation mixin for WorldQualityService."""

import logging
from typing import Any

import src.services.llm_client as _llm_client
from src.memory.story_state import Item, StoryState
from src.memory.world_quality import ItemQualityScores, RefinementHistory
from src.utils.exceptions import WorldGenerationError
from src.utils.json_parser import extract_json
from src.utils.validation import validate_unique_name

from ._base import WorldQualityServiceBase

logger = logging.getLogger(__name__)


class ItemMixin(WorldQualityServiceBase):
    """Mixin providing item generation with quality refinement."""

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
        best_scores: ItemQualityScores | None = None

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
            item = _llm_client.generate_structured(
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
            refined = _llm_client.generate_structured(
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
