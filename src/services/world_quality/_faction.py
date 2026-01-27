"""Faction generation mixin for WorldQualityService."""

import logging
import random
from typing import Any, ClassVar

import src.services.llm_client as _llm_client
from src.memory.story_state import Faction, StoryState
from src.memory.world_quality import FactionQualityScores, RefinementHistory
from src.utils.exceptions import WorldGenerationError
from src.utils.json_parser import extract_json
from src.utils.validation import validate_unique_name

from ._base import WorldQualityServiceBase

logger = logging.getLogger(__name__)


class FactionMixin(WorldQualityServiceBase):
    """Mixin providing faction generation with quality refinement."""

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
            faction = _llm_client.generate_structured(
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
            refined = _llm_client.generate_structured(
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
