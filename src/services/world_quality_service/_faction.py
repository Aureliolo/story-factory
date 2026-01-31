"""Faction generation, judgment, and refinement functions."""

import logging
import random
from typing import Any

from src.memory.story_state import Faction, StoryState
from src.memory.world_quality import FactionQualityScores, RefinementHistory
from src.services.llm_client import generate_structured
from src.utils.exceptions import WorldGenerationError
from src.utils.validation import validate_unique_name

logger = logging.getLogger(__name__)


# Diversity hints for faction naming to avoid generic names
FACTION_NAMING_HINTS: list[str] = [
    "Use an evocative, specific name - avoid generic names like 'The Guild' or 'The Order'",
    "Draw from historical organizations for inspiration (e.g., Hanseatic League, Templars)",
    "Name could reflect their methods (e.g., 'The Whispered Accord', 'Iron Covenant')",
    "Name could be ironic or misleading (e.g., a violent group called 'The Peacemakers')",
    "Name could reference their origin or founding myth",
    "Name could be in a constructed language or archaic form",
    "Name should be memorable and distinct from common fantasy tropes",
]

FACTION_STRUCTURE_HINTS: list[str] = [
    "Consider a non-hierarchical structure (cells, councils, rotating leadership)",
    "Structure could mirror their beliefs (egalitarian values = flat structure)",
    "Consider secret inner circles or public-facing vs. true leadership",
    "Structure could be based on expertise, seniority, or divine mandate",
    "Consider how new members join and advance within the organization",
]

FACTION_IDEOLOGY_HINTS: list[str] = [
    "Core beliefs should create internal tensions or contradictions",
    "Ideology could be a corruption or evolution of older beliefs",
    "Consider what the faction fears or opposes, not just what they support",
    "Ideology should naturally conflict with at least one other group",
    "Consider the gap between stated ideals and actual practices",
]


def generate_faction_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    existing_locations: list[str] | None = None,
) -> tuple[dict[str, Any], FactionQualityScores, int]:
    """
    Generate a faction using an iterative create-refine-judge loop and return the best iteration.

    Parameters:
        svc: WorldQualityService instance.
        story_state (StoryState): Current story state containing the brief and context.
        existing_names (list[str]): Existing faction names to avoid duplicates.
        existing_locations (list[str] | None): Optional list of location names for spatial grounding.

    Returns:
        tuple[dict[str, Any], FactionQualityScores, int]: (faction_dict, quality_scores, iterations_used)

    Raises:
        WorldGenerationError: If faction generation fails to produce any valid iterations.
    """
    config = svc.get_config()
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
    creation_retries = 0  # Track duplicate-name retries for temperature escalation

    while iteration < config.max_iterations:
        try:
            # Create new faction on first iteration OR if previous returned empty
            # (e.g., duplicate name detection returns {} to force retry)
            if iteration == 0 or not faction.get("name"):
                # Increase temperature on retries to avoid regenerating the same name
                retry_temp = min(config.creator_temperature + (creation_retries * 0.15), 1.5)
                faction = svc._create_faction(
                    story_state, existing_names, retry_temp, existing_locations
                )
            else:
                if faction and scores:
                    # Use dynamic temperature that decreases over iterations
                    dynamic_temp = config.get_refinement_temperature(iteration + 1)
                    faction = svc._refine_faction(
                        faction,
                        scores,
                        story_state,
                        dynamic_temp,
                    )

            if not faction.get("name"):
                creation_retries += 1
                last_error = f"Faction creation returned empty on iteration {iteration + 1}"
                logger.warning(
                    "%s (retry %d, next temp=%.2f)",
                    last_error,
                    creation_retries,
                    min(config.creator_temperature + (creation_retries * 0.15), 1.5),
                )
                iteration += 1
                continue

            # Update history entity name
            if not history.entity_name:
                history.entity_name = faction.get("name", "Unknown")

            scores = svc._judge_faction_quality(faction, story_state, config.judge_temperature)

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
                svc._log_refinement_analytics(
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
    svc._log_refinement_analytics(
        history,
        story_state.id,
        threshold_met=history.final_score >= config.quality_threshold,
        early_stop_triggered=was_early_stop,
        quality_threshold=config.quality_threshold,
        max_iterations=config.max_iterations,
    )

    return faction, scores, history.final_iteration


def _create_faction(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
    existing_locations: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate a unique faction definition for the given story using the configured creator model.

    Parameters:
        svc: WorldQualityService instance.
        story_state (StoryState): Story context and brief.
        existing_names (list[str]): Existing faction names to avoid.
        temperature (float): Sampling temperature for the creator model.
        existing_locations (list[str] | None): Optional list of world locations.

    Returns:
        dict[str, Any]: A faction dictionary. Returns empty dict when retry needed.

    Raises:
        WorldGenerationError: If faction generation fails due to unrecoverable errors.
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
    naming_hint = random.choice(FACTION_NAMING_HINTS)
    structure_hint = random.choice(FACTION_STRUCTURE_HINTS)
    ideology_hint = random.choice(FACTION_IDEOLOGY_HINTS)

    # Format existing names with clear guidance
    existing_names_formatted = _format_existing_names(existing_names)

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
        model = svc._get_creator_model(entity_type="faction")
        # Use structured generation with Pydantic model for reliable output
        faction = generate_structured(
            settings=svc.settings,
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
                check_semantic=svc.settings.semantic_duplicate_enabled,
                semantic_threshold=svc.settings.semantic_duplicate_threshold,
                ollama_url=svc.settings.ollama_url,
                embedding_model=svc.settings.embedding_model,
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
    svc,
    faction: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> FactionQualityScores:
    """Judge faction quality using the validator model.

    Args:
        svc: WorldQualityService instance.
        faction: Faction dict to evaluate.
        story_state: Current story state for context.
        temperature: Judge temperature (low for consistency).

    Returns:
        FactionQualityScores with ratings and feedback.

    Raises:
        WorldGenerationError: If quality judgment fails or returns invalid data.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    prompt = f"""You are a strict quality judge evaluating a faction for a {genre} story.

FACTION TO EVALUATE:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

Rate each dimension 0-10:
- coherence: Internal consistency, clear structure
- influence: World impact, power level
- conflict_potential: Story conflict opportunities
- distinctiveness: Memorable, unique qualities

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"coherence": <number>, "influence": <number>, "conflict_potential": <number>, "distinctiveness": <number>, "feedback": "<string>"}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    try:
        model = svc._get_judge_model(entity_type="faction")
        return generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=FactionQualityScores,
            temperature=temperature,
        )
    except Exception as e:
        logger.exception(
            "Faction quality judgment failed for '%s': %s",
            faction.get("name") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"Faction quality judgment failed: {e}") from e


def _refine_faction(
    svc,
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
        model = svc._get_creator_model(entity_type="faction")
        # Use structured generation with Pydantic model for reliable output
        refined = generate_structured(
            settings=svc.settings,
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


def _format_existing_names(existing_names: list[str]) -> str:
    """
    Format a list of existing faction names into a prompt-ready string.

    Parameters:
        existing_names (list[str]): Existing faction names to include in the prompt.

    Returns:
        str: Newline-separated names each prefixed with "-".
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
