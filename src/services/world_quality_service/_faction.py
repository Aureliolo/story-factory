"""Faction generation, judgment, and refinement functions."""

import logging
import random
from typing import Any

from src.memory.story_state import Faction, StoryState
from src.memory.world_quality import FactionQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
    retry_temperature,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
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
    """Generate a faction with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    factions until the quality threshold is met or stopping criteria is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_names: Existing faction names to avoid duplicates.
        existing_locations: Optional list of location names for spatial grounding.

    Returns:
        Tuple of (faction_dict, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If faction generation fails after all attempts.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for faction generation")

    return quality_refinement_loop(
        entity_type="faction",
        create_fn=lambda retries: svc._create_faction(
            story_state,
            existing_names,
            retry_temperature(config, retries),
            existing_locations,
        ),
        judge_fn=lambda fac: svc._judge_faction_quality(
            fac,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda fac, scores, iteration: svc._refine_faction(
            fac,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda fac: fac.get("name", "Unknown"),
        serialize=lambda fac: fac.copy(),
        is_empty=lambda fac: not fac.get("name"),
        score_cls=FactionQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
    )


def _create_faction(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
    existing_locations: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a unique faction definition for the given story using the configured creator model.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Story context and brief.
        existing_names: Existing faction names to avoid.
        temperature: Sampling temperature for the creator model.
        existing_locations: Optional list of world locations.

    Returns:
        A faction dictionary. Returns empty dict when retry needed.

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
    """Judge faction quality using the judge model.

    Supports multi-call averaging when judge_multi_call_enabled is True in settings.

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

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- coherence: Internal consistency, clear structure
- influence: World impact, power level
- conflict_potential: Story conflict opportunities
- distinctiveness: Memorable, unique qualities

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"coherence": <float 0-10>, "influence": <float 0-10>, "conflict_potential": <float 0-10>, "distinctiveness": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    # Resolve judge model and config once to avoid repeated resolution
    judge_model = svc._get_judge_model(entity_type="faction")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> FactionQualityScores:
        """Execute a single judge call for faction quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=FactionQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            if multi_call:
                logger.warning(
                    "Faction quality judgment failed for '%s': %s",
                    faction.get("name") or "Unknown",
                    e,
                )
            else:
                logger.exception(
                    "Faction quality judgment failed for '%s': %s",
                    faction.get("name") or "Unknown",
                    e,
                )
            raise WorldGenerationError(f"Faction quality judgment failed: {e}") from e

    return judge_with_averaging(_single_judge_call, FactionQualityScores, judge_config)


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
    threshold = svc.get_config().quality_threshold
    improvement_focus = []
    if scores.coherence < threshold:
        improvement_focus.append("Make internal logic more consistent")
    if scores.influence < threshold:
        improvement_focus.append("Increase world impact and power level")
    if scores.conflict_potential < threshold:
        improvement_focus.append("Add more story conflict opportunities")
    if scores.distinctiveness < threshold:
        improvement_focus.append("Make more unique and memorable")

    prompt = f"""TASK: Improve this faction to score HIGHER on the weak dimensions.

ORIGINAL FACTION:
Name: {faction.get("name", "Unknown")}
Description: {faction.get("description", "")}
Leader: {faction.get("leader", "Unknown")}
Goals: {", ".join(faction.get("goals", []))}
Values: {", ".join(faction.get("values", []))}

CURRENT SCORES (need {threshold}+ in all areas):
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
    """Format a list of existing faction names into a prompt-ready string.

    Parameters:
        existing_names: Existing faction names to include in the prompt.

    Returns:
        Newline-separated names each prefixed with "-".
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
