"""Item generation, judgment, and refinement functions."""

import logging
from typing import Any

from src.memory.story_state import Item, StoryState
from src.memory.world_quality import ItemQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
    retry_temperature,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error
from src.utils.validation import validate_unique_name

logger = logging.getLogger(__name__)


def generate_item_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
) -> tuple[dict[str, Any], ItemQualityScores, int]:
    """Generate an item with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    items until the quality threshold is met or stopping criteria is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_names: Existing item names to avoid duplicates.

    Returns:
        Tuple of (item_dict, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If item generation fails after all attempts.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for item generation")

    return quality_refinement_loop(
        entity_type="item",
        create_fn=lambda retries: svc._create_item(
            story_state,
            existing_names,
            retry_temperature(config, retries),
        ),
        judge_fn=lambda item: svc._judge_item_quality(
            item,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda item, scores, iteration: svc._refine_item(
            item,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda item: item.get("name", "Unknown"),
        serialize=lambda item: item.copy(),
        is_empty=lambda item: not item.get("name"),
        score_cls=ItemQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
    )


def _create_item(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    """Create a new item using the creator model with structured generation."""
    logger.debug("Creating item for story %s (existing: %d)", story_state.id, len(existing_names))
    brief = story_state.brief
    if not brief:
        return {}

    # Format existing names with explicit warnings
    existing_names_formatted = svc._format_existing_names_warning(existing_names, "item")

    calendar_context = svc.get_calendar_context()

    prompt = f"""Create a significant item/object for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}
{calendar_context}
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
5. Timeline placement - creation year and era (if calendar available)

Write all text in {brief.language}."""

    try:
        model = svc._get_creator_model(entity_type="item")
        item = generate_structured(
            settings=svc.settings,
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
                check_semantic=svc.settings.semantic_duplicate_enabled,
                semantic_threshold=svc.settings.semantic_duplicate_threshold,
                ollama_url=svc.settings.ollama_url,
                embedding_model=svc.settings.embedding_model,
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
        summary = summarize_llm_error(e)
        logger.error("Item creation failed for story %s: %s", story_state.id, summary)
        raise WorldGenerationError(f"Item creation failed: {summary}") from e


def _judge_item_quality(
    svc,
    item: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> ItemQualityScores:
    """
    Evaluate an item's quality along several story-relevant dimensions and return numeric scores plus actionable feedback.

    Supports multi-call averaging when judge multi-call is enabled in the judge configuration.

    Parameters:
        item (dict): The item to evaluate (expects keys like "name", "description", "significance", "properties").
        story_state (StoryState): Current story context used to determine genre and evaluation framing.
        temperature (float): Sampling temperature for the judge model (lower values favor consistency).

    Returns:
        ItemQualityScores: Scores for `story_significance`, `uniqueness`, `narrative_potential`, `integration`, `temporal_plausibility`, and a `feedback` field.

    Raises:
        WorldGenerationError: If judgment fails or returns invalid data.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    formatted_properties = svc._format_properties(item.get("properties", []))
    calendar_context = svc.get_calendar_context()

    prompt = f"""You are evaluating an item for a {genre} story.

ITEM TO EVALUATE:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {formatted_properties}
Creation Year: {item.get("creation_year", "N/A")}
Creation Era: {item.get("creation_era", "N/A")}
Temporal Notes: {item.get("temporal_notes", "N/A")}
{calendar_context}
{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- story_significance: Story importance, plot relevance
- uniqueness: Distinctive qualities
- narrative_potential: Opportunities for scenes
- integration: How well it fits the world
- temporal_plausibility: VERIFY against CALENDAR above — creation/discovery dates MUST fall within a defined era's [start_year, end_year] range. Score 8-10 ONLY if temporal references are consistent with the calendar eras AND are self-consistent. Score 5-7 if time references exist but era alignment is ambiguous. Score 2-4 if dates conflict with defined era ranges. If the CALENDAR block states "No calendar available", score exactly 3.0 (insufficient context to verify).

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"story_significance": <float 0-10>, "uniqueness": <float 0-10>, "narrative_potential": <float 0-10>, "integration": <float 0-10>, "temporal_plausibility": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    # Resolve judge model and config once to avoid repeated resolution
    judge_model = svc._get_judge_model(entity_type="item")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> ItemQualityScores:
        """Execute a single judge call for item quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=ItemQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Item quality judgment failed for '%s': %s",
                    item.get("name") or "Unknown",
                    summary,
                )
            else:
                logger.error(
                    "Item quality judgment failed for '%s': %s",
                    item.get("name") or "Unknown",
                    summary,
                )
            raise WorldGenerationError(f"Item quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, ItemQualityScores, judge_config)


def _refine_item(
    svc,
    item: dict[str, Any],
    scores: ItemQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine an item based on quality feedback using structured generation."""
    logger.debug("Refining item '%s' for story %s", item.get("name", "Unknown"), story_state.id)
    brief = story_state.brief

    # Build specific improvement instructions from feedback
    threshold = svc.get_config().get_threshold("item")
    improvement_focus = []
    if scores.significance < threshold:
        improvement_focus.append("Increase story importance and plot relevance")
    if scores.uniqueness < threshold:
        improvement_focus.append("Make more distinctive with unique qualities")
    if scores.narrative_potential < threshold:
        improvement_focus.append("Add more opportunities for scenes and conflict")
    if scores.integration < threshold:
        improvement_focus.append("Improve how naturally it fits into the world")
    if scores.temporal_plausibility < threshold:
        improvement_focus.append("Improve timeline placement and era consistency")

    calendar_context = svc.get_calendar_context()

    prompt = f"""TASK: Improve this item to score HIGHER on the weak dimensions.

ORIGINAL ITEM:
Name: {item.get("name", "Unknown")}
Description: {item.get("description", "")}
Significance: {item.get("significance", "")}
Properties: {svc._format_properties(item.get("properties", []))}
Creation Year: {item.get("creation_year", "N/A")}
Creation Era: {item.get("creation_era", "N/A")}
Temporal Notes: {item.get("temporal_notes", "N/A")}
{calendar_context}
CURRENT SCORES (need {threshold}+ in all areas):
- Story Significance: {scores.significance}/10
- Uniqueness: {scores.uniqueness}/10
- Narrative Potential: {scores.narrative_potential}/10
- Integration: {scores.integration}/10
- Temporal Plausibility: {scores.temporal_plausibility}/10

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
        model = svc._get_creator_model(entity_type="item")
        refined = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=Item,
            temperature=temperature,
        )

        # Ensure name and temporal fields are preserved from original item
        result = refined.model_dump()
        result["name"] = item.get("name", "Unknown")
        result["type"] = "item"
        for key in ("creation_year", "creation_era", "temporal_notes"):
            if result.get(key) in (None, "") and item.get(key) not in (None, ""):
                result[key] = item[key]
                logger.debug(
                    "Preserved temporal field '%s' from original item '%s'", key, result["name"]
                )
        return result
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error("Item refinement failed for '%s': %s", item.get("name") or "Unknown", summary)
        raise WorldGenerationError(f"Item refinement failed: {summary}") from e
