"""Location generation, judgment, and refinement functions."""

import logging
from typing import Any

from src.memory.story_state import Location, StoryState
from src.memory.world_quality import LocationQualityScores
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


def generate_location_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
) -> tuple[dict[str, Any], LocationQualityScores, int]:
    """Generate a location with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    locations until the quality threshold is met or stopping criteria is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_names: Names of existing locations to avoid duplicates.

    Returns:
        Tuple of (location_dict, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If no valid location could be produced after all attempts.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for location generation")

    prep_creator, prep_judge = svc._make_model_preparers("location")

    return quality_refinement_loop(
        entity_type="location",
        create_fn=lambda retries: svc._create_location(
            story_state,
            existing_names,
            retry_temperature(config, retries),
        ),
        judge_fn=lambda loc: svc._judge_location_quality(
            loc,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda loc, scores, iteration: svc._refine_location(
            loc,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda loc: loc.get("name", "Unknown"),
        serialize=lambda loc: loc.copy(),
        is_empty=lambda loc: not loc.get("name"),
        score_cls=LocationQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
        prepare_creator=prep_creator,
        prepare_judge=prep_judge,
    )


def _create_location(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    """Create a new location using the creator model with structured generation."""
    logger.debug(
        "Creating location for story %s (existing: %d)", story_state.id, len(existing_names)
    )
    brief = story_state.brief
    if not brief:
        return {}

    # Format existing names with explicit warnings
    existing_names_formatted = svc._format_existing_names_warning(existing_names, "location")

    calendar_context = svc.get_calendar_context()

    prompt = f"""Create a compelling location for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
{calendar_context}
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
5. Timeline placement - founding year and era (if calendar available)

Write all text in {brief.language}."""

    try:
        model = svc._get_creator_model(entity_type="location")
        location = generate_structured(
            settings=svc.settings,
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
                check_semantic=svc.settings.semantic_duplicate_enabled,
                semantic_threshold=svc.settings.semantic_duplicate_threshold,
                ollama_url=svc.settings.ollama_url,
                embedding_model=svc.settings.embedding_model,
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
        summary = summarize_llm_error(e)
        logger.error("Location creation failed for story %s: %s", story_state.id, summary)
        raise WorldGenerationError(f"Location creation failed: {summary}") from e


def _judge_location_quality(
    svc,
    location: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> LocationQualityScores:
    """
    Evaluate a generated location's quality across the defined dimensions.

    Uses the configured judge model to produce numeric scores for atmosphere, narrative_significance, story_relevance, distinctiveness, and temporal_plausibility, and returns targeted improvement feedback. Honors multi-call averaging when enabled in the judge configuration.

    Parameters:
        svc: Service client providing model resolution and settings.
        location (dict): Location data (expects keys like `name`, `description`, `significance`).
        story_state (StoryState): Story context used to determine genre and language.
        temperature (float): Sampling temperature to use with the judge model.

    Returns:
        LocationQualityScores: Numeric scores for each dimension and a `feedback` string.

    Raises:
        WorldGenerationError: If the judge model call fails or returns invalid data.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    calendar_context = svc.get_calendar_context()

    prompt = f"""You are evaluating a location for a {genre} story.

LOCATION TO EVALUATE:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}
Founding Year: {location.get("founding_year", "N/A")}
Destruction Year: {location.get("destruction_year", "N/A")}
Founding Era: {location.get("founding_era", "N/A")}
Temporal Notes: {location.get("temporal_notes", "N/A")}
{calendar_context}
{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- atmosphere: Sensory richness, mood, immersion
- narrative_significance: Plot or symbolic meaning
- story_relevance: Connections to themes and characters
- distinctiveness: Memorable, unique qualities
- temporal_plausibility: VERIFY against CALENDAR above — all temporal dates (including founding and destruction) MUST fall within defined era [start_year, end_year] ranges, and chronology must be self-consistent (e.g., destruction_year >= founding_year when both exist). Score 8-10 ONLY if all temporal references satisfy these checks. Score 5-7 if time references exist but era alignment is ambiguous. Score 2-4 if any date conflicts with defined era ranges or chronology. If temporal dates are "N/A" (not yet assigned), score 5.0 (neutral — dates are pending, not wrong). If the CALENDAR block states "No calendar available", score 5.0 (insufficient context to verify).

Provide specific improvement feedback in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"atmosphere": <float 0-10>, "narrative_significance": <float 0-10>, "story_relevance": <float 0-10>, "distinctiveness": <float 0-10>, "temporal_plausibility": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    # Resolve judge model and config once to avoid repeated resolution
    judge_model = svc._get_judge_model(entity_type="location")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> LocationQualityScores:
        """Execute a single judge call for location quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=LocationQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Location quality judgment failed for '%s': %s",
                    location.get("name", "Unknown"),
                    summary,
                )
            else:
                logger.error(
                    "Location quality judgment failed for '%s': %s",
                    location.get("name", "Unknown"),
                    summary,
                )
            raise WorldGenerationError(f"Location quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, LocationQualityScores, judge_config)


def _refine_location(
    svc,
    location: dict[str, Any],
    scores: LocationQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine a location based on quality feedback using structured generation."""
    logger.debug(
        "Refining location '%s' for story %s", location.get("name", "Unknown"), story_state.id
    )
    brief = story_state.brief

    # Build specific improvement instructions from feedback
    threshold = svc.get_config().get_threshold("location")
    improvement_focus = []
    if scores.atmosphere < threshold:
        improvement_focus.append("Add richer sensory details and mood")
    if scores.significance < threshold:
        improvement_focus.append("Deepen the plot or symbolic meaning")
    if scores.story_relevance < threshold:
        improvement_focus.append("Strengthen connections to themes and characters")
    if scores.distinctiveness < threshold:
        improvement_focus.append("Make more memorable with unique qualities")
    if scores.temporal_plausibility < threshold:
        improvement_focus.append("Improve timeline placement and era consistency")

    calendar_context = svc.get_calendar_context()

    prompt = f"""TASK: Improve this location to score HIGHER on the weak dimensions.

ORIGINAL LOCATION:
Name: {location.get("name", "Unknown")}
Description: {location.get("description", "")}
Significance: {location.get("significance", "")}
Founding Year: {location.get("founding_year", "N/A")}
Destruction Year: {location.get("destruction_year", "N/A")}
Founding Era: {location.get("founding_era", "N/A")}
Temporal Notes: {location.get("temporal_notes", "N/A")}
{calendar_context}
CURRENT SCORES (need {threshold}+ in all areas):
- Atmosphere: {scores.atmosphere}/10
- Narrative Significance: {scores.significance}/10
- Story Relevance: {scores.story_relevance}/10
- Distinctiveness: {scores.distinctiveness}/10
- Temporal Plausibility: {scores.temporal_plausibility}/10

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
        model = svc._get_creator_model(entity_type="location")
        refined = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=Location,
            temperature=temperature,
        )

        # Ensure name and temporal fields are preserved from original location
        result = refined.model_dump()
        result["name"] = location.get("name", "Unknown")
        result["type"] = "location"
        for key in ("founding_year", "destruction_year", "founding_era", "temporal_notes"):
            if result.get(key) in (None, "") and location.get(key) not in (None, ""):
                result[key] = location[key]
                logger.debug(
                    "Preserved temporal field '%s' from original location '%s'", key, result["name"]
                )
        return result
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error(
            "Location refinement failed for '%s': %s", location.get("name") or "Unknown", summary
        )
        raise WorldGenerationError(f"Location refinement failed: {summary}") from e
