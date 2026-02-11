"""Concept generation, judgment, and refinement functions."""

import logging
from typing import Any

from src.memory.story_state import Concept, StoryState
from src.memory.world_quality import ConceptQualityScores
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


def generate_concept_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
) -> tuple[dict[str, Any], ConceptQualityScores, int]:
    """Generate a concept with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    concepts until the quality threshold is met or stopping criteria is reached.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_names: Names of existing concepts to avoid.

    Returns:
        Tuple of (concept_dict, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If concept generation fails after all retries.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for concept generation")

    return quality_refinement_loop(
        entity_type="concept",
        create_fn=lambda retries: svc._create_concept(
            story_state,
            existing_names,
            retry_temperature(config, retries),
        ),
        judge_fn=lambda concept: svc._judge_concept_quality(
            concept,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda concept, scores, iteration: svc._refine_concept(
            concept,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda concept: concept.get("name", "Unknown"),
        serialize=lambda concept: concept.copy(),
        is_empty=lambda concept: not concept.get("name"),
        score_cls=ConceptQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
    )


def _create_concept(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    """Create a new concept using the creator model with structured generation."""
    logger.debug(
        "Creating concept for story %s (existing: %d)", story_state.id, len(existing_names)
    )
    brief = story_state.brief
    if not brief:
        return {}

    # Format existing names with explicit warnings
    existing_names_formatted = svc._format_existing_names_warning(existing_names, "concept")

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
        model = svc._get_creator_model(entity_type="concept")
        concept = generate_structured(
            settings=svc.settings,
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
                check_semantic=svc.settings.semantic_duplicate_enabled,
                semantic_threshold=svc.settings.semantic_duplicate_threshold,
                ollama_url=svc.settings.ollama_url,
                embedding_model=svc.settings.embedding_model,
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
        summary = summarize_llm_error(e)
        logger.error("Concept creation failed for story %s: %s", story_state.id, summary)
        raise WorldGenerationError(f"Concept creation failed: {summary}") from e


def _judge_concept_quality(
    svc,
    concept: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> ConceptQualityScores:
    """Judge concept quality using the judge model.

    Supports multi-call averaging when judge_multi_call_enabled is True in settings.

    Args:
        svc: WorldQualityService instance.
        concept: Concept dict to evaluate.
        story_state: Current story state for context.
        temperature: Judge temperature (low for consistency).

    Returns:
        ConceptQualityScores with ratings and feedback.

    Raises:
        WorldGenerationError: If quality judgment fails or returns invalid data.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    prompt = f"""You are evaluating a thematic concept for a {genre} story.

CONCEPT TO EVALUATE:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- relevance: Alignment with story themes
- depth: Philosophical richness
- manifestation: How well it can appear in story
- resonance: Emotional impact potential

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"relevance": <float 0-10>, "depth": <float 0-10>, "manifestation": <float 0-10>, "resonance": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    # Resolve judge model and config once to avoid repeated resolution
    judge_model = svc._get_judge_model(entity_type="concept")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> ConceptQualityScores:
        """Execute a single judge call for concept quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=ConceptQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Concept quality judgment failed for '%s': %s",
                    concept.get("name") or "Unknown",
                    summary,
                )
            else:
                logger.error(
                    "Concept quality judgment failed for '%s': %s",
                    concept.get("name") or "Unknown",
                    summary,
                )
            raise WorldGenerationError(f"Concept quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, ConceptQualityScores, judge_config)


def _refine_concept(
    svc,
    concept: dict[str, Any],
    scores: ConceptQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine a concept based on quality feedback using structured generation."""
    logger.debug(
        "Refining concept '%s' for story %s", concept.get("name", "Unknown"), story_state.id
    )
    brief = story_state.brief

    # Build specific improvement instructions from feedback
    threshold = svc.get_config().get_threshold("concept")
    improvement_focus = []
    if scores.relevance < threshold:
        improvement_focus.append("Strengthen alignment with story themes")
    if scores.depth < threshold:
        improvement_focus.append("Add more philosophical richness and complexity")
    if scores.manifestation < threshold:
        improvement_focus.append("Provide clearer ways the concept appears in the story")
    if scores.resonance < threshold:
        improvement_focus.append("Increase emotional impact potential")

    prompt = f"""TASK: Improve this concept to score HIGHER on the weak dimensions.

ORIGINAL CONCEPT:
Name: {concept.get("name", "Unknown")}
Description: {concept.get("description", "")}
Manifestations: {concept.get("manifestations", "")}

CURRENT SCORES (need {threshold}+ in all areas):
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
        model = svc._get_creator_model(entity_type="concept")
        refined = generate_structured(
            settings=svc.settings,
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
        summary = summarize_llm_error(e)
        logger.error(
            "Concept refinement failed for '%s': %s", concept.get("name") or "Unknown", summary
        )
        raise WorldGenerationError(f"Concept refinement failed: {summary}") from e
