"""Concept generation, judgment, and refinement functions."""

import logging
from typing import Any

from src.memory.story_state import Concept, StoryState
from src.memory.world_quality import ConceptQualityScores, RefinementHistory
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
    retry_temperature,
)
from src.utils.exceptions import WorldGenerationError
from src.utils.validation import validate_unique_name

logger = logging.getLogger(__name__)


def generate_concept_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
) -> tuple[dict[str, Any], ConceptQualityScores, int]:
    """Generate a concept with quality refinement loop.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_names: Names of existing concepts to avoid.

    Returns:
        Tuple of (concept_dict, QualityScores, iterations_used)

    Raises:
        WorldGenerationError: If concept generation fails after all retries.
    """
    config = svc.get_config()
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
    creation_retries = 0  # Track duplicate-name retries for temperature escalation

    while iteration < config.max_iterations:
        try:
            if needs_fresh_creation:
                # Increase temperature on retries to avoid regenerating the same name
                retry_temp = retry_temperature(config, creation_retries)
                concept = svc._create_concept(story_state, existing_names, retry_temp)
            else:
                if concept and scores:
                    # Use dynamic temperature that decreases over iterations
                    dynamic_temp = config.get_refinement_temperature(iteration + 1)
                    concept = svc._refine_concept(
                        concept,
                        scores,
                        story_state,
                        dynamic_temp,
                    )

            if not concept.get("name"):
                creation_retries += 1
                last_error = f"Concept creation returned empty on iteration {iteration + 1}"
                logger.warning(
                    "%s (retry %d, next temp=%.2f)",
                    last_error,
                    creation_retries,
                    retry_temperature(config, creation_retries),
                )
                needs_fresh_creation = True  # Retry with fresh creation
                iteration += 1
                continue

            # Update history entity name
            if not history.entity_name:
                history.entity_name = concept.get("name", "Unknown")

            scores = svc._judge_concept_quality(concept, story_state, config.judge_temperature)
            needs_fresh_creation = False  # Successfully created, now can refine

            # Track this iteration
            history.add_iteration(
                entity_data=concept.copy(),
                scores=scores.to_dict(),
                average_score=scores.average,
                feedback=scores.feedback,
            )

            current_iter = history.iterations[-1].iteration
            logger.info(
                f"Concept '{concept.get('name')}' iteration {current_iter}: "
                f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                f"at iteration {history.best_iteration})"
            )

            if scores.average >= config.quality_threshold:
                logger.info(f"Concept '{concept.get('name')}' met quality threshold")
                history.final_iteration = current_iter
                history.final_score = scores.average
                svc._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=True,
                    early_stop_triggered=False,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return concept, scores, current_iter

            # Check for early stopping after tracking iteration (enhanced with variance tolerance)
            if history.should_stop_early(
                config.early_stopping_patience,
                min_iterations=config.early_stopping_min_iterations,
                variance_tolerance=config.early_stopping_variance_tolerance,
            ):
                reason = (
                    f"plateaued for {history.consecutive_plateaus} consecutive iterations"
                    if history.consecutive_plateaus >= config.early_stopping_patience
                    else f"degraded for {history.consecutive_degradations} consecutive iterations"
                )
                logger.info(
                    f"Early stopping: Concept '{concept.get('name')}' quality {reason} "
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

    if best_entity and history.iterations[-1].average_score < history.peak_score:
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
        best_scores: ConceptQualityScores | None = None
        for record in history.iterations:
            if record.iteration == history.best_iteration:
                best_scores = ConceptQualityScores(**record.scores)
                break
        if best_scores:
            history.final_iteration = history.best_iteration
            history.final_score = history.peak_score
            was_early_stop = len(history.iterations) < config.max_iterations
            svc._log_refinement_analytics(
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
        svc._log_refinement_analytics(
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
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    """Create a new concept using the creator model with structured generation."""
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
        logger.exception("Concept creation failed for story %s: %s", story_state.id, e)
        raise WorldGenerationError(f"Concept creation failed: {e}") from e


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
{{"relevance": <number>, "depth": <number>, "manifestation": <number>, "resonance": <number>, "feedback": "<string>"}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    # Resolve judge model once to avoid repeated resolution and duplicate conflict warnings
    judge_model = svc._get_judge_model(entity_type="concept")

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
            logger.exception(
                "Concept quality judgment failed for '%s': %s",
                concept.get("name") or "Unknown",
                e,
            )
            raise WorldGenerationError(f"Concept quality judgment failed: {e}") from e

    judge_config = svc.get_judge_config()
    return judge_with_averaging(_single_judge_call, ConceptQualityScores, judge_config)


def _refine_concept(
    svc,
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
        logger.exception(
            "Concept refinement failed for '%s': %s", concept.get("name") or "Unknown", e
        )
        raise WorldGenerationError(f"Concept refinement failed: {e}") from e
