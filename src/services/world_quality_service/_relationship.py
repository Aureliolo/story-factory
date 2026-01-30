"""Relationship generation, judgment, and refinement functions."""

import logging
from typing import Any

import ollama

from src.memory.story_state import StoryState
from src.memory.world_quality import RefinementHistory, RelationshipQualityScores
from src.services.llm_client import generate_structured
from src.utils.exceptions import WorldGenerationError
from src.utils.json_parser import extract_json

logger = logging.getLogger(__name__)


def generate_relationship_with_quality(
    svc,
    story_state: StoryState,
    entity_names: list[str],
    existing_rels: list[tuple[str, str]],
) -> tuple[dict[str, Any], RelationshipQualityScores, int]:
    """Generate a relationship with quality refinement loop.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        entity_names: Names of entities that can have relationships.
        existing_rels: Existing (source, target) pairs to avoid.

    Returns:
        Tuple of (relationship_dict, QualityScores, iterations_used)

    Raises:
        WorldGenerationError: If relationship generation fails after all retries.
    """
    config = svc.get_config()
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
                relationship = svc._create_relationship(
                    story_state, entity_names, existing_rels, config.creator_temperature
                )
            else:
                if relationship and scores:
                    # Use dynamic temperature that decreases over iterations
                    dynamic_temp = config.get_refinement_temperature(iteration + 1)
                    relationship = svc._refine_relationship(
                        relationship,
                        scores,
                        story_state,
                        dynamic_temp,
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
            if _is_duplicate_relationship(source, target, rel_type, existing_rels):
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

            scores = svc._judge_relationship_quality(
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
                svc._log_refinement_analytics(
                    history,
                    story_state.id,
                    threshold_met=True,
                    early_stop_triggered=False,
                    quality_threshold=config.quality_threshold,
                    max_iterations=config.max_iterations,
                )
                return relationship, scores, iteration + 1

            # Check for early stopping after tracking iteration (enhanced with variance tolerance)
            if history.should_stop_early(
                config.early_stopping_patience,
                min_iterations=config.early_stopping_min_iterations,
                variance_tolerance=config.early_stopping_variance_tolerance,
            ):
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
        best_scores: RelationshipQualityScores | None = None
        for record in history.iterations:
            if record.iteration == history.best_iteration:
                best_scores = RelationshipQualityScores(**record.scores)
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
    if relationship.get("source") and relationship.get("target") and scores:
        logger.warning(
            f"Relationship '{relationship.get('source')} -> {relationship.get('target')}' "
            f"did not meet quality threshold ({scores.average:.1f} < {config.quality_threshold}), "
            f"returning anyway"
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
        return relationship, scores, len(history.iterations)

    raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
        f"Failed to generate relationship after {config.max_iterations} attempts. "
        f"Last error: {last_error}"
    )


def _is_duplicate_relationship(
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
    svc,
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
        model = svc._get_creator_model(entity_type="relationship")
        response = svc.client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": svc.settings.llm_tokens_relationship_create,
            },
        )

        data = extract_json(response["response"], strict=False)
        if data and isinstance(data, dict):
            return data
        else:
            logger.error(f"Relationship creation returned invalid JSON structure: {data}")
            raise WorldGenerationError(f"Invalid relationship JSON structure: {data}")
    except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
        logger.error("Relationship creation LLM error for story %s: %s", story_state.id, e)
        raise WorldGenerationError(f"LLM error during relationship creation: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        logger.error("Relationship creation JSON parsing error for story %s: %s", story_state.id, e)
        raise WorldGenerationError(f"Invalid relationship response format: {e}") from e
    except WorldGenerationError:
        # Re-raise domain exceptions as-is
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error in relationship creation for story %s: %s", story_state.id, e
        )
        raise WorldGenerationError(f"Unexpected relationship creation error: {e}") from e


def _judge_relationship_quality(
    svc,
    relationship: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> RelationshipQualityScores:
    """Judge relationship quality using the validator model.

    Args:
        svc: WorldQualityService instance.
        relationship: Relationship dict to evaluate.
        story_state: Current story state for context.
        temperature: Judge temperature (low for consistency).

    Returns:
        RelationshipQualityScores with ratings and feedback.

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
- tension: Conflict potential
- dynamics: Complexity, power balance, history
- story_potential: Opportunities for scenes and development
- authenticity: Believability of the connection

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"tension": <number>, "dynamics": <number>, "story_potential": <number>, "authenticity": <number>, "feedback": "<string>"}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    try:
        model = svc._get_judge_model(entity_type="relationship")
        return generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=RelationshipQualityScores,
            temperature=temperature,
        )
    except Exception as e:
        logger.exception(
            "Relationship quality judgment failed for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"Relationship quality judgment failed: {e}") from e


def _refine_relationship(
    svc,
    relationship: dict[str, Any],
    scores: RelationshipQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine a relationship based on quality feedback."""
    brief = story_state.brief
    weak = scores.weak_dimensions(svc.get_config().quality_threshold)

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
        model = svc._get_creator_model(entity_type="relationship")
        response = svc.client.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={
                "temperature": temperature,
                "num_predict": svc.settings.llm_tokens_relationship_refine,
            },
        )

        data = extract_json(response["response"], strict=False)
        if data and isinstance(data, dict):
            return data
        else:
            logger.error(f"Relationship refinement returned invalid JSON structure: {data}")
            raise WorldGenerationError(f"Invalid relationship refinement JSON structure: {data}")
    except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
        logger.error(
            "Relationship refinement LLM error for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"LLM error during relationship refinement: {e}") from e
    except (ValueError, KeyError, TypeError) as e:
        logger.error(
            "Relationship refinement JSON parsing error for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"Invalid relationship refinement response format: {e}") from e
    except WorldGenerationError:
        # Re-raise domain exceptions as-is
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error in relationship refinement for %s->%s: %s",
            relationship.get("source") or "Unknown",
            relationship.get("target") or "Unknown",
            e,
        )
        raise WorldGenerationError(f"Unexpected relationship refinement error: {e}") from e
