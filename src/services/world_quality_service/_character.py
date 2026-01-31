"""Character generation, judgment, and refinement functions."""

import logging
import random

from src.memory.story_state import Character, StoryState
from src.memory.world_quality import CharacterQualityScores, RefinementHistory
from src.services.llm_client import generate_structured
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)


def generate_character_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    custom_instructions: str | None = None,
) -> tuple[Character, CharacterQualityScores, int]:
    """
    Generate a character and iteratively refine it until a quality threshold or stopping criteria is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state (StoryState): Current story state containing the brief and identifiers used for generation and analytics.
        existing_names (list[str]): Names to avoid when generating a new character.
        custom_instructions (str | None): Optional additional instructions to influence creator model output.

    Returns:
        tuple[Character, CharacterQualityScores, int]: The selected character, its quality scores, and the number of iterations performed.

    Raises:
        WorldGenerationError: If generation fails and no valid iterations were produced.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for character generation")

    logger.info(f"Generating character with quality threshold {config.quality_threshold}")

    # Track all iterations for best-selection using RefinementHistory
    history = RefinementHistory(entity_type="character", entity_name="")
    iteration = 0
    character: Character | None = None
    scores: CharacterQualityScores | None = None
    last_error: str = ""

    while iteration < config.max_iterations:
        try:
            # Create new character on first iteration OR if previous returned None
            # (e.g., validation failure or duplicate name)
            if iteration == 0 or character is None:
                # Initial generation
                character = svc._create_character(
                    story_state,
                    existing_names,
                    config.creator_temperature,
                    custom_instructions,
                )
            else:
                # Refinement based on feedback
                if character and scores:
                    # Use dynamic temperature that decreases over iterations
                    dynamic_temp = config.get_refinement_temperature(iteration + 1)
                    character = svc._refine_character(
                        character,
                        scores,
                        story_state,
                        dynamic_temp,
                    )

            if character is None:
                last_error = f"Character creation returned None on iteration {iteration + 1}"
                logger.error(last_error)
                iteration += 1
                continue

            # Update history entity name
            if not history.entity_name:
                history.entity_name = character.name

            # Judge quality - this can raise if parsing fails
            scores = svc._judge_character_quality(character, story_state, config.judge_temperature)

            # Track this iteration
            history.add_iteration(
                entity_data=character.model_dump(),
                scores=scores.to_dict(),
                average_score=scores.average,
                feedback=scores.feedback,
            )

            logger.info(
                f"Character '{character.name}' iteration {iteration + 1}: "
                f"score {scores.average:.1f} (best so far: {history.peak_score:.1f} "
                f"at iteration {history.best_iteration})"
            )

            if scores.average >= config.quality_threshold:
                logger.info(f"Character '{character.name}' met quality threshold")
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
                return character, scores, iteration + 1

            # Check for early stopping after tracking iteration (enhanced with variance tolerance)
            if history.should_stop_early(
                config.early_stopping_patience,
                min_iterations=config.early_stopping_min_iterations,
                variance_tolerance=config.early_stopping_variance_tolerance,
            ):
                logger.info(
                    f"Early stopping: Character '{character.name}' quality degraded "
                    f"for {history.consecutive_degradations} consecutive iterations "
                    f"(patience: {config.early_stopping_patience}). "
                    f"Stopping at iteration {iteration + 1}."
                )
                break

        except WorldGenerationError as e:
            last_error = str(e)
            logger.error(f"Character generation error on iteration {iteration + 1}: {e}")

        iteration += 1

    # Didn't meet threshold - return BEST iteration, not last
    if not history.iterations:
        raise WorldGenerationError(
            f"Failed to generate character after {config.max_iterations} attempts. "
            f"Last error: {last_error}"
        )

    # Pick best iteration (not necessarily the last one)
    best_entity = history.get_best_entity()

    if best_entity and history.iterations[-1].average_score < history.peak_score:
        logger.warning(
            f"Character '{history.entity_name}' iterations got WORSE after peak. "
            f"Best: iteration {history.best_iteration} ({history.peak_score:.1f}), "
            f"Final: iteration {len(history.iterations)} "
            f"({history.iterations[-1].average_score:.1f}). "
            f"Returning best iteration."
        )

    # Return best entity or last one
    if best_entity:
        best_character = Character(**best_entity)
        # Reconstruct scores from best iteration
        best_scores: CharacterQualityScores | None = None
        for record in history.iterations:
            if record.iteration == history.best_iteration:
                best_scores = CharacterQualityScores(**record.scores)
                break
        if best_scores:
            history.final_iteration = history.best_iteration
            history.final_score = history.peak_score
            # Early stop = exited before max iterations due to degradation patience.
            # If we're in this block (returning best entity), we didn't meet
            # threshold early (that returns from within the loop), so
            # iterations < max means degradation-based early stop triggered.
            was_early_stop = len(history.iterations) < config.max_iterations
            svc._log_refinement_analytics(
                history,
                story_state.id,
                threshold_met=history.peak_score >= config.quality_threshold,
                early_stop_triggered=was_early_stop,
                quality_threshold=config.quality_threshold,
                max_iterations=config.max_iterations,
            )
            return best_character, best_scores, history.best_iteration

    # Fallback to last iteration
    if character and scores:
        logger.warning(
            f"Character '{character.name}' did not meet quality threshold "
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
        return character, scores, len(history.iterations)

    raise WorldGenerationError(  # pragma: no cover - defensive, unreachable
        f"Failed to generate character after {config.max_iterations} attempts. "
        f"Last error: {last_error}"
    )


def _create_character(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    temperature: float,
    custom_instructions: str | None = None,
) -> Character | None:
    """Create a new character using the creator model.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_names: Names to avoid.
        temperature: Generation temperature.
        custom_instructions: Optional custom instructions to refine generation.

    Returns:
        New Character or None on failure.
    """
    brief = story_state.brief
    if not brief:
        return None

    # Random naming hint to encourage variety
    naming_styles = [
        "Use an unexpected, fresh name - avoid common fantasy names like Elara, Kael, Thorne, or Lyra.",
        "Draw inspiration from diverse cultures for a unique name.",
        "Create a memorable name that reflects the character's personality.",
        "Use a short, punchy name or a longer, elaborate one - be creative.",
    ]
    naming_hint = random.choice(naming_styles)

    # Build custom instructions section if provided
    custom_section = ""
    if custom_instructions:
        custom_section = f"\n\nSPECIFIC REQUIREMENTS:\n{custom_instructions}\n"

    prompt = f"""Create a compelling NEW character for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}
SETTING: {brief.setting_place}, {brief.setting_time}

EXISTING CHARACTERS IN THIS WORLD: {", ".join(existing_names) if existing_names else "None yet"}
(Create a NEW character with a different name that complements these existing ones)

NAMING: {naming_hint}
{custom_section}
Create a character with:
1. Deep psychological complexity - internal contradictions, layers
2. Clear goals - what they want vs what they need
3. Meaningful flaws that drive conflict
4. Uniqueness - not a genre archetype
5. Arc potential - room for transformation

Write all text in {brief.language}."""

    try:
        model = svc._get_creator_model(entity_type="character")
        character = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=Character,
            temperature=temperature,
        )
        return character
    except Exception as e:
        logger.error("Character creation error for story %s: %s", story_state.id, e)
        raise WorldGenerationError(f"Character creation failed: {e}") from e


def _judge_character_quality(
    svc,
    character: Character,
    story_state: StoryState,
    temperature: float,
) -> CharacterQualityScores:
    """Judge character quality using the validator model.

    Args:
        svc: WorldQualityService instance.
        character: Character to evaluate.
        story_state: Current story state for context.
        temperature: Judge temperature (low for consistency).

    Returns:
        CharacterQualityScores with ratings and feedback.

    Raises:
        WorldGenerationError: If quality judgment fails or returns invalid data.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    prompt = f"""You are a literary critic evaluating character quality for a {genre} story.

CHARACTER TO EVALUATE:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

SCORING CALIBRATION - BE STRICT:
- 1-3: Poor quality, generic or incoherent
- 4-5: Below average, lacks depth or originality
- 6-7: Average, functional but unremarkable (most first drafts land here)
- 8-9: Good, well-crafted with clear strengths
- 10: Exceptional, publication-ready
Most entities should score 5-7 on first attempt. Only give 8+ if genuinely impressive.
Do NOT default to high scores — a 7 is already a good score.

Rate each dimension 0-10:
- depth: Psychological complexity, internal contradictions, layers
- goals: Clarity, story relevance, want vs need tension
- flaws: Meaningful vulnerabilities that drive conflict
- uniqueness: Distinctiveness from genre archetypes
- arc_potential: Room for transformation and growth

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"depth": <number>, "goals": <number>, "flaws": <number>, "uniqueness": <number>, "arc_potential": <number>, "feedback": "<string>"}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    try:
        model = svc._get_judge_model(entity_type="character")
        return generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=CharacterQualityScores,
            temperature=temperature,
        )
    except Exception as e:
        logger.exception("Character quality judgment failed for '%s': %s", character.name, e)
        raise WorldGenerationError(f"Character quality judgment failed: {e}") from e


def _refine_character(
    svc,
    character: Character,
    scores: CharacterQualityScores,
    story_state: StoryState,
    temperature: float,
) -> Character:
    """Refine a character based on quality feedback.

    Args:
        svc: WorldQualityService instance.
        character: Character to refine.
        scores: Quality scores with feedback.
        story_state: Current story state.
        temperature: Refinement temperature.

    Returns:
        Refined Character.
    """
    brief = story_state.brief
    weak = scores.weak_dimensions(svc.get_config().quality_threshold)

    prompt = f"""Improve this character based on quality feedback.

ORIGINAL CHARACTER:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

QUALITY SCORES (0-10):
- Depth: {scores.depth}
- Goals: {scores.goals}
- Flaws: {scores.flaws}
- Uniqueness: {scores.uniqueness}
- Arc Potential: {scores.arc_potential}

FEEDBACK: {scores.feedback}

WEAK AREAS TO IMPROVE: {", ".join(weak) if weak else "None - minor improvements only"}

Keep the name "{character.name}" and role "{character.role}", but enhance the weak areas.
Make the character more compelling while maintaining consistency.
Write all text in {brief.language if brief else "English"}."""

    try:
        model = svc._get_creator_model(entity_type="character")
        return generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=Character,
            temperature=temperature,
        )
    except Exception as e:
        logger.exception("Character refinement failed for '%s': %s", character.name, e)
        raise WorldGenerationError(f"Character refinement failed: {e}") from e
