"""Character generation, judgment, and refinement functions."""

import logging
import random

from src.memory.story_state import Character, StoryState
from src.memory.world_quality import CharacterQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger(__name__)


def generate_character_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    custom_instructions: str | None = None,
) -> tuple[Character, CharacterQualityScores, int]:
    """Generate a character with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    characters until the quality threshold is met or stopping criteria is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_names: Names to avoid when generating.
        custom_instructions: Optional additional instructions for the creator model.

    Returns:
        Tuple of (character, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If generation fails after all attempts.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for character generation")

    return quality_refinement_loop(
        entity_type="character",
        create_fn=lambda retries: svc._create_character(
            story_state,
            existing_names,
            config.creator_temperature,
            custom_instructions,
        ),
        judge_fn=lambda char: svc._judge_character_quality(
            char,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda char, scores, iteration: svc._refine_character(
            char,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda char: char.name,
        serialize=lambda char: char.model_dump(),
        is_empty=lambda char: not char.name,
        score_cls=CharacterQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
    )


def review_character_quality(
    svc,
    character: Character,
    story_state: StoryState,
) -> tuple[Character, CharacterQualityScores, int]:
    """Review and optionally refine an existing character (e.g. from Architect output).

    Uses the generic quality refinement loop in review mode: skips creation,
    judges the provided character, and refines if below threshold.

    Parameters:
        svc: WorldQualityService instance.
        character: Existing character to review.
        story_state: Current story state with brief.

    Returns:
        Tuple of (character, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If refinement fails after all attempts.
    """
    config = svc.get_config()

    return quality_refinement_loop(
        entity_type="character",
        create_fn=lambda retries: character,
        judge_fn=lambda char: svc._judge_character_quality(
            char,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda char, scores, iteration: svc._refine_character(
            char,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda char: char.name,
        serialize=lambda char: char.model_dump(),
        is_empty=lambda char: not char.name,
        score_cls=CharacterQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
        initial_entity=character,
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
    """Judge character quality using the judge model.

    Supports multi-call averaging when judge_multi_call_enabled is True in settings.
    Multiple judge calls are aggregated using ScoreStatistics with outlier detection.

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

    prompt = _build_character_judge_prompt(character, genre)

    # Resolve judge model and config once to avoid repeated resolution
    judge_model = svc._get_judge_model(entity_type="character")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> CharacterQualityScores:
        """Execute a single judge call for character quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=CharacterQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            if multi_call:
                logger.warning("Character quality judgment failed for '%s': %s", character.name, e)
            else:
                logger.exception(
                    "Character quality judgment failed for '%s': %s", character.name, e
                )
            raise WorldGenerationError(f"Character quality judgment failed: {e}") from e

    return judge_with_averaging(_single_judge_call, CharacterQualityScores, judge_config)


def _build_character_judge_prompt(character: Character, genre: str) -> str:
    """Build the judge prompt for character quality evaluation.

    Args:
        character: Character to evaluate.
        genre: Story genre for context.

    Returns:
        Formatted prompt string.
    """
    logger.debug("Building character judge prompt for '%s'", character.name)
    return f"""You are a literary critic evaluating character quality for a {genre} story.

CHARACTER TO EVALUATE:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- depth: Psychological complexity, internal contradictions, layers
- goal_clarity: Clarity, story relevance, want vs need tension
- flaws: Meaningful vulnerabilities that drive conflict
- uniqueness: Distinctiveness from genre archetypes
- arc_potential: Room for transformation and growth

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"depth": 6.3, "goal_clarity": 7.8, "flaws": 5.1, "uniqueness": 8.2, "arc_potential": 6.9, "feedback": "The character's..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""


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
    threshold = svc.get_config().quality_threshold

    # Build specific improvement instructions from feedback
    improvement_focus = []
    if scores.depth < threshold:
        improvement_focus.append(
            "Add deeper psychological complexity — internal contradictions, hidden motivations"
        )
    if scores.goals < threshold:
        improvement_focus.append("Clarify goals with specific want-vs-need tension")
    if scores.flaws < threshold:
        improvement_focus.append("Add meaningful flaws that create real conflict")
    if scores.uniqueness < threshold:
        improvement_focus.append("Avoid genre archetypes, add surprising traits")
    if scores.arc_potential < threshold:
        improvement_focus.append("Expand transformation potential with specific turning points")

    prompt = f"""TASK: Improve this character to score HIGHER on the weak dimensions.

ORIGINAL CHARACTER:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.personality_traits)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}

CURRENT SCORES (need {threshold}+ in all areas):
- Depth: {scores.depth}/10
- Goal Clarity: {scores.goals}/10
- Flaws: {scores.flaws}/10
- Uniqueness: {scores.uniqueness}/10
- Arc Potential: {scores.arc_potential}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the exact name: "{character.name}"
2. Keep the role: "{character.role}"
3. Make SUBSTANTIAL improvements to weak areas
4. Add concrete details, not vague generalities
5. Output in {brief.language if brief else "English"}

Return ONLY the improved character."""

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
