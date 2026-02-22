"""Character generation, judgment, and refinement functions."""

import logging
import random

from src.memory.story_state import Character, CharacterCreation, StoryState
from src.memory.world_quality import CharacterQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

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
    logger.debug(
        "Creating character for story %s (existing: %d)", story_state.id, len(existing_names)
    )
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

    calendar_context = svc.get_calendar_context()

    prompt = f"""Create a compelling NEW character for a {brief.genre} story.

STORY PREMISE: {brief.premise}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes)}
SETTING: {brief.setting_place}, {brief.setting_time}
{calendar_context}

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
6. Arc notes (arc_notes) - describe the character's potential transformation arc: how they might change over the story, what events could trigger growth or regression
7. Timeline placement - birth year and era from the calendar system (if available)

Write all text in {brief.language}."""

    try:
        model = svc._get_creator_model(entity_type="character")
        creation = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=CharacterCreation,
            temperature=temperature,
        )
        return creation.to_character()
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error("Character creation error for story %s: %s", story_state.id, summary)
        raise WorldGenerationError(f"Character creation failed: {summary}") from e


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

    calendar_context = svc.get_calendar_context()
    prompt = _build_character_judge_prompt(character, genre, calendar_context)

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
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Character quality judgment failed for '%s': %s", character.name, summary
                )
            else:
                logger.error(
                    "Character quality judgment failed for '%s': %s", character.name, summary
                )
            raise WorldGenerationError(f"Character quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, CharacterQualityScores, judge_config)


def _build_character_judge_prompt(
    character: Character, genre: str, calendar_context: str = ""
) -> str:
    """
    Constructs the textual prompt used by the judge model to evaluate a character's quality within a genre context.

    Parameters:
        character (Character): The character to be evaluated; the prompt will include name, role, description, traits, goals, arc notes, and temporal data.
        genre (str): The story genre used to frame evaluation criteria and tone.
        calendar_context (str): Formatted calendar/timeline context block for temporal validation.

    Returns:
        str: A formatted prompt string instructing the judge model to rate multiple quality dimensions and return a flat JSON object with numeric scores and feedback.
    """
    logger.debug("Building character judge prompt for '%s'", character.name)
    return f"""You are a literary critic evaluating character quality for a {genre} story.

CHARACTER TO EVALUATE:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.trait_names)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}
Birth Year: {character.birth_year if character.birth_year is not None else "N/A"}
Death Year: {character.death_year if character.death_year is not None else "N/A"}
Birth Era: {character.birth_era or "N/A"}
Temporal Notes: {character.temporal_notes or "N/A"}
{calendar_context}
{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- depth: Psychological complexity, internal contradictions, layers
- goal_clarity: Clarity, story relevance, want vs need tension
- flaws: Meaningful vulnerabilities that drive conflict
- uniqueness: Distinctiveness from genre archetypes
- arc_potential: Room for transformation and growth
- temporal_plausibility: Timeline consistency, era-appropriate placement

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"depth": <float 0-10>, "goal_clarity": <float 0-10>, "flaws": <float 0-10>, "uniqueness": <float 0-10>, "arc_potential": <float 0-10>, "temporal_plausibility": <float 0-10>, "feedback": "Your assessment..."}}

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
    logger.debug("Refining character '%s' for story %s", character.name, story_state.id)
    brief = story_state.brief
    threshold = svc.get_config().get_threshold("character")
    calendar_context = svc.get_calendar_context()

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
    if scores.temporal_plausibility < threshold:
        improvement_focus.append("Improve timeline placement and era consistency")

    calendar_section = f"\nCALENDAR & TIMELINE:\n{calendar_context}\n" if calendar_context else ""
    prompt = f"""TASK: Improve this character to score HIGHER on the weak dimensions.
{calendar_section}
ORIGINAL CHARACTER:
Name: {character.name}
Role: {character.role}
Description: {character.description}
Traits: {", ".join(character.trait_names)}
Goals: {", ".join(character.goals)}
Arc Notes: {character.arc_notes}
Birth Year: {character.birth_year if character.birth_year is not None else "N/A"}
Death Year: {character.death_year if character.death_year is not None else "N/A"}
Birth Era: {character.birth_era or "N/A"}
Temporal Notes: {character.temporal_notes or "N/A"}

CURRENT SCORES (need {threshold}+ in all areas):
- Depth: {scores.depth}/10
- Goal Clarity: {scores.goals}/10
- Flaws: {scores.flaws}/10
- Uniqueness: {scores.uniqueness}/10
- Arc Potential: {scores.arc_potential}/10
- Temporal Plausibility: {scores.temporal_plausibility}/10

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
        creation = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=CharacterCreation,
            temperature=temperature,
        )
        refined = creation.to_character()
        # Preserve fields not included in the refinement prompt — arc data
        # is populated during the write loop, and relationships aren't
        # part of the quality evaluation dimensions.
        refined.arc_progress = character.arc_progress.copy()
        refined.arc_type = character.arc_type
        refined.relationships = character.relationships.copy()
        return refined
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error("Character refinement failed for '%s': %s", character.name, summary)
        raise WorldGenerationError(f"Character refinement failed: {summary}") from e
