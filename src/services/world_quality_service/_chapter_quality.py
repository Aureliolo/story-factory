"""Chapter quality review, judgment, and refinement functions."""

import logging

from src.memory.story_state import Chapter, StoryState
from src.memory.world_quality import ChapterQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

logger = logging.getLogger(__name__)


def review_chapter_quality(
    svc,
    chapter: Chapter,
    story_state: StoryState,
) -> tuple[Chapter, ChapterQualityScores, int]:
    """Review and optionally refine a chapter outline from the Architect.

    Uses the generic quality refinement loop in review mode: skips creation,
    judges the provided chapter, and refines if below threshold.

    Args:
        svc: WorldQualityService instance.
        chapter: Existing Chapter to review.
        story_state: Current story state with brief and plot outline.

    Returns:
        Tuple of (chapter, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If refinement fails after all attempts.
    """
    logger.debug("Reviewing chapter quality for 'Ch%d: %s'", chapter.number, chapter.title)
    config = svc.get_config()

    return quality_refinement_loop(
        entity_type="chapter",
        create_fn=lambda retries: chapter,
        judge_fn=lambda ch: svc._judge_chapter_quality(
            ch,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda ch, scores, iteration: svc._refine_chapter_outline(
            ch,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda ch: f"Ch{ch.number}: {ch.title}",
        serialize=lambda ch: ch.model_dump(),
        is_empty=lambda ch: not ch.title and not ch.outline,
        score_cls=ChapterQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
        initial_entity=chapter,
    )


def _judge_chapter_quality(
    svc,
    chapter: Chapter,
    story_state: StoryState,
    temperature: float,
) -> ChapterQualityScores:
    """Judge chapter outline quality using the judge model.

    Supports multi-call averaging when judge_multi_call_enabled is True in settings.

    Args:
        svc: WorldQualityService instance.
        chapter: Chapter to evaluate.
        story_state: Current story state for context (includes plot outline).
        temperature: Judge temperature (low for consistency).

    Returns:
        ChapterQualityScores with ratings and feedback.

    Raises:
        WorldGenerationError: If quality judgment fails.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    # Get plot summary for context (chapter is evaluated within the story arc)
    plot_summary = ""
    if story_state.plot_summary:
        plot_summary = story_state.plot_summary

    prompt = _build_chapter_judge_prompt(chapter, genre, plot_summary)

    judge_model = svc._get_judge_model(entity_type="chapter")
    logger.debug(
        "Judging chapter quality for 'Ch%d: %s' (model=%s)",
        chapter.number,
        chapter.title,
        judge_model,
    )
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> ChapterQualityScores:
        """Execute a single judge call for chapter quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=ChapterQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Chapter quality judgment failed for 'Ch%d: %s': %s",
                    chapter.number,
                    chapter.title,
                    summary,
                )
            else:
                logger.error(
                    "Chapter quality judgment failed for 'Ch%d: %s': %s",
                    chapter.number,
                    chapter.title,
                    summary,
                )
            raise WorldGenerationError(f"Chapter quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, ChapterQualityScores, judge_config)


def _build_chapter_judge_prompt(
    chapter: Chapter,
    genre: str,
    plot_summary: str,
) -> str:
    """
    Constructs the text prompt used by the judge model to evaluate a chapter outline's quality.

    The prompt includes story-arc context, chapter metadata (number, title, outline), an optional scenes list, a calibration block, evaluation dimensions, and explicit output-format instructions required for the judge model.

    Parameters:
        chapter (Chapter): The chapter instance to evaluate (number, title, outline, optional scenes).
        genre (str): Story genre to provide contextual framing for the evaluation.
        plot_summary (str): Overall plot/arc summary to include as additional context; may be empty.

    Returns:
        str: A formatted prompt string ready to send to the judge model.
    """
    # Format scenes if present
    scenes_text = ""
    if chapter.scenes:
        scene_lines = []
        for scene in chapter.scenes:
            scene_lines.append(f"  - {scene.goal}" if scene.goal else f"  - Scene {scene.order}")
        scenes_text = "\nScenes:\n" + "\n".join(scene_lines)

    logger.debug(
        "Building chapter judge prompt for 'Ch%d: %s'",
        chapter.number,
        chapter.title,
    )

    return f"""You are a literary critic evaluating a chapter outline for a {genre} story.

STORY ARC CONTEXT:
{plot_summary if plot_summary else "Not available"}

CHAPTER TO EVALUATE:
Number: {chapter.number}
Title: {chapter.title}
Outline: {chapter.outline}
{scenes_text}

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- purpose: Advances plot and/or character development meaningfully
- pacing: Good distribution of action, dialogue, reflection
- hook: Opening grabs attention, ending compels continuation
- coherence: Internal consistency and logical flow

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"purpose": <float 0-10>, "pacing": <float 0-10>, "hook": <float 0-10>, "coherence": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""


def _build_dimension_instructions(weak: list[str]) -> str:
    """Build dimension-specific, actionable refinement instructions.

    Maps each weak dimension name to a concrete, actionable instruction that
    tells the LLM exactly what to fix rather than relying on generic guidance.

    Args:
        weak: List of dimension names that scored below threshold.

    Returns:
        Formatted instruction block for the weak dimensions, or a minor
        improvements note if no dimensions are weak.
    """
    dimension_guidance = {
        "purpose": (
            "PURPOSE: Clarify what this chapter accomplishes for the overall story arc. "
            "Add a specific character revelation, plot turning point, or thematic statement."
        ),
        "pacing": (
            "PACING: Vary the rhythm — if all beats are action, add reflection. "
            "If all beats are dialogue, add movement or tension. Balance tension and release."
        ),
        "hook": (
            "HOOK: Start with an immediate question or conflict, not setup. "
            "End on unresolved tension that compels the reader to continue."
        ),
        "coherence": (
            "COHERENCE: Ensure each scene leads logically to the next. "
            "Remove contradictions and tighten cause-and-effect connections."
        ),
    }

    if not weak:
        logger.debug("No weak dimensions found; requesting minor improvements only")
        return "No critical weaknesses — make minor improvements to polish the outline."

    instructions = []
    for dim in weak:
        instructions.append(f"- {dimension_guidance[dim]}")
    logger.debug("Built dimension instructions for weak areas: %s", weak)
    return "\n".join(instructions)


def _refine_chapter_outline(
    svc,
    chapter: Chapter,
    scores: ChapterQualityScores,
    story_state: StoryState,
    temperature: float,
) -> Chapter:
    """Refine a chapter outline based on quality feedback.

    Args:
        svc: WorldQualityService instance.
        chapter: Chapter to refine.
        scores: Quality scores with feedback.
        story_state: Current story state.
        temperature: Refinement temperature.

    Returns:
        Refined Chapter.
    """
    logger.debug("Refining chapter outline 'Ch%d: %s'", chapter.number, chapter.title)
    brief = story_state.brief
    weak = scores.weak_dimensions(svc.get_config().get_threshold("chapter"))

    # Get plot summary for context
    plot_summary = ""
    if story_state.plot_summary:
        plot_summary = story_state.plot_summary

    dimension_instructions = _build_dimension_instructions(weak)

    prompt = f"""Improve this chapter outline based on quality feedback.

STORY ARC CONTEXT:
{plot_summary if plot_summary else "Not available"}

ORIGINAL CHAPTER:
Number: {chapter.number}
Title: {chapter.title}
Outline: {chapter.outline}

QUALITY SCORES (0-10):
- Purpose: {scores.purpose}
- Pacing: {scores.pacing}
- Hook: {scores.hook}
- Coherence: {scores.coherence}

FEEDBACK: {scores.feedback}

DIMENSION-SPECIFIC INSTRUCTIONS:
{dimension_instructions}

IMPORTANT: Make SUBSTANTIAL changes to the outline — do NOT simply reword the original. \
Restructure scenes, add new story beats, change the dramatic arc, or introduce new elements. \
The refined outline must be meaningfully different from the original.

Keep the chapter number {chapter.number} and maintain consistency with the overall story arc.
Write all text in {brief.language if brief else "English"}."""

    try:
        model = svc._get_creator_model(entity_type="chapter")
        return generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=Chapter,
            temperature=temperature,
        )
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error(
            "Chapter refinement failed for 'Ch%d: %s': %s",
            chapter.number,
            chapter.title,
            summary,
        )
        raise WorldGenerationError(f"Chapter refinement failed: {summary}") from e
