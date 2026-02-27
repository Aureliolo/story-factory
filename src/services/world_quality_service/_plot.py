"""Plot outline quality review, judgment, and refinement functions."""

import logging

from src.memory.story_state import PlotOutline, StoryState
from src.memory.world_quality import PlotQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

logger = logging.getLogger(__name__)


def review_plot_quality(
    svc,
    plot_outline: PlotOutline,
    story_state: StoryState,
) -> tuple[PlotOutline, PlotQualityScores, int]:
    """Review and optionally refine a plot outline from the Architect.

    Uses the generic quality refinement loop in review mode: skips creation,
    judges the provided plot outline, and refines if below threshold.

    Args:
        svc: WorldQualityService instance.
        plot_outline: Existing PlotOutline to review.
        story_state: Current story state with brief.

    Returns:
        Tuple of (plot_outline, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If refinement fails after all attempts.
    """
    logger.debug("Reviewing plot quality for story %s", story_state.id)
    config = svc.get_config()
    prep_creator, prep_judge = svc._make_model_preparers("plot")

    return quality_refinement_loop(
        entity_type="plot",
        create_fn=lambda retries: plot_outline,
        judge_fn=lambda plot: svc._judge_plot_quality(
            plot,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda plot, scores, iteration: svc._refine_plot(
            plot,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda plot: (
            plot.plot_summary[:50] + "..." if len(plot.plot_summary) > 50 else plot.plot_summary
        ),
        serialize=lambda plot: plot.model_dump(),
        is_empty=lambda plot: not plot.plot_summary and not plot.plot_points,
        score_cls=PlotQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
        initial_entity=plot_outline,
        prepare_creator=prep_creator,
        prepare_judge=prep_judge,
    )


def _judge_plot_quality(
    svc,
    plot_outline: PlotOutline,
    story_state: StoryState,
    temperature: float,
) -> PlotQualityScores:
    """Judge plot outline quality using the judge model.

    Supports multi-call averaging when judge_multi_call_enabled is True in settings.

    Args:
        svc: WorldQualityService instance.
        plot_outline: PlotOutline to evaluate.
        story_state: Current story state for context.
        temperature: Judge temperature (low for consistency).

    Returns:
        PlotQualityScores with ratings and feedback.

    Raises:
        WorldGenerationError: If quality judgment fails.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    prompt = _build_plot_judge_prompt(plot_outline, genre, brief.themes if brief else [])

    judge_model = svc._get_judge_model(entity_type="plot")
    logger.debug("Judging plot quality for story %s (model=%s)", story_state.id, judge_model)
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> PlotQualityScores:
        """Execute a single judge call for plot quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=PlotQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning("Plot quality judgment failed: %s", summary)
            else:
                logger.error("Plot quality judgment failed: %s", summary)
            raise WorldGenerationError(f"Plot quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, PlotQualityScores, judge_config)


def _build_plot_judge_prompt(
    plot_outline: PlotOutline,
    genre: str,
    themes: list[str],
) -> str:
    """
    Constructs the prompt text used by the judge model to evaluate a plot outline's quality.

    The prompt includes the outline summary, formatted plot points with optional chapter annotations, themes, a calibration block, rating criteria for coherence, tension arc, character integration, and originality, and explicit instructions for the judge's JSON-only output format.

    Parameters:
        plot_outline (PlotOutline): The plot outline to evaluate.
        genre (str): Story genre to provide contextual framing.
        themes (list[str]): Story themes to include in the prompt.

    Returns:
        prompt (str): A formatted prompt string ready to be sent to the judge model.
    """
    # Format plot points with chapter assignments
    plot_points_text = []
    for i, pp in enumerate(plot_outline.plot_points, 1):
        chapter_info = f" (Chapter {pp.chapter})" if pp.chapter is not None else ""
        plot_points_text.append(f"  {i}. {pp.description}{chapter_info}")
    plot_points_formatted = (
        "\n".join(plot_points_text) if plot_points_text else "  (no plot points)"
    )

    logger.debug(
        "Building plot judge prompt (%d plot points)",
        len(plot_outline.plot_points),
    )

    return f"""You are a literary critic evaluating a plot outline for a {genre} story.

PLOT OUTLINE TO EVALUATE:
Summary: {plot_outline.plot_summary}

Plot Points:
{plot_points_formatted}

Themes: {", ".join(themes) if themes else "Not specified"}

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- coherence: Logical progression from inciting incident to resolution
- tension_arc: Stakes escalate, tension builds and releases properly
- character_integration: Plot events meaningfully advance character arcs
- originality: Avoids predictable or cliched progressions

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"coherence": <float 0-10>, "tension_arc": <float 0-10>, "character_integration": <float 0-10>, "originality": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""


def _refine_plot(
    svc,
    plot_outline: PlotOutline,
    scores: PlotQualityScores,
    story_state: StoryState,
    temperature: float,
) -> PlotOutline:
    """Refine a plot outline based on quality feedback.

    Args:
        svc: WorldQualityService instance.
        plot_outline: PlotOutline to refine.
        scores: Quality scores with feedback.
        story_state: Current story state.
        temperature: Refinement temperature.

    Returns:
        Refined PlotOutline.
    """
    logger.debug("Refining plot outline for story %s", story_state.id)
    brief = story_state.brief
    weak = scores.weak_dimensions(svc.get_config().get_threshold("plot"))

    # Format plot points for the prompt
    plot_points_text = []
    for i, pp in enumerate(plot_outline.plot_points, 1):
        chapter_info = f" (Chapter {pp.chapter})" if pp.chapter is not None else ""
        plot_points_text.append(f"  {i}. {pp.description}{chapter_info}")
    plot_points_formatted = (
        "\n".join(plot_points_text) if plot_points_text else "  (no plot points)"
    )

    prompt = f"""Improve this plot outline based on quality feedback.

ORIGINAL PLOT OUTLINE:
Summary: {plot_outline.plot_summary}

Plot Points:
{plot_points_formatted}

QUALITY SCORES (0-10):
- Coherence: {scores.coherence}
- Tension Arc: {scores.tension_arc}
- Character Integration: {scores.character_integration}
- Originality: {scores.originality}

FEEDBACK: {scores.feedback}

WEAK AREAS TO IMPROVE: {", ".join(weak) if weak else "None - minor improvements only"}

Enhance the weak areas while maintaining the core story direction.
Keep the same number of plot points but improve their quality and progression.
Write all text in {brief.language if brief else "English"}."""

    try:
        model = svc._get_creator_model(entity_type="plot")
        return generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=PlotOutline,
            temperature=temperature,
        )
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error("Plot refinement failed: %s", summary)
        raise WorldGenerationError(f"Plot refinement failed: {summary}") from e
