"""Calendar generation, judgment, and refinement functions."""

import logging
from typing import Any

from src.memory.story_state import StoryState
from src.memory.world_calendar import (
    CalendarMonth,
    HistoricalEra,
    WorldCalendar,
)
from src.memory.world_quality import CalendarQualityScores
from src.services.calendar_service import GeneratedCalendarData
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
    retry_temperature,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

logger = logging.getLogger(__name__)


def generate_calendar_with_quality(
    svc,
    story_state: StoryState,
) -> tuple[dict[str, Any], CalendarQualityScores, int]:
    """Generate a calendar with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    a calendar system until the quality threshold is met or stopping criteria
    is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.

    Returns:
        Tuple of (calendar_dict, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If no valid calendar could be produced after all attempts.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for calendar generation")

    return quality_refinement_loop(
        entity_type="calendar",
        create_fn=lambda retries: _create_calendar(
            svc,
            story_state,
            retry_temperature(config, retries),
        ),
        judge_fn=lambda cal: _judge_calendar_quality(
            svc,
            cal,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda cal, scores, iteration: _refine_calendar(
            svc,
            cal,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda c: c.get("current_era_name", "Unknown"),
        serialize=lambda c: c.copy(),
        is_empty=lambda c: not c.get("months"),
        score_cls=CalendarQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
    )


def _create_calendar(
    svc,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Create a new calendar system using the creator model with structured generation.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        temperature: Sampling temperature for the LLM call.

    Returns:
        Calendar dict (via WorldCalendar.to_dict()).

    Raises:
        WorldGenerationError: If the LLM call fails or story has no brief.
    """
    logger.debug("Creating calendar for story %s", story_state.id)
    brief = story_state.brief
    if not brief:
        raise WorldGenerationError("Cannot create calendar: story has no brief")

    prompt = f"""Design a fictional calendar system for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
THEMES: {", ".join(brief.themes) if brief.themes else "None"}

Create a calendar that:
1. Has 8-14 months with evocative names fitting the setting
2. Includes 5-8 day names for the week
3. Defines 2-5 historical eras with the current era being the most recent
4. Sets an appropriate "current year" for where the story takes place
5. Uses a date format that feels natural for this culture

The calendar should feel authentic to this world - not just a renamed version of our calendar.

For each month, provide:
- name: An evocative name (e.g., "Frostfall", "Highsun", "Reaping Moon")
- days: Number of days (20-40 range works well)
- description: Brief flavor text about this month's nature

For historical eras, provide:
- name: Era name (e.g., "Age of Dragons", "The Long Winter", "Era of Expansion")
- start_year: When this era began
- end_year: When this era ended (null for current era)
- description: What characterized this era

Make the era_abbreviation 2-3 letters that could follow a year number.

Write all text in {brief.language}."""

    try:
        model = svc._get_creator_model(entity_type="calendar")
        result = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=GeneratedCalendarData,
            temperature=temperature,
        )

        # Convert GeneratedCalendarData to WorldCalendar then to dict
        calendar = _generated_data_to_world_calendar(result)
        calendar_dict = calendar.to_dict()
        logger.info(
            "Created calendar: %s (%s), %d months, %d eras",
            result.era_name,
            result.era_abbreviation,
            len(result.months),
            len(result.historical_eras),
        )
        return calendar_dict
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error("Calendar creation failed for story %s: %s", story_state.id, summary)
        raise WorldGenerationError(f"Calendar creation failed: {summary}") from e


def _generated_data_to_world_calendar(result: GeneratedCalendarData) -> WorldCalendar:
    """Convert a GeneratedCalendarData Pydantic model to a WorldCalendar.

    Resolves ``era_start_year`` from the era list using a multi-step fallback:
    (1) exact name match on ``result.era_name``, (2) case-insensitive match,
    (3) ongoing era (``end_year is None``), (4) last era's ``start_year``.

    Parameters:
        result: Structured LLM response with calendar data.

    Returns:
        WorldCalendar instance.
    """
    # Note: LLM outputs may have missing fields, so we use fallbacks here.
    # This is intentional for LLM parsing (unlike config values which must be explicit).
    months = []
    for i, m in enumerate(result.months):
        name = m.get("name") or f"Month {i + 1}"
        days = m.get("days")
        description = m.get("description") or ""
        if name != m.get("name"):
            logger.debug("Month %d: using fallback name '%s'", i + 1, name)
        if days is None:
            days = 30
            logger.warning(
                "Month %d ('%s'): missing 'days' field, defaulting to %d", i + 1, name, days
            )
        months.append(CalendarMonth(name=name, days=days, description=description))

    eras = []
    for i, e in enumerate(result.historical_eras):
        name = e.get("name") or f"Era {i + 1}"
        start_year = e.get("start_year")
        end_year = e.get("end_year")  # None is valid for ongoing era
        description = e.get("description") or ""
        if name != e.get("name"):
            logger.debug("Era %d: using fallback name '%s'", i + 1, name)
        if start_year is None:
            start_year = 1
            logger.warning(
                "Era %d ('%s'): missing 'start_year' field, defaulting to %d",
                i + 1,
                name,
                start_year,
            )
        eras.append(
            HistoricalEra(
                name=name,
                start_year=start_year,
                end_year=end_year,
                description=description,
                display_order=i,
            )
        )

    # Match era_start_year to the era matching current_era_name for consistency
    era_start_year = 1
    if eras:
        # Try exact match first
        current_era_obj = next((era for era in eras if era.name == result.era_name), None)
        # Try case-insensitive match
        if not current_era_obj:
            current_era_obj = next(
                (era for era in eras if era.name.lower() == result.era_name.lower()),
                None,
            )
            if current_era_obj:
                logger.debug(
                    "Matched current era '%s' via case-insensitive lookup to '%s'",
                    result.era_name,
                    current_era_obj.name,
                )
        # Fall back to the ongoing era (end_year=None = current era)
        if not current_era_obj:
            current_era_obj = next((era for era in eras if era.end_year is None), None)
            if current_era_obj:
                logger.warning(
                    "Current era '%s' not found by name; using ongoing era '%s'",
                    result.era_name,
                    current_era_obj.name,
                )
        if current_era_obj:
            era_start_year = current_era_obj.start_year
        else:
            logger.warning(
                "Current era '%s' not found in historical eras and no ongoing era; "
                "falling back to last era's start year.",
                result.era_name,
            )
            era_start_year = eras[-1].start_year
    else:
        logger.warning("Calendar has no eras — using fallback era_start_year=1")

    return WorldCalendar(
        current_era_name=result.era_name,
        era_abbreviation=result.era_abbreviation,
        era_start_year=era_start_year,
        months=months,
        days_per_week=len(result.day_names),
        day_names=result.day_names,
        current_story_year=result.current_year,
        eras=eras,
        date_format=result.date_format,
    )


def _judge_calendar_quality(
    svc,
    calendar: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> CalendarQualityScores:
    """Evaluate a generated calendar's quality across defined dimensions.

    Uses the configured judge model to produce numeric scores for
    internal_consistency, thematic_fit, completeness, and uniqueness.

    Parameters:
        svc: WorldQualityService instance.
        calendar: Calendar dict data.
        story_state: Story context for genre/setting reference.
        temperature: Sampling temperature for the judge model.

    Returns:
        CalendarQualityScores with scores and feedback.

    Raises:
        WorldGenerationError: If the judge model call fails.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    logger.debug(
        "Judging calendar quality: svc=%s, genre=%s, era=%s, months=%d, temperature=%.2f",
        type(svc).__name__,
        genre,
        calendar.get("current_era_name", "Unknown"),
        len(calendar.get("months", [])),
        temperature,
    )

    # Format months for display
    months_text = ""
    months_list = calendar.get("months", [])
    for i, m in enumerate(months_list[:5]):  # Show first 5 months
        name = m.get("name", f"Month {i + 1}") if isinstance(m, dict) else str(m)
        days = m.get("days", "?") if isinstance(m, dict) else "?"
        months_text += f"  - {name} ({days} days)\n"
    if len(months_list) > 5:
        months_text += f"  ... and {len(months_list) - 5} more months\n"

    # Format eras for display
    eras_text = ""
    eras_list = calendar.get("eras", [])
    for e in eras_list:
        if isinstance(e, dict):
            name = e.get("name", "Unknown Era")
            start = e.get("start_year", "?")
            end = e.get("end_year")
            end = end if end is not None else "present"
            eras_text += f"  - {name}: {start} - {end}\n"

    prompt = f"""You are evaluating a calendar system for a {genre} story.

CALENDAR TO EVALUATE:
Era Name: {calendar.get("current_era_name", "Unknown")}
Era Abbreviation: {calendar.get("era_abbreviation", "?")}
Current Year: {calendar.get("current_story_year", "?")}
Months ({len(months_list)}):
{months_text}
Historical Eras:
{eras_text}
Days per Week: {calendar.get("days_per_week", "?")}

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- internal_consistency: Does the calendar system make internal sense? Are months/eras coherent?
- thematic_fit: How well does it match the story's genre, setting, and tone?
- completeness: Are months, eras, and day names thorough and well-developed?
- uniqueness: Is it distinctive from real-world or generic fantasy calendars?

Provide specific improvement feedback in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"internal_consistency": <float 0-10>, "thematic_fit": <float 0-10>, "completeness": <float 0-10>, "uniqueness": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    judge_model = svc._get_judge_model(entity_type="calendar")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> CalendarQualityScores:
        """Execute a single judge call for calendar quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=CalendarQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Calendar quality judgment failed for '%s': %s",
                    calendar.get("current_era_name", "Unknown"),
                    summary,
                )
            else:
                logger.error(
                    "Calendar quality judgment failed for '%s': %s",
                    calendar.get("current_era_name", "Unknown"),
                    summary,
                )
            raise WorldGenerationError(f"Calendar quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, CalendarQualityScores, judge_config)


def _refine_calendar(
    svc,
    calendar: dict[str, Any],
    scores: CalendarQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine a calendar based on quality feedback using structured generation.

    Parameters:
        svc: WorldQualityService instance.
        calendar: Calendar dict to refine.
        scores: Quality scores from the judge.
        story_state: Story context.
        temperature: Sampling temperature for the refinement call.

    Returns:
        Refined calendar dict.

    Raises:
        WorldGenerationError: If refinement fails.
    """
    logger.debug(
        "Refining calendar '%s' for story %s",
        calendar.get("current_era_name", "Unknown"),
        story_state.id,
    )
    brief = story_state.brief

    # Build specific improvement instructions from weak dimensions
    threshold = svc.get_config().get_threshold("calendar")
    improvement_focus = []
    for dim in scores.weak_dimensions(threshold):
        if dim == "internal_consistency":
            improvement_focus.append("Improve internal coherence of months, eras, and day names")
        elif dim == "thematic_fit":
            improvement_focus.append("Better align calendar with the story's genre and setting")
        elif dim == "completeness":
            improvement_focus.append("Add more detail to months, eras, and day descriptions")
        elif dim == "uniqueness":
            improvement_focus.append("Make the calendar more distinctive and original")

    # Format current months
    months_text = ""
    months_list = calendar.get("months", [])
    for i, m in enumerate(months_list):
        if isinstance(m, dict):
            months_text += f"  {i + 1}. {m.get('name', '?')} ({m.get('days', '?')} days)\n"

    prompt = f"""TASK: Improve this calendar system to score HIGHER on the weak dimensions.

ORIGINAL CALENDAR:
Era Name: {calendar.get("current_era_name", "Unknown")}
Era Abbreviation: {calendar.get("era_abbreviation", "?")}
Current Year: {calendar.get("current_story_year", "?")}
Months:
{months_text}

CURRENT SCORES (need {threshold}+ in all areas):
- Internal Consistency: {scores.internal_consistency}/10
- Thematic Fit: {scores.thematic_fit}/10
- Completeness: {scores.completeness}/10
- Uniqueness: {scores.uniqueness}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the exact era name: "{calendar.get("current_era_name", "Unknown")}"
2. Make SUBSTANTIAL improvements to weak areas
3. Maintain 8-14 months and 2-5 historical eras
4. Output in {brief.language if brief else "English"}

Return ONLY the improved calendar."""

    try:
        model = svc._get_creator_model(entity_type="calendar")
        refined_data = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=GeneratedCalendarData,
            temperature=temperature,
        )

        # Convert to WorldCalendar dict, preserving era name
        refined_calendar = _generated_data_to_world_calendar(refined_data)
        result = refined_calendar.to_dict()
        # Preserve the original era name identity
        original_era = calendar.get("current_era_name")
        if not original_era:
            logger.warning("Original calendar dict missing 'current_era_name' — using 'Unknown'")
            original_era = "Unknown"
        result["current_era_name"] = original_era
        return result
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error(
            "Calendar refinement failed for '%s': %s",
            calendar.get("current_era_name", "Unknown"),
            summary,
        )
        raise WorldGenerationError(f"Calendar refinement failed: {summary}") from e
