"""Event generation, judgment, and refinement functions."""

import logging
from typing import Any

from src.memory.story_state import StoryState, WorldEventCreation
from src.memory.world_quality import EventQualityScores
from src.services.llm_client import generate_structured
from src.services.world_quality_service._common import (
    JUDGE_CALIBRATION_BLOCK,
    judge_with_averaging,
    retry_temperature,
)
from src.services.world_quality_service._quality_loop import quality_refinement_loop
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

logger = logging.getLogger(__name__)

# Truncation length for event description dedup — descriptions matching in the
# first N characters (case-insensitive) are considered duplicates.  Keep in sync
# with the ``get_name`` lambdas that truncate for logging/display.
_EVENT_DESCRIPTION_PREFIX_LEN = 60


def _format_participants(participants: list[Any]) -> str:
    """Format an event's participant list for prompt display.

    Args:
        participants: List of participant dicts or strings from an event.

    Returns:
        Formatted multi-line string, or empty string if no participants.
    """
    if not participants:
        return ""
    parts = []
    for p in participants:
        name = p.get("entity_name", "Unknown") if isinstance(p, dict) else str(p)
        role = p.get("role", "affected") if isinstance(p, dict) else "affected"
        parts.append(f"  - {name} ({role})")
    logger.debug("Formatted %d participants for prompt display", len(parts))
    return "Participants:\n" + "\n".join(parts)


def _format_consequences(consequences: list[Any]) -> str:
    """Format an event's consequence list for prompt display.

    Args:
        consequences: List of consequence strings from an event.

    Returns:
        Formatted multi-line string, or empty string if no consequences.
    """
    if not consequences:
        return ""
    logger.debug("Formatted %d consequences for prompt display", len(consequences))
    return "Consequences:\n" + "\n".join(f"  - {c}" for c in consequences)


def _is_duplicate_description(description: str, existing: list[str]) -> bool:
    """Check if an event description is a near-duplicate of an existing one.

    Uses case-insensitive truncated comparison (first ``_EVENT_DESCRIPTION_PREFIX_LEN``
    chars) since event descriptions are identified by their truncated form in
    the quality loop.

    Args:
        description: The new event description to check.
        existing: List of existing event descriptions.

    Returns:
        True if the description is a duplicate.
    """
    n = _EVENT_DESCRIPTION_PREFIX_LEN
    normalized = description.strip().casefold()[:n]
    return any(d.strip().casefold()[:n] == normalized for d in existing)


def generate_event_with_quality(
    svc,
    story_state: StoryState,
    existing_descriptions: list[str],
    entity_context: str,
    rejected_descriptions: list[str] | None = None,
) -> tuple[dict[str, Any], EventQualityScores, int]:
    """Generate a world event with iterative quality refinement.

    Uses the generic quality refinement loop to create, judge, and refine
    events until the quality threshold is met or stopping criteria is reached.

    Parameters:
        svc: WorldQualityService instance.
        story_state: Current story state with brief.
        existing_descriptions: Descriptions of existing events to avoid duplicates.
        entity_context: Pre-formatted string of all entities/relationships/lifecycle data.
        rejected_descriptions: Descriptions rejected as duplicates within the current
            batch, fed back into the creator prompt to avoid regeneration.

    Returns:
        Tuple of (event_dict, quality_scores, iterations_used).

    Raises:
        WorldGenerationError: If no valid event could be produced after all attempts.
    """
    config = svc.get_config()
    brief = story_state.brief
    if not brief:
        raise ValueError("Story must have a brief for event generation")

    if rejected_descriptions is None:
        rejected_descriptions = []

    # Combine existing + rejected for dedup checking and prompt injection
    all_known = existing_descriptions + rejected_descriptions

    prep_creator, prep_judge = svc._make_model_preparers("event")

    def _is_empty(evt: dict[str, Any]) -> bool:
        """Check if event has no description or is a duplicate of an already-known event.

        Returns True for events with empty/whitespace-only descriptions (no side
        effect) or for duplicates of known events.

        Side effect: appends duplicate descriptions to ``rejected_descriptions``
        and ``all_known`` so subsequent creator prompts and dedup checks within
        the same quality-loop invocation avoid regenerating them.
        """
        raw_desc = evt.get("description", "")
        desc = raw_desc.strip() if isinstance(raw_desc, str) else ""
        if not desc:
            return True
        if _is_duplicate_description(desc, all_known):
            logger.warning(
                "Generated duplicate event description '%s', rejecting",
                desc[:_EVENT_DESCRIPTION_PREFIX_LEN],
            )
            rejected_descriptions.append(desc)
            all_known.append(desc)  # Keep snapshot in sync for within-invocation dedup
            return True
        return False

    return quality_refinement_loop(
        entity_type="event",
        create_fn=lambda retries: svc._create_event(
            story_state,
            all_known,
            entity_context,
            retry_temperature(config, retries),
        ),
        judge_fn=lambda evt: svc._judge_event_quality(
            evt,
            story_state,
            config.judge_temperature,
        ),
        refine_fn=lambda evt, scores, iteration: svc._refine_event(
            evt,
            scores,
            story_state,
            config.get_refinement_temperature(iteration),
        ),
        get_name=lambda evt: evt.get("description", "Unknown")[:_EVENT_DESCRIPTION_PREFIX_LEN],
        serialize=lambda evt: evt.copy(),
        is_empty=_is_empty,
        score_cls=EventQualityScores,
        config=config,
        svc=svc,
        story_id=story_state.id,
        prepare_creator=prep_creator,
        prepare_judge=prep_judge,
    )


def _create_event(
    svc,
    story_state: StoryState,
    existing_descriptions: list[str],
    entity_context: str,
    temperature: float,
) -> dict[str, Any]:
    """Create a new world event using the creator model with structured generation."""
    logger.debug(
        "Creating event for story %s (existing: %d)",
        story_state.id,
        len(existing_descriptions),
    )
    brief = story_state.brief
    if not brief:
        logger.error("_create_event called without story brief for story %s", story_state.id)
        raise ValueError("Story must have a brief for event creation")

    # Format existing events for dedup
    existing_block = ""
    if existing_descriptions:
        existing_list = "\n".join(f"- {d}" for d in existing_descriptions)
        existing_block = f"""
=== EXISTING EVENTS (DO NOT DUPLICATE) ===
{existing_list}

Create something COMPLETELY DIFFERENT from the above events.
"""

    calendar_context = svc.get_calendar_context()

    prompt = f"""Create a world-shaping event for a {brief.genre} story.

STORY PREMISE: {brief.premise}
SETTING: {brief.setting_place}, {brief.setting_time}
TONE: {brief.tone}
{calendar_context}
{existing_block}
WORLD CONTEXT (entities, relationships, timeline):
{entity_context}

Create a significant world event with:
1. A clear, specific description of what happened
2. When it occurred (year and/or era from the calendar if available)
3. Which existing entities participated and in what roles (actor, location, affected, witness)
4. What consequences followed from this event
5. How it shaped the world and created story opportunities

The event should be a pivotal moment in the world's history that connects
multiple entities and creates narrative tension.

Write all text in {brief.language}."""

    try:
        model = svc._get_creator_model(entity_type="event")
        event = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=WorldEventCreation,
            temperature=temperature,
        )

        if not event.description:
            logger.warning("Event creation returned empty description, clearing to force retry")
            return {}

        return event.model_dump()
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error("Event creation failed for story %s: %s", story_state.id, summary)
        raise WorldGenerationError(f"Event creation failed: {summary}") from e


def _judge_event_quality(
    svc,
    event: dict[str, Any],
    story_state: StoryState,
    temperature: float,
) -> EventQualityScores:
    """Evaluate a generated event's quality across the defined dimensions.

    Parameters:
        svc: Service client providing model resolution and settings.
        event: Event data dict (description, year, era_name, participants, consequences).
        story_state: Story context used to determine genre and language.
        temperature: Sampling temperature to use with the judge model.

    Returns:
        EventQualityScores with numeric scores for each dimension and feedback.

    Raises:
        WorldGenerationError: If the judge model call fails or returns invalid data.
    """
    brief = story_state.brief
    genre = brief.genre if brief else "fiction"

    calendar_context = svc.get_calendar_context()
    participants_str = _format_participants(event.get("participants", []))
    consequences_str = _format_consequences(event.get("consequences", []))

    prompt = f"""You are evaluating a world event for a {genre} story.

EVENT TO EVALUATE:
Description: {event.get("description", "Unknown")}
Year: {event.get("year", "N/A")}
Era: {event.get("era_name", "N/A")}
{participants_str}
{consequences_str}
{calendar_context}
{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- significance: How world-shaping is this event? Does it change the status quo?
- temporal_plausibility: VERIFY against CALENDAR above — event year MUST fall within a defined era's [start_year, end_year] range. Score 8-10 ONLY if the event's year is inside a valid era AND the event type makes sense for that era. Score 5-7 if timing is plausible but era alignment is unclear. Score 2-4 if the event year falls outside all defined eras or contradicts the calendar. If the event year is "N/A" (not yet assigned), score 5.0 (neutral — date is pending, not wrong). If the CALENDAR block states "No calendar available", score 5.0 (insufficient context to verify).
- causal_coherence: Are causes and consequences logically connected?
- narrative_potential: Does it create story opportunities and tension?
- entity_integration: Do participant roles make sense for their entity types?

Provide specific improvement feedback in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"significance": <float 0-10>, "temporal_plausibility": <float 0-10>, "causal_coherence": <float 0-10>, "narrative_potential": <float 0-10>, "entity_integration": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

    judge_model = svc._get_judge_model(entity_type="event")
    judge_config = svc.get_judge_config()
    multi_call = judge_config.enabled and judge_config.multi_call_enabled

    def _single_judge_call() -> EventQualityScores:
        """Execute a single judge call for event quality."""
        try:
            return generate_structured(
                settings=svc.settings,
                model=judge_model,
                prompt=prompt,
                response_model=EventQualityScores,
                temperature=temperature,
            )
        except Exception as e:
            summary = summarize_llm_error(e)
            if multi_call:
                logger.warning(
                    "Event quality judgment failed for '%s': %s",
                    event.get("description", "Unknown")[:_EVENT_DESCRIPTION_PREFIX_LEN],
                    summary,
                )
            else:
                logger.error(
                    "Event quality judgment failed for '%s': %s",
                    event.get("description", "Unknown")[:_EVENT_DESCRIPTION_PREFIX_LEN],
                    summary,
                )
            raise WorldGenerationError(f"Event quality judgment failed: {summary}") from e

    return judge_with_averaging(_single_judge_call, EventQualityScores, judge_config)


def _refine_event(
    svc,
    event: dict[str, Any],
    scores: EventQualityScores,
    story_state: StoryState,
    temperature: float,
) -> dict[str, Any]:
    """Refine an event based on quality feedback using structured generation."""
    logger.debug(
        "Refining event '%s' for story %s",
        event.get("description", "Unknown")[:_EVENT_DESCRIPTION_PREFIX_LEN],
        story_state.id,
    )
    brief = story_state.brief

    threshold = svc.get_config().get_threshold("event")
    improvement_focus = []
    if scores.significance < threshold:
        improvement_focus.append("Make the event more world-shaping and consequential")
    if scores.temporal_plausibility < threshold:
        improvement_focus.append("Improve timeline placement and era consistency")
    if scores.causal_coherence < threshold:
        improvement_focus.append("Strengthen the logical chain of causes and consequences")
    if scores.narrative_potential < threshold:
        improvement_focus.append("Create more story opportunities and dramatic tension")
    if scores.entity_integration < threshold:
        improvement_focus.append("Better integrate participants with meaningful roles")

    participants_str = _format_participants(event.get("participants", []))
    consequences_str = _format_consequences(event.get("consequences", []))

    prompt = f"""TASK: Improve this world event to score HIGHER on the weak dimensions.

ORIGINAL EVENT:
Description: {event.get("description", "Unknown")}
Year: {event.get("year", "N/A")}
Era: {event.get("era_name", "N/A")}
{participants_str}
{consequences_str}

CURRENT SCORES (need {threshold}+ in all areas):
- Significance: {scores.significance}/10
- Temporal Plausibility: {scores.temporal_plausibility}/10
- Causal Coherence: {scores.causal_coherence}/10
- Narrative Potential: {scores.narrative_potential}/10
- Entity Integration: {scores.entity_integration}/10

JUDGE'S FEEDBACK: {scores.feedback}

SPECIFIC IMPROVEMENTS NEEDED:
{"\n".join(f"- {imp}" for imp in improvement_focus) if improvement_focus else "- Enhance all areas"}

REQUIREMENTS:
1. Keep the core event concept but make SUBSTANTIAL improvements
2. Preserve existing participants and add more if needed
3. Strengthen causes and consequences
4. Output in {brief.language if brief else "English"}

Return ONLY the improved event."""

    try:
        model = svc._get_creator_model(entity_type="event")
        refined = generate_structured(
            settings=svc.settings,
            model=model,
            prompt=prompt,
            response_model=WorldEventCreation,
            temperature=temperature,
        )

        result = refined.model_dump()
        # Preserve temporal fields from original if refined version drops them.
        # LLMs frequently omit unchanged fields during refinement, so we
        # backfill from the original to prevent data loss.
        for key in ("year", "month", "era_name"):
            if result.get(key) in (None, "") and event.get(key) not in (None, ""):
                result[key] = event[key]
                logger.warning("Preserved temporal field '%s' from original event", key)
        # Preserve participants from original if refined version drops them
        if not result.get("participants") and event.get("participants"):
            result["participants"] = event["participants"]
            logger.warning("Preserved participants from original event")
        return result
    except Exception as e:
        summary = summarize_llm_error(e)
        logger.error(
            "Event refinement failed for '%s': %s",
            event.get("description", "Unknown")[:_EVENT_DESCRIPTION_PREFIX_LEN],
            summary,
        )
        raise WorldGenerationError(f"Event refinement failed: {summary}") from e
