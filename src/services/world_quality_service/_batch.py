"""Batch generation functions for multiple entities."""

import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from src.memory.story_state import Chapter, Character, StoryState
from src.memory.world_quality import (
    BaseQualityScores,
    ChapterQualityScores,
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RelationshipQualityScores,
)
from src.utils.exceptions import WorldGenerationError, summarize_llm_error

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S", bound=BaseQualityScores)


def _log_batch_summary(
    results: Sequence[tuple[Any, BaseQualityScores]],
    entity_type: str,
    quality_threshold: float,
    elapsed: float,
    get_name: Callable[[Any], str] | None = None,
) -> None:
    """Log an aggregate summary at the end of a batch generation or review.

    Args:
        results: List of (entity, scores) tuples produced by the batch.
        entity_type: Human-readable entity type (e.g. "character").
        quality_threshold: The configured quality threshold for pass/fail.
        elapsed: Total batch wall-clock time in seconds.
        get_name: Callable to extract display name from an entity. When provided,
            used for all entity types (chapters, relationships, etc.). Falls back
            to ``entity.get("name")`` / ``getattr(entity, "name")`` when ``None``.
    """
    if not results:
        logger.info(
            "Batch %s summary: 0 entities produced (%.1fs)",
            entity_type,
            elapsed,
        )
        return

    averages = [scores.average for _, scores in results]
    passed = sum(1 for avg in averages if round(avg, 1) >= quality_threshold)
    total = len(results)
    min_score = min(averages)
    max_score = max(averages)
    avg_score = sum(averages) / total

    failed_names: list[str] = []
    for entity, scores in results:
        if round(scores.average, 1) < quality_threshold:
            if get_name is not None:
                name = get_name(entity)
            elif isinstance(entity, dict):
                name = entity.get("name", "Unknown")
            else:
                name = getattr(entity, "name", "Unknown")
            failed_names.append(name)

    summary_parts = [
        f"passed={passed}/{total}",
        f"scores: min={min_score:.1f} max={max_score:.1f} avg={avg_score:.1f}",
        f"threshold={quality_threshold:.1f}",
        f"time={elapsed:.1f}s",
    ]
    if failed_names:
        summary_parts.append(f"below threshold: {', '.join(failed_names)}")

    logger.info(
        "Batch %s summary: %s",
        entity_type,
        ", ".join(summary_parts),
    )


def _aggregate_errors(errors: list[str]) -> str:
    """Deduplicate and aggregate identical error messages.

    Instead of repeating ``"Failed to generate relationship after 3 attempts"``
    nine times, produces ``"Failed to generate relationship after 3 attempts (x9)"``.

    Args:
        errors: List of raw error messages (may contain duplicates).

    Returns:
        A single joined string with counts appended for repeated messages.
    """
    counts = Counter(errors)
    parts: list[str] = []
    for msg, count in counts.items():
        if count > 1:
            parts.append(f"{msg} (x{count})")
        else:
            parts.append(msg)
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _generate_batch(
    svc,
    count: int,
    entity_type: str,
    generate_fn: Callable[[int], tuple[T, S, int]],
    get_name: Callable[[T], str],
    on_success: Callable[[T], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
    quality_threshold: float | None = None,
) -> list[tuple[T, S]]:
    """Generic batch generation loop for creating new entities.

    Args:
        svc: WorldQualityService instance (used for ``_calculate_eta``).
        count: Number of entities to generate.
        entity_type: Human-readable type name for logging/progress (e.g. "faction").
        generate_fn: Called with the loop index ``i``; must return
            ``(entity, scores, iterations)``.
        get_name: Extract a display name from the generated entity.
        on_success: Optional hook called after a successful generation (e.g. to
            track names for deduplication).
        cancel_check: Optional callable that returns ``True`` to cancel.
        progress_callback: Optional callback to receive
            :class:`EntityGenerationProgress` updates.
        quality_threshold: Quality threshold for pass/fail in batch summary log.
            Resolved via ``svc.get_config().get_threshold(entity_type)`` if not provided.

    Returns:
        List of ``(entity, scores)`` tuples.

    Raises:
        WorldGenerationError: If **no** entities could be generated.
    """
    from src.services.world_quality_service import EntityGenerationProgress

    # Resolve threshold for batch summary (raises ValueError if type missing)
    if quality_threshold is None:
        quality_threshold = svc.get_config().get_threshold(entity_type)

    results: list[tuple[T, S]] = []
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        if cancel_check and cancel_check():
            logger.info(
                "%s generation cancelled after %d/%d",
                entity_type.capitalize(),
                len(results),
                count,
            )
            break

        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type=entity_type,
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info("Generating %s %d/%d with quality refinement", entity_type, i + 1, count)
            entity, scores, iterations = generate_fn(i)
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((entity, scores))
            entity_name = get_name(entity)

            if on_success:
                on_success(entity)

            logger.info(
                "%s '%s' complete after %d iteration(s), quality: %.1f, generation_time: %.2fs",
                entity_type.capitalize(),
                entity_name,
                iterations,
                scores.average,
                entity_elapsed,
            )

            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type=entity_type,
                        entity_name=entity_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
        except WorldGenerationError as e:
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error("Failed to generate %s %d/%d: %s", entity_type, i + 1, count, error_msg)

    if not results and errors:
        raise WorldGenerationError(
            f"Failed to generate any {entity_type}s. Errors: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            "Generated %d/%d %ss. %d failed: %s",
            len(results),
            count,
            entity_type,
            len(errors),
            _aggregate_errors(errors),
        )

    _log_batch_summary(
        results, entity_type, quality_threshold, time.time() - batch_start_time, get_name=get_name
    )

    return results


def _review_batch(
    svc,
    entities: list[T],
    entity_type: str,
    review_fn: Callable[[T], tuple[T, S, int]],
    get_name: Callable[[T], str],
    zero_scores_fn: Callable[[str], S],
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
    quality_threshold: float | None = None,
) -> list[tuple[T, S]]:
    """Generic batch review loop for existing entities.

    Args:
        svc: WorldQualityService instance (used for ``_calculate_eta``).
        entities: Entities to review.
        entity_type: Human-readable type name for logging/progress (e.g. "character").
        review_fn: Called with each entity; must return
            ``(reviewed_entity, scores, iterations)``.
        get_name: Extract a display name from an entity (used for progress
            reports and logging).
        zero_scores_fn: Factory called with an error message to produce
            zero-valued scores when a review fails.
        cancel_check: Optional callable that returns ``True`` to cancel.
        progress_callback: Optional callback to receive
            :class:`EntityGenerationProgress` updates.
        quality_threshold: Quality threshold for pass/fail in batch summary log.
            Resolved via ``svc.get_config().get_threshold(entity_type)`` if not provided.

    Returns:
        List of ``(entity, scores)`` tuples.  On failure the *original*
        entity is kept with zero scores.
    """
    from src.services.world_quality_service import EntityGenerationProgress

    # Resolve threshold for batch summary (raises ValueError if type missing)
    if quality_threshold is None:
        quality_threshold = svc.get_config().get_threshold(entity_type)

    results: list[tuple[T, S]] = []
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []
    count = len(entities)

    for i, entity in enumerate(entities):
        if cancel_check and cancel_check():
            logger.info(
                "%s review cancelled after %d/%d",
                entity_type.capitalize(),
                len(results),
                count,
            )
            break

        entity_name = get_name(entity)

        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    phase="reviewing",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info("Reviewing %s %d/%d '%s' quality", entity_type, i + 1, count, entity_name)
            reviewed, scores, iterations = review_fn(entity)
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((reviewed, scores))
            reviewed_name = get_name(reviewed)
            logger.info(
                "%s '%s' review complete after %d iteration(s), quality: %.1f, review_time: %.2fs",
                entity_type.capitalize(),
                reviewed_name,
                iterations,
                scores.average,
                entity_elapsed,
            )

            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type=entity_type,
                        entity_name=reviewed_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
        except WorldGenerationError as e:
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            logger.error("Failed to review %s '%s': %s", entity_type, entity_name, error_msg)
            results.append((entity, zero_scores_fn(f"Review failed: {e}")))

    if errors:
        logger.warning(
            "Reviewed %d/%d %ss. %d failed: %s",
            len(results),
            count,
            entity_type,
            len(errors),
            _aggregate_errors(errors),
        )

    _log_batch_summary(
        results, entity_type, quality_threshold, time.time() - batch_start_time, get_name=get_name
    )

    return results


# ---------------------------------------------------------------------------
# Generate wrappers
# ---------------------------------------------------------------------------


def generate_factions_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    count: int = 2,
    existing_locations: list[str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], FactionQualityScores]]:
    """Generate multiple factions with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_names: Names to avoid.
        count: Number of factions to generate.
        existing_locations: Names of existing locations for spatial grounding.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (faction_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no factions could be generated.
    """
    names = existing_names.copy()
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="faction",
        generate_fn=lambda _i: svc.generate_faction_with_quality(
            story_state, names, existing_locations
        ),
        get_name=lambda f: f.get("name", "Unknown"),
        on_success=lambda f: names.append(f["name"]) if "name" in f else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_items_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    count: int = 3,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], ItemQualityScores]]:
    """Generate multiple items with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_names: Names to avoid.
        count: Number of items to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (item_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no items could be generated.
    """
    names = existing_names.copy()
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="item",
        generate_fn=lambda _i: svc.generate_item_with_quality(story_state, names),
        get_name=lambda item: item.get("name", "Unknown"),
        on_success=lambda item: names.append(item["name"]) if "name" in item else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_concepts_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    count: int = 2,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], ConceptQualityScores]]:
    """Generate multiple concepts with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_names: Names to avoid.
        count: Number of concepts to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (concept_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no concepts could be generated.
    """
    names = existing_names.copy()
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="concept",
        generate_fn=lambda _i: svc.generate_concept_with_quality(story_state, names),
        get_name=lambda c: c.get("name", "Unknown"),
        on_success=lambda c: names.append(c["name"]) if "name" in c else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_characters_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    count: int = 2,
    custom_instructions: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[Character, CharacterQualityScores]]:
    """Generate multiple characters with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_names: Names to avoid.
        count: Number of characters to generate.
        custom_instructions: Optional custom instructions to refine generation.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (Character, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no characters could be generated.
    """
    names = existing_names.copy()
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="character",
        generate_fn=lambda _i: svc.generate_character_with_quality(
            story_state, names, custom_instructions
        ),
        get_name=lambda char: char.name,
        on_success=lambda char: names.append(char.name),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_locations_with_quality(
    svc,
    story_state: StoryState,
    existing_names: list[str],
    count: int = 3,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], LocationQualityScores]]:
    """Generate multiple locations with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_names: Names to avoid.
        count: Number of locations to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (location_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no locations could be generated.
    """
    names = existing_names.copy()
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="location",
        generate_fn=lambda _i: svc.generate_location_with_quality(story_state, names),
        get_name=lambda loc: loc.get("name", "Unknown"),
        on_success=lambda loc: names.append(loc["name"]) if "name" in loc else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_relationships_with_quality(
    svc,
    story_state: StoryState,
    entity_names: list[str],
    existing_rels: list[tuple[str, str]],
    count: int = 5,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], RelationshipQualityScores]]:
    """Generate multiple relationships with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        entity_names: Entity names available for relationships.
        existing_rels: Existing relationships to avoid.
        count: Number of relationships to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (relationship_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no relationships could be generated.
    """
    rels = existing_rels.copy()
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="relationship",
        generate_fn=lambda _i: svc.generate_relationship_with_quality(
            story_state, entity_names, rels
        ),
        get_name=lambda r: f"{r['source']} -> {r['target']}",
        on_success=lambda r: rels.append((r["source"], r["target"])),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# Review wrappers
# ---------------------------------------------------------------------------


def review_characters_batch(
    svc,
    characters: list[Character],
    story_state: StoryState,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[Character, CharacterQualityScores]]:
    """Review multiple characters from the Architect with quality refinement.

    Each character is judged and refined if below the quality threshold.

    Args:
        svc: WorldQualityService instance.
        characters: List of Character objects to review.
        story_state: Current story state.
        cancel_check: Optional callable that returns True to cancel.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (reviewed_character, quality_scores) tuples.

    Raises:
        WorldGenerationError: If no characters could be reviewed.
    """
    return _review_batch(
        svc=svc,
        entities=characters,
        entity_type="character",
        review_fn=lambda char: svc.review_character_quality(char, story_state),
        get_name=lambda char: char.name,
        zero_scores_fn=lambda msg: CharacterQualityScores(
            depth=0, goals=0, flaws=0, uniqueness=0, arc_potential=0, feedback=msg
        ),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def review_chapters_batch(
    svc,
    chapters: list[Chapter],
    story_state: StoryState,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[Chapter, ChapterQualityScores]]:
    """Review multiple chapters from the Architect with quality refinement.

    Each chapter is judged and refined if below the quality threshold.

    Args:
        svc: WorldQualityService instance.
        chapters: List of Chapter objects to review.
        story_state: Current story state.
        cancel_check: Optional callable that returns True to cancel.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (reviewed_chapter, quality_scores) tuples.

    Raises:
        WorldGenerationError: If no chapters could be reviewed.
    """
    return _review_batch(
        svc=svc,
        entities=chapters,
        entity_type="chapter",
        review_fn=lambda ch: svc.review_chapter_quality(ch, story_state),
        get_name=lambda ch: f"Ch{ch.number}: {ch.title}",
        zero_scores_fn=lambda msg: ChapterQualityScores(
            purpose=0, pacing=0, hook=0, coherence=0, feedback=msg
        ),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
