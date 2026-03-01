"""Batch generation functions for multiple entities."""

import logging
import time
from collections.abc import Callable
from typing import Any

from src.memory.story_state import Chapter, Character, StoryState
from src.memory.world_quality import (
    BaseQualityScores,
    ChapterQualityScores,
    CharacterQualityScores,
    ConceptQualityScores,
    EventQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RelationshipQualityScores,
)
from src.services.world_quality_service._event import _EVENT_DESCRIPTION_PREFIX_LEN
from src.services.world_quality_service._formatting import aggregate_errors as _aggregate_errors
from src.services.world_quality_service._formatting import log_batch_summary as _log_batch_summary
from src.utils.exceptions import (
    DuplicateNameError,
    VRAMAllocationError,
    WorldGenerationError,
    summarize_llm_error,
)

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_BATCH_FAILURES = 3
MAX_BATCH_SHUFFLE_RETRIES = 1


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _generate_batch[T, S: BaseQualityScores](
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
        entity_type: Human-readable type name for logging/progress (e.g., "faction").
        generate_fn: Called with the loop index ``i``; must return
            ``(entity, scores, iterations)``.
        get_name: Extract a display name from the generated entity.
        on_success: Optional hook called after a successful generation (e.g., to
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
    consecutive_failures = 0
    shuffles_remaining = MAX_BATCH_SHUFFLE_RETRIES

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
            entity_name = get_name(entity)

            if on_success:
                on_success(entity)

            # Only count as success after get_name and on_success pass
            completed_times.append(entity_elapsed)
            results.append((entity, scores))
            consecutive_failures = 0

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
        except DuplicateNameError as e:
            # Don't count duplicates as consecutive failures —
            # they're expected outcomes when entity-pair slots fill up.
            duplicate_msg = summarize_llm_error(e, max_length=200)
            errors.append(duplicate_msg)
            logger.warning("Duplicate %s %d/%d: %s", entity_type, i + 1, count, duplicate_msg)
        except WorldGenerationError as e:
            # VRAMAllocationError is non-retryable — stop the entire batch
            if isinstance(e.__cause__, VRAMAllocationError):
                raise
            error_msg = summarize_llm_error(e, max_length=200)
            errors.append(error_msg)
            consecutive_failures += 1
            logger.error("Failed to generate %s %d/%d: %s", entity_type, i + 1, count, error_msg)

            if consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                if shuffles_remaining > 0:
                    shuffles_remaining -= 1
                    logger.warning(
                        "%s batch: %d consecutive failures. "
                        "Attempting recovery by continuing with different pairs "
                        "(generated %d/%d so far).",
                        entity_type.capitalize(),
                        consecutive_failures,
                        len(results),
                        count,
                    )
                    consecutive_failures = 0
                    # Continue generating — if prior successes populated the deduplication
                    # lists, subsequent calls will naturally target different entity pairs.
                    # Otherwise, the second round of failures will trigger termination.
                else:
                    logger.warning(
                        "%s batch early termination: %d consecutive failures "
                        "after recovery attempt. "
                        "Generated %d/%d successfully before stopping.",
                        entity_type.capitalize(),
                        consecutive_failures,
                        len(results),
                        count,
                    )
                    break

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


def _review_batch[T, S: BaseQualityScores](
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
            # VRAMAllocationError is non-retryable — stop the entire batch
            if isinstance(e.__cause__, VRAMAllocationError):
                raise
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
    name_provider: Callable[[], list[str]],
    count: int = 2,
    location_provider: Callable[[], list[str]] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], FactionQualityScores]]:
    """Generate multiple factions with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        name_provider: Callable returning current entity names from DB (live query).
        count: Number of factions to generate.
        location_provider: Callable returning current location names (live query).
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (faction_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no factions could be generated.
    """
    batch_names: list[str] = []
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="faction",
        generate_fn=lambda _i: svc.generate_faction_with_quality(
            story_state,
            name_provider() + batch_names,
            location_provider() if location_provider else None,
        ),
        get_name=lambda f: f.get("name", "Unknown"),
        on_success=lambda f: batch_names.append(f["name"]) if "name" in f else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_items_with_quality(
    svc,
    story_state: StoryState,
    name_provider: Callable[[], list[str]],
    count: int = 3,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], ItemQualityScores]]:
    """Generate multiple items with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        name_provider: Callable returning current entity names from DB (live query).
        count: Number of items to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (item_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no items could be generated.
    """
    batch_names: list[str] = []
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="item",
        generate_fn=lambda _i: svc.generate_item_with_quality(
            story_state, name_provider() + batch_names
        ),
        get_name=lambda item: item.get("name", "Unknown"),
        on_success=lambda item: batch_names.append(item["name"]) if "name" in item else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_concepts_with_quality(
    svc,
    story_state: StoryState,
    name_provider: Callable[[], list[str]],
    count: int = 2,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], ConceptQualityScores]]:
    """Generate multiple concepts with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        name_provider: Callable returning current entity names from DB (live query).
        count: Number of concepts to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (concept_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no concepts could be generated.
    """
    batch_names: list[str] = []
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="concept",
        generate_fn=lambda _i: svc.generate_concept_with_quality(
            story_state, name_provider() + batch_names
        ),
        get_name=lambda c: c.get("name", "Unknown"),
        on_success=lambda c: batch_names.append(c["name"]) if "name" in c else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_events_with_quality(
    svc,
    story_state: StoryState,
    existing_descriptions: list[str],
    entity_context_provider: Callable[[], str],
    count: int = 5,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], EventQualityScores]]:
    """Generate multiple events with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        existing_descriptions: Descriptions of existing events to avoid duplicates.
        entity_context_provider: Callable returning fresh entity context string.
            Cached by content hash to avoid rebuilding when content is unchanged.
        count: Number of events to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (event_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no events could be generated.
    """
    descriptions = existing_descriptions.copy()
    rejected: list[str] = []

    # Cache entity context by content hash so it's not rebuilt when unchanged.
    # Uses hash comparison to detect real content changes (not just line count).
    _cached_context: list[str] = [""]  # mutable container for closure
    _cached_hash: list[int] = [0]

    def _get_entity_context() -> str:
        """Return entity context, rebuilding only when content actually changes."""
        fresh = entity_context_provider()
        fresh_hash = hash(fresh)
        if fresh_hash != _cached_hash[0]:
            _cached_context[0] = fresh
            _cached_hash[0] = fresh_hash
            logger.debug("Event entity context refreshed (%d lines)", fresh.count("\n"))
        return _cached_context[0]

    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="event",
        generate_fn=lambda _i: svc.generate_event_with_quality(
            story_state, descriptions, _get_entity_context(), rejected_descriptions=rejected
        ),
        get_name=lambda evt: evt.get("description", "Unknown")[:_EVENT_DESCRIPTION_PREFIX_LEN],
        on_success=lambda evt: (
            descriptions.append(evt["description"])
            if evt.get("description")
            else logger.warning(
                "Event passed quality loop but has no description — dedup list stale"
            )
        ),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_characters_with_quality(
    svc,
    story_state: StoryState,
    name_provider: Callable[[], list[str]],
    count: int = 2,
    custom_instructions: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[Character, CharacterQualityScores]]:
    """Generate multiple characters with quality refinement.

    Uses parallel generation when count > 1 and concurrent requests are enabled.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        name_provider: Callable returning current entity names from DB (live query).
        count: Number of characters to generate.
        custom_instructions: Optional custom instructions to refine generation.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (Character, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no characters could be generated.
    """
    import threading

    from src.services.world_quality_service._batch_parallel import _generate_batch_parallel

    batch_names: list[str] = []
    batch_names_lock = threading.Lock()

    def _thread_safe_on_success(char: Character) -> None:
        """Append character name with lock for thread-safe dedup."""
        with batch_names_lock:
            if char.name not in batch_names:
                batch_names.append(char.name)

    def _get_names() -> list[str]:
        """Get deduplicated names from DB provider and batch-local names."""
        with batch_names_lock:
            provider_names = name_provider()
            seen = set(provider_names)
            return provider_names + [n for n in batch_names if n not in seen]

    # Use parallel generation when count > 1 and concurrency is enabled
    max_workers = min(svc.settings.llm_max_concurrent_requests, count)
    if count > 1 and max_workers > 1:
        logger.info(
            "Using parallel character generation: %d characters with max_workers=%d",
            count,
            max_workers,
        )
        return _generate_batch_parallel(
            svc=svc,
            count=count,
            entity_type="character",
            generate_fn=lambda _i: svc.generate_character_with_quality(
                story_state, _get_names(), custom_instructions
            ),
            get_name=lambda char: char.name,
            on_success=_thread_safe_on_success,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            max_workers=max_workers,
        )

    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="character",
        generate_fn=lambda _i: svc.generate_character_with_quality(
            story_state, _get_names(), custom_instructions
        ),
        get_name=lambda char: char.name,
        on_success=_thread_safe_on_success,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_locations_with_quality(
    svc,
    story_state: StoryState,
    name_provider: Callable[[], list[str]],
    count: int = 3,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], LocationQualityScores]]:
    """Generate multiple locations with quality refinement.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        name_provider: Callable returning current entity names from DB (live query).
        count: Number of locations to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (location_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no locations could be generated.
    """
    batch_names: list[str] = []
    return _generate_batch(
        svc=svc,
        count=count,
        entity_type="location",
        generate_fn=lambda _i: svc.generate_location_with_quality(
            story_state, name_provider() + batch_names
        ),
        get_name=lambda loc: loc.get("name", "Unknown"),
        on_success=lambda loc: batch_names.append(loc["name"]) if "name" in loc else None,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def generate_relationships_with_quality(
    svc,
    story_state: StoryState,
    entity_names_provider: Callable[[], list[str]],
    existing_rels: list[tuple[str, str, str]],
    count: int = 5,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable | None = None,
) -> list[tuple[dict[str, Any], RelationshipQualityScores]]:
    """Generate multiple relationships with quality refinement.

    Uses parallel generation (up to ``llm_max_concurrent_requests`` workers)
    with a thread-safe deduplication list. Each worker gets a point-in-time
    snapshot of existing relationships; the ``_is_duplicate_relationship()``
    check in the quality refinement loop rejects duplicates from the snapshot,
    and the atomic ``append_if_new_pair()`` dedup catches any duplicates that
    concurrent workers may produce from identical snapshots.

    When the creator and judge models differ, a **phased pipeline** is used:
    batch all creates under one model load, then batch all judges under
    another model load, reducing GPU swaps from ~2N to ~4.

    Args:
        svc: WorldQualityService instance.
        story_state: Current story state.
        entity_names_provider: Callable returning fresh entity names from DB.
        existing_rels: Existing (source, target, relation_type) 3-tuples to avoid duplicates.
        count: Number of relationships to generate.
        cancel_check: Optional callable that returns True to cancel generation.
        progress_callback: Optional callback to receive progress updates.

    Returns:
        List of (relationship_dict, QualityScores) tuples.

    Raises:
        WorldGenerationError: If no relationships could be generated.
    """
    from src.memory.conflict_types import normalize_relation_type
    from src.services.world_quality_service._batch_parallel import (
        _generate_batch_parallel,
        _ThreadSafeRelsList,
    )
    from src.services.world_quality_service._relationship import (
        _is_duplicate_relationship,
    )

    safe_rels = _ThreadSafeRelsList(existing_rels)
    max_workers = min(svc.settings.llm_max_concurrent_requests, count)
    config = svc.get_config()

    # Pairs registered by Phase 1 of the phased pipeline via register_created_fn.
    # When on_success is called for these pairs in Phase 3/3b, we skip the
    # duplicate error because the pair was already legitimately tracked.
    _phase1_pairs: set[tuple[str, str, str]] = set()

    def _on_relationship_success(r: dict[str, Any]) -> None:
        """Atomic dedup: reject if a concurrent worker already produced this pair."""
        pair = (r["source"], r["target"], r["relation_type"])
        accepted = safe_rels.append_if_new_pair(pair)
        if not accepted:
            if pair in _phase1_pairs:
                # Already registered by Phase 1 — not a real duplicate.
                return
            raise DuplicateNameError(
                f"Duplicate relationship from parallel worker: {r['source']} -> {r['target']}"
            )

    # ── Phased pipeline callables ───────────────────────────────────────
    # These are used only when creator != judge to enable the two-phase
    # batch pipeline (batch creates → batch judges → refine failures).

    def _create_only(_i: int) -> dict[str, Any] | None:
        """Create a single relationship without judging.

        Wraps ``svc._create_relationship`` with the current snapshot of
        existing relationships for deduplication.
        """
        try:
            rel: dict[str, Any] = svc._create_relationship(
                story_state,
                entity_names_provider(),
                safe_rels.snapshot(),
                config.creator_temperature,
            )
            return rel if rel else None
        except (WorldGenerationError, ValueError) as e:
            logger.warning("Phased create failed: %s", e)
            raise  # Let Phase 1's error handler catch and record it

    def _is_empty_rel(rel: dict[str, Any]) -> bool:
        """Check if a relationship is empty or a duplicate.

        Simplified version of the ``_is_empty`` closure in
        ``generate_relationship_with_quality`` — no ``required_entity``
        support since the batch path does not use it.
        """
        if not rel.get("source") or not rel.get("target"):
            logger.debug(
                "Relationship missing source/target (source=%r, target=%r), treating as empty",
                rel.get("source"),
                rel.get("target"),
            )
            return True
        source = rel.get("source", "")
        target = rel.get("target", "")
        raw_type = rel.get("relation_type") or "related_to"
        rel["relation_type"] = normalize_relation_type(raw_type)
        if _is_duplicate_relationship(source, target, safe_rels.snapshot()):
            logger.warning(
                "Phased pipeline: duplicate relationship %s -> %s, rejecting",
                source,
                target,
            )
            return True
        return False

    def _judge_only(rel: dict[str, Any]) -> RelationshipQualityScores:
        """Judge a single relationship without creating or refining."""
        result: RelationshipQualityScores = svc._judge_relationship_quality(
            rel,
            story_state,
            config.judge_temperature,
        )
        return result

    def _refine_with_initial(
        initial_rel: dict[str, Any],
    ) -> tuple[dict[str, Any], RelationshipQualityScores, int]:
        """Run the full quality refinement loop with an initial entity.

        Falls back to the per-entity ``quality_refinement_loop`` which
        handles early-stopping, hail-mary, analytics, and model preparation.
        """
        from src.services.world_quality_service._quality_loop import quality_refinement_loop

        prep_creator, prep_judge = svc._make_model_preparers("relationship")
        return quality_refinement_loop(
            entity_type="relationship",
            create_fn=lambda retries: svc._create_relationship(
                story_state,
                entity_names_provider(),
                safe_rels.snapshot(),
                config.creator_temperature,
            ),
            judge_fn=lambda rel: svc._judge_relationship_quality(
                rel,
                story_state,
                config.judge_temperature,
            ),
            refine_fn=lambda rel, scores, iteration: svc._refine_relationship(
                rel,
                scores,
                story_state,
                config.get_refinement_temperature(iteration),
            ),
            get_name=lambda rel: f"{rel.get('source', '?')} -> {rel.get('target', '?')}",
            serialize=lambda rel: rel.copy(),
            is_empty=_is_empty_rel,
            score_cls=RelationshipQualityScores,
            config=config,
            svc=svc,
            story_id=story_state.id,
            initial_entity=initial_rel,
            prepare_creator=prep_creator,
            prepare_judge=prep_judge,
        )

    def _register_phase1_pair(r: dict[str, Any]) -> None:
        """Register a relationship created in Phase 1 for dedup tracking.

        Adds the pair to both the thread-safe dedup list (``safe_rels``) and
        the Phase 1 pair set (``_phase1_pairs``).  The latter lets
        ``_on_relationship_success`` skip the duplicate error for pairs that
        were legitimately registered during Phase 1 batch creation.
        """
        pair = (r["source"], r["target"], r.get("relation_type", "related_to"))
        safe_rels.append(pair)
        _phase1_pairs.add(pair)

    # Resolve model preparers and determine whether models differ.  The
    # phased pipeline callables are only passed when creator != judge, so
    # `_generate_batch_parallel` can use the phased path.  When models are
    # the same, phased batching has no benefit (no GPU swap to eliminate).
    _phased_kwargs: dict[str, Any] = {}
    if max_workers > 1:
        try:
            prep_creator, prep_judge = svc._make_model_preparers("relationship")
            # _make_model_preparers returns (None, None) when models are the
            # same.  Only enable phased callables when at least one preparer
            # is non-None (i.e. models actually differ).
            if prep_creator is not None or prep_judge is not None:
                _phased_kwargs = {
                    "create_only_fn": _create_only,
                    "judge_only_fn": _judge_only,
                    "is_empty_fn": _is_empty_rel,
                    "refine_with_initial_fn": _refine_with_initial,
                    "prepare_creator_fn": prep_creator,
                    "prepare_judge_fn": prep_judge,
                    "register_created_fn": _register_phase1_pair,
                }
                logger.debug(
                    "Phased pipeline callables prepared for relationship batch (creator != judge)"
                )
        except (ValueError, KeyError, LookupError) as e:
            logger.warning(
                "Failed to resolve model preparers for phased pipeline: %s. "
                "Falling back to sequential path.",
                e,
            )

    return _generate_batch_parallel(
        svc=svc,
        count=count,
        entity_type="relationship",
        generate_fn=lambda _i: svc.generate_relationship_with_quality(
            story_state, entity_names_provider(), safe_rels.snapshot()
        ),
        get_name=lambda r: f"{r['source']} -> {r['target']}",
        on_success=_on_relationship_success,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        max_workers=max_workers,
        **_phased_kwargs,
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
            depth=0,
            goals=0,
            flaws=0,
            uniqueness=0,
            arc_potential=0,
            temporal_plausibility=0,
            feedback=msg,
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
