"""Batch generation functions for multiple entities."""

import logging
import time
from collections.abc import Callable
from typing import Any

from src.memory.story_state import Character, StoryState
from src.memory.world_quality import (
    CharacterQualityScores,
    ConceptQualityScores,
    FactionQualityScores,
    ItemQualityScores,
    LocationQualityScores,
    RelationshipQualityScores,
)
from src.utils.exceptions import WorldGenerationError

logger = logging.getLogger("src.services.world_quality_service._batch")


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
    from src.services.world_quality_service import EntityGenerationProgress

    results: list[tuple[dict[str, Any], FactionQualityScores]] = []
    names = existing_names.copy()
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        # Check cancellation before starting entity
        if cancel_check and cancel_check():
            logger.info(f"Faction generation cancelled after {len(results)}/{count}")
            break

        # Report progress before generation
        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type="faction",
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info(f"Generating faction {i + 1}/{count} with quality refinement")
            faction, scores, iterations = svc.generate_faction_with_quality(
                story_state, names, existing_locations
            )
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((faction, scores))
            faction_name = faction.get("name", "Unknown")
            names.append(faction_name)
            logger.info(
                f"Faction '{faction_name}' complete after {iterations} iteration(s), "
                f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
            )

            # Report completion with entity name
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="faction",
                        entity_name=faction_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
            # Analytics already recorded via _log_refinement_analytics in generate_faction_with_quality
        except WorldGenerationError as e:
            errors.append(str(e))
            logger.error(f"Failed to generate faction {i + 1}/{count}: {e}")

    if not results and errors:
        raise WorldGenerationError(f"Failed to generate any factions. Errors: {'; '.join(errors)}")

    if errors:
        logger.warning(
            f"Generated {len(results)}/{count} factions. {len(errors)} failed: {'; '.join(errors)}"
        )

    return results


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
    from src.services.world_quality_service import EntityGenerationProgress

    results: list[tuple[dict[str, Any], ItemQualityScores]] = []
    names = existing_names.copy()
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        # Check cancellation before starting entity
        if cancel_check and cancel_check():
            logger.info(f"Item generation cancelled after {len(results)}/{count}")
            break

        # Report progress before generation
        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type="item",
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info(f"Generating item {i + 1}/{count} with quality refinement")
            item, scores, iterations = svc.generate_item_with_quality(story_state, names)
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((item, scores))
            item_name = item.get("name", "Unknown")
            names.append(item_name)
            logger.info(
                f"Item '{item_name}' complete after {iterations} iteration(s), "
                f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
            )

            # Report completion with entity name
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="item",
                        entity_name=item_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
            # Analytics already recorded via _log_refinement_analytics in generate_item_with_quality
        except WorldGenerationError as e:
            errors.append(str(e))
            logger.error(f"Failed to generate item {i + 1}/{count}: {e}")

    if not results and errors:
        raise WorldGenerationError(f"Failed to generate any items. Errors: {'; '.join(errors)}")

    if errors:
        logger.warning(
            f"Generated {len(results)}/{count} items. {len(errors)} failed: {'; '.join(errors)}"
        )

    return results


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
    from src.services.world_quality_service import EntityGenerationProgress

    results: list[tuple[dict[str, Any], ConceptQualityScores]] = []
    names = existing_names.copy()
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        # Check cancellation before starting entity
        if cancel_check and cancel_check():
            logger.info(f"Concept generation cancelled after {len(results)}/{count}")
            break

        # Report progress before generation
        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type="concept",
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info(f"Generating concept {i + 1}/{count} with quality refinement")
            concept, scores, iterations = svc.generate_concept_with_quality(story_state, names)
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((concept, scores))
            concept_name = concept.get("name", "Unknown")
            names.append(concept_name)
            logger.info(
                f"Concept '{concept_name}' complete after {iterations} iteration(s), "
                f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
            )

            # Report completion with entity name
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="concept",
                        entity_name=concept_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
            # Analytics already recorded via _log_refinement_analytics in generate_concept_with_quality
        except WorldGenerationError as e:
            errors.append(str(e))
            logger.error(f"Failed to generate concept {i + 1}/{count}: {e}")

    if not results and errors:
        raise WorldGenerationError(f"Failed to generate any concepts. Errors: {'; '.join(errors)}")

    if errors:
        logger.warning(
            f"Generated {len(results)}/{count} concepts. {len(errors)} failed: {'; '.join(errors)}"
        )

    return results


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
    from src.services.world_quality_service import EntityGenerationProgress

    results: list[tuple[Character, CharacterQualityScores]] = []
    names = existing_names.copy()
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        # Check cancellation before starting entity
        if cancel_check and cancel_check():
            logger.info(f"Character generation cancelled after {len(results)}/{count}")
            break

        # Report progress before generation
        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type="character",
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info(f"Generating character {i + 1}/{count} with quality refinement")
            char, scores, iterations = svc.generate_character_with_quality(
                story_state, names, custom_instructions
            )
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((char, scores))
            names.append(char.name)
            logger.info(
                f"Character '{char.name}' complete after {iterations} iteration(s), "
                f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
            )

            # Report completion with entity name
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="character",
                        entity_name=char.name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
            # Analytics already recorded via _log_refinement_analytics in generate_character_with_quality
        except WorldGenerationError as e:
            errors.append(str(e))
            logger.error(f"Failed to generate character {i + 1}/{count}: {e}")

    if not results and errors:
        raise WorldGenerationError(
            f"Failed to generate any characters. Errors: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            f"Generated {len(results)}/{count} characters. "
            f"{len(errors)} failed: {'; '.join(errors)}"
        )

    return results


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
    from src.services.world_quality_service import EntityGenerationProgress

    results: list[tuple[dict[str, Any], LocationQualityScores]] = []
    names = existing_names.copy()
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        # Check cancellation before starting entity
        if cancel_check and cancel_check():
            logger.info(f"Location generation cancelled after {len(results)}/{count}")
            break

        # Report progress before generation
        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type="location",
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info(f"Generating location {i + 1}/{count} with quality refinement")
            loc, scores, iterations = svc.generate_location_with_quality(story_state, names)
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((loc, scores))
            loc_name = loc.get("name", "Unknown")
            names.append(loc_name)
            logger.info(
                f"Location '{loc_name}' complete after {iterations} iteration(s), "
                f"quality: {scores.average:.1f}, generation_time: {entity_elapsed:.2f}s"
            )

            # Report completion with entity name
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="location",
                        entity_name=loc_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
            # Analytics already recorded via _log_refinement_analytics in generate_location_with_quality
        except WorldGenerationError as e:
            errors.append(str(e))
            logger.error(f"Failed to generate location {i + 1}/{count}: {e}")

    if not results and errors:
        raise WorldGenerationError(f"Failed to generate any locations. Errors: {'; '.join(errors)}")

    if errors:
        logger.warning(
            f"Generated {len(results)}/{count} locations. {len(errors)} failed: {'; '.join(errors)}"
        )

    return results


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
    from src.services.world_quality_service import EntityGenerationProgress

    results: list[tuple[dict[str, Any], RelationshipQualityScores]] = []
    rels = existing_rels.copy()
    errors: list[str] = []
    batch_start_time = time.time()
    completed_times: list[float] = []

    for i in range(count):
        # Check cancellation before starting entity
        if cancel_check and cancel_check():
            logger.info(f"Relationship generation cancelled after {len(results)}/{count}")
            break

        # Report progress before generation
        if progress_callback:
            progress_callback(
                EntityGenerationProgress(
                    current=i + 1,
                    total=count,
                    entity_type="relationship",
                    phase="generating",
                    elapsed_seconds=time.time() - batch_start_time,
                    estimated_remaining_seconds=svc._calculate_eta(completed_times, count - i),
                )
            )

        entity_start = time.time()
        try:
            logger.info(f"Generating relationship {i + 1}/{count} with quality refinement")
            rel, scores, iterations = svc.generate_relationship_with_quality(
                story_state, entity_names, rels
            )
            entity_elapsed = time.time() - entity_start
            completed_times.append(entity_elapsed)
            results.append((rel, scores))
            rels.append((rel.get("source", ""), rel.get("target", "")))
            rel_name = f"{rel.get('source', '')} -> {rel.get('target', '')}"
            logger.info(
                f"Relationship '{rel_name}' complete "
                f"after {iterations} iteration(s), quality: {scores.average:.1f}, "
                f"generation_time: {entity_elapsed:.2f}s"
            )

            # Report completion with entity name
            if progress_callback:
                progress_callback(
                    EntityGenerationProgress(
                        current=i + 1,
                        total=count,
                        entity_type="relationship",
                        entity_name=rel_name,
                        phase="complete",
                        elapsed_seconds=time.time() - batch_start_time,
                        estimated_remaining_seconds=svc._calculate_eta(
                            completed_times, count - i - 1
                        ),
                    )
                )
            # Analytics already recorded via _log_refinement_analytics in generate_relationship_with_quality
        except WorldGenerationError as e:
            errors.append(str(e))
            logger.error(f"Failed to generate relationship {i + 1}/{count}: {e}")

    if not results and errors:
        raise WorldGenerationError(
            f"Failed to generate any relationships. Errors: {'; '.join(errors)}"
        )

    if errors:
        logger.warning(
            f"Generated {len(results)}/{count} relationships. "
            f"{len(errors)} failed: {'; '.join(errors)}"
        )

    return results
