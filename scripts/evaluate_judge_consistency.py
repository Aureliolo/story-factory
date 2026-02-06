#!/usr/bin/env python3
"""Evaluate judge scoring consistency for world building entities.

Submits the same entity to the judge N times to measure scoring variance.
Determines if the judge is noisy (std > 0.5) or consistent (std < 0.2).

Supports temperature sweep mode (--temperatures) to test variance across
multiple judge temperatures. Creates each entity once, then judges at
each temperature, producing a grouped summary and decision guidance.

Usage:
    python scripts/evaluate_judge_consistency.py [options]
      --entity-types faction,concept  (default: all 6)
      --judge-calls 5                 (default: 5)
      --temperatures 0.1,0.3,0.5,0.7 (default: configured judge temp)
      --output results.json           (default: output/diagnostics/<timestamp>_judge.json)
      --verbose
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_refinement import (
    ALL_ENTITY_TYPES,
    SCORE_DIMENSIONS,
    make_canonical_brief,
    make_story_state,
)
from src.memory.story_state import Character, StoryState
from src.services import ServiceContainer
from src.settings import Settings
from src.utils.exceptions import StoryFactoryError

logger = logging.getLogger(__name__)

# Verdict thresholds
NOISY_STD_THRESHOLD = 0.5
CONSISTENT_STD_THRESHOLD = 0.2

# Guidance thresholds for temperature sweep analysis
LOW_VARIANCE_STD_THRESHOLD = 0.05
HIGH_TEMP_SWEET_SPOT = 0.7


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max, coefficient of variation for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Dict with mean, std, min, max, cv statistics.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}

    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": round(mean, 3), "std": 0.0, "min": mean, "max": mean, "cv": 0.0}

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = variance**0.5
    cv = (std / mean) if mean > 0 else 0.0

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "cv": round(cv, 3),
    }


def compute_feedback_similarity(feedbacks: list[str]) -> float:
    """Compute pairwise Jaccard word overlap between feedback strings.

    Args:
        feedbacks: List of feedback strings.

    Returns:
        Average Jaccard similarity (0-1) across all pairs.
    """
    if len(feedbacks) < 2:
        return 1.0

    word_sets = [set(fb.lower().split()) for fb in feedbacks if fb]
    if len(word_sets) < 2:
        return 1.0

    similarities = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            union = word_sets[i] | word_sets[j]
            if union:
                intersection = word_sets[i] & word_sets[j]
                similarities.append(len(intersection) / len(union))
            else:
                similarities.append(1.0)

    return round(sum(similarities) / len(similarities), 3) if similarities else 1.0


def determine_verdict(per_dimension_stats: dict[str, dict[str, float]]) -> str:
    """Determine judge consistency verdict from per-dimension statistics.

    Args:
        per_dimension_stats: Dict mapping dimension name to statistics.

    Returns:
        One of: "consistent", "noisy", "borderline".
    """
    stds = [stats["std"] for stats in per_dimension_stats.values()]
    if not stds:
        return "consistent"

    avg_std = sum(stds) / len(stds)
    max_std = max(stds)

    if max_std > NOISY_STD_THRESHOLD or avg_std > NOISY_STD_THRESHOLD:
        return "noisy"
    if max_std < CONSISTENT_STD_THRESHOLD and avg_std < CONSISTENT_STD_THRESHOLD:
        return "consistent"
    return "borderline"


def make_synthetic_entity(entity_type: str) -> tuple[dict[str, Any] | None, str]:
    """Create a synthetic entity for judge testing without LLM calls.

    Returns a hardcoded entity matching the canonical Mnemorian Empire brief.
    Used as fallback when the creator model fails structured output.

    Args:
        entity_type: Entity type to create.

    Returns:
        Tuple of (entity_data, entity_name).
    """
    synthetics: dict[str, tuple[dict[str, Any], str]] = {
        "character": (
            {
                "name": "Veyra Ashcroft",
                "role": "protagonist",
                "description": (
                    "A disgraced archivist who once catalogued the empire's most sensitive "
                    "memory vaults. After discovering evidence of systematic memory erasure, "
                    "she was stripped of her credentials and exiled to the outer provinces. "
                    "She carries a stolen memory shard that holds proof of the council's crimes."
                ),
                "personality_traits": [
                    "obsessively meticulous",
                    "distrustful of authority",
                    "quietly compassionate",
                    "prone to self-isolation",
                ],
                "goals": [
                    "Restore the erased collective memories before they decay permanently",
                    "Expose the ruling council's memory manipulation program",
                ],
                "relationships": {},
                "arc_notes": (
                    "Begins as a solitary crusader convinced she must work alone. "
                    "Gradually learns that restoring memory requires trust and connection. "
                    "Must confront that some erased memories were suppressed for reasons "
                    "she doesn't fully understand."
                ),
                "arc_type": "disillusionment",
                "arc_progress": {},
            },
            "Veyra Ashcroft",
        ),
        "faction": (
            {
                "name": "The Ashen Quorum",
                "description": (
                    "A secret society of former archivists and scholars who preserve "
                    "forbidden memories in hidden vaults beneath the empire's libraries."
                ),
                "leader": "Archivist Matron Lys",
                "goals": ["Preserve erased memories", "Undermine the council's grip on history"],
                "values": ["Truth", "Knowledge preservation", "Academic freedom"],
                "resources": ["Hidden memory vaults", "Network of sympathetic librarians"],
            },
            "The Ashen Quorum",
        ),
        "location": (
            {
                "name": "The Sunken Archive",
                "description": (
                    "A vast underground library beneath the capital, flooded during the "
                    "First Erasure. Its waterlogged halls still contain memory crystals "
                    "that glow faintly in the darkness."
                ),
                "significance": "Contains pre-erasure records the council thought destroyed",
                "atmosphere": "Eerie, waterlogged, faintly luminescent",
            },
            "The Sunken Archive",
        ),
        "item": (
            {
                "name": "The Anamnesis Lens",
                "description": (
                    "A crystalline monocle that allows the wearer to perceive traces of "
                    "erased memories lingering in places and objects. Prolonged use causes "
                    "the wearer's own memories to blur."
                ),
                "significance": (
                    "Key to recovering erased memories hidden in the empire's archives."
                ),
                "properties": [
                    "Reveals erased memory traces",
                    "Decodes memory crystal recordings",
                ],
                "powers": ["Reveals erased memory traces", "Decodes memory crystal recordings"],
                "drawbacks": ["Gradual memory erosion in the user"],
            },
            "The Anamnesis Lens",
        ),
        "concept": (
            {
                "name": "The Great Erasure",
                "description": (
                    "The council's systematic program of removing dangerous or inconvenient "
                    "memories from the collective consciousness. Performed through ritual "
                    "magic that draws on the empire's ley lines."
                ),
                "manifestations": (
                    "Citizens lose shared history; archives fracture as public rituals "
                    "blur memory and communal identity erodes."
                ),
                "themes": ["Censorship", "Collective trauma", "Historical revisionism"],
                "implications": ["Weakens the empire's magical foundation over time"],
            },
            "The Great Erasure",
        ),
        "relationship": (
            {
                "source": "Veyra Ashcroft",
                "target": "High Censor Drystan",
                "relation_type": "enemy_of",
                "description": (
                    "Veyra was Drystan's most promising student before she discovered his "
                    "role in the memory erasure program. He views her as a dangerous loose "
                    "end; she sees him as a betrayer of everything the archives stood for."
                ),
            },
            "Veyra Ashcroft -> High Censor Drystan",
        ),
    }

    if entity_type in synthetics:
        logger.info("Using synthetic %s entity (LLM creation unavailable)", entity_type)
        return synthetics[entity_type]

    return None, ""


def create_entity(
    svc_container: ServiceContainer,
    story_state: StoryState,
    entity_type: str,
) -> tuple[dict[str, Any] | None, str]:
    """Create a single entity using the production _create_X method.

    Args:
        svc_container: Service container.
        story_state: Story state with brief.
        entity_type: Entity type to create.

    Returns:
        Tuple of (entity_data, entity_name).
    """
    wqs = svc_container.world_quality
    config = wqs.get_config()
    existing_names: list[str] = []

    if entity_type == "character":
        entity_obj = wqs._create_character(story_state, existing_names, config.creator_temperature)
        if entity_obj:
            return entity_obj.model_dump(), entity_obj.name
        return None, ""
    elif entity_type == "faction":
        data = wqs._create_faction(story_state, existing_names, config.creator_temperature)
        return data, data.get("name", "") if data else ""
    elif entity_type == "location":
        data = wqs._create_location(story_state, existing_names, config.creator_temperature)
        return data, data.get("name", "") if data else ""
    elif entity_type == "item":
        data = wqs._create_item(story_state, existing_names, config.creator_temperature)
        return data, data.get("name", "") if data else ""
    elif entity_type == "concept":
        data = wqs._create_concept(story_state, existing_names, config.creator_temperature)
        return data, data.get("name", "") if data else ""
    elif entity_type == "relationship":
        entity_names = ["Archivist Sera", "Councillor Vex", "The Pale Librarian"]
        data = wqs._create_relationship(story_state, entity_names, [], config.creator_temperature)
        name = f"{data.get('source', '')} -> {data.get('target', '')}" if data else ""
        return data, name

    return None, ""


def judge_entity(
    svc_container: ServiceContainer,
    story_state: StoryState,
    entity_type: str,
    entity_data: dict[str, Any],
    temperature: float,
) -> dict[str, Any] | None:
    """Judge an entity once and return scores as dict.

    Args:
        svc_container: Service container.
        story_state: Story state.
        entity_type: Entity type.
        entity_data: Entity data dict.
        temperature: Judge temperature.

    Returns:
        Scores dict or None on error.
    """
    wqs = svc_container.world_quality

    try:
        if entity_type == "character":
            scores = wqs._judge_character_quality(
                Character(**entity_data), story_state, temperature
            )
        elif entity_type == "faction":
            scores = wqs._judge_faction_quality(entity_data, story_state, temperature)
        elif entity_type == "location":
            scores = wqs._judge_location_quality(entity_data, story_state, temperature)
        elif entity_type == "item":
            scores = wqs._judge_item_quality(entity_data, story_state, temperature)
        elif entity_type == "concept":
            scores = wqs._judge_concept_quality(entity_data, story_state, temperature)
        elif entity_type == "relationship":
            scores = wqs._judge_relationship_quality(entity_data, story_state, temperature)
        else:
            return None

        score_dict = scores.to_dict()
        return {
            "dimensions": {k: v for k, v in score_dict.items() if k not in ("average", "feedback")},
            "average": round(scores.average, 2),
            "feedback": scores.feedback,
        }
    except StoryFactoryError as e:
        logger.error("Judge error for %s: %s", entity_type, e)
        return None


def _make_empty_result(
    entity_type: str,
    judge_calls: int,
    judge_temperature: float,
    error: str | None = None,
) -> dict[str, Any]:
    """Create the canonical empty result dict for a consistency test.

    Used both as the initial result in run_judge_consistency_test and as the
    error placeholder when entity creation fails in sweep mode.
    """
    return {
        "entity_type": entity_type,
        "entity_name": "",
        "entity_data": None,
        "judge_calls": judge_calls,
        "judge_temperature": judge_temperature,
        "individual_scores": [],
        "per_dimension_stats": {},
        "average_stats": {},
        "feedback_similarity": 0.0,
        "verdict": "",
        "error": error,
    }


def run_judge_consistency_test(
    svc_container: ServiceContainer,
    story_state: StoryState,
    entity_type: str,
    judge_calls: int,
    verbose: bool = False,
    temperature: float | None = None,
    entity_data: dict[str, Any] | None = None,
    entity_name: str = "",
) -> dict[str, Any]:
    """Run judge consistency test for one entity type.

    1. Generate one entity with _create_X() (or use pre-created entity_data)
    2. Call _judge_X_quality() N times on the frozen entity
    3. Compute statistics

    Args:
        svc_container: Service container.
        story_state: Story state.
        entity_type: Entity type.
        judge_calls: Number of judge calls.
        verbose: Verbose output.
        temperature: Override judge temperature (uses config value when None).
        entity_data: Pre-created entity data to skip creation step.
        entity_name: Name of pre-created entity (used with entity_data).

    Returns:
        Dict with consistency test results.
    """
    config = svc_container.world_quality.get_config()
    judge_temp = temperature if temperature is not None else config.judge_temperature

    result = _make_empty_result(entity_type, judge_calls, judge_temp)

    # Step 1: Create entity (or reuse pre-created)
    if entity_data is not None:
        result["entity_name"] = entity_name
        result["entity_data"] = entity_data
        logger.debug("Reusing pre-created %s entity '%s'", entity_type, entity_name)
    else:
        print(f"  Creating {entity_type}...")
        start = time.monotonic()
        try:
            entity_data, entity_name = create_entity(svc_container, story_state, entity_type)
        except StoryFactoryError as e:
            result["error"] = f"Creation failed: {e}"
            logger.error("Failed to create %s: %s", entity_type, e)
            return result

        if not entity_data:
            result["error"] = "Creation returned empty entity"
            return result

        result["entity_name"] = entity_name
        result["entity_data"] = entity_data
        create_time = round(time.monotonic() - start, 2)
        print(f"    Created '{entity_name}' in {create_time}s")

    # Step 2: Judge N times
    feedbacks: list[str] = []
    averages: list[float] = []
    dimension_scores: dict[str, list[float]] = {
        dim: [] for dim in SCORE_DIMENSIONS.get(entity_type, [])
    }

    for call_idx in range(judge_calls):
        call_start = time.monotonic()
        judge_result = judge_entity(
            svc_container, story_state, entity_type, entity_data, judge_temp
        )
        call_time = round(time.monotonic() - call_start, 2)

        if judge_result is None:
            result["individual_scores"].append({"error": "Judge call failed"})
            logger.warning("Judge call %d/%d failed for %s", call_idx + 1, judge_calls, entity_type)
            continue

        result["individual_scores"].append(judge_result)
        averages.append(judge_result["average"])
        feedbacks.append(judge_result["feedback"])

        for dim, value in judge_result["dimensions"].items():
            if dim in dimension_scores:
                dimension_scores[dim].append(float(value))

        if verbose:
            dims_str = ", ".join(
                f"{d}={judge_result['dimensions'].get(d, '?')}"
                for d in SCORE_DIMENSIONS.get(entity_type, [])
            )
            print(
                f"    Call {call_idx + 1}/{judge_calls}: avg={judge_result['average']:.1f} "
                f"({dims_str}) [{call_time}s]"
            )

    # Step 3: Compute statistics
    if not averages:
        logger.warning(
            "No successful judge calls for %s — cannot compute consistency stats",
            entity_type,
        )
        result["verdict"] = "insufficient_data"
        return result

    result["average_stats"] = compute_statistics(averages)

    per_dim_stats = {}
    for dim, values in dimension_scores.items():
        if values:
            per_dim_stats[dim] = compute_statistics(values)
    result["per_dimension_stats"] = per_dim_stats

    result["feedback_similarity"] = compute_feedback_similarity(feedbacks)
    result["verdict"] = determine_verdict(per_dim_stats)

    return result


def parse_temperatures(temp_str: str) -> list[float]:
    """Parse and validate a comma-separated list of temperatures.

    Args:
        temp_str: Comma-separated temperature values (e.g. "0.1,0.3,0.5").

    Returns:
        Sorted list of validated temperature floats.

    Raises:
        SystemExit: If any temperature is outside the 0.0-2.0 range.
    """
    temperatures: list[float] = []
    for part in temp_str.split(","):
        part = part.strip()
        try:
            temp = float(part)
        except ValueError:
            print(f"ERROR: Invalid temperature value: '{part}'")
            sys.exit(1)
        if temp < 0.0 or temp > 2.0:
            print(f"ERROR: Temperature {temp} out of range (0.0-2.0)")
            sys.exit(1)
        temperatures.append(temp)

    logger.debug("Parsed temperatures: %s", temperatures)
    return sorted(temperatures)


def print_temperature_sweep_summary(results: list[dict[str, Any]]) -> None:
    """Print a grouped summary table for temperature sweep results.

    Args:
        results: List of result dicts, each with entity_type, judge_temperature,
                 verdict, average_stats, and feedback_similarity.
    """
    print("\n=== JUDGE CONSISTENCY SUMMARY (TEMPERATURE SWEEP) ===")
    print(
        f"{'Type':<15} {'Temp':<7} {'Verdict':<12} {'Mean':<8} "
        f"{'Std':<8} {'Range':<12} {'FB Sim':<8}"
    )
    print("-" * 70)
    for r in results:
        if r.get("error") is not None:
            et = r["entity_type"]
            temp = r.get("judge_temperature", 0)
            print(f"{et:<15} {temp:<7.1f} {'ERROR':<12} {'--':>8} {'--':>8} {'--':>12} {'--':>8}")
            continue
        et = r["entity_type"]
        temp = r.get("judge_temperature", 0)
        v = r.get("verdict", "error").upper()
        avg = r.get("average_stats", {})
        fb = r.get("feedback_similarity", 0)
        range_str = f"{avg.get('min', 0):.1f}-{avg.get('max', 0):.1f}"
        print(
            f"{et:<15} {temp:<7.1f} {v:<12} {avg.get('mean', 0):<8.1f} "
            f"{avg.get('std', 0):<8.2f} {range_str:<12} {fb:<8.2f}"
        )


def print_temperature_sweep_guidance(
    results: list[dict[str, Any]],
    temperatures: list[float],
) -> None:
    """Analyze temperature sweep results and print decision guidance for #266.

    Examines per-temperature variance to recommend whether multi-call judge
    averaging adds value and at which temperature it becomes useful.

    Args:
        results: List of result dicts from the sweep (multiple temps per entity type).
        temperatures: Sorted list of temperatures tested.
    """
    print("\n=== TEMPERATURE SWEEP GUIDANCE (for Issue #266 / A1) ===")

    # Group results by temperature
    by_temp: dict[float, list[dict[str, Any]]] = {}
    for r in results:
        temp = r.get("judge_temperature", 0)
        by_temp.setdefault(temp, []).append(r)

    # Compute average std per temperature (None = no usable data at this temp)
    temp_avg_stds: dict[float, float | None] = {}
    for temp in temperatures:
        temp_results = by_temp.get(temp, [])
        stds = [
            r.get("average_stats", {}).get("std", 0)
            for r in temp_results
            if r.get("verdict") not in ("insufficient_data", "") and r.get("error") is None
        ]
        temp_avg_stds[temp] = sum(stds) / len(stds) if stds else None

    # Filter to temps with actual data for variance checks
    valid_stds = [std for std in temp_avg_stds.values() if std is not None]
    all_low_variance = bool(valid_stds) and all(
        std < LOW_VARIANCE_STD_THRESHOLD for std in valid_stds
    )
    all_noisy = bool(valid_stds) and all(std > NOISY_STD_THRESHOLD for std in valid_stds)

    if all_low_variance:
        print(
            f"RESULT: All temperatures show near-zero variance "
            f"(avg std < {LOW_VARIANCE_STD_THRESHOLD})"
        )
        print("RECOMMENDATION: Multi-call averaging is WASTEFUL at all tested temperatures.")
        print("  -> Disable multi-call averaging (Issue #266 C4). Halves all judge costs.")
        print("  -> Single judge call is sufficient for consistent scores.")
    elif all_noisy:
        print(f"RESULT: All temperatures show high variance (avg std > {NOISY_STD_THRESHOLD})")
        print("RECOMMENDATION: Lower judge temperature further or fix judge prompts.")
        print("  -> Multi-call averaging may help but does not fix the root cause.")
    else:
        # Find where meaningful variance first appears
        sweet_spot_temp = None
        for temp in temperatures:
            std = temp_avg_stds[temp]
            if std is not None and std >= LOW_VARIANCE_STD_THRESHOLD:
                sweet_spot_temp = temp
                break

        if sweet_spot_temp is not None and sweet_spot_temp >= HIGH_TEMP_SWEET_SPOT:
            print(
                f"RESULT: Variance only appears at temp >= {sweet_spot_temp} "
                f"(avg std = {temp_avg_stds[sweet_spot_temp]:.3f})"
            )
            print("RECOMMENDATION: Remove multi-call averaging.")
            print("  -> Variance at high temps is not worth 2-4x cost.")
            print("  -> Keep judge temperature low for reliable single-call scoring.")
        elif sweet_spot_temp is not None:
            print(
                f"RESULT: Meaningful variance appears at temp >= {sweet_spot_temp} "
                f"(avg std = {temp_avg_stds[sweet_spot_temp]:.3f})"
            )
            print(
                f"RECOMMENDATION: If judge temp is raised to {sweet_spot_temp}+, "
                "multi-call averaging adds value."
            )
            print("  -> Trade-off: higher temp = more diverse feedback but noisier scores.")
            print("  -> Consider keeping low temp + single call for best cost/reliability.")

    # Per-temperature breakdown
    print("\nPer-temperature average std:")
    for temp in temperatures:
        std = temp_avg_stds[temp]
        std_str = f"{std:.3f}" if std is not None else "no data"
        print(f"  temp={temp:.1f}: avg_std={std_str}")

    # KV cache caveat
    print("\nCAVEAT: Ollama KV cache effect")
    print("  Ollama caches loaded model state. Repeated identical prompts to the same")
    print("  model may produce deterministic output regardless of temperature or seed.")
    print("  This affects production equally (same prompt = same score). Results above")
    print("  are valid for the tested model but may differ with other judge models.")
    print("  To verify, re-run with a different model via --judge-model.")


def main() -> None:
    """Main entry point for the judge consistency evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate judge scoring consistency for world building entities."
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        help=f"Comma-separated entity types (default: all). Options: {', '.join(ALL_ENTITY_TYPES)}",
    )
    parser.add_argument(
        "--judge-calls",
        type=int,
        default=5,
        help="Number of judge calls per entity (default: 5)",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        help=(
            "Comma-separated temperatures to sweep (default: configured judge temp). "
            "Example: 0.1,0.3,0.5,0.7"
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Override judge model (default: auto-selected). Useful for cross-model comparison.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: output/diagnostics/<timestamp>_judge.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress to console",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse entity types
    if args.entity_types:
        entity_types = [t.strip() for t in args.entity_types.split(",")]
        invalid = [t for t in entity_types if t not in ALL_ENTITY_TYPES]
        if invalid:
            print(f"ERROR: Invalid entity types: {invalid}")
            sys.exit(1)
    else:
        entity_types = ALL_ENTITY_TYPES

    # Initialize services
    print("Loading settings and initializing services...")
    settings = Settings.load()

    # Override judge model if specified via the settings agent_models config.
    # This flows through the normal model resolution: _get_judge_model reads
    # settings.agent_models["judge"], so setting it before ServiceContainer
    # creation ensures all judge calls use the override.
    judge_model_override = args.judge_model
    if judge_model_override:
        logger.info("Overriding judge model to: %s", judge_model_override)
        settings.agent_models["judge"] = judge_model_override
        if not settings.use_per_agent_models:
            settings.use_per_agent_models = True

    svc = ServiceContainer(settings)
    config = svc.world_quality.get_config()

    # Parse temperatures
    is_sweep = args.temperatures is not None
    if is_sweep:
        temperatures = parse_temperatures(args.temperatures)
    else:
        temperatures = [config.judge_temperature]

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        suffix = "sweep" if is_sweep else "judge"
        output_path = diagnostics_dir / f"judge_consistency_{suffix}_{timestamp}.json"

    # Create canonical story state
    brief = make_canonical_brief()
    story_state = make_story_state(brief)

    judge_model_display = judge_model_override or svc.world_quality._get_judge_model()
    print(f"Judge model: {judge_model_display}{' (override)' if judge_model_override else ''}")
    print(f"Temperatures: {temperatures}")
    print(f"Judge calls per entity per temperature: {args.judge_calls}")
    print(f"Entity types: {entity_types}")
    print(f"Output: {output_path}")
    print()

    # Run consistency tests
    results: list[dict[str, Any]] = []

    if is_sweep:
        # Sweep mode: create entity once, judge at each temperature
        for et in entity_types:
            print(f"--- Creating {et} for temperature sweep ---")
            frozen_data = None
            frozen_name = ""
            try:
                frozen_data, frozen_name = create_entity(svc, story_state, et)
            except StoryFactoryError as e:
                logger.warning("LLM creation failed for %s: %s", et, e)

            if not frozen_data:
                # Fall back to synthetic entity so the judge can still be tested
                print(f"  LLM creation failed, using synthetic {et} entity...")
                frozen_data, frozen_name = make_synthetic_entity(et)

            if not frozen_data:
                logger.error("No entity available for %s (LLM and synthetic both failed)", et)
                for temp in temperatures:
                    results.append(
                        _make_empty_result(et, args.judge_calls, temp, error="No entity available")
                    )
                continue

            print(f"  Created '{frozen_name}'")

            for temp in temperatures:
                print(
                    f"  Judging {et} '{frozen_name}' at temp={temp} ({args.judge_calls} calls)..."
                )
                result = run_judge_consistency_test(
                    svc,
                    story_state,
                    et,
                    args.judge_calls,
                    args.verbose,
                    temperature=temp,
                    entity_data=frozen_data,
                    entity_name=frozen_name,
                )
                results.append(result)

                verdict = result.get("verdict", "error")
                avg_stats = result.get("average_stats", {})
                fb_sim = result.get("feedback_similarity", 0)
                print(
                    f"    temp={temp} -> {verdict.upper()} | "
                    f"avg={avg_stats.get('mean', 0):.1f} "
                    f"std={avg_stats.get('std', 0):.3f} | "
                    f"fb_sim={fb_sim:.2f}"
                )
            print()
    else:
        # Single-temperature mode (backward-compatible)
        for et in entity_types:
            print(f"--- Testing {et} judge consistency ({args.judge_calls} calls) ---")
            result = run_judge_consistency_test(
                svc, story_state, et, args.judge_calls, args.verbose
            )
            results.append(result)

            verdict = result.get("verdict", "error")
            avg_stats = result.get("average_stats", {})
            fb_sim = result.get("feedback_similarity", 0)
            print(
                f"  Verdict: {verdict.upper()} | "
                f"avg={avg_stats.get('mean', 0):.1f} std={avg_stats.get('std', 0):.2f} | "
                f"feedback_similarity={fb_sim:.2f}"
            )
            print()

    # Build output
    output = {
        "run_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "model_judge": svc.world_quality._get_judge_model(),
            "model_judge_override": judge_model_override,
            "temperatures": temperatures,
            "judge_calls_per_entity": args.judge_calls,
            "entity_types": entity_types,
            "is_temperature_sweep": is_sweep,
            "thresholds": {
                "noisy": NOISY_STD_THRESHOLD,
                "consistent": CONSISTENT_STD_THRESHOLD,
            },
        },
        "results": results,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_path}")

    if is_sweep:
        # Temperature sweep summary and guidance
        print_temperature_sweep_summary(results)
        print_temperature_sweep_guidance(results, temperatures)
    else:
        # Single-temperature summary (backward-compatible)
        print("\n=== JUDGE CONSISTENCY SUMMARY ===")
        print(f"{'Type':<15} {'Verdict':<12} {'Mean':<8} {'Std':<8} {'Range':<12} {'FB Sim':<8}")
        print("-" * 63)
        for r in results:
            et = r["entity_type"]
            v = r.get("verdict", "error").upper()
            avg = r.get("average_stats", {})
            fb = r.get("feedback_similarity", 0)
            range_str = f"{avg.get('min', 0):.1f}-{avg.get('max', 0):.1f}"
            print(
                f"{et:<15} {v:<12} {avg.get('mean', 0):<8.1f} "
                f"{avg.get('std', 0):<8.2f} {range_str:<12} {fb:<8.2f}"
            )

        # Decision guidance
        print("\n=== DECISION GUIDANCE ===")
        noisy_types = [r["entity_type"] for r in results if r.get("verdict") == "noisy"]
        consistent_types = [r["entity_type"] for r in results if r.get("verdict") == "consistent"]
        borderline_types = [r["entity_type"] for r in results if r.get("verdict") == "borderline"]

        if noisy_types:
            print(f"NOISY judges ({', '.join(noisy_types)}): Fix judge first!")
            print("  -> Consider: multi-call averaging, lower temperature, or structured output")
        if consistent_types:
            print(
                f"CONSISTENT judges ({', '.join(consistent_types)}): "
                "Problem is in creation/refinement"
            )
            print("  -> Run evaluate_refinement.py to diagnose further")
        if borderline_types:
            print(f"BORDERLINE ({', '.join(borderline_types)}): Some noise but not critical")
            print("  -> May benefit from multi-call averaging but not blocking")


if __name__ == "__main__":
    main()
