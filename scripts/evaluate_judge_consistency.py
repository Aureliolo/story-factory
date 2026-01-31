#!/usr/bin/env python3
"""Evaluate judge scoring consistency for world building entities.

Submits the same entity to the judge N times to measure scoring variance.
Determines if the judge is noisy (std > 0.5) or consistent (std < 0.2).

Usage:
    python scripts/evaluate_judge_consistency.py [options]
      --entity-types faction,concept  (default: all 6)
      --judge-calls 5                 (default: 5)
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


def run_judge_consistency_test(
    svc_container: ServiceContainer,
    story_state: StoryState,
    entity_type: str,
    judge_calls: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run judge consistency test for one entity type.

    1. Generate one entity with _create_X()
    2. Call _judge_X_quality() N times on the frozen entity
    3. Compute statistics

    Args:
        svc_container: Service container.
        story_state: Story state.
        entity_type: Entity type.
        judge_calls: Number of judge calls.
        verbose: Verbose output.

    Returns:
        Dict with consistency test results.
    """
    config = svc_container.world_quality.get_config()

    result: dict[str, Any] = {
        "entity_type": entity_type,
        "entity_name": "",
        "entity_data": None,
        "judge_calls": judge_calls,
        "judge_temperature": config.judge_temperature,
        "individual_scores": [],
        "per_dimension_stats": {},
        "average_stats": {},
        "feedback_similarity": 0.0,
        "verdict": "",
        "error": None,
    }

    # Step 1: Create entity
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
            svc_container, story_state, entity_type, entity_data, config.judge_temperature
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

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"judge_consistency_{timestamp}.json"

    # Initialize services
    print("Loading settings and initializing services...")
    settings = Settings.load()
    svc = ServiceContainer(settings)
    config = svc.world_quality.get_config()

    # Create canonical story state
    brief = make_canonical_brief()
    story_state = make_story_state(brief)

    print(f"Judge temperature: {config.judge_temperature}")
    print(f"Judge calls per entity: {args.judge_calls}")
    print(f"Entity types: {entity_types}")
    print(f"Output: {output_path}")
    print()

    # Run consistency tests
    results: list[dict[str, Any]] = []
    for et in entity_types:
        print(f"--- Testing {et} judge consistency ({args.judge_calls} calls) ---")
        result = run_judge_consistency_test(svc, story_state, et, args.judge_calls, args.verbose)
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
            "judge_temperature": config.judge_temperature,
            "judge_calls_per_entity": args.judge_calls,
            "entity_types": entity_types,
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

    # Print summary table
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
            f"CONSISTENT judges ({', '.join(consistent_types)}): Problem is in creation/refinement"
        )
        print("  -> Run evaluate_refinement.py to diagnose further")
    if borderline_types:
        print(f"BORDERLINE ({', '.join(borderline_types)}): Some noise but not critical")
        print("  -> May benefit from multi-call averaging but not blocking")


if __name__ == "__main__":
    main()
