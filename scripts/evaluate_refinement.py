#!/usr/bin/env python3
"""Evaluate refinement loop effectiveness for world building entities.

Runs real refinement loops against a live Ollama model with full instrumentation
to diagnose WHY the quality refinement loop has a high fail rate.

Captures per entity per iteration:
- Entity data snapshot (full dict/model dump)
- Per-dimension scores and average
- Judge feedback string (verbatim)
- Description diff ratio vs previous iteration
- Temperature used
- Wall clock time

Usage:
    python scripts/evaluate_refinement.py [options]
      --entity-types faction,concept  (default: all 6)
      --count-per-type 3              (default: 3)
      --output results.json           (default: output/diagnostics/<timestamp>.json)
      --verbose
"""

import argparse
import difflib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.story_state import Character, StoryBrief, StoryState
from src.services import ServiceContainer
from src.settings import Settings
from src.utils.exceptions import StoryFactoryError

logger = logging.getLogger(__name__)

ALL_ENTITY_TYPES = ["character", "faction", "location", "item", "concept", "relationship"]

# Score dimension names per entity type (for reporting)
SCORE_DIMENSIONS: dict[str, list[str]] = {
    "character": ["depth", "goals", "flaws", "uniqueness", "arc_potential"],
    "faction": ["coherence", "influence", "conflict_potential", "distinctiveness"],
    "location": ["atmosphere", "significance", "story_relevance", "distinctiveness"],
    "item": ["significance", "uniqueness", "narrative_potential", "integration"],
    "concept": ["relevance", "depth", "manifestation", "resonance"],
    "relationship": ["tension", "dynamics", "story_potential", "authenticity"],
}


def make_canonical_brief() -> StoryBrief:
    """Create a canonical StoryBrief for reproducible diagnostics.

    Fantasy premise with political intrigue themes that exercises all entity types well.
    """
    return StoryBrief(
        premise=(
            "In a crumbling empire where magic is fueled by memory, a disgraced archivist "
            "discovers that the ruling council has been erasing collective memories to maintain "
            "power. She must navigate rival factions, ancient artifacts, and forbidden knowledge "
            "to restore what was lost before the empire collapses into civil war."
        ),
        genre="Fantasy",
        subgenres=["Political Intrigue", "Dark Fantasy"],
        tone="Dark and suspenseful with moments of wonder",
        themes=["Memory and identity", "Power and corruption", "Truth vs stability"],
        setting_time="Late imperial era, roughly equivalent to 18th century",
        setting_place="The Mnemorian Empire, a vast realm of stone cities and forgotten libraries",
        target_length="novel",
        language="English",
        content_rating="moderate",
        content_preferences=["Complex politics", "Morally grey characters", "Magic systems"],
        content_avoid=["Gratuitous violence", "Sexual content"],
        additional_notes="Focus on worldbuilding depth and faction dynamics.",
    )


def make_story_state(brief: StoryBrief) -> StoryState:
    """Create a minimal StoryState wrapping the canonical brief."""
    return StoryState(
        id="diagnostic-run",
        brief=brief,
        project_name="Refinement Diagnostic",
    )


def compute_description_diff(prev_desc: str, curr_desc: str) -> float:
    """Compute diff ratio between two description strings.

    Returns a float in [0, 1] where 0 means identical and 1 means completely different.
    Uses SequenceMatcher which gives a ratio of matching characters.
    We invert it: diff_ratio = 1.0 - similarity_ratio.

    Args:
        prev_desc: Previous description text.
        curr_desc: Current description text.

    Returns:
        Diff ratio where higher means more change.
    """
    if not prev_desc and not curr_desc:
        return 0.0
    similarity = difflib.SequenceMatcher(None, prev_desc, curr_desc).ratio()
    return round(1.0 - similarity, 4)


def extract_description(entity_data: dict[str, Any] | None, entity_type: str) -> str:
    """Extract the primary description field from entity data.

    Args:
        entity_data: Entity dictionary or None.
        entity_type: Type of entity.

    Returns:
        Description string, empty if not found.
    """
    if not entity_data:
        return ""
    desc = str(entity_data.get("description", ""))
    # For relationships, also include the relation type and targets
    if entity_type == "relationship":
        parts = [
            str(entity_data.get("source", "")),
            str(entity_data.get("target", "")),
            str(entity_data.get("relation_type", "")),
            desc,
        ]
        return " | ".join(p for p in parts if p)
    return desc


def compute_feedback_specificity(feedback: str, entity_data: dict[str, Any]) -> float:
    """Compute what fraction of feedback words reference entity-specific content.

    A rough heuristic: count words in the feedback that also appear in the entity's
    name or description, divided by total feedback words. Higher = more specific.

    Args:
        feedback: Judge feedback string.
        entity_data: Entity data dict.

    Returns:
        Float ratio of entity-specific words in feedback.
    """
    if not feedback:
        return 0.0
    feedback_words = set(feedback.lower().split())
    if not feedback_words:
        return 0.0

    # Collect entity-specific words from name, description, goals, values
    entity_text_parts = []
    for key in ["name", "description", "significance", "manifestations", "leader"]:
        val = entity_data.get(key, "")
        if isinstance(val, str):
            entity_text_parts.append(val)
    for key in ["goals", "values", "personality_traits", "properties"]:
        val = entity_data.get(key, [])
        if isinstance(val, list):
            entity_text_parts.extend(str(v) for v in val)

    entity_words = set(" ".join(entity_text_parts).lower().split())
    # Remove very common words that would inflate the count
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "has",
        "have",
        "had",
        "not",
        "no",
        "as",
        "if",
        "they",
        "their",
        "them",
    }
    entity_words -= stop_words
    feedback_words -= stop_words

    if not feedback_words:
        return 0.0

    overlap = feedback_words & entity_words
    return round(len(overlap) / len(feedback_words), 4)


def run_single_entity(
    svc_container: ServiceContainer,
    story_state: StoryState,
    entity_type: str,
    entity_index: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a full create-judge-refine loop for a single entity with instrumentation.

    Args:
        svc_container: Service container with WorldQualityService.
        story_state: Story state with canonical brief.
        entity_type: One of the 6 entity types.
        entity_index: Index for this entity (for logging).
        verbose: Whether to print verbose progress.

    Returns:
        Dict with full diagnostic data for this entity.
    """
    wqs = svc_container.world_quality
    config = wqs.get_config()
    existing_names: list[str] = []
    start_time = time.monotonic()

    result: dict[str, Any] = {
        "entity_type": entity_type,
        "entity_index": entity_index,
        "entity_name": "",
        "threshold_met": False,
        "iterations": [],
        "score_progression": [],
        "feedback_strings": [],
        "description_diff_ratios": [],
        "feedback_specificity_scores": [],
        "total_time_s": 0.0,
        "error": None,
    }

    prev_description = ""
    entity_data: dict[str, Any] | None = None
    scores = None

    for iteration in range(config.max_iterations):
        iter_start = time.monotonic()
        iter_record: dict[str, Any] = {
            "iteration": iteration + 1,
            "phase": "create" if iteration == 0 else "refine",
            "temperature": 0.0,
            "scores": {},
            "average_score": 0.0,
            "feedback": "",
            "description_diff_ratio": None,
            "feedback_specificity": 0.0,
            "wall_clock_s": 0.0,
            "entity_snapshot": None,
            "error": None,
        }

        try:
            # Phase 1: Create or Refine
            # Re-create if no entity data or no scores from previous judge failure
            if iteration == 0 or entity_data is None or scores is None:
                iter_record["phase"] = "create"
                iter_record["temperature"] = config.creator_temperature
                if entity_type == "character":
                    entity_obj = wqs._create_character(
                        story_state, existing_names, config.creator_temperature
                    )
                    entity_data = entity_obj.model_dump() if entity_obj else None
                elif entity_type == "faction":
                    entity_data = wqs._create_faction(
                        story_state, existing_names, config.creator_temperature
                    )
                elif entity_type == "location":
                    entity_data = wqs._create_location(
                        story_state, existing_names, config.creator_temperature
                    )
                elif entity_type == "item":
                    entity_data = wqs._create_item(
                        story_state, existing_names, config.creator_temperature
                    )
                elif entity_type == "concept":
                    entity_data = wqs._create_concept(
                        story_state, existing_names, config.creator_temperature
                    )
                elif entity_type == "relationship":
                    entity_names = ["Archivist Sera", "Councillor Vex", "The Pale Librarian"]
                    entity_data = wqs._create_relationship(
                        story_state, entity_names, [], config.creator_temperature
                    )
            else:
                dynamic_temp = config.get_refinement_temperature(iteration + 1)
                iter_record["temperature"] = dynamic_temp
                iter_record["phase"] = "refine"

                if entity_type == "character":
                    entity_obj = wqs._refine_character(
                        Character(**entity_data), scores, story_state, dynamic_temp
                    )
                    entity_data = entity_obj.model_dump() if entity_obj else None
                elif entity_type == "faction":
                    entity_data = wqs._refine_faction(
                        entity_data, scores, story_state, dynamic_temp
                    )
                elif entity_type == "location":
                    entity_data = wqs._refine_location(
                        entity_data, scores, story_state, dynamic_temp
                    )
                elif entity_type == "item":
                    entity_data = wqs._refine_item(entity_data, scores, story_state, dynamic_temp)
                elif entity_type == "concept":
                    entity_data = wqs._refine_concept(
                        entity_data, scores, story_state, dynamic_temp
                    )
                elif entity_type == "relationship":
                    entity_data = wqs._refine_relationship(
                        entity_data, scores, story_state, dynamic_temp
                    )

            if not entity_data or (
                isinstance(entity_data, dict)
                and not entity_data.get("name")
                and entity_type != "relationship"
            ):
                iter_record["error"] = "Entity creation returned empty"
                iter_record["wall_clock_s"] = round(time.monotonic() - iter_start, 2)
                result["iterations"].append(iter_record)
                logger.warning(
                    "Entity %s #%d iteration %d: creation returned empty",
                    entity_type,
                    entity_index,
                    iteration + 1,
                )
                continue

            # Set entity name on first successful creation
            if not result["entity_name"]:
                name = entity_data.get("name", "")
                if entity_type == "relationship":
                    name = f"{entity_data.get('source', '')} -> {entity_data.get('target', '')}"
                result["entity_name"] = name

            iter_record["entity_snapshot"] = (
                entity_data.copy() if isinstance(entity_data, dict) else entity_data
            )

            # Compute description diff
            curr_description = extract_description(entity_data, entity_type)
            if iteration == 0:
                iter_record["description_diff_ratio"] = None
            else:
                diff_ratio = compute_description_diff(prev_description, curr_description)
                iter_record["description_diff_ratio"] = diff_ratio
            prev_description = curr_description

            # Phase 2: Judge
            if entity_type == "character":
                scores = wqs._judge_character_quality(
                    Character(**entity_data), story_state, config.judge_temperature
                )
            elif entity_type == "faction":
                scores = wqs._judge_faction_quality(
                    entity_data, story_state, config.judge_temperature
                )
            elif entity_type == "location":
                scores = wqs._judge_location_quality(
                    entity_data, story_state, config.judge_temperature
                )
            elif entity_type == "item":
                scores = wqs._judge_item_quality(entity_data, story_state, config.judge_temperature)
            elif entity_type == "concept":
                scores = wqs._judge_concept_quality(
                    entity_data, story_state, config.judge_temperature
                )
            elif entity_type == "relationship":
                scores = wqs._judge_relationship_quality(
                    entity_data, story_state, config.judge_temperature
                )

            if scores is None:
                iter_record["error"] = f"No judge for entity type: {entity_type}"
                iter_record["wall_clock_s"] = round(time.monotonic() - iter_start, 2)
                result["iterations"].append(iter_record)
                logger.warning(
                    "Entity %s #%d iteration %d: judge returned None",
                    entity_type,
                    entity_index,
                    iteration + 1,
                )
                continue

            score_dict = scores.to_dict()
            avg = scores.average
            feedback = scores.feedback

            iter_record["scores"] = {
                k: v for k, v in score_dict.items() if k not in ("average", "feedback")
            }
            iter_record["average_score"] = round(avg, 2)
            iter_record["feedback"] = feedback

            # Feedback specificity
            specificity = compute_feedback_specificity(feedback, entity_data)
            iter_record["feedback_specificity"] = specificity

            iter_record["wall_clock_s"] = round(time.monotonic() - iter_start, 2)
            result["iterations"].append(iter_record)
            result["score_progression"].append(round(avg, 2))
            result["feedback_strings"].append(feedback)
            result["description_diff_ratios"].append(iter_record["description_diff_ratio"])
            result["feedback_specificity_scores"].append(specificity)

            if verbose:
                print(
                    f"  Iter {iteration + 1}: avg={avg:.1f} "
                    f"diff={iter_record['description_diff_ratio']} "
                    f"time={iter_record['wall_clock_s']}s"
                )

            # Check threshold
            if avg >= config.quality_threshold:
                result["threshold_met"] = True
                logger.info(
                    "Entity %s '%s' met threshold at iteration %d (%.1f >= %.1f)",
                    entity_type,
                    result["entity_name"],
                    iteration + 1,
                    avg,
                    config.quality_threshold,
                )
                break

        except StoryFactoryError as e:
            iter_record["error"] = str(e)
            iter_record["wall_clock_s"] = round(time.monotonic() - iter_start, 2)
            result["iterations"].append(iter_record)
            logger.error(
                "Entity %s #%d iteration %d error: %s",
                entity_type,
                entity_index,
                iteration + 1,
                e,
            )

    result["total_time_s"] = round(time.monotonic() - start_time, 2)
    return result


def compute_summary(entity_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from entity results.

    Args:
        entity_results: List of per-entity diagnostic results.

    Returns:
        Summary dict with aggregated metrics per entity type.
    """
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in entity_results:
        et = r["entity_type"]
        by_type.setdefault(et, []).append(r)

    summary: dict[str, Any] = {
        "pass_rate_by_type": {},
        "avg_score_improvement_per_iteration": {},
        "avg_description_diff_ratio": {},
        "avg_feedback_length": {},
        "avg_feedback_specificity": {},
        "plateau_rate": {},
        "regression_rate": {},
    }

    for et, results in by_type.items():
        total = len(results)
        passed = sum(1 for r in results if r["threshold_met"])
        summary["pass_rate_by_type"][et] = round(passed / total, 2) if total else 0.0

        # Score improvement per iteration
        improvements = []
        for r in results:
            prog = r["score_progression"]
            for i in range(1, len(prog)):
                improvements.append(prog[i] - prog[i - 1])
        summary["avg_score_improvement_per_iteration"][et] = (
            round(sum(improvements) / len(improvements), 3) if improvements else 0.0
        )

        # Description diff ratio (skip first None)
        diffs = []
        for r in results:
            for d in r["description_diff_ratios"]:
                if d is not None:
                    diffs.append(d)
        summary["avg_description_diff_ratio"][et] = (
            round(sum(diffs) / len(diffs), 3) if diffs else 0.0
        )

        # Feedback length
        fb_lengths = []
        for r in results:
            for fb in r["feedback_strings"]:
                fb_lengths.append(len(fb.split()))
        summary["avg_feedback_length"][et] = (
            round(sum(fb_lengths) / len(fb_lengths), 1) if fb_lengths else 0.0
        )

        # Feedback specificity
        specificities = []
        for r in results:
            specificities.extend(r["feedback_specificity_scores"])
        summary["avg_feedback_specificity"][et] = (
            round(sum(specificities) / len(specificities), 3) if specificities else 0.0
        )

        # Plateau rate: iterations where |delta| < 0.3 / total refinement iterations
        plateau_count = 0
        refinement_iters = 0
        for r in results:
            prog = r["score_progression"]
            for i in range(1, len(prog)):
                refinement_iters += 1
                if abs(prog[i] - prog[i - 1]) < 0.3:
                    plateau_count += 1
        summary["plateau_rate"][et] = (
            round(plateau_count / refinement_iters, 3) if refinement_iters else 0.0
        )

        # Regression rate: iterations where delta < 0 / total refinement iterations
        regression_count = 0
        for r in results:
            prog = r["score_progression"]
            for i in range(1, len(prog)):
                if prog[i] < prog[i - 1]:
                    regression_count += 1
        summary["regression_rate"][et] = (
            round(regression_count / refinement_iters, 3) if refinement_iters else 0.0
        )

    return summary


def main() -> None:
    """Main entry point for the refinement evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate refinement loop effectiveness for world building entities."
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        help=f"Comma-separated entity types (default: all). Options: {', '.join(ALL_ENTITY_TYPES)}",
    )
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=3,
        help="Number of entities to generate per type (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: output/diagnostics/<timestamp>.json)",
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
            print(f"Valid types: {ALL_ENTITY_TYPES}")
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
        output_path = diagnostics_dir / f"refinement_{timestamp}.json"

    # Initialize services
    print("Loading settings and initializing services...")
    settings = Settings.load()
    svc = ServiceContainer(settings)
    config = svc.world_quality.get_config()

    # Create canonical story state
    brief = make_canonical_brief()
    story_state = make_story_state(brief)

    print(f"Configuration: threshold={config.quality_threshold}, max_iter={config.max_iterations}")
    print(f"Entity types: {entity_types}")
    print(f"Count per type: {args.count_per_type}")
    print(f"Output: {output_path}")
    print()

    # Run evaluations
    entity_results: list[dict[str, Any]] = []
    total = len(entity_types) * args.count_per_type
    completed = 0

    for et in entity_types:
        print(f"--- Evaluating {et} ({args.count_per_type} entities) ---")
        for i in range(args.count_per_type):
            completed += 1
            print(f"  [{completed}/{total}] {et} #{i + 1}...")
            result = run_single_entity(svc, story_state, et, i + 1, verbose=args.verbose)
            entity_results.append(result)
            name = result["entity_name"] or "(unnamed)"
            passed = "PASS" if result["threshold_met"] else "FAIL"
            scores = result["score_progression"]
            score_str = " -> ".join(f"{s:.1f}" for s in scores) if scores else "no scores"
            print(f"    {passed}: '{name}' scores: {score_str} ({result['total_time_s']}s)")

    # Compute summary
    summary = compute_summary(entity_results)

    # Build output
    output = {
        "run_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "model_creator": svc.world_quality._get_creator_model(),
            "model_judge": svc.world_quality._get_judge_model(),
            "config": {
                "quality_threshold": config.quality_threshold,
                "max_iterations": config.max_iterations,
                "creator_temperature": config.creator_temperature,
                "judge_temperature": config.judge_temperature,
                "refinement_temp_start": config.refinement_temp_start,
                "refinement_temp_end": config.refinement_temp_end,
                "refinement_temp_decay": config.refinement_temp_decay,
            },
            "brief_premise": brief.premise[:200],
            "entity_types": entity_types,
            "count_per_type": args.count_per_type,
        },
        "entity_results": entity_results,
        "summary": summary,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(
        f"{'Type':<15} {'Pass%':<8} {'Avg Imp':<10} {'Diff Ratio':<12} "
        f"{'Plateau%':<10} {'Regress%':<10}"
    )
    print("-" * 65)
    for et in entity_types:
        pr = summary["pass_rate_by_type"].get(et, 0)
        ai = summary["avg_score_improvement_per_iteration"].get(et, 0)
        dr = summary["avg_description_diff_ratio"].get(et, 0)
        plat = summary["plateau_rate"].get(et, 0)
        reg = summary["regression_rate"].get(et, 0)
        print(f"{et:<15} {pr:<8.0%} {ai:<10.3f} {dr:<12.3f} {plat:<10.1%} {reg:<10.1%}")


if __name__ == "__main__":
    main()
