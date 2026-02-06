#!/usr/bin/env python3
"""Evaluate different judge calibration block variants to find optimal scoring behavior.

Tests how different calibration strategies affect score distribution, quality
differentiation, and whether entities can reach configurable thresholds.

Uses ground-truth samples from evaluate_judge_accuracy.py to ensure we can
measure whether the judge still differentiates quality tiers correctly.

Calls the same Ollama API + instructor pipeline as production code to ensure
results reflect real behavior.

Usage:
    python scripts/evaluate_calibration_variants.py [options]
      --variants A,B,C,D,E,F       (default: all)
      --judge-calls 3               (default: 3 per sample per variant)
      --temperature 0.1             (judge temperature, default: 0.1)
      --output results.json         (default: output/diagnostics/<timestamp>_calibration.json)
      --verbose
"""

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.world_quality import (
    CharacterQualityScores,
    FactionQualityScores,
)
from src.services import ServiceContainer
from src.services.llm_client import generate_structured
from src.settings import Settings

logger = logging.getLogger(__name__)

# Suppress noisy HTTP library debug logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =====================================================================
# Calibration Variants
# =====================================================================

# Pre-#246 production calibration block (replaced by D_minimal in PR #261)
VARIANT_A_LEGACY = """SCORING GUIDE — USE THE FULL 0-10 RANGE WITH DECIMALS:
- 1-3: Fundamentally broken or generic (contradictory, cliched, no thought)
- 4-5: Below average (functional but bland, forgettable, one-dimensional)
- 6-7: Competent (clear strengths, some areas need work — most first drafts land here)
- 7-8: Strong (well-crafted, multiple strong dimensions — refined work reaches here)
- 8-9: Excellent (genuinely impressive, few weaknesses — justify in feedback)
- 10: Virtually flawless (publishable as-is — almost never appropriate)

RULES:
1. Score to one decimal place (e.g., 5.3, 7.1, 8.6). Do NOT round to whole numbers.
2. Differentiate between dimensions — an entity can have 8.2 in one area but 5.4 in another.
3. If you give 8+ on a dimension, your feedback MUST explain what makes it exceptional.
4. If all your scores are within 1 point of each other, you are not differentiating enough."""

VARIANT_B_SOFTENED = """SCORING GUIDE — USE THE FULL 0-10 RANGE WITH DECIMALS:
- 1-3: Fundamentally broken or generic (contradictory, cliched, no thought)
- 4-5: Below average (functional but bland, forgettable, one-dimensional)
- 6-7: Competent (clear strengths, some areas need work)
- 7-8: Strong (well-crafted, multiple strong dimensions)
- 8-9: Excellent (genuinely impressive, few weaknesses)
- 10: Virtually flawless (almost never appropriate)

RULES:
1. Score to one decimal place (e.g., 6.4, 8.2, 9.1). Do NOT round to whole numbers.
2. Differentiate between dimensions — an entity can have 8.2 in one area but 5.4 in another.
3. Use 8+ scores when deserved — strong work should score in the 7-9 range.
4. If all your scores are within 1 point of each other, you are not differentiating enough."""

VARIANT_C_NO_CALIBRATION = ""

VARIANT_D_MINIMAL = """Score each dimension 0-10 with one decimal place.
Differentiate between dimensions — scores should vary based on actual quality."""

VARIANT_E_NO_TIERS = """SCORING GUIDE — USE THE FULL 0-10 RANGE WITH DECIMALS:

RULES:
1. Score to one decimal place (e.g., 6.4, 8.2, 9.1). Do NOT round to whole numbers.
2. Differentiate between dimensions — an entity can have 8.2 in one area but 5.4 in another.
3. Use the full 0-10 range. Do not cluster all scores around the same value.
4. If all your scores are within 1 point of each other, you are not differentiating enough."""

VARIANT_F_ENCOURAGING = """SCORING GUIDE — USE THE FULL 0-10 RANGE WITH DECIMALS:
- 1-4: Broken, contradictory, or extremely generic
- 5-6: Generic or bland (forgettable, one-dimensional)
- 7-9: Most well-crafted entities should score in this range
- 10: Virtually flawless (almost never appropriate)

RULES:
1. Score to one decimal place (e.g., 7.3, 8.1, 9.2). Do NOT round to whole numbers.
2. Differentiate between dimensions — an entity can have 8.5 in one area but 6.2 in another.
3. Well-developed entities should score 7-9. Reserve sub-7 for genuinely weak dimensions.
4. If all your scores are within 1 point of each other, you are not differentiating enough."""

CALIBRATION_VARIANTS: dict[str, str] = {
    "A_legacy": VARIANT_A_LEGACY,
    "B_softened": VARIANT_B_SOFTENED,
    "C_no_calibration": VARIANT_C_NO_CALIBRATION,
    "D_minimal": VARIANT_D_MINIMAL,
    "E_no_tiers": VARIANT_E_NO_TIERS,
    "F_encouraging": VARIANT_F_ENCOURAGING,
}

# =====================================================================
# Ground-Truth Samples (subset from evaluate_judge_accuracy.py)
# 3 quality tiers x 2 entity types = 6 samples
# =====================================================================

ENTITY_DIMENSIONS: dict[str, dict[str, str]] = {
    "faction": {
        "coherence": "Internal consistency, clear structure",
        "influence": "World impact, power level",
        "conflict_potential": "Story conflict opportunities",
        "distinctiveness": "Memorable, unique qualities",
    },
    "character": {
        "depth": "Psychological complexity, internal contradictions, layers",
        "goal_clarity": "Clarity, story relevance, want vs need tension",
        "flaws": "Meaningful vulnerabilities that drive conflict",
        "uniqueness": "Distinctiveness from genre archetypes",
        "arc_potential": "Room for transformation and growth",
    },
}

SCORE_MODELS: dict[str, type] = {
    "faction": FactionQualityScores,
    "character": CharacterQualityScores,
}

JUDGE_ROLES: dict[str, str] = {
    "faction": "You are a strict quality judge evaluating a faction for a Fantasy story.",
    "character": "You are a literary critic evaluating character quality for a Fantasy story.",
}

SAMPLES: list[dict[str, Any]] = [
    # FACTION — Terrible (ground truth avg ~2.75)
    {
        "id": "faction_terrible",
        "entity_type": "faction",
        "tier": "terrible",
        "entity_block": (
            "FACTION TO EVALUATE:\n"
            "Name: The Shadow Legion\n"
            "Description: An army of darkness that serves the Dark Lord. They are evil "
            "and want to destroy everything good in the world.\n"
            "Leader: The Dark Lord\n"
            "Goals: Conquer the world, Destroy all good\n"
            "Values: Power, Fear"
        ),
        "ground_truth_avg": 2.75,
    },
    # FACTION — Mediocre (ground truth avg ~5.0)
    {
        "id": "faction_mediocre",
        "entity_type": "faction",
        "tier": "mediocre",
        "entity_block": (
            "FACTION TO EVALUATE:\n"
            "Name: The Merchant Consortium\n"
            "Description: A guild of traders and merchants who control trade routes "
            "across the empire. They seek profit above all else and maintain "
            "a network of warehouses and trading posts throughout major cities.\n"
            "Leader: Guildmaster Aldric\n"
            "Goals: Maintain their trade monopoly, Expand into new markets\n"
            "Values: Profit, Business acumen"
        ),
        "ground_truth_avg": 5.0,
    },
    # FACTION — Excellent (ground truth avg ~8.4)
    {
        "id": "faction_excellent",
        "entity_type": "faction",
        "tier": "excellent",
        "entity_block": (
            "FACTION TO EVALUATE:\n"
            "Name: The Veilkeepers\n"
            "Description: A fractured order of memory mages who guard the boundary "
            "between remembered and forgotten truths. Originally founded "
            "to preserve the empire's collective memory, they split into "
            "two secret internal factions: the Archivists who believe all "
            "memories must be preserved regardless of consequence, and the "
            "Censors who argue some truths are too dangerous to remember. "
            "This internal schism mirrors the empire's own struggle between "
            "truth and stability.\n"
            "Leader: The Divided Council — twin sisters Mira (Archivist) and "
            "Sera (Censor) who hold opposing chairs\n"
            "Goals: Protect the Veil of Remembrance from collapse, "
            "Resolve their internal schism before it destroys the order, "
            "Prevent the ruling council from weaponizing forgotten memories\n"
            "Values: Memory as sacred trust, Knowledge tempered by responsibility, "
            "The burden of truth-keeping"
        ),
        "ground_truth_avg": 8.4,
    },
    # CHARACTER — Terrible (ground truth avg ~2.0)
    {
        "id": "character_terrible",
        "entity_type": "character",
        "tier": "terrible",
        "entity_block": (
            "CHARACTER TO EVALUATE:\n"
            "Name: Aldric the Brave\n"
            "Role: protagonist\n"
            "Description: A brave warrior who fights for justice. He is strong, handsome, "
            "and everyone likes him. He never gives up and always does the right thing.\n"
            "Traits: brave, strong, kind, heroic\n"
            "Goals: Defeat the evil villain, Save the kingdom\n"
            "Arc Notes: Aldric starts brave and becomes even braver through his journey."
        ),
        "ground_truth_avg": 2.0,
    },
    # CHARACTER — Mediocre (ground truth avg ~5.4)
    {
        "id": "character_mediocre",
        "entity_type": "character",
        "tier": "mediocre",
        "entity_block": (
            "CHARACTER TO EVALUATE:\n"
            "Name: Commander Thessa\n"
            "Role: supporting\n"
            "Description: A seasoned military commander who leads the city guard. She is "
            "disciplined and follows orders without question, but struggles "
            "with the moral implications of enforcing increasingly harsh laws. "
            "She lost her family in the last war.\n"
            "Traits: disciplined, loyal, haunted, pragmatic\n"
            "Goals: Keep the city safe at any cost, Find meaning after personal loss\n"
            "Arc Notes: Thessa must choose between duty and conscience when ordered to "
            "suppress a civilian uprising she secretly sympathizes with."
        ),
        "ground_truth_avg": 5.4,
    },
    # CHARACTER — Excellent (ground truth avg ~8.7)
    {
        "id": "character_excellent",
        "entity_type": "character",
        "tier": "excellent",
        "entity_block": (
            "CHARACTER TO EVALUATE:\n"
            "Name: Vesper\n"
            "Role: antagonist\n"
            "Description: A former healer who discovered she could absorb others' traumatic "
            "memories to cure their suffering — but each absorbed memory slowly "
            "overwrites her own identity. Now more composite than individual, she "
            "leads a cult of willing 'donors' who worship the peace she brings, "
            "while secretly terrified that the person she was no longer exists. "
            "She opposes the protagonist not out of malice but because restoring "
            "erased memories would undo the peace she gave thousands of followers.\n"
            "Traits: compassionate to the point of self-destruction, "
            "manipulative through genuine care, identity-fractured, philosophically rigid\n"
            "Goals: Protect her followers' peace at any cost, "
            "Find a way to remember who she was before the absorptions, "
            "Prevent memory restoration that would re-traumatize thousands\n"
            "Arc Notes: Vesper's arc explores whether identity is what you remember or "
            "what you choose. Her climax forces her to decide between keeping "
            "her followers' donated peace or returning their pain alongside "
            "their truth — knowing she'll lose herself either way."
        ),
        "ground_truth_avg": 8.7,
    },
]


def build_judge_prompt(
    sample: dict[str, Any],
    calibration_block: str,
) -> str:
    """Build a judge prompt for a sample entity with a specific calibration variant.

    Mirrors the production prompt structure exactly:
    1. Role line
    2. Entity block
    3. Calibration block (the variable being tested)
    4. Dimension descriptions
    5. Feedback instruction
    6. Output format

    Args:
        sample: Sample with entity_type, entity_block.
        calibration_block: The calibration variant text to use.

    Returns:
        Complete judge prompt string.
    """
    entity_type = sample["entity_type"]
    role_line = JUDGE_ROLES[entity_type]
    entity_block = sample["entity_block"]

    dims = ENTITY_DIMENSIONS[entity_type]
    dim_lines = "\n".join(f"- {k}: {v}" for k, v in dims.items())

    # Output format matches production: parametric <float 0-10> placeholders
    dim_placeholders = ", ".join(f'"{k}": <float 0-10>' for k in dims)
    output_format = (
        f"OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:\n"
        f'{{{dim_placeholders}, "feedback": "Your assessment..."}}\n\n'
        f'DO NOT wrap in "properties" or "description" - return ONLY the flat '
        f"scores object with YOUR OWN assessment."
    )

    # Assemble in same order as production (_character.py:267, _faction.py:253)
    parts = [role_line, "", entity_block]
    if calibration_block:
        parts.extend(["", calibration_block])
    parts.extend(
        [
            "",
            "Rate each dimension 0-10:",
            dim_lines,
            "",
            "Provide specific, actionable feedback for improvement in the feedback field.",
            "",
            output_format,
        ]
    )

    return "\n".join(parts)


def judge_sample(
    settings: Settings,
    judge_model: str,
    sample: dict[str, Any],
    calibration_block: str,
    temperature: float,
) -> dict[str, Any] | None:
    """Judge a single sample with a specific calibration variant via production pipeline.

    Uses generate_structured() with the same Pydantic models as production,
    ensuring the instructor middleware and JSON mode behavior match exactly.

    Args:
        settings: App settings for generate_structured.
        judge_model: Model name to use.
        sample: Ground truth sample.
        calibration_block: Calibration variant text.
        temperature: Judge temperature.

    Returns:
        Dict with dimension scores + average + feedback, or None on failure.
    """
    entity_type = sample["entity_type"]
    score_model = SCORE_MODELS[entity_type]
    prompt = build_judge_prompt(sample, calibration_block)

    try:
        result: Any = generate_structured(
            settings=settings,
            model=judge_model,
            prompt=prompt,
            response_model=score_model,
            temperature=temperature,
        )
        score_dict = result.to_dict()
        return {
            "dimensions": {k: v for k, v in score_dict.items() if k not in ("average", "feedback")},
            "average": round(result.average, 2),
            "feedback": result.feedback,
        }
    except Exception as e:
        logger.warning("Judge call failed for %s: %s", sample["id"], e)
        return None


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of floats.

    Args:
        values: Numeric values.

    Returns:
        Dict with mean, std, min, max.
    """
    if not values:
        # Intentional: return zeros for empty input so aggregation code can
        # proceed without special-casing missing data from failed judge calls.
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def compute_rank_correlation(pairs: list[tuple[float, float]]) -> float:
    """Spearman rank correlation for ground-truth vs predicted averages.

    Args:
        pairs: List of (ground_truth, predicted) tuples.

    Returns:
        Spearman rho in [-1, 1], or 0.0 if insufficient data.
    """
    if len(pairs) < 3:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        """Convert values to fractional ranks, averaging ties."""
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_indices):
            j = i
            while (
                j < len(sorted_indices) - 1
                and values[sorted_indices[j]] == values[sorted_indices[j + 1]]
            ):
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    gt_vals = [p[0] for p in pairs]
    pred_vals = [p[1] for p in pairs]
    gt_ranks = _rank(gt_vals)
    pred_ranks = _rank(pred_vals)

    n = len(pairs)
    d_sq_sum = sum((g - p) ** 2 for g, p in zip(gt_ranks, pred_ranks, strict=True))
    rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
    return round(rho, 3)


def run_variant(
    settings: Settings,
    judge_model: str,
    variant_name: str,
    calibration_block: str,
    judge_calls: int,
    temperature: float,
    verbose: bool,
) -> dict[str, Any]:
    """Run all samples through a single calibration variant.

    Args:
        settings: App settings.
        judge_model: Judge model name.
        variant_name: Variant identifier.
        calibration_block: The calibration text.
        judge_calls: Number of judge calls per sample.
        temperature: Judge temperature.
        verbose: Print progress.

    Returns:
        Dict with per-sample results and aggregate metrics.
    """
    variant_result: dict[str, Any] = {
        "variant": variant_name,
        "sample_results": [],
        "aggregate": {},
    }

    all_averages_by_tier: dict[str, list[float]] = {"terrible": [], "mediocre": [], "excellent": []}
    rank_pairs: list[tuple[float, float]] = []
    all_dim_spreads: list[float] = []

    for sample in SAMPLES:
        sid = sample["id"]
        tier = sample["tier"]
        gt_avg = sample["ground_truth_avg"]

        call_results: list[dict[str, Any]] = []
        call_averages: list[float] = []

        for call_idx in range(judge_calls):
            call_start = time.monotonic()
            result = judge_sample(settings, judge_model, sample, calibration_block, temperature)
            call_time = round(time.monotonic() - call_start, 1)

            if result:
                call_results.append(result)
                call_averages.append(result["average"])

                if verbose:
                    dims_str = ", ".join(f"{k}={v}" for k, v in result["dimensions"].items())
                    print(
                        f"      {sid} call {call_idx + 1}/{judge_calls}: "
                        f"avg={result['average']:.1f} ({dims_str}) [{call_time}s]"
                    )
            else:
                if verbose:
                    print(f"      {sid} call {call_idx + 1}/{judge_calls}: FAILED [{call_time}s]")

        if not call_averages:
            variant_result["sample_results"].append(
                {
                    "sample_id": sid,
                    "tier": tier,
                    "gt_avg": gt_avg,
                    "error": "All calls failed",
                }
            )
            continue

        avg_stats = compute_statistics(call_averages)

        # Per-dimension spread within this sample (average across calls)
        if call_results:
            dim_values: dict[str, list[float]] = {}
            for cr in call_results:
                for k, v in cr["dimensions"].items():
                    dim_values.setdefault(k, []).append(float(v))
            dim_means = {k: sum(vs) / len(vs) for k, vs in dim_values.items()}
            if len(dim_means) >= 2:
                dim_spread = max(dim_means.values()) - min(dim_means.values())
                all_dim_spreads.append(dim_spread)
            else:
                dim_spread = 0.0
        else:
            dim_spread = 0.0

        sample_result = {
            "sample_id": sid,
            "tier": tier,
            "gt_avg": gt_avg,
            "predicted_avg": avg_stats["mean"],
            "predicted_std": avg_stats["std"],
            "predicted_range": f"{avg_stats['min']}-{avg_stats['max']}",
            "dim_spread": round(dim_spread, 2),
            "calls_ok": len(call_averages),
            "calls_total": judge_calls,
            "individual_calls": call_results,
        }
        variant_result["sample_results"].append(sample_result)

        all_averages_by_tier[tier].append(avg_stats["mean"])
        rank_pairs.append((gt_avg, avg_stats["mean"]))

    # Aggregate metrics
    terrible_avg = (
        sum(all_averages_by_tier["terrible"]) / len(all_averages_by_tier["terrible"])
        if all_averages_by_tier["terrible"]
        else 0
    )
    mediocre_avg = (
        sum(all_averages_by_tier["mediocre"]) / len(all_averages_by_tier["mediocre"])
        if all_averages_by_tier["mediocre"]
        else 0
    )
    excellent_avg = (
        sum(all_averages_by_tier["excellent"]) / len(all_averages_by_tier["excellent"])
        if all_averages_by_tier["excellent"]
        else 0
    )

    # Quality ordering: terrible < mediocre < excellent?
    ordering_correct = terrible_avg < mediocre_avg < excellent_avg

    # Score gap between tiers
    gap_terrible_mediocre = round(mediocre_avg - terrible_avg, 2)
    gap_mediocre_excellent = round(excellent_avg - mediocre_avg, 2)

    # Would threshold=7.5 work? Would threshold=8.0 work?
    excellent_above_7_5 = excellent_avg >= 7.5
    excellent_above_8_0 = excellent_avg >= 8.0
    mediocre_below_7_5 = mediocre_avg < 7.5
    mediocre_below_8_0 = mediocre_avg < 8.0

    variant_result["aggregate"] = {
        "avg_terrible": round(terrible_avg, 2),
        "avg_mediocre": round(mediocre_avg, 2),
        "avg_excellent": round(excellent_avg, 2),
        "ordering_correct": ordering_correct,
        "rank_correlation": compute_rank_correlation(rank_pairs),
        "gap_terrible_mediocre": gap_terrible_mediocre,
        "gap_mediocre_excellent": gap_mediocre_excellent,
        "avg_dim_spread": (
            round(sum(all_dim_spreads) / len(all_dim_spreads), 2) if all_dim_spreads else 0
        ),
        "threshold_analysis": {
            "7.5": {
                "excellent_passes": excellent_above_7_5,
                "mediocre_blocked": mediocre_below_7_5,
                "workable": excellent_above_7_5 and mediocre_below_7_5,
            },
            "8.0": {
                "excellent_passes": excellent_above_8_0,
                "mediocre_blocked": mediocre_below_8_0,
                "workable": excellent_above_8_0 and mediocre_below_8_0,
            },
        },
    }

    return variant_result


def main() -> None:
    """Main entry point for the calibration variant evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate different judge calibration block variants."
    )
    parser.add_argument(
        "--variants",
        type=str,
        help=(
            "Comma-separated variant names "
            f"(default: all). Options: {', '.join(CALIBRATION_VARIANTS.keys())}"
        ),
    )
    parser.add_argument(
        "--judge-calls",
        type=int,
        default=3,
        help="Number of judge calls per sample per variant (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Judge temperature (default: 0.1 for reproducibility)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-call results",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse variants
    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
        invalid = [v for v in variant_names if v not in CALIBRATION_VARIANTS]
        if invalid:
            print(f"ERROR: Invalid variants: {invalid}")
            print(f"Valid: {list(CALIBRATION_VARIANTS.keys())}")
            sys.exit(1)
    else:
        variant_names = list(CALIBRATION_VARIANTS.keys())

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"calibration_variants_{timestamp}.json"

    # Initialize services (to get the judge model)
    print("Loading settings and initializing services...")
    settings = Settings.load()
    svc = ServiceContainer(settings)
    judge_model = svc.world_quality._get_judge_model()
    judge_temp = args.temperature

    print("=" * 80)
    print("CALIBRATION VARIANT BENCHMARK")
    print("=" * 80)
    print(f"Judge model: {judge_model}")
    print(f"Judge temperature: {judge_temp}")
    print(f"Judge calls per sample: {args.judge_calls}")
    print(f"Samples: {len(SAMPLES)}")
    print(f"Variants: {variant_names}")
    print(f"Total calls: {len(variant_names) * len(SAMPLES) * args.judge_calls}")
    print(f"Output: {output_path}")
    print("=" * 80)
    print()

    # Run benchmark
    all_variant_results: list[dict[str, Any]] = []
    overall_start = time.monotonic()

    for vi, variant_name in enumerate(variant_names):
        calibration_block = CALIBRATION_VARIANTS[variant_name]
        block_preview = calibration_block[:80].replace("\n", " ") if calibration_block else "(none)"
        print(f"[{vi + 1}/{len(variant_names)}] Variant: {variant_name}")
        print(f"  Calibration: {block_preview}...")
        print()

        variant_start = time.monotonic()
        result = run_variant(
            settings,
            judge_model,
            variant_name,
            calibration_block,
            args.judge_calls,
            judge_temp,
            args.verbose,
        )
        variant_time = round(time.monotonic() - variant_start, 1)

        agg = result["aggregate"]
        print(
            f"  Results: terrible={agg['avg_terrible']:.1f}  mediocre={agg['avg_mediocre']:.1f}  "
            f"excellent={agg['avg_excellent']:.1f}  "
            f"ordering={'OK' if agg['ordering_correct'] else 'BROKEN'}  "
            f"rank_corr={agg['rank_correlation']:.2f}  "
            f"dim_spread={agg['avg_dim_spread']:.1f}  "
            f"[{variant_time}s]"
        )
        print(
            f"  Threshold 7.5: {'WORKABLE' if agg['threshold_analysis']['7.5']['workable'] else 'BROKEN'}  "
            f"Threshold 8.0: {'WORKABLE' if agg['threshold_analysis']['8.0']['workable'] else 'BROKEN'}"
        )
        print()

        all_variant_results.append(result)

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build output
    output = {
        "benchmark_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "judge_model": judge_model,
            "judge_temperature": judge_temp,
            "judge_calls_per_sample": args.judge_calls,
            "samples": len(SAMPLES),
            "variants_tested": variant_names,
        },
        "variant_results": all_variant_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_path}")

    # Summary table
    print()
    print("=" * 120)
    print("SUMMARY — CALIBRATION VARIANT BENCHMARK")
    print("=" * 120)
    print(
        f"{'Variant':<20} {'Terrible':>8} {'Mediocre':>9} {'Excellent':>9} "
        f"{'Order':>6} {'Rank':>6} {'DimSprd':>7} "
        f"{'Gap T-M':>7} {'Gap M-E':>7} {'T=7.5':>6} {'T=8.0':>6}"
    )
    print("-" * 120)

    for result in all_variant_results:
        v = result["variant"]
        a = result["aggregate"]
        order = "OK" if a["ordering_correct"] else "BAD"
        t75 = "OK" if a["threshold_analysis"]["7.5"]["workable"] else "FAIL"
        t80 = "OK" if a["threshold_analysis"]["8.0"]["workable"] else "FAIL"
        print(
            f"{v:<20} {a['avg_terrible']:>8.1f} {a['avg_mediocre']:>9.1f} "
            f"{a['avg_excellent']:>9.1f} {order:>6} {a['rank_correlation']:>6.2f} "
            f"{a['avg_dim_spread']:>7.1f} {a['gap_terrible_mediocre']:>7.1f} "
            f"{a['gap_mediocre_excellent']:>7.1f} {t75:>6} {t80:>6}"
        )

    # Decision guidance
    print()
    print("=" * 80)
    print("DECISION GUIDANCE")
    print("=" * 80)

    # Find best variant: ordering correct + highest excellent + largest gaps
    viable = [r for r in all_variant_results if r["aggregate"]["ordering_correct"]]
    if not viable:
        print("WARNING: No variant maintained correct quality ordering!")
        viable = all_variant_results

    # Sort by: excellent avg descending, then gap_mediocre_excellent descending
    viable.sort(
        key=lambda r: (r["aggregate"]["avg_excellent"], r["aggregate"]["gap_mediocre_excellent"]),
        reverse=True,
    )

    for i, result in enumerate(viable[:3]):
        v = result["variant"]
        a = result["aggregate"]
        print(f"\n  #{i + 1}: {v}")
        print(
            f"    Excellent avg: {a['avg_excellent']:.1f}  "
            f"Mediocre avg: {a['avg_mediocre']:.1f}  "
            f"Rank corr: {a['rank_correlation']:.2f}"
        )
        print(
            f"    Threshold 7.5: {'WORKS' if a['threshold_analysis']['7.5']['workable'] else 'FAILS'}  "
            f"Threshold 8.0: {'WORKS' if a['threshold_analysis']['8.0']['workable'] else 'FAILS'}"
        )

        if a["threshold_analysis"]["7.5"]["workable"]:
            print("    -> Good candidate with threshold=7.5")
        if a["threshold_analysis"]["8.0"]["workable"]:
            print("    -> Also works with threshold=8.0")
        if not a["ordering_correct"]:
            print("    -> WARNING: Quality ordering is broken (mediocre >= excellent)")


if __name__ == "__main__":
    main()
