"""Investigate whether Ollama's grammar-constrained generation quantizes judge scores.

Compares score distributions between two output modes:
  A) format=<json_schema>  — Ollama grammar-constrained (production path)
  B) format="json"         — unconstrained JSON (baseline)

For each mode, sends the same judge prompt N times and collects all scores.
Compares unique values, decimal precision, and distribution shape.

Usage:
    python scripts/investigate_structured_output_quantization.py --verbose
    python scripts/investigate_structured_output_quantization.py --model gemma3:12b --rounds 10
"""

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import ollama
from pydantic import ValidationError

from src.memory.world_quality._entity_scores import FactionQualityScores
from src.services.world_quality_service._common import JUDGE_CALIBRATION_BLOCK
from src.utils.streaming import consume_stream

logger = logging.getLogger(__name__)

# =====================================================================
# Test fixture — a mediocre faction for repeated judging
# =====================================================================
TEST_FACTION = {
    "name": "The Iron Covenant",
    "description": "A secretive group of former soldiers who control trade routes through intimidation. They claim to protect merchants but extract heavy tolls.",
    "leader": "Commander Voss",
    "goals": ["Control all northern trade routes", "Accumulate wealth"],
    "values": ["Loyalty to the group", "Strength"],
}

JUDGE_PROMPT = f"""You are a strict quality judge evaluating a faction for a fantasy story.

FACTION TO EVALUATE:
Name: {TEST_FACTION["name"]}
Description: {TEST_FACTION["description"]}
Leader: {TEST_FACTION["leader"]}
Goals: {", ".join(TEST_FACTION["goals"])}
Values: {", ".join(TEST_FACTION["values"])}

{JUDGE_CALIBRATION_BLOCK}

Rate each dimension 0-10:
- coherence: Internal consistency, clear structure
- influence: World impact, power level
- conflict_potential: Story conflict opportunities
- distinctiveness: Memorable, unique qualities
- temporal_plausibility: Timeline consistency, era-appropriate placement

Provide specific, actionable feedback for improvement in the feedback field.

OUTPUT FORMAT - Return ONLY a flat JSON object with these exact fields:
{{"coherence": <float 0-10>, "influence": <float 0-10>, "conflict_potential": <float 0-10>, "distinctiveness": <float 0-10>, "temporal_plausibility": <float 0-10>, "feedback": "Your assessment..."}}

DO NOT wrap in "properties" or "description" - return ONLY the flat scores object with YOUR OWN assessment."""

SCORE_DIMS = [
    "coherence",
    "influence",
    "conflict_potential",
    "distinctiveness",
    "temporal_plausibility",
]


def run_constrained(client: ollama.Client, model: str, rounds: int) -> list[dict]:
    """Run judge calls with format=json_schema (production path)."""
    schema = FactionQualityScores.model_json_schema()
    results = []
    for i in range(rounds):
        try:
            start = time.time()
            stream = client.chat(
                model=model,
                messages=[{"role": "user", "content": JUDGE_PROMPT}],
                format=schema,
                options={"temperature": 0.1, "num_ctx": 4096},
                stream=True,
            )
            resp = consume_stream(stream)
            elapsed = time.time() - start
            content = resp["message"]["content"]
            scores = FactionQualityScores.model_validate_json(content)
            results.append(
                {
                    "round": i + 1,
                    "coherence": scores.coherence,
                    "influence": scores.influence,
                    "conflict_potential": scores.conflict_potential,
                    "distinctiveness": scores.distinctiveness,
                    "temporal_plausibility": scores.temporal_plausibility,
                    "time": round(elapsed, 2),
                }
            )
        except (ValidationError, KeyError, TypeError) as e:
            logger.warning("Constrained round %d failed: %s", i + 1, e)
        except ollama.ResponseError as e:
            logger.warning("Constrained round %d Ollama error: %s", i + 1, e)
    return results


def run_unconstrained(client: ollama.Client, model: str, rounds: int) -> list[dict]:
    """Run judge calls with format='json' (unconstrained baseline)."""
    results = []
    for i in range(rounds):
        try:
            start = time.time()
            stream = client.chat(
                model=model,
                messages=[{"role": "user", "content": JUDGE_PROMPT}],
                format="json",
                options={"temperature": 0.1, "num_ctx": 4096},
                stream=True,
            )
            resp = consume_stream(stream)
            elapsed = time.time() - start
            content = resp["message"]["content"]
            data = json.loads(content)
            # Validate manually — unconstrained JSON may have extra/missing keys
            scores = {}
            for dim in SCORE_DIMS:
                val = data.get(dim)
                if val is not None:
                    scores[dim] = float(val)
            if len(scores) == len(SCORE_DIMS):
                results.append(
                    {
                        "round": i + 1,
                        **scores,
                        "time": round(elapsed, 2),
                    }
                )
            else:
                logger.warning(
                    "Unconstrained round %d missing dimensions: got %s", i + 1, list(data.keys())
                )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning("Unconstrained round %d failed: %s", i + 1, e)
        except ollama.ResponseError as e:
            logger.warning("Unconstrained round %d Ollama error: %s", i + 1, e)
    return results


def analyze_scores(results: list[dict], label: str) -> dict:
    """Analyze score distribution from a set of results."""
    if not results:
        return {"label": label, "rounds": 0, "error": "no valid results"}

    all_scores = []
    per_dim: dict[str, list[float]] = {d: [] for d in SCORE_DIMS}

    for r in results:
        for dim in SCORE_DIMS:
            val = r[dim]
            all_scores.append(val)
            per_dim[dim].append(val)

    # Decimal precision analysis
    decimal_places = []
    for s in all_scores:
        s_str = f"{s:.10f}".rstrip("0")
        if "." in s_str:
            decimal_places.append(len(s_str.split(".")[1]))
        else:
            decimal_places.append(0)

    unique_scores = sorted(set(all_scores))
    score_counter = Counter(all_scores)
    modal_scores = score_counter.most_common(3)

    analysis = {
        "label": label,
        "rounds": len(results),
        "total_scores": len(all_scores),
        "unique_values": len(unique_scores),
        "unique_scores": unique_scores,
        "mean": round(statistics.mean(all_scores), 3),
        "std_dev": round(statistics.stdev(all_scores), 3) if len(all_scores) > 1 else 0.0,
        "min": min(all_scores),
        "max": max(all_scores),
        "range": round(max(all_scores) - min(all_scores), 2),
        "modal_scores": [(s, c) for s, c in modal_scores],
        "avg_decimal_places": round(statistics.mean(decimal_places), 1),
        "max_decimal_places": max(decimal_places),
        "decimal_distribution": dict(Counter(decimal_places)),
        "per_dimension": {},
    }

    for dim in SCORE_DIMS:
        vals = per_dim[dim]
        analysis["per_dimension"][dim] = {
            "unique": len(set(vals)),
            "mean": round(statistics.mean(vals), 3),
            "std_dev": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    return analysis


def print_analysis(analysis: dict, verbose: bool = False) -> None:
    """Print analysis results."""
    label = analysis["label"]
    print(f"\n--- {label} ---")
    if "error" in analysis:
        print(f"  ERROR: {analysis['error']}")
        return

    print(f"  Rounds: {analysis['rounds']}, Total scores: {analysis['total_scores']}")
    print(f"  Unique values: {analysis['unique_values']}")
    print(f"  Mean: {analysis['mean']}, Std Dev: {analysis['std_dev']}")
    print(f"  Range: {analysis['min']}-{analysis['max']} (spread: {analysis['range']})")
    print(f"  Modal scores: {analysis['modal_scores']}")
    print(
        f"  Decimal places: avg={analysis['avg_decimal_places']}, max={analysis['max_decimal_places']}"
    )
    print(f"  Decimal distribution: {analysis['decimal_distribution']}")

    if verbose:
        print("  Per-dimension:")
        for dim, stats in analysis["per_dimension"].items():
            print(
                f"    {dim}: unique={stats['unique']}, mean={stats['mean']}, std={stats['std_dev']}, values={stats['values']}"
            )


def print_comparison(constrained: dict, unconstrained: dict) -> None:
    """Print comparison between constrained and unconstrained modes."""
    print("\n" + "=" * 70)
    print("COMPARISON: Constrained (json_schema) vs Unconstrained (json)")
    print("=" * 70)

    if "error" in constrained or "error" in unconstrained:
        print("  Cannot compare — one or both modes had errors")
        return

    rows = [
        ("Unique score values", constrained["unique_values"], unconstrained["unique_values"]),
        ("Score std dev", constrained["std_dev"], unconstrained["std_dev"]),
        ("Score range", constrained["range"], unconstrained["range"]),
        (
            "Avg decimal places",
            constrained["avg_decimal_places"],
            unconstrained["avg_decimal_places"],
        ),
        (
            "Max decimal places",
            constrained["max_decimal_places"],
            unconstrained["max_decimal_places"],
        ),
        ("Mean score", constrained["mean"], unconstrained["mean"]),
    ]

    print(f"  {'Metric':<25} {'Constrained':>12} {'Unconstrained':>14} {'Delta':>8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 14} {'-' * 8}")
    for name, c_val, u_val in rows:
        delta = round(c_val - u_val, 3) if isinstance(c_val, (int, float)) else "—"
        print(f"  {name:<25} {c_val:>12} {u_val:>14} {delta:>8}")

    # Quantization verdict
    c_unique = constrained["unique_values"]
    u_unique = unconstrained["unique_values"]
    c_decimals = constrained["avg_decimal_places"]
    u_decimals = unconstrained["avg_decimal_places"]

    print("\n--- VERDICT ---")
    if c_unique < u_unique * 0.5:
        print(
            "  [!] QUANTIZATION DETECTED: constrained mode produces significantly fewer unique values"
        )
    elif c_decimals < u_decimals - 0.5:
        print("  [!] PRECISION LOSS: constrained mode uses fewer decimal places")
    elif abs(c_unique - u_unique) <= 2 and abs(c_decimals - u_decimals) <= 0.3:
        print("  [OK] No significant quantization — both modes produce similar distributions")
    else:
        print("  [?] INCONCLUSIVE — minor differences, may need more rounds")


def main() -> None:
    """Run the structured output quantization investigation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test structured output quantization")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model to test (default: first non-embedding installed model)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of judge calls per mode (default: 10)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-dimension details")
    args = parser.parse_args()

    # Resolve model
    client = ollama.Client(host="http://localhost:11434")

    if args.model:
        model = args.model
    else:
        # Pick the recommended judge model if installed, otherwise first non-embedding model
        installed = [m.model for m in client.list().models or [] if m.model]
        preferred = ["gemma3:12b", "phi4:14b", "phi4-mini:latest", "gemma3:4b"]
        model = next((m for m in preferred if m in installed), None)
        if model is None:
            embedding_prefixes = ("mxbai-embed", "snowflake-arctic", "bge-m3")
            model = next(
                (m for m in installed if not m.startswith(embedding_prefixes)),
                None,
            )
        if model is None:
            logger.error("No suitable model found. Install a model first.")
            sys.exit(1)

    print("=" * 70)
    print("STRUCTURED OUTPUT QUANTIZATION TEST — Issue #294")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Rounds per mode: {args.rounds}")
    print(f"Test entity: {TEST_FACTION['name']}")

    # Run constrained mode (production path)
    print(f"\nRunning {args.rounds} constrained calls (format=json_schema)...")
    constrained_results = run_constrained(client, model, args.rounds)
    constrained_analysis = analyze_scores(constrained_results, "Constrained (json_schema)")
    print_analysis(constrained_analysis, args.verbose)

    # Unload and reload to avoid cache effects
    try:
        client.generate(model=model, prompt="", keep_alive=0)
        time.sleep(1)
    except ollama.ResponseError:
        pass  # Best-effort model eviction — failure is harmless

    # Run unconstrained mode (baseline)
    print(f"\nRunning {args.rounds} unconstrained calls (format='json')...")
    unconstrained_results = run_unconstrained(client, model, args.rounds)
    unconstrained_analysis = analyze_scores(unconstrained_results, "Unconstrained (json)")
    print_analysis(unconstrained_analysis, args.verbose)

    # Compare
    print_comparison(constrained_analysis, unconstrained_analysis)

    # Unload model
    try:
        client.generate(model=model, prompt="", keep_alive=0)
    except ollama.ResponseError:
        pass  # Best-effort model eviction — failure is harmless

    # Save results
    output_dir = Path("output/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"quantization_test_{timestamp}.json"
    output_file.write_text(
        json.dumps(
            {
                "model": model,
                "rounds": args.rounds,
                "constrained": constrained_analysis,
                "unconstrained": unconstrained_analysis,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
