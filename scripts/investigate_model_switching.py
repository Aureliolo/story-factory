#!/usr/bin/env python3
"""Model Switching Investigation — measures GPU residency, cold-start penalty, and diversity.

Investigates issue #399 model switching concerns:
  1. GPU residency per model transition — monitors nvidia-smi during creator<->judge alternation
  2. Cold-start penalty — measures first-call latency with and without warm-up ping
  3. Alternating vs batched call patterns — times interleaved vs batched model usage
  4. Creator temperature diversity — compares entity name/description uniqueness across temps

Works best with at least 2 installed models; falls back to using a single model for both
creator and judge roles with a warning.

Usage:
    python scripts/investigate_model_switching.py [options]
      --creator-model MODEL    (default: auto-select from installed)
      --judge-model MODEL      (default: auto-select different from creator)
      --rounds 5               (alternation rounds per scenario, default: 5)
      --diversity-builds 3     (diversity builds per temperature, default: 3)
      --temps 0.1,0.5,0.7,0.9 (creator temps to compare, default: 0.1,0.5,0.7,0.9)
      --skip-diversity          (skip the diversity comparison — saves time)
      --skip-gpu               (skip GPU monitoring — timing only)
      --output results.json    (default: output/diagnostics/model_switching_<ts>.json)
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

from scripts._model_switching_helpers import check_gpu_residency
from scripts._model_switching_scenarios import (
    run_alternation_test,
    run_cold_start_test,
    run_diversity_test,
    run_gpu_residency_test,
)
from scripts._ollama_helpers import (
    get_installed_models,
    get_model_info,
    get_total_gpu_vram_mb,
    unload_model,
)
from scripts.investigate_vram_usage import get_gpu_vram_usage, nvidia_smi_available

logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _select_models(args: argparse.Namespace) -> tuple[str, str]:
    """Select creator and judge models from args or auto-detect.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (creator_model, judge_model).
    """
    if args.creator_model and args.judge_model:
        logger.info(
            "Using explicit models: creator=%s, judge=%s", args.creator_model, args.judge_model
        )
        return args.creator_model, args.judge_model

    installed = get_installed_models()
    if not installed:
        logger.error("No models found — is Ollama running?")
        print("ERROR: No models found. Is Ollama running?")
        sys.exit(1)

    creator = args.creator_model or installed[0]

    if args.judge_model:
        judge = args.judge_model
    elif len(installed) >= 2:
        judge = next((m for m in installed if m != creator), installed[1])
    else:
        logger.warning("Only 1 model installed (%s) — using same for both roles", creator)
        print(
            f"WARNING: Only 1 model installed ({creator}). "
            "Using same model for creator and judge — model switching test will show no swap."
        )
        judge = creator

    logger.info("Selected models: creator=%s, judge=%s", creator, judge)
    return creator, judge


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Model Switching Investigation — measures GPU residency, "
            "cold-start penalty, and creator diversity."
        )
    )
    parser.add_argument("--creator-model", type=str, help="Creator model (default: auto-select)")
    parser.add_argument("--judge-model", type=str, help="Judge model (default: auto-select)")
    parser.add_argument(
        "--rounds", type=int, default=5, help="Alternation rounds per scenario (default: 5)"
    )
    parser.add_argument(
        "--diversity-builds",
        type=int,
        default=3,
        help="Builds per temperature for diversity test (default: 3)",
    )
    parser.add_argument(
        "--temps",
        type=str,
        default="0.1,0.5,0.7,0.9",
        help="Comma-separated creator temperatures to compare (default: 0.1,0.5,0.7,0.9)",
    )
    parser.add_argument("--skip-diversity", action="store_true", help="Skip diversity comparison")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU monitoring")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed measurements")
    return parser.parse_args()


def _validate_temps(temps_str: str) -> list[float]:
    """Parse and validate temperature values from CLI argument.

    Args:
        temps_str: Comma-separated temperature string.

    Returns:
        List of validated temperature floats.
    """
    try:
        temps = [float(t.strip()) for t in temps_str.split(",")]
    except ValueError as e:
        logger.error("Invalid temperature values in --temps: %s", e)
        print(f"ERROR: Invalid temperature values in --temps: {e}")
        print("Expected comma-separated floats, e.g., --temps 0.1,0.5,0.7,0.9")
        sys.exit(1)

    for t in temps:
        if not 0.0 <= t <= 2.0:
            logger.error("Temperature %.1f is outside valid range [0.0, 2.0]", t)
            print(f"ERROR: Temperature {t} is outside valid range [0.0, 2.0]")
            sys.exit(1)

    return temps


def _enforce_gpu_residency(creator_model: str, judge_model: str, has_gpu: bool) -> None:
    """Check both models against the 80% GPU residency rule.

    Args:
        creator_model: Creator model name.
        judge_model: Judge model name.
        has_gpu: Whether GPU monitoring is available.
    """
    if not has_gpu:
        return

    gpu_total_mb = get_total_gpu_vram_mb()
    for model_name in {creator_model, judge_model}:
        if not check_gpu_residency(model_name, gpu_total_mb):
            print(
                f"ERROR: Model {model_name} fails 80% GPU residency requirement. "
                "Aborting per GPU residency rule. Choose a smaller model."
            )
            sys.exit(1)

    logger.info("GPU residency check passed for both models")


def _print_header(
    creator_model: str,
    judge_model: str,
    creator_info: dict[str, Any],
    judge_info: dict[str, Any],
    has_gpu: bool,
    args: argparse.Namespace,
    temps: list[float],
    output_path: Path,
) -> None:
    """Print investigation header with configuration summary.

    Args:
        creator_model: Creator model name.
        judge_model: Judge model name.
        creator_info: Creator model info dict.
        judge_info: Judge model info dict.
        has_gpu: Whether GPU monitoring is available.
        args: Parsed CLI arguments.
        temps: Validated temperatures.
        output_path: Where results will be written.
    """
    print("=" * 70)
    print("MODEL SWITCHING INVESTIGATION (Issue #399)")
    print("=" * 70)
    print(f"GPU monitoring: {'YES' if has_gpu else 'NO (timing only)'}")
    if has_gpu:
        gpu = get_gpu_vram_usage()
        print(f"GPU VRAM: {gpu['total_mb']}MB total, {gpu['used_mb']}MB used")
    print(f"Creator: {creator_model} ({creator_info.get('parameter_size', '?')})")
    print(f"Judge:   {judge_model} ({judge_info.get('parameter_size', '?')})")
    print(f"Rounds:  {args.rounds}")
    if not args.skip_diversity:
        print(f"Diversity temps: {temps} ({args.diversity_builds} builds each)")
    print(f"Output:  {output_path}")
    print("=" * 70)


def _run_scenarios(
    creator_model: str,
    judge_model: str,
    args: argparse.Namespace,
    temps: list[float],
    monitor_gpu: bool,
) -> list[dict[str, Any]]:
    """Execute all investigation scenarios with top-level exception handling.

    Catches unexpected errors and preserves any partial results collected
    before the failure.

    Args:
        creator_model: Creator model name.
        judge_model: Judge model name.
        args: Parsed CLI arguments.
        temps: Validated temperatures.
        monitor_gpu: Whether to record GPU VRAM.

    Returns:
        List of scenario result dicts (may be partial if an error occurred).
    """
    scenarios: list[dict[str, Any]] = []

    try:
        cold_start = run_cold_start_test(creator_model, monitor_gpu, args.verbose)
        scenarios.append(cold_start)
        _print_cold_start_inline(cold_start)

        alternation = run_alternation_test(
            creator_model, judge_model, args.rounds, monitor_gpu, args.verbose
        )
        scenarios.append(alternation)
        _print_alternation_inline(alternation)

        if monitor_gpu:
            gpu_residency = run_gpu_residency_test(
                creator_model, judge_model, args.rounds, args.verbose
            )
            scenarios.append(gpu_residency)
            print(f"  => {gpu_residency['finding']}")
        else:
            print("\n  Scenario 3: GPU residency — SKIPPED (no GPU or --skip-gpu)")

        if not args.skip_diversity:
            diversity = run_diversity_test(
                creator_model, temps, args.diversity_builds, args.verbose
            )
            scenarios.append(diversity)
            _print_diversity_inline(diversity)
        else:
            print("\n  Scenario 4: Diversity comparison — SKIPPED (--skip-diversity)")

    except Exception:
        logger.exception("Investigation failed unexpectedly during scenario execution")
        print("\nERROR: Investigation failed unexpectedly. Saving partial results...")

    return scenarios


def _print_cold_start_inline(cold_start: dict[str, Any]) -> None:
    """Print cold-start scenario inline summary.

    Args:
        cold_start: Cold-start scenario result dict.
    """
    print(
        f"  => Cold penalty: {cold_start['cold_penalty_s']:.2f}s "
        f"(cold={cold_start['cold_call']['duration_s']:.2f}s, "
        f"warm_median={cold_start['warm_median_s']:.2f}s)"
    )
    wp = cold_start["warmup_ping"]
    print(
        f"  => Warm-up ping: {wp['warmup_time_s']:.2f}s + "
        f"post-warmup call: {wp['post_warmup_call_s']:.2f}s "
        f"(savings: {cold_start['warmup_savings_s']:.2f}s)"
    )


def _print_alternation_inline(alternation: dict[str, Any]) -> None:
    """Print alternation scenario inline summary.

    Args:
        alternation: Alternation scenario result dict.
    """
    print(
        f"  => Alternating: {alternation['alternating']['total_time_s']:.1f}s, "
        f"Batched: {alternation['batched']['total_time_s']:.1f}s, "
        f"Savings: {alternation['time_saved_by_batching_s']:.1f}s "
        f"({alternation['batching_speedup_pct']:.1f}%)"
    )
    bimodal = alternation["bimodal_judge_detection"]
    if bimodal.get("is_bimodal"):
        print(
            f"  => BIMODAL judge durations detected: "
            f"fast={bimodal['fast_cluster']['median_s']:.2f}s "
            f"slow={bimodal['slow_cluster']['median_s']:.2f}s"
        )


def _print_diversity_inline(diversity: dict[str, Any]) -> None:
    """Print diversity scenario inline summary.

    Args:
        diversity: Diversity scenario result dict.
    """
    print("  => Diversity results:")
    for tr in diversity["results"]:
        m = tr["metrics"]
        print(
            f"      temp={tr['temperature']}: "
            f"{m['unique_names']}/{m['total_factions']} unique names "
            f"({m['name_uniqueness_ratio']:.1%}), "
            f"{m['unique_types']} type variants"
        )


def _print_final_summary(scenarios: list[dict[str, Any]], overall_time: float) -> None:
    """Print final investigation summary with recommendations.

    Args:
        scenarios: List of completed scenario result dicts.
        overall_time: Total investigation time in seconds.
    """
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total investigation time: {overall_time:.0f}s")
    print()

    cold_start = next((s for s in scenarios if s.get("scenario") == "cold_start_penalty"), None)
    alternation = next((s for s in scenarios if s.get("scenario") == "alternation_test"), None)

    if cold_start:
        _print_cold_start_recommendation(cold_start)

    if alternation:
        _print_alternation_recommendation(alternation)


def _print_cold_start_recommendation(cold_start: dict[str, Any]) -> None:
    """Print cold-start recommendation block.

    Args:
        cold_start: Cold-start scenario result dict.
    """
    print("COLD-START:")
    print(f"  Cold penalty: {cold_start['cold_penalty_s']:.2f}s")
    wp = cold_start["warmup_ping"]
    print(f"  Warm-up ping cost: {wp['warmup_time_s']:.2f}s")
    print(f"  Net savings from warm-up: {cold_start['warmup_savings_s']:.2f}s")
    if cold_start["warmup_savings_s"] > 0:
        print("  RECOMMENDATION: Add warm-up ping before first build step")
    else:
        print("  FINDING: Warm-up ping does not save time (model loads fast enough)")
    print()


def _print_alternation_recommendation(alternation: dict[str, Any]) -> None:
    """Print alternation recommendation block.

    Args:
        alternation: Alternation scenario result dict.
    """
    print("MODEL SWITCHING:")
    print(
        f"  Batching saves {alternation['time_saved_by_batching_s']:.1f}s "
        f"({alternation['batching_speedup_pct']:.1f}%) over {alternation['rounds']} rounds"
    )
    if alternation["batching_speedup_pct"] > 10:
        print("  RECOMMENDATION: Consider batching entity creation before judging")
    else:
        print("  FINDING: Switching overhead is minimal — batching not necessary")
    print()

    bimodal = alternation["bimodal_judge_detection"]
    if bimodal.get("is_bimodal"):
        print("BIMODAL JUDGE DURATIONS:")
        print(
            f"  Fast cluster: {bimodal['fast_cluster']['n']} calls, "
            f"median {bimodal['fast_cluster']['median_s']:.2f}s"
        )
        print(
            f"  Slow cluster: {bimodal['slow_cluster']['n']} calls, "
            f"median {bimodal['slow_cluster']['median_s']:.2f}s"
        )
        print("  LIKELY CAUSE: Partial GPU residency during model swap")
    print()


def _write_results(output: dict[str, Any], output_path: Path) -> None:
    """Write investigation results to JSON file with error handling.

    Falls back to current directory, then stdout, if the target path fails.

    Args:
        output: Complete investigation output dict.
        output_path: Target file path.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", output_path)
    except OSError as e:
        logger.error("Failed to write results to %s: %s", output_path, e)
        print(f"ERROR: Could not write results to {output_path}: {e}")
        fallback = Path(f"model_switching_results_{int(time.time())}.json")
        try:
            with open(fallback, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to fallback location: {fallback}")
        except OSError:
            print("CRITICAL: Could not write results to any file. Dumping to stdout:")
            print(json.dumps(output, indent=2, ensure_ascii=False))


def main() -> None:
    """Run the model switching investigation."""
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting model switching investigation")

    creator_model, judge_model = _select_models(args)
    temps = _validate_temps(args.temps)

    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"model_switching_{timestamp}.json"

    has_gpu = nvidia_smi_available() and not args.skip_gpu
    monitor_gpu = has_gpu

    _enforce_gpu_residency(creator_model, judge_model, has_gpu)

    creator_info = get_model_info(creator_model)
    judge_info = get_model_info(judge_model)

    _print_header(
        creator_model, judge_model, creator_info, judge_info, has_gpu, args, temps, output_path
    )

    overall_start = time.monotonic()
    scenarios = _run_scenarios(creator_model, judge_model, args, temps, monitor_gpu)

    print("\nCleaning up — unloading models...")
    unload_model(creator_model)
    unload_model(judge_model)

    overall_time = round(time.monotonic() - overall_start, 1)

    output = {
        "investigation_metadata": {
            "script": "investigate_model_switching.py",
            "issue": "#399 — model switching, cold-start, diversity",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "gpu_available": has_gpu,
            "creator_model": creator_model,
            "creator_info": creator_info,
            "judge_model": judge_model,
            "judge_info": judge_info,
        },
        "scenarios": scenarios,
    }

    _write_results(output, output_path)
    _print_final_summary(scenarios, overall_time)
    print(f"Results written to {output_path}")
    print()
    logger.info("Investigation complete in %.1fs", overall_time)


if __name__ == "__main__":
    main()
