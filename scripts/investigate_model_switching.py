#!/usr/bin/env python3
"""Model Switching Investigation — measures GPU residency, cold-start penalty, and diversity.

Investigates issue #399 model switching concerns:
  1. GPU residency per model transition — monitors nvidia-smi during creator↔judge alternation
  2. Cold-start penalty — measures first-call latency with and without warm-up ping
  3. Alternating vs batched call patterns — times interleaved vs batched model usage
  4. Creator temperature diversity — compares entity name/description uniqueness across temps

Requires Ollama running with at least 2 installed models (one for creator, one for judge).
Falls back gracefully on non-NVIDIA systems (timing data only).

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
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from scripts._ollama_helpers import (
    CANONICAL_BRIEF,
    INVESTIGATION_NUM_CTX,
    OLLAMA_BASE,
    get_installed_models,
    get_model_info,
    unload_model,
    warm_model,
)
from scripts.investigate_vram_usage import (
    get_gpu_vram_usage,
    get_ollama_loaded_models,
    nvidia_smi_available,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# =====================================================================
# LLM call helpers
# =====================================================================
def _call_model(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    num_ctx: int = INVESTIGATION_NUM_CTX,
    timeout: int = 180,
) -> dict[str, Any]:
    """Send a prompt to a model and return timing + response data.

    Args:
        model: Ollama model name.
        prompt: The user prompt.
        temperature: Generation temperature.
        num_ctx: Context window size.
        timeout: Request timeout in seconds.

    Returns:
        Dict with duration_s, response, tokens, error.
    """
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": 256,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        elapsed = round(time.monotonic() - start, 3)
        content = body.get("message", {}).get("content", "")
        eval_count = body.get("eval_count", 0)
        prompt_eval_count = body.get("prompt_eval_count", 0)
        return {
            "duration_s": elapsed,
            "response": content[:500],
            "eval_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "error": None,
        }
    except httpx.TimeoutException:
        elapsed = round(time.monotonic() - start, 3)
        return {
            "duration_s": elapsed,
            "response": "",
            "eval_tokens": 0,
            "prompt_tokens": 0,
            "error": "timeout",
        }
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        elapsed = round(time.monotonic() - start, 3)
        return {
            "duration_s": elapsed,
            "response": "",
            "eval_tokens": 0,
            "prompt_tokens": 0,
            "error": str(e)[:200],
        }


# =====================================================================
# Scenario 1: Cold-start penalty
# =====================================================================
def run_cold_start_test(model: str, monitor_gpu: bool, verbose: bool) -> dict[str, Any]:
    """Measure cold-start penalty: first call after full unload vs warm call.

    Sequence: unload → cold call → warm call → warm call → unload → warm-up ping → call.

    Args:
        model: Model to test.
        monitor_gpu: Whether to record GPU VRAM.
        verbose: Print detailed output.

    Returns:
        Dict with cold_call, warm_calls, warmed_call timings and VRAM snapshots.
    """
    print("\n  Scenario 1: Cold-start penalty measurement")

    # Start clean
    unload_model(model)
    time.sleep(2)
    baseline_vram = get_gpu_vram_usage() if monitor_gpu else {}

    prompt = f"Given this story premise, name 3 characters:\n{CANONICAL_BRIEF[:200]}"

    # Cold call (model not in VRAM)
    if verbose:
        print(f"    Cold call to {model}...")
    cold_result = _call_model(model, prompt, temperature=0.7)
    time.sleep(1)
    post_cold_vram = get_gpu_vram_usage() if monitor_gpu else {}
    if verbose:
        print(f"      Cold: {cold_result['duration_s']:.2f}s")

    # Warm calls (model already loaded)
    warm_results = []
    for i in range(3):
        if verbose:
            print(f"    Warm call {i + 1}...")
        result = _call_model(model, prompt, temperature=0.7)
        warm_results.append(result)
        if verbose:
            print(f"      Warm {i + 1}: {result['duration_s']:.2f}s")

    warm_durations = [r["duration_s"] for r in warm_results if r["error"] is None]
    warm_median = statistics.median(warm_durations) if warm_durations else 0

    # Unload, then warm-up ping, then call
    unload_model(model)
    time.sleep(2)
    if verbose:
        print(f"    Warm-up ping to {model}...")
    warmup_start = time.monotonic()
    warm_model(model, num_ctx=INVESTIGATION_NUM_CTX)
    warmup_time = round(time.monotonic() - warmup_start, 3)

    if verbose:
        print(f"      Warm-up ping took: {warmup_time:.2f}s")
        print(f"    Post-warmup call to {model}...")
    warmed_result = _call_model(model, prompt, temperature=0.7)
    if verbose:
        print(f"      Post-warmup: {warmed_result['duration_s']:.2f}s")

    cold_penalty = round(cold_result["duration_s"] - warm_median, 3) if warm_median else 0

    return {
        "scenario": "cold_start_penalty",
        "model": model,
        "cold_call": {
            "duration_s": cold_result["duration_s"],
            "error": cold_result["error"],
            "vram_after": post_cold_vram,
        },
        "warm_calls": [{"duration_s": r["duration_s"], "error": r["error"]} for r in warm_results],
        "warm_median_s": warm_median,
        "cold_penalty_s": cold_penalty,
        "warmup_ping": {
            "warmup_time_s": warmup_time,
            "post_warmup_call_s": warmed_result["duration_s"],
            "error": warmed_result["error"],
        },
        "warmup_savings_s": round(
            cold_result["duration_s"] - (warmup_time + warmed_result["duration_s"]), 3
        ),
        "baseline_vram": baseline_vram,
    }


# =====================================================================
# Scenario 2: Alternating vs batched model transitions
# =====================================================================
def run_alternation_test(
    creator_model: str,
    judge_model: str,
    rounds: int,
    monitor_gpu: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Compare alternating creator↔judge vs batched creator-then-judge patterns.

    Pattern A (alternating): create→judge→create→judge→... (simulates quality loop)
    Pattern B (batched): create→create→...→judge→judge→... (hypothetical batch mode)

    Args:
        creator_model: Model for creative generation.
        judge_model: Model for quality judging.
        rounds: Number of create+judge pairs.
        monitor_gpu: Whether to record GPU VRAM.
        verbose: Print detailed output.

    Returns:
        Dict with per-call timings, VRAM snapshots, and transition overhead.
    """
    print("\n  Scenario 2: Alternating vs batched model transitions")

    create_prompt = (
        f"Create a unique fantasy faction for this story:\n{CANONICAL_BRIEF[:200]}\n"
        "Reply with a faction name, leader, and 2-sentence description."
    )
    judge_prompt = (
        "Rate this faction on a scale of 1-10 for originality, coherence, and depth. "
        "Reply with just the three scores and a 1-sentence rationale."
    )

    # --- Pattern A: Alternating ---
    print("    Pattern A: Alternating (create→judge→create→judge→...)")
    unload_model(creator_model)
    unload_model(judge_model)
    time.sleep(2)

    alternating_steps: list[dict[str, Any]] = []
    alt_start = time.monotonic()

    for i in range(rounds):
        # Creator call
        vram_before = get_gpu_vram_usage() if monitor_gpu else {}
        create_result = _call_model(creator_model, create_prompt, temperature=0.9)
        vram_after_create = get_gpu_vram_usage() if monitor_gpu else {}

        # Judge call using creator's output
        faction_text = create_result["response"][:300]
        judge_input = f"{judge_prompt}\n\nFaction: {faction_text}"
        judge_result = _call_model(judge_model, judge_input, temperature=0.1)
        vram_after_judge = get_gpu_vram_usage() if monitor_gpu else {}

        step = {
            "round": i + 1,
            "create": {
                "duration_s": create_result["duration_s"],
                "error": create_result["error"],
            },
            "judge": {
                "duration_s": judge_result["duration_s"],
                "error": judge_result["error"],
            },
        }
        if monitor_gpu:
            step["vram_before"] = vram_before
            step["vram_after_create"] = vram_after_create
            step["vram_after_judge"] = vram_after_judge

        alternating_steps.append(step)
        if verbose:
            print(
                f"      Round {i + 1}: create={create_result['duration_s']:.2f}s "
                f"judge={judge_result['duration_s']:.2f}s"
            )

    alt_total = round(time.monotonic() - alt_start, 3)

    # --- Pattern B: Batched ---
    print("    Pattern B: Batched (create→create→...→judge→judge→...)")
    unload_model(creator_model)
    unload_model(judge_model)
    time.sleep(2)

    batched_creates: list[dict[str, Any]] = []
    batched_judges: list[dict[str, Any]] = []
    batch_start = time.monotonic()

    # All creates first
    faction_texts: list[str] = []
    for i in range(rounds):
        create_result = _call_model(creator_model, create_prompt, temperature=0.9)
        batched_creates.append(
            {
                "round": i + 1,
                "duration_s": create_result["duration_s"],
                "error": create_result["error"],
            }
        )
        faction_texts.append(create_result["response"][:300])
        if verbose:
            print(f"      Create {i + 1}: {create_result['duration_s']:.2f}s")

    # All judges second
    for i in range(rounds):
        judge_input = f"{judge_prompt}\n\nFaction: {faction_texts[i]}"
        judge_result = _call_model(judge_model, judge_input, temperature=0.1)
        batched_judges.append(
            {
                "round": i + 1,
                "duration_s": judge_result["duration_s"],
                "error": judge_result["error"],
            }
        )
        if verbose:
            print(f"      Judge {i + 1}: {judge_result['duration_s']:.2f}s")

    batch_total = round(time.monotonic() - batch_start, 3)

    # Compute stats
    alt_create_times = [
        s["create"]["duration_s"] for s in alternating_steps if s["create"]["error"] is None
    ]
    alt_judge_times = [
        s["judge"]["duration_s"] for s in alternating_steps if s["judge"]["error"] is None
    ]
    batch_create_times = [s["duration_s"] for s in batched_creates if s["error"] is None]
    batch_judge_times = [s["duration_s"] for s in batched_judges if s["error"] is None]

    def _stats(durations: list[float]) -> dict[str, float]:
        if not durations:
            return {"median": 0, "mean": 0, "stdev": 0, "min": 0, "max": 0}
        return {
            "median": round(statistics.median(durations), 3),
            "mean": round(statistics.mean(durations), 3),
            "stdev": round(statistics.stdev(durations), 3) if len(durations) > 1 else 0,
            "min": round(min(durations), 3),
            "max": round(max(durations), 3),
        }

    # Bimodal detection: check if judge times cluster into two groups
    bimodal_info = _detect_bimodal(alt_judge_times)

    return {
        "scenario": "alternation_test",
        "creator_model": creator_model,
        "judge_model": judge_model,
        "rounds": rounds,
        "alternating": {
            "total_time_s": alt_total,
            "steps": alternating_steps,
            "create_stats": _stats(alt_create_times),
            "judge_stats": _stats(alt_judge_times),
        },
        "batched": {
            "total_time_s": batch_total,
            "creates": batched_creates,
            "judges": batched_judges,
            "create_stats": _stats(batch_create_times),
            "judge_stats": _stats(batch_judge_times),
        },
        "time_saved_by_batching_s": round(alt_total - batch_total, 3),
        "batching_speedup_pct": round((1 - batch_total / alt_total) * 100, 1) if alt_total else 0,
        "bimodal_judge_detection": bimodal_info,
    }


def _detect_bimodal(durations: list[float]) -> dict[str, Any]:
    """Check if durations show bimodal distribution (fast vs slow clusters).

    Uses a simple gap-detection heuristic: sort durations, find the largest
    gap. If the gap is > 2x the median, flag as bimodal.

    Args:
        durations: List of timing values.

    Returns:
        Dict with is_bimodal flag and cluster info.
    """
    if len(durations) < 4:
        return {"is_bimodal": False, "reason": "too few samples", "n": len(durations)}

    sorted_d = sorted(durations)
    gaps = [(sorted_d[i + 1] - sorted_d[i], i) for i in range(len(sorted_d) - 1)]
    max_gap, gap_idx = max(gaps, key=lambda x: x[0])
    median_val = statistics.median(sorted_d)

    is_bimodal = max_gap > median_val * 0.5 and len(sorted_d) >= 4

    fast_cluster = sorted_d[: gap_idx + 1]
    slow_cluster = sorted_d[gap_idx + 1 :]

    return {
        "is_bimodal": is_bimodal,
        "max_gap_s": round(max_gap, 3),
        "median_s": round(median_val, 3),
        "gap_to_median_ratio": round(max_gap / median_val, 2) if median_val else 0,
        "fast_cluster": {
            "n": len(fast_cluster),
            "median_s": round(statistics.median(fast_cluster), 3),
        },
        "slow_cluster": {
            "n": len(slow_cluster),
            "median_s": round(statistics.median(slow_cluster), 3),
        },
        "n": len(durations),
    }


# =====================================================================
# Scenario 3: GPU residency monitoring during transitions
# =====================================================================
def run_gpu_residency_test(
    creator_model: str, judge_model: str, rounds: int, verbose: bool
) -> dict[str, Any]:
    """Monitor GPU VRAM during model transitions to detect partial residency.

    Polls nvidia-smi before and after each model call to track exactly how
    much VRAM each model occupies and whether models overlap in VRAM.

    Args:
        creator_model: Model for creative generation.
        judge_model: Model for quality judging.
        rounds: Number of transitions to observe.
        verbose: Print detailed output.

    Returns:
        Dict with per-transition VRAM snapshots and loaded model lists.
    """
    print("\n  Scenario 3: GPU residency monitoring during transitions")

    if not nvidia_smi_available():
        print("    SKIPPED: nvidia-smi not available")
        return {"scenario": "gpu_residency", "skipped": True, "reason": "no nvidia-smi"}

    prompt = f"Name a unique fantasy location for:\n{CANONICAL_BRIEF[:150]}"

    # Unload everything
    unload_model(creator_model)
    unload_model(judge_model)
    time.sleep(2)

    baseline = get_gpu_vram_usage()
    transitions: list[dict[str, Any]] = []

    for i in range(rounds):
        # Before creator
        pre_create = get_gpu_vram_usage()
        get_ollama_loaded_models()  # ensure Ollama API is responsive

        create_result = _call_model(creator_model, prompt, temperature=0.9)
        time.sleep(0.5)

        # After creator, before judge
        post_create = get_gpu_vram_usage()
        loaded_mid = get_ollama_loaded_models()

        judge_input = f"Rate this name 1-10: {create_result['response'][:100]}"
        judge_result = _call_model(judge_model, judge_input, temperature=0.1)
        time.sleep(0.5)

        # After judge
        post_judge = get_gpu_vram_usage()
        loaded_post = get_ollama_loaded_models()

        transition = {
            "round": i + 1,
            "pre_create_vram_mb": pre_create["used_mb"],
            "post_create_vram_mb": post_create["used_mb"],
            "post_judge_vram_mb": post_judge["used_mb"],
            "creator_vram_delta_mb": post_create["used_mb"] - pre_create["used_mb"],
            "judge_vram_delta_mb": post_judge["used_mb"] - post_create["used_mb"],
            "models_after_create": [m["name"] for m in loaded_mid],
            "models_after_judge": [m["name"] for m in loaded_post],
            "both_models_resident": len(loaded_mid) > 1 or len(loaded_post) > 1,
            "create_duration_s": create_result["duration_s"],
            "judge_duration_s": judge_result["duration_s"],
        }
        transitions.append(transition)

        if verbose:
            print(
                f"      Round {i + 1}: "
                f"VRAM {pre_create['used_mb']}→{post_create['used_mb']}→{post_judge['used_mb']}MB "
                f"models_after_judge={[m['name'] for m in loaded_post]} "
                f"create={create_result['duration_s']:.2f}s judge={judge_result['duration_s']:.2f}s"
            )

    # Detect if Ollama evicts the previous model or keeps both resident
    both_resident_count = sum(1 for t in transitions if t["both_models_resident"])
    total_vram_mb = baseline.get("total_mb", 0)

    return {
        "scenario": "gpu_residency",
        "creator_model": creator_model,
        "judge_model": judge_model,
        "rounds": rounds,
        "baseline_vram_mb": baseline["used_mb"],
        "total_vram_mb": total_vram_mb,
        "transitions": transitions,
        "both_models_resident_count": both_resident_count,
        "finding": (
            f"Both models co-resident in {both_resident_count}/{rounds} transitions — "
            "Ollama keeps both in VRAM (no swap overhead)"
            if both_resident_count > rounds * 0.5
            else f"Models swapped in {rounds - both_resident_count}/{rounds} transitions — "
            "Ollama evicts one model to load the other"
        ),
    }


# =====================================================================
# Scenario 4: Creator temperature diversity comparison
# =====================================================================
def run_diversity_test(
    creator_model: str,
    temps: list[float],
    builds_per_temp: int,
    verbose: bool,
) -> dict[str, Any]:
    """Compare world-building diversity across different creator temperatures.

    For each temperature, generates N sets of entity names and descriptions,
    then measures uniqueness and structural variety.

    Args:
        creator_model: Model for creative generation.
        temps: List of temperatures to test.
        builds_per_temp: Number of builds per temperature.
        verbose: Print detailed output.

    Returns:
        Dict with per-temperature diversity metrics.
    """
    print("\n  Scenario 4: Creator temperature diversity comparison")

    entity_prompt = (
        f"Story premise: {CANONICAL_BRIEF[:300]}\n\n"
        "Create 5 unique fantasy factions for this world. For each, provide:\n"
        "- Name (unique, creative)\n"
        "- Type (government, religious, military, criminal, scholarly, etc.)\n"
        "- One-sentence description\n\n"
        "Format: Name | Type | Description (one per line)"
    )

    temp_results: list[dict[str, Any]] = []

    for temp in temps:
        print(f"    Temperature {temp}:")
        builds: list[dict[str, Any]] = []
        all_names: list[str] = []
        all_types: list[str] = []

        for b in range(builds_per_temp):
            result = _call_model(creator_model, entity_prompt, temperature=temp)
            if result["error"]:
                builds.append({"build": b + 1, "error": result["error"]})
                continue

            # Parse faction names from response
            lines = [
                line.strip()
                for line in result["response"].split("\n")
                if "|" in line and line.strip()
            ]
            names = []
            types = []
            for line in lines:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    # Strip leading numbering (e.g., "1. ", "- ")
                    name = parts[0].lstrip("0123456789.-) ").strip()
                    if name:
                        names.append(name.lower())
                        types.append(parts[1].lower().strip() if len(parts) > 1 else "")

            all_names.extend(names)
            all_types.extend(types)
            builds.append(
                {
                    "build": b + 1,
                    "names": names,
                    "types": types,
                    "duration_s": result["duration_s"],
                }
            )
            if verbose:
                print(f"      Build {b + 1}: {len(names)} factions, {result['duration_s']:.2f}s")

        # Compute diversity metrics
        unique_names = len(set(all_names))
        total_names = len(all_names)
        unique_types = len(set(all_types))
        total_types = len(all_types)

        # Name similarity: what fraction of names are unique across builds?
        name_uniqueness = unique_names / total_names if total_names else 0
        type_diversity = unique_types / total_types if total_types else 0

        temp_results.append(
            {
                "temperature": temp,
                "builds": builds,
                "metrics": {
                    "total_factions": total_names,
                    "unique_names": unique_names,
                    "name_uniqueness_ratio": round(name_uniqueness, 3),
                    "unique_types": unique_types,
                    "type_diversity_ratio": round(type_diversity, 3),
                    "all_unique_names": sorted(set(all_names)),
                    "all_unique_types": sorted(set(all_types)),
                },
            }
        )

        if verbose:
            print(
                f"      → {unique_names}/{total_names} unique names "
                f"({name_uniqueness:.1%}), {unique_types} type variants"
            )

    return {
        "scenario": "diversity_comparison",
        "creator_model": creator_model,
        "temperatures_tested": temps,
        "builds_per_temp": builds_per_temp,
        "results": temp_results,
    }


# =====================================================================
# Main
# =====================================================================
def _select_models(
    args: argparse.Namespace,
) -> tuple[str, str]:
    """Select creator and judge models from args or auto-detect.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (creator_model, judge_model).
    """
    if args.creator_model and args.judge_model:
        return args.creator_model, args.judge_model

    installed = get_installed_models()
    if not installed:
        print("ERROR: No models found. Is Ollama running?")
        sys.exit(1)

    creator = args.creator_model or installed[0]

    if args.judge_model:
        judge = args.judge_model
    elif len(installed) >= 2:
        # Pick a different model for judge
        judge = next((m for m in installed if m != creator), installed[1])
    else:
        print(
            f"WARNING: Only 1 model installed ({creator}). "
            "Using same model for creator and judge — model switching test will show no swap."
        )
        judge = creator

    return creator, judge


def main() -> None:
    """Run the model switching investigation."""
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
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Select models
    creator_model, judge_model = _select_models(args)
    temps = [float(t.strip()) for t in args.temps.split(",")]

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"model_switching_{timestamp}.json"

    has_gpu = nvidia_smi_available() and not args.skip_gpu
    monitor_gpu = has_gpu

    # Model info
    creator_info = get_model_info(creator_model)
    judge_info = get_model_info(judge_model)

    # Print header
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

    overall_start = time.monotonic()
    scenarios: list[dict[str, Any]] = []

    # Scenario 1: Cold-start penalty
    cold_start = run_cold_start_test(creator_model, monitor_gpu, args.verbose)
    scenarios.append(cold_start)
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

    # Scenario 2: Alternating vs batched
    alternation = run_alternation_test(
        creator_model, judge_model, args.rounds, monitor_gpu, args.verbose
    )
    scenarios.append(alternation)
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

    # Scenario 3: GPU residency (if GPU available)
    if monitor_gpu:
        gpu_residency = run_gpu_residency_test(
            creator_model, judge_model, args.rounds, args.verbose
        )
        scenarios.append(gpu_residency)
        print(f"  => {gpu_residency['finding']}")
    else:
        print("\n  Scenario 3: GPU residency — SKIPPED (no GPU or --skip-gpu)")

    # Scenario 4: Diversity comparison
    if not args.skip_diversity:
        diversity = run_diversity_test(creator_model, temps, args.diversity_builds, args.verbose)
        scenarios.append(diversity)
        print("  => Diversity results:")
        for tr in diversity["results"]:
            m = tr["metrics"]
            print(
                f"      temp={tr['temperature']}: "
                f"{m['unique_names']}/{m['total_factions']} unique names "
                f"({m['name_uniqueness_ratio']:.1%}), "
                f"{m['unique_types']} type variants"
            )
    else:
        print("\n  Scenario 4: Diversity comparison — SKIPPED (--skip-diversity)")

    # Clean up
    print("\nCleaning up — unloading models...")
    unload_model(creator_model)
    unload_model(judge_model)

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build output
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

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total investigation time: {overall_time:.0f}s")
    print()

    print("COLD-START:")
    print(f"  Cold penalty: {cold_start['cold_penalty_s']:.2f}s")
    print(f"  Warm-up ping cost: {wp['warmup_time_s']:.2f}s")
    print(f"  Net savings from warm-up: {cold_start['warmup_savings_s']:.2f}s")
    if cold_start["warmup_savings_s"] > 0:
        print("  RECOMMENDATION: Add warm-up ping before first build step")
    else:
        print("  FINDING: Warm-up ping does not save time (model loads fast enough)")
    print()

    print("MODEL SWITCHING:")
    print(
        f"  Batching saves {alternation['time_saved_by_batching_s']:.1f}s "
        f"({alternation['batching_speedup_pct']:.1f}%) over {args.rounds} rounds"
    )
    if alternation["batching_speedup_pct"] > 10:
        print("  RECOMMENDATION: Consider batching entity creation before judging")
    else:
        print("  FINDING: Switching overhead is minimal — batching not necessary")
    print()

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

    print(f"Results written to {output_path}")
    print()


if __name__ == "__main__":
    main()
