"""Scenario functions for the model switching investigation.

Contains the four investigation scenarios:
1. Cold-start penalty measurement
2. Alternating vs batched model transitions
3. GPU residency monitoring during transitions
4. Creator temperature diversity comparison
"""

import logging
import statistics
import time
from typing import Any

from scripts._model_switching_helpers import (
    CallResult,
    call_model,
    compute_stats,
    detect_bimodal,
)
from scripts._ollama_helpers import CANONICAL_BRIEF, INVESTIGATION_NUM_CTX, unload_model, warm_model
from scripts.investigate_vram_usage import (
    get_gpu_vram_usage,
    get_ollama_loaded_models,
    nvidia_smi_available,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Scenario 1: Cold-start penalty
# =====================================================================
def _measure_cold_call(
    model: str, prompt: str, monitor_gpu: bool, verbose: bool
) -> tuple[CallResult, dict[str, Any]]:
    """Execute a cold call (model not in VRAM) and return result with VRAM snapshot.

    Args:
        model: Model to call.
        prompt: Prompt to send.
        monitor_gpu: Whether to record GPU VRAM.
        verbose: Print detailed output.

    Returns:
        Tuple of (call_result, post_call_vram_snapshot).
    """
    if verbose:
        print(f"    Cold call to {model}...")
    result = call_model(model, prompt, temperature=0.7)
    if result["error"]:
        logger.warning("Cold call to %s failed: %s", model, result["error"])
        print(f"    WARNING: Cold call failed: {result['error']}")
    time.sleep(1)
    post_vram = get_gpu_vram_usage() if monitor_gpu else {}
    if verbose:
        print(f"      Cold: {result['duration_s']:.2f}s")
    return result, post_vram


def _measure_warm_calls(model: str, prompt: str, count: int, verbose: bool) -> list[CallResult]:
    """Execute warm calls (model already loaded) and return results.

    Args:
        model: Model to call.
        prompt: Prompt to send.
        count: Number of warm calls.
        verbose: Print detailed output.

    Returns:
        List of call results.
    """
    results: list[CallResult] = []
    for i in range(count):
        if verbose:
            print(f"    Warm call {i + 1}...")
        result = call_model(model, prompt, temperature=0.7)
        if result["error"]:
            logger.warning("Warm call %d to %s failed: %s", i + 1, model, result["error"])
        results.append(result)
        if verbose:
            print(f"      Warm {i + 1}: {result['duration_s']:.2f}s")
    return results


def _measure_warmup_ping(model: str, prompt: str, verbose: bool) -> tuple[float, bool, CallResult]:
    """Execute warmup ping followed by a call, measuring both.

    Args:
        model: Model to warm up and call.
        prompt: Prompt to send after warmup.
        verbose: Print detailed output.

    Returns:
        Tuple of (warmup_time_s, warmup_succeeded, post_warmup_call_result).
    """
    if verbose:
        print(f"    Warm-up ping to {model}...")
    warmup_start = time.monotonic()
    warmup_ok = warm_model(model, num_ctx=INVESTIGATION_NUM_CTX)
    warmup_time = round(time.monotonic() - warmup_start, 3)
    if not warmup_ok:
        logger.warning("Warm-up ping failed for %s — post-warmup measurement may be invalid", model)
        print(f"    WARNING: Warm-up ping failed for {model}")
    if verbose:
        print(f"      Warm-up ping took: {warmup_time:.2f}s")
        print(f"    Post-warmup call to {model}...")
    result = call_model(model, prompt, temperature=0.7)
    if result["error"]:
        logger.warning("Post-warmup call to %s failed: %s", model, result["error"])
    if verbose:
        print(f"      Post-warmup: {result['duration_s']:.2f}s")
    return warmup_time, warmup_ok, result


def run_cold_start_test(model: str, monitor_gpu: bool, verbose: bool) -> dict[str, Any]:
    """Measure cold-start penalty: first call after full unload vs warm call.

    Sequence: unload -> cold call -> warm calls -> unload -> warm-up ping -> call.

    Args:
        model: Model to test.
        monitor_gpu: Whether to record GPU VRAM.
        verbose: Print detailed output.

    Returns:
        Dict with cold_call, warm_calls, warm_median_s, cold_penalty_s,
        warmup_ping timings, warmup_savings_s, and baseline_vram.
    """
    logger.info("Starting cold-start penalty test for %s", model)
    print("\n  Scenario 1: Cold-start penalty measurement")

    if not unload_model(model):
        logger.warning("Failed to unload %s — cold-start measurement may be invalid", model)
        print(f"    WARNING: Failed to unload {model}. Cold-start data may be inaccurate.")
    time.sleep(2)
    baseline_vram = get_gpu_vram_usage() if monitor_gpu else {}

    prompt = f"Given this story premise, name 3 characters:\n{CANONICAL_BRIEF[:200]}"

    cold_result, post_cold_vram = _measure_cold_call(model, prompt, monitor_gpu, verbose)
    warm_results = _measure_warm_calls(model, prompt, 3, verbose)

    warm_durations = [r["duration_s"] for r in warm_results if r["error"] is None]
    warm_median = statistics.median(warm_durations) if warm_durations else 0.0
    if not warm_durations:
        logger.warning("All warm calls failed — cold penalty calculation unavailable")

    if not unload_model(model):
        logger.warning("Failed to unload %s before warmup ping test", model)
    time.sleep(2)

    warmup_time, warmup_ok, warmed_result = _measure_warmup_ping(model, prompt, verbose)

    cold_penalty = round(cold_result["duration_s"] - warm_median, 3) if warm_median else 0.0
    logger.info(
        "Cold-start test complete: penalty=%.3fs, warm_median=%.3fs", cold_penalty, warm_median
    )

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
            "warmup_succeeded": warmup_ok,
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
def _run_alternating_pattern(
    creator_model: str,
    judge_model: str,
    rounds: int,
    create_prompt: str,
    judge_prompt: str,
    monitor_gpu: bool,
    verbose: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Pattern A: alternating create->judge per round.

    Args:
        creator_model: Model for creative generation.
        judge_model: Model for quality judging.
        rounds: Number of create+judge pairs.
        create_prompt: Prompt for creator.
        judge_prompt: Prompt template for judge.
        monitor_gpu: Whether to record GPU VRAM.
        verbose: Print detailed output.

    Returns:
        Tuple of (steps_list, total_time_s).
    """
    steps: list[dict[str, Any]] = []
    start = time.monotonic()

    for i in range(rounds):
        vram_before = get_gpu_vram_usage() if monitor_gpu else None
        create_result = call_model(creator_model, create_prompt, temperature=0.9)
        vram_after_create = get_gpu_vram_usage() if monitor_gpu else None
        if create_result["error"]:
            logger.warning("Alternation round %d create failed: %s", i + 1, create_result["error"])

        faction_text = create_result["response"][:300]
        judge_input = f"{judge_prompt}\n\nFaction: {faction_text}"
        judge_result = call_model(judge_model, judge_input, temperature=0.1)
        vram_after_judge = get_gpu_vram_usage() if monitor_gpu else None
        if judge_result["error"]:
            logger.warning("Alternation round %d judge failed: %s", i + 1, judge_result["error"])

        steps.append(
            {
                "round": i + 1,
                "create": {
                    "duration_s": create_result["duration_s"],
                    "error": create_result["error"],
                },
                "judge": {
                    "duration_s": judge_result["duration_s"],
                    "error": judge_result["error"],
                },
                "vram_before": vram_before,
                "vram_after_create": vram_after_create,
                "vram_after_judge": vram_after_judge,
            }
        )

        if verbose:
            print(
                f"      Round {i + 1}: create={create_result['duration_s']:.2f}s "
                f"judge={judge_result['duration_s']:.2f}s"
            )

    total = round(time.monotonic() - start, 3)
    logger.info("Alternating pattern complete: %d rounds in %.1fs", rounds, total)
    return steps, total


def _run_batched_pattern(
    creator_model: str,
    judge_model: str,
    rounds: int,
    create_prompt: str,
    judge_prompt: str,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Pattern B: all creates first, then all judges.

    Args:
        creator_model: Model for creative generation.
        judge_model: Model for quality judging.
        rounds: Number of create+judge pairs.
        create_prompt: Prompt for creator.
        judge_prompt: Prompt template for judge.
        verbose: Print detailed output.

    Returns:
        Tuple of (creates_list, judges_list, total_time_s).
    """
    creates: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []
    faction_texts: list[str] = []
    start = time.monotonic()

    for i in range(rounds):
        result = call_model(creator_model, create_prompt, temperature=0.9)
        if result["error"]:
            logger.warning("Batched create %d failed: %s", i + 1, result["error"])
        creates.append(
            {
                "round": i + 1,
                "duration_s": result["duration_s"],
                "error": result["error"],
            }
        )
        faction_texts.append(result["response"][:300])
        if verbose:
            print(f"      Create {i + 1}: {result['duration_s']:.2f}s")

    for i in range(rounds):
        judge_input = f"{judge_prompt}\n\nFaction: {faction_texts[i]}"
        result = call_model(judge_model, judge_input, temperature=0.1)
        if result["error"]:
            logger.warning("Batched judge %d failed: %s", i + 1, result["error"])
        judges.append(
            {
                "round": i + 1,
                "duration_s": result["duration_s"],
                "error": result["error"],
            }
        )
        if verbose:
            print(f"      Judge {i + 1}: {result['duration_s']:.2f}s")

    total = round(time.monotonic() - start, 3)
    logger.info("Batched pattern complete: %d rounds in %.1fs", rounds, total)
    return creates, judges, total


def run_alternation_test(
    creator_model: str,
    judge_model: str,
    rounds: int,
    monitor_gpu: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Compare alternating creator<->judge vs batched creator-then-judge patterns.

    Pattern A (alternating): create->judge->create->judge->... (simulates quality loop)
    Pattern B (batched): create->create->...->judge->judge->... (hypothetical batch mode)

    Args:
        creator_model: Model for creative generation.
        judge_model: Model for quality judging.
        rounds: Number of create+judge pairs.
        monitor_gpu: Whether to record GPU VRAM.
        verbose: Print detailed output.

    Returns:
        Dict with per-call timings, VRAM snapshots, transition overhead,
        batching speedup metrics, and bimodal detection results.
    """
    logger.info(
        "Starting alternation test: %s vs %s (%d rounds)", creator_model, judge_model, rounds
    )
    print("\n  Scenario 2: Alternating vs batched model transitions")

    create_prompt = (
        f"Create a unique fantasy faction for this story:\n{CANONICAL_BRIEF[:200]}\n"
        "Reply with a faction name, leader, and 2-sentence description."
    )
    judge_prompt = (
        "Rate this faction on a scale of 1-10 for originality, coherence, and depth. "
        "Reply with just the three scores and a 1-sentence rationale."
    )

    # Pattern A: Alternating
    print("    Pattern A: Alternating (create->judge->create->judge->...)")
    if not unload_model(creator_model):
        logger.warning("Failed to unload %s before alternation test", creator_model)
    if not unload_model(judge_model):
        logger.warning("Failed to unload %s before alternation test", judge_model)
    time.sleep(2)

    alt_steps, alt_total = _run_alternating_pattern(
        creator_model, judge_model, rounds, create_prompt, judge_prompt, monitor_gpu, verbose
    )

    # Pattern B: Batched
    print("    Pattern B: Batched (create->create->...->judge->judge->...)")
    if not unload_model(creator_model):
        logger.warning("Failed to unload %s before batched test", creator_model)
    if not unload_model(judge_model):
        logger.warning("Failed to unload %s before batched test", judge_model)
    time.sleep(2)

    batched_creates, batched_judges, batch_total = _run_batched_pattern(
        creator_model, judge_model, rounds, create_prompt, judge_prompt, verbose
    )

    # Compute stats
    alt_create_times = [
        s["create"]["duration_s"] for s in alt_steps if s["create"]["error"] is None
    ]
    alt_judge_times = [s["judge"]["duration_s"] for s in alt_steps if s["judge"]["error"] is None]
    batch_create_times = [s["duration_s"] for s in batched_creates if s["error"] is None]
    batch_judge_times = [s["duration_s"] for s in batched_judges if s["error"] is None]

    bimodal_info = detect_bimodal(alt_judge_times)
    speedup = round((1 - batch_total / alt_total) * 100, 1) if alt_total else 0.0

    logger.info(
        "Alternation test complete: alt=%.1fs, batch=%.1fs, speedup=%.1f%%",
        alt_total,
        batch_total,
        speedup,
    )

    return {
        "scenario": "alternation_test",
        "creator_model": creator_model,
        "judge_model": judge_model,
        "rounds": rounds,
        "alternating": {
            "total_time_s": alt_total,
            "steps": alt_steps,
            "create_stats": compute_stats(alt_create_times),
            "judge_stats": compute_stats(alt_judge_times),
        },
        "batched": {
            "total_time_s": batch_total,
            "creates": batched_creates,
            "judges": batched_judges,
            "create_stats": compute_stats(batch_create_times),
            "judge_stats": compute_stats(batch_judge_times),
        },
        "time_saved_by_batching_s": round(alt_total - batch_total, 3),
        "batching_speedup_pct": speedup,
        "bimodal_judge_detection": bimodal_info,
    }


# =====================================================================
# Scenario 3: GPU residency monitoring during transitions
# =====================================================================
def _observe_transition(
    creator_model: str,
    judge_model: str,
    prompt: str,
    round_num: int,
    verbose: bool,
) -> dict[str, Any]:
    """Observe a single creator->judge transition with VRAM monitoring.

    Args:
        creator_model: Model for creative generation.
        judge_model: Model for quality judging.
        prompt: Prompt to send.
        round_num: Current round number (1-indexed).
        verbose: Print detailed output.

    Returns:
        Dict with VRAM snapshots, loaded model lists, and timing.
    """
    pre_create = get_gpu_vram_usage()
    if pre_create.get("error"):
        logger.warning("VRAM read failed before creator: %s", pre_create["error"])

    create_result = call_model(creator_model, prompt, temperature=0.9)
    if create_result["error"]:
        logger.warning(
            "GPU residency round %d create failed: %s", round_num, create_result["error"]
        )
    time.sleep(0.5)

    post_create = get_gpu_vram_usage()
    if post_create.get("error"):
        logger.warning("VRAM read failed after creator: %s", post_create["error"])
    loaded_mid = get_ollama_loaded_models()

    judge_input = f"Rate this name 1-10: {create_result['response'][:100]}"
    judge_result = call_model(judge_model, judge_input, temperature=0.1)
    if judge_result["error"]:
        logger.warning("GPU residency round %d judge failed: %s", round_num, judge_result["error"])
    time.sleep(0.5)

    post_judge = get_gpu_vram_usage()
    if post_judge.get("error"):
        logger.warning("VRAM read failed after judge: %s", post_judge["error"])
    loaded_post = get_ollama_loaded_models()

    # Approximate: >1 loaded model suggests co-residency (may include unrelated models)
    both_resident = len(loaded_mid) > 1 or len(loaded_post) > 1
    if not loaded_mid and not loaded_post:
        logger.warning(
            "Round %d: could not determine model residency (API returned empty)", round_num
        )

    transition = {
        "round": round_num,
        "pre_create_vram_mb": pre_create["used_mb"],
        "post_create_vram_mb": post_create["used_mb"],
        "post_judge_vram_mb": post_judge["used_mb"],
        "creator_vram_delta_mb": post_create["used_mb"] - pre_create["used_mb"],
        "judge_vram_delta_mb": post_judge["used_mb"] - post_create["used_mb"],
        "models_after_create": [m["name"] for m in loaded_mid],
        "models_after_judge": [m["name"] for m in loaded_post],
        "both_models_resident": both_resident,
        "create_duration_s": create_result["duration_s"],
        "judge_duration_s": judge_result["duration_s"],
    }

    if verbose:
        print(
            f"      Round {round_num}: "
            f"VRAM {pre_create['used_mb']}->{post_create['used_mb']}->{post_judge['used_mb']}MB "
            f"models_after_judge={[m['name'] for m in loaded_post]} "
            f"create={create_result['duration_s']:.2f}s judge={judge_result['duration_s']:.2f}s"
        )

    return transition


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
        Dict with per-transition VRAM snapshots, loaded model lists, and finding.
    """
    logger.info(
        "Starting GPU residency test: %s vs %s (%d rounds)", creator_model, judge_model, rounds
    )
    print("\n  Scenario 3: GPU residency monitoring during transitions")

    if not nvidia_smi_available():
        logger.info("GPU residency test skipped: nvidia-smi not available")
        print("    SKIPPED: nvidia-smi not available")
        return {"scenario": "gpu_residency", "skipped": True, "reason": "no nvidia-smi"}

    prompt = f"Name a unique fantasy location for:\n{CANONICAL_BRIEF[:150]}"

    if not unload_model(creator_model):
        logger.warning("Failed to unload %s before GPU residency test", creator_model)
    if not unload_model(judge_model):
        logger.warning("Failed to unload %s before GPU residency test", judge_model)
    time.sleep(2)

    baseline = get_gpu_vram_usage()
    transitions: list[dict[str, Any]] = []

    for i in range(rounds):
        transition = _observe_transition(creator_model, judge_model, prompt, i + 1, verbose)
        transitions.append(transition)

    both_resident_count = sum(1 for t in transitions if t["both_models_resident"])
    empty_residency_count = sum(
        1 for t in transitions if not t["models_after_create"] and not t["models_after_judge"]
    )

    finding = (
        f"Both models co-resident in {both_resident_count}/{rounds} transitions — "
        "Ollama keeps both in VRAM (no swap overhead)"
        if both_resident_count > rounds * 0.5
        else f"Models swapped in {rounds - both_resident_count}/{rounds} transitions — "
        "Ollama evicts one model to load the other"
    )
    if empty_residency_count > 0:
        finding += f" (NOTE: {empty_residency_count} transitions had incomplete residency data)"

    logger.info("GPU residency test complete: co-resident=%d/%d", both_resident_count, rounds)

    return {
        "scenario": "gpu_residency",
        "creator_model": creator_model,
        "judge_model": judge_model,
        "rounds": rounds,
        "baseline_vram_mb": baseline["used_mb"],
        "total_vram_mb": baseline.get("total_mb", 0),
        "transitions": transitions,
        "both_models_resident_count": both_resident_count,
        "finding": finding,
    }


# =====================================================================
# Scenario 4: Creator temperature diversity comparison
# =====================================================================
def _run_diversity_for_temp(
    creator_model: str,
    entity_prompt: str,
    temp: float,
    builds_per_temp: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Run diversity builds at a single temperature.

    Args:
        creator_model: Model for creative generation.
        entity_prompt: Prompt for entity generation.
        temp: Temperature to test.
        builds_per_temp: Number of builds at this temperature.
        verbose: Print detailed output.

    Returns:
        Tuple of (builds_list, all_names, all_types).
    """
    builds: list[dict[str, Any]] = []
    all_names: list[str] = []
    all_types: list[str] = []

    for b in range(builds_per_temp):
        result = call_model(creator_model, entity_prompt, temperature=temp)
        if result["error"]:
            logger.warning(
                "Diversity build %d at temp %.1f failed: %s", b + 1, temp, result["error"]
            )
            builds.append({"build": b + 1, "error": result["error"]})
            continue

        lines = [
            line.strip() for line in result["response"].split("\n") if "|" in line and line.strip()
        ]
        names = []
        types = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
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
        logger.debug("Diversity build %d at temp %.1f: %d factions", b + 1, temp, len(names))
        if verbose:
            print(f"      Build {b + 1}: {len(names)} factions, {result['duration_s']:.2f}s")

    return builds, all_names, all_types


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
    logger.info("Starting diversity test: %s at temps %s", creator_model, temps)
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
        builds, all_names, all_types = _run_diversity_for_temp(
            creator_model, entity_prompt, temp, builds_per_temp, verbose
        )

        unique_names = len(set(all_names))
        total_names = len(all_names)
        unique_types = len(set(all_types))
        total_types = len(all_types)

        # Name uniqueness: what fraction of names are unique across builds?
        name_uniqueness = unique_names / total_names if total_names else 0.0
        type_diversity = unique_types / total_types if total_types else 0.0

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
                f"      -> {unique_names}/{total_names} unique names "
                f"({name_uniqueness:.1%}), {unique_types} type variants"
            )

    logger.info("Diversity test complete: %d temperatures tested", len(temps))

    return {
        "scenario": "diversity_comparison",
        "creator_model": creator_model,
        "temperatures_tested": temps,
        "builds_per_temp": builds_per_temp,
        "results": temp_results,
    }
