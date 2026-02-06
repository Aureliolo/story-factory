#!/usr/bin/env python3
"""VRAM Usage Investigation — measures actual VRAM during multi-model sessions.

Tests whether the app's VRAM management actually works by measuring VRAM via
nvidia-smi during model load/unload scenarios. This investigates issue #267 (A5):
VRAM overcommitment.

3 scenarios:
  A) Sequential with explicit unload (keep_alive=0)
  B) Sequential without unload (rely on Ollama LRU)
  C) 3 models loaded back-to-back (overcommit test)

Also checks whether _vram.py's unload_all_except() actually frees VRAM
(hypothesis: it does NOT — only clears internal tracking).

Falls back gracefully on non-NVIDIA systems (reports timing only).

Usage:
    python scripts/investigate_vram_usage.py [options]
      --models model1,model2,model3  (default: first 3 installed non-embedding models)
      --timeout 120                  (seconds per call, default: 120)
      --output results.json          (default: output/diagnostics/vram_usage_<ts>.json)
      --verbose
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from scripts._ollama_helpers import OLLAMA_BASE, get_model_info
from scripts._ollama_helpers import get_installed_models as _shared_get_installed_models
from scripts._ollama_helpers import unload_model as _shared_unload_model

logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Simple prompt to force model loading
WARMUP_PROMPT = "Say exactly one word: hello"


# =====================================================================
# GPU / VRAM helpers
# =====================================================================
def nvidia_smi_available() -> bool:
    """Check if nvidia-smi is available on this system.

    Returns:
        True if nvidia-smi is found in PATH.
    """
    return shutil.which("nvidia-smi") is not None


def get_gpu_vram_usage() -> dict[str, Any]:
    """Query nvidia-smi for current GPU VRAM usage.

    Returns:
        Dict with total_mb, used_mb, free_mb, utilization_pct, error.
        On non-NVIDIA systems, returns error="not_available".
    """
    if not nvidia_smi_available():
        return {
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "utilization_pct": 0.0,
            "error": "nvidia-smi not available",
        }

    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=creationflags,
        )
        if proc.returncode != 0:
            return {
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "utilization_pct": 0.0,
                "error": f"nvidia-smi exit code {proc.returncode}",
            }

        # Parse first GPU line (multi-GPU: only use first)
        line = proc.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        total = int(parts[0])
        used = int(parts[1])
        free = int(parts[2])
        util = float(parts[3])

        return {
            "total_mb": total,
            "used_mb": used,
            "free_mb": free,
            "utilization_pct": util,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "utilization_pct": 0.0,
            "error": "nvidia-smi timeout",
        }
    except Exception as e:
        return {
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "utilization_pct": 0.0,
            "error": str(e)[:200],
        }


def get_ollama_loaded_models() -> list[dict[str, Any]]:
    """Query Ollama for currently loaded models via /api/ps.

    Returns:
        List of dicts with model name, size_vram, etc.
    """
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/ps", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        return [
            {
                "name": m.get("name", "unknown"),
                "size": m.get("size", 0),
                "size_vram": m.get("size_vram", 0),
                "expires_at": m.get("expires_at", ""),
            }
            for m in models
        ]
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("Failed to query loaded models: %s", e)
        return []


def load_model_with_prompt(model: str, timeout: int) -> dict[str, Any]:
    """Force-load a model by sending a simple prompt.

    Args:
        model: Ollama model name.
        timeout: Request timeout in seconds.

    Returns:
        Dict with load_time, response, error.
    """
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": WARMUP_PROMPT}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 512},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        elapsed = round(time.monotonic() - start, 2)
        content = body.get("message", {}).get("content", "")
        logger.debug("Model %s loaded in %.1fs: %s", model, elapsed, content[:50])
        return {"load_time": elapsed, "response": content[:100], "error": None}
    except httpx.TimeoutException:
        elapsed = round(time.monotonic() - start, 2)
        return {"load_time": elapsed, "response": "", "error": "timeout"}
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        elapsed = round(time.monotonic() - start, 2)
        return {"load_time": elapsed, "response": "", "error": str(e)[:200]}


def unload_model_timed(model: str) -> dict[str, Any]:
    """Explicitly unload a model via keep_alive=0, returning timing data.

    Unlike the shared unload_model(), this variant returns a dict with
    unload_time and error info for VRAM measurement purposes.

    Args:
        model: Model name to unload.

    Returns:
        Dict with unload_time, error.
    """
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=60,
        )
        resp.raise_for_status()
        elapsed = round(time.monotonic() - start, 2)
        logger.debug("Unloaded model '%s' in %.1fs", model, elapsed)
        return {"unload_time": elapsed, "error": None}
    except httpx.HTTPStatusError as e:
        elapsed = round(time.monotonic() - start, 2)
        status_code = e.response.status_code if e.response is not None else "unknown"
        body = (e.response.text or "")[:200] if e.response is not None else ""
        logger.warning("Failed to unload model '%s' (status %s): %s", model, status_code, body)
        return {"unload_time": elapsed, "error": f"HTTP {status_code}: {body}"}
    except httpx.HTTPError as e:
        elapsed = round(time.monotonic() - start, 2)
        logger.warning("Failed to unload model '%s': %s", model, e)
        return {"unload_time": elapsed, "error": str(e)[:200]}


def unload_all_models(models: list[str]) -> None:
    """Unload all specified models to start clean.

    Args:
        models: List of model names to unload.
    """
    for model in models:
        _shared_unload_model(model)
    # Wait briefly for VRAM to settle
    time.sleep(2)


# =====================================================================
# Scenario runners
# =====================================================================
def run_scenario_sequential_unload(
    models: list[str], timeout: int, verbose: bool
) -> dict[str, Any]:
    """Scenario A: Load each model sequentially, explicitly unload between each.

    This is the ideal case — measures per-model VRAM usage in isolation.

    Args:
        models: List of model names to test.
        timeout: Request timeout per call.
        verbose: Print detailed output.

    Returns:
        Scenario result dict with per-step VRAM measurements.
    """
    print("  Scenario A: Sequential with explicit unload")
    steps: list[dict[str, Any]] = []

    # Clean slate
    unload_all_models(models)
    baseline_vram = get_gpu_vram_usage()
    steps.append(
        {
            "action": "baseline",
            "vram": baseline_vram,
            "loaded_models": get_ollama_loaded_models(),
        }
    )

    peak_used = baseline_vram["used_mb"]

    for model in models:
        # Load model
        if verbose:
            print(f"    Loading {model}...")
        load_result = load_model_with_prompt(model, timeout)
        time.sleep(1)  # Let VRAM settle
        post_load_vram = get_gpu_vram_usage()
        loaded_models = get_ollama_loaded_models()

        steps.append(
            {
                "action": f"load_{model}",
                "load_result": load_result,
                "vram": post_load_vram,
                "loaded_models": loaded_models,
                "vram_delta_mb": post_load_vram["used_mb"] - baseline_vram["used_mb"],
            }
        )

        peak_used = max(peak_used, post_load_vram["used_mb"])

        if verbose:
            print(
                f"      VRAM: {post_load_vram['used_mb']}MB "
                f"(+{post_load_vram['used_mb'] - baseline_vram['used_mb']}MB from baseline)"
            )

        # Unload model
        if verbose:
            print(f"    Unloading {model}...")
        unload_result = unload_model_timed(model)
        time.sleep(2)  # Wait for VRAM release
        post_unload_vram = get_gpu_vram_usage()

        steps.append(
            {
                "action": f"unload_{model}",
                "unload_result": unload_result,
                "vram": post_unload_vram,
                "loaded_models": get_ollama_loaded_models(),
                "vram_recovered_mb": (post_load_vram["used_mb"] - post_unload_vram["used_mb"]),
            }
        )

        if verbose:
            recovered = post_load_vram["used_mb"] - post_unload_vram["used_mb"]
            print(
                f"      VRAM after unload: {post_unload_vram['used_mb']}MB (recovered {recovered}MB)"
            )

    return {
        "scenario": "A_sequential_unload",
        "description": "Load each model, explicitly unload via keep_alive=0 between each",
        "steps": steps,
        "peak_vram_mb": peak_used,
        "baseline_vram_mb": baseline_vram["used_mb"],
    }


def run_scenario_sequential_no_unload(
    models: list[str], timeout: int, verbose: bool
) -> dict[str, Any]:
    """Scenario B: Load each model sequentially, do NOT explicitly unload.

    Tests whether Ollama's LRU caching handles model replacement.

    Args:
        models: List of model names to test.
        timeout: Request timeout per call.
        verbose: Print detailed output.

    Returns:
        Scenario result dict.
    """
    print("  Scenario B: Sequential without explicit unload")
    steps: list[dict[str, Any]] = []

    # Clean slate
    unload_all_models(models)
    baseline_vram = get_gpu_vram_usage()
    steps.append(
        {
            "action": "baseline",
            "vram": baseline_vram,
            "loaded_models": get_ollama_loaded_models(),
        }
    )

    peak_used = baseline_vram["used_mb"]

    for model in models:
        if verbose:
            print(f"    Loading {model} (no unload)...")
        load_result = load_model_with_prompt(model, timeout)
        time.sleep(1)
        post_load_vram = get_gpu_vram_usage()
        loaded_models = get_ollama_loaded_models()

        steps.append(
            {
                "action": f"load_{model}",
                "load_result": load_result,
                "vram": post_load_vram,
                "loaded_models": loaded_models,
                "vram_delta_mb": post_load_vram["used_mb"] - baseline_vram["used_mb"],
                "models_in_vram": len(loaded_models),
            }
        )

        peak_used = max(peak_used, post_load_vram["used_mb"])

        if verbose:
            print(
                f"      VRAM: {post_load_vram['used_mb']}MB "
                f"(+{post_load_vram['used_mb'] - baseline_vram['used_mb']}MB) "
                f"models_in_vram={len(loaded_models)}"
            )

    return {
        "scenario": "B_sequential_no_unload",
        "description": "Load each model sequentially without explicit unload (rely on Ollama LRU)",
        "steps": steps,
        "peak_vram_mb": peak_used,
        "baseline_vram_mb": baseline_vram["used_mb"],
    }


def run_scenario_concurrent_load(models: list[str], timeout: int, verbose: bool) -> dict[str, Any]:
    """Scenario C: Load all models back-to-back rapidly (overcommit test).

    Tests the worst case: all models requested before any can be unloaded.

    Args:
        models: List of model names to test.
        timeout: Request timeout per call.
        verbose: Print detailed output.

    Returns:
        Scenario result dict.
    """
    print("  Scenario C: All models loaded back-to-back (overcommit test)")
    steps: list[dict[str, Any]] = []

    # Clean slate
    unload_all_models(models)
    baseline_vram = get_gpu_vram_usage()
    steps.append(
        {
            "action": "baseline",
            "vram": baseline_vram,
            "loaded_models": get_ollama_loaded_models(),
        }
    )

    peak_used = baseline_vram["used_mb"]

    # Load all models rapidly without waiting
    for model in models:
        if verbose:
            print(f"    Rapid loading {model}...")
        load_result = load_model_with_prompt(model, timeout)
        # Do NOT sleep or unload — measure immediately
        post_load_vram = get_gpu_vram_usage()
        loaded_models = get_ollama_loaded_models()

        steps.append(
            {
                "action": f"rapid_load_{model}",
                "load_result": load_result,
                "vram": post_load_vram,
                "loaded_models": loaded_models,
                "vram_delta_mb": post_load_vram["used_mb"] - baseline_vram["used_mb"],
                "models_in_vram": len(loaded_models),
            }
        )

        peak_used = max(peak_used, post_load_vram["used_mb"])

        if verbose:
            print(
                f"      VRAM: {post_load_vram['used_mb']}MB "
                f"(+{post_load_vram['used_mb'] - baseline_vram['used_mb']}MB) "
                f"models_in_vram={len(loaded_models)}"
            )

    # Check state after all loaded
    time.sleep(2)
    final_vram = get_gpu_vram_usage()
    final_models = get_ollama_loaded_models()
    steps.append(
        {
            "action": "final_state",
            "vram": final_vram,
            "loaded_models": final_models,
            "models_in_vram": len(final_models),
        }
    )

    total_vram = baseline_vram.get("total_mb", 0)
    overcommitted = peak_used > total_vram * 0.9 if total_vram > 0 else False

    return {
        "scenario": "C_concurrent_load",
        "description": "Load all models back-to-back without unloading (overcommit test)",
        "steps": steps,
        "peak_vram_mb": peak_used,
        "baseline_vram_mb": baseline_vram["used_mb"],
        "overcommit_detected": overcommitted,
        "final_models_in_vram": len(final_models),
    }


# =====================================================================
# _vram.py audit
# =====================================================================
def audit_vram_module() -> dict[str, Any]:
    """Audit _vram.py to verify whether unload_all_except actually calls Ollama unload.

    This is a static analysis — reads the source file and checks for Ollama API calls.

    Returns:
        Dict with audit findings.
    """
    vram_path = Path("src/services/model_mode_service/_vram.py")
    if not vram_path.exists():
        return {"error": f"File not found: {vram_path}", "calls_ollama_unload": "unknown"}

    source = vram_path.read_text(encoding="utf-8")

    # Check if unload_all_except calls any Ollama API
    has_httpx_call = "httpx" in source
    has_requests_call = "requests" in source
    has_ollama_api = "/api/" in source
    has_keep_alive = "keep_alive" in source

    # Check the unload function behavior
    has_tracking_only = (
        "just clear our tracking" in source.lower() or "only updates" in source.lower()
    )

    return {
        "file": str(vram_path),
        "calls_ollama_unload": has_httpx_call or has_requests_call or has_keep_alive,
        "has_api_call": has_ollama_api,
        "has_keep_alive": has_keep_alive,
        "tracking_only_comment_found": has_tracking_only,
        "finding": (
            "unload_all_except() only clears internal tracking (_loaded_models set). "
            "It does NOT call Ollama's keep_alive=0 API to actually free VRAM. "
            "VRAM freeing depends entirely on Ollama's internal LRU cache."
            if has_tracking_only and not has_keep_alive
            else "unload_all_except() appears to call Ollama's unload API."
        ),
    }


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the VRAM usage investigation."""
    parser = argparse.ArgumentParser(
        description=(
            "VRAM Usage Investigation — measures actual VRAM during multi-model sessions. "
            "Tests whether the app's VRAM management works."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names (default: first 3 installed non-embedding models)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per model load call in seconds (default: 120)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed VRAM measurements",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        all_models = _shared_get_installed_models()
        if not all_models:
            print("ERROR: No models found. Is Ollama running?")
            sys.exit(1)
        models = all_models[:3]
        if len(models) < 2:
            print(
                f"WARNING: Only {len(models)} model(s) installed. "
                "VRAM overcommit test needs at least 2 models."
            )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        diagnostics_dir = Path("output/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = diagnostics_dir / f"vram_usage_{timestamp}.json"

    # Check GPU availability
    has_gpu = nvidia_smi_available()

    # Get model sizes
    model_sizes = {}
    for m in models:
        model_sizes[m] = get_model_info(m)

    # Print header
    print("=" * 70)
    print("VRAM USAGE INVESTIGATION")
    print("=" * 70)
    print(f"GPU available: {'YES (nvidia-smi found)' if has_gpu else 'NO (timing only)'}")
    if has_gpu:
        gpu_info = get_gpu_vram_usage()
        print(f"GPU VRAM total: {gpu_info['total_mb']}MB")
        print(f"GPU VRAM used: {gpu_info['used_mb']}MB")
        print(f"GPU VRAM free: {gpu_info['free_mb']}MB")
    print(f"Models to test: {len(models)}")
    for m in models:
        size_info = model_sizes[m]
        print(f"  - {m} ({size_info['parameter_size']}, {size_info['quantization']})")
    print(f"Timeout per call: {args.timeout}s")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Audit _vram.py first (static analysis, no model loading)
    print("Auditing _vram.py...")
    vram_audit = audit_vram_module()
    print(f"  Finding: {vram_audit['finding']}")
    print()

    # Run scenarios
    overall_start = time.monotonic()
    scenarios: list[dict[str, Any]] = []

    scenario_a = run_scenario_sequential_unload(models, args.timeout, args.verbose)
    scenarios.append(scenario_a)
    print(
        f"  => Peak VRAM: {scenario_a['peak_vram_mb']}MB "
        f"(baseline: {scenario_a['baseline_vram_mb']}MB)"
    )
    print()

    scenario_b = run_scenario_sequential_no_unload(models, args.timeout, args.verbose)
    scenarios.append(scenario_b)
    print(
        f"  => Peak VRAM: {scenario_b['peak_vram_mb']}MB "
        f"(baseline: {scenario_b['baseline_vram_mb']}MB)"
    )
    print()

    scenario_c = run_scenario_concurrent_load(models, args.timeout, args.verbose)
    scenarios.append(scenario_c)
    print(
        f"  => Peak VRAM: {scenario_c['peak_vram_mb']}MB "
        f"(baseline: {scenario_c['baseline_vram_mb']}MB) "
        f"overcommit={'YES' if scenario_c.get('overcommit_detected') else 'no'}"
    )
    print()

    # Clean up — unload all models
    print("Cleaning up — unloading all models...")
    unload_all_models(models)

    overall_time = round(time.monotonic() - overall_start, 1)

    # Build output
    output = {
        "investigation_metadata": {
            "script": "investigate_vram_usage.py",
            "issue": "#267 — A5: VRAM Overcommitment",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_time_seconds": overall_time,
            "gpu_available": has_gpu,
            "models_tested": models,
            "model_sizes": model_sizes,
        },
        "vram_module_audit": vram_audit,
        "scenarios": scenarios,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_path}")

    # Print summary
    print()
    print("=" * 70)
    print("VRAM USAGE SUMMARY")
    print("=" * 70)

    if has_gpu:
        gpu_total = get_gpu_vram_usage()["total_mb"]
        print(f"GPU Total VRAM: {gpu_total}MB")
        print()

        print(f"{'Scenario':<50} {'Peak':>8} {'Baseline':>10} {'Delta':>8}")
        print("-" * 78)

        for s in scenarios:
            delta = s["peak_vram_mb"] - s["baseline_vram_mb"]
            print(
                f"{s['scenario']:<50} {s['peak_vram_mb']:>7}MB {s['baseline_vram_mb']:>9}MB {delta:>7}MB"
            )

        # Overcommit analysis
        print()
        if scenario_c.get("overcommit_detected"):
            print(
                "WARNING: VRAM overcommit detected in Scenario C! "
                "Peak usage >90% of total VRAM when loading multiple models."
            )
            print("RECOMMENDATION: Implement VRAM budget constraint before loading models.")
        else:
            print(
                "No VRAM overcommit detected. Ollama's LRU caching appears to handle model swapping."
            )

        # Unload effectiveness
        print()
        print("UNLOAD EFFECTIVENESS:")
        a_steps = [s for s in scenario_a["steps"] if s["action"].startswith("unload_")]
        for step in a_steps:
            recovered = step.get("vram_recovered_mb", 0)
            model_name = step["action"].replace("unload_", "")
            print(f"  {model_name}: recovered {recovered}MB on unload")
    else:
        print("No GPU detected. Timing data only (see JSON output).")
        print()
        for s in scenarios:
            load_times = [
                step.get("load_result", {}).get("load_time", 0)
                for step in s["steps"]
                if "load_result" in step
            ]
            avg_time = sum(load_times) / len(load_times) if load_times else 0
            print(f"  {s['scenario']}: avg load time = {avg_time:.1f}s")

    # _vram.py audit finding
    print()
    print("_VRAM.PY AUDIT:")
    print(f"  {vram_audit['finding']}")

    print()


if __name__ == "__main__":
    main()
