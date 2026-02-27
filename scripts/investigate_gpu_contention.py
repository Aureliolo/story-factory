#!/usr/bin/env python3
"""GPU Contention Investigation — measures latency impact of concurrent model loading.

Tests the hypothesis from issue #424: two models (qwen3:30b + mistral-small:24b)
competing for ~24GB VRAM causes bimodal latency in quality scoring.

3 phases:
  1) Baseline: single-model latency (creator only, then judge only)
  2) Contention: concurrent creator + judge calls
  3) Serialized: creator then judge sequentially (proposed fix)

Continuously logs nvidia-smi GPU stats and correlates with call latencies.

Usage:
    python scripts/investigate_gpu_contention.py [options]
      --creator MODEL   Creator model (default: auto-detect from settings)
      --judge MODEL     Judge model (default: auto-detect from settings)
      --samples N       Calls per phase (default: 5)
      --timeout SECS    Per-call timeout (default: 180)
      --output FILE     Output path (default: output/diagnostics/gpu_contention_<ts>.json)
      --verbose         Enable debug logging
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    get_model_info,
    unload_model,
    warm_model,
)
from src.memory.world_quality._entity_scores import CharacterQualityScores
from src.memory.world_quality._story_scores import RelationshipQualityScores

logger = logging.getLogger(__name__)

# Schema for creator calls — matches CHARACTER_CREATION_PROMPT output fields.
# CharacterQualityScores is for judge calls only; using it for creation would
# mismatch the prompt (creation asks for name/role/backstory, not quality dimensions).
CHARACTER_CREATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "description": {"type": "string"},
        "backstory": {"type": "string"},
        "goals": {"type": "string"},
        "flaws": {"type": "string"},
        "arc": {"type": "string"},
    },
    "required": ["name", "role", "description", "backstory", "goals", "flaws", "arc"],
}
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# =====================================================================
# Prompts that mirror real quality scoring calls
# =====================================================================

CHARACTER_CREATION_PROMPT = f"""You are a creative writer building characters for a story.

Story brief: {CANONICAL_BRIEF}

Create a detailed character for this story. Return a JSON object with these fields:
- name: The character's full name
- role: Their role in the story (protagonist, antagonist, supporting, etc.)
- description: A vivid 2-3 sentence description
- backstory: A paragraph about their history
- goals: What they want to achieve
- flaws: Their weaknesses and vulnerabilities
- arc: How they might change through the story
"""

CHARACTER_JUDGE_PROMPT = """You are a quality judge evaluating a character for a fantasy story.

Character to evaluate:
- Name: Lyra Ashenmere
- Role: Protagonist / Disgraced Archivist
- Description: A sharp-eyed woman in her thirties with ink-stained fingers and a \
perpetual frown. Her once-prestigious robes are now threadbare, but she carries \
herself with the rigid posture of someone who refuses to accept her fall from grace.
- Backstory: Formerly the youngest Head Archivist in the empire's history, Lyra \
was expelled after she discovered discrepancies in the historical records and \
refused to stay silent. The ruling council branded her a fabricator and stripped \
her credentials.
- Goals: To expose the council's memory-erasure program and restore the lost \
collective memories of the empire's citizens.
- Flaws: Obsessive attention to detail that blinds her to the bigger picture; \
deep-seated trust issues; tendency to work alone when collaboration would serve \
her better.

Rate this character on each dimension (0-10 scale) and provide feedback.
"""

RELATIONSHIP_JUDGE_PROMPT = """You are a quality judge evaluating a relationship between two \
characters in a fantasy story.

Relationship:
- Source: Lyra Ashenmere (Disgraced Archivist)
- Target: Councillor Maren Valdis (Head of the Memory Bureau)
- Type: Antagonistic / Former Mentor
- Description: Maren was Lyra's mentor and sponsor at the Archive. When Lyra \
discovered the memory erasure program, Maren chose to protect the council rather \
than her protege. Their relationship shifted from mutual respect to bitter enmity, \
complicated by the fact that Maren still harbors guilt over her betrayal.

Rate this relationship on each dimension (0-10 scale) and provide feedback.
"""


# =====================================================================
# GPU monitoring
# =====================================================================


class GpuMonitor:
    """Background thread that samples nvidia-smi at regular intervals."""

    def __init__(self, interval: float = 0.5):
        """Initialize GPU monitor with sampling interval in seconds."""
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start GPU monitoring in background thread."""
        if not shutil.which("nvidia-smi"):
            logger.warning("nvidia-smi not found — GPU monitoring disabled")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitor started (interval=%.1fs)", self.interval)

    def stop(self) -> list[dict[str, Any]]:
        """Stop monitoring and return all collected samples."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("GPU monitor stopped (%d samples)", len(self.samples))
        return list(self.samples)

    def _monitor_loop(self) -> None:
        """Poll nvidia-smi until stop event is set."""
        while not self._stop.is_set():
            sample = self._query_gpu()
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.interval)

    def _query_gpu(self) -> dict[str, Any] | None:
        """Single nvidia-smi query."""
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,"
                    "utilization.memory,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode != 0:
                return None
            parts = [p.strip() for p in proc.stdout.strip().split("\n")[0].split(",")]
            return {
                "timestamp": time.monotonic(),
                "wall_time": datetime.now(UTC).isoformat(),
                "vram_total_mb": int(parts[0]),
                "vram_used_mb": int(parts[1]),
                "vram_free_mb": int(parts[2]),
                "gpu_util_pct": float(parts[3]),
                "mem_util_pct": float(parts[4]),
                "temp_c": float(parts[5]),
                "power_w": float(parts[6]) if parts[6] != "[N/A]" else None,
            }
        except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as e:
            logger.debug("GPU query failed: %s: %s", type(e).__name__, e)
            return None


# =====================================================================
# Ollama call helpers
# =====================================================================


def get_loaded_models() -> list[dict[str, Any]]:
    """Query Ollama for currently loaded models."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/ps", timeout=10)
        resp.raise_for_status()
        return [
            {
                "name": m.get("name", "unknown"),
                "size_bytes": m.get("size", 0),
                "size_vram_bytes": m.get("size_vram", 0),
                "vram_pct": (
                    round(m.get("size_vram", 0) / m.get("size", 1) * 100, 1)
                    if m.get("size", 0) > 0
                    else 0
                ),
            }
            for m in resp.json().get("models", [])
        ]
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("Failed to query loaded models from Ollama: %s: %s", type(e).__name__, e)
        return []


def timed_chat(
    model: str,
    prompt: str,
    json_schema: dict[str, Any],
    temperature: float,
    timeout: int,
    label: str,
) -> dict[str, Any]:
    """Make a single timed Ollama chat call with structured output.

    Returns dict with timing, token counts, GPU snapshot, and loaded model info.
    """
    logger.info("[%s] Starting call to %s (temp=%.1f)", label, model, temperature)
    start = time.monotonic()
    start_wall = datetime.now(UTC).isoformat()

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": json_schema,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": INVESTIGATION_NUM_CTX,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        elapsed = time.monotonic() - start
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        logger.info(
            "[%s] Completed in %.2fs (tokens: %d+%d=%d)",
            label,
            elapsed,
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
        )

        return {
            "label": label,
            "model": model,
            "temperature": temperature,
            "start_time": start,
            "start_wall": start_wall,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "loaded_models": get_loaded_models(),
            "error": None,
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("[%s] Failed after %.2fs: %s", label, elapsed, e)
        return {
            "label": label,
            "model": model,
            "temperature": temperature,
            "start_time": start,
            "start_wall": start_wall,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "loaded_models": get_loaded_models(),
            "error": f"{type(e).__name__}: {str(e)[:280]}",
        }


# =====================================================================
# Test phases
# =====================================================================


def run_baseline(
    creator_model: str,
    judge_model: str,
    samples: int,
    timeout: int,
) -> dict[str, list[dict[str, Any]]]:
    """Phase 1: Single-model baseline (no contention).

    Unload all models first, then test each model alone.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: BASELINE (single model, no contention)")
    logger.info("=" * 60)

    creation_schema = CHARACTER_CREATION_SCHEMA
    judge_char_schema = CharacterQualityScores.model_json_schema()
    rel_schema = RelationshipQualityScores.model_json_schema()

    # --- Creator baseline ---
    logger.info("--- Creator baseline: %s ---", creator_model)
    unload_model(judge_model)
    unload_model(creator_model)
    time.sleep(2)
    warm_model(creator_model, num_ctx=INVESTIGATION_NUM_CTX)

    creator_results = []
    for i in range(samples):
        result = timed_chat(
            model=creator_model,
            prompt=CHARACTER_CREATION_PROMPT,
            json_schema=creation_schema,
            temperature=0.9,
            timeout=timeout,
            label=f"baseline_creator_{i + 1}",
        )
        creator_results.append(result)

    # --- Judge baseline ---
    logger.info("--- Judge baseline: %s ---", judge_model)
    unload_model(creator_model)
    time.sleep(2)
    warm_model(judge_model, num_ctx=INVESTIGATION_NUM_CTX)

    judge_char_results = []
    for i in range(samples):
        result = timed_chat(
            model=judge_model,
            prompt=CHARACTER_JUDGE_PROMPT,
            json_schema=judge_char_schema,
            temperature=0.3,
            timeout=timeout,
            label=f"baseline_judge_char_{i + 1}",
        )
        judge_char_results.append(result)

    judge_rel_results = []
    for i in range(samples):
        result = timed_chat(
            model=judge_model,
            prompt=RELATIONSHIP_JUDGE_PROMPT,
            json_schema=rel_schema,
            temperature=0.3,
            timeout=timeout,
            label=f"baseline_judge_rel_{i + 1}",
        )
        judge_rel_results.append(result)

    return {
        "creator_baseline": creator_results,
        "judge_character_baseline": judge_char_results,
        "judge_relationship_baseline": judge_rel_results,
    }


def run_contention(
    creator_model: str,
    judge_model: str,
    samples: int,
    timeout: int,
) -> list[dict[str, Any]]:
    """Phase 2: Concurrent model contention (reproduces the bug).

    Fires creator + judge calls simultaneously to force both models into VRAM.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: CONTENTION (both models concurrent)")
    logger.info("=" * 60)

    creation_schema = CHARACTER_CREATION_SCHEMA
    judge_schema = CharacterQualityScores.model_json_schema()

    # Ensure both models are loaded
    unload_model(creator_model)
    unload_model(judge_model)
    time.sleep(2)
    warm_model(creator_model, num_ctx=INVESTIGATION_NUM_CTX)
    warm_model(judge_model, num_ctx=INVESTIGATION_NUM_CTX)

    logger.info("Loaded models before contention: %s", get_loaded_models())

    contention_results = []
    for i in range(samples):
        logger.info("--- Contention round %d/%d ---", i + 1, samples)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    timed_chat,
                    creator_model,
                    CHARACTER_CREATION_PROMPT,
                    creation_schema,
                    0.9,
                    timeout,
                    f"contention_creator_{i + 1}",
                ): "creator",
                pool.submit(
                    timed_chat,
                    judge_model,
                    CHARACTER_JUDGE_PROMPT,
                    judge_schema,
                    0.3,
                    timeout,
                    f"contention_judge_{i + 1}",
                ): "judge",
            }
            round_result: dict[str, Any] = {"round": i + 1}
            for future in as_completed(futures):
                role = futures[future]
                round_result[role] = future.result()
            contention_results.append(round_result)

    return contention_results


def run_serialized(
    creator_model: str,
    judge_model: str,
    samples: int,
    timeout: int,
) -> list[dict[str, Any]]:
    """Phase 3: Serialized with explicit unload (proposed fix).

    Creator call, unload creator, judge call, unload judge. No overlap.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: SERIALIZED (unload between calls)")
    logger.info("=" * 60)

    creation_schema = CHARACTER_CREATION_SCHEMA
    judge_schema = CharacterQualityScores.model_json_schema()

    serialized_results = []
    for i in range(samples):
        logger.info("--- Serialized round %d/%d ---", i + 1, samples)

        # Creator call
        unload_model(judge_model)
        time.sleep(1)
        warm_model(creator_model, num_ctx=INVESTIGATION_NUM_CTX)
        creator_result = timed_chat(
            model=creator_model,
            prompt=CHARACTER_CREATION_PROMPT,
            json_schema=creation_schema,
            temperature=0.9,
            timeout=timeout,
            label=f"serialized_creator_{i + 1}",
        )

        # Judge call
        unload_model(creator_model)
        time.sleep(1)
        warm_model(judge_model, num_ctx=INVESTIGATION_NUM_CTX)
        judge_result = timed_chat(
            model=judge_model,
            prompt=CHARACTER_JUDGE_PROMPT,
            json_schema=judge_schema,
            temperature=0.3,
            timeout=timeout,
            label=f"serialized_judge_{i + 1}",
        )

        serialized_results.append(
            {
                "round": i + 1,
                "creator": creator_result,
                "judge": judge_result,
            }
        )

    return serialized_results


# =====================================================================
# Analysis
# =====================================================================


def analyze_results(results: dict[str, Any]) -> dict[str, Any]:
    """Compute summary statistics and draw conclusions."""
    analysis: dict[str, Any] = {}

    def stats(timings: list[float]) -> dict[str, float]:
        """Compute summary statistics for a list of timings."""
        if not timings:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "spread": 0, "median": 0}
        s = sorted(timings)
        n = len(s)
        mean = sum(s) / n
        median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        return {
            "count": n,
            "mean": round(mean, 3),
            "min": round(s[0], 3),
            "max": round(s[-1], 3),
            "spread": round(s[-1] / s[0], 2) if s[0] > 0 else 0,
            "median": round(median, 3),
        }

    # Baseline stats
    baseline = results.get("baseline", {})
    for key in ["creator_baseline", "judge_character_baseline", "judge_relationship_baseline"]:
        timings = [r["elapsed_s"] for r in baseline.get(key, []) if not r.get("error")]
        analysis[f"{key}_stats"] = stats(timings)

    # Contention stats
    contention = results.get("contention", [])
    creator_contention_times = [
        r["creator"]["elapsed_s"] for r in contention if not r.get("creator", {}).get("error")
    ]
    judge_contention_times = [
        r["judge"]["elapsed_s"] for r in contention if not r.get("judge", {}).get("error")
    ]
    analysis["contention_creator_stats"] = stats(creator_contention_times)
    analysis["contention_judge_stats"] = stats(judge_contention_times)

    # Serialized stats
    serialized = results.get("serialized", [])
    creator_serial_times = [
        r["creator"]["elapsed_s"] for r in serialized if not r.get("creator", {}).get("error")
    ]
    judge_serial_times = [
        r["judge"]["elapsed_s"] for r in serialized if not r.get("judge", {}).get("error")
    ]
    analysis["serialized_creator_stats"] = stats(creator_serial_times)
    analysis["serialized_judge_stats"] = stats(judge_serial_times)

    # Compute slowdown ratios
    baseline_creator_mean = analysis.get("creator_baseline_stats", {}).get("mean", 0)
    baseline_judge_mean = analysis.get("judge_character_baseline_stats", {}).get("mean", 0)
    contention_creator_mean = analysis.get("contention_creator_stats", {}).get("mean", 0)
    contention_judge_mean = analysis.get("contention_judge_stats", {}).get("mean", 0)
    serial_creator_mean = analysis.get("serialized_creator_stats", {}).get("mean", 0)
    serial_judge_mean = analysis.get("serialized_judge_stats", {}).get("mean", 0)

    if baseline_creator_mean > 0:
        analysis["contention_creator_slowdown"] = round(
            contention_creator_mean / baseline_creator_mean, 2
        )
        analysis["serialized_creator_slowdown"] = round(
            serial_creator_mean / baseline_creator_mean, 2
        )
    if baseline_judge_mean > 0:
        analysis["contention_judge_slowdown"] = round(
            contention_judge_mean / baseline_judge_mean, 2
        )
        analysis["serialized_judge_slowdown"] = round(serial_judge_mean / baseline_judge_mean, 2)

    # GPU stats during contention
    gpu_samples = results.get("gpu_samples", [])
    if gpu_samples:
        vram_used = [s["vram_used_mb"] for s in gpu_samples]
        gpu_utils = [s["gpu_util_pct"] for s in gpu_samples]
        analysis["gpu_vram_peak_mb"] = max(vram_used)
        analysis["gpu_vram_mean_mb"] = round(sum(vram_used) / len(vram_used), 0)
        analysis["gpu_util_peak_pct"] = max(gpu_utils)
        analysis["gpu_util_mean_pct"] = round(sum(gpu_utils) / len(gpu_utils), 1)

    # Conclusions
    conclusions = []
    contention_slowdown = max(
        analysis.get("contention_creator_slowdown", 1),
        analysis.get("contention_judge_slowdown", 1),
    )
    if contention_slowdown >= 2.0:
        conclusions.append(
            f"CONFIRMED: GPU contention causes {contention_slowdown:.1f}x slowdown. "
            "Two large models competing for VRAM causes severe performance degradation."
        )
    elif contention_slowdown >= 1.3:
        conclusions.append(
            f"MODERATE: GPU contention causes {contention_slowdown:.1f}x slowdown. "
            "Noticeable but not catastrophic."
        )
    else:
        conclusions.append(
            f"MINIMAL: GPU contention causes only {contention_slowdown:.1f}x slowdown. "
            "Models may be fitting in VRAM or Ollama is serializing internally."
        )

    serial_overhead = max(
        analysis.get("serialized_creator_slowdown", 1),
        analysis.get("serialized_judge_slowdown", 1),
    )
    if serial_overhead <= 1.2:
        conclusions.append("Serialized approach adds minimal overhead — viable fix for contention.")
    else:
        conclusions.append(
            f"Serialized approach has {serial_overhead:.1f}x overhead from model loading. "
            "Consider keeping models warm longer or using same-model for both roles."
        )

    # Spread analysis (bimodal detection)
    baseline_spread = analysis.get("judge_character_baseline_stats", {}).get("spread", 1)
    contention_spread = analysis.get("contention_judge_stats", {}).get("spread", 1)
    if contention_spread > 2.0 and baseline_spread < 1.5:
        conclusions.append(
            f"BIMODAL CONFIRMED: Contention judge spread={contention_spread:.1f}x "
            f"vs baseline spread={baseline_spread:.1f}x. "
            "GPU time-sharing creates two distinct latency bands."
        )

    analysis["conclusions"] = conclusions
    return analysis


# =====================================================================
# Main
# =====================================================================


def resolve_models(cli_creator: str | None, cli_judge: str | None) -> tuple[str, str]:
    """Resolve creator and judge models from CLI args or settings.

    Args:
        cli_creator: Explicit creator model from --creator flag.
        cli_judge: Explicit judge model from --judge flag.

    Returns:
        Tuple of (creator_model, judge_model).

    Raises:
        ValueError: If models cannot be resolved from settings and no CLI overrides given.
    """
    # CLI overrides take priority
    if cli_creator and cli_judge:
        logger.info("Using CLI models: creator=%s, judge=%s", cli_creator, cli_judge)
        return cli_creator, cli_judge

    try:
        from src.settings import Settings

        settings = Settings.load()

        # Check per-agent model config
        if settings.use_per_agent_models:
            agent_models = settings.agent_models
            if "writer" not in agent_models or "judge" not in agent_models:
                raise ValueError(
                    "use_per_agent_models is enabled but 'writer' and/or 'judge' keys "
                    f"are missing from agent_models. Found keys: {list(agent_models.keys())}. "
                    "Use --creator and --judge to specify models explicitly."
                )
            creator = agent_models["writer"]
            judge = agent_models["judge"]
            if creator == "auto" or judge == "auto":
                raise ValueError(
                    f"agent_models has 'auto' values (writer={creator!r}, judge={judge!r}). "
                    "Use --creator and --judge to specify models explicitly."
                )
            # Apply CLI override for single model if provided
            creator = cli_creator or creator
            judge = cli_judge or judge
            logger.info("Resolved from settings: creator=%s, judge=%s", creator, judge)
            return creator, judge

        # Check default model
        if settings.default_model and settings.default_model != "auto":
            logger.info("Single model mode: %s (no contention possible)", settings.default_model)
            return settings.default_model, settings.default_model

        raise ValueError(
            "Settings loaded but no model configuration found "
            "(use_per_agent_models=False, no default_model). "
            "Use --creator and --judge to specify models explicitly."
        )

    except Exception as e:
        logger.error("Could not resolve models from settings: %s", e, exc_info=True)
        raise ValueError(
            f"Failed to resolve models: {e}. "
            "Use --creator and --judge to specify models explicitly."
        ) from e


def main() -> None:
    """Run the GPU contention investigation."""
    parser = argparse.ArgumentParser(description="GPU contention investigation for issue #424")
    parser.add_argument("--creator", help="Creator model override")
    parser.add_argument("--judge", help="Judge model override")
    parser.add_argument("--samples", type=int, default=5, help="Calls per phase (default: 5)")
    parser.add_argument("--timeout", type=int, default=180, help="Per-call timeout secs")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve models
    creator_model, judge_model = resolve_models(args.creator, args.judge)

    print(f"\n{'=' * 60}")
    print("GPU CONTENTION INVESTIGATION (Issue #424)")
    print(f"{'=' * 60}")
    print(f"Creator model: {creator_model}")
    print(f"Judge model:   {judge_model}")
    print(f"Samples/phase: {args.samples}")
    print(f"Timeout:       {args.timeout}s")
    print(f"Same model:    {creator_model == judge_model}")

    if creator_model == judge_model:
        print("\nWARNING: Creator and judge are the same model — no contention possible.")
        print("Use --creator and --judge to specify different models.")

    # Collect model info
    creator_info = get_model_info(creator_model)
    judge_info = get_model_info(judge_model)
    print(f"\nCreator info: {json.dumps(creator_info, indent=2)}")
    print(f"Judge info:   {json.dumps(judge_info, indent=2)}")

    # Start GPU monitor
    gpu_monitor = GpuMonitor(interval=0.5)
    gpu_monitor.start()

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "creator_model": creator_model,
            "judge_model": judge_model,
            "creator_info": creator_info,
            "judge_info": judge_info,
            "samples_per_phase": args.samples,
            "timeout_s": args.timeout,
            "num_ctx": INVESTIGATION_NUM_CTX,
        },
    }

    try:
        # Phase 1: Baseline
        print(f"\n{'=' * 60}")
        print("Phase 1: BASELINE (single model, no contention)")
        print(f"{'=' * 60}\n")
        results["baseline"] = run_baseline(creator_model, judge_model, args.samples, args.timeout)

        # Phase 2: Contention
        print(f"\n{'=' * 60}")
        print("Phase 2: CONTENTION (both models concurrent)")
        print(f"{'=' * 60}\n")
        results["contention"] = run_contention(
            creator_model, judge_model, args.samples, args.timeout
        )

        # Phase 3: Serialized
        print(f"\n{'=' * 60}")
        print("Phase 3: SERIALIZED (unload between calls)")
        print(f"{'=' * 60}\n")
        results["serialized"] = run_serialized(
            creator_model, judge_model, args.samples, args.timeout
        )
    finally:
        # Guarantee monitor shutdown and model unload even if a phase raises
        results["gpu_samples"] = gpu_monitor.stop()
        unload_model(creator_model)
        unload_model(judge_model)

    # Analyze
    results["analysis"] = analyze_results(results)

    # Print summary
    analysis = results["analysis"]
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    print("\nBaseline (single model):")
    for key in [
        "creator_baseline_stats",
        "judge_character_baseline_stats",
        "judge_relationship_baseline_stats",
    ]:
        s = analysis.get(key, {})
        print(
            f"  {key}: mean={s.get('mean', 0):.2f}s, "
            f"min={s.get('min', 0):.2f}s, max={s.get('max', 0):.2f}s, "
            f"spread={s.get('spread', 0):.1f}x"
        )

    print("\nContention (both models):")
    for key in ["contention_creator_stats", "contention_judge_stats"]:
        s = analysis.get(key, {})
        print(
            f"  {key}: mean={s.get('mean', 0):.2f}s, "
            f"min={s.get('min', 0):.2f}s, max={s.get('max', 0):.2f}s, "
            f"spread={s.get('spread', 0):.1f}x"
        )

    print("\nSerialized (unload between):")
    for key in ["serialized_creator_stats", "serialized_judge_stats"]:
        s = analysis.get(key, {})
        print(
            f"  {key}: mean={s.get('mean', 0):.2f}s, "
            f"min={s.get('min', 0):.2f}s, max={s.get('max', 0):.2f}s, "
            f"spread={s.get('spread', 0):.1f}x"
        )

    print("\nSlowdown ratios (vs baseline):")
    print(f"  Contention creator: {analysis.get('contention_creator_slowdown', 'N/A')}x")
    print(f"  Contention judge:   {analysis.get('contention_judge_slowdown', 'N/A')}x")
    print(f"  Serialized creator: {analysis.get('serialized_creator_slowdown', 'N/A')}x")
    print(f"  Serialized judge:   {analysis.get('serialized_judge_slowdown', 'N/A')}x")

    if analysis.get("gpu_vram_peak_mb"):
        print(
            f"\nGPU VRAM peak: {analysis['gpu_vram_peak_mb']}MB, "
            f"mean: {analysis['gpu_vram_mean_mb']}MB"
        )
        print(
            f"GPU util peak: {analysis['gpu_util_peak_pct']}%, "
            f"mean: {analysis['gpu_util_mean_pct']}%"
        )

    print("\nConclusions:")
    for c in analysis.get("conclusions", []):
        print(f"  - {c}")

    # Save output
    output_dir = Path("output/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else output_dir / f"gpu_contention_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to: {output_path}")
    print(f"GPU samples collected: {len(results['gpu_samples'])}")


if __name__ == "__main__":
    main()
