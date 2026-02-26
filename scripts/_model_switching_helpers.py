"""Helpers for the model switching investigation script.

Contains the core model call function, statistical helpers, bimodal detection,
GPU residency checking, and type definitions used across scenario functions.
"""

import json
import logging
import statistics
import time
from typing import Any, TypedDict

import httpx

from scripts._ollama_helpers import INVESTIGATION_NUM_CTX, OLLAMA_BASE
from src.services.model_mode_service._vram import MIN_GPU_RESIDENCY

logger = logging.getLogger(__name__)


class CallResult(TypedDict):
    """Result from a single model call."""

    duration_s: float
    response: str
    eval_tokens: int
    prompt_tokens: int
    error: str | None


class StatsResult(TypedDict):
    """Descriptive statistics for a list of durations."""

    median: float
    mean: float
    stdev: float
    min: float
    max: float


def call_model(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    num_ctx: int = INVESTIGATION_NUM_CTX,
    timeout: int = 180,
) -> CallResult:
    """Send a prompt to a model and return timing + response data.

    Args:
        model: Ollama model name.
        prompt: The user prompt.
        temperature: Generation temperature.
        num_ctx: Context window size.
        timeout: Request timeout in seconds.

    Returns:
        CallResult with duration_s, response, eval_tokens, prompt_tokens, error.
    """
    logger.debug("Calling model %s (temp=%.1f, num_ctx=%d)", model, temperature, num_ctx)
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

        message = body.get("message")
        if message is None:
            logger.warning("Ollama response missing 'message' key: %s", list(body.keys()))
        content = (message or {}).get("content", "")

        eval_count = body.get("eval_count", 0)
        prompt_eval_count = body.get("prompt_eval_count", 0)
        logger.debug(
            "Model %s responded in %.3fs (eval=%d, prompt=%d tokens)",
            model,
            elapsed,
            eval_count,
            prompt_eval_count,
        )
        return CallResult(
            duration_s=elapsed,
            response=content[:500],
            eval_tokens=eval_count,
            prompt_tokens=prompt_eval_count,
            error=None,
        )
    except (httpx.TimeoutException, httpx.HTTPError, json.JSONDecodeError) as e:
        elapsed = round(time.monotonic() - start, 3)
        error_msg = "timeout" if isinstance(e, httpx.TimeoutException) else str(e)[:200]
        logger.warning("Model %s call failed after %.2fs: %s", model, elapsed, error_msg)
        return CallResult(
            duration_s=elapsed,
            response="",
            eval_tokens=0,
            prompt_tokens=0,
            error=error_msg,
        )


def compute_stats(durations: list[float]) -> StatsResult:
    """Compute descriptive statistics for a list of durations.

    Args:
        durations: List of timing values in seconds.

    Returns:
        StatsResult with median, mean, stdev, min, max.
    """
    if not durations:
        return StatsResult(median=0.0, mean=0.0, stdev=0.0, min=0.0, max=0.0)
    return StatsResult(
        median=round(statistics.median(durations), 3),
        mean=round(statistics.mean(durations), 3),
        stdev=round(statistics.stdev(durations), 3) if len(durations) > 1 else 0.0,
        min=round(min(durations), 3),
        max=round(max(durations), 3),
    )


def detect_bimodal(durations: list[float]) -> dict[str, Any]:
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

    is_bimodal = max_gap > median_val * 2

    fast_cluster = sorted_d[: gap_idx + 1]
    slow_cluster = sorted_d[gap_idx + 1 :]

    return {
        "is_bimodal": is_bimodal,
        "max_gap_s": round(max_gap, 3),
        "median_s": round(median_val, 3),
        "gap_to_median_ratio": round(max_gap / median_val, 2) if median_val else 0.0,
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


def check_gpu_residency(model: str, gpu_total_mb: int) -> bool:
    """Check if a model meets the 80% GPU residency requirement.

    Queries /api/tags to find the model's file size and compares against
    available GPU VRAM.

    Args:
        model: Model name to check.
        gpu_total_mb: Total GPU VRAM in MB.

    Returns:
        True if the model passes the residency check, False if too large.
    """
    if gpu_total_mb <= 0:
        logger.debug("No GPU VRAM info — skipping residency check for %s", model)
        return True

    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=30)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("Could not check GPU residency for %s: %s", model, e)
        return True

    for m in models:
        if m.get("name") == model:
            size_bytes = m.get("size", 0)
            size_gib = size_bytes / (1024**3)
            gpu_vram_gib = gpu_total_mb / 1024
            if size_gib > 0:
                residency = gpu_vram_gib / size_gib
                if residency < MIN_GPU_RESIDENCY:
                    logger.error(
                        "Model %s fails GPU residency: %.0f%% (%.1f GiB model, "
                        "%.0f GiB GPU, minimum %.0f%% required)",
                        model,
                        residency * 100,
                        size_gib,
                        gpu_vram_gib,
                        MIN_GPU_RESIDENCY * 100,
                    )
                    return False
            return True

    logger.warning("Model %s not found in installed models — cannot verify GPU residency", model)
    return True
