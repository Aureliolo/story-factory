"""Shared Ollama API helpers for investigation scripts.

Consolidates common functionality used across all investigation scripts:
model discovery, model unloading, model info retrieval, and shared constants.

80% GPU residency rule: never run a model unless at least 80% of it fits in
GPU VRAM. Models split heavily between GPU and system RAM run drastically
slower (5-10x). For a 24 GB GPU the practical max model size is ~30 GB.
"""

import json
import logging
import subprocess
from typing import Any

import httpx

from src.services.model_mode_service._vram import MIN_GPU_RESIDENCY

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
EMBEDDING_MODELS = {"bge-m3", "snowflake-arctic-embed", "mxbai-embed-large"}

# Default context window for investigation scripts.
# Investigation prompts are small (~1K tokens). Using the model's default
# (e.g. 128K for mistral-nemo) wastes massive VRAM on KV cache.
INVESTIGATION_NUM_CTX = 4096

# Canonical story brief used for generating test prompts across investigation scripts
CANONICAL_BRIEF = (
    "In a crumbling empire where magic is fueled by memory, a disgraced archivist "
    "discovers that the ruling council has been erasing collective memories to maintain "
    "power. She must navigate rival factions, ancient artifacts, and forbidden knowledge "
    "to restore what was lost before the empire collapses into civil war. "
    "Genre: Fantasy with Political Intrigue. Tone: Dark and suspenseful."
)


def get_total_gpu_vram_mb() -> int:
    """Detect total GPU VRAM in MB via nvidia-smi.

    Returns:
        Total VRAM in MB, or 0 if detection fails (no GPU / nvidia-smi not found).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        vram_mb = int(result.stdout.strip().split("\n")[0])
        logger.debug("Detected GPU VRAM: %dMB", vram_mb)
        return vram_mb
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError, OSError) as e:
        logger.info("Could not detect GPU VRAM: %s", e)
        return 0


def get_installed_models(enforce_gpu_residency: bool = True) -> list[str]:
    """Retrieve installed Ollama model tags, excluding embedding models and oversized models.

    Args:
        enforce_gpu_residency: If True, exclude models that cannot achieve at least
            MIN_GPU_RESIDENCY (80%) GPU residency. Requires nvidia-smi.

    Returns:
        Sorted list of model name:tag strings. Empty list on failure.
    """
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=30)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.error("Failed to list Ollama models: %s", e)
        return []

    gpu_vram_gb = get_total_gpu_vram_mb() / 1024 if enforce_gpu_residency else 0

    result = []
    for m in models:
        name = m.get("name", "")
        base_name = name.split(":")[0].split("/")[-1]
        if base_name in EMBEDDING_MODELS:
            continue

        # 80% GPU residency rule: skip models too large for the GPU.
        if enforce_gpu_residency and gpu_vram_gb > 0:
            size_bytes = m.get("size", 0)
            size_gb = size_bytes / (1024**3)
            if size_gb > 0:
                gpu_residency = gpu_vram_gb / size_gb
                if gpu_residency < MIN_GPU_RESIDENCY:
                    logger.info(
                        "Excluding %s: %.0f%% GPU residency (%.1fGB model, %.0fGB GPU). "
                        "Minimum %.0f%% required.",
                        name,
                        gpu_residency * 100,
                        size_gb,
                        gpu_vram_gb,
                        MIN_GPU_RESIDENCY * 100,
                    )
                    continue

        result.append(name)

    logger.info(
        "Found %d installed models (enforce_gpu_residency=%s)", len(result), enforce_gpu_residency
    )
    return sorted(result)


def warm_model(model: str, num_ctx: int = INVESTIGATION_NUM_CTX, timeout: int = 120) -> bool:
    """Pre-load a model via the native Ollama API with a specific context window.

    Ollama's OpenAI-compatible endpoint (/v1/) ignores the ``options`` field,
    so models load with their Modelfile default context (e.g. 128K for
    mistral-nemo).  Calling the native ``/api/chat`` endpoint first forces
    Ollama to load the model with the desired ``num_ctx``, and subsequent
    OpenAI-compatible calls reuse the already-loaded model at that context size.

    Args:
        model: Model name to pre-load.
        num_ctx: Context window size (default: INVESTIGATION_NUM_CTX = 4096).
        timeout: Request timeout in seconds.

    Returns:
        True if warm-up succeeded, False on failure.
    """
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": 1},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        logger.info(
            "Warmed model '%s' with num_ctx=%d (status=%s)", model, num_ctx, resp.status_code
        )
        return True
    except httpx.HTTPError as e:
        logger.warning("Failed to warm model '%s': %s", model, e)
        return False


def unload_model(model: str) -> bool:
    """Request Ollama to unload a model from VRAM via keep_alive=0.

    Calls raise_for_status() to detect 4xx/5xx failures that would leave
    the model resident in VRAM.

    Args:
        model: Name or tag of the model to unload.

    Returns:
        True if unload succeeded, False on any failure.
    """
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=60,
        )
        resp.raise_for_status()
        logger.debug("Unloaded model '%s' (status=%s)", model, resp.status_code)
        return True
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        body = (e.response.text or "")[:200]
        logger.warning("Failed to unload model '%s' (status %s): %s", model, status_code, body)
        return False
    except httpx.HTTPError as e:
        logger.warning("Failed to unload model '%s': %s", model, e)
        return False


def get_model_info(model: str) -> dict[str, Any]:
    """Get model details from Ollama (parameter size, quantization, family).

    Args:
        model: Model name to query.

    Returns:
        Dict with parameter_size, quantization, family, total_parameters.
        Empty dict on failure.
    """
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/show",
            json={"name": model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})
        model_info_data = data.get("model_info", {})
        return {
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization": details.get("quantization_level", "unknown"),
            "family": details.get("family", "unknown"),
            "total_parameters": model_info_data.get("general.parameter_count", 0),
        }
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("Failed to get info for model '%s': %s", model, e)
        return {}
