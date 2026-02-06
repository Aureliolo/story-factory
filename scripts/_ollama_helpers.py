"""Shared Ollama API helpers for investigation scripts.

Consolidates common functionality used across all investigation scripts:
model discovery, model unloading, model info retrieval, and shared constants.
"""

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
EMBEDDING_MODELS = {"bge-m3", "snowflake-arctic-embed", "mxbai-embed-large"}

# Canonical story brief used for generating test prompts across investigation scripts
CANONICAL_BRIEF = (
    "In a crumbling empire where magic is fueled by memory, a disgraced archivist "
    "discovers that the ruling council has been erasing collective memories to maintain "
    "power. She must navigate rival factions, ancient artifacts, and forbidden knowledge "
    "to restore what was lost before the empire collapses into civil war. "
    "Genre: Fantasy with Political Intrigue. Tone: Dark and suspenseful."
)


def get_installed_models() -> list[str]:
    """Retrieve installed Ollama model tags, excluding known embedding models.

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

    result = []
    for m in models:
        name = m.get("name", "")
        base_name = name.split(":")[0].split("/")[-1]
        if base_name in EMBEDDING_MODELS:
            continue
        result.append(name)

    logger.info("Found %d installed models", len(result))
    return sorted(result)


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
