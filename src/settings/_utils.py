"""Utility functions for Ollama model detection and VRAM queries."""

import logging
import subprocess

from src.settings._model_registry import RECOMMENDED_MODELS
from src.settings._types import ModelInfo

logger = logging.getLogger("src.settings._utils")

# Cross-platform subprocess flags (CREATE_NO_WINDOW only exists on Windows)
_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def get_installed_models(timeout: int | None = None) -> list[str]:
    """Get list of models currently installed in Ollama.

    Args:
        timeout: Timeout in seconds. If None, uses default (10s).
    """
    actual_timeout = timeout if timeout is not None else 10
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except FileNotFoundError:
        logger.warning("Ollama not found. Please ensure Ollama is installed and in PATH.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"Ollama list command timed out after {actual_timeout}s.")
        return []
    except (OSError, ValueError) as e:
        logger.warning(f"Error listing Ollama models: {e}")
        return []


def get_installed_models_with_sizes(timeout: int | None = None) -> dict[str, float]:
    """Get installed models with their sizes in GB.

    Args:
        timeout: Timeout in seconds. If None, uses default (10s).

    Returns:
        Dict mapping model ID to size in GB.
    """
    actual_timeout = timeout if timeout is not None else 10
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        # Check return code before parsing
        if result.returncode != 0:
            logger.warning(f"Ollama list returned non-zero exit code: {result.returncode}")
            return {}

        models = {}
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    model_name = parts[0]
                    # Size is typically in format like "4.1 GB" or "890 MB"
                    # Find size column - it's the one with GB/MB
                    # Use decimal units: 1 GB = 1000 MB (not 1024)
                    size_gb = 0.0
                    for i, part in enumerate(parts):
                        if part.upper() == "GB" and i > 0:
                            try:
                                size_gb = float(parts[i - 1])
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse GB size '{parts[i - 1]}' "
                                    f"for model '{model_name}'"
                                )
                            break
                        elif part.upper() == "MB" and i > 0:
                            try:
                                size_gb = float(parts[i - 1]) / 1000
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse MB size '{parts[i - 1]}' "
                                    f"for model '{model_name}'"
                                )
                            break
                        # Also handle combined format like "4.1GB"
                        elif part.upper().endswith("GB"):
                            try:
                                size_gb = float(part[:-2])
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse combined GB size '{part}' "
                                    f"for model '{model_name}'"
                                )
                            break
                        elif part.upper().endswith("MB"):
                            try:
                                size_gb = float(part[:-2]) / 1000
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse combined MB size '{part}' "
                                    f"for model '{model_name}'"
                                )
                            break

                    models[model_name] = size_gb
        return models
    except FileNotFoundError:
        logger.warning("Ollama not found. Please ensure Ollama is installed and in PATH.")
        return {}
    except subprocess.TimeoutExpired:
        logger.warning(f"Ollama list command timed out after {actual_timeout}s.")
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Error listing Ollama models: {e}")
        return {}


def get_available_vram(timeout: int | None = None) -> int:
    """Detect available VRAM in GB. Returns 8GB default if detection fails.

    Args:
        timeout: Timeout in seconds. If None, uses default (10s).
    """
    actual_timeout = timeout if timeout is not None else 10
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        vram_mb = int(result.stdout.strip().split("\n")[0])
        return vram_mb // 1024
    except FileNotFoundError:
        logger.info("nvidia-smi not found. Using default VRAM assumption of 8GB.")
        return 8
    except subprocess.TimeoutExpired:
        logger.warning(f"nvidia-smi timed out after {actual_timeout}s. Using default 8GB.")
        return 8
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse VRAM from nvidia-smi output: {e}. Using default 8GB.")
        return 8
    except OSError as e:
        logger.info(f"Could not detect VRAM: {e}. Using default 8GB.")
        return 8


def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a model.

    If the model is in RECOMMENDED_MODELS, returns that info.
    Otherwise, estimates info based on model name/size.
    """
    # Check exact match first
    if model_id in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[model_id]

    # Check base name match (e.g., "qwen3:4b" matches "qwen3:4b")
    base_name = model_id.split(":")[0] if ":" in model_id else model_id
    for rec_id, info in RECOMMENDED_MODELS.items():
        if rec_id.startswith(base_name) or base_name in rec_id:
            return info

    # Try to get size from installed models
    installed = get_installed_models_with_sizes()
    size_gb = installed.get(model_id, 0)

    # Estimate quality/speed from size
    if size_gb > 0:
        # Larger models = higher quality, lower speed
        quality = min(10, int(size_gb / 4) + 4)  # 4GB->5, 20GB->9
        speed = max(1, 10 - int(size_gb / 5))  # Smaller = faster
        vram_required = int(size_gb * 1.2)
    else:
        quality = 5
        speed = 5
        vram_required = 8  # Default assumption
        size_gb = 5.0

    return ModelInfo(
        name=model_id,
        size_gb=size_gb,
        vram_required=vram_required,
        quality=quality,
        speed=speed,
        uncensored=True,
        description="Automatically detected model",
        tags=[],  # No specific tags for unknown models
    )
