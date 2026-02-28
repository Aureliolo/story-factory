"""VRAM management functions for ModelModeService.

80% GPU residency rule: never run a model unless at least 80% of it fits in
GPU VRAM. Models split heavily between GPU and system RAM run drastically
slower (5-10x). For a 24 GB GPU the practical max model size is ~30 GB.
"""

import logging
import threading
from typing import TYPE_CHECKING

import ollama

from src.memory.mode_models import VramStrategy
from src.utils.validation import validate_not_empty

# Track last prepared model to suppress repeated log messages
_last_prepared_model_lock = threading.Lock()
_last_prepared_model_id: str | None = None

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

logger = logging.getLogger(__name__)

# Minimum fraction of a model that must fit in GPU VRAM.
# Models below this threshold are excluded from auto-selection because
# heavy GPU/CPU splitting causes 5-10x inference slowdown.
MIN_GPU_RESIDENCY = 0.8


def prepare_model(svc: ModelModeService, model_id: str) -> None:
    """Prepare a model for use according to the configured VRAM strategy.

    Reads the VRAM strategy from ``settings.vram_strategy`` (user-configurable)
    and may unload other loaded models depending on that strategy:

    - ``SEQUENTIAL``: unloads all other models.
    - ``ADAPTIVE``: unloads other models only if available VRAM is less than the model's
      estimated requirement.

    The given model is then marked as loaded in the service's internal tracker.

    Args:
        svc: The ModelModeService instance.
        model_id: Identifier of the model to prepare.
    """
    from src.settings import get_installed_models_with_sizes, get_model_info

    validate_not_empty(model_id, "model_id")

    # Use settings.vram_strategy as the source of truth (user-configurable override)
    strategy_str = svc.settings.vram_strategy
    try:
        strategy = VramStrategy(strategy_str)
    except ValueError as e:
        valid_options = ", ".join(s.value for s in VramStrategy)
        error_msg = (
            f"Invalid vram_strategy '{strategy_str}' in settings. Valid options: {valid_options}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    global _last_prepared_model_id
    with _last_prepared_model_lock:
        if model_id == _last_prepared_model_id:
            logger.debug("Model %s already prepared, skipping VRAM preparation", model_id)
            return
        logger.debug("Preparing model %s with VRAM strategy: %s", model_id, strategy.value)
        _last_prepared_model_id = model_id

    if strategy == VramStrategy.SEQUENTIAL:
        # Unload all other models
        unload_all_except(svc, model_id)
    elif strategy == VramStrategy.ADAPTIVE:
        # Check if we need to free VRAM
        from src.settings import get_available_vram

        available = get_available_vram()

        # Get model size from installed models or RECOMMENDED_MODELS
        installed = get_installed_models_with_sizes()
        if model_id in installed:
            size_gb = installed[model_id]
            required = int(size_gb * 1.2)  # 20% overhead
        else:
            model_info = get_model_info(model_id)
            required = model_info["vram_required"]

        if available < required:
            # Need to free up space
            unload_all_except(svc, model_id)

    # Model will be loaded on first use by Ollama
    svc._loaded_models.add(model_id)

    # M6: warn if model residency is below minimum threshold
    try:
        installed = get_installed_models_with_sizes()
        if model_id in installed:
            model_size_gb = installed[model_id]
            from src.settings import get_available_vram

            available_vram = get_available_vram()
            if model_size_gb > 0:
                residency = min(available_vram / model_size_gb, 1.0)
                if residency < MIN_GPU_RESIDENCY:
                    logger.warning(
                        "Model %s GPU residency %.0f%% is below minimum %.0f%% "
                        "(model=%.1fGB, available_vram=%.1fGB)",
                        model_id,
                        residency * 100,
                        MIN_GPU_RESIDENCY * 100,
                        model_size_gb,
                        available_vram,
                    )
    except Exception as e:
        logger.debug("Could not check GPU residency for %s: %s", model_id, e)


def unload_all_except(svc: ModelModeService, keep_model: str) -> None:
    """Unload all tracked models except the specified one via Ollama API.

    Calls ``ollama.Client.generate(model=..., keep_alive=0)`` for each model
    to be removed, which tells Ollama to immediately evict the model from VRAM.
    Errors are logged but do not interrupt the process (best-effort unload).

    Args:
        svc: The ModelModeService instance.
        keep_model: The model ID to keep loaded.
    """
    models_to_remove = svc._loaded_models - {keep_model}
    if not models_to_remove:
        logger.debug("No models to unload (keeping %s)", keep_model)
        return

    logger.info("Unloading %d model(s) from VRAM: %s", len(models_to_remove), models_to_remove)

    successfully_unloaded: set[str] = set()
    for model_id in models_to_remove:
        try:
            svc._ollama_client.generate(model=model_id, keep_alive=0)
            logger.debug("Unloaded model %s from VRAM (keep_alive=0)", model_id)
            successfully_unloaded.add(model_id)
        except ollama.ResponseError as e:
            logger.warning("Failed to unload model %s: %s", model_id, e)
        except ConnectionError as e:
            logger.warning("Connection error unloading model %s: %s", model_id, e)
        except TimeoutError as e:
            logger.warning("Timeout unloading model %s: %s", model_id, e)

    # Only remove successfully unloaded models from tracking
    svc._loaded_models = (svc._loaded_models - successfully_unloaded) | {keep_model}

    # Clear last-prepared cache when models are evicted so the next
    # prepare_model() call for the kept model actually runs setup.
    global _last_prepared_model_id
    with _last_prepared_model_lock:
        if _last_prepared_model_id in successfully_unloaded:
            logger.debug("Cleared last-prepared model cache (evicted %s)", _last_prepared_model_id)
            _last_prepared_model_id = None
