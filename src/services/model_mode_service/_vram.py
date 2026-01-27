"""VRAM management functions for ModelModeService."""

import logging
from typing import TYPE_CHECKING

from src.memory.mode_models import VramStrategy
from src.utils.validation import validate_not_empty

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService

logger = logging.getLogger("src.services.model_mode_service._vram")


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

    logger.debug(f"Preparing model {model_id} with VRAM strategy: {strategy.value}")

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


def unload_all_except(svc: ModelModeService, keep_model: str) -> None:
    """Unload all models except the specified one.

    Note: Ollama manages model lifecycle automatically via LRU caching.
    This method only updates our tracking. Actual VRAM freeing depends
    on Ollama's internal memory management, not explicit unload calls.

    For truly sequential model usage, consider using Ollama's --noprune
    flag with manual model loading/unloading via the API if available.

    Args:
        svc: The ModelModeService instance.
        keep_model: The model ID to keep loaded.
    """
    # For now, just clear our tracking
    # Ollama will unload based on its own LRU cache
    models_to_remove = svc._loaded_models - {keep_model}
    if models_to_remove:
        logger.debug(
            f"Marking models for potential unload by Ollama: {models_to_remove} "
            f"(actual unloading depends on Ollama's memory management)"
        )
        svc._loaded_models = {keep_model}
