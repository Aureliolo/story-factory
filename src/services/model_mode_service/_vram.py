"""VRAM management mixin for ModelModeService."""

import logging

from src.memory.mode_models import VramStrategy
from src.utils.validation import validate_not_empty

from ._base import ModelModeServiceBase

logger = logging.getLogger(__name__)


class VramMixin(ModelModeServiceBase):
    """Mixin providing VRAM management functionality."""

    def prepare_model(self, model_id: str) -> None:
        """
        Prepare a model for use according to the configured VRAM strategy.

        This method reads the VRAM strategy from ``settings.vram_strategy`` (user-configurable)
        and may unload other loaded models depending on that strategy:

        - ``SEQUENTIAL``: unloads all other models.
        - ``ADAPTIVE``: unloads other models only if available VRAM is less than the model's
          estimated requirement.

        The given model is then marked as loaded in the service's internal tracker.

        Parameters:
            model_id (str): Identifier of the model to prepare.
        """
        from src.settings import get_installed_models_with_sizes, get_model_info

        validate_not_empty(model_id, "model_id")

        # Use settings.vram_strategy as the source of truth (user-configurable override)
        strategy_str = self.settings.vram_strategy
        try:
            strategy = VramStrategy(strategy_str)
        except ValueError as e:
            valid_options = ", ".join(s.value for s in VramStrategy)
            error_msg = (
                f"Invalid vram_strategy '{strategy_str}' in settings. "
                f"Valid options: {valid_options}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        logger.debug(f"Preparing model {model_id} with VRAM strategy: {strategy.value}")

        if strategy == VramStrategy.SEQUENTIAL:
            # Unload all other models
            self._unload_all_except(model_id)
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
                self._unload_all_except(model_id)

        # Model will be loaded on first use by Ollama
        self._loaded_models.add(model_id)

    def _unload_all_except(self, keep_model: str) -> None:
        """Unload all models except the specified one.

        Note: Ollama manages model lifecycle automatically via LRU caching.
        This method only updates our tracking. Actual VRAM freeing depends
        on Ollama's internal memory management, not explicit unload calls.

        For truly sequential model usage, consider using Ollama's --noprune
        flag with manual model loading/unloading via the API if available.
        """
        # For now, just clear our tracking
        # Ollama will unload based on its own LRU cache
        models_to_remove = self._loaded_models - {keep_model}
        if models_to_remove:
            logger.debug(
                f"Marking models for potential unload by Ollama: {models_to_remove} "
                f"(actual unloading depends on Ollama's memory management)"
            )
            self._loaded_models = {keep_model}
