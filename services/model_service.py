"""Model service - handles Ollama model operations."""

import logging
from collections.abc import Generator
from dataclasses import dataclass

import ollama

from settings import AVAILABLE_MODELS, ModelInfo, Settings, get_available_vram, get_model_info

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Status information about a model."""

    model_id: str
    name: str
    installed: bool
    size_gb: float
    vram_required: int
    quality: int | float
    speed: int
    uncensored: bool
    description: str


@dataclass
class OllamaHealth:
    """Ollama service health status."""

    is_healthy: bool
    message: str
    version: str | None = None
    available_vram: int | None = None


class ModelService:
    """Ollama model operations service.

    This service handles checking Ollama health, listing models,
    pulling new models, and getting VRAM information.
    """

    def __init__(self, settings: Settings):
        """Initialize model service.

        Args:
            settings: Application settings.
        """
        self.settings = settings

    def check_health(self) -> OllamaHealth:
        """Check Ollama service health and connectivity.

        Returns:
            OllamaHealth with status information.
        """
        try:
            # Try to list models as a health check
            client = ollama.Client(host=self.settings.ollama_url, timeout=30.0)
            client.list()  # Just check connectivity

            # Get VRAM
            vram = get_available_vram()

            return OllamaHealth(
                is_healthy=True,
                message="Ollama is running",
                version=None,  # Ollama doesn't expose version easily
                available_vram=vram,
            )
        except ollama.ResponseError as e:
            logger.warning(f"Ollama API error during health check: {e}")
            return OllamaHealth(
                is_healthy=False,
                message=f"Ollama API error: {e}",
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Cannot connect to Ollama at {self.settings.ollama_url}: {e}")
            return OllamaHealth(
                is_healthy=False,
                message=f"Cannot connect to Ollama: {e}",
            )

    def list_installed(self) -> list[str]:
        """List installed Ollama models.

        Returns:
            List of installed model IDs.
        """
        try:
            client = ollama.Client(host=self.settings.ollama_url, timeout=30.0)
            response = client.list()
            return [model.model for model in response.models if model.model]
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to list models from Ollama: {e}")
            return []

    def list_available(self) -> list[ModelStatus]:
        """List all available models with their status.

        Returns:
            List of ModelStatus objects for all known models.
        """
        installed = set(self.list_installed())
        models = []

        for model_id, info in AVAILABLE_MODELS.items():
            # Check if any variant of this model is installed
            is_installed = model_id in installed or any(model_id in m for m in installed)

            models.append(
                ModelStatus(
                    model_id=model_id,
                    name=info["name"],
                    installed=is_installed,
                    size_gb=info["size_gb"],
                    vram_required=info["vram_required"],
                    quality=info["quality"],
                    speed=info["speed"],
                    uncensored=info["uncensored"],
                    description=info["description"],
                )
            )

        return models

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model.

        Args:
            model_id: The model identifier.

        Returns:
            ModelInfo dictionary.
        """
        return get_model_info(model_id)

    def pull_model(self, model_id: str) -> Generator[dict]:
        """Pull a model from Ollama registry with progress updates.

        Args:
            model_id: The model ID to pull.

        Yields:
            Progress dictionaries with status, completed, total.
        """
        try:
            client = ollama.Client(host=self.settings.ollama_url, timeout=600.0)  # 10 min for pulls

            for progress in client.pull(model_id, stream=True):
                yield {
                    "status": progress.get("status", ""),
                    "completed": progress.get("completed", 0),
                    "total": progress.get("total", 0),
                }

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error pulling model {model_id}: {e}")
            yield {"status": f"Error: {e}", "error": True}
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error pulling model {model_id}: {e}")
            yield {"status": f"Failed to pull model: {e}", "error": True}

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from Ollama.

        Args:
            model_id: The model ID to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            client = ollama.Client(host=self.settings.ollama_url, timeout=30.0)
            client.delete(model_id)
            logger.info(f"Deleted model: {model_id}")
            return True
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def get_vram(self) -> int:
        """Get available VRAM in GB.

        Returns:
            VRAM in gigabytes.
        """
        return get_available_vram()

    def get_recommended_model(self, role: str | None = None) -> str:
        """Get recommended model based on available VRAM and role.

        Args:
            role: Optional agent role for role-specific recommendation.

        Returns:
            Recommended model ID.
        """
        vram = self.get_vram()

        if role:
            return self.settings.get_model_for_agent(role, vram)

        # General recommendation based on VRAM
        if vram >= 24:
            return "huihui_ai/qwen3-abliterated:32b"
        elif vram >= 12:
            return "huihui_ai/qwen3-abliterated:14b"
        else:
            return "huihui_ai/qwen3-abliterated:8b"

    def get_models_for_vram(self, min_vram: int | None = None) -> list[ModelStatus]:
        """Get models that fit within available VRAM.

        Args:
            min_vram: Optional minimum VRAM override. Defaults to detected.

        Returns:
            List of ModelStatus for models that fit.
        """
        vram = min_vram if min_vram is not None else self.get_vram()
        all_models = self.list_available()

        return [m for m in all_models if m.vram_required <= vram]

    def test_model(self, model_id: str) -> tuple[bool, str]:
        """Test if a model works by running a simple prompt.

        Args:
            model_id: The model ID to test.

        Returns:
            Tuple of (success, message).
        """
        try:
            client = ollama.Client(host=self.settings.ollama_url, timeout=60.0)
            response = client.generate(
                model=model_id,
                prompt="Say 'hello' in one word.",
                options={"num_predict": 10},
            )

            if response.response:
                return True, f"Model working: {response.response[:50]}"
            return False, "Model returned empty response"

        except ollama.ResponseError as e:
            logger.warning(f"Model test failed for {model_id}: {e}")
            return False, f"Model error: {e}"
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error testing model {model_id}: {e}")
            return False, f"Test failed: {e}"

    def get_model_by_quality(
        self,
        min_quality: int,
        max_vram: int | None = None,
        uncensored_required: bool = False,
    ) -> list[ModelStatus]:
        """Get models matching quality and VRAM requirements.

        Args:
            min_quality: Minimum quality score (1-10).
            max_vram: Maximum VRAM requirement.
            uncensored_required: Whether NSFW support is required.

        Returns:
            List of matching ModelStatus objects.
        """
        vram = max_vram if max_vram is not None else self.get_vram()
        models = self.list_available()

        filtered = []
        for model in models:
            if model.quality >= min_quality and model.vram_required <= vram:
                if uncensored_required and not model.uncensored:
                    continue
                filtered.append(model)

        # Sort by quality descending
        filtered.sort(key=lambda m: m.quality, reverse=True)
        return filtered

    def compare_models(
        self,
        model_ids: list[str],
        prompt: str,
    ) -> list[dict]:
        """Compare multiple models on the same prompt.

        Args:
            model_ids: List of model IDs to compare.
            prompt: The prompt to test.

        Returns:
            List of result dictionaries with model_id, response, time.
        """
        import time

        results = []
        client = ollama.Client(host=self.settings.ollama_url, timeout=180.0)

        for model_id in model_ids:
            try:
                start = time.time()
                response = client.generate(
                    model=model_id,
                    prompt=prompt,
                    options={
                        "num_predict": self.settings.max_tokens,
                        "temperature": 0.8,
                    },
                )
                elapsed = time.time() - start

                results.append(
                    {
                        "model_id": model_id,
                        "response": response.response,
                        "time_seconds": round(elapsed, 2),
                        "success": True,
                    }
                )
            except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
                logger.warning(f"Failed to compare model {model_id}: {e}")
                results.append(
                    {
                        "model_id": model_id,
                        "response": "",
                        "error": str(e),
                        "success": False,
                    }
                )

        return results
