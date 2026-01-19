"""Model service - handles Ollama model operations."""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import ollama

from settings import AVAILABLE_MODELS, ModelInfo, Settings, get_available_vram, get_model_info
from utils.validation import (
    validate_in_range,
    validate_non_negative,
    validate_not_empty,
    validate_not_empty_collection,
)

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
        logger.debug("Initializing ModelService")
        self.settings = settings
        logger.debug("ModelService initialized successfully")

    def check_health(self) -> OllamaHealth:
        """Check Ollama service health and connectivity.

        Returns:
            OllamaHealth with status information.
        """
        logger.debug(f"check_health called: ollama_url={self.settings.ollama_url}")
        try:
            # Try to list models as a health check
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_health_check_timeout
            )
            client.list()  # Just check connectivity

            # Get VRAM
            vram = get_available_vram()

            logger.info(
                f"Ollama health check successful: {self.settings.ollama_url}, VRAM={vram}GB"
            )
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
        logger.debug("list_installed called")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_list_models_timeout
            )
            response = client.list()
            models = [model.model for model in response.models if model.model]
            logger.info(f"Found {len(models)} installed models")
            return models
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to list models from Ollama: {e}")
            return []

    def list_installed_with_sizes(self) -> dict[str, float]:
        """List installed Ollama models with their actual sizes.

        Returns:
            Dict mapping model ID to size in GB.
        """
        logger.debug("list_installed_with_sizes called")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_list_models_timeout
            )
            response = client.list()
            models = {}
            for model in response.models:
                if model.model:
                    # Ollama returns size in bytes, convert to GB
                    size_bytes = getattr(model, "size", 0) or 0
                    size_gb = round(size_bytes / (1024**3), 1)
                    models[model.model] = size_gb
            logger.info(f"Found {len(models)} installed models with sizes")
            return models
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to list models from Ollama: {e}")
            return {}

    def list_available(self) -> list[ModelStatus]:
        """List all available models with their status.

        Returns:
            List of ModelStatus objects for all known and installed models.
        """
        logger.debug("list_available called")
        installed_with_sizes = self.list_installed_with_sizes()
        installed_ids = set(installed_with_sizes.keys())
        models = []
        seen_ids = set()

        # First, add all known models from AVAILABLE_MODELS
        for model_id, info in AVAILABLE_MODELS.items():
            # Check if any variant of this model is installed
            is_installed = model_id in installed_ids or any(model_id in m for m in installed_ids)

            # Use actual size from Ollama if available, otherwise use hardcoded
            actual_size = installed_with_sizes.get(model_id, info["size_gb"])

            models.append(
                ModelStatus(
                    model_id=model_id,
                    name=info["name"],
                    installed=is_installed,
                    size_gb=actual_size,
                    vram_required=info["vram_required"],
                    quality=info["quality"],
                    speed=info["speed"],
                    uncensored=info["uncensored"],
                    description=info["description"],
                )
            )
            seen_ids.add(model_id)

        # Then, add any installed models that aren't in AVAILABLE_MODELS
        for model_id, size_gb in installed_with_sizes.items():
            if model_id not in seen_ids:
                # Check if this is a variant of a known model (e.g., :latest tag)
                base_name = model_id.split(":")[0] if ":" in model_id else model_id
                known_base = None
                for known_id in AVAILABLE_MODELS:
                    if known_id.startswith(base_name):
                        known_base = AVAILABLE_MODELS[known_id]
                        break

                if known_base:
                    # Use info from known base model
                    models.append(
                        ModelStatus(
                            model_id=model_id,
                            name=f"{known_base['name']} ({model_id.split(':')[-1] if ':' in model_id else 'custom'})",
                            installed=True,
                            size_gb=size_gb,
                            vram_required=known_base["vram_required"],
                            quality=known_base["quality"],
                            speed=known_base["speed"],
                            uncensored=known_base["uncensored"],
                            description=known_base["description"],
                        )
                    )
                else:
                    # Completely unknown model - use defaults with actual size
                    models.append(
                        ModelStatus(
                            model_id=model_id,
                            name=model_id,
                            installed=True,
                            size_gb=size_gb,
                            vram_required=int(size_gb * 1.2) if size_gb > 0 else 8,  # Estimate VRAM
                            quality=5,  # Default quality
                            speed=5,  # Default speed
                            uncensored=True,  # Assume uncensored
                            description="Custom/unknown model",
                        )
                    )

        logger.info(f"Listed {len(models)} available models ({len(installed_ids)} installed)")
        return models

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model.

        Args:
            model_id: The model identifier.

        Returns:
            ModelInfo dictionary.
        """
        validate_not_empty(model_id, "model_id")
        logger.debug(f"get_model_info called: model_id={model_id}")
        return get_model_info(model_id)

    def pull_model(self, model_id: str) -> Generator[dict[str, Any]]:
        """Pull a model from Ollama registry with progress updates.

        Args:
            model_id: The model ID to pull.

        Yields:
            Progress dictionaries with status, completed, total.
        """
        validate_not_empty(model_id, "model_id")
        logger.info(f"Starting download of model: {model_id}")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_pull_model_timeout
            )

            last_status = ""
            for progress in client.pull(model_id, stream=True):
                status = progress.get("status", "")
                # Handle None values explicitly - Ollama API may return None
                completed = progress.get("completed") or 0
                total = progress.get("total") or 0

                # Log status changes (not every progress update)
                if status != last_status:
                    if total > 0:
                        pct = (completed / total) * 100
                        logger.debug(f"[{model_id}] {status} ({pct:.1f}%)")
                    else:
                        logger.debug(f"[{model_id}] {status}")
                    last_status = status

                yield {
                    "status": status,
                    "completed": completed,
                    "total": total,
                }

            logger.info(f"Download completed: {model_id}")

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error pulling model {model_id}: {e}")
            yield {"status": f"Error: {e}", "error": True}
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error pulling model {model_id}: {e}")
            yield {"status": f"Failed to pull model: {e}", "error": True}
        except Exception as e:
            logger.exception(f"Unexpected error pulling model {model_id}: {e}")
            yield {"status": f"Unexpected error: {e}", "error": True}

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from Ollama.

        Args:
            model_id: The model ID to delete.

        Returns:
            True if deleted successfully.
        """
        validate_not_empty(model_id, "model_id")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_delete_model_timeout
            )
            client.delete(model_id)
            logger.info(f"Deleted model: {model_id}")
            return True
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def check_model_update(self, model_id: str) -> dict[str, Any]:
        """Check if a model has an update available.

        This works by attempting a pull - if Ollama reports 'already exists'
        or shows no download progress, the model is up to date.

        Args:
            model_id: The model ID to check.

        Returns:
            Dict with 'has_update' (bool), 'message' (str), 'error' (bool optional).
        """
        validate_not_empty(model_id, "model_id")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_check_update_timeout
            )

            # Pull with stream to check status without full download
            # Track whether we've seen actual downloading (not just layer verification)
            seen_downloading = False
            for progress in client.pull(model_id, stream=True):
                status = progress.get("status", "").lower()
                total = progress.get("total") or 0
                completed = progress.get("completed") or 0

                # "downloading" status means actually fetching new data
                if "downloading" in status:
                    seen_downloading = True
                    return {
                        "has_update": True,
                        "message": "Update available",
                    }

                # If pulling a layer and completed is significantly less than total,
                # it's an actual download (not just verification)
                if (
                    "pulling" in status
                    and total > 0
                    and completed < total * self.settings.model_download_threshold
                ):
                    # Wait a moment to confirm it's not just instant verification
                    import time

                    time.sleep(self.settings.model_verification_sleep)
                    seen_downloading = True
                    return {
                        "has_update": True,
                        "message": "Update available",
                    }

                # If already up to date
                if "up to date" in status or "already exists" in status:
                    return {
                        "has_update": False,
                        "message": "Already up to date",
                    }

                # Success without actual download means up to date
                if "success" in status and not seen_downloading:
                    return {
                        "has_update": False,
                        "message": "Already up to date",
                    }

            return {
                "has_update": False,
                "message": "Already up to date",
            }

        except ollama.ResponseError as e:
            logger.warning(f"Error checking update for {model_id}: {e}")
            return {
                "has_update": False,
                "message": f"Check failed: {e}",
                "error": True,
            }
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error checking update for {model_id}: {e}")
            return {
                "has_update": False,
                "message": f"Connection failed: {e}",
                "error": True,
            }

    def get_vram(self) -> int:
        """Get available VRAM in GB.

        Returns:
            VRAM in gigabytes.
        """
        logger.debug("get_vram called")
        vram = get_available_vram()
        logger.debug(f"Available VRAM: {vram}GB")
        return vram

    def get_recommended_model(self, role: str | None = None) -> str:
        """Get recommended model based on available VRAM and role.

        Args:
            role: Optional agent role for role-specific recommendation.

        Returns:
            Recommended model ID.
        """
        logger.debug(f"get_recommended_model called: role={role}")
        vram = self.get_vram()

        if role:
            model = self.settings.get_model_for_agent(role, vram)
            logger.info(f"Recommended model for {role} with {vram}GB VRAM: {model}")
            return model

        # General recommendation based on VRAM
        if vram >= 24:
            model = "huihui_ai/llama3.3-abliterated:70b-q4_K_M"
        elif vram >= 14:
            model = "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0"
        else:
            model = "huihui_ai/dolphin3-abliterated:8b"

        logger.info(f"General recommended model for {vram}GB VRAM: {model}")
        return model

    def get_models_for_vram(self, min_vram: int | None = None) -> list[ModelStatus]:
        """Get models that fit within available VRAM.

        Args:
            min_vram: Optional minimum VRAM override. Defaults to detected.

        Returns:
            List of ModelStatus for models that fit.
        """
        if min_vram is not None:
            validate_non_negative(min_vram, "min_vram")
        logger.debug(f"get_models_for_vram called: min_vram={min_vram}")
        vram = min_vram if min_vram is not None else self.get_vram()
        all_models = self.list_available()

        filtered = [m for m in all_models if m.vram_required <= vram]
        logger.info(f"Found {len(filtered)}/{len(all_models)} models that fit in {vram}GB VRAM")
        return filtered

    def test_model(self, model_id: str) -> tuple[bool, str]:
        """Test if a model works by running a simple prompt.

        Args:
            model_id: The model ID to test.

        Returns:
            Tuple of (success, message).
        """
        validate_not_empty(model_id, "model_id")
        logger.debug(f"test_model called: model_id={model_id}")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_generate_timeout
            )
            response = client.generate(
                model=model_id,
                prompt="Say 'hello' in one word.",
                options={"num_predict": 10},
            )

            if response.response:
                logger.info(f"Model {model_id} test successful: {response.response[:50]}")
                return True, f"Model working: {response.response[:50]}"

            logger.warning(f"Model {model_id} test returned empty response")
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
        validate_in_range(min_quality, "min_quality", min_val=1, max_val=10)
        if max_vram is not None:
            validate_non_negative(max_vram, "max_vram")
        logger.debug(
            f"get_model_by_quality called: min_quality={min_quality}, "
            f"max_vram={max_vram}, uncensored_required={uncensored_required}"
        )
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
        logger.info(
            f"Found {len(filtered)} models matching quality>={min_quality}, "
            f"vram<={vram}GB, uncensored={uncensored_required}"
        )
        return filtered

    def compare_models(
        self,
        model_ids: list[str],
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Compare multiple models on the same prompt.

        Args:
            model_ids: List of model IDs to compare.
            prompt: The prompt to test.

        Returns:
            List of result dictionaries with model_id, response, time.
        """
        import time

        validate_not_empty_collection(model_ids, "model_ids")
        validate_not_empty(prompt, "prompt")
        logger.debug(f"compare_models called: models={model_ids}, prompt_length={len(prompt)}")
        results = []
        client = ollama.Client(
            host=self.settings.ollama_url, timeout=self.settings.ollama_capability_check_timeout
        )

        for model_id in model_ids:
            try:
                logger.info(f"Comparing model: {model_id}")
                start = time.time()
                response = client.generate(
                    model=model_id,
                    prompt=prompt,
                    options={
                        "num_predict": self.settings.max_tokens,
                        "temperature": self.settings.temp_model_evaluation,
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
                logger.info(f"Model {model_id} completed in {elapsed:.2f}s")
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

        logger.info(
            f"Model comparison complete: {len([r for r in results if r['success']])}/{len(model_ids)} successful"
        )
        return results
