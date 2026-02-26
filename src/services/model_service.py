"""Model service - handles Ollama model operations."""

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import ollama

from src.settings import (
    RECOMMENDED_MODELS,
    ModelInfo,
    Settings,
    get_available_vram,
    get_model_info,
)
from src.utils.validation import (
    validate_in_range,
    validate_non_negative,
    validate_not_empty,
    validate_not_empty_collection,
)

logger = logging.getLogger(__name__)


class _TTLCached[T]:
    """TTL cache for a single value with time-based invalidation.

    Used by :class:`ModelService` to avoid redundant Ollama API calls from
    the header refresh loop.  Not thread-safe — relies on the single-threaded
    NiceGUI event loop for safe access from UI refresh callbacks.
    """

    __slots__ = ("_time", "_value", "ttl")

    def __init__(self, ttl: float) -> None:
        self._value: T | None = None
        self._time: float = 0.0
        self.ttl = ttl

    def get(self, now: float) -> T | None:
        """Return cached value if still within TTL, else None."""
        if self._value is not None and (now - self._time) < self.ttl:
            return self._value
        return None

    def set(self, value: T, now: float) -> None:
        """Store a value with the current timestamp."""
        self._value = value
        self._time = now

    def invalidate(self) -> None:
        """Clear the cached value so the next ``get`` returns None."""
        self._value = None


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
    cold_start_models: list[str] = field(default_factory=list)


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
        # State tracking for change-only logging
        self._last_health_healthy: bool | None = None
        self._last_health_vram: int | None = None
        self._last_model_count: int | None = None
        self._last_model_count_with_sizes: int | None = None

        # TTL caches for frequently-polled methods (TTLs from settings)
        self._health_cache: _TTLCached[OllamaHealth] = _TTLCached(settings.model_health_cache_ttl)
        self._installed_cache: _TTLCached[list[str]] = _TTLCached(
            settings.model_installed_cache_ttl
        )
        self._vram_cache: _TTLCached[int] = _TTLCached(settings.model_vram_cache_ttl)

        logger.debug("ModelService initialized successfully")

    def _log_health_failure(self, message: str) -> None:
        """Log a health check failure at the appropriate level.

        Logs at WARNING on first failure or transition from healthy to unhealthy,
        and at DEBUG on consecutive failures to reduce log noise.

        Args:
            message: The log message describing the failure.
        """
        if self._last_health_healthy is not False:
            logger.warning(message)
        else:
            logger.debug(message)

    def check_health(self) -> OllamaHealth:
        """Check Ollama service health and connectivity.

        Results are cached for ``settings.model_health_cache_ttl`` seconds
        to avoid redundant API calls from the header refresh loop.
        Unhealthy results are **not** cached so recovery is detected immediately.

        Returns:
            OllamaHealth with status information.
        """
        now = time.monotonic()
        cached = self._health_cache.get(now)
        if cached is not None:
            logger.debug("check_health: returning cached result")
            return cached

        logger.debug(f"check_health called: ollama_url={self.settings.ollama_url}")
        try:
            # Try to list models as a health check
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_health_check_timeout
            )
            client.list()  # Just check connectivity

            # Get VRAM
            vram = get_available_vram()

            if not self._last_health_healthy or self._last_health_vram != vram:
                logger.info(
                    f"Ollama health check successful: {self.settings.ollama_url}, VRAM={vram}GB"
                )
            else:
                logger.debug(
                    f"Ollama health check successful: {self.settings.ollama_url}, VRAM={vram}GB"
                )
            self._last_health_healthy = True
            self._last_health_vram = vram

            # Detect cold-start models: configured models not currently loaded in VRAM
            cold_start: list[str] = []
            try:
                running = self.get_running_models()
                running_names = {str(m["name"]) for m in running}
                # Collect unique non-"auto" models from agent_models + default_model
                configured: set[str] = set()
                if self.settings.default_model != "auto":
                    configured.add(self.settings.default_model)
                for model_name in self.settings.agent_models.values():
                    if model_name != "auto":
                        configured.add(model_name)
                for model_name in sorted(configured):
                    if model_name not in running_names:
                        cold_start.append(model_name)
            except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
                logger.warning("Cold-start detection failed (non-fatal): %s", e)

            result = OllamaHealth(
                is_healthy=True,
                message="Ollama is running",
                version=None,  # Ollama doesn't expose version easily
                available_vram=vram,
                cold_start_models=cold_start,
            )
            self._health_cache.set(result, now)
            return result
        except ollama.ResponseError as e:
            self._log_health_failure(f"Ollama API error during health check: {e}")
            self._last_health_healthy = False
            # Error result intentionally NOT cached — next call will retry,
            # allowing the UI to detect recovery immediately.
            return OllamaHealth(
                is_healthy=False,
                message=f"Ollama API error: {e}",
            )
        except (ConnectionError, TimeoutError) as e:
            self._log_health_failure(f"Cannot connect to Ollama at {self.settings.ollama_url}: {e}")
            self._last_health_healthy = False
            # Error result intentionally NOT cached — next call will retry.
            return OllamaHealth(
                is_healthy=False,
                message=f"Cannot connect to Ollama: {e}",
            )

    def list_installed(self) -> list[str]:
        """List installed Ollama models.

        Results are cached for ``settings.model_installed_cache_ttl`` seconds.

        Returns:
            List of installed model IDs.
        """
        now = time.monotonic()
        cached = self._installed_cache.get(now)
        if cached is not None:
            logger.debug("list_installed: returning cached result")
            return list(cached)  # Defensive copy

        logger.debug("list_installed called")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_list_models_timeout
            )
            response = client.list()
            models = [model.model for model in response.models if model.model]
            count = len(models)
            if self._last_model_count != count:
                logger.info(f"Found {count} installed models")
                self._last_model_count = count
            else:
                logger.debug(f"Found {count} installed models")
            self._installed_cache.set(models, now)
            return list(models)  # Defensive copy, same as cache-hit path
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to list models from Ollama: {e}")
            # Error result intentionally NOT cached — next call will retry.
            # Callers receive [] and cannot distinguish from "no models installed".
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
            count = len(models)
            if self._last_model_count_with_sizes != count:
                logger.info(f"Found {count} installed models with sizes")
                self._last_model_count_with_sizes = count
            else:
                logger.debug(f"Found {count} installed models with sizes")
            return models
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to list models from Ollama: {e}")
            return {}

    def list_available(self) -> list[ModelStatus]:
        """List all available models with their status.

        Returns:
            List of ModelStatus objects for all known and installed models.
            Includes both RECOMMENDED_MODELS and any installed models.
        """
        logger.debug("list_available called")
        installed_with_sizes = self.list_installed_with_sizes()
        installed_ids = set(installed_with_sizes.keys())
        models = []
        seen_ids: set[str] = set()

        # First, add all RECOMMENDED_MODELS
        for model_id, info in RECOMMENDED_MODELS.items():
            # Check if any variant of this model is installed
            is_installed = model_id in installed_ids or any(model_id in m for m in installed_ids)

            # Use actual size from Ollama if available, otherwise use recommended
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

        # Then, add any installed models not in RECOMMENDED_MODELS
        for model_id, size_gb in installed_with_sizes.items():
            if model_id not in seen_ids:
                # Check if this is a variant of a recommended model
                base_name = model_id.split(":")[0] if ":" in model_id else model_id
                known_base = None
                for known_id in RECOMMENDED_MODELS:
                    if known_id.startswith(base_name) or base_name in known_id:
                        known_base = RECOMMENDED_MODELS[known_id]
                        break

                if known_base:
                    # Use info from known base model
                    tag = model_id.split(":")[-1] if ":" in model_id else "custom"
                    models.append(
                        ModelStatus(
                            model_id=model_id,
                            name=f"{known_base['name']} ({tag})",
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
                    # Completely unknown model - estimate from size
                    quality = min(10, int(size_gb / 4) + 4) if size_gb > 0 else 5
                    speed = max(1, 10 - int(size_gb / 5)) if size_gb > 0 else 5
                    models.append(
                        ModelStatus(
                            model_id=model_id,
                            name=model_id,
                            installed=True,
                            size_gb=size_gb,
                            vram_required=int(size_gb * 1.2) if size_gb > 0 else 8,
                            quality=quality,
                            speed=speed,
                            uncensored=True,  # Assume uncensored for local models
                            description="Automatically detected model",
                        )
                    )
                seen_ids.add(model_id)

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
            self.invalidate_caches()

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
            self.invalidate_caches()
            return True
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def invalidate_caches(self) -> None:
        """Clear all TTL caches.

        Called after model install/delete to ensure subsequent queries
        reflect the new state immediately.
        """
        self._health_cache.invalidate()
        self._installed_cache.invalidate()
        self._vram_cache.invalidate()
        logger.debug("ModelService caches invalidated")

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

    def get_running_models(self) -> list[dict[str, str | float]]:
        """Query Ollama /api/ps to get currently loaded (running) models.

        Returns:
            List of dicts with 'name' and 'size_gb' for each loaded model.
            Empty list on connection failure.
        """
        logger.debug("get_running_models called")
        try:
            client = ollama.Client(
                host=self.settings.ollama_url, timeout=self.settings.ollama_health_check_timeout
            )
            response = client.ps()
            models = []
            for model in response.models:
                name = getattr(model, "name", "") or getattr(model, "model", "")
                size_bytes = getattr(model, "size", 0) or 0
                size_gb = round(size_bytes / (1024**3), 1)
                models.append({"name": name, "size_gb": size_gb})
            logger.debug("Found %d running models via /api/ps", len(models))
            return models
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            logger.warning("Failed to query running models from Ollama: %s", e)
            return []
        except AttributeError:
            # Older Ollama client versions may not have .ps()
            logger.debug("Ollama client does not support ps() — skipping running model check")
            return []

    def log_model_load_state(self, target_model: str | None = None) -> None:
        """Log whether target model is already loaded in Ollama's VRAM.

        Queries /api/ps and logs which models are resident. If a target model
        is specified and not loaded, logs a warning that the first call will
        incur a cold-start penalty.

        Args:
            target_model: Optional model ID to check for. If None, just logs
                what's currently loaded.
        """
        running = self.get_running_models()
        running_names = [str(m["name"]) for m in running]

        if not running:
            logger.info("Ollama model load state: no models currently loaded in VRAM")
            if target_model:
                logger.warning(
                    "No models loaded in VRAM — first LLM call for any model will "
                    "incur a cold-start penalty (~30-60s)",
                )
            return

        running_desc = ", ".join(f"{m['name']} ({m['size_gb']}GB)" for m in running)
        logger.info("Ollama model load state: %d model(s) loaded — %s", len(running), running_desc)

        if target_model:
            if target_model in running_names:
                logger.info(
                    "Target model '%s' is already loaded — no cold-start penalty expected",
                    target_model,
                )
            else:
                logger.warning(
                    "Target model '%s' is NOT loaded (loaded: %s) — first LLM call will "
                    "incur a cold-start penalty (~30-60s for loading model into VRAM)",
                    target_model,
                    ", ".join(running_names),
                )

    def get_vram(self) -> int:
        """Get available VRAM in GB.

        Results are cached for ``settings.model_vram_cache_ttl`` seconds.

        Returns:
            VRAM in gigabytes.
        """
        now = time.monotonic()
        cached = self._vram_cache.get(now)
        if cached is not None:
            logger.debug("get_vram: returning cached result (%dGB)", cached)
            return cached

        logger.debug("get_vram called")
        vram = get_available_vram()
        logger.debug(f"Available VRAM: {vram}GB")
        self._vram_cache.set(vram, now)
        return vram

    def get_recommended_model(self, role: str | None = None) -> str:
        """Get recommended model based on available VRAM and role.

        Uses fully automatic model selection based on installed models.
        No hardcoded model names - selection is based on what's actually installed.

        Args:
            role: Optional agent role for role-specific recommendation.

        Returns:
            Recommended model ID from installed models.

        Raises:
            ValueError: If no models are installed.
        """
        logger.debug(f"get_recommended_model called: role={role}")
        vram = self.get_vram()

        if role:
            model = self.settings.get_model_for_agent(role, vram)
            logger.info(f"Recommended model for {role} with {vram}GB VRAM: {model}")
            return model

        # General recommendation: use auto-selection for "writer" role
        # (writer requires highest quality, so it's a good default)
        model = self.settings.get_model_for_agent("writer", vram)
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
        logger.debug(f"Found {len(filtered)}/{len(all_models)} models that fit in {vram}GB VRAM")
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
        logger.debug(
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
