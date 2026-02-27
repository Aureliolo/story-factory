"""Model resolution caching for WorldQualityService.

Stores resolved models to avoid redundant tier score calculations during
world building operations. Cache invalidates when mode, VRAM, settings, or
installed models change.
"""

import logging
import threading
import time

from src.services.model_mode_service import ModelModeService
from src.settings import Settings

logger = logging.getLogger(__name__)

# Time-to-live for environment context cache (VRAM and installed models).
# These checks involve subprocess calls (nvidia-smi, ollama list), so we
# cache them to avoid overhead on every cache access.  During world builds
# (which can run 5-15 minutes) the environment rarely changes, so 5 minutes
# is a safe TTL — explicit invalidation via invalidate() still forces a
# fresh check when settings change.
_ENV_CONTEXT_TTL_SECONDS = 300.0


class ModelResolutionCache:
    """Cache for resolved model selections.

    Stores resolved creator and judge models by role to avoid redundant
    tier score calculations. Automatically invalidates when the context
    (mode, VRAM, settings, installed models) changes.

    Thread-safe: All cache operations are protected by a reentrant lock.
    """

    def __init__(self, settings: Settings, mode_service: ModelModeService):
        """Initialize the model resolution cache.

        Args:
            settings: Application settings.
            mode_service: Model mode service for model selection.
        """
        self._settings = settings
        self._mode_service = mode_service
        self._resolved_creator_models: dict[str, str] = {}  # role -> model_id
        self._resolved_judge_models: dict[str, str] = {}  # role -> model_id
        self._resolution_context: tuple | None = None
        self._warned_conflicts: set[str] = set()
        self._lock = threading.RLock()
        # Environment context cache (VRAM + installed models) with TTL
        self._env_context_cache: tuple[int, tuple] | None = None
        self._env_context_timestamp: float = 0.0

    def _get_env_context(self) -> tuple[int, tuple]:
        """Get environment context (VRAM and installed models) with caching.

        Subprocess calls to nvidia-smi and ollama are expensive, so we cache
        the results for a short duration to avoid overhead on every cache access.

        Returns:
            Tuple of (available_vram, installed_models_key).

        Note: This method assumes the caller holds the lock.
        """
        from src.settings import get_available_vram, get_installed_models_with_sizes

        now = time.monotonic()
        if (
            self._env_context_cache is None
            or (now - self._env_context_timestamp) > _ENV_CONTEXT_TTL_SECONDS
        ):
            current_vram = get_available_vram()
            installed_models_key = tuple(sorted(get_installed_models_with_sizes().items()))
            new_env = (current_vram, installed_models_key)
            if self._env_context_cache is not None and new_env != self._env_context_cache:
                logger.debug(
                    "Environment context changed (vram=%s, models=%d)",
                    current_vram,
                    len(installed_models_key),
                )
            self._env_context_cache = new_env
            self._env_context_timestamp = now

        return self._env_context_cache

    def _check_context(self) -> None:
        """Invalidate cache if mode/VRAM/settings/models context changed.

        Checks the current mode, VRAM strategy, available VRAM, model settings,
        custom model tags, and installed models against the stored context.
        If any have changed, clears all resolved model storage to force
        re-resolution on next access.

        Note: This method assumes the caller holds the lock.
        """
        current_mode = self._mode_service.get_current_mode()

        # Get environment context with TTL caching to avoid subprocess overhead
        current_vram, installed_models_key = self._get_env_context()

        # Include model settings in context to detect user configuration changes
        settings_key = (
            self._settings.use_per_agent_models,
            self._settings.default_model,
            tuple(sorted(self._settings.agent_models.items())),
            tuple(sorted((k, tuple(v)) for k, v in self._settings.custom_model_tags.items())),
        )

        current_context = (
            current_mode.id,
            self._settings.vram_strategy,
            current_vram,
            settings_key,
            installed_models_key,
        )

        if self._resolution_context != current_context:
            logger.debug(
                "Model resolution context changed, invalidating cache "
                "(mode=%s, vram_strategy=%s, vram=%s)",
                current_mode.id,
                self._settings.vram_strategy,
                current_vram,
            )
            self._resolved_creator_models.clear()
            self._resolved_judge_models.clear()
            self._warned_conflicts.clear()
            self._resolution_context = current_context

    def invalidate(self) -> None:
        """Invalidate all resolved model storage.

        Call this when settings change (e.g., user changes model configuration)
        to force re-resolution of models on next access.
        """
        with self._lock:
            logger.debug("Explicitly invalidating model resolution storage")
            self._resolved_creator_models.clear()
            self._resolved_judge_models.clear()
            self._warned_conflicts.clear()
            self._resolution_context = None
            # Also clear environment context cache to force fresh check
            self._env_context_cache = None
            self._env_context_timestamp = 0.0

    def get_creator_model(self, role: str) -> str | None:
        """Get cached creator model for a role, if available.

        Args:
            role: The agent role (writer, architect, etc.).

        Returns:
            The cached model ID, or None if not cached.
        """
        with self._lock:
            self._check_context()
            return self._resolved_creator_models.get(role)

    def store_creator_model(self, role: str, model: str) -> str:
        """Store a resolved creator model for a role, returning the canonical value.

        Uses a double-check pattern: if another thread already stored a model
        for this role while the caller was resolving, returns the existing value
        instead of overwriting it.  This prevents duplicate resolution log entries
        when parallel workers resolve the same role concurrently.

        Note: ``_check_context()`` is intentionally not called here. If the
        resolution context changed between the cache miss (``get``) and this
        store, the stale entry will be detected and cleared by the next
        ``get()`` call.

        Args:
            role: The agent role.
            model: The resolved model ID.

        Returns:
            The model ID that is now cached for this role (may differ from *model*
            if another thread stored first).
        """
        with self._lock:
            existing = self._resolved_creator_models.get(role)
            if existing is not None:
                return existing
            self._resolved_creator_models[role] = model
            return model

    def get_judge_model(self, role: str) -> str | None:
        """Get cached judge model for a role, if available.

        Args:
            role: The agent role (judge, etc.).

        Returns:
            The cached model ID, or None if not cached.
        """
        with self._lock:
            self._check_context()
            return self._resolved_judge_models.get(role)

    def store_judge_model(self, role: str, model: str) -> str:
        """Store a resolved judge model for a role, returning the canonical value.

        Uses a double-check pattern: if another thread already stored a model
        for this role while the caller was resolving, returns the existing value
        instead of overwriting it.

        Note: ``_check_context()`` is intentionally not called here. If the
        resolution context changed between the cache miss (``get``) and this
        store, the stale entry will be detected and cleared by the next
        ``get()`` call.

        Args:
            role: The agent role.
            model: The resolved model ID.

        Returns:
            The model ID that is now cached for this role.
        """
        with self._lock:
            existing = self._resolved_judge_models.get(role)
            if existing is not None:
                return existing
            self._resolved_judge_models[role] = model
            return model

    def has_warned_conflict(self, conflict_key: str) -> bool:
        """Check if a conflict warning has already been issued.

        Args:
            conflict_key: The conflict identifier (e.g., "character:model-id").

        Returns:
            True if the warning was already issued.
        """
        with self._lock:
            return conflict_key in self._warned_conflicts

    def mark_conflict_warned(self, conflict_key: str) -> None:
        """Mark a conflict as having been warned about.

        Args:
            conflict_key: The conflict identifier.
        """
        with self._lock:
            self._warned_conflicts.add(conflict_key)
