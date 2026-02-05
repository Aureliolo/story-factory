"""Model resolution caching for WorldQualityService.

Stores resolved models to avoid redundant tier score calculations during
world building operations. Cache invalidates when mode, VRAM, or settings change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.model_mode_service import ModelModeService
    from src.settings import Settings

logger = logging.getLogger(__name__)


class ModelResolutionCache:
    """Cache for resolved model selections.

    Stores resolved creator and judge models by role to avoid redundant
    tier score calculations. Automatically invalidates when the context
    (mode, VRAM, settings) changes.
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
        self._resolution_context: tuple | None = None  # (mode_id, vram_strategy, vram, settings)
        self._warned_conflicts: set[str] = set()

    def validate(self) -> None:
        """Invalidate cache if mode/VRAM/settings context changed.

        Checks the current mode, VRAM strategy, available VRAM, and model settings
        against the stored context. If any have changed, clears all resolved model
        storage to force re-resolution on next access.
        """
        from src.settings import get_available_vram

        current_mode = self._mode_service.get_current_mode()
        current_vram = get_available_vram()
        # Include model settings in context to detect user configuration changes
        settings_key = (
            self._settings.use_per_agent_models,
            self._settings.default_model,
            tuple(sorted(self._settings.agent_models.items())),
        )
        current_context = (
            current_mode.id,
            self._settings.vram_strategy,
            current_vram,
            settings_key,
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
        logger.debug("Explicitly invalidating model resolution storage")
        self._resolved_creator_models.clear()
        self._resolved_judge_models.clear()
        self._resolution_context = None

    def get_creator_model(self, role: str) -> str | None:
        """Get stored creator model for a role, if cached.

        Args:
            role: The agent role (writer, architect, etc.).

        Returns:
            The cached model ID, or None if not cached.
        """
        self.validate()
        return self._resolved_creator_models.get(role)

    def store_creator_model(self, role: str, model: str) -> None:
        """Store a resolved creator model for a role.

        Args:
            role: The agent role.
            model: The resolved model ID.
        """
        self._resolved_creator_models[role] = model

    def get_judge_model(self, role: str) -> str | None:
        """Get stored judge model for a role, if cached.

        Args:
            role: The agent role (judge, etc.).

        Returns:
            The cached model ID, or None if not cached.
        """
        self.validate()
        return self._resolved_judge_models.get(role)

    def store_judge_model(self, role: str, model: str) -> None:
        """Store a resolved judge model for a role.

        Args:
            role: The agent role.
            model: The resolved model ID.
        """
        self._resolved_judge_models[role] = model

    def has_warned_conflict(self, conflict_key: str) -> bool:
        """Check if a conflict warning has already been issued.

        Args:
            conflict_key: The conflict identifier (e.g., "character:model-id").

        Returns:
            True if the warning was already issued.
        """
        return conflict_key in self._warned_conflicts

    def mark_conflict_warned(self, conflict_key: str) -> None:
        """Mark a conflict as having been warned about.

        Args:
            conflict_key: The conflict identifier.
        """
        self._warned_conflicts.add(conflict_key)
