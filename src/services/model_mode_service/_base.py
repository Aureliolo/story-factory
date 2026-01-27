"""Base class and initialization for ModelModeService."""

import logging
from pathlib import Path

from src.memory.mode_database import ModeDatabase
from src.memory.mode_models import (
    GenerationMode,
    LearningSettings,
    get_preset_mode,
    list_preset_modes,
)
from src.settings import Settings

logger = logging.getLogger(__name__)


class ModelModeServiceBase:
    """Base class for model mode service with initialization and core state."""

    def __init__(
        self,
        settings: Settings,
        db_path: Path | str | None = None,
    ):
        """Initialize model mode service.

        Args:
            settings: Application settings.
            db_path: Path to scoring database. Defaults to output/model_scores.db
        """
        logger.debug(f"Initializing ModelModeService: db_path={db_path}")
        self.settings = settings
        # Default to output/model_scores.db at project root
        default_db = Path(__file__).parent.parent.parent.parent / "output" / "model_scores.db"
        self._db_path = Path(db_path) if db_path else default_db
        self._db = ModeDatabase(self._db_path)

        # Current mode
        self._current_mode: GenerationMode | None = None

        # Learning settings
        self._learning_settings = LearningSettings()

        # Track chapters for periodic triggers
        self._chapters_since_analysis = 0

        # Loaded model tracking (for VRAM management)
        self._loaded_models: set[str] = set()
        logger.debug("ModelModeService initialized successfully")

    def get_current_mode(self) -> GenerationMode:
        """Get the current generation mode.

        Returns preset 'balanced' if no mode is set.
        """
        if self._current_mode is None:
            self._current_mode = get_preset_mode("balanced") or list_preset_modes()[0]
        return self._current_mode
