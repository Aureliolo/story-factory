"""Settings package for Story Factory.

This package provides application settings management, model registry,
and utility functions for Ollama model management.

All functionality is now in focused modules:
- _paths.py: Path constants for settings and output directories
- _types.py: TypedDicts and role configurations
- _model_registry.py: Recommended models registry
- _validation.py: Settings validation mixin
- _settings.py: Main Settings dataclass
- _utils.py: Utility functions for Ollama model management
"""

# Re-export path constants
# Re-export model registry
from src.settings._model_registry import RECOMMENDED_MODELS
from src.settings._paths import (
    BACKUPS_DIR,
    SETTINGS_FILE,
    STORIES_DIR,
    WORLDS_DIR,
)

# Re-export main Settings class
from src.settings._settings import Settings

# Re-export type definitions and role configs
from src.settings._types import (
    AGENT_ROLES,
    REFINEMENT_TEMP_DECAY_CURVES,
    AgentRoleInfo,
    AgentSettings,
    ModelInfo,
)

# Re-export utility functions
from src.settings._utils import (
    get_available_vram,
    get_installed_models,
    get_installed_models_with_sizes,
    get_model_info,
)

__all__ = [
    "AGENT_ROLES",
    "BACKUPS_DIR",
    "RECOMMENDED_MODELS",
    "REFINEMENT_TEMP_DECAY_CURVES",
    "SETTINGS_FILE",
    "STORIES_DIR",
    "WORLDS_DIR",
    "AgentRoleInfo",
    "AgentSettings",
    "ModelInfo",
    "Settings",
    "get_available_vram",
    "get_installed_models",
    "get_installed_models_with_sizes",
    "get_model_info",
]
