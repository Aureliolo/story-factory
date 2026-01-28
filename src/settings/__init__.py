"""Web-configurable settings for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
This package re-exports all public symbols for backward compatibility.
"""

from src.settings._model_registry import RECOMMENDED_MODELS
from src.settings._paths import BACKUPS_DIR, SETTINGS_FILE, STORIES_DIR, WORLDS_DIR
from src.settings._settings import AgentSettings, Settings
from src.settings._types import (
    AGENT_ROLES,
    REFINEMENT_TEMP_DECAY_CURVES,
    AgentRoleInfo,
    ModelInfo,
)
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
