"""Web-configurable settings for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
This package re-exports all public symbols for backward compatibility.
"""

from src.settings._model_registry import RECOMMENDED_MODELS
from src.settings._paths import BACKUPS_DIR, SETTINGS_FILE, STORIES_DIR, WORLDS_DIR
from src.settings._settings import AgentSettings, Settings
from src.settings._types import (
    AGENT_ROLES,
    LOG_LEVELS,
    MINIMUM_ROLE_QUALITY,
    REFINEMENT_TEMP_DECAY_CURVES,
    AgentRoleInfo,
    ModelInfo,
    check_minimum_quality,
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
    "LOG_LEVELS",
    "MINIMUM_ROLE_QUALITY",
    "RECOMMENDED_MODELS",
    "REFINEMENT_TEMP_DECAY_CURVES",
    "SETTINGS_FILE",
    "STORIES_DIR",
    "WORLDS_DIR",
    "AgentRoleInfo",
    "AgentSettings",
    "ModelInfo",
    "Settings",
    "check_minimum_quality",
    "get_available_vram",
    "get_installed_models",
    "get_installed_models_with_sizes",
    "get_model_info",
]
