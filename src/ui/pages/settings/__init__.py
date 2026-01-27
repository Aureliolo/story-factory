"""Settings page package - application configuration.

This package provides the SettingsPage class composed from multiple mixins
for modularity and maintainability.
"""

from src.services import ServiceContainer
from src.ui.state import AppState

from ._advanced import AdvancedMixin
from ._connection import ConnectionMixin
from ._interaction import InteractionMixin
from ._models import ModelsMixin
from ._modes import ModesMixin
from ._page import SettingsPageBase
from ._world_gen import WorldGenMixin


class SettingsPage(
    ConnectionMixin,
    ModelsMixin,
    InteractionMixin,
    ModesMixin,
    WorldGenMixin,
    AdvancedMixin,
    SettingsPageBase,
):
    """Settings page for application configuration.

    Features:
    - Ollama connection settings
    - Model selection (per-agent or global)
    - Temperature settings
    - Interaction mode
    - Context limits
    - Generation modes (presets for model combinations)
    - Adaptive learning settings (autonomy, triggers, thresholds)

    This class composes functionality from multiple mixins:
    - ConnectionMixin: Ollama connection settings
    - ModelsMixin: Model selection and temperature settings
    - InteractionMixin: Interaction mode and context settings
    - ModesMixin: Generation modes and learning settings
    - WorldGenMixin: World generation and story structure settings
    - AdvancedMixin: Data integrity, advanced LLM, and relationship validation settings
    """

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize settings page.

        Args:
            state: Application state.
            services: Service container.
        """
        super().__init__(state, services)


__all__ = ["SettingsPage"]
