"""UI module for Story Factory.

NiceGUI-based web interface with:
- Write Story page (Fundamentals + Live Writing)
- World Builder page (entity/relationship management)
- Projects page (project management)
- Settings page (configuration)
- Models page (Ollama model management)
"""

from .app import StoryFactoryApp, create_app
from .state import AppState

__all__ = [
    "AppState",
    "StoryFactoryApp",
    "create_app",
]
