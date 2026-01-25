"""Minimal NiceGUI app entry point for component tests.

This file is used as the main_file for NiceGUI's User fixture testing.
It initializes the app and calls ui.run() for the testing framework.
The testing framework intercepts ui.run() so it doesn't actually start a server.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from nicegui import ui

from src.services import ServiceContainer
from src.settings import Settings
from src.ui.app import StoryFactoryApp


def setup_test_app():
    """Initialize the app for testing.

    Uses a temporary database path to prevent writing to the real analytics database.
    """
    settings = Settings()
    # Use temp directory for mode database to avoid polluting real analytics
    temp_dir = Path(tempfile.mkdtemp())
    mode_db_path = temp_dir / "test_mode.db"

    with patch("src.memory.mode_database.DEFAULT_DB_PATH", mode_db_path):
        services = ServiceContainer(settings)
        story_app = StoryFactoryApp(services)
        story_app.build()
        return story_app


# Only run when executed as main (via NiceGUI testing framework's runpy.run_path)
if __name__ in {"__main__", "__mp_main__"}:
    _app = setup_test_app()
    # Required by NiceGUI testing framework - intercepted in test mode
    ui.run(reload=False, show=False)
