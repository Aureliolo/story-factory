"""Minimal NiceGUI app entry point for component tests.

This file is used as the main_file for NiceGUI's User fixture testing.
It initializes the app and calls ui.run() for the testing framework.
The testing framework intercepts ui.run() so it doesn't actually start a server.
"""

import atexit
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from nicegui import ui

from src.services import ServiceContainer
from src.settings import Settings
from src.ui.app import StoryFactoryApp


def setup_test_app():
    """Initialize the app for testing.

    Uses temporary paths to prevent writing to real analytics database and project directories.
    Registers cleanup handler to remove temp directory on process exit.
    """
    settings = Settings()
    # Use temp directory for all data to avoid polluting real output
    temp_dir = Path(tempfile.mkdtemp())
    # Register cleanup to prevent temp dir accumulation (runpy.run_path may not
    # execute finally blocks, but atexit handlers run on normal interpreter shutdown)
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    mode_db_path = temp_dir / "test_mode.db"
    stories_dir = temp_dir / "stories"
    worlds_dir = temp_dir / "worlds"
    stories_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    # Patch at ALL import locations to prevent real data pollution
    with (
        patch("src.memory.mode_database.DEFAULT_DB_PATH", mode_db_path),
        patch("src.settings.STORIES_DIR", stories_dir),
        patch("src.settings.WORLDS_DIR", worlds_dir),
        patch("src.services.project_service.STORIES_DIR", stories_dir),
        patch("src.services.project_service.WORLDS_DIR", worlds_dir),
        patch("src.services.backup_service.STORIES_DIR", stories_dir),
        patch("src.services.backup_service.WORLDS_DIR", worlds_dir),
    ):
        services = ServiceContainer(settings)
        story_app = StoryFactoryApp(services)
        story_app.build()
        return story_app


# Only run when executed as main (via NiceGUI testing framework's runpy.run_path)
if __name__ in {"__main__", "__mp_main__"}:
    _app = setup_test_app()
    # Required by NiceGUI testing framework - intercepted in test mode
    ui.run(reload=False, show=False)
