"""Path constants for Story Factory settings and output directories.

IMPORTANT: All writable paths defined here are isolated in tests via autouse
fixtures in tests/conftest.py (isolate_all_production_paths,
isolate_project_directories). If you add a new writable path constant,
add it to the appropriate fixture to prevent tests from writing to real files.
"""

from pathlib import Path

# Settings file path (sibling to the settings package directory)
SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

# Centralized paths for story and world output files
# Go up from src/settings/ to project root, then into output/
STORIES_DIR = Path(__file__).parent.parent.parent / "output" / "stories"
WORLDS_DIR = Path(__file__).parent.parent.parent / "output" / "worlds"
BACKUPS_DIR = Path(__file__).parent.parent.parent / "output" / "backups"
