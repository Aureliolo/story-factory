"""Path constants for Story Factory settings and output directories."""

from pathlib import Path

# Settings file path (sibling to the settings package directory)
SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

# Centralized paths for story and world output files
# Go up from src/settings/ to project root, then into output/
STORIES_DIR = Path(__file__).parent.parent.parent / "output" / "stories"
WORLDS_DIR = Path(__file__).parent.parent.parent / "output" / "worlds"
BACKUPS_DIR = Path(__file__).parent.parent.parent / "output" / "backups"
