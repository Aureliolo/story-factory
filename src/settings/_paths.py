"""Path constants for Story Factory settings and output directories."""

import logging
import subprocess
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# Cross-platform subprocess flags (CREATE_NO_WINDOW only exists on Windows)
_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

# Centralized paths for story and world output files
# Go up from src/settings to src/, then up to project root, then into output/
STORIES_DIR = Path(__file__).parent.parent.parent / "output" / "stories"
WORLDS_DIR = Path(__file__).parent.parent.parent / "output" / "worlds"
BACKUPS_DIR = Path(__file__).parent.parent.parent / "output" / "backups"

__all__ = [
    "BACKUPS_DIR",
    "SETTINGS_FILE",
    "STORIES_DIR",
    "WORLDS_DIR",
    "_SUBPROCESS_FLAGS",
    "logger",
]
