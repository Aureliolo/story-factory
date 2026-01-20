"""Prompt templates package.

This package contains YAML-based prompt templates for all agents and services.
Templates are organized by agent role under templates/.

Directory structure:
    prompts/
    ├── __init__.py
    └── templates/
        ├── writer/
        │   ├── system.yaml
        │   ├── write_chapter.yaml
        │   └── ...
        ├── editor/
        ├── interviewer/
        ├── architect/
        ├── continuity/
        ├── validator/
        ├── suggestion/
        └── world_quality/
            ├── character/
            ├── location/
            ├── faction/
            ├── item/
            ├── concept/
            ├── relationship/
            └── shared/
"""

from pathlib import Path

# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
