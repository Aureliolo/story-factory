"""Built-in story templates and structure presets loaded from YAML files.

This module provides backward-compatible exports for:
- BUILTIN_STRUCTURE_PRESETS: dict[str, StructurePreset]
- BUILTIN_STORY_TEMPLATES: dict[str, StoryTemplate]

Templates are loaded from YAML files in the structures/ and stories/ subdirectories.
"""

from src.memory.builtin_templates._registry import (
    TemplateRegistry,
    TemplateRegistryError,
    get_builtin_story_templates,
    get_builtin_structure_presets,
)

# Backward-compatible exports - eagerly loaded at import time
BUILTIN_STRUCTURE_PRESETS = get_builtin_structure_presets()
BUILTIN_STORY_TEMPLATES = get_builtin_story_templates()

__all__ = [
    "BUILTIN_STORY_TEMPLATES",
    "BUILTIN_STRUCTURE_PRESETS",
    "TemplateRegistry",
    "TemplateRegistryError",
    "get_builtin_story_templates",
    "get_builtin_structure_presets",
]
