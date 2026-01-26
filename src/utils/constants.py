"""Shared constants used across the application."""

import logging

logger = logging.getLogger(__name__)

# ========== Entity Type Colors ==========
# Used for graph visualization and entity cards
# NOTE: Keep in sync with src/ui/theme.py ENTITY_COLORS if modified
ENTITY_COLORS: dict[str, str] = {
    "character": "#4CAF50",  # Green
    "location": "#2196F3",  # Blue
    "item": "#FF9800",  # Orange
    "faction": "#9C27B0",  # Purple
    "concept": "#607D8B",  # Grey
    "event": "#FF5722",  # Orange for events
}


def get_entity_color(entity_type: str) -> str:
    """
    Retrieve the hex color associated with an entity type.

    Parameters:
        entity_type (str): Entity type name (case-insensitive).

    Returns:
        str: Hex color code for the given entity type; returns the color for "concept" if the type is not found.
    """
    key = entity_type.lower()
    color = ENTITY_COLORS.get(key)
    if color is None:
        logger.warning(
            f"Unknown entity type '{entity_type}' - defaulting to concept color. "
            f"Valid types: {list(ENTITY_COLORS.keys())}"
        )
        return ENTITY_COLORS["concept"]
    logger.debug(f"get_entity_color: '{entity_type}' -> {color}")
    return color


# Language name to ISO 639-1 code mapping
# Used for EPUB metadata and other internationalization needs
LANGUAGE_CODES: dict[str, str] = {
    "English": "en",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Arabic": "ar",
}


def get_language_code(language_name: str, default: str = "en") -> str:
    """Get ISO 639-1 language code from language name.

    Args:
        language_name: Full language name (e.g., "English", "German")
        default: Default code if language not found

    Returns:
        ISO 639-1 language code (e.g., "en", "de")
    """
    return LANGUAGE_CODES.get(language_name, default)
