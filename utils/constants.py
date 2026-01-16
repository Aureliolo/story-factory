"""Shared constants used across the application."""

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
