"""Message analysis utilities for inferring user intent."""

import re


def detect_content_rating(message: str) -> str | None:
    """Detect content rating from keywords in user message.

    Args:
        message: User message text.

    Returns:
        Content rating ('adult', 'mature', 'teen', 'general') or None if unclear.
    """
    message_lower = message.lower()

    # Adult indicators (explicit content)
    adult_keywords = [
        "smut",
        "nsfw",
        "explicit",
        "erotic",
        "erotica",
        "porn",
        "xxx",
        "18+",
        "adult only",
        "adult content",
        "sex scene",
        "sexual",
        "steamy",
        "spicy",  # Common romance term for explicit
        "heat level",  # Romance term
        "lemon",  # Fanfic term for explicit
    ]
    if any(kw in message_lower for kw in adult_keywords):
        return "adult"

    # Mature indicators (adult themes but not explicit)
    mature_keywords = [
        "dark",
        "violence",
        "violent",
        "gore",
        "gory",
        "mature themes",
        "mature content",
        "dark romance",
        "grimdark",
        "horror",
        "death",
        "murder",
        "trauma",
        "abuse",
        "addiction",
        "drugs",
        "war",
    ]
    if any(kw in message_lower for kw in mature_keywords):
        return "mature"

    # Teen indicators
    teen_keywords = [
        "teen",
        "young adult",
        "ya ",
        "ya,",
        " ya",
        "coming of age",
        "high school",
        "teenager",
        "teenagers",
    ]
    if any(kw in message_lower for kw in teen_keywords):
        return "teen"

    # General/family-friendly indicators
    general_keywords = [
        "family friendly",
        "family-friendly",
        "kid",
        "kids",
        "children",
        "child friendly",
        "child-friendly",
        "all ages",
        "clean",
        "pg",
        "pg-13",
        "wholesome",
    ]
    if any(kw in message_lower for kw in general_keywords):
        return "general"

    return None


def detect_language(message: str) -> str:
    """Detect likely language from the message content.

    Uses simple heuristics to identify the language the user is writing in.

    Args:
        message: User message text.

    Returns:
        Detected language name (defaults to 'English').
    """
    message_lower = message.lower()

    # German indicators
    german_words = ["ich", "und", "der", "die", "das", "ist", "ein", "eine", "nicht", "mit"]
    german_count = sum(1 for word in german_words if re.search(rf"\b{word}\b", message_lower))
    if german_count >= 3:
        return "German"

    # Spanish indicators
    spanish_words = ["el", "la", "los", "las", "que", "de", "en", "es", "por", "con"]
    spanish_count = sum(1 for word in spanish_words if re.search(rf"\b{word}\b", message_lower))
    if spanish_count >= 3:
        return "Spanish"

    # French indicators
    french_words = ["le", "la", "les", "de", "du", "des", "est", "sont", "avec", "pour"]
    french_count = sum(1 for word in french_words if re.search(rf"\b{word}\b", message_lower))
    if french_count >= 3:
        return "French"

    # Italian indicators
    italian_words = ["il", "la", "di", "che", "non", "con", "sono", "una", "per", "come"]
    italian_count = sum(1 for word in italian_words if re.search(rf"\b{word}\b", message_lower))
    if italian_count >= 3:
        return "Italian"

    # Portuguese indicators
    portuguese_words = ["o", "a", "de", "que", "e", "do", "da", "em", "um", "uma"]
    portuguese_count = sum(
        1 for word in portuguese_words if re.search(rf"\b{word}\b", message_lower)
    )
    if portuguese_count >= 3:
        return "Portuguese"

    # Dutch indicators
    dutch_words = ["de", "het", "een", "van", "en", "is", "dat", "niet", "op", "met"]
    dutch_count = sum(1 for word in dutch_words if re.search(rf"\b{word}\b", message_lower))
    if dutch_count >= 3:
        return "Dutch"

    # Default to English (most common case)
    return "English"


def analyze_message(message: str) -> dict[str, str | None]:
    """Analyze a user message to infer content rating and language.

    Args:
        message: User message text.

    Returns:
        Dictionary with 'language' and 'content_rating' keys.
    """
    return {
        "language": detect_language(message),
        "content_rating": detect_content_rating(message),
    }


def format_inference_context(analysis: dict[str, str | None]) -> str:
    """Format analysis results as context for the interviewer.

    Args:
        analysis: Result from analyze_message().

    Returns:
        Context string to prepend to interviewer prompt.
    """
    parts = []

    if analysis.get("language"):
        parts.append(f"[INFERRED: User is writing in {analysis['language']}]")

    if analysis.get("content_rating"):
        parts.append(f"[INFERRED: Content rating is clearly '{analysis['content_rating']}']")

    if parts:
        return "\n".join(parts) + "\nDo NOT ask about these - they are already determined.\n\n"
    return ""
