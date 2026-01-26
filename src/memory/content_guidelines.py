"""Content guidelines for story generation.

Defines content profiles that control the appropriateness levels
for violence, language, themes, and romance in generated content.
"""

import logging
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ViolenceLevel(str, Enum):
    """Level of violence allowed in content."""

    MINIMAL = "minimal"  # No graphic violence, conflicts resolved peacefully
    MODERATE = "moderate"  # Action violence, consequences not dwelt upon
    GRAPHIC = "graphic"  # Detailed violence, injuries described


class LanguageLevel(str, Enum):
    """Level of language/profanity allowed in content."""

    CLEAN = "clean"  # No profanity or crude language
    MILD = "mild"  # Mild profanity (damn, hell), no slurs
    UNRESTRICTED = "unrestricted"  # Realistic language including strong profanity


class ThemeLevel(str, Enum):
    """Level of mature themes allowed in content."""

    LIGHT = "light"  # Family-friendly themes, no darkness
    MATURE = "mature"  # Complex themes (loss, betrayal, moral ambiguity)
    DARK = "dark"  # Heavy themes (trauma, abuse, death) explored in depth


class RomanceLevel(str, Enum):
    """Level of romantic/sexual content allowed."""

    NONE = "none"  # No romantic content
    SWEET = "sweet"  # Chaste romance, hand-holding, closed-door
    SENSUAL = "sensual"  # Romantic tension, implied intimacy
    EXPLICIT = "explicit"  # Detailed romantic scenes


class ContentProfile(BaseModel):
    """Content profile defining what is appropriate for a story.

    Used to guide generation and validate output content.
    """

    name: str = Field(default="custom", description="Profile name")
    violence: ViolenceLevel = Field(default=ViolenceLevel.MODERATE, description="Violence level")
    language: LanguageLevel = Field(
        default=LanguageLevel.MILD, description="Language/profanity level"
    )
    themes: ThemeLevel = Field(default=ThemeLevel.MATURE, description="Mature themes level")
    romance: RomanceLevel = Field(default=RomanceLevel.SWEET, description="Romance/intimacy level")
    custom_avoid: list[str] = Field(
        default_factory=list, description="Specific topics/elements to avoid"
    )
    custom_include: list[str] = Field(
        default_factory=list, description="Specific topics/elements to include"
    )


# =============================================================================
# PRESET PROFILES
# =============================================================================

ALL_AGES = ContentProfile(
    name="all_ages",
    violence=ViolenceLevel.MINIMAL,
    language=LanguageLevel.CLEAN,
    themes=ThemeLevel.LIGHT,
    romance=RomanceLevel.NONE,
    custom_avoid=[
        "death of children",
        "animal cruelty",
        "substance abuse",
        "self-harm",
        "graphic injuries",
    ],
    custom_include=[],
)

YOUNG_ADULT = ContentProfile(
    name="young_adult",
    violence=ViolenceLevel.MODERATE,
    language=LanguageLevel.MILD,
    themes=ThemeLevel.MATURE,
    romance=RomanceLevel.SWEET,
    custom_avoid=[
        "explicit sexual content",
        "detailed torture",
        "gratuitous violence",
        "substance glorification",
    ],
    custom_include=[],
)

ADULT = ContentProfile(
    name="adult",
    violence=ViolenceLevel.GRAPHIC,
    language=LanguageLevel.UNRESTRICTED,
    themes=ThemeLevel.DARK,
    romance=RomanceLevel.SENSUAL,
    custom_avoid=[],
    custom_include=[],
)

UNRESTRICTED = ContentProfile(
    name="unrestricted",
    violence=ViolenceLevel.GRAPHIC,
    language=LanguageLevel.UNRESTRICTED,
    themes=ThemeLevel.DARK,
    romance=RomanceLevel.EXPLICIT,
    custom_avoid=[],
    custom_include=[],
)


# Preset profiles registry
PRESET_PROFILES: dict[str, ContentProfile] = {
    "all_ages": ALL_AGES,
    "young_adult": YOUNG_ADULT,
    "adult": ADULT,
    "unrestricted": UNRESTRICTED,
}


def get_preset_profile(name: str) -> ContentProfile | None:
    """Get a preset content profile by name.

    Args:
        name: Profile name (all_ages, young_adult, adult, unrestricted).

    Returns:
        The preset profile or None if not found.
    """
    profile = PRESET_PROFILES.get(name)
    if profile:
        logger.debug(f"Retrieved preset profile: {name}")
    else:
        logger.warning(f"Preset profile not found: {name}")
    return profile


def list_preset_profiles() -> list[ContentProfile]:
    """List all preset content profiles.

    Returns:
        List of all preset profiles.
    """
    logger.debug(f"Listed {len(PRESET_PROFILES)} preset profiles")
    return list(PRESET_PROFILES.values())


def format_profile_for_prompt(profile: ContentProfile) -> str:
    """Format a content profile into prompt guidance text.

    Args:
        profile: The content profile to format.

    Returns:
        Formatted string for prompt injection.
    """
    lines = [
        "=== CONTENT GUIDELINES ===",
        f"Profile: {profile.name}",
        "",
        "Content Parameters:",
        f"  Violence: {profile.violence.value} - {_get_violence_description(profile.violence)}",
        f"  Language: {profile.language.value} - {_get_language_description(profile.language)}",
        f"  Themes: {profile.themes.value} - {_get_themes_description(profile.themes)}",
        f"  Romance: {profile.romance.value} - {_get_romance_description(profile.romance)}",
    ]

    if profile.custom_avoid:
        lines.append("")
        lines.append("MUST AVOID:")
        for item in profile.custom_avoid:
            lines.append(f"  - {item}")

    if profile.custom_include:
        lines.append("")
        lines.append("INCLUDE/ALLOW:")
        for item in profile.custom_include:
            lines.append(f"  - {item}")

    guidance = "\n".join(lines)
    logger.debug(f"Formatted content profile guidance ({len(guidance)} chars)")
    return guidance


def _get_violence_description(level: ViolenceLevel) -> str:
    """Get human-readable description for violence level."""
    descriptions = {
        ViolenceLevel.MINIMAL: "No graphic violence. Conflicts should be resolved without detailed harm.",
        ViolenceLevel.MODERATE: "Action violence is acceptable. Don't dwell on injuries or suffering.",
        ViolenceLevel.GRAPHIC: "Violence can be detailed when appropriate to the story.",
    }
    return descriptions[level]


def _get_language_description(level: LanguageLevel) -> str:
    """Get human-readable description for language level."""
    descriptions = {
        LanguageLevel.CLEAN: "No profanity or crude language. Keep dialogue family-friendly.",
        LanguageLevel.MILD: "Mild profanity acceptable (damn, hell). No slurs or crude terms.",
        LanguageLevel.UNRESTRICTED: "Realistic language including strong profanity when appropriate.",
    }
    return descriptions[level]


def _get_themes_description(level: ThemeLevel) -> str:
    """Get human-readable description for themes level."""
    descriptions = {
        ThemeLevel.LIGHT: "Light themes only. Avoid darkness, trauma, or moral complexity.",
        ThemeLevel.MATURE: "Complex themes acceptable (loss, betrayal, moral ambiguity).",
        ThemeLevel.DARK: "Dark themes can be explored in depth (trauma, death, existential threats).",
    }
    return descriptions[level]


def _get_romance_description(level: RomanceLevel) -> str:
    """Get human-readable description for romance level."""
    descriptions = {
        RomanceLevel.NONE: "No romantic content. Focus on other relationship types.",
        RomanceLevel.SWEET: "Chaste romance only. Affection shown through words and gestures.",
        RomanceLevel.SENSUAL: "Romantic tension and implied intimacy. Fade to black.",
        RomanceLevel.EXPLICIT: "Detailed romantic scenes may be included.",
    }
    return descriptions[level]
