"""Content guidelines service for checking content appropriateness."""

import logging
import re
from dataclasses import dataclass, field

from src.memory.content_guidelines import (
    ContentProfile,
    LanguageLevel,
    RomanceLevel,
    ThemeLevel,
    ViolenceLevel,
)
from src.settings import Settings
from src.utils.exceptions import StoryFactoryError

logger = logging.getLogger(__name__)


@dataclass
class ContentViolation:
    """Represents a content guideline violation."""

    category: str  # violence, language, themes, romance, custom
    severity: str  # warning, violation
    description: str
    excerpt: str = ""  # Relevant text excerpt if available


@dataclass
class ContentCheckResult:
    """Result of a content check operation."""

    passed: bool
    violations: list[ContentViolation] = field(default_factory=list)
    warnings: list[ContentViolation] = field(default_factory=list)

    def add_violation(self, violation: ContentViolation) -> None:
        """Add a violation and mark as failed."""
        if violation.severity == "warning":
            self.warnings.append(violation)
        else:
            self.violations.append(violation)
            self.passed = False


class ContentGuidelinesService:
    """Service for checking content against guidelines.

    Provides both heuristic-based and LLM-based content checking.
    """

    # Patterns for heuristic checks
    PROFANITY_MILD = re.compile(r"\b(damn|hell|crap|bastard)\b", re.IGNORECASE)
    PROFANITY_STRONG = re.compile(r"\b(fuck|shit|ass|bitch|cock|dick|pussy|cunt)\b", re.IGNORECASE)

    VIOLENCE_MODERATE = re.compile(
        r"\b(blood|wound|stab|slash|punch|kick|fight|battle|kill)\b", re.IGNORECASE
    )
    VIOLENCE_GRAPHIC = re.compile(
        r"\b(gore|entrails|dismember|mutilate|torture|agony|splatter|eviscerate)\b",
        re.IGNORECASE,
    )

    ROMANCE_MILD = re.compile(r"\b(kiss|embrace|caress|hold|touch|desire)\b", re.IGNORECASE)
    ROMANCE_EXPLICIT = re.compile(
        r"\b(thrust|moan|naked|strip|orgasm|climax|breast|erection)\b", re.IGNORECASE
    )

    DARK_THEMES = re.compile(
        r"\b(suicide|self-harm|abuse|rape|incest|trafficking|torture)\b", re.IGNORECASE
    )

    def __init__(self, settings: Settings | None = None):
        """Initialize ContentGuidelinesService.

        Args:
            settings: Application settings. If None, loads from src/settings.json.
        """
        logger.debug("Initializing ContentGuidelinesService")
        self.settings = settings or Settings.load()
        logger.debug("ContentGuidelinesService initialized")

    def check_content(
        self,
        content: str,
        profile: ContentProfile,
        use_llm: bool = False,
    ) -> ContentCheckResult:
        """Check content against a content profile.

        Args:
            content: The text content to check.
            profile: The content profile to check against.
            use_llm: Whether to use LLM for additional checking (slower but more accurate).

        Returns:
            ContentCheckResult with any violations found.
        """
        logger.info(f"Checking content ({len(content)} chars) against profile '{profile.name}'")

        # Fail fast if use_llm is requested but not implemented
        if use_llm:
            raise StoryFactoryError(
                "LLM-based content checking is not implemented yet. "
                "Set use_llm=False or content_check_use_llm=False in settings."
            )

        result = ContentCheckResult(passed=True)

        # Run heuristic checks
        self._check_language(content, profile, result)
        self._check_violence(content, profile, result)
        self._check_romance(content, profile, result)
        self._check_themes(content, profile, result)
        self._check_custom_avoid(content, profile, result)

        if result.passed:
            logger.info(
                f"Content check passed (profile={profile.name}, warnings={len(result.warnings)})"
            )
        else:
            logger.warning(
                f"Content check failed (profile={profile.name}, "
                f"violations={len(result.violations)}, warnings={len(result.warnings)})"
            )

        return result

    def _check_language(
        self,
        content: str,
        profile: ContentProfile,
        result: ContentCheckResult,
    ) -> None:
        """Check content for language/profanity violations."""
        if profile.language == LanguageLevel.UNRESTRICTED:
            return  # No restrictions

        # Check for strong profanity
        strong_matches = self.PROFANITY_STRONG.findall(content)
        if strong_matches and profile.language in (LanguageLevel.CLEAN, LanguageLevel.MILD):
            result.add_violation(
                ContentViolation(
                    category="language",
                    severity="violation",
                    description=f"Strong profanity detected: {', '.join(set(strong_matches[:5]))}",
                    excerpt=self._get_excerpt(content, strong_matches[0]),
                )
            )

        # Check for mild profanity (only violation for CLEAN)
        if profile.language == LanguageLevel.CLEAN:
            mild_matches = self.PROFANITY_MILD.findall(content)
            if mild_matches:
                result.add_violation(
                    ContentViolation(
                        category="language",
                        severity="violation",
                        description=f"Mild profanity detected: {', '.join(set(mild_matches[:5]))}",
                        excerpt=self._get_excerpt(content, mild_matches[0]),
                    )
                )

    def _check_violence(
        self,
        content: str,
        profile: ContentProfile,
        result: ContentCheckResult,
    ) -> None:
        """Check content for violence violations."""
        if profile.violence == ViolenceLevel.GRAPHIC:
            return  # No restrictions

        # Check for graphic violence
        graphic_matches = self.VIOLENCE_GRAPHIC.findall(content)
        if graphic_matches:
            result.add_violation(
                ContentViolation(
                    category="violence",
                    severity="violation",
                    description=f"Graphic violence detected: {', '.join(set(graphic_matches[:5]))}",
                    excerpt=self._get_excerpt(content, graphic_matches[0]),
                )
            )

        # Check for moderate violence (violation for MINIMAL profile)
        if profile.violence == ViolenceLevel.MINIMAL:
            moderate_matches = self.VIOLENCE_MODERATE.findall(content)
            if moderate_matches:
                result.add_violation(
                    ContentViolation(
                        category="violence",
                        severity="violation",
                        description=f"Action violence detected: {', '.join(set(moderate_matches[:5]))}",
                        excerpt=self._get_excerpt(content, moderate_matches[0]),
                    )
                )

    def _check_romance(
        self,
        content: str,
        profile: ContentProfile,
        result: ContentCheckResult,
    ) -> None:
        """Check content for romance/intimacy violations."""
        if profile.romance == RomanceLevel.EXPLICIT:
            return  # No restrictions

        # Check for explicit content
        explicit_matches = self.ROMANCE_EXPLICIT.findall(content)
        if explicit_matches and profile.romance in (
            RomanceLevel.NONE,
            RomanceLevel.SWEET,
            RomanceLevel.SENSUAL,
        ):
            result.add_violation(
                ContentViolation(
                    category="romance",
                    severity="violation",
                    description=f"Explicit romantic content detected: {', '.join(set(explicit_matches[:5]))}",
                    excerpt=self._get_excerpt(content, explicit_matches[0]),
                )
            )

        # Check for mild romance (only issue for NONE)
        if profile.romance == RomanceLevel.NONE:
            mild_matches = self.ROMANCE_MILD.findall(content)
            if mild_matches:
                result.add_violation(
                    ContentViolation(
                        category="romance",
                        severity="warning",
                        description="Romantic elements detected in no-romance profile",
                        excerpt=self._get_excerpt(content, mild_matches[0]),
                    )
                )

    def _check_themes(
        self,
        content: str,
        profile: ContentProfile,
        result: ContentCheckResult,
    ) -> None:
        """Check content for dark theme violations."""
        if profile.themes == ThemeLevel.DARK:
            return  # No restrictions

        dark_matches = self.DARK_THEMES.findall(content)
        if dark_matches:
            severity = "violation" if profile.themes == ThemeLevel.LIGHT else "warning"
            result.add_violation(
                ContentViolation(
                    category="themes",
                    severity=severity,
                    description=f"Dark themes detected: {', '.join(set(dark_matches[:5]))}",
                    excerpt=self._get_excerpt(content, dark_matches[0]),
                )
            )

    def _check_custom_avoid(
        self,
        content: str,
        profile: ContentProfile,
        result: ContentCheckResult,
    ) -> None:
        """Check content for custom avoid items."""
        content_lower = content.lower()
        for avoid_item in profile.custom_avoid:
            if avoid_item.lower() in content_lower:
                result.add_violation(
                    ContentViolation(
                        category="custom",
                        severity="violation",
                        description=f"Custom avoid item detected: '{avoid_item}'",
                        excerpt=self._get_excerpt(content, avoid_item),
                    )
                )

    def _get_excerpt(self, content: str, term: str, context_chars: int = 50) -> str:
        """Get an excerpt of content around a term.

        Args:
            content: Full content text.
            term: Term to find and excerpt around.
            context_chars: Characters of context on each side.

        Returns:
            Excerpt string with context.
        """
        lower_content = content.lower()
        lower_term = term.lower()
        idx = lower_content.find(lower_term)
        if idx == -1:
            return ""

        start = max(0, idx - context_chars)
        end = min(len(content), idx + len(term) + context_chars)
        excerpt = content[start:end]

        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."

        return excerpt.replace("\n", " ")

    def get_profile_summary(self, profile: ContentProfile) -> str:
        """Get a human-readable summary of a content profile.

        Args:
            profile: The content profile to summarize.

        Returns:
            Summary string.
        """
        summary = (
            f"Content Profile: {profile.name}\n"
            f"  Violence: {profile.violence.value}\n"
            f"  Language: {profile.language.value}\n"
            f"  Themes: {profile.themes.value}\n"
            f"  Romance: {profile.romance.value}"
        )
        if profile.custom_avoid:
            summary += f"\n  Avoid: {', '.join(profile.custom_avoid)}"
        if profile.custom_include:
            summary += f"\n  Include: {', '.join(profile.custom_include)}"
        return summary
