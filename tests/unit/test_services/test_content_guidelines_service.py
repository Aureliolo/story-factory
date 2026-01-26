"""Unit tests for ContentGuidelinesService."""

import pytest

from src.memory.content_guidelines import (
    ContentProfile,
    LanguageLevel,
    RomanceLevel,
    ThemeLevel,
    ViolenceLevel,
    get_preset_profile,
)
from src.services.content_guidelines_service import (
    ContentCheckResult,
    ContentGuidelinesService,
    ContentViolation,
)


@pytest.fixture
def service():
    """Create a ContentGuidelinesService instance."""
    return ContentGuidelinesService()


@pytest.fixture
def all_ages_profile():
    """Get the ALL_AGES preset profile."""
    return get_preset_profile("all_ages")


@pytest.fixture
def adult_profile():
    """Get the ADULT preset profile."""
    return get_preset_profile("adult")


@pytest.fixture
def unrestricted_profile():
    """Get the UNRESTRICTED preset profile."""
    return get_preset_profile("unrestricted")


class TestContentViolation:
    """Tests for ContentViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a content violation."""
        violation = ContentViolation(
            category="violence",
            severity="violation",
            description="Graphic violence detected",
            excerpt="...blood splattered...",
        )
        assert violation.category == "violence"
        assert violation.severity == "violation"
        assert "blood" in violation.excerpt


class TestContentCheckResult:
    """Tests for ContentCheckResult dataclass."""

    def test_result_default_passed(self):
        """Test that result defaults to passed."""
        result = ContentCheckResult(passed=True)
        assert result.passed is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0

    def test_add_violation_marks_failed(self):
        """Test that adding a violation marks the result as failed."""
        result = ContentCheckResult(passed=True)
        result.add_violation(
            ContentViolation(
                category="language",
                severity="violation",
                description="Strong profanity",
            )
        )
        assert result.passed is False
        assert len(result.violations) == 1

    def test_add_warning_does_not_fail(self):
        """Test that adding a warning does not mark as failed."""
        result = ContentCheckResult(passed=True)
        result.add_violation(
            ContentViolation(
                category="violence",
                severity="warning",
                description="Action violence",
            )
        )
        assert result.passed is True
        assert len(result.warnings) == 1


class TestLanguageChecking:
    """Tests for language/profanity checking."""

    def test_clean_content_passes(self, service, all_ages_profile):
        """Test that clean content passes all_ages check."""
        content = "The children played happily in the garden."
        result = service.check_content(content, all_ages_profile)
        assert result.passed is True

    def test_mild_profanity_fails_clean(self, service):
        """Test that mild profanity fails CLEAN language level."""
        profile = ContentProfile(language=LanguageLevel.CLEAN)
        content = "What the hell is going on here?"
        result = service.check_content(content, profile)
        assert result.passed is False
        assert any(v.category == "language" for v in result.violations)

    def test_mild_profanity_passes_mild(self, service):
        """Test that mild profanity passes MILD language level."""
        profile = ContentProfile(language=LanguageLevel.MILD)
        content = "What the hell is going on here?"
        result = service.check_content(content, profile)
        # Should pass - mild profanity is allowed
        language_violations = [v for v in result.violations if v.category == "language"]
        assert len(language_violations) == 0

    def test_strong_profanity_fails_mild(self, service):
        """Test that strong profanity fails MILD language level."""
        profile = ContentProfile(language=LanguageLevel.MILD)
        content = "What the fuck is happening?"
        result = service.check_content(content, profile)
        assert result.passed is False
        assert any(v.category == "language" for v in result.violations)

    def test_strong_profanity_passes_unrestricted(self, service, unrestricted_profile):
        """Test that strong profanity passes UNRESTRICTED level."""
        content = "Shit, that was fucking crazy!"
        result = service.check_content(content, unrestricted_profile)
        language_violations = [v for v in result.violations if v.category == "language"]
        assert len(language_violations) == 0


class TestViolenceChecking:
    """Tests for violence checking."""

    def test_peaceful_content_passes(self, service, all_ages_profile):
        """Test that peaceful content passes minimal violence check."""
        content = "They shared a meal and talked about their dreams."
        result = service.check_content(content, all_ages_profile)
        violence_issues = [
            v for v in result.violations + result.warnings if v.category == "violence"
        ]
        assert len(violence_issues) == 0

    def test_graphic_violence_fails_moderate(self, service):
        """Test that graphic violence fails MODERATE level."""
        profile = ContentProfile(violence=ViolenceLevel.MODERATE)
        content = "The sword eviscerated the soldier, gore spilling everywhere."
        result = service.check_content(content, profile)
        assert result.passed is False
        assert any(v.category == "violence" for v in result.violations)

    def test_action_violence_warning_minimal(self, service):
        """Test that action violence creates warning for MINIMAL level."""
        profile = ContentProfile(violence=ViolenceLevel.MINIMAL)
        content = "The hero punched the villain and won the fight."
        result = service.check_content(content, profile)
        # Should be a warning, not a hard violation for action words
        assert any(v.category == "violence" for v in result.warnings)

    def test_graphic_violence_passes_graphic(self, service):
        """Test that graphic violence passes GRAPHIC level."""
        profile = ContentProfile(violence=ViolenceLevel.GRAPHIC)
        content = "Blood splattered as the torture continued."
        result = service.check_content(content, profile)
        violence_violations = [v for v in result.violations if v.category == "violence"]
        assert len(violence_violations) == 0


class TestRomanceChecking:
    """Tests for romance/intimacy checking."""

    def test_no_romance_content_passes(self, service, all_ages_profile):
        """Test that content without romance passes NONE level."""
        content = "They worked together to solve the mystery."
        result = service.check_content(content, all_ages_profile)
        romance_issues = [v for v in result.violations + result.warnings if v.category == "romance"]
        assert len(romance_issues) == 0

    def test_sweet_romance_warning_none(self, service):
        """Test that romantic elements create warning for NONE level."""
        profile = ContentProfile(romance=RomanceLevel.NONE)
        content = "She embraced him warmly as desire filled her heart."
        result = service.check_content(content, profile)
        assert any(v.category == "romance" for v in result.warnings)

    def test_explicit_content_fails_sweet(self, service):
        """Test that explicit content fails SWEET level."""
        profile = ContentProfile(romance=RomanceLevel.SWEET)
        content = "He thrust against her as they both reached climax."
        result = service.check_content(content, profile)
        assert result.passed is False
        assert any(v.category == "romance" for v in result.violations)

    def test_explicit_content_passes_explicit(self, service):
        """Test that explicit content passes EXPLICIT level."""
        profile = ContentProfile(romance=RomanceLevel.EXPLICIT)
        content = "Their naked bodies intertwined as desire took over."
        result = service.check_content(content, profile)
        romance_violations = [v for v in result.violations if v.category == "romance"]
        assert len(romance_violations) == 0


class TestThemeChecking:
    """Tests for mature theme checking."""

    def test_light_themes_pass(self, service, all_ages_profile):
        """Test that light themes pass LIGHT level."""
        content = "The friends celebrated their victory with a party."
        result = service.check_content(content, all_ages_profile)
        theme_issues = [v for v in result.violations + result.warnings if v.category == "themes"]
        assert len(theme_issues) == 0

    def test_dark_themes_fail_light(self, service):
        """Test that dark themes fail LIGHT level."""
        profile = ContentProfile(themes=ThemeLevel.LIGHT)
        content = "He contemplated suicide after the abuse he suffered."
        result = service.check_content(content, profile)
        assert result.passed is False
        assert any(v.category == "themes" for v in result.violations)

    def test_dark_themes_warning_mature(self, service):
        """Test that dark themes create warning for MATURE level."""
        profile = ContentProfile(themes=ThemeLevel.MATURE)
        content = "The story explored themes of abuse and trauma."
        result = service.check_content(content, profile)
        # Should be a warning for mature, not a hard violation
        assert any(v.category == "themes" for v in result.warnings)

    def test_dark_themes_pass_dark(self, service):
        """Test that dark themes pass DARK level."""
        profile = ContentProfile(themes=ThemeLevel.DARK)
        content = "The narrative explored suicide and self-harm in depth."
        result = service.check_content(content, profile)
        theme_violations = [v for v in result.violations if v.category == "themes"]
        assert len(theme_violations) == 0


class TestCustomAvoid:
    """Tests for custom avoid checking."""

    def test_custom_avoid_detected(self, service):
        """Test that custom avoid items are detected."""
        profile = ContentProfile(custom_avoid=["dragons", "magic"])
        content = "The wizard used magic to summon a dragon."
        result = service.check_content(content, profile)
        assert result.passed is False
        custom_violations = [v for v in result.violations if v.category == "custom"]
        assert len(custom_violations) >= 1

    def test_custom_avoid_case_insensitive(self, service):
        """Test that custom avoid is case insensitive."""
        profile = ContentProfile(custom_avoid=["dragons"])
        content = "The DRAGONS flew overhead."
        result = service.check_content(content, profile)
        assert result.passed is False

    def test_content_without_avoid_passes(self, service):
        """Test that content without avoid items passes."""
        profile = ContentProfile(custom_avoid=["dragons", "magic"])
        content = "The scientist studied the stars."
        result = service.check_content(content, profile)
        custom_violations = [v for v in result.violations if v.category == "custom"]
        assert len(custom_violations) == 0


class TestExcerptExtraction:
    """Tests for excerpt extraction."""

    def test_excerpt_includes_context(self, service):
        """Test that excerpts include surrounding context."""
        profile = ContentProfile(language=LanguageLevel.CLEAN)
        content = "He walked slowly. Then he said 'damn it' under his breath. He continued walking."
        result = service.check_content(content, profile)

        violations_with_excerpts = [v for v in result.violations if v.excerpt]
        assert len(violations_with_excerpts) > 0
        excerpt = violations_with_excerpts[0].excerpt
        assert "damn" in excerpt.lower()

    def test_excerpt_truncates_long_content(self, service):
        """Test that excerpts are truncated for long content."""
        profile = ContentProfile(language=LanguageLevel.CLEAN)
        content = "A" * 100 + " damn " + "B" * 100
        result = service.check_content(content, profile)

        violations_with_excerpts = [v for v in result.violations if v.excerpt]
        if violations_with_excerpts:
            excerpt = violations_with_excerpts[0].excerpt
            # Excerpt should be much shorter than full content
            assert len(excerpt) < len(content)


class TestProfileSummary:
    """Tests for get_profile_summary method."""

    def test_summary_includes_all_levels(self, service, adult_profile):
        """Test that summary includes all content levels."""
        summary = service.get_profile_summary(adult_profile)

        assert "Violence:" in summary
        assert "Language:" in summary
        assert "Themes:" in summary
        assert "Romance:" in summary

    def test_summary_includes_custom_lists(self, service):
        """Test that summary includes custom avoid/include lists."""
        profile = ContentProfile(
            name="test",
            custom_avoid=["item1", "item2"],
            custom_include=["theme1"],
        )
        summary = service.get_profile_summary(profile)

        assert "Avoid:" in summary
        assert "item1" in summary
        assert "Include:" in summary
        assert "theme1" in summary


class TestMultipleViolations:
    """Tests for content with multiple violations."""

    def test_multiple_category_violations(self, service, all_ages_profile):
        """Test content that violates multiple categories."""
        content = (
            "He said 'fuck you' as the torture continued with gore everywhere. "
            "Later he committed suicide after the abuse."
        )
        result = service.check_content(content, all_ages_profile)

        assert result.passed is False
        # Count violations and warnings together for category diversity
        all_issues = result.violations + result.warnings
        categories = {v.category for v in all_issues}
        # Should have issues in multiple categories (language, violence, themes)
        assert len(categories) >= 2

    def test_clean_content_all_profiles(self, service):
        """Test that truly clean content passes all profiles."""
        content = "The children played with their toys in the sunny garden."

        for profile_name in ["all_ages", "young_adult", "adult", "unrestricted"]:
            profile = get_preset_profile(profile_name)
            result = service.check_content(content, profile)
            assert result.passed is True, f"Clean content should pass {profile_name}"


class TestGetExcerpt:
    """Tests for _get_excerpt helper method."""

    def test_get_excerpt_term_not_found(self, service):
        """Test _get_excerpt returns empty string when term is not found."""
        content = "This is some sample content."
        result = service._get_excerpt(content, "nonexistent_term_xyz")
        assert result == ""

    def test_get_excerpt_term_found(self, service):
        """Test _get_excerpt returns excerpt when term is found."""
        content = "The quick brown fox jumps over the lazy dog."
        result = service._get_excerpt(content, "fox")
        assert "fox" in result
        assert len(result) > 0
