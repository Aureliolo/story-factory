"""Unit tests for content guidelines models."""

from src.memory.content_guidelines import (
    ADULT,
    ALL_AGES,
    PRESET_PROFILES,
    UNRESTRICTED,
    YOUNG_ADULT,
    ContentProfile,
    LanguageLevel,
    RomanceLevel,
    ThemeLevel,
    ViolenceLevel,
    format_profile_for_prompt,
    get_preset_profile,
    list_preset_profiles,
)


class TestEnums:
    """Tests for content level enums."""

    def test_violence_level_values(self):
        """Test ViolenceLevel enum values."""
        assert ViolenceLevel.MINIMAL.value == "minimal"
        assert ViolenceLevel.MODERATE.value == "moderate"
        assert ViolenceLevel.GRAPHIC.value == "graphic"

    def test_language_level_values(self):
        """Test LanguageLevel enum values."""
        assert LanguageLevel.CLEAN.value == "clean"
        assert LanguageLevel.MILD.value == "mild"
        assert LanguageLevel.UNRESTRICTED.value == "unrestricted"

    def test_theme_level_values(self):
        """Test ThemeLevel enum values."""
        assert ThemeLevel.LIGHT.value == "light"
        assert ThemeLevel.MATURE.value == "mature"
        assert ThemeLevel.DARK.value == "dark"

    def test_romance_level_values(self):
        """Test RomanceLevel enum values."""
        assert RomanceLevel.NONE.value == "none"
        assert RomanceLevel.SWEET.value == "sweet"
        assert RomanceLevel.SENSUAL.value == "sensual"
        assert RomanceLevel.EXPLICIT.value == "explicit"


class TestContentProfile:
    """Tests for ContentProfile model."""

    def test_default_profile(self):
        """Test creating a profile with defaults."""
        profile = ContentProfile()
        assert profile.name == "custom"
        assert profile.violence == ViolenceLevel.MODERATE
        assert profile.language == LanguageLevel.MILD
        assert profile.themes == ThemeLevel.MATURE
        assert profile.romance == RomanceLevel.SWEET
        assert profile.custom_avoid == []
        assert profile.custom_include == []

    def test_custom_profile(self):
        """Test creating a fully custom profile."""
        profile = ContentProfile(
            name="my_profile",
            violence=ViolenceLevel.MINIMAL,
            language=LanguageLevel.CLEAN,
            themes=ThemeLevel.LIGHT,
            romance=RomanceLevel.NONE,
            custom_avoid=["gore", "drugs"],
            custom_include=["friendship", "adventure"],
        )
        assert profile.name == "my_profile"
        assert profile.violence == ViolenceLevel.MINIMAL
        assert "gore" in profile.custom_avoid
        assert "friendship" in profile.custom_include


class TestPresetProfiles:
    """Tests for preset content profiles."""

    def test_all_ages_profile(self):
        """Test ALL_AGES preset is appropriately restrictive."""
        assert ALL_AGES.violence == ViolenceLevel.MINIMAL
        assert ALL_AGES.language == LanguageLevel.CLEAN
        assert ALL_AGES.themes == ThemeLevel.LIGHT
        assert ALL_AGES.romance == RomanceLevel.NONE
        assert len(ALL_AGES.custom_avoid) > 0

    def test_young_adult_profile(self):
        """Test YOUNG_ADULT preset has moderate restrictions."""
        assert YOUNG_ADULT.violence == ViolenceLevel.MODERATE
        assert YOUNG_ADULT.language == LanguageLevel.MILD
        assert YOUNG_ADULT.themes == ThemeLevel.MATURE
        assert YOUNG_ADULT.romance == RomanceLevel.SWEET

    def test_adult_profile(self):
        """Test ADULT preset allows mature content."""
        assert ADULT.violence == ViolenceLevel.GRAPHIC
        assert ADULT.language == LanguageLevel.UNRESTRICTED
        assert ADULT.themes == ThemeLevel.DARK
        assert ADULT.romance == RomanceLevel.SENSUAL

    def test_unrestricted_profile(self):
        """Test UNRESTRICTED preset has no limits."""
        assert UNRESTRICTED.violence == ViolenceLevel.GRAPHIC
        assert UNRESTRICTED.language == LanguageLevel.UNRESTRICTED
        assert UNRESTRICTED.themes == ThemeLevel.DARK
        assert UNRESTRICTED.romance == RomanceLevel.EXPLICIT
        assert len(UNRESTRICTED.custom_avoid) == 0

    def test_preset_profiles_dict(self):
        """Test that all presets are in the registry."""
        assert "all_ages" in PRESET_PROFILES
        assert "young_adult" in PRESET_PROFILES
        assert "adult" in PRESET_PROFILES
        assert "unrestricted" in PRESET_PROFILES


class TestGetPresetProfile:
    """Tests for get_preset_profile function."""

    def test_get_existing_profile(self):
        """Test getting an existing preset profile."""
        profile = get_preset_profile("all_ages")
        assert profile is not None
        assert profile.name == "all_ages"

    def test_get_nonexistent_profile(self):
        """Test getting a nonexistent profile returns None."""
        profile = get_preset_profile("nonexistent")
        assert profile is None

    def test_get_all_presets(self):
        """Test getting all preset profiles."""
        for name in PRESET_PROFILES:
            profile = get_preset_profile(name)
            assert profile is not None
            assert profile.name == name


class TestListPresetProfiles:
    """Tests for list_preset_profiles function."""

    def test_list_returns_all_presets(self):
        """Test that list returns all preset profiles."""
        profiles = list_preset_profiles()
        assert len(profiles) == len(PRESET_PROFILES)

    def test_list_contains_expected_profiles(self):
        """Test that list contains expected profiles."""
        profiles = list_preset_profiles()
        names = {p.name for p in profiles}
        assert "all_ages" in names
        assert "young_adult" in names
        assert "adult" in names
        assert "unrestricted" in names


class TestFormatProfileForPrompt:
    """Tests for format_profile_for_prompt function."""

    def test_format_includes_profile_name(self):
        """Test that formatted output includes profile name."""
        profile = get_preset_profile("young_adult")
        assert profile is not None
        formatted = format_profile_for_prompt(profile)
        assert "young_adult" in formatted

    def test_format_includes_all_levels(self):
        """Test that formatted output includes all content levels."""
        profile = get_preset_profile("adult")
        assert profile is not None
        formatted = format_profile_for_prompt(profile)

        assert "Violence:" in formatted
        assert "Language:" in formatted
        assert "Themes:" in formatted
        assert "Romance:" in formatted

    def test_format_includes_custom_avoid(self):
        """Test that custom avoid items are included."""
        profile = get_preset_profile("all_ages")
        assert profile is not None
        formatted = format_profile_for_prompt(profile)

        assert "MUST AVOID" in formatted
        for avoid_item in profile.custom_avoid[:3]:  # Check at least some items
            assert avoid_item in formatted

    def test_format_includes_custom_include(self):
        """Test that custom include items are included."""
        profile = ContentProfile(
            name="test",
            custom_include=["adventure", "humor"],
        )
        formatted = format_profile_for_prompt(profile)

        assert "INCLUDE/ALLOW" in formatted
        assert "adventure" in formatted
        assert "humor" in formatted

    def test_format_empty_custom_lists(self):
        """Test formatting when custom lists are empty."""
        profile = get_preset_profile("unrestricted")
        assert profile is not None
        formatted = format_profile_for_prompt(profile)

        # Should not include MUST AVOID section if list is empty
        if not profile.custom_avoid:
            assert "MUST AVOID" not in formatted


class TestProfileProgressions:
    """Test that profile levels form logical progressions."""

    def test_violence_progression(self):
        """Test that violence levels progress from minimal to graphic."""
        profiles = [ALL_AGES, YOUNG_ADULT, ADULT]
        violence_order = [ViolenceLevel.MINIMAL, ViolenceLevel.MODERATE, ViolenceLevel.GRAPHIC]

        for i, profile in enumerate(profiles):
            expected = violence_order[min(i, len(violence_order) - 1)]
            assert profile.violence == expected or profile.violence.value >= expected.value

    def test_language_progression(self):
        """Test that language levels progress appropriately."""
        assert ALL_AGES.language == LanguageLevel.CLEAN
        assert YOUNG_ADULT.language == LanguageLevel.MILD
        assert ADULT.language == LanguageLevel.UNRESTRICTED

    def test_all_ages_most_restrictive(self):
        """Test that ALL_AGES is the most restrictive profile."""
        assert ALL_AGES.violence == ViolenceLevel.MINIMAL
        assert ALL_AGES.language == LanguageLevel.CLEAN
        assert ALL_AGES.themes == ThemeLevel.LIGHT
        assert ALL_AGES.romance == RomanceLevel.NONE

    def test_unrestricted_least_restrictive(self):
        """Test that UNRESTRICTED is the least restrictive profile."""
        assert UNRESTRICTED.violence == ViolenceLevel.GRAPHIC
        assert UNRESTRICTED.language == LanguageLevel.UNRESTRICTED
        assert UNRESTRICTED.themes == ThemeLevel.DARK
        assert UNRESTRICTED.romance == RomanceLevel.EXPLICIT
