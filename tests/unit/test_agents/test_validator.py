"""Tests for ValidatorAgent (rule-based, no AI)."""

import pytest

from src.agents.validator import ValidatorAgent, validate_or_raise
from src.settings import Settings
from src.utils.exceptions import ResponseValidationError


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def validator(settings):
    """Create ValidatorAgent with test settings."""
    return ValidatorAgent(settings=settings)


class TestValidatorAgentInit:
    """Tests for ValidatorAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        agent = ValidatorAgent(settings=settings)
        assert isinstance(agent.settings, Settings)

    def test_init_with_custom_settings(self, settings):
        """Test agent initializes with provided settings."""
        agent = ValidatorAgent(settings=settings)
        assert agent.settings is settings


class TestValidatorValidateResponse:
    """Tests for validate_response method."""

    def test_accepts_valid_english_response(self, validator):
        """Test accepts valid English response."""
        response = """The sun set over the mountains, casting long shadows across the valley.
Sarah watched from her window, thinking about the journey ahead."""

        result = validator.validate_response(response, "English")

        assert result is True

    def test_rejects_empty_response(self, validator):
        """Test rejects empty response."""
        with pytest.raises((ResponseValidationError, ValueError), match=r"(Empty|cannot be empty)"):
            validator.validate_response("", "English")

    def test_rejects_whitespace_only_response(self, validator):
        """Test rejects whitespace-only response."""
        with pytest.raises((ResponseValidationError, ValueError), match=r"(Empty|cannot be empty)"):
            validator.validate_response("   \n\t  ", "English")

    def test_rejects_cjk_when_english_expected(self, validator):
        """Test rejects CJK characters when English expected."""
        # Response with many Chinese characters
        response = "这是一个中文测试文本，包含很多汉字。"

        with pytest.raises(ResponseValidationError, match="CJK"):
            validator.validate_response(response, "English")

    def test_allows_few_cjk_characters(self, validator):
        """Test allows a few CJK characters (for names/terms)."""
        response = """The ancient scroll bore the name 山本 in golden characters.
The rest of the story was in English and told of great adventures."""

        # Should not raise - only 2 CJK characters
        result = validator.validate_response(response, "English")
        assert result is True

    def test_rejects_non_printable_characters(self, validator):
        """Test rejects responses with too many non-printable characters."""
        # Response with many control characters
        response = "Normal text" + "\x00\x01\x02\x03\x04\x05\x06\x07\x08" * 20

        with pytest.raises(ResponseValidationError, match="non-printable"):
            validator.validate_response(response)

    def test_no_ai_validation_for_long_responses(self, validator):
        """Test no AI validation even for long responses (rule-based only)."""
        long_response = "This is a valid English response. " * 20

        result = validator.validate_response(long_response, "English", "Story content")

        assert result is True


class TestValidateOrRaise:
    """Tests for validate_or_raise convenience function."""

    def test_returns_response_when_valid(self, validator):
        """Test returns original response when valid."""
        response = "This is valid English content."

        result = validate_or_raise(response, "English", validator=validator)

        assert result == response

    def test_raises_for_invalid_response(self, validator):
        """Test raises error for invalid response."""
        with pytest.raises((ResponseValidationError, ValueError)):
            validate_or_raise("", "English", validator=validator)

    def test_accepts_custom_validator(self, validator):
        """Test accepts pre-created validator agent."""
        response = "Valid response."

        result = validate_or_raise(response, "English", validator=validator)

        assert result == response

    def test_creates_validator_when_not_provided(self, settings):
        """Test creates new validator when not provided."""
        response = "Valid response."

        result = validate_or_raise(response, "English", validator=ValidatorAgent(settings=settings))

        assert result == response


class TestValidatorLanguageChecks:
    """Tests for language-specific validation."""

    def test_accepts_german_when_german_expected(self, validator):
        """Test accepts German text when German is expected."""
        response = """Die Sonne ging unter und tauchte die Berge in goldenes Licht.
Maria schaute aus dem Fenster und dachte an die Reise, die vor ihr lag."""

        result = validator.validate_response(response, "German")

        assert result is True

    def test_cjk_check_only_for_english(self, validator):
        """Test CJK character check only applies to English."""
        response = "这是一个中文测试文本，包含很多汉字。"

        # For Chinese expected language, CJK characters are fine
        result = validator.validate_response(response, "Chinese")

        assert result is True


class TestValidatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_unicode_correctly(self, validator):
        """Test handles various unicode correctly."""
        response = 'Café résumé naïve — "quotes" and emojis'

        result = validator.validate_response(response, "English")

        assert result is True

    def test_handles_mixed_content(self, validator):
        """Test handles code blocks and special formatting."""
        response = """Here's an example:
```python
def hello():
    print("Hello, world!")
```
This is valid content."""

        result = validator.validate_response(response, "English")

        assert result is True
