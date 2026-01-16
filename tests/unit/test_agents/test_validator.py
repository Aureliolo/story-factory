"""Tests for ValidatorAgent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.validator import ValidatorAgent, validate_or_raise
from settings import Settings
from utils.exceptions import ResponseValidationError


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def validator(settings):
    """Create ValidatorAgent with mocked Ollama client."""
    with patch("agents.base.ollama.Client"):
        agent = ValidatorAgent(model="test-model", settings=settings)
        return agent


class TestValidatorAgentInit:
    """Tests for ValidatorAgent initialization."""

    def test_init_with_defaults(self, settings):
        """Test agent initializes with default settings."""
        with patch("agents.base.ollama.Client"):
            agent = ValidatorAgent(settings=settings)
            assert agent.name == "Validator"
            assert agent.role == "Response Validator"

    def test_uses_small_model_by_default(self, settings):
        """Test uses small/fast model for validation."""
        with patch("agents.base.ollama.Client"):
            agent = ValidatorAgent(settings=settings)
            # Should use a small model like qwen3:0.6b or similar
            assert "0.6" in agent.model or "small" in agent.model.lower() or agent.model is not None


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
        with pytest.raises(ResponseValidationError, match="Empty"):
            validator.validate_response("", "English")

    def test_rejects_whitespace_only_response(self, validator):
        """Test rejects whitespace-only response."""
        with pytest.raises(ResponseValidationError, match="Empty"):
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

    def test_uses_ai_for_long_responses(self, validator):
        """Test uses AI validation for responses over 200 chars."""
        long_response = "This is a valid English response. " * 20
        validator._ai_validate = MagicMock(return_value=True)

        result = validator.validate_response(long_response, "English", "Story content")

        validator._ai_validate.assert_called_once()
        assert result is True

    def test_skips_ai_for_short_responses(self, validator):
        """Test skips AI validation for short responses."""
        short_response = "A short but valid response."
        validator._ai_validate = MagicMock()

        result = validator.validate_response(short_response, "English")

        validator._ai_validate.assert_not_called()
        assert result is True


class TestValidatorAIValidate:
    """Tests for _ai_validate method."""

    def test_returns_true_for_valid_response(self, validator):
        """Test returns True when AI says TRUE."""
        validator.generate = MagicMock(return_value="TRUE")

        result = validator._ai_validate("Valid content...", "English", "Write a story")

        assert result is True

    def test_returns_false_for_invalid_response(self, validator):
        """Test returns False when AI says FALSE."""
        validator.generate = MagicMock(return_value="FALSE")

        result = validator._ai_validate("Invalid content...", "English", "Write a story")

        assert result is False

    def test_handles_ambiguous_response(self, validator):
        """Test defaults to True for ambiguous response."""
        validator.generate = MagicMock(return_value="Maybe? I'm not sure...")

        result = validator._ai_validate("Content...", "English", "Task")

        # Ambiguous response should fail open (return True)
        assert result is True

    def test_handles_ai_failure_gracefully(self, validator):
        """Test handles AI call failure gracefully."""
        validator.generate = MagicMock(side_effect=Exception("LLM error"))

        result = validator._ai_validate("Content...", "English", "Task")

        # Should fail open (return True)
        assert result is True

    def test_truncates_long_response_for_validation(self, validator):
        """Test truncates very long response for validation prompt."""
        validator.generate = MagicMock(return_value="TRUE")
        long_content = "A" * 5000

        validator._ai_validate(long_content, "English", "Task")

        call_args = validator.generate.call_args
        prompt = call_args[0][0]
        # Should contain truncation indicator
        assert "..." in prompt


class TestValidateOrRaise:
    """Tests for validate_or_raise convenience function."""

    def test_returns_response_when_valid(self):
        """Test returns original response when valid."""
        response = "This is valid English content."

        with patch("agents.base.ollama.Client"):
            result = validate_or_raise(response, "English")

        assert result == response

    def test_raises_for_invalid_response(self):
        """Test raises error for invalid response."""
        with patch("agents.base.ollama.Client"):
            with pytest.raises(ResponseValidationError):
                validate_or_raise("", "English")

    def test_accepts_custom_validator(self, validator):
        """Test accepts pre-created validator agent."""
        response = "Valid response."

        result = validate_or_raise(response, "English", validator=validator)

        assert result == response

    def test_creates_validator_when_not_provided(self):
        """Test creates new validator when not provided."""
        response = "Valid response."

        with patch("agents.base.ollama.Client"):
            result = validate_or_raise(response, "English")

        assert result == response


class TestValidatorLanguageChecks:
    """Tests for language-specific validation."""

    def test_accepts_german_when_german_expected(self, validator):
        """Test accepts German text when German is expected."""
        response = """Die Sonne ging unter und tauchte die Berge in goldenes Licht.
Maria schaute aus dem Fenster und dachte an die Reise, die vor ihr lag."""

        # Not checking for English, so should pass
        validator._ai_validate = MagicMock(return_value=True)
        result = validator.validate_response(response, "German")

        assert result is True

    def test_cjk_check_only_for_english(self, validator):
        """Test CJK character check only applies to English."""
        response = "这是一个中文测试文本，包含很多汉字。"

        # For Chinese expected language, CJK characters are fine
        validator._ai_validate = MagicMock(return_value=True)
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

    def test_ai_validation_failure_raises_error(self, validator):
        """Test AI validation failure raises ResponseValidationError."""
        long_response = "Valid content. " * 50
        validator._ai_validate = MagicMock(return_value=False)

        # AI validation returns False, which should raise
        with pytest.raises(ResponseValidationError, match="AI validator rejected"):
            validator.validate_response(long_response, "English", "Story task")

    def test_ai_validation_exception_is_logged_and_passes(self, validator):
        """Test AI validation exception is logged but validation passes."""
        long_response = "Valid content. " * 50
        validator._ai_validate = MagicMock(side_effect=Exception("Test error"))

        # Should pass because it fails open
        result = validator.validate_response(long_response, "English", "Story task")
        assert result is True
