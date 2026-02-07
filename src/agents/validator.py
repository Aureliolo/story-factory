"""Response Validator - Rule-based sanity checks on AI responses.

Investigation #267 proved AI validation (smollm2) adds zero value over regex
checks: both achieve ~70% accuracy with ~40% false positive rate. This module
now uses only fast rule-based checks (CJK detection, printable ratio).
"""

import logging
import re

from src.settings import Settings
from src.utils.exceptions import ResponseValidationError
from src.utils.validation import validate_not_empty

logger = logging.getLogger(__name__)

# Re-export exception for backward compatibility
__all__ = ["ResponseValidationError", "ValidatorAgent", "validate_or_raise"]


class ValidatorAgent:
    """Rule-based validator for AI responses.

    Performs fast sanity checks without requiring an LLM:
    - CJK character detection for English responses
    - Non-printable character ratio check
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the Validator.

        Args:
            settings: Application settings. If None, loads default settings.
        """
        self.settings = settings or Settings.load()
        logger.debug("ValidatorAgent initialized (rule-based only)")

    def validate_response(
        self,
        response: str,
        expected_language: str = "English",
        task_description: str = "",
    ) -> bool:
        """Validate an AI response using rule-based checks.

        Args:
            response: The AI-generated response to validate
            expected_language: The language the response should be in
            task_description: Brief description of what the response should contain

        Returns:
            True if validation passes

        Raises:
            ResponseValidationError: If validation fails
        """
        validate_not_empty(response, "response")
        validate_not_empty(expected_language, "expected_language")

        # Check for obvious wrong-language characters
        if expected_language == "English":
            # Check for CJK characters (Chinese, Japanese, Korean)
            cjk_pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
            cjk_matches = cjk_pattern.findall(response)
            if len(cjk_matches) > self.settings.validator_cjk_char_threshold:
                raise ResponseValidationError(
                    f"Response contains {len(cjk_matches)} CJK characters but expected {expected_language}. "
                    f"Sample: {''.join(cjk_matches[:10])}"
                )

        # Check for excessive non-printable characters
        printable_ratio = sum(1 for c in response if c.isprintable() or c.isspace()) / len(response)
        if printable_ratio < self.settings.validator_printable_ratio:
            raise ResponseValidationError(
                f"Response contains too many non-printable characters ({1 - printable_ratio:.0%})"
            )

        logger.debug(
            "Response validated (lang=%s, len=%d, printable=%.0f%%)",
            expected_language,
            len(response),
            printable_ratio * 100,
        )
        return True


def validate_or_raise(
    response: str,
    expected_language: str = "English",
    task_description: str = "",
    validator: ValidatorAgent | None = None,
) -> str:
    """Convenience function to validate a response or raise an exception.

    Args:
        response: The response to validate
        expected_language: Expected language of the response
        task_description: What the response should contain
        validator: Optional pre-created validator agent

    Returns:
        The original response if valid

    Raises:
        ResponseValidationError: If validation fails
    """
    if validator is None:
        validator = ValidatorAgent()

    validator.validate_response(response, expected_language, task_description)
    return response
