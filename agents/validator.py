"""Response Validator Agent - Quick sanity checks on AI responses."""

import logging
import re
from typing import TYPE_CHECKING

from utils.exceptions import ResponseValidationError
from utils.validation import validate_not_empty

from .base import BaseAgent

if TYPE_CHECKING:
    from settings import Settings

logger = logging.getLogger(__name__)

# Re-export exception for backward compatibility
__all__ = ["ValidatorAgent", "ResponseValidationError", "validate_or_raise"]


VALIDATOR_SYSTEM_PROMPT = """You are a response validator. Your ONLY job is to answer TRUE or FALSE.

You check if AI-generated content:
1. Is written in the CORRECT language (most important!)
2. Is relevant to the task (not random gibberish)
3. Does not contain obvious errors like wrong character encoding

Be strict about language - if the expected language is English but you see Chinese/Japanese/Korean characters, that's FALSE.
Be lenient about content quality - you're just checking basic sanity, not quality.

ALWAYS respond with ONLY one word: TRUE or FALSE. Nothing else."""


class ValidatorAgent(BaseAgent):
    """Agent that validates AI responses for basic correctness."""

    def __init__(self, model: str | None = None, settings: "Settings | None" = None) -> None:
        super().__init__(
            name="Validator",
            role="Response Validator",
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            agent_role="validator",
            model=model,
            settings=settings,
        )

    def validate_response(
        self,
        response: str,
        expected_language: str = "English",
        task_description: str = "",
    ) -> bool:
        """Validate an AI response.

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
            if len(cjk_matches) > 5:  # Allow a few for names/terms
                raise ResponseValidationError(
                    f"Response contains {len(cjk_matches)} CJK characters but expected {expected_language}. "
                    f"Sample: {''.join(cjk_matches[:10])}"
                )

        # Check for excessive non-printable characters
        printable_ratio = sum(1 for c in response if c.isprintable() or c.isspace()) / len(response)
        if printable_ratio < 0.9:
            raise ResponseValidationError(
                f"Response contains too many non-printable characters ({1 - printable_ratio:.0%})"
            )

        # Use AI for more nuanced validation (only for longer responses)
        if len(response) > 200:
            try:
                result = self._ai_validate(response, expected_language, task_description)
                if not result:
                    raise ResponseValidationError(
                        f"AI validator rejected response. Expected language: {expected_language}"
                    )
            except ResponseValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                # If validator fails, log but don't block (fail open)
                logger.warning(f"Validator check failed: {e}")

        return True

    def _ai_validate(
        self,
        response: str,
        expected_language: str,
        task_description: str,
    ) -> bool:
        """Use AI to validate response."""
        # Truncate response for validation (we don't need to check everything)
        sample = response[:1000] + ("..." if len(response) > 1000 else "")

        prompt = f"""Check this AI response:

EXPECTED LANGUAGE: {expected_language}
TASK: {task_description or "Generate story content"}

RESPONSE SAMPLE:
---
{sample}
---

Is this response:
1. Written in {expected_language}? (CRITICAL)
2. Relevant to the task?
3. Not gibberish or corrupted text?

Answer only TRUE or FALSE."""

        try:
            result = self.generate(prompt, temperature=self.temperature)
            result_clean = result.strip().upper()

            # Parse response
            if "TRUE" in result_clean:
                return True
            elif "FALSE" in result_clean:
                return False
            else:
                # Ambiguous response, assume OK
                logger.debug(f"Validator gave ambiguous response: {result}")
                return True
        except Exception as e:
            logger.warning(f"Validator AI call failed: {e}")
            return True  # Fail open


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
