"""Progressive retry strategies for LLM calls.

Provides mechanisms to adjust parameters on retry attempts when initial
generation fails or returns empty/invalid results.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetryContext:
    """Context for progressive retry strategies.

    Tracks retry attempts and provides adjusted parameters for each attempt:
    - Attempt 1: Same parameters (standard retry)
    - Attempt 2: Increase temperature to encourage different output
    - Attempt 3+: Simplify prompt to reduce complexity

    Attributes:
        attempt: Current attempt number (1-indexed).
        base_temperature: Original temperature setting.
        temp_increase: Amount to increase temperature on attempt 2+.
        simplify_on_attempt: Attempt number at which to start simplifying prompts.
        max_attempts: Maximum number of attempts before giving up.
    """

    attempt: int = 1
    base_temperature: float = 0.7
    temp_increase: float = 0.15
    simplify_on_attempt: int = 3
    max_attempts: int = 5
    _original_prompt: str = field(default="", repr=False)

    def get_temperature(self) -> float:
        """Get adjusted temperature for current attempt.

        Returns:
            Temperature value, increased by temp_increase for attempts 2+.
        """
        if self.attempt == 1:
            return self.base_temperature

        # Increase temperature on subsequent attempts, capped at 1.0
        adjusted = min(1.0, self.base_temperature + self.temp_increase)
        logger.debug(
            "Retry attempt %d: adjusting temperature from %.2f to %.2f",
            self.attempt,
            self.base_temperature,
            adjusted,
        )
        return adjusted

    def should_simplify(self) -> bool:
        """Check if prompt should be simplified on this attempt.

        Returns:
            True if current attempt is at or beyond simplify_on_attempt.
        """
        should = self.attempt >= self.simplify_on_attempt
        logger.debug(
            "Retry attempt %d: should_simplify=%s (threshold=%d)",
            self.attempt,
            should,
            self.simplify_on_attempt,
        )
        return should

    def increment(self) -> RetryContext:
        """Create a new context for the next attempt.

        Returns:
            New RetryContext with incremented attempt number.
        """
        next_ctx = RetryContext(
            attempt=self.attempt + 1,
            base_temperature=self.base_temperature,
            temp_increase=self.temp_increase,
            simplify_on_attempt=self.simplify_on_attempt,
            max_attempts=self.max_attempts,
            _original_prompt=self._original_prompt,
        )
        logger.debug(
            "Incrementing retry context: attempt %d -> %d",
            self.attempt,
            next_ctx.attempt,
        )
        return next_ctx

    def should_retry(self) -> bool:
        """Check if another retry attempt should be made.

        Returns:
            True if current attempt is less than max_attempts.
        """
        should = self.attempt < self.max_attempts
        logger.debug(
            "Retry attempt %d/%d: should_retry=%s",
            self.attempt,
            self.max_attempts,
            should,
        )
        return should

    def store_original_prompt(self, prompt: str) -> None:
        """Store the original prompt for potential simplification.

        Args:
            prompt: The original prompt to store.
        """
        self._original_prompt = prompt
        logger.debug("Stored original prompt (%d chars)", len(prompt))

    def get_original_prompt(self) -> str:
        """Get the stored original prompt.

        Returns:
            The original prompt, or empty string if not stored.
        """
        logger.debug("Retrieved original prompt (%d chars)", len(self._original_prompt))
        return self._original_prompt


def simplify_prompt(prompt: str, entity_type: str) -> str:
    """Simplify a prompt by removing complex instructions.

    This is used when multiple retry attempts with normal prompts have failed.
    Simplified prompts focus on the core requirements and remove:
    - Detailed scoring guidance
    - Multiple examples
    - Complex formatting instructions

    Args:
        prompt: The original prompt to simplify.
        entity_type: Type of entity being generated (for context).

    Returns:
        Simplified version of the prompt.
    """
    logger.info("Simplifying prompt for %s generation after multiple failures", entity_type)

    # Extract key sections from the prompt
    lines = prompt.split("\n")
    simplified_lines = []
    skip_section = False

    for line in lines:
        line_lower = line.lower().strip()

        # Skip sections that add complexity (match section headers only)
        if any(
            line_lower.startswith(marker)
            for marker in [
                "=== critical",
                "strict rules:",
                "do not",
                "- do not",
                "scoring guide",
                "calibration",
            ]
        ):
            skip_section = True
            continue

        # Resume after blank line following skipped section
        if skip_section and not line.strip():
            skip_section = False
            continue

        if not skip_section:
            simplified_lines.append(line)

    simplified = "\n".join(simplified_lines)

    # Add a simplified instruction
    simplified += f"\n\nCreate a simple, valid {entity_type}. Focus on core attributes only."

    logger.debug("Simplified prompt from %d to %d characters", len(prompt), len(simplified))
    return simplified


def create_retry_context(
    base_temperature: float,
    temp_increase: float,
    simplify_on_attempt: int,
    max_attempts: int = 5,
) -> RetryContext:
    """Create a new retry context with the given settings.

    Args:
        base_temperature: Starting temperature for generation.
        temp_increase: Amount to increase temperature on retries.
        simplify_on_attempt: Attempt number to start simplifying prompts.
        max_attempts: Maximum retry attempts.

    Returns:
        Configured RetryContext instance.
    """
    logger.debug(
        "Creating retry context: base_temp=%.2f, temp_increase=%.2f, "
        "simplify_on=%d, max_attempts=%d",
        base_temperature,
        temp_increase,
        simplify_on_attempt,
        max_attempts,
    )
    return RetryContext(
        attempt=1,
        base_temperature=base_temperature,
        temp_increase=temp_increase,
        simplify_on_attempt=simplify_on_attempt,
        max_attempts=max_attempts,
    )
