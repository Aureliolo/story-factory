"""Tests for the retry strategies module."""

from src.utils.retry_strategies import (
    RetryContext,
    create_retry_context,
    simplify_prompt,
)


class TestRetryContext:
    """Tests for the RetryContext class."""

    def test_default_values(self):
        """RetryContext has sensible defaults."""
        ctx = RetryContext()
        assert ctx.attempt == 1
        assert ctx.base_temperature == 0.7
        assert ctx.temp_increase == 0.15
        assert ctx.simplify_on_attempt == 3
        assert ctx.max_attempts == 5

    def test_get_temperature_first_attempt(self):
        """First attempt uses base temperature."""
        ctx = RetryContext(base_temperature=0.7, temp_increase=0.15)
        assert ctx.get_temperature() == 0.7

    def test_get_temperature_second_attempt(self):
        """Second attempt increases temperature."""
        ctx = RetryContext(attempt=2, base_temperature=0.7, temp_increase=0.15)
        assert ctx.get_temperature() == 0.85

    def test_get_temperature_capped_at_one(self):
        """Temperature is capped at 1.0."""
        ctx = RetryContext(attempt=5, base_temperature=0.9, temp_increase=0.2)
        assert ctx.get_temperature() == 1.0

    def test_should_simplify_before_threshold(self):
        """should_simplify returns False before simplify_on_attempt."""
        ctx = RetryContext(attempt=2, simplify_on_attempt=3)
        assert ctx.should_simplify() is False

    def test_should_simplify_at_threshold(self):
        """should_simplify returns True at simplify_on_attempt."""
        ctx = RetryContext(attempt=3, simplify_on_attempt=3)
        assert ctx.should_simplify() is True

    def test_should_simplify_after_threshold(self):
        """should_simplify returns True after simplify_on_attempt."""
        ctx = RetryContext(attempt=4, simplify_on_attempt=3)
        assert ctx.should_simplify() is True

    def test_increment_creates_new_context(self):
        """increment creates a new context with incremented attempt."""
        ctx = RetryContext(attempt=1, base_temperature=0.7, temp_increase=0.15)
        ctx2 = ctx.increment()

        assert ctx.attempt == 1
        assert ctx2.attempt == 2
        assert ctx2.base_temperature == 0.7
        assert ctx2.temp_increase == 0.15

    def test_should_retry_before_max(self):
        """should_retry returns True when below max_attempts."""
        ctx = RetryContext(attempt=3, max_attempts=5)
        assert ctx.should_retry() is True

    def test_should_retry_at_max(self):
        """should_retry returns False at max_attempts."""
        ctx = RetryContext(attempt=5, max_attempts=5)
        assert ctx.should_retry() is False

    def test_should_retry_after_max(self):
        """should_retry returns False above max_attempts."""
        ctx = RetryContext(attempt=6, max_attempts=5)
        assert ctx.should_retry() is False

    def test_store_and_get_original_prompt(self):
        """Original prompt can be stored and retrieved."""
        ctx = RetryContext()
        assert ctx.get_original_prompt() == ""

        ctx.store_original_prompt("Test prompt")
        assert ctx.get_original_prompt() == "Test prompt"


class TestSimplifyPrompt:
    """Tests for the simplify_prompt function."""

    def test_removes_critical_sections(self):
        """simplify_prompt removes CRITICAL sections."""
        prompt = """Create a faction.

=== CRITICAL: UNIQUENESS REQUIREMENTS ===
Do not duplicate names.
Names to avoid: Faction A, Faction B.

Create something unique."""

        simplified = simplify_prompt(prompt, "faction")

        assert "CRITICAL" not in simplified
        assert "Create a faction" in simplified

    def test_removes_strict_rules(self):
        """simplify_prompt removes STRICT RULES sections."""
        prompt = """Create an item.

STRICT RULES:
- DO NOT use existing names
- DO NOT copy anything

Create something new."""

        simplified = simplify_prompt(prompt, "item")

        assert "STRICT RULES" not in simplified
        assert "DO NOT" not in simplified

    def test_removes_do_not_examples(self):
        """simplify_prompt removes DO NOT examples."""
        prompt = """Create a concept.

DO NOT use names like:
- Example 1
- Example 2

Create something unique."""

        simplified = simplify_prompt(prompt, "concept")

        # The DO NOT line should be removed
        assert "DO NOT use names like" not in simplified

    def test_adds_simplification_note(self):
        """simplify_prompt adds a simplification note."""
        prompt = "Create something."
        simplified = simplify_prompt(prompt, "faction")

        assert "Create a simple, valid faction" in simplified

    def test_preserves_core_content(self):
        """simplify_prompt preserves core prompt content."""
        prompt = """Create a character for a fantasy story.

STORY PREMISE: A hero's journey.
TONE: Epic

The character should be memorable."""

        simplified = simplify_prompt(prompt, "character")

        assert "Create a character" in simplified
        assert "STORY PREMISE" in simplified or "hero's journey" in simplified

    def test_reduces_prompt_length(self):
        """simplify_prompt reduces prompt length for complex prompts."""
        complex_prompt = """Create a faction.

=== CRITICAL: UNIQUENESS REQUIREMENTS ===
Existing factions:
- Faction A
- Faction B

STRICT RULES:
- DO NOT duplicate
- DO NOT use similar names

Scoring guide:
- Coherence: 8+
- Influence: 8+

Calibration examples:
- Good: Example 1
- Bad: Example 2

Create something unique."""

        simplified = simplify_prompt(complex_prompt, "faction")
        assert len(simplified) < len(complex_prompt)


class TestCreateRetryContext:
    """Tests for the create_retry_context function."""

    def test_creates_context_with_parameters(self):
        """create_retry_context creates a properly configured context."""
        ctx = create_retry_context(
            base_temperature=0.8, temp_increase=0.2, simplify_on_attempt=4, max_attempts=6
        )

        assert ctx.attempt == 1
        assert ctx.base_temperature == 0.8
        assert ctx.temp_increase == 0.2
        assert ctx.simplify_on_attempt == 4
        assert ctx.max_attempts == 6

    def test_creates_context_with_defaults(self):
        """create_retry_context uses default for max_attempts."""
        ctx = create_retry_context(base_temperature=0.7, temp_increase=0.15, simplify_on_attempt=3)

        assert ctx.max_attempts == 5


class TestRetryContextIntegration:
    """Integration tests for retry context usage patterns."""

    def test_progressive_retry_pattern(self):
        """Demonstrates the progressive retry pattern."""
        ctx = create_retry_context(base_temperature=0.7, temp_increase=0.15, simplify_on_attempt=3)

        # Simulate a retry loop
        temperatures = []
        simplify_flags = []

        while ctx.should_retry():
            temperatures.append(ctx.get_temperature())
            simplify_flags.append(ctx.should_simplify())
            ctx = ctx.increment()

        # Verify progression
        assert temperatures == [0.7, 0.85, 0.85, 0.85]
        assert simplify_flags == [False, False, True, True]

    def test_custom_retry_configuration(self):
        """Custom configuration works as expected."""
        ctx = create_retry_context(
            base_temperature=0.5, temp_increase=0.1, simplify_on_attempt=2, max_attempts=3
        )

        # Attempt 1: base temp, no simplify
        assert ctx.get_temperature() == 0.5
        assert not ctx.should_simplify()

        # Attempt 2: increased temp, simplify
        ctx = ctx.increment()
        assert ctx.get_temperature() == 0.6
        assert ctx.should_simplify()

        # Attempt 3: should not retry
        ctx = ctx.increment()
        assert ctx.get_temperature() == 0.6
        assert ctx.should_simplify()
        assert not ctx.should_retry()
