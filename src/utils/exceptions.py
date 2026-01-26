"""Centralized exception hierarchy for Story Factory.

Exception Hierarchy:

    StoryFactoryError (base for all application errors)
    ├── LLMError (LLM/Ollama related errors)
    │   ├── LLMConnectionError (connection failures)
    │   ├── LLMGenerationError (generation failures after retries)
    │   └── CircuitOpenError (circuit breaker blocking requests)
    ├── ValidationError (validation failures)
    │   └── ResponseValidationError (AI response validation)
    ├── ConfigError (configuration parsing/validation failures)
    ├── ExportError (export/file related errors)
    ├── WorldGenerationError (world entity generation failures)
    │   ├── EmptyGenerationError (empty/invalid entity content)
    │   └── DuplicateNameError (duplicate entity name detected)
    ├── GenerationCancelledError (user cancelled generation)
    ├── SuggestionError (AI suggestion generation failures)
    └── JSONParseError (JSON parsing failures)

Usage:
    from src.utils.exceptions import LLMError, LLMConnectionError

    try:
        agent.generate(prompt)
    except LLMConnectionError:
        logger.error("Failed to connect to Ollama")
    except LLMError:
        logger.error("LLM operation failed")
"""

import logging

logger = logging.getLogger(__name__)


class StoryFactoryError(Exception):
    """Base exception for all Story Factory errors.

    All custom exceptions should inherit from this class to allow
    catching all application-specific errors with a single except clause.
    """

    pass


class LLMError(StoryFactoryError):
    """Base exception for LLM-related errors.

    Raised when any LLM operation fails. Subclasses provide more
    specific error types.
    """

    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to Ollama.

    This typically indicates the Ollama server is not running or
    the connection was refused.
    """

    pass


class LLMGenerationError(LLMError):
    """Raised when generation fails after retries.

    This indicates the LLM request failed despite multiple retry
    attempts. Check logs for specific failure reasons.
    """

    pass


class CircuitOpenError(LLMError):
    """Raised when the circuit breaker is open and blocking requests.

    This indicates too many consecutive LLM failures have occurred,
    and requests are being blocked to prevent cascading failures.
    The circuit will automatically transition to half-open state
    after the configured timeout to test if the service has recovered.

    Attributes:
        time_until_retry: Seconds until the circuit may allow requests.
    """

    def __init__(self, message: str, time_until_retry: float | None = None):
        """Initialize CircuitOpenError with timing information.

        Args:
            message: Human-readable error message.
            time_until_retry: Seconds until circuit may transition to half-open.
        """
        super().__init__(message)
        self.time_until_retry = time_until_retry
        logger.debug(
            "CircuitOpenError initialized: message=%s, time_until_retry=%s",
            message,
            time_until_retry,
        )


class ValidationError(StoryFactoryError):
    """Base exception for validation errors.

    Raised when input validation fails.
    """

    pass


class ResponseValidationError(ValidationError):
    """Raised when an AI response fails validation.

    This indicates the LLM returned a response that doesn't meet
    the expected format or content requirements.
    """

    pass


class ExportError(StoryFactoryError):
    """Raised when export operations fail.

    This covers failures in exporting stories to various formats
    (EPUB, PDF, etc.) or file I/O errors during export.
    """

    pass


class WorldGenerationError(StoryFactoryError):
    """Raised when world entity generation fails.

    This indicates a failure to generate characters, locations,
    relationships, or other world entities after all retries.
    """

    pass


class GenerationCancelledError(StoryFactoryError):
    """Raised when generation is cancelled by user.

    This indicates the user clicked Cancel during a long-running
    generation operation (e.g., Build Story Structure).
    """

    pass


class SuggestionError(StoryFactoryError):
    """Raised when AI suggestion generation fails.

    This indicates a failure to generate project name suggestions,
    writing prompts, or other AI-assisted suggestions.
    """

    pass


class ConfigError(StoryFactoryError):
    """Raised when configuration parsing or validation fails.

    This indicates issues with pyproject.toml, settings files,
    or other configuration that cannot be loaded or is invalid.
    """

    pass


class JSONParseError(StoryFactoryError):
    """Raised when JSON extraction or parsing fails.

    This indicates the LLM response could not be parsed as valid JSON,
    or the JSON structure did not match the expected format.

    Attributes:
        response_preview: First 500 chars of the raw response for debugging.
        expected_type: The expected type (dict, list, or model class name).
    """

    def __init__(
        self,
        message: str,
        response_preview: str | None = None,
        expected_type: str | None = None,
    ):
        """
        Initialize the JSONParseError with an error message and optional parsing context.

        Parameters:
            message (str): Human-readable error message describing the parse failure.
            response_preview (str | None): Optional preview (typically up to the first ~500 characters) of the raw response that failed to parse, for debugging.
            expected_type (str | None): Optional description of the expected JSON type or structure (e.g., "dict", "list", or model class name).

        Notes:
            The provided `response_preview` and `expected_type` are stored on the instance as
            `response_preview` and `expected_type` respectively.
        """
        super().__init__(message)
        self.response_preview = response_preview
        self.expected_type = expected_type


class EmptyGenerationError(WorldGenerationError):
    """Raised when entity generation returns empty or invalid content.

    This indicates the LLM returned an empty name, empty description,
    or otherwise invalid entity that cannot be used.
    """

    pass


class DuplicateNameError(WorldGenerationError):
    """Raised when a generated entity has a duplicate name.

    This indicates the generated entity name matches or is too similar
    to an existing entity name in the world.

    Attributes:
        generated_name: The name that was generated.
        existing_name: The existing name it conflicts with.
        reason: Why it's considered a duplicate (exact, case-insensitive, prefix, substring).
    """

    def __init__(
        self,
        message: str,
        generated_name: str | None = None,
        existing_name: str | None = None,
        reason: str | None = None,
    ):
        """
        Initialize a DuplicateNameError with an error message and optional context about the name conflict.

        Parameters:
            message (str): Human-readable error message.
            generated_name (str | None): The generated name that caused the conflict, if available.
            existing_name (str | None): The existing name that conflicts with the generated name, if known.
            reason (str | None): Description of why the names are considered duplicates (e.g., "exact match", "case-insensitive match", "prefix", "substring").
        """
        super().__init__(message)
        self.generated_name = generated_name
        self.existing_name = existing_name
        self.reason = reason


class RelationshipValidationError(WorldGenerationError):
    """Raised when relationship validation fails.

    This indicates an attempt to create a relationship with invalid
    source/target entities, self-referential relationship, or other
    validation failures.

    Attributes:
        source_id: The source entity ID that was provided.
        target_id: The target entity ID that was provided.
        reason: Why the validation failed (e.g., "source_not_found", "self_loop").
        suggestions: List of suggested fixes or alternatives.
    """

    def __init__(
        self,
        message: str,
        source_id: str | None = None,
        target_id: str | None = None,
        reason: str | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize RelationshipValidationError with context about the failed relationship.

        Parameters:
            message (str): Human-readable error message.
            source_id (str | None): The source entity ID that was provided.
            target_id (str | None): The target entity ID that was provided.
            reason (str | None): Why validation failed (e.g., "source_not_found",
                "target_not_found", "self_loop").
            suggestions (list[str] | None): List of suggested fixes or alternatives.
        """
        super().__init__(message)
        self.source_id = source_id
        self.target_id = target_id
        self.reason = reason
        self.suggestions = suggestions or []
        logger.debug(
            "RelationshipValidationError initialized: source_id=%s, target_id=%s, reason=%s",
            source_id,
            target_id,
            reason,
        )
