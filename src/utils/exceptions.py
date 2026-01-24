"""Centralized exception hierarchy for Story Factory.

Exception Hierarchy:

    StoryFactoryError (base for all application errors)
    ├── LLMError (LLM/Ollama related errors)
    │   ├── LLMConnectionError (connection failures)
    │   └── LLMGenerationError (generation failures after retries)
    ├── ValidationError (validation failures)
    │   └── ResponseValidationError (AI response validation)
    ├── ExportError (export/file related errors)
    └── WorldGenerationError (world entity generation failures)

Usage:
    from src.utils.exceptions import LLMError, LLMConnectionError

    try:
        agent.generate(prompt)
    except LLMConnectionError:
        logger.error("Failed to connect to Ollama")
    except LLMError:
        logger.error("LLM operation failed")
"""


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
        super().__init__(message)
        self.response_preview = response_preview
        self.expected_type = expected_type
