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
    from utils.exceptions import LLMError, LLMConnectionError

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
