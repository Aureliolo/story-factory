"""Error handling utilities for Story Factory."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Error message constants
OLLAMA_UNAVAILABLE_MSG = "Please ensure Ollama is running and accessible."


def handle_ollama_errors(
    default_return: Any = None, raise_on_error: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to gracefully handle Ollama connection errors.

    Args:
        default_return: Value to return if Ollama is unavailable (default: None)
        raise_on_error: If True, re-raise the exception after logging (default: False)

    Example:
        @handle_ollama_errors(default_return="Ollama unavailable")
        def generate_text():
            # Will return "Ollama unavailable" if connection fails
            return ollama.generate(...)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Decorate function with Ollama error handling.

        Args:
            func: Function to wrap with error handling.

        Returns:
            Wrapped function.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute function with error handling.

            Args:
                *args: Positional arguments to pass to function.
                **kwargs: Keyword arguments to pass to function.

            Returns:
                Function result or default_return on error.

            Raises:
                Exception: Re-raises if raise_on_error is True.
            """
            try:
                return func(*args, **kwargs)
            except (ConnectionError, OSError, TimeoutError) as e:
                # Standard connection-related errors
                logger.error(
                    f"Ollama connection error in {func.__name__}: {e}. {OLLAMA_UNAVAILABLE_MSG}"
                )
                if raise_on_error:
                    raise
                return cast(T, default_return)
            except Exception as e:
                # Check for ollama-specific errors using hasattr for robustness
                # Ollama errors typically have 'error' or 'status_code' attributes
                is_ollama_error = hasattr(e, "status_code") or (
                    hasattr(e, "__module__") and "ollama" in e.__module__
                )

                if is_ollama_error:
                    logger.error(f"Ollama error in {func.__name__}: {e}. {OLLAMA_UNAVAILABLE_MSG}")
                else:
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)

                if raise_on_error:
                    raise
                return cast(T, default_return)

        return wrapper

    return decorator


def retry_with_fallback(
    max_retries: int = 3, fallback_value: Any = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with fallback on failure.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        fallback_value: Value to return after all retries fail (default: None)

    Example:
        @retry_with_fallback(max_retries=5, fallback_value=[])
        def fetch_data():
            # Will retry up to 5 times, return [] if all fail
            return requests.get(...).json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Decorate function with retry and fallback logic.

        Args:
            func: Function to wrap with retry logic.

        Returns:
            Wrapped function.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute function with retry logic.

            Args:
                *args: Positional arguments to pass to function.
                **kwargs: Keyword arguments to pass to function.

            Returns:
                Function result or fallback_value after all retries fail.
            """
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} failed on attempt {attempt + 1}/{max_retries}: {e}"
                    )

            logger.error(
                f"{func.__name__} failed after {max_retries} attempts. "
                f"Last error: {last_exception}",
                exc_info=True,
            )
            return cast(T, fallback_value)

        return wrapper

    return decorator


class ErrorBoundary:
    """Context manager for handling errors with fallback behavior.

    Example:
        with ErrorBoundary(fallback="Error occurred", log_level="ERROR"):
            risky_operation()
    """

    def __init__(self, fallback: Any = None, log_level: str = "ERROR", raise_on_exit: bool = False):
        """Initialize error boundary.

        Args:
            fallback: Value to return/assign on error
            log_level: Logging level for error messages
            raise_on_exit: If True, re-raise exception after logging
        """
        self.fallback = fallback
        self.log_level = log_level
        self.raise_on_exit = raise_on_exit
        self.exception: BaseException | None = None

    def __enter__(self) -> ErrorBoundary:
        """Enter the error boundary context.

        Returns:
            The ErrorBoundary instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit the error boundary context.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception instance if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            True to suppress exception, False to re-raise.
        """
        if exc_type is not None:
            self.exception = exc_val
            log_func = getattr(logger, self.log_level.lower(), logger.error)
            log_func(f"Error in boundary: {exc_val}", exc_info=True)

            if self.raise_on_exit:
                return False  # Re-raise exception

            return True  # Suppress exception

        return False
