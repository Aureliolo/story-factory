"""Tests for error handling utilities."""

import pytest

from src.agents.base import BaseAgent, LLMConnectionError
from src.utils.error_handling import ErrorBoundary, handle_ollama_errors, retry_with_fallback


class TestHandleOllamaErrors:
    """Tests for handle_ollama_errors decorator."""

    def test_returns_default_on_connection_error(self):
        """Should return default value when LLMConnectionError occurs."""

        @handle_ollama_errors(default_return="fallback")
        def failing_function():
            raise LLMConnectionError("Connection failed")

        result = failing_function()
        assert result == "fallback"

    def test_raises_when_configured(self):
        """Should re-raise exception when raise_on_error=True."""

        @handle_ollama_errors(raise_on_error=True)
        def failing_function():
            raise LLMConnectionError("Connection failed")

        with pytest.raises(LLMConnectionError):
            failing_function()

    def test_handles_generic_exceptions(self):
        """Should handle generic exceptions too."""

        @handle_ollama_errors(default_return=None)
        def failing_function():
            raise ValueError("Some error")

        result = failing_function()
        assert result is None

    def test_returns_normal_value_on_success(self):
        """Should return normal value when no error occurs."""

        @handle_ollama_errors(default_return="fallback")
        def working_function():
            return "success"

        result = working_function()
        assert result == "success"


class TestRetryWithFallback:
    """Tests for retry_with_fallback decorator."""

    def test_retries_on_failure(self):
        """Should retry the specified number of times."""
        attempts = []

        @retry_with_fallback(max_retries=3, fallback_value="fallback")
        def flaky_function():
            attempts.append(1)
            raise ValueError("Error")

        result = flaky_function()
        assert len(attempts) == 3
        assert result == "fallback"

    def test_succeeds_on_eventual_success(self):
        """Should return value when function eventually succeeds."""
        attempts = []

        @retry_with_fallback(max_retries=5, fallback_value="fallback")
        def eventually_succeeds():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_succeeds()
        assert len(attempts) == 3
        assert result == "success"

    def test_returns_immediately_on_success(self):
        """Should not retry when function succeeds first time."""
        attempts = []

        @retry_with_fallback(max_retries=3, fallback_value="fallback")
        def working_function():
            attempts.append(1)
            return "success"

        result = working_function()
        assert len(attempts) == 1
        assert result == "success"


class TestErrorBoundary:
    """Tests for ErrorBoundary context manager."""

    def test_catches_exceptions(self):
        """Should catch exceptions and not propagate."""
        with ErrorBoundary():
            raise ValueError("Test error")
        # Should not raise

    def test_provides_fallback(self):
        """Should provide access to fallback value."""
        boundary = ErrorBoundary(fallback="fallback_value")
        with boundary:
            raise ValueError("Test error")
        # fallback is accessible after context exit (ErrorBoundary suppresses exceptions)
        assert boundary.fallback == "fallback_value"

    def test_stores_exception(self):
        """Should store the exception that occurred."""
        boundary = ErrorBoundary()
        with boundary:
            raise ValueError("Test error")
        # exception is accessible after context exit (ErrorBoundary suppresses exceptions)
        assert isinstance(boundary.exception, ValueError)
        assert str(boundary.exception) == "Test error"

    def test_reraises_when_configured(self):
        """Should re-raise exception when raise_on_exit=True."""
        with pytest.raises(ValueError):
            with ErrorBoundary(raise_on_exit=True):
                raise ValueError("Test error")

    def test_no_error_on_success(self):
        """Should work normally when no error occurs."""
        boundary = ErrorBoundary()
        with boundary:
            _ = 1 + 1  # Some operation

        assert boundary.exception is None


class TestDecoratorIntegration:
    """Tests for decorator integration with BaseAgent methods."""

    def test_check_ollama_health_returns_error_on_failure(self):
        """check_ollama_health should gracefully handle connection errors."""
        # Use an invalid URL to trigger connection error
        is_healthy, message = BaseAgent.check_ollama_health("http://invalid-host:99999")

        assert is_healthy is False
        assert "Ollama connection failed" in message
