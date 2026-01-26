"""Circuit breaker pattern for LLM calls.

Provides protection against cascading failures when the LLM service
(Ollama) is struggling or unavailable.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failures exceeded threshold, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading LLM failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation. Requests pass through. Failures are counted.
    - OPEN: Too many failures. Requests are rejected immediately with CircuitOpenError.
    - HALF_OPEN: After timeout, allow limited requests to test if service recovered.

    State transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: After timeout_seconds have passed
    - HALF_OPEN -> CLOSED: When success_count >= success_threshold
    - HALF_OPEN -> OPEN: On any failure

    Attributes:
        name: Identifier for this circuit breaker (for logging).
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Number of successes in half-open to close circuit.
        timeout_seconds: Seconds to wait before transitioning from open to half-open.
        enabled: Whether the circuit breaker is active.
    """

    name: str = "llm"
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    enabled: bool = True

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_in_flight: bool = field(default=False, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        """Initialize logging after dataclass init."""
        logger.info(
            "Circuit breaker '%s' initialized: enabled=%s, failure_threshold=%d, "
            "success_threshold=%d, timeout=%.1fs",
            self.name,
            self.enabled,
            self.failure_threshold,
            self.success_threshold,
            self.timeout_seconds,
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                self._check_timeout()
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self.state == CircuitState.CLOSED

    def _check_timeout(self) -> None:
        """Check if timeout has passed and transition to half-open if so.

        Must be called while holding _lock.
        """
        if self._last_failure_time is None:
            return

        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.timeout_seconds:
            logger.info(
                "Circuit breaker '%s': timeout elapsed (%.1fs), transitioning OPEN -> HALF_OPEN",
                self.name,
                elapsed,
            )
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._half_open_in_flight = False

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        if not self.enabled:
            return True

        with self._lock:
            state = self.state  # This triggers timeout check

            if state == CircuitState.CLOSED:
                return True
            elif state == CircuitState.HALF_OPEN:
                # Only allow a single in-flight probe request in HALF_OPEN state
                if self._half_open_in_flight:
                    logger.warning(
                        "Circuit breaker '%s': half-open probe already in flight, rejecting request",
                        self.name,
                    )
                    return False
                self._half_open_in_flight = True
                logger.debug(
                    "Circuit breaker '%s': allowing single test request in HALF_OPEN state",
                    self.name,
                )
                return True
            else:  # OPEN
                logger.warning(
                    "Circuit breaker '%s': rejecting request, circuit is OPEN", self.name
                )
                return False

    def record_success(self) -> None:
        """Record a successful request.

        In HALF_OPEN state, increments success count and may close circuit.
        In CLOSED state, resets failure count.
        """
        if not self.enabled:
            return

        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = False
                self._success_count += 1
                logger.debug(
                    "Circuit breaker '%s': success in HALF_OPEN (%d/%d)",
                    self.name,
                    self._success_count,
                    self.success_threshold,
                )
                if self._success_count >= self.success_threshold:
                    logger.info(
                        "Circuit breaker '%s': service recovered, transitioning HALF_OPEN -> CLOSED",
                        self.name,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    logger.debug(
                        "Circuit breaker '%s': resetting failure count on success", self.name
                    )
                    self._failure_count = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed request.

        In CLOSED state, increments failure count and may open circuit.
        In HALF_OPEN state, immediately opens circuit.

        Args:
            error: Optional exception that caused the failure.
        """
        if not self.enabled:
            return

        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = False
                logger.warning(
                    "Circuit breaker '%s': failure in HALF_OPEN, transitioning HALF_OPEN -> OPEN. "
                    "Error: %s",
                    self.name,
                    error,
                )
                self._state = CircuitState.OPEN
                self._failure_count = self.failure_threshold  # Keep at threshold
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                logger.debug(
                    "Circuit breaker '%s': failure %d/%d. Error: %s",
                    self.name,
                    self._failure_count,
                    self.failure_threshold,
                    error,
                )
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        "Circuit breaker '%s': failure threshold reached, "
                        "transitioning CLOSED -> OPEN",
                        self.name,
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker to initial closed state.

        Useful for testing or manual recovery.
        """
        with self._lock:
            logger.info("Circuit breaker '%s': manually reset to CLOSED", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_in_flight = False

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status for monitoring.

        Returns:
            Dictionary with state, counts, and timing info.
        """
        with self._lock:
            state = self.state
            status = {
                "name": self.name,
                "enabled": self.enabled,
                "state": state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "success_count": self._success_count,
                "success_threshold": self.success_threshold,
            }

            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                status["time_since_last_failure"] = elapsed
                if state == CircuitState.OPEN:
                    status["time_until_half_open"] = max(0, self.timeout_seconds - elapsed)

            return status


# Global circuit breaker instance for LLM calls
_global_circuit_breaker: CircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def get_circuit_breaker(
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout_seconds: float = 60.0,
    enabled: bool = True,
) -> CircuitBreaker:
    """Get or create the global circuit breaker for LLM calls.

    The circuit breaker is lazily initialized on first access.

    Args:
        failure_threshold: Failures before opening (used on creation only).
        success_threshold: Successes to close from half-open (used on creation only).
        timeout_seconds: Timeout before half-open (used on creation only).
        enabled: Whether circuit breaker is active (used on creation only).

    Returns:
        The global CircuitBreaker instance.
    """
    global _global_circuit_breaker

    if _global_circuit_breaker is None:
        with _circuit_breaker_lock:
            if _global_circuit_breaker is None:
                _global_circuit_breaker = CircuitBreaker(
                    name="llm",
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout_seconds=timeout_seconds,
                    enabled=enabled,
                )

    return _global_circuit_breaker


def reset_global_circuit_breaker() -> None:
    """Reset the global circuit breaker.

    Useful for testing to ensure a clean state.
    """
    global _global_circuit_breaker
    with _circuit_breaker_lock:
        if _global_circuit_breaker is not None:
            _global_circuit_breaker.reset()
        _global_circuit_breaker = None
