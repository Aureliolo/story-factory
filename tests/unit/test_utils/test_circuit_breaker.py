"""Tests for the circuit breaker module."""

import threading
import time
from typing import cast

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker,
    reset_global_circuit_breaker,
)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open

    def test_allow_request_when_closed(self):
        """Requests are allowed when circuit is closed."""
        cb = CircuitBreaker(name="test")
        assert cb.allow_request()

    def test_allow_request_when_disabled(self):
        """Requests are always allowed when circuit breaker is disabled."""
        cb = CircuitBreaker(name="test", enabled=False)
        # Even after failures, requests should be allowed
        for _ in range(10):
            cb.record_failure(Exception("test"))
        assert cb.allow_request()

    def test_record_success_resets_failure_count(self):
        """Recording success resets the failure count in closed state."""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb.record_failure(Exception("test"))
        cb.record_failure(Exception("test"))
        assert cb._failure_count == 2

        cb.record_success()
        assert cb._failure_count == 0

    def test_circuit_opens_after_failure_threshold(self):
        """Circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        for i in range(3):
            cb.record_failure(Exception(f"test {i}"))

        assert cb.state == CircuitState.OPEN
        assert cb.is_open
        assert not cb.allow_request()

    def test_circuit_transitions_to_half_open_after_timeout(self, monkeypatch):
        """Circuit transitions from OPEN to HALF_OPEN after timeout."""
        # Use deterministic time control instead of real time.sleep
        current_time = {"t": 1000.0}
        monkeypatch.setattr(time, "time", lambda: current_time["t"])

        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=60.0)

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))
        assert cb.state == CircuitState.OPEN

        # Advance time past timeout
        current_time["t"] += 61.0

        # Should transition to HALF_OPEN (cast breaks type narrowing)
        current_state = cast(CircuitState, cb.state)
        assert current_state == CircuitState.HALF_OPEN
        assert cb.allow_request()

    def test_half_open_closes_on_success_threshold(self, monkeypatch):
        """Circuit closes from HALF_OPEN after success threshold."""
        # Use deterministic time control instead of real time.sleep
        current_time = {"t": 1000.0}
        monkeypatch.setattr(time, "time", lambda: current_time["t"])

        cb = CircuitBreaker(
            name="test", failure_threshold=2, success_threshold=2, timeout_seconds=60.0
        )

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        # Advance time past timeout to transition to HALF_OPEN
        current_time["t"] += 61.0
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Still half-open after 1 success

        cb.record_success()
        # Cast breaks type narrowing from earlier state checks
        current_state = cast(CircuitState, cb.state)
        assert current_state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self, monkeypatch):
        """Circuit reopens from HALF_OPEN on any failure."""
        # Use deterministic time control instead of real time.sleep
        current_time = {"t": 1000.0}
        monkeypatch.setattr(time, "time", lambda: current_time["t"])

        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=60.0)

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        # Advance time past timeout to transition to HALF_OPEN
        current_time["t"] += 61.0
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in HALF_OPEN should immediately reopen (cast breaks type narrowing)
        cb.record_failure(Exception("test in half-open"))
        current_state = cast(CircuitState, cb.state)
        assert current_state == CircuitState.OPEN

    def test_reset(self):
        """Reset returns circuit to closed state."""
        cb = CircuitBreaker(name="test", failure_threshold=2)

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))
        assert cb.state == CircuitState.OPEN

        # Reset (cast breaks type narrowing)
        cb.reset()
        current_state = cast(CircuitState, cb.state)
        assert current_state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0

    def test_get_status(self):
        """get_status returns correct information."""
        cb = CircuitBreaker(name="test", failure_threshold=5)

        cb.record_failure(Exception("test"))
        cb.record_failure(Exception("test"))

        status = cb.get_status()
        assert status["name"] == "test"
        assert status["enabled"] is True
        assert status["state"] == "closed"
        assert status["failure_count"] == 2
        assert status["failure_threshold"] == 5

    def test_get_status_includes_time_until_half_open_when_open(self):
        """get_status includes time_until_half_open when circuit is OPEN."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=60.0)

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        status = cb.get_status()
        assert status["state"] == "open"
        assert "time_until_half_open" in status
        # Should be close to 60 seconds (minus small elapsed time)
        assert 55.0 <= status["time_until_half_open"] <= 60.0

    def test_record_success_noop_when_disabled(self):
        """record_success does nothing when circuit breaker is disabled."""
        cb = CircuitBreaker(name="test", enabled=False, failure_threshold=2)
        # Open the circuit (won't actually open since disabled)
        cb._state = CircuitState.HALF_OPEN
        cb._success_count = 0

        # record_success should return early without doing anything
        cb.record_success()

        # State should remain unchanged (disabled means no state tracking)
        assert cb._success_count == 0

    def test_check_timeout_noop_when_no_failures(self):
        """_check_timeout returns early when no failures have been recorded."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=0.1)

        # Manually set to OPEN without recording failures
        cb._state = CircuitState.OPEN
        cb._last_failure_time = None

        # Get state property triggers _check_timeout, but should not transition
        # because _last_failure_time is None
        assert cb.state == CircuitState.OPEN

    def test_half_open_allows_single_in_flight_probe(self, monkeypatch):
        """HALF_OPEN state only allows a single in-flight probe request."""
        # Use deterministic time control
        current_time = {"t": 1000.0}
        monkeypatch.setattr(time, "time", lambda: current_time["t"])

        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=60.0)

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        # Advance time past timeout to transition to HALF_OPEN
        current_time["t"] += 61.0
        assert cb.state == CircuitState.HALF_OPEN

        # First request should be allowed
        assert cb.allow_request() is True
        assert cb._half_open_in_flight is True

        # Second request should be rejected (probe already in flight)
        assert cb.allow_request() is False

        # After success, probe in flight flag should be cleared
        cb.record_success()
        assert cb._half_open_in_flight is False

        # Now another request should be allowed (state is still HALF_OPEN after 1 success)
        assert cb.allow_request() is True  # type: ignore[unreachable]

    def test_half_open_probe_cleared_on_failure(self, monkeypatch):
        """HALF_OPEN probe in-flight flag is cleared on failure."""
        # Use deterministic time control
        current_time = {"t": 1000.0}
        monkeypatch.setattr(time, "time", lambda: current_time["t"])

        cb = CircuitBreaker(name="test", failure_threshold=2, timeout_seconds=60.0)

        # Open the circuit
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        # Advance time past timeout to transition to HALF_OPEN
        current_time["t"] += 61.0
        assert cb.state == CircuitState.HALF_OPEN

        # Start a probe
        assert cb.allow_request() is True
        assert cb._half_open_in_flight is True

        # Failure should clear the flag and reopen circuit
        cb.record_failure(Exception("probe failed"))
        assert cb._half_open_in_flight is False
        assert cb.state == CircuitState.OPEN  # type: ignore[unreachable]

    def test_thread_safety(self):
        """Circuit breaker operations are thread-safe."""
        cb = CircuitBreaker(name="test", failure_threshold=100)
        errors = []

        def record_failures():
            try:
                for _ in range(50):
                    cb.record_failure(Exception("test"))
            except Exception as e:
                errors.append(e)

        def record_successes():
            try:
                for _ in range(50):
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_failures),
            threading.Thread(target=record_failures),
            threading.Thread(target=record_successes),
            threading.Thread(target=record_successes),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"


class TestGlobalCircuitBreaker:
    """Tests for the global circuit breaker functions."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_global_circuit_breaker()

    def teardown_method(self):
        """Clean up global state after each test."""
        reset_global_circuit_breaker()

    def test_get_circuit_breaker_creates_singleton(self):
        """get_circuit_breaker returns the same instance."""
        cb1 = get_circuit_breaker()
        cb2 = get_circuit_breaker()
        assert cb1 is cb2

    def test_get_circuit_breaker_uses_parameters_on_creation(self):
        """Parameters are used when creating the circuit breaker."""
        cb = get_circuit_breaker(failure_threshold=10, timeout_seconds=120.0)
        assert cb.failure_threshold == 10
        assert cb.timeout_seconds == 120.0

    def test_reset_global_circuit_breaker(self):
        """reset_global_circuit_breaker clears the singleton."""
        cb1 = get_circuit_breaker(failure_threshold=5)
        cb1.record_failure(Exception("test"))

        reset_global_circuit_breaker()

        cb2 = get_circuit_breaker(failure_threshold=10)
        assert cb2._failure_count == 0
        assert cb2.failure_threshold == 10
