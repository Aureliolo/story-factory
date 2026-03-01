"""Tests for streaming utilities (consume_stream and StreamTimeoutError)."""

import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import httpcore
import pytest

from src.utils.streaming import (
    _DEFAULT_INTER_CHUNK_TIMEOUT,
    _DEFAULT_WALL_CLOCK_TIMEOUT,
    StreamTimeoutError,
    consume_stream,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockMessage:
    """Mimics an Ollama ChatResponse message."""

    content: str = ""


@dataclass
class MockChunk:
    """Mimics an Ollama ChatResponse chunk from a streaming iterator."""

    message: Any = None
    done: bool = False
    prompt_eval_count: int | None = None
    eval_count: int | None = None


def _make_chunks(
    texts: list[str],
    *,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> list[MockChunk]:
    """Build a list of mock stream chunks, ending with a done=True chunk."""
    chunks: list[MockChunk] = []
    for text in texts:
        chunks.append(MockChunk(message=MockMessage(content=text)))
    # Final done chunk carries token counts
    chunks.append(
        MockChunk(
            message=MockMessage(content=""),
            done=True,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
        )
    )
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConsumeStreamBasic:
    """Tests for normal (non-timeout) stream consumption."""

    def test_consume_stream_basic(self):
        """Normal stream with multiple content chunks returns joined content."""
        chunks = _make_chunks(["Hello, ", "world", "!"], prompt_eval_count=10, eval_count=5)
        result = consume_stream(iter(chunks))

        assert result["message"]["content"] == "Hello, world!"
        assert result["prompt_eval_count"] == 10
        assert result["eval_count"] == 5

    def test_consume_stream_empty(self):
        """Stream with no content text returns empty string."""
        chunks = [
            MockChunk(message=MockMessage(content=""), done=False),
            MockChunk(message=MockMessage(content=""), done=True),
        ]
        result = consume_stream(iter(chunks))

        assert result["message"]["content"] == ""

    def test_consume_stream_none_message(self):
        """Chunks with message=None are handled gracefully."""
        chunks = [
            MockChunk(message=None, done=False),
            MockChunk(message=MockMessage(content="data"), done=False),
            MockChunk(message=None, done=True),
        ]
        result = consume_stream(iter(chunks))

        assert result["message"]["content"] == "data"

    def test_consume_stream_token_counts(self):
        """Token counts are extracted from the final done chunk."""
        chunks = _make_chunks(["ok"], prompt_eval_count=42, eval_count=99)
        result = consume_stream(iter(chunks))

        assert result["prompt_eval_count"] == 42
        assert result["eval_count"] == 99

    def test_consume_stream_token_counts_none_when_missing(self):
        """Token counts are None when not provided on the final chunk."""
        chunks = [
            MockChunk(message=MockMessage(content="text"), done=False),
            MockChunk(message=MockMessage(content=""), done=True),
        ]
        result = consume_stream(iter(chunks))

        assert result["prompt_eval_count"] is None
        assert result["eval_count"] is None


class TestStreamTimeouts:
    """Tests for inter-chunk and wall-clock timeout behaviour."""

    def test_wall_clock_timeout(self):
        """Exceeding wall_clock_timeout raises StreamTimeoutError with type='wall_clock'."""
        # Use a tiny wall-clock timeout so it triggers immediately on the second chunk.
        barrier = threading.Event()

        def slow_stream():
            """Generator that yields a chunk, waits, then yields more chunks."""
            yield MockChunk(message=MockMessage(content="a"))
            # Wait until the wall-clock window is certainly exceeded
            barrier.wait(timeout=5)
            yield MockChunk(message=MockMessage(content="b"))
            yield MockChunk(done=True)

        # Patch time.monotonic so that elapsed time jumps past the limit
        # after the first chunk is consumed.
        real_monotonic = __import__("time").monotonic
        call_count = 0

        def fake_monotonic():
            """Return mocked monotonic time that jumps to 999.0 after a few calls."""
            nonlocal call_count
            call_count += 1
            # First few calls return 0 (startup + first chunk).
            # After that, return a value well past the wall-clock limit.
            if call_count <= 3:
                return real_monotonic() - real_monotonic() + 0.0
            return 999.0

        barrier.set()  # don't actually block the iterator
        with patch("src.utils.streaming.time.monotonic", side_effect=fake_monotonic):
            with pytest.raises(StreamTimeoutError) as exc_info:
                consume_stream(iter(slow_stream()), wall_clock_timeout=1)

        assert exc_info.value.timeout_type == "wall_clock"
        assert exc_info.value.partial_content_length >= 1

    def test_inter_chunk_timeout(self):
        """Stalled stream triggers the inter-chunk watchdog."""
        received_first = threading.Event()

        def stalling_stream():
            """Generator that yields once, blocks, then tries to yield again."""
            yield MockChunk(message=MockMessage(content="start"))
            received_first.set()
            # Block long enough for the inter-chunk watchdog to fire.
            # With inter_chunk_timeout=0.05 the watchdog fires quickly.
            threading.Event().wait(timeout=2)  # just blocks
            yield MockChunk(message=MockMessage(content="late"))
            yield MockChunk(done=True)

        with pytest.raises(StreamTimeoutError) as exc_info:
            consume_stream(
                stalling_stream(),
                inter_chunk_timeout=1,
                wall_clock_timeout=60,
            )

        received_first.wait(timeout=5)
        assert exc_info.value.timeout_type == "inter_chunk"
        assert exc_info.value.partial_content_length >= 5  # "start" = 5 chars

    def test_stream_timeout_error_attributes(self):
        """StreamTimeoutError carries all expected attributes."""
        err = StreamTimeoutError(
            "test",
            partial_content_length=42,
            elapsed_seconds=7.5,
            timeout_type="wall_clock",
        )
        assert err.partial_content_length == 42
        assert err.elapsed_seconds == 7.5
        assert err.timeout_type == "wall_clock"
        assert str(err) == "test"

    def test_stream_timeout_error_defaults(self):
        """StreamTimeoutError default attribute values are sensible."""
        err = StreamTimeoutError("minimal")
        assert err.partial_content_length == 0
        assert err.elapsed_seconds == 0.0
        assert err.timeout_type == "inter_chunk"

    def test_stream_timeout_error_is_timeout_error(self):
        """StreamTimeoutError is a subclass of TimeoutError."""
        assert issubclass(StreamTimeoutError, TimeoutError)


class TestNetworkErrors:
    """Tests for httpcore network error wrapping."""

    @pytest.mark.parametrize(
        "error_cls",
        [
            httpcore.RemoteProtocolError,
            httpcore.ReadError,
            httpcore.NetworkError,
        ],
    )
    def test_network_error_wrapped(self, error_cls):
        """httpcore network errors are caught and re-raised as ConnectionError."""

        def failing_stream():
            """Generator that yields once then raises a network error."""
            yield MockChunk(message=MockMessage(content="partial"))
            raise error_cls("connection lost")

        with pytest.raises(ConnectionError, match="Ollama stream interrupted"):
            consume_stream(failing_stream())

    def test_stream_timeout_not_caught_by_network_handler(self):
        """StreamTimeoutError propagates without being caught by the httpcore handler."""

        def timeout_stream():
            """Generator that yields once then raises a StreamTimeoutError."""
            yield MockChunk(message=MockMessage(content="partial"))
            raise StreamTimeoutError(
                "stalled",
                partial_content_length=0,
                elapsed_seconds=1.0,
                timeout_type="inter_chunk",
            )

        with pytest.raises(StreamTimeoutError):
            consume_stream(timeout_stream())

    def test_unrelated_exception_propagates(self):
        """Exceptions that are not httpcore or StreamTimeoutError propagate unchanged."""

        def bad_stream():
            """Generator that yields once then raises an unrelated RuntimeError."""
            yield MockChunk(message=MockMessage(content="ok"))
            raise RuntimeError("unexpected failure")

        with pytest.raises(RuntimeError, match="unexpected failure"):
            consume_stream(bad_stream())


class TestTimeoutDefaults:
    """Tests for default and custom timeout values."""

    def test_default_timeouts(self):
        """Module-level defaults are 120s inter-chunk and 600s wall-clock."""
        assert _DEFAULT_INTER_CHUNK_TIMEOUT == 120
        assert _DEFAULT_WALL_CLOCK_TIMEOUT == 600

    def test_custom_timeouts_are_used(self):
        """Custom timeout values are passed through to the watchdog logic."""
        # We verify by patching threading.Timer to capture the interval argument.
        recorded_intervals: list[float] = []

        class SpyTimer(threading.Timer):
            """Timer subclass that records interval values for testing."""

            def __init__(self, interval, function, *args, **kwargs):
                """Initialize timer and record the interval."""
                recorded_intervals.append(interval)
                super().__init__(interval, function, *args, **kwargs)

        chunks = _make_chunks(["a", "b"])

        with patch("src.utils.streaming.threading.Timer", SpyTimer):
            consume_stream(
                iter(chunks),
                inter_chunk_timeout=77,
                wall_clock_timeout=999,
            )

        # Every Timer should have been created with the custom inter-chunk value.
        assert all(iv == 77 for iv in recorded_intervals)
        assert len(recorded_intervals) >= 1

    def test_none_timeouts_use_defaults(self):
        """Passing None for timeouts falls back to module-level defaults."""
        recorded_intervals: list[float] = []

        class SpyTimer(threading.Timer):
            """Timer subclass that records interval values for testing."""

            def __init__(self, interval, function, *args, **kwargs):
                """Initialize timer and record the interval."""
                recorded_intervals.append(interval)
                super().__init__(interval, function, *args, **kwargs)

        chunks = _make_chunks(["x"])

        with patch("src.utils.streaming.threading.Timer", SpyTimer):
            consume_stream(
                iter(chunks),
                inter_chunk_timeout=None,
                wall_clock_timeout=None,
            )

        assert all(iv == _DEFAULT_INTER_CHUNK_TIMEOUT for iv in recorded_intervals)


class TestWatchdogCleanup:
    """Tests that the watchdog timer is cleaned up in all code paths."""

    def test_timer_cancelled_on_success(self):
        """Watchdog timer is cancelled after successful stream consumption."""
        cancel_calls = 0

        class TrackingTimer(threading.Timer):
            """Timer subclass that tracks cancel() calls for testing."""

            def cancel(self):
                """Cancel timer and increment the call counter."""
                nonlocal cancel_calls
                cancel_calls += 1
                super().cancel()

        chunks = _make_chunks(["ok"])

        with patch("src.utils.streaming.threading.Timer", TrackingTimer):
            consume_stream(iter(chunks))

        # At least one cancel in the finally block
        assert cancel_calls >= 1

    def test_timer_cancelled_on_error(self):
        """Watchdog timer is cancelled even when an exception occurs."""
        cancel_calls = 0

        class TrackingTimer(threading.Timer):
            """Timer subclass that tracks cancel() calls for testing."""

            def cancel(self):
                """Cancel timer and increment the call counter."""
                nonlocal cancel_calls
                cancel_calls += 1
                super().cancel()

        def failing_stream():
            """Generator that yields once then raises a ReadError."""
            yield MockChunk(message=MockMessage(content="x"))
            raise httpcore.ReadError("oops")

        with patch("src.utils.streaming.threading.Timer", TrackingTimer):
            with pytest.raises(ConnectionError):
                consume_stream(failing_stream())

        assert cancel_calls >= 1
