"""Streaming utilities for Ollama API responses.

Provides helpers for consuming streaming chat responses from the Ollama client.
Using stream=True prevents HTTP read-timeouts on long-running generations
(e.g. qwen3 thinking mode can produce thousands of internal tokens before output).

Includes inter-chunk and wall-clock watchdog timeouts to prevent indefinite
blocking when Ollama inference stalls.
"""

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpcore

logger = logging.getLogger(__name__)

# Default timeouts (overridden by settings when available)
_DEFAULT_INTER_CHUNK_TIMEOUT = 120  # seconds between chunks
_DEFAULT_WALL_CLOCK_TIMEOUT = 600  # 10 minutes absolute max


class StreamTimeoutError(TimeoutError):
    """Raised when a streaming response exceeds the configured timeout.

    Attributes:
        partial_content_length: Number of characters received before timeout.
        elapsed_seconds: Wall-clock time elapsed before timeout.
        timeout_type: Either "inter_chunk" or "wall_clock".
    """

    def __init__(
        self,
        message: str,
        *,
        partial_content_length: int = 0,
        elapsed_seconds: float = 0.0,
        timeout_type: str = "inter_chunk",
    ):
        super().__init__(message)
        self.partial_content_length = partial_content_length
        self.elapsed_seconds = elapsed_seconds
        self.timeout_type = timeout_type


def consume_stream(
    stream: Iterator[Any],
    *,
    inter_chunk_timeout: int | None = None,
    wall_clock_timeout: int | None = None,
) -> dict[str, Any]:
    """Consume a streaming Ollama chat response into a non-streaming-compatible dict.

    Collects content chunks and extracts token counts from the final chunk.
    Using stream=True prevents HTTP read-timeouts on long-running generations
    (the timeout resets with each received chunk).

    Includes two watchdog timeouts:
    - ``inter_chunk_timeout``: max seconds between consecutive chunks. If no
      chunk arrives within this window, the stream is considered stalled.
    - ``wall_clock_timeout``: absolute max seconds for the entire stream.

    Args:
        stream: Iterator of ChatResponse chunks from client.chat(stream=True).
        inter_chunk_timeout: Max seconds between chunks (None = default 120s).
        wall_clock_timeout: Max total seconds for the stream (None = default 600s).

    Returns:
        Dict with 'message.content', 'prompt_eval_count', and 'eval_count',
        compatible with the non-streaming response access patterns.

    Raises:
        StreamTimeoutError: If inter-chunk or wall-clock timeout is exceeded.
        ConnectionError: If the stream is interrupted by a network error.
    """
    inter_chunk = (
        inter_chunk_timeout if inter_chunk_timeout is not None else _DEFAULT_INTER_CHUNK_TIMEOUT
    )
    wall_clock = (
        wall_clock_timeout if wall_clock_timeout is not None else _DEFAULT_WALL_CLOCK_TIMEOUT
    )

    content_parts: list[str] = []
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    # Watchdog: fires if no chunk received within inter_chunk seconds.
    # Uses a threading.Event that the main loop resets on each chunk.
    watchdog_fired = threading.Event()
    watchdog_timer: threading.Timer | None = None

    def _watchdog_callback() -> None:
        """Called when the inter-chunk timer expires."""
        watchdog_fired.set()

    def _reset_watchdog() -> None:
        """Cancel the current timer and start a new one."""
        nonlocal watchdog_timer
        if watchdog_timer is not None:
            watchdog_timer.cancel()
        watchdog_timer = threading.Timer(inter_chunk, _watchdog_callback)
        watchdog_timer.daemon = True
        watchdog_timer.start()

    start_time = time.monotonic()
    _reset_watchdog()

    try:
        for chunk in stream:
            # Check wall-clock timeout
            elapsed = time.monotonic() - start_time
            if elapsed > wall_clock:
                content_len = sum(len(p) for p in content_parts)
                logger.error(
                    "Stream wall-clock timeout after %.1fs (limit=%ds, partial_content=%d chars)",
                    elapsed,
                    wall_clock,
                    content_len,
                )
                raise StreamTimeoutError(
                    f"Stream exceeded wall-clock timeout of {wall_clock}s "
                    f"(elapsed={elapsed:.1f}s, partial_content={content_len} chars)",
                    partial_content_length=content_len,
                    elapsed_seconds=elapsed,
                    timeout_type="wall_clock",
                )

            # Check inter-chunk watchdog
            if watchdog_fired.is_set():
                content_len = sum(len(p) for p in content_parts)
                logger.error(
                    "Stream inter-chunk timeout: no chunk received for %ds "
                    "(partial_content=%d chars, elapsed=%.1fs)",
                    inter_chunk,
                    content_len,
                    elapsed,
                )
                raise StreamTimeoutError(
                    f"No stream chunk received for {inter_chunk}s "
                    f"(partial_content={content_len} chars)",
                    partial_content_length=content_len,
                    elapsed_seconds=elapsed,
                    timeout_type="inter_chunk",
                )

            # Reset watchdog on each received chunk
            _reset_watchdog()

            if chunk.message and chunk.message.content:
                content_parts.append(chunk.message.content)
            if chunk.done:
                prompt_eval_count = getattr(chunk, "prompt_eval_count", None)
                eval_count = getattr(chunk, "eval_count", None)
    except StreamTimeoutError:
        raise
    except (
        httpcore.RemoteProtocolError,
        httpcore.ReadError,
        httpcore.NetworkError,
    ) as e:
        logger.error("Ollama stream interrupted mid-response: %s", e)
        raise ConnectionError(f"Ollama stream interrupted: {e}") from e
    finally:
        if watchdog_timer is not None:
            watchdog_timer.cancel()

    elapsed = time.monotonic() - start_time
    content = "".join(content_parts)
    logger.debug(
        "Stream consumed: %d content chunks, %d chars, %.2fs",
        len(content_parts),
        len(content),
        elapsed,
    )

    return {
        "message": {"content": content},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
