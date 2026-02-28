"""Streaming utilities for Ollama API responses.

Provides helpers for consuming streaming chat responses from the Ollama client.
Using stream=True prevents HTTP read-timeouts on long-running generations
(e.g. qwen3 thinking mode can produce thousands of internal tokens before output).
"""

import logging
from collections.abc import Iterator
from typing import Any

import httpcore

logger = logging.getLogger(__name__)


def consume_stream(stream: Iterator[Any]) -> dict[str, Any]:
    """Consume a streaming Ollama chat response into a non-streaming-compatible dict.

    Collects content chunks and extracts token counts from the final chunk.
    Using stream=True prevents HTTP read-timeouts on long-running generations
    (the timeout resets with each received chunk).

    Args:
        stream: Iterator of ChatResponse chunks from client.chat(stream=True).

    Returns:
        Dict with 'message.content', 'prompt_eval_count', and 'eval_count',
        compatible with the non-streaming response access patterns.
    """
    content_parts: list[str] = []
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    try:
        for chunk in stream:
            if chunk.message and chunk.message.content:
                content_parts.append(chunk.message.content)
            if chunk.done:
                prompt_eval_count = getattr(chunk, "prompt_eval_count", None)
                eval_count = getattr(chunk, "eval_count", None)
    except (
        httpcore.RemoteProtocolError,
        httpcore.ReadError,
        httpcore.NetworkError,
    ) as e:
        logger.error("Ollama stream interrupted mid-response: %s", e)
        raise ConnectionError(f"Ollama stream interrupted: {e}") from e

    content = "".join(content_parts)
    logger.debug("Stream consumed: %d content chunks, %d chars", len(content_parts), len(content))

    return {
        "message": {"content": content},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
