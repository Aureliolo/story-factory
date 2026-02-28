"""Tests for src/utils/streaming.py."""

import logging
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from src.utils.streaming import consume_stream
from tests.shared.mock_ollama import MockStreamChunk


class TestConsumeStream:
    """Tests for consume_stream function."""

    def test_single_chunk_with_content(self):
        """Test consuming a stream with a single content chunk."""
        stream = iter(
            [
                MockStreamChunk(
                    content="Hello world", done=True, prompt_eval_count=10, eval_count=5
                ),
            ]
        )

        result = consume_stream(stream)

        assert result["message"]["content"] == "Hello world"
        assert result["prompt_eval_count"] == 10
        assert result["eval_count"] == 5

    def test_multiple_chunks_concatenated(self):
        """Test that multiple content chunks are concatenated."""
        stream = iter(
            [
                MockStreamChunk(content="Hello "),
                MockStreamChunk(content="world"),
                MockStreamChunk(content="!", done=True, prompt_eval_count=20, eval_count=10),
            ]
        )

        result = consume_stream(stream)

        assert result["message"]["content"] == "Hello world!"
        assert result["prompt_eval_count"] == 20
        assert result["eval_count"] == 10

    def test_empty_stream(self):
        """Test consuming an empty stream produces empty content."""
        stream: Iterator[MockStreamChunk] = iter([])

        result = consume_stream(stream)

        assert result["message"]["content"] == ""
        assert result["prompt_eval_count"] is None
        assert result["eval_count"] is None

    def test_chunk_with_empty_content_skipped(self):
        """Test that chunks with empty content are skipped."""
        stream = iter(
            [
                MockStreamChunk(content=""),
                MockStreamChunk(
                    content="actual content", done=True, prompt_eval_count=5, eval_count=3
                ),
            ]
        )

        result = consume_stream(stream)

        assert result["message"]["content"] == "actual content"

    def test_token_counts_only_from_done_chunk(self):
        """Test that token counts are extracted only from the final (done=True) chunk."""
        stream = iter(
            [
                MockStreamChunk(content="part1"),
                MockStreamChunk(content="part2", done=True, prompt_eval_count=100, eval_count=50),
            ]
        )

        result = consume_stream(stream)

        assert result["prompt_eval_count"] == 100
        assert result["eval_count"] == 50

    def test_no_done_chunk_returns_none_counts(self):
        """Test that missing done chunk results in None token counts."""
        stream = iter(
            [
                MockStreamChunk(content="content"),
            ]
        )

        result = consume_stream(stream)

        assert result["message"]["content"] == "content"
        assert result["prompt_eval_count"] is None
        assert result["eval_count"] is None

    def test_logs_stream_consumption(self, caplog):
        """Test that stream consumption is logged at DEBUG level."""
        stream = iter(
            [
                MockStreamChunk(content="chunk1"),
                MockStreamChunk(content="chunk2", done=True),
            ]
        )

        with caplog.at_level(logging.DEBUG, logger="src.utils.streaming"):
            consume_stream(stream)

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("Stream consumed" in r.message for r in debug_records)
        assert any("2 content chunks" in r.message for r in debug_records)

    def test_none_message_content_skipped(self):
        """Test that chunks with None message content are skipped."""
        chunk = MockStreamChunk(content="real content", done=True)
        # Create a chunk with None content
        none_chunk = MagicMock()
        none_chunk.message = MagicMock()
        none_chunk.message.content = None
        none_chunk.done = False

        stream = iter([none_chunk, chunk])

        result = consume_stream(stream)

        assert result["message"]["content"] == "real content"

    def test_result_dict_structure(self):
        """Test that the result dict has the expected keys."""
        stream = iter(
            [
                MockStreamChunk(content="test", done=True, prompt_eval_count=1, eval_count=2),
            ]
        )

        result = consume_stream(stream)

        assert "message" in result
        assert "content" in result["message"]
        assert "prompt_eval_count" in result
        assert "eval_count" in result


class TestConsumeStreamHttpcoreErrors:
    """Tests for httpcore exception handling in consume_stream (lines 42-48)."""

    def _make_failing_stream(self, exc: Exception):
        """Return an iterator that raises exc on the first iteration."""

        def _gen():
            raise exc
            yield  # type: ignore[unreachable]  # makes _gen a generator

        return _gen()

    def test_remote_protocol_error_raises_connection_error(self, caplog):
        """httpcore.RemoteProtocolError mid-stream raises ConnectionError."""
        import httpcore

        exc = httpcore.RemoteProtocolError("connection broken")
        stream = self._make_failing_stream(exc)

        with caplog.at_level(logging.ERROR, logger="src.utils.streaming"):
            with pytest.raises(ConnectionError, match="Ollama stream interrupted"):
                consume_stream(stream)

        assert any("stream interrupted" in m.lower() for m in caplog.messages)

    def test_read_error_raises_connection_error(self, caplog):
        """httpcore.ReadError mid-stream raises ConnectionError."""
        import httpcore

        exc = httpcore.ReadError("read timed out")
        stream = self._make_failing_stream(exc)

        with caplog.at_level(logging.ERROR, logger="src.utils.streaming"):
            with pytest.raises(ConnectionError, match="Ollama stream interrupted"):
                consume_stream(stream)

        assert any("stream interrupted" in m.lower() for m in caplog.messages)

    def test_network_error_raises_connection_error(self, caplog):
        """httpcore.NetworkError mid-stream raises ConnectionError."""
        import httpcore

        exc = httpcore.NetworkError("network unreachable")
        stream = self._make_failing_stream(exc)

        with caplog.at_level(logging.ERROR, logger="src.utils.streaming"):
            with pytest.raises(ConnectionError, match="Ollama stream interrupted"):
                consume_stream(stream)

        assert any("stream interrupted" in m.lower() for m in caplog.messages)

    def test_connection_error_preserves_original_exception_as_cause(self):
        """The raised ConnectionError chains the original httpcore exception."""
        import httpcore

        original_exc = httpcore.RemoteProtocolError("original error")
        stream = self._make_failing_stream(original_exc)

        with pytest.raises(ConnectionError) as exc_info:
            consume_stream(stream)

        assert exc_info.value.__cause__ is original_exc

    def test_partial_content_lost_on_error(self, caplog):
        """Content collected before the error is not returned (exception is raised)."""
        import httpcore

        def _partial_stream():
            yield MockStreamChunk(content="partial ")
            raise httpcore.ReadError("dropped mid-stream")

        with caplog.at_level(logging.ERROR, logger="src.utils.streaming"):
            with pytest.raises(ConnectionError):
                consume_stream(_partial_stream())

    def test_error_message_includes_original_exc(self):
        """ConnectionError message includes the original exception text."""
        import httpcore

        exc = httpcore.NetworkError("host unreachable")
        stream = self._make_failing_stream(exc)

        with pytest.raises(ConnectionError, match="host unreachable"):
            consume_stream(stream)
