"""Tests for services/llm_client.py."""

import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import ollama
import pytest
from pydantic import BaseModel

from src.services import llm_client
from src.services.llm_client import (
    _model_context_cache,
    _warned_context_size_models,
    generate_structured,
    get_model_context_size,
    get_ollama_client,
    validate_context_size,
)
from src.settings import Settings
from src.utils.exceptions import LLMError


class SampleModel(BaseModel):
    """Test response model."""

    name: str
    value: int


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_timeout = 120
    settings.context_size = 32768
    settings.streaming_inter_chunk_timeout = 120
    settings.streaming_wall_clock_timeout = 600
    return settings


@pytest.fixture(autouse=True)
def clear_client_cache():
    """Clear the Ollama client, model context, and warning caches before each test."""
    llm_client._ollama_clients.clear()
    _model_context_cache.clear()
    _warned_context_size_models.clear()
    yield
    llm_client._ollama_clients.clear()
    _model_context_cache.clear()
    _warned_context_size_models.clear()


def _make_chat_response(json_content: str) -> Iterator:
    """Create a mock streaming Ollama chat response.

    Returns an iterator of MockStreamChunk objects compatible with consume_stream().

    Args:
        json_content: JSON string for the response content.

    Returns:
        Iterator of stream chunks matching Ollama streaming format.
    """
    from tests.shared.mock_ollama import MockStreamChunk

    return iter(
        [
            MockStreamChunk(
                content=json_content,
                done=True,
                prompt_eval_count=100,
                eval_count=50,
            ),
        ]
    )


class TestGetOllamaClient:
    """Tests for get_ollama_client function."""

    def test_creates_new_client(self, mock_settings):
        """Test that a new client is created when cache is empty."""
        client = get_ollama_client(mock_settings)

        assert client is not None
        assert len(llm_client._ollama_clients) == 1

    def test_caches_client(self, mock_settings):
        """Test that the client is cached and reused."""
        client1 = get_ollama_client(mock_settings)
        client2 = get_ollama_client(mock_settings)

        assert client1 is client2
        assert len(llm_client._ollama_clients) == 1

    def test_different_settings_create_different_clients(self):
        """Test that different settings create different cached clients."""
        settings1 = MagicMock(spec=Settings)
        settings1.ollama_url = "http://localhost:11434"
        settings1.ollama_timeout = 120

        settings2 = MagicMock(spec=Settings)
        settings2.ollama_url = "http://localhost:11434"
        settings2.ollama_timeout = 300  # Different timeout → different cache key

        client1 = get_ollama_client(settings1)
        client2 = get_ollama_client(settings2)

        assert client1 is not client2
        assert len(llm_client._ollama_clients) == 2


class TestGenerateStructured:
    """Tests for generate_structured function."""

    @patch("src.services.llm_client.get_ollama_client")
    def test_basic_generation(self, mock_get_client, mock_settings):
        """Test basic structured generation."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="Generate something",
            response_model=SampleModel,
        )

        assert result.name == "test"
        assert result.value == 42
        mock_client.chat.assert_called_once()

    @patch("src.services.llm_client.get_ollama_client")
    def test_includes_system_prompt(self, mock_get_client, mock_settings):
        """Test that system prompt is included in messages."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt="System instructions",
        )

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instructions"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    @patch("src.services.llm_client.get_ollama_client")
    def test_no_system_prompt(self, mock_get_client, mock_settings):
        """Test generation without system prompt."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
        )

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("src.services.llm_client.get_ollama_client")
    def test_qwen_model_no_think_not_added(self, mock_get_client, mock_settings):
        """Test that Qwen models do NOT get /no_think prefix in system prompt."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="qwen2.5:14b",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt="Original system prompt",
        )

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Original system prompt"

    @patch("src.services.llm_client.get_ollama_client")
    def test_qwen_model_no_system_prompt_no_change(self, mock_get_client, mock_settings):
        """Test that Qwen models without system prompt don't crash."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="qwen2.5:14b",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt=None,
        )

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        # Only user message, no system message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("src.services.llm_client.get_ollama_client")
    def test_non_qwen_model_no_prefix(self, mock_get_client, mock_settings):
        """Test that non-Qwen models don't get /no_think prefix."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="llama3:8b",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt="Original system prompt",
        )

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["content"] == "Original system prompt"

    @patch("src.services.llm_client.get_ollama_client")
    def test_custom_temperature(self, mock_get_client, mock_settings):
        """Test that custom temperature is passed through."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            temperature=0.9,
        )

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.9

    @patch("src.services.llm_client.get_ollama_client")
    def test_passes_format_schema(self, mock_get_client, mock_settings):
        """Test that JSON schema is passed via format parameter."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
        )

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["format"] == SampleModel.model_json_schema()
        assert call_kwargs["model"] == "test-model"

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_validation_error(self, mock_get_client, mock_settings):
        """Test that validation errors trigger retries."""
        mock_client = MagicMock()
        # First call returns invalid JSON, second returns valid
        mock_client.chat.side_effect = [
            _make_chat_response('{"invalid_field": "bad"}'),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert result.value == 42
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_raises_after_retries_exhausted(self, mock_get_client, mock_settings):
        """Test that LLMError is raised when all retries fail with validation errors."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            _make_chat_response('{"invalid": "bad"}'),
            _make_chat_response('{"also_invalid": "bad"}'),
            _make_chat_response('{"still_invalid": "bad"}'),
        ]
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMError, match="Structured generation failed"):
            generate_structured(
                settings=mock_settings,
                model="test-model",
                prompt="User prompt",
                response_model=SampleModel,
                max_retries=3,
            )

        assert mock_client.chat.call_count == 3

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_connection_error(self, mock_get_client, mock_settings):
        """Test that ConnectionError triggers retries."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ConnectionError("Connection refused"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_timeout_error(self, mock_get_client, mock_settings):
        """Test that TimeoutError triggers retries."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            TimeoutError("Request timed out"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_httpx_timeout(self, mock_get_client, mock_settings):
        """Test that httpx.TimeoutException triggers retries (mid-stream read timeout)."""
        import httpx

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            httpx.ReadTimeout("Read timed out"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_httpx_transport_error(self, mock_get_client, mock_settings):
        """Test that httpx.TransportError triggers retries (mid-stream disconnect)."""
        import httpx

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            httpx.RemoteProtocolError("Server disconnected"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_httpcore_remote_protocol_error(self, mock_get_client, mock_settings):
        """Test that httpcore.RemoteProtocolError triggers retries."""
        import httpcore

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            httpcore.RemoteProtocolError("peer closed connection"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_httpcore_read_error(self, mock_get_client, mock_settings):
        """Test that httpcore.ReadError triggers retries."""
        import httpcore

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            httpcore.ReadError("read operation failed"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_retries_on_httpcore_network_error(self, mock_get_client, mock_settings):
        """Test that httpcore.NetworkError triggers retries."""
        import httpcore

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            httpcore.NetworkError("network unreachable"),
            _make_chat_response('{"name": "test", "value": 42}'),
        ]
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=2,
        )

        assert result.name == "test"
        assert mock_client.chat.call_count == 2

    @patch("src.services.llm_client.get_ollama_client")
    def test_response_error_fails_fast(self, mock_get_client, mock_settings):
        """Test that ollama.ResponseError does not retry."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ollama.ResponseError("model not found")
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMError, match="Structured generation failed"):
            generate_structured(
                settings=mock_settings,
                model="test-model",
                prompt="User prompt",
                response_model=SampleModel,
                max_retries=3,
            )

        assert mock_client.chat.call_count == 1

    @patch("src.services.llm_client.get_ollama_client")
    def test_max_retries_zero_raises_value_error(self, mock_get_client, mock_settings):
        """Test that max_retries=0 raises ValueError."""
        mock_get_client.return_value = MagicMock()

        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            generate_structured(
                settings=mock_settings,
                model="test-model",
                prompt="User prompt",
                response_model=SampleModel,
                max_retries=0,
            )

    @patch("src.services.llm_client.get_ollama_client")
    def test_logs_model_and_tokens_at_info(self, mock_get_client, mock_settings, caplog):
        """Test that successful generation logs model name and token counts at INFO."""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"name": "test", "value": 42}')
        mock_get_client.return_value = mock_client

        with caplog.at_level(logging.INFO, logger="src.services.llm_client"):
            generate_structured(
                settings=mock_settings,
                model="test-model:8b",
                prompt="Generate something",
                response_model=SampleModel,
            )

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("test-model:8b" in r.message for r in info_records)
        assert any("tokens:" in r.message.lower() for r in info_records)
        assert any("SampleModel" in r.message for r in info_records)

    @patch("src.services.llm_client.consume_stream")
    @patch("src.services.llm_client.validate_context_size", return_value=4096)
    @patch("src.services.llm_client.get_ollama_client")
    def test_generate_structured_uses_validated_context_size(
        self, mock_get_client, mock_validate, mock_consume, mock_settings
    ):
        """Test that generate_structured passes the validated context size as num_ctx."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_consume.return_value = {
            "message": {"content": '{"name": "test", "value": 42}'},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        class TestResponse(BaseModel):
            """Test Pydantic model for structured generation."""

            name: str
            value: int

        result = generate_structured(
            settings=mock_settings,
            model="test-model:7b",
            prompt="test prompt",
            response_model=TestResponse,
        )

        assert result.name == "test"
        assert result.value == 42

        # Verify num_ctx was set to 4096 (the value returned by validate_context_size)
        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["options"]["num_ctx"] == 4096

        # Verify validate_context_size was called with the client, model, and configured size
        mock_validate.assert_called_once_with(
            mock_client, "test-model:7b", mock_settings.context_size
        )


def _make_show_response(model_info: dict | None = None) -> MagicMock:
    """Create a mock ShowResponse with a modelinfo attribute.

    ollama.ShowResponse stores model info in the ``modelinfo`` Python attribute
    (the JSON alias is ``model_info``).  The production code accesses
    ``info.modelinfo``, so mocks must expose the same attribute.
    """
    resp = MagicMock()
    resp.modelinfo = model_info
    return resp


class TestGetModelContextSize:
    """Tests for get_model_context_size function."""

    def test_get_model_context_size_returns_context_length(self):
        """Test that the context length is extracted from modelinfo."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"some.context_length": 8192},
        )

        result = get_model_context_size(mock_client, "test-model:8b")

        assert result == 8192

    def test_get_model_context_size_caches_result(self):
        """Test that the context size is cached and client.show is called only once."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": 4096},
        )

        result1 = get_model_context_size(mock_client, "cached-model:8b")
        result2 = get_model_context_size(mock_client, "cached-model:8b")

        assert result1 == 4096
        assert result2 == 4096
        mock_client.show.assert_called_once()

    def test_get_model_context_size_returns_none_when_modelinfo_is_none(self):
        """Test that modelinfo=None (the Pydantic default) returns None via or {} fallback."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(None)

        result = get_model_context_size(mock_client, "nil-info-model:8b")

        assert result is None

    def test_get_model_context_size_caches_none_on_network_error(self):
        """Test that None is cached when client.show raises a network error.

        Caching None prevents repeated API calls on every embed_text() call.
        """
        mock_client = MagicMock()
        mock_client.show.side_effect = ConnectionError("Connection refused")

        result = get_model_context_size(mock_client, "error-model:8b")

        assert result is None
        # None should be cached so subsequent calls don't retry
        assert "error-model:8b" in _model_context_cache
        assert _model_context_cache["error-model:8b"] is None

    def test_get_model_context_size_no_retry_after_cached_none(self):
        """Test that after network error caches None, subsequent call returns from cache."""
        mock_client = MagicMock()
        mock_client.show.side_effect = ConnectionError("Connection refused")

        get_model_context_size(mock_client, "retry-model:8b")
        mock_client.show.assert_called_once()

        # Second call should return cached None without calling show() again
        result = get_model_context_size(mock_client, "retry-model:8b")
        assert result is None
        mock_client.show.assert_called_once()  # Still only one call

    def test_get_model_context_size_does_not_cache_unexpected_errors(self):
        """Test that unexpected errors (e.g., ValueError) are NOT cached.

        Unexpected errors may be transient programming issues — allow retry.
        """
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": "not_a_number"},
        )

        result = get_model_context_size(mock_client, "bad-meta-model:8b")
        assert result is None
        # Should NOT be cached — next call should retry
        assert "bad-meta-model:8b" not in _model_context_cache

    def test_get_model_context_size_caches_none_on_response_error(self):
        """Test that ollama.ResponseError is cached as None (model doesn't support query)."""
        mock_client = MagicMock()
        mock_client.show.side_effect = ollama.ResponseError("model not found")

        result = get_model_context_size(mock_client, "missing-model:8b")
        assert result is None
        # ResponseError should be cached — model genuinely doesn't support it
        assert "missing-model:8b" in _model_context_cache
        assert _model_context_cache["missing-model:8b"] is None

    def test_get_model_context_size_caches_none_on_timeout_error(self):
        """Test that TimeoutError is cached as None (network error)."""
        mock_client = MagicMock()
        mock_client.show.side_effect = TimeoutError("Connection timed out")

        result = get_model_context_size(mock_client, "timeout-model:8b")
        assert result is None
        assert "timeout-model:8b" in _model_context_cache
        assert _model_context_cache["timeout-model:8b"] is None

    def test_get_model_context_size_caches_none_on_httpx_timeout(self):
        """Test that httpx.TimeoutException is cached as None (network error)."""
        import httpx

        mock_client = MagicMock()
        mock_client.show.side_effect = httpx.ReadTimeout("Read timed out")

        result = get_model_context_size(mock_client, "httpx-timeout-model:8b")
        assert result is None
        assert "httpx-timeout-model:8b" in _model_context_cache
        assert _model_context_cache["httpx-timeout-model:8b"] is None

    def test_get_model_context_size_caches_none_on_httpx_transport_error(self):
        """Test that httpx.TransportError is cached as None (network error)."""
        import httpx

        mock_client = MagicMock()
        mock_client.show.side_effect = httpx.RemoteProtocolError("Server disconnected")

        result = get_model_context_size(mock_client, "httpx-transport-model:8b")
        assert result is None
        assert "httpx-transport-model:8b" in _model_context_cache
        assert _model_context_cache["httpx-transport-model:8b"] is None

    def test_get_model_context_size_double_check_lock(self):
        """Test double-checked locking when cache is populated during lock acquisition."""
        mock_client = MagicMock()
        real_lock = llm_client._model_context_cache_lock

        class InjectingLock:
            """Lock wrapper that injects a cache entry on acquire."""

            def __enter__(self):
                """Populate cache before acquiring, simulating a concurrent thread."""
                _model_context_cache["race-model:8b"] = 4096
                return real_lock.__enter__()

            def __exit__(self, *args):
                """Release the underlying lock."""
                return real_lock.__exit__(*args)

        with patch.object(llm_client, "_model_context_cache_lock", InjectingLock()):
            result = get_model_context_size(mock_client, "race-model:8b")

        assert result == 4096
        mock_client.show.assert_not_called()


class TestValidateContextSize:
    """Tests for validate_context_size function."""

    def test_validate_context_size_caps_to_model_limit(self, caplog):
        """Test that configured value is capped to model limit when model limit is smaller."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": 4096},
        )

        with caplog.at_level(logging.WARNING, logger="src.services.llm_client"):
            result = validate_context_size(mock_client, "small-model:8b", 32768)

        assert result == 4096
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert any("Capping to model limit" in r.message for r in warning_records)

    def test_validate_context_size_returns_configured_when_larger(self):
        """Test that configured value is returned when model limit is larger."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": 131072},
        )

        result = validate_context_size(mock_client, "big-model:8b", 32768)

        assert result == 32768

    def test_validate_context_size_returns_configured_when_none(self):
        """Test that configured value is returned when model limit is unknown (None)."""
        mock_client = MagicMock()
        mock_client.show.side_effect = Exception("Model not found")

        result = validate_context_size(mock_client, "unknown-model:8b", 32768)

        assert result == 32768

    def test_validate_context_size_warns_only_once_per_model(self, caplog):
        """Second call with same model should not emit a warning."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": 4096},
        )

        with caplog.at_level(logging.WARNING, logger="src.services.llm_client"):
            validate_context_size(mock_client, "warn-once-model:8b", 32768)

        assert any("Capping to model limit" in r.message for r in caplog.records)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="src.services.llm_client"):
            result = validate_context_size(mock_client, "warn-once-model:8b", 32768)

        assert result == 4096
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0

    def test_validate_context_size_warns_for_each_model(self, caplog):
        """Different models should each get their own warning."""
        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": 4096},
        )

        with caplog.at_level(logging.WARNING, logger="src.services.llm_client"):
            validate_context_size(mock_client, "model-a:8b", 32768)
            validate_context_size(mock_client, "model-b:8b", 32768)

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Capping to model limit" in r.message
        ]
        assert len(warning_records) == 2

    def test_validate_context_size_concurrent_warns_once(self):
        """Concurrent calls for the same model should emit at most one warning."""
        import threading

        mock_client = MagicMock()
        mock_client.show.return_value = _make_show_response(
            {"llama.context_length": 4096},
        )

        warnings_logged: list[str] = []
        original_warning = logging.getLogger("src.services.llm_client").warning

        def capture_warning(msg, *args, **kwargs):
            """Intercept logger.warning calls and record formatted messages."""
            warnings_logged.append(msg % args if args else msg)
            original_warning(msg, *args, **kwargs)

        barrier = threading.Barrier(4)
        results: list[int] = []

        def worker():
            """Call validate_context_size from a thread, syncing via barrier."""
            barrier.wait()
            r = validate_context_size(mock_client, "concurrent-model:8b", 32768)
            results.append(r)

        with patch.object(
            logging.getLogger("src.services.llm_client"), "warning", side_effect=capture_warning
        ):
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        # Ensure all worker threads completed; fail fast on partial execution
        assert all(not t.is_alive() for t in threads), "All worker threads must finish"
        assert len(results) == 4, "Expected one result per worker thread"
        # All threads should return the capped value
        assert all(r == 4096 for r in results)
        # Warning should be emitted at most once
        cap_warnings = [w for w in warnings_logged if "Capping to model limit" in w]
        assert len(cap_warnings) == 1
