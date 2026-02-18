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
    return settings


@pytest.fixture(autouse=True)
def clear_client_cache():
    """Clear the Ollama client and model context caches before each test."""
    llm_client._ollama_clients.clear()
    _model_context_cache.clear()
    yield
    llm_client._ollama_clients.clear()
    _model_context_cache.clear()


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


class TestGetModelContextSize:
    """Tests for get_model_context_size function."""

    def test_get_model_context_size_returns_context_length(self):
        """Test that the context length is extracted from model_info."""
        mock_client = MagicMock()
        mock_client.show.return_value = {
            "model_info": {"some.context_length": 8192},
        }

        result = get_model_context_size(mock_client, "test-model:8b")

        assert result == 8192

    def test_get_model_context_size_caches_result(self):
        """Test that the context size is cached and client.show is called only once."""
        mock_client = MagicMock()
        mock_client.show.return_value = {
            "model_info": {"llama.context_length": 4096},
        }

        result1 = get_model_context_size(mock_client, "cached-model:8b")
        result2 = get_model_context_size(mock_client, "cached-model:8b")

        assert result1 == 4096
        assert result2 == 4096
        mock_client.show.assert_called_once()

    def test_get_model_context_size_returns_none_on_error(self):
        """Test that None is returned when client.show raises an exception.

        Transient errors should NOT be cached so the next call can retry.
        """
        mock_client = MagicMock()
        mock_client.show.side_effect = Exception("Connection refused")

        result = get_model_context_size(mock_client, "error-model:8b")

        assert result is None
        # Transient errors must not be cached — the key should be absent
        assert "error-model:8b" not in _model_context_cache

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
        mock_client.show.return_value = {
            "model_info": {"llama.context_length": 4096},
        }

        with caplog.at_level(logging.WARNING, logger="src.services.llm_client"):
            result = validate_context_size(mock_client, "small-model:8b", 32768)

        assert result == 4096
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert any("Capping to model limit" in r.message for r in warning_records)

    def test_validate_context_size_returns_configured_when_larger(self):
        """Test that configured value is returned when model limit is larger."""
        mock_client = MagicMock()
        mock_client.show.return_value = {
            "model_info": {"llama.context_length": 131072},
        }

        result = validate_context_size(mock_client, "big-model:8b", 32768)

        assert result == 32768

    def test_validate_context_size_returns_configured_when_none(self):
        """Test that configured value is returned when model limit is unknown (None)."""
        mock_client = MagicMock()
        mock_client.show.side_effect = Exception("Model not found")

        result = validate_context_size(mock_client, "unknown-model:8b", 32768)

        assert result == 32768
