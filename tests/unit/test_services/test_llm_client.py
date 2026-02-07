"""Tests for services/llm_client.py."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.services import llm_client
from src.services.llm_client import generate_structured, get_ollama_client
from src.settings import Settings


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
    """Clear the Ollama client cache before each test."""
    llm_client._ollama_clients.clear()
    yield
    llm_client._ollama_clients.clear()


def _make_chat_response(json_content: str) -> dict:
    """Create a mock Ollama chat response.

    Args:
        json_content: JSON string for the response content.

    Returns:
        Dict matching Ollama ChatResponse format.
    """
    return {
        "message": {"content": json_content, "role": "assistant"},
        "done": True,
        "prompt_eval_count": 100,
        "eval_count": 50,
    }


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
    def test_qwen_model_adds_no_think_prefix(self, mock_get_client, mock_settings):
        """Test that Qwen models get /no_think prefix in system prompt."""
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
        assert messages[0]["content"] == "/no_think\nOriginal system prompt"

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
