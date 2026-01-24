"""Tests for services/llm_client.py."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.services import llm_client
from src.services.llm_client import generate_structured, get_instructor_client
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
    return settings


@pytest.fixture(autouse=True)
def clear_client_cache():
    """Clear the instructor client cache before each test."""
    llm_client._instructor_clients.clear()
    yield
    llm_client._instructor_clients.clear()


class TestGetInstructorClient:
    """Tests for get_instructor_client function."""

    @patch("src.services.llm_client.instructor")
    @patch("src.services.llm_client.OpenAI")
    def test_creates_new_client(self, mock_openai_class, mock_instructor, mock_settings):
        """Test that a new client is created when cache is empty."""
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance
        mock_instructor_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_instructor_client

        client = get_instructor_client(mock_settings)

        mock_openai_class.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=120.0,
        )
        mock_instructor.from_openai.assert_called_once()
        assert client == mock_instructor_client

    @patch("src.services.llm_client.instructor")
    @patch("src.services.llm_client.OpenAI")
    def test_caches_client(self, mock_openai_class, mock_instructor, mock_settings):
        """Test that the client is cached and reused."""
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance
        mock_instructor_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_instructor_client

        client1 = get_instructor_client(mock_settings)
        client2 = get_instructor_client(mock_settings)

        # Should only create once
        assert mock_openai_class.call_count == 1
        assert mock_instructor.from_openai.call_count == 1
        assert client1 is client2

    @patch("src.services.llm_client.instructor")
    @patch("src.services.llm_client.OpenAI")
    def test_different_settings_create_different_clients(self, mock_openai_class, mock_instructor):
        """Test that different settings create different cached clients."""
        mock_openai_class.return_value = MagicMock()
        mock_instructor.from_openai.side_effect = [MagicMock(), MagicMock()]

        settings1 = MagicMock(spec=Settings)
        settings1.ollama_url = "http://localhost:11434"
        settings1.ollama_timeout = 120

        settings2 = MagicMock(spec=Settings)
        settings2.ollama_url = "http://other:11434"
        settings2.ollama_timeout = 120

        client1 = get_instructor_client(settings1)
        client2 = get_instructor_client(settings2)

        # Should create two separate clients
        assert mock_openai_class.call_count == 2
        assert mock_instructor.from_openai.call_count == 2
        assert client1 is not client2


class TestGenerateStructured:
    """Tests for generate_structured function."""

    @patch("src.services.llm_client.get_instructor_client")
    def test_basic_generation(self, mock_get_client, mock_settings):
        """Test basic structured generation."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        result = generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="Generate something",
            response_model=SampleModel,
        )

        assert result.name == "test"
        assert result.value == 42
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.services.llm_client.get_instructor_client")
    def test_includes_system_prompt(self, mock_get_client, mock_settings):
        """Test that system prompt is included in messages."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt="System instructions",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instructions"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    @patch("src.services.llm_client.get_instructor_client")
    def test_no_system_prompt(self, mock_get_client, mock_settings):
        """Test generation without system prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("src.services.llm_client.get_instructor_client")
    def test_qwen_model_adds_no_think_prefix(self, mock_get_client, mock_settings):
        """Test that Qwen models get /no_think prefix in system prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="qwen2.5:14b",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt="Original system prompt",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "/no_think\nOriginal system prompt"

    @patch("src.services.llm_client.get_instructor_client")
    def test_qwen_model_no_system_prompt_no_change(self, mock_get_client, mock_settings):
        """Test that Qwen models without system prompt don't crash."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="qwen2.5:14b",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        # Only user message, no system message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("src.services.llm_client.get_instructor_client")
    def test_non_qwen_model_no_prefix(self, mock_get_client, mock_settings):
        """Test that non-Qwen models don't get /no_think prefix."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="llama3:8b",
            prompt="User prompt",
            response_model=SampleModel,
            system_prompt="Original system prompt",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["content"] == "Original system prompt"

    @patch("src.services.llm_client.get_instructor_client")
    def test_custom_temperature(self, mock_get_client, mock_settings):
        """Test that custom temperature is passed through."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            temperature=0.9,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @patch("src.services.llm_client.get_instructor_client")
    def test_custom_max_retries(self, mock_get_client, mock_settings):
        """Test that custom max_retries is passed through."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
            max_retries=5,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_retries"] == 5

    @patch("src.services.llm_client.get_instructor_client")
    def test_passes_response_model(self, mock_get_client, mock_settings):
        """Test that response_model is passed to the client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SampleModel(name="test", value=42)
        mock_get_client.return_value = mock_client

        generate_structured(
            settings=mock_settings,
            model="test-model",
            prompt="User prompt",
            response_model=SampleModel,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_model"] == SampleModel
        assert call_kwargs["model"] == "test-model"
