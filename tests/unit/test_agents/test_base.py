"""Tests for the base agent class."""

import threading
from typing import Any, cast
from unittest.mock import MagicMock, patch

import ollama
import pytest

from agents.base import BaseAgent
from settings import Settings
from utils.exceptions import LLMGenerationError


class MockedBaseAgent(BaseAgent):
    """BaseAgent subclass with MagicMock client for testing.

    This provides proper type hints for the mocked client while
    allowing access to MagicMock methods like .return_value and .side_effect.
    """

    client: MagicMock  # Override type to MagicMock for testing


def create_mock_agent(**overrides: Any) -> MockedBaseAgent:
    """Create a BaseAgent with mocked internals for testing.

    This is the proper way to create test agents without triggering
    actual Ollama connections. All required attributes are set to
    sensible test defaults.

    Args:
        **overrides: Override any default attributes.

    Returns:
        A MockedBaseAgent instance ready for testing.
    """
    # Create agent with minimal required args
    agent = BaseAgent(
        name=overrides.get("name", "TestAgent"),
        role=overrides.get("role", "Tester"),
        system_prompt=overrides.get("system_prompt", "You are a test agent"),
        model=overrides.get("model", "test-model:7b"),
        temperature=overrides.get("temperature", 0.7),
    )

    # Replace the real Ollama client with a mock
    agent.client = MagicMock()

    # Apply any additional overrides
    for key, value in overrides.items():
        if hasattr(agent, key):
            setattr(agent, key, value)

    return cast(MockedBaseAgent, agent)


class TestBaseAgentInit:
    """Tests for BaseAgent initialization."""

    def test_init_with_explicit_model(self):
        """Test agent uses explicit model when provided."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            agent_role="writer",
            system_prompt="Test prompt",
            model="custom-model:7b",
        )

        assert agent.model == "custom-model:7b"

    def test_init_with_explicit_temperature(self):
        """Test agent uses explicit temperature when provided."""
        agent = BaseAgent(
            name="Test",
            role="Tester",
            system_prompt="Test prompt",
            temperature=0.5,
        )

        assert agent.temperature == 0.5

    def test_init_uses_settings_model_when_not_specified(self):
        """Test agent gets model from settings when not specified."""
        settings = Settings()
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
            settings=settings,
        )

        expected_model = settings.get_model_for_agent("writer")
        assert agent.model == expected_model

    def test_init_uses_settings_temperature_when_not_specified(self):
        """Test agent gets temperature from settings when not specified."""
        settings = Settings()
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
            settings=settings,
        )

        expected_temp = settings.get_temperature_for_agent("writer")
        assert agent.temperature == expected_temp

    def test_init_derives_agent_role_from_role(self):
        """Test agent_role is derived from role if not provided."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
        )

        assert agent.agent_role == "writer"

    def test_init_creates_ollama_client_with_settings(self):
        """Test Ollama client is created with settings URL and timeout."""
        settings = Settings()
        agent = BaseAgent(
            name="Test",
            role="Editor",
            system_prompt="Test prompt",
            settings=settings,
        )

        assert agent.client is not None


class TestBaseAgentCheckOllamaHealth:
    """Tests for check_ollama_health classmethod."""

    @patch("agents.base.ollama.Client")
    def test_returns_healthy_when_ollama_responds(self, mock_client_class):
        """Test returns healthy tuple when Ollama responds."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
        mock_client_class.return_value = mock_client

        is_healthy, message = BaseAgent.check_ollama_health("http://localhost:11434")

        assert is_healthy is True
        assert "2 models available" in message

    @patch("agents.base.ollama.Client")
    def test_returns_healthy_with_empty_models(self, mock_client_class):
        """Test returns healthy even with no models."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        mock_client_class.return_value = mock_client

        is_healthy, message = BaseAgent.check_ollama_health()

        assert is_healthy is True
        assert "0 models available" in message


class TestBaseAgentValidateModel:
    """Tests for validate_model method."""

    def test_validate_installed_model_returns_true(self):
        """Test validates installed model."""
        agent = create_mock_agent()
        agent.client.list.return_value = {"models": [{"name": "test-model:7b"}]}

        is_valid, message = agent.validate_model("test-model:7b")

        assert is_valid is True
        assert "is available" in message

    def test_validate_with_latest_suffix(self):
        """Test validates model with :latest suffix."""
        agent = create_mock_agent()
        agent.client.list.return_value = {"models": [{"name": "test-model:latest"}]}

        is_valid, message = agent.validate_model("test-model")

        assert is_valid is True
        assert "is available" in message

    def test_validate_missing_model_returns_false(self):
        """Test returns false for missing model."""
        agent = create_mock_agent()
        agent.client.list.return_value = {"models": [{"name": "other-model:7b"}]}

        is_valid, message = agent.validate_model("missing-model")

        assert is_valid is False
        assert "not found" in message

    def test_validate_handles_connection_error(self):
        """Test handles connection error gracefully."""
        agent = create_mock_agent()
        agent.client.list.side_effect = ConnectionError("Connection refused")

        is_valid, message = agent.validate_model("any-model")

        assert is_valid is False
        assert "Error checking model availability" in message

    def test_validate_handles_timeout_error(self):
        """Test handles timeout error gracefully."""
        agent = create_mock_agent()
        agent.client.list.side_effect = TimeoutError("Timeout")

        is_valid, message = agent.validate_model("any-model")

        assert is_valid is False
        assert "Error checking model availability" in message

    def test_validate_handles_response_error(self):
        """Test handles Ollama response error gracefully."""
        agent = create_mock_agent()
        agent.client.list.side_effect = ollama.ResponseError("Error")

        is_valid, message = agent.validate_model("any-model")

        assert is_valid is False
        assert "Error checking model availability" in message


class TestBaseAgentGenerate:
    """Tests for generate method."""

    def test_generate_returns_content(self):
        """Test generate returns response content."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Test response"}}

        result = agent.generate("Test prompt")

        assert result == "Test response"

    def test_generate_includes_context(self):
        """Test generate includes context in messages."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Response"}}

        agent.generate("Prompt", context="Story context here")

        call_args = agent.client.chat.call_args
        messages = call_args.kwargs["messages"]
        context_messages = [m for m in messages if "CURRENT STORY CONTEXT" in m.get("content", "")]
        assert len(context_messages) == 1

    def test_generate_uses_custom_model(self):
        """Test generate uses custom model when provided."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Response"}}

        agent.generate("Prompt", model="custom-model:7b")

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["model"] == "custom-model:7b"

    def test_generate_uses_custom_temperature(self):
        """Test generate uses custom temperature when provided."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Response"}}

        agent.generate("Prompt", temperature=0.9)

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.9

    @patch("agents.base.time.sleep")
    def test_generate_retries_on_connection_error(self, mock_sleep):
        """Test generate retries on connection error."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = [
            ConnectionError("First failure"),
            {"message": {"content": "Success after retry"}},
        ]

        result = agent.generate("Prompt")

        assert result == "Success after retry"
        assert agent.client.chat.call_count == 2
        mock_sleep.assert_called_once()

    @patch("agents.base.time.sleep")
    def test_generate_retries_on_timeout(self, mock_sleep):
        """Test generate retries on timeout."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = [
            TimeoutError("Timeout"),
            {"message": {"content": "Success after retry"}},
        ]

        result = agent.generate("Prompt")

        assert result == "Success after retry"
        assert agent.client.chat.call_count == 2

    @patch("agents.base.time.sleep")
    def test_generate_exhausts_retries(self, mock_sleep):
        """Test generate raises after exhausting retries."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = ConnectionError("Persistent failure")

        with pytest.raises(LLMGenerationError, match="Failed to generate after"):
            agent.generate("Prompt")

        assert agent.client.chat.call_count == 3

    def test_generate_raises_immediately_on_response_error(self):
        """Test generate raises immediately on Ollama response error."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = ollama.ResponseError("Model not found")

        with pytest.raises(LLMGenerationError, match="Model error"):
            agent.generate("Prompt")

        # Should not retry on model errors
        assert agent.client.chat.call_count == 1


class TestBaseAgentRateLimiting:
    """Tests for rate limiting in generate method."""

    def test_generate_respects_rate_limit(self):
        """Test generate uses semaphore for rate limiting."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Response"}}

        # Verify semaphore is used (indirectly by checking concurrent calls)
        results = []

        def call_generate():
            result = agent.generate("Prompt")
            results.append(result)

        threads = [threading.Thread(target=call_generate) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r == "Response" for r in results)


class TestBaseAgentGetModelInfo:
    """Tests for get_model_info method."""

    @patch("agents.base.get_model_info")
    def test_get_model_info_returns_info(self, mock_get_info):
        """Test get_model_info returns model information."""
        mock_info = {
            "name": "Test Model",
            "quality": 8,
            "speed": 7,
            "vram_required": 12,
        }
        mock_get_info.return_value = mock_info

        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test",
            model="test-model:7b",
        )

        result = agent.get_model_info()

        assert result == mock_info
        mock_get_info.assert_called_with("test-model:7b")


class TestBaseAgentRepr:
    """Tests for __repr__ method."""

    def test_repr_shows_name_and_model(self):
        """Test __repr__ shows agent name and model."""
        agent = BaseAgent(
            name="TestAgent",
            role="Writer",
            system_prompt="Test",
            model="test-model:7b",
        )

        repr_str = repr(agent)

        assert "TestAgent" in repr_str
        assert "test-model:7b" in repr_str
