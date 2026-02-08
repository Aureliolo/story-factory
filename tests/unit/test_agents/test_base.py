"""Tests for the base agent class."""

import threading
from typing import Any, cast
from unittest.mock import MagicMock, patch

import ollama
import pytest
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.settings import Settings
from src.utils.circuit_breaker import reset_global_circuit_breaker
from src.utils.exceptions import CircuitOpenError, LLMGenerationError
from tests.shared.mock_ollama import TEST_MODEL


# Test model for structured output tests
class SampleOutputModel(BaseModel):
    """Test model for structured output."""

    name: str
    count: int = 0
    items: list[str] = Field(default_factory=list)


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
    # Use TEST_MODEL from mock_ollama to ensure model is in RECOMMENDED_MODELS
    agent = BaseAgent(
        name=overrides.get("name", "TestAgent"),
        role=overrides.get("role", "Tester"),
        system_prompt=overrides.get("system_prompt", "You are a test agent"),
        model=overrides.get("model", TEST_MODEL),
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
            role="Writer",
            agent_role="writer",
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

        assert isinstance(agent.client, ollama.Client)


class TestBaseAgentCheckOllamaHealth:
    """Tests for check_ollama_health classmethod."""

    @patch("src.agents.base.ollama.Client")
    def test_returns_healthy_when_ollama_responds(self, mock_client_class):
        """Test returns healthy tuple when Ollama responds."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
        mock_client_class.return_value = mock_client

        is_healthy, message = BaseAgent.check_ollama_health("http://localhost:11434")

        assert is_healthy is True
        assert "2 models available" in message

    @patch("src.agents.base.ollama.Client")
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
        agent.client.chat.return_value = {"message": {"content": "Valid response content"}}

        agent.generate("Prompt", context="Story context here")

        call_args = agent.client.chat.call_args
        messages = call_args.kwargs["messages"]
        context_messages = [m for m in messages if "CURRENT STORY CONTEXT" in m.get("content", "")]
        assert len(context_messages) == 1

    def test_generate_uses_custom_model(self):
        """Test generate uses custom model when provided."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Valid response content"}}

        agent.generate("Prompt", model="custom-model:7b")

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["model"] == "custom-model:7b"

    def test_generate_uses_custom_temperature(self):
        """Test generate uses custom temperature when provided."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Valid response content"}}

        agent.generate("Prompt", temperature=0.9)

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.9

    @patch("src.agents.base.time.sleep")
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

    @patch("src.agents.base.time.sleep")
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

    @patch("src.agents.base.time.sleep")
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


class TestBaseAgentQwenNoThink:
    """Tests that /no_think is NOT injected for any model."""

    def test_generate_no_think_not_added_for_qwen_model(self):
        """Test /no_think is NOT added to system prompt for Qwen models."""
        agent = create_mock_agent(model="huihui_ai/qwen3-abliterated:8b")
        agent.client.chat.return_value = {"message": {"content": "Valid response text"}}

        agent.generate("Test prompt")

        call_args = agent.client.chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert not system_msg.startswith("/no_think")
        assert system_msg == agent.system_prompt

    def test_generate_no_think_not_added_for_non_qwen(self):
        """Test /no_think is NOT added for non-Qwen models."""
        agent = create_mock_agent(model="huihui_ai/dolphin3-abliterated:8b")
        agent.client.chat.return_value = {"message": {"content": "Valid response text"}}

        agent.generate("Test prompt")

        call_args = agent.client.chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert not system_msg.startswith("/no_think")
        assert system_msg == agent.system_prompt


class TestBaseAgentShortResponseValidation:
    """Tests for short response validation and retry."""

    @patch("src.agents.base.time.sleep")
    def test_generate_retries_on_short_response(self, mock_sleep):
        """Test generate retries when response is too short after cleaning."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = [
            {"message": {"content": "<think>"}},  # Too short after cleaning
            {"message": {"content": "Valid response with enough content"}},
        ]

        result = agent.generate("Prompt")

        assert result == "Valid response with enough content"
        assert agent.client.chat.call_count == 2

    @patch("src.agents.base.time.sleep")
    def test_generate_raises_after_retries_exhausted_with_short_response(self, mock_sleep):
        """Test generate raises LLMGenerationError after all retries exhausted with short response."""
        from src.utils.exceptions import LLMGenerationError

        agent = create_mock_agent()
        # All attempts return short response
        agent.client.chat.return_value = {"message": {"content": "<think>"}}

        with pytest.raises(LLMGenerationError, match="Response too short"):
            agent.generate("Prompt")

        assert agent.client.chat.call_count == 3  # max_retries default

    def test_generate_returns_cleaned_content_without_think_tags(self):
        """Test generate returns content with think tags removed (#248).

        LLM responses may contain <think>...</think> tags from training data
        contamination. These should be stripped before returning to callers.
        """
        agent = create_mock_agent()
        agent.client.chat.return_value = {
            "message": {
                "content": "<think>Let me think about this carefully...</think>The actual response"
            }
        }

        result = agent.generate("Prompt")

        assert result == "The actual response"
        assert "<think>" not in result
        assert "</think>" not in result


class TestBaseAgentRateLimiting:
    """Tests for rate limiting in generate method."""

    def test_generate_respects_rate_limit(self):
        """Test generate uses semaphore for rate limiting."""
        agent = create_mock_agent()
        agent.client.chat.return_value = {"message": {"content": "Valid response content"}}

        # Verify semaphore is used (indirectly by checking concurrent calls)
        results = []

        def call_generate():
            """Call agent.generate and append result to shared list."""
            result = agent.generate("Prompt")
            results.append(result)

        threads = [threading.Thread(target=call_generate) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r == "Valid response content" for r in results)


class TestBaseAgentGetModelInfo:
    """Tests for get_model_info method."""

    @patch("src.agents.base.get_model_info")
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


class TestBaseAgentGenerateStructured:
    """Tests for generate_structured method with native Ollama format parameter."""

    @staticmethod
    def _make_chat_response(json_content: str) -> dict:
        """Create a mock Ollama chat response with JSON content.

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

    def test_generate_structured_returns_model_instance(self):
        """Test generate_structured returns validated Pydantic model instance."""
        agent = create_mock_agent()
        agent.client.chat.return_value = self._make_chat_response(
            '{"name": "Test", "count": 5, "items": ["a", "b"]}'
        )

        result = agent.generate_structured("Test prompt", SampleOutputModel)

        assert isinstance(result, SampleOutputModel)
        assert result.name == "Test"
        assert result.count == 5
        assert result.items == ["a", "b"]

    def test_generate_structured_uses_low_temperature_by_default(self):
        """Test generate_structured uses low temperature for schema adherence."""
        agent = create_mock_agent()
        agent.client.chat.return_value = self._make_chat_response('{"name": "Test"}')

        agent.generate_structured("Test prompt", SampleOutputModel)

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.1

    def test_generate_structured_allows_custom_temperature(self):
        """Test generate_structured accepts custom temperature."""
        agent = create_mock_agent()
        agent.client.chat.return_value = self._make_chat_response('{"name": "Test"}')

        agent.generate_structured("Test prompt", SampleOutputModel, temperature=0.5)

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.5

    def test_generate_structured_includes_context(self):
        """Test generate_structured includes context in messages."""
        agent = create_mock_agent()
        agent.client.chat.return_value = self._make_chat_response('{"name": "Test"}')

        agent.generate_structured("Test prompt", SampleOutputModel, context="Story context here")

        call_args = agent.client.chat.call_args
        messages = call_args.kwargs["messages"]
        context_found = any("CURRENT STORY CONTEXT" in m.get("content", "") for m in messages)
        assert context_found

    def test_generate_structured_no_think_not_added_for_qwen(self):
        """Test generate_structured does NOT add /no_think for Qwen models."""
        agent = create_mock_agent(model="fake-qwen:7b")
        agent.client.chat.return_value = self._make_chat_response('{"name": "Test"}')

        agent.generate_structured("Test prompt", SampleOutputModel)

        call_args = agent.client.chat.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "/no_think" not in system_msg

    def test_generate_structured_raises_on_error(self):
        """Test generate_structured raises LLMGenerationError on failure."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = ollama.ResponseError("API error")

        with pytest.raises(LLMGenerationError, match="Structured generation failed"):
            agent.generate_structured("Test prompt", SampleOutputModel)

    def test_generate_structured_passes_format_schema(self):
        """Test generate_structured passes JSON schema via format parameter."""
        agent = create_mock_agent()
        agent.client.chat.return_value = self._make_chat_response('{"name": "Test"}')

        agent.generate_structured("Test prompt", SampleOutputModel)

        call_args = agent.client.chat.call_args
        assert call_args.kwargs["format"] == SampleOutputModel.model_json_schema()

    def test_generate_structured_retries_on_validation_error(self):
        """Test generate_structured retries on Pydantic validation error."""
        agent = create_mock_agent()
        # First call returns invalid JSON, second returns valid
        agent.client.chat.side_effect = [
            self._make_chat_response('{"invalid_field": "bad"}'),
            self._make_chat_response('{"name": "Test"}'),
        ]

        result = agent.generate_structured("Test prompt", SampleOutputModel, max_retries=2)

        assert result.name == "Test"
        assert agent.client.chat.call_count == 2

    def test_generate_structured_records_token_metrics(self):
        """Test generate_structured records token counts from Ollama response."""
        agent = create_mock_agent()
        agent.client.chat.return_value = self._make_chat_response('{"name": "Test"}')

        agent.generate_structured("Test prompt", SampleOutputModel)

        metrics = agent.last_generation_metrics
        assert metrics is not None
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150

    @patch("src.agents.base.time.sleep")
    def test_generate_structured_retries_on_connection_error(self, mock_sleep):
        """Test generate_structured retries on ConnectionError with backoff."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = [
            ConnectionError("Connection refused"),
            self._make_chat_response('{"name": "Recovered"}'),
        ]

        result = agent.generate_structured("Test prompt", SampleOutputModel, max_retries=2)

        assert result.name == "Recovered"
        assert agent.client.chat.call_count == 2
        mock_sleep.assert_called_once_with(1)  # min(2**0, 10) = 1

    @patch("src.agents.base.time.sleep")
    def test_generate_structured_retries_on_timeout_error(self, mock_sleep):
        """Test generate_structured retries on TimeoutError with backoff."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = [
            TimeoutError("Request timed out"),
            self._make_chat_response('{"name": "Recovered"}'),
        ]

        result = agent.generate_structured("Test prompt", SampleOutputModel, max_retries=2)

        assert result.name == "Recovered"
        assert agent.client.chat.call_count == 2
        mock_sleep.assert_called_once_with(1)  # min(2**0, 10) = 1

    @patch("src.agents.base.time.sleep")
    def test_generate_structured_exhausts_retries_raises(self, mock_sleep):
        """Test generate_structured raises LLMGenerationError after exhausting all retries."""
        agent = create_mock_agent()
        agent.client.chat.side_effect = ConnectionError("Connection refused")

        with pytest.raises(LLMGenerationError, match="Structured generation failed"):
            agent.generate_structured("Test prompt", SampleOutputModel, max_retries=2)

        assert agent.client.chat.call_count == 2
        mock_sleep.assert_called_once_with(1)  # backoff on first attempt before retry

    def test_generate_structured_max_retries_zero_raises(self):
        """Test generate_structured raises ValueError when max_retries < 1."""
        agent = create_mock_agent()

        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            agent.generate_structured("Test prompt", SampleOutputModel, max_retries=0)


class TestBaseAgentPromptTemplate:
    """Tests for prompt template methods."""

    def test_get_registry_returns_registry(self):
        """Test get_registry returns a PromptRegistry instance."""
        from src.utils.prompt_registry import PromptRegistry

        registry = BaseAgent.get_registry()

        assert isinstance(registry, PromptRegistry)
        assert len(registry) > 0  # Should have loaded templates

    def test_get_registry_returns_same_instance(self):
        """Test get_registry returns the same singleton instance."""
        registry1 = BaseAgent.get_registry()
        registry2 = BaseAgent.get_registry()

        assert registry1 is registry2

    def test_has_prompt_template_returns_true_for_existing(self):
        """Test has_prompt_template returns True for existing template."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )

        # Writer should have system template
        assert agent.has_prompt_template("system") is True

    def test_has_prompt_template_returns_false_for_missing(self):
        """Test has_prompt_template returns False for missing template."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )

        # Writer shouldn't have this task
        assert agent.has_prompt_template("nonexistent_task") is False

    def test_render_prompt_renders_template(self):
        """Test render_prompt renders a template with variables."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )

        # Render the write_chapter template
        result = agent.render_prompt(
            "write_chapter",
            chapter_number=1,
            chapter_title="The Beginning",
            chapter_outline="First chapter outline",
            story_context="Fantasy story context",
            language="English",
            genre="Fantasy",
            tone="Epic",
            content_rating="Teen",
        )

        assert "Chapter 1" in result
        assert "The Beginning" in result
        assert "English" in result

    def test_get_prompt_hash_returns_hash(self):
        """Test get_prompt_hash returns a valid MD5 hash."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )

        hash_value = agent.get_prompt_hash("system")

        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # MD5 hex length

    def test_get_prompt_hash_consistent(self):
        """Test get_prompt_hash returns same hash for same template."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )

        hash1 = agent.get_prompt_hash("system")
        hash2 = agent.get_prompt_hash("system")

        assert hash1 == hash2

    def test_get_system_prompt_from_template_returns_prompt(self):
        """Test get_system_prompt_from_template returns rendered prompt."""
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )

        result = agent.get_system_prompt_from_template()

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_system_prompt_from_template_returns_none_for_missing(self):
        """Test get_system_prompt_from_template returns None when no template."""
        # Create agent with a valid role that exists in settings
        agent = BaseAgent(
            name="Test",
            role="Writer",
            system_prompt="Test prompt",
            agent_role="writer",
        )
        # Override agent_role after creation to simulate missing template
        agent.agent_role = "nonexistent_agent_role"

        result = agent.get_system_prompt_from_template()

        assert result is None


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration in BaseAgent."""

    def setup_method(self):
        """Reset global circuit breaker before each test."""
        reset_global_circuit_breaker()

    def teardown_method(self):
        """Reset global circuit breaker after each test."""
        reset_global_circuit_breaker()

    def test_generate_raises_circuit_open_error_when_open(self):
        """Test that generate() raises CircuitOpenError when circuit is open."""
        from src.utils.circuit_breaker import get_circuit_breaker

        # Create agent with circuit breaker enabled
        settings = Settings(
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_timeout=60.0,
        )
        agent = create_mock_agent()
        agent.settings = settings

        # Open the circuit breaker
        cb = get_circuit_breaker(
            failure_threshold=2,
            timeout_seconds=60.0,
            enabled=True,
        )
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        # Now generate() should raise CircuitOpenError
        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            agent.generate("test prompt")

    def test_generate_structured_raises_circuit_open_error_when_open(self):
        """Test that generate_structured() raises CircuitOpenError when circuit is open."""
        from src.utils.circuit_breaker import get_circuit_breaker

        # Create agent with circuit breaker enabled
        settings = Settings(
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_timeout=60.0,
        )
        agent = create_mock_agent()
        agent.settings = settings

        # Open the circuit breaker
        cb = get_circuit_breaker(
            failure_threshold=2,
            timeout_seconds=60.0,
            enabled=True,
        )
        cb.record_failure(Exception("test 1"))
        cb.record_failure(Exception("test 2"))

        # Now generate_structured() should raise CircuitOpenError
        with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
            agent.generate_structured("test prompt", SampleOutputModel)

    def test_generate_works_when_circuit_closed(self):
        """Test that generate() works normally when circuit is closed."""
        from src.utils.circuit_breaker import get_circuit_breaker

        # Ensure circuit is closed
        cb = get_circuit_breaker(enabled=True)
        cb.reset()

        agent = create_mock_agent()
        agent.settings = Settings(circuit_breaker_enabled=True)
        agent.client.chat.return_value = {"message": {"content": "test response"}}

        result = agent.generate("test prompt")

        assert result == "test response"


class TestBaseAgentGenerationMetrics:
    """Tests for generation metrics tracking."""

    def test_last_generation_metrics_initially_none(self):
        """Test last_generation_metrics is None before any generation."""
        agent = create_mock_agent()

        assert agent.last_generation_metrics is None
