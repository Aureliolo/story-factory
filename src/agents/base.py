"""Base agent class for all story factory agents."""

import logging
import threading
import time
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from src.memory.cost_models import GenerationMetrics
from src.settings import ModelInfo, Settings, get_model_info
from src.utils.circuit_breaker import get_circuit_breaker
from src.utils.error_handling import handle_ollama_errors
from src.utils.exceptions import CircuitOpenError, LLMConnectionError, LLMError, LLMGenerationError
from src.utils.json_parser import clean_llm_text
from src.utils.logging_config import log_performance
from src.utils.prompt_registry import PromptRegistry
from src.utils.validation import validate_not_empty

# Type variable for generic structured output
T = TypeVar("T", bound=BaseModel)

# Minimum response length after cleaning (to detect truncated/empty responses)
MIN_RESPONSE_LENGTH = 10

logger = logging.getLogger(__name__)

# Rate limiting: lazily initialized semaphore based on settings
_llm_semaphore: threading.Semaphore | None = None
_llm_semaphore_lock = threading.Lock()


def _get_llm_semaphore(settings: Settings) -> threading.Semaphore:
    """Get or create the LLM rate limiting semaphore.

    The semaphore is created lazily on first use with the concurrency limit
    from the provided settings. Once created, the limit cannot be changed
    without restarting the application.
    """
    global _llm_semaphore
    if _llm_semaphore is None:
        with _llm_semaphore_lock:
            # Double-check after acquiring lock
            if _llm_semaphore is None:
                _llm_semaphore = threading.Semaphore(settings.llm_max_concurrent_requests)
                logger.debug(
                    f"Initialized LLM semaphore with limit {settings.llm_max_concurrent_requests}"
                )
    return _llm_semaphore


# Re-export exceptions for backward compatibility
__all__ = ["BaseAgent", "CircuitOpenError", "LLMConnectionError", "LLMError", "LLMGenerationError"]

# Class-level singleton for prompt registry
_prompt_registry: PromptRegistry | None = None
_prompt_registry_lock = threading.Lock()


def _get_prompt_registry() -> PromptRegistry:
    """Get or create the prompt registry singleton.

    The registry is created lazily on first use and shared across all agents.
    This ensures templates are loaded only once per application lifetime.
    Uses the prompt_templates_dir setting to locate templates.
    """
    global _prompt_registry
    if _prompt_registry is None:
        # Load settings outside lock to minimize lock hold time
        settings = Settings.load()
        with _prompt_registry_lock:
            # Double-check after acquiring lock
            if _prompt_registry is None:
                _prompt_registry = PromptRegistry(settings.prompt_templates_dir)
                logger.info(f"Initialized prompt registry with {len(_prompt_registry)} templates")
    return _prompt_registry


class BaseAgent:
    """Base class for all agents in the story factory."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        agent_role: str | None = None,  # For auto model selection
        model: str | None = None,
        temperature: float | None = None,
        settings: Settings | None = None,
    ):
        """
        Create a BaseAgent with identity, prompts, model selection, and LLM client setup.

        Parameters:
            name (str): Display name for the agent.
            role (str): Human-readable role description for the agent.
            system_prompt (str): System prompt that guides the agent's behavior.
            agent_role (str | None): Identifier used for agent-specific defaults (model/temperature). If omitted, a normalized form of `role` is used.
            model (str | None): Explicit model to use; when None the agent's model is resolved from settings for `agent_role`.
            temperature (float | None): Explicit sampling temperature; when None the temperature is resolved from settings for `agent_role`.
            settings (Settings | None): Application settings to use; when None the default Settings are loaded.

        Notes:
            - Initializes an Ollama client using the configured URL and timeout.
            - Storage for last generation metrics is initialized to None.
        """
        validate_not_empty(name, "name")
        validate_not_empty(role, "role")
        validate_not_empty(system_prompt, "system_prompt")

        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.agent_role = agent_role or role.lower().replace(" ", "_")

        # Load settings
        self.settings = settings or Settings.load()

        # Get model and temperature from settings if not specified
        if model:
            self.model = model
        else:
            self.model = self.settings.get_model_for_agent(self.agent_role)

        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = self.settings.get_temperature_for_agent(self.agent_role)

        # Create Ollama client with scaled timeout based on model size
        self.client = ollama.Client(
            host=self.settings.ollama_url, timeout=self.settings.get_scaled_timeout(self.model)
        )

        # Store metrics from the last generation for cost tracking
        self._last_generation_metrics: GenerationMetrics | None = None

    @property
    def last_generation_metrics(self) -> GenerationMetrics | None:
        """
        Expose metrics from the most recent generation call.

        Returns:
            GenerationMetrics containing token counts, duration, model_id, and agent_role, or `None` if no generation has occurred.
        """
        return self._last_generation_metrics

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> T:
        """Generate structured output using native Ollama format parameter.

        Uses ollama.Client.chat() with `format=` set to the Pydantic model's
        JSON schema. Ollama constrains output at the grammar level, ensuring
        valid JSON matching the schema (investigation #267: 858 calls, 0 failures).

        Args:
            prompt: The user prompt to send.
            response_model: Pydantic model class defining the expected output structure.
            context: Optional context to include as a system message.
            temperature: Override temperature (defaults to low value for structured output).
            max_retries: Maximum number of retries on validation failure.

        Returns:
            Instance of response_model with validated data.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
            LLMGenerationError: If generation fails after all retries.
            ValueError: If max_retries < 1.
        """
        validate_not_empty(prompt, "prompt")

        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")

        # Check circuit breaker before proceeding
        circuit_breaker = get_circuit_breaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            success_threshold=self.settings.circuit_breaker_success_threshold,
            timeout_seconds=self.settings.circuit_breaker_timeout,
            enabled=self.settings.circuit_breaker_enabled,
        )
        if not circuit_breaker.allow_request():
            status = circuit_breaker.get_status()
            time_until_retry = status.get("time_until_half_open", 0)
            logger.warning(
                "%s: Circuit breaker open; refusing LLM call for %.0fs",
                self.name,
                time_until_retry,
            )
            raise CircuitOpenError(
                f"Circuit breaker is open. Too many LLM failures. "
                f"Will retry in {time_until_retry:.0f}s.",
                time_until_retry=time_until_retry,
            )

        # Build messages
        use_model = self.model
        if "qwen" in use_model.lower():
            system_content = f"/no_think\n{self.system_prompt}"
        else:
            system_content = self.system_prompt

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        if context:
            messages.append({"role": "system", "content": f"CURRENT STORY CONTEXT:\n{context}"})

        messages.append({"role": "user", "content": prompt})

        # Use low temperature for structured output to ensure schema adherence
        use_temp = temperature if temperature is not None else 0.1

        # Clear previous metrics before new generation
        self._last_generation_metrics = None

        # Get JSON schema for format parameter
        json_schema = response_model.model_json_schema()

        # Acquire semaphore to limit concurrent requests
        with _get_llm_semaphore(self.settings):
            with log_performance(logger, f"{self.name} structured generation"):
                last_error: Exception | None = None

                for attempt in range(max_retries):
                    try:
                        logger.debug(
                            f"{self.name}: Calling LLM ({use_model}) for structured output "
                            f"(model={response_model.__name__}, temp={use_temp}, "
                            f"attempt={attempt + 1}/{max_retries})"
                        )

                        start_time = time.time()
                        response = self.client.chat(
                            model=use_model,
                            messages=messages,
                            format=json_schema,
                            options={
                                "temperature": use_temp,
                                "num_ctx": self.settings.context_size,
                            },
                        )
                        duration = time.time() - start_time

                        # Extract token counts from native Ollama response
                        prompt_tokens = response.get("prompt_eval_count")
                        completion_tokens = response.get("eval_count")
                        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

                        # Parse and validate the response
                        content = response["message"]["content"]
                        result = response_model.model_validate_json(content)

                        # Store metrics for structured generation
                        self._last_generation_metrics = GenerationMetrics(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            time_seconds=duration,
                            model_id=use_model,
                            agent_role=self.agent_role,
                        )

                        logger.info(
                            f"{self.name}: Structured output received "
                            f"({response_model.__name__}, {duration:.2f}s, "
                            f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens})"
                        )
                        circuit_breaker.record_success()
                        return result

                    except (ValidationError, KeyError, TypeError) as e:
                        last_error = e
                        logger.warning(
                            "%s: Structured output validation/parsing failed (attempt %d/%d): %s",
                            self.name,
                            attempt + 1,
                            max_retries,
                            e,
                        )
                        if attempt < max_retries - 1:
                            continue  # Retry with same prompt

                    except (ConnectionError, TimeoutError) as e:
                        last_error = e
                        circuit_breaker.record_failure(e)
                        logger.warning(
                            "%s: Connection/timeout error on structured attempt %d/%d: %s",
                            self.name,
                            attempt + 1,
                            max_retries,
                            e,
                        )
                        if attempt < max_retries - 1:
                            backoff = min(2**attempt, 10)
                            time.sleep(backoff)
                            continue

                    except ollama.ResponseError as e:
                        # Model-specific errors — don't retry
                        logger.error(f"{self.name}: Ollama response error: {e}")
                        circuit_breaker.record_failure(e)
                        raise LLMGenerationError(
                            f"Structured generation failed for {response_model.__name__}: {e}"
                        ) from e

                # All retries exhausted
                logger.error(
                    "%s: Structured generation failed after %d attempts",
                    self.name,
                    max_retries,
                )
                circuit_breaker.record_failure(last_error)
                raise LLMGenerationError(
                    f"Structured generation failed for {response_model.__name__}: {last_error}"
                ) from last_error

    @classmethod
    @handle_ollama_errors(default_return=(False, "Ollama connection failed"), raise_on_error=False)
    def check_ollama_health(
        cls, ollama_url: str = "http://localhost:11434", timeout: int = 30
    ) -> tuple[bool, str]:
        """Check if Ollama is running and accessible.

        Args:
            ollama_url: Ollama API URL.
            timeout: Connection timeout in seconds.

        Returns:
            Tuple of (is_healthy, message)
        """
        client = ollama.Client(host=ollama_url, timeout=float(timeout))
        models = client.list()
        model_count = len(models.get("models", []))
        return True, f"Ollama connected. {model_count} models available."

    def validate_model(self, model_name: str) -> tuple[bool, str]:
        """Validate that a model is available.

        Args:
            model_name: The model name to validate

        Returns:
            Tuple of (is_valid, message)
        """
        validate_not_empty(model_name, "model_name")
        try:
            models = self.client.list()
            available_models = [m["name"] for m in models.get("models", [])]

            # Check direct match or with :latest suffix
            if model_name in available_models:
                return True, f"Model '{model_name}' is available"

            if f"{model_name}:latest" in available_models:
                return True, f"Model '{model_name}:latest' is available"

            # Model not found
            return False, (
                f"Model '{model_name}' not found. Available models: {', '.join(available_models[:5])}"
            )
        except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
            error_msg = f"Error checking model availability: {e}"
            logger.warning(error_msg)
            return False, error_msg

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
        min_response_length: int | None = None,
    ) -> str:
        """
        Generate a plain-text response from the agent using the configured LLM.

        Retries on transient connection/timeouts, enforces a minimum cleaned response length, and records token/time metrics for the last generation.

        Uses a semaphore to limit concurrent LLM requests and prevent overloading
        the Ollama server. Also uses a circuit breaker to protect against cascading failures.

        Parameters:
            prompt (str): The prompt to send to the LLM.
            context (str | None): Optional additional context to include in the system messages.
            temperature (float | None): Optional temperature override for this call.
            model (str | None): Optional model identifier override for this call.
            min_response_length (int | None): Minimum cleaned response length to accept. Defaults to MIN_RESPONSE_LENGTH; set to 1 for very short expected outputs.

        Returns:
            str: The cleaned text content returned by the LLM, with think tags
                 and other artifacts stripped via clean_llm_text().

        Raises:
            CircuitOpenError: If the circuit breaker is open.
            LLMGenerationError: If generation fails due to model errors or after exhausting retries for transient errors or consistently too-short responses.
        """
        validate_not_empty(prompt, "prompt")

        # Check circuit breaker before proceeding
        circuit_breaker = get_circuit_breaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            success_threshold=self.settings.circuit_breaker_success_threshold,
            timeout_seconds=self.settings.circuit_breaker_timeout,
            enabled=self.settings.circuit_breaker_enabled,
        )
        if not circuit_breaker.allow_request():
            status = circuit_breaker.get_status()
            time_until_retry = status.get("time_until_half_open", 0)
            logger.warning(
                "%s: Circuit breaker open; refusing LLM call for %.0fs",
                self.name,
                time_until_retry,
            )
            raise CircuitOpenError(
                f"Circuit breaker is open. Too many LLM failures. "
                f"Will retry in {time_until_retry:.0f}s.",
                time_until_retry=time_until_retry,
            )

        # Add /no_think prefix for Qwen models to disable thinking mode
        # This prevents models from outputting <think>...</think> tags instead of actual content
        use_model = model or self.model
        if "qwen" in use_model.lower():
            system_content = f"/no_think\n{self.system_prompt}"
        else:
            system_content = self.system_prompt
        messages = [{"role": "system", "content": system_content}]

        if context:
            messages.append({"role": "system", "content": f"CURRENT STORY CONTEXT:\n{context}"})

        messages.append({"role": "user", "content": prompt})

        use_temp = temperature or self.temperature
        use_min_length = (
            min_response_length if min_response_length is not None else MIN_RESPONSE_LENGTH
        )

        last_error: Exception | None = None
        delay = self.settings.llm_retry_delay
        max_retries = self.settings.llm_max_retries

        # Acquire semaphore to limit concurrent requests
        with _get_llm_semaphore(self.settings):
            with log_performance(logger, f"{self.name} generation"):
                for attempt in range(max_retries):
                    try:
                        logger.info(
                            f"{self.name}: Calling LLM ({use_model}) attempt {attempt + 1}/{max_retries}"
                        )

                        start_time = time.time()
                        response = self.client.chat(
                            model=use_model,
                            messages=messages,
                            options={
                                "temperature": use_temp,
                                "num_predict": self.settings.max_tokens,
                                "num_ctx": self.settings.context_size,
                            },
                        )
                        duration = time.time() - start_time

                        content: str = response["message"]["content"]

                        # Extract token counts from Ollama response for cost tracking
                        prompt_tokens = response.get("prompt_eval_count")
                        completion_tokens = response.get("eval_count")
                        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

                        # Store metrics for callers who need cost tracking
                        self._last_generation_metrics = GenerationMetrics(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            time_seconds=duration,
                            model_id=use_model,
                            agent_role=self.agent_role,
                        )

                        logger.info(
                            f"{self.name}: LLM response received "
                            f"({len(content)} chars, {duration:.2f}s, "
                            f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens})"
                        )

                        # Validate response isn't just thinking tokens or truncated
                        cleaned_content = clean_llm_text(content)
                        if len(cleaned_content) < use_min_length:
                            logger.warning(
                                f"{self.name}: Response too short after cleaning "
                                f"({len(cleaned_content)} chars < {use_min_length}), "
                                f"raw: {content[:50]!r}..."
                            )
                            if attempt < max_retries - 1:
                                logger.info(f"{self.name}: Retrying in {delay}s...")
                                time.sleep(delay)
                                delay *= self.settings.llm_retry_backoff
                                continue  # Retry
                            else:
                                # Last attempt - raise error, don't return garbage
                                error_msg = (
                                    f"{self.name}: Response too short after {max_retries} attempts "
                                    f"({len(cleaned_content)} chars < {use_min_length})"
                                )
                                logger.error(error_msg)
                                circuit_breaker.record_failure(LLMGenerationError(error_msg))
                                raise LLMGenerationError(error_msg)

                        # Success - record it and return cleaned content (think tags removed)
                        circuit_breaker.record_success()
                        return cleaned_content

                    except ConnectionError as e:
                        last_error = e
                        circuit_breaker.record_failure(e)
                        logger.warning(
                            f"{self.name}: Connection error on attempt {attempt + 1}: {e}"
                        )
                        if attempt < max_retries - 1:
                            logger.info(f"{self.name}: Retrying in {delay}s...")
                            time.sleep(delay)
                            delay *= self.settings.llm_retry_backoff

                    except ollama.ResponseError as e:
                        # Model-specific errors (model not found, etc.) - don't retry
                        logger.error(f"{self.name}: Ollama response error: {e}")
                        circuit_breaker.record_failure(e)
                        raise LLMGenerationError(f"Model error: {e}") from e

                    except TimeoutError as e:
                        last_error = e
                        circuit_breaker.record_failure(e)
                        logger.warning(f"{self.name}: Timeout on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            logger.info(f"{self.name}: Retrying in {delay}s...")
                            time.sleep(delay)
                            delay *= self.settings.llm_retry_backoff

                # All retries failed
                logger.error(f"{self.name}: All {max_retries} attempts failed")
                raise LLMGenerationError(
                    f"Failed to generate after {max_retries} attempts: {last_error}"
                ) from last_error

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        return get_model_info(self.model)

    # === Prompt Template Methods ===

    def render_prompt(self, task: str, **kwargs) -> str:
        """Render a prompt template for this agent.

        Uses the prompt registry to find and render the appropriate template
        based on this agent's role and the specified task.

        Args:
            task: Task identifier (e.g., "write_chapter", "edit_passage").
            **kwargs: Variables to substitute into the template.

        Returns:
            Rendered prompt string.

        Raises:
            PromptTemplateError: If template not found or rendering fails.
        """
        registry = _get_prompt_registry()
        return registry.render(self.agent_role, task, **kwargs)

    def get_prompt_hash(self, task: str) -> str:
        """Get the hash of a prompt template for metrics tracking.

        Args:
            task: Task identifier.

        Returns:
            MD5 hash of the template content.

        Raises:
            PromptTemplateError: If template not found.
        """
        registry = _get_prompt_registry()
        return registry.get_hash(self.agent_role, task)

    def has_prompt_template(self, task: str) -> bool:
        """Check if a prompt template exists for this agent and task.

        Args:
            task: Task identifier.

        Returns:
            True if template exists, False otherwise.
        """
        registry = _get_prompt_registry()
        return registry.has_template(self.agent_role, task)

    def get_system_prompt_from_template(self) -> str | None:
        """Get system prompt from template if available.

        Returns:
            Rendered system prompt string, or None if no template exists.
        """
        registry = _get_prompt_registry()
        if registry.has_system(self.agent_role):
            return registry.render_system(self.agent_role)
        return None

    @classmethod
    def get_registry(cls) -> PromptRegistry:
        """Get the shared prompt registry.

        Returns:
            The singleton PromptRegistry instance.
        """
        return _get_prompt_registry()

    def __repr__(self) -> str:
        """Return string representation of the agent.

        Returns:
            String showing agent class name, display name, and model.
        """
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
