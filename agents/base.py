"""Base agent class for all story factory agents."""

import logging
import threading
import time

import ollama

from settings import ModelInfo, Settings, get_model_info
from utils.error_handling import handle_ollama_errors
from utils.exceptions import LLMConnectionError, LLMError, LLMGenerationError
from utils.logging_config import log_performance

logger = logging.getLogger(__name__)

# Rate limiting: maximum concurrent LLM requests
MAX_CONCURRENT_LLM_REQUESTS = 2
_llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

# Re-export exceptions for backward compatibility
__all__ = ["BaseAgent", "LLMError", "LLMConnectionError", "LLMGenerationError"]


class BaseAgent:
    """Base class for all agents in the story factory."""

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    RETRY_BACKOFF = 2  # multiplier

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

        # Create Ollama client with configurable timeout to prevent hanging
        self.client = ollama.Client(
            host=self.settings.ollama_url, timeout=float(self.settings.ollama_timeout)
        )

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
    ) -> str:
        """Generate a response from the agent with retry logic, rate limiting, and performance tracking.

        Uses a semaphore to limit concurrent LLM requests and prevent overloading
        the Ollama server.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({"role": "system", "content": f"CURRENT STORY CONTEXT:\n{context}"})

        messages.append({"role": "user", "content": prompt})

        use_model = model or self.model
        use_temp = temperature or self.temperature

        last_error: Exception | None = None
        delay = self.RETRY_DELAY

        # Acquire semaphore to limit concurrent requests
        with _llm_semaphore:
            with log_performance(logger, f"{self.name} generation"):
                for attempt in range(self.MAX_RETRIES):
                    try:
                        logger.info(
                            f"{self.name}: Calling LLM ({use_model}) attempt {attempt + 1}/{self.MAX_RETRIES}"
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
                        logger.info(
                            f"{self.name}: LLM response received ({len(content)} chars, {duration:.2f}s)"
                        )
                        return content

                    except ConnectionError as e:
                        last_error = e
                        logger.warning(
                            f"{self.name}: Connection error on attempt {attempt + 1}: {e}"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            logger.info(f"{self.name}: Retrying in {delay}s...")
                            time.sleep(delay)
                            delay *= self.RETRY_BACKOFF

                    except ollama.ResponseError as e:
                        # Model-specific errors (model not found, etc.) - don't retry
                        logger.error(f"{self.name}: Ollama response error: {e}")
                        raise LLMGenerationError(f"Model error: {e}") from e

                    except TimeoutError as e:
                        last_error = e
                        logger.warning(f"{self.name}: Timeout on attempt {attempt + 1}: {e}")
                        if attempt < self.MAX_RETRIES - 1:
                            logger.info(f"{self.name}: Retrying in {delay}s...")
                            time.sleep(delay)
                            delay *= self.RETRY_BACKOFF

                # All retries failed
                logger.error(f"{self.name}: All {self.MAX_RETRIES} attempts failed")
                raise LLMGenerationError(
                    f"Failed to generate after {self.MAX_RETRIES} attempts: {last_error}"
                ) from last_error

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        return get_model_info(self.model)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
