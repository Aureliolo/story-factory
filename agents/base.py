"""Base agent class for all story factory agents."""

import logging
import time

import ollama

from settings import ModelInfo, Settings, get_model_info
from utils.error_handling import handle_ollama_errors
from utils.logging_config import log_performance

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to Ollama."""

    pass


class LLMGenerationError(LLMError):
    """Raised when generation fails after retries."""

    pass


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

        self.client = ollama.Client(host=self.settings.ollama_url)

    @classmethod
    @handle_ollama_errors(default_return=(False, "Ollama connection failed"), raise_on_error=False)
    def check_ollama_health(cls, ollama_url: str = "http://localhost:11434") -> tuple[bool, str]:
        """Check if Ollama is running and accessible.

        Returns:
            Tuple of (is_healthy, message)
        """
        client = ollama.Client(host=ollama_url)
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
        except Exception as e:
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
        """Generate a response from the agent with retry logic and performance tracking."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({"role": "system", "content": f"CURRENT STORY CONTEXT:\n{context}"})

        messages.append({"role": "user", "content": prompt})

        use_model = model or self.model
        use_temp = temperature or self.temperature

        last_error: Exception | None = None
        delay = self.RETRY_DELAY

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
                    logger.warning(f"{self.name}: Connection error on attempt {attempt + 1}: {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        logger.info(f"{self.name}: Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= self.RETRY_BACKOFF

                except ollama.ResponseError as e:
                    # Model-specific errors (model not found, etc.) - don't retry
                    logger.error(f"{self.name}: Ollama response error: {e}")
                    raise LLMGenerationError(f"Model error: {e}") from e

                except Exception as e:
                    last_error = e
                    logger.warning(f"{self.name}: Error on attempt {attempt + 1}: {e}")
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
