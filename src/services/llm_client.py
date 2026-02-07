"""Shared LLM client utilities for services.

Provides native Ollama client for structured outputs in services that don't use agents.
Uses ollama.Client.chat() with `format=` parameter for grammar-constrained JSON output.
"""

import logging
import time

import ollama
from pydantic import BaseModel, ValidationError

from src.settings import Settings

logger = logging.getLogger(__name__)

# Module-level cache for Ollama clients (keyed by (url, timeout))
_ollama_clients: dict[tuple[str, float], ollama.Client] = {}


def get_ollama_client(settings: Settings, model_id: str | None = None) -> ollama.Client:
    """Get or create an Ollama client for the given settings.

    The client is cached based on URL and timeout to avoid recreating it for each call.
    Timeout is scaled based on model size when model_id is provided.

    Args:
        settings: Application settings with ollama_url and ollama_timeout.
        model_id: Optional model ID for timeout scaling. If None, uses base timeout.

    Returns:
        Ollama client configured for the given settings.
    """
    # Get timeout (scaled by model size if model_id provided)
    timeout = settings.get_scaled_timeout(model_id) if model_id else float(settings.ollama_timeout)

    cache_key = (settings.ollama_url, timeout)

    if cache_key not in _ollama_clients:
        client = ollama.Client(host=settings.ollama_url, timeout=timeout)
        _ollama_clients[cache_key] = client
        logger.debug(f"Created Ollama client for {settings.ollama_url} (timeout={timeout:.0f}s)")

    return _ollama_clients[cache_key]


def generate_structured[T: BaseModel](
    settings: Settings,
    model: str,
    prompt: str,
    response_model: type[T],
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> T:
    """Generate structured output using native Ollama format parameter.

    This is a standalone function for services that don't use BaseAgent.
    Uses ollama.Client.chat() with `format=` set to the Pydantic model's
    JSON schema for grammar-constrained output.

    Args:
        settings: Application settings.
        model: The Ollama model to use.
        prompt: The user prompt to send.
        response_model: Pydantic model class defining the expected output structure.
        system_prompt: Optional system prompt.
        temperature: Temperature for generation (default 0.1 for structured output).
        max_retries: Maximum number of retries on validation failure.

    Returns:
        Instance of response_model with validated data.

    Raises:
        Exception: If generation fails after all retries.
    """
    client = get_ollama_client(settings, model_id=model)

    # Add /no_think prefix for Qwen models
    if system_prompt and "qwen" in model.lower():
        system_prompt = f"/no_think\n{system_prompt}"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Get JSON schema for format parameter
    json_schema = response_model.model_json_schema()

    logger.debug(
        f"Generating structured output: model={model}, response_model={response_model.__name__}, "
        f"temperature={temperature}, max_retries={max_retries}"
    )

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat(
                model=model,
                messages=messages,
                format=json_schema,
                options={
                    "temperature": temperature,
                    "num_ctx": settings.context_size,
                },
            )
            duration = time.time() - start_time

            # Extract token counts from native Ollama response
            prompt_tokens = response.get("prompt_eval_count")
            completion_tokens = response.get("eval_count")

            content = response["message"]["content"]
            result = response_model.model_validate_json(content)

            logger.debug(
                f"Structured output received: {response_model.__name__} "
                f"({duration:.2f}s, tokens: {prompt_tokens}+{completion_tokens})"
            )
            return result

        except ValidationError as e:
            last_error = e
            logger.warning(
                "Structured output validation failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                continue  # Retry with same prompt

    # All retries exhausted
    raise last_error  # type: ignore[misc]
