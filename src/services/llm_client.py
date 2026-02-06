"""Shared LLM client utilities for services.

Provides Instructor client for structured outputs in services that don't use agents.
"""

import logging

import instructor
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from src.settings import Settings
from src.utils.exceptions import summarize_llm_error

logger = logging.getLogger(__name__)

# Module-level cache for instructor clients (keyed by settings hash)
_instructor_clients: dict[int, instructor.Instructor] = {}


def get_instructor_client(settings: Settings, model_id: str | None = None) -> instructor.Instructor:
    """Get or create an Instructor client for the given settings.

    The client is cached based on settings and model to avoid recreating it for each call.
    Timeout is scaled based on model size when model_id is provided.

    Args:
        settings: Application settings with ollama_url and ollama_timeout.
        model_id: Optional model ID for timeout scaling. If None, uses base timeout.

    Returns:
        Instructor client configured for Ollama's OpenAI-compatible endpoint.
    """
    # Get timeout (scaled by model size if model_id provided)
    timeout = settings.get_scaled_timeout(model_id) if model_id else float(settings.ollama_timeout)

    # Create a simple hash based on relevant settings and timeout
    settings_key = hash((settings.ollama_url, timeout))

    if settings_key not in _instructor_clients:
        openai_client = OpenAI(
            base_url=f"{settings.ollama_url}/v1",
            api_key="ollama",  # Required but not used by Ollama
            timeout=timeout,
        )
        client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )
        client.on(
            "parse:error",
            lambda e: logger.warning(
                "Structured output validation failed (will retry): %s",
                summarize_llm_error(e) if isinstance(e, Exception) else str(e)[:300],
            ),
        )
        client.on(
            "completion:error",
            lambda e: logger.warning(
                "LLM API error during structured output (will retry): %s",
                summarize_llm_error(e) if isinstance(e, Exception) else str(e)[:300],
            ),
        )
        _instructor_clients[settings_key] = client
        logger.debug(
            f"Created instructor client for {settings.ollama_url} (timeout={timeout:.0f}s)"
        )

    return _instructor_clients[settings_key]


def generate_structured[T: BaseModel](
    settings: Settings,
    model: str,
    prompt: str,
    response_model: type[T],
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> T:
    """Generate structured output with automatic validation and retry.

    This is a standalone function for services that don't use BaseAgent.
    Uses Instructor library to enforce JSON schema at the API level.

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
    client = get_instructor_client(settings, model_id=model)

    # Add /no_think prefix for Qwen models
    if system_prompt and "qwen" in model.lower():
        system_prompt = f"/no_think\n{system_prompt}"

    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        f"Generating structured output: model={model}, response_model={response_model.__name__}, "
        f"temperature={temperature}, max_retries={max_retries}"
    )

    result = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        temperature=temperature,
    )

    logger.debug(f"Structured output received: {response_model.__name__}")
    return result
