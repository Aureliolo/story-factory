"""Shared LLM client utilities for services.

Provides native Ollama client for structured outputs in services that don't use agents.
Uses ollama.Client.chat() with `format=` parameter for grammar-constrained JSON output.
All calls use stream=True to prevent HTTP read-timeout on long-running generations
(e.g. qwen3 thinking mode can produce thousands of internal tokens before output).
"""

import logging
import threading
import time

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from src.settings import Settings
from src.utils.exceptions import LLMError
from src.utils.streaming import consume_stream

logger = logging.getLogger(__name__)

# Module-level cache for Ollama clients (keyed by (url, timeout))
_ollama_clients: dict[tuple[str, float], ollama.Client] = {}
_ollama_clients_lock = threading.Lock()

# Cache for model context sizes (keyed by model name)
_model_context_cache: dict[str, int | None] = {}
_model_context_cache_lock = threading.Lock()


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text string.

    Uses the rough heuristic of ~4 characters per token, which is a reasonable
    approximation for English text with most tokenizers.

    Args:
        text: Input text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    return len(text) // 4


def warn_if_prompt_too_large(prompt: str, model: str, context_size: int, max_tokens: int) -> None:
    """Log a warning if the prompt + max_tokens may exceed the context window.

    Uses a 90% threshold to account for system prompt overhead and formatting
    tokens that are not included in the prompt text.

    Args:
        prompt: The prompt text to check.
        model: Model identifier (for log message).
        context_size: Configured context window size in tokens.
        max_tokens: Maximum tokens to generate (num_predict).
    """
    estimated_prompt_tokens = estimate_token_count(prompt)
    total_estimated = estimated_prompt_tokens + max_tokens
    threshold = int(context_size * 0.9)

    if total_estimated > threshold:
        logger.warning(
            "Prompt (~%d tokens) + max_tokens (%d) may exceed context_size (%d) "
            "for model %s. Output may be truncated.",
            estimated_prompt_tokens,
            max_tokens,
            context_size,
            model,
        )


def get_model_context_size(client: ollama.Client, model: str) -> int | None:
    """Query and cache the context size for a model.

    Args:
        client: Ollama client instance.
        model: Model identifier.

    Returns:
        Context size in tokens, or None if unavailable.
    """
    if model in _model_context_cache:
        return _model_context_cache[model]

    with _model_context_cache_lock:
        # Double-check after acquiring lock
        if model in _model_context_cache:
            return _model_context_cache[model]

        try:
            info = client.show(model)
            # Extract context length from model info
            model_info = info.get("model_info", {})
            context_length = None
            for key, value in model_info.items():
                if "context_length" in key:
                    context_length = int(value)
                    break
            _model_context_cache[model] = context_length
            if context_length:
                logger.debug("Model %s context size: %d tokens", model, context_length)
            return context_length
        except Exception as e:
            # Don't cache on transient errors — let the next call retry
            logger.debug("Could not query context size for model %s: %s", model, e)
            return None


def validate_context_size(client: ollama.Client, model: str, configured_context_size: int) -> int:
    """Validate configured context size against the model's actual limit.

    If the model's native context limit is smaller than the configured value,
    logs a warning and returns the model's limit to prevent silent truncation.

    Args:
        client: Ollama client instance.
        model: Model identifier.
        configured_context_size: The context size from settings.

    Returns:
        The effective context size to use (min of configured and model limit).
    """
    model_limit = get_model_context_size(client, model)
    if model_limit is not None and model_limit < configured_context_size:
        logger.warning(
            "Model %s has context limit of %d tokens, but configured context_size "
            "is %d. Capping to model limit to prevent truncation.",
            model,
            model_limit,
            configured_context_size,
        )
        return model_limit
    return configured_context_size


def get_ollama_client(settings: Settings, model_id: str | None = None) -> ollama.Client:
    """Get or create an Ollama client for the given settings.

    The client is cached based on URL and timeout to avoid recreating it for each call.
    Timeout is scaled based on model size when model_id is provided.
    Thread-safe via double-checked locking.

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
        with _ollama_clients_lock:
            if cache_key not in _ollama_clients:
                client = ollama.Client(host=settings.ollama_url, timeout=timeout)
                _ollama_clients[cache_key] = client
                logger.debug(
                    f"Created Ollama client for {settings.ollama_url} (timeout={timeout:.0f}s)"
                )

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
        LLMError: If generation fails after all retries or on non-retryable Ollama errors.
        ValueError: If max_retries < 1.
    """
    client = get_ollama_client(settings, model_id=model)

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

    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    # NOTE: Context size validation is intentionally applied only in generate_structured()
    # because JSON truncation from exceeding context limits causes parse failures here.
    # Free-form text generation in BaseAgent.generate() tolerates truncation gracefully.
    effective_context_size = validate_context_size(client, model, settings.context_size)

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            stream = client.chat(
                model=model,
                messages=messages,
                format=json_schema,
                options={
                    "temperature": temperature,
                    "num_ctx": effective_context_size,
                },
                stream=True,
            )
            response = consume_stream(stream)
            duration = time.time() - start_time

            # Extract token counts from streamed response
            prompt_tokens = response.get("prompt_eval_count")
            completion_tokens = response.get("eval_count")

            content = response["message"]["content"]
            result = response_model.model_validate_json(content)

            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            logger.info(
                "LLM call complete: model=%s, schema=%s, %.2fs, tokens: %s+%s=%s",
                model,
                response_model.__name__,
                duration,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
            return result

        except (ValidationError, KeyError, TypeError) as e:
            last_error = e
            logger.warning(
                "Structured output validation/parsing failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                continue  # Retry with same prompt

        except (ConnectionError, TimeoutError, httpx.TimeoutException, httpx.TransportError) as e:
            last_error = e
            logger.warning(
                "Transient error in structured output (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                backoff = min(2**attempt, 10)
                logger.debug("Backing off %.1fs before retry", backoff)
                time.sleep(backoff)
                continue

        except ollama.ResponseError as e:
            logger.error("Ollama response error during structured generation: %s", e)
            raise LLMError(
                f"Structured generation failed for {response_model.__name__}: {e}"
            ) from e

    # All retries exhausted
    logger.error("Structured output generation failed after %d attempts", max_retries)
    raise LLMError(
        f"Structured generation failed for {response_model.__name__} "
        f"after {max_retries} attempts: {last_error}"
    ) from last_error
