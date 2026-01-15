"""Utility functions for working with model identifiers."""


def extract_model_name(model_id: str) -> str:
    """Extract display name from a model ID.

    Handles both simple names and HuggingFace-style paths (org/model).

    Args:
        model_id: The full model identifier (e.g., "llama3.3" or "qwen/qwen3-30b")

    Returns:
        The display name (e.g., "llama3.3" or "qwen3-30b")

    Examples:
        >>> extract_model_name("llama3.3")
        'llama3.3'
        >>> extract_model_name("qwen/qwen3-30b")
        'qwen3-30b'
    """
    return model_id.split("/")[-1] if "/" in model_id else model_id
