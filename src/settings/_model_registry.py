"""Recommended models registry for Story Factory.

This is a curated list for the UI - auto-selection works with ANY installed model.
Tags indicate which roles the model is particularly good for.
"""

from src.settings._types import ModelInfo

RECOMMENDED_MODELS: dict[str, ModelInfo] = {
    # === CREATIVE WRITING SPECIALISTS ===
    # Prose-optimized models: writer, editor, suggestion, interviewer. NOT architect.
    "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0": {
        "name": "Celeste V1.9 12B",
        "size_gb": 13,
        "vram_required": 14,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "Purpose-built for fiction writing, excellent prose quality",
        "tags": ["writer", "editor", "suggestion", "interviewer"],
    },
    "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit": {
        "name": "Dark Champion 18B MOE",
        "size_gb": 11,
        "vram_required": 14,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "Exceptional fiction/RP, outstanding prose quality",
        "tags": ["writer", "editor", "suggestion", "interviewer"],
    },
    # === GENERAL PURPOSE ===
    # Quality 7: continuity, interviewer, suggestion. No writer/editor/architect (need Q8+).
    "huihui_ai/dolphin3-abliterated:8b": {
        "name": "Dolphin 3.0 8B Abliterated",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "uncensored": True,
        "description": "Fast, compliant, no Chinese output - great all-rounder",
        "tags": ["continuity", "interviewer", "suggestion"],
    },
    # Quality 8: All roles except writer (editing-focused, not prose creation).
    "CognitiveComputations/dolphin-mistral-nemo:12b": {
        "name": "Dolphin Mistral Nemo 12B",
        "size_gb": 7,
        "vram_required": 10,
        "quality": 8,
        "speed": 8,
        "uncensored": True,
        "description": "128K context, excellent for editing and refinement",
        "tags": ["editor", "architect", "continuity", "interviewer", "suggestion"],
    },
    # === REASONING SPECIALISTS ===
    # Reasoning-optimized: architect, continuity, interviewer, suggestion. NOT writer/editor.
    "huihui_ai/qwen3-abliterated:30b": {
        "name": "Qwen3 30B Abliterated (MoE)",
        "size_gb": 18,
        "vram_required": 18,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "MoE (30B/3B active), strong reasoning - excellent for architect",
        "tags": ["architect", "continuity", "interviewer", "suggestion"],
    },
    # Quality 7 reasoning: architect, continuity, interviewer, validator. No suggestion.
    "huihui_ai/qwen3-abliterated:8b": {
        "name": "Qwen3 8B Abliterated",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "uncensored": True,
        "description": "Good reasoning at smaller size",
        "tags": ["architect", "continuity", "interviewer", "validator"],
    },
    # === HIGH-END ===
    # 70B+ models: Large enough to excel at everything
    "huihui_ai/llama3.3-abliterated:70b": {
        "name": "Llama 3.3 70B Abliterated",
        "size_gb": 40,
        "vram_required": 48,
        "quality": 10,
        "speed": 5,
        "uncensored": True,
        "description": "Premium reasoning, excellent for complex story architecture",
        "tags": ["writer", "editor", "architect", "continuity", "interviewer", "suggestion"],
    },
    "huihui_ai/llama3.3-abliterated:70b-instruct-q4_K_M": {
        "name": "Llama 3.3 70B Q4_K_M",
        "size_gb": 43,
        "vram_required": 24,
        "quality": 9,
        "speed": 4,
        "uncensored": True,
        "description": "Quantized 70B, fits 24GB VRAM",
        "tags": ["writer", "editor", "architect", "continuity", "interviewer", "suggestion"],
    },
    # Creative 70B: Best at prose, good at everything due to size
    "vanilj/midnight-miqu-70b-v1.5": {
        "name": "Midnight Miqu 70B",
        "size_gb": 42,
        "vram_required": 48,
        "quality": 10,
        "speed": 4,
        "uncensored": True,
        "description": "Premium creative writer - writes like a novelist",
        "tags": ["writer", "editor", "architect", "continuity", "interviewer", "suggestion"],
    },
    # === SMALL / FAST ===
    # Quality 3-5: Validator only, maybe basic interviewer
    "qwen3:0.6b": {
        "name": "Qwen3 0.6B",
        "size_gb": 0.5,
        "vram_required": 2,
        "quality": 3,
        "speed": 10,
        "uncensored": False,
        "description": "Tiny, ultra-fast - for validator only",
        "tags": ["validator"],
    },
    "smollm2:1.7b": {
        "name": "SmolLM2 1.7B",
        "size_gb": 1.2,
        "vram_required": 2,
        "quality": 4,
        "speed": 10,
        "uncensored": True,
        "description": "Small but capable - good for validation",
        "tags": ["validator"],
    },
    "qwen3:4b": {
        "name": "Qwen3 4B",
        "size_gb": 2.5,
        "vram_required": 4,
        "quality": 5,
        "speed": 9,
        "uncensored": True,
        "description": "Fast inference, good for quick tasks",
        "tags": ["validator", "interviewer"],
    },
    # === EMBEDDING MODELS ===
    # Used for semantic duplicate detection during world generation.
    # Not agent models — selected via the embedding_model setting, not auto-selection.
    "mxbai-embed-large": {
        "name": "MxBAI Embed Large",
        "size_gb": 0.7,
        "vram_required": 1,
        "quality": 9,
        "speed": 9,
        "uncensored": False,
        "description": "High-quality 335M embeddings, 1024-dim vectors",
        "tags": ["embedding"],
    },
    "snowflake-arctic-embed:335m": {
        "name": "Snowflake Arctic Embed 335M",
        "size_gb": 0.7,
        "vram_required": 1,
        "quality": 9,
        "speed": 9,
        "uncensored": False,
        "description": "Top-ranked embedding model from Snowflake",
        "tags": ["embedding"],
    },
    "bge-m3": {
        "name": "BGE-M3",
        "size_gb": 1.2,
        "vram_required": 2,
        "quality": 10,
        "speed": 8,
        "uncensored": False,
        "description": "Top-ranked 567M multilingual embedding model, 1024-dim vectors",
        "tags": ["embedding"],
    },
}


def get_embedding_prefix(model_name: str) -> str:
    """Look up the embedding prompt prefix for a model.

    Some embedding models require a specific prompt prefix for optimal results.
    Models not in the registry, or models without a prefix, return ``""``.

    Args:
        model_name: The Ollama model name (e.g., ``"mxbai-embed-large"``).

    Returns:
        The prompt prefix string, or ``""`` if no prefix is needed.
    """
    info = RECOMMENDED_MODELS.get(model_name)
    if info is None:
        return ""
    return info.get("embedding_prefix", "")
