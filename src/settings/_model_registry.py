"""Recommended models registry for Story Factory.

This is a curated list for the UI - auto-selection works with ANY installed model.
Tags indicate which roles the model is particularly good for.

Judge tag policy (Issue #228, updated #294, Feb 2026):
  Models with the ``judge`` tag have been empirically validated via
  ``scripts/evaluate_judge_accuracy.py``.  Only models that achieve MAE < 2.5
  and Spearman rank correlation > 0.85 on the parametric prompt variant
  (no example scores) earn the tag.  Untested models (e.g. 70B models that
  timeout during benchmarking) do NOT receive the tag.
  See ``output/diagnostics/judge_accuracy_20260210_063748.json`` for latest results.
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
        # No judge tag: MAE=2.45, rank=0.76 parametric — marginal (Issue #294).
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
        # No judge tag: MAE=2.53, rank=0.74 parametric — fails both criteria (Issue #294).
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
        # Judge: MAE=2.12, rank=0.96 parametric (Issue #294).
        "tags": ["architect", "continuity", "interviewer", "suggestion", "judge"],
    },
    # === GOOGLE GEMMA ===
    # Gemma3 12B: best empirical judge model (Issue #294).
    "gemma3:12b": {
        "name": "Gemma 3 12B",
        "size_gb": 8,
        "vram_required": 10,
        "quality": 8,
        "speed": 8,
        "uncensored": False,
        "description": "Best empirical judge — MAE 1.58, rank 0.98, zero copying",
        # Judge: MAE=1.58, rank=0.98 parametric, best tested (Issue #294).
        "tags": ["continuity", "judge"],
    },
    "gemma3:4b": {
        "name": "Gemma 3 4B",
        "size_gb": 3,
        "vram_required": 4,
        "quality": 6,
        "speed": 9,
        "uncensored": False,
        "description": "Strong judge for its size, good structured output",
        # Judge: MAE=2.04, rank=0.94 parametric (Issue #294).
        "tags": ["judge"],
    },
    # === MICROSOFT PHI ===
    "phi4:14b": {
        "name": "Phi-4 14B",
        "size_gb": 9,
        "vram_required": 12,
        "quality": 8,
        "speed": 7,
        "uncensored": False,
        "description": "Near-perfect rank correlation as judge, strong reasoning",
        # Judge: MAE=1.96, rank=0.97 parametric (Issue #294).
        "tags": ["architect", "continuity", "judge"],
    },
    # Quality 7 reasoning: architect, continuity, interviewer. No suggestion.
    "huihui_ai/qwen3-abliterated:8b": {
        "name": "Qwen3 8B Abliterated",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "uncensored": True,
        "description": "Good reasoning at smaller size",
        # No judge tag: MAE=2.44, rank=0.85 parametric — borderline (Issue #294).
        "tags": ["architect", "continuity", "interviewer"],
    },
    # === HIGH-END ===
    # 70B+ models: Large enough to excel at everything.
    # WARNING: 80% GPU residency rule applies. On a 24GB GPU, max model ~30GB.
    # These models require 48GB+ VRAM to avoid heavy CPU offloading (5-10x slower).
    # See MIN_GPU_RESIDENCY in src/services/model_mode_service/_vram.py.
    "huihui_ai/llama3.3-abliterated:70b": {
        "name": "Llama 3.3 70B Abliterated",
        "size_gb": 40,
        "vram_required": 48,
        "quality": 10,
        "speed": 5,
        "uncensored": True,
        "description": "Premium reasoning, excellent for complex story architecture",
        # No judge tag: untested — timed out during benchmark (Issue #294).
        "tags": [
            "writer",
            "editor",
            "architect",
            "continuity",
            "interviewer",
            "suggestion",
        ],
    },
    "huihui_ai/llama3.3-abliterated:70b-instruct-q4_K_M": {
        "name": "Llama 3.3 70B Q4_K_M",
        "size_gb": 43,
        "vram_required": 48,
        "quality": 9,
        "speed": 4,
        "uncensored": True,
        "description": "Quantized 70B, needs 48GB+ VRAM (43GB weights alone)",
        # No judge tag: untested — timed out during benchmark (Issue #294).
        "tags": [
            "writer",
            "editor",
            "architect",
            "continuity",
            "interviewer",
            "suggestion",
        ],
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
        # No judge tag: untested — not installed during benchmark (Issue #294).
        "tags": [
            "writer",
            "editor",
            "architect",
            "continuity",
            "interviewer",
            "suggestion",
        ],
    },
    # === SMALL / FAST ===
    # Quality 6: Strong structured output for its size, best small-model judge (Issue #294).
    "phi4-mini": {
        "name": "Phi-4 Mini 3.8B",
        "size_gb": 2.3,
        "vram_required": 4,
        "quality": 6,
        "speed": 9,
        "uncensored": False,
        "description": "Microsoft reasoning model, best small-model judge — produces independent scores",
        # Judge: MAE=1.99, rank=1.00 parametric (Issue #294).
        "tags": ["judge"],
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
