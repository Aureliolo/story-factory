"""Web-configurable settings for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypedDict
from urllib.parse import urlparse

from src.memory.mode_models import LearningTrigger, VramStrategy

# Configure module logger
logger = logging.getLogger(__name__)

# Cross-platform subprocess flags (CREATE_NO_WINDOW only exists on Windows)
_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Centralized paths for story and world output files
# Go up from src/ to project root, then into output/
STORIES_DIR = Path(__file__).parent.parent / "output" / "stories"
WORLDS_DIR = Path(__file__).parent.parent / "output" / "worlds"
BACKUPS_DIR = Path(__file__).parent.parent / "output" / "backups"


class ModelInfo(TypedDict):
    """Type definition for model information."""

    name: str
    size_gb: float
    vram_required: int
    quality: int | float
    speed: int
    uncensored: bool
    description: str
    # Tags for role suitability - list of agent roles this model is good for
    tags: list[str]


class AgentRoleInfo(TypedDict):
    """Type definition for agent role information."""

    name: str
    description: str


# Temperature decay curve options for world quality refinement
REFINEMENT_TEMP_DECAY_CURVES: dict[str, str] = {
    "linear": "Linear",
    "exponential": "Exponential",
    "step": "Step",
}


# Agent role definitions
AGENT_ROLES: dict[str, AgentRoleInfo] = {
    "interviewer": {
        "name": "Interviewer",
        "description": "Gathers story requirements",
    },
    "architect": {
        "name": "Architect",
        "description": "Designs story structure",
    },
    "writer": {
        "name": "Writer",
        "description": "Writes prose",
    },
    "editor": {
        "name": "Editor",
        "description": "Polishes prose",
    },
    "continuity": {
        "name": "Continuity Checker",
        "description": "Checks for plot holes",
    },
    "validator": {
        "name": "Validator",
        "description": "Validates AI responses",
    },
    "suggestion": {
        "name": "Suggestion Assistant",
        "description": "Generates writing prompts and suggestions",
    },
}


# Recommended/supported models registry
# This is a curated list for the UI - auto-selection works with ANY installed model
# Tags indicate which roles the model is particularly good for
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
    # Quality 7 reasoning: architect, continuity, interviewer. No suggestion (reasoning focus).
    "huihui_ai/qwen3-abliterated:8b": {
        "name": "Qwen3 8B Abliterated",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "uncensored": True,
        "description": "Good reasoning at smaller size",
        "tags": ["architect", "continuity", "interviewer"],
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
}


@dataclass
class AgentSettings:
    """Settings for a single agent."""

    model: str = "auto"  # "auto" means auto-select based on role
    temperature: float = 0.8


@dataclass
class Settings:
    """Application settings, stored as JSON."""

    # General
    ollama_url: str = "http://localhost:11434"
    context_size: int = 32768
    max_tokens: int = 8192  # Increased to support longer chapters (2000+ words)

    # Default model for all agents - "auto" means select automatically
    # If set to a specific model, that model will be used as fallback
    default_model: str = "auto"

    # Per-agent model settings
    use_per_agent_models: bool = True
    agent_models: dict[str, str] = field(
        default_factory=lambda: {
            "interviewer": "auto",
            "architect": "auto",
            "writer": "auto",
            "editor": "auto",
            "continuity": "auto",
            "validator": "auto",  # Small, fast model for validation
            "suggestion": "auto",  # Creative model for writing suggestions
        }
    )

    # Custom model tags - maps model_id to list of role tags
    # Example: {"my-model:7b": ["writer", "editor"], "fast-model:3b": ["validator"]}
    # These are merged with RECOMMENDED_MODELS tags for selection
    custom_model_tags: dict[str, list[str]] = field(default_factory=dict)

    # Agent temperatures
    agent_temperatures: dict[str, float] = field(
        default_factory=lambda: {
            "interviewer": 0.7,
            "architect": 0.85,
            "writer": 0.9,
            "editor": 0.6,
            "continuity": 0.3,
            "validator": 0.1,  # Very low for consistent validation
            "suggestion": 0.8,  # Creative for writing prompts
        }
    )

    # Task-specific temperatures (override agent defaults for specific operations)
    temp_brief_extraction: float = 0.3  # Lower for structured JSON output
    temp_edit_suggestions: float = 0.5  # Moderate for creative suggestions
    temp_plot_checking: float = 0.2  # Very low for deterministic analysis
    temp_capability_check: float = 0.1  # Very low for yes/no capability check
    temp_model_evaluation: float = 0.8  # Higher for creative model evaluation

    # Revision settings
    revision_temperature: float = 0.7  # Lower temperature for more focused revisions

    # Context truncation limits (characters)
    # These control how much context is sent to the LLM to stay within limits
    previous_chapter_context_chars: int = 2000  # End of previous chapter for continuity
    chapter_analysis_chars: int = 4000  # Chapter content for analysis
    full_text_preview_chars: int = 3000  # Text preview for editing suggestions

    # Interaction settings
    interaction_mode: str = "checkpoint"
    chapters_between_checkpoints: int = 3
    max_revision_iterations: int = 3

    # Comparison mode
    comparison_models: list[str] = field(default_factory=list)

    # UI settings
    dark_mode: bool = True
    last_project_id: str | None = None  # Remember last opened project

    # Backup settings
    backup_folder: str = str(BACKUPS_DIR)  # Folder to store project backups
    backup_verify_on_restore: bool = True  # Verify backup integrity before restore

    # Data integrity settings
    entity_version_retention: int = 10  # Keep last N versions per entity

    # Generation mode settings
    current_mode: str = "balanced"  # ID of active generation mode
    use_mode_system: bool = True  # Whether to use mode-based model selection
    vram_strategy: str = "adaptive"  # sequential, parallel, adaptive

    # Learning/tuning settings
    learning_triggers: list[str] = field(
        default_factory=lambda: [
            LearningTrigger.AFTER_PROJECT.value
        ]  # off, after_project, periodic, continuous
    )
    learning_autonomy: str = "balanced"  # manual, cautious, balanced, aggressive, experimental
    learning_periodic_interval: int = 5  # Chapters between periodic analysis
    learning_min_samples: int = 5  # Minimum samples before making recommendations
    learning_confidence_threshold: float = 0.8  # For auto-applying in balanced mode

    # Timeout settings (in seconds)
    ollama_timeout: int = 120  # Timeout for Ollama API requests
    subprocess_timeout: int = 10  # Timeout for subprocess calls (ollama list, etc.)

    # World quality refinement settings
    world_quality_enabled: bool = True  # Enable quality refinement for world generation
    world_quality_max_iterations: int = 3  # Maximum refinement iterations per entity
    world_quality_threshold: float = 7.5  # Min score (0-10) - entities plateau at 7.5-7.8
    world_quality_creator_temp: float = 0.9  # Temperature for creative generation
    world_quality_judge_temp: float = 0.1  # Temperature for quality judgment
    world_quality_refinement_temp: float = 0.7  # Temperature for refinement passes
    world_quality_early_stopping_patience: int = 2  # Stop after N consecutive score degradations

    # Dynamic temperature settings for refinement (#167, #177)
    world_quality_refinement_temp_start: float = 0.7  # Starting temperature for refinement
    world_quality_refinement_temp_end: float = 0.35  # Ending temperature after decay
    world_quality_refinement_temp_decay: str = "linear"  # Decay curve: linear, exponential, step

    # Enhanced early stopping settings (#167)
    world_quality_early_stopping_min_iterations: int = 2  # Min iterations before early stop
    world_quality_early_stopping_variance_tolerance: float = 0.3  # Score variance tolerance

    # Circuit breaker settings (#175)
    circuit_breaker_enabled: bool = True  # Enable circuit breaker for LLM calls
    circuit_breaker_failure_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_success_threshold: int = 2  # Successes to close from half-open
    circuit_breaker_timeout: float = 60.0  # Seconds before half-open

    # Retry strategy settings (#168)
    retry_temp_increase: float = 0.15  # Temperature increase on retry attempt 2+
    retry_simplify_on_attempt: int = 3  # Attempt number to start simplifying prompts

    # Semantic duplicate detection settings (#176)
    semantic_duplicate_enabled: bool = False  # Opt-in: use embedding-based similarity
    semantic_duplicate_threshold: float = 0.85  # Cosine similarity threshold for duplicates
    embedding_model: str = "nomic-embed-text"  # Model for generating embeddings

    # Content guidelines checking settings
    content_check_enabled: bool = True  # Enable content guideline checking for generated content
    content_check_use_llm: bool = False  # Use LLM for more accurate checking (slower)

    # Judge consistency settings (for more reliable quality judgments)
    judge_consistency_enabled: bool = False  # Opt-in: enable judge consistency features
    judge_multi_call_enabled: bool = False  # Make multiple judge calls and average (expensive)
    judge_multi_call_count: int = 3  # Number of judge calls if multi_call_enabled
    judge_confidence_threshold: float = 0.7  # Min confidence for reliable decisions
    judge_outlier_detection: bool = True  # Detect and handle outlier scores
    judge_outlier_std_threshold: float = 2.0  # Std devs from mean to consider outlier
    judge_outlier_strategy: str = "median"  # How to handle outliers: median, mean, retry

    # World generation counts (randomized within range for variety)
    world_gen_characters_min: int = 4
    world_gen_characters_max: int = 12
    world_gen_locations_min: int = 3
    world_gen_locations_max: int = 10
    world_gen_factions_min: int = 1
    world_gen_factions_max: int = 6
    world_gen_items_min: int = 2
    world_gen_items_max: int = 10
    world_gen_concepts_min: int = 1
    world_gen_concepts_max: int = 6
    world_gen_relationships_min: int = 8
    world_gen_relationships_max: int = 25

    # World health and validation settings
    relationship_validation_enabled: bool = True  # Validate relationships on creation
    orphan_detection_enabled: bool = True  # Enable orphan entity detection
    circular_detection_enabled: bool = True  # Enable circular relationship detection
    circular_relationship_types: list[str] = field(
        default_factory=lambda: ["owns", "reports_to", "parent_of", "located_in", "contains"]
    )
    fuzzy_match_threshold: float = 0.8  # Threshold for fuzzy entity name matching

    # Relationship minimums per entity type/role (for suggestions)
    relationship_minimums: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "character": {"protagonist": 5, "antagonist": 4, "supporting": 2, "minor": 1},
            "location": {"central": 4, "significant": 2, "minor": 1},
            "faction": {"major": 4, "minor": 2},
            "item": {"key": 3, "common": 1},
            "concept": {"central_theme": 4, "minor_theme": 2},
        }
    )
    max_relationships_per_entity: int = 10  # Max relationships to suggest per entity

    # Prompt template settings
    prompt_templates_dir: str = "src/prompts/templates"
    prompt_metrics_enabled: bool = True

    # Graph visualization settings
    graph_filter_types: list[str] = field(
        default_factory=lambda: ["character", "location", "item", "faction", "concept"]
    )

    # LLM generation token limits (num_predict values)
    llm_tokens_character_create: int = 500
    llm_tokens_character_judge: int = 300
    llm_tokens_character_refine: int = 500
    llm_tokens_location_create: int = 400
    llm_tokens_location_judge: int = 200
    llm_tokens_location_refine: int = 600
    llm_tokens_faction_create: int = 600
    llm_tokens_faction_judge: int = 200
    llm_tokens_faction_refine: int = 600
    llm_tokens_item_create: int = 500
    llm_tokens_item_judge: int = 200
    llm_tokens_item_refine: int = 500
    llm_tokens_concept_create: int = 500
    llm_tokens_concept_judge: int = 200
    llm_tokens_concept_refine: int = 500
    llm_tokens_relationship_create: int = 500
    llm_tokens_relationship_judge: int = 200
    llm_tokens_relationship_refine: int = 500
    llm_tokens_mini_description: int = 50

    # Entity extraction limits
    entity_extract_locations_max: int = 10
    entity_extract_items_max: int = 5
    entity_extract_events_max: int = 5

    # Mini description settings
    mini_description_words_max: int = 15
    mini_description_temperature: float = 0.3

    # Workflow and orchestration limits
    orchestrator_cache_size: int = 10
    workflow_max_events: int = 100

    # LLM request limits
    llm_max_concurrent_requests: int = 2
    llm_max_retries: int = 3

    # Content truncation limits
    content_truncation_for_judgment: int = 3000

    # Ollama client timeouts (in seconds)
    ollama_health_check_timeout: float = 30.0
    ollama_list_models_timeout: float = 30.0
    ollama_pull_model_timeout: float = 600.0  # 10 minutes for large model downloads
    ollama_delete_model_timeout: float = 30.0
    ollama_check_update_timeout: float = 60.0
    ollama_generate_timeout: float = 60.0
    ollama_capability_check_timeout: float = 180.0  # 3 minutes for capability checks

    # Retry configuration
    llm_retry_delay: float = 2.0  # Base delay in seconds between retries
    llm_retry_backoff: float = 2.0  # Exponential backoff multiplier

    # Verification delays
    model_verification_sleep: float = 0.1  # Delay for model download verification

    # Validation thresholds
    validator_cjk_char_threshold: int = 5  # Max CJK chars allowed in English text
    validator_printable_ratio: float = 0.9  # Min ratio of printable characters
    validator_ai_check_min_length: int = 200  # Min response length for AI validation

    # Outline generation
    outline_variations_min: int = 3  # Min outline variations to generate
    outline_variations_max: int = 5  # Max outline variations to generate

    # Import extraction thresholds
    import_confidence_threshold: float = 0.7  # Flag for review if confidence below this
    import_default_confidence: float = 0.5  # Default confidence when not provided

    # World/plot generation limits
    world_description_summary_length: int = 500  # Truncate world desc if longer
    event_sentence_min_length: int = 20  # Min length for event sentences
    event_sentence_max_length: int = 200  # Max length for event sentences

    # User rating bounds
    user_rating_min: int = 1  # Minimum user rating value
    user_rating_max: int = 5  # Maximum user rating value

    # Model download detection
    model_download_threshold: float = 0.9  # Completion ratio to detect actual download

    # Story length chapter counts
    chapters_short_story: int = 3  # Chapters for short story
    chapters_novella: int = 10  # Chapters for novella
    chapters_novel: int = 25  # Chapters for novel
    chapters_default: int = 5  # Default if length not recognized

    # Import temperatures (override agent defaults for specific operations)
    temp_import_extraction: float = 0.3  # Low temp for consistent import extraction
    temp_interviewer_override: float = 0.9  # Higher temp for creative interviewer prompts

    # Token multipliers for extraction
    import_character_token_multiplier: int = 4  # Multiply base tokens for character extraction

    def save(self) -> None:
        """Save settings to JSON file."""
        # Validate before saving
        self.validate()
        with open(SETTINGS_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self) -> None:
        """
        Validate the Settings instance fields and enforce allowed ranges and formats.

        Raises:
            ValueError: If any field contains an invalid value (range, enum/choice, or format).
        """
        # Validate URL format
        try:
            parsed = urlparse(self.ollama_url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme in ollama_url: {self.ollama_url}")
            if not parsed.netloc:
                raise ValueError(f"Invalid URL (missing host) in ollama_url: {self.ollama_url}")
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Invalid ollama_url: {self.ollama_url} - {e}") from e

        # Validate numeric ranges
        if not 1024 <= self.context_size <= 128000:
            raise ValueError(
                f"context_size must be between 1024 and 128000, got {self.context_size}"
            )

        if not 256 <= self.max_tokens <= 32000:
            raise ValueError(f"max_tokens must be between 256 and 32000, got {self.max_tokens}")

        if not 1 <= self.chapters_between_checkpoints <= 20:
            raise ValueError(
                f"chapters_between_checkpoints must be between 1 and 20, got {self.chapters_between_checkpoints}"
            )

        if not 0 <= self.max_revision_iterations <= 10:
            raise ValueError(
                f"max_revision_iterations must be between 0 and 10, got {self.max_revision_iterations}"
            )

        # Validate interaction mode
        valid_modes = ["minimal", "checkpoint", "interactive", "collaborative"]
        if self.interaction_mode not in valid_modes:
            raise ValueError(
                f"interaction_mode must be one of {valid_modes}, got {self.interaction_mode}"
            )

        # Validate VRAM strategy (derived from enum to prevent drift)
        valid_vram_strategies = [strategy.value for strategy in VramStrategy]
        if self.vram_strategy not in valid_vram_strategies:
            raise ValueError(
                f"vram_strategy must be one of {valid_vram_strategies}, got {self.vram_strategy}"
            )

        # Validate temperatures
        expected_agents = set(AGENT_ROLES)

        unknown_temp_agents = set(self.agent_temperatures) - expected_agents
        if unknown_temp_agents:
            raise ValueError(
                f"Unknown agent(s) in agent_temperatures: {sorted(unknown_temp_agents)}; "
                f"expected only: {sorted(expected_agents)}"
            )

        for agent, temp in self.agent_temperatures.items():
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"Temperature for {agent} must be between 0.0 and 2.0, got {temp}")

        # Validate task-specific temperatures
        task_temps = [
            ("temp_brief_extraction", self.temp_brief_extraction),
            ("temp_edit_suggestions", self.temp_edit_suggestions),
            ("temp_plot_checking", self.temp_plot_checking),
            ("temp_capability_check", self.temp_capability_check),
            ("temp_model_evaluation", self.temp_model_evaluation),
        ]
        for name, temp in task_temps:
            if not 0.0 <= temp <= 2.0:
                raise ValueError(f"{name} must be between 0.0 and 2.0, got {temp}")

        # Validate learning settings
        valid_autonomy = ["manual", "cautious", "balanced", "aggressive", "experimental"]
        if self.learning_autonomy not in valid_autonomy:
            raise ValueError(
                f"learning_autonomy must be one of {valid_autonomy}, got {self.learning_autonomy}"
            )

        valid_triggers = ["off", "after_project", "periodic", "continuous"]
        for trigger in self.learning_triggers:
            if trigger not in valid_triggers:
                raise ValueError(
                    f"Invalid learning trigger: {trigger}; must be one of {valid_triggers}"
                )

        if not 1 <= self.learning_periodic_interval <= 20:
            raise ValueError(
                f"learning_periodic_interval must be between 1 and 20, got {self.learning_periodic_interval}"
            )

        if not 0.0 <= self.learning_confidence_threshold <= 1.0:
            raise ValueError(
                f"learning_confidence_threshold must be between 0.0 and 1.0, got {self.learning_confidence_threshold}"
            )

        # Validate data integrity settings
        if not 1 <= self.entity_version_retention <= 100:
            raise ValueError(
                f"entity_version_retention must be between 1 and 100, "
                f"got {self.entity_version_retention}"
            )
        if not isinstance(self.backup_verify_on_restore, bool):
            raise ValueError(
                f"backup_verify_on_restore must be a boolean, "
                f"got {type(self.backup_verify_on_restore).__name__}"
            )

        # Validate timeout settings
        if not 10 <= self.ollama_timeout <= 600:
            raise ValueError(
                f"ollama_timeout must be between 10 and 600 seconds, got {self.ollama_timeout}"
            )

        if not 5 <= self.subprocess_timeout <= 60:
            raise ValueError(
                f"subprocess_timeout must be between 5 and 60 seconds, got {self.subprocess_timeout}"
            )

        # Validate world quality settings
        if not 1 <= self.world_quality_max_iterations <= 10:
            raise ValueError(
                f"world_quality_max_iterations must be between 1 and 10, "
                f"got {self.world_quality_max_iterations}"
            )

        if not 0.0 <= self.world_quality_threshold <= 10.0:
            raise ValueError(
                f"world_quality_threshold must be between 0.0 and 10.0, "
                f"got {self.world_quality_threshold}"
            )

        if not 1 <= self.world_quality_early_stopping_patience <= 10:
            raise ValueError(
                f"world_quality_early_stopping_patience must be between 1 and 10, "
                f"got {self.world_quality_early_stopping_patience}"
            )

        for temp_name, temp_value in [
            ("world_quality_creator_temp", self.world_quality_creator_temp),
            ("world_quality_judge_temp", self.world_quality_judge_temp),
            ("world_quality_refinement_temp", self.world_quality_refinement_temp),
            ("world_quality_refinement_temp_start", self.world_quality_refinement_temp_start),
            ("world_quality_refinement_temp_end", self.world_quality_refinement_temp_end),
        ]:
            if not 0.0 <= temp_value <= 2.0:
                raise ValueError(f"{temp_name} must be between 0.0 and 2.0, got {temp_value}")

        # Validate dynamic temperature decay curve
        if self.world_quality_refinement_temp_decay not in REFINEMENT_TEMP_DECAY_CURVES:
            raise ValueError(
                f"world_quality_refinement_temp_decay must be one of "
                f"{list(REFINEMENT_TEMP_DECAY_CURVES.keys())}, "
                f"got {self.world_quality_refinement_temp_decay}"
            )

        # Validate enhanced early stopping settings
        if not 1 <= self.world_quality_early_stopping_min_iterations <= 10:
            raise ValueError(
                f"world_quality_early_stopping_min_iterations must be between 1 and 10, "
                f"got {self.world_quality_early_stopping_min_iterations}"
            )
        if not 0.0 <= self.world_quality_early_stopping_variance_tolerance <= 2.0:
            raise ValueError(
                f"world_quality_early_stopping_variance_tolerance must be between 0.0 and 2.0, "
                f"got {self.world_quality_early_stopping_variance_tolerance}"
            )

        # Validate circuit breaker settings
        if not 1 <= self.circuit_breaker_failure_threshold <= 20:
            raise ValueError(
                f"circuit_breaker_failure_threshold must be between 1 and 20, "
                f"got {self.circuit_breaker_failure_threshold}"
            )
        if not 1 <= self.circuit_breaker_success_threshold <= 10:
            raise ValueError(
                f"circuit_breaker_success_threshold must be between 1 and 10, "
                f"got {self.circuit_breaker_success_threshold}"
            )
        if not 10.0 <= self.circuit_breaker_timeout <= 600.0:
            raise ValueError(
                f"circuit_breaker_timeout must be between 10.0 and 600.0 seconds, "
                f"got {self.circuit_breaker_timeout}"
            )

        # Validate retry strategy settings
        if not 0.0 <= self.retry_temp_increase <= 1.0:
            raise ValueError(
                f"retry_temp_increase must be between 0.0 and 1.0, got {self.retry_temp_increase}"
            )
        if not 2 <= self.retry_simplify_on_attempt <= 10:
            raise ValueError(
                f"retry_simplify_on_attempt must be between 2 and 10, "
                f"got {self.retry_simplify_on_attempt}"
            )

        # Validate semantic duplicate settings
        if not 0.5 <= self.semantic_duplicate_threshold <= 1.0:
            raise ValueError(
                f"semantic_duplicate_threshold must be between 0.5 and 1.0, "
                f"got {self.semantic_duplicate_threshold}"
            )
        if self.semantic_duplicate_enabled and not self.embedding_model.strip():
            raise ValueError("embedding_model must be set when semantic_duplicate_enabled is True")

        # Validate temperature decay semantics (start should be >= end for decay)
        if self.world_quality_refinement_temp_start < self.world_quality_refinement_temp_end:
            raise ValueError(
                f"world_quality_refinement_temp_start ({self.world_quality_refinement_temp_start}) "
                f"must be >= world_quality_refinement_temp_end ({self.world_quality_refinement_temp_end}) "
                "to preserve decay semantics"
            )

        # Validate judge consistency settings
        if not 2 <= self.judge_multi_call_count <= 5:
            raise ValueError(
                f"judge_multi_call_count must be between 2 and 5, got {self.judge_multi_call_count}"
            )

        if not 0.0 <= self.judge_confidence_threshold <= 1.0:
            raise ValueError(
                f"judge_confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.judge_confidence_threshold}"
            )

        if not 1.0 <= self.judge_outlier_std_threshold <= 4.0:
            raise ValueError(
                f"judge_outlier_std_threshold must be between 1.0 and 4.0, "
                f"got {self.judge_outlier_std_threshold}"
            )

        valid_outlier_strategies = ["median", "mean", "retry"]
        if self.judge_outlier_strategy not in valid_outlier_strategies:
            raise ValueError(
                f"judge_outlier_strategy must be one of {valid_outlier_strategies}, "
                f"got {self.judge_outlier_strategy}"
            )

        # Validate world generation count settings
        world_gen_ranges = [
            ("characters", self.world_gen_characters_min, self.world_gen_characters_max),
            ("locations", self.world_gen_locations_min, self.world_gen_locations_max),
            ("factions", self.world_gen_factions_min, self.world_gen_factions_max),
            ("items", self.world_gen_items_min, self.world_gen_items_max),
            ("concepts", self.world_gen_concepts_min, self.world_gen_concepts_max),
            ("relationships", self.world_gen_relationships_min, self.world_gen_relationships_max),
        ]
        for entity_type, min_val, max_val in world_gen_ranges:
            if not 0 <= min_val <= 20:
                raise ValueError(
                    f"world_gen_{entity_type}_min must be between 0 and 20, got {min_val}"
                )
            if not 1 <= max_val <= 50:
                raise ValueError(
                    f"world_gen_{entity_type}_max must be between 1 and 50, got {max_val}"
                )
            if min_val > max_val:
                raise ValueError(
                    f"world_gen_{entity_type}_min ({min_val}) cannot exceed "
                    f"world_gen_{entity_type}_max ({max_val})"
                )

        # Validate LLM token limits
        token_settings = [
            ("llm_tokens_character_create", self.llm_tokens_character_create),
            ("llm_tokens_character_judge", self.llm_tokens_character_judge),
            ("llm_tokens_character_refine", self.llm_tokens_character_refine),
            ("llm_tokens_location_create", self.llm_tokens_location_create),
            ("llm_tokens_location_judge", self.llm_tokens_location_judge),
            ("llm_tokens_location_refine", self.llm_tokens_location_refine),
            ("llm_tokens_faction_create", self.llm_tokens_faction_create),
            ("llm_tokens_faction_judge", self.llm_tokens_faction_judge),
            ("llm_tokens_faction_refine", self.llm_tokens_faction_refine),
            ("llm_tokens_item_create", self.llm_tokens_item_create),
            ("llm_tokens_item_judge", self.llm_tokens_item_judge),
            ("llm_tokens_item_refine", self.llm_tokens_item_refine),
            ("llm_tokens_concept_create", self.llm_tokens_concept_create),
            ("llm_tokens_concept_judge", self.llm_tokens_concept_judge),
            ("llm_tokens_concept_refine", self.llm_tokens_concept_refine),
            ("llm_tokens_relationship_create", self.llm_tokens_relationship_create),
            ("llm_tokens_relationship_judge", self.llm_tokens_relationship_judge),
            ("llm_tokens_relationship_refine", self.llm_tokens_relationship_refine),
            ("llm_tokens_mini_description", self.llm_tokens_mini_description),
        ]
        for name, value in token_settings:
            if not 10 <= value <= 4096:
                raise ValueError(f"{name} must be between 10 and 4096, got {value}")

        # Validate entity extraction limits
        for name, value in [
            ("entity_extract_locations_max", self.entity_extract_locations_max),
            ("entity_extract_items_max", self.entity_extract_items_max),
            ("entity_extract_events_max", self.entity_extract_events_max),
        ]:
            if not 1 <= value <= 100:
                raise ValueError(f"{name} must be between 1 and 100, got {value}")

        # Validate mini description settings
        if not 5 <= self.mini_description_words_max <= 50:
            raise ValueError(
                f"mini_description_words_max must be between 5 and 50, "
                f"got {self.mini_description_words_max}"
            )
        if not 0.0 <= self.mini_description_temperature <= 2.0:
            raise ValueError(
                f"mini_description_temperature must be between 0.0 and 2.0, "
                f"got {self.mini_description_temperature}"
            )

        # Validate workflow limits
        if not 1 <= self.orchestrator_cache_size <= 100:
            raise ValueError(
                f"orchestrator_cache_size must be between 1 and 100, "
                f"got {self.orchestrator_cache_size}"
            )
        if not 10 <= self.workflow_max_events <= 10000:
            raise ValueError(
                f"workflow_max_events must be between 10 and 10000, got {self.workflow_max_events}"
            )

        # Validate LLM request limits
        if not 1 <= self.llm_max_concurrent_requests <= 10:
            raise ValueError(
                f"llm_max_concurrent_requests must be between 1 and 10, "
                f"got {self.llm_max_concurrent_requests}"
            )
        if not 1 <= self.llm_max_retries <= 10:
            raise ValueError(
                f"llm_max_retries must be between 1 and 10, got {self.llm_max_retries}"
            )

        # Validate content truncation
        if not 500 <= self.content_truncation_for_judgment <= 50000:
            raise ValueError(
                f"content_truncation_for_judgment must be between 500 and 50000, "
                f"got {self.content_truncation_for_judgment}"
            )

        # Validate Ollama client timeouts
        ollama_timeout_settings = [
            ("ollama_health_check_timeout", self.ollama_health_check_timeout, 5.0, 300.0),
            ("ollama_list_models_timeout", self.ollama_list_models_timeout, 5.0, 300.0),
            ("ollama_pull_model_timeout", self.ollama_pull_model_timeout, 60.0, 3600.0),
            ("ollama_delete_model_timeout", self.ollama_delete_model_timeout, 5.0, 300.0),
            ("ollama_check_update_timeout", self.ollama_check_update_timeout, 5.0, 300.0),
            ("ollama_generate_timeout", self.ollama_generate_timeout, 5.0, 600.0),
            ("ollama_capability_check_timeout", self.ollama_capability_check_timeout, 30.0, 600.0),
        ]
        for timeout_name, timeout_value, min_timeout, max_timeout in ollama_timeout_settings:
            if not min_timeout <= timeout_value <= max_timeout:
                raise ValueError(
                    f"{timeout_name} must be between {min_timeout} and {max_timeout} seconds, "
                    f"got {timeout_value}"
                )

        # Validate retry configuration
        if not 0.1 <= self.llm_retry_delay <= 60.0:
            raise ValueError(
                f"llm_retry_delay must be between 0.1 and 60.0 seconds, got {self.llm_retry_delay}"
            )
        if not 1.0 <= self.llm_retry_backoff <= 10.0:
            raise ValueError(
                f"llm_retry_backoff must be between 1.0 and 10.0, got {self.llm_retry_backoff}"
            )

        # Validate verification delays
        if not 0.01 <= self.model_verification_sleep <= 5.0:
            raise ValueError(
                f"model_verification_sleep must be between 0.01 and 5.0 seconds, "
                f"got {self.model_verification_sleep}"
            )

        # Validate validation thresholds
        if not 0 <= self.validator_cjk_char_threshold <= 1000:
            raise ValueError(
                f"validator_cjk_char_threshold must be between 0 and 1000, "
                f"got {self.validator_cjk_char_threshold}"
            )
        if not 0.0 <= self.validator_printable_ratio <= 1.0:
            raise ValueError(
                f"validator_printable_ratio must be between 0.0 and 1.0, "
                f"got {self.validator_printable_ratio}"
            )
        if not 0 <= self.validator_ai_check_min_length <= 10000:
            raise ValueError(
                f"validator_ai_check_min_length must be between 0 and 10000, "
                f"got {self.validator_ai_check_min_length}"
            )

        # Validate outline generation
        if not 1 <= self.outline_variations_min <= 10:
            raise ValueError(
                f"outline_variations_min must be between 1 and 10, "
                f"got {self.outline_variations_min}"
            )
        if not 1 <= self.outline_variations_max <= 20:
            raise ValueError(
                f"outline_variations_max must be between 1 and 20, "
                f"got {self.outline_variations_max}"
            )
        if self.outline_variations_min > self.outline_variations_max:
            raise ValueError(
                f"outline_variations_min ({self.outline_variations_min}) cannot exceed "
                f"outline_variations_max ({self.outline_variations_max})"
            )

        # Validate import thresholds
        if not 0.0 <= self.import_confidence_threshold <= 1.0:
            raise ValueError(
                f"import_confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.import_confidence_threshold}"
            )
        if not 0.0 <= self.import_default_confidence <= 1.0:
            raise ValueError(
                f"import_default_confidence must be between 0.0 and 1.0, "
                f"got {self.import_default_confidence}"
            )

        # Validate world/plot generation limits
        if not 10 <= self.world_description_summary_length <= 10000:
            raise ValueError(
                f"world_description_summary_length must be between 10 and 10000, "
                f"got {self.world_description_summary_length}"
            )
        if not 5 <= self.event_sentence_min_length <= 500:
            raise ValueError(
                f"event_sentence_min_length must be between 5 and 500, "
                f"got {self.event_sentence_min_length}"
            )
        if not 10 <= self.event_sentence_max_length <= 5000:
            raise ValueError(
                f"event_sentence_max_length must be between 10 and 5000, "
                f"got {self.event_sentence_max_length}"
            )
        if self.event_sentence_min_length > self.event_sentence_max_length:
            raise ValueError(
                f"event_sentence_min_length ({self.event_sentence_min_length}) cannot exceed "
                f"event_sentence_max_length ({self.event_sentence_max_length})"
            )

        # Validate rating bounds
        if not 1 <= self.user_rating_min <= 10:
            raise ValueError(
                f"user_rating_min must be between 1 and 10, got {self.user_rating_min}"
            )
        if not 1 <= self.user_rating_max <= 10:
            raise ValueError(
                f"user_rating_max must be between 1 and 10, got {self.user_rating_max}"
            )
        if self.user_rating_min > self.user_rating_max:
            raise ValueError(
                f"user_rating_min ({self.user_rating_min}) cannot exceed "
                f"user_rating_max ({self.user_rating_max})"
            )

        # Validate model download threshold
        if not 0.0 <= self.model_download_threshold <= 1.0:
            raise ValueError(
                f"model_download_threshold must be between 0.0 and 1.0, "
                f"got {self.model_download_threshold}"
            )

        # Validate story chapter counts
        if not 1 <= self.chapters_short_story <= 100:
            raise ValueError(
                f"chapters_short_story must be between 1 and 100, got {self.chapters_short_story}"
            )
        if not 1 <= self.chapters_novella <= 100:
            raise ValueError(
                f"chapters_novella must be between 1 and 100, got {self.chapters_novella}"
            )
        if not 1 <= self.chapters_novel <= 200:
            raise ValueError(f"chapters_novel must be between 1 and 200, got {self.chapters_novel}")
        if not 1 <= self.chapters_default <= 100:
            raise ValueError(
                f"chapters_default must be between 1 and 100, got {self.chapters_default}"
            )

        # Validate import temperatures
        if not 0.0 <= self.temp_import_extraction <= 2.0:
            raise ValueError(
                f"temp_import_extraction must be between 0.0 and 2.0, "
                f"got {self.temp_import_extraction}"
            )
        if not 0.0 <= self.temp_interviewer_override <= 2.0:
            raise ValueError(
                f"temp_interviewer_override must be between 0.0 and 2.0, "
                f"got {self.temp_interviewer_override}"
            )

        # Validate token multipliers
        if not 1 <= self.import_character_token_multiplier <= 20:
            raise ValueError(
                f"import_character_token_multiplier must be between 1 and 20, "
                f"got {self.import_character_token_multiplier}"
            )

        # Validate content check settings
        if not isinstance(self.content_check_enabled, bool):
            raise ValueError(
                f"content_check_enabled must be a boolean, got {type(self.content_check_enabled)}"
            )
        if not isinstance(self.content_check_use_llm, bool):
            raise ValueError(
                f"content_check_use_llm must be a boolean, got {type(self.content_check_use_llm)}"
            )
        if self.content_check_use_llm and not self.content_check_enabled:
            logger.warning(
                "content_check_use_llm is enabled but content_check_enabled is False. "
                "LLM checking will have no effect."
            )

        # Validate world health settings
        if not isinstance(self.relationship_validation_enabled, bool):
            raise ValueError(
                f"relationship_validation_enabled must be a boolean, "
                f"got {type(self.relationship_validation_enabled)}"
            )
        if not isinstance(self.orphan_detection_enabled, bool):
            raise ValueError(
                f"orphan_detection_enabled must be a boolean, "
                f"got {type(self.orphan_detection_enabled)}"
            )
        if not isinstance(self.circular_detection_enabled, bool):
            raise ValueError(
                f"circular_detection_enabled must be a boolean, "
                f"got {type(self.circular_detection_enabled)}"
            )
        if not 0.5 <= self.fuzzy_match_threshold <= 1.0:
            raise ValueError(
                f"fuzzy_match_threshold must be between 0.5 and 1.0, "
                f"got {self.fuzzy_match_threshold}"
            )
        if not 1 <= self.max_relationships_per_entity <= 50:
            raise ValueError(
                f"max_relationships_per_entity must be between 1 and 50, "
                f"got {self.max_relationships_per_entity}"
            )

        # Validate circular_relationship_types
        if not isinstance(self.circular_relationship_types, list):
            raise ValueError(
                f"circular_relationship_types must be a list, "
                f"got {type(self.circular_relationship_types)}"
            )
        if not all(isinstance(t, str) for t in self.circular_relationship_types):
            raise ValueError("circular_relationship_types must contain only strings")

        # Validate relationship_minimums structure
        if not isinstance(self.relationship_minimums, dict):
            raise ValueError(
                f"relationship_minimums must be a dict, got {type(self.relationship_minimums)}"
            )
        for entity_type, roles in self.relationship_minimums.items():
            if not isinstance(entity_type, str):
                raise ValueError(
                    f"relationship_minimums keys must be strings, got {type(entity_type)}"
                )
            if not isinstance(roles, dict):
                raise ValueError(
                    f"relationship_minimums[{entity_type}] must be a dict, got {type(roles)}"
                )
            for role, min_count in roles.items():
                if not isinstance(role, str):
                    raise ValueError(
                        f"relationship_minimums[{entity_type}] keys must be strings, "
                        f"got {type(role)}"
                    )
                if not isinstance(min_count, int) or min_count < 0:
                    raise ValueError(
                        f"relationship_minimums[{entity_type}][{role}] must be a non-negative "
                        f"integer, got {min_count}"
                    )
                # Ensure max_relationships_per_entity >= minimum count
                if min_count > self.max_relationships_per_entity:
                    raise ValueError(
                        f"relationship_minimums[{entity_type}][{role}] ({min_count}) exceeds "
                        f"max_relationships_per_entity ({self.max_relationships_per_entity})"
                    )

    # Class-level cache for settings (speeds up repeated load() calls)
    _cached_instance: ClassVar[Settings | None] = None

    @classmethod
    def load(cls, use_cache: bool = True) -> Settings:
        """Load settings from JSON file, or create defaults.

        Attempts to repair invalid settings by merging valid fields with
        defaults rather than silently overwriting the entire file.

        Args:
            use_cache: If True, return cached instance if available. Set to False
                to force reload from disk (useful after save() or in tests).

        Returns:
            Settings instance.
        """
        # Return cached instance if available and caching enabled
        if use_cache and cls._cached_instance is not None:
            return cls._cached_instance

        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE) as f:
                    data = json.load(f)
                settings = cls(**data)
                # Validate loaded settings
                settings.validate()
                cls._cached_instance = settings
                return settings
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted settings file (invalid JSON): {e}")
                logger.warning("Creating backup and using default settings")
                cls._backup_corrupted_settings()
            except TypeError as e:
                logger.warning(f"Unknown fields in settings file: {e}")
                # Try partial recovery - use only known fields
                return cls._recover_partial_settings(data)
            except ValueError as e:
                logger.warning(f"Invalid setting values: {e}")
                # Try partial recovery - use valid fields
                return cls._recover_partial_settings(data)

        # Create default settings
        settings = cls()
        settings.save()
        cls._cached_instance = settings
        return settings

    @classmethod
    def _backup_corrupted_settings(cls) -> None:
        """Create a backup of corrupted settings file."""
        if SETTINGS_FILE.exists():
            backup_path = SETTINGS_FILE.with_suffix(".json.bak")
            try:
                import shutil

                shutil.copy(SETTINGS_FILE, backup_path)
                logger.info(f"Backed up corrupted settings to {backup_path}")
            except OSError as e:
                logger.warning(f"Failed to backup settings: {e}")

    @classmethod
    def _recover_partial_settings(cls, data: dict[str, Any]) -> Settings:
        """Attempt to recover partial settings from corrupted data.

        Merges valid fields from the loaded data with default values,
        preserving as much user configuration as possible.

        Args:
            data: The raw dict loaded from the settings file.

        Returns:
            Settings object with recovered valid fields.
        """
        defaults = cls()
        default_dict = asdict(defaults)
        recovered_fields: list[str] = []
        invalid_fields: list[str] = []

        # Try each field individually
        for field_name, default_value in default_dict.items():
            if field_name in data:
                try:
                    # Create a test settings with just this field changed
                    test_data = default_dict.copy()
                    test_data[field_name] = data[field_name]
                    # Validate by attempting to construct with this field
                    cls(**test_data)

                    # If construction succeeded, the field is valid
                    setattr(defaults, field_name, data[field_name])
                    recovered_fields.append(field_name)
                except (TypeError, ValueError):  # pragma: no cover
                    # Defensive: dataclasses don't type-check, but keep for safety
                    invalid_fields.append(field_name)
                    setattr(defaults, field_name, default_value)

        if recovered_fields:
            logger.info(f"Recovered {len(recovered_fields)} valid settings: {recovered_fields}")
        if invalid_fields:  # pragma: no cover
            logger.warning(
                f"Reset {len(invalid_fields)} invalid settings to defaults: {invalid_fields}"
            )

        # Validate the recovered settings
        try:
            defaults.validate()
        except ValueError as e:
            logger.error(f"Recovery failed validation: {e}")
            logger.warning("Falling back to complete defaults")
            defaults = cls()

        # Save the recovered/repaired settings
        defaults.save()
        cls._cached_instance = defaults
        return defaults

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached settings instance.

        Use this in tests that need to verify settings loading behavior,
        or after programmatically modifying settings files.
        """
        cls._cached_instance = None

    def get_model_tags(self, model_id: str) -> list[str]:
        """Get all tags for a model, merging RECOMMENDED_MODELS and custom tags.

        Args:
            model_id: The model identifier (e.g., "my-model:7b")

        Returns:
            List of role tags for this model (e.g., ["writer", "editor"])
        """
        tags: set[str] = set()

        # Check RECOMMENDED_MODELS (exact match and base name match)
        for rec_id, info in RECOMMENDED_MODELS.items():
            if model_id == rec_id or model_id.startswith(rec_id.split(":")[0]):
                tags.update(info.get("tags", []))
                break

        # Add custom tags
        if model_id in self.custom_model_tags:
            tags.update(self.custom_model_tags[model_id])

        return list(tags)

    def set_model_tags(self, model_id: str, tags: list[str]) -> None:
        """Set custom tags for a model.

        Args:
            model_id: The model identifier
            tags: List of role tags to assign
        """
        if tags:
            self.custom_model_tags[model_id] = tags
        elif model_id in self.custom_model_tags:
            del self.custom_model_tags[model_id]
        self.save()
        logger.info(f"Updated tags for {model_id}: {tags}")

    def get_model_for_agent(self, agent_role: str, available_vram: int = 24) -> str:
        """Get the appropriate model for an agent role using tag-based selection.

        ONLY models tagged for the specified role will be selected.
        No fallback to untagged models - users must configure tags.

        Args:
            agent_role: The agent role (writer, architect, etc.)
            available_vram: Available VRAM in GB

        Returns:
            Model ID to use for this agent role.

        Raises:
            ValueError: If no tagged model is available for the role.
        """

        if not self.use_per_agent_models:
            # If per-agent disabled, use default model
            if self.default_model != "auto":
                return self.default_model
            # Fall through to tag-based selection

        model_setting: str = self.agent_models.get(agent_role, "auto")

        if model_setting != "auto":
            return model_setting

        # Get installed models with their sizes
        installed_models = get_installed_models_with_sizes()

        if not installed_models:
            # Return a recommended model as default for CI/testing
            # Ollama will error later when actually trying to use it
            default = next(iter(RECOMMENDED_MODELS.keys()))
            logger.warning(
                f"No models installed in Ollama - returning default model '{default}' for {agent_role}. "
                f"Install a model with 'ollama pull <model_name>'."
            )
            return default

        # Find installed models tagged for this role
        tagged_models_fit: list[tuple[str, float, float]] = []  # (model_id, size, quality)
        tagged_models_all: list[tuple[str, float, float]] = []  # All tagged models
        for model_id, size_gb in installed_models.items():
            tags = self.get_model_tags(model_id)
            if agent_role in tags:
                estimated_vram = int(size_gb * 1.2)
                # Get quality from RECOMMENDED_MODELS if available, else default to 5
                quality = 5.0
                for rec_id, info in RECOMMENDED_MODELS.items():
                    if model_id == rec_id or model_id.startswith(rec_id.split(":")[0]):
                        quality = info.get("quality", 5.0)
                        break
                tagged_models_all.append((model_id, size_gb, quality))
                if estimated_vram <= available_vram:
                    tagged_models_fit.append((model_id, size_gb, quality))

        if tagged_models_fit:
            # Sort by quality descending, then size descending
            tagged_models_fit.sort(key=lambda x: (x[2], x[1]), reverse=True)
            best = tagged_models_fit[0]
            logger.info(
                f"Auto-selected {best[0]} ({best[1]:.1f}GB, quality={best[2]}) "
                f"for {agent_role} (tagged model)"
            )
            return best[0]

        if tagged_models_all:
            # No tagged model fits VRAM - select smallest as last resort
            tagged_models_all.sort(key=lambda x: x[1])  # Sort by size ascending
            smallest = tagged_models_all[0]
            logger.warning(
                f"No tagged model fits VRAM ({available_vram}GB) for {agent_role}. "
                f"Selecting smallest tagged model: {smallest[0]} ({smallest[1]:.1f}GB)"
            )
            return smallest[0]

        # No tagged model available - raise error with helpful message
        installed_list = ", ".join(installed_models.keys())
        raise ValueError(
            f"No model tagged for role '{agent_role}'. "
            f"Installed models: [{installed_list}]. "
            f"Configure model tags in Settings > Models tab, or download a recommended model."
        )

    def get_temperature_for_agent(self, agent_role: str) -> float:
        """Get temperature setting for an agent.

        Raises:
            ValueError: If agent_role is not configured in agent_temperatures.
        """
        if agent_role not in self.agent_temperatures:
            raise ValueError(
                f"Unknown agent role '{agent_role}' - must be one of: "
                f"{sorted(self.agent_temperatures.keys())}"
            )
        return float(self.agent_temperatures[agent_role])


def get_installed_models(timeout: int | None = None) -> list[str]:
    """Get list of models currently installed in Ollama.

    Args:
        timeout: Timeout in seconds. If None, uses default (10s).
    """
    actual_timeout = timeout if timeout is not None else 10
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except FileNotFoundError:
        logger.warning("Ollama not found. Please ensure Ollama is installed and in PATH.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"Ollama list command timed out after {actual_timeout}s.")
        return []
    except (OSError, ValueError) as e:
        logger.warning(f"Error listing Ollama models: {e}")
        return []


def get_installed_models_with_sizes(timeout: int | None = None) -> dict[str, float]:
    """Get installed models with their sizes in GB.

    Args:
        timeout: Timeout in seconds. If None, uses default (10s).

    Returns:
        Dict mapping model ID to size in GB.
    """
    actual_timeout = timeout if timeout is not None else 10
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        # Check return code before parsing
        if result.returncode != 0:
            logger.warning(f"Ollama list returned non-zero exit code: {result.returncode}")
            return {}

        models = {}
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    model_name = parts[0]
                    # Size is typically in format like "4.1 GB" or "890 MB"
                    # Find size column - it's the one with GB/MB
                    # Use decimal units: 1 GB = 1000 MB (not 1024)
                    size_gb = 0.0
                    for i, part in enumerate(parts):
                        if part.upper() == "GB" and i > 0:
                            try:
                                size_gb = float(parts[i - 1])
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse GB size '{parts[i - 1]}' for model '{model_name}'"
                                )
                            break
                        elif part.upper() == "MB" and i > 0:
                            try:
                                size_gb = float(parts[i - 1]) / 1000
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse MB size '{parts[i - 1]}' for model '{model_name}'"
                                )
                            break
                        # Also handle combined format like "4.1GB"
                        elif part.upper().endswith("GB"):
                            try:
                                size_gb = float(part[:-2])
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse combined GB size '{part}' for model '{model_name}'"
                                )
                            break
                        elif part.upper().endswith("MB"):
                            try:
                                size_gb = float(part[:-2]) / 1000
                            except ValueError:
                                logger.debug(
                                    f"Failed to parse combined MB size '{part}' for model '{model_name}'"
                                )
                            break

                    models[model_name] = size_gb
        return models
    except FileNotFoundError:
        logger.warning("Ollama not found. Please ensure Ollama is installed and in PATH.")
        return {}
    except subprocess.TimeoutExpired:
        logger.warning(f"Ollama list command timed out after {actual_timeout}s.")
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Error listing Ollama models: {e}")
        return {}


def get_available_vram(timeout: int | None = None) -> int:
    """Detect available VRAM in GB. Returns 8GB default if detection fails.

    Args:
        timeout: Timeout in seconds. If None, uses default (10s).
    """
    actual_timeout = timeout if timeout is not None else 10
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        vram_mb = int(result.stdout.strip().split("\n")[0])
        return vram_mb // 1024
    except FileNotFoundError:
        logger.info("nvidia-smi not found. Using default VRAM assumption of 8GB.")
        return 8
    except subprocess.TimeoutExpired:
        logger.warning(f"nvidia-smi timed out after {actual_timeout}s. Using default 8GB.")
        return 8
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse VRAM from nvidia-smi output: {e}. Using default 8GB.")
        return 8
    except OSError as e:
        logger.info(f"Could not detect VRAM: {e}. Using default 8GB.")
        return 8


def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a model.

    If the model is in RECOMMENDED_MODELS, returns that info.
    Otherwise, estimates info based on model name/size.
    """
    # Check exact match first
    if model_id in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[model_id]

    # Check base name match (e.g., "qwen3:4b" matches "qwen3:4b")
    base_name = model_id.split(":")[0] if ":" in model_id else model_id
    for rec_id, info in RECOMMENDED_MODELS.items():
        if rec_id.startswith(base_name) or base_name in rec_id:
            return info

    # Try to get size from installed models
    installed = get_installed_models_with_sizes()
    size_gb = installed.get(model_id, 0)

    # Estimate quality/speed from size
    if size_gb > 0:
        # Larger models = higher quality, lower speed
        quality = min(10, int(size_gb / 4) + 4)  # 4GB->5, 20GB->9
        speed = max(1, 10 - int(size_gb / 5))  # Smaller = faster
        vram_required = int(size_gb * 1.2)
    else:
        quality = 5
        speed = 5
        vram_required = 8  # Default assumption
        size_gb = 5.0

    return ModelInfo(
        name=model_id,
        size_gb=size_gb,
        vram_required=vram_required,
        quality=quality,
        speed=speed,
        uncensored=True,
        description="Automatically detected model",
        tags=[],  # No specific tags for unknown models
    )
