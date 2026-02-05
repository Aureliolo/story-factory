"""Main Settings dataclass for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from src.memory.mode_models import LearningTrigger
from src.settings import _validation as _validation_mod
from src.settings._model_registry import RECOMMENDED_MODELS
from src.settings._paths import BACKUPS_DIR, SETTINGS_FILE
from src.settings._types import check_minimum_quality
from src.settings._utils import get_installed_models_with_sizes

# Configure module logger
logger = logging.getLogger(__name__)


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
    log_level: str = "INFO"

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
            "judge": "auto",  # Capable model for quality judging (needs reasoning)
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
            "judge": 0.1,  # Very low for consistent quality judgments
            "embedding": 0.0,  # Not used for generation — embeddings are deterministic
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
    world_quality_threshold: float = 8.0  # Min score (0-10) to pass quality review
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
    embedding_model: str = ""  # Model for generating embeddings (must be set when semantic duplicate detection is enabled)

    # Content guidelines checking settings
    content_check_enabled: bool = True  # Enable content guideline checking for generated content
    content_check_use_llm: bool = False  # Use LLM for more accurate checking (slower)

    # Judge consistency settings (for more reliable quality judgments)
    judge_consistency_enabled: bool = True  # Enable judge consistency features
    judge_multi_call_enabled: bool = True  # Make multiple judge calls and average
    judge_multi_call_count: int = 3  # Number of judge calls if multi_call_enabled
    judge_confidence_threshold: float = 0.7  # Min confidence for reliable decisions
    judge_outlier_detection: bool = True  # Detect and handle outlier scores
    judge_outlier_std_threshold: float = 2.0  # Std devs from mean to consider outlier
    judge_outlier_strategy: str = "median"  # How to handle outliers: median or mean

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

    # Calendar and temporal validation settings
    generate_calendar_on_world_build: bool = True  # Auto-generate calendar during world build
    validate_temporal_consistency: bool = True  # Validate temporal consistency of entities
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
    llm_tokens_relationship_create: int = 800
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

    def validate(self) -> bool:
        """Validate all settings fields. Delegates to _validation module.

        Returns:
            True if any settings were mutated during validation (e.g. stale
            embedding model migrated), False otherwise.

        Raises:
            ValueError: If any field contains an invalid value (range, enum/choice, or format).
        """
        return _validation_mod.validate(self)

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
                # Validate loaded settings (may auto-migrate stale values)
                changed = settings.validate()
                if changed:
                    logger.info("Settings migrated during validation, saving to disk")
                    with open(SETTINGS_FILE, "w") as f:
                        json.dump(asdict(settings), f, indent=2)
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
        """
        Selects the model ID to use for a given agent role based on configured tags and available VRAM.

        Respects per-agent model settings and the global default model; when a role's setting is "auto" the method selects from installed models that are tagged for the role, preferring models that fit within the provided available_vram. Embedding-tagged models are excluded for non-"embedding" roles. If no models are installed, a recommended default model ID is returned.

        Parameters:
            agent_role (str): Agent role to select a model for (for example, "writer" or "embedding").
            available_vram (int): Available VRAM in GB used to prefer models that fit the system.

        Returns:
            str: Model ID to use for the specified agent role.

        Raises:
            ValueError: If no installed model is tagged for the requested role.
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
                f"No models installed in Ollama - returning default model '{default}' "
                f"for {agent_role}. "
                f"Install a model with 'ollama pull <model_name>'."
            )
            return default

        # Find installed models tagged for this role
        tagged_models_fit: list[tuple[str, float, float]] = []  # (model_id, size, quality)
        tagged_models_all: list[tuple[str, float, float]] = []  # All tagged models
        for model_id, size_gb in installed_models.items():
            tags = self.get_model_tags(model_id)
            if agent_role in tags:
                # Embedding models only generate vectors — skip them for chat roles
                if agent_role != "embedding" and "embedding" in tags:
                    logger.debug(
                        "Skipping embedding model %s for chat role %s", model_id, agent_role
                    )
                    continue
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
            logger.debug(
                f"Auto-selected {best[0]} ({best[1]:.1f}GB, quality={best[2]}) "
                f"for {agent_role} (tagged model)"
            )
            check_minimum_quality(best[0], best[2], agent_role)
            return best[0]

        if tagged_models_all:
            # No tagged model fits VRAM - select smallest as last resort
            tagged_models_all.sort(key=lambda x: x[1])  # Sort by size ascending
            smallest = tagged_models_all[0]
            logger.warning(
                f"No tagged model fits VRAM ({available_vram}GB) for {agent_role}. "
                f"Selecting smallest tagged model: {smallest[0]} ({smallest[1]:.1f}GB)"
            )
            check_minimum_quality(smallest[0], smallest[2], agent_role)
            return smallest[0]

        # No tagged model available - raise error with helpful message
        installed_list = ", ".join(installed_models.keys())
        raise ValueError(
            f"No model tagged for role '{agent_role}'. "
            f"Installed models: [{installed_list}]. "
            f"Configure model tags in Settings > Models tab, or download a recommended model."
        )

    def get_models_for_role(self, role: str) -> list[str]:
        """Return all installed models tagged for the given role, sorted by quality descending.

        Args:
            role: The agent role tag to search for (e.g., "judge", "writer").

        Returns:
            List of model IDs that have the requested role tag, sorted by
            quality score descending. Returns empty list if none found.
        """
        installed_models = get_installed_models_with_sizes()
        if not installed_models:
            logger.debug("get_models_for_role: no installed models found")
            return []

        tagged: list[tuple[str, float]] = []  # (model_id, quality)
        for model_id in installed_models:
            tags = self.get_model_tags(model_id)
            if role in tags:
                # Embedding models only generate vectors — skip for chat roles
                if role != "embedding" and "embedding" in tags:
                    continue
                quality = 5.0
                for rec_id, info in RECOMMENDED_MODELS.items():
                    if model_id == rec_id or model_id.startswith(rec_id.split(":")[0]):
                        quality = info.get("quality", 5.0)
                        break
                tagged.append((model_id, quality))

        # Sort by quality descending
        tagged.sort(key=lambda x: x[1], reverse=True)
        result = [model_id for model_id, _ in tagged]
        logger.debug("get_models_for_role(%s): found %d models: %s", role, len(result), result)
        return result

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
