"""Web-configurable settings for Story Factory.

Settings are stored in settings.json and can be modified via the web UI.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from memory.mode_models import LearningTrigger

# Configure module logger
logger = logging.getLogger(__name__)


SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Centralized paths for story and world output files
STORIES_DIR = Path(__file__).parent / "output" / "stories"
WORLDS_DIR = Path(__file__).parent / "output" / "worlds"
BACKUPS_DIR = Path(__file__).parent / "output" / "backups"


class ModelInfo(TypedDict):
    """Type definition for model information."""

    name: str
    release: str
    size_gb: int
    vram_required: int
    quality: int | float
    speed: int
    uncensored: bool
    description: str


class AgentRoleInfo(TypedDict):
    """Type definition for agent role information."""

    name: str
    description: str
    recommended_quality: int


# Available models registry - curated for creative writing (uncensored)
# Organized by use case: creative specialists, general purpose, high-end
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    # === CREATIVE WRITING SPECIALISTS ===
    # These models are specifically fine-tuned for fiction and prose
    "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0": {
        "name": "Celeste V1.9 12B",
        "release": "2025",
        "size_gb": 13,
        "vram_required": 14,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "Purpose-built for fiction writing, excellent prose quality",
    },
    "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit": {
        "name": "Dark Champion 18B MOE",
        "release": "2025",
        "size_gb": 11,
        "vram_required": 14,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "Exceptional fiction/RP, outstanding prose quality",
    },
    # === GENERAL PURPOSE (FAST) ===
    # Good for interviewer, editor, continuity - fast and reliable
    "huihui_ai/dolphin3-abliterated:8b": {
        "name": "Dolphin 3.0 8B Abliterated",
        "release": "2025",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7.5,
        "speed": 9,
        "uncensored": True,
        "description": "Eric Hartford's latest, highly compliant, no Chinese output",
    },
    "CognitiveComputations/dolphin-mistral-nemo:12b": {
        "name": "Dolphin Mistral Nemo 12B",
        "release": "2025",
        "size_gb": 7,
        "vram_required": 10,
        "quality": 8,
        "speed": 8,
        "uncensored": True,
        "description": "128K context, excellent for editing and refinement",
    },
    # === REASONING / ARCHITECT SPECIALISTS ===
    # MoE models offer excellent reasoning at reduced VRAM
    "huihui_ai/qwen3-abliterated:30b": {
        "name": "Qwen3 30B Abliterated (MoE)",
        "release": "January 2025",
        "size_gb": 18,
        "vram_required": 18,
        "quality": 9,
        "speed": 7,
        "uncensored": True,
        "description": "MoE (30B/3B active), matches 70B reasoning at half VRAM - RECOMMENDED for architect",
    },
    # === HIGH-END (LARGE VRAM / QUANTIZED) ===
    # Premium models for those with VRAM to spare
    "vanilj/midnight-miqu-70b-v1.5": {
        "name": "Midnight Miqu 70B V1.5",
        "release": "2024",
        "size_gb": 42,
        "vram_required": 48,
        "quality": 9.5,
        "speed": 4,
        "uncensored": True,
        "description": "Premium creative writer - writes like a novelist, 32K context",
    },
    "huihui_ai/llama3.3-abliterated:70b": {
        "name": "Llama 3.3 70B Abliterated",
        "release": "December 2024",
        "size_gb": 40,
        "vram_required": 48,
        "quality": 9.5,
        "speed": 5,
        "uncensored": True,
        "description": "Best reasoning, excellent for story architecture",
    },
    "huihui_ai/llama3.3-abliterated:70b-instruct-q4_K_M": {
        "name": "Llama 3.3 70B Q4_K_M",
        "release": "December 2024",
        "size_gb": 43,
        "vram_required": 24,
        "quality": 9,
        "speed": 4,
        "uncensored": True,
        "description": "Quantized 70B, fits 24GB VRAM, great for architect role",
    },
    # === SMALL / VALIDATOR MODELS ===
    # Minimal models for basic validation tasks
    "qwen3:0.6b": {
        "name": "Qwen3 0.6B",
        "release": "2025",
        "size_gb": 1,
        "vram_required": 2,
        "quality": 3,
        "speed": 10,
        "uncensored": False,
        "description": "Tiny model for validator role - basic sanity checks only",
    },
    # === LEGACY (kept for compatibility) ===
    "huihui_ai/qwen3-abliterated:8b": {
        "name": "Qwen3 8B Abliterated (v1)",
        "release": "April 2025",
        "size_gb": 5,
        "vram_required": 8,
        "quality": 7,
        "speed": 9,
        "uncensored": True,
        "description": "Fast but may output Chinese characters - use Dolphin instead",
    },
}

# Agent role definitions
AGENT_ROLES: dict[str, AgentRoleInfo] = {
    "interviewer": {
        "name": "Interviewer",
        "description": "Gathers story requirements",
        "recommended_quality": 7,  # Doesn't need highest quality
    },
    "architect": {
        "name": "Architect",
        "description": "Designs story structure",
        "recommended_quality": 8,  # Needs good reasoning
    },
    "writer": {
        "name": "Writer",
        "description": "Writes prose",
        "recommended_quality": 9,  # Needs best quality
    },
    "editor": {
        "name": "Editor",
        "description": "Polishes prose",
        "recommended_quality": 8,
    },
    "continuity": {
        "name": "Continuity Checker",
        "description": "Checks for plot holes",
        "recommended_quality": 7,
    },
    "validator": {
        "name": "Validator",
        "description": "Validates AI responses",
        "recommended_quality": 3,  # Uses small/fast model for basic sanity checks
    },
    "suggestion": {
        "name": "Suggestion Assistant",
        "description": "Generates writing prompts and suggestions",
        "recommended_quality": 7,  # Needs creativity but not highest quality
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

    # Default model for all agents (if not using per-agent)
    default_model: str = "huihui_ai/dolphin3-abliterated:8b"

    # Per-agent model settings
    use_per_agent_models: bool = True
    agent_models: dict = field(
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

    # Agent temperatures
    agent_temperatures: dict = field(
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
    comparison_models: list = field(default_factory=list)

    # UI settings
    dark_mode: bool = True
    last_project_id: str | None = None  # Remember last opened project

    # Backup settings
    backup_folder: str = str(BACKUPS_DIR)  # Folder to store project backups

    # Generation mode settings
    current_mode: str = "balanced"  # ID of active generation mode
    use_mode_system: bool = True  # Whether to use mode-based model selection

    # Learning/tuning settings
    learning_triggers: list = field(
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
    world_quality_threshold: float = 7.0  # Minimum quality score (0-10) to accept entity
    world_quality_creator_temp: float = 0.9  # Temperature for creative generation
    world_quality_judge_temp: float = 0.1  # Temperature for quality judgment
    world_quality_refinement_temp: float = 0.7  # Temperature for refinement passes

    # World generation counts (randomized within range for variety)
    world_gen_characters_min: int = 4
    world_gen_characters_max: int = 7
    world_gen_locations_min: int = 3
    world_gen_locations_max: int = 6
    world_gen_factions_min: int = 1
    world_gen_factions_max: int = 3
    world_gen_items_min: int = 2
    world_gen_items_max: int = 5
    world_gen_concepts_min: int = 1
    world_gen_concepts_max: int = 3
    world_gen_relationships_min: int = 8
    world_gen_relationships_max: int = 15

    # LLM generation token limits (num_predict values)
    llm_tokens_character_create: int = 500
    llm_tokens_character_judge: int = 300
    llm_tokens_character_refine: int = 500
    llm_tokens_location_create: int = 400
    llm_tokens_location_judge: int = 200
    llm_tokens_location_refine: int = 400
    llm_tokens_faction_create: int = 400
    llm_tokens_faction_judge: int = 200
    llm_tokens_faction_refine: int = 400
    llm_tokens_item_create: int = 400
    llm_tokens_item_judge: int = 200
    llm_tokens_item_refine: int = 400
    llm_tokens_concept_create: int = 400
    llm_tokens_concept_judge: int = 200
    llm_tokens_concept_refine: int = 400
    llm_tokens_relationship_create: int = 300
    llm_tokens_relationship_judge: int = 200
    llm_tokens_relationship_refine: int = 300
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

    def save(self) -> None:
        """Save settings to JSON file."""
        # Validate before saving
        self.validate()
        with open(SETTINGS_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self) -> None:
        """Validate settings values.

        Raises:
            ValueError: If any setting value is invalid.
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

        for temp_name, temp_value in [
            ("world_quality_creator_temp", self.world_quality_creator_temp),
            ("world_quality_judge_temp", self.world_quality_judge_temp),
            ("world_quality_refinement_temp", self.world_quality_refinement_temp),
        ]:
            if not 0.0 <= temp_value <= 2.0:
                raise ValueError(f"{temp_name} must be between 0.0 and 2.0, got {temp_value}")

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

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from JSON file, or create defaults.

        Attempts to repair invalid settings by merging valid fields with
        defaults rather than silently overwriting the entire file.
        """
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE) as f:
                    data = json.load(f)
                settings = cls(**data)
                # Validate loaded settings
                settings.validate()
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
    def _recover_partial_settings(cls, data: dict) -> "Settings":
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
        return defaults

    def get_model_for_agent(self, agent_role: str, available_vram: int = 24) -> str:
        """Get the appropriate model for an agent role.

        If set to 'auto', selects based on agent's quality requirements, available VRAM,
        and which models are actually installed. Only returns installed models.
        Special handling for writer role to prefer creative writing specialists.
        """
        if not self.use_per_agent_models:
            return self.default_model

        model_setting: str = self.agent_models.get(agent_role, "auto")

        if model_setting != "auto":
            return model_setting

        # Get installed models to filter candidates
        installed = get_installed_models()

        def is_installed(model_id: str) -> bool:
            """Check if model is installed (exact match or base name match)."""
            return any(model_id in m or m.startswith(model_id.split(":")[0]) for m in installed)

        # Special case: Writer role prefers creative writing specialists
        if agent_role == "writer":
            creative_models = [
                "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
                "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit",
            ]
            for model_id in creative_models:
                info = AVAILABLE_MODELS.get(model_id)
                if info and info["vram_required"] <= available_vram and is_installed(model_id):
                    return model_id

        # Special case: Architect role prefers high-reasoning models
        if agent_role == "architect":
            architect_models = [
                "huihui_ai/qwen3-abliterated:30b",  # MoE, matches 70B reasoning at 18GB - RECOMMENDED
                "huihui_ai/llama3.3-abliterated:70b-instruct-q4_K_M",  # Fits 24GB
                "huihui_ai/llama3.3-abliterated:70b",  # Needs 48GB
            ]
            for model_id in architect_models:
                info = AVAILABLE_MODELS.get(model_id)
                if info and info["vram_required"] <= available_vram and is_installed(model_id):
                    return model_id

        # Special case: Validator role prefers smallest/fastest models
        if agent_role == "validator":
            validator_models = [
                "qwen3:0.6b",  # Tiny, fast - ideal for simple validation
            ]
            for model_id in validator_models:
                info = AVAILABLE_MODELS.get(model_id)
                if info and info["vram_required"] <= available_vram and is_installed(model_id):
                    return model_id

        # Auto-select based on agent role and VRAM
        role_info: AgentRoleInfo | None = AGENT_ROLES.get(agent_role)
        required_quality: int = role_info["recommended_quality"] if role_info else 7

        # Filter models that fit VRAM, meet quality requirement, AND are installed
        candidates = []
        for model_id, info in AVAILABLE_MODELS.items():
            if (
                info["vram_required"] <= available_vram
                and info["quality"] >= required_quality
                and is_installed(model_id)
            ):
                candidates.append((model_id, info))

        if not candidates:
            # No suitable installed model found - fall back to default
            return self.default_model

        # For high quality roles (9+), prioritize quality
        # For lower quality roles, prioritize speed while meeting quality threshold
        if required_quality >= 9:
            # Writer needs best quality - sort by quality desc
            candidates.sort(key=lambda x: x[1]["quality"], reverse=True)
        else:
            # Other roles - prefer speed while meeting quality threshold
            # Find candidates closest to required quality with best speed
            candidates.sort(key=lambda x: (-x[1]["speed"], x[1]["quality"]))

        return candidates[0][0]

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
            ["ollama", "list"], capture_output=True, text=True, timeout=actual_timeout
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
    """Get information about a model."""
    return AVAILABLE_MODELS.get(
        model_id,
        ModelInfo(
            name=model_id,
            release="Unknown",
            size_gb=0,
            vram_required=0,
            quality=5,
            speed=5,
            uncensored=True,
            description="Unknown model",
        ),
    )
