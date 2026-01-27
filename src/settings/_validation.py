"""Validation mixin for Settings class."""

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from src.memory.mode_models import VramStrategy
from src.settings._paths import logger
from src.settings._types import AGENT_ROLES, REFINEMENT_TEMP_DECAY_CURVES

if TYPE_CHECKING:
    from src.settings._settings import Settings


class ValidationMixin:
    """Mixin providing validation methods for Settings."""

    def validate(self: Settings) -> None:
        """
        Validate the Settings instance fields and enforce allowed ranges and formats.

        Raises:
            ValueError: If any field contains an invalid value (range, enum/choice, or format).
        """
        self._validate_url()
        self._validate_numeric_ranges()
        self._validate_interaction_mode()
        self._validate_vram_strategy()
        self._validate_temperatures()
        self._validate_learning_settings()
        self._validate_data_integrity_settings()
        self._validate_timeout_settings()
        self._validate_world_quality_settings()
        self._validate_circuit_breaker_settings()
        self._validate_retry_strategy_settings()
        self._validate_semantic_duplicate_settings()
        self._validate_judge_consistency_settings()
        self._validate_world_generation_counts()
        self._validate_llm_token_limits()
        self._validate_entity_extraction_limits()
        self._validate_mini_description_settings()
        self._validate_workflow_limits()
        self._validate_llm_request_limits()
        self._validate_content_truncation()
        self._validate_ollama_client_timeouts()
        self._validate_retry_configuration()
        self._validate_verification_delays()
        self._validate_validation_thresholds()
        self._validate_outline_generation()
        self._validate_import_thresholds()
        self._validate_world_plot_generation()
        self._validate_rating_bounds()
        self._validate_model_download_threshold()
        self._validate_story_chapter_counts()
        self._validate_import_temperatures()
        self._validate_token_multipliers()
        self._validate_content_check_settings()
        self._validate_world_health_settings()

    def _validate_url(self: Settings) -> None:
        """Validate URL format for ollama_url."""
        try:
            parsed = urlparse(self.ollama_url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme in ollama_url: {self.ollama_url}")
            if not parsed.netloc:
                raise ValueError(f"Invalid URL (missing host) in ollama_url: {self.ollama_url}")
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Invalid ollama_url: {self.ollama_url} - {e}") from e

    def _validate_numeric_ranges(self: Settings) -> None:
        """Validate numeric range settings."""
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

    def _validate_interaction_mode(self: Settings) -> None:
        """Validate interaction mode setting."""
        valid_modes = ["minimal", "checkpoint", "interactive", "collaborative"]
        if self.interaction_mode not in valid_modes:
            raise ValueError(
                f"interaction_mode must be one of {valid_modes}, got {self.interaction_mode}"
            )

    def _validate_vram_strategy(self: Settings) -> None:
        """Validate VRAM strategy setting."""
        valid_vram_strategies = [strategy.value for strategy in VramStrategy]
        if self.vram_strategy not in valid_vram_strategies:
            raise ValueError(
                f"vram_strategy must be one of {valid_vram_strategies}, got {self.vram_strategy}"
            )

    def _validate_temperatures(self: Settings) -> None:
        """Validate temperature settings."""
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

    def _validate_learning_settings(self: Settings) -> None:
        """Validate learning/tuning settings."""
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

    def _validate_data_integrity_settings(self: Settings) -> None:
        """Validate data integrity settings."""
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

    def _validate_timeout_settings(self: Settings) -> None:
        """Validate timeout settings."""
        if not 10 <= self.ollama_timeout <= 600:
            raise ValueError(
                f"ollama_timeout must be between 10 and 600 seconds, got {self.ollama_timeout}"
            )

        if not 5 <= self.subprocess_timeout <= 60:
            raise ValueError(
                f"subprocess_timeout must be between 5 and 60 seconds, got {self.subprocess_timeout}"
            )

    def _validate_world_quality_settings(self: Settings) -> None:
        """Validate world quality refinement settings."""
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

        # Validate temperature decay semantics (start should be >= end for decay)
        if self.world_quality_refinement_temp_start < self.world_quality_refinement_temp_end:
            raise ValueError(
                f"world_quality_refinement_temp_start ({self.world_quality_refinement_temp_start}) "
                f"must be >= world_quality_refinement_temp_end ({self.world_quality_refinement_temp_end}) "
                "to preserve decay semantics"
            )

    def _validate_circuit_breaker_settings(self: Settings) -> None:
        """Validate circuit breaker settings."""
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

    def _validate_retry_strategy_settings(self: Settings) -> None:
        """Validate retry strategy settings."""
        if not 0.0 <= self.retry_temp_increase <= 1.0:
            raise ValueError(
                f"retry_temp_increase must be between 0.0 and 1.0, got {self.retry_temp_increase}"
            )
        if not 2 <= self.retry_simplify_on_attempt <= 10:
            raise ValueError(
                f"retry_simplify_on_attempt must be between 2 and 10, "
                f"got {self.retry_simplify_on_attempt}"
            )

    def _validate_semantic_duplicate_settings(self: Settings) -> None:
        """Validate semantic duplicate detection settings."""
        if not 0.5 <= self.semantic_duplicate_threshold <= 1.0:
            raise ValueError(
                f"semantic_duplicate_threshold must be between 0.5 and 1.0, "
                f"got {self.semantic_duplicate_threshold}"
            )
        if self.semantic_duplicate_enabled and not self.embedding_model.strip():
            raise ValueError("embedding_model must be set when semantic_duplicate_enabled is True")

    def _validate_judge_consistency_settings(self: Settings) -> None:
        """Validate judge consistency settings."""
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

    def _validate_world_generation_counts(self: Settings) -> None:
        """Validate world generation count settings."""
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

    def _validate_llm_token_limits(self: Settings) -> None:
        """Validate LLM token limit settings."""
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

    def _validate_entity_extraction_limits(self: Settings) -> None:
        """Validate entity extraction limit settings."""
        for name, value in [
            ("entity_extract_locations_max", self.entity_extract_locations_max),
            ("entity_extract_items_max", self.entity_extract_items_max),
            ("entity_extract_events_max", self.entity_extract_events_max),
        ]:
            if not 1 <= value <= 100:
                raise ValueError(f"{name} must be between 1 and 100, got {value}")

    def _validate_mini_description_settings(self: Settings) -> None:
        """Validate mini description settings."""
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

    def _validate_workflow_limits(self: Settings) -> None:
        """Validate workflow limit settings."""
        if not 1 <= self.orchestrator_cache_size <= 100:
            raise ValueError(
                f"orchestrator_cache_size must be between 1 and 100, "
                f"got {self.orchestrator_cache_size}"
            )
        if not 10 <= self.workflow_max_events <= 10000:
            raise ValueError(
                f"workflow_max_events must be between 10 and 10000, got {self.workflow_max_events}"
            )

    def _validate_llm_request_limits(self: Settings) -> None:
        """Validate LLM request limit settings."""
        if not 1 <= self.llm_max_concurrent_requests <= 10:
            raise ValueError(
                f"llm_max_concurrent_requests must be between 1 and 10, "
                f"got {self.llm_max_concurrent_requests}"
            )
        if not 1 <= self.llm_max_retries <= 10:
            raise ValueError(
                f"llm_max_retries must be between 1 and 10, got {self.llm_max_retries}"
            )

    def _validate_content_truncation(self: Settings) -> None:
        """Validate content truncation settings."""
        if not 500 <= self.content_truncation_for_judgment <= 50000:
            raise ValueError(
                f"content_truncation_for_judgment must be between 500 and 50000, "
                f"got {self.content_truncation_for_judgment}"
            )

    def _validate_ollama_client_timeouts(self: Settings) -> None:
        """Validate Ollama client timeout settings."""
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

    def _validate_retry_configuration(self: Settings) -> None:
        """Validate retry configuration settings."""
        if not 0.1 <= self.llm_retry_delay <= 60.0:
            raise ValueError(
                f"llm_retry_delay must be between 0.1 and 60.0 seconds, got {self.llm_retry_delay}"
            )
        if not 1.0 <= self.llm_retry_backoff <= 10.0:
            raise ValueError(
                f"llm_retry_backoff must be between 1.0 and 10.0, got {self.llm_retry_backoff}"
            )

    def _validate_verification_delays(self: Settings) -> None:
        """Validate verification delay settings."""
        if not 0.01 <= self.model_verification_sleep <= 5.0:
            raise ValueError(
                f"model_verification_sleep must be between 0.01 and 5.0 seconds, "
                f"got {self.model_verification_sleep}"
            )

    def _validate_validation_thresholds(self: Settings) -> None:
        """Validate validation threshold settings."""
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

    def _validate_outline_generation(self: Settings) -> None:
        """Validate outline generation settings."""
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

    def _validate_import_thresholds(self: Settings) -> None:
        """Validate import threshold settings."""
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

    def _validate_world_plot_generation(self: Settings) -> None:
        """Validate world/plot generation limit settings."""
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

    def _validate_rating_bounds(self: Settings) -> None:
        """Validate user rating bound settings."""
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

    def _validate_model_download_threshold(self: Settings) -> None:
        """Validate model download detection threshold."""
        if not 0.0 <= self.model_download_threshold <= 1.0:
            raise ValueError(
                f"model_download_threshold must be between 0.0 and 1.0, "
                f"got {self.model_download_threshold}"
            )

    def _validate_story_chapter_counts(self: Settings) -> None:
        """Validate story chapter count settings."""
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

    def _validate_import_temperatures(self: Settings) -> None:
        """Validate import temperature settings."""
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

    def _validate_token_multipliers(self: Settings) -> None:
        """Validate token multiplier settings."""
        if not 1 <= self.import_character_token_multiplier <= 20:
            raise ValueError(
                f"import_character_token_multiplier must be between 1 and 20, "
                f"got {self.import_character_token_multiplier}"
            )

    def _validate_content_check_settings(self: Settings) -> None:
        """Validate content check settings."""
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

    def _validate_world_health_settings(self: Settings) -> None:
        """Validate world health and validation settings."""
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


__all__ = ["ValidationMixin"]
