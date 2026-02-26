"""Validation functions for Settings."""

import logging
from typing import TYPE_CHECKING

from src.memory.mode_models import VramStrategy
from src.settings._types import LOG_LEVELS, REFINEMENT_TEMP_DECAY_CURVES

if TYPE_CHECKING:
    from src.settings._settings import Settings

logger = logging.getLogger(__name__)


def validate(settings: Settings) -> bool:
    """Validate all settings fields.

    Delegates to individual validation functions for each category of settings.

    Returns:
        True if any settings were mutated during validation (e.g. stale values
        migrated), False otherwise. Callers can use this to decide whether to
        re-save the settings file.

    Raises:
        ValueError: If any field contains an invalid value.
    """
    changed = False
    _validate_log_level(settings)
    _validate_url(settings)
    _validate_numeric_ranges(settings)
    _validate_interaction_mode(settings)
    _validate_vram_strategy(settings)
    _validate_temperatures(settings)
    _validate_dict_structure(settings)
    _validate_task_temperatures(settings)
    _validate_learning_settings(settings)
    _validate_data_integrity(settings)
    _validate_timeouts(settings)
    _validate_world_quality(settings)
    _validate_dynamic_temperature(settings)
    _validate_early_stopping(settings)
    _validate_circuit_breaker(settings)
    _validate_retry_strategy(settings)
    changed = _validate_rag_context(settings) or changed
    changed = _validate_semantic_duplicate(settings) or changed
    _validate_temperature_decay_semantics(settings)
    _validate_judge_consistency(settings)
    _validate_world_gen_counts(settings)
    _validate_entity_extraction(settings)
    _validate_mini_description(settings)
    _validate_workflow_limits(settings)
    _validate_llm_request_limits(settings)
    _validate_content_truncation(settings)
    _validate_ollama_client_timeouts(settings)
    _validate_retry_configuration(settings)
    _validate_verification_delays(settings)
    _validate_model_cache_ttls(settings)
    _validate_validation_thresholds(settings)
    _validate_outline_generation(settings)
    _validate_import_thresholds(settings)
    _validate_world_plot_limits(settings)
    _validate_rating_bounds(settings)
    _validate_model_download(settings)
    _validate_story_chapters(settings)
    _validate_import_temperatures(settings)
    _validate_token_multipliers(settings)
    _validate_content_check(settings)
    _validate_world_health(settings)
    return changed


def _validate_log_level(settings: Settings) -> None:
    """Validate log_level is a known logging level."""
    if settings.log_level not in LOG_LEVELS:
        raise ValueError(
            f"log_level must be one of {list(LOG_LEVELS.keys())}, got {settings.log_level}"
        )


def _validate_url(settings: Settings) -> None:
    """Validate URL format for ollama_url."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(settings.ollama_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme in ollama_url: {settings.ollama_url}")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL (missing host) in ollama_url: {settings.ollama_url}")
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Invalid ollama_url: {settings.ollama_url} - {e}") from e


def _validate_numeric_ranges(settings: Settings) -> None:
    """Validate numeric range constraints."""
    if not 1024 <= settings.context_size <= 128000:
        raise ValueError(
            f"context_size must be between 1024 and 128000, got {settings.context_size}"
        )

    if not 256 <= settings.max_tokens <= 32000:
        raise ValueError(f"max_tokens must be between 256 and 32000, got {settings.max_tokens}")

    if not 1 <= settings.chapters_between_checkpoints <= 20:
        raise ValueError(
            f"chapters_between_checkpoints must be between 1 and 20, "
            f"got {settings.chapters_between_checkpoints}"
        )

    if not 0 <= settings.max_revision_iterations <= 10:
        raise ValueError(
            f"max_revision_iterations must be between 0 and 10, "
            f"got {settings.max_revision_iterations}"
        )


def _validate_interaction_mode(settings: Settings) -> None:
    """Validate interaction mode is a known value."""
    valid_modes = ["minimal", "checkpoint", "interactive", "collaborative"]
    if settings.interaction_mode not in valid_modes:
        raise ValueError(
            f"interaction_mode must be one of {valid_modes}, got {settings.interaction_mode}"
        )


def _validate_vram_strategy(settings: Settings) -> None:
    """Validate VRAM strategy (derived from enum to prevent drift)."""
    valid_vram_strategies = [strategy.value for strategy in VramStrategy]
    if settings.vram_strategy not in valid_vram_strategies:
        raise ValueError(
            f"vram_strategy must be one of {valid_vram_strategies}, got {settings.vram_strategy}"
        )


def _validate_temperatures(settings: Settings) -> None:
    """Validate agent temperature values are within range.

    Structural checks (missing/extra keys) are handled by the merge step in
    Settings.load(), so this only validates value ranges.
    """
    if not isinstance(settings.agent_temperatures, dict):
        raise ValueError(
            f"agent_temperatures must be a dict, got {type(settings.agent_temperatures).__name__}"
        )
    for agent, temp in settings.agent_temperatures.items():
        if not isinstance(temp, (int, float)):
            raise ValueError(f"Temperature for {agent} must be a number, got {type(temp).__name__}")
        if not 0.0 <= temp <= 2.0:
            raise ValueError(f"Temperature for {agent} must be between 0.0 and 2.0, got {temp}")


def _validate_dict_structure(settings: Settings) -> None:
    """Validate that fixed-key dict fields have exactly the expected keys.

    The merge step in Settings.load() handles adding/removing keys gracefully,
    but validate() is also called from save() and the UI.  This catch-all
    rejects unknown or missing keys so programmatic mutations that bypass
    load() can't silently persist invalid structure.

    Raises:
        ValueError: If any dict field has unknown or missing keys.
    """
    logger.debug("Validating structured and nested dict settings")
    from src.settings._settings import (
        _NESTED_DICT_FIELDS,
        _STRUCTURED_DICT_FIELDS,
    )
    from src.settings._settings import (
        Settings as _Settings,
    )

    default_instance = _Settings()

    for field_name in _STRUCTURED_DICT_FIELDS:
        current = getattr(settings, field_name)
        if not isinstance(current, dict):
            raise ValueError(f"{field_name} must be a dict, got {type(current).__name__}")
        expected = set(getattr(default_instance, field_name))
        actual = set(current)
        unknown = actual - expected
        if unknown:
            raise ValueError(
                f"Unknown keys in {field_name}: {sorted(unknown)}; "
                f"expected only: {sorted(expected)}"
            )
        missing = expected - actual
        if missing:
            raise ValueError(
                f"Missing keys in {field_name}: {sorted(missing)}; expected: {sorted(expected)}"
            )

    for field_name in _NESTED_DICT_FIELDS:
        default_outer = getattr(default_instance, field_name)
        current_outer = getattr(settings, field_name)
        if not isinstance(current_outer, dict):
            raise ValueError(f"{field_name} must be a dict, got {type(current_outer).__name__}")
        unknown_outer = set(current_outer) - set(default_outer)
        if unknown_outer:
            raise ValueError(
                f"Unknown keys in {field_name}: {sorted(unknown_outer)}; "
                f"expected only: {sorted(default_outer)}"
            )
        missing_outer = set(default_outer) - set(current_outer)
        if missing_outer:
            raise ValueError(
                f"Missing keys in {field_name}: {sorted(missing_outer)}; "
                f"expected: {sorted(default_outer)}"
            )
        for outer_key, default_inner in default_outer.items():
            if isinstance(default_inner, dict) and outer_key in current_outer:
                inner = current_outer[outer_key]
                if not isinstance(inner, dict):
                    raise ValueError(
                        f"{field_name}[{outer_key}] must be a dict, got {type(inner).__name__}"
                    )
                unknown_inner = set(inner) - set(default_inner)
                if unknown_inner:
                    raise ValueError(
                        f"Unknown keys in {field_name}[{outer_key}]: "
                        f"{sorted(unknown_inner)}; expected only: {sorted(default_inner)}"
                    )
                missing_inner = set(default_inner) - set(inner)
                if missing_inner:
                    raise ValueError(
                        f"Missing keys in {field_name}[{outer_key}]: "
                        f"{sorted(missing_inner)}; expected: {sorted(default_inner)}"
                    )


def _validate_task_temperatures(settings: Settings) -> None:
    """Validate task-specific temperature settings."""
    task_temps = [
        ("temp_brief_extraction", settings.temp_brief_extraction),
        ("temp_edit_suggestions", settings.temp_edit_suggestions),
        ("temp_plot_checking", settings.temp_plot_checking),
        ("temp_capability_check", settings.temp_capability_check),
        ("temp_model_evaluation", settings.temp_model_evaluation),
    ]
    for name, temp in task_temps:
        if not 0.0 <= temp <= 2.0:
            raise ValueError(f"{name} must be between 0.0 and 2.0, got {temp}")


def _validate_learning_settings(settings: Settings) -> None:
    """Validate learning/tuning settings."""
    valid_autonomy = ["manual", "cautious", "balanced", "aggressive", "experimental"]
    if settings.learning_autonomy not in valid_autonomy:
        raise ValueError(
            f"learning_autonomy must be one of {valid_autonomy}, got {settings.learning_autonomy}"
        )

    valid_triggers = ["off", "after_project", "periodic", "continuous"]
    for trigger in settings.learning_triggers:
        if trigger not in valid_triggers:
            raise ValueError(
                f"Invalid learning trigger: {trigger}; must be one of {valid_triggers}"
            )

    if not 1 <= settings.learning_periodic_interval <= 20:
        raise ValueError(
            f"learning_periodic_interval must be between 1 and 20, "
            f"got {settings.learning_periodic_interval}"
        )

    if not 0.0 <= settings.learning_confidence_threshold <= 1.0:
        raise ValueError(
            f"learning_confidence_threshold must be between 0.0 and 1.0, "
            f"got {settings.learning_confidence_threshold}"
        )


def _validate_data_integrity(settings: Settings) -> None:
    """Validate data integrity settings."""
    if not 1 <= settings.entity_version_retention <= 100:
        raise ValueError(
            f"entity_version_retention must be between 1 and 100, "
            f"got {settings.entity_version_retention}"
        )
    if not isinstance(settings.backup_verify_on_restore, bool):
        raise ValueError(
            f"backup_verify_on_restore must be a boolean, "
            f"got {type(settings.backup_verify_on_restore).__name__}"
        )


def _validate_timeouts(settings: Settings) -> None:
    """Validate timeout settings."""
    if not 10 <= settings.ollama_timeout <= 600:
        raise ValueError(
            f"ollama_timeout must be between 10 and 600 seconds, got {settings.ollama_timeout}"
        )

    if not 5 <= settings.subprocess_timeout <= 60:
        raise ValueError(
            f"subprocess_timeout must be between 5 and 60 seconds, "
            f"got {settings.subprocess_timeout}"
        )

    if settings.small_model_size_threshold_gb <= 0:
        raise ValueError(
            f"small_model_size_threshold_gb must be positive, "
            f"got {settings.small_model_size_threshold_gb}"
        )

    if settings.small_model_timeout_cap <= 0:
        raise ValueError(
            f"small_model_timeout_cap must be positive, got {settings.small_model_timeout_cap}"
        )


def _validate_world_quality(settings: Settings) -> None:
    """Validate world quality settings."""
    if not 1 <= settings.world_quality_max_iterations <= 10:
        raise ValueError(
            f"world_quality_max_iterations must be between 1 and 10, "
            f"got {settings.world_quality_max_iterations}"
        )

    if not 0.0 <= settings.world_quality_threshold <= 10.0:
        raise ValueError(
            f"world_quality_threshold must be between 0.0 and 10.0, "
            f"got {settings.world_quality_threshold}"
        )

    # Validate per-entity quality thresholds
    _validate_world_quality_thresholds(settings)

    if not 0.0 <= settings.world_quality_dimension_minimum <= 10.0:
        raise ValueError(
            f"world_quality_dimension_minimum must be between 0.0 and 10.0, "
            f"got {settings.world_quality_dimension_minimum}"
        )

    if settings.world_quality_dimension_minimum > settings.world_quality_threshold:
        raise ValueError(
            f"world_quality_dimension_minimum ({settings.world_quality_dimension_minimum}) "
            f"cannot exceed world_quality_threshold ({settings.world_quality_threshold})"
        )

    if settings.world_quality_thresholds:
        min_per_entity = min(settings.world_quality_thresholds.values())
        if settings.world_quality_dimension_minimum > min_per_entity:
            lowest_entity = min(
                settings.world_quality_thresholds,
                key=lambda k: settings.world_quality_thresholds[k],
            )
            raise ValueError(
                f"world_quality_dimension_minimum ({settings.world_quality_dimension_minimum}) "
                f"cannot exceed the lowest per-entity threshold "
                f"({min_per_entity} for '{lowest_entity}')"
            )

    if not 1 <= settings.world_quality_early_stopping_patience <= 10:
        raise ValueError(
            f"world_quality_early_stopping_patience must be between 1 and 10, "
            f"got {settings.world_quality_early_stopping_patience}"
        )

    for temp_name, temp_value in [
        ("world_quality_creator_temp", settings.world_quality_creator_temp),
        ("world_quality_judge_temp", settings.world_quality_judge_temp),
        ("world_quality_refinement_temp", settings.world_quality_refinement_temp),
        ("world_quality_refinement_temp_start", settings.world_quality_refinement_temp_start),
        ("world_quality_refinement_temp_end", settings.world_quality_refinement_temp_end),
    ]:
        if not 0.0 <= temp_value <= 2.0:
            raise ValueError(f"{temp_name} must be between 0.0 and 2.0, got {temp_value}")


def _validate_dynamic_temperature(settings: Settings) -> None:
    """Validate dynamic temperature decay curve setting."""
    if settings.world_quality_refinement_temp_decay not in REFINEMENT_TEMP_DECAY_CURVES:
        raise ValueError(
            f"world_quality_refinement_temp_decay must be one of "
            f"{list(REFINEMENT_TEMP_DECAY_CURVES.keys())}, "
            f"got {settings.world_quality_refinement_temp_decay}"
        )


def _validate_early_stopping(settings: Settings) -> None:
    """Validate enhanced early stopping settings."""
    if not 1 <= settings.world_quality_early_stopping_min_iterations <= 10:
        raise ValueError(
            f"world_quality_early_stopping_min_iterations must be between 1 and 10, "
            f"got {settings.world_quality_early_stopping_min_iterations}"
        )
    if not 0.0 <= settings.world_quality_early_stopping_variance_tolerance <= 2.0:
        raise ValueError(
            f"world_quality_early_stopping_variance_tolerance must be between 0.0 and 2.0, "
            f"got {settings.world_quality_early_stopping_variance_tolerance}"
        )
    if not 0.0 <= settings.world_quality_score_plateau_tolerance <= 1.0:
        raise ValueError(
            f"world_quality_score_plateau_tolerance must be between 0.0 and 1.0, "
            f"got {settings.world_quality_score_plateau_tolerance}"
        )


def _validate_circuit_breaker(settings: Settings) -> None:
    """Validate circuit breaker settings."""
    if not 1 <= settings.circuit_breaker_failure_threshold <= 20:
        raise ValueError(
            f"circuit_breaker_failure_threshold must be between 1 and 20, "
            f"got {settings.circuit_breaker_failure_threshold}"
        )
    if not 1 <= settings.circuit_breaker_success_threshold <= 10:
        raise ValueError(
            f"circuit_breaker_success_threshold must be between 1 and 10, "
            f"got {settings.circuit_breaker_success_threshold}"
        )
    if not 10.0 <= settings.circuit_breaker_timeout <= 600.0:
        raise ValueError(
            f"circuit_breaker_timeout must be between 10.0 and 600.0 seconds, "
            f"got {settings.circuit_breaker_timeout}"
        )


def _validate_retry_strategy(settings: Settings) -> None:
    """Validate retry strategy settings."""
    if not 0.0 <= settings.retry_temp_increase <= 1.0:
        raise ValueError(
            f"retry_temp_increase must be between 0.0 and 1.0, got {settings.retry_temp_increase}"
        )
    if not 2 <= settings.retry_simplify_on_attempt <= 10:
        raise ValueError(
            f"retry_simplify_on_attempt must be between 2 and 10, "
            f"got {settings.retry_simplify_on_attempt}"
        )


def _validate_semantic_duplicate(settings: Settings) -> bool:
    """Validate semantic duplicate detection settings.

    Auto-disables semantic_duplicate_enabled when embedding_model is empty,
    since semantic duplicate detection cannot function without an embedding model.

    Returns:
        True if settings were mutated (auto-disabled), False otherwise.
    """
    if not 0.5 <= settings.semantic_duplicate_threshold <= 1.0:
        raise ValueError(
            f"semantic_duplicate_threshold must be between 0.5 and 1.0, "
            f"got {settings.semantic_duplicate_threshold}"
        )
    if settings.semantic_duplicate_enabled and not settings.embedding_model.strip():
        logger.warning(
            "semantic_duplicate_enabled requires an embedding_model; "
            "auto-disabling semantic duplicate detection"
        )
        settings.semantic_duplicate_enabled = False
        return True
    return False


def _validate_temperature_decay_semantics(settings: Settings) -> None:
    """Validate temperature decay semantics (start should be >= end for decay)."""
    if settings.world_quality_refinement_temp_start < settings.world_quality_refinement_temp_end:
        raise ValueError(
            f"world_quality_refinement_temp_start ({settings.world_quality_refinement_temp_start}) "
            f"must be >= world_quality_refinement_temp_end "
            f"({settings.world_quality_refinement_temp_end}) "
            "to preserve decay semantics"
        )


def _validate_judge_consistency(settings: Settings) -> None:
    """Validate judge consistency settings."""
    if not 2 <= settings.judge_multi_call_count <= 5:
        raise ValueError(
            f"judge_multi_call_count must be between 2 and 5, got {settings.judge_multi_call_count}"
        )

    if not 0.0 <= settings.judge_confidence_threshold <= 1.0:
        raise ValueError(
            f"judge_confidence_threshold must be between 0.0 and 1.0, "
            f"got {settings.judge_confidence_threshold}"
        )

    if not 1.0 <= settings.judge_outlier_std_threshold <= 4.0:
        raise ValueError(
            f"judge_outlier_std_threshold must be between 1.0 and 4.0, "
            f"got {settings.judge_outlier_std_threshold}"
        )

    valid_outlier_strategies = ["median", "mean"]
    if settings.judge_outlier_strategy not in valid_outlier_strategies:
        raise ValueError(
            f"judge_outlier_strategy must be one of {valid_outlier_strategies}, "
            f"got {settings.judge_outlier_strategy}"
        )


def _validate_world_gen_counts(settings: Settings) -> None:
    """Validate world generation count settings."""
    world_gen_ranges = [
        ("characters", settings.world_gen_characters_min, settings.world_gen_characters_max),
        ("locations", settings.world_gen_locations_min, settings.world_gen_locations_max),
        ("factions", settings.world_gen_factions_min, settings.world_gen_factions_max),
        ("items", settings.world_gen_items_min, settings.world_gen_items_max),
        ("concepts", settings.world_gen_concepts_min, settings.world_gen_concepts_max),
        ("events", settings.world_gen_events_min, settings.world_gen_events_max),
        (
            "relationships",
            settings.world_gen_relationships_min,
            settings.world_gen_relationships_max,
        ),
    ]
    for entity_type, min_val, max_val in world_gen_ranges:
        if not 0 <= min_val <= 20:
            raise ValueError(f"world_gen_{entity_type}_min must be between 0 and 20, got {min_val}")
        if not 1 <= max_val <= 50:
            raise ValueError(f"world_gen_{entity_type}_max must be between 1 and 50, got {max_val}")
        if min_val > max_val:
            raise ValueError(
                f"world_gen_{entity_type}_min ({min_val}) cannot exceed "
                f"world_gen_{entity_type}_max ({max_val})"
            )


def _validate_entity_extraction(settings: Settings) -> None:
    """Validate entity extraction limit settings."""
    for name, value in [
        ("entity_extract_locations_max", settings.entity_extract_locations_max),
        ("entity_extract_items_max", settings.entity_extract_items_max),
        ("entity_extract_events_max", settings.entity_extract_events_max),
    ]:
        if not 1 <= value <= 100:
            raise ValueError(f"{name} must be between 1 and 100, got {value}")


def _validate_mini_description(settings: Settings) -> None:
    """Validate mini description settings."""
    if not 5 <= settings.mini_description_words_max <= 50:
        raise ValueError(
            f"mini_description_words_max must be between 5 and 50, "
            f"got {settings.mini_description_words_max}"
        )
    if not 0.0 <= settings.mini_description_temperature <= 2.0:
        raise ValueError(
            f"mini_description_temperature must be between 0.0 and 2.0, "
            f"got {settings.mini_description_temperature}"
        )


def _validate_workflow_limits(settings: Settings) -> None:
    """Validate workflow and orchestration limits."""
    if not 1 <= settings.orchestrator_cache_size <= 100:
        raise ValueError(
            f"orchestrator_cache_size must be between 1 and 100, "
            f"got {settings.orchestrator_cache_size}"
        )
    if not 10 <= settings.workflow_max_events <= 10000:
        raise ValueError(
            f"workflow_max_events must be between 10 and 10000, got {settings.workflow_max_events}"
        )


def _validate_llm_request_limits(settings: Settings) -> None:
    """Validate LLM request limit settings."""
    if not 1 <= settings.llm_max_concurrent_requests <= 10:
        raise ValueError(
            f"llm_max_concurrent_requests must be between 1 and 10, "
            f"got {settings.llm_max_concurrent_requests}"
        )
    if not 1 <= settings.llm_max_retries <= 10:
        raise ValueError(
            f"llm_max_retries must be between 1 and 10, got {settings.llm_max_retries}"
        )
    if not 30 <= settings.llm_semaphore_timeout <= 600:
        raise ValueError(
            f"llm_semaphore_timeout must be between 30 and 600, "
            f"got {settings.llm_semaphore_timeout}"
        )


def _validate_content_truncation(settings: Settings) -> None:
    """Validate content truncation settings."""
    if not 500 <= settings.content_truncation_for_judgment <= 50000:
        raise ValueError(
            f"content_truncation_for_judgment must be between 500 and 50000, "
            f"got {settings.content_truncation_for_judgment}"
        )

    context_char_fields = [
        ("previous_chapter_context_chars", settings.previous_chapter_context_chars),
        ("chapter_analysis_chars", settings.chapter_analysis_chars),
        ("full_story_analysis_chars", settings.full_story_analysis_chars),
        ("full_text_preview_chars", settings.full_text_preview_chars),
    ]
    logger.debug("Validating context char fields: %s", [n for n, _ in context_char_fields])
    for field_name, value in context_char_fields:
        if not 1 <= value <= 100000:
            raise ValueError(f"{field_name} must be between 1 and 100000, got {value}")


def _validate_ollama_client_timeouts(settings: Settings) -> None:
    """Validate Ollama client timeout settings."""
    ollama_timeout_settings = [
        ("ollama_health_check_timeout", settings.ollama_health_check_timeout, 5.0, 300.0),
        ("ollama_list_models_timeout", settings.ollama_list_models_timeout, 5.0, 300.0),
        ("ollama_pull_model_timeout", settings.ollama_pull_model_timeout, 60.0, 3600.0),
        ("ollama_delete_model_timeout", settings.ollama_delete_model_timeout, 5.0, 300.0),
        ("ollama_check_update_timeout", settings.ollama_check_update_timeout, 5.0, 300.0),
        ("ollama_generate_timeout", settings.ollama_generate_timeout, 5.0, 600.0),
        (
            "ollama_capability_check_timeout",
            settings.ollama_capability_check_timeout,
            30.0,
            600.0,
        ),
    ]
    for timeout_name, timeout_value, min_timeout, max_timeout in ollama_timeout_settings:
        if not min_timeout <= timeout_value <= max_timeout:
            raise ValueError(
                f"{timeout_name} must be between {min_timeout} and {max_timeout} seconds, "
                f"got {timeout_value}"
            )


def _validate_retry_configuration(settings: Settings) -> None:
    """Validate retry configuration settings."""
    if not 0.1 <= settings.llm_retry_delay <= 60.0:
        raise ValueError(
            f"llm_retry_delay must be between 0.1 and 60.0 seconds, got {settings.llm_retry_delay}"
        )
    if not 1.0 <= settings.llm_retry_backoff <= 10.0:
        raise ValueError(
            f"llm_retry_backoff must be between 1.0 and 10.0, got {settings.llm_retry_backoff}"
        )


def _validate_verification_delays(settings: Settings) -> None:
    """Validate verification delay settings."""
    if not 0.01 <= settings.model_verification_sleep <= 5.0:
        raise ValueError(
            f"model_verification_sleep must be between 0.01 and 5.0 seconds, "
            f"got {settings.model_verification_sleep}"
        )


def _validate_model_cache_ttls(settings: Settings) -> None:
    """Validate model service cache TTL settings."""
    for name, value in [
        ("model_health_cache_ttl", settings.model_health_cache_ttl),
        ("model_installed_cache_ttl", settings.model_installed_cache_ttl),
        ("model_vram_cache_ttl", settings.model_vram_cache_ttl),
    ]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(value).__name__}")
        if not 1.0 <= float(value) <= 300.0:
            raise ValueError(f"{name} must be between 1.0 and 300.0 seconds, got {value}")


def _validate_validation_thresholds(settings: Settings) -> None:
    """Validate validation threshold settings."""
    if not 0 <= settings.validator_cjk_char_threshold <= 1000:
        raise ValueError(
            f"validator_cjk_char_threshold must be between 0 and 1000, "
            f"got {settings.validator_cjk_char_threshold}"
        )
    if not 0.0 <= settings.validator_printable_ratio <= 1.0:
        raise ValueError(
            f"validator_printable_ratio must be between 0.0 and 1.0, "
            f"got {settings.validator_printable_ratio}"
        )


def _validate_outline_generation(settings: Settings) -> None:
    """Validate outline generation settings."""
    if not 1 <= settings.outline_variations_min <= 10:
        raise ValueError(
            f"outline_variations_min must be between 1 and 10, "
            f"got {settings.outline_variations_min}"
        )
    if not 1 <= settings.outline_variations_max <= 20:
        raise ValueError(
            f"outline_variations_max must be between 1 and 20, "
            f"got {settings.outline_variations_max}"
        )
    if settings.outline_variations_min > settings.outline_variations_max:
        raise ValueError(
            f"outline_variations_min ({settings.outline_variations_min}) cannot exceed "
            f"outline_variations_max ({settings.outline_variations_max})"
        )


def _validate_import_thresholds(settings: Settings) -> None:
    """Validate import threshold settings."""
    if not 0.0 <= settings.import_confidence_threshold <= 1.0:
        raise ValueError(
            f"import_confidence_threshold must be between 0.0 and 1.0, "
            f"got {settings.import_confidence_threshold}"
        )
    if not 0.0 <= settings.import_default_confidence <= 1.0:
        raise ValueError(
            f"import_default_confidence must be between 0.0 and 1.0, "
            f"got {settings.import_default_confidence}"
        )


def _validate_world_plot_limits(settings: Settings) -> None:
    """Validate world/plot generation limit settings."""
    if not 10 <= settings.world_description_summary_length <= 10000:
        raise ValueError(
            f"world_description_summary_length must be between 10 and 10000, "
            f"got {settings.world_description_summary_length}"
        )
    if not 5 <= settings.event_sentence_min_length <= 500:
        raise ValueError(
            f"event_sentence_min_length must be between 5 and 500, "
            f"got {settings.event_sentence_min_length}"
        )
    if not 10 <= settings.event_sentence_max_length <= 5000:
        raise ValueError(
            f"event_sentence_max_length must be between 10 and 5000, "
            f"got {settings.event_sentence_max_length}"
        )
    if settings.event_sentence_min_length > settings.event_sentence_max_length:
        raise ValueError(
            f"event_sentence_min_length ({settings.event_sentence_min_length}) cannot exceed "
            f"event_sentence_max_length ({settings.event_sentence_max_length})"
        )


def _validate_rating_bounds(settings: Settings) -> None:
    """Validate user rating bound settings."""
    if not 1 <= settings.user_rating_min <= 10:
        raise ValueError(
            f"user_rating_min must be between 1 and 10, got {settings.user_rating_min}"
        )
    if not 1 <= settings.user_rating_max <= 10:
        raise ValueError(
            f"user_rating_max must be between 1 and 10, got {settings.user_rating_max}"
        )
    if settings.user_rating_min > settings.user_rating_max:
        raise ValueError(
            f"user_rating_min ({settings.user_rating_min}) cannot exceed "
            f"user_rating_max ({settings.user_rating_max})"
        )


def _validate_model_download(settings: Settings) -> None:
    """Validate model download threshold."""
    if not 0.0 <= settings.model_download_threshold <= 1.0:
        raise ValueError(
            f"model_download_threshold must be between 0.0 and 1.0, "
            f"got {settings.model_download_threshold}"
        )


def _validate_story_chapters(settings: Settings) -> None:
    """Validate story chapter count settings."""
    if not 1 <= settings.chapters_short_story <= 100:
        raise ValueError(
            f"chapters_short_story must be between 1 and 100, got {settings.chapters_short_story}"
        )
    if not 1 <= settings.chapters_novella <= 100:
        raise ValueError(
            f"chapters_novella must be between 1 and 100, got {settings.chapters_novella}"
        )
    if not 1 <= settings.chapters_novel <= 200:
        raise ValueError(f"chapters_novel must be between 1 and 200, got {settings.chapters_novel}")
    if not 1 <= settings.chapters_default <= 100:
        raise ValueError(
            f"chapters_default must be between 1 and 100, got {settings.chapters_default}"
        )


def _validate_import_temperatures(settings: Settings) -> None:
    """Validate import temperature settings."""
    if not 0.0 <= settings.temp_import_extraction <= 2.0:
        raise ValueError(
            f"temp_import_extraction must be between 0.0 and 2.0, "
            f"got {settings.temp_import_extraction}"
        )
    if not 0.0 <= settings.temp_interviewer_override <= 2.0:
        raise ValueError(
            f"temp_interviewer_override must be between 0.0 and 2.0, "
            f"got {settings.temp_interviewer_override}"
        )


def _validate_token_multipliers(settings: Settings) -> None:
    """Validate token multiplier settings."""
    if not 1 <= settings.import_character_token_multiplier <= 20:
        raise ValueError(
            f"import_character_token_multiplier must be between 1 and 20, "
            f"got {settings.import_character_token_multiplier}"
        )


def _validate_content_check(settings: Settings) -> None:
    """Validate content check settings."""
    if not isinstance(settings.content_check_enabled, bool):
        raise ValueError(
            f"content_check_enabled must be a boolean, got {type(settings.content_check_enabled)}"
        )
    if not isinstance(settings.content_check_use_llm, bool):
        raise ValueError(
            f"content_check_use_llm must be a boolean, got {type(settings.content_check_use_llm)}"
        )
    if settings.content_check_use_llm and not settings.content_check_enabled:
        logger.warning(
            "content_check_use_llm is enabled but content_check_enabled is False. "
            "LLM checking will have no effect."
        )


def _validate_world_health(settings: Settings) -> None:
    """Validate world health and relationship settings."""
    if not isinstance(settings.relationship_validation_enabled, bool):
        raise ValueError(
            f"relationship_validation_enabled must be a boolean, "
            f"got {type(settings.relationship_validation_enabled)}"
        )
    if not isinstance(settings.orphan_detection_enabled, bool):
        raise ValueError(
            f"orphan_detection_enabled must be a boolean, "
            f"got {type(settings.orphan_detection_enabled)}"
        )
    if not isinstance(settings.circular_detection_enabled, bool):
        raise ValueError(
            f"circular_detection_enabled must be a boolean, "
            f"got {type(settings.circular_detection_enabled)}"
        )
    if not 0.5 <= settings.fuzzy_match_threshold <= 1.0:
        raise ValueError(
            f"fuzzy_match_threshold must be between 0.5 and 1.0, "
            f"got {settings.fuzzy_match_threshold}"
        )
    if not 1 <= settings.max_relationships_per_entity <= 50:
        raise ValueError(
            f"max_relationships_per_entity must be between 1 and 50, "
            f"got {settings.max_relationships_per_entity}"
        )

    # Validate circular_check_all_types
    if not isinstance(settings.circular_check_all_types, bool):
        raise ValueError(
            f"circular_check_all_types must be a boolean, "
            f"got {type(settings.circular_check_all_types)}"
        )

    # Validate circular_relationship_types
    if not isinstance(settings.circular_relationship_types, list):
        raise ValueError(
            f"circular_relationship_types must be a list, "
            f"got {type(settings.circular_relationship_types)}"
        )
    if not all(isinstance(t, str) for t in settings.circular_relationship_types):
        raise ValueError("circular_relationship_types must contain only strings")

    # _validate_dict_structure already rejects non-dict relationship_minimums,
    # so we only validate inner values here.
    for entity_type, roles in settings.relationship_minimums.items():
        # _validate_dict_structure ensures outer and inner key sets are correct,
        # but does NOT validate inner value types — those checks remain below.
        for role, min_count in roles.items():
            if not isinstance(min_count, int) or min_count < 0:
                raise ValueError(
                    f"relationship_minimums[{entity_type}][{role}] must be a non-negative "
                    f"integer, got {min_count}"
                )
            # Ensure max_relationships_per_entity >= minimum count
            if min_count > settings.max_relationships_per_entity:
                raise ValueError(
                    f"relationship_minimums[{entity_type}][{role}] ({min_count}) exceeds "
                    f"max_relationships_per_entity ({settings.max_relationships_per_entity})"
                )

    # Validate calendar settings
    if not isinstance(settings.generate_calendar_on_world_build, bool):
        raise ValueError(
            f"generate_calendar_on_world_build must be a boolean, "
            f"got {type(settings.generate_calendar_on_world_build)}"
        )
    if not isinstance(settings.validate_temporal_consistency, bool):
        raise ValueError(
            f"validate_temporal_consistency must be a boolean, "
            f"got {type(settings.validate_temporal_consistency)}"
        )


def _validate_rag_context(settings: Settings) -> bool:
    """Validate RAG context retrieval settings.

    Auto-migrates empty ``embedding_model`` (a stale default from pre-RAG
    settings files) to the current default rather than raising an error.  This
    prevents existing users from hitting an opaque ``ConfigError`` on upgrade.

    Returns:
        True if ``embedding_model`` was migrated, False otherwise.

    Raises:
        ValueError: If any RAG setting is out of range.
    """
    if not 100 <= settings.rag_context_max_tokens <= 16000:
        raise ValueError(
            f"rag_context_max_tokens must be between 100 and 16000, "
            f"got {settings.rag_context_max_tokens}"
        )
    if not 1 <= settings.rag_context_max_items <= 100:
        raise ValueError(
            f"rag_context_max_items must be between 1 and 100, got {settings.rag_context_max_items}"
        )
    if not 0.0 <= settings.rag_context_similarity_threshold <= 1.0:
        raise ValueError(
            f"rag_context_similarity_threshold must be between 0.0 and 1.0, "
            f"got {settings.rag_context_similarity_threshold}"
        )
    if not 1 <= settings.rag_context_graph_depth <= 3:
        raise ValueError(
            f"rag_context_graph_depth must be between 1 and 3, "
            f"got {settings.rag_context_graph_depth}"
        )

    if not settings.embedding_model.strip():
        from src.settings._settings import Settings as _Settings

        default_model = _Settings().embedding_model
        logger.warning(
            "embedding_model is empty (stale settings file from before RAG pipeline). "
            "Auto-migrating to default '%s'. "
            "Change this in Settings > RAG if you prefer a different model.",
            default_model,
        )
        settings.embedding_model = default_model
        return True
    return False


def _validate_world_quality_thresholds(settings: Settings) -> None:
    """Validate per-entity quality threshold values are within range.

    Structural checks (missing/extra keys) are handled by the merge step in
    Settings.load(), so this only validates value ranges.

    Raises:
        ValueError: If any threshold is out of range.
    """
    for entity_type, threshold in settings.world_quality_thresholds.items():
        if not 0.0 <= threshold <= 10.0:
            raise ValueError(
                f"world_quality_thresholds[{entity_type}] must be between 0.0 and 10.0, "
                f"got {threshold}"
            )
