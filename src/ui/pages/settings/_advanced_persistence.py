"""Advanced settings - save and refresh logic for persistence/undo-redo.

Extracted from _advanced.py to stay within the 1000-line file size limit.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def save_to_settings(page: SettingsPage) -> None:
    """Extract advanced settings from UI and save to settings.

    Handles world generation, story structure, data integrity, advanced LLM,
    and relationship validation settings.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # World generation settings
    for key, (min_input, max_input) in page._world_gen_inputs.items():  # type: ignore[attr-defined]
        min_val = int(min_input.value)
        max_val = int(max_input.value)
        # Ensure min <= max
        if min_val > max_val:
            max_val = min_val
        setattr(settings, f"world_gen_{key}_min", min_val)
        setattr(settings, f"world_gen_{key}_max", max_val)

    # Quality refinement settings — per-entity thresholds
    if hasattr(page, "_quality_threshold_inputs"):
        for entity_type, input_widget in page._quality_threshold_inputs.items():
            settings.world_quality_thresholds[entity_type] = float(input_widget.value)
        # Keep legacy single threshold in sync — older code paths and migration
        # use world_quality_threshold as the seed value for new per-entity keys.
        # max() ensures it stays representative of the strictest configured threshold.
        settings.world_quality_threshold = max(settings.world_quality_thresholds.values())
    if hasattr(page, "_quality_max_iterations_input"):
        settings.world_quality_max_iterations = int(page._quality_max_iterations_input.value)
    if hasattr(page, "_quality_patience_input"):
        settings.world_quality_early_stopping_patience = int(page._quality_patience_input.value)

    # Story structure settings (chapter counts)
    if hasattr(page, "_chapter_inputs"):
        for key, input_widget in page._chapter_inputs.items():
            setattr(settings, f"chapters_{key}", int(input_widget.value))

    # Data integrity settings
    if hasattr(page, "_entity_version_retention_input"):
        settings.entity_version_retention = int(page._entity_version_retention_input.value)
    if hasattr(page, "_backup_verify_on_restore_switch"):
        settings.backup_verify_on_restore = page._backup_verify_on_restore_switch.value

    # Advanced LLM settings (WP1/WP2)
    advanced_llm_settings_map = [
        ("_circuit_breaker_enabled_switch", "circuit_breaker_enabled", None),
        ("_circuit_breaker_failure_threshold_input", "circuit_breaker_failure_threshold", int),
        ("_circuit_breaker_success_threshold_input", "circuit_breaker_success_threshold", int),
        ("_circuit_breaker_timeout_input", "circuit_breaker_timeout", float),
        ("_retry_temp_increase_input", "retry_temp_increase", float),
        ("_retry_simplify_on_attempt_input", "retry_simplify_on_attempt", int),
        ("_semantic_duplicate_enabled_switch", "semantic_duplicate_enabled", None),
        ("_semantic_duplicate_threshold_input", "semantic_duplicate_threshold", float),
        ("_embedding_model_select", "embedding_model", None),
        ("_refinement_temp_start_input", "world_quality_refinement_temp_start", float),
        ("_refinement_temp_end_input", "world_quality_refinement_temp_end", float),
        ("_refinement_temp_decay_select", "world_quality_refinement_temp_decay", None),
        (
            "_early_stopping_min_iterations_input",
            "world_quality_early_stopping_min_iterations",
            int,
        ),
        (
            "_early_stopping_variance_tolerance_input",
            "world_quality_early_stopping_variance_tolerance",
            float,
        ),
        (
            "_score_plateau_tolerance_input",
            "world_quality_score_plateau_tolerance",
            float,
        ),
        (
            "_dimension_minimum_input",
            "world_quality_dimension_minimum",
            float,
        ),
        # Judge consistency settings
        ("_judge_consistency_switch", "judge_consistency_enabled", None),
        ("_judge_multi_call_switch", "judge_multi_call_enabled", None),
        ("_judge_multi_call_count_input", "judge_multi_call_count", int),
        ("_judge_confidence_threshold_input", "judge_confidence_threshold", float),
        ("_judge_outlier_detection_switch", "judge_outlier_detection", None),
        ("_judge_outlier_std_threshold_input", "judge_outlier_std_threshold", float),
        ("_judge_outlier_strategy_select", "judge_outlier_strategy", None),
        ("_llm_max_concurrent_requests_input", "llm_max_concurrent_requests", int),
        ("_llm_semaphore_timeout_input", "llm_semaphore_timeout", int),
        # Small model timeout cap
        ("_small_model_size_threshold_input", "small_model_size_threshold_gb", float),
        ("_small_model_timeout_cap_input", "small_model_timeout_cap", float),
        # Model service cache TTLs
        ("_model_health_cache_ttl_input", "model_health_cache_ttl", float),
        ("_model_installed_cache_ttl_input", "model_installed_cache_ttl", float),
        ("_model_vram_cache_ttl_input", "model_vram_cache_ttl", float),
        # RAG context settings
        ("_rag_context_enabled_switch", "rag_context_enabled", None),
        ("_rag_context_max_tokens_input", "rag_context_max_tokens", int),
        ("_rag_context_max_items_input", "rag_context_max_items", int),
        ("_rag_context_threshold_input", "rag_context_similarity_threshold", float),
        ("_rag_graph_expansion_switch", "rag_context_graph_expansion", None),
        ("_rag_graph_depth_input", "rag_context_graph_depth", int),
    ]

    for ui_attr, setting_attr, type_conv in advanced_llm_settings_map:
        if hasattr(page, ui_attr):
            ui_element = getattr(page, ui_attr)
            value = ui_element.value
            if type_conv:
                value = type_conv(value)
            setattr(settings, setting_attr, value)

    logger.debug("Advanced settings saved")


def refresh_from_settings(page: SettingsPage) -> None:
    """Refresh advanced UI elements from current settings values.

    Handles world generation, story structure, data integrity, advanced LLM,
    and relationship validation settings. Uses clear-and-rebuild for dynamic
    elements like relationship minimums.

    Args:
        page: The SettingsPage instance.
    """
    settings = page.settings

    # World generation settings
    if hasattr(page, "_world_gen_inputs"):
        for key, (min_input, max_input) in page._world_gen_inputs.items():
            min_attr = f"world_gen_{key}_min"
            max_attr = f"world_gen_{key}_max"
            if hasattr(settings, min_attr):
                min_input.value = getattr(settings, min_attr)
                max_input.value = getattr(settings, max_attr)

    # Quality refinement settings — per-entity thresholds
    if hasattr(page, "_quality_threshold_inputs"):
        for entity_type, input_widget in page._quality_threshold_inputs.items():
            input_widget.value = settings.world_quality_thresholds[entity_type]
    if hasattr(page, "_quality_max_iterations_input"):
        page._quality_max_iterations_input.value = settings.world_quality_max_iterations
    if hasattr(page, "_quality_patience_input"):
        page._quality_patience_input.value = settings.world_quality_early_stopping_patience

    # Data integrity settings
    if hasattr(page, "_entity_version_retention_input"):
        page._entity_version_retention_input.value = settings.entity_version_retention
    if hasattr(page, "_backup_verify_on_restore_switch"):
        page._backup_verify_on_restore_switch.value = settings.backup_verify_on_restore

    # Advanced LLM settings (WP1/WP2)
    advanced_llm_ui_map = [
        ("_circuit_breaker_enabled_switch", "circuit_breaker_enabled"),
        ("_circuit_breaker_failure_threshold_input", "circuit_breaker_failure_threshold"),
        ("_circuit_breaker_success_threshold_input", "circuit_breaker_success_threshold"),
        ("_circuit_breaker_timeout_input", "circuit_breaker_timeout"),
        ("_retry_temp_increase_input", "retry_temp_increase"),
        ("_retry_simplify_on_attempt_input", "retry_simplify_on_attempt"),
        ("_semantic_duplicate_enabled_switch", "semantic_duplicate_enabled"),
        ("_semantic_duplicate_threshold_input", "semantic_duplicate_threshold"),
        ("_embedding_model_select", "embedding_model"),
        ("_refinement_temp_start_input", "world_quality_refinement_temp_start"),
        ("_refinement_temp_end_input", "world_quality_refinement_temp_end"),
        ("_refinement_temp_decay_select", "world_quality_refinement_temp_decay"),
        ("_early_stopping_min_iterations_input", "world_quality_early_stopping_min_iterations"),
        (
            "_early_stopping_variance_tolerance_input",
            "world_quality_early_stopping_variance_tolerance",
        ),
        ("_score_plateau_tolerance_input", "world_quality_score_plateau_tolerance"),
        ("_dimension_minimum_input", "world_quality_dimension_minimum"),
        # Judge consistency settings
        ("_judge_consistency_switch", "judge_consistency_enabled"),
        ("_judge_multi_call_switch", "judge_multi_call_enabled"),
        ("_judge_multi_call_count_input", "judge_multi_call_count"),
        ("_judge_confidence_threshold_input", "judge_confidence_threshold"),
        ("_judge_outlier_detection_switch", "judge_outlier_detection"),
        ("_judge_outlier_std_threshold_input", "judge_outlier_std_threshold"),
        ("_judge_outlier_strategy_select", "judge_outlier_strategy"),
        ("_llm_max_concurrent_requests_input", "llm_max_concurrent_requests"),
        ("_llm_semaphore_timeout_input", "llm_semaphore_timeout"),
        # Small model timeout cap
        ("_small_model_size_threshold_input", "small_model_size_threshold_gb"),
        ("_small_model_timeout_cap_input", "small_model_timeout_cap"),
        # Model service cache TTLs
        ("_model_health_cache_ttl_input", "model_health_cache_ttl"),
        ("_model_installed_cache_ttl_input", "model_installed_cache_ttl"),
        ("_model_vram_cache_ttl_input", "model_vram_cache_ttl"),
        # RAG context settings
        ("_rag_context_enabled_switch", "rag_context_enabled"),
        ("_rag_context_max_tokens_input", "rag_context_max_tokens"),
        ("_rag_context_max_items_input", "rag_context_max_items"),
        ("_rag_context_threshold_input", "rag_context_similarity_threshold"),
        ("_rag_graph_expansion_switch", "rag_context_graph_expansion"),
        ("_rag_graph_depth_input", "rag_context_graph_depth"),
    ]

    for ui_attr, setting_attr in advanced_llm_ui_map:
        if hasattr(page, ui_attr):
            getattr(page, ui_attr).value = getattr(settings, setting_attr)

    logger.debug("Advanced UI refreshed from settings")
