"""Settings page - save, snapshot capture, restore, and UI refresh for undo/redo."""

import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def save_settings(page: SettingsPage) -> None:
    """Persist current UI-configured settings to the application's settings store
    and record an undo snapshot.

    Read values from the page's UI controls and apply them to the settings object.
    Validate and save the updated settings.
    Record an undo action that contains before and after snapshots for undo and redo.
    On success, display a positive notification.
    On validation error or any other failure, log the issue and display a negative
    notification.

    Args:
        page: The SettingsPage instance.
    """
    from src.ui.state import ActionType, UndoAction

    try:
        # Capture old state for undo
        old_snapshot = capture_settings_snapshot(page)

        # Update settings from UI
        page.settings.ollama_url = page._ollama_url_input.value
        page.settings.default_model = page._default_model_select.value
        page.settings.use_per_agent_models = page._use_per_agent.value

        # Per-agent models
        for role, select in page._agent_model_selects.items():
            page.settings.agent_models[role] = select.value

        # Temperatures
        for role, slider in page._temp_sliders.items():
            page.settings.agent_temperatures[role] = slider.value

        # Interaction
        page.settings.interaction_mode = page._interaction_mode_select.value
        page.settings.chapters_between_checkpoints = int(page._checkpoint_input.value)
        page.settings.max_revision_iterations = int(page._revision_input.value)

        # Context
        page.settings.context_size = int(page._context_size_input.value)
        page.settings.max_tokens = int(page._max_tokens_input.value)
        page.settings.previous_chapter_context_chars = int(page._prev_chapter_chars.value)
        page.settings.chapter_analysis_chars = int(page._chapter_analysis_chars.value)
        page.settings.full_text_preview_chars = int(page._full_text_preview_chars.value)

        # Generation mode
        page.settings.use_mode_system = page._use_mode_system.value
        if hasattr(page, "_mode_select"):
            page.settings.current_mode = page._mode_select.value
        if hasattr(page, "_vram_strategy_select"):
            page.settings.vram_strategy = page._vram_strategy_select.value

        # Learning settings
        page.settings.learning_autonomy = page._autonomy_select.value

        # Collect enabled triggers
        enabled_triggers = []
        for trigger_value, checkbox in page._trigger_checkboxes.items():
            if checkbox.value:
                enabled_triggers.append(trigger_value)
        if not enabled_triggers:
            enabled_triggers = ["off"]
        page.settings.learning_triggers = enabled_triggers

        page.settings.learning_periodic_interval = int(page._periodic_interval.value)
        page.settings.learning_min_samples = int(page._min_samples.value)
        page.settings.learning_confidence_threshold = page._confidence_slider.value

        # World generation settings
        for key, (min_input, max_input) in page._world_gen_inputs.items():
            min_val = int(min_input.value)
            max_val = int(max_input.value)
            # Ensure min <= max
            if min_val > max_val:
                max_val = min_val
            setattr(page.settings, f"world_gen_{key}_min", min_val)
            setattr(page.settings, f"world_gen_{key}_max", max_val)

        # Quality refinement settings
        if hasattr(page, "_quality_threshold_input"):
            page.settings.world_quality_threshold = float(page._quality_threshold_input.value)
        if hasattr(page, "_quality_max_iterations_input"):
            page.settings.world_quality_max_iterations = int(
                page._quality_max_iterations_input.value
            )
        if hasattr(page, "_quality_patience_input"):
            page.settings.world_quality_early_stopping_patience = int(
                page._quality_patience_input.value
            )

        # Story structure settings (chapter counts)
        if hasattr(page, "_chapter_inputs"):
            for key, input_widget in page._chapter_inputs.items():
                setattr(page.settings, f"chapters_{key}", int(input_widget.value))

        # Data integrity settings
        if hasattr(page, "_entity_version_retention_input"):
            page.settings.entity_version_retention = int(page._entity_version_retention_input.value)
        if hasattr(page, "_backup_verify_on_restore_switch"):
            page.settings.backup_verify_on_restore = page._backup_verify_on_restore_switch.value

        # Advanced LLM settings (WP1/WP2) - use mapping to reduce repetition
        advanced_llm_settings_map = [
            ("_circuit_breaker_enabled_switch", "circuit_breaker_enabled", None),
            (
                "_circuit_breaker_failure_threshold_input",
                "circuit_breaker_failure_threshold",
                int,
            ),
            (
                "_circuit_breaker_success_threshold_input",
                "circuit_breaker_success_threshold",
                int,
            ),
            ("_circuit_breaker_timeout_input", "circuit_breaker_timeout", float),
            ("_retry_temp_increase_input", "retry_temp_increase", float),
            ("_retry_simplify_on_attempt_input", "retry_simplify_on_attempt", int),
            ("_semantic_duplicate_enabled_switch", "semantic_duplicate_enabled", None),
            ("_semantic_duplicate_threshold_input", "semantic_duplicate_threshold", float),
            ("_embedding_model_select", "embedding_model", None),
            (
                "_refinement_temp_start_input",
                "world_quality_refinement_temp_start",
                float,
            ),
            ("_refinement_temp_end_input", "world_quality_refinement_temp_end", float),
            (
                "_refinement_temp_decay_select",
                "world_quality_refinement_temp_decay",
                None,
            ),
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
        ]

        for ui_attr, setting_attr, type_conv in advanced_llm_settings_map:
            if hasattr(page, ui_attr):
                ui_element = getattr(page, ui_attr)
                value = ui_element.value
                if type_conv:
                    value = type_conv(value)
                setattr(page.settings, setting_attr, value)
                logger.debug(f"Updated {setting_attr} to {value}")

        # Relationship validation settings
        if hasattr(page, "_relationship_validation_switch"):
            page.settings.relationship_validation_enabled = (
                page._relationship_validation_switch.value
            )
        if hasattr(page, "_orphan_detection_switch"):
            page.settings.orphan_detection_enabled = page._orphan_detection_switch.value
        if hasattr(page, "_circular_detection_switch"):
            page.settings.circular_detection_enabled = page._circular_detection_switch.value
        if hasattr(page, "_fuzzy_threshold_input"):
            page.settings.fuzzy_match_threshold = float(page._fuzzy_threshold_input.value)
        if hasattr(page, "_max_relationships_input"):
            page.settings.max_relationships_per_entity = int(page._max_relationships_input.value)
        # circular_relationship_types is modified directly by the chip UI, no extraction needed

        # Relationship minimums (extract from number inputs)
        if hasattr(page, "_relationship_min_inputs"):
            for entity_type, roles in page._relationship_min_inputs.items():
                if entity_type not in page.settings.relationship_minimums:
                    raise ValueError(f"Unknown relationship minimums entity type: {entity_type}")
                for role, num_input in roles.items():
                    if role not in page.settings.relationship_minimums[entity_type]:
                        raise ValueError(
                            f"Unknown relationship minimums role: {entity_type}/{role}"
                        )
                    if num_input.value is None:
                        raise ValueError(f"Relationship minimum required for {entity_type}/{role}")
                    page.settings.relationship_minimums[entity_type][role] = int(num_input.value)

        # Calendar and temporal validation settings
        if hasattr(page, "_generate_calendar_switch"):
            page.settings.generate_calendar_on_world_build = page._generate_calendar_switch.value
        if hasattr(page, "_temporal_validation_switch"):
            page.settings.validate_temporal_consistency = page._temporal_validation_switch.value

        # Validate and save first - only record undo if successful
        page.settings.validate()
        page.settings.save()

        # Capture new state for redo AFTER successful save
        new_snapshot = capture_settings_snapshot(page)

        # Record undo action only after successful validation and save
        page.state.record_action(
            UndoAction(
                action_type=ActionType.UPDATE_SETTINGS,
                data=new_snapshot,
                inverse_data=old_snapshot,
            )
        )

        ui.notify("Settings saved!", type="positive")

    except ValueError as e:
        logger.warning(f"Invalid setting value: {e}")
        ui.notify(f"Invalid setting: {e}", type="negative")
    except Exception as e:
        logger.exception("Failed to save settings")
        ui.notify(f"Error saving: {e}", type="negative")


def capture_settings_snapshot(page: SettingsPage) -> dict[str, Any]:
    """Create a serializable snapshot of current settings suitable for undo/redo.

    Args:
        page: The SettingsPage instance.

    Returns:
        A dictionary snapshot of all settings values.
    """
    settings = page.settings
    snapshot = {
        "ollama_url": settings.ollama_url,
        "default_model": settings.default_model,
        "use_per_agent_models": settings.use_per_agent_models,
        "agent_models": dict(settings.agent_models),
        "agent_temperatures": dict(settings.agent_temperatures),
        "interaction_mode": settings.interaction_mode,
        "chapters_between_checkpoints": settings.chapters_between_checkpoints,
        "max_revision_iterations": settings.max_revision_iterations,
        "context_size": settings.context_size,
        "max_tokens": settings.max_tokens,
        "previous_chapter_context_chars": settings.previous_chapter_context_chars,
        "chapter_analysis_chars": settings.chapter_analysis_chars,
        "full_text_preview_chars": settings.full_text_preview_chars,
        "use_mode_system": settings.use_mode_system,
        "current_mode": settings.current_mode,
        "vram_strategy": settings.vram_strategy,
        "learning_autonomy": settings.learning_autonomy,
        "learning_triggers": list(settings.learning_triggers),
        "learning_periodic_interval": settings.learning_periodic_interval,
        "learning_min_samples": settings.learning_min_samples,
        "learning_confidence_threshold": settings.learning_confidence_threshold,
        # World generation
        "world_gen_characters_min": settings.world_gen_characters_min,
        "world_gen_characters_max": settings.world_gen_characters_max,
        "world_gen_locations_min": settings.world_gen_locations_min,
        "world_gen_locations_max": settings.world_gen_locations_max,
        "world_gen_factions_min": settings.world_gen_factions_min,
        "world_gen_factions_max": settings.world_gen_factions_max,
        "world_gen_items_min": settings.world_gen_items_min,
        "world_gen_items_max": settings.world_gen_items_max,
        "world_gen_concepts_min": settings.world_gen_concepts_min,
        "world_gen_concepts_max": settings.world_gen_concepts_max,
        "world_gen_relationships_min": settings.world_gen_relationships_min,
        "world_gen_relationships_max": settings.world_gen_relationships_max,
        # Quality refinement
        "world_quality_threshold": settings.world_quality_threshold,
        "world_quality_max_iterations": settings.world_quality_max_iterations,
        "world_quality_early_stopping_patience": (settings.world_quality_early_stopping_patience),
        # Data integrity
        "entity_version_retention": settings.entity_version_retention,
        "backup_verify_on_restore": settings.backup_verify_on_restore,
        # Relationship validation
        "relationship_validation_enabled": settings.relationship_validation_enabled,
        "orphan_detection_enabled": settings.orphan_detection_enabled,
        "circular_detection_enabled": settings.circular_detection_enabled,
        "fuzzy_match_threshold": settings.fuzzy_match_threshold,
        "max_relationships_per_entity": settings.max_relationships_per_entity,
        "circular_relationship_types": settings.circular_relationship_types.copy(),
        "relationship_minimums": {
            et: roles.copy() for et, roles in settings.relationship_minimums.items()
        },
        # Calendar and temporal validation
        "generate_calendar_on_world_build": settings.generate_calendar_on_world_build,
        "validate_temporal_consistency": settings.validate_temporal_consistency,
    }

    # Advanced LLM settings (WP1/WP2) - add using key iteration
    advanced_llm_keys = [
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_success_threshold",
        "circuit_breaker_timeout",
        "retry_temp_increase",
        "retry_simplify_on_attempt",
        "semantic_duplicate_enabled",
        "semantic_duplicate_threshold",
        "embedding_model",
        "world_quality_refinement_temp_start",
        "world_quality_refinement_temp_end",
        "world_quality_refinement_temp_decay",
        "world_quality_early_stopping_min_iterations",
        "world_quality_early_stopping_variance_tolerance",
    ]
    for key in advanced_llm_keys:
        snapshot[key] = getattr(settings, key)

    return snapshot


def restore_settings_snapshot(page: SettingsPage, snapshot: dict[str, Any]) -> None:
    """Restore the SettingsPage state from a snapshot and persist the restored values.

    Args:
        page: The SettingsPage instance.
        snapshot: Snapshot containing saved settings values.
    """
    settings = page.settings

    settings.ollama_url = snapshot["ollama_url"]
    settings.default_model = snapshot["default_model"]
    settings.use_per_agent_models = snapshot["use_per_agent_models"]
    settings.agent_models = dict(snapshot["agent_models"])
    settings.agent_temperatures = dict(snapshot["agent_temperatures"])
    settings.interaction_mode = snapshot["interaction_mode"]
    settings.chapters_between_checkpoints = snapshot["chapters_between_checkpoints"]
    settings.max_revision_iterations = snapshot["max_revision_iterations"]
    settings.context_size = snapshot["context_size"]
    settings.max_tokens = snapshot["max_tokens"]
    settings.previous_chapter_context_chars = snapshot["previous_chapter_context_chars"]
    settings.chapter_analysis_chars = snapshot["chapter_analysis_chars"]
    if "full_text_preview_chars" in snapshot:
        settings.full_text_preview_chars = snapshot["full_text_preview_chars"]
    settings.use_mode_system = snapshot["use_mode_system"]
    settings.current_mode = snapshot["current_mode"]
    if "vram_strategy" in snapshot:
        settings.vram_strategy = snapshot["vram_strategy"]
    settings.learning_autonomy = snapshot["learning_autonomy"]
    settings.learning_triggers = list(snapshot["learning_triggers"])
    settings.learning_periodic_interval = snapshot["learning_periodic_interval"]
    settings.learning_min_samples = snapshot["learning_min_samples"]
    settings.learning_confidence_threshold = snapshot["learning_confidence_threshold"]

    # World generation (with backward compatibility for old snapshots)
    if "world_gen_characters_min" in snapshot:
        settings.world_gen_characters_min = snapshot["world_gen_characters_min"]
        settings.world_gen_characters_max = snapshot["world_gen_characters_max"]
        settings.world_gen_locations_min = snapshot["world_gen_locations_min"]
        settings.world_gen_locations_max = snapshot["world_gen_locations_max"]
        settings.world_gen_factions_min = snapshot["world_gen_factions_min"]
        settings.world_gen_factions_max = snapshot["world_gen_factions_max"]
        settings.world_gen_items_min = snapshot["world_gen_items_min"]
        settings.world_gen_items_max = snapshot["world_gen_items_max"]
        settings.world_gen_concepts_min = snapshot["world_gen_concepts_min"]
        settings.world_gen_concepts_max = snapshot["world_gen_concepts_max"]
        settings.world_gen_relationships_min = snapshot["world_gen_relationships_min"]
        settings.world_gen_relationships_max = snapshot["world_gen_relationships_max"]

    # Quality refinement (with backward compatibility for old snapshots)
    if "world_quality_threshold" in snapshot:
        settings.world_quality_threshold = snapshot["world_quality_threshold"]
    if "world_quality_max_iterations" in snapshot:
        settings.world_quality_max_iterations = snapshot["world_quality_max_iterations"]
    if "world_quality_early_stopping_patience" in snapshot:
        settings.world_quality_early_stopping_patience = snapshot[
            "world_quality_early_stopping_patience"
        ]

    # Data integrity (with backward compatibility for old snapshots)
    if "entity_version_retention" in snapshot:
        settings.entity_version_retention = snapshot["entity_version_retention"]
    if "backup_verify_on_restore" in snapshot:
        settings.backup_verify_on_restore = snapshot["backup_verify_on_restore"]

    # Advanced LLM settings (with backward compatibility for old snapshots)
    advanced_llm_keys = [
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_success_threshold",
        "circuit_breaker_timeout",
        "retry_temp_increase",
        "retry_simplify_on_attempt",
        "semantic_duplicate_enabled",
        "semantic_duplicate_threshold",
        "embedding_model",
        "world_quality_refinement_temp_start",
        "world_quality_refinement_temp_end",
        "world_quality_refinement_temp_decay",
        "world_quality_early_stopping_min_iterations",
        "world_quality_early_stopping_variance_tolerance",
    ]
    for key in advanced_llm_keys:
        if key in snapshot:
            setattr(settings, key, snapshot[key])

    # Relationship validation (with backward compatibility for old snapshots)
    if "relationship_validation_enabled" in snapshot:
        settings.relationship_validation_enabled = snapshot["relationship_validation_enabled"]
    if "orphan_detection_enabled" in snapshot:
        settings.orphan_detection_enabled = snapshot["orphan_detection_enabled"]
    if "circular_detection_enabled" in snapshot:
        settings.circular_detection_enabled = snapshot["circular_detection_enabled"]
    if "fuzzy_match_threshold" in snapshot:
        settings.fuzzy_match_threshold = snapshot["fuzzy_match_threshold"]
    if "max_relationships_per_entity" in snapshot:
        settings.max_relationships_per_entity = snapshot["max_relationships_per_entity"]
    if "circular_relationship_types" in snapshot:
        # Create a copy to keep the snapshot immutable (chip UI mutates in-place)
        settings.circular_relationship_types = list(snapshot["circular_relationship_types"])
    if "relationship_minimums" in snapshot:
        # Shallow copy the nested dict (roles.copy() creates shallow copies of inner dicts)
        settings.relationship_minimums = {
            et: roles.copy() for et, roles in snapshot["relationship_minimums"].items()
        }

    # Calendar and temporal validation (with backward compatibility for old snapshots)
    if "generate_calendar_on_world_build" in snapshot:
        settings.generate_calendar_on_world_build = snapshot["generate_calendar_on_world_build"]
    if "validate_temporal_consistency" in snapshot:
        settings.validate_temporal_consistency = snapshot["validate_temporal_consistency"]

    # Save changes
    settings.save()

    # Update UI elements to reflect restored values
    refresh_ui_from_settings(page)

    ui.notify("Settings restored", type="info")
    logger.debug("Settings snapshot restored")


def refresh_ui_from_settings(page: SettingsPage) -> None:
    """Refresh all UI input elements from current settings values.

    Args:
        page: The SettingsPage instance.
    """
    logger.debug("Refreshing UI elements from settings")
    settings = page.settings

    # Core settings
    if hasattr(page, "_ollama_url_input") and page._ollama_url_input:
        page._ollama_url_input.value = settings.ollama_url
    if hasattr(page, "_default_model_select") and page._default_model_select:
        page._default_model_select.value = settings.default_model
    if hasattr(page, "_use_per_agent") and page._use_per_agent:
        page._use_per_agent.value = settings.use_per_agent_models

    # Per-agent model selects
    if hasattr(page, "_agent_model_selects"):
        for role, select in page._agent_model_selects.items():
            if select and role in settings.agent_models:
                select.value = settings.agent_models[role]

    # Temperature sliders
    if hasattr(page, "_temp_sliders"):
        for role, slider in page._temp_sliders.items():
            if slider and role in settings.agent_temperatures:
                slider.value = settings.agent_temperatures[role]

    # Workflow settings
    if hasattr(page, "_interaction_mode_select") and page._interaction_mode_select:
        page._interaction_mode_select.value = settings.interaction_mode
    if hasattr(page, "_checkpoint_input") and page._checkpoint_input:
        page._checkpoint_input.value = settings.chapters_between_checkpoints
    if hasattr(page, "_revision_input") and page._revision_input:
        page._revision_input.value = settings.max_revision_iterations

    # Context settings
    if hasattr(page, "_context_size_input") and page._context_size_input:
        page._context_size_input.value = settings.context_size
    if hasattr(page, "_max_tokens_input") and page._max_tokens_input:
        page._max_tokens_input.value = settings.max_tokens
    if hasattr(page, "_full_text_preview_chars") and page._full_text_preview_chars:
        page._full_text_preview_chars.value = settings.full_text_preview_chars

    # Mode settings
    if hasattr(page, "_mode_select") and page._mode_select:
        page._mode_select.value = settings.current_mode
    if hasattr(page, "_vram_strategy_select") and page._vram_strategy_select:
        page._vram_strategy_select.value = settings.vram_strategy

    # Learning settings
    if hasattr(page, "_autonomy_select") and page._autonomy_select:
        page._autonomy_select.value = settings.learning_autonomy
    if hasattr(page, "_confidence_slider") and page._confidence_slider:
        page._confidence_slider.value = settings.learning_confidence_threshold
    if hasattr(page, "_periodic_interval") and page._periodic_interval:
        page._periodic_interval.value = settings.learning_periodic_interval
    if hasattr(page, "_min_samples") and page._min_samples:
        page._min_samples.value = settings.learning_min_samples

    # World generation settings
    if hasattr(page, "_world_gen_inputs"):
        for key, (min_input, max_input) in page._world_gen_inputs.items():
            min_attr = f"world_gen_{key}_min"
            max_attr = f"world_gen_{key}_max"
            if hasattr(settings, min_attr):
                min_input.value = getattr(settings, min_attr)
                max_input.value = getattr(settings, max_attr)

    # Quality refinement settings
    if hasattr(page, "_quality_threshold_input") and page._quality_threshold_input:
        page._quality_threshold_input.value = settings.world_quality_threshold
    if hasattr(page, "_quality_max_iterations_input") and page._quality_max_iterations_input:
        page._quality_max_iterations_input.value = settings.world_quality_max_iterations
    if hasattr(page, "_quality_patience_input") and page._quality_patience_input:
        page._quality_patience_input.value = settings.world_quality_early_stopping_patience

    # Data integrity settings
    if hasattr(page, "_entity_version_retention_input") and page._entity_version_retention_input:
        page._entity_version_retention_input.value = settings.entity_version_retention
    if hasattr(page, "_backup_verify_on_restore_switch") and page._backup_verify_on_restore_switch:
        page._backup_verify_on_restore_switch.value = settings.backup_verify_on_restore

    # Advanced LLM settings (WP1/WP2) - use mapping to reduce repetition
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
        (
            "_early_stopping_min_iterations_input",
            "world_quality_early_stopping_min_iterations",
        ),
        (
            "_early_stopping_variance_tolerance_input",
            "world_quality_early_stopping_variance_tolerance",
        ),
    ]

    for ui_attr, setting_attr in advanced_llm_ui_map:
        if hasattr(page, ui_attr) and getattr(page, ui_attr):
            getattr(page, ui_attr).value = getattr(settings, setting_attr)

    # Relationship validation settings
    if hasattr(page, "_relationship_validation_switch") and page._relationship_validation_switch:
        page._relationship_validation_switch.value = settings.relationship_validation_enabled
    if hasattr(page, "_orphan_detection_switch") and page._orphan_detection_switch:
        page._orphan_detection_switch.value = settings.orphan_detection_enabled
    if hasattr(page, "_circular_detection_switch") and page._circular_detection_switch:
        page._circular_detection_switch.value = settings.circular_detection_enabled
    if hasattr(page, "_fuzzy_threshold_input") and page._fuzzy_threshold_input:
        page._fuzzy_threshold_input.value = settings.fuzzy_match_threshold
    if hasattr(page, "_max_relationships_input") and page._max_relationships_input:
        page._max_relationships_input.value = settings.max_relationships_per_entity
    # circular_relationship_types chips are rebuilt from settings directly

    # Relationship minimums
    if hasattr(page, "_relationship_min_inputs"):
        logger.debug("Refreshing relationship minimum inputs from settings")
        for entity_type, roles in page._relationship_min_inputs.items():
            for role, num_input in roles.items():
                if entity_type in settings.relationship_minimums:
                    if role in settings.relationship_minimums[entity_type]:
                        num_input.value = settings.relationship_minimums[entity_type][role]

    # Rebuild circular type chips from settings
    if hasattr(page, "_circular_types_container"):
        logger.debug("Rebuilding circular type chips from settings")
        from src.ui.pages.settings._advanced import _build_circular_type_chips

        _build_circular_type_chips(page)

    # Calendar and temporal validation settings
    if hasattr(page, "_generate_calendar_switch") and page._generate_calendar_switch:
        page._generate_calendar_switch.value = settings.generate_calendar_on_world_build
    if hasattr(page, "_temporal_validation_switch") and page._temporal_validation_switch:
        page._temporal_validation_switch.value = settings.validate_temporal_consistency
