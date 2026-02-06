"""Settings page - save, snapshot capture, restore, and UI refresh for undo/redo.

This module coordinates persistence operations by delegating to section-specific
modules. Each settings section module (_connection, _models, _modes, _interaction,
_advanced, _world_validation) handles its own save and refresh logic.
"""

import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui

if TYPE_CHECKING:
    from src.ui.pages.settings import SettingsPage

logger = logging.getLogger(__name__)


def save_settings(page: SettingsPage) -> None:
    """Persist current UI-configured settings to the application's settings store
    and record an undo snapshot.

    Delegates to section-specific save functions, validates settings, and records
    an undo action with before and after snapshots.

    Args:
        page: The SettingsPage instance.
    """
    from src.ui.pages.settings import (
        _advanced,
        _connection,
        _interaction,
        _models,
        _modes,
        _world_validation,
    )
    from src.ui.state import ActionType, UndoAction

    try:
        # Capture old state for undo
        old_snapshot = capture_settings_snapshot(page)

        # Delegate to section-specific save functions
        _connection.save_to_settings(page)
        _models.save_to_settings(page)
        _interaction.save_to_settings(page)
        _modes.save_to_settings(page)
        _advanced.save_to_settings(page)
        _world_validation.save_to_settings(page)

        # Validate and save first - only record undo if successful
        page.settings.validate()
        page.settings.save()

        # Apply the log level at runtime
        from src.utils.logging_config import set_log_level

        set_log_level(page.settings.log_level)

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
        logger.debug("Settings saved successfully")

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
        # Connection
        "ollama_url": settings.ollama_url,
        "log_level": settings.log_level,
        # Models
        "default_model": settings.default_model,
        "use_per_agent_models": settings.use_per_agent_models,
        "agent_models": dict(settings.agent_models),
        "agent_temperatures": dict(settings.agent_temperatures),
        # Interaction
        "interaction_mode": settings.interaction_mode,
        "chapters_between_checkpoints": settings.chapters_between_checkpoints,
        "max_revision_iterations": settings.max_revision_iterations,
        # Context
        "context_size": settings.context_size,
        "max_tokens": settings.max_tokens,
        "previous_chapter_context_chars": settings.previous_chapter_context_chars,
        "chapter_analysis_chars": settings.chapter_analysis_chars,
        "full_text_preview_chars": settings.full_text_preview_chars,
        # Mode
        "use_mode_system": settings.use_mode_system,
        "current_mode": settings.current_mode,
        "vram_strategy": settings.vram_strategy,
        # Learning
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
        "world_quality_early_stopping_patience": settings.world_quality_early_stopping_patience,
        # Data integrity
        "entity_version_retention": settings.entity_version_retention,
        "backup_verify_on_restore": settings.backup_verify_on_restore,
        # Relationship validation
        "relationship_validation_enabled": settings.relationship_validation_enabled,
        "orphan_detection_enabled": settings.orphan_detection_enabled,
        "circular_detection_enabled": settings.circular_detection_enabled,
        "circular_check_all_types": settings.circular_check_all_types,
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

    # Advanced LLM settings (WP1/WP2)
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

    # Connection
    settings.ollama_url = snapshot["ollama_url"]
    settings.log_level = snapshot["log_level"]

    # Models
    settings.default_model = snapshot["default_model"]
    settings.use_per_agent_models = snapshot["use_per_agent_models"]
    settings.agent_models = dict(snapshot["agent_models"])
    settings.agent_temperatures = dict(snapshot["agent_temperatures"])

    # Interaction
    settings.interaction_mode = snapshot["interaction_mode"]
    settings.chapters_between_checkpoints = snapshot["chapters_between_checkpoints"]
    settings.max_revision_iterations = snapshot["max_revision_iterations"]

    # Context
    settings.context_size = snapshot["context_size"]
    settings.max_tokens = snapshot["max_tokens"]
    settings.previous_chapter_context_chars = snapshot["previous_chapter_context_chars"]
    settings.chapter_analysis_chars = snapshot["chapter_analysis_chars"]
    if "full_text_preview_chars" in snapshot:
        settings.full_text_preview_chars = snapshot["full_text_preview_chars"]

    # Mode
    settings.use_mode_system = snapshot["use_mode_system"]
    settings.current_mode = snapshot["current_mode"]
    if "vram_strategy" in snapshot:
        settings.vram_strategy = snapshot["vram_strategy"]

    # Learning
    settings.learning_autonomy = snapshot["learning_autonomy"]
    settings.learning_triggers = list(snapshot["learning_triggers"])
    settings.learning_periodic_interval = snapshot["learning_periodic_interval"]
    settings.learning_min_samples = snapshot["learning_min_samples"]
    settings.learning_confidence_threshold = snapshot["learning_confidence_threshold"]

    # World generation (with backward compatibility)
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

    # Quality refinement (with backward compatibility)
    if "world_quality_threshold" in snapshot:
        settings.world_quality_threshold = snapshot["world_quality_threshold"]
    if "world_quality_max_iterations" in snapshot:
        settings.world_quality_max_iterations = snapshot["world_quality_max_iterations"]
    if "world_quality_early_stopping_patience" in snapshot:
        settings.world_quality_early_stopping_patience = snapshot[
            "world_quality_early_stopping_patience"
        ]

    # Data integrity (with backward compatibility)
    if "entity_version_retention" in snapshot:
        settings.entity_version_retention = snapshot["entity_version_retention"]
    if "backup_verify_on_restore" in snapshot:
        settings.backup_verify_on_restore = snapshot["backup_verify_on_restore"]

    # Advanced LLM settings (with backward compatibility)
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

    # Relationship validation (with backward compatibility)
    if "relationship_validation_enabled" in snapshot:
        settings.relationship_validation_enabled = snapshot["relationship_validation_enabled"]
    if "orphan_detection_enabled" in snapshot:
        settings.orphan_detection_enabled = snapshot["orphan_detection_enabled"]
    if "circular_detection_enabled" in snapshot:
        settings.circular_detection_enabled = snapshot["circular_detection_enabled"]
    if "circular_check_all_types" in snapshot:
        settings.circular_check_all_types = snapshot["circular_check_all_types"]
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

    # Calendar and temporal validation (with backward compatibility)
    if "generate_calendar_on_world_build" in snapshot:
        settings.generate_calendar_on_world_build = snapshot["generate_calendar_on_world_build"]
    if "validate_temporal_consistency" in snapshot:
        settings.validate_temporal_consistency = snapshot["validate_temporal_consistency"]

    # Save changes
    settings.save()

    # Apply the log level at runtime
    from src.utils.logging_config import set_log_level

    set_log_level(settings.log_level)

    # Update UI elements to reflect restored values
    refresh_ui_from_settings(page)

    ui.notify("Settings restored", type="info")
    logger.debug("Settings snapshot restored")


def refresh_ui_from_settings(page: SettingsPage) -> None:
    """Refresh all UI input elements from current settings values.

    Delegates to section-specific refresh functions for better encapsulation.

    Args:
        page: The SettingsPage instance.
    """
    from src.ui.pages.settings import (
        _advanced,
        _connection,
        _interaction,
        _models,
        _modes,
        _world_validation,
    )

    logger.debug("Refreshing UI elements from settings")

    # Delegate to section-specific refresh functions
    _connection.refresh_from_settings(page)
    _models.refresh_from_settings(page)
    _interaction.refresh_from_settings(page)
    _modes.refresh_from_settings(page)
    _advanced.refresh_from_settings(page)
    _world_validation.refresh_from_settings(page)

    logger.debug("UI refresh complete")
