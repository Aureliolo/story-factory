"""Tests for the settings module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.settings import (
    AGENT_ROLES,
    RECOMMENDED_MODELS,
    Settings,
    get_available_vram,
    get_installed_models,
    get_installed_models_with_sizes,
    get_model_info,
)


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Should have sensible default values."""
        settings = Settings()
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.context_size == 32768
        assert settings.max_tokens == 8192
        assert settings.use_per_agent_models is True
        assert settings.interaction_mode == "checkpoint"
        assert settings.world_quality_threshold == 7.5

    def test_default_agent_models_includes_judge(self):
        """Default agent_models should include 'judge' role set to 'auto'."""
        settings = Settings()
        assert "judge" in settings.agent_models
        assert settings.agent_models["judge"] == "auto"

    def test_default_agent_temperatures_includes_judge(self):
        """Default agent_temperatures should include 'judge' role."""
        settings = Settings()
        assert "judge" in settings.agent_temperatures
        assert settings.agent_temperatures["judge"] == 0.1

    def test_embedding_model_defaults_to_empty(self):
        """Embedding model should default to empty string (no hardcoded model)."""
        settings = Settings()
        assert settings.embedding_model == ""

    def test_get_temperature_for_agent(self):
        """Should return correct temperature for each agent role."""
        settings = Settings()
        assert settings.get_temperature_for_agent("writer") == 0.9
        assert settings.get_temperature_for_agent("editor") == 0.6
        assert settings.get_temperature_for_agent("continuity") == 0.3

    def test_get_temperature_for_agent_unknown_raises(self):
        """Should raise ValueError for unknown agent role."""
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown agent role"):
            settings.get_temperature_for_agent("unknown")

    def test_get_model_for_agent_with_override(self):
        """Should return overridden model when set."""
        settings = Settings()
        settings.agent_models["writer"] = "custom-model:7b"
        result = settings.get_model_for_agent("writer")
        assert result == "custom-model:7b"

    def test_get_model_for_agent_uses_default_when_not_per_agent(self):
        """Should return default model when per-agent models disabled."""
        settings = Settings()
        settings.use_per_agent_models = False
        settings.default_model = "test-model:8b"
        result = settings.get_model_for_agent("writer")
        assert result == "test-model:8b"

    def test_validate_raises_on_invalid_url(self):
        """Should raise ValueError for invalid Ollama URL."""
        settings = Settings()
        settings.ollama_url = "not-a-url"
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            settings.validate()

    def test_validate_raises_on_invalid_context_size(self):
        """Should raise ValueError for context_size out of range."""
        settings = Settings()
        settings.context_size = 500  # Too small
        with pytest.raises(ValueError, match="context_size must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_max_tokens(self):
        """Should raise ValueError for max_tokens out of range."""
        settings = Settings()
        settings.max_tokens = 100000  # Too large
        with pytest.raises(ValueError, match="max_tokens must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_interaction_mode(self):
        """Should raise ValueError for invalid interaction mode."""
        settings = Settings()
        settings.interaction_mode = "invalid_mode"
        with pytest.raises(ValueError, match="interaction_mode must be one of"):
            settings.validate()

    def test_validate_raises_on_invalid_temperature(self):
        """Should raise ValueError for temperature out of range."""
        settings = Settings()
        settings.agent_temperatures["writer"] = 3.0  # Too high
        with pytest.raises(ValueError, match="Temperature for writer must be between"):
            settings.validate()

    def test_validate_raises_on_unknown_agent_in_temperatures(self):
        """Should raise ValueError for unknown agent in agent_temperatures."""
        settings = Settings()
        settings.agent_temperatures["unknown_agent"] = 0.7
        with pytest.raises(ValueError, match="Unknown agent\\(s\\) in agent_temperatures"):
            settings.validate()

    def test_validate_raises_on_unknown_agent_in_models(self):
        """Should raise ValueError for unknown agent in agent_models."""
        settings = Settings()
        settings.agent_models["unknown_agent"] = "some-model:8b"
        with pytest.raises(ValueError, match="Unknown agent\\(s\\) in agent_models"):
            settings.validate()

    def test_validate_backfills_missing_agent_models(self):
        """Should backfill missing roles in agent_models from defaults."""
        settings = Settings()
        del settings.agent_models["judge"]  # Simulate old settings file
        changed = settings.validate()
        assert changed is True
        assert "judge" in settings.agent_models
        assert settings.agent_models["judge"] == "auto"

    def test_validate_agent_models_no_change_when_complete(self):
        """Should not modify agent_models when all expected roles are present."""
        settings = Settings()
        original_models = dict(settings.agent_models)
        settings.validate()
        assert settings.agent_models == original_models

    def test_validate_passes_for_valid_settings(self):
        """Should not raise for valid default settings."""
        settings = Settings()
        settings.validate()  # Should not raise


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_known_model_info(self):
        """Should return info for known models."""
        # Pick a model from the registry
        model_id = next(iter(RECOMMENDED_MODELS.keys()))
        info = get_model_info(model_id)
        assert "name" in info
        assert "quality" in info
        assert "speed" in info
        assert "tags" in info

    def test_returns_estimated_info_for_unknown_model(self, monkeypatch):
        """Should return estimated info for unknown models based on size."""
        # Mock installed models to return the unknown model with a size
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {"completely-unknown-model:99b": 10.0},
        )
        info = get_model_info("completely-unknown-model:99b")
        assert info["name"] == "completely-unknown-model:99b"
        # Quality/speed estimated from 10GB size
        assert info["quality"] > 0
        assert info["speed"] > 0
        assert info["tags"] == []  # No tags for unknown models

    def test_matches_model_by_base_name(self):
        """Should match model by base name if exact match not found."""
        # Use a base name that matches an existing model
        # "huihui_ai/qwen3-abliterated:xyz" should match "huihui_ai/qwen3-abliterated:30b"
        info = get_model_info("huihui_ai/qwen3-abliterated:xyz")
        # Should return info from the matching base name (not estimated defaults)
        assert info is not None
        assert "quality" in info
        # The quality should come from the RECOMMENDED_MODELS entry, not defaults
        assert info["quality"] >= 7  # Known models have good quality scores

    def test_returns_defaults_for_unknown_model_with_no_size(self, monkeypatch):
        """Should return default values when model size is unknown."""
        # Mock installed models to return empty or no size for the model
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {},
        )
        info = get_model_info("completely-unknown-model:xyz")
        assert info["name"] == "completely-unknown-model:xyz"
        # Should use default values (quality=5, speed=5, vram=8)
        assert info["quality"] == 5
        assert info["speed"] == 5
        assert info["vram_required"] == 8

    def test_estimates_quality_speed_from_installed_size(self, monkeypatch):
        """Should estimate quality/speed from installed model file size when size_gb > 0."""
        # Patch in the correct module where get_model_info actually calls the function
        monkeypatch.setattr(
            "src.settings._utils.get_installed_models_with_sizes",
            lambda timeout=None: {"custom-unknown-model:7b": 7.5},
        )
        info = get_model_info("custom-unknown-model:7b")
        assert info["name"] == "custom-unknown-model:7b"
        # quality = min(10, int(7.5 / 4) + 4) = min(10, 1 + 4) = 5
        assert info["quality"] == 5
        # speed = max(1, 10 - int(7.5 / 5)) = max(1, 10 - 1) = 9
        assert info["speed"] == 9
        # vram_required = int(7.5 * 1.2) = 9
        assert info["vram_required"] == 9
        # size_gb should be the actual installed size, not the default
        assert info["size_gb"] == 7.5
        # Unknown models have no tags
        assert info["tags"] == []


class TestAgentRoles:
    """Tests for agent role definitions."""

    def test_all_roles_defined(self):
        """Should have all required agent roles defined."""
        required_roles = [
            "interviewer",
            "architect",
            "writer",
            "editor",
            "continuity",
            "validator",
            "suggestion",
            "embedding",
        ]
        for role in required_roles:
            assert role in AGENT_ROLES
            assert "name" in AGENT_ROLES[role]
            assert "description" in AGENT_ROLES[role]


class TestEmbeddingTemperature:
    """Tests for the embedding agent temperature default."""

    def test_embedding_role_has_default_temperature(self):
        """Embedding role must have a default temperature in agent_temperatures."""
        settings = Settings()
        assert "embedding" in settings.agent_temperatures
        assert settings.agent_temperatures["embedding"] == 0.0

    def test_get_temperature_for_embedding_role(self):
        """get_temperature_for_agent returns the embedding temperature without error."""
        settings = Settings()
        temp = settings.get_temperature_for_agent("embedding")
        assert temp == 0.0


class TestSettingsValidation:
    """Additional tests for Settings validation."""

    def test_validate_raises_on_url_missing_host(self):
        """Should raise ValueError for URL without host."""
        settings = Settings()
        settings.ollama_url = "http://"
        with pytest.raises(ValueError, match="Invalid URL"):
            settings.validate()

    def test_validate_raises_on_url_none_type(self):
        """Should raise ValueError for URL that is None."""
        settings = Settings()
        settings.ollama_url = None  # type: ignore
        # urlparse(None) returns empty scheme, so triggers "Invalid URL scheme"
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            settings.validate()

    def test_validate_raises_on_url_non_string(self):
        """Should raise ValueError for URL that is not a string."""
        settings = Settings()
        # Pass a custom object that causes urlparse to raise TypeError
        settings.ollama_url = object()  # type: ignore
        with pytest.raises(ValueError, match="Invalid ollama_url"):
            settings.validate()

    def test_validate_raises_on_invalid_chapters_between_checkpoints(self):
        """Should raise ValueError for chapters_between_checkpoints out of range."""
        settings = Settings()
        settings.chapters_between_checkpoints = 25  # Too high
        with pytest.raises(ValueError, match="chapters_between_checkpoints must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_max_revision_iterations(self):
        """Should raise ValueError for max_revision_iterations out of range."""
        settings = Settings()
        settings.max_revision_iterations = 15  # Too high
        with pytest.raises(ValueError, match="max_revision_iterations must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_learning_autonomy(self):
        """Should raise ValueError for invalid learning_autonomy."""
        settings = Settings()
        settings.learning_autonomy = "invalid_mode"
        with pytest.raises(ValueError, match="learning_autonomy must be one of"):
            settings.validate()

    def test_validate_raises_on_invalid_learning_trigger(self):
        """Should raise ValueError for invalid learning trigger."""
        settings = Settings()
        settings.learning_triggers = ["invalid_trigger"]
        with pytest.raises(ValueError, match="Invalid learning trigger"):
            settings.validate()

    def test_validate_raises_on_invalid_learning_periodic_interval(self):
        """Should raise ValueError for learning_periodic_interval out of range."""
        settings = Settings()
        settings.learning_periodic_interval = 50  # Too high
        with pytest.raises(ValueError, match="learning_periodic_interval must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_learning_confidence_threshold(self):
        """Should raise ValueError for learning_confidence_threshold out of range."""
        settings = Settings()
        settings.learning_confidence_threshold = 1.5  # Too high
        with pytest.raises(ValueError, match="learning_confidence_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_entity_version_retention_too_low(self):
        """Should raise ValueError for entity_version_retention below 1."""
        settings = Settings()
        settings.entity_version_retention = 0  # Too low
        with pytest.raises(ValueError, match="entity_version_retention must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_entity_version_retention_too_high(self):
        """Should raise ValueError for entity_version_retention above 100."""
        settings = Settings()
        settings.entity_version_retention = 101  # Too high
        with pytest.raises(ValueError, match="entity_version_retention must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_backup_verify_on_restore_type(self):
        """Should raise ValueError for backup_verify_on_restore that's not a boolean."""
        settings = Settings()
        settings.backup_verify_on_restore = "true"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="backup_verify_on_restore must be a boolean"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_timeout(self):
        """Should raise ValueError for ollama_timeout out of range."""
        settings = Settings()
        settings.ollama_timeout = 5  # Too low
        with pytest.raises(ValueError, match="ollama_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_subprocess_timeout(self):
        """Should raise ValueError for subprocess_timeout out of range."""
        settings = Settings()
        settings.subprocess_timeout = 2  # Too low
        with pytest.raises(ValueError, match="subprocess_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_non_bool_content_check_enabled(self):
        """Should raise ValueError for non-boolean content_check_enabled."""
        settings = Settings()
        settings.content_check_enabled = "true"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="content_check_enabled must be a boolean"):
            settings.validate()

    def test_validate_raises_on_non_bool_content_check_use_llm(self):
        """Should raise ValueError for non-boolean content_check_use_llm."""
        settings = Settings()
        settings.content_check_use_llm = 1  # type: ignore[assignment]
        with pytest.raises(ValueError, match="content_check_use_llm must be a boolean"):
            settings.validate()

    def test_validate_warns_on_use_llm_without_check_enabled(self, caplog):
        """Should log warning when use_llm is enabled but check is disabled."""
        import logging

        settings = Settings()
        settings.content_check_enabled = False
        settings.content_check_use_llm = True

        with caplog.at_level(logging.WARNING):
            settings.validate()

        assert "LLM checking will have no effect" in caplog.text

    def test_validate_raises_on_non_bool_relationship_validation_enabled(self):
        """Should raise ValueError for non-boolean relationship_validation_enabled."""
        settings = Settings()
        settings.relationship_validation_enabled = "true"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="relationship_validation_enabled must be a boolean"):
            settings.validate()

    def test_validate_raises_on_non_bool_orphan_detection_enabled(self):
        """Should raise ValueError for non-boolean orphan_detection_enabled."""
        settings = Settings()
        settings.orphan_detection_enabled = 1  # type: ignore[assignment]
        with pytest.raises(ValueError, match="orphan_detection_enabled must be a boolean"):
            settings.validate()

    def test_validate_raises_on_non_bool_circular_detection_enabled(self):
        """Should raise ValueError for non-boolean circular_detection_enabled."""
        settings = Settings()
        settings.circular_detection_enabled = "yes"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="circular_detection_enabled must be a boolean"):
            settings.validate()

    def test_validate_raises_on_fuzzy_match_threshold_too_low(self):
        """Should raise ValueError for fuzzy_match_threshold below 0.5."""
        settings = Settings()
        settings.fuzzy_match_threshold = 0.3
        with pytest.raises(
            ValueError, match=r"fuzzy_match_threshold must be between 0\.5 and 1\.0"
        ):
            settings.validate()

    def test_validate_raises_on_fuzzy_match_threshold_too_high(self):
        """Should raise ValueError for fuzzy_match_threshold above 1.0."""
        settings = Settings()
        settings.fuzzy_match_threshold = 1.5
        with pytest.raises(
            ValueError, match=r"fuzzy_match_threshold must be between 0\.5 and 1\.0"
        ):
            settings.validate()

    def test_validate_raises_on_max_relationships_too_low(self):
        """Should raise ValueError for max_relationships_per_entity below 1."""
        settings = Settings()
        settings.max_relationships_per_entity = 0
        with pytest.raises(
            ValueError, match="max_relationships_per_entity must be between 1 and 50"
        ):
            settings.validate()

    def test_validate_raises_on_max_relationships_too_high(self):
        """Should raise ValueError for max_relationships_per_entity above 50."""
        settings = Settings()
        settings.max_relationships_per_entity = 100
        with pytest.raises(
            ValueError, match="max_relationships_per_entity must be between 1 and 50"
        ):
            settings.validate()

    def test_validate_raises_on_circular_relationship_types_not_list(self):
        """Should raise ValueError for circular_relationship_types that's not a list."""
        settings = Settings()
        settings.circular_relationship_types = "owns,reports_to"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="circular_relationship_types must be a list"):
            settings.validate()

    def test_validate_raises_on_circular_relationship_types_contains_non_strings(self):
        """Should raise ValueError for circular_relationship_types containing non-strings."""
        settings = Settings()
        settings.circular_relationship_types = ["owns", 123, "reports_to"]  # type: ignore[list-item]
        with pytest.raises(
            ValueError, match="circular_relationship_types must contain only strings"
        ):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_not_dict(self):
        """Should raise ValueError for relationship_minimums that's not a dict."""
        settings = Settings()
        settings.relationship_minimums = [("character", {"protagonist": 5})]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="relationship_minimums must be a dict"):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_key_not_string(self):
        """Should raise ValueError for relationship_minimums with non-string keys."""
        settings = Settings()
        settings.relationship_minimums = {123: {"protagonist": 5}}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="relationship_minimums keys must be strings"):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_value_not_dict(self):
        """Should raise ValueError for relationship_minimums with non-dict values."""
        settings = Settings()
        settings.relationship_minimums = {"character": [("protagonist", 5)]}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match=r"relationship_minimums\[character\] must be a dict"):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_role_not_string(self):
        """Should raise ValueError for relationship_minimums with non-string role keys."""
        settings = Settings()
        settings.relationship_minimums = {"character": {123: 5}}  # type: ignore[dict-item]
        with pytest.raises(
            ValueError, match=r"relationship_minimums\[character\] keys must be strings"
        ):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_count_not_int(self):
        """Should raise ValueError for relationship_minimums with non-int min_count."""
        settings = Settings()
        settings.relationship_minimums = {"character": {"protagonist": "five"}}  # type: ignore[dict-item]
        with pytest.raises(
            ValueError,
            match=r"relationship_minimums\[character\]\[protagonist\] must be a non-negative integer",
        ):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_count_negative(self):
        """Should raise ValueError for relationship_minimums with negative min_count."""
        settings = Settings()
        settings.relationship_minimums = {"character": {"protagonist": -1}}
        with pytest.raises(
            ValueError,
            match=r"relationship_minimums\[character\]\[protagonist\] must be a non-negative integer",
        ):
            settings.validate()

    def test_validate_raises_on_minimum_exceeds_max_relationships(self):
        """Should raise ValueError when minimum exceeds max_relationships_per_entity."""
        settings = Settings()
        settings.max_relationships_per_entity = 5
        settings.relationship_minimums = {"character": {"protagonist": 10}}  # Exceeds max of 5
        with pytest.raises(
            ValueError,
            match=r"relationship_minimums\[character\]\[protagonist\] \(10\) exceeds "
            r"max_relationships_per_entity \(5\)",
        ):
            settings.validate()

    def test_validate_raises_on_non_bool_generate_calendar_on_world_build(self):
        """Should raise ValueError for non-boolean generate_calendar_on_world_build."""
        settings = Settings()
        settings.generate_calendar_on_world_build = "true"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="generate_calendar_on_world_build must be a boolean"):
            settings.validate()

    def test_validate_raises_on_non_bool_validate_temporal_consistency(self):
        """Should raise ValueError for non-boolean validate_temporal_consistency."""
        settings = Settings()
        settings.validate_temporal_consistency = 1  # type: ignore[assignment]
        with pytest.raises(ValueError, match="validate_temporal_consistency must be a boolean"):
            settings.validate()


class TestSettingsSaveLoad:
    """Tests for Settings save and load methods."""

    def test_save_creates_file(self, tmp_path, monkeypatch):
        """Test save creates settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings()
        settings.save()

        assert settings_file.exists()
        with open(settings_file) as f:
            data = json.load(f)
        assert data["ollama_url"] == "http://localhost:11434"

    def test_load_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        """Test load returns defaults when settings file doesn't exist."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings.load()

        assert settings.ollama_url == "http://localhost:11434"
        # File should be created
        assert settings_file.exists()

    def test_load_reads_existing_file(self, tmp_path, monkeypatch):
        """Test load reads existing settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a settings file with custom value
        with open(settings_file, "w") as f:
            json.dump({"ollama_url": "http://custom:11434", "context_size": 16384}, f)

        settings = Settings.load()

        assert settings.ollama_url == "http://custom:11434"
        assert settings.context_size == 16384

    def test_load_handles_corrupted_json(self, tmp_path, monkeypatch):
        """Test load handles corrupted JSON file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a corrupted JSON file
        settings_file.write_text("not valid json {{{")

        settings = Settings.load()

        # Should return defaults
        assert settings.ollama_url == "http://localhost:11434"

    def test_load_handles_unknown_fields(self, tmp_path, monkeypatch):
        """Test load handles unknown fields in settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a settings file with unknown field
        data = {"ollama_url": "http://localhost:11434", "unknown_field": "value"}
        with open(settings_file, "w") as f:
            json.dump(data, f)

        # Should use partial recovery
        settings = Settings.load()
        assert settings.ollama_url == "http://localhost:11434"

    def test_load_handles_invalid_values(self, tmp_path, monkeypatch):
        """Test load handles invalid setting values."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a settings file with invalid values
        data = {"context_size": 500}  # Too low
        with open(settings_file, "w") as f:
            json.dump(data, f)

        # Should use partial recovery
        settings = Settings.load()
        # Either uses default context_size or recovers partially
        assert settings.context_size >= 1024

    def test_load_caches_settings(self, tmp_path, monkeypatch):
        """Test load() returns cached instance on subsequent calls."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # First load creates a new instance
        settings1 = Settings.load()
        # Second load returns the same instance (cached)
        settings2 = Settings.load()

        assert settings1 is settings2

    def test_load_with_use_cache_false_reloads(self, tmp_path, monkeypatch):
        """Test load(use_cache=False) forces reload from disk."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # First load
        settings1 = Settings.load()

        # Modify the cached instance
        settings1.world_quality_enabled = not settings1.world_quality_enabled

        # Load with use_cache=False should create a new instance
        settings2 = Settings.load(use_cache=False)

        assert settings1 is not settings2
        # New instance has default value
        assert settings2.world_quality_enabled != settings1.world_quality_enabled

    def test_clear_cache_clears_cached_instance(self, tmp_path, monkeypatch):
        """Test clear_cache() clears the cached instance."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # First load creates a cached instance
        settings1 = Settings.load()

        # Clear cache
        Settings.clear_cache()

        # Next load creates a new instance
        settings2 = Settings.load()

        assert settings1 is not settings2


class TestRecoverPartialSettings:
    """Tests for _recover_partial_settings method."""

    def test_recovers_valid_fields_and_logs(self, tmp_path, monkeypatch, caplog):
        """Test recovers valid fields and logs the recovery."""
        import logging

        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Data with valid fields
        data = {
            "ollama_url": "http://custom:11434",  # Valid
            "world_quality_enabled": False,  # Valid
        }

        with caplog.at_level(logging.INFO):
            settings = Settings._recover_partial_settings(data)

        assert settings.ollama_url == "http://custom:11434"
        assert settings.world_quality_enabled is False
        assert "Recovered" in caplog.text

    def test_falls_back_to_defaults_when_recovery_fails_validation(self, tmp_path, monkeypatch):
        """Test falls back to complete defaults when recovered settings fail validation."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create data that looks valid but fails overall validation
        # Use agent_temperatures with unknown agent which fails validation
        data = {
            "ollama_url": "http://localhost:11434",
            "agent_temperatures": {"unknown_agent": 0.5},  # Unknown agent fails validation
        }

        settings = Settings._recover_partial_settings(data)

        # Should fall back to defaults since recovered settings fail validation
        assert "unknown_agent" not in settings.agent_temperatures


class TestSettingsGetModelForAgent:
    """Tests for get_model_for_agent method."""

    def test_returns_default_model_when_per_agent_disabled(self):
        """Test returns default model when per-agent models disabled."""
        settings = Settings()
        settings.use_per_agent_models = False
        settings.default_model = "my-default:8b"

        result = settings.get_model_for_agent("writer")

        assert result == "my-default:8b"

    def test_returns_specific_model_when_set(self):
        """Test returns specific model when set for agent."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models["writer"] = "writer-model:7b"

        result = settings.get_model_for_agent("writer")

        assert result == "writer-model:7b"

    def test_auto_selects_tagged_model_for_writer(self, monkeypatch):
        """Test auto-selects tagged model for writer role."""
        # Mock installed models with sizes - includes a model tagged for writer
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0": 13.0,  # Tagged for writer
                "huihui_ai/dolphin3-abliterated:8b": 5.0,  # Tagged for interviewer
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        result = settings.get_model_for_agent("writer", available_vram=24)

        # Should select Celeste (tagged for writer)
        assert result == "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0"

    def test_selects_architect_model(self, monkeypatch):
        """Test selects high-reasoning model for architect role."""
        # Mock installed models with sizes - includes models tagged for architect
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "huihui_ai/qwen3-abliterated:30b": 18.0,  # Tagged for architect
                "huihui_ai/dolphin3-abliterated:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        # With 24GB VRAM, should prefer Qwen3-30B (tagged for architect)
        result = settings.get_model_for_agent("architect", available_vram=24)

        assert result == "huihui_ai/qwen3-abliterated:30b"

    def test_raises_error_when_no_tagged_model_available(self, monkeypatch):
        """Test raises error when no installed model has the required tag."""
        # Mock installed models with no matching tags
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:30b": 18.0,
                "custom-medium:12b": 10.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        with pytest.raises(ValueError, match="No model tagged for role 'architect'"):
            settings.get_model_for_agent("architect", available_vram=24)

    def test_selects_custom_tagged_model(self, monkeypatch):
        """Test selects model with custom tags when configured."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "my-custom-model:7b": 5.0,
                "another-model:12b": 10.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}
        # Add custom tag for the model
        settings.custom_model_tags = {"my-custom-model:7b": ["architect"]}

        result = settings.get_model_for_agent("architect", available_vram=24)

        assert result == "my-custom-model:7b"

    def test_validator_prefers_tiny_models(self, monkeypatch):
        """Test validator role prefers tiny models."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "qwen3:0.6b": 0.5,  # Tagged for validator, tiny tier
                "huihui_ai/dolphin3-abliterated:8b": 5.0,  # Small tier
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "auto"}

        result = settings.get_model_for_agent("validator", available_vram=24)

        # Should select the tiny model tagged for validator
        assert result == "qwen3:0.6b"

    def test_returns_default_when_no_models_installed(self, monkeypatch, caplog):
        """Test returns default model when no models installed."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {},
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        # Should return first recommended model and log warning
        result = settings.get_model_for_agent("writer", available_vram=24)
        assert result == "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0"  # First in RECOMMENDED_MODELS
        assert "No models installed in Ollama" in caplog.text

    def test_selects_smallest_tagged_when_nothing_fits_vram(self, monkeypatch):
        """Test selects smallest tagged model as last resort when nothing fits VRAM."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "large-model:30b": 18.0,  # Needs 21.6GB
                "medium-model:12b": 10.0,  # Needs 12GB
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}
        # Tag both models for writer role
        settings.custom_model_tags = {
            "large-model:30b": ["writer"],
            "medium-model:12b": ["writer"],
        }

        # Only 8GB VRAM - nothing fits, but should select smallest tagged model
        result = settings.get_model_for_agent("writer", available_vram=8)

        # Should select smallest tagged model as last resort
        assert result == "medium-model:12b"

    def test_raises_error_for_unknown_role(self, monkeypatch):
        """Test raises error for unknown role with no tagged models."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-medium:12b": 10.0,
                "custom-small:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"unknown_role": "auto"}

        # Unknown role has no tagged models - should raise error
        with pytest.raises(ValueError, match="No model tagged for role 'unknown_role'"):
            settings.get_model_for_agent("unknown_role", available_vram=24)


class TestGetAvailableVram:
    """Tests for get_available_vram function."""

    @patch("src.settings._utils.subprocess.run")
    def test_returns_vram_from_nvidia_smi(self, mock_run):
        """Test returns VRAM from nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n"
        mock_run.return_value = mock_result

        vram = get_available_vram()

        # Should parse the MB value and convert to GB (8192 MB = 8 GB)
        assert vram == 8

    @patch("src.settings._utils.subprocess.run")
    def test_returns_default_on_file_not_found(self, mock_run):
        """Test returns default when nvidia-smi not found."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        vram = get_available_vram()

        assert vram == 8  # Default

    @patch("src.settings._utils.subprocess.run")
    def test_returns_default_on_timeout(self, mock_run):
        """Test returns default on subprocess timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        vram = get_available_vram()

        assert vram == 8  # Default

    @patch("src.settings._utils.subprocess.run")
    def test_returns_default_on_parse_error(self, mock_run):
        """Test returns default when output can't be parsed."""
        mock_result = MagicMock()
        mock_result.stdout = "invalid\n"  # Not a number
        mock_run.return_value = mock_result

        vram = get_available_vram()

        assert vram == 8  # Default

    @patch("src.settings._utils.subprocess.run")
    def test_returns_default_on_os_error(self, mock_run):
        """Test returns default on OSError."""
        mock_run.side_effect = OSError("Permission denied")

        vram = get_available_vram()

        assert vram == 8  # Default


class TestGetInstalledModels:
    """Tests for get_installed_models function."""

    @patch("src.settings._utils.subprocess.run")
    def test_returns_model_list(self, mock_run):
        """Test returns list of installed models."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME\nmodel1:latest\nmodel2:7b\n"
        mock_run.return_value = mock_result

        models = get_installed_models()

        assert "model1:latest" in models
        assert "model2:7b" in models

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_file_not_found(self, mock_run):
        """Test returns empty list when ollama not found."""
        mock_run.side_effect = FileNotFoundError("ollama not found")

        models = get_installed_models()

        assert models == []

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_timeout(self, mock_run):
        """Test returns empty list on timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)

        models = get_installed_models()

        assert models == []

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_os_error(self, mock_run):
        """Test returns empty list on OSError."""
        mock_run.side_effect = OSError("Permission denied")

        models = get_installed_models()

        assert models == []


class TestGetInstalledModelsWithSizes:
    """Tests for get_installed_models_with_sizes function."""

    @patch("src.settings._utils.subprocess.run")
    def test_returns_model_sizes(self, mock_run):
        """Test returns dict of models with sizes."""

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "NAME              SIZE\nmodel1:latest     8.5 GB\nmodel2:7b     4.0 GB\n"
        )
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        assert "model1:latest" in models
        assert models["model1:latest"] == 8.5
        assert models["model2:7b"] == 4.0

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_file_not_found(self, mock_run):
        """Test returns empty dict when ollama not found."""
        mock_run.side_effect = FileNotFoundError("ollama not found")

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_timeout(self, mock_run):
        """Test returns empty dict on timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_os_error(self, mock_run):
        """Test returns empty dict on OSError."""
        mock_run.side_effect = OSError("Permission denied")

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("src.settings._utils.subprocess.run")
    def test_returns_empty_on_nonzero_exit_code(self, mock_run):
        """Test returns empty dict when ollama list returns non-zero exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Error: something went wrong"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("src.settings._utils.subprocess.run")
    def test_parses_mb_sizes(self, mock_run):
        """Test parses MB size values and converts to GB."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME              SIZE\nsmall-model:1b    512 MB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        assert "small-model:1b" in models
        # 512 MB = 0.512 GB (using decimal: 1 GB = 1000 MB)
        assert models["small-model:1b"] == 0.512

    @patch("src.settings._utils.subprocess.run")
    def test_parses_combined_size_format(self, mock_run):
        """Test parses combined format like '4.1GB' without space."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME              SIZE\nmodel:tag     4.1GB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        assert "model:tag" in models
        assert models["model:tag"] == 4.1

    @patch("src.settings._utils.subprocess.run")
    def test_handles_invalid_gb_separate_format(self, mock_run):
        """Test handles invalid GB separate format gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # Invalid size format - not a number before GB
        mock_result.stdout = "NAME              SIZE\nmodel:tag     invalid GB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        # Should still return the model but with default size of 0.0
        assert "model:tag" in models
        assert models["model:tag"] == 0.0

    @patch("src.settings._utils.subprocess.run")
    def test_handles_invalid_mb_separate_format(self, mock_run):
        """Test handles invalid MB separate format gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # Invalid size format - not a number before MB
        mock_result.stdout = "NAME              SIZE\nmodel:tag     invalid MB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        # Should still return the model but with default size of 0.0
        assert "model:tag" in models
        assert models["model:tag"] == 0.0

    @patch("src.settings._utils.subprocess.run")
    def test_handles_invalid_combined_gb_format(self, mock_run):
        """Test handles invalid combined GB format (like 'xyzGB') gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # Invalid combined format - not a valid number
        mock_result.stdout = "NAME              SIZE\nmodel:tag     invalidGB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        # Should still return the model but with default size of 0.0
        assert "model:tag" in models
        assert models["model:tag"] == 0.0

    @patch("src.settings._utils.subprocess.run")
    def test_handles_invalid_combined_mb_format(self, mock_run):
        """Test handles invalid combined MB format (like 'xyzMB') gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # Invalid combined format - not a valid number
        mock_result.stdout = "NAME              SIZE\nmodel:tag     invalidMB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        # Should still return the model but with default size of 0.0
        assert "model:tag" in models
        assert models["model:tag"] == 0.0


class TestNewSettingsValidation:
    """Tests for newly added settings validation (PR #110)."""

    # --- Ollama client timeouts ---

    def test_validate_raises_on_invalid_ollama_health_check_timeout(self):
        """Should raise ValueError for ollama_health_check_timeout out of range."""
        settings = Settings()
        settings.ollama_health_check_timeout = 3.0  # Too low (min 5.0)
        with pytest.raises(ValueError, match="ollama_health_check_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_list_models_timeout(self):
        """Should raise ValueError for ollama_list_models_timeout out of range."""
        settings = Settings()
        settings.ollama_list_models_timeout = 400.0  # Too high (max 300.0)
        with pytest.raises(ValueError, match="ollama_list_models_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_pull_model_timeout(self):
        """Should raise ValueError for ollama_pull_model_timeout out of range."""
        settings = Settings()
        settings.ollama_pull_model_timeout = 30.0  # Too low (min 60.0)
        with pytest.raises(ValueError, match="ollama_pull_model_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_delete_model_timeout(self):
        """Should raise ValueError for ollama_delete_model_timeout out of range."""
        settings = Settings()
        settings.ollama_delete_model_timeout = 2.0  # Too low (min 5.0)
        with pytest.raises(ValueError, match="ollama_delete_model_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_check_update_timeout(self):
        """Should raise ValueError for ollama_check_update_timeout out of range."""
        settings = Settings()
        settings.ollama_check_update_timeout = 400.0  # Too high (max 300.0)
        with pytest.raises(ValueError, match="ollama_check_update_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_generate_timeout(self):
        """Should raise ValueError for ollama_generate_timeout out of range."""
        settings = Settings()
        settings.ollama_generate_timeout = 700.0  # Too high (max 600.0)
        with pytest.raises(ValueError, match="ollama_generate_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_ollama_capability_check_timeout(self):
        """Should raise ValueError for ollama_capability_check_timeout out of range."""
        settings = Settings()
        settings.ollama_capability_check_timeout = 20.0  # Too low (min 30.0)
        with pytest.raises(ValueError, match="ollama_capability_check_timeout must be between"):
            settings.validate()

    # --- Retry configuration ---

    def test_validate_raises_on_invalid_llm_retry_delay_too_low(self):
        """Should raise ValueError for llm_retry_delay too low."""
        settings = Settings()
        settings.llm_retry_delay = 0.05  # Too low (min 0.1)
        with pytest.raises(ValueError, match="llm_retry_delay must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_llm_retry_delay_too_high(self):
        """Should raise ValueError for llm_retry_delay too high."""
        settings = Settings()
        settings.llm_retry_delay = 70.0  # Too high (max 60.0)
        with pytest.raises(ValueError, match="llm_retry_delay must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_llm_retry_backoff(self):
        """Should raise ValueError for llm_retry_backoff out of range."""
        settings = Settings()
        settings.llm_retry_backoff = 15.0  # Too high (max 10.0)
        with pytest.raises(ValueError, match="llm_retry_backoff must be between"):
            settings.validate()

    # --- Verification delays ---

    def test_validate_raises_on_invalid_model_verification_sleep(self):
        """Should raise ValueError for model_verification_sleep out of range."""
        settings = Settings()
        settings.model_verification_sleep = 0.001  # Too low (min 0.01)
        with pytest.raises(ValueError, match="model_verification_sleep must be between"):
            settings.validate()

    # --- Validation thresholds ---

    def test_validate_raises_on_invalid_validator_cjk_char_threshold(self):
        """Should raise ValueError for validator_cjk_char_threshold out of range."""
        settings = Settings()
        settings.validator_cjk_char_threshold = -1  # Below 0
        with pytest.raises(ValueError, match="validator_cjk_char_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_validator_printable_ratio(self):
        """Should raise ValueError for validator_printable_ratio out of range."""
        settings = Settings()
        settings.validator_printable_ratio = 1.5  # Above 1.0
        with pytest.raises(ValueError, match="validator_printable_ratio must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_validator_ai_check_min_length(self):
        """Should raise ValueError for validator_ai_check_min_length out of range."""
        settings = Settings()
        settings.validator_ai_check_min_length = 20000  # Too high (max 10000)
        with pytest.raises(ValueError, match="validator_ai_check_min_length must be between"):
            settings.validate()

    # --- Outline generation ---

    def test_validate_raises_on_invalid_outline_variations_min(self):
        """Should raise ValueError for outline_variations_min out of range."""
        settings = Settings()
        settings.outline_variations_min = 0  # Below 1
        with pytest.raises(ValueError, match="outline_variations_min must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_outline_variations_max(self):
        """Should raise ValueError for outline_variations_max out of range."""
        settings = Settings()
        settings.outline_variations_max = 25  # Above 20
        with pytest.raises(ValueError, match="outline_variations_max must be between"):
            settings.validate()

    def test_validate_raises_on_outline_variations_min_exceeds_max(self):
        """Should raise ValueError when outline_variations_min exceeds max."""
        settings = Settings()
        settings.outline_variations_min = 5
        settings.outline_variations_max = 3
        with pytest.raises(ValueError, match=r"outline_variations_min.*cannot exceed"):
            settings.validate()

    # --- Import thresholds ---

    def test_validate_raises_on_invalid_import_confidence_threshold(self):
        """Should raise ValueError for import_confidence_threshold out of range."""
        settings = Settings()
        settings.import_confidence_threshold = 1.5  # Above 1.0
        with pytest.raises(ValueError, match="import_confidence_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_import_default_confidence(self):
        """Should raise ValueError for import_default_confidence out of range."""
        settings = Settings()
        settings.import_default_confidence = -0.1  # Below 0.0
        with pytest.raises(ValueError, match="import_default_confidence must be between"):
            settings.validate()

    # --- World/plot generation limits ---

    def test_validate_raises_on_invalid_world_description_summary_length(self):
        """Should raise ValueError for world_description_summary_length out of range."""
        settings = Settings()
        settings.world_description_summary_length = 5  # Below 10
        with pytest.raises(ValueError, match="world_description_summary_length must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_event_sentence_min_length(self):
        """Should raise ValueError for event_sentence_min_length out of range."""
        settings = Settings()
        settings.event_sentence_min_length = 600  # Above 500
        with pytest.raises(ValueError, match="event_sentence_min_length must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_event_sentence_max_length(self):
        """Should raise ValueError for event_sentence_max_length out of range."""
        settings = Settings()
        settings.event_sentence_max_length = 6000  # Above 5000
        with pytest.raises(ValueError, match="event_sentence_max_length must be between"):
            settings.validate()

    def test_validate_raises_on_event_sentence_min_exceeds_max(self):
        """Should raise ValueError when event_sentence_min_length exceeds max."""
        settings = Settings()
        settings.event_sentence_min_length = 150
        settings.event_sentence_max_length = 100
        with pytest.raises(ValueError, match=r"event_sentence_min_length.*cannot exceed"):
            settings.validate()

    # --- User rating bounds ---

    def test_validate_raises_on_invalid_user_rating_min(self):
        """Should raise ValueError for user_rating_min out of range."""
        settings = Settings()
        settings.user_rating_min = 0  # Below 1
        with pytest.raises(ValueError, match="user_rating_min must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_user_rating_max(self):
        """Should raise ValueError for user_rating_max out of range."""
        settings = Settings()
        settings.user_rating_max = 15  # Above 10
        with pytest.raises(ValueError, match="user_rating_max must be between"):
            settings.validate()

    def test_validate_raises_on_user_rating_min_exceeds_max(self):
        """Should raise ValueError when user_rating_min exceeds max."""
        settings = Settings()
        settings.user_rating_min = 5
        settings.user_rating_max = 3
        with pytest.raises(ValueError, match=r"user_rating_min.*cannot exceed"):
            settings.validate()

    # --- Model download threshold ---

    def test_validate_raises_on_invalid_model_download_threshold(self):
        """Should raise ValueError for model_download_threshold out of range."""
        settings = Settings()
        settings.model_download_threshold = 1.5  # Above 1.0
        with pytest.raises(ValueError, match="model_download_threshold must be between"):
            settings.validate()

    # --- Story chapter counts ---

    def test_validate_raises_on_invalid_chapters_short_story(self):
        """Should raise ValueError for chapters_short_story out of range."""
        settings = Settings()
        settings.chapters_short_story = 0  # Below 1
        with pytest.raises(ValueError, match="chapters_short_story must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_chapters_novella(self):
        """Should raise ValueError for chapters_novella out of range."""
        settings = Settings()
        settings.chapters_novella = 150  # Above 100
        with pytest.raises(ValueError, match="chapters_novella must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_chapters_novel(self):
        """Should raise ValueError for chapters_novel out of range."""
        settings = Settings()
        settings.chapters_novel = 250  # Above 200
        with pytest.raises(ValueError, match="chapters_novel must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_chapters_default(self):
        """Should raise ValueError for chapters_default out of range."""
        settings = Settings()
        settings.chapters_default = 0  # Below 1
        with pytest.raises(ValueError, match="chapters_default must be between"):
            settings.validate()

    # --- Import temperatures ---

    def test_validate_raises_on_invalid_temp_import_extraction(self):
        """Should raise ValueError for temp_import_extraction out of range."""
        settings = Settings()
        settings.temp_import_extraction = 3.0  # Above 2.0
        with pytest.raises(ValueError, match="temp_import_extraction must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_temp_interviewer_override(self):
        """Should raise ValueError for temp_interviewer_override out of range."""
        settings = Settings()
        settings.temp_interviewer_override = -0.5  # Below 0.0
        with pytest.raises(ValueError, match="temp_interviewer_override must be between"):
            settings.validate()

    # --- Token multipliers ---

    def test_validate_raises_on_invalid_import_character_token_multiplier(self):
        """Should raise ValueError for import_character_token_multiplier out of range."""
        settings = Settings()
        settings.import_character_token_multiplier = 25  # Above 20
        with pytest.raises(ValueError, match="import_character_token_multiplier must be between"):
            settings.validate()

    # --- Boundary value tests ---

    def test_validate_passes_with_boundary_values(self):
        """Should not raise for valid boundary values."""
        settings = Settings()
        # Set boundary values
        settings.ollama_health_check_timeout = 5.0  # Min boundary
        settings.llm_retry_delay = 60.0  # Max boundary
        settings.validator_printable_ratio = 0.0  # Min boundary
        settings.validator_printable_ratio = 1.0  # Max boundary
        settings.outline_variations_min = 1  # Min boundary
        settings.outline_variations_max = 20  # Max boundary
        settings.user_rating_min = 1  # Min boundary
        settings.user_rating_max = 10  # Max boundary
        settings.chapters_novel = 200  # Max boundary
        settings.validate()  # Should not raise


class TestMissingValidationCoverage:
    """Tests for missing validation coverage lines."""

    # --- Task-specific temperature validation (line 492) ---

    def test_validate_raises_on_invalid_temp_brief_extraction(self):
        """Should raise ValueError for temp_brief_extraction out of range."""
        settings = Settings()
        settings.temp_brief_extraction = 3.0  # Above 2.0
        with pytest.raises(ValueError, match="temp_brief_extraction must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_temp_edit_suggestions(self):
        """Should raise ValueError for temp_edit_suggestions out of range."""
        settings = Settings()
        settings.temp_edit_suggestions = -0.5  # Below 0.0
        with pytest.raises(ValueError, match="temp_edit_suggestions must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_temp_plot_checking(self):
        """Should raise ValueError for temp_plot_checking out of range."""
        settings = Settings()
        settings.temp_plot_checking = 2.5  # Above 2.0
        with pytest.raises(ValueError, match="temp_plot_checking must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_temp_capability_check(self):
        """Should raise ValueError for temp_capability_check out of range."""
        settings = Settings()
        settings.temp_capability_check = 5.0  # Above 2.0
        with pytest.raises(ValueError, match="temp_capability_check must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_temp_model_evaluation(self):
        """Should raise ValueError for temp_model_evaluation out of range."""
        settings = Settings()
        settings.temp_model_evaluation = -1.0  # Below 0.0
        with pytest.raises(ValueError, match="temp_model_evaluation must be between"):
            settings.validate()

    # --- World quality settings (lines 531, 537, 548) ---

    def test_validate_raises_on_invalid_world_quality_max_iterations(self):
        """Should raise ValueError for world_quality_max_iterations out of range."""
        settings = Settings()
        settings.world_quality_max_iterations = 15  # Above 10
        with pytest.raises(ValueError, match="world_quality_max_iterations must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_quality_threshold(self):
        """Should raise ValueError for world_quality_threshold out of range."""
        settings = Settings()
        settings.world_quality_threshold = 15.0  # Above 10.0
        with pytest.raises(ValueError, match="world_quality_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_quality_creator_temp(self):
        """Should raise ValueError for world_quality_creator_temp out of range."""
        settings = Settings()
        settings.world_quality_creator_temp = 3.0  # Above 2.0
        with pytest.raises(ValueError, match="world_quality_creator_temp must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_quality_judge_temp(self):
        """Should raise ValueError for world_quality_judge_temp out of range."""
        settings = Settings()
        settings.world_quality_judge_temp = -0.5  # Below 0.0
        with pytest.raises(ValueError, match="world_quality_judge_temp must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_quality_refinement_temp(self):
        """Should raise ValueError for world_quality_refinement_temp out of range."""
        settings = Settings()
        settings.world_quality_refinement_temp = 2.5  # Above 2.0
        with pytest.raises(ValueError, match="world_quality_refinement_temp must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_quality_early_stopping_patience_low(self):
        """Should raise ValueError for world_quality_early_stopping_patience below 1."""
        settings = Settings()
        settings.world_quality_early_stopping_patience = 0  # Below 1
        with pytest.raises(
            ValueError, match="world_quality_early_stopping_patience must be between"
        ):
            settings.validate()

    def test_validate_raises_on_invalid_world_quality_early_stopping_patience_high(self):
        """Should raise ValueError for world_quality_early_stopping_patience above 10."""
        settings = Settings()
        settings.world_quality_early_stopping_patience = 11  # Above 10
        with pytest.raises(
            ValueError, match="world_quality_early_stopping_patience must be between"
        ):
            settings.validate()

    # --- World gen entity min/max validation (lines 561, 565, 569) ---

    def test_validate_raises_on_invalid_world_gen_characters_min(self):
        """Should raise ValueError for world_gen_characters_min out of range."""
        settings = Settings()
        settings.world_gen_characters_min = 25  # Above 20
        with pytest.raises(ValueError, match="world_gen_characters_min must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_gen_characters_max(self):
        """Should raise ValueError for world_gen_characters_max out of range."""
        settings = Settings()
        settings.world_gen_characters_max = 60  # Above 50
        with pytest.raises(ValueError, match="world_gen_characters_max must be between"):
            settings.validate()

    def test_validate_raises_on_world_gen_characters_min_exceeds_max(self):
        """Should raise ValueError when world_gen_characters_min exceeds max."""
        settings = Settings()
        settings.world_gen_characters_min = 10
        settings.world_gen_characters_max = 5
        with pytest.raises(ValueError, match=r"world_gen_characters_min.*cannot exceed"):
            settings.validate()

    def test_validate_raises_on_invalid_world_gen_locations_min(self):
        """Should raise ValueError for world_gen_locations_min out of range."""
        settings = Settings()
        settings.world_gen_locations_min = -1  # Below 0
        with pytest.raises(ValueError, match="world_gen_locations_min must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_world_gen_locations_max(self):
        """Should raise ValueError for world_gen_locations_max out of range."""
        settings = Settings()
        settings.world_gen_locations_max = 0  # Below 1
        with pytest.raises(ValueError, match="world_gen_locations_max must be between"):
            settings.validate()

    # --- LLM token settings validation (line 598) ---

    def test_validate_raises_on_invalid_llm_tokens_character_create(self):
        """Should raise ValueError for llm_tokens_character_create out of range."""
        settings = Settings()
        settings.llm_tokens_character_create = 5  # Below 10
        with pytest.raises(ValueError, match="llm_tokens_character_create must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_llm_tokens_mini_description(self):
        """Should raise ValueError for llm_tokens_mini_description out of range."""
        settings = Settings()
        settings.llm_tokens_mini_description = 5000  # Above 4096
        with pytest.raises(ValueError, match="llm_tokens_mini_description must be between"):
            settings.validate()

    # --- Entity extraction limits validation (line 607) ---

    def test_validate_raises_on_invalid_entity_extract_locations_max(self):
        """Should raise ValueError for entity_extract_locations_max out of range."""
        settings = Settings()
        settings.entity_extract_locations_max = 0  # Below 1
        with pytest.raises(ValueError, match="entity_extract_locations_max must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_entity_extract_items_max(self):
        """Should raise ValueError for entity_extract_items_max out of range."""
        settings = Settings()
        settings.entity_extract_items_max = 150  # Above 100
        with pytest.raises(ValueError, match="entity_extract_items_max must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_entity_extract_events_max(self):
        """Should raise ValueError for entity_extract_events_max out of range."""
        settings = Settings()
        settings.entity_extract_events_max = 200  # Above 100
        with pytest.raises(ValueError, match="entity_extract_events_max must be between"):
            settings.validate()

    # --- Mini description settings validation (lines 611, 616) ---

    def test_validate_raises_on_invalid_mini_description_words_max(self):
        """Should raise ValueError for mini_description_words_max out of range."""
        settings = Settings()
        settings.mini_description_words_max = 3  # Below 5
        with pytest.raises(ValueError, match="mini_description_words_max must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_mini_description_temperature(self):
        """Should raise ValueError for mini_description_temperature out of range."""
        settings = Settings()
        settings.mini_description_temperature = 3.0  # Above 2.0
        with pytest.raises(ValueError, match="mini_description_temperature must be between"):
            settings.validate()

    # --- Workflow limits validation (lines 623, 628) ---

    def test_validate_raises_on_invalid_orchestrator_cache_size(self):
        """Should raise ValueError for orchestrator_cache_size out of range."""
        settings = Settings()
        settings.orchestrator_cache_size = 0  # Below 1
        with pytest.raises(ValueError, match="orchestrator_cache_size must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_workflow_max_events(self):
        """Should raise ValueError for workflow_max_events out of range."""
        settings = Settings()
        settings.workflow_max_events = 5  # Below 10
        with pytest.raises(ValueError, match="workflow_max_events must be between"):
            settings.validate()

    # --- LLM request limits validation (lines 634, 639) ---

    def test_validate_raises_on_invalid_llm_max_concurrent_requests(self):
        """Should raise ValueError for llm_max_concurrent_requests out of range."""
        settings = Settings()
        settings.llm_max_concurrent_requests = 0  # Below 1
        with pytest.raises(ValueError, match="llm_max_concurrent_requests must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_llm_max_retries(self):
        """Should raise ValueError for llm_max_retries out of range."""
        settings = Settings()
        settings.llm_max_retries = 15  # Above 10
        with pytest.raises(ValueError, match="llm_max_retries must be between"):
            settings.validate()

    # --- Content truncation validation (line 645) ---

    def test_validate_raises_on_invalid_content_truncation_for_judgment(self):
        """Should raise ValueError for content_truncation_for_judgment out of range."""
        settings = Settings()
        settings.content_truncation_for_judgment = 100  # Below 500
        with pytest.raises(ValueError, match="content_truncation_for_judgment must be between"):
            settings.validate()

    # --- Judge consistency settings validation (lines 611, 617, 623, 630) ---

    def test_validate_raises_on_invalid_judge_multi_call_count_low(self):
        """Should raise ValueError for judge_multi_call_count below 2."""
        settings = Settings()
        settings.judge_multi_call_count = 1  # Below 2
        with pytest.raises(ValueError, match="judge_multi_call_count must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_judge_multi_call_count_high(self):
        """Should raise ValueError for judge_multi_call_count above 5."""
        settings = Settings()
        settings.judge_multi_call_count = 6  # Above 5
        with pytest.raises(ValueError, match="judge_multi_call_count must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_judge_confidence_threshold_low(self):
        """Should raise ValueError for judge_confidence_threshold below 0.0."""
        settings = Settings()
        settings.judge_confidence_threshold = -0.1  # Below 0.0
        with pytest.raises(ValueError, match="judge_confidence_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_judge_confidence_threshold_high(self):
        """Should raise ValueError for judge_confidence_threshold above 1.0."""
        settings = Settings()
        settings.judge_confidence_threshold = 1.5  # Above 1.0
        with pytest.raises(ValueError, match="judge_confidence_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_judge_outlier_std_threshold_low(self):
        """Should raise ValueError for judge_outlier_std_threshold below 1.0."""
        settings = Settings()
        settings.judge_outlier_std_threshold = 0.5  # Below 1.0
        with pytest.raises(ValueError, match="judge_outlier_std_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_judge_outlier_std_threshold_high(self):
        """Should raise ValueError for judge_outlier_std_threshold above 4.0."""
        settings = Settings()
        settings.judge_outlier_std_threshold = 5.0  # Above 4.0
        with pytest.raises(ValueError, match="judge_outlier_std_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_judge_outlier_strategy(self):
        """Should raise ValueError for invalid judge_outlier_strategy."""
        settings = Settings()
        settings.judge_outlier_strategy = "invalid"  # Not in valid list
        with pytest.raises(ValueError, match="judge_outlier_strategy must be one of"):
            settings.validate()

    def test_validate_raises_on_retry_judge_outlier_strategy(self):
        """Should raise ValueError for 'retry' outlier strategy (not implemented)."""
        settings = Settings()
        settings.judge_outlier_strategy = "retry"
        with pytest.raises(ValueError, match="judge_outlier_strategy must be one of"):
            settings.validate()


class TestValidatorModelSelection:
    """Tests for validator model selection."""

    def test_selects_tagged_validator_model(self, monkeypatch):
        """Test selects tiny model tagged for validator role."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {"qwen3:0.6b": 0.5},
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "auto"}

        result = settings.get_model_for_agent("validator", available_vram=24)

        assert result == "qwen3:0.6b"

    def test_validator_selects_tagged_model(self, monkeypatch):
        """Test validator selects model tagged for validator role."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-tiny:1b": 1.0,
                "custom-medium:12b": 10.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "auto"}
        # Tag the tiny model for validator
        settings.custom_model_tags = {"custom-tiny:1b": ["validator"]}

        result = settings.get_model_for_agent("validator", available_vram=24)

        # Validator selects the tagged model
        assert result == "custom-tiny:1b"


class TestBackupCorruptedSettings:
    """Tests for _backup_corrupted_settings method."""

    def test_creates_backup_file(self, tmp_path, monkeypatch):
        """Test creates backup of corrupted settings."""
        settings_file = tmp_path / "settings.json"
        backup_file = tmp_path / "settings.json.bak"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a "corrupted" settings file
        settings_file.write_text("corrupted content")

        Settings._backup_corrupted_settings()

        assert backup_file.exists()
        assert backup_file.read_text() == "corrupted content"

    def test_handles_missing_file(self, tmp_path, monkeypatch):
        """Test handles missing settings file gracefully."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Should not raise when file doesn't exist
        Settings._backup_corrupted_settings()

    def test_handles_backup_failure(self, tmp_path, monkeypatch):
        """Test handles OSError when backup fails."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a settings file
        settings_file.write_text("content")

        # Mock shutil.copy to raise OSError
        with patch("shutil.copy") as mock_copy:
            mock_copy.side_effect = OSError("Permission denied")
            # Should not raise, just log warning
            Settings._backup_corrupted_settings()


class TestWriterModelSelection:
    """Tests for writer model selection using tags."""

    def test_selects_tagged_creative_model_for_writer(self, monkeypatch):
        """Test selects creative writing specialist model tagged for writer role."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0": 13.0,  # Tagged for writer
                "other-model:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        result = settings.get_model_for_agent("writer", available_vram=24)

        # Should select the Celeste creative model (tagged for writer)
        assert result == "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0"

    def test_selects_alternative_tagged_writer_model(self, monkeypatch):
        """Test selects alternative tagged model when first isn't available."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit": 11.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        result = settings.get_model_for_agent("writer", available_vram=24)

        # Should select the Dark Champion model (tagged for writer)
        assert result == "TheAzazel/l3.2-moe-dark-champion-inst-18.4b-uncen-ablit"


class TestTagBasedModelSelection:
    """Tests for tag-based model selection.

    Models must be tagged for a specific role to be selected.
    Tags can come from RECOMMENDED_MODELS or custom_model_tags.
    """

    def test_selects_highest_quality_tagged_model(self, monkeypatch):
        """Test selects highest quality model when multiple tagged models available."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:70b": 25.0,
                "custom-medium:12b": 10.0,
                "custom-small:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}
        # Tag multiple models for writer with different implicit quality (size-based)
        settings.custom_model_tags = {
            "custom-large:70b": ["writer"],
            "custom-medium:12b": ["writer"],
            "custom-small:8b": ["writer"],
        }

        result = settings.get_model_for_agent("writer", available_vram=50)

        # Should select largest tagged model (highest quality by size when quality equal)
        assert result == "custom-large:70b"

    def test_respects_vram_limit_for_tagged_models(self, monkeypatch):
        """Test only selects tagged models that fit in VRAM."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:70b": 25.0,  # Needs 30GB VRAM
                "custom-medium:12b": 10.0,  # Needs 12GB VRAM
                "custom-small:8b": 5.0,  # Needs 6GB VRAM
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"continuity": "auto"}
        settings.custom_model_tags = {
            "custom-large:70b": ["continuity"],
            "custom-medium:12b": ["continuity"],
            "custom-small:8b": ["continuity"],
        }

        # Only 15GB VRAM - large doesn't fit
        result = settings.get_model_for_agent("continuity", available_vram=15)

        # Should select medium (largest that fits)
        assert result == "custom-medium:12b"

    def test_multiple_roles_on_same_model(self, monkeypatch):
        """Test model can be tagged for multiple roles."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "versatile-model:12b": 10.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.custom_model_tags = {
            "versatile-model:12b": ["editor", "writer", "continuity"],
        }

        # Should work for all tagged roles
        settings.agent_models = {"editor": "auto"}
        assert settings.get_model_for_agent("editor", available_vram=24) == "versatile-model:12b"

        settings.agent_models = {"writer": "auto"}
        assert settings.get_model_for_agent("writer", available_vram=24) == "versatile-model:12b"

        settings.agent_models = {"continuity": "auto"}
        assert (
            settings.get_model_for_agent("continuity", available_vram=24) == "versatile-model:12b"
        )

    def test_skips_embedding_model_for_chat_roles(self, monkeypatch):
        """Embedding-tagged models are skipped when selecting for chat roles."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "mxbai-embed-large": 0.7,  # Embedding model
                "chat-model:8b": 5.0,  # Chat model
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}
        # Tag both for architect, but mxbai also has embedding tag
        settings.custom_model_tags = {
            "mxbai-embed-large": ["architect", "embedding"],
            "chat-model:8b": ["architect"],
        }

        result = settings.get_model_for_agent("architect", available_vram=24)

        # Should skip the embedding model and select the chat model
        assert result == "chat-model:8b"

    def test_raises_error_when_no_tagged_model_fits_vram(self, monkeypatch):
        """Test raises error when tagged models exist but none fit VRAM."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "large-model:70b": 25.0,  # Needs 30GB VRAM
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"editor": "auto"}
        settings.custom_model_tags = {"large-model:70b": ["editor"]}

        # Only 8GB VRAM - tagged model doesn't fit, should still select as last resort
        result = settings.get_model_for_agent("editor", available_vram=8)

        # Should select the only tagged model even if it doesn't fit
        assert result == "large-model:70b"


class TestModelTags:
    """Tests for get_model_tags and set_model_tags methods."""

    def test_get_model_tags_returns_empty_for_unknown_model(self):
        """Test returns empty list for unknown model."""
        settings = Settings()
        tags = settings.get_model_tags("unknown-model:7b")
        assert tags == []

    def test_get_model_tags_returns_recommended_tags(self):
        """Test returns tags from RECOMMENDED_MODELS for known models."""
        settings = Settings()
        # Use a model from RECOMMENDED_MODELS that has tags
        tags = settings.get_model_tags("vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0")
        assert "writer" in tags

    def test_get_model_tags_matches_by_base_name(self):
        """Test matches model by base name prefix."""
        settings = Settings()
        # Match by base name (without the tag suffix)
        tags = settings.get_model_tags("vanilj/mistral-nemo-12b-celeste-v1.9:latest")
        assert "writer" in tags

    def test_get_model_tags_returns_custom_tags(self):
        """Test returns custom tags when set."""
        settings = Settings()
        settings.custom_model_tags = {"my-model:7b": ["writer", "editor"]}
        tags = settings.get_model_tags("my-model:7b")
        assert "writer" in tags
        assert "editor" in tags

    def test_get_model_tags_merges_recommended_and_custom(self):
        """Test merges tags from RECOMMENDED_MODELS and custom tags."""
        settings = Settings()
        # Add a custom tag to a recommended model
        model_id = "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0"
        settings.custom_model_tags = {model_id: ["validator"]}
        tags = settings.get_model_tags(model_id)
        # Should have both recommended tags (writer) and custom tag (validator)
        assert "writer" in tags
        assert "validator" in tags

    def test_set_model_tags_saves_tags(self, tmp_path, monkeypatch):
        """Test set_model_tags saves tags to settings."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings()
        settings.set_model_tags("my-model:7b", ["writer", "editor"])

        assert settings.custom_model_tags == {"my-model:7b": ["writer", "editor"]}
        # Verify it was saved to file
        assert settings_file.exists()

    def test_set_model_tags_removes_empty_tags(self, tmp_path, monkeypatch):
        """Test set_model_tags removes entry when tags are empty."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings()
        settings.custom_model_tags = {"my-model:7b": ["writer"]}
        settings.set_model_tags("my-model:7b", [])

        assert "my-model:7b" not in settings.custom_model_tags

    def test_set_model_tags_updates_existing_tags(self, tmp_path, monkeypatch):
        """Test set_model_tags updates existing tags."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings()
        settings.custom_model_tags = {"my-model:7b": ["writer"]}
        settings.set_model_tags("my-model:7b", ["editor", "validator"])

        assert settings.custom_model_tags == {"my-model:7b": ["editor", "validator"]}


class TestSettingsFixtureIsolation:
    """Tests to ensure settings fixtures don't depend on local settings.json."""

    def test_cached_settings_uses_defaults(self, cached_settings):
        """cached_settings fixture should use default values, not load from file."""
        # Verify it has default values, not potentially different values from local file
        assert cached_settings.ollama_url == "http://localhost:11434"
        assert cached_settings.context_size == 32768
        assert cached_settings.max_tokens == 8192
        assert cached_settings.use_per_agent_models is True
        # Verify the prompt_templates_dir is the default
        assert cached_settings.prompt_templates_dir == "src/prompts/templates"

    def test_tmp_settings_uses_defaults(self, tmp_settings):
        """tmp_settings fixture should use default values, not load from file."""
        # Verify it has default values, not potentially different values from local file
        assert tmp_settings.ollama_url == "http://localhost:11434"
        assert tmp_settings.context_size == 32768
        assert tmp_settings.max_tokens == 8192
        assert tmp_settings.use_per_agent_models is True
        # Verify the prompt_templates_dir is the default
        assert tmp_settings.prompt_templates_dir == "src/prompts/templates"

    def test_settings_constructor_does_not_load_from_file(self):
        """Settings() should create instance with defaults, not load from file."""
        # Clear cache to ensure we're not getting a cached loaded instance
        Settings.clear_cache()

        # Create new settings instance
        settings = Settings()

        # Verify it has default values
        assert settings.prompt_templates_dir == "src/prompts/templates"
        assert settings.ollama_url == "http://localhost:11434"

    def test_settings_load_with_mocked_file(self, tmp_path, monkeypatch):
        """Settings.load() should respect SETTINGS_FILE mock."""
        # Create a test settings file with custom values
        test_settings_file = tmp_path / "test_settings.json"
        test_data = {
            "prompt_templates_dir": "custom/path/templates",
            "ollama_url": "http://custom:11434",
        }
        test_settings_file.write_text(json.dumps(test_data))

        # Mock SETTINGS_FILE to point to our test file
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", test_settings_file)
        Settings.clear_cache()

        # Load settings - should load from mocked file
        settings = Settings.load()

        # Verify it loaded from our test file, not defaults
        assert settings.prompt_templates_dir == "custom/path/templates"
        assert settings.ollama_url == "http://custom:11434"

    def test_settings_load_without_mock_isolated_from_constructor(self, tmp_path, monkeypatch):
        """Test that Settings.load() and Settings() are independent."""
        # Setup: Create a temporary settings file with different values
        test_settings_file = tmp_path / "test_settings.json"
        test_data = {"prompt_templates_dir": "different/path"}
        test_settings_file.write_text(json.dumps(test_data))

        # Mock SETTINGS_FILE
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", test_settings_file)
        Settings.clear_cache()

        # Create instance with constructor - should use defaults
        settings_default = Settings()
        assert settings_default.prompt_templates_dir == "src/prompts/templates"

        # Load from file - should use file values
        settings_loaded = Settings.load()
        assert settings_loaded.prompt_templates_dir == "different/path"


class TestSettingsMigration:
    """Tests for settings migration when new fields are added."""

    def test_missing_agent_temperature_backfilled_on_validate(self):
        """Validation should backfill missing agent temperatures from defaults."""
        settings = Settings()
        # Simulate an old settings file that lacks the "embedding" key
        del settings.agent_temperatures["embedding"]
        assert "embedding" not in settings.agent_temperatures

        # Validation should backfill it
        settings.validate()

        assert "embedding" in settings.agent_temperatures
        assert settings.agent_temperatures["embedding"] == 0.0

    def test_missing_multiple_agent_temperatures_backfilled(self):
        """Validation should backfill all missing agent temperatures."""
        settings = Settings()
        # Remove multiple agents
        del settings.agent_temperatures["embedding"]
        del settings.agent_temperatures["suggestion"]

        settings.validate()

        assert settings.agent_temperatures["embedding"] == 0.0
        assert settings.agent_temperatures["suggestion"] == 0.8

    def test_load_old_settings_file_without_embedding_temp(self, tmp_path, monkeypatch):
        """Loading settings saved before embedding role added should not crash."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Create settings file missing the "embedding" key in agent_temperatures
        old_temps = {
            "interviewer": 0.7,
            "architect": 0.85,
            "writer": 0.9,
            "editor": 0.6,
            "continuity": 0.3,
            "validator": 0.1,
            "suggestion": 0.8,
            # No "embedding" key — simulates pre-migration file
        }

        with open(settings_file, "w") as f:
            json.dump({"agent_temperatures": old_temps}, f)

        # Should load without crashing and backfill the missing key
        settings = Settings.load()
        assert "embedding" in settings.agent_temperatures
        assert settings.agent_temperatures["embedding"] == 0.0

    def test_relationship_token_limit_default_is_800(self):
        """Default llm_tokens_relationship_create should be 800 (increased from 500)."""
        settings = Settings()
        assert settings.llm_tokens_relationship_create == 800


class TestWP1WP2SettingsValidation:
    """Tests for WP1/WP2 settings validation."""

    def test_validate_raises_on_invalid_decay_curve(self):
        """Should raise ValueError for invalid refinement temp decay curve."""
        settings = Settings()
        settings.world_quality_refinement_temp_decay = "invalid"
        with pytest.raises(ValueError, match="world_quality_refinement_temp_decay must be one of"):
            settings.validate()

    def test_validate_raises_on_invalid_min_iterations(self):
        """Should raise ValueError for early stopping min_iterations out of range."""
        settings = Settings()
        settings.world_quality_early_stopping_min_iterations = 0
        with pytest.raises(
            ValueError, match="world_quality_early_stopping_min_iterations must be between"
        ):
            settings.validate()

    def test_validate_raises_on_invalid_variance_tolerance(self):
        """Should raise ValueError for early stopping variance_tolerance out of range."""
        settings = Settings()
        settings.world_quality_early_stopping_variance_tolerance = 5.0
        with pytest.raises(
            ValueError, match="world_quality_early_stopping_variance_tolerance must be between"
        ):
            settings.validate()

    def test_validate_raises_on_invalid_circuit_breaker_failure_threshold(self):
        """Should raise ValueError for circuit breaker failure_threshold out of range."""
        settings = Settings()
        settings.circuit_breaker_failure_threshold = 0
        with pytest.raises(ValueError, match="circuit_breaker_failure_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_circuit_breaker_success_threshold(self):
        """Should raise ValueError for circuit breaker success_threshold out of range."""
        settings = Settings()
        settings.circuit_breaker_success_threshold = 0
        with pytest.raises(ValueError, match="circuit_breaker_success_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_circuit_breaker_timeout(self):
        """Should raise ValueError for circuit breaker timeout out of range."""
        settings = Settings()
        settings.circuit_breaker_timeout = 5.0  # Below minimum of 10
        with pytest.raises(ValueError, match="circuit_breaker_timeout must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_retry_temp_increase(self):
        """Should raise ValueError for retry_temp_increase out of range."""
        settings = Settings()
        settings.retry_temp_increase = 2.0  # Above maximum of 1.0
        with pytest.raises(ValueError, match="retry_temp_increase must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_retry_simplify_on_attempt(self):
        """Should raise ValueError for retry_simplify_on_attempt out of range."""
        settings = Settings()
        settings.retry_simplify_on_attempt = 1  # Below minimum of 2
        with pytest.raises(ValueError, match="retry_simplify_on_attempt must be between"):
            settings.validate()

    def test_validate_raises_on_invalid_semantic_duplicate_threshold(self):
        """Should raise ValueError for semantic_duplicate_threshold out of range."""
        settings = Settings()
        settings.semantic_duplicate_threshold = 0.3  # Below minimum of 0.5
        with pytest.raises(ValueError, match="semantic_duplicate_threshold must be between"):
            settings.validate()

    def test_validate_raises_on_empty_embedding_model_when_semantic_enabled(self):
        """Should raise ValueError when semantic_duplicate_enabled but embedding_model is empty."""
        settings = Settings()
        settings.semantic_duplicate_enabled = True
        settings.embedding_model = ""  # Empty model
        with pytest.raises(ValueError, match="embedding_model must be set"):
            settings.validate()

    def test_validate_raises_on_invalid_temp_decay_order(self):
        """Should raise ValueError when temp_start < temp_end (invalid decay)."""
        settings = Settings()
        settings.world_quality_refinement_temp_start = 0.3  # Less than end
        settings.world_quality_refinement_temp_end = 0.7
        with pytest.raises(ValueError, match="must be >= world_quality_refinement_temp_end"):
            settings.validate()

    def test_valid_new_settings_pass_validation(self):
        """Valid new settings should pass validation."""
        settings = Settings(
            world_quality_refinement_temp_decay="exponential",
            world_quality_early_stopping_min_iterations=3,
            world_quality_early_stopping_variance_tolerance=0.5,
            circuit_breaker_failure_threshold=10,
            circuit_breaker_success_threshold=3,
            circuit_breaker_timeout=120.0,
            retry_temp_increase=0.2,
            retry_simplify_on_attempt=4,
            semantic_duplicate_threshold=0.9,
        )
        # Should not raise
        settings.validate()


class TestEmbeddingModelMigration:
    """Tests for stale embedding model migration during validation."""

    def test_valid_embedding_model_unchanged(self):
        """Valid embedding model should not be changed by validation."""
        settings = Settings()
        settings.embedding_model = "mxbai-embed-large"
        settings.validate()
        assert settings.embedding_model == "mxbai-embed-large"

    def test_stale_embedding_model_migrated(self):
        """Removed embedding model should be migrated to first valid one."""
        settings = Settings()
        settings.embedding_model = "nomic-embed-text"
        settings.validate()
        # Should have been migrated to a model with "embedding" tag
        assert settings.embedding_model != "nomic-embed-text"
        model_info = RECOMMENDED_MODELS.get(settings.embedding_model)
        assert model_info is not None
        assert "embedding" in model_info["tags"]

    def test_unknown_embedding_model_migrated(self):
        """Completely unknown embedding model should be migrated."""
        settings = Settings()
        settings.embedding_model = "some-nonexistent-model:latest"
        settings.validate()
        assert settings.embedding_model != "some-nonexistent-model:latest"
        model_info = RECOMMENDED_MODELS.get(settings.embedding_model)
        assert model_info is not None
        assert "embedding" in model_info["tags"]

    def test_non_embedding_model_migrated(self):
        """A model in the registry but without embedding tag should be migrated."""
        settings = Settings()
        settings.embedding_model = "huihui_ai/dolphin3-abliterated:8b"
        settings.validate()
        assert settings.embedding_model != "huihui_ai/dolphin3-abliterated:8b"
        model_info = RECOMMENDED_MODELS.get(settings.embedding_model)
        assert model_info is not None
        assert "embedding" in model_info["tags"]

    def test_empty_embedding_model_migrated(self):
        """Empty embedding model string should be migrated to a valid one."""
        settings = Settings()
        settings.embedding_model = ""
        settings.validate()
        # Empty string is not in the registry, so it should be migrated
        model_info = RECOMMENDED_MODELS.get(settings.embedding_model)
        assert model_info is not None
        assert "embedding" in model_info["tags"]

    def test_migration_logs_warning(self, caplog):
        """Migration should log a warning message."""
        import logging

        settings = Settings()
        settings.embedding_model = "nomic-embed-text"
        with caplog.at_level(logging.WARNING):
            settings.validate()
        assert "not in registry" in caplog.text
        assert "migrating to" in caplog.text

    def test_migration_persists_to_disk_on_load(self, tmp_path, monkeypatch):
        """Loading settings with a stale embedding model should auto-save the fix."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Write settings with stale embedding model
        defaults = Settings()
        defaults.embedding_model = "nomic-embed-text"
        from dataclasses import asdict

        with open(settings_file, "w") as f:
            json.dump(asdict(defaults), f)

        # Clear cache so load() reads from disk
        Settings.clear_cache()

        loaded = Settings.load(use_cache=False)

        # In-memory value should be migrated
        assert loaded.embedding_model != "nomic-embed-text"
        assert "embedding" in RECOMMENDED_MODELS[loaded.embedding_model]["tags"]

        # On-disk value should also be updated
        with open(settings_file) as f:
            on_disk = json.load(f)
        assert on_disk["embedding_model"] == loaded.embedding_model

    def test_no_embedding_models_in_registry_keeps_current(self, monkeypatch, caplog):
        """When registry has no embedding-tagged models, keep current model and warn."""
        import logging

        # Mock RECOMMENDED_MODELS to have no embedding-tagged models
        fake_registry = {
            "some-chat-model:8b": {
                "size_gb": 4.0,
                "tags": ["world_creator"],
            },
        }
        monkeypatch.setattr(
            "src.settings._model_registry.RECOMMENDED_MODELS",
            fake_registry,
        )

        settings = Settings()
        settings.embedding_model = "some-old-model"
        with caplog.at_level(logging.WARNING):
            settings.validate()

        # Model should be unchanged since no valid alternative exists
        assert settings.embedding_model == "some-old-model"
        assert "No embedding models found in registry" in caplog.text

    def test_validate_returns_true_when_migrated(self):
        """validate() should return True when embedding model is migrated."""
        settings = Settings()
        settings.embedding_model = "nomic-embed-text"
        changed = settings.validate()
        assert changed is True

    def test_validate_returns_false_when_no_changes(self):
        """validate() should return False when no settings are mutated."""
        settings = Settings()
        settings.embedding_model = "mxbai-embed-large"  # Already valid, no migration
        changed = settings.validate()
        assert changed is False

    def test_validate_returns_false_no_embedding_models_in_registry(self, monkeypatch):
        """validate() returns False when no embedding models exist (keeps current, no change)."""
        fake_registry = {
            "some-chat-model:8b": {"size_gb": 4.0, "tags": ["world_creator"]},
        }
        monkeypatch.setattr(
            "src.settings._model_registry.RECOMMENDED_MODELS",
            fake_registry,
        )
        settings = Settings()
        settings.embedding_model = "some-old-model"
        changed = settings.validate()
        assert changed is False
