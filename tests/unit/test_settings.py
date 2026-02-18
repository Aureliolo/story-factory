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
        assert settings.log_level == "INFO"

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

    def test_embedding_model_defaults_to_mxbai(self):
        """Embedding model should default to mxbai-embed-large."""
        settings = Settings()
        assert settings.embedding_model == "mxbai-embed-large"

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

    def test_validate_raises_on_invalid_vram_strategy(self):
        """Should raise ValueError for unrecognized vram_strategy."""
        settings = Settings()
        settings.vram_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="vram_strategy must be one of"):
            settings.validate()

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

    def test_circular_check_all_types_default_true(self):
        """circular_check_all_types should default to True."""
        settings = Settings()
        assert settings.circular_check_all_types is True

    def test_validate_raises_on_non_bool_circular_check_all_types(self):
        """Should raise ValueError for non-boolean circular_check_all_types."""
        settings = Settings()
        settings.circular_check_all_types = "yes"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="circular_check_all_types must be a boolean"):
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
        with pytest.raises(ValueError, match="Unknown keys in relationship_minimums"):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_value_not_dict(self):
        """Should raise ValueError for relationship_minimums with non-dict values."""
        settings = Settings()
        # Set only the "character" value to non-dict, keep others intact
        settings.relationship_minimums["character"] = [("protagonist", 5)]  # type: ignore[assignment]
        with pytest.raises(ValueError, match=r"relationship_minimums\[character\] must be a dict"):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_role_not_string(self):
        """Should raise ValueError for relationship_minimums with non-string role keys."""
        settings = Settings()
        settings.relationship_minimums["character"] = {123: 5}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match=r"Unknown keys in relationship_minimums\[character\]"):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_count_not_int(self):
        """Should raise ValueError for relationship_minimums with non-int min_count."""
        settings = Settings()
        settings.relationship_minimums["character"]["protagonist"] = "five"  # type: ignore[assignment]
        with pytest.raises(
            ValueError,
            match=r"relationship_minimums\[character\]\[protagonist\] must be a non-negative integer",
        ):
            settings.validate()

    def test_validate_raises_on_relationship_minimums_count_negative(self):
        """Should raise ValueError for relationship_minimums with negative min_count."""
        settings = Settings()
        settings.relationship_minimums["character"]["protagonist"] = -1
        with pytest.raises(
            ValueError,
            match=r"relationship_minimums\[character\]\[protagonist\] must be a non-negative integer",
        ):
            settings.validate()

    def test_validate_raises_on_minimum_exceeds_max_relationships(self):
        """Should raise ValueError when minimum exceeds max_relationships_per_entity."""
        settings = Settings()
        settings.max_relationships_per_entity = 5
        settings.relationship_minimums["character"]["protagonist"] = 10  # Exceeds max of 5
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

    def test_load_backs_up_corrupted_json(self, tmp_path, monkeypatch):
        """Test load creates a .corrupt backup when JSON is invalid."""
        settings_file = tmp_path / "settings.json"
        backup_file = tmp_path / "settings.json.corrupt"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        corrupted_content = "not valid json {{{"
        settings_file.write_text(corrupted_content)

        Settings.load()

        assert backup_file.exists()
        assert backup_file.read_text() == corrupted_content

    def test_load_handles_backup_failure_gracefully(self, tmp_path, monkeypatch):
        """Test load proceeds even if backup copy fails."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        settings_file.write_text("not valid json {{{")

        with patch("src.settings._settings.shutil.copy", side_effect=OSError("denied")):
            settings = Settings.load()

        # Should still return defaults despite backup failure
        assert settings.ollama_url == "http://localhost:11434"

    def test_load_handles_non_dict_json_root(self, tmp_path, monkeypatch):
        """Test load handles JSON file with non-object root (e.g. list)."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        settings_file.write_text("[1, 2, 3]")

        settings = Settings.load()

        # Should return defaults after treating non-dict as corrupted
        assert settings.ollama_url == "http://localhost:11434"

    def test_load_backs_up_non_dict_json_root(self, tmp_path, monkeypatch):
        """Test load creates .corrupt backup when JSON root is not an object."""
        settings_file = tmp_path / "settings.json"
        backup_file = tmp_path / "settings.json.corrupt"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        settings_file.write_text("[1, 2, 3]")

        Settings.load()

        assert backup_file.exists()
        assert backup_file.read_text() == "[1, 2, 3]"

    def test_load_handles_non_dict_json_root_backup_failure(self, tmp_path, monkeypatch):
        """Test load proceeds even if backup copy fails for non-dict JSON root."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        settings_file.write_text('"just a string"')

        with patch("src.settings._settings.shutil.copy", side_effect=OSError("denied")):
            settings = Settings.load()

        # Should still return defaults despite backup failure
        assert settings.ollama_url == "http://localhost:11434"

    def test_load_wraps_type_error_as_value_error(self, tmp_path, monkeypatch):
        """Test load wraps TypeError from constructor as ValueError."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Patch _merge_with_defaults to inject a bad type that bypasses merge
        # but causes TypeError in the dataclass constructor
        original_merge = __import__(
            "src.settings._settings", fromlist=["_merge_with_defaults"]
        )._merge_with_defaults

        def bad_merge(data, cls):
            result = original_merge(data, cls)
            # Inject a value that causes TypeError when passed to dataclass
            data["unexpected_kwarg_xyz"] = "boom"
            return result

        with patch("src.settings._settings._merge_with_defaults", side_effect=bad_merge):
            with pytest.raises(ValueError, match="A setting has an invalid type"):
                Settings.load()

    def test_load_handles_unknown_fields(self, tmp_path, monkeypatch):
        """Test load filters out unknown fields and preserves known ones."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Create a settings file with known + unknown fields
        data = {
            "ollama_url": "http://custom:11434",
            "semantic_duplicate_enabled": True,
            "world_quality_enabled": False,
            "unknown_field": "value",
            "removed_setting": 42,
        }
        with open(settings_file, "w") as f:
            json.dump(data, f)

        # Should drop unknown fields and preserve all known ones
        settings = Settings.load()
        assert settings.ollama_url == "http://custom:11434"
        assert settings.semantic_duplicate_enabled is True
        assert settings.world_quality_enabled is False

        # Saved file should have unknown fields removed
        with open(settings_file) as f:
            saved = json.load(f)
        assert "unknown_field" not in saved
        assert "removed_setting" not in saved
        assert saved["ollama_url"] == "http://custom:11434"

    def test_load_adds_defaults_for_new_fields(self, tmp_path, monkeypatch):
        """Test load uses defaults for fields missing from JSON."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Create a minimal settings file missing most fields
        data = {"ollama_url": "http://custom:11434"}
        with open(settings_file, "w") as f:
            json.dump(data, f)

        settings = Settings.load()
        assert settings.ollama_url == "http://custom:11434"
        # Missing fields get their defaults
        assert settings.semantic_duplicate_enabled is True
        assert settings.context_size == 32768

        # Saved file should now include all fields
        with open(settings_file) as f:
            saved = json.load(f)
        assert "semantic_duplicate_enabled" in saved
        assert "context_size" in saved

    def test_load_raises_on_invalid_values(self, tmp_path, monkeypatch):
        """Test load raises ValueError for genuinely invalid setting values."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Create a settings file with an invalid value
        data = {"context_size": 500}  # Too low (min 1024)
        with open(settings_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="context_size must be between"):
            Settings.load()

    def test_load_wraps_wrong_type_scalar_as_value_error(self, tmp_path, monkeypatch):
        """Wrong-type scalar in JSON (e.g. string for int) raises ValueError, not TypeError."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        data = {"context_size": "not_a_number"}
        with open(settings_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="A setting has an invalid type"):
            Settings.load()

    def test_load_does_not_save_on_validation_failure(self, tmp_path, monkeypatch):
        """Settings file on disk is NOT rewritten when validate() raises."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Write settings with an invalid value (too low)
        data = {"context_size": 500}
        with open(settings_file, "w") as f:
            json.dump(data, f)
        original_content = settings_file.read_text()

        with pytest.raises(ValueError):
            Settings.load()

        # File should be unchanged after the failed load
        assert settings_file.read_text() == original_content

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

    def test_load_with_old_agent_models_preserves_settings(self, tmp_path, monkeypatch):
        """Loading settings with missing agent roles doesn't nuke everything."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Write an old settings file missing "judge" and with extra "validator"
        old_data = {
            "log_level": "DEBUG",
            "ollama_url": "http://custom:9999",
            "agent_models": {
                "interviewer": "auto",
                "architect": "auto",
                "writer": "my-writer:7b",
                "editor": "auto",
                "continuity": "auto",
                "suggestion": "auto",
                "validator": "auto",
            },
            "agent_temperatures": {
                "interviewer": 0.7,
                "architect": 0.85,
                "writer": 0.9,
                "editor": 0.6,
                "continuity": 0.3,
                "suggestion": 0.8,
                "validator": 0.1,
            },
        }
        with open(settings_file, "w") as f:
            json.dump(old_data, f)

        settings = Settings.load()

        # User customizations preserved
        assert settings.log_level == "DEBUG"
        assert settings.ollama_url == "http://custom:9999"
        assert settings.agent_models["writer"] == "my-writer:7b"
        # Obsolete "validator" removed, new "judge" added
        assert "validator" not in settings.agent_models
        assert settings.agent_models["judge"] == "auto"
        assert "validator" not in settings.agent_temperatures
        assert settings.agent_temperatures["judge"] == 0.1


class TestMergeWithDefaults:
    """Tests for the _merge_with_defaults function used during Settings.load()."""

    def test_field_tuples_reference_real_settings_fields(self):
        """_STRUCTURED_DICT_FIELDS and _NESTED_DICT_FIELDS reference real dict fields."""
        from dataclasses import fields as dc_fields

        from src.settings._settings import _NESTED_DICT_FIELDS, _STRUCTURED_DICT_FIELDS

        known = {f.name for f in dc_fields(Settings)}
        for name in _STRUCTURED_DICT_FIELDS + _NESTED_DICT_FIELDS:
            assert name in known, f"{name} is not a field on Settings"
            default_val = getattr(Settings(), name)
            assert isinstance(default_val, dict), f"{name} default is not a dict"

    def test_adds_missing_top_level_fields(self):
        """Missing settings are added with their default values."""
        from src.settings._settings import _merge_with_defaults

        data = {"ollama_url": "http://custom:11434"}
        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "context_size" in data
        assert data["ollama_url"] == "http://custom:11434"  # Preserved

    def test_removes_obsolete_top_level_fields(self):
        """Settings that no longer exist in the dataclass are removed."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["removed_feature"] = "old_value"
        data["another_gone"] = 42

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "removed_feature" not in data
        assert "another_gone" not in data

    def test_adds_missing_dict_sub_keys(self):
        """New sub-keys in structured dict fields are added with defaults."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        # Simulate old settings file missing the "judge" agent
        del data["agent_models"]["judge"]
        del data["agent_temperatures"]["judge"]

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert data["agent_models"]["judge"] == "auto"
        assert data["agent_temperatures"]["judge"] == 0.1

    def test_removes_obsolete_dict_sub_keys(self):
        """Obsolete sub-keys in structured dict fields are removed."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        # Simulate old settings with a removed agent role
        data["agent_models"]["validator"] = "auto"
        data["agent_temperatures"]["validator"] = 0.1

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "validator" not in data["agent_models"]
        assert "validator" not in data["agent_temperatures"]

    def test_removes_embedding_from_agent_models(self):
        """Stale 'embedding' entry in agent_models is removed by merge."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["agent_models"]["embedding"] = "some-embed-model"

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "embedding" not in data["agent_models"]

    def test_preserves_user_customizations(self):
        """User-modified values survive the merge unchanged."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["log_level"] = "DEBUG"
        data["agent_temperatures"]["writer"] = 1.5
        data["world_quality_thresholds"]["character"] = 9.0

        changed = _merge_with_defaults(data, Settings)

        assert changed is False
        assert data["log_level"] == "DEBUG"
        assert data["agent_temperatures"]["writer"] == 1.5
        assert data["world_quality_thresholds"]["character"] == 9.0

    def test_handles_wrong_type_for_dict_field(self):
        """Dict fields with wrong type are replaced with defaults."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["agent_models"] = "not_a_dict"

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert isinstance(data["agent_models"], dict)
        assert "writer" in data["agent_models"]

    def test_merges_nested_dict_fields(self):
        """Nested dict fields (relationship_minimums) are merged at both levels."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        # Remove an outer key
        del data["relationship_minimums"]["item"]
        # Remove an inner key
        del data["relationship_minimums"]["character"]["minor"]

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "item" in data["relationship_minimums"]
        assert "minor" in data["relationship_minimums"]["character"]

    def test_removes_obsolete_nested_dict_outer_keys(self):
        """Obsolete outer keys in nested dict fields are removed."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["relationship_minimums"]["alien"] = {"warrior": 3}

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "alien" not in data["relationship_minimums"]

    def test_removes_obsolete_nested_dict_inner_keys(self):
        """Obsolete inner keys in nested dict fields are removed."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["relationship_minimums"]["character"]["legendary"] = 99

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert "legendary" not in data["relationship_minimums"]["character"]

    def test_handles_wrong_type_for_nested_dict_field(self):
        """Nested dict fields with wrong type are replaced with defaults."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["relationship_minimums"] = "not_a_dict"

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert isinstance(data["relationship_minimums"], dict)
        assert "character" in data["relationship_minimums"]

    def test_handles_wrong_type_for_nested_dict_inner_value(self):
        """Inner values of wrong type in nested dicts are reset to defaults."""
        from dataclasses import asdict

        from src.settings._settings import _merge_with_defaults

        data = asdict(Settings())
        data["relationship_minimums"]["character"] = "not_a_dict"

        changed = _merge_with_defaults(data, Settings)

        assert changed is True
        assert isinstance(data["relationship_minimums"]["character"], dict)
        assert "protagonist" in data["relationship_minimums"]["character"]


class TestDictStructureValidation:
    """Tests for _validate_dict_structure in validate()."""

    def test_rejects_unknown_key_in_agent_models(self):
        """Unknown agent role in agent_models should raise ValueError."""
        settings = Settings()
        settings.agent_models["unknown_agent"] = "some-model:8b"
        with pytest.raises(ValueError, match="Unknown keys in agent_models"):
            settings.validate()

    def test_rejects_missing_key_in_agent_models(self):
        """Missing agent role in agent_models should raise ValueError."""
        settings = Settings()
        del settings.agent_models["judge"]
        with pytest.raises(ValueError, match="Missing keys in agent_models"):
            settings.validate()

    def test_raises_on_non_dict_agent_temperatures(self):
        """Non-dict agent_temperatures raises ValueError, not AttributeError."""
        settings = Settings()
        settings.agent_temperatures = "not a dict"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="agent_temperatures must be a dict"):
            settings.validate()

    def test_raises_on_non_numeric_agent_temperature(self):
        """Non-numeric temperature value raises ValueError, not TypeError."""
        settings = Settings()
        settings.agent_temperatures["writer"] = "hot"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Temperature for writer must be a number"):
            settings.validate()

    def test_rejects_unknown_key_in_agent_temperatures(self):
        """Unknown agent in agent_temperatures should raise ValueError."""
        settings = Settings()
        settings.agent_temperatures["unknown_agent"] = 0.5
        with pytest.raises(ValueError, match="Unknown keys in agent_temperatures"):
            settings.validate()

    def test_rejects_unknown_entity_type_in_thresholds(self):
        """Unknown entity type in world_quality_thresholds should raise ValueError."""
        settings = Settings()
        settings.world_quality_thresholds["alien"] = 5.0
        with pytest.raises(ValueError, match="Unknown keys in world_quality_thresholds"):
            settings.validate()

    def test_rejects_unknown_outer_key_in_nested_dict(self):
        """Unknown outer key in relationship_minimums should raise ValueError."""
        settings = Settings()
        settings.relationship_minimums["alien"] = {"warrior": 3}
        with pytest.raises(ValueError, match="Unknown keys in relationship_minimums"):
            settings.validate()

    def test_rejects_unknown_inner_key_in_nested_dict(self):
        """Unknown inner key in relationship_minimums should raise ValueError."""
        settings = Settings()
        settings.relationship_minimums["character"]["legendary"] = 99
        with pytest.raises(ValueError, match=r"Unknown keys in relationship_minimums\[character\]"):
            settings.validate()

    def test_passes_for_valid_default_settings(self):
        """Default settings should pass structural validation."""
        settings = Settings()
        settings.validate()  # Should not raise

    def test_raises_on_non_dict_structured_field(self):
        """Non-dict structured field raises ValueError immediately."""
        from src.settings._validation import _validate_dict_structure

        settings = Settings()
        settings.world_quality_thresholds = "not a dict"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="world_quality_thresholds must be a dict"):
            _validate_dict_structure(settings)

    def test_raises_on_non_dict_nested_field(self):
        """Non-dict nested field raises ValueError immediately."""
        from src.settings._validation import _validate_dict_structure

        settings = Settings()
        settings.relationship_minimums = "not a dict"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="relationship_minimums must be a dict"):
            _validate_dict_structure(settings)

    def test_rejects_missing_outer_key_in_nested_dict(self):
        """Missing outer key in relationship_minimums should raise ValueError."""
        settings = Settings()
        del settings.relationship_minimums["character"]
        with pytest.raises(ValueError, match="Missing keys in relationship_minimums"):
            settings.validate()

    def test_rejects_missing_inner_key_in_nested_dict(self):
        """Missing inner key in relationship_minimums should raise ValueError."""
        settings = Settings()
        del settings.relationship_minimums["character"]["protagonist"]
        with pytest.raises(ValueError, match=r"Missing keys in relationship_minimums\[character\]"):
            settings.validate()


class TestAtomicWrite:
    """Tests for _atomic_write_json function."""

    def test_writes_json_to_file(self, tmp_path):
        """Should write valid JSON to the target path."""
        from src.settings._settings import _atomic_write_json

        target = tmp_path / "test.json"
        _atomic_write_json(target, {"key": "value"})

        assert target.exists()
        with open(target) as f:
            data = json.load(f)
        assert data == {"key": "value"}

    def test_no_temp_file_left_on_success(self, tmp_path):
        """No .tmp files should remain after successful write."""
        from src.settings._settings import _atomic_write_json

        target = tmp_path / "test.json"
        _atomic_write_json(target, {"key": "value"})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_no_partial_write_on_json_error(self, tmp_path):
        """If JSON serialization fails, the target file is not created."""
        from src.settings._settings import _atomic_write_json

        target = tmp_path / "test.json"

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            _atomic_write_json(target, {"bad": Unserializable()})

        assert not target.exists()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_cleanup_failure_during_write_error(self, tmp_path):
        """Original error propagates even if temp file cleanup fails."""
        from src.settings._settings import _atomic_write_json

        target = tmp_path / "test.json"

        class Unserializable:
            pass

        with patch("src.settings._settings.os.unlink", side_effect=OSError("permission denied")):
            with pytest.raises(TypeError):
                _atomic_write_json(target, {"bad": Unserializable()})


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
                "fake-writer:14b": 9.0,
                "fake-interviewer:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}
        settings.custom_model_tags = {
            "fake-writer:14b": ["writer"],
            "fake-interviewer:8b": ["interviewer"],
        }

        result = settings.get_model_for_agent("writer", available_vram=24)

        assert result == "fake-writer:14b"

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

    def test_selects_model_matched_by_base_name_prefix(self, monkeypatch):
        """Test quality lookup works for models matched by base-name prefix.

        When an installed model has a tag like ':latest' that doesn't match
        any exact key in RECOMMENDED_MODELS, the base-name prefix fallback
        should still resolve quality correctly.
        """
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "huihui_ai/qwen3-abliterated:latest": 9.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        # :latest matches qwen3-abliterated:30b by prefix — should get its tags
        result = settings.get_model_for_agent("architect", available_vram=24)
        assert result == "huihui_ai/qwen3-abliterated:latest"

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
        assert result == "huihui_ai/deepseek-r1-abliterated:7b"  # First in RECOMMENDED_MODELS
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

    # --- Per-entity quality thresholds ---

    def test_validate_per_entity_thresholds_default(self):
        """Default thresholds should come from PER_ENTITY_QUALITY_DEFAULTS."""
        settings = Settings()
        settings.validate()
        # Default factory is PER_ENTITY_QUALITY_DEFAULTS.copy, already populated
        assert settings.world_quality_thresholds["character"] == 7.5
        assert settings.world_quality_thresholds["item"] == 8.0  # Items have higher default
        assert settings.world_quality_thresholds["calendar"] == 7.5
        assert len(settings.world_quality_thresholds) == 9

    def test_validate_raises_on_per_entity_threshold_out_of_range(self):
        """Should raise ValueError for per-entity threshold out of 0-10 range."""
        settings = Settings()
        settings.world_quality_thresholds = {
            "character": 11.0,
            "location": 7.5,
            "faction": 7.5,
            "item": 7.5,
            "concept": 7.5,
            "calendar": 7.5,
            "relationship": 7.5,
            "plot": 7.5,
            "chapter": 7.5,
        }
        with pytest.raises(
            ValueError, match=r"world_quality_thresholds\[character\] must be between"
        ):
            settings.validate()

    def test_validate_raises_on_per_entity_threshold_negative(self):
        """Should raise ValueError for negative per-entity threshold."""
        settings = Settings()
        settings.world_quality_thresholds = {
            "character": 7.5,
            "location": 7.5,
            "faction": 7.5,
            "item": -1.0,
            "concept": 7.5,
            "calendar": 7.5,
            "relationship": 7.5,
            "plot": 7.5,
            "chapter": 7.5,
        }
        with pytest.raises(ValueError, match=r"world_quality_thresholds\[item\] must be between"):
            settings.validate()

    def test_no_migration_when_all_types_present(self):
        """No threshold migration needed when all entity types are present."""
        settings = Settings()
        # Set all thresholds explicitly so migration doesn't trigger
        settings.world_quality_thresholds = {
            "character": 7.5,
            "location": 7.5,
            "faction": 7.5,
            "item": 8.0,
            "concept": 7.5,
            "calendar": 7.5,
            "relationship": 7.5,
            "plot": 7.5,
            "chapter": 7.5,
        }
        # Set a valid embedding model to prevent embedding migration from returning True
        settings.embedding_model = "mxbai-embed-large"
        changed = settings.validate()
        assert changed is False

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

    # --- LLM semaphore timeout validation (line 548) ---

    def test_validate_raises_on_invalid_llm_semaphore_timeout(self):
        """Should raise ValueError for llm_semaphore_timeout out of range."""
        settings = Settings()
        settings.llm_semaphore_timeout = 10  # Below 30
        with pytest.raises(ValueError, match="llm_semaphore_timeout must be between 30 and 600"):
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


class TestWriterModelSelection:
    """Tests for writer model selection using tags."""

    def test_selects_tagged_creative_model_for_writer(self, monkeypatch):
        """Test selects creative writing specialist model tagged for writer role."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "fake-writer:14b": 9.0,
                "other-model:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}
        settings.custom_model_tags = {"fake-writer:14b": ["writer"]}

        result = settings.get_model_for_agent("writer", available_vram=24)

        assert result == "fake-writer:14b"

    def test_selects_alternative_tagged_writer_model(self, monkeypatch):
        """Test selects alternative tagged model when first isn't available."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "fake-writer-alt:24b": 14.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}
        settings.custom_model_tags = {"fake-writer-alt:24b": ["writer"]}

        result = settings.get_model_for_agent("writer", available_vram=24)

        assert result == "fake-writer-alt:24b"


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
        tags = settings.get_model_tags("huihui_ai/qwen2.5-1m-abliterated:14b")
        assert "writer" in tags

    def test_get_model_tags_matches_by_base_name(self):
        """Test matches model by base name prefix."""
        settings = Settings()
        # Match by base name (without the tag suffix)
        tags = settings.get_model_tags("huihui_ai/qwen2.5-1m-abliterated:latest")
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
        model_id = "huihui_ai/qwen2.5-1m-abliterated:14b"
        settings.custom_model_tags = {model_id: ["validator"]}
        tags = settings.get_model_tags(model_id)
        # Should have both recommended tags (writer) and custom tag (validator)
        assert "writer" in tags
        assert "validator" in tags

    def test_get_model_tags_exact_match_preferred_over_prefix(self):
        """Test exact key match is preferred over base-name prefix match.

        Models like qwen3-abliterated:14b and qwen3-abliterated:30b share
        the same base name. Without exact-match-first logic, the 14b variant
        would incorrectly inherit the 30b's tags (including 'judge').
        """
        settings = Settings()
        # 30B has judge tag, 14B does not
        tags_30b = settings.get_model_tags("huihui_ai/qwen3-abliterated:30b")
        tags_14b = settings.get_model_tags("huihui_ai/qwen3-abliterated:14b")
        assert "judge" in tags_30b
        assert "judge" not in tags_14b

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

    def test_load_backfills_missing_agent_temperature(self, tmp_path, monkeypatch):
        """Load should add missing agent temperatures from defaults via merge."""
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
            "suggestion": 0.8,
            "judge": 0.1,
        }

        with open(settings_file, "w") as f:
            json.dump({"agent_temperatures": old_temps}, f)

        settings = Settings.load()
        assert "embedding" in settings.agent_temperatures
        assert settings.agent_temperatures["embedding"] == 0.0

    def test_load_backfills_multiple_missing_temperatures(self, tmp_path, monkeypatch):
        """Load should add all missing agent temperatures via merge."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # Only a few temperatures present
        old_temps = {
            "writer": 0.9,
            "editor": 0.6,
        }

        with open(settings_file, "w") as f:
            json.dump({"agent_temperatures": old_temps}, f)

        settings = Settings.load()
        assert settings.agent_temperatures["embedding"] == 0.0
        assert settings.agent_temperatures["suggestion"] == 0.8
        # User values preserved
        assert settings.agent_temperatures["writer"] == 0.9

    def test_semaphore_timeout_default_is_300(self):
        """Default llm_semaphore_timeout should be 300 seconds."""
        settings = Settings()
        assert settings.llm_semaphore_timeout == 300


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

    def test_validate_raises_on_invalid_score_plateau_tolerance(self):
        """Should raise ValueError for score_plateau_tolerance out of range."""
        settings = Settings()
        settings.world_quality_score_plateau_tolerance = 2.0
        with pytest.raises(
            ValueError, match="world_quality_score_plateau_tolerance must be between"
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

    def test_validate_keeps_semantic_enabled_with_default_embedding(self):
        """Semantic duplicate stays enabled when default embedding model is set."""
        settings = Settings()
        settings.semantic_duplicate_enabled = True
        # embedding_model defaults to "mxbai-embed-large" so no auto-disable
        settings.validate()
        assert settings.semantic_duplicate_enabled is True
        assert settings.embedding_model == "mxbai-embed-large"

    def test_validate_auto_disables_semantic_when_embedding_cleared(self):
        """Should auto-disable semantic_duplicate_enabled when embedding_model is cleared."""
        settings = Settings()
        settings.semantic_duplicate_enabled = True
        settings.embedding_model = ""
        # _validate_semantic_duplicate auto-disables semantic duplicate,
        # _validate_rag_context auto-migrates empty embedding_model to default
        from src.settings._validation import _validate_semantic_duplicate

        changed = _validate_semantic_duplicate(settings)
        assert changed is True
        assert settings.semantic_duplicate_enabled is False

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


class TestGetModelsForRole:
    """Tests for Settings.get_models_for_role() method."""

    def test_returns_models_with_matching_role_tag(self, monkeypatch):
        """Returns models that have the requested role tag."""
        fake_registry = {
            "judge-model:8b": {"quality": 8, "tags": ["judge", "writer"]},
            "writer-model:30b": {"quality": 9, "tags": ["writer"]},
            "judge-small:4b": {"quality": 6, "tags": ["judge"]},
        }
        # Patch both the source module and the local binding in _settings.py
        monkeypatch.setattr("src.settings._model_registry.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr("src.settings._settings.RECOMMENDED_MODELS", fake_registry)

        # Mock installed models to include all registry models
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: dict.fromkeys(fake_registry, 4.0),
        )

        settings = Settings()
        result = settings.get_models_for_role("judge")

        assert "judge-model:8b" in result
        assert "judge-small:4b" in result
        assert "writer-model:30b" not in result

    def test_returns_sorted_by_quality_descending(self, monkeypatch):
        """Models are sorted by quality score descending."""
        fake_registry = {
            "low-quality:4b": {"quality": 4, "tags": ["judge"]},
            "high-quality:30b": {"quality": 9, "tags": ["judge"]},
            "mid-quality:8b": {"quality": 7, "tags": ["judge"]},
        }
        monkeypatch.setattr("src.settings._model_registry.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr("src.settings._settings.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: dict.fromkeys(fake_registry, 4.0),
        )

        settings = Settings()
        result = settings.get_models_for_role("judge")

        assert result[0] == "high-quality:30b"
        assert result[1] == "mid-quality:8b"
        assert result[2] == "low-quality:4b"

    def test_returns_empty_when_no_models_found(self, monkeypatch):
        """Returns empty list when no installed models have the role tag."""
        fake_registry = {
            "writer-only:8b": {"quality": 7, "tags": ["writer"]},
        }
        monkeypatch.setattr("src.settings._model_registry.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr("src.settings._settings.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: dict.fromkeys(fake_registry, 4.0),
        )

        settings = Settings()
        result = settings.get_models_for_role("judge")

        assert result == []

    def test_returns_empty_when_no_installed_models(self, monkeypatch):
        """Returns empty list when no models are installed."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: {},
        )

        settings = Settings()
        result = settings.get_models_for_role("judge")

        assert result == []

    def test_skips_embedding_models_for_chat_roles(self, monkeypatch):
        """Embedding models are excluded when searching for non-embedding roles."""
        fake_registry = {
            "embed-model:8b": {"quality": 9, "tags": ["writer", "embedding"]},
            "pure-writer:8b": {"quality": 7, "tags": ["writer"]},
        }
        monkeypatch.setattr("src.settings._model_registry.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr("src.settings._settings.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: dict.fromkeys(fake_registry, 4.0),
        )

        settings = Settings()
        result = settings.get_models_for_role("writer")

        # embed-model should be skipped because it has "embedding" tag
        assert "pure-writer:8b" in result
        assert "embed-model:8b" not in result

    def test_resolves_quality_via_base_name_prefix(self, monkeypatch):
        """Quality lookup falls back to base-name prefix when no exact key match."""
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "huihui_ai/qwen3-abliterated:latest": 9.0,
            },
        )

        settings = Settings()
        result = settings.get_models_for_role("architect")

        # :latest matches qwen3-abliterated:30b by prefix, gets architect tag
        assert "huihui_ai/qwen3-abliterated:latest" in result


class TestMinimumRoleQuality:
    """Tests for MINIMUM_ROLE_QUALITY and check_minimum_quality (Issue #228)."""

    def test_minimum_role_quality_has_judge(self):
        """MINIMUM_ROLE_QUALITY must include judge with threshold >= 7."""
        from src.settings._types import MINIMUM_ROLE_QUALITY

        assert "judge" in MINIMUM_ROLE_QUALITY
        assert MINIMUM_ROLE_QUALITY["judge"] >= 7

    def test_minimum_role_quality_excludes_embedding(self):
        """Embedding has no minimum quality — separate selection path."""
        from src.settings._types import MINIMUM_ROLE_QUALITY

        assert "embedding" not in MINIMUM_ROLE_QUALITY

    def test_check_minimum_quality_warns_low_quality_judge(self, caplog):
        """Low-quality model auto-selected for judge should log a warning."""
        import logging

        from src.settings._types import check_minimum_quality

        with caplog.at_level(logging.WARNING, logger="src.settings._types"):
            check_minimum_quality("tiny-model:1b", 4.0, "judge")

        assert any("below minimum quality" in r.message for r in caplog.records)
        assert any("judge" in r.message for r in caplog.records)

    def test_check_minimum_quality_no_warn_adequate_judge(self, caplog):
        """Quality 8 model should not trigger a warning for judge role."""
        import logging

        from src.settings._types import check_minimum_quality

        with caplog.at_level(logging.WARNING, logger="src.settings._types"):
            check_minimum_quality("good-model:12b", 8.0, "judge")

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("below minimum quality" in m for m in warning_messages)

    def test_auto_selection_calls_check_minimum_quality(self, monkeypatch):
        """get_model_for_agent should call check_minimum_quality on auto-selection."""
        from unittest.mock import MagicMock

        fake_registry = {
            "low-judge:4b": {
                "quality": 4,
                "tags": ["judge"],
                "size_gb": 3.0,
                "vram_required": 4,
            },
        }
        monkeypatch.setattr("src.settings._settings.RECOMMENDED_MODELS", fake_registry)
        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: {"low-judge:4b": 3.0},
        )
        monkeypatch.setattr("src.settings._utils.get_available_vram", lambda: 24)

        mock_check = MagicMock()
        monkeypatch.setattr("src.settings._settings.check_minimum_quality", mock_check)

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"judge": "auto"}
        settings.get_model_for_agent("judge")

        mock_check.assert_called_once_with("low-judge:4b", 4, "judge")

    def test_explicit_model_skips_check(self, monkeypatch):
        """Explicitly set model should NOT call check_minimum_quality."""
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "src.settings._settings.get_installed_models_with_sizes",
            lambda: {"my-model:8b": 5.0},
        )

        mock_check = MagicMock()
        monkeypatch.setattr("src.settings._settings.check_minimum_quality", mock_check)

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"judge": "my-model:8b"}
        result = settings.get_model_for_agent("judge")

        assert result == "my-model:8b"
        mock_check.assert_not_called()

    def test_registry_gemma3_12b_has_judge_tag(self):
        """Gemma 3 12B should have judge tag (best empirical judge, Issue #228)."""
        assert "judge" in RECOMMENDED_MODELS["gemma3:12b"]["tags"]

    def test_registry_phi4_14b_has_judge_tag(self):
        """Phi-4 14B should have judge tag (rank=0.99, Issue #228)."""
        assert "judge" in RECOMMENDED_MODELS["phi4:14b"]["tags"]

    def test_registry_gemma3_4b_has_judge_tag(self):
        """Gemma 3 4B should have judge tag (rank=0.94, Issue #228)."""
        assert "judge" in RECOMMENDED_MODELS["gemma3:4b"]["tags"]


class TestLogLevelValidation:
    """Tests for log_level setting and validation."""

    def test_log_level_default_is_info(self):
        """Default log_level should be INFO."""
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_validate_accepts_all_valid_log_levels(self):
        """Validation should accept all valid log levels."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            settings = Settings()
            settings.log_level = level
            settings.validate()  # Should not raise

    def test_validate_rejects_invalid_log_level(self):
        """Validation should reject invalid log levels like TRACE."""
        settings = Settings()
        settings.log_level = "TRACE"
        with pytest.raises(ValueError, match="log_level must be one of"):
            settings.validate()

    def test_validate_rejects_lowercase_log_level(self):
        """Validation should reject lowercase log level strings."""
        settings = Settings()
        settings.log_level = "debug"
        with pytest.raises(ValueError, match="log_level must be one of"):
            settings.validate()

    def test_validate_rejects_critical_log_level(self):
        """Validation should reject CRITICAL (not in allowed set)."""
        settings = Settings()
        settings.log_level = "CRITICAL"
        with pytest.raises(ValueError, match="log_level must be one of"):
            settings.validate()


class TestGetScaledTimeout:
    """Tests for Settings.get_scaled_timeout() method."""

    def test_small_model_scales_moderately(self):
        """Small model (~5GB) should get ~150s timeout with 120s base."""
        from unittest.mock import patch

        from src.settings._types import ModelInfo

        mock_info = ModelInfo(
            name="small-model:5b",
            size_gb=5.0,
            vram_required=6,
            quality=5,
            speed=7,
            uncensored=True,
            description="Test model",
            tags=["writer"],
        )
        with patch("src.settings._utils.get_model_info", return_value=mock_info):
            settings = Settings()
            settings.ollama_timeout = 120
            timeout = settings.get_scaled_timeout("small-model:5b")

        # Formula: 120 * (1 + 5/20) = 120 * 1.25 = 150
        assert timeout == 150.0

    def test_large_model_scales_significantly(self):
        """Large model (~40GB) should get ~360s timeout with 120s base."""
        from unittest.mock import patch

        from src.settings._types import ModelInfo

        mock_info = ModelInfo(
            name="large-model:70b",
            size_gb=40.0,
            vram_required=48,
            quality=9,
            speed=3,
            uncensored=True,
            description="Test model",
            tags=["writer"],
        )
        with patch("src.settings._utils.get_model_info", return_value=mock_info):
            settings = Settings()
            settings.ollama_timeout = 120
            timeout = settings.get_scaled_timeout("large-model:70b")

        # Formula: 120 * (1 + 40/20) = 120 * 3 = 360
        assert timeout == 360.0

    def test_medium_model_scales_appropriately(self):
        """Medium model (~20GB) should get ~240s timeout with 120s base."""
        from unittest.mock import patch

        from src.settings._types import ModelInfo

        mock_info = ModelInfo(
            name="medium-model:20b",
            size_gb=20.0,
            vram_required=24,
            quality=7,
            speed=5,
            uncensored=True,
            description="Test model",
            tags=["writer"],
        )
        with patch("src.settings._utils.get_model_info", return_value=mock_info):
            settings = Settings()
            settings.ollama_timeout = 120
            timeout = settings.get_scaled_timeout("medium-model:20b")

        # Formula: 120 * (1 + 20/20) = 120 * 2 = 240
        assert timeout == 240.0

    def test_unknown_model_returns_base_timeout(self):
        """Unknown model that raises exception should return base timeout."""
        from unittest.mock import patch

        with patch("src.settings._utils.get_model_info", side_effect=ValueError("Model not found")):
            settings = Settings()
            settings.ollama_timeout = 120
            timeout = settings.get_scaled_timeout("unknown-model:xyz")

        assert timeout == 120.0

    def test_zero_size_returns_base_timeout(self):
        """Model with zero size should return base timeout."""
        from unittest.mock import patch

        from src.settings._types import ModelInfo

        mock_info = ModelInfo(
            name="zero-model:0b",
            size_gb=0.0,
            vram_required=0,
            quality=5,
            speed=5,
            uncensored=True,
            description="Test model",
            tags=[],
        )
        with patch("src.settings._utils.get_model_info", return_value=mock_info):
            settings = Settings()
            settings.ollama_timeout = 120
            timeout = settings.get_scaled_timeout("zero-model:0b")

        assert timeout == 120.0

    def test_respects_custom_base_timeout(self):
        """Should scale from custom base timeout, not hardcoded value."""
        from unittest.mock import patch

        from src.settings._types import ModelInfo

        mock_info = ModelInfo(
            name="test-model:8b",
            size_gb=10.0,
            vram_required=12,
            quality=6,
            speed=6,
            uncensored=True,
            description="Test model",
            tags=["writer"],
        )
        with patch("src.settings._utils.get_model_info", return_value=mock_info):
            settings = Settings()
            settings.ollama_timeout = 60  # Custom lower timeout
            timeout = settings.get_scaled_timeout("test-model:8b")

        # Formula: 60 * (1 + 10/20) = 60 * 1.5 = 90
        assert timeout == 90.0


class TestRagContextValidation:
    """Tests for RAG context settings validation."""

    def test_validate_rag_default_settings_pass(self):
        """Default RAG settings should pass validation."""
        settings = Settings()
        settings.validate()  # Should not raise

    def test_validate_raises_on_rag_max_tokens_too_low(self):
        """Should reject rag_context_max_tokens below 100."""
        settings = Settings()
        settings.rag_context_max_tokens = 50
        with pytest.raises(ValueError, match="rag_context_max_tokens"):
            settings.validate()

    def test_validate_raises_on_rag_max_tokens_too_high(self):
        """Should reject rag_context_max_tokens above 16000."""
        settings = Settings()
        settings.rag_context_max_tokens = 20000
        with pytest.raises(ValueError, match="rag_context_max_tokens"):
            settings.validate()

    def test_validate_raises_on_rag_max_items_too_low(self):
        """Should reject rag_context_max_items below 1."""
        settings = Settings()
        settings.rag_context_max_items = 0
        with pytest.raises(ValueError, match="rag_context_max_items"):
            settings.validate()

    def test_validate_raises_on_rag_max_items_too_high(self):
        """Should reject rag_context_max_items above 100."""
        settings = Settings()
        settings.rag_context_max_items = 101
        with pytest.raises(ValueError, match="rag_context_max_items"):
            settings.validate()

    def test_validate_raises_on_rag_similarity_threshold_too_low(self):
        """Should reject rag_context_similarity_threshold below 0.0."""
        settings = Settings()
        settings.rag_context_similarity_threshold = -0.1
        with pytest.raises(ValueError, match="rag_context_similarity_threshold"):
            settings.validate()

    def test_validate_raises_on_rag_similarity_threshold_too_high(self):
        """Should reject rag_context_similarity_threshold above 1.0."""
        settings = Settings()
        settings.rag_context_similarity_threshold = 1.5
        with pytest.raises(ValueError, match="rag_context_similarity_threshold"):
            settings.validate()

    def test_validate_raises_on_rag_graph_depth_too_low(self):
        """Should reject rag_context_graph_depth below 1."""
        settings = Settings()
        settings.rag_context_graph_depth = 0
        with pytest.raises(ValueError, match="rag_context_graph_depth"):
            settings.validate()

    def test_validate_raises_on_rag_graph_depth_too_high(self):
        """Should reject rag_context_graph_depth above 3."""
        settings = Settings()
        settings.rag_context_graph_depth = 4
        with pytest.raises(ValueError, match="rag_context_graph_depth"):
            settings.validate()

    def test_validate_auto_migrates_empty_embedding_model(self):
        """Should auto-migrate empty embedding_model to default."""
        from src.settings._validation import _validate_rag_context

        settings = Settings()
        settings.embedding_model = ""
        changed = _validate_rag_context(settings)
        assert changed is True
        assert settings.embedding_model == "mxbai-embed-large"

    def test_validate_keeps_rag_enabled_when_embedding_model_set(self):
        """Should not change embedding_model when it is already configured."""
        from src.settings._validation import _validate_rag_context

        settings = Settings()
        settings.rag_context_enabled = True
        settings.embedding_model = "mxbai-embed-large"
        result = _validate_rag_context(settings)
        assert settings.rag_context_enabled is True
        assert settings.embedding_model == "mxbai-embed-large"
        assert result is False

    def test_validate_full_auto_migrates_empty_embedding_model(self):
        """Full validate() auto-migrates empty embedding_model instead of raising."""
        settings = Settings()
        settings.embedding_model = ""
        changed = settings.validate()
        assert changed is True
        assert settings.embedding_model == "mxbai-embed-large"

    def test_validate_preserves_semantic_duplicate_after_embedding_migration(self):
        """Ordering: rag_context migration runs before semantic_duplicate check.

        When a stale settings file has embedding_model="" and
        semantic_duplicate_enabled=True, the auto-migration must populate the
        embedding model BEFORE the semantic duplicate validator inspects it.
        Otherwise the user's preference is silently lost.
        """
        settings = Settings()
        settings.embedding_model = ""
        settings.semantic_duplicate_enabled = True
        changed = settings.validate()
        assert changed is True
        assert settings.embedding_model == "mxbai-embed-large"
        assert settings.semantic_duplicate_enabled is True

    def test_validate_auto_migrates_whitespace_only_embedding_model(self):
        """Should auto-migrate whitespace-only embedding_model to default."""
        from src.settings._validation import _validate_rag_context

        settings = Settings()
        settings.embedding_model = "   "
        changed = _validate_rag_context(settings)
        assert changed is True
        assert settings.embedding_model == "mxbai-embed-large"
