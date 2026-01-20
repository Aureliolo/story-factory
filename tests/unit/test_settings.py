"""Tests for the settings module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from settings import (
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

    def test_validate_passes_for_valid_settings(self):
        """Should not raise for valid default settings."""
        settings = Settings()
        settings.validate()  # Should not raise


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_known_model_info(self):
        """Should return info for known models."""
        # Pick a model from the registry
        model_id = list(RECOMMENDED_MODELS.keys())[0]
        info = get_model_info(model_id)
        assert "name" in info
        assert "quality" in info
        assert "speed" in info
        assert "tags" in info

    def test_returns_estimated_info_for_unknown_model(self, monkeypatch):
        """Should return estimated info for unknown models based on size."""
        # Mock installed models to return the unknown model with a size
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
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
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {},
        )
        info = get_model_info("completely-unknown-model:xyz")
        assert info["name"] == "completely-unknown-model:xyz"
        # Should use default values (quality=5, speed=5, vram=8)
        assert info["quality"] == 5
        assert info["speed"] == 5
        assert info["vram_required"] == 8


class TestAgentRoles:
    """Tests for agent role definitions."""

    def test_all_roles_defined(self):
        """Should have all required agent roles defined."""
        required_roles = ["interviewer", "architect", "writer", "editor", "continuity"]
        for role in required_roles:
            assert role in AGENT_ROLES
            assert "name" in AGENT_ROLES[role]
            assert "description" in AGENT_ROLES[role]


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


class TestSettingsSaveLoad:
    """Tests for Settings save and load methods."""

    def test_save_creates_file(self, tmp_path, monkeypatch):
        """Test save creates settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        settings = Settings()
        settings.save()

        assert settings_file.exists()
        with open(settings_file) as f:
            data = json.load(f)
        assert data["ollama_url"] == "http://localhost:11434"

    def test_load_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        """Test load returns defaults when settings file doesn't exist."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        settings = Settings.load()

        assert settings.ollama_url == "http://localhost:11434"
        # File should be created
        assert settings_file.exists()

    def test_load_reads_existing_file(self, tmp_path, monkeypatch):
        """Test load reads existing settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        # Create a settings file with custom value
        with open(settings_file, "w") as f:
            json.dump({"ollama_url": "http://custom:11434", "context_size": 16384}, f)

        settings = Settings.load()

        assert settings.ollama_url == "http://custom:11434"
        assert settings.context_size == 16384

    def test_load_handles_corrupted_json(self, tmp_path, monkeypatch):
        """Test load handles corrupted JSON file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        # Create a corrupted JSON file
        settings_file.write_text("not valid json {{{")

        settings = Settings.load()

        # Should return defaults
        assert settings.ollama_url == "http://localhost:11434"

    def test_load_handles_unknown_fields(self, tmp_path, monkeypatch):
        """Test load handles unknown fields in settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

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
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

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
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # First load creates a new instance
        settings1 = Settings.load()
        # Second load returns the same instance (cached)
        settings2 = Settings.load()

        assert settings1 is settings2

    def test_load_with_use_cache_false_reloads(self, tmp_path, monkeypatch):
        """Test load(use_cache=False) forces reload from disk."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)
        Settings.clear_cache()

        # First load
        settings1 = Settings.load()

        # Modify the cached instance
        settings1.dark_mode = not settings1.dark_mode

        # Load with use_cache=False should create a new instance
        settings2 = Settings.load(use_cache=False)

        assert settings1 is not settings2
        # New instance has default value
        assert settings2.dark_mode != settings1.dark_mode

    def test_clear_cache_clears_cached_instance(self, tmp_path, monkeypatch):
        """Test clear_cache() clears the cached instance."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)
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
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        # Data with valid fields
        data = {
            "ollama_url": "http://custom:11434",  # Valid
            "dark_mode": False,  # Valid
        }

        with caplog.at_level(logging.INFO):
            settings = Settings._recover_partial_settings(data)

        assert settings.ollama_url == "http://custom:11434"
        assert settings.dark_mode is False
        assert "Recovered" in caplog.text

    def test_falls_back_to_defaults_when_recovery_fails_validation(self, tmp_path, monkeypatch):
        """Test falls back to complete defaults when recovered settings fail validation."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

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
            "settings.get_installed_models_with_sizes",
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
            "settings.get_installed_models_with_sizes",
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

    def test_auto_selects_by_size_tier_when_no_tagged_model(self, monkeypatch):
        """Test auto-selects by size tier when no tagged model available."""
        # Mock installed models with sizes - no recommended models
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:30b": 18.0,  # Large tier
                "custom-medium:12b": 10.0,  # Medium tier
                "custom-small:8b": 5.0,  # Small tier
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        result = settings.get_model_for_agent("architect", available_vram=24)

        # Architect prefers large models - should select custom-large
        assert result == "custom-large:30b"

    def test_falls_back_to_smaller_tier_when_large_doesnt_fit(self, monkeypatch):
        """Test falls back to smaller tier when large models don't fit VRAM."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:30b": 18.0,  # Large tier - needs ~21.6GB VRAM
                "custom-medium:12b": 10.0,  # Medium tier - needs ~12GB VRAM
                "custom-small:8b": 5.0,  # Small tier - needs ~6GB VRAM
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        # Only 16GB VRAM - large model won't fit (needs 21.6GB with 20% overhead)
        result = settings.get_model_for_agent("architect", available_vram=16)

        # Should select medium tier
        assert result == "custom-medium:12b"

    def test_validator_prefers_tiny_models(self, monkeypatch):
        """Test validator role prefers tiny models."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
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

    def test_raises_when_no_models_installed(self, monkeypatch):
        """Test raises ValueError when no models installed."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {},
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        with pytest.raises(ValueError, match="No models installed"):
            settings.get_model_for_agent("writer", available_vram=24)

    def test_selects_smallest_when_nothing_fits_vram(self, monkeypatch):
        """Test selects smallest model as last resort when nothing fits VRAM."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "large-model:30b": 18.0,  # Needs 21.6GB
                "medium-model:12b": 10.0,  # Needs 12GB
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        # Only 8GB VRAM - nothing fits
        result = settings.get_model_for_agent("writer", available_vram=8)

        # Should select smallest as last resort with warning
        assert result == "medium-model:12b"

    def test_auto_selects_for_unknown_role(self, monkeypatch):
        """Test auto-selects for unknown agent role using default tier preferences."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-medium:12b": 10.0,
                "custom-small:8b": 5.0,
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"unknown_role": "auto"}

        result = settings.get_model_for_agent("unknown_role", available_vram=24)

        # Unknown role uses default tier preferences (medium first)
        assert result == "custom-medium:12b"


class TestGetAvailableVram:
    """Tests for get_available_vram function."""

    @patch("settings.subprocess.run")
    def test_returns_vram_from_nvidia_smi(self, mock_run):
        """Test returns VRAM from nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n"
        mock_run.return_value = mock_result

        vram = get_available_vram()

        # Should parse the MB value and convert to GB (8192 MB = 8 GB)
        assert vram == 8

    @patch("settings.subprocess.run")
    def test_returns_default_on_file_not_found(self, mock_run):
        """Test returns default when nvidia-smi not found."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        vram = get_available_vram()

        assert vram == 8  # Default

    @patch("settings.subprocess.run")
    def test_returns_default_on_timeout(self, mock_run):
        """Test returns default on subprocess timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        vram = get_available_vram()

        assert vram == 8  # Default

    @patch("settings.subprocess.run")
    def test_returns_default_on_parse_error(self, mock_run):
        """Test returns default when output can't be parsed."""
        mock_result = MagicMock()
        mock_result.stdout = "invalid\n"  # Not a number
        mock_run.return_value = mock_result

        vram = get_available_vram()

        assert vram == 8  # Default

    @patch("settings.subprocess.run")
    def test_returns_default_on_os_error(self, mock_run):
        """Test returns default on OSError."""
        mock_run.side_effect = OSError("Permission denied")

        vram = get_available_vram()

        assert vram == 8  # Default


class TestGetInstalledModels:
    """Tests for get_installed_models function."""

    @patch("settings.subprocess.run")
    def test_returns_model_list(self, mock_run):
        """Test returns list of installed models."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME\nmodel1:latest\nmodel2:7b\n"
        mock_run.return_value = mock_result

        models = get_installed_models()

        assert "model1:latest" in models
        assert "model2:7b" in models

    @patch("settings.subprocess.run")
    def test_returns_empty_on_file_not_found(self, mock_run):
        """Test returns empty list when ollama not found."""
        mock_run.side_effect = FileNotFoundError("ollama not found")

        models = get_installed_models()

        assert models == []

    @patch("settings.subprocess.run")
    def test_returns_empty_on_timeout(self, mock_run):
        """Test returns empty list on timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)

        models = get_installed_models()

        assert models == []

    @patch("settings.subprocess.run")
    def test_returns_empty_on_os_error(self, mock_run):
        """Test returns empty list on OSError."""
        mock_run.side_effect = OSError("Permission denied")

        models = get_installed_models()

        assert models == []


class TestGetInstalledModelsWithSizes:
    """Tests for get_installed_models_with_sizes function."""

    @patch("settings.subprocess.run")
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

    @patch("settings.subprocess.run")
    def test_returns_empty_on_file_not_found(self, mock_run):
        """Test returns empty dict when ollama not found."""
        mock_run.side_effect = FileNotFoundError("ollama not found")

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("settings.subprocess.run")
    def test_returns_empty_on_timeout(self, mock_run):
        """Test returns empty dict on timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("settings.subprocess.run")
    def test_returns_empty_on_os_error(self, mock_run):
        """Test returns empty dict on OSError."""
        mock_run.side_effect = OSError("Permission denied")

        models = get_installed_models_with_sizes()

        assert models == {}

    @patch("settings.subprocess.run")
    def test_parses_mb_sizes(self, mock_run):
        """Test parses MB size values and converts to GB."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME              SIZE\nsmall-model:1b    512 MB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        assert "small-model:1b" in models
        # 512 MB = 0.5 GB
        assert models["small-model:1b"] == 0.5

    @patch("settings.subprocess.run")
    def test_parses_combined_size_format(self, mock_run):
        """Test parses combined format like '4.1GB' without space."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME              SIZE\nmodel:tag     4.1GB\n"
        mock_run.return_value = mock_result

        models = get_installed_models_with_sizes()

        assert "model:tag" in models
        assert models["model:tag"] == 4.1

    @patch("settings.subprocess.run")
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

    @patch("settings.subprocess.run")
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

    @patch("settings.subprocess.run")
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

    @patch("settings.subprocess.run")
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
        with pytest.raises(ValueError, match="outline_variations_min.*cannot exceed"):
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
        with pytest.raises(ValueError, match="event_sentence_min_length.*cannot exceed"):
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
        with pytest.raises(ValueError, match="user_rating_min.*cannot exceed"):
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
        with pytest.raises(ValueError, match="world_gen_characters_min.*cannot exceed"):
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


class TestValidatorModelSelection:
    """Tests for validator model selection."""

    def test_selects_tagged_validator_model(self, monkeypatch):
        """Test selects tiny model tagged for validator role."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {"qwen3:0.6b": 0.5},
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "auto"}

        result = settings.get_model_for_agent("validator", available_vram=24)

        assert result == "qwen3:0.6b"

    def test_validator_falls_back_to_tiny_tier(self, monkeypatch):
        """Test validator falls back to tiny tier when no tagged model available."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-tiny:1b": 1.0,  # Tiny tier
                "custom-medium:12b": 10.0,  # Medium tier
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"validator": "auto"}

        result = settings.get_model_for_agent("validator", available_vram=24)

        # Validator prefers tiny tier
        assert result == "custom-tiny:1b"


class TestBackupCorruptedSettings:
    """Tests for _backup_corrupted_settings method."""

    def test_creates_backup_file(self, tmp_path, monkeypatch):
        """Test creates backup of corrupted settings."""
        settings_file = tmp_path / "settings.json"
        backup_file = tmp_path / "settings.json.bak"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        # Create a "corrupted" settings file
        settings_file.write_text("corrupted content")

        Settings._backup_corrupted_settings()

        assert backup_file.exists()
        assert backup_file.read_text() == "corrupted content"

    def test_handles_missing_file(self, tmp_path, monkeypatch):
        """Test handles missing settings file gracefully."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

        # Should not raise when file doesn't exist
        Settings._backup_corrupted_settings()

    def test_handles_backup_failure(self, tmp_path, monkeypatch):
        """Test handles OSError when backup fails."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("settings.SETTINGS_FILE", settings_file)

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
            "settings.get_installed_models_with_sizes",
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
            "settings.get_installed_models_with_sizes",
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


class TestSizeTierBasedModelSelection:
    """Tests for size tier-based model selection.

    Size tier boundaries:
    - LARGE: >= 20GB
    - MEDIUM: 8-20GB
    - SMALL: 3-8GB
    - TINY: < 3GB
    """

    def test_interviewer_prefers_small_tier(self, monkeypatch):
        """Test interviewer role prefers small tier models."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:70b": 25.0,  # Large tier (>= 20GB)
                "custom-medium:12b": 10.0,  # Medium tier (8-20GB)
                "custom-small:8b": 5.0,  # Small tier (3-8GB)
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"interviewer": "auto"}

        result = settings.get_model_for_agent("interviewer", available_vram=50)

        # Interviewer prefers small tier
        assert result == "custom-small:8b"

    def test_continuity_prefers_medium_tier(self, monkeypatch):
        """Test continuity role prefers medium tier models."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:70b": 25.0,  # Large tier (>= 20GB)
                "custom-medium:12b": 10.0,  # Medium tier (8-20GB)
                "custom-small:8b": 5.0,  # Small tier (3-8GB)
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"continuity": "auto"}

        result = settings.get_model_for_agent("continuity", available_vram=50)

        # Continuity prefers medium tier
        assert result == "custom-medium:12b"

    def test_editor_prefers_medium_tier(self, monkeypatch):
        """Test editor role prefers medium tier models."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                "custom-large:70b": 25.0,  # Large tier (>= 20GB)
                "custom-medium:12b": 10.0,  # Medium tier (8-20GB)
                "custom-small:8b": 5.0,  # Small tier (3-8GB)
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"editor": "auto"}

        result = settings.get_model_for_agent("editor", available_vram=50)

        # Editor prefers medium tier
        assert result == "custom-medium:12b"

    def test_fallback_when_no_tier_matches(self, monkeypatch):
        """Test fallback when no model in preferred tiers fits VRAM."""
        monkeypatch.setattr(
            "settings.get_installed_models_with_sizes",
            lambda timeout=None: {
                # Only tiny tier models available
                "tiny-model:1b": 2.0,  # Tiny tier (< 3GB)
            },
        )

        settings = Settings()
        settings.use_per_agent_models = True
        # Editor prefers medium tier, but only tiny model available
        settings.agent_models = {"editor": "auto"}

        # VRAM is enough for the tiny model (2GB * 1.2 = 2.4GB < 50GB)
        result = settings.get_model_for_agent("editor", available_vram=50)

        # Should fall back to the only available model
        assert result == "tiny-model:1b"
