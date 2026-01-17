"""Tests for the settings module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from settings import (
    AGENT_ROLES,
    AVAILABLE_MODELS,
    Settings,
    get_available_vram,
    get_installed_models,
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
        model_id = list(AVAILABLE_MODELS.keys())[0]
        info = get_model_info(model_id)
        assert "name" in info
        assert "quality" in info
        assert "speed" in info

    def test_returns_default_for_unknown_model(self):
        """Should return default info for unknown models."""
        info = get_model_info("completely-unknown-model:99b")
        assert info["name"] == "completely-unknown-model:99b"
        assert info["quality"] == 5
        assert info["speed"] == 5


class TestAgentRoles:
    """Tests for agent role definitions."""

    def test_all_roles_defined(self):
        """Should have all required agent roles defined."""
        required_roles = ["interviewer", "architect", "writer", "editor", "continuity"]
        for role in required_roles:
            assert role in AGENT_ROLES
            assert "name" in AGENT_ROLES[role]
            assert "recommended_quality" in AGENT_ROLES[role]


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

    def test_selects_model_by_quality_when_not_set(self):
        """Test selects model by quality recommendation when not set."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {}  # No specific models set

        # Should select based on AGENT_ROLES recommended_quality
        result = settings.get_model_for_agent("writer")

        # Should return some model (exact model depends on registry)
        assert result is not None

    def test_selects_architect_model(self):
        """Test selects high-reasoning model for architect role."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        # With 24GB VRAM, should prefer Qwen3-30B (MoE, only 18GB needed)
        result = settings.get_model_for_agent("architect", available_vram=24)

        # Should select Qwen3-30B as the recommended architect model
        assert result == "huihui_ai/qwen3-abliterated:30b"

    def test_selects_architect_model_high_vram(self):
        """Test selects architect model with high VRAM."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        # With 48GB VRAM, still prefers Qwen3-30B as recommended
        result = settings.get_model_for_agent("architect", available_vram=48)

        # Qwen3-30B is first in preference list
        assert result == "huihui_ai/qwen3-abliterated:30b"

    def test_selects_architect_model_low_vram_falls_through(self):
        """Test architect falls through to auto-select with very low VRAM."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"architect": "auto"}

        # With only 8GB VRAM, architect models don't fit
        result = settings.get_model_for_agent("architect", available_vram=8)

        # Should fall through to auto-selection and pick something
        assert result is not None

    def test_auto_selects_speed_for_interviewer(self):
        """Test auto-selects for speed with interviewer role."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"interviewer": "auto"}

        result = settings.get_model_for_agent("interviewer", available_vram=24)

        # Should select something (speed-optimized selection for quality < 9)
        assert result is not None

    def test_auto_selects_quality_for_writer(self):
        """Test auto-selects for quality with writer role (high VRAM)."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}

        # With high VRAM (24GB), creative specialists fit
        # First checks creative specialists, but if those don't fit or aren't available,
        # falls through to quality-based sorting (required_quality >= 9)
        result = settings.get_model_for_agent("writer", available_vram=24)

        # Should select a quality-optimized model
        assert result is not None

    def test_auto_selects_quality_sorting_for_high_quality_role(self):
        """Test quality-based sorting when creative specialists don't fit but other high-quality models do."""
        import settings as settings_module

        s = Settings()
        s.use_per_agent_models = True
        s.agent_models = {"writer": "auto"}

        # Mock AVAILABLE_MODELS with a quality >= 9 model that fits low VRAM
        # Creative specialists are hardcoded and need 14GB, so we need a model
        # with quality >= 9 but lower VRAM to hit the quality sorting path
        mock_models = {
            "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0": {
                "name": "Celeste V1.9 12B",
                "release": "2025",
                "size_gb": 13,
                "vram_required": 14,  # Won't fit in 10GB
                "quality": 9,
                "speed": 7,
                "uncensored": True,
                "description": "Creative writing model",
            },
            "small-quality-model:7b": {
                "name": "Small Quality Model",
                "release": "2025",
                "size_gb": 5,
                "vram_required": 8,  # Fits in 10GB
                "quality": 9,
                "speed": 8,
                "uncensored": True,
                "description": "Small high-quality model",
            },
        }

        with (
            patch.object(settings_module, "AVAILABLE_MODELS", mock_models),
            patch.object(
                settings_module,
                "get_installed_models",
                return_value=[
                    "small-quality-model:7b",
                    "vanilj/mistral-nemo-12b-celeste-v1.9:Q8_0",
                ],
            ),
        ):
            # With 10GB VRAM: Celeste doesn't fit (14GB), but small-quality-model fits (8GB)
            result = s.get_model_for_agent("writer", available_vram=10)

        # Should select the small quality model via quality sorting
        assert result == "small-quality-model:7b"

    def test_falls_back_to_default_when_no_candidates(self):
        """Test falls back to default model when no candidates fit."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"writer": "auto"}
        settings.default_model = "fallback-model:7b"

        # With 0 VRAM, nothing fits
        result = settings.get_model_for_agent("writer", available_vram=0)

        assert result == "fallback-model:7b"

    def test_auto_selects_for_unknown_role(self):
        """Test auto-selects for unknown agent role."""
        settings = Settings()
        settings.use_per_agent_models = True
        settings.agent_models = {"unknown_role": "auto"}

        # Unknown role should still work, using default quality of 7
        result = settings.get_model_for_agent("unknown_role", available_vram=24)

        assert result is not None


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
