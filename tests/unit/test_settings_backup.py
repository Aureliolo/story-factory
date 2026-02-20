"""Tests for settings backup, recovery, and change logging."""

import json
import logging
from unittest.mock import patch

from src.settings import Settings
from src.settings._backup import (
    _create_settings_backup,
    _log_settings_changes,
    _recover_from_backup,
)


class TestCreateSettingsBackup:
    """Tests for _create_settings_backup()."""

    def test_creates_bak_file(self, tmp_path):
        """Should copy settings.json to settings.json.bak."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"ollama_url": "http://localhost:11434"}')

        result = _create_settings_backup(settings_file)

        assert result is True
        bak = settings_file.with_suffix(".json.bak")
        assert bak.exists()
        assert json.loads(bak.read_text())["ollama_url"] == "http://localhost:11434"

    def test_skips_when_file_missing(self, tmp_path):
        """Should return False when settings file doesn't exist."""
        settings_file = tmp_path / "settings.json"

        result = _create_settings_backup(settings_file)

        assert result is False
        assert not settings_file.with_suffix(".json.bak").exists()

    def test_skips_when_file_empty(self, tmp_path):
        """Should return False when settings file is empty."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("")

        result = _create_settings_backup(settings_file)

        assert result is False

    def test_handles_os_error_gracefully(self, tmp_path):
        """Should log warning and return False on copy failure."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"key": "value"}')

        with patch("src.settings._backup.shutil.copy2", side_effect=OSError("denied")):
            result = _create_settings_backup(settings_file)

        assert result is False

    def test_overwrites_existing_bak(self, tmp_path):
        """Should overwrite an existing .bak file with current content."""
        settings_file = tmp_path / "settings.json"
        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text('{"old": true}')
        settings_file.write_text('{"new": true}')

        _create_settings_backup(settings_file)

        assert json.loads(bak_file.read_text()) == {"new": True}


class TestRecoverFromBackup:
    """Tests for _recover_from_backup()."""

    def test_recovers_valid_backup(self, tmp_path):
        """Should return parsed dict from valid .bak file."""
        settings_file = tmp_path / "settings.json"
        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text('{"context_size": 16384, "log_level": "DEBUG"}')

        result = _recover_from_backup(settings_file)

        assert result == {"context_size": 16384, "log_level": "DEBUG"}

    def test_returns_none_when_no_backup(self, tmp_path):
        """Should return None when .bak file doesn't exist."""
        settings_file = tmp_path / "settings.json"

        result = _recover_from_backup(settings_file)

        assert result is None

    def test_returns_none_for_empty_backup(self, tmp_path):
        """Should return None when .bak file is empty."""
        settings_file = tmp_path / "settings.json"
        settings_file.with_suffix(".json.bak").write_text("")

        result = _recover_from_backup(settings_file)

        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        """Should return None when .bak contains invalid JSON."""
        settings_file = tmp_path / "settings.json"
        settings_file.with_suffix(".json.bak").write_text("not valid json {{{")

        result = _recover_from_backup(settings_file)

        assert result is None

    def test_returns_none_for_non_dict_json(self, tmp_path):
        """Should return None when .bak contains a JSON array instead of dict."""
        settings_file = tmp_path / "settings.json"
        settings_file.with_suffix(".json.bak").write_text("[1, 2, 3]")

        result = _recover_from_backup(settings_file)

        assert result is None

    def test_returns_none_on_os_error(self, tmp_path):
        """Should return None when backup file cannot be read (OSError)."""
        settings_file = tmp_path / "settings.json"
        settings_file.with_suffix(".json.bak").write_text('{"key": "value"}')

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            result = _recover_from_backup(settings_file)

        assert result is None


class TestLogSettingsChanges:
    """Tests for _log_settings_changes()."""

    def test_detects_added_keys(self):
        """Should log added keys."""
        original: dict[str, str] = {}
        final = {"new_key": "value"}

        count = _log_settings_changes(original, final, "test")

        assert count == 1

    def test_detects_removed_keys(self):
        """Should log removed keys."""
        original = {"old_key": "value"}
        final: dict[str, str] = {}

        count = _log_settings_changes(original, final, "test")

        assert count == 1

    def test_detects_changed_values(self):
        """Should log changed values."""
        original = {"key": "old"}
        final = {"key": "new"}

        count = _log_settings_changes(original, final, "test")

        assert count == 1

    def test_detects_nested_dict_changes(self):
        """Should log changes in nested dict values."""
        original = {"nested": {"a": 1, "b": 2}}
        final = {"nested": {"a": 1, "b": 3, "c": 4}}

        count = _log_settings_changes(original, final, "test")

        # b changed (2->3) + c added = 2 changes
        assert count == 2

    def test_detects_nested_dict_key_removal(self):
        """Should detect when a sub-key is removed from a nested dict."""
        original = {"nested": {"a": 1, "b": 2}}
        final = {"nested": {"a": 1}}

        count = _log_settings_changes(original, final, "test")

        assert count == 1

    def test_detects_type_change_dict_to_scalar(self):
        """Should detect when a value changes from dict to scalar."""
        original = {"key": {"nested": 1}}
        final = {"key": "flat_value"}

        count = _log_settings_changes(original, final, "test")

        assert count == 1

    def test_returns_zero_for_identical(self):
        """Should return 0 when dicts are identical."""
        data = {"key": "val", "nested": {"a": 1}}

        count = _log_settings_changes(data, data.copy(), "test")

        assert count == 0

    def test_logs_at_info_level(self, caplog):
        """Should log changes at INFO level."""
        with caplog.at_level(logging.INFO, logger="src.settings._backup"):
            _log_settings_changes({"k": "old"}, {"k": "new"}, "test-label")

        assert any("[test-label] changed k:" in r.message for r in caplog.records)


class TestSettingsLoadWithBackupRecovery:
    """Integration tests for Settings.load() backup recovery."""

    def test_recovers_from_backup_on_corrupt_file(self, tmp_path, monkeypatch):
        """Should recover user settings from .bak when primary is corrupt."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create a backup with custom settings
        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text(json.dumps({"context_size": 16384, "log_level": "DEBUG"}))

        # Corrupt the primary file
        settings_file.write_text("not valid json {{{")

        settings = Settings.load()

        # Should have recovered the custom values from backup
        assert settings.context_size == 16384
        assert settings.log_level == "DEBUG"

    def test_recovers_from_backup_on_missing_file(self, tmp_path, monkeypatch):
        """Should recover from .bak when primary is missing."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Create only a backup
        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text(json.dumps({"context_size": 65536}))

        settings = Settings.load()

        assert settings.context_size == 65536

    def test_falls_back_to_defaults_without_backup(self, tmp_path, monkeypatch):
        """Should use defaults when neither primary nor backup exist."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings.load()

        assert settings.context_size == 32768  # default
        assert settings.ollama_url == "http://localhost:11434"

    def test_recovers_from_backup_on_empty_file(self, tmp_path, monkeypatch):
        """Should recover from .bak when primary file is empty (0 bytes)."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text(json.dumps({"context_size": 16384}))

        # Empty file — common during interrupted writes
        settings_file.write_text("")

        settings = Settings.load()

        assert settings.context_size == 16384

    def test_recovers_from_backup_on_non_dict_json(self, tmp_path, monkeypatch):
        """Should recover from .bak when primary contains non-dict JSON."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text(json.dumps({"context_size": 16384}))
        settings_file.write_text("[1, 2, 3]")

        settings = Settings.load()

        assert settings.context_size == 16384

    def test_recovery_restores_primary_file(self, tmp_path, monkeypatch):
        """After recovery from .bak, the primary file should be restored."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        bak_file = settings_file.with_suffix(".json.bak")
        bak_file.write_text(json.dumps({"context_size": 16384}))
        settings_file.write_text("not valid json {{{")

        Settings.load()

        # Primary should now contain valid JSON with the recovered value
        restored = json.loads(settings_file.read_text())
        assert restored["context_size"] == 16384
        assert "ollama_url" in restored  # Merged defaults should be present

    def test_does_not_overwrite_bak_with_defaults(self, tmp_path, monkeypatch):
        """When primary is empty and no backup: defaults should be saved but
        an existing .bak should not be replaced with defaults."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Write custom settings to .bak
        bak_file = settings_file.with_suffix(".json.bak")
        bak_data = {"context_size": 16384, "log_level": "DEBUG"}
        bak_file.write_text(json.dumps(bak_data))

        # Primary is corrupt -> recovery kicks in -> loads from bak
        settings_file.write_text("{invalid")
        Settings.load()

        # .bak should still contain the original custom data (not defaults)
        recovered = json.loads(bak_file.read_text())
        assert recovered["context_size"] == 16384


class TestSettingsSaveCreatesBackup:
    """Tests for Settings.save() creating backups."""

    def test_save_creates_backup(self, tmp_path, monkeypatch):
        """save() should create a .bak copy before writing."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # First save to create the file
        settings = Settings()
        settings.save()

        original_data = json.loads(settings_file.read_text())

        # Modify and save again
        settings.context_size = 65536
        settings.save()

        # .bak should contain the original data (before the second save)
        bak = settings_file.with_suffix(".json.bak")
        assert bak.exists()
        bak_data = json.loads(bak.read_text())
        assert bak_data["context_size"] == original_data["context_size"]

    def test_save_works_without_existing_file(self, tmp_path, monkeypatch):
        """save() should work even when no prior file exists (no backup needed)."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        settings = Settings()
        settings.save()

        assert settings_file.exists()
        # No .bak should exist (nothing to back up on first save)
        assert not settings_file.with_suffix(".json.bak").exists()


class TestLoadLastProjectFix:
    """Tests for the _load_last_project() transient error fix in app.py."""

    def test_file_not_found_clears_project(self):
        """FileNotFoundError should still clear last_project_id."""
        from unittest.mock import MagicMock

        services = MagicMock()
        services.settings.last_project_id = "deleted-project-id"
        services.project.load_project.side_effect = FileNotFoundError("gone")

        from src.ui.app import StoryFactoryApp

        with patch.object(StoryFactoryApp, "build"):
            StoryFactoryApp(services)

        assert services.settings.last_project_id is None
        services.settings.save.assert_called_once()

    def test_transient_error_preserves_project(self):
        """Non-FileNotFoundError should NOT clear last_project_id."""
        from unittest.mock import MagicMock

        services = MagicMock()
        services.settings.last_project_id = "valid-project-id"
        services.project.load_project.side_effect = RuntimeError("db locked")

        from src.ui.app import StoryFactoryApp

        with patch.object(StoryFactoryApp, "build"):
            StoryFactoryApp(services)

        # last_project_id should NOT have been cleared
        assert services.settings.last_project_id == "valid-project-id"
        services.settings.save.assert_not_called()

    def test_permission_error_preserves_project(self):
        """PermissionError (file locked) should NOT clear last_project_id."""
        from unittest.mock import MagicMock

        services = MagicMock()
        services.settings.last_project_id = "valid-project-id"
        services.project.load_project.side_effect = PermissionError("file locked")

        from src.ui.app import StoryFactoryApp

        with patch.object(StoryFactoryApp, "build"):
            StoryFactoryApp(services)

        assert services.settings.last_project_id == "valid-project-id"
        services.settings.save.assert_not_called()


class TestSettingsLoadChangeLogging:
    """Tests for change logging during Settings.load()."""

    def test_load_logs_changes_on_merge(self, tmp_path, monkeypatch, caplog):
        """Settings.load() should log changes when merging new defaults."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Write a file with only one key — all others will be added
        settings_file.write_text(json.dumps({"context_size": 16384}))

        with caplog.at_level(logging.INFO, logger="src.settings._backup"):
            Settings.load()

        # Should have logged added keys
        assert any("[load]" in r.message and "added" in r.message for r in caplog.records)

    def test_load_logs_no_changes_for_complete_file(self, tmp_path, monkeypatch, caplog):
        """Settings.load() should log 'no changes' for a complete settings file."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("src.settings._settings.SETTINGS_FILE", settings_file)

        # Save a complete settings file first
        settings = Settings()
        settings.save()
        Settings.clear_cache()

        with caplog.at_level(logging.DEBUG, logger="src.settings._backup"):
            Settings.load()

        # Should have the "no changes" debug log
        assert any("no changes" in r.message for r in caplog.records)
