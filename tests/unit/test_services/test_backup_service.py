"""Tests for BackupService."""

import json
import zipfile

import pytest

from services.backup_service import BackupService, _validate_backup_path


class TestBackupService:
    """Tests for BackupService."""

    def test_create_backup(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a backup."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)

        # Create test project files
        project_id = "test-project-123"
        story_data = {
            "id": project_id,
            "project_name": "Test Project",
            "status": "writing",
            "chapters": [],
        }
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        world_path = worlds_dir / f"{project_id}.db"
        world_path.write_text("mock db content")

        # Update settings
        tmp_settings.backup_folder = str(backups_dir)

        # Create backup
        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test Project")

        # Verify backup was created
        assert backup_path.exists()
        assert backup_path.suffix == ".zip"

        # Verify backup contents
        with zipfile.ZipFile(backup_path, "r") as zf:
            names = zf.namelist()
            assert f"{project_id}.json" in names
            assert f"{project_id}.db" in names
            assert "backup_metadata.json" in names

            # Verify metadata
            metadata = json.loads(zf.read("backup_metadata.json"))
            assert metadata["project_id"] == project_id
            assert metadata["project_name"] == "Test Project"

    def test_create_backup_without_world_db(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a backup when world database doesn't exist."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)

        # Create test project file (no world DB)
        project_id = "test-project-456"
        story_data = {"id": project_id, "project_name": "Test Project"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        # Update settings
        tmp_settings.backup_folder = str(backups_dir)

        # Create backup
        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test Project")

        # Verify backup contents
        with zipfile.ZipFile(backup_path, "r") as zf:
            names = zf.namelist()
            assert f"{project_id}.json" in names
            assert f"{project_id}.db" not in names

    def test_create_backup_project_not_found(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a backup for non-existent project."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)

        stories_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        service = BackupService(tmp_settings)

        with pytest.raises(FileNotFoundError):
            service.create_backup("nonexistent-id", "Test")

    def test_list_backups(self, tmp_settings, monkeypatch, tmp_path):
        """Test listing backups."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)
        backups_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project and backup
        project_id = "test-project-789"
        story_data = {"id": project_id, "project_name": "Test Project"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        service = BackupService(tmp_settings)
        service.create_backup(project_id, "Test Project")

        # List backups
        backups = service.list_backups()

        assert len(backups) == 1
        assert backups[0].project_id == project_id
        assert backups[0].project_name == "Test Project"
        assert backups[0].size_bytes > 0

    def test_list_backups_empty(self, tmp_settings, tmp_path):
        """Test listing backups when none exist."""
        backups_dir = tmp_path / "backups"
        tmp_settings.backup_folder = str(backups_dir)

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        assert backups == []

    def test_restore_backup(self, tmp_settings, monkeypatch, tmp_path):
        """Test restoring a backup."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)
        backups_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project and backup
        project_id = "original-project"
        story_data = {
            "id": project_id,
            "project_name": "Original Project",
            "status": "writing",
        }
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        world_path = worlds_dir / f"{project_id}.db"
        world_path.write_text("world data")

        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Original Project")

        # Delete original files to simulate restoration scenario
        story_path.unlink()
        world_path.unlink()

        # Restore backup
        new_project_id = service.restore_backup(backup_path.name)

        # Verify restored project
        assert new_project_id != project_id  # Should have new UUID

        restored_story_path = stories_dir / f"{new_project_id}.json"
        assert restored_story_path.exists()

        restored_data = json.loads(restored_story_path.read_text())
        assert restored_data["id"] == new_project_id
        assert restored_data["project_name"] == "Original Project (Restored)"

        restored_world_path = worlds_dir / f"{new_project_id}.db"
        assert restored_world_path.exists()

    def test_restore_backup_with_custom_name(self, tmp_settings, monkeypatch, tmp_path):
        """Test restoring a backup with a custom name."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project and backup
        project_id = "test-restore"
        story_data = {"id": project_id, "project_name": "Test"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test")

        # Restore with custom name
        new_project_id = service.restore_backup(backup_path.name, "Custom Name")

        restored_story_path = stories_dir / f"{new_project_id}.json"
        restored_data = json.loads(restored_story_path.read_text())
        assert restored_data["project_name"] == "Custom Name"

    def test_restore_backup_not_found(self, tmp_settings, tmp_path):
        """Test restoring a backup that doesn't exist."""
        tmp_settings.backup_folder = str(tmp_path / "backups")

        service = BackupService(tmp_settings)

        with pytest.raises(FileNotFoundError):
            service.restore_backup("nonexistent.zip")

    def test_delete_backup(self, tmp_settings, monkeypatch, tmp_path):
        """Test deleting a backup."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("services.backup_service.STORIES_DIR", stories_dir)

        stories_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project and backup
        project_id = "test-delete"
        story_data = {"id": project_id, "project_name": "Test"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test")

        assert backup_path.exists()

        # Delete backup
        result = service.delete_backup(backup_path.name)

        assert result is True
        assert not backup_path.exists()

    def test_delete_backup_not_found(self, tmp_settings, tmp_path):
        """Test deleting a backup that doesn't exist."""
        tmp_settings.backup_folder = str(tmp_path / "backups")

        service = BackupService(tmp_settings)
        result = service.delete_backup("nonexistent.zip")

        assert result is False

    def test_validate_backup_path(self, tmp_path):
        """Test backup path validation."""
        base_dir = tmp_path / "backups"
        base_dir.mkdir()

        # Valid path
        valid_path = base_dir / "backup.zip"
        result = _validate_backup_path(valid_path, base_dir)
        assert result == valid_path.resolve()

        # Invalid path (outside base dir)
        invalid_path = tmp_path / "other" / "backup.zip"
        with pytest.raises(ValueError, match="outside"):
            _validate_backup_path(invalid_path, base_dir)
