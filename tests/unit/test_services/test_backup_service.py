"""Tests for BackupService."""

import hashlib
import json
import sqlite3
import zipfile

import pytest

from src.services.backup_service import (
    BACKUP_FORMAT_VERSION,
    BackupService,
    BackupVerifier,
    _validate_backup_path,
)


class TestBackupService:
    """Tests for BackupService."""

    def test_create_backup(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a backup."""
        # Setup directories
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

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

        # Create a valid SQLite database for world
        world_path = worlds_dir / f"{project_id}.db"
        conn = sqlite3.connect(str(world_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

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

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

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

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)

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

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

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

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

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

        # Create a valid SQLite database for world
        world_path = worlds_dir / f"{project_id}.db"
        conn = sqlite3.connect(str(world_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

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

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

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

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)

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

    def test_list_backups_directory_does_not_exist(self, tmp_settings, tmp_path):
        """Test listing backups when backup directory doesn't exist (line 149)."""
        # Point to a directory that does not exist
        nonexistent_dir = tmp_path / "nonexistent_backups"
        tmp_settings.backup_folder = str(nonexistent_dir)

        # Manually prevent directory creation by not calling BackupService constructor
        # Instead, create service and then remove the directory
        service = BackupService(tmp_settings)
        # BackupService creates the directory, so remove it to test the condition
        nonexistent_dir.rmdir()

        backups = service.list_backups()
        assert backups == []

    def test_list_backups_missing_metadata_file(self, tmp_settings, tmp_path):
        """Test listing backups when a backup is missing metadata file (lines 156-157)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip file without metadata
        zip_path = backups_dir / "no_metadata.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        # Backup should be skipped, no entries returned
        assert len(backups) == 0

    def test_list_backups_missing_project_id_in_metadata(self, tmp_settings, tmp_path):
        """Test listing backups when metadata is missing project_id (lines 164-167)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with metadata missing project_id
        zip_path = backups_dir / "missing_project_id.zip"
        metadata = {
            "project_name": "Test Project",
            "backup_created_at": "2024-01-01T12:00:00",
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        # Backup should be skipped
        assert len(backups) == 0

    def test_list_backups_missing_project_name_in_metadata(self, tmp_settings, tmp_path):
        """Test listing backups when metadata is missing project_name (lines 170-173)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with metadata missing project_name
        zip_path = backups_dir / "missing_project_name.zip"
        metadata = {
            "project_id": "test-123",
            "backup_created_at": "2024-01-01T12:00:00",
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        # Backup should be skipped
        assert len(backups) == 0

    def test_list_backups_missing_created_at_uses_file_mtime(self, tmp_settings, tmp_path):
        """Test listing backups falls back to file mtime when created_at is missing (line 183)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with metadata missing backup_created_at
        zip_path = backups_dir / "missing_created_at.zip"
        metadata = {
            "project_id": "test-123",
            "project_name": "Test Project",
            # No backup_created_at field
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        # Backup should be included with mtime as created_at
        assert len(backups) == 1
        assert backups[0].project_id == "test-123"
        assert backups[0].project_name == "Test Project"
        # The created_at should be close to file mtime
        assert backups[0].created_at is not None

    def test_list_backups_bad_zip_file(self, tmp_settings, tmp_path):
        """Test listing backups handles BadZipFile exception (lines 194-195)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create an invalid zip file (just random content)
        bad_zip_path = backups_dir / "corrupted.zip"
        bad_zip_path.write_text("this is not a valid zip file")

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        # Should handle the error gracefully and return empty list
        assert len(backups) == 0

    def test_list_backups_json_decode_error(self, tmp_settings, tmp_path):
        """Test listing backups handles JSONDecodeError (lines 194-195)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with invalid JSON in metadata
        zip_path = backups_dir / "invalid_json.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", "this is { not valid json")
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)
        backups = service.list_backups()

        # Should handle the error gracefully and return empty list
        assert len(backups) == 0

    def test_restore_backup_missing_metadata(self, tmp_settings, tmp_path):
        """Test restoring backup without metadata file raises ValueError (line 235)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)
        tmp_settings.backup_verify_on_restore = False  # Skip verification for this test

        # Create a zip without metadata
        zip_path = backups_dir / "no_metadata.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)

        with pytest.raises(ValueError, match="missing metadata file"):
            service.restore_backup("no_metadata.zip")

    def test_restore_backup_missing_project_name_in_metadata(
        self, tmp_settings, monkeypatch, tmp_path
    ):
        """Test restoring backup with missing project_name raises ValueError (line 242)."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)
        backups_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with metadata missing project_name
        zip_path = backups_dir / "missing_project_name.zip"
        metadata = {
            "project_id": "test-123",
            # project_name is missing
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("test-123.json", json.dumps({"id": "test-123"}))

        service = BackupService(tmp_settings)

        with pytest.raises(ValueError, match="missing project_name"):
            service.restore_backup("missing_project_name.zip")

    def test_restore_backup_no_story_state_file(self, tmp_settings, tmp_path):
        """Test restoring backup without story state file raises ValueError (line 253)."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with metadata but no story json file
        zip_path = backups_dir / "no_story.zip"
        metadata = {
            "project_id": "test-123",
            "project_name": "Test Project",
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            # Only metadata, no story .json file

        service = BackupService(tmp_settings)

        with pytest.raises(ValueError, match="no story state file found"):
            service.restore_backup("no_story.zip")

    def test_get_backup_path(self, tmp_settings, tmp_path):
        """Test get_backup_path returns correct path (lines 340-341)."""
        backups_dir = tmp_path / "backups"
        tmp_settings.backup_folder = str(backups_dir)

        service = BackupService(tmp_settings)

        result = service.get_backup_path("test_backup.zip")

        expected = backups_dir / "test_backup.zip"
        assert result == expected


class TestBackupServiceGetMetadata:
    """Tests for get_backup_metadata method."""

    def test_get_backup_metadata_valid(self, tmp_settings, monkeypatch, tmp_path):
        """Test getting metadata from a valid backup."""
        stories_dir = tmp_path / "stories"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        stories_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project and backup
        project_id = "test-metadata-123"
        story_data = {"id": project_id, "project_name": "Test Metadata Project"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test Metadata Project")

        # Get metadata
        metadata = service.get_backup_metadata(backup_path.name)

        assert metadata is not None
        assert metadata["project_id"] == project_id
        assert metadata["project_name"] == "Test Metadata Project"
        assert "backup_created_at" in metadata

    def test_get_backup_metadata_not_found(self, tmp_settings, tmp_path):
        """Test getting metadata from non-existent backup."""
        backups_dir = tmp_path / "backups"
        tmp_settings.backup_folder = str(backups_dir)

        service = BackupService(tmp_settings)

        metadata = service.get_backup_metadata("nonexistent.zip")

        assert metadata is None

    def test_get_backup_metadata_missing_metadata_file(self, tmp_settings, tmp_path):
        """Test getting metadata from backup without metadata file."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip without metadata
        zip_path = backups_dir / "no_metadata.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("project.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)

        metadata = service.get_backup_metadata("no_metadata.zip")

        assert metadata is None

    def test_get_backup_metadata_bad_zip(self, tmp_settings, tmp_path):
        """Test getting metadata from corrupted zip file."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a corrupted zip
        bad_zip_path = backups_dir / "corrupted.zip"
        bad_zip_path.write_text("not a valid zip file")

        service = BackupService(tmp_settings)

        metadata = service.get_backup_metadata("corrupted.zip")

        assert metadata is None

    def test_get_backup_metadata_invalid_json(self, tmp_settings, tmp_path):
        """Test getting metadata from backup with invalid JSON metadata."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)
        tmp_settings.backup_folder = str(backups_dir)

        # Create a zip with invalid JSON metadata
        zip_path = backups_dir / "invalid_json.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", "{ not valid json }")

        service = BackupService(tmp_settings)

        metadata = service.get_backup_metadata("invalid_json.zip")

        assert metadata is None


class TestBackupVerifier:
    """Tests for BackupVerifier class."""

    def test_verify_valid_backup(self, tmp_settings, monkeypatch, tmp_path):
        """Test verification of a valid backup passes all checks."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project with actual SQLite database
        project_id = "test-verify"
        story_data = {"id": project_id, "project_name": "Test Verify"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        # Create a valid SQLite database
        world_path = worlds_dir / f"{project_id}.db"
        conn = sqlite3.connect(str(world_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        # Create backup
        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test Verify")

        # Verify backup
        verifier = BackupVerifier()
        result = verifier.verify(backup_path)

        assert result.valid is True
        assert result.manifest_valid is True
        assert result.files_complete is True
        assert result.checksums_valid is True
        assert result.sqlite_integrity_valid is True
        assert result.json_parseable is True
        assert result.version_compatible is True
        assert len(result.errors) == 0

    def test_verify_missing_manifest(self, tmp_path):
        """Test verification fails without metadata file."""
        # Create a zip without metadata
        zip_path = tmp_path / "no_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("project.json", json.dumps({"id": "test"}))

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        assert result.valid is False
        assert result.manifest_valid is False
        assert "Missing backup_metadata.json" in result.errors[0]

    def test_verify_checksum_mismatch(self, tmp_path):
        """Test verification fails on checksum mismatch."""
        zip_path = tmp_path / "bad_checksum.zip"

        # Create backup with correct file but wrong checksum
        content = b'{"id": "test", "project_name": "Test"}'
        wrong_checksum = "0" * 64  # Definitely wrong

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["test.json"],
            "checksums": {"test.json": wrong_checksum},
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("test.json", content)

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        assert result.valid is False
        assert result.checksums_valid is False
        assert any("Checksum mismatch" in e for e in result.errors)

    def test_verify_corrupted_sqlite(self, tmp_path):
        """Test verification fails on corrupted SQLite database."""
        zip_path = tmp_path / "corrupted_db.zip"

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["test.db"],
            "checksums": {},
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            # Write invalid SQLite data
            zf.writestr("test.db", b"this is not a valid sqlite database")

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        assert result.valid is False
        assert result.sqlite_integrity_valid is False
        assert any("SQLite" in e for e in result.errors)

    def test_verify_invalid_json(self, tmp_path):
        """Test verification fails on invalid JSON files."""
        zip_path = tmp_path / "invalid_json.zip"

        # Calculate checksum for invalid JSON
        invalid_json = b"{ not valid json }"
        checksum = hashlib.sha256(invalid_json).hexdigest()

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["project.json"],
            "checksums": {"project.json": checksum},
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", invalid_json)

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        assert result.valid is False
        assert result.json_parseable is False
        assert any("Invalid JSON" in e for e in result.errors)

    def test_verify_missing_files(self, tmp_path):
        """Test verification fails when files listed in metadata are missing."""
        zip_path = tmp_path / "missing_files.zip"

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["project.json", "missing.db"],  # missing.db doesn't exist
            "checksums": {},
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", json.dumps({"id": "test"}))
            # Note: missing.db is not added

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        assert result.valid is False
        assert result.files_complete is False
        assert any("Missing file" in e for e in result.errors)

    def test_verify_incompatible_version(self, tmp_path):
        """Test verification fails for future backup format version."""
        zip_path = tmp_path / "future_version.zip"

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION + 10,  # Future version
            "files": ["project.json"],
            "checksums": {},
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", json.dumps({"id": "test"}))

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        assert result.valid is False
        assert result.version_compatible is False
        assert any("newer than supported" in e for e in result.errors)

    def test_verify_nonexistent_file(self, tmp_path):
        """Test verification handles nonexistent backup file."""
        verifier = BackupVerifier()
        result = verifier.verify(tmp_path / "nonexistent.zip")

        assert result.valid is False
        assert any("not found" in e for e in result.errors)

    def test_verify_bad_zip_file(self, tmp_path):
        """Test verification handles corrupted zip file."""
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_text("not a zip file")

        verifier = BackupVerifier()
        result = verifier.verify(bad_zip)

        assert result.valid is False
        assert any("Invalid zip file" in e for e in result.errors)

    def test_verify_no_checksums_warns(self, tmp_path):
        """Test verification warns when no checksums in metadata."""
        zip_path = tmp_path / "no_checksums.zip"

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["project.json"],
            # No checksums field
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("project.json", json.dumps({"id": "test"}))

        verifier = BackupVerifier()
        result = verifier.verify(zip_path)

        # Should pass but with warning
        assert result.valid is True
        assert any("No checksums" in w for w in result.warnings)


class TestBackupServiceWithVerification:
    """Tests for BackupService with verification enabled."""

    def test_restore_with_verification(self, tmp_settings, monkeypatch, tmp_path):
        """Test restore runs verification when enabled."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)
        tmp_settings.backup_verify_on_restore = True

        # Create test project
        project_id = "test-verify-restore"
        story_data = {"id": project_id, "project_name": "Test"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test")

        # Restore should succeed with valid backup
        new_id = service.restore_backup(backup_path.name)
        assert new_id is not None

    def test_restore_fails_on_invalid_backup(self, tmp_settings, tmp_path):
        """Test restore fails when verification fails."""
        backups_dir = tmp_path / "backups"
        backups_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)
        tmp_settings.backup_verify_on_restore = True

        # Create a corrupted backup
        zip_path = backups_dir / "corrupted.zip"
        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["test.json"],
            "checksums": {"test.json": "wrong_checksum"},
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("test.json", json.dumps({"id": "test"}))

        service = BackupService(tmp_settings)

        with pytest.raises(ValueError, match="Backup verification failed"):
            service.restore_backup("corrupted.zip")

    def test_restore_skip_verification(self, tmp_settings, monkeypatch, tmp_path):
        """Test restore with skip_verification bypasses verification."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)
        backups_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)
        tmp_settings.backup_verify_on_restore = True

        # Create a backup with wrong checksum (would fail verification)
        zip_path = backups_dir / "bad_checksum.zip"
        story_content = json.dumps({"id": "test", "project_name": "Test"})

        metadata = {
            "project_id": "test",
            "project_name": "Test",
            "backup_format_version": BACKUP_FORMAT_VERSION,
            "files": ["test.json"],
            "checksums": {"test.json": "wrong_checksum"},  # Wrong checksum
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("backup_metadata.json", json.dumps(metadata))
            zf.writestr("test.json", story_content)

        service = BackupService(tmp_settings)

        # Should succeed when skipping verification
        new_id = service.restore_backup("bad_checksum.zip", skip_verification=True)
        assert new_id is not None

    def test_backup_includes_checksums(self, tmp_settings, monkeypatch, tmp_path):
        """Test that created backups include checksums in metadata."""
        stories_dir = tmp_path / "stories"
        worlds_dir = tmp_path / "worlds"
        backups_dir = tmp_path / "backups"

        monkeypatch.setattr("src.services.backup_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("src.services.backup_service.WORLDS_DIR", worlds_dir)

        stories_dir.mkdir(parents=True)
        worlds_dir.mkdir(parents=True)

        tmp_settings.backup_folder = str(backups_dir)

        # Create test project
        project_id = "test-checksums"
        story_data = {"id": project_id, "project_name": "Test Checksums"}
        story_path = stories_dir / f"{project_id}.json"
        story_path.write_text(json.dumps(story_data))

        service = BackupService(tmp_settings)
        backup_path = service.create_backup(project_id, "Test Checksums")

        # Verify metadata includes checksums
        with zipfile.ZipFile(backup_path, "r") as zf:
            metadata = json.loads(zf.read("backup_metadata.json"))

            assert "checksums" in metadata
            assert f"{project_id}.json" in metadata["checksums"]
            assert "backup_format_version" in metadata
            assert metadata["backup_format_version"] == BACKUP_FORMAT_VERSION

            # Verify checksum is correct
            story_content = zf.read(f"{project_id}.json")
            expected_checksum = hashlib.sha256(story_content).hexdigest()
            assert metadata["checksums"][f"{project_id}.json"] == expected_checksum
