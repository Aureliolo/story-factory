"""Tests for ProjectService."""

import pytest

from memory.world_database import WorldDatabase
from services.project_service import ProjectService, _validate_path


class TestProjectService:
    """Tests for ProjectService."""

    def test_create_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a new project."""
        # Monkeypatch the output directories
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)
        state, world_db = service.create_project("Test Story")

        assert isinstance(state.id, str) and len(state.id) > 0
        assert state.project_name == "Test Story"
        assert state.status == "interview"
        assert isinstance(world_db, WorldDatabase)
        assert world_db.db_path.exists()

    def test_create_project_default_name(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a project with default name."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)
        state, _ = service.create_project()

        assert state.project_name.startswith("New Story")

    def test_list_projects_empty(self, tmp_settings, monkeypatch, tmp_path):
        """Test listing projects when none exist."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        (tmp_path / "stories").mkdir(parents=True, exist_ok=True)

        service = ProjectService(tmp_settings)
        projects = service.list_projects()

        assert projects == []

    def test_list_projects(self, tmp_settings, monkeypatch, tmp_path):
        """Test listing existing projects."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create some projects
        service.create_project("Project 1")
        service.create_project("Project 2")

        projects = service.list_projects()

        assert len(projects) == 2
        names = [p.name for p in projects]
        assert "Project 1" in names
        assert "Project 2" in names

    def test_load_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test loading an existing project."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        state, _ = service.create_project("Test Project")
        project_id = state.id

        # Load it
        loaded_state, loaded_db = service.load_project(project_id)

        assert loaded_state.id == project_id
        assert loaded_state.project_name == "Test Project"
        assert isinstance(loaded_db, WorldDatabase)
        assert loaded_db.db_path.exists()

    def test_load_nonexistent_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test loading a project that doesn't exist."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        (tmp_path / "stories").mkdir(parents=True, exist_ok=True)

        service = ProjectService(tmp_settings)

        with pytest.raises(FileNotFoundError):
            service.load_project("nonexistent-id")

    def test_delete_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test deleting a project."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        state, world_db = service.create_project("To Delete")
        project_id = state.id

        # Verify it exists
        assert len(service.list_projects()) == 1

        # Close the database connection before deleting (Windows file locking)
        world_db.close()

        # Delete it
        result = service.delete_project(project_id)

        assert result is True
        assert len(service.list_projects()) == 0

    def test_duplicate_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test duplicating a project."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create original project
        original, _ = service.create_project("Original")

        # Duplicate it
        duplicate, dup_db = service.duplicate_project(original.id)

        assert duplicate.id != original.id
        assert duplicate.project_name == "Copy of Original"
        assert len(service.list_projects()) == 2


class TestValidatePath:
    """Tests for _validate_path function (path traversal prevention)."""

    def test_valid_path_within_base(self, tmp_path):
        """Test that valid paths within base directory are accepted."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        valid_path = base_dir / "file.txt"

        result = _validate_path(valid_path, base_dir)
        assert result == valid_path.resolve()

    def test_valid_nested_path(self, tmp_path):
        """Test that nested paths within base are accepted."""
        base_dir = tmp_path / "base"
        nested = base_dir / "sub" / "nested"
        nested.mkdir(parents=True)
        valid_path = nested / "file.txt"

        result = _validate_path(valid_path, base_dir)
        assert result == valid_path.resolve()

    def test_rejects_path_traversal_unix(self, tmp_path):
        """Test that Unix-style path traversal is rejected."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        malicious_path = base_dir / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match="outside"):
            _validate_path(malicious_path, base_dir)

    def test_rejects_path_traversal_windows(self, tmp_path):
        """Test that Windows-style path traversal is rejected."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        # Even on Unix, this tests the traversal logic
        malicious_path = base_dir / ".." / ".." / ".." / "windows" / "system32"

        with pytest.raises(ValueError, match="outside"):
            _validate_path(malicious_path, base_dir)

    def test_rejects_absolute_path_outside(self, tmp_path):
        """Test that absolute paths outside base are rejected."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        outside_path = tmp_path / "outside" / "file.txt"

        with pytest.raises(ValueError, match="outside"):
            _validate_path(outside_path, base_dir)

    def test_returns_resolved_path(self, tmp_path):
        """Test that the function returns resolved absolute paths."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        relative_path = base_dir / "." / "sub" / ".." / "file.txt"

        result = _validate_path(relative_path, base_dir)
        assert result.is_absolute()
        assert ".." not in str(result)
        assert result == (base_dir / "file.txt").resolve()


class TestProjectServiceAdditional:
    """Additional tests for edge cases and uncovered methods."""

    def test_list_projects_stories_dir_not_exists(self, tmp_settings, monkeypatch, tmp_path):
        """Test list_projects returns empty list when STORIES_DIR doesn't exist."""
        # Create service first (which creates directories)
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr("services.project_service.STORIES_DIR", nonexistent)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Now delete the directory to simulate it not existing
        import shutil

        if nonexistent.exists():
            shutil.rmtree(nonexistent)

        projects = service.list_projects()

        assert projects == []

    def test_list_projects_skips_corrupt_files(self, tmp_settings, monkeypatch, tmp_path):
        """Test list_projects skips corrupt JSON files."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True)
        monkeypatch.setattr("services.project_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        # Create a valid project first
        service = ProjectService(tmp_settings)
        service.create_project("Valid Project")

        # Create a corrupt JSON file
        corrupt_file = stories_dir / "corrupt.json"
        corrupt_file.write_text("{ not valid json", encoding="utf-8")

        # Should still list valid projects without error
        projects = service.list_projects()
        assert len(projects) == 1
        assert projects[0].name == "Valid Project"

    def test_load_legacy_project_without_world_db(self, tmp_settings, monkeypatch, tmp_path):
        """Test loading a legacy project without world_db_path creates one."""
        import json

        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True)
        worlds_dir = tmp_path / "worlds"
        worlds_dir.mkdir(parents=True)
        monkeypatch.setattr("services.project_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", worlds_dir)

        # Create a legacy project file without world_db_path
        project_id = "legacy-test-id"
        legacy_data = {
            "id": project_id,
            "project_name": "Legacy Project",
            "status": "interview",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            # No world_db_path field
        }
        project_file = stories_dir / f"{project_id}.json"
        project_file.write_text(json.dumps(legacy_data), encoding="utf-8")

        service = ProjectService(tmp_settings)
        state, world_db = service.load_project(project_id)

        # Should create world DB path
        assert isinstance(state.world_db_path, str) and len(state.world_db_path) > 0
        assert "legacy-test-id.db" in state.world_db_path
        assert isinstance(world_db, WorldDatabase)
        assert world_db.db_path.exists()

    def test_update_project_name(self, tmp_settings, monkeypatch, tmp_path):
        """Test updating a project's name."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        state, _ = service.create_project("Original Name")
        project_id = state.id

        # Update name
        updated = service.update_project_name(project_id, "New Name")

        assert updated.project_name == "New Name"

        # Verify persistence
        loaded, _ = service.load_project(project_id)
        assert loaded.project_name == "New Name"

    def test_get_project_path(self, tmp_settings, monkeypatch, tmp_path):
        """Test get_project_path returns correct path."""
        stories_dir = tmp_path / "stories"
        monkeypatch.setattr("services.project_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)
        path = service.get_project_path("test-uuid-123")

        assert path == stories_dir / "test-uuid-123.json"

    def test_get_world_db_path(self, tmp_settings, monkeypatch, tmp_path):
        """Test get_world_db_path returns correct path."""
        worlds_dir = tmp_path / "worlds"
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", worlds_dir)

        service = ProjectService(tmp_settings)
        path = service.get_world_db_path("test-uuid-123")

        assert path == worlds_dir / "test-uuid-123.db"


class TestProjectServiceExceptionHandling:
    """Tests for exception handling and edge cases in ProjectService."""

    def test_create_project_with_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a project with a template applies the template (lines 120-126)."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Use a built-in template
        state, world_db = service.create_project("Templated Story", template_id="fantasy-epic")

        # Verify project was created
        assert state.project_name == "Templated Story"
        assert state.status == "interview"
        # Template should have been applied - brief should have genre set
        assert state.brief is not None
        assert state.brief.genre == "Fantasy"

    def test_create_project_with_nonexistent_template(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a project with a nonexistent template still creates the project."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Use a nonexistent template ID
        state, world_db = service.create_project(
            "Story Without Template", template_id="nonexistent-template"
        )

        # Project should still be created
        assert state.project_name == "Story Without Template"
        assert state.status == "interview"
        # Brief should not have been set since template wasn't found
        assert state.brief is None

    def test_create_project_exception_reraises(self, tmp_settings, monkeypatch, tmp_path):
        """Test that create_project re-raises exceptions after logging (lines 133-135)."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Mock WorldDatabase to raise an exception
        def mock_world_db_init(*args, **kwargs):
            raise RuntimeError("Database creation failed")

        monkeypatch.setattr("services.project_service.WorldDatabase", mock_world_db_init)

        with pytest.raises(RuntimeError, match="Database creation failed"):
            service.create_project("Test Project")

    def test_load_project_exception_reraises(self, tmp_settings, monkeypatch, tmp_path):
        """Test that load_project re-raises non-FileNotFoundError exceptions (lines 187-189)."""
        import json

        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True)
        worlds_dir = tmp_path / "worlds"
        worlds_dir.mkdir(parents=True)
        monkeypatch.setattr("services.project_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", worlds_dir)

        # Create a valid project file
        project_id = "test-exception-id"
        valid_data = {
            "id": project_id,
            "project_name": "Test Project",
            "status": "interview",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "world_db_path": str(worlds_dir / f"{project_id}.db"),
        }
        project_file = stories_dir / f"{project_id}.json"
        project_file.write_text(json.dumps(valid_data), encoding="utf-8")

        service = ProjectService(tmp_settings)

        # Mock WorldDatabase to raise an exception (after JSON is loaded)
        def mock_world_db_init(*args, **kwargs):
            raise RuntimeError("World database initialization failed")

        monkeypatch.setattr("services.project_service.WorldDatabase", mock_world_db_init)

        with pytest.raises(RuntimeError, match="World database initialization failed"):
            service.load_project(project_id)

    def test_save_project_exception_reraises(self, tmp_settings, monkeypatch, tmp_path):
        """Test that save_project re-raises exceptions after logging (lines 216-218)."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        state, _ = service.create_project("Test Project")

        # Make the stories directory read-only to cause a write failure
        # Instead, mock the open function to raise an exception
        original_open = open

        def mock_open(*args, **kwargs):
            if "w" in str(args[1:]) or kwargs.get("mode", "") == "w":
                raise PermissionError("Cannot write to file")
            return original_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(PermissionError, match="Cannot write to file"):
            service.save_project(state)

    def test_delete_project_exception_reraises(self, tmp_settings, monkeypatch, tmp_path):
        """Test that delete_project re-raises exceptions after logging (lines 313-315)."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        state, world_db = service.create_project("Test Project")
        project_id = state.id

        # Close the database
        world_db.close()

        # Mock Path.unlink to raise an exception
        from pathlib import Path

        def mock_unlink(self):
            raise PermissionError("Cannot delete file")

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        with pytest.raises(PermissionError, match="Cannot delete file"):
            service.delete_project(project_id)

    def test_duplicate_project_exception_reraises(self, tmp_settings, monkeypatch, tmp_path):
        """Test that duplicate_project re-raises exceptions after logging (lines 366-368)."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        original, _ = service.create_project("Original Project")

        # Mock shutil.copy2 to raise an exception
        def mock_copy2(*args, **kwargs):
            raise OSError("Cannot copy file")

        monkeypatch.setattr("shutil.copy2", mock_copy2)

        with pytest.raises(OSError, match="Cannot copy file"):
            service.duplicate_project(original.id)

    def test_update_project_name_exception_reraises(self, tmp_settings, monkeypatch, tmp_path):
        """Test that update_project_name re-raises exceptions after logging (lines 393-395)."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create a project
        state, _ = service.create_project("Original Name")
        project_id = state.id

        # Mock save_project to raise an exception
        def mock_save(state):
            raise OSError("Cannot save project")

        monkeypatch.setattr(service, "save_project", mock_save)

        with pytest.raises(OSError, match="Cannot save project"):
            service.update_project_name(project_id, "New Name")

    def test_duplicate_project_with_custom_name(self, tmp_settings, monkeypatch, tmp_path):
        """Test duplicating a project with a custom name."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Create original project
        original, _ = service.create_project("Original")

        # Duplicate with custom name
        duplicate, _ = service.duplicate_project(original.id, "My Custom Copy")

        assert duplicate.id != original.id
        assert duplicate.project_name == "My Custom Copy"

    def test_delete_nonexistent_project_returns_false(self, tmp_settings, monkeypatch, tmp_path):
        """Test deleting a project that doesn't exist returns False."""
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        result = service.delete_project("nonexistent-project-id")

        assert result is False

    def test_load_project_path_traversal_rejected(self, tmp_settings, monkeypatch, tmp_path):
        """Test that load_project rejects path traversal attempts."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True)
        monkeypatch.setattr("services.project_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Attempt path traversal
        with pytest.raises(ValueError, match="outside"):
            service.load_project("../../../etc/passwd")

    def test_delete_project_path_traversal_rejected(self, tmp_settings, monkeypatch, tmp_path):
        """Test that delete_project rejects path traversal attempts."""
        stories_dir = tmp_path / "stories"
        stories_dir.mkdir(parents=True)
        monkeypatch.setattr("services.project_service.STORIES_DIR", stories_dir)
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)

        # Attempt path traversal
        with pytest.raises(ValueError, match="outside"):
            service.delete_project("../../../etc/passwd")
