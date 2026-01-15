"""Tests for ProjectService."""

import pytest

from services.project_service import ProjectService


class TestProjectService:
    """Tests for ProjectService."""

    def test_create_project(self, tmp_settings, monkeypatch, tmp_path):
        """Test creating a new project."""
        # Monkeypatch the output directories
        monkeypatch.setattr("services.project_service.STORIES_DIR", tmp_path / "stories")
        monkeypatch.setattr("services.project_service.WORLDS_DIR", tmp_path / "worlds")

        service = ProjectService(tmp_settings)
        state, world_db = service.create_project("Test Story")

        assert state.id is not None
        assert state.project_name == "Test Story"
        assert state.status == "interview"
        assert world_db is not None

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
        assert loaded_db is not None

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
