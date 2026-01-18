"""Project service - handles project CRUD operations."""

import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from memory.story_state import StoryState
from memory.world_database import WorldDatabase
from settings import STORIES_DIR, WORLDS_DIR, Settings
from utils.validation import validate_not_empty, validate_not_none, validate_type

logger = logging.getLogger(__name__)


def _validate_path(path: Path, base_dir: Path) -> Path:
    """Validate that a path is within the base directory.

    Args:
        path: Path to validate
        base_dir: Base directory to check against

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path escapes base directory
    """
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        resolved.relative_to(base_resolved)
        return resolved
    except ValueError:
        raise ValueError(f"Invalid path: {path} is outside {base_dir}") from None


@dataclass
class ProjectSummary:
    """Summary information about a project for listing."""

    id: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    premise: str
    chapter_count: int
    word_count: int


class ProjectService:
    """Project CRUD operations.

    This service handles creating, loading, saving, listing, and deleting
    story projects. Each project consists of a StoryState JSON file and
    an associated WorldDatabase SQLite file.
    """

    def __init__(self, settings: Settings):
        """Initialize project service.

        Args:
            settings: Application settings.
        """
        validate_not_none(settings, "settings")
        validate_type(settings, "settings", Settings)
        logger.debug("Initializing ProjectService")
        self.settings = settings
        self._ensure_directories()
        logger.debug("ProjectService initialized successfully")

    def _ensure_directories(self) -> None:
        """Ensure output directories exist."""
        logger.debug("Ensuring output directories exist")
        STORIES_DIR.mkdir(parents=True, exist_ok=True)
        WORLDS_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directories ready: stories={STORIES_DIR}, worlds={WORLDS_DIR}")

    def create_project(
        self, name: str = "", template_id: str | None = None
    ) -> tuple[StoryState, WorldDatabase]:
        """Create a new project with story state and world database.

        Args:
            name: Optional project name. Defaults to timestamp-based name.
            template_id: Optional template ID to apply to the project.

        Returns:
            Tuple of (StoryState, WorldDatabase) for the new project.
        """
        logger.debug(f"create_project called: name={name}")
        now = datetime.now()
        project_id = str(uuid.uuid4())

        # Generate default name if not provided
        if not name:
            name = f"New Story - {now.strftime('%b %d, %Y %I:%M %p')}"

        try:
            # Create world database
            world_db_path = WORLDS_DIR / f"{project_id}.db"
            world_db = WorldDatabase(world_db_path)

            # Create story state
            story_state = StoryState(
                id=project_id,
                created_at=now,
                updated_at=now,
                project_name=name,
                world_db_path=str(world_db_path),
                status="interview",
            )

            # Apply template if provided
            if template_id:
                from services.template_service import TemplateService

                template_service = TemplateService(self.settings)
                template = template_service.get_template(template_id)
                if template:
                    template_service.apply_template_to_state(template, story_state, world_db)
                    logger.info(f"Applied template {template_id} to new project")

            # Save immediately so it appears in project list
            self.save_project(story_state)

            logger.info(f"Created new project: {project_id} - {name}")
            return story_state, world_db
        except Exception as e:
            logger.error(f"Failed to create project {name}: {e}", exc_info=True)
            raise

    def load_project(self, project_id: str) -> tuple[StoryState, WorldDatabase]:
        """Load an existing project.

        Args:
            project_id: The project UUID.

        Returns:
            Tuple of (StoryState, WorldDatabase).

        Raises:
            FileNotFoundError: If project doesn't exist.
            ValueError: If path validation fails.
        """
        validate_not_empty(project_id, "project_id")
        logger.debug(f"load_project called: project_id={project_id}")
        story_path = STORIES_DIR / f"{project_id}.json"

        try:
            # Validate path is within STORIES_DIR
            story_path = _validate_path(story_path, STORIES_DIR)

            if not story_path.exists():
                error_msg = f"Project not found: {project_id}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Load story state
            with open(story_path, encoding="utf-8") as f:
                data = json.load(f)

            story_state = StoryState.model_validate(data)

            # Load or create world database
            world_db_path = story_state.world_db_path
            if not world_db_path:
                # Legacy project without world DB - create one
                world_db_path = str(WORLDS_DIR / f"{project_id}.db")
                story_state.world_db_path = world_db_path
                self.save_project(story_state)
                logger.info(f"Created world DB for legacy project: {project_id}")

            # Validate world DB path
            world_db_path_obj = Path(world_db_path)
            world_db_path_obj = _validate_path(world_db_path_obj, WORLDS_DIR)
            world_db = WorldDatabase(world_db_path_obj)

            logger.info(f"Loaded project: {project_id} - {story_state.project_name}")
            return story_state, world_db
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load project {project_id}: {e}", exc_info=True)
            raise

    def save_project(self, state: StoryState) -> Path:
        """Save project to disk.

        Args:
            state: The story state to save.

        Returns:
            Path where the project was saved.
        """
        validate_not_none(state, "state")
        validate_type(state, "state", StoryState)
        logger.debug(f"save_project called: project_id={state.id}, name={state.project_name}")
        try:
            state.updated_at = datetime.now()
            if not state.last_saved:
                state.last_saved = datetime.now()

            output_path = STORIES_DIR / f"{state.id}.json"
            story_data = state.model_dump(mode="json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(story_data, f, indent=2, default=str)

            logger.debug(f"Saved project: {state.id} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save project {state.id}: {e}", exc_info=True)
            raise

    def list_projects(self) -> list[ProjectSummary]:
        """List all saved projects.

        Returns:
            List of ProjectSummary objects sorted by updated_at descending.
        """
        logger.debug("list_projects called")
        projects: list[ProjectSummary] = []

        if not STORIES_DIR.exists():
            logger.debug("Stories directory does not exist, returning empty list")
            return projects

        for filepath in STORIES_DIR.glob("*.json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)

                # Calculate word count
                chapters = data.get("chapters", [])
                word_count = sum(ch.get("word_count", 0) for ch in chapters)

                # Parse dates
                created_at = datetime.fromisoformat(
                    data.get("created_at", datetime.now().isoformat())
                )
                updated_at = datetime.fromisoformat(
                    data.get("updated_at", datetime.now().isoformat())
                )

                # Get premise from brief
                brief = data.get("brief")
                premise = brief.get("premise", "") if brief else ""

                projects.append(
                    ProjectSummary(
                        id=data.get("id", ""),
                        name=data.get("project_name", "Untitled"),
                        status=data.get("status", "unknown"),
                        created_at=created_at,
                        updated_at=updated_at,
                        premise=premise[:200] if premise else "",
                        chapter_count=len(chapters),
                        word_count=word_count,
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not read project file {filepath}: {e}")

        # Sort by updated_at descending (most recent first)
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        logger.info(f"Listed {len(projects)} projects")
        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and its associated world database.

        Args:
            project_id: The project UUID to delete.

        Returns:
            True if deleted successfully, False if project not found.

        Raises:
            ValueError: If path validation fails.
        """
        validate_not_empty(project_id, "project_id")
        logger.debug(f"delete_project called: project_id={project_id}")
        try:
            story_path = STORIES_DIR / f"{project_id}.json"
            world_path = WORLDS_DIR / f"{project_id}.db"

            # Validate paths are within their respective directories
            story_path = _validate_path(story_path, STORIES_DIR)
            world_path = _validate_path(world_path, WORLDS_DIR)

            deleted = False

            if story_path.exists():
                story_path.unlink()
                deleted = True
                logger.info(f"Deleted story file: {story_path}")

            if world_path.exists():
                world_path.unlink()
                logger.info(f"Deleted world database: {world_path}")

            if deleted:
                logger.info(f"Successfully deleted project: {project_id}")
            else:
                logger.warning(f"Project not found for deletion: {project_id}")

            return deleted
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}", exc_info=True)
            raise

    def duplicate_project(
        self, project_id: str, new_name: str = ""
    ) -> tuple[StoryState, WorldDatabase]:
        """Duplicate an existing project.

        Args:
            project_id: The project UUID to duplicate.
            new_name: Optional name for the duplicate. Defaults to "Copy of [original]".

        Returns:
            Tuple of (StoryState, WorldDatabase) for the new project.

        Raises:
            FileNotFoundError: If original project doesn't exist.
        """
        validate_not_empty(project_id, "project_id")
        logger.debug(f"duplicate_project called: project_id={project_id}, new_name={new_name}")
        try:
            # Load original
            original_state, original_world = self.load_project(project_id)

            # Create new project
            now = datetime.now()
            new_id = str(uuid.uuid4())

            if not new_name:
                new_name = f"Copy of {original_state.project_name}"

            # Copy world database
            new_world_path = WORLDS_DIR / f"{new_id}.db"
            if Path(original_state.world_db_path).exists():
                shutil.copy2(original_state.world_db_path, new_world_path)
                logger.debug(f"Copied world database to {new_world_path}")
            new_world = WorldDatabase(new_world_path)

            # Create new state based on original
            new_state = original_state.model_copy(deep=True)
            new_state.id = new_id
            new_state.project_name = new_name
            new_state.created_at = now
            new_state.updated_at = now
            new_state.last_saved = None
            new_state.world_db_path = str(new_world_path)

            # Save new project
            self.save_project(new_state)

            logger.info(f"Duplicated project {project_id} to {new_id} as '{new_name}'")
            return new_state, new_world
        except Exception as e:
            logger.error(f"Failed to duplicate project {project_id}: {e}", exc_info=True)
            raise

    def update_project_name(self, project_id: str, name: str) -> StoryState:
        """Update a project's name.

        Args:
            project_id: The project UUID.
            name: The new project name.

        Returns:
            Updated StoryState.

        Raises:
            FileNotFoundError: If project doesn't exist.
        """
        validate_not_empty(project_id, "project_id")
        validate_not_empty(name, "name")
        logger.debug(f"update_project_name called: project_id={project_id}, new_name={name}")
        try:
            state, _ = self.load_project(project_id)
            old_name = state.project_name
            state.project_name = name
            self.save_project(state)
            logger.info(f"Updated project name: {project_id} from '{old_name}' to '{name}'")
            return state
        except Exception as e:
            logger.error(f"Failed to update project name for {project_id}: {e}", exc_info=True)
            raise

    def get_project_path(self, project_id: str) -> Path:
        """Get the file path for a project.

        Args:
            project_id: The project UUID.

        Returns:
            Path to the project JSON file.
        """
        validate_not_empty(project_id, "project_id")
        logger.debug(f"get_project_path called: project_id={project_id}")
        return STORIES_DIR / f"{project_id}.json"

    def get_world_db_path(self, project_id: str) -> Path:
        """Get the world database path for a project.

        Args:
            project_id: The project UUID.

        Returns:
            Path to the project's SQLite database.
        """
        validate_not_empty(project_id, "project_id")
        logger.debug(f"get_world_db_path called: project_id={project_id}")
        return WORLDS_DIR / f"{project_id}.db"

    def get_project_by_name(self, name: str) -> ProjectSummary | None:
        """Find a project by its name.

        Args:
            name: The project name to search for.

        Returns:
            ProjectSummary if found, None otherwise.
        """
        validate_not_empty(name, "name")
        logger.debug(f"get_project_by_name called: name={name}")

        projects = self.list_projects()
        for project in projects:
            if project.name == name:
                logger.debug(f"Found project by name: {name} -> {project.id}")
                return project

        logger.debug(f"No project found with name: {name}")
        return None

    def delete_project_by_name(self, name: str) -> bool:
        """Delete a project by its name.

        Args:
            name: The project name to delete.

        Returns:
            True if deleted successfully, False if project not found.
        """
        validate_not_empty(name, "name")
        logger.debug(f"delete_project_by_name called: name={name}")

        project = self.get_project_by_name(name)
        if not project:
            logger.warning(f"Cannot delete - no project found with name: {name}")
            return False

        return self.delete_project(project.id)
