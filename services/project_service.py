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
        raise ValueError(f"Invalid path: {path} is outside {base_dir}")


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
        self.settings = settings
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure output directories exist."""
        STORIES_DIR.mkdir(parents=True, exist_ok=True)
        WORLDS_DIR.mkdir(parents=True, exist_ok=True)

    def create_project(self, name: str = "") -> tuple[StoryState, WorldDatabase]:
        """Create a new project with story state and world database.

        Args:
            name: Optional project name. Defaults to timestamp-based name.

        Returns:
            Tuple of (StoryState, WorldDatabase) for the new project.
        """
        now = datetime.now()
        project_id = str(uuid.uuid4())

        # Generate default name if not provided
        if not name:
            name = f"New Story - {now.strftime('%b %d, %Y %I:%M %p')}"

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

        # Save immediately so it appears in project list
        self.save_project(story_state)

        logger.info(f"Created new project: {project_id} - {name}")
        return story_state, world_db

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
        story_path = STORIES_DIR / f"{project_id}.json"
        # Validate path is within STORIES_DIR
        story_path = _validate_path(story_path, STORIES_DIR)

        if not story_path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")

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

        # Validate world DB path
        world_db_path_obj = Path(world_db_path)
        world_db_path_obj = _validate_path(world_db_path_obj, WORLDS_DIR)
        world_db = WorldDatabase(world_db_path_obj)

        logger.info(f"Loaded project: {project_id}")
        return story_state, world_db

    def save_project(self, state: StoryState) -> Path:
        """Save project to disk.

        Args:
            state: The story state to save.

        Returns:
            Path where the project was saved.
        """
        state.updated_at = datetime.now()
        if not state.last_saved:
            state.last_saved = datetime.now()

        output_path = STORIES_DIR / f"{state.id}.json"
        story_data = state.model_dump(mode="json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, default=str)

        logger.debug(f"Saved project: {state.id}")
        return output_path

    def list_projects(self) -> list[ProjectSummary]:
        """List all saved projects.

        Returns:
            List of ProjectSummary objects sorted by updated_at descending.
        """
        projects: list[ProjectSummary] = []

        if not STORIES_DIR.exists():
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

        return deleted

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

        logger.info(f"Duplicated project {project_id} to {new_id}")
        return new_state, new_world

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
        state, _ = self.load_project(project_id)
        state.project_name = name
        self.save_project(state)
        return state

    def get_project_path(self, project_id: str) -> Path:
        """Get the file path for a project.

        Args:
            project_id: The project UUID.

        Returns:
            Path to the project JSON file.
        """
        return STORIES_DIR / f"{project_id}.json"

    def get_world_db_path(self, project_id: str) -> Path:
        """Get the world database path for a project.

        Args:
            project_id: The project UUID.

        Returns:
            Path to the project's SQLite database.
        """
        return WORLDS_DIR / f"{project_id}.db"
