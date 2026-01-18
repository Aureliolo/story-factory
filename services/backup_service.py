"""Backup service - handles project backup and restore."""

import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from settings import STORIES_DIR, WORLDS_DIR, Settings

logger = logging.getLogger(__name__)


def _validate_backup_path(path: Path, base_dir: Path) -> Path:
    """Validate that a backup path is within the backup directory.

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
        raise ValueError(f"Invalid backup path: {path} is outside {base_dir}")


@dataclass
class BackupInfo:
    """Information about a backup."""

    filename: str
    project_id: str
    project_name: str
    created_at: datetime
    size_bytes: int


class BackupService:
    """Project backup and restore operations.

    This service handles creating zip backups of projects (story state + world DB)
    and restoring them.
    """

    def __init__(self, settings: Settings):
        """Initialize backup service.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._ensure_backup_directory()

    def _ensure_backup_directory(self) -> None:
        """Ensure backup directory exists."""
        backup_dir = Path(self.settings.backup_folder)
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Backup directory: {backup_dir}")

    def create_backup(self, project_id: str, project_name: str = "") -> Path:
        """Create a backup of a project.

        Creates a zip file containing:
        - {project_id}.json (story state)
        - {project_id}.db (world database, if exists)

        Args:
            project_id: The project UUID to backup.
            project_name: Optional project name for backup filename.

        Returns:
            Path to the created backup zip file.

        Raises:
            FileNotFoundError: If project files don't exist.
            ValueError: If backup path is invalid.
        """
        logger.info(f"Creating backup for project: {project_id}")

        # Prepare paths
        story_path = STORIES_DIR / f"{project_id}.json"
        world_path = WORLDS_DIR / f"{project_id}.db"

        if not story_path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")

        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in project_name)
        safe_name = safe_name[:50] if safe_name else "backup"
        backup_filename = f"{safe_name}_{timestamp}.zip"

        backup_dir = Path(self.settings.backup_folder)
        backup_path = backup_dir / backup_filename

        # Validate backup path
        backup_path = _validate_backup_path(backup_path, backup_dir)

        # Create zip backup
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add story state
            zf.write(story_path, f"{project_id}.json")
            logger.debug(f"Added story state to backup: {story_path.name}")

            # Add world database if it exists
            if world_path.exists():
                zf.write(world_path, f"{project_id}.db")
                logger.debug(f"Added world database to backup: {world_path.name}")

            # Add metadata file for easier restoration
            files_list: list[str] = [f"{project_id}.json"]
            if world_path.exists():
                files_list.append(f"{project_id}.db")

            metadata = {
                "project_id": project_id,
                "project_name": project_name,
                "backup_created_at": datetime.now().isoformat(),
                "files": files_list,
            }

            zf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
            logger.debug("Added metadata to backup")

        logger.info(f"Backup created: {backup_path} ({backup_path.stat().st_size} bytes)")
        return backup_path

    def list_backups(self) -> list[BackupInfo]:
        """List all available backups.

        Returns:
            List of BackupInfo objects sorted by created_at descending.
        """
        logger.debug("Listing available backups")
        backups: list[BackupInfo] = []
        backup_dir = Path(self.settings.backup_folder)

        if not backup_dir.exists():
            return backups

        for backup_file in backup_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, "r") as zf:
                    # Try to read metadata
                    if "backup_metadata.json" not in zf.namelist():
                        logger.warning(f"Backup {backup_file.name} missing metadata file, skipping")
                        continue

                    metadata_content = zf.read("backup_metadata.json")
                    metadata = json.loads(metadata_content)

                    # Validate required metadata fields
                    if "project_id" not in metadata:
                        logger.warning(
                            f"Backup {backup_file.name} missing project_id in metadata, skipping"
                        )
                        continue

                    if "project_name" not in metadata:
                        logger.warning(
                            f"Backup {backup_file.name} missing project_name in metadata, skipping"
                        )
                        continue

                    project_id = metadata["project_id"]
                    project_name = metadata["project_name"]
                    created_at_str = metadata.get("backup_created_at")

                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str)
                    else:
                        # Fallback to file modification time
                        created_at = datetime.fromtimestamp(backup_file.stat().st_mtime)

                    backups.append(
                        BackupInfo(
                            filename=backup_file.name,
                            project_id=project_id,
                            project_name=project_name,
                            created_at=created_at,
                            size_bytes=backup_file.stat().st_size,
                        )
                    )
            except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not read backup file {backup_file}: {e}")

        # Sort by created_at descending (most recent first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        logger.debug(f"Found {len(backups)} backups")
        return backups

    def restore_backup(self, backup_filename: str, new_project_name: str = "") -> str:
        """Restore a project from a backup.

        Extracts the backup zip and creates a new project with a new UUID.
        This allows restoring multiple times without conflicts.

        Args:
            backup_filename: Name of the backup zip file.
            new_project_name: Optional new name for the restored project.
                             If not provided, uses the original name with " (Restored)" suffix.

        Returns:
            The new project UUID.

        Raises:
            FileNotFoundError: If backup file doesn't exist.
            ValueError: If backup is invalid or path validation fails.
        """
        logger.info(f"Restoring backup: {backup_filename}")

        backup_dir = Path(self.settings.backup_folder)
        backup_path = backup_dir / backup_filename

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_filename}")

        # Validate backup path
        backup_path = _validate_backup_path(backup_path, backup_dir)

        # Extract backup contents
        with zipfile.ZipFile(backup_path, "r") as zf:
            # Read and validate metadata
            if "backup_metadata.json" not in zf.namelist():
                raise ValueError(f"Invalid backup: {backup_filename} is missing metadata file")

            metadata_content = zf.read("backup_metadata.json")
            metadata = json.loads(metadata_content)

            # Validate required metadata fields
            if "project_name" not in metadata:
                raise ValueError(
                    f"Invalid backup: {backup_filename} is missing project_name in metadata"
                )

            original_name = metadata["project_name"]

            # Find the story state file
            story_files = [
                f for f in zf.namelist() if f.endswith(".json") and f != "backup_metadata.json"
            ]
            if not story_files:
                raise ValueError(f"Invalid backup: no story state file found in {backup_filename}")

            story_filename = story_files[0]

            # Extract and load story state
            story_content = zf.read(story_filename)
            story_data = json.loads(story_content)

            # Generate new project ID
            import uuid

            new_project_id = str(uuid.uuid4())
            logger.debug(f"Restoring with new project ID: {new_project_id}")

            # Update story state with new ID and name
            story_data["id"] = new_project_id
            if new_project_name:
                story_data["project_name"] = new_project_name
            else:
                story_data["project_name"] = f"{original_name} (Restored)"

            # Update timestamps
            now = datetime.now()
            story_data["created_at"] = now.isoformat()
            story_data["updated_at"] = now.isoformat()
            story_data["last_saved"] = None

            # Update world database path
            new_world_path = WORLDS_DIR / f"{new_project_id}.db"
            story_data["world_db_path"] = str(new_world_path)

            # Save restored story state
            new_story_path = STORIES_DIR / f"{new_project_id}.json"
            STORIES_DIR.mkdir(parents=True, exist_ok=True)
            with open(new_story_path, "w", encoding="utf-8") as f:
                json.dump(story_data, f, indent=2, default=str)
            logger.debug(f"Restored story state to: {new_story_path}")

            # Extract world database if it exists
            db_files = [fname for fname in zf.namelist() if fname.endswith(".db")]
            if db_files:
                db_filename = db_files[0]
                WORLDS_DIR.mkdir(parents=True, exist_ok=True)
                with open(new_world_path, "wb") as db_file:
                    db_file.write(zf.read(db_filename))
                logger.debug(f"Restored world database to: {new_world_path}")

        logger.info(f"Backup restored successfully as project: {new_project_id}")
        return new_project_id

    def delete_backup(self, backup_filename: str) -> bool:
        """Delete a backup file.

        Args:
            backup_filename: Name of the backup zip file to delete.

        Returns:
            True if deleted successfully, False if backup not found.

        Raises:
            ValueError: If path validation fails.
        """
        logger.info(f"Deleting backup: {backup_filename}")

        backup_dir = Path(self.settings.backup_folder)
        backup_path = backup_dir / backup_filename

        # Validate backup path
        backup_path = _validate_backup_path(backup_path, backup_dir)

        if not backup_path.exists():
            logger.warning(f"Backup file not found: {backup_filename}")
            return False

        backup_path.unlink()
        logger.info(f"Deleted backup: {backup_filename}")
        return True

    def get_backup_path(self, backup_filename: str) -> Path:
        """Get the full path to a backup file.

        Args:
            backup_filename: Name of the backup file.

        Returns:
            Path to the backup file.
        """
        backup_dir = Path(self.settings.backup_folder)
        return backup_dir / backup_filename

    def get_backup_metadata(self, backup_filename: str) -> dict | None:
        """Get metadata from a backup file.

        Args:
            backup_filename: Name of the backup zip file.

        Returns:
            Dictionary with backup metadata, or None if invalid/not found.
        """
        logger.debug(f"Getting metadata for backup: {backup_filename}")

        backup_dir = Path(self.settings.backup_folder)
        backup_path = backup_dir / backup_filename

        if not backup_path.exists():
            logger.warning(f"Backup file not found: {backup_filename}")
            return None

        try:
            with zipfile.ZipFile(backup_path, "r") as zf:
                if "backup_metadata.json" not in zf.namelist():
                    logger.warning(f"Backup {backup_filename} missing metadata file")
                    return None

                metadata_content = zf.read("backup_metadata.json")
                metadata: dict = json.loads(metadata_content)
                logger.debug(f"Retrieved metadata for backup {backup_filename}")
                return metadata
        except (zipfile.BadZipFile, json.JSONDecodeError) as e:
            logger.warning(f"Could not read backup metadata from {backup_filename}: {e}")
            return None
