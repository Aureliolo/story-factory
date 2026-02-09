"""Backup service - handles project backup and restore."""

import hashlib
import json
import logging
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.settings import STORIES_DIR, WORLDS_DIR, Settings

logger = logging.getLogger(__name__)

# Current backup format version for compatibility checking
BACKUP_FORMAT_VERSION = 1


@dataclass
class FileCheckResult:
    """Result of checking a single file in a backup."""

    filename: str
    exists: bool
    checksum_valid: bool | None = None
    checksum_expected: str = ""
    checksum_actual: str = ""
    error: str = ""


@dataclass
class BackupVerificationResult:
    """Result of verifying a backup's integrity."""

    valid: bool
    manifest_valid: bool = True
    files_complete: bool = True
    checksums_valid: bool = True
    sqlite_integrity_valid: bool = True
    json_parseable: bool = True
    version_compatible: bool = True
    file_results: list[FileCheckResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BackupVerifier:
    """Verifies backup file integrity before restoration."""

    def verify(self, backup_path: Path) -> BackupVerificationResult:
        """Verify a backup file's integrity.

        Performs the following checks:
        1. Manifest validity (metadata file exists and is valid JSON)
        2. File completeness (all listed files present)
        3. Checksum verification (SHA-256 matches if checksums present)
        4. SQLite integrity (database files pass integrity_check)
        5. JSON parseability (JSON files are valid)
        6. Version compatibility (format version is supported)

        Args:
            backup_path: Path to the backup zip file.

        Returns:
            BackupVerificationResult with detailed check results.
        """
        result = BackupVerificationResult(valid=True)

        if not backup_path.exists():
            result.valid = False
            result.errors.append(f"Backup file not found: {backup_path}")
            return result

        try:
            with zipfile.ZipFile(backup_path, "r") as zf:
                self._check_manifest(zf, result)
                if not result.manifest_valid:
                    result.valid = False
                    return result

                self._check_file_completeness(zf, result)
                self._check_checksums(zf, result)
                self._check_sqlite_integrity(zf, result)
                self._check_json_parseability(zf, result)
                self._check_version_compatibility(zf, result)

        except zipfile.BadZipFile as e:
            result.valid = False
            result.errors.append(f"Invalid zip file: {e}")
            return result

        # Set overall validity based on checks
        result.valid = (
            result.manifest_valid
            and result.files_complete
            and result.checksums_valid
            and result.sqlite_integrity_valid
            and result.json_parseable
            and result.version_compatible
        )

        if result.valid:
            logger.info(f"Backup verification passed: {backup_path}")
        else:
            logger.warning(f"Backup verification failed: {backup_path} - Errors: {result.errors}")

        return result

    def _check_manifest(self, zf: zipfile.ZipFile, result: BackupVerificationResult) -> None:
        """Check if manifest (metadata) exists, is valid JSON, and is a dict."""
        if "backup_metadata.json" not in zf.namelist():
            result.manifest_valid = False
            result.errors.append("Missing backup_metadata.json")
            return

        try:
            metadata_content = zf.read("backup_metadata.json")
            metadata = json.loads(metadata_content)
            if not isinstance(metadata, dict):
                result.manifest_valid = False
                result.errors.append("Metadata must be a JSON object (dict)")
                return
            logger.debug("Manifest check passed")
        except json.JSONDecodeError as e:
            result.manifest_valid = False
            result.errors.append(f"Invalid JSON in metadata: {e}")

    def _check_file_completeness(
        self, zf: zipfile.ZipFile, result: BackupVerificationResult
    ) -> None:
        """Check if all files listed in metadata are present."""
        try:
            metadata_content = zf.read("backup_metadata.json")
            metadata = json.loads(metadata_content)

            # Validate metadata structure
            if not isinstance(metadata, dict):
                result.files_complete = False
                result.errors.append("Metadata must be a JSON object")
                return

            files_list = metadata.get("files")
            if files_list is None:
                result.files_complete = False
                result.errors.append("Missing 'files' list in metadata")
                return
            if not isinstance(files_list, list):
                result.files_complete = False
                result.errors.append("Invalid 'files' field in metadata - must be a list")
                return

            for filename in files_list:
                file_result = FileCheckResult(
                    filename=filename,
                    exists=filename in zf.namelist(),
                )
                result.file_results.append(file_result)

                if not file_result.exists:
                    result.files_complete = False
                    result.errors.append(f"Missing file: {filename}")

            logger.debug(
                f"File completeness check: {len(files_list)} files, "
                f"complete={result.files_complete}"
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            result.files_complete = False
            result.errors.append(f"Error checking file completeness: {e}")

    def _check_checksums(self, zf: zipfile.ZipFile, result: BackupVerificationResult) -> None:
        """Verify file checksums if present in metadata."""
        try:
            metadata_content = zf.read("backup_metadata.json")
            metadata = json.loads(metadata_content)

            # Validate metadata structure
            if not isinstance(metadata, dict):
                result.checksums_valid = False
                result.errors.append("Metadata must be a JSON object for checksum verification")
                return

            checksums = metadata.get("checksums")
            if checksums is None:
                # No checksums in metadata - skip check but warn
                result.warnings.append("No checksums in backup metadata")
                logger.debug("No checksums in metadata, skipping checksum verification")
                return
            if not isinstance(checksums, dict):
                result.checksums_valid = False
                result.errors.append("Invalid 'checksums' field in metadata - must be a dict")
                return

            if not checksums:
                # Empty checksums dict - skip check but warn
                result.warnings.append("No checksums in backup metadata")
                logger.debug("No checksums in metadata, skipping checksum verification")
                return

            for filename, expected_checksum in checksums.items():
                if filename not in zf.namelist():
                    continue  # Already caught by completeness check

                # Calculate actual checksum
                file_content = zf.read(filename)
                actual_checksum = hashlib.sha256(file_content).hexdigest()

                # Find existing file result or create new one
                file_result = next(
                    (fr for fr in result.file_results if fr.filename == filename),
                    None,
                )
                if file_result is None:
                    file_result = FileCheckResult(filename=filename, exists=True)
                    result.file_results.append(file_result)

                file_result.checksum_expected = expected_checksum
                file_result.checksum_actual = actual_checksum
                file_result.checksum_valid = expected_checksum == actual_checksum

                if not file_result.checksum_valid:
                    result.checksums_valid = False
                    result.errors.append(
                        f"Checksum mismatch for {filename}: "
                        f"expected {expected_checksum[:16]}..., "
                        f"got {actual_checksum[:16]}..."
                    )

            logger.debug(
                f"Checksum verification: {len(checksums)} files, valid={result.checksums_valid}"
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            result.warnings.append(f"Error checking checksums: {e}")

    def _check_sqlite_integrity(
        self, zf: zipfile.ZipFile, result: BackupVerificationResult
    ) -> None:
        """Verify SQLite database files pass integrity check."""
        db_files = [f for f in zf.namelist() if f.endswith(".db")]

        for db_filename in db_files:
            temp_path: Path | None = None
            conn: sqlite3.Connection | None = None
            try:
                # Extract to temp file for integrity check
                db_content = zf.read(db_filename)
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    temp_file.write(db_content)

                conn = sqlite3.connect(str(temp_path))
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]

                if integrity_result != "ok":
                    result.sqlite_integrity_valid = False
                    result.errors.append(
                        f"SQLite integrity check failed for {db_filename}: {integrity_result}"
                    )
                else:
                    logger.debug(f"SQLite integrity check passed: {db_filename}")

            except sqlite3.DatabaseError as e:
                result.sqlite_integrity_valid = False
                result.errors.append(f"SQLite database error for {db_filename}: {e}")
            finally:
                if conn:
                    conn.close()
                if temp_path:
                    temp_path.unlink(missing_ok=True)

    def _check_json_parseability(
        self, zf: zipfile.ZipFile, result: BackupVerificationResult
    ) -> None:
        """Verify JSON files are valid."""
        json_files = [
            f for f in zf.namelist() if f.endswith(".json") and f != "backup_metadata.json"
        ]

        for json_filename in json_files:
            try:
                content = zf.read(json_filename)
                json.loads(content)
                logger.debug(f"JSON validation passed: {json_filename}")
            except json.JSONDecodeError as e:
                result.json_parseable = False
                result.errors.append(f"Invalid JSON in {json_filename}: {e}")

    def _check_version_compatibility(
        self, zf: zipfile.ZipFile, result: BackupVerificationResult
    ) -> None:
        """Check if backup format version is compatible."""
        try:
            metadata_content = zf.read("backup_metadata.json")
            metadata = json.loads(metadata_content)

            # Validate metadata structure
            if not isinstance(metadata, dict):
                result.version_compatible = False
                result.errors.append("Metadata must be a JSON object for version check")
                return

            backup_version = metadata.get("backup_format_version", BACKUP_FORMAT_VERSION)

            # Validate and coerce backup_version to int
            if not isinstance(backup_version, int):
                try:
                    backup_version = int(backup_version)
                except TypeError, ValueError:
                    result.version_compatible = False
                    result.errors.append(
                        f"Invalid backup_format_version type in metadata: "
                        f"expected int, got {type(backup_version).__name__}"
                    )
                    return

            if backup_version > BACKUP_FORMAT_VERSION:
                result.version_compatible = False
                result.errors.append(
                    f"Backup format version {backup_version} is newer than "
                    f"supported version {BACKUP_FORMAT_VERSION}"
                )
            else:
                logger.debug(f"Backup format version {backup_version} is compatible")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # No version info - assume compatible (older backup)
            result.warnings.append(f"Unable to read backup format version: {e}")


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
        raise ValueError(f"Invalid backup path: {path} is outside {base_dir}") from None


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

        # Create zip backup with checksums
        files_list: list[str] = []
        checksums: dict[str, str] = {}

        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add story state and calculate checksum
            story_content = story_path.read_bytes()
            story_filename = f"{project_id}.json"
            zf.writestr(story_filename, story_content)
            checksums[story_filename] = hashlib.sha256(story_content).hexdigest()
            files_list.append(story_filename)
            logger.debug(f"Added story state to backup: {story_path.name}")

            # Add world database if it exists
            if world_path.exists():
                world_content = world_path.read_bytes()
                world_filename = f"{project_id}.db"
                zf.writestr(world_filename, world_content)
                checksums[world_filename] = hashlib.sha256(world_content).hexdigest()
                files_list.append(world_filename)
                logger.debug(f"Added world database to backup: {world_path.name}")

            # Add metadata file with checksums and format version
            metadata = {
                "project_id": project_id,
                "project_name": project_name,
                "backup_created_at": datetime.now().isoformat(),
                "backup_format_version": BACKUP_FORMAT_VERSION,
                "files": files_list,
                "checksums": checksums,
            }

            zf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
            logger.debug("Added metadata with checksums to backup")

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

    def restore_backup(
        self,
        backup_filename: str,
        new_project_name: str = "",
        skip_verification: bool = False,
    ) -> str:
        """Restore a project from a backup.

        Extracts the backup zip and creates a new project with a new UUID.
        This allows restoring multiple times without conflicts.

        Args:
            backup_filename: Name of the backup zip file.
            new_project_name: Optional new name for the restored project.
                             If not provided, uses the original name with " (Restored)" suffix.
            skip_verification: If True, skip backup verification. Use with caution.

        Returns:
            The new project UUID.

        Raises:
            FileNotFoundError: If backup file doesn't exist.
            ValueError: If backup is invalid, verification fails, or path validation fails.
        """
        logger.info(f"Restoring backup: {backup_filename}")

        backup_dir = Path(self.settings.backup_folder)
        backup_path = backup_dir / backup_filename

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_filename}")

        # Validate backup path
        backup_path = _validate_backup_path(backup_path, backup_dir)

        # Run verification if enabled
        if self.settings.backup_verify_on_restore and not skip_verification:
            verifier = BackupVerifier()
            result = verifier.verify(backup_path)

            if not result.valid:
                error_summary = "; ".join(result.errors[:3])
                if len(result.errors) > 3:
                    error_summary += f" (+{len(result.errors) - 3} more errors)"
                raise ValueError(
                    f"Backup verification failed: {error_summary}. "
                    f"Use skip_verification=True to restore anyway."
                )

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
