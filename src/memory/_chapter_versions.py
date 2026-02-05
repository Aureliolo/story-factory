"""Chapter version management using composition.

This module provides a ChapterVersionManager class that handles version history
for chapters, including saving, rollback, and comparison operations.
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.story_state import Chapter, ChapterVersion

logger = logging.getLogger(__name__)


class ChapterVersionManager:
    """Manages version history for a Chapter using composition.

    This class provides version management operations without modifying
    the Chapter model's interface. Access via the chapter's versions property.

    Example:
        chapter = Chapter(number=1, title="Introduction", outline="...")
        chapter.version_manager.save_version()
        chapter.version_manager.rollback(version_id)
    """

    def __init__(self, chapter: Chapter) -> None:
        """Initialize the version manager with a chapter reference.

        Args:
            chapter: The Chapter instance to manage versions for.
        """
        self._chapter = chapter

    def save_version(self, feedback: str = "") -> str:
        """Save the current chapter content as a new version.

        Args:
            feedback: Optional feedback that prompted this version.

        Returns:
            The ID of the newly created version.
        """
        # Import here to avoid circular imports
        from src.memory.story_state import ChapterVersion

        # Mark current version as not current (more efficient than looping all)
        current = self.get_current()
        if current:
            current.is_current = False

        # Create new version
        version_id = str(uuid.uuid4())
        version_number = len(self._chapter.versions) + 1

        new_version = ChapterVersion(
            id=version_id,
            content=self._chapter.content,
            word_count=self._chapter.word_count,
            feedback=feedback,
            version_number=version_number,
            is_current=True,
        )

        self._chapter.versions.append(new_version)
        self._chapter.current_version_id = version_id

        logger.debug(
            "Saved chapter %d version %d (id=%s)",
            self._chapter.number,
            version_number,
            version_id,
        )
        return version_id

    def rollback(self, version_id: str) -> bool:
        """Rollback to a previous version.

        Args:
            version_id: The ID of the version to rollback to.

        Returns:
            True if successful, False if version not found.
        """
        # Find the version
        target_version = self.get_version(version_id)

        if not target_version:
            logger.warning(
                "Version %s not found for chapter %d",
                version_id,
                self._chapter.number,
            )
            return False

        # Mark current version as not current (more efficient than looping all)
        current = self.get_current()
        if current:
            current.is_current = False

        # Restore content from target version
        self._chapter.content = target_version.content
        self._chapter.word_count = target_version.word_count
        target_version.is_current = True
        self._chapter.current_version_id = version_id

        logger.info(
            "Rolled back chapter %d to version %d",
            self._chapter.number,
            target_version.version_number,
        )
        return True

    def get_version(self, version_id: str) -> ChapterVersion | None:
        """Get a version by its ID.

        Args:
            version_id: The version ID to find.

        Returns:
            The version if found, None otherwise.
        """
        return next(
            (v for v in self._chapter.versions if v.id == version_id),
            None,
        )

    def get_current(self) -> ChapterVersion | None:
        """Get the current version.

        Returns:
            The current version if it exists, None otherwise.
        """
        if self._chapter.current_version_id:
            return self.get_version(self._chapter.current_version_id)
        return None

    def compare(self, version_id_a: str, version_id_b: str) -> dict[str, Any]:
        """Compare two versions.

        Args:
            version_id_a: First version ID.
            version_id_b: Second version ID.

        Returns:
            Dictionary with comparison data including word count differences.
            If either version is not found, returns {"error": "..."} instead.
        """
        version_a = self.get_version(version_id_a)
        version_b = self.get_version(version_id_b)

        if not version_a or not version_b:
            return {"error": "One or both versions not found"}

        return {
            "version_a": {
                "id": version_a.id,
                "version_number": version_a.version_number,
                "content": version_a.content,
                "word_count": version_a.word_count,
                "created_at": version_a.created_at.isoformat(),
                "feedback": version_a.feedback,
            },
            "version_b": {
                "id": version_b.id,
                "version_number": version_b.version_number,
                "content": version_b.content,
                "word_count": version_b.word_count,
                "created_at": version_b.created_at.isoformat(),
                "feedback": version_b.feedback,
            },
            "word_count_diff": version_b.word_count - version_a.word_count,
        }

    @property
    def count(self) -> int:
        """Get the total number of versions.

        Returns:
            Number of versions stored.
        """
        return len(self._chapter.versions)

    @property
    def all_versions(self) -> list[ChapterVersion]:
        """Get all versions.

        Returns:
            List of all chapter versions.
        """
        return self._chapter.versions
