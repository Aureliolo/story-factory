"""Tests for Chapter version history functionality."""

import pytest

from src.memory._chapter_versions import ChapterVersionManager
from src.memory.story_state import Chapter, ChapterVersion


class TestChapterVersion:
    """Tests for ChapterVersion model."""

    def test_create_version(self):
        """Test creating a chapter version."""
        version = ChapterVersion(
            id="v1",
            content="This is test content.",
            word_count=4,
            feedback="Make it more exciting",
            version_number=1,
            is_current=True,
        )
        assert version.id == "v1"
        assert version.content == "This is test content."
        assert version.word_count == 4
        assert version.feedback == "Make it more exciting"
        assert version.version_number == 1
        assert version.is_current is True

    def test_version_defaults(self):
        """Test version with default values."""
        version = ChapterVersion(
            id="v2",
            content="Content",
        )
        assert version.word_count == 0
        assert version.feedback == ""
        assert version.version_number == 1
        assert version.is_current is False
        assert version.created_at is not None


class TestChapterVersionManagement:
    """Tests for Chapter version management methods."""

    @pytest.fixture
    def sample_chapter(self):
        """Create a sample chapter with content."""
        return Chapter(
            number=1,
            title="Test Chapter",
            outline="Test outline",
            content="Original chapter content here.",
            word_count=4,
        )

    def test_save_current_as_version(self, sample_chapter):
        """Test saving current content as a version."""
        version_id = sample_chapter.save_current_as_version(feedback="Needs more action")

        assert len(sample_chapter.versions) == 1
        assert sample_chapter.current_version_id == version_id

        version = sample_chapter.versions[0]
        assert version.id == version_id
        assert version.content == "Original chapter content here."
        assert version.word_count == 4
        assert version.feedback == "Needs more action"
        assert version.version_number == 1
        assert version.is_current is True

    def test_save_multiple_versions(self, sample_chapter):
        """Test saving multiple versions."""
        # Save first version
        v1_id = sample_chapter.save_current_as_version(feedback="First feedback")

        # Update content and save second version
        sample_chapter.content = "Updated content with more action!"
        sample_chapter.word_count = 5
        v2_id = sample_chapter.save_current_as_version(feedback="Second feedback")

        assert len(sample_chapter.versions) == 2
        assert sample_chapter.current_version_id == v2_id

        # Check first version is marked as not current
        v1 = sample_chapter.get_version_by_id(v1_id)
        assert v1 is not None
        assert v1.is_current is False
        assert v1.version_number == 1

        # Check second version is current
        v2 = sample_chapter.get_version_by_id(v2_id)
        assert v2 is not None
        assert v2.is_current is True
        assert v2.version_number == 2

    def test_rollback_to_version(self, sample_chapter):
        """Test rolling back to a previous version."""
        # Save original
        v1_id = sample_chapter.save_current_as_version()

        # Update and save new version
        sample_chapter.content = "New content"
        sample_chapter.word_count = 2
        v2_id = sample_chapter.save_current_as_version()

        assert sample_chapter.content == "New content"
        assert sample_chapter.current_version_id == v2_id

        # Rollback to v1
        success = sample_chapter.rollback_to_version(v1_id)
        assert success is True
        assert sample_chapter.content == "Original chapter content here."
        assert sample_chapter.word_count == 4
        assert sample_chapter.current_version_id == v1_id

        # Check v1 is marked current
        v1 = sample_chapter.get_version_by_id(v1_id)
        assert v1 is not None
        assert v1.is_current is True

        # Check v2 is not current
        v2 = sample_chapter.get_version_by_id(v2_id)
        assert v2 is not None
        assert v2.is_current is False

    def test_rollback_to_nonexistent_version(self, sample_chapter):
        """Test rollback to non-existent version fails gracefully."""
        success = sample_chapter.rollback_to_version("nonexistent-id")
        assert success is False
        # Content should remain unchanged
        assert sample_chapter.content == "Original chapter content here."

    def test_get_version_by_id(self, sample_chapter):
        """Test getting a version by ID."""
        v_id = sample_chapter.save_current_as_version()

        version = sample_chapter.get_version_by_id(v_id)
        assert version is not None
        assert version.id == v_id

        # Non-existent ID
        none_version = sample_chapter.get_version_by_id("fake-id")
        assert none_version is None

    def test_get_current_version(self, sample_chapter):
        """Test getting the current version."""
        # No versions yet
        assert sample_chapter.get_current_version() is None

        # Save a version
        v_id = sample_chapter.save_current_as_version()
        current = sample_chapter.get_current_version()
        assert current is not None
        assert current.id == v_id
        assert current.is_current is True

    def test_compare_versions(self, sample_chapter):
        """Test comparing two versions."""
        # Save first version
        v1_id = sample_chapter.save_current_as_version(feedback="Make it longer")

        # Update and save second version
        sample_chapter.content = "Much longer updated content with lots of details here."
        sample_chapter.word_count = 9
        v2_id = sample_chapter.save_current_as_version()

        # Compare versions
        comparison = sample_chapter.compare_versions(v1_id, v2_id)

        assert "version_a" in comparison
        assert "version_b" in comparison
        assert comparison["version_a"]["id"] == v1_id
        assert comparison["version_a"]["version_number"] == 1
        assert comparison["version_a"]["word_count"] == 4
        assert comparison["version_a"]["feedback"] == "Make it longer"

        assert comparison["version_b"]["id"] == v2_id
        assert comparison["version_b"]["version_number"] == 2
        assert comparison["version_b"]["word_count"] == 9

        assert comparison["word_count_diff"] == 5  # 9 - 4

    def test_compare_versions_with_nonexistent(self, sample_chapter):
        """Test comparing with non-existent version."""
        v_id = sample_chapter.save_current_as_version()

        comparison = sample_chapter.compare_versions(v_id, "fake-id")
        assert "error" in comparison
        assert comparison["error"] == "One or both versions not found"

    def test_version_empty_feedback(self, sample_chapter):
        """Test saving version without feedback."""
        v_id = sample_chapter.save_current_as_version()
        version = sample_chapter.get_version_by_id(v_id)
        assert version is not None
        assert version.feedback == ""


class TestChapterVersionManager:
    """Tests for ChapterVersionManager using composition pattern."""

    def test_manager_created_from_chapter(self):
        """Test that a version manager can be created from a chapter."""
        chapter = Chapter(number=1, title="Test Chapter", outline="Test outline")
        manager = ChapterVersionManager(chapter)
        assert manager is not None
        assert manager._chapter is chapter

    def test_chapter_version_manager_property(self):
        """Test that Chapter.version_manager returns a ChapterVersionManager."""
        chapter = Chapter(number=1, title="Test Chapter", outline="Test outline")
        manager = chapter.version_manager
        assert isinstance(manager, ChapterVersionManager)

    def test_save_version_via_manager(self):
        """Test saving a version through the manager directly."""
        chapter = Chapter(
            number=1,
            title="Test Chapter",
            outline="Test outline",
            content="Original content",
            word_count=2,
        )
        manager = chapter.version_manager

        version_id = manager.save_version()

        assert version_id is not None
        assert len(chapter.versions) == 1
        assert chapter.versions[0].content == "Original content"
        assert chapter.versions[0].is_current is True
        assert chapter.current_version_id == version_id

    def test_count_property(self):
        """Test count returns the number of versions."""
        chapter = Chapter(number=1, title="Test Chapter", outline="Test outline", content="Content")
        manager = chapter.version_manager

        assert manager.count == 0

        manager.save_version()
        assert manager.count == 1

        chapter.content = "V2"
        manager.save_version()
        assert manager.count == 2

    def test_all_versions_property(self):
        """Test all_versions returns the version list."""
        chapter = Chapter(number=1, title="Test Chapter", outline="Test outline", content="Content")
        manager = chapter.version_manager
        manager.save_version()
        chapter.content = "V2"
        manager.save_version()

        versions = manager.all_versions

        assert len(versions) == 2
        assert versions is chapter.versions  # Should be the same list

    def test_rollback_via_manager(self):
        """Test rollback through the manager directly."""
        chapter = Chapter(
            number=1, title="Test Chapter", outline="Test outline", content="Original"
        )
        manager = chapter.version_manager

        v1_id = manager.save_version()
        chapter.content = "Modified"
        chapter.word_count = 1
        manager.save_version()

        result = manager.rollback(v1_id)

        assert result is True
        assert chapter.content == "Original"
        assert chapter.current_version_id == v1_id

    def test_get_version_via_manager(self):
        """Test get_version through the manager directly."""
        chapter = Chapter(number=1, title="Test Chapter", outline="Test outline", content="Content")
        manager = chapter.version_manager
        version_id = manager.save_version()

        version = manager.get_version(version_id)

        assert version is not None
        assert version.id == version_id

    def test_get_current_via_manager(self):
        """Test get_current through the manager directly."""
        chapter = Chapter(number=1, title="Test Chapter", outline="Test outline", content="Content")
        manager = chapter.version_manager
        manager.save_version()

        current = manager.get_current()

        assert current is not None
        assert current.is_current is True

    def test_compare_via_manager(self):
        """Test compare through the manager directly."""
        chapter = Chapter(
            number=1,
            title="Test Chapter",
            outline="Test outline",
            content="Short",
            word_count=1,
        )
        manager = chapter.version_manager
        v1_id = manager.save_version()

        chapter.content = "Longer content here"
        chapter.word_count = 3
        v2_id = manager.save_version()

        result = manager.compare(v1_id, v2_id)

        assert "version_a" in result
        assert "version_b" in result
        assert result["version_a"]["word_count"] == 1
        assert result["version_b"]["word_count"] == 3
        assert result["word_count_diff"] == 2
