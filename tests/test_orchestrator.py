"""Tests for the orchestrator export functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from memory.story_state import Chapter, StoryBrief, StoryState
from workflows.orchestrator import StoryOrchestrator


class TestExportFunctionality:
    """Tests for story export to files."""

    @pytest.fixture
    def sample_story_state(self):
        """Create a sample story state for testing."""
        state = StoryState(
            id="test-story-123",
            created_at=datetime.now(),
            status="complete",
            brief=StoryBrief(
                premise="A robot learns to love",
                genre="Science Fiction",
                tone="Hopeful",
                setting_time="2150",
                setting_place="Mars Colony",
                target_length="short_story",
                nsfw_level="none",
            ),
        )
        state.chapters = [
            Chapter(
                number=1,
                title="First Encounter",
                outline="The robot meets a human",
                content="The robot's optical sensors focused on the human...",
                status="final",
                word_count=500,
            )
        ]
        return state

    def test_export_to_markdown_file(self, sample_story_state):
        """Should export story to markdown file."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.md"
            result = orchestrator.export_story_to_file(format="markdown", filepath=str(filepath))

            assert Path(result).exists()
            assert result.endswith(".md")

            content = Path(result).read_text(encoding="utf-8")
            assert "A robot learns to love" in content
            assert "Chapter 1: First Encounter" in content

    def test_export_to_text_file(self, sample_story_state):
        """Should export story to plain text file."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            result = orchestrator.export_story_to_file(format="text", filepath=str(filepath))

            assert Path(result).exists()
            assert result.endswith(".txt")

            content = Path(result).read_text(encoding="utf-8")
            assert "Science Fiction" in content
            assert "FIRST ENCOUNTER" in content

    def test_export_raises_on_no_story(self):
        """Should raise ValueError when no story to export."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = None

        with pytest.raises(ValueError, match="No story to export"):
            orchestrator.export_story_to_file(format="markdown")

    def test_export_raises_on_invalid_format(self, sample_story_state):
        """Should raise ValueError for unsupported format."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with pytest.raises(ValueError, match="Unsupported export format"):
            orchestrator.export_story_to_file(format="pdf")

    def test_export_creates_directory_if_not_exists(self, sample_story_state):
        """Should create parent directories if they don't exist."""
        orchestrator = StoryOrchestrator()
        orchestrator.story_state = sample_story_state

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dir" / "story.md"
            result = orchestrator.export_story_to_file(format="markdown", filepath=str(filepath))

            assert Path(result).exists()
            assert Path(result).parent.exists()
