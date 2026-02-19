"""Tests for temporal context wiring in orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from src.services.orchestrator._writing import (
    _retrieve_temporal_context,
)


@pytest.fixture
def mock_orc():
    """Create a mock StoryOrchestrator."""
    orc = MagicMock()
    orc.settings = MagicMock()
    orc.world_db = MagicMock()
    orc.story_state = MagicMock()
    orc.context_retrieval = MagicMock()
    orc.timeline = MagicMock()
    return orc


class TestRetrieveTemporalContext:
    """Tests for _retrieve_temporal_context helper."""

    def test_returns_context_when_world_db_exists(self, mock_orc):
        """Returns temporal context when world_db and timeline are available."""
        mock_orc.timeline.build_temporal_context.return_value = "CHARACTERS:\n- Hero: Year 100"

        result = _retrieve_temporal_context(mock_orc)

        assert result == "CHARACTERS:\n- Hero: Year 100"
        mock_orc.timeline.build_temporal_context.assert_called_once_with(mock_orc.world_db)

    def test_returns_empty_when_no_world_db(self, mock_orc):
        """Returns empty string when world_db is None."""
        mock_orc.world_db = None

        result = _retrieve_temporal_context(mock_orc)

        assert result == ""

    def test_returns_empty_on_exception(self, mock_orc):
        """Returns empty string when timeline service raises."""
        mock_orc.timeline.build_temporal_context.side_effect = RuntimeError("Boom")

        result = _retrieve_temporal_context(mock_orc)

        assert result == ""

    def test_returns_empty_when_temporal_disabled(self, mock_orc):
        """Returns empty string when validate_temporal_consistency is False."""
        mock_orc.settings.validate_temporal_consistency = False

        result = _retrieve_temporal_context(mock_orc)

        assert result == ""

    def test_returns_empty_when_build_returns_empty(self, mock_orc):
        """Returns empty string when build_temporal_context returns empty."""
        mock_orc.timeline.build_temporal_context.return_value = ""

        result = _retrieve_temporal_context(mock_orc)

        assert result == ""

    def test_returns_empty_when_no_timeline_service(self, mock_orc):
        """Returns empty string when timeline service is not configured."""
        mock_orc.timeline = None

        result = _retrieve_temporal_context(mock_orc)

        assert result == ""


class TestCombineContexts:
    """Tests for _combine_contexts helper."""

    def test_both_contexts_present(self):
        """Combines both contexts with separator."""
        from src.services.orchestrator._writing import _combine_contexts

        result = _combine_contexts("RAG data", "Timeline data")
        assert "RAG data" in result
        assert "Timeline data" in result
        assert "\n\n" in result

    def test_only_world_context(self):
        """Returns world context when temporal is empty."""
        from src.services.orchestrator._writing import _combine_contexts

        result = _combine_contexts("RAG data", "")
        assert result == "RAG data"

    def test_only_temporal_context(self):
        """Returns temporal context when world is empty."""
        from src.services.orchestrator._writing import _combine_contexts

        result = _combine_contexts("", "Timeline data")
        assert result == "Timeline data"

    def test_both_empty(self):
        """Returns empty string when both are empty."""
        from src.services.orchestrator._writing import _combine_contexts

        result = _combine_contexts("", "")
        assert result == ""


class TestWriteAllChaptersFinalReviewContextWarning:
    """Tests for aggregated context-failure warning in write_all_chapters."""

    def test_warns_when_both_contexts_empty_but_sources_configured(self, mock_orc):
        """Logs warning when both RAG and temporal return empty but sources exist."""
        from src.memory.story_state import Chapter

        chapter = MagicMock(spec=Chapter)
        chapter.number = 1
        chapter.content = "Story content"
        mock_orc.story_state.chapters = [chapter]
        mock_orc.story_state.status = "writing"
        mock_orc.interaction_mode = "auto"
        mock_orc.settings.max_revision_iterations = 0
        mock_orc.settings.chapters_between_checkpoints = 5
        mock_orc.settings.validate_temporal_consistency = True
        mock_orc._total_chapters = 0
        mock_orc._completed_chapters = 0
        mock_orc.events = []
        mock_orc._emit = MagicMock()
        mock_orc._set_phase = MagicMock()
        # context_retrieval and world_db are set (sources configured)
        mock_orc.context_retrieval = MagicMock()
        mock_orc.world_db = MagicMock()

        def mock_write_chapter(num):
            """Yield a mock event for the chapter."""
            mock_orc._emit("agent_complete", "System", f"Chapter {num} complete")
            mock_orc.events.append(MagicMock())
            yield mock_orc.events[-1]

        mock_orc.write_chapter = mock_write_chapter
        mock_orc.continuity.check_full_story.return_value = []

        with patch(
            "src.services.orchestrator._writing._retrieve_world_context",
            return_value="",
        ):
            with patch(
                "src.services.orchestrator._writing._retrieve_temporal_context",
                return_value="",
            ):
                with patch("src.services.orchestrator._writing.logger") as mock_logger:
                    from src.services.orchestrator._writing import write_all_chapters

                    list(write_all_chapters(mock_orc))

                    mock_logger.warning.assert_any_call(
                        "Final story review proceeding without any world/temporal context "
                        "despite context sources being configured"
                    )


class TestReviewFullStoryContextWarning:
    """Tests for aggregated context-failure warning in review_full_story."""

    def test_warns_when_both_contexts_empty_but_sources_configured(self, mock_orc):
        """Logs warning when both RAG and temporal return empty but sources exist."""
        mock_orc.events = []
        mock_orc.context_retrieval = MagicMock()
        mock_orc.world_db = MagicMock()
        mock_orc.settings.validate_temporal_consistency = True

        def fake_emit(*args, **kwargs):
            """Append a mock event on each emit call."""
            mock_orc.events.append(MagicMock())

        mock_orc._emit = fake_emit
        mock_orc.continuity.check_full_story.return_value = []

        with patch(
            "src.services.orchestrator._writing._retrieve_world_context",
            return_value="",
        ):
            with patch(
                "src.services.orchestrator._writing._retrieve_temporal_context",
                return_value="",
            ):
                with patch("src.services.orchestrator._editing.logger") as mock_logger:
                    from src.services.orchestrator._editing import review_full_story

                    list(review_full_story(mock_orc))

                    mock_logger.warning.assert_any_call(
                        "Full story review proceeding without any world/temporal context "
                        "despite context sources being configured"
                    )


class TestWriteAllChaptersFinalReview:
    """Tests for temporal context in write_all_chapters final review."""

    def test_final_review_passes_combined_context(self, mock_orc):
        """Final review combines RAG + temporal context for check_full_story."""
        from src.memory.story_state import Chapter

        # Set up story state with one chapter
        chapter = MagicMock(spec=Chapter)
        chapter.number = 1
        chapter.content = "Story content"
        mock_orc.story_state.chapters = [chapter]
        mock_orc.story_state.status = "writing"
        mock_orc.interaction_mode = "auto"
        mock_orc.settings.max_revision_iterations = 0
        mock_orc.settings.chapters_between_checkpoints = 5
        mock_orc._total_chapters = 0
        mock_orc._completed_chapters = 0
        mock_orc.events = []
        mock_orc._emit = MagicMock()
        mock_orc._set_phase = MagicMock()

        # Mock write_chapter to yield events
        def mock_write_chapter(num):
            """Yield a mock event for the chapter."""
            mock_orc._emit("agent_complete", "System", f"Chapter {num} complete")
            mock_orc.events.append(MagicMock())
            yield mock_orc.events[-1]

        mock_orc.write_chapter = mock_write_chapter

        # Mock continuity check_full_story
        mock_orc.continuity.check_full_story.return_value = []

        with patch(
            "src.services.orchestrator._writing._retrieve_world_context",
            return_value="RAG context",
        ):
            with patch(
                "src.services.orchestrator._writing._retrieve_temporal_context",
                return_value="TEMPORAL context",
            ):
                from src.services.orchestrator._writing import write_all_chapters

                # Consume generator
                list(write_all_chapters(mock_orc))

        # check_full_story should have been called with combined context
        mock_orc.continuity.check_full_story.assert_called_once()
        call_kwargs = mock_orc.continuity.check_full_story.call_args
        world_context = call_kwargs.kwargs.get("world_context", "")
        assert "RAG context" in world_context
        assert "TEMPORAL context" in world_context

    def test_final_review_works_without_temporal_context(self, mock_orc):
        """Final review works when temporal context is empty."""
        from src.memory.story_state import Chapter

        chapter = MagicMock(spec=Chapter)
        chapter.number = 1
        chapter.content = "Story content"
        mock_orc.story_state.chapters = [chapter]
        mock_orc.story_state.status = "writing"
        mock_orc.interaction_mode = "auto"
        mock_orc.settings.max_revision_iterations = 0
        mock_orc.settings.chapters_between_checkpoints = 5
        mock_orc._total_chapters = 0
        mock_orc._completed_chapters = 0
        mock_orc.events = []
        mock_orc._emit = MagicMock()
        mock_orc._set_phase = MagicMock()

        def mock_write_chapter(num):
            """Yield a mock event for the chapter."""
            mock_orc._emit("agent_complete", "System", f"Chapter {num} complete")
            mock_orc.events.append(MagicMock())
            yield mock_orc.events[-1]

        mock_orc.write_chapter = mock_write_chapter
        mock_orc.continuity.check_full_story.return_value = []

        with patch(
            "src.services.orchestrator._writing._retrieve_world_context",
            return_value="RAG context only",
        ):
            with patch(
                "src.services.orchestrator._writing._retrieve_temporal_context",
                return_value="",
            ):
                from src.services.orchestrator._writing import write_all_chapters

                list(write_all_chapters(mock_orc))

        call_kwargs = mock_orc.continuity.check_full_story.call_args
        world_context = call_kwargs.kwargs.get("world_context", "")
        assert "RAG context only" in world_context
        # No temporal context appended
        assert "TEMPORAL" not in world_context


class TestReviewFullStoryTemporal:
    """Tests for temporal context in review_full_story."""

    def test_review_full_story_passes_combined_context(self, mock_orc):
        """review_full_story combines RAG + temporal context."""
        mock_orc.events = []

        def fake_emit(*args, **kwargs):
            """Append a mock event on each emit call."""
            mock_orc.events.append(MagicMock())

        mock_orc._emit = fake_emit
        mock_orc.continuity.check_full_story.return_value = []

        with patch(
            "src.services.orchestrator._writing._retrieve_world_context",
            return_value="RAG data",
        ):
            with patch(
                "src.services.orchestrator._writing._retrieve_temporal_context",
                return_value="Timeline data",
            ):
                from src.services.orchestrator._editing import review_full_story

                list(review_full_story(mock_orc))

        call_kwargs = mock_orc.continuity.check_full_story.call_args
        world_context = call_kwargs.kwargs.get("world_context", "")
        assert "RAG data" in world_context
        assert "Timeline data" in world_context
