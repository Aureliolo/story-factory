"""Tests for Architect output quality review (character, plot, chapter).

Tests cover:
- review_character_quality (passes above threshold, refines below)
- review_plot_quality (passes above threshold, refines below)
- review_chapter_quality (passes above threshold, refines below)
- review_characters_batch (batch review with progress, errors, callbacks)
- review_chapters_batch (batch review with progress, errors, callbacks, cancellation)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.story_state import (
    Chapter,
    Character,
    PlotOutline,
    PlotPoint,
    Scene,
    StoryBrief,
    StoryState,
)
from src.memory.world_quality import (
    ChapterQualityScores,
    CharacterQualityScores,
    JudgeConsistencyConfig,
    PlotQualityScores,
)
from src.services.world_quality_service import WorldQualityService
from src.services.world_quality_service._chapter_quality import (
    _build_chapter_judge_prompt,
)
from src.settings import Settings
from src.utils.exceptions import WorldGenerationError


@pytest.fixture
def settings():
    """Create settings with test values."""
    return Settings(
        ollama_url="http://localhost:11434",
        ollama_timeout=60,
        world_quality_enabled=True,
        world_quality_max_iterations=3,
        world_quality_threshold=8.0,
        world_quality_thresholds={
            "character": 8.0,
            "location": 8.0,
            "faction": 8.0,
            "item": 8.0,
            "concept": 8.0,
            "event": 8.0,
            "relationship": 8.0,
            "plot": 8.0,
            "chapter": 8.0,
        },
        world_quality_creator_temp=0.9,
        world_quality_judge_temp=0.1,
        world_quality_refinement_temp=0.7,
        mini_description_words_max=15,
    )


@pytest.fixture
def mock_mode_service():
    """Create a mock mode service."""
    svc = MagicMock()
    svc.get_model_for_agent.return_value = "test-model:8b"
    return svc


@pytest.fixture
def service(settings, mock_mode_service):
    """Create WorldQualityService with mocked dependencies."""
    svc = WorldQualityService(settings, mock_mode_service)
    svc._analytics_db = MagicMock()
    return svc


@pytest.fixture
def story_state():
    """Create story state with brief for testing."""
    state = StoryState(id="test-story-id")
    state.brief = StoryBrief(
        premise="A hero's journey through a magical land",
        genre="fantasy",
        subgenres=["adventure"],
        tone="epic",
        themes=["courage", "sacrifice"],
        setting_time="Medieval",
        setting_place="Fantasy realm",
        target_length="novella",
        language="English",
        content_rating="mild",
    )
    state.plot_summary = "A young hero discovers their destiny"
    state.plot_points = [
        PlotPoint(description="Hero receives the call", chapter=1),
        PlotPoint(description="Hero crosses the threshold", chapter=2),
    ]
    return state


@pytest.fixture
def test_character():
    """Create a test character."""
    return Character(
        name="Test Hero",
        role="protagonist",
        description="A brave warrior on a quest",
        personality_traits=["brave", "stubborn"],
        goals=["save the kingdom"],
        arc_notes="Learns humility through failure",
    )


@pytest.fixture
def test_plot_outline():
    """Create a test plot outline."""
    return PlotOutline(
        plot_summary="A young hero discovers their destiny",
        plot_points=[
            PlotPoint(description="Hero receives the call", chapter=1),
            PlotPoint(description="Hero crosses the threshold", chapter=2),
        ],
    )


@pytest.fixture
def test_chapter():
    """Create a test chapter."""
    return Chapter(
        number=1,
        title="The Beginning",
        outline="The hero receives a mysterious call to adventure",
    )


class TestReviewCharacterQuality:
    """Test review_character_quality for Architect output review."""

    def test_passes_above_threshold(self, service, story_state, test_character):
        """Character above threshold passes without refinement."""
        high_scores = CharacterQualityScores(
            depth=8.5,
            goals=8.2,
            flaws=8.8,
            uniqueness=8.0,
            arc_potential=8.3,
            temporal_plausibility=8.2,
            feedback="Excellent character",
        )

        with patch(
            "src.services.world_quality_service._character.generate_structured",
            return_value=high_scores,
        ):
            result_char, result_scores, iterations = service.review_character_quality(
                test_character, story_state
            )

        assert result_char.name == "Test Hero"
        assert result_scores.average >= 8.0
        assert iterations == 1

    def test_refines_below_threshold(self, service, story_state, test_character):
        """Character below threshold triggers refinement."""
        low_scores = CharacterQualityScores(
            depth=5.0,
            goals=5.0,
            flaws=5.0,
            uniqueness=5.0,
            arc_potential=5.0,
            temporal_plausibility=5.0,
            feedback="Needs more depth",
        )
        high_scores = CharacterQualityScores(
            depth=8.5,
            goals=8.5,
            flaws=8.5,
            uniqueness=8.5,
            arc_potential=8.5,
            temporal_plausibility=8.5,
            feedback="Much improved",
        )
        refined_char = Character(
            name="Test Hero",
            role="protagonist",
            description="A deeply complex warrior with inner conflict",
            personality_traits=["brave", "stubborn", "compassionate"],
            goals=["save the kingdom", "find inner peace"],
            arc_notes="Learns humility through failure, gains wisdom",
        )

        judge_count = 0

        def mock_judge(char, story_state, temperature):
            """Return low scores on first call, high on second."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return low_scores
            return high_scores

        def mock_refine(char, scores, story_state, temperature):
            """Return the pre-built refined character."""
            return refined_char

        with (
            patch.object(service, "_judge_character_quality", side_effect=mock_judge),
            patch.object(service, "_refine_character", side_effect=mock_refine),
        ):
            _result_char, result_scores, iterations = service.review_character_quality(
                test_character, story_state
            )

        assert iterations == 2
        assert result_scores.average >= 8.0


class TestReviewPlotQuality:
    """Test review_plot_quality for Architect output review."""

    def test_passes_above_threshold(self, service, story_state, test_plot_outline):
        """Plot above threshold passes without refinement."""
        high_scores = PlotQualityScores(
            coherence=8.5,
            tension_arc=8.2,
            character_integration=8.8,
            originality=8.0,
            feedback="Strong plot",
        )

        with patch(
            "src.services.world_quality_service._plot.generate_structured",
            return_value=high_scores,
        ):
            result_plot, result_scores, iterations = service.review_plot_quality(
                test_plot_outline, story_state
            )

        assert result_plot.plot_summary == test_plot_outline.plot_summary
        assert result_scores.average >= 8.0
        assert iterations == 1

    def test_refines_below_threshold(self, service, story_state, test_plot_outline):
        """Plot below threshold triggers refinement."""
        low_scores = PlotQualityScores(
            coherence=5.0,
            tension_arc=5.0,
            character_integration=5.0,
            originality=5.0,
            feedback="Needs better tension arc",
        )
        high_scores = PlotQualityScores(
            coherence=8.5,
            tension_arc=8.5,
            character_integration=8.5,
            originality=8.5,
            feedback="Improved",
        )
        refined_plot = PlotOutline(
            plot_summary="A refined plot with better tension",
            plot_points=[
                PlotPoint(description="Improved call to adventure", chapter=1),
                PlotPoint(description="Enhanced threshold crossing", chapter=2),
            ],
        )

        judge_count = 0

        def mock_judge(plot, story_state, temperature):
            """Return low scores on first call, high on second."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return low_scores
            return high_scores

        def mock_refine(plot, scores, story_state, temperature):
            """Return the pre-built refined plot outline."""
            return refined_plot

        with (
            patch.object(service, "_judge_plot_quality", side_effect=mock_judge),
            patch.object(service, "_refine_plot", side_effect=mock_refine),
        ):
            _result_plot, result_scores, iterations = service.review_plot_quality(
                test_plot_outline, story_state
            )

        assert iterations == 2
        assert result_scores.average >= 8.0

    def test_refine_plot_generates_structured_output(self, service, story_state, test_plot_outline):
        """Directly calling _refine_plot delegates to generate_structured and returns PlotOutline."""
        low_scores = PlotQualityScores(
            coherence=5.0,
            tension_arc=5.0,
            character_integration=5.0,
            originality=5.0,
            feedback="Needs work",
        )
        refined_plot = PlotOutline(
            plot_summary="Refined journey",
            plot_points=[PlotPoint(description="The refined call", chapter=1)],
        )

        with patch(
            "src.services.world_quality_service._plot.generate_structured",
            return_value=refined_plot,
        ) as mock_gen:
            result = service._refine_plot(test_plot_outline, low_scores, story_state, 0.7)

        assert result is refined_plot
        mock_gen.assert_called_once()

    def test_judge_plot_quality_logs_exception_on_error(
        self, service, story_state, test_plot_outline
    ):
        """Judge plot quality raises WorldGenerationError when generate_structured fails."""
        with patch(
            "src.services.world_quality_service._plot.generate_structured",
            side_effect=Exception("LLM error"),
        ):
            with pytest.raises(WorldGenerationError, match="Plot quality judgment failed"):
                service._judge_plot_quality(test_plot_outline, story_state, 0.1)


class TestReviewChapterQuality:
    """Test review_chapter_quality for Architect output review."""

    def test_passes_above_threshold(self, service, story_state, test_chapter):
        """Chapter above threshold passes without refinement."""
        high_scores = ChapterQualityScores(
            purpose=8.5,
            pacing=8.2,
            hook=8.8,
            coherence=8.0,
            feedback="Strong chapter",
        )

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            return_value=high_scores,
        ):
            result_ch, result_scores, iterations = service.review_chapter_quality(
                test_chapter, story_state
            )

        assert result_ch.title == "The Beginning"
        assert result_scores.average >= 8.0
        assert iterations == 1

    def test_refines_below_threshold(self, service, story_state, test_chapter):
        """Chapter below threshold triggers refinement."""
        low_scores = ChapterQualityScores(
            purpose=5.0,
            pacing=5.0,
            hook=5.0,
            coherence=5.0,
            feedback="Weak hook",
        )
        high_scores = ChapterQualityScores(
            purpose=8.5,
            pacing=8.5,
            hook=8.5,
            coherence=8.5,
            feedback="Improved",
        )
        refined_chapter = Chapter(
            number=1,
            title="The Beginning",
            outline="An enhanced opening with a gripping hook",
        )

        judge_count = 0

        def mock_judge(chapter, story_state, temperature):
            """Return low scores on first call, high on second."""
            nonlocal judge_count
            judge_count += 1
            if judge_count == 1:
                return low_scores
            return high_scores

        def mock_refine(chapter, scores, story_state, temperature):
            """Return the pre-built refined chapter."""
            return refined_chapter

        with (
            patch.object(service, "_judge_chapter_quality", side_effect=mock_judge),
            patch.object(service, "_refine_chapter_outline", side_effect=mock_refine),
        ):
            _result_ch, result_scores, iterations = service.review_chapter_quality(
                test_chapter, story_state
            )

        assert iterations == 2
        assert result_scores.average >= 8.0

    def test_refine_chapter_outline_generates_structured_output(
        self, service, story_state, test_chapter
    ):
        """Directly calling _refine_chapter_outline delegates to generate_structured."""
        low_scores = ChapterQualityScores(
            purpose=5.0,
            pacing=5.0,
            hook=5.0,
            coherence=5.0,
            feedback="Needs improvement",
        )
        refined_chapter = Chapter(
            number=1,
            title="The Refined Beginning",
            outline="A much stronger opening scene",
        )

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            return_value=refined_chapter,
        ) as mock_gen:
            result = service._refine_chapter_outline(test_chapter, low_scores, story_state, 0.7)

        assert result is refined_chapter
        mock_gen.assert_called_once()

    def test_judge_chapter_quality_logs_exception_on_error(
        self, service, story_state, test_chapter
    ):
        """Judge chapter quality raises WorldGenerationError when generate_structured fails."""
        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            side_effect=Exception("LLM error"),
        ):
            with pytest.raises(WorldGenerationError, match="Chapter quality judgment failed"):
                service._judge_chapter_quality(test_chapter, story_state, 0.1)

    def test_chapter_with_scenes_builds_scenes_text(self):
        """Chapter with scenes includes scene details in the judge prompt."""
        chapter = Chapter(
            number=1,
            title="The Beginning",
            outline="The hero begins the journey",
            scenes=[
                Scene(
                    id="scene-1",
                    title="Awakening",
                    order=1,
                    goal="Hero wakes up in the forest",
                ),
                Scene(
                    id="scene-2",
                    title="Discovery",
                    order=2,
                    goal="Hero finds the ancient map",
                ),
            ],
        )

        prompt = _build_chapter_judge_prompt(chapter, "fantasy", "A hero's journey")

        assert "Hero wakes up in the forest" in prompt
        assert "Hero finds the ancient map" in prompt
        assert "Scenes:" in prompt


class TestReviewCharactersBatch:
    """Test review_characters_batch for batch character review."""

    def test_reviews_all_characters(self, service, story_state):
        """Batch review processes all characters."""
        characters = [
            Character(
                name=f"Hero {i}",
                role="protagonist",
                description=f"Hero number {i}",
                personality_traits=["brave"],
                goals=["survive"],
                arc_notes="Grows",
            )
            for i in range(3)
        ]

        high_scores = CharacterQualityScores(
            depth=8.5,
            goals=8.5,
            flaws=8.5,
            uniqueness=8.5,
            arc_potential=8.5,
            temporal_plausibility=8.5,
            feedback="Good",
        )

        with patch(
            "src.services.world_quality_service._character.generate_structured",
            return_value=high_scores,
        ):
            results = service.review_characters_batch(characters, story_state)

        assert len(results) == 3
        for _char, scores in results:
            assert scores.average >= 8.0

    def test_cancellation_stops_early(self, service, story_state):
        """Cancellation stops batch review early."""
        characters = [
            Character(
                name=f"Hero {i}",
                role="protagonist",
                description=f"Hero number {i}",
                personality_traits=["brave"],
                goals=["survive"],
                arc_notes="Grows",
            )
            for i in range(5)
        ]

        high_scores = CharacterQualityScores(
            depth=8.5,
            goals=8.5,
            flaws=8.5,
            uniqueness=8.5,
            arc_potential=8.5,
            temporal_plausibility=8.5,
            feedback="Good",
        )

        call_count = 0

        def cancel_after_two():
            """Signal cancellation after two calls have been made."""
            return call_count >= 2

        with patch(
            "src.services.world_quality_service._character.generate_structured",
            return_value=high_scores,
        ) as mock_gen:

            def counting_side_effect(**kwargs):
                """Increment call counter and return high scores."""
                nonlocal call_count
                call_count += 1
                return high_scores

            mock_gen.side_effect = counting_side_effect

            results = service.review_characters_batch(
                characters, story_state, cancel_check=cancel_after_two
            )

        # Should have fewer than 5 results due to cancellation
        assert len(results) < 5

    def test_progress_callback_receives_updates(self, service, story_state):
        """Progress callback is invoked during batch character review."""
        characters = [
            Character(
                name=f"Hero {i}",
                role="protagonist",
                description=f"Hero number {i}",
                personality_traits=["brave"],
                goals=["survive"],
                arc_notes="Grows",
            )
            for i in range(2)
        ]

        high_scores = CharacterQualityScores(
            depth=8.5,
            goals=8.5,
            flaws=8.5,
            uniqueness=8.5,
            arc_potential=8.5,
            temporal_plausibility=8.5,
            feedback="Good",
        )

        callback_updates = []

        def progress_callback(progress):
            """Collect progress updates."""
            callback_updates.append(progress)

        with patch.object(service, "review_character_quality") as mock_review:
            mock_review.side_effect = lambda char, state: (char, high_scores, 1)
            results = service.review_characters_batch(
                characters, story_state, progress_callback=progress_callback
            )

        assert len(results) == 2
        # Each character gets a "reviewing" and "complete" callback = 4 total
        assert len(callback_updates) == 4
        phases = [u.phase for u in callback_updates]
        assert "reviewing" in phases
        assert "complete" in phases

    def test_error_keeps_original_with_zero_scores(self, service, story_state):
        """When character review raises WorldGenerationError, original is kept with zero scores."""
        characters = [
            Character(
                name="Failing Hero",
                role="protagonist",
                description="A hero whose review fails",
                personality_traits=["brave"],
                goals=["survive"],
                arc_notes="Grows",
            )
        ]

        with patch.object(service, "review_character_quality") as mock_review:
            mock_review.side_effect = WorldGenerationError("LLM error")
            results = service.review_characters_batch(characters, story_state)

        assert len(results) == 1
        char, scores = results[0]
        assert char.name == "Failing Hero"
        assert scores.depth == 0
        assert scores.goals == 0
        assert scores.flaws == 0
        assert scores.uniqueness == 0
        assert scores.arc_potential == 0
        assert scores.temporal_plausibility == 0
        assert "Review failed" in scores.feedback
        assert "LLM error" in scores.feedback


class TestReviewChaptersBatch:
    """Test review_chapters_batch for batch chapter review."""

    def test_reviews_all_chapters(self, service, story_state):
        """Batch review processes all chapters."""
        chapters = [
            Chapter(number=i, title=f"Chapter {i}", outline=f"Chapter {i} outline")
            for i in range(1, 4)
        ]

        high_scores = ChapterQualityScores(
            purpose=8.5,
            pacing=8.5,
            hook=8.5,
            coherence=8.5,
            feedback="Good",
        )

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            return_value=high_scores,
        ):
            results = service.review_chapters_batch(chapters, story_state)

        assert len(results) == 3
        for _ch, scores in results:
            assert scores.average >= 8.0

    def test_failure_keeps_original_chapter(self, service, story_state):
        """When review fails for a chapter, the original is kept."""
        chapters = [
            Chapter(number=1, title="Chapter 1", outline="Chapter 1 outline"),
        ]

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            side_effect=Exception("LLM error"),
        ):
            results = service.review_chapters_batch(chapters, story_state)

        assert len(results) == 1
        ch, scores = results[0]
        assert ch.title == "Chapter 1"
        assert "Review failed" in scores.feedback

    def test_cancellation_stops_chapters_early(self, service, story_state):
        """Cancellation stops batch chapter review before processing all chapters."""
        chapters = [
            Chapter(number=i, title=f"Chapter {i}", outline=f"Chapter {i} outline")
            for i in range(1, 4)
        ]

        high_scores = ChapterQualityScores(
            purpose=8.5,
            pacing=8.5,
            hook=8.5,
            coherence=8.5,
            feedback="Good",
        )

        call_count = 0

        def cancel_after_one():
            """Signal cancellation after one review has been made."""
            return call_count >= 1

        with patch.object(service, "review_chapter_quality") as mock_review:

            def counting_side_effect(chapter, state):
                """Increment call counter and return success."""
                nonlocal call_count
                call_count += 1
                return (chapter, high_scores, 1)

            mock_review.side_effect = counting_side_effect
            results = service.review_chapters_batch(
                chapters, story_state, cancel_check=cancel_after_one
            )

        # Should have fewer than 3 results due to cancellation
        assert len(results) < 3

    def test_chapter_progress_callback_receives_updates(self, service, story_state):
        """Progress callback is invoked during batch chapter review."""
        chapters = [
            Chapter(number=i, title=f"Chapter {i}", outline=f"Chapter {i} outline")
            for i in range(1, 3)
        ]

        high_scores = ChapterQualityScores(
            purpose=8.5,
            pacing=8.5,
            hook=8.5,
            coherence=8.5,
            feedback="Good",
        )

        callback_updates = []

        def progress_callback(progress):
            """Collect progress updates."""
            callback_updates.append(progress)

        with patch.object(service, "review_chapter_quality") as mock_review:
            mock_review.side_effect = lambda ch, state: (ch, high_scores, 1)
            results = service.review_chapters_batch(
                chapters, story_state, progress_callback=progress_callback
            )

        assert len(results) == 2
        # Each chapter gets a "reviewing" and "complete" callback = 4 total
        assert len(callback_updates) == 4
        phases = [u.phase for u in callback_updates]
        assert "reviewing" in phases
        assert "complete" in phases

    def test_chapter_error_keeps_original_with_zero_scores(self, service, story_state):
        """When chapter review raises WorldGenerationError, original is kept with zero scores."""
        chapters = [
            Chapter(number=1, title="Failing Chapter", outline="A chapter whose review fails"),
        ]

        with patch.object(service, "review_chapter_quality") as mock_review:
            mock_review.side_effect = WorldGenerationError("LLM error")
            results = service.review_chapters_batch(chapters, story_state)

        assert len(results) == 1
        ch, scores = results[0]
        assert ch.title == "Failing Chapter"
        assert scores.purpose == 0
        assert scores.pacing == 0
        assert scores.hook == 0
        assert scores.coherence == 0
        assert "Review failed" in scores.feedback
        assert "LLM error" in scores.feedback


class TestNonMultiCallJudgeErrors:
    """Test judge error paths when multi_call is disabled (logger.exception branch)."""

    def test_plot_judge_error_without_multi_call_logs_exception(
        self, service, story_state, test_plot_outline
    ):
        """Plot judge error uses logger.exception when multi_call is disabled."""
        # Override get_judge_config to return disabled multi_call
        disabled_config = JudgeConsistencyConfig(enabled=False, multi_call_enabled=False)
        with (
            patch.object(service, "get_judge_config", return_value=disabled_config),
            patch(
                "src.services.world_quality_service._plot.generate_structured",
                side_effect=Exception("LLM error"),
            ),
        ):
            with pytest.raises(WorldGenerationError, match="Plot quality judgment failed"):
                service._judge_plot_quality(test_plot_outline, story_state, 0.1)

    def test_chapter_judge_error_without_multi_call_logs_exception(
        self, service, story_state, test_chapter
    ):
        """Chapter judge error uses logger.exception when multi_call is disabled."""
        # Override get_judge_config to return disabled multi_call
        disabled_config = JudgeConsistencyConfig(enabled=False, multi_call_enabled=False)
        with (
            patch.object(service, "get_judge_config", return_value=disabled_config),
            patch(
                "src.services.world_quality_service._chapter_quality.generate_structured",
                side_effect=Exception("LLM error"),
            ),
        ):
            with pytest.raises(WorldGenerationError, match="Chapter quality judgment failed"):
                service._judge_chapter_quality(test_chapter, story_state, 0.1)


class TestRefineErrorPaths:
    """Test error paths in refinement functions."""

    def test_refine_plot_raises_world_generation_error(
        self, service, story_state, test_plot_outline
    ):
        """Plot refinement raises WorldGenerationError when generate_structured fails."""
        low_scores = PlotQualityScores(
            coherence=5.0,
            tension_arc=5.0,
            character_integration=5.0,
            originality=5.0,
            feedback="Needs work",
        )

        with patch(
            "src.services.world_quality_service._plot.generate_structured",
            side_effect=Exception("Refinement LLM error"),
        ):
            with pytest.raises(WorldGenerationError, match="Plot refinement failed"):
                service._refine_plot(test_plot_outline, low_scores, story_state, 0.7)

    def test_refine_chapter_raises_world_generation_error(self, service, story_state, test_chapter):
        """Chapter refinement raises WorldGenerationError when generate_structured fails."""
        low_scores = ChapterQualityScores(
            purpose=5.0,
            pacing=5.0,
            hook=5.0,
            coherence=5.0,
            feedback="Needs improvement",
        )

        with patch(
            "src.services.world_quality_service._chapter_quality.generate_structured",
            side_effect=Exception("Refinement LLM error"),
        ):
            with pytest.raises(WorldGenerationError, match="Chapter refinement failed"):
                service._refine_chapter_outline(test_chapter, low_scores, story_state, 0.7)
