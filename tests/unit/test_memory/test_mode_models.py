"""Unit tests for mode models."""

import pytest

from memory.mode_models import (
    PRESET_MODES,
    AutonomyLevel,
    GenerationMode,
    GenerationScore,
    ImplicitSignals,
    LearningSettings,
    LearningTrigger,
    ModelPerformanceSummary,
    ModelSizeTier,
    PerformanceMetrics,
    QualityScores,
    RecommendationType,
    TuningRecommendation,
    VramStrategy,
    get_preset_mode,
    get_size_tier,
    list_preset_modes,
)


class TestEnums:
    """Tests for enum types."""

    def test_vram_strategy_values(self) -> None:
        """Test VramStrategy enum values."""
        assert VramStrategy.SEQUENTIAL.value == "sequential"
        assert VramStrategy.PARALLEL.value == "parallel"
        assert VramStrategy.ADAPTIVE.value == "adaptive"

    def test_learning_trigger_values(self) -> None:
        """Test LearningTrigger enum values."""
        assert LearningTrigger.OFF.value == "off"
        assert LearningTrigger.AFTER_PROJECT.value == "after_project"
        assert LearningTrigger.PERIODIC.value == "periodic"
        assert LearningTrigger.CONTINUOUS.value == "continuous"

    def test_autonomy_level_values(self) -> None:
        """Test AutonomyLevel enum values."""
        assert AutonomyLevel.MANUAL.value == "manual"
        assert AutonomyLevel.CAUTIOUS.value == "cautious"
        assert AutonomyLevel.BALANCED.value == "balanced"
        assert AutonomyLevel.AGGRESSIVE.value == "aggressive"
        assert AutonomyLevel.EXPERIMENTAL.value == "experimental"

    def test_recommendation_type_values(self) -> None:
        """Test RecommendationType enum values."""
        assert RecommendationType.MODEL_SWAP.value == "model_swap"
        assert RecommendationType.TEMP_ADJUST.value == "temp_adjust"
        assert RecommendationType.MODE_CHANGE.value == "mode_change"
        assert RecommendationType.VRAM_STRATEGY.value == "vram_strategy"


class TestGenerationMode:
    """Tests for GenerationMode model."""

    def test_basic_mode(self) -> None:
        """Test creating a basic generation mode."""
        mode = GenerationMode(
            id="test-mode",
            name="Test Mode",
            agent_models={"writer": "model-a"},
            agent_temperatures={"writer": 0.9},
        )

        assert mode.id == "test-mode"
        assert mode.name == "Test Mode"
        assert mode.vram_strategy == VramStrategy.ADAPTIVE
        assert mode.is_preset is False
        assert mode.is_experimental is False

    def test_mode_with_all_options(self) -> None:
        """Test creating a mode with all options."""
        mode = GenerationMode(
            id="full-mode",
            name="Full Mode",
            description="A fully configured mode",
            agent_models={"writer": "model-a", "editor": "model-b"},
            agent_temperatures={"writer": 0.9, "editor": 0.6},
            vram_strategy=VramStrategy.SEQUENTIAL,
            is_preset=True,
            is_experimental=True,
        )

        assert mode.description == "A fully configured mode"
        assert mode.vram_strategy == VramStrategy.SEQUENTIAL
        assert mode.is_preset is True
        assert mode.is_experimental is True


class TestQualityScores:
    """Tests for QualityScores model."""

    def test_default_values(self) -> None:
        """Test default values are None."""
        scores = QualityScores()
        assert scores.prose_quality is None
        assert scores.instruction_following is None
        assert scores.consistency_score is None

    def test_valid_scores(self) -> None:
        """Test valid score values."""
        scores = QualityScores(
            prose_quality=8.5,
            instruction_following=9.0,
            consistency_score=7.5,
        )
        assert scores.prose_quality == 8.5
        assert scores.instruction_following == 9.0
        assert scores.consistency_score == 7.5

    def test_score_validation(self) -> None:
        """Test score validation boundaries."""
        # Valid at boundaries
        QualityScores(prose_quality=0)
        QualityScores(prose_quality=10)

        # Invalid scores should raise
        with pytest.raises(ValueError):
            QualityScores(prose_quality=-1)
        with pytest.raises(ValueError):
            QualityScores(prose_quality=11)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics model."""

    def test_default_values(self) -> None:
        """Test default values are None."""
        metrics = PerformanceMetrics()
        assert metrics.tokens_generated is None
        assert metrics.time_seconds is None
        assert metrics.tokens_per_second is None
        assert metrics.vram_used_gb is None

    def test_with_values(self) -> None:
        """Test with actual values."""
        metrics = PerformanceMetrics(
            tokens_generated=1000,
            time_seconds=10.5,
            tokens_per_second=95.2,
            vram_used_gb=12.5,
        )
        assert metrics.tokens_generated == 1000
        assert metrics.tokens_per_second == 95.2


class TestImplicitSignals:
    """Tests for ImplicitSignals model."""

    def test_default_values(self) -> None:
        """Test default values."""
        signals = ImplicitSignals()
        assert signals.was_regenerated is False
        assert signals.edit_distance is None
        assert signals.user_rating is None

    def test_with_values(self) -> None:
        """Test with actual values."""
        signals = ImplicitSignals(
            was_regenerated=True,
            edit_distance=150,
            user_rating=4,
        )
        assert signals.was_regenerated is True
        assert signals.edit_distance == 150
        assert signals.user_rating == 4

    def test_rating_validation(self) -> None:
        """Test rating validation."""
        # Valid ratings
        ImplicitSignals(user_rating=1)
        ImplicitSignals(user_rating=5)

        # Invalid ratings
        with pytest.raises(ValueError):
            ImplicitSignals(user_rating=0)
        with pytest.raises(ValueError):
            ImplicitSignals(user_rating=6)


class TestGenerationScore:
    """Tests for GenerationScore model."""

    def test_minimal_score(self) -> None:
        """Test creating a minimal generation score."""
        score = GenerationScore(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            mode_name="balanced",
        )

        assert score.project_id == "test-project"
        assert score.chapter_id is None
        assert score.quality == QualityScores()
        assert score.performance == PerformanceMetrics()
        assert score.signals == ImplicitSignals()

    def test_full_score(self) -> None:
        """Test creating a full generation score."""
        score = GenerationScore(
            project_id="test-project",
            chapter_id="ch-1",
            agent_role="writer",
            model_id="test-model",
            mode_name="quality_max",
            genre="fantasy",
            quality=QualityScores(prose_quality=9.0),
            performance=PerformanceMetrics(tokens_per_second=50.0),
            signals=ImplicitSignals(user_rating=5),
        )

        assert score.chapter_id == "ch-1"
        assert score.genre == "fantasy"
        assert score.quality.prose_quality == 9.0


class TestTuningRecommendation:
    """Tests for TuningRecommendation model."""

    def test_basic_recommendation(self) -> None:
        """Test creating a basic recommendation."""
        rec = TuningRecommendation(
            recommendation_type=RecommendationType.MODEL_SWAP,
            current_value="model-a",
            suggested_value="model-b",
            reason="Better quality",
            confidence=0.85,
        )

        assert rec.recommendation_type == RecommendationType.MODEL_SWAP
        assert rec.was_applied is False
        assert rec.user_feedback is None

    def test_confidence_validation(self) -> None:
        """Test confidence validation."""
        # Valid confidence
        TuningRecommendation(
            recommendation_type=RecommendationType.TEMP_ADJUST,
            current_value="0.8",
            suggested_value="0.9",
            reason="Test",
            confidence=0.0,
        )
        TuningRecommendation(
            recommendation_type=RecommendationType.TEMP_ADJUST,
            current_value="0.8",
            suggested_value="0.9",
            reason="Test",
            confidence=1.0,
        )

        # Invalid confidence
        with pytest.raises(ValueError):
            TuningRecommendation(
                recommendation_type=RecommendationType.TEMP_ADJUST,
                current_value="0.8",
                suggested_value="0.9",
                reason="Test",
                confidence=-0.1,
            )
        with pytest.raises(ValueError):
            TuningRecommendation(
                recommendation_type=RecommendationType.TEMP_ADJUST,
                current_value="0.8",
                suggested_value="0.9",
                reason="Test",
                confidence=1.1,
            )


class TestModelPerformanceSummary:
    """Tests for ModelPerformanceSummary model."""

    def test_basic_summary(self) -> None:
        """Test creating a basic summary."""
        summary = ModelPerformanceSummary(
            model_id="test-model",
            agent_role="writer",
        )

        assert summary.model_id == "test-model"
        assert summary.sample_count == 0
        assert summary.avg_prose_quality is None


class TestLearningSettings:
    """Tests for LearningSettings model."""

    def test_default_settings(self) -> None:
        """Test default learning settings."""
        settings = LearningSettings()

        assert LearningTrigger.AFTER_PROJECT in settings.triggers
        assert settings.autonomy == AutonomyLevel.BALANCED
        assert settings.periodic_interval == 5
        assert settings.min_samples_for_recommendation == 5
        assert settings.confidence_threshold == 0.8

    def test_custom_settings(self) -> None:
        """Test custom learning settings."""
        settings = LearningSettings(
            triggers=[LearningTrigger.PERIODIC, LearningTrigger.CONTINUOUS],
            autonomy=AutonomyLevel.AGGRESSIVE,
            periodic_interval=10,
            min_samples_for_recommendation=3,
            confidence_threshold=0.6,
        )

        assert len(settings.triggers) == 2
        assert settings.autonomy == AutonomyLevel.AGGRESSIVE
        assert settings.confidence_threshold == 0.6


class TestPresetModes:
    """Tests for preset mode definitions."""

    def test_preset_modes_exist(self) -> None:
        """Test that preset modes are defined."""
        assert len(PRESET_MODES) >= 5
        assert "quality_max" in PRESET_MODES
        assert "quality_creative" in PRESET_MODES
        assert "balanced" in PRESET_MODES
        assert "draft_fast" in PRESET_MODES
        assert "experimental" in PRESET_MODES

    def test_get_preset_mode(self) -> None:
        """Test getting a preset mode by ID."""
        mode = get_preset_mode("balanced")
        assert mode is not None
        assert mode.name == "Balanced"
        assert mode.is_preset is True

        # Non-existent mode
        assert get_preset_mode("nonexistent") is None

    def test_list_preset_modes(self) -> None:
        """Test listing all preset modes."""
        modes = list_preset_modes()
        assert len(modes) >= 5
        assert all(m.is_preset for m in modes)

    def test_preset_modes_have_all_agent_temperatures(self) -> None:
        """Test that preset modes have temperatures for all agents.

        Note: agent_models is intentionally empty for preset modes to use
        automatic model selection based on installed models. Only temperatures
        need to be defined for all roles.
        """
        expected_roles = {"architect", "writer", "editor", "continuity", "interviewer", "validator"}

        for mode_id, mode in PRESET_MODES.items():
            # agent_models can be empty (auto-selection) or have specific overrides
            # but agent_temperatures must be defined for all roles
            assert set(mode.agent_temperatures.keys()) == expected_roles, (
                f"Mode {mode_id} missing temps"
            )

    def test_experimental_mode_flag(self) -> None:
        """Test that experimental mode has the flag set."""
        exp_mode = PRESET_MODES["experimental"]
        assert exp_mode.is_experimental is True

        # Others should not be experimental
        for mode_id, mode in PRESET_MODES.items():
            if mode_id != "experimental":
                assert mode.is_experimental is False, f"Mode {mode_id} should not be experimental"


class TestGetSizeTier:
    """Tests for get_size_tier function."""

    def test_large_tier(self) -> None:
        """Test models >= 20GB are classified as LARGE."""
        assert get_size_tier(20.0) == ModelSizeTier.LARGE
        assert get_size_tier(25.0) == ModelSizeTier.LARGE
        assert get_size_tier(50.0) == ModelSizeTier.LARGE

    def test_medium_tier(self) -> None:
        """Test models 8-20GB are classified as MEDIUM."""
        assert get_size_tier(8.0) == ModelSizeTier.MEDIUM
        assert get_size_tier(12.0) == ModelSizeTier.MEDIUM
        assert get_size_tier(19.9) == ModelSizeTier.MEDIUM

    def test_small_tier(self) -> None:
        """Test models 3-8GB are classified as SMALL."""
        assert get_size_tier(3.0) == ModelSizeTier.SMALL
        assert get_size_tier(5.0) == ModelSizeTier.SMALL
        assert get_size_tier(7.9) == ModelSizeTier.SMALL

    def test_tiny_tier(self) -> None:
        """Test models < 3GB are classified as TINY."""
        assert get_size_tier(0.5) == ModelSizeTier.TINY
        assert get_size_tier(1.0) == ModelSizeTier.TINY
        assert get_size_tier(2.9) == ModelSizeTier.TINY
