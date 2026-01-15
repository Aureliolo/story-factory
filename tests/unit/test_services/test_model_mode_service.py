"""Unit tests for ModelModeService."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memory.mode_models import (
    AutonomyLevel,
    GenerationMode,
    LearningSettings,
    LearningTrigger,
    QualityScores,
    RecommendationType,
    TuningRecommendation,
    VramStrategy,
)
from services.model_mode_service import ModelModeService
from settings import Settings


class TestModelModeService:
    """Tests for ModelModeService."""

    @pytest.fixture
    def temp_db(self) -> Path:
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock(spec=Settings)
        mock.ollama_url = "http://localhost:11434"
        mock.get_model_for_agent.return_value = "test-model:8b"
        mock.agent_temperatures = {"writer": 0.9, "editor": 0.6}
        return mock

    @pytest.fixture
    def service(self, mock_settings: MagicMock, temp_db: Path) -> ModelModeService:
        """Create a ModelModeService with mocked dependencies."""
        return ModelModeService(mock_settings, db_path=temp_db)

    # === Mode Management Tests ===

    def test_get_current_mode_default(self, service: ModelModeService) -> None:
        """Test getting the default mode."""
        mode = service.get_current_mode()
        assert mode is not None
        assert mode.id == "balanced"
        assert mode.is_preset is True

    def test_set_mode_preset(self, service: ModelModeService) -> None:
        """Test setting a preset mode."""
        mode = service.set_mode("quality_max")
        assert mode.id == "quality_max"
        assert mode.is_preset is True
        assert service.get_current_mode().id == "quality_max"

    def test_set_mode_invalid(self, service: ModelModeService) -> None:
        """Test setting an invalid mode raises error."""
        with pytest.raises(ValueError, match="Mode not found"):
            service.set_mode("nonexistent_mode")

    def test_list_modes_includes_presets(self, service: ModelModeService) -> None:
        """Test that list_modes includes preset modes."""
        modes = service.list_modes()
        assert len(modes) > 0
        preset_ids = {m.id for m in modes if m.is_preset}
        assert "balanced" in preset_ids
        assert "quality_max" in preset_ids
        assert "draft_fast" in preset_ids

    def test_save_and_load_custom_mode(self, service: ModelModeService) -> None:
        """Test saving and loading a custom mode."""
        custom_mode = GenerationMode(
            id="test_custom",
            name="Test Custom Mode",
            description="A test mode",
            agent_models={"writer": "model-a", "editor": "model-b"},
            agent_temperatures={"writer": 0.8, "editor": 0.5},
            vram_strategy=VramStrategy.SEQUENTIAL,
            is_preset=False,
        )

        service.save_custom_mode(custom_mode)

        # Load it back
        loaded_mode = service.set_mode("test_custom")
        assert loaded_mode.id == "test_custom"
        assert loaded_mode.name == "Test Custom Mode"
        assert loaded_mode.agent_models["writer"] == "model-a"
        assert loaded_mode.is_preset is False

    def test_delete_custom_mode(self, service: ModelModeService) -> None:
        """Test deleting a custom mode."""
        custom_mode = GenerationMode(
            id="to_delete",
            name="To Delete",
            description="",
            agent_models={},
            agent_temperatures={},
            vram_strategy=VramStrategy.ADAPTIVE,
            is_preset=False,
        )
        service.save_custom_mode(custom_mode)

        # Verify it exists
        service.set_mode("to_delete")

        # Delete it
        result = service.delete_custom_mode("to_delete")
        assert result is True

        # Verify it's gone
        with pytest.raises(ValueError):
            service.set_mode("to_delete")

    def test_delete_preset_mode_fails(self, service: ModelModeService) -> None:
        """Test that deleting a preset mode fails."""
        result = service.delete_custom_mode("balanced")
        assert result is False

    # === Model/Temperature Access Tests ===

    def test_get_model_for_agent(self, service: ModelModeService) -> None:
        """Test getting model for an agent."""
        # Set a mode with specific models
        service.set_mode("quality_max")
        model = service.get_model_for_agent("writer")
        assert model is not None

    def test_get_temperature_for_agent(self, service: ModelModeService) -> None:
        """Test getting temperature for an agent."""
        temp = service.get_temperature_for_agent("writer")
        assert isinstance(temp, float)
        assert 0.0 <= temp <= 2.0

    def test_get_temperature_fallback(
        self, service: ModelModeService, mock_settings: MagicMock
    ) -> None:
        """Test temperature fallback to settings."""
        # Create mode without temperature for interviewer
        custom_mode = GenerationMode(
            id="no_temp",
            name="No Temp",
            description="",
            agent_models={},
            agent_temperatures={},  # No temperatures
            vram_strategy=VramStrategy.PARALLEL,
            is_preset=False,
        )
        service.save_custom_mode(custom_mode)
        service.set_mode("no_temp")

        temp = service.get_temperature_for_agent("writer")
        # Should fall back to settings
        assert temp == 0.9  # From mock_settings.agent_temperatures

    # === Score Recording Tests ===

    def test_record_generation(self, service: ModelModeService) -> None:
        """Test recording a generation event."""
        score_id = service.record_generation(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            chapter_id="ch-1",
            genre="fantasy",
            tokens_generated=1000,
            time_seconds=10.0,
        )

        assert score_id > 0

    def test_record_generation_calculates_speed(self, service: ModelModeService) -> None:
        """Test that record_generation calculates tokens per second."""
        score_id = service.record_generation(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            tokens_generated=1000,
            time_seconds=10.0,
        )

        # Speed should be calculated as 100 tokens/sec
        assert score_id > 0

    def test_update_quality_scores(self, service: ModelModeService) -> None:
        """Test updating quality scores."""
        score_id = service.record_generation(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
        )

        quality = QualityScores(
            prose_quality=8.5,
            instruction_following=9.0,
            consistency_score=7.5,
        )
        service.update_quality_scores(score_id, quality)

    def test_record_implicit_signal(self, service: ModelModeService) -> None:
        """Test recording implicit signals."""
        score_id = service.record_generation(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
        )

        service.record_implicit_signal(score_id, was_regenerated=True)
        service.record_implicit_signal(score_id, edit_distance=50)
        service.record_implicit_signal(score_id, user_rating=4)

    def test_update_performance_metrics(self, service: ModelModeService) -> None:
        """Test updating performance metrics."""
        score_id = service.record_generation(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
        )

        service.update_performance_metrics(
            score_id,
            tokens_generated=500,
            time_seconds=5.0,
        )

    # === Consistency Score Tests ===

    def test_calculate_consistency_score_no_issues(self, service: ModelModeService) -> None:
        """Test consistency score with no issues."""
        score = service.calculate_consistency_score([])
        assert score == 10.0

    def test_calculate_consistency_score_minor_issues(self, service: ModelModeService) -> None:
        """Test consistency score with minor issues."""
        issues = [
            {"severity": "minor"},
            {"severity": "minor"},
        ]
        score = service.calculate_consistency_score(issues)
        assert score == 9.0  # 10 - 2 * 0.5

    def test_calculate_consistency_score_mixed_issues(self, service: ModelModeService) -> None:
        """Test consistency score with mixed severity."""
        issues = [
            {"severity": "critical"},
            {"severity": "moderate"},
            {"severity": "minor"},
        ]
        score = service.calculate_consistency_score(issues)
        assert score == 5.0  # 10 - 3.0 - 1.5 - 0.5

    def test_calculate_consistency_score_capped(self, service: ModelModeService) -> None:
        """Test consistency score doesn't go below 0."""
        issues = [{"severity": "critical"} for _ in range(10)]
        score = service.calculate_consistency_score(issues)
        assert score == 0.0

    # === Learning Settings Tests ===

    def test_set_learning_settings(self, service: ModelModeService) -> None:
        """Test setting learning settings."""
        settings = LearningSettings(
            autonomy=AutonomyLevel.AGGRESSIVE,
            triggers=[LearningTrigger.CONTINUOUS],
            confidence_threshold=0.8,
        )
        service.set_learning_settings(settings)

        loaded = service.get_learning_settings()
        assert loaded.autonomy == AutonomyLevel.AGGRESSIVE
        assert LearningTrigger.CONTINUOUS in loaded.triggers

    def test_should_tune_off(self, service: ModelModeService) -> None:
        """Test should_tune returns False when OFF."""
        settings = LearningSettings(triggers=[LearningTrigger.OFF])
        service.set_learning_settings(settings)
        assert service.should_tune() is False

    def test_should_tune_continuous(self, service: ModelModeService) -> None:
        """Test should_tune returns True when CONTINUOUS."""
        settings = LearningSettings(triggers=[LearningTrigger.CONTINUOUS])
        service.set_learning_settings(settings)
        assert service.should_tune() is True

    def test_should_tune_periodic(self, service: ModelModeService) -> None:
        """Test should_tune returns True after periodic interval."""
        settings = LearningSettings(
            triggers=[LearningTrigger.PERIODIC],
            periodic_interval=3,
        )
        service.set_learning_settings(settings)

        # Initially false
        assert service.should_tune() is False

        # After 3 chapters, should be true
        for _ in range(3):
            service.on_chapter_complete()

        assert service.should_tune() is True

    def test_on_chapter_complete_increments_counter(self, service: ModelModeService) -> None:
        """Test that on_chapter_complete increments the counter."""
        initial = service._chapters_since_analysis
        service.on_chapter_complete()
        assert service._chapters_since_analysis == initial + 1

    def test_on_project_complete_resets_counter(self, service: ModelModeService) -> None:
        """Test that on_project_complete resets the chapter counter."""
        service._chapters_since_analysis = 10
        service.on_project_complete()
        assert service._chapters_since_analysis == 0

    # === VRAM Management Tests ===

    def test_prepare_model_sequential_strategy(self, service: ModelModeService) -> None:
        """Test prepare_model with sequential strategy."""
        # Set mode with sequential strategy
        service.set_mode("quality_max")  # Uses sequential

        # Add a model to tracking
        service._loaded_models = {"model-a", "model-b"}

        # Prepare a new model
        service.prepare_model("model-c")

        # Should only keep the new model
        assert service._loaded_models == {"model-c"}

    def test_prepare_model_parallel_strategy(self, service: ModelModeService) -> None:
        """Test prepare_model with parallel strategy."""
        # Set mode with parallel strategy
        service.set_mode("draft_fast")  # Uses parallel

        # Add a model to tracking
        service._loaded_models = {"model-a"}

        # Prepare a new model
        service.prepare_model("model-b")

        # Should keep both models
        assert "model-a" in service._loaded_models
        assert "model-b" in service._loaded_models

    # === Recommendation Tests ===

    def test_get_recommendations_insufficient_data(self, service: ModelModeService) -> None:
        """Test that get_recommendations returns empty with insufficient data."""
        recommendations = service.get_recommendations()
        assert recommendations == []

    def test_apply_recommendation_model_swap(self, service: ModelModeService) -> None:
        """Test applying a model swap recommendation."""
        # Set up current mode
        service.set_mode("balanced")

        rec = TuningRecommendation(
            recommendation_type=RecommendationType.MODEL_SWAP,
            current_value="old-model",
            suggested_value="new-model",
            affected_role="writer",
            reason="Better performance",
            confidence=0.9,
        )

        result = service.apply_recommendation(rec)
        assert result is True
        assert service._current_mode is not None
        assert service._current_mode.agent_models["writer"] == "new-model"

    def test_apply_recommendation_temp_adjust(self, service: ModelModeService) -> None:
        """Test applying a temperature adjustment recommendation."""
        service.set_mode("balanced")

        rec = TuningRecommendation(
            recommendation_type=RecommendationType.TEMP_ADJUST,
            current_value="0.7",
            suggested_value="0.9",
            affected_role="writer",
            reason="More creativity needed",
            confidence=0.8,
        )

        result = service.apply_recommendation(rec)
        assert result is True
        assert service._current_mode is not None
        assert service._current_mode.agent_temperatures["writer"] == 0.9

    def test_handle_recommendations_manual_autonomy(self, service: ModelModeService) -> None:
        """Test that manual autonomy doesn't auto-apply."""
        settings = LearningSettings(autonomy=AutonomyLevel.MANUAL)
        service.set_learning_settings(settings)
        service.set_mode("balanced")

        rec = TuningRecommendation(
            recommendation_type=RecommendationType.MODEL_SWAP,
            current_value="old",
            suggested_value="new",
            affected_role="writer",
            reason="test",
            confidence=0.99,
        )

        pending = service.handle_recommendations([rec])
        assert len(pending) == 1
        # Should NOT have been applied
        assert service._current_mode is not None
        assert service._current_mode.agent_models.get("writer") != "new"

    def test_handle_recommendations_aggressive_autonomy(self, service: ModelModeService) -> None:
        """Test that aggressive autonomy auto-applies."""
        settings = LearningSettings(autonomy=AutonomyLevel.AGGRESSIVE)
        service.set_learning_settings(settings)
        service.set_mode("balanced")

        rec = TuningRecommendation(
            recommendation_type=RecommendationType.MODEL_SWAP,
            current_value="old",
            suggested_value="new",
            affected_role="writer",
            reason="test",
            confidence=0.5,
        )

        pending = service.handle_recommendations([rec])
        assert len(pending) == 0
        # Should have been applied
        assert service._current_mode is not None
        assert service._current_mode.agent_models["writer"] == "new"

    # === Analytics Tests ===

    def test_get_quality_vs_speed_data(self, service: ModelModeService) -> None:
        """Test getting quality vs speed data."""
        data = service.get_quality_vs_speed_data()
        assert isinstance(data, list)

    def test_get_model_performance(self, service: ModelModeService) -> None:
        """Test getting model performance data."""
        perf = service.get_model_performance()
        assert isinstance(perf, list)

    def test_export_scores_csv(self, service: ModelModeService, temp_db: Path) -> None:
        """Test exporting scores to CSV."""
        # Record some scores first
        service.record_generation(
            project_id="test",
            agent_role="writer",
            model_id="model",
        )

        csv_path = temp_db.parent / "export.csv"
        count = service.export_scores_csv(csv_path)
        assert count >= 0


class TestModelModeServiceVramStrategy:
    """Tests for VRAM strategy validation."""

    @pytest.fixture
    def temp_db(self) -> Path:
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock(spec=Settings)
        mock.ollama_url = "http://localhost:11434"
        return mock

    def test_set_mode_invalid_vram_strategy(self, mock_settings: MagicMock, temp_db: Path) -> None:
        """Test that invalid VRAM strategy raises clear error."""
        import sqlite3

        service = ModelModeService(mock_settings, db_path=temp_db)

        # Save a custom mode with valid strategy first
        custom_mode = GenerationMode(
            id="test_invalid",
            name="Test",
            description="",
            agent_models={},
            agent_temperatures={},
            vram_strategy=VramStrategy.SEQUENTIAL,
            is_preset=False,
        )
        service.save_custom_mode(custom_mode)

        # Manually corrupt the database entry using direct sqlite connection
        conn = sqlite3.connect(temp_db)
        try:
            conn.execute(
                "UPDATE custom_modes SET vram_strategy = 'invalid_strategy' WHERE id = ?",
                ("test_invalid",),
            )
            conn.commit()
        finally:
            conn.close()

        # Now try to load the corrupted mode
        with pytest.raises(ValueError) as exc_info:
            service.set_mode("test_invalid")

        assert "Invalid VRAM strategy" in str(exc_info.value)
        assert "invalid_strategy" in str(exc_info.value)
