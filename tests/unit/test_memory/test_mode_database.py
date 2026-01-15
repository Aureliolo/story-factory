"""Unit tests for ModeDatabase."""

from pathlib import Path

import pytest

from memory.mode_database import ModeDatabase
from memory.mode_models import GenerationScore, ModelPerformanceSummary, TuningRecommendation


class TestModeDatabase:
    """Tests for ModeDatabase."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> ModeDatabase:
        """Create a test database using pytest's tmp_path fixture."""
        db_path = tmp_path / "test_scores.db"
        return ModeDatabase(db_path)

    def test_init_creates_tables(self, db: ModeDatabase) -> None:
        """Test that database initialization creates all required tables."""
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

        assert "generation_scores" in tables
        assert "model_performance" in tables
        assert "recommendations" in tables
        assert "custom_modes" in tables

    def test_record_score(self, db: ModeDatabase) -> None:
        """Test recording a generation score."""
        score_id = db.record_score(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            mode_name="balanced",
            chapter_id="ch-1",
            genre="fantasy",
            prose_quality=8.5,
        )

        assert score_id > 0

        # Verify it's stored
        scores = db.get_scores_for_model("test-model")
        assert len(scores) == 1
        assert scores[0]["prose_quality"] == 8.5

    def test_update_score(self, db: ModeDatabase) -> None:
        """Test updating an existing score."""
        score_id = db.record_score(
            project_id="test-project",
            agent_role="writer",
            model_id="test-model",
            mode_name="balanced",
        )

        db.update_score(
            score_id,
            prose_quality=9.0,
            instruction_following=8.5,
            was_regenerated=True,
        )

        scores = db.get_scores_for_model("test-model")
        assert scores[0]["prose_quality"] == 9.0
        assert scores[0]["instruction_following"] == 8.5
        assert scores[0]["was_regenerated"] == 1

    def test_get_score_count(self, db: ModeDatabase) -> None:
        """Test counting scores with filters."""
        # Add some scores
        for _ in range(5):
            db.record_score(
                project_id="test-project",
                agent_role="writer",
                model_id="model-a",
                mode_name="balanced",
            )
        for _ in range(3):
            db.record_score(
                project_id="test-project",
                agent_role="editor",
                model_id="model-b",
                mode_name="balanced",
            )

        assert db.get_score_count() == 8
        assert db.get_score_count(model_id="model-a") == 5
        assert db.get_score_count(agent_role="editor") == 3
        assert db.get_score_count(model_id="model-a", agent_role="writer") == 5

    def test_record_recommendation(self, db: ModeDatabase) -> None:
        """Test recording a tuning recommendation."""
        rec_id = db.record_recommendation(
            recommendation_type="model_swap",
            current_value="model-a",
            suggested_value="model-b",
            reason="Better quality scores",
            confidence=0.85,
            affected_role="writer",
        )

        assert rec_id > 0

        recs = db.get_pending_recommendations()
        assert len(recs) == 1
        assert recs[0]["suggested_value"] == "model-b"

    def test_update_recommendation_outcome(self, db: ModeDatabase) -> None:
        """Test updating recommendation outcome."""
        rec_id = db.record_recommendation(
            recommendation_type="temp_adjust",
            current_value="0.8",
            suggested_value="0.9",
            reason="Increase creativity",
            confidence=0.75,
        )

        db.update_recommendation_outcome(rec_id, was_applied=True, user_feedback="accepted")

        recs = db.get_pending_recommendations()
        assert len(recs) == 0  # No longer pending

        history = db.get_recommendation_history()
        assert history[0]["was_applied"] == 1
        assert history[0]["user_feedback"] == "accepted"

    def test_save_and_get_custom_mode(self, db: ModeDatabase) -> None:
        """Test saving and retrieving custom modes."""
        db.save_custom_mode(
            mode_id="my-custom",
            name="My Custom Mode",
            agent_models={"writer": "model-a", "editor": "model-b"},
            agent_temperatures={"writer": 0.9, "editor": 0.6},
            description="A custom mode",
        )

        mode = db.get_custom_mode("my-custom")
        assert mode is not None
        assert mode["name"] == "My Custom Mode"
        assert mode["agent_models"]["writer"] == "model-a"
        assert mode["agent_temperatures"]["editor"] == 0.6

    def test_list_custom_modes(self, db: ModeDatabase) -> None:
        """Test listing custom modes."""
        db.save_custom_mode(
            mode_id="mode-1",
            name="Mode One",
            agent_models={"writer": "model-a"},
            agent_temperatures={"writer": 0.8},
        )
        db.save_custom_mode(
            mode_id="mode-2",
            name="Mode Two",
            agent_models={"writer": "model-b"},
            agent_temperatures={"writer": 0.9},
        )

        modes = db.list_custom_modes()
        assert len(modes) == 2
        names = {m["name"] for m in modes}
        assert "Mode One" in names
        assert "Mode Two" in names

    def test_delete_custom_mode(self, db: ModeDatabase) -> None:
        """Test deleting a custom mode."""
        db.save_custom_mode(
            mode_id="to-delete",
            name="Delete Me",
            agent_models={"writer": "model-a"},
            agent_temperatures={"writer": 0.8},
        )

        assert db.delete_custom_mode("to-delete") is True
        assert db.get_custom_mode("to-delete") is None
        assert db.delete_custom_mode("nonexistent") is False

    def test_get_unique_genres(self, db: ModeDatabase) -> None:
        """Test getting unique genres."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            genre="fantasy",
        )
        db.record_score(
            project_id="p2",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            genre="sci-fi",
        )
        db.record_score(
            project_id="p3",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            genre="fantasy",  # Duplicate
        )

        genres = db.get_unique_genres()
        assert set(genres) == {"fantasy", "sci-fi"}

    def test_get_average_score(self, db: ModeDatabase) -> None:
        """Test getting average scores."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="p2",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            prose_quality=9.0,
        )

        avg = db.get_average_score("prose_quality")
        assert avg == 8.5

        # Invalid metric returns None
        assert db.get_average_score("invalid_metric") is None

    def test_get_model_summaries(self, db: ModeDatabase) -> None:
        """Test getting model performance summaries."""
        # Record some scores
        for i in range(3):
            db.record_score(
                project_id=f"p{i}",
                agent_role="writer",
                model_id="model-a",
                mode_name="balanced",
                prose_quality=8.0 + i * 0.5,
            )

        # Update aggregates
        db.update_model_performance("model-a", "writer")

        summaries = db.get_model_summaries()
        assert len(summaries) == 1
        assert isinstance(summaries[0], ModelPerformanceSummary)
        assert summaries[0].model_id == "model-a"
        assert summaries[0].sample_count == 3

    def test_get_recent_recommendations(self, db: ModeDatabase) -> None:
        """Test getting recent recommendations as objects."""
        db.record_recommendation(
            recommendation_type="model_swap",
            current_value="old",
            suggested_value="new",
            reason="Better",
            confidence=0.9,
        )

        recs = db.get_recent_recommendations(limit=5)
        assert len(recs) == 1
        assert isinstance(recs[0], TuningRecommendation)
        assert recs[0].current_value == "old"
        assert recs[0].confidence == 0.9

    def test_get_all_scores(self, db: ModeDatabase) -> None:
        """Test getting all scores as objects."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            prose_quality=8.5,
        )

        scores = db.get_all_scores()
        assert len(scores) == 1
        assert isinstance(scores[0], GenerationScore)
        assert scores[0].project_id == "p1"
        assert scores[0].quality.prose_quality == 8.5
