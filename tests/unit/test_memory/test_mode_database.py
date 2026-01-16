"""Unit tests for ModeDatabase."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

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

    # === Exception Handling Tests ===

    def test_record_score_raises_on_db_error(self, tmp_path: Path) -> None:
        """Test record_score raises sqlite3.Error on database failure."""
        db = ModeDatabase(tmp_path / "test.db")

        with patch.object(db, "db_path", "/nonexistent/path/db.db"):
            with pytest.raises(sqlite3.Error):
                db.record_score(
                    project_id="test",
                    agent_role="writer",
                    model_id="model",
                    mode_name="balanced",
                )

    def test_update_score_raises_on_db_error(self, db: ModeDatabase) -> None:
        """Test update_score raises sqlite3.Error on database failure."""
        score_id = db.record_score(
            project_id="test",
            agent_role="writer",
            model_id="model",
            mode_name="balanced",
        )

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database locked")
            with pytest.raises(sqlite3.Error, match="Database locked"):
                db.update_score(score_id, prose_quality=8.0)

    def test_update_score_noop_with_no_changes(self, db: ModeDatabase) -> None:
        """Test update_score returns early when no fields provided."""
        score_id = db.record_score(
            project_id="test",
            agent_role="writer",
            model_id="model",
            mode_name="balanced",
        )
        # Should not raise, just return
        db.update_score(score_id)

    def test_update_performance_metrics_raises_on_db_error(self, db: ModeDatabase) -> None:
        """Test update_performance_metrics raises sqlite3.Error on database failure."""
        score_id = db.record_score(
            project_id="test",
            agent_role="writer",
            model_id="model",
            mode_name="balanced",
        )

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")
            with pytest.raises(sqlite3.Error, match="Database error"):
                db.update_performance_metrics(score_id, tokens_generated=100)

    def test_update_performance_metrics_noop_with_no_changes(self, db: ModeDatabase) -> None:
        """Test update_performance_metrics returns early when no fields provided."""
        score_id = db.record_score(
            project_id="test",
            agent_role="writer",
            model_id="model",
            mode_name="balanced",
        )
        # Should not raise, just return
        db.update_performance_metrics(score_id)

    def test_record_recommendation_raises_on_db_error(self, db: ModeDatabase) -> None:
        """Test record_recommendation raises sqlite3.Error on database failure."""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Connection failed")
            with pytest.raises(sqlite3.Error, match="Connection failed"):
                db.record_recommendation(
                    recommendation_type="model_swap",
                    current_value="old",
                    suggested_value="new",
                    reason="test",
                    confidence=0.9,
                )

    def test_save_custom_mode_raises_on_db_error(self, db: ModeDatabase) -> None:
        """Test save_custom_mode raises sqlite3.Error on database failure."""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Write failed")
            with pytest.raises(sqlite3.Error, match="Write failed"):
                db.save_custom_mode(
                    mode_id="test",
                    name="Test",
                    agent_models={"writer": "model"},
                    agent_temperatures={"writer": 0.8},
                )

    # === Query Filter Tests ===

    def test_get_scores_for_model_with_filters(self, db: ModeDatabase) -> None:
        """Test get_scores_for_model with agent_role and genre filters."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            genre="fantasy",
        )
        db.record_score(
            project_id="p2",
            agent_role="editor",
            model_id="model-a",
            mode_name="balanced",
            genre="sci-fi",
        )

        # Filter by agent_role
        writer_scores = db.get_scores_for_model("model-a", agent_role="writer")
        assert len(writer_scores) == 1
        assert writer_scores[0]["agent_role"] == "writer"

        # Filter by genre
        fantasy_scores = db.get_scores_for_model("model-a", genre="fantasy")
        assert len(fantasy_scores) == 1
        assert fantasy_scores[0]["genre"] == "fantasy"

    def test_get_scores_for_project(self, db: ModeDatabase) -> None:
        """Test get_scores_for_project returns all scores for a project."""
        for i in range(3):
            db.record_score(
                project_id="project-1",
                agent_role="writer",
                model_id=f"model-{i}",
                mode_name="balanced",
            )
        db.record_score(
            project_id="project-2",
            agent_role="writer",
            model_id="model-x",
            mode_name="balanced",
        )

        scores = db.get_scores_for_project("project-1")
        assert len(scores) == 3
        assert all(s["project_id"] == "project-1" for s in scores)

    def test_get_score_count_with_genre_filter(self, db: ModeDatabase) -> None:
        """Test get_score_count with genre filter."""
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

        assert db.get_score_count(genre="fantasy") == 1
        assert db.get_score_count(genre="sci-fi") == 1

    def test_get_model_performance_with_filters(self, db: ModeDatabase) -> None:
        """Test get_model_performance with model_id, agent_role, and genre filters."""
        # Record scores and update aggregates
        for i in range(3):
            db.record_score(
                project_id=f"p{i}",
                agent_role="writer",
                model_id="model-a",
                mode_name="balanced",
                genre="fantasy",
                prose_quality=8.0,
            )
        db.update_model_performance("model-a", "writer", genre="fantasy")

        # Filter by model_id
        perf = db.get_model_performance(model_id="model-a")
        assert len(perf) == 1

        # Filter by agent_role
        perf = db.get_model_performance(agent_role="writer")
        assert len(perf) == 1

        # Filter by genre
        perf = db.get_model_performance(genre="fantasy")
        assert len(perf) == 1

    def test_get_quality_vs_speed_data_with_role_filter(self, db: ModeDatabase) -> None:
        """Test get_quality_vs_speed_data with agent_role filter."""
        # Record scores and update aggregates
        for i in range(5):
            db.record_score(
                project_id=f"p{i}",
                agent_role="writer",
                model_id="model-a",
                mode_name="balanced",
                prose_quality=8.0,
                tokens_per_second=50.0,
            )
        db.update_model_performance("model-a", "writer")

        data = db.get_quality_vs_speed_data(agent_role="writer")
        assert len(data) == 1
        assert data[0]["agent_role"] == "writer"

    def test_get_genre_breakdown(self, db: ModeDatabase) -> None:
        """Test get_genre_breakdown returns breakdown by genre."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            genre="fantasy",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="p2",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            genre="fantasy",
            prose_quality=9.0,
        )
        db.record_score(
            project_id="p3",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            genre="sci-fi",
            prose_quality=7.0,
        )

        breakdown = db.get_genre_breakdown("model-a")
        assert len(breakdown) == 2

        fantasy = next(g for g in breakdown if g["genre"] == "fantasy")
        assert fantasy["avg_quality"] == 8.5
        assert fantasy["sample_count"] == 2

    def test_get_average_score_with_filters(self, db: ModeDatabase) -> None:
        """Test get_average_score with agent_role and genre filters."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            genre="fantasy",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="p2",
            agent_role="editor",
            model_id="m1",
            mode_name="balanced",
            genre="fantasy",
            prose_quality=6.0,
        )

        # Filter by agent_role
        avg = db.get_average_score("prose_quality", agent_role="writer")
        assert avg == 8.0

        # Filter by genre
        avg = db.get_average_score("prose_quality", genre="fantasy")
        assert avg == 7.0  # Average of 8.0 and 6.0

    def test_get_model_summaries_with_filters(self, db: ModeDatabase) -> None:
        """Test get_model_summaries with agent_role and genre filters."""
        for i in range(3):
            db.record_score(
                project_id=f"p{i}",
                agent_role="writer",
                model_id="model-a",
                mode_name="balanced",
                genre="fantasy",
                prose_quality=8.0,
            )
        db.update_model_performance("model-a", "writer", genre="fantasy")

        # Filter by agent_role
        summaries = db.get_model_summaries(agent_role="writer")
        assert len(summaries) == 1

        # Filter by genre
        summaries = db.get_model_summaries(genre="fantasy")
        assert len(summaries) == 1

    def test_get_all_scores_with_filters(self, db: ModeDatabase) -> None:
        """Test get_all_scores with agent_role and genre filters."""
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="m1",
            mode_name="balanced",
            genre="fantasy",
        )
        db.record_score(
            project_id="p2",
            agent_role="editor",
            model_id="m1",
            mode_name="balanced",
            genre="sci-fi",
        )

        # Filter by agent_role
        scores = db.get_all_scores(agent_role="writer")
        assert len(scores) == 1
        assert scores[0].agent_role == "writer"

        # Filter by genre
        scores = db.get_all_scores(genre="sci-fi")
        assert len(scores) == 1
        assert scores[0].genre == "sci-fi"

    def test_get_recent_recommendations_invalid_type(self, db: ModeDatabase) -> None:
        """Test get_recent_recommendations handles invalid recommendation type."""
        import sqlite3 as sql

        # Insert a recommendation with invalid type directly
        with sql.connect(db.db_path) as conn:
            conn.execute(
                """
                INSERT INTO recommendations (
                    recommendation_type, current_value, suggested_value, reason, confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("invalid_type", "old", "new", "test", 0.8),
            )
            conn.commit()

        recs = db.get_recent_recommendations()
        assert len(recs) == 1
        # Should fallback to MODEL_SWAP
        from memory.mode_models import RecommendationType

        assert recs[0].recommendation_type == RecommendationType.MODEL_SWAP

    def test_export_scores_csv_empty(self, db: ModeDatabase, tmp_path: Path) -> None:
        """Test export_scores_csv returns 0 for empty database."""
        output = tmp_path / "export.csv"
        count = db.export_scores_csv(output)
        assert count == 0

    def test_export_scores_csv_with_data(self, db: ModeDatabase, tmp_path: Path) -> None:
        """Test export_scores_csv writes data and returns row count."""
        # Add some scores
        db.record_score(
            project_id="p1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.5,
        )
        db.record_score(
            project_id="p2",
            agent_role="editor",
            model_id="model-b",
            mode_name="quality",
            prose_quality=9.0,
        )

        output = tmp_path / "export.csv"
        count = db.export_scores_csv(output)

        assert count == 2
        assert output.exists()

        # Verify CSV content
        import csv

        with open(output, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["prose_quality"] == "8.5"
            assert rows[1]["prose_quality"] == "9.0"

    def test_update_score_all_optional_fields(self, db: ModeDatabase) -> None:
        """Test update_score with consistency_score, edit_distance, and user_rating."""
        score_id = db.record_score(
            project_id="test",
            agent_role="writer",
            model_id="model",
            mode_name="balanced",
        )

        db.update_score(
            score_id,
            consistency_score=7.5,
            edit_distance=150,
            user_rating=4,
        )

        scores = db.get_scores_for_model("model")
        assert scores[0]["consistency_score"] == 7.5
        assert scores[0]["edit_distance"] == 150
        assert scores[0]["user_rating"] == 4

    def test_update_performance_metrics_all_fields(self, db: ModeDatabase) -> None:
        """Test update_performance_metrics with all optional fields."""
        score_id = db.record_score(
            project_id="test",
            agent_role="writer",
            model_id="model",
            mode_name="balanced",
        )

        db.update_performance_metrics(
            score_id,
            tokens_generated=500,
            time_seconds=10.5,
            tokens_per_second=47.6,
            vram_used_gb=4.2,
        )

        scores = db.get_scores_for_model("model")
        assert scores[0]["tokens_generated"] == 500
        assert scores[0]["time_seconds"] == 10.5
        assert scores[0]["tokens_per_second"] == 47.6
        assert scores[0]["vram_used_gb"] == 4.2

    def test_get_top_models_for_role(self, db: ModeDatabase) -> None:
        """Test get_top_models_for_role returns top performers."""
        # Create multiple models with different performance
        for model, quality in [("model-a", 8.0), ("model-b", 9.0), ("model-c", 7.5)]:
            for i in range(5):  # min_samples is 3 by default
                db.record_score(
                    project_id=f"p-{model}-{i}",
                    agent_role="writer",
                    model_id=model,
                    mode_name="balanced",
                    prose_quality=quality,
                )
            db.update_model_performance(model, "writer")

        top = db.get_top_models_for_role("writer", limit=2, min_samples=3)

        assert len(top) == 2
        # Should be sorted by avg_prose_quality DESC
        assert top[0]["model_id"] == "model-b"  # 9.0 avg
        assert top[1]["model_id"] == "model-a"  # 8.0 avg

    def test_get_top_models_for_role_respects_min_samples(self, db: ModeDatabase) -> None:
        """Test get_top_models_for_role filters by minimum sample count."""
        # Add model with only 2 samples (below min)
        for i in range(2):
            db.record_score(
                project_id=f"p-few-{i}",
                agent_role="writer",
                model_id="model-few",
                mode_name="balanced",
                prose_quality=10.0,  # High quality but too few samples
            )
        db.update_model_performance("model-few", "writer")

        # Add model with 5 samples (above min)
        for i in range(5):
            db.record_score(
                project_id=f"p-enough-{i}",
                agent_role="writer",
                model_id="model-enough",
                mode_name="balanced",
                prose_quality=7.0,
            )
        db.update_model_performance("model-enough", "writer")

        top = db.get_top_models_for_role("writer", min_samples=3)

        # Only model-enough should be returned (has >= 3 samples)
        assert len(top) == 1
        assert top[0]["model_id"] == "model-enough"
