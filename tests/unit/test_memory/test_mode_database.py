"""Unit tests for ModeDatabase."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from src.memory.mode_database import ModeDatabase
from src.memory.mode_models import GenerationScore, ModelPerformanceSummary, TuningRecommendation


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
        from src.memory.mode_models import RecommendationType

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

    # === World Entity Scores Tests ===

    def test_record_world_entity_score(self, db: ModeDatabase) -> None:
        """Test recording a world entity score."""
        scores = {"depth": 8.0, "consistency": 7.5, "creativity": 9.0, "average": 8.17}
        score_id = db.record_world_entity_score(
            project_id="test-project",
            entity_type="character",
            entity_name="Hero",
            model_id="test-model",
            scores=scores,
            entity_id="char-001",
            iterations_used=3,
            generation_time_seconds=5.5,
            feedback="Good character development",
        )

        assert score_id > 0

        # Verify it's stored
        entity_scores = db.get_world_entity_scores(project_id="test-project")
        assert len(entity_scores) == 1
        assert entity_scores[0]["entity_name"] == "Hero"
        assert entity_scores[0]["entity_type"] == "character"
        assert entity_scores[0]["score_1"] == 8.0
        assert entity_scores[0]["average_score"] == 8.17
        assert entity_scores[0]["iterations_used"] == 3

    def test_record_world_entity_score_raises_on_db_error(self, db: ModeDatabase) -> None:
        """Test record_world_entity_score raises sqlite3.Error on database failure."""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database write failed")
            with pytest.raises(sqlite3.Error, match="Database write failed"):
                db.record_world_entity_score(
                    project_id="test",
                    entity_type="character",
                    entity_name="Test",
                    model_id="model",
                    scores={"score": 8.0},
                )

    def test_record_world_entity_score_with_fewer_scores(self, db: ModeDatabase) -> None:
        """Test record_world_entity_score with fewer than 4 score values."""
        # Only 2 scores
        scores = {"depth": 7.5, "consistency": 8.0}
        score_id = db.record_world_entity_score(
            project_id="test-project",
            entity_type="location",
            entity_name="Castle",
            model_id="test-model",
            scores=scores,
        )

        assert score_id > 0

        entity_scores = db.get_world_entity_scores(project_id="test-project")
        assert len(entity_scores) == 1
        assert entity_scores[0]["score_1"] == 7.5
        assert entity_scores[0]["score_2"] == 8.0
        assert entity_scores[0]["score_3"] is None
        assert entity_scores[0]["score_4"] is None

    def test_get_world_entity_scores_with_all_filters(self, db: ModeDatabase) -> None:
        """Test get_world_entity_scores with project_id, entity_type, and model_id filters."""
        # Add various entity scores
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"quality": 8.0},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="location",
            entity_name="Castle",
            model_id="model-a",
            scores={"quality": 7.5},
        )
        db.record_world_entity_score(
            project_id="project-2",
            entity_type="character",
            entity_name="Villain",
            model_id="model-b",
            scores={"quality": 9.0},
        )

        # Filter by project_id
        scores = db.get_world_entity_scores(project_id="project-1")
        assert len(scores) == 2

        # Filter by entity_type
        scores = db.get_world_entity_scores(entity_type="character")
        assert len(scores) == 2
        assert all(s["entity_type"] == "character" for s in scores)

        # Filter by model_id
        scores = db.get_world_entity_scores(model_id="model-b")
        assert len(scores) == 1
        assert scores[0]["entity_name"] == "Villain"

        # Combined filters
        scores = db.get_world_entity_scores(project_id="project-1", entity_type="character")
        assert len(scores) == 1
        assert scores[0]["entity_name"] == "Hero"

    def test_get_world_entity_scores_respects_limit(self, db: ModeDatabase) -> None:
        """Test get_world_entity_scores respects the limit parameter."""
        for i in range(10):
            db.record_world_entity_score(
                project_id="project-1",
                entity_type="character",
                entity_name=f"Character-{i}",
                model_id="model-a",
                scores={"quality": 8.0},
            )

        scores = db.get_world_entity_scores(limit=5)
        assert len(scores) == 5

    def test_get_world_quality_summary_basic(self, db: ModeDatabase) -> None:
        """Test get_world_quality_summary returns correct summary statistics."""
        # Add various entity scores
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"average": 8.0},
            iterations_used=2,
            generation_time_seconds=3.0,
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Villain",
            model_id="model-a",
            scores={"average": 9.0},
            iterations_used=4,
            generation_time_seconds=5.0,
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="location",
            entity_name="Castle",
            model_id="model-b",
            scores={"average": 7.0},
            iterations_used=3,
            generation_time_seconds=4.0,
        )

        summary = db.get_world_quality_summary()

        # Overall statistics
        assert summary["total_entities"] == 3
        assert summary["avg_quality"] == 8.0  # (8 + 9 + 7) / 3
        assert summary["min_quality"] == 7.0
        assert summary["max_quality"] == 9.0
        assert summary["avg_iterations"] == 3.0  # (2 + 4 + 3) / 3
        assert summary["avg_generation_time"] == 4.0  # (3 + 5 + 4) / 3

        # By entity type breakdown
        assert len(summary["by_entity_type"]) == 2
        char_breakdown = next(
            b for b in summary["by_entity_type"] if b["entity_type"] == "character"
        )
        assert char_breakdown["count"] == 2
        assert char_breakdown["avg_quality"] == 8.5  # (8 + 9) / 2

        loc_breakdown = next(b for b in summary["by_entity_type"] if b["entity_type"] == "location")
        assert loc_breakdown["count"] == 1
        assert loc_breakdown["avg_quality"] == 7.0

        # By model breakdown
        assert len(summary["by_model"]) == 2
        model_a_breakdown = next(b for b in summary["by_model"] if b["model_id"] == "model-a")
        assert model_a_breakdown["count"] == 2
        assert model_a_breakdown["avg_quality"] == 8.5

    def test_get_world_quality_summary_with_entity_type_filter(self, db: ModeDatabase) -> None:
        """Test get_world_quality_summary with entity_type filter."""
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"average": 8.0},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="location",
            entity_name="Castle",
            model_id="model-a",
            scores={"average": 6.0},
        )

        summary = db.get_world_quality_summary(entity_type="character")

        assert summary["total_entities"] == 1
        assert summary["avg_quality"] == 8.0

    def test_get_world_quality_summary_with_model_filter(self, db: ModeDatabase) -> None:
        """Test get_world_quality_summary with model_id filter."""
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"average": 8.0},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Villain",
            model_id="model-b",
            scores={"average": 9.0},
        )

        summary = db.get_world_quality_summary(model_id="model-a")

        assert summary["total_entities"] == 1
        assert summary["avg_quality"] == 8.0

    def test_get_world_quality_summary_empty_database(self, db: ModeDatabase) -> None:
        """Test get_world_quality_summary with no data."""
        summary = db.get_world_quality_summary()

        assert summary["total_entities"] == 0
        assert summary["avg_quality"] is None
        assert summary["min_quality"] is None
        assert summary["max_quality"] is None
        assert summary["by_entity_type"] == []
        assert summary["by_model"] == []

    def test_get_world_entity_count_basic(self, db: ModeDatabase) -> None:
        """Test get_world_entity_count returns correct count."""
        for i in range(5):
            db.record_world_entity_score(
                project_id="project-1",
                entity_type="character",
                entity_name=f"Character-{i}",
                model_id="model-a",
                scores={"quality": 8.0},
            )

        assert db.get_world_entity_count() == 5

    def test_get_world_entity_count_with_entity_type_filter(self, db: ModeDatabase) -> None:
        """Test get_world_entity_count with entity_type filter."""
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"quality": 8.0},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="location",
            entity_name="Castle",
            model_id="model-a",
            scores={"quality": 7.5},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Villain",
            model_id="model-a",
            scores={"quality": 9.0},
        )

        assert db.get_world_entity_count(entity_type="character") == 2
        assert db.get_world_entity_count(entity_type="location") == 1

    def test_get_world_entity_count_with_model_filter(self, db: ModeDatabase) -> None:
        """Test get_world_entity_count with model_id filter."""
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"quality": 8.0},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Villain",
            model_id="model-b",
            scores={"quality": 9.0},
        )

        assert db.get_world_entity_count(model_id="model-a") == 1
        assert db.get_world_entity_count(model_id="model-b") == 1

    def test_get_world_entity_count_combined_filters(self, db: ModeDatabase) -> None:
        """Test get_world_entity_count with combined filters."""
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Hero",
            model_id="model-a",
            scores={"quality": 8.0},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="location",
            entity_name="Castle",
            model_id="model-a",
            scores={"quality": 7.5},
        )
        db.record_world_entity_score(
            project_id="project-1",
            entity_type="character",
            entity_name="Villain",
            model_id="model-b",
            scores={"quality": 9.0},
        )

        assert db.get_world_entity_count(entity_type="character", model_id="model-a") == 1

    # === Content Statistics Tests ===

    def test_get_content_statistics_basic(self, db: ModeDatabase) -> None:
        """Test get_content_statistics returns correct statistics."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=500,
            time_seconds=10.0,
        )
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=300,
            time_seconds=6.0,
        )
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=700,
            time_seconds=14.0,
        )

        stats = db.get_content_statistics()

        assert stats["generation_count"] == 3
        assert stats["total_tokens"] == 1500
        assert stats["avg_tokens"] == 500.0
        assert stats["min_tokens"] == 300
        assert stats["max_tokens"] == 700
        assert stats["avg_generation_time"] == 10.0

    def test_get_content_statistics_with_project_filter(self, db: ModeDatabase) -> None:
        """Test get_content_statistics with project_id filter."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=500,
            time_seconds=10.0,
        )
        db.record_score(
            project_id="project-2",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=300,
            time_seconds=6.0,
        )

        stats = db.get_content_statistics(project_id="project-1")

        assert stats["generation_count"] == 1
        assert stats["total_tokens"] == 500

    def test_get_content_statistics_with_agent_role_filter(self, db: ModeDatabase) -> None:
        """Test get_content_statistics with agent_role filter."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=500,
            time_seconds=10.0,
        )
        db.record_score(
            project_id="project-1",
            agent_role="editor",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=200,
            time_seconds=4.0,
        )

        stats = db.get_content_statistics(agent_role="writer")

        assert stats["generation_count"] == 1
        assert stats["total_tokens"] == 500

    def test_get_content_statistics_excludes_null_tokens(self, db: ModeDatabase) -> None:
        """Test get_content_statistics excludes records with null tokens_generated."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            tokens_generated=500,
        )
        # This one has no tokens_generated
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
        )

        stats = db.get_content_statistics()

        assert stats["generation_count"] == 1
        assert stats["total_tokens"] == 500

    def test_get_content_statistics_empty_database(self, db: ModeDatabase) -> None:
        """Test get_content_statistics with no data."""
        stats = db.get_content_statistics()

        assert stats["generation_count"] == 0
        assert stats["total_tokens"] == 0
        assert stats["avg_tokens"] is None

    # === Time Series Tests ===

    def test_get_quality_time_series_basic(self, db: ModeDatabase) -> None:
        """Test get_quality_time_series returns time series data."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=9.0,
        )

        data = db.get_quality_time_series(metric="prose_quality")

        assert len(data) == 2
        assert all("timestamp" in d and "value" in d for d in data)
        assert data[0]["value"] == 9.0  # Most recent first (DESC order)
        assert data[1]["value"] == 8.0

    def test_get_quality_time_series_with_agent_role_filter(self, db: ModeDatabase) -> None:
        """Test get_quality_time_series with agent_role filter."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="project-1",
            agent_role="editor",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=7.0,
        )

        data = db.get_quality_time_series(agent_role="writer")

        assert len(data) == 1
        assert data[0]["value"] == 8.0

    def test_get_quality_time_series_with_genre_filter(self, db: ModeDatabase) -> None:
        """Test get_quality_time_series with genre filter."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            genre="fantasy",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            genre="sci-fi",
            prose_quality=9.0,
        )

        data = db.get_quality_time_series(genre="fantasy")

        assert len(data) == 1
        assert data[0]["value"] == 8.0

    def test_get_quality_time_series_invalid_metric(self, db: ModeDatabase) -> None:
        """Test get_quality_time_series defaults to prose_quality for invalid metric."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.5,
        )

        # Invalid metric should default to prose_quality
        data = db.get_quality_time_series(metric="invalid_metric")

        assert len(data) == 1
        assert data[0]["value"] == 8.5

    def test_get_quality_time_series_respects_limit(self, db: ModeDatabase) -> None:
        """Test get_quality_time_series respects the limit parameter."""
        for i in range(10):
            db.record_score(
                project_id=f"project-{i}",
                agent_role="writer",
                model_id="model-a",
                mode_name="balanced",
                prose_quality=8.0 + i * 0.1,
            )

        data = db.get_quality_time_series(limit=5)

        assert len(data) == 5

    def test_get_quality_time_series_different_metrics(self, db: ModeDatabase) -> None:
        """Test get_quality_time_series with different valid metrics."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
            instruction_following=7.5,
            consistency_score=9.0,
            tokens_per_second=50.0,
        )

        # Test instruction_following
        data = db.get_quality_time_series(metric="instruction_following")
        assert len(data) == 1
        assert data[0]["value"] == 7.5

        # Test consistency_score
        data = db.get_quality_time_series(metric="consistency_score")
        assert len(data) == 1
        assert data[0]["value"] == 9.0

        # Test tokens_per_second
        data = db.get_quality_time_series(metric="tokens_per_second")
        assert len(data) == 1
        assert data[0]["value"] == 50.0

    # === Daily Quality Averages Tests ===

    def test_get_daily_quality_averages_basic(self, db: ModeDatabase) -> None:
        """Test get_daily_quality_averages returns daily averages."""
        # Record scores (they'll all be on the same day since we can't easily mock time)
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="project-2",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=9.0,
        )

        data = db.get_daily_quality_averages()

        assert len(data) == 1
        assert "date" in data[0]
        assert data[0]["avg_value"] == 8.5  # (8 + 9) / 2
        assert data[0]["sample_count"] == 2

    def test_get_daily_quality_averages_with_agent_role_filter(self, db: ModeDatabase) -> None:
        """Test get_daily_quality_averages with agent_role filter."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
        )
        db.record_score(
            project_id="project-2",
            agent_role="editor",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=6.0,
        )

        data = db.get_daily_quality_averages(agent_role="writer")

        assert len(data) == 1
        assert data[0]["avg_value"] == 8.0
        assert data[0]["sample_count"] == 1

    def test_get_daily_quality_averages_invalid_metric(self, db: ModeDatabase) -> None:
        """Test get_daily_quality_averages defaults to prose_quality for invalid metric."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.5,
        )

        # Invalid metric should default to prose_quality
        data = db.get_daily_quality_averages(metric="invalid_metric")

        assert len(data) == 1
        assert data[0]["avg_value"] == 8.5

    def test_get_daily_quality_averages_different_metrics(self, db: ModeDatabase) -> None:
        """Test get_daily_quality_averages with different valid metrics."""
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
            instruction_following=7.5,
            consistency_score=9.0,
            tokens_per_second=50.0,
        )

        # Test instruction_following
        data = db.get_daily_quality_averages(metric="instruction_following")
        assert len(data) == 1
        assert data[0]["avg_value"] == 7.5

        # Test consistency_score
        data = db.get_daily_quality_averages(metric="consistency_score")
        assert len(data) == 1
        assert data[0]["avg_value"] == 9.0

        # Test tokens_per_second
        data = db.get_daily_quality_averages(metric="tokens_per_second")
        assert len(data) == 1
        assert data[0]["avg_value"] == 50.0

    def test_get_daily_quality_averages_empty_database(self, db: ModeDatabase) -> None:
        """Test get_daily_quality_averages with no data."""
        data = db.get_daily_quality_averages()
        assert data == []

    def test_get_daily_quality_averages_respects_days_param(self, db: ModeDatabase) -> None:
        """Test get_daily_quality_averages filters by days parameter."""
        # Record a score (will be within the default 30 days)
        db.record_score(
            project_id="project-1",
            agent_role="writer",
            model_id="model-a",
            mode_name="balanced",
            prose_quality=8.0,
        )

        # With days=30, should include today's record
        data = db.get_daily_quality_averages(days=30)
        assert len(data) == 1

        # With days=0, should still include today's record (DATE('now', '-0 days') = today)
        data = db.get_daily_quality_averages(days=0)
        assert len(data) == 1

    # === Prompt Metrics Tests ===

    def test_record_prompt_metrics(self, db: ModeDatabase) -> None:
        """Test recording prompt metrics."""
        record_id = db.record_prompt_metrics(
            prompt_hash="abc123",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=500,
            generation_time_seconds=10.5,
            success=True,
            project_id="test-project",
        )

        assert record_id > 0

    def test_record_prompt_metrics_with_error(self, db: ModeDatabase) -> None:
        """Test recording failed prompt metrics with error message."""
        record_id = db.record_prompt_metrics(
            prompt_hash="abc123",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=0,
            generation_time_seconds=5.0,
            success=False,
            error_message="Generation failed due to timeout",
        )

        assert record_id > 0

    def test_record_prompt_metrics_raises_on_db_error(self, db: ModeDatabase) -> None:
        """Test record_prompt_metrics raises sqlite3.Error on database failure."""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database write failed")
            with pytest.raises(sqlite3.Error, match="Database write failed"):
                db.record_prompt_metrics(
                    prompt_hash="abc123",
                    agent_role="writer",
                    task="test",
                    template_version="1.0",
                    model_id="model",
                    tokens_generated=100,
                    generation_time_seconds=1.0,
                    success=True,
                )

    def test_get_prompt_analytics(self, db: ModeDatabase) -> None:
        """Test getting prompt analytics."""
        # Record some metrics
        for i in range(5):
            db.record_prompt_metrics(
                prompt_hash="hash1",
                agent_role="writer",
                task="write_chapter",
                template_version="1.0",
                model_id="test-model",
                tokens_generated=500 + i * 10,
                generation_time_seconds=10.0 + i,
                success=True,
            )
        db.record_prompt_metrics(
            prompt_hash="hash1",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=0,
            generation_time_seconds=2.0,
            success=False,
            error_message="Failed",
        )

        # Get analytics
        analytics = db.get_prompt_analytics()
        assert len(analytics) == 1
        assert analytics[0]["agent_role"] == "writer"
        assert analytics[0]["task"] == "write_chapter"
        assert analytics[0]["total_calls"] == 6
        assert analytics[0]["successful_calls"] == 5

    def test_get_prompt_analytics_with_filters(self, db: ModeDatabase) -> None:
        """Test get_prompt_analytics with agent_role and task filters."""
        # Record metrics for different agents/tasks
        db.record_prompt_metrics(
            prompt_hash="h1",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="m1",
            tokens_generated=100,
            generation_time_seconds=1.0,
            success=True,
        )
        db.record_prompt_metrics(
            prompt_hash="h2",
            agent_role="editor",
            task="edit_chapter",
            template_version="1.0",
            model_id="m1",
            tokens_generated=200,
            generation_time_seconds=2.0,
            success=True,
        )

        # Filter by agent_role
        analytics = db.get_prompt_analytics(agent_role="writer")
        assert len(analytics) == 1
        assert analytics[0]["agent_role"] == "writer"

        # Filter by task
        analytics = db.get_prompt_analytics(task="edit_chapter")
        assert len(analytics) == 1
        assert analytics[0]["task"] == "edit_chapter"

    def test_get_prompt_metrics_summary(self, db: ModeDatabase) -> None:
        """Test getting prompt metrics summary."""
        # Record some metrics
        for i in range(3):
            db.record_prompt_metrics(
                prompt_hash=f"hash{i}",
                agent_role="writer" if i < 2 else "editor",
                task="write_chapter" if i < 2 else "edit_chapter",
                template_version="1.0",
                model_id="test-model",
                tokens_generated=100 * (i + 1),
                generation_time_seconds=5.0 + i,
                success=True,
            )

        summary = db.get_prompt_metrics_summary()

        assert summary["total_generations"] == 3
        assert summary["successful_generations"] == 3
        assert summary["success_rate"] == 100.0
        assert summary["total_tokens"] == 600  # 100 + 200 + 300
        assert "by_agent" in summary
        assert len(summary["by_agent"]) == 2  # writer and editor

    def test_get_prompt_metrics_summary_empty(self, db: ModeDatabase) -> None:
        """Test get_prompt_metrics_summary with no data."""
        summary = db.get_prompt_metrics_summary()

        assert summary["total_generations"] == 0
        assert summary["successful_generations"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["by_agent"] == []

    def test_get_prompt_metrics_by_hash(self, db: ModeDatabase) -> None:
        """Test getting metrics for a specific template hash."""
        target_hash = "target_hash_123"
        other_hash = "other_hash_456"

        # Record metrics for target hash
        for i in range(3):
            db.record_prompt_metrics(
                prompt_hash=target_hash,
                agent_role="writer",
                task="write_chapter",
                template_version="1.0",
                model_id="test-model",
                tokens_generated=100 + i,
                generation_time_seconds=1.0 + i,
                success=True,
            )

        # Record metrics for other hash
        db.record_prompt_metrics(
            prompt_hash=other_hash,
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=500,
            generation_time_seconds=5.0,
            success=True,
        )

        # Get metrics for target hash only
        metrics = db.get_prompt_metrics_by_hash(target_hash)
        assert len(metrics) == 3
        assert all(m["prompt_hash"] == target_hash for m in metrics)

    def test_get_prompt_metrics_by_hash_respects_limit(self, db: ModeDatabase) -> None:
        """Test get_prompt_metrics_by_hash respects limit parameter."""
        for _i in range(10):
            db.record_prompt_metrics(
                prompt_hash="test_hash",
                agent_role="writer",
                task="write_chapter",
                template_version="1.0",
                model_id="test-model",
                tokens_generated=100,
                generation_time_seconds=1.0,
                success=True,
            )

        metrics = db.get_prompt_metrics_by_hash("test_hash", limit=5)
        assert len(metrics) == 5

    def test_get_prompt_error_summary(self, db: ModeDatabase) -> None:
        """Test getting prompt error summary."""
        # Record some successful metrics
        db.record_prompt_metrics(
            prompt_hash="h1",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=500,
            generation_time_seconds=10.0,
            success=True,
        )

        # Record some failed metrics
        for i in range(3):
            db.record_prompt_metrics(
                prompt_hash="h1",
                agent_role="writer",
                task="write_chapter",
                template_version="1.0",
                model_id="test-model",
                tokens_generated=0,
                generation_time_seconds=1.0,
                success=False,
                error_message=f"Error {i}",
            )

        errors = db.get_prompt_error_summary()
        assert len(errors) == 1
        assert errors[0]["error_count"] == 3
        assert errors[0]["agent_role"] == "writer"
        assert errors[0]["task"] == "write_chapter"

    def test_get_prompt_error_summary_empty(self, db: ModeDatabase) -> None:
        """Test get_prompt_error_summary with no errors."""
        # Record only successful metrics
        db.record_prompt_metrics(
            prompt_hash="h1",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=500,
            generation_time_seconds=10.0,
            success=True,
        )

        errors = db.get_prompt_error_summary()
        assert len(errors) == 0

    def test_record_prompt_metrics_disabled(self, db: ModeDatabase, monkeypatch) -> None:
        """Test that prompt metrics are not recorded when disabled in settings."""
        from src.settings import Settings

        # Create settings with prompt_metrics_enabled=False
        mock_settings = Settings()
        mock_settings.prompt_metrics_enabled = False
        monkeypatch.setattr(Settings, "load", lambda: mock_settings)

        # Attempt to record metrics - should return 0 and not insert
        record_id = db.record_prompt_metrics(
            prompt_hash="abc123",
            agent_role="writer",
            task="write_chapter",
            template_version="1.0",
            model_id="test-model",
            tokens_generated=500,
            generation_time_seconds=10.5,
            success=True,
        )

        assert record_id == 0

        # Verify no record was inserted
        analytics = db.get_prompt_analytics()
        assert len(analytics) == 0

    def test_get_prompt_analytics_negative_days_raises(self, db: ModeDatabase) -> None:
        """Test that get_prompt_analytics raises ValueError for negative days."""
        with pytest.raises(ValueError, match="days must be a non-negative integer"):
            db.get_prompt_analytics(days=-1)

    def test_get_prompt_error_summary_negative_days_raises(self, db: ModeDatabase) -> None:
        """Test that get_prompt_error_summary raises ValueError for negative days."""
        with pytest.raises(ValueError, match="days must be a non-negative integer"):
            db.get_prompt_error_summary(days=-1)


class TestModeDatabaseMigrations:
    """Tests for database migrations."""

    def test_migration_adds_size_preference_column(self, tmp_path: Path) -> None:
        """Test that migration adds size_preference column to existing databases."""
        db_path = tmp_path / "old_schema.db"

        # Create a database with the old schema (without size_preference column)
        # Using the correct column names: agent_models_json, agent_temperatures_json
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE custom_modes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    agent_models_json TEXT NOT NULL,
                    agent_temperatures_json TEXT NOT NULL,
                    vram_strategy TEXT NOT NULL DEFAULT 'adaptive',
                    is_experimental INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Insert a mode without size_preference
            conn.execute(
                """
                INSERT INTO custom_modes
                    (id, name, agent_models_json, agent_temperatures_json, vram_strategy)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("test-mode", "Test Mode", '{"writer": "model-a"}', '{"writer": 0.9}', "adaptive"),
            )
            conn.commit()
        finally:
            conn.close()

        # Now open the database with ModeDatabase - should trigger migration
        db = ModeDatabase(db_path)

        # Check that the column was added
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("PRAGMA table_info(custom_modes)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "size_preference" in columns
        finally:
            conn.close()

        # Check that existing mode has default value
        mode = db.get_custom_mode("test-mode")
        assert mode is not None
        assert mode.get("size_preference") == "medium"
