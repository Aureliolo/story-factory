"""Tests for cost tracking models and functionality."""

from datetime import datetime

import pytest

from src.memory.cost_models import (
    CostSummary,
    EntityTypeCostBreakdown,
    GenerationMetrics,
    GenerationRunCosts,
    ModelCostBreakdown,
)
from src.memory.mode_database import ModeDatabase


class TestGenerationMetrics:
    """Tests for GenerationMetrics model."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        metrics = GenerationMetrics()
        assert metrics.prompt_tokens is None
        assert metrics.completion_tokens is None
        assert metrics.total_tokens == 0
        assert metrics.time_seconds == 0.0

    def test_init_with_values(self):
        """Test initialization with specific values."""
        metrics = GenerationMetrics(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            time_seconds=2.5,
            model_id="test-model",
            agent_role="writer",
        )
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 200
        assert metrics.total_tokens == 300
        assert metrics.time_seconds == 2.5
        assert metrics.model_id == "test-model"
        assert metrics.agent_role == "writer"

    def test_tokens_per_second(self):
        """Test tokens_per_second calculation."""
        metrics = GenerationMetrics(
            completion_tokens=100,
            time_seconds=2.0,
        )
        assert metrics.tokens_per_second == 50.0

    def test_tokens_per_second_zero_time(self):
        """Test tokens_per_second returns None for zero time."""
        metrics = GenerationMetrics(
            completion_tokens=100,
            time_seconds=0.0,
        )
        assert metrics.tokens_per_second is None

    def test_tokens_per_second_no_tokens(self):
        """Test tokens_per_second returns None for no tokens."""
        metrics = GenerationMetrics(
            completion_tokens=None,
            time_seconds=2.0,
        )
        assert metrics.tokens_per_second is None


class TestEntityTypeCostBreakdown:
    """Tests for EntityTypeCostBreakdown model."""

    def test_avg_tokens_per_entity(self):
        """Test average tokens per entity calculation."""
        breakdown = EntityTypeCostBreakdown(
            entity_type="character",
            count=10,
            total_tokens=1000,
        )
        assert breakdown.avg_tokens_per_entity == 100.0

    def test_avg_tokens_per_entity_zero_count(self):
        """Test average when count is zero."""
        breakdown = EntityTypeCostBreakdown(
            entity_type="character",
            count=0,
            total_tokens=0,
        )
        assert breakdown.avg_tokens_per_entity == 0.0

    def test_avg_time_per_entity(self):
        """Test average time per entity calculation."""
        breakdown = EntityTypeCostBreakdown(
            entity_type="character",
            count=5,
            total_time_seconds=50.0,
        )
        assert breakdown.avg_time_per_entity == 10.0

    def test_avg_time_per_entity_zero_count(self):
        """Test average time when count is zero."""
        breakdown = EntityTypeCostBreakdown(
            entity_type="character",
            count=0,
            total_time_seconds=0.0,
        )
        assert breakdown.avg_time_per_entity == 0.0


class TestModelCostBreakdown:
    """Tests for ModelCostBreakdown model."""

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        breakdown = ModelCostBreakdown(
            model_id="test-model",
            total_tokens=1000,
            total_time_seconds=10.0,
        )
        assert breakdown.tokens_per_second == 100.0

    def test_tokens_per_second_zero_time(self):
        """Test tokens per second with zero time."""
        breakdown = ModelCostBreakdown(
            model_id="test-model",
            total_tokens=1000,
            total_time_seconds=0.0,
        )
        assert breakdown.tokens_per_second is None

    def test_avg_tokens_per_call(self):
        """Test average tokens per call calculation."""
        breakdown = ModelCostBreakdown(
            model_id="test-model",
            total_tokens=1000,
            call_count=10,
        )
        assert breakdown.avg_tokens_per_call == 100.0

    def test_avg_tokens_per_call_zero_count(self):
        """Test average tokens per call when count is zero."""
        breakdown = ModelCostBreakdown(
            model_id="test-model",
            total_tokens=0,
            call_count=0,
        )
        assert breakdown.avg_tokens_per_call == 0.0


class TestGenerationRunCosts:
    """Tests for GenerationRunCosts model."""

    def test_duration_seconds(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 5, 0)  # 5 minutes later
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="story_generation",
            started_at=start,
            completed_at=end,
        )
        assert costs.duration_seconds == 300.0  # 5 minutes

    def test_duration_seconds_not_completed(self):
        """Test duration when not completed."""
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="story_generation",
            completed_at=None,
        )
        assert costs.duration_seconds is None

    def test_efficiency_ratio(self):
        """Test efficiency ratio calculation."""
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="world_build",
            total_iterations=10,
            wasted_iterations=2,
        )
        assert costs.efficiency_ratio == 0.8  # 8/10

    def test_efficiency_ratio_perfect(self):
        """Test efficiency ratio with no waste."""
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="world_build",
            total_iterations=10,
            wasted_iterations=0,
        )
        assert costs.efficiency_ratio == 1.0

    def test_efficiency_ratio_no_iterations(self):
        """Test efficiency ratio with no iterations."""
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="world_build",
            total_iterations=0,
        )
        assert costs.efficiency_ratio == 1.0

    def test_avg_tokens_per_call(self):
        """Test average tokens per call calculation."""
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="story_generation",
            total_tokens=1000,
            total_calls=10,
        )
        assert costs.avg_tokens_per_call == 100.0

    def test_avg_tokens_per_call_zero_calls(self):
        """Test average tokens per call when no calls."""
        costs = GenerationRunCosts(
            run_id="test-run",
            project_id="test-project",
            run_type="story_generation",
            total_tokens=0,
            total_calls=0,
        )
        assert costs.avg_tokens_per_call == 0.0


class TestCostSummary:
    """Tests for CostSummary model."""

    def test_avg_tokens_per_run(self):
        """Test average tokens per run calculation."""
        summary = CostSummary(
            total_runs=10,
            total_tokens=10000,
        )
        assert summary.avg_tokens_per_run == 1000.0

    def test_avg_tokens_per_run_zero_runs(self):
        """Test average tokens per run when no runs."""
        summary = CostSummary(
            total_runs=0,
            total_tokens=0,
        )
        assert summary.avg_tokens_per_run == 0.0

    def test_overall_efficiency(self):
        """Test overall efficiency calculation."""
        summary = CostSummary(
            total_iterations=100,
            total_wasted_iterations=20,
        )
        assert summary.overall_efficiency == 0.8

    def test_overall_efficiency_zero_iterations(self):
        """Test overall efficiency when no iterations."""
        summary = CostSummary(
            total_iterations=0,
            total_wasted_iterations=0,
        )
        assert summary.overall_efficiency == 1.0

    def test_format_total_time_seconds(self):
        """Test time formatting for seconds."""
        summary = CostSummary(total_time_seconds=45)
        assert summary.format_total_time() == "45s"

    def test_format_total_time_minutes(self):
        """Test time formatting for minutes."""
        summary = CostSummary(total_time_seconds=332)  # 5m 32s
        assert summary.format_total_time() == "5m 32s"

    def test_format_total_time_hours(self):
        """Test time formatting for hours."""
        summary = CostSummary(total_time_seconds=8100)  # 2h 15m
        assert summary.format_total_time() == "2h 15m"

    def test_format_total_tokens_small(self):
        """Test token formatting for small numbers."""
        summary = CostSummary(total_tokens=500)
        assert summary.format_total_tokens() == "500"

    def test_format_total_tokens_thousands(self):
        """Test token formatting for thousands."""
        summary = CostSummary(total_tokens=1200)
        assert summary.format_total_tokens() == "1.2K"

    def test_format_total_tokens_millions(self):
        """Test token formatting for millions."""
        summary = CostSummary(total_tokens=3500000)
        assert summary.format_total_tokens() == "3.5M"


class TestModeDatabaseCostTracking:
    """Tests for cost tracking in ModeDatabase."""

    @pytest.fixture
    def db(self, tmp_path):
        """
        Create a ModeDatabase instance pointing at a temporary test database file.

        Parameters:
            tmp_path (pathlib.Path): Temporary directory path provided by pytest.

        Returns:
            ModeDatabase: A ModeDatabase configured to use a test SQLite file under the provided temporary path.
        """
        db_path = tmp_path / "test_mode.db"
        return ModeDatabase(db_path)

    def test_start_generation_run(self, db):
        """Test starting a generation run."""
        run_id = db.start_generation_run(
            run_id="test-run-1",
            project_id="test-project",
            run_type="story_generation",
        )
        assert run_id > 0

    def test_get_generation_run(self, db):
        """Test getting a generation run."""
        db.start_generation_run(
            run_id="test-run-1",
            project_id="test-project",
            run_type="world_build",
        )

        run = db.get_generation_run("test-run-1")

        assert run is not None
        assert run["run_id"] == "test-run-1"
        assert run["project_id"] == "test-project"
        assert run["run_type"] == "world_build"

    def test_get_generation_run_not_found(self, db):
        """Test getting a non-existent run."""
        run = db.get_generation_run("nonexistent")
        assert run is None

    def test_update_generation_run(self, db):
        """Test updating a generation run."""
        db.start_generation_run(
            run_id="test-run-1",
            project_id="test-project",
            run_type="story_generation",
        )

        db.update_generation_run(
            run_id="test-run-1",
            total_tokens=1000,
            total_time_seconds=30.0,
            total_calls=5,
        )

        run = db.get_generation_run("test-run-1")
        assert run["total_tokens"] == 1000
        assert run["total_time_seconds"] == 30.0
        assert run["total_calls"] == 5

    def test_update_generation_run_with_breakdowns(self, db):
        """Test updating run with JSON breakdowns."""
        db.start_generation_run(
            run_id="test-run-1",
            project_id="test-project",
            run_type="world_build",
        )

        db.update_generation_run(
            run_id="test-run-1",
            by_entity_type={"character": {"count": 5, "tokens": 500}},
            by_model={"test-model": {"tokens": 500, "calls": 5}},
        )

        run = db.get_generation_run("test-run-1")
        assert "character" in run["by_entity_type"]
        assert "test-model" in run["by_model"]

    def test_complete_generation_run(self, db):
        """Test completing a generation run."""
        db.start_generation_run(
            run_id="test-run-1",
            project_id="test-project",
            run_type="story_generation",
        )

        db.complete_generation_run("test-run-1")

        run = db.get_generation_run("test-run-1")
        assert run["completed_at"] is not None

    def test_get_generation_runs(self, db):
        """Test getting multiple generation runs."""
        db.start_generation_run("run-1", "project-1", "story_generation")
        db.start_generation_run("run-2", "project-1", "world_build")
        db.start_generation_run("run-3", "project-2", "story_generation")

        # Get all runs
        runs = db.get_generation_runs()
        assert len(runs) == 3

        # Filter by project
        runs = db.get_generation_runs(project_id="project-1")
        assert len(runs) == 2

        # Filter by type
        runs = db.get_generation_runs(run_type="story_generation")
        assert len(runs) == 2

    def test_get_cost_summary(self, db):
        """Test getting cost summary."""
        db.start_generation_run("run-1", "project-1", "story_generation")
        db.update_generation_run("run-1", total_tokens=1000, total_time_seconds=30.0)
        db.complete_generation_run("run-1")

        db.start_generation_run("run-2", "project-1", "world_build")
        db.update_generation_run("run-2", total_tokens=500, total_time_seconds=15.0)
        db.complete_generation_run("run-2")

        summary = db.get_cost_summary(project_id="project-1")

        assert summary["total_runs"] == 2
        assert summary["total_tokens"] == 1500
        assert summary["total_time_seconds"] == 45.0

    def test_get_cost_summary_efficiency(self, db):
        """Test cost summary includes efficiency ratio."""
        db.start_generation_run("run-1", "project-1", "world_build")
        db.update_generation_run(
            "run-1",
            total_iterations=10,
            wasted_iterations=2,
        )
        db.complete_generation_run("run-1")

        summary = db.get_cost_summary()

        assert summary["total_iterations"] == 10
        assert summary["wasted_iterations"] == 2
        assert summary["efficiency_ratio"] == 0.8

    def test_get_cost_summary_empty(self, db):
        """Test cost summary for empty database."""
        summary = db.get_cost_summary()

        assert summary["total_runs"] == 0
        assert summary["total_tokens"] == 0
        assert summary["efficiency_ratio"] == 1.0

    def test_get_cost_summary_negative_days_raises(self, db):
        """Test that negative days raises ValidationError."""
        from src.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="days must be a non-negative integer"):
            db.get_cost_summary(days=-1)

    def test_update_generation_run_with_completed(self, db):
        """Test updating a run with completed=True sets timestamp."""
        db.start_generation_run("run-1", "project-1", "story_generation")
        db.update_generation_run("run-1", total_tokens=500, completed=True)

        run = db.get_generation_run("run-1")
        assert run["completed_at"] is not None
        assert run["total_tokens"] == 500

    def test_update_generation_run_no_updates(self, db):
        """Test update with no parameters does nothing."""
        db.start_generation_run("run-1", "project-1", "story_generation")
        # Call with no updates - should return early
        db.update_generation_run("run-1")

        run = db.get_generation_run("run-1")
        assert run["total_tokens"] == 0

    def test_start_generation_run_error(self, db, tmp_path):
        """Test error handling when start_generation_run fails."""
        import sqlite3
        from unittest.mock import patch

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")
            with pytest.raises(sqlite3.Error):
                db.start_generation_run("run-fail", "project-1", "story_generation")

    def test_update_generation_run_error(self, db):
        """Test error handling when update_generation_run fails."""
        import sqlite3
        from unittest.mock import patch

        db.start_generation_run("run-1", "project-1", "story_generation")

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")
            with pytest.raises(sqlite3.Error):
                db.update_generation_run("run-1", total_tokens=100)

    def test_complete_generation_run_error(self, db):
        """Test error handling when complete_generation_run fails."""
        import sqlite3
        from unittest.mock import patch

        db.start_generation_run("run-1", "project-1", "story_generation")

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")
            with pytest.raises(sqlite3.Error):
                db.complete_generation_run("run-1")

    def test_get_generation_runs_with_json_fields(self, db):
        """Test get_generation_runs parses JSON fields correctly."""
        db.start_generation_run("run-1", "project-1", "world_build")
        db.update_generation_run(
            "run-1",
            by_entity_type={"character": {"count": 5}},
            by_model={"model-1": {"tokens": 100}},
        )

        runs = db.get_generation_runs()
        assert len(runs) == 1
        assert runs[0]["by_entity_type"] == {"character": {"count": 5}}
        assert runs[0]["by_model"] == {"model-1": {"tokens": 100}}

    def test_get_model_cost_breakdown(self, db):
        """Test get_model_cost_breakdown returns model statistics."""
        # This method queries generation_scores table
        # First, need to add some scores with all required NOT NULL fields
        with __import__("sqlite3").connect(db.db_path) as conn:
            conn.execute(
                """
                INSERT INTO generation_scores
                (project_id, agent_role, model_id, mode_name, tokens_generated, time_seconds, tokens_per_second, prose_quality, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                ("proj-1", "writer", "test-model", "creative", 1000, 10.0, 100.0, 0.8),
            )
            conn.commit()

        breakdown = db.get_model_cost_breakdown()
        assert len(breakdown) == 1
        assert breakdown[0]["model_id"] == "test-model"
        assert breakdown[0]["total_tokens"] == 1000
        assert breakdown[0]["avg_quality"] == 0.8

    def test_get_model_cost_breakdown_negative_days_raises(self, db):
        """Test that negative days raises ValidationError."""
        from src.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="days must be a non-negative integer"):
            db.get_model_cost_breakdown(days=-1)

    def test_get_model_cost_breakdown_with_project_filter(self, db):
        """Test get_model_cost_breakdown with project filter."""
        import sqlite3 as sql

        with sql.connect(db.db_path) as conn:
            conn.execute(
                """
                INSERT INTO generation_scores
                (project_id, agent_role, model_id, mode_name, tokens_generated, time_seconds, tokens_per_second, prose_quality, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                ("proj-1", "writer", "model-a", "creative", 500, 5.0, 100.0, 0.7),
            )
            conn.execute(
                """
                INSERT INTO generation_scores
                (project_id, agent_role, model_id, mode_name, tokens_generated, time_seconds, tokens_per_second, prose_quality, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                ("proj-2", "editor", "model-b", "precise", 300, 3.0, 100.0, 0.9),
            )
            conn.commit()

        breakdown = db.get_model_cost_breakdown(project_id="proj-1")
        assert len(breakdown) == 1
        assert breakdown[0]["model_id"] == "model-a"

    def test_get_entity_type_cost_breakdown(self, db):
        """Test get_entity_type_cost_breakdown returns entity statistics."""
        import sqlite3 as sql

        with sql.connect(db.db_path) as conn:
            conn.execute(
                """
                INSERT INTO world_entity_scores
                (project_id, entity_type, entity_name, model_id, generation_time_seconds, iterations_used, threshold_met, average_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                ("proj-1", "character", "Test Hero", "test-model", 15.0, 3, 1, 0.85),
            )
            conn.commit()

        breakdown = db.get_entity_type_cost_breakdown()
        assert len(breakdown) == 1
        assert breakdown[0]["entity_type"] == "character"
        assert breakdown[0]["count"] == 1
        assert breakdown[0]["avg_iterations"] == 3.0

    def test_get_entity_type_cost_breakdown_negative_days_raises(self, db):
        """Test that negative days raises ValidationError."""
        from src.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="days must be a non-negative integer"):
            db.get_entity_type_cost_breakdown(days=-1)

    def test_get_entity_type_cost_breakdown_with_project_filter(self, db):
        """Test get_entity_type_cost_breakdown with project filter."""
        import sqlite3 as sql

        with sql.connect(db.db_path) as conn:
            conn.execute(
                """
                INSERT INTO world_entity_scores
                (project_id, entity_type, entity_name, model_id, generation_time_seconds, iterations_used, threshold_met, average_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                ("proj-1", "character", "Hero One", "model-a", 10.0, 2, 1, 0.8),
            )
            conn.execute(
                """
                INSERT INTO world_entity_scores
                (project_id, entity_type, entity_name, model_id, generation_time_seconds, iterations_used, threshold_met, average_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                ("proj-2", "location", "Tavern", "model-b", 8.0, 1, 0, 0.6),
            )
            conn.commit()

        breakdown = db.get_entity_type_cost_breakdown(project_id="proj-1")
        assert len(breakdown) == 1
        assert breakdown[0]["entity_type"] == "character"
