"""Analytics mixin for ModeDatabase.

Handles analytics queries for quality, speed, and genre data.
"""

import logging
import sqlite3
from typing import Any

from src.memory.mode_database._base import ModeDatabaseBase

logger = logging.getLogger(__name__)


class AnalyticsMixin(ModeDatabaseBase):
    """Mixin providing analytics query operations."""

    def get_quality_vs_speed_data(
        self,
        agent_role: str | None = None,
        min_samples: int = 3,
    ) -> list[dict[str, Any]]:
        """Get data for quality vs speed scatter plot."""
        query = """
            SELECT
                model_id,
                agent_role,
                avg_prose_quality,
                avg_tokens_per_second,
                sample_count
            FROM model_performance
            WHERE sample_count >= ?
        """
        params: list[Any] = [min_samples]

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_genre_breakdown(self, model_id: str) -> list[dict[str, Any]]:
        """Get performance breakdown by genre for a model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    genre,
                    AVG(prose_quality) as avg_quality,
                    AVG(tokens_per_second) as avg_speed,
                    COUNT(*) as sample_count
                FROM generation_scores
                WHERE model_id = ? AND genre IS NOT NULL
                GROUP BY genre
                ORDER BY avg_quality DESC
                """,
                (model_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_unique_genres(self) -> list[str]:
        """Get list of unique genres from scores."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT genre FROM generation_scores WHERE genre IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_average_score(
        self,
        metric: str,
        agent_role: str | None = None,
        genre: str | None = None,
    ) -> float | None:
        """Get average of a specific metric.

        Args:
            metric: Column name (prose_quality, instruction_following, etc.)
            agent_role: Optional filter.
            genre: Optional filter.

        Returns:
            Average value or None if no data.
        """
        valid_metrics = {
            "prose_quality",
            "instruction_following",
            "consistency_score",
            "tokens_per_second",
        }
        if metric not in valid_metrics:
            return None

        query = f"SELECT AVG({metric}) FROM generation_scores WHERE 1=1"
        params: list[Any] = []

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)
        if genre:
            query += " AND genre = ?"
            params.append(genre)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return float(result) if result is not None else None

    def get_quality_time_series(
        self,
        metric: str = "prose_quality",
        agent_role: str | None = None,
        genre: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get time series data for quality metrics.

        Args:
            metric: Metric to track (prose_quality, instruction_following, consistency_score).
            agent_role: Optional filter by agent role.
            genre: Optional filter by genre.
            limit: Maximum number of data points.

        Returns:
            List of time series data points with timestamp and value.
        """
        valid_metrics = {
            "prose_quality",
            "instruction_following",
            "consistency_score",
            "tokens_per_second",
        }
        if metric not in valid_metrics:
            logger.warning(f"Invalid metric {metric}, defaulting to prose_quality")
            metric = "prose_quality"

        query = f"SELECT timestamp, {metric} FROM generation_scores WHERE {metric} IS NOT NULL"
        params: list = []

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)
        if genre:
            query += " AND genre = ?"
            params.append(genre)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [{"timestamp": row[0], "value": row[1]} for row in cursor.fetchall()]

    def get_daily_quality_averages(
        self,
        metric: str = "prose_quality",
        days: int = 30,
        agent_role: str | None = None,
    ) -> list[dict]:
        """Get daily average quality metrics.

        Args:
            metric: Metric to track.
            days: Number of days to include.
            agent_role: Optional filter by agent role.

        Returns:
            List of daily averages with date and average value.
        """
        valid_metrics = {
            "prose_quality",
            "instruction_following",
            "consistency_score",
            "tokens_per_second",
        }
        if metric not in valid_metrics:
            logger.warning(f"Invalid metric {metric}, defaulting to prose_quality")
            metric = "prose_quality"

        query = f"""
            SELECT
                DATE(timestamp) as date,
                AVG({metric}) as avg_value,
                COUNT(*) as sample_count
            FROM generation_scores
            WHERE {metric} IS NOT NULL
            AND DATE(timestamp) >= DATE('now', '-{days} days')
        """
        params: list = []

        if agent_role:
            query += " AND agent_role = ?"
            params.append(agent_role)

        query += " GROUP BY DATE(timestamp) ORDER BY date DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [
                {"date": row[0], "avg_value": row[1], "sample_count": row[2]}
                for row in cursor.fetchall()
            ]
