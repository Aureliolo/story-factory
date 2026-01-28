"""Analytics page - CSV export helpers."""

import csv
import io
import logging
from datetime import datetime

from nicegui import ui

logger = logging.getLogger(__name__)


def export_csv(page) -> None:
    """Export score data to CSV.

    Args:
        page: The AnalyticsPage instance.
    """
    try:
        scores = page._db.get_all_scores(
            agent_role=page._filter_agent_role, genre=page._filter_genre
        )
        logger.info(f"Exporting {len(scores)} scores to CSV")

        if not scores:
            logger.warning("No data to export")
            ui.notify("No data to export", type="warning")
            return

        # Build CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "timestamp",
                "project_id",
                "chapter_id",
                "agent_role",
                "model_id",
                "mode_name",
                "genre",
                "prose_quality",
                "instruction_following",
                "consistency_score",
                "tokens_generated",
                "time_seconds",
                "tokens_per_second",
                "was_regenerated",
                "edit_distance",
                "user_rating",
            ]
        )

        # Data rows
        for score in scores:
            writer.writerow(
                [
                    score.timestamp.isoformat() if score.timestamp else "",
                    score.project_id,
                    score.chapter_id or "",
                    score.agent_role,
                    score.model_id,
                    score.mode_name,
                    score.genre or "",
                    score.quality.prose_quality if score.quality.prose_quality else "",
                    score.quality.instruction_following
                    if score.quality.instruction_following
                    else "",
                    score.quality.consistency_score if score.quality.consistency_score else "",
                    score.performance.tokens_generated
                    if score.performance.tokens_generated
                    else "",
                    score.performance.time_seconds if score.performance.time_seconds else "",
                    score.performance.tokens_per_second
                    if score.performance.tokens_per_second
                    else "",
                    score.signals.was_regenerated,
                    score.signals.edit_distance if score.signals.edit_distance else "",
                    score.signals.user_rating if score.signals.user_rating else "",
                ]
            )

        csv_content = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"story_factory_analytics_{timestamp}.csv"

        # Trigger download
        ui.download(csv_content.encode(), filename)
        logger.info(f"Successfully exported {len(scores)} records to {filename}")
        ui.notify(f"Exported {len(scores)} records to {filename}", type="positive")

    except Exception as e:
        logger.error(f"Failed to export CSV: {e}", exc_info=True)
        ui.notify(f"Export failed: {e!s}", type="negative")
