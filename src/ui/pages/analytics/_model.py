"""Analytics page - model performance section helpers."""

import logging

from nicegui import ui

from src.utils import extract_model_name

logger = logging.getLogger(__name__)


def build_model_section(page) -> None:
    """Build the model performance table.

    Args:
        page: The AnalyticsPage instance.
    """
    if page._model_section is None:
        return

    page._model_section.clear()

    try:
        # Get model performance summaries
        summaries = page._db.get_model_summaries(
            agent_role=page._filter_agent_role, genre=page._filter_genre
        )
        logger.debug(f"Loaded {len(summaries)} model performance summaries")
    except Exception as e:
        logger.error(f"Failed to load model performance data: {e}", exc_info=True)
        with page._model_section:
            with ui.card().classes("w-full"):
                ui.label("Failed to load model performance data. Check logs for details.").classes(
                    "text-red-500 p-4"
                )
        return

    with page._model_section:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center mb-4"):
                ui.icon("leaderboard").classes("text-blue-500")
                ui.label("Model Performance").classes("text-lg font-semibold")
                ui.space()
                ui.label(f"{len(summaries)} models tracked").classes("text-sm text-gray-500")

            if not summaries:
                ui.label(
                    "No performance data yet. Generate some stories to collect metrics!"
                ).classes("text-gray-500 dark:text-gray-400 py-8 text-center")
                return

            # Performance table
            columns = [
                {"name": "model", "label": "Model", "field": "model", "sortable": True},
                {"name": "role", "label": "Role", "field": "role", "sortable": True},
                {"name": "prose", "label": "Prose", "field": "prose", "sortable": True},
                {
                    "name": "instruction",
                    "label": "Instruction",
                    "field": "instruction",
                    "sortable": True,
                },
                {
                    "name": "consistency",
                    "label": "Consistency",
                    "field": "consistency",
                    "sortable": True,
                },
                {"name": "speed", "label": "Speed (t/s)", "field": "speed", "sortable": True},
                {"name": "samples", "label": "Samples", "field": "samples", "sortable": True},
                {"name": "overall", "label": "Overall", "field": "overall", "sortable": True},
            ]

            rows = []
            for s in summaries:
                model_name = extract_model_name(s.model_id)
                # Calculate overall score as average of quality metrics
                quality_scores = [
                    score
                    for score in [
                        s.avg_prose_quality,
                        s.avg_instruction_following,
                        s.avg_consistency,
                    ]
                    if score is not None
                ]
                overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0

                rows.append(
                    {
                        "model": model_name,
                        "role": s.agent_role.title(),
                        "prose": f"{s.avg_prose_quality:.1f}" if s.avg_prose_quality else "-",
                        "instruction": f"{s.avg_instruction_following:.1f}"
                        if s.avg_instruction_following
                        else "-",
                        "consistency": f"{s.avg_consistency:.1f}" if s.avg_consistency else "-",
                        "speed": f"{s.avg_tokens_per_second:.1f}"
                        if s.avg_tokens_per_second
                        else "-",
                        "samples": s.sample_count,
                        "overall": f"{overall_score:.1f}" if overall_score > 0 else "-",
                    }
                )

            # Wrap table in scrollable container for mobile
            with ui.element("div").classes("w-full overflow-x-auto"):
                ui.table(columns=columns, rows=rows, row_key="model").classes("w-full")

            # Add model comparison insights
            if len(summaries) >= 2:
                _build_model_insights(summaries)


def _build_model_insights(summaries: list) -> None:
    """Build model comparison insights.

    Args:
        summaries: List of ModelPerformanceSummary objects.
    """
    ui.label("Model Insights").classes("font-semibold mt-4 mb-2")

    # Find best models for each metric
    best_prose = max(
        (s for s in summaries if s.avg_prose_quality),
        key=lambda s: s.avg_prose_quality or 0,
        default=None,
    )
    best_speed = max(
        (s for s in summaries if s.avg_tokens_per_second),
        key=lambda s: s.avg_tokens_per_second or 0,
        default=None,
    )
    best_consistency = max(
        (s for s in summaries if s.avg_consistency),
        key=lambda s: s.avg_consistency or 0,
        default=None,
    )

    insights = []
    if best_prose:
        insights.append(
            f"\U0001f3c6 Best Prose Quality: {extract_model_name(best_prose.model_id)} "
            f"({best_prose.avg_prose_quality:.1f}/10) for {best_prose.agent_role}"
        )
    if best_speed:
        insights.append(
            f"\u26a1 Fastest: {extract_model_name(best_speed.model_id)} "
            f"({best_speed.avg_tokens_per_second:.1f} t/s) for {best_speed.agent_role}"
        )
    if best_consistency:
        insights.append(
            f"\U0001f3af Most Consistent: {extract_model_name(best_consistency.model_id)} "
            f"({best_consistency.avg_consistency:.1f}/10) for {best_consistency.agent_role}"
        )

    with ui.card().classes("w-full mt-2").props("flat bordered"):
        for insight in insights:
            ui.label(insight).classes("text-sm mb-1")
