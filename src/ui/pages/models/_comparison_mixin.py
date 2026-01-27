"""Models page - Model comparison mixin."""

import logging

from nicegui import ui

from src.ui.pages.models._page import ModelsPageBase

logger = logging.getLogger(__name__)


class ComparisonMixin(ModelsPageBase):
    """Mixin providing model comparison functionality for ModelsPage."""

    def _build_comparison_section(self) -> None:
        """Build the model comparison section."""
        with ui.card().classes("w-full"):
            ui.label("Model Comparison").classes("text-lg font-semibold mb-4")
            ui.label("Compare models on the same prompt to see output quality and speed").classes(
                "text-sm text-gray-500 dark:text-gray-400 mb-4"
            )

            installed = self.services.model.list_installed()

            if len(installed) < 2:
                ui.label("Install at least 2 models to compare").classes(
                    "text-gray-500 dark:text-gray-400"
                )
                return

            # Model selection
            with ui.row().classes("w-full gap-4 mb-4"):
                self._compare_models_select = ui.select(
                    label="Select models to compare",
                    options=installed,
                    multiple=True,
                ).classes("flex-grow")

            # Prompt input
            self._compare_prompt = ui.textarea(
                label="Test prompt",
                value="Write a short paragraph describing a sunset over the ocean.",
            ).classes("w-full")

            ui.button(
                "Run Comparison",
                on_click=self._run_comparison,
                icon="compare",
            ).props("color=primary")

            # Results
            self._comparison_result = ui.column().classes("w-full mt-4")

    async def _run_comparison(self) -> None:
        """Run model comparison."""
        if not self._comparison_result:
            return

        models = self._compare_models_select.value if self._compare_models_select else []
        prompt = self._compare_prompt.value if self._compare_prompt else ""

        if len(models) < 2:
            ui.notify("Select at least 2 models", type="warning")
            return

        if not prompt:
            ui.notify("Enter a prompt", type="warning")
            return

        logger.info(f"Running model comparison: {models}")

        self._comparison_result.clear()

        with self._comparison_result:
            ui.label("Running comparison...").classes("text-gray-500 dark:text-gray-400")
            ui.spinner()

        # Run comparison
        results = self.services.model.compare_models(models, prompt)
        logger.info(f"Model comparison complete: {len(results)} results")

        self._comparison_result.clear()

        with self._comparison_result:
            for result in results:
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label(result["model_id"]).classes("font-semibold")
                        if result["success"]:
                            ui.badge(f"{result['time_seconds']}s").props("color=grey-7")
                        else:
                            ui.badge("Failed").props("color=negative")

                    if result["success"]:
                        ui.markdown(result["response"]).classes("text-sm")
                    else:
                        ui.label(result.get("error", "Unknown error")).classes("text-red-500")
