"""Models page - Ollama model management."""

import asyncio
import logging

from nicegui import ui
from nicegui.elements.column import Column

from services import ServiceContainer
from settings import AVAILABLE_MODELS, ModelInfo
from ui.state import AppState
from ui.theme import get_quality_color

logger = logging.getLogger(__name__)


class ModelsPage:
    """Models management page.

    Features:
    - View installed models
    - Pull new models
    - Delete models
    - Model comparison
    - VRAM recommendations
    """

    # Allow 10% VRAM tolerance - if you have 23GB and model needs 24GB, it should work
    VRAM_TOLERANCE = 0.10

    def __init__(self, state: AppState, services: ServiceContainer):
        """Initialize models page.

        Args:
            state: Application state.
            services: Service container.
        """
        self.state = state
        self.services = services

        self._model_list: Column | None = None
        self._pull_progress: Column | None = None
        self._comparison_result: Column | None = None

        # Filter state
        self._filter_fits_vram = True
        self._filter_quality_min = 0
        self._filter_uncensored_only = False
        self._filter_search = ""

    def _model_fits_vram(self, vram_required: float, available_vram: float) -> bool:
        """Check if model fits in VRAM with tolerance.

        Args:
            vram_required: VRAM required by model.
            available_vram: Available VRAM.

        Returns:
            True if model fits (with tolerance), False otherwise.
        """
        # Allow models that are within 10% of available VRAM
        return vram_required <= available_vram * (1 + self.VRAM_TOLERANCE)

    def build(self) -> None:
        """Build the models page UI."""
        with ui.column().classes("w-full gap-6 p-4"):
            # Header with VRAM info
            self._build_header()

            # Installed models
            self._build_installed_section()

            # Available models
            self._build_available_section()

            # Model comparison
            self._build_comparison_section()

    def _build_header(self) -> None:
        """Build the header with VRAM info."""
        vram = self.services.model.get_vram()
        health = self.services.model.check_health()

        with ui.row().classes("w-full items-center"):
            ui.label("Models").classes("text-2xl font-bold")
            ui.space()

            # VRAM indicator
            with ui.row().classes("items-center gap-2 p-2 bg-gray-100 dark:bg-gray-700 rounded"):
                ui.icon("memory")
                ui.label(f"{vram} GB VRAM").classes("font-medium")

                if health.is_healthy:
                    ui.icon("check_circle", color="green")
                else:
                    ui.icon("error", color="red")

    def _build_installed_section(self) -> None:
        """Build the installed models section."""
        with ui.card().classes("w-full"):
            ui.label("Installed Models").classes("text-lg font-semibold mb-4")

            installed = self.services.model.list_installed()

            if not installed:
                ui.label("No models installed").classes("text-gray-500 dark:text-gray-400")
                return

            # Model table
            columns = [
                {"name": "name", "label": "Model", "field": "name", "align": "left"},
                {"name": "size", "label": "Size", "field": "size", "align": "center"},
                {"name": "quality", "label": "Quality", "field": "quality", "align": "center"},
                {"name": "actions", "label": "", "field": "actions", "align": "right"},
            ]

            rows = []
            for model_id in installed:
                info = self.services.model.get_model_info(model_id)
                rows.append(
                    {
                        "id": model_id,
                        "name": info["name"],
                        "size": f"{info['size_gb']} GB",
                        "quality": info["quality"],
                    }
                )

            ui.table(columns=columns, rows=rows).classes("w-full")

            # Note: Delete buttons would need custom cell rendering
            # For now, users can delete from the Available Models section

    def _build_available_section(self) -> None:
        """Build the available models section."""
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center mb-4"):
                ui.icon("download").classes("text-blue-500")
                ui.label("Available Models").classes("text-lg font-semibold")
                ui.icon("help_outline", size="xs").classes("text-gray-400 cursor-help").tooltip(
                    "Models from the Ollama registry you can download"
                )

            # Filter controls - responsive grid
            with ui.element("div").classes(
                "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
            ):
                # Search
                ui.input(
                    label="Search",
                    placeholder="Model name...",
                    on_change=lambda e: self._apply_filters(search=e.value),
                ).classes("w-full").props("dense clearable")

                # Quality filter
                ui.select(
                    label="Min Quality",
                    options={0: "Any", 5: "5+", 7: "7+", 8: "8+"},
                    value=0,
                    on_change=lambda e: self._apply_filters(quality=e.value),
                ).classes("w-full").props("dense")

                # Checkboxes row
                with ui.column().classes("gap-2"):
                    ui.checkbox(
                        "Fits my VRAM (~10% tolerance)",
                        value=True,
                        on_change=lambda e: self._apply_filters(fits_vram=e.value),
                    ).tooltip("Show models that fit your available VRAM with 10% tolerance")

                    ui.checkbox(
                        "Uncensored only",
                        value=False,
                        on_change=lambda e: self._apply_filters(uncensored=e.value),
                    ).tooltip("Only show models without content filters")

            # Model cards
            self._model_list = ui.column().classes("w-full gap-4")
            self._refresh_model_list()

            # Pull progress
            self._pull_progress = ui.column().classes("w-full mt-4")
            self._pull_progress.set_visibility(False)

    def _apply_filters(
        self,
        search: str | None = None,
        quality: int | None = None,
        fits_vram: bool | None = None,
        uncensored: bool | None = None,
    ) -> None:
        """Apply filters and refresh model list."""
        if search is not None:
            self._filter_search = search
        if quality is not None:
            self._filter_quality_min = quality
        if fits_vram is not None:
            self._filter_fits_vram = fits_vram
        if uncensored is not None:
            self._filter_uncensored_only = uncensored

        self._refresh_model_list()

    def _refresh_model_list(self) -> None:
        """Refresh the model list with current filters."""
        if not self._model_list:
            return

        self._model_list.clear()

        vram = self.services.model.get_vram()
        installed = set(self.services.model.list_installed())

        with self._model_list:
            shown_count = 0
            for model_id, info in AVAILABLE_MODELS.items():
                # Apply filters
                if self._filter_search and self._filter_search.lower() not in info["name"].lower():
                    continue
                if self._filter_quality_min and info["quality"] < self._filter_quality_min:
                    continue
                if self._filter_uncensored_only and not info["uncensored"]:
                    continue

                fits_vram = self._model_fits_vram(info["vram_required"], vram)
                if self._filter_fits_vram and not fits_vram:
                    continue

                is_installed = any(model_id in m for m in installed)
                self._build_model_card(model_id, info, fits_vram, is_installed, vram)
                shown_count += 1

            if shown_count == 0:
                ui.label("No models match your filters").classes(
                    "text-gray-500 dark:text-gray-400 text-center py-8"
                )

    def _build_model_card(
        self,
        model_id: str,
        info: ModelInfo,
        fits_vram: bool,
        is_installed: bool,
        available_vram: float = 0,
    ) -> None:
        """Build a model card.

        Args:
            model_id: Model identifier.
            info: Model info dictionary.
            fits_vram: Whether model fits available VRAM.
            is_installed: Whether model is installed.
            available_vram: Available VRAM for comparison display.
        """
        quality_color = get_quality_color(info["quality"])

        card_classes = "w-full"
        if not fits_vram:
            card_classes += " opacity-60"

        with ui.card().classes(card_classes):
            # Header row with name and badges
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.label(info["name"]).classes("text-lg font-semibold")
                if is_installed:
                    ui.badge("Installed").props("color=positive")
                if info["uncensored"]:
                    ui.badge("Uncensored").props("color=warning").tooltip(
                        "This model has no content filters"
                    )
                ui.space()
                # Actions in header
                if not is_installed:
                    if fits_vram:
                        ui.button(
                            "Download",
                            on_click=lambda m=model_id: self._pull_model(m),
                            icon="download",
                        ).props("color=primary dense")
                    else:
                        ui.label(f"Needs {info['vram_required']}GB").classes(
                            "text-sm text-orange-500 dark:text-orange-400"
                        ).tooltip(
                            f"Requires {info['vram_required']}GB VRAM, you have ~{available_vram}GB available"
                        )
                else:
                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="play_arrow", on_click=lambda m=model_id: self._test_model(m)
                        ).props("flat dense round").tooltip("Test model")
                        ui.button(
                            icon="delete", on_click=lambda m=model_id: self._delete_model(m)
                        ).props("flat dense round color=negative").tooltip("Delete model")

            # Description
            ui.label(info["description"]).classes("text-sm text-gray-600 dark:text-gray-400 mb-3")

            # Stats grid
            with ui.element("div").classes("grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm"):
                # Size
                with ui.column().classes("gap-0.5"):
                    ui.label("Disk Size").classes("text-xs text-gray-500 dark:text-gray-400")
                    ui.label(f"{info['size_gb']} GB").classes("font-medium")

                # VRAM
                with ui.column().classes("gap-0.5"):
                    ui.label("VRAM Needed").classes("text-xs text-gray-500 dark:text-gray-400")
                    vram_class = "font-medium"
                    if not fits_vram:
                        vram_class += " text-orange-500"
                    ui.label(f"{info['vram_required']} GB").classes(vram_class)

                # Quality bar
                with ui.column().classes("gap-0.5"):
                    ui.label("Quality").classes("text-xs text-gray-500 dark:text-gray-400")
                    with ui.row().classes("gap-0.5 items-center"):
                        for i in range(10):
                            if i < info["quality"]:
                                ui.element("div").style(
                                    f"width: 8px; height: 8px; background: {quality_color}; border-radius: 2px;"
                                )
                            else:
                                ui.element("div").classes("bg-gray-200 dark:bg-gray-600").style(
                                    "width: 8px; height: 8px; border-radius: 2px;"
                                )
                        ui.label(f"{info['quality']}/10").classes("text-xs ml-1")

                # Speed bar
                with ui.column().classes("gap-0.5"):
                    ui.label("Speed").classes("text-xs text-gray-500 dark:text-gray-400")
                    with ui.row().classes("gap-0.5 items-center"):
                        for i in range(10):
                            if i < info["speed"]:
                                ui.element("div").style(
                                    "width: 8px; height: 8px; background: #4CAF50; border-radius: 2px;"
                                )
                            else:
                                ui.element("div").classes("bg-gray-200 dark:bg-gray-600").style(
                                    "width: 8px; height: 8px; border-radius: 2px;"
                                )
                        ui.label(f"{info['speed']}/10").classes("text-xs ml-1")

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

    async def _pull_model(self, model_id: str) -> None:
        """Pull a model from Ollama."""
        if not self._pull_progress:
            return

        self._pull_progress.clear()
        self._pull_progress.set_visibility(True)

        with self._pull_progress:
            ui.label(f"Pulling {model_id}...").classes("font-medium")
            progress_bar = ui.linear_progress(value=0).classes("w-full")
            status_label = ui.label("Starting...").classes(
                "text-sm text-gray-500 dark:text-gray-400"
            )

        try:
            async for progress in self._async_pull(model_id):
                if "error" in progress:
                    status_label.text = progress["status"]
                    ui.notify(progress["status"], type="negative")
                    break

                status_label.text = progress.get("status", "")
                if progress.get("total", 0) > 0:
                    pct = progress.get("completed", 0) / progress["total"]
                    progress_bar.value = pct

            ui.notify(f"Model {model_id} pulled successfully!", type="positive")
            # Refresh would be needed

        except Exception as e:
            logger.exception(f"Failed to pull model {model_id}")
            ui.notify(f"Error: {e}", type="negative")

        finally:
            await asyncio.sleep(2)
            self._pull_progress.set_visibility(False)

    async def _async_pull(self, model_id: str):
        """Async wrapper for pull generator."""
        for progress in self.services.model.pull_model(model_id):
            yield progress
            await asyncio.sleep(0.1)  # Allow UI updates

    async def _test_model(self, model_id: str) -> None:
        """Test a model with a simple prompt."""
        ui.notify(f"Testing {model_id}...", type="info")

        success, message = self.services.model.test_model(model_id)

        if success:
            ui.notify(message, type="positive")
        else:
            ui.notify(message, type="negative")

    async def _delete_model(self, model_id: str) -> None:
        """Delete a model."""
        with ui.dialog() as dialog, ui.card():
            ui.label("Delete Model?").classes("text-lg font-semibold")
            ui.label(f"Are you sure you want to delete {model_id}?").classes(
                "text-gray-600 dark:text-gray-400"
            )

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Delete",
                    on_click=lambda: self._confirm_delete(dialog, model_id),
                ).props("color=negative")

        dialog.open()

    def _confirm_delete(self, dialog, model_id: str) -> None:
        """Confirm model deletion."""
        if self.services.model.delete_model(model_id):
            ui.notify(f"Deleted {model_id}", type="positive")
        else:
            ui.notify("Delete failed", type="negative")

        dialog.close()

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

        self._comparison_result.clear()

        with self._comparison_result:
            ui.label("Running comparison...").classes("text-gray-500 dark:text-gray-400")
            ui.spinner()

        # Run comparison
        results = self.services.model.compare_models(models, prompt)

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
