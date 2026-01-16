"""Models page - Ollama model management."""

import asyncio
import logging
from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.card import Card
from nicegui.elements.column import Column
from nicegui.elements.input import Input

from services import ServiceContainer
from settings import AVAILABLE_MODELS, ModelInfo
from ui.state import AppState
from ui.theme import get_quality_color

logger = logging.getLogger(__name__)


@dataclass
class DownloadTask:
    """Tracks state of a single download."""

    model_id: str
    status: str = "queued"  # queued, downloading, completed, error
    progress: float = 0.0
    status_text: str = "Queued..."
    # UI elements (typed as Any to avoid NiceGUI import issues)
    card: Card | None = None
    progress_bar: object | None = None
    status_label: object | None = None
    cancel_requested: bool = False


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
        self._installed_section: Card | None = None
        self._custom_model_input: Input | None = None

        # Download queue - tracks all active/pending downloads
        self._download_queue: dict[str, DownloadTask] = {}
        self._download_lock = asyncio.Lock()
        self._max_concurrent_downloads = 3  # Allow 3 concurrent downloads

        # Filter state
        self._filter_fits_vram = True
        self._filter_quality_min = 0
        self._filter_uncensored_only = False
        self._filter_search = ""

        # Download all state
        self._download_all_btn: ui.button | None = None
        self._queue_status_label: ui.label | None = None

        # Cached values to avoid redundant service calls
        self._cached_vram: float | None = None
        self._cached_installed: set[str] | None = None

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

    def _get_vram(self) -> float:
        """Get VRAM from cache or service."""
        if self._cached_vram is None:
            self._cached_vram = self.services.model.get_vram()
        return self._cached_vram

    def _get_installed(self) -> set[str]:
        """Get installed models from cache or service."""
        if self._cached_installed is None:
            self._cached_installed = set(self.services.model.list_installed())
        return self._cached_installed

    def _invalidate_cache(self) -> None:
        """Invalidate cached values - call after model installs/deletes."""
        self._cached_vram = None
        self._cached_installed = None

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
        vram = self._get_vram()
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
        self._installed_section = ui.card().classes("w-full")
        self._refresh_installed_section()

    def _refresh_installed_section(self) -> None:
        """Refresh the installed models section."""
        if self._installed_section is None:
            return

        self._installed_section.clear()

        with self._installed_section:
            with ui.row().classes("w-full items-center mb-4"):
                ui.label("Installed Models").classes("text-lg font-semibold")
                ui.space()
                ui.button(
                    "Check for Updates",
                    icon="system_update",
                    on_click=self._check_all_updates,
                ).props("flat dense").tooltip("Check all models for updates")
                ui.button(icon="refresh", on_click=self._refresh_all).props(
                    "flat dense round"
                ).tooltip("Refresh model list")

            installed_with_sizes = self.services.model.list_installed_with_sizes()

            if not installed_with_sizes:
                ui.label("No models installed").classes("text-gray-500 dark:text-gray-400")
                ui.label("Download models below to get started").classes(
                    "text-sm text-gray-400 dark:text-gray-500"
                )
                return

            # Model cards for installed
            for model_id, size_gb in installed_with_sizes.items():
                info = self.services.model.get_model_info(model_id)
                with ui.card().classes("w-full mb-2").props("flat bordered"):
                    with ui.row().classes("w-full items-center"):
                        with ui.column().classes("flex-grow gap-0"):
                            ui.label(info["name"]).classes("font-medium")
                            ui.label(model_id).classes("text-xs text-gray-500 dark:text-gray-400")
                        ui.label(f"{size_gb} GB").classes(
                            "text-sm text-gray-600 dark:text-gray-400"
                        )
                        with ui.row().classes("gap-1"):
                            ui.button(
                                icon="play_arrow",
                                on_click=lambda m=model_id: self._test_model(m),
                            ).props("flat dense round").tooltip("Test model")
                            ui.button(
                                icon="update",
                                on_click=lambda m=model_id: self._update_model(m),
                            ).props("flat dense round").tooltip("Update model")
                            ui.button(
                                icon="delete",
                                on_click=lambda m=model_id: self._delete_model(m),
                            ).props("flat dense round color=negative").tooltip("Delete")

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

            # Download All button row
            with ui.row().classes("w-full items-center gap-4 mb-2"):
                self._download_all_btn = ui.button(
                    "Download All Filtered",
                    icon="download",
                    on_click=self._download_all_filtered,
                ).props("color=primary")
                self._queue_status_label = ui.label("").classes(
                    "text-sm text-gray-600 dark:text-gray-400"
                )
                self._update_download_all_btn()

            # Custom model pull
            with ui.expansion("Pull Custom Model", icon="add_circle").classes("w-full"):
                ui.label("Enter any Ollama model name to download").classes(
                    "text-sm text-gray-500 dark:text-gray-400 mb-2"
                )
                with ui.row().classes("w-full gap-2"):
                    self._custom_model_input = (
                        ui.input(placeholder="e.g., llama3.2:8b or vanilj/model-name:tag")
                        .classes("flex-grow")
                        .props("dense")
                    )
                    ui.button(
                        "Pull",
                        on_click=self._pull_custom_model,
                        icon="download",
                    ).props("color=primary dense")

            # Pull progress
            self._pull_progress = ui.column().classes("w-full mt-4")
            self._pull_progress.set_visibility(False)

            # Model cards
            self._model_list = ui.column().classes("w-full gap-4")
            self._refresh_model_list()

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
        self._update_download_all_btn()

    def _get_filtered_downloadable_models(self) -> list[str]:
        """Get list of model IDs that match current filters and are not installed."""
        vram = self.services.model.get_vram()
        installed = set(self.services.model.list_installed())
        downloadable = []

        for model_id, info in AVAILABLE_MODELS.items():
            # Apply same filters as _refresh_model_list
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
            if not is_installed and fits_vram:
                downloadable.append(model_id)

        return downloadable

    def _update_download_all_btn(self) -> None:
        """Update the Download All button state based on filters."""
        if self._download_all_btn is None:
            return

        downloadable = self._get_filtered_downloadable_models()
        count = len(downloadable)

        if count > 0:
            self._download_all_btn.text = f"Download All ({count})"
            self._download_all_btn.enable()
        else:
            self._download_all_btn.text = "Download All (0)"
            self._download_all_btn.disable()

    def _update_queue_status(self) -> None:
        """Update the queue status label."""
        if self._queue_status_label is None:
            return

        if not self._download_queue:
            self._queue_status_label.text = ""
            return

        completed = sum(1 for t in self._download_queue.values() if t.status == "completed")
        total = len(self._download_queue)
        downloading = sum(1 for t in self._download_queue.values() if t.status == "downloading")
        queued = sum(1 for t in self._download_queue.values() if t.status == "queued")

        parts = []
        if downloading > 0:
            parts.append(f"{downloading} downloading")
        if queued > 0:
            parts.append(f"{queued} queued")
        if completed > 0:
            parts.append(f"{completed}/{total} complete")

        self._queue_status_label.text = ", ".join(parts) if parts else ""

    async def _download_all_filtered(self) -> None:
        """Download all models that match current filters."""
        downloadable = self._get_filtered_downloadable_models()

        if not downloadable:
            ui.notify("No models to download", type="info")
            return

        logger.info(f"Queueing {len(downloadable)} models for download: {downloadable}")

        # Show confirmation with model list
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"Download {len(downloadable)} Models?").classes("text-lg font-semibold mb-2")
            ui.label(f"Downloads will run {self._max_concurrent_downloads} at a time.").classes(
                "text-sm text-gray-500 dark:text-gray-400 mb-2"
            )

            with ui.scroll_area().classes("max-h-48 w-full"):
                for model_id in downloadable:
                    ui.label(f"• {model_id}").classes("text-sm")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Download All",
                    on_click=lambda: self._confirm_download_all(dialog, downloadable),
                    icon="download",
                ).props("color=primary")

        dialog.open()

    async def _confirm_download_all(self, dialog, model_ids: list[str]) -> None:
        """Confirm and start downloading all models."""
        dialog.close()
        ui.notify(
            f"Starting download of {len(model_ids)} models ({self._max_concurrent_downloads} concurrent)...",
            type="info",
        )

        # Queue all downloads
        for model_id in model_ids:
            await self._queue_download(model_id)

    async def _queue_download(self, model_id: str) -> None:
        """Add a model to the download queue and start processing if needed."""
        # Check if already in queue (without lock for quick check)
        if model_id in self._download_queue:
            logger.debug(f"Model {model_id} already in queue")
            return

        task = DownloadTask(model_id=model_id)
        self._download_queue[model_id] = task
        logger.info(f"Queued model for download: {model_id}")

        # Create UI elements for this download (must be done in main thread context)
        self._create_download_card(task)
        self._update_queue_status()

        # Try to start downloads
        await self._process_download_queue()

    def _create_download_card(self, task: DownloadTask) -> None:
        """Create a progress card for a download task."""
        if not self._pull_progress:
            return

        try:
            self._pull_progress.set_visibility(True)

            with self._pull_progress:
                task.card = ui.card().classes("w-full mb-2").props("flat bordered")
                with task.card:
                    with ui.row().classes("w-full items-center gap-2"):
                        with ui.column().classes("flex-grow gap-1"):
                            with ui.row().classes("items-center gap-2"):
                                ui.spinner(size="sm")
                                ui.label(task.model_id).classes("font-medium")
                            task.progress_bar = ui.linear_progress(value=0).classes("w-full")
                            task.status_label = ui.label(task.status_text).classes(
                                "text-sm text-gray-500 dark:text-gray-400"
                            )
                        ui.button(
                            icon="close",
                            on_click=lambda t=task: self._cancel_download(t),
                        ).props("flat dense round").tooltip("Cancel")
        except Exception as e:
            logger.exception(f"Error creating download card for {task.model_id}: {e}")

    def _cancel_download(self, task: DownloadTask) -> None:
        """Cancel a download task."""
        logger.info(f"Cancelling download: {task.model_id}")
        task.cancel_requested = True
        task.status = "error"
        task.status_text = "Cancelled"
        self._safe_update_label(task.status_label, "Cancelled")
        self._update_queue_status()

    async def _process_download_queue(self) -> None:
        """Process the download queue, starting downloads up to the concurrency limit."""
        active_downloads = sum(
            1 for t in self._download_queue.values() if t.status == "downloading"
        )

        # Start new downloads up to the limit
        tasks_to_start = []
        for task in list(self._download_queue.values()):
            if active_downloads >= self._max_concurrent_downloads:
                break
            if task.status == "queued" and not task.cancel_requested:
                task.status = "downloading"
                active_downloads += 1
                tasks_to_start.append(task)

        # Start downloads outside the iteration
        for task in tasks_to_start:
            asyncio.create_task(self._execute_download(task))

    async def _execute_download(self, task: DownloadTask) -> None:
        """Execute a single download task."""
        logger.info(f"Starting download: {task.model_id}")
        task.status_text = "Starting download..."
        self._safe_update_label(task.status_label, task.status_text)

        try:
            success = False
            async for progress in self._async_pull(task.model_id):
                if task.cancel_requested:
                    break

                if "error" in progress:
                    task.status_text = progress.get("status", "Error")
                    task.status = "error"
                    self._safe_update_label(task.status_label, task.status_text)
                    break

                status = progress.get("status", "")
                task.status_text = status
                self._safe_update_label(task.status_label, status)

                total = progress.get("total") or 0
                completed = progress.get("completed") or 0
                if total > 0:
                    task.progress = completed / total
                    self._safe_update_progress(task.progress_bar, task.progress)

                if "success" in status.lower() or (total > 0 and completed == total):
                    success = True

            if success and not task.cancel_requested:
                task.status = "completed"
                task.status_text = "Download complete!"
                logger.info(f"Model {task.model_id} downloaded successfully")
                # Don't call ui.notify here - we're in a background context
                # The status label update is sufficient feedback
            elif not task.cancel_requested and task.status != "error":
                task.status = "error"
                task.status_text = "Download failed"
                logger.warning(f"Model {task.model_id} download failed")

        except Exception as e:
            logger.exception(f"Error downloading {task.model_id}")
            task.status = "error"
            task.status_text = f"Error: {e}"

        self._safe_update_label(task.status_label, task.status_text)
        self._update_queue_status()

        # Clean up completed/error downloads after a delay
        await asyncio.sleep(3)
        self._cleanup_download_card(task)

        # Continue processing queue
        await self._process_download_queue()

        # Refresh lists if any downloads completed (no notify from background)
        if task.status == "completed":
            self._refresh_all(notify=False)

    def _safe_update_label(self, label, text: str) -> None:
        """Safely update a label's text, handling deleted elements."""
        try:
            if label:
                label.text = text
        except Exception:
            pass  # Element may have been deleted

    def _safe_update_progress(self, progress_bar, value: float) -> None:
        """Safely update a progress bar's value, handling deleted elements."""
        try:
            if progress_bar:
                progress_bar.value = value
        except Exception:
            pass  # Element may have been deleted

    def _cleanup_download_card(self, task: DownloadTask) -> None:
        """Remove a download card from the UI."""
        try:
            if task.card:
                task.card.delete()
                task.card = None
        except Exception:
            pass  # Card may already be deleted

        # Safely remove from queue
        self._download_queue.pop(task.model_id, None)

        self._update_queue_status()

        # Hide progress section if empty
        try:
            if self._pull_progress and not self._download_queue:
                self._pull_progress.set_visibility(False)
        except Exception:
            pass

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

    def _refresh_all(self, notify: bool = True) -> None:
        """Refresh all model lists with visual feedback.

        Args:
            notify: Whether to show a notification. Set False when called from background.
        """
        self._invalidate_cache()
        self._refresh_installed_section()
        self._refresh_model_list()
        self._update_download_all_btn()
        if notify:
            ui.notify("Model lists refreshed", type="info")

    async def _pull_custom_model(self) -> None:
        """Pull a custom model from user input."""
        if self._custom_model_input is None:
            return

        model_id = self._custom_model_input.value.strip()
        if not model_id:
            ui.notify("Enter a model name", type="warning")
            return

        self._custom_model_input.value = ""
        await self._pull_model(model_id)

    async def _pull_model(self, model_id: str) -> None:
        """Pull a model from Ollama using the download queue."""
        logger.info(f"Queueing download of model: {model_id}")
        await self._queue_download(model_id)

    async def _async_pull(self, model_id: str):
        """Async wrapper for pull generator."""
        for progress in self.services.model.pull_model(model_id):
            yield progress
            await asyncio.sleep(0.1)  # Allow UI updates

    async def _test_model(self, model_id: str) -> None:
        """Test a model with a simple prompt."""
        logger.info(f"Testing model: {model_id}")
        ui.notify(f"Testing {model_id}...", type="info")

        success, message = self.services.model.test_model(model_id)

        if success:
            logger.info(f"Model test passed: {model_id}")
            ui.notify(message, type="positive")
        else:
            logger.warning(f"Model test failed: {model_id} - {message}")
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
        logger.info(f"Deleting model: {model_id}")
        if self.services.model.delete_model(model_id):
            logger.info(f"Model {model_id} deleted successfully")
            ui.notify(f"Deleted {model_id}", type="positive")
            # Refresh lists to reflect deletion
            self._refresh_all()
        else:
            logger.error(f"Failed to delete model {model_id}")
            ui.notify("Delete failed", type="negative")

        dialog.close()

    async def _check_all_updates(self) -> None:
        """Check all installed models for updates and offer to download them."""
        logger.info("Checking all models for updates")
        installed = self.services.model.list_installed()
        if not installed:
            ui.notify("No models installed", type="info")
            return

        # Show checking progress
        with ui.dialog() as check_dialog, ui.card().classes("w-96"):
            ui.label("Checking for Updates").classes("text-lg font-semibold mb-2")
            progress_label = ui.label(f"Checking 0/{len(installed)} models...")
            progress_bar = ui.linear_progress(value=0).classes("w-full")
            check_dialog.open()

        updates_available = []
        for i, model_id in enumerate(installed):
            progress_label.text = f"Checking {i + 1}/{len(installed)}: {model_id}"
            progress_bar.value = (i + 1) / len(installed)
            await asyncio.sleep(0.05)  # Allow UI update

            result = await asyncio.get_event_loop().run_in_executor(
                None, self.services.model.check_model_update, model_id
            )
            if result.get("has_update"):
                updates_available.append(model_id)

        check_dialog.close()

        if not updates_available:
            logger.info("All models are up to date")
            ui.notify("All models are up to date!", type="positive")
            return

        # Show dialog offering to download updates
        logger.info(f"Updates available for: {updates_available}")

        with ui.dialog() as update_dialog, ui.card().classes("w-96"):
            ui.label("Updates Available").classes("text-lg font-semibold mb-2")
            ui.label(f"Found updates for {len(updates_available)} model(s):").classes(
                "text-gray-600 dark:text-gray-400 mb-2"
            )
            for model_id in updates_available:
                ui.label(f"• {model_id}").classes("text-sm ml-2")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=update_dialog.close).props("flat")
                ui.button(
                    "Download All Updates",
                    on_click=lambda: self._download_updates(update_dialog, updates_available),
                    icon="download",
                ).props("color=primary")

        update_dialog.open()

    async def _download_updates(self, dialog, model_ids: list[str]) -> None:
        """Download updates for the given models."""
        dialog.close()
        logger.info(f"Downloading updates for {len(model_ids)} models")
        ui.notify(f"Queueing {len(model_ids)} updates for download...", type="info")

        for model_id in model_ids:
            await self._queue_download(model_id)

    async def _update_model(self, model_id: str) -> None:
        """Update a specific model by re-pulling it."""
        logger.info(f"Updating model: {model_id}")
        ui.notify(f"Updating {model_id}...", type="info")

        # Show progress in pull progress area
        if self._pull_progress:
            self._pull_progress.set_visibility(True)
            self._pull_progress.clear()

            with self._pull_progress:
                status_label = ui.label(f"Updating {model_id}...")
                progress_bar = ui.linear_progress(value=0).classes("w-full")

            success = False
            async for progress in self._async_pull(model_id):
                if "error" in progress:
                    status_label.text = progress["status"]
                    ui.notify(progress["status"], type="negative")
                    break

                status = progress.get("status", "")
                status_label.text = status
                total = progress.get("total") or 0
                completed = progress.get("completed") or 0
                if total > 0:
                    pct = completed / total
                    progress_bar.value = pct

                if "success" in status.lower() or "up to date" in status.lower():
                    success = True

            if success:
                logger.info(f"Model {model_id} update complete")
                ui.notify(f"Model {model_id} is up to date!", type="positive")
                self._refresh_all()

            self._pull_progress.set_visibility(False)

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
