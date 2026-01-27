"""Models page - Download functionality mixin."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from nicegui import ui
from nicegui.elements.card import Card

from src.ui.pages.models._page import ModelsPageBase

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


class DownloadMixin(ModelsPageBase):
    """Mixin providing download functionality for ModelsPage."""

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

    async def _confirm_download_all(self, dialog: ui.dialog, model_ids: list[str]) -> None:
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
        """
        Start queued downloads until the concurrency limit is reached.

        Marks eligible queued DownloadTask entries as "downloading" and schedules their
        execution as background asyncio Tasks, adding each Task to the `_background_tasks`
        set for lifecycle tracking.
        """
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
            async_task = asyncio.create_task(self._execute_download(task))
            self._background_tasks.add(async_task)
            async_task.add_done_callback(self._background_tasks.discard)

    async def _execute_download(self, task: DownloadTask) -> None:
        """
        Perform a single model download: stream progress updates, handle cancellation
        and errors, update task state and UI, and then clean up and continue the
        download queue.

        Parameters:
            task (DownloadTask): The download task to execute; its `status`, `status_text`,
                `progress`, and UI references will be updated in-place.
        """
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

    def _safe_update_label(self, label: Any, text: str) -> None:
        """Safely update a label's text, handling deleted elements."""
        try:
            if label:
                label.text = text
        except Exception:
            logger.debug("Label element already deleted, skipping update to: %s", text[:50])

    def _safe_update_progress(self, progress_bar: Any, value: float) -> None:
        """Safely update a progress bar's value, handling deleted elements."""
        try:
            if progress_bar:
                progress_bar.value = value
        except Exception:
            logger.debug("Progress bar element already deleted, skipping update to: %.1f", value)

    def _cleanup_download_card(self, task: DownloadTask) -> None:
        """Remove a download card from the UI."""
        try:
            if task.card:
                task.card.delete()
                task.card = None
        except Exception:
            logger.debug("Download card for %s already deleted during cleanup", task.model_id)

        # Safely remove from queue
        self._download_queue.pop(task.model_id, None)

        self._update_queue_status()

        # Hide progress section if empty
        try:
            if self._pull_progress and not self._download_queue:
                self._pull_progress.set_visibility(False)
        except Exception:
            logger.debug("Progress section element already deleted during visibility update")

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

    async def _async_pull(self, model_id: str) -> AsyncGenerator[dict[str, Any]]:
        """Async wrapper for pull generator."""
        for progress in self.services.model.pull_model(model_id):
            yield progress
            await asyncio.sleep(0.1)  # Allow UI updates
