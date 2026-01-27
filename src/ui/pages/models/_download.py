"""Download queue management for the Models page."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from nicegui import ui

logger = logging.getLogger("src.ui.pages.models._download")


def update_queue_status(page: Any) -> None:
    """Update the queue status label.

    Args:
        page: ModelsPage instance.
    """
    if page._queue_status_label is None:
        return

    if not page._download_queue:
        page._queue_status_label.text = ""
        return

    completed = sum(1 for t in page._download_queue.values() if t.status == "completed")
    total = len(page._download_queue)
    downloading = sum(1 for t in page._download_queue.values() if t.status == "downloading")
    queued = sum(1 for t in page._download_queue.values() if t.status == "queued")

    parts = []
    if downloading > 0:
        parts.append(f"{downloading} downloading")
    if queued > 0:
        parts.append(f"{queued} queued")
    if completed > 0:
        parts.append(f"{completed}/{total} complete")

    page._queue_status_label.text = ", ".join(parts) if parts else ""


async def download_all_filtered(page: Any) -> None:
    """Download all models that match current filters.

    Args:
        page: ModelsPage instance.
    """
    from src.ui.pages.models._listing import get_filtered_downloadable_models

    downloadable = get_filtered_downloadable_models(page)

    if not downloadable:
        ui.notify("No models to download", type="info")
        return

    logger.info(f"Queueing {len(downloadable)} models for download: {downloadable}")

    # Show confirmation with model list
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(f"Download {len(downloadable)} Models?").classes("text-lg font-semibold mb-2")
        ui.label(f"Downloads will run {page._max_concurrent_downloads} at a time.").classes(
            "text-sm text-gray-500 dark:text-gray-400 mb-2"
        )

        with ui.scroll_area().classes("max-h-48 w-full"):
            for model_id in downloadable:
                ui.label(f"• {model_id}").classes("text-sm")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Download All",
                on_click=lambda: confirm_download_all(page, dialog, downloadable),
                icon="download",
            ).props("color=primary")

    dialog.open()


async def confirm_download_all(page: Any, dialog: ui.dialog, model_ids: list[str]) -> None:
    """Confirm and start downloading all models.

    Args:
        page: ModelsPage instance.
        dialog: The confirmation dialog.
        model_ids: List of model IDs to download.
    """
    dialog.close()
    ui.notify(
        f"Starting download of {len(model_ids)} models ({page._max_concurrent_downloads} concurrent)...",
        type="info",
    )

    # Queue all downloads
    for model_id in model_ids:
        await queue_download(page, model_id)


async def queue_download(page: Any, model_id: str) -> None:
    """Add a model to the download queue and start processing if needed.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier to queue.
    """
    from src.ui.pages.models import DownloadTask

    # Check if already in queue (without lock for quick check)
    if model_id in page._download_queue:
        logger.debug(f"Model {model_id} already in queue")
        return

    task = DownloadTask(model_id=model_id)
    page._download_queue[model_id] = task
    logger.info(f"Queued model for download: {model_id}")

    # Create UI elements for this download (must be done in main thread context)
    create_download_card(page, task)
    update_queue_status(page)

    # Try to start downloads
    await process_download_queue(page)


def create_download_card(page: Any, task: Any) -> None:
    """Create a progress card for a download task.

    Args:
        page: ModelsPage instance.
        task: DownloadTask instance.
    """
    if not page._pull_progress:
        return

    try:
        page._pull_progress.set_visibility(True)

        with page._pull_progress:
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
                        on_click=lambda t=task: cancel_download(page, t),
                    ).props("flat dense round").tooltip("Cancel")
    except Exception as e:
        logger.exception(f"Error creating download card for {task.model_id}: {e}")


def cancel_download(page: Any, task: Any) -> None:
    """Cancel a download task.

    Args:
        page: ModelsPage instance.
        task: DownloadTask instance.
    """
    logger.info(f"Cancelling download: {task.model_id}")
    task.cancel_requested = True
    task.status = "error"
    task.status_text = "Cancelled"
    safe_update_label(task.status_label, "Cancelled")
    update_queue_status(page)


async def process_download_queue(page: Any) -> None:
    """Start queued downloads until the concurrency limit is reached.

    Marks eligible queued DownloadTask entries as "downloading" and schedules their
    execution as background asyncio Tasks, adding each Task to the _background_tasks
    set for lifecycle tracking.

    Args:
        page: ModelsPage instance.
    """
    active_downloads = sum(1 for t in page._download_queue.values() if t.status == "downloading")

    # Start new downloads up to the limit
    tasks_to_start = []
    for task in list(page._download_queue.values()):
        if active_downloads >= page._max_concurrent_downloads:
            break
        if task.status == "queued" and not task.cancel_requested:
            task.status = "downloading"
            active_downloads += 1
            tasks_to_start.append(task)

    # Start downloads outside the iteration
    for task in tasks_to_start:
        async_task = asyncio.create_task(execute_download(page, task))
        page._background_tasks.add(async_task)
        async_task.add_done_callback(page._background_tasks.discard)


async def execute_download(page: Any, task: Any) -> None:
    """Perform a single model download with progress streaming.

    Streams progress updates, handles cancellation and errors, updates task state
    and UI, then cleans up and continues the download queue.

    Args:
        page: ModelsPage instance.
        task: DownloadTask instance.
    """
    logger.info(f"Starting download: {task.model_id}")
    task.status_text = "Starting download..."
    safe_update_label(task.status_label, task.status_text)

    try:
        success = False
        async for progress in async_pull(page, task.model_id):
            if task.cancel_requested:
                break

            if "error" in progress:
                task.status_text = progress.get("status", "Error")
                task.status = "error"
                safe_update_label(task.status_label, task.status_text)
                break

            status = progress.get("status", "")
            task.status_text = status
            safe_update_label(task.status_label, status)

            total = progress.get("total") or 0
            completed = progress.get("completed") or 0
            if total > 0:
                task.progress = completed / total
                safe_update_progress(task.progress_bar, task.progress)

            if "success" in status.lower() or (total > 0 and completed == total):
                success = True

        if success and not task.cancel_requested:
            task.status = "completed"
            task.status_text = "Download complete!"
            logger.info(f"Model {task.model_id} downloaded successfully")
        elif not task.cancel_requested and task.status != "error":
            task.status = "error"
            task.status_text = "Download failed"
            logger.warning(f"Model {task.model_id} download failed")

    except Exception as e:
        logger.exception(f"Error downloading {task.model_id}")
        task.status = "error"
        task.status_text = f"Error: {e}"

    safe_update_label(task.status_label, task.status_text)
    update_queue_status(page)

    # Clean up completed/error downloads after a delay
    await asyncio.sleep(3)
    cleanup_download_card(page, task)

    # Continue processing queue
    await process_download_queue(page)

    # Refresh lists if any downloads completed (no notify from background)
    if task.status == "completed":
        from src.ui.pages.models._operations import refresh_all

        refresh_all(page, notify=False)


def safe_update_label(label: Any, text: str) -> None:
    """Safely update a label's text, handling deleted elements.

    Args:
        label: NiceGUI label element.
        text: New text to set.
    """
    try:
        if label:
            label.text = text
    except Exception:
        logger.debug("Label element already deleted, skipping update to: %s", text[:50])


def safe_update_progress(progress_bar: Any, value: float) -> None:
    """Safely update a progress bar's value, handling deleted elements.

    Args:
        progress_bar: NiceGUI progress bar element.
        value: New progress value (0.0 to 1.0).
    """
    try:
        if progress_bar:
            progress_bar.value = value
    except Exception:
        logger.debug("Progress bar element already deleted, skipping update to: %.1f", value)


def cleanup_download_card(page: Any, task: Any) -> None:
    """Remove a download card from the UI.

    Args:
        page: ModelsPage instance.
        task: DownloadTask instance.
    """
    try:
        if task.card:
            task.card.delete()
            task.card = None
    except Exception:
        logger.debug("Download card for %s already deleted during cleanup", task.model_id)

    # Safely remove from queue
    page._download_queue.pop(task.model_id, None)

    update_queue_status(page)

    # Hide progress section if empty
    try:
        if page._pull_progress and not page._download_queue:
            page._pull_progress.set_visibility(False)
    except Exception:
        logger.debug("Progress section element already deleted during visibility update")


async def async_pull(page: Any, model_id: str) -> AsyncGenerator[dict[str, Any]]:
    """Async wrapper for pull generator.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier to pull.

    Yields:
        Progress dictionaries from the pull operation.
    """
    for progress in page.services.model.pull_model(model_id):
        yield progress
        await asyncio.sleep(0.1)  # Allow UI updates


async def pull_custom_model(page: Any) -> None:
    """Pull a custom model from user input.

    Args:
        page: ModelsPage instance.
    """
    if page._custom_model_input is None:
        return

    model_id = page._custom_model_input.value.strip()
    if not model_id:
        ui.notify("Enter a model name", type="warning")
        return

    page._custom_model_input.value = ""
    await pull_model(page, model_id)


async def pull_model(page: Any, model_id: str) -> None:
    """Pull a model from Ollama using the download queue.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier to download.
    """
    logger.info(f"Queueing download of model: {model_id}")
    await queue_download(page, model_id)


async def download_updates(page: Any, dialog: ui.dialog, model_ids: list[str]) -> None:
    """Download updates for the given models.

    Args:
        page: ModelsPage instance.
        dialog: The dialog to close.
        model_ids: List of model IDs to update.
    """
    dialog.close()
    logger.info(f"Downloading updates for {len(model_ids)} models")
    ui.notify(f"Queueing {len(model_ids)} updates for download...", type="info")

    for model_id in model_ids:
        await queue_download(page, model_id)


async def update_model(page: Any, model_id: str) -> None:
    """Update a specific model by re-pulling it.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier to update.
    """
    from src.ui.pages.models._operations import refresh_all

    logger.info(f"Updating model: {model_id}")
    ui.notify(f"Updating {model_id}...", type="info")

    # Show progress in pull progress area
    if page._pull_progress:
        page._pull_progress.set_visibility(True)
        page._pull_progress.clear()

        with page._pull_progress:
            status_label = ui.label(f"Updating {model_id}...")
            progress_bar = ui.linear_progress(value=0).classes("w-full")

        success = False
        async for progress in async_pull(page, model_id):
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
            refresh_all(page)

        page._pull_progress.set_visibility(False)
