"""Models page - Model operations mixin (test, delete, update, tags)."""

import asyncio
import logging

from nicegui import ui

from src.settings import Settings
from src.ui.pages.models._page import ModelsPageBase

logger = logging.getLogger(__name__)


class OperationsMixin(ModelsPageBase):
    """Mixin providing model operations (test, delete, update, tags) for ModelsPage."""

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

    def _update_model_tags(self, model_id: str, tags: list[str]) -> None:
        """Update the tags for a model.

        Args:
            model_id: Model identifier.
            tags: List of role tags to assign.
        """
        logger.info(f"Updating tags for {model_id}: {tags}")
        settings = Settings.load()
        settings.set_model_tags(model_id, tags)
        ui.notify(f"Updated roles for {model_id}", type="positive")

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

    def _confirm_delete(self, dialog: ui.dialog, model_id: str) -> None:
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

    async def _download_updates(self, dialog: ui.dialog, model_ids: list[str]) -> None:
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
