"""Model operations (test, delete, compare, updates) for the Models page."""

import asyncio
import logging
from typing import Any

from nicegui import ui

from src.settings import AGENT_ROLES, Settings

logger = logging.getLogger(__name__)


def build_installed_section(page: Any) -> None:
    """Build the installed models section.

    Args:
        page: ModelsPage instance.
    """
    page._installed_section = ui.card().classes("w-full")
    refresh_installed_section(page)


def refresh_installed_section(page: Any) -> None:
    """Refresh the installed models section.

    Args:
        page: ModelsPage instance.
    """
    from src.ui.pages.models._download import update_model

    if page._installed_section is None:
        return

    page._installed_section.clear()

    with page._installed_section:
        with ui.row().classes("w-full items-center mb-4"):
            ui.label("Installed Models").classes("text-lg font-semibold")
            ui.space()
            ui.button(
                "Check for Updates",
                icon="system_update",
                on_click=lambda: check_all_updates(page),
            ).props("flat dense").tooltip("Check all models for updates")
            ui.button(icon="refresh", on_click=lambda: refresh_all(page)).props(
                "flat dense round"
            ).tooltip("Refresh model list")

        installed_with_sizes = page.services.model.list_installed_with_sizes()

        if not installed_with_sizes:
            ui.label("No models installed").classes("text-gray-400")
            ui.label("Download models below to get started").classes("text-sm text-gray-500")
            return

        # Model cards for installed
        settings = Settings.load()
        for model_id, size_gb in installed_with_sizes.items():
            info = page.services.model.get_model_info(model_id)
            current_tags = settings.get_model_tags(model_id)
            with ui.card().classes("w-full mb-2").props("flat bordered"):
                with ui.row().classes("w-full items-center"):
                    with ui.column().classes("flex-grow gap-0"):
                        ui.label(info["name"]).classes("font-medium")
                        ui.label(model_id).classes("text-xs text-gray-400")
                    ui.label(f"{size_gb} GB").classes("text-sm text-gray-400")
                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="play_arrow",
                            on_click=lambda m=model_id: test_model(page, m),
                        ).props("flat dense round").tooltip("Test model")
                        ui.button(
                            icon="update",
                            on_click=lambda m=model_id: update_model(page, m),
                        ).props("flat dense round").tooltip("Update model")
                        ui.button(
                            icon="delete",
                            on_click=lambda m=model_id: delete_model(page, m),
                        ).props("flat dense round color=negative").tooltip("Delete")

                # Tags configuration
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    ui.label("Roles:").classes("text-xs text-gray-400")
                    role_options = {role: AGENT_ROLES[role]["name"] for role in AGENT_ROLES}
                    ui.select(
                        options=role_options,
                        value=current_tags,
                        multiple=True,
                        on_change=lambda e, m=model_id: update_model_tags(page, m, e.value),
                    ).classes("flex-grow").props("dense use-chips").tooltip(
                        "Select agent roles this model can be used for"
                    )


def build_comparison_section(page: Any) -> None:
    """Build the model comparison section.

    Args:
        page: ModelsPage instance.
    """
    with ui.card().classes("w-full"):
        ui.label("Model Comparison").classes("text-lg font-semibold mb-4")
        ui.label("Compare models on the same prompt to see output quality and speed").classes(
            "text-sm text-gray-400 mb-4"
        )

        installed = page.services.model.list_installed()

        if len(installed) < 2:
            ui.label("Install at least 2 models to compare").classes("text-gray-400")
            return

        # Model selection
        with ui.row().classes("w-full gap-4 mb-4"):
            page._compare_models_select = ui.select(
                label="Select models to compare",
                options=installed,
                multiple=True,
            ).classes("flex-grow")

        # Prompt input
        page._compare_prompt = ui.textarea(
            label="Test prompt",
            value="Write a short paragraph describing a sunset over the ocean.",
        ).classes("w-full")

        ui.button(
            "Run Comparison",
            on_click=lambda: run_comparison(page),
            icon="compare",
        ).props("color=primary")

        # Results
        page._comparison_result = ui.column().classes("w-full mt-4")


def refresh_all(page: Any, notify: bool = True) -> None:
    """Refresh all model lists with visual feedback.

    Args:
        page: ModelsPage instance.
        notify: Whether to show a notification. Set False when called from background.
    """
    from src.ui.pages.models._listing import refresh_model_list, update_download_all_btn

    page._invalidate_cache()
    refresh_installed_section(page)
    refresh_model_list(page)
    update_download_all_btn(page)
    if notify:
        try:
            ui.notify("Model lists refreshed", type="info")
        except RuntimeError:
            logger.debug("Skipped notification — UI context unavailable")


async def test_model(page: Any, model_id: str) -> None:
    """Test a model with a simple prompt.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier to test.
    """
    logger.info(f"Testing model: {model_id}")
    ui.notify(f"Testing {model_id}...", type="info")

    success, message = page.services.model.test_model(model_id)

    if success:
        logger.info(f"Model test passed: {model_id}")
        ui.notify(message, type="positive")
    else:
        logger.warning(f"Model test failed: {model_id} - {message}")
        ui.notify(message, type="negative")


def update_model_tags(page: Any, model_id: str, tags: list[str]) -> None:
    """Update the tags for a model.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier.
        tags: List of role tags to assign.
    """
    logger.info(f"Updating tags for {model_id}: {tags}")
    settings = Settings.load()
    settings.set_model_tags(model_id, tags)
    ui.notify(f"Updated roles for {model_id}", type="positive")


async def delete_model(page: Any, model_id: str) -> None:
    """Delete a model with confirmation dialog.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier to delete.
    """
    with ui.dialog() as dialog, ui.card():
        ui.label("Delete Model?").classes("text-lg font-semibold")
        ui.label(f"Are you sure you want to delete {model_id}?").classes("text-gray-400")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Delete",
                on_click=lambda: confirm_delete(page, dialog, model_id),
            ).props("color=negative")

    dialog.open()


def confirm_delete(page: Any, dialog: ui.dialog, model_id: str) -> None:
    """Confirm model deletion.

    Args:
        page: ModelsPage instance.
        dialog: The confirmation dialog.
        model_id: Model identifier to delete.
    """
    logger.info(f"Deleting model: {model_id}")
    if page.services.model.delete_model(model_id):
        logger.info(f"Model {model_id} deleted successfully")
        dialog.close()
        ui.notify(f"Deleted {model_id}", type="positive")
        refresh_all(page)
    else:
        logger.error(f"Failed to delete model {model_id}")
        dialog.close()
        ui.notify("Delete failed", type="negative")


async def check_all_updates(page: Any) -> None:
    """Check all installed models for updates and offer to download them.

    Args:
        page: ModelsPage instance.
    """
    from src.ui.pages.models._download import download_updates

    logger.info("Checking all models for updates")
    installed = page.services.model.list_installed()
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
            None, page.services.model.check_model_update, model_id
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
            "text-gray-400 mb-2"
        )
        for model_id in updates_available:
            ui.label(f"• {model_id}").classes("text-sm ml-2")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=update_dialog.close).props("flat")
            ui.button(
                "Download All Updates",
                on_click=lambda: download_updates(page, update_dialog, updates_available),
                icon="download",
            ).props("color=primary")

    update_dialog.open()


async def run_comparison(page: Any) -> None:
    """Run model comparison.

    Args:
        page: ModelsPage instance.
    """
    if not page._comparison_result:
        return

    models = page._compare_models_select.value if page._compare_models_select else []
    prompt = page._compare_prompt.value if page._compare_prompt else ""

    if len(models) < 2:
        ui.notify("Select at least 2 models", type="warning")
        return

    if not prompt:
        ui.notify("Enter a prompt", type="warning")
        return

    logger.info(f"Running model comparison: {models}")

    page._comparison_result.clear()

    with page._comparison_result:
        ui.label("Running comparison...").classes("text-gray-400")
        ui.spinner()

    # Run comparison
    results = page.services.model.compare_models(models, prompt)
    logger.info(f"Model comparison complete: {len(results)} results")

    page._comparison_result.clear()

    with page._comparison_result:
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
