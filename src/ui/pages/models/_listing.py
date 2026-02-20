"""Available model listing and filtering for the Models page."""

import logging
from typing import Any

from nicegui import ui

from src.settings import RECOMMENDED_MODELS, ModelInfo
from src.ui.local_prefs import load_prefs_deferred, save_pref
from src.ui.theme import get_quality_color

logger = logging.getLogger(__name__)

_PAGE_KEY = "models_listing"


def build_available_section(page: Any) -> None:
    """Build the available models section.

    Args:
        page: ModelsPage instance.
    """
    from src.ui.pages.models._download import pull_custom_model

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center mb-4"):
            ui.icon("download").classes("text-blue-500")
            ui.label("Available Models").classes("text-lg font-semibold")
            ui.icon("help_outline", size="xs").classes("text-gray-400 cursor-help").tooltip(
                "Models from the Ollama registry you can download"
            )

        # Filter controls - responsive grid
        with ui.element("div").classes(
            "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4 p-4 bg-gray-800 rounded-lg"
        ):
            # Search
            ui.input(
                label="Search",
                placeholder="Model name...",
                on_change=lambda e: apply_filters(page, search=e.value),
            ).classes("w-full").props("dense clearable")

            # Quality filter
            page._quality_min_select = (
                ui.select(
                    label="Min Quality",
                    options={0: "Any", 5: "5+", 7: "7+", 8: "8+"},
                    value=0,
                    on_change=lambda e: apply_filters(page, quality=e.value),
                )
                .classes("w-full")
                .props("dense")
            )

            # Checkboxes row
            with ui.column().classes("gap-2"):
                page._fits_vram_cb = ui.checkbox(
                    "Fits my VRAM (~10% tolerance)",
                    value=True,
                    on_change=lambda e: apply_filters(page, fits_vram=e.value),
                ).tooltip("Show models that fit your available VRAM with 10% tolerance")

                page._uncensored_cb = ui.checkbox(
                    "Uncensored only",
                    value=False,
                    on_change=lambda e: apply_filters(page, uncensored=e.value),
                ).tooltip("Only show models without content filters")

        # Download All button row
        with ui.row().classes("w-full items-center gap-4 mb-2"):
            from src.ui.pages.models._download import download_all_filtered

            page._download_all_btn = ui.button(
                "Download All Filtered",
                icon="download",
                on_click=lambda: download_all_filtered(page),
            ).props("color=primary")
            page._queue_status_label = ui.label("").classes("text-sm text-gray-400")
            update_download_all_btn(page)

        # Custom model pull
        with ui.expansion("Pull Custom Model", icon="add_circle").classes("w-full"):
            ui.label("Enter any Ollama model name to download").classes(
                "text-sm text-gray-400 mb-2"
            )
            with ui.row().classes("w-full gap-2"):
                page._custom_model_input = (
                    ui.input(placeholder="e.g., llama3.2:8b or vanilj/model-name:tag")
                    .classes("flex-grow")
                    .props("dense")
                )
                ui.button(
                    "Pull",
                    on_click=lambda: pull_custom_model(page),
                    icon="download",
                ).props("color=primary dense")

        # Pull progress
        page._pull_progress = ui.column().classes("w-full mt-4")
        page._pull_progress.set_visibility(False)

        # Model cards
        page._model_list = ui.column().classes("w-full gap-4")
        refresh_model_list(page)

    # Restore persisted preferences from localStorage
    load_prefs_deferred(_PAGE_KEY, lambda prefs: _apply_prefs(page, prefs))


def apply_filters(
    page: Any,
    search: str | None = None,
    quality: int | None = None,
    fits_vram: bool | None = None,
    uncensored: bool | None = None,
) -> None:
    """Apply filters and refresh model list.

    Args:
        page: ModelsPage instance.
        search: Search text filter.
        quality: Minimum quality filter.
        fits_vram: VRAM fit filter.
        uncensored: Uncensored-only filter.
    """
    if search is not None:
        page._filter_search = search
    if quality is not None:
        page._filter_quality_min = quality
        save_pref(_PAGE_KEY, "filter_quality_min", quality)
    if fits_vram is not None:
        page._filter_fits_vram = fits_vram
        save_pref(_PAGE_KEY, "filter_fits_vram", fits_vram)
    if uncensored is not None:
        page._filter_uncensored_only = uncensored
        save_pref(_PAGE_KEY, "filter_uncensored_only", uncensored)

    refresh_model_list(page)
    update_download_all_btn(page)


def get_filtered_downloadable_models(page: Any) -> list[str]:
    """Get list of model IDs that match current filters and are not installed.

    Args:
        page: ModelsPage instance.

    Returns:
        List of downloadable model IDs matching filters.
    """
    vram = page._get_vram()
    installed = page._get_installed()
    downloadable = []

    for model_id, info in RECOMMENDED_MODELS.items():
        # Apply same filters as refresh_model_list
        if page._filter_search and page._filter_search.lower() not in info["name"].lower():
            continue
        if page._filter_quality_min and info["quality"] < page._filter_quality_min:
            continue
        if page._filter_uncensored_only and not info["uncensored"]:
            continue

        fits_vram = page._model_fits_vram(info["vram_required"], vram)
        if page._filter_fits_vram and not fits_vram:
            continue

        is_installed = any(model_id in m for m in installed)
        if not is_installed and fits_vram:
            downloadable.append(model_id)

    return downloadable


def update_download_all_btn(page: Any) -> None:
    """Update the Download All button state based on filters.

    Args:
        page: ModelsPage instance.
    """
    if page._download_all_btn is None:
        return

    downloadable = get_filtered_downloadable_models(page)
    count = len(downloadable)

    if count > 0:
        page._download_all_btn.text = f"Download All ({count})"
        page._download_all_btn.enable()
    else:
        page._download_all_btn.text = "Download All (0)"
        page._download_all_btn.disable()


def refresh_model_list(page: Any) -> None:
    """Refresh the model list with current filters.

    Args:
        page: ModelsPage instance.
    """
    if not page._model_list:
        return

    page._model_list.clear()

    vram = page._get_vram()
    installed = page._get_installed()

    with page._model_list:
        shown_count = 0
        for model_id, info in RECOMMENDED_MODELS.items():
            # Apply filters
            if page._filter_search and page._filter_search.lower() not in info["name"].lower():
                continue
            if page._filter_quality_min and info["quality"] < page._filter_quality_min:
                continue
            if page._filter_uncensored_only and not info["uncensored"]:
                continue

            fits_vram = page._model_fits_vram(info["vram_required"], vram)
            if page._filter_fits_vram and not fits_vram:
                continue

            is_installed = any(model_id in m for m in installed)
            build_model_card(page, model_id, info, fits_vram, is_installed, vram)
            shown_count += 1

        if shown_count == 0:
            ui.label("No models match your filters").classes("text-gray-400 text-center py-8")


def build_model_card(
    page: Any,
    model_id: str,
    info: ModelInfo,
    fits_vram: bool,
    is_installed: bool,
    available_vram: float = 0,
) -> None:
    """Build a model card.

    Args:
        page: ModelsPage instance.
        model_id: Model identifier.
        info: Model info dictionary.
        fits_vram: Whether model fits available VRAM.
        is_installed: Whether model is installed.
        available_vram: Available VRAM for comparison display.
    """
    from src.ui.pages.models._download import pull_model
    from src.ui.pages.models._operations import delete_model, test_model

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
                        on_click=lambda m=model_id: pull_model(page, m),
                        icon="download",
                    ).props("color=primary dense")
                else:
                    ui.label(f"Needs {info['vram_required']}GB").classes(
                        "text-sm text-orange-400"
                    ).tooltip(
                        f"Requires {info['vram_required']}GB VRAM, you have ~{available_vram}GB available"
                    )
            else:
                with ui.row().classes("gap-1"):
                    ui.button(
                        icon="play_arrow", on_click=lambda m=model_id: test_model(page, m)
                    ).props("flat dense round").tooltip("Test model")
                    ui.button(
                        icon="delete", on_click=lambda m=model_id: delete_model(page, m)
                    ).props("flat dense round color=negative").tooltip("Delete model")

        # Description
        ui.label(info["description"]).classes("text-sm text-gray-400 mb-1")

        # Role tags
        tags = info.get("tags", [])
        if tags:
            with ui.row().classes("flex-wrap gap-1 mb-3"):
                for tag in tags:
                    ui.badge(tag).props("outline color=grey-7").classes("text-xs")
        else:
            ui.space().classes("mb-3")

        # Stats grid
        with ui.element("div").classes("grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm"):
            # Size
            with ui.column().classes("gap-0.5"):
                ui.label("Disk Size").classes("text-xs text-gray-400")
                ui.label(f"{info['size_gb']} GB").classes("font-medium")

            # VRAM
            with ui.column().classes("gap-0.5"):
                ui.label("VRAM Needed").classes("text-xs text-gray-400")
                vram_class = "font-medium"
                if not fits_vram:
                    vram_class += " text-orange-500"
                ui.label(f"{info['vram_required']} GB").classes(vram_class)

            # Quality bar
            with ui.column().classes("gap-0.5"):
                ui.label("Quality").classes("text-xs text-gray-400")
                with ui.row().classes("gap-0.5 items-center"):
                    for i in range(10):
                        if i < info["quality"]:
                            ui.element("div").style(
                                f"width: 8px; height: 8px; background: {quality_color}; border-radius: 2px;"
                            )
                        else:
                            ui.element("div").classes("bg-gray-600").style(
                                "width: 8px; height: 8px; border-radius: 2px;"
                            )
                    ui.label(f"{info['quality']}/10").classes("text-xs ml-1")

            # Speed bar
            with ui.column().classes("gap-0.5"):
                ui.label("Speed").classes("text-xs text-gray-400")
                with ui.row().classes("gap-0.5 items-center"):
                    for i in range(10):
                        if i < info["speed"]:
                            ui.element("div").style(
                                "width: 8px; height: 8px; background: #4CAF50; border-radius: 2px;"
                            )
                        else:
                            ui.element("div").classes("bg-gray-600").style(
                                "width: 8px; height: 8px; border-radius: 2px;"
                            )
                    ui.label(f"{info['speed']}/10").classes("text-xs ml-1")


def _apply_prefs(page: Any, prefs: dict) -> None:
    """Apply loaded preferences to model listing filter state and UI widgets.

    Args:
        page: ModelsPage instance.
        prefs: Dict of field→value from localStorage.
    """
    if not prefs:
        return

    changed = False

    if "filter_quality_min" in prefs and isinstance(prefs["filter_quality_min"], int):
        val = prefs["filter_quality_min"]
        if val in (0, 5, 7, 8) and val != page._filter_quality_min:
            page._filter_quality_min = val
            changed = True
            if hasattr(page, "_quality_min_select") and page._quality_min_select:
                page._quality_min_select.value = val

    if "filter_fits_vram" in prefs and isinstance(prefs["filter_fits_vram"], bool):
        val = prefs["filter_fits_vram"]
        if val != page._filter_fits_vram:
            page._filter_fits_vram = val
            changed = True
            if hasattr(page, "_fits_vram_cb") and page._fits_vram_cb:
                page._fits_vram_cb.value = val

    if "filter_uncensored_only" in prefs and isinstance(prefs["filter_uncensored_only"], bool):
        val = prefs["filter_uncensored_only"]
        if val != page._filter_uncensored_only:
            page._filter_uncensored_only = val
            changed = True
            if hasattr(page, "_uncensored_cb") and page._uncensored_cb:
                page._uncensored_cb.value = val

    if changed:
        logger.info("Restored model listing preferences from localStorage")
        refresh_model_list(page)
        update_download_all_btn(page)
