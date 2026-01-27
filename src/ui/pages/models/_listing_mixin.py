"""Models page - Model listing and display mixin."""

import logging

from nicegui import ui

from src.settings import AGENT_ROLES, RECOMMENDED_MODELS, ModelInfo, Settings
from src.ui.pages.models._page import ModelsPageBase
from src.ui.theme import get_quality_color

logger = logging.getLogger(__name__)


class ListingMixin(ModelsPageBase):
    """Mixin providing model listing and display functionality for ModelsPage."""

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
            settings = Settings.load()
            for model_id, size_gb in installed_with_sizes.items():
                info = self.services.model.get_model_info(model_id)
                current_tags = settings.get_model_tags(model_id)
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

                    # Tags configuration
                    with ui.row().classes("w-full items-center gap-2 mt-2"):
                        ui.label("Roles:").classes("text-xs text-gray-500 dark:text-gray-400")
                        role_options = {role: AGENT_ROLES[role]["name"] for role in AGENT_ROLES}
                        ui.select(
                            options=role_options,
                            value=current_tags,
                            multiple=True,
                            on_change=lambda e, m=model_id: self._update_model_tags(m, e.value),
                        ).classes("flex-grow").props("dense use-chips").tooltip(
                            "Select agent roles this model can be used for"
                        )

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

        for model_id, info in RECOMMENDED_MODELS.items():
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

    def _refresh_model_list(self) -> None:
        """Refresh the model list with current filters."""
        if not self._model_list:
            return

        self._model_list.clear()

        vram = self.services.model.get_vram()
        installed = set(self.services.model.list_installed())

        with self._model_list:
            shown_count = 0
            for model_id, info in RECOMMENDED_MODELS.items():
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

    def _refresh_all(self, notify: bool = True) -> None:
        """Refresh all model lists with visual feedback.

        Args:
            notify: Whether to show a notification. Set False when called from background.
        """
        # This method is defined in base but needs to call methods from this mixin
        # We override to ensure proper method resolution
        self._invalidate_cache()
        self._refresh_installed_section()
        self._refresh_model_list()
        self._update_download_all_btn()
        if notify:
            ui.notify("Model lists refreshed", type="info")

    def _invalidate_cache(self) -> None:
        """Invalidate cached values - call after model installs/deletes."""
        # Defined in base class, type hint for mixin
        raise NotImplementedError
