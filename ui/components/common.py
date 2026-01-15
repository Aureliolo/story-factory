"""Common reusable UI components."""

from nicegui import ui


class LoadingSpinner:
    """A loading spinner component.

    Usage:
        spinner = LoadingSpinner()
        spinner.show()
        # ... do async work ...
        spinner.hide()
    """

    def __init__(self, message: str = "Loading...", size: str = "lg"):
        """Initialize loading spinner.

        Args:
            message: Loading message to display.
            size: Icon size (sm, md, lg, xl).
        """
        self.message = message
        self.size = size
        self._container: ui.element | None = None
        self._visible = False

    def show(self) -> None:
        """Show the loading spinner."""
        if self._container is not None:
            self._container.set_visibility(True)
            self._visible = True

    def hide(self) -> None:
        """Hide the loading spinner."""
        if self._container is not None:
            self._container.set_visibility(False)
            self._visible = False

    def build(self, inline: bool = False) -> None:
        """Build the spinner UI.

        Args:
            inline: If True, display inline. Otherwise in a centered column.
        """
        if inline:
            self._container = ui.row().classes("items-center gap-2")
        else:
            self._container = ui.column().classes("items-center justify-center gap-2 p-4")

        with self._container:
            ui.spinner(size=self.size, color="primary")
            ui.label(self.message).classes("text-sm text-gray-500")

        if not self._visible:
            self._container.set_visibility(False)


def tooltip_button(
    text: str,
    tooltip: str,
    on_click=None,
    icon: str | None = None,
    color: str = "primary",
    **kwargs,
) -> ui.button:
    """Create a button with a tooltip.

    Args:
        text: Button text.
        tooltip: Tooltip text.
        on_click: Click handler.
        icon: Optional icon name.
        color: Button color.
        **kwargs: Additional button props.

    Returns:
        Button element with tooltip.
    """
    btn = ui.button(text, on_click=on_click, icon=icon).props(f"color={color}")
    btn.tooltip(tooltip)
    for key, value in kwargs.items():
        btn.props(f"{key}={value}")
    return btn


def confirmation_dialog(
    title: str,
    message: str,
    on_confirm,
    on_cancel=None,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
) -> None:
    """Show a confirmation dialog.

    Args:
        title: Dialog title.
        message: Dialog message.
        on_confirm: Callback when confirmed.
        on_cancel: Optional callback when cancelled.
        confirm_text: Confirmation button text.
        cancel_text: Cancel button text.
    """
    with ui.dialog() as dialog, ui.card():
        ui.label(title).classes("text-lg font-bold mb-2")
        ui.label(message).classes("mb-4")

        with ui.row().classes("w-full justify-end gap-2"):

            def handle_cancel() -> None:
                dialog.close()
                if on_cancel:
                    on_cancel()

            def handle_confirm() -> None:
                dialog.close()
                on_confirm()

            ui.button(cancel_text, on_click=handle_cancel).props("flat")
            ui.button(confirm_text, on_click=handle_confirm).props("color=primary")

    dialog.open()


def empty_state(
    icon: str,
    title: str,
    description: str,
    action_text: str | None = None,
    on_action=None,
) -> None:
    """Create an empty state display.

    Args:
        icon: Material icon name.
        title: Empty state title.
        description: Empty state description.
        action_text: Optional action button text.
        on_action: Optional action callback.
    """
    with ui.column().classes("w-full items-center justify-center gap-4 py-12"):
        ui.icon(icon, size="xl").classes("text-gray-400")
        ui.label(title).classes("text-xl text-gray-500")
        ui.label(description).classes("text-gray-400")

        if action_text and on_action:
            ui.button(action_text, on_click=on_action).props("color=primary")


def loading_skeleton(width: str = "100%", height: str = "20px") -> None:
    """Create a loading skeleton placeholder.

    Args:
        width: Skeleton width.
        height: Skeleton height.
    """
    # Use sanitize=False since we're generating safe HTML ourselves
    ui.html(
        f'<div class="animate-pulse bg-gray-200 rounded" '
        f'style="width: {width}; height: {height};"></div>',
        sanitize=False,
    )


def section_header(title: str, icon: str | None = None, actions: list | None = None) -> None:
    """Create a consistent section header.

    Args:
        title: Section title.
        icon: Optional icon name.
        actions: Optional list of button configs [{"text": "...", "on_click": ...}, ...]
    """
    with ui.row().classes("w-full items-center mb-4"):
        if icon:
            ui.icon(icon, size="md").classes("text-gray-700")
        ui.label(title).classes("text-lg font-semibold")
        ui.space()

        if actions:
            for action in actions:
                ui.button(
                    action.get("text", ""),
                    on_click=action.get("on_click"),
                    icon=action.get("icon"),
                ).props(action.get("props", "flat"))


def status_badge(status: str, color: str | None = None, icon: str | None = None) -> ui.element:
    """Create a status badge with optional icon.

    Args:
        status: Status text.
        color: Optional badge color.
        icon: Optional icon name.

    Returns:
        Badge element.
    """
    badge = ui.badge(status.title())

    if color:
        badge.style(f"background-color: {color}22; color: {color};")

    if icon:
        with badge:
            ui.icon(icon, size="xs")

    return badge


def progress_bar(value: float, max_value: float = 100.0, show_text: bool = True) -> None:
    """Create a progress bar.

    Args:
        value: Current progress value.
        max_value: Maximum value.
        show_text: Whether to show percentage text.
    """
    percentage = min(100, (value / max_value) * 100) if max_value > 0 else 0

    with ui.column().classes("w-full gap-1"):
        if show_text:
            ui.label(f"{percentage:.0f}%").classes("text-sm text-gray-600")

        ui.linear_progress(value=percentage / 100.0).props("color=primary")


def info_card(
    title: str,
    content: str,
    icon: str | None = None,
    color: str = "blue",
    dismissible: bool = False,
) -> None:
    """Create an informational card.

    Args:
        title: Card title.
        content: Card content.
        icon: Optional icon name.
        color: Card accent color.
        dismissible: Whether card can be dismissed.
    """
    color_map = {
        "blue": "bg-blue-50 border-blue-200 text-blue-800",
        "green": "bg-green-50 border-green-200 text-green-800",
        "yellow": "bg-yellow-50 border-yellow-200 text-yellow-800",
        "red": "bg-red-50 border-red-200 text-red-800",
    }

    card_classes = f"w-full p-4 rounded-lg border {color_map.get(color, color_map['blue'])}"

    card = ui.card().classes(card_classes)
    with card:
        with ui.row().classes("w-full items-start gap-3"):
            if icon:
                ui.icon(icon, size="md")

            with ui.column().classes("flex-grow gap-1"):
                ui.label(title).classes("font-semibold")
                ui.label(content).classes("text-sm")

            if dismissible:
                ui.button(icon="close", on_click=lambda: card.delete()).props("flat dense")
