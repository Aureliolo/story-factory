"""Chat component for interview interface."""

import logging
from collections.abc import Callable
from typing import Any

from nicegui import Client, context, ui
from nicegui.elements.button import Button
from nicegui.elements.column import Column
from nicegui.elements.row import Row
from nicegui.elements.textarea import Textarea

logger = logging.getLogger(__name__)


class ChatComponent:
    """Chat interface component for the interview system.

    Features:
    - Message history display
    - User input with send button
    - Typing indicator
    - Auto-scroll to latest message
    """

    def __init__(
        self,
        on_send: Callable[[str], Any] | None = None,
        placeholder: str = "Type your response...",
        disabled: bool = False,
    ):
        """Initialize chat component.

        Args:
            on_send: Callback when user sends a message.
            placeholder: Placeholder text for input.
            disabled: Whether input is disabled.
        """
        self.on_send = on_send
        self.placeholder = placeholder
        self.disabled = disabled

        self._message_container: Column | None = None
        self._input: Textarea | None = None
        self._send_button: Button | None = None
        self._typing_indicator: Row | None = None
        self._is_processing = False
        self._client: Client | None = None  # Store client for background task safety

    def build(self) -> None:
        """Build the chat UI."""
        # Capture client for background task safety
        try:
            self._client = context.client
        except RuntimeError:
            logger.warning("Could not capture client context during build")

        with ui.column().classes("w-full h-full"):
            # Message display area
            self._message_container = ui.column().classes(
                "w-full flex-grow overflow-auto p-4 bg-gray-50 dark:bg-gray-800 rounded-lg gap-3"
            )

            # Typing indicator (hidden by default)
            with ui.row().classes("items-center gap-2 px-4 py-2") as row:
                self._typing_indicator = row
                self._typing_indicator.set_visibility(False)
                ui.spinner(size="sm")
                ui.label("AI is thinking...").classes(
                    "text-gray-500 dark:text-gray-400 text-sm italic"
                )

            # Input area
            with ui.row().classes("w-full items-end gap-2 mt-2"):
                self._input = (
                    ui.textarea(
                        placeholder=self.placeholder,
                        on_change=self._on_input_change,
                    )
                    .classes("flex-grow")
                    .props("outlined autogrow rows=2")
                )

                self._send_button = (
                    ui.button(
                        icon="send",
                        on_click=self._handle_send,
                    )
                    .props("round color=primary")
                    .classes("mb-1")
                )

                if self.disabled:
                    self._input.disable()
                    self._send_button.disable()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat.

        Safe to call from background tasks.

        Args:
            role: 'user' or 'assistant'.
            content: Message content.
        """
        if not self._message_container:
            return

        # Use client context for background task safety
        def _do_add_message():
            if self._message_container is None:
                return
            with self._message_container:
                if role == "user":
                    self._create_user_message(content)
                else:
                    self._create_assistant_message(content)

            # Auto-scroll to bottom
            ui.run_javascript(
                "document.querySelector('.overflow-auto').scrollTop = "
                "document.querySelector('.overflow-auto').scrollHeight"
            )

        # If we have a client, use it to ensure proper context
        if self._client:
            with self._client:
                _do_add_message()
        else:
            # Fallback to direct execution (may fail in background tasks)
            _do_add_message()

    def _create_user_message(self, content: str) -> None:
        """Create a user message bubble."""
        with ui.row().classes("w-full justify-end"):
            with ui.card().classes("max-w-3/4 bg-blue-500 text-white"):
                ui.markdown(content).classes("text-white")

    def _create_assistant_message(self, content: str) -> None:
        """Create an assistant message bubble."""
        with ui.row().classes("w-full justify-start"):
            with ui.card().classes("max-w-3/4 bg-gray-100 dark:bg-gray-700"):
                ui.markdown(content).classes("prose-sm dark:prose-invert")

    def set_messages(self, messages: list[dict[str, str]]) -> None:
        """Set all messages at once (replacing existing).

        Args:
            messages: List of message dicts with 'role' and 'content'.
        """
        if not self._message_container:
            return

        self._message_container.clear()

        with self._message_container:
            for msg in messages:
                if msg["role"] == "user":
                    self._create_user_message(msg["content"])
                else:
                    self._create_assistant_message(msg["content"])

    def show_typing(self, show: bool = True) -> None:
        """Show or hide the typing indicator.

        Safe to call from background tasks.

        Args:
            show: Whether to show the indicator.
        """

        def _do_show_typing():
            if self._typing_indicator:
                self._typing_indicator.set_visibility(show)
            self._is_processing = show

            # Disable input while processing
            if self._input and self._send_button:
                if show:
                    self._input.disable()
                    self._send_button.disable()
                elif not self.disabled:
                    self._input.enable()
                    self._send_button.enable()

        # Use client context for background task safety
        if self._client:
            with self._client:
                _do_show_typing()
        else:
            _do_show_typing()

    def clear(self) -> None:
        """Clear all messages."""
        if self._message_container:
            self._message_container.clear()

    def set_disabled(self, disabled: bool) -> None:
        """Set disabled state.

        Args:
            disabled: Whether to disable input.
        """
        self.disabled = disabled
        if self._input and self._send_button:
            if disabled:
                self._input.disable()
                self._send_button.disable()
            else:
                self._input.enable()
                self._send_button.enable()

    def _on_input_change(self, e) -> None:
        """Handle input changes (for future features like char count)."""
        pass

    async def _handle_send(self) -> None:
        """Handle send button click."""
        if not self._input or self._is_processing:
            return

        content = self._input.value.strip()
        if not content:
            return

        # Clear input
        self._input.value = ""

        # Add user message
        self.add_message("user", content)

        # Call callback
        if self.on_send:
            await self.on_send(content)

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._input:
            self._input.run_method("focus")
