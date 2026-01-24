"""Component tests for chat interface.

Tests that would catch runtime NiceGUI errors like:
- Background task UI context loss
- Client context not captured during build
"""

import pytest
from nicegui import ui
from nicegui.testing import User


@pytest.mark.component
class TestChatComponent:
    """Tests for the ChatComponent class."""

    async def test_chat_component_builds(self, user: User):
        """Chat component builds without errors."""
        from src.ui.components.chat import ChatComponent

        @ui.page("/test-chat")
        def test_page():
            """Build test page with chat component."""
            chat = ChatComponent(placeholder="Test placeholder")
            chat.build()

        await user.open("/test-chat")
        # If we get here without exception, the build succeeded

    async def test_chat_add_message_user(self, user: User):
        """Chat component can add user messages."""
        from src.ui.components.chat import ChatComponent

        chat_ref = []

        @ui.page("/test-chat-user")
        def test_page():
            """Build test page with chat component for user messages."""
            chat = ChatComponent()
            chat.build()
            chat_ref.append(chat)

        await user.open("/test-chat-user")

        # Add a user message
        chat_ref[0].add_message("user", "Hello from test")

    async def test_chat_add_message_assistant(self, user: User):
        """Chat component can add assistant messages."""
        from src.ui.components.chat import ChatComponent

        chat_ref = []

        @ui.page("/test-chat-assistant")
        def test_page():
            """Build test page with chat component for assistant messages."""
            chat = ChatComponent()
            chat.build()
            chat_ref.append(chat)

        await user.open("/test-chat-assistant")

        # Add an assistant message
        chat_ref[0].add_message("assistant", "Hello from AI")

    async def test_chat_typing_indicator(self, user: User):
        """Chat component typing indicator works."""
        from src.ui.components.chat import ChatComponent

        chat_ref = []

        @ui.page("/test-chat-typing")
        def test_page():
            """Build test page with chat component for typing indicator."""
            chat = ChatComponent()
            chat.build()
            chat_ref.append(chat)

        await user.open("/test-chat-typing")

        chat = chat_ref[0]
        # Toggle typing indicator
        chat.show_typing(True)
        assert chat._is_processing is True

        chat.show_typing(False)
        assert chat._is_processing is False

    async def test_chat_set_messages(self, user: User):
        """Chat component can set multiple messages at once."""
        from src.ui.components.chat import ChatComponent

        chat_ref = []

        @ui.page("/test-chat-set")
        def test_page():
            """Build test page with chat component for setting messages."""
            chat = ChatComponent()
            chat.build()
            chat_ref.append(chat)

        await user.open("/test-chat-set")

        messages = [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "How can I help?"},
        ]

        chat_ref[0].set_messages(messages)

    async def test_chat_disabled_state(self, user: User):
        """Chat component disabled state works."""
        from src.ui.components.chat import ChatComponent

        chat_ref = []

        @ui.page("/test-chat-disabled")
        def test_page():
            """Build test page with disabled chat component."""
            chat = ChatComponent(disabled=True)
            chat.build()
            chat_ref.append(chat)

        await user.open("/test-chat-disabled")

        chat = chat_ref[0]
        assert chat.disabled is True

        # Enable it
        chat.set_disabled(False)
        assert chat.disabled is False

    async def test_chat_client_context_captured(self, user: User):
        """Chat component captures client context for background tasks."""
        from src.ui.components.chat import ChatComponent

        chat_ref = []

        @ui.page("/test-chat-context")
        def test_page():
            """Build test page with chat component for client context capture."""
            chat = ChatComponent()
            chat.build()
            chat_ref.append(chat)

        await user.open("/test-chat-context")

        # The client should be captured during build
        assert chat_ref[0]._client is not None, "Client context should be captured"
