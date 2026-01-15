"""Component tests for the Write page.

Tests that would catch runtime NiceGUI errors like:
- Blocking LLM calls freezing the UI
- Background task context issues
"""

import pytest
from nicegui import ui
from nicegui.testing import User


@pytest.mark.component
class TestWritePageNoProject:
    """Tests for Write page without a project selected."""

    async def test_write_page_shows_no_project_message(self, user: User, test_services):
        """Write page shows 'no project selected' when no project is loaded."""
        from ui.pages.write import WritePage
        from ui.state import AppState

        @ui.page("/test-write-no-project")
        def test_page():
            state = AppState()  # No project set
            page = WritePage(state, test_services)
            page.build()

        await user.open("/test-write-no-project")
        await user.should_see("No Project Selected")


@pytest.mark.component
class TestWritePageWithProject:
    """Tests for Write page with a project loaded."""

    async def test_write_page_builds_with_project(self, user: User, test_app_state, test_services):
        """Write page builds successfully with a project."""
        from ui.pages.write import WritePage

        @ui.page("/test-write-with-project")
        def test_page():
            page = WritePage(test_app_state, test_services)
            page.build()

        await user.open("/test-write-with-project")
        # Should show the fundamentals tab content
        await user.should_see("Fundamentals")

    async def test_write_page_shows_interview_section(
        self, user: User, test_app_state, test_services
    ):
        """Write page shows interview section."""
        from ui.pages.write import WritePage

        @ui.page("/test-write-interview")
        def test_page():
            page = WritePage(test_app_state, test_services)
            page.build()

        await user.open("/test-write-interview")
        await user.should_see("Interview")

    async def test_write_page_captures_client_context(
        self, user: User, test_app_state, test_services
    ):
        """Write page captures client context for background tasks."""
        from ui.pages.write import WritePage

        page_ref = []

        @ui.page("/test-write-context")
        def test_page():
            page = WritePage(test_app_state, test_services)
            page.build()
            page_ref.append(page)

        await user.open("/test-write-context")

        # The client should be captured during build
        assert page_ref[0]._client is not None, "Client context should be captured"


@pytest.mark.component
class TestWritePageTabs:
    """Tests for Write page tab navigation."""

    async def test_write_page_has_fundamentals_tab(self, user: User, test_app_state, test_services):
        """Write page has Fundamentals tab."""
        from ui.pages.write import WritePage

        @ui.page("/test-write-fundamentals-tab")
        def test_page():
            page = WritePage(test_app_state, test_services)
            page.build()

        await user.open("/test-write-fundamentals-tab")
        await user.should_see("Fundamentals")

    async def test_write_page_has_live_writing_tab(self, user: User, test_app_state, test_services):
        """Write page has Live Writing tab."""
        from ui.pages.write import WritePage

        @ui.page("/test-write-writing-tab")
        def test_page():
            page = WritePage(test_app_state, test_services)
            page.build()

        await user.open("/test-write-writing-tab")
        await user.should_see("Live Writing")
