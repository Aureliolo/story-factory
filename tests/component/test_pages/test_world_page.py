"""Component tests for the World page.

Tests that would catch runtime NiceGUI errors like:
- Interview required message not showing
- Entity browser rendering failures
"""

import pytest
from nicegui import ui
from nicegui.testing import User


@pytest.mark.component
class TestWorldPageNoProject:
    """Tests for World page without a project selected."""

    async def test_world_page_shows_no_project_message(self, user: User, test_services):
        """World page shows 'no project selected' when no project is loaded."""
        from src.ui.pages.world import WorldPage
        from src.ui.state import AppState

        @ui.page("/test-world-no-project")
        def test_page():
            """Build test page with world page and no project."""
            state = AppState()  # No project set
            page = WorldPage(state, test_services)
            page.build()

        await user.open("/test-world-no-project")
        await user.should_see("No Project Selected")


@pytest.mark.component
class TestWorldPageInterviewRequired:
    """Tests for World page when interview is not complete."""

    async def test_world_page_shows_interview_required(
        self, user: User, test_app_state, test_services
    ):
        """World page shows interview required message when interview not complete."""
        from src.ui.pages.world import WorldPage

        # Ensure interview is not complete
        test_app_state.interview_complete = False

        @ui.page("/test-world-interview-required")
        def test_page():
            """Build test page with world page requiring interview."""
            page = WorldPage(test_app_state, test_services)
            page.build()

        await user.open("/test-world-interview-required")
        await user.should_see("Complete the Interview First")

    async def test_world_page_has_go_to_interview_button(
        self, user: User, test_app_state, test_services
    ):
        """World page has button to navigate to interview."""
        from src.ui.pages.world import WorldPage

        # Ensure interview is not complete
        test_app_state.interview_complete = False

        @ui.page("/test-world-interview-button")
        def test_page():
            """Build test page with world page interview button."""
            page = WorldPage(test_app_state, test_services)
            page.build()

        await user.open("/test-world-interview-button")
        await user.should_see("Go to Interview")


@pytest.mark.component
class TestWorldPageWithData:
    """Tests for World page with complete interview and world data."""

    async def test_world_page_builds_with_data(self, user: User, test_app_state, test_services):
        """World page builds successfully with world data."""
        from src.ui.pages.world import WorldPage

        # Mark interview as complete
        test_app_state.interview_complete = True

        @ui.page("/test-world-with-data")
        def test_page():
            """Build test page with world page and world data."""
            page = WorldPage(test_app_state, test_services)
            page.build()

        await user.open("/test-world-with-data")
        # Should show entity browser elements
        await user.should_see("Entity Browser")

    async def test_world_page_entity_browser_builds_successfully(
        self, user: User, test_app_state, test_services
    ):
        """Entity browser builds without errors."""
        from src.ui.pages.world import WorldPage

        test_app_state.interview_complete = True

        @ui.page("/test-world-entity-browser")
        def test_page():
            """Build test page with world page entity browser."""
            page = WorldPage(test_app_state, test_services)
            page.build()

        await user.open("/test-world-entity-browser")
        await user.should_see("Entity Browser")
