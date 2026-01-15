"""E2E tests for basic app loading and navigation.

These tests verify the app loads correctly in a browser.
Requires: playwright install (run: playwright install chromium)
"""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestAppLoads:
    """Test that the app loads without errors."""

    def test_homepage_loads(self, page: Page, base_url: str):
        """Homepage loads successfully."""
        page.goto(base_url)
        expect(page).to_have_title("Story Factory")

    def test_no_console_errors_on_load(self, page: Page, base_url: str):
        """No JavaScript errors on page load."""
        errors = []
        page.on("pageerror", lambda err: errors.append(err))

        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        assert len(errors) == 0, f"Console errors: {errors}"


class TestTabNavigation:
    """Test tab navigation works correctly."""

    def test_all_tabs_accessible(self, page: Page, base_url: str):
        """All main tabs are accessible."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        tabs = ["Write", "World", "Projects", "Settings", "Models"]

        for tab_name in tabs:
            tab = page.get_by_role("tab", name=tab_name)
            expect(tab).to_be_visible()

    def test_settings_tab_loads(self, page: Page, base_url: str):
        """Settings tab loads without errors."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        page.get_by_role("tab", name="Settings").click()
        # Playwright's expect has auto-waiting, no need for fixed timeout
        expect(page.get_by_text("Ollama Connection")).to_be_visible()

    def test_models_tab_loads(self, page: Page, base_url: str):
        """Models tab loads without errors."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        page.get_by_role("tab", name="Models").click()
        # Playwright's expect has auto-waiting, no need for fixed timeout
        expect(page.get_by_text("Model")).to_be_visible()


class TestSettingsPage:
    """Test settings page functionality."""

    def test_settings_form_visible(self, page: Page, base_url: str):
        """Settings form elements are visible."""
        page.goto(base_url)
        page.get_by_role("tab", name="Settings").click()
        # Playwright's expect has auto-waiting
        expect(page.get_by_label("Ollama URL")).to_be_visible()

    def test_model_dropdown_has_options(self, page: Page, base_url: str):
        """Model dropdown has selectable options."""
        page.goto(base_url)
        page.get_by_role("tab", name="Settings").click()
        # Playwright's expect has auto-waiting
        model_select = (
            page.locator("text=Default Model").locator("..").locator("select, [role='listbox']")
        )
        expect(model_select.first).to_be_visible()
