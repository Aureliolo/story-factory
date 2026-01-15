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


class TestNavigation:
    """Test path-based navigation works correctly."""

    def test_all_nav_links_accessible(self, page: Page, base_url: str):
        """All main navigation links are accessible in the header."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        nav_items = ["Write", "World", "Projects", "Settings", "Models"]

        for nav_name in nav_items:
            # Navigation links are in the header
            link = page.get_by_role("link", name=nav_name)
            expect(link).to_be_visible()

    def test_settings_page_loads(self, page: Page, base_url: str):
        """Settings page loads via direct URL."""
        page.goto(f"{base_url}/settings")
        page.wait_for_load_state("networkidle")
        # Playwright's expect has auto-waiting - use exact match to avoid "Connection lost" conflict
        expect(page.get_by_text("Connection", exact=True)).to_be_visible()

    def test_models_page_loads(self, page: Page, base_url: str):
        """Models page loads via direct URL."""
        page.goto(f"{base_url}/models")
        page.wait_for_load_state("networkidle")
        # Playwright's expect has auto-waiting
        expect(page.get_by_text("Installed Models")).to_be_visible()


class TestSettingsPage:
    """Test settings page functionality."""

    def test_settings_form_visible(self, page: Page, base_url: str):
        """Settings form elements are visible."""
        page.goto(f"{base_url}/settings")
        page.wait_for_load_state("networkidle")
        # Playwright's expect has auto-waiting
        expect(page.get_by_label("Ollama URL")).to_be_visible()

    def test_model_selection_visible(self, page: Page, base_url: str):
        """Model selection section is visible on settings page."""
        page.goto(f"{base_url}/settings")
        page.wait_for_load_state("networkidle")
        # NiceGUI uses Quasar's q-select, not native select elements
        # Just verify the model selection section is present
        expect(page.get_by_text("Default Model")).to_be_visible()
