"""Tests for models page operations (confirm_delete, refresh_all)."""

import logging
from unittest.mock import MagicMock, patch


class TestConfirmDelete:
    """Tests for confirm_delete dialog ordering."""

    def test_dialog_closes_before_refresh_on_success(self):
        """Dialog should close BEFORE refresh_all when deletion succeeds."""
        from src.ui.pages.models._operations import confirm_delete

        page = MagicMock()
        page.services.model.delete_model.return_value = True

        dialog = MagicMock()
        call_order = []

        dialog.close.side_effect = lambda: call_order.append("dialog.close")

        with (
            patch(
                "src.ui.pages.models._operations.refresh_all",
                side_effect=lambda p: call_order.append("refresh_all"),
            ),
            patch("src.ui.pages.models._operations.ui") as mock_ui,
        ):
            mock_ui.notify.side_effect = lambda *a, **kw: call_order.append("ui.notify")
            confirm_delete(page, dialog, "test-model:8b")

        assert "dialog.close" in call_order
        assert "refresh_all" in call_order
        # dialog.close must come before refresh_all
        assert call_order.index("dialog.close") < call_order.index("refresh_all")

    def test_dialog_closes_before_notify_on_failure(self):
        """Dialog should close BEFORE ui.notify when deletion fails."""
        from src.ui.pages.models._operations import confirm_delete

        page = MagicMock()
        page.services.model.delete_model.return_value = False

        dialog = MagicMock()
        call_order = []

        dialog.close.side_effect = lambda: call_order.append("dialog.close")

        with patch("src.ui.pages.models._operations.ui") as mock_ui:
            mock_ui.notify.side_effect = lambda *a, **kw: call_order.append("ui.notify")
            confirm_delete(page, dialog, "test-model:8b")

        assert "dialog.close" in call_order
        assert "ui.notify" in call_order
        assert call_order.index("dialog.close") < call_order.index("ui.notify")

    def test_successful_delete_notifies_and_refreshes(self):
        """Successful deletion should notify positively and refresh."""
        from src.ui.pages.models._operations import confirm_delete

        page = MagicMock()
        page.services.model.delete_model.return_value = True
        dialog = MagicMock()

        with (
            patch(
                "src.ui.pages.models._operations.refresh_all",
            ) as mock_refresh,
            patch("src.ui.pages.models._operations.ui") as mock_ui,
        ):
            confirm_delete(page, dialog, "test-model:8b")

        mock_ui.notify.assert_called_once_with("Deleted test-model:8b", type="positive")
        mock_refresh.assert_called_once_with(page)

    def test_failed_delete_notifies_negatively(self):
        """Failed deletion should notify negatively and not refresh."""
        from src.ui.pages.models._operations import confirm_delete

        page = MagicMock()
        page.services.model.delete_model.return_value = False
        dialog = MagicMock()

        with (
            patch(
                "src.ui.pages.models._operations.refresh_all",
            ) as mock_refresh,
            patch("src.ui.pages.models._operations.ui") as mock_ui,
        ):
            confirm_delete(page, dialog, "test-model:8b")

        mock_ui.notify.assert_called_once_with("Delete failed", type="negative")
        mock_refresh.assert_not_called()

    def test_delete_logs_success(self, caplog):
        """Successful deletion should log info messages."""
        from src.ui.pages.models._operations import confirm_delete

        page = MagicMock()
        page.services.model.delete_model.return_value = True
        dialog = MagicMock()

        with (
            caplog.at_level(logging.INFO),
            patch("src.ui.pages.models._operations.refresh_all"),
            patch("src.ui.pages.models._operations.ui"),
        ):
            confirm_delete(page, dialog, "test-model:8b")

        assert "Deleting model: test-model:8b" in caplog.text
        assert "deleted successfully" in caplog.text

    def test_delete_logs_failure(self, caplog):
        """Failed deletion should log error message."""
        from src.ui.pages.models._operations import confirm_delete

        page = MagicMock()
        page.services.model.delete_model.return_value = False
        dialog = MagicMock()

        with (
            caplog.at_level(logging.ERROR),
            patch(
                "src.ui.pages.models._operations.ui",
            ),
        ):
            confirm_delete(page, dialog, "test-model:8b")

        assert "Failed to delete model test-model:8b" in caplog.text


class TestRefreshAll:
    """Tests for refresh_all with defensive notify."""

    def test_refresh_all_with_notify(self):
        """refresh_all should notify when notify=True."""
        from src.ui.pages.models._operations import refresh_all

        page = MagicMock()

        with (
            patch("src.ui.pages.models._operations.refresh_installed_section"),
            patch("src.ui.pages.models._listing.refresh_model_list"),
            patch("src.ui.pages.models._listing.update_download_all_btn"),
            patch("src.ui.pages.models._operations.ui") as mock_ui,
        ):
            refresh_all(page, notify=True)

        mock_ui.notify.assert_called_once_with("Model lists refreshed", type="info")

    def test_refresh_all_without_notify(self):
        """refresh_all should skip notify when notify=False."""
        from src.ui.pages.models._operations import refresh_all

        page = MagicMock()

        with (
            patch("src.ui.pages.models._operations.refresh_installed_section"),
            patch("src.ui.pages.models._listing.refresh_model_list"),
            patch("src.ui.pages.models._listing.update_download_all_btn"),
            patch("src.ui.pages.models._operations.ui") as mock_ui,
        ):
            refresh_all(page, notify=False)

        mock_ui.notify.assert_not_called()

    def test_refresh_all_survives_runtime_error_on_notify(self, caplog):
        """refresh_all should not crash if ui.notify raises RuntimeError."""
        from src.ui.pages.models._operations import refresh_all

        page = MagicMock()

        with (
            caplog.at_level(logging.DEBUG),
            patch("src.ui.pages.models._operations.refresh_installed_section"),
            patch("src.ui.pages.models._listing.refresh_model_list"),
            patch("src.ui.pages.models._listing.update_download_all_btn"),
            patch("src.ui.pages.models._operations.ui") as mock_ui,
        ):
            mock_ui.notify.side_effect = RuntimeError("UI context unavailable")
            refresh_all(page, notify=True)

        assert "Skipped notification" in caplog.text
