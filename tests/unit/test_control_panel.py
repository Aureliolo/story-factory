"""Tests for the control panel module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.control_panel import LogWatcher, OllamaManager, ProcessManager


class TestProcessManager:
    """Tests for ProcessManager class."""

    def test_init(self):
        """Should initialize with no running process."""
        manager = ProcessManager()
        assert manager.get_pid() is None
        assert manager.get_uptime() is None

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_start_app_success(self, mock_port_check, mock_popen):
        """Should start the application and return PID."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        pid = manager.start_app()

        assert pid == 12345
        mock_popen.assert_called_once()
        assert manager.is_running() is True
        assert manager.get_pid() == 12345

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=True)
    def test_start_app_port_in_use(self, mock_port_check, mock_popen):
        """Should return None when port is already in use."""
        manager = ProcessManager()
        pid = manager.start_app()

        assert pid is None
        mock_popen.assert_not_called()

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_start_app_already_running(self, mock_port_check, mock_popen):
        """Should return existing PID if already running."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        # Try to start again
        pid = manager.start_app()

        assert pid == 12345
        assert mock_popen.call_count == 1  # Only called once

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_stop_app_success(self, mock_port_check, mock_popen):
        """Should stop the running application."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.side_effect = [None, None, 0]  # Running, then terminated
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        result = manager.stop_app()

        assert result is True
        mock_process.terminate.assert_called_once()
        assert manager.get_pid() is None

    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_stop_app_not_running(self, mock_port_check):
        """Should return False when no process is running."""
        manager = ProcessManager()
        result = manager.stop_app()
        assert result is False

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_get_uptime(self, mock_port_check, mock_popen):
        """Should return uptime when process is running."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        uptime = manager.get_uptime()
        assert uptime is not None
        assert uptime.total_seconds() >= 0

    def test_is_running_false_initially(self):
        """Should return False when no process has been started."""
        with patch.object(ProcessManager, "_is_port_in_use", return_value=False):
            manager = ProcessManager()
            assert manager.is_running() is False


class TestOllamaManager:
    """Tests for OllamaManager class."""

    def test_init_default_url(self):
        """Should use default URL when settings file doesn't exist."""
        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            assert manager.get_url() == "http://localhost:11434"

    def test_init_url_from_settings(self):
        """Should load URL from settings file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"ollama_url": "http://custom:11435"}, f)
            f.flush()

            with patch("scripts.control_panel.SETTINGS_FILE", Path(f.name)):
                manager = OllamaManager()
                assert manager.get_url() == "http://custom:11435"

    @patch("urllib.request.urlopen")
    def test_check_health_success(self, mock_urlopen):
        """Should return True when Ollama responds with 200."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            assert manager.check_health() is True

    @patch("urllib.request.urlopen")
    def test_check_health_failure(self, mock_urlopen):
        """Should return False when Ollama is not responding."""
        mock_urlopen.side_effect = Exception("Connection refused")

        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            assert manager.check_health() is False


class TestLogWatcher:
    """Tests for LogWatcher class."""

    def test_init(self):
        """Should initialize with log path."""
        watcher = LogWatcher(Path("/tmp/test.log"))
        assert watcher._log_path == Path("/tmp/test.log")

    def test_get_recent_lines_nonexistent_file(self):
        """Should return empty list when log file doesn't exist."""
        watcher = LogWatcher(Path("/nonexistent/path/test.log"))
        lines = watcher.get_recent_lines()
        assert lines == []

    def test_get_recent_lines_with_content(self):
        """Should return recent lines from log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 [INFO] Line 1\n")
            f.write("2024-01-01 [INFO] Line 2\n")
            f.write("2024-01-01 [ERROR] Line 3\n")
            f.flush()

            watcher = LogWatcher(Path(f.name))
            lines = watcher.get_recent_lines(n=3)

            assert len(lines) == 3
            assert "Line 1" in lines[0]
            assert "ERROR" in lines[2]

    def test_get_recent_lines_truncates_long_lines(self):
        """Should truncate lines longer than 150 characters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            long_line = "A" * 200
            f.write(f"{long_line}\n")
            f.flush()

            watcher = LogWatcher(Path(f.name))
            lines = watcher.get_recent_lines(n=1)

            assert len(lines) == 1
            assert len(lines[0]) == 150
            assert lines[0].endswith("...")

    def test_clear_log_success(self):
        """Should clear log file and return bytes cleared."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("Some log content here\n")
            f.flush()
            temp_path = Path(f.name)

            watcher = LogWatcher(temp_path)
            bytes_cleared = watcher.clear_log()

            assert bytes_cleared > 0
            assert temp_path.stat().st_size == 0

    def test_clear_log_nonexistent(self):
        """Should return 0 when log file doesn't exist."""
        watcher = LogWatcher(Path("/nonexistent/path/test.log"))
        bytes_cleared = watcher.clear_log()
        assert bytes_cleared == 0

    def test_get_file_size_with_file(self):
        """Should return correct file size."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            content = "Test content"
            f.write(content)
            f.flush()

            watcher = LogWatcher(Path(f.name))
            size = watcher.get_file_size()

            assert size == len(content)

    def test_get_file_size_nonexistent(self):
        """Should return 0 when log file doesn't exist."""
        watcher = LogWatcher(Path("/nonexistent/path/test.log"))
        size = watcher.get_file_size()
        assert size == 0
