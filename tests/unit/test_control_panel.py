"""Tests for the control panel module."""

import json
import subprocess
import tempfile
import time
from datetime import timedelta
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
    def test_start_app_oserror(self, mock_port_check, mock_popen):
        """Should return None when subprocess raises OSError."""
        mock_popen.side_effect = OSError("Failed to start")

        manager = ProcessManager()
        pid = manager.start_app()

        assert pid is None

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

    @patch.object(ProcessManager, "_is_port_in_use", return_value=True)
    @patch.object(ProcessManager, "_kill_process_on_port", return_value=True)
    def test_stop_app_orphan_process(self, mock_kill, mock_port_check):
        """Should kill orphan process when port is in use but no tracked process."""
        manager = ProcessManager()
        result = manager.stop_app()

        assert result is True
        mock_kill.assert_called_once()

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_stop_app_already_terminated(self, mock_port_check, mock_popen):
        """Should return True when process is already terminated."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        # First poll during start_app check (skipped since _process is None initially)
        # First poll during stop_app: returns 0 (terminated)
        mock_process.poll.return_value = 0  # Process already terminated
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        # Now the process has "terminated" between start and stop
        result = manager.stop_app()

        assert result is True
        assert manager.get_pid() is None
        # terminate should NOT be called since process already terminated
        mock_process.terminate.assert_not_called()

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_stop_app_force_kill_on_timeout(self, mock_port_check, mock_popen):
        """Should force kill when graceful shutdown times out."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        result = manager.stop_app()

        assert result is True
        mock_process.kill.assert_called_once()

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    def test_stop_app_oserror(self, mock_port_check, mock_popen):
        """Should return False when stopping raises OSError."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.terminate.side_effect = OSError("Failed")
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        result = manager.stop_app()

        assert result is False

    @patch("scripts.control_panel.subprocess.Popen")
    @patch.object(ProcessManager, "_is_port_in_use", return_value=False)
    @patch("scripts.control_panel.sys.platform", "linux")
    def test_stop_app_sigterm_on_linux(self, mock_port_check, mock_popen):
        """Should use SIGTERM on non-Windows platforms."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager = ProcessManager()
        manager.start_app()

        with patch("scripts.control_panel.sys.platform", "linux"):
            result = manager.stop_app()

        assert result is True
        mock_process.send_signal.assert_called()

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

    @patch("scripts.control_panel.socket.socket")
    def test_is_port_in_use_true(self, mock_socket_class):
        """Should return True when port is in use."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        mock_socket_class.return_value = mock_socket

        manager = ProcessManager()
        result = manager._is_port_in_use(7860)

        assert result is True

    @patch("scripts.control_panel.socket.socket")
    def test_is_port_in_use_false(self, mock_socket_class):
        """Should return False when port is not in use."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 1  # Connection refused
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        mock_socket_class.return_value = mock_socket

        manager = ProcessManager()
        result = manager._is_port_in_use(7860)

        assert result is False

    @patch("scripts.control_panel.socket.socket")
    def test_is_port_in_use_oserror(self, mock_socket_class):
        """Should return False when socket raises OSError."""
        mock_socket_class.side_effect = OSError("Failed")

        manager = ProcessManager()
        result = manager._is_port_in_use(7860)

        assert result is False

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_kill_process_on_port_success(self, mock_run):
        """Should kill process on port successfully."""
        # First call: netstat
        netstat_result = MagicMock()
        netstat_result.stdout = "  TCP    0.0.0.0:7860    0.0.0.0:0    LISTENING    12345\n"
        # Second call: taskkill
        taskkill_result = MagicMock()

        mock_run.side_effect = [netstat_result, taskkill_result]

        manager = ProcessManager()
        result = manager._kill_process_on_port(7860)

        assert result is True
        assert mock_run.call_count == 2

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_kill_process_on_port_no_match(self, mock_run):
        """Should return False when no process matches the port."""
        netstat_result = MagicMock()
        netstat_result.stdout = "  TCP    0.0.0.0:8080    0.0.0.0:0    LISTENING    99999\n"
        mock_run.return_value = netstat_result

        manager = ProcessManager()
        result = manager._kill_process_on_port(7860)

        assert result is False

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_kill_process_on_port_timeout(self, mock_run):
        """Should return False when netstat times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("netstat", 10)

        manager = ProcessManager()
        result = manager._kill_process_on_port(7860)

        assert result is False

    @patch("scripts.control_panel.sys.platform", "linux")
    def test_kill_process_on_port_non_windows(self):
        """Should return False on non-Windows platforms."""
        manager = ProcessManager()
        result = manager._kill_process_on_port(7860)

        assert result is False


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

    def test_init_url_invalid_json(self):
        """Should use default URL when settings file has invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()

            with patch("scripts.control_panel.SETTINGS_FILE", Path(f.name)):
                manager = OllamaManager()
                assert manager.get_url() == "http://localhost:11434"

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

    @patch.object(OllamaManager, "check_health", return_value=True)
    def test_start_ollama_already_running(self, mock_health):
        """Should return True when Ollama is already running."""
        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is True

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_via_service(self, mock_popen, mock_run):
        """Should start Ollama via Windows service."""
        # Service check returns not running
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"

        mock_run.side_effect = [service_check, MagicMock()]  # query, start

        health_results = [False, False, True]  # First checks fail, last succeeds

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is True

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_not_responding(self, mock_popen, mock_run):
        """Should return False when service starts but doesn't respond."""
        # Service check returns not running
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"

        mock_run.side_effect = [service_check, MagicMock()]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is False

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_via_serve(self, mock_popen, mock_run):
        """Should start Ollama via 'ollama serve' when service fails."""
        # Service check fails (timeout)
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)

        health_results = [False, False, False, True]  # Last check succeeds

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is True
            mock_popen.assert_called()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_serve_not_responding(self, mock_popen, mock_run):
        """Should return False when 'ollama serve' starts but doesn't respond."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is False

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_not_found(self, mock_popen, mock_run):
        """Should return False when Ollama is not installed."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_popen.side_effect = FileNotFoundError("ollama not found")

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is False

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_oserror(self, mock_popen, mock_run):
        """Should return False when Popen raises OSError."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_popen.side_effect = OSError("Permission denied")

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is False

    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "linux")
    def test_start_ollama_linux(self, mock_popen):
        """Should start Ollama on Linux without service check."""
        health_results = [False, False, True]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            result = manager.start_ollama()

            assert result is True
            mock_popen.assert_called()


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

    def test_get_recent_lines_empty_file(self):
        """Should return empty list for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.flush()  # Create empty file

            watcher = LogWatcher(Path(f.name))
            lines = watcher.get_recent_lines()

            assert lines == []

    def test_get_recent_lines_oserror(self):
        """Should return empty list on OSError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("test\n")
            f.flush()
            temp_path = Path(f.name)

            watcher = LogWatcher(temp_path)

            with patch("builtins.open", side_effect=OSError("Permission denied")):
                lines = watcher.get_recent_lines()

            assert lines == []

    def test_get_recent_lines_large_file(self):
        """Should handle large files efficiently."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            # Write many lines
            for i in range(1000):
                f.write(f"2024-01-01 [INFO] Line {i}\n")
            f.flush()

            watcher = LogWatcher(Path(f.name))
            lines = watcher.get_recent_lines(n=10)

            assert len(lines) == 10
            assert "Line 999" in lines[-1]

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

    def test_clear_log_oserror(self):
        """Should return 0 on OSError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("test\n")
            f.flush()
            temp_path = Path(f.name)

            watcher = LogWatcher(temp_path)

            with patch("builtins.open", side_effect=OSError("Permission denied")):
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

    def test_get_file_size_oserror(self):
        """Should return 0 on OSError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("test\n")
            f.flush()
            temp_path = Path(f.name)

            watcher = LogWatcher(temp_path)

            with patch.object(Path, "stat", side_effect=OSError("Permission denied")):
                size = watcher.get_file_size()

            assert size == 0


class TestControlPanel:
    """Tests for ControlPanel class - mocking GUI components."""

    def _create_mock_panel(self):
        """Create a mock ControlPanel instance for testing."""
        from scripts.control_panel import ControlPanel

        with patch.object(ControlPanel, "__init__", return_value=None):
            panel = ControlPanel.__new__(ControlPanel)
            panel._process_manager = MagicMock()
            panel._ollama_manager = MagicMock()
            panel._log_watcher = MagicMock()
            panel._status_label = MagicMock()
            panel._sf_card = {
                "indicator": MagicMock(),
                "status_label": MagicMock(),
                "info_label": MagicMock(),
            }
            panel._ollama_card = {
                "indicator": MagicMock(),
                "status_label": MagicMock(),
                "info_label": MagicMock(),
            }
            panel._log_text = MagicMock()
            panel._log_text._textbox = MagicMock()
            panel._auto_scroll_var = MagicMock()
            panel.after = MagicMock()
            panel.destroy = MagicMock()
            return panel

    @patch("scripts.control_panel.ctk.CTk.__init__", return_value=None)
    @patch("scripts.control_panel.ctk.set_appearance_mode")
    @patch("scripts.control_panel.ctk.set_default_color_theme")
    @patch.object(ProcessManager, "__init__", return_value=None)
    @patch.object(OllamaManager, "__init__", return_value=None)
    @patch.object(LogWatcher, "__init__", return_value=None)
    def test_control_panel_init_creates_managers(
        self,
        mock_log_init,
        mock_ollama_init,
        mock_proc_init,
        mock_theme,
        mock_appearance,
        mock_ctk_init,
    ):
        """Should create manager instances on init."""
        from scripts.control_panel import ControlPanel

        # Mock all the UI creation methods
        with (
            patch.object(ControlPanel, "_create_header"),
            patch.object(ControlPanel, "_create_status_frame"),
            patch.object(ControlPanel, "_create_controls_frame"),
            patch.object(ControlPanel, "_create_log_frame"),
            patch.object(ControlPanel, "_create_status_bar"),
            patch.object(ControlPanel, "_poll_status"),
            patch.object(ControlPanel, "_poll_logs"),
            patch.object(ControlPanel, "title"),
            patch.object(ControlPanel, "geometry"),
            patch.object(ControlPanel, "minsize"),
            patch.object(ControlPanel, "protocol"),
        ):
            ControlPanel()  # Create instance to verify managers are initialized

            mock_proc_init.assert_called_once()
            mock_ollama_init.assert_called_once()
            mock_log_init.assert_called_once()

    def test_get_log_tag_error(self):
        """Should return 'error' tag for error lines."""
        panel = self._create_mock_panel()

        assert panel._get_log_tag("2024-01-01 [ERROR] Something failed") == "error"
        assert panel._get_log_tag("EXCEPTION occurred") == "error"
        assert panel._get_log_tag("Traceback (most recent call last)") == "error"

    def test_get_log_tag_warning(self):
        """Should return 'warning' tag for warning lines."""
        panel = self._create_mock_panel()

        assert panel._get_log_tag("2024-01-01 [WARNING] Be careful") == "warning"

    def test_get_log_tag_info(self):
        """Should return 'info' tag for info lines."""
        panel = self._create_mock_panel()

        assert panel._get_log_tag("2024-01-01 [INFO] Started") == "info"

    def test_get_log_tag_debug(self):
        """Should return 'debug' tag for other lines."""
        panel = self._create_mock_panel()

        assert panel._get_log_tag("Some random log line") == "debug"

    def test_set_status_message(self):
        """Should update status label with message and color."""
        panel = self._create_mock_panel()

        panel._set_status_message("Test message", "#ff0000")

        panel._status_label.configure.assert_called_once_with(
            text="> Test message", text_color="#ff0000"
        )

    def test_on_start_already_running(self):
        """Should show warning when app is already running."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True

        panel._on_start()

        panel._status_label.configure.assert_called_once()
        args = panel._status_label.configure.call_args
        assert "Already running" in args[1]["text"]

    def test_on_start_starts_app(self):
        """Should start app when not running."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = False

        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_start()

            mock_run.assert_called_once()

    def test_on_stop_not_running(self):
        """Should show warning when app is not running."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = False

        panel._on_stop()

        panel._status_label.configure.assert_called_once()
        args = panel._status_label.configure.call_args
        assert "Not running" in args[1]["text"]

    def test_on_stop_stops_app(self):
        """Should stop app when running."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True

        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_stop()

            mock_run.assert_called_once()

    def test_on_restart(self):
        """Should trigger restart in thread."""
        panel = self._create_mock_panel()

        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_restart()

            mock_run.assert_called_once()
            # Verify restart function is passed
            restart_func = mock_run.call_args[0][0]
            assert callable(restart_func)

    def test_on_restart_inner_function(self):
        """Should execute restart function correctly."""
        panel = self._create_mock_panel()
        panel._process_manager.stop_app.return_value = True
        panel._process_manager.start_app.return_value = 12345

        # Capture the restart function
        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_restart()
            restart_func = mock_run.call_args[0][0]

        # Execute the captured restart function
        with patch("time.sleep"):
            result = restart_func()

        panel._process_manager.stop_app.assert_called_once()
        panel._process_manager.start_app.assert_called_once()
        assert result == 12345

    @patch("scripts.control_panel.webbrowser.open")
    def test_on_browser(self, mock_open):
        """Should open browser to app URL."""
        panel = self._create_mock_panel()

        panel._on_browser()

        mock_open.assert_called_once_with("http://localhost:7860")

    def test_on_clear_logs_with_content(self):
        """Should clear logs and show bytes cleared."""
        panel = self._create_mock_panel()
        panel._log_watcher.clear_log.return_value = 2048

        panel._on_clear_logs()

        panel._log_watcher.clear_log.assert_called_once()
        args = panel._status_label.configure.call_args
        assert "Logs cleared" in args[1]["text"]

    def test_on_clear_logs_empty(self):
        """Should show message when no logs to clear."""
        panel = self._create_mock_panel()
        panel._log_watcher.clear_log.return_value = 0

        panel._on_clear_logs()

        args = panel._status_label.configure.call_args
        assert "No logs to clear" in args[1]["text"]

    def test_on_start_ollama_already_running(self):
        """Should show message when Ollama is already running."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = True
        panel._ollama_manager.get_url.return_value = "http://localhost:11434"

        panel._on_start_ollama()

        args = panel._status_label.configure.call_args
        assert "already running" in args[1]["text"]

    def test_on_start_ollama_starts(self):
        """Should start Ollama when not running."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = False

        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_start_ollama()

            mock_run.assert_called_once()

    def test_on_close(self):
        """Should destroy window on close."""
        panel = self._create_mock_panel()

        panel._on_close()

        panel.destroy.assert_called_once()

    def test_poll_status_running(self):
        """Should update status when app is running."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(seconds=65)
        panel._ollama_manager.check_health.return_value = True
        panel._ollama_manager.get_url.return_value = "http://localhost:11434"

        panel._poll_status()

        # Verify Story Factory status updated to running
        panel._sf_card["indicator"].configure.assert_called()
        panel._sf_card["status_label"].configure.assert_called_with(text="RUNNING")
        # Verify poll scheduled
        panel.after.assert_called()

    def test_poll_status_stopped(self):
        """Should update status when app is stopped."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = False
        panel._ollama_manager.check_health.return_value = False

        panel._poll_status()

        panel._sf_card["status_label"].configure.assert_called_with(text="STOPPED")
        panel._ollama_card["status_label"].configure.assert_called_with(text="STOPPED")

    def test_poll_status_uptime_hours(self):
        """Should display uptime in hours when > 1 hour."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(hours=2, minutes=30)
        panel._ollama_manager.check_health.return_value = False

        panel._poll_status()

        info_call = panel._sf_card["info_label"].configure.call_args
        assert "2h" in info_call[1]["text"]

    def test_poll_status_uptime_minutes(self):
        """Should display uptime in minutes when > 1 minute."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(minutes=5, seconds=30)
        panel._ollama_manager.check_health.return_value = False

        panel._poll_status()

        info_call = panel._sf_card["info_label"].configure.call_args
        assert "5m" in info_call[1]["text"]

    def test_poll_status_uptime_seconds(self):
        """Should display uptime in seconds when < 1 minute."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(seconds=45)
        panel._ollama_manager.check_health.return_value = False

        panel._poll_status()

        info_call = panel._sf_card["info_label"].configure.call_args
        assert "45s" in info_call[1]["text"]

    def test_poll_status_no_pid(self):
        """Should handle case when running but no PID available."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = None
        panel._process_manager.get_uptime.return_value = None
        panel._ollama_manager.check_health.return_value = False

        panel._poll_status()

        # Should still update to running
        panel._sf_card["status_label"].configure.assert_called_with(text="RUNNING")

    def test_poll_logs_with_content(self):
        """Should update log display with content."""
        panel = self._create_mock_panel()
        panel._log_watcher.get_recent_lines.return_value = ["Line 1", "Line 2"]
        panel._auto_scroll_var.get.return_value = True

        panel._poll_logs()

        panel._log_text.configure.assert_called()
        panel._log_text.delete.assert_called_once_with("1.0", "end")
        assert panel._log_text.insert.call_count == 2
        panel._log_text.see.assert_called_once_with("end")

    def test_poll_logs_empty(self):
        """Should show placeholder when no logs."""
        panel = self._create_mock_panel()
        panel._log_watcher.get_recent_lines.return_value = []
        panel._auto_scroll_var.get.return_value = True

        panel._poll_logs()

        panel._log_text.insert.assert_called_once()
        args = panel._log_text.insert.call_args
        assert "(no logs yet)" in args[0][1]

    def test_poll_logs_no_auto_scroll(self):
        """Should not auto-scroll when disabled."""
        panel = self._create_mock_panel()
        panel._log_watcher.get_recent_lines.return_value = ["Line 1"]
        panel._auto_scroll_var.get.return_value = False

        panel._poll_logs()

        panel._log_text.see.assert_not_called()

    def test_run_in_thread_success(self):
        """Should call success callback when function succeeds."""
        panel = self._create_mock_panel()

        def success_func():
            return True

        panel._run_in_thread(success_func, "Success!", "Failed")

        # Wait for thread to complete
        time.sleep(0.1)

        # Verify after() was scheduled
        panel.after.assert_called()

    def test_run_in_thread_failure(self):
        """Should call error callback when function returns False."""
        panel = self._create_mock_panel()

        def fail_func():
            return False

        panel._run_in_thread(fail_func, "Success!", "Failed")

        # Wait for thread to complete
        time.sleep(0.1)

        panel.after.assert_called()

    def test_run_in_thread_exception(self):
        """Should handle exception in background task."""
        panel = self._create_mock_panel()

        def error_func():
            raise ValueError("Test error")

        panel._run_in_thread(error_func, "Success!", "Failed")

        # Wait for thread to complete
        time.sleep(0.1)

        panel.after.assert_called()


class TestMainFunction:
    """Tests for the main() function."""

    @patch("scripts.control_panel.ControlPanel")
    @patch("scripts.control_panel.LOG_FILE")
    def test_main_creates_directories(self, mock_log_file, mock_control_panel):
        """Should create output directories if they don't exist."""
        from scripts.control_panel import main

        mock_parent = MagicMock()
        mock_log_file.parent = mock_parent

        mock_app = MagicMock()
        mock_control_panel.return_value = mock_app

        main()

        mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_app.mainloop.assert_called_once()
