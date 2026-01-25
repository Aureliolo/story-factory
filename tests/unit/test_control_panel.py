"""Tests for the control panel module."""

import json
import subprocess
import sys
import tempfile
import urllib.error
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock customtkinter before importing control_panel (CI doesn't have GUI libraries)
# Create proper mock classes that can be subclassed
class MockCTk:
    """Mock CTk base class."""

    def __init__(self):
        pass

    def title(self, *args):
        """Set the window title."""
        pass

    def geometry(self, *args):
        """Set the window size and position."""
        pass

    def minsize(self, *args):
        """Set the minimum window dimensions."""
        pass

    def protocol(self, *args):
        """Register a callback for window manager protocol events."""
        pass

    def after(self, *args):
        """Schedule a callback to run after a delay."""
        pass

    def destroy(self):
        """Destroy the window and all its children."""
        pass

    def mainloop(self):
        """Start the Tkinter event loop."""
        pass


mock_ctk = MagicMock()
mock_ctk.CTk = MockCTk
mock_ctk.CTkFrame = MagicMock(return_value=MagicMock())
mock_ctk.CTkLabel = MagicMock(return_value=MagicMock())
mock_ctk.CTkButton = MagicMock(return_value=MagicMock())
mock_ctk.CTkTextbox = MagicMock(return_value=MagicMock())
mock_ctk.CTkCheckBox = MagicMock(return_value=MagicMock())
mock_ctk.CTkFont = MagicMock(return_value=MagicMock())
mock_ctk.BooleanVar = MagicMock(return_value=MagicMock())
mock_ctk.set_appearance_mode = MagicMock()
mock_ctk.set_default_color_theme = MagicMock()
sys.modules["customtkinter"] = mock_ctk

from scripts.control_panel import LogWatcher, OllamaManager, ProcessManager  # noqa: E402


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
    @patch("scripts.control_panel.sys.platform", "win32")
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
    @patch("scripts.control_panel.sys.platform", "win32")
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

    def test_init_url_missing_key(self):
        """Should use default URL when settings file exists but lacks ollama_url key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"other_setting": "value"}, f)
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
        mock_urlopen.side_effect = OSError("Connection refused")

        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            assert manager.check_health() is False

    @patch("urllib.request.urlopen")
    def test_check_health_urlerror(self, mock_urlopen):
        """Should return False on URLError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            assert manager.check_health() is False

    @patch("urllib.request.urlopen")
    def test_check_health_timeout(self, mock_urlopen):
        """Should return False on timeout."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")

        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            assert manager.check_health() is False

    @patch.object(OllamaManager, "check_health", return_value=True)
    def test_start_ollama_already_running(self, mock_health):
        """Should return success tuple when Ollama is already running."""
        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "already running" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_via_service(self, mock_popen, mock_run):
        """Should start Ollama via Windows service."""
        # Service check returns not running (service exists but stopped)
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"
        service_check.returncode = 0  # Service exists

        start_result = MagicMock()
        start_result.stderr = ""
        start_result.stdout = ""
        mock_run.side_effect = [service_check, start_result]  # query, start

        health_results = [False, False, True]  # First checks fail, last succeeds

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "service started" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_not_responding(self, mock_popen, mock_run):
        """Should return False when service starts but doesn't respond."""
        # Service check returns not running
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"
        service_check.returncode = 0

        start_result = MagicMock()
        start_result.stderr = ""
        start_result.stdout = ""
        mock_run.side_effect = [service_check, start_result]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is False
            assert "not responding" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_running_not_responding(self, mock_run):
        """Should return False when service is RUNNING but health check fails."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 4  RUNNING"
        service_check.returncode = 0
        mock_run.return_value = service_check

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is False
            assert "not responding" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_running_eventually_responds(self, mock_run):
        """Should return success when service is RUNNING and health check eventually succeeds."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 4  RUNNING"
        service_check.returncode = 0
        mock_run.return_value = service_check

        health_results = [False, False, True]  # Eventually succeeds

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "running" in message.lower()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_not_installed(self, mock_popen, mock_run):
        """Should fall back to desktop app when service doesn't exist."""
        service_check = MagicMock()
        service_check.stdout = "The specified service does not exist"
        service_check.returncode = 1
        mock_run.return_value = service_check

        health_results = [False, False, True]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, _ = manager.start_ollama()

            assert success is True
            # Should have fallen back to ollama serve
            mock_popen.assert_called()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_start_timeout(self, mock_popen, mock_run):
        """Should fall back to ollama serve when service start times out."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"
        service_check.returncode = 0

        # First call is query (success), second is start (timeout)
        mock_run.side_effect = [service_check, subprocess.TimeoutExpired("sc", 10)]

        health_results = [False, False, True]  # Eventually succeeds via serve

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            # Should have fallen back to ollama serve
            assert success is True
            assert "ollama serve" in message
            mock_popen.assert_called()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_service_oserror(self, mock_popen, mock_run):
        """Should fall back to ollama serve when service check fails with OSError."""
        mock_run.side_effect = OSError("Access denied")

        health_results = [False, False, True]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "ollama serve" in message
            mock_popen.assert_called()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.os.environ.get")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_via_desktop_app(self, mock_environ_get, mock_popen, mock_run):
        """Should start Ollama desktop app when available."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_environ_get.return_value = "C:\\Users\\Test\\AppData\\Local"

        health_results = [False, False, True]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
            patch("scripts.control_panel.Path") as mock_path_class,
        ):
            mock_path.exists.return_value = False
            mock_app_path = MagicMock()
            mock_app_path.exists.return_value = True
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_app_path
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "desktop app" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.os.environ.get")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_desktop_app_not_responding(self, mock_environ_get, mock_popen, mock_run):
        """Should return False when desktop app starts but doesn't respond."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_environ_get.return_value = "C:\\Users\\Test\\AppData\\Local"

        # Health check always fails
        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("time.sleep"),
            patch("scripts.control_panel.Path") as mock_path_class,
        ):
            mock_path.exists.return_value = False
            mock_app_path = MagicMock()
            mock_app_path.exists.return_value = True
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_app_path
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is False
            assert "not responding" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.os.environ.get")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_desktop_app_fails_fallback(self, mock_environ_get, mock_popen, mock_run):
        """Should fall back to ollama serve when desktop app fails."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_environ_get.return_value = "C:\\Users\\Test\\AppData\\Local"

        # First Popen fails (desktop app), second succeeds (ollama serve)
        mock_popen.side_effect = [OSError("Desktop app failed"), MagicMock()]

        health_results = [False, False, True]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
            patch("scripts.control_panel.Path") as mock_path_class,
        ):
            mock_path.exists.return_value = False
            mock_app_path = MagicMock()
            mock_app_path.exists.return_value = True
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_app_path
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "ollama serve" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_via_serve(self, mock_popen, mock_run):
        """Should start Ollama via 'ollama serve' when service fails and no desktop app."""
        # Service check fails (timeout)
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)

        health_results = [False, False, False, True]  # Last check succeeds

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is True
            assert "ollama serve" in message
            mock_popen.assert_called()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_serve_not_responding(self, mock_popen, mock_run):
        """Should return False when 'ollama serve' starts but doesn't respond."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("time.sleep"),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is False
            assert "not responding" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_not_found(self, mock_popen, mock_run):
        """Should return False when Ollama is not installed."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_popen.side_effect = FileNotFoundError("ollama not found")

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is False
            assert "not found" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.subprocess.Popen")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_start_ollama_oserror(self, mock_popen, mock_run):
        """Should return False when Popen raises OSError."""
        mock_run.side_effect = subprocess.TimeoutExpired("sc", 10)
        mock_popen.side_effect = OSError("Permission denied")

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_settings_path,
            patch.object(OllamaManager, "check_health", return_value=False),
            patch("scripts.control_panel.os.environ.get", return_value=""),
        ):
            mock_settings_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.start_ollama()

            assert success is False
            assert "Failed to start" in message

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
            success, message = manager.start_ollama()

            assert success is True
            assert "ollama serve" in message
            mock_popen.assert_called()

    @patch.object(OllamaManager, "check_health", return_value=False)
    def test_stop_ollama_not_running(self, mock_health):
        """Should return success when Ollama is not running."""
        with patch("scripts.control_panel.SETTINGS_FILE") as mock_path:
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is True
            assert "not running" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_via_service(self, mock_run):
        """Should stop Ollama via Windows service."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 4  RUNNING"

        stop_result = MagicMock()
        stop_result.stderr = ""
        mock_run.side_effect = [service_check, stop_result]

        health_results = [True, True, False]  # Last check shows stopped

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is True
            assert "service stopped" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_via_taskkill(self, mock_run):
        """Should stop Ollama via taskkill when service stop fails."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"  # Not running as service

        taskkill_result = MagicMock()
        taskkill_result.stdout = "SUCCESS: The process was terminated."
        taskkill_result.stderr = ""
        mock_run.side_effect = [service_check, taskkill_result, taskkill_result]

        health_results = [True, False]  # Initially running, then stopped

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is True
            assert "taskkill" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "linux")
    def test_stop_ollama_linux(self, mock_run):
        """Should stop Ollama on Linux via pkill."""
        health_results = [True, False]  # Initially running, then stopped

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is True
            assert "stopped" in message.lower()

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_service_timeout(self, mock_run):
        """Should fall back to taskkill when service stop times out."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 4  RUNNING"

        # First call query succeeds, second (stop) times out, then taskkill succeeds
        taskkill_result = MagicMock()
        taskkill_result.stdout = "SUCCESS: The process was terminated."
        taskkill_result.stderr = ""
        mock_run.side_effect = [
            service_check,
            subprocess.TimeoutExpired("sc stop", 10),
            taskkill_result,
            taskkill_result,
        ]

        health_results = [True, False]  # Eventually stopped

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is True
            assert "taskkill" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_service_still_running(self, mock_run):
        """Should return failure when service won't stop."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 4  RUNNING"

        stop_result = MagicMock()
        stop_result.stderr = "Access denied"
        mock_run.side_effect = [service_check, stop_result]

        # Health check keeps returning True (service won't stop)
        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=True),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is False
            assert "still running" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_taskkill_still_running(self, mock_run):
        """Should return failure when taskkill runs but Ollama still responds."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"  # Not running as service

        taskkill_result = MagicMock()
        taskkill_result.stdout = "SUCCESS: The process was terminated."
        taskkill_result.stderr = ""
        mock_run.side_effect = [service_check, taskkill_result, taskkill_result]

        # Health check keeps returning True (process respawns somehow)
        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=True),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is False
            assert "still responding" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_service_oserror_fallback(self, mock_run):
        """Should fall back to taskkill when service stop raises OSError."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 4  RUNNING"

        taskkill_result = MagicMock()
        taskkill_result.stdout = "SUCCESS: The process was terminated."
        taskkill_result.stderr = ""
        mock_run.side_effect = [
            service_check,
            OSError("Access denied"),
            taskkill_result,
            taskkill_result,
        ]

        health_results = [True, False]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", side_effect=health_results),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is True
            assert "taskkill" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "win32")
    def test_stop_ollama_taskkill_timeout(self, mock_run):
        """Should return failure when taskkill times out."""
        service_check = MagicMock()
        service_check.stdout = "STATE              : 1  STOPPED"

        mock_run.side_effect = [
            service_check,
            subprocess.TimeoutExpired("taskkill", 10),
        ]

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=True),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is False
            assert "Failed" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "linux")
    def test_stop_ollama_linux_still_running(self, mock_run):
        """Should return failure when pkill runs but Ollama still responds."""
        # Health check keeps returning True
        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=True),
            patch("time.sleep"),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is False
            assert "still responding" in message

    @patch("scripts.control_panel.subprocess.run")
    @patch("scripts.control_panel.sys.platform", "linux")
    def test_stop_ollama_linux_pkill_error(self, mock_run):
        """Should return failure when pkill fails."""
        mock_run.side_effect = OSError("pkill not found")

        with (
            patch("scripts.control_panel.SETTINGS_FILE") as mock_path,
            patch.object(OllamaManager, "check_health", return_value=True),
        ):
            mock_path.exists.return_value = False
            manager = OllamaManager()
            success, message = manager.stop_ollama()

            assert success is False
            assert "Failed" in message


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

    def test_get_recent_lines_returns_full_lines(self):
        """Should return full lines without truncation (truncation is UI concern)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            long_line = "A" * 200
            f.write(f"{long_line}\n")
            f.flush()

            watcher = LogWatcher(Path(f.name))
            lines = watcher.get_recent_lines(n=1)

            assert len(lines) == 1
            # Full line returned - truncation happens in UI layer
            assert len(lines[0]) == 200

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

    def test_get_recent_lines_chunk_boundary(self):
        """Should correctly stitch lines split across chunk boundaries."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            # Create a file where we need to read multiple chunks.
            # Chunk size is 8192 bytes. Each line is ~60 bytes.
            # To force multiple chunk reads, we need to request more lines than fit
            # in one chunk. 8192 / 60 = ~136 lines per chunk.
            # Request n=200 lines to force reading 2+ chunks.
            for i in range(300):
                f.write(f"2024-01-01 12:00:00 [INFO] Padding line number {i:04d} extra\n")

            # Write a unique line at the end
            f.write("2024-01-01 12:00:00 [INFO] MARKER_LINE_FINAL\n")
            f.flush()

            watcher = LogWatcher(Path(f.name))
            # Request 200 lines to force reading multiple chunks
            lines = watcher.get_recent_lines(n=200)

            # Verify lines are intact (stitching worked correctly)
            marker_found = any("MARKER_LINE_FINAL" in line for line in lines)
            assert marker_found, f"Marker line should be intact in: {lines}"
            # All lines should be complete (contain the full timestamp format)
            for line in lines:
                assert "2024-01-01" in line, f"Line appears truncated: {line}"

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
        import queue as queue_module

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
            # Thread-safe UI queue and cached Ollama status
            panel._ui_queue = queue_module.Queue()
            panel._ollama_healthy = False
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
            patch.object(ControlPanel, "_drain_ui_queue"),
            patch.object(ControlPanel, "_poll_status"),
            patch.object(ControlPanel, "_poll_ollama_status"),
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

    def test_on_restart_clear(self):
        """Should trigger restart and clear logs in thread."""
        panel = self._create_mock_panel()

        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_restart_clear()

            mock_run.assert_called_once()
            # Verify restart_and_clear function is passed
            restart_func = mock_run.call_args[0][0]
            assert callable(restart_func)

    def test_on_restart_clear_inner_function(self):
        """Should execute restart and clear function correctly."""
        panel = self._create_mock_panel()
        panel._process_manager.stop_app.return_value = True
        panel._process_manager.start_app.return_value = 12345
        panel._log_watcher.clear_log.return_value = 1024

        # Capture the restart_and_clear function
        with patch.object(panel, "_run_in_thread") as mock_run:
            panel._on_restart_clear()
            restart_func = mock_run.call_args[0][0]

        # Execute the captured restart_and_clear function
        with patch("time.sleep"):
            result = restart_func()

        panel._process_manager.stop_app.assert_called_once()
        panel._log_watcher.clear_log.assert_called_once()
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
        """Should show starting message and launch thread (health check now in thread)."""
        panel = self._create_mock_panel()

        with patch("threading.Thread") as mock_thread:
            panel._on_start_ollama()

            # Should show "Starting..." immediately
            args = panel._status_label.configure.call_args
            assert "Starting" in args[1]["text"]
            mock_thread.assert_called_once()

    def test_on_start_ollama_starts(self):
        """Should start Ollama when not running."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = False

        with patch("threading.Thread") as mock_thread:
            panel._on_start_ollama()

            mock_thread.assert_called_once()
            # Verify thread was started
            mock_thread.return_value.start.assert_called_once()

    def test_on_start_ollama_thread_already_running(self):
        """Should report already running when health check succeeds."""
        import queue as queue_module

        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = True
        panel._ollama_manager.get_url.return_value = "http://localhost:11434"
        panel._ui_queue = queue_module.Queue()

        with patch("threading.Thread") as mock_thread:
            panel._on_start_ollama()
            target_func = mock_thread.call_args[1]["target"]
            target_func()

            assert not panel._ui_queue.empty()
            callback = panel._ui_queue.get()
            assert callable(callback)

    def test_on_start_ollama_thread_success(self):
        """Should report success after starting Ollama."""
        import queue as queue_module

        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = False
        panel._ollama_manager.start_ollama.return_value = (True, "Started successfully")
        panel._ui_queue = queue_module.Queue()

        with patch("threading.Thread") as mock_thread:
            panel._on_start_ollama()
            target_func = mock_thread.call_args[1]["target"]
            target_func()

            assert not panel._ui_queue.empty()

    def test_on_start_ollama_thread_exception(self):
        """Should handle exceptions in start thread gracefully."""
        import queue as queue_module

        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.side_effect = Exception("Test error")
        panel._ui_queue = queue_module.Queue()

        # Call the actual method (not mocked thread)
        panel._on_start_ollama()

        # Get the thread target function and call it
        with patch("threading.Thread") as mock_thread:
            panel._on_start_ollama()
            # Get the target function from the Thread call
            target_func = mock_thread.call_args[1]["target"]

            # Run the target function - it should catch the exception
            target_func()

            # Check that an error message was queued
            assert not panel._ui_queue.empty()
            callback = panel._ui_queue.get()
            # The callback should set an error message
            assert callable(callback)

    def test_on_stop_ollama_thread_success(self):
        """Should report success after stopping Ollama."""
        import queue as queue_module

        panel = self._create_mock_panel()
        panel._ollama_manager.stop_ollama.return_value = (True, "Stopped successfully")
        panel._ui_queue = queue_module.Queue()

        with patch("threading.Thread") as mock_thread:
            panel._on_stop_ollama()
            target_func = mock_thread.call_args[1]["target"]
            target_func()

            assert not panel._ui_queue.empty()

    def test_on_stop_ollama_thread_exception(self):
        """Should handle exceptions in stop thread gracefully."""
        import queue as queue_module

        panel = self._create_mock_panel()
        panel._ollama_manager.stop_ollama.side_effect = Exception("Stop error")
        panel._ui_queue = queue_module.Queue()

        with patch("threading.Thread") as mock_thread:
            panel._on_stop_ollama()
            target_func = mock_thread.call_args[1]["target"]

            # Run the target function - it should catch the exception
            target_func()

            # Check that an error message was queued
            assert not panel._ui_queue.empty()

    def test_on_stop_ollama_attempts_stop(self):
        """Should attempt to stop Ollama even when health check fails."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = False

        with (
            patch.object(panel, "_set_status_message") as mock_status,
            patch("threading.Thread") as mock_thread,
        ):
            panel._on_stop_ollama()

            # Should show "Stopping..." and start a thread
            mock_status.assert_called_with("Stopping Ollama...", "#17a2b8")
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_on_stop_ollama_stops(self):
        """Should stop Ollama when running."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = True

        with patch("threading.Thread") as mock_thread:
            panel._on_stop_ollama()

            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

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
        panel._ollama_healthy = True  # Cached Ollama status
        panel._ollama_manager.get_url.return_value = "http://localhost:11434"

        panel._poll_status()

        # Verify Story Factory status updated to running
        panel._sf_card["indicator"].configure.assert_called()
        panel._sf_card["status_label"].configure.assert_called_with(text="RUNNING")
        # Verify Ollama status updated to running
        panel._ollama_card["status_label"].configure.assert_called_with(text="RUNNING")
        # Verify poll scheduled
        panel.after.assert_called()

    def test_poll_status_stopped(self):
        """Should update status when app is stopped."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = False
        panel._ollama_healthy = False  # Cached Ollama status

        panel._poll_status()

        panel._sf_card["status_label"].configure.assert_called_with(text="STOPPED")
        panel._ollama_card["status_label"].configure.assert_called_with(text="STOPPED")

    def test_poll_status_uptime_hours(self):
        """Should display uptime in hours when > 1 hour."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(hours=2, minutes=30)
        # _ollama_healthy defaults to False in _create_mock_panel

        panel._poll_status()

        info_call = panel._sf_card["info_label"].configure.call_args
        assert "2h" in info_call[1]["text"]

    def test_poll_status_uptime_minutes(self):
        """Should display uptime in minutes when > 1 minute."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(minutes=5, seconds=30)
        # _ollama_healthy defaults to False in _create_mock_panel

        panel._poll_status()

        info_call = panel._sf_card["info_label"].configure.call_args
        assert "5m" in info_call[1]["text"]

    def test_poll_status_uptime_seconds(self):
        """Should display uptime in seconds when < 1 minute."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = 12345
        panel._process_manager.get_uptime.return_value = timedelta(seconds=45)
        # _ollama_healthy defaults to False in _create_mock_panel

        panel._poll_status()

        info_call = panel._sf_card["info_label"].configure.call_args
        assert "45s" in info_call[1]["text"]

    def test_poll_status_no_pid(self):
        """Should handle case when running but no PID available."""
        panel = self._create_mock_panel()
        panel._process_manager.is_running.return_value = True
        panel._process_manager.get_pid.return_value = None
        panel._process_manager.get_uptime.return_value = None
        # _ollama_healthy defaults to False in _create_mock_panel

        panel._poll_status()

        # Should still update to running
        panel._sf_card["status_label"].configure.assert_called_with(text="RUNNING")

    def test_poll_ollama_status_healthy(self):
        """Should update cached status when Ollama is healthy."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = True

        panel._poll_ollama_status()

        # Wait for thread to queue the callback
        import time

        start = time.time()
        while panel._ui_queue.empty() and (time.time() - start) < 1:
            time.sleep(0.01)

        assert not panel._ui_queue.empty(), "Thread did not queue a callback"

        # Execute the queued callback
        callback = panel._ui_queue.get_nowait()
        callback()

        assert panel._ollama_healthy is True

    def test_poll_ollama_status_unhealthy(self):
        """Should update cached status when Ollama is not healthy."""
        panel = self._create_mock_panel()
        panel._ollama_manager.check_health.return_value = False
        panel._ollama_healthy = True  # Start with healthy state

        panel._poll_ollama_status()

        # Wait for thread to queue the callback
        import time

        start = time.time()
        while panel._ui_queue.empty() and (time.time() - start) < 1:
            time.sleep(0.01)

        assert not panel._ui_queue.empty(), "Thread did not queue a callback"

        # Execute the queued callback
        callback = panel._ui_queue.get_nowait()
        callback()

        assert panel._ollama_healthy is False

    def test_drain_ui_queue_executes_callbacks(self):
        """Should execute all queued callbacks."""
        panel = self._create_mock_panel()
        results = []

        # Queue some callbacks
        panel._ui_queue.put(lambda: results.append(1))
        panel._ui_queue.put(lambda: results.append(2))
        panel._ui_queue.put(lambda: results.append(3))

        panel._drain_ui_queue()

        assert results == [1, 2, 3]
        assert panel._ui_queue.empty()

    def test_drain_ui_queue_empty(self):
        """Should handle empty queue gracefully."""
        panel = self._create_mock_panel()

        # Should not raise
        panel._drain_ui_queue()

        assert panel._ui_queue.empty()

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

    def test_poll_logs_unchanged_skips_update(self):
        """Should skip update when log content hasn't changed."""
        panel = self._create_mock_panel()
        lines = ["Line 1", "Line 2"]
        panel._log_watcher.get_recent_lines.return_value = lines
        # Simulate that we already have the same content cached
        panel._last_log_lines = lines

        panel._poll_logs()

        # Should not call configure/delete since content is unchanged
        panel._log_text.configure.assert_not_called()
        panel._log_text.delete.assert_not_called()
        # But should still schedule next poll
        panel.after.assert_called_with(1000, panel._poll_logs)

    def test_run_in_thread_success(self):
        """Should queue success callback when function succeeds."""
        panel = self._create_mock_panel()

        def success_func():
            """Return True to simulate a successful operation."""
            return True

        panel._run_in_thread(success_func, "Success!", "Failed")

        # Wait for thread to complete by polling the queue
        import time

        start = time.time()
        while panel._ui_queue.empty() and (time.time() - start) < 1:
            time.sleep(0.01)

        assert not panel._ui_queue.empty(), "Thread did not queue a callback"

        # Execute the queued callback
        callback = panel._ui_queue.get_nowait()
        callback()

        # Verify success message was set
        panel._status_label.configure.assert_called()
        call_args = panel._status_label.configure.call_args
        assert "Success!" in call_args[1]["text"]

    def test_run_in_thread_failure_false(self):
        """Should queue error callback when function returns False."""
        panel = self._create_mock_panel()

        def fail_func():
            """Return False to simulate a failed operation."""
            return False

        panel._run_in_thread(fail_func, "Success!", "Failed")

        # Wait for thread to complete
        import time

        start = time.time()
        while panel._ui_queue.empty() and (time.time() - start) < 1:
            time.sleep(0.01)

        assert not panel._ui_queue.empty(), "Thread did not queue a callback"

        # Execute the queued callback
        callback = panel._ui_queue.get_nowait()
        callback()

        # Verify error message was set
        panel._status_label.configure.assert_called()
        call_args = panel._status_label.configure.call_args
        assert "Failed" in call_args[1]["text"]

    def test_run_in_thread_failure_none(self):
        """Should queue error callback when function returns None."""
        panel = self._create_mock_panel()

        def fail_func():
            """Return None to simulate a failed operation."""
            return None

        panel._run_in_thread(fail_func, "Success!", "Failed")

        # Wait for thread to complete
        import time

        start = time.time()
        while panel._ui_queue.empty() and (time.time() - start) < 1:
            time.sleep(0.01)

        assert not panel._ui_queue.empty(), "Thread did not queue a callback"

        # Execute the queued callback
        callback = panel._ui_queue.get_nowait()
        callback()

        # Verify error message was set
        panel._status_label.configure.assert_called()
        call_args = panel._status_label.configure.call_args
        assert "Failed" in call_args[1]["text"]

    def test_run_in_thread_exception(self):
        """Should queue error callback when exception occurs."""
        panel = self._create_mock_panel()

        def error_func():
            """Raise an exception to simulate an error during execution."""
            raise ValueError("Test error")

        panel._run_in_thread(error_func, "Success!", "Failed")

        # Wait for thread to complete
        import time

        start = time.time()
        while panel._ui_queue.empty() and (time.time() - start) < 1:
            time.sleep(0.01)

        assert not panel._ui_queue.empty(), "Thread did not queue a callback"

        # Execute the queued callback
        callback = panel._ui_queue.get_nowait()
        callback()

        # Verify error message was set
        panel._status_label.configure.assert_called()
        call_args = panel._status_label.configure.call_args
        assert "Error:" in call_args[1]["text"]
        assert "Test error" in call_args[1]["text"]


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
