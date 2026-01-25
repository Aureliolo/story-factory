"""Story Factory Control Panel - Native Python/CustomTkinter GUI.

A modern, flicker-free control panel for managing the Story Factory application.
Replaces the PowerShell-based control panel with a native desktop experience.
"""

import json
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import customtkinter as ctk

# Cross-platform subprocess flags (CREATE_NO_WINDOW only exists on Windows)
SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Configure logging for the control panel itself
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_FILE = PROJECT_ROOT / "output" / "logs" / "story_factory.log"
SETTINGS_FILE = PROJECT_ROOT / "src" / "settings.json"

# Application constants
APP_PORT = 7860
APP_URL = f"http://localhost:{APP_PORT}"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class ProcessManager:
    """Manages the Story Factory application process."""

    def __init__(self) -> None:
        """Initialize the process manager."""
        self._process: subprocess.Popen | None = None
        self._start_time: datetime | None = None
        self._lock = threading.Lock()
        logger.debug("ProcessManager initialized")

    def start_app(self) -> int | None:
        """Start the Story Factory application.

        Returns:
            The process PID if started successfully, None otherwise.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                logger.warning("Application already running (PID: %d)", self._process.pid)
                return self._process.pid

            # Check if port is already in use (orphan process)
            if self._is_port_in_use(APP_PORT):
                logger.warning("Port %d already in use - another instance may be running", APP_PORT)
                return None

            try:
                # Start the application with CREATE_NO_WINDOW on Windows
                creation_flags = 0
                if sys.platform == "win32":
                    creation_flags = SUBPROCESS_FLAGS

                self._process = subprocess.Popen(
                    [sys.executable, "main.py"],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
                self._start_time = datetime.now()
                logger.info("Started Story Factory (PID: %d)", self._process.pid)
                return self._process.pid

            except (OSError, subprocess.SubprocessError) as e:
                logger.error("Failed to start application: %s", e)
                return None

    def stop_app(self) -> bool:
        """Stop the Story Factory application.

        Uses graceful shutdown (SIGTERM) with fallback to force kill (SIGKILL).

        Returns:
            True if the application was stopped, False otherwise.
        """
        with self._lock:
            if self._process is None:
                logger.info("No tracked process to stop")
                # Check for orphan process on port
                if self._is_port_in_use(APP_PORT):
                    return self._kill_process_on_port(APP_PORT)
                return False

            if self._process.poll() is not None:
                logger.info("Process already terminated")
                self._process = None
                self._start_time = None
                return True

            pid = self._process.pid
            logger.info("Stopping application (PID: %d)...", pid)

            try:
                # Try graceful shutdown first
                if sys.platform == "win32":
                    self._process.terminate()
                else:
                    self._process.send_signal(signal.SIGTERM)

                # Wait up to 5 seconds for graceful shutdown
                try:
                    self._process.wait(timeout=5)
                    logger.info("Application stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    logger.warning("Graceful shutdown timed out, force killing...")
                    self._process.kill()
                    self._process.wait(timeout=2)
                    logger.info("Application force killed")

                self._process = None
                self._start_time = None
                return True

            except (OSError, subprocess.SubprocessError) as e:
                logger.error("Error stopping application: %s", e)
                return False

    def is_running(self) -> bool:
        """Check if the application is running.

        Returns:
            True if the application is running, False otherwise.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return True

            # Also check if port is in use (detects orphan processes)
            return self._is_port_in_use(APP_PORT)

    def get_pid(self) -> int | None:
        """Get the PID of the running application.

        Returns:
            The PID if running, None otherwise.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self._process.pid
            return None

    def get_uptime(self) -> timedelta | None:
        """Get the uptime of the running application.

        Returns:
            Timedelta of uptime if running, None otherwise.
        """
        with self._lock:
            if self._start_time is not None and self._process is not None:
                if self._process.poll() is None:
                    return datetime.now() - self._start_time
            return None

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use.

        Args:
            port: The port number to check.

        Returns:
            True if the port is in use, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", port))
                return result == 0
        except OSError:
            return False

    def _kill_process_on_port(self, port: int) -> bool:
        """Kill any process using the specified port (Windows only).

        On non-Windows platforms, this method returns False without action.

        Args:
            port: The port number.

        Returns:
            True if a process was killed, False otherwise.
        """
        if sys.platform == "win32":
            try:
                # Find PID using netstat
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=SUBPROCESS_FLAGS,
                )
                for line in result.stdout.split("\n"):
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        if parts:
                            pid = int(parts[-1])
                            subprocess.run(
                                ["taskkill", "/F", "/PID", str(pid)],
                                capture_output=True,
                                timeout=10,
                                creationflags=SUBPROCESS_FLAGS,
                                check=True,
                            )
                            logger.info("Killed orphan process (PID: %d)", pid)
                            return True
            except (subprocess.TimeoutExpired, ValueError, OSError) as e:
                logger.error("Failed to kill process on port %d: %s", port, e)
        return False


class OllamaManager:
    """Manages Ollama service health and startup."""

    def __init__(self) -> None:
        """Initialize the Ollama manager."""
        self._url = self._load_ollama_url()
        logger.debug("OllamaManager initialized with URL: %s", self._url)

    def _load_ollama_url(self) -> str:
        """Load the Ollama URL from settings.

        Returns:
            The Ollama URL from settings, or DEFAULT_OLLAMA_URL if not configured.
        """
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, encoding="utf-8") as f:
                    settings = json.load(f)
                    if "ollama_url" in settings:
                        return str(settings["ollama_url"])
                    logger.debug("ollama_url not in settings, using default")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load settings: %s", e)
        return DEFAULT_OLLAMA_URL

    def check_health(self) -> bool:
        """Check if Ollama is running and responding.

        Returns:
            True if Ollama is healthy, False otherwise.
        """
        try:
            url = f"{self._url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                status = int(response.status)
                logger.debug("Health check %s: status=%d", url, status)
                return status == 200
        except urllib.error.URLError as e:
            logger.debug("Health check failed (URLError): %s", e.reason)
            return False
        except TimeoutError:
            logger.debug("Health check failed (timeout)")
            return False
        except OSError as e:
            logger.debug("Health check failed (OSError): %s", e)
            return False

    def start_ollama(self) -> tuple[bool, str]:
        """Start the Ollama service.

        Returns:
            Tuple of (success, message) with details about what happened.
        """
        logger.info("Checking if Ollama is already running...")
        if self.check_health():
            msg = f"Ollama already running at {self._url}"
            logger.info(msg)
            return True, msg

        # Try Windows service first
        service_exists = False
        if sys.platform == "win32":
            try:
                logger.info("Checking Windows service status...")
                result = subprocess.run(
                    ["sc", "query", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=SUBPROCESS_FLAGS,
                )
                logger.debug(
                    "sc query result: %s", result.stdout[:200] if result.stdout else "empty"
                )

                if "does not exist" in result.stdout or result.returncode != 0:
                    logger.info("Ollama Windows service not installed")
                    service_exists = False
                elif "RUNNING" in result.stdout:
                    # Service running but health check failed - wait and retry
                    logger.info("Service shows RUNNING but health check failed, waiting...")
                    for _ in range(5):
                        time.sleep(1)
                        if self.check_health():
                            msg = f"Ollama service is running at {self._url}"
                            logger.info(msg)
                            return True, msg
                    msg = "Ollama service is running but not responding to API requests"
                    logger.warning(msg)
                    return False, msg
                else:
                    # Service exists but not running - try to start it
                    logger.info("Attempting to start Ollama Windows service...")
                    try:
                        start_result = subprocess.run(
                            ["sc", "start", "ollama"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            creationflags=SUBPROCESS_FLAGS,
                        )
                        logger.debug(
                            "sc start result: %s",
                            start_result.stdout[:200] if start_result.stdout else "empty",
                        )

                        # Give it time to start
                        for i in range(5):
                            time.sleep(1)
                            logger.debug("Waiting for service... attempt %d", i + 1)
                            if self.check_health():
                                msg = f"Ollama service started at {self._url}"
                                logger.info(msg)
                                return True, msg

                        # Service started but not responding
                        stderr = start_result.stderr.strip() if start_result.stderr else ""
                        stdout = start_result.stdout.strip() if start_result.stdout else ""
                        details = stderr or stdout or "no details"
                        msg = f"Ollama service start attempted but not responding. {details}"
                        logger.warning(msg)
                        return False, msg

                    except subprocess.TimeoutExpired:
                        logger.warning("Windows service start timed out, trying other methods...")
                        # Fall through to try desktop app / ollama serve
                    except OSError as e:
                        logger.warning(
                            "Windows service start failed: %s, trying other methods...", e
                        )
                        # Fall through to try desktop app / ollama serve

            except subprocess.TimeoutExpired:
                logger.warning("Windows service check timed out after 10s")
            except OSError as e:
                logger.debug("Windows service check failed: %s", e)

        # Try starting Ollama (if service doesn't exist or we're not on Windows)
        if not service_exists:
            creation_flags = SUBPROCESS_FLAGS if sys.platform == "win32" else 0

            # On Windows, try starting the desktop app first (has tray icon)
            if sys.platform == "win32":
                # The desktop app is "ollama app.exe" (with space), not "Ollama.exe"
                ollama_app = (
                    Path(os.environ.get("LOCALAPPDATA", ""))
                    / "Programs"
                    / "Ollama"
                    / "ollama app.exe"
                )
                if ollama_app.exists():
                    try:
                        logger.info("Starting Ollama desktop app: %s", ollama_app)
                        subprocess.Popen(
                            [str(ollama_app)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=creation_flags,
                        )

                        for i in range(8):  # Desktop app may take longer to start
                            time.sleep(1)
                            logger.debug("Waiting for Ollama desktop app... attempt %d", i + 1)
                            if self.check_health():
                                msg = f"Ollama desktop app started at {self._url}"
                                logger.info(msg)
                                return True, msg

                        msg = "Started Ollama desktop app but not responding after 8 seconds"
                        logger.warning(msg)
                        return False, msg

                    except (OSError, subprocess.SubprocessError) as e:
                        logger.warning("Failed to start Ollama desktop app: %s", e)
                        # Fall through to try ollama serve
                else:
                    logger.debug("Ollama desktop app not found at %s", ollama_app)

            # Fall back to ollama serve (CLI)
            try:
                logger.info("Attempting to start Ollama via 'ollama serve'...")
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )

                for i in range(5):
                    time.sleep(1)
                    logger.debug("Waiting for ollama serve... attempt %d", i + 1)
                    if self.check_health():
                        msg = f"Ollama started via 'ollama serve' at {self._url}"
                        logger.info(msg)
                        return True, msg

                msg = "Ran 'ollama serve' but Ollama not responding after 5 seconds"
                logger.warning(msg)
                return False, msg

            except FileNotFoundError:
                msg = "Ollama not found in PATH. Install from https://ollama.ai"
                logger.error(msg)
                return False, msg
            except (OSError, subprocess.SubprocessError) as e:
                msg = f"Failed to start Ollama: {e}"
                logger.error(msg)
                return False, msg

        # Should not reach here, but just in case
        msg = "Failed to start Ollama - unknown error"  # pragma: no cover
        logger.error(msg)  # pragma: no cover
        return False, msg  # pragma: no cover

    def stop_ollama(self) -> tuple[bool, str]:
        """Stop the Ollama service.

        Returns:
            Tuple of (success, message) with details about what happened.
        """
        # Don't rely solely on health check - try to stop anyway
        # (service might be running but API not responding)
        initially_healthy = self.check_health()
        logger.info(
            "Attempting to stop Ollama (currently %s)...",
            "responding" if initially_healthy else "not responding",
        )

        if sys.platform == "win32":
            # Try stopping Windows service first
            try:
                result = subprocess.run(
                    ["sc", "query", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=SUBPROCESS_FLAGS,
                )
                if "RUNNING" in result.stdout:
                    logger.info("Attempting to stop Ollama Windows service...")
                    stop_result = subprocess.run(
                        ["sc", "stop", "ollama"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        creationflags=SUBPROCESS_FLAGS,
                    )
                    # Give it time to stop
                    for _ in range(5):
                        time.sleep(1)
                        if not self.check_health():
                            msg = "Ollama service stopped"
                            logger.info(msg)
                            return True, msg
                    stderr = stop_result.stderr.strip() if stop_result.stderr else ""
                    msg = f"Ollama service stop requested but still running. {stderr}".strip()
                    logger.warning(msg)
                    return False, msg
            except subprocess.TimeoutExpired:
                logger.debug("Windows service stop timed out, trying taskkill...")
            except OSError as e:
                logger.debug("Windows service stop failed: %s", e)

            # Try killing ollama processes directly
            try:
                logger.info("Attempting to stop Ollama via taskkill...")
                any_killed = False

                # Kill all Ollama-related processes
                for proc_name in ["ollama app.exe", "ollama.exe", "ollama_llama_server.exe"]:
                    result = subprocess.run(
                        ["taskkill", "/F", "/IM", proc_name],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        creationflags=SUBPROCESS_FLAGS,
                    )
                    output = result.stdout.strip() if result.stdout else result.stderr.strip()
                    logger.debug("taskkill %s: %s", proc_name, output)
                    if "SUCCESS" in (result.stdout or ""):
                        any_killed = True

                time.sleep(1)
                if not self.check_health():
                    if any_killed:
                        msg = "Ollama stopped via taskkill"
                    else:
                        msg = "Ollama is not running (no process found)"
                    logger.info(msg)
                    return True, msg
                msg = "Taskkill ran but Ollama still responding"
                logger.warning(msg)
                return False, msg
            except (subprocess.TimeoutExpired, OSError) as e:
                msg = f"Failed to stop Ollama: {e}"
                logger.error(msg)
                return False, msg
        else:
            # Unix: try pkill
            try:
                logger.info("Attempting to stop Ollama via pkill...")
                subprocess.run(["pkill", "-f", "ollama"], timeout=10)
                time.sleep(1)
                if not self.check_health():
                    msg = "Ollama stopped"
                    logger.info(msg)
                    return True, msg
                msg = "pkill ran but Ollama still responding"
                logger.warning(msg)
                return False, msg
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                msg = f"Failed to stop Ollama: {e}"
                logger.error(msg)
                return False, msg

    def get_url(self) -> str:
        """Get the Ollama API URL.

        Returns:
            The Ollama URL.
        """
        return self._url


class LogWatcher:
    """Monitors and manages the Story Factory log file."""

    def __init__(self, log_path: Path = LOG_FILE) -> None:
        """Initialize the log watcher.

        Args:
            log_path: Path to the log file.
        """
        self._log_path = log_path
        logger.debug("LogWatcher initialized for: %s", log_path)

    def get_recent_lines(self, n: int = 20) -> list[str]:
        """Get the most recent lines from the log file.

        Args:
            n: Number of lines to retrieve.

        Returns:
            List of recent log lines.
        """
        if not self._log_path.exists():
            return []

        try:
            with open(self._log_path, "rb") as f:
                # Seek from end for efficiency
                f.seek(0, 2)
                file_size = f.tell()

                if file_size == 0:
                    return []

                # Read in chunks from the end
                chunk_size = min(8192, file_size)
                lines: list[str] = []
                current_pos = file_size

                while len(lines) < n + 1 and current_pos > 0:
                    # Move back by chunk_size
                    seek_pos = max(0, current_pos - chunk_size)
                    bytes_to_read = current_pos - seek_pos
                    f.seek(seek_pos)
                    chunk = f.read(bytes_to_read)
                    current_pos = seek_pos

                    # Decode and split into lines (errors="replace" handles invalid bytes)
                    decoded = chunk.decode("utf-8", errors="replace")

                    new_lines = decoded.split("\n")
                    if lines and current_pos > 0:
                        # Stitch together a line that was split across a chunk boundary
                        new_lines[-1] += lines.pop(0)
                    lines = new_lines + lines

                    if seek_pos == 0:
                        break

                # Filter empty lines and take last n
                non_empty_lines = [stripped for ln in lines if (stripped := ln.strip())]
                return non_empty_lines[-n:]

        except OSError as e:
            logger.warning("Failed to read log file: %s", e)
            return []

    def clear_log(self) -> int:
        """Clear the log file.

        Returns:
            Number of bytes cleared.
        """
        if not self._log_path.exists():
            return 0

        try:
            size = self._log_path.stat().st_size
            with open(self._log_path, "w", encoding="utf-8"):
                pass
            logger.info("Cleared log file (%d bytes)", size)
            return size
        except OSError as e:
            logger.error("Failed to clear log file: %s", e)
            return 0

    def get_file_size(self) -> int:
        """Get the current log file size.

        Returns:
            File size in bytes.
        """
        if not self._log_path.exists():
            return 0
        try:
            return self._log_path.stat().st_size
        except OSError:
            return 0


class ControlPanel(ctk.CTk):
    """Main control panel window using CustomTkinter."""

    def __init__(self) -> None:
        """Initialize the control panel window."""
        super().__init__()

        # Initialize managers
        self._process_manager = ProcessManager()
        self._ollama_manager = OllamaManager()
        self._log_watcher = LogWatcher()

        # Thread-safe UI dispatch queue
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()

        # Window setup
        self.title("Story Factory Control Panel")
        self.geometry("600x700")
        self.minsize(500, 500)

        # Set dark mode
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Configure grid layout for dynamic resizing
        # Row 0: Header (fixed)
        # Row 1: Status (fixed)
        # Row 2: Controls (fixed)
        # Row 3: Logs (expands to fill available space)
        # Row 4: Status bar (fixed)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Header - fixed
        self.grid_rowconfigure(1, weight=0)  # Status - fixed
        self.grid_rowconfigure(2, weight=0)  # Controls - fixed
        self.grid_rowconfigure(3, weight=1)  # Logs - expands
        self.grid_rowconfigure(4, weight=0)  # Status bar - fixed

        # Build the UI
        self._create_header()
        self._create_status_frame()
        self._create_controls_frame()
        self._create_log_frame()
        self._create_status_bar()

        # Cached Ollama status (updated by background thread)
        self._ollama_healthy = False

        # Start thread-safe UI dispatch and polling
        self._drain_ui_queue()
        self._poll_status()
        self._poll_ollama_status()
        self._poll_logs()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        logger.info("Control panel initialized")

    def _create_header(self) -> None:  # pragma: no cover
        """Create the header section (UI widget creation only)."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        title_label = ctk.CTkLabel(
            header_frame,
            text="STORY FACTORY CONTROL PANEL",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label.pack()

    def _create_status_frame(self) -> None:  # pragma: no cover
        """Create the status indicators section (UI widget creation only)."""
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        status_label = ctk.CTkLabel(
            status_frame,
            text="Status",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        status_label.pack(anchor="w", padx=15, pady=(10, 5))

        # Container for status cards
        cards_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=(0, 10))
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)

        # Story Factory status card
        self._sf_card = self._create_status_card(cards_frame, "Story Factory", "STOPPED", "red", 0)

        # Ollama status card
        self._ollama_card = self._create_status_card(cards_frame, "Ollama", "STOPPED", "red", 1)

    def _create_status_card(  # pragma: no cover
        self,
        parent: ctk.CTkFrame,
        title: str,
        status: str,
        color: str,
        column: int,
    ) -> dict:
        """Create a status card widget (UI widget creation only).

        Args:
            parent: Parent frame.
            title: Card title.
            status: Initial status text.
            color: Status color.
            column: Grid column.

        Returns:
            Dictionary with card widget references.
        """
        card = ctk.CTkFrame(parent)
        card.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        title_label.pack(anchor="w", padx=10, pady=(10, 5))

        status_container = ctk.CTkFrame(card, fg_color="transparent")
        status_container.pack(anchor="w", padx=10, pady=2)

        indicator = ctk.CTkLabel(
            status_container,
            text="\u25cf",  # Bullet character
            font=ctk.CTkFont(size=14),
            text_color=color,
        )
        indicator.pack(side="left")

        status_label = ctk.CTkLabel(
            status_container,
            text=status,
            font=ctk.CTkFont(size=12),
        )
        status_label.pack(side="left", padx=(5, 0))

        info_label = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        info_label.pack(anchor="w", padx=10, pady=(0, 10))

        return {
            "card": card,
            "indicator": indicator,
            "status_label": status_label,
            "info_label": info_label,
        }

    def _create_controls_frame(self) -> None:  # pragma: no cover
        """Create the control buttons section (UI widget creation only)."""
        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

        controls_label = ctk.CTkLabel(
            controls_frame,
            text="Controls",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        controls_label.pack(anchor="w", padx=15, pady=(10, 10))

        # First row of buttons
        row1 = ctk.CTkFrame(controls_frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 5))

        self._start_btn = ctk.CTkButton(
            row1,
            text="Start",
            command=self._on_start,
            width=100,
            fg_color="#28a745",
            hover_color="#218838",
        )
        self._start_btn.pack(side="left", padx=5)

        self._stop_btn = ctk.CTkButton(
            row1,
            text="Stop",
            command=self._on_stop,
            width=100,
            fg_color="#dc3545",
            hover_color="#c82333",
        )
        self._stop_btn.pack(side="left", padx=5)

        self._restart_btn = ctk.CTkButton(
            row1,
            text="Restart",
            command=self._on_restart,
            width=100,
            fg_color="#fd7e14",
            hover_color="#e76a00",
        )
        self._restart_btn.pack(side="left", padx=5)

        self._restart_clear_btn = ctk.CTkButton(
            row1,
            text="Restart & Clear",
            command=self._on_restart_clear,
            width=120,
            fg_color="#fd7e14",
            hover_color="#e76a00",
        )
        self._restart_clear_btn.pack(side="left", padx=5)

        self._browser_btn = ctk.CTkButton(
            row1,
            text="Browser",
            command=self._on_browser,
            width=100,
        )
        self._browser_btn.pack(side="left", padx=5)

        # Second row of buttons
        row2 = ctk.CTkFrame(controls_frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 10))

        self._clear_btn = ctk.CTkButton(
            row2,
            text="Clear Logs",
            command=self._on_clear_logs,
            width=100,
            fg_color="#6c757d",
            hover_color="#5a6268",
        )
        self._clear_btn.pack(side="left", padx=5)

        self._ollama_start_btn = ctk.CTkButton(
            row2,
            text="Start Ollama",
            command=self._on_start_ollama,
            width=110,
            fg_color="#17a2b8",
            hover_color="#138496",
        )
        self._ollama_start_btn.pack(side="left", padx=5)

        self._ollama_stop_btn = ctk.CTkButton(
            row2,
            text="Stop Ollama",
            command=self._on_stop_ollama,
            width=110,
            fg_color="#dc3545",
            hover_color="#c82333",
        )
        self._ollama_stop_btn.pack(side="left", padx=5)

    def _create_log_frame(self) -> None:  # pragma: no cover
        """Create the log viewer section (UI widget creation only)."""
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=10)

        # Configure log_frame to expand its contents
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)  # Row 1 is the textbox

        # Header with title and auto-scroll toggle
        header = ctk.CTkFrame(log_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        log_label = ctk.CTkLabel(
            header,
            text="Logs",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        log_label.pack(side="left")

        self._auto_scroll_var = ctk.BooleanVar(value=True)
        auto_scroll_check = ctk.CTkCheckBox(
            header,
            text="Auto-scroll",
            variable=self._auto_scroll_var,
            onvalue=True,
            offvalue=False,
            width=100,
        )
        auto_scroll_check.pack(side="right")

        # Log text area - expands to fill all available space
        self._log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            state="disabled",
        )
        self._log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Configure log colors
        self._log_text._textbox.tag_config("error", foreground="#dc3545")
        self._log_text._textbox.tag_config("warning", foreground="#ffc107")
        self._log_text._textbox.tag_config("info", foreground="#6c757d")
        self._log_text._textbox.tag_config("debug", foreground="#495057")

    def _create_status_bar(self) -> None:  # pragma: no cover
        """Create the status bar at the bottom (UI widget creation only)."""
        status_bar = ctk.CTkFrame(self, height=30)
        status_bar.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 10))

        self._status_label = ctk.CTkLabel(
            status_bar,
            text="> Ready",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self._status_label.pack(fill="x", padx=10, pady=5)

    def _set_status_message(self, message: str, color: str = "gray") -> None:
        """Set the status bar message.

        Args:
            message: The message to display.
            color: Text color.
        """
        self._status_label.configure(text=f"> {message}", text_color=color)

    def _poll_status(self) -> None:
        """Poll and update status indicators."""
        # Update Story Factory status
        if self._process_manager.is_running():
            pid = self._process_manager.get_pid()
            uptime = self._process_manager.get_uptime()

            self._sf_card["indicator"].configure(text_color="#28a745")
            self._sf_card["status_label"].configure(text="RUNNING")

            info_text = APP_URL
            if pid:
                info_text += f" | PID: {pid}"
            if uptime:
                total_seconds = int(uptime.total_seconds())
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 0:
                    info_text += f" | {hours}h {minutes}m"
                elif minutes > 0:
                    info_text += f" | {minutes}m {seconds}s"
                else:
                    info_text += f" | {seconds}s"

            self._sf_card["info_label"].configure(text=info_text)
        else:
            self._sf_card["indicator"].configure(text_color="#dc3545")
            self._sf_card["status_label"].configure(text="STOPPED")
            self._sf_card["info_label"].configure(text="")

        # Update Ollama status (uses cached value from background thread)
        if self._ollama_healthy:
            self._ollama_card["indicator"].configure(text_color="#28a745")
            self._ollama_card["status_label"].configure(text="RUNNING")
            self._ollama_card["info_label"].configure(text=self._ollama_manager.get_url())
        else:
            self._ollama_card["indicator"].configure(text_color="#dc3545")
            self._ollama_card["status_label"].configure(text="STOPPED")
            self._ollama_card["info_label"].configure(text="")

        # Schedule next poll
        self.after(2000, self._poll_status)

    def _poll_ollama_status(self) -> None:
        """Poll Ollama health in background thread to avoid blocking UI."""

        def check_and_update() -> None:
            """Check Ollama health and queue UI update."""
            healthy = self._ollama_manager.check_health()
            self._ui_queue.put(lambda: setattr(self, "_ollama_healthy", healthy))

        thread = threading.Thread(target=check_and_update, daemon=True)
        thread.start()

        # Schedule next poll (longer interval since it's a network call)
        self.after(3000, self._poll_ollama_status)

    def _drain_ui_queue(self) -> None:
        """Execute UI callbacks scheduled from worker threads (thread-safe)."""
        while True:
            try:
                callback = self._ui_queue.get_nowait()
                callback()
            except queue.Empty:
                break
        self.after(50, self._drain_ui_queue)

    def _poll_logs(self) -> None:
        """Poll and update log display."""
        lines = self._log_watcher.get_recent_lines(20)

        # Only update if content has changed (prevents scrollbar twitching)
        if lines == getattr(self, "_last_log_lines", None):
            self.after(1000, self._poll_logs)
            return

        self._last_log_lines = lines

        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")

        if not lines:
            self._log_text.insert("end", "(no logs yet)")
        else:
            for line in lines:
                tag = self._get_log_tag(line)
                # Truncate long lines for display (after tag detection)
                display_line = line[:147] + "..." if len(line) > 150 else line
                self._log_text.insert("end", display_line + "\n", tag)

        self._log_text.configure(state="disabled")

        # Auto-scroll if enabled
        if self._auto_scroll_var.get():
            self._log_text.see("end")

        # Schedule next poll
        self.after(1000, self._poll_logs)

    def _get_log_tag(self, line: str) -> str:
        """Determine the color tag for a log line.

        Args:
            line: The log line.

        Returns:
            Tag name for coloring.
        """
        upper = line.upper()
        if "ERROR" in upper or "EXCEPTION" in upper or "TRACEBACK" in upper:
            return "error"
        elif "WARNING" in upper:
            return "warning"
        elif "INFO" in upper:
            return "info"
        return "debug"

    def _run_in_thread(self, func: Callable, success_msg: str, error_msg: str) -> None:
        """Run a function in a background thread.

        Args:
            func: The function to run.
            success_msg: Message on success.
            error_msg: Message on error.
        """

        def wrapper() -> None:
            """Execute function and queue success/error status update."""
            try:
                result = func()
                # Success: function returned a truthy value (PID or True)
                # Failure: function returned a falsy value (None or False)
                if result:
                    self._ui_queue.put(lambda: self._set_status_message(success_msg, "#28a745"))
                else:
                    self._ui_queue.put(lambda: self._set_status_message(error_msg, "#dc3545"))
            except Exception as exc:
                logger.exception("Error in background task")
                err_msg = str(exc)

                def make_error_callback(msg: str) -> Callable[[], None]:
                    """Create callback that displays error message in status bar."""
                    return lambda: self._set_status_message(f"Error: {msg}", "#dc3545")

                self._ui_queue.put(make_error_callback(err_msg))

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    def _on_start(self) -> None:
        """Handle Start button click."""
        if self._process_manager.is_running():
            self._set_status_message("Already running - stop first", "#ffc107")
            return

        self._set_status_message("Starting...", "#17a2b8")
        self._run_in_thread(
            self._process_manager.start_app,
            f"Started! Open {APP_URL}",
            "Failed to start",
        )

    def _on_stop(self) -> None:
        """Handle Stop button click."""
        if not self._process_manager.is_running():
            self._set_status_message("Not running", "#ffc107")
            return

        self._set_status_message("Stopping...", "#17a2b8")
        self._run_in_thread(
            self._process_manager.stop_app,
            "Stopped",
            "Failed to stop",
        )

    def _on_restart(self) -> None:
        """Handle Restart button click."""

        def restart() -> int | None:
            """Stop app, wait, and restart."""
            self._process_manager.stop_app()
            time.sleep(1)
            return self._process_manager.start_app()

        self._set_status_message("Restarting...", "#17a2b8")
        self._run_in_thread(
            restart,
            f"Restarted! Open {APP_URL}",
            "Failed to restart",
        )

    def _on_restart_clear(self) -> None:
        """Handle Restart & Clear button click."""

        def restart_and_clear() -> int | None:
            """Stop app, clear logs, and restart."""
            self._process_manager.stop_app()
            self._log_watcher.clear_log()
            time.sleep(1)
            return self._process_manager.start_app()

        self._set_status_message("Restarting & clearing logs...", "#17a2b8")
        self._run_in_thread(
            restart_and_clear,
            f"Restarted with fresh logs! Open {APP_URL}",
            "Failed to restart",
        )

    def _on_browser(self) -> None:
        """Handle Browser button click."""
        webbrowser.open(APP_URL)
        self._set_status_message("Opening browser...", "#17a2b8")

    def _on_clear_logs(self) -> None:
        """Handle Clear Logs button click."""
        size = self._log_watcher.clear_log()
        if size > 0:
            kb = size / 1024
            self._set_status_message(f"Logs cleared ({kb:.1f} KB)", "#28a745")
        else:
            self._set_status_message("No logs to clear", "#ffc107")

    def _on_start_ollama(self) -> None:
        """Handle Ollama button click."""
        # Don't block UI - do health check in background thread too
        self._set_status_message("Starting Ollama...", "#17a2b8")

        def start_and_report() -> None:
            """Start Ollama and report detailed status."""
            try:
                # Check if already running first
                if self._ollama_manager.check_health():
                    url = self._ollama_manager.get_url()
                    msg = f"Ollama already running at {url}"
                    self._ui_queue.put(lambda: self._set_status_message(msg, "#28a745"))
                    return

                success, message = self._ollama_manager.start_ollama()
                color = "#28a745" if success else "#dc3545"
                self._ui_queue.put(lambda: self._set_status_message(message, color))
            except Exception as e:
                logger.exception("Error starting Ollama")
                error_msg = f"Error: {e}"
                self._ui_queue.put(lambda: self._set_status_message(error_msg, "#dc3545"))

        thread = threading.Thread(target=start_and_report, daemon=True)
        thread.start()

    def _on_stop_ollama(self) -> None:
        """Handle Stop Ollama button click."""
        # Don't rely solely on health check - try to stop anyway
        self._set_status_message("Stopping Ollama...", "#17a2b8")

        def stop_and_report() -> None:
            """Stop Ollama and report detailed status."""
            try:
                success, message = self._ollama_manager.stop_ollama()
                color = "#28a745" if success else "#dc3545"
                self._ui_queue.put(lambda: self._set_status_message(message, color))
            except Exception as e:
                logger.exception("Error stopping Ollama")
                error_msg = f"Error: {e}"
                self._ui_queue.put(lambda: self._set_status_message(error_msg, "#dc3545"))

        thread = threading.Thread(target=stop_and_report, daemon=True)
        thread.start()

    def _on_close(self) -> None:
        """Handle window close event.

        Note: Closing the control panel does NOT stop the Story Factory app.
        """
        logger.info("Control panel closing (app will continue running)")
        self.destroy()


def main() -> None:
    """Main entry point for the control panel."""
    # Ensure output directories exist
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create and run the control panel
    app = ControlPanel()
    app.mainloop()


if __name__ == "__main__":
    main()
