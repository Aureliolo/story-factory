"""Story Factory Control Panel - Native Python/CustomTkinter GUI.

A modern, flicker-free control panel for managing the Story Factory application.
Replaces the PowerShell-based control panel with a native desktop experience.
"""

import json
import logging
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
                with open(SETTINGS_FILE) as f:
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
            req = urllib.request.Request(f"{self._url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                return bool(response.status == 200)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def start_ollama(self) -> bool:
        """Start the Ollama service.

        Returns:
            True if Ollama was started, False otherwise.
        """
        if self.check_health():
            logger.info("Ollama already running at %s", self._url)
            return True

        # Try Windows service first
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["sc", "query", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=SUBPROCESS_FLAGS,
                )
                if "RUNNING" not in result.stdout:
                    logger.info("Starting Ollama service...")
                    subprocess.run(
                        ["sc", "start", "ollama"],
                        capture_output=True,
                        timeout=10,
                        creationflags=SUBPROCESS_FLAGS,
                    )
                    # Give it time to start
                    for _ in range(5):
                        time.sleep(1)
                        if self.check_health():
                            logger.info("Ollama service started")
                            return True
                    logger.warning("Ollama service started but not responding")
                    return False
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.debug("Service check failed: %s", e)

        # Try starting ollama serve directly
        try:
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = SUBPROCESS_FLAGS

            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )

            for _ in range(5):
                time.sleep(1)
                if self.check_health():
                    logger.info("Ollama started via 'ollama serve'")
                    return True

            logger.warning("Ollama started but not responding")
            return False

        except FileNotFoundError:
            logger.error("Ollama not found. Install from https://ollama.ai")
            return False
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("Failed to start Ollama: %s", e)
            return False

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
                    lines = new_lines + lines

                    if seek_pos == 0:
                        break

                # Filter empty lines first, then take last n and truncate
                non_empty_lines = [stripped for ln in lines if (stripped := ln.strip())]
                result = []
                for line in non_empty_lines[-n:]:
                    if len(line) > 150:
                        line = line[:147] + "..."
                    result.append(line)
                return result

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
            with open(self._log_path, "w") as f:
                f.truncate(0)
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

        # Window setup
        self.title("Story Factory Control Panel")
        self.geometry("600x700")
        self.minsize(500, 500)

        # Set dark mode
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Build the UI
        self._create_header()
        self._create_status_frame()
        self._create_controls_frame()
        self._create_log_frame()
        self._create_status_bar()

        # Start polling
        self._poll_status()
        self._poll_logs()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        logger.info("Control panel initialized")

    def _create_header(self) -> None:  # pragma: no cover
        """Create the header section (UI widget creation only)."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))

        title_label = ctk.CTkLabel(
            header_frame,
            text="STORY FACTORY CONTROL PANEL",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label.pack()

    def _create_status_frame(self) -> None:  # pragma: no cover
        """Create the status indicators section (UI widget creation only)."""
        status_frame = ctk.CTkFrame(self)
        status_frame.pack(fill="x", padx=20, pady=10)

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
        controls_frame.pack(fill="x", padx=20, pady=10)

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

        self._ollama_btn = ctk.CTkButton(
            row2,
            text="Ollama",
            command=self._on_start_ollama,
            width=100,
            fg_color="#17a2b8",
            hover_color="#138496",
        )
        self._ollama_btn.pack(side="left", padx=5)

    def _create_log_frame(self) -> None:  # pragma: no cover
        """Create the log viewer section (UI widget creation only)."""
        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Header with title and auto-scroll toggle
        header = ctk.CTkFrame(log_frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(10, 5))

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

        # Log text area
        self._log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            state="disabled",
        )
        self._log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Configure log colors
        self._log_text._textbox.tag_config("error", foreground="#dc3545")
        self._log_text._textbox.tag_config("warning", foreground="#ffc107")
        self._log_text._textbox.tag_config("info", foreground="#6c757d")
        self._log_text._textbox.tag_config("debug", foreground="#495057")

    def _create_status_bar(self) -> None:  # pragma: no cover
        """Create the status bar at the bottom (UI widget creation only)."""
        status_bar = ctk.CTkFrame(self, height=30)
        status_bar.pack(fill="x", padx=20, pady=(0, 10))

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

        # Update Ollama status
        if self._ollama_manager.check_health():
            self._ollama_card["indicator"].configure(text_color="#28a745")
            self._ollama_card["status_label"].configure(text="RUNNING")
            self._ollama_card["info_label"].configure(text=self._ollama_manager.get_url())
        else:
            self._ollama_card["indicator"].configure(text_color="#dc3545")
            self._ollama_card["status_label"].configure(text="STOPPED")
            self._ollama_card["info_label"].configure(text="")

        # Schedule next poll
        self.after(2000, self._poll_status)

    def _poll_logs(self) -> None:
        """Poll and update log display."""
        lines = self._log_watcher.get_recent_lines(20)

        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")

        if not lines:
            self._log_text.insert("end", "(no logs yet)")
        else:
            for line in lines:
                tag = self._get_log_tag(line)
                self._log_text.insert("end", line + "\n", tag)

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

        def wrapper():
            try:
                result = func()
                # Success: function returned a truthy value (PID or True)
                # Failure: function returned a falsy value (None or False)
                if result:
                    self.after(0, lambda: self._set_status_message(success_msg, "#28a745"))
                else:
                    self.after(0, lambda: self._set_status_message(error_msg, "#dc3545"))
            except Exception as exc:
                logger.exception("Error in background task")
                err_msg = str(exc)
                self.after(0, lambda: self._set_status_message(f"Error: {err_msg}", "#dc3545"))

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

        def restart():
            self._process_manager.stop_app()
            time.sleep(1)
            return self._process_manager.start_app()

        self._set_status_message("Restarting...", "#17a2b8")
        self._run_in_thread(
            restart,
            f"Restarted! Open {APP_URL}",
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
        if self._ollama_manager.check_health():
            url = self._ollama_manager.get_url()
            self._set_status_message(f"Ollama already running at {url}", "#28a745")
            return

        self._set_status_message("Starting Ollama...", "#17a2b8")
        self._run_in_thread(
            self._ollama_manager.start_ollama,
            f"Ollama started at {self._ollama_manager.get_url()}",
            "Failed to start Ollama",
        )

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
