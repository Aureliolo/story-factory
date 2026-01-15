"""E2E test fixtures for Playwright browser automation."""

import os
import socket
import subprocess
import time

import pytest


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context - headless mode is default."""
    return {**browser_context_args}


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Force headless mode for all browser launches."""
    return {**browser_type_launch_args, "headless": True}


def find_free_port() -> int:
    """Find an available port by letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        port: int = s.getsockname()[1]
        return port


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for server to start accepting connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def app_server():
    """Start the app server for E2E tests.

    This fixture starts the NiceGUI server in a subprocess and waits for it
    to be ready before yielding control to tests.

    Note: Session-scoped mocks don't affect subprocesses. E2E tests run against
    the real app with real Ollama (or fail gracefully if unavailable).
    """
    port = find_free_port()

    # Clear pytest env vars so NiceGUI doesn't enter test mode in subprocess
    # NiceGUI checks PYTEST_CURRENT_TEST to detect pytest and expects
    # NICEGUI_SCREEN_TEST_PORT to be set, which breaks our subprocess approach
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}

    process = subprocess.Popen(
        ["python", "main.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    if not wait_for_server(port):
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"Server failed to start within timeout.\nstdout: {stdout}\nstderr: {stderr}")

    yield f"http://localhost:{port}"

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Give up on waiting; avoid hanging the test suite if the process is unkillable.
            pass


@pytest.fixture(scope="session")
def base_url(app_server):
    """Get the base URL for the running app.

    Must be session-scoped to match app_server fixture scope.
    """
    return app_server
