"""E2E test fixtures for Playwright browser automation."""

import socket
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest


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
def mock_ollama_for_e2e():
    """Mock Ollama at module level for E2E tests."""
    with patch("services.model_service.ollama") as mock_ollama:
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [
                {"name": "test-model:latest"},
                {"name": "test-model:7b"},
            ]
        }
        mock_client.generate.return_value = {"response": "Test response from mock LLM"}
        mock_ollama.Client.return_value = mock_client
        yield mock_client


@pytest.fixture(scope="session")
def app_server(mock_ollama_for_e2e):
    """Start the app server for E2E tests.

    This fixture starts the NiceGUI server in a subprocess and waits for it
    to be ready before yielding control to tests.
    """
    port = 7861

    if is_port_in_use(port):
        pytest.skip(f"Port {port} is already in use")

    process = subprocess.Popen(
        ["python", "main.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not wait_for_server(port):
        process.terminate()
        pytest.fail("Server failed to start within timeout")

    yield f"http://localhost:{port}"

    process.terminate()
    process.wait(timeout=5)


@pytest.fixture
def base_url(app_server):
    """Get the base URL for the running app."""
    return app_server
