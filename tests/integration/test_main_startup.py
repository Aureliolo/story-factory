"""Integration tests for main.py application startup."""

import subprocess
import sys
from pathlib import Path


class TestMainAppStartup:
    """Test main.py application startup in different modes."""

    def test_main_imports_successfully(self):
        """Test that main.py can be imported without errors."""
        # This tests the module-level code execution
        result = subprocess.run(
            [sys.executable, "-c", "import main"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should not crash on import
        assert result.returncode == 0, f"Import failed: {result.stderr}"

    def test_cli_mode_help(self):
        """Test CLI mode with --help flag."""
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--cli" in result.stdout or "usage:" in result.stdout.lower()

    def test_run_web_ui_function_exists(self):
        """Test that run_web_ui function exists and is callable."""
        import main

        assert hasattr(main, "run_web_ui")
        assert callable(main.run_web_ui)

    def test_run_cli_function_exists(self):
        """Test that run_cli function exists and is callable."""
        import main

        assert hasattr(main, "run_cli")
        assert callable(main.run_cli)

    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        import main

        assert hasattr(main, "main")
        assert callable(main.main)


class TestMainModuleAttributes:
    """Test main module attributes and structure."""

    def test_main_has_logger(self):
        """Test that main module has logger configured."""
        import main

        assert hasattr(main, "logger")
        assert main.logger is not None

    def test_main_has_docstring(self):
        """Test that main module has documentation."""
        import main

        assert main.__doc__ is not None
        assert len(main.__doc__) > 0
        assert "Story Factory" in main.__doc__

    def test_main_module_name(self):
        """Test main module has correct name."""
        import main

        # When run as script, __name__ would be "__main__"
        # When imported, it's "main"
        assert main.__name__ == "main"


class TestMainCommandLineInterface:
    """Test CLI argument parsing."""

    def test_cli_accepts_list_flag(self):
        """Test that CLI accepts --list-stories flag."""
        result = subprocess.run(
            [sys.executable, "main.py", "--cli", "--list-stories"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should not crash
        assert result.returncode == 0 or "No saved stories" in result.stdout

    def test_cli_version_check(self):
        """Test that main.py doesn't crash on basic execution check."""
        result = subprocess.run(
            [sys.executable, "-c", "import main; print('OK')"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout


class TestMainFunctionality:
    """Test main.py functionality without full startup."""

    def test_imports_required_modules(self):
        """Test that main can import all required modules."""
        test_code = """
import main
# Test internal imports work
from services import ServiceContainer
from settings import Settings
from ui import create_app
from workflows.orchestrator import StoryOrchestrator
print("All imports successful")
"""
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "All imports successful" in result.stdout

    def test_logging_setup(self):
        """Test that logging is configured properly."""
        test_code = """
import main
import logging
# Check logger exists and has a level set
assert main.logger is not None
assert isinstance(main.logger, logging.Logger)
print("Logging configured")
"""
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Logging configured" in result.stdout
