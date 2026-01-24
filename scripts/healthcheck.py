#!/usr/bin/env python3
"""Health check script for Story Factory.

Verifies that all dependencies are properly installed and configured.
"""

import logging
import sys
from pathlib import Path

from src.settings import STORIES_DIR

# Suppress logging during healthcheck unless there are errors
logging.basicConfig(level=logging.ERROR)


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is 3.14+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 14:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    return False, f"Python {version.major}.{version.minor} (requires 3.14+) ✗"


def check_dependencies() -> tuple[bool, str]:
    """Check if all required Python packages are installed."""
    required = ["ollama", "nicegui", "pydantic"]
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        return False, f"Missing packages: {', '.join(missing)} ✗"
    return True, "All dependencies installed ✓"


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and accessible."""
    try:
        from src.agents.base import BaseAgent

        is_healthy, message = BaseAgent.check_ollama_health()
        if is_healthy:
            return True, f"Ollama: {message} ✓"
        return False, f"Ollama: {message} ✗"
    except Exception as e:
        logging.error(f"Ollama health check failed: {e}")
        return False, f"Ollama check failed: {e} ✗"


def check_settings() -> tuple[bool, str]:
    """Check if settings file exists and is valid."""
    from src.settings import Settings

    try:
        settings = Settings.load()
        settings.validate()
        return True, "Settings loaded and valid ✓"
    except Exception as e:
        logging.error(f"Settings validation failed: {e}")
        return False, f"Settings error: {e} ✗"


def check_output_directory() -> tuple[bool, str]:
    """Check if output directory exists and is writable."""
    output_dir = STORIES_DIR
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True, "Output directory writable ✓"
    except Exception as e:
        logging.error(f"Output directory check failed: {e}")
        return False, f"Output directory not writable: {e} ✗"


def check_logs_directory() -> tuple[bool, str]:
    """Check if logs directory exists and is writable."""
    logs_dir = Path(__file__).parent.parent / "output" / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_file = logs_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True, "Logs directory writable ✓"
    except Exception as e:
        logging.error(f"Logs directory check failed: {e}")
        return False, f"Logs directory not writable: {e} ✗"


def run_health_check() -> bool:
    """Run all health checks and print results.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("STORY FACTORY - HEALTH CHECK")
    print("=" * 60)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Settings", check_settings),
        ("Output Directory", check_output_directory),
        ("Logs Directory", check_logs_directory),
        ("Ollama Service", check_ollama),
    ]

    results = []
    for name, check_func in checks:
        try:
            passed, message = check_func()
            results.append(passed)
            status = "✓" if passed else "✗"
            print(f"[{status}] {name}: {message}")
        except Exception as e:
            logging.exception(f"Health check '{name}' raised unexpected exception: {e}")
            results.append(False)
            print(f"[✗] {name}: Unexpected error: {e}")

    print()
    print("=" * 60)

    all_passed = all(results)
    if all_passed:
        print("✓ ALL CHECKS PASSED - System ready!")
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        failed_count = len([r for r in results if not r])
        print(f"  {failed_count}/{len(results)} checks failed")

    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)
