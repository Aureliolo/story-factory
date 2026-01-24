#!/usr/bin/env python
"""Quick smoke test to verify the app can start and basic functionality works."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all critical modules can be imported."""
    logger.info("Testing imports...")

    try:
        from src.services.story_service import GenerationCancelled
        from src.ui.components.generation_status import GenerationStatus
        from src.ui.state import AppState

        # Verify imported items are valid types
        assert issubclass(GenerationCancelled, Exception)
        assert callable(GenerationStatus)
        assert callable(AppState)

        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def _verify_flag(value: bool, expected: bool, name: str) -> None:
    """Helper to verify flag value without triggering mypy's narrow-type analysis."""
    if value != expected:
        raise AssertionError(f"{name} was {value}, expected {expected}")


def test_app_state():
    """Test AppState generation control methods."""
    logger.info("Testing AppState...")

    try:
        from src.ui.state import AppState

        state = AppState()

        # Test initial state
        _verify_flag(state.generation_cancel_requested, False, "cancel_requested initial")
        _verify_flag(state.generation_pause_requested, False, "pause_requested initial")

        # Test cancel request
        state.request_cancel_generation()
        _verify_flag(state.generation_cancel_requested, True, "cancel_requested after request")

        # Test pause request
        state.request_pause_generation()
        _verify_flag(state.generation_pause_requested, True, "pause_requested after request")

        # Test reset
        state.reset_generation_flags()
        _verify_flag(state.generation_cancel_requested, False, "cancel_requested after reset")
        _verify_flag(state.generation_pause_requested, False, "pause_requested after reset")

        logger.info("✓ AppState tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ AppState test failed: {e}")
        return False


def test_generation_cancelled():
    """Test GenerationCancelled exception."""
    logger.info("Testing GenerationCancelled...")

    try:
        from src.services.story_service import GenerationCancelled

        exc = GenerationCancelled("Test")
        assert isinstance(exc, Exception)
        assert "Test" in str(exc)

        logger.info("✓ GenerationCancelled tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ GenerationCancelled test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    logger.info("=" * 60)
    logger.info("Running smoke tests for async generation feature")
    logger.info("=" * 60)

    tests = [
        test_imports,
        test_app_state,
        test_generation_cancelled,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()  # Blank line between tests

    # Summary
    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
