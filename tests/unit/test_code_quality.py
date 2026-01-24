"""Code quality checks for the codebase.

These tests enforce code quality standards such as maximum file length
to encourage modular code design.
"""

from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Maximum number of lines allowed per file before suggesting a split
MAX_FILE_LINES = 1000

# Directories to check for Python files
SOURCE_DIRS = [
    PROJECT_ROOT / "src" / "agents",
    PROJECT_ROOT / "src" / "memory",
    PROJECT_ROOT / "src" / "services",
    PROJECT_ROOT / "src" / "ui",
    PROJECT_ROOT / "src" / "utils",
]

# Files that are explicitly allowed to exceed the limit (with justification)
# TODO: These files should be refactored into smaller modules
ALLOWED_LARGE_FILES: dict[str, str] = {
    "src/ui/pages/world.py": "Complex graph visualization with entity CRUD, analysis, and import features - should be split into components",
    "src/services/world_quality_service.py": "Quality scoring and entity generation with multiple entity types - should be split by entity type",
    "src/services/orchestrator.py": "Main story generation workflow coordinating all agents - should be split into sub-workflows",
    "src/ui/pages/write.py": "Main writing interface with interview, structure, and editing - should be split into components",
    "src/memory/world_database.py": "SQLite + NetworkX world database with CRUD, queries, and migrations - should be split by concern",
    "src/memory/mode_database.py": "Mode presets and custom mode management - should be split into presets and storage",
    "src/services/world_service.py": "World building service with entity extraction and generation - should be split by operation type",
    "src/ui/pages/models.py": "Model management with download queue, batch operations, and tag configuration - should be split into components",
}


def _count_lines(file_path: Path) -> int:
    """Count the number of lines in a file.

    Args:
        file_path: Path to the file.

    Returns:
        Number of lines in the file.
    """
    try:
        return len(file_path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def _get_python_files() -> list[tuple[Path, int]]:
    """Get all Python files with their line counts.

    Returns:
        List of (file_path, line_count) tuples for files exceeding the limit.
    """
    large_files = []

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        for py_file in source_dir.rglob("*.py"):
            # Skip __pycache__ and other generated files
            if "__pycache__" in str(py_file):
                continue

            line_count = _count_lines(py_file)

            if line_count > MAX_FILE_LINES:
                # Check if file is in the allowed list
                relative_path = py_file.relative_to(PROJECT_ROOT)
                if str(relative_path).replace("\\", "/") not in ALLOWED_LARGE_FILES:
                    large_files.append((py_file, line_count))

    return large_files


class TestFileLength:
    """Tests for enforcing maximum file length."""

    def test_no_files_exceed_max_length(self) -> None:
        """Ensure no Python files exceed the maximum allowed line count.

        Files that exceed the limit should be refactored into smaller,
        more focused modules. This improves maintainability, testability,
        and makes code reviews easier.

        To add an exception for a specific file, add it to ALLOWED_LARGE_FILES
        with a justification comment.
        """
        large_files = _get_python_files()

        if large_files:
            # Build a helpful error message
            file_list = "\n".join(
                f"  - {file.relative_to(PROJECT_ROOT)}: {count} lines "
                f"(exceeds {MAX_FILE_LINES} by {count - MAX_FILE_LINES})"
                for file, count in sorted(large_files, key=lambda x: -x[1])
            )

            pytest.fail(
                f"The following files exceed the maximum allowed length of {MAX_FILE_LINES} lines:\n"
                f"{file_list}\n\n"
                f"Consider refactoring these files into smaller modules.\n"
                f"If a file must remain large, add it to ALLOWED_LARGE_FILES with justification."
            )

    def test_allowed_large_files_have_justification(self) -> None:
        """Ensure all allowed large files have a non-empty justification."""
        for file_path, justification in ALLOWED_LARGE_FILES.items():
            assert justification.strip(), (
                f"ALLOWED_LARGE_FILES entry '{file_path}' must have a justification"
            )

    def test_allowed_large_files_exist(self) -> None:
        """Ensure all files in ALLOWED_LARGE_FILES actually exist."""
        for file_path in ALLOWED_LARGE_FILES:
            full_path = PROJECT_ROOT / file_path
            assert full_path.exists(), (
                f"ALLOWED_LARGE_FILES contains '{file_path}' which does not exist. "
                "Remove it from the allowed list."
            )

    def test_allowed_large_files_actually_large(self) -> None:
        """Ensure files in ALLOWED_LARGE_FILES actually exceed the limit.

        If a file has been refactored below the limit, it should be removed
        from the exception list.
        """
        for file_path in ALLOWED_LARGE_FILES:
            full_path = PROJECT_ROOT / file_path
            if not full_path.exists():
                continue  # Handled by test_allowed_large_files_exist

            line_count = _count_lines(full_path)
            assert line_count > MAX_FILE_LINES, (
                f"'{file_path}' is in ALLOWED_LARGE_FILES but only has {line_count} lines "
                f"(limit is {MAX_FILE_LINES}). Remove it from the exception list."
            )
