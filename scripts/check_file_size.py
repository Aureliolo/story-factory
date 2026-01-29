#!/usr/bin/env python3
"""Pre-commit hook to enforce maximum file length.

This script prevents committing Python files that exceed 1000 lines.
Large files should be split into smaller, focused modules.

Usage:
    python scripts/check_file_size.py [files...]

Exit codes:
    0: All files are within the limit
    1: One or more files exceed the limit
"""

import sys
from pathlib import Path

MAX_FILE_LINES = 1000

# Only check these directories (same as test_code_quality.py)
CHECKED_PREFIXES = (
    "src/agents/",
    "src/memory/",
    "src/services/",
    "src/ui/",
    "src/utils/",
)

SPLIT_GUIDANCE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HOW TO SPLIT A LARGE FILE                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DO NOT add this file to any "allowed" list or skip this check.              ║
║  Large files harm maintainability, testability, and code review.             ║
║                                                                              ║
║  INSTEAD, split the file into a package:                                     ║
║                                                                              ║
║  1. Create a directory with the same name as the file:                       ║
║       src/services/my_service.py → src/services/my_service/                  ║
║                                                                              ║
║  2. Create __init__.py that re-exports the public API:                       ║
║       from src.services.my_service._core import MyService                    ║
║       __all__ = ["MyService"]                                                ║
║                                                                              ║
║  3. Split into focused modules by concern:                                   ║
║       _core.py      - Main class/entry point                                 ║
║       _helpers.py   - Internal helper functions                              ║
║       _types.py     - Type definitions and models                            ║
║       _validation.py - Validation logic                                      ║
║                                                                              ║
║  4. Use underscore prefix (_) for internal modules to signal private API     ║
║                                                                              ║
║  5. Keep each submodule under 500 lines ideally, 1000 max                    ║
║                                                                              ║
║  Example: See how src/services/orchestrator/ was split from orchestrator.py  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def count_lines(file_path: Path) -> int:
    """Count non-empty lines in a file."""
    try:
        return len(file_path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def should_check_file(file_path: str) -> bool:
    """Check if the file is in a directory we enforce limits on."""
    # Normalize path separators
    normalized = file_path.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in CHECKED_PREFIXES)


def main() -> int:
    """Check files for maximum length."""
    files_to_check = sys.argv[1:] if len(sys.argv) > 1 else []

    # Filter to only Python files in checked directories
    python_files = [f for f in files_to_check if f.endswith(".py") and should_check_file(f)]

    if not python_files:
        return 0

    violations = []
    for file_path in python_files:
        path = Path(file_path)
        if not path.exists():
            continue

        line_count = count_lines(path)
        if line_count > MAX_FILE_LINES:
            violations.append((file_path, line_count))

    if violations:
        print("\n" + "=" * 80)
        print("ERROR: The following files exceed the maximum allowed length:")
        print("=" * 80 + "\n")

        for file_path, line_count in sorted(violations, key=lambda x: -x[1]):
            excess = line_count - MAX_FILE_LINES
            print(f"  ✗ {file_path}: {line_count} lines (exceeds limit by {excess})")

        print(SPLIT_GUIDANCE)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
