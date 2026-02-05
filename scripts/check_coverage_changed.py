#!/usr/bin/env python3
"""Check test coverage for changed Python files.

This script runs pytest with coverage only on files that have changed compared
to the target branch (usually main). It enforces 100% coverage on new/modified
code while keeping pre-push hooks fast.

Usage:
    python scripts/check_coverage_changed.py
    python scripts/check_coverage_changed.py --base-branch=develop
    python scripts/check_coverage_changed.py --verbose  # Enable debug logging
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Files/directories to exclude from coverage checks
COVERAGE_EXCLUDES = [
    "tests/",
    "scripts/",
    "main.py",
    "src/ui/",  # UI code excluded per pyproject.toml
]


def get_changed_files(base_branch: str) -> list[str]:
    """Get list of changed Python files compared to base branch.

    Args:
        base_branch: The branch to compare against (e.g., 'main', 'origin/main')

    Returns:
        List of changed .py file paths relative to repo root.

    Raises:
        SystemExit: If git diff fails against all attempted refs or git is not found.
    """
    logger.debug("Determining changed files against base branch: %s", base_branch)

    # Try with origin/ prefix first, fall back to local branch
    for ref in [f"origin/{base_branch}", base_branch]:
        logger.debug("Trying git diff against ref: %s", ref)
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{ref}...HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            logger.error("git command not found. Please ensure git is installed and in your PATH.")
            sys.exit(1)

        if result.returncode == 0:
            logger.debug("Successfully got diff against %s", ref)
            break
    else:
        # Fatal error - cannot determine changed files, so cannot enforce coverage
        logger.error(
            "Could not determine changed files against 'origin/%s' or '%s'. "
            "Git error: %s\n"
            "Hint: Run 'git fetch origin' to update remote refs.",
            base_branch,
            base_branch,
            result.stderr.strip() or "(no error message)",
        )
        sys.exit(1)

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    python_files = [f for f in files if f.endswith(".py")]

    if not python_files:
        logger.info("No Python files changed")
    else:
        logger.debug("Found %d changed Python files", len(python_files))

    return python_files


def filter_coverable_files(files: list[str]) -> list[str]:
    """Filter out files that are excluded from coverage.

    Args:
        files: List of file paths.

    Returns:
        List of files that should have coverage enforced.
    """
    logger.debug("Filtering %d changed file(s) for coverage eligibility", len(files))

    filtered = [
        f
        for f in files
        if not any(f.startswith(exc) for exc in COVERAGE_EXCLUDES) and Path(f).exists()
    ]

    logger.debug("Coverable files after excludes: %d", len(filtered))
    return filtered


def run_coverage_check(files: list[str]) -> int:
    """Run pytest coverage on specific files.

    Args:
        files: List of source files to check coverage for.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    if not files:
        logger.info("No coverable files changed, skipping coverage check")
        return 0

    logger.info("Checking coverage for %d changed file(s):", len(files))
    for f in files:
        logger.info("  - %s", f)

    # Build --cov arguments for each file
    cov_args = [arg for f in files for arg in ("--cov", f)]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *cov_args,
        "--cov-fail-under=100",
        "--cov-report=term-missing",
        "-q",
        "tests/unit/",
        "tests/smoke/",
        "tests/integration/",
        "tests/component/",
    ]

    logger.info("Running coverage check...")
    logger.debug("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> None:
    """Parse arguments and run coverage check on changed files."""
    parser = argparse.ArgumentParser(description="Check test coverage for changed Python files")
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Base branch to compare against (default: main)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging for troubleshooting",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    changed = get_changed_files(args.base_branch)
    coverable = filter_coverable_files(changed)

    if not coverable:
        logger.info("No source files changed that require coverage, skipping")
        sys.exit(0)

    exit_code = run_coverage_check(coverable)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
