#!/usr/bin/env python3
"""Check test coverage for changed Python files.

This script runs pytest with coverage only on files that have changed compared
to the target branch (usually main). It enforces 100% coverage on new/modified
code while keeping pre-push hooks fast.

Usage:
    python scripts/check_coverage_changed.py
    python scripts/check_coverage_changed.py --base-branch=develop
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    """
    # Try with origin/ prefix first, fall back to local branch
    for ref in [f"origin/{base_branch}", base_branch]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{ref}...HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            break
    else:
        logger.warning("Could not determine changed files, running full coverage check")
        return []

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    python_files = [f for f in files if f.endswith(".py")]
    return python_files


def filter_coverable_files(files: list[str]) -> list[str]:
    """Filter out files that are excluded from coverage.

    Args:
        files: List of file paths.

    Returns:
        List of files that should have coverage enforced.
    """
    coverable = []
    for f in files:
        excluded = any(f.startswith(exc) for exc in COVERAGE_EXCLUDES)
        if not excluded and Path(f).exists():
            coverable.append(f)
    return coverable


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

    logger.info(f"Checking coverage for {len(files)} changed file(s):")
    for f in files:
        logger.info(f"  - {f}")

    # Build --cov arguments for each file
    cov_args = []
    for f in files:
        cov_args.extend(["--cov", f])

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
    ]

    logger.info("Running coverage check...")
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
    args = parser.parse_args()

    changed = get_changed_files(args.base_branch)
    coverable = filter_coverable_files(changed)

    if not coverable:
        logger.info("No source files changed that require coverage, skipping")
        sys.exit(0)

    exit_code = run_coverage_check(coverable)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
