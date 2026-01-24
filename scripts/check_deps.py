#!/usr/bin/env python3
"""Check and optionally install dependencies from pyproject.toml.

This script compares installed package versions against those specified in
pyproject.toml and reports any mismatches. With --auto-install, it will
automatically install/upgrade packages to the required versions.

Usage:
    python scripts/check_deps.py              # Check only
    python scripts/check_deps.py --auto-install  # Check and install if needed
"""

import argparse
import logging
import re
import subprocess
import sys
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Normalize package name per PEP 503.

    All runs of hyphens, underscores, and periods are replaced with a single hyphen,
    and the result is lowercased.

    Args:
        name: Package name to normalize

    Returns:
        Normalized package name
    """
    logger.debug("Normalizing package name: %s", name)
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(req: str) -> tuple[str, str]:
    """
    Parse a dependency requirement string into a (package_name, version) pair.

    Parameters:
        req (str): Requirement like "nicegui==3.6.1", "ruff==0.14.14", or just a package name.

    Returns:
        tuple[str, str]: (normalized_name, version) where name is PEP 503 normalized
            and version is the specified version or empty string if none.
    """
    logger.debug("Parsing requirement: %s", req)
    if "==" not in req:
        return _normalize_name(req), ""
    name, version = req.split("==", 1)
    return _normalize_name(name), version


def get_installed_version(package: str) -> str | None:
    """
    Return the installed version string for the given package.

    Parameters:
        package (str): Package name to query (will be normalized for lookup).

    Returns:
        str | None: Version string if the package is installed, None otherwise.
    """
    logger.debug("Checking installed version for: %s", package)
    normalized = _normalize_name(package)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", normalized],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # Try with original name in case pip prefers it
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.debug("Package %s not found", package)
                return None
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
                logger.debug("Found %s version: %s", package, version)
                return version
        return None
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Error checking %s: %s", package, e)
        return None


def load_required_deps(pyproject_path: Path) -> dict[str, str]:
    """Load required dependencies from pyproject.toml.

    Args:
        pyproject_path: Path to pyproject.toml

    Returns:
        Dict of {normalized_package_name: required_version}

    Raises:
        SystemExit: If pyproject.toml cannot be parsed or is missing required sections.
    """
    logger.debug("Loading dependencies from: %s", pyproject_path)
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        logger.error("Invalid TOML in %s: %s", pyproject_path, e)
        sys.exit(1)

    # Validate required sections exist
    if "project" not in data:
        logger.error("Missing [project] section in %s", pyproject_path)
        sys.exit(1)

    project = data["project"]
    if "dependencies" not in project:
        logger.error("Missing [project].dependencies in %s", pyproject_path)
        sys.exit(1)

    deps: dict[str, str] = {}

    # Main dependencies (required)
    for req in project["dependencies"]:
        name, version = parse_requirement(req)
        if version:
            deps[name] = version

    # Optional dependencies (dev, test) - these are truly optional
    if "optional-dependencies" in project:
        optional = project["optional-dependencies"]
        for group in ["dev", "test"]:
            if group in optional:
                logger.debug("Processing optional group: %s", group)
                for req in optional[group]:
                    name, version = parse_requirement(req)
                    if version:
                        deps[name] = version

    logger.debug("Loaded %d dependencies", len(deps))
    return deps


def check_deps(auto_install: bool = False) -> int:
    """
    Check dependencies declared in pyproject.toml and report missing or outdated packages.

    If any required packages are missing or have a different version than specified,
    logs diagnostics. If `auto_install` is True, attempts to install/upgrade the
    missing or outdated packages using pip.

    Parameters:
        auto_install (bool): If True, automatically install or upgrade packages.

    Returns:
        int: Exit code - 0 if all dependencies are present and up-to-date or
            installation succeeded, 1 if dependencies are missing/outdated.
    """
    logger.debug("check_deps called with auto_install=%s", auto_install)
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        logger.error("pyproject.toml not found at %s", pyproject_path)
        return 1

    required = load_required_deps(pyproject_path)
    missing: list[str] = []
    outdated: list[tuple[str, str, str]] = []  # (name, installed, required)

    for package, required_version in sorted(required.items()):
        installed = get_installed_version(package)
        if installed is None:
            missing.append(f"{package}=={required_version}")
        elif installed != required_version:
            outdated.append((package, installed, required_version))

    if not missing and not outdated:
        logger.debug("All dependencies are up to date")
        return 0

    if missing:
        logger.warning("Missing packages: %s", ", ".join(missing))

    if outdated:
        logger.warning("Outdated packages:")
        for name, installed, required_ver in outdated:
            logger.warning("  %s: installed %s, requires %s", name, installed, required_ver)

    if auto_install:
        to_install = missing + [f"{name}=={req}" for name, _, req in outdated]
        logger.info("Installing: %s", ", ".join(to_install))
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *to_install],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("Failed to install packages:\n%s", result.stderr)
            return 1
        logger.info("Dependencies updated successfully")
        return 0

    logger.info("Run with --auto-install to fix, or: pip install -e '.[all]'")
    return 1


def main() -> None:
    """
    Parse command-line arguments and run the dependency check.

    Recognizes the `--auto-install` flag to enable automatic installation or
    upgrade of packages to the required versions before exiting with the same
    status code produced by `check_deps`.
    """
    # Configure logging for CLI output
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    parser = argparse.ArgumentParser(description="Check dependencies from pyproject.toml")
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Automatically install/upgrade packages to required versions",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(check_deps(auto_install=args.auto_install))


if __name__ == "__main__":
    main()
