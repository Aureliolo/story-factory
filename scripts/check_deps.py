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
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def _normalize_name(name: str) -> str:
    """Normalize package name per PEP 503.

    All runs of hyphens, underscores, and periods are replaced with a single hyphen,
    and the result is lowercased.

    Args:
        name: Package name to normalize

    Returns:
        Normalized package name
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(req: str) -> tuple[str, str]:
    """
    Parse a dependency requirement string into a (package_name, version) pair.

    Parameters:
        req (str): Requirement like "nicegui==3.6.1", "ruff==0.14.14", or just a package name.

    Returns:
        tuple[str, str]: (normalized_name, version) where name is PEP 503 normalized and version is the specified version or empty string if none.
    """
    if "==" not in req:
        return _normalize_name(req), ""
    name, version = req.split("==", 1)
    return _normalize_name(name), version


def get_installed_version(package: str) -> str | None:
    """
    Return the installed version string for the given package, or None if it is not installed or cannot be determined.

    Parameters:
        package (str): Package name to query (will be normalized for lookup).

    Returns:
        str | None: Version string if the package is installed, `None` otherwise.
    """
    # Normalize the package name for pip show
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
                return None
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
        return None
    except (subprocess.SubprocessError, OSError):
        return None


def load_required_deps(pyproject_path: Path) -> dict[str, str]:
    """Load required dependencies from pyproject.toml.

    Args:
        pyproject_path: Path to pyproject.toml

    Returns:
        Dict of {normalized_package_name: required_version}

    Raises:
        SystemExit: If pyproject.toml cannot be parsed.
    """
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error: Invalid TOML in {pyproject_path}: {e}")
        sys.exit(1)

    deps: dict[str, str] = {}

    # Main dependencies
    for req in data.get("project", {}).get("dependencies", []):
        name, version = parse_requirement(req)
        if version:
            deps[name] = version

    # Optional dependencies (dev, test)
    optional = data.get("project", {}).get("optional-dependencies", {})
    for group in ["dev", "test"]:
        for req in optional.get(group, []):
            name, version = parse_requirement(req)
            if version:
                deps[name] = version

    return deps


def check_deps(auto_install: bool = False) -> int:
    """
    Check dependencies declared in pyproject.toml and report missing or outdated packages.

    If any required packages are missing or have a different version than specified, prints diagnostics.
    If `auto_install` is True, attempts to install/upgrade the missing or outdated packages using pip.

    Parameters:
        auto_install (bool): If True, automatically install or upgrade missing/outdated packages.

    Returns:
        int: Exit code — `0` if all dependencies are present and up-to-date or installation succeeded, `1` if dependencies are missing/outdated and not installed or an installation error occurred.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
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
        return 0

    if missing:
        print(f"Missing packages: {', '.join(missing)}")

    if outdated:
        print("Outdated packages:")
        for name, installed, required_ver in outdated:
            print(f"  {name}: installed {installed}, requires {required_ver}")

    if auto_install:
        to_install = missing + [f"{name}=={req}" for name, _, req in outdated]
        print(f"\nInstalling: {', '.join(to_install)}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *to_install],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"Error: Failed to install packages\n{result.stderr}")
            return 1
        print("Dependencies updated successfully")
        return 0

    print("\nRun with --auto-install to fix, or: pip install -e '.[all]'")
    return 1


def main() -> None:
    """
    Parse command-line arguments and run the dependency check, exiting with the check's status.

    Recognizes the `--auto-install` flag to enable automatic installation or upgrade of packages to the required versions before exiting with the same status code produced by `check_deps`.
    """
    parser = argparse.ArgumentParser(description="Check dependencies from pyproject.toml")
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Automatically install/upgrade packages to required versions",
    )
    args = parser.parse_args()

    sys.exit(check_deps(auto_install=args.auto_install))


if __name__ == "__main__":
    main()
