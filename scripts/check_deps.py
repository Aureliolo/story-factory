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
import subprocess
import sys
import tomllib
from pathlib import Path


def parse_requirement(req: str) -> tuple[str, str]:
    """Parse a requirement string into (package_name, version).

    Args:
        req: Requirement string like "nicegui==3.6.1" or "ruff==0.14.14"

    Returns:
        Tuple of (package_name, required_version)
    """
    if "==" not in req:
        return req.lower(), ""
    name, version = req.split("==", 1)
    return name.lower().replace("-", "_"), version


def get_installed_version(package: str) -> str | None:
    """Get the installed version of a package.

    Args:
        package: Package name

    Returns:
        Version string or None if not installed
    """
    try:
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
    except Exception:
        return None


def load_required_deps(pyproject_path: Path) -> dict[str, str]:
    """Load required dependencies from pyproject.toml.

    Args:
        pyproject_path: Path to pyproject.toml

    Returns:
        Dict of {package_name: required_version}
    """
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

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
    """Check dependencies and optionally install missing/outdated ones.

    Args:
        auto_install: If True, automatically install/upgrade packages

    Returns:
        Exit code (0 = success, 1 = missing/outdated deps without auto-install)
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
            [sys.executable, "-m", "pip", "install", "--quiet", *to_install],
            check=False,
        )
        if result.returncode != 0:
            print("Error: Failed to install packages")
            return 1
        print("Dependencies updated successfully")
        return 0

    print("\nRun with --auto-install to fix, or: pip install -e '.[all]'")
    return 1


def main() -> None:
    """Main entry point."""
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
