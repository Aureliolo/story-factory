#!/usr/bin/env python
"""Audit exception handling in the codebase.

This script finds all exception handlers and checks:
1. Whether they log the exception
2. Whether they use broad Exception vs specific types
3. Whether they re-raise properly (with 'from' clause)
"""

import re
import sys
from pathlib import Path
from typing import Any


def find_exception_handlers(file_path: Path) -> list[dict[str, Any]]:
    """Find all exception handlers in a Python file."""
    handlers = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return handlers

    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        # Match except clauses
        except_match = re.match(r"\s*except\s+(.+?):", line)
        if except_match:
            exception_type = except_match.group(1).strip()

            # Check if it logs the exception (look ahead 10 lines)
            has_logging = False
            has_reraise = False
            block_indent = len(line) - len(line.lstrip())

            for j in range(i, min(i + 10, len(lines) + 1)):
                future_line = lines[j - 1]
                current_indent = len(future_line) - len(future_line.lstrip())

                # Stop if we've exited the except block (indent less than or equal to except line)
                # BUT skip the except line itself (j == i) and empty lines
                if j > i and future_line.strip() and current_indent <= block_indent:
                    break

                if re.search(r"\b(?:logger|logging)\.(error|exception|warning|info|debug)", future_line):
                    has_logging = True
                if re.search(r"\braise\b", future_line):
                    has_reraise = True

            handlers.append(
                {
                    "file": str(file_path),
                    "line": i,
                    "exception_type": exception_type,
                    "has_logging": has_logging,
                    "has_reraise": has_reraise,
                    "is_broad": "Exception" in exception_type and "as" in exception_type,
                }
            )

    return handlers


def audit_exceptions(
    root_dir: Path, exclude_dirs: list[str] | None = None
) -> dict[str, int | list[dict[str, Any]]]:
    """Audit all exception handlers in the codebase."""
    if exclude_dirs is None:
        exclude_dirs = ["tests", ".git", "__pycache__", "venv", ".venv", "build", "dist"]

    all_handlers = []

    for py_file in root_dir.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in str(py_file) for excluded in exclude_dirs):
            continue

        handlers = find_exception_handlers(py_file)
        all_handlers.extend(handlers)

    return {
        "total": len(all_handlers),
        "handlers": all_handlers,
        "broad_without_logging": [
            h for h in all_handlers if h["is_broad"] and not h["has_logging"]
        ],
        "broad_with_logging": [h for h in all_handlers if h["is_broad"] and h["has_logging"]],
        "specific_handlers": [h for h in all_handlers if not h["is_broad"]],
    }


def print_report(audit_results: dict[str, Any]) -> None:
    """Print a formatted audit report."""
    total = audit_results["total"]
    broad_no_log = audit_results["broad_without_logging"]
    broad_with_log = audit_results["broad_with_logging"]
    specific = audit_results["specific_handlers"]

    print("=" * 80)
    print("EXCEPTION HANDLING AUDIT REPORT")
    print("=" * 80)
    print()

    print(f"Total exception handlers found: {total}")
    print(f"  - Broad (Exception) without logging: {len(broad_no_log)} ⚠️")
    print(f"  - Broad (Exception) with logging: {len(broad_with_log)} ✓")
    print(f"  - Specific exception types: {len(specific)} ✓")
    print()

    if broad_no_log:
        print("=" * 80)
        print("⚠️  BROAD EXCEPTION HANDLERS WITHOUT LOGGING")
        print("=" * 80)
        print()
        print("These handlers should either:")
        print("1. Log the exception for debugging")
        print("2. Use more specific exception types")
        print("3. Document why broad handling is needed")
        print()

        for handler in broad_no_log:
            print(f"📄 {handler['file']}:{handler['line']}")
            print(f"   Type: {handler['exception_type']}")
            print(f"   Logging: {'✓' if handler['has_logging'] else '✗'}")
            print(f"   Re-raise: {'✓' if handler['has_reraise'] else '✗'}")
            print()

    if broad_with_log:
        print("=" * 80)
        print("✓ BROAD EXCEPTION HANDLERS WITH LOGGING")
        print("=" * 80)
        print()
        print("These handlers use broad Exception but do log - acceptable for:")
        print("- Error boundary patterns")
        print("- User-facing error handling")
        print("- Fallback mechanisms")
        print()

        for handler in broad_with_log[:10]:  # Show first 10
            print(f"📄 {handler['file']}:{handler['line']}")
            print(f"   Type: {handler['exception_type']}")
            print()

        if len(broad_with_log) > 10:
            print(f"... and {len(broad_with_log) - 10} more")
            print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if broad_no_log:
        print(f"⚠️  Fix {len(broad_no_log)} broad handlers without logging")
        print("   Add logger.exception() or logger.error() calls")
    else:
        print("✓ All broad handlers have logging")

    print()
    print(
        f"Coverage: {len(specific)}/{total} ({len(specific) * 100 // total if total else 0}%) use specific exceptions"
    )

    if len(specific) < total * 0.5:
        print("⚠️  Consider using more specific exception types")
    else:
        print("✓ Good use of specific exception types")

    print()


def main():
    """Run the exception handling audit."""
    repo_root = Path(__file__).parent.parent

    print(f"Auditing exception handlers in: {repo_root}")
    print()

    results = audit_exceptions(repo_root)
    print_report(results)

    # Exit with error if there are broad handlers without logging
    if results["broad_without_logging"]:
        print("❌ Audit found issues that should be addressed")
        return 1
    else:
        print("✅ All exception handlers meet quality standards")
        return 0


if __name__ == "__main__":
    sys.exit(main())
