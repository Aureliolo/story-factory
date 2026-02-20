"""Backup and diagnostic helpers for settings persistence.

Provides:
- Pre-save backup creation (.bak file)
- Recovery from backup when primary file is missing/corrupt
- Change logging for audit trail during load/save
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _create_settings_backup(settings_path: Path) -> bool:
    """Copy settings.json to settings.json.bak before writing.

    Skips backup if the source file is missing or empty (nothing to preserve).
    Failures are logged but never raised — backup is best-effort.

    Args:
        settings_path: Path to the primary settings file.

    Returns:
        True if a backup was created, False otherwise.
    """
    try:
        if not settings_path.exists():
            logger.debug("No settings file to back up at %s", settings_path)
            return False
        if settings_path.stat().st_size == 0:
            logger.debug("Settings file is empty, skipping backup")
            return False
        backup_path = settings_path.with_suffix(".json.bak")
        shutil.copy2(settings_path, backup_path)
        logger.debug("Created settings backup at %s", backup_path)
        return True
    except OSError as e:
        logger.warning("Failed to create settings backup: %s", e)
        return False


def _recover_from_backup(settings_path: Path) -> dict[str, Any] | None:
    """Attempt to recover settings from the .bak file.

    Used when the primary settings file is missing, empty, or corrupt.

    Args:
        settings_path: Path to the primary settings file.

    Returns:
        Parsed settings dict if recovery succeeded, None otherwise.
    """
    backup_path = settings_path.with_suffix(".json.bak")
    try:
        if not backup_path.exists():
            logger.debug("No backup file found at %s", backup_path)
            return None
        if backup_path.stat().st_size == 0:
            logger.debug("Backup file is empty, cannot recover")
            return None
        with open(backup_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "Backup file contains %s instead of dict, cannot recover",
                type(data).__name__,
            )
            return None
        logger.info(
            "Recovered %d settings from backup file %s",
            len(data),
            backup_path,
        )
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to recover from backup: %s", e)
        return None


def _log_settings_changes(
    original: dict[str, Any],
    final: dict[str, Any],
    label: str,
) -> int:
    """Log every key that changed between two settings snapshots.

    Compares flat keys and one level of nested dict keys (dot-notation).
    Only logs at INFO level so it shows up in normal operation.

    Args:
        original: Settings dict before the operation.
        final: Settings dict after the operation.
        label: Human-readable label for the log messages (e.g. "merge", "validation").

    Returns:
        Number of changes detected.
    """
    all_keys = set(original) | set(final)
    changes = 0

    for key in sorted(all_keys):
        old_val = original.get(key)
        new_val = final.get(key)

        if key not in original:
            logger.info("[%s] added %s = %r", label, key, new_val)
            changes += 1
        elif key not in final:
            logger.info("[%s] removed %s (was %r)", label, key, old_val)
            changes += 1
        elif isinstance(old_val, dict) and isinstance(new_val, dict):
            # Compare one level of nested dict keys
            sub_keys = set(old_val) | set(new_val)
            for sub_key in sorted(sub_keys):
                old_sub = old_val.get(sub_key)
                new_sub = new_val.get(sub_key)
                if old_sub != new_sub:
                    logger.info(
                        "[%s] changed %s.%s: %r -> %r",
                        label,
                        key,
                        sub_key,
                        old_sub,
                        new_sub,
                    )
                    changes += 1
        elif old_val != new_val:
            logger.info("[%s] changed %s: %r -> %r", label, key, old_val, new_val)
            changes += 1

    if changes:
        logger.info("[%s] total changes: %d", label, changes)
    else:
        logger.debug("[%s] no changes detected", label)

    return changes
