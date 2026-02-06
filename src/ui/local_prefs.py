"""Generic localStorage persistence for UI preferences.

Provides fire-and-forget saving and deferred loading of user preferences
(filters, sort orders, layout choices) to browser localStorage.

All keys are prefixed with ``sf_prefs_`` to avoid collisions.
Values are JSON-serialised so lists, dicts, bools, and numbers round-trip safely.
"""

import asyncio
import json
import logging
from collections.abc import Callable

from nicegui import ui

logger = logging.getLogger(__name__)

_PREFIX = "sf_prefs_"

# How long to wait (seconds) before the first deferred load attempt.
_INITIAL_DELAY = 0.5

# Timeout (seconds) for each ui.run_javascript() call.
_JS_TIMEOUT = 3.0

# How many total attempts to make when loading prefs.
_MAX_RETRIES = 2

# Pause (seconds) between retry attempts.
_RETRY_DELAY = 1.0


def _storage_key(page_key: str) -> str:
    """Build the localStorage key for a page.

    Args:
        page_key: Short identifier for the page/component (e.g. "entity_browser").

    Returns:
        Prefixed key string.
    """
    return f"{_PREFIX}{page_key}"


def save_pref(page_key: str, field: str, value: object) -> None:
    """Save a single preference field to localStorage.

    This is synchronous (fire-and-forget) — the JS runs on the client
    without waiting for a result.

    Args:
        page_key: Page/component identifier.
        field: Field name inside the prefs object.
        value: JSON-serialisable value to store.
    """
    key = _storage_key(page_key)
    # Read existing prefs, merge, and write back in one JS call.
    # Use JSON.stringify for key/field to prevent injection via special chars.
    # Guard against non-object parse results (arrays, numbers, null).
    key_json = json.dumps(key)
    field_json = json.dumps(field)
    value_json = json.dumps(value)
    js = (
        f"(function(){{ "
        f"var k={key_json}; "
        f"var raw; try{{ raw=JSON.parse(localStorage.getItem(k)); }}catch(e){{}} "
        f"var p=(raw && typeof raw==='object' && !Array.isArray(raw)) ? raw : {{}}; "
        f"p[{field_json}]={value_json}; "
        f"localStorage.setItem(k, JSON.stringify(p)); "
        f"}})()"
    )
    ui.run_javascript(js)
    logger.debug("Saved pref %s.%s", page_key, field)


def save_prefs(page_key: str, fields: dict[str, object]) -> None:
    """Save multiple preference fields to localStorage at once.

    Args:
        page_key: Page/component identifier.
        fields: Mapping of field name → value.
    """
    key = _storage_key(page_key)
    key_json = json.dumps(key)
    fields_json = json.dumps(fields)
    js = (
        f"(function(){{ "
        f"var k={key_json}; "
        f"var raw; try{{ raw=JSON.parse(localStorage.getItem(k)); }}catch(e){{}} "
        f"var p=(raw && typeof raw==='object' && !Array.isArray(raw)) ? raw : {{}}; "
        f"Object.assign(p, {fields_json}); "
        f"localStorage.setItem(k, JSON.stringify(p)); "
        f"}})()"
    )
    ui.run_javascript(js)
    logger.debug("Saved %d prefs for %s", len(fields), page_key)


async def _load_prefs(page_key: str) -> dict:
    """Load all stored preferences for *page_key* with retry on timeout.

    On the first attempt a ``TimeoutError`` is logged at debug level and
    retried after :data:`_RETRY_DELAY` seconds.  Only the final failure is
    logged at warning level so cold-start timeouts do not spam the log.

    Returns:
        A dict of ``{field: value}`` or an empty dict when nothing is stored
        or the stored JSON is corrupt.
    """
    key = _storage_key(page_key)
    key_json = json.dumps(key)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = await ui.run_javascript(f"localStorage.getItem({key_json})", timeout=_JS_TIMEOUT)
        except TimeoutError:
            if attempt < _MAX_RETRIES:
                logger.debug(
                    "localStorage read timed out for %s on attempt %d/%d, retrying in %ss",
                    page_key,
                    attempt,
                    _MAX_RETRIES,
                    _RETRY_DELAY,
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue
            logger.warning(
                "localStorage read timed out for %s after %d attempts, preferences lost",
                page_key,
                _MAX_RETRIES,
            )
            return {}

        # Successfully got a response from JS
        if not raw:
            logger.debug("No saved preferences for %s", page_key)
            return {}
        try:
            prefs = json.loads(raw)
            if not isinstance(prefs, dict):
                logger.warning("Stored prefs for %s is not a dict, ignoring", page_key)
                return {}
            logger.debug("Loaded %d prefs for %s", len(prefs), page_key)
            return prefs
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Corrupt prefs for %s, ignoring: %s", page_key, exc)
            return {}

    # Unreachable — the loop always returns — but keeps the type-checker happy.
    return {}


def load_prefs_deferred(page_key: str, callback: Callable[[dict], None]) -> None:
    """Schedule async preference loading and invoke *callback* with the result.

    Call this at the end of a page's ``build()`` method.  A one-shot
    ``ui.timer`` fires shortly after the page renders, reads localStorage,
    and delivers the prefs dict to *callback*.

    Args:
        page_key: Page/component identifier.
        callback: Function receiving a dict of ``{field: value}``.
                  Called with an empty dict when nothing is stored.
    """

    async def _deferred() -> None:
        """Load preferences from localStorage and deliver to callback."""
        try:
            prefs = await _load_prefs(page_key)
            callback(prefs)
        except (RuntimeError, TimeoutError):
            logger.debug("Pref load skipped for %s (timeout or UI element destroyed)", page_key)

    ui.timer(_INITIAL_DELAY, _deferred, once=True)
    logger.debug("Scheduled deferred pref load for %s", page_key)
