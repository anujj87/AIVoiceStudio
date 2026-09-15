"""Cached, non-blocking "is this package installed in the managed venv?" probes.

The Settings dialog builds every category up front, and each OmniVoice status
label plus each pip-installed engine used to start its own ``python -c`` inside
the managed virtualenv while the dialog was being built.  A dozen interpreter
start-ups (each a second or more) is what made opening Settings feel sluggish
next to NVDA's settings dialog.

This module answers the question from an in-memory cache: a miss returns
``None`` and starts *one* background probe, so the GUI keeps drawing and fills
installed/not-installed labels in when the answer arrives.  Results are
invalidated explicitly after an install or uninstall.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_CACHE_TTL = 300.0

#: package name -> (timestamp, version or None)
_cache: Dict[str, Tuple[float, Optional[str]]] = {}
_loading: set[str] = set()
_listeners: Dict[str, List[Callable[[Optional[str]], None]]] = {}
_lock = threading.RLock()


def _env_exists() -> bool:
    """True when the managed virtualenv has been created."""
    try:
        from .python_runtime import get_runtime  # noqa: PLC0415

        return bool(get_runtime().is_created)
    except Exception:  # noqa: BLE001
        return False


def _notify(package: str, version: Optional[str]) -> None:
    with _lock:
        listeners = _listeners.pop(package, [])
    for callback in listeners:
        try:
            callback(version)
        except Exception:  # noqa: BLE001
            log.debug("Package probe listener failed", exc_info=True)


def _job(package: str) -> None:
    found: Optional[str] = None
    try:
        from .python_runtime import get_runtime  # noqa: PLC0415

        runtime = get_runtime()
        if runtime.is_created:
            script = (
                "import importlib.metadata as m\n"
                f"try:\n print(m.version({package!r}))\nexcept Exception:\n pass\n"
            )
            result = runtime.run_in_env(script)
            text = (result.stdout or "").strip()
            if result.returncode == 0 and text:
                found = text
    except Exception:  # noqa: BLE001
        log.debug("Probe for %s failed", package, exc_info=True)
    with _lock:
        _cache[package] = (time.time(), found)
        _loading.discard(package)
    _notify(package, found)


def request(package: str, on_ready: Optional[Callable[[Optional[str]], None]] = None) -> None:
    """Probe ``package`` once.

    ``on_ready(version_or_None)`` runs on the worker thread (GUI callers wrap
    it in ``wx.CallAfter``).  A cached value is delivered immediately.
    """
    if on_ready is not None:
        with _lock:
            _listeners.setdefault(package, []).append(on_ready)
    if not _env_exists():
        with _lock:
            _cache[package] = (time.time(), None)
        _notify(package, None)
        return
    cached: Optional[str] = None
    fresh = False
    with _lock:
        entry = _cache.get(package)
        if entry is not None and (time.time() - entry[0]) < _CACHE_TTL:
            cached, fresh = entry[1], True
        loading = package in _loading
        if not fresh and not loading:
            _loading.add(package)
    if fresh:
        _notify(package, cached)
        return
    if loading:
        return  # the running probe will notify the listener
    threading.Thread(
        target=_job, args=(package,), daemon=True, name=f"aivs-probe-{package}"
    ).start()


def version(package: str) -> Optional[str]:
    """Installed version of ``package`` in the managed venv.

    ``None`` means "not installed" *or* "not known yet" - the cache fills in
    from the background probe.  Never blocks.
    """
    with _lock:
        entry = _cache.get(package)
    if entry is None:
        request(package)
        return None
    if (time.time() - entry[0]) >= _CACHE_TTL:
        request(package)
    return entry[1]


def installed(package: str) -> bool:
    return bool(version(package))


def is_known(package: str) -> bool:
    """True when a fresh cached answer exists for ``package``.

    Callers use this to avoid attaching a listener that would fire
    immediately from the cache (a callback that refreshes the caller would
    then recurse).
    """
    with _lock:
        entry = _cache.get(package)
    return entry is not None and (time.time() - entry[0]) < _CACHE_TTL


def invalidate(*packages: str) -> None:
    """Forget cached results (call after installing or removing a package)."""
    with _lock:
        if not packages:
            _cache.clear()
            return
        for package in packages:
            _cache.pop(package, None)


def warm(packages: Optional[List[str]] = None) -> None:
    """Start background probes so the first panel that asks is already answered."""
    if not _env_exists():
        return
    if packages is None:
        packages = []
        try:
            from .tts import catalog  # noqa: PLC0415

            for entry in catalog.get_tts_list():
                package = entry.get("requires_package")
                if package and package not in packages:
                    packages.append(package)
        except Exception:  # noqa: BLE001
            log.debug("Could not list pip-installed engines", exc_info=True)
    for package in packages:
        request(package)
