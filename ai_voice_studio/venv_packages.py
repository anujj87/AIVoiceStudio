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
import os
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_CACHE_TTL = 300.0

#: ``(env, package)`` -> (timestamp, version or None).  ``env`` is the
#: *environment* id of a per-TTS virtualenv (engines that share one
#: environment share the key), or ``""`` for the shared addon environment.
_cache: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}
_loading: set[Tuple[str, str]] = set()
_listeners: Dict[Tuple[str, str], List[Callable[[Optional[str]], None]]] = {}
_lock = threading.RLock()


#: ``site-packages folder`` -> (timestamp, {normalized name: version}).  Read
#: straight from the disk so the answer is available on the first question and
#: survives an app restart.
_disk_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}


def _normalize(name: str) -> str:
    """PEP 503 normalisation (``f5_tts`` == ``f5-tts`` == ``F5.TTS``)."""
    parts = (part for part in re.split(r"[-_.]+", (name or "").strip().lower()))
    return "-".join(part for part in parts if part)


def _disk_distributions(engine: Optional[str] = None) -> Dict[str, str]:
    """``{normalized distribution name: version}`` from an env's site-packages.

    A plain directory listing: no subprocess, no interpreter start-up, so this
    can run on the GUI thread.  The listing is cached per folder with the
    folder's modification time as the key, which changes whenever pip adds or
    removes a package.
    """
    try:
        from .python_runtime import get_runtime  # noqa: PLC0415

        folders = get_runtime(engine).site_packages_dirs
    except Exception:  # noqa: BLE001
        return {}
    found: Dict[str, str] = {}
    for folder in folders:
        try:
            stamp = os.path.getmtime(folder)
        except OSError:
            continue
        with _lock:
            entry = _disk_cache.get(folder)
        if entry is not None and entry[0] == stamp:
            found.update(entry[1])
            continue
        listed: Dict[str, str] = {}
        try:
            for name in os.listdir(folder):
                if not name.endswith(".dist-info"):
                    continue
                distribution, _, version_text = name[: -len(".dist-info")].partition("-")
                listed[_normalize(distribution)] = version_text
        except OSError:
            continue
        with _lock:
            _disk_cache[folder] = (stamp, listed)
        found.update(listed)
    return found


def _disk_version(package: str, engine: Optional[str] = None) -> Optional[str]:
    return _disk_distributions(engine).get(_normalize(package))


def _env_exists(engine: Optional[str] = None) -> bool:
    """True when the managed virtualenv has been created."""
    try:
        from .python_runtime import get_runtime  # noqa: PLC0415

        return bool(get_runtime(engine).is_created)
    except Exception:  # noqa: BLE001
        return False


def _notify(key: Tuple[str, str], version: Optional[str]) -> None:
    with _lock:
        listeners = _listeners.pop(key, [])
    for callback in listeners:
        try:
            callback(version)
        except Exception:  # noqa: BLE001
            log.debug("Package probe listener failed", exc_info=True)


def _job(key: Tuple[str, str]) -> None:
    env, package = key
    found: Optional[str] = None
    try:
        from .python_runtime import get_runtime  # noqa: PLC0415

        runtime = get_runtime(env or None)
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
        _cache[key] = (time.time(), found)
        _loading.discard(key)
    _notify(key, found)


def _engine_key(engine: Optional[str]) -> str:
    """Cache key of an engine.

    Engines that share one virtualenv (the two OmniVoice ones) also share
    their answers, so installing either of them refreshes the other.
    """
    if not engine:
        return ""
    try:
        from .python_runtime import environment_id  # noqa: PLC0415

        return environment_id(engine)
    except Exception:  # noqa: BLE001
        return engine


def _key(package: str, engine: Optional[str]) -> Tuple[str, str]:
    return (_engine_key(engine), package)


def request(
    package: str,
    on_ready: Optional[Callable[[Optional[str]], None]] = None,
    engine: Optional[str] = None,
) -> None:
    """Probe ``package`` once.

    ``on_ready(version_or_None)`` runs on the worker thread (GUI callers wrap
    it in ``wx.CallAfter``).  A cached value is delivered immediately.
    ``engine`` names a per-TTS environment; without it the shared addon
    environment is probed.
    """
    key = _key(package, engine)
    if on_ready is not None:
        with _lock:
            _listeners.setdefault(key, []).append(on_ready)
    if not _env_exists(engine):
        with _lock:
            _cache[key] = (time.time(), None)
        _notify(key, None)
        return
    cached: Optional[str] = None
    fresh = False
    with _lock:
        entry = _cache.get(key)
        if entry is not None and (time.time() - entry[0]) < _CACHE_TTL:
            cached, fresh = entry[1], True
        loading = key in _loading
        if not fresh and not loading:
            _loading.add(key)
    if fresh:
        _notify(key, cached)
        return
    if loading:
        return  # the running probe will notify the listener
    threading.Thread(
        target=_job, args=(key,), daemon=True,
        name=f"aivs-probe-{engine or 'addon'}-{package}",
    ).start()


def version(package: str, engine: Optional[str] = None) -> Optional[str]:
    """Installed version of ``package`` in the managed venv.

    Before the background probe has answered, the version is read straight
    from the environment's ``site-packages`` on disk (see
    ``_disk_distributions``), so a cold cache answers with the truth instead of
    ``None``.  ``None`` therefore means "not installed".  Never blocks.
    """
    key = _key(package, engine)
    with _lock:
        entry = _cache.get(key)
    if entry is None:
        request(package, engine=engine)
        return _disk_version(package, engine)
    if (time.time() - entry[0]) >= _CACHE_TTL:
        request(package, engine=engine)
    return entry[1]


def installed(package: str, engine: Optional[str] = None) -> bool:
    return bool(version(package, engine))


def package_present(package: str, engine: Optional[str] = None) -> bool:
    """Synchronous, cache-free "is ``package`` in this environment?"

    Answers from ``site-packages`` on disk.  Used where the answer must be
    right the *first* time - most importantly to choose which interpreter an
    engine's worker subprocess runs with, because a worker started in the
    wrong environment reports a baffling "No module named 'torch'" instead of
    the engine's own error.
    """
    name = _normalize(package)
    if not name:
        return False
    if name in _disk_distributions(engine):
        return True
    # A bare module folder or single-file module also counts (dist-info can be
    # missing in a hand-patched environment).
    try:
        from .python_runtime import get_runtime  # noqa: PLC0415

        folders = get_runtime(engine).site_packages_dirs
    except Exception:  # noqa: BLE001
        return False
    stem = name.replace("-", "_")
    for folder in folders:
        if os.path.isdir(os.path.join(folder, stem)):
            return True
        if os.path.isfile(os.path.join(folder, stem + ".py")):
            return True
    return False


def is_known(package: str, engine: Optional[str] = None) -> bool:
    """True when a fresh cached answer exists for ``package``.

    Callers use this to avoid attaching a listener that would fire
    immediately from the cache (a callback that refreshes the caller would
    then recurse).
    """
    with _lock:
        entry = _cache.get(_key(package, engine))
    return entry is not None and (time.time() - entry[0]) < _CACHE_TTL


def invalidate(*packages: str, engine: Optional[str] = None) -> None:
    """Forget cached results (call after installing or removing a package).

    Without ``engine`` the shared environment's entries are dropped; with one,
    that engine's entries (plus, for every engine, the shared ones when no
    packages are named - tests rely on the blanket clear).
    """
    with _lock:
        _disk_cache.clear()
        if not packages:
            if engine:
                for key in [k for k in _cache if k[0] == engine]:
                    _cache.pop(key, None)
            else:
                _cache.clear()
            return
        for package in packages:
            _cache.pop(_key(package, engine), None)
            _cache.pop(_key(package, None), None)


def warm(packages: Optional[List[str]] = None) -> None:
    """Start background probes so the first panel that asks is already answered.

    The shared addon environment is probed for every catalog package (that is
    where the OmniVoice engines live); per-TTS environments are probed on
    demand by the panels that list those engines' voices.
    """
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
