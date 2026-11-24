"""User data directory resolution.

All user data lives under ``%APPDATA%\\AIVoiceStudio`` on Windows (falls back
to ``~/.aivoicestudio`` elsewhere) so the installed program folder stays
read-only and models/ffmpeg can be managed by the user (SPEC 2, SPEC 3.6).
"""

from __future__ import annotations

import os
import sys

from .constants import (
    FFMPEG_DIR_NAME,
    LOGS_DIR_NAME,
    MODELS_DIR_NAME,
    PROJECTS_DIR_NAME,
    SETTINGS_FILE_NAME,
    USER_DATA_DIR_NAME,
)


def user_data_dir() -> str:
    """Return (and create) the per-user application data directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        root = os.path.join(base, USER_DATA_DIR_NAME)
    else:
        root = os.path.join(os.path.expanduser("~"), ".aivoicestudio")
    os.makedirs(root, exist_ok=True)
    return root


def _custom_dir(key: str) -> str:
    """Return the user-configured custom folder for ``key`` ("" if unset).

    Reads ``settings.paths.<key>``. The import is deferred so ``settings``
    (which imports this module) can be loaded first without a cycle.
    """
    try:
        from .settings import Settings  # lazy: settings.py imports this module

        value = Settings().get(f"paths.{key}", "")
    except Exception:
        return ""
    if value and os.path.isabs(value):
        return value
    return ""


def models_dir() -> str:
    d = _custom_dir("models_dir") or os.path.join(user_data_dir(), MODELS_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def recordings_dir() -> str:
    d = _custom_dir("recordings_dir") or os.path.join(user_data_dir(), PROJECTS_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def projects_dir() -> str:
    """Alias of :func:`recordings_dir` (projects hold the recorded audio)."""
    return recordings_dir()


def ffmpeg_dir() -> str:
    d = os.path.join(user_data_dir(), FFMPEG_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def runtime_dir(kind: str) -> str:
    """Return (and create) the per-user folder for an optional compute
    runtime (e.g. "cuda" for the GPU dependency)."""
    d = os.path.join(user_data_dir(), "runtime", kind)
    os.makedirs(d, exist_ok=True)
    return d


def app_dir() -> str:
    """The folder that holds the frozen executable ("" in a source checkout)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return ""


#: The interpreter inside a shipped/staged runtime folder.
PYTHON_EXE_NAME = "python.exe"


def shipped_python_runtime() -> str:
    """Folder of the Python runtime shipped inside this installation, or "".

    The build stages a full CPython under ``python-runtime`` next to the
    executable (see ``packaging/stage_python_runtime.py``) so the application
    can create its per-engine and per-model virtualenvs on a machine that has
    no Python installed at all.  ``python -m venv`` needs a *full* interpreter -
    one that still carries ``Lib/venv`` and ``Lib/ensurepip`` - which is exactly
    what the PyInstaller-frozen executable is not, so the runtime is shipped as
    a real file tree rather than taken from the bundle.

    A source checkout, and a build made without the runtime staged, return "".
    """
    candidates: list[str] = []
    root = app_dir()
    if root:
        candidates.append(os.path.join(root, "python-runtime"))
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(os.path.join(meipass, "python-runtime"))
    for folder in candidates:
        if os.path.isfile(os.path.join(folder, PYTHON_EXE_NAME)):
            return folder
    return ""


def staged_python_runtime() -> str:
    """Folder of the per-user copy of that runtime (made on first use).

    Every environment the application creates is a *virtual* one, and a
    virtualenv records the absolute path of the interpreter it runs on top of
    (``pyvenv.cfg``).  The environments live under ``%APPDATA%\\AIVoiceStudio``
    so that uninstalling never throws away gigabytes of packages - which means
    they also outlive the installation that made them, and a base interpreter
    inside the program folder turns every one of them into a broken
    environment the moment somebody reinstalls, moves or upgrades the
    application.  A copy that lives in the user's own data folder cannot move,
    so the environments built from it keep working
    (see ``python_runtime._stage_python_runtime``).
    """
    return os.path.join(user_data_dir(), "python_runtime")


def bundled_python_exe() -> str:
    """Path of a Python runtime the application may use, or "".

    The shipped runtime first, then the per-user copy of it, so that a build
    made without the runtime staged (and a source checkout) return "" and the
    caller falls back to a Python found on the machine.
    """
    for folder in (shipped_python_runtime(), staged_python_runtime()):
        exe = os.path.join(folder, PYTHON_EXE_NAME)
        if os.path.isfile(exe):
            return exe
    return ""

def logs_dir() -> str:
    d = os.path.join(user_data_dir(), LOGS_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def settings_file() -> str:
    return os.path.join(user_data_dir(), SETTINGS_FILE_NAME)


def models_state_file() -> str:
    return os.path.join(user_data_dir(), "models.json")


def log_file() -> str:
    return os.path.join(logs_dir(), "app.log")

def addons_dir() -> str:
    d = os.path.join(user_data_dir(), "addons")
    os.makedirs(d, exist_ok=True)
    return d


def project_dir(name: str) -> str:
    safe = _safe_dir_name(name)
    d = os.path.join(recordings_dir(), safe)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_dir_name(name: str) -> str:
    import re

    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    return safe or "untitled"
