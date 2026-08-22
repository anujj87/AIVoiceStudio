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
    except Exception:  # noqa: BLE001
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

def addon_env_dir() -> str:
    d = os.path.join(user_data_dir(), "addon_env")
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
