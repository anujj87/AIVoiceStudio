"""Small shared helpers used across the application."""

from __future__ import annotations

import os
import re
import shutil
import sys

_ILLEGAL_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def sanitize_filename(name: str, max_len: int = 60) -> str:
    """Turn arbitrary text into a safe file name (keeps spaces and unicode)."""
    name = _ILLEGAL_FILENAME.sub("_", name).strip().strip(".")
    name = _MULTI_SPACE.sub(" ", name)
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[: max_len - 1].rstrip()
    return name


def sanitize_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def format_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def find_python() -> str:
    """Return a Python executable that has pip available.

    In a PyInstaller bundle ``sys.executable`` is the bundled EXE which
    does not contain pip.  We always use the managed PythonRuntime
    virtualenv under ``%%APPDATA%%/AIVoiceStudio/addon_env`` so that
    packages are isolated from any system Python the user may install.
    The venv is created on first use if it does not yet exist.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    # Always use the managed virtualenv (created on demand).
    try:
        from .python_runtime import get_runtime
        rt = get_runtime()
        if not rt.is_created:
            rt.ensure_env()  # creates venv using system Python
        if os.path.isfile(rt.python_exe):
            return rt.python_exe
    except Exception:  # noqa: BLE001
        pass
    raise FileNotFoundError(
        "Could not create the managed Python environment.  "
        "Make sure Python 3.10+ is installed on this system."
    )


def worker_command(module: str, *args: str) -> list[str]:
    """Build a command for a worker in development or a frozen app."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--aivs-worker", module, *args]
    return [sys.executable, "-m", module, *args]
