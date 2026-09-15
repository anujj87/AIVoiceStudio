"""Small shared helpers used across the application."""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import sys
import threading
import time

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


# ---------------------------------------------------------------------------
# Child processes started from background threads
# ---------------------------------------------------------------------------
#: Children still running, so the exit hook below can reach them.
_children: "set[subprocess.Popen]" = set()
_children_lock = threading.Lock()
_drain_registered = False


def run_tracked(cmd, *, timeout=None, **kwargs) -> subprocess.CompletedProcess:
    """Run ``cmd`` and keep the child process reachable at interpreter exit.

    Voice discovery and the managed-venv package probes run child processes
    from daemon threads.  A daemon thread caught inside ``subprocess`` while
    Python finalises can crash the process on exit (a segmentation fault when
    the app was closed during voice discovery), so such children are
    registered here and drained by :func:`drain_children` before shutdown.

    Same contract as ``subprocess.run`` with ``capture_output=True, text=True``.
    """
    global _drain_registered
    if not _drain_registered:
        _drain_registered = True
        atexit.register(drain_children)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
    )
    with _children_lock:
        _children.add(proc)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
    finally:
        with _children_lock:
            _children.discard(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def drain_children(timeout: float = 3.0) -> None:
    """Let tracked children finish, then kill whatever is still running.

    Called at interpreter shutdown: waiting a moment lets a quick probe finish
    on its own (the clean path), and killing the rest stops a hung child from
    leaving its worker thread mid-``subprocess`` during finalisation.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _children_lock:
            if not _children:
                return
        time.sleep(0.05)
    with _children_lock:
        procs = list(_children)
    for proc in procs:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        with _children_lock:
            if not _children:
                break
        time.sleep(0.05)
