"""Embedded Python runtime manager.

Inspired by NVDA's approach: the application ships with or can locate a
Python interpreter.  Addons install their own dependencies into a
per-user virtualenv under ``%APPDATA%/AIVoiceStudio/addon_env``.

All pip operations run through this managed environment so the main
application's packages are never touched.

Usage::

    from ai_voice_studio.python_runtime import PythonRuntime
    rt = PythonRuntime()
    rt.ensure_pip()
    rt.pip_install("pyttsx3")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from typing import Callable, Dict, List, Optional

from .. import paths

log = logging.getLogger(__name__)

_ADDON_ENV_DIR = "addon_env"
_MANIFEST_NAME = "installed_packages.json"


def _python_exe_for_env(env_dir: str) -> str:
    """Return the Python executable inside a virtualenv."""
    if sys.platform == "win32":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python3")


def _pip_exe_for_env(env_dir: str) -> str:
    """Return the pip executable inside a virtualenv."""
    if sys.platform == "win32":
        return os.path.join(env_dir, "Scripts", "pip.exe")
    return os.path.join(env_dir, "bin", "pip3")


def _find_system_python() -> str:
    """Find a usable system Python interpreter."""
    if not getattr(sys, "frozen", False):
        # Development mode: use the running Python
        return sys.executable
    # In a PyInstaller bundle, sys.executable is the frozen EXE
    for name in ("python3", "python", "py"):
        path = shutil.which(name)
        if path:
            return path
    # Last resort: try common Windows install paths
    for candidate in (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python",
                     "Python313", "python.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python",
                     "Python312", "python.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python",
                     "Python311", "python.exe"),
        r"C:\Python313\python.exe",
        r"C:\Python312\python.exe",
        r"C:\Python311\python.exe",
    ):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No Python interpreter found. Install Python 3.11+ and ensure it "
        "is on the PATH, or add it to a standard location."
    )


class PythonRuntime:
    """Manages the embedded/managed Python environment for AI Voice Studio.

    The environment lives under ``%APPDATA%/AIVoiceStudio/addon_env``.
    Addons install their dependencies here via pip, keeping them isolated
    from the main application's packages.
    """

    def __init__(self, env_dir: str | None = None):
        self._env_dir = env_dir or os.path.join(
            paths.user_data_dir(), _ADDON_ENV_DIR
        )
        self._system_python: str | None = None
        self._lock = threading.RLock()

    # -- properties ---------------------------------------------------------

    @property
    def env_dir(self) -> str:
        return self._env_dir

    @property
    def python_exe(self) -> str:
        return _python_exe_for_env(self._env_dir)

    @property
    def pip_exe(self) -> str:
        return _pip_exe_for_env(self._env_dir)

    @property
    def is_created(self) -> bool:
        return os.path.isfile(self.python_exe)

    # -- system python ------------------------------------------------------

    @property
    def system_python(self) -> str:
        if self._system_python is None:
            self._system_python = _find_system_python()
        return self._system_python

    # -- environment lifecycle ----------------------------------------------

    def ensure_env(self) -> str:
        """Create the virtualenv if it does not exist. Returns the env dir."""
        if self.is_created:
            return self._env_dir
        os.makedirs(self._env_dir, exist_ok=True)
        log.info("Creating addon virtualenv at %s", self._env_dir)
        system_py = self.system_python
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                [system_py, "-m", "venv", self._env_dir, "--clear"],
                capture_output=True, text=True, check=True,
                cwd=tempfile.gettempdir(), creationflags=creationflags,
            )
            log.info("Addon virtualenv created successfully")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create virtualenv: {exc.stderr or exc.stdout}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to start Python: {exc}") from exc
        return self._env_dir

    def ensure_pip(self) -> str:
        """Ensure pip is available in the virtualenv. Returns pip path."""
        self.ensure_env()
        if os.path.isfile(self.pip_exe):
            return self.pip_exe
        log.info("Bootstrapping pip in addon virtualenv")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            [self.python_exe, "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, check=True,
            cwd=tempfile.gettempdir(), creationflags=creationflags,
        )
        return self.pip_exe

    def remove_env(self) -> None:
        """Remove the entire virtualenv."""
        if os.path.isdir(self._env_dir):
            shutil.rmtree(self._env_dir, ignore_errors=True)
            log.info("Addon virtualenv removed: %s", self._env_dir)

    # -- pip operations -----------------------------------------------------

    def pip_install(
        self,
        packages: List[str] | str,
        progress: Callable[[str, int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Dict[str, str]:
        """Install packages into the addon virtualenv.

        Parameters
        ----------
        packages:
            Package names with optional version specifiers,
            e.g. ``["pyttsx3", "edge-tts>=6.0"]`` or ``"pyttsx3"``.
        progress:
            Optional callback ``(message, done, total)`` for progress updates.
        cancel_event:
            Optional event to cancel the installation.

        Returns
        -------
        dict:
            ``{"ok": True/False, "output": "...", "error": "..."}``
        """
        if isinstance(packages, str):
            packages = [packages]
        # Users often type a full command like "pip install numpy" or
        # "install numpy" into the developer panel; strip those tokens so the
        # command does not become "pip install ... pip install numpy".
        cleaned: List[str] = []
        for entry in packages:
            for token in str(entry).split():
                if token.lower() in ("pip", "install", "--user"):
                    continue
                if token.startswith("-") and token not in ("-U", "--upgrade", "--no-deps"):
                    continue
                cleaned.append(token)
        packages = cleaned or []
        pip = self.ensure_pip()
        cmd = [pip, "install", "--no-warn-script-location"] + packages
        log.info("Running pip install: %s", " ".join(cmd))

        if progress:
            progress("Installing packages...", 0, 0)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=tempfile.gettempdir(),
                creationflags=creationflags,
            )
        except OSError as exc:
            return {"ok": False, "output": "", "error": f"Could not start pip: {exc}"}

        output_lines: List[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if progress:
                progress(line, 0, 0)  # pulsing
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                return {"ok": False, "output": "\n".join(output_lines),
                        "error": "Installation cancelled."}

        proc.wait()
        combined = "\n".join(output_lines)
        if proc.returncode != 0:
            log.error("pip install failed (exit %d): %s", proc.returncode, combined[-500:])
            return {"ok": False, "output": combined,
                    "error": f"pip install failed (exit {proc.returncode})"}
        log.info("pip install succeeded")
        if progress:
            progress("Installation complete.", 1, 1)
        return {"ok": True, "output": combined, "error": ""}

    def pip_uninstall(
        self,
        packages: List[str] | str,
    ) -> Dict[str, str]:
        """Uninstall packages from the addon virtualenv."""
        if isinstance(packages, str):
            packages = [packages]
        if not self.is_created:
            return {"ok": True, "output": "", "error": ""}
        pip = self.pip_exe
        cmd = [pip, "uninstall", "-y"] + packages
        log.info("Running pip uninstall: %s", " ".join(cmd))
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=tempfile.gettempdir(), creationflags=creationflags,
            )
            return {"ok": result.returncode == 0,
                    "output": result.stdout, "error": result.stderr}
        except OSError as exc:
            return {"ok": False, "output": "", "error": str(exc)}

    def pip_list(self) -> List[Dict[str, str]]:
        """List installed packages in the addon virtualenv."""
        if not self.is_created:
            return []
        pip = self.pip_exe
        cmd = [pip, "list", "--format=json"]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=tempfile.gettempdir(), creationflags=creationflags,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def run_in_env(self, script: str) -> subprocess.CompletedProcess:
        """Run a Python script inside the managed virtualenv."""
        if not self.is_created:
            self.ensure_env()
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(
            [self.python_exe, "-c", script],
            capture_output=True, text=True,
            cwd=tempfile.gettempdir(), creationflags=creationflags,
        )

    def add_to_sys_path(self) -> None:
        """Add the virtualenv's site-packages to sys.path at runtime.

        This lets the main process import packages installed by addons.
        Call this after pip install completes if you need immediate access.
        """
        if not self.is_created:
            return
        # Determine the site-packages path
        result = self.run_in_env(
            "import sysconfig; print(sysconfig.get_path('purelib'))"
        )
        if result.returncode == 0:
            site_packages = result.stdout.strip()
            if site_packages and os.path.isdir(site_packages):
                if site_packages not in sys.path:
                    sys.path.insert(0, site_packages)
                    log.info("Added addon site-packages to sys.path: %s", site_packages)

    def get_executable(self) -> str:
        """Return the Python executable path for the addon environment."""
        self.ensure_env()
        return self.python_exe


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_default_runtime: PythonRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> PythonRuntime:
    """Return the default PythonRuntime instance (lazy singleton)."""
    global _default_runtime
    if _default_runtime is None:
        with _runtime_lock:
            if _default_runtime is None:
                _default_runtime = PythonRuntime()
    return _default_runtime
