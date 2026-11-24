"""Embedded Python runtime manager.

Inspired by NVDA's approach: the application ships with or can locate a
Python interpreter.  Addons install their own dependencies into a
per-user virtualenv under ``%APPDATA%/AIVoiceStudio/addon_env``.

All pip operations run through this managed environment so the main
application's packages are never touched.

**Every environment outlives the installation that created it.**  They live in
``%APPDATA%/AIVoiceStudio`` on purpose - uninstalling must never throw away the
gigabytes of packages and models a user has downloaded - and a virtualenv runs
*on top of* the interpreter it was built from, whose absolute path it records
in its ``pyvenv.cfg``.  That makes a base interpreter inside the program folder
a trap: reinstalling, moving or upgrading the application leaves every
environment pointing at a folder that is gone, where its ``python.exe`` prints
``did not find executable at '<path>': The system cannot find the path
specified.`` and exits with code 103.  Two rules follow, and both live here:

* environments are built from a **per-user copy** of the bundled runtime
  (``_stage_python_runtime``), a path that cannot move; and
* an environment built from a Python that has gone anyway is **re-pointed** at
  one that exists (``PythonRuntime.repair_env``) instead of being rebuilt, so
  the packages already installed in it survive.

Usage::

    from ai_voice_studio.python_runtime import PythonRuntime
    rt = PythonRuntime()
    rt.ensure_pip()
    rt.pip_install("pyttsx3")
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable

from .. import paths

log = logging.getLogger(__name__)

#: How long a helper command may run before it is abandoned.  Without these a
#: hung ``venv``/``pip``/probe call held its caller's thread for ever - the GUI
#: stayed alive but the operation it started could never finish or be cancelled.
#: Creating an environment and bootstrapping pip are slow on a cold disk; the
#: probes are quick unless something is badly wrong.
VENV_CREATE_TIMEOUT = 300
ENSURE_PIP_TIMEOUT = 300
SHORT_COMMAND_TIMEOUT = 120
PROBE_TIMEOUT = 120

_ADDON_ENV_DIR = "addon_env"
#: Per-TTS environments live in a sibling folder: one virtualenv per engine,
#: so engines with conflicting dependencies (different torch or transformers
#: releases) never fight over the same site-packages.
_TTS_ENVS_DIR = "tts_envs"
_MANIFEST_NAME = "installed_packages.json"

#: Engines that share one environment.  The two OmniVoice engines are the same
#: model behind two front-ends (the direct Triton runner and the HTTP server):
#: both want the same ``omnivoice`` base package and the same CUDA PyTorch, so
#: a second environment would only duplicate several gigabytes of wheels.
#: Every other TTS engine gets an environment of its own.
SHARED_ENVIRONMENTS: dict[str, str] = {
    "omnivoice_server": "omnivoice",
}

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

#: ``...\Microsoft\WindowsApps\python.exe`` (and ``python3.exe``) are not
#: interpreters: they are zero-byte *App Execution Aliases* that open the
#: Microsoft Store.  Running one prints "Python was not found; run without
#: arguments to install from the Microsoft Store" and exits with code 9009.
#: ``shutil.which`` happily returns them because they *are* on PATH, so a
#: candidate under this folder must never be handed to ``python -m venv``.
_STORE_ALIAS_DIR = os.path.join("microsoft", "windowsapps")

#: The oldest interpreter the application can build its environments with:
#: the code uses 3.10 syntax (``str | None``) and the STT backends want 3.10+.
_MINIMUM_PYTHON: tuple[int, int] = (3, 10)

#: Pulls the ``major.minor`` out of e.g. ``...\Python313\python.exe``.
_VERSION_IN_PATH = re.compile(r"python(\d)(\d+)\\python\.exe$")

def _creation_flags() -> int:
    """Flags that keep a probe from flashing a console window."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0

def _is_store_alias(path: str) -> bool:
    """True for the Microsoft Store ``python``/``python3`` aliases."""
    return _STORE_ALIAS_DIR in path.replace("/", "\\").lower()

#: Wording pip uses when a run failed for a reason the user can act on (a
#: source build with no compiler, a package with no wheel for this Python, a
#: version that does not exist).  Ordered most specific first: pip's *last*
#: line is usually just the package name, and its generic ``error:`` lines say
#: much less than these.
_PIP_FAILURE_PATTERNS: tuple[str, ...] = (
    "no matching distribution",
    "could not find a version",
    "failed building wheel",
    "microsoft visual c++",
    "requires a c++ compiler",
    "is not a supported wheel",
    "error:",
)

def _pip_failure_line(output_lines: list[str]) -> str:
    """The most telling line of a failed pip run, or "" when there is none.

    Scanning for pip's own error wording turns a bare "pip install failed
    (exit 1)" into a message that names what actually went wrong - the package
    that could not be built, or the version that does not exist - so the user
    has something to act on instead of only a status code.
    """
    for pattern in _PIP_FAILURE_PATTERNS:
        for line in reversed(output_lines):
            stripped = line.strip()
            if stripped and pattern in stripped.lower():
                return stripped
    for line in reversed(output_lines):
        if line.strip():
            return line.strip()
    return ""

def _usable_python(path: str, minimum: tuple[int, int] = _MINIMUM_PYTHON
                   ) -> tuple[int, int] | None:
    """Return ``(major, minor)`` when ``path`` is a real, usable interpreter.

    A candidate must actually run *and* carry the standard library the
    environment manager drives (``venv`` and ``ensurepip``).  That is what
    rejects the Microsoft Store aliases: they exist on disk and sit on PATH,
    but they are not Python - running one only prints the "install from the
    Store" notice and exits 9009.  ``None`` means "not usable; try the next
    candidate".
    """
    if not path or not os.path.isfile(path) or _is_store_alias(path):
        return None
    probe = "import sys, venv, ensurepip; print('%d.%d' % sys.version_info[:2])"
    try:
        result = subprocess.run(
            [path, "-c", probe], capture_output=True, text=True,
            cwd=tempfile.gettempdir(), creationflags=_creation_flags(),
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor = (int(part) for part in result.stdout.strip().split(".")[:2])
    except (TypeError, ValueError):
        return None
    if (major, minor) < minimum:
        return None
    return (major, minor)

def _runs(exe: str, timeout: float = 120.0) -> bool:
    """True when ``exe`` starts at all - an orphaned environment does not.

    A virtualenv's ``Scripts/python.exe`` is a small launcher that reads
    ``pyvenv.cfg`` and starts the base interpreter named there, so an
    environment whose base has gone fails at exactly this point, with the
    launcher's own message and exit code 103.  Starting it is the only way to
    tell; a file that exists proves nothing.
    """
    if not exe or not os.path.isfile(exe):
        return False
    probe = "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))"
    try:
        result = subprocess.run(
            [exe, "-c", probe], capture_output=True, text=True,
            cwd=tempfile.gettempdir(), creationflags=_creation_flags(),
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())

#: What ``venv`` writes next to an environment's interpreter.  It is what makes
#: the folder an environment rather than a stray copy of ``python.exe``, and its
#: ``home``/``executable`` lines name the interpreter the environment runs on.
PYVENV_CFG_NAME = "pyvenv.cfg"

def _read_pyvenv_cfg(env_dir: str) -> dict[str, str]:
    """``pyvenv.cfg`` of an environment as ``{lower-cased key: value}``."""
    config: dict[str, str] = {}
    try:
        with open(os.path.join(env_dir, PYVENV_CFG_NAME),
                  encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, separator, value = line.partition("=")
                if separator:
                    config[key.strip().lower()] = value.strip()
    except OSError:
        return {}
    return config

def environment_base_python(env_dir: str) -> str:
    """The interpreter an environment was built from, per its ``pyvenv.cfg``.

    ``executable`` is exact and is read first; older environments carry only
    ``home``, the folder holding that interpreter, which is also what the
    launcher itself trusts.
    """
    config = _read_pyvenv_cfg(env_dir)
    executable = config.get("executable") or ""
    if executable:
        return executable
    home = config.get("home") or ""
    if not home:
        return ""
    name = "python.exe" if sys.platform == "win32" else "python3"
    return os.path.join(home, name)

def missing_base_python(env_dir: str) -> str:
    """The Python an environment was built from, when it is no longer there.

    Returns "" while the environment is fine, and also when it carries no
    ``pyvenv.cfg`` at all - a folder that never was an environment is a
    different problem (see ``PythonRuntime._set_aside_leftover``).

    This is the one failure a virtualenv cannot report for itself.  Reinstall
    the application somewhere else, and every environment it had created still
    points at the interpreter inside the folder that is gone; the only thing
    the user sees is the run that used it failing as a model crash.
    """
    base = environment_base_python(env_dir)
    if not base or os.path.isfile(base):
        return ""
    return base

class EnvironmentUnusable(RuntimeError):
    """An environment exists but its interpreter cannot be made to run."""

#: What a C extension built *at run time* needs from the base runtime: the
#: headers and the import library.  Triton compiles its ``cuda_utils`` helper the
#: first time a kernel runs, and with neither present its bundled compiler fails
#: with ``include file 'Python.h' not found`` - which reaches the user as the
#: whole engine refusing to start.  ``Include`` is the folder ``sysconfig``
#: reports on Windows.
_RUNTIME_HEADER_DIR = "Include"
_RUNTIME_LIBS_DIR = "libs"

def _copy_runtime_folder(source_dir: str, runtime_dir: str, source_name: str,
                         target_name: str) -> bool:
    """Copy one build-support folder into a runtime; False when there is none."""
    source = os.path.join(source_dir, source_name)
    if not os.path.isdir(source):
        return False
    try:
        shutil.copytree(source, os.path.join(runtime_dir, target_name),
                        dirs_exist_ok=True)
    except OSError as exc:
        log.warning("Could not copy %s into the Python runtime at %s: %s",
                    source, runtime_dir, exc)
        return False
    return True

def runtime_build_support_missing(runtime_dir: str) -> bool:
    """True when a runtime cannot build a C extension for itself."""
    if not runtime_dir or not os.path.isdir(runtime_dir):
        return False
    if not os.path.isfile(os.path.join(runtime_dir, _RUNTIME_HEADER_DIR,
                                       "Python.h")):
        return True
    return not glob.glob(os.path.join(runtime_dir, _RUNTIME_LIBS_DIR,
                                      "python3*.lib"))

def ensure_runtime_build_support(runtime_dir: str, source_dir: str) -> bool:
    """Give a runtime the headers and import library a run-time build needs.

    A copy staged by an older version of the application has no ``Include`` and
    no ``libs`` (the stager used to ship only the interpreter, its DLLs and
    ``Lib``), so a runtime that already exists is topped up in place instead of
    asking for a reinstall - the environments built from it keep working, and
    the several megabytes this adds is the difference between an engine that can
    compile its own helper and one that cannot.  Returns True when anything was
    written.
    """
    if not runtime_dir or not source_dir:
        return False
    if not os.path.isdir(runtime_dir) or not os.path.isdir(source_dir):
        return False
    written = False
    if not os.path.isfile(os.path.join(runtime_dir, _RUNTIME_HEADER_DIR,
                                       "Python.h")):
        written |= _copy_runtime_folder(source_dir, runtime_dir, "include",
                                        _RUNTIME_HEADER_DIR)
    if not glob.glob(os.path.join(runtime_dir, _RUNTIME_LIBS_DIR,
                                  "python3*.lib")):
        written |= _copy_runtime_folder(source_dir, runtime_dir, "libs",
                                        _RUNTIME_LIBS_DIR)
    if written:
        log.info("Added headers/libs to the Python runtime at %s", runtime_dir)
    return written

def _stage_python_runtime() -> str:
    """Copy the shipped Python runtime into the user's data folder.

    Returns the path of the copy's interpreter, or "" when there is no shipped
    runtime to copy.  Environments record the absolute path of the interpreter
    they run on, so building them from a copy that cannot move - one inside
    ``%APPDATA%`` rather than inside the program folder - is what lets them keep
    working when the application is reinstalled, moved or upgraded
    (see ``paths.staged_python_runtime``).  The copy is made once, then only
    verified, so the usual cost is one short-lived interpreter start.
    """
    staged_dir = paths.staged_python_runtime()
    staged_exe = os.path.join(staged_dir, paths.PYTHON_EXE_NAME)
    if _usable_python(staged_exe):
        # A copy staged before the headers and import libraries were shipped is
        # topped up from the bundled runtime (see
        # ``ensure_runtime_build_support``).  A complete copy is not looked at
        # at all, so it keeps standing on its own.
        if runtime_build_support_missing(staged_dir):
            ensure_runtime_build_support(staged_dir,
                                         paths.shipped_python_runtime())
        return staged_exe
    source = paths.shipped_python_runtime()
    if not source:
        return ""
    if os.path.isdir(staged_dir):
        # A half-written copy (an interrupted run, a scanner holding a DLL, no
        # disk space) is not something to build environments on: start again.
        shutil.rmtree(staged_dir, ignore_errors=True)
    try:
        shutil.copytree(source, staged_dir)
    except OSError as exc:
        log.warning("Could not copy the Python runtime to %s: %s", staged_dir, exc)
        return ""
    if not _usable_python(staged_exe):
        log.warning("The copied Python runtime at %s does not run", staged_dir)
        return ""
    ensure_runtime_build_support(staged_dir, source)
    log.info("Staged the bundled Python runtime at %s", staged_dir)
    return staged_exe

def _version_sort_key(path: str) -> tuple[int, int]:
    """Sort key that puts ``Python313`` above ``Python312`` and ``Python39``."""
    match = _VERSION_IN_PATH.search(path.replace("/", "\\").lower())
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

def _dedupe_newest_first(paths: list[str]) -> list[str]:
    unique: list[str] = []
    for path in paths:
        if path and path not in unique:
            unique.append(path)
    return sorted(unique, key=_version_sort_key, reverse=True)

def _py_launcher_pythons() -> list[str]:
    """Interpreters the ``py`` launcher knows about, newest first.

    The python.org installer creates ``C:\\Windows\\py.exe`` even when "Add
    Python to PATH" is left unchecked - the default - and ``py -0p`` lists
    every install it found.  This therefore finds a real interpreter on the
    many machines where no ``python`` command resolves at all.
    """
    launcher = shutil.which("py")
    if not launcher or _is_store_alias(launcher):
        return []
    try:
        result = subprocess.run(
            [launcher, "-0p"], capture_output=True, text=True,
            cwd=tempfile.gettempdir(), creationflags=_creation_flags(),
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    found: list[str] = []
    for line in result.stdout.splitlines():
        # e.g. " -V:3.13 *        C:\Python313\python.exe"
        for token in reversed(line.split()):
            if token.lower().endswith("python.exe") and os.path.isfile(token):
                found.append(token)
                break
    return _dedupe_newest_first(found)

def _glob_installs() -> list[str]:
    """Interpreters in the folders the python.org installer uses."""
    patterns: list[str] = []
    for base in (os.environ.get("LOCALAPPDATA", ""),
                 os.environ.get("PROGRAMFILES", ""),
                 os.environ.get("PROGRAMFILES(X86)", "")):
        if not base:
            continue
        patterns.append(os.path.join(base, "Programs", "Python", "Python3*",
                                     "python.exe"))
        patterns.append(os.path.join(base, "Python3*", "python.exe"))
    patterns.append(r"C:\Python3*\python.exe")
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return _dedupe_newest_first(found)

def _registry_installs() -> list[str]:
    """Interpreters registered under ``Software\\Python\\PythonCore``."""
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return []
    roots = (
        (winreg.HKEY_CURRENT_USER, r"Software\Python\PythonCore"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Python\PythonCore"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Python\PythonCore"),
    )
    found: list[str] = []
    for hive, subkey in roots:
        try:
            key = winreg.OpenKey(hive, subkey)
        except OSError:
            continue
        try:
            for index in range(winreg.QueryInfoKey(key)[0]):
                try:
                    name = winreg.EnumKey(key, index)
                except OSError:
                    continue
                if not name[:1].isdigit():
                    continue  # e.g. the "PyLauncher" key
                try:
                    install = winreg.OpenKey(key, name + r"\InstallPath")
                except OSError:
                    continue
                try:
                    folder, _ = winreg.QueryValueEx(install, "")
                except OSError:
                    folder = ""
                finally:
                    winreg.CloseKey(install)
                exe = os.path.join(folder, "python.exe")
                if folder and os.path.isfile(exe):
                    found.append(exe)
        finally:
            winreg.CloseKey(key)
    return _dedupe_newest_first(found)

def _find_system_python() -> str:
    """Find a usable base interpreter for the managed virtualenvs.

    Search order, and why:

    1. The per-user copy of the runtime shipped with the application, made
       here on first use.  This is what lets somebody who has never installed
       Python create the per-engine and per-model environments - and a base
       that cannot move is also what keeps those environments working when the
       application is reinstalled or moved, so it comes first.
    2. The shipped runtime itself, used in place when it could not be copied.
    3. ``python``/``python3`` on PATH - but only when they are real
       interpreters.  The Microsoft Store aliases are skipped; handing one of
       those to ``python -m venv`` is exactly what failed with code 9009.
    4. The ``py`` launcher (``py -0p``), which lists installs even when they
       are not on PATH.
    5. The folders the python.org installer uses, then the registry.
    """
    if not getattr(sys, "frozen", False):
        # Development mode: the running interpreter is already a full Python.
        return sys.executable

    staged = _stage_python_runtime()
    if staged:
        log.info("Using the staged Python runtime at %s", staged)
        return staged

    bundled = paths.bundled_python_exe()
    if bundled and _usable_python(bundled):
        log.info("Using the bundled Python runtime at %s", bundled)
        return bundled

    candidates: list[str] = []
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            candidates.append(path)
    candidates += _py_launcher_pythons()
    launcher = shutil.which("py")
    if launcher:
        candidates.append(launcher)
    candidates += _glob_installs()
    candidates += _registry_installs()

    for candidate in candidates:
        version = _usable_python(candidate)
        if version:
            log.info("Using Python %d.%d at %s", version[0], version[1], candidate)
            return candidate
    raise FileNotFoundError(
        "No usable Python was found. AI Voice Studio builds each engine's own "
        "environment with a base Python 3.10 or newer that has the 'venv' "
        "module. Install Python from https://www.python.org/downloads/ (the "
        "installer's 'py' launcher is enough - Python on PATH is not "
        "required), or reinstall AI Voice Studio so its bundled Python "
        "runtime is present."
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
        return os.path.isfile(self.python_exe) and os.path.isfile(self.marker_file)

    @property
    def missing_base_python(self) -> str:
        """The Python this environment was built from, when it is gone.

        "" while the environment is usable - and while it does not exist, since
        a folder that is not an environment is a different problem (see
        ``_set_aside_leftover``).
        """
        return missing_base_python(self._env_dir)

    @property
    def is_usable(self) -> bool:
        """True when the environment exists *and* its interpreter can run."""
        return self.is_created and not self.missing_base_python

    @property
    def marker_file(self) -> str:
        """The file ``venv`` writes to prove the folder really is an env.

        ``Scripts/python.exe`` inside a virtualenv is a small redirector that
        finds its base interpreter through ``pyvenv.cfg``.  Without that file
        it refuses to run at all - it prints "failed to locate pyvenv.cfg" and
        exits with status 106 - which is exactly the shape a folder left
        behind by a removal that could not delete everything has.  So the
        interpreter alone is *not* evidence that the environment is usable.
        """
        return os.path.join(self._env_dir, "pyvenv.cfg")

    @property
    def site_packages_dirs(self) -> list[str]:
        """The environment's ``site-packages`` folder(s), when they exist.

        Used for *synchronous* "is this package installed here?" checks: a
        background probe answers that question for the GUI, but the decision
        which interpreter to start an engine's worker with must not depend on
        a probe that may not have answered yet (see ``venv_packages``).
        """
        found: list[str] = []
        for pattern in ("Lib/site-packages", "lib/python*/site-packages"):
            found.extend(sorted(glob.glob(os.path.join(self._env_dir, pattern))))
        return [path for path in found if os.path.isdir(path)]

    # -- system python ------------------------------------------------------

    @property
    def system_python(self) -> str:
        if self._system_python is None:
            self._system_python = _find_system_python()
        return self._system_python

    # -- environment lifecycle ----------------------------------------------

    def ensure_env(self) -> str:
        """Create the virtualenv if it is not usable. Returns the env dir."""
        if self.is_created:
            if not self.missing_base_python:
                return self._env_dir
            # The Python this environment was built from has gone - the
            # application was reinstalled, moved or upgraded somewhere else
            # (see ``missing_base_python``).  Point the existing environment at
            # a Python that exists instead of building a new one, so the
            # packages already in it survive; a fresh environment is the
            # fallback for when even that cannot be done.
            if self.repair_env():
                return self._env_dir
            log.warning(
                "The environment at %s could not be pointed at a Python that "
                "exists; building a new one", self._env_dir,
            )
        if os.path.isdir(self._env_dir):
            # A folder that is not an environment (no ``pyvenv.cfg``) is what a
            # removal that could not delete everything leaves behind: the
            # files that survived are precisely the ones a running engine holds
            # open.  Move it out of the way rather than trying to delete it -
            # the environment path has to be free for the fresh one.
            self._set_aside_leftover()
        os.makedirs(self._env_dir, exist_ok=True)
        log.info("Creating addon virtualenv at %s", self._env_dir)
        system_py = self.system_python
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [system_py, "-m", "venv", self._env_dir, "--clear"]
        try:
            subprocess.run(
                command,
                capture_output=True, text=True, check=True,
                cwd=tempfile.gettempdir(), creationflags=creationflags,
                timeout=VENV_CREATE_TIMEOUT,
            )
            log.info("Addon virtualenv created successfully")
        except subprocess.TimeoutExpired as exc:
            # A ``venv`` that never returns (a hung antivirus scan, a broken
            # base interpreter) used to hang the caller's thread for ever.
            raise RuntimeError(
                f"Creating the Python environment at {self._env_dir} did not "
                f"finish within {VENV_CREATE_TIMEOUT} seconds."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                f"Failed to create the Python environment at {self._env_dir}: "
                f"{' '.join(command)} exited with {exc.returncode}"
                + (f" ({detail[-500:]})" if detail else "")
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to start Python: {exc}") from exc
        if not os.path.isfile(self.marker_file):
            raise RuntimeError(
                f"Python did not finish creating the environment at "
                f"{self._env_dir} (it has no pyvenv.cfg), so it cannot be "
                "used. Close AI Voice Studio, delete that folder and try "
                "again."
            )
        return self._env_dir

    def repair_env(self) -> bool:
        """Point an existing environment at a Python that exists again.

        Returns True when the environment's interpreter runs afterwards.

        The work is ``python -m venv --upgrade``, which rewrites ``pyvenv.cfg``
        and the launchers **in place**: the packages already installed in the
        environment - several gigabytes of PyTorch, usually - are left exactly
        as they are, which is the difference between a second of work and
        downloading a model's whole back-end again.

        ``--without-pip`` is deliberate.  An environment that has pip keeps it,
        and bootstrapping pip again would replace a newer pip with the bundled
        one; pip is needed to install a package, never to run one.
        """
        if not self.missing_base_python:
            return self.is_created
        try:
            system_python = self.system_python
        except Exception as exc:
            log.warning("No Python is available to repair %s: %s",
                        self._env_dir, exc)
            return False
        log.info(
            "The environment at %s was built from %s, which is gone; pointing "
            "it at %s", self._env_dir, self.missing_base_python, system_python,
        )
        command = [system_python, "-m", "venv", "--upgrade", "--without-pip",
                   self._env_dir]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                cwd=tempfile.gettempdir(), creationflags=_creation_flags(),
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("Could not re-point %s: %s", self._env_dir, exc)
            return False
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            log.warning("Could not re-point %s (exit %s): %s", self._env_dir,
                        result.returncode, detail[-300:])
            return False
        if self.missing_base_python or not _runs(self.python_exe):
            log.warning("The environment at %s still does not run", self._env_dir)
            return False
        log.info("The environment at %s now runs on %s", self._env_dir,
                 system_python)
        return True

    def usable_python_exe(self) -> str:
        """The environment's interpreter, re-pointed at this installation first.

        What a worker launch wants.  Unlike :meth:`ensure_env` this never
        *builds* an environment, so a model that was never installed still
        reaches the caller's own "not installed" message instead of being
        handed an empty environment that cannot hold its back-end.

        Raises :class:`EnvironmentUnusable` when the environment exists but its
        interpreter cannot be made to run at all.
        """
        if self.is_created and self.missing_base_python and not self.repair_env():
            raise EnvironmentUnusable(
                f"The Python environment at {self._env_dir} was built by an "
                "earlier installation of AI Voice Studio, and the Python it "
                "ran on is no longer on this machine, so it cannot start. "
                "Reinstall the model from Settings."
            )
        return self.python_exe

    def _set_aside_leftover(self) -> None:
        """Move an unusable environment folder out of the way.

        Windows cannot delete a running interpreter - the ``.exe`` and every
        DLL that process mapped are held open - so removing an engine while
        its worker is alive always leaves a partial folder behind, and that
        folder cannot be deleted later either while the process lives.  A
        folder can still be *renamed*, which is all that is needed: the
        environment path is free again and the remains are swept up by the
        next install, or by the system once nothing holds them open.
        """
        if not os.path.isdir(self._env_dir):
            return
        # Anything moved aside by an earlier attempt that is free by now.
        for stale in glob.glob(self._env_dir + ".leftover*"):
            shutil.rmtree(stale, ignore_errors=True)
        for index in range(1, 100):
            aside = f"{self._env_dir}.leftover{index}"
            if os.path.exists(aside):
                continue
            try:
                os.replace(self._env_dir, aside)
            except OSError as exc:
                log.warning(
                    "Could not move the unusable environment %s aside: %s",
                    self._env_dir, exc,
                )
                return
            log.info("Moved the unusable environment aside to %s", aside)
            shutil.rmtree(aside, ignore_errors=True)
            return

    def ensure_pip(self) -> str:
        """Ensure pip is available in the virtualenv. Returns pip path."""
        self.ensure_env()
        if os.path.isfile(self.pip_exe):
            return self.pip_exe
        log.info("Bootstrapping pip in the virtualenv at %s", self._env_dir)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [self.python_exe, "-m", "ensurepip", "--upgrade"]
        try:
            subprocess.run(
                command, capture_output=True, text=True, check=True,
                cwd=tempfile.gettempdir(), creationflags=creationflags,
                timeout=ENSURE_PIP_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"pip could not be installed into the environment at "
                f"{self._env_dir}: ensurepip did not finish within "
                f"{ENSURE_PIP_TIMEOUT} seconds."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip().splitlines()
            raise RuntimeError(
                f"pip could not be installed into the environment at "
                f"{self._env_dir}: {' '.join(command)} exited with "
                f"{exc.returncode}"
                + (f" - {detail[-1].strip()}" if detail else ".")
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to start Python: {exc}") from exc
        return self.pip_exe

    def remove_env(self) -> str | None:
        """Remove the entire virtualenv.

        Best effort *and* honest.  Files an engine still has open (its
        worker's interpreter, the DLLs that process loaded) cannot be deleted,
        so whatever survives is moved aside under a ``.leftover`` name - the
        environment path is free at once, and the remains go on the next
        install or when nothing holds them open any more.  Returns a note for
        the user when the removal was not clean, ``None`` when it was.
        """
        if not os.path.isdir(self._env_dir):
            return None
        shutil.rmtree(self._env_dir, ignore_errors=True)
        if not os.path.isdir(self._env_dir):
            log.info("Addon virtualenv removed: %s", self._env_dir)
            return None
        log.warning(
            "Addon virtualenv %s could not be fully removed (the engine still "
            "has files open); moving the remains aside", self._env_dir,
        )
        self._set_aside_leftover()
        return (
            "Some files of the engine's environment were still in use by the "
            "engine itself, so they were moved aside; they are tidied up by "
            "the next install."
        )

    # -- pip operations -----------------------------------------------------

    def pip_install(
        self,
        packages: list[str] | str,
        progress: Callable[[str, int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        index_url: str | None = None,
    ) -> dict[str, str]:
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
        index_url:
            Optional alternative package index (``--index-url``), used to pull
            the CUDA build of PyTorch from the official PyTorch wheels.

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
        cleaned: list[str] = []
        for entry in packages:
            for token in str(entry).split():
                if token.lower() in ("pip", "install", "--user"):
                    continue
                if token.startswith("-") and token not in ("-U", "--upgrade", "--no-deps"):
                    continue
                cleaned.append(token)
        packages = cleaned or []
        pip = self.ensure_pip()
        cmd = [pip, "install", "--no-warn-script-location"]
        if index_url:
            cmd += ["--index-url", index_url]
        cmd += packages
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

        output_lines: list[str] = []
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
            message = f"pip install failed (exit {proc.returncode})"
            detail = _pip_failure_line(output_lines)
            if detail:
                message += f": {detail}"
            return {"ok": False, "output": combined, "error": message}
        log.info("pip install succeeded")
        if progress:
            progress("Installation complete.", 1, 1)
        return {"ok": True, "output": combined, "error": ""}

    def pip_uninstall(
        self,
        packages: list[str] | str,
    ) -> dict[str, str]:
        """Uninstall packages from the addon virtualenv."""
        if isinstance(packages, str):
            packages = [packages]
        if not self.is_created:
            return {"ok": True, "output": "", "error": ""}
        pip = self.pip_exe
        cmd = [pip, "uninstall", "-y", *packages]
        log.info("Running pip uninstall: %s", " ".join(cmd))
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=tempfile.gettempdir(), creationflags=creationflags,
                timeout=SHORT_COMMAND_TIMEOUT,
                check=False,
            )
            return {"ok": result.returncode == 0,
                    "output": result.stdout, "error": result.stderr}
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "",
                    "error": f"The command took longer than {SHORT_COMMAND_TIMEOUT} s"}
        except OSError as exc:
            return {"ok": False, "output": "", "error": str(exc)}

    def pip_list(self) -> list[dict[str, str]]:
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
                timeout=SHORT_COMMAND_TIMEOUT,
                check=False,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
            log.debug("Ignoring a failure in PythonRuntime.pip_list", exc_info=True)
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
            timeout=PROBE_TIMEOUT,
            check=False,
        )

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_default_runtime: PythonRuntime | None = None
#: One environment per pip-installed TTS engine ("pocket_tts", "f5tts", ...).
_engine_runtimes: dict[str, PythonRuntime] = {}
_runtime_lock = threading.Lock()

def _safe_env_name(engine_id: str | None) -> str:
    """Folder name for an engine's environment (path-safe, lower-case)."""
    name = "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in (engine_id or "").strip().lower()
    )
    return name or "default"

def environment_id(engine_id: str | None) -> str:
    """Folder id of the virtualenv that holds ``engine_id``'s packages.

    Usually the engine's own id; the OmniVoice engines are the one exception
    (see ``SHARED_ENVIRONMENTS``) and map onto their shared environment.
    """
    name = _safe_env_name(engine_id)
    return SHARED_ENVIRONMENTS.get(name, name)

def engine_env_dir(engine_id: str) -> str:
    """The folder of the virtualenv that holds one TTS engine."""
    return os.path.join(
        paths.user_data_dir(), _TTS_ENVS_DIR, environment_id(engine_id)
    )

def get_runtime(engine_id: str | None = None) -> PythonRuntime:
    """Return the PythonRuntime instance (lazy singleton per environment).

    Without ``engine_id`` this is the shared addon environment used by addons
    and the Developer tab.  With an engine id it is that TTS engine's *own*
    environment (``%APPDATA%/AIVoiceStudio/tts_envs/<engine>``), so engines
    whose pip dependencies conflict never share site-packages.  Engines listed
    in ``SHARED_ENVIRONMENTS`` (the two OmniVoice ones) resolve to the same
    runtime object.
    """
    global _default_runtime
    if not engine_id:
        if _default_runtime is None:
            with _runtime_lock:
                if _default_runtime is None:
                    _default_runtime = PythonRuntime()
        return _default_runtime
    key = environment_id(engine_id)
    with _runtime_lock:
        runtime = _engine_runtimes.get(key)
        if runtime is None:
            runtime = PythonRuntime(engine_env_dir(engine_id))
            _engine_runtimes[key] = runtime
        return runtime

def engine_runtime(engine_id: str) -> PythonRuntime:
    """The environment that actually holds one TTS engine's packages.

    Every pip-installed TTS engine gets its *own* virtualenv
    (``%APPDATA%/AIVoiceStudio/tts_envs/<engine>``, shared by the two
    OmniVoice engines) so engines with conflicting dependencies never share
    site-packages.  An engine that was installed before those per-TTS
    environments existed still lives in the shared addon environment, so that
    one is used as the fallback until the engine is (re)installed into its
    own environment.
    """
    runtime = get_runtime(engine_id)
    if runtime.is_created:
        return runtime
    return get_runtime()

def engine_env_exists(engine_id: str) -> bool:
    """True when ``engine_id`` has its own (created) virtualenv."""
    return get_runtime(engine_id).is_created

# ---------------------------------------------------------------------------
# Speech-to-Text environments: one virtualenv per model
# ---------------------------------------------------------------------------
#: Every speech-to-text model gets an environment of its *own* under this
#: folder.  STT models pull in very different, very heavy stacks (NeMo,
#: transformers + torch, SpeechBrain, Moonshine) whose transitive pins
#: (torch, torchaudio, transformers, numpy, protobuf, ...) regularly conflict,
#: so sharing site-packages between two models would let installing one model
#: break another.  A model is therefore never allowed to reuse another
#: model's environment.
_STT_ENVS_DIR = "stt_envs"

_stt_runtimes: dict[str, PythonRuntime] = {}

def stt_environment_id(model_id: str) -> str:
    """Folder id of the environment that holds one STT model's packages."""
    return _safe_env_name(model_id)

def stt_env_dir(model_id: str) -> str:
    """The folder of the virtualenv that holds one STT model.

    ``%APPDATA%/AIVoiceStudio/stt_envs/<model id>`` - a sibling of the TTS
    engine environments, so the two worlds can never share packages either.
    """
    return os.path.join(
        paths.user_data_dir(), _STT_ENVS_DIR, stt_environment_id(model_id)
    )

def get_stt_runtime(model_id: str) -> PythonRuntime:
    """The :class:`PythonRuntime` of one STT model's own environment.

    Cached per model id, so the same model always resolves to the same
    environment object and repeated installs/removals stay consistent.
    """
    key = stt_environment_id(model_id)
    with _runtime_lock:
        runtime = _stt_runtimes.get(key)
        if runtime is None:
            runtime = PythonRuntime(stt_env_dir(model_id))
            _stt_runtimes[key] = runtime
        return runtime

# ---------------------------------------------------------------------------
# Audio-description environments: one virtualenv per describing model
# ---------------------------------------------------------------------------
#: Every video description model gets an environment of its own under this
#: folder.  A describing model pulls in a vision-language stack (transformers +
#: torch + a multimodal processor, and often a quantisation runtime) whose pins
#: clash with both the TTS engines and the speech-to-text models, so it is kept
#: separate from ``tts_envs`` and ``stt_envs``.  Installing a describing model
#: therefore can never break a voice or a transcription model.
_DESCRIBE_ENVS_DIR = "describe_envs"

_describe_runtimes: dict[str, PythonRuntime] = {}

def describe_environment_id(model_id: str) -> str:
    """Folder id of the environment that holds one describing model."""
    return _safe_env_name(model_id)

def describe_env_dir(model_id: str) -> str:
    """The folder of the virtualenv that holds one describing model.

    ``%APPDATA%/AIVoiceStudio/describe_envs/<model id>`` - a sibling of the TTS
    and STT environments, so the three families can never share packages.
    """
    return os.path.join(
        paths.user_data_dir(), _DESCRIBE_ENVS_DIR, describe_environment_id(model_id)
    )

def get_describe_runtime(model_id: str) -> PythonRuntime:
    """The :class:`PythonRuntime` of one describing model's own environment."""
    key = describe_environment_id(model_id)
    with _runtime_lock:
        runtime = _describe_runtimes.get(key)
        if runtime is None:
            runtime = PythonRuntime(describe_env_dir(model_id))
            _describe_runtimes[key] = runtime
        return runtime

def describe_env_exists(model_id: str) -> bool:
    """True when ``model_id`` has its own (created) virtualenv."""
    return get_describe_runtime(model_id).is_created
