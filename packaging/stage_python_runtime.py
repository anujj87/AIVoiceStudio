"""Stage a full, portable Python runtime next to the built application.

AI Voice Studio creates one virtualenv per TTS engine and per STT model.  That
needs a *full* CPython - one that still carries ``Lib/venv`` and
``Lib/ensurepip`` - because ``python -m venv`` lives there.  The Python that
PyInstaller freezes into the application cannot do it: the frozen executable is
not an interpreter you can hand to ``-m venv``.

This script copies a trimmed copy of the interpreter that runs it into
``dist/AIVS-64/AI-Voice-Studio/python-runtime`` (or the 32-bit twin).  The Inno
Setup script ships that folder with everything else, so the installer is
self-sufficient and an end user never has to install Python.  The application
finds it through ``paths.bundled_python_exe()``.

Usage (from the repository root, with the build venv)::

    .venv/Scripts/python.exe packaging/stage_python_runtime.py --arch x64

Every build ships this runtime, so a skip is a build failure: the setup it
would otherwise produce installs an application that cannot create a single
engine environment.  ``main`` therefore exits nonzero unless a complete runtime
was written (the app's own fallback to a Python found on the machine is for a
source checkout, not for what is shipped).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

#: Folders under ``Lib`` that a build/venv base does not need at run time.
_EXCLUDED_LIB_DIRS = {
    "site-packages",   # the build venv's packages must never leak into the base
    "test", "tests",   # CPython's own test suite
    "tkinter", "turtledemo", "idlelib", "lib2to3",
}

#: Top-level files that make a minimal, runnable Windows interpreter.
_EXE_FILES = ("python.exe", "python3.dll", "vcruntime140.dll",
              "vcruntime140_1.dll")

#: Folders a compiler needs to build a C extension *at run time*.  Engines
#: compile small helpers the first time they run - Triton builds its
#: ``cuda_utils`` module against ``Python.h`` and ``python3XX.lib`` - and a
#: runtime without them fails with "include file 'Python.h' not found".  The
#: headers land under ``Include``, the folder ``sysconfig`` reports on Windows.
#: (source name, name in the staged runtime)
_BUILD_DIRS = (("include", "Include"), ("libs", "libs"))


def _dist_target(arch: str) -> str:
    suffix = "64" if arch == "x64" else "32"
    return os.path.join("dist", f"AIVS-{suffix}", "AI-Voice-Studio",
                        "python-runtime")


def _source_prefix() -> str:
    """The base install whose interpreter is running this script."""
    return os.path.abspath(sys.base_prefix)


def _has_venv_support(prefix: str) -> bool:
    return (os.path.isfile(os.path.join(prefix, "Lib", "venv", "__init__.py"))
            and os.path.isdir(os.path.join(prefix, "Lib", "ensurepip")))


def missing_parts(target: str) -> list[str]:
    """The pieces a staged runtime must have; empty when the copy is complete.

    ``python.exe`` alone is not enough: the environments are created with
    ``python -m venv``, which lives in ``Lib/venv`` and needs ``Lib/ensurepip``,
    and an engine that compiles a helper of its own needs ``Include/Python.h``
    and ``libs/python3XX.lib``.
    """
    required_files = ("python.exe", os.path.join("Lib", "venv", "__init__.py"))
    missing = [relative.replace(os.sep, "/") for relative in required_files
               if not os.path.isfile(os.path.join(target, relative))]
    if not os.path.isdir(os.path.join(target, "Lib", "ensurepip")):
        missing.append("Lib/ensurepip")
    if not os.path.isfile(os.path.join(target, "Include", "Python.h")):
        missing.append("Include/Python.h")
    libs = os.path.join(target, "libs")
    has_import_library = os.path.isdir(libs) and any(
        name.lower().startswith("python3") and name.lower().endswith(".lib")
        for name in os.listdir(libs)
    )
    if not has_import_library:
        missing.append("libs/python3XX.lib")
    return missing


def _copy_tree_filtered(src: str, dst: str) -> None:
    """Copy ``src`` into ``dst``, dropping the folders we do not need."""
    for root, dirs, files in os.walk(src):
        dirs[:] = [name for name in dirs
                   if name not in _EXCLUDED_LIB_DIRS and name != "__pycache__"]
        relative = os.path.relpath(root, src)
        target = dst if relative == "." else os.path.join(dst, relative)
        os.makedirs(target, exist_ok=True)
        for name in files:
            shutil.copy2(os.path.join(root, name), os.path.join(target, name))


def stage(arch: str, *, quiet: bool = False) -> bool:
    """Copy the runtime; return True when something was written."""
    def say(message: str) -> None:
        if not quiet:
            print(message)

    if sys.platform != "win32":
        say("stage_python_runtime: not Windows; nothing to stage.")
        return False
    if not sys.maxsize > 2 ** 32 and arch == "x64":
        say("stage_python_runtime: x64 build needs a 64-bit base interpreter; "
            "skipping.")
        return False

    prefix = _source_prefix()
    if not _has_venv_support(prefix):
        say(f"stage_python_runtime: {prefix} has no 'venv'/'ensurepip'; the "
            "app will fall back to a Python found on the machine.")
        return False

    target = _dist_target(arch)
    dist_app = os.path.dirname(target)
    if not os.path.isdir(dist_app):
        say(f"stage_python_runtime: {dist_app} does not exist - run "
            "PyInstaller first; skipping.")
        return False

    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)

    say(f"stage_python_runtime: copying {prefix} -> {target}")
    # The interpreter and the DLLs every extension links against.
    for name in _EXE_FILES:
        source = os.path.join(prefix, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(target, name))
    # The interpreter library (python313.dll etc.) and the C runtime licence.
    for name in os.listdir(prefix):
        if name.lower().startswith("python3") and name.lower().endswith(".dll"):
            shutil.copy2(os.path.join(prefix, name),
                         os.path.join(target, name))
    for name in ("LICENSE.txt",):
        source = os.path.join(prefix, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(target, name))
    # The extension modules (DLLs) and the standard library (Lib).
    for folder in ("DLLs", "Lib"):
        source = os.path.join(prefix, folder)
        if os.path.isdir(source):
            _copy_tree_filtered(source, os.path.join(target, folder))
    # Headers and import libraries, so an engine that compiles a helper of its
    # own at run time can do so (see _BUILD_DIRS).
    for source_name, target_name in _BUILD_DIRS:
        source = os.path.join(prefix, source_name)
        if os.path.isdir(source):
            _copy_tree_filtered(source, os.path.join(target, target_name))
    # ``pythonw.exe`` is harmless and keeps the runtime a complete install.
    pythonw = os.path.join(prefix, "pythonw.exe")
    if os.path.isfile(pythonw):
        shutil.copy2(pythonw, os.path.join(target, "pythonw.exe"))

    missing = missing_parts(target)
    if missing:
        say("stage_python_runtime: the copy is missing " + ", ".join(missing) +
            "; a build must ship a complete runtime, so this is a failure.")
        return False
    say(f"stage_python_runtime: staged a Python runtime at {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("x64", "x86"), default="x64")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not stage(args.arch, quiet=args.quiet):
        print("stage_python_runtime: no bundled Python runtime was staged, and "
              "every build ships one. Fix the reason above and rebuild "
              "(packaging\\build.ps1 -Arch " + args.arch + ").",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
