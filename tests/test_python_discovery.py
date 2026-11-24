"""The base-interpreter search must never hand a Store alias to ``venv``.

Regression: on Windows ``shutil.which("python3")`` resolves to
``...\\Microsoft\\WindowsApps\\python3.EXE`` - a zero-byte App Execution Alias
that opens the Microsoft Store and exits 9009.  The old search returned it
first, so ``python -m venv`` failed and *every* STT model and TTS engine failed
to install with "Failed to create the Python environment ... exited with 9009".

These tests pin the fix: candidates are validated (they must run and carry
``venv``/``ensurepip``), the Store alias is skipped, a runtime shipped with the
app is preferred (copied into the user's own folder first, so the environments
built from it survive the application being reinstalled - see
``test_env_repair``), and installs are found through the ``py`` launcher, the
standard folders and the registry - newest first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import paths
from ai_voice_studio import python_runtime as pr

_STORE_ALIAS = (r"C:\Users\victor\AppData\Local\Microsoft\WindowsApps"
                r"\python3.EXE")
_REAL_PYTHON = r"C:\Python313\python.exe"


def _completed(returncode: int, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr="")


class StoreAliasTest(unittest.TestCase):
    def test_the_store_alias_is_recognised(self):
        self.assertTrue(pr._is_store_alias(_STORE_ALIAS))
        self.assertTrue(pr._is_store_alias(
            r"C:/Users/v/AppData/Local/Microsoft/WindowsApps/python.exe"))
        self.assertFalse(pr._is_store_alias(_REAL_PYTHON))

    def test_a_store_alias_is_never_usable(self):
        with mock.patch.object(pr.os.path, "isfile", return_value=True):
            self.assertIsNone(pr._usable_python(_STORE_ALIAS))

    def test_a_candidate_that_cannot_import_venv_is_rejected(self):
        # The Store stub exits 9009; a real Python without ``venv`` fails too.
        with mock.patch.object(pr.os.path, "isfile", return_value=True), \
             mock.patch.object(pr.subprocess, "run",
                               return_value=_completed(1, "")):
            self.assertIsNone(pr._usable_python(_REAL_PYTHON))

    def test_a_real_interpreter_is_accepted(self):
        with mock.patch.object(pr.os.path, "isfile", return_value=True), \
             mock.patch.object(pr.subprocess, "run",
                               return_value=_completed(0, "3.13\n")):
            self.assertEqual(pr._usable_python(_REAL_PYTHON), (3, 13))

    def test_an_interpreter_older_than_the_minimum_is_rejected(self):
        with mock.patch.object(pr.os.path, "isfile", return_value=True), \
             mock.patch.object(pr.subprocess, "run",
                               return_value=_completed(0, "3.8\n")):
            self.assertIsNone(pr._usable_python(_REAL_PYTHON))


class VersionOrderTest(unittest.TestCase):
    def test_newest_first_and_3_9_is_not_above_3_10(self):
        ordered = pr._dedupe_newest_first([
            r"C:\Python39\python.exe",
            r"C:\Python313\python.exe",
            r"C:\Python310\python.exe",
            r"C:\Python313\python.exe",  # a duplicate is dropped
        ])
        self.assertEqual(ordered, [
            r"C:\Python313\python.exe",
            r"C:\Python310\python.exe",
            r"C:\Python39\python.exe",
        ])


class LauncherTest(unittest.TestCase):
    def test_it_reads_the_paths_py_minus_0p_prints(self):
        listing = (" -V:3.13 *        C:\\Python313\\python.exe\n"
                   " -V:3.10          C:\\Python310\\python.exe\n")
        with mock.patch.object(pr.shutil, "which",
                               return_value=r"C:\Windows\py.exe"), \
             mock.patch.object(pr.subprocess, "run",
                               return_value=_completed(0, listing)), \
             mock.patch.object(pr.os.path, "isfile", return_value=True):
            self.assertEqual(pr._py_launcher_pythons(),
                             [r"C:\Python313\python.exe",
                              r"C:\Python310\python.exe"])


class DiscoveryTest(unittest.TestCase):
    """The search itself, with every source faked."""

    def setUp(self):
        patcher = mock.patch.object(sys, "frozen", True, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _find(self, which_map, usable, bundled="", staged=""):
        with mock.patch.object(pr, "_stage_python_runtime", return_value=staged), \
             mock.patch.object(paths, "bundled_python_exe",
                               return_value=bundled), \
             mock.patch.object(pr.shutil, "which",
                               side_effect=which_map.get), \
             mock.patch.object(pr, "_usable_python", side_effect=usable), \
             mock.patch.object(pr, "_glob_installs", return_value=[]), \
             mock.patch.object(pr, "_registry_installs", return_value=[]), \
             mock.patch.object(pr, "_py_launcher_pythons", return_value=[]):
            return pr._find_system_python()

    def test_the_bundled_runtime_is_preferred(self):
        # The path inside the program folder, used when the per-user copy could
        # not be made (see test_env_repair for the copy itself).
        bundled = r"C:\Program Files\AI Voice Studio\python-runtime\python.exe"
        result = self._find({}, lambda path: (3, 13), bundled=bundled,
                            staged="")
        self.assertEqual(result, bundled)

    def test_the_store_stub_is_skipped_for_a_real_python(self):
        real = _REAL_PYTHON

        def usable(path):
            return None if pr._is_store_alias(path) else (3, 13)

        result = self._find({"python3": _STORE_ALIAS, "python": real}, usable)
        self.assertEqual(result, real)

    def test_it_raises_when_nothing_is_usable(self):
        with self.assertRaises(FileNotFoundError):
            self._find({"python3": _STORE_ALIAS, "python": _STORE_ALIAS},
                       lambda path: None)


class BundledRuntimeTest(unittest.TestCase):
    def test_a_missing_runtime_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(paths, "user_data_dir", return_value=tmp), \
                 mock.patch.object(paths, "app_dir", return_value=tmp), \
                 mock.patch.object(sys, "_MEIPASS", "", create=True):
                self.assertEqual(paths.bundled_python_exe(), "")

    def test_the_installed_runtime_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_dir = os.path.join(tmp, "python-runtime")
            os.makedirs(runtime_dir)
            exe = os.path.join(runtime_dir, "python.exe")
            with open(exe, "w", encoding="utf-8") as fh:
                fh.write("")
            with mock.patch.object(paths, "user_data_dir", return_value=tmp), \
                 mock.patch.object(paths, "app_dir", return_value=tmp), \
                 mock.patch.object(sys, "_MEIPASS", "", create=True):
                self.assertEqual(paths.bundled_python_exe(), exe)


if __name__ == "__main__":
    unittest.main()
