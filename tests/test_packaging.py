"""Tests that keep the unattended install path honest.

A scripted install is the one workflow nothing else covers. ``test_winget.py``
checks that the installer *answers* the switches and ``PROJECT_SPEC`` requires
them, but neither notices how the switches are handed over: a shell that
translates POSIX paths rewrites a leading ``/VERYSILENT`` into
``C:/Program Files/Git/VERYSILENT`` before Setup sees it.  Setup then ignores
what it does not recognise and opens its wizard, so an "unattended" install
waits on a dialog nobody is there to answer - it never exits, and nothing in
the build fails.

``packaging/silent_install.ps1`` is the answer to that, and it also waits for
Inno's uninstaller, which copies itself to a temporary file, exits 0 and only
then removes the files.  These tests hold the driver, the switches winget
advertises and the README that tells a maintainer to use it together.

The other half of the shipped application that nothing else covers is the
executable's own name and version.  Windows - and every screen reader
answering "which application is this, and which version?" - reads them from
the file's VERSIONINFO resource, which ``packaging/ai_voice_studio.spec``
takes from ``tools/make_version_info.py``; the tests below hold that resource
against the version the application itself reports.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.constants import APP_NAME, APP_VERSION
from tools import make_version_info
from tools.winget_manifests import SILENT_SWITCH

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVER = os.path.join(ROOT, "packaging", "silent_install.ps1")
README = os.path.join(ROOT, "README.md")
SPEC = os.path.join(ROOT, "PROJECT_SPEC.md")
PYINSTALLER_SPEC = os.path.join(ROOT, "packaging", "ai_voice_studio.spec")
BUILD = os.path.join(ROOT, "packaging", "build.ps1")
STAGE = os.path.join(ROOT, "packaging", "stage_python_runtime.py")
INSTALLER_COMMON = os.path.join(ROOT, "packaging", "installer_common.iss")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class SilentInstallDriverTest(unittest.TestCase):
    """``packaging/silent_install.ps1`` must pass the whole switch set."""

    @classmethod
    def setUpClass(cls):
        cls.driver = _read(DRIVER)

    def test_it_exists_and_is_a_powershell_script(self):
        self.assertTrue(os.path.isfile(DRIVER))
        self.assertIn("param(", self.driver)

    def test_it_passes_every_switch_the_manifests_advertise(self):
        # The driver and the winget manifest must install the same way; if
        # winget gains or loses a switch, this fails rather than drifting.
        for switch in SILENT_SWITCH.split():
            self.assertIn(switch, self.driver, f"driver does not pass {switch}")

    def test_it_chooses_the_install_mode_per_user_by_default(self):
        # An unattended install must never stop on the mode dialog, and the
        # recommended choice without elevation is the per-user one.
        self.assertIn('"/CURRENTUSER"', self.driver)
        self.assertIn('"/ALLUSERS"', self.driver)

    def test_it_clears_the_bytecode_caches_the_runtime_writes(self):
        """An uninstall must finish, even after the app has been run.

        The bundled runtime writes ``__pycache__`` files next to its own
        modules the first time anything runs it.  Inno's uninstaller removes
        only the files Setup installed, so those caches stay - and a wait that
        insists on an empty folder then runs out its clock and calls a
        complete uninstall a timeout (measured: 47 ``.pyc`` files, 600 seconds).
        """
        self.assertIn("Get-RuntimeCache", self.driver)
        self.assertIn(".pyc", self.driver)
        self.assertIn("bytecode cache", self.driver)
        self.assertIn('"/SUPPRESSMSGBOXES"', self.driver)

    def test_it_can_aim_the_install_and_its_log(self):
        self.assertIn('"/DIR=$Dir"', self.driver)
        self.assertIn('"/LOG=$Log"', self.driver)

    def test_it_waits_for_the_uninstaller_instead_of_its_exit_code(self):
        # Inno's unins000.exe returns 0 while a copy of it is still deleting
        # files, so the driver has to wait for the copy.
        self.assertIn("Wait-ForRemoval", self.driver)
        self.assertIn("Wait-ForRemoval -Path $target", self.driver)
        self.assertIn("finishes in a copy of itself", self.driver)

    def test_a_failure_is_reported_as_a_nonzero_exit(self):
        self.assertIn("exit 1", self.driver)

    def test_the_packaging_scripts_point_at_the_driver(self):
        # A maintainer editing the switches, or who has just built an
        # installer, should learn how they are exercised unattended.
        for name in ("installer_common.iss", "build.ps1"):
            text = _read(os.path.join(ROOT, "packaging", name))
            self.assertIn("silent_install.ps1", text, f"{name} does not point at the driver")


class SilentInstallDocumentationTest(unittest.TestCase):
    """The README has to warn about the trap the driver avoids."""

    @classmethod
    def setUpClass(cls):
        cls.readme = _read(README)
        cls.spec = _read(SPEC)

    def test_the_readme_names_the_driver(self):
        self.assertIn("silent_install.ps1", self.readme)

    def test_the_readme_warns_that_a_posix_shell_rewrites_the_switches(self):
        self.assertIn("Git Bash", self.readme)
        self.assertIn("/VERYSILENT", self.readme)
        self.assertIn("C:/Program Files/Git", self.readme)

    def test_the_spec_requires_the_unattended_path(self):
        self.assertIn("silent_install.ps1", self.spec)
        self.assertIn("/VERYSILENT", self.spec)

    def test_the_readme_explains_what_an_uninstall_leaves_behind(self):
        # The caches are the runtime's, not the application's, so the README
        # has to say so where a maintainer runs the driver.
        self.assertIn("bytecode cache", self.readme)


class ApplicationVersionResourceTest(unittest.TestCase):
    """The executable's own name and version - what Windows and NVDA read.

    NVDA's *report the application version* gesture asks Windows for the
    product name and the product version of the running executable.  A
    PyInstaller build whose EXE carries no VERSIONINFO resource answers
    nothing at all, and the application is announced as "Application unknown,
    version not detected" however correct its About box is.
    """

    def test_the_resource_names_the_application_and_its_version(self):
        got = make_version_info.fields()
        # ProductName and ProductVersion are the two NVDA asks for.
        self.assertEqual(got["ProductName"], APP_NAME)
        self.assertEqual(got["ProductVersion"], APP_VERSION)
        self.assertEqual(got["FileVersion"], APP_VERSION)
        self.assertEqual(got["FileDescription"], APP_NAME)
        self.assertEqual(got["InternalName"], "AI-Voice-Studio")
        self.assertEqual(got["OriginalFilename"], "AI-Voice-Studio.exe")
        self.assertTrue(got["CompanyName"])
        self.assertTrue(got["LegalCopyright"])

    def test_the_binary_version_is_four_numbers(self):
        self.assertEqual(make_version_info.version_tuple("2027.1.0"),
                         (2027, 1, 0, 0))
        self.assertEqual(make_version_info.version_tuple("2027.1"),
                         (2027, 1, 0, 0))
        self.assertEqual(make_version_info.version_tuple("2027.1.0-rc1"),
                         (2027, 1, 0, 0))
        # The version the whole application reports has to survive the
        # conversion, or the resource would disagree with the About box.
        expected = tuple(int(part) for part in APP_VERSION.split("."))
        numbers = make_version_info.version_tuple(APP_VERSION)
        self.assertEqual(numbers[:len(expected)], expected)
        self.assertEqual(numbers[len(expected):], (0,) * (4 - len(expected)))

    def test_the_spec_gives_the_resource_to_pyinstaller(self):
        text = _read(PYINSTALLER_SPEC)
        self.assertIn("make_version_info", text)
        self.assertIn("version=version_info()", text)

    def test_a_built_executable_carries_the_resource(self):
        """Skipped until the application has been built into ``dist/``."""
        path = os.path.join(ROOT, "dist", "AIVS-64", "AI-Voice-Studio",
                            "AI-Voice-Studio.exe")
        if not os.path.isfile(path):
            self.skipTest("the application has not been built into dist/AIVS-64")
        found = make_version_info.read_version_info(path)
        self.assertEqual(found.get("ProductName"), APP_NAME,
                         "a screen reader reports this file as unknown")
        self.assertEqual(found.get("ProductVersion"), APP_VERSION)
        self.assertEqual(found.get("FileVersion"), APP_VERSION)


class BundledPythonRuntimeTest(unittest.TestCase):
    """The setup always carries the Python runtime the engines build on.

    Every engine and STT model gets its own virtualenv, and only a full
    interpreter can create one, so the build stages a complete CPython next to
    the application and the installer ships that folder whole.  While the
    staging step could skip quietly the build still produced a setup - it just
    installed an application that could create no environment at all (the first
    symptom was Triton failing with "include file 'Python.h' not found").
    """

    @classmethod
    def setUpClass(cls):
        cls.build = _read(BUILD)
        cls.installer = _read(INSTALLER_COMMON)
        spec = importlib.util.spec_from_file_location("stage_python_runtime", STAGE)
        cls.stage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.stage_module)

    def test_the_build_stages_the_runtime_before_it_compiles_the_setup(self):
        self.assertIn("stage_python_runtime.py", self.build)
        self.assertLess(self.build.index("stage_python_runtime.py"),
                        self.build.index("& $iscc"))

    def test_the_build_checks_the_payload_before_compiling_it(self):
        self.assertIn("function Assert-BundledRuntime", self.build)
        self.assertLess(self.build.index("Assert-BundledRuntime $distArch"),
                        self.build.index("& $iscc"))
        self.assertIn("throw", self.build)

    def test_the_setup_script_refuses_a_dist_without_the_runtime(self):
        # iscc can also be run by hand (the .iss headers say so), so the
        # compiler itself has to refuse a payload that has no runtime.
        self.assertIn("#if !FileExists(", self.installer)
        self.assertIn("python-runtime", self.installer)
        self.assertIn("#error", self.installer)
        self.assertIn("stage_python_runtime.py", self.installer)

    def test_a_staging_failure_is_a_build_failure(self):
        argv = ["stage_python_runtime.py", "--arch", "x64"]
        with mock.patch.object(self.stage_module, "stage", return_value=False), \
                mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.stage_module.main(), 1)
        with mock.patch.object(self.stage_module, "stage", return_value=True), \
                mock.patch.object(sys, "argv", argv):
            self.assertEqual(self.stage_module.main(), 0)

    def test_an_incomplete_copy_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = self.stage_module.missing_parts(os.path.join(tmp, "runtime"))
        for part in ("python.exe", "Lib/venv/__init__.py", "Lib/ensurepip",
                     "Include/Python.h", "libs/python3XX.lib"):
            self.assertIn(part, missing)

    def test_a_complete_copy_is_accepted(self):
        parts = ("python.exe", "Lib/venv/__init__.py", "Lib/ensurepip/__init__.py",
                 "Include/Python.h", "libs/python313.lib")
        with tempfile.TemporaryDirectory() as tmp:
            for relative in parts:
                path = pathlib.Path(tmp, *relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            self.assertEqual(self.stage_module.missing_parts(tmp), [])


if __name__ == "__main__":
    unittest.main()
