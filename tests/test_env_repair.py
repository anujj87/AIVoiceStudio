"""An environment must survive the installation that created it going away.

Every environment the application creates is a *virtual* one, and a virtualenv
runs on top of the interpreter it was built from, whose absolute path it records
in its ``pyvenv.cfg``.  The environments live in ``%APPDATA%`` on purpose - so
uninstalling never throws away gigabytes of packages a user downloaded - which
means they also outlive the installation: reinstall the application somewhere
else, and every one of them is left pointing at a Python that is gone.

That failure is invisible from the files.  ``Scripts/python.exe`` is still
there and ``pyvenv.cfg`` is still there, so the environment looks installed;
starting it is the only way to find out, and what the user gets is

    Audio description failed: the model's process exited with code 103 without
    producing a description
    did not find executable at 'C:\\Program Files\\AI Voice Studio\\python-runtime
    \\python.exe': The system cannot find the path specified.

These tests pin both halves of the fix: environments are built from a per-user
copy of the bundled runtime, which cannot move, and an environment built from a
Python that has gone anyway is re-pointed at one that exists
(``python -m venv --upgrade``), which leaves the packages installed inside it
where they are.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import paths
from ai_voice_studio import python_runtime as pr

#: The kind of path that ends up in an environment's ``pyvenv.cfg`` when the
#: application was installed for all users and then reinstalled per-user.
_GONE_BASE = os.path.join(r"C:\Program Files\AI Voice Studio", "python-runtime",
                          "python.exe")


def _venv_launcher() -> str:
    """The small launcher a virtualenv's ``Scripts/python.exe`` is copied from.

    Part of every CPython installation (``venvlauncher.exe``; older releases
    shipped it as ``nt/python.exe``) and small, which is what lets these tests
    reproduce the real failure without copying a whole runtime around.
    """
    folder = os.path.join(sys.base_prefix, "Lib", "venv", "scripts", "nt")
    for name in ("venvlauncher.exe", "python.exe"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return os.path.join(folder, "venvlauncher.exe")


def _packaged_runtime() -> str:
    """The Python runtime inside a built payload, or "" when none is built."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for relative in (os.path.join("dist", "AIVS-64", "AI-Voice-Studio",
                                  "python-runtime"),
                     os.path.join("dist", "AIVS-32", "AI-Voice-Studio",
                                  "python-runtime")):
        exe = os.path.join(root, relative, "python.exe")
        if os.path.isfile(exe):
            return os.path.dirname(exe)
    return ""


class BasePythonTest(unittest.TestCase):
    """Reading back the interpreter an environment was built from."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = os.path.join(self.tmp.name, "env")
        os.makedirs(os.path.join(self.env, "Scripts"))

    def _config(self, text: str) -> None:
        with open(os.path.join(self.env, pr.PYVENV_CFG_NAME), "w",
                  encoding="utf-8") as handle:
            handle.write(text)

    def test_the_executable_line_is_what_venv_records(self):
        self._config("home = C:\\Python313\n"
                     "executable = C:\\Python313\\python.exe\n")
        self.assertEqual(pr.environment_base_python(self.env),
                         r"C:\Python313\python.exe")

    def test_an_older_environment_is_read_through_its_home_folder(self):
        self._config("home = C:\\Python313\nversion = 3.13.15\n")
        name = "python.exe" if sys.platform == "win32" else "python3"
        self.assertEqual(pr.environment_base_python(self.env),
                         os.path.join(r"C:\Python313", name))

    def test_a_base_that_has_gone_is_reported(self):
        self._config(f"home = {os.path.dirname(_GONE_BASE)}\n"
                     f"executable = {_GONE_BASE}\n")
        self.assertEqual(pr.missing_base_python(self.env), _GONE_BASE)

    def test_a_base_that_is_still_there_is_not_a_problem(self):
        self._config(f"home = {os.path.dirname(sys.executable)}\n"
                     f"executable = {sys.executable}\n")
        self.assertEqual(pr.missing_base_python(self.env), "")

    def test_a_folder_that_was_never_an_environment_reports_nothing(self):
        # No ``pyvenv.cfg`` at all.  That is a different failure - the folder is
        # not an environment - and it stays with ``_set_aside_leftover``.
        self.assertEqual(pr.missing_base_python(self.env), "")
        self.assertEqual(pr.environment_base_python(self.env), "")

    def test_a_config_without_a_home_reports_nothing(self):
        # Nothing to check against: better silent than a wrong repair.
        self._config("include-system-site-packages = false\n")
        self.assertEqual(pr.missing_base_python(self.env), "")


@unittest.skipUnless(sys.platform == "win32" and os.path.isfile(_venv_launcher()),
                     "needs the Windows virtualenv launcher")
class OrphanedLauncherTest(unittest.TestCase):
    """The failure itself, with the real launcher and no runtime to copy.

    ``Scripts/python.exe`` inside a virtualenv is a copy of that launcher: it
    reads ``pyvenv.cfg``, then starts the base interpreter named there.  When
    that folder has gone, the environment fails on its own - before a single
    line of the model's code runs - which is why the model looked like it had
    crashed.
    """

    def test_it_exits_103_and_names_the_interpreter_it_cannot_find(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.path.join(tmp, "env")
            os.makedirs(os.path.join(env, "Scripts"))
            shutil.copy2(_venv_launcher(),
                         os.path.join(env, "Scripts", "python.exe"))
            gone = os.path.join(tmp, "uninstalled", "python.exe")
            with open(os.path.join(env, pr.PYVENV_CFG_NAME), "w",
                      encoding="utf-8") as handle:
                handle.write(f"home = {os.path.dirname(gone)}\n"
                             f"executable = {gone}\nversion = 3.13.15\n")
            result = subprocess.run(
                [os.path.join(env, "Scripts", "python.exe"), "-c", "print(1)"],
                capture_output=True, text=True, cwd=tmp,
            )
        self.assertEqual(result.returncode, 103)
        self.assertIn("did not find executable at", result.stderr)
        self.assertIn(gone, result.stderr)


class RepairTest(unittest.TestCase):
    """An orphaned environment is re-pointed, not thrown away."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(paths, "user_data_dir",
                                    return_value=self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        pr._engine_runtimes.clear()
        pr._default_runtime = None
        self.addCleanup(pr._engine_runtimes.clear)

    def _orphaned(self, engine_id="kokoclone", packages=("torch",)):
        """What the application finds after being reinstalled somewhere else."""
        runtime = pr.get_runtime(engine_id)
        site = os.path.join(runtime.env_dir, "Lib", "site-packages")
        for package in packages:
            os.makedirs(os.path.join(site, package))
        os.makedirs(os.path.dirname(runtime.python_exe), exist_ok=True)
        with open(runtime.python_exe, "w", encoding="utf-8") as handle:
            handle.write("")
        with open(runtime.marker_file, "w", encoding="utf-8") as handle:
            handle.write(f"home = {os.path.dirname(_GONE_BASE)}\n"
                         f"executable = {_GONE_BASE}\nversion = 3.13.15\n")
        return runtime

    def _upgrade(self, base: str):
        """Stand in for ``python -m venv --upgrade``: record it, re-point the env.

        Writing the configuration ``venv`` would write is the whole point of the
        call, so the test does exactly that and nothing else.
        """
        def run(command, **_kwargs):
            self.calls.append(list(command))
            if len(command) > 1 and command[1] == "-m" and command[2] == "venv":
                # The environment is the one argument that is not a switch.
                folder = next((part for part in command[3:]
                               if not part.startswith("-")), "")
                if "--upgrade" in command:
                    with open(os.path.join(folder, pr.PYVENV_CFG_NAME), "w",
                              encoding="utf-8") as handle:
                        handle.write(f"home = {os.path.dirname(base)}\n"
                                     f"executable = {base}\nversion = 3.13.15\n")
                else:  # a fresh environment
                    os.makedirs(os.path.join(folder, "Scripts"), exist_ok=True)
                    with open(os.path.join(folder, pr.PYVENV_CFG_NAME), "w",
                              encoding="utf-8") as handle:
                        handle.write(f"home = {os.path.dirname(base)}\n"
                                     f"executable = {base}\nversion = 3.13.15\n")
                    for name in ("python.exe", "pip.exe"):
                        with open(os.path.join(folder, "Scripts", name), "w",
                                  encoding="utf-8") as handle:
                            handle.write("")
            return mock.Mock(returncode=0, stdout="", stderr="")
        return run

    def test_an_orphaned_environment_is_not_a_usable_one(self):
        runtime = self._orphaned()
        self.assertTrue(runtime.is_created, "the files are all still there")
        self.assertEqual(runtime.missing_base_python, _GONE_BASE)
        self.assertFalse(runtime.is_usable)

    def test_ensuring_the_environment_repoints_it_instead_of_rebuilding_it(self):
        runtime = self._orphaned()
        site = os.path.join(runtime.env_dir, "Lib", "site-packages")
        self.calls = []
        with mock.patch.object(pr.subprocess, "run",
                               side_effect=self._upgrade(sys.executable)), \
             mock.patch.object(pr, "_runs", return_value=True):
            self.assertEqual(runtime.ensure_env(), runtime.env_dir)
        # Exactly one call, and it is the in-place upgrade: never ``--clear``,
        # never a fresh environment, and pip is left as it is.
        self.assertEqual(self.calls, [[sys.executable, "-m", "venv", "--upgrade",
                                       "--without-pip", runtime.env_dir]])
        self.assertTrue(os.path.isdir(os.path.join(site, "torch")),
                        "the packages already installed must survive")
        self.assertFalse(os.path.isdir(runtime.env_dir + ".leftover1"))
        self.assertTrue(runtime.is_usable)
        self.assertNotIn("--clear", " ".join(self.calls[0]))

    def test_ensuring_the_environment_starts_over_when_it_cannot_be_repointed(self):
        runtime = self._orphaned()

        def run(command, **kwargs):
            if "--upgrade" in command:
                self.calls.append(list(command))
                return mock.Mock(returncode=1, stdout="",
                                 stderr="cannot re-point")
            # ``_upgrade`` records the call itself.
            return self._upgrade(sys.executable)(command, **kwargs)

        self.calls = []
        with mock.patch.object(pr.subprocess, "run", side_effect=run), \
             mock.patch.object(pr, "_runs", return_value=True):
            runtime.ensure_env()
        self.assertIn("--upgrade", self.calls[0])
        self.assertEqual(self.calls[1][1:4], ["-m", "venv", runtime.env_dir])
        self.assertTrue(runtime.is_created)
        self.assertTrue(runtime.is_usable)

    def test_a_launch_gets_an_interpreter_that_actually_runs(self):
        runtime = self._orphaned()
        self.calls = []
        with mock.patch.object(pr.subprocess, "run",
                               side_effect=self._upgrade(sys.executable)), \
             mock.patch.object(pr, "_runs", return_value=True):
            self.assertEqual(runtime.usable_python_exe(), runtime.python_exe)
        self.assertEqual(self.calls[0][1:4], ["-m", "venv", "--upgrade"])

    def test_a_launch_that_cannot_be_repaired_says_what_to_do(self):
        runtime = self._orphaned()
        with mock.patch.object(pr.subprocess, "run",
                               return_value=mock.Mock(returncode=1, stdout="",
                                                      stderr="nope")):
            with self.assertRaises(pr.EnvironmentUnusable) as caught:
                runtime.usable_python_exe()
        message = str(caught.exception)
        self.assertIn(runtime.env_dir, message)
        self.assertIn("Reinstall the model from Settings", message)

    def test_a_launch_never_builds_an_environment_of_its_own(self):
        # A model that was never installed must reach the caller's own "not
        # installed" message, not be handed an empty environment.
        runtime = pr.get_runtime("f5tts")
        with mock.patch.object(pr.subprocess, "run",
                               side_effect=AssertionError("must not run")):
            self.assertEqual(runtime.usable_python_exe(), runtime.python_exe)
        self.assertFalse(runtime.is_created)
        self.assertFalse(os.path.exists(runtime.marker_file))

    def test_a_healthy_environment_is_left_completely_alone(self):
        runtime = self._orphaned(engine_id="f5tts")
        with open(runtime.marker_file, "w", encoding="utf-8") as handle:
            handle.write(f"home = {os.path.dirname(sys.executable)}\n"
                         f"executable = {sys.executable}\n")
        with mock.patch.object(pr.subprocess, "run",
                               side_effect=AssertionError("must not run")):
            self.assertTrue(runtime.repair_env())
            self.assertEqual(runtime.usable_python_exe(), runtime.python_exe)


class StagingTest(unittest.TestCase):
    """Environments are built from a copy that cannot move."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shipped = os.path.join(self.tmp.name, "shipped")
        os.makedirs(self.shipped)
        with open(os.path.join(self.shipped, paths.PYTHON_EXE_NAME), "w",
                  encoding="utf-8") as handle:
            handle.write("the shipped interpreter")
        self.staged = os.path.join(self.tmp.name, "staged")

    def _patches(self, shipped=None):
        """The three patches a staging test needs, the shipped one adjustable."""
        return [
            shipped if shipped is not None else mock.patch.object(
                paths, "shipped_python_runtime", return_value=self.shipped),
            mock.patch.object(paths, "staged_python_runtime",
                              return_value=self.staged),
            mock.patch.object(
                pr, "_usable_python",
                side_effect=lambda path: (3, 13) if os.path.isfile(path) else None,
            ),
        ]

    @staticmethod
    def _add_build_support(folder):
        """Headers/libs, which a runtime needs to build a C extension itself.

        Triton compiles its ``cuda_utils`` helper the first time a kernel runs,
        so a runtime without them cannot run an engine that uses it.
        """
        include = os.path.join(folder, "Include")
        os.makedirs(include, exist_ok=True)
        with open(os.path.join(include, "Python.h"), "w", encoding="utf-8") as handle:
            handle.write("/* header */")
        libs = os.path.join(folder, "libs")
        os.makedirs(libs, exist_ok=True)
        with open(os.path.join(libs, "python313.lib"), "w", encoding="utf-8") as handle:
            handle.write("import library")

    def _start(self, *patches):
        """Start patches and stop them when the test is over.

        ``patch.start()`` returns the *replacement*, not the patcher, so the
        patchers are kept to stop them - a test that asks for the return value
        leaves the mock in place and quietly breaks the next test that reads
        the same function.
        """
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_shipped_runtime_is_copied_into_the_user_folder(self):
        self._start(*self._patches())
        exe = pr._stage_python_runtime()
        self.assertEqual(exe, os.path.join(self.staged, paths.PYTHON_EXE_NAME))
        self.assertTrue(os.path.isfile(exe))

    def test_an_existing_copy_is_used_without_copying_again(self):
        os.makedirs(self.staged)
        with open(os.path.join(self.staged, paths.PYTHON_EXE_NAME), "w",
                  encoding="utf-8") as handle:
            handle.write("already here")
        self._add_build_support(self.staged)
        # A complete copy stands on its own: the shipped runtime must not even
        # be looked at.
        self._start(*self._patches(
            shipped=mock.patch.object(paths, "shipped_python_runtime",
                                      side_effect=AssertionError("copied again")),
        ))
        self.assertEqual(pr._stage_python_runtime(),
                         os.path.join(self.staged, paths.PYTHON_EXE_NAME))

    def test_a_staged_copy_without_headers_is_topped_up_from_the_shipped_one(self):
        """A copy staged before the headers shipped gains them, in place."""
        os.makedirs(self.staged)
        with open(os.path.join(self.staged, paths.PYTHON_EXE_NAME), "w",
                  encoding="utf-8") as handle:
            handle.write("already here")
        self._add_build_support(self.shipped)
        self._start(*self._patches())
        self.assertEqual(pr._stage_python_runtime(),
                         os.path.join(self.staged, paths.PYTHON_EXE_NAME))
        self.assertTrue(os.path.isfile(
            os.path.join(self.staged, "Include", "Python.h")))
        self.assertTrue(os.path.isfile(
            os.path.join(self.staged, "libs", "python313.lib")))
        # The interpreter itself is left exactly as it was.
        with open(os.path.join(self.staged, paths.PYTHON_EXE_NAME),
                  encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "already here")

    def test_a_fresh_copy_carries_the_headers_and_libraries(self):
        self._add_build_support(self.shipped)
        self._start(*self._patches())
        pr._stage_python_runtime()
        self.assertTrue(os.path.isfile(
            os.path.join(self.staged, "Include", "Python.h")))
        self.assertTrue(os.path.isfile(
            os.path.join(self.staged, "libs", "python313.lib")))

    def test_a_build_without_a_shipped_runtime_stages_nothing(self):
        self._start(*self._patches(
            shipped=mock.patch.object(paths, "shipped_python_runtime",
                                      return_value=""),
        ))
        self.assertEqual(pr._stage_python_runtime(), "")
        self.assertFalse(os.path.isdir(self.staged))

    def test_the_copy_is_preferred_over_the_shipped_runtime(self):
        staged_exe = os.path.join(self.staged, paths.PYTHON_EXE_NAME)
        frozen = mock.patch.object(sys, "frozen", True, create=True)
        frozen.start()
        self.addCleanup(frozen.stop)
        with mock.patch.object(pr, "_stage_python_runtime",
                               return_value=staged_exe), \
             mock.patch.object(paths, "bundled_python_exe",
                               side_effect=AssertionError("used the shipped one")):
            self.assertEqual(pr._find_system_python(), staged_exe)

    def test_the_shipped_runtime_is_used_when_it_cannot_be_copied(self):
        shipped_exe = os.path.join(self.shipped, paths.PYTHON_EXE_NAME)
        frozen = mock.patch.object(sys, "frozen", True, create=True)
        frozen.start()
        self.addCleanup(frozen.stop)
        with mock.patch.object(pr, "_stage_python_runtime", return_value=""), \
             mock.patch.object(paths, "bundled_python_exe",
                               return_value=shipped_exe), \
             mock.patch.object(pr, "_usable_python", return_value=(3, 13)):
            self.assertEqual(pr._find_system_python(), shipped_exe)


@unittest.skipUnless(sys.platform == "win32" and _packaged_runtime(),
                     "needs a built payload (packaging\\build.ps1)")
class RealEnvironmentTest(unittest.TestCase):
    """The whole story with a real runtime: it breaks, it heals, packages stay.

    Kept out of the way of the fast tests because it copies the shipped runtime
    (about 35 MB) and lets ``venv`` build an environment from it.
    """

    def test_a_deleted_base_is_repointed_and_the_packages_survive(self):
        packaged = _packaged_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "python-runtime")
            shutil.copytree(packaged, base)
            env = os.path.join(tmp, "env")
            created = subprocess.run(
                [os.path.join(base, "python.exe"), "-m", "venv", env],
                capture_output=True, text=True,
            )
            self.assertEqual(created.returncode, 0, created.stderr)

            # A package the user would rather not download again.
            site = os.path.join(env, "Lib", "site-packages")
            os.makedirs(os.path.join(site, "torch"))
            with open(os.path.join(site, "torch", "__init__.py"), "w",
                      encoding="utf-8") as handle:
                handle.write("VERSION = '2.9'\n")

            # The application is uninstalled: the Python the environment names
            # is gone, exactly as it is after reinstalling somewhere else.
            recorded = os.path.join(base, "python.exe")
            gone = os.path.join(tmp, "old-installation")
            shutil.move(base, gone)
            broken = subprocess.run(
                [os.path.join(env, "Scripts", "python.exe"), "-c", "print(1)"],
                capture_output=True, text=True, cwd=tmp,
            )
            self.assertEqual(broken.returncode, 103)
            self.assertIn("did not find executable at", broken.stderr)

            runtime = pr.PythonRuntime(env)
            self.assertEqual(os.path.normcase(runtime.missing_base_python),
                             os.path.normcase(recorded))
            self.assertFalse(runtime.is_usable)

            # What the new installation does: the environment is pointed at the
            # runtime it ships.  (Patching the search is how the frozen
            # application arrives at the same interpreter.)
            with mock.patch.object(
                pr, "_find_system_python",
                return_value=os.path.join(packaged, "python.exe"),
            ):
                self.assertTrue(runtime.repair_env())
            self.assertTrue(runtime.is_usable)

            alive = subprocess.run(
                [runtime.python_exe, "-c",
                 "import torch, sys; print(torch.VERSION, sys.prefix != sys.base_prefix)"],
                capture_output=True, text=True, cwd=tmp,
            )
            self.assertEqual(alive.returncode, 0, alive.stderr)
            self.assertIn("2.9", alive.stdout)
            self.assertIn("True", alive.stdout)
            self.assertTrue(os.path.isfile(os.path.join(env, "Scripts", "pip.exe")),
                            "the environment's own pip must be left alone")


if __name__ == "__main__":
    unittest.main()
