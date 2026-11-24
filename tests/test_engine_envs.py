"""Every pip-installed TTS engine gets its own Python virtualenv.

The application installs each pip-installed TTS engine (the Voice Lab engines
and the two OmniVoice engines) into a virtualenv of its own, so two engines can
never break each other's dependencies.  The one exception is the OmniVoice
pair: the direct engine and the HTTP server are the same model behind two
front-ends and share a single environment, because installing them separately
would only duplicate the same base package and CUDA PyTorch.

These tests cover the environment layout, the per-engine runtime lookup, the
OmniVoice sharing and the shared-environment fallback for engines installed
before those environments existed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import paths
from ai_voice_studio import python_runtime as pr
from ai_voice_studio.tts import catalog

#: Every pip-installed TTS engine: the Voice Lab engines plus the two
#: OmniVoice engines.  The ONNX engines (Piper, Kokoro, ...) ship with the
#: application and are deliberately not here.
_PIP_ENGINES = ("pocket_tts", "f5tts", "kokoclone",
                "chatterbox", "qwen3tts", "vibevoice",
                "moss_tts", "gptsovits",
                "indic_parler", "kitten_tts_2",
                "omnivoice", "omnivoice_server")

#: The engines that each get an environment the others cannot touch.
_OWN_ENV_ENGINES = ("pocket_tts", "f5tts", "kokoclone",
                    "chatterbox", "qwen3tts", "vibevoice",
                    "moss_tts", "gptsovits",
                    "indic_parler", "kitten_tts_2",
                    "omnivoice")


def _make_env(runtime):
    """Create what ``venv`` really leaves behind: an interpreter and its
    ``pyvenv.cfg`` (which is what makes the folder a usable environment)."""
    os.makedirs(os.path.dirname(runtime.python_exe), exist_ok=True)
    for path in (runtime.python_exe, runtime.marker_file):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
    return runtime


class EngineEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(paths, "user_data_dir",
                                    return_value=self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The runtimes are cached module-wide: start from a clean slate and
        # never leak the fakes into another test.
        pr._engine_runtimes.clear()
        pr._default_runtime = None
        self.addCleanup(pr._engine_runtimes.clear)

    def test_every_engine_gets_its_own_folder(self):
        folders = {engine_id: pr.engine_env_dir(engine_id)
                   for engine_id in _PIP_ENGINES}
        for engine_id in _OWN_ENV_ENGINES:
            self.assertTrue(folders[engine_id].startswith(self.tmp.name), engine_id)
            self.assertIn("tts_envs", folders[engine_id])
            self.assertTrue(os.path.basename(folders[engine_id]), engine_id)
        shared = [folders[engine_id] for engine_id in _OWN_ENV_ENGINES]
        self.assertEqual(len(set(shared)), len(_OWN_ENV_ENGINES),
                         "two engines share an environment unexpectedly")

    def test_the_two_omnivoice_engines_share_one_environment(self):
        # Same model, two front-ends: installing them separately would only
        # duplicate the same base package and CUDA PyTorch (~5 GB).
        self.assertEqual(pr.engine_env_dir("omnivoice_server"),
                         pr.engine_env_dir("omnivoice"))
        self.assertEqual(pr.environment_id("omnivoice_server"),
                         pr.environment_id("omnivoice"))
        self.assertIs(pr.get_runtime("omnivoice_server"),
                      pr.get_runtime("omnivoice"))

    def test_the_folder_name_is_path_safe(self):
        folder = pr.engine_env_dir("Some/Engine: v2")
        self.assertEqual(os.path.dirname(folder),
                         os.path.join(self.tmp.name, "tts_envs"))
        self.assertNotIn("/", os.path.basename(folder))
        self.assertNotIn(":", os.path.basename(folder))

    def test_each_engine_gets_its_own_runtime_object(self):
        runtimes = [pr.get_runtime(engine_id) for engine_id in _OWN_ENV_ENGINES]
        self.assertEqual(len({id(runtime) for runtime in runtimes}),
                         len(_OWN_ENV_ENGINES))
        addon = pr.get_runtime()
        self.assertNotIn(id(addon), {id(runtime) for runtime in runtimes})
        self.assertIs(pr.get_runtime(), addon, "the addon runtime is a singleton")
        for runtime in runtimes:
            self.assertNotEqual(runtime.env_dir, addon.env_dir)

    def test_an_engine_without_its_own_env_falls_back_to_the_addon_one(self):
        # Nothing created yet: the engine is looked up in the shared addon
        # environment (where installs from before the per-TTS environments
        # existed live).
        self.assertIs(pr.engine_runtime("omnivoice"), pr.get_runtime())
        self.assertFalse(pr.engine_env_exists("omnivoice"))

    def test_a_created_environment_wins_over_the_addon_one(self):
        runtime = pr.get_runtime("omnivoice")
        _make_env(runtime)
        self.assertTrue(pr.engine_env_exists("omnivoice"))
        self.assertIs(pr.engine_runtime("omnivoice"), runtime)
        self.assertIsNot(pr.engine_runtime("omnivoice"), pr.get_runtime())


class HalfDeletedEnvironmentTest(unittest.TestCase):
    """What an interrupted removal leaves behind must not block an install.

    Removing an engine while its worker is still running cannot delete the
    worker's interpreter - Windows keeps a running executable (and the DLLs it
    mapped) open - so the folder is left holding ``Scripts/python.exe`` and,
    crucially, *no* ``pyvenv.cfg``.  That leftover used to be mistaken for a
    finished environment: the next install skipped creating one and ran
    ``ensurepip`` with the half-deleted redirector instead, which fails with
    "failed to locate pyvenv.cfg" (exit status 106) - the raw message the user
    saw, "Command '[...ensurepip --upgrade]' returned non-zero exit status
    106".
    """

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

    def _runtime(self, engine_id="kokoclone"):
        return pr.get_runtime(engine_id)

    def _leftover(self, engine_id="kokoclone"):
        """A half-deleted environment: an interpreter, no ``pyvenv.cfg``."""
        runtime = self._runtime(engine_id)
        os.makedirs(os.path.dirname(runtime.python_exe), exist_ok=True)
        with open(runtime.python_exe, "w", encoding="utf-8") as fh:
            fh.write("")
        return runtime

    @staticmethod
    def _fake_venv(calls):
        """Stand in for ``python -m venv``: record it and build the env."""

        def run(command, **_kwargs):
            calls.append(list(command))
            if len(command) > 3 and command[1] == "-m" and command[2] == "venv":
                folder = command[3]
                os.makedirs(os.path.join(folder, "Scripts"), exist_ok=True)
                with open(os.path.join(folder, "pyvenv.cfg"), "w",
                          encoding="utf-8") as fh:
                    fh.write("home = fake\n")
                # A real environment comes with its interpreter and pip, so
                # no ensurepip bootstrap is needed.
                for name in ("python.exe", "pip.exe"):
                    with open(os.path.join(folder, "Scripts", name), "w",
                              encoding="utf-8") as fh:
                        fh.write("")
            return mock.Mock(returncode=0, stdout="", stderr="")

        return run

    def test_a_folder_without_pyvenv_cfg_is_not_a_finished_environment(self):
        runtime = self._leftover()
        self.assertTrue(os.path.isfile(runtime.python_exe))
        self.assertFalse(runtime.is_created,
                         "an interpreter alone is not a usable environment")

    def test_installing_over_a_half_deleted_environment_rebuilds_it(self):
        runtime = self._leftover()
        calls = []
        with mock.patch.object(pr.subprocess, "run",
                               side_effect=self._fake_venv(calls)):
            runtime.ensure_pip()
        # The environment was created again, and ensurepip was never asked to
        # finish the job with the leftover interpreter.
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][1:4], ["-m", "venv", runtime.env_dir])
        self.assertNotIn("ensurepip", " ".join(calls[0]))
        self.assertTrue(runtime.is_created)

    def test_a_leftover_folder_is_moved_aside_before_the_fresh_one(self):
        runtime = self._leftover()
        with mock.patch.object(pr.shutil, "rmtree", lambda *a, **k: None):
            calls = []
            with mock.patch.object(pr.subprocess, "run",
                                   side_effect=self._fake_venv(calls)):
                runtime.ensure_env()
            self.assertTrue(os.path.isdir(runtime.env_dir + ".leftover1"),
                            "the half-deleted folder must be moved aside")
        self.assertTrue(runtime.is_created)

    def test_a_removal_that_cannot_delete_everything_says_so(self):
        runtime = self._leftover()
        with mock.patch.object(pr.shutil, "rmtree", lambda *a, **k: None):
            note = runtime.remove_env()
        self.assertIsNotNone(note)
        self.assertIn("moved aside", note)
        # The environment path is free again, which is what lets the next
        # install succeed; the remains sit next to it.
        self.assertFalse(os.path.isdir(runtime.env_dir))
        self.assertTrue(os.path.isdir(runtime.env_dir + ".leftover1"))

    def test_a_clean_removal_reports_nothing(self):
        runtime = self._leftover()
        self.assertIsNone(runtime.remove_env())
        self.assertFalse(os.path.isdir(runtime.env_dir))

    def test_a_failed_pip_bootstrap_is_explained_instead_of_raw(self):
        runtime = _make_env(self._runtime("f5tts"))
        failure = pr.subprocess.CalledProcessError(
            106, ["python", "-m", "ensurepip"],
            stderr="failed to locate pyvenv.cfg",
        )
        with mock.patch.object(pr.subprocess, "run", side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                runtime.ensure_pip()
        message = str(caught.exception)
        self.assertIn("pip could not be installed", message)
        self.assertIn("106", message)
        self.assertIn("failed to locate pyvenv.cfg", message)


class SharedProbeCacheTest(unittest.TestCase):
    """Engines sharing an environment share the package probe answers."""

    def setUp(self):
        from ai_voice_studio import venv_packages as vp

        self.vp = vp
        with vp._lock:
            vp._cache.clear()
            vp._listeners.clear()
            vp._loading.clear()
        self.addCleanup(self._clear)

    def _clear(self):
        with self.vp._lock:
            self.vp._cache.clear()
            self.vp._listeners.clear()
            self.vp._loading.clear()

    def test_the_two_omnivoice_engines_collapse_onto_one_key(self):
        self.assertEqual(self.vp._key("omnivoice-triton", "omnivoice"),
                         self.vp._key("omnivoice-triton", "omnivoice_server"))
        # ... while the Voice Lab engines stay separate.
        self.assertNotEqual(self.vp._key("torch", "chatterbox"),
                            self.vp._key("torch", "f5tts"))
        self.assertNotEqual(self.vp._key("torch", "chatterbox"),
                            self.vp._key("torch", None))


class CatalogEnvironmentTest(unittest.TestCase):
    """The catalog tells the GUI which environment holds an engine."""

    def test_pip_engines_name_their_environment(self):
        for tts in catalog.get_tts_list():
            env_id = catalog.engine_env_id(tts)
            if tts.get("requires_package"):
                self.assertEqual(env_id, tts["id"], tts["id"])
            else:
                self.assertIsNone(env_id, tts["id"])

    def test_the_onnx_engines_are_not_pip_installed(self):
        for tts_id in ("piper", "kokoro", "kitten", "matcha"):
            tts = catalog.find_tts(tts_id)
            self.assertIsNotNone(tts, tts_id)
            self.assertIsNone(catalog.engine_env_id(tts), tts_id)

    def test_every_pip_engine_is_covered(self):
        pip_engines = {tts["id"] for tts in catalog.get_tts_list()
                       if tts.get("requires_package")}
        self.assertEqual(pip_engines, set(_PIP_ENGINES))


if __name__ == "__main__":
    unittest.main()
