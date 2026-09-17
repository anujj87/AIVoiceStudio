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

from ai_voice_studio import paths  # noqa: E402
from ai_voice_studio import python_runtime as pr  # noqa: E402
from ai_voice_studio.tts import catalog  # noqa: E402

#: Every pip-installed TTS engine: the Voice Lab engines plus the two
#: OmniVoice engines.  The ONNX engines (Piper, Kokoro, ...) ship with the
#: application and are deliberately not here.
_PIP_ENGINES = ("pocket_tts", "bark", "f5tts",
                "omnivoice", "omnivoice_server")

#: The engines that each get an environment the others cannot touch.
_OWN_ENV_ENGINES = ("pocket_tts", "bark", "f5tts", "omnivoice")


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
        os.makedirs(os.path.dirname(runtime.python_exe), exist_ok=True)
        with open(runtime.python_exe, "w", encoding="utf-8") as fh:
            fh.write("")
        self.assertTrue(pr.engine_env_exists("omnivoice"))
        self.assertIs(pr.engine_runtime("omnivoice"), runtime)
        self.assertIsNot(pr.engine_runtime("omnivoice"), pr.get_runtime())


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
        self.assertNotEqual(self.vp._key("torch", "bark"),
                            self.vp._key("torch", "f5tts"))
        self.assertNotEqual(self.vp._key("torch", "bark"),
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
