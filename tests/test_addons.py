"""Addon integration: a manifest's voices reach the model store.

The Addon Development Guide promises that *"addon-provided voices appear in the
Recording window's voice selector"*.  Discovery alone only *collects* them
(``AddonManager.get_all_voices``); somebody has to put them in the store the
Recording window reads, and ``main.py`` does that at start-up.  Nothing called
``addons.register_addon_voices`` before these tests existed, so an addon could
declare voices and see none of them - the tests hold both halves: the
registration itself, and the fact that start-up performs it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main
from ai_voice_studio import addons
from ai_voice_studio.addons import AddonManager, register_addon_voices
from ai_voice_studio.tts.models import ModelStore


def _temp_store() -> ModelStore:
    """A store with no voices, in a file of its own (never the user's)."""
    fd, path = tempfile.mkstemp(prefix="aivs_addon_models_", suffix=".json")
    os.close(fd)
    return ModelStore(state_file=path)


class _AddonCase(unittest.TestCase):
    """A throwaway addons directory with one addon in it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aivs_addons_")
        self.addCleanup(self.tmp.cleanup)
        self.addons_dir = self.tmp.name

    def _write_addon(self, name: str, data: dict, *, files: bool = True) -> str:
        addon_dir = os.path.join(self.addons_dir, name)
        os.makedirs(addon_dir, exist_ok=True)
        with open(os.path.join(addon_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        if files:
            # A voice folder with something in it, so the registration has a real
            # directory to hand the store.
            voice_dir = os.path.join(addon_dir, "voice_data")
            os.makedirs(voice_dir, exist_ok=True)
            with open(os.path.join(voice_dir, "model.onnx"), "wb") as fh:
                fh.write(b"not really a model")
        return addon_dir

    def _manager(self) -> AddonManager:
        manager = AddonManager(addons_dir=self.addons_dir)
        manager.discover()
        return manager


class AddonVoiceRegistrationTest(_AddonCase):

    def test_a_manifest_voice_reaches_the_store(self):
        self._write_addon("demo", {
            "name": "Demo Addon",
            "version": "1.0",
            "engine_id": "demo_engine",
            "voices": [{
                "id": "demo_en",
                "name": "Demo English",
                "language": "en",
                "data_dir": "voice_data",
            }],
        })
        store = _temp_store()
        with mock.patch.object(addons, "get_addon_manager",
                               return_value=self._manager()):
            register_addon_voices(store)

        voices = store.custom_voices()
        self.assertEqual([v["name"] for v in voices], ["Demo English"])
        voice = voices[0]
        self.assertEqual(voice["tts"], "demo_engine")
        self.assertEqual(voice["kind"], "addon")
        self.assertEqual(voice["addon"], "Demo Addon")
        self.assertEqual(voice["voice_id"], "demo_en")
        self.assertEqual(voice["language"], "en")
        # The manifest's relative data_dir is resolved against the addon folder.
        self.assertTrue(os.path.isdir(voice["dir"]))
        self.assertTrue(voice["dir"].startswith(self.addons_dir))

    def test_a_voice_can_name_its_own_engine(self):
        self._write_addon("demo", {
            "name": "Demo Addon",
            "engine_id": "demo_engine",
            "voices": [{
                "id": "piper_en",
                "name": "Piper English",
                "engine_id": "piper",
                "data_dir": "voice_data",
            }],
        })
        store = _temp_store()
        with mock.patch.object(addons, "get_addon_manager",
                               return_value=self._manager()):
            register_addon_voices(store)
        self.assertEqual(store.custom_voices()[0]["tts"], "piper")

    def test_a_voice_that_ships_no_model_files_is_left_out(self):
        # An addon engine that synthesises by itself declares a voice with no
        # folder: there is nothing to load, so nothing is registered.
        self._write_addon("demo", {
            "name": "Demo Addon",
            "engine_id": "demo_engine",
            "voices": [{"id": "tone", "name": "Test Tone", "data_dir": ""}],
        }, files=False)
        store = _temp_store()
        with mock.patch.object(addons, "get_addon_manager",
                               return_value=self._manager()):
            register_addon_voices(store)
        self.assertEqual(store.custom_voices(), [])

    def test_a_voice_whose_folder_is_missing_is_left_out(self):
        self._write_addon("demo", {
            "name": "Demo Addon",
            "voices": [{
                "id": "gone", "name": "Missing Voice", "data_dir": "no_such_dir",
            }],
        }, files=False)
        store = _temp_store()
        with mock.patch.object(addons, "get_addon_manager",
                               return_value=self._manager()):
            register_addon_voices(store)
        self.assertEqual(store.custom_voices(), [])

    def test_a_disabled_addon_contributes_no_voices(self):
        self._write_addon("demo", {
            "name": "Demo Addon",
            "enabled": False,
            "voices": [{
                "id": "demo_en", "name": "Demo English", "data_dir": "voice_data",
            }],
        })
        store = _temp_store()
        with mock.patch.object(addons, "get_addon_manager",
                               return_value=self._manager()):
            register_addon_voices(store)
        self.assertEqual(store.custom_voices(), [])


class StartupRegistrationTest(unittest.TestCase):
    """The step itself has to be reached from the application's start-up."""

    def test_start_up_registers_addon_voices(self):
        calls = []

        def fake_register(store):
            calls.append(store)

        with mock.patch.object(addons, "register_addon_voices", fake_register):
            main._register_addon_voices("the store")

        self.assertEqual(calls, ["the store"])

    def test_a_broken_addon_never_stops_the_application_starting(self):
        with mock.patch.object(addons, "register_addon_voices",
                               side_effect=RuntimeError("bad manifest")):
            main._register_addon_voices(_temp_store())  # must not raise

    def test_the_start_up_sequence_asks_for_them(self):
        """``main()`` registers the voices before it builds the main window."""
        registered: list[str] = []
        order: list[str] = []

        class _Frame:
            def __init__(self, *_args, **_kwargs):
                order.append("frame")

            def Show(self):
                pass

        fake_settings = mock.MagicMock()
        fake_settings.get.return_value = False   # basic logging, no developer mode
        fake_settings.theme = "light"

        class _App:
            def __init__(self, *_args, **_kwargs):
                pass

            def MainLoop(self):
                return 0

        with mock.patch.object(main, "Settings", return_value=fake_settings), \
                mock.patch.object(main, "_setup_logging"), \
                mock.patch.object(main, "_require_terms_acceptance", return_value=True), \
                mock.patch.object(main, "_init_python_runtime"), \
                mock.patch.object(main, "_init_addons"), \
                mock.patch.object(main, "_warm_package_cache"), \
                mock.patch.object(main, "_schedule_update_check"), \
                mock.patch.object(main, "_register_addon_voices",
                                  lambda store: (order.append("voices"),
                                                 registered.append(store))), \
                mock.patch.object(main, "ModelStore"), \
                mock.patch.object(main, "MainFrame", _Frame), \
                mock.patch.object(main, "apply_theme"), \
                mock.patch.object(main.wx, "App", _App):
            main.main()

        self.assertEqual(len(registered), 1, "start-up must register addon voices once")
        self.assertIn("voices", order)
        self.assertLess(order.index("voices"), order.index("frame"),
                        "the store must hold the voices before the window lists them")


if __name__ == "__main__":
    unittest.main()
