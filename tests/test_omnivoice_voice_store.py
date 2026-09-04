"""Unit tests for the universal OmniVoice voice library (omnivoice/voice_store).

No GPU / pip packages are needed: creation is pure file/store work and the
engine-probe helper is stubbed out by passing explicit ``engine_ids``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import paths  # noqa: E402
from ai_voice_studio.omnivoice import voice_store  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402


class _StoreBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="aivs_voices_")
        fd, state = tempfile.mkstemp(prefix="aivs_models_", suffix=".json")
        os.close(fd)
        self._state = state
        self.store = ModelStore(state_file=state)
        # Route the voice folders into the throwaway temp dir.
        patcher = mock.patch.object(
            paths, "models_dir", return_value=os.path.join(self._tmp, "models")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)
        if os.path.exists(self._state):
            os.remove(self._state)

    def _sample(self, name="ref.wav") -> str:
        path = os.path.join(self._tmp, name)
        with open(path, "wb") as fh:
            fh.write(b"RIFFxxxxWAVEfmt dummy")
        return path


class CreateVoiceTest(_StoreBase):
    def test_create_clone_copies_sample_and_registers(self):
        sample = self._sample()
        entry = voice_store.create_voice(
            self.store, name="Mum", mode="clone", ref_audio=sample,
            ref_text="Hello there.",
        )
        self.assertEqual(entry["name"], "Mum")
        self.assertEqual(entry["kind"], voice_store.KIND_CLONE)
        self.assertEqual(entry["mode"], "clone")
        stored = voice_store.find_voice(self.store, "Mum")
        self.assertIsNotNone(stored)
        self.assertTrue(os.path.isfile(stored["sample"]))
        # The sample is a copy, not the original file.
        self.assertNotEqual(stored["sample"], sample)
        self.assertEqual(entry["ref_text"], "Hello there.")

    def test_create_design_stores_description(self):
        entry = voice_store.create_voice(
            self.store, name="Deep Narrator", mode="design",
            instruct="male, middle-aged, low pitch, british accent",
        )
        self.assertEqual(entry["kind"], voice_store.KIND_DESIGN)
        self.assertEqual(entry["instruct"],
                         "male, middle-aged, low pitch, british accent")
        self.assertFalse(entry.get("sample"))
        self.assertEqual(len(voice_store.omni_custom_voices(self.store)), 1)

    def test_create_clone_requires_existing_sample(self):
        with self.assertRaises(ValueError):
            voice_store.create_voice(
                self.store, name="X", mode="clone", ref_audio="C:/nope.wav"
            )
        self.assertEqual(voice_store.omni_custom_voices(self.store), [])

    def test_duplicate_names_rejected(self):
        sample = self._sample()
        voice_store.create_voice(self.store, name="Mum", mode="clone",
                                 ref_audio=sample)
        with self.assertRaises(ValueError):
            voice_store.create_voice(self.store, name="Mum", mode="design",
                                     instruct="female")
        # Case: name differs only by whitespace is sanitized to the same name.
        with self.assertRaises(ValueError):
            voice_store.create_voice(self.store, name="  Mum  ", mode="design",
                                     instruct="female")

    def test_folder_is_id_keyed_and_survives_rename(self):
        sample = self._sample()
        entry = voice_store.create_voice(self.store, name="First", mode="clone",
                                         ref_audio=sample)
        folder = entry["dir"]
        sample_path = entry["sample"]
        renamed = voice_store.rename_voice(self.store, "First", "Second name")
        self.assertEqual(renamed["name"], "Second name")
        self.assertEqual(renamed["dir"], folder)  # folder unchanged
        self.assertEqual(renamed["sample"], sample_path)
        self.assertTrue(os.path.isfile(sample_path))
        self.assertIsNone(voice_store.find_voice(self.store, "First"))
        self.assertIsNotNone(voice_store.find_voice(self.store, "Second name"))

    def test_delete_removes_folder(self):
        sample = self._sample()
        entry = voice_store.create_voice(self.store, name="Gone", mode="clone",
                                         ref_audio=sample)
        folder = entry["dir"]
        self.assertTrue(voice_store.delete_voice(self.store, "Gone"))
        self.assertFalse(os.path.isdir(folder))
        self.assertFalse(voice_store.delete_voice(self.store, "Gone"))

    def test_rename_collision_rejected(self):
        sample = self._sample()
        voice_store.create_voice(self.store, name="A", mode="clone",
                                 ref_audio=sample)
        voice_store.create_voice(self.store, name="B", mode="design",
                                 instruct="female")
        with self.assertRaises(ValueError):
            voice_store.rename_voice(self.store, "A", "B")


class ConsumerEntriesTest(_StoreBase):
    def _two_voices(self):
        sample = self._sample()
        voice_store.create_voice(self.store, name="Clone", mode="clone",
                                 ref_audio=sample, ref_text="Hi")
        voice_store.create_voice(self.store, name="Design", mode="design",
                                 instruct="female, elderly")

    def test_entries_cover_every_engine_and_variant(self):
        self._two_voices()
        entries = voice_store.consumer_entries(self.store, ["omnivoice",
                                                            "omnivoice_server"])
        # 2 voices x omnivoice(triton,hybrid) x server(server,server_clone)
        self.assertEqual(len(entries), 8)
        names = {e["voice"] for e in entries}
        self.assertEqual(names, {"Clone", "Design"})
        # Every entry carries the engine-specific identity.
        for e in entries:
            self.assertTrue(e["custom_omni"])
            self.assertIn(e["engine"], ("omnivoice", "omnivoice_server"))
            self.assertEqual(e["tts"], e["engine"])
            self.assertEqual(e["language"], "auto")
            omni = e["omni"]
            if e["voice"] == "Clone":
                self.assertEqual(omni["mode"], "clone")
                self.assertTrue(e["ref_audio"])
                self.assertEqual(omni["ref_text"], "Hi")
            else:
                self.assertEqual(omni["mode"], "design")
                self.assertIn("female, elderly", omni["instruct"])

    def test_only_requested_engines_are_emitted(self):
        self._two_voices()
        entries = voice_store.consumer_entries(self.store, ["omnivoice_server"])
        self.assertEqual({e["engine"] for e in entries}, {"omnivoice_server"})
        # Each voice under both server variants.
        self.assertEqual(len(entries), 4)

    def test_empty_library_yields_no_entries(self):
        self.assertEqual(voice_store.consumer_entries(self.store, ["omnivoice"]), [])

    def test_preview_entry_single(self):
        self._two_voices()
        voice = voice_store.find_voice(self.store, "Clone")
        entry = voice_store.preview_entry(voice, "omnivoice")
        self.assertEqual(entry["variant"], "triton")
        self.assertEqual(entry["omni"]["mode"], "clone")
        self.assertTrue(entry["ref_audio"])
        server = voice_store.preview_entry(voice, "omnivoice_server")
        self.assertEqual(server["variant"], "server")
        self.assertEqual(server["engine"], "omnivoice_server")


if __name__ == "__main__":
    unittest.main()
