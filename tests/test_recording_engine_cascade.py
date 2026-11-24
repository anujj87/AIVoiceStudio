"""The Recording window's cascade for the ordinary (non-OmniVoice) engines.

The TTS combo lists every installed engine, and picking one has to fill the
Language, Variant and Voice boxes from that engine's downloaded voices.  The
filter that decides which voice entries belong to the selected engine used to
compare the entry's ``engine`` field - its runtime kind, ``"vits"`` for Piper
and the Chinese VITS voices - against a set of catalog ids, so Piper showed up
in the TTS combo with all three boxes empty.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import wx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wx_test_app import get_app

from ai_voice_studio import project, venv_packages
from ai_voice_studio.constants import MODE_PAGE_WITH_H1
from ai_voice_studio.gui.recording_dialog import RecordingDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts import catalog
from ai_voice_studio.tts.models import ModelStore


class RecordingEngineCascadeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def _dialog(self, tts_id: str) -> RecordingDialog:
        """A Recording window with one installed variant of ``tts_id``."""
        tts = catalog.find_tts(tts_id)
        self.assertIsNotNone(tts)
        language = tts["languages"][0]
        variant = language["variants"][0]
        root = tempfile.mkdtemp(prefix=f"aivs_cascade_{tts_id}_")
        self.addCleanup(shutil.rmtree, root, True)
        project_dir = os.path.join(root, "project")
        os.makedirs(project_dir)
        project.create_project(
            project_dir, "Cascade project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        state_file = os.path.join(root, "models.json")
        store = ModelStore(state_file=state_file)
        variant_dir = os.path.join(root, "models", tts_id, language["code"],
                                   variant["id"])
        os.makedirs(variant_dir, exist_ok=True)
        store.mark_installed(tts_id, language["code"], variant["id"], variant_dir)
        settings = Settings(path=os.path.join(root, "settings.json"))
        # No pip-installed engine: the cascade must work from downloaded
        # voices alone (and the probe must not go looking for packages).
        with mock.patch.object(venv_packages, "installed",
                               lambda pkg, engine=None: False), \
                mock.patch.object(venv_packages, "is_known",
                                  lambda pkg, engine=None: True):
            dlg = RecordingDialog(self.frame, project_dir, settings, store)
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_a_plain_vits_engine_fills_language_variant_and_voice(self):
        """Piper and the Chinese VITS voices are the ``vits`` runtime kind.

        Their entries carry ``engine == "vits"``, not the catalog id, which is
        what the old filter compared against - and why both engines listed
        their TTS entry with empty boxes.
        """
        for tts_id in ("piper", "vits_zh"):
            with self.subTest(tts=tts_id):
                dlg = self._dialog(tts_id)
                self.assertEqual(dlg._selected_tts_id(), tts_id)
                self.assertGreater(dlg.lang_combo.GetCount(), 0)
                self.assertGreater(dlg.variant_combo.GetCount(), 0)
                self.assertGreater(dlg.voice_combo.GetCount(), 0)
                voice = dlg._selected_voice()
                self.assertIsNotNone(voice)
                self.assertEqual(voice["tts"], tts_id)
                self.assertEqual(voice["language"],
                                 catalog.find_tts(tts_id)["languages"][0]["code"])

    def test_an_engine_whose_kind_is_its_catalog_id_still_fills(self):
        """Kokoro keeps working (its ``engine`` field equals its id)."""
        dlg = self._dialog("kokoro")
        self.assertEqual(dlg._selected_tts_id(), "kokoro")
        self.assertGreater(dlg.lang_combo.GetCount(), 0)
        self.assertGreater(dlg.variant_combo.GetCount(), 0)
        self.assertGreater(dlg.voice_combo.GetCount(), 0)


if __name__ == "__main__":
    unittest.main()
