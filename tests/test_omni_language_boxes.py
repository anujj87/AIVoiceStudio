"""The OmniVoice Language box: Auto plus all 646 languages, everywhere.

Every place the application offers a language for OmniVoice - the Recording
window and Settings > Available TTS - offers the same list, and every one of
them means the same thing: a *hint* to the engine.  Picking a language pins it
(OmniVoice's own detection can read a short line with the wrong accent, or as
gibberish); picking Auto leaves that detection in charge.  The list never
removes voices, because an OmniVoice voice can speak any of the model's
languages.  The voice library (Settings > OmniVoice engines) can pin a created
voice to one language as well.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

import wx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import project, venv_packages  # noqa: E402
from ai_voice_studio.constants import MODE_PAGE_WITH_H1  # noqa: E402
from ai_voice_studio.gui import language_choice  # noqa: E402
from ai_voice_studio.gui.model_panels import AvailablePanel  # noqa: E402
from ai_voice_studio.gui.recording_dialog import RecordingDialog  # noqa: E402
from ai_voice_studio.gui.settings_dialog import _OmniVoiceEnginesPanel  # noqa: E402
from ai_voice_studio.omnivoice import languages, spec, voice_store  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402


class _AppMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = wx.App(False)
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def _store(self, state_path: str | None = None) -> ModelStore:
        """A store backed by a throwaway state file (no voices unless added)."""
        if state_path is None:
            fd, state_path = tempfile.mkstemp(prefix="aivs_models_", suffix=".json")
            os.close(fd)
            self.addCleanup(lambda: os.path.exists(state_path) and os.remove(state_path))
        return ModelStore(state_file=state_path)

    def _installed(self):
        """Pretend every pip engine (OmniVoice, Voice Lab) is installed."""
        for patch in (
            mock.patch.object(venv_packages, "installed",
                              lambda pkg, engine=None: True),
            mock.patch.object(venv_packages, "is_known",
                              lambda pkg, engine=None: True),
        ):
            patch.start()
            self.addCleanup(patch.stop)


class LanguageChoiceHelperTest(_AppMixin):
    """The shared helper both language boxes are built from."""

    def _combo(self):
        combo = wx.ComboBox(self.frame, style=wx.CB_READONLY, name="Language")
        self.addCleanup(combo.Destroy)
        return combo

    def test_auto_comes_first_then_every_language_of_the_model(self):
        combo = self._combo()
        language_choice.fill(combo)
        self.assertEqual(combo.GetCount(), languages.COUNT + 1)
        self.assertEqual(combo.GetString(0), languages.AUTO_LABEL)
        self.assertEqual(combo.GetValue(), languages.AUTO_LABEL)
        listed = [combo.GetString(i) for i in range(1, combo.GetCount())]
        self.assertEqual(listed, [languages.label(code)
                                  for code in languages.language_ids()])

    def test_a_selection_survives_the_bulk_fill(self):
        combo = self._combo()
        language_choice.fill(combo, "hi")
        self.assertEqual(combo.GetValue(), "Hindi (hi)")
        self.assertEqual(language_choice.hint_of(combo), "hi")
        self.assertEqual(language_choice.selected_code(combo), "hi")
        # Selecting by code works for any language of the table.
        language_choice.select(combo, "zh")
        self.assertEqual(combo.GetValue(), languages.label("zh"))
        # ... and Auto is the auto key, which normalises to "no hint".
        language_choice.select(combo, "auto")
        self.assertEqual(combo.GetStringSelection(), languages.AUTO_LABEL)
        self.assertIsNone(language_choice.hint_of(combo))
        self.assertEqual(language_choice.selected_code(combo),
                         language_choice.AUTO_KEY)


class OmniVoiceRecordingLanguageTest(_AppMixin):
    """The Recording window's Language box for the OmniVoice engines."""

    def _dialog(self, pin=None):
        tmp = tempfile.mkdtemp(prefix="aivs_omni_lang_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Omni project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        if pin:
            data = project.load_project(tmp)
            data.setdefault("tts", {})["omni"] = {"language": pin}
            project.save_project(tmp, data)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        dlg = RecordingDialog(self.frame, tmp, settings, self._store())
        self.addCleanup(dlg.Destroy)
        self._installed()
        dlg._rebuild_voice_choices()
        for index in range(dlg.tts_combo.GetCount()):
            if dlg.tts_combo.GetClientData(index) == "omnivoice":
                dlg.tts_combo.SetSelection(index)
                dlg._on_tts(None)
                break
        return tmp, dlg

    def _start(self, dlg):
        captured = {}

        def fake_worker(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch("ai_voice_studio.gui.recording_dialog.SynthesisWorker",
                        side_effect=fake_worker), \
                mock.patch.object(dlg, "_show_progress_dialog"):
            dlg._cancel_event = threading.Event()
            dlg._on_start(None)
        return captured

    def test_the_language_box_offers_auto_and_every_language(self):
        _tmp, dlg = self._dialog()
        self.assertEqual(dlg._selected_tts_id(), "omnivoice")
        self.assertEqual(dlg.lang_combo.GetCount(), languages.COUNT + 1)
        self.assertEqual(dlg.lang_combo.GetString(0), languages.AUTO_LABEL)
        self.assertEqual(dlg.lang_combo.GetSelection(), 0)
        self.assertEqual(dlg.lang_combo.GetValue(), languages.AUTO_LABEL)

    def test_a_stored_pin_reappears_in_the_language_box(self):
        _tmp, dlg = self._dialog(pin="hi")
        self.assertEqual(dlg.lang_combo.GetValue(), "Hindi (hi)")
        self.assertEqual(dlg._omni_language_hint(), "hi")

    def test_pinning_a_language_stores_it_with_the_project(self):
        tmp, dlg = self._dialog()
        variants_before = dlg.variant_combo.GetCount()
        voices_before = dlg.voice_combo.GetCount()
        language_choice.select(dlg.lang_combo, "hi")
        dlg._on_lang(None)
        self.assertEqual(dlg.data["tts"]["omni"]["language"], "hi")
        # The hint never removes voices: an OmniVoice voice speaks every
        # language of the model.
        self.assertEqual(dlg.variant_combo.GetCount(), variants_before)
        self.assertEqual(dlg.voice_combo.GetCount(), voices_before)
        dlg._save_tts_to_project()
        stored = project.load_project(tmp)["tts"]
        self.assertEqual(stored["omni"]["language"], "hi")
        # The voice's own catalog language stays "auto"; only the hint moves.
        self.assertEqual(stored["language"], "auto")
        # Auto clears the pin again.
        language_choice.select(dlg.lang_combo, "auto")
        dlg._on_lang(None)
        self.assertNotIn("language", dlg.data["tts"]["omni"])
        self.assertFalse(dlg.data["tts"]["omni"])

    def test_the_pin_travels_to_the_synthesis_worker(self):
        _tmp, dlg = self._dialog()
        language_choice.select(dlg.lang_combo, "hi")
        dlg._on_lang(None)
        captured = self._start(dlg)
        self.assertEqual(captured["voice_entry"].get("language"), "hi")
        # An explicit "auto" sends no hint at all.
        _tmp2, dlg2 = self._dialog()
        captured = self._start(dlg2)
        self.assertIsNone(captured["voice_entry"].get("language"))

    def test_a_library_voice_carries_its_own_pin(self):
        tmp = tempfile.mkdtemp(prefix="aivs_omni_lib_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Library project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        store = self._store()
        voice_store.create_voice(store, name="Hindi narrator", mode="design",
                                instruct="female, calm", language="hi")
        dlg = RecordingDialog(self.frame, tmp,
                              Settings(path=os.path.join(tmp, "settings.json")),
                              store)
        self.addCleanup(dlg.Destroy)
        self._installed()
        dlg._rebuild_voice_choices()
        for index in range(dlg.tts_combo.GetCount()):
            if dlg.tts_combo.GetClientData(index) == "omnivoice":
                dlg.tts_combo.SetSelection(index)
                dlg._on_tts(None)
                break
        # Pick the library voice: its own pin shows up in the Language box.
        picked = None
        for index in range(dlg.voice_combo.GetCount()):
            entry = dlg.voice_combo.GetClientData(index)
            if entry and entry.get("custom_omni"):
                picked = (index, entry)
                break
        self.assertIsNotNone(picked, "the library voice is missing")
        dlg.voice_combo.SetSelection(picked[0])
        dlg._update_omni_ui()
        self.assertEqual(dlg.lang_combo.GetValue(), "Hindi (hi)")
        self.assertIn("Hindi (hi)", dlg.omni_summary.GetLabel())
        # An explicit project choice overrides the voice's own pin.
        language_choice.select(dlg.lang_combo, "ja")
        dlg._on_lang(None)
        captured = self._start(dlg)
        self.assertEqual(captured["voice_entry"]["omni"]["language"], "ja")


class AvailableTtsLanguageTest(_AppMixin):
    """Settings > Available TTS offers the same list for OmniVoice."""

    def _panel(self):
        store = self._store()
        voice_store.create_voice(store, name="Narrator", mode="design",
                                instruct="female", language="hi")
        settings = Settings(path=os.path.join(
            tempfile.mkdtemp(prefix="aivs_avail_"), "settings.json"))
        panel = AvailablePanel(self.frame, store, settings)
        self.addCleanup(panel.Destroy)
        self._installed()
        panel.refresh()
        for index in range(panel.tts_combo.GetCount()):
            if panel.tts_combo.GetClientData(index) == "omnivoice":
                panel.tts_combo.SetSelection(index)
                panel._on_tts(None)
                break
        return panel

    def test_the_language_box_offers_auto_and_every_language(self):
        panel = self._panel()
        self.assertEqual(panel.lang_combo.GetCount(), languages.COUNT + 1)
        self.assertEqual(panel.lang_combo.GetString(0), languages.AUTO_LABEL)

    def test_pinning_keeps_the_engine_voices_selectable(self):
        panel = self._panel()
        self.assertGreater(panel.variant_combo.GetCount(), 0)
        self.assertGreater(panel.voice_combo.GetCount(), 0)
        language_choice.select(panel.lang_combo, "hi")
        panel._on_lang(None)
        self.assertEqual(panel._omni_language, "hi")
        self.assertGreater(panel.variant_combo.GetCount(), 0)
        self.assertGreater(panel.voice_combo.GetCount(), 0)

    def test_the_preview_speaks_the_pinned_language(self):
        panel = self._panel()
        language_choice.select(panel.lang_combo, "hi")
        panel._on_lang(None)
        with mock.patch("ai_voice_studio.gui.model_panels.threading.Thread") as fake:
            panel._on_preview(None)
        voice = fake.call_args.kwargs.get("args", (None, None))[0]
        self.assertEqual(voice.get("language"), "hi")
        # The panel's own entry is untouched (the pin is applied to a copy).
        self.assertEqual(panel.selected_voice().get("language"), "auto")


class VoiceLibraryLanguageTest(_AppMixin):
    """Settings > OmniVoice engines pins the language into a created voice."""

    def test_create_voice_remembers_the_pin(self):
        store = self._store()
        entry = voice_store.create_voice(
            store, name="Hindi narrator", mode="design",
            instruct="female, calm", language="Hindi",
        )
        self.assertEqual(entry["omni_language"], "hi")
        consumer = voice_store.entry_for_engine(entry, "omnivoice", "triton")
        self.assertEqual(consumer["omni"]["language"], "hi")
        # The catalog language stays "auto" so the voice cascade is not
        # filtered by it.
        self.assertEqual(consumer["language"], "auto")
        # Auto (the default) stores no pin at all.
        plain = voice_store.create_voice(store, name="Plain", mode="design",
                                         instruct="male")
        self.assertNotIn("omni_language", plain)
        self.assertIsNone(
            voice_store.entry_for_engine(plain, "omnivoice", "triton")["omni"]["language"]
        )

    def test_the_panel_pins_what_the_form_says(self):
        store = self._store()
        settings = Settings(path=os.path.join(
            tempfile.mkdtemp(prefix="aivs_lib_"), "settings.json"))
        panel = _OmniVoiceEnginesPanel(self.frame, settings, store)
        self.addCleanup(panel.Destroy)
        self.assertEqual(panel.omni_language_combo.GetCount(),
                         languages.COUNT + 1)
        self.assertEqual(panel.omni_language_combo.GetString(0),
                         languages.AUTO_LABEL)
        language_choice.select(panel.omni_language_combo, "fr")
        panel.design_name_ctrl.SetValue("French narrator")
        panel.design_desc_ctrl.SetValue("female, middle-aged, french accent")
        with mock.patch("ai_voice_studio.gui.settings_dialog.wx.MessageBox"):
            panel._on_create("design")
        entry = voice_store.find_voice(store, "French narrator")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["omni_language"], "fr")
        self.assertTrue(any("French narrator" in name
                            and languages.label("fr") in name
                            for name in panel._lib_names))


class OmniEngineIdsTest(unittest.TestCase):
    def test_the_engine_ids_match_the_catalog(self):
        from ai_voice_studio.tts import catalog

        self.assertEqual(set(spec.ENGINE_IDS),
                         {"omnivoice", "omnivoice_server"})
        self.assertTrue(spec.is_engine("omnivoice_server"))
        self.assertFalse(spec.is_engine("piper"))
        self.assertTrue(language_choice.is_omnivoice("omnivoice"))
        self.assertFalse(language_choice.is_omnivoice(None))
        for entry in catalog.get_tts_list():
            if entry["id"] in spec.ENGINE_IDS:
                self.assertEqual(entry["engine"], entry["id"])


if __name__ == "__main__":
    unittest.main()
