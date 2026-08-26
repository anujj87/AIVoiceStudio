"""GUI construction smoke tests (no event loop, no window shown).

These guard against regressions where dialog/wizard page construction code
ended up as dead code (e.g. an early ``return`` before the UI was built),
which made pages render blank.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import wx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.constants import AUDIO_MODE_CHOICES, MODE_PAGE_WITH_H1  # noqa: E402
from ai_voice_studio.gui.main_frame import MainFrame  # noqa: E402
from ai_voice_studio.gui.new_project_wizard import NewProjectWizard  # noqa: E402
from ai_voice_studio.gui.recording_dialog import RecordingDialog  # noqa: E402
from ai_voice_studio.gui.settings_dialog import SettingsDialog  # noqa: E402
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

    def _empty_store(self) -> ModelStore:
        """A store backed by a throwaway state file: no voices, always."""
        fd, path = tempfile.mkstemp(prefix="aivs_models_", suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return ModelStore(state_file=path)


class NewProjectWizardTest(_AppMixin):
    def test_initial_name_prefills_name_field(self):
        wizard = NewProjectWizard(self.frame, Settings(), self._empty_store(),
                                  initial_name="My Book")
        try:
            self.assertEqual(wizard.page_details.name_ctrl.GetValue(), "My Book")
        finally:
            wizard.Destroy()

    def test_pages_build_their_controls(self):
        wizard = NewProjectWizard(self.frame, Settings(), self._empty_store())
        try:
            details, mode = wizard.page_details, wizard.page_mode

            # Page 1: project name + open document + punctuation.
            self.assertTrue(hasattr(details, "name_ctrl"), "name_ctrl missing")
            self.assertTrue(hasattr(details, "open_btn"), "open_btn missing")
            self.assertTrue(hasattr(details, "file_label"), "file_label missing")
            self.assertTrue(hasattr(details, "punct_combo"), "punct_combo missing")
            self.assertEqual(details.punct_combo.GetName(), "Punctuation")
            self.assertGreater(details.punct_combo.GetCount(), 0)
            self.assertIsNotNone(details.GetSizer())
            self.assertGreater(details.GetSizer().GetItemCount(), 0)

            # Page 2: one radio per audio mode.
            self.assertTrue(mode.radios, "no mode radios built")
            self.assertEqual(len(mode.radios), len(AUDIO_MODE_CHOICES))
            self.assertIsNotNone(mode.GetSizer())
            self.assertGreater(mode.GetSizer().GetItemCount(), 0)
            self.assertIn(mode.selected(), dict(AUDIO_MODE_CHOICES))
            self.assertTrue(all(r[0].GetName() for r in mode.radios))

            # Navigation wiring.
            self.assertIs(details.GetNext(), mode)
            self.assertIs(mode.GetPrev(), details)
            self.assertIsNone(details.GetPrev())
            self.assertIsNone(mode.GetNext())
        finally:
            wizard.Destroy()

    def test_finish_without_document_shows_message(self):
        # No crash path: _on_finish requires a source document.
        wizard = NewProjectWizard(self.frame, Settings(), self._empty_store())
        try:
            self.assertEqual(wizard.source_path, "")
            self.assertIsNotNone(wizard.page_details.name_ctrl)
            self.assertIsNotNone(wizard.page_mode.selected())
        finally:
            wizard.Destroy()


class SettingsDialogTest(_AppMixin):
    def test_categories_build_all_controls(self):
        settings = Settings()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        try:
            # NVDA-style: a category list on the left, one panel per category.
            self.assertIsNotNone(dlg.cat_list)
            self.assertEqual(dlg.cat_list.GetItemCount(), 12)
            self.assertEqual(len(dlg._panels), 12)
            self.assertTrue(hasattr(dlg, "container"))
            # Every panel must have at least one child control.
            for panel in dlg._panels:
                self.assertTrue(panel.GetChildren(), f"{panel.title} has no children")
            # Only the first category is visible.
            self.assertTrue(dlg._panels[0].IsShown())
            self.assertFalse(dlg._panels[1].IsShown())
            # Category 3 is OmniVoice Engines (was Voice Clone).
            self.assertEqual(dlg._panels[3].title, "OmniVoice engines")
            # Recording-settings category exposes the preview controls
            # (punctuation moved out to its own category).
            recording = dlg.recording_panel
            self.assertTrue(hasattr(recording, "preview_btn"))
            self.assertTrue(hasattr(recording, "sample_text"))
            self.assertTrue(hasattr(recording, "preview_status"))
            self.assertFalse(hasattr(recording, "punct_combo"))
            # Punctuation category exposes the mode combo, the voice cascade
            # (TTS/variant/voice) and a preview button.
            punct = dlg.punctuation_panel
            self.assertEqual(punct.punct_combo.GetName(), "Punctuation mode")
            self.assertEqual(punct.punct_combo.GetCount(), 4)
            self.assertTrue(hasattr(punct, "example_text"))
            self.assertTrue(hasattr(punct, "example_out"))
            self.assertTrue(punct.example_out.GetLabel())
            self.assertIn(punct.tts_combo.GetName(), ("TTS engine", "TTS engine for preview"))
            self.assertIn(punct.variant_combo.GetName(), ("Variant", "Variant for preview"))
            self.assertIn(punct.voice_combo.GetName(), ("Voice", "Voice for preview"))
            self.assertTrue(hasattr(punct, "preview_btn"))
            self.assertTrue(hasattr(punct, "preview_status"))
            # Punctuation panel has a preview button (may be enabled if
            # pip-installed OmniVoice voices are available).
            self.assertTrue(hasattr(punct, "preview_btn"))
            self.assertTrue(hasattr(punct, "preview_status"))
            # Available TTS category exposes the preview button.
            self.assertTrue(hasattr(dlg.available_panel, "preview_btn"))
            # General category exposes the storage-location pickers.
            general = dlg.general_panel
            self.assertTrue(hasattr(general, "recordings_ctrl"))
            self.assertTrue(hasattr(general, "models_ctrl"))
            self.assertTrue(general.recordings_ctrl.GetValue())
            self.assertTrue(general.models_ctrl.GetValue())
            self.assertEqual(general.recordings_ctrl.GetName(), "Recorded files location")
            self.assertEqual(general.models_ctrl.GetName(), "Model files location")
            # Compute category exposes the GPU runtime controls.
            compute = dlg.compute_panel
            self.assertTrue(hasattr(compute, "gpu_download_btn"))
            self.assertTrue(hasattr(compute, "gpu_remove_btn"))
            self.assertTrue(hasattr(compute, "gpu_status"))
            self.assertTrue(hasattr(compute, "gauge"))
            self.assertEqual(compute.gpu_download_btn.GetName(), "Download GPU runtime")
            self.assertEqual(compute.gpu_remove_btn.GetName(), "Remove GPU runtime")
            # Switching category shows the right panel (NVDA pattern).
            dlg._show_category(3)
            self.assertFalse(dlg._panels[0].IsShown())
            self.assertTrue(dlg._panels[3].IsShown())
        finally:
            dlg.Destroy()


    def test_general_panel_applies_path_settings(self):
        settings = Settings()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        try:
            general = dlg.general_panel
            with tempfile.TemporaryDirectory() as rec, tempfile.TemporaryDirectory() as mod:
                general.recordings_ctrl.SetValue(rec)
                general.models_ctrl.SetValue(mod)
                general.apply_to_settings()
                self.assertEqual(settings.get("paths.recordings_dir"), rec)
                self.assertEqual(settings.get("paths.models_dir"), mod)
        finally:
            dlg.Destroy()

    def test_enter_on_button_keeps_dialog_open(self):
        """Enter on a focused button must activate the button, not close the
        Settings dialog (regression: the NVDA-style hook hijacked Enter on
        every control, so pressing the Download button closed Settings)."""
        settings = Settings()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        try:
            fired = []
            dlg.Bind(wx.EVT_BUTTON, lambda evt: fired.append(True), id=wx.ID_OK)
            # Enter on a button -> button keeps the key (no OK click).
            dlg.compute_panel.gpu_download_btn.SetFocus()
            evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
            evt.SetKeyCode(wx.WXK_RETURN)
            dlg._on_char_hook(evt)
            self.assertEqual(fired, [], "Enter on a button must not trigger OK")
            # Enter on a plain text field still means OK (NVDA pattern).
            dlg.general_panel.recordings_ctrl.SetFocus()
            evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
            evt.SetKeyCode(wx.WXK_RETURN)
            dlg._on_char_hook(evt)
            self.assertEqual(fired, [True], "Enter in a text field should confirm")
        finally:
            dlg.Destroy()


class MainFrameTest(_AppMixin):
    def test_frame_builds_all_ui(self):
        frame = MainFrame(settings=Settings(), store=self._empty_store())
        try:
            self.assertIsNotNone(frame.GetMenuBar())
            self.assertTrue(hasattr(frame, "recent_list"))
            self.assertTrue(hasattr(frame, "remove_btn"), "Remove Project button missing")
            self.assertTrue(frame.recent_list.GetParent().GetChildren())
            self.assertEqual(frame.GetStatusBar().GetStatusText(), "Ready.")
        finally:
            frame.Destroy()

    def test_edit_menu_has_project_actions(self):
        frame = MainFrame(settings=Settings(), store=self._empty_store())
        try:
            menubar = frame.GetMenuBar()
            edit_index = menubar.FindMenu("Edit")
            self.assertGreaterEqual(edit_index, 0, "Edit menu missing")
            labels = [item.GetItemLabelText()
                      for item in menubar.GetMenu(edit_index).GetMenuItems()]
            for expected in ("Resume Recording", "Restart Project...",
                             "Restart All Recording", "Restart Selected Recording...",
                             "Remove Project..."):
                self.assertIn(expected, labels)
            # Help menu has Read Me + User Guide + About.
            help_index = menubar.FindMenu("Help")
            self.assertGreaterEqual(help_index, 0, "Help menu missing")
            help_labels = [item.GetItemLabelText()
                           for item in menubar.GetMenu(help_index).GetMenuItems()]
            for expected in ("Read Me", "User Guide", "About AI Voice Studio"):
                self.assertIn(expected, help_labels)
            # Shortcuts live in the menu labels (single registration; no
            # duplicate frame-level accelerator table that could double-fire
            # with wxMSW's menu accelerators).
            new_item = [i for i in menubar.GetMenu(0).GetMenuItems()
                        if i.GetItemLabelText() == "New Project"][0]
            self.assertIn("Ctrl+Shift+N", new_item.GetItemLabel())
        finally:
            frame.Destroy()

    def test_recording_picker_dialog_builds(self):
        from ai_voice_studio.gui.main_frame import _RecordingPickerDialog

        dlg = _RecordingPickerDialog(self.frame, ["Chapter 1.wav", "Chapter 2.wav"])
        try:
            self.assertEqual(dlg.combo.GetCount(), 2)
            self.assertEqual(dlg.combo.GetName(), "Recorded files")
            self.assertEqual(dlg.combo.GetSelection(), 0)
        finally:
            dlg.Destroy()


class RecordingDialogTest(_AppMixin):
    def test_dialog_builds_controls_from_a_project(self):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        try:
            project.create_project(
                tmp, "Smoke project", "doc.txt", MODE_PAGE_WITH_H1,
                [{"index": 1, "title": "One", "text": "Hello."}],
            )
            dlg = RecordingDialog(self.frame, tmp, Settings(), self._empty_store())
            try:
                for attr in ("start_btn", "pause_btn", "resume_btn", "stop_btn",
                             "gauge", "status", "compute_combo", "tts_combo",
                             "lang_combo", "variant_combo", "voice_combo",
                             "format_combo", "text_preview", "rate", "pitch", "volume"):
                    self.assertTrue(hasattr(dlg, attr), f"RecordingDialog missing {attr}")
                self.assertIsNotNone(dlg.GetSizer())
                self.assertGreater(dlg.GetSizer().GetItemCount(), 0)
                # Punctuation moved out of the Recording window.
                self.assertFalse(hasattr(dlg, "punct_combo"))
                # The read-only text preview is first and shows the document.
                self.assertEqual(dlg.text_preview.GetName(), "Text to record")
                self.assertIn("Hello.", dlg.text_preview.GetValue())
                # No voices installed -> start disabled (progress label is set
                # afterwards by _update_progress, so only the button state and
                # a sensible status label are checked).
                self.assertFalse(dlg.start_btn.IsEnabled())
                self.assertTrue(dlg.status.GetLabel())
                # Volume must default to 100%.
                self.assertEqual(dlg.volume.GetValue(), 100)
                self.assertEqual(dlg.rate.GetValue(), 100)
                self.assertEqual(dlg.pitch.GetValue(), 100)
                # Reading params must not crash even when the project has no
                # stored compute back-end (combo left unselected).
                params = dlg._current_params()
                # Default compute should be a valid compute mode (auto resolves)
                self.assertIn(params["compute"], ("cpu", "cuda", "dml", "auto"))
                self.assertEqual(params["punctuation"], "default")
            finally:
                dlg.Destroy()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
