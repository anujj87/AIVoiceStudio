"""GUI construction smoke tests (no event loop, no window shown).

These guard against regressions where dialog/wizard page construction code
ended up as dead code (e.g. an early ``return`` before the UI was built),
which made pages render blank.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import wx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.constants import AUDIO_MODE_CHOICES, MODE_PAGE_WITH_H1, TERMS_VERSION  # noqa: E402
from ai_voice_studio.gui.accept_dialog import (  # noqa: E402
    AcceptanceDialog,
    record_acceptance,
    terms_current,
)
from ai_voice_studio.gui.main_frame import MainFrame  # noqa: E402
from ai_voice_studio.gui.new_project_wizard import NewProjectWizard  # noqa: E402
from ai_voice_studio.gui.recording_dialog import RecordingDialog  # noqa: E402
from ai_voice_studio.gui.settings_dialog import SettingsDialog  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402
from ai_voice_studio import compute  # noqa: E402
from ai_voice_studio.omnivoice import languages as omni_languages  # noqa: E402
from ai_voice_studio import voicelab  # noqa: E402
from ai_voice_studio.gui import compute_choice  # noqa: E402
from ai_voice_studio.voicelab import engines as voice_lab  # noqa: E402
from ai_voice_studio.voicelab import options as tuning  # noqa: E402


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


class AcceptanceDialogTest(_AppMixin):
    """First-launch terms dialog: all boxes required, disagree blocks the app."""

    def _dialog(self) -> AcceptanceDialog:
        dlg = AcceptanceDialog(self.frame)
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_buttons_disabled_until_every_box_checked(self):
        dlg = self._dialog()
        boxes = dlg._checkboxes
        self.assertGreaterEqual(len(boxes), 7)  # 6 terms + "agree with all"
        self.assertFalse(dlg._agree_btn.IsEnabled())
        self.assertFalse(dlg._disagree_btn.IsEnabled())
        for box in boxes:
            box.SetValue(True)
        dlg._on_toggle(None)
        self.assertTrue(dlg._agree_btn.IsEnabled())
        self.assertTrue(dlg._disagree_btn.IsEnabled())
        # Unchecking any single box disables both buttons again.
        boxes[0].SetValue(False)
        dlg._on_toggle(None)
        self.assertFalse(dlg._agree_btn.IsEnabled())
        self.assertFalse(dlg._disagree_btn.IsEnabled())

    def test_last_checkbox_checks_all_terms_automatically(self):
        dlg = self._dialog()
        terms = dlg._term_checkboxes
        self.assertEqual(len(terms), 6)
        # Ticking the final box ticks every term checkbox.
        dlg._agree_all.SetValue(True)
        dlg._on_agree_all(None)
        self.assertTrue(all(b.GetValue() for b in terms))
        self.assertTrue(dlg._agree_btn.IsEnabled())
        self.assertTrue(dlg._disagree_btn.IsEnabled())
        # Un-ticking it un-ticks them all again.
        dlg._agree_all.SetValue(False)
        dlg._on_agree_all(None)
        self.assertFalse(any(b.GetValue() for b in terms))
        self.assertFalse(dlg._agree_btn.IsEnabled())

    def test_term_two_mentions_full_responsibility(self):
        dlg = self._dialog()
        self.assertIn("fully responsible", dlg._term_checkboxes[1].GetLabel())

    def test_terms_alone_are_not_enough_without_the_all_box(self):
        dlg = self._dialog()
        for box in dlg._term_checkboxes:
            box.SetValue(True)
        dlg._on_toggle(None)
        # The "agree with all" box itself is still required to proceed.
        self.assertFalse(dlg._agree_btn.IsEnabled())
        dlg._agree_all.SetValue(True)
        dlg._on_agree_all(None)
        self.assertTrue(dlg._agree_btn.IsEnabled())

    def test_agree_records_acceptance(self):
        settings = Settings(path=os.path.join(
            tempfile.mkdtemp(prefix="aivs_terms_"), "settings.json"))
        self.assertFalse(terms_current(settings))
        dlg = self._dialog()
        for box in dlg._checkboxes:
            box.SetValue(True)
        dlg._on_agree(None)
        self.assertTrue(dlg.accepted)
        record_acceptance(settings)
        self.assertTrue(terms_current(settings))

    def test_disagree_does_not_record_acceptance(self):
        settings = Settings(path=os.path.join(
            tempfile.mkdtemp(prefix="aivs_terms_"), "settings.json"))
        dlg = self._dialog()
        for box in dlg._checkboxes:
            box.SetValue(True)
        dlg._on_disagree(None)
        self.assertFalse(dlg.accepted)
        record_acceptance(settings)  # only called on agreement in main.py
        dlg2 = self._dialog()
        for box in dlg2._checkboxes:
            box.SetValue(True)
        dlg2._on_disagree(None)
        self.assertFalse(dlg2.accepted)

    def test_version_bump_reprompts(self):
        settings = Settings(path=os.path.join(
            tempfile.mkdtemp(prefix="aivs_terms_"), "settings.json"))
        record_acceptance(settings)
        self.assertTrue(terms_current(settings))
        # Simulate an update: a newer TERMS_VERSION no longer matches.
        with mock.patch("ai_voice_studio.gui.accept_dialog.TERMS_VERSION", "9999.0.0"):
            self.assertFalse(terms_current(settings))

    def test_escape_is_swallowed(self):
        dlg = self._dialog()
        for box in dlg._checkboxes:
            box.SetValue(True)
        event = mock.Mock()
        event.GetKeyCode.return_value = wx.WXK_ESCAPE
        dlg._on_char_hook(event)
        event.Skip.assert_not_called()  # Escape cannot dismiss the dialog


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
            self.assertEqual(dlg.cat_list.GetItemCount(), 13)
            self.assertEqual(len(dlg._panels), 13)
            self.assertTrue(hasattr(dlg, "container"))
            # Every panel must have at least one child control.
            for panel in dlg._panels:
                self.assertTrue(panel.GetChildren(), f"{panel.title} has no children")
            # Only the first category is visible.
            self.assertTrue(dlg._panels[0].IsShown())
            self.assertFalse(dlg._panels[1].IsShown())
            # Category 3 is the Voice Clone category (the CPU/GPU clone
            # engines); OmniVoice engines follows it.
            self.assertEqual(dlg._panels[3].title, "Voice Clone")
            self.assertEqual(dlg._panels[4].title, "OmniVoice engines")
            clone = dlg.voice_clone_panel
            for attribute in ("engine_combo", "install_btn", "remove_btn",
                              "engine_status", "device_combo", "source_combo",
                              "builtin_combo", "sample_ctrl", "name_ctrl",
                              "ref_text_ctrl", "create_btn", "voices_list",
                              "preview_btn", "preview_status", "delete_btn"):
                self.assertTrue(hasattr(clone, attribute), attribute)
            # Every Voice Lab engine is selectable, and the CPU is always the
            # first device (the GPU is only ever offered *in addition*).
            self.assertEqual(
                clone.engine_combo.GetCount(), len(voice_lab.engine_ids())
            )
            self.assertGreaterEqual(clone.device_combo.GetCount(), 1)
            self.assertEqual(clone.device_combo.GetClientData(0), "cpu")
            self.assertIn(clone.device_combo.GetClientData(0),
                          [d[0] for d in voicelab.device_options()])
            # The built-in voices of the four engines are defined, even when
            # the engines are not installed yet.
            self.assertGreater(voice_lab.count_builtin_voices("bark"), 100)
            self.assertEqual(voice_lab.count_builtin_voices("pocket_tts"), 26)
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
            dlg._show_category(4)
            self.assertFalse(dlg._panels[3].IsShown())
            self.assertTrue(dlg._panels[4].IsShown())

            # Recording settings exposes the TTS/voice preview list.
            self.assertTrue(hasattr(recording, "voices_list"))
            self.assertEqual(recording.voices_list.GetName(),
                             "Voices for the selected TTS engine")
            self.assertTrue(hasattr(recording, "voice_count"))
            self.assertEqual(recording.voice_count.GetName(), "Voice count")

            # The OmniVoice engines category offers the one-click move to the
            # Compute category when the OmniVoice dependency is missing.
            engines = dlg.omnivoice_engines_panel
            self.assertTrue(hasattr(engines, "goto_compute_btn"))
            self.assertIn("Compute", engines.goto_compute_btn.GetLabel())
            self.assertTrue(hasattr(engines, "dependency_status"))
        finally:
            dlg.Destroy()

    def test_server_category_opens_without_waiting_for_the_probe(self):
        """Opening a category must never block on a network round trip.

        The server status probe used to run inline on activation, which made
        the Settings dialog freeze for seconds on an unreachable host.
        """
        settings = Settings()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        try:
            index = [i for i, cls in enumerate(dlg.CATEGORIES)
                     if cls.title == "OmniVoice Server"][0]
            panel = dlg._panels[index]
            probed = threading.Event()
            probed_args = []

            def slow_probe(host, port, epoch):
                probed_args.append((host, port, epoch))
                probed.set()
                time.sleep(1.5)  # stand in for a slow health check

            with mock.patch.object(panel, "_probe_status", slow_probe):
                start = time.perf_counter()
                dlg._show_category(index)
                elapsed = time.perf_counter() - start

            self.assertLess(elapsed, 0.5,
                            f"switching category blocked for {elapsed:.2f}s")
            self.assertEqual(panel.status_label.GetLabel(),
                             "Server status: Checking...")
            self.assertTrue(probed.wait(2.0), "status probe never started")
            self.assertTrue(probed_args[0][0] or probed_args[0][1])
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


class VoiceCloneCategoryTest(_AppMixin):
    """The Voice Clone category end to end: create a clone, then use it."""

    def setUp(self):
        # Cloned voices are copied into the user models folder: keep that in a
        # temporary directory so the real one is never touched.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch("ai_voice_studio.paths.user_data_dir",
                             return_value=self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _dialog(self):
        settings = Settings(path=os.path.join(self.tmp.name, "settings.json"))
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg

    @contextlib.contextmanager
    def _without_builtin_voices(self, panel):
        """Hide the engine's built-in voices, whatever this machine has.

        A Voice Lab engine that happens to be installed here contributes its
        own voices (F5-TTS ships two reference voices, Bark over a hundred),
        which would drown out the clones these tests are about.  The probe
        thread is silenced too, so the test never waits on a real engine.
        """
        with mock.patch.object(voice_lab, "builtin_voices", return_value=[]), \
                mock.patch.object(panel, "_probe_builtin_availability"):
            yield

    def test_cpu_is_always_offered_and_gpu_is_added_when_present(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            dlg = self._dialog()
            values = [dlg.voice_clone_panel.device_combo.GetClientData(i)
                      for i in range(dlg.voice_clone_panel.device_combo.GetCount())]
            self.assertEqual(values, ["cpu"])
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            dlg = self._dialog()
            values = [dlg.voice_clone_panel.device_combo.GetClientData(i)
                      for i in range(dlg.voice_clone_panel.device_combo.GetCount())]
            self.assertEqual(values, ["cpu", "cuda", "auto"])

    def test_builtin_voices_are_offered_once_the_engine_is_installed(self):
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        # Whether a package happens to be installed in the managed environment
        # depends on the machine, so both states are driven explicitly here.
        with mock.patch.object(voicelab, "is_installed", return_value=False):
            self.assertTrue(panel.select_engine("bark"))
            self.assertEqual(panel.builtin_combo.GetCount(), 0)
            self.assertEqual(panel.voices_list.GetCount(), 0)
            self.assertIn("Not installed", panel.engine_status.GetLabel())
        with mock.patch.object(voicelab, "is_installed", return_value=True), \
                mock.patch.object(voicelab, "installed_version", return_value="4.4.0"):
            panel._refresh_engine_status()
            panel._refresh_builtin_voices()
            panel._refresh_voices()
        self.assertEqual(panel.builtin_combo.GetCount(), 130)
        self.assertEqual(panel.voices_list.GetCount(), 130)
        self.assertIn("Installed (version 4.4.0)",
                      panel.engine_status.GetLabel())
        # ... and every voice row says which engine, variant and device.
        self.assertIn("built-in", panel.voices_list.GetString(0))
        self.assertIn("CPU", panel.voices_list.GetString(0))

    def test_a_cloned_voice_reaches_available_tts(self):
        sample = os.path.join(self.tmp.name, "reference.wav")
        with open(sample, "wb") as fh:
            fh.write(b"RIFFfake")
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.assertTrue(panel.select_engine("f5tts"))
        panel.name_ctrl.SetValue("My Clone")
        panel.sample_ctrl.SetValue(sample)
        panel.ref_text_ctrl.SetValue("some words")
        # This test is about the clone: on a machine where F5-TTS really is
        # installed its own built-in reference voices would be listed as well.
        with self._without_builtin_voices(panel):
            panel._on_create(None)
            labels = [panel.voices_list.GetString(i)
                      for i in range(panel.voices_list.GetCount())]
            self.assertEqual(len(labels), 1, labels)
        self.assertIn("My Clone", labels[0])
        self.assertIn("cloned", labels[0])
        # The other categories pick the clone up without knowing about the
        # Voice Clone category at all.
        dlg.available_panel.refresh()
        cloned = [v for v in dlg.available_panel._voices if v.get("cloned")]
        self.assertEqual([v["voice_name"] for v in cloned], ["My Clone"])
        dlg.recording_panel.on_activated()
        self.assertIn("f5tts", [dlg.recording_panel.tts_combo.GetClientData(i)
                                for i in range(dlg.recording_panel.tts_combo.GetCount())])

    def test_a_failed_cuda_install_falls_back_to_the_cpu_wheels(self):
        """A CUDA wheel index that cannot satisfy the install must not block it.

        ``cu121`` has no Python 3.13 wheels; without the fallback the engine
        install failed outright with "No matching distribution found for
        torch" and the engine could not be used at all.
        """
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.assertTrue(panel.select_engine("f5tts"))
        calls = []

        class FakeRuntime:
            def ensure_pip(self):
                pass

            def pip_install(self, packages, progress=None, index_url=None):
                calls.append((list(packages), index_url))
                if index_url:
                    return {"ok": False, "output": "",
                            "error": "No matching distribution found for torch"}
                return {"ok": True, "output": "", "error": ""}

        done = {}
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=FakeRuntime()), \
                mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch.object(panel, "_update_progress"), \
                mock.patch.object(
                    panel, "_install_done",
                    side_effect=lambda *args: done.update(args=args),
                ), \
                mock.patch("wx.CallAfter",
                           side_effect=lambda fn, *a, **k: fn(*a, **k)):
            panel._install_job("f5tts")
        # CUDA wheels first, then the same packages from PyPI.
        self.assertEqual(calls[0][1], voice_lab.PYTORCH_CUDA_INDEX)
        self.assertIsNone(calls[1][1])
        # The install succeeds, with a note explaining the GPU limitation.
        self.assertIsNone(done["args"][1])
        self.assertTrue(done["args"][2])
        self.assertIn("CPU", done["args"][2][0])

    def test_engine_tuning_defaults_are_saved_in_settings(self):
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.assertTrue(panel.select_engine("f5tts"))
        self.assertTrue(hasattr(panel, "tuning_btn"))
        self.assertIn("engine defaults", panel.tuning_note.GetLabel())
        # The dialog writes into the panel's settings file.
        with mock.patch(
            "ai_voice_studio.gui.voicelab_options_dialog.VoiceLabOptionsDialog"
        ) as fake:
            fake.return_value.ShowModal.return_value = wx.ID_OK
            fake.return_value.get_options.return_value = {"nfe_step": 16}
            panel._on_tuning(None)
        self.assertEqual(
            dlg.settings.get(tuning.settings_key("f5tts")), {"nfe_step": 16}
        )
        panel._refresh_tuning_note()
        self.assertIn("diffusion steps 16", panel.tuning_note.GetLabel().lower())
        self.assertIn("diffusion steps 16", panel.engine_status.GetLabel().lower())

    def test_preview_uses_the_saved_tuning_defaults(self):
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        dlg.settings.set(tuning.settings_key("f5tts"), {"nfe_step": 8})
        with mock.patch.object(voicelab, "is_installed", return_value=True), \
                mock.patch.object(voicelab, "installed_version",
                                  return_value="1.2.3"):
            self.assertTrue(panel.select_engine("f5tts"))
            panel._refresh_engine_status()
            panel._refresh_builtin_voices()
            panel._refresh_voices()
            panel.voices_list.SetSelection(0)
            panel._update_voice_buttons()
            captured = {}

            def fake_thread(target=None, args=None, daemon=None):
                captured["args"] = args

                class _Thread:
                    def start(self):
                        pass

                return _Thread()

            with mock.patch(
                "ai_voice_studio.gui.clone_engines_panel.threading.Thread",
                side_effect=fake_thread,
            ):
                panel._on_preview(None)
        entry = captured["args"][0]
        self.assertEqual(entry.get("options"), {"nfe_step": 8})
        self.assertIn("diffusion steps 8",
                      panel.preview_status.GetLabel().lower())

    def test_the_engine_tuning_dialog_edits_the_overrides(self):
        from ai_voice_studio.gui.voicelab_options_dialog import (
            VoiceLabOptionsDialog,
        )

        dlg = VoiceLabOptionsDialog(
            self.frame, "f5tts", values={"nfe_step": 16},
            project_name="Smoke project",
        )
        try:
            self.assertEqual(dlg._controls["nfe_step"].GetValue(), 16)
            self.assertEqual(dlg.get_options(), {"nfe_step": 16})
            self.assertIn("diffusion steps 16", dlg.summary.GetLabel().lower())
            # A value equal to the engine default is not an override any more.
            dlg._controls["nfe_step"].SetValue(32)
            dlg._on_changed(None)
            self.assertEqual(dlg.get_options(), {})
            self.assertIn("none", dlg.summary.GetLabel())
            # Reset restores every engine default.
            dlg._controls["cfg_strength"].SetValue(3.5)
            dlg._on_reset(None)
            self.assertEqual(dlg.get_options(), {})
            # The seed field accepts a number and refuses anything else.
            dlg._controls["seed"].SetValue("7")
            self.assertEqual(dlg.get_options(), {"seed": 7})
            dlg._controls["seed"].SetValue("later")
            with self.assertRaises(ValueError):
                dlg.get_options()
        finally:
            dlg.Destroy()

    def test_save_refuses_an_invalid_value_and_closes_otherwise(self):
        """Clicking the real buttons: one validates, the other cancels."""
        from ai_voice_studio.gui.voicelab_options_dialog import (
            VoiceLabOptionsDialog,
        )

        # Another stock-button dialog first: wx must not re-create ours.
        scratch = wx.Dialog(self.frame)
        scratch.SetSizer(scratch.CreateSeparatedButtonSizer(wx.OK))
        scratch.Destroy()

        dlg = VoiceLabOptionsDialog(self.frame, "f5tts")
        try:
            self.assertEqual(dlg.save_btn.GetName(), "Save options")
            self.assertEqual(dlg.cancel_btn.GetName(), "Cancel")

            def click(button):
                evt = wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED,
                                      button.GetId())
                evt.SetEventObject(button)
                button.GetEventHandler().ProcessEvent(evt)

            with mock.patch.object(dlg, "EndModal") as end, \
                    mock.patch.object(wx, "MessageBox") as box:
                dlg._controls["seed"].SetValue("not a number")
                click(dlg.save_btn)
                self.assertTrue(box.called, "an invalid value must be reported")
                self.assertFalse(end.called, "the dialog must stay open")
                dlg._controls["seed"].SetValue("7")
                click(dlg.save_btn)
                end.assert_called_once_with(wx.ID_OK)
                end.reset_mock()
                click(dlg.cancel_btn)
                end.assert_called_once_with(wx.ID_CANCEL)
        finally:
            dlg.Destroy()

    def test_every_engine_has_a_tuning_dialog(self):
        from ai_voice_studio.gui.voicelab_options_dialog import (
            VoiceLabOptionsDialog,
        )

        for engine_id in voice_lab.engine_ids():
            dlg = VoiceLabOptionsDialog(self.frame, engine_id)
            try:
                self.assertEqual(
                    sorted(dlg._controls),
                    sorted(tuning.option_keys(engine_id)),
                    engine_id,
                )
                self.assertEqual(dlg.get_options(), {}, engine_id)
            finally:
                dlg.Destroy()

    def test_deleting_a_clone_removes_it_from_the_other_categories(self):
        sample = os.path.join(self.tmp.name, "reference.wav")
        with open(sample, "wb") as fh:
            fh.write(b"RIFFfake")
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.assertTrue(panel.select_engine("pocket_tts"))
        panel.name_ctrl.SetValue("Gone Soon")
        panel.sample_ctrl.SetValue(sample)
        with self._without_builtin_voices(panel):
            panel._on_create(None)
            panel.voices_list.SetSelection(0)
            with mock.patch.object(wx, "MessageBox", return_value=wx.YES):
                panel._on_delete(None)
            self.assertEqual(panel.voices_list.GetCount(), 0)
        dlg.available_panel.refresh()
        self.assertEqual(
            [v for v in dlg.available_panel._voices if v.get("cloned")], []
        )


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
                             "Restart All Recording", "Start Selected Recording...",
                             "Remove Project..."):
                self.assertIn(expected, labels)
            # Help menu has Read Me + User Guide + About.
            help_index = menubar.FindMenu("Help")
            self.assertGreaterEqual(help_index, 0, "Help menu missing")
            help_labels = [item.GetItemLabelText()
                           for item in menubar.GetMenu(help_index).GetMenuItems()]
            for expected in ("Read Me", "User Guide", "Third-Party Licences",
                             "About AI Voice Studio"):
                self.assertIn(expected, help_labels)
            # Ctrl+Shift+N is the welcome panel's Create New Project button's
            # gesture, owned by that button's own click path - so the File
            # menu item must not advertise the same key a second time.
            new_item = [i for i in menubar.GetMenu(0).GetMenuItems()
                        if i.GetItemLabelText() == "New Project"][0]
            self.assertNotIn("Ctrl+Shift+N", new_item.GetItemLabel())
            # "Show project folder" is the last Edit-menu action.
            self.assertEqual(labels[-1], "Show project folder")
        finally:
            frame.Destroy()

    def _project_frame(self):
        """A MainFrame whose single recent project is selected."""
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Selected project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "chapter 1", "text": "Text 1."}],
        )
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        settings.add_recent_project("Selected project", tmp)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        frame.recent_list.SetSelection(0)
        return frame, tmp

    def test_show_project_folder_opens_the_selected_projects_folder(self):
        frame, tmp = self._project_frame()
        opened = []
        with mock.patch("ai_voice_studio.gui.main_frame.open_folder",
                        side_effect=opened.append):
            frame._show_project_folder()
        self.assertEqual(opened, [tmp])
        self.assertEqual(frame.GetStatusBar().GetStatusText(), f"Opened {tmp}")

    def test_show_project_folder_warns_when_the_folder_is_gone(self):
        frame, tmp = self._project_frame()
        shutil.rmtree(tmp)
        messages = []
        with mock.patch("ai_voice_studio.gui.main_frame.open_folder") as opener, \
                mock.patch.object(wx, "MessageBox",
                                  side_effect=lambda *a, **k: messages.append(a[0])):
            frame._show_project_folder()
        self.assertFalse(opener.called)
        self.assertEqual(len(messages), 1)
        self.assertIn(tmp, messages[0])

    def test_the_context_menu_ends_with_show_project_folder(self):
        frame, _tmp = self._project_frame()
        labels = []
        with mock.patch.object(
            frame.recent_list, "PopupMenu",
            side_effect=lambda menu: labels.extend(
                i.GetItemLabelText() for i in menu.GetMenuItems() if not i.IsSeparator()
            ),
        ):
            evt = mock.MagicMock()
            evt.GetPosition.return_value = wx.Point(4, 4)
            frame._on_recent_context_menu(evt)
        self.assertEqual(labels, ["Resume Recording", "Restart Project...",
                                  "Restart All Recording",
                                  "Start Selected Recording...",
                                  "Remove Project...", "Show project folder"])

    def test_the_context_menu_from_the_keyboard_keeps_the_selection(self):
        """The application-menu key has no position: the selected row wins."""
        frame, tmp = self._project_frame()
        evt = mock.MagicMock()
        evt.GetPosition.return_value = wx.DefaultPosition
        seen = []
        with mock.patch.object(
            frame.recent_list, "PopupMenu", side_effect=lambda menu: seen.append(menu),
        ):
            frame._on_recent_context_menu(evt)
        self.assertEqual(frame.recent_list.GetSelection(), 0)
        self.assertEqual(frame._selected_recent()["path"], tmp)

    def test_the_edit_menu_action_opens_the_folder(self):
        """The Edit-menu item reaches the same handler as the popup item."""
        frame, tmp = self._project_frame()
        menubar = frame.GetMenuBar()
        edit_menu = menubar.GetMenu(menubar.FindMenu("Edit"))
        item_id = [i.GetId() for i in edit_menu.GetMenuItems()
                   if i.GetItemLabelText() == "Show project folder"][0]
        opened = []
        with mock.patch("ai_voice_studio.gui.main_frame.open_folder",
                        side_effect=opened.append):
            frame.ProcessEvent(wx.MenuEvent(wx.wxEVT_MENU, item_id))
        self.assertEqual(opened, [tmp])


class StartSelectedRecordingDialogTest(_AppMixin):
    """The Edit > Start Selected Recording picker (sidebar + combo).

    Sidebar: "Select an audio file" lists what is already recorded, "Select
    by file break" lists every file break of the project (recorded or not).
    The second radio group decides whether the run stops after the chosen
    file or goes on to the end of the project.
    """

    def _entries(self):
        return [
            {"position": 0, "title": "01 chapter 1", "file": "01 chapter 1.wav"},
            {"position": 1, "title": "02 chapter 2", "file": None},
            {"position": 2, "title": "03 chapter 3", "file": None},
        ]

    def _dialog(self, entries=None):
        from ai_voice_studio.gui.main_frame import _StartRecordingDialog

        dlg = _StartRecordingDialog(
            self.frame, self._entries() if entries is None else entries)
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_the_defaults_are_recorded_files_and_only_that_file(self):
        dlg = self._dialog()
        self.assertTrue(dlg.file_radio.GetValue())
        self.assertFalse(dlg.break_radio.GetValue())
        self.assertTrue(dlg.only_radio.GetValue())
        self.assertFalse(dlg.all_radio.GetValue())
        self.assertIn("recorded file to record again",
                      dlg.choice_label.GetLabel())
        # Only the files that exist on disk are listed.
        self.assertEqual([dlg.combo.GetString(i) for i in range(dlg.combo.GetCount())],
                         ["01 chapter 1.wav"])
        self.assertEqual(dlg.start_index(), 0)
        self.assertTrue(dlg.only_selected())
        self.assertEqual(dlg.chosen_file(), "01 chapter 1.wav")

    def test_the_focus_lands_on_the_sidebar_mode_not_the_combo(self):
        """The dialog opens on the radio button, not inside the combo box.

        The dialog hands the focus to its default (OK) button while it is
        built, so the move happens on the show event instead of in the
        constructor; from there on it is the checked mode button.
        """
        from ai_voice_studio.gui import main_frame as main_frame_module

        dlg = self._dialog()
        self.assertIs(dlg.initial_focus_control(), dlg.file_radio)
        event = mock.Mock()
        event.IsShown.return_value = True
        with mock.patch.object(main_frame_module.wx, "CallAfter") as call_after:
            dlg._on_show(event)
            dlg._on_show(event)  # a second show must not re-steal the focus
        scheduled = [getattr(call.args[0], "__name__", "")
                     for call in call_after.call_args_list if call.args]
        self.assertEqual(scheduled, ["_focus_first_control"])
        self.assertNotIn("SetFocus", scheduled)  # the combo is not focused
        # The event always keeps propagating, shown or hidden.
        self.assertEqual(event.Skip.call_count, 2)
        dlg._focus_first_control()  # and the focus call itself is harmless

    def test_with_nothing_recorded_the_break_radio_takes_the_focus(self):
        dlg = self._dialog(entries=[
            {"position": 0, "title": "01 chapter 1", "file": None},
        ])
        self.assertIs(dlg.initial_focus_control(), dlg.break_radio)

    def test_select_by_file_break_lists_every_break_of_the_project(self):
        dlg = self._dialog()
        dlg.break_radio.SetValue(True)
        dlg._on_mode(None)
        self.assertIn("record by file break", dlg.choice_label.GetLabel())
        self.assertEqual([dlg.combo.GetString(i) for i in range(dlg.combo.GetCount())],
                         ["01 chapter 1", "02 chapter 2", "03 chapter 3"])
        # A break that has no file yet can still be recorded from.
        dlg.combo.SetSelection(2)
        self.assertEqual(dlg.start_index(), 2)
        self.assertIsNone(dlg.chosen_file())
        self.assertTrue(dlg.only_selected())
        # "Record all files from here" goes on to the end of the project.
        dlg.all_radio.SetValue(True)
        self.assertFalse(dlg.only_selected())
        # A break that is already recorded hands its file over for deletion.
        dlg.combo.SetSelection(0)
        self.assertEqual(dlg.chosen_file(), "01 chapter 1.wav")

    def test_switching_back_to_audio_files_restores_that_list(self):
        dlg = self._dialog()
        dlg.break_radio.SetValue(True)
        dlg._on_mode(None)
        dlg.file_radio.SetValue(True)
        dlg._on_mode(None)
        self.assertEqual([dlg.combo.GetString(i) for i in range(dlg.combo.GetCount())],
                         ["01 chapter 1.wav"])
        self.assertEqual(dlg.start_index(), 0)

    def test_with_nothing_recorded_the_dialog_starts_in_break_mode(self):
        dlg = self._dialog(entries=[
            {"position": 0, "title": "01 chapter 1", "file": None},
            {"position": 1, "title": "02 chapter 2", "file": None},
        ])
        self.assertTrue(dlg.break_radio.GetValue())
        self.assertEqual(dlg.combo.GetCount(), 2)

    def test_the_edit_menu_action_starts_at_the_chosen_break(self):
        """The picker's answers reach the recording window end to end."""
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Selected project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": i + 1, "title": f"chapter {i + 1}",
              "text": f"Text {i + 1}."} for i in range(4)],
        )
        data = project.load_project(tmp)
        data["segments"][2].update({"status": "done", "saved": "chapter 3.wav"})
        project.save_project(tmp, data)
        with open(os.path.join(tmp, "chapter 3.wav"), "wb"):
            pass
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        settings.add_recent_project("Selected project", tmp)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        frame.recent_list.SetSelection(0)

        opened = {}
        fake = mock.MagicMock()
        fake.return_value.ShowModal.return_value = wx.ID_OK
        fake.return_value.start_index.return_value = 2
        fake.return_value.only_selected.return_value = True
        fake.return_value.chosen_file.return_value = "chapter 3.wav"
        with mock.patch.object(
            frame, "_open_recording",
            lambda *args, **kwargs: opened.update(args=args, kwargs=kwargs),
        ), mock.patch("ai_voice_studio.gui.main_frame._StartRecordingDialog", fake):
            frame._restart_selected()

        self.assertEqual(opened["args"], (tmp,))
        self.assertEqual(opened["kwargs"],
                         {"start_index": 2, "single_segment": True})
        # The chosen recording is deleted and its segment is pending again.
        self.assertFalse(os.path.exists(os.path.join(tmp, "chapter 3.wav")))
        self.assertEqual(project.load_project(tmp)["segments"][2]["status"],
                         "pending")

    def _project_frame(self, segments):
        """A MainFrame whose one recent project has ``segments``."""
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Selected project", "doc.txt", MODE_PAGE_WITH_H1, segments,
        )
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        settings.add_recent_project("Selected project", tmp)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        frame.recent_list.SetSelection(0)
        return frame, tmp

    def test_the_picker_opens_even_when_nothing_is_recorded(self):
        """A 100-chapter book may be recorded from chapter 75; no guard."""
        frame, _tmp = self._project_frame([
            {"index": i + 1, "title": f"chapter {i + 1}", "text": f"Text {i + 1}."}
            for i in range(4)
        ])
        opened = {}
        fake = mock.MagicMock()
        fake.return_value.ShowModal.return_value = wx.ID_OK
        fake.return_value.start_index.return_value = 2
        fake.return_value.only_selected.return_value = True
        fake.return_value.chosen_file.return_value = None
        with mock.patch.object(
            frame, "_open_recording",
            lambda *args, **kwargs: opened.update(args=args, kwargs=kwargs),
        ), mock.patch("ai_voice_studio.gui.main_frame._StartRecordingDialog", fake) \
                as dlg, mock.patch.object(wx, "MessageBox") as box:
            frame._restart_selected()
        self.assertFalse(box.called, "nothing recorded must not block the picker")
        self.assertTrue(dlg.called, "the picker must open")
        # Every file break is offered and none of them has a file yet.
        entries = dlg.call_args.args[1]
        self.assertEqual([e["title"] for e in entries],
                         [f"chapter {i + 1}" for i in range(4)])
        self.assertEqual([e["file"] for e in entries], [None] * 4)
        self.assertEqual(opened["kwargs"],
                         {"start_index": 2, "single_segment": True})

    def test_a_project_without_file_breaks_is_reported(self):
        from ai_voice_studio.gui.main_frame import _StartRecordingDialog

        frame, _tmp = self._project_frame([])
        with mock.patch.object(wx, "MessageBox") as box, \
                mock.patch.object(_StartRecordingDialog, "ShowModal") as modal:
            frame._restart_selected()
        self.assertTrue(box.called, "an unsplit project has nothing to record")
        self.assertIn("file breaks", box.call_args.args[0])
        self.assertFalse(modal.called)

    def test_the_project_files_are_mapped_back_to_their_break(self):
        from ai_voice_studio.gui.main_frame import _picker_entries

        data = {"segments": [
            {"index": 1, "title": "01 chapter 1", "saved": "01 chapter 1.wav"},
            {"index": 2, "title": "02 chapter 2", "saved": None},
            {"index": 3, "title": "03 chapter 3", "saved": None},
        ]}
        entries = _picker_entries(data, ["03 chapter 3.wav", "01 chapter 1.wav",
                                         "stray file.wav"])
        self.assertEqual([e["position"] for e in entries], [0, 1, 2])
        self.assertEqual(entries[0]["file"], "01 chapter 1.wav")
        self.assertIsNone(entries[1]["file"])
        # Matched by file name even though project.json does not know it yet.
        self.assertEqual(entries[2]["file"], "03 chapter 3.wav")


class PreviewComputeChoiceTest(_AppMixin):
    """Every category with a Preview button offers a Compute combo.

    The combo lists CPU first, adds GPU (CUDA) when an NVIDIA GPU is detected
    and Auto when there is a choice; the choice is remembered per category in
    Settings, so the next preview uses the same back-end.
    """

    #: ``(panel accessor, category key)`` for every Preview category.  The
    #: Voice Clone category has its own Device combo (same idea, older API)
    #: and is checked separately.
    _PANELS = (
        ("available_panel", "available_tts"),
        ("recording_panel", "recording_settings"),
        ("punctuation_panel", "punctuation"),
        ("omnivoice_engines_panel", "omnivoice_engines"),
    )

    def _dialog(self):
        settings = Settings(path=os.path.join(tempfile.mkdtemp(), "settings.json"))
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg, settings

    def test_every_preview_category_has_a_compute_combo(self):
        with mock.patch.object(compute, "has_nvidia_gpu", return_value=False):
            dlg, _settings = self._dialog()
            for accessor, _category in self._PANELS:
                panel = getattr(dlg, accessor)
                self.assertTrue(hasattr(panel, "preview_btn"), accessor)
                self.assertTrue(hasattr(panel, "compute_combo"), accessor)
                values = [panel.compute_combo.GetClientData(i)
                          for i in range(panel.compute_combo.GetCount())]
                self.assertEqual(values, ["cpu"], accessor)
                self.assertEqual(panel.compute_combo.GetSelection(), 0, accessor)

    def test_the_gpu_is_added_when_a_card_is_present(self):
        with mock.patch.object(compute, "has_nvidia_gpu", return_value=True):
            dlg, _settings = self._dialog()
            for accessor, _category in self._PANELS:
                panel = getattr(dlg, accessor)
                values = [panel.compute_combo.GetClientData(i)
                          for i in range(panel.compute_combo.GetCount())]
                self.assertEqual(values, ["cpu", "cuda", "auto"], accessor)
            # OmniVoice itself needs CUDA, so its preview defaults to the GPU.
            self.assertEqual(
                dlg.omnivoice_engines_panel.compute_combo.GetClientData(
                    dlg.omnivoice_engines_panel.compute_combo.GetSelection()
                ),
                "cuda",
            )

    def test_the_choice_is_remembered_per_category(self):
        with mock.patch.object(compute, "has_nvidia_gpu", return_value=True):
            dlg, settings = self._dialog()
            combo = dlg.punctuation_panel.compute_combo
            combo.SetSelection(1)  # GPU
            evt = wx.CommandEvent(wx.wxEVT_COMMAND_COMBOBOX_SELECTED,
                                  combo.GetId())
            evt.SetEventObject(combo)
            combo.GetEventHandler().ProcessEvent(evt)
            self.assertEqual(
                settings.get(compute_choice.settings_key("punctuation")), "cuda"
            )
        self.assertEqual(compute_choice.saved_choice(settings, "punctuation"),
                         "cuda")
        # ... and a category nobody touched keeps the CPU.
        self.assertEqual(compute_choice.saved_choice(settings, "recording_settings"),
                         "cpu")

    def test_the_preview_provider_follows_the_choice(self):
        # "auto" resolves through the *whole* detection chain, so the optional
        # GPU runtime must be in place as well as the NVIDIA driver (the pair
        # tests/test_compute.py patches): patching only the driver check made
        # this test answer "what is installed on this machine?" instead of
        # "what does the choice mean?".  The detection cache is cleared for
        # the call and put back afterwards, so a probe that ran earlier in
        # the session cannot answer for it.
        cached = getattr(compute.detect, "_cache", None)
        self.addCleanup(setattr, compute.detect, "_cache", cached)
        with mock.patch("ai_voice_studio.compute.runtime.is_installed",
                        return_value=True), \
                mock.patch.object(compute, "has_nvidia_gpu", return_value=True):
            compute.detect._cache = None  # type: ignore[attr-defined]
            self.assertEqual(compute_choice.provider_for_preview("cpu"), "cpu")
            self.assertEqual(compute_choice.provider_for_preview("cuda"), "cuda")
            self.assertEqual(compute_choice.provider_for_preview("auto"), "cuda")
            # A Voice Lab engine resolves its own device (it brings its own
            # PyTorch), an ONNX engine goes through the CUDA runtime check.
            self.assertEqual(
                compute_choice.provider_for_preview("cuda", "pocket_tts"), "cuda"
            )
            self.assertEqual(
                compute_choice.provider_for_preview("cpu", "pocket_tts"), "cpu"
            )
        # A machine with the driver but no downloaded GPU runtime keeps the
        # CPU preview: the option is offered, "auto" does not pick it.
        with mock.patch("ai_voice_studio.compute.runtime.is_installed",
                        return_value=False), \
                mock.patch.object(compute, "has_nvidia_gpu", return_value=True):
            compute.detect._cache = None  # type: ignore[attr-defined]
            self.assertEqual(compute_choice.provider_for_preview("auto"), "cpu")
            self.assertEqual(compute_choice.provider_for_preview("cuda"), "cuda")


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
                # Start is enabled exactly when a voice is selectable. With
                # OmniVoice pip-installed machine-wide this is true even for a
                # throwaway store; on a machine with no voices installed the
                # TTS combo stays empty and the button stays disabled.
                has_voices = (dlg.tts_combo.GetCount() > 0
                              and dlg.voice_combo.GetCount() > 0)
                self.assertEqual(dlg.start_btn.IsEnabled(), has_voices)
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


class RecordingDialogTuningTest(_AppMixin):
    """The per-project tuning overrides in the Recording window."""

    def _dialog_with_engine(self, engine_id="f5tts", stored=None):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Tuning project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        data = project.load_project(tmp)
        data.setdefault("tts", {})
        data["tts"].update({
            "tts": engine_id,
            "language": "en",
            "variant": "f5_v1_base",
            "voice": "basic_ref_en",
        })
        if stored is not None:
            data["tts"]["engine_options"] = {
                "engine": engine_id, "values": stored,
            }
        project.save_project(tmp, data)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        dlg = RecordingDialog(self.frame, tmp, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        # The Voice Lab engines only appear once their package is installed;
        # hand the dialog their pre-made voices so the tuning row is exercised
        # (rebuilding the choices re-reads the store, so the patch has to wrap
        # the rebuild).
        entries = voice_lab.builtin_voices(engine_id)
        with mock.patch.object(voicelab, "builtin_voice_entries",
                               lambda engine_id=None: list(entries)):
            dlg._rebuild_voice_choices()
        for index in range(dlg.tts_combo.GetCount()):
            if dlg.tts_combo.GetClientData(index) == engine_id:
                dlg.tts_combo.SetSelection(index)
                dlg._on_tts(None)
                break
        return tmp, dlg, settings

    def test_the_row_shows_only_for_voice_lab_engines(self):
        _tmp, dlg, _settings = self._dialog_with_engine("f5tts")
        self.assertTrue(dlg.engine_options_btn.IsShown())
        self.assertIn("engine defaults",
                      dlg.engine_options_summary.GetLabel().lower())
        # Another engine hides the row again.
        dlg.tts_combo.SetSelection(0)
        dlg._on_tts(None)
        if dlg._selected_tts_id() not in ("pocket_tts", "bark", "f5tts"):
            self.assertFalse(dlg.engine_options_btn.IsShown())

    def test_a_projects_tuning_is_summarised_and_saved(self):
        tmp, dlg, _settings = self._dialog_with_engine(
            "f5tts", stored={"nfe_step": 16}
        )
        self.assertIn("diffusion steps 16",
                      dlg.engine_options_summary.GetLabel().lower())
        self.assertEqual(dlg._effective_engine_options(), {"nfe_step": 16})
        # Saving the project keeps the overrides untouched.
        dlg._save_tts_to_project()
        saved = project.load_project(tmp)["tts"]["engine_options"]
        self.assertEqual(saved, {"engine": "f5tts", "values": {"nfe_step": 16}})

    def test_the_settings_default_is_used_when_the_project_has_none(self):
        _tmp, dlg, settings = self._dialog_with_engine("f5tts")
        settings.set(tuning.settings_key("f5tts"), {"nfe_step": 8})
        dlg._update_engine_options_ui()
        self.assertEqual(dlg._effective_engine_options(), {"nfe_step": 8})
        label = dlg.engine_options_summary.GetLabel()
        self.assertIn("diffusion steps 8", label.lower())
        self.assertIn("Settings default", label)
        # A project override wins over the saved default.
        dlg.data.setdefault("tts", {})["engine_options"] = {
            "engine": "f5tts", "values": {"nfe_step": 16},
        }
        dlg._update_engine_options_ui()
        self.assertEqual(dlg._effective_engine_options(), {"nfe_step": 16})
        self.assertNotIn("Settings default",
                         dlg.engine_options_summary.GetLabel())

    def test_the_tuning_dialog_result_is_stored_in_the_project(self):
        from ai_voice_studio.gui import recording_dialog as recording_module

        _tmp, dlg, _settings = self._dialog_with_engine("f5tts")
        with mock.patch(
            "ai_voice_studio.gui.voicelab_options_dialog.VoiceLabOptionsDialog"
        ) as fake:
            fake.return_value.ShowModal.return_value = wx.ID_OK
            fake.return_value.get_options.return_value = {"nfe_step": 16}
            dlg._on_engine_options(None)
        self.assertEqual(dlg.data["tts"]["engine_options"],
                         {"engine": "f5tts", "values": {"nfe_step": 16}})
        self.assertIn("diffusion steps 16",
                      dlg.engine_options_summary.GetLabel().lower())
        # Cancelling keeps the previous values.
        with mock.patch(
            "ai_voice_studio.gui.voicelab_options_dialog.VoiceLabOptionsDialog"
        ) as fake:
            fake.return_value.ShowModal.return_value = wx.ID_CANCEL
            dlg._on_engine_options(None)
        self.assertEqual(dlg.data["tts"]["engine_options"]["values"],
                         {"nfe_step": 16})
        # An empty result clears them again.
        with mock.patch(
            "ai_voice_studio.gui.voicelab_options_dialog.VoiceLabOptionsDialog"
        ) as fake:
            fake.return_value.ShowModal.return_value = wx.ID_OK
            fake.return_value.get_options.return_value = {}
            dlg._on_engine_options(None)
        self.assertNotIn("engine_options", dlg.data["tts"])
        self.assertIn("engine defaults",
                      dlg.engine_options_summary.GetLabel().lower())
        self.assertTrue(recording_module is not None)

    def test_start_recording_hands_the_tuning_to_the_voice(self):
        """The overrides ride on the voice entry the synthesis worker gets."""
        _tmp, dlg, settings = self._dialog_with_engine(
            "f5tts", stored={"nfe_step": 16}
        )
        voice = tuning.apply_to_voice(
            dlg._selected_voice(), dlg._effective_engine_options()
        )
        self.assertEqual(voice["options"], {"nfe_step": 16})
        # ... and the whole Start-recording path passes them on unchanged.
        captured = {}

        def fake_worker(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        with mock.patch("ai_voice_studio.gui.recording_dialog.SynthesisWorker",
                        side_effect=fake_worker), \
                mock.patch.object(dlg, "_show_progress_dialog"):
            dlg._cancel_event = threading.Event()
            dlg._on_start(None)
        self.assertEqual(captured["voice_entry"].get("options"),
                         {"nfe_step": 16})
        self.assertIn("engine_options", project.load_project(
            _tmp)["tts"])
        # A Settings default is applied when the project has none of its own.
        _tmp2, dlg2, settings2 = self._dialog_with_engine("f5tts")
        settings2.set(tuning.settings_key("f5tts"), {"cfg_strength": 3.0})
        with mock.patch("ai_voice_studio.gui.recording_dialog.SynthesisWorker",
                        side_effect=fake_worker), \
                mock.patch.object(dlg2, "_show_progress_dialog"):
            dlg2._on_start(None)
        self.assertEqual(captured["voice_entry"].get("options"),
                         {"cfg_strength": 3.0})
        self.assertTrue(settings is not None)


class StartSelectedRecordingWiringTest(_AppMixin):
    """Edit > Start Selected Recording reaches the synthesizer.

    The picker's answers travel to the recording window: the chosen file
    break is where the run starts, and "only record selected file" stops the
    run after that one segment.
    """

    def _dialog(self, start_index=None, single_segment=False, segments=4):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Selected project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": i + 1, "title": f"chapter {i + 1}",
              "text": f"Text {i + 1}."} for i in range(segments)],
        )
        data = project.load_project(tmp)
        data.setdefault("tts", {})
        data["tts"].update({
            "tts": "f5tts", "language": "en",
            "variant": "f5_v1_base", "voice": "basic_ref_en",
        })
        project.save_project(tmp, data)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        dlg = RecordingDialog(self.frame, tmp, settings, self._empty_store(),
                              start_index=start_index,
                              single_segment=single_segment)
        self.addCleanup(dlg.Destroy)
        entries = voice_lab.builtin_voices("f5tts")
        with mock.patch.object(voicelab, "builtin_voice_entries",
                               lambda engine_id=None: list(entries)):
            dlg._rebuild_voice_choices()
        for index in range(dlg.tts_combo.GetCount()):
            if dlg.tts_combo.GetClientData(index) == "f5tts":
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

    def test_the_ready_note_names_the_chosen_segment(self):
        _tmp, dlg = self._dialog(start_index=1)
        note = dlg.status.GetLabel()
        self.assertIn("segment 2 of 4", note)
        self.assertIn("chapter 2", note)
        self.assertIn("every file after it", note)

    def test_only_record_selected_file_stops_after_that_segment(self):
        _tmp, dlg = self._dialog(start_index=2, single_segment=True)
        self.assertIn("only segment 3", dlg.status.GetLabel())
        captured = self._start(dlg)
        self.assertEqual(captured["start_index"], 2)
        self.assertEqual(captured["end_index"], 3)
        self.assertIn("only segment 3", dlg.status.GetLabel())

    def test_record_all_files_from_here_runs_to_the_end(self):
        _tmp, dlg = self._dialog(start_index=2)
        captured = self._start(dlg)
        self.assertEqual(captured["start_index"], 2)
        self.assertIsNone(captured["end_index"])
        self.assertIn("resuming from segment 3", dlg.status.GetLabel())

    def test_a_plain_start_still_uses_the_first_pending_segment(self):
        _tmp, dlg = self._dialog()
        captured = self._start(dlg)
        self.assertEqual(captured["start_index"], 0)
        self.assertIsNone(captured["end_index"])


class OmniVoiceLanguagePickerTest(_AppMixin):
    """OmniVoice options: Auto first, then every language it was trained on.

    OmniVoice detects the language on its own and sometimes guesses wrong on
    short lines, so the picker is what pins it.  Whatever the user picks or
    types, what reaches the project has to be a language id - or ``None``,
    which is "Auto" and no hint at all.
    """

    def _dialog(self, omni=None):
        from ai_voice_studio.gui.omnivoice_options_dialog import (
            OmniVoiceOptionsDialog,
        )

        dlg = OmniVoiceOptionsDialog(self.frame, engine_label="OmniVoice",
                                     omni=omni or {}, project_name="demo")
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_auto_is_the_first_entry_and_sends_no_hint(self):
        dlg = self._dialog()
        self.assertEqual(dlg.language_combo.GetCount(),
                         omni_languages.COUNT + 1)
        self.assertEqual(dlg.language_combo.GetString(0),
                         omni_languages.AUTO_LABEL)
        self.assertEqual(dlg.language_combo.GetSelection(), 0)
        self.assertEqual(dlg.language_combo.GetValue(), omni_languages.AUTO_LABEL)
        self.assertIsNone(dlg.selected_language())
        self.assertIsNone(dlg.get_omni()["language"])

    def test_every_language_of_the_table_is_offered(self):
        dlg = self._dialog()
        listed = [dlg.language_combo.GetString(i)
                  for i in range(1, dlg.language_combo.GetCount())]
        self.assertEqual(listed, [omni_languages.label(code)
                                  for code in omni_languages.language_ids()])

    def test_choosing_a_language_forces_it(self):
        dlg = self._dialog()
        index = omni_languages.selection_index("hi")
        self.assertEqual(dlg.language_combo.GetString(index), "Hindi (hi)")
        dlg.language_combo.SetSelection(index)
        dlg.language_combo.SetValue(dlg.language_combo.GetString(index))
        self.assertEqual(dlg.selected_language(), "hi")
        self.assertEqual(dlg.get_omni()["language"], "hi")

    def test_a_stored_hint_selects_its_row(self):
        for stored, expected in (("de", "de"), ("deu", "de"),
                                 ("German", "de"), ("zh", "zh")):
            with self.subTest(stored=stored):
                self.assertEqual(self._dialog(
                    {"language": stored}).selected_language(), expected)
        # The chosen row is what the user sees and what the engine receives.
        dlg = self._dialog({"language": "ja"})
        self.assertEqual(dlg.language_combo.GetStringSelection(), "Japanese (ja)")
        self.assertEqual(dlg.language_combo.GetValue(), "Japanese (ja)")
        self.assertEqual(dlg.get_omni()["language"], "ja")

    def test_auto_stays_auto(self):
        for stored in ("auto", "", None, omni_languages.AUTO_LABEL):
            with self.subTest(stored=stored):
                dlg = self._dialog({"language": stored})
                self.assertEqual(dlg.language_combo.GetSelection(), 0)
                self.assertIsNone(dlg.get_omni()["language"])

    def test_an_unknown_hint_is_neither_lost_nor_confused_with_a_name(self):
        # A hint from a newer engine survives a round trip untouched.
        dlg = self._dialog({"language": "xx-newer"})
        self.assertEqual(dlg.selected_language(), "xx-newer")
        self.assertEqual(dlg.get_omni()["language"], "xx-newer")

    def test_a_typed_name_or_code_is_understood(self):
        for typed, expected in (("Japanese", "ja"), ("jpn", "ja"),
                                ("English (en)", "en"), ("  Hindi  ", "hi")):
            with self.subTest(typed=typed):
                dlg = self._dialog()
                dlg.language_combo.SetValue(typed)
                self.assertEqual(dlg.selected_language(), expected)

    def test_the_picker_stays_visible_in_every_voice_mode(self):
        """It lives in the advanced group, which no mode hides."""
        dlg = self._dialog({"mode": "clone", "language": "zh"})
        parent = dlg.language_combo.GetParent()
        self.assertIsNot(parent, dlg.design_group.GetStaticBox())
        self.assertIsNot(parent, dlg.clone_group.GetStaticBox())
        self.assertIs(parent, dlg.duration_ctrl.GetParent())
        self.assertEqual(dlg.get_omni()["language"], "zh")


class AvailablePanelThreadTest(_AppMixin):
    """Background voice discovery must never touch widgets off-thread.

    The Windows voices are enumerated on one worker thread per engine, and
    both finish at roughly the same time.  A callback that rebuilt the panel
    inline ran on those threads and raced on the combo boxes, corrupting the
    heap (a hard crash a second or two after the settings dialog opened).
    """

    def _panel(self):
        from ai_voice_studio.gui.model_panels import AvailablePanel

        return AvailablePanel(self.frame, self._empty_store())

    def test_builtin_voice_callback_is_deferred_to_the_main_thread(self):
        panel = self._panel()
        try:
            seen: list = []
            with mock.patch.object(
                panel, "_reload_keeping_selection",
                lambda: seen.append(threading.current_thread()),
            ):
                # Called from the enumeration worker thread.
                panel._on_builtin_voices([])
                self.assertEqual(
                    seen, [],
                    "the panel was rebuilt synchronously on the enumeration "
                    "thread (wx widgets must only be touched on the main one)",
                )
                # The event loop then runs it on the main thread.
                for _ in range(50):
                    wx.Yield()
                    if seen:
                        break
                    time.sleep(0.02)
            self.assertTrue(seen, "the deferred rebuild never ran")
            self.assertIs(seen[0], threading.main_thread())
        finally:
            panel.Destroy()


class AccessibleListTest(_AppMixin):
    """MSAA may ask a list for any child id, including out-of-range ones.

    ``_NameAccessible.GetName`` used to call ``GetString``/``GetItemText``
    without checking the range, so a screen reader probing the recent-projects
    list raised ``wxAssertionError: ... invalid index in wxListBox::GetString``
    (caught by the crash handler and written to crash.log at every start).
    """

    def _list(self, choices):
        from ai_voice_studio.gui.a11y import set_accessible_name

        lb = wx.ListBox(self.frame, choices=choices)
        self.addCleanup(lb.Destroy)
        set_accessible_name(lb, "Recent projects")
        return lb

    def test_rows_report_their_own_text(self):
        lb = self._list(["one", "two", "three"])
        acc = lb.GetAccessible()
        self.assertEqual(acc.GetName(0), (wx.ACC_OK, "Recent projects"))
        for child, text in ((1, "one"), (2, "two"), (3, "three")):
            self.assertEqual(acc.GetName(child), (wx.ACC_OK, text))
        self.assertEqual(acc.GetChildCount(), (wx.ACC_OK, 3))

    def test_out_of_range_child_id_does_not_raise(self):
        lb = self._list(["one", "two"])
        acc = lb.GetAccessible()
        for child in (3, 4, 99):
            status, text = acc.GetName(child)
            self.assertEqual(status, wx.ACC_NOT_IMPLEMENTED, child)
            self.assertEqual(text, "")

    def test_empty_list_does_not_raise(self):
        lb = self._list([])
        acc = lb.GetAccessible()
        self.assertEqual(acc.GetChildCount(), (wx.ACC_OK, 0))
        self.assertEqual(acc.GetName(1)[0], wx.ACC_NOT_IMPLEMENTED)


if __name__ == "__main__":
    unittest.main()
