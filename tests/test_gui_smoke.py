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
import webbrowser
from unittest import mock

import wx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wx_test_app import get_app

from ai_voice_studio import (
    compute,
    gestures,
    hf_access,
    project,
    voicelab,
)
from ai_voice_studio.constants import AUDIO_MODE_CHOICES, MODE_PAGE_WITH_H1, PROJECT_TYPE_STT
from ai_voice_studio.describe import catalog as describe_catalog
from ai_voice_studio.gui import compute_choice
from ai_voice_studio.gui import settings_dialog as settings_module
from ai_voice_studio.gui.accept_dialog import (
    AcceptanceDialog,
    record_acceptance,
    terms_current,
)
from ai_voice_studio.gui.main_frame import MainFrame
from ai_voice_studio.gui.new_project_wizard import NewProjectWizard
from ai_voice_studio.gui.recording_dialog import RecordingDialog
from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.gui.stt_dialog import STTTranscribeDialog
from ai_voice_studio.omnivoice import languages as omni_languages
from ai_voice_studio.settings import Settings
from ai_voice_studio.stt import catalog as stt_catalog
from ai_voice_studio.stt import engine as stt_engine
from ai_voice_studio.tts.models import ModelStore
from ai_voice_studio.voicelab import engines as voice_lab
from ai_voice_studio.voicelab import options as tuning


def _temp_settings() -> Settings:
    """A throwaway settings store - never the user's own settings file.

    A test that builds ``Settings()`` with no path reads - and, through
    ``theme.set_wallpaper``'s immediate save, writes - the real
    ``%APPDATA%\\AIVoiceStudio\\settings.json``.  The suite then depends on
    whatever the machine's user has chosen there (a rebound gesture made a
    test fail) and can overwrite it.  Every GUI test gets its own file.
    """
    return Settings(path=os.path.join(tempfile.mkdtemp(prefix="aivs_settings_"),
                                      "settings.json"))


def _ffmpeg_available() -> bool:
    """True when this machine has an FFmpeg the audio-description flow accepts.

    The audio-description wizard asks for one (``describe_dialog._ensure_ffmpeg``)
    and, when it is missing, opens a modal install prompt that no test can
    answer.  The process then dies with a Windows fatal exception *inside* wx
    and pytest prints no summary at all, so the results of every test that ran
    before it are lost.  Skipping the class on such a machine is the honest
    answer; the code paths that *report* a missing FFmpeg have their own tests.
    """
    from ai_voice_studio.audio.ffmpeg import ensure_ffmpeg

    return ensure_ffmpeg() is not None


def _inline_call_after(fn, *args, **kwargs):
    """Run a ``wx.CallAfter`` body inline - but only from the test thread.

    The Voice Clone panel marshals work out of its background threads (the
    built-in voice probe asks a worker which voices the installed package
    really has) with ``wx.CallAfter``.  A mock that runs *everything* inline
    therefore touches widgets from a thread wx never created a window in, and
    Windows does not fail the assertion - it kills the process with an access
    violation, which shows up as an unrelated test hanging.  Calls from the
    test thread are the GUI thread here, so those still run immediately and
    the tests stay synchronous.
    """
    if threading.current_thread() is threading.main_thread():
        return fn(*args, **kwargs)
    return None


class _AppMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def tearDown(self):
        """Let the windows destroyed by this test give their handles back.

        Windows counts USER objects (windows, menus, ...) per process and the
        limit is 10,000.  A ``Destroy()`` only *queues* the destruction: the
        handles come back when the pending events are processed, and a test run
        with no event loop never processes them.  Enough dialog-building tests
        in one process therefore end with wx failing to create any window at
        all - ``CreateWindowEx ... failed with error 0x00000486 (The current
        process has used all of its system allowance of handles for Window
        Manager objects)`` - and every later GUI test fails for no reason of
        its own.
        """
        with contextlib.suppress(Exception):
            wx.SafeYield()

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
        wizard = NewProjectWizard(self.frame, _temp_settings(), self._empty_store(),
                                  initial_name="My Book")
        try:
            self.assertEqual(wizard.page_details.name_ctrl.GetValue(), "My Book")
        finally:
            wizard.Destroy()

    def test_pages_build_their_controls(self):
        wizard = NewProjectWizard(self.frame, _temp_settings(), self._empty_store())
        try:
            details, mode = wizard.page_details, wizard.page_mode
            # Pin the project base to "Text to speech" so the routing check
            # below does not depend on the machine's saved settings.
            details.base_combo.SetSelection(0)

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
        wizard = NewProjectWizard(self.frame, _temp_settings(), self._empty_store())
        try:
            self.assertEqual(wizard.source_path, "")
            self.assertIsNotNone(wizard.page_details.name_ctrl)
            self.assertIsNotNone(wizard.page_mode.selected())
        finally:
            wizard.Destroy()


class InputGesturesPanelTest(_AppMixin):
    """Settings > Keyboard gestures: view, rebind and reset the commands."""

    def _panel(self, bindings=None):
        settings = _temp_settings()
        if bindings:
            settings.set("gestures.bindings", bindings)
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return settings, dlg, dlg.input_gestures_panel

    def test_the_tree_groups_the_commands_by_category(self):
        _settings, _dlg, panel = self._panel()
        root = panel.tree.GetRootItem()
        categories = []
        branch, cookie = panel.tree.GetFirstChild(root)
        while branch.IsOk():
            categories.append(panel.tree.GetItemText(branch))
            branch, cookie = panel.tree.GetNextChild(root, cookie)
        self.assertEqual(categories, gestures.categories_present())

    def test_selecting_a_command_lists_its_gestures(self):
        _settings, _dlg, panel = self._panel()
        panel.tree.SelectItem(panel._find_node("project.new"))
        panel._refresh_gestures()
        self.assertEqual(panel.selected_command(), "project.new")
        self.assertEqual(panel.gesture_list.GetCount(), 1)
        self.assertIn("Ctrl+Shift+N", panel.gesture_list.GetString(0))

    def test_removing_a_gesture_leaves_the_command_unbound(self):
        settings, _dlg, panel = self._panel()
        panel.tree.SelectItem(panel._find_node("project.new"))
        panel._refresh_gestures()
        panel.gesture_list.SetSelection(0)
        panel._on_remove(None)
        self.assertEqual(panel.gesture_list.GetCount(), 0)
        panel.apply_to_settings()
        self.assertEqual(settings.get("gestures.bindings")["project.new"], [])

    def test_restore_defaults_forgets_the_override(self):
        settings, _dlg, panel = self._panel({"project.new": ["kb:control+n"]})
        panel.tree.SelectItem(panel._find_node("project.new"))
        panel._refresh_gestures()
        self.assertTrue(panel.reset_btn.IsEnabled())
        panel._on_reset(None)
        self.assertEqual(panel.gesture_map.bindings_for("project.new"),
                         ["kb:control+shift+n"])
        panel.apply_to_settings()
        self.assertNotIn("project.new", settings.get("gestures.bindings"))

    def test_the_filter_narrows_the_tree(self):
        _settings, _dlg, panel = self._panel()
        panel.filter.SetValue("what's")
        panel._rebuild_tree()
        self.assertEqual(panel.selected_command(), "help.whats_new")

    def test_a_conflicting_gesture_is_reported(self):
        _settings, _dlg, panel = self._panel(
            {"project.open": ["kb:control+shift+n"]}
        )
        self.assertIn("kb:control+shift+n", panel.gesture_map.conflicts())
        panel.tree.SelectItem(panel._find_node("project.open"))
        panel._refresh_gestures()
        self.assertIn("Conflicts", panel.status.GetLabel())

    def test_adding_a_taken_gesture_asks_before_stealing_it(self):
        _settings, _dlg, panel = self._panel()
        panel.tree.SelectItem(panel._find_node("project.open"))
        panel._refresh_gestures()
        capture = mock.MagicMock()
        capture.gesture = "kb:control+shift+n"
        capture.ShowModal.return_value = wx.ID_OK
        with mock.patch.object(settings_module, "_GestureCaptureDialog",
                               return_value=capture), \
                mock.patch.object(wx, "MessageBox", return_value=wx.YES):
            panel._on_add(None)
        self.assertIn("kb:control+shift+n",
                      panel.gesture_map.bindings_for("project.open"))
        self.assertNotIn("kb:control+shift+n",
                         panel.gesture_map.bindings_for("project.new"))

    def test_declining_the_steal_leaves_the_gesture_alone(self):
        _settings, _dlg, panel = self._panel()
        panel.tree.SelectItem(panel._find_node("project.open"))
        capture = mock.MagicMock()
        capture.gesture = "kb:control+shift+n"
        capture.ShowModal.return_value = wx.ID_OK
        with mock.patch.object(settings_module, "_GestureCaptureDialog",
                               return_value=capture), \
                mock.patch.object(wx, "MessageBox", return_value=wx.NO):
            panel._on_add(None)
        self.assertNotIn("kb:control+shift+n",
                         panel.gesture_map.bindings_for("project.open"))
        self.assertIn("kb:control+shift+n",
                      panel.gesture_map.bindings_for("project.new"))

    def test_capturing_a_gesture_from_a_key_event(self):
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetKeyCode(ord("J"))
        evt.SetControlDown(True)
        evt.SetAltDown(True)
        self.assertEqual(settings_module.gesture_from_event(evt),
                         "kb:control+alt+j")

    def test_a_bare_letter_is_refused_as_a_gesture(self):
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetKeyCode(ord("J"))
        self.assertEqual(settings_module.gesture_from_event(evt), "kb:j")
        self.assertTrue(settings_module.gesture_problem("kb:j"))
        self.assertEqual(settings_module.gesture_problem("kb:control+j"), "")
        self.assertEqual(settings_module.gesture_problem("kb:f5"), "")

    def test_modifier_only_presses_are_not_a_gesture(self):
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetKeyCode(wx.WXK_CONTROL)
        evt.SetControlDown(True)
        self.assertEqual(settings_module.gesture_from_event(evt), "")


class MainFrameGestureTest(_AppMixin):
    """A configured gesture runs its command in the main window."""

    def _frame(self, bindings=None):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        if bindings:
            settings.set("gestures.bindings", bindings)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        return frame

    @staticmethod
    def _key(key, **modifiers):
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetKeyCode(ord(key))
        evt.SetControlDown(bool(modifiers.get("ctrl")))
        evt.SetAltDown(bool(modifiers.get("alt")))
        evt.SetShiftDown(bool(modifiers.get("shift")))
        return evt

    def test_a_rebound_command_runs_its_action(self):
        frame = self._frame({"help.about": ["kb:control+shift+j"]})
        called = []
        frame._about = lambda: called.append("about")
        frame._on_global_char_hook(
            self._key("J", ctrl=True, shift=True))
        self.assertEqual(called, ["about"])

    def test_an_unbound_command_no_longer_answers_its_default_key(self):
        frame = self._frame({"help.about": []})
        called = []
        frame._about = lambda: called.append("about")
        self.assertFalse(frame._dispatch_gesture(
            self._key("J", ctrl=True, shift=True)))
        self.assertEqual(called, [])

    def test_an_unmodified_gesture_never_shadows_typing(self):
        frame = self._frame()
        called = []
        frame._about = lambda: called.append("about")
        self.assertFalse(frame._dispatch_gesture(self._key("J")))
        self.assertEqual(called, [])

    def test_settings_changes_are_picked_up_without_a_restart(self):
        frame = self._frame()
        frame.settings.set("gestures.bindings", {"help.about": ["kb:alt+j"]})
        frame._reload_gestures()
        called = []
        frame._about = lambda: called.append("about")
        frame._on_global_char_hook(self._key("J", alt=True))
        self.assertEqual(called, ["about"])

    def test_the_built_in_shortcut_still_works_after_a_rebind(self):
        frame = self._frame({"project.open": ["kb:control+shift+p"]})
        reached = []
        frame.new_project_btn.Unbind(wx.EVT_BUTTON)
        frame.new_project_btn.Bind(wx.EVT_BUTTON,
                                   lambda _: reached.append("button"))
        frame._on_global_char_hook(
            self._key("N", ctrl=True, shift=True))
        self.assertEqual(reached, ["button"])


class HuggingFacePanelTest(_AppMixin):
    """Settings > Hugging Face account: the token a gated model needs.

    Gemma 3 downloads from a gated repository, so without this panel the only
    advice the application could give was ``huggingface-cli login`` - which a
    person who installed the setup file does not have.  These tests hold the
    panel to what replaces it: a field, a check that asks huggingface.co whose
    token it is, and a value that survives Apply.
    """

    def _panel(self, token: str = ""):
        settings = _temp_settings()
        if token:
            settings.set(hf_access.TOKEN_SETTING, token)
            settings.save()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg, settings, dlg.hugging_face_panel

    def _inline_threads(self):
        """Run a background worker inline, so the test stays synchronous.

        The token check talks to huggingface.co, so it runs off the GUI thread
        and comes back through ``wx.CallAfter``.  Patching only *that* module's
        ``threading`` reference keeps the rest of threading alone.
        """
        class _InlineThread:
            def __init__(self, target=None, name=None, daemon=None, **_kwargs):
                self._target = target

            def start(self):
                if self._target is not None:
                    self._target()

        return (mock.patch.object(settings_module.threading, "Thread",
                                  _InlineThread), settings_module)

    def test_a_pasted_token_is_checked_and_saved_with_its_account(self):
        _dlg, settings, panel = self._panel()
        patcher, module = self._inline_threads()
        with patcher, \
                mock.patch.object(module.hf_access, "verify",
                                  return_value=(True, "anujj87")), \
                mock.patch.object(wx, "CallAfter", _inline_call_after):
            panel.token_ctrl.SetValue("hf_read_token")
            panel._on_check_token()
        self.assertIn("anujj87", panel.status.GetLabel())
        panel.apply_to_settings()
        self.assertEqual(settings.get(hf_access.TOKEN_SETTING), "hf_read_token")
        self.assertEqual(settings.get(hf_access.ACCOUNT_SETTING), "anujj87")

    def test_a_refused_token_is_reported_and_no_account_is_claimed(self):
        _dlg, settings, panel = self._panel()
        patcher, module = self._inline_threads()
        with patcher, \
                mock.patch.object(module.hf_access, "verify",
                                  return_value=(False, "Hugging Face did not "
                                                        "accept this token")), \
                mock.patch.object(wx, "CallAfter", _inline_call_after):
            panel.token_ctrl.SetValue("hf_wrong")
            panel._on_check_token()
        self.assertIn("did not accept", panel.status.GetLabel())
        panel.apply_to_settings()
        self.assertEqual(settings.get(hf_access.TOKEN_SETTING), "hf_wrong")
        self.assertEqual(settings.get(hf_access.ACCOUNT_SETTING), "")

    def test_saving_a_token_that_was_never_checked_claims_no_account(self):
        _dlg, settings, panel = self._panel()
        panel.token_ctrl.SetValue("hf_typed_by_hand")
        panel.apply_to_settings()
        self.assertEqual(settings.get(hf_access.TOKEN_SETTING), "hf_typed_by_hand")
        self.assertEqual(settings.get(hf_access.ACCOUNT_SETTING), "")

    def test_clearing_the_field_removes_the_token_and_its_account(self):
        _dlg, settings, panel = self._panel("hf_old")
        settings.set(hf_access.ACCOUNT_SETTING, "anujj87")
        panel._on_clear_token()
        panel.apply_to_settings()
        self.assertEqual(settings.get(hf_access.TOKEN_SETTING), "")
        self.assertEqual(settings.get(hf_access.ACCOUNT_SETTING), "")
        self.assertIn("cleared", panel.status.GetLabel())

    def test_show_the_token_unmasks_the_field(self):
        _dlg, _settings, panel = self._panel("hf_secret")
        self.assertTrue(panel.token_ctrl.GetWindowStyle() & wx.TE_PASSWORD)
        panel.show_cb.SetValue(True)
        panel._on_show_token()
        self.assertFalse(panel.token_ctrl.GetWindowStyle() & wx.TE_PASSWORD)
        panel.show_cb.SetValue(False)
        panel._on_show_token()
        self.assertTrue(panel.token_ctrl.GetWindowStyle() & wx.TE_PASSWORD)

    def test_a_token_from_the_environment_is_named_in_the_status(self):
        # Nothing in the field, yet gated models can download: saying where the
        # token comes from is what stops a user pasting it in twice.
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
            _dlg, _settings, panel = self._panel()
            self.assertIn("HF_TOKEN", panel.status.GetLabel())

    def test_a_stored_token_is_shown_in_the_field(self):
        _dlg, _settings, panel = self._panel("hf_saved")
        self.assertEqual(panel.token_ctrl.GetValue(), "hf_saved")
        self.assertIn("A token is configured", panel.status.GetLabel())

    def test_the_gated_models_are_listed_and_gemma_3_12b_is_the_one_shown(self):
        # There is no per-model token, so the panel offers the *models*: the
        # page whose licence has to be accepted is one button away.
        _dlg, _settings, panel = self._panel()
        repos = [row["repo"] for row in hf_access.gated_repos()]
        self.assertEqual(panel.repo_choice.GetCount(), len(repos))
        self.assertEqual(panel._selected_repo(), "google/gemma-3-12b-it")
        self.assertIn("gemma-3-12b-it",
                      panel.repo_choice.GetString(panel.repo_choice.GetSelection()))

    def test_opening_the_model_page_goes_to_that_repository(self):
        _dlg, _settings, panel = self._panel()
        panel.repo_choice.SetSelection(
            [row["repo"] for row in hf_access.gated_repos()]
            .index("kyutai/pocket-tts"))
        panel._on_repo_changed()
        opened: list = []
        with mock.patch.object(webbrowser, "open",
                               side_effect=opened.append):
            panel._on_open_model_page()
        self.assertEqual(opened, ["https://huggingface.co/kyutai/pocket-tts"])
        # What to do on the page is said here, because that is the half of the
        # job the token page itself cannot do.
        self.assertIn("Agree and access repository",
                      panel.access_status.GetLabel())

    def test_check_access_reports_what_hugging_face_answered(self):
        _dlg, _settings, panel = self._panel("hf_good")
        patcher, module = self._inline_threads()
        answer = (module.hf_access.ACCESS_REFUSED,
                  "The token works (signed in as anujj87), but that account "
                  "has not been granted google/gemma-3-12b-it yet.")
        with patcher, \
                mock.patch.object(module.hf_access, "check_access",
                                  return_value=answer) as check, \
                mock.patch.object(wx, "CallAfter", _inline_call_after):
            panel._on_check_access()
        self.assertIn("anujj87", panel.access_status.GetLabel())
        self.assertIn("not been granted", panel.access_status.GetLabel())
        # The token in the field is the one checked, off the GUI thread.
        check.assert_called_once_with("google/gemma-3-12b-it", "hf_good")
        self.assertTrue(panel.access_btn.IsEnabled())

    def test_saving_a_new_token_restarts_the_voice_lab_workers(self):
        # A worker gets HF_TOKEN once, at launch, and keeps its (possibly
        # cloning-free) Pocket TTS weights loaded - so a token pasted now has to
        # drop them, or the next clone fails exactly as before.
        _dlg, _settings, panel = self._panel("hf_old")
        with mock.patch.object(voicelab, "close_workers") as closed:
            panel.token_ctrl.SetValue("hf_new")
            panel.apply_to_settings()
        closed.assert_called_once_with()

    def test_saving_the_same_token_again_does_not_restart_anything(self):
        _dlg, _settings, panel = self._panel("hf_same")
        with mock.patch.object(voicelab, "close_workers") as closed:
            panel.apply_to_settings()
        closed.assert_not_called()

    def test_the_access_line_before_a_check_says_what_the_button_does(self):
        _dlg, _settings, panel = self._panel()
        self.assertIn("Check access", panel.access_status.GetLabel())

    def test_choosing_another_model_clears_the_previous_answer(self):
        _dlg, _settings, panel = self._panel()
        panel._access_status = "Access is in place: ..."
        panel.repo_choice.SetSelection(0)
        panel._on_repo_changed()
        self.assertNotIn("Access is in place", panel.access_status.GetLabel())


class SettingsDialogTest(_AppMixin):
    def test_categories_build_all_controls(self):
        settings = _temp_settings()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        try:
            # NVDA-style: a category list on the left, one panel per category.
            self.assertIsNotNone(dlg.cat_list)
            self.assertEqual(dlg.cat_list.GetItemCount(), 17)
            self.assertEqual(len(dlg._panels), 17)
            self.assertTrue(hasattr(dlg, "container"))
            # Every panel must have at least one child control.  They are built
            # on first visit now, so this sweeps them all explicitly rather
            # than relying on the dialog having built them up front.
            for panel in dlg.panels():
                self.assertTrue(panel.GetChildren(), f"{panel.title} has no children")
            # Only the first category is visible.
            self.assertTrue(dlg._panels[0].IsShown())
            self.assertFalse(dlg._panels[1].IsShown())
            # The second category is Appearance (theme + wallpaper), which
            # moved out of General; General itself stays first.
            self.assertEqual(dlg._panels[0].title, "General")
            self.assertEqual(dlg._panels[1].title, "Appearance")
            self.assertEqual(dlg.cat_list.GetItemText(1), "Appearance")
            # The Voice Clone category (the CPU/GPU clone engines) follows the
            # three voice/model categories, and carries the OmniVoice engines
            # as its second half: one page for every pip-installed cloning
            # engine, with OmniVoice installable right there.
            clone_category = dlg.voice_clone_category_panel
            self.assertEqual(clone_category.title, "Voice Clone")
            self.assertEqual(clone_category.omnivoice.title, "OmniVoice engines")
            self.assertIs(dlg.voice_clone_panel, clone_category.voicelab)
            self.assertIs(dlg.omnivoice_engines_panel, clone_category.omnivoice)
            titles = [cls.title for cls in dlg.CATEGORIES]
            self.assertNotIn("OmniVoice engines", titles,
                             "the OmniVoice engines category was merged away")
            self.assertEqual(len(titles), len(set(titles)),
                             "two categories share a title")
            # The Manage the STT category lists every speech-to-text model.
            stt = dlg.manage_stt_panel
            self.assertEqual(stt.title, "Manage the STT")
            # The video describer sits beside the STT category: both list the
            # models that run in a managed Python environment of their own.
            describer = dlg.manage_describe_panel
            self.assertEqual(describer.title, "Manage the video describer")
            # Variant, Size, VRAM, Fit, Status, Note.
            self.assertEqual(describer.table.GetColumnCount(), 6)
            self.assertGreater(describer.variant_combo.GetCount(), 0)
            self.assertEqual(
                describer.variant_combo.GetCount(),
                len(describe_catalog.variants(describer._selected_model())),
            )
            # Every variant row carries a size and a video-memory figure, which
            # is what lets a user pick by hardware, and a state that says
            # whether that variant's own checkpoint is here yet.
            for row in range(describer.table.GetItemCount()):
                self.assertTrue(describer.table.GetItemText(row, 1))
                self.assertTrue(describer.table.GetItemText(row, 2))
                self.assertTrue(describer.table.GetItemText(row, 4))
            # The two layers are managed by two pairs of buttons: the model's
            # environment, and one variant's checkpoint.
            self.assertEqual(describer.download_btn.GetName(),
                             "Download selected variant")
            self.assertEqual(describer.remove_variant_btn.GetName(),
                             "Remove variant download")
            self.assertIn("environment", describer.install_btn.GetLabel())
            self.assertIn("environment", describer.remove_btn.GetLabel())
            self.assertTrue(describer.status.GetLabel())
            # The Hugging Face category sits beside the describer: it is where
            # the token for a gated repository (Gemma 3) is pasted.
            hugging = dlg.hugging_face_panel
            self.assertEqual(hugging.title, "Hugging Face account")
            self.assertTrue(hugging.token_ctrl.GetWindowStyle() & wx.TE_PASSWORD)
            self.assertEqual(hugging.token_ctrl.GetValue(), "")
            self.assertTrue(hugging.status.GetLabel())
            self.assertTrue(hugging.check_btn.IsEnabled())
            gestures = dlg.input_gestures_panel
            self.assertEqual(gestures.title, "Keyboard gestures")
            self.assertEqual(
                dlg._panels.index(gestures), dlg._panels.index(hugging) + 1,
                "the Keyboard gestures category follows Hugging Face",
            )
            self.assertTrue(gestures.tree.GetRootItem().IsOk())
            self.assertTrue(gestures.selected_command())
            self.assertGreater(gestures.gesture_list.GetCount(), 0)
            self.assertEqual(stt.table.GetItemCount(),
                             len(stt_catalog.model_ids()))
            # Status + Model + License + Languages + Fit + Notes.
            self.assertEqual(stt.table.GetColumnCount(), 6)
            self.assertIn(stt.table.GetItemText(0, 0),
                          ("Installed", "Not installed"))
            self.assertTrue(stt.status.GetLabel())
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
            # The built-in voices of the engines are defined, even when
            # the engines are not installed yet.
            self.assertGreater(voice_lab.count_builtin_voices("chatterbox"), 20)
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
            self.assertEqual(general.recordings_ctrl.GetName(), "Project save location")
            self.assertEqual(general.models_ctrl.GetName(), "Model files location")
            # Each folder row names its own Browse button (a screen reader must
            # not offer two identical "Browse..." buttons in one panel).
            browse_labels = {
                child.GetLabel() for child in general.GetChildren()
                if isinstance(child, wx.Button)
            }
            self.assertIn("Browse project location...", browse_labels)
            self.assertIn("Browse model location...", browse_labels)
            self.assertNotIn("Browse...", browse_labels)
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

            # The OmniVoice engines now live on the Voice Clone page and are
            # installed from there, beside the Voice Lab engines; the Compute
            # category keeps only the GPU (CUDA) runtime.
            engines = dlg.omnivoice_engines_panel
            self.assertEqual(engines.omnivoice_install_btn.GetName(),
                             "Install OmniVoice dependency")
            self.assertEqual(engines.server_install_btn.GetName(),
                             "Install OmniVoice Server")
            self.assertTrue(hasattr(engines, "dependency_status"))
            self.assertTrue(hasattr(engines, "server_status"))
            self.assertFalse(hasattr(dlg.compute_panel, "omnivoice_install_btn"))
            self.assertFalse(hasattr(dlg.compute_panel, "_on_server_install"))
        finally:
            dlg.Destroy()

    def test_server_category_opens_without_waiting_for_the_probe(self):
        """Opening a category must never block on a network round trip.

        The server status probe used to run inline on activation, which made
        the Settings dialog freeze for seconds on an unreachable host.
        """
        settings = _temp_settings()
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        try:
            index = next(i for i, cls in enumerate(dlg.CATEGORIES)
                     if cls.title == "OmniVoice Server")
            # The panel is built *by* the switch, so the slow probe is patched
            # on the class: the panel the category builds must already carry it.
            panel_cls = dlg.CATEGORIES[index]
            probed = threading.Event()
            probed_args = []

            def slow_probe(_self, host, port, epoch):
                probed_args.append((host, port, epoch))
                probed.set()
                time.sleep(1.5)  # stand in for a slow health check

            with mock.patch.object(panel_cls, "_probe_status", slow_probe):
                start = time.perf_counter()
                dlg._show_category(index)
                elapsed = time.perf_counter() - start

            self.assertLess(elapsed, 0.5,
                            f"switching category blocked for {elapsed:.2f}s")
            panel = dlg.ensure_panel(index)
            self.assertEqual(panel.status_label.GetLabel(),
                             "Server status: Checking...")
            self.assertTrue(probed.wait(2.0), "status probe never started")
            self.assertTrue(probed_args[0][0] or probed_args[0][1])
        finally:
            dlg.Destroy()


    def test_general_panel_applies_path_settings(self):
        settings = _temp_settings()
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
        settings = _temp_settings()
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
        own voices (F5-TTS ships two reference voices, Chatterbox one per
        model), which would drown out the clones these tests are about.  The probe
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

    def test_the_panel_offers_a_speed_measurement(self):
        """The Device box has the 'Measure speed...' control and its note."""
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.addCleanup(voicelab.clear_cpu_rtf)
        self.assertTrue(hasattr(panel, "benchmark_btn"))
        self.assertTrue(hasattr(panel, "benchmark_note"))
        self.assertEqual(panel.benchmark_btn.GetName(), "Measure engine speed")
        # Nothing is measured yet: the note explains what the button does and
        # the button follows installation.
        self.assertIn("No speed measurement", panel.benchmark_note.GetLabel())
        with mock.patch.object(voicelab, "is_installed", return_value=True), \
                mock.patch.object(voicelab, "installed_version",
                                  return_value="1.0"):
            panel._refresh_engine_status()
            self.assertTrue(panel.benchmark_btn.IsEnabled())
        with mock.patch.object(voicelab, "is_installed", return_value=False):
            panel._refresh_engine_status()
            self.assertFalse(panel.benchmark_btn.IsEnabled())

    def test_a_measured_fast_cpu_is_offered_for_a_gpu_only_engine(self):
        """The measurement, not the model's size, decides the device list."""
        from ai_voice_studio.voicelab import benchmark

        settings = Settings(path=os.path.join(self.tmp.name, "settings.json"))
        settings.set(benchmark.settings_key("chatterbox"),
                     {"cpu": {"rtf": 1.4, "seconds": 4.2,
                               "audio_seconds": 3.0, "sample_rate": 24000}})
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        self.addCleanup(voicelab.clear_cpu_rtf)
        panel = dlg.voice_clone_panel
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertTrue(panel.select_engine("chatterbox"))
            values = [panel.device_combo.GetClientData(i)
                      for i in range(panel.device_combo.GetCount())]
        self.assertEqual(values[0], "cuda")
        self.assertIn("cpu", values)
        self.assertIn("1.40x real time", panel.benchmark_note.GetLabel())
        self.assertIn("1.40x real time", panel.device_note.GetLabel())

    def test_a_finished_measurement_updates_the_panel(self):
        """``_benchmark_done`` refreshes the device list and the note."""

        dlg = self._dialog()
        self.addCleanup(voicelab.clear_cpu_rtf)
        panel = dlg.voice_clone_panel
        summary = {
            "engine": "f5tts",
            "device": "cuda",
            "probes": {
                "cuda": {"ok": True, "rtf": 0.3, "seconds": 1.0,
                         "audio_seconds": 3.3, "sample_rate": 24000},
                "cpu": {"ok": True, "rtf": 4.5, "seconds": 15.0,
                        "audio_seconds": 3.3, "sample_rate": 24000},
            },
        }
        voicelab.set_cpu_rtf("f5tts", 4.5)
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch.object(panel, "_selected_engine_id",
                                  return_value="f5tts"), \
                mock.patch.object(wx, "MessageBox", return_value=wx.ID_OK):
            panel._benchmark_done("f5tts", None, summary)
        values = [panel.device_combo.GetClientData(i)
                  for i in range(panel.device_combo.GetCount())]
        # Measured above the limit with a card present: no CPU entry.
        self.assertNotIn("cpu", values)
        self.assertIn("4.50x real time", panel.benchmark_note.GetLabel())

    def test_builtin_voices_are_offered_once_the_engine_is_installed(self):
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        # Whether a package happens to be installed in the managed environment
        # depends on the machine, so both states are driven explicitly here.
        with mock.patch.object(voicelab, "is_installed", return_value=False):
            self.assertTrue(panel.select_engine("f5tts"))
            self.assertEqual(panel.builtin_combo.GetCount(), 0)
            self.assertEqual(panel.voices_list.GetCount(), 0)
            self.assertIn("Not installed", panel.engine_status.GetLabel())
        with mock.patch.object(voicelab, "is_installed", return_value=True), \
                mock.patch.object(voicelab, "installed_version", return_value="4.4.0"):
            panel._refresh_engine_status()
            panel._refresh_builtin_voices()
            panel._refresh_voices()
        self.assertEqual(panel.builtin_combo.GetCount(), 2)
        self.assertEqual(panel.voices_list.GetCount(), 2)
        self.assertIn("Installed (version 4.4.0)",
                      panel.engine_status.GetLabel())
        # ... and every voice row says which engine, variant and device.
        self.assertIn("built-in", panel.voices_list.GetString(0))
        self.assertIn("CPU", panel.voices_list.GetString(0))

    def test_a_cloned_voice_reaches_every_category(self):
        """A clone made here is offered by every category that lists voices.

        Both a long-standing engine (F5-TTS) and the newest one (KokoClone)
        are checked: the categories find their voices through the engine
        registry, not through a list of ids kept in the GUI.
        """
        sample = os.path.join(self.tmp.name, "reference.wav")
        with open(sample, "wb") as fh:
            fh.write(b"RIFFfake")
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        for engine_id, name in (("f5tts", "My Clone"),
                                ("kokoclone", "Koko Clone")):
            self.assertTrue(panel.select_engine(engine_id))
            panel.name_ctrl.SetValue(name)
            panel.sample_ctrl.SetValue(sample)
            panel.ref_text_ctrl.SetValue("some words")
            panel._on_create(None)
        # This test is about the clones: on a machine where these engines
        # really are installed their built-in voices would be listed as well.
        with self._without_builtin_voices(panel):
            panel.select_engine("f5tts")
            labels = [panel.voices_list.GetString(i)
                      for i in range(panel.voices_list.GetCount())]
            self.assertEqual(len(labels), 1, labels)
        self.assertIn("My Clone", labels[0])
        self.assertIn("cloned", labels[0])
        # The other categories pick the clones up without knowing about the
        # Voice Clone category at all.
        dlg.available_panel.refresh()
        cloned = sorted(v["voice_name"] for v in dlg.available_panel._voices
                        if v.get("cloned"))
        self.assertEqual(cloned, ["Koko Clone", "My Clone"])
        for category in (dlg.recording_panel, dlg.punctuation_panel):
            category.on_activated()
            engines = [category.tts_combo.GetClientData(i)
                       for i in range(category.tts_combo.GetCount())]
            self.assertIn("f5tts", engines)
            self.assertIn("kokoclone", engines)

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
                mock.patch("wx.CallAfter", side_effect=_inline_call_after):
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

    def test_a_clone_keeps_the_language_it_was_created_for(self):
        """KokoClone speaks seven languages; the clone says which one."""
        sample = os.path.join(self.tmp.name, "reference.wav")
        with open(sample, "wb") as fh:
            fh.write(b"RIFFfake")
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.assertTrue(panel.select_engine("kokoclone"))
        codes = [panel.language_combo.GetClientData(i)
                 for i in range(panel.language_combo.GetCount())]
        self.assertEqual(codes,
                         [lang["code"] for lang in
                          voice_lab.engine("kokoclone")["languages"]])
        self.assertEqual(panel.language_combo.GetSelection(), 0)
        panel.name_ctrl.SetValue("Zh Voice")
        panel.sample_ctrl.SetValue(sample)
        for index in range(panel.language_combo.GetCount()):
            if panel.language_combo.GetClientData(index) == "zh":
                panel.language_combo.SetSelection(index)
                break
        panel._on_create(None)
        entry = next(v for v in voicelab.clone_voices(dlg.store, "kokoclone")
                     if v.get("name") == "Zh Voice")
        self.assertEqual(entry["language"], "zh")

    def test_kokoclone_is_installed_into_and_removed_from_its_own_env(self):
        """KokoClone gets its own environment and never touches the shared
        addon environment (which is where OmniVoice keeps its packages).

        The other engines keep the behaviour they have always had: installed
        into their own environment, and swept out of the old shared one too.
        """
        dlg = self._dialog()
        panel = dlg.voice_clone_panel
        self.assertTrue(panel.select_engine("kokoclone"))

        own, shared = [], []

        class FakeRuntime:
            def __init__(self, name):
                self.name = name

            def ensure_pip(self):
                pass

            def pip_install(self, packages, progress=None, index_url=None):
                (own if self.name == "own" else shared).append(
                    (list(packages), index_url))
                return {"ok": True, "output": "", "error": ""}

            def pip_uninstall(self, packages):
                (own if self.name == "own" else shared).append(
                    ("uninstall", list(packages)))
                return {"ok": True, "output": "", "error": ""}

            def remove_env(self):
                own.append("removed")

        def fake_get_runtime(engine_id=None):
            return FakeRuntime("own" if engine_id == "kokoclone" else "shared")

        done = {}
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        side_effect=fake_get_runtime), \
                mock.patch.object(panel, "_update_progress"), \
                mock.patch.object(panel, "_install_done",
                                  side_effect=lambda *a: done.update(args=a)), \
                mock.patch("wx.CallAfter", side_effect=_inline_call_after):
            panel._install_job("kokoclone")
            panel._remove_job("kokoclone", voicelab.packages_for("kokoclone"))

        self.assertTrue(own, "the install never reached its own environment")
        self.assertEqual(shared, [],
                         "KokoClone touched the shared/OmniVoice environment")
        # Removing it takes the whole environment away - models included - and
        # never asks the shared environment to uninstall anything.
        self.assertEqual(own[-1], "removed")

        # An older engine still uninstalls its packages from both places.
        self.assertTrue(panel.select_engine("f5tts"))

        def fake_other(engine_id=None):
            return FakeRuntime("own" if engine_id == "f5tts" else "shared")

        own.clear()
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        side_effect=fake_other), \
                mock.patch.object(panel, "_install_done"), \
                mock.patch("wx.CallAfter", side_effect=_inline_call_after):
            panel._remove_job("f5tts", voice_lab.packages("f5tts"))
        self.assertEqual([c[0] for c in own], ["uninstall"])
        self.assertEqual([c[0] for c in shared], ["uninstall"])
        self.assertEqual(len(shared), 1)


class MainFrameTest(_AppMixin):
    def test_frame_builds_all_ui(self):
        frame = MainFrame(settings=_temp_settings(), store=self._empty_store())
        try:
            self.assertIsNotNone(frame.GetMenuBar())
            self.assertTrue(hasattr(frame, "recent_list"))
            self.assertTrue(hasattr(frame, "remove_btn"), "Remove Project button missing")
            self.assertTrue(frame.recent_list.GetParent().GetChildren())
            self.assertEqual(frame.GetStatusBar().GetStatusText(), "Ready.")
        finally:
            frame.Destroy()

    def test_edit_menu_has_project_actions(self):
        frame = MainFrame(settings=_temp_settings(), store=self._empty_store())
        try:
            menubar = frame.GetMenuBar()
            edit_index = menubar.FindMenu("Edit")
            self.assertGreaterEqual(edit_index, 0, "Edit menu missing")
            labels = [item.GetItemLabelText()
                      for item in menubar.GetMenu(edit_index).GetMenuItems()]
            # A recording project (the default) speaks of recordings.
            for expected in ("Resume the project", "Restart Project...",
                             "Restart All Recording",
                             "Start Selected Recording...",
                             "Remove Project...", "Show project folder"):
                self.assertIn(expected, labels)
            # Help menu has Read Me + User Guide + About.
            help_index = menubar.FindMenu("Help")
            self.assertGreaterEqual(help_index, 0, "Help menu missing")
            help_labels = [item.GetItemLabelText()
                           for item in menubar.GetMenu(help_index).GetMenuItems()]
            for expected in ("Read Me", "User Guide", "Third-Party Licences",
                             "What's New", "About AI Voice Studio"):
                self.assertIn(expected, help_labels)
            # Ctrl+Shift+N is the welcome panel's Create New Project button's
            # gesture, owned by that button's own click path - so the File
            # menu item must not advertise the same key a second time.
            new_item = next(i for i in menubar.GetMenu(0).GetMenuItems()
                        if i.GetItemLabelText() == "New Project")
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
        self.assertEqual(labels, ["Resume the project", "Restart Project...",
                                  "Restart All Recording",
                                  "Start Selected Recording...",
                                  "Remove Project...", "Show project folder"])

    def test_the_edit_menu_follows_the_selected_project_base(self):
        """A transcription project has *files*, a recording project recordings."""
        frame, tmp = self._project_frame()
        settings = frame.settings
        settings.set("recent_projects", [
            {"name": "Speech", "path": tmp, "opened": ""},
        ])
        # A recording project first.
        self.assertEqual(frame._selected_project_base(), "tts")
        frame._refresh_edit_menu()
        labels = [item.GetItemLabelText()
                  for item in frame._edit_menu.GetMenuItems()]
        self.assertIn("Restart All Recording", labels)
        self.assertNotIn("Restart All Files", labels)
        # Now the same project, but a speech-to-text one.
        data = project.load_project(tmp)
        data["project_type"] = "stt_transcription"
        project.save_project(tmp, data)
        self.assertEqual(frame._selected_project_base(), "stt")
        frame._refresh_edit_menu()
        labels = [item.GetItemLabelText()
                  for item in frame._edit_menu.GetMenuItems()]
        self.assertIn("Resume the project", labels)
        self.assertIn("Restart All Files", labels)
        self.assertIn("Start Selected Files...", labels)
        self.assertNotIn("Restart All Recording", labels)

    def test_the_context_menu_follows_the_selected_project_base(self):
        """The right-click menu uses the same words as the Edit menu."""
        frame, tmp = self._project_frame()
        data = project.load_project(tmp)
        data["project_type"] = "stt_transcription"
        project.save_project(tmp, data)
        labels = []
        with mock.patch.object(
            frame.recent_list, "PopupMenu",
            side_effect=lambda menu: labels.extend(
                i.GetItemLabelText() for i in menu.GetMenuItems()
                if not i.IsSeparator()
            ),
        ):
            evt = mock.MagicMock()
            evt.GetPosition.return_value = wx.Point(4, 4)
            frame._on_recent_context_menu(evt)
        self.assertIn("Restart All Files", labels)
        self.assertIn("Start Selected Files...", labels)
        self.assertNotIn("Restart All Recording", labels)

    def test_the_context_menu_from_the_keyboard_keeps_the_selection(self):
        """The application-menu key has no position: the selected row wins."""
        frame, tmp = self._project_frame()
        evt = mock.MagicMock()
        evt.GetPosition.return_value = wx.DefaultPosition
        seen = []
        with mock.patch.object(
            frame.recent_list, "PopupMenu", side_effect=seen.append,
        ):
            frame._on_recent_context_menu(evt)
        self.assertEqual(frame.recent_list.GetSelection(), 0)
        self.assertEqual(frame._selected_recent()["path"], tmp)

    def test_the_edit_menu_action_opens_the_folder(self):
        """The Edit-menu item reaches the same handler as the popup item."""
        frame, tmp = self._project_frame()
        menubar = frame.GetMenuBar()
        edit_menu = menubar.GetMenu(menubar.FindMenu("Edit"))
        item_id = next(i.GetId() for i in edit_menu.GetMenuItems()
                   if i.GetItemLabelText() == "Show project folder")
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

    def test_the_recording_picker_names_the_project_base(self):
        from ai_voice_studio.gui.main_frame import _StartRecordingDialog

        dlg = _StartRecordingDialog(self.frame, self._entries())
        self.addCleanup(dlg.Destroy)
        self.assertEqual(dlg.base_combo.GetCount(), 1)
        self.assertEqual(dlg.base_combo.GetString(0), "Text to speech")
        self.assertEqual(dlg.base_combo.GetSelection(), 0)

    def test_restart_project_confirmation_mentions_transcribe_files(self):
        frame, _tmp = self._project_frame([
            {"index": 1, "title": "chapter 1", "text": "Text 1."},
        ])
        with mock.patch.object(wx, "MessageBox", return_value=wx.NO) as box:
            frame._restart_project()
        text = box.call_args.args[0]
        self.assertIn(
            "All recordings or transcribe files and project settings will "
            "be deleted", text,
        )
        self.assertIn("audio creation or transcribe options again", text)

    def test_restart_all_confirmation_uses_the_files_wording(self):
        frame, _tmp = self._project_frame([
            {"index": 1, "title": "chapter 1", "text": "Text 1."},
        ])
        with mock.patch.object(wx, "MessageBox", return_value=wx.NO) as box:
            frame._restart_all()
        self.assertEqual(box.call_args.args[1], "Restart all files")
        text = box.call_args.args[0]
        self.assertIn("Restart all recording or transcribe files for "
                      "'Selected project'?", text)
        self.assertIn("recorded or transcribed again from the beginning", text)


class StartSelectedFilesDialogTest(_AppMixin):
    """Edit > Start Selected Files for a speech-to-text project."""

    def _entries(self):
        return [
            {"position": 0, "title": "testingPad.wav",
             "source": "C:\\in\\testingPad.wav", "file": "testingPad.wav"},
            {"position": 1, "title": "notes.wav",
             "source": "C:\\in\\notes.wav", "file": "notes.wav"},
            {"position": 2, "title": "empty", "source": "", "file": None},
        ]

    def _dialog(self, entries=None):
        from ai_voice_studio.gui.main_frame import _StartTranscribeDialog

        dlg = _StartTranscribeDialog(
            self.frame, self._entries() if entries is None else entries)
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_the_dialog_states_the_base_and_starts_on_transcribe_files(self):
        dlg = self._dialog()
        self.assertEqual(dlg.GetTitle(), "Start selected files")
        self.assertEqual(dlg.base_combo.GetCount(), 1)
        self.assertEqual(dlg.base_combo.GetString(0), "Speech to text")
        self.assertTrue(dlg.file_radio.GetValue())
        self.assertFalse(dlg.break_radio.GetValue())
        self.assertTrue(dlg.only_radio.GetValue())
        self.assertFalse(dlg.all_radio.GetValue())
        self.assertEqual(dlg.choice_label.GetLabel(),
                         "Choose which transcribed file to transcribe again:")
        self.assertEqual(dlg.combo.GetName(),
                         "Choose which transcribed file to transcribe again")
        self.assertEqual(
            [dlg.combo.GetString(i) for i in range(dlg.combo.GetCount())],
            ["testingPad.wav", "notes.wav"],
        )
        self.assertEqual(dlg.start_index(), 0)
        self.assertTrue(dlg.only_selected())

    def test_by_file_break_lists_every_break_and_the_scope(self):
        dlg = self._dialog()
        dlg.break_radio.SetValue(True)
        dlg._on_mode(None)
        self.assertEqual(dlg.choice_label.GetLabel(),
                         "Choose which file to transcribe by file break:")
        self.assertEqual(
            [dlg.combo.GetString(i) for i in range(dlg.combo.GetCount())],
            ["testingPad.wav", "notes.wav", "empty"],
        )
        dlg.combo.SetSelection(2)
        self.assertEqual(dlg.start_index(), 2)
        dlg.all_radio.SetValue(True)
        self.assertFalse(dlg.only_selected())

    def test_the_entries_come_from_the_project_segments(self):
        from ai_voice_studio.gui.main_frame import _transcribe_entries

        entries = _transcribe_entries({"segments": [
            {"index": 1, "title": "testingPad.wav",
             "source": "D:\\in\\testingPad.wav"},
            {"index": 2, "source": "D:\\in\\silent.mp3"},
        ]})
        self.assertEqual([e["position"] for e in entries], [0, 1])
        self.assertEqual(entries[0]["file"], "testingPad.wav")
        self.assertEqual(entries[1]["title"], "silent.mp3")
        self.assertEqual(entries[1]["file"], "silent.mp3")

    def test_the_action_opens_the_transcribe_window_for_the_chosen_files(self):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Speech project", "first.wav", "per_source",
            [{"index": 1, "title": "one.wav", "text": ""},
             {"index": 2, "title": "two.wav", "text": ""}],
            project_type=PROJECT_TYPE_STT,
        )
        data = project.load_project(tmp)
        data["segments"][0]["source"] = os.path.join(tmp, "one.wav")
        data["segments"][1]["source"] = os.path.join(tmp, "two.wav")
        project.save_project(tmp, data)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        settings.add_recent_project("Speech project", tmp)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        frame.recent_list.SetSelection(0)

        opened = {}
        fake = mock.MagicMock()
        fake.return_value.ShowModal.return_value = wx.ID_OK
        fake.return_value.start_index.return_value = 1
        fake.return_value.only_selected.return_value = False
        with mock.patch.object(
            frame, "_open_transcribe",
            lambda *args, **kwargs: opened.update(args=args, kwargs=kwargs),
        ), mock.patch("ai_voice_studio.gui.main_frame._StartTranscribeDialog", fake):
            frame._restart_selected()

        self.assertEqual(opened["args"][0], tmp)
        self.assertEqual(opened["args"][1], [os.path.join(tmp, "two.wav")])

    def test_only_the_selected_file_is_transcribed(self):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Speech project", "first.wav", "per_source",
            [{"index": 1, "title": "one.wav", "text": ""},
             {"index": 2, "title": "two.wav", "text": ""}],
            project_type=PROJECT_TYPE_STT,
        )
        data = project.load_project(tmp)
        data["segments"][0]["source"] = os.path.join(tmp, "one.wav")
        data["segments"][1]["source"] = os.path.join(tmp, "two.wav")
        project.save_project(tmp, data)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        settings.add_recent_project("Speech project", tmp)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        frame.recent_list.SetSelection(0)

        opened = {}
        fake = mock.MagicMock()
        fake.return_value.ShowModal.return_value = wx.ID_OK
        fake.return_value.start_index.return_value = 0
        fake.return_value.only_selected.return_value = True
        with mock.patch.object(
            frame, "_open_transcribe",
            lambda *args, **kwargs: opened.update(args=args, kwargs=kwargs),
        ), mock.patch("ai_voice_studio.gui.main_frame._StartTranscribeDialog", fake):
            frame._restart_selected()

        self.assertEqual(opened["args"][1], [os.path.join(tmp, "one.wav")])


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
            # Read back while the card is still there: saved_choice()
            # re-validates the stored value against what this machine offers,
            # so a stored "cuda" on a machine without the GPU deliberately
            # reports "cpu" - that is the fallback, not the memory.
            self.assertEqual(
                compute_choice.saved_choice(settings, "punctuation"), "cuda"
            )
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
                mock.patch.object(compute, "has_nvidia_gpu", return_value=True), \
                mock.patch("ai_voice_studio.voicelab.has_cuda",
                           return_value=True):
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
            dlg = RecordingDialog(self.frame, tmp, _temp_settings(), self._empty_store())
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
        if not voice_lab.is_engine(dlg._selected_tts_id()):
            self.assertFalse(dlg.engine_options_btn.IsShown())

    def test_kokoclone_is_offered_in_the_recording_window(self):
        """The record dialog lists it exactly like the other cloners."""
        _tmp, dlg, _settings = self._dialog_with_engine("kokoclone")
        engines = [dlg.tts_combo.GetClientData(i)
                   for i in range(dlg.tts_combo.GetCount())]
        self.assertIn("kokoclone", engines)
        self.assertEqual(dlg._selected_tts_id(), "kokoclone")
        # Its device row offers the CPU (and the GPU when one is detected),
        # like every Voice Lab engine.
        values = [dlg.compute_combo.GetClientData(i)
                  for i in range(dlg.compute_combo.GetCount())]
        self.assertEqual(values[0], "cpu")
        self.assertTrue(dlg.engine_options_btn.IsShown())

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


class _GpuOnlyEnginesTest(_AppMixin):
    """Chatterbox and Qwen3-TTS reach every place a voice can be chosen.

    The two engines are installed nowhere on a test machine, so their voices
    are handed to the panels exactly as ``builtin_voice_entries`` would once
    they were - which is what every one of these lists is built from.
    """

    _ENGINES = ("chatterbox", "qwen3tts")

    def setUp(self):
        self.entries = [
            voice for engine_id in self._ENGINES
            for voice in voice_lab.builtin_voices(engine_id)
        ]
        patcher = mock.patch.object(
            voicelab, "builtin_voice_entries",
            lambda engine_id=None: [
                v for v in self.entries
                if engine_id is None or v["engine"] == engine_id
            ],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _ids(combo):
        return [combo.GetClientData(i) for i in range(combo.GetCount())]

    def _settings(self):
        settings = Settings(path=os.path.join(tempfile.mkdtemp(), "settings.json"))
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_every_settings_category_offers_both_engines(self):
        dlg = self._settings()
        for accessor in ("available_panel", "voice_clone_panel",
                         "recording_panel", "punctuation_panel"):
            panel = getattr(dlg, accessor)
            if accessor == "voice_clone_panel":
                ids = self._ids(panel.engine_combo)
            else:
                ids = self._ids(panel.tts_combo)
            for engine_id in self._ENGINES:
                self.assertIn(engine_id, ids, (accessor, ids))

    def test_the_voice_clone_category_offers_a_gpu_only_device_choice(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False), \
                mock.patch.object(voicelab, "is_installed", return_value=True):
            dlg = self._settings()
            panel = dlg.voice_clone_panel
            with mock.patch.object(panel, "_probe_builtin_availability"):
                for engine_id in self._ENGINES:
                    self.assertTrue(panel.select_engine(engine_id), engine_id)
                    # The CPU is not offered for an engine that cannot use it.
                    self.assertEqual(self._ids(panel.device_combo), ["cuda"],
                                     engine_id)
                    note = panel.device_note.GetLabel()
                    self.assertIn("GPU-only", note)
                    self.assertIn("No NVIDIA GPU", note)
                    self.assertGreater(panel.builtin_combo.GetCount(), 0,
                                       engine_id)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertTrue(panel.select_engine("chatterbox"))
            self.assertEqual(self._ids(panel.device_combo), ["cuda"])
            # Switching back to a CPU-first engine restores the CPU entry.
            self.assertTrue(panel.select_engine("pocket_tts"))
            self.assertEqual(self._ids(panel.device_combo)[0], "cpu")

    def test_the_recording_window_offers_both_engines(self):
        tmp = tempfile.mkdtemp(prefix="aivs_smoke_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "GPU only", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        dlg = RecordingDialog(self.frame, tmp, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            dlg._rebuild_voice_choices()
            for engine_id in self._ENGINES:
                self.assertIn(engine_id, self._ids(dlg.tts_combo), engine_id)
                for index in range(dlg.tts_combo.GetCount()):
                    if dlg.tts_combo.GetClientData(index) != engine_id:
                        continue
                    dlg.tts_combo.SetSelection(index)
                    dlg._on_tts(None)
                    break
                # The compute row shows the only device the engine has, and a
                # voice is selectable, so the project can actually record.
                self.assertEqual(self._ids(dlg.compute_combo), ["cuda"],
                                 engine_id)
                self.assertEqual(dlg._selected_compute(), "cuda", engine_id)
                self.assertGreater(dlg.voice_combo.GetCount(), 0, engine_id)

    def test_the_preview_provider_is_the_gpu_for_a_gpu_only_engine(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            for engine_id in self._ENGINES:
                self.assertEqual(
                    compute_choice.provider_for_preview("cpu", engine_id),
                    "cuda", engine_id,
                )


class PunctuationMarkupTest(_AppMixin):
    """The Punctuation category leaves an engine's own markup alone.

    Kitten TTS 2.0 reads ``[emotion]`` tags, ``<event>`` tags and
    ``(((emphasis)))`` spans as director's notes, so the mode chosen here - the
    spoken-word expansion of "All", the symbol stripping of "None" - must not
    reach them.  The example line shows what the *selected voice's* engine is
    about to be handed, so it follows the voice cascade as well as the mode.
    """

    _TEXT = ('My name is "Anuj". [joyful] Truly? <laugh> I can (((hardly))) '
             'believe it, really.')

    def setUp(self):
        self.entries = [
            voice
            for engine_id in ("kitten_tts_2", "f5tts")
            for voice in voice_lab.builtin_voices(engine_id)
        ]
        patcher = mock.patch.object(
            voicelab, "builtin_voice_entries",
            lambda engine_id=None: [
                v for v in self.entries
                if engine_id is None or v["engine"] == engine_id
            ],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _panel(self, engine_id: str = "kitten_tts_2"):
        dlg = SettingsDialog(self.frame, _temp_settings(), self._empty_store())
        self.addCleanup(dlg.Destroy)
        panel = dlg.punctuation_panel
        panel.on_activated()
        ids = [panel.tts_combo.GetClientData(i)
               for i in range(panel.tts_combo.GetCount())]
        self.assertIn(engine_id, ids)
        panel.tts_combo.SetSelection(ids.index(engine_id))
        panel._on_tts(None)
        panel.example_text.SetValue(self._TEXT)
        panel._update_example()
        return panel

    @staticmethod
    def _modes(panel):
        return [panel.punct_combo.GetClientData(i)
                for i in range(panel.punct_combo.GetCount())]

    def _example_for(self, panel, mode: str) -> str:
        modes = self._modes(panel)
        panel.punct_combo.SetSelection(modes.index(mode))
        panel._update_example()
        return panel.example_out.GetLabel()

    def test_a_markup_engine_keeps_its_tags_in_every_mode(self):
        panel = self._panel("kitten_tts_2")
        for mode in self._modes(panel):
            with self.subTest(mode=mode):
                out = self._example_for(panel, mode)
                for span in ("[joyful]", "<laugh>", "(((hardly)))"):
                    self.assertIn(span, out)

    def test_the_mode_still_expands_the_rest_of_the_example(self):
        panel = self._panel("kitten_tts_2")
        out = self._example_for(panel, "all")
        self.assertIn("quote", out)
        self.assertIn("comma", out)
        self.assertNotIn('"', out)

    def test_another_engine_has_its_punctuation_expanded_as_before(self):
        panel = self._panel("f5tts")
        out = self._example_for(panel, "all")
        self.assertIn("left bracket", out)
        self.assertNotIn("[joyful]", out)

    def test_the_example_follows_the_voice_cascade(self):
        """Switching the engine re-reads the example, mode untouched."""
        panel = self._panel("kitten_tts_2")
        self.assertIn("[joyful]", self._example_for(panel, "all"))
        ids = [panel.tts_combo.GetClientData(i)
               for i in range(panel.tts_combo.GetCount())]
        panel.tts_combo.SetSelection(ids.index("f5tts"))
        panel._on_tts(None)
        self.assertNotIn("[joyful]", panel.example_out.GetLabel())

    def test_the_category_says_which_tags_are_kept(self):
        panel = self._panel("kitten_tts_2")
        labels = [child.GetLabel() for child in panel.GetChildren()
                  if isinstance(child, wx.StaticText)]
        self.assertTrue(any("Expression tags" in label for label in labels),
                        labels)


class STTTranscribeDialogDeviceTest(_AppMixin):
    """The transcribe dialog offers both devices, except for a GPU-only model.

    The Compute combo labels have to be truthful: a model that runs on a CPU
    gets both the CPU and the GPU, while a model that cannot run without a GPU
    says so and blocks Start on a machine that has no card.
    """

    def _dialog(self, model_id="whisper_large_v3", variant=""):
        tmp = tempfile.mkdtemp(prefix="aivs_stt_dlg_")
        self.addCleanup(shutil.rmtree, tmp, True)
        stt = {"model": model_id, "sources": [os.path.join(tmp, "a.wav")]}
        if variant:
            stt["variant"] = variant
        project.create_project(
            tmp, "STT", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
            project_type=PROJECT_TYPE_STT, stt_settings=stt,
        )
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        patcher = mock.patch.object(stt_engine, "gpu_runtime_missing",
                                    return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        dlg = STTTranscribeDialog(self.frame, tmp, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg

    @staticmethod
    def _values(dlg):
        combo = dlg.compute_combo
        return [combo.GetClientData(i) for i in range(combo.GetCount())]

    def test_a_cpu_capable_model_offers_both_devices(self):
        with mock.patch.object(stt_engine, "cuda_available", return_value=True):
            dlg = self._dialog("whisper_large_v3")
            self.assertEqual(self._values(dlg), ["cuda", "cpu"])
            self.assertTrue(dlg.start_btn.IsEnabled())

    def test_the_gpu_only_variant_offers_the_gpu_alone(self):
        with mock.patch.object(stt_engine, "cuda_available", return_value=True):
            dlg = self._dialog("canary_qwen", "2.5b")
            self.assertEqual(self._values(dlg), ["cuda"])
            # The CPU-friendly 1B variant of the same model offers both.
            dlg.variant_combo.SetSelection(
                [dlg.variant_combo.GetClientData(i)
                 for i in range(dlg.variant_combo.GetCount())].index("1b"))
            dlg._on_variant()
            self.assertEqual(self._values(dlg), ["cpu", "cuda"])

    def test_a_gpu_only_model_blocks_start_without_a_card(self):
        with mock.patch.object(stt_engine, "cuda_available", return_value=False):
            dlg = self._dialog("canary_qwen", "2.5b")
            self.assertEqual(self._values(dlg), ["cuda"])
            self.assertFalse(dlg.start_btn.IsEnabled())
            self.assertIn("NVIDIA GPU", dlg.status.GetLabel())

    def test_no_card_still_offers_the_cpu_for_a_cpu_capable_model(self):
        with mock.patch.object(stt_engine, "cuda_available", return_value=False):
            dlg = self._dialog("whisper_large_v3")
            self.assertEqual(self._values(dlg), ["cpu"])
            self.assertTrue(dlg.start_btn.IsEnabled())


class ManageSTTRepairTest(_AppMixin):
    """A model installed before its CUDA wheel was known can be repaired.

    Whisper Large V3's environment is missing the cuBLAS runtime CTranslate2
    needs, so its GPU run silently fell back to the CPU.  The Settings table
    has to say so and turn its Install button into a repair.
    """

    def _panel(self):
        settings = Settings(path=os.path.join(tempfile.mkdtemp(), "settings.json"))
        dlg = SettingsDialog(self.frame, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg.manage_stt_panel

    def test_a_missing_cuda_runtime_turns_install_into_repair(self):
        panel = self._panel()
        with mock.patch.object(stt_engine, "is_installed", return_value=True), \
                mock.patch.object(stt_engine, "env_exists", return_value=True), \
                mock.patch.object(stt_engine, "gpu_runtime_missing",
                                  return_value=["nvidia-cublas-cu12"]):
            panel._show_status()
            self.assertTrue(panel.install_btn.IsEnabled())
            self.assertIn("Repair", panel.install_btn.GetLabel())
            self.assertIn("GPU runtime", panel.status.GetLabel())

    def test_an_installed_complete_model_offers_no_repair(self):
        panel = self._panel()
        with mock.patch.object(stt_engine, "is_installed", return_value=True), \
                mock.patch.object(stt_engine, "env_exists", return_value=True), \
                mock.patch.object(stt_engine, "gpu_runtime_missing",
                                  return_value=[]):
            panel._show_status()
            self.assertFalse(panel.install_btn.IsEnabled())
            self.assertIn("Install", panel.install_btn.GetLabel())


@unittest.skipUnless(_ffmpeg_available(), "FFmpeg is not installed")
class AudioDescriptionWizardTest(_AppMixin):
    """The third project base: Audio description, end to end inside the wizard."""

    def _wizard(self):
        settings = Settings(path=os.path.join(tempfile.mkdtemp(), "settings.json"))
        wizard = NewProjectWizard(self.frame, settings, self._empty_store())
        self.addCleanup(wizard.Destroy)
        return settings, wizard

    def test_the_base_combo_offers_audio_description(self):
        _settings, wizard = self._wizard()
        combo = wizard.page_details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        self.assertIn("Audio description", labels)
        self.assertEqual(
            combo.GetClientData(labels.index("Audio description")), "ad")

    def test_the_audio_description_page_is_ready_to_use(self):
        _settings, wizard = self._wizard()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        details._update_visibility()

        self.assertTrue(details._is_ad())
        # Audio description finishes straight from this page, exactly like the
        # Clipboard type, because the describe window asks for the rest.
        self.assertIsNone(details.GetNext())
        self.assertGreater(details.ad_model_combo.GetCount(), 0)
        self.assertGreater(details.ad_variant_combo.GetCount(), 0)
        self.assertGreater(details.ad_compute_combo.GetCount(), 0)
        self.assertGreater(details.ad_fps_combo.GetCount(), 0)
        self.assertGreater(details.ad_chunk_combo.GetCount(), 0)
        self.assertGreater(details.ad_output_combo.GetCount(), 0)
        # The preselect is a model that can actually run on this machine.
        self.assertIn(details.selected_ad_model(), describe_catalog.model_ids())
        self.assertTrue(describe_catalog.cpu_ok(
            details.selected_ad_model(), details.selected_ad_variant()))
        # The two numbers a user picks by are on screen.
        self.assertIn("GB", details.ad_model_note.GetLabel())
        self.assertIn("video memory", details.ad_model_note.GetLabel())
        self.assertIn("Hallucination", details.ad_model_note.GetLabel())
        # Nothing to describe yet, so the estimate says to add a video.
        self.assertTrue(details.ad_note.GetLabel())

    def test_the_describer_panel_manages_both_layers_separately(self):
        """Installing a model must not take away another variant's download."""
        from ai_voice_studio.describe import engine as describe_engine
        from ai_voice_studio.gui import settings_dialog as settings_module

        dlg = SettingsDialog(self.frame, _temp_settings(), self._empty_store())
        self.addCleanup(dlg.Destroy)
        panel = dlg.manage_describe_panel
        model_id = panel._selected_model()
        variant = panel._selected_variant()
        self.assertTrue(model_id)
        self.assertTrue(variant)

        # This test is about the two layers a variant's own files and the
        # model's environment are, so the environment is held *complete* here: a
        # missing package is a state of its own, pinned by the two tests below.
        # The declared-package probe reads the real environment on disk, so it is
        # answered too - a test must not depend on what is installed on the
        # machine running it.
        complete = mock.patch.object(describe_engine, "gpu_runtime_missing",
                                     return_value=[])
        complete.start()
        self.addCleanup(complete.stop)
        declared = mock.patch.object(describe_engine.environments,
                                     "missing_packages", return_value=[])
        declared.start()
        self.addCleanup(declared.stop)

        # The environment is missing: it is what has to be built first, and the
        # reason the variant buttons cannot act is stated on them.
        with mock.patch.object(describe_engine, "is_installed",
                               return_value=False), \
                mock.patch.object(describe_engine, "env_exists",
                                  return_value=False), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=False):
            panel._refresh_statuses()
        self.assertTrue(panel.install_btn.IsEnabled())
        self.assertFalse(panel.remove_btn.IsEnabled())
        self.assertFalse(panel.download_btn.IsEnabled())
        self.assertFalse(panel.remove_variant_btn.IsEnabled())
        self.assertIn("environment first", panel.download_btn.GetToolTipText())

        # The environment is installed and this variant is not downloaded yet:
        # downloading is possible and removing is not.  This is the state the
        # old panel left the user stuck in.
        with mock.patch.object(describe_engine, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "env_exists",
                                  return_value=True), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=False):
            panel._refresh_statuses()
        self.assertFalse(panel.install_btn.IsEnabled())
        self.assertTrue(panel.remove_btn.IsEnabled())
        self.assertTrue(panel.download_btn.IsEnabled())
        self.assertFalse(panel.remove_variant_btn.IsEnabled())

        # Downloaded: the same variant can be deleted again.
        with mock.patch.object(describe_engine, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "env_exists",
                                  return_value=True), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=True):
            panel._refresh_statuses()
        self.assertFalse(panel.download_btn.IsEnabled())
        self.assertTrue(panel.remove_variant_btn.IsEnabled())

        # The confirmation names the repository and where it is kept, and the
        # engine is asked for that model and variant.
        # The progress window and the completion handler are stubbed: this test
        # is about which model and variant the engine is asked for, not about a
        # real download or a real message box in a headless test run.
        with mock.patch.object(describe_engine, "variant_downloaded",
                               return_value=False), \
                mock.patch.object(settings_module.wx, "MessageBox",
                                  return_value=wx.YES) as box, \
                mock.patch.object(settings_module, "TaskProgressDialog"), \
                mock.patch.object(settings_module.threading, "Thread") as thread:
            panel._on_download(None)
        message = box.call_args.args[0]
        self.assertIn("Download the checkpoint files", message)
        self.assertIn(describe_engine.variant_repo(model_id, variant), message)
        self.assertTrue(thread.called)
        with mock.patch.object(describe_engine, "download_variant",
                               return_value={"ok": True}) as download, \
                mock.patch.object(panel, "_download_done"):
            panel._download_job(model_id, variant)
        self.assertEqual(download.call_args.args[:2], (model_id, variant))
        # The download runs in the model's own environment, so it is handed a
        # cancel event and a progress callback.
        self.assertTrue(callable(download.call_args.kwargs.get("progress")))
        self.assertIsNotNone(download.call_args.kwargs.get("cancel_event"))

        # And removing a variant asks the engine for just that variant.
        with mock.patch.object(settings_module.wx, "MessageBox",
                               return_value=wx.YES), \
                mock.patch.object(describe_engine, "remove_variant",
                                  return_value=None) as remove, \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=False):
            panel._on_remove_variant(None)
        self.assertEqual(remove.call_args.args[:2], (model_id, variant))

    def test_a_missing_quantiser_turns_install_into_repair(self):
        """A GPU install without bitsandbytes cannot load a large variant.

        A checkpoint bigger than the card - Gemma 3 12B on a 12 GB card - is
        loaded in 4-bit, and the quantiser is a CUDA-only package of the
        model's own environment; without it the panel must offer to add it
        instead of presenting the environment as complete.
        """
        from ai_voice_studio.describe import engine as describe_engine

        dlg = SettingsDialog(self.frame, _temp_settings(), self._empty_store())
        self.addCleanup(dlg.Destroy)
        panel = dlg.manage_describe_panel
        with mock.patch.object(describe_engine, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "env_exists",
                                  return_value=True), \
                mock.patch.object(describe_engine.environments,
                                  "missing_packages", return_value=[]), \
                mock.patch.object(describe_engine, "gpu_runtime_missing",
                                  return_value=["bitsandbytes"]):
            panel._refresh_statuses()
        self.assertTrue(panel.install_btn.IsEnabled())
        self.assertIn("Repair", panel.install_btn.GetLabel())
        self.assertIn("bitsandbytes", panel.status.GetLabel())

    def test_a_missing_declared_package_turns_install_into_repair(self):
        """MiniCPM-o 2.6: an environment built before its own code's packages
        were declared reports itself installed and then fails on the
        checkpoint's first import - "Could not load the model: This modeling
        file requires the following packages that were not found in your
        environment: torchaudio".

        Repair (rather than a Remove-and-rebuild of several gigabytes of
        PyTorch) is the way back, so the panel has to offer it and name what is
        missing.
        """
        from ai_voice_studio.describe import engine as describe_engine

        dlg = SettingsDialog(self.frame, _temp_settings(), self._empty_store())
        self.addCleanup(dlg.Destroy)
        panel = dlg.manage_describe_panel
        with mock.patch.object(describe_engine, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "env_exists",
                                  return_value=True), \
                mock.patch.object(describe_engine, "gpu_runtime_missing",
                                  return_value=[]), \
                mock.patch.object(describe_engine.environments,
                                  "missing_packages",
                                  return_value=["torchaudio", "librosa"]):
            panel._refresh_statuses()
        self.assertTrue(panel.install_btn.IsEnabled())
        self.assertIn("Repair", panel.install_btn.GetLabel())
        status = panel.status.GetLabel()
        self.assertIn("torchaudio", status)
        self.assertIn("librosa", status)

        # And an environment with everything its catalogue declares offers no
        # repair at all: the button goes back to saying Install, disabled.
        with mock.patch.object(describe_engine, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "env_exists",
                                  return_value=True), \
                mock.patch.object(describe_engine, "gpu_runtime_missing",
                                  return_value=[]), \
                mock.patch.object(describe_engine.environments,
                                  "missing_packages", return_value=[]):
            panel._refresh_statuses()
        self.assertFalse(panel.install_btn.IsEnabled())
        self.assertIn("Install", panel.install_btn.GetLabel())
        self.assertNotIn("Repair", panel.install_btn.GetLabel())

    def test_the_wizard_says_which_models_and_variants_are_ready(self):
        from ai_voice_studio.describe import engine as describe_engine

        _settings, wizard = self._wizard()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        details._update_visibility()

        def _model_labels() -> list:
            return [details.ad_model_combo.GetString(i)
                    for i in range(details.ad_model_combo.GetCount())]

        def _variant_labels() -> list:
            return [details.ad_variant_combo.GetString(i)
                    for i in range(details.ad_variant_combo.GetCount())]

        # Nothing installed yet: every model says so, and so does every variant.
        # The state itself is computed from the environment probe and the
        # checkpoint probe, so those are what this drives.
        with mock.patch.object(describe_engine.environments, "is_installed",
                               return_value=False), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=False):
            details._refresh_ad_states()
        self.assertTrue(_model_labels())
        for label in _model_labels():
            self.assertTrue(label.endswith("- not installed"), label)
        for label in _variant_labels():
            self.assertIn("not downloaded yet", label)
        self.assertIn("not installed yet", details.ad_state_note.GetLabel())
        self.assertIn("Manage the video describer",
                      details.ad_state_note.GetLabel())

        # Installed but with no checkpoint: the model is ready to use, and the
        # variant still says it has to be fetched.
        with mock.patch.object(describe_engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=False):
            details._refresh_ad_states()
        for label in _model_labels():
            self.assertTrue(label.endswith("- ready to use"), label)
        for label in _variant_labels():
            self.assertIn("not downloaded yet", label)
        self.assertIn("is not downloaded yet", details.ad_state_note.GetLabel())

        # Downloaded: both layers say so.
        with mock.patch.object(describe_engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=True):
            details._refresh_ad_states()
        for label in _variant_labels():
            self.assertTrue(label.endswith("- downloaded"), label)
        self.assertIn("already downloaded", details.ad_state_note.GetLabel())
        # The costs and the hallucination note are still stated.
        note = details.ad_model_note.GetLabel()
        self.assertIn("GB", note)
        self.assertIn("Hallucination", note)

    def test_the_wizard_names_a_variant_whose_download_stopped_part_way(self):
        """A cancelled fetch is not "not downloaded yet" - the files are here."""
        from ai_voice_studio.describe import engine as describe_engine

        _settings, wizard = self._wizard()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        details._update_visibility()
        variant_combo = details.ad_variant_combo

        with mock.patch.object(describe_engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(describe_engine, "variant_downloaded",
                                  return_value=False), \
                mock.patch.object(describe_engine, "variant_partial",
                                  return_value=True):
            details._refresh_ad_states()
        selected = variant_combo.GetString(variant_combo.GetSelection())
        self.assertIn("download stopped part-way", selected)
        # The state line says what to do about it rather than calling it missing.
        note = details.ad_state_note.GetLabel()
        self.assertIn("stopped part-way", note)
        self.assertIn("Manage the video describer", note)

    def test_finishing_without_a_video_does_not_create_a_project(self):
        _settings, wizard = self._wizard()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        with mock.patch.object(
                sys.modules["ai_voice_studio.gui.new_project_wizard"].wx,
                "MessageBox") as box:
            wizard._on_finish(None)
        self.assertTrue(box.called)
        self.assertIn("video", box.call_args.args[0].lower())

    def _ad_project(self, root: str) -> tuple[str, str]:
        """An audio description project with one described source file."""
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        pdir = os.path.join(root, "project")
        project.create_project(
            pdir, "Test", source, "chunked",
            [{"index": 1, "title": "clip.mp4", "text": "", "source": source}],
            project_type="audio_description",
            describe_settings={"model": "smolvlm2", "variant": "2.2b",
                               "sources": [source]},
        )
        return source, pdir

    def _ad_frame(self, prefix: str = "aivs_ad_menu_"):
        """A MainFrame whose only recent project is an audio description one."""
        root = tempfile.mkdtemp(prefix=prefix)
        self.addCleanup(shutil.rmtree, root, True)
        source, pdir = self._ad_project(root)
        settings = Settings(path=os.path.join(root, "settings.json"))
        settings.add_recent_project("Test", pdir)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        frame.recent_list.SetSelection(0)
        return frame, source, pdir

    def test_the_edit_menu_speaks_the_audio_description_wording(self):
        frame, _source, _pdir = self._ad_frame()
        self.assertEqual(frame._selected_project_base(), "ad")
        labels = [frame._edit_label(entry, "ad")
                  for entry in frame._edit_items if entry[0] is not None]
        self.assertIn("Restart &All Descriptions", labels)
        self.assertIn("Start &Selected Descriptions...", labels)
        # The Edit menu really is re-labelled when the AD project is selected,
        # and the TTS/STT wording is untouched by the new base.
        frame._refresh_edit_menu()
        shown = [item.GetItemLabelText()
                 for item in frame._edit_menu.GetMenuItems()]
        self.assertIn("Restart All Descriptions", shown)
        self.assertNotIn("Restart All Recording", shown)
        stt_labels = [frame._edit_label(entry, "stt")
                      for entry in frame._edit_items if entry[0] is not None]
        self.assertIn("Restart &All Files", stt_labels)
        self.assertNotIn("Restart &All Descriptions", stt_labels)
        tts_labels = [frame._edit_label(entry, "tts")
                      for entry in frame._edit_items if entry[0] is not None]
        self.assertIn("Restart &All Recording", tts_labels)

    def test_opening_an_audio_description_project_uses_the_describe_window(self):
        frame, _source, pdir = self._ad_frame("aivs_ad_open_")
        describe_module = sys.modules["ai_voice_studio.gui.main_frame"]
        with mock.patch.object(describe_module, "DescribeDialog") as dlg, \
                mock.patch.object(describe_module, "RecordingDialog") as rec:
            frame._open_recording(pdir)
        self.assertTrue(dlg.called)
        self.assertFalse(rec.called)

    def test_restart_all_removes_the_description_documents(self):
        frame, source, pdir = self._ad_frame("aivs_ad_reset_")
        describe_output = sys.modules["ai_voice_studio.describe.output"]
        written = describe_output.write_document(
            pdir, "clip", [{"start": 0, "end": 10, "text": "Something."}],
            "script", source=source)
        data = project.load_project(pdir)
        data["describe_outputs"] = [{"source": source, "files": [written]}]
        data["segments"][0]["status"] = "done"
        project.save_project(pdir, data)
        removed = frame._clear_describe_outputs(pdir)
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(written))
        after = project.load_project(pdir)
        self.assertEqual(after["describe_outputs"], [])
        self.assertEqual(after["segments"][0]["status"], "pending")

    def test_restart_all_leaves_other_project_types_alone(self):
        frame = MainFrame(settings=Settings(path=os.path.join(
            tempfile.mkdtemp(), "settings.json")), store=self._empty_store())
        self.addCleanup(frame.Destroy)
        root = tempfile.mkdtemp(prefix="aivs_ad_reset2_")
        pdir = os.path.join(root, "tts")
        project.create_project(pdir, "Book", "a.txt", "h1_only",
                               [{"index": 1, "title": "c", "text": "hi"}])
        self.assertEqual(frame._clear_describe_outputs(pdir), 0)

    def _describe_dialog(self, root: str, duration: float):
        """A DescribeDialog whose only source is reported as ``duration`` long."""
        from ai_voice_studio.gui import describe_dialog as describe_module

        _source, pdir = self._ad_project(root)
        settings = Settings(path=os.path.join(root, "settings.json"))
        dlg = describe_module.DescribeDialog(
            self.frame, pdir, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        patcher = mock.patch.object(describe_module.media, "total_duration",
                                    return_value=duration)
        patcher.start()
        self.addCleanup(patcher.stop)
        dlg._refresh_estimate()
        return dlg

    def test_a_long_video_is_warned_about_before_it_starts(self):
        dlg = self._describe_dialog(tempfile.mkdtemp(prefix="aivs_ad_warn_"),
                                    7200.0)
        note = dlg.estimate_label.GetLabel()
        self.assertIn("Estimated time", note)
        self.assertIn("long video", note.lower())
        # Two hours at the default two-minute chunks is sixty of them.
        self.assertIn("60 chunk(s)", note)

    def test_a_short_video_is_not_warned_about(self):
        dlg = self._describe_dialog(tempfile.mkdtemp(prefix="aivs_ad_short_"),
                                    30.0)
        note = dlg.estimate_label.GetLabel()
        self.assertIn("Estimated time", note)
        self.assertNotIn("long video", note.lower())

    def test_the_long_video_warning_offers_shorter_chunks(self):
        from ai_voice_studio.gui import describe_dialog as describe_module

        dlg = self._describe_dialog(tempfile.mkdtemp(prefix="aivs_ad_chunk_"),
                                    7200.0)
        options = dlg._options()
        self.assertGreater(options["chunk_seconds"], 60.0)
        with mock.patch.object(describe_module, "run_action_dialog",
                               return_value="chunk") as gate:
            self.assertTrue(dlg._long_video_gate(options))
        # The offer is an action: accepting it really shortens the chunks.
        self.assertEqual(options["chunk_seconds"], 60.0)
        self.assertEqual(dlg._selected_chunk(), 60.0)
        actions = gate.call_args.args[3]
        self.assertIn("chunk", [key for key, _label in actions])
        self.assertIn("cancel", [key for key, _label in actions])

    def test_the_describe_dialog_reports_every_chunk_as_it_finishes(self):
        """The run shows chunk by chunk what it has done, not just a bar."""
        dlg = self._describe_dialog(tempfile.mkdtemp(prefix="aivs_ad_log_"),
                                    180.0)
        self.assertEqual(dlg.chunk_log.GetValue(), "")
        dlg._log_line("Starting: 1 video file(s).")
        dlg._update_progress("Describing chunk 1 of 3...", 0.3)
        # The same sentence twice in a row is written once: a stalled run must
        # not fill the report with a repeat.
        dlg._update_progress("Describing chunk 1 of 3...", 0.3)
        self.assertEqual(
            dlg.chunk_log.GetValue().count("Describing chunk 1 of 3..."), 1)
        dlg._on_chunk_done(
            {"chunk": 0, "start": 0.0, "end": 60.0,
             "text": "A lecturer points at a slide."}, 1, 3)
        dlg._update_progress("Skipped 2 of 3 chunk(s): the picture had not "
                             "changed.", 0.6)
        dlg._on_chunk_done(
            {"chunk": 2, "start": 120.0, "end": 180.0,
             "text": "The slide changes to a graph."}, 2, 3)
        report = dlg.chunk_log.GetValue()
        self.assertIn("Starting: 1 video file(s).", report)
        self.assertIn("Describing chunk 1 of 3...", report)
        self.assertIn("Chunk 1 of 3 (chunk 0) described: A lecturer points at "
                      "a slide.", report)
        self.assertIn("Skipped 2 of 3 chunk(s)", report)
        self.assertIn("Chunk 2 of 3 (chunk 2) described: The slide changes to "
                      "a graph.", report)
        # The report keeps a chronological line per event, so the second run's
        # own skip is a second line - but the status line carries the latest
        # sentence, which is what a screen reader reads out.
        self.assertEqual(dlg.status.GetLabel(),
                         "Skipped 2 of 3 chunk(s): the picture had not changed.")

    def test_the_chunk_report_is_cleared_for_each_run(self):
        from ai_voice_studio.describe import engine as describe_engine
        from ai_voice_studio.gui import describe_dialog as describe_module

        dlg = self._describe_dialog(tempfile.mkdtemp(prefix="aivs_ad_log_"),
                                    60.0)
        dlg.chunk_log.SetValue("a previous run")
        with mock.patch.object(describe_engine, "model_available",
                               return_value=True), \
                mock.patch.object(describe_module, "TaskProgressDialog"), \
                mock.patch.object(describe_module, "DescribeJob") as started:
            dlg._on_start(None)
        self.assertTrue(started.called)
        report = dlg.chunk_log.GetValue()
        self.assertNotIn("a previous run", report)
        self.assertIn("Starting:", report)
        # ...and the job is handed a callback for every finished chunk.
        self.assertTrue(callable(started.call_args.kwargs.get("on_chunk")))

    def test_cancelling_the_warning_stops_the_run(self):
        from ai_voice_studio.gui import describe_dialog as describe_module

        dlg = self._describe_dialog(tempfile.mkdtemp(prefix="aivs_ad_cancel_"),
                                    7200.0)
        options = dlg._options()
        with mock.patch.object(describe_module, "run_action_dialog",
                               return_value="cancel"):
            self.assertFalse(dlg._long_video_gate(options))
        self.assertEqual(dlg._selected_chunk(), options["chunk_seconds"])

    # -- spoken narration --------------------------------------------------
    VOICE = {
        "tts": "piper", "tts_name": "Piper", "language": "en_US",
        "variant": "medium", "voice": "amy", "voice_name": "Amy", "sid": 0,
        "engine": "vits", "dir": "C:/models/amy",
    }
    OTHER_VOICE = {
        "tts": "kokoro", "tts_name": "Kokoro", "language": "en_US",
        "variant": "82m", "voice": "af_heart", "voice_name": "Heart",
        "sid": 0, "engine": "kokoro", "dir": "C:/models/kokoro",
    }

    def _patch_duration(self, seconds: float):
        """The describe dialog asks ``media.total_duration`` for its estimate."""
        from ai_voice_studio.gui import describe_dialog as describe_module

        return mock.patch.object(describe_module.media, "total_duration",
                                 return_value=seconds)

    def _narration_dialog(self, root: str, voices, remembered=None,
                          fmt: str = "wav", priority: str | None = None,
                          volume: float | None = None):
        """A DescribeDialog for a machine with ``voices`` installed."""
        from ai_voice_studio.gui import describe_dialog as describe_module

        _source, pdir = self._ad_project(root)
        settings = Settings(path=os.path.join(root, "settings.json"))
        settings.set("recording.output_format", fmt)
        if priority is not None or volume is not None:
            # The project is what the dialog reads, so a remembered track
            # priority and narration volume are put where the wizard would
            # have written them.
            data = project.load_project(pdir)
            if priority is not None:
                data["describe"]["track_priority"] = priority
            if volume is not None:
                data["describe"]["narration_volume"] = volume
            project.save_project(pdir, data)
        if remembered is not None:
            # The wizard copies the remembered voice into the project; the
            # dialog reads the project, so that is where the test puts it.
            data = project.load_project(pdir)
            data["describe"]["narration_voice"] = dict(remembered)
            project.save_project(pdir, data)
        patch = mock.patch.object(ModelStore, "installed_voices",
                                  return_value=[dict(v) for v in voices])
        patch.start()
        self.addCleanup(patch.stop)
        dlg = describe_module.DescribeDialog(
            self.frame, pdir, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg

    def test_the_describe_dialog_offers_a_narration_voice(self):
        dlg = self._narration_dialog(
            tempfile.mkdtemp(prefix="aivs_ad_narr_"),
            [self.VOICE, self.OTHER_VOICE])
        self.assertTrue(dlg.narrate_check.IsEnabled())
        # The voice is chosen as a cascade - the TTS model, then one of its
        # downloaded variants, then one of that variant's voices - so only the
        # voices of the chosen model are listed at a time.
        self.assertEqual(dlg.narration_model_combo.GetCount(), 2)
        self.assertEqual(dlg.narration_model_combo.GetString(0), "Piper")
        self.assertEqual(
            dlg.narration_model_combo.GetClientData(
                dlg.narration_model_combo.GetSelection()), "piper")
        self.assertEqual(dlg.narration_variant_combo.GetCount(), 1)
        self.assertEqual(dlg.narration_variant_combo.GetString(0), "medium")
        self.assertEqual(dlg.voice_combo.GetCount(), 1)
        self.assertIn("Piper - Amy", dlg.voice_combo.GetString(0))
        # Choosing the other engine moves all three levels onto it.
        labels = [dlg.narration_model_combo.GetString(i)
                  for i in range(dlg.narration_model_combo.GetCount())]
        dlg.narration_model_combo.SetSelection(labels.index("Kokoro"))
        dlg._on_narration_model()
        self.assertEqual(dlg.narration_variant_combo.GetString(0), "82m")
        self.assertIn("Kokoro - Heart", dlg.voice_combo.GetString(0))
        self.assertEqual(dlg._selected_voice()["voice"], "af_heart")
        # Back to the first engine, for the rest of this check.
        dlg.narration_model_combo.SetSelection(labels.index("Piper"))
        dlg._on_narration_model()
        # Speaking the descriptions is what produces the narration track *and*
        # the described video, so a machine that has a voice starts with it on:
        # a run that wrote only the text looked like a run that had failed at the
        # other two outputs.
        self.assertTrue(dlg.narrate_check.GetValue())
        self.assertTrue(dlg.voice_combo.IsEnabled())
        self.assertTrue(dlg.narration_model_combo.IsEnabled())
        self.assertTrue(dlg._options()["narration"])
        note = dlg.narration_note.GetLabel()
        self.assertIn("Piper - Amy", note)
        # ...and it says what the two spoken outputs are, so nothing is a surprise
        # in the project folder afterwards.
        self.assertIn("described video", note)

        # It can still be turned off, and then nothing is promised.
        dlg.narrate_check.SetValue(False)
        dlg._on_narration()
        self.assertFalse(dlg.voice_combo.IsEnabled())
        self.assertFalse(dlg.narration_model_combo.IsEnabled())
        self.assertFalse(dlg._options()["narration"])
        self.assertEqual(dlg.narration_note.GetLabel(), "")

        dlg.narrate_check.SetValue(True)
        dlg._on_narration()
        options = dlg._options()
        self.assertTrue(options["narration"])
        self.assertEqual(options["voice"]["voice"], "amy")
        self.assertEqual(options["narration_format"], "wav")
        self.assertTrue(dlg.voice_combo.IsEnabled())
        self.assertTrue(dlg.narration_model_combo.IsEnabled())
        self.assertTrue(dlg.narration_variant_combo.IsEnabled())
        note = dlg.narration_note.GetLabel()
        self.assertIn("Piper - Amy", note)
        self.assertIn("WAV", note)
        # The recording settings drive the delivery, so they are carried through.
        self.assertEqual(options["tts"]["rate"], 1.0)
        self.assertIn("compute", options["tts"])

    def test_narration_needs_an_installed_voice(self):
        from ai_voice_studio.describe import engine as describe_engine

        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_narr_"), [])
        self.assertFalse(dlg.narrate_check.IsEnabled())
        self.assertFalse(dlg.narrate_check.GetValue())
        self.assertEqual(dlg.voice_combo.GetCount(), 0)
        self.assertEqual(dlg.narration_model_combo.GetCount(), 0)
        self.assertEqual(dlg.narration_variant_combo.GetCount(), 0)
        self.assertIn("Settings", dlg.narrate_check.GetToolTipText())
        # Even forced on, Start refuses to promise a track it cannot speak.
        dlg.narrate_check.SetValue(True)
        with mock.patch.object(describe_engine, "model_available",
                               return_value=True), \
                mock.patch.object(wx, "MessageBox") as box:
            dlg._on_start(None)
        self.assertTrue(box.called)
        self.assertIn("offline voice", box.call_args.args[0])
        self.assertIsNone(dlg.job)

    # -- FFmpeg -------------------------------------------------------------
    def test_a_missing_ffmpeg_is_offered_for_download_not_reported_as_failure(self):
        """Video processing without FFmpeg asks to install it.

        FFmpeg is what decodes the frames, reads the video's length and writes the
        described video, so a run without it cannot start - and the answer is an
        offer to install it, which is what the recording side already does for
        MP3 and FLAC.
        """
        from ai_voice_studio.audio import ffmpeg as ffmpeg_mod
        from ai_voice_studio.gui import describe_dialog as describe_module

        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_ffmpeg_"),
                                     [self.VOICE])
        started: list = []

        class _FakeThread:
            def __init__(self, target=None, **kwargs):
                started.append(target)

            def start(self):
                pass

        with mock.patch.object(ffmpeg_mod, "ensure_ffmpeg", return_value=None), \
                mock.patch.object(wx, "MessageBox", return_value=wx.YES) as box, \
                mock.patch.object(describe_module.threading, "Thread",
                                  _FakeThread):
            self.assertFalse(dlg._ensure_ffmpeg())
        self.assertTrue(box.called)
        message = box.call_args.args[0]
        self.assertIn("FFmpeg", message)
        self.assertIn("Download FFmpeg now", message)
        self.assertIn("described video", message)
        # The download runs on its own thread and the start button waits for it.
        self.assertEqual(len(started), 1)
        self.assertTrue(callable(started[0]))
        self.assertFalse(dlg.start_btn.IsEnabled())
        self.assertIn("Downloading FFmpeg", dlg.status.GetLabel())

    def test_declining_the_ffmpeg_install_does_not_start_the_run(self):
        from ai_voice_studio.audio import ffmpeg as ffmpeg_mod

        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_ffmpeg_"),
                                     [self.VOICE])
        with mock.patch.object(ffmpeg_mod, "ensure_ffmpeg", return_value=None), \
                mock.patch.object(wx, "MessageBox", return_value=wx.NO):
            self.assertFalse(dlg._ensure_ffmpeg())
        self.assertIn("FFmpeg is required", dlg.status.GetLabel())

    def test_ffmpeg_that_is_already_there_is_not_asked_about(self):
        from ai_voice_studio.audio import ffmpeg as ffmpeg_mod

        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_ffmpeg_"),
                                     [self.VOICE])
        with mock.patch.object(ffmpeg_mod, "ensure_ffmpeg",
                               return_value="C:/ffmpeg/bin/ffmpeg.exe"), \
                mock.patch.object(wx, "MessageBox") as box:
            self.assertTrue(dlg._ensure_ffmpeg())
        self.assertFalse(box.called)

    def test_the_estimate_says_how_many_descriptions_are_coming(self):
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_est_"),
                                     [self.VOICE])
        with self._patch_duration(600.0):
            dlg._refresh_estimate()
        text = dlg.estimate_label.GetLabel()
        # 600 seconds of video in twenty-second pieces is thirty descriptions.
        self.assertIn("30 description(s)", text)
        self.assertIn("20 seconds", text)
        # ...and the three outputs are named, so nothing is a surprise in the
        # project folder afterwards.
        self.assertIn("described video", text)

    def test_the_run_is_told_how_much_video_one_description_covers(self):
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_est_"),
                                     [self.VOICE])
        from ai_voice_studio.constants import AD_WINDOW_SECONDS

        self.assertEqual(dlg._options()["window_seconds"], AD_WINDOW_SECONDS)

    def test_the_remembered_narration_voice_is_preselected(self):
        dlg = self._narration_dialog(
            tempfile.mkdtemp(prefix="aivs_ad_narr_"),
            [self.VOICE, self.OTHER_VOICE],
            remembered={"tts": "kokoro", "language": "en_US",
                        "variant": "82m", "voice": "af_heart"})
        self.assertEqual(dlg._selected_voice()["voice"], "af_heart")
        self.assertIn("Kokoro", dlg.voice_combo.GetString(
            dlg.voice_combo.GetSelection()))

    def test_the_describe_dialog_offers_a_narration_volume_slider(self):
        """The description's own level, right under the voice that speaks it.

        It is the one narration level this window asks for on its own: *Track
        priority* moves the film's track, never the description, so a louder
        description would otherwise only be reachable in Settings > Recording.
        """
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_vol_"),
                                     [self.VOICE])
        slider = dlg.narration_volume_slider
        self.assertIn("Narration voice volume", slider.GetName())
        self.assertEqual(slider.GetValue(), 100)
        self.assertIn("Narration voice volume: 1.00",
                      dlg.narration_volume_label.GetLabel())
        self.assertEqual(dlg._options()["tts"]["volume"], 1.0)
        # It follows the voice it belongs to: nothing to speak, nothing to set.
        self.assertTrue(slider.IsEnabled())
        dlg.narrate_check.SetValue(False)
        dlg._on_narration()
        self.assertFalse(slider.IsEnabled())
        dlg.narrate_check.SetValue(True)
        dlg._on_narration()
        self.assertTrue(slider.IsEnabled())
        # Moving it changes the level the run is told to speak with, and the
        # label and accessible name say the new number.
        slider.SetValue(150)
        dlg._on_narration_volume()
        self.assertEqual(dlg._options()["tts"]["volume"], 1.5)
        self.assertIn("Narration voice volume: 1.50",
                      dlg.narration_volume_label.GetLabel())
        self.assertIn("1.50", slider.GetName() or "")
        # ...and the next project starts where this one left off.
        dlg._remember(dlg._options())
        self.assertEqual(dlg.settings.get("describe.narration_volume"), 1.5)

    def test_the_remembered_narration_volume_is_preselected(self):
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_vol_"),
                                     [self.VOICE], volume=1.5)
        self.assertEqual(dlg.narration_volume_slider.GetValue(), 150)
        self.assertEqual(dlg._options()["tts"]["volume"], 1.5)
        # A project with no volume of its own speaks at the recording level.
        plain = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_vol_"),
                                       [self.VOICE])
        self.assertEqual(plain.narration_volume_slider.GetValue(), 100)
        self.assertEqual(plain._options()["tts"]["volume"], 1.0)

    # -- extended audio description ----------------------------------------
    def _choose_extended(self, dlg):
        combo = dlg.placement_combo
        for index in range(combo.GetCount()):
            if combo.GetClientData(index) == "extended":
                combo.SetSelection(index)
                break
        dlg._on_placement()
        return dlg

    def test_the_describe_dialog_offers_extended_audio_description(self):
        dlg = self._narration_dialog(
            tempfile.mkdtemp(prefix="aivs_ad_ext_"), [self.VOICE])
        combo = dlg.placement_combo
        values = [combo.GetClientData(i) for i in range(combo.GetCount())]
        self.assertEqual(values, ["standard", "extended"])
        self.assertEqual(dlg._selected_placement(), "standard")
        self.assertFalse(dlg._is_extended())
        self.assertIn("Pause the video", combo.GetString(1))

    def test_stopping_the_film_turns_on_the_voice_and_off_the_pauses(self):
        dlg = self._narration_dialog(
            tempfile.mkdtemp(prefix="aivs_ad_ext_"), [self.VOICE])
        self.assertTrue(dlg.gap_check.IsEnabled())
        self._choose_extended(dlg)
        # Extended audio description *is* spoken: the description is heard while
        # the film is stopped, so the voice cannot be optional there...
        self.assertTrue(dlg.narrate_check.GetValue())
        self.assertFalse(dlg.narrate_check.IsEnabled())
        self.assertTrue(dlg.voice_combo.IsEnabled())
        # ...and the pauses it replaces are switched off, not silently ignored.
        self.assertFalse(dlg.gap_check.IsEnabled())
        self.assertIn("no pause to fit into", dlg.gap_check.GetToolTipText())
        options = dlg._options()
        self.assertEqual(options["placement"], "extended")
        self.assertTrue(options["narration"])
        self.assertEqual(options["voice"]["voice"], "amy")
        note = dlg.narration_note.GetLabel()
        self.assertIn("Extended audio description", note)
        self.assertIn("described video", note)

    def test_the_estimate_says_how_much_longer_the_film_becomes(self):
        dlg = self._narration_dialog(
            tempfile.mkdtemp(prefix="aivs_ad_ext_"), [self.VOICE])
        with self._patch_duration(600.0):
            dlg._refresh_estimate()
        standard = dlg.estimate_label.GetLabel()
        self.assertNotIn("stops the film", standard)
        self._choose_extended(dlg)
        with self._patch_duration(600.0):
            dlg._refresh_estimate()
        extended_note = dlg.estimate_label.GetLabel()
        self.assertIn("stops the film", extended_note)
        self.assertIn("becomes about", extended_note)

    def test_extended_mode_without_a_voice_explains_itself(self):
        from ai_voice_studio.describe import engine as describe_engine

        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_ext_"), [])
        self._choose_extended(dlg)
        note = dlg.narration_note.GetLabel()
        self.assertIn("needs an offline voice", note)
        with mock.patch.object(describe_engine, "model_available",
                               return_value=True), \
                mock.patch.object(wx, "MessageBox") as box:
            dlg._on_start(None)
        self.assertTrue(box.called)
        message = box.call_args.args[0]
        self.assertIn("stops the film", message)
        self.assertIn("standard audio description", message)
        self.assertIsNone(dlg.job)

    # -- track priority -----------------------------------------------------
    def test_the_describe_dialog_offers_a_track_priority(self):
        """The combo box that decides which of the two tracks gives way.

        It sits with the placement choice, before the button that starts the
        run, and it opens on "same volume" - both tracks as they are - which is
        what every run before this choice existed produced.
        """
        from ai_voice_studio.constants import AD_TRACK_PRIORITY_CHOICES

        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_prio_"),
                                     [self.VOICE])
        combo = dlg.priority_combo
        self.assertEqual(
            [combo.GetClientData(i) for i in range(combo.GetCount())],
            [value for value, _label in AD_TRACK_PRIORITY_CHOICES])
        self.assertEqual(combo.GetCount(), 4)
        # From the gentlest answer to the film to the most forceful...
        self.assertIn("Same volume", combo.GetString(0))
        self.assertIn("Duck, don't lower", combo.GetString(1))
        self.assertIn("25% higher", combo.GetString(2))
        self.assertIn("50% higher", combo.GetString(3))
        # ...and the first entry is the default: a project that never chose one
        # opens on *same volume*, because that is what those runs produced.
        self.assertEqual(dlg._selected_track_priority(), "same")
        self.assertEqual(combo.GetSelection(), 0)
        self.assertEqual(dlg._track_priority_gain(), 1.0)
        self.assertEqual(dlg._track_priority_duck(), 1.0)
        self.assertEqual(dlg._options()["track_priority"], "same")
        # It is above the Start button, so the choice is made before anything
        # is written rather than discovered afterwards.
        dlg.Layout()
        self.assertLess(combo.GetPosition().y, dlg.start_btn.GetPosition().y)

    def test_the_chosen_priority_is_what_the_run_is_told(self):
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_prio_"),
                                     [self.VOICE])
        combo = dlg.priority_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(
            labels.index("Description 25% higher (the film's own sound is "
                         "quieter by 25%)"))
        dlg._on_change()
        self.assertEqual(dlg._selected_track_priority(), "higher25")
        self.assertEqual(dlg._track_priority_gain(), 0.75)
        self.assertEqual(dlg._options()["track_priority"], "higher25")
        # The estimate names the choice, and says the description is not the
        # track that moves.
        with self._patch_duration(600.0):
            dlg._refresh_estimate()
        text = dlg.estimate_label.GetLabel()
        self.assertIn("Track priority", text)
        self.assertIn("75% of it", text)
        # ...and the next project starts where this one left off.
        dlg._remember(dlg._options())
        self.assertEqual(dlg.settings.get("describe.track_priority"), "higher25")

    def test_the_track_priority_is_remembered_in_the_project(self):
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_prio_"),
                                     [self.VOICE], priority="higher50")
        self.assertEqual(dlg._selected_track_priority(), "higher50")
        self.assertEqual(dlg._track_priority_gain(), 0.5)
        self.assertEqual(
            dlg.priority_combo.GetString(dlg.priority_combo.GetSelection()),
            "Description 50% higher (the film's own sound is quieter by 50%)")

    def test_the_ducking_choice_says_when_the_film_moves(self):
        """*Duck, don't lower*: chosen, reported, and remembered like the rest.

        The film is not lowered at all - its resting gain is 1.0 - so the
        estimate has to say *when* the film's own sound moves, or a reader sees
        a fraction and hears the choice they did not make.
        """
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_prio_"),
                                     [self.VOICE])
        combo = dlg.priority_combo
        combo.SetSelection(1)
        dlg._on_change()
        self.assertEqual(dlg._selected_track_priority(), "duck")
        self.assertEqual(dlg._track_priority_gain(), 1.0)
        self.assertEqual(dlg._track_priority_duck(), 0.25)
        self.assertEqual(dlg._options()["track_priority"], "duck")
        self.assertIn("Duck, don't lower", combo.GetToolTipText())
        self.assertIn("dips the film's own sound to a quarter of its own level "
                      "while a description is actually being spoken",
                      combo.GetToolTipText())
        with self._patch_duration(600.0):
            dlg._refresh_estimate()
        text = dlg.estimate_label.GetLabel()
        self.assertIn("Track priority", text)
        self.assertIn("25% of its own level only while a description is "
                      "actually being spoken", text)
        # ...and the next project starts where this one left off.
        dlg._remember(dlg._options())
        self.assertEqual(dlg.settings.get("describe.track_priority"), "duck")
        again = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_prio_"),
                                       [self.VOICE], priority="duck")
        self.assertEqual(again.priority_combo.GetSelection(), 1)
        self.assertEqual(again._track_priority_duck(), 0.25)

    def test_stopping_the_film_switches_the_track_priority_off(self):
        """The two tracks are never heard together in extended mode."""
        dlg = self._narration_dialog(tempfile.mkdtemp(prefix="aivs_ad_prio_"),
                                     [self.VOICE])
        self.assertTrue(dlg.priority_combo.IsEnabled())
        self._choose_extended(dlg)
        self.assertFalse(dlg.priority_combo.IsEnabled())
        self.assertIn("nothing needs rebalancing",
                      dlg.priority_combo.GetToolTipText())
        # It is switched off, not cleared: choosing standard placement again
        # brings the choice back exactly as it was.
        self.assertEqual(dlg._options()["track_priority"], "same")
        combo = dlg.placement_combo
        for index in range(combo.GetCount()):
            if combo.GetClientData(index) == "standard":
                combo.SetSelection(index)
                break
        dlg._on_placement()
        self.assertTrue(dlg.priority_combo.IsEnabled())
        self.assertIn("Which of the two tracks gives way",
                      dlg.priority_combo.GetToolTipText())

    def test_the_placement_is_remembered_for_the_next_project(self):
        settings, wizard = self._wizard()
        settings.set("describe.placement", "extended")
        root = tempfile.mkdtemp(prefix="aivs_ad_ext_wiz_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        details._update_visibility()
        wizard.describe_sources = [source]
        wizard_dir = mock.patch.object(
            sys.modules["ai_voice_studio.gui.new_project_wizard"],
            "project_dir", return_value=os.path.join(root, "project"))
        with wizard_dir, mock.patch.object(
                sys.modules["ai_voice_studio.gui.new_project_wizard"],
                "DescribeDialog") as dlg, mock.patch.object(wizard, "EndModal"):
            wizard._on_finish(None)
        self.assertTrue(dlg.called)
        data = project.load_project(os.path.join(root, "project"))
        self.assertEqual(data["describe"]["placement"], "extended")

    def test_a_pause_less_video_is_pointed_at_extended_description(self):
        """A standard run says so when nothing was left to describe into."""
        from ai_voice_studio.describe import job as describe_job
        from ai_voice_studio.describe import pipeline as describe_pipeline

        root = tempfile.mkdtemp(prefix="aivs_ad_hint_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        pdir = os.path.join(root, "project")
        project.create_project(
            pdir, "Test", source, "chunked",
            [{"index": 1, "title": "clip.mp4", "text": "", "source": source}],
            project_type="audio_description",
            describe_settings={"model": "smolvlm2", "variant": "2.2b",
                               "sources": [source]},
        )
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": describe_pipeline.plan_chunks(240.0, 120.0)}
        options = {"model": "smolvlm2", "variant": "2.2b",
                   "output_format": "script", "gap_fitting": True}
        with mock.patch.object(describe_job.engine, "plan_for_video",
                               return_value=plan), \
                mock.patch.object(describe_job.engine, "describe_video",
                                  side_effect=lambda video, **kw: []):
            run = describe_job.DescribeJob(pdir, [source], options)
            run.run()
        self.assertEqual(len(run.warnings), 1)
        self.assertIn("extended audio description", run.warnings[0])
        # The unit the warning counts is the piece that produced no description,
        # which is a window now, not a chunk.
        self.assertIn("2 of 2 piece(s)", run.warnings[0])

    def test_a_new_project_carries_the_remembered_narration_choice(self):
        """Settings > new project > describe window, without a real dialog."""
        settings, wizard = self._wizard()
        settings.set("describe.narration", True)
        settings.set("describe.narration_voice", {
            "tts": "piper", "language": "en_US", "variant": "medium",
            "voice": "amy"})
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_wiz_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        details._update_visibility()
        wizard.describe_sources = [source]
        wizard_dir = mock.patch.object(
            sys.modules["ai_voice_studio.gui.new_project_wizard"],
            "project_dir", return_value=os.path.join(root, "project"))
        patcher = mock.patch.object(
            sys.modules["ai_voice_studio.gui.new_project_wizard"],
            "DescribeDialog")
        with wizard_dir, patcher as dlg, mock.patch.object(wizard, "EndModal"):
            wizard._on_finish(None)
        self.assertTrue(dlg.called)
        data = project.load_project(os.path.join(root, "project"))
        self.assertTrue(data["describe"]["narration"])
        self.assertEqual(data["describe"]["narration_voice"]["voice"], "amy")

    def test_a_new_project_carries_the_remembered_track_priority(self):
        """The remembered choice reaches the project and then the window."""
        settings, wizard = self._wizard()
        settings.set("describe.track_priority", "higher25")
        root = tempfile.mkdtemp(prefix="aivs_ad_prio_wiz_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        details = wizard.page_details
        combo = details.base_combo
        labels = [combo.GetString(i) for i in range(combo.GetCount())]
        combo.SetSelection(labels.index("Audio description"))
        details._update_visibility()
        wizard.describe_sources = [source]
        wizard_dir = mock.patch.object(
            sys.modules["ai_voice_studio.gui.new_project_wizard"],
            "project_dir", return_value=os.path.join(root, "project"))
        with wizard_dir, mock.patch.object(
                sys.modules["ai_voice_studio.gui.new_project_wizard"],
                "DescribeDialog") as dlg, mock.patch.object(wizard, "EndModal"):
            wizard._on_finish(None)
        self.assertTrue(dlg.called)
        data = project.load_project(os.path.join(root, "project"))
        self.assertEqual(data["describe"]["track_priority"], "higher25")

    def test_the_describe_dialog_builds_and_gates_an_uninstalled_model(self):
        from ai_voice_studio.describe import engine as describe_engine
        from ai_voice_studio.gui.describe_dialog import DescribeDialog

        root = tempfile.mkdtemp(prefix="aivs_ad_gui_")
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        pdir = os.path.join(root, "project")
        project.create_project(
            pdir, "Test", source, "chunked",
            [{"index": 1, "title": "clip.mp4", "text": "", "source": source}],
            project_type="audio_description",
            describe_settings={"model": "smolvlm2", "variant": "2.2b",
                               "sources": [source]},
        )
        settings = Settings(path=os.path.join(root, "settings.json"))
        dlg = DescribeDialog(self.frame, pdir, settings, self._empty_store())
        self.addCleanup(dlg.Destroy)
        # The wizard already asked which model describes the picture, so this
        # window reports it instead of asking again - and never repeats the
        # page's own inputs.
        self.assertFalse(hasattr(dlg, "model_combo"))
        self.assertFalse(hasattr(dlg, "variant_combo"))
        self.assertFalse(hasattr(dlg, "fps_combo"))
        self.assertFalse(hasattr(dlg, "chunk_combo"))
        self.assertFalse(hasattr(dlg, "output_combo"))
        self.assertFalse(hasattr(dlg, "compute_combo"))
        summary = dlg.run_label.GetLabel()
        self.assertIn(describe_catalog.model("smolvlm2")["name"], summary)
        # The variant is named as the catalog names it, and its state is stated
        # rather than left to be discovered by starting a run.
        variant_label = dict(describe_catalog.variants("smolvlm2"))["2.2b"]
        self.assertIn(variant_label, summary)
        self.assertEqual(dlg._current_model(), "smolvlm2")
        self.assertEqual(dlg._current_variant(), "2.2b")
        self.assertEqual([dlg.sources], [[source]])
        self.assertTrue(dlg.gauge)
        self.assertTrue(dlg.start_btn)
        self.assertTrue(dlg.folder_btn)
        with mock.patch.object(describe_engine, "model_available",
                               return_value=False), \
                mock.patch.object(
                    sys.modules["ai_voice_studio.gui.describe_dialog"].wx,
                    "MessageBox") as box:
            dlg._on_start(None)
        self.assertTrue(box.called)
        self.assertIn("Manage the video describer", box.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
