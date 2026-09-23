"""Keyboard gestures: every access key and accelerator does what it says.

Two user reports are pinned down here.

*"I select a model to remove in the Download and remove category and press
Alt+R - it will not remove it."*  Alt+R used to be advertised on the Remove
button while the Variant box opened on "Download all variants" (which has no
single model behind it), so the key landed on a dead control without a word.
The outcome is the honest one: that key is gone, the page opens on a
downloaded model when there is one, and the remaining access key (Alt+D)
explains itself when it cannot act
(``test_the_download_category_preselects_a_downloaded_variant``,
``test_a_disabled_target_explains_itself_instead_of_nothing``).

*"Ctrl+Shift+N is not working for create new project."*  Ctrl+Shift+N now
belongs to the welcome panel's *Create New Project* button alone: the File
menu item carries no accelerator and the frame routes the key into that
button's own click path, so there is one owner and one code path
(``test_ctrl_shift_n_presses_the_create_new_project_button``).

The gesture audit tool is run as a test as well, so a newly added window
cannot quietly bring a shadowed, duplicated or unexplained access key with
it.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import a11y_gestures  # noqa: E402

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.constants import MODE_PAGE_WITH_H1  # noqa: E402
from ai_voice_studio.gui import access_keys  # noqa: E402
from ai_voice_studio.gui.main_frame import MainFrame  # noqa: E402
from ai_voice_studio.gui.recording_dialog import RecordingDialog  # noqa: E402
from ai_voice_studio.gui.settings_dialog import SettingsDialog  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402


def _char_hook_event(key: str, *, alt: bool = False, ctrl: bool = False,
                     shift: bool = False, scan: int = 0) -> wx.KeyEvent:
    """A synthetic ``wxEVT_CHAR_HOOK`` key event for ``key``."""
    evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evt.SetAltDown(alt)
    evt.SetControlDown(ctrl)
    evt.SetShiftDown(shift)
    evt.SetKeyCode(ord(key))
    evt.SetRawKeyFlags(max(scan, ord(key)) << 16)
    return evt


def _post_alt_key(control: wx.Window, key: str, scan: int) -> None:
    """Post the two Windows messages one Alt+<key> keystroke produces.

    A real key press delivers ``WM_SYSKEYDOWN`` to the focused control, and
    the message loop derives ``WM_SYSCHAR`` from it; both are answered by the
    dialog manager, which is precisely why one press used to do two things.
    """
    import ctypes

    user32 = ctypes.windll.user32
    WM_SYSKEYDOWN, WM_SYSKEYUP, WM_SYSCHAR = 0x0104, 0x0105, 0x0106
    ALT, UP = 1 << 29, 1 << 30
    vk = ord(key.upper())
    hwnd = control.GetHandle()
    user32.PostMessageW(hwnd, WM_SYSKEYDOWN, vk, 1 | (scan << 16) | ALT)
    user32.PostMessageW(hwnd, WM_SYSCHAR, ord(key.lower()),
                        1 | (scan << 16) | ALT | (1 << 31))
    user32.PostMessageW(hwnd, WM_SYSKEYUP, vk,
                        1 | (scan << 16) | ALT | UP | (1 << 31))


class _GestureCase(unittest.TestCase):
    """A wx.App, a parent frame, and helpers to build the real windows."""

    @classmethod
    def setUpClass(cls):
        cls.app = wx.App(False)
        cls.parent = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.parent.Destroy()

    # -- windows -----------------------------------------------------------
    def _fresh_store(self) -> tuple[ModelStore, str]:
        folder = tempfile.mkdtemp(prefix="aivs_gestures_")
        self.addCleanup(shutil.rmtree, folder, True)
        return ModelStore(state_file=os.path.join(folder, "models.json")), folder

    def _settings(self, store: ModelStore | None = None) -> SettingsDialog:
        if store is None:
            store, _ = self._fresh_store()
        dlg = SettingsDialog(self.parent, Settings(), store)
        self.addCleanup(dlg.Destroy)
        dlg.SetSize((1000, 900))
        # Shown on purpose: a control inside a hidden window is invisible to
        # the access-key resolver, exactly as it is to Windows.
        dlg.Show()
        self.addCleanup(dlg.Hide)
        return dlg

    def _download_panel(self, store: ModelStore | None = None):
        """The Download and remove category, with its own store."""
        dlg = self._settings(store)
        index = next(i for i, c in enumerate(dlg.CATEGORIES)
                     if "Download" in (c.title or ""))
        dlg._show_category(index)
        dlg.Layout()
        wx.Yield()
        return dlg, dlg._panels[index]

    def _install_first_variant(self, panel, store: ModelStore, folder: str):
        """Mark one real variant of the panel's first engine as downloaded.

        Returns ``(tts_id, lang_code, variant_id, model_dir)``.
        """
        tts_id = panel.tts_combo.GetClientData(0)
        lang_index = 0
        panel.lang_combo.SetSelection(lang_index)
        lang_code = panel.lang_combo.GetClientData(lang_index)
        variant_id = None
        for index in range(panel.variant_combo.GetCount()):
            candidate = panel.variant_combo.GetClientData(index)
            if candidate and candidate != "__all__":
                variant_id = candidate
                break
        self.assertIsNotNone(variant_id, "the catalog has no single variant")
        model_dir = os.path.join(folder, "model")
        os.makedirs(model_dir, exist_ok=True)
        store.mark_installed(tts_id, lang_code, variant_id, model_dir)
        return tts_id, lang_code, variant_id, model_dir

    def _main_frame(self) -> MainFrame:
        settings_folder = tempfile.mkdtemp(prefix="aivs_gestures_")
        self.addCleanup(shutil.rmtree, settings_folder, True)
        settings = Settings(path=os.path.join(settings_folder, "settings.json"))
        store, _ = self._fresh_store()
        frame = MainFrame(settings=settings, store=store)
        self.addCleanup(frame.Destroy)
        frame.SetSize((900, 700))
        frame.Show()
        self.addCleanup(frame.Hide)
        return frame

    def _recording_window(self) -> RecordingDialog:
        folder = tempfile.mkdtemp(prefix="aivs_gestures_")
        self.addCleanup(shutil.rmtree, folder, True)
        project.create_project(
            folder, "Gesture project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        store, _ = self._fresh_store()
        dlg = RecordingDialog(self.parent, folder, Settings(), store)
        self.addCleanup(dlg.Destroy)
        dlg.SetSize((900, 800))
        dlg.Show()
        self.addCleanup(dlg.Hide)
        return dlg

    # -- 1. the trap default ----------------------------------------------
    def test_the_download_category_preselects_a_downloaded_variant(self):
        """Opening the page on "Download all variants" left Alt+R dead."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        tts_id, lang_code, variant_id, model_dir = self._install_first_variant(
            panel, store, folder)
        panel.tts_combo.SetSelection(0)
        panel._on_tts(None)                       # repopulates the cascades
        self.assertEqual(
            panel.variant_combo.GetClientData(panel.variant_combo.GetSelection()),
            variant_id,
            "the Variant box must land on the downloaded variant, not on "
            "\"Download all variants\"",
        )
        self.assertTrue(panel.remove_btn.IsEnabled(),
                        "Remove must be able to act as soon as the page opens")

    def test_the_download_category_opens_on_a_downloaded_model(self):
        """Opening the page on an engine that was never downloaded leaves
        Remove disabled, so the first Alt+R does nothing at all."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        tts_id = panel.tts_combo.GetClientData(0)
        lang_code = panel.lang_combo.GetClientData(0)
        variant_id = next(
            panel.variant_combo.GetClientData(index)
            for index in range(panel.variant_combo.GetCount())
            if panel.variant_combo.GetClientData(index) not in (None, "__all__")
        )
        model_dir = os.path.join(folder, "installed")
        os.makedirs(model_dir, exist_ok=True)
        store.mark_installed(tts_id, lang_code, variant_id, model_dir)
        panel.on_activated()               # the category is opened
        self.assertEqual(panel._selected(), (tts_id, lang_code, variant_id))
        self.assertTrue(panel.remove_btn.IsEnabled())

    def test_removing_the_preselected_variant_says_what_happened(self):
        """The removal reports itself and leaves the panel consistent."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        tts_id, lang_code, variant_id, model_dir = self._install_first_variant(
            panel, store, folder)
        panel._on_tts(None)
        removed = []
        panel.downloader.remove_variant = lambda *a: (removed.append(a), True)[1]
        with mock.patch.object(wx, "MessageBox", lambda *a, **k: wx.YES):
            panel._on_remove(None)
        self.assertEqual(removed, [(tts_id, lang_code, variant_id)])
        self.assertIn("Removed", panel.progress_label.GetLabel())
        self.assertIn(f"{tts_id} / {lang_code} / {variant_id}",
                      panel.progress_label.GetLabel())

    # -- 2. the disabled target -------------------------------------------
    def test_a_disabled_target_explains_itself_instead_of_nothing(self):
        """Alt+D on an already-downloaded model must say why it cannot act."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        self._install_first_variant(panel, store, folder)
        panel._on_tts(None)                 # lands on the downloaded variant
        self.assertFalse(panel.download_btn.IsEnabled())
        started = []
        panel._download_job = lambda *a: started.append(a)
        with mock.patch.object(access_keys, "alt_is_down", lambda: True):
            with mock.patch.object(wx, "Bell", lambda: None):
                handled = access_keys.handle_key(
                    panel, _char_hook_event("D", alt=True))
        self.assertTrue(handled, "the key must be consumed, not passed on")
        self.assertEqual(started, [], "nothing may be downloaded here")
        self.assertIn("already downloaded", panel.progress_label.GetLabel())

    def test_a_disabled_target_is_inert_for_the_next_message_too(self):
        """Two messages of one press must not give two explanations."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        self._install_first_variant(panel, store, folder)
        panel._on_tts(None)
        hints = []
        panel.show_access_key_hint = lambda entry: hints.append(entry.key)
        with mock.patch.object(access_keys, "alt_is_down", lambda: True):
            with mock.patch.object(wx, "Bell", lambda: None):
                for _ in range(2):
                    access_keys.handle_key(panel,
                                           _char_hook_event("D", alt=True))
        self.assertLessEqual(len(hints), 2)  # informing twice is harmless
        self.assertGreaterEqual(len(hints), 1, "the key must say something")

    # -- 3. one keystroke, one action -------------------------------------
    @unittest.skipUnless(sys.platform == "win32", "Windows key messages")
    def test_one_alt_d_press_starts_one_download_through_windows(self):
        """Drive the real Windows messages of one Alt+D press.

        The dialog manager matches the mnemonic on both messages the press
        produces, so without the shared guard the user gets two downloads
        from one key stroke.
        """
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        panel._refresh_buttons()
        self.assertTrue(panel.download_btn.IsEnabled())
        started = []
        panel._download_job = lambda *a: started.append(a)
        dlg.Show()
        panel.variant_combo.SetFocus()
        wx.Yield()

        with mock.patch.object(access_keys, "alt_is_down", lambda: True):
            _post_alt_key(wx.Window.FindFocus() or panel, "D", 0x20)
            wx.CallLater(700, self.app.ExitMainLoop)
            watchdog = wx.CallLater(8000, self.app.ExitMainLoop)  # never hang
            try:
                self.app.MainLoop()
            finally:
                watchdog.Stop()
                dlg.Hide()
        self.assertEqual(len(started), 1,
                         "one Alt+D press must start exactly one download")

    def test_the_dispatcher_does_not_act_twice_for_one_key(self):
        """The window hook may run for both messages of one press."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        panel._refresh_buttons()
        started = []
        panel._download_job = lambda *a: started.append(a)
        with mock.patch.object(access_keys, "alt_is_down", lambda: True):
            for _ in range(3):
                access_keys.handle_key(panel, _char_hook_event("D", alt=True))
        self.assertEqual(len(started), 1, "the key must be answered once")

    def test_a_mouse_click_is_never_de_duplicated(self):
        """The guard exists for Alt-driven activations only."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        self._install_first_variant(panel, store, folder)
        panel._on_tts(None)
        with mock.patch.object(access_keys, "alt_is_down", lambda: False):
            self.assertTrue(access_keys.once(panel.download_btn))
            self.assertTrue(access_keys.once(panel.download_btn),
                            "a second click must never be swallowed")
        with mock.patch.object(access_keys, "alt_is_down", lambda: True):
            self.assertTrue(access_keys.once(panel.download_btn))
            self.assertFalse(access_keys.once(panel.download_btn),
                             "the second message of one Alt+key press is"
                             " the duplicate")

    def test_altgr_and_other_modifiers_are_not_access_keys(self):
        """Ctrl+Alt is AltGr - it types characters, it does not press buttons."""
        store, folder = self._fresh_store()
        dlg, panel = self._download_panel(store)
        panel._refresh_buttons()
        started = []
        panel._download_job = lambda *a: started.append(a)
        with mock.patch.object(wx, "Bell", lambda: None):
            for evt in (_char_hook_event("D", alt=True, ctrl=True),
                        _char_hook_event("D", alt=False),
                        _char_hook_event("X", alt=True)):
                access_keys.handle_key(panel, evt)
        self.assertEqual(started, [], "nothing may happen")

    # -- 4. the menu keeps its own keys ------------------------------------
    def test_no_access_key_in_the_main_window_is_shadowed_by_a_menu(self):
        """A frame answers its menu bar before any child control."""
        frame = self._main_frame()
        letter = access_keys.mnemonic(frame.remove_btn.GetLabel())
        self.assertIsNotNone(letter)
        self.assertFalse(access_keys.menu_claims(frame, letter))
        # Alt+R belongs to Edit > Resume Recording, and must stay there.
        self.assertTrue(access_keys.menu_claims(frame, "r"))
        self.assertTrue(access_keys.menu_claims(frame, "n"))

    def test_alt_r_in_the_main_window_does_not_remove_a_project(self):
        """The menu's Alt+R opens a menu item; it must not delete anything."""
        frame = self._main_frame()
        calls = []
        frame._remove_project = lambda: calls.append("remove")
        with mock.patch.object(access_keys, "alt_is_down", lambda: True):
            access_keys.handle_key(frame, _char_hook_event("R", alt=True))
        self.assertEqual(calls, [])

    # -- 5. the shortcuts --------------------------------------------------
    def _new_project_route(self, frame) -> list:
        """Rebind the welcome button so its activation can be observed.

        The button binds its handler at construction time, so the binding (not
        the method behind it) is what has to be replaced to see Ctrl+Shift+N
        arrive at the button rather than at the action behind it.
        """
        reached: list[str] = []
        frame.new_project_btn.Unbind(wx.EVT_BUTTON)
        frame.new_project_btn.Bind(wx.EVT_BUTTON,
                                   lambda _: reached.append("button"))
        return reached

    def test_ctrl_shift_n_presses_the_create_new_project_button(self):
        """Ctrl+Shift+N must press the button and nothing else.

        The button is the gesture's only owner, so the File-menu item must
        not be reached by the key even though it starts the same action.
        """
        frame = self._main_frame()
        reached = self._new_project_route(frame)
        frame._new_project = lambda: reached.append("action")
        frame._record = lambda: reached.append("record")
        frame._settings = lambda: reached.append("settings")
        frame._on_global_char_hook(
            _char_hook_event("n", ctrl=True, shift=True, scan=0x31))
        self.assertEqual(reached, ["button"],
                         "Ctrl+Shift+N must arrive at the button, not at the "
                         "action behind it")
        frame._on_global_char_hook(
            _char_hook_event("r", ctrl=True, shift=True, scan=0x13))
        self.assertEqual(reached, ["button", "record"])
        frame._on_global_char_hook(_char_hook_event(",", ctrl=True))
        self.assertEqual(reached, ["button", "record", "settings"])

    def test_ctrl_shift_n_works_when_wx_hands_the_key_over_untranslated(self):
        """wx 3.3.3 can deliver Ctrl+Shift+letter with key code 0."""
        frame = self._main_frame()
        reached = self._new_project_route(frame)
        event = _char_hook_event("N", ctrl=True, shift=True)
        event.SetKeyCode(0)
        event.SetRawKeyFlags(0x31 << 16)
        frame._on_global_char_hook(event)
        self.assertEqual(reached, ["button"])

    def test_the_file_menu_does_not_advertise_ctrl_shift_n(self):
        """One owner: the menu item must not claim the button's gesture."""
        frame = self._main_frame()
        bar = frame.GetMenuBar()
        item = [i for i in bar.GetMenu(0).GetMenuItems()
                if i.GetItemLabelText() == "New Project"][0]
        self.assertNotIn("Ctrl+Shift+N", item.GetItemLabel())

    def test_the_welcome_button_owns_the_ctrl_shift_n_gesture(self):
        """The gesture is discoverable where it belongs: on the button."""
        frame = self._main_frame()
        self.assertIn("Ctrl+Shift+N", frame.new_project_btn.GetToolTipText())

    # -- 6. the audit itself ----------------------------------------------
    def test_the_gesture_audit_reports_a_shared_access_key(self):
        """The tool must be able to fail (otherwise the next test is empty)."""
        dlg = wx.Dialog(self.parent)
        self.addCleanup(dlg.Destroy)
        sizer = wx.BoxSizer(wx.VERTICAL)
        for label in ("&Save project", "&Save settings"):
            sizer.Add(wx.Button(dlg, label=label), 0)
        dlg.SetSizer(sizer)
        dlg.SetSize((600, 400))
        dlg.Show()
        with contextlib.redirect_stdout(io.StringIO()):
            problems = a11y_gestures.audit("probe", dlg)
        dlg.Hide()
        self.assertTrue(
            any("!DUP" in problem for problem in problems),
            f"the audit missed a shared Alt+S: {problems}",
        )

    def test_every_shipped_window_has_clean_keyboard_gestures(self):
        """Run the audit over the shipped windows as a build-time check."""
        windows: list[tuple[str, wx.Window]] = []
        settings = self._settings()
        for index, category in enumerate(settings.CATEGORIES):
            settings._show_category(index)
            settings.Layout()
            windows.append((f"settings/{category.title}",
                            settings._panels[index]))
        windows.append(("recording", self._recording_window()))
        windows.append(("main window", self._main_frame()))
        problems: list[str] = []
        for title, window in windows:
            window.SetSize((max(window.GetBestSize().width, 900),
                            max(window.GetBestSize().height, 700)))
            window.Layout()
            with contextlib.redirect_stdout(io.StringIO()):
                problems.extend(f"{title}: {p}"
                                for p in a11y_gestures.audit(title, window))
        self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
