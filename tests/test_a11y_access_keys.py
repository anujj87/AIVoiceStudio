"""Access keys (Alt+letter) and accessible-name coverage.

The application is audited by ``tools/a11y_audit.py`` (screen-reader names),
``tools/a11y_adjacency.py`` (visible label vs accessible name) and
``tools/a11y_mnemonics.py`` (access keys).  Those tools are run by hand; the
checks below turn the parts that must not regress into build-time failures:

* the access keys the user asked for (Alt+P preview, Alt+S start recording,
  Alt+D download selected), and that the Download and remove page's Remove
  button deliberately carries none,
* Alt+R no longer starts a recording: the main window's Edit menu entry
  (&Resume Recording) owns that key, which is what made the key move the
  focus instead of starting a run,
* no two *visible* buttons in one window claim the same Alt+key - Windows
  would fire only the first of them and the other would be unreachable,
* the mnemonic marker never leaks into an accessible name (Windows strips
  the '&' when it derives a button's name from its text),
* the first-launch terms checkboxes are each announced with their own term
  instead of six identical "check box" announcements,
* the "start selected recording" combo announces exactly the sentence that
  is printed above it.
"""

from __future__ import annotations

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

from a11y_mnemonics import access_key  # noqa: E402

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.constants import MODE_PAGE_WITH_H1  # noqa: E402
from ai_voice_studio.gui import access_keys  # noqa: E402
from ai_voice_studio.gui.accept_dialog import AcceptanceDialog  # noqa: E402
from ai_voice_studio.gui.main_frame import (  # noqa: E402
    MainFrame,
    _picker_entries,
    _StartRecordingDialog,
)
from ai_voice_studio.gui.recording_dialog import RecordingDialog  # noqa: E402
from ai_voice_studio.gui.settings_dialog import SettingsDialog  # noqa: E402
from ai_voice_studio.gui import theme as app_theme  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402


def _contrast(colour_a: str, colour_b: str) -> float:
    """WCAG 2.x contrast ratio between two ``#rrggbb`` colours."""
    def luminance(colour: str) -> float:
        raw = colour.lstrip("#")
        channels = [int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return (0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])

    lower, higher = sorted((luminance(colour_a), luminance(colour_b)))
    return (higher + 0.05) / (lower + 0.05)


def _visible(window: wx.Window) -> bool:
    win = window
    while win is not None:
        if not win.IsShown():
            return False
        win = win.GetParent()
    return True


def _buttons(window: wx.Window) -> list:
    """Every visible button of ``window`` (its own children, recursively)."""
    found = []

    def walk(container: wx.Window) -> None:
        for child in container.GetChildren():
            if not isinstance(child, wx.Window):
                continue
            if isinstance(child, wx.Button) and _visible(child):
                found.append(child)
            if not isinstance(child, (wx.ComboBox, wx.Choice, wx.SpinCtrl,
                                      wx.SpinCtrlDouble, wx.RadioBox)):
                walk(child)

    walk(window)
    return found


class _AppMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = wx.App(False)
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def _empty_store(self) -> ModelStore:
        fd, path = tempfile.mkstemp(prefix="aivs_models_", suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return ModelStore(state_file=path)

    def _settings(self) -> SettingsDialog:
        dlg = SettingsDialog(self.frame, Settings(), self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg

    def _recording_window(self) -> RecordingDialog:
        tmp = tempfile.mkdtemp(prefix="aivs_a11y_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Access key project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
        )
        dlg = RecordingDialog(self.frame, tmp, Settings(), self._empty_store())
        self.addCleanup(dlg.Destroy)
        return dlg

    def _main_frame(self) -> MainFrame:
        tmp = tempfile.mkdtemp(prefix="aivs_a11y_")
        self.addCleanup(shutil.rmtree, tmp, True)
        settings = Settings(path=os.path.join(tmp, "settings.json"))
        settings.add_recent_project("Access key project", tmp)
        frame = MainFrame(settings=settings, store=self._empty_store())
        self.addCleanup(frame.Destroy)
        return frame

    # ------------------------------------------------------- requested keys
    def test_alt_p_previews_in_every_category_that_offers_a_preview(self):
        """Preview is Alt+P everywhere a preview button exists."""
        dlg = self._settings()
        dlg.Layout()
        checked = 0
        for name in ("available_panel", "recording_panel", "punctuation_panel",
                     "omnivoice_engines_panel", "voice_clone_panel"):
            panel = getattr(dlg, name)
            button = getattr(panel, "preview_btn", None)
            if button is None:
                continue
            checked += 1
            self.assertEqual(
                access_key(button.GetLabel()), "p",
                f"{name} preview button has no Alt+P access key",
            )
        self.assertGreaterEqual(checked, 4)

    def test_alt_s_starts_recording(self):
        dlg = self._recording_window()
        self.assertEqual(access_key(dlg.start_btn.GetLabel()), "s")
        self.assertIn("Start recording", dlg.start_btn.GetLabel())

    def test_alt_d_downloads_and_remove_carries_no_access_key(self):
        dlg = self._settings()
        panel = dlg.download_panel
        self.assertEqual(access_key(panel.download_btn.GetLabel()), "d")
        # Remove has no Alt key on purpose: the key looked available but was
        # not dependable, and a key that does nothing is worse than none.
        self.assertIsNone(access_key(panel.remove_btn.GetLabel()),
                          "the Remove button must advertise no access key")
        self.assertNotIn(
            panel.remove_btn.GetLabel(),
            [entry.label for entry in access_keys.access_keys(panel)],
            "no Alt+letter may resolve to the Remove button",
        )
        # The main window's own red button cannot use Alt+R: in a frame the
        # menu bar answers first (&Resume Recording), so the key would open a
        # menu instead of removing anything.  Its letter must be one no menu
        # item claims - that is what the gesture audit checks for every window.
        frame = self._main_frame()
        letter = access_key(frame.remove_btn.GetLabel())
        self.assertIsNotNone(letter)
        self.assertFalse(
            access_keys.menu_claims(frame, letter),
            f"Alt+{letter.upper()} on the welcome remove button is shadowed by "
            "a menu item, so the label advertises a key that cannot reach it",
        )

    # -------------------------------------------------- no duplicate keys
    def test_no_two_visible_buttons_share_an_access_key(self):
        windows: list[tuple[str, wx.Window]] = []
        settings = self._settings()
        for index, category in enumerate(settings.CATEGORIES):
            settings._show_category(index)
            settings.Layout()
            windows.append((f"settings/{category.title}", settings._panels[index]))
        windows.append(("recording", self._recording_window()))
        windows.append(("main frame", self._main_frame()))
        for title, window in windows:
            window.SetSize((max(window.GetBestSize().width, 900),
                            max(window.GetBestSize().height, 700)))
            window.Layout()
            seen: dict[str, str] = {}
            for button in _buttons(window):
                key = access_key(button.GetLabel())
                if key is None:
                    continue
                label = button.GetLabel()
                self.assertNotIn(
                    key, seen,
                    f"{title}: Alt+{key.upper()} is claimed by both "
                    f"{label!r} and {seen.get(key)!r}",
                )
                seen[key] = label

    def test_the_mnemonic_marker_never_reaches_an_accessible_name(self):
        windows: list[wx.Window] = [self._recording_window(), self._main_frame()]
        settings = self._settings()
        for index in range(len(settings.CATEGORIES)):
            settings._show_category(index)
            windows.append(settings._panels[index])
        windows.append(AcceptanceDialog(self.frame))
        for window in windows:
            for button in _buttons(window):
                self.assertNotIn("&", button.GetName(),
                                 f"{button.GetLabel()!r} leaks '&' into its name")

    # --------------------------------------------------- first-launch terms
    def test_every_terms_checkbox_is_announced_with_its_own_term(self):
        dlg = AcceptanceDialog(self.frame)
        self.addCleanup(dlg.Destroy)
        names = [box.GetName() for box in dlg._checkboxes]
        self.assertEqual(len(names), 7)
        for name in names:
            self.assertNotIn(name.lower(), ("", "check", "checkbox"))
        self.assertEqual(len(set(names)), len(names))
        # The numbers make it possible to say *which* of the six terms is
        # focused, and the text is the term itself.
        self.assertTrue(names[0].startswith("Term 1:"))
        self.assertEqual(names[-1], "I agree with all above terms")

    # ------------------------------------------------------- picker wording
    def test_the_picker_combo_announces_its_visible_label(self):
        entries = [
            {"position": 0, "title": "01 chapter 1", "file": "01 chapter 1.wav"},
            {"position": 1, "title": "02 chapter 2", "file": None},
        ]
        dlg = _StartRecordingDialog(self.frame, entries)
        self.addCleanup(dlg.Destroy)
        self.assertEqual(dlg.combo.GetName(),
                         dlg.choice_label.GetLabel().rstrip(":"))
        dlg.break_radio.SetValue(True)
        dlg._on_mode(None)
        self.assertEqual(dlg.combo.GetName(),
                         dlg.choice_label.GetLabel().rstrip(":"))
        self.assertEqual(dlg.combo.GetName(),
                         "Choose which file to record by file break")

    # ------------------------------------------------- Alt+S really starts
    def _start_counting_window(self) -> tuple[RecordingDialog, dict]:
        """A recording window whose recording body only counts calls.

        ``_begin_recording`` is the real start handler's body (a worker is
        never spawned here), and it is an attribute so the count survives
        every path into it.
        """
        dlg = self._recording_window()
        calls = {"n": 0}
        dlg._begin_recording = lambda: calls.__setitem__("n", calls["n"] + 1)
        return dlg, calls

    def test_alt_s_hook_starts_the_recording(self):
        """Alt+S reaches the start handler even without the mnemonic path.

        The button's own mnemonic is answered by Windows; this handler is
        the independent path that survives a keyboard layout or a window
        state where Windows does not match '&Start recording'.
        """
        dlg, calls = self._start_counting_window()
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetAltDown(True)
        evt.SetKeyCode(ord("S"))
        evt.SetRawKeyFlags(0x1F << 16)
        dlg._on_char_hook(evt)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(evt.GetSkipped(), "the access key must be consumed")

    def test_alt_s_is_recognised_from_an_untranslated_key_event(self):
        """wx 3.3.3 can hand Alt+letter over with key code 0."""
        dlg, calls = self._start_counting_window()
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetAltDown(True)
        evt.SetKeyCode(0)
        evt.SetRawKeyFlags(0x1F << 16)  # the S key's scan code
        dlg._on_char_hook(evt)
        self.assertEqual(calls["n"], 1)

    def test_the_old_alt_r_no_longer_starts_a_recording(self):
        """Alt+R belongs to Edit > Resume Recording, not to this window."""
        dlg, calls = self._start_counting_window()
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetAltDown(True)
        evt.SetKeyCode(ord("R"))
        evt.SetRawKeyFlags(0x13 << 16)
        dlg._on_char_hook(evt)
        self.assertEqual(calls["n"], 0)
        self.assertTrue(evt.GetSkipped(), "Alt+R must be passed on")

    def test_the_hook_ignores_every_other_key(self):
        dlg, calls = self._start_counting_window()
        others = [
            ("plain S", False, False, ord("S"), 0x1F << 16),
            ("Ctrl+Alt+S (AltGr)", True, True, ord("S"), 0x1F << 16),
            ("Alt+X", True, False, ord("X"), 0x2D << 16),
            ("Alt with no letter", True, False, 0, 0x38 << 16),
        ]
        for name, alt, ctrl, key, raw in others:
            evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
            evt.SetAltDown(alt)
            evt.SetControlDown(ctrl)
            evt.SetKeyCode(key)
            evt.SetRawKeyFlags(raw)
            dlg._on_char_hook(evt)
            self.assertEqual(calls["n"], 0, f"{name} must not start recording")
            self.assertTrue(evt.GetSkipped(), f"{name} must be passed on")

    def test_one_keystroke_cannot_start_two_runs(self):
        """Alt+S fires the mnemonic *and* the hook; only one may start."""
        dlg, calls = self._start_counting_window()
        dlg._on_start(None)   # the Windows mnemonic path
        dlg._on_start(None)   # and the CHAR_HOOK path for the same keystroke
        self.assertEqual(calls["n"], 1)
        # A deliberate press later on is not swallowed.
        dlg._start_requested_at = 0.0
        dlg._on_start(None)
        self.assertEqual(calls["n"], 2)

    def test_a_disabled_start_button_is_not_triggered_by_the_key(self):
        dlg, calls = self._start_counting_window()
        dlg.start_btn.Disable()
        evt = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evt.SetAltDown(True)
        evt.SetKeyCode(ord("S"))
        dlg._on_char_hook(evt)
        self.assertEqual(calls["n"], 0)

    def test_a_start_that_fails_says_so(self):
        """A silent failure is what 'the shortcut does nothing' looks like."""
        dlg = self._recording_window()

        def boom():
            raise RuntimeError("no engine available")

        dlg._begin_recording = boom
        shown = []
        with mock.patch.object(wx, "MessageBox",
                               lambda *a, **k: shown.append(a)):
            dlg._on_start(None)
        self.assertEqual(len(shown), 1)
        self.assertIn("Could not start recording", shown[0][0])
        self.assertIn("no engine available", shown[0][0])
        self.assertIn("Could not start recording", dlg.status.GetLabel())

    # ------------------------------------------------- the real Windows path
    @unittest.skipUnless(sys.platform == "win32", "Windows key messages")
    def test_alt_s_reaches_the_start_handler_through_windows(self):
        """Drive the real Alt+S messages: WM_SYSKEYDOWN + WM_SYSCHAR.

        This is the path the user's keyboard takes (the mnemonic is answered
        by the Windows dialog manager, which fires on both messages), and it
        is the guard that one keystroke starts exactly one recording.
        """
        import ctypes

        dlg, calls = self._start_counting_window()
        dlg.Show()
        self.addCleanup(dlg.Destroy)

        user32 = ctypes.windll.user32
        ALT = 1 << 29
        scan = 0x1F
        try:
            hwnd = wx.Window.FindFocus().GetHandle()
        except Exception:  # noqa: BLE001
            hwnd = dlg.GetHandle()

        state = {"posted": False}

        def post_and_stop():
            user32.PostMessageW(hwnd, 0x0104, 0x53, 1 | (scan << 16) | ALT)
            user32.PostMessageW(hwnd, 0x0106, ord("s"),
                                1 | (scan << 16) | ALT | (1 << 31))
            user32.PostMessageW(hwnd, 0x0105, 0x53,
                                1 | (scan << 16) | ALT | (1 << 30) | (1 << 31))
            state["posted"] = True
            wx.CallLater(700, self.app.ExitMainLoop)

        wx.CallLater(400, post_and_stop)
        watchdog = wx.CallLater(8000, self.app.ExitMainLoop)  # never hang
        try:
            self.app.MainLoop()
        finally:
            watchdog.Stop()
        self.assertTrue(state["posted"], "the probe never ran")
        self.assertEqual(calls["n"], 1)

    # -------------------------------------------------------------- contrast
    def test_both_themes_meet_wcag_aa_contrast(self):
        """The themed text colours stay legible (4.5:1 is WCAG AA)."""
        for name, colours in (("light", app_theme._LIGHT),
                              ("dark", app_theme._DARK)):
            for fg_key, bg_key in (("fg", "bg"),
                                   ("control_fg", "control_bg")):
                ratio = _contrast(colours[fg_key], colours[bg_key])
                self.assertGreaterEqual(
                    ratio, 4.5,
                    f"{name} theme {fg_key} on {bg_key} is only {ratio:.1f}:1",
                )

    def test_picker_entries_are_what_the_window_shows(self):
        """The picker's combo names come from the project, not the title."""
        tmp = tempfile.mkdtemp(prefix="aivs_a11y_")
        self.addCleanup(shutil.rmtree, tmp, True)
        project.create_project(
            tmp, "Picker project", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "01 chapter 1", "text": "One."},
             {"index": 2, "title": "02 chapter 2", "text": "Two."}],
        )
        data = project.load_project(tmp)
        entries = _picker_entries(data, {})
        self.assertEqual([e["title"] for e in entries],
                         ["01 chapter 1", "02 chapter 2"])


if __name__ == "__main__":
    unittest.main()
