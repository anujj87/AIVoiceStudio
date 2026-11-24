"""What a screen reader is told a window is called.

Two screen-reader commands are answered from names the application has to
supply.  NVDA+T reads the title of the active window, and the *report the
application version* gesture reads the executable's own name and version.

The first used to answer "frame": wxWidgets names every window after its own
class ("frame", "dialog", "listBox"), ``finalize_accessibility`` copied that
leftover name into the MSAA accName, and the window's real title never reached
the screen reader.  A window's accName is its title - read live, because the
describe and transcribe windows rename themselves as a project opens - and a
control nobody named must not borrow its class name either.

The executable's version resource is covered by ``tests/test_packaging.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import unittest

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wx_test_app import get_app

from ai_voice_studio.constants import APP_NAME
from ai_voice_studio.gui import a11y
from ai_voice_studio.gui.main_frame import MainFrame
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore

#: ``wx.Something(`` - how the application's own source names a widget class.
_WIDGET_CALL = re.compile(r"\bwx\.([A-Z]\w*)\s*\(")


def _control_classes() -> list[type]:
    """Every ``wx`` widget class the application instantiates.

    Read out of the application's source so that a control added to a dialog
    is swept by the test below without anybody remembering to list it here.
    """
    found: set[type] = set()
    for folder, _dirs, files in os.walk(os.path.join(ROOT, "ai_voice_studio")):
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                text = handle.read()
            for match in _WIDGET_CALL.finditer(text):
                cls = getattr(wx, match.group(1), None)
                if isinstance(cls, type) and issubclass(cls, wx.Window):
                    found.add(cls)
    return sorted(found, key=lambda cls: cls.__name__)


class WindowNameTest(unittest.TestCase):
    """The frame, and every dialog, answer with their title."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.parent = wx.Frame(None, title="Parent")

    @classmethod
    def tearDownClass(cls):
        cls.parent.Destroy()

    def _empty_store(self) -> ModelStore:
        fd, path = tempfile.mkstemp(prefix="aivs_names_", suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return ModelStore(state_file=path)

    def _main_frame(self) -> MainFrame:
        folder = tempfile.mkdtemp(prefix="aivs_names_")
        self.addCleanup(shutil.rmtree, folder, True)
        frame = MainFrame(
            settings=Settings(path=os.path.join(folder, "settings.json")),
            store=self._empty_store(),
        )
        self.addCleanup(frame.Destroy)
        return frame

    def test_the_main_window_is_announced_as_the_application(self):
        frame = self._main_frame()
        self.assertIsNotNone(frame.GetAccessible())
        self.assertEqual(frame.GetAccessible().GetName(0), (wx.ACC_OK, APP_NAME),
                         "NVDA+T would announce the window's class name")

    def test_a_dialog_is_announced_with_its_title(self):
        dlg = wx.Dialog(self.parent, title="Settings")
        self.addCleanup(dlg.Destroy)
        a11y.finalize_accessibility(dlg)
        self.assertEqual(dlg.GetAccessible().GetName(0), (wx.ACC_OK, "Settings"))

    def test_a_window_that_renames_itself_reports_the_new_title(self):
        # The describe window is built as "Describe" and then becomes
        # "Audio description - <film>"; the accName has to follow.
        dlg = wx.Dialog(self.parent, title="Describe")
        self.addCleanup(dlg.Destroy)
        a11y.finalize_accessibility(dlg)
        dlg.SetTitle("Audio description - Film")
        self.assertEqual(dlg.GetAccessible().GetName(0),
                         (wx.ACC_OK, "Audio description - Film"))

    def test_an_unnamed_control_does_not_borrow_its_class_name(self):
        panel = wx.Panel(self.parent)
        self.addCleanup(panel.Destroy)
        unnamed = wx.ListBox(panel)
        a11y.finalize_accessibility(panel)
        self.assertIsNone(unnamed.GetAccessible(),
                          "the listbox would be announced as \"listBox\"")

    def test_a_named_control_keeps_its_name(self):
        panel = wx.Panel(self.parent)
        self.addCleanup(panel.Destroy)
        named = wx.ListBox(panel)
        named.SetName("Recent projects list")
        a11y.finalize_accessibility(panel)
        self.assertEqual(named.GetAccessible().GetName(0),
                         (wx.ACC_OK, "Recent projects list"))

    def test_every_default_name_wx_uses_is_known(self):
        """The list of wx's own window names must cover this wx build.

        A name wxWidgets invented for a control is not an accessible name; if
        a widget's default is missing from ``a11y._DEFAULT_WINDOW_NAMES`` the
        application would announce it as if the label had been "check" or
        "panel".
        """
        checked, unconstructible = 0, []
        for cls in _control_classes():
            if issubclass(cls, wx.TopLevelWindow):
                continue  # a window is named by its title
            try:
                control = cls(self.parent)
            except Exception:
                unconstructible.append(cls.__name__)
                continue
            try:
                checked += 1
                with self.subTest(widget=cls.__name__):
                    self.assertFalse(
                        a11y._named_by_the_app(control),
                        f"{cls.__name__} starts life named "
                        f"{control.GetName()!r}, which "
                        "_DEFAULT_WINDOW_NAMES does not know",
                    )
                    control.SetName("A name the application chose")
                    self.assertTrue(a11y._named_by_the_app(control))
            finally:
                control.Destroy()
        self.assertGreater(checked, 10,
                           f"only {checked} widget classes were swept; "
                           f"unconstructible: {unconstructible}")


if __name__ == "__main__":
    unittest.main()
