"""Settings opens in under a second: categories are built when first visited.

The dialog used to construct all seventeen categories up front - 392 window
handles and 1.8-2.0 s of GUI-thread work before the first control could be
announced, with the heaviest category (Voice Clone, both halves) never built
belonging to a user who only came to change the theme.

These tests lock the new contract:

* opening builds **only** the first category, and does it in well under a
  second;
* a category is built on first visit, once;
* every category still builds its controls, with the same titles and the same
  MSAA names / theme as before;
* a category the user never opens **writes nothing**, so the values it owns
  stay exactly as they were through Apply, OK and Cancel - the reason this is
  safe rather than a shortcut;
* a category that *is* visited still validates and still saves.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wx_test_app import get_app

from ai_voice_studio.gui import theme as theme_module
from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore

#: Settings owned by categories far down the list, used to prove that a
#: category nobody opened neither loses nor rewrites what it owns.
FAR_AWAY_VALUES = {
    "daisy.language": "de",
    "daisy.publisher": "Testverlag",
    "omnivoice_server.port": 9911,
    "recording.punctuation": "none",
    "audio_mode_pages_per_file": 37,
    "hf.token": "hf_neverOpenThisCategory",
    "gestures.bindings": {"record": "Ctrl+Shift+R"},
}


class SettingsLazyPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None, title="Settings parent")

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aivs_lazy_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.settings_path = os.path.join(self.folder, "settings.json")
        self.store = ModelStore(state_file=os.path.join(self.folder, "models.json"))

    def _settings(self, **values) -> Settings:
        settings = Settings(path=self.settings_path)
        for key, value in values.items():
            settings.set(key.replace("__", "."), value)
        settings.save()
        return settings

    def _dialog(self, settings: Settings | None = None) -> SettingsDialog:
        dlg = SettingsDialog(self.frame, settings or self._settings(), self.store)
        self.addCleanup(dlg.Destroy)
        return dlg

    def _reset_index(self, dlg: SettingsDialog) -> int:
        return next(index for index, cls in enumerate(dlg.CATEGORIES)
                    if cls.title == "Reset")

    def _category_of(self, dlg: SettingsDialog, title_start: str) -> int:
        return next(index for index, cls in enumerate(dlg.CATEGORIES)
                    if (cls.title or "").startswith(title_start))

    # ---------------------------------------------------------------- opening
    def test_opening_builds_only_the_first_category(self):
        dlg = self._dialog()

        self.assertEqual(len(dlg.built_panels()), 1)
        self.assertIs(dlg.built_panels()[0], dlg.general_panel)
        self.assertEqual(dlg.general_panel.title, "General")
        # The others exist as slots, not as windows.
        self.assertEqual(len(dlg._panels), len(dlg.CATEGORIES))
        self.assertTrue(all(panel is None for panel in dlg._panels[1:]))

    def test_opening_the_settings_dialog_takes_under_a_second(self):
        """The whole point: a settings window ready in well under a second.

        Measured over several opens and compared on the median, so one slow
        moment on a busy machine cannot decide it either way.
        """
        settings = self._settings()
        opens = []
        for _ in range(5):
            start = time.perf_counter()
            dlg = SettingsDialog(self.frame, settings, self.store)
            opens.append(time.perf_counter() - start)
            dlg.Destroy()

        median = sorted(opens)[len(opens) // 2]
        shown = ", ".join(f"{opened:.2f}" for opened in opens)
        self.assertLess(median, 1.0,
                        f"opening Settings took {median:.2f}s (opens: {shown})")

    def test_the_category_list_is_complete_without_building_the_panels(self):
        dlg = self._dialog()

        self.assertEqual(dlg.cat_list.GetItemCount(), len(dlg.CATEGORIES))
        self.assertEqual([dlg.cat_list.GetItemText(i)
                          for i in range(dlg.cat_list.GetItemCount())],
                         [cls.title for cls in dlg.CATEGORIES])
        self.assertEqual(len(dlg.built_panels()), 1)

    # -------------------------------------------------------------- visiting
    def test_a_category_is_built_on_first_visit_only(self):
        dlg = self._dialog()
        index = self._category_of(dlg, "Voice Clone")

        dlg._show_category(index)
        first = dlg._panels[index]

        self.assertIsNotNone(first)
        self.assertEqual(len(dlg.built_panels()), 2)
        self.assertTrue(first.GetChildren())

        dlg._show_category(0)
        dlg._show_category(index)

        self.assertIs(dlg._panels[index], first,
                      "the category was built again on the second visit")

    def test_only_the_shown_category_is_visible(self):
        dlg = self._dialog()
        index = self._category_of(dlg, "Appearance")

        dlg._show_category(index)

        self.assertFalse(dlg.general_panel.IsShown())
        self.assertTrue(dlg.ensure_panel(index).IsShown())
        # A panel nobody visited is not on screen either.
        later = self._category_of(dlg, "Compute")
        self.assertFalse(dlg.ensure_panel(later).IsShown())

    def test_every_category_still_builds_its_controls(self):
        dlg = self._dialog()

        panels = dlg.panels()

        self.assertEqual(len(panels), len(dlg.CATEGORIES))
        for panel in panels:
            self.assertTrue(panel.GetChildren(),
                            f"{panel.title} has no children")
            self.assertEqual(panel.title,
                             dlg.CATEGORIES[dlg._panels.index(panel)].title)

    def test_a_later_panel_gets_real_accessible_names(self):
        dlg = self._dialog()
        panel = dlg.ensure_panel(self._category_of(dlg, "DAISY"))

        named = [child for child in panel.GetChildren()
                 if getattr(child, "_acc_name", None)]
        self.assertTrue(named, "a panel built on first visit has no accNames")
        for child in named:
            code, name = child.GetAccessible().GetName(0)
            self.assertEqual(code, wx.ACC_OK)
            self.assertEqual(name, child._acc_name)

    def test_a_later_panel_is_painted_with_the_current_theme(self):
        dlg = self._dialog(self._settings(theme="dark"))

        panel = dlg.ensure_panel(self._category_of(dlg, "DAISY"))

        texts = [child for child in panel.GetChildren()
                 if isinstance(child, wx.StaticText)]
        self.assertTrue(texts, "no text in the DAISY category to check")
        expected = theme_module.current_theme_colors()["fg"]
        self.assertEqual(texts[0].GetForegroundColour(), expected)

    # ------------------------------------------------- never-opened values
    def test_values_of_categories_never_opened_survive_apply(self):
        settings = self._settings(**FAR_AWAY_VALUES)
        dlg = self._dialog(settings)

        self.assertTrue(dlg._apply_now())

        reopened = Settings(path=self.settings_path)
        for key, value in FAR_AWAY_VALUES.items():
            self.assertEqual(reopened.get(key), value,
                             f"{key} was lost by a category nobody opened")

    def test_values_of_categories_never_opened_survive_ok(self):
        settings = self._settings(**FAR_AWAY_VALUES)
        dlg = self._dialog(settings)

        self.assertTrue(dlg._save_from_ui())

        reopened = Settings(path=self.settings_path)
        self.assertEqual(reopened.get("daisy.language"), "de")
        self.assertEqual(reopened.get("omnivoice_server.port"), 9911)

    def test_values_of_categories_never_opened_survive_cancel(self):
        settings = self._settings(**FAR_AWAY_VALUES)
        dlg = self._dialog(settings)

        with mock.patch.object(dlg, "EndModal"):
            dlg._on_cancel(None)

        reopened = Settings(path=self.settings_path)
        self.assertEqual(reopened.get("daisy.publisher"), "Testverlag")
        self.assertEqual(reopened.get("daisy.language"), "de")

    def test_visiting_a_category_does_not_rewrite_what_was_not_touched(self):
        """The other half of the invariant: a *visited* panel round-trips.

        Every category is shown (the Ctrl+Tab path a keyboard user takes) and
        Apply is pressed.  A panel that seeds its controls from the settings
        must write the same values back, otherwise opening a category and
        saving would quietly rewrite it.
        """
        settings = self._settings(**FAR_AWAY_VALUES)
        dlg = self._dialog(settings)

        for index in range(dlg.panel_count()):
            dlg._show_category(index)
        self.assertEqual(len(dlg.built_panels()), len(dlg.CATEGORIES))
        self.assertTrue(dlg._apply_now())

        reopened = Settings(path=self.settings_path)
        for key in ("daisy.language", "daisy.publisher", "omnivoice_server.port",
                    "recording.punctuation", "hf.token"):
            self.assertEqual(reopened.get(key), FAR_AWAY_VALUES[key],
                             f"{key} was rewritten by a category that was only visited")

    def test_an_unopened_category_writes_nothing_at_all(self):
        settings = self._settings(**FAR_AWAY_VALUES)
        dlg = self._dialog(settings)
        before = settings.as_dict()

        dlg._apply_now()

        # Compare the categories that were never built, key by key.
        after = Settings(path=self.settings_path).as_dict()
        for section in ("daisy", "omnivoice_server", "recording", "gestures"):
            self.assertEqual(after.get(section), before.get(section),
                             f"the {section} category was rewritten")

    # -------------------------------------------------------- visited values
    def test_a_visited_category_still_saves_its_change(self):
        settings = self._settings(updates={"auto_check": True})
        dlg = self._dialog(settings)
        general = dlg.general_panel
        general.update_auto_cb.SetValue(False)

        self.assertTrue(dlg._apply_now())

        self.assertFalse(Settings(path=self.settings_path).get("updates.auto_check"))

    def test_validation_still_blocks_for_a_visited_category(self):
        settings = self._settings(updates={"auto_check": True})
        dlg = self._dialog(settings)
        general = dlg.general_panel
        general.update_auto_cb.SetValue(False)

        with mock.patch.object(type(general), "isValid", return_value=False):
            self.assertFalse(dlg._apply_now())

        self.assertTrue(Settings(path=self.settings_path).get("updates.auto_check"),
                        "a panel that failed validation still wrote to disk")

    def test_an_unvisited_category_cannot_block_the_save(self):
        # It has no controls, so there is nothing to be invalid about; its
        # values are the ones already stored, validated when they were written.
        settings = self._settings(updates={"auto_check": True})
        dlg = self._dialog(settings)
        unvisited = [cls for index, cls in enumerate(dlg.CATEGORIES)
                     if dlg._panels[index] is None]
        self.assertTrue(unvisited)
        guards = {}
        for cls in unvisited:
            patcher = mock.patch.object(cls, "isValid", return_value=False)
            guards[cls] = patcher.start()
            self.addCleanup(patcher.stop)

        self.assertTrue(dlg._save_from_ui())

        for cls, guard in guards.items():
            self.assertFalse(guard.called,
                             f"{cls.title} was validated without being opened")

    # ------------------------------------------------------------------ reset
    def test_reset_drops_the_built_panels_and_they_return_with_defaults(self):
        settings = self._settings(theme="dark", daisy={"language": "de"})
        dlg = self._dialog(settings)
        dlg._show_category(self._category_of(dlg, "DAISY"))
        dlg._show_category(self._reset_index(dlg))

        dlg._settings_reset()

        self.assertEqual([panel.title for panel in dlg.built_panels()],
                         ["Reset"], "a reset must not leave stale panels up")
        self.assertNotEqual(settings.get("theme"), "dark")
        reopened = Settings(path=self.settings_path)
        self.assertEqual(reopened.get("daisy.language"),
                         Settings(path=self.settings_path).get("daisy.language"))

    def test_the_reset_category_uses_the_dialog_to_reset(self):
        settings = self._settings(theme="dark")
        dlg = self._dialog(settings)
        dlg._show_category(self._reset_index(dlg))
        reset_panel = dlg.ensure_panel(self._reset_index(dlg))

        with mock.patch("wx.MessageBox", return_value=wx.YES):
            reset_panel._on_reset(None)

        self.assertNotEqual(settings.get("theme"), "dark")
        self.assertEqual([panel.title for panel in dlg.built_panels()],
                         ["Reset"])

    def test_the_reset_button_is_only_a_question_and_a_default_reset(self):
        settings = self._settings(theme="dark")
        dlg = self._dialog(settings)
        dlg._show_category(self._reset_index(dlg))
        reset_panel = dlg.ensure_panel(self._reset_index(dlg))

        with mock.patch("wx.MessageBox", return_value=wx.NO):
            reset_panel._on_reset(None)

        self.assertEqual(settings.get("theme"), "dark")


class SettingsStorageTest(unittest.TestCase):
    """The settings file is not rewritten by merely opening and closing."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None, title="Settings parent")

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def test_closing_without_touching_anything_leaves_the_file_alone(self):
        folder = tempfile.mkdtemp(prefix="aivs_lazy_keep_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "settings.json")
        settings = Settings(path=path)
        settings.set("daisy.language", "fr")
        settings.save()
        with open(path, encoding="utf-8") as handle:
            before = json.load(handle)

        dlg = SettingsDialog(self.frame, settings,
                             ModelStore(state_file=os.path.join(folder, "m.json")))
        self.addCleanup(dlg.Destroy)
        with mock.patch.object(dlg, "EndModal"):
            dlg._on_cancel(None)

        with open(path, encoding="utf-8") as handle:
            after = json.load(handle)
        self.assertEqual(after["daisy"]["language"], "fr",
                         "Cancel lost a setting of a category nobody opened")
        self.assertEqual(after, before,
                         "closing the dialog changed the settings file")


if __name__ == "__main__":
    unittest.main()
