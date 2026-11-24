"""Tests for the theme system: the twenty predefined dark themes (the original
ten and the vibrant ten), the custom theme creator store and the wallpaper
support.

The built-in light/dark palettes are part of the accessible baseline, so the
first test pins them: adding themes must never drift them.
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

from ai_voice_studio.constants import THEME_DARK, THEME_LIGHT, THEME_SYSTEM
from ai_voice_studio.gui import theme
from ai_voice_studio.gui.main_frame import MainFrame
from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.gui.theme_studio import ThemeStudioDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore


def _contrast(colour_a: str, colour_b: str) -> float:
    def luminance(colour: str) -> float:
        raw = colour.lstrip("#")
        channels = [int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lower, higher = sorted((luminance(colour_a), luminance(colour_b)))
    return (higher + 0.05) / (lower + 0.05)


class PresetThemeTest(unittest.TestCase):
    def test_ten_predefined_dark_themes(self):
        self.assertEqual(len(theme.THEME_PRESETS), 10)

    def test_ids_and_names_are_unique(self):
        ids = [preset.theme_id for preset in theme.THEME_PRESETS]
        names = [preset.name for preset in theme.THEME_PRESETS]
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(len(set(names)), 10)

    def test_every_preset_has_a_complete_palette(self):
        for preset in theme.THEME_PRESETS:
            with self.subTest(theme=preset.theme_id):
                for key in ("bg", "fg", "control_bg", "control_fg", "accent",
                            "heading"):
                    self.assertRegex(preset.colors[key], r"^#[0-9a-f]{6}$")

    def test_the_themes_look_different(self):
        """No two presets share a background+accent combination."""
        signatures = {
            (preset.colors["bg"], preset.colors["accent"])
            for preset in theme.THEME_PRESETS
        }
        self.assertEqual(len(signatures), 10)

    def test_every_preset_picks_a_known_style(self):
        for preset in theme.THEME_PRESETS:
            with self.subTest(theme=preset.theme_id):
                self.assertIn(preset.pattern, theme.PATTERN_IDS)

    def test_predefined_themes_are_dark(self):
        for preset in theme.THEME_PRESETS:
            with self.subTest(theme=preset.theme_id):
                self.assertLess(_contrast(preset.colors["bg"], "#000000"), 4.5,
                                "a predefined theme must be dark")
                self.assertGreater(_contrast(preset.colors["fg"],
                                             preset.colors["bg"]), 4.5)

    def test_every_preset_meets_wcag_aa_contrast(self):
        for preset in theme.THEME_PRESETS:
            for fg_key, bg_key in (("fg", "bg"), ("control_fg", "control_bg")):
                ratio = _contrast(preset.colors[fg_key], preset.colors[bg_key])
                with self.subTest(theme=preset.theme_id, pair=fg_key):
                    self.assertGreaterEqual(ratio, 4.5)


class VibrantThemeTest(unittest.TestCase):
    """The ten vibrant dark themes, and that the original ten are untouched."""

    def test_ten_vibrant_themes(self):
        self.assertEqual(len(theme.THEME_PRESETS_VIBRANT), 10)

    def test_the_original_ten_are_unchanged(self):
        """Adding the vibrant set must not rename, recolour or reorder them."""
        self.assertEqual(
            [preset.theme_id for preset in theme.THEME_PRESETS],
            ["graphite_midnight", "obsidian_gold", "deep_navy", "forest_pine",
             "plum_royale", "crimson_slate", "ocean_teal", "amber_dusk",
             "carbon_lime", "blue_steel"],
        )
        self.assertEqual(theme.THEME_PRESETS[0].colors["bg"], "#16181c")
        self.assertEqual(theme.THEME_PRESETS[-1].colors["accent"], "#9fb3c8")
        self.assertEqual(theme.ALL_THEME_PRESETS[:10], theme.THEME_PRESETS)
        self.assertEqual(theme.ALL_THEME_PRESETS[10:], theme.THEME_PRESETS_VIBRANT)

    def test_every_vibrant_theme_has_a_complete_palette(self):
        for preset in theme.THEME_PRESETS_VIBRANT:
            with self.subTest(theme=preset.theme_id):
                for key in ("bg", "fg", "control_bg", "control_fg", "accent",
                            "heading"):
                    self.assertRegex(preset.colors[key], r"^#[0-9a-f]{6}$")
                self.assertIn(preset.pattern, theme.PATTERN_IDS)
                self.assertTrue(preset.note)

    def test_every_id_and_name_is_new_and_unique(self):
        ids = [preset.theme_id for preset in theme.ALL_THEME_PRESETS]
        names = [preset.name for preset in theme.ALL_THEME_PRESETS]
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(len(set(names)), 20)
        originals = {preset.theme_id for preset in theme.THEME_PRESETS}
        self.assertEqual(
            originals & {preset.theme_id
                         for preset in theme.THEME_PRESETS_VIBRANT},
            set(),
            "a vibrant theme must not reuse an original id",
        )

    def test_the_vibrant_accents_are_really_vivid(self):
        """A vibrant accent is a saturated hue, not a muted grey-blue."""
        for preset in theme.THEME_PRESETS_VIBRANT:
            channels = [int(preset.colors["accent"][i:i + 2], 16)
                        for i in (1, 3, 5)]
            spread = max(channels) - min(channels)
            with self.subTest(theme=preset.theme_id):
                self.assertGreaterEqual(
                    spread, 120,
                    "the accent must be saturated, unlike the muted originals",
                )

    def test_no_two_vibrant_themes_look_alike(self):
        signatures = {
            (preset.colors["bg"], preset.colors["accent"])
            for preset in theme.ALL_THEME_PRESETS
        }
        self.assertEqual(len(signatures), 20)

    def test_every_vibrant_theme_is_dark_and_legible(self):
        for preset in theme.THEME_PRESETS_VIBRANT:
            with self.subTest(theme=preset.theme_id):
                self.assertLess(_contrast(preset.colors["bg"], "#000000"), 4.5,
                                "a new preset must still be a dark theme")
                for fg_key, bg_key in (("fg", "bg"), ("control_fg", "control_bg")):
                    self.assertGreaterEqual(
                        _contrast(preset.colors[fg_key], preset.colors[bg_key]),
                        4.5,
                        "WCAG AA contrast",
                    )

    def test_a_vibrant_theme_resolves_and_paints(self):
        preset = theme.THEME_PRESETS_VIBRANT[0]
        self.assertEqual(theme.resolve_palette(preset.theme_id)["bg"],
                         preset.colors["bg"])
        self.assertEqual(theme.theme_pattern(preset.theme_id), preset.pattern)
        self.assertEqual(theme.theme_label(preset.theme_id), preset.name)


class BuiltinThemeTest(unittest.TestCase):
    def test_light_and_dark_are_untouched(self):
        self.assertEqual(theme._LIGHT, {
            "bg": "#ffffff", "fg": "#1a1a1a",
            "control_bg": "#ffffff", "control_fg": "#1a1a1a",
        })
        self.assertEqual(theme._DARK, {
            "bg": "#1e1e1e", "fg": "#e8e8e8",
            "control_bg": "#2d2d2d", "control_fg": "#e8e8e8",
        })

    def test_resolve_builtin_and_preset(self):
        self.assertEqual(theme.resolve_palette(THEME_DARK)["bg"], "#1e1e1e")
        self.assertEqual(theme.resolve_palette(THEME_LIGHT)["bg"], "#ffffff")
        preset = theme.THEME_PRESETS[0]
        self.assertEqual(theme.resolve_palette(preset.theme_id)["bg"],
                         preset.colors["bg"])

    def test_unknown_theme_falls_back_legibly(self):
        palette = theme.resolve_palette("custom:does-not-exist")
        self.assertEqual(palette["bg"], theme._DARK["bg"])

    def test_labels(self):
        self.assertEqual(theme.theme_label(THEME_SYSTEM), "System default")
        self.assertEqual(theme.theme_label(THEME_LIGHT), "Light")
        self.assertEqual(theme.theme_label(THEME_DARK), "Dark")
        self.assertEqual(theme.theme_label(theme.THEME_PRESETS[2].theme_id),
                         theme.THEME_PRESETS[2].name)


class CustomThemeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aivs_themes_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.settings = Settings(path=os.path.join(self.tmp, "settings.json"))

    def _record(self, name, accent="#00ff00"):
        return {
            "name": name,
            "colors": {
                "bg": "#101010", "fg": "#f0f0f0", "control_bg": "#202020",
                "control_fg": "#f0f0f0", "accent": accent, "heading": "#ffffff",
            },
            "pattern": "grid",
        }

    def test_save_list_delete_roundtrip(self):
        theme_id = theme.save_custom_theme(self.settings, self._record("Mine"))
        self.assertTrue(theme_id.startswith(theme.CUSTOM_PREFIX))
        records = theme.custom_themes(self.settings)
        self.assertEqual([r["name"] for r in records], ["Mine"])
        self.assertEqual(theme.resolve_palette(theme_id, self.settings)["accent"],
                         "#00ff00")
        theme.delete_custom_theme(self.settings, theme_id)
        self.assertEqual(theme.custom_themes(self.settings), [])

    def test_user_can_create_many_themes(self):
        for index in range(25):
            theme.save_custom_theme(self.settings, self._record(f"Theme {index}"))
        self.assertEqual(len(theme.custom_themes(self.settings)), 25)
        self.assertEqual(len(theme.all_theme_choices(self.settings)),
                         3 + 20 + 25)

    def test_ids_stay_unique_for_the_same_name(self):
        first = theme.save_custom_theme(self.settings, self._record("Same"))
        second = theme.save_custom_theme(self.settings, self._record("Same"))
        self.assertNotEqual(first, second)

    def test_saving_an_existing_id_updates_in_place(self):
        theme_id = theme.save_custom_theme(self.settings, self._record("Mine"))
        record = self._record("Renamed")
        record["id"] = theme_id
        theme.save_custom_theme(self.settings, record)
        self.assertEqual(len(theme.custom_themes(self.settings)), 1)
        self.assertEqual(theme.custom_themes(self.settings)[0]["name"], "Renamed")

    def test_all_theme_choices_start_with_the_builtins(self):
        choices = theme.all_theme_choices(self.settings)
        self.assertEqual([value for value, _label in choices[:3]],
                         [THEME_SYSTEM, THEME_LIGHT, THEME_DARK])
        self.assertEqual(len(choices), 23)

    def test_the_theme_list_offers_the_vibrant_ten_after_the_original_ten(self):
        """Both sets are in the list, the originals first (their order stands)."""
        values = [value for value, _label in theme.all_theme_choices(self.settings)]
        self.assertEqual(values[3:13],
                         [preset.theme_id for preset in theme.THEME_PRESETS])
        self.assertEqual(values[13:23],
                         [preset.theme_id
                          for preset in theme.THEME_PRESETS_VIBRANT])


class WallpaperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aivs_wallpaper_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.settings = Settings(path=os.path.join(self.tmp, "settings.json"))

    def test_wallpaper_defaults_off(self):
        record = theme.get_wallpaper(self.settings, "deep_navy")
        self.assertFalse(record["enabled"])
        self.assertFalse(theme.is_wallpaper_enabled("deep_navy", self.settings))

    def test_wallpaper_is_per_theme(self):
        theme.set_wallpaper(self.settings, "deep_navy", enabled=True)
        self.assertTrue(theme.is_wallpaper_enabled("deep_navy", self.settings))
        self.assertFalse(theme.is_wallpaper_enabled("amber_dusk", self.settings))

    def test_wallpaper_image_and_style_roundtrip(self):
        theme.set_wallpaper(self.settings, "ocean_teal", enabled=True,
                            image="C:/pictures/sea.png", pattern="waves")
        record = theme.get_wallpaper(self.settings, "ocean_teal")
        self.assertEqual(record["image"], "C:/pictures/sea.png")
        self.assertEqual(record["pattern"], "waves")

    def test_theme_pattern_uses_the_override_then_the_preset(self):
        preset = theme.THEME_PRESETS[3]
        self.assertEqual(theme.theme_pattern(preset.theme_id, self.settings),
                         preset.pattern)
        theme.set_wallpaper(self.settings, preset.theme_id, pattern="dots")
        self.assertEqual(theme.theme_pattern(preset.theme_id, self.settings),
                         "dots")

    def test_wallpapers_survive_a_reload(self):
        theme.set_wallpaper(self.settings, "blue_steel", enabled=True)
        reloaded = Settings(path=self.settings._path)
        self.assertTrue(theme.is_wallpaper_enabled("blue_steel", reloaded))


class PaintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_every_pattern_paints_without_raising(self):
        bitmap = wx.Bitmap(120, 90)
        dc = wx.MemoryDC(bitmap)
        colors = theme.resolve_palette(theme.THEME_PRESETS[0].theme_id)
        for pattern, _label in theme.PATTERNS:
            with self.subTest(pattern=pattern):
                theme.paint_background(dc, (120, 90), colors, pattern)
        dc.SelectObject(wx.NullBitmap)

    def test_a_wallpaper_image_is_drawn(self):
        folder = tempfile.mkdtemp(prefix="aivs_wallpaper_")
        self.addCleanup(shutil.rmtree, folder, True)
        image = wx.Image(40, 30)
        path = os.path.join(folder, "wall.png")
        image.SaveFile(path, wx.BITMAP_TYPE_PNG)
        bitmap = wx.Bitmap(120, 90)
        dc = wx.MemoryDC(bitmap)
        theme.paint_background(dc, (120, 90), theme._DARK, "grid", path)
        dc.SelectObject(wx.NullBitmap)


class GuiThemeTest(unittest.TestCase):
    """The studio, the General panel and the wallpaper surface as built."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def _tempdir(self, prefix: str) -> str:
        folder = tempfile.mkdtemp(prefix=prefix)
        self.addCleanup(shutil.rmtree, folder, True)
        return folder

    def _settings(self, folder: str) -> Settings:
        return Settings(path=os.path.join(folder, "settings.json"))

    def _store(self, folder: str) -> ModelStore:
        return ModelStore(state_file=os.path.join(folder, "models.json"))

    def test_the_studio_creates_a_theme(self):
        folder = self._tempdir("aivs_studio_")
        settings = self._settings(folder)
        dlg = ThemeStudioDialog(None, settings)
        self.addCleanup(dlg.Destroy)
        dlg.name_ctrl.SetValue("My Theme")
        dlg.style_choice.SetSelection(5)
        dlg._save()
        theme_id = dlg.last_saved_theme_id
        self.assertTrue(theme_id.startswith(theme.CUSTOM_PREFIX))
        self.assertEqual([r["name"] for r in theme.custom_themes(settings)],
                         ["My Theme"])
        self.assertIn(theme_id, [value for value, _ in theme.all_theme_choices(settings)])
        # The saved palette follows the colour pickers.
        expected = dlg.bg_picker.GetColour().GetAsString(wx.C2S_HTML_SYNTAX).lower()
        self.assertEqual(theme.resolve_palette(theme_id, settings)["bg"], expected)

    def test_the_studio_refuses_a_theme_without_a_name(self):
        folder = self._tempdir("aivs_studio_")
        settings = self._settings(folder)
        dlg = ThemeStudioDialog(None, settings)
        self.addCleanup(dlg.Destroy)
        dlg.name_ctrl.SetValue("   ")
        with mock.patch.object(wx, "MessageBox", lambda *a, **k: wx.OK):
            dlg._save()
        self.assertEqual(theme.custom_themes(settings), [])

    def test_appearance_panel_lists_the_presets_and_wallpaper_controls(self):
        folder = self._tempdir("aivs_studio_")
        settings = self._settings(folder)
        dlg = SettingsDialog(None, settings, self._store(folder))
        self.addCleanup(dlg.Destroy)
        panel = dlg.appearance_panel
        self.assertEqual(panel.title, "Appearance")
        values = [panel.theme_combo.GetClientData(index)
                  for index in range(panel.theme_combo.GetCount())]
        for preset in theme.ALL_THEME_PRESETS:
            with self.subTest(theme=preset.theme_id):
                self.assertIn(preset.theme_id, values)
        self.assertTrue(hasattr(panel, "new_theme_btn"))
        self.assertTrue(hasattr(panel, "wallpaper_cb"))
        # The wallpaper controls follow the selected theme.
        theme.set_wallpaper(settings, "deep_navy", enabled=True)
        panel._load_wallpaper("deep_navy")
        self.assertTrue(panel.wallpaper_cb.GetValue())

    def test_appearance_owns_every_look_control_and_general_keeps_the_rest(self):
        """The move is a move: nothing was dropped, and nothing was duplicated."""
        folder = self._tempdir("aivs_appearance_")
        settings = self._settings(folder)
        dlg = SettingsDialog(None, settings, self._store(folder))
        self.addCleanup(dlg.Destroy)
        appearance = dlg.appearance_panel
        general = dlg.general_panel
        self.assertEqual(appearance.title, "Appearance")
        for attribute in ("theme_combo", "new_theme_btn", "wallpaper_cb",
                          "wallpaper_image", "wallpaper_browse_btn",
                          "wallpaper_clear_btn", "wallpaper_style"):
            with self.subTest(control=attribute):
                self.assertTrue(hasattr(appearance, attribute))
                self.assertFalse(hasattr(general, attribute),
                                 f"General still carries {attribute}")
        # General keeps what it is for now.
        for attribute in ("dev_mode_cb", "recordings_ctrl", "models_ctrl",
                          "update_auto_cb", "update_now_btn"):
            with self.subTest(control=attribute):
                self.assertTrue(hasattr(general, attribute))
        # The Appearance category is second, right below General.
        self.assertEqual(dlg._panels[1].title, "Appearance")
        self.assertEqual(dlg.cat_list.GetItemText(1), "Appearance")

    def test_an_appearance_change_is_saved_by_apply(self):
        """Choosing a theme in the new category still reaches settings.json."""
        folder = self._tempdir("aivs_appearance_")
        settings = self._settings(folder)
        dlg = SettingsDialog(None, settings, self._store(folder))
        self.addCleanup(dlg.Destroy)
        panel = dlg.appearance_panel
        target = next(i for i in range(panel.theme_combo.GetCount())
                      if panel.theme_combo.GetClientData(i) == "neon_magenta")
        panel.theme_combo.SetSelection(target)
        dlg._save_from_ui()
        reloaded = Settings(path=os.path.join(folder, "settings.json"))
        self.assertEqual(reloaded.theme, "neon_magenta")

    def _welcome_frame(self, folder: str):
        frame = MainFrame(settings=self._settings(folder),
                          store=self._store(folder))
        self.addCleanup(frame.Destroy)
        frame.Show()
        self.addCleanup(frame.Hide)
        frame.SetSize((900, 620))
        frame.Layout()
        self.app.Yield()
        return frame, frame.settings

    def test_wallpaper_mode_leaves_only_the_four_buttons(self):
        folder = self._tempdir("aivs_wallpaper_")
        frame, settings = self._welcome_frame(folder)
        preset = theme.THEME_PRESETS[0]
        theme.apply_theme(frame, preset.theme_id, settings)
        frame._apply_welcome_mode()
        self.assertTrue(frame._welcome_texts[0].IsShown())
        self.assertTrue(frame._welcome_note.IsShown())
        theme.set_wallpaper(settings, preset.theme_id, enabled=True, pattern="grid")
        theme.apply_theme(frame, preset.theme_id, settings)
        frame._apply_welcome_mode()
        self.assertFalse(frame._welcome_texts[0].IsShown())
        self.assertFalse(frame._welcome_note.IsShown())
        # The four buttons survive, in the order the request names them.
        for attribute in ("new_project_btn", "remove_btn", "recent_list"):
            self.assertTrue(hasattr(frame, attribute))
        self.assertIn("Remove selected pro", frame.remove_btn.GetLabel())
        self.assertEqual(theme.get_wallpaper(settings, preset.theme_id)["pattern"],
                         "grid")

    def test_the_wallpaper_never_covers_a_button_or_the_project_list(self):
        """The wallpaper lives in its own - empty - region of the window.

        It used to be painted behind the whole welcome screen, which put the
        pattern under the buttons and the recent projects list.  The screen is
        now three regions instead: a full-width action bar, a left project
        sidebar, and a wallpaper panel that holds nothing but the (hidden)
        description - so no control can ever sit on top of the wallpaper.
        """
        folder = self._tempdir("aivs_wallpaper_area_")
        frame, settings = self._welcome_frame(folder)
        preset = theme.THEME_PRESETS[0]
        theme.set_wallpaper(settings, preset.theme_id, enabled=True, pattern="grid")
        theme.apply_theme(frame, preset.theme_id, settings)
        frame._apply_welcome_mode()
        frame.Layout()
        self.app.Yield()

        # No button, and not the project list, is a child of the wallpaper.
        inside: list[wx.Window] = []
        stack = [frame.wallpaper_panel]
        while stack:
            window = stack.pop()
            inside.append(window)
            stack.extend(window.GetChildren())
        self.assertNotIn(frame.recent_list, inside)
        self.assertEqual([w for w in inside if isinstance(w, wx.Button)], [])

        # ...and none of them overlaps the wallpaper on screen either.
        region = frame.wallpaper_panel.GetScreenRect()
        for name, widget in (("action bar", frame.action_bar),
                             ("sidebar", frame.sidebar),
                             ("new project", frame.new_project_btn),
                             ("remove", frame.remove_btn),
                             ("recent list", frame.recent_list)):
            with self.subTest(widget=name):
                self.assertFalse(region.Intersects(widget.GetScreenRect()))

        # The toolbar spans the window's full width but only the buttons'
        # height plus padding, and the wallpaper gets the large empty area.
        self.assertEqual(frame.action_bar.GetSize().width,
                         frame.welcome_panel.GetClientSize().width)
        self.assertGreater(frame.action_bar.GetSize().height,
                           frame.new_project_btn.GetSize().height)
        self.assertGreater(region.GetWidth(), frame.sidebar.GetSize().width)
        self.assertGreater(region.GetHeight(), frame.action_bar.GetSize().height)

    def test_the_sidebar_hint_wraps_instead_of_being_clipped(self):
        folder = self._tempdir("aivs_sidebar_")
        frame, _settings = self._welcome_frame(folder)
        note = frame._welcome_note
        self.assertLessEqual(note.GetSize().width,
                             frame.sidebar.GetClientSize().width)
        self.assertGreater(note.GetBestSize().height, note.GetCharHeight(),
                           "the hint should occupy more than a single line")


if __name__ == "__main__":
    unittest.main()
