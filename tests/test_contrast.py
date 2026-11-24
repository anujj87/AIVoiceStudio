"""Tests for following Windows high contrast.

High contrast is the user having already chosen every colour, so the tests here
are about *not* overriding that choice: the palette comes from the system
colours unchanged, every theme id resolves to it, the welcome screen loses its
wallpaper and its patterns, and none of it is written to the settings file, so
switching high contrast off restores the theme that was chosen.

Two of the tests are the reason the feature exists rather than a style
preference: the contrast ratio of the palette is the scheme's own (it is not
re-derived or mixed), and a real window's text ends up in the system's window
text colour.  Windows' own high-contrast state cannot be switched on from a
test, so it is forced through :func:`contrast.set_override`; the Win32 call
that reads it is covered separately with a fake ``SystemParametersInfoW``.
"""

from __future__ import annotations

import ctypes
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

from ai_voice_studio.gui import contrast, dpi, theme

#: Three of the schemes Windows ships, as the colours the user sees in them.
SCHEMES = {
    "High Contrast Black": {"bg": "#000000", "fg": "#ffffff", "accent": "#ffff00"},
    "High Contrast White": {"bg": "#ffffff", "fg": "#000000", "accent": "#0000ff"},
    "High Contrast #1 (blue)": {"bg": "#000000", "fg": "#00ffff", "accent": "#ffff00"},
}

PALETTE_KEYS = ("bg", "fg", "control_bg", "control_fg", "accent", "heading")


def _fake_colour(scheme: dict[str, str]):
    """A stand-in for ``wx.SystemSettings.GetColour`` for one scheme."""
    roles = {
        wx.SYS_COLOUR_WINDOW: scheme["bg"],
        wx.SYS_COLOUR_WINDOWTEXT: scheme["fg"],
        wx.SYS_COLOUR_HIGHLIGHT: scheme["accent"],
    }

    def colour(role, fallback="#000000"):
        return roles.get(role, fallback)

    return colour


class DetectionTest(unittest.TestCase):
    def tearDown(self):
        contrast.set_override(None)

    def test_windows_is_asked_with_the_high_contrast_action(self):
        """SPI_GETHIGHCONTRAST, with the flag word read back."""
        struct = contrast.HIGHCONTRASTW()
        seen = {}

        def system_parameters_info(action, size, pointer, flags):
            seen["action"] = action
            seen["size"] = size
            # The API fills the caller's structure in; do the same.
            target = ctypes.cast(pointer, ctypes.POINTER(contrast.HIGHCONTRASTW)).contents
            target.dwFlags = contrast._HCF_HIGHCONTRASTON
            seen["wrote"] = True
            return 1

        fake = mock.Mock()
        fake.SystemParametersInfoW.side_effect = system_parameters_info
        with mock.patch.object(contrast.ctypes, "windll") as windll:
            windll.user32 = fake
            on, name = contrast.read()

        self.assertEqual(seen["action"], 0x0042, "SPI_GETHIGHCONTRAST is 0x42")
        self.assertEqual(seen["size"], ctypes.sizeof(contrast.HIGHCONTRASTW))
        self.assertGreaterEqual(seen["size"], ctypes.sizeof(struct))
        self.assertTrue(on)
        self.assertEqual(name, "")

    def test_a_flag_that_is_off_means_off(self):
        def system_parameters_info(action, size, pointer, flags):
            target = ctypes.cast(pointer, ctypes.POINTER(contrast.HIGHCONTRASTW)).contents
            target.dwFlags = 0
            return 1

        fake = mock.Mock()
        fake.SystemParametersInfoW.side_effect = system_parameters_info
        with mock.patch.object(contrast.ctypes, "windll") as windll:
            windll.user32 = fake
            self.assertFalse(contrast.enabled())

    def test_a_failing_api_does_not_raise(self):
        fake = mock.Mock()
        fake.SystemParametersInfoW.side_effect = OSError("no such API")
        with mock.patch.object(contrast.ctypes, "windll") as windll:
            windll.user32 = fake
            self.assertFalse(contrast.enabled())
            self.assertEqual(contrast.scheme_name(), "")

    def test_the_override_answers_both_ways(self):
        contrast.set_override(True)
        self.assertTrue(contrast.enabled())
        self.assertIn("on", contrast.describe())
        contrast.set_override(False)
        self.assertFalse(contrast.enabled())
        self.assertEqual(contrast.describe(), "high contrast: off")

    def test_the_real_machine_is_read_without_raising(self):
        contrast.set_override(None)
        on, name = contrast.read()
        self.assertIsInstance(on, bool)
        self.assertIsInstance(name, str)


class PaletteTest(unittest.TestCase):
    def tearDown(self):
        contrast.set_override(None)

    def test_the_palette_is_the_system_colours_unmixed(self):
        for name, scheme in SCHEMES.items():
            with self.subTest(scheme=name), \
                 mock.patch.object(contrast, "colour", _fake_colour(scheme)):
                self.assertEqual(contrast.palette(), {
                    "bg": scheme["bg"],
                    "fg": scheme["fg"],
                    "control_bg": scheme["bg"],
                    "control_fg": scheme["fg"],
                    "accent": scheme["accent"],
                    "heading": scheme["fg"],
                })

    def test_every_key_the_theme_layer_uses_is_present(self):
        self.assertEqual(set(contrast.palette()), set(PALETTE_KEYS))
        for key, value in contrast.palette().items():
            with self.subTest(key=key):
                self.assertRegex(value, r"^#[0-9a-f]{6}$")

    def test_the_scheme_own_contrast_is_preserved(self):
        """The point of the mode: the text keeps the user's contrast."""
        for name, scheme in SCHEMES.items():
            with self.subTest(scheme=name), \
                 mock.patch.object(contrast, "colour", _fake_colour(scheme)):
                palette = contrast.palette()
                self.assertGreaterEqual(
                    contrast.contrast_ratio(palette["fg"], palette["bg"]), 4.5,
                    f"{name} would not be legible in the palette built from it")
                self.assertGreaterEqual(
                    contrast.contrast_ratio(palette["accent"], palette["bg"]), 4.5)

    def test_a_system_that_refuses_a_colour_falls_back(self):
        with mock.patch.object(contrast, "colour", lambda role, fallback="#000000":
                               fallback):
            self.assertEqual(set(contrast.palette()), set(PALETTE_KEYS))

    def test_contrast_ratio_matches_the_known_values(self):
        self.assertAlmostEqual(contrast.contrast_ratio("#000000", "#ffffff"), 21.0,
                               places=1)
        self.assertAlmostEqual(contrast.contrast_ratio("#ffffff", "#ffffff"), 1.0,
                               places=2)


class ThemeFollowsHighContrastTest(unittest.TestCase):
    """The theme layer hands every colour decision back to the user."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aivs_hc_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        from ai_voice_studio.settings import Settings

        self.settings = Settings(path=os.path.join(self.folder, "settings.json"))

    def tearDown(self):
        contrast.set_override(None)
        # The palette is module state (that is what every control asks for its
        # colours), so put the ordinary theme back for the next test.
        theme.apply_theme(self.frame, "system", self.settings)

    def test_every_theme_id_resolves_to_the_system_palette(self):
        self.settings.set("custom_themes", [{
            "id": "custom:mine", "name": "Mine",
            "colors": {"bg": "#123456", "fg": "#ffffff"},
            "pattern": "solid",
        }])
        contrast.set_override(True)
        expected = contrast.palette()
        for theme_id in ("system", "light", "dark",
                         theme.THEME_PRESETS[0].theme_id, "custom:mine",
                         "removed-theme"):
            with self.subTest(theme=theme_id):
                self.assertEqual(theme.resolve_palette(theme_id, self.settings),
                                 expected)

    def test_the_chosen_theme_is_not_written_to_the_settings_file(self):
        self.settings.set("theme", "dark")
        self.settings.save()
        path = os.path.join(self.folder, "settings.json")
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        contrast.set_override(True)

        self.assertEqual(theme.resolve_palette("dark", self.settings),
                         contrast.palette())
        self.assertEqual(theme.theme_pattern("dark", self.settings), "solid")

        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), before,
                             "following high contrast rewrote settings.json")

    def test_a_pattern_becomes_a_flat_fill(self):
        self.settings.set("wallpapers", {"dark": {"enabled": True,
                                                  "pattern": "stripes"}})
        contrast.set_override(True)
        self.assertEqual(theme.theme_pattern("dark", self.settings), "solid")

    def test_a_wallpaper_is_not_used(self):
        self.settings.set("wallpapers", {"dark": {"enabled": True, "image": "x.png"}})
        # With high contrast off the wallpaper is the user's own setting, and
        # this is the check that makes the next line mean something.
        self.assertTrue(theme.is_wallpaper_enabled("dark", self.settings))

        contrast.set_override(True)
        self.assertFalse(theme.is_wallpaper_enabled("dark", self.settings))

    def test_the_welcome_screen_is_painted_flat_instead_of_the_image(self):
        path = os.path.join(self.folder, "wall.png")
        image = wx.Image(20, 20)
        image.SetRGB(wx.Rect(0, 0, 20, 20), 255, 0, 0)  # unmistakable red
        self.assertTrue(image.SaveFile(path, wx.BITMAP_TYPE_PNG))
        palette = {"bg": "#000000", "fg": "#ffffff", "accent": "#ffff00"}

        def painted(with_high_contrast: bool) -> wx.Image:
            contrast.set_override(with_high_contrast)
            bitmap = wx.Bitmap(60, 40)
            dc = wx.MemoryDC(bitmap)
            theme.paint_background(dc, (60, 40), palette, "solid", path)
            dc.SelectObject(wx.NullBitmap)
            return bitmap.ConvertToImage()

        with_high_contrast = painted(True)
        without = painted(False)

        self.assertEqual(with_high_contrast.GetRed(10, 10), 0)
        self.assertEqual(with_high_contrast.GetGreen(10, 10), 0)
        self.assertEqual(without.GetRed(10, 10), 255,
                         "the sanity check: the image is drawn without it")

    def test_a_real_window_takes_the_system_colours(self):
        contrast.set_override(True)
        panel = wx.Panel(self.frame)
        self.addCleanup(panel.Destroy)
        label = wx.StaticText(panel, label="Hello")
        button = wx.Button(panel, label="OK")

        theme.apply_theme(panel, "dark", self.settings)

        palette = contrast.palette()
        self.assertEqual(label.GetForegroundColour(), wx.Colour(palette["fg"]))
        self.assertEqual(panel.GetBackgroundColour(), wx.Colour(palette["bg"]))
        self.assertEqual(theme.current_theme_colors(), palette)
        # Buttons keep their native look: that is the one part of the window
        # Windows already paints in the user's colours.
        self.assertNotEqual(button.GetBackgroundColour(), wx.Colour(palette["bg"]))

    def test_switching_it_off_brings_the_theme_back(self):
        contrast.set_override(True)
        panel = wx.Panel(self.frame)
        self.addCleanup(panel.Destroy)
        theme.apply_theme(panel, "dark", self.settings)
        self.assertEqual(panel.GetBackgroundColour(),
                         wx.Colour(contrast.palette()["bg"]))

        contrast.set_override(False)
        theme.apply_theme(panel, "dark", self.settings)
        self.assertEqual(panel.GetBackgroundColour(),
                         wx.Colour(theme.resolve_palette("dark", self.settings)["bg"]))

    def test_a_windows_colour_change_is_followed_live(self):
        """Turning high contrast on while a window is open re-themes it."""
        from ai_voice_studio.gui.main_frame import MainFrame
        from ai_voice_studio.tts.models import ModelStore

        frame = MainFrame(None, self.settings,
                          ModelStore(state_file=os.path.join(self.folder, "m.json")))
        self.addCleanup(frame.Destroy)
        theme.apply_theme(frame, "dark", self.settings)
        before = frame.GetBackgroundColour()
        self.assertEqual(before,
                         wx.Colour(theme.resolve_palette("dark", self.settings)["bg"]))

        contrast.set_override(True)
        event = wx.SysColourChangedEvent()
        event.SetEventObject(frame)
        frame.ProcessEvent(event)

        self.assertEqual(frame.GetBackgroundColour(),
                         wx.Colour(contrast.palette()["bg"]))
        self.assertNotEqual(frame.GetBackgroundColour(), before)
        self.assertEqual(self.settings.get("theme"), "system",
                         "the re-theme must not write a theme choice")

    def test_the_colour_change_is_announced(self):
        frame = wx.Frame(None)
        self.addCleanup(frame.Destroy)
        dpi.follow(frame)
        announced = []
        contrast.set_override(True)
        with mock.patch("ai_voice_studio.gui.announce.announce",
                        side_effect=lambda text, **kw: announced.append(text)):
            event = wx.SysColourChangedEvent()
            event.SetEventObject(frame)
            frame.ProcessEvent(event)

        self.assertTrue(any("high contrast" in text.lower() for text in announced),
                        f"a high-contrast change was not announced: {announced}")


if __name__ == "__main__":
    unittest.main()
