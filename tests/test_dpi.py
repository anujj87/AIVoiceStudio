"""Tests for scale-derived geometry: the DPI layer.

Two properties matter here and both are asserted against real windows:

* the application's written geometry is treated as DIPs, so at 150% and 200%
  every explicit minimum size, every sizer border and the window's own size
  grow by exactly the scale factor - and re-deriving them is exact rather than
  compounding, which is what a window that moves between monitors needs;
* scaling never *introduces* clipping - no control is pushed below its best
  size by the conversion that was not already below it at 100%.  That is
  checked as a set comparison between the scales rather than as an absolute
  rule, so the app's deliberate choices (a category list deliberately kept to
  200x10 and grown by its sizer) are not reported as defects.

The scale factor is forced (:func:`dpi.set_scale_override`) because the test
machine's display is 100%; the same code path runs on a real 150% monitor,
where the factor comes from ``wx.Window.GetDPIScaleFactor``.
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

from ai_voice_studio.gui import dpi
from ai_voice_studio.gui.progress import TaskProgressDialog
from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore

SCALES = (1.0, 1.5, 2.0)


def _largest_window(parent: wx.Window | None = None) -> tuple[int, int]:
    """The biggest a top-level window can actually become on this machine.

    Windows will not grow a window past the display's tracking size, so a
    dialog asked for 1640x1120 on a 1400x840 CI screen comes back at
    1400x840.  The limit is measured rather than assumed, so ``min(wanted,
    limit)`` is exact both on a large monitor (where nothing is capped) and
    on a small one.
    """
    probe = wx.Dialog(parent, size=(100, 100))
    try:
        probe.SetSize((100_000, 100_000))
        return tuple(probe.GetSize())
    finally:
        probe.Destroy()


def _within_display(size: tuple[int, int], window: wx.Window) -> tuple[int, int]:
    """``size``, reduced to what this machine's display will allow."""
    limit = _largest_window(window.GetParent())
    return (min(size[0], limit[0]), min(size[1], limit[1]))


def _size_note(wanted: tuple[int, int], window: wx.Window) -> str:
    """Context for a size mismatch: display, limit and the recorded DIPs."""
    area = wx.Display(0).GetClientArea()
    state = getattr(window, "_aivs_dpi_state", {}) or {}
    return (f"wanted {wanted} at scale {dpi.scale_of(window)}; "
            f"display {area.width}x{area.height}; "
            f"largest window {_largest_window(window.GetParent())}; "
            f"recorded DIP size {state.get('size')}, "
            f"from content {state.get('size_from_content')}")


class ScaleFactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def tearDown(self):
        dpi.set_scale_override(None)

    def test_a_length_in_dips_becomes_pixels(self):
        for scale, expected in ((1.0, 6), (1.5, 9), (2.0, 12)):
            with self.subTest(scale=scale), dpi.at_scale(scale):
                self.assertEqual(dpi.px(6, self.frame), expected)

    def test_a_half_pixel_rounds_to_whole_pixels(self):
        with dpi.at_scale(1.5):
            # 5 DIPs at 150% is 7.5 pixels: nothing in wx accepts half a pixel,
            # so it has to be rounded rather than truncated (5 -> 7, not 7.5).
            self.assertEqual(dpi.px(5, self.frame), 8)

    def test_a_size_keeps_its_negative_components(self):
        """-1 means "let the control decide" and must survive the conversion."""
        with dpi.at_scale(2.0):
            self.assertEqual(dpi.px_size((120, -1), self.frame), (240, -1))

    def test_the_override_is_restored_by_the_context_manager(self):
        with dpi.at_scale(1.5):
            self.assertEqual(dpi.scale_override(), 1.5)
        self.assertIsNone(dpi.scale_override())

    def test_without_an_override_the_window_answers(self):
        dpi.set_scale_override(None)
        factor = dpi.current_scale(self.frame)
        self.assertEqual(factor, self.frame.GetDPIScaleFactor())
        self.assertGreater(factor, 0)

    def test_a_window_that_cannot_answer_is_one_to_one(self):
        broken = mock.Mock()
        broken.GetDPIScaleFactor.side_effect = RuntimeError("destroyed")
        self.assertEqual(dpi.current_scale(broken), 1.0)

    def test_awareness_is_reported_in_words(self):
        self.assertIn(dpi.awareness(),
                      {"per-monitor-v2", "per-monitor", "system", "unaware",
                       "unknown", "not-windows"})

    def test_per_monitor_v2_is_requested_first(self):
        user32 = mock.Mock()
        user32.SetProcessDpiAwarenessContext.return_value = 1
        with mock.patch.object(dpi.sys, "platform", "win32"), \
             mock.patch.object(dpi.sys, "frozen", False, create=True), \
             mock.patch.object(dpi.ctypes, "windll") as windll:
            windll.user32 = user32
            self.assertEqual(dpi.enable_process_awareness(), "per-monitor-v2")
        user32.SetProcessDpiAwarenessContext.assert_called_once()
        context = user32.SetProcessDpiAwarenessContext.call_args.args[0]
        self.assertEqual(context.value & 0xFFFFFFFF, 0xFFFFFFFC,
                         "PerMonitorV2 is the context -4")

    def test_a_refusal_falls_back_to_the_older_apis(self):
        user32 = mock.Mock()
        user32.SetProcessDpiAwarenessContext.return_value = 0
        user32.SetProcessDPIAware.return_value = 1
        shcore = mock.Mock()
        shcore.SetProcessDpiAwareness.return_value = 1  # E_ACCESSDENIED
        with mock.patch.object(dpi.sys, "platform", "win32"), \
             mock.patch.object(dpi.sys, "frozen", False, create=True), \
             mock.patch.object(dpi.ctypes, "windll") as windll:
            windll.user32 = user32
            windll.shcore = shcore
            self.assertEqual(dpi.enable_process_awareness(), "system")

    def test_a_frozen_build_reports_the_manifest(self):
        with mock.patch.object(dpi.sys, "frozen", True, create=True):
            self.assertEqual(dpi.enable_process_awareness(), "manifest")


class GeometryTest(unittest.TestCase):
    """A real window, laid out at each scale."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()
        cls.frame = wx.Frame(None)

    @classmethod
    def tearDownClass(cls):
        cls.frame.Destroy()

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aivs_dpi_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.settings = Settings(path=os.path.join(self.folder, "settings.json"))
        self.store = ModelStore(state_file=os.path.join(self.folder, "models.json"))

    def tearDown(self):
        dpi.set_scale_override(None)

    def _dialog(self) -> SettingsDialog:
        dlg = SettingsDialog(self.frame, self.settings, self.store)
        self.addCleanup(dlg.Destroy)
        return dlg

    # -- recording ---------------------------------------------------------
    @staticmethod
    def _min_sizes(window: wx.Window) -> dict:
        """Every explicitly set minimum size in a window, by control path."""
        out = {}
        for control in _windows(window):
            size = control.GetMinSize()
            if size[0] >= 0 or size[1] >= 0:
                out[_path(control)] = (size[0], size[1])
        return out

    @staticmethod
    def _borders(window: wx.Window) -> list:
        return _borders(window)

    def test_at_100_percent_everything_is_left_alone(self):
        """The safety property for every existing machine: 100% is identity."""
        dlg = self._dialog()
        before_mins = self._min_sizes(dlg)
        before_borders = self._borders(dlg)
        before_size = tuple(dlg.GetSize())

        dpi.prepare(dlg)

        self.assertEqual(self._min_sizes(dlg), before_mins)
        self.assertEqual(self._borders(dlg), before_borders)
        self.assertEqual(tuple(dlg.GetSize()), before_size)

    def test_every_explicit_minimum_size_scales(self):
        dip = self._dialog()
        before = self._min_sizes(dip)
        dip.Destroy()
        self.assertTrue(before, "the dialog sets no sizes at all")

        for scale in (1.5, 2.0):
            with self.subTest(scale=scale), dpi.at_scale(scale):
                scaled = self._dialog()
                after = self._min_sizes(scaled)
                expected = {path: self._expected_dip(size, scaled)
                            for path, size in before.items()}
                self.assertEqual(set(after), set(before),
                                 "scaling changed which controls set a size")
                for path, size in expected.items():
                    self.assertEqual(after[path], size,
                                     f"{path} was not scaled to {scale}x")

    def test_sizer_borders_scale(self):
        dip = self._dialog()
        before = self._borders(dip)
        dip.Destroy()
        self.assertTrue(before, "the dialog has no sizer borders to check")

        with dpi.at_scale(2.0):
            scaled = self._dialog()

        self.assertEqual(self._borders(scaled),
                         [border * 2 for border in before])

    def test_the_window_keeps_its_footprint_in_dips(self):
        with dpi.at_scale(1.0):
            dip = self._dialog()
            dip_size = tuple(dip.GetSize())
        with dpi.at_scale(2.0):
            scaled = self._dialog()
            self.assertEqual(tuple(scaled.GetSize()),
                             _within_display((dip_size[0] * 2, dip_size[1] * 2),
                                             scaled),
                             _size_note((dip_size[0] * 2, dip_size[1] * 2),
                                        scaled))

    def test_preparing_twice_does_not_compound(self):
        with dpi.at_scale(1.5):
            dlg = self._dialog()
            once = self._min_sizes(dlg)
            dpi.prepare(dlg)
            dpi.prepare(dlg)
            self.assertEqual(self._min_sizes(dlg), once)

    def test_rescaling_is_exact_not_compounded(self):
        """100% -> 150% -> 200% must land on exactly 2x the DIP values."""
        with dpi.at_scale(1.0):
            dip = self._dialog()
            expected = self._min_sizes(dip)
            dip.Destroy()

        dlg = self._dialog()
        for scale in (1.5, 2.0):
            with dpi.at_scale(scale):
                dpi.rescale(dlg)
        with dpi.at_scale(2.0):
            reached = self._min_sizes(dlg)
            wanted = {path: self._expected_dip(size, dlg)
                      for path, size in expected.items()}
        self.assertEqual(reached, wanted,
                         "two scale changes must land on 2x the DIP geometry")

    def test_a_window_the_user_resized_keeps_that_size_in_dips(self):
        dlg = self._dialog()
        dpi.set_scale_override(1.5)
        dpi.prepare(dlg)
        dlg.SetSize((1200, 900))  # the user drags the corner at 150%
        with dpi.at_scale(2.0):
            dpi.rescale(dlg)

        self.assertEqual(tuple(dlg.GetSize()), _within_display((1600, 1200), dlg),
                         "the resized window must keep its DIP size on the new "
                         "display: " + _size_note((1600, 1200), dlg))

    def test_a_control_the_app_asked_for_scales_its_size(self):
        """``wx.ListCtrl(size=(200, 10))`` carries that as its minimum size."""
        with dpi.at_scale(1.5):
            dlg = self._dialog()
        self.assertEqual(dlg.cat_list.GetMinSize(), (300, 15))

    @staticmethod
    def _category_index(dlg: SettingsDialog, title: str) -> int:
        return next(index for index, cls in enumerate(dlg.CATEGORIES)
                    if cls.title == title)

    def test_a_category_built_later_is_scaled_too(self):
        """Settings builds its categories on first visit; each one scales."""
        with dpi.at_scale(1.0):
            dip = self._dialog()
            panel = dip.ensure_panel(self._category_index(dip, "Manage the STT"))
            before = self._min_sizes(panel)
            dip.Destroy()
        self.assertTrue(before, "the STT category sets no sizes to check")

        with dpi.at_scale(2.0):
            dlg = self._dialog()
            panel = dlg.ensure_panel(self._category_index(dlg, "Manage the STT"))
            self.assertEqual(dpi.scale_of(panel), 2.0)
            after = self._min_sizes(panel)
            wanted = {path: self._expected_dip(size, panel)
                      for path, size in before.items()}
        self.assertEqual(after, wanted,
                         "a category built on first visit is not scaled")

    def test_a_category_built_later_follows_a_display_change(self):
        """A category built at 100% must scale when the display does.

        This is the case the dialog delegates: the panel keeps its own
        recorded geometry, so the dialog's rescale cannot scale its sizes
        twice - it hands the new factor to the panel instead.
        """
        with dpi.at_scale(1.0):
            dlg = self._dialog()
            panel = dlg.ensure_panel(self._category_index(dlg, "Manage the STT"))
            before = self._min_sizes(panel)
        with dpi.at_scale(2.0):
            dpi.rescale(dlg)
            after = self._min_sizes(panel)
            wanted = {path: self._expected_dip(size, panel)
                      for path, size in before.items()}
            self.assertEqual(set(after), set(before))
        self.assertEqual(dpi.scale_of(panel), 2.0)
        self.assertEqual(after, wanted)

    @staticmethod
    def _expected_dip(size: tuple[int, int], window: wx.Window) -> tuple[int, int]:
        """What a recorded DIP size becomes at the window's current scale."""
        width, height = size
        return (
            dpi.px(width, window) if width >= 0 else width,
            dpi.px(height, window) if height >= 0 else height,
        )

    def test_a_progress_dialog_fits_its_content(self):
        """A content-sized dialog must not be left smaller than its content."""
        for scale in SCALES:
            with self.subTest(scale=scale), dpi.at_scale(scale):
                dlg = TaskProgressDialog(self.frame, title="Working")
                self.addCleanup(dlg.Destroy)
                width, height = dlg.GetSize()
                best_width, best_height = dlg.GetBestSize()
                self.assertGreaterEqual(width, best_width)
                self.assertGreaterEqual(height, best_height)

    def test_size_from_content_says_how_the_window_was_sized(self):
        """The audit needs to know which windows may differ from a multiple."""
        plain = wx.Frame(None)
        self.addCleanup(plain.Destroy)
        self.assertIsNone(dpi.size_from_content(plain),
                          "an unprepared window has no recorded geometry")

        with dpi.at_scale(1.0):
            dialog = self._dialog()      # size=(820, 560), its content is smaller
            progress = TaskProgressDialog(self.frame, title="Working")
            self.addCleanup(progress.Destroy)
        self.assertFalse(dpi.size_from_content(dialog))
        self.assertTrue(dpi.size_from_content(progress))

    # -- the proof ---------------------------------------------------------
    def test_scaling_never_introduces_clipping(self):
        """At 150% and 200% no more controls are under-sized than at 100%.

        Comparing the scales against each other rather than against an absolute
        rule is what makes this honest: the application deliberately keeps a few
        controls below their best size (the category list is 200x10 and grows by
        its sizer), and the question that matters is whether the conversion adds
        any to that set.
        """
        with dpi.at_scale(1.0):
            baseline = self._clipped(self._dialog())
        for scale in (1.5, 2.0):
            with self.subTest(scale=scale), dpi.at_scale(scale):
                found = self._clipped(self._dialog())
                self.assertLessEqual(
                    set(found), set(baseline),
                    f"{scale}x clipped controls that 100% did not: "
                    f"{sorted(set(found) - set(baseline))}",
                )

    @staticmethod
    def _clipped(window: wx.Window) -> dict:
        """Visible, non-empty controls smaller than the size they need."""
        report = {}
        for control in _windows(window):
            if control is window or not control.IsShown():
                continue
            size = control.GetSize()
            if size[0] <= 0 or size[1] <= 0:
                continue
            best = control.GetBestSize()
            if best[0] > size[0] or best[1] > size[1]:
                report[_path(control)] = (tuple(size), tuple(best))
        return report

    # -- following the system ----------------------------------------------
    def test_follow_binds_once(self):
        dlg = self._dialog()
        dpi.follow(dlg)
        dpi.follow(dlg)
        # A second bind would run the handler twice for one event.
        handler = mock.Mock()
        with mock.patch.object(dpi, "rescale", handler):
            event = wx.DPIChangedEvent()
            event.SetEventObject(dlg)
            dlg.ProcessEvent(event)
        self.assertEqual(handler.call_count, 1)

    def test_a_display_change_relays_out_the_window(self):
        with dpi.at_scale(1.0):
            dlg = self._dialog()
        dpi.set_scale_override(2.0)
        event = wx.DPIChangedEvent()
        event.SetEventObject(dlg)
        dlg.ProcessEvent(event)

        self.assertEqual(dpi.scale_of(dlg), 2.0)
        self.assertEqual(tuple(dlg.GetSize()), _within_display((1640, 1120), dlg),
                         _size_note((1640, 1120), dlg))
        self.assertEqual(dlg.cat_list.GetMinSize(), (400, 20))

    def test_the_display_change_is_announced(self):
        dlg = self._dialog()
        announced = []
        with mock.patch("ai_voice_studio.gui.announce.announce",
                        side_effect=lambda text, **kw: announced.append(text)):
            dpi.set_scale_override(1.5)
            event = wx.DPIChangedEvent()
            event.SetEventObject(dlg)
            dlg.ProcessEvent(event)

        self.assertTrue(any("150 percent" in text for text in announced),
                        f"the new scaling was not announced: {announced}")


def _windows(window: wx.Window) -> list[wx.Window]:
    """``window`` and every descendant, parents before children."""
    out = [window]
    stack = list(window.GetChildren())
    while stack:
        child = stack.pop(0)
        out.append(child)
        stack.extend(child.GetChildren())
    return out


def _path(control: wx.Window) -> str:
    """A stable name for a control, so scales can be compared with each other.

    The index counts siblings of the *same class* and the label is part of the
    name: the two runs build slightly different trees (SettingsDialog builds
    its categories on first visit), and a plain child index would then compare
    one control against another.
    """
    parts = []
    node = control
    while node is not None and not isinstance(node, wx.TopLevelWindow):
        parent = node.GetParent()
        siblings = [other for other in (parent.GetChildren() if parent else [])
                    if other.__class__ is node.__class__]
        index = next((i for i, other in enumerate(siblings) if other is node), 0)
        label = ""
        for getter in ("GetLabel", "GetName", "GetValue"):
            try:
                value = getattr(node, getter)()
            except Exception:
                continue
            if isinstance(value, str) and value.strip():
                # Only the leading part of a label: several carry live status
                # text after a colon, and a name that changes between two runs
                # would look like a control that disappeared.
                label = value.strip().split(":")[0].replace("/", "_")[:16]
                break
        parts.append(f"{node.__class__.__name__}[{index}]"
                     + (f":{label}" if label else ""))
        node = parent
    return "/".join(reversed(parts))


def _borders(window: wx.Window) -> list:
    """Every sizer border in the window's tree, in a stable order."""
    out: list[int] = []
    for control in _windows(window):
        sizer = control.GetSizer()
        if sizer is not None:
            _collect_borders(sizer, out)
    return out


def _collect_borders(sizer: wx.Sizer, out: list[int]) -> None:
    for index in range(sizer.GetItemCount()):
        item = sizer.GetItem(index)
        if item is None:
            continue
        out.append(int(item.GetBorder()))
        nested = item.GetSizer()
        if nested is not None:
            _collect_borders(nested, out)


if __name__ == "__main__":
    unittest.main()
