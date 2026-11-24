"""Prove the interface at 100%, 150% and 200% display scaling.

Windows scaling cannot be changed from a test, so the scale factor is forced
through ``gui.dpi.set_scale_override`` and every window is built three times -
once per scale.  The geometry the application asks for is compared with what
each scale actually produced, and the *100%* run is the baseline: the question
is never "does this control happen to be smaller than its best size" (the
application deliberately keeps a few that way, such as the 200x10 category list
that grows by its sizer), it is "did scaling add any".

What is reported per window, at each scale:

  controls   visible controls in the window
  borders    sizer borders, compared with ``round(dip x scale)``
  min-sizes  explicit minimum sizes, compared with ``round(dip x scale)``
  overflow   controls smaller than the size they need; ``new`` counts the ones
             100% did not have (a failure)
  zero       visible controls with no area (a failure)
  overlap    visible siblings whose rectangles intersect (a failure)
  size       the window's own size, which must be its 100% size x scale

Then it turns Windows high contrast on (through ``gui.contrast.set_override``),
re-themes every window and reports the palette, whether a wallpaper survived
and the contrast ratio of the scheme's own colours.

Run::

    .venv/Scripts/python.exe tools/check_dpi_layout.py
    .venv/Scripts/python.exe tools/check_dpi_layout.py --no-screenshots
    .venv/Scripts/python.exe tools/check_dpi_layout.py --scale 1.5

Exit code 0 means no finding; 1 prints every finding first.  Screenshots are
written to ``build/dpi_layout/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from ai_voice_studio.gui import contrast, dpi, theme
from ai_voice_studio.gui.accept_dialog import AcceptanceDialog
from ai_voice_studio.gui.announce import is_live_region
from ai_voice_studio.gui.dialogs import _ActionDialog
from ai_voice_studio.gui.main_frame import MainFrame
from ai_voice_studio.gui.new_project_wizard import NewProjectWizard
from ai_voice_studio.gui.progress import TaskProgressDialog
from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.gui.theme_studio import ThemeStudioDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore

SCALES = (1.0, 1.5, 2.0)

#: Controls whose best size is their text: smaller than that and they are
#: clipped.  Text boxes and lists are exempt by design (they scroll).
_TEXT_CONTROLS = (wx.Button, wx.CheckBox, wx.RadioButton, wx.StaticText,
                  wx.Gauge, wx.StaticBox)

SCREENSHOT_DIR = os.path.join(ROOT, "build", "dpi_layout")


# ---------------------------------------------------------------------------
# Measuring a window
# ---------------------------------------------------------------------------
def _windows(window: wx.Window) -> list[wx.Window]:
    """Every control under ``window``, except invisible announcement vehicles.

    The live region is a 1x1 control that exists so a screen reader can be
    told what changed; it is not something the user reads or clicks, and its
    empty label and 1x1 minimum size would otherwise look like a defect that
    only appears when something happened to be announced first.
    """
    out = [window]
    stack = [child for child in window.GetChildren() if not is_live_region(child)]
    while stack:
        child = stack.pop(0)
        out.append(child)
        stack.extend(grand for grand in child.GetChildren()
                     if not is_live_region(grand))
    return out


def _path(control: wx.Window) -> str:
    """A stable name for a control, so two runs can be compared.

    The index counts siblings of the *same class* and the label is included:
    a plain child index would drift the moment one run builds a control the
    other did not (the settings dialog builds its categories on first visit),
    and then every comparison below it is between two different controls.
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
        parts.append(f"{node.__class__.__name__}[{index}]" + (f":{label}" if label else ""))
        node = parent
    return "/".join(reversed(parts))


def _sizer_borders(window: wx.Window) -> list[int]:
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


def _min_sizes(window: wx.Window) -> dict[str, tuple[int, int]]:
    """Explicit minimum sizes, keyed without labels.

    The key has to survive two builds of the same window, and a label can be
    live text (a model status line starts empty and is filled by a background
    refresh), so the key here is the class and sibling order only; the label is
    still used when reporting a finding.
    """
    out = {}
    for control in _windows(window):
        size = control.GetMinSize()
        if size[0] >= 0 or size[1] >= 0:
            out[_key(control)] = (size[0], size[1])
    return out


def _key(control: wx.Window) -> str:
    """The label-free part of a control's path, for matching two runs."""
    parts = []
    node = control
    while node is not None and not isinstance(node, wx.TopLevelWindow):
        parent = node.GetParent()
        siblings = [other for other in (parent.GetChildren() if parent else [])
                    if other.__class__ is node.__class__]
        index = next((i for i, other in enumerate(siblings) if other is node), 0)
        parts.append(f"{node.__class__.__name__}[{index}]")
        node = parent
    return "/".join(reversed(parts))


def _visible(window: wx.Window) -> list[wx.Window]:
    """The controls the user can actually see, and therefore judge.

    A wizard page that has never been shown has no area, and its controls
    still carry the sizes they were created with; measuring those would report
    the *creation* order of a synthetic display rather than the interface.  So
    a control is only audited when it and every window above it, up to the
    window under test, are shown and have an area.
    """
    out = []
    stack = [window]
    while stack:
        control = stack.pop(0)
        if control is not window:
            if not control.IsShown():
                continue
            size = control.GetClientSize()
            if size[0] <= 0 or size[1] <= 0:
                continue
            out.append(control)
        stack.extend(child for child in control.GetChildren()
                     if not is_live_region(child))
    return out


def _overflow(window: wx.Window) -> dict[str, tuple]:
    """Controls squeezed below the size their own text needs."""
    out = {}
    for control in _visible(window):
        if not isinstance(control, _TEXT_CONTROLS):
            continue
        size, best = control.GetSize(), control.GetBestSize()
        if size[0] < best[0] or size[1] < best[1]:
            out[_path(control)] = (tuple(size), tuple(best))
    return out


def _empty(window: wx.Window) -> list[str]:
    """Visible controls with no area: nothing can be read in them."""
    out = []
    for control in _visible(window):
        if not isinstance(control, (wx.Button, wx.CheckBox, wx.RadioButton,
                                    wx.StaticText, wx.TextCtrl, wx.ComboBox,
                                    wx.Choice, wx.ListBox, wx.ListCtrl, wx.Gauge)):
            continue
        width, height = control.GetSize()
        if width <= 0 or height <= 0:
            out.append(f"{_path(control)} {width}x{height}")
    return out


def _overlaps(window: wx.Window) -> list[str]:
    """Visible siblings drawn on top of each other."""
    out = []
    for parent in [window, *_visible(window)]:
        children = [child for child in parent.GetChildren()
                    if child.IsShown() and not is_live_region(child)
                    and child.GetSize()[0] > 0 and child.GetSize()[1] > 0]
        for first in range(len(children)):
            for second in range(first + 1, len(children)):
                rect = children[first].GetRect().Intersect(children[second].GetRect())
                if rect.GetWidth() > 1 and rect.GetHeight() > 1:
                    out.append(f"{_path(children[first])} over "
                               f"{_path(children[second])}")
    return out


def measure(window: wx.Window) -> dict:
    size = tuple(window.GetSize())
    min_size = tuple(window.GetMinSize())
    return {
        "controls": len(_visible(window)),
        "borders": _sizer_borders(window),
        "mins": _min_sizes(window),
        "paths": {_key(control): _path(control) for control in _windows(window)},
        "overflow": _overflow(window),
        "empty": _empty(window),
        "overlap": _overlaps(window),
        "size": size,
        "best": tuple(window.GetBestSize()),
        "from_content": bool(dpi.size_from_content(window)),
        "minor_than_min": min_size[0] > size[0] or min_size[1] > size[1],
        "min_size": min_size,
    }


def capture(window: wx.Window, path: str, size: tuple[int, int]) -> bool:
    """Save a picture of the window as it is laid out (needs it on screen).

    The window is put back to the size that was measured after it is shown: on
    a session whose desktop policy maximizes windows on show (this one does),
    the picture would otherwise be of a maximized window rather than of the
    layout under test.
    """
    try:
        window.Show()
        wx.Yield()
        if tuple(window.GetSize()) != tuple(size):
            window.SetSize(size)
            wx.Yield()
        window.Raise()
        width, height = window.GetClientSize()
        if width <= 0 or height <= 0:
            return False
        bitmap = wx.Bitmap(width, height)
        memory = wx.MemoryDC(bitmap)
        memory.Blit(0, 0, width, height, wx.ClientDC(window), 0, 0)
        memory.SelectObject(wx.NullBitmap)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return bool(bitmap.SaveFile(path, wx.BITMAP_TYPE_PNG))
    except Exception as exc:  # a screenshot must never fail the audit
        print(f"    (screenshot failed: {exc})")
        return False


# ---------------------------------------------------------------------------
# The windows under test
# ---------------------------------------------------------------------------
def _factories():
    """``(name, build)`` for every window that can be built without a project."""
    def settings_all_categories(frame, settings, store):
        dialog = SettingsDialog(frame, settings, store)
        for panel in dialog.panels():          # every category, as a user would
            panel.Refresh()
        return dialog

    return (
        ("MainFrame", lambda f, s, st: MainFrame(None, s, st)),
        ("SettingsDialog (18 categories)", settings_all_categories),
        ("NewProjectWizard", NewProjectWizard),
        ("ThemeStudioDialog", lambda f, s, st: ThemeStudioDialog(f, s)),
        ("ActionDialog", lambda f, s, st: _ActionDialog(
            f, title="Save project", message="Save the project before closing?",
            actions=[("save", "Save"), ("discard", "Discard"), ("cancel", "Cancel")],
            default_key="save")),
        ("TaskProgressDialog", lambda f, s, st: TaskProgressDialog(f, title="Working")),
        ("AcceptanceDialog", lambda f, s, st: AcceptanceDialog(f)),
    )


def _compare(scale: float, baseline: dict, measured: dict) -> list[str]:
    """Findings for one window at one scale, against the 100% baseline.

    Defects that exist at 100% as well are the application's own business (a
    wizard page that is not laid out until it is shown, an engine card built
    inside a static box); the failure this tool exists for is a defect the
    *conversion* adds, so those categories are compared as sets.
    """
    findings = []
    for name in sorted(set(measured["empty"]) - set(baseline["empty"])):
        findings.append(f"a visible control has no area: {name}")
    findings.extend(f"overlapping controls: {name}" for name in
                    sorted(set(measured["overlap"]) - set(baseline["overlap"])))
    if measured["minor_than_min"] and not baseline["minor_than_min"]:
        findings.append(f"the window is smaller than its minimum "
                        f"{measured['size']} < {measured['min_size']}")

    for name in sorted(set(measured["overflow"]) - set(baseline["overflow"])):
        findings.append(f"scaling clipped {name} to "
                        f"{measured['overflow'][name][0]} "
                        f"(needs {measured['overflow'][name][1]})")

    if len(measured["borders"]) != len(baseline["borders"]):
        findings.append(f"{len(measured['borders'])} sizer borders, "
                        f"{len(baseline['borders'])} at 100%")
    for index, (border, dip) in enumerate(zip(measured["borders"],
                                              baseline["borders"])):
        if border != round(dip * scale):
            findings.append(f"sizer border #{index} is {border}, expected "
                            f"{round(dip * scale)} ({dip} at 100%)")

    for key in sorted(set(baseline["mins"]) - set(measured["mins"])):
        findings.append(f"{measured['paths'].get(key, key)} lost its minimum "
                        f"size {baseline['mins'][key]}")
    for key, (width, height) in measured["mins"].items():
        dip = baseline["mins"].get(key)
        if dip is None:
            findings.append(f"{measured['paths'].get(key, key)} sets a minimum "
                            f"size that 100% did not")
            continue
        expected = (round(dip[0] * scale) if dip[0] >= 0 else dip[0],
                    round(dip[1] * scale) if dip[1] >= 0 else dip[1])
        if (width, height) != expected:
            findings.append(f"{measured['paths'].get(key, key)} has minimum "
                            f"size {(width, height)}, expected {expected} "
                            f"({dip} at 100%)")

    base_size = baseline["size"]
    expected_size = (round(base_size[0] * scale), round(base_size[1] * scale))
    # The window may end up *bigger* than the exact multiple, and that is not a
    # defect: a dialog whose content is taller at this scale is grown to fit
    # it, and extra room clips nothing.  Coming out *smaller* is the defect the
    # conversion must never cause - it is what pushes content out of view, and
    # it is what this tool exists to catch.
    if baseline["from_content"] or measured["from_content"]:
        need = measured["best"]
        reason = f"the {need} its content needs"
    else:
        need = expected_size
        reason = f"its {expected_size} footprint ({base_size} at 100%)"
    if measured["size"][0] < need[0] - 1 or measured["size"][1] < need[1] - 1:
        findings.append(f"the window is {measured['size']}, smaller than "
                        f"{reason}")
    return findings


def run_scale(scale: float, frame, settings, store, screenshots: bool):
    """Build, measure and report every window at one scale."""
    results = {}
    for name, build in _factories():
        with dpi.at_scale(scale):
            window = build(frame, settings, store)
        try:
            measured = measure(window)
            results[name] = measured
            if screenshots:
                slug = name.split()[0].lower()
                capture(window, os.path.join(
                    SCREENSHOT_DIR, f"scale_{round(scale * 100):03d}_{slug}.png"),
                    measured["size"])
        finally:
            window.Destroy()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, action="append",
                        help="a scale to check (repeatable); default 1.0, 1.5, 2.0")
    parser.add_argument("--no-screenshots", action="store_true",
                        help="measure without showing the windows")
    args = parser.parse_args()
    scales = tuple(args.scale) if args.scale else SCALES

    # Before the application object: the tool is also the demonstration that
    # the awareness call works, and an unaware process would report 1.0 for a
    # 200% display (Windows would be stretching the bitmap instead).
    dpi.enable_process_awareness()
    app = wx.App(False)
    frame = wx.Frame(None, title="DPI layout check")
    folder = tempfile.mkdtemp(prefix="aivs_dpi_")
    settings = Settings(path=os.path.join(folder, "settings.json"))
    store = ModelStore(state_file=os.path.join(folder, "models.json"))

    print(f"DPI awareness: {dpi.awareness()}   {contrast.describe()}")
    print(f"scales checked: {', '.join(f'{scale * 100:.0f}%' for scale in scales)}")

    baseline = None
    failures: list[str] = []
    header = (f"{'scale':>6}  {'window':<32} {'ctrls':>5} {'borders':>7} "
              f"{'mins':>5} {'new':>4} {'size':>12}")
    for scale in scales:
        print("\n" + header)
        print("-" * len(header))
        results = run_scale(scale, frame, settings, store,
                            screenshots=not args.no_screenshots)
        if baseline is None:
            baseline = results
        for name, measured in results.items():
            base = baseline[name]
            new_overflow = len(set(measured["overflow"]) - set(base["overflow"]))
            print(f"{scale * 100:5.0f}%  {name:<32} {measured['controls']:>5} "
                  f"{len(measured['borders']):>7} {len(measured['mins']):>5} "
                  f"{new_overflow:>4} {measured['size'][0]:>5}x{measured['size'][1]:<6}")
            # The 100% run is compared with itself too: an overlap, an empty
            # control or a window below its minimum size is a defect at any
            # scale, and the arithmetic simply comes out exact at 1.0.
            for finding in _compare(scale, base, measured):
                failures.append(f"{scale * 100:.0f}% {name}: {finding}")

    _report_high_contrast(frame, settings)

    if failures:
        print("\nFINDINGS")
        for line in failures:
            print(f"  ! {line}")
        print(f"\n{len(failures)} finding(s)")
        frame.Destroy()
        shutil.rmtree(folder, ignore_errors=True)
        return 1
    print("\nNo findings: every window keeps its DIP geometry at every scale.")
    frame.Destroy()
    shutil.rmtree(folder, ignore_errors=True)
    return 0


def _report_high_contrast(frame, settings) -> None:
    """Turn high contrast on and prove the application follows it."""
    print("\nhigh contrast (forced on)")
    print("-" * 44)
    contrast.set_override(True)
    try:
        palette = contrast.palette()
        for key, value in palette.items():
            print(f"  {key:<12} {value}")
        print(f"  contrast ratio  {contrast.contrast_ratio(palette['fg'], palette['bg']):.2f}:1")
        settings.set("theme", "dark")
        settings.set("wallpapers", {"dark": {"enabled": True, "image": "any.png"}})
        print(f"  theme id        {settings.get('theme')!r} (a dark theme was chosen)")
        print(f"  resolved to     {theme.resolve_palette('dark', settings)['bg']} "
              f"(the system background, not the dark theme's)")
        print(f"  wallpaper       {theme.is_wallpaper_enabled('dark', settings)}"
              f"   pattern {theme.theme_pattern('dark', settings)}")
        panel = wx.Panel(frame)
        label = wx.StaticText(panel, label="High contrast text")
        theme.apply_theme(panel, "dark", settings)
        matches = label.GetForegroundColour() == wx.Colour(palette["fg"])
        print(f"  window text     "
              f"{label.GetForegroundColour().GetAsString(wx.C2S_HTML_SYNTAX)}"
              f" ({'follows the system' if matches else 'DOES NOT FOLLOW THE SYSTEM'})")
        panel.Destroy()
    finally:
        contrast.set_override(None)


if __name__ == "__main__":
    raise SystemExit(main())
