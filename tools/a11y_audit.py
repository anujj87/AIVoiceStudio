"""Headless screen-reader label audit.

Walks every constructible panel/dialog in the app and prints, for each
input control (text, combo, spin, check, radio, list, listctrl), the
accessible name a screen reader (NVDA/JAWS/Narrator) would announce.

Flags:
  * !DEFAULT  - name is empty or the wxWidgets default class name, i.e.
                the control has no explicit accessible label.
  * !DUP      - duplicate accessible name within the same panel (two
                controls a reader cannot tell apart).

Run: .venv/Scripts/python tools/a11y_audit.py
"""

from __future__ import annotations

import os
import sys

import wx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402

_DEFAULT_NAMES = {
    "", "text", "comboBox", "checkBox", "radioButton", "button",
    "listBox", "listCtrl", "spinButton", "staticText", "scrollBar",
    "slider", "gauge", "statusBar", "toolBar", "menuItem",
}

_INPUT_CLASSES = (
    wx.TextCtrl, wx.ComboBox, wx.CheckBox, wx.RadioButton, wx.RadioBox,
    wx.SpinCtrl, wx.SpinCtrlDouble, wx.ListBox, wx.ListCtrl, wx.Slider,
    wx.Choice, wx.Button, wx.ToggleButton,
)


# Composite controls whose child windows are implementation internals
# (e.g. the edit box inside a SpinCtrlDouble, or the item buttons inside a
# wx.RadioBox).  They are announced as a single control by screen readers,
# so walking into them only produces false "unnamed child" reports.
# Leaf composite inputs whose child windows are implementation internals
# (e.g. the edit box inside a SpinCtrlDouble, or the item buttons inside a
# wx.RadioBox).  They are announced as a single control by screen readers,
# so walking into them only produces false "unnamed child" reports.
# wx.StaticBox is intentionally NOT here: its children are the labelled
# rows inside a group box and must be audited.
_COMPOSITE = (
    wx.SpinCtrl, wx.SpinCtrlDouble, wx.RadioBox, wx.ComboBox, wx.Choice,
    wx.ListBox, wx.ListCtrl, wx.TreeCtrl, wx.Slider, wx.Gauge, wx.Button,
    wx.ToggleButton, wx.CheckBox,
)


def _effectively_shown(win: wx.Window) -> bool:
    """True when `win` and every ancestor is shown.

    wx.Window.IsShown() only reflects the window's own flag, so a box inside
    a hidden sub-panel still reports True.  Screen readers see the native
    WS_VISIBLE chain instead, so mirror that here.
    """
    w = win
    while w is not None:
        if not w.IsShown():
            return False
        w = w.GetParent()
    return True


def walk(win, out):
    for child in win.GetChildren():
        if not isinstance(child, wx.Window):
            continue
        if not _effectively_shown(child):
            continue
        is_composite = isinstance(child, _COMPOSITE)
        if isinstance(child, _INPUT_CLASSES):
            cls = child.GetClassInfo().GetClassName()
            name = (child.GetName() or "").strip()
            extra = ""
            label = ""
            if isinstance(child, wx.Button) or isinstance(child, wx.ToggleButton):
                label = child.GetLabel()
                if not name or name in _DEFAULT_NAMES:
                    if label:
                        # Windows usually derives a name from the button text,
                        # but be explicit everywhere for consistency.
                        extra += " [no explicit name; label={!r}]".format(label)
                        name = label
            line = f"{cls}| name={name!r}{extra}"
            flag = ""
            if not name or name in _DEFAULT_NAMES:
                flag = "  !DEFAULT"
            elif name.startswith("staticText"):
                flag = "  !DEFAULT"
            if flag:
                line += flag
            out.append((win, line))
        elif isinstance(child, wx.StaticText):
            txt = (child.GetLabel() or "").strip().rstrip(":")
            if txt and txt.lower() not in {
                "categories", "options", "select", "none", "description",
            }:
                child._audit_label = txt  # stash for parent-level checks
        # Descend only into genuine containers (panels, static boxes,
        # notebooks, ...).  Composite inputs (spin boxes, combos, radio
        # boxes, sliders) are announced as a single control by screen
        # readers, so their internal children must not be walked.
        if not is_composite and isinstance(
            child, (wx.Panel, wx.ScrolledWindow, wx.StaticBox, wx.Notebook),
        ):
            walk(child, out)


def dup_check(rows):
    seen = {}
    for _win, line in rows:
        if "name=" not in line:
            continue
        name = line.split("name=", 1)[1].split("|", 1)[0].strip().strip("'\"")
        if not name:
            continue
        seen.setdefault(name, []).append(line)
    for name, lines in seen.items():
        if len(lines) > 1:
            print(f"  !DUP name {name!r} used by {len(lines)} controls:")
            for l in lines[:6]:
                print("      ", l)


def dump(title, win):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    rows = []
    walk(win, rows)
    if not rows:
        print("  (no input controls)")
    for _w, line in rows:
        print("  " + line)
    dup_check(rows)


def main():
    os.makedirs("build/a11y_models", exist_ok=True)
    store = ModelStore(
        state_file=os.path.join(os.getcwd(), "build", "a11y_models", "models_state.json")
    )
    settings = Settings()

    app = wx.App(False)

    # -- Settings dialog: every category panel ---------------------------
    import ai_voice_studio.gui.settings_dialog as sd

    dlg = sd.SettingsDialog(None, settings, store)
    dlg.SetSize((1000, 900))
    dlg.Show()
    for i, cls in enumerate(dlg.CATEGORIES):
        dlg._show_category(i)
        dlg.Layout()
        dump(f"[{cls.title}] panel", dlg._panels[i])
    dlg.Destroy()

    # -- Recording dialog (uses a real sample project) -------------------
    from ai_voice_studio.gui.recording_dialog import RecordingDialog

    project_dir = os.path.join("build", "e2e_omnivoice", "design")
    if os.path.isfile(os.path.join(project_dir, "project.json")):
        rd = RecordingDialog(None, project_dir, settings, store)
        rd.SetSize((900, 800))
        rd.Show()
        dump("Recording dialog", rd)
        rd.Destroy()
    else:
        print("\nRecordingDialog skipped: no sample project at", project_dir)

    # -- OmniVoice options dialog (recording voice settings) -------------
    from ai_voice_studio.gui.omnivoice_options_dialog import OmniVoiceOptionsDialog

    od = OmniVoiceOptionsDialog(None, engine_label="OmniVoice", omni={}, project_name="demo")
    od.SetSize((760, 800))
    od.Show()
    dump("OmniVoice options dialog", od)
    od.Destroy()


if __name__ == "__main__":
    main()
