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


def walk(win, out):
    for child in win.GetChildren():
        if not isinstance(child, wx.Window):
            continue
        cls = child.GetClassInfo().GetClassName()
        name = (child.GetName() or "").strip()
        if isinstance(child, _INPUT_CLASSES):
            extra = ""
            label = ""
            if isinstance(child, wx.Button) or isinstance(child, wx.ToggleButton):
                label = child.GetLabel()
                if name and not label:
                    extra = f" (btn label {label!r})"
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
        # Recurse into every window that can contain child windows (panels,
        # scrolled windows, static boxes, notebooks, ...).
        if isinstance(child, wx.Panel) or isinstance(child, wx.ScrolledWindow) \
                or isinstance(child, wx.StaticBox) or isinstance(child, wx.Notebook) \
                or isinstance(child, wx.BoxSizer) or child.GetChildren():
            if isinstance(child, wx.Window) and child.GetChildren():
                walk(child, out)
            elif not isinstance(child, wx.Window):
                pass
            else:
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
