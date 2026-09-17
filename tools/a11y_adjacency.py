"""Geometric label-adjacency audit.

For every text/combo/spin input in each constructible panel/dialog, find the
StaticText that sits immediately to its LEFT on the same row (the label), and
compare that visible text with the input's accessible name.  Reports rows
where the label text (minus a trailing colon) does not match the accessible
name, which is exactly the "this box is labelled with the wrong word" defect.

Run: .venv/Scripts/python tools/a11y_adjacency.py
"""

from __future__ import annotations

import os
import sys

import wx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402

_INPUTS = (wx.TextCtrl, wx.ComboBox, wx.SpinCtrl, wx.SpinCtrlDouble)


def norm(text: str) -> str:
    return text.strip().rstrip(":").strip()


def _effectively_shown(win: wx.Window) -> bool:
    w = win
    while w is not None:
        if not w.IsShown():
            return False
        w = w.GetParent()
    return True


def find_nearby_label(win: wx.Window, ctrl: wx.Window):
    """Nearest StaticText directly to the LEFT, else directly ABOVE, `ctrl`.

    Mirrors NVDA's label association: horizontal (label left of control) for
    most inputs, vertical (label above) for list-like or multi-line ones.
    """
    best_left = best_above = None
    best_left_x = -1
    best_above_y = -1
    cx, cy = ctrl.GetPosition()
    cw, ch = ctrl.GetSize()
    for child in win.GetChildren():
        if not isinstance(child, wx.StaticText):
            continue
        if not _effectively_shown(child):
            continue
        label = norm(child.GetLabel())
        if not label:
            continue
        x, y = child.GetPosition()
        w, h = child.GetSize()
        if y + h <= cy or y >= cy + ch:
            # Not on the control's row: it is an above-label candidate if its
            # bottom edge is close above the control's top and it overlaps
            # the control horizontally.
            if y + h <= cy and cy - (y + h) <= 24 and x < cx + cw:
                if y + h > best_above_y:
                    best_above_y = y + h
                    best_above = child
            continue
        # Same row: candidate only if it ends left of the control.
        if x + w <= cx + 4 and x + w > best_left_x:
            best_left_x = x + w
            best_left = child
    return best_left or best_above


def audit(title, win, out):
    rows = []

    def walk(container):
        for child in container.GetChildren():
            if not isinstance(child, wx.Window):
                continue
            if not _effectively_shown(child):
                continue
            if isinstance(child, _INPUTS):
                parent = child.GetParent()
                label = find_nearby_label(parent, child)
                name = norm(child.GetName())
                ltext = norm(label.GetLabel()) if label else ""
                state = "OK" if (ltext and name and ltext == name) else "??"
                if state == "??":
                    rows.append(
                        f"  {child.GetClassInfo().GetClassName()} name={child.GetName()!r} "
                        f"nearbyLabel={ltext!r} state={state}"
                    )
            # recurse into panels/boxes/groups (skip internal children of
            # spinctrls and radios -- those are composite internals)
            if isinstance(child, (wx.SpinCtrl, wx.SpinCtrlDouble, wx.RadioBox)):
                continue
            if isinstance(child, wx.Window) and child.GetChildren():
                walk(child)

    walk(win)
    if rows:
        print("\n" + title)
        for r in rows:
            print(r)


def main():
    os.makedirs("build/a11y_models", exist_ok=True)
    store = ModelStore(
        state_file=os.path.join(os.getcwd(), "build", "a11y_models", "models_state.json")
    )
    settings = Settings()
    app = wx.App(False)

    import ai_voice_studio.gui.settings_dialog as sd

    dlg = sd.SettingsDialog(None, settings, store)
    dlg.SetSize((1000, 900))
    dlg.Show()
    for i, cls in enumerate(dlg.CATEGORIES):
        dlg._show_category(i)
        dlg.Layout()
        audit(f"[{cls.title}]", dlg._panels[i], None)
    dlg.Destroy()

    from ai_voice_studio.gui.recording_dialog import RecordingDialog

    project_dir = os.path.join("build", "e2e_omnivoice", "design")
    if os.path.isfile(os.path.join(project_dir, "project.json")):
        rd = RecordingDialog(None, project_dir, settings, store)
        rd.SetSize((900, 800))
        rd.Show()
        audit("Recording dialog", rd, None)
        rd.Destroy()
    else:
        print("\nRecordingDialog skipped: no sample project at", project_dir)

    from ai_voice_studio.gui.omnivoice_options_dialog import OmniVoiceOptionsDialog

    od = OmniVoiceOptionsDialog(None, engine_label="OmniVoice", omni={}, project_name="demo")
    od.SetSize((760, 800))
    od.Show()
    audit("OmniVoice options dialog", od, None)
    od.Destroy()

    from ai_voice_studio.gui.voicelab_options_dialog import VoiceLabOptionsDialog
    from ai_voice_studio.voicelab import engines as voice_lab

    for engine_id in voice_lab.engine_ids():
        vd = VoiceLabOptionsDialog(None, engine_id, values={}, project_name="demo")
        vd.SetSize((760, 800))
        vd.Show()
        audit(f"Voice Lab options dialog ({engine_id})", vd, None)
        vd.Destroy()

    from ai_voice_studio.gui.new_project_wizard import NewProjectWizard

    nw = NewProjectWizard(None, settings, store, initial_name="demo")
    nw.SetSize((900, 800))
    nw.Show()
    audit("New Project wizard", nw, None)
    nw.Destroy()

    print("\nDone.")


if __name__ == "__main__":
    main()
