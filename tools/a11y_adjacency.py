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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from a11y_sample import ensure_sample_project  # noqa: E402

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
    from a11y_windows import iter_windows

    app = wx.App(False)  # noqa: F841 - must stay referenced
    for title, window in iter_windows():
        audit(title, window, None)


if __name__ == "__main__":
    main()
