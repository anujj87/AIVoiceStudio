"""Accessibility helpers (SPEC 3.2).

wxPython on Windows exposes ``SetName()`` as the MSAA/UIA accessible name for a
control, which is what NVDA/JAWS/Narrator announce. The convention used here:
every labelled control gets a ``wx.StaticText`` label placed immediately before
it in the layout AND its accessible name set to the label text, so both browse
mode and focus mode announce it correctly.
"""

from __future__ import annotations

import wx

from .theme import current_theme_colors


def add_labeled(
    parent: wx.Window,
    sizer: wx.Sizer,
    label_text: str,
    control: wx.Window,
    flag: int = wx.ALL,
    border: int = 4,
    label_extra_flags: int = 0,
    control_flag: int = 0,
) -> wx.StaticText:
    """Add ``<label_text> <control>`` to ``sizer`` with a screen-reader name."""
    label = wx.StaticText(parent, label=label_text + ":")
    _tint(label)
    sizer.Add(label, 0, flag | wx.ALIGN_CENTER_VERTICAL | label_extra_flags, border)
    control.SetName(label_text)
    sizer.Add(control, 1, flag | control_flag, border)
    return label


def add_check(
    parent: wx.Window,
    sizer: wx.Sizer,
    label_text: str,
    checked: bool = False,
) -> wx.CheckBox:
    """CheckBox with an accessible name (SPEC: single choices use check boxes)."""
    box = wx.CheckBox(parent, label=label_text)
    box.SetValue(checked)
    box.SetName(label_text)
    sizer.Add(box, 0, wx.ALL, 4)
    return box


def make_radio(
    parent: wx.Window,
    label_text: str,
    group: wx.RadioBox,
) -> wx.RadioButton:
    """RadioButton whose accessible name matches its label."""
    btn = wx.RadioButton(parent, label=label_text, style=wx.RB_GROUP if False else 0)
    btn.SetName(label_text)
    return btn


def _tint(control: wx.Window) -> None:
    """Apply current theme colors to a freshly created control."""
    colors = current_theme_colors()
    try:
        if isinstance(control, wx.StaticText):
            control.SetForegroundColour(colors["fg"])
    except Exception:  # noqa: BLE001
        pass


def set_accessible_name(control: wx.Window, name: str) -> None:
    control.SetName(name)
