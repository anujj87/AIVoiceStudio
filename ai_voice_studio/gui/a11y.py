"""Accessibility helpers (SPEC 3.2).

wxPython on Windows exposes ``SetName()`` as the wx-level name, but on this
wx build that name does NOT reach MSAA/UIA: ``get_accName`` on a text box,
combo box or slider returns the auto-associated neighbour text (or nothing)
instead of ``SetName``.  What a screen reader like NVDA actually announces
comes from ``IAccessible::get_accName``.

The reliable mechanism is a custom ``wx.Accessible`` whose ``GetName``
override returns the label.  These helpers:

* ``set_accessible_name()`` -- give a control a real MSAA accName.
* ``update_accessible_name()`` -- cheap update for dynamic names (sliders).
* ``finalize_accessibility()`` -- walk a dialog/page once and attach real
  accNames to every named non-button control (buttons and static text
  already expose their own visible text to MSAA).
* ``add_labeled()`` / ``add_check()`` / ``make_radio()`` -- build a row
  whose label text and accName agree.
"""

from __future__ import annotations

import wx

from .theme import current_theme_colors


class _NameAccessible(wx.Accessible):
    """wx.Accessible that reports ``win._acc_name`` as the MSAA accName.

    For list controls (wx.ListBox / wx.ListCtrl) the child ids are the
    rows: their accNames must be the item text, otherwise a screen reader
    announces the list's own name for every row instead of the project/
    profile names when the user moves through them with arrow keys.
    """

    def __init__(self, win: wx.Window):
        super().__init__(win)
        self.win = win

    def GetName(self, childId):
        if childId > 0:
            win = self.win
            index = childId - 1
            # MSAA may ask for any child id, including ones past the end (a
            # screen reader probing the list, or a stale id after the rows
            # changed).  GetString/GetItemText assert on an invalid index, so
            # the range is checked here: asking for a row that is not there
            # returns "not implemented" instead of raising.
            if isinstance(win, wx.ListBox):
                if not 0 <= index < win.GetCount():
                    return (wx.ACC_NOT_IMPLEMENTED, "")
                text = win.GetString(index)
                return (wx.ACC_OK, text or "")
            if isinstance(win, wx.ListCtrl):
                if not 0 <= index < win.GetItemCount():
                    return (wx.ACC_NOT_IMPLEMENTED, "")
                text = win.GetItemText(index)
                return (wx.ACC_OK, text or "")
            return (wx.ACC_NOT_IMPLEMENTED, "")
        return (wx.ACC_OK, getattr(self.win, "_acc_name", "") or "")

    def GetChildCount(self):
        win = self.win
        if isinstance(win, wx.ListBox):
            return (wx.ACC_OK, win.GetCount())
        if isinstance(win, wx.ListCtrl):
            return (wx.ACC_OK, win.GetItemCount())
        return wx.Accessible.GetChildCount(self)


_SKIP_TYPES = (wx.StaticText, wx.Button, wx.Panel, wx.StaticBox)
# Composite controls whose OS children must not get their own accessibles
# (their internals are announced as part of the parent control by NVDA).
_COMPOSITE = (wx.ComboBox, wx.Choice, wx.SpinCtrl, wx.SpinCtrlDouble,
              wx.RadioBox, wx.Gauge)


def set_accessible_name(control: wx.Window, name: str) -> None:
    """Give ``control`` a real MSAA accName (what NVDA announces)."""
    control.SetName(name)
    control._acc_name = name  # type: ignore[attr-defined]
    if not isinstance(control.GetAccessible(), _NameAccessible):
        control.SetAccessible(_NameAccessible(control))


def update_accessible_name(control: wx.Window, name: str) -> None:
    """Update a previously attached accessible name (no re-attach cost)."""
    control._acc_name = name  # type: ignore[attr-defined]
    control.SetName(name)


def finalize_accessibility(container: wx.Window) -> None:
    """Attach real MSAA accNames to every named, non-button control in
    ``container``.

    Runs once after the UI is built: controls that already carry a custom
    ``wx.Accessible`` (e.g. settings panels) are left untouched, and
    buttons / static text are skipped because MSAA already exposes their
    visible text as the name.
    """
    def walk(win: wx.Window) -> None:
        if not isinstance(win, _SKIP_TYPES):
            if win.GetAccessible() is None:
                name = win.GetName()
                if name and name not in ("staticText", "text"):
                    win._acc_name = name  # type: ignore[attr-defined]
                    win.SetAccessible(_NameAccessible(win))
        if not isinstance(win, _COMPOSITE):
            for child in win.GetChildren():
                walk(child)

    walk(container)


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
    set_accessible_name(control, label_text)
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
    set_accessible_name(box, label_text)
    sizer.Add(box, 0, wx.ALL, 4)
    return box


def make_radio(
    parent: wx.Window,
    label_text: str,
    group: wx.RadioBox,
) -> wx.RadioButton:
    """RadioButton whose accessible name matches its label."""
    btn = wx.RadioButton(parent, label=label_text, style=wx.RB_GROUP if False else 0)
    set_accessible_name(btn, label_text)
    return btn


def _tint(control: wx.Window) -> None:
    """Apply current theme colors to a freshly created control."""
    colors = current_theme_colors()
    try:
        if isinstance(control, wx.StaticText):
            control.SetForegroundColour(colors["fg"])
    except Exception:  # noqa: BLE001
        pass