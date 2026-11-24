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
* ``add_labeled()`` / ``add_check()`` -- build a row whose label text and
  accName agree.

A *window's* own accName is its title, never the name wxWidgets invented for
it.  Every window is born with a name of wx's making - the window's class in
lower case, so ``GetName()`` answers ``"frame"`` for a frame and ``"dialog"``
for a dialog - and a screen reader asking for the title of the window
(NVDA+T) reads that accName.  Announcing "frame" instead of "AI Voice Studio"
is exactly what this module used to do; ``_named_by_the_app()`` is what stops
it, and the title is read live so a window that renames itself as a project
opens reports the new one.
"""

from __future__ import annotations

import contextlib

import wx

from .theme import current_theme_colors


class _NameAccessible(wx.Accessible):
    """wx.Accessible that reports ``win._acc_name`` as the MSAA accName.

    For list controls (wx.ListBox / wx.ListCtrl) the child ids are the
    rows: their accNames must be the item text, otherwise a screen reader
    announces the list's own name for every row instead of the project/
    profile names when the user moves through them with arrow keys.

    A top-level window is the one case where the accName is not stored in
    ``_acc_name``: it is the window's title, read on every request.
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
        win = self.win
        if isinstance(win, wx.TopLevelWindow):
            # The window's own name is its title, and it is read live: a
            # window renames itself as a project opens (the describe window
            # becomes "Audio description - <film>"), so a copy taken when the
            # window was built would announce the previous title.
            return (wx.ACC_OK, win.GetTitle() or "")
        return (wx.ACC_OK, getattr(win, "_acc_name", "") or "")

    def GetChildCount(self):
        win = self.win
        if isinstance(win, wx.ListBox):
            return (wx.ACC_OK, win.GetCount())
        if isinstance(win, wx.ListCtrl):
            return (wx.ACC_OK, win.GetItemCount())
        return wx.Accessible.GetChildCount(self)


#: ``STATE_SYSTEM_LIVE_REGION`` from ``oleacc.h``: the state flag that makes
#: JAWS and NVDA read an object's name when it changes, without the object
#: having focus and without the user asking for it (see ``gui/announce.py``).
STATE_SYSTEM_LIVE_REGION = 0x08000000


class LiveRegionAccessible(_NameAccessible):
    """Accessible for the announcer's off-screen status control.

    A static text with ``STATE_SYSTEM_LIVE_REGION`` is what MSAA calls a live
    region: setting its accName and raising ``EVENT_OBJECT_LIVEREGIONCHANGED``
    is the desktop equivalent of ``aria-live``, and it is the only way to push
    a message to a screen reader in an application that has no focused control
    to move.
    """

    def GetRole(self, childId):
        return (wx.ACC_OK, wx.ROLE_SYSTEM_STATICTEXT)

    def GetState(self, childId):
        code, state = wx.Accessible.GetState(self, childId)
        return (code, state | STATE_SYSTEM_LIVE_REGION)


_SKIP_TYPES = (wx.StaticText, wx.Button, wx.Panel, wx.StaticBox)
# Composite controls whose OS children must not get their own accessibles
# (their internals are announced as part of the parent control by NVDA).
_COMPOSITE = (wx.ComboBox, wx.Choice, wx.SpinCtrl, wx.SpinCtrlDouble,
              wx.RadioBox, wx.Gauge)

#: The names wxWidgets gives a window nobody has named: ``GetName()`` answers
#: one of these for every window the application has not itself named, so
#: treating one as an accName is how the main window came to be announced as
#: "frame".  They are collected from the real widget set rather than derived
#: from the class name, which does not fit (a wxCheckBox is ``"check"``, a
#: wxScrolledWindow is ``"panel"``, a wxTextCtrl is ``"text"``); the widget
#: walk in ``tests/test_a11y_names.py`` fails when wx grows another one.
_DEFAULT_WINDOW_NAMES = frozenset({
    "", "button", "collapsiblePane", "check", "choice", "colourpicker",
    "comboBox", "dialog",
    "dirpicker", "filepicker", "frame", "gauge", "groupBox", "listBox",
    "listCtrl", "listbook", "notebook", "panel", "radioButton",
    "scrolledWindow", "searchCtrl", "simplebook", "slider", "spinCtrl",
    "spinCtrlDouble", "staticBox", "staticLine", "staticText", "statusBar",
    "text", "toolbar", "treeCtrl", "wxSpinCtrl", "wxSpinCtrlDouble",
})


def _named_by_the_app(control: wx.Window) -> bool:
    """True when ``control``'s name was chosen here, not by wxWidgets.

    A window keeps the class-derived name it was born with until the
    application names it, and that leftover must never become an accessible
    name: a screen reader would announce it as if it were the control's own
    label (and, for a window, as if it were the window's title).
    """
    name = control.GetName()
    return bool(name) and name not in _DEFAULT_WINDOW_NAMES


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


def finalize_accessibility(
    container: wx.Window,
    *,
    install_access_keys: bool = True,
) -> None:
    """Attach real MSAA accNames to every named, non-button control in
    ``container``.

    Runs once after the UI is built: controls that already carry a custom
    ``wx.Accessible`` (e.g. settings panels) are left untouched, and
    buttons / static text are skipped because MSAA already exposes their
    visible text as the name.

    Every window that finishes this step also gets the application's own
    access-key handling (``access_keys.install``).  Windows answers a
    button's ``&`` mnemonic along one narrow path that can silently miss
    (see ``access_keys``), so the window-level hook is what makes Alt+D,
    Alt+R, Alt+P and friends dependable here.

    ``install_access_keys=False`` names the controls without adding the key
    hook: a window built *after* its dialog already installed one (a settings
    category built on first visit) would otherwise carry a second hook, and
    key events reach the dialog's own hook by bubbling up anyway.
    """
    if install_access_keys:
        from . import access_keys

        access_keys.install(container)

    def walk(win: wx.Window) -> None:
        if not isinstance(win, _SKIP_TYPES) and win.GetAccessible() is None:
            if isinstance(win, wx.TopLevelWindow):
                # The window's accName is its title (see _NameAccessible), so
                # NVDA+T announces "AI Voice Studio" and not the class name
                # wxWidgets left on the window.
                win._acc_name = win.GetTitle()  # type: ignore[attr-defined]
                win.SetAccessible(_NameAccessible(win))
            elif _named_by_the_app(win):
                win._acc_name = win.GetName()  # type: ignore[attr-defined]
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


def _tint(control: wx.Window) -> None:
    """Apply current theme colors to a freshly created control."""
    colors = current_theme_colors()
    # A control that refuses a colour is not a failure: the drawing continues in
    # the theme's own colours, which is what the caller wanted anyway.
    with contextlib.suppress(Exception):
        if isinstance(control, wx.StaticText):
            control.SetForegroundColour(colors["fg"])