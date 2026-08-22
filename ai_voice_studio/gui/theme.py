"""Theme manager (SPEC: General tab -> Light / Dark / System default).

Applies explicit colours to container/text controls (buttons keep the native
look so they stay accessible). System default follows the OS appearance on
Windows via ``wx.SystemSettings.GetAppearance()``.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

import wx

from ..constants import THEME_DARK, THEME_LIGHT, THEME_SYSTEM

log = logging.getLogger(__name__)

_LIGHT: Dict[str, str] = {
    "bg": "#ffffff",
    "fg": "#1a1a1a",
    "control_bg": "#ffffff",
    "control_fg": "#1a1a1a",
}
_DARK: Dict[str, str] = {
    "bg": "#1e1e1e",
    "fg": "#e8e8e8",
    "control_bg": "#2d2d2d",
    "control_fg": "#e8e8e8",
}

_state = {"theme": THEME_SYSTEM, "colors": _LIGHT}
_lock = threading.Lock()


def current_theme() -> str:
    with _lock:
        return _state["theme"]


def current_theme_colors() -> Dict[str, str]:
    with _lock:
        return dict(_state["colors"])


def _resolve(theme: str) -> str:
    if theme == THEME_SYSTEM:
        try:
            appearance = wx.SystemSettings.GetAppearance()
            return THEME_DARK if appearance.IsDark() else THEME_LIGHT
        except Exception:  # noqa: BLE001
            return THEME_LIGHT
    return theme


def apply_theme(top: wx.Window, theme: str) -> None:
    """Apply the theme to ``top`` and all of its descendants."""
    resolved = _resolve(theme)
    colors = _DARK if resolved == THEME_DARK else _LIGHT
    with _lock:
        _state["theme"] = theme
        _state["colors"] = colors

    _paint(top, colors)
    for child in _descendants(top):
        _paint(child, colors)
    top.Refresh()


def _descendants(window: wx.Window):
    stack = list(window.GetChildren())
    while stack:
        child = stack.pop()
        yield child
        stack.extend(child.GetChildren())


def _paint(window: wx.Window, colors: Dict[str, str]) -> None:
    try:
        if isinstance(window, (wx.Panel, wx.Dialog, wx.Frame, wx.ScrolledWindow)):
            window.SetBackgroundColour(colors["bg"])
        elif isinstance(
            window,
            (wx.StaticText, wx.CheckBox, wx.RadioButton, wx.StaticBox, wx.RadioBox),
        ):
            window.SetForegroundColour(colors["fg"])
        elif isinstance(
            window,
            (wx.TextCtrl, wx.ListBox, wx.ComboBox, wx.ListCtrl, wx.Choice, wx.SpinCtrl),
        ):
            window.SetBackgroundColour(colors["control_bg"])
            window.SetForegroundColour(colors["control_fg"])
    except Exception:  # noqa: BLE001
        pass
