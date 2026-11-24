"""Windows high contrast: detect it, and follow the user's own colours.

High contrast is not "a dark theme".  It is an accessibility mode in which the
*user* chooses every colour, and an application that keeps painting its own
palette is precisely the failure the mode exists to prevent: text the user
cannot read, a selection highlight that disappears, a wallpaper behind a
heading.  So the rule in this application is simple: while high contrast is on,
it stops choosing colours.  :func:`palette` builds a palette from the colours
Windows itself reports (``wx.SystemSettings.GetColour``), :mod:`theme` returns
that palette for *every* theme id, and the welcome screen is painted a flat
system colour with no wallpaper and no pattern.  Nothing is written to the
settings file, so the theme the user had chosen comes back the moment high
contrast is switched off.

The state is read from Windows:
``SystemParametersInfo(SPI_GETHIGHCONTRAST)`` and its ``HCF_HIGHCONTRASTON``
flag.  ``wx.SystemSettings.GetAppearance()`` is deliberately *not* used for
this: it reports light/dark appearance, not high contrast, so it would happily
answer "not dark" while the user is in a black-on-yellow high-contrast scheme.

``wx.EVT_SYS_COLOUR_CHANGED`` is what makes the change live instead of needing
a restart - Windows sends it when the scheme, the accent colour or the theme
changes.  ``gui.dpi.follow`` binds it for every top-level window.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading

import wx

log = logging.getLogger(__name__)

#: ``SystemParametersInfo`` action and the flag it fills in for us.
_SPI_GETHIGHCONTRAST = 0x0042
_HCF_HIGHCONTRASTON = 0x00000001

_lock = threading.RLock()
_override: bool | None = None


class HIGHCONTRASTW(ctypes.Structure):
    """``HIGHCONTRASTW`` from ``winuser.h``.

    ``lpszDefaultScheme`` is filled in by Windows with a pointer to its own
    string (the name the user sees in the Control Panel), which is why the
    field is a plain ``c_wchar_p`` and is never assigned by us.
    """

    _fields_ = (
        ("cbSize", ctypes.c_uint),
        ("dwFlags", ctypes.c_uint),
        ("lpszDefaultScheme", ctypes.c_wchar_p),
    )


# ---------------------------------------------------------------------------
# Reading the state
# ---------------------------------------------------------------------------
def set_override(value: bool | None) -> None:
    """Force the answer (``None`` asks Windows again).

    A test cannot switch the machine's accessibility mode on and off, and the
    audit tool that proves the high-contrast palette is in use needs both
    answers from one run.
    """
    global _override
    with _lock:
        _override = None if value is None else bool(value)


def override() -> bool | None:
    """The forced answer, or ``None`` when Windows decides."""
    with _lock:
        return _override


def read() -> tuple[bool, str]:
    """``(enabled, scheme name)`` straight from Windows (never raises)."""
    with _lock:
        forced = _override
    if forced is not None:
        return forced, "forced by override" if forced else ""
    if sys.platform != "win32":
        return False, ""
    try:
        high_contrast = HIGHCONTRASTW()
        high_contrast.cbSize = ctypes.sizeof(HIGHCONTRASTW)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            _SPI_GETHIGHCONTRAST, ctypes.sizeof(HIGHCONTRASTW),
            ctypes.byref(high_contrast), 0,
        )
        if not ok:
            return False, ""
        return (
            bool(high_contrast.dwFlags & _HCF_HIGHCONTRASTON),
            str(high_contrast.lpszDefaultScheme or ""),
        )
    except Exception:
        # A non-Windows build, or an API that is not there: the application
        # keeps its own themes, which is the pre-existing behaviour.
        log.debug("Could not read the high-contrast state", exc_info=True)
        return False, ""


def enabled() -> bool:
    """True when Windows high contrast is on (or a test forced it on)."""
    return read()[0]


def scheme_name() -> str:
    """The high-contrast scheme's own name, e.g. "High Contrast Black"."""
    return read()[1]


def describe() -> str:
    """One line for the start-up log and the audit tools."""
    on, name = read()
    if not on:
        return "high contrast: off"
    return f"high contrast: on ({name})" if name else "high contrast: on"


# ---------------------------------------------------------------------------
# The palette
# ---------------------------------------------------------------------------
def _hex(colour: wx.Colour) -> str:
    return f"#{colour.Red():02x}{colour.Green():02x}{colour.Blue():02x}"


def colour(role: int, fallback: str = "#000000") -> str:
    """One system colour as ``#rrggbb``, with a fallback if Windows says no."""
    try:
        return _hex(wx.SystemSettings.GetColour(role))
    except Exception:
        log.debug("Could not read system colour %r", role, exc_info=True)
        return fallback


def palette() -> dict[str, str]:
    """The palette the application paints with while high contrast is on.

    The keys are the ones :mod:`theme` uses everywhere, filled from the
    colours the user chose in Windows; nothing is derived, lightened or mixed,
    because that is what would destroy the contrast the user asked for.
    """
    window = colour(wx.SYS_COLOUR_WINDOW, "#000000")
    text = colour(wx.SYS_COLOUR_WINDOWTEXT, "#ffffff")
    return {
        "bg": window,
        "fg": text,
        "control_bg": window,
        "control_fg": text,
        # The system highlight is the one accent a high-contrast scheme is
        # guaranteed to keep legible against its own text colour.
        "accent": colour(wx.SYS_COLOUR_HIGHLIGHT, text),
        "heading": text,
    }


def contrast_ratio(colour_a: str, colour_b: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` colours.

    Used by the tests and by ``tools/check_dpi_layout.py`` to *prove* that the
    high-contrast palette really is high contrast rather than merely different.
    """
    def _luminance(value: str) -> float:
        text = str(value).lstrip("#")
        channels = [int(text[index:index + 2], 16) / 255 for index in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    first, second = _luminance(colour_a), _luminance(colour_b)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)
