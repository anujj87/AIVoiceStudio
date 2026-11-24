"""Display scaling: awareness, and scale-derived geometry.

wxMSW does **not** scale the pixel values an application passes to sizers
(borders) or to ``wx.Window`` (sizes, minimum sizes).  At 150% those numbers
are read as if the display were 96 DPI, so a window that asked for a 700-pixel
minimum and a 6-pixel margin keeps both while the text inside it grows by half
again: the padding disappears and controls are cramped or clipped.  The fix is
to treat every number in the layout code as a **DIP** (device-independent
pixel, i.e. a pixel at 100%) and multiply it by the window's scale factor.

Doing that by hand at ~160 call sites would mean wrapping every ``border=6``
and ``size=(200, 10)`` in a conversion, and it would still be wrong the moment
a window moved to another monitor.  So the conversion happens once per window,
after its controls are built: :func:`prepare` records the window's laid-out
geometry - which at that moment *is* the DIP geometry the code wrote - then
applies it multiplied by the scale factor.  From then on :func:`rescale`
re-derives the geometry from the same recorded numbers, so a window that moves
from a 100% monitor to a 200% one is exact rather than compounded.

What scales:

* every explicitly set minimum and maximum size, on every control in the
  window (a control created as ``wx.ListCtrl(size=(200, 10))`` carries that as
  its minimum size, so this covers ``size=`` arguments too);
* every sizer border, and the minimum size of every spacer;
* the window's own size, so the dialog keeps the same footprint in DIPs;
* *not* fonts: wx sizes point-based fonts through the target display's DPI
  already, which is why the text is correct today and only the geometry is not.

At 100% every one of those operations is the identity, so a machine at 100%
lays out exactly as it did before this module existed.

:func:`follow` keeps a window in step while it is open - ``wx.EVT_DPI_CHANGED``
for a move to a monitor with a different scale, ``wx.EVT_SYS_COLOUR_CHANGED``
for a Windows theme or high-contrast change - and re-themes it through
:mod:`ai_voice_studio.gui.theme`, which reads the high-contrast state from
:mod:`ai_voice_studio.gui.contrast`.

The scale factor can be forced with :func:`set_scale_override` so the layout
can be proved at 100%, 150% and 200% on a 100% machine
(``tools/check_dpi_layout.py`` does exactly that).
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import sys
import threading

import wx

log = logging.getLogger(__name__)

#: ``DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`` is a *handle*, not an enum:
#: Win32 hands it around as a pointer whose value is -4.
_PER_MONITOR_AWARE_V2 = -4
#: ``PROCESS_PER_MONITOR_DPI_AWARE`` for the Windows 8.1 fallback below.
_PROCESS_PER_MONITOR_AWARE = 2

#: Attribute the recorder leaves on a prepared window (this codebase's usual
#: way of annotating a wx object: ``child._acc_name`` is set the same way).
_STATE = "_aivs_dpi_state"
_FOLLOWING = "_aivs_dpi_follows"

_lock = threading.RLock()
_override: float | None = None


# ---------------------------------------------------------------------------
# The scale factor
# ---------------------------------------------------------------------------
def scale_override() -> float | None:
    """The forced scale factor, or ``None`` when the display decides."""
    with _lock:
        return _override


def set_scale_override(scale: float | None) -> None:
    """Force the scale factor (``None`` restores the real display).

    Used by the tests and by ``tools/check_dpi_layout.py`` to lay a window out
    at 150% or 200% on a 100% machine.
    """
    global _override
    with _lock:
        _override = None if scale is None else max(0.25, float(scale))


@contextlib.contextmanager
def at_scale(scale: float | None):
    """Run a block with the scale factor forced, restoring it afterwards."""
    previous = scale_override()
    set_scale_override(scale)
    try:
        yield
    finally:
        set_scale_override(previous)


def current_scale(window: wx.Window | None = None) -> float:
    """The scale factor in force: 1.0 at 100%, 1.5 at 150%, 2.0 at 200%.

    The forced value (tests, tools) wins; otherwise the window's own display
    answers, and a window that cannot answer - or a non-Windows run - is 1.0.
    """
    with _lock:
        if _override is not None:
            return _override
    if window is not None:
        try:
            factor = float(window.GetDPIScaleFactor())
            if factor > 0:
                return factor
        except Exception:
            log.debug("Could not read the scale factor; assuming 100%%", exc_info=True)
    return 1.0


def px(value: float, window: wx.Window | None = None) -> int:
    """A length written in DIPs, in the display's pixels."""
    return round(float(value) * current_scale(window))


def px_size(size: tuple[int, int], window: wx.Window | None = None) -> tuple[int, int]:
    """A ``(width, height)`` pair in DIPs, in the display's pixels.

    A negative component means "let the control decide" and is passed through.
    """
    width, height = size
    return (
        px(width, window) if width > 0 else width,
        px(height, window) if height > 0 else height,
    )


# ---------------------------------------------------------------------------
# Scaling a built window
# ---------------------------------------------------------------------------
def prepare(window: wx.Window, *, scale: float | None = None) -> float:
    """Lay ``window`` out at the display's scale, recording what it asked for.

    Call it once, after the window's sizers are assembled and before it is
    shown.  The geometry the code wrote is read back as the DIP geometry and
    applied multiplied by the scale factor; the recorded numbers stay on the
    window so a later :func:`rescale` is exact.  Returns the scale applied.
    """
    factor = current_scale(window) if scale is None else float(scale)
    state = getattr(window, _STATE, None)
    if state is None:
        state = _record(window)
        setattr(window, _STATE, state)
    _apply(window, state, factor)
    return factor


def rescale(window: wx.Window, *, scale: float | None = None) -> float:
    """Re-derive the geometry of a prepared window at a new scale.

    The window's current size becomes the new DIP size first, so a window the
    user had resized keeps that size (in DIPs) across the move to another
    monitor instead of snapping back to the size it opened with.
    """
    factor = current_scale(window) if scale is None else float(scale)
    state = getattr(window, _STATE, None)
    if state is None:
        return prepare(window, scale=factor)
    _rebase_size(window, state)
    _apply(window, state, factor)
    return factor


def scale_of(window: wx.Window) -> float | None:
    """The scale a window was last laid out at, or ``None`` if never prepared."""
    state = getattr(window, _STATE, None)
    return None if state is None else float(state["scale"])


def size_from_content(window: wx.Window) -> bool | None:
    """Whether the window was sized by its content when it was prepared.

    Such a window is fitted rather than multiplied (see :func:`_resize_root`),
    so its size does not have to be a multiple of the scale: 23 pixels of text
    become 34 at 150%, not 34.5.  ``None`` when the window was never prepared.
    """
    state = getattr(window, _STATE, None)
    return None if state is None else bool(state.get("size_from_content"))


def scale_added_item(sizer: wx.Sizer, dip_border: int) -> None:
    """Scale a sizer item that was added *after* its window was prepared.

    The one place this happens is a settings category built on first visit: it
    is added to a sizer that was already recorded, so its padding would stay
    at the DIP value while everything around it grew.  The item joins the
    owner's recorded geometry, so a later display change scales it too.

    The owner is the nearest prepared window above the sizer.  When there is
    none - a category added before its dialog finished being built - the item
    is deliberately left at its DIP value: the dialog's own prepare is about
    to record it, and scaling it here as well would scale it twice.
    """
    index = sizer.GetItemCount() - 1
    if index < 0:
        return
    owner = _prepared_owner(sizer)
    if owner is None:
        return
    try:
        item = sizer.GetItem(index)
        if item is not None and dip_border:
            item.SetBorder(round(dip_border * current_scale(owner)))
    except Exception:
        log.debug("Could not scale a newly added sizer item", exc_info=True)
        return
    owner._aivs_dpi_state["items"].append([sizer, index, int(dip_border), None])


def _prepared_owner(sizer: wx.Sizer) -> wx.Window | None:
    """The nearest window above ``sizer`` that carries recorded geometry."""
    node = None
    try:
        node = sizer.GetContainingWindow()
    except Exception:
        log.debug("Could not find the window a sizer belongs to", exc_info=True)
    while node is not None:
        if getattr(node, _STATE, None) is not None:
            return node
        node = node.GetParent()
    return None


def summary(window: wx.Window) -> str:
    """One line describing a window's scaling, for logs and audit tools."""
    state = getattr(window, _STATE, None)
    if state is None:
        return f"{window.__class__.__name__}: not prepared (scale {current_scale(window):.2f})"
    windows = state["windows"]
    return (
        f"{window.__class__.__name__}: scale {state['scale']:.2f}, "
        f"{len(windows)} controls, {state['items']} sizer items, "
        f"size {tuple(window.GetSize())}"
    )


# -- recording ---------------------------------------------------------------
def _walk(window: wx.Window) -> list[wx.Window]:
    """``window`` and its descendants that it is responsible for scaling.

    A descendant that was prepared on its own - a settings category, built
    after the dialog that holds it - keeps its own recorded geometry and is
    left out here, both so its sizes cannot be scaled twice and so a rescale
    of the dialog does not fight its state (it is delegated to instead, see
    :func:`_prepared_children`).
    """
    out = [window]
    stack = list(window.GetChildren())
    while stack:
        child = stack.pop(0)
        if getattr(child, _STATE, None) is not None:
            continue
        out.append(child)
        stack.extend(child.GetChildren())
    return out


def _prepared_children(window: wx.Window) -> list[wx.Window]:
    """The topmost descendants that scale themselves, in tree order."""
    out = []
    stack = list(window.GetChildren())
    while stack:
        child = stack.pop(0)
        if getattr(child, _STATE, None) is not None:
            out.append(child)  # its own rescale covers everything below it
            continue
        stack.extend(child.GetChildren())
    return out


def _unset(size) -> tuple[int, int] | None:
    """``None`` for wx's "not set" pair, so it is left alone when applying."""
    width, height = size
    if width < 0 and height < 0:
        return None
    return (width, height)


def _record(window: wx.Window) -> dict:
    """Read the DIP geometry the code wrote, before anything is scaled."""
    state: dict = {"scale": 1.0, "windows": [], "items": [], "fonts": [],
                   "size": None}
    for child in _walk(window):
        state["windows"].append((
            child,
            _unset(child.GetMinSize()),
            _unset(child.GetMaxSize()),
        ))
        try:
            state["fonts"].append((child, child.GetFont()))
        except Exception:
            log.debug("Could not read a control's font", exc_info=True)
        sizer = child.GetSizer()
        if sizer is not None:
            _record_sizer(state, sizer)
    # A window whose size *is* its content (Fit / SetSizeHints) must not have
    # that size multiplied: the content already grew with the scale, and
    # scaling it again would double it.  It is fitted instead (see
    # :func:`_resize_root`).
    size = tuple(window.GetSize())
    try:
        best = tuple(window.GetBestSize())
    except Exception:
        best = (-1, -1)
    state["size"] = size
    state["size_from_content"] = size[0] <= best[0] or size[1] <= best[1]
    return state


def _record_sizer(state: dict, sizer: wx.Sizer) -> None:
    """Record one sizer's borders (and its spacers' sizes), depth first."""
    for index in range(sizer.GetItemCount()):
        item = sizer.GetItem(index)
        if item is None:
            continue
        spacer = item.GetMinSize() if item.IsSpacer() else None
        state["items"].append([
            sizer,
            index,
            int(item.GetBorder()),
            _unset(spacer) if spacer else None,
        ])
        nested = item.GetSizer()
        if nested is not None:
            _record_sizer(state, nested)


# -- applying ----------------------------------------------------------------
def _scaled(size: tuple[int, int] | None, factor: float) -> tuple[int, int] | None:
    if size is None:
        return None
    width, height = size
    return (
        round(width * factor) if width > 0 else width,
        round(height * factor) if height > 0 else height,
    )


def _rebase_size(window: wx.Window, state: dict) -> None:
    """Turn the window's live size back into DIPs at the scale it was laid out.

    Only meaningful while the scale is changing: it is what keeps a window the
    user had resized the same size in DIPs after it lands on another display.
    """
    previous = float(state["scale"]) or 1.0
    if previous <= 0 or previous == current_scale(window) or state["size_from_content"]:
        return
    try:
        width, height = window.GetSize()
    except Exception:
        return
    state["size"] = (round(width / previous), round(height / previous))


def _apply(window: wx.Window, state: dict, factor: float) -> None:
    """Set every recorded length to ``factor`` times its DIP value."""
    _apply_fonts(state, factor)
    dead: list = []
    for entry in state["windows"]:
        control, min_size, max_size = entry
        try:
            scaled_min = _scaled(min_size, factor)
            if scaled_min is not None:
                control.SetMinSize(scaled_min)
            scaled_max = _scaled(max_size, factor)
            if scaled_max is not None:
                control.SetMaxSize(scaled_max)
        except RuntimeError:
            # The control was destroyed since it was recorded (a settings
            # category that was thrown away and rebuilt): forget it.
            dead.append(entry)
        except Exception:
            log.debug("Could not scale a control; leaving it alone", exc_info=True)
    for entry in dead:
        state["windows"].remove(entry)

    for item_state in state["items"]:
        sizer, index, border, spacer_min = item_state
        try:
            if index >= sizer.GetItemCount():
                continue
            item = sizer.GetItem(index)
            if item is None:
                continue
            if border:
                item.SetBorder(round(border * factor))
            scaled_spacer = _scaled(spacer_min, factor)
            if scaled_spacer is not None:
                item.SetMinSize(scaled_spacer)
        except Exception:
            log.debug("Could not scale a sizer item; leaving it alone", exc_info=True)

    if state["size"]:
        _resize_root(window, state, factor)
    # Lay the tree out again, parents first: a page that is not on screen has
    # never been laid out since the geometry changed, and its controls would
    # otherwise keep the size they were given at the old scale until something
    # happened to rearrange them.  A scrolled window needs its virtual size
    # recalculated first (its content just grew), then a second pass because
    # the scrollbars it decided on change every size inside it again.
    _relayout(state["windows"])
    _relayout(state["windows"])
    state["scale"] = factor
    # Subtrees with their own recorded geometry (a settings category built on
    # first visit) re-derive theirs from the same new factor.
    for child in _prepared_children(window):
        rescale(child, scale=factor)


def _relayout(windows: list) -> None:
    """Re-run the layout of every window in the list, parents first.

    A window that has never had an area - a wizard page that has not been
    shown - is skipped: laying it out would distribute nothing and squash the
    controls inside it to their minimum, which is worse than leaving them.
    """
    for control in windows:
        try:
            size = control.GetClientSize()
            if size[0] <= 0 or size[1] <= 0:
                continue
            setup = getattr(control, "SetupScrolling", None)
            if callable(setup):
                setup()
            control.Layout()
        except RuntimeError:
            continue
        except Exception:
            log.debug("Could not lay a control out again", exc_info=True)


def _apply_fonts(state: dict, factor: float) -> None:
    """Scale the fonts as well - only while the scale is forced.

    On a real display change wx resolves a point-based font through the new
    DPI by itself, and scaling it here as well would double it.  A forced
    factor is a *synthetic* display with no such help: without this the
    geometry would grow by 150% while every label stayed at 100%, which is not
    the world a user at 150% sees - and an audit run in that world would miss
    the text that no longer fits.
    """
    if scale_override() is None:
        return
    for control, font in state["fonts"]:
        try:
            control.SetFont(font.Scaled(factor))
            control.InvalidateBestSize()
        except RuntimeError:
            continue
        except Exception:
            log.debug("Could not scale a control's font", exc_info=True)
    for control, _font in state["fonts"]:
        _synthetic_size(control)


def _synthetic_size(control: wx.Window) -> None:
    """Re-derive the size of a control that has never been laid out.

    Such a control still carries the size it was created with, and in a
    synthetic display that size was measured with the *old* font, so it no
    longer matches the text inside it.  Sizing it to what it needs reproduces
    what creating the window at this scale would have produced; controls that
    have been laid out are left exactly as the audit needs to see them.
    """
    try:
        parent = control.GetParent()
        if parent is None:
            return
        area = parent.GetClientSize()
        if area[0] > 0 and area[1] > 0:
            return
        best = control.GetBestSize()
        if best[0] > 0 and best[1] > 0 and tuple(control.GetSize()) != tuple(best):
            control.SetSize(best)
    except RuntimeError:
        return
    except Exception:
        log.debug("Could not re-derive a control's size", exc_info=True)


def _resize_root(window: wx.Window, state: dict, factor: float) -> None:
    """Give the window its recorded size, or its content's, in pixels.

    An explicit size (``size=(820, 560)``) is scaled, so the window keeps the
    same footprint in DIPs.  A size that came from the content is *not*
    scaled - the content already grew - so the window is fitted to it instead.
    Either way a window too small for its content is fitted, so the conversion
    can never clip something.
    """
    dip_size = state["size"]
    try:
        if state["size_from_content"]:
            window.Fit()
            # Never smaller than the footprint it had in DIPs: a few parts of a
            # window are sized by wx itself and do not take part in the
            # conversion (the status bar, the scrollbar thickness), so a
            # content-fitted window can come out a little short - and a short
            # window squashes whatever is above those parts.
            target = (round(dip_size[0] * factor), round(dip_size[1] * factor))
            current = tuple(window.GetSize())
            if current[0] < target[0] or current[1] < target[1]:
                window.SetSize((max(current[0], target[0]),
                                max(current[1], target[1])))
            return
        target = (round(dip_size[0] * factor), round(dip_size[1] * factor))
        window.SetSize(target)
        best = window.GetBestSize()
        if best[0] > target[0] or best[1] > target[1]:
            window.Fit()
    except Exception:
        log.debug("Could not resize a window to its scaled size", exc_info=True)
    _grow_to_fit_children(window)


def _grow_to_fit_children(window: wx.Window) -> None:
    """Never leave a window with less room than its own children need.

    A dialog whose content sits in a panel the dialog itself has no sizer for
    (the first-launch terms window) reports nothing useful from
    ``GetBestSize()``, so the size the dialog was fitted to at 100% can come
    out a few pixels short at 150% - and the buttons at the bottom of that
    panel then lose their height.  Growing to the widest and tallest child
    costs nothing when there is already room for it.
    """
    try:
        for _ in range(2):  # a size change can raise what a child needs
            client = tuple(window.GetClientSize())
            need = [0, 0]
            for child in window.GetChildren():
                if not child.IsShown():
                    continue
                best = child.GetBestSize()
                need[0] = max(need[0], best[0])
                need[1] = max(need[1], best[1])
            missing = (max(0, need[0] - client[0]), max(0, need[1] - client[1]))
            if missing == (0, 0):
                return
            size = tuple(window.GetSize())
            window.SetSize((size[0] + missing[0], size[1] + missing[1]))
    except Exception:
        log.debug("Could not grow a window to fit its children", exc_info=True)


# ---------------------------------------------------------------------------
# Following the system while the window is open
# ---------------------------------------------------------------------------
def follow(window: wx.Window) -> None:
    """Keep ``window`` in step with the display and with Windows itself.

    ``wx.EVT_DPI_CHANGED`` fires when the window lands on a display with a
    different scale (per-monitor DPI awareness is what makes Windows send it),
    ``wx.EVT_SYS_COLOUR_CHANGED`` when the user switches Windows theme,
    high-contrast scheme or accent colour.  Both re-derive the geometry from
    the recorded DIP values and re-theme through :mod:`~ai_voice_studio.gui.theme`,
    so the change is applied while the window is open rather than at the next
    start.  Binding twice is a no-op.
    """
    if getattr(window, _FOLLOWING, False):
        return
    setattr(window, _FOLLOWING, True)
    window.Bind(wx.EVT_DPI_CHANGED, _on_dpi_changed)
    window.Bind(wx.EVT_SYS_COLOUR_CHANGED, _on_system_colours_changed)


def _root_of(window: wx.Window | None) -> wx.Window | None:
    """The top-level window to relayout: a child's geometry lives with it."""
    if window is None:
        return None
    try:
        return window.GetTopLevelParent() or window
    except Exception:
        return window


def _on_dpi_changed(event) -> None:
    root = _root_of(event.GetEventObject())
    if root is not None:
        try:
            factor = rescale(root)
            _retheme(root)
            root.Layout()
            root.Refresh()
            _announce(f"Display scaling is now {round(factor * 100)} percent")
            log.info("Display scaling changed to %.0f%% for %s",
                     factor * 100, root.__class__.__name__)
        except Exception:
            log.exception("Could not follow the display scaling change")
    # wx scales its own controls (borders of native widgets, scrollbars) when
    # the event is allowed to continue.
    event.Skip()


def _on_system_colours_changed(event) -> None:
    root = _root_of(event.GetEventObject())
    if root is not None:
        try:
            _retheme(root)
            root.Layout()
            root.Refresh()
            from . import contrast

            _announce("Windows high contrast is on" if contrast.enabled()
                      else "Windows colours changed")
            log.info("System colours changed (high contrast %s)",
                     "on" if contrast.enabled() else "off")
        except Exception:
            log.exception("Could not follow the Windows colour change")
    event.Skip()


def _retheme(root: wx.Window) -> None:
    """Re-apply the current theme, so the new scale/colours are painted."""
    from . import theme

    theme.apply_theme(root, theme.current_theme(), _settings_of(root))


def _settings_of(window: wx.Window):
    """The settings object the window was built with, if it carries one."""
    node = window
    while node is not None:
        settings = getattr(node, "settings", None)
        if settings is not None:
            return settings
        node = node.GetParent()
    return None


def _announce(text: str) -> None:
    """Say it through the announcement layer, when there is one."""
    try:
        from . import announce

        announce.announce(text, key="system")
    except Exception:
        log.debug("No announcement layer to report %r through", text, exc_info=True)


# ---------------------------------------------------------------------------
# Process DPI awareness
# ---------------------------------------------------------------------------
def enable_process_awareness() -> str:
    """Make the process per-monitor-v2 DPI aware, before any window exists.

    The packaged application declares this in its own manifest
    (``packaging/dpi_aware.manifest``, embedded by the PyInstaller spec); this
    is the path that matters when the application is run from source, where
    there is no manifest to read and an unaware process gets its windows
    bitmap-stretched by Windows at 150% and 200% (blurry, and reported to a
    screen reader as one image rather than as the controls inside it).

    Returns the awareness that was established, for the start-up log.  Calling
    it after a window exists fails harmlessly and reports "unchanged".
    """
    if sys.platform != "win32":
        return "not-windows"
    if getattr(sys, "frozen", False):
        return "manifest"
    user32 = ctypes.windll.user32
    try:
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2)):
            return "per-monitor-v2"
    except (AttributeError, OSError):
        log.debug("SetProcessDpiAwarenessContext is not available", exc_info=True)
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_AWARE) == 0:
            return "per-monitor"
    except (AttributeError, OSError):
        log.debug("SetProcessDpiAwareness is not available", exc_info=True)
    try:
        if user32.SetProcessDPIAware():
            return "system"
    except (AttributeError, OSError):
        log.debug("SetProcessDPIAware is not available", exc_info=True)
    return "unchanged"


def awareness() -> str:
    """How DPI-aware this process is, in words (start-up log, audit tools)."""
    if sys.platform != "win32":
        return "not-windows"
    try:
        user32 = ctypes.windll.user32
        context = user32.GetThreadDpiAwarenessContext()
        if context and user32.AreDpiAwarenessContextsEqual(
                ctypes.c_void_p(context), ctypes.c_void_p(_PER_MONITOR_AWARE_V2)):
            return "per-monitor-v2"
        level = user32.GetAwarenessFromDpiAwarenessContext(ctypes.c_void_p(context))
    except (AttributeError, OSError):
        log.debug("Could not read the process DPI awareness", exc_info=True)
        return "unknown"
    return {0: "unaware", 1: "system", 2: "per-monitor"}.get(level, f"unknown ({level})")


def describe_display(window: wx.Window | None = None) -> str:
    """Scale, awareness and monitor geometry - one line for the start-up log."""
    factor = current_scale(window)
    try:
        geometry = wx.Display().GetGeometry()
        display = f"{geometry.GetWidth()}x{geometry.GetHeight()}"
    except Exception:
        display = "unknown size"
    return (f"DPI awareness: {awareness()}; display scaling: "
            f"{round(factor * 100)}% ({display})")
