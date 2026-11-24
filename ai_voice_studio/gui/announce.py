"""Live announcements: push changes to the screen reader (SPEC 3.2).

A screen reader only learns about a control when the user navigates to it.  A
progress bar that moves, a status line that changes its text and a validation
error that appears in a label are all *silent*: the application repaints, and
the listener hears nothing.  This module pushes them instead.

How the push works
------------------
On Windows the message is delivered through **MSAA live regions**, the same
mechanism a browser uses for ``aria-live``.  The announcer owns one 1x1
``wx.StaticText`` per window (the *live region*) whose accessible name is set
to the message, then fires ``EVENT_OBJECT_LIVEREGIONCHANGED`` and, as a
fallback for readers that only watch name changes, ``EVENT_OBJECT_NAMECHANGE``
on it.  ``a11y.LiveRegionAccessible`` gives that control
``STATE_SYSTEM_LIVE_REGION``, which is what tells JAWS and NVDA to read it
without being asked.

The region is never the only place the message exists: callers also put it in
the visible status line / progress message, so a reader that ignores live
regions still finds it where it has always been.

Politeness and pacing
---------------------
``polite`` announcements wait their turn (progress, status); a progress bar can
update a hundred times a second and a screen reader reading every value would
never finish a sentence, so:

* repeats of the text just delivered are dropped;
* a polite message is delivered at most once every
  :data:`DEFAULT_MIN_INTERVAL` seconds;
* while that interval runs, a later polite message **replaces** the earlier one
  with the same ``key`` ("latest wins": the position of the progress bar, not
  every position it passed through), and messages with different keys queue
  behind it (at most :data:`MAX_PENDING`), so a status line update cannot be
  starved by a progress stream.

``assertive`` announcements (an error the user has to act on) bypass both the
interval and the queue, and drop whatever polite message was waiting: reading a
stale "45 percent" after an error is worse than not reading it at all.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import wx

from .a11y import LiveRegionAccessible, update_accessible_name

log = logging.getLogger(__name__)

#: MSAA event ids (``winuser.h``): no ``wx`` constant exists for these.
EVENT_OBJECT_NAMECHANGE = 0x800C
EVENT_OBJECT_LIVEREGIONCHANGED = 0x8019

#: ``NotifyWinEvent`` arguments for "the object itself".
OBJID_CLIENT = wx.OBJID_CLIENT
CHILDID_SELF = 0

POLITE = "polite"
ASSERTIVE = "assertive"

#: Shortest gap between two polite announcements, in seconds.  Fast enough that
#: a listener hears progress move, slow enough that a sentence can be finished.
DEFAULT_MIN_INTERVAL = 0.6

#: How many *different* polite messages may wait behind each other.
MAX_PENDING = 4

#: How often the pending queue is pumped while something is waiting, in ms.
PUMP_INTERVAL_MS = 120

#: How many delivered messages are remembered (tests and the a11y audit read
#: this; it is not shown to the user).
HISTORY_LENGTH = 20


def notify_win_event(hwnd: int, event_id: int) -> bool:
    """Fire one MSAA event at ``hwnd``'s client object.

    Separate from :meth:`Announcer._deliver` so tests can record the events
    instead of sending them to the desktop, and so a non-Windows run (and a
    window that has already gone) degrades to a logged no-op rather than an
    exception in the middle of a UI callback.
    """
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        import ctypes  # only needed on the delivery path

        ctypes.windll.user32.NotifyWinEvent(
            int(event_id), int(hwnd), OBJID_CLIENT, CHILDID_SELF
        )
        return True
    except Exception as exc:  # pragma: no cover - depends on the OS
        log.debug("Could not raise MSAA event %s: %s", hex(event_id), exc)
        return False


def _default_is_gui_thread() -> bool:
    """True when called from wx's GUI thread (``wx.CallAfter`` needs that)."""
    try:
        return wx.IsMainThread()
    except Exception:  # pragma: no cover - no app yet
        return threading.current_thread() is threading.main_thread()


def _default_post(callable_) -> None:
    """Run ``callable_`` on the GUI thread."""
    wx.CallAfter(callable_)


class _Pending:
    """One polite message waiting for its turn."""

    __slots__ = ("key", "text")

    def __init__(self, text: str, key: str | None):
        self.text = text
        self.key = key

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Pending({self.text!r}, key={self.key!r})"


def is_live_region(control) -> bool:
    """True for the invisible live region an announcer adds to a window.

    The region is a 1x1 static text at the window's corner whose label is the
    last message announced.  It is deliberately empty most of the time, so
    anything measuring the interface has to be able to leave it out: its 1x1
    minimum size is not a layout defect, it is what makes it invisible.
    """
    return getattr(control, "_live_region", False)


class Announcer:
    """Delivers announcements one at a time, newest first within a key."""

    def __init__(
        self,
        window: wx.Window | None = None,
        *,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        post=None,
        clock=time.monotonic,
        is_gui_thread=None,
        notify=None,
        auto_pump: bool = True,
        follow_active: bool = False,
    ):
        self._window = window
        #: Keep the live region on the window the user is in.  The application
        #: announcer sets this: otherwise the region created on the main window
        #: would stay there while a modal dialog has the focus, and the reader
        #: would be listening to a window that is no longer active.
        self.follow_active = follow_active
        self._region: wx.StaticText | None = None
        self._min_interval = float(min_interval)
        self._post = post or _default_post
        self._clock = clock
        self._is_gui_thread = is_gui_thread or _default_is_gui_thread
        self._notify = notify or notify_win_event
        self._auto_pump = auto_pump
        self._lock = threading.RLock()
        self._pending: list[_Pending] = []
        self._last_text: str | None = None
        self._last_time: float | None = None
        self._history: list[str] = []
        #: Set to False to make the whole layer silent (tests, and a user
        #: preference would go here if one is ever added).
        self.enabled = True

    # -- target window / live region ---------------------------------------

    def attach(self, window: wx.Window | None) -> None:
        """Announce through ``window``'s live region from now on."""
        with self._lock:
            if window is not self._window:
                self._region = None
            self._window = window

    def follow_the_active_window(self) -> wx.Window | None:
        """Point the region at the active window, if it moved.

        Called just before an announcement.  Only on the GUI thread, and only
        for an announcer that asked to follow: a background worker and a test
        with a fixed window both have a target already.
        """
        if not self.follow_active or not self._is_gui_thread():
            return self._window
        try:
            active = wx.GetActiveWindow()
        except Exception:  # pragma: no cover - no GUI yet
            return self._window
        if active is not None and active is not self._window and self._alive(active):
            self.attach(active)
        return self._window

    def _region_for(self) -> wx.StaticText | None:
        """The live-region control, created on first use or after a destroy.

        Parented to the announcer's window and *not* added to any sizer: it is
        1x1, sits in the corner and is painted in the window's own background
        colour, so it is invisible while staying in the MSAA tree - a hidden
        control is not, and a control that is not in the tree cannot be
        announced.
        """
        window = self._window
        if window is None or not self._alive(window):
            return None
        region = self._region
        if region is not None and self._alive(region):
            return region
        try:
            colours = window.GetBackgroundColour()
            region = wx.StaticText(window, size=(1, 1), pos=(0, 0), label="")
            region.SetBackgroundColour(colours)
            region.SetForegroundColour(colours)
            region.SetAccessible(LiveRegionAccessible(region))
            region._acc_name = ""  # type: ignore[attr-defined]
            region._live_region = True  # type: ignore[attr-defined]
            region.Show()
        except Exception as exc:
            log.debug("Could not create the live region on %s: %s", window, exc)
            return None
        self._region = region
        return region

    @staticmethod
    def _alive(window) -> bool:
        """True while ``window`` is a live wx object."""
        try:
            return bool(window) and not window.IsBeingDeleted()
        except Exception:
            return False

    # -- announcing ---------------------------------------------------------

    def announce(
        self,
        text: str,
        *,
        assertive: bool = False,
        key: str | None = None,
    ) -> bool:
        """Push ``text`` to the screen reader.

        Returns True when the message was taken (delivered, queued, or posted
        to the GUI thread); False for an empty message, a repeat of what was
        just read, or a disabled announcer.
        """
        text = " ".join(str(text or "").split())
        if not text:
            return False
        with self._lock:
            if not self.enabled:
                return False
            if not self._is_gui_thread():
                # Background threads (a download worker, an engine's progress
                # callback) cannot touch wx; the message is re-posted and the
                # caller is told it was accepted.
                self._post(
                    lambda: self.announce(text, assertive=assertive, key=key)
                )
                return True
            if not assertive and text == self._last_text:
                return False
            now = self._clock()
            if assertive:
                # An error wins: drop the queued percentages, they are stale.
                self._pending.clear()
                self._deliver(text, now=now, assertive=True)
                return True
            if self._ready(now):
                self._deliver(text, now=now, assertive=False)
                return True
            self._enqueue(_Pending(text, key))
            self._schedule_pump()
            return True

    def error(self, text: str, *, key: str | None = None) -> bool:
        """Announce a validation error (assertive: it interrupts)."""
        return self.announce(text, assertive=True, key=key)

    def pump(self, now: float | None = None) -> bool:
        """Deliver the oldest waiting polite message if its turn has come.

        Called by the internal ``wx.CallLater`` while a queue exists; tests
        call it directly with their own clock.
        """
        with self._lock:
            if not self._pending:
                return False
            now = self._clock() if now is None else now
            if not self._ready(now):
                if self._auto_pump:
                    self._schedule_pump()
                return False
            pending = self._pending.pop(0)
            self._deliver(pending.text, now=now, assertive=False)
            if self._pending:
                self._schedule_pump()
            return True

    def flush(self) -> list[str]:
        """Deliver everything waiting, in order, and return the texts sent.

        Used when a long operation ends: the final "Finished" must not be
        dropped because a percentage was read a moment ago.
        """
        with self._lock:
            sent = [pending.text for pending in self._pending]
            self._pending.clear()
            now = self._clock()
            for text in sent:
                self._deliver(text, now=now, assertive=False)
            return sent

    # -- delivery -----------------------------------------------------------

    def _ready(self, now: float) -> bool:
        """True when the polite interval has elapsed since the last delivery."""
        if self._last_time is None:
            return True
        return now - self._last_time >= self._min_interval

    def _enqueue(self, pending: _Pending) -> None:
        """Queue a polite message, replacing an earlier one with the same key."""
        if pending.key is not None:
            for index, waiting in enumerate(self._pending):
                if waiting.key == pending.key:
                    self._pending[index] = pending
                    return
        self._pending.append(pending)
        # Bounded: a burst of unrelated status text must not build a backlog.
        del self._pending[:-MAX_PENDING]

    def _deliver(self, text: str, *, now: float, assertive: bool) -> None:
        region = self._region_for()
        if region is not None:
            try:
                update_accessible_name(region, text)
                self._notify(region.GetHandle(), EVENT_OBJECT_LIVEREGIONCHANGED)
                self._notify(region.GetHandle(), EVENT_OBJECT_NAMECHANGE)
            except Exception as exc:
                log.debug("Could not hand %r to the screen reader: %s", text, exc)
        self._last_text = text
        self._last_time = now
        self._history.append(text)
        del self._history[:-HISTORY_LENGTH]
        log.debug("Announced (%s): %s", ASSERTIVE if assertive else POLITE, text)

    def _schedule_pump(self) -> None:
        """Ask for a ``pump`` once the interval has run out (GUI thread only)."""
        if not self._auto_pump:
            return
        wx.CallLater(int(self._min_interval * 1000) + PUMP_INTERVAL_MS, self._pump_tick)

    def _pump_tick(self) -> None:
        with self._lock:
            if self._alive(self._window):
                self.pump()

    # -- inspection (tests, tools/a11y_audit.py) ----------------------------

    @property
    def target(self) -> wx.Window | None:
        """The window whose live region announcements go through."""
        return self._window

    @property
    def live_region(self) -> wx.StaticText | None:
        """The MSAA live-region control, once one has been created."""
        return self._region

    @property
    def history(self) -> list[str]:
        """The messages delivered most recently, oldest first."""
        with self._lock:
            return list(self._history)

    @property
    def pending(self) -> list[str]:
        """Messages waiting for their turn."""
        with self._lock:
            return [pending.text for pending in self._pending]

    def reset(self) -> None:
        """Forget the history, the queue and the pacing (for a fresh run)."""
        with self._lock:
            self._pending.clear()
            self._history.clear()
            self._last_text = None
            self._last_time = None


# ---------------------------------------------------------------------------
# The application-wide announcer
# ---------------------------------------------------------------------------

_announcer: Announcer | None = None
_announcer_lock = threading.Lock()


def get_announcer() -> Announcer:
    """The process-wide announcer, attached to the active top-level window."""
    global _announcer
    with _announcer_lock:
        if _announcer is None:
            _announcer = Announcer(follow_active=True)
        return _announcer


def set_announcer(announcer: Announcer | None) -> None:
    """Replace the process-wide announcer (tests, and window teardown)."""
    global _announcer
    with _announcer_lock:
        _announcer = announcer


def attach(window: wx.Window | None) -> Announcer:
    """Point the application announcer at ``window``."""
    announcer = get_announcer()
    announcer.attach(window)
    return announcer


def announce(text: str, *, assertive: bool = False, key: str | None = None) -> bool:
    """Push ``text`` to the screen reader through the application announcer."""
    announcer = get_announcer()
    # Announcements come from dialogs, panels and background threads, so the
    # target is resolved here rather than wired by every caller - and it
    # follows the user from the main window into a modal dialog.
    announcer.follow_the_active_window()
    return announcer.announce(text, assertive=assertive, key=key)


def announce_error(text: str, *, key: str | None = None) -> bool:
    """Announce a validation error or a refusal (assertive)."""
    return announce(text, assertive=True, key=key)


def status_error(label: wx.StaticText, message: str, *, key: str | None = None) -> None:
    """Set a status label to a validation error *and* say it at once.

    A refusal ("Choose a document to read first", "Could not start
    recording: ...") has to interrupt: the user is waiting for the action to
    happen, and a polite message queued behind a progress stream would arrive
    after they have already pressed the button again.
    """
    label.SetLabel(message)
    announce_error(message, key=key)


def status_text(label: wx.StaticText, message: str, *, key: str | None = None) -> None:
    """Set a visible status label *and* announce what it now says.

    Every window's "status line" goes through here, so a status update is never
    visible-only: the label is what a sighted user reads and the announcement
    is what a listener hears, from one string.
    """
    label.SetLabel(message)
    announce(message, key=key)


__all__ = [
    "ASSERTIVE",
    "DEFAULT_MIN_INTERVAL",
    "EVENT_OBJECT_LIVEREGIONCHANGED",
    "EVENT_OBJECT_NAMECHANGE",
    "MAX_PENDING",
    "POLITE",
    "Announcer",
    "announce",
    "announce_error",
    "attach",
    "get_announcer",
    "notify_win_event",
    "set_announcer",
    "status_error",
    "status_text",
]
