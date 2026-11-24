"""A small screen-reader friendly progress dialog.

It is a plain, non-modal ``wx.Dialog`` with a labelled gauge and a status
text.  Two usage patterns:

* **Background threads** (recording, server restart): the worker updates the
  dialog through ``wx.CallAfter`` and the normal message loop repaints it.
* **Synchronous work** (document parsing in the wizard): call
  :meth:`TaskProgressDialog.show_progress` and the dialog repaints itself
  immediately (``wx.Window.Update``), so the bar moves even while the UI
  thread is busy inside a long parsing call.

An optional ``cancel`` button simply invokes the supplied callback; the
dialog stays up until :meth:`TaskProgressDialog.finish` is called so the
caller controls its lifetime.

Both the message and the position of the bar are **announced**: a listener who
is not looking at the dialog (or whose focus never left the button that started
the work) hears the same thing the status line shows, through
:mod:`ai_voice_studio.gui.announce`.  Progress is announced in 10-point steps,
so a fast job says "20 percent", "30 percent" instead of every value.
"""

from __future__ import annotations

import contextlib

import wx

from . import announce, dpi
from .a11y import finalize_accessibility

#: How far the bar must move before its position is worth announcing again.
ANNOUNCE_STEP = 10


class TaskProgressDialog(wx.Dialog):
    def __init__(
        self,
        parent: wx.Window,
        title: str = "Working",
        message: str = "Working...",
        cancel_label: str | None = None,
        on_cancel=None,
        style: int | None = None,
    ):
        """Create (but do not show) the dialog.

        ``on_cancel`` is called when the cancel button is pressed; the dialog
        remains open until :meth:`finish` is called.
        """
        if style is None:
            # No close box: lifetime is controlled by the caller.
            style = wx.CAPTION | wx.RESIZE_BORDER
        super().__init__(parent, title=title, style=style)
        self._on_cancel = on_cancel
        self._announced_value: int | None = None

        sizer = wx.BoxSizer(wx.VERTICAL)
        self._message = wx.StaticText(self, label=message)
        self._message.SetName("Progress message")
        self._message.Wrap(430)
        sizer.Add(self._message, 0, wx.ALL, 10)

        self.gauge = wx.Gauge(self, range=100, size=(420, 22))
        self.gauge.SetName("Progress")
        self.gauge.SetToolTip("Shows how much of the work has finished")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        if cancel_label:
            row = wx.BoxSizer(wx.HORIZONTAL)
            row.AddStretchSpacer(1)
            btn = wx.Button(self, label=cancel_label)
            btn.SetName(cancel_label)
            btn.SetToolTip("Cancel the current operation")
            row.Add(btn, 0, wx.ALL, 6)
            self._cancel_btn = btn
            btn.Bind(wx.EVT_BUTTON, self._on_cancel_clicked)
            sizer.Add(row, 0, wx.EXPAND)
        else:
            self._cancel_btn = None

        self.SetSizer(sizer)
        self.Fit()
        self.SetMinSize(self.GetSize())
        # Real MSAA accNames (gauge + message).
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)
        if parent is not None and parent.IsShown():
            self.CentreOnParent()
        else:
            self.Centre()

    # -- pulse (indeterminate-looking progress) -----------------------------
    def start_pulse(self, interval_ms: int = 150) -> None:
        """Animate the gauge between 0 and 95 while the length of the work
        is unknown (server restart, recording)."""
        self._pulse_value = 0
        self._pulse_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_pulse_tick, self._pulse_timer)
        self._pulse_timer.Start(interval_ms)

    def _on_pulse_tick(self, _):
        self._pulse_value = (self._pulse_value + 4) % 96
        self.show_progress(self._pulse_value)

    def stop_pulse(self) -> None:
        timer = getattr(self, "_pulse_timer", None)
        if timer is not None:
            # The timer may already be gone (the dialog was destroyed).
            with contextlib.suppress(Exception):
                timer.Stop()

    # -- helpers ------------------------------------------------------------
    def set_message(self, message: str) -> None:
        self._message.SetName("Progress message: " + message)
        self._message.Update()
        # The label and the announcement come from one string.
        announce.status_text(self._message, message, key="progress-message")

    def show_progress(self, value: int) -> None:
        """Set the gauge to ``value`` (0-100), repaint, and say where it is."""
        value = max(0, min(100, int(value)))
        self.gauge.SetValue(value)
        self.gauge.Update()
        if (self._announced_value is None
                or abs(value - self._announced_value) >= ANNOUNCE_STEP
                or value == 100):
            self._announced_value = value
            announce.announce(f"{value} percent", key="progress")

    def update(self, value: int, message: str | None = None) -> None:
        if message is not None:
            self.set_message(message)
        self.show_progress(value)

    def _on_cancel_clicked(self, _):
        if self._cancel_btn is not None:
            self._cancel_btn.Disable()
        if self._on_cancel is not None:
            self._on_cancel()

    def finish(self) -> None:
        """Close and destroy the dialog (safe to call more than once)."""
        self.stop_pulse()
        # Deliver anything still waiting rather than dropping it with the
        # window it was going to be read from.
        announce.get_announcer().flush()
        # Closing twice is allowed: the second call finds nothing to destroy.
        with contextlib.suppress(Exception):
            self.Destroy()

