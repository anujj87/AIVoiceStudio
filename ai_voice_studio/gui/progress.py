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
"""

from __future__ import annotations

import wx

from .a11y import _tint, finalize_accessibility


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
            try:
                timer.Stop()
            except Exception:  # noqa: BLE001
                pass

    # -- helpers ------------------------------------------------------------
    def set_message(self, message: str) -> None:
        self._message.SetLabel(message)
        self._message.SetName("Progress message: " + message)
        self._message.Update()

    def show_progress(self, value: int) -> None:
        """Set the gauge to ``value`` (0-100) and repaint immediately."""
        self.gauge.SetValue(max(0, min(100, int(value))))
        self.gauge.Update()

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
        try:
            self.Destroy()
        except Exception:  # noqa: BLE001 - already destroyed
            pass


def tint(control: wx.Window) -> None:
    """Apply the current theme colours to a fresh control (re-export)."""
    _tint(control)
