"""Reusable message dialogs (screen-reader friendly).

The stock ``wx.MessageBox`` only offers standard button labels.  Several
places in the app need *two meaningful buttons* instead of just OK:

* recording finished -> ``OK`` / ``Open project folder``
* an OmniVoice Server error -> ``Restart OmniVoice server now`` /
  ``I will restart later``

This module provides a small labelled-button dialog plus helpers that keep
the button semantics consistent across the whole application.
"""

from __future__ import annotations

import logging
import os
import threading
import webbrowser

import wx

from .progress import TaskProgressDialog

log = logging.getLogger(__name__)


def run_action_dialog(
    parent: wx.Window,
    title: str,
    message: str,
    actions: list[tuple[str, str]],
    default_key: str | None = None,
) -> str | None:
    """Show a dialog with custom buttons.

    ``actions`` is a list of ``(key, label)`` pairs.  The chosen key is
    returned, or ``None`` when the dialog was dismissed (Escape / close).
    Each button carries the same accessible name as its visible label.
    """
    dlg = _ActionDialog(parent, title, message, actions, default_key)
    try:
        if dlg.ShowModal() == wx.ID_OK:
            return dlg.selected_key
        return None
    finally:
        dlg.Destroy()


class _ActionDialog(wx.Dialog):
    """Dialog whose buttons are plain labelled wx.Buttons (NVDA friendly)."""

    def __init__(
        self,
        parent: wx.Window,
        title: str,
        message: str,
        actions: list[tuple[str, str]],
        default_key: str | None = None,
    ):
        super().__init__(parent, title=title)
        self.selected_key: str | None = None
        self._keys: dict[int, str] = {}

        sizer = wx.BoxSizer(wx.VERTICAL)
        text = wx.StaticText(self, label=message)
        text.SetName(title)
        text.Wrap(520)
        sizer.Add(text, 0, wx.EXPAND | wx.ALL, 12)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        btn_row.AddStretchSpacer(1)
        default_btn = None
        for key, label in actions:
            btn = wx.Button(self, label=label)
            btn.SetName(label)
            btn.SetToolTip(label)
            self._keys[btn.GetId()] = key
            if default_key is not None and key == default_key:
                default_btn = btn
            btn_row.Add(btn, 0, wx.ALL, 6)
            btn.Bind(wx.EVT_BUTTON, self._on_action)
        sizer.Add(btn_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.SetSizer(sizer)
        self.Fit()
        self.SetMinSize(self.GetSize())
        if parent is not None and parent.IsShown():
            self.CentreOnParent()
        else:
            self.Centre()
        # Focus the default action so Enter activates the safe choice.
        target = default_btn or (btn_row.GetChildren()[0].GetWindow()
                                 if btn_row.GetChildren() else None)
        if target is not None:
            wx.CallAfter(target.SetFocus)

    def _on_action(self, evt):
        self.selected_key = self._keys.get(evt.GetId())
        self.EndModal(wx.ID_OK)


def open_folder(path: str) -> None:
    """Open ``path`` in the OS file manager (best effort)."""
    if not path or not os.path.isdir(path):
        return
    try:
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: SIM115 - Windows
    except Exception:  # noqa: BLE001
        try:
            webbrowser.open("file:///" + os.path.normpath(path).replace("\\", "/"))
        except Exception:  # noqa: BLE001
            log.debug("Could not open folder %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Recording complete
# ---------------------------------------------------------------------------
def show_recording_complete(parent: wx.Window, message: str, project_folder: str) -> None:
    """Show the end-of-recording message with OK / Open project folder."""
    key = run_action_dialog(
        parent,
        "Recording complete",
        message,
        [
            ("open", "Open project folder"),
            ("ok", "OK"),
        ],
        default_key="ok",
    )
    if key == "open":
        open_folder(project_folder)


# ---------------------------------------------------------------------------
# OmniVoice Server error handling
# ---------------------------------------------------------------------------
_SERVER_HINTS = (
    "omnivoice server",
    "omnivoice_server",
    "server is not responding",
    "server did not become ready",
    "server exited prematurely",
    "failed to start server",
    "synthesis failed",
    "clone synthesis failed",
    "try restarting it from settings",
    "connection test failed",
)

# Request-level failures: the server answered (4xx/5xx with a JSON error
# body), so it is alive and a restart would not change the outcome.  The
# message now includes the server's own detail via _http_error_detail
# ("server returned HTTP 500: ...").
_REQUEST_LEVEL_HINTS = (
    "server returned http 4",
    "server returned http 5",
)


def looks_like_server_error(message: str) -> bool:
    """True when an engine error message points at the OmniVoice HTTP server.

    Request-level failures (the server answered with an HTTP error and its
    own message, e.g. "server returned HTTP 500: Synthesis failed: ...")
    do NOT count: the server is demonstrably up, so offering a restart
    would only confuse.  Those show as a plain error box with the real
    server-side detail.
    """
    low = (message or "").lower()
    if any(hint in low for hint in _REQUEST_LEVEL_HINTS):
        return False
    return any(hint in low for hint in _SERVER_HINTS)


def notify_engine_error(parent: wx.Window, title: str, message: str) -> None:
    """Show an engine error.

    Errors that mention the OmniVoice server get a two-button dialog
    (restart the server now / restart later) instead of a plain OK box.
    All other errors keep the plain error box.
    """
    if looks_like_server_error(message):
        ask_omni_server_restart(parent, message, title)
        return
    wx.MessageBox(message, title, style=wx.OK | wx.ICON_ERROR)


def ask_omni_server_restart(parent: wx.Window, message: str, title: str) -> None:
    """Offer to restart the OmniVoice server after a server-side error."""
    text = (
        str(message).rstrip() + "\n\n"
        "The OmniVoice server is not working. Restart it now?"
    )
    key = run_action_dialog(
        parent,
        title,
        text,
        [
            ("restart", "Restart OmniVoice server now"),
            ("later", "I will restart later"),
        ],
        default_key="later",
    )
    if key == "restart":
        restart_omni_server(parent)


def restart_omni_server(parent: wx.Window) -> None:
    """Restart the OmniVoice server (background) and report the result."""
    dlg = TaskProgressDialog(
        parent,
        title="Restarting OmniVoice server",
        message="Stopping the server and starting it again. This can take a "
                "couple of minutes, please wait.",
    )
    dlg.Show()
    dlg.start_pulse()

    def _job():
        ok, text = False, "OmniVoice server restart failed."
        try:
            from ..omnivoice_server import get_server_manager  # noqa: PLC0415
            from ..settings import Settings  # noqa: PLC0415

            cfg = Settings().get("omnivoice_server", {})
            mgr = get_server_manager(
                host=cfg.get("host", "127.0.0.1"),
                port=int(cfg.get("port", 8881)),
                device=cfg.get("device", "cuda"),
                num_steps=int(cfg.get("num_steps", 32)),
                max_concurrent=int(cfg.get("max_concurrent", 2)),
                api_key=cfg.get("api_key", ""),
                cors_origins=cfg.get("cors_origins", ""),
            )
            mgr.restart(timeout=180.0)
            ok, text = True, f"OmniVoice server restarted at {mgr.base_url}."
        except Exception as exc:  # noqa: BLE001
            ok, text = False, f"Could not restart the OmniVoice server: {exc}"
        wx.CallAfter(_done, ok, text)

    def _done(success: bool, text: str):
        dlg.finish()
        style = wx.OK | (wx.ICON_INFORMATION if success else wx.ICON_ERROR)
        wx.MessageBox(text, "OmniVoice Server", style=style)

    threading.Thread(target=_job, daemon=True).start()
