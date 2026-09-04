"""End-to-end test of the OmniVoice Server voice-profile manager panel.

Drives the REAL code path used by the Settings dialog's "OmniVoice Server"
panel (``_OmniVoiceServerPanel``):

    Start server -> list profiles -> create profile from reference audio
                 -> list again (shows the clone) -> delete -> list again

The omnivoice server itself is spawned by the app's server manager with the
*managed* Python runtime (same as pressing "Start server" in the UI), so a
GPU + the managed runtime with omnivoice-server installed are required.

Headless wx gotcha handled here: worker threads finish their HTTP work and
post results back with ``wx.CallAfter``; those callbacks only dispatch while
a real ``MainLoop`` runs (``wx.Yield`` alone is not enough on Windows), so
this harness runs a MainLoop and advances stages from the main thread via
``wx.CallAfter`` / ``wx.CallLater`` pollers.  ``wx.MessageBox`` is stubbed so
no modal dialog can block the run.

Usage:  .venv/Scripts/python tools/e2e_server_profiles.py
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import tempfile
import threading
import time

# Make the project root importable when run as tools/e2e_server_profiles.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join("build", "e2e_server_profiles")

# ---------------------------------------------------------------------------
# Watchdog: if anything hangs, kill the process with a clear marker instead
# of blocking the caller forever.  (Model load + clone inference on GPU.)
# ---------------------------------------------------------------------------
WATCHDOG_SECONDS = 560


def _watchdog() -> None:
    print("\nWATCHDOG: e2e_server_profiles.py exceeded "
          f"{WATCHDOG_SECONDS}s, aborting.", flush=True)
    os._exit(124)  # noqa: PLR2801 - hard exit is the point of a watchdog


_watchdog_timer = threading.Timer(WATCHDOG_SECONDS, _watchdog)
_watchdog_timer.daemon = True
_watchdog_timer.start()


def _log() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("e2e-server-profiles")


log = _log()

# ---------------------------------------------------------------------------
# Free port + clean Settings (never touch the user's real settings)
# ---------------------------------------------------------------------------
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
log.info("Using port %d", PORT)

tmp_settings = os.path.join(tempfile.mkdtemp(prefix="avs-e2e-"), "settings.json")

REF_AUDIO = os.path.abspath(os.path.join(
    "build", "e2e_omnivoice", "design",
    "01 This is an end to end test of OmniVoice.wav",
))
REF_TEXT = "This is an end to end test of OmniVoice in AI Voice Studio."

# ---------------------------------------------------------------------------
# Import wx + the panel, then stub modal dialogs BEFORE anything runs
# ---------------------------------------------------------------------------
import wx  # noqa: E402

_modal_calls: list[tuple[str, str, int]] = []


def _fake_message_box(message, caption="", style=wx.OK, **kwargs):  # noqa: ANN001
    """Auto-answer every MessageBox so a headless run can never block."""
    _modal_calls.append((caption, message, style))
    log.info("MessageBox [%s]: %s", caption, message.splitlines()[0][:120])
    return wx.YES if style & wx.YES_NO else wx.OK


wx.MessageBox = _fake_message_box  # type: ignore[method-assign]

from ai_voice_studio.gui import settings_dialog  # noqa: E402
from ai_voice_studio.omnivoice_server import get_server_manager  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402

# ---------------------------------------------------------------------------
# Harness: main thread runs the wx MainLoop; stages advance via CallAfter.
# ---------------------------------------------------------------------------

_failure: list[str] = []
_done = threading.Event()


def _fail(msg: str) -> None:
    log.error("FAIL: %s", msg)
    _failure.append(msg)
    _finish()


def _finish() -> None:
    try:
        _app.frame.Destroy()
    except Exception:  # noqa: BLE001
        pass
    wx.CallLater(150, _app.app.ExitMainLoop)
    _done.set()


class Harness:
    def __init__(self) -> None:
        self.app = wx.App(redirect=False)
        self.frame = wx.Frame(None, title="e2e omni server", size=(900, 760))
        self.panel = settings_dialog._OmniVoiceServerPanel(  # noqa: SLF001
            self.frame, Settings(path=tmp_settings)
        )
        self.panel.host_ctrl.SetValue("127.0.0.1")
        self.panel.port_ctrl.SetValue(PORT)
        self.frame.Show()
        # Trace which buttons get clicked (helpful when debugging the run).
        self.frame.Bind(wx.EVT_BUTTON, self._on_button)
        self.app.Yield()
        # Drainer prints the real server logs as they arrive.
        mgr = get_server_manager(host="127.0.0.1", port=PORT)
        threading.Thread(
            target=lambda: self._drain_when_up(mgr), daemon=True
        ).start()

    @staticmethod
    def _on_button(evt) -> None:  # noqa: ANN001
        btn = evt.GetEventObject()
        log.info("BUTTON CLICK: %s", getattr(btn, "GetLabel", lambda: "?")())
        evt.Skip()

    @staticmethod
    def _drain_when_up(mgr) -> None:  # noqa: ANN001
        for _ in range(1200):  # poll up to ~120 s for the subprocess handle
            proc = getattr(mgr, "_proc", None)  # noqa: SLF001
            if proc is not None:
                try:
                    for line in proc.stdout:
                        print(f"  [server] {line.rstrip()}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
                return
            time.sleep(0.1)

    # -- main-thread helpers (safe: run inside the MainLoop) -------------
    def later(self, fn, ms: int = 0) -> None:
        """Run ``fn`` on the main thread; any exception fails the run with
        its message instead of dying inside the wx event loop."""

        def _safe():
            if _failure:
                return
            try:
                name = getattr(fn, "__name__", repr(fn))
                if name != "_tick":
                    log.info(">> %s", name)
                fn()
            except Exception as exc:  # noqa: BLE001
                _fail(f"{getattr(fn, '__name__', '?')} raised: {exc}")

        if ms:
            wx.CallLater(ms, _safe)
        else:
            wx.CallAfter(_safe)

    def poll_until(self, cond, deadline: float, what: str, on_ok,
                   on_success=None, on_fail=None, interval: float = 0.2):  # noqa: ANN001
        """Re-schedule itself until ``cond`` (checked on the main thread).
        ``on_success`` (if any) runs only after the condition succeeds.
        Note: the app replaces "saved."/"deleted." status labels with a
        follow-up refresh in the SAME event-loop turn, so polls must not
        depend on transient label text -- use server-visible state."""

        def _tick():
            if _failure:
                return
            try:
                ok = cond()
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                on_ok()
                if on_success is not None:
                    self.later(on_success)
                return
            if on_fail is not None and on_fail():
                return
            if time.monotonic() >= deadline:
                _fail(f"Timed out waiting for: {what}")
                return
            self.later(_tick, max(50, int(interval * 1000)))

        _tick()

    # -- server-visible helpers -------------------------------------------
    def _server_ids(self) -> list:
        mgr = get_server_manager(host="127.0.0.1", port=PORT)
        try:
            return [p["profile_id"] for p in mgr.list_profiles()]
        except Exception:  # noqa: BLE001 - server mid-restart etc.
            return []

    # -- stages (each runs on the main thread) ------------------------------
    def stage_start(self) -> None:
        log.info("== Stage: start server (pressing 'Start server') ==")
        self.panel._on_start(None)  # noqa: SLF001 - same handler the button binds
        self.poll_until(
            cond=lambda: "Server started at" in self.panel.status_label.GetLabel()
            or "Running at" in self.panel.status_label.GetLabel()
            or "Failed to start" in self.panel.status_label.GetLabel(),
            deadline=time.monotonic() + 300.0,
            what="server to become ready (model load on GPU)",
            on_ok=lambda: log.info("Status: %s",
                                   self.panel.status_label.GetLabel()),
            on_success=self.check_running,
        )

    def check_running(self) -> None:
        label = self.panel.status_label.GetLabel()
        if "Failed to start" in label:
            _fail(f"Server failed to start: {label}")
            return
        assert "Server started at" in label or "Running at" in label, label
        mgr = get_server_manager(host="127.0.0.1", port=PORT)
        if not mgr.is_running:
            _fail("Server manager reports not running after 'Running' label.")
            return
        self.later(self.stage_initial_list)

    def stage_initial_list(self) -> None:
        log.info("== Stage: initial profile list ==")
        self.poll_until(
            cond=lambda: "profile(s)" in self.panel.profiles_status.GetLabel()
            or "Could not load" in self.panel.profiles_status.GetLabel(),
            deadline=time.monotonic() + 60.0,
            what="initial profile list refresh",
            on_ok=lambda: log.info("Profiles status: %s",
                                   self.panel.profiles_status.GetLabel()),
            on_success=self.check_initial_list,
        )

    def check_initial_list(self) -> None:
        # The profile store is persistent on disk, so earlier interrupted
        # runs can leave e2e_prof_* entries behind.  Purge only our own ids.
        mgr = get_server_manager(host="127.0.0.1", port=PORT)
        leftovers = [
            p["profile_id"] for p in mgr.list_profiles()
            if str(p.get("profile_id", "")).startswith("e2e_prof_")
        ]
        for pid in leftovers:
            mgr.delete_profile(pid)
            log.info("Purged leftover test profile %s", pid)
        before = mgr.list_profiles()
        log.info("Profiles on server before create: %d", len(before))
        assert all(str(p.get("profile_id", "")).startswith("e2e_prof_") is False
                   for p in before)
        self.later(self.stage_create)

    def stage_create(self) -> None:
        log.info("== Stage: create profile from reference audio ==")
        profile_id = f"e2e_prof_{int(time.time())}"
        self.profile_id = profile_id
        self.panel.profile_id_ctrl.SetValue(profile_id)
        self.panel.profile_audio_ctrl.SetValue(REF_AUDIO)
        self.panel.profile_ref_text_ctrl.SetValue(REF_TEXT)
        self.panel._on_add_profile(None)  # noqa: SLF001
        self.poll_until(
            cond=lambda: self.profile_id in self._server_ids()
            or "Could not save" in self.panel.profiles_status.GetLabel(),
            deadline=time.monotonic() + 240.0,
            what="profile save to finish on the server",
            on_ok=lambda: log.info("Profiles status: %s; on server: %s",
                                   self.panel.profiles_status.GetLabel(),
                                   self.profile_id in self._server_ids()),
            on_success=self.check_created,
            interval=0.5,
        )

    def check_created(self) -> None:
        status = self.panel.profiles_status.GetLabel()
        assert "Could not" not in status, f"profile save failed: {status}"
        ids = self._server_ids()
        log.info("Server profile ids now: %s", ids)
        assert self.profile_id in ids, f"{self.profile_id!r} not in {ids}"
        self.later(self.stage_list_after)

    def stage_list_after(self) -> None:
        log.info("== Stage: refresh list shows the new profile ==")
        self.panel._on_refresh_profiles(None)  # noqa: SLF001
        self.poll_until(
            cond=lambda: self.panel.profile_list.GetItemCount() >= 1,
            deadline=time.monotonic() + 60.0,
            what="list refresh after create",
            on_ok=lambda: log.info("List rows: %d",
                                   self.panel.profile_list.GetItemCount()),
            on_success=self.check_listed,
        )

    def check_listed(self) -> None:
        rows = [
            (self.panel.profile_list.GetItemText(i),
             self.panel.profile_list.GetItemText(i, 1))
            for i in range(self.panel.profile_list.GetItemCount())
        ]
        log.info("Panel list rows: %s", rows)
        assert any(self.profile_id in r[0] for r in rows), rows
        self.later(self.stage_delete)

    def stage_delete(self) -> None:
        log.info("== Stage: delete the profile via the panel ==")
        for i in range(self.panel.profile_list.GetItemCount()):
            if self.panel.profile_list.GetItemText(i) == self.profile_id:
                self.panel.profile_list.Select(i, on=1)
                break
        else:
            _fail(f"profile {self.profile_id} not in the panel list")
            return
        if self.panel.profile_list.GetFirstSelected() < 0:
            _fail("Could not select the profile row in the list")
            return
        self.panel._on_delete_profile(None)  # noqa: SLF001 (stub answers YES)
        self.poll_until(
            cond=lambda: self.profile_id not in self._server_ids()
            or "Could not delete" in self.panel.profiles_status.GetLabel(),
            deadline=time.monotonic() + 90.0,
            what="profile deletion",
            on_ok=lambda: log.info("Profiles status: %s; gone from server: %s",
                                   self.panel.profiles_status.GetLabel(),
                                   self.profile_id not in self._server_ids()),
            on_success=self.check_deleted,
            interval=0.5,
        )

    def check_deleted(self) -> None:
        status = self.panel.profiles_status.GetLabel()
        assert "Could not delete" not in status, f"delete failed: {status}"
        ids = self._server_ids()
        log.info("Server profile ids now: %s", ids)
        assert self.profile_id not in ids, f"{self.profile_id!r} still in {ids}"
        self.later(self.stage_stop)

    def stage_stop(self) -> None:
        log.info("== Stage: stop server ==")
        self.panel._on_stop(None)  # noqa: SLF001
        self.poll_until(
            cond=lambda: "Stopped" in self.panel.status_label.GetLabel()
            or "Error stopping" in self.panel.status_label.GetLabel(),
            deadline=time.monotonic() + 60.0,
            what="server stop",
            on_ok=lambda: log.info("Status: %s",
                                   self.panel.status_label.GetLabel()),
            on_success=self.check_stopped,
        )

    def check_stopped(self) -> None:
        label = self.panel.status_label.GetLabel()
        assert "Stopped" in label, label
        assert not self.panel.add_profile_btn.IsEnabled()
        log.info("Profile controls disabled after stop (expected).")
        _finish()


def main() -> None:
    log.info("Reference audio: %s", REF_AUDIO)
    if not os.path.isfile(REF_AUDIO):
        print("Reference audio missing; run tools/e2e_omnivoice.py first",
              flush=True)
        sys.exit(1)

    global _app  # noqa: PLW0603
    _app = Harness()

    def _go():
        try:
            _app.stage_start()
        except Exception as exc:  # noqa: BLE001
            _fail(f"stage_start raised: {exc}")

    wx.CallAfter(_go)
    _app.app.MainLoop()

    # -- MainLoop is over: stop the server, report result -------------------
    try:
        mgr = get_server_manager(host="127.0.0.1", port=PORT)
        if mgr.is_running:
            mgr.stop()
            log.info("Server stopped during cleanup.")
    except Exception:  # noqa: BLE001
        pass

    if _failure:
        print(f"\nE2E RESULT: FAIL - {_failure[0]}", flush=True)
        sys.exit(1)
    print(f"\nE2E RESULT: PASS", flush=True)
    print(f"Profile manager on the Settings OmniVoice Server panel verified "
          f"end-to-end against a live server (port {PORT}).", flush=True)
    print(f"Modal dialogs auto-answered: {len(_modal_calls)}", flush=True)


if __name__ == "__main__":
    main()
