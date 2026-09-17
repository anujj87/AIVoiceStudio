"""AI Voice Studio - application entry point.

Run with::

    python main.py

For packaged builds the PyInstaller spec in ``packaging/`` points here.
"""

from __future__ import annotations

import logging
import os
import sys
import threading


def _run_frozen_worker() -> bool:
    """Run a worker inside the frozen executable when requested."""
    if not getattr(sys, "frozen", False) or len(sys.argv) < 2:
        return False
    try:
        worker_index = sys.argv.index("--aivs-worker")
    except ValueError:
        return False
    if worker_index + 1 >= len(sys.argv):
        raise SystemExit("Missing AI Voice Studio worker name")
    worker_name = sys.argv[worker_index + 1]
    sys.argv[:] = [sys.argv[0]] + sys.argv[worker_index + 2:]
    raise SystemExit(f"Unknown AI Voice Studio worker: {worker_name}")


if _run_frozen_worker():
    raise SystemExit(0)

import wx

from ai_voice_studio import paths
from ai_voice_studio.gui.accept_dialog import AcceptanceDialog, record_acceptance, terms_current
from ai_voice_studio.gui.main_frame import MainFrame
from ai_voice_studio.gui.theme import apply_theme
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore


def _setup_logging(developer_mode: bool = False) -> None:
    """Configure logging based on developer mode setting.

    When developer_mode is True, enables heavy logging with:
    - DEBUG-level logging to multiple subsystem files
    - Full tracebacks for all caught exceptions
    - Thread crash logging
    - Crash log for unhandled exceptions

    When False, uses basic INFO-level logging to app.log only.
    """
    if developer_mode:
        from ai_voice_studio.heavy_logging import setup_heavy_logging
        setup_heavy_logging(enabled=True)
    else:
        try:
            logging.basicConfig(
                filename=paths.log_file(),
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        except OSError:
            logging.basicConfig(level=logging.INFO)


def _init_addons() -> None:
    """Discover and load addons if developer mode is enabled."""
    try:
        from ai_voice_studio.addons import get_addon_manager
        manager = get_addon_manager()
        manager.discover()
        count = manager.load_all()
        log = logging.getLogger("main")
        if count:
            log.info("Loaded %d addon(s)", count)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("main").warning("Addon initialization failed: %s", exc)


def _init_python_runtime() -> None:
    """Ensure the managed Python runtime is ready for addons."""
    try:
        from ai_voice_studio.python_runtime import get_runtime
        rt = get_runtime()
        # Don't create the venv eagerly — just ensure it's available
        log = logging.getLogger("main")
        log.info("Python runtime manager initialized (env: %s)", rt.env_dir)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("main").warning("Python runtime init failed: %s", exc)


def _warm_package_cache() -> None:
    """Probe the managed venv in the background.

    The Settings dialog and the Recording window ask whether the OmniVoice
    packages are installed; answering from a warm cache keeps both windows
    opening instantly (the probe itself starts a Python interpreter, which is
    why it must never run on the UI thread).
    """
    try:
        from ai_voice_studio.venv_packages import warm

        threading.Thread(
            target=warm, daemon=True, name="aivs-warm-packages"
        ).start()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("main").debug("Package warm-up failed: %s", exc)


def _require_terms_acceptance(settings: Settings) -> bool:
    """Show the first-launch terms dialog until the user accepts.

    Returns True when the application may continue to start. Every exit
    path of the dialog except the "I Agree" button (Disagree, Escape,
    closing the window) returns False so the application never launches
    without acceptance. Acceptance is remembered per TERMS_VERSION, so
    the dialog shows again only after an update or a fresh install.
    """
    if terms_current(settings):
        return True
    log = logging.getLogger("main")
    log.info("Showing first-launch terms acceptance dialog")
    app = wx.App(False)
    dialog = AcceptanceDialog(None)
    try:
        accepted = dialog.ShowModal() == wx.ID_OK and dialog.accepted
    finally:
        dialog.Destroy()
    if accepted:
        record_acceptance(settings)
        log.info("Terms accepted (version %s)", settings.get("terms_version"))
    else:
        log.info("Terms not accepted - application will not start")
    return accepted


def main() -> int:
    # First, do basic startup logging to determine developer mode
    settings = Settings()
    developer_mode = settings.get("developer_mode", False)

    # Setup logging (heavy or light based on developer mode)
    _setup_logging(developer_mode)

    log = logging.getLogger("main")
    log.info("AI Voice Studio starting (Python %s)", sys.version.split()[0])
    log.info("Developer mode: %s", developer_mode)

    # First launch (or first launch after an update): the application only
    # continues when the user accepts the terms of use.
    if not _require_terms_acceptance(settings):
        return 0

    # Initialize subsystems
    _init_python_runtime()
    _init_addons()
    _warm_package_cache()

    # If heavy logging was enabled, log system info
    if developer_mode:
        try:
            from ai_voice_studio.heavy_logging import log_system_info
            log_system_info()
        except Exception:  # noqa: BLE001
            pass

    app = wx.App(False)
    store = ModelStore()
    frame = MainFrame(None, settings, store)
    apply_theme(frame, settings.theme)
    frame.Show()
    result = app.MainLoop()
    log.info("AI Voice Studio exiting with code %s", result)
    return result or 0


if __name__ == "__main__":
    sys.exit(main())
