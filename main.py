"""AI Voice Studio - application entry point.

Run with::

    python main.py

For packaged builds the PyInstaller spec in ``packaging/`` points here.
"""

from __future__ import annotations

import logging
import sys

import wx

from ai_voice_studio import paths
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


def main() -> int:
    # First, do basic startup logging to determine developer mode
    settings = Settings()
    developer_mode = settings.get("developer_mode", False)

    # Setup logging (heavy or light based on developer mode)
    _setup_logging(developer_mode)

    log = logging.getLogger("main")
    log.info("AI Voice Studio starting (Python %s)", sys.version.split()[0])
    log.info("Developer mode: %s", developer_mode)

    # Initialize subsystems
    _init_python_runtime()
    _init_addons()

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
