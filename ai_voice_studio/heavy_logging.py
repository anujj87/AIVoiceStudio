"""Heavy logging system for AI Voice Studio.

Provides comprehensive error and diagnostic logging that captures every
error, warning, and diagnostic event in the application.  When enabled
(via Developer Mode in Settings), logs include:

* Full tracebacks for every caught exception (not just the last line).
* Thread names and timestamps with millisecond precision.
* Module/function names and line numbers.
* System info (Python version, platform, memory).
* Per-subsystem log files (gui.log, tts.log, engine.log).
* A crash log that is written on unexpected exit.

When disabled, only basic INFO-level logging to app.log is active.

Usage::

    from ai_voice_studio.heavy_logging import setup_heavy_logging
    setup_heavy_logging(enabled=True)

    # Later, anywhere in the code:
    from ai_voice_studio.heavy_logging import get_logger
    log = get_logger("my_module")
    log.error("Something broke", exc_info=True)
"""

from __future__ import annotations

import atexit
import datetime
import logging
import logging.handlers
import os
import platform
import sys
import threading
import traceback
from typing import Optional

from . import paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HEAVY_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)-12s] "
    "%(name)s:%(lineno)d %(message)s"
)
_LIGHT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_SUBSYSTEM_FILES = {
    "gui": "gui.log",
    "tts": "tts.log",
    "engine": "engine.log",
    "recording": "recording.log",
    "addons": "addons.log",
    "download": "download.log",
}

# Track enabled state globally
_heavy_enabled = False
_original_excepthook = sys.excepthook
_crash_log_path: str | None = None


# ---------------------------------------------------------------------------
# Crash handler
# ---------------------------------------------------------------------------

def _crash_handler(exc_type, exc_value, exc_tb):
    """Write a crash log on unhandled exception."""
    try:
        if _crash_log_path:
            with open(_crash_log_path, "w", encoding="utf-8") as fh:
                fh.write("=== AI Voice Studio CRASH LOG ===\n")
                fh.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                fh.write(f"Python: {sys.version}\n")
                fh.write(f"Platform: {platform.platform()}\n")
                fh.write(f"Executable: {sys.executable}\n")
                fh.write("\n--- Traceback ---\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=fh)
                fh.write("\n--- System Info ---\n")
                fh.write(f"PID: {os.getpid()}\n")
                fh.write(f"Thread: {threading.current_thread().name}\n")
                try:
                    import resource
                    fh.write(f"Memory RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KB\n")
                except (ImportError, AttributeError):
                    pass
    except Exception:
        pass
    # Call the original handler
    _original_excepthook(exc_type, exc_value, exc_tb)


# ---------------------------------------------------------------------------
# Subsystem file handler
# ---------------------------------------------------------------------------

class _SubsystemFilter(logging.Filter):
    """Routes log records to the appropriate subsystem file."""

    def __init__(self, subsystem: str):
        super().__init__()
        self.subsystem = subsystem

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name.lower()
        return name.startswith(self.subsystem)


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------

def setup_light_logging() -> None:
    """Configure basic (production) logging: INFO to app.log."""
    global _heavy_enabled
    _heavy_enabled = False

    log_file = paths.log_file()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format=_LIGHT_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )


def setup_heavy_logging(enabled: bool = True) -> None:
    """Configure comprehensive logging for development/debugging.

    Parameters
    ----------
    enabled:
        When True, enables verbose DEBUG-level logging to multiple
        subsystem files plus console output.  When False, falls back
        to basic logging.
    """
    global _heavy_enabled, _crash_log_path

    if not enabled:
        setup_light_logging()
        return

    _heavy_enabled = True

    # Clear existing handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    log_dir = paths.logs_dir()
    os.makedirs(log_dir, exist_ok=True)

    # -- Main app.log with full detail ------------------------------------
    app_log = os.path.join(log_dir, "app.log")
    main_handler = logging.handlers.RotatingFileHandler(
        app_log, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(logging.Formatter(_HEAVY_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(main_handler)

    # -- Console handler (WARNING+) for immediate visibility ---------------
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(_HEAVY_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console_handler)

    # -- Subsystem-specific files ------------------------------------------
    for subsystem, filename in _SUBSYSTEM_FILES.items():
        filepath = os.path.join(log_dir, filename)
        handler = logging.handlers.RotatingFileHandler(
            filepath, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_HEAVY_LOG_FORMAT, _DATE_FORMAT))
        handler.addFilter(_SubsystemFilter(subsystem))
        root.addHandler(handler)

    # -- Crash log path ----------------------------------------------------
    _crash_log_path = os.path.join(log_dir, "crash.log")

    # Install crash handler
    sys.excepthook = _crash_handler

    # -- Log system info at startup ----------------------------------------
    log = logging.getLogger("heavy_logging")
    log.info("=" * 70)
    log.info("AI Voice Studio - HEAVY LOGGING ENABLED")
    log.info("Python: %s", sys.version)
    log.info("Platform: %s", platform.platform())
    log.info("Executable: %s", sys.executable)
    log.info("PID: %d", os.getpid())
    log.info("Log directory: %s", log_dir)
    log.info("Subsystem log files: %s", list(_SUBSYSTEM_FILES.values()))
    log.info("Crash log: %s", _crash_log_path)
    log.info("=" * 70)

    # -- Install a global exception logger ---------------------------------
    _install_exception_logger()


def _install_exception_logger() -> None:
    """Install a thread-level exception logger that catches all unhandled
    exceptions in threads and logs them with full context."""

    _original_init = threading.Thread.__init__

    def _patched_thread_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        _original_run = self.run

        def _wrapped_run():
            try:
                _original_run()
            except Exception:
                log = logging.getLogger("heavy_logging.thread_crash")
                log.exception(
                    "Unhandled exception in thread '%s'", self.name
                )
                raise

        self.run = _wrapped_run

    threading.Thread.__init__ = _patched_thread_init


def is_heavy_logging_enabled() -> bool:
    """Check if heavy logging is currently active."""
    return _heavy_enabled


def get_log_path() -> str:
    """Return the path to the main log file."""
    return paths.log_file()


def get_log_dir() -> str:
    """Return the path to the logs directory."""
    return paths.logs_dir()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a subsystem (automatically routed to subsystem file)."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def log_system_info() -> None:
    """Write comprehensive system information to the log."""
    log = logging.getLogger("heavy_logging.system_info")
    log.info("--- System Information ---")
    log.info("Python: %s", sys.version)
    log.info("Platform: %s", platform.platform())
    log.info("Machine: %s", platform.machine())
    log.info("Processor: %s", platform.processor())
    log.info("Executable: %s", sys.executable)
    log.info("Prefix: %s", sys.prefix)
    log.info("PID: %d", os.getpid())
    try:
        import psutil
        mem = psutil.virtual_memory()
        log.info("RAM: %.1f GB total, %.1f GB available", mem.total / 1e9, mem.available / 1e9)
    except ImportError:
        pass
    try:
        log.info("GPU info:")
        import subprocess
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            creationflags=flags,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                log.info("  GPU: %s", line.strip())
        else:
            log.info("  No NVIDIA GPU detected")
    except Exception:
        log.debug("Could not query GPU info")


def log_exception_with_context(
    exc: Exception,
    context: str = "",
    module: str = "",
) -> None:
    """Log an exception with extra context information.

    Use this in except blocks for maximum diagnostic value::

        except Exception as exc:
            log_exception_with_context(exc, "while processing segment 5",
                                       module="jobs.synthesizer")
    """
    log = logging.getLogger(module or "heavy_logging")
    log.error(
        "ERROR %s: %s: %s\nFull context: %s",
        datetime.datetime.now().isoformat(),
        type(exc).__name__,
        exc,
        context,
        exc_info=True,
    )
