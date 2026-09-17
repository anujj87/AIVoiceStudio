"""Persistent application settings.

A small thread-safe JSON store. Defaults are defined in one place; ``reset``
restores them. Settings are read/written by the GUI, the TTS engine and the
recording worker.
"""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from typing import Any, Dict

from . import paths
from .constants import (
    COMPUTE_AUTO,
    DEFAULT_PITCH,
    DEFAULT_RATE,
    DEFAULT_VOLUME,
    FORMAT_WAV,
    MODE_PAGE_WITH_H1,
    PUNCTUATION_DEFAULT,
    THEME_SYSTEM,
)

log = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "theme": THEME_SYSTEM,
    # compute back-end: "auto" | "cpu" | "cuda" | "dml"
    "compute": COMPUTE_AUTO,
    # per-category compute choice for the Preview buttons
    # ({"punctuation": "cpu", "available_tts": "cuda", ...}); kept under its
    # own key because "compute" above is a plain string, not a table.
    "preview_compute": {},
    # developer mode: enables heavy logging, addon management, advanced options
    "developer_mode": False,
    # recording defaults used by the New Project wizard / Recording window
    "recording": {
        "punctuation": PUNCTUATION_DEFAULT,
        "rate": DEFAULT_RATE,
        "pitch": DEFAULT_PITCH,
        "volume": DEFAULT_VOLUME,
        "output_format": FORMAT_WAV,
        # per-TTS defaults: {tts_id: {"rate": x, "pitch": y, "volume": z}}
        "per_tts": {},
    },
    # audio file creation mode used by the New Project wizard
    "audio_mode": MODE_PAGE_WITH_H1,
    # default number of pages per audio file for "Page by page only"
    "audio_mode_pages_per_file": 1,
    # last used model selection (helps the wizard to preselect)
    "last_model": {
        "tts": None,
        "language": None,
        "variant": None,
        "voice": None,
    },
    # custom storage locations; empty string means "use the default folder"
    "paths": {
        "recordings_dir": "",
        "models_dir": "",
    },
    # cache of detected compute capabilities: {"cpu": true, "cuda": false, "dml": false}
    "detected": {},
    # DAISY 2.02 book settings
    "daisy": {
        "language": "en",
        "publisher": "",
        "include_text": True,
    },
    # list of recent projects: [{"name": str, "path": str, "opened": iso}]
    "recent_projects": [],
    # addon management
    "addons_enabled": {},
    # Voice Clone (Voice Lab) engines: last engine and the device they run on
    # ("cpu" always; "cuda" when an NVIDIA GPU is detected; "auto" = GPU first)
    "clone_engines": {
        "engine": "pocket_tts",
        "device": "cpu",
    },
    # OmniVoice Server settings
    "omnivoice_server": {
        "enabled": False,
        "auto_start": False,
        "host": "127.0.0.1",
        "port": 8881,
        "device": "cuda",
        "num_steps": 32,
        "max_concurrent": 2,
        "api_key": "",
        "cors_origins": "",
        "allow_network": False,
    },
}


class Settings:
    """JSON-backed settings store."""

    def __init__(self, path: str | None = None):
        self._path = path or paths.settings_file()
        self._lock = threading.RLock()
        self._data = deepcopy(_DEFAULTS)
        self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                merged = deepcopy(_DEFAULTS)
                merged.update(loaded)
                self._data = merged
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read settings %s: %s", self._path, exc)

    def save(self) -> None:
        with self._lock:
            try:
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, ensure_ascii=False)
            except OSError as exc:
                log.error("Could not save settings %s: %s", self._path, exc)

    # -- access ------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            value = self._data
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            parts = key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value

    def reset(self) -> None:
        with self._lock:
            self._data = deepcopy(_DEFAULTS)
        self.save()

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def set_many(self, items: Dict[str, Any]) -> None:
        for key, value in items.items():
            self.set(key, value)
        self.save()

    # -- convenience -------------------------------------------------------
    @property
    def theme(self) -> str:
        return self.get("theme", THEME_SYSTEM)

    @property
    def compute(self) -> str:
        return self.get("compute", COMPUTE_AUTO)

    @property
    def developer_mode(self) -> bool:
        return bool(self.get("developer_mode", False))

    def add_recent_project(self, name: str, path: str) -> None:
        recents = self.get("recent_projects", [])
        recents = [r for r in recents if r.get("path") != path]
        recents.insert(0, {"name": name, "path": path, "opened": __import__("datetime").datetime.now().isoformat()})
        self.set("recent_projects", recents[:10])
        self.save()

    def remove_recent_project(self, path: str) -> None:
        """Drop a project from the recent list (the folder may already be gone)."""
        recents = self.get("recent_projects", [])
        recents = [r for r in recents if r.get("path") != path]
        self.set("recent_projects", recents)
        self.save()
