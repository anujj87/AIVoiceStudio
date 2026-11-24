"""Persistent application settings.

A small thread-safe JSON store. Defaults are defined in one place; ``reset``
restores them. Settings are read/written by the GUI, the TTS engine and the
recording worker.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from copy import deepcopy
from typing import Any

from . import paths, secret_store
from .constants import (
    AD_CHUNK_SECONDS,
    AD_DEFAULT_FPS,
    AD_DEFAULT_OUTPUT,
    AD_DEFAULT_PLACEMENT,
    AD_DEFAULT_TRACK_PRIORITY,
    AD_MAX_DESCRIPTION_SECONDS,
    AD_MIN_GAP_SECONDS,
    COMPUTE_AUTO,
    DEFAULT_PITCH,
    DEFAULT_PROJECT_BASE,
    DEFAULT_RATE,
    DEFAULT_VOLUME,
    FORMAT_WAV,
    MODE_PAGE_WITH_H1,
    PUNCTUATION_DEFAULT,
    STT_DEFAULT_MINUTES,
    STT_DEFAULT_OUTPUT,
    STT_DEFAULT_SOURCE,
    STT_DEFAULT_SPLIT,
    THEME_SYSTEM,
)

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "theme": THEME_SYSTEM,
    # user-created themes: [{id, name, colors: {bg, fg, control_bg,
    # control_fg, accent, heading}, pattern}].  The twenty predefined dark
    # themes live in ai_voice_studio.gui.theme and are not stored here.
    "custom_themes": [],
    # welcome-screen wallpaper per theme id:
    # {theme_id: {enabled, image, pattern}}.  Works for the built-in, every
    # predefined and any custom theme.
    "wallpapers": {},
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
    # which project base the New Project wizard starts on ("tts" | "stt")
    "project_base": DEFAULT_PROJECT_BASE,
    # last used Speech-to-Text choices (preselected by the transcribe dialog)
    "stt": {
        "model": "whisper_large_v3",
        "variant": "",
        "language": "auto",
        "accuracy": "balanced",
        "translate_to": "",
        "compute": "auto",
        "output_format": STT_DEFAULT_OUTPUT,
        "include_timestamp": False,
        "source_type": STT_DEFAULT_SOURCE,
        "split_mode": STT_DEFAULT_SPLIT,
        "minutes": STT_DEFAULT_MINUTES,
    },
    # last used Audio description choices (preselected by the describe dialog)
    "describe": {
        # The smallest CPU-capable describer, so a first run works on any
        # machine; the Settings category and the wizard change it (see
        # describe.catalog.default_model_id).
        "model": "smolvlm2",
        "variant": "2.2b",
        "compute": "auto",
        "chunk_seconds": AD_CHUNK_SECONDS,
        "fps": AD_DEFAULT_FPS,
        "output_format": AD_DEFAULT_OUTPUT,
        "gap_fitting": False,
        # Where a description is spoken: into the pauses of the soundtrack
        # (standard) or while the film is stopped (extended, see constants).
        "placement": AD_DEFAULT_PLACEMENT,
        # Which of the two tracks gives way when the description is spoken over
        # the film's own soundtrack (see constants.AD_TRACK_PRIORITY_CHOICES).
        # Only standard placement has the two heard together at all.
        "track_priority": AD_DEFAULT_TRACK_PRIORITY,
        "min_gap": AD_MIN_GAP_SECONDS,
        "max_description": AD_MAX_DESCRIPTION_SECONDS,
        # Speak the descriptions too, with the voice chosen last (the ids below,
        # resolved against the installed voices when the dialog opens).
        "narration": False,
        "narration_voice": {
            "tts": None,
            "language": None,
            "variant": None,
            "voice": None,
        },
    },
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
    # user keyboard-gesture overrides: {command id: ["kb:control+shift+n"]}
    # (a command missing here uses the defaults in ai_voice_studio.gestures)
    "gestures": {
        "bindings": {},
    },
    # list of recent projects: [{"name": str, "path": str, "opened": iso}]
    "recent_projects": [],
    # update checks against the project's GitHub releases
    "updates": {
        # look for a newer release once a day when the application starts
        "auto_check": True,
        # ISO timestamp of the last successful check (throttles the auto check)
        "last_check": "",
        # a version the user asked never to be reminded about again
        "skip_version": "",
    },
    # addon management
    "addons_enabled": {},
    # Hugging Face access for *gated* model repositories: the user's own read
    # token - needed by nothing else, and needed by Gemma 3 - and the account it
    # was last verified against.  See ai_voice_studio.hf_access.
    "hf": {
        "token": "",
        "account": "",
    },
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
        """Read the file; credentials are decrypted on the way in.

        A settings file from an older version stores the Hugging Face token
        and the OmniVoice API key as plain text.  It is still read (nobody's
        token disappears because of an upgrade) and written back encrypted at
        once, so the plain text is not left on disk for another process to
        read (see :mod:`ai_voice_studio.secret_store`).
        """
        needs_migration = False
        try:
            with open(self._path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                needs_migration = secret_store.has_plaintext_secrets(loaded)
                merged = deepcopy(_DEFAULTS)
                merged.update(secret_store.unprotect_settings(loaded))
                self._data = merged
        except FileNotFoundError:
            log.debug("Ignoring a failure in Settings.load", exc_info=True)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read settings %s: %s", self._path, exc)
        if needs_migration and secret_store.available():
            log.info("Encrypting the credentials in %s", self._path)
            self.save()

    def save(self) -> None:
        with self._lock:
            try:
                # Only the values are encrypted: the structure, the key names
                # and every other setting stay readable in the file.
                payload = secret_store.protect_settings(self._data)
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False)
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

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def set_many(self, items: dict[str, Any]) -> None:
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
        recents.insert(0, {
            "name": name,
            "path": path,
            "opened": datetime.datetime.now().isoformat(),
        })
        self.set("recent_projects", recents[:10])
        self.save()

    def remove_recent_project(self, path: str) -> None:
        """Drop a project from the recent list (the folder may already be gone)."""
        recents = self.get("recent_projects", [])
        recents = [r for r in recents if r.get("path") != path]
        self.set("recent_projects", recents)
        self.save()
