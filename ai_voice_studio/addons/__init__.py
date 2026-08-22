"""Addon system for AI Voice Studio.

Inspired by NVDA's addon architecture, this module provides a framework for
third-party extensions.  Addons can:

* Register new TTS engines (with their own worker processes).
* Register new voices or voice providers.
* Add new punctuation modes.
* Hook into the recording pipeline.
* Install their own pip dependencies through the managed Python environment.

Addon structure on disk::

    %APPDATA%/AIVoiceStudio/addons/
        my_addon/
            manifest.json          # required
            __init__.py            # entry point (optional)
            engine.py              # custom TTS engine (optional)
            voices/                # bundled voice data (optional)

manifest.json layout::

    {
        "name": "My TTS Addon",
        "version": "1.0.0",
        "description": "Adds a new TTS engine",
        "author": "Jane Doe",
        "url": "https://example.com",
        "min_app_version": "0.1.0",
        "engine_id": "my_tts",
        "engine_name": "My TTS Engine",
        "engine_entry_point": "engine:MyTtsEngine",
        "dependencies": ["requests>=2.31"],
        "voices": [
            {
                "id": "my_voice_1",
                "name": "My Voice 1",
                "language": "en",
                "data_dir": "voices/my_voice_1"
            }
        ],
        "provides": ["tts_engine", "voice"]
    }

Usage::

    from ai_voice_studio.addons import AddonManager
    manager = AddonManager()
    manager.discover()
    manager.load_all()
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from .. import paths

log = logging.getLogger(__name__)

_ADDONS_DIR = "addons"
_MANIFEST = "manifest.json"
_MIN_APP_VERSION = "0.1.0"


def _addon_dir() -> str:
    d = os.path.join(paths.user_data_dir(), _ADDONS_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def _parse_version(v: str) -> tuple:
    """Parse a version string like '1.0.0' into a comparable tuple."""
    try:
        return tuple(int(x) for x in v.split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


class AddonManifest:
    """Parsed manifest.json for an addon."""

    def __init__(self, path: str, data: Dict[str, Any]):
        self.path = path
        self.name: str = data.get("name", "Unnamed Addon")
        self.version: str = data.get("version", "0.0.0")
        self.description: str = data.get("description", "")
        self.author: str = data.get("author", "")
        self.url: str = data.get("url", "")
        self.min_app_version: str = data.get("min_app_version", "0.0.0")
        self.engine_id: str = data.get("engine_id", "")
        self.engine_name: str = data.get("engine_name", "")
        self.engine_entry_point: str = data.get("engine_entry_point", "")
        self.dependencies: List[str] = data.get("dependencies", [])
        self.voices: List[Dict[str, Any]] = data.get("voices", [])
        self.provides: List[str] = data.get("provides", [])
        self.enabled: bool = data.get("enabled", True)
        self._raw = data

    @property
    def addon_dir(self) -> str:
        return os.path.dirname(self.path)

    @property
    def is_compatible(self) -> bool:
        return _parse_version(self.min_app_version) <= _parse_version(_MIN_APP_VERSION)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self._raw)
        d["enabled"] = self.enabled
        return d

    def __repr__(self) -> str:
        return f"AddonManifest({self.name!r}, {self.version})"


class AddonManager:
    """Discovers, loads, and manages addons.

    Addons live in ``%APPDATA%/AIVoiceStudio/addons/<name>/``.  Each addon
    directory must contain a ``manifest.json``.
    """

    def __init__(self, addons_dir: str | None = None):
        self._dir = addons_dir or _addon_dir()
        self._addons: Dict[str, AddonManifest] = {}
        self._loaded_modules: Dict[str, Any] = {}
        self._engines: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # -- discovery ----------------------------------------------------------

    def discover(self) -> List[AddonManifest]:
        """Scan the addons directory and load manifests."""
        self._addons.clear()
        if not os.path.isdir(self._dir):
            return []
        for name in os.listdir(self._dir):
            manifest_path = os.path.join(self._dir, name, _MANIFEST)
            if not os.path.isfile(manifest_path):
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                manifest = AddonManifest(manifest_path, data)
                if not manifest.is_compatible:
                    log.warning(
                        "Addon %s requires app version %s (current: %s); skipping",
                        manifest.name, manifest.min_app_version, _MIN_APP_VERSION,
                    )
                    continue
                self._addons[name] = manifest
                log.info("Discovered addon: %s %s", manifest.name, manifest.version)
            except (json.JSONDecodeError, OSError) as exc:
                log.error("Failed to load addon manifest %s: %s", manifest_path, exc)
        return list(self._addons.values())

    # -- access -------------------------------------------------------------

    def list_addons(self) -> List[AddonManifest]:
        return list(self._addons.values())

    def get_addon(self, name: str) -> AddonManifest | None:
        return self._addons.get(name)

    def get_engines(self) -> Dict[str, Any]:
        """Return loaded engine classes keyed by engine_id."""
        return dict(self._engines)

    def get_all_voices(self) -> List[Dict[str, Any]]:
        """Aggregate voices from all enabled addons."""
        voices = []
        for manifest in self._addons.values():
            if not manifest.enabled:
                continue
            for voice in manifest.voices:
                entry = dict(voice)
                entry["addon"] = manifest.name
                # Resolve relative data_dir to absolute path
                data_dir = entry.get("data_dir", "")
                if data_dir and not os.path.isabs(data_dir):
                    entry["data_dir"] = os.path.join(manifest.addon_dir, data_dir)
                voices.append(entry)
        return voices

    # -- loading ------------------------------------------------------------

    def load_addon(self, name: str) -> bool:
        """Load a single addon's Python module and register its engine."""
        manifest = self._addons.get(name)
        if manifest is None or not manifest.enabled:
            return False
        if name in self._loaded_modules:
            return True

        addon_dir = manifest.addon_dir

        # Ensure addon dir is on sys.path for imports
        if addon_dir not in sys.path:
            sys.path.insert(0, addon_dir)

        # Import the addon's __init__.py if present
        init_path = os.path.join(addon_dir, "__init__.py")
        if os.path.isfile(init_path):
            try:
                module_name = f"_aivs_addon_{name}"
                spec = importlib.util.spec_from_file_location(module_name, init_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    self._loaded_modules[name] = module
                    log.info("Loaded addon module: %s", name)
            except Exception as exc:
                log.error("Failed to load addon %s: %s", name, exc, exc_info=True)
                return False

        # Register the TTS engine if the addon provides one
        if manifest.engine_id and manifest.engine_entry_point:
            self._register_engine(manifest, addon_dir)

        # Register voices
        if manifest.voices:
            self._register_voices(manifest)

        return True

    def _register_engine(self, manifest: AddonManifest, addon_dir: str) -> None:
        """Import and cache the addon's engine class."""
        if manifest.engine_id in self._engines:
            return
        try:
            module_name, class_name = manifest.engine_entry_point.split(":")
            # Import the module from the addon directory
            full_path = os.path.join(addon_dir, module_name + ".py")
            if not os.path.isfile(full_path):
                log.error("Engine module not found: %s", full_path)
                return
            spec = importlib.util.spec_from_file_location(
                f"_aivs_engine_{manifest.engine_id}", full_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"_aivs_engine_{manifest.engine_id}"] = module
                spec.loader.exec_module(module)
                engine_class = getattr(module, class_name, None)
                if engine_class:
                    self._engines[manifest.engine_id] = {
                        "class": engine_class,
                        "manifest": manifest,
                    }
                    log.info(
                        "Registered engine '%s' from addon '%s'",
                        manifest.engine_id, manifest.name,
                    )
                else:
                    log.error(
                        "Class %s not found in %s", class_name, full_path
                    )
        except Exception as exc:
            log.error(
                "Failed to register engine from addon %s: %s",
                manifest.name, exc, exc_info=True,
            )

    def _register_voices(self, manifest: AddonManifest) -> None:
        """Mark addon voices as available (actual registration happens in
        the TTS catalog at runtime)."""
        log.info(
            "Addon '%s' provides %d voice(s)", manifest.name, len(manifest.voices)
        )

    def load_all(self) -> int:
        """Load all enabled addons. Returns the number loaded."""
        count = 0
        for name, manifest in self._addons.items():
            if manifest.enabled and self.load_addon(name):
                count += 1
        return count

    # -- enable/disable/install/uninstall -----------------------------------

    def enable_addon(self, name: str) -> None:
        manifest = self._addons.get(name)
        if manifest:
            manifest.enabled = True
            self._save_manifest(manifest)

    def disable_addon(self, name: str) -> None:
        manifest = self._addons.get(name)
        if manifest:
            manifest.enabled = False
            self._save_manifest(manifest)

    def install_addon(self, addon_path: str) -> bool:
        """Install an addon from a ZIP file or directory.

        Parameters
        ----------
        addon_path:
            Path to a .zip file or a directory containing an addon
            with a manifest.json.

        Returns
        -------
        bool:
            True if the addon was installed successfully.
        """
        import shutil
        import zipfile

        if addon_path.endswith(".zip"):
            # Extract the zip to a temp dir, find manifest.json
            import tempfile
            tmp = tempfile.mkdtemp(prefix="aivs_addon_")
            try:
                with zipfile.ZipFile(addon_path) as zf:
                    zf.extractall(tmp)
                return self._install_from_dir(tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        elif os.path.isdir(addon_path):
            return self._install_from_dir(addon_path)
        return False

    def _install_from_dir(self, source_dir: str) -> bool:
        """Install an addon from a directory (copies it into the addons folder)."""
        import shutil

        manifest_path = os.path.join(source_dir, _MANIFEST)
        if not os.path.isfile(manifest_path):
            log.error("No manifest.json found in %s", source_dir)
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            name = data.get("name", "").strip()
            if not name:
                log.error("Addon manifest has no name")
                return False
            # Use a safe directory name
            from ..util import sanitize_filename
            safe_name = sanitize_filename(name.replace(" ", "_").lower(), 40)
            dest = os.path.join(self._dir, safe_name)
            if os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(source_dir, dest)
            # Reload manifests
            self.discover()
            log.info("Installed addon: %s to %s", name, dest)
            return True
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Failed to install addon from %s: %s", source_dir, exc)
            return False

    def uninstall_addon(self, name: str) -> bool:
        """Remove an addon completely."""
        import shutil
        manifest = self._addons.get(name)
        if manifest is None:
            return False
        addon_dir = manifest.addon_dir
        # Unload first
        self._loaded_modules.pop(name, None)
        self._engines.pop(manifest.engine_id, None)
        self._addons.pop(name, None)
        # Remove directory
        shutil.rmtree(addon_dir, ignore_errors=True)
        log.info("Uninstalled addon: %s", name)
        return True

    def _save_manifest(self, manifest: AddonManifest) -> None:
        """Write back the manifest (e.g. after enable/disable)."""
        try:
            with open(manifest.path, "w", encoding="utf-8") as fh:
                json.dump(manifest.to_dict(), fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            log.error("Could not save manifest for %s: %s", manifest.name, exc)


# ---------------------------------------------------------------------------
# Integration with the TTS engine
# ---------------------------------------------------------------------------

def register_addon_voices(store) -> None:
    """Register addon-provided voices with the ModelStore.

    This is called after addon discovery so addon voices appear in
    the Recording window alongside built-in voices.
    """
    manager = get_addon_manager()
    for voice in manager.get_all_voices():
        name = voice.get("name", "addon_voice")
        engine_id = voice.get("engine_id", "vits")
        data_dir = voice.get("data_dir", "")
        if data_dir and os.path.isdir(data_dir):
            store.add_custom_voice(
                name=name,
                tts_id=engine_id,
                directory=data_dir,
                kind="addon",
                extra={
                    "addon": voice.get("addon", ""),
                    "voice_id": voice.get("id", ""),
                    "language": voice.get("language", ""),
                },
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_default_manager: AddonManager | None = None
_manager_lock = threading.Lock()


def get_addon_manager() -> AddonManager:
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = AddonManager()
    return _default_manager
