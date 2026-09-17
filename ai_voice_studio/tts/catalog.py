"""Access to the embedded model catalog (models_catalog.json).

The catalog is a static data file listing open-source, free, ONNX-based neural
TTS models and their official download URLs. Nothing is downloaded at import
time; the GUI uses these helpers to populate combo boxes and to build download
jobs.

The **Voice Lab** engines (Pocket TTS, Bark, F5-TTS) are appended to
the list from ``voicelab.engines``.  They are Python packages installed into
the managed virtualenv rather than downloaded ONNX artifacts, and their voices
are provided by the package itself, so keeping their metadata next to their
built-in voices avoids a second, hand-maintained copy here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "models_catalog.json")


def load_catalog() -> Dict[str, Any]:
    with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_tts_list() -> List[Dict[str, Any]]:
    """Every selectable TTS engine: the catalog plus the Voice Lab engines."""
    entries = list(load_catalog().get("tts", []))
    entries.extend(_voice_lab_entries())
    return entries


def _voice_lab_entries() -> List[Dict[str, Any]]:
    """Catalog-shaped metadata for the Voice Lab engines.

    Imported lazily: the GUI must still start when a third-party package used
    by one of those engines is missing from the environment.
    """
    try:
        from ..voicelab import engines as voicelab_engines  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return []
    entries: List[Dict[str, Any]] = []
    for entry in voicelab_engines.ENGINES:
        item = dict(entry)
        packages = list(entry.get("packages") or ())
        primary = entry.get("package", "")
        if primary and primary not in packages:
            packages.insert(0, primary)
        # ``requires_package`` is the single probe package the rest of the app
        # looks at (download list exclusion, install detection);
        # ``requires_packages`` is the full install list.
        item["requires_package"] = primary
        item["requires_packages"] = packages
        entries.append(item)
    return entries


def engine_env_id(tts: Dict[str, Any]) -> Optional[str]:
    """Environment key for this TTS engine's pip packages.

    Every pip-installed TTS engine gets its *own* Python environment
    (``%APPDATA%\\AIVoiceStudio\\tts_envs\\<engine>``), so the catalog id names
    the environment; the one pair that deliberately shares an environment (the
    two OmniVoice engines) is folded together by
    ``python_runtime.environment_id``.  Engines without ``requires_package``
    (the ONNX models and the built-in Windows voices) are not pip-installed at
    all and return ``None``, meaning "the shared addon environment".
    """
    if not tts.get("requires_package"):
        return None
    return tts["id"]


def requires_packages(tts: Dict[str, Any]) -> List[str]:
    """pip packages a TTS engine needs in the managed virtualenv.

    ``requires_package`` (singular) is the probe package used to decide
    whether the engine shows up at all; ``requires_packages`` lists everything
    to install for it (Bark, for example, needs transformers *and* torch).
    """
    packages = list(tts.get("requires_packages") or [])
    primary = tts.get("requires_package")
    if primary and primary not in packages:
        packages.insert(0, primary)
    return packages


def find_tts(tts_id: str) -> Optional[Dict[str, Any]]:
    for tts in get_tts_list():
        if tts["id"] == tts_id:
            return tts
    return None


def find_language(tts: Dict[str, Any], lang_code: str) -> Optional[Dict[str, Any]]:
    for lang in tts.get("languages", []):
        if lang["code"] == lang_code:
            return lang
    return None


def find_variant(
    tts: Dict[str, Any], lang_code: str, variant_id: str
) -> Optional[Dict[str, Any]]:
    lang = find_language(tts, lang_code)
    if not lang:
        # Built-in engines (the Windows system voices) have one variant whose
        # id is shared by every installed locale, so the language code never
        # matches a catalog language and the variant is looked up by id.
        if tts and tts.get("builtin"):
            for candidate in tts.get("languages", []):
                for variant in candidate.get("variants", []):
                    if variant["id"] == variant_id:
                        return variant
        return None
    for variant in lang.get("variants", []):
        if variant["id"] == variant_id:
            return variant
    return None


def find_voice(
    tts: Dict[str, Any], lang_code: str, variant_id: str, voice_id: str
) -> Optional[Dict[str, Any]]:
    variant = find_variant(tts, lang_code, variant_id)
    if not variant:
        return None
    for voice in variant.get("voices", []):
        if voice["id"] == voice_id:
            return voice
    return None


def cloning_capable_tts() -> List[Dict[str, Any]]:
    """TTS engines that support the voice-cloning workflow."""
    return [tts for tts in get_tts_list() if tts.get("voice_cloning")]


def language_display_name(tts: Dict[str, Any], lang_code: str) -> str:
    lang = find_language(tts, lang_code)
    return f"{lang['name']} ({lang_code})" if lang else lang_code


def voice_display_name(tts: Dict[str, Any], lang_code: str, variant_id: str, voice: Dict[str, Any]) -> str:
    return f"{voice.get('name', voice['id'])} ({voice['id']})"
