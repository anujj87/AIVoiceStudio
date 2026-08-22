"""Access to the embedded model catalog (models_catalog.json).

The catalog is a static data file listing open-source, free, ONNX-based neural
TTS models and their official download URLs. Nothing is downloaded at import
time; the GUI uses these helpers to populate combo boxes and to build download
jobs.
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
    return load_catalog().get("tts", [])


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
    """TTS engines that support the Voice clone workflow (SPEC: voice clone tab)."""
    return [tts for tts in get_tts_list() if tts.get("voice_cloning")]


def language_display_name(tts: Dict[str, Any], lang_code: str) -> str:
    lang = find_language(tts, lang_code)
    return f"{lang['name']} ({lang_code})" if lang else lang_code


def voice_display_name(tts: Dict[str, Any], lang_code: str, variant_id: str, voice: Dict[str, Any]) -> str:
    return f"{voice.get('name', voice['id'])} ({voice['id']})"
