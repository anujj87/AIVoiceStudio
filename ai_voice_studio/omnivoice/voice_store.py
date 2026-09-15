"""Universal OmniVoice voice library.

Users can create reusable voices in Settings -> OmniVoice engines:

* **Voice clone**  - copy a short reference recording (3-15 seconds) plus an
  optional transcript into the library.
* **Voice design** - store a natural-language voice description (gender, age,
  pitch, accent, style, dialect).

The library is *engine agnostic*.  Both OmniVoice engines (direct
``omnivoice`` via ``omnivoice-triton`` and ``omnivoice_server`` via the HTTP
server) share the same underlying model and accept the same reference audio /
design instructions, so a created voice works with either of them.  That is
what makes the list "universal": one created voice shows up under *every*
installed OmniVoice engine in the Recording window, the Available TTS panel
and the Punctuation panel.

Voices are persisted through ``ModelStore.custom_voices`` (``models.json``)
so they survive restarts and do not depend on the engine that was selected
when they were created.  Each voice's folder is keyed by a random id, never
by the user-visible name, so renaming a voice never touches the audio.

Everything in this module is GUI-free and import-safe (no PyTorch/CUDA), so
it can be used from the GUI thread, background preview threads and tests.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from typing import Any, Dict, Iterable, List, Tuple

from .. import paths
from ..tts import catalog
from ..util import sanitize_filename

log = logging.getLogger(__name__)

#: ModelStore ``kind`` values used by library voices.
KIND_CLONE = "omni_clone"
KIND_DESIGN = "omni_design"
OMNI_KINDS = (KIND_CLONE, KIND_DESIGN)

#: engine id -> (display name, pip package in the managed runtime).
ENGINE_INFO: Dict[str, Tuple[str, str]] = {
    "omnivoice": ("OmniVoice", "omnivoice-triton"),
    "omnivoice_server": ("OmniVoice Server", "omnivoice-server"),
}

#: Catalog voice-group variants each engine exposes.  A library voice is
#: registered under every variant of its engine so it is visible no matter
#: which variant the user has selected in a voice cascade.
_CATALOG_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "omnivoice": ("triton", "hybrid"),
    "omnivoice_server": ("server", "server_clone"),
}

#: The variant used when the panel previews a voice through an engine.
_DEFAULT_VARIANT: Dict[str, str] = {
    "omnivoice": "triton",
    "omnivoice_server": "server",
}

#: Languages of the two OmniVoice catalog entries.
_LANGUAGE = "auto"


def is_omni_custom(voice: Dict[str, Any]) -> bool:
    """True when ``voice`` is a ModelStore custom voice of this library."""
    return voice.get("kind") in OMNI_KINDS


def omni_custom_voices(store) -> List[Dict[str, Any]]:
    """All library voices stored in ``store`` (both clone and design)."""
    return [v for v in store.custom_voices() if is_omni_custom(v)]


def find_voice(store, name: str) -> Dict[str, Any] | None:
    return next((v for v in omni_custom_voices(store) if v["name"] == name), None)


def engine_ids_installed() -> List[str]:
    """Return the OmniVoice engine ids whose pip package is installed.

    Answers from the shared package cache (``venv_packages``), which probes
    the managed runtime in the background, so this never blocks the GUI while
    the settings dialog is being built.  ``[]`` means "nothing installed or
    not known yet".
    """
    from ..venv_packages import version  # noqa: PLC0415

    return [engine_id for engine_id, (_, pkg) in ENGINE_INFO.items() if version(pkg)]


# ---------------------------------------------------------------------------
# Create / rename / delete
# ---------------------------------------------------------------------------
def clean_name(name: str, fallback: str = "my_voice") -> str:
    """Sanitize a user-chosen voice name (keeps spaces/unicode)."""
    return sanitize_filename(name.strip(), 40) or fallback


def _ensure_name_free(store, name: str) -> None:
    if any(v["name"] == name for v in store.custom_voices()):
        raise ValueError(
            f"A voice named '{name}' already exists. Choose a different name "
            "or delete the existing voice first."
        )


def create_voice(
    store,
    *,
    name: str,
    mode: str,
    ref_audio: str = "",
    ref_text: str = "",
    instruct: str = "",
) -> Dict[str, Any]:
    """Create a universal OmniVoice voice and register it in ``store``.

    ``mode`` is ``"clone"`` (a reference recording is copied into the voice
    library) or ``"design"`` (a text description is stored).  The voice is
    engine agnostic: it is usable through the direct engine *and* the server
    engine.  Returns the stored custom-voice entry (its ``name`` may have
    been sanitized).
    """
    if mode not in ("clone", "design"):
        raise ValueError(f"Unknown voice mode '{mode}'")
    name = clean_name(name)
    _ensure_name_free(store, name)

    folder = os.path.join(
        paths.models_dir(), "custom", f"omni_{uuid.uuid4().hex[:12]}"
    )
    os.makedirs(folder, exist_ok=True)

    kind = KIND_CLONE if mode == "clone" else KIND_DESIGN
    extra: Dict[str, Any] = {
        "engine": "omnivoice",
        "custom_omni": True,
        "mode": mode,
        "instruct": (instruct or "").strip(),
        "ref_text": (ref_text or "").strip(),
        "sample": "",
        "reference": "",
    }
    if mode == "clone":
        ref = (ref_audio or "").strip()
        if not ref or not os.path.isfile(ref):
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError("Choose a reference audio sample file first.")
        base = os.path.basename(ref) or "sample"
        sample = os.path.join(folder, sanitize_filename(base, 60) or "sample")
        try:
            shutil.copy2(ref, sample)
        except OSError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError(f"Could not copy the sample: {exc}") from exc
        extra["sample"] = sample
        extra["reference"] = sample

    store.add_custom_voice(name, "omnivoice", folder, kind, extra=extra)
    entry = find_voice(store, name)
    if entry is None:  # pragma: no cover - defensive
        raise ValueError(f"Voice '{name}' could not be saved.")
    return entry


def rename_voice(store, old_name: str, new_name: str) -> Dict[str, Any]:
    """Rename a library voice; the folder keeps its id-based name."""
    new_name = clean_name(new_name)
    if new_name == old_name:
        return find_voice(store, old_name)  # nothing to do
    _ensure_name_free(store, new_name)
    if not store.rename_custom_voice(old_name, new_name):
        raise ValueError(f"Could not rename '{old_name}'.")
    entry = find_voice(store, new_name)
    if entry is None:  # pragma: no cover - defensive
        raise ValueError(f"Voice '{old_name}' is missing.")
    return entry


def delete_voice(store, name: str) -> bool:
    """Delete a library voice (and its sample folder)."""
    if not find_voice(store, name):
        return False
    return store.remove_custom_voice(name)


# ---------------------------------------------------------------------------
# Consumer voice entries (what the GUI cascades expect)
# ---------------------------------------------------------------------------
def _omni_for(voice: Dict[str, Any]) -> Dict[str, Any]:
    """The canonical per-voice ``omni`` dict the engines consume."""
    mode = voice.get("mode", "clone" if voice.get("kind") == KIND_CLONE else "design")
    sample = voice.get("sample") or voice.get("reference") or ""
    return {
        "mode": mode,
        "instruct": (voice.get("instruct") or "").strip(),
        "ref_audio": sample if mode == "clone" else "",
        "ref_text": (voice.get("ref_text") or "").strip() if mode == "clone" else "",
        "language": None,
    }


def entry_for_engine(
    voice: Dict[str, Any],
    engine_id: str,
    variant: str,
) -> Dict[str, Any]:
    """One consumer voice entry for ``voice`` under ``engine_id``/``variant``."""
    label, _ = ENGINE_INFO.get(engine_id, (engine_id, ""))
    tts = catalog.find_tts(engine_id)
    mode = voice.get("mode", "design")
    sample = voice.get("sample") or voice.get("reference") or ""
    omni = _omni_for(voice)
    return {
        "tts": engine_id,
        "tts_name": tts["name"] if tts else label,
        "language": _LANGUAGE,
        "variant": variant,
        "voice": voice["name"],
        "voice_name": voice["name"],
        "sid": 0,
        "engine": engine_id,
        "dir": voice.get("dir", ""),
        "custom": True,
        "custom_omni": True,
        "kind": voice.get("kind", ""),
        "omni": omni,
        "sample": sample,
        "reference": sample,
        "ref_audio": sample if mode == "clone" else "",
        "ref_text": omni.get("ref_text", ""),
        "instruct": omni.get("instruct", ""),
        "requires_gpu": True,
    }


def consumer_entries(
    store,
    engine_ids: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Every library voice as consumer entries under the installed engines.

    The same voice is registered under *every* engine (and every catalog
    variant of that engine) so it appears in the Recording window / Available
    TTS / Punctuation panels no matter which OmniVoice engine the user picks.
    """
    if engine_ids is None:
        engine_ids = engine_ids_installed()
    entries: List[Dict[str, Any]] = []
    for voice in omni_custom_voices(store):
        for engine_id in engine_ids:
            for variant in _CATALOG_VARIANTS.get(engine_id, ()):
                entries.append(entry_for_engine(voice, engine_id, variant))
    return entries


def preview_entry(
    voice: Dict[str, Any],
    engine_id: str,
) -> Dict[str, Any]:
    """A single entry for previewing ``voice`` through ``engine_id``."""
    variant = _DEFAULT_VARIANT.get(engine_id, "triton")
    return entry_for_engine(voice, engine_id, variant)


def describe(voice: Dict[str, Any]) -> str:
    """Short human summary of a library voice (for list tooltips/status)."""
    mode = voice.get("mode", "design")
    if mode == "clone":
        sample = voice.get("sample") or voice.get("reference") or ""
        return f"clone of '{os.path.basename(sample) or sample}'"
    return f"design: {voice.get('instruct') or '(empty description)'}"
