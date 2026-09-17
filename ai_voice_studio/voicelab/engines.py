"""Voice Lab engine registry (the voice-clone engines).

These engines are **not** sherpa-onnx ONNX artifacts: each one is a Python
package that runs in the app's managed virtualenv (see ``python_runtime``) and
is driven by ``voicelab.worker`` in a subprocess, exactly like OmniVoice.

Every one of them is a *voice cloning* engine and every one runs on a **CPU**;
when an NVIDIA GPU is present they can additionally run on the **GPU** (that
choice is offered both here and per project in the Recording window).

Because the voices of these engines are *provided by the installed package*
rather than downloaded as artifacts, they are defined here in Python instead of
``tts/models_catalog.json``:

* Bark ships 100+ speaker presets (ten speakers for each of twelve languages)
  which the catalog would have to spell out one by one, and
* F5-TTS built-in voices point at reference files *inside* the installed
  package (``f5_tts:infer/examples/basic/basic_ref_en.wav``).

Each voice entry has the same shape as the ones ``ModelStore.installed_voices``
produces, so the Download/Available/Recording panels need no special casing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Engine metadata
# ---------------------------------------------------------------------------

#: Bark v2 speaker presets: ten speakers for each language it speaks.
_BARK_LANGUAGES: Tuple[Tuple[str, str], ...] = (
    ("en", "English"),
    ("de", "German"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
)

#: Kyutai Pocket TTS pre-made voices: (voice id, language code).
_POCKET_VOICES: Tuple[Tuple[str, str], ...] = (
    ("alba", "en"),
    ("anna", "en"),
    ("azelma", "en"),
    ("bill_boerst", "en"),
    ("caro_davy", "en"),
    ("charles", "en"),
    ("cosette", "en"),
    ("eponine", "en"),
    ("eve", "en"),
    ("fantine", "en"),
    ("george", "en"),
    ("jane", "en"),
    ("javert", "en"),
    ("jean", "en"),
    ("marius", "en"),
    ("mary", "en"),
    ("michael", "en"),
    ("paul", "en"),
    ("peter_yearsley", "en"),
    ("stuart_bell", "en"),
    ("vera", "en"),
    ("estelle", "fr"),
    ("juergen", "de"),
    ("giovanni", "it"),
    ("lola", "es"),
    ("rafael", "pt"),
)

#: F5-TTS reference voices bundled with the ``f5-tts`` package.
_F5_REFS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "basic_ref_en",
        "en",
        "infer/examples/basic/basic_ref_en.wav",
        "Some call me nature, others call me mother nature.",
    ),
    (
        "basic_ref_zh",
        "zh",
        "infer/examples/basic/basic_ref_zh.wav",
        "对，这就是我，万人敬仰的太乙真人。",
    ),
)

#: ``voice`` id used on the Voice Clone panel when the user wants to clone a
#: voice from their own recording instead of a built-in voice.
CLONE_VOICE_ID = "__clone__"

#: The official PyTorch wheel index with CUDA builds.  The plain PyPI
#: ``torch`` wheel on Windows is CPU-only, so a GPU run needs the wheels from
#: this index (installed into the engine's own environment when an NVIDIA GPU
#: with a working driver is detected).
#:
#: ``cu128`` is deliberate and must stay in step with the Python the app runs
#: on: the older ``cu121`` index stops at Python 3.12 and carries no wheels for
#: Python 3.13, which made pip fail with "No matching distribution found for
#: torch" while installing an engine.  The OmniVoice installers use this same
#: constant, so all CUDA wheels come from one index.
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

#: Packages each engine takes from PyTorch's CUDA wheel index when an NVIDIA
#: GPU with a working driver is present.  Every Voice Lab engine runs on the
#: GPU (``devices: cpu+gpu``), so *every* engine gets the CUDA build of PyTorch:
#: without it the plain PyPI wheel is CPU-only on Windows and the "GPU" option
#: would be a label with nothing behind it.  Engines that need torchaudio
#: (F5-TTS) pull it from the same index, so the pair can never mix builds.
_CUDA_PACKAGES: Dict[str, Tuple[str, ...]] = {
    "pocket_tts": ("torch",),
    "bark": ("torch",),
    "f5tts": ("torch", "torchaudio"),
}

#: Modules an engine needs before it can speak - the worker answers "is this
#: engine available?" from the same list.  Kept here (not only in the worker)
#: so the application can *verify* an installation by importing them in the
#: engine's own environment; ``tests/test_voicelab.py`` fails when the two
#: copies drift apart.  The same list decides which environment an engine's
#: worker is started with (see ``voicelab.CloneWorker._runtime``), so a wrong
#: entry here is what makes an engine fail with a missing-module error.
_IMPORT_MODULES: Dict[str, Tuple[str, ...]] = {
    "pocket_tts": ("pocket_tts",),
    "bark": ("torch", "transformers"),
    "f5tts": ("f5_tts",),
}

_LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


def _languages(codes: Iterable[str], variant: str, variant_name: str) -> List[Dict[str, Any]]:
    """Catalog-shaped language list (one variant, voices added separately)."""
    return [
        {
            "code": code,
            "name": _LANGUAGE_NAMES.get(code, code),
            "variants": [{"id": variant, "name": variant_name, "voices": []}],
        }
        for code in codes
    ]


ENGINES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "pocket_tts",
        "name": "Pocket TTS (Kyutai)",
        "engine": "pocket_tts",
        "package": "pocket-tts",
        "packages": ("pocket-tts", "soundfile", "scipy"),
        "license": "Apache-2.0 (code); per-voice licenses",
        "homepage": "https://github.com/kyutai-labs/pocket-tts",
        "size_hint": "~450 MB (model) + PyTorch",
        "devices": "cpu+gpu",
        "voice_cloning": True,
        "note": (
            "A 100M-parameter voice-cloning TTS that runs faster than real "
            "time on a normal CPU. 26 pre-made voices in 6 languages; clone "
            "any voice from a 5-10 second sample."
        ),
        "languages": _languages(
            ("en", "fr", "de", "es", "it", "pt"),
            "pocket_v1", "Pocket TTS 100M",
        ),
    },
    {
        "id": "bark",
        "name": "Bark (Suno AI, transformer)",
        "engine": "bark",
        "package": "transformers",
        "packages": ("transformers", "torch", "soundfile", "scipy"),
        "license": "MIT",
        "homepage": "https://github.com/suno-ai/bark",
        "size_hint": "~2 GB (suno/bark) or ~700 MB (bark-small)",
        "devices": "cpu+gpu",
        "voice_cloning": True,
        "note": (
            "Suno's transformer text-prompted audio model. Over one hundred "
            "speaker presets across twelve languages, plus 'cloning' from a "
            "Bark speaker-embedding file (.npz)."
        ),
        # English also has Bark small: the same speakers on a much smaller
        # (and faster) English-only checkpoint.
        "languages": [
            {
                "code": "en",
                "name": "English",
                "variants": [
                    {"id": "bark_v2", "name": "Bark v2 (best quality, multilingual)",
                     "voices": []},
                    {"id": "bark_small", "name": "Bark small (English only, lighter)",
                     "voices": []},
                ],
            },
            *_languages(
                ("de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "zh"),
                "bark_v2", "Bark v2 (best quality, multilingual)",
            ),
        ],
    },
    {
        "id": "f5tts",
        "name": "F5-TTS (transformer)",
        "engine": "f5tts",
        "package": "f5-tts",
        "packages": ("f5-tts", "torch", "torchaudio", "soundfile"),
        "license": "MIT (code); CC-BY-NC-4.0 (weights)",
        "homepage": "https://github.com/SWivid/F5-TTS",
        "size_hint": "~1.4 GB",
        "devices": "cpu+gpu",
        "voice_cloning": True,
        "note": (
            "Flow-matching transformer TTS with zero-shot voice cloning from "
            "a 5-15 second reference clip plus its transcript. English and "
            "Chinese, with a bundled reference voice for each."
        ),
        "languages": _languages(("en", "zh"), "f5_v1_base", "F5-TTS v1 Base"),
    },
)

_ENGINES_BY_ID: Dict[str, Dict[str, Any]] = {e["id"]: e for e in ENGINES}


def engine_ids() -> Tuple[str, ...]:
    """Ids of every Voice Lab engine."""
    return tuple(e["id"] for e in ENGINES)


def is_engine(engine_id: Optional[str]) -> bool:
    """True when ``engine_id`` is one of the Voice Lab engines."""
    return bool(engine_id) and engine_id in _ENGINES_BY_ID


def engine(engine_id: str) -> Optional[Dict[str, Any]]:
    return _ENGINES_BY_ID.get(engine_id)


def packages(engine_id: str) -> Tuple[str, ...]:
    """pip packages this engine needs in the managed virtualenv."""
    info = _ENGINES_BY_ID.get(engine_id)
    if not info:
        return ()
    return tuple(info.get("packages") or (info.get("package"),))


def gpu_supported(engine_id: str) -> bool:
    """True when this engine can additionally run on an NVIDIA GPU.

    Every Voice Lab engine can (``devices`` is ``cpu+gpu`` for all of them);
    the CPU stays available either way.
    """
    info = _ENGINES_BY_ID.get(engine_id)
    return bool(info and "gpu" in str(info.get("devices", "")))


def import_modules(engine_id: str) -> Tuple[str, ...]:
    """Modules the engine needs before it can speak (worker's list)."""
    return _IMPORT_MODULES.get(engine_id, ())


def cuda_packages(engine_id: str) -> Tuple[str, ...]:
    """Packages to install from the CUDA wheel index on a GPU machine."""
    return _CUDA_PACKAGES.get(engine_id, ())


def cuda_index_url(engine_id: str) -> Optional[str]:
    """The wheel index for ``cuda_packages`` (``None`` when the engine has none)."""
    return PYTORCH_CUDA_INDEX if cuda_packages(engine_id) else None


def probe_package(engine_id: str) -> str:
    """The single package whose presence means "this engine is installed"."""
    info = _ENGINES_BY_ID.get(engine_id)
    return (info or {}).get("package", "")


def engine_name(engine_id: str) -> str:
    return (_ENGINES_BY_ID.get(engine_id) or {}).get("name", engine_id)


# ---------------------------------------------------------------------------
# Built-in (pre-made) voices
# ---------------------------------------------------------------------------
def _entry(
    engine_id: str,
    language: str,
    variant: str,
    voice: str,
    voice_name: str,
    **extra: Any,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "tts": engine_id,
        "tts_name": engine_name(engine_id),
        "language": language,
        "variant": variant,
        "voice": voice,
        "voice_name": voice_name,
        "sid": 0,
        "engine": engine_id,
        "dir": "",
        "builtin_voice": True,
        "requires_package": probe_package(engine_id),
    }
    entry.update(extra)
    return entry


def builtin_voices(engine_id: str) -> List[Dict[str, Any]]:
    """Every pre-made voice the engine ships with (no download needed)."""
    if not is_engine(engine_id):
        return []
    if engine_id == "pocket_tts":
        return [
            _entry(
                "pocket_tts", language, "pocket_v1", voice,
                f"{voice.replace('_', ' ').title()} ({_LANGUAGE_NAMES.get(language, language)})",
            )
            for voice, language in _POCKET_VOICES
        ]
    if engine_id == "bark":
        voices: List[Dict[str, Any]] = []
        for code, label in _BARK_LANGUAGES:
            for speaker in range(10):
                voices.append(
                    _entry(
                        "bark", code, "bark_v2", f"v2/{code}_speaker_{speaker}",
                        f"{label} speaker {speaker + 1} (of 10)",
                    )
                )
        # suno/bark-small is English-only, lighter and faster on a CPU.
        for speaker in range(10):
            voices.append(
                _entry(
                    "bark", "en", "bark_small", f"v2/en_speaker_{speaker}",
                    f"Bark small - English speaker {speaker + 1} (of 10)",
                    repo="suno/bark-small",
                )
            )
        return voices
    if engine_id == "f5tts":
        return [
            _entry(
                "f5tts", language, "f5_v1_base", voice,
                f"Bundled {_LANGUAGE_NAMES.get(language, language)} reference voice",
                ref_resource=f"f5_tts:{ref}",
                ref_text=text,
            )
            for (voice, language, ref, text) in _F5_REFS
        ]
    return []


def all_builtin_voices() -> List[Dict[str, Any]]:
    voices: List[Dict[str, Any]] = []
    for engine_id in engine_ids():
        voices.extend(builtin_voices(engine_id))
    return voices


def count_builtin_voices(engine_id: str) -> int:
    return len(builtin_voices(engine_id))
