"""TTS synthesis engine (sherpa-onnx).

The engine is loaded lazily so the GUI starts even when the ONNX runtime is not
installed. Engines are cached per ``(voice key, provider)``; changing the
compute back-end or the voice re-creates the engine.

Verified against sherpa-onnx 1.13.5 (Aug 2026):
  * OfflineTtsVitsModelConfig(model, lexicon, tokens, data_dir, dict_dir, ...)
  * OfflineTtsMatchaModelConfig(acoustic_model, vocoder, lexicon, tokens,
    data_dir) -- acoustic model + separate neural vocoder
  * OfflineTtsKokoroModelConfig(model, voices, tokens, lexicon, data_dir,
    dict_dir, length_scale, lang) -- lexicon may be a comma-separated list
    (Kokoro multi-lang), data_dir points at a bundled espeak-ng-data
  * OfflineTtsModelConfig(vits=..., matcha=..., kokoro=..., num_threads, provider)
  * OfflineTtsConfig(model, rule_fsts, max_num_sentences)
  * tts.generate(text, sid=0, speed=1.0) -> GeneratedAudio(.samples, .sample_rate)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from typing import Dict

import numpy as np

from ..constants import (
    PUNCTUATION_ALL,
    PUNCTUATION_DEFAULT,
    PUNCTUATION_MATH,
    PUNCTUATION_NONE,
)
from .models import resolve_voice_files

log = logging.getLogger(__name__)

_SHERPA = None  # lazy import
_engine_lock = threading.RLock()
_engine_cache: Dict[str, "TtsEngine"] = {}


class EngineUnavailableError(Exception):
    """Raised when sherpa_onnx / onnxruntime is missing or broken."""


def _ensure_dll_search_path() -> None:
    """Make sherpa-onnx's bundled native DLLs findable on Windows.

    sherpa-onnx ships its own ONNX Runtime next to the extension module. GPU
    builds additionally bundle the CUDA / cuDNN runtime DLLs in the same
    folder: onnxruntime loads its CUDA provider plugin from there, and cuDNN
    resolves its companion DLLs through the library search path. Neither is
    on the default PATH, so register the folder before the module is imported
    (os.add_dll_directory for the Windows loader, PATH for cuDNN's own
    loader).
    """
    if sys.platform != "win32":
        return
    try:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.find_spec("sherpa_onnx")
        if not spec or not spec.submodule_search_locations:
            return
        lib = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        if not os.path.isdir(lib):
            return
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib)  # type: ignore[attr-defined]
            except OSError:
                pass
        path = os.environ.get("PATH", "")
        if lib not in path:
            os.environ["PATH"] = lib + os.pathsep + path
    except Exception:  # noqa: BLE001
        log.debug("Could not set up sherpa-onnx DLL search path", exc_info=True)


def _activate_optional_runtimes() -> None:
    """Preload an optional downloaded runtime (GPU) so it shadows the
    bundled CPU runtime before sherpa-onnx is imported."""
    try:
        from .. import runtime  # noqa: PLC0415

        runtime.activate()
    except Exception:  # noqa: BLE001
        log.debug("Optional runtime activation failed", exc_info=True)


def _import_sherpa():
    global _SHERPA
    if _SHERPA is None:
        try:
            _activate_optional_runtimes()
            _ensure_dll_search_path()
            import sherpa_onnx  # noqa: PLC0415

            _SHERPA = sherpa_onnx
        except ImportError as exc:
            raise EngineUnavailableError(
                "The neural TTS runtime (sherpa-onnx / ONNX Runtime) is not installed. "
                "Reinstall the application, or install 'sherpa-onnx' into the Python "
                "environment (pip install sherpa-onnx)."
            ) from exc
    return _SHERPA


class TtsEngine:
    """Wraps one sherpa-onnx OfflineTts instance."""

    def __init__(self, voice_entry: Dict, provider: str = "cpu", num_threads: int = 2):
        self.voice_entry = voice_entry
        self.provider = provider
        self.files = resolve_voice_files(voice_entry)
        required = ("model", "tokens")
        if self.files.get("engine") == "matcha":
            required = required + ("vocoder",)  # Matcha needs a neural vocoder
        missing = [k for k in required if not self.files.get(k)]
        if missing:
            raise EngineUnavailableError(
                "Model files are incomplete (missing " + ", ".join(missing) + "). "
                "Re-download the voice from Settings."
            )
        sherpa = _import_sherpa()
        cfg = self._build_config(sherpa, num_threads)
        try:
            self._tts = sherpa.OfflineTts(cfg)
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailableError(
                f"Could not load the TTS model: {exc}. Try the CPU back-end."
            ) from exc
        self.sample_rate = 22050
        try:
            self.sample_rate = self._tts.sample_rate
        except Exception:  # noqa: BLE001
            pass
    # -- config ------------------------------------------------------------
    def _build_config(self, sherpa, num_threads: int):
        engine = self.files.get("engine", "vits")
        if engine == "kokoro":
            voices_file = self.files.get("voices_file") or ""
            kokoro = sherpa.OfflineTtsKokoroModelConfig(
                model=self.files["model"],
                voices=voices_file,
                tokens=self.files["tokens"],
                lexicon=self.files.get("lexicon") or "",
                data_dir=self.files.get("data_dir") or "",
                dict_dir="",
                length_scale=1.0,
                lang="",
            )
            model_cfg = sherpa.OfflineTtsModelConfig(
                kokoro=kokoro, num_threads=num_threads, provider=self.provider
            )
        elif engine == "kitten":
            kitten = sherpa.OfflineTtsKittenModelConfig(
                model=self.files["model"],
                voices=self.files.get("voices_file") or "",
                tokens=self.files["tokens"],
                data_dir=self.files.get("data_dir") or "",
                length_scale=1.0,
            )
            model_cfg = sherpa.OfflineTtsModelConfig(
                kitten=kitten, num_threads=num_threads, provider=self.provider
            )
        elif engine == "matcha":
            matcha = sherpa.OfflineTtsMatchaModelConfig(
                acoustic_model=self.files["model"],
                vocoder=self.files["vocoder"],
                lexicon=self.files.get("lexicon") or "",
                tokens=self.files["tokens"],
                data_dir=self.files.get("data_dir") or "",
            )
            model_cfg = sherpa.OfflineTtsModelConfig(
                matcha=matcha, num_threads=num_threads, provider=self.provider
            )
        else:
            vits = sherpa.OfflineTtsVitsModelConfig(
                model=self.files["model"],
                lexicon=self.files.get("lexicon") or "",
                tokens=self.files["tokens"],
                data_dir=self.files.get("data_dir") or "",
                dict_dir="",
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )
            model_cfg = sherpa.OfflineTtsModelConfig(
                vits=vits, num_threads=num_threads, provider=self.provider
            )
        return sherpa.OfflineTtsConfig(model=model_cfg, rule_fsts="", max_num_sentences=1)

    # -- synthesis ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> "np.ndarray":
        """Synthesize text; returns int16 samples at ``self.sample_rate``."""
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        audio = self._tts.generate(text=text, sid=int(sid), speed=float(speed))
        samples = samples_to_int16(audio.samples)
        if pitch != 1.0:
            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            samples = _apply_volume(samples, volume)
        return samples


def samples_to_int16(raw) -> "np.ndarray":
    """Convert raw TTS samples to int16 PCM.

    sherpa-onnx returns the generated audio as ``std::vector<float>`` samples
    normalized to [-1, 1] (see cxx-api.h). Casting those floats straight to
    int16 truncates every value to 0, which produces a WAV with the right
    duration but total silence. Older int16 back-ends are passed through.
    """
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return arr.astype(np.int16)


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------
def get_engine(
    voice_entry: Dict,
    provider: str = "cpu",
    num_threads: int = 2,
):
    # Cloned voices (XTTS v2) run in a separate worker process through the
    # coqui-tts runtime; they do not use sherpa-onnx at all.
    if voice_entry.get("engine") == "xtts":
        from ..clone import XtTsCloneEngine  # noqa: PLC0415

        try:
            return XtTsCloneEngine(voice_entry)
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailableError(str(exc)) from exc
    # Qwen3-TTS cloned voices run in their own worker subprocess too (its
    # onnxruntime must never share a process with sherpa-onnx).
    if voice_entry.get("engine") == "qwen3":
        from ..qwen import Qwen3Engine  # noqa: PLC0415

        try:
            return Qwen3Engine(voice_entry)
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailableError(str(exc)) from exc
    # OmniVoice (k2-fsa/OmniVoice) runs in its own worker subprocess.
    if voice_entry.get("engine") == "omnivoice":
        from ..omnivoice import OmniVoiceEngine  # noqa: PLC0415

        try:
            return OmniVoiceEngine(voice_entry)
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailableError(str(exc)) from exc
    # sid is a per-call generate() argument, not part of the engine's
    # identity: multi-speaker models (e.g. Kokoro multi-lang with 53 speakers)
    # must share one loaded engine, otherwise each speaker would reload the
    # whole model. The model files (``dir``) fully determine the engine.
    key = (
        voice_entry.get("tts", ""),
        voice_entry.get("language", ""),
        voice_entry.get("variant", ""),
        voice_entry.get("voice", ""),
        voice_entry.get("dir", ""),
        provider,
    )
    key = str(key)
    with _engine_lock:
        engine = _engine_cache.get(key)
        if engine is None:
            engine = TtsEngine(voice_entry, provider=provider, num_threads=num_threads)
            _engine_cache[key] = engine
        return engine


def clear_engine_cache() -> None:
    with _engine_lock:
        _engine_cache.clear()


# ---------------------------------------------------------------------------
# Punctuation processing (SPEC: punctuation is chosen in the New Project
# wizard and in Settings -> Punctuation; the Recording window no longer
# shows it)
# ---------------------------------------------------------------------------
_SENTENCE_PUNCT = re.compile(r"[.,;:!?…“”\"'()\[\]{}<>«»—-]+")
_MATH_KEEP = re.compile(r"[+×÷=<>^%$€£¥&|~±√∫∑∏∞≈≠≤≥]+")
_EXTRA_WHITESPACE = re.compile(r"\s+")

# Spoken names for every punctuation mark, following the way screen readers
# voice them (elevenways.be/how-screen-readers-read-special-characters):
#   my name is "anujsharma".  ->  my name is quote anujsharma quote dot
#   print('hello')             ->  print left paren tic hello tic right paren
_PUNCT_WORDS = {
    ".": "dot",
    ",": "comma",
    ":": "colon",
    ";": "semicolon",
    "?": "question mark",
    "!": "exclamation mark",
    "\"": "quote",
    "“": "quote",
    "”": "quote",
    "«": "quote",
    "»": "quote",
    "'": "tic",
    "‘": "tic",
    "’": "tic",
    "`": "tic",
    "(": "left paren",
    ")": "right paren",
    "[": "left bracket",
    "]": "right bracket",
    "{": "left brace",
    "}": "right brace",
    "<": "less than",
    ">": "greater than",
    "-": "dash",
    "–": "dash",
    "—": "dash",
    "_": "underscore",
    "…": "dot dot dot",
    "+": "plus",
    "=": "equals",
    "/": "slash",
    "\\": "backslash",
    "@": "at",
    "#": "hash",
    "%": "percent",
    "&": "ampersand",
    "*": "asterisk",
    "$": "dollar",
    "€": "euro",
    "£": "pound",
    "¥": "yen",
    "^": "caret",
    "~": "tilde",
    "|": "vertical bar",
    "·": "dot",
    "•": "bullet",
}


def process_punctuation(text: str, mode: str) -> str:
    """Apply the selected punctuation mode (default/none/math/all)."""
    if mode == PUNCTUATION_DEFAULT:
        return text
    if mode == PUNCTUATION_ALL:
        # Replace every punctuation mark with its spoken word so TTS engines
        # that cannot pronounce punctuation still read it correctly.
        out = [
            f" {_PUNCT_WORDS[ch]} " if ch in _PUNCT_WORDS else ch
            for ch in text
        ]
        return _EXTRA_WHITESPACE.sub(" ", "".join(out)).strip()
    if mode == PUNCTUATION_NONE:
        cleaned = _SENTENCE_PUNCT.sub("", text)
    elif mode == PUNCTUATION_MATH:
        # Remove sentence punctuation but keep math symbols.
        cleaned = _SENTENCE_PUNCT.sub("", text)
        # Sentence punctuation is gone; re-insert spaces around math symbols so
        # the TTS does not glue words together.
        cleaned = _MATH_KEEP.sub(lambda m: " " + m.group(0) + " ", cleaned)
    else:  # unknown -> pass through
        return text
    return _EXTRA_WHITESPACE.sub(" ", cleaned).strip()


# ---------------------------------------------------------------------------
# Pitch / volume helpers (numpy, no extra deps)
# ---------------------------------------------------------------------------
def _resample(samples: "np.ndarray", factor: float) -> "np.ndarray":
    """Linearly resample a 1-D int16 array by ``factor`` (>1 speeds up)."""
    n_out = int(round(len(samples) * factor))
    if n_out == len(samples):
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _shift_pitch(samples: "np.ndarray", factor: float) -> "np.ndarray":
    """Approximate pitch shift without changing duration.

    Resample up/down by ``factor`` (changes pitch) then time-stretch back to the
    original length. Reasonable quality for speech; documented as approximate.
    """
    if factor <= 0.01:
        return samples
    try:
        shifted = _resample(samples, factor)
        return _resample(shifted, 1.0 / factor)
    except Exception:  # noqa: BLE001
        return samples


def _apply_volume(samples: "np.ndarray", volume: float) -> "np.ndarray":
    scaled = samples.astype(np.float64) * float(volume)
    np.clip(scaled, -32768, 32767, out=scaled)
    return scaled.astype(np.int16)


# ---------------------------------------------------------------------------
# Convenience for the Preview button / quick tests
# ---------------------------------------------------------------------------
def preview_synthesis(
    text: str,
    voice_entry: Dict,
    provider: str = "cpu",
    speed: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
    punctuation: str = PUNCTUATION_DEFAULT,
) -> "np.ndarray":
    engine = get_engine(voice_entry, provider=provider)
    text = process_punctuation(text, punctuation)
    return engine.synthesize(text, sid=voice_entry.get("sid", 0), speed=speed, pitch=pitch, volume=volume)
