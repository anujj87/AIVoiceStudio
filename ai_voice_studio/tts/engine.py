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

import json
import logging
import os
import queue
import re
import sys
import threading

import numpy as np

from ..constants import (
    PUNCTUATION_ALL,
    PUNCTUATION_DEFAULT,
    PUNCTUATION_MATH,
    PUNCTUATION_NONE,
)
from . import markup
from .models import resolve_voice_files

log = logging.getLogger(__name__)

#: Supertonic's own quality knob: 5 (low) to 12 (high), 8 by default.  It is
#: the number of flow-matching steps the model runs, so it trades a little
#: speed for a little naturalness.
SUPERTONIC_NUM_STEPS = 8

_SHERPA = None  # lazy import
_engine_lock = threading.RLock()
_engine_cache: dict[str, TtsEngine] = {}


class EngineUnavailableError(Exception):
    """Raised when sherpa_onnx / onnxruntime is missing or broken."""


def default_num_threads() -> int:
    """Threads a CPU back-end uses: 80-95% of the machine's logical CPUs.

    Resolved through :mod:`ai_voice_studio.compute` (lazily, so this module
    stays importable on a machine where the runtime is missing).
    """
    try:
        from .. import compute

        return compute.cpu_threads()
    except Exception:
        return 2


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
        import importlib.util

        spec = importlib.util.find_spec("sherpa_onnx")
        if not spec or not spec.submodule_search_locations:
            return
        lib = os.path.join(next(iter(spec.submodule_search_locations)), "lib")
        if not os.path.isdir(lib):
            return
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib)  # type: ignore[attr-defined]
            except OSError:
                log.debug("Ignoring a failure in _ensure_dll_search_path", exc_info=True)
        path = os.environ.get("PATH", "")
        if lib not in path:
            os.environ["PATH"] = lib + os.pathsep + path
    except Exception:
        log.debug("Could not set up sherpa-onnx DLL search path", exc_info=True)


def _activate_optional_runtimes() -> None:
    """Preload an optional downloaded runtime (GPU) so it shadows the
    bundled CPU runtime before sherpa-onnx is imported."""
    try:
        from .. import runtime

        runtime.activate()
    except Exception:
        log.debug("Optional runtime activation failed", exc_info=True)


def _import_sherpa():
    global _SHERPA
    if _SHERPA is None:
        try:
            _activate_optional_runtimes()
            _ensure_dll_search_path()
            import sherpa_onnx

            _SHERPA = sherpa_onnx
        except ImportError as exc:
            raise EngineUnavailableError(
                "The neural TTS runtime (sherpa-onnx / ONNX Runtime) is not installed. "
                "Reinstall the application, or install 'sherpa-onnx' into the Python "
                "environment (pip install sherpa-onnx)."
            ) from exc
    return _SHERPA


def repair_voice_files(voice_entry: dict) -> bool:
    """Put a voice back together before calling its files incomplete.

    A voice can fail here with its files present and simply one folder too
    high: builds up to 2027.1.0 extracted every archive of a variant into the
    variant folder itself, where the last voice's files overwrote the earlier
    ones, so ``<variant>/<voice>`` - the folder the engine reads - stayed
    empty.  Those downloads keep their archives, and the voice is rebuilt from
    its own archive once, without asking for a ~100 MB download that would land
    in exactly the same place.  Returns ``True`` when a repair was made.
    """
    try:
        from . import downloader

        return bool(downloader.repair_voice_layout(voice_entry))
    except Exception:
        log.debug("Could not repair the files of %s", voice_entry.get("dir"), exc_info=True)
        return False


class TtsEngine:
    """Wraps one sherpa-onnx OfflineTts instance."""

    def __init__(self, voice_entry: dict, provider: str = "cpu", num_threads: int = 2):
        self.voice_entry = voice_entry
        self.provider = provider
        self.files = resolve_voice_files(voice_entry)
        engine = self.files.get("engine", "vits")
        required = ("model", "tokens")
        if engine == "matcha":
            required = (*required, "vocoder")  # Matcha needs a neural vocoder
        elif engine == "supertonic":
            # Supertonic 3 has no model.onnx/tokens.txt: its four ONNX parts
            # plus the two tokenizer tables and voice.bin are the model.
            required = (
                "duration_predictor", "text_encoder", "vector_estimator",
                "vocoder", "tts_json", "unicode_indexer", "voice_style",
            )
        missing = [k for k in required if not self.files.get(k)]
        if missing and repair_voice_files(voice_entry):
            # The files were there, in the wrong folder: read them again.
            self.files = resolve_voice_files(voice_entry)
            missing = [k for k in required if not self.files.get(k)]
        if missing:
            raise EngineUnavailableError(
                "Model files are incomplete (missing " + ", ".join(missing) + "). "
                "Re-download the voice from Settings."
            )
        sherpa = _import_sherpa()
        self._sherpa = sherpa
        # The language Supertonic speaks: the generation call takes it as an
        # extra argument, and it is the catalog language of the voice entry.
        self._lang = ""
        if engine == "supertonic":
            self._lang = str(voice_entry.get("language") or "en")
        cfg = self._build_config(sherpa, num_threads)
        try:
            self._tts = sherpa.OfflineTts(cfg)
        except Exception as exc:
            raise EngineUnavailableError(
                f"Could not load the TTS model: {exc}. Try the CPU back-end."
            ) from exc
        self.sample_rate = 22050
        try:
            self.sample_rate = self._tts.sample_rate
        except Exception:
            log.debug("Ignoring a failure in TtsEngine.__init__", exc_info=True)
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
        elif engine == "supertonic":
            # Supertonic 3 (Supertone): one multilingual model, 31 languages
            # and 10 voice styles.  The provider is passed exactly like the
            # other ONNX engines, so CPU and GPU (CUDA) both work here.
            factory = getattr(sherpa, "OfflineTtsSupertonicModelConfig", None)
            if factory is None:
                raise EngineUnavailableError(
                    "This build of the TTS runtime (sherpa-onnx "
                    f"{getattr(sherpa, '__version__', 'older')}) cannot load "
                    "Supertonic 3. Update sherpa-onnx to 1.13.6 or newer "
                    "(pip install -U sherpa-onnx) and try again."
                )
            supertonic = factory(
                duration_predictor=self.files["duration_predictor"],
                text_encoder=self.files["text_encoder"],
                vector_estimator=self.files["vector_estimator"],
                vocoder=self.files["vocoder"],
                tts_json=self.files["tts_json"],
                unicode_indexer=self.files["unicode_indexer"],
                voice_style=self.files["voice_style"],
            )
            model_cfg = sherpa.OfflineTtsModelConfig(
                supertonic=supertonic, num_threads=num_threads, provider=self.provider
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
        # Sentence batching (``max_num_sentences``) is deliberately left at 1.
        # The Kokoro implementation ignores every other value ("max_num_sentences
        # (8) != 1 is ignored for Kokoro TTS models"), and the engines that do
        # honour it gained nothing when measured (Kitten: 1.26x realtime at 1,
        # 1.20-1.33x realtime at 2-8) because the per-sentence CPU front-end
        # (normalisation, espeak-ng phonemisation, tokenisation) - not the card -
        # is the bottleneck.  What fills the GPU is several segments in flight:
        # see :class:`TtsEnginePool`.
        return sherpa.OfflineTtsConfig(model=model_cfg, rule_fsts="", max_num_sentences=1)

    # -- synthesis ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        """Synthesize text; returns int16 samples at ``self.sample_rate``."""
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        if self._lang:
            # A multilingual model is told which language to speak; the same
            # call also carries the speaker and the speed.
            gen = self._sherpa.GenerationConfig()
            gen.sid = int(sid)
            gen.speed = float(speed)
            gen.num_steps = SUPERTONIC_NUM_STEPS
            gen.extra["lang"] = self._lang
            audio = self._tts.generate(text, gen)
        else:
            audio = self._tts.generate(text=text, sid=int(sid), speed=float(speed))
        samples = samples_to_int16(audio.samples)
        if pitch != 1.0:
            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            samples = _apply_volume(samples, volume)
        return samples


def samples_to_int16(raw) -> np.ndarray:
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
    voice_entry: dict,
    provider: str = "cpu",
    num_threads: int = 0,
):
    if not num_threads:
        # CPU inference: use the 80-95% band of the machine (the GPU back-end
        # ignores this - it is only there for the CPU-side pre/post work).
        num_threads = default_num_threads()
    # Windows system voices (SAPI5 / Windows Core) are synthesised by the
    # operating system through PowerShell, not by sherpa-onnx.
    if voice_entry.get("engine") in ("sapi5", "windows_core"):
        from ..tts.windows_tts import WindowsVoiceEngine

        try:
            return WindowsVoiceEngine(voice_entry)
        except Exception as exc:
            raise EngineUnavailableError(str(exc)) from exc
    # OmniVoice runs in a separate worker process through the
    # omnivoice-triton runtime; it requires an NVIDIA GPU.
    if voice_entry.get("engine") == "omnivoice":
        from ..omnivoice import OmniVoiceEngine

        try:
            return OmniVoiceEngine(voice_entry)
        except Exception as exc:
            raise EngineUnavailableError(str(exc)) from exc
    # OmniVoice Server runs as an HTTP server (OpenAI-compatible API).
    if voice_entry.get("engine") == "omnivoice_server":
        from ..omnivoice_server import OmniVoiceServerEngine

        try:
            return OmniVoiceServerEngine(voice_entry)
        except Exception as exc:
            raise EngineUnavailableError(str(exc)) from exc
    # Voice Lab engines (Pocket TTS, F5-TTS) run through the
    # managed virtualenv. They work on the CPU and use the GPU when the user
    # asked for it; ``provider`` carries that choice here.
    try:
        from ..voicelab import engines as voicelab_engines

        is_voice_lab = voicelab_engines.is_engine(voice_entry.get("engine"))
    except Exception:
        is_voice_lab = False
    if is_voice_lab:
        from ..voicelab import VoicelabEngine

        try:
            return VoicelabEngine(
                voice_entry,
                device=voice_entry.get("device") or provider,
            )
        except Exception as exc:
            raise EngineUnavailableError(str(exc)) from exc
    # sid is a per-call generate() argument, not part of the engine's
    # identity: multi-speaker models (e.g. Kokoro multi-lang with 53 speakers)
    # must share one loaded engine, otherwise each speaker would reload the
    # whole model. The model files (``dir``) fully determine the engine.
    if voice_entry.get("engine") in ("omnivoice", "omnivoice_server"):
        # OmniVoice engines carry their voice identity *inside* the voice
        # entry: cloning reference audio, design instructions and the
        # per-project ``omni`` generation settings.  Two projects that use
        # the same catalog voice (e.g. "auto") but different clone samples
        # or instructions must NOT share one cached engine, or the second
        # project would speak with the first project's voice.
        key = str((
            voice_entry.get("tts", ""),
            voice_entry.get("variant", ""),
            voice_entry.get("voice", ""),
            voice_entry.get("instruct", ""),
            voice_entry.get("ref_audio", ""),
            voice_entry.get("ref_text", ""),
            voice_entry.get("language", ""),
            json.dumps(voice_entry.get("omni") or {}, sort_keys=True),
            provider,
        ))
    else:
        key = str((
            voice_entry.get("tts", ""),
            voice_entry.get("language", ""),
            voice_entry.get("variant", ""),
            voice_entry.get("voice", ""),
            voice_entry.get("dir", ""),
            provider,
        ))
    with _engine_lock:
        engine = _engine_cache.get(key)
        if engine is None:
            engine = TtsEngine(voice_entry, provider=provider, num_threads=num_threads)
            _engine_cache[key] = engine
        return engine


def is_local_sherpa_engine(voice_entry: dict) -> bool:
    """True when a local sherpa-onnx model does the speaking.

    Windows system voices, OmniVoice (its own worker process / HTTP server) and
    the Voice Lab engines bring their own runtime, where several engines at once
    would mean several worker processes.  How many of those a Voice Lab engine
    *may* run is decided by its own plan (see :func:`planned_workers`).
    """
    engine = voice_entry.get("engine", "vits")
    if engine in ("sapi5", "windows_core", "omnivoice", "omnivoice_server"):
        return False
    try:
        from ..voicelab import engines as voicelab_engines

        return not voicelab_engines.is_engine(engine)
    except Exception:
        return True


def planned_workers(
    voice_entry: dict,
    compute_choice: str = "cpu",
    total: int | None = None,
) -> int:
    """How many segments a run of this voice may synthesize at the same time.

    Where the number comes from depends on who speaks:

    * a **local sherpa-onnx model** - :func:`compute.synthesis_workers`: a GPU
      run keeps two to four segments in flight so the card is not left waiting
      for the CPU to phonemise every sentence, a CPU run keeps one;
    * a **Voice Lab engine that can fill a GPU** (Chatterbox, Qwen3-TTS,
      VibeVoice) - what
      its own plan affords on the card *now* (``voicelab.parallel_plan``):
      several worker processes for an engine that speaks one text per call, or
      several texts in one batched call for an engine whose own call takes a
      list of them;
    * anything else with a runtime of its own (OmniVoice, the Windows voices,
      a Voice Lab engine on a CPU) - one, because each of those is a worker
      process of its own.

    The caller still *saves and reports* the segments in index order; only how
    many are in flight changes.
    """
    if total is not None and int(total) <= 0:
        return 1
    if not is_local_sherpa_engine(voice_entry):
        engine_id = str(voice_entry.get("engine") or "")
        try:
            from ..voicelab import engines as voicelab_engines

            if voicelab_engines.is_engine(engine_id):
                from ..voicelab import parallel_plan

                # Deliberately without the variant: this is the same plan the
                # engine's shared worker pool settles on (``voicelab.pool_for``
                # is keyed by engine and device, not by voice), and the two
                # must not disagree about how many segments are in flight.
                slots, _mode = parallel_plan(
                    engine_id, voice_entry.get("device") or compute_choice
                )
                return max(1, min(slots, int(total)) if total else slots)
        except Exception:
            return 1
        return 1
    from .. import compute

    return compute.synthesis_workers(compute_choice, total=total)


class TtsEnginePool:
    """A few interchangeable engines, so several segments can be in flight.

    sherpa-onnx prepares and runs **one sentence** per ``generate`` call, and
    the Kokoro implementation ignores ``max_num_sentences``, so a single engine
    leaves the card idle most of the time: it waits while the CPU normalises,
    phonemises and tokenises the next sentence.  A pool keeps that CPU work of
    one segment overlapped with the GPU work of another, which is what raises
    utilisation and shortens the recording.  Every segment is still synthesised
    by the same model with the same arguments, one sentence at a time, so the
    audio is exactly what a single engine would have produced.
    """

    def __init__(
        self,
        voice_entry: dict,
        provider: str = "cuda",
        size: int = 2,
        num_threads: int = 0,
    ):
        if not num_threads:
            num_threads = default_num_threads()
        self.provider = provider
        self._free: queue.Queue[TtsEngine] = queue.Queue()
        engines: list[TtsEngine] = []
        try:
            for _ in range(max(1, int(size))):
                engines.append(
                    TtsEngine(voice_entry, provider=provider, num_threads=num_threads)
                )
        except EngineUnavailableError:
            # Out of video memory (or out of disk after a partial download):
            # keep whatever did load rather than failing the whole recording.
            if not engines:
                raise
            log.warning(
                "Only %d of %d TTS engines could be loaded; continuing with %d",
                len(engines), size, len(engines),
            )
        for engine in engines:
            self._free.put(engine)
        self.sample_rate = engines[0].sample_rate

    @property
    def size(self) -> int:
        """How many engines this pool actually holds."""
        return self._free.qsize()

    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        """Synthesize on the first free engine (same signature as the engine)."""
        engine = self._free.get()
        try:
            return engine.synthesize(
                text, sid=sid, speed=speed, pitch=pitch, volume=volume
            )
        finally:
            self._free.put(engine)


def get_engine_pool(
    voice_entry: dict,
    provider: str = "cpu",
    size: int = 2,
    num_threads: int = 0,
):
    """An engine that can synthesize ``size`` segments at the same time.

    ``size`` of 1 - or a voice that is not a local sherpa-onnx model - returns
    the shared cached engine, so a pool is only ever created when it can help.
    """
    if size <= 1 or not is_local_sherpa_engine(voice_entry):
        return get_engine(voice_entry, provider=provider, num_threads=num_threads)
    return TtsEnginePool(voice_entry, provider=provider, size=size, num_threads=num_threads)


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


def process_punctuation(text: str, mode: str,
                        engine_id: str | None = None) -> str:
    """Apply the selected punctuation mode (default/none/math/all).

    ``engine_id`` is the engine the text is destined for.  An engine that reads
    expression markup - Kitten TTS 2.0's ``[emotion]`` tags, ``<event>`` tags
    and ``(((emphasis)))`` spans - keeps them exactly as typed: the spans are
    set aside before the mode runs and written back afterwards (*every* mode, so
    neither the spoken-word expansion of ``all`` nor the symbol stripping of
    ``none``/``math`` can reach them), because they are director's notes rather
    than punctuation, and an engine conditions on them instead of speaking
    them.  Without the engine - or for one that has no such markup - nothing
    changes: the mode sees the whole text, as it always did.
    """
    if mode == PUNCTUATION_DEFAULT:
        return text
    text, spans = markup.protect_for(engine_id, text)
    return markup.restore(_apply_punctuation_mode(text, mode), spans)


def _apply_punctuation_mode(text: str, mode: str) -> str:
    """The mode itself, with any expression markup already set aside."""
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
def _resample(samples: np.ndarray, factor: float) -> np.ndarray:
    """Linearly resample a 1-D int16 array by ``factor`` (>1 speeds up)."""
    n_out = round(len(samples) * factor)
    if n_out == len(samples):
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _shift_pitch(samples: np.ndarray, factor: float) -> np.ndarray:
    """Approximate pitch shift without changing duration.

    Resample up/down by ``factor`` (changes pitch) then time-stretch back to the
    original length. Reasonable quality for speech; documented as approximate.
    """
    if factor <= 0.01:
        return samples
    try:
        shifted = _resample(samples, factor)
        return _resample(shifted, 1.0 / factor)
    except Exception:
        return samples


def _apply_volume(samples: np.ndarray, volume: float) -> np.ndarray:
    scaled = samples.astype(np.float64) * float(volume)
    np.clip(scaled, -32768, 32767, out=scaled)
    return scaled.astype(np.int16)


# ---------------------------------------------------------------------------
# Convenience for the Preview button / quick tests
# ---------------------------------------------------------------------------
def preview_synthesis(
    text: str,
    voice_entry: dict,
    provider: str = "cpu",
    speed: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
    punctuation: str = PUNCTUATION_DEFAULT,
) -> np.ndarray:
    engine = get_engine(voice_entry, provider=provider)
    text = process_punctuation(text, punctuation, voice_entry.get("engine"))
    return engine.synthesize(text, sid=voice_entry.get("sid", 0), speed=speed, pitch=pitch, volume=volume)
