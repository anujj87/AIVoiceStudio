"""Voice Lab worker subprocess (standalone, runs in the managed virtualenv).

Protocol (JSON lines on stdin/stdout)::

    {"cmd": "ping"}                                   -> engine availability
    {"cmd": "voices", "engine": "pocket_tts"}         -> built-in voice availability
    {"cmd": "synthesize", "engine": "bark", ...}      -> {"wav": "<base64 int16>"}
    {"cmd": "quit"}                                   -> the process exits

The engines are kept in *this* process so their (large) PyTorch models
stay loaded between segments.  The main application only ever pipes JSON to it,
which keeps PyTorch/CUDA out of the GUI process - the same isolation the
OmniVoice engine uses.

Every ``synthesize`` request may carry an ``options`` dict with the per-engine
tuning values (see ``ai_voice_studio/voicelab/options.py``).  The worker passes
those to the engine *only* when the installed version of the engine accepts
them, and answers with the ``applied`` and ``skipped`` option names so the app
can tell the user instead of silently ignoring a knob.

Keep this module importable by a bare Python interpreter: it must not import
anything from ``ai_voice_studio``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import struct
import sys
import tempfile
import wave

import numpy as np

DEFAULT_SAMPLE_RATE = 24000

# Kyutai Pocket TTS language names (its config files are named after the
# language, e.g. ``english_2026-04.yaml``).
_POCKET_LANGUAGES = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
}

#: Tuning options the worker itself consumes (it seeds the RNGs before calling
#: an engine), so they are never forwarded as engine arguments.
_WORKER_OPTIONS = ("seed",)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet():
    """Run engine code with stdout redirected to stderr.

    Engines print progress to stdout; the worker's stdout is the JSON channel,
    so anything printed there would corrupt the protocol.
    """
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


def _signature(callable_obj):
    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def _supported_kwargs(callable_obj, kwargs):
    """Split ``kwargs`` into what ``callable_obj`` accepts and what it does not.

    The engines are young and rename their arguments between releases (F5-TTS
    renamed ``ref_audio`` to ``ref_file``, for example), so the worker asks each
    object what it supports instead of guessing.
    """
    if not kwargs:
        return {}, []
    params = _signature(callable_obj)
    if params is None:
        return dict(kwargs), []
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.parameters.values()):
        return dict(kwargs), []
    supported, skipped = {}, []
    for name, value in kwargs.items():
        if name in params.parameters:
            supported[name] = value
        else:
            skipped.append(name)
    return supported, skipped


def _pick_param(callable_obj, names, default=None):
    """First parameter name of ``callable_obj`` that is in ``names``."""
    params = _signature(callable_obj)
    if params is None:
        return default
    for candidate in names:
        if candidate in params.parameters:
            return candidate
    return default


def _accepts(callable_obj, name):
    params = _signature(callable_obj)
    return bool(params is not None and name in params.parameters)


class OptionReport:
    """Which tuning options a request actually used, and which it did not."""

    def __init__(self):
        self.applied = []
        self.skipped = []

    def reset(self):
        self.applied = []
        self.skipped = []

    def mark(self, name):
        if name not in self.applied:
            self.applied.append(name)

    def record(self, applied=(), skipped=()):
        for name in applied:
            self.mark(name)
        for name in skipped:
            if name not in self.skipped:
                self.skipped.append(name)

    def as_dict(self):
        return {"applied": list(self.applied), "skipped": list(self.skipped)}


def _apply_seed(seed, report=None):
    """Seed every RNG the engines may use; returns True when it was applied.

    Only engines with a native ``seed`` argument consume the value themselves;
    for the others an identical text + seed still reproduces the same audio
    because Torch/NumPy/random are seeded before generation.
    """
    if seed is None or str(seed).strip() == "":
        return False
    try:
        value = int(seed)
    except (TypeError, ValueError):
        return False
    try:
        import random  # noqa: PLC0415

        random.seed(value)
        np.random.seed(value % (2 ** 32))
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except Exception:  # noqa: BLE001
        pass
    return True


def _maximize_cuda(torch):
    """Use the whole available NVIDIA card for generation.

    Whatever CUDA-capable card the driver exposes is used - nothing here is
    tied to a particular model, VRAM size or compute capability, so any
    NVIDIA GPU is accepted, and the goal is the maximum throughput the
    installed card can deliver: cuDNN's autotuner, the TensorFloat-32 fast
    paths and "high" float32 matmul precision are enabled, and the autograd
    engine is switched off (these are inference-only engines; its bookkeeping
    costs time and memory for nothing).
    """
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.set_device(0)
    except Exception:  # noqa: BLE001
        return False
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.set_grad_enabled(False)
    except Exception:  # noqa: BLE001
        pass
    return True


def _limit_cpu(torch, threads):
    """Let a CPU run use the application's share of the machine.

    ``threads`` is computed by the application (80-95% of the logical CPUs,
    see ``ai_voice_studio.compute.cpu_threads``); the worker only applies it,
    so both sides always agree on how much CPU a run may take.
    """
    if not threads:
        return False
    try:
        value = max(1, int(threads))
    except (TypeError, ValueError):
        return False
    try:
        torch.set_num_threads(value)
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.set_num_interop_threads(max(1, value // 2))
    except Exception:  # noqa: BLE001
        # Raised once parallel work has started; the per-op thread pool set
        # above is the one that decides how much CPU inference uses.
        pass
    return True


def _apply_compute_settings(device, threads=None):
    """Apply the compute back-end the user chose, inside the worker process."""
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    if str(device).lower() == "cuda":
        _maximize_cuda(torch)
    else:
        _limit_cpu(torch, threads)


def _resample_speed(samples, speed):
    """Approximate ``speed`` by linear resampling (pitch shifts slightly)."""
    if speed <= 0.01 or abs(speed - 1.0) < 1e-3:
        return samples
    n_out = max(1, int(round(len(samples) / speed)))
    if n_out == len(samples):
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _to_int16(audio):
    """Normalise whatever an engine returned into int16 PCM samples."""
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        return arr.reshape(-1).copy()
    arr = arr.astype(np.float32).reshape(-1)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)


def _resolve_resource(spec):
    """Resolve ``"package:relative/path"`` to a real file on disk.

    Returns ``None`` when the package is not installed or does not contain the
    file (some engines ship their sample references only in the git checkout,
    not in the wheel).
    """
    if not spec:
        return None
    if ":" not in spec:
        return spec if os.path.isfile(spec) else None
    package, _, relative = spec.partition(":")
    try:
        from importlib.resources import files as _files  # noqa: PLC0415

        target = _files(package)
        for part in relative.split("/"):
            target = target.joinpath(part)
        if target.is_file():
            return str(target)
        # Zipped import (rare): copy the resource into a folder we own.
        cache = os.path.join(tempfile.gettempdir(), "aivs_voicelab_res")
        os.makedirs(cache, exist_ok=True)
        dest = os.path.join(cache, os.path.basename(relative))
        if not os.path.isfile(dest):
            with target.open("rb") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        return dest
    except Exception:
        return None


def _read_text_resource(spec):
    if not spec:
        return ""
    resolved = _resolve_resource(spec)
    if resolved and os.path.isfile(resolved):
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""
    return ""


def _module_available(name):
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Audio I/O without FFmpeg
# ---------------------------------------------------------------------------
#: Set to ``0``/``false``/``no`` to keep torchaudio's own audio I/O (useful
#: when diagnosing a decoding problem on a machine that has FFmpeg).
_AUDIO_SHIM_ENV = "AIVS_VOICELAB_NO_AUDIO_SHIM"


def _write_probe_wav(path, sample_rate=16000, seconds=0.05):
    """Write a tiny silent WAV so a decoder can be tried on a real file."""
    frames = max(1, int(sample_rate * seconds))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))


def _torchcodec_works(torchaudio):
    """True when torchaudio's decoder can really open a file.

    From torchaudio 2.9 on, ``load``/``save`` decode through TorchCodec (its
    ``backend`` argument is accepted and ignored).  On Windows TorchCodec
    needs FFmpeg's *shared* libraries - ``avcodec-*.dll`` and friends, from an
    FFmpeg "full-shared" build - which are not part of this application.  On a
    computer without them every decode raises ``Could not load libtorchcodec``
    for ``libtorchcodec_core<N>.dll``, which is what made F5-TTS fail on its
    reference recording.
    """
    path = os.path.join(tempfile.gettempdir(), "aivs_voicelab_probe.wav")
    try:
        _write_probe_wav(path)
        torchaudio.load(path)
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _soundfile_load(sf, torch, uri, frame_offset=0, num_frames=-1,
                    normalize=True, channels_first=True, format=None,
                    buffer_size=4096, backend=None):
    """``torchaudio.load`` implemented with soundfile (no FFmpeg needed).

    ``soundfile`` ships libsndfile inside its own wheel, so it reads WAV,
    FLAC, OGG and MP3 with nothing installed on the computer.  The arguments
    mirror torchaudio's; ``normalize``/``format``/``buffer_size``/``backend``
    only exist for compatibility (samples always come back as normalised
    float32, which is what the engines ask for).
    """
    if hasattr(uri, "read"):
        data, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
    else:
        try:
            start = max(0, int(frame_offset or 0))
        except (TypeError, ValueError):
            start = 0
        try:
            frames = int(num_frames)
        except (TypeError, ValueError):
            frames = -1
        data, sample_rate = sf.read(
            str(uri), start=start, frames=(frames if frames > 0 else -1),
            dtype="float32", always_2d=True,
        )
    # soundfile returns (frames, channels), torchaudio (channels, frames).
    samples = np.ascontiguousarray(data.T if channels_first else data)
    tensor = torch.from_numpy(samples.astype(np.float32, copy=False))
    return tensor, int(sample_rate)


def _soundfile_save(sf, torch, uri, src, sample_rate, channels_first=True,
                    format=None, encoding=None, bits_per_sample=None,
                    buffer_size=4096, backend=None, compression=None):
    """``torchaudio.save`` implemented with soundfile (no FFmpeg needed)."""
    data = src
    if hasattr(data, "detach"):  # torch tensor
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    array = np.asarray(data)
    if channels_first and array.ndim > 1:
        array = array.T
    if array.dtype.kind in "iu":  # integer PCM stays integer
        sf.write(str(uri), array, int(sample_rate), subtype="PCM_16")
    else:
        sf.write(
            str(uri),
            np.clip(array.astype(np.float32), -1.0, 1.0),
            int(sample_rate),
        )


def _import_soundfile():
    """Import soundfile (its wheel contains libsndfile; no FFmpeg needed)."""
    import soundfile  # noqa: PLC0415

    return soundfile


#: Suffixes the engines' own pre-processing can read without FFmpeg.
_WAV_SUFFIXES = (".wav", ".wave")


def _convert_to_wav(path, soundfile=None):
    """Write ``path`` next to the temp files as a WAV; returns the new path.

    F5-TTS cleans its reference clip with pydub, which shells out to FFmpeg
    for anything that is not a WAV (an MP3 reference fails with
    "[WinError 2] The system cannot find the file specified").  soundfile
    reads MP3/FLAC/OGG on its own, so those references are converted once and
    the engine sees a WAV it can handle everywhere.
    """
    try:
        sf = soundfile if soundfile is not None else _import_soundfile()
        size = os.path.getsize(path)
        stamp = int(os.path.getmtime(path))
        key = f"{path}|{size}|{stamp}".encode("utf-8", "replace")
        digest = hashlib.md5(key).hexdigest()[:12]
        destination = os.path.join(
            tempfile.gettempdir(), f"aivs_reference_{digest}.wav"
        )
        if os.path.isfile(destination) and os.path.getsize(destination) > 0:
            return destination
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(destination, data, int(sample_rate))
        return destination
    except Exception:  # noqa: BLE001
        # Leave the original in place: the engine either reads it itself or
        # reports what it cannot do, which is more useful than a silent swap.
        return path


def _install_audio_io_shim(torchaudio=None, soundfile=None, torch=None):
    """Back torchaudio's audio I/O with soundfile when TorchCodec cannot load.

    TorchCodec is only replaced when it is measured to be broken
    (:func:`_torchcodec_works` tries a real decode first), so a computer with
    FFmpeg keeps torchaudio's own decoder and its wider format support.  Set
    ``AIVS_VOICELAB_NO_AUDIO_SHIM=0`` to skip the swap entirely.

    Returns a note describing what was done, or ``None`` when nothing was
    changed; the note travels back to the application with the response so the
    user can see why an engine still worked on a machine without FFmpeg.
    """
    if str(os.environ.get(_AUDIO_SHIM_ENV, "")).strip().lower() in ("0", "false", "no"):
        return None
    if torchaudio is None:
        try:
            import torchaudio as _torchaudio  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
        torchaudio = _torchaudio
    if getattr(torchaudio, "_aivs_soundfile_io", False):
        return None
    if _torchcodec_works(torchaudio):
        return None
    if soundfile is None:
        try:
            soundfile = _import_soundfile()
        except Exception:  # noqa: BLE001
            return (
                "Audio decoding is unavailable: the TorchCodec decoder cannot "
                "load (FFmpeg's shared libraries are missing) and the "
                "soundfile package is not installed, so this engine may not "
                "be able to read its reference recording."
            )
    if torch is None:
        try:
            import torch as _torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
        torch = _torch

    def load(*args, **kwargs):
        return _soundfile_load(soundfile, torch, *args, **kwargs)

    def save(*args, **kwargs):
        return _soundfile_save(soundfile, torch, *args, **kwargs)

    torchaudio.load = load
    torchaudio.save = save
    torchaudio._aivs_soundfile_io = True
    return (
        "Audio files are read with the bundled soundfile library: this "
        "computer has no FFmpeg, which the TorchCodec decoder torchaudio "
        "normally uses needs. WAV, FLAC, OGG and MP3 work; formats only "
        "FFmpeg can decode (for example m4a/AAC) do not."
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend:
    """Base class: one loaded engine inside the worker.

    ``synthesize`` returns ``(audio, sample_rate, speed_handled)``.
    """

    name = ""
    sample_rate = DEFAULT_SAMPLE_RATE
    #: Option keys this backend handles itself (so the generic option filter
    #: never forwards them to an engine method by accident).
    own_options = ()

    def __init__(self, device, request=None):
        self.device = device
        self.request = request or {}
        self.report = OptionReport()
        #: Explanations to send back with this backend's responses (for
        #: example: audio is decoded with soundfile because there is no
        #: FFmpeg).  The application logs them; they never fail a request.
        self.notes = []

    def note(self, message):
        """Record an explanation for the user; empty messages are ignored."""
        if message and message not in self.notes:
            self.notes.append(message)

    # -- tuning options ------------------------------------------------------
    def request_options(self, consume=()):
        """This request's tuning options, minus the ones we handle ourselves."""
        raw = self.request.get("options") or {}
        if not isinstance(raw, dict):
            return {}
        handled = set(self.own_options) | set(_WORKER_OPTIONS) | set(consume)
        return {k: v for k, v in raw.items() if k not in handled}

    def option_kwargs(self, callable_obj, consume=()):
        """Tuning options ``callable_obj`` accepts; the rest is reported.

        The engines rename and add generation parameters between releases, so
        the worker asks each engine what it supports instead of guessing.
        """
        wanted = self.request_options(consume=consume)
        supported, skipped = _supported_kwargs(callable_obj, wanted)
        self.report.record(applied=supported.keys(), skipped=skipped)
        return supported

    def available_voices(self):
        """Voice ids this backend can speak without user-supplied audio."""
        return {}

    def synthesize(self, text, request):
        raise NotImplementedError


class PocketBackend(Backend):
    """Kyutai Pocket TTS (``pocket-tts``), a 100M CPU-first cloner."""

    name = "pocket_tts"

    own_options = ("quantize",)

    def __init__(self, device, request=None):
        super().__init__(device, request)
        from pocket_tts import TTSModel  # noqa: PLC0415

        language = _POCKET_LANGUAGES.get(
            str((request or {}).get("language", "en")).lower()[:2], "english"
        )
        load_kwargs = {"language": language}
        options = (request or {}).get("options") or {}
        if isinstance(options, dict) and options.get("quantize"):
            if self.device == "cuda":
                # int8 quantization is a CPU-only optimisation upstream.
                load_kwargs["quantize"] = False
                self.report.record(skipped=["quantize"])
            else:
                load_kwargs["quantize"] = True
        kwargs, skipped = _supported_kwargs(TTSModel.load_model, load_kwargs)
        if load_kwargs.get("quantize") and "quantize" not in kwargs:
            self.report.record(skipped=["quantize"])
        self.report.record(applied=kwargs.keys(), skipped=skipped)
        with _quiet():
            self.model = TTSModel.load_model(**kwargs)
        if self.device == "cuda":
            try:
                self.model.to("cuda")
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"Pocket TTS could not use the GPU ({exc}); using the CPU.\n"
                )
                self.device = "cpu"
        try:
            self.sample_rate = int(self.model.sample_rate)
        except Exception:  # noqa: BLE001
            self.sample_rate = DEFAULT_SAMPLE_RATE
        self._states = {}

    def _state(self, prompt):
        state = self._states.get(prompt)
        if state is None:
            with _quiet():
                state = self.model.get_state_for_audio_prompt(prompt)
            self._states[prompt] = state
        return state

    def synthesize(self, text, request):
        prompt = (
            (request.get("ref_audio") or "").strip()
            or (request.get("ref_resource") or "").strip()
            or request.get("voice")
            or "alba"
        )
        state = self._state(prompt)
        with _quiet():
            audio = self.model.generate_audio(state, text)
        return audio, self.sample_rate, False


class BarkBackend(Backend):
    """Suno Bark (``transformers``): one transformer for text -> audio."""

    name = "bark"

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.repo = (request or {}).get("repo") or "suno/bark"
        self._processor = None
        self._model = None
        self.sample_rate = DEFAULT_SAMPLE_RATE

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor, BarkModel  # noqa: PLC0415

        with _quiet():
            self._processor = AutoProcessor.from_pretrained(self.repo)
            self._model = BarkModel.from_pretrained(self.repo).eval()
        if self.device == "cuda":
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                sys.stderr.write(
                    "CUDA is not available to PyTorch; Bark will run on the CPU.\n"
                )
                self.device = "cpu"
        try:
            self.sample_rate = int(
                self._model.generation_config.sample_rate or DEFAULT_SAMPLE_RATE
            )
        except Exception:  # noqa: BLE001
            self.sample_rate = DEFAULT_SAMPLE_RATE

    def synthesize(self, text, request):
        import torch  # noqa: PLC0415

        self._load()
        ref_audio = (request.get("ref_audio") or "").strip()
        kwargs = {"text": text}
        if ref_audio:
            # Bark "clones" from a speaker-embedding file (the same .npz
            # arrays that back its own voice-preset library), not from raw
            # audio.
            if not ref_audio.lower().endswith(".npz"):
                raise ValueError(
                    "Bark clones voices from a speaker-embedding file (.npz). "
                    "Choose a .npz file, or pick one of the built-in speakers."
                )
            if not os.path.isfile(ref_audio):
                raise FileNotFoundError(
                    f"The speaker embedding was not found: {ref_audio}"
                )
            kwargs["speaker_embeddings"] = [np.load(ref_audio, allow_pickle=True)]
        else:
            kwargs["voice_preset"] = request.get("voice") or "v2/en_speaker_0"
        with _quiet():
            inputs = self._processor(**kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        # The sampling temperatures are arguments of generate(), not of the
        # processor; anything the installed version cannot take is reported.
        gen_kwargs = self.option_kwargs(self._model.generate)
        with torch.no_grad(), _quiet():
            audio = self._model.generate(**inputs, **gen_kwargs)
        return audio, self.sample_rate, False


class F5TtsBackend(Backend):
    """F5-TTS (``f5-tts``) - a flow-matching transformer with voice cloning."""

    name = "f5tts"

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self._f5 = None
        self.sample_rate = DEFAULT_SAMPLE_RATE
        #: ``source path -> converted WAV`` so a clip is converted once.
        self._converted_refs = {}
        # F5-TTS reads its reference clip with ``torchaudio.load``, which from
        # torchaudio 2.9 on decodes through TorchCodec and therefore needs a
        # system FFmpeg.  Back it with soundfile *before* the engine runs so
        # the reference clip loads on a computer without FFmpeg.
        self.note(_install_audio_io_shim())

    def _engine(self):
        if self._f5 is None:
            from f5_tts.api import F5TTS  # noqa: PLC0415

            kwargs, _skipped = _supported_kwargs(
                F5TTS, {"model": "F5TTS_v1_Base", "device": self.device}
            )
            with _quiet():
                self._f5 = F5TTS(**kwargs)
        return self._f5

    def _readable_reference(self, path):
        """The reference clip as a WAV, which F5-TTS can clean without FFmpeg.

        A ``.wav`` reference is handed straight to the engine; anything else
        (MP3, FLAC, OGG - the formats the Voice Clone page offers) is converted
        with soundfile, because the pydub step inside F5-TTS needs FFmpeg for
        those.  The conversion is cached per path + size + timestamp.
        """
        if not path or str(path).lower().endswith(_WAV_SUFFIXES):
            return path
        if not os.path.isfile(path):
            return path
        cached = self._converted_refs.get(path)
        if cached and os.path.isfile(cached):
            return cached
        converted = _convert_to_wav(path)
        if converted == path:
            self.note(
                "The reference recording is not a WAV and could not be "
                "converted (install FFmpeg to use this format)."
            )
            return path
        self._converted_refs[path] = converted
        return converted

    def synthesize(self, text, request):
        engine = self._engine()
        ref = (request.get("ref_audio") or "").strip() or _resolve_resource(
            request.get("ref_resource") or ""
        )
        if not ref:
            raise RuntimeError(
                "F5-TTS needs a reference voice: pick a built-in reference "
                "voice or clone one from your own recording."
            )
        ref = self._readable_reference(ref)
        ref_text = (request.get("ref_text") or "").strip()
        if not ref_text:
            ref_text = _read_text_resource(request.get("ref_text_resource") or "")

        ref_param = _pick_param(engine.infer, ("ref_file", "ref_audio"), "ref_file")
        text_param = _pick_param(engine.infer, ("gen_text", "text"), "gen_text")
        kwargs = {
            ref_param: ref,
            "ref_text": ref_text,
            text_param: text,
        }
        speed = float(request.get("speed", 1.0) or 1.0)
        speed_handled = False
        if abs(speed - 1.0) > 1e-3 and _accepts(engine.infer, "speed"):
            kwargs["speed"] = speed
            speed_handled = True
        # The per-engine tuning knobs the installed F5-TTS understands
        # (nfe_step, cfg_strength, sway_sampling_coef, cross_fade_duration,
        # target_rms, fix_duration, remove_silence, seed ...).
        kwargs.update(self.option_kwargs(engine.infer, consume=("speed",)))
        if _accepts(engine.infer, "progress"):
            kwargs["progress"] = None
        if _accepts(engine.infer, "show_info"):
            kwargs["show_info"] = lambda *_a, **_k: None
        with _quiet():
            wav, sample_rate, _spectrogram = engine.infer(**kwargs)
        try:
            if sample_rate:
                self.sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            pass
        return wav, self.sample_rate, speed_handled


_BACKENDS = {
    "pocket_tts": PocketBackend,
    "bark": BarkBackend,
    "f5tts": F5TtsBackend,
}

#: Import the engine needs before it can speak.
_ENGINE_MODULES = {
    "pocket_tts": ("pocket_tts",),
    "bark": ("torch", "transformers"),
    "f5tts": ("f5_tts",),
}


def _engine_available(engine_id):
    return all(_module_available(name) for name in _ENGINE_MODULES.get(engine_id, ()))


def _backend_key(engine_id, device, request):
    """Cache key of a loaded backend: engine, device and the loaded model.

    Two requests that need *different* model weights (Bark v2 vs bark-small,
    an int8-quantized Pocket TTS vs the full-precision one) must not share a
    backend, or the second one would keep speaking with the first one's
    weights.
    """
    options = request.get("options") or {}
    if not isinstance(options, dict):
        options = {}
    if engine_id == "pocket_tts":
        return (
            engine_id, device,
            str(request.get("language", "en")).lower()[:2],
            bool(options.get("quantize") and device != "cuda"),
        )
    if engine_id == "bark":
        return (engine_id, device, request.get("repo") or "suno/bark")
    return (engine_id, device)


# ---------------------------------------------------------------------------
# Request loop
# ---------------------------------------------------------------------------
def main():
    out = sys.stdout
    backends = {}

    def respond(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    def backend_for(engine_id, device, request):
        key = _backend_key(engine_id, device, request)
        backend = backends.get(key)
        if backend is None:
            cls = _BACKENDS.get(engine_id)
            if cls is None:
                raise RuntimeError(f"Unknown Voice Lab engine: {engine_id}")
            backend = cls(device, request)
            backends[key] = backend
        return backend

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            respond({"ok": False, "error": "Invalid request."})
            continue
        cmd = req.get("cmd")

        if cmd == "quit":
            break

        if cmd == "ping":
            respond({
                "ok": True,
                "engines": {name: _engine_available(name) for name in _BACKENDS},
            })
            continue

        if cmd == "voices":
            engine_id = req.get("engine", "")
            cls = _BACKENDS.get(engine_id)
            if cls is None:
                respond({"ok": False, "error": f"Unknown engine: {engine_id}"})
                continue
            try:
                backend = backend_for(engine_id, req.get("device", "cpu"), req)
                respond({"ok": True, "available": backend.available_voices()})
            except Exception as exc:  # noqa: BLE001
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        if cmd == "synthesize":
            engine_id = req.get("engine", "")
            device = (
                "cuda" if str(req.get("device", "cpu")).lower() == "cuda" else "cpu"
            )
            # GPU: use the whole available card.  CPU: use the share the
            # application computed (80-95% of the logical CPUs).
            _apply_compute_settings(device, req.get("threads"))
            try:
                text = req["text"]
                if not str(text).strip():
                    raise ValueError("Nothing to synthesize.")
                backend = backend_for(engine_id, device, req)
                backend.report.reset()
                options = req.get("options") or {}
                if not isinstance(options, dict):
                    options = {}
                if _apply_seed(options.get("seed")):
                    backend.report.mark("seed")
                audio, sample_rate, speed_handled = backend.synthesize(text, req)
                samples = _to_int16(audio)
                if samples.size == 0:
                    raise ValueError("The engine returned no audio.")
                speed = float(req.get("speed", 1.0) or 1.0)
                if not speed_handled:
                    samples = _resample_speed(samples, speed)
                report = backend.report.as_dict()
                respond({
                    "ok": True,
                    "wav": base64.b64encode(samples.tobytes()).decode("ascii"),
                    "sample_rate": int(sample_rate or DEFAULT_SAMPLE_RATE),
                    "device_used": getattr(backend, "device", device),
                    "speed_resampled": (not speed_handled)
                    and abs(speed - 1.0) > 1e-3,
                    "applied": report["applied"],
                    "skipped": report["skipped"],
                    "notes": list(getattr(backend, "notes", []) or []),
                })
            except Exception as exc:  # noqa: BLE001
                respond({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue

        respond({"ok": False, "error": f"Unknown command: {cmd}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
