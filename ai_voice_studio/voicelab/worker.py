"""Voice Lab worker subprocess (standalone, runs in the managed virtualenv).

Protocol (JSON lines on stdin/stdout)::

    {"cmd": "ping"}                                   -> engine availability
    {"cmd": "voices", "engine": "pocket_tts"}         -> built-in voice availability
    {"cmd": "synthesize", "engine": "f5tts", ...}      -> {"wav": "<base64 int16>"}
    {"cmd": "quit"}                                   -> the process exits

The engines are kept in *this* process so their (large) PyTorch models
stay loaded between segments.  The main application only ever pipes JSON to it,
which keeps PyTorch/CUDA out of the GUI process - the same isolation the
OmniVoice engine uses.

A ``synthesize`` request carries **one** text (``text``) or a **list** of them
(``texts``); ``wav``/``wavs`` answer in the same shape.  A list is how the
application lets one card do more than one segment at a time (see
``voicelab.parallel_plan``): the engines whose own calls take a list of texts
generate them in one pass, and the others get one worker process per segment.

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
import importlib
import inspect
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import urllib.request
import wave
import zipfile

import numpy as np

# The CUDA caching allocator reads this when it first talks to the card, which
# is well after this module is imported but while no engine has loaded torch
# yet - so setting it here, at import time, is early enough.  ``expandable
# _segments`` releases the memory a finished generation used instead of keeping
# it in per-size pools: without it, a card that has already run a long segment
# fragments into holes too small for the next model copy or batch, and an
# engine that could have used the whole card reports "out of memory" instead.
# A user (or a future setting) may have chosen something else: only a missing or
# empty value is filled in.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_SAMPLE_RATE = 24000

#: Text the speed benchmark speaks when the application does not send its own.
#: One ordinary sentence is the right unit for a real-time factor: it is what
#: the per-call overheads of a whole recording are paid on, and it keeps the
#: probe quick on a slow CPU.
_BENCHMARK_TEXT = (
    "The quick brown fox jumps over the lazy dog, and the reader carries on "
    "at a steady pace."
)

#: What the benchmark speaks first, to load the model: its time is excluded
#: from the measurement (a first run may be downloading the weights), and one
#: short word keeps the warm-up from costing anything worth counting.
_BENCHMARK_WARMUP_TEXT = "Hello."

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

#: What a gated/unauthorised Hugging Face download says; matched
#: case-insensitively.  The last three are Kyutai's wording for Pocket TTS's
#: voice-cloning weights.  Kept in step with ``ai_voice_studio.hf_access``.
_GATED_MARKERS = (
    "gated",
    "gatedrepoerror",
    "must be authenticated",
    "cannot access",
    "unauthorized",
    "you must have access",
    "access to model",
    "please log in",
    "repository not found",
    "accept the terms",
    "logged in locally",
    "hf auth login",
)

#: Kyutai's *voice-cloning* Pocket TTS weights live in this gated repository
#: (the cloning-free ones do not).
_POCKET_VOICE_CLONING_REPO = "kyutai/pocket-tts"

#: What a clone attempt gets when those weights are not available.  This worker
#: is a standalone script in the engine's own environment and imports no part of
#: the application, so the wording is repeated here rather than imported (the
#: engine's ``ai_voice_studio.hf_access`` says the same thing).  Without it the
#: user gets the library's own advice - "make sure you're logged in locally with
#: `uvx hf auth login`" - which names a command this installation does not have.
_POCKET_NO_CLONING_HELP = (
    "Pocket TTS could not download the weights that can clone a voice, so it "
    "loaded the ones that cannot: only its 26 pre-made voices can be spoken. "
    f"Cloning needs the gated Hugging Face repository "
    f"{_POCKET_VOICE_CLONING_REPO}:\n"
    f"1. Open https://huggingface.co/{_POCKET_VOICE_CLONING_REPO} and accept "
    "its licence (its terms of use) with your account.\n"
    "2. Create a read token at https://huggingface.co/settings/tokens and "
    "paste it into Settings > Hugging Face account in the application (an "
    "HF_TOKEN environment variable or an earlier huggingface-cli login works "
    "too).\n"
    "Then run this again: the cloning weights download authenticated, and the "
    "cloned voice works."
)

#: One line for the run's notes: the model loaded, but without cloning.
_POCKET_NO_CLONING_NOTE = (
    "Pocket TTS is running without voice cloning: the gated weights at "
    f"https://huggingface.co/{_POCKET_VOICE_CLONING_REPO} were not "
    "downloaded. Accept their licence and add a token in Settings > Hugging "
    "Face account to clone a voice; the pre-made voices work either way."
)

#: Tuning options the worker itself consumes (it seeds the RNGs before calling
#: an engine), so they are never forwarded as engine arguments.
_WORKER_OPTIONS = ("seed",)


def _is_access_failure(text):
    """True when a failure is a missing Hugging Face access, not a bug."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _GATED_MARKERS)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _protocol_stream():
    """The stream the JSON protocol is written to, safe from engine noise.

    An engine can spawn helpers - the ``qwen-tts`` package imports the SoX
    tokenizer, whose wrapper shells out to ``sox`` - and a child process writes
    to the *file descriptor*, which redirecting ``sys.stdout`` (``_quiet``)
    cannot catch.  One stray line would be read by the application as an answer
    and shift every response after it by one request, which is what made the
    first batched Qwen3-TTS call report "Invalid response from the Voice Lab
    worker" four segments in a row.

    So the protocol keeps its own descriptor, and whatever an engine writes to
    descriptor 1 lands on stderr instead - which the application discards.  An
    interpreter that cannot duplicate the descriptor (an embedded one) keeps
    the old behaviour.
    """
    try:
        control = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        return sys.stdout
    try:
        return os.fdopen(control, "w", encoding="utf-8", newline="\n")
    except OSError:  # pragma: no cover - a descriptor that cannot be wrapped
        return sys.stdout


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
        import random

        random.seed(value)
        np.random.seed(value % (2 ** 32))
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _apply_seed: {exc}\n")
    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _apply_seed: {exc}\n")
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
    except Exception:
        return False
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _maximize_cuda: {exc}\n")
    try:
        # Let each attention call use the fastest kernel this build has for the
        # card - FlashAttention, then the memory-efficient one, then the plain
        # math path.  This chooses a *kernel*, not a precision: the audio is
        # the same, but an autoregressive engine spends most of its time in
        # attention, so a slow kernel is what leaves a card idle.
        for setting in ("enable_flash_sdp", "enable_mem_efficient_sdp",
                        "enable_math_sdp"):
            enable = getattr(torch.backends.cuda, setting, None)
            if enable is not None:
                enable(True)
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _maximize_cuda: {exc}\n")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _maximize_cuda: {exc}\n")
    try:
        torch.set_grad_enabled(False)
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _maximize_cuda: {exc}\n")
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
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _limit_cpu: {exc}\n")
    try:
        torch.set_num_interop_threads(max(1, value // 2))
    except Exception as exc:
        # Raised once parallel work has started; the per-op thread pool set
        # above is the one that decides how much CPU inference uses.
        sys.stderr.write(f"Ignoring a failure in _limit_cpu: {exc}\n")
    return True


def _no_grad():
    """A context in which PyTorch builds no autograd graph.

    ``_maximize_cuda`` switches the autograd engine off for a GPU run, but a
    CPU run had never done the same: an engine that does not disable it itself
    then pays the graph bookkeeping on every one of the thousands of tokens an
    autoregressive model writes.  This is the CPU half of that setting; a
    missing PyTorch (or one without the API) simply runs the call as it was.
    """
    try:
        import torch
    except Exception:
        return contextlib.nullcontext()
    try:
        return torch.no_grad()
    except Exception:
        return contextlib.nullcontext()


def _apply_compute_settings(device, threads=None):
    """Apply the compute back-end the user chose, inside the worker process."""
    try:
        import torch
    except Exception:
        return
    if str(device).lower() == "cuda":
        _maximize_cuda(torch)
    else:
        _limit_cpu(torch, threads)


def _resample_speed(samples, speed):
    """Approximate ``speed`` by linear resampling (pitch shifts slightly)."""
    if speed <= 0.01 or abs(speed - 1.0) < 1e-3:
        return samples
    n_out = max(1, round(len(samples) / speed))
    if n_out == len(samples):
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _read_wav_samples(path):
    """Read a PCM WAV as ``(float32 mono samples in [-1, 1], sample rate)``.

    KokoClone writes its finished audio to a file, so the worker reads it back
    instead of asking soundfile to do it: the file is one the engine wrote
    itself - always a plain PCM WAV - and the standard library decodes it
    exactly (a 16-bit file survives the round trip bit for bit, which is the
    precision the application works in anyway).
    """
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        data = np.array([], dtype=np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, int(sample_rate)


def _to_int16(audio):
    """Normalise whatever an engine returned into int16 PCM samples."""
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    # A half-precision tensor (float16, and bfloat16 especially - numpy has no
    # dtype for it at all) cannot be handed to numpy: ``.numpy()`` raises
    # "Got unsupported ScalarType BFloat16".  Engines such as VibeVoice run
    # their model in bfloat16 and return the waveform in that dtype, so widen
    # to float32 first.  This is loss-free for audio, and the conversion below
    # reads floats anyway.
    if hasattr(audio, "float") and hasattr(audio, "numpy"):
        audio = audio.float()
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
        from importlib.resources import files as _files

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
            with open(resolved, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""
    return ""


def _module_available(name):
    try:
        import importlib.util

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
    except Exception:
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
    import soundfile

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
        # A cache key for a converted file, not a security hash: the digest only
        # has to differ between two different audio files.
        digest = hashlib.md5(key, usedforsecurity=False).hexdigest()[:12]
        destination = os.path.join(
            tempfile.gettempdir(), f"aivs_reference_{digest}.wav"
        )
        if os.path.isfile(destination) and os.path.getsize(destination) > 0:
            return destination
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(destination, data, int(sample_rate))
        return destination
    except Exception:
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
            import torchaudio as _torchaudio
        except Exception:
            return None
        torchaudio = _torchaudio
    if getattr(torchaudio, "_aivs_soundfile_io", False):
        return None
    if _torchcodec_works(torchaudio):
        return None
    if soundfile is None:
        try:
            soundfile = _import_soundfile()
        except Exception:
            return (
                "Audio decoding is unavailable: the TorchCodec decoder cannot "
                "load (FFmpeg's shared libraries are missing) and the "
                "soundfile package is not installed, so this engine may not "
                "be able to read its reference recording."
            )
    if torch is None:
        try:
            import torch as _torch
        except Exception:
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
    #: How many texts one ``synthesize_many`` call may hold.  One means the
    #: engine speaks a **single** text per call, so it is given one worker
    #: process per segment instead of a batch (see ``voicelab.parallel_plan``
    #: and ``_GPU_PARALLEL`` in ``engines.py``, which declare the same thing
    #: for the application side).
    batch_limit = 1

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

    def synthesize_many(self, texts, request):
        """Speak several texts, returning one ``(audio, rate, handled)`` each.

        The base implementation is the honest one for an engine whose call
        takes a single text: they are spoken one after another.  The
        application gets its concurrency from separate worker processes for
        such an engine (``mode: "instances"``), so nothing is lost here.  An
        engine whose own call takes a list of texts (``batch_limit > 1``)
        overrides this and generates the whole batch in one pass.
        """
        return [self.synthesize(text, request) for text in texts]


class PocketBackend(Backend):
    """Kyutai Pocket TTS (``pocket-tts``), a 100M CPU-first cloner."""

    name = "pocket_tts"

    own_options = ("quantize",)

    def __init__(self, device, request=None):
        super().__init__(device, request)
        from pocket_tts import TTSModel

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
        # The library does not fail when the gated voice-cloning weights cannot
        # be downloaded: it quietly loads its cloning-free ones instead and only
        # *then*, if a clone is attempted, says so in terms of a CLI login.  Note
        # it here, and refuse a clone with the two steps that actually fix it.
        self.has_voice_cloning = bool(
            getattr(self.model, "has_voice_cloning", True)
        )
        if not self.has_voice_cloning:
            self.note(_POCKET_NO_CLONING_NOTE)
        if self.device == "cuda":
            try:
                self.model.to("cuda")
            except Exception as exc:
                sys.stderr.write(
                    f"Pocket TTS could not use the GPU ({exc}); using the CPU.\n"
                )
                self.device = "cpu"
        try:
            self.sample_rate = int(self.model.sample_rate)
        except Exception:
            self.sample_rate = DEFAULT_SAMPLE_RATE
        self._states = {}

    def _state(self, prompt):
        state = self._states.get(prompt)
        if state is None:
            try:
                with _quiet():
                    state = self.model.get_state_for_audio_prompt(prompt)
            except Exception as exc:
                if (_is_access_failure(f"{exc}")
                        or not getattr(self, "has_voice_cloning", True)):
                    raise RuntimeError(_POCKET_NO_CLONING_HELP) from exc
                raise
            self._states[prompt] = state
        return state

    def _wants_cloned_voice(self, request):
        """True when this request asks Pocket TTS to clone a voice.

        A cloned voice arrives as a reference clip (``ref_audio``, the file the
        Voice Lab stored it as, or ``ref_resource``); a pre-made voice arrives
        as its own id, which is not a path - and one may also arrive as a path
        of its own, so both spellings are honoured.
        """
        if (request.get("ref_audio") or "").strip():
            return True
        if (request.get("ref_resource") or "").strip():
            return True
        prompt = (request.get("voice") or "").strip()
        return bool(prompt) and os.path.isfile(prompt)

    def synthesize(self, text, request):
        if not self.has_voice_cloning and self._wants_cloned_voice(request):
            # Before the library's own misleading advice: the weights that clone
            # are the ones missing, and only a licence and a token fetch them.
            raise RuntimeError(_POCKET_NO_CLONING_HELP)
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
            from f5_tts.api import F5TTS

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
        except (TypeError, ValueError) as exc:
            sys.stderr.write(f"Ignoring a failure in F5TtsBackend.synthesize: {exc}\n")
        return wav, self.sample_rate, speed_handled


# ---------------------------------------------------------------------------
# GPU-only engines
# ---------------------------------------------------------------------------
def _cuda_or_error(name, package):
    """The ``torch`` module, or a plain-language error when CUDA is unusable.

    Chatterbox and Qwen3-TTS are far slower than real time on a CPU, so they
    are offered on the GPU alone.  A missing card, a driver PyTorch cannot use,
    or a CPU-only PyTorch wheel is reported here in the user's words instead of
    quietly turning a book into an overnight run.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"{name} needs PyTorch, which is not installed in its own Python "
            f"environment ({exc}). Install the engine again from Settings > "
            "Voice Clone."
        ) from exc
    try:
        usable = bool(torch.cuda.is_available())
    except Exception:
        usable = False
    if not usable:
        raise RuntimeError(
            f"{name} runs only on an NVIDIA GPU, and this environment cannot "
            "use one. Either the computer has no NVIDIA card with a working "
            f"driver, or {package} was installed with the CPU build of "
            "PyTorch. Install the engine again from Settings > Voice Clone on "
            "a computer with an NVIDIA GPU."
        )
    return torch


def _short_language(value):
    """``"en-US"`` / ``"EN_us"`` -> ``"en"`` (an engine's own short code)."""
    return str(value or "").strip().lower().split("-")[0].split("_")[0]


# ---------------------------------------------------------------------------
# Chatterbox (Resemble AI)
# ---------------------------------------------------------------------------
#: Chatterbox's three models: ``(variant id, module, class, languages)``.  The
#: two English models speak English alone; the multilingual one takes a
#: ``language_id`` for any of its 23 languages.  ``engines.py`` declares the
#: same three variants, and ``tests/test_voicelab.py`` fails when the two
#: drift apart.
_CHATTERBOX_LANGUAGES_ALL = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)

_CHATTERBOX_MODELS = (
    ("chatterbox_v1", "chatterbox.tts", "ChatterboxTTS", ("en",)),
    ("chatterbox_turbo", "chatterbox.tts_turbo", "ChatterboxTurboTTS", ("en",)),
    (
        "chatterbox_mtl",
        "chatterbox.mtl_tts",
        "ChatterboxMultilingualTTS",
        _CHATTERBOX_LANGUAGES_ALL,
    ),
)


class ChatterboxBackend(Backend):
    """Resemble AI's Chatterbox: zero-shot cloning from a reference clip.

    The variant picks the model: the 500M English model, its 350M Turbo
    sibling (paralinguistic tags such as ``[laugh]``) or the 500M multilingual
    model.  With no reference clip the model's own default voice speaks, which
    is what the engine's built-in voices ask for; with one, the clip is the
    voice.

    Chatterbox is GPU-only, so the model is loaded on the first synthesis
    rather than when the worker starts - answering "which voices are usable?"
    must not need a card, and a missing one has to be reported as what it is.
    """

    name = "chatterbox"

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = None
        self.variant = (
            str((request or {}).get("variant") or "").strip()
            or _CHATTERBOX_MODELS[0][0]
        )

    def _spec(self):
        for spec in _CHATTERBOX_MODELS:
            if spec[0] == self.variant:
                return spec
        return _CHATTERBOX_MODELS[0]

    def _load(self):
        """Load the selected model once per worker process."""
        if self._model is not None:
            return self._model
        _variant, module, class_name, _languages = self._spec()
        _cuda_or_error("Chatterbox", "chatterbox-tts")
        try:
            engine_class = getattr(importlib.import_module(module), class_name)
        except Exception as exc:
            raise RuntimeError(
                f"This Chatterbox installation does not have its "
                f"{module}.{class_name} model ({exc}). Install the engine "
                "again from Settings > Voice Clone."
            ) from exc
        kwargs, skipped = _supported_kwargs(
            engine_class.from_pretrained, {"device": "cuda"}
        )
        if skipped:
            # A version that names the argument differently still loads on the
            # device it defaults to; only report, never fail.
            sys.stderr.write(
                "Chatterbox from_pretrained does not accept "
                f"{', '.join(skipped)}; using its defaults.\n"
            )
        with _quiet():
            model = engine_class.from_pretrained(**kwargs)
        self._model = model
        try:
            self.sample_rate = int(getattr(model, "sr", None) or DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            self.sample_rate = DEFAULT_SAMPLE_RATE
        return self._model

    def available_voices(self):
        """Chatterbox has one voice of its own (every other is a clone)."""
        return {"default": True}

    def synthesize(self, text, request):
        model = self._load()
        _variant, _module, class_name, languages = self._spec()
        language = _short_language(request.get("language")) or "en"
        if language not in languages:
            label = "an English Chatterbox model" if languages == ("en",) else class_name
            raise RuntimeError(
                f"{label} only speaks English, and this voice asks for "
                f"'{language}'. Choose a voice from Chatterbox Multilingual "
                "for that language."
            )
        reference = (request.get("ref_audio") or "").strip() or _resolve_resource(
            request.get("ref_resource") or ""
        )
        kwargs = {"text": text}
        if reference:
            if not os.path.isfile(reference):
                raise FileNotFoundError(
                    f"The reference recording was not found: {reference}"
                )
            kwargs["audio_prompt_path"] = reference
        # The multilingual model needs to be told the language; the English
        # ones do not take the argument at all.
        if _accepts(model.generate, "language_id"):
            kwargs["language_id"] = language
        kwargs.update(self.option_kwargs(model.generate, consume=("language_id",)))
        with _quiet():
            wav = model.generate(**kwargs)
        return wav, self.sample_rate, False


# ---------------------------------------------------------------------------
# Qwen3-TTS (Alibaba Qwen)
# ---------------------------------------------------------------------------
#: Qwen3-TTS's four models: ``(variant id, Hugging Face repo, kind)``.  The
#: ``custom`` models speak one of their nine named speakers, the ``base`` ones
#: clone a voice from a recording.  ``engines.py`` declares the same variants.
_QWEN_MODELS = (
    ("qwen3_custom_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "custom"),
    ("qwen3_custom_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "custom"),
    ("qwen3_clone_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "clone"),
    ("qwen3_clone_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "clone"),
)

#: Qwen3-TTS's nine pre-made voices, by the id the package asks for them by.
#: Declared here as well as in ``engines.py`` (which builds the catalogue from
#: its own copy) so the worker can answer "is this voice usable?".
_QWEN_SPEAKERS = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)

#: The ten languages the models cover, as the package's ``language`` argument
#: spells them (an unknown code is sent as ``Auto``, which lets the model pick).
_QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


class QwenTtsBackend(Backend):
    """Qwen3-TTS: one of four models behind one call each.

    * a **custom** variant speaks with one of the package's nine named
      speakers (``generate_custom_voice``),
    * a **base** variant clones a voice from a reference clip and its
      transcript (``generate_voice_clone``), reusing the prompt features it
      extracted for a clip so a second line with the same voice does not pay
      for them again.

    The model is loaded on first use: Qwen3-TTS is GPU-only, and the weights
    (0.6-1.7B, downloaded from Hugging Face the first time) must not be pulled
    in just to answer a question about the engine.

    This is the engine that made batching worth having in the worker: its own
    ``generate_custom_voice``/``generate_voice_clone`` take a *list* of texts
    and decode them together, which is how a 12 GB card is filled by a model
    that alone leaves it two thirds idle.  Measured on an RTX 3060 (1.7B
    CustomVoice, four sentences of one segment each): 52 s in one batched call
    against 183 s for the same four calls one at a time - the same audio from
    the same weights at the same precision.
    """

    name = "qwen3tts"
    #: The package's own ceiling on a batch is the text list it can hold; eight
    #: is well inside it and keeps one bad batch from asking the card for more
    #: than the application's plan (``engines._GPU_PARALLEL``) allows.
    batch_limit = 8

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = None
        #: ``(reference, transcript)`` -> the prompt the engine built for it.
        self._prompts = {}
        self.variant = (
            str((request or {}).get("variant") or "").strip()
            or _QWEN_MODELS[0][0]
        )

    def _spec(self):
        for spec in _QWEN_MODELS:
            if spec[0] == self.variant:
                return spec
        return _QWEN_MODELS[0]

    def _load(self):
        """Load the selected model once per worker process."""
        if self._model is not None:
            return self._model
        _variant, repo, _kind = self._spec()
        torch = _cuda_or_error("Qwen3-TTS", "qwen-tts")
        try:
            # Imported quietly: this package pulls in its 25 Hz tokenizer,
            # which imports the SoX wrapper, and that prints a banner on a
            # computer without SoX (it is not needed for the 12 Hz models).
            with _quiet():
                from qwen_tts import Qwen3TTSModel
        except Exception as exc:
            raise RuntimeError(
                f"The qwen-tts package could not be imported ({exc}). Install "
                "the engine again from Settings > Voice Clone."
            ) from exc
        kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
        if _module_available("flash_attn"):
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            self.note(
                "FlashAttention 2 is not installed, so Qwen3-TTS uses "
                "PyTorch's own attention instead: the same audio, but it "
                "needs more GPU memory."
            )
        supported, skipped = _supported_kwargs(
            Qwen3TTSModel.from_pretrained, kwargs
        )
        for name in skipped:
            sys.stderr.write(
                f"Qwen3-TTS from_pretrained does not accept {name}; using "
                "its default.\n"
            )
        with _quiet():
            self._model = Qwen3TTSModel.from_pretrained(repo, **supported)
        return self._model

    def available_voices(self):
        """The nine named voices of a CustomVoice model (none on a Base one)."""
        _variant, _repo, kind = self._spec()
        if kind != "custom":
            return {}
        return dict.fromkeys(_QWEN_SPEAKERS, True)

    # -- cloning -------------------------------------------------------------
    def _clone_prompt(self, model, reference, ref_text):
        """The engine's own reusable prompt for a reference clip.

        Building it once per clip is what the package recommends for a long
        text read in one voice: extracting the speaker's features is the part
        that costs time, and the prompt carries exactly them.
        """
        key = (reference, ref_text)
        cached = self._prompts.get(key)
        if cached is not None:
            return cached
        kwargs = {"ref_audio": reference, "ref_text": ref_text}
        if hasattr(model, "create_voice_clone_prompt"):
            # ``x_vector_only_mode`` copies the speaker's timbre alone, which
            # is also the only way to clone without a transcript.
            if not ref_text:
                kwargs.setdefault("x_vector_only_mode", True)
            for name in ("x_vector_only_mode",):
                if name in kwargs and not _accepts(
                    model.create_voice_clone_prompt, name
                ):
                    kwargs.pop(name)
            kwargs.update(
                self.option_kwargs(
                    model.create_voice_clone_prompt,
                    consume=("ref_audio", "ref_text"),
                )
            )
            with _quiet():
                prompt = model.create_voice_clone_prompt(**kwargs)
        else:
            prompt = None
        self._prompts[key] = prompt
        return prompt

    def _speak_clone(self, model, text, language, request):
        reference = (request.get("ref_audio") or "").strip() or _resolve_resource(
            request.get("ref_resource") or ""
        )
        if not reference:
            raise RuntimeError(
                "Qwen3-TTS's Base models clone a voice, so they need a "
                "reference recording: create a voice from a sample on the "
                "Voice Clone page, or pick one of the nine built-in voices "
                "of a CustomVoice model instead."
            )
        if not os.path.isfile(reference):
            raise FileNotFoundError(
                f"The reference recording was not found: {reference}"
            )
        ref_text = (request.get("ref_text") or "").strip()
        if not ref_text:
            ref_text = _read_text_resource(request.get("ref_text_resource") or "")
        prompt = self._clone_prompt(model, reference, ref_text)
        kwargs = {"text": text, "language": language}
        kwargs.update(
            self.option_kwargs(
                model.generate_voice_clone,
                consume=("text", "language", "voice_clone_prompt",
                         "ref_audio", "ref_text"),
            )
        )
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt
        else:
            kwargs["ref_audio"] = reference
            if ref_text:
                kwargs["ref_text"] = ref_text
        with _quiet():
            wavs, rate = model.generate_voice_clone(**kwargs)
        return wavs, rate

    def _speak_custom(self, model, text, language, request):
        reference = (request.get("ref_audio") or "").strip()
        if reference and os.path.isfile(reference):
            raise RuntimeError(
                "Qwen3-TTS CustomVoice models speak one of their nine own "
                "voices and cannot clone a recording. Choose a voice from "
                "the 'Qwen3-TTS ... Base' variant to speak in a voice from "
                "your own sample."
            )
        speaker = str(request.get("voice") or "").strip() or _QWEN_SPEAKERS[0]
        kwargs = {"text": text, "language": language, "speaker": speaker}
        kwargs.update(
            self.option_kwargs(
                model.generate_custom_voice,
                consume=("text", "language", "speaker"),
            )
        )
        with _quiet():
            wavs, rate = model.generate_custom_voice(**kwargs)
        return wavs, rate

    def synthesize(self, text, request):
        model = self._load()
        _variant, _repo, kind = self._spec()
        language = _QWEN_LANGUAGE_NAMES.get(
            _short_language(request.get("language")), "Auto"
        )
        if kind == "clone":
            wavs, rate = self._speak_clone(model, text, language, request)
        else:
            wavs, rate = self._speak_custom(model, text, language, request)
        self._set_sample_rate(rate)
        if isinstance(wavs, (list, tuple)):
            if not len(wavs):
                raise ValueError("The engine returned no audio.")
            wavs = wavs[0]
        return wavs, self.sample_rate, False

    def synthesize_many(self, texts, request):
        """Speak several texts in **one** engine call.

        Every text is generated as its own sequence from its own prompt, so
        each answer is the audio a solo call would have produced; batching is
        how one model copy on one card decodes several segments at once
        instead of one after another.  A batch of one is not worth a list, so
        it goes the ordinary way.
        """
        if len(texts) < 2:
            return super().synthesize_many(texts, request)
        if len(texts) > self.batch_limit:
            raise RuntimeError(
                f"This engine speaks at most {self.batch_limit} texts in one "
                f"call, and {len(texts)} were asked for."
            )
        model = self._load()
        _variant, _repo, kind = self._spec()
        language = _QWEN_LANGUAGE_NAMES.get(
            _short_language(request.get("language")), "Auto"
        )
        wanted = list(texts)
        if kind == "clone":
            wavs, rate = self._speak_clone(model, wanted, language, request)
        else:
            wavs, rate = self._speak_custom(model, wanted, language, request)
        if not isinstance(wavs, (list, tuple)):
            wavs = [wavs]
        if len(wavs) != len(wanted):
            raise RuntimeError(
                f"The engine returned {len(wavs)} recording(s) for "
                f"{len(wanted)} text(s)."
            )
        self._set_sample_rate(rate)
        return [(wav, self.sample_rate, False) for wav in wavs]

    def _set_sample_rate(self, rate):
        try:
            self.sample_rate = int(rate or DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            self.sample_rate = DEFAULT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# KokoClone
# ---------------------------------------------------------------------------
#: The environment variables ``phonemizer`` itself reads.  The in-process
#: class attribute below is what phonemizer prefers, but ``kokoro-onnx`` -
#: the engine's own Kokoro stage - reads the variable, so both are set.
_ESPEAK_LIBRARY_ENV = "PHONEMIZER_ESPEAK_LIBRARY"
_ESPEAK_DATA_ENV = "PHONEMIZER_ESPEAK_DATA_PATH"


def _espeak_resolves_itself():
    """True when phonemizer finds an eSpeak NG library without our help."""
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        if EspeakWrapper._ESPEAK_LIBRARY:
            return True
        EspeakWrapper.library()
        return True
    except Exception:
        return False


def _prepare_espeak_phonemizer():
    """Aim the phonemiser at the eSpeak NG that ships with the engine.

    KokoClone routes every language through ``misaki.espeak`` ->
    ``phonemizer``, and both of those look for an eSpeak NG **installed on the
    computer**: misaki only ever probes a hard-coded macOS/Windows path and
    phonemizer falls back to ``ctypes.util.find_library``, so neither sees the
    copy that ships inside the ``espeakng_loader`` wheel - the only one this
    application installs.  Without this, the first call fails with
    ``RuntimeError: espeak not installed on your system``.

    Returns a note for the user, or ``None`` when there is nothing to report
    (a user-set ``PHONEMIZER_ESPEAK_LIBRARY`` is honoured, and a system eSpeak
    NG that phonemizer finds by itself is left alone).
    """
    configured = str(os.environ.get(_ESPEAK_LIBRARY_ENV) or "").strip()
    if configured and os.path.isfile(configured):
        return None
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        library = str(espeakng_loader.get_library_path())
        if os.path.isfile(library):
            EspeakWrapper.set_library(library)
            os.environ.setdefault(_ESPEAK_LIBRARY_ENV, library)
            try:
                data_path = str(espeakng_loader.get_data_path())
            except Exception:
                data_path = ""
            if os.path.isdir(data_path):
                EspeakWrapper.set_data_path(data_path)
                os.environ.setdefault(_ESPEAK_DATA_ENV, data_path)
            return None
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _prepare_espeak_phonemizer: {exc}\n")
    if _espeak_resolves_itself():
        return None
    return (
        "eSpeak NG was not found. KokoClone phonemises every language with "
        "it, and this environment has neither the copy that ships with the "
        "engine nor a system installation. Reinstall the KokoClone engine "
        "from the Voice Clone page, or install eSpeak NG and point "
        "PHONEMIZER_ESPEAK_LIBRARY at its libespeak-ng.dll."
    )


#: The built-in voices of KokoClone ``(voice id, language code)`` - the Kokoro
#: voices it speaks with before a reference voice is applied.  ``engines.py``
#: declares the same list (it builds the voice catalogue from it, and this copy
#: answers "is this voice usable?"); ``tests/test_voicelab.py`` fails when the
#: two drift apart.
_KOKOCLONE_VOICES = (
    ("af_bella", "en"),
    ("hf_alpha", "hi"),
    ("ff_siwis", "fr"),
    ("em_alex", "es"),
    ("im_nicola", "it"),
    ("pf_dora", "pt"),
    ("zf_001", "zh"),
)

#: Every language KokoClone itself can speak.  Japanese has no *built-in*
#: voice (its phonemiser is not installable from PyPI), but a voice entry may
#: still ask for it - for example a cloned one, on a machine where the user
#: installed ``pyopenjtalk`` themselves - so it is accepted here.
_KOKOCLONE_LANGUAGES = tuple(
    [language for _voice, language in _KOKOCLONE_VOICES] + ["ja"]
)

#: Modules KokoClone's own stack needs for every language (``misaki.espeak``
#: is the phonemiser it routes most languages through; its dependencies come
#: with ``kokoro-onnx``).
_KOKOCLONE_BASE_MODULES = ("kokoro_onnx", "misaki.espeak", "core.cloner")

#: Set once ``misaki``'s phonemiser has been made to return a ``(phonemes,
#: tokens)`` pair, and once the engine's language router has been made to hand
#: such a phonemiser out (see :func:`_ensure_misaki_g2p_compat`).
_MISAKI_G2P_PATCHED = False
_CLONER_ROUTER_PATCHED = False


def _as_pair(g2p):
    """A copy of ``g2p`` that always answers ``(phonemes, tokens)``.

    Works for a plain callable and for a method being installed on a class
    (which is then handed its ``self`` first), and passes a phonemiser that
    already returns the pair straight through, so a version that also gives
    the tokens back keeps them.
    """

    def __call__(*args, **kwargs):
        result = g2p(*args, **kwargs)
        if isinstance(result, tuple):
            return result
        return result, None

    __call__.__wrapped__ = g2p
    return __call__


def _ensure_misaki_g2p_compat():
    """Make the engine's phonemisers return a ``(phonemes, tokens)`` pair.

    ``misaki`` changed ``EspeakG2P.__call__`` from returning
    ``(phonemes, tokens)`` to returning the phoneme string alone, but
    KokoClone's own ``generate`` - and this worker - unpack the result as a
    pair.  A bare string then raises ``ValueError: too many values to unpack
    (expected 2)`` for **every** language routed through eSpeak (Hindi,
    French, Italian, Spanish, Portuguese), while English (no phonemiser) and
    Japanese (its own front-end) keep working - which is exactly the "first
    voice plays, the next one fails" report.

    Two repairs, each done once per process and each leaving an already
    pair-returning phonemiser untouched:

    * the class itself, because ``core.cloner`` binds ``EspeakG2P`` at import
      time and also builds its English for mixed text from it, and
    * the language router, whose Mandarin front-end is a closure built inside
      ``_get_config`` - invisible to any class patch, and unpacked as a pair
      by the engine's own ``generate`` when a Mandarin voice is cloned.
    """
    global _MISAKI_G2P_PATCHED, _CLONER_ROUTER_PATCHED
    if not _MISAKI_G2P_PATCHED:
        try:
            from misaki.espeak import EspeakG2P
        except Exception as exc:
            sys.stderr.write(f"Ignoring a failure in _ensure_misaki_g2p_compat: {exc}\n")
        else:
            EspeakG2P.__call__ = _as_pair(EspeakG2P.__call__)
            _MISAKI_G2P_PATCHED = True
    if _CLONER_ROUTER_PATCHED:
        return
    try:
        from core.cloner import KokoClone
    except Exception:
        return
    router = KokoClone._get_config
    if getattr(router, "__wrapped__", None) is not None:
        _CLONER_ROUTER_PATCHED = True
        return

    def _get_config(self, language):
        config = list(router(self, language))
        if len(config) > 3 and callable(config[3]):
            config[3] = _as_pair(config[3])
        return tuple(config)

    _get_config.__wrapped__ = router
    KokoClone._get_config = _get_config
    _CLONER_ROUTER_PATCHED = True


def _phonemize(g2p, text):
    """``(phonemes, tokens)`` from an engine phonemiser of any shape.

    ``_ensure_misaki_g2p_compat`` repairs the engine's phonemisers, but one
    that still hands back a bare string (a different engine version, or a
    user-set ``PHONEMIZER_*``) must not crash the run: accept either shape.
    """
    result = g2p(text)
    if isinstance(result, tuple):
        return result[0], (result[1] if len(result) > 1 else None)
    return result, None

#: Modules a single language needs on top of that - KokoClone's Mandarin
#: front-end (``misaki.zh``) is the only one with extra dependencies.
_KOKOCLONE_LANGUAGE_MODULES = {
    "zh": ("pypinyin", "jieba", "cn2an", "ordered_set"),
}


class KokoCloneBackend(Backend):
    """KokoClone: Kokoro ONNX speech + Kanade zero-shot voice conversion.

    Both ways of speaking go through the engine's own code:

    * a **cloned voice** (the request carries ``ref_audio``) - the engine's
      documented ``KokoClone.generate(text, lang, reference_audio,
      output_path)`` path: Kokoro synthesises the text and Kanade re-voices it
      to the reference recording,
    * a **built-in voice** - the Kokoro voice behind that language, without the
      voice conversion, so the engine is usable before a sample is recorded.

    Everything runs with the engine's *working directory* set to a folder the
    application owns: KokoClone downloads its weights into ``./model`` and
    ``./voice`` relative to the current directory, which is why the application
    sends a ``workspace`` with the request (its own folder may be read-only).

    The (large) Kanade model is loaded on first use, so answering "which
    built-in voices are usable?" costs nothing.
    """

    name = "kokoclone"

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._cloner = None
        self._kokoro = {}
        # Before anything of the engine is imported: misaki's phonemiser is
        # created inside the engine's own code, and it fails outright unless
        # phonemizer already knows which eSpeak NG to use.
        self.note(_prepare_espeak_phonemizer())
        self.workspace = self._prepare_workspace()

    # -- working directory ---------------------------------------------------
    def _prepare_workspace(self):
        """Move into the folder KokoClone may keep its model in."""
        base = str((self.request or {}).get("workspace") or "").strip()
        if not base:
            base = os.path.join(tempfile.gettempdir(), "aivs_kokoclone")
        for folder in (
            os.path.join(base, str(self.device)),
            os.path.join(tempfile.gettempdir(), "aivs_kokoclone", str(self.device)),
        ):
            try:
                os.makedirs(folder, exist_ok=True)
                os.chdir(folder)
                return folder
            except OSError:
                continue
        return base

    # -- engine --------------------------------------------------------------
    def _ensure_cloner(self):
        """Load KokoClone (Kokoro + Kanade), once per worker process."""
        # Before the engine's own ``generate`` can reach its phonemiser.
        _ensure_misaki_g2p_compat()
        if self._cloner is not None:
            return self._cloner
        from core.cloner import KokoClone

        with _quiet():
            cloner = KokoClone()
        self._cloner = self._use_requested_device(cloner)
        try:
            self.sample_rate = int(cloner.sample_rate)
        except Exception:
            self.sample_rate = DEFAULT_SAMPLE_RATE
        return self._cloner

    def _use_requested_device(self, cloner):
        """Run on the device the user picked, not the one KokoClone guessed.

        ``KokoClone`` puts its models on CUDA whenever PyTorch can see a card;
        the application offers the CPU as well, so a CPU request has to be
        honoured by moving the voice-conversion model and its vocoder back.
        """
        import torch

        on_gpu = self.device == "cuda" and torch.cuda.is_available()
        target = torch.device("cuda" if on_gpu else "cpu")
        if getattr(cloner, "device", None) == target:
            return cloner
        try:
            for attribute in ("kanade", "vocoder"):
                module = getattr(cloner, attribute, None)
                if module is not None:
                    setattr(cloner, attribute, module.to(target))
            cloner.device = target
        except Exception as exc:
            sys.stderr.write(
                f"KokoClone could not move to {target.type} ({exc}); "
                "keeping the device it chose.\n"
            )
            return cloner
        if target.type == "cpu":
            self.note("Voice cloning is running on the CPU.")
        else:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "NVIDIA GPU"
            self.note(f"Voice cloning is running on the GPU ({name}).")
        return cloner

    # -- built-in (un-cloned) Kokoro voices ----------------------------------
    def _kokoro_for(self, model_file, voices_file, vocab, cloner):
        """The engine's Kokoro model for one weights file, loaded once."""
        cached = self._kokoro.get(model_file)
        if cached is not None:
            return cached
        from kokoro_onnx import Kokoro

        with _quiet():
            if vocab:
                kokoro = Kokoro(model_file, voices_file, vocab_config=vocab)
            else:
                kokoro = Kokoro(model_file, voices_file)
        # KokoClone patches exports whose inputs differ from Kokoro's own
        # (the v1.1-zh graph takes integer input ids); use its patch when the
        # installed version has it.
        patcher = getattr(cloner, "_patch_kokoro_compat", None)
        if patcher is not None:
            kokoro = patcher(kokoro)
        self._kokoro[model_file] = kokoro
        return kokoro

    def _speak_builtin(self, text, language, voice_id):
        """Speak with the Kokoro voice behind a language (no cloning)."""
        cloner = self._ensure_cloner()
        router = getattr(cloner, "_get_config", None)
        if router is None:
            raise RuntimeError(
                "This KokoClone version does not expose the language routing "
                "its built-in voices need. Pick a cloned voice instead, or "
                "install a KokoClone version that has it."
            )
        # Ask KokoClone itself which model, voice file and phonemiser this
        # language uses, so a built-in voice can never speak with the wrong
        # model as the engine evolves.
        with _quiet():
            model_file, voices_file, vocab, g2p, default_voice, _en = router(language)
        kokoro = self._kokoro_for(model_file, voices_file, vocab, cloner)
        voice = str(voice_id or "").strip() or default_voice
        if voice.endswith(".pt"):
            voice = voice[:-3]
        with _quiet():
            if g2p is not None:
                phonemes, _tokens = _phonemize(g2p, text)
                audio, rate = kokoro.create(
                    phonemes, voice=voice, speed=1.0, is_phonemes=True
                )
            else:
                audio, rate = kokoro.create(text, voice=voice, speed=0.9,
                                            lang="en-us")
        return audio, rate

    # -- cloning -------------------------------------------------------------
    def _reference(self, request):
        return (
            (request.get("ref_audio") or "").strip()
            or _resolve_resource(request.get("ref_resource") or "")
            or ""
        )

    def _speak_clone(self, text, language, reference):
        """Kokoro + Kanade: the engine's own text-to-cloned-speech call."""
        cloner = self._ensure_cloner()
        if not os.path.isfile(reference):
            raise FileNotFoundError(
                f"The reference recording was not found: {reference}"
            )
        handle, output = tempfile.mkstemp(prefix="aivs_kokoclone_", suffix=".wav")
        os.close(handle)
        try:
            try:
                with _quiet():
                    cloner.generate(
                        text=text,
                        lang=language,
                        reference_audio=reference,
                        output_path=output,
                    )
            except TypeError as exc:
                raise RuntimeError(
                    "This KokoClone version does not accept its documented "
                    f"generate(text, lang, reference_audio, output_path) call: {exc}"
                ) from exc
            audio, rate = _read_wav_samples(output)
        finally:
            with contextlib.suppress(OSError):
                os.remove(output)
        return audio, rate

    # -- protocol ------------------------------------------------------------
    def available_voices(self):
        """Which built-in voices the installed packages can really speak."""
        missing_base = [
            module for module in _KOKOCLONE_BASE_MODULES
            if not _module_available(module)
        ]
        available = {}
        for voice, language in _KOKOCLONE_VOICES:
            extra = _KOKOCLONE_LANGUAGE_MODULES.get(language, ())
            available[voice] = not missing_base and all(
                _module_available(module) for module in extra
            )
        return available

    def synthesize(self, text, request):
        language = _kokoclone_language(request.get("language"))
        reference = self._reference(request)
        if reference:
            audio, rate = self._speak_clone(text, language, reference)
        else:
            audio, rate = self._speak_builtin(text, language, request.get("voice"))
        try:
            if rate:
                self.sample_rate = int(rate)
        except (TypeError, ValueError) as exc:
            sys.stderr.write(f"Ignoring a failure in KokoCloneBackend.synthesize: {exc}\n")
        return audio, self.sample_rate, False


def _kokoclone_language(value):
    """The language code KokoClone should speak (its own short codes)."""
    code = str(value or "").strip().lower().split("-")[0].split("_")[0]
    if code in ("jp", "jpn"):
        code = "ja"
    return code if code in _KOKOCLONE_LANGUAGES else "en"


# ---------------------------------------------------------------------------
# VibeVoice (Microsoft, community fork)
# ---------------------------------------------------------------------------
#: VibeVoice's two models: ``(variant id, Hugging Face repo)``.  Both are
#: declared by ``engines.py`` as well, and ``tests/test_voicelab.py`` fails
#: when the two lists drift apart.  The weights are fetched from Hugging Face
#: the first time the model is loaded.
_VIBEVOICE_MODELS = (
    ("vibevoice_1_5b", "vibevoice/VibeVoice-1.5B"),
    ("vibevoice_7b", "vibevoice/VibeVoice-7B"),
)

#: The voice presets the fork ships as reference recordings (``demo/voices``),
#: keyed by the id the catalogue offers them under.  The pip package does not
#: carry the ``demo`` folder, so a preset is fetched once from the pinned
#: revision this integration was checked against and cached in the engine's
#: own folder afterwards.  Declared here as well as in ``engines.py`` so the
#: worker can answer "is this built-in voice usable?" without loading a model.
_VIBEVOICE_VOICES = {
    "Alice": "en-Alice_woman.wav",
    "Maya": "en-Maya_woman.wav",
    "Carter": "en-Carter_man.wav",
    "Frank": "en-Frank_man.wav",
    "Samuel": "in-Samuel_man.wav",
    "Bowen": "zh-Bowen_man.wav",
    "Xinran": "zh-Xinran_woman.wav",
}

#: Where those presets are downloaded from (the pinned source revision).
_VIBEVOICE_VOICES_URL = (
    "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/"
    "952326ddb264062466a888cf32a5b2f4e803e16e/demo/voices/"
)

#: A speaker label VibeVoice's own training scripts use (``Speaker 1: ...``).
_VIBEVOICE_SPEAKER = re.compile(r"^\s*Speaker\s+\d+\s*:", re.MULTILINE)


def _vibevoice_script(text):
    """The script VibeVoice is asked to speak.

    The model is trained on scripts whose every line starts with a speaker
    label, so a plain paragraph is given the single-speaker label.  Text that
    already carries labels (the multi-speaker form the model documents) is
    passed through unchanged.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return stripped
    if _VIBEVOICE_SPEAKER.search(stripped):
        return stripped
    return f"Speaker 1: {stripped}"


class VibeVoiceBackend(Backend):
    """VibeVoice: long-form conversational TTS with voice conditioning.

    The variant picks the model: the 1.5B one (64K context, up to ~90 minutes
    per run) or the 7B "Large" one (32K context, ~45 minutes).  Both speak
    with a voice the model conditions on - the bundled presets are reference
    recordings, and one from the user's own library is a clone.

    VibeVoice is GPU-only and the weights are large, so the model is loaded on
    the first synthesis rather than when the worker starts: answering "which
    voices are usable?" must not need a card.
    """

    name = "vibevoice"
    #: The diffusion-sampler step count is applied through the model's own
    #: setter (``set_ddpm_inference_steps``), not as a ``generate`` argument.
    own_options = ("ddpm_steps",)

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = None
        self._processor = None
        self.variant = (
            str((request or {}).get("variant") or "").strip()
            or _VIBEVOICE_MODELS[0][0]
        )

    def _spec(self):
        for spec in _VIBEVOICE_MODELS:
            if spec[0] == self.variant:
                return spec
        return _VIBEVOICE_MODELS[0]

    def _load(self):
        """Load the processor and model once per worker process."""
        if self._model is not None:
            return self._model, self._processor
        _variant, repo = self._spec()
        torch = _cuda_or_error("VibeVoice", "vibevoice")
        try:
            with _quiet():
                from vibevoice.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference,
                )
                from vibevoice.processor.vibevoice_processor import (
                    VibeVoiceProcessor,
                )
        except Exception as exc:
            raise RuntimeError(
                f"The vibevoice package could not be imported ({exc}). Install "
                "the engine again from Settings > Voice Clone."
            ) from exc
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
        if _module_available("flash_attn"):
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            # The reference implementation falls back to SDPA in exactly the
            # same way; on Windows there is no prebuilt FlashAttention wheel,
            # so this is the normal path there.  Same audio, more memory.
            kwargs["attn_implementation"] = "sdpa"
            self.note(
                "FlashAttention 2 is not installed, so VibeVoice uses "
                "PyTorch's own attention instead: the same audio, but it "
                "needs a little more GPU memory."
            )
        supported, skipped = _supported_kwargs(
            VibeVoiceForConditionalGenerationInference.from_pretrained, kwargs
        )
        for name in skipped:
            sys.stderr.write(
                f"VibeVoice from_pretrained does not accept {name}; using "
                "its default.\n"
            )
        with _quiet():
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                repo, **supported
            )
            processor = VibeVoiceProcessor.from_pretrained(repo)
        try:
            model.eval()
        except Exception as exc:
            sys.stderr.write(f"Ignoring a failure in VibeVoiceBackend._load: {exc}\n")
        self._model, self._processor = model, processor
        return self._model, self._processor

    def _ddpm_steps(self):
        """The requested diffusion step count, or ``None`` (the model default)."""
        raw = (self.request or {}).get("options") or {}
        if not isinstance(raw, dict):
            return None
        try:
            return int(raw["ddpm_steps"])
        except (KeyError, TypeError, ValueError):
            return None

    def _apply_ddpm_steps(self, model):
        """Apply the sampler step count before a call; never fatal."""
        steps = self._ddpm_steps()
        if steps is None:
            return
        setter = getattr(model, "set_ddpm_inference_steps", None)
        if setter is None:
            self.report.record(skipped=["ddpm_steps"])
            return
        try:
            setter(num_steps=int(steps))
        except Exception:
            self.report.record(skipped=["ddpm_steps"])
            return
        self.report.record(applied=["ddpm_steps"])

    def available_voices(self):
        """The bundled voice presets (they are downloaded on first use)."""
        return dict.fromkeys(_VIBEVOICE_VOICES, True)

    # -- the voice the model conditions on -----------------------------------
    def _reference(self, request):
        """The reference recording for this call: a clone or a preset."""
        path = (request.get("ref_audio") or "").strip()
        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"The reference recording was not found: {path}"
                )
            return path
        spec = request.get("ref_resource") or ""
        resolved = _resolve_resource(spec)
        if resolved and os.path.isfile(resolved):
            return resolved
        preset = ""
        if spec.startswith("vibevoice:"):
            preset = spec.split("demo/voices/", 1)[-1]
        if preset:
            return self._preset(preset)
        raise RuntimeError(
            "VibeVoice needs a voice: pick one of its built-in voices, or "
            "clone a voice from your own recording."
        )

    def _cache_dir(self):
        """Where the downloaded presets live (the engine's own folder)."""
        workspace = str((self.request or {}).get("workspace") or "").strip()
        if workspace:
            return os.path.join(workspace, "voices")
        return os.path.join(tempfile.gettempdir(), "aivs_vibevoice", "voices")

    def _preset(self, filename):
        """Path of a bundled preset, downloading it once if needed."""
        name = os.path.basename(filename)
        destination = os.path.join(self._cache_dir(), name)
        if os.path.isfile(destination) and os.path.getsize(destination) > 0:
            return destination
        url = _VIBEVOICE_VOICES_URL + name
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response, \
                    open(destination, "wb") as out:
                shutil.copyfileobj(response, out)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download VibeVoice's '{name}' voice preset "
                f"({exc}). Check the internet connection and try again."
            ) from exc
        return destination

    def synthesize(self, text, request):
        model, processor = self._load()
        self._apply_ddpm_steps(model)
        reference = self._reference(request)
        inputs = processor(
            text=[_vibevoice_script(text)],
            voice_samples=[[reference]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to("cuda")
        kwargs = {
            "max_new_tokens": None,
            "tokenizer": processor.tokenizer,
            "generation_config": {"do_sample": False},
            "is_prefill": True,
        }
        # The tuning knobs the installed engine accepts (cfg_scale, ...); the
        # ones the worker sets itself are consumed above and never forwarded.
        kwargs.update(
            self.option_kwargs(
                model.generate,
                consume=("max_new_tokens", "tokenizer", "generation_config",
                         "is_prefill"),
            )
        )
        with _quiet():
            outputs = model.generate(**inputs, **kwargs)
        speech = list(getattr(outputs, "speech_outputs", None) or ())
        if not speech or speech[0] is None:
            raise RuntimeError("VibeVoice returned no audio for this text.")
        return speech[0], self.sample_rate, False


# ---------------------------------------------------------------------------
# MOSS-TTS (OpenMOSS / MOSI.AI)
# ---------------------------------------------------------------------------
#: The MOSS-TTS checkpoints this build offers: ``(variant id, Hugging Face
#: repo)``.  Both load through the standard ``AutoProcessor``/``AutoModel``
#: interface with the repository's own remote code.  ``engines.py`` declares
#: the same variants, and ``tests/test_voicelab.py`` fails when the two drift
#: apart.
_MOSS_MODELS = (
    (
        "moss_tts_local_v15_4b",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    ),
    (
        "moss_tts_local_1_7b",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    ),
)

#: MOSS-TTS language tags, by short code (the family spells them as names in
#: the Local-Transformer API).
_MOSS_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


class MossTtsBackend(Backend):
    """MOSS-TTS: the OpenMOSS speech family, two checkpoints that fit a card.

    The variant picks the model: the 48 kHz Local Transformer checkpoints
    (4B latest and 1.7B) load through Hugging Face's
    ``AutoProcessor``/``AutoModel`` and are asked for a ``reference`` when a
    voice is cloned.

    Both are GPU-only, and both load on the first synthesis rather than when
    the worker starts, so answering "which voices are usable?" never needs a
    card or a download.
    """

    name = "moss_tts"

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = None
        self._processor = None
        self.variant = (
            str((request or {}).get("variant") or "").strip()
            or _MOSS_MODELS[0][0]
        )

    def _spec(self):
        for spec in _MOSS_MODELS:
            if spec[0] == self.variant:
                return spec
        return _MOSS_MODELS[0]

    def _load(self):
        """Load the selected checkpoint once per worker process."""
        if self._model is not None:
            return self._model, self._processor
        # Both checkpoints write their audio through torchaudio, which from
        # 2.9 on saves through TorchCodec - and that needs FFmpeg's shared
        # libraries on Windows, which this application does not ship.  Without
        # the soundfile-backed shim every call ends in "TorchCodec is required
        # for save_with_torchcodec".
        self.note(_install_audio_io_shim())
        _variant, repo = self._spec()
        return self._load_local(repo)

    def _load_local(self, repo):
        """The 48 kHz Local Transformer checkpoint (its own remote code)."""
        import torch

        if self.device == "cuda":
            _cuda_or_error("MOSS-TTS", "moss-tts")
        try:
            with _quiet():
                from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                f"The transformers package could not be imported ({exc}). "
                "Install the engine again from Settings > Voice Clone."
            ) from exc
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn = "eager"
        if self.device == "cuda":
            if _module_available("flash_attn"):
                attn = "flash_attention_2"
            else:
                attn = "sdpa"
                self.note(
                    "FlashAttention 2 is not installed, so MOSS-TTS uses "
                    "PyTorch's own attention instead: the same audio, but it "
                    "needs a little more GPU memory."
                )
        with _quiet():
            processor = AutoProcessor.from_pretrained(
                repo, trust_remote_code=True
            )
            tokenizer = getattr(processor, "audio_tokenizer", None)
            if tokenizer is not None:
                tokenizer = tokenizer.to(self.device)
                processor.audio_tokenizer = tokenizer
            model = AutoModel.from_pretrained(
                repo,
                trust_remote_code=True,
                attn_implementation=attn,
                dtype=dtype,
            ).to(self.device)
        try:
            model.eval()
        except Exception as exc:
            sys.stderr.write(f"Ignoring a failure in MossTtsBackend._load_local: {exc}\n")
        try:
            config = getattr(processor, "model_config", None)
            self.sample_rate = int(getattr(config, "sampling_rate", 0)) or self.sample_rate
        except (TypeError, ValueError) as exc:
            sys.stderr.write(f"Ignoring a failure in MossTtsBackend._load_local: {exc}\n")
        self._model, self._processor = model, processor
        return self._model, self._processor

    def available_voices(self):
        """One default voice per variant: every other is a clone."""
        return {"default": True}

    # -- the reference recording --------------------------------------------
    def _reference(self, request):
        """The reference recording for a clone, or "" for the default voice."""
        path = (request.get("ref_audio") or "").strip()
        if not path:
            path = _resolve_resource(request.get("ref_resource") or "") or ""
        if path and not os.path.isfile(path):
            raise FileNotFoundError(
                f"The reference recording was not found: {path}"
            )
        return path

    def synthesize(self, text, request):
        model, processor = self._load()
        language = _short_language(request.get("language")) or "en"
        return self._speak_local(model, processor, text, language, request)

    def _speak_local(self, model, processor, text, language, request):
        """Speak with a Local Transformer checkpoint (48 kHz stereo)."""

        reference = self._reference(request)
        language_name = _MOSS_LANGUAGE_NAMES.get(language, "English")
        message_kwargs = {"text": text, "language": language_name}
        if reference:
            message_kwargs["reference"] = [reference]
        tokens = self._int_option("tokens")
        if tokens:
            message_kwargs["tokens"] = tokens
        with _quiet():
            conversation = processor.build_user_message(**message_kwargs)
            batch = processor([conversation], mode="generation")
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            kwargs.update(
                self.option_kwargs(
                    model.generate, consume=("input_ids", "attention_mask")
                )
            )
            outputs = model.generate(**kwargs)
            decoded = list(processor.decode(outputs))
        if not decoded:
            raise RuntimeError("MOSS-TTS returned no audio for this text.")
        audio = decoded[0].audio_codes_list[0]
        return audio, self.sample_rate, False

    def _int_option(self, key):
        """An integer tuning value, or ``None`` when unset/zero."""
        raw = (self.request or {}).get("options") or {}
        if not isinstance(raw, dict) or key not in raw:
            return None
        try:
            value = int(raw[key])
        except (TypeError, ValueError):
            return None
        return value or None


# ---------------------------------------------------------------------------
# GPT-SoVITS (RVC-Boss)
# ---------------------------------------------------------------------------
#: The GPT-SoVITS releases this build offers: ``(variant id, version)``.  The
#: version is the engine's own ``version`` key, which picks its weights and
#: languages.  ``engines.py`` declares the same variants.
_GPTSOVITS_MODELS = (
    ("gptsovits_v2proplus", "v2ProPlus"),
    ("gptsovits_v2pro", "v2Pro"),
    ("gptsovits_v4", "v4"),
    ("gptsovits_v3", "v3"),
    ("gptsovits_v2", "v2"),
    ("gptsovits_v1", "v1"),
)

#: GPT-SoVITS language codes, as its own ``*_lang`` arguments spell them.
#: Japanese is absent on purpose: its front-end imports ``pyopenjtalk``, which
#: has no wheel for this Python, and the import shim below leaves it out rather
#: than failing.  ``engines._GPTSOVITS_LANGUAGES`` declares the same set.
_GPTSOVITS_LANGUAGES = ("en", "zh", "ko", "yue")


#: The pinned GPT-SoVITS source archive (the commit this integration was
#: reviewed against).  It has no packaging metadata, so the worker fetches the
#: tree itself rather than pip installing it.  Kept in step with
#: ``engines.GPT_SOVITS_SOURCE_URL``, which the application declares.
_GPTSOVITS_SOURCE_URL = (
    "https://github.com/RVC-Boss/GPT-SoVITS/archive/"
    "48b1a0169a28582a8984402f82cf438d3bfa6aca.zip"
)

#: The Hugging Face repository holding every GPT-SoVITS pretrained weight.
_GPTSOVITS_WEIGHTS_REPO = "lj1995/GPT-SoVITS"

#: ``True`` once plain ``jieba`` has been registered as ``jieba_fast``.
_GPTSOVITS_JIEBA_PATCHED = False


def _gptsovits_tree_dir(request):
    """Where the fetched GPT-SoVITS source tree lives (the engine's folder)."""
    workspace = str((request or {}).get("workspace") or "").strip()
    if workspace:
        return workspace
    return os.path.join(tempfile.gettempdir(), "aivs_gptsovits")


def _download_file(url, destination):
    """Fetch ``url`` to ``destination`` (a ``.part`` file, then a rename)."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, \
            open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(partial, destination)


def _ensure_gptsovits_tree(workspace):
    """Fetch and unpack the pinned GPT-SoVITS source tree once.

    The archive unpacks to ``GPT-SoVITS-<sha>/``; its *contents* (which hold
    the ``GPT_SoVITS`` package and ``tools``) are what belong on ``sys.path``,
    so the single top-level directory is flattened into ``workspace``.
    """
    marker = os.path.join(workspace, ".aivs_gptsovits_tree")
    if os.path.isfile(marker) and os.path.isdir(
        os.path.join(workspace, "GPT_SoVITS")
    ):
        return workspace
    os.makedirs(workspace, exist_ok=True)
    archive = os.path.join(workspace, "gptsovits-src.zip")
    if not os.path.isfile(archive) or os.path.getsize(archive) < 1024:
        sys.stderr.write("Downloading the GPT-SoVITS source tree...\n")
        _download_file(_GPTSOVITS_SOURCE_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(workspace)
    # Flatten the single top-level folder the archive always has.
    roots = [
        os.path.join(workspace, name)
        for name in os.listdir(workspace)
        if os.path.isdir(os.path.join(workspace, name))
        and name.lower().startswith("gpt-sovits")
    ]
    if roots:
        root = roots[0]
        for name in os.listdir(root):
            source = os.path.join(root, name)
            destination = os.path.join(workspace, name)
            if os.path.exists(destination):
                continue
            shutil.move(source, destination)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("ready")
    return workspace


def _ensure_gptsovits_weights(workspace):
    """Download the pretrained weights once (a Hugging Face snapshot).

    The engine resolves its default weight paths *relative to the working
    directory* (``GPT_SoVITS/pretrained_models/...``), and the worker runs
    with the source tree as its working directory, so the snapshot is placed
    exactly where those paths look - inside the tree's own
    ``GPT_SoVITS/pretrained_models`` folder.  Then every default path the
    engine's config names resolves without being overridden.
    """
    weights = os.path.join(workspace, "GPT_SoVITS", "pretrained_models")
    marker = os.path.join(weights, ".aivs_gptsovits_weights")
    if os.path.isfile(marker):
        return weights
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            f"huggingface_hub is needed to download GPT-SoVITS's pretrained "
            f"weights ({exc}). Install the engine again from Settings > Voice "
            "Clone."
        ) from exc
    os.makedirs(weights, exist_ok=True)
    sys.stderr.write(
        "Downloading GPT-SoVITS's pretrained weights (~1.5-2 GB)...\n"
    )
    with _quiet():
        snapshot_download(
            repo_id=_GPTSOVITS_WEIGHTS_REPO,
            local_dir=weights,
            allow_patterns=(
                "*.ckpt", "*.pth", "*.bin", "*.pt", "*.json", "*.txt",
                "*.model", "*.yaml",
            ),
        )
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("ready")
    return weights


def _ensure_gptsovits_nltk():
    """Fetch the NLTK corpora the English front-end needs, if missing.

    ``GPT_SoVITS/text/english.py`` tags its text with ``nltk.pos_tag``, which
    needs the perceptron tagger and CMUdict.  Upstream's installer downloads
    them (its ``install.ps1`` unpacks a ``nltk_data.zip``); this integration
    fetches the same two packages on first use instead, quietly and only when
    they are actually missing.  A download that fails is not fatal - the
    engine's own ``LookupError`` then names the package, which is a better
    message than anything swallowed here.
    """
    try:
        import nltk
    except Exception:
        return
    for package in ("averaged_perceptron_tagger_eng", "cmudict"):
        try:
            nltk.data.find(f"taggers/{package}" if "tagger" in package
                           else f"corpora/{package}")
            continue
        except Exception as exc:
            sys.stderr.write(f"Ignoring a failure in _ensure_gptsovits_nltk: {exc}\n")
        with contextlib.suppress(Exception), _quiet():
            nltk.download(package, quiet=True)


def _ensure_gptsovits_jieba_fast():
    """Let the engine import on a Python ``jieba_fast`` cannot be built on.

    ``GPT_SoVITS/text/{chinese,chinese2,tone_sandhi}.py`` all do
    ``import jieba_fast``.  ``jieba_fast`` publishes a source distribution
    whose ``setup.py`` imports the removed ``pkgutil.ImpImporter``, so it
    cannot even be built on Python 3.13 and the engine fails at import time.
    Plain ``jieba`` is a drop-in for everything the engine uses - the
    segmentation and part-of-speech tagging it exposes under the same names -
    so it is registered under the ``jieba_fast`` name, once per process, and
    only when the real package is missing.
    """
    global _GPTSOVITS_JIEBA_PATCHED
    if _GPTSOVITS_JIEBA_PATCHED or _module_available("jieba_fast"):
        return
    try:
        import jieba
        import jieba.posseg
    except Exception:
        return
    sys.modules.setdefault("jieba_fast", jieba)
    sys.modules.setdefault("jieba_fast.posseg", jieba.posseg)
    _GPTSOVITS_JIEBA_PATCHED = True


def _gptsovits_import_tree(tree_dir):
    """Put the GPT-SoVITS source tree on ``sys.path`` (once per process).

    Two directories belong there, not one.  The repository root holds the
    ``GPT_SoVITS`` package, and ``GPT_SoVITS/`` itself holds the modules its
    own code imports as top level - ``AR``, ``module``, ``text``, ``tools``,
    ``BigVGAN`` - which upstream's ``api_v2.py`` adds the same way.  Without
    the second entry the first preview fails with ``No module named 'AR'``.
    """
    for directory in (os.path.join(tree_dir, "GPT_SoVITS"), tree_dir):
        if os.path.isdir(directory) and directory not in sys.path:
            sys.path.insert(0, directory)


class GptSovitsBackend(Backend):
    """GPT-SoVITS: few-shot voice cloning from five seconds of audio.

    GPT-SoVITS is a two-stage engine - a GPT stage turns the text into
    semantic tokens and a SoVITS stage renders them through a vocoder - and it
    always speaks from a reference recording plus its transcript.  It has no
    voices of its own, so every voice it uses is one the user created.

    The engine is a **source tree**, not a package: its worker adds the fetched
    tree to ``sys.path`` and points the engine at the pretrained weights in the
    engine's own folder.  The six releases differ only in which weights they
    load and which languages they cover.

    GPU-only: the project is built for CUDA and its GPT stage is far slower
    than real time on a CPU.
    """

    name = "gptsovits"

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = 32000
        self._tts = None
        self.variant = (
            str((request or {}).get("variant") or "").strip()
            or _GPTSOVITS_MODELS[0][0]
        )

    def _version(self):
        for variant, version in _GPTSOVITS_MODELS:
            if variant == self.variant:
                return version
        return _GPTSOVITS_MODELS[0][1]

    def _load(self):
        """Load the TTS pipeline once per worker process."""
        if self._tts is not None:
            return self._tts
        torch = _cuda_or_error("GPT-SoVITS", "gpt-sovits")
        # The engine reads its reference clip with ``torchaudio.load``.  From
        # torchaudio 2.9 on that decodes through TorchCodec, which needs
        # FFmpeg's shared libraries on Windows - so without this the very
        # first preview dies with "TorchCodec is required for
        # load_with_torchcodec".  The same soundfile-backed shim every other
        # engine installs is what makes the reference readable here.
        self.note(_install_audio_io_shim(torch=torch))
        tree_dir = _gptsovits_tree_dir(self.request)
        _ensure_gptsovits_tree(tree_dir)
        _ensure_gptsovits_weights(tree_dir)
        # The engine resolves its default weight paths (and writes its own
        # ``GPT_SoVITS/configs/tts_infer.yaml``) relative to the current
        # directory, so run from the source tree's root.
        try:
            os.chdir(tree_dir)
        except OSError as exc:
            sys.stderr.write(f"Ignoring a failure in GptSovitsBackend._load: {exc}\n")
        _gptsovits_import_tree(tree_dir)
        _ensure_gptsovits_jieba_fast()
        _ensure_gptsovits_nltk()
        try:
            with _quiet():
                from GPT_SoVITS.TTS_infer_pack.TTS import (
                    TTS,
                    TTS_Config,
                )
        except Exception as exc:
            raise RuntimeError(
                f"The GPT-SoVITS source tree could not be imported ({exc}). "
                "Install the engine again from Settings > Voice Clone."
            ) from exc
        configs = self._configs(tree_dir, torch)
        try:
            with _quiet():
                config = TTS_Config(configs)
                self._tts = TTS(config)
        except Exception as exc:
            raise RuntimeError(
                f"GPT-SoVITS could not load its {self._version()} weights "
                f"({exc}). Install the engine again from Settings > Voice "
                "Clone."
            ) from exc
        return self._tts

    def _configs(self, tree_dir, torch):
        """The engine's own config for the selected release.

        The ``custom`` key is what ``TTS_Config`` reads (without it the engine
        would silently use its v2 defaults whatever release was chosen).  Only
        the device, precision and version are set: the weight paths are left to
        the engine's own defaults, which the worker's working directory makes
        resolve inside the fetched tree.
        """
        half = self.device == "cuda"
        return {
            "custom": {
                "device": "cuda" if self.device == "cuda" else "cpu",
                "is_half": bool(half and torch.cuda.is_available()),
                "version": self._version(),
            }
        }

    def available_voices(self):
        """GPT-SoVITS has no voices of its own: every one is a clone."""
        return {}

    def synthesize(self, text, request):
        pipeline = self._load()
        reference = (request.get("ref_audio") or "").strip() or _resolve_resource(
            request.get("ref_resource") or ""
        )
        if not reference:
            raise RuntimeError(
                "GPT-SoVITS speaks in a voice cloned from a recording, and "
                "has no voices of its own: create a voice from a sample on "
                "the Voice Clone page first, then use it here."
            )
        if not os.path.isfile(reference):
            raise FileNotFoundError(
                f"The reference recording was not found: {reference}"
            )
        language = _short_language(request.get("language")) or "en"
        if language not in _GPTSOVITS_LANGUAGES:
            language = "en"
        prompt_text = (request.get("ref_text") or "").strip()
        inputs = {
            "text": text,
            "text_lang": language,
            "ref_audio_path": reference,
            "prompt_text": prompt_text,
            "prompt_lang": language,
        }
        speed = float(request.get("speed", 1.0) or 1.0)
        speed_handled = False
        if abs(speed - 1.0) > 1e-3:
            # The engine's own speed control time-stretches (a real speed
            # change); the application's resampling would shift the pitch.
            inputs["speed_factor"] = speed
            speed_handled = True
        inputs.update(
            self.option_kwargs(
                pipeline.run,
                consume=("text", "text_lang", "ref_audio_path", "prompt_text",
                         "prompt_lang", "speed_factor"),
            )
        )
        with _quiet():
            generator = pipeline.run(inputs)
            sample_rate, audio = next(iter(generator))
        try:
            self.sample_rate = int(sample_rate or self.sample_rate)
        except (TypeError, ValueError) as exc:
            sys.stderr.write(f"Ignoring a failure in GptSovitsBackend.synthesize: {exc}\n")
        return audio, self.sample_rate, speed_handled


# ---------------------------------------------------------------------------
# Indic Parler-TTS (AI4Bharat)
# ---------------------------------------------------------------------------
#: The repository Indic Parler-TTS downloads its checkpoint from.  It is gated,
#: so the worker's own text for the refusal is defined here (this module runs in
#: its own interpreter and may not import the application - see the module
#: docstring); ``hf_access`` keeps the application's wording and this one in
#: step.
_INDIC_PARLER_REPO = "ai4bharat/indic-parler-tts"

#: The tokenizer the *description* is tokenized with.  The checkpoint names the
#: encoder itself (``google/flan-t5-large``) and the model card loads its
#: tokenizer separately, so it is downloaded on first use like the checkpoint.
_INDIC_PARLER_DESCRIPTION_TOKENIZER = "google/flan-t5-large"

#: The two steps that make a gated repository downloadable.  The failure a
#: library reports is a bare 401 that names neither the page nor the panel, so a
#: person who installed the setup file cannot act on it.


def _gated_help(repo, what):
    """The licence page and the token panel for a gated repository."""
    return (
        f"{what} downloads from the gated Hugging Face repository {repo}:\n"
        f"1. Open https://huggingface.co/{repo} and accept its licence (its "
        "terms of use) with your account.\n"
        "2. Create a read token at https://huggingface.co/settings/tokens and "
        "paste it into Settings > Hugging Face account in the application (an "
        "HF_TOKEN environment variable or an earlier huggingface-cli login "
        "works too).\n"
        "Then run this again: the download starts authenticated."
    )


def _gated_failure(exc, repo, what):
    """``exc`` as a plain-language failure, or ``None`` when it is not gated.

    Returning ``None`` for anything else is what keeps a real bug (a disk-full,
    a broken wheel) from being dressed up as a missing licence.
    """
    if _is_access_failure(f"{exc}"):
        return RuntimeError(_gated_help(repo, what))
    return None


def _load_error(package, engine_name, exc):
    """The failure a missing package gives, in the user's own terms."""
    return RuntimeError(
        f"{engine_name} needs the '{package}' package, which is not installed "
        f"in its own Python environment ({exc}). Install the engine from "
        "Settings > Available TTS."
    )


class IndicParlerBackend(Backend):
    """Indic Parler-TTS (``parler-tts``): a description-conditioned 0.9B TTS.

    Two inputs: the **prompt** (what to say) and a **description** of the voice
    and delivery.  There is no voice-id argument - the description *is* the
    voice - so a built-in voice arrives as the model card's speaker name plus
    the sentence that asks for them, and the tuning dialog's *Voice description*
    replaces that sentence, which is the model's own voice-design mode.
    """

    name = "indic_parler"

    #: Handled here rather than passed on: the description is not an argument of
    #: the engine's ``generate`` (it is tokenized separately by the model card's
    #: own example).
    own_options = ("description",)

    #: Used when a request carries no description of its own - a clone, or a
    #: voice entry written by an older release.
    _FALLBACK = (
        "A speaker reads at a moderate pace with a slightly expressive "
        "delivery. The recording is of very high quality, with no background "
        "noise and the speaker very close to the microphone."
    )

    def __init__(self, device, request=None):
        super().__init__(device, request)
        # GPU only (see ``engines.py``): a 0.9B autoregressive transformer whose
        # CPU run is many times slower than real time.  ``_cuda_or_error`` says
        # so in plain words when the card or the CUDA build is missing.
        _cuda_or_error("Indic Parler-TTS", "parler-tts")
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except Exception as exc:
            raise _load_error("parler-tts", "Indic Parler-TTS", exc) from exc
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        with _quiet():
            try:
                self.model = (
                    ParlerTTSForConditionalGeneration
                    .from_pretrained(_INDIC_PARLER_REPO)
                    .to(self.device)
                )
            except Exception as exc:
                hint = _gated_failure(exc, _INDIC_PARLER_REPO,
                                      "Indic Parler-TTS")
                if hint is not None:
                    raise hint from exc
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(_INDIC_PARLER_REPO)
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                _INDIC_PARLER_DESCRIPTION_TOKENIZER
            )
        try:
            self.sample_rate = int(self.model.config.sampling_rate)
        except Exception:
            self.sample_rate = 44100

    def _description(self, request):
        """The description this request asks to be spoken with.

        The tuning dialog wins (it is an explicit instruction), then the voice
        entry's own description, and only then a neutral sentence that names
        the voice - so two different speakers never come out identical and a
        cloned entry (which has no description) still speaks.
        """
        options = request.get("options") or {}
        override = str(options.get("description") or "").strip()
        if override:
            return override
        described = str(request.get("description") or "").strip()
        if described:
            return described
        voice = str(request.get("voice") or "").strip()
        if voice and os.sep not in voice and "/" not in voice \
                and not voice.startswith("__"):
            return f"{voice} speaks naturally. {self._FALLBACK}"
        return self._FALLBACK

    def synthesize(self, text, request):
        description = self._description(request)
        with _quiet():
            description_ids = self.description_tokenizer(
                description, return_tensors="pt"
            ).to(self.device)
            prompt_ids = self.tokenizer(text, return_tensors="pt").to(self.device)
            audio = self.model.generate(
                input_ids=description_ids.input_ids,
                attention_mask=description_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
                **self.option_kwargs(self.model.generate,
                                     consume=("description",)),
            )
        return audio, self.sample_rate, False


# ---------------------------------------------------------------------------
# Kitten TTS 2.0 (KittenML)
# ---------------------------------------------------------------------------
#: The one checkpoint this engine speaks with.  Its reference clips, decoder and
#: packaging all live in the repository, so there is nothing to fetch by hand.
_KITTEN2_REPO = "KittenML/kitten-tts-2"

#: KittenTTS 2's 47 built-in voices, as the package's own ``available_voices``
#: lists them: thirty-eight English ones - the eight KittenTTS 0.8 already had
#: plus thirty persona voices - and nine named after a language.  Declared here
#: as well as in ``engines.py`` (which builds the catalogue from its own copy of
#: the list) so the worker can answer "is this voice usable?" without loading a
#: 1.7B model, and ``tests/test_voicelab.py`` fails when the two drift apart.
_KITTEN2_ENGLISH_VOICES = (
    "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
    "Herbert", "Diana", "Laurence", "Maeve", "Walter", "Edith", "Miles",
    "Grace", "Matthew", "Reginald", "Elliot", "Iris", "Willow", "Frank",
    "Dolores", "Serena", "Victor", "Julian", "Dante", "Eleanor", "Alfred",
    "Otis", "Saoirse", "Vincent", "Claire", "Martha", "Raven", "Sable",
    "Marcus", "Victoria",
)

#: The nine language voices.  The clip *is* the language: it is what supplies
#: the accent and phonotactics, so asking an English voice for German text does
#: not work, and ``normalize=False`` is what keeps the English-tuned text
#: normalizer from mangling numbers and dates in every other language.
_KITTEN2_LANGUAGE_VOICES = (
    "Arabic", "Chinese", "French", "German", "Hindi", "Italian",
    "Portuguese", "Russian", "Spanish",
)

_KITTEN2_VOICES = _KITTEN2_ENGLISH_VOICES + _KITTEN2_LANGUAGE_VOICES


class KittenTts2Backend(Backend):
    """KittenTTS 2 (``kittenml``): a 1.7B speech language model.

    The model reads text and writes S3 codec tokens, which its own vocoder
    renders at 24 kHz.  A voice is either one of the package's forty-seven
    built-in reference clips or a recording of your own (``reference=``), and
    the clip is *both* the voice and its transcript - the package transcribes
    one it is not given.

    Both devices are the engine's own: its documentation says it runs on a CPU
    out of the box (about 6 GB of RAM) and on a CUDA card, with ``device``
    passed explicitly here so a run never lands on the wrong one.  A CPU run is
    given the machine's cores before the model loads, and generation runs
    without an autograd graph; the PyTorch path is the one upstream calls
    demo-only, so the card remains the fast device.

    The model is loaded on first use rather than when the worker starts, so
    answering "which voices are usable?" never costs a 1.7B load.
    """

    name = "kitten_tts_2"

    #: Handled here rather than forwarded: ``weights`` and ``decoder`` pick
    #: which files the model is loaded from (they are part of the backend cache
    #: key for that reason), and ``normalize`` is decided per voice (see
    #: ``synthesize``).
    own_options = ("weights", "decoder", "normalize")

    def __init__(self, device, request=None):
        super().__init__(device, request)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = None

    def _options(self):
        raw = self.request.get("options") or {}
        return raw if isinstance(raw, dict) else {}

    def _configure_cpu_threads(self):
        """Give a CPU run the machine's cores before the model is loaded.

        The application sends the share of the CPU a run may use with every
        request (``ai_voice_studio.compute.cpu_threads``), and applies it
        before the backend is built.  A request that carries no figure - a
        probe from an older build, a hand-written one - would otherwise leave
        PyTorch on its own default, so every core but one is used instead: a
        CPU run of this engine is never single-threaded by accident.
        """
        try:
            import torch
        except Exception:
            return
        try:
            value = int(self.request.get("threads") or 0)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            try:
                value = max(1, int(os.cpu_count() or 1) - 1)
            except Exception:
                value = 1
        _limit_cpu(torch, value)

    def _load(self):
        """Load the selected weights once per worker process."""
        if self._model is not None:
            return self._model
        on_card = str(self.device).lower() == "cuda"
        if on_card:
            _cuda_or_error("Kitten TTS 2.0", "kittenml")
        else:
            # Threads must be set before the first forward pass, not after.
            self._configure_cpu_threads()
        try:
            from kittenml import KittenTTS
        except Exception as exc:
            raise _load_error("kittenml", "Kitten TTS 2.0", exc) from exc
        options = self._options()
        kwargs = {"device": "cuda" if on_card else "cpu"}
        weights = str(options.get("weights") or "").strip()
        if weights:
            kwargs["weights"] = weights
        decoder = str(options.get("decoder") or "").strip()
        if decoder and decoder != "default":
            kwargs["decoder"] = decoder
        with _quiet():
            model = KittenTTS(_KITTEN2_REPO, **kwargs)
        try:
            self.sample_rate = int(
                getattr(model, "sample_rate", 0) or DEFAULT_SAMPLE_RATE
            )
        except (TypeError, ValueError):
            self.sample_rate = DEFAULT_SAMPLE_RATE
        self._model = model
        return self._model

    def available_voices(self):
        """The forty-seven built-in voices, answered without loading a model."""
        return dict.fromkeys(_KITTEN2_VOICES, True)

    def synthesize(self, text, request):
        model = self._load()
        kwargs = {}
        # A clone arrives as the file the Voice Lab stored the recording as (or,
        # for a voice defined by a resource, as ``package:path``).
        reference = (request.get("ref_audio") or "").strip() or _resolve_resource(
            request.get("ref_resource") or ""
        ) or ""
        if reference:
            if not os.path.isfile(reference):
                raise FileNotFoundError(
                    f"The reference recording was not found: {reference}"
                )
            kwargs["reference"] = reference
            transcript = (request.get("ref_text") or "").strip() \
                or _read_text_resource(request.get("ref_text_resource") or "")
            # The transcript is part of the prompt; without one the package
            # transcribes the clip itself, which costs an extra Whisper pass.
            if transcript:
                kwargs["reference_text"] = transcript
        else:
            voice = (request.get("voice") or "").strip()
            # A built-in voice is a name, never a path and never the clone
            # marker: anything else lets the model speak with its own default.
            if voice and voice != "__clone__" \
                    and os.sep not in voice and "/" not in voice:
                kwargs["voice"] = voice
        # The normalizer is English-tuned, so a non-English voice turns it off
        # unless the user asked for it in Engine tuning; the language rides on
        # the voice, so the request's own language is what decides.
        normalize = self._options().get("normalize")
        if normalize is None:
            normalize = _short_language(request.get("language")) in ("", "en")
        kwargs["normalize"] = bool(normalize)
        self.report.mark("normalize")
        kwargs.update(self.option_kwargs(model.generate))
        # Reproducibility the way the engine documents it: ``seed`` is one of
        # its ``advanced`` options (an unknown top-level keyword raises), and
        # the worker's global Torch seeding alone does not reach its sampler.
        seed = self._options().get("seed")
        if seed is not None:
            params = _signature(model.generate)
            if params is not None and "advanced" in params.parameters:
                advanced = dict(kwargs.get("advanced") or {})
                advanced["seed"] = int(seed)
                kwargs["advanced"] = advanced
                self.report.mark("seed")
        # No autograd graph: this is a forward-only pass, and on the CPU the
        # bookkeeping is a real cost per token (see ``_no_grad``).
        with _quiet(), _no_grad():
            audio = model.generate(text, **kwargs)
        return audio, self.sample_rate, False


_BACKENDS = {
    "vibevoice": VibeVoiceBackend,
    "pocket_tts": PocketBackend,
    "f5tts": F5TtsBackend,
    "kokoclone": KokoCloneBackend,
    "chatterbox": ChatterboxBackend,
    "qwen3tts": QwenTtsBackend,
    "moss_tts": MossTtsBackend,
    "gptsovits": GptSovitsBackend,
    "indic_parler": IndicParlerBackend,
    "kitten_tts_2": KittenTts2Backend,
}

#: Import the engine needs before it can speak.
_ENGINE_MODULES = {
    "pocket_tts": ("pocket_tts",),
    "f5tts": ("f5_tts",),
    "kokoclone": ("core", "kanade_tokenizer"),
    "chatterbox": ("chatterbox", "torch", "torchaudio"),
    "qwen3tts": ("qwen_tts", "torch", "torchaudio"),
    "vibevoice": ("vibevoice", "torch", "transformers"),
    "moss_tts": ("transformers", "torch"),
    "gptsovits": (
        "torch", "torchaudio", "librosa", "ffmpeg", "pytorch_lightning",
    ),
    "indic_parler": ("parler_tts", "transformers", "torch", "torchaudio"),
    "kitten_tts_2": ("kittenml", "torch"),
}


def _engine_available(engine_id):
    return all(_module_available(name) for name in _ENGINE_MODULES.get(engine_id, ()))


def _backend_key(engine_id, device, request):
    """Cache key of a loaded backend: engine, device and the loaded model.

    Two requests that need *different* model weights (an int8-quantized Pocket
    TTS vs the full-precision one, or two variants of a transformer engine)
    must not share a backend, or the second one would keep speaking with the
    first one's weights.
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
    if engine_id in ("chatterbox", "qwen3tts", "vibevoice", "moss_tts",
                     "gptsovits", "indic_parler"):
        # Every variant is a *different model*: two variants must never share a
        # loaded backend, or the second one would speak with the first one's
        # weights.
        return (engine_id, device, str(request.get("variant") or ""))
    if engine_id == "kitten_tts_2":
        # One model, but two ways to load it: the weight packing and the audio
        # decoder decide *which files* are read, so switching either one has to
        # load again instead of keeping the first selection's weights.
        return (
            engine_id, device,
            str(options.get("weights") or ""),
            str(options.get("decoder") or ""),
        )
    return (engine_id, device)


def _request_texts(request):
    """The texts one synthesize request asks for.

    A single ``text`` (the common case: one segment, one preview) or a list
    ``texts`` - the application's way of giving one card several segments at
    once.  An empty or blank text is refused here, before an engine is asked
    to speak it.
    """
    texts = request.get("texts")
    if texts is None:
        texts = [request.get("text", "")]
    elif isinstance(texts, str):
        texts = [texts]
    else:
        texts = [str(item) for item in texts]
    if not texts:
        raise ValueError("Nothing to synthesize.")
    for text in texts:
        if not str(text).strip():
            raise ValueError("Nothing to synthesize.")
    return texts


def _cuda_peak_mb(reset=True):
    """The card's peak use for the work just done, in MB (0 without a card).

    Reported back with every synthesize response so the application (and the
    log) can say what a run actually took: the figure a "is my whole card
    being used?" question is answered with, and the one the next run's plan is
    judged against.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        peak = int(torch.cuda.max_memory_reserved() // (1024 * 1024))
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return peak
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Request loop
# ---------------------------------------------------------------------------
def main():
    out = _protocol_stream()
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
            except Exception as exc:
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        if cmd == "benchmark":
            # A quick speed measurement: the first call loads the model (and
            # may download its weights), the second one is timed.  The figure
            # is therefore the engine's steady-state speed, never the
            # download, and the application stores it per engine and device.
            engine_id = req.get("engine", "")
            device = (
                "cuda" if str(req.get("device", "cpu")).lower() == "cuda" else "cpu"
            )
            _apply_compute_settings(device, req.get("threads"))
            text = str(req.get("text") or _BENCHMARK_TEXT)
            try:
                backend = backend_for(engine_id, device, req)
                _request_texts({"text": text})
                warmup = dict(req, text=_BENCHMARK_WARMUP_TEXT)
                backend.synthesize(_BENCHMARK_WARMUP_TEXT, warmup)
                started = time.monotonic()
                audio, sample_rate, _speed_handled = backend.synthesize(
                    text, dict(req, text=text)
                )
                seconds = time.monotonic() - started
                samples = _to_int16(audio)
                rate = int(sample_rate or DEFAULT_SAMPLE_RATE)
                audio_seconds = (len(samples) / float(rate)) if rate else 0.0
                respond({
                    "ok": True,
                    "engine": engine_id,
                    "device_used": getattr(backend, "device", device),
                    "text": text,
                    "seconds": seconds,
                    "audio_seconds": audio_seconds,
                    "sample_rate": rate,
                    # Seconds of generation per second of audio: 1.0 is
                    # exactly real time, 0.3 is three times faster.
                    "rtf": (seconds / audio_seconds) if audio_seconds > 0 else None,
                    "vram_peak_mb": _cuda_peak_mb(),
                    "notes": list(getattr(backend, "notes", []) or []),
                })
            except Exception as exc:
                respond({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
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
                texts = _request_texts(req)
                backend = backend_for(engine_id, device, req)
                backend.report.reset()
                options = req.get("options") or {}
                if not isinstance(options, dict):
                    options = {}
                if _apply_seed(options.get("seed")):
                    backend.report.mark("seed")
                speed = float(req.get("speed", 1.0) or 1.0)
                produced = []
                for audio, sample_rate, speed_handled in backend.synthesize_many(
                    texts, req
                ):
                    samples = _to_int16(audio)
                    if samples.size == 0:
                        raise ValueError("The engine returned no audio.")
                    if not speed_handled:
                        samples = _resample_speed(samples, speed)
                    produced.append((
                        base64.b64encode(samples.tobytes()).decode("ascii"),
                        int(sample_rate or DEFAULT_SAMPLE_RATE),
                        (not speed_handled) and abs(speed - 1.0) > 1e-3,
                    ))
                report = backend.report.as_dict()
                respond({
                    "ok": True,
                    "wav": produced[0][0],
                    "wavs": [item[0] for item in produced],
                    "sample_rate": produced[0][1],
                    "device_used": getattr(backend, "device", device),
                    "speed_resampled": produced[0][2],
                    "applied": report["applied"],
                    "skipped": report["skipped"],
                    "notes": list(getattr(backend, "notes", []) or []),
                    "vram_peak_mb": _cuda_peak_mb(),
                })
            except Exception as exc:
                respond({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue

        respond({"ok": False, "error": f"Unknown command: {cmd}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
