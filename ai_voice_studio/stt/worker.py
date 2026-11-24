"""Standalone speech-to-text worker.

This script is run by the interpreter of **one model's own virtualenv** - never
by the application - so the model's heavy dependencies (PyTorch, NeMo,
transformers, ...) live and load only in that isolated environment.  It is
deliberately self-contained: a model environment cannot import the application
package, so nothing here uses a relative import.

Usage::

    <env>/python.exe worker.py <request.json>

``request.json``::

    {
      "audio":  "C:/.../lecture 1.wav",
      "backend": "faster_whisper" | "whisper" | "transformers" | "nemo"
                 | "speechbrain",
      "hf_model": "Qwen/Qwen3-ASR-1.7B-hf",   # repository-backed back-ends
      "variant": "large-v3",
      "language": "auto",
      "translate_to": "",
      "accuracy": "balanced",
      "compute": "cpu" | "cuda" | "auto",
      "timestamps": false,      # only honoured where the model really has them
      "trust_remote_code": false,
      "out": "C:/.../response.json"
    }

The worker writes ``{"ok": true, "segments": [...]}`` to ``out`` (each segment
``{"start", "end", "text"}``) or ``{"ok": false, "error": "..."}`` and exits
non-zero on failure.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile

BEAM_SIZES = {"fast": 1, "balanced": 5, "best": 8}

#: Every automatic-speech-recognition model here is trained on 16 kHz mono
#: audio.  Decoding to that rate ourselves keeps the transformers pipeline from
#: reaching for ``torchaudio`` to resample (it refuses to work without it).
SAMPLE_RATE = 16000

#: The peak level a recording is scaled to before a model sees it.
#:
#: Every model here was trained on speech at a healthy level, and the two
#: things that decide what gets transcribed are level-sensitive by
#: construction: Silero VAD, which faster-whisper's ``vad_filter`` runs, and
#: Whisper's own ``no_speech_threshold``, which skips a whole 30-second window
#: whose ``no_speech_prob`` crosses 0.6 and drops its words while the rest of
#: the transcript carries on.  The recordings in one folder differed by 7 dB -
#: the one being transcribed sat at -7 dBFS peak while its neighbours sat at
#: 0 dBFS - which is a spread neither detector should have to cope with.
#: Scaling a recording to a healthy peak takes the level out of the equation,
#: so a quiet recording is judged on its content and not on how far a
#: recorder's gain was turned down.
TARGET_PEAK = 0.95

#: How many tokens one second of speech may need, in the worst case.  One
#: token per *word* is the best any language manages, and only English gets
#: near it: a tokenizer splits Devanagari at about one token per character, so
#: Hindi needs several times the English budget for the same recording.  The
#: old 8-per-second rate allowed a 54-second Hindi clip 435 tokens, and its
#: transcript stopped mid-sentence on token 432 - the cap, not the audio, was
#: what ended it.  The generation always stops at the model's own
#: end-of-sequence token, so this is a runaway guard rather than a target and
#: is deliberately several times any observed real transcript.
TOKENS_PER_SECOND = 32

#: Floor for a very short clip (and for one whose length cannot be measured).
MIN_NEW_TOKENS = 4096

#: Ceiling that keeps the guard finite for a very long recording.
MAX_NEW_TOKENS = 32768

#: The longest stretch of a recording one call to a decoder-only speech model
#: may be handed.
#:
#: Qwen3-ASR and Voxtral are given the whole recording in a single request and
#: attend over all of it: a 29-minute Hindi chapter arrived as one 22,969-token
#: sequence, and self-attention over that - the sequence squared, times the
#: head count, times four bytes - asked for a **31.5 GiB** tensor while the
#: card holds 12, so the run died with "CUDA out of memory ... Tried to
#: allocate 31.45 GiB" before a single word was written.  The cost grows with
#: the *square* of the length, so a few minutes at a time brings that tensor
#: down to a few hundred megabytes on any card - and because each call still
#: stops at the model's own end-of-sequence token, every piece comes back as
#: finished text rather than as a fragment.
AUDIO_CHUNK_SECONDS = 180.0

#: The shortest piece an out-of-memory run will cut down to.  Past this the
#: problem is the model or the card rather than the length of the piece, so
#: the failure is reported - with what to do about it - instead of cutting
#: forever.
MIN_AUDIO_CHUNK_SECONDS = 30.0

#: What a run that ran out of memory even at the shortest piece says.
#:
#: PyTorch's own message explains its allocator, which is not a choice the
#: user has; the three things that actually change the outcome are named.
OUT_OF_MEMORY_HELP = (
    "The model ran out of memory even with the recording cut into "
    "{seconds}-second pieces. Close other programs that are using the "
    "graphics card, choose a smaller model variant, or set Compute to CPU "
    "for this transcription."
)

#: Handles returned by ``os.add_dll_directory``.  Windows keeps the directory on
#: the process's DLL search path only while the handle is alive, so they must
#: outlive the call that added them.
_DLL_HANDLES: list = []

#: The instruction an audio chat model is given.  Granite Speech documents
#: exactly this prompt, and it is what any audio-in / text-out model expects
#: for a plain transcription.
TRANSCRIBE_PROMPT = "<|audio|>can you transcribe the speech into a written format?"

#: How a transformers-backed model is driven.  The generic ASR pipeline cannot
#: run every architecture: the audio language models have their own request
#: API, Granite Speech's feature extractor rejects the pipeline's
#: ``sampling_rate`` argument, and asking either of them for timestamps raises
#: "We cannot return_timestamps yet on non-CTC models apart from Whisper!".
RUNNER_PIPELINE = "pipeline"
RUNNER_TRANSCRIPTION_REQUEST = "transcription_request"
RUNNER_CHAT = "chat"


def _patch_pyav_open() -> None:
    """Let faster-whisper decode audio on PyAV releases that removed
    ``metadata_errors``.

    PyAV 19 dropped the ``metadata_errors`` (and ``metadata_encoding``)
    arguments to :func:`av.open`, but faster-whisper still calls
    ``av.open(..., metadata_errors="ignore")`` in its audio decoder, so every
    Whisper transcription failed with::

        TypeError: open() got an unexpected keyword argument 'metadata_errors'

    We wrap ``av.open`` so those arguments are honoured on the PyAV builds that
    still understand them and quietly dropped on the ones that do not, which
    keeps every Whisper model working across PyAV versions.  Idempotent, so it
    is safe to call more than once.
    """
    try:
        import av
    except Exception:
        return
    original = getattr(av, "open", None)
    if original is None or getattr(original, "_aivs_pyav_compat", False):
        return
    removed = ("metadata_errors", "metadata_encoding")

    def open_compat(*args, **kwargs):
        if any(key in kwargs for key in removed):
            try:
                return original(*args, **kwargs)
            except TypeError as exc:
                if not any(key in str(exc) for key in removed):
                    raise
                kwargs = {k: v for k, v in kwargs.items() if k not in removed}
        return original(*args, **kwargs)

    open_compat._aivs_pyav_compat = True
    open_compat.__wrapped__ = original  # type: ignore[attr-defined]
    av.open = open_compat


# ---------------------------------------------------------------------------
# Native libraries
# ---------------------------------------------------------------------------
def _library_directories() -> list:
    """Folders that may hold the CUDA libraries the back-ends load at runtime.

    CTranslate2 (and so faster-whisper) is built with CUDA support on Windows
    and loads ``cublas64_12.dll`` the moment it is asked for a CUDA device.
    The wheel ships ``cudnn64_9.dll`` itself but *not* cuBLAS, so unless the
    CUDA runtime is on the DLL search path every Whisper run failed with::

        RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

    The same libraries are also bundled inside the CUDA build of PyTorch
    (``torch/lib``) and, when installed from PyPI, under ``nvidia/<name>/bin``;
    both are searched here so a model can use CUDA wherever the DLLs already
    are, without a CUDA toolkit install.
    """
    roots = set()
    try:
        import sysconfig

        for key in ("purelib", "platlib"):
            try:
                roots.add(sysconfig.get_path(key))
            except Exception:  # noqa: S112 - one missing root is not an error
                continue
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _library_directories: {exc}\n")
    directories: list = []
    for root in sorted(filter(None, roots)):
        directories.append(os.path.join(root, "ctranslate2"))
        directories.append(os.path.join(root, "torch", "lib"))
        vendor = os.path.join(root, "nvidia")
        try:
            names = sorted(os.listdir(vendor))
        except OSError:
            names = []
        for name in names:
            directories.append(os.path.join(vendor, name, "bin"))
    return [path for path in directories if os.path.isdir(path)]


def _prepare_native_libraries() -> list:
    """Put every discovered CUDA library folder on the DLL search path.

    Called once before any back-end is imported, because ``ctranslate2`` loads
    its CUDA libraries at import time.  Idempotent.
    """
    added: list = []
    for path in _library_directories():
        if path in added:
            continue
        try:
            _DLL_HANDLES.append(os.add_dll_directory(path))
        except (AttributeError, OSError) as exc:
            sys.stderr.write(f"Ignoring a failure in _prepare_native_libraries: {exc}\n")
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        added.append(path)
    return added


def _cuda_libraries_present() -> bool:
    """True when cuBLAS - the library CTranslate2 cannot start without - is
    somewhere the loader will find it."""
    return any(
        os.path.isfile(os.path.join(path, "cublas64_12.dll"))
        for path in _library_directories()
    )


def _resolve_device(compute: str) -> str:
    if compute in ("cuda", "gpu"):
        return "cuda"
    if compute == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------
def _resample(samples, source_rate: int, target_rate: int):
    """Resample mono float samples to ``target_rate``.

    ``soxr`` is used when the model's environment has it (it is the same
    resampler ``librosa`` uses); otherwise a plain linear interpolation is
    enough for speech recognition and needs nothing but NumPy.
    """
    try:
        import numpy as np
    except Exception:
        return samples
    samples = np.asarray(samples, dtype="float32")
    if source_rate == target_rate or samples.size == 0 or source_rate <= 0:
        return samples
    try:
        import soxr

        return np.asarray(soxr.resample(samples, source_rate, target_rate),
                          dtype="float32")
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _resample: {exc}\n")
    count = round(samples.size * target_rate / source_rate)
    if count <= 0:
        return samples[:0]
    positions = np.linspace(0.0, samples.size - 1, count)
    return np.interp(positions, np.arange(samples.size), samples).astype("float32")


def _normalise_level(samples):
    """Scale mono float samples so the loudest point sits near full scale.

    Only the level is touched - no filtering, no compression - which is what a
    quietly recorded file needs to be judged on its content rather than on how
    far the recorder's gain was turned down.  Silence is left alone.
    """
    try:
        import numpy as np
    except Exception:
        return samples
    samples = np.asarray(samples, dtype="float32")
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if not peak > 0.0:
        return samples
    return np.clip(samples * (TARGET_PEAK / peak), -1.0, 1.0)


def _decode_with_av(path):
    """Decode any container PyAV understands (video files included)."""
    import numpy as np

    try:
        import av
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the audio in {os.path.basename(path)}: no audio "
            "decoder (SoundFile, librosa or PyAV) is installed for this model."
        ) from exc
    with av.open(path) as container:
        stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if stream is None:
            raise RuntimeError(f"{os.path.basename(path)} has no audio track.")
        rate = int(getattr(stream, "rate", 0) or SAMPLE_RATE)
        chunks = []
        for frame in container.decode(stream):
            data = frame.to_ndarray()
            data = np.asarray(data, dtype="float32")
            if data.ndim > 1:
                # PyAV returns planar frames as (channels, samples) and packed
                # ones as (1, samples * channels); average whichever axis holds
                # the channels.
                data = data.mean(axis=0 if data.shape[0] <= 8 else 1)
            chunks.append(data.reshape(-1))
    if not chunks:
        return np.zeros(0, dtype="float32"), rate
    return np.concatenate(chunks), rate


def _decode_audio(path, target_rate: int = SAMPLE_RATE):
    """Mono float32 samples at ``target_rate``, plus the rate they are at.

    SoundFile reads WAV/FLAC/OGG/MP3 directly; librosa and PyAV cover the rest
    (and video containers).  Resampling here is what keeps the transformers
    pipeline from demanding ``torchaudio``.
    """
    import numpy as np

    data = None
    rate = 0
    try:
        import soundfile as sf

        data, rate = sf.read(path, always_2d=False)
    except Exception:
        data = None
    if data is None:
        try:
            import librosa

            data, rate = librosa.load(path, sr=None, mono=True)
        except Exception:
            data, rate = _decode_with_av(path)
    samples = np.asarray(data, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    try:
        source_rate = int(rate)
    except (TypeError, ValueError):
        source_rate = target_rate
    samples = _resample(samples, source_rate, target_rate)
    return _normalise_level(samples), target_rate


def _duration_with_av(path) -> float:
    """Length of a container SoundFile cannot open (an ``.mp4``, say).

    The audio language models are told how many tokens their transcript may
    hold from the recording's length, so a video whose length read as 0 in the
    past was given only the floor - a truncation waiting to happen.  A
    container knows its own duration without decoding a frame.
    """
    try:
        import av

        with av.open(path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if stream is not None and stream.duration:
                return float(stream.duration * (stream.time_base or 0))
    except Exception:
        return 0.0
    return 0.0


def _audio_duration(path) -> float:
    """Length of a recording in seconds, or 0 when it cannot be read."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return _duration_with_av(path)


def _token_budget(seconds: float) -> int:
    """How long a transcript of ``seconds`` of audio may become."""
    if seconds <= 0:
        return MAX_NEW_TOKENS
    wanted = int(seconds * TOKENS_PER_SECOND)
    return max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS, wanted))


def _max_new_tokens(path) -> int:
    """How long a generated transcript may become, from the audio's length.

    A decoder-only speech model writes its transcript one token at a time and
    stops at its own end-of-sequence token, so this bound exists only to stop
    a generation that has run away - it must never be what ends a complete
    transcript.  It therefore scales with the recording at a rate several
    times what even a finely tokenised script needs, and a recording whose
    length cannot be measured gets the ceiling rather than the floor.
    """
    return _token_budget(_audio_duration(path))


def _normalise(segments):
    out = []
    for seg in segments:
        if isinstance(seg, dict):
            text = str(seg.get("text", "")).strip()
            start = seg.get("start")
            end = seg.get("end")
        else:  # pragma: no cover - object-style results
            text = str(getattr(seg, "text", "")).strip()
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
        if text:
            out.append({"start": start, "end": end, "text": text})
    return out


def _truncation_note(new_tokens, budget: int) -> str:
    """A warning when generation stopped on the cap rather than on its own.

    A transcript that ends because the model ran out of budget is written to
    the project looking exactly like a finished one, which is how a
    half-transcribed recording reached the user unnoticed.  Saying so on the
    status line turns a silent truncation into one that can be acted on.
    """
    try:
        emitted = int(new_tokens.shape[-1])
    except (AttributeError, IndexError, TypeError):  # a plain list of tokens
        try:
            emitted = len(new_tokens)
        except TypeError:
            return ""
    if emitted < budget:
        return ""
    return (
        f"The transcript reached the {budget}-token limit set for this "
        "recording and may be incomplete."
    )


# ---------------------------------------------------------------------------
# Long recordings: pieces, and what to do when a piece will not fit
# ---------------------------------------------------------------------------
def _split_audio(samples, rate, seconds: float = AUDIO_CHUNK_SECONDS):
    """``samples`` cut into pieces of at most ``seconds`` each.

    The pieces share the recording equally instead of filling each one and
    leaving a stub behind: cutting 400 seconds into 180-second pieces would be
    180, 180 and a 40-second tail - a whole model call for a tail - whereas an
    equal split gives three 134-second ones.  A recording that already fits is
    returned untouched, as the one item of a one-item list.
    """
    frames = max(1, int(seconds * rate))
    total = len(samples)
    if total <= frames:
        return [samples]
    count = (total + frames - 1) // frames
    step = (total + count - 1) // count
    return [samples[start:start + step] for start in range(0, total, step)]


def _write_wav(path: str, samples, rate: int) -> None:
    """Write one piece of audio as a 16-bit mono WAV file.

    Only ``wave`` and NumPy are used, because this runs inside a model's own
    environment, where nothing may be installed that the model itself did not
    ask for.  The file carries its own sample rate, so a processor reading it
    back needs no rate argument of its own.
    """
    import wave

    import numpy as np

    data = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = np.round(data * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm.tobytes())


def _recording_pieces(path, tmpdir) -> list:
    """The recording as one path per piece of at most ``AUDIO_CHUNK_SECONDS``.

    A recording that fits - the ordinary case - is handed back unchanged: the
    processor reads the original file exactly as it always has, so nothing is
    decoded or rewritten for a short clip.  A longer one is decoded once and
    written out as equal pieces of 16 kHz mono WAV under ``tmpdir``, and each
    piece is given to the processor as a *path*, which is what a processor is
    always given here.
    """
    duration = _audio_duration(path)
    if duration <= 0 or duration <= AUDIO_CHUNK_SECONDS:
        return [path]
    samples, rate = _decode_audio(path)
    if len(samples) == 0:
        return [path]
    pieces = []
    for index, piece in enumerate(_split_audio(samples, rate)):
        piece_path = os.path.join(tmpdir, f"piece{index:03d}.wav")
        _write_wav(piece_path, piece, rate)
        pieces.append(piece_path)
    return pieces


def _halve_wav(path, tmpdir, label):
    """Two half-length copies of ``path``, or ``None`` when it is too short.

    This is what a run that ran out of memory falls back on: the piece that
    would not fit is cut in half and each half transcribed on its own, so the
    memory failure costs time rather than transcript.
    """
    samples, rate = _decode_audio(path)
    if len(samples) == 0 \
            or len(samples) / float(rate) <= MIN_AUDIO_CHUNK_SECONDS:
        return None
    middle = len(samples) // 2
    first = os.path.join(tmpdir, f"{label}a.wav")
    second = os.path.join(tmpdir, f"{label}b.wav")
    _write_wav(first, samples[:middle], rate)
    _write_wav(second, samples[middle:], rate)
    return first, second


def _is_out_of_memory(exc) -> bool:
    """True for the allocator failures a piece that is too long produces.

    CUDA, the CPU allocator and CTranslate2 word it differently, so the
    message is matched rather than the exception type - the worker runs in a
    model's own environment, whose PyTorch is not one the application imports.
    """
    text = str(exc).lower()
    return ("out of memory" in text
            or "can't allocate memory" in text
            or "cannot allocate memory" in text
            or "not enough memory" in text)


def _release_device_memory(device) -> None:
    """Hand back what the failed attempt reserved, so the halves may use it."""
    if device != "cuda":
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _release_device_memory: {exc}\n")


def _transcribe_in_halves(piece, transcribe, halve, device):
    """``transcribe(piece)`` -> ``(text, note)``, cut in half when it will not fit.

    ``transcribe`` turns one piece into its text and note; ``halve`` returns
    two smaller pieces, or ``None`` when the piece is already the smallest
    worth trying - at which point the memory failure is reported as something
    the user can act on (see :data:`OUT_OF_MEMORY_HELP`) instead of as an
    allocator dump.  Both halves are transcribed and their text joined in
    order, so a memory failure costs time, never transcript.
    """
    try:
        return transcribe(piece)
    except Exception as exc:
        if not _is_out_of_memory(exc):
            raise
        _release_device_memory(device)
        halves = halve(piece)
        if halves is None:
            raise RuntimeError(
                OUT_OF_MEMORY_HELP.format(seconds=int(MIN_AUDIO_CHUNK_SECONDS))
            ) from exc
        first = _transcribe_in_halves(halves[0], transcribe, halve, device)
        second = _transcribe_in_halves(halves[1], transcribe, halve, device)
        return _join_results(first, second)


def _join_results(left, right):
    """Two ``(text, note)`` results as one, in order."""
    text = " ".join(part for part in (left[0], right[0]) if part)
    note = " ".join(part for part in (left[1], right[1]) if part)
    return text, note


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
#: How much of a recording the Whisper models are handed at one time.
#:
#: faster-whisper decodes in fixed 30-second windows and its decoder emits at
#: most 448 tokens for each one.  In English that ceiling never binds - 30
#: seconds of English is nowhere near it - but Devanagari costs a tokenizer
#: about one token per character, and 30 seconds of Hindi needs more than a
#: window can hold.  The model then reaches the window edge mid-sentence, emits
#: its end-of-sequence token, and faster-whisper advances a whole window, so
#: everything the model had not written yet is never decoded at all: a
#: 54-second Hindi recording came back as 100 words where the audio holds 196,
#: with the loss sitting exactly on the 30-second boundary ("the first
#: paragraph stops and it jumps to another").
#:
#: Decoding the recording in pieces and putting the timings back together
#: recovers the whole transcript - measured, that recording went from 100 words
#: to 199 against a reference of 196 - and English is unharmed: 90 seconds of
#: English gave 319 words as one call against 324 in pieces, with the
#: punctuation and casing the model drops when one window has to cover too much
#: of it.
#:
#: How big those pieces may be is a budget, not a number.  One 30-second window
#: can hold :data:`WHISPER_TOKEN_CEILING` tokens, how many of them a second of
#: speech costs depends on the language, and a piece must be small enough that
#: its tokens fit with room to spare.  One fixed size therefore has to be the
#: size that is safe for the *most expensive* language, which hands an English
#: lecture four times as many pieces as it needs.  The pieces are sized from the
#: language instead: a dense script gets pieces that were measured to come back
#: complete, and a cheap one gets pieces long enough that the token ceiling
#: never binds.
#:
#: The window's budget is spent on twice what it looks like.  The decoder emits
#: a timestamp with almost every word as well as the word itself - and word-level
#: alignment is what keeps these pieces complete - so a second of Devanagari
#: speech costs around 24 tokens, not the 15 it spends on its written words.
#: Sizing from the written words alone is how a 29-minute Hindi chapter still
#: came back with about a tenth of its words missing: its 19.6-second pieces ran
#: out of window partway through, and the tail of each one was never decoded.
#: Measured against a 1.7B audio language model on two windows of that chapter,
#: shortening the pieces to 13 seconds recovered 70% -> 83% of the reference
#: words with no loss of accuracy, and going below 13 gained nothing.
WHISPER_TOKEN_CEILING = 448

#: The share of a window's token ceiling one piece is allowed to fill.  A piece
#: that decodes completely was measured to use about 0.7 of the ceiling, and
#: word-level alignment adds tokens of its own on top of the words written
#: down, so the budget stops short of the ceiling rather than reaching for it.
WHISPER_BUDGET_FRACTION = 0.7

#: The longest piece the model is ever handed.  Whisper's own window is 30
#: seconds; a piece has to be short enough that the window never has to cut it.
WHISPER_MAX_PIECE_SECONDS = 28.0

#: Tokens a second of speech costs, in the worst case measured: Devanagari
#: spends about 24 once the timestamp tokens that come with each word are
#: counted, and rounding up leaves the piece just inside the 13 seconds that
#: were measured to come back complete.  Used whenever the language is unknown,
#: because guessing "cheap" and being wrong loses the transcript.
WHISPER_DENSE_TOKENS_PER_SECOND = 24.0

#: Tokens a second of speech costs in the space-delimited majority of
#: languages, where a word is about a token.  This is generous on purpose: at
#: this rate the token ceiling stops binding altogether and pieces reach
#: :data:`WHISPER_MAX_PIECE_SECONDS`.
WHISPER_DEFAULT_TOKENS_PER_SECOND = 3.0

#: How far back from a piece's end the quietest moment is looked for.  A piece
#: that ends in the middle of a word loses the rest of that word - the model can
#: only write what the audio held - so the cut is moved onto a pause instead.
#: Three seconds is what was measured on the Hindi chapter: it leaves a piece
#: close to its budget while still finding the pause around every boundary.
WHISPER_CUT_SEARCH_SECONDS = 3.0

#: The stretch of audio the quiet search measures energy over.  20 ms is short
#: enough to sit inside a pause between words.
WHISPER_CUT_FRAME_SECONDS = 0.02

#: The languages whose writing system makes Whisper's tokenizer spend about a
#: token per character - or per syllable - rather than about a token per word:
#: the Brahmic and CJK scripts, none of which separate words with spaces.
#: Devanagari costs some 15.4 written tokens a second of speech, measured,
#: where English costs 3-5.  The list is deliberately generous, because naming a
#: language here only makes its pieces smaller (a little slower, never less
#: complete) while leaving one out risks the truncation these pieces exist to
#: prevent.
WHISPER_DENSE_LANGUAGES = frozenset({
    # Devanagari and its neighbours
    "hi", "mr", "ne", "sa", "bn", "as", "gu", "pa", "ta", "te", "kn",
    "ml", "si",
    # South-East Asian and Himalayan scripts
    "th", "lo", "km", "my", "bo",
    # CJK
    "zh", "yue", "ja", "ko",
})


def _whisper_tokens_per_second(language) -> float:
    """What a second of this language is expected to cost the decoder.

    ``None`` here means the language was left to the model, which is the one
    case where nothing is known about what a second of it costs: the cautious
    rate is used until the model has heard a piece and said which language it
    was (see :func:`_transcribe_with_whisper`).
    """
    if not language:
        return WHISPER_DENSE_TOKENS_PER_SECOND
    if language in WHISPER_DENSE_LANGUAGES:
        return WHISPER_DENSE_TOKENS_PER_SECOND
    return WHISPER_DEFAULT_TOKENS_PER_SECOND


def _whisper_piece_seconds(language) -> float:
    """How much of a recording one Whisper call may be handed.

    The piece has to fit inside both Whisper's 30-second window and that
    window's token ceiling, and only the second bound depends on the language.
    """
    rate = _whisper_tokens_per_second(language)
    fit = WHISPER_TOKEN_CEILING * WHISPER_BUDGET_FRACTION / rate
    return min(WHISPER_MAX_PIECE_SECONDS, fit)


def _whisper_piece_frames(language, rate: int) -> int:
    """The piece size above, in frames at the decoded sample rate."""
    return max(1, int(_whisper_piece_seconds(language) * rate))


def _whisper_piece_size(remaining: int, piece_frames: int) -> int:
    """Frames to hand the model next, without leaving a stub at the end.

    Cutting a 57-second recording into 28-second pieces leaves 28, 28 and then
    one second - a whole model call for a syllable.  When the tail of what is
    left would come out that small, the rest is split evenly instead: 28, 14.5
    and 14.5, the same number of calls and no wasted one.
    """
    if remaining <= piece_frames:
        return remaining
    if remaining - piece_frames < piece_frames // 2:
        return (remaining + 1) // 2
    return piece_frames


def _quietest_frame(samples, rate: int, start: int, end: int) -> int:
    """The last quietest 20 ms frame in ``samples[start:end]``, or -1.

    The frame that gave up the least energy is where the speech is quietest, so
    it is the safest place to stop a piece.  The *last* one wins: the search
    looks for a pause just before where the piece was going to end, and moving
    the cut as little as possible leaves the piece as long as it can be.

    -1 means there is nothing to move to: either the energy could not be
    measured, or the end of the stretch is already as quiet as its quietest
    moment - a boundary that sits in silence, or a recording with no sound in
    it at all - and the piece should end where it was going to end.
    """
    try:
        import numpy as np
    except Exception:
        return -1
    hop = max(1, int(WHISPER_CUT_FRAME_SECONDS * rate))
    if end - start < hop * 2:
        return -1
    energy = np.square(np.asarray(samples[start:end], dtype="float32"))
    usable = (energy.size // hop) * hop
    if not usable:
        return -1
    frames = energy[:usable].reshape(-1, hop).mean(axis=1)
    quietest = float(frames.min())
    if float(frames[-1]) <= quietest:
        return -1
    return start + (frames.size - 1 - int(frames[::-1].argmin())) * hop


def _whisper_cut_frame(samples, rate: int, start: int, end: int) -> int:
    """Where a piece may end so that it does not end in the middle of a word.

    The piece budget says how much speech the decoder can finish; it says
    nothing about *where* speech stops and silence begins.  Ending the piece a
    second or two early, on the quietest moment of the last
    :data:`WHISPER_CUT_SEARCH_SECONDS` seconds, costs almost no speech and stops
    a word from being cut in half - the model can only write the words it was
    given, so a piece that ends mid-word leaves a fragment behind and the next
    piece starts after the rest of it.
    """
    search = int(WHISPER_CUT_SEARCH_SECONDS * rate)
    limit = end - search
    if limit <= start:
        return end
    quiet = _quietest_frame(samples, rate, limit, end)
    return end if quiet <= 0 else quiet


def _whisper_options(req) -> dict:
    """The decode settings every Whisper piece is transcribed with."""
    options = {
        "beam_size": BEAM_SIZES.get(req.get("accuracy", "balanced"), 5),
        "language": (None if req.get("language") in ("", "auto", None)
                     else req.get("language")),
        "task": "translate" if req.get("translate_to") else "transcribe",
        "vad_filter": True,
    }
    if not req.get("translate_to"):
        # Word timings are not just for the subtitles: they are what makes the
        # decode complete.  They let faster-whisper carry on from the last word
        # it really wrote and align the words the model skipped, rather than
        # stopping where the window ran out.  Translation keeps the plain path,
        # where the words spoken and the tokens written are different languages.
        options["word_timestamps"] = True
        options["without_timestamps"] = False
    return options


def _transcribe_with_whisper(model, req):
    """Transcribe a recording in pieces the Whisper decoder can finish.

    The recording is decoded once (to the 16 kHz mono Whisper wants, at a
    healthy level) and each piece is handed to the model on its own; every
    piece's timings are then shifted back onto the original recording, so the
    segments come out as they always did - just complete.

    The piece size comes from the language (see :func:`_whisper_piece_seconds`),
    and a recording short enough to fit in one piece is not cut at all.  When
    the language was left to the model the first piece is sized cautiously and
    the rest are priced from the language the model reports it heard, so a
    detected English lecture gets the long pieces its language has earned.

    Each piece ends a little before its budget runs out, on the quietest moment
    of the last few seconds (:func:`_whisper_cut_frame`), so the cut falls in a
    pause rather than through a word.
    """
    samples, rate = _decode_audio(req["audio"])
    options = _whisper_options(req)
    total = len(samples)
    if total == 0:
        return []
    # ``None`` is Whisper's own "detect it", and the language it detects comes
    # back with the first piece.
    language = options.get("language")
    step = _whisper_piece_frames(language, rate)
    segments: list = []
    start = 0
    while start < total:
        frames = _whisper_piece_size(total - start, step)
        frames = _whisper_cut_frame(samples, rate, start, start + frames) - start
        piece = samples[start:start + frames]
        offset = start / float(rate)
        result, info = model.transcribe(piece, **options)
        for segment in _normalise(result):
            start_time = segment["start"]
            end_time = segment["end"]
            segments.append({
                "start": None if start_time is None else start_time + offset,
                "end": None if end_time is None else end_time + offset,
                "text": segment["text"],
            })
        if language is None:
            detected = getattr(info, "language", None)
            if detected:
                language = detected
                step = _whisper_piece_frames(language, rate)
        start += frames
    return segments


def _run_faster_whisper(req):
    _patch_pyav_open()
    _prepare_native_libraries()
    from faster_whisper import WhisperModel

    device = _resolve_device(req.get("compute", "cpu"))
    # CTranslate2 cannot even *construct* a CUDA model without the CUDA runtime
    # (see ``_library_directories``), and a card that is busy or too small fails
    # the same way.  Rather than leaving the user with an error dialog, fall
    # back to the CPU build, which always works - slower, but a transcript
    # beats no transcript.
    attempts = []
    note = ""
    if device == "cuda":
        if _cuda_libraries_present():
            attempts.append(("cuda", "float16"))
        else:
            note = (
                "The CUDA runtime this model needs is not installed in its "
                "environment, so the recording was transcribed on the CPU "
                "instead. Install or repair this model from Settings > "
                "Manage the STT to enable GPU transcription."
            )
    attempts.append(("cpu", "int8"))

    last_error = None
    for index, (target, compute_type) in enumerate(attempts):
        try:
            model = WhisperModel(req["variant"], device=target,
                                 compute_type=compute_type)
            segments = _transcribe_with_whisper(model, req)
        except Exception as exc:
            last_error = exc
            if index + 1 >= len(attempts):
                raise
            note = (
                f"GPU transcription was not available ({exc}); "
                "the recording was transcribed on the CPU instead."
            )
            continue
        return {"segments": segments, "device": target, "note": note}
    raise last_error  # pragma: no cover - the loop always returns or raises


def _run_openai_whisper(req):
    import whisper

    _prepare_native_libraries()
    device = _resolve_device(req.get("compute", "cpu"))
    note = ""
    try:
        model = whisper.load_model(req["variant"], device=device)
        result = model.transcribe(
            req["audio"],
            language=None if req.get("language") in ("", "auto") else req.get("language"),
            task="translate" if req.get("translate_to") else "transcribe",
            verbose=False,
        )
    except Exception as exc:
        if device != "cuda":
            raise
        note = (
            f"GPU transcription was not available ({exc}); "
            "the recording was transcribed on the CPU instead."
        )
        model = whisper.load_model(req["variant"], device="cpu")
        result = model.transcribe(
            req["audio"],
            language=None if req.get("language") in ("", "auto") else req.get("language"),
            task="translate" if req.get("translate_to") else "transcribe",
            verbose=False,
        )
        device = "cpu"
    return {"segments": _normalise(result.get("segments", [])),
            "device": device, "note": note}


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------
def _requested_language(req):
    language = req.get("language")
    return None if language in ("", "auto", None) else language


def _signature_parameters(callable_object) -> dict:
    try:
        return dict(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return {}


def _dtype_kwargs() -> dict:
    """The checkpoint's own dtype, under the name this transformers uses.

    The default widened the weights to float32 on the way in, which turned a
    1.7B model into 7.3 GiB of video memory - more than twice the 3.4 GiB it
    is published as - and made every tensor allocated afterwards twice as
    large as well.  ``auto`` keeps the bfloat16 the checkpoint ships in, which
    is what the model was trained with.

    transformers 5 renamed ``torch_dtype`` to ``dtype``; the old name warns
    there and the new one does not exist in 4.x, so the major version decides
    which is spelled.  No version at all (a stand-in module, an odd string)
    keeps the default rather than guessing.
    """
    try:
        import transformers

        major = int(str(transformers.__version__).split(".", 1)[0])
    except Exception:
        return {}
    return {"dtype": "auto"} if major >= 5 else {"torch_dtype": "auto"}


def _load_repository_model(auto_class, hf_id, req, device):
    """Load a repository's model onto the chosen device.

    ``device_map`` needs ``accelerate``; a model environment installed before
    that package was part of the plan still has to work, so the model is then
    placed on the device by hand instead.  The weights are loaded as the
    checkpoint stores them (:func:`_dtype_kwargs`) rather than widened to
    float32 on the way in.
    """
    hf_kwargs = {"trust_remote_code": bool(req.get("trust_remote_code")),
                 **_dtype_kwargs()}
    try:
        return auto_class.from_pretrained(
            hf_id, device_map="cuda:0" if device == "cuda" else "cpu",
            **hf_kwargs,
        )
    except (ImportError, ValueError) as exc:
        if "accelerate" not in str(exc):
            raise
        model = auto_class.from_pretrained(hf_id, **hf_kwargs)
        return model.to("cuda" if device == "cuda" else "cpu")


def _transcription_request(processor, req, hf_id, audio):
    """Build a processor's transcription inputs, however it spells the call.

    Qwen3-ASR's ``apply_transcription_request`` takes the audio alone;
    Voxtral's also wants the repository id.  Reading the signature keeps one
    path working for both (and for the next audio model to arrive) instead of
    hard-coding each processor's shape.  ``audio`` is the piece of the
    recording this call transcribes: the whole file when it fits, or one of
    the pieces it was cut into when it does not.
    """
    parameters = _signature_parameters(processor.apply_transcription_request)
    extra = {}
    if "model_id" in parameters:
        extra["model_id"] = hf_id
    if not parameters or "language" in parameters:
        extra["language"] = _requested_language(req)
    return processor.apply_transcription_request(audio=audio, **extra)


def _run_chat_audio_model(req, hf_id, processor, device):
    """Transcribe with an audio model driven by a chat prompt plus the audio.

    Granite Speech is documented exactly this way, and its feature extractor
    rejects the generic pipeline's ``sampling_rate`` argument, so the pipeline
    cannot run it at all.

    The recording is handed over in pieces of at most ``AUDIO_CHUNK_SECONDS``
    (see there for why one call over the whole of it is what exhausted a
    card), and a piece that still will not fit is cut in half until it does.
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq

    tokenizer = getattr(processor, "tokenizer", None)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": TRANSCRIBE_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    samples, rate = _decode_audio(req["audio"])
    model = _load_repository_model(
        AutoModelForSpeechSeq2Seq, hf_id, req, device,
    )
    kwargs = {"return_tensors": "pt"}
    if "sampling_rate" in _signature_parameters(processor.__call__):
        kwargs["sampling_rate"] = rate

    def transcribe(piece):
        inputs = processor(prompt, piece, **kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device, model.dtype)
        budget = _token_budget(len(piece) / float(rate))
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=budget,
                do_sample=False, num_beams=1,
            )
        prompt_length = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[0, prompt_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, _truncation_note(new_tokens, budget)

    def halve(piece):
        if not len(piece) \
                or len(piece) / float(rate) <= MIN_AUDIO_CHUNK_SECONDS:
            return None
        middle = len(piece) // 2
        return piece[:middle], piece[middle:]

    texts, notes = [], []
    for piece in _split_audio(samples, rate):
        text, note = _transcribe_in_halves(piece, transcribe, halve, device)
        if text:
            texts.append(text)
        if note:
            notes.append(note)
    return {"segments": _normalise([{"text": " ".join(texts)}]),
            "note": " ".join(notes), "timestamps": False}


def _run_audio_language_model(req, hf_id, processor, device):
    """Transcribe with an audio-in / text-out language model.

    Qwen3-ASR and Voxtral are decoder-only speech models: the ASR pipeline
    cannot drive them (its beam search reshapes a three-dimensional decoder
    input and fails) and they have no timestamps.  Their processors expose
    ``apply_transcription_request`` instead, which builds the prompt the model
    was trained on.

    The recording is handed over in pieces of at most ``AUDIO_CHUNK_SECONDS``
    - one call over the whole of a 29-minute chapter asked for a 31.5 GiB
    attention tensor (see :data:`AUDIO_CHUNK_SECONDS`) - and a piece that
    still will not fit is cut in half until it does.  The pieces' text is
    joined in order: each call ends with the model's own end-of-sequence
    token, so the joins fall between sentences rather than inside one.
    """
    import torch
    from transformers import AutoModelForMultimodalLM

    model = _load_repository_model(
        AutoModelForMultimodalLM, hf_id, req, device,
    )

    def transcribe(piece):
        inputs = _transcription_request(processor, req, hf_id, piece)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device, model.dtype)
        budget = _max_new_tokens(piece)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=budget)
        prompt_length = int(inputs["input_ids"].shape[1])
        new_tokens = generated[:, prompt_length:]
        text = ""
        try:
            text = processor.decode(
                new_tokens, return_format="transcription_only")[0]
        except (TypeError, ValueError, IndexError):  # pragma: no cover - older API
            text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text, _truncation_note(new_tokens, budget)

    texts, notes = [], []
    with tempfile.TemporaryDirectory(prefix="aivs_stt_") as tmpdir:
        for index, piece in enumerate(_recording_pieces(req["audio"], tmpdir)):
            text, note = _transcribe_in_halves(
                piece, transcribe,
                lambda current, i=index: _halve_wav(current, tmpdir, f"p{i}"),
                device,
            )
            if text:
                texts.append(text)
            if note:
                notes.append(note)
    return {"segments": _normalise([{"text": " ".join(texts)}]),
            "note": " ".join(notes), "timestamps": False}


def _run_asr_pipeline(req, hf_id, device):
    """Transcribe through the generic transformers ASR pipeline."""
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=hf_id,
        device=0 if device == "cuda" else -1,
        trust_remote_code=bool(req.get("trust_remote_code")),
    )
    samples, rate = _decode_audio(req["audio"])
    pipeline_type = getattr(asr, "type", "")
    # Only Whisper (and CTC heads) can produce timings through the pipeline;
    # asking any other architecture raises
    #   "We cannot return_timestamps yet on non-CTC models apart from Whisper!"
    wants_timestamps = bool(req.get("timestamps"))
    kwargs = {}
    if wants_timestamps and pipeline_type in ("seq2seq_whisper", "ctc"):
        kwargs["return_timestamps"] = True
    language = _requested_language(req)
    if language and pipeline_type == "seq2seq_whisper":
        # Whisper is the only pipeline that understands a language hint.
        kwargs["generate_kwargs"] = {"language": language}
    try:
        result = asr({"array": samples, "sampling_rate": rate}, **kwargs)
    except ValueError as exc:
        if "return_timestamps" not in str(exc) or "return_timestamps" not in kwargs:
            raise
        kwargs.pop("return_timestamps", None)
        result = asr({"array": samples, "sampling_rate": rate}, **kwargs)
    if isinstance(result, dict) and result.get("chunks"):
        return {"segments": _normalise(result["chunks"]), "note": "",
                "timestamps": True}
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    return {"segments": _normalise([{"text": text}]) if str(text).strip() else [],
            "note": "", "timestamps": False}


def _load_processor(hf_id, trust):
    """The repository's processor, or ``None`` when it cannot be loaded."""
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
    except Exception:
        return None


def _with_device(result, device):
    """Record which device the run really used (the dialog reports it)."""
    if isinstance(result, dict):
        result.setdefault("device", device)
    return result


def _run_transformers(req):
    hf_id = req.get("hf_model")
    if not hf_id:
        raise RuntimeError(
            "No model repository is configured for this model and variant."
        )
    device = _resolve_device(req.get("compute", "cpu"))
    runner = req.get("runner") or RUNNER_PIPELINE
    if runner in (RUNNER_TRANSCRIPTION_REQUEST, RUNNER_CHAT):
        processor = _load_processor(hf_id, bool(req.get("trust_remote_code")))
        if runner == RUNNER_TRANSCRIPTION_REQUEST and callable(
            getattr(processor, "apply_transcription_request", None)
        ):
            return _with_device(
                _run_audio_language_model(req, hf_id, processor, device), device,
            )
        tokenizer = getattr(processor, "tokenizer", None)
        if runner == RUNNER_CHAT and callable(
            getattr(tokenizer, "apply_chat_template", None)
        ):
            return _with_device(
                _run_chat_audio_model(req, hf_id, processor, device), device,
            )
    # A model whose own request API is unavailable still goes through the
    # generic pipeline rather than failing outright.
    return _with_device(_run_asr_pipeline(req, hf_id, device), device)


# ---------------------------------------------------------------------------
# NVIDIA NeMo
# ---------------------------------------------------------------------------
def _nemo_timestamps(hypothesis, duration: float):
    """Timed segments from a NeMo ``Hypothesis`` (word or segment times)."""
    if isinstance(hypothesis, str):
        text = hypothesis.strip()
        return [{"start": 0.0, "end": duration or None, "text": text}] if text else []
    stamp = getattr(hypothesis, "timestamp", None)
    if isinstance(stamp, dict):
        for key in ("segment", "word"):
            entries = stamp.get(key) or []
            segments = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                text = str(
                    entry.get("segment") or entry.get("word")
                    or entry.get("text") or ""
                ).strip()
                if text:
                    segments.append({
                        "start": entry.get("start"),
                        "end": entry.get("end"),
                        "text": text,
                    })
            if segments:
                return segments
    text = str(getattr(hypothesis, "text", "") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": duration or None, "text": text}]


def _nemo_asr_segments(model, req) -> dict:
    """Run a NeMo ASR model's own ``transcribe`` and time the segments.

    NeMo returns word/segment timestamps for CTC and transducer models;
    asking a model that has none is an error, so try, then retry plainly.
    """
    try:
        results = model.transcribe([req["audio"]], timestamps=True)
    except Exception:
        results = model.transcribe([req["audio"]])
    duration = _audio_duration(req["audio"])
    segments = []
    for hypothesis in results or []:
        segments.extend(_nemo_timestamps(hypothesis, duration))
    return {"segments": segments, "note": "", "timestamps": True}


def _run_nemo_salm(name, req, device) -> dict:
    """Transcribe with a NeMo speech-language model.

    ``nvidia/canary-qwen-2.5b`` is a SALM: its repository holds a
    transformers-style ``config.json`` and ``model.safetensors`` and no
    ``model_config.yaml``, so the ASR entry point cannot open it at all.
    ``SALM`` documents a high-level ``generate`` that takes a chat turn with
    the audio attached.
    """
    import torch
    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained(name)
    model = model.to("cuda" if device == "cuda" else "cpu")
    model.eval()
    prompt = [{
        "role": "user",
        "content": f"Transcribe the following: {model.audio_locator_tag}",
        "audio": [req["audio"]],
    }]
    budget = _max_new_tokens(req["audio"])
    with torch.no_grad():
        answer = model.generate(
            prompts=[prompt], max_new_tokens=budget,
        )
    tokens = answer[0].tolist() if hasattr(answer, "__getitem__") else list(answer)
    text = model.tokenizer.ids_to_text(tokens)
    return {"segments": _normalise([{"text": str(text).strip()}]),
            "note": _truncation_note(tokens, budget), "timestamps": False}


def _run_nemo(req):
    """Transcribe with a NeMo model (Canary-Qwen, Parakeet TDT, ...)."""
    _prepare_native_libraries()

    name = req.get("hf_model") or req.get("variant")
    if req.get("translate_to"):
        raise RuntimeError(
            "Speech translation is not wired up for the NVIDIA NeMo models "
            "yet. Choose \"Don't translate\" to transcribe this recording."
        )
    device = _resolve_device(req.get("compute", "cpu"))
    runner = req.get("runner") or ""
    # The catalog knows which kind of NeMo model this is.  A speech-language
    # model must not go anywhere near the ASR entry point (it has no
    # ``.nemo`` checkpoint), and an ordinary ASR checkpoint must not fall
    # through to the SALM loader (its config has no audio locator tag).
    if runner == "salm":
        return _run_nemo_salm(name, req, device)
    model = None
    asr_error = None
    try:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(name)
    except Exception as exc:
        if runner == "asr":
            raise
        asr_error = exc
        model = None
    if model is None:
        try:
            return _run_nemo_salm(name, req, device)
        except Exception:
            # The repository is a SALM neither; the ASR loader's error is the
            # one that names the real problem, so report that instead of the
            # confusing "Missing key audio_locator_tag" from the SALM config.
            if asr_error is not None:
                raise RuntimeError(
                    f"{type(asr_error).__name__}: {asr_error}"
                ) from asr_error
            raise
    try:
        model = model.to(device) if device == "cuda" else model
    except Exception as exc:
        sys.stderr.write(f"Ignoring a failure in _run_nemo: {exc}\n")
    return _nemo_asr_segments(model, req)


# ---------------------------------------------------------------------------
# SpeechBrain
# ---------------------------------------------------------------------------
def _run_speechbrain(req):
    """Transcribe with a SpeechBrain recipe from the Hugging Face Hub."""
    from speechbrain.inference.ASR import EncoderDecoderASR

    source = req.get("hf_model")
    if not source:
        raise RuntimeError(
            "No SpeechBrain model is configured for this model and variant."
        )
    device = _resolve_device(req.get("compute", "cpu"))
    savedir = os.path.join(
        os.path.expanduser("~"), ".cache", "aivs_speechbrain",
        str(source).replace("/", "--"),
    )
    os.makedirs(savedir, exist_ok=True)
    # SpeechBrain links the downloaded files into ``savedir`` by default, and
    # a symlink needs Developer Mode (or an elevated process) on Windows - the
    # first download failed with "[WinError 1314] A required privilege is not
    # held by the client".  A plain copy always works.
    link: dict = {}
    try:
        from speechbrain.utils.fetching import LocalStrategy

        link["local_strategy"] = LocalStrategy.COPY
    except Exception:
        link = {}
    asr = EncoderDecoderASR.from_hparams(
        source=source, savedir=savedir, run_opts={"device": device}, **link,
    )
    text = _speechbrain_text(asr, req)
    return {"segments": _normalise([{"text": text}]), "note": "",
            "timestamps": False}


def _speechbrain_text(asr, req) -> str:
    """Transcribe one file with a loaded SpeechBrain recipe.

    ``transcribe_file`` runs the path through SpeechBrain's own *fetcher*,
    which treats it as relative to the model source and mangled an absolute
    Windows path into ``<current folder>\\C:\\...``, so the recording was never
    found.  Decoding the audio here and using the model's batch API avoids its
    path handling completely; a very old SpeechBrain without that API still
    gets the plain call.
    """
    if not (hasattr(asr, "transcribe_batch") and hasattr(asr, "audio_normalizer")):
        return asr.transcribe_file(req["audio"])
    import torch

    samples, rate = _decode_audio(req["audio"])
    signal = torch.from_numpy(samples).unsqueeze(1)  # [time, channels]
    signal = signal.to(getattr(asr, "device", "cpu"))
    signal = asr.audio_normalizer(signal, rate)
    texts, _tokens = asr.transcribe_batch(signal.unsqueeze(0),
                                          torch.tensor([1.0]))
    return str(texts[0])


_RUNNERS = {
    "faster_whisper": _run_faster_whisper,
    "whisper": _run_openai_whisper,
    "transformers": _run_transformers,
    "nemo": _run_nemo,
    "speechbrain": _run_speechbrain,
}


def run(request):
    backend = request.get("backend")
    runner = _RUNNERS.get(backend)
    if runner is None:
        raise RuntimeError(
            f"The {backend!r} back-end is not supported by the worker yet."
        )
    result = runner(request)
    if isinstance(result, dict) and "segments" in result:
        return result
    return {"segments": result, "note": "", "timestamps": False}


#: Fragments that mark a line as progress-bar or warning chatter rather than
#: the cause of a failure.  A download's tqdm bar and transformers' deprecation
#: warnings otherwise became the headline the dialog showed.
_NOISE_MARKERS = ("%|", "it/s", "s/it", "B/s")


def _clean_message(text: str) -> str:
    """Strip progress bars and deprecation warnings from an exception's text."""
    kept = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in line for marker in _NOISE_MARKERS):
            continue
        if lowered.startswith("warning:") or "deprecated" in lowered \
                or "futurewarning" in lowered or "userwarning" in lowered:
            continue
        kept.append(line)
    return "\n".join(kept) if kept else "the model produced no result"


def _explain(exc: BaseException) -> str:
    """The text the dialog shows for a failed run.

    A gated repository reports a bare ``401``/``GatedRepoError``, which tells
    the user nothing they can act on, so it is spelled out.  Progress output and
    deprecation warnings captured into the message are dropped, so the cause is
    what the user reads.
    """
    text = _clean_message(f"{type(exc).__name__}: {exc}")
    lowered = text.lower()
    if "gated" in lowered or "unauthorized" in lowered \
            or "401" in text or "403" in text:
        text += (
            "\n\nThis model is gated on Hugging Face. Accept its licence on "
            "the model's own page, then paste a read token from "
            "https://huggingface.co/settings/tokens into Settings > Hugging "
            "Face account in the application (an HF_TOKEN environment "
            "variable or an earlier `huggingface-cli login` works too), and "
            "transcribe again."
        )
    return text


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2:
        print("usage: worker.py <request.json>", file=sys.stderr)
        return 2
    request_path = argv[1]
    try:
        with open(request_path, encoding="utf-8") as fh:
            request = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read the request: {exc}", file=sys.stderr)
        return 2

    out_path = request.get("out")
    try:
        result = run(request)
        response = {"ok": True, "segments": result["segments"]}
        for key in ("device", "note"):
            if result.get(key):
                response[key] = result[key]
    except Exception as exc:
        response = {"ok": False, "error": _explain(exc)}

    try:
        if out_path:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(response, fh, ensure_ascii=False)
        else:
            print(json.dumps(response, ensure_ascii=False))
    except OSError as exc:
        print(f"could not write the response: {exc}", file=sys.stderr)
        return 1
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
