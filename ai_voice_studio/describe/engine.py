"""Audio description execution.

The catalog says *what* a describing model is; this module says *how* it runs.
Every model runs inside **its own** Python environment (see
:mod:`ai_voice_studio.describe.environments`), so two describers with conflicting
dependencies can never break each other.  The application never imports a
model's back-end itself: it starts :mod:`ai_voice_studio.describe.worker` with
the model environment's interpreter, hands it the chunk plan as JSON and reads
the description segments back the same way.

One worker run describes one source file, loading the model **once** and then
looping over that file's chunks.  That matters: loading a 7B model per chunk
would dwarf the describing itself.  The worker reports each finished chunk on
stdout as it goes, so the user sees progress on a long video.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .. import hf_access
from ..audio.ffmpeg import find_ffmpeg
from ..constants import (
    AD_CHUNK_SECONDS,
    AD_DEFAULT_FPS,
    AD_MAX_DESCRIPTION_SECONDS,
    AD_MIN_GAP_SECONDS,
    AD_WINDOW_SECONDS,
)
from . import catalog, environments, media, pipeline, prompts, weights

log = logging.getLogger(__name__)

#: ``<percent>%|<message>`` lines the worker writes as it finishes each chunk.
PROGRESS_PREFIX = "AIVS-PROGRESS|"

#: ``<json>`` lines the worker writes the moment a description exists.
SEGMENT_PREFIX = "AIVS-SEGMENT|"


class DescribeEngineError(RuntimeError):
    """A description run could not be performed (bad input, failure, ...)."""


class BackendMissing(DescribeEngineError):
    """The model's environment, or its back-end package, is not installed."""


# ---------------------------------------------------------------------------
# Environment / availability (mirrors the STT engine)
# ---------------------------------------------------------------------------
def env_dir(model_id: str) -> str:
    return environments.env_dir(model_id)


def is_installed(model_id: str) -> bool:
    return environments.is_installed(model_id)


def env_exists(model_id: str) -> bool:
    return environments.is_created(model_id)


def installed_version(model_id: str) -> str | None:
    return environments.installed_version(model_id)


def install_plan(model_id: str):
    return environments.install_plan(model_id)


def install(model_id: str, progress: Callable[[str], None] | None = None,
            cancel_event: threading.Event | None = None) -> dict[str, object]:
    return environments.install(model_id, progress=progress,
                                cancel_event=cancel_event)


def remove(model_id: str) -> str | None:
    return environments.remove(model_id)


def model_available(model_id: str) -> bool:
    """True when the model's back-end is installed in its own environment."""
    return environments.is_installed(model_id)


def model_state_label(model_id: str) -> str:
    """``ready to use`` or ``not installed`` - what a combo entry can carry.

    A model list that does not say which entries can describe a video right now
    makes the user guess, and the guess is wrong in the case that matters (a
    model whose environment was never built).  Both the wizard and the describe
    window label every entry with this.
    """
    return "ready to use" if environments.is_installed(model_id) else \
        "not installed"


# ---------------------------------------------------------------------------
# One variant's checkpoint files
# ---------------------------------------------------------------------------
def variant_repo(model_id: str, variant: str) -> str:
    """The Hugging Face repository a variant's weights come from."""
    return str(catalog.hf_model(model_id, variant) or "")


def variant_downloaded(model_id: str, variant: str) -> bool:
    """True when this variant's checkpoint is already on disk."""
    repo = variant_repo(model_id, variant)
    return bool(repo) and weights.is_downloaded(repo)


def variant_partial(model_id: str, variant: str) -> bool:
    """True when a fetch of this variant stopped before it had finished.

    The files are on disk but the checkpoint cannot be loaded, which is a state
    of its own: it is what a cancelled or failed download leaves behind, and the
    download that finishes it is what the panel has to keep offering.
    """
    return weights.is_partly_downloaded(variant_repo(model_id, variant))


def variant_state_label(model_id: str, variant: str) -> str:
    """What one variant of a model is, in the user's terms.

    Deliberately separate from the *environment* state: the packages are shared
    by every variant of a model, the checkpoint is not, and saying so is the
    whole point of the status column in Settings.
    """
    variant = variant or catalog.default_variant(model_id)
    if not catalog.hf_model(model_id, variant):
        return "No download for this choice"
    # A gated repository is said here, where the choice is made and *before*
    # anything is installed: the download will refuse itself until its licence is
    # accepted and a token is configured, and finding that out after pressing
    # Download (or after building the environment) wastes the wait.  It is
    # therefore computed before the environment check, not after it.
    gated = catalog.gated_note(model_id, variant)
    extra = f". {gated}" if gated else ""
    if not environments.is_installed(model_id):
        return f"Install the model first{extra}"
    if variant_downloaded(model_id, variant):
        return "Downloaded"
    size = catalog.variant_size_gb(model_id, variant)
    # A fetch that stopped part-way is neither of the two states above: the files
    # are on disk, but the checkpoint cannot be loaded, and the download that
    # would finish it is available.  Reported as its own state because "Not
    # downloaded" sends the user looking for gigabytes that are already there.
    if weights.is_partly_downloaded(variant_repo(model_id, variant)):
        return (f"Partly downloaded - about {size:.1f} GB in all; download it "
                f"again to continue where it stopped{extra}")
    return f"Not downloaded (about {size:.1f} GB){extra}"


def download_variant(model_id: str, variant: str = "",
                     progress: Callable[[str], None] | None = None,
                     cancel_event: threading.Event | None = None
                     ) -> dict[str, object]:
    """Fetch one variant's checkpoint files now, not at the first run.

    The download runs inside the model's own environment, because that is the
    one that carries ``huggingface_hub``; the environment itself has to exist
    first, and the returned error says so when it does not.
    """
    entry = catalog.model(model_id)
    variant = variant or catalog.default_variant(model_id)
    repo = variant_repo(model_id, variant)
    if not repo:
        return {"ok": False,
                "error": f"{entry['name']} has no repository to download for "
                         f"variant '{variant}'."}
    if not environments.is_installed(model_id):
        return {"ok": False,
                "error": f"Install {entry['name']} first (Settings > Manage "
                         "the video describer): its own environment carries "
                         "the downloader."}
    request = {
        "mode": "download",
        "model": model_id,
        "variant": variant,
        "hf_model": repo,
    }
    _report(progress, f"Downloading {repo}...", None)
    try:
        response, _streamed = _worker_response(model_id, request, progress,
                                               cancel_event=cancel_event)
    except DescribeEngineError as exc:
        return {"ok": False, "error": str(exc)}
    if not response.get("ok"):
        return {"ok": False,
                "error": _worker_error_detail(str(response.get("error") or ""))
                         or "The download failed."}
    note = str(response.get("note") or "")
    _report(progress, note or "Downloaded.", 1.0)
    return {"ok": True, "repo": repo, "path": str(response.get("path") or ""),
            "note": note}


def remove_variant(model_id: str, variant: str = "") -> str | None:
    """Delete one variant's cached checkpoint; a note when it was not clean."""
    repo = variant_repo(model_id, variant or catalog.default_variant(model_id))
    if not repo:
        return None
    return weights.remove(repo)


def gpu_runtime_missing(model_id: str) -> list[str]:
    return environments.missing_cuda_runtime(model_id)


def missing_packages(model_id: str) -> list[str]:
    """Packages the catalogue declares that this model's environment lacks.

    ``gpu_runtime_missing`` answers this for the CUDA-only extras; this covers
    everything the catalogue declares, including the packages a model's *own*
    modelling code imports.  Without it an environment that is missing one of
    them looks installed, and the only way back is to delete it and build it
    again.
    """
    return environments.missing_packages(model_id)


def availability_note(model_id: str) -> str:
    """A one-line status for the describe dialog and the settings table."""
    if environments.is_installed(model_id):
        missing = environments.missing_packages(model_id)
        if missing:
            return (
                "Installed, but its environment is missing "
                + ", ".join(missing)
                + " - use Repair model environment before a run "
                f"({environments.env_dir(model_id)})."
            )
        version = environments.installed_version(model_id)
        suffix = f", version {version}" if version else ""
        return (
            f"Ready in its own environment ({environments.env_dir(model_id)}"
            f"{suffix})."
        )
    packages = ", ".join(catalog.pip_packages(model_id))
    return (
        "Not installed. Its own environment is "
        f"{environments.env_dir(model_id)}. Install: {packages}"
    )


def cuda_available() -> bool:
    """True when the dialog may offer a GPU run at all."""
    from .. import compute as _compute

    return _compute.has_nvidia_gpu()


def compute_options(model_id: str, variant: str = "") -> list[dict[str, str]]:
    """Compute choices the dialog shows for this model and variant."""
    return [
        {"value": value, "label": label}
        for value, label in catalog.device_options(
            model_id, variant=variant, has_cuda=cuda_available()
        )
    ]


def resolve_device(compute: str) -> str:
    """``cuda``/``cpu`` for the worker, from the dialog's compute choice."""
    if compute == "cuda":
        return "cuda"
    if compute == "auto":
        return "cuda" if cuda_available() else "cpu"
    return "cpu"


def _report(on_progress, message: str, fraction: float | None = None) -> None:
    if on_progress is not None:
        try:
            on_progress(message, fraction)
        except Exception:
            log.debug("describe progress callback failed", exc_info=True)


# ---------------------------------------------------------------------------
# Planning a whole file
# ---------------------------------------------------------------------------
def plan_for_video(video_path: str,
                   chunk_seconds: float = AD_CHUNK_SECONDS,
                   window_seconds: float = AD_WINDOW_SECONDS) -> dict[str, Any]:
    """Probe a video and return its chunk plan, windowed for describing.

    Each chunk carries the ``windows`` it is described in: a chunk bounds memory,
    progress and resume, while a window is what one description covers - see
    :data:`ai_voice_studio.constants.AD_WINDOW_SECONDS`.  The one call that
    matters to a listener is the model's, and it is asked once per window.

    Raises :class:`DescribeEngineError` when the length cannot be read at all -
    FFmpeg missing or the file unreadable - because every later decision (the
    estimate, the chunking, the warning) needs it and guessing would be worse
    than saying so.
    """
    if not video_path or not os.path.isfile(video_path):
        raise DescribeEngineError(f"Video file not found: {video_path}")
    ffmpeg_exe = find_ffmpeg()
    duration = media.probe_duration(video_path, ffmpeg_exe)
    if duration is None:
        raise DescribeEngineError(
            "Could not read the length of this video. Audio description needs "
            "FFmpeg to decode the video and to sample its frames."
        )
    chunks = pipeline.plan_chunks(duration, chunk_seconds)
    for chunk in chunks:
        chunk["windows"] = pipeline.plan_windows(chunk, window_seconds)
    return {
        "video": video_path,
        "duration": duration,
        "ffmpeg": ffmpeg_exe or "",
        "chunks": chunks,
        "windows": pipeline.window_count(chunks),
        "window_seconds": float(window_seconds),
    }


# ---------------------------------------------------------------------------
# Running one file
# ---------------------------------------------------------------------------
def describe_video(
    video_path: str,
    *,
    model_id: str,
    variant: str = "",
    fps: float = AD_DEFAULT_FPS,
    chunk_seconds: float = AD_CHUNK_SECONDS,
    window_seconds: float = AD_WINDOW_SECONDS,
    device: str = "cpu",
    language: str = "en",
    gap_fitting: bool = False,
    speech_segments: Sequence[dict[str, Any]] | None = None,
    max_description_seconds: float = AD_MAX_DESCRIPTION_SECONDS,
    min_gap: float = AD_MIN_GAP_SECONDS,
    on_progress=None,
    on_segment=None,
) -> list[dict[str, Any]]:
    """Describe one video into timed description segments.

    ``speech_segments`` (optional, from the speech-to-text model) are the spoken
    phrases; when ``gap_fitting`` is on and they are given, each description is
    placed into the silence that follows it instead of at its own window, so the
    narration never talks over the dialogue.

    Every chunk is described window by window (``window_seconds``, twenty by
    default), which is what makes a description cover a change rather than a sum
    up two minutes of video in a sentence.
    """
    entry = catalog.model(model_id)
    variant = variant or catalog.default_variant(model_id)
    if not environments.is_created(model_id):
        raise BackendMissing(
            f"{entry['name']} has no Python environment of its own yet "
            f"({environments.env_dir(model_id)}).\n\nOpen Settings > Manage the "
            "video describer and install this model there."
        )
    if not environments.is_installed(model_id):
        packages = ", ".join(catalog.pip_packages(model_id))
        raise BackendMissing(
            f"{entry['name']} is not installed in its own environment "
            f"({environments.env_dir(model_id)}).\n\nInstall: {packages}"
        )

    plan = plan_for_video(video_path, chunk_seconds, window_seconds)
    duration = float(plan["duration"])
    chunks: list[dict[str, Any]] = plan["chunks"]

    request: dict[str, Any] = {
        "video": video_path,
        "ffmpeg": plan["ffmpeg"],
        "backend": catalog.backend(model_id),
        "model": model_id,
        "hf_model": catalog.hf_model(model_id, variant),
        "variant": variant,
        # The worker decides how to load the checkpoint from how big it is: a
        # model larger than the card is asked for in 4-bit instead of being
        # allowed to spill to the disk, where it cannot be generated from.
        "model_size_gb": catalog.variant_size_gb(model_id, variant),
        "device": device,
        "fps": fps,
        "language": language,
        "chunks": chunks,
        "window_seconds": plan["window_seconds"],
        "guideline_prompt": prompts.GUIDELINE_PROMPT,
        # One description covers one window, and the worker tells the model
        # what the previous window's description said so it reports the change
        # rather than describing the same scene again.
        "chunk_instruction": prompts.WINDOW_INSTRUCTION,
        "trust_remote_code": catalog.needs_remote_code(model_id),
        # How to load it and how to ask it, both taken from the checkpoint
        # rather than assumed: a repository that ships its own modelling code
        # names the ``transformers`` class it registered (MiniCPM-o registers
        # ``AutoModel``, not the task-specific class a run would otherwise ask
        # for) and can need to be driven through its own ``chat`` method.  The
        # extra ``from_pretrained`` keywords are the model's own too - MiniCPM-o
        # is loaded vision-only, which is the whole of what a description uses.
        "auto_class": catalog.auto_class(model_id),
        "driver": catalog.driver(model_id),
        "load_kwargs": catalog.load_kwargs(model_id),
        # The dtype is the checkpoint's own too, for the one model whose code
        # fixes it: Moondream 2 casts its image crops to bfloat16, so a CPU run
        # forcing float32 fails in the first layer rather than running slowly.
        "dtype": catalog.dtype(model_id),
        "timestamps": catalog.supports_timestamps(model_id),
        "max_description_seconds": max_description_seconds,
    }
    segments = _run_worker(model_id, request, on_progress, on_segment)

    if gap_fitting and speech_segments:
        fitted = pipeline.fit_into_gaps(
            segments, speech_segments, duration=duration,
            min_gap=min_gap, max_seconds=max_description_seconds,
        )
        placed = list(fitted.get("placed", []))
        dropped = int(fitted.get("dropped", 0))
        if dropped:
            _report(
                on_progress,
                f"{dropped} description(s) did not fit any pause and were left "
                "out of the timed track.",
                None,
            )
        if placed:
            return placed
    return segments


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
#: Fragments that mark a line as progress chatter rather than a cause.
_NOISE_MARKERS = ("%|", "it/s", "s/it", "B/s", "deprecated", "userwarning",
                  "futurewarning", "warnings.warn")


def _meaningful_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(PROGRESS_PREFIX):
            continue
        lowered = stripped.lower()
        if lowered.startswith(("warning:", "note:")) or any(
            marker in lowered for marker in _NOISE_MARKERS
        ):
            continue
        lines.append(stripped)
    return lines


#: A line that names a Python exception, e.g. ``OSError: [WinError 1314] ...``.
_EXCEPTION_RE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b")

#: What :func:`_worker_error_detail` says when the worker left nothing useful.
_NO_WORKER_RESULT = "the worker produced no result"


#: Exit codes that mean the model's process *died* rather than returned, keyed
#: by the unsigned value Windows reports.  0xC0000005 is an access violation -
#: what a run of this feature produced on this machine when the model it was
#: asked for did not fit in the memory available - and 0xC00000FD a stack
#: overflow.  A negative code on POSIX is a signal.
_CRASH_CODES = {
    0xC0000005: "an access violation",
    0xC00000FD: "a stack overflow",
    0xC0000374: "heap corruption",
}


def _crash_note(returncode: int) -> str:
    """What a non-zero exit code means, in the user's terms (``""`` if none).

    A process that dies prints no result and no traceback of its own, so the
    text it left behind was the only thing to go on: a real run here failed with
    ``0xC0000005`` after the model had offloaded part of itself to the disk, and
    the window reported - as the cause - a warning about Windows symlinks that
    sat three lines above the crash.  The exit code is the fact; that warning was
    not.
    """
    if not returncode:
        return ""
    unsigned = returncode & 0xFFFFFFFF
    what = _CRASH_CODES.get(unsigned)
    if not what:
        if returncode < 0:
            return (f"the model's process was killed by signal {-returncode} "
                    "before it produced a description")
        return (f"the model's process exited with code {returncode} without "
                "producing a description")
    note = (f"the model's process stopped (exit code {returncode}: {what}) "
            "before it produced a description")
    # The one cause worth naming: this feature runs models that can be far larger
    # than the machine, and the smallest variant is a click away.
    note += (".  The usual cause is that the chosen model did not fit in this "
            "machine's video memory and system memory; a smaller variant (or "
            "the CPU/GPU choice) from Settings > Manage the video describer "
            "runs instead.")
    return note


def _worker_error_detail(text: str) -> str:
    """The worker's own cause, in preference to the advice printed after it.

    Windows is why this is fussy.  A permission error while the cache writes a
    checkpoint is followed by several lines telling the reader to activate
    Developer Mode, and the tail of that is advice and a URL: a run here failed
    and the dialog reported nothing but the advice.  When a line names an
    exception, that line and the rest of its report are the answer; when nothing
    does (a killed process, a silent exit) the tail is still the best guess.
    """
    lines = _meaningful_lines(text)
    if not lines:
        return _NO_WORKER_RESULT
    for index in range(len(lines) - 1, -1, -1):
        if _EXCEPTION_RE.match(lines[index]):
            return "\n".join(lines[index:index + 3])
    return "\n".join(lines[-3:])


def _run_worker(model_id: str, request: dict[str, Any], on_progress,
                on_segment=None,
                cancel_event: threading.Event | None = None
                ) -> list[dict[str, Any]]:
    """Start the worker in the model's environment and read its JSON response.

    The worker's stdout is read **line by line** while it runs: progress lines
    drive the dialog's gauge, and ``AIVS-SEGMENT`` lines hand each finished
    description to ``on_segment`` straight away, which is what lets the job save
    a partial script as it goes.
    """
    response, streamed = _worker_response(
        model_id, request, on_progress, on_segment, cancel_event,
    )
    segments = list(response.get("segments", []))
    # The response file is authoritative; the streamed lines are only a progress
    # convenience, so an empty response with streamed segments is still a result.
    return segments or streamed


def _worker_response(model_id: str, request: dict[str, Any], on_progress,
                     on_segment=None,
                     cancel_event: threading.Event | None = None
                     ) -> tuple:
    """Run the worker once; return ``(response, streamed segments)``."""
    repo = str(request.get("hf_model") or "")
    # A gated repository cannot be fetched anonymously at all, so a run that
    # would need a token stops here - before the environment is touched, before
    # a download is attempted, and while the answer can still be two short
    # paragraphs rather than a bare 401 from the middle of a worker's log.  A
    # token being configured is not proof the licence was accepted, so the
    # message names that step too.  This is the failure the user reported:
    # Gemma 3 12B answering 401 after its download had already begun.
    if catalog.gated(model_id, str(request.get("variant") or "")) \
            and not hf_access.has_token():
        raise DescribeEngineError(
            hf_access.gated_help(repo, catalog.model(model_id)["name"])
        )
    try:
        script = environments.worker_script()
        # ``usable_python_exe`` (not ``python_exe``): an environment built by an
        # earlier installation of the application runs on a Python that is no
        # longer there, and is re-pointed at this one before it is used.
        python_exe = environments.runtime(model_id).usable_python_exe()
    except Exception as exc:
        raise DescribeEngineError(
            f"Could not start the model's environment: {exc}"
        ) from exc
    if not os.path.isfile(python_exe):
        raise BackendMissing(
            f"The Python environment for this model is not usable "
            f"({python_exe} is missing). Reinstall it from Settings > Manage "
            "the video describer."
        )
    if not request.get("hf_model"):
        raise DescribeEngineError(
            f"{catalog.model(model_id)['name']} has no downloadable repository "
            f"for variant '{request.get('variant')}' yet."
        )
    tmpdir = tempfile.mkdtemp(prefix="aivs_ad_")
    request_path = os.path.join(tmpdir, "request.json")
    out_path = os.path.join(tmpdir, "response.json")
    payload = dict(request)
    payload["out"] = out_path
    try:
        with open(request_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except OSError as exc:
        raise DescribeEngineError(f"Could not prepare the description: {exc}") from exc

    _report(on_progress,
            "Downloading the model files..."
            if str(request.get("mode") or "") == "download"
            else "Loading the describer and sampling frames...", None)
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # The worker's stderr goes to a **file**, not a pipe.  A pipe fills up, and
    # the child then blocks writing to it: loading a model reports itself as one
    # long progress bar, updated with carriage returns and no newline in it -
    # measured on this machine, a single Qwen3-VL-4B load writes 7,639 bytes
    # before its first newline.  Draining that pipe from a thread is not enough
    # to be sure of either: a reader that decodes can stop on one byte it cannot
    # decode, and the run then parks at "Loading the describer..." with the model
    # already on the card and nothing anywhere to say why.  A file has no size
    # limit and no reader to fall behind, so the load simply finishes; the text
    # is read back - it is what explains a failure - once the process has ended.
    stderr_path = os.path.join(tmpdir, "worker-stderr.log")
    try:
        # Not a context manager on purpose: the handle goes to the worker
        # process and is closed once it has exited (below), not when this
        # function returns.
        stderr_file = open(stderr_path, "wb")  # noqa: SIM115
    except OSError as exc:
        raise DescribeEngineError(
            f"Could not prepare the description: {exc}"
        ) from exc
    try:
        process = subprocess.Popen(
            [python_exe, script, request_path],
            stdout=subprocess.PIPE, stderr=stderr_file, text=True,
            creationflags=flags,
            # A gated repository is fetched *by the worker*, so the token has to
            # be in the worker's environment: ``transformers`` and
            # ``huggingface_hub`` read ``HF_TOKEN`` there and nowhere else.
            env=hf_access.worker_environment(),
        )
    except OSError as exc:
        stderr_file.close()
        raise DescribeEngineError(
            f"Could not run the model's environment: {exc}"
        ) from exc
    streamed: list[dict[str, Any]] = []
    streamed_stdout: list[str] = []
    # A cancel has to reach a child that is sitting inside a long download or a
    # model load, where no progress line arrives for minutes: a thread waits on
    # the event and ends the process, so the button answers at once.
    killer = None
    if cancel_event is not None:
        def _kill_when_cancelled() -> None:
            while not cancel_event.wait(0.5):
                if process.poll() is not None:
                    return
            try:
                process.kill()
            except Exception:
                log.debug("could not kill the cancelled worker", exc_info=True)

        killer = threading.Thread(target=_kill_when_cancelled, daemon=True,
                                  name="aivs-describe-cancel")
        killer.start()
    try:
        for line in process.stdout or ():
            streamed_stdout.append(line)
            if line.startswith(PROGRESS_PREFIX):
                _forward_progress(line, on_progress)
            elif line.startswith(SEGMENT_PREFIX) and on_segment is not None:
                try:
                    segment = json.loads(line[len(SEGMENT_PREFIX):])
                except json.JSONDecodeError:
                    continue
                streamed.append(segment)
                try:
                    on_segment(segment)
                except Exception:
                    log.debug("segment callback failed", exc_info=True)
        process.wait()
    finally:
        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            log.debug("Ignoring a failure in _worker_response", exc_info=True)
        stderr_file.close()
    result = subprocess.CompletedProcess(
        [python_exe, script, request_path], process.returncode,
        "".join(streamed_stdout), _read_text(stderr_path),
    )

    response: dict[str, Any] = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                response = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise DescribeEngineError(
                f"Could not read the description result: {exc}"
            ) from exc
    if not response:
        detail = _worker_error_detail(result.stderr or result.stdout or "")
        note = _crash_note(result.returncode)
        if note and detail == _NO_WORKER_RESULT:
            detail = note
        elif note:
            # The exit code is the cause; what the worker printed is context.
            detail = f"{note}\n{detail}"
        raise DescribeEngineError(
            f"Audio description failed: {_explain_access(detail, repo, model_id)}"
        )
    if not response.get("ok"):
        detail = _worker_error_detail(str(response.get("error") or ""))
        raise DescribeEngineError(
            _explain_access(detail, repo, model_id) or "Audio description failed."
        )
    note = response.get("note")
    if note:
        log.info("Audio description: %s", note)
        _report(on_progress, str(note), None)
    return response, streamed


def _explain_access(detail: str, repo: str, model_id: str) -> str:
    """Swap a bare gated-model 401 for the two steps that fix it.

    The worker reports what ``huggingface_hub`` raised - ``Cannot access gated
    repo ... Please log in`` - which names no page, no panel and no account, and
    is what the user asked to have explained.  Anything that is not an access
    failure is returned untouched, so a real crash keeps its own cause.
    """
    try:
        name = str(catalog.model(model_id)["name"])
    except KeyError:  # pragma: no cover - a model that left the catalog
        name = ""
    return hf_access.access_message(detail, repo, name) or detail


def _read_text(path: str) -> str:
    """A written file back as text, whatever it turns out to contain.

    The worker's stderr is decoded leniently: its encoding is the child's
    business (a progress bar may be in any of them) and a failure message is
    worth showing even when a byte in it is not.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _forward_progress(line: str, on_progress) -> None:
    """Turn one worker ``AIVS-PROGRESS`` line into a progress callback."""
    if on_progress is None or not line:
        return
    body = line[len(PROGRESS_PREFIX):].strip()
    percent, _, message = body.partition("%|")
    try:
        fraction = float(percent) / 100.0
    except ValueError:
        fraction = None
    _report(on_progress, message.strip() or "Describing...", fraction)
