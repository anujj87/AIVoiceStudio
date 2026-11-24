"""Speech-to-text execution.

The catalog (:mod:`ai_voice_studio.stt.catalog`) says *what* a model is; this
module says *how* it runs.  Every model runs inside **its own** Python
environment (see :mod:`ai_voice_studio.stt.environments`), so two models with
conflicting dependencies - NeMo against transformers, or two different PyTorch
pins - can never break each other.  The application never imports a model's
back-end itself: it starts :mod:`ai_voice_studio.stt.worker` with the model
environment's interpreter and reads the timed segments back as JSON.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from .. import hf_access
from . import catalog, environments

log = logging.getLogger(__name__)

#: An hour is far longer than any real transcription (the models run at a small
#: multiple of real time once loaded) and still short enough that a parked
#: worker reports an error instead of holding its thread for the rest of the
#: session.  The whole worker is one call, load included.
WORKER_RUN_TIMEOUT = 60 * 60

#: Back-end id -> the importable module that proves it is installed.  Kept as a
#: module attribute for callers that only want the vocabulary; the actual check
#: is done per model environment by :mod:`environments`.
BACKEND_MODULES = catalog.BACKEND_MODULES

#: Hugging Face repositories for the repository-backed models (transformers,
#: NeMo and SpeechBrain).  A missing entry means local inference for that
#: model/variant is not wired yet.
HF_MODELS: dict[str, dict[str, str]] = {
    "qwen3_asr": {
        "1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
        "0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    },
    "voxtral": {
        "mini-3b": "mistralai/Voxtral-Mini-3B-2507",
    },
    "granite_speech": {
        "1b": "ibm-granite/granite-4.0-1b-speech",
        "2b": "ibm-granite/granite-speech-3.3-2b",
        "8b": "ibm-granite/granite-speech-3.3-8b",
    },
    "canary_qwen": {
        "2.5b": "nvidia/canary-qwen-2.5b",
        "1b": "nvidia/canary-1b-v2",
    },
    "parakeet_tdt": {
        "0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",
        "0.6b": "nvidia/parakeet-tdt-0.6b-v2",
    },
    "moonshine": {
        "base": "moonshine-ai/moonshine-base",
        "tiny": "moonshine-ai/moonshine-tiny",
    },
    "speechbrain": {
        "base": "speechbrain/asr-crdnn-rnnlm-librispeech",
        "hindi": "speechbrain/asr-whisper-large-v2-commonvoice-hi",
    },
}

#: Back-ends that are driven by a Hugging Face repository rather than by a
#: checkpoint name of their own.
REPOSITORY_BACKENDS = frozenset({"transformers", "nemo", "speechbrain"})


class STTEngineError(RuntimeError):
    """A transcription could not be run (bad input, decode failure, ...)."""


class BackendMissing(STTEngineError):
    """The model's environment, or its back-end package, is not installed."""


# ---------------------------------------------------------------------------
# Environment / availability
# ---------------------------------------------------------------------------
def env_dir(model_id: str) -> str:
    """The model's own environment folder."""
    return environments.env_dir(model_id)


def is_installed(model_id: str) -> bool:
    """True when a back-end of this model is importable in its own environment."""
    return environments.is_installed(model_id)


def env_exists(model_id: str) -> bool:
    """True when the model has its own Python environment folder."""
    return environments.is_created(model_id)


def installed_version(model_id: str) -> str | None:
    return environments.installed_version(model_id)


def install_plan(model_id: str):
    """The pip steps that install this model into its own environment."""
    return environments.install_plan(model_id)


def install(model_id: str, progress: Callable[[str], None] | None = None,
            cancel_event: threading.Event | None = None) -> dict[str, object]:
    """Install this model's back-end into its own environment."""
    return environments.install(model_id, progress=progress,
                                cancel_event=cancel_event)


def remove(model_id: str) -> str | None:
    """Delete this model's environment."""
    return environments.remove(model_id)


def model_backends(model_id: str) -> list[str]:
    """Back-ends that can run this model, best first."""
    entry = catalog.model(model_id)
    return [entry["backend"], *entry.get("alt_backends", [])]


def model_available(model_id: str) -> bool:
    """True when at least one of the model's back-ends is installed."""
    return environments.is_installed(model_id)


def missing_packages(model_id: str) -> list[str]:
    """The pip packages the user has to install to run this model."""
    return catalog.pip_packages(model_id)


def availability_note(model_id: str) -> str:
    """A one-line status for the transcribe dialog and the STT settings table."""
    if environments.is_installed(model_id):
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


def installed_backends() -> dict[str, bool]:
    """Which back-ends are installed, across every model environment.

    A back-end counts as installed when any model that can use it has its
    module present in that model's own environment.  This is a diagnostic
    helper; the decision that matters is made per model by
    :func:`environments.is_installed`.
    """
    result: dict[str, bool] = dict.fromkeys(catalog.BACKEND_MODULES, False)
    for model_id in catalog.model_ids():
        for backend in model_backends(model_id):
            if backend not in result or result[backend]:
                continue
            module = catalog.BACKEND_MODULES.get(backend)
            if module and environments.module_present(model_id, module):
                result[backend] = True
    return result


def _installed_backend(model_id: str) -> str | None:
    """The first of the model's back-ends that is installed, or ``None``."""
    for backend in model_backends(model_id):
        module = catalog.BACKEND_MODULES.get(backend)
        if module and environments.module_present(model_id, module):
            return backend
    return None


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def _resolve_device(compute: str) -> str:
    """The device the worker should put the model on.

    The speech models are pip-installed PyTorch, which brings its own CUDA
    runtime, so an NVIDIA card with a working driver is the whole requirement
    - the *optional sherpa-onnx GPU runtime* (what ``compute.resolve_compute``
    waits for) plays no part here, and requiring it would silently send a GPU
    run to the CPU on a machine that downloaded nothing from Settings.
    """
    from .. import compute as _compute

    if compute == _compute.COMPUTE_CUDA:
        return "cuda"
    if compute == _compute.COMPUTE_AUTO:
        return "cuda" if _compute.has_nvidia_gpu() else "cpu"
    return "cpu"


def cuda_available() -> bool:
    """True when the transcribe dialog may offer a GPU run at all."""
    from .. import compute as _compute

    return _compute.has_nvidia_gpu()


def compute_options(model_id: str, variant: str = "") -> list[dict[str, str]]:
    """Compute choices the dialog shows for this model and variant.

    Each choice is ``{"value", "label"}``: the catalog's
    :func:`catalog.device_options` decides which devices the model may be run
    on (both, unless the variant is GPU-only), and this adds the value the
    transcribe request expects (``cuda`` for the GPU, ``cpu`` otherwise).  The
    variant matters because one catalog entry can hold a GPU-only checkpoint
    and a CPU-friendly one.
    """
    return [
        {"value": value, "label": label}
        for value, label in catalog.device_options(
            model_id, variant=variant, has_cuda=cuda_available()
        )
    ]


def gpu_runtime_missing(model_id: str) -> list[str]:
    """CUDA runtime packages this model's environment still lacks.

    A GPU run of a Whisper model needs CTranslate2's cuBLAS runtime; when it
    is missing the worker silently transcribes on the CPU instead.  Naming the
    packages lets the dialog tell the user to repair the model, rather than
    leaving them to wonder why the GPU was never used.
    """
    return environments.missing_cuda_runtime(model_id)


def _beam_size(accuracy: str) -> int:
    return catalog.BEAM_SIZES.get(accuracy, catalog.BEAM_SIZES[catalog.DEFAULT_ACCURACY])


def _report(on_progress, message: str, fraction: float | None = None) -> None:
    if on_progress is not None:
        try:
            on_progress(message, fraction)
        except Exception:
            log.debug("transcription progress callback failed", exc_info=True)


def transcribe_audio(
    audio_path: str,
    *,
    model_id: str,
    variant: str | None = None,
    language: str = "auto",
    translate_to: str = "",
    accuracy: str = catalog.DEFAULT_ACCURACY,
    compute: str = "cpu",
    on_progress=None,
) -> list[dict[str, Any]]:
    """Transcribe one audio/video file into timed segments.

    Runs the model in **its own** environment through the worker subprocess, so
    the model's dependencies never enter the GUI process.  Returns a list of
    ``{"start", "end", "text"}``.  Raises :class:`BackendMissing` when the
    model's environment is missing or holds no back-end, and
    :class:`STTEngineError` for anything else.
    """
    if not os.path.isfile(audio_path):
        raise STTEngineError(f"Audio file not found: {audio_path}")
    entry = catalog.model(model_id)
    variant = variant or entry["default_variant"]

    if not environments.is_created(model_id):
        raise BackendMissing(
            f"{entry['name']} has no Python environment of its own yet "
            f"({environments.env_dir(model_id)}).\n\nOpen Settings > Manage "
            "the STT and install this model there."
        )
    backend = _installed_backend(model_id)
    if backend is None:
        packages = ", ".join(catalog.pip_packages(model_id))
        raise BackendMissing(
            f"{entry['name']} is not installed in its own environment "
            f"({environments.env_dir(model_id)}).\n\nInstall: {packages}"
        )

    request: dict[str, Any] = {
        "audio": audio_path,
        "backend": backend,
        "variant": variant,
        "language": language,
        "translate_to": translate_to,
        "accuracy": accuracy,
        "compute": _resolve_device(compute),
        "model": model_id,
    }
    if backend in REPOSITORY_BACKENDS:
        request["hf_model"] = HF_MODELS.get(model_id, {}).get(variant, "")
    if backend in ("transformers", "nemo"):
        # Which of the worker's strategies can run this architecture at all.  A
        # NeMo family mixes plain ASR models with speech-language models (and a
        # single entry may offer both as variants), so the hint is per variant.
        request["runner"] = catalog.runner_for(model_id, variant)
    # Never ask for timings a model cannot give: transformers raises "We cannot
    # return_timestamps yet on non-CTC models apart from Whisper!" otherwise.
    request["timestamps"] = catalog.supports_timestamps(model_id)
    request["trust_remote_code"] = catalog.needs_remote_code(model_id)
    return _run_worker(model_id, request, on_progress)


#: Fragments that mark a line as a progress bar rather than a cause.  A model
#: download prints a tqdm bar to the same stream a crash uses, so the last
#: lines of a killed worker were the *download* finishing, not the error.
_PROGRESS_MARKERS = ("%|", "it/s", "s/it", "B/s")

#: Lines a back-end prints that are never the reason a run failed: deprecation
#: and user warnings from transformers/NeMo/PyTorch.
_WARNING_MARKERS = (
    "deprecated", "futurewarning", "userwarning", "deprecationwarning",
    "warnings.warn", "[transformers]",
)

#: Log prefixes whose lines are routine chatter (NeMo's ``[NeMo I ...]`` info
#: logs, PyTorch's ``W1002`` warnings), never the cause of a failure.
_LOG_PREFIXES = ("[nemo ", "[torch", "w1002", "w1003", "a100")


def _is_noise(line: str) -> bool:
    """Whether a worker output line is progress/warning chatter, not a cause."""
    lowered = line.lower()
    if lowered.startswith("note:"):
        return True
    if any(marker in line for marker in _PROGRESS_MARKERS):
        return True
    if lowered.startswith("warning:") or any(
        marker in lowered for marker in _WARNING_MARKERS
    ):
        return True
    return lowered.startswith(_LOG_PREFIXES)


def _meaningful_lines(text: str) -> list:
    """The stripped, non-noise lines of a worker's output, in order."""
    return [
        line.strip()
        for line in (text or "").splitlines()
        if line.strip() and not _is_noise(line.strip())
    ]


def _worker_error_detail(text: str) -> str:
    """The most useful line(s) of a failed worker's output.

    A native panic (a Rust extension such as the tokenizer or safetensors
    binding blowing up, say) prints its cause and *then* the
    "note: run with RUST_BACKTRACE=1" hint, so reporting only the last line
    showed the hint and hid the cause.  The same applies to a model download's
    progress bar and to third-party deprecation warnings: they are dropped so
    the real reason is what the dialog shows.
    """
    lines = _meaningful_lines(text)
    if not lines:
        return "the worker produced no result"
    return "\n".join(lines[-3:])


def _explain_access(detail: str) -> str:
    """Swap a bare gated-model 401 for the steps that fix it.

    Mirrors the describe engine: the worker's own wording names no page and no
    panel, and the application now has a place to put a token.
    """
    return hf_access.access_message(detail) or detail


def _clean_error_message(text: str) -> str:
    """A worker-reported error with any progress-bar/warning noise removed.

    The worker writes ``str(exc)`` as its error text, and a back-end exception
    can carry captured progress output or warnings inside its message; those
    would otherwise become the dialog's headline.
    """
    lines = _meaningful_lines(text)
    if not lines:
        return "Transcription failed."
    return "\n".join(lines[-3:])


def _run_worker(model_id: str, request: dict[str, Any], on_progress) -> list[dict[str, Any]]:
    """Start the worker in the model's environment and read its JSON response."""
    try:
        script = environments.worker_script()
        # ``usable_python_exe`` (not ``python_exe``): an environment built by an
        # earlier installation of the application runs on a Python that is no
        # longer there, and is re-pointed at this one before it is used.
        python_exe = environments.runtime(model_id).usable_python_exe()
    except Exception as exc:
        raise STTEngineError(f"Could not start the model's environment: {exc}") from exc
    if not os.path.isfile(python_exe):
        raise BackendMissing(
            f"The Python environment for this model is not usable "
            f"({python_exe} is missing). Reinstall it from Settings > Manage "
            "the STT."
        )

    tmpdir = tempfile.mkdtemp(prefix="aivs_stt_")
    request_path = os.path.join(tmpdir, "request.json")
    out_path = os.path.join(tmpdir, "response.json")
    request = dict(request)
    request["out"] = out_path
    try:
        with open(request_path, "w", encoding="utf-8") as fh:
            json.dump(request, fh, ensure_ascii=False)
    except OSError as exc:
        raise STTEngineError(f"Could not prepare the transcription: {exc}") from exc

    _report(on_progress, "Loading the model and transcribing...", None)
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [python_exe, script, request_path],
            capture_output=True, text=True, creationflags=flags,
            # A gated checkpoint is fetched by the worker, so the token has to
            # be in *its* environment (transformers and huggingface_hub read
            # HF_TOKEN there and nowhere else).
            env=hf_access.worker_environment(),
            # Generous, but not unbounded: a worker that parks (a stuck CUDA
            # call, a download with no timeout) must end in an error the user
            # can act on rather than a thread that never returns.
            timeout=WORKER_RUN_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise STTEngineError(
            f"Transcription did not finish within {WORKER_RUN_TIMEOUT // 60} "
            "minutes and was stopped. Try a shorter file, or check that the "
            "model's environment is healthy."
        ) from exc
    except OSError as exc:
        raise STTEngineError(f"Could not run the model's environment: {exc}") from exc

    response: dict[str, Any] = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                response = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise STTEngineError(f"Could not read the transcription result: {exc}") from exc
    if not response:
        detail = _worker_error_detail(result.stderr or result.stdout or "")
        if result.returncode:
            detail = f"{detail}\n(the model's process exited with code {result.returncode})"
        raise STTEngineError(f"Transcription failed: {detail}")
    if not response.get("ok"):
        raise STTEngineError(_explain_access(
            _clean_error_message(str(response.get("error") or ""))
        ))
    note = response.get("note")
    if note:
        log.info("Speech to text: %s", note)
        _report(on_progress, str(note), None)
    return list(response.get("segments", []))
