"""Voice Lab: the CPU-first voice-clone engines.

Three engines live here, all of them voice cloners that run on a plain CPU and
switch to the GPU when one is present:

============ ============================ ====================================
Engine       Package                      Notes
============ ============================ ====================================
Pocket TTS   ``pocket-tts``               Kyutai, 100M params, 26 voices
Bark         ``transformers``             Suno, 120+ speaker presets
F5-TTS       ``f5-tts``                   Flow-matching transformer
============ ============================ ====================================

Each one is installed into the app's managed virtualenv and driven from a
subprocess (``voicelab.worker``), so PyTorch never enters the GUI process.  The
engine wrapper below plugs into ``tts.engine.get_engine`` and implements the
same ``synthesize`` contract as the sherpa-onnx engines.

Per-engine tuning (diffusion steps, sampling temperatures, quantization, seeds,
...) lives in ``voicelab.options``: the Settings page stores defaults per
engine, the Recording window stores per-project overrides, and both ride along
with a voice entry into the worker, which applies whatever the installed
engine version actually supports.

Nothing in this module is imported by the GUI at start-up: the heavy pieces
(the worker, the venv) are created on first use.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import engines
from . import options as _options

log = logging.getLogger(__name__)

#: Re-exported at package level so callers can write
#: ``from ..voicelab import is_engine`` (the Settings panels and the
#: Recording window do exactly that).
is_engine = engines.is_engine

#: ``kind`` used for voices the user cloned from their own recording.
CLONE_KIND = "clone_reference"

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_AUTO = "auto"

#: Settings key holding the device chosen on the Voice Clone settings page.
DEVICE_SETTING = "clone_engines.device"

#: Interpreter used to run the worker; defaults to the managed virtualenv.
_python_override = os.environ.get("AIVS_VOICELAB_PYTHON", "")


def set_worker_python(path: str) -> None:
    """Force the interpreter the worker runs with (tests / advanced setups)."""
    global _python_override
    _python_override = path or ""

_DEFAULT_SAMPLE_RATE = 24000

#: Fields copied from a voice entry into a worker request.
_VOICE_KEYS = (
    "voice",
    "language",
    "variant",
    "ref_audio",
    "ref_text",
    "ref_resource",
    "ref_text_resource",
    "repo",
)

_WORKER_NAME = "_voicelab_worker.py"


class VoicelabError(Exception):
    """Raised when a Voice Lab engine is missing or fails to synthesize."""


# ---------------------------------------------------------------------------
# Device detection / selection
# ---------------------------------------------------------------------------
_cuda_cache: Optional[bool] = None


def has_cuda(force: bool = False) -> bool:
    """True when an NVIDIA GPU with a working driver is present.

    Checked with ``nvidia-smi`` (a few milliseconds, no PyTorch import) and
    cached for the session: it only decides whether the *option* to run on the
    GPU is offered, never whether an engine works at all.
    """
    global _cuda_cache
    if _cuda_cache is not None and not force:
        return _cuda_cache
    found = False
    try:
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=flags,
        )
        found = result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        found = False
    _cuda_cache = found
    return found


def device_options() -> List[Tuple[str, str]]:
    """``(value, label)`` pairs for the device selector.

    The CPU is always offered.  When an NVIDIA GPU is detected, the GPU is
    offered *in addition* - never instead - plus an Auto entry that picks the
    GPU when one is available at synthesis time.
    """
    options: List[Tuple[str, str]] = [
        (DEVICE_CPU, "CPU (works on every computer)"),
    ]
    if has_cuda():
        options.append((DEVICE_CUDA, "GPU (CUDA - NVIDIA)"))
        options.append((DEVICE_AUTO, "Auto (GPU when available)"))
    return options


def resolve_device(choice: Optional[str]) -> str:
    """Turn a device choice (``cpu`` / ``cuda`` / ``auto``) into a concrete one."""
    value = (choice or "").strip().lower()
    if value == DEVICE_CUDA:
        return DEVICE_CUDA if has_cuda() else DEVICE_CPU
    if value == DEVICE_AUTO:
        return DEVICE_CUDA if has_cuda() else DEVICE_CPU
    if value in ("gpu", "cuda_gpu"):
        return DEVICE_CUDA if has_cuda() else DEVICE_CPU
    return DEVICE_CPU


def device_label(device: str) -> str:
    return "GPU (CUDA)" if device == DEVICE_CUDA else "CPU"


# ---------------------------------------------------------------------------
# Engine availability
# ---------------------------------------------------------------------------
def is_installed(engine_id: str) -> bool:
    """True when the engine's package is installed in its own environment.

    Every Voice Lab engine lives in its own virtualenv
    (``%APPDATA%\\AIVoiceStudio\\tts_envs\\<engine>``), so engines with
    conflicting dependencies cannot break each other.  An engine that was
    installed into the older *shared* addon environment keeps working: the
    probe falls back to it.
    """
    package = engines.probe_package(engine_id)
    if not package:
        return False
    try:
        from .. import venv_packages  # noqa: PLC0415

        if venv_packages.installed(package, engine=engine_id):
            return True
        # Engines installed before the per-TTS environments exist in the
        # shared addon environment; honour them until they are reinstalled.
        return venv_packages.installed(package)
    except Exception:  # noqa: BLE001
        return False


def installed_version(engine_id: str) -> Optional[str]:
    package = engines.probe_package(engine_id)
    if not package:
        return None
    try:
        from .. import venv_packages  # noqa: PLC0415

        return (
            venv_packages.version(package, engine=engine_id)
            or venv_packages.version(package)
        )
    except Exception:  # noqa: BLE001
        return None


def install_plan(engine_id: str) -> List[Tuple[str, Optional[str]]]:
    """The pip steps that install an engine into its own environment.

    Returns ``[(packages, index_url_or_None), ...]``.  On a machine with an
    NVIDIA GPU the CUDA build of PyTorch is installed *first*, from the
    official PyTorch wheel index; the remaining packages then come from PyPI -
    two steps, because the CUDA index does not mirror the other packages.
    (PyPI's plain ``torch`` wheel is CPU-only on Windows, so a GPU run needs
    the wheels from this index.)

    The engine's compatibility pins ride along in the second step, so pip
    resolves them together with the engine's own dependencies (see
    ``engines.pins``).
    """
    from .. import venv_packages  # noqa: PLC0415

    packages = list(packages_for(engine_id))
    steps: List[Tuple[List[str], Optional[str]]] = []
    cuda = list(engines.cuda_packages(engine_id))
    index = engines.cuda_index_url(engine_id)
    if cuda and index and has_cuda() and not venv_packages.installed(
        "torch", engine=engine_id
    ):
        steps.append((cuda, index))
        packages = [p for p in packages if p not in cuda]
    if packages:
        steps.append((packages, None))
    return steps


def packages_for(engine_id: str) -> Tuple[str, ...]:
    """The ``pip`` packages this engine needs in its own environment.

    The same list feeds "Install engine..." and "Remove engine", so it holds
    plain names only - never ``name<version`` requirements, which
    ``pip uninstall`` cannot take.
    """
    return engines.packages(engine_id)


def verify_engine_import(engine_id: str, timeout: float = 300.0) -> Optional[str]:
    """Import the engine in its own environment; return a problem, or ``None``.

    Installing the packages is not the same as having a working engine: pip
    can resolve a dependency combination that the engine's own code cannot
    import.  Running the import once, right after the install, turns that into
    a message the user sees immediately instead of a puzzling failure on the
    first preview.
    """
    modules = engines.import_modules(engine_id)
    if not modules:
        return None
    try:
        from ..python_runtime import get_runtime  # noqa: PLC0415

        runtime = get_runtime(engine_id)
        if not runtime.is_created:
            return None
        python_exe = runtime.python_exe
    except Exception as exc:  # noqa: BLE001
        return f"Could not start the engine's Python environment: {exc}"
    if not os.path.isfile(python_exe):
        return None
    script = "import " + ", ".join(modules)
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Loading {engines.engine_name(engine_id)} took too long "
            f"({int(timeout)}s) and was stopped."
        )
    except OSError as exc:
        return f"Could not run the engine's Python environment: {exc}"
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    last = detail[-1] if detail else f"exit code {result.returncode}"
    return f"{engines.engine_name(engine_id)} is installed but failed to load: {last}"


def installed_engines() -> List[Dict[str, Any]]:
    """Metadata of every engine whose package is installed."""
    return [info for info in engines.ENGINES if is_installed(info["id"])]


# ---------------------------------------------------------------------------
# Built-in voices
# ---------------------------------------------------------------------------
def builtin_voice_entries(engine_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Voice entries for the pre-made voices usable right now.

    Only engines whose package is installed contribute entries: a built-in
    voice is offered exactly when the engine that speaks it is ready.
    """
    ids = (engine_id,) if engine_id else engines.engine_ids()
    entries: List[Dict[str, Any]] = []
    for engine_key in ids:
        if not is_installed(engine_key):
            continue
        entries.extend(engines.builtin_voices(engine_key))
    return entries


# ---------------------------------------------------------------------------
# Cloned voices (user recordings)
# ---------------------------------------------------------------------------
def clone_voices(store, engine_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Custom voices the user cloned with a Voice Lab engine."""
    try:
        voices = store.custom_voices(engine_id) if engine_id else store.custom_voices()
    except Exception:  # noqa: BLE001
        return []
    return [v for v in voices if v.get("kind") == CLONE_KIND]


def create_clone(
    store,
    engine_id: str,
    name: str,
    ref_audio: str,
    ref_text: str = "",
    language: str = "en",
    variant: str = "",
) -> Dict[str, Any]:
    """Register a cloned voice from a reference recording.

    The audio file is copied into the voices folder so the clone keeps working
    when the original file is moved or deleted.  Returns the stored entry.
    """
    if not engines.is_engine(engine_id):
        raise ValueError(f"'{engine_id}' is not a Voice Lab engine.")
    voice_name = (name or "").strip()
    if not voice_name:
        raise ValueError("Type a voice name first.")
    source = (ref_audio or "").strip()
    if not source or not os.path.isfile(source):
        raise ValueError("Choose a reference audio file first (Browse...).")
    info = engines.engine(engine_id) or {}
    if engine_id == "bark":
        if not source.lower().endswith(".npz"):
            raise ValueError(
                "Bark clones voices from a speaker-embedding file (.npz). "
                "Choose a .npz file, or use one of Bark's built-in speakers."
            )
    directory = store.custom_dir(voice_name)
    extension = os.path.splitext(source)[1].lower() or ".wav"
    destination = os.path.join(directory, "reference" + extension)
    try:
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise ValueError(f"Could not copy the reference audio: {exc}") from exc
    variants = [
        variant_entry["id"]
        for lang in info.get("languages", [])
        for variant_entry in lang.get("variants", [])
    ]
    entry = {
        "engine": engine_id,
        "language": language or "en",
        "variant": variant or (variants[0] if variants else "default"),
        "voice": engines.CLONE_VOICE_ID,
        "ref_audio": destination,
        "ref_text": (ref_text or "").strip(),
        "source_audio": source,
        "mode": "clone",
    }
    store.add_custom_voice(
        voice_name, engine_id, directory, CLONE_KIND, extra=entry
    )
    for stored in store.custom_voices(engine_id):
        if stored.get("name") == voice_name:
            return stored
    return {"name": voice_name, "tts": engine_id, **entry}


def rename_clone(store, old_name: str, new_name: str) -> bool:
    return bool(store.rename_custom_voice(old_name, new_name))


def delete_clone(store, name: str) -> bool:
    return bool(store.remove_custom_voice(name))


# ---------------------------------------------------------------------------
# Worker request builder (pure, unit-testable)
# ---------------------------------------------------------------------------
def cpu_thread_count() -> int:
    """Threads a CPU run may use (80-95% of the machine, see ``compute``)."""
    try:
        from .. import compute  # noqa: PLC0415

        return compute.cpu_threads()
    except Exception:  # noqa: BLE001
        return 0


def build_synthesize_request(
    *,
    engine_id: str,
    device: str,
    text: str,
    voice_entry: Optional[Dict[str, Any]] = None,
    speed: float = 1.0,
    tuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON request sent to the Voice Lab worker.

    ``tuning`` holds the per-engine generation overrides (``voicelab.options``);
    entries equal to the engine's own default are dropped, so an untouched
    project sends no options at all.  A voice entry may carry them itself
    instead - the Recording window does exactly that.
    """
    entry = voice_entry or {}
    resolved = resolve_device(device)
    request: Dict[str, Any] = {
        "cmd": "synthesize",
        "engine": engine_id,
        "device": resolved,
        "text": text,
        "speed": float(speed),
        # How much of the machine a run may use: the worker applies this with
        # torch.set_num_threads (CPU) or maximizes the CUDA card (GPU).
        "threads": cpu_thread_count(),
    }
    for key in _VOICE_KEYS:
        value = entry.get(key)
        if value:
            request[key] = value
    if engine_id == "bark" and not request.get("repo"):
        # suno/bark-small is English-only but much lighter on a CPU: honour the
        # variant the user picked.
        request["repo"] = (
            "suno/bark-small" if entry.get("variant") == "bark_small" else "suno/bark"
        )
    overrides = _options.clean(engine_id, tuning)
    if not overrides:
        # The voice entry is the fallback carrier of the options.
        overrides = _options.clean(engine_id, entry.get("options"))
    if overrides:
        request["options"] = overrides
    return request


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------
def _bundled_worker_path() -> Optional[str]:
    """Path of ``worker.py`` on disk, when it is a real (readable) file.

    In a frozen build this module lives inside PyInstaller's PYZ archive, so
    ``__file__`` does not exist; the packaging spec copies ``worker.py`` next to
    the other data files instead (see ``packaging/ai_voice_studio.spec``).
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"),
    ]
    bundle_dir = getattr(sys, "_MEIPASS", "")
    if bundle_dir:
        candidates.append(
            os.path.join(bundle_dir, "ai_voice_studio", "voicelab", "worker.py")
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _read_worker_source() -> Optional[str]:
    """Worker script source: from disk first, then the embedded copy."""
    path = _bundled_worker_path()
    if path:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return _EMBEDDED_WORKER_SOURCE or None


def _worker_script_path(env_dir: str) -> str:
    """A ``worker.py`` the managed virtualenv's interpreter can run.

    The bundled file is used directly when present; otherwise the source is
    written next to the venv (a frozen app keeps ``worker.py`` inside
    PyInstaller's PYZ archive, which another interpreter cannot read).
    """
    bundled = _bundled_worker_path()
    if bundled:
        return bundled
    script = os.path.join(env_dir, _WORKER_NAME)
    source = _read_worker_source()
    if not source:
        raise VoicelabError(
            "The Voice Lab worker script is missing from this installation. "
            "Reinstall the application."
        )
    try:
        needs_write = True
        if os.path.isfile(script):
            with open(script, encoding="utf-8") as fh:
                needs_write = fh.read() != source
        if needs_write:
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(source)
    except OSError as exc:
        raise VoicelabError(f"Could not prepare the Voice Lab worker: {exc}") from exc
    return script


class CloneWorker:
    """A persistent worker subprocess for one engine + device.

    The worker loads its model once and answers ``synthesize`` requests over a
    JSON-lines pipe, so consecutive segments do not pay the model-load cost.
    """

    def __init__(self, engine_id: str, device: str = DEVICE_CPU,
                 timeout: float = 1800.0):
        self.engine_id = engine_id
        self.device = resolve_device(device)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._timeout = timeout
        self._next_id = 0

    # -- lifecycle ----------------------------------------------------------
    def _runtime(self):
        """The interpreter environment for this engine.

        Its own per-TTS environment when the engine lives there, else the
        shared addon environment (engines installed before the per-TTS
        environments existed).

        The check reads each environment's ``site-packages`` **directly**
        (``venv_packages.package_present``) instead of asking the cached
        package probe: the answer has to be right the first time.  A probe
        that has not finished yet would otherwise send the worker into the
        shared environment, where it dies with a baffling
        "ModuleNotFoundError: No module named 'torch'" - the module the
        engine's *own* environment does have.
        """
        from .. import venv_packages  # noqa: PLC0415
        from ..python_runtime import get_runtime  # noqa: PLC0415

        required = tuple(engines.import_modules(self.engine_id)) or (
            engines.probe_package(self.engine_id),
        )
        required = tuple(name for name in required if name)
        engine_runtime = get_runtime(self.engine_id)
        if engine_runtime.is_created and (
            not required
            or all(
                venv_packages.package_present(name, engine=self.engine_id)
                for name in required
            )
        ):
            return engine_runtime
        runtime = get_runtime()
        if not runtime.is_created:
            runtime.ensure_env()
        if not required or all(venv_packages.package_present(name) for name in required):
            return runtime
        raise VoicelabError(
            f"{engines.engine_name(self.engine_id)} is not installed in its "
            "own Python environment any more. Install it again from "
            "Settings > Voice Clone."
        )

    def _venv_python(self) -> str:
        if _python_override and os.path.isfile(_python_override):
            return _python_override
        return self._runtime().python_exe

    def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        runtime = self._runtime()
        python_exe = runtime.python_exe
        script = _worker_script_path(runtime.env_dir)
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                [python_exe, script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=flags,
            )
        except OSError as exc:
            raise VoicelabError(
                f"Could not start the {engines.engine_name(self.engine_id)} "
                f"worker: {exc}"
            ) from exc
        response = self._request({"cmd": "ping"})
        if not response.get("ok"):
            raise VoicelabError(
                response.get("error") or "The Voice Lab worker failed to start."
            )

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- protocol -----------------------------------------------------------
    def _request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise VoicelabError("The Voice Lab worker is not running.")
            payload = dict(request)
            payload["id"] = self._next_id
            self._next_id += 1
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise VoicelabError(f"The Voice Lab worker failed: {exc}") from exc
            if not line:
                raise VoicelabError(
                    "The Voice Lab worker stopped. Install the engine again "
                    "from Settings > Voice Clone."
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise VoicelabError(
                    "Invalid response from the Voice Lab worker."
                ) from exc

    # -- public API ---------------------------------------------------------
    def available_engines(self) -> Dict[str, bool]:
        """Which engines the worker can import (``{"pocket_tts": True, ...}``)."""
        self._ensure_proc()
        return dict(self._request({"cmd": "ping"}).get("engines") or {})

    def available_voices(self, engine_id: Optional[str] = None) -> Dict[str, bool]:
        """Which built-in voices of an engine are usable right now."""
        self._ensure_proc()
        if engine_id and engine_id != self.engine_id:
            worker = worker_for(engine_id, self.device)
            return worker.available_voices(engine_id)
        response = self._request({"cmd": "voices", "engine": self.engine_id,
                                  "device": self.device})
        if not response.get("ok"):
            raise VoicelabError(response.get("error") or "Could not list voices.")
        return dict(response.get("available") or {})

    def synthesize(self, request: Dict[str, Any]):
        """Send a synthesize request.

        Returns ``(int16 samples, sample rate, report)`` where ``report`` lists
        the tuning options the engine applied and those it had to skip.
        """
        self._ensure_proc()
        response = self._request(request)
        if not response.get("ok"):
            raise VoicelabError(response.get("error") or "Synthesis failed.")
        samples = np.frombuffer(
            base64.b64decode(response["wav"]), dtype=np.int16
        ).copy()
        try:
            sample_rate = int(response.get("sample_rate") or _DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            sample_rate = _DEFAULT_SAMPLE_RATE
        report = {
            "applied": list(response.get("applied") or []),
            "skipped": list(response.get("skipped") or []),
            "device_used": response.get("device_used"),
            # Explanations from the engine process (for example "audio is
            # decoded with soundfile because there is no FFmpeg here").
            "notes": list(response.get("notes") or []),
        }
        return samples, sample_rate, report


_worker_lock = threading.RLock()
_workers: Dict[Tuple[str, str], CloneWorker] = {}


def worker_for(engine_id: str, device: str = DEVICE_CPU) -> CloneWorker:
    """Shared worker per ``(engine, device)``: models stay loaded."""
    resolved = resolve_device(device)
    key = (engine_id, resolved)
    with _worker_lock:
        worker = _workers.get(key)
        if worker is None:
            worker = CloneWorker(engine_id, resolved)
            _workers[key] = worker
        return worker


def close_workers() -> None:
    """Shut every worker down (used by tests and on application exit)."""
    with _worker_lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        worker.close()


# ---------------------------------------------------------------------------
# Engine wrapper (drops into the app's get_engine interface)
# ---------------------------------------------------------------------------
class VoicelabEngine:
    """Wraps a Voice Lab worker so it plugs into the TTS engine cache.

    Implements the same ``synthesize`` contract as ``TtsEngine``: returns int16
    samples at ``sample_rate``, honouring the pitch/volume helpers the app
    applies to every engine.
    """

    def __init__(self, voice_entry: Dict[str, Any], device: str = DEVICE_CPU):
        engine_id = voice_entry.get("engine")
        if not engines.is_engine(engine_id):
            raise VoicelabError(f"'{engine_id}' is not a Voice Lab engine.")
        self.voice_entry = voice_entry
        self.engine_id = engine_id
        #: Per-engine tuning overrides carried by this voice entry.
        self.tuning = _options.clean(engine_id, voice_entry.get("options"))
        self.device = resolve_device(
            voice_entry.get("device") or device or DEVICE_CPU
        )
        self.sample_rate = _DEFAULT_SAMPLE_RATE
        self._worker = worker_for(engine_id, self.device)

    # -- synthesis ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        request = build_synthesize_request(
            engine_id=self.engine_id,
            device=self.device,
            text=text,
            voice_entry=self.voice_entry,
            speed=speed,
            tuning=self.tuning,
        )
        samples, sample_rate, report = self._worker.synthesize(request)
        self._report_unused_options(report)
        self._report_notes(report)
        if sample_rate:
            self.sample_rate = sample_rate
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415

            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415

            samples = _apply_volume(samples, volume)
        return samples

    @staticmethod
    def _report_unused_options(report: Optional[Dict[str, Any]]) -> None:
        """Log the tuning values the installed engine could not apply.

        Never fatal: an older/newer engine version may simply not have the
        knob, and the recording should still be produced with the default.
        """
        skipped = list((report or {}).get("skipped") or [])
        if skipped:
            log.warning(
                "The installed engine does not accept these tuning options "
                "(they were skipped): %s", ", ".join(sorted(skipped))
            )

    @staticmethod
    def _report_notes(report: Optional[Dict[str, Any]]) -> None:
        """Log the explanations the engine process sent back.

        A note means the run took a different route than the obvious one - for
        example audio was decoded with soundfile because this computer has no
        FFmpeg - which is worth putting in the log even though it is not an
        error.
        """
        for note in list((report or {}).get("notes") or []):
            log.info("Voice Lab engine note: %s", note)

    def close(self) -> None:
        self._worker.close()


# ---------------------------------------------------------------------------
# Embedded worker source (filled in by tools/build with the file contents)
# ---------------------------------------------------------------------------
#: Kept as a marker: when empty, a frozen build reads the worker from the
#: bundled data file instead (see ``packaging/ai_voice_studio.spec``).
_EMBEDDED_WORKER_SOURCE = ""


def set_embedded_worker_source(source: str) -> None:
    """Used by the packaging step (and tests) to embed the worker source."""
    global _EMBEDDED_WORKER_SOURCE
    _EMBEDDED_WORKER_SOURCE = source or ""


def worker_source_available() -> bool:
    """True when a worker script can be handed to the venv interpreter."""
    return bool(_bundled_worker_path() or _EMBEDDED_WORKER_SOURCE)
