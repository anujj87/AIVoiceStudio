"""OmniVoice TTS engine (omnivoice-triton).

OmniVoice is a massively multilingual zero-shot text-to-speech (TTS) model
supporting over 600 languages, with voice cloning and voice design.  It runs
in a separate worker subprocess so its PyTorch/CUDA runtime never conflicts
with the main app's sherpa-onnx.

The ``omnivoice-triton`` package must be installed separately::

    pip install omnivoice-triton

The worker supports several inference modes (selected at model load time,
``mode`` in the synthesize request).  ``triton`` (stable) and ``hybrid``
(fastest; known VRAM-leak bug) are the well-known ones; extra modes are
accepted when the installed package provides them.

Enabled OmniVoice features (see ``omnivoice/spec.py`` for the vocabulary):

* **Auto voice** - plain text synthesis with no reference.
* **Voice design** - ``instruct`` attribute descriptions (gender, age,
  pitch, accent, style, dialect).  Catalog voices such as ``female`` /
  ``male`` are mapped to design instructions automatically.
* **Voice cloning** - ``ref_audio`` (3-15 s sample) plus optional
  ``ref_text`` (omitted -> Whisper auto-transcribes inside the engine).
* **Generation knobs** - ``num_step``, ``guidance_scale``,
  ``class_temperature``, ``duration``, ``seed`` and a ``language`` hint are
  forwarded to the installed runner only when it supports them; anything it
  cannot apply is reported in the worker response's ``skipped`` list.
* **Speed** - applied natively when supported, otherwise approximated by
  resampling in the worker (reported in ``warnings``).
* **Non-verbal symbols & pronunciation hints** - passed through untouched
  (``[laughter]``, pinyin/CMU corrections), as OmniVoice reads them inline.

Flow (Settings -> OmniVoice):
1. Check that ``omnivoice-triton`` is importable (``pip install``).
2. First synthesis triggers model download (~2 GB from HuggingFace).
3. The model stays loaded in the worker subprocess for fast subsequent calls.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import threading
from typing import Dict, Optional

import numpy as np

from .. import omnivoice_quality as quality

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # OmniVoice output sample rate

#: This engine's id: it names the OmniVoice Python environment
#: (``%APPDATA%/AIVoiceStudio/tts_envs/omnivoice``).  The direct engine and the
#: HTTP server are the same model behind two front-ends and share this one
#: environment; every other TTS engine gets an environment of its own, so a
#: conflicting PyTorch/transformers release can never break it.
ENGINE_ID = "omnivoice"


# ---------------------------------------------------------------------------
# Embedded worker source (standalone script, no ai_voice_studio imports).
# Used in frozen apps where PyInstaller compiles .py into a PYZ archive
# that the managed venv's Python cannot read.
#
# IMPORTANT: keep this string functionally identical to ``worker.py``
# (the file read in development mode).  It is written without type
# annotations so it stays valid on any Python the managed venv may hold.
# ---------------------------------------------------------------------------
_WORKER_SOURCE = r'''"""OmniVoice TTS worker subprocess (standalone).

Protocol (JSON lines on stdin/stdout):
  {"cmd": "ping"}                        -> {"ok": true, "engine": true}
  {"cmd": "synthesize", "text": "..."}    -> {"ok": true, "wav": "<base64>", ...}
  {"cmd": "quit"}                         -> (process exits)
"""
import base64
import inspect
import json
import sys
import numpy as np


def _resample_speed(samples, speed):
    """Approximate ``speed`` by linear resampling (pitch changes slightly)."""
    if speed <= 0.01:
        return samples
    n_out = max(1, int(round(len(samples) / speed)))
    if n_out == len(samples):
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _supported_kwargs(method, kwargs):
    if not kwargs:
        return {}, []
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return dict(kwargs), []
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    supported = {}
    skipped = []
    for name, value in kwargs.items():
        if name in params:
            supported[name] = value
        else:
            skipped.append(name)
    return supported, skipped


def _language(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("auto", "automatic", "any"):
        return None
    return text


def main():
    out = sys.stdout
    err = sys.stderr
    runner = None

    def respond(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

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
            try:
                from omnivoice_triton import create_runner  # noqa: PLC0415
                respond({"ok": True, "engine": True})
            except ImportError as exc:
                respond({
                    "ok": True, "engine": False,
                    "error": "OmniVoice engine not installed.\n" + str(exc),
                })
            continue

        if cmd == "synthesize":
            try:
                if runner is None:
                    from omnivoice_triton import create_runner  # noqa: PLC0415
                    mode = req.get("mode", "triton")
                    runner = create_runner(mode)
                    runner.load_model()

                text = req["text"]
                speed = float(req.get("speed", 1.0))
                language = _language(req.get("language"))
                num_step = req.get("num_step", req.get("num_steps"))
                instruct = req.get("instruct", "")
                ref_audio = req.get("ref_audio", "")
                ref_text = req.get("ref_text", "")

                options = {}
                if language is not None:
                    options["language"] = language
                if num_step is not None:
                    options["num_step"] = int(num_step)
                if req.get("guidance_scale") is not None:
                    options["guidance_scale"] = float(req["guidance_scale"])
                if req.get("class_temperature") is not None:
                    options["class_temperature"] = float(req["class_temperature"])
                if req.get("duration") is not None:
                    options["duration"] = float(req["duration"])
                if req.get("seed") is not None:
                    options["seed"] = int(req["seed"])
                if speed != 1.0:
                    options["speed"] = speed

                if ref_audio:
                    method = getattr(runner, "generate_voice_clone", None)
                    base = {"text": text, "ref_audio": ref_audio}
                    if ref_text:
                        base["ref_text"] = ref_text
                    mode_name = "voice clone"
                elif instruct:
                    method = getattr(runner, "generate_voice_design", None)
                    base = {"text": text, "instruct": instruct}
                    mode_name = "voice design"
                else:
                    method = getattr(runner, "generate", None)
                    base = {"text": text}
                    mode_name = "auto voice"
                if method is None:
                    raise RuntimeError(
                        "The installed OmniVoice runner has no %s "
                        "generation method." % mode_name
                    )

                supported, skipped = _supported_kwargs(
                    method, dict(base, **options)
                )

                saved_stdout = sys.stdout
                sys.stdout = err
                try:
                    result = method(**supported)
                finally:
                    sys.stdout = saved_stdout

                audio = result["audio"]
                sample_rate = result.get("sample_rate", 24000)
                peak_vram_gb = result.get("peak_vram_gb", 0)

                arr = np.asarray(audio, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    raise ValueError("The engine returned no audio.")
                samples = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)

                warnings = []
                if speed != 1.0 and "speed" not in supported:
                    samples = _resample_speed(samples, speed)
                    warnings.append(
                        "speed applied approximately by resampling "
                        "(the installed runner has no native speed control)"
                    )
                if "duration" in skipped:
                    warnings.append(
                        "fixed duration is not supported by the installed runner; ignored"
                    )

                wav_b64 = base64.b64encode(samples.tobytes()).decode("ascii")
                time_s = result.get("time_s")
                if time_s is None:
                    time_ms = result.get("time_ms", 0)
                else:
                    time_ms = int(round(float(time_s) * 1000.0))
                respond({
                    "ok": True,
                    "wav": wav_b64,
                    "sample_rate": sample_rate,
                    "time_ms": time_ms,
                    "peak_vram_gb": peak_vram_gb,
                    "mode_used": mode_name,
                    "skipped": sorted(set(skipped)),
                    "warnings": warnings,
                })
            except Exception as exc:
                respond({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
            continue

        respond({"ok": False, "error": "Unknown command: %s" % cmd})

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _read_worker_source() -> str:
    """Return the worker script source.

    In development mode, reads ``worker.py`` from disk.  In a frozen
    app, returns the embedded ``_WORKER_SOURCE`` string (PyInstaller
    compiles .py into the PYZ archive which the venv Python can't read).
    """
    if getattr(sys, "frozen", False):
        return _WORKER_SOURCE
    worker_py = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "worker.py",
    )
    with open(worker_py, encoding="utf-8") as fh:
        return fh.read()


class OmniVoiceError(Exception):
    """Raised when the OmniVoice engine is not installed or fails."""


# ---------------------------------------------------------------------------
# CUDA detection (lightweight -- does NOT import torch)
# ---------------------------------------------------------------------------
def has_nvidia_gpu() -> bool:
    """Return True when an NVIDIA GPU with a working driver is detected."""
    try:
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=flags,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def is_available() -> bool:
    """True when the ``omnivoice_triton`` package is importable from this
    engine's own virtualenv (checked by the worker subprocess)."""
    try:
        from ..python_runtime import engine_runtime  # noqa: PLC0415
        rt = engine_runtime(ENGINE_ID)
        if not rt.is_created:
            return False
        result = rt.run_in_env(
            "import importlib.metadata; "
            "print(importlib.metadata.version('omnivoice-triton'))"
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Worker request builder (pure, unit-testable)
# ---------------------------------------------------------------------------
def build_synthesize_request(
    *,
    text: str,
    mode: str = "triton",
    speed: float = 1.0,
    language: Optional[str] = None,
    num_step: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    class_temperature: Optional[float] = None,
    duration: Optional[float] = None,
    seed: Optional[int] = None,
    ref_audio: str = "",
    ref_text: str = "",
    instruct: str = "",
) -> Dict:
    """Build the JSON request the app sends to the OmniVoice worker.

    The worker itself decides which parameters the installed runner supports,
    so this builder only ever includes values the caller explicitly set.
    """
    from . import spec  # noqa: PLC0415

    req: Dict = {
        "cmd": "synthesize",
        "text": text,
        "mode": mode,
        "speed": float(speed),
        "language": spec.clean_language(language),
    }
    if num_step is not None:
        req["num_step"] = int(num_step)
    if guidance_scale is not None:
        req["guidance_scale"] = float(guidance_scale)
    if class_temperature is not None:
        req["class_temperature"] = float(class_temperature)
    if duration is not None:
        req["duration"] = float(duration)
    if seed is not None:
        req["seed"] = int(seed)
    if ref_audio:
        req["ref_audio"] = ref_audio
        if ref_text:
            req["ref_text"] = ref_text
    if instruct:
        req["instruct"] = instruct
    return req


# ---------------------------------------------------------------------------
# Worker subprocess management (mirrors clone/__init__.py CloneClient)
# ---------------------------------------------------------------------------
class OmniVoiceWorker:
    """Manages the persistent OmniVoice worker subprocess.

    The worker loads the OmniVoice model once (via omnivoice-triton) and
    answers ``synthesize`` requests over a JSON-lines pipe.  PyTorch/CUDA
    stay isolated in the subprocess.
    """

    def __init__(self, mode: str = "triton", timeout: float = 600.0):
        """
        Parameters
        ----------
        mode : str
            Inference mode: ``"triton"`` (stable), ``"hybrid"`` (fastest but
            may leak VRAM) or any other mode the installed package offers.
        timeout : float
            Seconds to wait for a single worker response.
        """
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._mode = mode
        self._timeout = timeout
        self._next_id = 0

    # -- lifecycle ----------------------------------------------------------

    def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # In a frozen app, sys.executable is the .exe -- use the engine's
        # own environment's Python instead so it can find omnivoice-triton.
        if getattr(sys, "frozen", False):
            from ..python_runtime import engine_runtime  # noqa: PLC0415
            rt = engine_runtime(ENGINE_ID)
            worker_python = rt.python_exe
            # The worker.py source is inside PyInstaller's PYZ archive
            # and invisible to the venv Python.  Write it as a standalone
            # script so the venv Python can run it directly.
            worker_script = os.path.join(rt.env_dir, "_omnivoice_worker.py")
            if not os.path.isfile(worker_script):
                src = _read_worker_source()
                with open(worker_script, "w", encoding="utf-8") as fh:
                    fh.write(src)
            cmd = [worker_python, worker_script]
        else:
            worker_python = sys.executable
            # In dev mode, point at the actual worker.py file.
            worker_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "worker.py",
            )
            cmd = [worker_python, worker_script]
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            creationflags=flags,
        )
        resp = self._request({"cmd": "ping"})
        if not resp.get("ok"):
            raise OmniVoiceError(
                resp.get("error", "The OmniVoice worker failed to start.")
            )

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    self._proc.kill()
                self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- protocol -----------------------------------------------------------

    def _request(self, req: dict) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise OmniVoiceError("The OmniVoice worker is not running.")
            req = dict(req)
            req["id"] = self._next_id
            self._next_id += 1
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise OmniVoiceError(f"The OmniVoice worker failed: {exc}") from exc
            if not line:
                raise OmniVoiceError(
                    "The OmniVoice worker stopped. "
                    "Is omnivoice-triton installed? (pip install omnivoice-triton)"
                )
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OmniVoiceError(
                    "Invalid response from the OmniVoice worker."
                ) from exc
            return resp

    # -- public API ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        ref_audio: str = "",
        ref_text: str = "",
        instruct: str = "",
        language: Optional[str] = None,
        num_step: Optional[int] = None,
        num_steps: Optional[int] = None,  # legacy alias
        guidance_scale: Optional[float] = None,
        class_temperature: Optional[float] = None,
        duration: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Synthesize text; returns int16 samples at 24 kHz.

        Parameters
        ----------
        text : str
            Text to synthesize.
        speed : float
            Speaking speed (0.5-2.0); applied natively when the installed
            runner supports it, otherwise approximated by resampling.
        ref_audio : str
            Path to reference audio for voice cloning (3-15 s optimal).
        ref_text : str
            Transcript of reference audio (optional; when cloning without it
            the engine auto-transcribes via Whisper).
        instruct : str
            Voice design instruction, e.g. ``"female, low pitch, british accent"``.
        language : str
            Language hint (``None``/``"auto"`` = auto-detect, ``"en"``, ``"zh"``, ...).
        num_step / num_steps : int
            Diffusion steps (16 = faster, 32 = balanced, 64 = best).
        guidance_scale : float
            CFG strength 0-10.
        class_temperature : float
            Token sampling temperature 0-2 (0 = greedy).
        duration : float
            Fixed output duration in seconds, when the runner supports it.
        seed : int
            RNG seed for reproducible output, when the runner supports it.
        """
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        req = build_synthesize_request(
            text=text,
            mode=self._mode,
            speed=float(speed),
            language=language,
            num_step=num_step if num_step is not None else num_steps,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
            duration=duration,
            seed=seed,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
        )
        resp = self._request(req)
        if not resp.get("ok"):
            raise OmniVoiceError(resp.get("error", "Synthesis failed."))
        # Surface version-adaptive behaviour: parameters the installed runner
        # could not apply and approximate fallbacks (e.g. resampled speed).
        skipped = resp.get("skipped") or []
        warnings = resp.get("warnings") or []
        if skipped or warnings:
            log.info(
                "OmniVoice worker (%s): skipped=%s warnings=%s",
                resp.get("mode_used") or self._mode,
                sorted(set(skipped)),
                warnings,
            )
        raw = base64.b64decode(resp["wav"])
        samples = np.frombuffer(raw, dtype=np.int16).copy()
        return samples


# ---------------------------------------------------------------------------
# Engine wrapper (drops into the app's get_engine interface)
# ---------------------------------------------------------------------------
class OmniVoiceEngine:
    """Wraps the OmniVoice worker so it plugs into the TTS engine cache.

    Implements the same ``synthesize`` contract as ``TtsEngine`` so the
    recording worker and preview buttons need no changes: returns int16
    samples at ``sample_rate`` (24 kHz).

    The voice entry may carry an ``omni`` dict (see ``spec.apply_omni_to_voice``)
    with the per-project voice mode, reference audio / design instructions,
    language hint and generation knobs.  When it is absent the engine behaves
    exactly like before (auto voice with the runner's defaults), except that
    catalog design voices (``female``, ``male``, ...) now map to real
    voice-design instructions.
    """

    sample_rate = _SAMPLE_RATE

    def __init__(
        self,
        voice_entry: dict,
        worker: OmniVoiceWorker | None = None,
        drone_attempts: int = quality.DRONE_ATTEMPTS,
        repair_attempts: int = quality.DRONE_REPAIR_ATTEMPTS,
    ):
        from . import spec  # noqa: PLC0415

        self.voice_entry = voice_entry
        variant = voice_entry.get("variant", "triton")
        self._worker = worker or OmniVoiceWorker(mode=variant)
        self._worker._ensure_proc()

        omni = voice_entry.get("omni") or {}
        self.language = spec.clean_language(
            omni.get("language") or voice_entry.get("language")
        )
        self.instruct = spec.resolve_instruct(
            "omnivoice",
            voice_entry.get("voice"),
            omni.get("instruct") or voice_entry.get("instruct"),
        )
        self.ref_audio = (
            omni.get("ref_audio") or voice_entry.get("ref_audio") or ""
        ).strip()
        self.ref_text = (
            omni.get("ref_text") or voice_entry.get("ref_text") or ""
        ).strip()
        # Advanced generation knobs (only sent when explicitly set).
        self.num_step = omni.get("num_step")
        self.guidance_scale = omni.get("guidance_scale")
        self.class_temperature = omni.get("class_temperature")
        self.duration = omni.get("duration")
        self.seed = omni.get("seed")
        # Drone ("no speech") recovery: this engine is the same model as the
        # HTTP server with a different front-end, and it makes the same bad
        # draw, so it gets the same repair policy (``omnivoice_quality``).
        self.drone_attempts = max(1, int(drone_attempts))
        self.repair_attempts = max(1, int(repair_attempts))
        #: What the last segment needed fixing (see ``_repair_message``), so a
        #: caller can warn the user instead of shipping a droning take.
        self.last_repairs: list = []
        self.last_warnings: list = []

    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        """Synthesize text; returns int16 samples at 24 kHz."""
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        samples = self._synthesize_checked(text, float(speed))
        self._report_quality(samples)
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415

            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415

            samples = _apply_volume(samples, volume)
        return samples

    def _draw(self, text: str, speed: float, seed):
        """One worker request, shaped for the shared drone policy.

        Returns ``(samples, server_flagged)`` — the direct engine has no server
        to flag a bad take, so its own audio is the only evidence.
        """
        samples = self._worker.synthesize(
            text=text,
            speed=speed,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            instruct=self.instruct,
            language=self.language,
            num_step=self.num_step,
            guidance_scale=self.guidance_scale,
            class_temperature=self.class_temperature,
            duration=self.duration,
            seed=seed,
        )
        return samples, False

    def _synthesize_checked(self, text: str, speed: float) -> np.ndarray:
        """One segment's text as samples, without the OmniVoice drone.

        The direct engine and the HTTP server are the same model behind two
        front-ends, so a chunk can come back as a loud low-frequency drone in
        either of them (upstream k2-fsa/OmniVoice issues #37, #73, #144), and
        this one has nobody to warn it: the worker returns WAV bytes and the
        buzz used to be written straight into the recorded segment.

        The recovery policy is shared with the server engine
        (``omnivoice_quality``): draw the segment again — a drone is a bad roll,
        not a bad setting, and the retry costs nothing until it is needed — and
        when every draw is a drone, re-record the text in sentence-sized pieces
        and join them.  Pieces are joined in order, so one segment is still
        exactly one file.

        The whole segment is one request here (the worker has no request-size
        cap), so a re-draw re-renders all of it: a segment that drones pays
        roughly one extra render per retry, and a repaired segment is generated
        piece by piece.
        """
        self.last_repairs = []
        self.last_warnings = []

        def draw(piece: str, piece_seed):
            return self._draw(piece, speed, piece_seed)

        samples, record = quality.recover(
            text,
            draw,
            seed=self.seed,
            attempts=self.drone_attempts,
            repair_attempts=self.repair_attempts,
        )
        if record:
            self.last_repairs = [record]
            self.last_warnings = [self._repair_message(record)]
            for message in self.last_warnings:
                log.warning(message)
        return samples

    def _report_quality(self, samples) -> None:
        """Judge the finished segment as a whole.

        Safety net over the repaired take: a drone at the seam between two
        pieces, or one shorter than a single piece's verdict, would only be
        visible on the whole segment.
        """
        verdict = quality.judge(samples)
        if verdict.bad:
            message = (
                "OmniVoice thinks part of this segment is noise rather than "
                f"speech ({verdict.reason}). Listen to it before publishing."
            )
            self.last_warnings.append(message)
            log.warning(message)

    @staticmethod
    def _repair_message(record: dict) -> str:
        """One user-readable line about a repaired (or unrepaired) drone."""
        return quality.repair_message(record)

    def close(self) -> None:
        self._worker.close()
