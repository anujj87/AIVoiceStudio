"""OmniVoice TTS engine (omnivoice-triton).

OmniVoice is a massively multilingual zero-shot text-to-speech (TTS) model
supporting over 600 languages, with voice cloning and voice design.  It runs
in a separate worker subprocess so its PyTorch/CUDA runtime never conflicts
with the main app's sherpa-onnx.

The ``omnivoice-triton`` package must be installed separately::

    pip install omnivoice-triton

The worker supports two inference modes (selected at model load time):

* ``triton``  – Stable, ~1.5x faster than stock OmniVoice.  Uses Triton
  kernel fusion only.  Recommended for production use.
* ``hybrid``  – ~3.4x faster than stock.  Triton kernels + CUDA Graph.
  Has a known VRAM-leak bug (memory grows per request until OOM); use with
  caution or unload the model between synthesis batches.

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

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # OmniVoice output sample rate


# ---------------------------------------------------------------------------
# Embedded worker source (standalone script, no ai_voice_studio imports).
# Used in frozen apps where PyInstaller compiles .py into a PYZ archive
# that the managed venv's Python cannot read.
# ---------------------------------------------------------------------------
_WORKER_SOURCE = r'''"""OmniVoice TTS worker subprocess (standalone).

Protocol (JSON lines on stdin/stdout):
  {"cmd": "ping"}                        -> {"ok": true, "engine": true}
  {"cmd": "synthesize", "text": "..."}    -> {"ok": true, "wav": "<base64>", ...}
  {"cmd": "quit"}                         -> (process exits)
'''
_WORKER_SOURCE += r'''"""
import base64
import json
import sys
import numpy as np


def main() -> int:
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
                ref_audio = req.get("ref_audio", "")
                ref_text = req.get("ref_text", "")
                instruct = req.get("instruct", "")
                language = req.get("language", "auto")

                saved_stdout = sys.stdout
                sys.stdout = err
                try:
                    if ref_audio:
                        result = runner.generate_voice_clone(
                            text=text, ref_audio=ref_audio, ref_text=ref_text)
                    elif instruct:
                        result = runner.generate_voice_design(
                            text=text, instruct=instruct)
                    else:
                        result = runner.generate(
                            text=text, language=language)
                finally:
                    sys.stdout = saved_stdout

                audio = result["audio"]
                arr = np.asarray(audio, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    raise ValueError("The engine returned no audio.")
                clipped = np.clip(arr, -1.0, 1.0) * 32767.0
                wav_b64 = base64.b64encode(
                    clipped.astype(np.int16).tobytes()).decode("ascii")
                respond({
                    "ok": True,
                    "wav": wav_b64,
                    "sample_rate": result.get("sample_rate", 24000),
                    "time_ms": result.get("time_ms", 0),
                    "peak_vram_gb": result.get("peak_vram_gb", 0),
                })
            except Exception as exc:
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        respond({"ok": False, "error": f"Unknown command: {cmd}"})

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
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def is_available() -> bool:
    """True when the ``omnivoice_triton`` package is importable from the
    managed venv (checked by the worker subprocess, not in-process)."""
    try:
        from ..python_runtime import get_runtime  # noqa: PLC0415
        rt = get_runtime()
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
# Worker subprocess management (mirrors clone/__init__.py CloneClient)
# ---------------------------------------------------------------------------
class OmniVoiceWorker:
    """Manages the persistent OmniVoice worker subprocess.

    The worker loads the OmniVoice model once (via omnivoice-triton) and
    answers ``synthesize`` / ``clone`` / ``design`` requests over a
    JSON-lines pipe.  PyTorch/CUDA stay isolated in the subprocess.
    """

    def __init__(self, mode: str = "triton", timeout: float = 600.0):
        """
        Parameters
        ----------
        mode : str
            Inference mode: ``"triton"`` (stable) or ``"hybrid"`` (fastest
            but may leak VRAM).
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
        # In a frozen app, sys.executable is the .exe -- use the managed
        # venv's Python instead so it can find omnivoice-triton.
        if getattr(sys, "frozen", False):
            from ..python_runtime import get_runtime  # noqa: PLC0415
            rt = get_runtime()
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
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
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
        language: str = "auto",
        num_steps: int = 32,
    ) -> np.ndarray:
        """Synthesize text; returns int16 samples at 24 kHz.

        Parameters
        ----------
        text : str
            Text to synthesize.
        speed : float
            Speaking speed (0.5-2.0).
        ref_audio : str
            Path to reference audio for voice cloning (3-15 s optimal).
        ref_text : str
            Transcript of reference audio (for cloning).
        instruct : str
            Voice design instruction, e.g. ``"female, low pitch, british accent"``.
        language : str
            Language hint (``"auto"``, ``"en"``, ``"zh"``, etc.).
        num_steps : int
            Diffusion steps (16 = faster, 32 = balanced, 64 = best).
        """
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        req: Dict = {
            "cmd": "synthesize",
            "text": text,
            "mode": self._mode,
            "speed": float(speed),
            "language": language,
            "num_steps": num_steps,
        }
        if ref_audio:
            req["ref_audio"] = ref_audio
            req["ref_text"] = ref_text
        if instruct:
            req["instruct"] = instruct
        resp = self._request(req)
        if not resp.get("ok"):
            raise OmniVoiceError(resp.get("error", "Synthesis failed."))
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
    """

    sample_rate = _SAMPLE_RATE

    def __init__(
        self,
        voice_entry: dict,
        worker: OmniVoiceWorker | None = None,
    ):
        self.voice_entry = voice_entry
        variant = voice_entry.get("variant", "triton")
        self._worker = worker or OmniVoiceWorker(mode=variant)
        self._worker._ensure_proc()
        self.language = voice_entry.get("language", "auto")
        self.instruct = voice_entry.get("instruct", "")
        self.ref_audio = voice_entry.get("ref_audio", "")
        self.ref_text = voice_entry.get("ref_text", "")

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
        samples = self._worker.synthesize(
            text=text,
            speed=float(speed),
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            instruct=self.instruct,
            language=self.language,
        )
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415

            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415

            samples = _apply_volume(samples, volume)
        return samples

    def close(self) -> None:
        self._worker.close()
