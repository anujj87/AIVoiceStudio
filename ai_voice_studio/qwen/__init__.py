"""Qwen3-TTS (ONNX).

Qwen3-TTS is Alibaba's open-source multilingual TTS (Apache-2.0). It runs as
pure ONNX (no PyTorch) and supports both plain synthesis and **voice
cloning**: give it a short reference recording and it speaks in that voice in
any of 10 languages (Chinese, English, German, Italian, Portuguese, Spanish,
Japanese, Korean, French, Russian).

Like the XTTS cloner it runs in a separate worker subprocess using its own
onnxruntime (downloaded into ``runtime/qwen``), so the two ONNX runtimes never
share a process. The model itself (~5.6 GB fp16, voice_clone subset) is
downloaded through the normal Settings -> Download panel.
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

import numpy as np

from .. import paths
from ..util import find_python, sanitize_filename

log = logging.getLogger(__name__)

#: Qwen3-TTS supported languages (code, English name) - used in the clone tab.
QWEN_LANGUAGES = [
    ("english", "English"),
    ("chinese", "Chinese"),
    ("japanese", "Japanese"),
    ("korean", "Korean"),
    ("german", "German"),
    ("french", "French"),
    ("russian", "Russian"),
    ("portuguese", "Portuguese"),
    ("spanish", "Spanish"),
    ("italian", "Italian"),
]

_SAMPLE_RATE = 24000

#: onnxruntime-gpu builds that work with the CUDA 12 runtime downloaded by the
#: app (sherpa-onnx 1.13.5 cuda12). Newer ORT (1.26+) wants CUDA 13 DLLs.
_ONNXRUNTIME_GPU = "onnxruntime-gpu==1.24.4"


class Qwen3Error(Exception):
    """Raised when the Qwen3 engine/runtime is missing or fails."""


def engine_dir() -> str:
    return os.path.join(paths.user_data_dir(), "runtime", "qwen")


def engine_installed() -> bool:
    """True when ``onnxruntime`` is importable from the qwen runtime folder."""
    pkg = os.path.join(engine_dir(), "onnxruntime")
    return os.path.isdir(pkg)


def cuda_installed() -> bool:
    """True when the app's CUDA 12 runtime (shared with sherpa-onnx) exists."""
    cuda_dir = os.path.join(paths.user_data_dir(), "runtime", "cuda")
    return os.path.isfile(os.path.join(cuda_dir, "onnxruntime.dll"))


def ensure_engine(progress=None, cancel_event: threading.Event | None = None) -> None:
    """Install the Qwen3 runtime into the user folder with pip --target.

    Installs onnxruntime (CUDA build when the app's GPU runtime is present,
    plain CPU otherwise) plus librosa, soundfile and tokenizers. The main app
    never imports these - only the worker subprocess does.
    """
    target = engine_dir()
    os.makedirs(target, exist_ok=True)
    if progress:
        progress("Preparing pip install of the Qwen3 engine", 0, 1)
    rt = "onnxruntime-gpu==1.24.4" if cuda_installed() else "onnxruntime"
    packages = [rt, "librosa", "soundfile", "tokenizers"]
    if progress:
        progress(
            f"Installing {', '.join(packages)} "
            f"({'GPU' if rt.startswith('onnxruntime-gpu') else 'CPU'} build)",
            0, 0,
        )
    import tempfile  # noqa: PLC0415
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = [
        find_python(), "-m", "pip", "install",
        "--target", target, "--upgrade",
    ] + packages
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            cwd=tempfile.gettempdir(), creationflags=creationflags,
            timeout=7200,
        )
    except subprocess.TimeoutExpired as exc:
        raise Qwen3Error("The pip install timed out after 2 hours.") from exc
    except OSError as exc:
        raise Qwen3Error(
            f"Could not start pip to install the Qwen3 engine: {exc}"
        ) from exc
    if cancel_event and cancel_event.is_set():
        raise Qwen3Error("Installation cancelled.")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise Qwen3Error(
            "Installing the Qwen3 engine failed:\n" + "\n".join(tail)
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise Qwen3Error(
            "Installing the Qwen3 engine failed:\n" + "\n".join(tail)
        )
    if progress:
        progress("Qwen3 engine installed.", 1, 1)


def remove_engine() -> None:
    shutil.rmtree(engine_dir(), ignore_errors=True)


class QwenClient:
    """Manages the persistent Qwen3 worker subprocess."""

    def __init__(self, timeout: float = 1800.0):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._timeout = timeout
        self._next_id = 0

    # -- lifecycle ----------------------------------------------------------
    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [sys.executable, "-m", "ai_voice_studio.qwen.worker"]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        resp = self._request({"cmd": "ping"})
        if not resp.get("ok"):
            raise Qwen3Error(
                resp.get("error", "The Qwen3 worker failed to start.")
            )

    def close(self):
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

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- protocol -----------------------------------------------------------
    def _request(self, req: dict) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise Qwen3Error("The Qwen3 worker is not running.")
            req = dict(req)
            req["id"] = self._next_id
            self._next_id += 1
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise Qwen3Error(f"The Qwen3 worker failed: {exc}") from exc
            if not line:
                raise Qwen3Error(
                    "The Qwen3 worker stopped. Is the Qwen3 engine downloaded?"
                )
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Qwen3Error(
                    "Invalid response from the Qwen3 worker."
                ) from exc
            if resp.get("id") != req["id"]:
                raise Qwen3Error(
                    "The Qwen3 worker returned an out-of-order response."
                )
            return resp

    # -- public API ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        model_dir: str,
        ref_wav: str,
        ref_text: str = "",
        language: str = "english",
        speed: float = 1.0,
    ) -> np.ndarray:
        self._ensure_proc()
        resp = self._request(
            {
                "cmd": "synthesize",
                "text": text,
                "model_dir": model_dir,
                "ref_wav": ref_wav,
                "ref_text": ref_text,
                "language": language,
                "speed": float(speed),
            }
        )
        if not resp.get("ok"):
            raise Qwen3Error(resp.get("error", "Synthesis failed."))
        raw = base64.b64decode(resp["wav"])
        samples = np.frombuffer(raw, dtype=np.int16).copy()
        return samples


class Qwen3Engine:
    """Drops into the app's engine interface (``get_engine``) for Qwen3 voices.

    Implements the same ``synthesize`` contract as ``TtsEngine`` so the
    recording worker and preview buttons need no changes: returns int16
    samples at ``sample_rate`` (24 kHz).
    """

    sample_rate = _SAMPLE_RATE

    def __init__(self, voice_entry: dict, client: QwenClient | None = None):
        self.voice_entry = voice_entry
        self._client = client or QwenClient()
        self.model_dir = voice_entry.get("model_dir") or voice_entry.get("dir")
        self.ref_wav = voice_entry.get("reference") or voice_entry.get("sample")
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise Qwen3Error(
                "The Qwen3 model is missing. Download it in Settings -> "
                "Download and remove first."
            )
        if not self.ref_wav or not os.path.isfile(self.ref_wav):
            raise Qwen3Error(
                "The reference recording for this voice is missing. "
                "Re-create the cloned voice."
            )
        self.ref_text = voice_entry.get("ref_text") or ""
        self.language = voice_entry.get("qwen_lang") or "english"

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
        samples = self._client.synthesize(
            text=text,
            model_dir=self.model_dir,
            ref_wav=self.ref_wav,
            ref_text=self.ref_text,
            language=self.language,
            speed=float(speed),
        )
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415

            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415

            samples = _apply_volume(samples, volume)
        return samples

    def close(self):
        self._client.close()


def create_qwen_voice(
    name: str,
    sample_path: str,
    language: str,
    model_dir: str,
    ref_text: str,
    store,
) -> dict:
    """Copy ``sample_path`` into the models folder and register a Qwen3 voice.

    ``store`` is a ``ModelStore``; ``model_dir`` is the downloaded Qwen3
    model folder. Returns the new custom-voice entry.
    """
    name = sanitize_filename(name.strip(), 40) or "qwen_voice"
    dest = store.custom_dir(name)
    sample = os.path.join(dest, "sample.wav")
    shutil.copy2(sample_path, sample)
    store.add_custom_voice(
        name, "qwen3", dest, "qwen3",
        extra={
            "sample": sample,
            "reference": sample,
            "language": language,
            "qwen_lang": language,
            "ref_text": ref_text,
            "model_dir": model_dir,
        },
    )
    return {"name": name, "tts": "qwen3", "dir": dest, "kind": "qwen3",
            "sample": sample, "reference": sample, "language": language,
            "qwen_lang": language, "ref_text": ref_text,
            "model_dir": model_dir}
