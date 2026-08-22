"""OmniVoice TTS engine (k2-fsa/OmniVoice) — rewritten.

OmniVoice is a state-of-the-art massively multilingual zero-shot TTS model
supporting 600+ languages with voice cloning and voice design. Two variants:

* **omnivoice-gpu**  -- Full PyTorch model (k2-fsa/OmniVoice). Requires an
  NVIDIA GPU with CUDA. Higher quality, voice cloning from a reference audio,
  and voice design via text instructions.
* **omnivoice-onnx** -- ONNX-converted model (Prince-1/OmniVoice-Onnx). Runs
  on CPU (GPU optional via onnxruntime-genai CUDA provider). Uses the full
  OmniVoice pipeline: text encoder → language model → audio decoder →
  vocoder.

Both run in a worker subprocess (like Qwen3) so their runtimes never share a
process with sherpa-onnx.

License: Apache-2.0 (both models).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

from .. import paths
from ..util import find_python, sanitize_filename

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000

# Supported languages for OmniVoice (common subset; full list is 600+)
OMNIVOICE_LANGUAGES = [
    ("en", "English"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("hi", "Hindi"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("it", "Italian"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("bn", "Bengali"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
]


class OmniVoiceError(Exception):
    """Raised when the OmniVoice engine/runtime is missing or fails."""


# ---------------------------------------------------------------------------
# Engine directory management (per-variant: gpu vs onnx)
# ---------------------------------------------------------------------------

def _bundled_onnx_dir() -> str | None:
    """Return the path to the bundled ONNX runtime, or None."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        d = os.path.join(meipass, "vendor", "omnivoice-onnx")
        if os.path.isdir(d):
            return d
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "vendor", "omnivoice-onnx")
    if os.path.isdir(d):
        return d
    return None


def engine_dir(variant: str = "onnx") -> str:
    """Return the runtime install directory for the given variant."""
    if variant == "onnx":
        bundled = _bundled_onnx_dir()
        if bundled:
            return bundled
    return os.path.join(paths.user_data_dir(), "runtime", f"omnivoice-{variant}")


def engine_installed(variant: str = "onnx") -> bool:
    """True when the OmniVoice runtime for *variant* is importable."""
    if variant == "onnx":
        return _bundled_onnx_dir() is not None
    d = engine_dir(variant)
    return os.path.isdir(os.path.join(d, "omnivoice"))


def cuda_available() -> bool:
    """True when the app's CUDA runtime is installed."""
    cuda_dir = os.path.join(paths.user_data_dir(), "runtime", "cuda")
    return os.path.isfile(os.path.join(cuda_dir, "onnxruntime.dll"))


def ensure_engine(
    variant: str = "onnx",
    progress=None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Install the OmniVoice runtime into the user folder."""
    if variant == "onnx":
        if not _bundled_onnx_dir():
            raise OmniVoiceError(
                "The bundled OmniVoice ONNX runtime was not found. "
                "Please reinstall the application."
            )
        if progress:
            progress("OmniVoice ONNX runtime is bundled (ready to use).", 1, 1)
        return

    target = engine_dir(variant)
    os.makedirs(target, exist_ok=True)
    _install_gpu_engine(target, progress, cancel_event)
    if progress:
        progress("OmniVoice GPU engine installed.", 1, 1)


def _install_gpu_engine(target, progress, cancel_event):
    """Install PyTorch + omnivoice for GPU inference.

    Uses the managed PythonRuntime virtualenv when available so pip
    operations stay within the application's Python environment.
    """
    # Ensure the PythonRuntime virtualenv exists and has pip.
    try:
        from ..python_runtime import get_runtime
        rt = get_runtime()
        rt.ensure_env()
        rt.ensure_pip()
    except Exception:  # noqa: BLE001
        log.debug("PythonRuntime not available, falling back to system Python")

    python = find_python()
    cmd = [
        python, "-m", "pip", "install",
        "--target", target, "--upgrade",
        "torch", "torchaudio",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ]
    if progress:
        progress("Installing PyTorch with CUDA (~2.5 GB)...", 0, 0)
    _run_pip(cmd, "PyTorch+CUDA", progress, cancel_event)

    cmd2 = [
        python, "-m", "pip", "install",
        "--target", target, "--upgrade",
        "omnivoice",
    ]
    if progress:
        progress("Installing omnivoice package...", 0, 0)
    _run_pip(cmd2, "omnivoice", progress, cancel_event)


def _run_pip(cmd, label, progress, cancel_event):
    """Run a pip command with real-time progress streaming."""
    import io

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    safe_cwd = tempfile.gettempdir()

    if cancel_event and cancel_event.is_set():
        raise OmniVoiceError("Installation cancelled.")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=safe_cwd,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise OmniVoiceError(f"Could not start pip: {exc}") from exc

    stderr_chunks: list[str] = []
    last_line = [""]

    def _reader():
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                stderr_chunks.append(text)
                for part in text.replace("\r", "\n").split("\n"):
                    part = part.strip()
                    if part:
                        last_line[0] = part
                        if progress:
                            progress(part, 0, 0)
        except Exception:  # noqa: BLE001
            pass

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    while proc.poll() is None:
        if cancel_event and cancel_event.is_set():
            proc.terminate()
            raise OmniVoiceError("Installation cancelled.")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass  # Process still running; loop back and check cancel_event

    reader.join(timeout=10)

    if cancel_event and cancel_event.is_set():
        raise OmniVoiceError("Installation cancelled.")

    if proc.returncode != 0:
        combined = "".join(stderr_chunks)
        tail = combined.strip().splitlines()[-10:]
        raise OmniVoiceError(
            f"pip install {label} failed (exit {proc.returncode}):\n"
            + "\n".join(tail)
        )


def remove_engine(variant: str = "onnx") -> None:
    shutil.rmtree(engine_dir(variant), ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker subprocess client (JSON protocol over stdin/stdout)
# ---------------------------------------------------------------------------

class OmniVoiceClient:
    """Manages the persistent OmniVoice worker subprocess."""

    def __init__(self, variant: str = "onnx", timeout: float = 3600.0):
        self._variant = variant
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._timeout = timeout
        self._next_id = 0

    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [sys.executable, "-m", "ai_voice_studio.omnivoice.worker",
               "--variant", self._variant]
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
            raise OmniVoiceError(
                resp.get("error", "The OmniVoice worker failed to start.")
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
                    "The OmniVoice worker stopped. Is the engine downloaded?"
                )
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OmniVoiceError(
                    "Invalid response from the OmniVoice worker."
                ) from exc
            if resp.get("id") != req["id"]:
                raise OmniVoiceError(
                    "The OmniVoice worker returned an out-of-order response."
                )
            return resp

    # -- public API ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        model_dir: str,
        ref_wav: str = "",
        ref_text: str = "",
        instruct: str = "",
        language: str = "en",
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize speech. Supports voice cloning, voice design, or auto."""
        self._ensure_proc()
        resp = self._request({
            "cmd": "synthesize",
            "text": text,
            "model_dir": model_dir,
            "ref_wav": ref_wav,
            "ref_text": ref_text,
            "instruct": instruct,
            "language": language,
            "speed": float(speed),
        })
        if not resp.get("ok"):
            raise OmniVoiceError(resp.get("error", "Synthesis failed."))
        raw = base64.b64decode(resp["wav"])
        samples = np.frombuffer(raw, dtype=np.int16).copy()
        return samples


class OmniVoiceEngine:
    """Drops into the app's engine interface (``get_engine``) for OmniVoice voices.

    Implements the same ``synthesize`` contract as ``TtsEngine`` so the
    recording worker and preview buttons need no changes.
    """

    sample_rate = _SAMPLE_RATE

    def __init__(self, voice_entry: dict, client: OmniVoiceClient | None = None):
        self.voice_entry = voice_entry
        variant = voice_entry.get("omnivoice_variant", "onnx")
        self._client = client or OmniVoiceClient(variant=variant)
        self.model_dir = voice_entry.get("model_dir") or voice_entry.get("dir")
        self.ref_wav = voice_entry.get("reference") or voice_entry.get("sample") or ""
        self.ref_text = voice_entry.get("ref_text") or ""
        self.instruct = voice_entry.get("instruct") or ""
        self.language = voice_entry.get("omnivoice_lang") or "en"
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise OmniVoiceError(
                "The OmniVoice model is missing. Download it in Settings -> "
                "Download and remove first."
            )

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
        try:
            samples = self._client.synthesize(
                text=text,
                model_dir=self.model_dir,
                ref_wav=self.ref_wav,
                ref_text=self.ref_text,
                instruct=self.instruct,
                language=self.language,
                speed=float(speed),
            )
        except OmniVoiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OmniVoiceError(f"OmniVoice synthesis failed: {exc}") from exc
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415
            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415
            samples = _apply_volume(samples, volume)
        return samples

    def close(self):
        self._client.close()


# ---------------------------------------------------------------------------
# Voice creation helpers
# ---------------------------------------------------------------------------

def create_omnivoice_voice(
    name: str,
    sample_path: str,
    language: str,
    model_dir: str,
    ref_text: str,
    instruct: str,
    variant: str,
    store,
) -> dict:
    """Copy sample into models folder and register an OmniVoice voice."""
    name = sanitize_filename(name.strip(), 40) or "omnivoice_voice"
    dest = store.custom_dir(name)
    sample = os.path.join(dest, "sample.wav")
    shutil.copy2(sample_path, sample)
    store.add_custom_voice(
        name, "omnivoice", dest, "omnivoice",
        extra={
            "sample": sample,
            "reference": sample,
            "language": language,
            "omnivoice_lang": language,
            "ref_text": ref_text,
            "instruct": instruct,
            "model_dir": model_dir,
            "omnivoice_variant": variant,
        },
    )
    return {
        "name": name, "tts": "omnivoice", "dir": dest, "kind": "omnivoice",
        "sample": sample, "reference": sample, "language": language,
        "omnivoice_lang": language, "ref_text": ref_text, "instruct": instruct,
        "model_dir": model_dir, "omnivoice_variant": variant,
    }
